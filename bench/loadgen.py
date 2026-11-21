"""Streaming load generator for OpenAI-compatible chat servers (llama.cpp, vLLM, Ollama's /v1).

For every request it records time to first token (TTFT), inter-token latency (ITL), end-to-end latency
and generated tokens. Each concurrency level runs a fixed number of requests with N in flight.

    python bench/loadgen.py --url http://127.0.0.1:58080 --label metal-q4 --levels 1,2,4,8 \
        --max-tokens 128 --out results/raw/metal-q4.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

PROMPTS = [
    "Write a detailed technical explanation of how a Kafka consumer group rebalances partitions.",
    "Describe, step by step, how Apache Iceberg performs snapshot isolation for concurrent writers.",
    "Explain the trade-offs between PagedAttention and a contiguous KV cache in LLM serving.",
    "Write a short story about a network engineer who finds a sleeping cell at 3 a.m.",
    "List and explain ten best practices for running PostgreSQL on Kubernetes.",
    "Explain how gradient boosting builds trees and why learning rate matters.",
    "Describe the lifecycle of a churn model from data quality checks to online serving.",
    "Explain how quantization to 4 bits changes the memory footprint and speed of a language model.",
]
FILLER = Path(__file__).resolve().parent / "text" / "filler.txt"


def long_prompt(tokens: int, salt: int) -> str:
    """About `tokens` tokens of real documentation text (approx. 3.6 chars/token for this corpus)."""
    text = FILLER.read_text()
    n = int(tokens * 3.6)
    while len(text) < n + 2000:
        text += "\n" + text
    start = (salt * 997) % 2000
    return f"[{salt}] Summarise the following documentation in one paragraph.\n\n" + text[start:start + n]


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


async def one(client, url, model, prompt, max_tokens, ignore_eos, cache_prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    if ignore_eos:
        body["ignore_eos"] = True          # llama.cpp extension: generate exactly max_tokens
    if cache_prompt is not None:
        body["cache_prompt"] = cache_prompt
    t0 = time.perf_counter()
    first, stamps, usage = None, [], {}
    async with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                if ch.get("delta", {}).get("content"):
                    now = time.perf_counter()
                    first = first or now
                    stamps.append(now)
    t1 = time.perf_counter()
    itl = [b - a for a, b in zip(stamps, stamps[1:])]
    return {"ttft_s": (first - t0) if first else None, "e2e_s": t1 - t0,
            "itl_ms_median": statistics.median(itl) * 1000 if itl else None,
            "completion_tokens": usage.get("completion_tokens", len(stamps)),
            "prompt_tokens": usage.get("prompt_tokens")}


async def level(url, model, conc, n, max_tokens, prompt_tokens, ignore_eos, cache_prompt):
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as client:
        async def task(i):
            async with sem:
                p = long_prompt(prompt_tokens, i) if prompt_tokens else f"[{i}] {PROMPTS[i % len(PROMPTS)]}"
                return await one(client, url, model, p, max_tokens, ignore_eos, cache_prompt)
        t0 = time.perf_counter()
        res = await asyncio.gather(*(task(i) for i in range(n)), return_exceptions=True)
        wall = time.perf_counter() - t0
    ok = [r for r in res if isinstance(r, dict)]
    toks = sum(r["completion_tokens"] for r in ok)
    return {
        "concurrency": conc, "requests": n, "errors": n - len(ok), "wall_s": round(wall, 2),
        "output_tok_s": round(toks / wall, 2), "req_s": round(len(ok) / wall, 3),
        "per_request_tok_s_median": round(statistics.median(r["completion_tokens"] / (r["e2e_s"] - r["ttft_s"])
                                                            for r in ok if r["ttft_s"] and r["e2e_s"] > r["ttft_s"]), 2),
        "ttft_s": {"p50": round(pct([r["ttft_s"] for r in ok if r["ttft_s"]], 50), 3),
                   "p95": round(pct([r["ttft_s"] for r in ok if r["ttft_s"]], 95), 3)},
        "itl_ms_p50": round(statistics.median(r["itl_ms_median"] for r in ok if r["itl_ms_median"]), 2),
        "e2e_s": {"p50": round(pct([r["e2e_s"] for r in ok], 50), 2), "p95": round(pct([r["e2e_s"] for r in ok], 95), 2)},
        "prompt_tokens_mean": round(statistics.mean(r["prompt_tokens"] or 0 for r in ok), 1),
        "completion_tokens_mean": round(toks / max(1, len(ok)), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--label", required=True)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--requests-per-level", type=int, default=0, help="default: max(8, 3 x concurrency)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt-tokens", type=int, default=0, help="0 = short prompts")
    ap.add_argument("--no-ignore-eos", action="store_true")
    ap.add_argument("--cache-prompt", choices=["true", "false"], default=None)
    ap.add_argument("--meta", default="{}", help="JSON merged into the output (server settings)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cp = None if a.cache_prompt is None else a.cache_prompt == "true"
    asyncio.run(level(a.url, a.model, 1, 1, 8, 0, True, cp))            # warm-up, not recorded
    levels = []
    for c in [int(x) for x in a.levels.split(",")]:
        n = a.requests_per_level or max(8, 3 * c)
        r = asyncio.run(level(a.url, a.model, c, n, a.max_tokens, a.prompt_tokens, not a.no_ignore_eos, cp))
        print(f"{a.label} c={c}: {r['output_tok_s']} tok/s, TTFT p50 {r['ttft_s']['p50']} s, ITL {r['itl_ms_p50']} ms")
        levels.append(r)
    out = {"label": a.label, "url": a.url, "max_tokens": a.max_tokens, "prompt_tokens_target": a.prompt_tokens,
           "ignore_eos": not a.no_ignore_eos, "cache_prompt": cp, **json.loads(a.meta), "levels": levels}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
