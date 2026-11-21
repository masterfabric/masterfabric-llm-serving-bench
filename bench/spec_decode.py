"""Speculative decoding: target model alone vs target + small draft model, llama.cpp (Metal) and MLX.

    python bench/spec_decode.py --engine llamacpp --target 7b --out results/spec_llamacpp_7b.json
    python bench/spec_decode.py --engine mlx --target 7b --out results/spec_mlx_7b.json

Greedy decoding (temperature 0), 256 new tokens per prompt, one request at a time. Eight prompts in two
groups: "structured" (code, SQL, format conversion, enumerations), where a small model predicts the next
tokens well, and "open" (explanations and a story), where it does not. Reported per configuration:
decode tok/s (prompt processing excluded), draft acceptance rate, and whether the text is identical to
the target-only run (it should be: greedy speculative decoding must not change the output).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
M = Path(os.environ.get("MODELS", Path.home() / ".cache" / "mf-models"))
PORT = 58084
MAX_TOKENS = 256
DRAFTS = [3, 8, 16]

PROMPTS = {
    "structured": [
        "Write a Python function `parse_iso_date(s: str) -> datetime.date` with type hints and a docstring, "
        "followed by five pytest test functions for it.",
        "Write PostgreSQL DDL for three tables: customers, orders and order_items, with primary keys, foreign "
        "keys, NOT NULL constraints and one index per foreign key.",
        "Convert this JSON to YAML and output only the YAML: {\"service\": {\"name\": \"rag-api\", \"replicas\": 2, "
        "\"ports\": [8000, 9090], \"env\": {\"TOP_K\": 4, \"MIN_SCORE\": 0.55, \"LOG_LEVEL\": \"info\"}, "
        "\"resources\": {\"requests\": {\"cpu\": \"250m\", \"memory\": \"256Mi\"}, \"limits\": {\"cpu\": \"1\", "
        "\"memory\": \"512Mi\"}}}}",
        "Write the numbers from one to eighty in English words, separated by commas.",
    ],
    "open": [
        "Explain the trade-offs between PagedAttention and a contiguous KV cache in LLM serving.",
        "Write a short story about a network engineer who finds a sleeping cell at 3 a.m.",
        "Describe the lifecycle of a churn model from data quality checks to online serving.",
        "Why do data teams argue about lakehouse versus warehouse? Give a balanced view.",
    ],
}
MODELS = {
    "7b": {"gguf": "qwen2.5-7b-instruct-q4_k_m.gguf", "mlx": "Qwen2.5-7B-Instruct-4bit"},
    "1.5b": {"gguf": "qwen2.5-1.5b-instruct-q4_k_m.gguf", "mlx": "Qwen2.5-1.5B-Instruct-4bit"},
    "draft": {"gguf": "qwen2.5-0.5b-instruct-q8_0.gguf", "mlx": "Qwen2.5-0.5B-Instruct-8bit"},
}


def items():
    for group, ps in PROMPTS.items():
        for i, p in enumerate(ps):
            yield group, f"{group[0]}{i + 1}", p


def llamacpp_run(target: Path, draft: Path | None, n_max: int | None, log: Path) -> list[dict]:
    args = ["llama-server", "-m", str(target), "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99", "-np", "1",
            "-c", "4096", "-fa", "on"]
    if draft:
        # without --spec-type the draft model is loaded but never used (default: none)
        args += ["--spec-type", "draft-simple", "-md", str(draft), "-ngld", "99", "--spec-draft-n-max", str(n_max), "--spec-draft-n-min", "0"]
    url = f"http://127.0.0.1:{PORT}"
    with log.open("w") as f:
        proc = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try:
                if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        httpx.post(f"{url}/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi"}],
                                                      "max_tokens": 8}, timeout=120)
        rows = []
        for group, pid, p in items():
            r = httpx.post(f"{url}/v1/chat/completions", timeout=600, json={
                "messages": [{"role": "user", "content": p}], "max_tokens": MAX_TOKENS, "temperature": 0,
                "cache_prompt": False}).json()
            t = r["timings"]
            rows.append({"group": group, "id": pid, "text": r["choices"][0]["message"]["content"],
                         "tokens": t["predicted_n"], "decode_tok_s": round(t["predicted_per_second"], 2),
                         "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted")})
        return rows
    finally:
        proc.terminate()
        proc.wait()


def mlx_run(target, draft, n_draft: int | None) -> list[dict]:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tok = target
    greedy = make_sampler(temp=0.0)
    kw = {"draft_model": draft[0], "num_draft_tokens": n_draft} if draft else {}
    for _ in stream_generate(model, tok, "Hi", max_tokens=8, sampler=greedy, **kw):
        pass
    rows = []
    for group, pid, p in items():
        prompt = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        text, last, from_draft = "", None, 0
        for resp in stream_generate(model, tok, prompt, max_tokens=MAX_TOKENS, sampler=greedy, **kw):
            text += resp.text
            from_draft += bool(resp.from_draft)
            last = resp
        rows.append({"group": group, "id": pid, "text": text, "tokens": last.generation_tokens,
                     "decode_tok_s": round(last.generation_tps, 2),
                     "draft_accepted": from_draft if draft else None, "peak_memory_mib": round(last.peak_memory * 1024, 1)})
    return rows


def summarise(rows: list[dict], base: list[dict] | None) -> dict:
    out = {}
    for group in ("structured", "open", "all"):
        rs = [r for r in rows if group == "all" or r["group"] == group]
        s = {"decode_tok_s_mean": round(statistics.mean(r["decode_tok_s"] for r in rs), 2)}
        if rs[0].get("draft_n"):
            s["acceptance"] = round(sum(r["draft_accepted"] for r in rs) / max(1, sum(r["draft_n"] for r in rs)), 3)
        elif rs[0].get("draft_accepted") is not None:
            s["tokens_from_draft"] = round(sum(r["draft_accepted"] for r in rs) / sum(r["tokens"] for r in rs), 3)
        if base:
            b = {r["id"]: r for r in base}
            s["speedup"] = round(statistics.mean(r["decode_tok_s"] / b[r["id"]]["decode_tok_s"] for r in rs), 3)
            s["identical_text"] = sum(r["text"] == b[r["id"]]["text"] for r in rs)
            s["n"] = len(rs)
        out[group] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["llamacpp", "mlx"], required=True)
    ap.add_argument("--target", choices=["7b", "1.5b"], default="7b")
    ap.add_argument("--drafts", default=",".join(map(str, DRAFTS)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = ROOT / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    configs = [None] + [int(x) for x in a.drafts.split(",")]
    res = {"engine": a.engine, "target": MODELS[a.target][a.engine if a.engine == "mlx" else "gguf"],
           "draft": MODELS["draft"]["mlx" if a.engine == "mlx" else "gguf"], "max_tokens": MAX_TOKENS, "configs": []}
    if a.engine == "mlx":
        from mlx_lm import load
        target = load(str(M / "mlx" / MODELS[a.target]["mlx"]))
        draft = load(str(M / "mlx" / MODELS["draft"]["mlx"]))
    base = None
    for n in configs:
        if a.engine == "llamacpp":
            rows = llamacpp_run(M / MODELS[a.target]["gguf"], M / MODELS["draft"]["gguf"] if n else None, n,
                                raw / f"spec_{a.engine}_{a.target}_{n or 0}.log")
        else:
            rows = mlx_run(target, draft if n else None, n)
        summ = summarise(rows, base)
        base = base or rows
        print(a.engine, a.target, f"draft={n}", json.dumps(summ))
        res["configs"].append({"draft_tokens": n, "summary": summ, "rows": rows})
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
