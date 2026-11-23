"""Why does greedy output with a draft model differ from the target alone in llama.cpp?

    python bench/spec_mismatch.py --target 7b --out results/spec_mismatch_7b.json

Hypothesis: speculative decoding verifies k+1 tokens in one batch, and the Metal kernels for a batch of
tokens (matrix-matrix) round differently from the kernels for one token (matrix-vector). Where the top
two tokens are nearly tied, the argmax can flip, and greedy text diverges from then on.

Four measurements, greedy, 256 tokens, the eight prompts of spec_decode.py, one request at a time:
  1. repeat    target alone, twice: is single-token decoding deterministic?
  2. draft     target + 0.5B draft (3 draft tokens): where does the text first differ (token index)?
  3. margin    the target-alone log-probability gap between its top-1 and top-2 token at that index,
               against the gap distribution over all generated tokens
  4. no-draft  target alone, prompt + the common prefix fed as one batch (prefill), 1 token: does a
               batched pass pick the draft run's token, without any draft model involved?
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

from spec_decode import MODELS, PROMPTS, items

ROOT = Path(__file__).resolve().parents[1]
M = Path(os.environ.get("MODELS", Path.home() / ".cache" / "mf-models"))
LLAMA = os.environ.get("LLAMA_SERVER", "llama-server")
PORT = 58084
URL = f"http://127.0.0.1:{PORT}"
N = 256


def start(target: Path, draft: Path | None, log: Path, extra: list[str] | None = None):
    args = [LLAMA, "-m", str(target), "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99", "-np", "1",
            "-c", "4096", "-fa", "on"] + (extra or [])
    if draft:
        args += ["--spec-type", "draft-simple", "-md", str(draft), "-ngld", "99", "--spec-draft-n-max", "3",
                 "--spec-draft-n-min", "0"]
    f = log.open("w")
    p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT)
    for _ in range(600):
        try:
            if httpx.get(f"{URL}/health", timeout=2).status_code == 200:
                return p
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    p.terminate()
    raise RuntimeError("server did not start")


def stop(p):
    p.terminate()
    p.wait()
    time.sleep(1)


def prompt_ids(text: str) -> list[int]:
    templ = httpx.post(f"{URL}/apply-template", json={"messages": [{"role": "user", "content": text}]},
                       timeout=60).json()["prompt"]
    return httpx.post(f"{URL}/tokenize", json={"content": templ, "add_special": True, "parse_special": True},
                      timeout=60).json()["tokens"]


def generate(ids: list[int], n: int, probs: bool) -> dict:
    body = {"prompt": ids, "n_predict": n, "temperature": 0, "top_k": 1, "cache_prompt": False,
            "samplers": ["top_k"], "return_tokens": True}
    if probs:
        body["n_probs"] = 2
    r = httpx.post(f"{URL}/completion", json=body, timeout=900).json()
    out = {"tokens": r["tokens"], "text": r["content"], "tok_s": r["timings"]["predicted_per_second"]}
    if probs:
        gaps = []
        for c in r["completion_probabilities"]:
            top = c["top_logprobs"]
            gaps.append(round(top[0]["logprob"] - top[1]["logprob"], 5) if len(top) > 1 else None)
        out["gaps"] = gaps
        out["top2"] = [[t["id"] for t in c["top_logprobs"][:2]] for c in r["completion_probabilities"]]
    return out


def first_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["7b", "1.5b"], default="7b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = ROOT / "results" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tgt, drf = M / MODELS[a.target]["gguf"], M / MODELS["draft"]["gguf"]

    p = start(tgt, None, raw / f"mismatch_{a.target}_base.log")
    try:
        ids = {pid: prompt_ids(text) for _, pid, text in items()}
        generate(ids["s1"], 8, False)  # warm-up
        base = {pid: generate(ids[pid], N, True) for pid in ids}
        again = {pid: generate(ids[pid], N, False) for pid in ids}
    finally:
        stop(p)
    p = start(tgt, drf, raw / f"mismatch_{a.target}_draft.log")
    try:
        generate(ids["s1"], 8, False)
        spec = {pid: generate(ids[pid], N, False) for pid in ids}
    finally:
        stop(p)

    rows = []
    p = start(tgt, None, raw / f"mismatch_{a.target}_prefill.log")
    try:
        for group, pid, _ in items():
            b, s = base[pid], spec[pid]
            i = first_diff(b["tokens"], s["tokens"])
            row = {"group": group, "id": pid, "repeat_identical": again[pid]["tokens"] == b["tokens"],
                   "draft_identical": i is None, "first_diff_token": i}
            if i is not None and i < len(b["gaps"]):
                row.update({"gap_at_diff": b["gaps"][i], "target_token": b["tokens"][i], "draft_run_token": s["tokens"][i],
                            "draft_token_is_target_top2": s["tokens"][i] in b["top2"][i],
                            "gap_rank_pct": round(100 * sum(g is not None and g <= b["gaps"][i] for g in b["gaps"])
                                                  / sum(g is not None for g in b["gaps"]), 1)})
                # same prefix, fed as one batch to the target alone (no draft model loaded)
                pf = generate(ids[pid] + b["tokens"][:i], 1, False)["tokens"][0]
                row["batched_prefill_token"] = pf
                row["batched_prefill_matches"] = ("draft run" if pf == s["tokens"][i] else
                                                 "target alone" if pf == b["tokens"][i] else "neither")
            rows.append(row)
    finally:
        stop(p)
    allgaps = [g for pid in base for g in base[pid]["gaps"] if g is not None]
    res = {"engine": "llama.cpp", "target": tgt.name, "draft": drf.name, "draft_tokens": 3, "max_tokens": N,
           "rows": rows,
           "summary": {"repeat_identical": sum(r["repeat_identical"] for r in rows),
                       "draft_identical": sum(r["draft_identical"] for r in rows), "n": len(rows),
                       "diverged_at_top2": sum(bool(r.get("draft_token_is_target_top2")) for r in rows),
                       "gap_at_diff_median": statistics.median([r["gap_at_diff"] for r in rows if "gap_at_diff" in r] or [0]),
                       "gap_all_tokens_median": round(statistics.median(allgaps), 4),
                       "tokens_with_gap_below_0.1": round(sum(g < 0.1 for g in allgaps) / len(allgaps), 4),
                       "batched_prefill_matches_draft_run": sum(r.get("batched_prefill_matches") == "draft run" for r in rows),
                       "batched_prefill_matches_target_alone": sum(r.get("batched_prefill_matches") == "target alone" for r in rows),
                       "decode_tok_s_target": round(statistics.mean(base[k]["tok_s"] for k in base), 2),
                       "decode_tok_s_draft": round(statistics.mean(spec[k]["tok_s"] for k in spec), 2)},
           "texts": {pid: {"target": base[pid]["text"], "draft": spec[pid]["text"]} for pid in base}}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["summary"], indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
