"""Perplexity of the MLX 4-bit and 8-bit weights on the same text and method as llama-perplexity.

    python bench/mlx_perplexity.py --out results/perplexity_mlx.json

Like llama.cpp's perplexity tool: the text is split into 512-token chunks, each chunk is evaluated on its
own, and only the second half of each chunk is scored (every scored token has at least 256 tokens of
context). The error is the standard error of the mean NLL, propagated to PPL.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

ROOT = Path(__file__).resolve().parents[1]
M = Path(os.environ.get("MODELS", Path.home() / ".cache" / "mf-models"))
CTX = 512


def ppl(model_dir: Path, text: str) -> dict:
    model, tok = load(str(model_dir))
    ids = tok.encode(text, add_special_tokens=False)
    nll = []
    for c in range(len(ids) // CTX):
        chunk = mx.array(ids[c * CTX:(c + 1) * CTX])[None]
        logits = model(chunk)[0].astype(mx.float32)
        lp = logits[CTX // 2 - 1:CTX - 1] - mx.logsumexp(logits[CTX // 2 - 1:CTX - 1], axis=-1, keepdims=True)
        tgt = chunk[0, CTX // 2:CTX]
        nll.extend((-mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]).tolist())
    nll = np.array(nll)
    mean, se = float(nll.mean()), float(nll.std(ddof=1) / math.sqrt(len(nll)))
    size = sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))
    return {"weights": model_dir.name, "file_mib": round(size / 2**20, 1), "chunks": len(ids) // CTX,
            "ppl": round(math.exp(mean), 4), "ppl_err": round(math.exp(mean) * se, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "perplexity_mlx.json"))
    a = ap.parse_args()
    text = (ROOT / "bench" / "text" / "filler.txt").read_text()
    rows = [ppl(M / "mlx" / d, text) for d in ("Qwen2.5-1.5B-Instruct-4bit", "Qwen2.5-1.5B-Instruct-8bit")]
    print(rows)
    Path(a.out).write_text(json.dumps({"text": "bench/text/filler.txt", "ctx": CTX, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
