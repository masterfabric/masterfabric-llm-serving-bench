"""Long-context curve: prefill speed, decode speed and peak memory from 512 to ~31k prompt tokens,
llama.cpp (Metal) against MLX, one fresh process per point so the peak memory belongs to that point.

    python bench/ctx_curve.py --out results/ctx_curve.json

Both engines get the same token ids (Qwen2.5 tokenizer, tokenized once through llama-server /tokenize),
generate exactly 128 tokens greedily (EOS ignored), and run under /usr/bin/time -l, whose
"peak memory footprint" includes Metal buffers in unified memory. Each engine also reports its own
accounting: llama.cpp's buffer sizes from the server log, MLX's mx.get_peak_memory().
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
M = Path(os.environ.get("MODELS", Path.home() / ".cache" / "mf-models"))
RAW = ROOT / "results" / "raw"
FILLER = ROOT / "bench" / "text" / "filler.txt"
PORT = 58083
POINTS = [512, 4096, 8192, 16384, 24576, 31744]
GEN = 128


def peak_footprint(stderr_text: str) -> int | None:
    m = re.search(r"(\d+)\s+peak memory footprint", stderr_text)
    return int(m.group(1)) if m else None


def wait_up(url: str, proc: subprocess.Popen, timeout: float = 180) -> None:
    t = time.time()
    while time.time() - t < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early, see {RAW}")
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise SystemExit("server did not start")


def stop_timed(proc: subprocess.Popen) -> None:
    """proc is /usr/bin/time; stop its child so time prints the rusage block and exits."""
    kids = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True).stdout.split()
    for k in kids:
        os.kill(int(k), signal.SIGTERM)
    proc.wait(timeout=60)


def token_ids(n_max: int, model: Path) -> list[int]:
    cache = RAW / "ctx_prompt_ids.json"
    if cache.exists():
        ids = json.loads(cache.read_text())
        if len(ids) >= n_max:
            return ids
    text = "Summarise the following documentation in one paragraph.\n\n"
    body = FILLER.read_text()
    while len(text) < n_max * 4.2:
        text += body + "\n"
    proc = subprocess.Popen(["llama-server", "-m", str(model), "--host", "127.0.0.1", "--port", str(PORT), "-c", "512"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_up(f"http://127.0.0.1:{PORT}", proc)
        ids = httpx.post(f"http://127.0.0.1:{PORT}/tokenize", json={"content": text}, timeout=120).json()["tokens"]
    finally:
        proc.terminate()
        proc.wait()
    RAW.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(ids[:n_max]))
    return ids[:n_max]


def llamacpp_point(model: Path, ids: list[int], kv: str) -> dict:
    log = RAW / f"ctx_llamacpp_{kv}_{len(ids)}.log"
    args = ["/usr/bin/time", "-l", "llama-server", "-m", str(model), "--host", "127.0.0.1", "--port", str(PORT),
            "-ngl", "99", "-np", "1", "-c", str(len(ids) + GEN + 64), "-fa", "on", "-ctk", kv, "-ctv", kv,
            "--metrics", "-lm", "none"]
    rusage = log.with_suffix(".time")
    with rusage.open("w") as f:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=f)
        try:
            url = f"http://127.0.0.1:{PORT}"
            wait_up(url, proc)
            httpx.post(f"{url}/completion", json={"prompt": ids[:16], "n_predict": 4, "cache_prompt": False}, timeout=60)
            r = httpx.post(f"{url}/completion", json={"prompt": ids, "n_predict": GEN, "ignore_eos": True,
                                                      "temperature": 0, "cache_prompt": False}, timeout=1800).json()
        finally:
            stop_timed(proc)
    t = r["timings"]
    # buffer sizes are logged only at verbosity 4, which slows decoding by ~20%: read them from a second,
    # load-only start with the same settings
    acct = subprocess.Popen(args[2:] + ["--log-file", str(log), "-lv", "4"], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        wait_up(f"http://127.0.0.1:{PORT}", acct)
    finally:
        acct.terminate()
        acct.wait()
    text = log.read_text()
    peak = peak_footprint(rusage.read_text())
    kv_mib = sum(float(x) for x in re.findall(r"KV buffer size =\s*([\d.]+) MiB", text))
    model_mib = sum(float(x) for x in re.findall(r"model buffer size =\s*([\d.]+) MiB", text))
    compute_mib = sum(float(x) for x in re.findall(r"compute buffer size =\s*([\d.]+) MiB", text))
    return {"engine": "llama.cpp", "kv": kv, "prompt_tokens": t["prompt_n"], "prefill_s": round(t["prompt_ms"] / 1000, 3),
            "prefill_tok_s": round(t["prompt_per_second"], 1), "decode_tok_s": round(t["predicted_per_second"], 1),
            "generated": t["predicted_n"], "peak_footprint_mib": round(peak / 2**20, 1),
            "engine_kv_mib": round(kv_mib, 1), "engine_model_mib": round(model_mib, 1),
            "engine_compute_mib": round(compute_mib, 1)}


def mlx_point(model: Path, ids: list[int], kv_bits: int | None) -> dict:
    tmp = RAW / "ctx_ids_tmp.json"
    tmp.write_text(json.dumps(ids))
    cmd = ["/usr/bin/time", "-l", sys.executable, __file__, "--point-mlx", str(model), "--ids", str(tmp)]
    if kv_bits:
        cmd += ["--kv-bits", str(kv_bits)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if p.returncode:
        raise SystemExit(p.stderr[-2000:])
    res = json.loads(p.stdout.strip().splitlines()[-1])
    res["peak_footprint_mib"] = round(peak_footprint(p.stderr) / 2**20, 1)
    return res


def run_mlx_point(model: str, ids_file: str, kv_bits: int | None) -> None:
    """Runs inside the child process."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    ids = json.loads(Path(ids_file).read_text())
    m, _tok = load(model)
    greedy = make_sampler(temp=0.0)
    kw = {"kv_bits": kv_bits, "quantized_kv_start": 0} if kv_bits else {}
    for _ in generate_step(mx.array(ids[:16]), m, max_tokens=4, sampler=greedy):  # warm-up
        pass
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    first = None
    n = 0
    for tok, _ in generate_step(mx.array(ids), m, max_tokens=GEN, sampler=greedy, **kw):
        mx.eval(tok)
        n += 1
        if first is None:
            first = time.perf_counter()
    t1 = time.perf_counter()
    print(json.dumps({"engine": "mlx", "kv": f"q{kv_bits}" if kv_bits else "f16", "prompt_tokens": len(ids),
                      "prefill_s": round(first - t0, 3), "prefill_tok_s": round(len(ids) / (first - t0), 1),
                      "decode_tok_s": round((n - 1) / (t1 - first), 1), "generated": n,
                      "engine_peak_mib": round(mx.get_peak_memory() / 2**20, 1)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--point-mlx")
    ap.add_argument("--ids")
    ap.add_argument("--kv-bits", type=int)
    ap.add_argument("--gguf", default=str(M / "qwen2.5-1.5b-instruct-q4_k_m.gguf"))
    ap.add_argument("--mlx", default=str(M / "mlx" / "Qwen2.5-1.5B-Instruct-4bit"))
    ap.add_argument("--points", default=",".join(map(str, POINTS)))
    ap.add_argument("--configs", default="llamacpp:f16,llamacpp:q8_0,mlx:f16,mlx:q8")
    ap.add_argument("--out", default=str(ROOT / "results" / "ctx_curve.json"))
    a = ap.parse_args()
    if a.point_mlx:
        run_mlx_point(a.point_mlx, a.ids, a.kv_bits)
        return
    points = [int(x) for x in a.points.split(",")]
    ids = token_ids(max(points), Path(a.gguf))
    out = Path(a.out)
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    for cfg in a.configs.split(","):
        engine, kv = cfg.split(":")
        for n in points:
            if any(r["engine"] == ("llama.cpp" if engine == "llamacpp" else "mlx") and r["kv"] == kv
                   and r["prompt_tokens"] == n for r in rows):
                continue
            r = llamacpp_point(Path(a.gguf), ids[:n], kv) if engine == "llamacpp" else \
                mlx_point(Path(a.mlx), ids[:n], 8 if kv == "q8" else None)
            print(json.dumps(r))
            rows.append(r)
            out.write_text(json.dumps({"model": "Qwen2.5-1.5B-Instruct", "gguf": Path(a.gguf).name,
                                       "mlx": Path(a.mlx).name, "generated_tokens": GEN, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
