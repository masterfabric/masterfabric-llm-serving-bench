#!/usr/bin/env bash
# Run the benchmark matrix. Every server is started and stopped by this script; nothing else is touched.
#   metal: Homebrew llama-server on the host, all layers on the Apple GPU (Metal), 127.0.0.1:58080
#   cpu:   ghcr.io/ggml-org/llama.cpp server container (Colima VM, 6 vCPU, no GPU), 127.0.0.1:58081
# Usage: bench/run_matrix.sh [concurrency|prefill|kvcache|promptcache|perplexity|all]
set -euo pipefail
cd "$(dirname "$0")/.."
M=${MODELS:-$HOME/.cache/mf-models}
PY=.venv/bin/python
IMG=ghcr.io/ggml-org/llama.cpp:server-b12464
RES=results; RAW=results/raw; mkdir -p "$RES" "$RAW"
q() { echo "$M/qwen2.5-1.5b-instruct-$1.gguf"; }

wait_up() { for _ in $(seq 1 120); do curl -sf "$1/health" >/dev/null 2>&1 && return 0; sleep 1; done; echo "server $1 did not start"; return 1; }

start_metal() {  # $1 quant, rest: extra args
  local quant=$1; shift
  llama-server -m "$(q "$quant")" -a qwen --host 127.0.0.1 --port 58080 -ngl 99 -np 4 -c 16384 --metrics "$@" \
    > "$RAW/server-metal-$quant.log" 2>&1 &
  echo $! > "$RAW/metal.pid"; wait_up http://127.0.0.1:58080
}
stop_metal() { kill "$(cat "$RAW/metal.pid")" 2>/dev/null || true; sleep 2; }

start_cpu() {
  local quant=$1; shift
  docker run -d --rm --name mf-bench-cpu -p 127.0.0.1:58081:8080 -v "$M":/models:ro $IMG \
    -m "/models/qwen2.5-1.5b-instruct-$quant.gguf" -a qwen --host 0.0.0.0 --port 8080 -t 6 -np 4 -c 16384 --metrics "$@" >/dev/null
  wait_up http://127.0.0.1:58081
}
stop_cpu() { docker logs mf-bench-cpu > "$RAW/server-cpu-$1.log" 2>&1 || true; docker stop mf-bench-cpu >/dev/null 2>&1 || true; }

concurrency() {
  for quant in q4_k_m q8_0; do
    start_metal $quant
    $PY bench/loadgen.py --url http://127.0.0.1:58080 --label "metal-$quant" --levels 1,2,4,8 \
      --meta "{\"backend\":\"metal\",\"quant\":\"$quant\",\"slots\":4}" --out "$RES/concurrency_metal_$quant.json"
    stop_metal
    start_cpu $quant
    $PY bench/loadgen.py --url http://127.0.0.1:58081 --label "cpu-$quant" --levels 1,2,4,8 \
      --meta "{\"backend\":\"cpu\",\"quant\":\"$quant\",\"slots\":4}" --out "$RES/concurrency_cpu_$quant.json"
    stop_cpu $quant
  done
}

prefill() {  # 32k context over 4 slots = 8k per sequence
  start_metal q4_k_m -c 32768
  for n in 256 1024 4096 6144; do
    $PY bench/loadgen.py --url http://127.0.0.1:58080 --label "metal-prefill-$n" --levels 1 --requests-per-level 3 \
      --max-tokens 16 --prompt-tokens $n --cache-prompt false --meta '{"backend":"metal","quant":"q4_k_m"}' \
      --out "$RAW/prefill_metal_$n.json"
  done
  stop_metal
  start_cpu q4_k_m -c 32768
  for n in 256 1024 4096; do
    $PY bench/loadgen.py --url http://127.0.0.1:58081 --label "cpu-prefill-$n" --levels 1 --requests-per-level 3 \
      --max-tokens 16 --prompt-tokens $n --cache-prompt false --meta '{"backend":"cpu","quant":"q4_k_m"}' \
      --out "$RAW/prefill_cpu_$n.json"
  done
  stop_cpu q4_k_m
  $PY - <<'PY'
import glob, json
rows = []
for f in sorted(glob.glob("results/raw/prefill_*.json")):
    d = json.load(open(f)); l = d["levels"][0]
    rows.append({"backend": d["backend"], "prompt_tokens": l["prompt_tokens_mean"], "ttft_p50_s": l["ttft_s"]["p50"],
                 "prefill_tok_s": round(l["prompt_tokens_mean"] / l["ttft_s"]["p50"], 1)})
json.dump({"quant": "q4_k_m", "cache_prompt": False, "rows": rows}, open("results/prefill.json", "w"), indent=2)
print(rows)
PY
}

kvcache() {  # 4 concurrent requests with ~6k-token prompts in a 32k context: f16 vs q8_0 KV cache
  for kv in f16 q8_0; do
    start_metal q4_k_m -c 32768 -fa on -ctk $kv -ctv $kv
    $PY bench/loadgen.py --url http://127.0.0.1:58080 --label "metal-kv-$kv" --levels 4 --requests-per-level 8 \
      --max-tokens 128 --prompt-tokens 6000 --cache-prompt false \
      --meta "{\"backend\":\"metal\",\"quant\":\"q4_k_m\",\"kv_type\":\"$kv\",\"ctx\":32768}" --out "$RAW/kv_$kv.json"
    ps -o rss= -p "$(cat "$RAW/metal.pid")" | tr -d ' ' > "$RAW/kv_$kv.rss_kib"   # resident set after the run
    stop_metal
  done
  $PY - <<'PY'
import json, re
rows = []
for kv in ("f16", "q8_0"):
    d = json.load(open(f"results/raw/kv_{kv}.json")); l = d["levels"][0]
    rss = int(open(f"results/raw/kv_{kv}.rss_kib").read()) / 1024
    # Qwen2.5-1.5B: 28 layers x 2 KV heads x 128 dims, K and V -> 14,336 values per token
    bytes_per_value = {"f16": 2.0, "q8_0": 34 / 32}[kv]
    rows.append({"kv_type": kv, "ctx": 32768, "kv_buffer_mib": round(32768 * 14336 * bytes_per_value / 2**20, 1),
                 "kv_buffer_note": "computed from the model shape, not read from the server",
                 "server_rss_mib": round(rss, 1),
                 "output_tok_s": l["output_tok_s"], "ttft_p50_s": l["ttft_s"]["p50"], "itl_ms_p50": l["itl_ms_p50"],
                 "prompt_tokens_mean": l["prompt_tokens_mean"]})
json.dump({"rows": rows}, open("results/kvcache.json", "w"), indent=2)
print(json.dumps(rows, indent=1))
PY
}

promptcache() {  # same 4k-token prompt twice, cache on vs off; fresh server per mode so the first request is cold
  for cp in false true; do
    start_metal q4_k_m
    $PY bench/loadgen.py --url http://127.0.0.1:58080 --label "metal-cache-$cp" --levels 1 --requests-per-level 1 \
      --max-tokens 16 --prompt-tokens 4096 --cache-prompt $cp --out "$RAW/pc_${cp}_1.json"
    $PY bench/loadgen.py --url http://127.0.0.1:58080 --label "metal-cache-$cp" --levels 1 --requests-per-level 1 \
      --max-tokens 16 --prompt-tokens 4096 --cache-prompt $cp --out "$RAW/pc_${cp}_2.json"
    stop_metal
  done
  $PY - <<'PY'
import json
out = {}
for cp in ("false", "true"):
    a, b = (json.load(open(f"results/raw/pc_{cp}_{i}.json"))["levels"][0]["ttft_s"]["p50"] for i in (1, 2))
    out[f"cache_prompt_{cp}"] = {"first_ttft_s": a, "repeat_ttft_s": b}
json.dump({"prompt_tokens": 4096, "backend": "metal", "quant": "q4_k_m", **out}, open("results/promptcache.json", "w"), indent=2)
print(out)
PY
}

perplexity() {
  for quant in q4_k_m q8_0 fp16; do
    [ -f "$(q $quant)" ] || { echo "skip $quant (not downloaded)"; continue; }
    llama-perplexity -m "$(q $quant)" -f bench/text/filler.txt -c 512 -ngl 99 > "$RAW/ppl_$quant.log" 2>&1
  done
  $PY - <<'PY'
import json, os, re
rows = []
for quant in ("q4_k_m", "q8_0", "fp16"):
    f = f"results/raw/ppl_{quant}.log"
    if not os.path.exists(f):
        continue
    t = open(f).read()
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", t)
    size = os.path.getsize(os.path.expanduser(f"~/.cache/mf-models/qwen2.5-1.5b-instruct-{quant}.gguf"))
    rows.append({"quant": quant, "file_mib": round(size / 2**20, 1), "ppl": float(m.group(1)), "ppl_err": float(m.group(2))})
json.dump({"text": "bench/text/filler.txt", "ctx": 512, "rows": rows}, open("results/perplexity.json", "w"), indent=2)
print(rows)
PY
}

case "${1:-all}" in
  all) concurrency; prefill; kvcache; promptcache; perplexity ;;
  *) "$1" ;;
esac
