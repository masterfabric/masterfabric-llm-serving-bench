#!/usr/bin/env bash
# MLX vs llama.cpp on the same Apple GPU. Servers are started and stopped here; nothing else is touched.
#   llama-server (Homebrew, Metal) 127.0.0.1:58080  ·  mlx_lm.server 127.0.0.1:58082
# Usage: bench/run_mlx.sh [concurrency|single|ctx|spec|all]
set -euo pipefail
cd "$(dirname "$0")/.."
M=${MODELS:-$HOME/.cache/mf-models}
PY=.venv/bin/python
RES=results; RAW=results/raw; mkdir -p "$RES" "$RAW"

wait_up() { for _ in $(seq 1 180); do curl -sf "$1/health" >/dev/null 2>&1 && return 0; sleep 1; done; echo "server $1 did not start"; return 1; }
wait_mlx() { for _ in $(seq 1 180); do curl -sf "$1/v1/models" >/dev/null 2>&1 && return 0; sleep 1; done; echo "mlx server did not start"; return 1; }

concurrency() {  # same prompts, 128 tokens, EOS respected on both (mlx_lm.server has no ignore_eos)
  for pair in "q4_k_m:Qwen2.5-1.5B-Instruct-4bit" "q8_0:Qwen2.5-1.5B-Instruct-8bit"; do
    gguf=${pair%%:*}; mlx=${pair##*:}
    llama-server -m "$M/qwen2.5-1.5b-instruct-$gguf.gguf" -a qwen --host 127.0.0.1 --port 58080 -ngl 99 -np 4 -c 16384 \
      > "$RAW/mlxcmp-llamacpp-$gguf.log" 2>&1 &
    pid=$!; wait_up http://127.0.0.1:58080
    $PY bench/loadgen.py --url http://127.0.0.1:58080 --label "llamacpp-$gguf" --levels 1,2,4,8 --no-ignore-eos \
      --meta "{\"engine\":\"llama.cpp\",\"weights\":\"$gguf\",\"slots\":4}" --out "$RAW/mlxcmp_llamacpp_$gguf.json"
    kill $pid; wait $pid 2>/dev/null || true; sleep 2
    $PY -m mlx_lm.server --model "$M/mlx/$mlx" --host 127.0.0.1 --port 58082 --decode-concurrency 4 \
      --prompt-concurrency 4 --log-level WARNING > "$RAW/mlxcmp-mlx-$mlx.log" 2>&1 &
    pid=$!; wait_mlx http://127.0.0.1:58082
    $PY bench/loadgen.py --url http://127.0.0.1:58082 --model "$M/mlx/$mlx" --label "mlx-$mlx" --levels 1,2,4,8 \
      --no-ignore-eos --meta "{\"engine\":\"mlx\",\"weights\":\"$mlx\",\"decode_concurrency\":4}" \
      --out "$RAW/mlxcmp_mlx_$mlx.json"
    kill $pid; wait $pid 2>/dev/null || true; sleep 2
  done
  $PY - <<'PY'
import glob, json
runs = [json.load(open(f)) for f in sorted(glob.glob("results/raw/mlxcmp_*.json"))]
json.dump({"note": "128 max tokens, EOS respected, short prompts from bench/loadgen.py", "runs": runs},
          open("results/mlx_concurrency.json", "w"), indent=2)
for r in runs:
    print(r["label"], [(l["concurrency"], l["output_tok_s"], l["ttft_s"]["p50"]) for l in r["levels"]])
PY
}

single() {  # 8-bit pair at two prompt lengths, same harness as the long-context curve
  $PY bench/ctx_curve.py --gguf "$M/qwen2.5-1.5b-instruct-q8_0.gguf" --mlx "$M/mlx/Qwen2.5-1.5B-Instruct-8bit" \
    --points 512,4096 --configs llamacpp:f16,mlx:f16 --out "$RES/single_8bit.json"
}

ctx() { $PY bench/ctx_curve.py --out "$RES/ctx_curve.json"; }

spec() {
  $PY bench/spec_decode.py --engine llamacpp --target 7b --out "$RES/spec_llamacpp_7b.json"
  $PY bench/spec_decode.py --engine mlx --target 7b --out "$RES/spec_mlx_7b.json"
  $PY bench/spec_decode.py --engine llamacpp --target 1.5b --drafts 3,8 --out "$RES/spec_llamacpp_1.5b.json"
}

case "${1:-all}" in
  all) single; ctx; concurrency; spec ;;
  *) "$1" ;;
esac
