#!/usr/bin/env bash
# Download the Qwen2.5-1.5B-Instruct GGUF files used by the benchmarks (Apache-2.0, not redistributed here).
set -euo pipefail
M=${MODELS:-$HOME/.cache/mf-models}
BASE=https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main
mkdir -p "$M"
for quant in ${QUANTS:-q4_k_m q8_0 fp16}; do
  f="$M/qwen2.5-1.5b-instruct-$quant.gguf"
  [ -f "$f" ] && { echo "have $f"; continue; }
  curl -fL --retry 3 -o "$f.part" "$BASE/qwen2.5-1.5b-instruct-$quant.gguf" && mv "$f.part" "$f"
done
ls -l "$M"
