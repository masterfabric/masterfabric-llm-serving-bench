#!/usr/bin/env bash
# Models for the MLX, speculative-decoding and long-context experiments (all Apache-2.0, not redistributed).
#   GGUF: Qwen2.5-7B-Instruct Q4_K_M (target) and Qwen2.5-0.5B-Instruct Q8_0 (draft) for llama.cpp
#   MLX:  Qwen2.5-1.5B-Instruct 4bit/8bit, Qwen2.5-7B-Instruct 4bit, Qwen2.5-0.5B-Instruct 8bit (mlx-community)
set -euo pipefail
cd "$(dirname "$0")/.."
M=${MODELS:-$HOME/.cache/mf-models}
mkdir -p "$M/mlx"
get() {  # url file
  [ -f "$M/$2" ] && { echo "have $2"; return; }
  curl -fL --retry 3 -o "$M/$2.part" "$1" && mv "$M/$2.part" "$M/$2"
}
get https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf qwen2.5-7b-instruct-q4_k_m.gguf
get https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf qwen2.5-0.5b-instruct-q8_0.gguf
.venv/bin/python - "$M/mlx" <<'PY'
import sys
from huggingface_hub import snapshot_download
for repo in ["mlx-community/Qwen2.5-1.5B-Instruct-4bit", "mlx-community/Qwen2.5-1.5B-Instruct-8bit",
             "mlx-community/Qwen2.5-0.5B-Instruct-8bit", "mlx-community/Qwen2.5-7B-Instruct-4bit"]:
    p = snapshot_download(repo, local_dir=f"{sys.argv[1]}/{repo.split('/')[1]}")
    print("have", p)
PY
# pinned: the Hugging Face LFS sha256 of the files the results were measured with
( cd "$M" && shasum -a 256 -c - <<'SUMS'
65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423  qwen2.5-7b-instruct-q4_k_m.gguf
ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e  qwen2.5-0.5b-instruct-q8_0.gguf
SUMS
)
