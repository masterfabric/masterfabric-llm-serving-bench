# 0004. Comparing MLX with llama.cpp: same tokens, same GPU, quality checked next to speed

- Status: Accepted
- Date: 2026-11-21

## Context

On Apple Silicon the two practical local engines are llama.cpp (GGUF, Metal) and Apple's MLX
(`mlx-lm`). Speed comparisons between them are easy to get wrong: the "4-bit" formats are not the same
size, servers differ in how they batch, and one engine can stop generating earlier than the other.

## Decision

- **Same machine, same model family, matched formats**: Qwen2.5-1.5B-Instruct as GGUF Q4_K_M
  (1,066 MiB) against MLX 4-bit (828 MiB, group size 64), and Q8_0 (1,807 MiB) against MLX 8-bit
  (1,564 MiB). The files are not the same size, so the results say so.
- **Single stream, in process**: `bench/ctx_curve.py` feeds both engines the *same token ids*
  (tokenized once), generates exactly 128 tokens greedily with EOS ignored, and runs each point in
  a fresh process under `/usr/bin/time -l` for the peak memory footprint. llama.cpp runs with
  `-lm none` (no mmap) so its weights count in the footprint as MLX's do.
- **Server**: `llama-server -np 4` against `mlx_lm.server --decode-concurrency 4
  --prompt-concurrency 4`, both driven by `bench/loadgen.py` with EOS respected (the MLX server has
  no `ignore_eos`), two runs each.
- **Quality next to speed**: perplexity of the MLX weights with the llama-perplexity method on the
  same text (`bench/mlx_perplexity.py`). Each engine's 4-bit is compared with its own 8-bit, which
  cancels small differences between the two perplexity implementations.
- The user's own MLX server on port 11234 is not used; the benchmark starts its own on 58082.

## Result

| | llama.cpp Q4_K_M | MLX 4-bit | llama.cpp Q8_0 | MLX 8-bit |
|---|---|---|---|---|
| Decode, 512-token prompt | 171 tok/s | **210 tok/s** | 122 tok/s | 128 tok/s |
| Prefill, 4k prompt | 2,010 tok/s | **2,131 tok/s** | 2,086 tok/s | 2,252 tok/s |
| Server peak (4–8 concurrent, mean of 2 runs) | 378 tok/s at 8 | **474 tok/s at 4** | 351 tok/s at 8 | 329 tok/s at 8 |
| Perplexity vs own 8-bit | **+4.5%** (24.69 vs 23.63) | +17.5% (27.26 vs 23.21) | – | – |

## Consequences

- MLX is faster at 4-bit, but its 4-bit file is 22% smaller and loses clearly more quality. Q4_K_M
  keeps more precision where it matters; per unit of quality the comparison is much closer.
- At 8-bit the engines are within 5–10% of each other on one request; under load llama.cpp's slot
  scheduler kept higher throughput (340 vs 269 tok/s at 4 concurrent).
- MLX's throughput fell from 4 to 8 concurrent requests (474 → 360 tok/s mean), and one of the 24
  requests at 8 concurrent failed on the client side in both runs and for both MLX models; llama.cpp
  had no errors. The cause was not investigated.
- llama.cpp stays the default for the on-prem stack (one engine for Metal, CPU and CUDA, GGUF in
  PVCs). MLX is the faster choice for a single Mac serving a few users with 4-bit weights, if the
  quality loss is acceptable for the task.
