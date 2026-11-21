# 0006. Long context: measure the process, not the formula

- Status: Accepted
- Date: 2026-11-21

## Context

ADR 0003 computed the KV buffer for 32k context from the model shape. A server needs the memory of the
whole process: weights, KV cache, compute buffers and anything temporary during prefill.

## Decision

Measure each prompt length (512 to 31,744 tokens, 128 generated) in a fresh process and record the
**peak memory footprint** from `/usr/bin/time -l`, which counts Metal buffers in unified memory.
Record each engine's own accounting next to it.

## Result (`results/ctx_curve.json`, Qwen2.5-1.5B, 4-bit weights)

| Prompt tokens | llama.cpp f16 KV | llama.cpp q8_0 KV | MLX f16 KV | MLX 8-bit KV |
|---|---|---|---|---|
| 512 | 1.40 GiB | 1.40 GiB | 1.86 GiB | 1.86 GiB |
| 16,384 | 1.83 GiB | 1.63 GiB | 2.62 GiB | 4.11 GiB |
| 31,744 | 2.25 GiB | **1.85 GiB** | 3.40 GiB | 5.14 GiB |
| Decode at 31,744 | 95.5 tok/s | 87.4 tok/s | **113.9 tok/s** | 92.5 tok/s |
| Prefill of 31,744 tokens | 31.1 s | 31.3 s | **25.2 s** | 32.6 s |

## Consequences

- llama.cpp grows by almost exactly its KV buffer (+866 MiB from 512 to 31,744 tokens; the server
  logs 875 MiB of KV), because it allocates the cache up front. A q8_0 cache halves that growth and
  costs 8–10% decode speed for one request (ADR 0003 measured no speed cost at 4 concurrent
  requests, where the GPU is busier with batched work).
- MLX grows by more than the KV cache (+1,577 MiB) because prefill keeps temporary attention buffers.
  It is still the faster engine at every length.
- MLX's 8-bit KV cache (`kv_bits=8`, quantized from the first token) *raised* peak memory in this
  version (mlx-lm 0.32.2): 5.14 vs 3.40 GiB at 31k. Do not assume a quantized cache saves memory;
  measure it. The cause was not investigated further.
- Decode speed falls with context on every engine (llama.cpp f16: 171 → 96 tok/s from 512 to 31k
  tokens), because every new token attends over the whole cache.
