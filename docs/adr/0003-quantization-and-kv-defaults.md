# 0003. Serving defaults: weight and KV quantization, slots, prompt cache

- Status: Accepted (for this lab's hardware; re-measure on new hardware)
- Date: 2026-11-21

## Context

The on-prem stack (masterfabric-onprem-llm-stack) shipped with Q4_K_M weights, an f16 KV cache,
4 slots and the prompt cache on, without measurements behind those choices. This repo measured them.

## Evidence (results/*.json)

| Question | Measurement | Result |
|---|---|---|
| Q4_K_M or Q8_0 weights? | perplexity, 38 KB lab docs, ctx 512 | 24.69 ± 1.09 vs 23.63 ± 1.04 (F16: 23.62 ± 1.04), within the error bars; file 1,066 vs 1,807 MiB (F16 3,396) |
| | Metal throughput, 4 concurrent | 381 vs 358 tok/s |
| | CPU container throughput, 4 concurrent | 184 vs 328 tok/s (reproduced: 181 vs 335) |
| f16 or q8_0 KV cache? | 32k context, 4 × ~5.5k-token prompts, Metal | KV buffer 896 vs 476 MiB (computed), server RSS 2,746 vs 2,042 MiB, 38.2 vs 37.9 tok/s |
| How many concurrent requests per replica? | 1-8 concurrent, 4 slots | throughput peaks at 4; at 8, TTFT p50 jumps from ~14 ms to ~1.5 s (queueing) |
| Prompt cache on? | 4k-token prompt sent twice | 1.89 s TTFT cold, 13 ms on the repeat with cache; no cost on the cold request |

## Decision

- **GPU/Metal nodes: Q4_K_M weights.** Slightly faster than Q8_0 and 41% smaller, with no
  measurable perplexity loss on our text.
- **CPU-only nodes: Q8_0 weights.** Counter-intuitive but measured twice: on this ARM CPU build Q8_0
  decodes batched requests much faster than Q4_K_M (11.7 vs 20.7 ms per token at 4 concurrent). The
  extra 740 MiB of RAM is cheap compared with the throughput gain. Re-measure on x86 nodes before
  copying this decision.
- **KV cache q8_0 with flash attention** when long contexts are needed: half the KV memory at the
  same decode speed. Keep f16 for short contexts, where the KV buffer is small anyway.
- **Concurrency = slots.** Size the replica count so steady-state concurrency per replica stays at
  or below the slot count; above it, requests wait for a free slot and TTFT becomes seconds.
- **Prompt cache on** (llama.cpp default). It is free for cold requests and removes prefill for
  repeated system prompts and RAG templates.

## Consequences

- The on-prem stack's Helm values keep Q4_K_M as the default; a CPU-node values file should switch
  to Q8_0. That change belongs in that repository and is not made here.
- Perplexity on 38 KB of text is a coarse quality signal. Task-level evaluation (masterfabric-onprem-llm-stack
  `eval/`) should be re-run before a quantization change goes to production.
