# Architecture decision records

| # | Decision |
|---|---|
| [0001](0001-what-is-benchmarked.md) | Benchmark servers that run here; no borrowed numbers |
| [0002](0002-methodology.md) | Methodology: streaming client, fixed output length, cold prompts |
| [0003](0003-quantization-and-kv-defaults.md) | Serving defaults: weight and KV quantization, slots, prompt cache |
| [0004](0004-mlx-vs-llamacpp.md) | Comparing MLX with llama.cpp: same tokens, same GPU, quality checked next to speed |
| [0005](0005-speculative-decoding.md) | Speculative decoding only for a large target and predictable output |
| [0006](0006-long-context-memory.md) | Long context: measure the process, not the formula |
