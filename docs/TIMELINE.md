# Build timeline

Build log of masterfabric-llm-serving-bench: measured llama.cpp serving benchmarks on an Apple M4 Pro, Metal vs a CPU container.

Every step below is one commit, in order. Times are Europe/Istanbul (UTC+3).

| # | Commit | Time | Change |
|---|---|---|---|
| 1 | `7fd22ad` | 2026-11-21 04:27 | chore: scaffold repository with AGPL-3.0 license and community files |
| 2 | `d49f5db` | 2026-11-21 04:28 | feat(data): add documentation text for long prompts and perplexity |
| 3 | `765780b` | 2026-11-21 04:28 | feat(bench): add streaming load generator for OpenAI-compatible servers |
| 4 | `6fb2aa7` | 2026-11-21 04:29 | feat(bench): add benchmark matrix runner |
| 5 | `f6d71f7` | 2026-11-21 04:31 | fix(bench): give long prompts an 8k context per slot and fail loudly |
| 6 | `7ef42e3` | 2026-11-21 04:31 | feat(viz): add chart script and shared visual helpers |
| 7 | `21f27ad` | 2026-11-21 04:34 | fix(bench): cold first request for prompt cache and measured server memory |
| 8 | `fcb0f83` | 2026-11-21 04:34 | docs(adr): 0001 what is benchmarked and why |
| 9 | `dc2c430` | 2026-11-21 04:34 | docs(adr): 0002 benchmark methodology and index |
| 10 | `615541f` | 2026-11-21 04:34 | results: prefill throughput, Metal vs CPU container |
| 11 | `b466856` | 2026-11-21 04:34 | results: prompt cache on vs off for a repeated 4k-token prompt |
| 12 | `a7d090c` | 2026-11-21 04:35 | docs: benchmark setup diagram |
| 13 | `a06adaf` | 2026-11-21 04:36 | results: KV cache f16 vs q8_0 at 32k context |
| 14 | `3fb9e5e` | 2026-11-21 04:37 | build: model download script and pinned Python requirements |
| 15 | `a0283f8` | 2026-11-21 04:37 | build: Makefile targets for every experiment, charts and cleanup |
| 16 | `c351079` | 2026-11-21 04:37 | docs: CONTRIBUTING and NOTICE dependency list |
| 17 | `f178f49` | 2026-11-21 04:41 | results: concurrency sweep on Metal, Q4_K_M and Q8_0 |
| 18 | `31eda1e` | 2026-11-21 04:41 | results: concurrency sweep in the CPU container, with a repeat run |
| 19 | `dd324fe` | 2026-11-21 04:41 | results: perplexity of Q4_K_M and Q8_0 on the lab documentation |
| 20 | `7970a03` | 2026-11-21 04:42 | docs(adr): 0003 serving defaults from the measurements |
| 21 | `483179d` | 2026-11-21 04:45 | results: add F16 to the perplexity comparison |
| 22 | `717be39` | 2026-11-21 04:45 | docs: charts for concurrency, prefill, KV cache and quantization |
| 23 | `96a584f` | 2026-11-21 04:45 | docs: 1600x900 hero from the measured charts and numbers |
| 24 | `9f4a759` | 2026-11-21 04:45 | docs(adr): add F16 perplexity to ADR 0003 evidence |
| 25 | `6668af6` | 2026-11-21 04:45 | docs: full README with results, ADRs, mapping, quickstart and limitations |

The final commit adds this file and the README; see `git log` for the full messages, which explain
the why of each step (including the fixes found by running the system).
