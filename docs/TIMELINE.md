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


## Addition: MLX vs llama.cpp, speculative decoding, long-context memory curve

Goal: compare Apple's MLX with llama.cpp on the same GPU, measure speculative decoding with a draft
model, and measure memory and speed from 512 to 31k tokens of context. Times are Europe/Istanbul (UTC+3).

| # | Commit | Time | Change |
|---|---|---|---|
| 1 | `68b08a6` | 2026-11-21 12:33 | bench: long-context curve (512 to 31k tokens), llama.cpp vs MLX, one process per point with peak footprint |
| 2 | `070b6e1` | 2026-11-21 12:51 | bench: speculative decoding (target vs target + 0.5B draft) for llama.cpp and MLX; run_mlx.sh |
| 3 | `99f9d09` | 2026-11-21 12:51 | results: llama.cpp speculative decoding, 7B and 1.5B targets with a 0.5B draft |
| 4 | `9fca0e2` | 2026-11-21 12:54 | results: long-context curve 512-31k tokens, llama.cpp f16/q8_0 KV vs MLX f16/8-bit KV |
| 5 | `0a64390` | 2026-11-21 12:55 | results: 8-bit pair (Q8_0 vs MLX 8-bit) single stream at 512 and 4k tokens |
| 6 | `43f50d0` | 2026-11-21 12:59 | results: MLX server vs llama-server, 1-8 concurrent, 4-bit and 8-bit, two runs |
| 7 | `3825e19` | 2026-11-21 13:00 | bench: MLX perplexity with the llama-perplexity method; 4-bit 27.26 vs 8-bit 23.21 |
| 8 | `671efd7` | 2026-11-21 13:05 | bench: pin sha256 of the 7B and 0.5B GGUF files |
| 9 | `18e38fc` | 2026-11-21 13:07 | results: MLX speculative decoding, 7B 4-bit target with 0.5B 8-bit draft |
| 10 | `81e2d1f` | 2026-11-21 13:08 | docs: MLX vs llama.cpp, long-context memory and speculative decoding charts |
| 11 | `6aa1702` | 2026-11-21 13:09 | docs: setup diagram with the MLX server and in-process runs |
| 12 | `6322e42` | 2026-11-21 13:10 | docs(adr): 0004 MLX vs llama.cpp, 0005 speculative decoding, 0006 long-context memory |
| 13 | `0fd3d74` | 2026-11-21 13:10 | NOTICE and Makefile: MLX, extra models, mlx/ctx/spec targets |
| 14 | `da239b5` | 2026-11-21 13:12 | README: MLX vs llama.cpp, long-context curve and speculative decoding with figures; limitations |
| 15 | (this commit) | | docs(timeline): this section |

### Verification (measured on the M4 Pro)

| Check | Result |
|---|---|
| MLX install | mlx 0.32.4, mlx-lm 0.32.2 in `.venv`, default device GPU |
| Model files | 7B Q4_K_M and 0.5B Q8_0 sha256 match the Hugging Face LFS hashes (pinned in `get_models_extra.sh`) |
| MLX vs llama.cpp, one request (4-bit) | decode 210 vs 171 tok/s; prefill at 4k 2,131 vs 2,010 tok/s |
| MLX vs llama.cpp, server (4-bit) | 474 vs 351 tok/s at 4 concurrent, mean of 2 runs |
| Perplexity vs own 8-bit | MLX 4-bit +17.5%, Q4_K_M +4.5% |
| Long context at 31,744 tokens | llama.cpp 2.25 GiB (1.85 with q8_0 KV), 96 tok/s; MLX 3.40 GiB, 114 tok/s; MLX 8-bit KV 5.14 GiB |
| Speculative, 7B + 0.5B draft | structured 1.60x (llama.cpp, 8 drafts) / 1.73x (MLX, 3 drafts); open 0.41-0.49x at 16 drafts |
| Speculative, 1.5B + 0.5B draft | 0.31-0.71x (always slower) |
| MacBook Air | offline (not reachable), not measured |

### Found while running it

- With only `-md`, llama-server loads the draft model and never uses it: the default `--spec-type` is
  `none`. The first run measured exactly 1.00x; `--spec-type draft-simple` fixed it.
- llama-server prints its buffer sizes only at log verbosity 4, and that verbosity slowed decoding by
  about 20% (125 vs 153 tok/s). Timings now come from a quiet run and buffer sizes from a second,
  load-only start.
- With the default mmap loading, the weights did not show in llama.cpp's peak memory footprint
  (181 MiB at 512 tokens). The curve runs with `-lm none` so both engines count their weights.
- MLX's 8-bit KV cache raised peak memory instead of lowering it (5.14 vs 3.40 GiB at 31k tokens).
- Greedy decoding with a draft model was not always byte-identical to the target alone, more often
  on llama.cpp than on MLX.
- `mlx_lm.server` has no `ignore_eos`, so the server comparison respects EOS on both engines; at 8
  concurrent requests 3 of 96 MLX requests failed on the client side.
- The user's own MLX server on port 11234 was not used; the benchmark started and stopped its own on
  58082.


## Addition: why draft output differs, and the MacBook Air M5

Goal: find the cause of the 5-of-8 greedy output mismatches with a draft model in llama.cpp, and
measure the MacBook Air M5 if it is reachable. On 2026-11-23 the M4 Pro was offline and the Air was
reachable, so the measurements ran on the Air (on battery) and the commits were prepared from those
results. Times are Europe/Istanbul (UTC+3); the run times on the Air were 09:50-10:27.

| # | Commit | Time | Change |
|---|---|---|---|
| 1 | `38b9cae` | 2026-11-23 10:29 | bench: spec_mismatch.py finds where draft output diverges and how close the target's top two tokens were |
| 2 | `a2fe7e6` | 2026-11-23 10:29 | results: speculative-decoding mismatch study, 7B and 1.5B targets (MacBook Air M5) |
| 3 | `98330a1` | 2026-11-23 10:29 | results: MacBook Air M5 runs, llama.cpp Metal concurrency and prefill, MLX vs llama.cpp |
| 4 | `42f740a` | 2026-11-23 10:29 | docs: Air M5 vs M4 Pro chart and speculative-mismatch chart |
| 5 | `9948d49` | 2026-11-23 10:30 | docs(adr): 0005 update with the measured cause of the draft/no-draft output mismatch |
| 6 | `969272a` | 2026-11-23 10:30 | build: make spec-mismatch target |
| 7 | `3feb360` | 2026-11-23 10:30 | README: why draft output differs (near-ties, batched kernels) and MacBook Air M5 vs M4 Pro, with figures 8-9; limitations for the battery runs |
| 8 | `176a8a0` | 2026-11-23 10:30 | build: spec-mismatch in the single .PHONY line |
| 9 | (this commit) | | docs(timeline): this section |

### Verification

| Check | Result |
|---|---|
| Air hardware | Apple M5, 4P+6E CPU cores, 10-core GPU, 24 GB, macOS 27.2, battery 43% → 21% |
| llama.cpp | official b12359 macOS arm64 release (same build number as the M4 Pro host) |
| Model files | 7B Q4_K_M and 0.5B Q8_0 sha256 match the pinned hashes |
| Target alone, repeated | 8/8 identical (7B and 1.5B) |
| With a 3-token draft | 7B 3/8 identical (M4 Pro run: 3/8), 1.5B 6/8 (M4 Pro: 6/8) |
| Gap at the divergences | 0.002-0.034 nats (7B), 0.005-0.008 (1.5B); median over all tokens 5.36 / 3.22 |
| Batched prefill, no draft | picks the draft run's token in 3/5 (7B) and 2/2 (1.5B) cases |
| Air vs M4 Pro decode | 0.57-0.75x (llama.cpp, 1 and 4 requests); prefill 1.25x at ~1k tokens, 0.82x at 3.8k |
| Air clean-up | the temporary folder (llama.cpp binary, uv, Python, models, repo copy, logs) and the uv receipt deleted; no process left running |

### Found while running it

- The divergence counts on the Air equal those on the M4 Pro (5/8 and 2/8), with a different GPU, so
  the effect is not tied to one chip.
- A batched prefill of the target alone reproduces the draft run's token in most but not all cases,
  because a whole-prompt batch and a 4-token verification batch are different shapes.
- `n_probs` responses from llama-server give log-probabilities before sampling, which is what the
  margin needs; decode speed with `n_probs` is lower, so these runs are not speed benchmarks.
- Q8_0 at 4 concurrent on the Air measured 213 tok/s in one sweep and 131 tok/s in another 15 minutes
  later; on battery and fanless this was not explained and is listed as a limitation.
- The first 7B mismatch run was interrupted halfway and restarted; only the complete second run is
  in `results/air/`.
