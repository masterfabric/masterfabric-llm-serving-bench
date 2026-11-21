# 0005. Speculative decoding only for a large target and predictable output

- Status: Accepted
- Date: 2026-11-21

## Decision

Use a draft model (Qwen2.5-0.5B-Instruct) only when the target is several times larger (7B here)
and the output is predictable (code, SQL, configuration, lists). Do not use it for the 1.5B model,
and do not use long drafts for open-ended text.

## Evidence (`results/spec_*.json`, greedy, 256 tokens, one request at a time)

Decode speed relative to the target alone:

| Target | Draft tokens | Structured prompts | Open prompts |
|---|---|---|---|
| llama.cpp 7B Q4_K_M (51.7 tok/s alone) | 3 / 8 / 16 | 1.48x / **1.60x** / 1.33x | 1.06x / 0.75x / 0.41x |
| MLX 7B 4-bit (59.5 tok/s alone) | 3 / 8 / 16 | **1.73x** / 1.44x / 1.50x | 1.23x / 0.68x / 0.49x |
| llama.cpp 1.5B Q4_K_M (169.9 tok/s alone) | 3 / 8 | 0.68x / 0.71x | 0.42x / 0.31x |

Share of output tokens that came from accepted drafts (7B, 3 draft tokens): 0.66 on both engines.

## Why

Each step costs one draft pass per drafted token plus one target pass over all of them. When the
target is only 3x the draft (1.5B vs 0.5B), that overhead is never paid back, even at 86% acceptance
on structured prompts. On open text the draft is right less often, so long drafts mostly produce
tokens that are thrown away: at 16 draft tokens the 7B runs are 2.0–2.4x slower than without a draft.

## Consequences

- llama.cpp needs `--spec-type draft-simple`; with only `-md` the draft model is loaded and never
  used (found while running this: the first run showed exactly 1.00x).
- Greedy speculative output is not always byte-identical to the target alone. llama.cpp diverged on
  5 of 8 prompts with the 7B target (after 500–1,400 identical characters), MLX on 0–3 of 8,
  depending on the draft length. The target's verification pass runs a batch of tokens, and batched
  and single-token kernels round differently, so a near-tie between two tokens can flip.
- On a GPU server with many concurrent users, batching already fills the GPU and speculative
  decoding gains less; this was measured only with one request at a time.
