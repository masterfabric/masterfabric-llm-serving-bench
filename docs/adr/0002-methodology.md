# 0002. Methodology: streaming client, fixed output length, cold prompts

- Status: Accepted
- Date: 2026-11-21

## Decision

- **Streaming requests** to `/v1/chat/completions`. The client timestamps every content delta, which
  gives TTFT (time to the first token), inter-token latency (median gap per request) and end-to-end
  latency.
- **Fixed output length.** `ignore_eos` makes every request generate exactly `max_tokens` (128), so
  tokens/s compare like with like. Greedy decoding (temperature 0).
- **Fixed request counts.** Each concurrency level runs max(8, 3 × concurrency) requests with N in
  flight, after one unrecorded warm-up request.
- **4 server slots, 16k context** (`-np 4 -c 16384`) for the concurrency sweep. Prefill and KV tests
  use a 32k context, 8k per slot.
- **Cold prompts for prefill.** Long prompts are cut from real documentation with a different offset
  per request, and sent with `cache_prompt: false`. Prompt-cache reuse is measured separately, with a
  fresh server for each mode.
- **Quality of a quantization**: `llama-perplexity` on the same 38 KB documentation text, context 512,
  for every GGUF file.

## Not controlled

Other apps on the laptop, thermal state, and the download that ran in the background during the
first runs. Each configuration ran once; there are no confidence intervals beyond the perplexity error.
