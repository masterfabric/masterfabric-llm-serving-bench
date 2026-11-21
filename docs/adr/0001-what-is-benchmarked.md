# 0001. Benchmark servers that run here; no borrowed numbers

- Status: Accepted
- Date: 2026-11-21

## Context

The brief asks for throughput, latency, quantization and KV-cache effects on servers that actually
run. Tailscale machines may be used **read-only**, without changing their services.

## What was available (checked with `tailscale status`, 2026-11-21)

- The M4 Pro itself (14 cores, 24 GB) is the only online machine in the tailnet. Every other node
  (Mac Studio, Mac mini, the other MacBooks, iOS devices) was offline, last seen between 1 and 74
  days earlier. No remote server was benchmarked, and none was contacted.
- Ollama is installed on the host but was not running. It was **not started**: this repo does not
  start or change long-lived services on the machine.
- No NVIDIA GPU, so vLLM could not be run (see masterfabric-onprem-llm-stack ADR 0001).

## Decision

Benchmark the same llama.cpp server code, with the same GGUF files, on the two execution paths that
exist on this machine:

1. **Metal**: Homebrew `llama-server` (build 12359) on the host, all layers offloaded to the Apple GPU.
2. **CPU in a container**: `ghcr.io/ggml-org/llama.cpp:server-b12464` in the Colima VM (6 vCPU), which
   is how the k3d stack in masterfabric-onprem-llm-stack runs.

Both expose the OpenAI API, so the load generator would run unchanged against vLLM, and the
results tables have the same columns a vLLM run would fill.

## Consequences

- Results compare hardware paths and settings for one model family (Qwen2.5-1.5B-Instruct), not servers
  from different vendors.
- The two builds differ slightly (12359 vs 12464). Both are recorded.
