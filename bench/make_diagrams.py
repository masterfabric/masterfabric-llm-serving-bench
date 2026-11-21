"""Benchmark setup diagram (Pillow)."""
from mfviz import GREEN, GREY, ICE, IMG, ORANGE, PURPLE, RED, WHITE, YELLOW, arrow, box, canvas, font


def architecture():
    im, d = canvas(1600, 1000)
    d.text((60, 40), "MasterFabric LLM Serving Bench · setup", font=font(36, 800), fill=WHITE)
    d.text((60, 90), "One load generator, OpenAI-compatible APIs, three execution paths on the same Apple M4 Pro (24 GB).",
           font=font(20, 450), fill=GREY)
    box(d, (50, 230, 400, 630), "bench/loadgen.py", ["/v1/chat/completions, streaming", "N requests in flight", "128 tokens",
                                                     "timestamps every delta", "→ TTFT, ITL, e2e,", "   tokens/s, req/s",
                                                     "cold prompts from", "   real lab docs"], GREEN, line_size=18)
    box(d, (480, 230, 820, 520), "bench/run_*.sh", ["concurrency 1-8", "prefill 256-6k", "KV f16 vs q8_0",
                                                               "prompt cache · perplexity", "MLX vs llama.cpp",
                                                               "context 512 → 31k tokens", "speculative decoding"],
        YELLOW, line_size=18)
    box(d, (480, 560, 820, 760), "In-process MLX", ["ctx_curve.py, spec_decode.py,", "mlx_perplexity.py",
                                                    "/usr/bin/time -l for", "peak memory footprint"], YELLOW, line_size=17)
    box(d, (900, 130, 1550, 340), "Metal · llama-server :58080", ["Homebrew build 12359, -ngl 99 (Apple GPU)",
                                                                  "-np 4 slots · 16k / 32k context",
                                                                  "draft model via --spec-type draft-simple"], ICE, line_size=18)
    box(d, (900, 380, 1550, 590), "Metal · mlx_lm.server :58082", ["mlx 0.32.4 · mlx-lm 0.32.2",
                                                                   "decode/prompt concurrency 4",
                                                                   "MLX 4-bit / 8-bit safetensors"], ORANGE, line_size=18)
    box(d, (900, 630, 1550, 840), "CPU · Colima container :58081", ["ghcr.io/ggml-org/llama.cpp:server-b12464",
                                                                     "6 vCPU of the Colima VM, -t 6", "-np 4 slots"],
        PURPLE, line_size=18)
    box(d, (50, 680, 400, 840), "results/*.json", ["→ make_charts.py", "→ docs/images, README"], RED, line_size=18)
    arrow(d, (400, 400), (480, 400), GREEN)
    arrow(d, (820, 300), (900, 235), ICE)
    arrow(d, (820, 420), (900, 485), ORANGE)
    arrow(d, (820, 500), (900, 720), PURPLE)
    arrow(d, (225, 630), (225, 680), RED)
    d.text((60, 890), "Models: Qwen2.5-1.5B-Instruct (GGUF Q4_K_M · Q8_0 · F16; MLX 4-bit · 8-bit), Qwen2.5-7B-Instruct "
           "(Q4_K_M; MLX 4-bit), Qwen2.5-0.5B-Instruct draft (Q8_0; MLX 8-bit).", font=font(17, 500), fill=GREY)
    d.text((60, 920), "Not run: vLLM (no NVIDIA GPU), Ollama, and the MacBook Air and other remote hosts (offline). "
           "The user's own MLX server on :11234 was left alone.", font=font(17, 500), fill=GREY)
    im.save(IMG / "architecture.png", optimize=True)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    architecture()
    print("diagram written")
