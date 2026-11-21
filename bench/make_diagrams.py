"""Benchmark setup diagram (Pillow)."""
from mfviz import GREEN, GREY, ICE, IMG, ORANGE, PURPLE, RED, WHITE, YELLOW, arrow, box, canvas, font


def architecture():
    im, d = canvas(1600, 900)
    d.text((60, 40), "MasterFabric LLM Serving Bench · setup", font=font(36, 800), fill=WHITE)
    d.text((60, 90), "One load generator, one OpenAI-compatible API, two execution paths on the same Apple M4 Pro (24 GB).",
           font=font(20, 450), fill=GREY)
    box(d, (50, 250, 400, 650), "bench/loadgen.py", ["async httpx, streaming", "N requests in flight", "ignore_eos · 128 tokens",
                                                     "timestamps every delta", "→ TTFT, ITL, e2e,", "   tokens/s, req/s",
                                                     "cold prompts from", "   real lab docs"], GREEN, line_size=18)
    box(d, (500, 330, 800, 570), "run_matrix.sh", ["concurrency 1-8", "prefill 256-6k", "KV f16 vs q8_0",
                                                   "prompt cache", "perplexity"], YELLOW, line_size=18)
    box(d, (900, 140, 1550, 400), "Metal · host :58080", ["Homebrew llama-server build 12359", "-ngl 99 (all layers on Apple GPU)",
                                                         "-np 4 slots · 16k / 32k context", "flash attention for KV tests"],
        ICE, line_size=18)
    box(d, (900, 460, 1550, 720), "CPU · Colima container :58081", ["ghcr.io/ggml-org/llama.cpp:server-b12464",
                                                                     "6 vCPU of the Colima VM, -t 6", "-np 4 slots · 16k / 32k context",
                                                                     "same image as the k3d stack"], ORANGE, line_size=18)
    box(d, (500, 690, 800, 860), "GGUF files", ["Qwen2.5-1.5B-Instruct", "Q4_K_M · Q8_0 · F16"], PURPLE, line_size=18)
    box(d, (50, 720, 400, 860), "results/*.json", ["→ make_charts.py", "→ docs/images, README"], RED, line_size=18)
    arrow(d, (400, 450), (500, 450), GREEN)
    arrow(d, (800, 370), (900, 280), ICE, "/v1/chat/completions", at=0.4)
    arrow(d, (800, 540), (900, 640), ORANGE, "/v1/chat/completions", at=0.35)
    arrow(d, (800, 770), (900, 700), PURPLE, "mount")
    arrow(d, (225, 650), (225, 720), RED)
    d.text((900, 760), "Not run: vLLM (no NVIDIA GPU), Ollama (not running,", font=font(18, 500), fill=GREY)
    d.text((900, 788), "left alone), remote tailnet hosts (all offline).", font=font(18, 500), fill=GREY)
    im.save(IMG / "architecture.png", optimize=True)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    architecture()
    print("diagram written")
