"""Compose docs/hero.png (1600x900) from real screenshots and measured numbers.

Every number on the hero is read from results/*.json at render time.
"""

from __future__ import annotations

import json

from PIL import Image, ImageDraw, ImageFilter

from mfviz import BG, BG2, DOCS, GREEN, GREY, ICE, IMG, ORANGE, PURPLE, RED, RESULTS, WHITE, YELLOW, font

W, H = 1600, 900


def load(n):
    return json.loads((RESULTS / f"{n}.json").read_text())


def background():
    im = Image.new("RGB", (W, H), BG)
    px = im.load()
    for y in range(H):
        for x in range(W):
            t = min(1.0, (x / W) * 0.75 + (y / H) * 0.35)
            px[x, y] = tuple(int(BG[i] * (1 - t) + BG2[i] * t) for i in range(3))
    return im


def card(base, path, box, label, color, crop=None):
    x0, y0, x1, y1 = box
    shot = Image.open(path).convert("RGB")
    if crop:
        shot = shot.crop(crop)
    shot = shot.resize((x1 - x0, round(shot.height * (x1 - x0) / shot.width)), Image.LANCZOS)
    shot = shot.crop((0, 0, x1 - x0, min(shot.height, y1 - y0)))
    shadow = Image.new("RGBA", (shot.width + 40, shot.height + 40), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((20, 20, shot.width + 20, shot.height + 20), radius=16, fill=(0, 0, 0, 170))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    base.paste(shadow, (x0 - 20, y0 - 12), shadow)
    mask = Image.new("L", shot.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, shot.width, shot.height), radius=14, fill=255)
    base.paste(shot, (x0, y0), mask)
    d = ImageDraw.Draw(base)
    f = font(20, 800)
    w = d.textlength(label, font=f)
    d.rounded_rectangle((x0 - 10, y0 - 22, x0 + w + 18, y0 + 14), radius=18, fill=color)
    d.text((x0 + 4, y0 - 18), label, font=f, fill=(255, 255, 255))


def main():
    cm = load("concurrency_metal_q4_k_m")["levels"]
    cc = load("concurrency_cpu_q4_k_m")["levels"]
    pre = load("prefill")["rows"]
    kv = {r["kv_type"]: r for r in load("kvcache")["rows"]}
    pc = load("promptcache")
    ppl = {r["quant"]: r for r in load("perplexity")["rows"]}
    bc8 = max(load("concurrency_cpu_q8_0")["levels"], key=lambda l: l["output_tok_s"])
    bm, bc = max(cm, key=lambda l: l["output_tok_s"]), max(cc, key=lambda l: l["output_tok_s"])
    pm = max(r["prefill_tok_s"] for r in pre if r["backend"] == "metal")
    pcpu = max(r["prefill_tok_s"] for r in pre if r["backend"] == "cpu")
    im = background()
    card(im, IMG / "concurrency.png", (960, 70, 1270, 360), "throughput", GREEN, crop=(0, 0, 850, 747))
    card(im, IMG / "kv_quant.png", (1300, 70, 1570, 300), "KV cache", PURPLE, crop=(0, 0, 850, 715))
    card(im, IMG / "prefill.png", (960, 420, 1570, 620), "prefill: Metal vs CPU", ORANGE)
    card(im, IMG / "kv_quant.png", (990, 680, 1500, 880), "prompt cache · perplexity", ICE, crop=(850, 0, 2542, 715))
    d = ImageDraw.Draw(im)
    d.text((80, 70), "MASTERFABRIC", font=font(36, 800), fill=WHITE)
    d.text((76, 120), "LLM Serving Bench", font=font(84, 800), fill=WHITE)
    x, y = 80, 238
    for word, col in (("llama.cpp", ICE), (" · ", GREY), ("Metal vs CPU", ORANGE), (" · ", GREY), ("KV", PURPLE),
                      (" · ", GREY), ("quantization", GREEN)):
        f = font(38, 800)
        d.text((x, y), word, font=f, fill=col)
        x += d.textlength(word, font=f)
    d.text((80, 318), "Throughput, TTFT, inter-token latency, prefill, KV-cache and", font=font(26, 500), fill=GREY)
    d.text((80, 352), "quantization trade-offs, measured on one M4 Pro. No borrowed numbers.", font=font(26, 500), fill=GREY)
    q4, q8 = ppl["q4_k_m"], ppl["q8_0"]
    facts = [
        (f"{bm['output_tok_s']:.0f} tok/s Metal vs {bc['output_tok_s']:.0f} tok/s CPU",
         f"Qwen2.5-1.5B Q4_K_M, 4 slots · CPU container with Q8_0: {bc8['output_tok_s']:.0f} tok/s", GREEN),
        (f"prefill {pm:,.0f} vs {pcpu:,.0f} prompt tok/s",
         f"Metal vs CPU · a 4k-token prompt costs {pc['cache_prompt_false']['first_ttft_s']:.2f} s, "
         f"{pc['cache_prompt_true']['repeat_ttft_s'] * 1000:.0f} ms from the prompt cache", ORANGE),
        (f"KV q8_0: {kv['f16']['kv_buffer_mib']:.0f} → {kv['q8_0']['kv_buffer_mib']:.0f} MiB at 32k",
         f"same decode speed ({kv['f16']['output_tok_s']:.1f} vs {kv['q8_0']['output_tok_s']:.1f} tok/s), "
         f"server RSS {kv['f16']['server_rss_mib']:,.0f} → {kv['q8_0']['server_rss_mib']:,.0f} MiB", PURPLE),
        (f"Q4_K_M PPL {q4['ppl']:.2f} vs Q8_0 {q8['ppl']:.2f}",
         f"{q4['file_mib']:,.0f} vs {q8['file_mib']:,.0f} MiB file" +
         (f" · F16 {ppl['fp16']['ppl']:.2f} ({ppl['fp16']['file_mib']:,.0f} MiB)" if "fp16" in ppl else ""), ICE),
    ]
    y = 430
    for title, sub, col in facts:
        d.rounded_rectangle((80, y + 6, 88, y + 62), radius=4, fill=col)
        d.text((106, y), title, font=font(29, 750), fill=WHITE)
        d.text((106, y + 38), sub, font=font(20, 450), fill=GREY)
        y += 92
    d.text((80, 820), "github.com/masterfabric/masterfabric-llm-serving-bench", font=font(30, 700), fill=WHITE)
    im.save(DOCS / "hero.png", optimize=True)
    print("docs/hero.png written")


if __name__ == "__main__":
    main()
