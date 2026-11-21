"""Charts from results/*.json (matplotlib, dark theme)."""
from __future__ import annotations

import json

from mfviz import HEX, IMG, RESULTS, mpl_setup

plt = mpl_setup()
COL = {("metal", "q4_k_m"): HEX["GREEN"], ("metal", "q8_0"): HEX["ICE"],
       ("cpu", "q4_k_m"): HEX["ORANGE"], ("cpu", "q8_0"): HEX["YELLOW"]}


def J(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def concurrency():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for b in ("metal", "cpu"):
        for q in ("q4_k_m", "q8_0"):
            try:
                d = J(f"concurrency_{b}_{q}")
            except FileNotFoundError:
                continue
            x = [l["concurrency"] for l in d["levels"]]
            lab = f"{'Metal GPU' if b == 'metal' else 'CPU container'} · {q.upper()}"
            axes[0].plot(x, [l["output_tok_s"] for l in d["levels"]], "-o", color=COL[(b, q)], label=lab)
            axes[1].plot(x, [l["ttft_s"]["p50"] * 1000 for l in d["levels"]], "-o", color=COL[(b, q)], label=lab)
            axes[2].plot(x, [l["itl_ms_p50"] for l in d["levels"]], "-o", color=COL[(b, q)], label=lab)
    for a, t, y in zip(axes, ("Aggregate output throughput", "Time to first token (p50)", "Inter-token latency (p50)"),
                       ("output tokens/s", "ms", "ms per token")):
        a.set_xscale("log", base=2)
        a.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        a.set_xlabel("concurrent requests (4 server slots)")
        a.set_title(t, loc="left")
        a.set_ylabel(y)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "concurrency.png")
    plt.close(fig)


def prefill():
    d = J("prefill")["rows"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6))
    for b, col, lab in (("metal", HEX["GREEN"], "Metal GPU"), ("cpu", HEX["ORANGE"], "CPU container")):
        rows = sorted((r for r in d if r["backend"] == b), key=lambda r: r["prompt_tokens"])
        x = [r["prompt_tokens"] for r in rows]
        a1.plot(x, [r["ttft_p50_s"] for r in rows], "-o", color=col, label=lab)
        a2.plot(x, [r["prefill_tok_s"] for r in rows], "-o", color=col, label=lab)
        for r in rows:
            a1.annotate(f"{r['ttft_p50_s']:.2f}s", (r["prompt_tokens"], r["ttft_p50_s"]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, color=col)
    for a in (a1, a2):
        a.set_xscale("log", base=2)
        a.set_xlabel("prompt tokens (prompt cache off)")
        a.legend()
    a1.set_yscale("log")
    a1.set_ylabel("TTFT p50 (s)")
    a1.set_title("Prefill cost grows with prompt length", loc="left")
    a2.set_ylabel("prompt tokens/s")
    a2.set_title("Prefill throughput (Q4_K_M)", loc="left")
    fig.tight_layout()
    fig.savefig(IMG / "prefill.png")
    plt.close(fig)


def kv_and_quant():
    kv = J("kvcache")["rows"]
    pc = J("promptcache")
    ppl = J("perplexity")["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    a = axes[0]
    names = [r["kv_type"] for r in kv]
    a.bar(names, [r["kv_buffer_mib"] or 0 for r in kv], color=[HEX["ICE"], HEX["GREEN"]])
    for i, r in enumerate(kv):
        a.text(i, (r["kv_buffer_mib"] or 0) * 1.02, f"{r['kv_buffer_mib']:.0f} MiB (computed)\nserver RSS {r['server_rss_mib']:.0f} MiB\n{r['output_tok_s']:.1f} tok/s · TTFT {r['ttft_p50_s']:.2f}s",
               ha="center", va="bottom", fontsize=9)
    a.set_ylim(0, max(r["kv_buffer_mib"] or 0 for r in kv) * 1.6)
    a.set_title("KV cache type, 32k context (4 × ~5.6k-token prompts)", loc="left", fontsize=11)
    a.set_ylabel("KV buffer for 32k tokens (MiB)")
    a = axes[1]
    labels = ["cache off\n1st", "cache off\nrepeat", "cache on\n1st", "cache on\nrepeat"]
    vals = [pc["cache_prompt_false"]["first_ttft_s"], pc["cache_prompt_false"]["repeat_ttft_s"],
            pc["cache_prompt_true"]["first_ttft_s"], pc["cache_prompt_true"]["repeat_ttft_s"]]
    a.bar(labels, vals, color=[HEX["GREY"], HEX["GREY"], HEX["PURPLE"], HEX["GREEN"]])
    for i, v in enumerate(vals):
        a.text(i, v * 1.02, f"{v:.3f}s", ha="center", va="bottom", fontsize=9)
    a.set_title("Prompt-cache reuse, 4k-token prompt (TTFT)", loc="left", fontsize=11)
    a.set_ylabel("seconds")
    a = axes[2]
    a.bar([r["quant"].upper() for r in ppl], [r["ppl"] for r in ppl], yerr=[r["ppl_err"] for r in ppl],
          color=[HEX["GREEN"], HEX["ICE"], HEX["PURPLE"]][:len(ppl)])
    for i, r in enumerate(ppl):
        a.text(i, r["ppl"] * 0.5, f"PPL {r['ppl']:.3f}\n{r['file_mib']:.0f} MiB", ha="center", fontsize=9, color=HEX["BG"])
    a.set_title("Quantization quality: perplexity on lab docs", loc="left", fontsize=11)
    a.set_ylabel("perplexity (lower is better)")
    fig.tight_layout()
    fig.savefig(IMG / "kv_quant.png")
    plt.close(fig)


def mlx_vs_llamacpp():
    import numpy as np
    runs = [J("mlx_concurrency")["runs"], J("mlx_concurrency_repeat")["runs"]]
    col = {"llamacpp-q4_k_m": HEX["GREEN"], "llamacpp-q8_0": HEX["ICE"],
           "mlx-Qwen2.5-1.5B-Instruct-4bit": HEX["ORANGE"], "mlx-Qwen2.5-1.5B-Instruct-8bit": HEX["YELLOW"]}
    name = {"llamacpp-q4_k_m": "llama.cpp · Q4_K_M", "llamacpp-q8_0": "llama.cpp · Q8_0",
            "mlx-Qwen2.5-1.5B-Instruct-4bit": "MLX · 4-bit", "mlx-Qwen2.5-1.5B-Instruct-8bit": "MLX · 8-bit"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    for i, r in enumerate(runs[0]):
        x = [l["concurrency"] for l in r["levels"]]
        ys = np.array([[l["output_tok_s"] for l in rr[i]["levels"]] for rr in runs])
        a1.plot(x, ys.mean(0), "-o", color=col[r["label"]], label=name[r["label"]])
        a1.fill_between(x, ys.min(0), ys.max(0), color=col[r["label"]], alpha=0.2)
    a1.set_xscale("log", base=2)
    a1.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    a1.set_xlabel("concurrent requests (4 slots / decode-concurrency 4)")
    a1.set_ylabel("output tokens/s (mean of 2 runs, band = both)")
    a1.set_title("Server throughput, Qwen2.5-1.5B on the M4 Pro GPU", loc="left")
    a1.legend(fontsize=9)
    ctx = [r for r in J("ctx_curve")["rows"] if r["kv"] == "f16" and r["prompt_tokens"] == 512]
    s8 = [r for r in J("single_8bit")["rows"] if r["prompt_tokens"] == 512]
    ppl_l = {r["quant"]: r["ppl"] for r in J("perplexity")["rows"]}
    ppl_m = {r["weights"]: r["ppl"] for r in J("perplexity_mlx")["rows"]}
    bars = [("llama.cpp\nQ4_K_M", next(r for r in ctx if r["engine"] == "llama.cpp"), ppl_l["q4_k_m"] / ppl_l["q8_0"], HEX["GREEN"]),
            ("MLX\n4-bit", next(r for r in ctx if r["engine"] == "mlx"),
             ppl_m["Qwen2.5-1.5B-Instruct-4bit"] / ppl_m["Qwen2.5-1.5B-Instruct-8bit"], HEX["ORANGE"]),
            ("llama.cpp\nQ8_0", next(r for r in s8 if r["engine"] == "llama.cpp"), 1.0, HEX["ICE"]),
            ("MLX\n8-bit", next(r for r in s8 if r["engine"] == "mlx"), 1.0, HEX["YELLOW"])]
    x = np.arange(len(bars))
    b = a2.bar(x, [r["decode_tok_s"] for _, r, _, _ in bars], 0.6, color=[c for *_, c in bars])
    for xi, (_, r, rel, _) in zip(x, bars):
        a2.text(xi, r["decode_tok_s"] + 3, f"{r['decode_tok_s']:.0f} tok/s", ha="center", fontsize=10)
        if rel != 1.0:
            a2.text(xi, r["decode_tok_s"] / 2, f"PPL +{(rel - 1) * 100:.1f}%\nvs own 8-bit", ha="center", fontsize=9,
                    color=HEX["BG"])
    a2.set_xticks(x, [n for n, *_ in bars])
    a2.set_ylabel("single-stream decode tok/s (512-token prompt)")
    a2.set_title("One request: speed and what 4-bit costs", loc="left")
    fig.tight_layout()
    fig.savefig(IMG / "mlx_vs_llamacpp.png")
    plt.close(fig)


def ctx_curve():
    rows = J("ctx_curve")["rows"]
    cfg = [("llama.cpp", "f16", HEX["GREEN"], "llama.cpp · f16 KV"), ("llama.cpp", "q8_0", HEX["ICE"], "llama.cpp · q8_0 KV"),
           ("mlx", "f16", HEX["ORANGE"], "MLX · f16 KV"), ("mlx", "q8", HEX["YELLOW"], "MLX · 8-bit KV")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for e, kv, c, lab in cfg:
        rs = sorted((r for r in rows if r["engine"] == e and r["kv"] == kv), key=lambda r: r["prompt_tokens"])
        x = [r["prompt_tokens"] / 1000 for r in rs]
        axes[0].plot(x, [r["peak_footprint_mib"] / 1024 for r in rs], "-o", color=c, label=lab)
        axes[1].plot(x, [r["prefill_tok_s"] for r in rs], "-o", color=c, label=lab)
        axes[2].plot(x, [r["decode_tok_s"] for r in rs], "-o", color=c, label=lab)
    for a, t, y in zip(axes, ("Peak memory footprint of the process", "Prefill speed", "Decode speed after the prompt"),
                       ("GiB", "prompt tokens/s", "output tokens/s")):
        a.set_xlabel("prompt length (thousand tokens)")
        a.set_title(t, loc="left")
        a.set_ylabel(y)
        a.axvline(16.384, color=HEX["LINE"], ls=":")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "ctx_memory.png")
    plt.close(fig)


def spec_decode():
    import numpy as np
    sets = [("spec_llamacpp_7b", "llama.cpp · 7B Q4_K_M", HEX["GREEN"]), ("spec_mlx_7b", "MLX · 7B 4-bit", HEX["ORANGE"]),
            ("spec_llamacpp_1.5b", "llama.cpp · 1.5B Q4_K_M", HEX["GREY"])]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for a, group in zip(axes, ("structured", "open")):
        for f, lab, c in sets:
            d = J(f)
            xs = [0] + [cf["draft_tokens"] for cf in d["configs"][1:]]
            ys = [1.0] + [cf["summary"][group]["speedup"] for cf in d["configs"][1:]]
            a.plot(xs, ys, "-o", color=c, label=lab)
            for x_, y_ in zip(xs[1:], ys[1:]):
                a.annotate(f"{y_:.2f}x", (x_, y_), textcoords="offset points",
                           xytext=(0, 7 if "MLX" in lab else -14), ha="center", fontsize=8, color=c)
        a.axhline(1.0, color=HEX["LINE"], ls="--")
        a.set_xticks([0, 3, 8, 16], ["off", "3", "8", "16"])
        a.set_xlabel("draft tokens per step (0.5B draft model)")
        a.set_title(f"{group.capitalize()} prompts ({'code, SQL, YAML, enumeration' if group == 'structured' else 'explanations, a story'})",
                    loc="left")
    axes[0].set_ylabel("decode speed vs target alone")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "spec_decode.png")
    plt.close(fig)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    for f in (concurrency, prefill, kv_and_quant, mlx_vs_llamacpp, ctx_curve, spec_decode):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
    print("charts written")
