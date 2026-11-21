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


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    for f in (concurrency, prefill, kv_and_quant):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
    print("charts written")
