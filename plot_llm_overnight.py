"""Render only completed measured cohorts; no synthetic/fallback performance data."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, default=Path("results/overnight"), nargs="?")
    args = parser.parse_args()
    root = args.root
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {256: "#0072B2", 512: "#009E73", 2048: "#D55E00"}
    cross = root / "backend_crossover" / "summaries.json"
    if cross.exists():
        rows = read(cross)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        for ax, metric, title, unit in zip(axes,
            ["a_post_max_gap_ms", "b_ttft_ms", "output_tokens_s"],
            ["A: largest gap after B arrives", "B: time to first token", "A+B output throughput"],
            ["ms", "ms", "tokens/s"]):
            for backend, style in [("math", "--o"), ("efficient", "-s")]:
                vals, lower, upper = [], [], []
                for chunk in (256,512,2048):
                    seq = [r[metric] for r in rows if r["backend"] == backend and r["chunk"] == chunk]
                    med = statistics.median(seq)
                    vals.append(med)
                    lower.append(med-min(seq))
                    upper.append(max(seq)-med)
                ax.errorbar([0,1,2], vals, yerr=[lower, upper], fmt=style, capsize=4,
                            label=f"{backend}: median / min-max")
            ax.set(xticks=[0,1,2], xticklabels=["256","512","2048"], xlabel="Prefill chunk tokens", ylabel=unit, title=title)
            ax.legend()
            ax.grid(alpha=.2)
        fig.suptitle("Qwen 1.5B FP16 / RTX 3060 / real HTTP SSE\nRandomized backend crossover; native sequential worker, NOT vLLM")
        fig.savefig(root/"backend_crossover.png", dpi=160)
        fig.savefig(root/"backend_crossover.svg")
        plt.close(fig)
    load = root / "http_mixed_load" / "summaries.json"
    if load.exists():
        rows = read(load)
        rates = sorted({r["rate"] for r in rows})
        agg = read(root/"http_mixed_load"/"aggregate.json")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for chunk in (256,512,2048):
            for ax, metric, label in [(axes[0,0], "goodput_req_s", "Goodput (requests/s)"),
                                      (axes[0,1], "output_tokens_s", "Output tokens/s")]:
                vals = [next(r["pooled_"+metric] for r in agg if r["chunk"] == chunk and r["rate"] == rate) for rate in rates]
                ax.plot(rates, vals, "o-", color=colors[chunk], label=f"chunk {chunk}")
                ax.set(xlabel="Offered requests/s", ylabel=label)
            for ax, metric, label in [(axes[1,0], "scheduled_ttft_p95_ms", "Short-request scheduled TTFT p95 (ms)"),
                                      (axes[1,1], "max_gap_p95_ms", "Short-request max-gap p95 (ms)")]:
                vals = [next(r["128"][metric] for r in agg if r["chunk"] == chunk and r["rate"] == rate) for rate in rates]
                ax.plot(rates, vals, "o-", color=colors[chunk], label=f"chunk {chunk}")
                ax.set(xlabel="Offered requests/s", ylabel=label)
                ax.set_yscale("log")
        axes[1,0].axhline(2000, color="black", ls="--", lw=1, label="Synthetic TTFT target")
        axes[1,1].axhline(350, color="black", ls="--", lw=1, label="Synthetic gap target")
        for ax in axes.flat:
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle("Fixed finite 80% short / 20% long HTTP arrival traces; throughput denominator includes drain\nFP16 efficient SDPA; 256 output tokens/request; cap 8; no GPU batching; exploratory p95, 2 runs/condition")
        fig.savefig(root/"mixed_load.png", dpi=160)
        fig.savefig(root/"mixed_load.svg")
        plt.close(fig)
    batch = root / "static_batch_fp16" / "summaries.json"
    if batch.exists():
        rows = read(batch)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        x = [1,2,4,8]
        for ax, metric, title in [(axes[0], "output_tokens_s", "Static batch aggregate throughput"),
                                  (axes[1], "decode_step_median_ms", "Per-request decode step latency")]:
            y = [statistics.median(r[metric] for r in rows if r["batch"] == b and r["cpu_threads"] == 4 and r["input_tokens"] == 128) for b in x]
            ax.plot(x, y, "o-")
            ax.set(xlabel="Batch size", ylabel="tokens/s" if metric == "output_tokens_s" else "ms", title=title, xticks=x)
        y = [statistics.median(r["output_tokens_s"] for r in rows if r["batch"] == 1 and r["cpu_threads"] == t and r["input_tokens"] == 128) for t in x]
        axes[2].plot(x, y, "o-", color="#D55E00")
        axes[2].set(xlabel="CPU threads (batch=1)", ylabel="tokens/s", title="CPU dispatch sensitivity", xticks=x)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle("Qwen 1.5B FP16 / RTX 3060 / efficient SDPA\nStatic direct-call batching, separate from HTTP serving")
        fig.savefig(root/"static_batch.png", dpi=160)
        fig.savefig(root/"static_batch.svg")
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
        lengths = [128, 512, 1024, 2048]
        for ax, metric, title, unit in [
            (axes[0], "decode_step_median_ms", "Context and decode latency", "ms"),
            (axes[1], "kv_tensor_bytes", "Measured KV tensors = formula", "MiB")]:
            values = [statistics.median(r[metric] for r in rows
                      if r["batch"] == 1 and r["cpu_threads"] == 4 and r["input_tokens"] == length)
                      for length in lengths]
            if metric == "kv_tensor_bytes":
                values = [v/2**20 for v in values]
            ax.plot(lengths, values, "o-")
            ax.set(xlabel="Input tokens (+255 cached output tokens)", ylabel=unit,
                   title=title, xticks=lengths)
        values = [statistics.median(r["peak_allocated_mib"] for r in rows
                  if r["batch"] == b and r["cpu_threads"] == 4 and r["input_tokens"] == 128) for b in x]
        axes[2].plot(x, values, "o-", color="#D55E00")
        axes[2].set(xlabel="Batch size (128 input tokens)", ylabel="MiB",
                    title="Peak PyTorch allocated memory", xticks=x)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle("Static direct inference / FP16 / RTX 3060 / efficient SDPA\n"
                     "KV = 28,672 bytes per cached token per sequence; allocated memory is NOT total device VRAM")
        fig.savefig(root/"context_memory.png", dpi=160)
        fig.savefig(root/"context_memory.svg")
        plt.close(fig)


if __name__ == "__main__":
    main()
