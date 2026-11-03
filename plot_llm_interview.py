"""One-page evidence: received-token gaps, synchronized prefill intervals, and costs."""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--backend", default="efficient")
    args = p.parse_args()
    rows = [json.loads(path.read_text()) for path in args.folder.glob("pair_*.json")]
    rows = [r for r in rows if not r["summary"].get("warmup")
            and r["summary"].get("backend") == args.backend
            and r["summary"].get("case") == "long"
            and r["summary"].get("client_placement") is None]
    chunks = [256, 512, 2048]
    colors = {256: "#0072B2", 512: "#009E73", 2048: "#D55E00"}
    fig = plt.figure(figsize=(14, 10), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, .8])
    gap_ax = fig.add_subplot(grid[0, :])
    time_ax = fig.add_subplot(grid[1, :])
    timeline_end = 1200
    for lane, chunk in enumerate(chunks):
        group = [r for r in rows if r["summary"]["chunk"] == chunk]
        median = statistics.median(r["summary"]["a_post_max_gap_ms"] for r in group)
        chosen = min(group, key=lambda r: abs(r["summary"]["a_post_max_gap_ms"]-median))
        a, b = chosen["requests"]
        origin = b["send_ns"]
        tokens = a["tokens"]
        gap_ax.plot(range(2, 257), a["gaps_ms"], color=colors[chunk], lw=1.3,
                    label=f"chunk {chunk}: representative run {chosen['summary']['run_id']}")
        server_a, server_b = chosen["server"]
        bars = [((s["start_ns"]-origin)/1e6, (s["end_ns"]-s["start_ns"])/1e6)
                for s in server_b["steps"] if s["phase"] == "prefill"]
        time_ax.broken_barh(bars, (lane-.25, .45), facecolors=colors[chunk], alpha=.8)
        first_b = (b["tokens"][0]["received_ns"]-origin)/1e6
        stamps = [(t["received_ns"]-origin)/1e6 for t in tokens
                  if -100 <= (t["received_ns"]-origin)/1e6 <= max(1200, first_b*1.1)]
        time_ax.plot(stamps, [lane+.28]*len(stamps), "|", color="black", markersize=7)
        timeline_end = max(timeline_end, first_b*1.1)
        time_ax.plot(first_b, lane, "*", ms=12, color="black")
    gap_ax.axvline(16, color="black", ls="--", lw=1, label="B scheduled after receipt of A token 16")
    gap_ax.set(xlabel="A output token index (gap ends at this token)", ylabel="Client receive gap (ms)",
               title="Streaming interruption: all 255 receive intervals, no token pooling")
    gap_ax.legend(fontsize=8)
    gap_ax.grid(alpha=.2)
    time_ax.axvline(0, color="black", ls="--", lw=1)
    time_ax.set(xlim=(-100, timeline_end), yticks=range(3), yticklabels=[f"chunk {c}" for c in chunks],
                xlabel="Milliseconds from actual B send (each representative run aligned independently)",
                title="Server B-prefill wall intervals (colored) / A token receipt (ticks) / B first token (star)")
    time_ax.grid(axis="x", alpha=.2)
    for col, metric, title, unit in [(0,"a_post_max_gap_ms","A largest gap after B arrives","ms"),
                                    (1,"b_ttft_ms","B time to first token","ms"),
                                    (2,"output_tokens_s","A+B output throughput","tokens/s")]:
        ax = fig.add_subplot(grid[2, col])
        for x, chunk in enumerate(chunks):
            seq = [r["summary"][metric] for r in rows if r["summary"]["chunk"] == chunk]
            median = statistics.median(seq)
            ax.scatter([x]*len(seq), seq, color=colors[chunk], alpha=.5, s=20)
            ax.plot([x-.18,x+.18], [median,median], color="black", lw=2)
            ax.annotate(f"{median:.1f}", (x,median), xytext=(8,5), textcoords="offset points", fontsize=9)
        ax.set(xticks=range(3), xticklabels=chunks, xlabel="Prefill chunk tokens", ylabel=unit, title=title)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"RTX 3060 / Qwen2.5-1.5B FP16 / {args.backend} SDPA / HTTP SSE\n"
                 "Native sequential worker: independent caches, no GPU batching, NOT vLLM; dots are separate runs", fontsize=13)
    for suffix in ("png", "svg"):
        fig.savefig(args.folder/f"interview.{suffix}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
