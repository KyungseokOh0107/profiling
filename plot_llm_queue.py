"""Reconstruct actual admission queues from server event timestamps."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trial", type=Path)
    args = p.parse_args()
    data = json.loads(args.trial.read_text())
    requests = data["requests"]
    servers = {s["id"]: s for s in data["server"]}
    origin = min(r["scheduled_ns"] for r in requests)
    def sec(t):
        return (t-origin)/1e9
    queue_events, active_events = [], []
    for server in servers.values():
        queue_events += [(sec(server["accepted_ns"]), 1), (sec(server["admitted_ns"]), -1)]
        active_events += [(sec(server["admitted_ns"]), 1), (sec(server["completed_ns"]), -1)]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    for events, label, color in [(queue_events,"Accepted, waiting for admission","#D55E00"),
                                  (active_events,"Active sequences","#0072B2")]:
        times, counts, current = [0], [0], 0
        for t, delta in sorted(events):
            current += delta
            times.append(t)
            counts.append(current)
        assert current == 0
        axes[0].step(times, counts, where="post", label=label, color=color)
    axes[0].set(ylabel="Requests", title="Actual queue and active population; completion drain included")
    for length, marker, label, color in [(128,"o","Short (128 input)","#0072B2"),
                                        (2048,"^","Long (2048 input)","#D55E00")]:
        group = [r for r in requests if r["prompt_tokens"] == length and r["status"] == "ok"]
        axes[1].scatter([sec(r["scheduled_ns"]) for r in group],
                        [r["scheduled_ttft_ms"]/1000 for r in group], marker=marker, color=color, label=label)
        axes[2].scatter([sec(r["scheduled_ns"]) for r in group],
                        [r["max_gap_ms"] for r in group], marker=marker, color=color, label=label)
    axes[1].axhline(2, ls="--", lw=1, color="#0072B2", label="Synthetic short TTFT target")
    axes[1].axhline(4, ls="--", lw=1, color="#D55E00", label="Synthetic long TTFT target")
    axes[2].axhline(350, ls="--", lw=1, color="black", label="Synthetic max-gap target")
    axes[1].set(ylabel="Scheduled TTFT (seconds)", title="Each point is one completed request")
    axes[2].set(ylabel="Request maximum receive gap (ms)")
    for ax in axes:
        ax.set_xlabel("Seconds from first scheduled arrival")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle(f"{data['summary']['run_id']} / native sequential HTTP worker / NOT vLLM")
    for suffix in ("png", "svg"):
        fig.savefig(args.trial.parent/f"{args.trial.stem}_queue.{suffix}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
