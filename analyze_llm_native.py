"""Summarize measured native LLM runs; never mix warmup/profile with performance."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run(run):
    summary, events, steps = run["summary"], run["events"], run["steps"]
    names = ["A"] if summary["case"] == "alone" else ["A", "B"]
    for name in names:
        tokens = [e for e in events if e["kind"] == "token" and e["request"] == name]
        assert [e["index"] for e in tokens] == list(range(1, 257)), "Missing or duplicate tokens"
        assert all(b["t_ns"] >= a["t_ns"] for a, b in zip(tokens, tokens[1:])), "Invalid clock order"
        expected = 2048 if name == "B" and summary["case"] == "long" else 128
        assert sum(s["input_tokens"] for s in steps if s["request"] == name and s["phase"] == "prefill") == expected
        assert sum(s["phase"] == "decode" and s["request"] == name for s in steps) == 255
    if len(names) == 2:
        assert summary["b_arrived_before_a_generated_last"], "Requests did not overlap"
        assert summary["trigger_delay_ms"] >= 0, "B sent before A token 16"
    a = [e for e in events if e["kind"] == "token" and e["request"] == "A"]
    first, last = max(zip(a, a[1:]), key=lambda pair: pair[1]["t_ns"]-pair[0]["t_ns"])
    overlap = {}
    for step in steps:
        ms = max(0, min(step["end_ns"], last["t_ns"])-max(step["start_ns"], first["t_ns"]))/1e6
        key = step["request"] + "." + step["phase"]
        overlap[key] = overlap.get(key, 0) + ms
    return {"run_id": summary["run_id"], "valid": True,
            "max_gap_ends_at_a_token": last["index"], "max_gap_ms": (last["t_ns"]-first["t_ns"])/1e6,
            "worker_wall_overlap_ms": overlap,
            "max_token_delivery_lag_ms": max((e["t_ns"]-e["generated_ns"])/1e6
                                              for e in events if e["kind"] == "token")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/llm_native"))
    args = parser.parse_args()
    runs = [read(p) for p in sorted(args.input.glob("r*.json")) if "summary" in read(p)]
    runs = [r for r in runs if not r["summary"]["profiled"] and not r["summary"]["warmup"]]
    if not runs:
        raise ValueError("No measured runs")
    validity = [validate_run(r) for r in runs]
    (args.input / "measurement_validation.json").write_text(json.dumps(validity, indent=2), encoding="utf-8")
    keys = [("alone", 2048), ("short", 2048), ("long", 256), ("long", 512), ("long", 2048)]
    metrics = ["a_post_max_gap_ms", "a_pre_median_gap_ms", "a_ttft_ms", "b_ttft_ms",
               "b_queue_ms", "trigger_delay_ms", "output_tokens_s", "elapsed_s", "peak_allocated_mib"]
    groups = {}
    aggregates = []
    for case, chunk in keys:
        group = [r for r in runs if r["summary"]["case"] == case and r["summary"]["chunk_tokens"] == chunk]
        if not group:
            continue
        groups[(case, chunk)] = group
        agg = {"case": case, "chunk_tokens": chunk, "n": len(group)}
        for metric in metrics:
            vals = [r["summary"][metric] for r in group if metric in r["summary"]]
            if vals:
                agg[metric] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
        aggregates.append(agg)
    (args.input / "aggregate.json").write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
    with (args.input / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["case", "chunk_tokens", "n"] + [f"{m}_median" for m in metrics]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for agg in aggregates:
            writer.writerow({**{k: agg[k] for k in fields[:3]},
                             **{f"{m}_median": agg[m]["median"] for m in metrics if m in agg}})

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 8), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[2.2, 1])
    ax = fig.add_subplot(grid[0, :])
    colors = {("alone", 2048): "#808080", ("long", 256): "#0072B2",
              ("long", 512): "#009E73", ("long", 2048): "#D55E00"}
    chosen = {}
    for key in [("alone", 2048), ("long", 256), ("long", 2048)]:
        group = groups[key]
        median = statistics.median(r["summary"]["a_post_max_gap_ms"] for r in group)
        r = min(group, key=lambda r: abs(r["summary"]["a_post_max_gap_ms"]-median))
        chosen[str(key)] = r["summary"]["run_id"]
        seq = [e for e in r["events"] if e["kind"] == "token" and e["request"] == "A"]
        gaps = [(b["t_ns"]-a["t_ns"])/1e6 for a, b in zip(seq, seq[1:])]
        label = "A alone" if key[0] == "alone" else f"Long B, prefill chunk {key[1]}"
        ax.plot(range(2, 257), gaps, color=colors[key], label=label, linewidth=1.2, alpha=.9)
    ax.axvline(16, color="black", linestyle="--", linewidth=1, label="B submitted just after A token 16")
    ax.set(xlabel="A output token index", ylabel="Client queue receive interval (ms)",
           title="Long-prompt interference on an ongoing answer (representative runs)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=.15)
    for j, (metric, title, unit) in enumerate([
        ("a_post_max_gap_ms", "A: largest gap after B submission", "ms"),
        ("b_ttft_ms", "B: time to first token", "ms"),
        ("output_tokens_s", "A+B output throughput", "tokens/s")]):
        bar = fig.add_subplot(grid[1, j])
        for i, chunk in enumerate((256, 512, 2048)):
            vals = [r["summary"][metric] for r in groups[("long", chunk)]]
            med = statistics.median(vals)
            bar.bar(i, med, width=.6, color=colors[("long", chunk)], alpha=.7)
            bar.scatter([i+(k-(len(vals)-1)/2)*.035 for k in range(len(vals))], vals,
                        color="#222222", s=14, zorder=3)
            bar.text(i, max(vals), f" {med:.1f}", ha="center", va="bottom", fontsize=9)
        bar.set(xticks=[0, 1, 2], xticklabels=["256", "512", "2048"], ylabel=unit, title=title,
                xlabel="B prefill chunk tokens (custom sequential worker)")
        bar.margins(y=.25)
    n = len(groups[("long", 256)])
    fig.suptitle(f"RTX 3060 | Qwen2.5-1.5B FP16 | SDPA math | {n} runs/condition\n"
                 "In-process token streaming, no GPU batching; NOT a vLLM scheduler benchmark", fontsize=13)
    fig.savefig(args.input / "interference.png", dpi=160)
    fig.savefig(args.input / "interference.svg")
    (args.input / "representative_runs.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
