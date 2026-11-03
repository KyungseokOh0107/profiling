"""Validate HTTP/SSE measurements and aggregate at trial and request levels."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values)*fraction)-1)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    args = p.parse_args()
    files = sorted(list(args.folder.glob("pair_*.json"))+list(args.folder.glob("load_*.json")))
    groups, validation = defaultdict(list), []
    request_rows, gap_attribution = [], []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data["summary"]
        if summary.get("warmup"):
            continue
        assert summary["requests"] == len(data["requests"])
        assert summary["completed"] == sum(r["status"] == "ok" for r in data["requests"])
        assert summary["errors"] == summary["requests"]-summary["completed"]
        good = sum(r["status"] == "ok" and
                   r["scheduled_ttft_ms"] <= (2000 if r["prompt_tokens"] == 128 else 4000) and
                   r["max_gap_ms"] <= 350 for r in data["requests"])
        assert good == summary["good_requests"], "SLO count mismatch"
        first = min(r["scheduled_ns"] if r["scheduled_ns"] is not None else r["send_ns"]
                    for r in data["requests"])
        elapsed = (max(r["done_ns"] for r in data["requests"])-first)/1e9
        assert abs(elapsed-summary["elapsed_s"]) < 1e-8, "Drain denominator mismatch"
        assert abs(good/elapsed-summary["goodput_req_s"]) < 1e-8
        group = (summary.get("backend"), summary["phase"], summary.get("case", "mixed"), summary.get("rate"), summary["chunk"], summary.get("client_placement"), summary.get("max_seqs"))
        groups[group].append(data)
        server = {r["id"]: r for r in data["server"]}
        if summary["phase"] in ("pilot", "load"):
            traces = json.loads((args.folder/f"arrivals_r{summary['repeat']}.json").read_text())
            expected = traces[str(summary["rate"])]
            assert len(expected) == len(data["requests"])
            base = data["requests"][0]["scheduled_ns"]
            for spec, req in zip(expected, data["requests"]):
                assert req["scheduled_ns"]-base == int(spec["offset_s"]*1e9), "Arrival replay changed"
                assert req["prompt_tokens"] == (128 if spec["kind"] == "short" else 2048)
            assert sum(r["prompt_tokens"] == 2048 for r in data["requests"]) * 5 == len(expected)
        for r in data["requests"]:
            if r["status"] == "ok":
                assert [t["index"] for t in r["tokens"]] == list(range(1,257))
                assert len(r["tokens"]) == 256
                digest = hashlib.sha256(json.dumps([t["token_id"] for t in r["tokens"]]).encode()).hexdigest()
                assert digest == r["output_sha256"], "Output token hash mismatch"
                gaps = [(b["received_ns"]-a["received_ns"])/1e6
                        for a, b in zip(r["tokens"], r["tokens"][1:])]
                assert gaps == r["gaps_ms"] and min(gaps) >= 0
                assert max(gaps) == r["max_gap_ms"]
                assert abs(r["scheduled_ttft_ms"]-r["ttft_ms"]-r["send_lag_ms"]) < 1e-5
                s = server[r["request_id"]]
                assert sum(x["input_tokens"] for x in s["steps"] if x["phase"] == "prefill") == r["prompt_tokens"]
                assert sum(x["phase"] == "decode" for x in s["steps"]) == 255
                assert s["accepted_ns"] <= s["admitted_ns"] <= s["steps"][0]["start_ns"]
                assert all(t["generated_ns"] <= t["received_ns"] for t in r["tokens"])
                assert [e["event"] for e in r["aux_events"]] == ["accepted", "done"]
                prefill = [step for step in s["steps"] if step["phase"] == "prefill"]
                compute_ms = sum(step["end_ns"]-step["start_ns"] for step in prefill)/1e6
                components = {
                    "http_ingress_ms": (s["accepted_ns"]-r["send_ns"])/1e6,
                    "admission_wait_ms": (s["admitted_ns"]-s["accepted_ns"])/1e6,
                    "admission_to_first_forward_ms": (prefill[0]["start_ns"]-s["admitted_ns"])/1e6,
                    "prefill_forward_wall_ms": compute_ms,
                    "prefill_interleave_wait_ms": (prefill[-1]["end_ns"]-prefill[0]["start_ns"])/1e6-compute_ms,
                    "first_token_delivery_ms": (r["tokens"][0]["received_ns"]-prefill[-1]["end_ns"])/1e6,
                }
                assert abs(sum(components.values())-r["ttft_ms"]) < 1e-5, "TTFT decomposition mismatch"
                request_rows.append({"run_id": summary["run_id"], "request_id": r["request_id"],
                    "backend": summary.get("backend"), "client_placement": summary.get("client_placement"), "max_seqs": summary.get("max_seqs"), "chunk": summary["chunk"], "rate": summary.get("rate"), "prompt_tokens": r["prompt_tokens"],
                    "ttft_ms": r["ttft_ms"], "scheduled_ttft_ms": r["scheduled_ttft_ms"],
                    "max_gap_ms": r["max_gap_ms"], "gap_p95_ms": percentile(r["gaps_ms"], .95),
                    "send_lag_ms": r["send_lag_ms"], "delivery_max_ms": r["delivery_max_ms"],
                    **components,
                    "output_sha256": r["output_sha256"]})
        if summary["phase"] == "pairs" and len(data["requests"]) == 2:
            assert summary.get("overlap"), "B did not overlap A generation"
            a, b = data["requests"]
            if a["status"] == b["status"] == "ok":
                assert b["scheduled_ns"] == a["tokens"][15]["received_ns"], "B trigger was not A token 16"
                eligible = [(gap, i) for i, gap in enumerate(a["gaps_ms"])
                            if a["tokens"][i+1]["received_ns"] >= b["send_ns"]]
                gap, i = max(eligible)
                begin, end = a["tokens"][i]["received_ns"], a["tokens"][i+1]["received_ns"]
                overlap = defaultdict(float)
                for r in (a, b):
                    role = "A" if r is a else "B"
                    for step in server[r["request_id"]]["steps"]:
                        clipped = max(0, min(end, step["end_ns"])-max(begin, step["start_ns"]))
                        overlap[f"{role}_{step['phase']}_wall_overlap_ms"] += clipped/1e6
                gap_attribution.append({"run_id": summary["run_id"], "backend": summary.get("backend"),
                    "client_placement": summary.get("client_placement"), "chunk": summary["chunk"],
                    "a_token_index": i+2, "receive_gap_ms": gap, **overlap,
                    "outside_forward_wall_ms": gap-sum(overlap.values()),
                    "interpretation": "Synchronized CPU forward wall intervals, NOT GPU kernel-time attribution"})
        validation.append({"run_id": summary["run_id"], "successful": summary["completed"], "errors": summary["errors"]})
    aggregates = []
    metrics = ["output_tokens_s", "goodput_req_s", "a_post_max_gap_ms", "b_ttft_ms",
               "max_admission_wait_ms", "max_pending_at_capacity", "max_client_send_lag_ms", "peak_allocated_mib"]
    for key, trials in groups.items():
        backend, phase, case, rate, chunk, placement, max_seqs = key
        result = {"backend": backend, "phase": phase, "case": case, "rate": rate, "chunk": chunk, "client_placement": placement, "max_seqs": max_seqs,
                  "trials": len(trials), "requests": sum(t["summary"]["requests"] for t in trials),
                  "errors": sum(t["summary"]["errors"] for t in trials)}
        total_seconds = sum(t["summary"]["elapsed_s"] for t in trials)
        result["total_elapsed_s"] = total_seconds
        result["good_requests"] = sum(t["summary"]["good_requests"] for t in trials)
        result["pooled_goodput_req_s"] = sum(t["summary"]["good_requests"] for t in trials)/total_seconds
        result["pooled_output_tokens_s"] = sum(len(r["tokens"]) for t in trials for r in t["requests"])/total_seconds
        for metric in metrics:
            values = [t["summary"][metric] for t in trials if metric in t["summary"]]
            if values:
                result[metric] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        for length in (128, 2048):
            reqs = [r for t in trials for r in t["requests"] if r["prompt_tokens"] == length and r["status"] == "ok"]
            if reqs:
                result[str(length)] = {"n": len(reqs),
                    "scheduled_ttft_median_ms": statistics.median(r["scheduled_ttft_ms"] for r in reqs),
                    "scheduled_ttft_p95_ms": percentile([r["scheduled_ttft_ms"] for r in reqs], .95),
                    "ttft_median_ms": statistics.median(r["ttft_ms"] for r in reqs),
                    "ttft_p95_ms": percentile([r["ttft_ms"] for r in reqs], .95),
                    "max_gap_median_ms": statistics.median(r["max_gap_ms"] for r in reqs),
                    "max_gap_p95_ms": percentile([r["max_gap_ms"] for r in reqs], .95)}
        aggregates.append(result)
    for name, value in [("aggregate.json", aggregates), ("validation.json", validation),
                        ("gap_attribution.json", gap_attribution)]:
        (args.folder/name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    if request_rows:
        with (args.folder/"requests.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(request_rows[0]))
            writer.writeheader()
            writer.writerows(request_rows)
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
