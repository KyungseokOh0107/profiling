"""Collect completed cohorts into compact, mechanically computed result tables."""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def distribution(values):
    return {"n": len(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, nargs="?", default=Path("results/overnight"))
    args = p.parse_args()
    root = args.root
    result = {"created": datetime.now().astimezone().isoformat(), "cohorts": {}}
    names = ["http_pairs_math", "http_pairs_efficient", "http_load_pilot", "http_load_pilot_extended",
             "http_mixed_load", "backend_crossover", "chat_pairs", "client_process_crossover", "admission_crossover"]
    for name in names:
        folder = root/name
        if not (folder/"summaries.json").exists():
            continue
        subprocess.run([sys.executable, "analyze_llm_http.py", str(folder)], check=True,
                       stdout=subprocess.DEVNULL)
        summaries = read(folder/"summaries.json")
        rows = list(folder.glob("pair_*.json"))+list(folder.glob("load_*.json"))
        measured = [read(path) for path in rows]
        measured = [r for r in measured if not r["summary"].get("warmup")]
        assert len(summaries) == len(measured), f"Incomplete summary index in {name}"
        hashes = defaultdict(set)
        for trial in measured:
            for request in trial["requests"]:
                if request["status"] == "ok":
                    hashes[request["prompt_tokens"]].add(request["output_sha256"])
        result["cohorts"][name] = {"trials": len(summaries),
            "requests": sum(s["requests"] for s in summaries),
            "errors": sum(s["errors"] for s in summaries),
            "output_tokens": sum(len(r["tokens"]) for t in measured for r in t["requests"]),
            "output_hashes_by_prompt_length": {k: sorted(v) for k,v in hashes.items()},
            "aggregate": read(folder/"aggregate.json")}
        breakdown = defaultdict(lambda: {"requests": 0, "good": 0,
            "ttft_only_fail": 0, "gap_only_fail": 0, "both_fail": 0, "errors": 0})
        for trial in measured:
            s = trial["summary"]
            key = f"{s.get('backend')}/{s.get('rate')}/{s['chunk']}/{s.get('client_placement')}/{s.get('max_seqs')}"
            entry = breakdown[key]
            for request in trial["requests"]:
                entry["requests"] += 1
                if request["status"] != "ok":
                    entry["errors"] += 1
                    continue
                ttft_bad = request["scheduled_ttft_ms"] > (2000 if request["prompt_tokens"] == 128 else 4000)
                gap_bad = request["max_gap_ms"] > 350
                field = ("both_fail" if ttft_bad and gap_bad else "ttft_only_fail" if ttft_bad
                         else "gap_only_fail" if gap_bad else "good")
                entry[field] += 1
        result["cohorts"][name]["slo_breakdown"] = dict(breakdown)
        with (folder/"requests.csv").open(encoding="utf-8", newline="") as stream:
            request_metrics = list(csv.DictReader(stream))
        components = defaultdict(list)
        for request in request_metrics:
            key = "/".join(request[k] for k in ("backend", "rate", "chunk", "client_placement", "max_seqs", "prompt_tokens"))
            components[key].append(request)
        result["cohorts"][name]["ttft_components"] = {
            key: {metric: distribution([float(r[metric]) for r in group]) for metric in
                ("http_ingress_ms", "admission_wait_ms", "admission_to_first_forward_ms",
                 "prefill_forward_wall_ms", "prefill_interleave_wait_ms", "first_token_delivery_ms", "send_lag_ms")}
            for key, group in components.items()}
    for name in ("static_batch_fp16", "static_batch_bf16"):
        path = root/name/"summaries.json"
        if not path.exists():
            continue
        rows = read(path)
        assert len(rows) == len(list(path.parent.glob("r*_b*_t*_l*.json"))), "Static raw/index count mismatch"
        groups = defaultdict(list)
        for r in rows:
            assert r["kv_tensor_bytes"] == r["kv_theoretical_bytes"]
            assert r["kv_stored_tokens"] == r["input_tokens"]+255
            assert r["output_tokens_per_request"] == 256 and len(r["step_ms"]) == 256
            groups[(r["batch"],r["cpu_threads"],r["input_tokens"])].append(r)
        aggregate = []
        for (batch, threads, length), group in sorted(groups.items()):
            entry = {"batch": batch, "cpu_threads": threads, "input_tokens": length,
                     "trials": len(group), "kv_bytes": group[0]["kv_tensor_bytes"],
                     "sequence_hashes": sorted({r["sequence_hash"] for r in group})}
            for metric in ("output_tokens_s", "ttft_ms", "decode_step_median_ms", "peak_allocated_mib"):
                entry[metric] = distribution([r[metric] for r in group])
            aggregate.append(entry)
        result["cohorts"][name] = {"trials": len(rows), "aggregate": aggregate,
            "output_tokens": sum(r["batch"]*256 for r in rows)}
    for name in ("backend_probe", "backend_probe_v2"):
        path = root/name/"probe.json"
        if path.exists():
            result[name] = read(path)
    cross = root/"backend_crossover"/"summaries.json"
    if cross.exists():
        rows = read(cross)
        paired = []
        for chunk in (256,512,2048):
            lookup = {(r["repeat"],r["backend"]):r for r in rows if r["chunk"] == chunk}
            reps = sorted({r["repeat"] for r in rows})
            entry = {"chunk": chunk}
            for metric in ("a_post_max_gap_ms", "b_ttft_ms", "output_tokens_s"):
                ratios = [100*(lookup[(rep,"efficient")][metric]/lookup[(rep,"math")][metric]-1)
                          for rep in reps if (rep,"efficient") in lookup and (rep,"math") in lookup]
                if ratios:
                    entry[metric+"_paired_percent_change"] = distribution(ratios)
            paired.append(entry)
        result["paired_backend_changes"] = paired
    (root/"RESULTS_DATA.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name:{k:v for k,v in cohort.items() if k not in ("aggregate","output_hashes_by_prompt_length", "slo_breakdown", "ttft_components")}
                      for name,cohort in result["cohorts"].items()}, indent=2))


if __name__ == "__main__":
    main()
