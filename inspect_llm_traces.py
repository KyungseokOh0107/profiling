"""Validate CUDA activity and A/B annotations in separately captured traces."""

import argparse
import json
from collections import Counter
from pathlib import Path


def inspect(path):
    trace = json.loads(path.read_text(encoding="utf-8"))
    events = trace["traceEvents"]
    counts = Counter(e.get("cat", "") for e in events)
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    labels = [e for e in events if e.get("name", "").startswith(("A.prefill.", "A.decode.",
                                                                 "B.prefill.", "B.decode."))
              and e.get("ph") == "X" and e.get("cat") == "user_annotation"]
    labels.sort(key=lambda e: e["ts"])
    sections = []
    for event in labels:
        start, end = event["ts"], event["ts"] + event["dur"]
        # Same trace clock; sum only kernel intervals, never nested CPU aggregates.
        kernel_us = sum(max(0, min(end, k["ts"]+k["dur"])-max(start, k["ts"])) for k in kernels)
        sections.append({"name": event["name"], "trace_start_us": start,
                         "cpu_wall_ms": event["dur"]/1000,
                         "overlapping_kernel_sum_ms": kernel_us/1000})
    assert kernels, f"No GPU kernels in {path}"
    assert any(s["name"].startswith("A.decode.") for s in sections), "Missing A decode markers"
    assert any(s["name"].startswith("B.prefill.") for s in sections), "Missing B prefill markers"
    return {"trace": path.name, "categories": dict(counts), "kernel_count": len(kernels),
            "sections": sections,
            "limitations": "Profile runs include instrumentation/export overhead; not performance samples. "
                           "Kernel sums are activity inside synchronized forward markers, not bandwidth/occupancy measurements.",
            "runtime_warning": "External init callback must run in same thread as registerClient was emitted; "
                               "CUDA kernel events and A/B markers were independently verified."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/llm_native_sdpa"))
    args = parser.parse_args()
    results = [inspect(p) for p in sorted(args.input.glob("profile*.pt.trace.json"))]
    if len(results) != 2:
        raise ValueError(f"Expected exactly two profile traces, found {len(results)}")
    (args.input / "trace_validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        print(json.dumps({"trace": result["trace"], "kernels": result["kernel_count"],
                          "prefill_sections": [s for s in result["sections"] if s["name"].startswith("B.prefill.")]}))


if __name__ == "__main__":
    main()
