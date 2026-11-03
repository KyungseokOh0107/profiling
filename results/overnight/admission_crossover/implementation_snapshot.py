"""Admission-cap crossover under a fixed finite overload burst; not vLLM."""
import argparse
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmark_llm_native import MODEL, REVISION, input_cases, save_json
from benchmark_llm_http import Engine, serve, pair, persist, replay, arrivals
from benchmark_llm_crossover import set_backend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--deadline", default="2026-11-03T07:00:00+09:00")
    args = p.parse_args()
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    if time.time()+600 > deadline:
        print("Insufficient time for one complete paired cap block; not starting.", flush=True)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    set_backend("efficient")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    cases = input_cases(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        local_files_only=True, torch_dtype=torch.float16, attn_implementation="sdpa").eval().cuda()
    save_json(args.output / "manifest.json", {"model": MODEL, "revision": REVISION,
        "dtype": "FP16", "backend": "efficient", "chunk": 512, "caps": [2, 8],
        "rate": .24, "requests_per_trial": 20, "planned_repeats": args.repeats,
        "created": datetime.now().astimezone().isoformat(),
        "design": "same arrival trace within each cap crossover block; finite overload, not steady state",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "synthetic_slo": {"short_scheduled_ttft_ms": 2000, "long_scheduled_ttft_ms": 4000, "max_gap_ms": 350}})
    (args.output / "implementation_snapshot.py").write_bytes(Path(__file__).read_bytes())
    save_json(args.output / "inputs.json", cases)
    engine = Engine(model, chunk=512, max_seqs=8)
    server = serve(engine)
    port = server.server_address[1]
    summaries = []
    try:
        persist(args.output, "warmup", pair(port, cases, "long", 512, "warmup"), engine,
                {"phase": "pairs", "case": "long", "chunk": 512, "backend": "efficient", "warmup": True})
        for rep in range(args.repeats):
            if time.time()+500 > deadline:
                print("Deadline: not starting another paired cap block.", flush=True)
                break
            trace = arrivals(20, .24, 1402+rep)
            save_json(args.output / f"arrivals_r{rep}.json", {.24: trace})
            order = [2, 8] if rep % 2 == 0 else [8, 2]
            for cap in order:
                engine.configure(512, cap)
                torch.cuda.reset_peak_memory_stats()
                rid = f"load_r{rep}_cap{cap}"
                results = replay(port, cases, trace, rid)
                summaries.append(persist(args.output, rid, results, engine,
                    {"phase": "load", "rate": .24, "chunk": 512, "backend": "efficient",
                     "max_seqs": cap, "repeat": rep, "warmup": False}))
                save_json(args.output / "summaries.json", summaries)
    finally:
        engine.stop.set()
        engine.thread.join(timeout=30)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
