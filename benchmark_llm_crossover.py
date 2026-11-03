"""Randomized within-process backend/transport-controlled HTTP crossover."""
import argparse
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmark_llm_native import MODEL, REVISION, input_cases, save_json, validate_chunked_cache
from benchmark_llm_http import Engine, serve, pair, persist


def set_backend(name):
    torch.backends.cuda.enable_math_sdp(name == "math")
    torch.backends.cuda.enable_mem_efficient_sdp(name == "efficient")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--chat", action="store_true")
    p.add_argument("--deadline", default="2026-11-03T07:00:00+09:00")
    args = p.parse_args()
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Choose a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    cases = input_cases(tok)
    if args.chat:
        prefix = tok.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nDocument:\n", add_special_tokens=False)
        suffix = tok.encode("\nSummarize this document in three bullet points and explain the tradeoffs.<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
        body = tok.encode(("The inference service accepts short questions and long documents. "
            "Operators monitor request queues and GPU memory. Some customers need fast first responses, "
            "while others need smooth streamed answers. The team evaluates scheduling and reports latency distributions. "
            "They use fixed workloads, repeat measurements, and preserve failed tests.\n")*100, add_special_tokens=False)
        def prompt(n):
            result = prefix+body[:n-len(prefix)-len(suffix)]+suffix
            assert len(result) == n
            return result
        cases = {"A": prompt(128), "short": prompt(128), "long": prompt(2048)}
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                torch_dtype=torch.float16, attn_implementation="sdpa").eval().cuda()
    save_json(args.output / "inputs.json", cases)
    save_json(args.output / "manifest.json", {"model": MODEL, "revision": REVISION,
        "prompt_style": "chat" if args.chat else "completion", "repeats": args.repeats,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "created": datetime.now().astimezone().isoformat(), "dtype": "FP16", "order_seed": 442,
        "design": "math/efficient crossed with 256/512/2048; fixed backend efficient only for chat family"})
    (args.output / "implementation_snapshot.py").write_bytes(Path(__file__).read_bytes())
    engine = Engine(model, max_seqs=8)
    server = serve(engine)
    port = server.server_address[1]
    backends = ["efficient"] if args.chat else ["math", "efficient"]
    summaries = []
    try:
        for backend in backends:
            set_backend(backend)
            v = validate_chunked_cache(model, torch.tensor([cases["long"]], device="cuda"))
            save_json(args.output / f"cache_{backend}.json", v)
            for chunk in (256, 512, 2048):
                engine.configure(chunk)
                rid = f"warmup_{backend}_{chunk}"
                persist(args.output, rid, pair(port, cases, "long", chunk, rid), engine,
                    {"phase": "pairs", "case": "long", "chunk": chunk, "backend": backend, "warmup": True})
        configs = [(b,c) for b in backends for c in (256,512,2048)]
        rng = random.Random(442)
        for rep in range(args.repeats):
            order = configs.copy()
            rng.shuffle(order)
            for backend, chunk in order:
                if time.time()+90 > datetime.fromisoformat(args.deadline).timestamp():
                    return
                engine.configure(chunk)
                set_backend(backend)
                torch.cuda.reset_peak_memory_stats()
                rid = f"pair_r{rep}_{backend}_{chunk}"
                rs = pair(port, cases, "long", chunk, rid)
                summaries.append(persist(args.output, rid, rs, engine,
                    {"phase": "pairs", "case": "long", "chunk": chunk, "backend": backend,
                     "repeat": rep, "warmup": False}))
                save_json(args.output / "summaries.json", summaries)
    finally:
        engine.stop.set()
        engine.thread.join(timeout=30)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
