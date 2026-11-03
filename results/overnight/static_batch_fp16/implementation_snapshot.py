"""Static batching / CPU dispatch / context-length study, separate from serving."""
import argparse
import hashlib
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from benchmark_llm_native import MODEL, REVISION, input_cases, save_json


def trial(model, ids, batch, threads, length, output_tokens=256):
    torch.set_num_threads(threads)
    x = torch.tensor([ids[:length]]*batch, device="cuda")
    cache = DynamicCache()
    emitted, stamps, durations = [], [], []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for i in range(output_tokens):
            before = time.perf_counter_ns()
            out = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
            if not out.logits.isfinite().all().item():
                raise ValueError("Nonfinite logits")
            cache = out.past_key_values
            x = out.logits[:, -1:].argmax(dim=-1)
            emitted.append(x[:, 0].tolist())
            now = time.perf_counter_ns()
            stamps.append(now)
            durations.append((now-before)/1e6)
        elapsed = (stamps[-1]-start)/1e9
        dtype = str(cache.key_cache[0].dtype)
        cache_bytes = sum(t.numel()*t.element_size() for t in cache.key_cache+cache.value_cache)
        stored_tokens = int(cache.get_seq_length())
        expected_bytes = (2*model.config.num_hidden_layers*model.config.num_key_value_heads*
                          (model.config.hidden_size//model.config.num_attention_heads)*2*stored_tokens*batch)
        assert cache_bytes == expected_bytes, "Measured KV bytes disagree with architecture formula"
    sequences = list(zip(*emitted))
    assert all(seq == sequences[0] for seq in sequences), "Identical batch rows diverged"
    return {"batch": batch, "cpu_threads": threads, "input_tokens": length,
            "output_tokens_per_request": output_tokens, "ttft_ms": (stamps[0]-start)/1e6,
            "decode_step_median_ms": statistics.median(durations[1:]),
            "decode_step_max_ms": max(durations[1:]), "elapsed_s": elapsed,
            "output_tokens_s": batch*output_tokens/elapsed, "kv_dtype": dtype,
            "kv_stored_tokens": stored_tokens, "kv_tensor_bytes": cache_bytes,
            "kv_theoretical_bytes": expected_bytes,
            "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
            "sequence_hash": hashlib.sha256(json.dumps(sequences[0]).encode()).hexdigest(),
            "step_ms": durations}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--deadline", default="2026-11-03T07:00:00+09:00")
    args = p.parse_args()
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Choose new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    ids = input_cases(tok)["long"]
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
            torch_dtype=torch.float16 if args.dtype == "fp16" else torch.bfloat16,
            attn_implementation="sdpa").eval().cuda()
    configs = [(1, 4, 128), (2, 4, 128), (4, 4, 128), (8, 4, 128),
               (1, 1, 128), (1, 2, 128), (1, 8, 128), (1, 4, 512), (1, 4, 1024), (1, 4, 2048)]
    manifest = {"model": MODEL, "revision": REVISION, "dtype": args.dtype,
                "backend": "efficient SDPA only", "configs_batch_threads_input": configs,
                "repeats": args.repeats, "output_tokens": 256, "transport": "direct model calls",
                "created": datetime.now().astimezone().isoformat(), "ignore_eos": True,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(args.output / "manifest.json", manifest)
    (args.output / "implementation_snapshot.py").write_bytes(Path(__file__).read_bytes())
    for batch, threads, length in configs:
        r = trial(model, ids, batch, threads, length, output_tokens=16)
        save_json(args.output / f"warmup_b{batch}_t{threads}_l{length}.json", r)
    results = []
    rng = random.Random(714)
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    for rep in range(args.repeats):
        order = configs.copy()
        rng.shuffle(order)
        for batch, threads, length in order:
            if time.time()+90 >= deadline:
                return
            r = trial(model, ids, batch, threads, length)
            r.update({"repeat": rep, "dtype": args.dtype, "warmup": False})
            results.append(r)
            save_json(args.output / f"r{rep}_b{batch}_t{threads}_l{length}.json", r)
            save_json(args.output / "summaries.json", results)
            print(json.dumps({k:v for k,v in r.items() if k != "step_ms"}), flush=True)


if __name__ == "__main__":
    main()
