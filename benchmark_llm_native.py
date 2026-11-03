"""Windows-only preliminary LLM interference experiment, NOT vLLM/HTTP serving.

One GPU worker round-robins independent A/B KV caches. Each turn advances one
decode token or one prefill chunk. A client thread receives explicit token IDs
through an in-process Queue and submits B immediately after receiving A#16.
Chunk size is NOT vLLM's max_num_batched_tokens; requests are never GPU-batched.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import queue
import random
import statistics
import threading
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def input_cases(tokenizer):
    text = (
        "An AI inference service shares one GPU across requests. A long document "
        "requires prompt processing, while another user is waiting for the next "
        "output token. Explain how request scheduling affects latency and throughput.\n"
    )
    ids = tokenizer.encode(text * 120, add_special_tokens=False)
    assert len(ids) >= 2048
    # Fixed, exact token arrays; no chat template or implicit BOS insertion.
    return {"A": ids[:128], "short": ids[:128], "long": ids[:2048]}


def validate_chunked_cache(model, tokens):
    """Check cached chunked prefill against a full causal forward, before timing."""
    results = []
    with torch.inference_mode():
        full = model(input_ids=tokens, use_cache=False, logits_to_keep=1).logits.float()
        for chunk in (256, 512):
            cache = DynamicCache()
            for start in range(0, tokens.shape[1], chunk):
                out = model(input_ids=tokens[:, start:start+chunk], past_key_values=cache,
                            use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
            logits = out.logits.float()
            finite = bool(torch.isfinite(logits).all().item())
            close = torch.allclose(logits, full, atol=0.15, rtol=0.02)
            results.append({"chunk_tokens": chunk, "finite": finite,
                            "max_abs_logit_difference": (logits-full).abs().max().item(),
                            "allclose_atol_0_15_rtol_0_02": close,
                            "first_token_equal": logits.argmax().item() == full.argmax().item(),
                            "kv_length": cache.get_seq_length()})
            if not finite or not close or cache.get_seq_length() != tokens.shape[1]:
                raise ValueError(f"Chunked causal-cache validation failed: {results[-1]}")
    return results


def summarize(events, steps, case, chunk, repeat, run_id, profiled):
    sends = {e["request"]: e["t_ns"] for e in events if e["kind"] == "send"}
    tokens = {r: [e for e in events if e["kind"] == "token" and e["request"] == r]
              for r in sends}
    for r, seq in tokens.items():
        if len(seq) != 256 or [e["index"] for e in seq] != list(range(1, 257)):
            raise ValueError(f"Invalid output sequence for {r}: {len(seq)}")
    a = tokens["A"]
    gaps = [(a[i]["t_ns"] - a[i-1]["t_ns"]) / 1e6 for i in range(1, len(a))]
    trigger = sends.get("B", a[15]["t_ns"])
    post = [gap for gap, token in zip(gaps, a[1:]) if token["t_ns"] >= trigger]
    admissions = {s["request"]: s["t_ns"] for s in events if s["kind"] == "admit"}
    done = [e["t_ns"] for e in events if e["kind"] == "complete"]
    elapsed = (max(done) - sends["A"]) / 1e9
    result = {
        "run_id": run_id, "case": case, "chunk_tokens": chunk,
        "repeat": repeat, "profiled": profiled, "status": "ok",
        "a_max_gap_ms": max(gaps), "a_post_max_gap_ms": max(post),
        "a_pre_median_gap_ms": statistics.median(gaps[:15]),
        "a_ttft_ms": (a[0]["t_ns"] - sends["A"]) / 1e6,
        "elapsed_s": elapsed, "output_tokens_s": sum(map(len, tokens.values())) / elapsed,
        "completion_tokens": {r: len(seq) for r, seq in tokens.items()},
        "a_token_ids_sha256": hashlib.sha256(json.dumps([e["token_id"] for e in a]).encode()).hexdigest(),
    }
    if "B" in tokens:
        b = tokens["B"]
        result.update({
            "b_ttft_ms": (b[0]["t_ns"] - sends["B"]) / 1e6,
            "b_queue_ms": (admissions["B"] - sends["B"]) / 1e6,
            "trigger_delay_ms": (sends["B"] - a[15]["t_ns"]) / 1e6,
            "b_arrived_before_a_generated_last": sends["B"] < a[-1]["generated_ns"],
            "b_token_ids_sha256": hashlib.sha256(json.dumps([e["token_id"] for e in b]).encode()).hexdigest(),
            "b_prefill_wall_ms": sum((s["end_ns"] - s["start_ns"]) / 1e6
                                     for s in steps if s["request"] == "B" and s["phase"] == "prefill"),
        })
    return result


def run_pair(model, tensors, case, chunk, repeat, run_id, output, profile=False, warmup=False):
    incoming, outgoing = queue.Queue(), queue.Queue()
    events, steps = [], []
    expected = 1 if case == "alone" else 2
    ready = threading.Event()
    cancel = threading.Event()
    profiler_info = {}

    def export_trace(recording):
        recording.export_chrome_trace(str(output / f"{run_id}.pt.trace.json"))
        profiler_info["cuda_event_count"] = sum(
            str(e.device_type).lower().endswith("cuda") for e in recording.events())
        profiler_info["capture"] = "12 forward calls wait, 2 warmup, 24 active, one capture"
        profiler_info["top_cuda_ops"] = sorted(
            [{"name": e.key, "self_device_time_us": e.self_device_time_total}
             for e in recording.key_averages() if e.self_device_time_total > 0],
            key=lambda e: e["self_device_time_us"], reverse=True)[:12]

    def worker():
        try:
            context = (torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=12, warmup=2, active=24, repeat=1),
                on_trace_ready=export_trace,
                record_shapes=False, profile_memory=False, with_stack=False)
                if profile else nullcontext())
            with torch.inference_mode(), context as recording:
                states = {}
                admitted = 0
                ready.set()
                while not cancel.is_set():
                    while True:
                        try:
                            request = incoming.get_nowait()
                        except queue.Empty:
                            break
                        name = request["request"]
                        states[name] = {"offset": 0, "count": 0, "cache": DynamicCache(),
                                        "input": request["input"], "last": None}
                        admitted += 1
                        outgoing.put({"kind": "admit", "request": name, "t_ns": time.perf_counter_ns()})
                    if not states:
                        if admitted == expected:
                            break
                        time.sleep(0.0001)
                        continue
                    for name in list(states):
                        state = states[name]
                        length = state["input"].shape[1]
                        prefill = state["offset"] < length
                        start_offset = state["offset"]
                        if prefill:
                            end = min(length, start_offset + chunk)
                            ids = state["input"][:, start_offset:end]
                            state["offset"] = end
                        else:
                            ids = state["last"]
                        phase = "prefill" if prefill else "decode"
                        label = f"{name}.{phase}.offset={start_offset}.output={state['count']}"
                        start = time.perf_counter_ns()
                        marker = torch.profiler.record_function(label) if profile else nullcontext()
                        with marker:
                            result = model(input_ids=ids, past_key_values=state["cache"],
                                           use_cache=True, logits_to_keep=1)
                            if not bool(torch.isfinite(result.logits).all().item()):
                                raise ValueError(f"Nonfinite logits at {label}")
                            state["cache"] = result.past_key_values
                            emit = not prefill or state["offset"] == length
                            if emit:
                                state["last"] = result.logits[:, -1:].argmax(dim=-1)
                                token_id = state["last"].item()  # GPU completion before publishing.
                            else:
                                torch.cuda.synchronize()
                        end_ns = time.perf_counter_ns()
                        steps.append({"request": name, "phase": phase, "start_ns": start,
                                      "end_ns": end_ns, "input_tokens": ids.shape[1],
                                      "prefill_offset": start_offset, "outputs_before": state["count"]})
                        if emit:
                            state["count"] += 1
                            outgoing.put({"kind": "token", "request": name,
                                          "index": state["count"], "token_id": token_id,
                                          "generated_ns": end_ns})
                            if state["count"] == 256:  # Deliberately ignore EOS for fixed work.
                                outgoing.put({"kind": "complete", "request": name,
                                              "t_ns": time.perf_counter_ns()})
                                del states[name]
                        del result
                        if profile:
                            recording.step()
                torch.cuda.synchronize()
        except Exception:
            outgoing.put({"kind": "error", "message": traceback.format_exc()})
        finally:
            ready.set()
            outgoing.put({"kind": "worker_done"})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    if not ready.wait(30):
        raise TimeoutError("Worker did not initialize")

    def send(name, tensor):
        stamp = time.perf_counter_ns()
        events.append({"kind": "send", "request": name, "t_ns": stamp,
                       "prompt_tokens": tensor.shape[1], "target_output_tokens": 256})
        incoming.put({"request": name, "input": tensor})

    send("A", tensors["A"])
    try:
        while True:
            event = outgoing.get(timeout=120)
            received_ns = time.perf_counter_ns()
            if event["kind"] == "error":
                raise RuntimeError(event["message"])
            if event["kind"] == "worker_done":
                break
            if event["kind"] == "token":
                event["t_ns"] = received_ns
            events.append(event)
            if event["kind"] == "token" and event["request"] == "A" and event["index"] == 16:
                events.append({"kind": "trigger", "request": "A", "t_ns": received_ns})
                if case != "alone":
                    send("B", tensors[case])
    finally:
        cancel.set()
        thread.join(timeout=120)
    if thread.is_alive():
        raise TimeoutError("GPU worker failed to stop")
    result = summarize(events, steps, case, chunk, repeat, run_id, profile)
    result["warmup"] = warmup
    result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    if profile:
        result["profiler"] = profiler_info
    save_json(output / f"{run_id}.json", {"summary": result, "events": events, "steps": steps})
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/llm_native"))
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(MODEL, revision=REVISION,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    print(f"Model snapshot ready: {snapshot}", flush=True)
    if args.download_only:
        return
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    # Windows eager FP16 produced nonfinite logits in the rejected initial probe.
    # Pin SDPA math (FP32 intermediates) while keeping weights and KV in FP16.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, torch_dtype=torch.float16, attn_implementation="sdpa",
        local_files_only=True).eval().to("cuda")
    cases = input_cases(tokenizer)
    tensors = {name: torch.tensor([ids], device="cuda") for name, ids in cases.items()}
    save_json(args.output / "inputs.json", cases)
    save_json(args.output / "cache_validation.json", validate_chunked_cache(model, tensors["long"]))
    metadata = {
        "experiment": "native Transformers sequential worker, in-process queue; NOT vLLM/HTTP",
        "model": MODEL, "revision": REVISION, "dtype": str(next(model.parameters()).dtype),
        "model_config": model.config.to_dict(), "python": platform.python_version(),
        "platform": platform.platform(), "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "attention_backend": model.config._attn_implementation,
        "sdpa_kernel": "math only (FP32 intermediates), FP16 weights and KV",
        "finite_logits_check": "all logits checked at every model forward, included in timing",
        "cuda_graph": False, "prefix_cache": False, "seed": 42, "ignore_eos": True,
        "output_tokens": 256, "max_logical_context": 2304, "max_sequences": 2,
        "input_storage": "preloaded CUDA token tensors", "cuda_sync": "every model forward",
        "scheduler": "A then B, one decode token or prefill chunk per request per round; no batching",
        "torch_cpu_threads": 4, "sampling": "greedy argmax", "parameter_count": sum(p.numel() for p in model.parameters()),
        "input_sha256": {name: hashlib.sha256(json.dumps(ids).encode()).hexdigest() for name, ids in cases.items()},
        "repeats": args.repeats,
    }
    save_json(args.output / "metadata.json", metadata)
    if args.smoke:
        run_pair(model, tensors, "long", 2048, -1, "smoke_long_2048", args.output, warmup=True)
        return
    summaries = []
    # Every shape is warmed before measurements. Warmup files are explicitly labelled.
    for chunk in (256, 512, 2048):
        run_pair(model, tensors, "long", chunk, -1, f"warmup_long_{chunk}", args.output, warmup=True)
    # Alone and short B need only one chunk value: input length is below all budgets.
    conditions = [("alone", 2048), ("short", 2048), ("long", 256), ("long", 512), ("long", 2048)]
    rng = random.Random(42)
    order = []
    for repeat in range(args.repeats):
        block = conditions.copy()
        rng.shuffle(block)
        for case, chunk in block:
            run_id = f"r{repeat}_{case}_{chunk}"
            order.append(run_id)
            torch.cuda.reset_peak_memory_stats()
            summaries.append(run_pair(model, tensors, case, chunk, repeat, run_id, args.output))
            save_json(args.output / "summaries.json", summaries)
    save_json(args.output / "order.json", order)
    if args.profile:
        long_runs = [r for r in summaries if r["case"] == "long"]
        medians = {c: statistics.median(r["a_post_max_gap_ms"] for r in long_runs if r["chunk_tokens"] == c)
                   for c in (256, 512, 2048)}
        best, worst = min(medians, key=medians.get), max(medians, key=medians.get)
        for chunk in dict.fromkeys((best, worst)):
            run_pair(model, tensors, "long", chunk, -1, f"profile_long_{chunk}", args.output, profile=True)


if __name__ == "__main__":
    main()
