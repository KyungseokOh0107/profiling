"""Loopback HTTP/SSE experiment with a real Qwen GPU worker, NOT vLLM.

The worker advances independent requests sequentially; no GPU batching.
Client and server share a monotonic clock in one process but communicate via TCP.
"""
import argparse
import hashlib
import importlib.metadata
import http.client
import json
import math
import queue
import random
import socket
import statistics
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from benchmark_llm_native import MODEL, REVISION, input_cases, save_json, validate_chunked_cache


class Engine:
    def __init__(self, model, chunk=512, max_seqs=8):
        self.model, self.chunk, self.max_seqs = model, chunk, max_seqs
        self.incoming = queue.Queue()
        self.stop = threading.Event()
        self.idle = threading.Event()
        self.idle.set()
        self.records = {}
        self.health_error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, payload):
        if self.health_error or self.stop.is_set():
            raise RuntimeError(self.health_error or "Engine is stopping")
        rid = payload["request_id"]
        if rid in self.records:
            raise ValueError("Duplicate request ID")
        if len(payload["prompt_ids"]) not in (128, 512, 1024, 2048):
            raise ValueError("Unsupported prompt length")
        if payload.get("max_tokens", 256) != 256:
            raise ValueError("Output length must remain 256")
        state = {"id": rid, "accepted_ns": time.perf_counter_ns(), "events": queue.Queue(),
                 "prompt_ids": payload["prompt_ids"], "steps": [], "count": 0, "offset": 0}
        self.records[rid] = state
        self.idle.clear()
        self.incoming.put(state)
        return state

    def run(self):
        pending, active = deque(), []
        try:
            with torch.inference_mode():
                while not self.stop.is_set():
                    while True:
                        try:
                            pending.append(self.incoming.get_nowait())
                        except queue.Empty:
                            break
                    while pending and len(active) < self.max_seqs:
                        s = pending.popleft()
                        s["admitted_ns"] = time.perf_counter_ns()
                        s["input"] = torch.tensor([s["prompt_ids"]], device="cuda")
                        s["cache"] = DynamicCache()
                        s["last"] = None
                        active.append(s)
                    if not active:
                        if not pending and self.incoming.empty():
                            self.idle.set()
                        time.sleep(.0002)
                        continue
                    for s in list(active):
                        if self.stop.is_set():
                            break
                        prefill = s["offset"] < len(s["prompt_ids"])
                        offset = s["offset"]
                        if prefill:
                            end = min(len(s["prompt_ids"]), offset+self.chunk)
                            ids = s["input"][:, offset:end]
                            s["offset"] = end
                        else:
                            ids = s["last"]
                        start = time.perf_counter_ns()
                        out = self.model(input_ids=ids, past_key_values=s["cache"],
                                         use_cache=True, logits_to_keep=1)
                        if not bool(out.logits.isfinite().all().item()):
                            raise ValueError(f"Nonfinite logits in {s['id']}")
                        s["cache"] = out.past_key_values
                        emit = not prefill or s["offset"] == len(s["prompt_ids"])
                        if emit:
                            s["last"] = out.logits[:, -1:].argmax(dim=-1)
                            token = s["last"].item()
                        else:
                            torch.cuda.synchronize()
                        end_ns = time.perf_counter_ns()
                        s["steps"].append({"phase": "prefill" if prefill else "decode",
                                           "start_ns": start, "end_ns": end_ns,
                                           "input_tokens": ids.shape[1], "offset": offset,
                                           "active": len(active), "pending": len(pending)+self.incoming.qsize()})
                        if emit:
                            s["count"] += 1
                            s["events"].put(("token", {"index": s["count"], "token_id": token,
                                                       "generated_ns": end_ns}))
                        del out
                        if s["count"] == 256:
                            s["completed_ns"] = time.perf_counter_ns()
                            s["events"].put(("done", {"output_tokens": 256, "completed_ns": s["completed_ns"]}))
                            active.remove(s)
                            del s["input"], s["cache"], s["last"]
        except Exception:
            self.health_error = traceback.format_exc()
            for s in self.records.values():
                if "completed_ns" not in s:
                    s["events"].put(("error", {"error": self.health_error}))
        finally:
            self.idle.set()

    def configure(self, chunk, max_seqs=8):
        if not self.idle.wait(120):
            raise TimeoutError("Engine failed to drain")
        if self.health_error:
            raise RuntimeError(self.health_error)
        self.chunk, self.max_seqs = chunk, max_seqs

    def evidence(self, request_ids):
        return [{k: v for k, v in self.records[rid].items()
                 if k not in ("events", "input", "cache", "last", "prompt_ids")}
                for rid in request_ids if rid in self.records]


def serve(engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404)
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state = engine.submit(payload)
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                def emit(kind, data):
                    self.wfile.write((f"event: {kind}\ndata: {json.dumps(data)}\n\n").encode())
                    self.wfile.flush()

                emit("accepted", {"request_id": state["id"], "accepted_ns": state["accepted_ns"]})
                while True:
                    kind, data = state["events"].get(timeout=600)
                    emit(kind, data)
                    if kind in ("done", "error"):
                        break
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                self.close_connection = True
            finally:
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def request(port, rid, ids, scheduled_ns=None, token_callback=None):
    result = {"request_id": rid, "prompt_tokens": len(ids), "scheduled_ns": scheduled_ns,
              "send_ns": time.perf_counter_ns(), "tokens": [], "aux_events": []}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    try:
        conn.request("POST", "/generate", json.dumps({"request_id": rid, "prompt_ids": ids, "max_tokens": 256}),
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        result["http_status"] = response.status
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        event, data_lines = None, []
        while True:
            line = response.readline()
            received = time.perf_counter_ns()
            if not line:
                break
            line = line.decode().rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif not line and data_lines:
                data = json.loads("\n".join(data_lines))
                data["received_ns"] = received
                if event == "token":
                    result["tokens"].append(data)
                    if token_callback:
                        token_callback(data)
                else:
                    result["aux_events"].append({"event": event, **data})
                if event == "error":
                    raise RuntimeError(data["error"])
                if event == "done":
                    result["done_ns"] = received
                    break
                event, data_lines = None, []
        seq = result["tokens"]
        if [x["index"] for x in seq] != list(range(1, 257)) or "done_ns" not in result:
            raise ValueError("Missing/duplicate tokens or missing done event")
        result["ttft_ms"] = (seq[0]["received_ns"]-result["send_ns"])/1e6
        result["scheduled_ttft_ms"] = (seq[0]["received_ns"]-(scheduled_ns or result["send_ns"]))/1e6
        result["send_lag_ms"] = (result["send_ns"]-(scheduled_ns or result["send_ns"]))/1e6
        result["gaps_ms"] = [(b["received_ns"]-a["received_ns"])/1e6 for a, b in zip(seq, seq[1:])]
        result["max_gap_ms"] = max(result["gaps_ms"])
        result["delivery_max_ms"] = max((x["received_ns"]-x["generated_ns"])/1e6 for x in seq)
        result["output_sha256"] = hashlib.sha256(json.dumps([x["token_id"] for x in seq]).encode()).hexdigest()
        result["status"] = "ok"
    except Exception as exc:
        result["status"], result["error"] = "error", repr(exc)
        result["done_ns"] = time.perf_counter_ns()
    finally:
        conn.close()
    return result


def pair(port, cases, name, chunk, rid):
    b = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        def trigger(token):
            if token["index"] == 16 and name != "alone":
                b.append(pool.submit(request, port, rid+"_B", cases[name], token["received_ns"]))
        a = pool.submit(request, port, rid+"_A", cases["A"], None, trigger).result()
        results = [a] + ([b[0].result()] if b else [])
    return results


def arrivals(n, rate, seed):
    rng = random.Random(seed)
    kinds = ["short"]*(n*4//5) + ["long"]*(n//5)
    rng.shuffle(kinds)
    gaps = [rng.expovariate(rate) for _ in range(n-1)]
    # Keep stochastic relative spacing, but exact finite-window average rate.
    scale = ((n-1)/rate)/sum(gaps) if gaps else 1
    ts = [0.0]
    for g in gaps:
        ts.append(ts[-1]+g*scale)
    return [{"offset_s": t, "kind": k} for t, k in zip(ts, kinds)]


def replay(port, cases, trace, prefix):
    start = time.perf_counter_ns()+100_000_000
    with ThreadPoolExecutor(max_workers=len(trace)) as pool:
        futures = []
        def completed(future):
            result = future.result()
            print(json.dumps({"progress": "request_completed", "request_id": result["request_id"],
                              "status": result["status"], "ttft_ms": result.get("ttft_ms"),
                              "max_gap_ms": result.get("max_gap_ms")}), flush=True)
        for i, entry in enumerate(trace):
            target = start+int(entry["offset_s"]*1e9)
            time.sleep(max(0, (target-time.perf_counter_ns())/1e9))
            future = pool.submit(request, port, f"{prefix}_{i:03d}", cases[entry["kind"]], target)
            future.add_done_callback(completed)
            futures.append(future)
        return [f.result() for f in futures]


def persist(out, rid, results, engine, settings):
    ids = [r["request_id"] for r in results]
    evidence = engine.evidence(ids)
    first = min(r["scheduled_ns"] if r["scheduled_ns"] is not None else r["send_ns"] for r in results)
    end = max(r["done_ns"] for r in results)
    seconds = (end-first)/1e9
    success = [r for r in results if r["status"] == "ok"]
    # Predeclared synthetic targets, not a production SLA.
    good = [r for r in success if r["scheduled_ttft_ms"] <= (2000 if r["prompt_tokens"] == 128 else 4000)
            and r["max_gap_ms"] <= 350]
    summary = {"run_id": rid, **settings, "requests": len(results), "completed": len(success),
               "errors": len(results)-len(success), "elapsed_s": seconds,
               "output_tokens_s": sum(len(r["tokens"]) for r in results)/seconds,
               "good_requests": len(good), "goodput_req_s": len(good)/seconds,
               "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
               "max_pending": max((s["pending"] for r in evidence for s in r["steps"]), default=0),
               "max_pending_at_capacity": max((s["pending"] for r in evidence for s in r["steps"]
                                                if s["active"] == engine.max_seqs), default=0),
               "max_admission_wait_ms": max(((r["admitted_ns"]-r["accepted_ns"])/1e6 for r in evidence), default=0),
               "max_client_send_lag_ms": max((r.get("send_lag_ms", 0) for r in results), default=0),
               "max_active": max((s["active"] for r in evidence for s in r["steps"]), default=0)}
    for kind, length in (("short", 128), ("long", 2048)):
        group = [r for r in success if r["prompt_tokens"] == length]
        if group:
            summary[kind] = {"n": len(group), "ttft_median_ms": statistics.median(r["ttft_ms"] for r in group),
                             "gap_max_median_ms": statistics.median(r["max_gap_ms"] for r in group),
                             "ttft_max_ms": max(r["ttft_ms"] for r in group)}
    if settings["phase"] == "pairs":
        a = results[0]
        if a["status"] == "ok":
            trigger = results[1]["send_ns"] if len(results) == 2 else a["tokens"][15]["received_ns"]
            summary["a_post_max_gap_ms"] = max(gap for gap, token in zip(a["gaps_ms"], a["tokens"][1:])
                                                if token["received_ns"] >= trigger)
        if len(results) == 2 and results[1]["status"] == "ok":
            summary["b_ttft_ms"] = results[1]["ttft_ms"]
            summary["trigger_delay_ms"] = (results[1]["send_ns"]-results[1]["scheduled_ns"])/1e6
            summary["overlap"] = results[1]["send_ns"] < a["tokens"][-1]["generated_ns"]
    save_json(out / f"{rid}.json", {"summary": summary, "requests": results, "server": evidence})
    for request_id in ids:
        engine.records.pop(request_id, None)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["math", "efficient"], default="math")
    parser.add_argument("--phase", choices=["pairs", "pilot", "load"], default="pairs")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.07, 0.14])
    parser.add_argument("--chunks", type=int, nargs="+", default=[256, 512, 2048])
    parser.add_argument("--max-seqs", type=int, default=8)
    parser.add_argument("--deadline", default="2026-11-03T07:00:00+09:00")
    args = parser.parse_args()
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Choose a new output directory; do not overwrite evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.enable_math_sdp(args.backend == "math")
    torch.backends.cuda.enable_mem_efficient_sdp(args.backend == "efficient")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    cases = input_cases(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
              torch_dtype=torch.float16, attn_implementation="sdpa").eval().cuda()
    validation = validate_chunked_cache(model, torch.tensor([cases["long"]], device="cuda"))
    manifest = {**vars(args), "output": str(args.output), "model": MODEL, "revision": REVISION,
                "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                "transformers": importlib.metadata.version("transformers"),
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "dtype": "FP16", "created": datetime.now().astimezone().isoformat(),
                "cache_validation": validation, "transport": "HTTP/1.1 SSE over loopback",
                "scheduler": "sequential round-robin, no GPU batching, no vLLM",
                "output_tokens": 256, "prefix_cache": False, "cuda_graph": False,
                "synthetic_slo": {"short_ttft_ms": 2000, "long_ttft_ms": 4000, "max_gap_ms": 350},
                "inputs_sha256": {k: hashlib.sha256(json.dumps(v).encode()).hexdigest() for k,v in cases.items()}}
    save_json(args.output / "manifest.json", manifest)
    save_json(args.output / "inputs.json", cases)
    (args.output / "implementation_snapshot.py").write_bytes(Path(__file__).read_bytes())
    engine = Engine(model, max_seqs=args.max_seqs)
    server = serve(engine)
    port = server.server_address[1]
    summaries = []
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    try:
        for chunk in args.chunks:
            engine.configure(chunk, args.max_seqs)
            rs = pair(port, cases, "long", chunk, f"warmup_{chunk}")
            persist(args.output, f"warmup_{chunk}", rs, engine,
                    {"phase": "pairs", "chunk": chunk, "warmup": True, "backend": args.backend})
        if args.phase == "pairs":
            configs = [("alone", 2048), ("short", 2048)]+[("long", c) for c in args.chunks]
            rng = random.Random(471)
            for rep in range(args.repeats):
                rng.shuffle(configs)
                for case, chunk in configs:
                    if time.time()+90 > deadline:
                        return
                    engine.configure(chunk, args.max_seqs)
                    torch.cuda.reset_peak_memory_stats()
                    rid = f"pair_r{rep}_{case}_{chunk}"
                    rs = pair(port, cases, case, chunk, rid)
                    summaries.append(persist(args.output, rid, rs, engine,
                        {"phase": "pairs", "case": case, "chunk": chunk, "repeat": rep,
                         "warmup": False, "backend": args.backend}))
                    save_json(args.output / "summaries.json", summaries)
        else:
            configs = [(rate, chunk) for rate in args.rates for chunk in args.chunks]
            for rep in range(args.repeats):
                traces = {rate: arrivals(args.count, rate, 902+rep) for rate in args.rates}
                save_json(args.output / f"arrivals_r{rep}.json", traces)
                order = configs if rep % 2 == 0 else list(reversed(configs))
                for rate, chunk in order:
                    if time.time()+max(args.count/rate, args.count*11)+90 > deadline:
                        print("Deadline: not starting another load run", flush=True)
                        return
                    engine.configure(chunk, args.max_seqs)
                    torch.cuda.reset_peak_memory_stats()
                    rid = f"load_r{rep}_rate{rate}_chunk{chunk}"
                    print(f"START {rid} {datetime.now().isoformat()}", flush=True)
                    rs = replay(port, cases, traces[rate], rid)
                    summaries.append(persist(args.output, rid, rs, engine,
                        {"phase": args.phase, "rate": rate, "chunk": chunk, "repeat": rep,
                         "warmup": False, "backend": args.backend}))
                    save_json(args.output / "summaries.json", summaries)
    finally:
        engine.stop.set()
        engine.thread.join(timeout=30)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
