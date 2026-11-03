"""Randomized same-server crossover of in-process versus separate-process HTTP clients.

The control API exists only on loopback and outside timed request intervals.
No CUDA initialization happens in the separate client process.
"""
import argparse
import hashlib
import http.client
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from benchmark_llm_http import Engine, serve, pair, persist
from benchmark_llm_native import MODEL, REVISION, input_cases, save_json


def control(port, path, payload):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
    try:
        conn.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(data)
        return data
    finally:
        conn.close()


def server_main(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmark_llm_crossover import set_backend

    deadline = datetime.fromisoformat(args.deadline).timestamp()
    def watchdog():
        time.sleep(max(0, deadline-time.time()))
        os._exit(124)  # This owned process must not survive a killed controller.
    threading.Thread(target=watchdog, daemon=True).start()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    set_backend("efficient")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    cases = input_cases(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        local_files_only=True, torch_dtype=torch.float16, attn_implementation="sdpa").eval().cuda()
    engine = Engine(model, max_seqs=8)
    server = serve(engine)
    port = server.server_address[1]
    stop = threading.Event()

    class ControlHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                p = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/configure":
                    engine.configure(p["chunk"], 8)
                    torch.cuda.reset_peak_memory_stats()
                    result = {"ok": True}
                elif self.path in ("/local", "/persist"):
                    settings, rid = p["settings"], p["rid"]
                    results = (pair(port, cases, "long", settings["chunk"], rid)
                               if self.path == "/local" else p["results"])
                    engine.idle.wait(120)
                    result = persist(args.output, rid, results, engine, settings)
                elif self.path == "/stop":
                    stop.set()
                    result = {"ok": True}
                else:
                    raise ValueError("Unknown control endpoint")
                status = 200
            except Exception:
                result, status = {"error": traceback.format_exc()}, 500
            body = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    api = ThreadingHTTPServer(("127.0.0.1", 0), ControlHandler)
    api.daemon_threads = True
    threading.Thread(target=api.serve_forever, daemon=True).start()
    save_json(args.output / "ready.json", {"port": port, "control_port": api.server_address[1],
        "cases": cases, "pid": os.getpid()})
    try:
        while not stop.wait(.2):
            if engine.health_error:
                raise RuntimeError(engine.health_error)
    finally:
        engine.stop.set()
        engine.thread.join(30)
        server.shutdown()
        server.server_close()
        api.shutdown()
        api.server_close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--server", action="store_true")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--deadline", default="2026-11-03T07:00:00+09:00")
    args = p.parse_args()
    if args.server:
        return server_main(args)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    save_json(args.output / "manifest.json", {"model": MODEL, "revision": REVISION,
        "dtype": "FP16", "backend": "efficient", "max_seqs": 8,
        "repeats": args.repeats, "order_seed": 731,
        "created": datetime.now().astimezone().isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "design": "same persistent GPU server; randomized client placement x chunk",
        "clock": "Windows QueryPerformanceCounter via perf_counter_ns, system-wide",
        "scheduler": "sequential independent caches, no GPU batching, not vLLM"})
    (args.output / "implementation_snapshot.py").write_bytes(Path(__file__).read_bytes())
    summaries, info = [], None
    with (args.output / "server.log").open("w", encoding="utf-8") as stream:
        proc = subprocess.Popen([sys.executable, "-u", __file__, "--server", "--output",
            str(args.output), "--deadline", args.deadline], stdout=stream, stderr=subprocess.STDOUT)
        try:
            start = time.monotonic()
            while info is None:
                if proc.poll() is not None or time.monotonic()-start > 180:
                    raise RuntimeError("GPU server failed to start; inspect server.log")
                try:
                    info = json.loads((args.output / "ready.json").read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    time.sleep(.2)
            save_json(args.output / "inputs.json", info["cases"])
            configs = [(placement, chunk) for placement in ("same_process", "separate_process")
                       for chunk in (256, 512, 2048)]
            rng = random.Random(731)
            for rep in range(-1, args.repeats):
                order = configs.copy()
                rng.shuffle(order)
                for placement, chunk in order:
                    if time.time()+90 > datetime.fromisoformat(args.deadline).timestamp():
                        return
                    rid = f"pair_r{rep}_{placement}_{chunk}"
                    settings = {"phase": "pairs", "case": "long", "chunk": chunk,
                        "backend": "efficient", "client_placement": placement,
                        "repeat": rep, "warmup": rep == -1}
                    control(info["control_port"], "/configure", {"chunk": chunk})
                    payload = {"rid": rid, "settings": settings}
                    if placement == "separate_process":
                        payload["results"] = pair(info["port"], info["cases"], "long", chunk, rid)
                    result = control(info["control_port"],
                                     "/persist" if placement == "separate_process" else "/local", payload)
                    print(json.dumps(result), flush=True)
                    if rep != -1:
                        summaries.append(result)
                        save_json(args.output / "summaries.json", summaries)
        finally:
            if info and proc.poll() is None:
                try:
                    control(info["control_port"], "/stop", {})
                except Exception:
                    pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                from run_experiment_queue import stop_owned_process
                stop_owned_process(proc)


if __name__ == "__main__":
    main()
