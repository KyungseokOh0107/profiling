"""Run GPU jobs serially, preserve logs, and enforce the authorized deadline."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def stop_owned_process(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        # A Windows venv executable can be a launcher with a child interpreter.
        # Stop only the process tree rooted at the child this controller owns.
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        proc.terminate()
    proc.wait(timeout=30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs", type=Path)
    p.add_argument("--deadline", default="2026-11-03T07:00:00+09:00")
    args = p.parse_args()
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    folder = args.jobs.parent
    state_path = folder / (args.jobs.stem+"_status.json")
    state = {"deadline": args.deadline, "jobs": [], "running": None}

    def save():
        state["updated"] = datetime.now().astimezone().isoformat()
        temp = state_path.with_suffix(".tmp")
        for attempt in range(20):
            try:
                temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
                temp.replace(state_path)
                return
            except PermissionError:
                # Windows readers/sync clients can transiently deny delete sharing.
                # A status-file collision must not terminate the GPU experiment.
                time.sleep(.1)
        print("WARNING: status snapshot locked; retrying on next heartbeat", flush=True)

    for job in jobs:
        if time.time() >= deadline-60:
            break
        log = folder / (job["name"]+".log")
        if log.exists():
            raise FileExistsError(log)
        record = {"name": job["name"], "started": datetime.now().astimezone().isoformat(), "log": str(log)}
        state["running"] = record
        save()
        print(f"START {job['name']}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            proc = subprocess.Popen([sys.executable, "-u"]+job["args"], stdout=stream,
                                    stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            try:
                heartbeat = 0
                while proc.poll() is None:
                    if time.time() >= deadline:
                        stop_owned_process(proc)
                        record["deadline_terminated"] = True
                        break
                    if time.monotonic() >= heartbeat:
                        record["pid"] = proc.pid
                        record["log_bytes"] = log.stat().st_size
                        save()
                        print(f"RUNNING {job['name']} {datetime.now().isoformat()}", flush=True)
                        heartbeat = time.monotonic()+45
                    time.sleep(1)
            finally:
                if proc.poll() is None:
                    stop_owned_process(proc)
        record["returncode"] = proc.returncode
        record["finished"] = datetime.now().astimezone().isoformat()
        state["jobs"].append(record)
        state["running"] = None
        save()
        print(f"FINISHED {job['name']} rc={proc.returncode}", flush=True)
        if proc.returncode != 0:
            break


if __name__ == "__main__":
    main()
