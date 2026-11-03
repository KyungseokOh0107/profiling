"""Read-only compact campaign progress; counts completion events, not request IDs."""
import json
from datetime import datetime
from pathlib import Path


def main():
    root = Path("results/overnight")
    result = {"time": datetime.now().astimezone().isoformat()}
    for stage in (2, 3):
        path = root/f"jobs_stage{stage}_status.json"
        if not path.exists():
            continue
        try:
            state = json.loads(path.read_text())
        except (PermissionError, json.JSONDecodeError):
            continue
        result[f"stage{stage}"] = {"finished_jobs": len(state["jobs"]), "running": state["running"]}
        if state["running"]:
            log = Path(state["running"]["log"])
            current, done, errors, trials = None, 0, 0, 0
            for line in log.read_text(encoding="utf-8").splitlines():
                if line.startswith("START load_"):
                    current, done, errors = line.split()[1], 0, 0
                if line.startswith("{"):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("progress") == "request_completed":
                        prefix = data["request_id"].rsplit("_", 1)[0]
                        if prefix != current:
                            current, done, errors = prefix, 0, 0
                        done += 1
                        errors += data.get("status") != "ok"
                    if "run_id" in data and not data.get("warmup"):
                        trials += 1
                        if data.get("phase") == "pairs":
                            current = data["run_id"]
                    elif "output_tokens_per_request" in data and not data.get("warmup"):
                        trials += 1
                        current = f"r{data['repeat']}_batch{data['batch']}_threads{data['cpu_threads']}_input{data['input_tokens']}"
            result[f"stage{stage}"].update(current_trial=current, completed_requests_current_trial=done,
                errors_current_trial=errors, completed_trials_current_job=trials)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
