"""Freeze workload rates only after the pilot proves quiet and queued regimes."""
from datetime import datetime
from pathlib import Path
import json

root = Path("results/overnight")
pilot = json.loads((root/"http_load_pilot"/"summaries.json").read_text())
extended = json.loads((root/"http_load_pilot_extended"/"summaries.json").read_text())
quiet = next(r for r in pilot if r["rate"] == .08)
busy = extended[0]
assert quiet["max_pending_at_capacity"] == 0 and quiet["max_active"] < 8
assert busy["max_pending_at_capacity"] > 0 and busy["max_admission_wait_ms"] > 1000, "No demonstrated admission queue"
path = root/"LOAD_RATES_FROZEN.json"
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps({"frozen_at": datetime.now().astimezone().isoformat(),
    "rates": [.08,.24], "quiet_pilot": quiet, "busy_pilot": busy,
    "interpretation": "0.24 is a finite overloaded arrival burst, not a stable infinite-horizon load",
    "main_trials": {"requests": 50, "repeats": 2, "chunks": [256,512,2048], "max_seqs": 8},
    "synthetic_slo": {"short_scheduled_ttft_ms":2000,"long_scheduled_ttft_ms":4000,"max_gap_ms":350}}, indent=2), encoding="utf-8")
print(path)
