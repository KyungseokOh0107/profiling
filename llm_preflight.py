"""Record environment evidence before LLM experiments; never install packages."""

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def collect(gpu_smoke=False):
    packages = {}
    for name in ("torch", "transformers", "vllm", "huggingface-hub", "httpx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": sys.version,
        "executable": sys.executable, "packages": packages,
        "tools": {name: shutil.which(name) for name in ("nvidia-smi", "nsys", "ncu", "nvcc")},
        "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                               "--format=csv"]),
        "llm_inference_executed": False,
        "vllm_serving_executed": False,
    }
    if sys.platform == "win32":
        import winreg
        distributions = []
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r"Software\Microsoft\Windows\CurrentVersion\Lxss") as root:
                for i in range(winreg.QueryInfoKey(root)[0]):
                    with winreg.OpenKey(root, winreg.EnumKey(root, i)) as distro:
                        try:
                            distributions.append({
                                "name": winreg.QueryValueEx(distro, "DistributionName")[0],
                                "version": winreg.QueryValueEx(distro, "Version")[0],
                            })
                        except FileNotFoundError:
                            pass
        except FileNotFoundError:
            pass
        report["wsl_distributions"] = distributions
        report["vllm_blocker"] = "Official CUDA vLLM requires Linux; this interpreter is Windows."
    if gpu_smoke:
        try:
            import torch
            from torch.profiler import ProfilerActivity, profile
            gpu = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "cuda_available": torch.cuda.is_available()}
            if gpu["cuda_available"]:
                gpu["name"] = torch.cuda.get_device_name(0)
                gpu["capability"] = list(torch.cuda.get_device_capability(0))
                a = torch.ones((256, 256), dtype=torch.float16, device="cuda")
                b = a @ a
                torch.cuda.synchronize()
                gpu["fp16_matmul_correct"] = bool((b == 256).all().item())
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                    b = a @ a
                    torch.cuda.synchronize()
                gpu["cuda_profiler_events"] = sum(
                    str(e.device_type).lower().endswith("cuda") for e in trace.events())
            report["gpu_smoke"] = gpu
        except Exception as exc:
            report["gpu_smoke_error"] = repr(exc)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.gpu_smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
