#!/usr/bin/env python3
"""Read-only Runpod inventory. Does not install packages, load models, or start training.

Run with the intended training interpreter; stdout is JSON. Additional --path arguments
inspect model/data/output mounts without recursively listing their contents.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(args: list[str]) -> dict:
    if shutil.which(args[0]) is None:
        return {"available": False}
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"available": True, "returncode": result.returncode,
                "stdout": result.stdout[:20000], "stderr": result.stderr[:2000]}
    except subprocess.TimeoutExpired:
        return {"available": True, "timed_out": True}


def inspect_path(value: str) -> dict:
    path = Path(value)
    result = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    try:
        usage = shutil.disk_usage(path)
        result.update(total_bytes=usage.total, used_bytes=usage.used, free_bytes=usage.free)
    except OSError as exc:
        result["disk_usage_error"] = type(exc).__name__
    # Exclude mount sources/options: they may contain connection information.
    result["mount"] = command(["findmnt", "--json", "--target", str(path), "--output", "TARGET,FSTYPE"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", default=[])
    args = parser.parse_args()
    packages = {}
    for name in ("torch", "vllm", "transformers", "flash-attn", "ray", "verl", "numpy", "pyarrow"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    memory = {}
    info = Path("/proc/meminfo")
    if info.exists():
        for line in info.read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                memory[key] = value.strip()
    for location in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        limit = Path(location)
        if limit.exists():
            memory["cgroup_limit_bytes"] = limit.read_text().strip()
            break
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "system": platform.system(), "machine": platform.machine(),
        "python": sys.version.split()[0], "interpreter": sys.executable,
        "cpu_count": os.cpu_count(), "memory": memory, "packages": packages,
        "gpus": command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,driver_version",
                         "--format=csv,noheader,nounits"]),
        "gpu_topology": command(["nvidia-smi", "topo", "-m"]),
        "paths": [inspect_path(p) for p in dict.fromkeys(["/", "/workspace", "/workspace-global", *args.path])],
        "validation_scope": "inventory_only",
        "not_validated": ["CUDA kernel compatibility", "NCCL collectives", "model/data completeness",
                          "training configuration", "storage write semantics", "end-to-end training step"],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
