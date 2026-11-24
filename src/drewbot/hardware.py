"""Actual device probes and optional metrics; no model, search or training behavior."""

from __future__ import annotations

import platform
import shutil
import sys

import psutil
import torch

from drewbot.config import ROOT


def inspect_hardware(preferred: str = "auto", allow_cpu_fallback: bool = True) -> dict:
    """WHAT: Report CPU/MPS operation and capacity. HOW: Execute tensors on each device.

    WHY: An availability flag alone does not establish that a tensor operation works.
    MPS failures are recorded and CPU is retained; optional memory metrics may be null.
    """

    def probe(device):
        try:
            result = torch.ones((2, 2), device=device) @ torch.ones((2, 2), device=device)
            if device == "mps":
                torch.mps.synchronize()
            if not torch.equal(result.cpu(), torch.full((2, 2), 2.0)):
                raise RuntimeError("Unexpected tensor operation result")
            return {"operational": True, "error": None}
        except (RuntimeError, AssertionError, NotImplementedError) as error:
            return {"operational": False, "error": str(error)}

    if preferred not in {"auto", "cpu", "mps"} or type(allow_cpu_fallback) is not bool:
        raise ValueError("Use auto/cpu/mps and a boolean fallback setting")
    cpu = probe("cpu")
    available = bool(torch.backends.mps.is_available())
    mps = probe("mps") if available else {"operational": False, "error": "MPS unavailable"}
    memory = {}
    for name in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
        try:
            memory[name] = int(getattr(torch.mps, name)()) if mps["operational"] else None
        except (AttributeError, RuntimeError, NotImplementedError):
            memory[name] = None
    if not cpu["operational"]:
        raise RuntimeError("CPU tensor operation failed: " + cpu["error"])
    if preferred == "mps" and not mps["operational"] and not allow_cpu_fallback:
        raise ValueError("MPS requested without fallback but is not operational: " + mps["error"])
    selected = "mps" if preferred != "cpu" and mps["operational"] else "cpu"
    return {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cpu": cpu,
        "mps_available": available,
        "mps": mps,
        "selected_device": selected,
        "requested_device": preferred,
        "memory_bytes": psutil.virtual_memory().total,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "mps_memory": memory,
    }
