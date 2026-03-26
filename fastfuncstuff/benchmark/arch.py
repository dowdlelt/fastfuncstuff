"""Architecture fingerprinting for benchmark result caching."""

from __future__ import annotations

import os
import platform

import torch


def get_arch_id() -> str:
    """Return a string identifying the current hardware architecture.

    Format: {os}-{machine}-{accelerator}
    Examples:
        linux-x86_64-cuda-NVIDIA_RTX_5070_Ti
        darwin-arm64-mps-Apple_M4_Pro
        linux-x86_64-cpu
    """
    system = platform.system().lower()
    machine = platform.machine()

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
        accel = f"cuda-{gpu_name}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        proc = platform.processor() or "unknown"
        accel = f"mps-{proc.replace(' ', '_')}"
    else:
        accel = "cpu"

    return f"{system}-{machine}-{accel}"


def get_arch_info() -> dict:
    """Return detailed architecture information."""
    info = {
        "arch_id": get_arch_id(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        info["gpu_memory_gb"] = round(mem / 1024**3, 1)
    return info
