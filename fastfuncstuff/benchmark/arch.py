"""Architecture fingerprinting for benchmark result caching."""

from __future__ import annotations

import os
import platform

import torch


def _get_cpu_model() -> str:
    """Best-effort CPU model name."""
    # Linux: parse /proc/cpuinfo
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    # macOS / fallback
    return platform.processor() or "unknown"


def get_ref_arch_id() -> str:
    """Arch ID relevant for *reference* tools (CPU-bound: AFNI, melodic, MATLAB).

    Format: {os}-{cpu_arch}
    Examples: linux-x86_64, darwin-arm64
    """
    system = platform.system().lower()
    machine = platform.machine()
    return f"{system}-{machine}"


def get_ffs_arch_id(device_spec: str | None = None) -> str:
    """Arch ID relevant for *FFS* tools (GPU-accelerated).

    Format: cuda-{gpu_name} | mps-{processor} | cpu
    Examples: cuda-NVIDIA_GeForce_RTX_5070_Ti, mps-Apple_M4_Pro, cpu
    """
    requested = (device_spec or "auto").split(",", 1)[0].strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        index = 0
        if requested == "cuda" and device_spec and "," in device_spec:
            index = int(device_spec.split(",", 1)[1])
        gpu_name = torch.cuda.get_device_name(index).replace(" ", "_")
        return f"cuda-{gpu_name}"
    if requested == "mps":
        proc = platform.processor() or "unknown"
        return f"mps-{proc.replace(' ', '_')}"
    return "cpu"


def get_arch_id() -> str:
    """Combined arch ID (legacy — kept for backward compat with v1 cache).

    Format: {os}-{machine}-{ffs_arch_id}
    """
    return f"{get_ref_arch_id()}-{get_ffs_arch_id()}"


def get_hardware_info(device_spec: str | None = None) -> dict:
    """Return comprehensive hardware info for a cache entry."""
    n_logical = os.cpu_count() or 1
    try:
        import psutil

        n_physical = psutil.cpu_count(logical=False) or n_logical
    except ImportError:
        n_physical = n_logical

    info: dict = {
        "ref_arch_id": get_ref_arch_id(),
        "ffs_arch_id": get_ffs_arch_id(device_spec),
        "requested_device": device_spec or "auto",
        "os": platform.system().lower(),
        "cpu_arch": platform.machine(),
        "cpu_model": _get_cpu_model(),
        "n_logical_cores": n_logical,
        "n_physical_cores": n_physical,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        info["gpu_memory_gb"] = round(mem / 1024**3, 1)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["gpu"] = platform.processor() or "Apple Silicon"
    return info


def get_arch_info() -> dict:
    """Backward-compatible alias for get_hardware_info()."""
    return get_hardware_info()
