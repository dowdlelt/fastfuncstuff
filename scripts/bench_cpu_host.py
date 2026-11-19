#!/usr/bin/env python3
"""How should ffs run on *this* CPU host?  Measure it instead of guessing.

A deployment question -- "more threads or more jobs?", "is MKL slow on this
AMD box?", "did jemalloc help?" -- usually gets answered from folklore. This
script answers it on the machine in front of you, in three parts:

1. **Host report.** CPU, cores, NUMA nodes, L3 groups (CCDs on EPYC), governor,
   transparent huge pages, cgroup quota, preloaded allocator, which BLAS torch
   and numpy/scipy are built on (and which kernel OpenBLAS picked), and the
   thread count ffs itself would choose.

2. **Thread scaling.** Representative kernels -- dense GEMM, the voxelwise-OLS
   shape, a memory-bandwidth triad, a time-axis reduction, QR/eigh/SVD, batched
   per-voxel solves, an FFT along time, 3-D grid_sample -- at 1, 2, 4, ... threads.
   Efficiency at full width says whether one job can use the machine.

3. **Concurrent jobs.** The same kernels run as J processes x T threads that
   together fill the physical cores, each pinned to its own contiguous slice
   (consecutive L3 groups / NUMA nodes), the way ``numactl``/``taskset`` would
   deploy them. Aggregate throughput against the 1-job layout is the cohort
   speedup you'd get from running subjects side by side.

Typical use::

    python scripts/bench_cpu_host.py --json host_a.json           # ~5-10 min
    python scripts/bench_cpu_host.py --quick                      # ~1-2 min smoke run
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \\
        python scripts/bench_cpu_host.py --json host_a_jemalloc.json
    python scripts/bench_cpu_host.py --compare host_a.json host_a_jemalloc.json

Any environment A/B (allocator, ``OMP_PROC_BIND``, THP, governor, a torch
upgrade, another host) is the same pattern: two ``--json`` runs, one
``--compare``. Run the pair back to back on an otherwise idle box, and alternate
them if the difference is small -- one run per setting measures drift as much
as the setting.

Timings are the *minimum* over repeats (kernel cost, not scheduler noise). The
scaling section is unpinned by default because that is how an ffs CLI runs
today; ``--pin-scaling`` pins T threads to the first T cores so you can see the
cost of letting the OS scatter them.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import shutil
import sys
import time
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Kernels. Each builder imports torch itself: job workers must set their CPU
# affinity before torch (and its OpenMP pool) exists, so nothing at module
# scope may import it.
# ---------------------------------------------------------------------------

Built = tuple[Callable[[], Any], float, int]  # (call, work per call, bytes of data)


def _gemm(dtype: str, quick: bool) -> Built:
    import torch

    n = 1024 if quick else 2048
    dt = getattr(torch, dtype)
    a, b = torch.randn(n, n, dtype=dt), torch.randn(n, n, dtype=dt)
    out = torch.empty(n, n, dtype=dt)
    return (lambda: torch.matmul(a, b, out=out)), 2 * n**3 / 1e9, 3 * n * n * a.element_size()


def _glm(quick: bool) -> Built:
    # The shape of every voxelwise OLS/ridge apply: a small (p, T) operator
    # against a wide (T, V) data block. Moderate arithmetic intensity, so it
    # sits between GEMM and the pure-bandwidth kernels.
    import torch

    t, p, v = 500, 50, 25_000 if quick else 100_000
    pinv = torch.linalg.pinv(torch.randn(t, p))
    y = torch.randn(t, v)
    out = torch.empty(p, v)
    return (lambda: torch.matmul(pinv, y, out=out)), 2 * p * t * v / 1e9, 4 * (t * v + p * v)


def _stream(quick: bool) -> Built:
    import torch

    n = 8_000_000 if quick else 32_000_000
    x, y, out = torch.randn(n), torch.randn(n), torch.empty(n)
    return (lambda: torch.add(y, x, alpha=1.5, out=out)), 12 * n / 1e9, 12 * n


def _reduce(quick: bool) -> Built:
    import torch

    v, t = 25_000 if quick else 100_000, 500
    y = torch.randn(v, t)
    return (lambda: y.std(dim=1)), 4 * v * t / 1e9, 4 * v * t


def _qr(quick: bool) -> Built:
    import torch

    m, n = (2000, 100) if quick else (4000, 200)
    a = torch.randn(m, n, dtype=torch.float64)
    return (lambda: torch.linalg.qr(a)), (2 * m * n * n - 2 * n**3 / 3) / 1e9, 2 * a.numel() * 8


def _eigh(quick: bool) -> Built:
    import torch

    n = 500 if quick else 1000
    a = torch.randn(n, n, dtype=torch.float64)
    a = a @ a.T
    return (lambda: torch.linalg.eigh(a)), 1.0, 2 * n * n * 8


def _svd(quick: bool) -> Built:
    import torch

    a = torch.randn(1000 if quick else 2000, 500)
    return (lambda: torch.linalg.svd(a, full_matrices=False)), 1.0, a.numel() * 8


def _eigh_batch(quick: bool) -> Built:
    import torch

    b = 5_000 if quick else 20_000
    a = torch.randn(b, 8, 8)
    a = a @ a.transpose(-1, -2)
    return (lambda: torch.linalg.eigh(a)), b / 1e6, a.numel() * 4 * 2


def _cholsolve_batch(quick: bool) -> Built:
    # Per-voxel small SPD solves in float64: the REML / ARMA inner loop shape.
    import torch

    bsz, k = (10_000 if quick else 50_000), 10
    x = torch.randn(bsz, k, k, dtype=torch.float64)
    a = x @ x.transpose(-1, -2) + k * torch.eye(k, dtype=torch.float64)
    rhs = torch.randn(bsz, k, 1, dtype=torch.float64)

    def call():
        return torch.cholesky_solve(rhs, torch.linalg.cholesky(a))

    return call, bsz / 1e6, a.numel() * 8 * 2


def _rfft(quick: bool) -> Built:
    import torch

    v, t = (10_000 if quick else 50_000), 512
    x = torch.randn(v, t)
    return (lambda: torch.fft.rfft(x, dim=-1)), v / 1e6, x.numel() * 4 * 2


def _grid_sample(quick: bool) -> Built:
    # Trilinear resampling of a volume through a dense warp: the registration /
    # nwarp / moco hot path on CPU.
    import torch
    import torch.nn.functional as F

    d, h, w = (48, 64, 64) if quick else (96, 128, 128)
    vol = torch.randn(1, 1, d, h, w)
    theta = torch.eye(3, 4).unsqueeze(0)
    grid = F.affine_grid(theta, [1, 1, d, h, w], align_corners=False)
    grid = grid + 0.01 * torch.randn_like(grid)

    def call():
        return F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=False)

    return call, d * h * w / 1e6, vol.numel() * 4 * 5


def _ffs_resample(quick: bool) -> Built:
    # The same warp through ffs's own wrapper, which deals the points across
    # batch lanes because ATen threads grid_sampler_3d over N only -- stock
    # torch runs an N=1 volume on one core however many threads it has.
    import torch
    import torch.nn.functional as F

    from fastfuncstuff.processing.interp import _grid_sample_3d

    d, h, w = (48, 64, 64) if quick else (96, 128, 128)
    vol = torch.randn(1, 1, d, h, w)
    grid = F.affine_grid(torch.eye(3, 4).unsqueeze(0), [1, 1, d, h, w], align_corners=True)
    grid = grid + 0.01 * torch.randn_like(grid)
    return (lambda: _grid_sample_3d(vol, grid)), d * h * w / 1e6, vol.numel() * 4 * 5


def _numpy_gemm(quick: bool) -> Built:
    import numpy as np

    n = 1024 if quick else 2048
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    out = np.empty((n, n))
    return (lambda: np.matmul(a, b, out=out)), 2 * n**3 / 1e9, 3 * n * n * 8


def _scipy_eigh(quick: bool) -> Built:
    import numpy as np
    import scipy.linalg

    n = 500 if quick else 1000
    a = np.random.default_rng(0).standard_normal((n, n))
    a = a @ a.T
    return (lambda: scipy.linalg.eigh(a)), 1.0, 2 * n * n * 8


# name -> (description, unit, library whose threads apply, builder)
KERNELS: dict[str, tuple[str, str, str, Callable[[bool], Built]]] = {
    "gemm_f32": ("square GEMM 2048 f32", "GFLOP/s", "torch", lambda q: _gemm("float32", q)),
    "gemm_f64": ("square GEMM 2048 f64", "GFLOP/s", "torch", lambda q: _gemm("float64", q)),
    "glm_f32": ("pinv(X)[50x500] @ Y[500x100k]", "GFLOP/s", "torch", _glm),
    "stream_f32": ("triad out=y+a*x, 32M floats", "GB/s", "torch", _stream),
    "reduce_f32": ("std over time, (100k,500)", "GB/s", "torch", _reduce),
    "qr_f64": ("QR (4000,200) f64", "GFLOP/s", "torch", _qr),
    "eigh_f64": ("eigh 1000x1000 f64", "call/s", "torch", _eigh),
    "svd_f32": ("SVD (2000,500) f32", "call/s", "torch", _svd),
    "eigh_batch_f32": ("batched eigh 20k x (8,8)", "Mmat/s", "torch", _eigh_batch),
    "cholsolve_f64": ("batched chol+solve 50k x (10,10) f64", "Mmat/s", "torch", _cholsolve_batch),
    "rfft_f32": ("rfft along time, (50k,512)", "Mseries/s", "torch", _rfft),
    "grid_sample_f32": ("stock F.grid_sample 3-D warp 96x128x128", "Mvox/s", "torch", _grid_sample),
    "ffs_resample_f32": ("ffs _grid_sample_3d, same warp", "Mvox/s", "torch", _ffs_resample),
    "numpy_gemm_f64": ("numpy GEMM 2048 f64", "GFLOP/s", "numpy", _numpy_gemm),
    "scipy_eigh_f64": ("scipy eigh 1000x1000 f64", "call/s", "numpy", _scipy_eigh),
}
DEFAULT_JOB_KERNELS = ["gemm_f32", "glm_f32", "stream_f32", "ffs_resample_f32", "cholsolve_f64"]


# ---------------------------------------------------------------------------
# Host introspection
# ---------------------------------------------------------------------------


def _read(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _parse_cpulist(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def cpu_topology() -> list[dict]:
    """Physical cores in the affinity mask, ordered node -> L3 group -> core.

    Contiguous slices of this list are what a job should be pinned to: on EPYC
    a slice the size of one CCD shares one L3, a slice the size of a node
    shares one memory controller set.
    """
    cores: dict[tuple, dict] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = f"/sys/devices/system/cpu/cpu{cpu}"
        pkg = _read(f"{base}/topology/physical_package_id") or "0"
        core = _read(f"{base}/topology/core_id") or str(cpu)
        l3 = None
        for idx in glob.glob(f"{base}/cache/index*"):
            if _read(f"{idx}/level") == "3":
                l3 = _read(f"{idx}/id") or _read(f"{idx}/shared_cpu_list")
                break
        nodes = glob.glob(f"{base}/node*")
        node = int(nodes[0].rsplit("node", 1)[1]) if nodes else 0
        rec = cores.setdefault(
            (pkg, core), {"cpus": [], "node": node, "l3": f"{pkg}:{l3}", "pkg": pkg}
        )
        rec["cpus"].append(cpu)

    def l3_order(rec):
        return (rec["node"], rec["pkg"], rec["l3"], min(rec["cpus"]))

    return sorted(cores.values(), key=l3_order)


def _cpus_for(topo: list[dict], start: int, n_threads: int) -> list[int]:
    """One logical CPU per core for ``n_threads`` cores from ``start``; SMT siblings past that."""
    n_cores = len(topo)
    if n_threads <= n_cores - start:
        return [min(c["cpus"]) for c in topo[start : start + n_threads]]
    return sorted(cpu for c in topo for cpu in c["cpus"])


def _set_affinity_all_threads(cpus: set[int]) -> None:
    # sched_setaffinity(0) moves only the calling thread; an existing OpenMP
    # pool keeps its old mask unless every task is moved.
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
        except OSError:
            pass


def _cpu_quota() -> float | None:
    try:
        from fastfuncstuff.utils import cgroup_cpu_quota

        return cgroup_cpu_quota()
    except ImportError:  # running without ffs installed: top-level cgroup v2 only
        raw = (_read("/sys/fs/cgroup/cpu.max") or "max").split()
        return None if raw[0] == "max" else int(raw[0]) / int(raw[1] if len(raw) > 1 else 100000)


def host_report() -> dict:
    import numpy as np
    import scipy.linalg  # noqa: F401 -- loads scipy's BLAS so threadpoolctl sees it
    import torch

    info: dict[str, Any] = {"hostname": platform.node(), "python": platform.python_version()}
    cpuinfo = _read("/proc/cpuinfo") or ""
    m = re.search(r"model name\s*:\s*(.+)", cpuinfo)
    info["cpu_model"] = m.group(1).strip() if m else platform.processor()
    fm = re.search(r"flags\s*:\s*(.+)", cpuinfo)
    flags = set(fm.group(1).split()) if fm else set()
    info["isa"] = [
        f
        for f in ("avx2", "fma", "avx512f", "avx512_bf16", "avx512_fp16", "amx_tile")
        if f in flags
    ]

    topo = cpu_topology()
    info["physical_cores"] = len(topo)
    info["logical_cpus"] = sum(len(c["cpus"]) for c in topo)
    info["sockets"] = len({c["pkg"] for c in topo})
    info["numa_nodes"] = {
        os.path.basename(n): _read(f"{n}/cpulist")
        for n in sorted(glob.glob("/sys/devices/system/node/node[0-9]*"))
    }
    l3_groups: dict[str, int] = {}
    for c in topo:
        l3_groups[c["l3"]] = l3_groups.get(c["l3"], 0) + 1
    info["l3_groups"] = len(l3_groups)
    info["cores_per_l3"] = sorted(set(l3_groups.values()))
    info["l3_size"] = _read("/sys/devices/system/cpu/cpu0/cache/index3/size")
    fmax = _read("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    info["max_ghz"] = round(int(fmax) / 1e6, 2) if fmax else None
    info["governor"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    info["energy_pref"] = _read(
        "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"
    )
    info["thp_enabled"] = _read("/sys/kernel/mm/transparent_hugepage/enabled")
    info["thp_defrag"] = _read("/sys/kernel/mm/transparent_hugepage/defrag")
    mem = _read("/proc/meminfo") or ""
    for key in ("MemTotal", "MemAvailable"):
        mm = re.search(rf"{key}:\s*(\d+)", mem)
        info[key] = round(int(mm.group(1)) / 2**20, 1) if mm else None  # GiB

    maps = _read("/proc/self/maps") or ""
    info["allocator"] = next(
        (a for a in ("jemalloc", "tcmalloc", "mimalloc") if a in maps), "glibc"
    )
    info["env"] = {
        k: v
        for k, v in os.environ.items()
        if re.match(r"(OMP|MKL|KMP|GOMP|OPENBLAS|FFS|TORCHINDUCTOR)_", k) or k == "LD_PRELOAD"
    }

    cfg = torch.__config__.show()
    blas = re.search(r"BLAS_INFO=(\w+)", cfg)
    mkl = re.search(r"Math Kernel Library Version ([\w.\-]+)", cfg)
    info["torch"] = torch.__version__
    info["torch_blas"] = (blas.group(1) if blas else "?") + (f" {mkl.group(1)}" if mkl else "")
    info["torch_cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    info["cxx_for_torch_compile"] = shutil.which(os.environ.get("CXX", "g++")) or shutil.which(
        "c++"
    )
    info["numpy"] = np.__version__
    try:
        from threadpoolctl import threadpool_info

        info["threadpools"] = [
            {
                k: p.get(k)
                for k in ("internal_api", "version", "architecture", "num_threads", "filepath")
            }
            for p in threadpool_info()
        ]
    except ImportError:
        info["threadpools"] = "threadpoolctl not installed"

    info["cgroup_cpu_quota"] = _cpu_quota()
    # A container quota (Kubernetes cpu limit) hides nothing from nproc or the
    # affinity mask, but threads past it get the whole process frozen each
    # period -- so the quota, not the core count, is how far to push.
    q = info["cgroup_cpu_quota"]
    info["usable_cores"] = min(len(topo), max(1, int(q))) if q else len(topo)
    try:
        from fastfuncstuff.utils import resolve_cpu_threads

        n, src = resolve_cpu_threads()
        info["ffs_default_threads"] = f"{n} ({src})"
    except Exception as exc:  # noqa: BLE001 -- the report must not die on an ffs import
        info["ffs_default_threads"] = f"unavailable: {exc}"
    return info


def print_host(info: dict) -> None:
    print("\n== Host ==")
    print(f"  {info['cpu_model']}  [{info['hostname']}]")
    print(
        f"  {info['sockets']} socket(s), {info['physical_cores']} cores / {info['logical_cpus']} CPUs, "
        f"{len(info['numa_nodes'])} NUMA node(s), {info['l3_groups']} L3 group(s) of "
        f"{'/'.join(map(str, info['cores_per_l3']))} cores ({info['l3_size']} each)"
    )
    print(
        f"  ISA: {' '.join(info['isa'])}   max {info['max_ghz']} GHz   governor={info['governor']}"
        f"  epp={info['energy_pref']}"
    )
    print(
        f"  RAM {info['MemTotal']} GiB ({info['MemAvailable']} available)   THP enabled={info['thp_enabled']}"
        f"  defrag={info['thp_defrag']}"
    )
    print(
        f"  allocator={info['allocator']}   cgroup quota={info.get('cgroup_cpu_quota')}"
        f" -> testing up to {info['usable_cores']} cores"
        f"   ffs default threads={info['ffs_default_threads']}"
    )
    print(
        f"  torch {info['torch']}  BLAS={info['torch_blas']}  capability={info['torch_cpu_capability']}"
        f"  C++ for torch.compile={info['cxx_for_torch_compile'] or 'MISSING'}"
    )
    if isinstance(info["threadpools"], list):
        for p in info["threadpools"]:
            print(
                f"  pool: {p['internal_api']} {p['version'] or ''} arch={p['architecture']} "
                f"threads={p['num_threads']}  ({os.path.basename(p['filepath'] or '')})"
            )
    else:
        print(f"  pools: {info['threadpools']}")
    if info["env"]:
        print("  env: " + "  ".join(f"{k}={v}" for k, v in sorted(info["env"].items())))


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


class _Threads:
    """Set the thread count for torch or for the BLAS under numpy/scipy."""

    def __init__(self):
        self._limits = None

    def set(self, lib: str, n: int) -> bool:
        if lib == "torch":
            import torch

            torch.set_num_threads(n)
            return True
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            return False
        if self._limits is not None:
            self._limits.restore_original_limits()
        self._limits = threadpool_limits(limits=n, user_api="blas")
        return True


def time_call(fn: Callable[[], Any], min_time: float, min_reps: int = 3) -> float:
    fn()  # warm-up: allocator, lazy init, thread pool spin-up
    best, total, reps = float("inf"), 0.0, 0
    while reps < min_reps or total < min_time:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best, total, reps = min(best, dt), total + dt, reps + 1
        if reps >= 50:
            break
    return best


def run_scaling(names, thread_counts, quick, min_time, pin, topo) -> list[dict]:
    rows = []
    threads = _Threads()
    all_cpus = set(os.sched_getaffinity(0))
    print(f"\n== Thread scaling ({'pinned to first T cores' if pin else 'unpinned'}) ==")
    for name in names:
        desc, unit, lib, build = KERNELS[name]
        try:
            fn, work, _ = build(quick)
        except ImportError as exc:
            print(f"  {name:<16} skipped: {exc}")
            continue
        line = f"  {name:<16} {unit:<10}"
        print(line, end="", flush=True)
        for t in thread_counts:
            if not threads.set(lib, t):
                if t != thread_counts[0]:
                    break
                print(" (threads not controllable: pip install threadpoolctl)", end="")
            if pin:
                _set_affinity_all_threads(set(_cpus_for(topo, 0, t)))
            sec = time_call(fn, min_time)
            rows.append(
                {
                    "kernel": name,
                    "threads": t,
                    "sec_per_call": sec,
                    "rate": work / sec,
                    "unit": unit,
                }
            )
            print(f" {t}:{work / sec:.4g}", end="", flush=True)
        if pin:
            _set_affinity_all_threads(all_cpus)
        print()
        del fn
    return rows


def _job_worker(name, quick, n_threads, cpus, barrier, queue, duration):
    try:
        if cpus:
            os.sched_setaffinity(0, set(cpus))
        _, _, lib, build = KERNELS[name]
        if lib == "torch":
            import torch

            torch.set_num_threads(n_threads)
        fn, work, _ = build(quick)
        fn()
        fn()
        barrier.wait(timeout=900)
        t0 = time.perf_counter()
        count = 0
        while True:
            fn()
            count += 1
            elapsed = time.perf_counter() - t0
            if elapsed >= duration:
                break
        queue.put(("ok", count, elapsed, work))
    except Exception as exc:  # noqa: BLE001 -- report, don't hang the parent
        queue.put(("error", repr(exc), 0.0, 0.0))
        try:
            barrier.abort()
        except Exception:  # noqa: BLE001
            pass


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def run_jobs(names, layouts, quick, duration, pin, topo, mem_avail_gib) -> list[dict]:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    rows = []
    per_l3 = max(set(sum(1 for c in topo if c["l3"] == g) for g in {c["l3"] for c in topo}))
    nodes = {c["node"] for c in topo}
    per_node = len(topo) // max(1, len(nodes))
    print(
        f"\n== Concurrent jobs ({'each pinned to its own core slice' if pin else 'unpinned'}, "
        f"{duration:g} s each) =="
    )
    for name in names:
        desc, unit, _, _ = KERNELS[name]
        print(f"  {name}  ({desc}, {unit})")
        print(f"    {'layout':<10} {'aggregate':>11} {'vs 1 job':>9} {'per-job s/call':>15}")
        base = None
        for n_jobs, n_threads in layouts:
            # Rough memory guard: each worker holds its kernel's arrays plus
            # interpreter + torch (~0.5 GiB).
            est_gib = n_jobs * (_kernel_gib(name, quick) * 1.5 + 0.5)
            if mem_avail_gib and est_gib > 0.6 * mem_avail_gib:
                print(f"    {n_jobs}x{n_threads:<8} skipped: needs ~{est_gib:.0f} GiB")
                continue
            env_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            saved = {k: os.environ.get(k) for k in env_keys}
            for k in env_keys:
                os.environ[k] = str(n_threads)
            barrier, queue = ctx.Barrier(n_jobs), ctx.Queue()
            procs = []
            for j in range(n_jobs):
                cpus = _cpus_for(topo, j * n_threads, n_threads) if pin else None
                p = ctx.Process(
                    target=_job_worker,
                    args=(name, quick, n_threads, cpus, barrier, queue, duration),
                )
                p.start()
                procs.append(p)
            results = [queue.get(timeout=1800) for _ in procs]
            for p in procs:
                p.join()
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            errors = [r[1] for r in results if r[0] != "ok"]
            if errors:
                print(f"    {n_jobs}x{n_threads:<8} FAILED: {errors[0]}")
                continue
            agg = sum(count * work / elapsed for _, count, elapsed, work in results)
            per_call = sum(elapsed / count for _, count, elapsed, _ in results) / len(results)
            base = base or agg
            tag = ""
            if n_threads == per_l3 and per_l3 != len(topo):
                tag = "  = 1 L3 group/job"
            elif n_threads == per_node and len(nodes) > 1:
                tag = "  = 1 NUMA node/job"
            print(
                f"    {n_jobs}x{n_threads:<8} {agg:>11.4g} {agg / base:>8.2f}x {per_call:>15.4g}{tag}"
            )
            rows.append(
                {
                    "kernel": name,
                    "jobs": n_jobs,
                    "threads": n_threads,
                    "aggregate_rate": agg,
                    "per_job_sec_per_call": per_call,
                    "unit": unit,
                }
            )
    return rows


def _kernel_gib(name: str, quick: bool) -> float:
    # Static estimates (GiB) so the guard need not build every kernel in the parent.
    full = {
        "glm_f32": 0.22,
        "stream_f32": 0.39,
        "reduce_f32": 0.2,
        "grid_sample_f32": 0.04,
        "ffs_resample_f32": 0.04,
        "cholsolve_f64": 0.09,
        "rfft_f32": 0.2,
        "eigh_batch_f32": 0.01,
    }
    return full.get(name, 0.1) / (4 if quick else 1)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def observations(info: dict, scaling: list[dict], jobs: list[dict]) -> list[str]:
    notes = []
    n_phys = info["usable_cores"]
    by = {(r["kernel"], r["threads"]): r for r in scaling}
    for name in dict.fromkeys(r["kernel"] for r in scaling):
        r1 = by.get((name, 1))
        top = max((t for k, t in by if k == name and t <= n_phys), default=1)
        if r1 and top > 1:
            eff = by[(name, top)]["rate"] / (top * r1["rate"])
            if eff < 0.5:
                notes.append(
                    f"{name}: {eff:.0%} parallel efficiency at {top} threads -- one job cannot use this box"
                )
        smt = by.get((name, info["logical_cpus"]))
        if smt and (name, n_phys) in by and smt["rate"] < 0.95 * by[(name, n_phys)]["rate"]:
            notes.append(
                f"{name}: SMT siblings make it slower ({smt['rate'] / by[(name, n_phys)]['rate']:.2f}x)"
            )
    g = by.get(("gemm_f32", 1))
    if g and info.get("max_ghz"):
        lanes = 16 if "avx512f" in info["isa"] else 8 if "avx2" in info["isa"] else 4
        peak = 2 * 2 * lanes * info["max_ghz"]  # 2 FMA units x 2 flop x lanes x GHz
        notes.append(
            f"gemm_f32 1-thread: {g['rate']:.0f} GFLOP/s = {g['rate'] / peak:.0%} of a nominal {peak:.0f} "
            "(assumes two full-width FMA units at max boost; Zen 4 has half that at 512-bit). "
            "Well under ~50% on a full-width core suggests the BLAS is not using its best kernel."
        )
    for name in dict.fromkeys(r["kernel"] for r in jobs):
        rs = [r for r in jobs if r["kernel"] == name]
        one = next((r for r in rs if r["jobs"] == 1), None)
        best = max(rs, key=lambda r: r["aggregate_rate"])
        if one and best is not one:
            notes.append(
                f"{name}: best layout {best['jobs']}x{best['threads']} = "
                f"{best['aggregate_rate'] / one['aggregate_rate']:.2f}x the throughput of 1x{one['threads']}"
            )
    if info["allocator"] == "glibc":
        notes.append(
            "allocator is glibc malloc -- try an A/B with LD_PRELOAD of libjemalloc.so.2 or libtcmalloc"
        )
    if info.get("governor") not in (None, "performance"):
        notes.append(f"cpufreq governor is '{info['governor']}', not 'performance'")
    if not info.get("cxx_for_torch_compile"):
        notes.append(
            "no C++ compiler found: torch.compile cannot build CPU kernels (apt install build-essential)"
        )
    return notes


def compare(paths: list[str]) -> None:
    runs = []
    for p in paths:
        with open(p) as fh:
            runs.append((os.path.splitext(os.path.basename(p))[0], json.load(fh)))
    labels = [lab for lab, _ in runs]
    w = max(10, *(len(lab) for lab in labels))
    print("\n== Hosts ==")
    for lab, r in runs:
        h = r["host"]
        print(
            f"  {lab}: {h['cpu_model']}, {h['physical_cores']} cores, torch {h['torch']} {h['torch_blas']}, "
            f"allocator={h['allocator']}, env={h['env']}"
        )

    def table(title, key_fn, rate_key, rows_of):
        keys = list(dict.fromkeys(k for _, r in runs for k in map(key_fn, rows_of(r))))
        if not keys:
            return
        print(f"\n== {title} ==  (ratio vs {labels[0]})")
        print(f"  {'':<28}" + "".join(f"{lab:>{w + 8}}" for lab in labels))
        for key in keys:
            vals = []
            for _, r in runs:
                hit = next((x for x in rows_of(r) if key_fn(x) == key), None)
                vals.append(hit[rate_key] if hit else None)
            cells = []
            for v in vals:
                if v is None:
                    cells.append(f"{'-':>{w + 8}}")
                elif vals[0]:
                    cells.append(f"{v:>{w}.4g} {v / vals[0]:>5.2f}x ")
                else:
                    cells.append(f"{v:>{w}.4g}        ")
            print(f"  {key:<28}" + "".join(cells))

    table(
        "Thread scaling",
        lambda x: f"{x['kernel']} @{x['threads']}",
        "rate",
        lambda r: r.get("scaling", []),
    )
    table(
        "Concurrent jobs",
        lambda x: f"{x['kernel']} {x['jobs']}x{x['threads']}",
        "aggregate_rate",
        lambda r: r.get("jobs", []),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--quick", action="store_true", help="smaller problems, fewer thread counts and layouts"
    )
    ap.add_argument("--kernels", help=f"comma list for scaling (default all): {','.join(KERNELS)}")
    ap.add_argument(
        "--job-kernels",
        help=f"comma list for the jobs section (default {','.join(DEFAULT_JOB_KERNELS)})",
    )
    ap.add_argument(
        "--threads",
        help="comma list of thread counts (default 1,2,4,... up to all cores, plus SMT)",
    )
    ap.add_argument(
        "--layouts",
        help="comma list of JOBSxTHREADS, e.g. 1x64,4x16,8x8 (default: every even split)",
    )
    ap.add_argument("--duration", type=float, help="seconds per job layout (default 5, quick 2)")
    ap.add_argument(
        "--min-time", type=float, default=0.3, help="min seconds of repeats per scaling point"
    )
    ap.add_argument(
        "--max-cpus",
        type=int,
        help="never use more than this many cores (default: the cgroup CPU quota, else all)",
    )
    ap.add_argument("--pin-scaling", action="store_true", help="pin T threads to the first T cores")
    ap.add_argument("--no-pin-jobs", action="store_true", help="let the OS place job processes")
    ap.add_argument("--skip-scaling", action="store_true")
    ap.add_argument("--skip-jobs", action="store_true")
    ap.add_argument("--json", help="write host info and results here (feed to --compare)")
    ap.add_argument(
        "--compare", nargs="+", metavar="JSON", help="compare earlier --json runs and exit"
    )
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return

    info = host_report()
    print_host(info)
    topo = cpu_topology()
    n_all = len(topo)
    n_phys = min(info["usable_cores"], args.max_cpus or n_all)
    info["usable_cores"] = n_phys
    # Contiguous in L3/NUMA order, so a capped run fills whole CCDs / nodes
    # rather than one core from each.
    topo = topo[:n_phys]
    # SMT siblings are only worth testing when nothing caps us below the full
    # core count; under a quota extra threads just oversubscribe it.
    n_log = info["logical_cpus"] if n_phys == n_all else n_phys
    if n_phys < n_all:
        print(f"\n  using {n_phys} of {n_all} cores (cgroup quota / --max-cpus)")

    if args.threads:
        thread_counts = [int(t) for t in args.threads.split(",")]
    else:
        thread_counts = [1 << i for i in range(n_phys.bit_length()) if 1 << i < n_phys] + [n_phys]
        if n_log > n_phys:
            thread_counts.append(n_log)
        if args.quick:
            thread_counts = sorted({1, max(1, n_phys // 4), n_phys})
    scale_names = args.kernels.split(",") if args.kernels else list(KERNELS)
    if args.quick and not args.kernels:
        scale_names = [
            "gemm_f32",
            "gemm_f64",
            "glm_f32",
            "stream_f32",
            "cholsolve_f64",
            "grid_sample_f32",
            "ffs_resample_f32",
            "numpy_gemm_f64",
        ]
    job_names = args.job_kernels.split(",") if args.job_kernels else list(DEFAULT_JOB_KERNELS)
    if args.quick and not args.job_kernels:
        job_names = ["glm_f32", "ffs_resample_f32"]
    for n in scale_names + job_names:
        if n not in KERNELS:
            sys.exit(f"unknown kernel {n!r}; choose from {', '.join(KERNELS)}")

    if args.layouts:
        layouts = [tuple(int(x) for x in s.split("x")) for s in args.layouts.split(",")]
    else:
        layouts = [(d, n_phys // d) for d in _divisors(n_phys)]
        if args.quick and len(layouts) > 3:
            layouts = [layouts[0], layouts[len(layouts) // 2], layouts[-1]]
    duration = args.duration or (2.0 if args.quick else 5.0)

    t_start = time.perf_counter()
    scaling = (
        []
        if args.skip_scaling
        else run_scaling(
            scale_names, thread_counts, args.quick, args.min_time, args.pin_scaling, topo
        )
    )
    jobs = (
        []
        if args.skip_jobs
        else run_jobs(
            job_names,
            layouts,
            args.quick,
            duration,
            not args.no_pin_jobs,
            topo,
            info["MemAvailable"],
        )
    )

    notes = observations(info, scaling, jobs)
    if notes:
        print("\n== Observations ==")
        for n in notes:
            print(f"  - {n}")
    print(f"\n({time.perf_counter() - t_start:.0f} s)")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "host": info,
                    "quick": args.quick,
                    "scaling": scaling,
                    "jobs": jobs,
                    "args": vars(args),
                    "date": time.strftime("%Y-%m-%d %H:%M"),
                },
                fh,
                indent=1,
            )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
