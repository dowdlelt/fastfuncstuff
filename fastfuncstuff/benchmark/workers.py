"""Run several benchmark workers side by side, each pinned to its own cores.

On a core-rich host (a 32-CPU server) one worker at a time leaves most of the machine idle,
but letting every tool see every core makes timings depend on whatever else is running.
Instead each worker gets ``threads`` physical cores of its own (with their SMT siblings, so
no other worker lands on them) and every tool it starts — AFNI, ffs, MATLAB, FSL — is told
to use exactly that many threads. All workers share the one worklist: claiming a unit is an
atomic transaction, so two workers never take the same unit.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
from pathlib import Path

# Every thread-count knob a reference or ffs reads. FFS_NUM_THREADS outranks OMP_NUM_THREADS
# inside ffs (utils.thread_budget), so both are set to the same value.
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "FFS_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",  # ANTs
)


def allowed_cpus() -> set[int]:
    """Logical CPUs this process may run on. macOS has no affinity mask (and Apple Silicon
    no SMT), so there it is every CPU."""
    if hasattr(os, "sched_getaffinity"):
        return set(os.sched_getaffinity(0))
    return set(range(os.cpu_count() or 1))


def pin_to(cpus: list[int]) -> None:
    """Pin this process to ``cpus`` where the OS allows it (not macOS: there the thread
    variables alone bound each worker)."""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)


def cpu_cores(
    cpus: set[int] | None = None, sysfs: str = "/sys/devices/system/cpu"
) -> list[list[int]]:
    """Allowed logical CPUs grouped by physical core, in core order.

    Reads each CPU's SMT siblings from sysfs; a CPU whose topology cannot be read is its
    own core."""
    allowed = sorted(cpus if cpus is not None else allowed_cpus())
    groups: dict[int, list[int]] = {}
    for c in allowed:
        sib = [c]
        for name in ("core_cpus_list", "thread_siblings_list"):
            try:
                sib = _parse_cpu_list(Path(f"{sysfs}/cpu{c}/topology/{name}").read_text())
                break
            except OSError:
                continue
        groups.setdefault(min(sib), []).append(c)
    return [groups[k] for k in sorted(groups)]


def _parse_cpu_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def default_workers(cores: list[list[int]], threads: int, quota: float | None) -> int:
    """How many ``threads``-wide workers fit in the cores and the container's CPU quota."""
    usable = len(cores) if quota is None else min(len(cores), math.floor(quota))
    return max(1, usable // threads)


def core_slots(cores: list[list[int]], workers: int, threads: int) -> list[list[int]]:
    """Disjoint logical-CPU sets, ``threads`` physical cores (plus siblings) per worker."""
    if workers * threads > len(cores):
        raise ValueError(
            f"{workers} workers x {threads} threads needs {workers * threads} physical cores; "
            f"{len(cores)} are available"
        )
    return [
        sorted(c for core in cores[i * threads : (i + 1) * threads] for c in core)
        for i in range(workers)
    ]


def thread_env(threads: int) -> dict[str, str]:
    return {v: str(threads) for v in THREAD_VARS}


def launch(argv: list[str], workers: int | None, threads: int, device: str, max_units: int) -> int:
    """Start the workers as child processes of this one and wait for them all.

    ``argv`` is the ``-run`` command line minus the pool flags; each child gets it back with
    ``-workers 1`` and its share of ``max_units``. Output lines are prefixed ``wN``."""
    from ..utils import cgroup_cpu_quota

    if device.startswith(("cuda", "mps")) and (workers or 0) > 1:
        raise SystemExit(
            "-workers > 1 needs -device cpu: workers sharing one GPU would time each other"
        )
    cores = cpu_cores()
    quota = cgroup_cpu_quota()
    n = workers or default_workers(cores, threads, quota)
    if quota is not None and n * threads > quota:
        raise SystemExit(f"{n} x {threads} threads exceeds the CPU quota of {quota:g}")
    slots = core_slots(cores, n, threads)
    share = math.ceil(max_units / n) if max_units < 10**9 else max_units
    # The MATLAB memory guard is per process; split the host's budget between workers.
    matlab_gb = float(os.environ.get("FFS_MATLAB_MAX_GB") or 160) / n
    print(
        f"{n} workers x {threads} threads (quota {quota or 'none'}, {len(cores)} cores)", flush=True
    )

    procs = []
    for i, cpus in enumerate(slots):
        env = {
            **os.environ,
            **thread_env(threads),
            "FFS_MATLAB_MAX_GB": f"{matlab_gb:g}",
            "FFS_BENCH_WORKER": f"{i}/{n}",
            "FFS_BENCH_CPUS": ",".join(map(str, cpus)),
            "PYTHONUNBUFFERED": "1",
        }
        cmd = [sys.executable, "-m", "fastfuncstuff.benchmark.work", *argv,
               "-workers", "1", "-threads", str(threads), "-max_units", str(share)]  # fmt: skip
        p = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )  # the child pins itself to FFS_BENCH_CPUS (preexec_fn is unsafe beside threads)
        print(f"w{i}: pid {p.pid} cpus {env['FFS_BENCH_CPUS']}", flush=True)
        t = threading.Thread(target=_relay, args=(p, f"w{i} "), daemon=True)
        t.start()
        procs.append((p, t))
    rc = 0
    for p, t in procs:
        rc = p.wait() or rc
        t.join()
    return rc


def _relay(p: subprocess.Popen, prefix: str) -> None:
    assert p.stdout is not None
    for line in p.stdout:
        sys.stdout.write(prefix + line)
        sys.stdout.flush()
