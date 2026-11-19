"""Dense linear algebra that runs where :data:`utils._MPS_CPU_OPS` says it should.

Use these instead of the ``torch.linalg`` solvers. Off Metal each one is the torch
function behind a single device check. On Metal the matrices go to the CPU and the
answer comes back, because at every shape in ``scripts/bench_mps_policy.py`` the CPU
wins, by 800x for a (300,12) QR and 200x for a batched triangular solve, and a batched
SVD does not run at all. Batched ``eigh`` is the one that stays, at 3x.

A call site cannot know which of those it is in, so this is enforced rather than
remembered: ``tests/test_mps_cast_order.py`` forbids the bare torch names. Norms and
``matrix_exp`` are not here; they are elementwise work Metal is good at.

On the CPU a *batched* call is also dealt across threads here. ATen runs batched
LAPACK work -- every factorisation and solve in this module -- largely one matrix at
a time, whatever ``torch.set_num_threads`` says: on six cores a batched solve ran
1.5x its one-thread speed and a batched ``eigh`` 0.6x, against 6.4x and 3.4x once
the batch was split over six Python threads (ATen drops the GIL inside the call).
Lanes run MKL single-threaded (otherwise each opens its own OpenMP team and a
six-thread job ran 36), so a split call is bit-identical to the *one-thread*
native call. Against multi-threaded native it is too for small matrices; from
k ~ 64, where MKL threads inside a single matrix, the two differ at ~1e-15
relative -- the same as changing the thread count. See :func:`_cpu_batch_split`;
``scripts/bench_cpu_host.py`` measures the gap on a host.
"""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import pairwise

import torch

from fastfuncstuff.utils import cpu_if_mps, linalg_lstsq

# Below this much work (batch x k^3) the pool's dispatch costs more than the
# threads save; measured break-evens on a 6-core Xeon, float64, k = 3..100. The
# iterative decompositions win early (eigh/svd/lstsq 1.5-2x at 5e5, 4-5x by
# 1e7). Solves and Cholesky are a few k^3/3 flops per matrix and ATen already
# threads some of them, so they need 3e7 to come out ahead, 1.5-4x past it.
_SPLIT_MIN_WORK = {"eigh_batched": 5e5, "svd_batched": 5e5, "pinv": 5e5, "lstsq": 5e5, "qr": 1e7}
_SPLIT_MIN_WORK_DEFAULT = 3e7
# torch 2.14.1's CPU batched LU (getrf: solve, inv, det, lu_*) hangs, after MKL
# prints "Parameter 6 was incorrect on entry to DLASWP", once k > 128 and more
# than one thread runs -- even for a batch of two, float32 or float64. Unbatched
# getrf does not, so those calls always loop one matrix at a time in a lane.
_LU_OPS = {"solve", "inv", "det", "lu"}
_LU_HANG_K = 128
# torch.unique(dim=0) is not a per-matrix op: splitting it would dedupe per chunk.
_NEVER_SPLIT = {"unique_rows"}
# Each lane must get a few matrices, or the split is all overhead.
_MIN_PER_LANE = 4

_lane_thread_setters: list | None = None


def _thread_local_setters() -> list:
    """Per-thread thread-count setters for the OpenMP runtime and MKL torch loaded.

    ``torch.set_num_threads`` is process-wide, so a lane cannot use it without
    shrinking every other caller. But OpenMP's ``nthreads-var`` and MKL's
    ``mkl_set_num_threads_local`` belong to the calling thread. Found by path
    from ``/proc/self/maps`` so it is the copy torch actually uses; on a host
    without it (macOS) the list is empty and lanes run as they are.
    """
    global _lane_thread_setters
    if _lane_thread_setters is not None:
        return _lane_thread_setters
    import ctypes
    import re

    setters = []
    try:
        with open("/proc/self/maps") as fh:
            maps = fh.read()
    except OSError:
        maps = ""
    for path in sorted(set(re.findall(r"(/\S*lib(?:gomp|iomp5|omp)[\w.-]*\.so[\w.]*)", maps))):
        try:
            setters.append(
                ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LAZY).omp_set_num_threads
            )
        except (OSError, AttributeError):
            pass
    for path in sorted(
        set(re.findall(r"(/\S*lib(?:torch_cpu|mkl_rt|mkl_core)[\w.-]*\.so[\w.]*)", maps))
    ):
        try:
            # The C entry point; lowercase mkl_set_num_threads_local is the
            # Fortran binding, which takes a pointer and segfaults on an int.
            setters.append(
                ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_LAZY).MKL_Set_Num_Threads_Local
            )
        except (OSError, AttributeError):
            pass
    _lane_thread_setters = setters
    return setters


def _init_lane() -> None:
    """Make a pool thread a single-threaded lane, for good.

    Without this every lane's ATen/MKL call opened its own OpenMP team: six lanes
    on six cores peaked at 36 running threads and 8.8 CPUs of time -- harmless on
    an idle desktop, but under a container CPU quota it is the over-subscription
    that freezes the whole process each period.
    """
    _in_lane.active = True
    for set_threads in _thread_local_setters():
        set_threads(1)


_pool: ThreadPoolExecutor | None = None
_pool_key: tuple[int, int] | None = None
_pool_lock = threading.Lock()
_in_lane = threading.local()


def _lane_pool(lanes: int) -> ThreadPoolExecutor:
    """One pool per (process, lane count). A forked child inherits the pool object
    but not its threads, so it builds its own rather than queue work to nobody."""
    global _pool, _pool_key
    key = (os.getpid(), lanes)
    with _pool_lock:
        if _pool_key != key:
            if _pool is not None and _pool_key is not None and _pool_key[0] == key[0]:
                _pool.shutdown(wait=False)
            _pool = ThreadPoolExecutor(
                lanes, thread_name_prefix="ffs-linalg", initializer=_init_lane
            )
            _pool_key = key
        assert _pool is not None
        return _pool


@contextmanager
def _modes(inference: bool, grad: bool):
    with torch.inference_mode(inference), torch.enable_grad() if grad else torch.no_grad():
        yield


def _gather(parts: list, bounds: list[int], batch: torch.Size, pool: ThreadPoolExecutor, modes):
    """Reassemble the lanes' results in batch order, copied by the lanes themselves.

    Not ``torch.cat`` on the calling thread: that opens an OpenMP region there,
    and libgomp then spins the caller's idle team for ~10 ms -- across a stream
    of short calls it held CPU/wall at 8.8 on a six-thread budget, against 5.3
    when the lanes do the copy (they are single-threaded, so no team wakes).
    """
    single = isinstance(parts[0], torch.Tensor)
    rows = [[p] if single else list(p) for p in parts]
    n = bounds[-1]
    outs: list = []
    filled: list[bool] = []
    for j in range(len(rows[0])):
        items = [r[j] for r in rows]
        first = items[0]
        if not isinstance(first, torch.Tensor):
            outs.append(first)
            filled.append(False)
        elif any(t.numel() == 0 for t in items):
            # lstsq leaves residuals / singular values empty unless its driver and
            # every matrix qualify; a chunk that didn't qualify empties them all.
            outs.append(first if all(t.numel() == 0 for t in items) else first.new_empty(0))
            filled.append(False)
        else:
            outs.append(first.new_empty((n, *first.shape[1:])))
            filled.append(True)

    def copy(i: int) -> None:
        with modes():
            for j, out in enumerate(outs):
                if filled[j]:
                    out[bounds[i] : bounds[i + 1]].copy_(rows[i][j])

    list(pool.map(copy, range(len(parts))))
    outs = [o.reshape(*batch, *o.shape[1:]) if f else o for o, f in zip(outs, filled, strict=True)]
    # torch's return_types are structseqs: they take one sequence, not *args.
    return outs[0] if single else type(parts[0])(outs)


def _cpu_batch_split(op: str, fn, args: tuple, kwargs: dict):
    """``fn(*args, **kwargs)`` with the batch dealt across ``torch.get_num_threads()``
    lanes, or ``None`` when the call should simply run as it is.

    It declines whenever splitting could change anything but the speed: while
    tracing, under autograd, inside one of its own lanes (no nested pool), with a
    tensor keyword argument, or when an operand broadcasts in a way the split
    cannot mirror. An unbatched operand -- a single right-hand side against a batch
    of matrices -- is shared by every lane. If a lane raises (a singular matrix),
    the whole call is redone unsplit so the error names the caller's batch index.

    Batched LU past ``_LU_HANG_K`` always goes through the lanes, whatever the
    batch size, threshold or ``FFS_LINALG_SPLIT``: there it is the hang guard,
    not an optimisation.
    """
    if op in _NEVER_SPLIT:
        return None
    if getattr(_in_lane, "active", False) or torch.compiler.is_compiling():
        return None
    lanes = torch.get_num_threads()
    if lanes < 2:
        return None
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    if not tensors or any(isinstance(v, torch.Tensor) for v in kwargs.values()):
        return None
    if any(t.requires_grad for t in tensors) and torch.is_grad_enabled():
        return None
    primary = max(tensors, key=lambda t: t.ndim)
    if primary.ndim < 3:
        return None
    batch = primary.shape[:-2]
    nb = len(batch)
    n = math.prod(batch)
    k = max(primary.shape[-2:])
    lu_guard = op in _LU_OPS and k > _LU_HANG_K
    if lu_guard:
        # One lane, looping: concurrent large-k getrf from several lanes is safe
        # but contends inside MKL -- 96 x (200,200) took 1.48 s over six lanes
        # against 0.042 s in one (0.026 s for single-threaded native).
        lanes = 1
    else:
        if os.environ.get("FFS_LINALG_SPLIT", "1") == "0":
            return None
        lanes = min(lanes, n // _MIN_PER_LANE)
        if lanes < 2 or n * k**3 < _SPLIT_MIN_WORK.get(op, _SPLIT_MIN_WORK_DEFAULT):
            return None

    flat: list = []
    for a in args:
        if not isinstance(a, torch.Tensor):
            flat.append((False, a))
        elif a.shape[:nb] == batch and (
            a.ndim == primary.ndim or (a.ndim == primary.ndim - 1 and a.shape == primary.shape[:-1])
        ):
            flat.append((True, a.reshape(n, *a.shape[nb:])))
        elif a.ndim == primary.ndim:
            # Broadcast batch dims, e.g. one right-hand side per row of a grid.
            try:
                flat.append((True, a.expand(*batch, *a.shape[nb:]).reshape(n, *a.shape[nb:])))
            except RuntimeError:
                return None  # torch itself will raise the shape error
        elif a.ndim <= 2:
            flat.append((False, a))
        else:
            return None

    # Grad and inference mode are thread-local: carry the caller's into the lanes.
    inference, grad = torch.is_inference_mode_enabled(), torch.is_grad_enabled()

    def modes():
        return _modes(inference, grad)

    def lane(job):
        offset, chunk = job
        with modes():
            if not lu_guard:
                return fn(*chunk, **kwargs)
            # One matrix per call: ATen's *batched* LU is the path that hangs.
            m = next(c.shape[0] for (split, _), c in zip(flat, chunk, strict=True) if split)
            outs = []
            for i in range(m):
                one = [c[i] if split else c for (split, _), c in zip(flat, chunk, strict=True)]
                try:
                    outs.append(fn(*one, **kwargs))
                except RuntimeError as exc:
                    raise type(exc)(f"(Batch element {offset + i}): {exc}") from exc
            if isinstance(outs[0], torch.Tensor):
                return torch.stack(outs)
            return type(outs[0])([torch.stack(col) for col in zip(*outs, strict=True)])

    bounds = [round(i * n / lanes) for i in range(lanes + 1)]
    jobs = [(lo, [a[lo:hi] if split else a for split, a in flat]) for lo, hi in pairwise(bounds)]
    pool = _lane_pool(torch.get_num_threads())
    try:
        parts = list(pool.map(lane, jobs))
    except RuntimeError:
        if lu_guard:  # already names the caller's batch index; a batched redo could hang
            raise
        return None  # redone unsplit by the caller, so the error names its batch index
    with modes():
        return _gather(parts, bounds, batch, pool, modes)


def _routed(op: str, fn, *args, **kwargs):
    """``fn(*args, **kwargs)`` on the device *op* belongs on, returned to the input's."""
    devices = [a.device for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)]
    # Not just the first: on a Mac a float64-only helper hands back a CPU tensor, so
    # solve(b_cpu, L_mps) is a normal thing to arrive here, and Metal is where it goes home to.
    home = next((d for d in devices if d.type == "mps"), devices[0])
    work = cpu_if_mps(home, op)
    if work == home:
        if home.type == "cpu":
            split = _cpu_batch_split(op, fn, args, kwargs)
            if split is not None:
                return split
        return fn(*args, **kwargs)
    moved = [a.to(work) if isinstance(a, torch.Tensor) else a for a in args]
    named = {k: v.to(work) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
    result = fn(*moved, **named)
    if isinstance(result, torch.Tensor):
        return result.to(home)
    # torch's return_types are structseqs: they take one sequence, not *args.
    return type(result)([x.to(home) if isinstance(x, torch.Tensor) else x for x in result])


def qr(a, *args, **kwargs):
    return _routed("qr", torch.linalg.qr, a, *args, **kwargs)


def svd(a, *args, **kwargs):
    op = "svd_batched" if a.ndim > 2 else "svd"
    return _routed(op, torch.linalg.svd, a, *args, **kwargs)


def svdvals(a, *args, **kwargs):
    op = "svd_batched" if a.ndim > 2 else "svdvals"
    return _routed(op, torch.linalg.svdvals, a, *args, **kwargs)


def matrix_rank(a, *args, **kwargs):
    op = "svd_batched" if a.ndim > 2 else "svd"
    return _routed(op, torch.linalg.matrix_rank, a, *args, **kwargs)


def cond(a, *args, **kwargs):
    op = "svd_batched" if a.ndim > 2 else "svd"
    return _routed(op, torch.linalg.cond, a, *args, **kwargs)


def pinv(a, *args, **kwargs):
    return _routed("pinv", torch.linalg.pinv, a, *args, **kwargs)


def lstsq(a, b, **kwargs):
    if a.device.type == "cpu" and b.device.type == "cpu":
        split = _cpu_batch_split("lstsq", torch.linalg.lstsq, (a, b), kwargs)
        if split is not None:
            return split
    return linalg_lstsq(a, b, **kwargs)


def solve(a, b, **kwargs):
    return _routed("solve", torch.linalg.solve, a, b, **kwargs)


def solve_triangular(a, b, **kwargs):
    return _routed("solve_triangular", torch.linalg.solve_triangular, a, b, **kwargs)


def cholesky(a, **kwargs):
    return _routed("cholesky", torch.linalg.cholesky, a, **kwargs)


def cholesky_ex(a, **kwargs):
    return _routed("cholesky", torch.linalg.cholesky_ex, a, **kwargs)


def cholesky_solve(b, l, **kwargs):  # noqa: E741 -- torch's own name for the factor
    return _routed("cholesky_solve", torch.cholesky_solve, b, l, **kwargs)


def cholesky_inverse(l, **kwargs):  # noqa: E741 -- torch's own name for the factor
    return _routed("cholesky_inverse", torch.cholesky_inverse, l, **kwargs)


def inv(a, **kwargs):
    return _routed("inv", torch.linalg.inv, a, **kwargs)


def inv_ex(a, **kwargs):
    return _routed("inv", torch.linalg.inv_ex, a, **kwargs)


def det(a):
    return _routed("det", torch.linalg.det, a)


def slogdet(a):
    return _routed("det", torch.linalg.slogdet, a)


def logdet(a):
    return _routed("det", torch.logdet, a)


def lu_factor(a, **kwargs):
    return _routed("lu", torch.linalg.lu_factor, a, **kwargs)


def lu_solve(lu, pivots, b, **kwargs):
    return _routed("lu", torch.linalg.lu_solve, lu, pivots, b, **kwargs)


def eigh(a, *args, **kwargs):
    op = "eigh_batched" if a.ndim > 2 else "eigh"
    return _routed(op, torch.linalg.eigh, a, *args, **kwargs)


def eigvalsh(a, *args, **kwargs):
    op = "eigh_batched" if a.ndim > 2 else "eigh"
    return _routed(op, torch.linalg.eigvalsh, a, *args, **kwargs)


def unique_rows(x, **kwargs):
    """``torch.unique(x, dim=0, ...)``. The plain 1-D unique is 5x faster on Metal; this is 770x slower."""
    return _routed("unique_rows", torch.unique, x, dim=0, **kwargs)
