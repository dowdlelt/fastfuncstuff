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
Each matrix still gets the same LAPACK call, so the answer is bit-identical; see
:func:`_cpu_batch_split`. ``scripts/bench_cpu_host.py`` measures the gap on a host.
"""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
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
# torch.unique(dim=0) is not a per-matrix op: splitting it would dedupe per chunk.
_NEVER_SPLIT = {"unique_rows"}
# Each lane must get a few matrices, or the split is all overhead.
_MIN_PER_LANE = 4

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
            _pool = ThreadPoolExecutor(lanes, thread_name_prefix="ffs-linalg")
            _pool_key = key
        assert _pool is not None
        return _pool


def _join(parts: list, batch: torch.Size):
    if not isinstance(parts[0], torch.Tensor):
        return parts[0]
    if any(p.numel() == 0 for p in parts):
        # lstsq leaves residuals / singular values empty unless its driver and
        # every matrix qualify; a chunk that didn't qualify empties them all.
        return parts[0] if all(p.numel() == 0 for p in parts) else parts[0].new_empty(0)
    joined = torch.cat(parts)
    return joined.reshape(*batch, *joined.shape[1:])


def _cpu_batch_split(op: str, fn, args: tuple, kwargs: dict):
    """``fn(*args, **kwargs)`` with the batch dealt across ``torch.get_num_threads()``
    lanes, or ``None`` when the call should simply run as it is.

    It declines whenever splitting could change anything but the speed: while
    tracing, under autograd, inside one of its own lanes (no nested pool), with a
    tensor keyword argument, or when an operand broadcasts in a way the split
    cannot mirror. An unbatched operand -- a single right-hand side against a batch
    of matrices -- is shared by every lane. If a lane raises (a singular matrix),
    the whole call is redone unsplit so the error names the caller's batch index.
    """
    if op in _NEVER_SPLIT or os.environ.get("FFS_LINALG_SPLIT", "1") == "0":
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
    lanes = min(lanes, n // _MIN_PER_LANE)
    min_work = _SPLIT_MIN_WORK.get(op, _SPLIT_MIN_WORK_DEFAULT)
    if lanes < 2 or n * max(primary.shape[-2:]) ** 3 < min_work:
        return None

    flat: list = []
    for a in args:
        if not isinstance(a, torch.Tensor):
            flat.append((False, a))
        elif a.shape[:nb] == batch and (
            a.ndim == primary.ndim or (a.ndim == primary.ndim - 1 and a.shape == primary.shape[:-1])
        ):
            flat.append((True, a.reshape(n, *a.shape[nb:])))
        elif a.ndim <= 2:
            flat.append((False, a))
        else:
            return None

    bounds = [round(i * n / lanes) for i in range(lanes + 1)]
    chunks = [[a[lo:hi] if split else a for split, a in flat] for lo, hi in pairwise(bounds)]
    inference = torch.is_inference_mode_enabled()

    def lane(chunk):
        _in_lane.active = True
        try:
            with torch.inference_mode(inference):
                return fn(*chunk, **kwargs)
        finally:
            _in_lane.active = False

    try:
        parts = list(_lane_pool(torch.get_num_threads()).map(lane, chunks))
    except RuntimeError:
        return None
    if isinstance(parts[0], torch.Tensor):
        return _join(parts, batch)
    # torch's return_types are structseqs: they take one sequence, not *args.
    return type(parts[0])([_join([p[i] for p in parts], batch) for i in range(len(parts[0]))])


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
