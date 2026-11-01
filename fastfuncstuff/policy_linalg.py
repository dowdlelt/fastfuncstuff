"""Dense linear algebra that runs where :data:`utils._MPS_CPU_OPS` says it should.

Use these instead of the ``torch.linalg`` solvers. Off Metal each one is the torch
function behind a single device check. On Metal the matrices go to the CPU and the
answer comes back, because at every shape in ``scripts/bench_mps_policy.py`` the CPU
wins, by 800x for a (300,12) QR and 200x for a batched triangular solve, and a batched
SVD does not run at all. Batched ``eigh`` is the one that stays, at 3x.

A call site cannot know which of those it is in, so this is enforced rather than
remembered: ``tests/test_mps_cast_order.py`` forbids the bare torch names. Norms and
``matrix_exp`` are not here; they are elementwise work Metal is good at.
"""

from __future__ import annotations

import torch

from fastfuncstuff.utils import cpu_if_mps, linalg_lstsq


def _routed(op: str, fn, *args, **kwargs):
    """``fn(*args, **kwargs)`` on the device *op* belongs on, returned to the input's."""
    home = next(a.device for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor))
    work = cpu_if_mps(home, op)
    if work == home:
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
