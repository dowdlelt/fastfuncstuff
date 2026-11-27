"""Correlation, partial-correlation and shrunk-correlation matrices from cleaned series.

Everything starts from :func:`unit_rows`: each series centred and scaled to unit
length, after which a correlation is a dot product and a whole matrix is one GEMM.
With per-TR weights the same holds for the *weighted* Pearson r (centre on the
weighted mean, scale by sqrt(w)) -- that is AFNI's ``-weight_corr`` and, with a
condition's block regressor as the weight, a task-weighted connectivity.

Matrices are small (ROIs squared) and computed in float64 wherever the device can
hold it; the inverse behind a partial correlation always is.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff.utils import full_fp32_matmul, linalg_device

#: AFNI's Fisher-z ceiling: atanh is capped at 4 (|r| = 0.999329). Without a cap
#: the diagonal is infinite; past it every r is "identical" for any practical use.
FZ_CAP = 4.0


def fisher_z(r: Tensor | np.ndarray, cap: float = FZ_CAP):
    """``atanh(r)`` with ``|z|`` capped at ``cap`` (AFNI's BOBatanh)."""
    lim = math.tanh(cap)
    if isinstance(r, Tensor):
        return torch.atanh(r.clamp(-lim, lim))
    return np.arctanh(np.clip(r, -lim, lim))


@full_fp32_matmul()
def unit_rows(
    x: Tensor,
    weights: Tensor | np.ndarray | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Centre each row and scale it to unit length; ``(rows, valid)``.

    ``weights`` (``(T,)``, non-negative) gives the weighted version: rows are centred
    on their weighted mean and multiplied by ``sqrt(w)``, so ``a @ b`` is the weighted
    Pearson r. Constant rows (nothing to correlate) come back as zeros with
    ``valid = False`` rather than as NaN, so they correlate 0 with everything.
    A float64 request on a device without float64 (Metal) is served on the CPU, and
    the rows come back there: ROI-by-time series are small, and the matrices built
    from them are meant to be float64.
    """
    dtype = dtype or x.dtype
    if dtype == torch.float64:
        x = x.to(linalg_device(x.device))  # device before dtype: MPS float64 casts fail
    x = x.to(dtype)
    if weights is not None:
        w = torch.as_tensor(np.asarray(weights) if not isinstance(weights, Tensor) else weights)
        w = w.to(device=x.device, dtype=dtype).reshape(-1)
        if w.numel() != x.shape[1]:
            raise ValueError(f"{w.numel()} weights for {x.shape[1]} time points")
        if bool((w < 0).any()) or not float(w.sum()) > 0:
            raise ValueError("weights must be non-negative and not all zero")
        mean = (x @ w) / w.sum()
        xc = (x - mean[:, None]) * w.sqrt()
    else:
        xc = x - x.mean(dim=1, keepdim=True)
    norm = xc.norm(dim=1, keepdim=True)
    scale = norm.abs().amax().clamp_min(1e-30)
    valid = (norm > 1e-7 * scale).reshape(-1)
    xc = xc / norm.clamp_min(1e-30)
    xc[~valid] = 0.0
    return xc, valid


@full_fp32_matmul()
def correlation_matrix(
    x: Tensor,
    weights: Tensor | np.ndarray | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """``(N, T)`` -> ``(N, N)`` Pearson (or weighted Pearson) r.

    The diagonal is 1 for every row with variance; a constant row is 0 throughout
    (AFNI's convention for an empty ROI).
    """
    xn, valid = unit_rows(x, weights, dtype=dtype)
    r = (xn @ xn.T).clamp_(-1.0, 1.0)
    r.fill_diagonal_(0.0)
    r.diagonal().copy_(valid.to(r.dtype))
    return r


def partial_correlation(
    r: Tensor | np.ndarray, dof: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Partial correlation and partial-regression ("PC beta") matrices from ``r``.

    With ``M = inv(r)``: ``pc_ij = -M_ij / sqrt(M_ii M_jj)`` and
    ``pcb_ij = -M_ij / M_ii`` (3dNetCorr ``-part_corr``). The diagonals are set to 1,
    the correlation / regression of a series with itself; AFNI's formula leaves -1
    there. ``r`` must be well conditioned: with as many ROIs as residual degrees of
    freedom it is singular, and a shrunk matrix (:func:`ledoit_wolf`) is the input to
    use. ``dof`` (the residual degrees of freedom of the series behind ``r``) makes
    that a rank test rather than an eigenvalue threshold: N series in a dof-dimensional
    space have rank at most dof, and a float32 pipeline can leave the missing
    eigenvalues at 1e-6 instead of 0, where no threshold separates them from real ones.
    """
    from fastfuncstuff import policy_linalg as plinalg

    rr = torch.as_tensor(np.asarray(r) if not isinstance(r, Tensor) else r)
    rr = rr.detach().to("cpu").to(torch.float64)
    n = rr.shape[0]
    if dof is not None and n >= dof:
        raise ValueError(
            f"{n} ROIs from {dof} residual degrees of freedom: the correlation matrix has "
            f"rank at most {dof - 1}, so no partial correlation exists. Use a shrunk "
            "estimate (-shrink lw), fewer ROIs, or less nuisance (a narrower stopband)."
        )
    evals = plinalg.eigvalsh(rr)
    if not float(evals[0]) > 1e-10 * max(float(evals[-1]), 1e-30):
        raise ValueError(
            f"the {n}x{n} correlation matrix is singular (smallest eigenvalue "
            f"{float(evals[0]):.2e}): more ROIs than degrees of freedom, or ROIs that are "
            "linear combinations of others. Use a shrunk estimate (-shrink lw)."
        )
    m = plinalg.inv(rr)
    d = m.diagonal()
    pc = -m / torch.sqrt(d[:, None] * d[None, :])
    pcb = -m / d[:, None]
    pc.fill_diagonal_(1.0)
    pcb.fill_diagonal_(1.0)
    return pc.numpy(), pcb.numpy()


@full_fp32_matmul()
def ledoit_wolf(
    x: Tensor, n_effective: int | None = None, dtype: torch.dtype = torch.float64
) -> tuple[Tensor, float]:
    """Ledoit-Wolf shrunk correlation of ``(N, T)`` series; ``(R_shrunk, shrinkage)``.

    The series are standardised first, so this shrinks the *correlation* matrix toward
    the identity with the analytic optimal intensity (Ledoit & Wolf 2004, as
    scikit-learn computes it with ``n = T`` samples). The result is positive definite
    whenever the shrinkage is above 0, which is what makes a partial correlation
    possible with more ROIs than time points.

    ``n_effective`` counts the sampling noise in independent samples rather than time
    points. Cleaned series are not T independent samples: a 0.01-0.1 Hz passband at
    TR 1 s leaves 114 residual dof in 720 TRs, and with ``n = T`` the analytic
    intensity comes out ~6x too small (0.009 vs 0.058 on TASKFORCE rest, 360 Glasser
    ROIs; the dof-based estimate also predicted the held-out run's matrix better).
    The sampling-variance term is scaled by ``T / n_effective``; None = scikit-learn.
    """
    xn, _ = unit_rows(x, dtype=dtype)
    n_samples = x.shape[1]
    z = xn.T * math.sqrt(n_samples)  # (T, N), unit population variance per column
    p = z.shape[1]
    s = (z.T @ z) / n_samples
    z2 = z * z
    var = z2.sum(0) / n_samples
    mu = float(var.sum()) / p
    beta_ = float((z2.T @ z2).sum())
    delta_ = float((s * s).sum())
    beta = (beta_ / n_samples - delta_) / (p * n_samples)
    if n_effective is not None:
        beta *= n_samples / max(int(n_effective), 1)
    delta = (delta_ - 2.0 * mu * float(var.sum()) + p * mu * mu) / p
    beta = min(beta, delta)
    shrink = 0.0 if beta == 0 else beta / delta
    shrunk = (1.0 - shrink) * s
    shrunk.diagonal().add_(shrink * mu)
    d = shrunk.diagonal().clamp_min(1e-30).sqrt()
    return shrunk / (d[:, None] * d[None, :]), float(shrink)
