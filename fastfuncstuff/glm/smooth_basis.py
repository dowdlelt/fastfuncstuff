"""Roughness-penalized FIR / TENT fits with a per-voxel smoothing strength.

The knot values ``beta`` of a FIR/TENT response are estimated by

    min ||y - X beta - N gamma||^2 + lam * ||D beta||^2

where ``D`` is the ``order``-th difference across neighbouring knots of each
condition (no penalty across conditions) and ``N`` is the unpenalized
nuisance (drift polynomials, -ortvec).  This is the smooth FIR of Goutte,
Nielsen & Hansen (2000) / Marrelec et al. (2003).  The penalty is largest on
exactly the knot pattern mid-TR onsets cannot see (+,-,+,-), leaves straight
lines free, and makes grids finer than the TR solvable at all.

``lam`` is chosen per voxel without held-out data, by REML (the default --
Wood 2011 finds it avoids GCV's occasional severe undersmoothing) or GCV.
Both are closed form on a lambda grid after ONE simultaneous diagonalization
of the task Gram matrix and the penalty, shared by every voxel:

    B = A + P = R'R,   R^-T P R^-1 = V diag(s) V',   0 <= s <= 1
    beta(lam) = W diag(d) z,   W = R^-1 V,   z = (X W)' y,   d = 1/((1-s) + lam s)

so each lambda costs O(K) per voxel.  Working from ``A + P`` rather than
``A`` keeps this valid when ``A`` alone is singular (knots finer than the
timing resolves): the penalty supplies what the data cannot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import block_diag
from tqdm.auto import tqdm

from fastfuncstuff.glm.xval import cod_from_ss_residual
from fastfuncstuff.memory import estimate_chunk_size

#: log10 lambda grid.  lambda is relative (penalty scaled to the Gram trace),
#: so the same grid spans "no smoothing" to "straight lines" for any design.
DEFAULT_LOG10_GRID = np.linspace(-5.0, 7.0, 49)
SELECTION_METHODS = ("reml", "gcv", "fixed")


def roughness_penalty(
    n_basis_per_condition: list[int], order: int = 2, zero_edges: bool = False
) -> np.ndarray:
    """Block-diagonal ``D'D`` over each condition's knots.

    ``zero_edges`` (TENTzero/CSPLINzero) differences the FULL knot vector,
    whose pinned-zero edge knots are dropped from the design, so the
    response is also kept smooth into its zero ends.
    """
    blocks = []
    for n in n_basis_per_condition:
        full = n + 2 if zero_edges else n
        if full <= order:
            blocks.append(np.zeros((n, n)))
            continue
        d = np.diff(np.eye(full), order, axis=0)
        if zero_edges:
            d = d[:, 1:-1]
        blocks.append(d.T @ d)
    return block_diag(*blocks)


@dataclass
class SmoothBasisFit:
    betas: torch.Tensor  # (V, K) task knot values
    lam: torch.Tensor  # (V,) chosen relative lambda
    edf: torch.Tensor  # (V,) effective degrees of freedom of the task block
    r2: torch.Tensor  # (V,) in-sample R^2, same definition as fit_glm
    method: str


@dataclass
class _Spectrum:
    g: np.ndarray  # (T, K) X_tilde @ W
    w: np.ndarray  # (K, K)
    s: np.ndarray  # (K,)
    q_nuis: np.ndarray  # (T, p) orthonormal nuisance basis
    n_eff: int  # T - rank(nuisance)
    rank_penalty: int


def _spectrum(design: np.ndarray, n_task: int, penalty: np.ndarray) -> _Spectrum:
    x = design[:, :n_task]
    nuis = design[:, n_task:]
    nuis = nuis[:, np.abs(nuis).sum(axis=0) > 0]
    if nuis.shape[1]:
        u, sv, _ = np.linalg.svd(nuis, full_matrices=False)
        q = u[:, sv > sv.max() * 1e-10]
    else:
        q = np.zeros((design.shape[0], 0))
    x = x - q @ (q.T @ x)
    gram = x.T @ x
    # Relative lambda: the penalty's scale matches the data term's.
    pen = penalty * (np.trace(gram) / max(np.trace(penalty), 1e-300))
    b = gram + pen
    b += np.eye(n_task) * (1e-10 * np.trace(b) / n_task)  # null(A) & null(P) guard
    r = np.linalg.cholesky(b).T  # b = r' r
    r_inv = np.linalg.inv(r)
    m = r_inv.T @ pen @ r_inv
    s, v = np.linalg.eigh((m + m.T) / 2)
    s = np.clip(s, 0.0, 1.0)
    w = r_inv @ v
    p_evals = np.linalg.eigvalsh(pen)
    rank_p = int((p_evals > p_evals.max() * 1e-9).sum()) if p_evals.size else 0
    return _Spectrum(
        g=x @ w,
        w=w,
        s=s,
        q_nuis=q,
        n_eff=design.shape[0] - q.shape[1],
        rank_penalty=rank_p,
    )


def _criterion(
    method: str,
    z2: torch.Tensor,  # (V, K)
    yy: torch.Tensor,  # (V,)
    s: torch.Tensor,  # (K,)
    lams: torch.Tensor,  # (L,)
    n_eff: int,
    rank_p: int,
) -> torch.Tensor:
    one_minus_s = 1.0 - s
    denom = one_minus_s[None, :] + lams[:, None] * s[None, :]  # (L, K)
    d = 1.0 / denom
    if method == "reml":
        # -2 * restricted log-likelihood with sigma^2 profiled out, constants dropped:
        # (n - M_p) log(RSS + lam b'Pb) + log|A + lam P| - rank(P) log lam
        q = (yy[:, None] - z2 @ d.T).clamp_min(1e-30)
        logdet = torch.log(denom).sum(dim=1) - rank_p * torch.log(lams)
        return (n_eff - (s.numel() - rank_p)) * torch.log(q) + logdet[None, :]
    rss = (yy[:, None] - z2 @ (2 * d - d * d * one_minus_s[None, :]).T).clamp_min(1e-30)
    edf = (one_minus_s[None, :] * d).sum(dim=1)
    return n_eff * rss / (n_eff - edf).clamp_min(1.0)[None, :] ** 2


def _refine_log_lambda(crit: torch.Tensor, log_grid: torch.Tensor) -> torch.Tensor:
    """Per-voxel argmin on the grid, refined by a parabola through its neighbours."""
    idx = crit.argmin(dim=1)
    n = log_grid.numel()
    inner = (idx > 0) & (idx < n - 1)
    i0 = idx.clamp(1, n - 2)
    rows = torch.arange(crit.shape[0], device=crit.device)
    f0, f1, f2 = crit[rows, i0 - 1], crit[rows, i0], crit[rows, i0 + 1]
    curv = f0 - 2 * f1 + f2
    step = torch.where(curv > 0, 0.5 * (f0 - f2) / curv, torch.zeros_like(curv)).clamp(-1, 1)
    h = log_grid[1] - log_grid[0]
    refined = log_grid[i0] + step * h
    return torch.where(inner, refined, log_grid[idx])


def fit_smooth_basis(
    data: torch.Tensor,
    design: torch.Tensor,
    n_task: int,
    penalty: np.ndarray,
    *,
    method: str = "reml",
    lam: float | None = None,
    log10_grid: np.ndarray | None = None,
    device: torch.device | None = None,
    chunk_size: int | None = None,
    verbose: bool = False,
) -> SmoothBasisFit:
    """Penalized fit of the first ``n_task`` design columns, lambda per voxel.

    ``data`` is ``(V, T)``, ``design`` ``(T, n_task + n_nuisance)`` -- the packed
    concatenated GLM form (task block first, block-diagonal nuisance after).
    ``method="fixed"`` uses ``lam`` everywhere; ``"reml"``/``"gcv"`` pick it
    per voxel on ``log10_grid``.  The decomposition is float64 on the CPU
    (K x K, tiny); voxel work streams in chunks on ``device``.
    """
    if method not in SELECTION_METHODS:
        raise ValueError(f"method must be one of {SELECTION_METHODS}, got {method!r}")
    if method == "fixed" and (lam is None or lam <= 0):
        raise ValueError("method='fixed' needs a positive lam")
    device = device if device is not None else torch.device("cpu")
    spec = _spectrum(design.detach().cpu().double().numpy(), n_task, penalty)

    log_grid = torch.as_tensor(
        DEFAULT_LOG10_GRID if log10_grid is None else log10_grid, dtype=torch.float64
    ).to(device) * np.log(10.0)
    g = torch.as_tensor(spec.g, dtype=torch.float32, device=device)
    q = torch.as_tensor(spec.q_nuis, dtype=torch.float32, device=device)
    s = torch.as_tensor(spec.s, dtype=torch.float64, device=device)
    w = torch.as_tensor(spec.w, dtype=torch.float64, device=device)

    n_vox, n_t = data.shape
    if chunk_size is None:
        chunk_size = estimate_chunk_size(
            n_vox, n_t, n_task + int(log_grid.numel()), device, operation="glm"
        )
    out_b = torch.empty((n_vox, n_task), dtype=torch.float32)
    out_lam = torch.empty(n_vox, dtype=torch.float32)
    out_edf = torch.empty(n_vox, dtype=torch.float32)
    out_r2 = torch.empty(n_vox, dtype=torch.float32)
    starts = range(0, n_vox, chunk_size)
    for a in tqdm(
        starts,
        desc="  Smooth fit",
        unit="chunk",
        leave=True,
        disable=not verbose or len(starts) < 2,
    ):
        b = min(a + chunk_size, n_vox)
        y = data[a:b].to(device=device, dtype=torch.float32)
        y_t = y - (y @ q) @ q.T if q.shape[1] else y
        z = (y_t @ g).double()
        z2 = z * z
        yy = (y_t.double() ** 2).sum(dim=1)
        if method == "fixed":
            log_lam = torch.full((b - a,), float(np.log(lam)), dtype=torch.float64, device=device)
        else:
            crit = _criterion(method, z2, yy, s, torch.exp(log_grid), spec.n_eff, spec.rank_penalty)
            log_lam = _refine_log_lambda(crit, log_grid)
        lam_v = torch.exp(log_lam)
        d = 1.0 / ((1.0 - s)[None, :] + lam_v[:, None] * s[None, :])
        betas = (d * z) @ w.T
        rss = (yy - (z2 * (2 * d - d * d * (1.0 - s)[None, :])).sum(dim=1)).clamp_min(0.0)
        out_b[a:b] = betas.float().cpu()
        out_lam[a:b] = lam_v.float().cpu()
        out_edf[a:b] = ((1.0 - s)[None, :] * d).sum(dim=1).float().cpu()
        out_r2[a:b] = cod_from_ss_residual(y, rss.float()).cpu()
    return SmoothBasisFit(betas=out_b, lam=out_lam, edf=out_edf, r2=out_r2, method=method)
