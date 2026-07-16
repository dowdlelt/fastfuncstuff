"""Partial-volume tissue-synthesis cost for boundary-based registration.

The sparse BBR / NGF costs (:mod:`fastfuncstuff.processing.bbr`) drive the fit
from ONE boundary (WM/GM) or from edge directions. Where soft tissue
probabilities are available (e.g. SPM ``c1/c2/c3`` or FreeSurfer posteriors),
this DENSE cost uses *all* tissue at once:

Predict the EPI as a partial-volume mixture of the (warped) tissue fractions,
``EPI_pred = Σ_k f_k · μ_k`` (+ intercept), fit the per-tissue means ``μ`` by
least squares, and register by minimizing the EPI variance the mixture cannot
explain. Because the fraction fields are STATIC (attached to fixed sample
coordinates) the projection that profiles ``μ`` out is precomputed once, so the
per-iteration cost is a single trilinear sample + two small matmuls — and it is
differentiable in the warp through the EPI sampling.

Why it complements BBR at high (sub-mm / laminar) resolution:
  * uses the whole WM→GM→CSF intensity ramp, not just the WM/GM step;
  * **learns EPI polarity for free** (``μ`` are fitted — no wm_bright/wm_dark);
  * models partial volume, matching the EPI's own PSF blur;
  * CSF (bright in T2*/EPI) is a strong anchor where the WM/GM edge is weak.

Coordinate convention matches :mod:`bbr` / ``interp``: points/coords are
``(x, y, z)`` voxel-index over a ``(nz, ny, nx)`` volume.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .bbr import rst_matrix
from .interp import trilinear_interpolate


def build_tissue_design(
    tissues: list[Tensor],
    *,
    mask: Tensor | None = None,
    n_sample: int | None = 20000,
    intercept: bool = True,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Sample coordinates + tissue-fraction design matrix for the synthesis cost.

    Args:
        tissues: list of ``(nz, ny, nx)`` tissue-fraction volumes (e.g. WM, GM,
            CSF), already cast into the EPI grid.
        mask: optional ``(nz, ny, nx)`` boolean/float brain mask; default = where
            the tissue fractions sum above 0.5.
        n_sample: cap on the number of sample voxels (random subsample for speed
            on big high-res volumes); None = use all masked voxels.
        intercept: append a column of ones so ``μ`` can absorb a DC offset.
        device: placement of the returned tensors.
        seed: RNG seed for the subsample (reproducible).

    Returns:
        coords: ``(P, 3)`` sample points ``(x, y, z)``.
        F: ``(P, K)`` design — one column per tissue (+ intercept) at ``coords``.
    """
    stack = torch.stack([t.to(device) for t in tissues], dim=0)  # (K, nz, ny, nx)
    total = stack.sum(dim=0)
    m = (total > 0.5) if mask is None else (mask.to(device) > 0.5)
    kk, jj, ii = torch.nonzero(m, as_tuple=True)  # (z, y, x) indices
    if kk.numel() == 0:
        raise ValueError("tissue mask is empty — check the probability maps / mask")

    if n_sample is not None and kk.numel() > n_sample:
        g = torch.Generator(device=kk.device).manual_seed(seed)
        sel = torch.randperm(kk.numel(), generator=g, device=kk.device)[:n_sample]
        kk, jj, ii = kk[sel], jj[sel], ii[sel]

    coords = torch.stack([ii, jj, kk], dim=1).to(device=device, dtype=torch.float32)
    cols = [t[kk, jj, ii] for t in stack]  # (P,) per tissue
    if intercept:
        cols.append(torch.ones_like(cols[0]))
    F = torch.stack(cols, dim=1).to(dtype=torch.float32)  # (P, K)
    return coords, F


def tissue_projector(F: Tensor) -> Tensor:
    """Precompute the pseudo-inverse ``F⁺`` (K, P) — profiles the tissue means out
    of a sampled EPI vector via one matmul. ``F`` is static, so this is done once.

    Uses ``pinv`` (not a normal-equations solve) because the tissue columns are
    typically COLLINEAR with the intercept — SPM ``c1+c2+c3 ≈ 1`` inside the brain,
    so ``FᵀF`` is (near-)singular. The projection ``F·F⁺·y`` onto the column space
    is still unique and correct; only the individual means are non-identifiable,
    which does not matter for the residual cost.
    """
    return torch.linalg.pinv(F)  # (K, P)


def _sample_zero_outside(vol: Tensor, pts: Tensor) -> tuple[Tensor, Tensor]:
    """Trilinear sample at ``pts`` (P,3); returns ``(values, oob_mask)`` with 0
    outside the FoV. Callers penalize ``oob`` explicitly — a zeroed sample alone
    is NOT enough, since the intercept can explain all-zeros for free."""
    nz, ny, nx = vol.shape
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    oob = (x < -0.5) | (x > nx - 0.5) | (y < -0.5) | (y > ny - 0.5) | (z < -0.5) | (z > nz - 0.5)
    out = trilinear_interpolate(vol, x, y, z)
    return torch.where(oob, torch.zeros_like(out), out), oob


def synthesis_residual_cost(
    y: Tensor, F: Tensor, Fpinv: Tensor, *, weight: Tensor | None = None
) -> Tensor:
    """Fraction of EPI variance UNEXPLAINED by the tissue mixture (≈ 1 − R²).

    ``y`` = EPI sampled at the (warped) coords; ``F``/``Fpinv`` from the static
    tissue design. ``μ = Fpinv·y`` (profiled out), residual ``y − F·μ``. Returned
    as RSS / total-variance so the cost is **O(1)** and scale-free — essential when
    combined with the O(1) BBR/NGF terms (a raw RSS is ~intensity², millions on
    real data, and would swamp the boundary terms → the fit diverges). The variance
    normalizer is detached, so the gradient is the RSS direction, just rescaled.
    Lower = the tissue fractions explain the EPI better = better alignment.
    """
    beta = Fpinv @ y  # (K,) fitted tissue means (+ intercept)
    resid = y - F @ beta  # (P,)
    if weight is not None:
        w = weight.to(dtype=resid.dtype, device=resid.device)
        wsum = w.sum().clamp_min(1e-12)
        rss = (w * resid * resid).sum() / wsum
        ybar = (w * y).sum() / wsum
        tss = (w * (y - ybar) * (y - ybar)).sum() / wsum
    else:
        rss = (resid * resid).mean()
        tss = y.var(unbiased=False)
    return rss / tss.detach().clamp_min(1e-12)


def _oob_penalty(oob: Tensor, weight: float) -> Tensor:
    """Penalty so pushing samples off the grid isn't free (the intercept would
    otherwise explain the zeroed samples). Unitless (a fraction), matching the
    normalized ≈1−R² residual so ``weight`` is meaningful; 0 when all in-bounds."""
    return weight * oob.float().mean()


def synthesis_cost_at_points(
    epi: Tensor,
    pts: Tensor,
    F: Tensor,
    Fpinv: Tensor,
    *,
    weight: Tensor | None = None,
    oob_penalty: float = 1.0,
) -> Tensor:
    """Partial-volume synthesis cost at *already-positioned* sample points.

    Shared core of the affine (:func:`tissue_synthesis_cost`) and warp (RBR field-
    shifted) stages: samples the EPI at ``pts``, returns the tissue-unexplained
    variance plus an out-of-FoV penalty. Differentiable in ``pts``.
    """
    y, oob = _sample_zero_outside(epi, pts)
    return synthesis_residual_cost(y, F, Fpinv, weight=weight) + _oob_penalty(oob, oob_penalty)


def tissue_synthesis_cost(
    epi: Tensor,
    coords: Tensor,
    F: Tensor,
    Fpinv: Tensor,
    params: Tensor,
    *,
    pivot: Tensor | None = None,
    weight: Tensor | None = None,
    oob_penalty: float = 1.0,
) -> Tensor:
    """Affine (single-transform) partial-volume synthesis cost.

    Applies the 9-param transform to ``coords``, then scores at those points
    (:func:`synthesis_cost_at_points`). Differentiable in ``params`` via the EPI
    sampling. For the nonlinear stage, the RBR field shifts ``coords`` instead.
    """
    if pivot is None:
        pivot = coords.mean(dim=0)
    mat = rst_matrix(params, pivot)
    pts = coords @ mat[:3, :3].T + mat[:3, 3]  # points-only affine (x, y, z)
    return synthesis_cost_at_points(epi, pts, F, Fpinv, weight=weight, oob_penalty=oob_penalty)
