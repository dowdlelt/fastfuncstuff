"""Vein exclusion masks from magnitude-phase coupling.

An alternative to correcting the magnitude at all.  [[Phase regression]] estimates
a macrovascular component and subtracts it, which means a point estimate of the
slope is applied as though it were known exactly — over-subtract and you eat
grey-matter signal, under-subtract and the vein survives.  For applications that
only need to *exclude* contaminated voxels (laminar profiles, layer extraction,
ROI definition), the cleaner question is a hypothesis test:

    does this voxel's magnitude covary with phase more than chance?

Randomly-oriented microvasculature produces magnitude change with no coherent
phase change, so a significant magnitude-phase correlation is direct evidence of
an oriented (macro)vessel.  Voxels that fail the test are kept; voxels that pass
are excluded.  This is the same logic as the H_c vs H_a contrast of Rowe 2005,
reached without fitting the full complex-valued GLM.

Nothing here modifies the corrected time series — the mask is an extra output.
"""

from __future__ import annotations

import math

import torch


def _t_sf(t: torch.Tensor, df: int) -> torch.Tensor:
    """Upper-tail survival function of Student's t, vectorised.

    Uses the regularised incomplete beta relation
    ``P(T > t) = 0.5 * I_{df/(df+t^2)}(df/2, 1/2)`` for t > 0.  torch has no
    betainc, so this goes through the continued-fraction expansion in float64;
    at fMRI voxel counts that is still far cheaper than a host round-trip to
    scipy, and it keeps the whole path on-device.
    """
    x = df / (df + t.double() ** 2)
    return 0.5 * _betainc(x, 0.5 * df, 0.5)


def _betainc(x: torch.Tensor, a: float, b: float, n_iter: int = 200) -> torch.Tensor:
    """Regularised incomplete beta I_x(a, b) via Lentz's continued fraction."""
    # I_x(a,b) = x^a (1-x)^b / (a B(a,b)) * CF, valid for x < (a+1)/(a+b+2).
    # Outside that range use the reflection I_x(a,b) = 1 - I_{1-x}(b,a).
    x = x.clamp(1e-300, 1.0)
    swap = x > (a + 1.0) / (a + b + 2.0)
    xs = torch.where(swap, 1.0 - x, x)
    aa = torch.where(swap, torch.full_like(x, b), torch.full_like(x, a))
    bb = torch.where(swap, torch.full_like(x, a), torch.full_like(x, b))

    tiny = 1e-300
    c = torch.ones_like(xs)
    d = 1.0 - (aa + bb) * xs / (aa + 1.0)
    d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
    d = 1.0 / d
    h = d.clone()

    for m in range(1, n_iter + 1):
        m2 = 2 * m
        # even step
        num = m * (bb - m) * xs / ((aa + m2 - 1.0) * (aa + m2))
        d = 1.0 + num * d
        d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
        c = 1.0 + num / c
        c = torch.where(c.abs() < tiny, torch.full_like(c, tiny), c)
        d = 1.0 / d
        h = h * d * c
        # odd step
        num = -(aa + m) * (aa + bb + m) * xs / ((aa + m2) * (aa + m2 + 1.0))
        d = 1.0 + num * d
        d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
        c = 1.0 + num / c
        c = torch.where(c.abs() < tiny, torch.full_like(c, tiny), c)
        d = 1.0 / d
        h = h * d * c

    log_pref = (
        aa * torch.log(xs)
        + bb * torch.log1p(-xs)
        - torch.log(aa)
        + torch.lgamma(aa + bb)
        - torch.lgamma(aa)
        - torch.lgamma(bb)
    )
    out = torch.exp(log_pref) * h
    return torch.where(swap, 1.0 - out, out).clamp(0.0, 1.0)


def coupling_pvalue(
    r: torch.Tensor,
    df: int,
    n_candidates: int = 1,
) -> torch.Tensor:
    """Two-sided p-value that |corr(magnitude, phase)| exceeds chance.

    Parameters
    ----------
    r : Tensor (n_voxels,)
        Pearson correlation between the magnitude and phase series the slope was
        fit on.
    df : int
        Residual degrees of freedom: timepoints minus nuisance regressors minus 2.
    n_candidates : int
        Number of donor candidates the correlation was maximised over.  1 for
        standard PR.  For sPR the reported r is an argmax over the neighbourhood,
        which inflates it under the null; a Sidak correction
        ``1 - (1-p)^n_candidates`` compensates.  Conservative but honest — without
        it the vein mask is badly over-inclusive at 26-connectivity.

    Returns
    -------
    p : Tensor (n_voxels,)
    """
    if df <= 0:
        return torch.ones_like(r)

    r_c = r.double().clamp(-0.999999, 0.999999)
    t = r_c.abs() * math.sqrt(df) / torch.sqrt(1.0 - r_c**2)
    p = (2.0 * _t_sf(t, df)).clamp(0.0, 1.0)
    if n_candidates > 1:
        p = 1.0 - (1.0 - p) ** n_candidates
    return p.to(r.dtype)


def fdr_threshold(p: torch.Tensor, mask: torch.Tensor, q: float = 0.05) -> float:
    """Benjamini-Hochberg critical p-value over the masked voxels.

    Returns 0.0 when nothing survives, so ``p <= threshold`` selects nothing.
    """
    vals = p[mask]
    n = vals.numel()
    if n == 0:
        return 0.0
    ordered, _ = torch.sort(vals)
    ranks = torch.arange(1, n + 1, device=ordered.device, dtype=ordered.dtype)
    passing = ordered <= (ranks / n) * q
    if not bool(passing.any()):
        return 0.0
    return float(ordered[passing.nonzero()[-1]].item())
