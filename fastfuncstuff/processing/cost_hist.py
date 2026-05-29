"""Histogram-based image-matching costs (MI / NMI / CR / Hellinger / JE).

Ports AFNI's ``thd_incorrelate.c`` 2D-histogram machinery:
  - bin count  ``nbin = round(ndata**(1/3))`` clamped to [5, 255]
  - clip levels from ``INCOR_clipate`` (positive images only: cliplevel
    0.321 / quantile 0.987, capped at 6.543x); for images containing
    negatives the clip is disabled and equal-size bins span [min, max]
  - bilinear ("interpolated") deposit into the joint histogram, exactly as
    ``INCOR_addto_2Dhist``
  - measures from ``INCOR_mutual_info`` / ``norm_mutinf`` / ``corr_ratio`` /
    ``hellinger``

The deposit uses a *soft* (bilinear) split, so the joint histogram — and every
measure built from it — is differentiable in the warped source (the integer
bin index is treated as constant; only the fractional bin offset carries the
gradient, which is exactly the linear region AFNI samples).

AFNI minimises its cost; ffs maximises (higher == better).  Each public
``*_cost`` returns the ffs-convention value; :func:`hist2d_measures` returns the
raw measures so the AFNI-printed values can be reproduced for validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

_LOG2E = 1.4426950408889634  # 1/log(2): nats -> bits, matches AFNI's 1.4427


def compute_nbin(ndata: int) -> int:
    """``nbin = round(ndata**(1/3))`` clamped to [5, 255] (AFNI)."""
    nbin = int(round(ndata ** (1.0 / 3.0)))
    return max(5, min(255, nbin))


def _clip_level(data: Tensor, frac: float) -> float:
    """AFNI THD_cliplevel-ish: a robust low clip from the positive values."""
    pos = data[data > 0]
    if pos.numel() == 0:
        return 0.0
    # THD_cliplevel iterates; a quantile of the positive part is a faithful,
    # cheap stand-in for the histogram-peak heuristic used by AFNI.
    return float(pos.quantile(frac).item())


def clip_range(data: Tensor) -> tuple[float, float] | None:
    """Return (cbot, ctop) clip à la INCOR_clipate, or None if disabled.

    Clipping is only applied to strictly-positive images (matching AFNI:
    images with negative values get clipping disabled).
    """
    d = data.reshape(-1)
    if d.numel() < 666:
        return None
    if float(d.min()) < 0.0:
        return None
    cbot = _clip_level(d, 0.321)
    ctop = float(d.quantile(0.987).item())
    if ctop > 6.543 * cbot:
        ctop = 6.543 * cbot
    if cbot >= ctop:
        return None
    return (cbot, ctop)


@dataclass
class JointHist:
    """Normalized joint + marginal histograms (sum to 1)."""

    xyc: Tensor  # (nbp, nbp)
    xc: Tensor  # (nbp,)
    yc: Tensor  # (nbp,)
    nbp: int


def build_joint_hist(
    base: Tensor,
    warped: Tensor,
    weight: Tensor | None = None,
    nbin: int | None = None,
    base_clip: tuple[float, float] | None = None,
    source_clip: tuple[float, float] | None = None,
) -> JointHist:
    """Bilinear-deposit 2D histogram of (base, warped), differentiable in warped.

    Equal-size bins span the data range (or the clip range when provided),
    matching ``INCOR_addto_2Dhist`` for the non-clipped case.
    """
    x = base.reshape(-1)
    y = warped.reshape(-1)
    if weight is not None:
        w = weight.reshape(-1)
        good = w > 0
        x, y, w = x[good], y[good], w[good]
    else:
        w = torch.ones_like(x)

    n = x.numel()
    if nbin is None:
        nbin = compute_nbin(n)
    nbm = nbin - 1
    nbp = nbin + 1
    device = x.device

    # Bin extents define the (fixed) histogram support and carry no gradient.
    xd, yd = x.detach(), y.detach()
    xb = base_clip[0] if base_clip else float(xd.min())
    xt = base_clip[1] if base_clip else float(xd.max())
    yb = source_clip[0] if source_clip else float(yd.min())
    yt = source_clip[1] if source_clip else float(yd.max())
    if xt <= xb or yt <= yb:
        z = torch.zeros(nbp, nbp, device=device)
        return JointHist(z, z[:, 0].clone(), z[0].clone(), nbp)

    xi = nbm / (xt - xb)
    yi = nbm / (yt - yb)

    xx = ((x - xb) * xi).clamp(0.0, nbm)
    yy = ((y - yb) * yi).clamp(0.0, nbm)
    jj = xx.floor().clamp(0, nbm - 1).long()  # bin index (detached path)
    kk = yy.floor().clamp(0, nbm - 1).long()
    fx = xx - jj.to(xx.dtype)  # fractional offset (carries gradient via warped)
    fy = yy - kk.to(yy.dtype)
    x1 = 1.0 - fx
    y1 = 1.0 - fy

    # Linear indices into the (nbp, nbp) joint histogram for the 4 corners.
    base_idx = jj * nbp + kk
    xyc = torch.zeros(nbp * nbp, device=device)
    xyc.index_add_(0, base_idx, x1 * y1 * w)
    xyc.index_add_(0, base_idx + nbp, fx * y1 * w)  # (jj+1, kk)
    xyc.index_add_(0, base_idx + 1, x1 * fy * w)  # (jj, kk+1)
    xyc.index_add_(0, base_idx + nbp + 1, fx * fy * w)
    xyc = xyc.reshape(nbp, nbp)

    xc = torch.zeros(nbp, device=device)
    xc.index_add_(0, jj, x1 * w)
    xc.index_add_(0, jj + 1, fx * w)
    yc = torch.zeros(nbp, device=device)
    yc.index_add_(0, kk, y1 * w)
    yc.index_add_(0, kk + 1, fy * w)

    nww = w.sum().clamp(min=1e-12)
    return JointHist(xyc / nww, xc / nww, yc / nww, nbp)


# ---------------------------------------------------------------------------
# Measures (ports of the INCOR_* functions in thd_incorrelate.c)
# ---------------------------------------------------------------------------


def _entropy_terms(h: JointHist):
    """Return (Hx+Hy weighted sum vv, joint sum uu) using natural log.

    vv = sum xc log xc + sum yc log yc      (== -(Hx+Hy))
    uu = sum xyc log xyc                     (== -H(x,y))
    """
    xc, yc, xyc = h.xc, h.yc, h.xyc
    vv = (xc[xc > 0] * xc[xc > 0].log()).sum() + (yc[yc > 0] * yc[yc > 0].log()).sum()
    pos = xyc[xyc > 0]
    uu = (pos * pos.log()).sum()
    return vv, uu


def mutual_info(h: JointHist) -> Tensor:
    """MI in bits = 1.4427 * sum xyc log(xyc/(xc*yc))  (INCOR_mutual_info)."""
    xc, yc, xyc = h.xc, h.yc, h.xyc
    denom = xc[:, None] * yc[None, :]
    mask = (xyc > 0) & (denom > 0)
    val = (xyc[mask] * (xyc[mask] / denom[mask]).log()).sum()
    return _LOG2E * val


def joint_entropy(h: JointHist) -> Tensor:
    """H(base, source) using natural log (INCOR / je)."""
    pos = h.xyc[h.xyc > 0]
    return -(pos * pos.log()).sum()


def norm_mutinf(h: JointHist) -> Tensor:
    """H(x,y) / [H(x)+H(y)] = uu/vv  (INCOR_norm_mutinf; small == redundant)."""
    vv, uu = _entropy_terms(h)
    if vv == 0:
        return torch.zeros((), device=h.xyc.device)
    return uu / vv


def _corr_ratio_yx(h: JointHist) -> Tensor:
    """Var(y|x)/Var(y) using bin-index moments (INCOR_corr_ratio)."""
    xc, xyc = h.xc, h.xyc
    nbp = h.nbp
    jdx = torch.arange(nbp, device=xyc.device, dtype=xyc.dtype)
    # Var(y|x): for each x-bin column, moments of y over j
    mm = (jdx[None, :] * xyc).sum(dim=1)  # E(y|x)*xc
    vv = (jdx[None, :] ** 2 * xyc).sum(dim=1)  # E(y^2|x)*xc
    pos = xc > 0
    cyvar = (vv[pos] - mm[pos] ** 2 / xc[pos]).sum()
    # Var(y)
    mY = (jdx * h.yc).sum()
    vY = (jdx**2 * h.yc).sum()
    uyvar = vY - mY**2
    return cyvar / uyvar if uyvar > 0 else torch.ones((), device=xyc.device)


def _corr_ratio_xy(h: JointHist) -> Tensor:
    """Var(x|y)/Var(x)."""
    yc, xyc = h.yc, h.xyc
    nbp = h.nbp
    idx = torch.arange(nbp, device=xyc.device, dtype=xyc.dtype)
    mm = (idx[:, None] * xyc).sum(dim=0)
    vv = (idx[:, None] ** 2 * xyc).sum(dim=0)
    pos = yc > 0
    cyvar = (vv[pos] - mm[pos] ** 2 / yc[pos]).sum()
    mX = (idx * h.xc).sum()
    vX = (idx**2 * h.xc).sum()
    uxvar = vX - mX**2
    return cyvar / uxvar if uxvar > 0 else torch.ones((), device=xyc.device)


def hellinger(h: JointHist) -> Tensor:
    """Hellinger affinity sum sqrt(xyc*xc*yc)  (INCOR_hellinger returns 1-this)."""
    xc, yc, xyc = h.xc, h.yc, h.xyc
    prod = xyc * xc[:, None] * yc[None, :]
    # sqrt has an infinite slope at 0; route the gradient only through the
    # strictly-positive entries (the zero entries contribute 0 and no grad).
    pos = prod > 0
    safe = torch.where(pos, prod, torch.ones_like(prod))
    return torch.where(pos, safe.sqrt(), torch.zeros_like(prod)).sum()


@dataclass
class HistMeasures:
    mi: float
    je: float
    nmi: float
    hel: float
    cr_yx: float  # Var(y|x)/Var(y)  ratio
    cr_xy: float  # Var(x|y)/Var(x)  ratio


def hist2d_measures(
    base, warped, weight=None, nbin=None, base_clip=None, source_clip=None
) -> HistMeasures:
    """Compute all histogram measures (floats) for reporting / validation."""
    h = build_joint_hist(base, warped, weight, nbin, base_clip, source_clip)
    return HistMeasures(
        mi=float(mutual_info(h)),
        je=float(joint_entropy(h)),
        nmi=float(norm_mutinf(h)),
        hel=float(hellinger(h)),
        cr_yx=float(_corr_ratio_yx(h)),
        cr_xy=float(_corr_ratio_xy(h)),
    )


# ---------------------------------------------------------------------------
# ffs-convention costs (higher == better), differentiable in warped
# ---------------------------------------------------------------------------


def _hist(base, warped, weight, nbin, base_clip, source_clip):
    return build_joint_hist(base, warped, weight, nbin, base_clip, source_clip)


def mi_cost(base, warped, weight=None, nbin=None, base_clip=None, source_clip=None) -> Tensor:
    """Mutual information (bits); higher == better match."""
    return mutual_info(_hist(base, warped, weight, nbin, base_clip, source_clip))


def nmi_cost(base, warped, weight=None, nbin=None, base_clip=None, source_clip=None) -> Tensor:
    """-(H(x,y)/[H(x)+H(y)]); higher == better (AFNI minimises the ratio)."""
    return -norm_mutinf(_hist(base, warped, weight, nbin, base_clip, source_clip))


def je_cost(base, warped, weight=None, nbin=None, base_clip=None, source_clip=None) -> Tensor:
    """-H(x,y); higher == better (AFNI minimises joint entropy)."""
    return -joint_entropy(_hist(base, warped, weight, nbin, base_clip, source_clip))


def hel_cost(base, warped, weight=None, nbin=None, base_clip=None, source_clip=None) -> Tensor:
    """Hellinger distance 1 - affinity; higher == better.

    The affinity sum is maximised (== 1) when base and source are *independent*,
    so the alignment cost is its complement, which grows with dependence.
    """
    aff = hellinger(_hist(base, warped, weight, nbin, base_clip, source_clip))
    return 1.0 - aff


def cr_cost(
    base, warped, weight=None, mode="u", nbin=None, base_clip=None, source_clip=None
) -> Tensor:
    """Correlation-ratio cost; higher == better.

    mode "u" (unsymmetric, CR(source|base)), "a" (additive), "m"
    (multiplicative).  AFNI minimises ``1-|assoc|`` / the raw ratio; we return
    the association strength (1 - ratio), so higher == better.
    """
    h = _hist(base, warped, weight, nbin, base_clip, source_clip)
    yx = _corr_ratio_yx(h)  # Var(y|x)/Var(y) -- the AFNI-printed crU ratio
    if mode == "u":
        return 1.0 - yx
    xy = _corr_ratio_xy(h)
    if mode == "a":  # AFNI crU/crA printed = 0.5*(yx+xy); cost = 1 - that
        return 1.0 - 0.5 * (yx + xy)
    return 1.0 - yx * xy  # multiplicative: AFNI crM printed = yx*xy
