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
    """Normalized joint + marginal histograms (sum to 1).

    Every field carries an optional leading batch axis: ``xyc`` is (nbp, nbp) or
    (B, nbp, nbp), the marginals (nbp,) or (B, nbp). The measures below reduce
    over the trailing axes and return a scalar or (B,) to match, so one
    histogram and a stack of them go through exactly the same arithmetic.
    """

    xyc: Tensor  # (..., nbp, nbp)
    xc: Tensor  # (..., nbp)
    yc: Tensor  # (..., nbp)
    nbp: int

    @property
    def batched(self) -> bool:
        return self.xyc.dim() == 3


def build_joint_hist(
    base: Tensor,
    warped: Tensor,
    weight: Tensor | None = None,
    nbin: int | None = None,
    base_clip: tuple[float, float] | None = None,
    source_clip: tuple[float, float] | None = None,
    batched: bool = False,
) -> JointHist:
    """Bilinear-deposit 2D histogram of (base, warped), differentiable in warped.

    Equal-size bins span the data range (or the clip range when provided),
    matching ``INCOR_addto_2Dhist`` for the non-clipped case.

    With ``batched``, ``warped`` is (B, M): B candidate transforms scored against
    the same base points, which is how the refiners ask for it. All B are
    deposited by one ``index_add_`` into a (B, nbp*nbp) buffer rather than by B
    of them -- the deposit is launch-bound at these point counts, so B
    histograms cost barely more than one. The flag is explicit because ``warped``
    is just as often a 3-D volume, which flattens to a single histogram.
    """
    x = base.reshape(-1)
    y = warped.reshape(warped.shape[0], -1) if batched else warped.reshape(1, -1)
    if weight is not None:
        w = weight.reshape(-1)
        good = w > 0
        x, w, y = x[good], w[good], y[:, good]
    else:
        w = torch.ones_like(x)

    nb, n = y.shape
    if nbin is None:
        nbin = compute_nbin(n)
    nbm = nbin - 1
    nbp = nbin + 1
    device = x.device

    def _out(xyc: Tensor, xc: Tensor, yc: Tensor) -> JointHist:
        return JointHist(xyc, xc, yc, nbp) if batched else JointHist(xyc[0], xc[0], yc[0], nbp)

    # Bin extents define the (fixed) histogram support and carry no gradient.
    xd, yd = x.detach(), y.detach()
    xb = base_clip[0] if base_clip else float(xd.min())
    xt = base_clip[1] if base_clip else float(xd.max())
    if source_clip:
        yb = torch.full((nb, 1), float(source_clip[0]), device=device, dtype=y.dtype)
        yt = torch.full((nb, 1), float(source_clip[1]), device=device, dtype=y.dtype)
    else:
        # Without a clip range each candidate spans its own warped values, exactly
        # as a one-at-a-time build would.
        yb = yd.min(dim=1, keepdim=True).values
        yt = yd.max(dim=1, keepdim=True).values
    if xt <= xb:
        z = torch.zeros(nb, nbp, nbp, device=device)
        return _out(z, z[:, :, 0].clone(), z[:, 0].clone())
    # A candidate whose warped values are all one number (a transform that fell
    # entirely outside the source) has no histogram; its row stays zero, which is
    # what every measure below reads as "no information".
    live = (yt > yb).to(y.dtype)

    xi = nbm / (xt - xb)
    yi = nbm / (yt - yb).clamp(min=1e-12)

    xx = ((x - xb) * xi).clamp(0.0, nbm)
    yy = ((y - yb) * yi).clamp(0.0, nbm)
    jj = xx.floor().clamp(0, nbm - 1).long()  # bin index (detached path)
    kk = yy.floor().clamp(0, nbm - 1).long()
    fx = xx - jj.to(xx.dtype)  # fractional offset (carries gradient via warped)
    fy = yy - kk.to(yy.dtype)
    x1 = 1.0 - fx
    y1 = 1.0 - fy

    # Linear indices into each candidate's (nbp, nbp) joint histogram, offset by
    # the candidate so one deposit fills the whole stack.
    row = torch.arange(nb, device=device)[:, None]
    base_idx = (jj[None, :] * nbp + kk) + row * (nbp * nbp)
    wl = w * live
    xyc = torch.zeros(nb * nbp * nbp, device=device)
    xyc.index_add_(0, base_idx.reshape(-1), (x1 * y1 * wl).reshape(-1))
    xyc.index_add_(0, (base_idx + nbp).reshape(-1), (fx * y1 * wl).reshape(-1))  # (jj+1, kk)
    xyc.index_add_(0, (base_idx + 1).reshape(-1), (x1 * fy * wl).reshape(-1))  # (jj, kk+1)
    xyc.index_add_(0, (base_idx + nbp + 1).reshape(-1), (fx * fy * wl).reshape(-1))
    xyc = xyc.reshape(nb, nbp, nbp)

    # The x marginal depends only on the base points, so it is one histogram
    # broadcast across the candidates (scaled by the same liveness).
    xc1 = torch.zeros(nbp, device=device)
    xc1.index_add_(0, jj, x1 * w)
    xc1.index_add_(0, jj + 1, fx * w)
    xc = xc1[None, :] * live

    kidx = kk + row * nbp
    yc = torch.zeros(nb * nbp, device=device)
    yc.index_add_(0, kidx.reshape(-1), (y1 * wl).reshape(-1))
    yc.index_add_(0, (kidx + 1).reshape(-1), (fy * wl).reshape(-1))
    yc = yc.reshape(nb, nbp)

    nww = w.sum().clamp(min=1e-12)
    return _out(xyc / nww, xc / nww, yc / nww)


# ---------------------------------------------------------------------------
# Measures (ports of the INCOR_* functions in thd_incorrelate.c)
# ---------------------------------------------------------------------------


def _masked(mask: Tensor, value: Tensor) -> Tensor:
    """``value`` where ``mask``, else 0 -- and no gradient from the masked-out entries.

    Boolean *indexing* (``x[x > 0]``) would do the same for one histogram but goes
    ragged the moment there is a batch axis, so the masking is done in place with
    the where/where pattern: the masked-out entries never see the log or the
    division, so neither the value nor its gradient can be a NaN.
    """
    return torch.where(mask, value, torch.zeros_like(value))


def _safe(mask: Tensor, value: Tensor) -> Tensor:
    """``value`` where ``mask``, else 1 -- the operand to feed log/divide."""
    return torch.where(mask, value, torch.ones_like(value))


def _xlogx(p: Tensor) -> Tensor:
    """p log p with 0 log 0 = 0, elementwise."""
    pos = p > 0
    safe = _safe(pos, p)
    return _masked(pos, safe * safe.log())


def _entropy_terms(h: JointHist):
    """Return (Hx+Hy weighted sum vv, joint sum uu) using natural log.

    vv = sum xc log xc + sum yc log yc      (== -(Hx+Hy))
    uu = sum xyc log xyc                     (== -H(x,y))
    """
    vv = _xlogx(h.xc).sum(-1) + _xlogx(h.yc).sum(-1)
    uu = _xlogx(h.xyc).sum((-2, -1))
    return vv, uu


def mutual_info(h: JointHist) -> Tensor:
    """MI in bits = 1.4427 * sum xyc log(xyc/(xc*yc))  (INCOR_mutual_info)."""
    xc, yc, xyc = h.xc, h.yc, h.xyc
    denom = xc[..., :, None] * yc[..., None, :]
    mask = (xyc > 0) & (denom > 0)
    val = _masked(mask, _safe(mask, xyc) * (_safe(mask, xyc) / _safe(mask, denom)).log())
    return _LOG2E * val.sum((-2, -1))


def joint_entropy(h: JointHist) -> Tensor:
    """H(base, source) using natural log (INCOR / je)."""
    return -_xlogx(h.xyc).sum((-2, -1))


def norm_mutinf(h: JointHist) -> Tensor:
    """H(x,y) / [H(x)+H(y)] = uu/vv  (INCOR_norm_mutinf; small == redundant)."""
    vv, uu = _entropy_terms(h)
    ok = vv != 0
    return _masked(ok, uu / _safe(ok, vv))


def _corr_ratio(marg: Tensor, other: Tensor, xyc: Tensor, axis: int, nbp: int) -> Tensor:
    """Conditional-variance ratio along ``axis`` of the joint histogram.

    ``axis`` is the one summed over to get the conditional moments: -1 for
    Var(y|x)/Var(y), -2 for Var(x|y)/Var(x).
    """
    idx = torch.arange(nbp, device=xyc.device, dtype=xyc.dtype)
    shaped = idx[:, None] if axis == -2 else idx
    mm = (shaped * xyc).sum(axis)
    vv = (shaped**2 * xyc).sum(axis)
    pos = marg > 0
    cyvar = _masked(pos, vv - mm**2 / _safe(pos, marg)).sum(-1)
    m1 = (idx * other).sum(-1)
    v1 = (idx**2 * other).sum(-1)
    uvar = v1 - m1**2
    ok = uvar > 0
    # An unvarying marginal has no ratio to take; AFNI's guard returns 1 there.
    return torch.where(ok, cyvar / _safe(ok, uvar), torch.ones_like(cyvar))


def _corr_ratio_yx(h: JointHist) -> Tensor:
    """Var(y|x)/Var(y) using bin-index moments (INCOR_corr_ratio)."""
    return _corr_ratio(h.xc, h.yc, h.xyc, -1, h.nbp)


def _corr_ratio_xy(h: JointHist) -> Tensor:
    """Var(x|y)/Var(x)."""
    return _corr_ratio(h.yc, h.xc, h.xyc, -2, h.nbp)


def hellinger(h: JointHist) -> Tensor:
    """Hellinger affinity sum sqrt(xyc*xc*yc)  (INCOR_hellinger returns 1-this)."""
    xc, yc, xyc = h.xc, h.yc, h.xyc
    prod = xyc * xc[..., :, None] * yc[..., None, :]
    # sqrt has an infinite slope at 0; route the gradient only through the
    # strictly-positive entries (the zero entries contribute 0 and no grad).
    pos = prod > 0
    return _masked(pos, _safe(pos, prod).sqrt()).sum((-2, -1))


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


def combo_terms(
    base,
    warped,
    weights: tuple[float, float, float, float],
    weight=None,
    nbin=None,
    base_clip=None,
    source_clip=None,
    batched: bool = False,
) -> Tensor:
    """Weighted (hel, mi, nmi, crA) sum in ffs convention, from ONE histogram.

    ``batched`` scores a (B, M) stack of candidates and returns (B,).

    This is the extra half of AFNI's lpc+/lpa+ combination. Calling the four
    ``*_cost`` helpers instead would build four *identical* joint histograms —
    the dominant cost in the combination (a build is ~1.4 ms on 900k points,
    against ~0.9 ms for the whole local-Pearson cost it is being added to), so
    sharing the histogram is most of the difference between a usable combined
    cost and an unusable one.
    """
    w_hel, w_mi, w_nmi, w_cra = weights
    h = build_joint_hist(base, warped, weight, nbin, base_clip, source_clip, batched)
    total = None

    def _add(acc, term):
        return term if acc is None else acc + term

    if w_hel:
        total = _add(total, w_hel * (1.0 - hellinger(h)))
    if w_mi:
        total = _add(total, w_mi * mutual_info(h))
    if w_nmi:
        total = _add(total, w_nmi * -norm_mutinf(h))
    if w_cra:
        total = _add(total, w_cra * (1.0 - 0.5 * (_corr_ratio_yx(h) + _corr_ratio_xy(h))))
    if total is None:
        shape = h.xyc.shape[:1] if h.batched else ()
        return torch.zeros(shape, device=h.xyc.device)
    return total


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
