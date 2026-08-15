"""Deformation-regularity metrics for a nonlinear warp.

The other half of the referee in [[automatic registration tuning]]: similarity
says whether two images ended up looking alike, and this says whether the
deformation that got them there is anatomically plausible. Both are needed,
because a warp with enough degrees of freedom can drive any single similarity
metric to its optimum by folding tissue around — ranking candidates on
similarity alone selects for over-warping, monotonically.

These are *filters*, not soft terms. A folded warp is not a slightly worse warp,
it is a wrong one, and averaging it into a score lets a big similarity gain buy
its way past a defect that invalidates the result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from .optiwarp import jacobian_determinant
from .penalty import _central_diff_batched

# A handful of voxels grazing zero is numerical noise on the difference stencil,
# not a folded warp — typically at the very edge of the field, at values like
# -0.01. Disqualifying on those would reject every usable warp.
#
# The tolerance is max(count, fraction) rather than either alone, because a pure
# fraction misbehaves at both ends: on a 17k-voxel test volume 1e-4 is under two
# voxels, while on a 1.5M-voxel brain it silently permits 150. "A handful" is an
# absolute idea; the fraction only takes over once the volume is large enough
# that a handful really is negligible.
DEFAULT_MAX_NEG_VOXELS = 64
DEFAULT_MAX_NEG_FRAC = 1e-5
DEFAULT_MAX_JAC = 4.0  # local expansion beyond this is implausible
DEFAULT_MIN_JAC = 0.25  # ... and likewise compression

# Folding above the "handful" budget but below this fraction is MARGINAL rather
# than FAIL: a small, localised fold in an otherwise sound field is a warp that
# was pushed slightly too hard, not a broken one — and the fix is known (more
# regularization), so throwing the candidate away discards both a good result
# and the information about which way to move. See `regularity_verdict`.
DEFAULT_MARGINAL_NEG_FRAC = 5e-3  # 0.5% of in-mask voxels

PASS, MARGINAL, FAIL = "pass", "marginal", "fail"

# What to do about each failure mode. Regularity failures are *directional*:
# every one of them is fixed by moving a knob the tuner is already walking, so a
# rejected candidate still tells you where the good settings are.
REMEDY = {
    "folding": "increase regularization (qwarp -penfac; SyN -update_var/-total_var; "
    "flow -update_sigma/-total_sigma)",
    "over-compression": "increase regularization, or reduce the number of levels/iterations",
    "over-expansion": "increase regularization, or reduce the number of levels/iterations",
}


@dataclass
class WarpQC:
    """Regularity summary for one displacement field."""

    n_voxels: int
    jac_neg_count: int  # voxels with det(J) <= 0 -- folding
    jac_neg_frac: float
    jac_min: float
    jac_p01: float
    jac_p50: float
    jac_p99: float
    jac_max: float
    bending_energy: float  # mean squared second derivative, mm^-1
    disp_mean_mm: float
    disp_p99_mm: float
    disp_max_mm: float

    def as_dict(self) -> dict:
        return asdict(self)


def _quantiles(v: Tensor, qs: tuple[float, ...]) -> list[float]:
    """Quantiles of a flat tensor, with no size ceiling.

    ``torch.quantile`` refuses inputs beyond ~16M elements, which an unmasked
    warp field on a padded grid exceeds (a 193^3 base pads to 17M). Sorting has
    no such limit and is a few milliseconds on the GPU at this size.
    """
    s, _ = v.float().flatten().sort()
    n = s.numel()
    idx = torch.tensor([min(n - 1, max(0, int(round(q * (n - 1))))) for q in qs], device=s.device)
    return s[idx].tolist()


def _second_derivative_energy(d: Tensor) -> Tensor:
    """Mean squared second derivative of one displacement component.

    The standard bending energy integrand: the six distinct entries of the
    Hessian, with the off-diagonal (mixed) terms counted twice because the
    Hessian is symmetric.
    """
    total = torch.zeros((), device=d.device, dtype=d.dtype)
    firsts = [_central_diff_batched(d, dim=k) for k in range(3)]
    for i in range(3):
        for j in range(i, 3):
            second = _central_diff_batched(firsts[i], dim=j)
            total = total + (1.0 if i == j else 2.0) * (second**2).mean()
    return total


def warp_regularity(
    xd: Tensor,
    yd: Tensor,
    zd: Tensor,
    mask: Tensor | None = None,
    voxdims: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> WarpQC:
    """Summarise how well behaved a displacement field is.

    Args:
        xd, yd, zd: (nz, ny, nx) displacements in **voxel** units.
        mask: optional (nz, ny, nx); statistics are taken over ``mask > 0`` only.
            Strongly recommended — the field outside the brain is unconstrained,
            and letting it into the percentiles buries real folding in noise.
        voxdims: (dx, dy, dz) in mm, for reporting displacement magnitudes.

    Returns:
        :class:`WarpQC`.
    """
    if mask is not None and tuple(mask.shape) != tuple(xd.shape):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} != warp shape {tuple(xd.shape)}. "
            "Some tools (qwarp) save the field on a padded grid, so a mask built "
            "on the base grid has to be padded to match — see pad_mask_to_field()."
        )

    jac = jacobian_determinant(xd, yd, zd)

    dx, dy, dz = (float(v) for v in voxdims)
    disp_mm = torch.sqrt((xd * dx) ** 2 + (yd * dy) ** 2 + (zd * dz) ** 2)

    if mask is not None:
        sel = mask.reshape(-1) > 0
        jac_v = jac.reshape(-1)[sel]
        disp_v = disp_mm.reshape(-1)[sel]
    else:
        jac_v = jac.reshape(-1)
        disp_v = disp_mm.reshape(-1)

    n = int(jac_v.numel())
    if n == 0:
        raise ValueError("warp_regularity: mask selected no voxels")

    j01, j50, j99 = _quantiles(jac_v, (0.01, 0.5, 0.99))
    (d99,) = _quantiles(disp_v, (0.99,))

    # Bending energy is a property of the field, not of the mask, and the second
    # differences need neighbours — so it is computed on the whole grid.
    be = sum(float(_second_derivative_energy(d)) for d in (xd, yd, zd))

    n_neg = int((jac_v <= 0).sum())
    return WarpQC(
        n_voxels=n,
        jac_neg_count=n_neg,
        jac_neg_frac=n_neg / n,
        jac_min=float(jac_v.min()),
        jac_p01=j01,
        jac_p50=j50,
        jac_p99=j99,
        jac_max=float(jac_v.max()),
        bending_energy=be,
        disp_mean_mm=float(disp_v.mean()),
        disp_p99_mm=d99,
        disp_max_mm=float(disp_v.max()),
    )


def pad_mask_to_field(mask: Tensor, field_shape: tuple[int, int, int]) -> Tensor:
    """Centre a base-grid mask inside a larger (padded) warp grid.

    ``ffs_qwarp`` solves on a padded volume and saves the field on that grid, so
    a brain mask built on the base grid is smaller than the field it has to
    select from. The padding is symmetric, so centring recovers the alignment.
    """
    if tuple(mask.shape) == tuple(field_shape):
        return mask
    out = torch.zeros(field_shape, dtype=mask.dtype, device=mask.device)
    offs = [(f - m) // 2 for f, m in zip(field_shape, mask.shape, strict=True)]
    if any(o < 0 for o in offs):
        raise ValueError(f"mask {tuple(mask.shape)} is larger than the field {tuple(field_shape)}")
    z, y, x = offs
    nz, ny, nx = mask.shape
    out[z : z + nz, y : y + ny, x : x + nx] = mask
    return out


def regularity_verdict(
    qc: WarpQC,
    max_neg_voxels: int = DEFAULT_MAX_NEG_VOXELS,
    max_neg_frac: float = DEFAULT_MAX_NEG_FRAC,
    min_jac: float = DEFAULT_MIN_JAC,
    max_jac: float = DEFAULT_MAX_JAC,
    marginal_neg_frac: float = DEFAULT_MARGINAL_NEG_FRAC,
) -> tuple[str, list[str]]:
    """Grade a warp's regularity: returns (``pass``/``marginal``/``fail``, reasons).

    Three grades rather than two, because the interesting case is a candidate
    that wins on similarity and folds *slightly*. Discarding it loses a good
    result; accepting it silently ships a defect. Marginal says: this is the best
    thing we have seen and it is nearly sound — push regularization up a notch
    and re-fit, rather than throwing it away.

    That is the general shape of these failures. Every one of them is fixed by a
    knob the search is already walking, so a bad grade is a **direction**, not
    just a veto — see :data:`REMEDY`.

    The Jacobian bounds are checked at the 1st/99th percentile rather than the
    extremes, so one pathological voxel does not veto an otherwise sound warp,
    while a genuinely over-warped field — which distorts whole regions — moves
    the percentiles.
    """
    reasons: list[str] = []
    grade = PASS

    neg_budget = max(max_neg_voxels, max_neg_frac * qc.n_voxels)
    if qc.jac_neg_count > neg_budget:
        reasons.append(
            f"folding: {qc.jac_neg_count} voxels ({100 * qc.jac_neg_frac:.3f}%) "
            f"have det(J) <= 0, budget {neg_budget:.0f}"
        )
        # Localised folding is recoverable; widespread folding is not.
        grade = MARGINAL if qc.jac_neg_frac <= marginal_neg_frac else FAIL

    # A Jacobian percentile out of bounds is systematic distortion across whole
    # regions, not a local defect, so it is never merely marginal.
    if qc.jac_p01 < min_jac:
        reasons.append(f"over-compression: 1st pct det(J) = {qc.jac_p01:.3f} < {min_jac}")
        grade = FAIL
    if qc.jac_p99 > max_jac:
        reasons.append(f"over-expansion: 99th pct det(J) = {qc.jac_p99:.3f} > {max_jac}")
        grade = FAIL

    return grade, reasons


def remedies(reasons: list[str]) -> list[str]:
    """Map failure reasons to the knob that fixes each one."""
    out = []
    for r in reasons:
        key = r.split(":", 1)[0]
        if key in REMEDY and REMEDY[key] not in out:
            out.append(REMEDY[key])
    return out


def format_warpqc(qc: WarpQC, name: str = "") -> str:
    """One-block human summary."""
    grade, reasons = regularity_verdict(qc)
    head = f"{name}\n" if name else ""
    verdict = grade.upper() if not reasons else f"{grade.upper()} — {'; '.join(reasons)}"
    fix = remedies(reasons)
    lines = [
        f"{head}  det(J)      min={qc.jac_min:.3f}  1%={qc.jac_p01:.3f}  "
        f"50%={qc.jac_p50:.3f}  99%={qc.jac_p99:.3f}  max={qc.jac_max:.3f}",
        f"  folded      {qc.jac_neg_count} voxels ({100 * qc.jac_neg_frac:.4f}% of {qc.n_voxels})",
        f"  |disp| mm   mean={qc.disp_mean_mm:.2f}  99%={qc.disp_p99_mm:.2f}  "
        f"max={qc.disp_max_mm:.2f}",
        f"  bending     {qc.bending_energy:.5g}",
        f"  verdict     {verdict}",
    ]
    if fix:
        lines.append(f"  try         {'; '.join(fix)}")
    return "\n".join(lines)
