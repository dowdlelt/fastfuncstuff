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

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor

from .formwarp import FOLD_GUARD_FLOOR
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

# Saturation for regularity_margin(), in log units. Real margins live within a
# couple of units of zero, so this is "off the scale" in either direction while
# staying finite — a searcher regressing on these needs a number for a fit that
# exploded, not a NaN.
MARGIN_LIMIT = 10.0
FAILED_MARGIN = -MARGIN_LIMIT  # a fit that produced no field at all
UNCONSTRAINED_MARGIN = MARGIN_LIMIT  # a result with no field to fold
DEFAULT_MAX_JAC = 4.0  # local expansion beyond this is implausible
DEFAULT_MIN_JAC = 0.25  # ... and likewise compression

# Folding above the "handful" budget but below this fraction is MARGINAL rather
# than FAIL: a small, localised fold in an otherwise sound field is a warp that
# was pushed slightly too hard, not a broken one — and the fix is known (more
# regularization), so throwing the candidate away discards both a good result
# and the information about which way to move. See `regularity_verdict`.
DEFAULT_MARGINAL_NEG_FRAC = 5e-3  # 0.5% of in-mask voxels

PASS, MARGINAL, FAIL = "pass", "marginal", "fail"

# A returned warp whose min det(J) is within this factor of the floor did not
# merely come close to folding -- it is sitting ON the guard, which means the
# damping is the only thing holding it together.
GUARD_PINNED_RATIO = 1.10


def guard_pinned(qc: WarpQC, floor: float = FOLD_GUARD_FLOOR) -> bool:
    """True when the anti-fold damping, not the regularization, is what saved this warp.

    The most useful diagnostic in the whole QC, and the one that took longest to
    see, because it looks like an ordinary number. ``jac_min = 0.050`` reads as "a
    warp with a small minimum Jacobian"; it actually means the field spent the run
    trying to invert and was clamped every time. The solver *targets* this value,
    so landing on it is not a coincidence -- it is the signature of a config whose
    regularization is too low, running on the guard.

    Measured on a 7T epi2epi run: the five configs pinned on every subject averaged
    a bending energy of 1.32, against 0.021 for the 122 that never approached the
    floor -- 64x the roughness for 0.05 of lncc. And the mechanism is not
    theoretical: one pinned config met a subject where the guard could not hold,
    reported NO LEGAL ITERATE, and folded outright.

    This is what lets the report demote those configs without inventing a
    roughness threshold. The floor is already a declared parameter of the solver;
    all this does is notice when a result is resting on it.
    """
    return qc.jac_min < floor * GUARD_PINNED_RATIO


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


def pad_mask_to_field(
    mask: Tensor,
    field_shape: tuple[int, int, int],
    *,
    mask_affine: np.ndarray | None = None,
    field_affine: np.ndarray | None = None,
    lower_padding_xyz: tuple[int, int, int] | None = None,
) -> Tensor:
    """Place a base-grid mask inside a larger padded warp grid.

    Prefer affine-derived placement for saved fields, or explicit lower-face
    padding for in-memory fields. Centre placement remains only for legacy
    symmetric warps whose geometry is unavailable.
    """
    if tuple(mask.shape) == tuple(field_shape):
        return mask
    out = torch.zeros(field_shape, dtype=mask.dtype, device=mask.device)
    if mask_affine is not None and field_affine is not None:
        origin = np.linalg.solve(field_affine, mask_affine @ np.array([0.0, 0.0, 0.0, 1.0]))
        ox, oy, oz = (int(round(v)) for v in origin[:3])
        offs = [oz, oy, ox]
    elif lower_padding_xyz is not None:
        ox, oy, oz = lower_padding_xyz
        offs = [oz, oy, ox]
    else:
        offs = [(f - m) // 2 for f, m in zip(field_shape, mask.shape, strict=True)]
    if any(o < 0 for o in offs):
        raise ValueError(f"mask {tuple(mask.shape)} is larger than the field {tuple(field_shape)}")
    z, y, x = offs
    nz, ny, nx = mask.shape
    if z + nz > field_shape[0] or y + ny > field_shape[1] or x + nx > field_shape[2]:
        raise ValueError(
            f"mask {tuple(mask.shape)} at offset {(z, y, x)} exceeds field {field_shape}"
        )
    out[z : z + nz, y : y + ny, x : x + nx] = mask
    return out


def regularity_verdict(
    qc: WarpQC,
    max_neg_voxels: int = DEFAULT_MAX_NEG_VOXELS,
    max_neg_frac: float = DEFAULT_MAX_NEG_FRAC,
    marginal_neg_frac: float = DEFAULT_MARGINAL_NEG_FRAC,
) -> tuple[str, list[str]]:
    """Grade a warp's regularity: returns (``pass``/``marginal``/``fail``, reasons).

    Three grades rather than two, because the interesting case is a candidate
    that wins on similarity and folds *slightly*. Discarding it loses a good
    result; accepting it silently ships a defect. Marginal says: this is the best
    thing we have seen and it is nearly sound — push regularization up a notch
    and re-fit, rather than throwing it away.

    **Only folding can fail.** The dividing line is whether a criterion states
    something anatomy cannot do or merely something it rarely does. Tissue cannot
    turn inside out, so a folded field is wrong whatever it scores. Everything
    else here is a threshold on a continuum with real anatomical variation on both
    sides, and a threshold like that earns a caution, not a veto.

    That is the general shape of these failures. Every one of them is fixed by a
    knob the search is already walking, so a bad grade is a **direction**, not
    just a veto — see :data:`REMEDY`. For the bounds that no longer grade, see
    :func:`regularity_cautions`.
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

    return grade, reasons


def regularity_cautions(
    qc: WarpQC,
    min_jac: float = DEFAULT_MIN_JAC,
    max_jac: float = DEFAULT_MAX_JAC,
) -> list[str]:
    """Things worth saying about a warp that do not make it wrong.

    Extreme-but-positive Jacobians live here rather than in the verdict. A det(J)
    of 0.2 is *unusual*, not impossible: heads are different shapes and ventricles
    vary enormously between people, so a fifth of the volume across one percent of
    the brain is plausible anatomy meeting a constant nobody derived. Measured on
    T1->MNI, this bound alone condemned 67 of 435 warps that had not a single
    folded voxel between them -- and the settings it condemned were the ones that
    matched the template best.

    Saying it out loud still matters. The field is working hard somewhere, the
    percentile makes that regional rather than one bad voxel, and it is the
    direction the search is being pulled in. It just does not get a veto.
    """
    out = []
    if guard_pinned(qc):
        out.append(
            f"guard-limited: min det(J) = {qc.jac_min:.3f} sits on the solver's "
            f"fold-guard floor ({FOLD_GUARD_FLOOR}) -- the damping is holding this "
            "warp together, not its regularization"
        )
    if qc.jac_p01 < min_jac:
        out.append(f"over-compression: 1st pct det(J) = {qc.jac_p01:.3f} < {min_jac}")
    if qc.jac_p99 > max_jac:
        out.append(f"over-expansion: 99th pct det(J) = {qc.jac_p99:.3f} > {max_jac}")
    return out


def gate_margin(
    qc: WarpQC,
    max_neg_voxels: int = DEFAULT_MAX_NEG_VOXELS,
    max_neg_frac: float = DEFAULT_MAX_NEG_FRAC,
) -> float:
    """Distance to the only thing that can actually fail a warp: folding.

    Deliberately narrower than :func:`regularity_margin`, and the two answer
    different questions. That one includes the Jacobian cautions because the
    *searcher* needs to feel them coming; this one includes only the criterion
    that decides pass/fail, because a config's clearance has to be measured
    against the line it can actually be rejected at. Measuring clearance on a
    criterion that no longer gates would demote exactly the warps that were just
    ruled acceptable.

    A field with nothing folded is as clear as clear gets, so it returns the
    saturation value rather than a number that would invite comparing two clean
    warps on how nearly they folded.
    """
    if qc.jac_neg_count == 0:
        return UNCONSTRAINED_MARGIN
    neg_budget = max(max_neg_voxels, max_neg_frac * qc.n_voxels)
    return max(-MARGIN_LIMIT, math.log((neg_budget + 1.0) / (qc.jac_neg_count + 1.0)))


def regularity_margin(
    qc: WarpQC,
    max_neg_voxels: int = DEFAULT_MAX_NEG_VOXELS,
    max_neg_frac: float = DEFAULT_MAX_NEG_FRAC,
    min_jac: float = DEFAULT_MIN_JAC,
    max_jac: float = DEFAULT_MAX_JAC,
) -> float:
    """How much room a warp has before it trips a criterion: > 0 is clear of all of
    them, <= 0 is not.

    Note this is *not* the same line as :func:`regularity_verdict`'s pass/fail:
    only folding fails there, while the margin keeps measuring distance to the
    Jacobian bounds too. A field can be MARGINAL for over-compression and still be
    the one to ship. Kept that way on purpose — the searcher needs to feel the
    bounds approaching to steer away from them, and dropping them from the margin
    would make it blind to exactly the direction the cautions warn about.

    :func:`regularity_verdict` answers the question a user asks; this answers the
    question a *searcher* asks. A pass/fail label says only which side of the
    boundary a config landed on, so a search steering by it is blind until it
    trips — and the boundary is a cliff (measured: 100% folding at
    ``total_sigma=0.5``, 0% at 1.0), so by the time the label changes the useful
    gradient is gone.

    Each criterion becomes a log ratio of achieved-to-allowed, and the margin is
    the tightest of them. Logs because these quantities span orders of magnitude
    and because it makes "twice the budget" and "half the budget" symmetric
    distances, which is what a surrogate wants to interpolate over.
    """
    neg_budget = max(max_neg_voxels, max_neg_frac * qc.n_voxels)
    margins = [
        math.log((neg_budget + 1.0) / (qc.jac_neg_count + 1.0)),
        # A non-positive Jacobian percentile is not "slightly out of bounds", it
        # is inside-out tissue; the log is undefined and the answer is "very bad".
        math.log(qc.jac_p01 / min_jac) if qc.jac_p01 > 0 else -MARGIN_LIMIT,
        math.log(max_jac / qc.jac_p99) if qc.jac_p99 > 0 else -MARGIN_LIMIT,
    ]
    return max(-MARGIN_LIMIT, min(margins))


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
