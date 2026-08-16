"""One registry of image-similarity metrics, for optimising *and* for judging.

Before this, a metric's identity was split across the codebase: the 14 AFNI cost
functionals lived in :mod:`allcost` and could only be evaluated, the engines each
kept their own short list of things they could optimise (``cc``, ``mse``, ``lpa``,
``lpc``, ``pearson``), and NGF existed only inside ``ffs_bbr`` as a point-cloud
cost. The same idea therefore had two or three spellings, and which of them a
given tool could reach depended on which module it happened to import.

The consequence was not cosmetic. ``cc`` — local normalised cross-correlation,
the ANTs SyN default — was what ``formwarp`` *optimised with*, and it was not in
the panel that judged the result. The tuner could drive a backend with a metric
it was then unable to score.

So a metric is declared once here, with the properties that decide where it may
be used:

* **contrast** — the modality regimes it is *meaningful* in. This is the tag that
  earns its keep. Signed measures like ``lpc`` mean "more anti-correlated is
  better", which is right across a contrast inversion and actively wrong between
  two T1s, where they rank the worst warp first (measured at rho = -0.99 against
  every unsigned functional).
* **family** — what it is computed from. Excluding a metric from a jury has to
  exclude its whole family, because "different name" is not "different evidence".
* **differentiable** — whether an optimiser can descend on it, not merely score it.
* **needs_grid** — whether it needs a 3-D volume or works on flat in-mask vectors.
  The AFNI functionals are scattered-point measures; neighbourhood descriptors are
  not, and cannot be evaluated on a bag of voxels.

Why the new ones are worth having: on a real T1->MNI search, all eleven unsigned
AFNI functionals agreed at rank correlation >= 0.96. Fourteen judges were one
judge counted eleven times. LNCC, NGF and MIND are built from *structurally*
different evidence — windowed covariance, gradient orientation, and neighbourhood
self-similarity — so they are the first additions that can actually disagree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .cost import _separable_smooth_3d

_EPS = 1e-6

SAME = "same"
CROSS = "cross"
CONTRAST_REGIMES = (SAME, CROSS)


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    """One similarity measure and the facts that decide where it may be used.

    ``lower_is_better`` is not configurable: every metric in this project reports
    in AFNI's convention so that any two can be compared without a sign table.
    Implementations convert once, at the point of definition.
    """

    name: str
    long_name: str
    describe: str
    family: str  # correlation | histogram | localpearson | windowed | gradient | descriptor
    contrast: tuple[str, ...] = (SAME, CROSS)
    signed: bool = False  # rewards anti-correlation; only meaningful cross-modal
    needs_grid: bool = False  # needs a volume, not flat in-mask values
    differentiable: bool = False  # an optimiser can descend on it
    afni: bool = False  # part of the 3dAllineate set, evaluated via allcost

    def usable_for(self, contrast: str) -> bool:
        return contrast in self.contrast


# The 3dAllineate set. Kept exactly as AFNI defines it — these are the numbers
# `ffs_util_cost` prints and they have to stay comparable to `3dAllineate
# -allcostX`. Tags added here; the arithmetic still lives in allcost.py.
_AFNI: list[Metric] = [
    Metric("ls", "leastsq", "1 - |Pearson r|", "correlation", afni=True),
    Metric("sp", "spearman", "1 - |Spearman rho|", "correlation", afni=True),
    Metric("mi", "mutualinfo", "-Mutual information", "histogram", afni=True),
    Metric("crM", "corratio_mul", "1 - |CR(b,s)*CR(s,b)|", "histogram", afni=True),
    Metric("nmi", "norm_mutualinfo", "H(b,s)/[H(b)+H(s)]", "histogram", afni=True),
    Metric("je", "jointentropy", "Joint entropy H(b,s)", "histogram", afni=True),
    Metric("hel", "hellinger", "-Hellinger distance", "histogram", afni=True),
    Metric("crA", "corratio_add", "1 - |CR(b,s)+CR(s,b)|", "histogram", afni=True),
    Metric("crU", "corratio_uns", "1 - |CR(s,b)|", "histogram", afni=True),
    Metric(
        "lss",
        "signedPcor",
        "Signed Pearson r (rewards anti-correlation)",
        "correlation",
        contrast=(CROSS,),
        signed=True,
        afni=True,
    ),
    Metric(
        "lpc",
        "localPcorSigned",
        "Local Pearson, signed",
        "localpearson",
        contrast=(CROSS,),
        signed=True,
        afni=True,
    ),
    Metric("lpa", "localPcorAbs", "1 - |local Pearson|", "localpearson", afni=True),
    Metric(
        "lpc+",
        "localPcor+Others",
        "lpc + weighted hel/mi/nmi/crA",
        "localpearson",
        contrast=(CROSS,),
        signed=True,
        afni=True,
    ),
    Metric(
        "lpa+", "localPcorAbs+Others", "lpa + weighted hel/mi/nmi/crA", "localpearson", afni=True
    ),
]

# Everything AFNI does not have. All grid-based and all differentiable, so they
# serve as optimiser objectives as well as judges — which is the point of one
# registry rather than two lists.
_EXTRA: list[Metric] = [
    Metric(
        "lncc",
        "localNCC",
        "1 - local normalised cross-correlation over a (2r+1)^3 box",
        "windowed",
        needs_grid=True,
        differentiable=True,
    ),
    Metric(
        "mse",
        "meansquare",
        "Weighted mean squared intensity difference",
        "intensity",
        contrast=(SAME,),
        needs_grid=True,
        differentiable=True,
    ),
    Metric(
        "ngf",
        "normgradfield",
        "1 - mean squared normalised-gradient alignment (edge orientation only)",
        "gradient",
        needs_grid=True,
        differentiable=True,
    ),
    Metric(
        "mind",
        "MIND",
        "SSD between modality-independent neighbourhood descriptors",
        "descriptor",
        needs_grid=True,
        differentiable=True,
    ),
    Metric(
        "mindssc",
        "MIND-SSC",
        "SSD between self-similarity-context descriptors (12-element MIND variant)",
        "descriptor",
        needs_grid=True,
        differentiable=True,
    ),
]

METRICS: dict[str, Metric] = {m.name: m for m in (*_AFNI, *_EXTRA)}

ALL_METRICS = list(METRICS)
AFNI_METRICS = [m.name for m in _AFNI]
GRID_METRICS = [m.name for m in _EXTRA]


def metric(name: str) -> Metric:
    if name not in METRICS:
        raise ValueError(f"unknown metric {name!r}; have {', '.join(ALL_METRICS)}")
    return METRICS[name]


def differentiable_metrics(contrast: str | None = None) -> list[str]:
    """Metrics an engine can optimise, optionally restricted to a contrast regime."""
    return [
        n
        for n, m in METRICS.items()
        if m.differentiable and (contrast is None or m.usable_for(contrast))
    ]


def panel_for(
    optimized: str | None = None,
    contrast: str = SAME,
    exclude: Sequence[str] = (),
    grid: bool = True,
) -> list[str]:
    """The metrics allowed to judge a fit produced under these conditions.

    Three filters, each of which changed an answer on real data:

    * **Contrast.** A metric is dropped unless it is meaningful in this regime.
      That is what keeps the signed measures out of a same-modality jury, where
      they rank the worst warp first.
    * **Family.** The optimised metric is excluded along with everything computed
      from the same quantity, so a cost cannot vote on its own fit through a
      sibling — ``lpa`` and ``lpa+`` are one number at rank correlation 1.00.
    * **Grid availability.** Neighbourhood metrics need a volume; a caller with
      only scattered in-mask values passes ``grid=False`` and gets the rest.
    """
    if contrast not in CONTRAST_REGIMES:
        raise ValueError(f"contrast must be one of {CONTRAST_REGIMES}, got {contrast!r}")

    drop = set(exclude)
    if optimized:
        family = metric(optimized).family
        drop |= {n for n, m in METRICS.items() if m.family == family}

    panel = [
        n
        for n, m in METRICS.items()
        if n not in drop and m.usable_for(contrast) and (grid or not m.needs_grid)
    ]
    if not panel:
        raise ValueError(
            f"empty judging panel: optimizing {optimized!r} under contrast "
            f"{contrast!r} excludes every metric"
        )
    return panel


def describe_metrics(contrast: str | None = None) -> str:
    """A table of what exists and where each one applies."""
    lines = [
        f"  {'name':9s} {'family':13s} {'contrast':11s} {'opt':4s} {'grid':5s} description",
        "  " + "-" * 96,
    ]
    for name, m in METRICS.items():
        if contrast and not m.usable_for(contrast):
            continue
        lines.append(
            f"  {name:9s} {m.family:13s} {'/'.join(m.contrast):11s} "
            f"{'yes' if m.differentiable else '-':4s} "
            f"{'yes' if m.needs_grid else '-':5s} {m.describe}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Implementations of the non-AFNI metrics
# ---------------------------------------------------------------------------


def _grad3(vol: Tensor) -> Tensor:
    """(3, nz, ny, nx) central-difference gradient, replicating at the faces."""
    out = []
    for dim in range(3):
        g = torch.zeros_like(vol)
        sl = [slice(None)] * 3
        lo, hi = list(sl), list(sl)
        lo[dim], hi[dim] = slice(0, -2), slice(2, None)
        mid = list(sl)
        mid[dim] = slice(1, -1)
        g[tuple(mid)] = 0.5 * (vol[tuple(hi)] - vol[tuple(lo)])
        out.append(g)
    return torch.stack(out, dim=0)


def ngf_volume_cost(
    base: Tensor, moving: Tensor, weight: Tensor | None = None, eta: float | None = None
) -> Tensor:
    """Dense normalised-gradient-field cost (Haber & Modersitzki), lower = better.

    Compares only the *orientation* of intensity change, through
    ``(grad_a . grad_b)^2 / (|grad_a|^2_eta |grad_b|^2_eta)``, which is why it
    survives a contrast inversion: flipping a modality negates both gradients and
    the square is unchanged. The ``eta`` floor is what stops flat noisy regions
    from contributing a full-strength random orientation.

    Differs from :func:`bbr.ngf_cost`, which scores a *point cloud* of extracted
    anatomical edges against one gradient field. This is the volume-to-volume form
    the registry needs, where neither side has been reduced to edges first.
    """
    ga, gb = _grad3(base), _grad3(moving)
    if eta is None:
        # Scale from the data rather than a constant: the images may be in any
        # units, and a fixed floor would silently become "ignore everything" or
        # "trust everything" depending on the scaling.
        mag = torch.sqrt((ga * ga).sum(0))
        pos = mag[mag > 0]
        eta = float(pos.median()) if pos.numel() else 1.0
    e2 = float(eta) ** 2

    dot = (ga * gb).sum(0)
    na = (ga * ga).sum(0) + e2
    nb = (gb * gb).sum(0) + e2
    aligned = (dot * dot) / (na * nb)

    if weight is None:
        return 1.0 - aligned.mean()
    w = weight.clamp(min=0)
    return 1.0 - (w * aligned).sum() / w.sum().clamp(min=_EPS)


# The six-neighbourhood MIND uses, as (dz, dy, dx) voxel offsets.
_MIND_SIX = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

# The twelve SSC pairs: every pair of six-neighbours at Euclidean distance sqrt(2),
# i.e. the ones sharing a face-diagonal. SSC measures neighbour-to-neighbour rather
# than centre-to-neighbour, which is what removes the centre patch's noise from
# every element of the descriptor.
_SSC_PAIRS = tuple(
    (i, j)
    for i in range(6)
    for j in range(i + 1, 6)
    if sum((_MIND_SIX[i][k] - _MIND_SIX[j][k]) ** 2 for k in range(3)) == 2
)


def _shift(vol: Tensor, off: tuple[int, int, int]) -> Tensor:
    """Translate by whole voxels, replicating the edge rather than wrapping."""
    out = vol
    for dim, delta in enumerate(off):
        if delta:
            out = torch.roll(out, shifts=delta, dims=dim)
            idx = [slice(None)] * 3
            idx[dim] = slice(0, delta) if delta > 0 else slice(delta, None)
            edge = [slice(None)] * 3
            edge[dim] = slice(delta, delta + 1) if delta > 0 else slice(delta - 1, delta)
            out = out.clone()
            out[tuple(idx)] = out[tuple(edge)]
    return out


def _patch_ssd(vol: Tensor, off: tuple[int, int, int], radius: float) -> Tensor:
    """Mean squared difference between a patch at x and the patch at x+off."""
    d = vol - _shift(vol, off)
    return _separable_smooth_3d(d * d, radius, kernel_type="box")


def mind_descriptor(vol: Tensor, radius: float = 1.0, ssc: bool = False) -> Tensor:
    """(C, nz, ny, nx) MIND or MIND-SSC descriptor field.

    Each channel is ``exp(-Dp / V)`` for one neighbour offset, where ``Dp`` is the
    patch distance in that direction and ``V`` the local variance estimate. The
    descriptor is normalised by its own per-voxel maximum, which is what makes it
    modality independent: it encodes the *shape* of the local self-similarity, and
    any monotonic intensity remapping leaves that shape intact.

    ``ssc=True`` measures between pairs of neighbours instead of from the centre
    outward. That drops the centre patch — and hence the centre voxel's noise —
    out of every channel, which is the improvement SSC exists for.
    """
    dists = [_patch_ssd(vol, off, radius) for off in _MIND_SIX]
    if ssc:
        # Distance between two neighbour patches, via the patch centred between
        # them; the difference of the two shifted volumes is exactly that.
        chans = []
        for i, j in _SSC_PAIRS:
            d = _shift(vol, _MIND_SIX[i]) - _shift(vol, _MIND_SIX[j])
            chans.append(_separable_smooth_3d(d * d, radius, kernel_type="box"))
    else:
        chans = dists

    stack = torch.stack(chans, dim=0)
    # Variance estimate from the six-neighbourhood, floored so that a perfectly
    # flat region yields a finite (and uninformative) descriptor rather than NaN.
    var = torch.stack(dists, dim=0).mean(dim=0).clamp(min=_EPS)
    desc = torch.exp(-stack / var)
    return desc / desc.max(dim=0, keepdim=True).values.clamp(min=_EPS)


def mind_cost(
    base: Tensor,
    moving: Tensor,
    weight: Tensor | None = None,
    radius: float = 1.0,
    ssc: bool = False,
) -> Tensor:
    """Mean absolute difference between MIND descriptors, lower = better.

    The standard modality-independent measure for *deformable* multimodal
    registration. Intensity-statistic measures (MI, correlation ratio) compare
    global histograms and carry no local structural information, which is why they
    suit affine alignment better than deformation; MIND turns each image into a
    field of local self-similarity and compares those pointwise.
    """
    da = mind_descriptor(base, radius, ssc)
    db = mind_descriptor(moving, radius, ssc)
    diff = (da - db).abs().mean(dim=0)
    if weight is None:
        return diff.mean()
    w = weight.clamp(min=0)
    return (w * diff).sum() / w.sum().clamp(min=_EPS)


# ---------------------------------------------------------------------------
# One evaluation surface
# ---------------------------------------------------------------------------


@dataclass
class MetricInputs:
    """Everything any metric here might need, on a common grid.

    Carrying the volumes rather than only the flat in-mask values is what lets one
    call serve both families. The AFNI functionals are computed from scattered
    points and the neighbourhood metrics from the grid; asking the caller to
    prepare each separately is how the two lists drifted apart in the first place.
    """

    base: Tensor  # (nz, ny, nx)
    moving: Tensor  # (nz, ny, nx), already resampled onto the base grid
    weight: Tensor | None = None  # (nz, ny, nx)
    voxdims: tuple[float, float, float] = (1.0, 1.0, 1.0)
    overlap: float | None = None
    cc_radius: int = 4
    mind_radius: float = 1.0
    ngf_eta: float | None = None
    bloktype: str = "tohd"
    extra: dict[str, Any] = field(default_factory=dict)


def evaluate_metrics(inp: MetricInputs, names: Sequence[str] | None = None) -> dict[str, float]:
    """Evaluate any mix of AFNI and grid metrics in one call, AFNI sign convention.

    The AFNI names are delegated to :mod:`allcost` rather than reimplemented, so
    the numbers stay bit-comparable with ``3dAllineate -allcostX`` and with every
    result already recorded.
    """
    want = list(names) if names is not None else list(ALL_METRICS)
    unknown = [n for n in want if n not in METRICS]
    if unknown:
        raise ValueError(f"unknown metric(s): {', '.join(unknown)}")

    out: dict[str, float] = {}

    afni_want = [n for n in want if METRICS[n].afni]
    if afni_want:
        from .allcost import build_cost_inputs, evaluate_all_costs

        cost_inp = build_cost_inputs(
            inp.base,
            inp.moving,
            inp.weight,
            inp.voxdims,
            inp.overlap if inp.overlap is not None else 1.0,
            inp.bloktype,
        )
        out.update(evaluate_all_costs(cost_inp, costs=afni_want))
        del cost_inp

    for name in want:
        if METRICS[name].afni:
            continue
        out[name] = float(_grid_metric(name, inp))
    return {n: out[n] for n in want}


def _grid_metric(name: str, inp: MetricInputs) -> Tensor:
    w = inp.weight if inp.weight is not None else torch.ones_like(inp.base)
    if name == "lncc":
        from .formwarp import _local_cc_cost

        return _local_cc_cost(inp.base, inp.moving, inp.cc_radius, w)
    if name == "mse":
        return (w * (inp.base - inp.moving) ** 2).sum() / w.sum().clamp(min=_EPS)
    if name == "ngf":
        return ngf_volume_cost(inp.base, inp.moving, inp.weight, inp.ngf_eta)
    if name in ("mind", "mindssc"):
        return mind_cost(inp.base, inp.moving, inp.weight, inp.mind_radius, ssc=(name == "mindssc"))
    raise ValueError(f"no grid implementation for {name!r}")


def differentiable_cost(
    name: str,
    base: Tensor,
    moving: Tensor,
    weight: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """A metric as a differentiable scalar, for use as an optimiser objective.

    The other half of "one surface": the same declaration that says a metric may
    judge a result also says whether an engine may descend on it, and this is
    where that promise is honoured.
    """
    spec = metric(name)
    if not spec.differentiable:
        raise ValueError(
            f"{name} is not differentiable; optimisable metrics are "
            f"{', '.join(differentiable_metrics())}"
        )
    inp = MetricInputs(base=base, moving=moving, weight=weight, **kwargs)
    return _grid_metric(name, inp)


def check_contrast(name: str, contrast: str) -> None:
    """Raise if a metric is being used in a regime where it is meaningless.

    Worth failing loudly over: a signed metric on same-modality data does not
    degrade gracefully, it optimises for the wrong answer.
    """
    spec = metric(name)
    if not spec.usable_for(contrast):
        raise ValueError(
            f"metric {name!r} is not meaningful for {contrast}-modality data "
            f"(it applies to: {', '.join(spec.contrast)}). "
            + (
                "Signed measures reward anti-correlation, which is right across a "
                "contrast inversion and wrong between two images of the same type."
                if spec.signed
                else ""
            )
        )


__all__ = [
    "AFNI_METRICS",
    "PATCH_METRICS",
    "batched_patch_cost",
    "ALL_METRICS",
    "CONTRAST_REGIMES",
    "CROSS",
    "GRID_METRICS",
    "METRICS",
    "SAME",
    "Metric",
    "MetricInputs",
    "check_contrast",
    "describe_metrics",
    "differentiable_cost",
    "differentiable_metrics",
    "evaluate_metrics",
    "metric",
    "mind_cost",
    "mind_descriptor",
    "ngf_volume_cost",
    "panel_for",
]


# ---------------------------------------------------------------------------
# Patch-wise forms, so the qwarp engine can optimise these too
# ---------------------------------------------------------------------------
#
# qwarp optimises shrinking overlapping patches rather than a whole field, and its
# cost is evaluated on flat ``(B, V)`` patch vectors. The neighbourhood metrics
# need the 3-D structure back, which is available: a patch is an ``(nzh, nyh, nxh)``
# block that was flattened, so it can simply be reshaped. Everything below returns
# **higher = better**, matching what the engine's other patch costs return before it
# negates them.

PATCH_METRICS = ("lncc", "mse", "ngf", "mind", "mindssc")


def _shift_batched(v: Tensor, off: tuple[int, int, int]) -> Tensor:
    """Whole-voxel translate of a (B, 1, D, H, W) block, replicating the edge."""
    out = v
    for axis, delta in enumerate(off):
        if not delta:
            continue
        dim = axis + 2
        out = torch.roll(out, shifts=delta, dims=dim)
        idx: list = [slice(None)] * 5
        edge: list = [slice(None)] * 5
        if delta > 0:
            idx[dim], edge[dim] = slice(0, delta), slice(delta, delta + 1)
        else:
            idx[dim], edge[dim] = slice(delta, None), slice(delta - 1, delta)
        out = out.clone()
        out[tuple(idx)] = out[tuple(edge)]
    return out


def _grad_batched(v: Tensor) -> Tensor:
    """(B, 3, D, H, W) central-difference gradient of a (B, 1, D, H, W) block."""
    comps = []
    for axis in range(3):
        hi = _shift_batched(v, tuple(-1 if a == axis else 0 for a in range(3)))
        lo = _shift_batched(v, tuple(1 if a == axis else 0 for a in range(3)))
        comps.append(0.5 * (hi - lo))
    return torch.cat(comps, dim=1)


def batched_patch_cost(
    name: str,
    base_patches: Tensor,
    warped_patches: Tensor,
    weight_patches: Tensor,
    nzh: int,
    nyh: int,
    nxh: int,
    cc_radius: int = 4,
    mind_radius: float = 1.0,
    ngf_eta: float | None = None,
) -> Tensor:
    """One grid metric evaluated per patch: ``(B, V)`` in, ``(B,)`` out.

    Higher is better, so the caller negates it exactly as it does for the local
    Pearson costs.
    """
    from .cost import _batched_separable_smooth_3d, _make_kernel_1d

    b = base_patches.shape[0]
    x = base_patches.reshape(b, 1, nzh, nyh, nxh)
    y = warped_patches.reshape(b, 1, nzh, nyh, nxh)
    w = weight_patches.reshape(b, 1, nzh, nyh, nxh).clamp(min=0)
    wsum = w.sum(dim=(1, 2, 3, 4)).clamp(min=_EPS)

    if name == "mse":
        return -(w * (x - y) ** 2).sum(dim=(1, 2, 3, 4)) / wsum

    if name == "lncc":
        # The window has to fit inside the patch, or every voxel sees the same
        # (whole-patch) statistics and the local metric silently becomes a global
        # one -- which is precisely the information qwarp's small patches carry.
        radius = max(1.0, min(float(cc_radius), (min(nzh, nyh, nxh) - 1) / 2.0))
        kernel = _make_kernel_1d("box", radius, base_patches.device)

        def sm(v: Tensor) -> Tensor:
            return _batched_separable_smooth_3d(v, kernel)

        sw = sm(w).clamp(min=_EPS)
        mx, my = sm(w * x) / sw, sm(w * y) / sw
        vxx = (sm(w * x * x) / sw - mx * mx).clamp(min=_EPS)
        vyy = (sm(w * y * y) / sw - my * my).clamp(min=_EPS)
        vxy = sm(w * x * y) / sw - mx * my
        local = (vxy * vxy) / (vxx * vyy)
        return (w * local).sum(dim=(1, 2, 3, 4)) / wsum

    if name == "ngf":
        ga, gb = _grad_batched(x), _grad_batched(y)
        if ngf_eta is None:
            mag = ga.pow(2).sum(dim=1, keepdim=True).sqrt()
            eta = mag.flatten(1).median(dim=1).values.clamp(min=_EPS).view(b, 1, 1, 1, 1)
        else:
            eta = torch.full((b, 1, 1, 1, 1), float(ngf_eta), device=x.device)
        e2 = eta * eta
        dot = (ga * gb).sum(dim=1, keepdim=True)
        na = ga.pow(2).sum(dim=1, keepdim=True) + e2
        nb = gb.pow(2).sum(dim=1, keepdim=True) + e2
        return (w * (dot * dot) / (na * nb)).sum(dim=(1, 2, 3, 4)) / wsum

    if name in ("mind", "mindssc"):
        kernel = _make_kernel_1d("box", float(mind_radius), base_patches.device)

        def desc(v: Tensor) -> Tensor:
            dists = [
                _batched_separable_smooth_3d((v - _shift_batched(v, o)) ** 2, kernel)
                for o in _MIND_SIX
            ]
            if name == "mindssc":
                chans = [
                    _batched_separable_smooth_3d(
                        (_shift_batched(v, _MIND_SIX[i]) - _shift_batched(v, _MIND_SIX[j])) ** 2,
                        kernel,
                    )
                    for i, j in _SSC_PAIRS
                ]
            else:
                chans = dists
            stack = torch.cat(chans, dim=1)
            var = torch.cat(dists, dim=1).mean(dim=1, keepdim=True).clamp(min=_EPS)
            d = torch.exp(-stack / var)
            return d / d.max(dim=1, keepdim=True).values.clamp(min=_EPS)

        # The base descriptor is constant -- the base does not move -- so it is
        # built once under no_grad. Keeping it in the graph doubled peak memory and
        # bought nothing; MIND-SSC's twelve channels put a 193^3 run out of memory.
        with torch.no_grad():
            dx = desc(x)
        diff = (dx - desc(y)).abs().mean(dim=1, keepdim=True)
        del dx
        return -(w * diff).sum(dim=(1, 2, 3, 4)) / wsum

    raise ValueError(f"{name!r} has no patch-wise form; have {', '.join(PATCH_METRICS)}")
