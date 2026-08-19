"""GPU-accelerated affine/rigid alignment engine.

Implements a multi-stage alignment pipeline:
  1. Center-of-mass initialization
  2. GPU-parallel coarse rotation search (thousands of candidates at once)
  3. Progressive refinement: normalized Adam (GPU) + Powell polish (CPU/GPU)

Hybrid approach:
  - GPU batched grid_sample for coarse search (fast parallel evaluation)
  - Normalized Adam on [0,1] params for smoothing passes (GPU-fast, fixes
    the scale mismatch between mm/degrees/ratios that breaks single-lr Adam)
  - scipy Powell for final full-resolution polish (derivative-free, robust
    convergence without gradient noise issues)
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from torch import Tensor

from fastfuncstuff.memory import compute_registration_candidate_batch_size

from . import cost_hist
from .affine import (
    apply_affine,
    apply_affine_batched,
    apply_affine_interp,
    apply_affine_wsinc5,
    identity_params,
    params_to_matrix,
    params_to_matrix_batched,
    sample_affine_at_points,
    sample_affine_at_points_batched,
)
from .cost import (
    _auto_clip,
    _separable_smooth_3d,
    clipped_pearson_correlation,
    lpa_correlation,
    lpc_correlation,
)
from .cost_blok import (
    BlokSet,
    assign_bloks,
    assign_bloks_points,
    local_pearson_value,
    local_pearson_value_batched,
    lpa_cost,
    lpa_cost_batched,
    lpc_cost,
    lpc_cost_batched,
)
from .cost_hist import clip_range
from .mask import automask
from .weight import compute_weight_image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _tqdm_bar(iterable, total=None, desc=None, disable=False):
    """Wrap iterable in tqdm if available, otherwise passthrough."""
    if tqdm is not None and not disable:
        return tqdm(iterable, total=total, desc=desc, file=sys.stderr, leave=True, ncols=80)
    return iterable


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AffineAlignConfig:
    """Configuration for affine/rigid alignment."""

    # Degrees of freedom
    dof: str = "affine"  # "rigid" (6), "affine" (12), "epi" (9)

    # Cost function. AFNI-faithful: ls, lpa, lpc, mi, nmi, je, hel,
    # cru/cra/crm (correlation ratio). ffs-special per-voxel Gaussian local
    # Pearson: lps (absolute), lpsc (signed).
    cost: str = "lpa"
    lpa_sigma: float = 4.0  # Gaussian sigma / box radius for lps/lpsc
    lpa_kernel: str = "gauss"  # "gauss" or "box" (lps/lpsc only)

    # Blok geometry for AFNI-faithful lpa/lpc (radius in mm; None auto-sizes
    # to ~555 voxels per blok, matching 3dAllineate).
    bloktype: str = "tohd"  # "tohd" (AFNI default), "rhdd", or "cube"
    blokrad: float | None = None
    ppow: float = 1.0  # |z| emphasis exponent (AFNI AFNI_LPC_POWER)

    # Overlap penalty weight (AFNI lpc+/lpa+ "ov"). 0 = off (default). When > 0,
    # a differentiable (max(0, 9.95-10*overlap))**2 term is subtracted from the
    # cost so the refiner is pushed back toward full base/source overlap.
    ov: float = 0.0

    # Match-point subsampling for blok-cost (lpa/lpc) refinement (AFNI npt_match).
    # The cost is evaluated on a fixed random subset of weight-domain points via
    # point-wise sampling, so each iteration is O(n_match) instead of O(all
    # voxels). Interpreted unit-free: <=1.0 is a *fraction* of the in-mask voxels
    # (0.47 = AFNI default, 1.0 = all); >1.0 is an absolute count (e.g. 150000).
    n_match: float = 0.47

    # torch.compile the batched refinement forward (matrix build + point sample +
    # blok cost) to cut per-iteration kernel-launch overhead. Also enabled by the
    # FFS_ALLINEATE_COMPILE=1 environment variable.
    compile: bool = False
    # "adam" = autograd + Adam (legacy); "pattern" = batched derivative-free
    # coordinate search. See _refine_pattern_batched for why the latter suits a
    # 6-12 parameter, launch-bound, jagged-surface problem.
    optimizer: str = "adam"

    # Coarse search. Ranges mirror 3dAllineate's defaults: angle ±30°,
    # shift ±32% of grid size, scale ±20%. ``range_scale`` shrinks all of them
    # (-smallrange -> 0.5, -verysmallrange -> 0.25).
    twopass: bool = True
    coarse_range: float = 30.0  # rotation half-range, degrees
    coarse_step: float = 5.0  # rotation step, degrees
    coarse_shift_frac: float = 0.321  # translation half-range (fraction of grid)
    coarse_shift_steps: int = 7  # samples per translation axis (odd, incl. 0)
    coarse_scale_range: float = 0.20  # scale half-range (fraction)
    range_scale: float = 1.0  # global shrink for all coarse ranges + bounds
    # Best coarse candidates carried into refinement. None -> resolved per cost
    # by `_default_tbest`, because "more trials" is only free on the blok path.
    tbest: int | None = None
    # Joint-coarse seed grid: samples per translation axis and per rotation axis,
    # so the seed count is n_trans^3 * n_rot^3 (+ a few random). Rotation is
    # sampled finer because cmass already fixes translation but never rotation.
    coarse_n_trans: int = 3
    coarse_n_rot: int | None = None  # None -> derived from coarse_range/coarse_step
    coarse_n_random: int = 32  # extra random joint seeds, for robustness

    # Refinement tuning. Iteration counts are ceilings; Adam stops early on a
    # relative-tolerance plateau, so generous caps are cheap when converged.
    adam_iters_2x: int = 300  # Adam iters at 2x downsampled (ceiling)
    adam_iters_1x: int = 400  # Adam iters at full resolution (ceiling)
    # lr 0.005 climbs steadily to a good optimum; higher (e.g. 0.02) overshoots
    # from the cmass start, oscillates, and trips early-stop at a worse point.
    adam_lr_2x: float = 0.01  # Adam learning rate at 2x
    adam_lr: float = 0.005  # Adam learning rate at full resolution
    powell_maxfev: int = 500  # Powell max function evaluations (0=skip)

    # Center-of-mass
    cmass: bool = True
    # Manual cmass shift [dx, dy, dz] in base-grid voxels — the same space and
    # sign the auto path prints. When set, skips the automatic center-of-mass
    # estimate and uses these directly, so you can reproduce or hand-tune the
    # initial placement (manual positioning) instead of relying on COM matching.
    cmass_direct: tuple[float, float, float] | None = None

    # Interpolation
    interp: str = "linear"  # "linear" or "cubic"
    final_interp: str = "linear"

    # Masking
    source_automask: bool = False
    autoweight: bool = True

    # Cropping
    autocrop: bool = True  # crop zero margins from base for optimization

    # Device
    device: str | None = None

    # Verbosity
    verb: int = 1

    def __post_init__(self):
        if self.tbest is None:
            self.tbest = _default_tbest(self.cost)


# ---------------------------------------------------------------------------
# Cost function dispatch
# ---------------------------------------------------------------------------

# Costs that need the blok lattice (AFNI-faithful local Pearson).
_BLOK_COSTS = ("lpa", "lpc")

# Default coarse candidates to refine, by cost family. The blok costs refine
# their trials as one batched, point-subsampled Adam, so trials are nearly free
# up to ~10 (measured: tbest 1..10 all inside the run-to-run spread) and keeping
# more of the coarse lottery costs nothing. The other costs refine on the full
# grid without that batching, where each extra trial is paid in full — measured
# on one pair, tbest 3 -> 10 took ls from 34s to 86s and nmi from 60s to 159s.
_TBEST_DEFAULT_BLOK = 10
_TBEST_DEFAULT_OTHER = 3


def _default_tbest(cost: str) -> int:
    return _TBEST_DEFAULT_BLOK if _base_cost(cost) in _BLOK_COSTS else _TBEST_DEFAULT_OTHER


# AFNI's lpc+/lpa+ combination weights (DEFAULT_MICHO_* in 3dAllineate.c), as
# (hel, mi, nmi, crA). The overlap term is carried separately by -ov, since it
# needs the source coverage map rather than the joint histogram. lpa+ drops the
# MI term (3dAllineate.c, 27 May 2021), which is the only difference.
_MICHO_LPC = (0.4, 0.2, 0.2, 0.4)
_MICHO_LPA = (0.4, 0.0, 0.2, 0.4)
_MICHO_OV = 0.4

_COMBO_COSTS = {
    "lpc+": ("lpc", _MICHO_LPC, False),
    "lpa+": ("lpa", _MICHO_LPA, False),
    "lpc+zz": ("lpc", _MICHO_LPC, True),
    "lpa+zz": ("lpa", _MICHO_LPA, True),
}


def _parse_cost(name: str) -> tuple[str, tuple[float, float, float, float] | None, bool]:
    """Split a cost name into (base cost, combination weights, ZZ final flag).

    ``lpc+``/``lpa+`` add weighted hel/mi/nmi/crA terms to the local-Pearson
    cost, which makes the basin wider and the search more robust. ``+ZZ`` then
    drops those terms for the final polish, so the answer is a pure lpc/lpa
    optimum that the combination merely helped to find.
    """
    key = name.lower()
    if key in _COMBO_COSTS:
        return _COMBO_COSTS[key]
    return name, None, False


def _base_cost(name: str) -> str:
    return _parse_cost(name)[0]


def _micho_terms(ctx: CostContext, base_v: Tensor, warped_v: Tensor, weight_v) -> Tensor:
    """The lpc+/lpa+ extra terms, in ffs (higher == better) units.

    Each of AFNI's four extra functionals, converted to our sign convention, is
    just the standalone ffs cost, so the combination collapses to a plain
    weighted sum ``base + sum(w_i * term_i)`` (the constant offsets AFNI carries
    do not affect the optimum).
    """
    base_v = base_v.reshape(-1)
    warped_v = warped_v.reshape(-1)
    weight_v = None if weight_v is None else weight_v.reshape(-1)
    idx = ctx.micho_subset(base_v.numel(), base_v.device)
    if idx is not None:
        base_v = base_v[idx]
        warped_v = warped_v[idx]
        weight_v = None if weight_v is None else weight_v[idx]
    return cost_hist.combo_terms(
        base_v,
        warped_v,
        ctx.micho,  # type: ignore[arg-type]
        weight=weight_v,
        base_clip=ctx.base_clip,
        source_clip=ctx.source_clip,
    )


# Costs built from the 2D joint histogram.
_HIST_COSTS = ("mi", "nmi", "je", "hel", "cru", "cra", "crm")


@dataclass
class CostContext:
    """Everything a cost evaluation needs, built once per ``allineate`` call.

    Holds the cost name, the Gaussian-kernel params for the ``lps`` family, the
    blok geometry params (resolved radius is in mm), the per-grid blok cache,
    and the intensity clip ranges / bin count for the histogram costs.
    """

    name: str
    sigma: float = 4.0
    kernel: str = "gauss"
    bloktype: str = "tohd"
    blokrad_mm: float | None = None
    base_voxdims: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ppow: float = 1.0
    base_clip: tuple[float, float] | None = None
    source_clip: tuple[float, float] | None = None
    # Interpolation for the source resample *inside* the cost, i.e. -interp.
    # Lives here rather than being read off the config at each call site because
    # the cost evaluators are closures that capture ctx and nothing else.
    interp: str = "linear"
    # Overlap penalty (AFNI lpc+/lpa+ "ov"): differentiable additive term so the
    # gradient-based refiner is steered away from low-overlap configurations.
    # ``src_cov`` is a soft source-coverage map on the optimisation grid that is
    # warped with the candidate transform; ``base_dom`` is the base brain domain;
    # ``ov_denom`` mirrors AFNI's MIN(nbsmask, najmask) normaliser. ``ov_weight``
    # is 0 (off) unless the user passes -ov.
    ov_weight: float = 0.0
    src_cov: Tensor | None = None
    base_dom: Tensor | None = None
    ov_denom: float = 1.0
    # lpc+/lpa+ combination weights (hel, mi, nmi, crA); None == a pure cost.
    # ``micho_zfinal`` is AFNI's +ZZ: drop them for the final polish.
    micho: tuple[float, float, float, float] | None = None
    micho_zfinal: bool = False
    # Points used for the combination's joint histogram. The extra functionals
    # are a basin-widening prior, not the precision term, and a 97x97 histogram
    # is already well determined by ~10^5 samples — while the index_add backward
    # over every match point is what makes the combined costs slow (measured
    # fwd+bwd at 11 trials: 95 ms/iteration on 896k points, 12 ms on 100k).
    micho_npts: int = 100_000
    _micho_idx: dict = None  # type: ignore[assignment]
    _blok_cache: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self._blok_cache is None:
            self._blok_cache = {}
        if self._micho_idx is None:
            self._micho_idx = {}

    def micho_subset(self, n: int, device) -> Tensor | None:
        """Fixed random subset of ``n`` point positions for the combination terms.

        Cached and deterministic, so the combined cost surface is stable across
        iterations the way the main match-point set is. ``None`` means "use all
        of them" (the input is already small enough).
        """
        if n <= self.micho_npts:
            return None
        idx = self._micho_idx.get(n)
        if idx is None:
            g = torch.Generator(device="cpu").manual_seed(4242)
            idx = torch.randperm(n, generator=g)[: self.micho_npts].to(device)
            self._micho_idx[n] = idx
        return idx

    def blokset(self, shape, voxdims, device, blokrad_mm=None) -> BlokSet:
        """Get (or build + cache) the blok assignment for one grid.

        Cached per (shape, rounded voxdims, blokrad) so the lattice is computed
        once and reused across every optimiser iteration on that grid. The blur
        pyramid passes a per-stage ``blokrad_mm`` (inflated by the smoothing
        radius, AFNI-style), so the radius is part of the cache key.
        """
        rad = self.blokrad_mm if blokrad_mm is None else blokrad_mm
        key = (
            tuple(shape),
            tuple(round(v, 4) for v in voxdims),
            None if rad is None else round(rad, 4),
        )
        bs = self._blok_cache.get(key)
        if bs is None:
            bs = assign_bloks(tuple(shape), voxdims, self.bloktype, rad, device=device)
            self._blok_cache[key] = bs
        return bs


@dataclass
class SampleSet:
    """A fixed random subset of weight-domain points for subsampled blok costs.

    Built once per run from the voxels the optimiser actually cares about
    (weight > 0), so the per-iteration cost of the local-Pearson refinement is
    O(M) instead of O(all voxels). The matching points are fixed across
    iterations (a stable cost surface, like AFNI's npt_match), and the per-stage
    blok lattice over them is cached by blok radius.
    """

    idx_flat: Tensor  # (M,) flat indices into the optimisation grid
    points_xyz: Tensor  # (M, 3) base voxel coords (x, y, z) for the point sampler
    coords_mm: Tensor  # (M, 3) physical mm coords for blok assignment
    weight_s: Tensor  # (M,) gathered weight
    bloktype: str = "tohd"
    _blok_cache: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self._blok_cache is None:
            self._blok_cache = {}

    def blokset(self, blokrad_mm, device) -> BlokSet:
        key = None if blokrad_mm is None else round(blokrad_mm, 4)
        bs = self._blok_cache.get(key)
        if bs is None:
            bs = assign_bloks_points(self.coords_mm, self.bloktype, blokrad_mm, device=device)
            self._blok_cache[key] = bs
        return bs


_SAMPLE_DEFAULT_FRAC = 0.47  # AFNI 3dAllineate default: 47% of the in-mask voxels

# Cap on rotation samples per axis in the joint coarse grid. Seed count goes as
# n_rot^3, and the batched evaluation is launch-bound (so nearly free) up to
# ~20k seeds, then roughly linear: measured 3.4k -> 0.83s, 20k -> 0.99s,
# 91k -> 2.94s, 166k -> 4.68s. 19 samples/axis is ~185k seeds, the most that
# buys tickets at a defensible price; -hugerange is what reaches for it.
_MAX_COARSE_ROT_SAMPLES = 19


def _build_sample_set(
    base_opt: Tensor,
    weight_opt: Tensor | None,
    voxdims: tuple[float, float, float],
    n_match: float,
    bloktype: str,
    device: torch.device,
) -> SampleSet | None:
    """Draw a fixed random subset of the weight domain for subsampled costs.

    The domain is ``weight > 0`` (the autoweight already drops the background and
    fades the FOV edge); without a weight it falls back to nonzero base voxels.
    ``n_match`` is interpreted like 3dAllineate's npt_match but unit-free:
    ``<= 0`` -> default 47% of the domain; ``<= 1.0`` -> that *fraction* of the
    domain (so 1.0 == all); ``> 1.0`` -> that many points (e.g. 150000).
    Returns None if the domain is too small to bother subsampling.
    """
    _, ny, nx = base_opt.shape
    if weight_opt is not None:
        domain = (weight_opt.reshape(-1) > 0).nonzero(as_tuple=False).reshape(-1)
    else:
        domain = (base_opt.reshape(-1) != 0).nonzero(as_tuple=False).reshape(-1)
    n_dom = int(domain.numel())
    if n_dom < 2000:  # tiny / synthetic: not worth the point machinery
        return None

    if n_match <= 0.0:
        budget = int(_SAMPLE_DEFAULT_FRAC * n_dom)
    elif n_match <= 1.0:
        budget = int(n_match * n_dom)
    else:
        budget = int(n_match)
    if budget >= n_dom:
        idx_flat = domain
    else:
        # Deterministic subset so the cost surface is stable across iterations
        # and reproducible across runs.
        g = torch.Generator(device="cpu").manual_seed(12345)
        perm = torch.randperm(n_dom, generator=g)[:budget]
        idx_flat = domain[perm.to(domain.device)]

    nxy = ny * nx
    z = torch.div(idx_flat, nxy, rounding_mode="floor")
    rem = idx_flat - z * nxy
    y = torch.div(rem, nx, rounding_mode="floor")
    x = rem - y * nx
    points_xyz = torch.stack([x, y, z], dim=1).to(dtype=torch.float32)
    dx, dy, dz = (float(v) for v in voxdims)
    coords_mm = points_xyz * torch.tensor([dx, dy, dz], device=points_xyz.device)
    weight_s = (
        weight_opt.reshape(-1)[idx_flat]
        if weight_opt is not None
        else torch.ones(idx_flat.numel(), device=device)
    )
    return SampleSet(
        idx_flat=idx_flat,
        points_xyz=points_xyz,
        coords_mm=coords_mm,
        weight_s=weight_s,
        bloktype=bloktype,
    )


def _voxdims_from_header(header: dict | None) -> tuple[float, float, float]:
    """Voxel sizes (dx, dy, dz) in mm from a load_image header (x fastest).

    Falls back to isotropic 1 mm when no affine is available.
    """
    if header is None or "affine" not in header:
        return (1.0, 1.0, 1.0)
    aff = np.asarray(header["affine"], dtype=np.float64)
    dx = float(np.linalg.norm(aff[:3, 0]))
    dy = float(np.linalg.norm(aff[:3, 1]))
    dz = float(np.linalg.norm(aff[:3, 2]))
    return (dx or 1.0, dy or 1.0, dz or 1.0)


def _overlap_penalty(ctx: CostContext, matrix: Tensor, out_shape) -> Tensor:
    """AFNI lpc+/lpa+ overlap penalty as a differentiable scalar (>= 0).

    Warps the soft source-coverage map by ``matrix`` and measures the fraction
    of the base brain domain it covers, then applies AFNI's
    ``(max(0, 9.95 - 10*ov))**2`` shape (mri_genalign.c GA_scalar_costfun). The
    fraction depends on the warp through grid_sample, so the term is
    differentiable and the Adam/Powell refiner is actively pushed back toward
    overlap rather than only being re-ranked after the fact.
    """
    warped_cov = apply_affine(ctx.src_cov, matrix, out_shape, zero_outside=True)
    ov = (ctx.base_dom * warped_cov).sum() / max(ctx.ov_denom, 1e-6)
    ovv = torch.clamp(9.95 - 10.0 * ov, min=0.0)
    return ovv * ovv


def _compute_cost(
    base: Tensor,
    warped: Tensor,
    weight: Tensor | None,
    ctx: CostContext,
    voxdims: tuple[float, float, float] = (1.0, 1.0, 1.0),
    matrix: Tensor | None = None,
    blokrad_mm: float | None = None,
) -> Tensor:
    """Compute alignment cost for one warped volume (higher == better).

    ``matrix`` (base->source voxel map) is only needed when the overlap penalty
    is active; ``blokrad_mm`` overrides the blok radius for the current blur
    pyramid stage.
    """
    name = ctx.name
    if name in ("ls", "pearclp"):
        cost = clipped_pearson_correlation(
            base.reshape(-1),
            warped.reshape(-1),
            weight.reshape(-1) if weight is not None else None,
        )
    elif name == "lps":
        cost = lpa_correlation(base, warped, weight, sigma=ctx.sigma, kernel_type=ctx.kernel)
    elif name == "lpsc":
        cost = lpc_correlation(base, warped, weight, sigma=ctx.sigma, kernel_type=ctx.kernel)
    elif name in _BLOK_COSTS:
        bs = ctx.blokset(base.shape, voxdims, base.device, blokrad_mm=blokrad_mm)
        fn = lpa_cost if name == "lpa" else lpc_cost
        cost = fn(base, warped, weight, bs, ppow=ctx.ppow)
    elif name in _HIST_COSTS:
        kw = dict(weight=weight, base_clip=ctx.base_clip, source_clip=ctx.source_clip)
        if name == "mi":
            cost = cost_hist.mi_cost(base, warped, **kw)
        elif name == "nmi":
            cost = cost_hist.nmi_cost(base, warped, **kw)
        elif name == "je":
            cost = cost_hist.je_cost(base, warped, **kw)
        elif name == "hel":
            cost = cost_hist.hel_cost(base, warped, **kw)
        else:
            mode = {"cru": "u", "cra": "a", "crm": "m"}[name]
            cost = cost_hist.cr_cost(base, warped, mode=mode, **kw)
    else:
        raise ValueError(f"Unknown cost function: {name}")

    if ctx.micho is not None:
        cost = cost + _micho_terms(ctx, base, warped, weight)

    # Overlap penalty (subtracted because we maximise; AFNI adds it to a cost it
    # minimises). Only when -ov is set and a transform is available.
    if ctx.ov_weight > 0.0 and ctx.src_cov is not None and matrix is not None:
        cost = cost - ctx.ov_weight * _overlap_penalty(ctx, matrix, base.shape)
    return cost


def _batched_cost(
    base: Tensor,
    warped_batch: Tensor,
    weight: Tensor | None,
    ctx: CostContext,
    voxdims: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tensor:
    """Compute cost for B warped images against a single base -> (B,)."""
    B = warped_batch.shape[0]

    if ctx.name in ("ls", "pearclp"):
        base_flat = base.reshape(-1)
        w_flat = weight.reshape(-1) if weight is not None else None
        base_clip = _auto_clip(base_flat, w_flat)
        bc = base_flat.clamp(base_clip[0], base_clip[1])

        warped_flat = warped_batch.reshape(B, -1)
        src_clip = _auto_clip(warped_flat[0], w_flat)
        sc = warped_flat.clamp(src_clip[0], src_clip[1])

        if w_flat is not None:
            w = w_flat[None, :]
            wsum = w.sum()
            bm = (w * bc[None, :]).sum(dim=1) / wsum
            sm = (w * sc).sum(dim=1) / wsum
            bd = bc[None, :] - bm[:, None]
            sd = sc - sm[:, None]
            bb = (w * bd * bd).sum(dim=1)
            ss = (w * sd * sd).sum(dim=1)
            bs = (w * bd * sd).sum(dim=1)
        else:
            N = base_flat.numel()
            bm = bc.sum() / N
            sm = sc.sum(dim=1) / N
            bd = bc[None, :] - bm
            sd = sc - sm[:, None]
            bb = (bd * bd).sum(dim=1)
            ss = (sd * sd).sum(dim=1)
            bs = (bd * sd).sum(dim=1)

        denom = (bb * ss).sqrt().clamp(min=1e-10)
        return bs / denom

    if ctx.name in _BLOK_COSTS:
        bs = ctx.blokset(base.shape, voxdims, base.device)
        fn = lpa_cost_batched if ctx.name == "lpa" else lpc_cost_batched
        return fn(base, warped_batch, weight, bs, ctx.ppow)

    # Histogram / lps costs: evaluate per candidate (coarse grids are small).
    costs = torch.zeros(B, device=base.device)
    for i in range(B):
        costs[i] = _compute_cost(base, warped_batch[i], weight, ctx, voxdims)
    return costs


# ---------------------------------------------------------------------------
# Multi-resolution utilities
# ---------------------------------------------------------------------------


def _downsample_3d(vol: Tensor, factor: int) -> Tensor:
    """Downsample a 3D volume by an integer factor using avg pooling."""
    if factor <= 1:
        return vol
    v = vol[None, None]
    v = F.avg_pool3d(v, kernel_size=factor, stride=factor)
    return v[0, 0]


def _smooth_to_resolution(vol: Tensor, factor: int) -> Tensor:
    """Apply Gaussian blur matching the target resolution."""
    if factor <= 1:
        return vol
    sigma = factor * 0.5
    return _separable_smooth_3d(vol, sigma)


# ---------------------------------------------------------------------------
# Auto-crop: remove zero margins for faster optimization
# ---------------------------------------------------------------------------


def _compute_nonzero_bbox(
    vol: Tensor, pad: int = 8
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Compute tight bounding box of nonzero voxels, with padding.

    Uses generous padding (default 8 voxels, at least 5% of each axis)
    to avoid clipping brain edges during optimization.

    Returns:
        (start, end): each is (z, y, x) indices for slicing [start:end].
    """
    nz, ny, nx = vol.shape
    nonzero = vol != 0

    # Project along each axis pair to find bounds per axis
    z_any = nonzero.any(dim=2).any(dim=1)  # (nz,)
    y_any = nonzero.any(dim=2).any(dim=0)  # (ny,)
    x_any = nonzero.any(dim=1).any(dim=0)  # (nx,)

    z_idx = torch.where(z_any)[0]
    y_idx = torch.where(y_any)[0]
    x_idx = torch.where(x_any)[0]

    if len(z_idx) == 0 or len(y_idx) == 0 or len(x_idx) == 0:
        return (0, 0, 0), (nz, ny, nx)

    # Per-axis padding: at least `pad` voxels, at least 5% of dimension
    pz = max(pad, int(math.ceil(nz * 0.05)))
    py = max(pad, int(math.ceil(ny * 0.05)))
    px = max(pad, int(math.ceil(nx * 0.05)))

    z0 = max(0, int(z_idx[0].item()) - pz)
    z1 = min(nz, int(z_idx[-1].item()) + 1 + pz)
    y0 = max(0, int(y_idx[0].item()) - py)
    y1 = min(ny, int(y_idx[-1].item()) + 1 + py)
    x0 = max(0, int(x_idx[0].item()) - px)
    x1 = min(nx, int(x_idx[-1].item()) + 1 + px)

    return (z0, y0, x0), (z1, y1, x1)


def _crop_volumes(
    base: Tensor,
    source: Tensor,
    weight: Tensor | None,
    pad: int = 8,
) -> tuple[Tensor, Tensor, Tensor | None, tuple[int, int, int]]:
    """Crop base/source/weight to the nonzero bounding box of base.

    The crop is only used during optimization — the final output is always
    produced on the full original base grid. The crop offset is returned
    so the optimized matrix can be adjusted back to full-grid coordinates.

    Uses generous padding (default 8 voxels + 5% per axis) to avoid
    clipping brain edges.

    Args:
        base: (nz, ny, nx) base image.
        source: (nz, ny, nx) source on base grid.
        weight: optional (nz, ny, nx) weight image.
        pad: minimum padding voxels around the bounding box.

    Returns:
        (base_crop, source_crop, weight_crop, offset):
            offset is (x_off, y_off, z_off) in voxels — add to translations
            to convert from cropped to full-grid space.
    """
    (z0, y0, x0), (z1, y1, x1) = _compute_nonzero_bbox(base, pad=pad)

    base_crop = base[z0:z1, y0:y1, x0:x1].contiguous()
    source_crop = source[z0:z1, y0:y1, x0:x1].contiguous()
    weight_crop = weight[z0:z1, y0:y1, x0:x1].contiguous() if weight is not None else None

    # Offset: params are [dx, dy, dz] = [x, y, z] shifts
    # In cropped space, coordinate i maps to full-space coordinate i + offset
    # So to transform: full_coord = crop_coord + offset
    # The matrix maps base voxels → source voxels:
    #   source_voxel = M @ base_voxel
    # In cropped space: base_crop_voxel = base_full_voxel - offset
    # So: source_full_voxel = M @ (base_crop_voxel + offset)
    # Translation in cropped space = M_trans - (M_3x3 - I) @ offset
    # But for optimization, we work in cropped space and adjust after.
    offset = (x0, y0, z0)

    return base_crop, source_crop, weight_crop, offset


# ---------------------------------------------------------------------------
# Source validity mask
# ---------------------------------------------------------------------------


def _compute_source_validity_mask(
    source_shape: tuple[int, int, int],
    base_shape: tuple[int, int, int],
    grid_matrix: Tensor,
) -> Tensor:
    """Compute which base-grid voxels fall inside the source volume.

    When source and base have different FOVs, some base voxels map to
    locations outside the source volume. This mask identifies valid voxels.

    Args:
        source_shape: (nz, ny, nx) of native source.
        base_shape: (nz, ny, nx) of base grid.
        grid_matrix: (4, 4) mapping base voxels → source voxels.

    Returns:
        (nz, ny, nx) bool tensor on same device as grid_matrix.
    """
    device = grid_matrix.device
    dtype = grid_matrix.dtype
    onz, ony, onx = base_shape
    snz, sny, snx = source_shape

    kk, jj, ii = torch.meshgrid(
        torch.arange(onz, dtype=dtype, device=device),
        torch.arange(ony, dtype=dtype, device=device),
        torch.arange(onx, dtype=dtype, device=device),
        indexing="ij",
    )
    N = onz * ony * onx
    coords = torch.stack(
        [ii.reshape(-1), jj.reshape(-1), kk.reshape(-1), torch.ones(N, device=device, dtype=dtype)],
        dim=0,
    )
    src_coords = grid_matrix @ coords  # (4, N)

    src_x = src_coords[0]
    src_y = src_coords[1]
    src_z = src_coords[2]

    valid = (
        (src_x >= -0.5)
        & (src_x <= snx - 0.5)
        & (src_y >= -0.5)
        & (src_y <= sny - 0.5)
        & (src_z >= -0.5)
        & (src_z <= snz - 0.5)
    )

    return valid.reshape(onz, ony, onx)


# ---------------------------------------------------------------------------
# Parameter bounds and normalization (AFNI-style)
# ---------------------------------------------------------------------------


def _compute_param_bounds(
    base_shape: tuple[int, int, int],
    cmass_shift: np.ndarray | None = None,
    range_scale: float = 1.0,
    rot_range: float = 30.0,
    shift_frac: float = 0.321,
) -> np.ndarray:
    """Compute AFNI-style parameter bounds.

    All units match our internal convention (voxels for translations,
    degrees for rotations, ratios for scales/shears). ``range_scale`` shrinks
    the translation/rotation/scale ranges (-smallrange -> 0.5, etc.).

    Returns:
        (12, 2) array of [min, max] per parameter.
    """
    nz, ny, nx = base_shape
    if cmass_shift is None:
        cmass_shift = np.zeros(3)

    bounds = np.zeros((12, 2))
    rs = range_scale

    # Translation range: ±32% of FOV (* range_scale), centered at cmass.
    sf = shift_frac * rs
    tx, ty, tz = sf * (nx - 1), sf * (ny - 1), sf * (nz - 1)
    bounds[0] = [cmass_shift[0] - tx, cmass_shift[0] + tx]
    bounds[1] = [cmass_shift[1] - ty, cmass_shift[1] + ty]
    bounds[2] = [cmass_shift[2] - tz, cmass_shift[2] + tz]

    bounds[3:6] = [-rot_range * rs, rot_range * rs]  # rotations (degrees)
    sc = 0.20 * rs  # scale half-range (±20%)
    bounds[6:9] = [1.0 - sc, 1.0 + sc]
    bounds[9:12] = [-0.1111 * rs, 0.1111 * rs]  # shears

    return bounds


def _get_free_mask(dof: str) -> np.ndarray:
    """Return boolean mask of which parameters are free."""
    free = np.ones(12, dtype=bool)
    if dof == "rigid":
        free[6:12] = False
    elif dof == "epi":
        free[6] = False  # sx
        free[8] = False  # sz
        free[11] = False  # shzy
    return free


def _identity_physical() -> np.ndarray:
    """Return identity parameter values: (0,0,0, 0,0,0, 1,1,1, 0,0,0)."""
    p = np.zeros(12)
    p[6:9] = 1.0
    return p


def _normalize(params: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Physical → [0,1] normalized space."""
    span = bounds[:, 1] - bounds[:, 0]
    span[span < 1e-10] = 1.0
    return (params - bounds[:, 0]) / span


def _denormalize(x_norm: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """[0,1] normalized → physical space."""
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * x_norm


# Torch versions for Adam (stay on GPU, differentiable)
def _bounds_to_torch(bounds: np.ndarray, device: torch.device) -> tuple[Tensor, Tensor]:
    """Convert bounds to torch tensors: (bmin, span)."""
    bmin = torch.tensor(bounds[:, 0], dtype=torch.float32, device=device)
    bmax = torch.tensor(bounds[:, 1], dtype=torch.float32, device=device)
    span = bmax - bmin
    span[span < 1e-10] = 1.0
    return bmin, span


def _normalize_t(params: Tensor, bmin: Tensor, span: Tensor) -> Tensor:
    """Physical → [0,1] (differentiable)."""
    return (params - bmin) / span


def _denormalize_t(x_norm: Tensor, bmin: Tensor, span: Tensor) -> Tensor:
    """[0,1] → physical (differentiable)."""
    return bmin + span * x_norm


# ---------------------------------------------------------------------------
# Stage 1: Center-of-mass initialization
# ---------------------------------------------------------------------------


def _center_of_mass(vol: Tensor, weight: Tensor | None = None) -> Tensor:
    """Compute weighted center of mass in voxel coordinates."""
    device = vol.device
    dtype = vol.dtype
    nz, ny, nx = vol.shape

    w = weight * vol.abs() if weight is not None else vol.abs()
    wsum = w.sum()
    if wsum < 1e-10:
        return torch.tensor([nx / 2.0, ny / 2.0, nz / 2.0], device=device, dtype=dtype)

    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=dtype, device=device),
        torch.arange(ny, dtype=dtype, device=device),
        torch.arange(nx, dtype=dtype, device=device),
        indexing="ij",
    )
    return torch.stack([(w * ii).sum() / wsum, (w * jj).sum() / wsum, (w * kk).sum() / wsum])


def _cmass_translation(base: Tensor, source: Tensor, grid_matrix: Tensor | None = None) -> Tensor:
    """Translation (base-grid voxels) mapping the source centroid onto the base.

    Each centroid is the value-weighted centre of mass of its OWN volume on its
    OWN grid (AFNI's mri_get_cmass_3D). Crucially the source centroid must come
    from the *native* source, not from the source resampled onto the base grid:
    that resample clips the source to the base FOV, so anything the base doesn't
    cover (e.g. the top of the brain under a short EPI) is dropped from the source
    centroid too and the shift comes up short ("not moved far enough").

    ``grid_matrix`` (base→source voxels) maps the native-source centroid back
    into base-grid voxels before differencing; pass None when the two already
    share a grid.
    """
    c_base = _center_of_mass(base)
    c_src = _center_of_mass(source)
    if grid_matrix is not None:
        c_h = torch.cat([c_src, c_src.new_ones(1)])
        c_src = (torch.linalg.inv(grid_matrix) @ c_h)[:3]
    return c_src - c_base


# ---------------------------------------------------------------------------
# Stage 2: GPU-parallel coarse search
# ---------------------------------------------------------------------------


def _coarse_search_joint(
    base_opt: Tensor,
    source_opt: Tensor,
    sample: SampleSet,
    ctx: CostContext,
    config: AffineAlignConfig,
    bounds: np.ndarray,
    device: torch.device,
    verb: int = 1,
) -> list[np.ndarray]:
    """Joint rigid coarse search, point-subsampled (AFNI ransetup-style).

    Instead of the split translation→rotation sweep (which seeds the refiner far
    from the basin when both are needed, because the best translation at zero
    rotation is not the best translation at the true rotation), this:

      1. builds a *joint* seed set over (translation, rotation) — a coarse grid
         plus a few random points — so every seed is a real (T, R) pair;
      2. scores them all on a small subset of the match points (batched, on a
         mild blur for a wide basin) — full-resolution points, not a distorted
         2x-downsample, so the ranking signal is real;
      3. polishes the best NKEEP with a short subsampled joint Adam;
      4. dedups by parameter distance and returns the best ``tbest``.

    All evaluation is point-wise (O(M)), so the joint search is cheaper than the
    old split+downsample one despite covering more of the space jointly.
    """
    free_mask = _get_free_mask(config.dof)
    is_lpc = ctx.name == "lpc"
    voxdims = ctx.base_voxdims
    vmean = sum(voxdims) / 3.0
    sigma = 2.0  # mild blur: wide capture basin for the seed polish

    # Coarse match points: a small subset of the refinement sample (AFNI scores
    # the coarse pass on far fewer points than the fine pass).
    n_coarse = min(int(sample.idx_flat.numel()), 40000)
    pts_xyz = sample.points_xyz[:n_coarse]
    coords_mm = sample.coords_mm[:n_coarse]
    weight_c = sample.weight_s[:n_coarse]
    idx_c = sample.idx_flat[:n_coarse]

    blokrad_c = math.sqrt(ctx.blokrad_mm**2 + (sigma * vmean) ** 2) if ctx.blokrad_mm else None
    base_blur = _separable_smooth_3d(base_opt, sigma)
    source_blur = _separable_smooth_3d(source_opt, sigma)
    base_pts = base_blur.reshape(-1)[idx_c]
    blokset_c = assign_bloks_points(coords_mm, ctx.bloktype, blokrad_c, device=device)

    # --- joint seed set: grid over (n_t translations x n_r rotations per axis) ---
    def _axis(lo, hi, n):
        return torch.linspace(float(lo), float(hi), n, device=device)

    # Rotation density follows -coarse_step across the (already -coarse_range
    # wide) bounds; -coarse_n_rot overrides it outright. Seeds are cheap: the
    # batched evaluation is launch-bound, so 3.4k -> 20k seeds costs ~0.2s.
    # What is NOT cheap is being wrong about the *range* — seeds packed more
    # densely into a box that excludes the answer never find it.
    n_t = config.coarse_n_trans
    if config.coarse_n_rot is not None:
        n_r = config.coarse_n_rot
    else:
        span = float(bounds[3, 1] - bounds[3, 0])
        n_r = int(
            min(_MAX_COARSE_ROT_SAMPLES, max(3, round(span / max(config.coarse_step, 0.5)) + 1))
        )
    tax = [_axis(bounds[a, 0], bounds[a, 1], n_t) for a in range(3)]
    rax = [_axis(bounds[3 + a, 0], bounds[3 + a, 1], n_r) for a in range(3)]
    tg = torch.meshgrid(*tax, indexing="ij")
    rg = torch.meshgrid(*rax, indexing="ij")
    trans = torch.stack([g.reshape(-1) for g in tg], dim=1)  # (n_t^3, 3)
    rots = torch.stack([g.reshape(-1) for g in rg], dim=1)  # (n_r^3, 3)
    nt3, nr3 = trans.shape[0], rots.shape[0]
    seeds = _base_params(nt3 * nr3, trans.repeat_interleave(nr3, 0), device)
    seeds[:, 3:6] = rots.repeat(nt3, 1)

    # plus a handful of random joint seeds (deterministic) for robustness
    n_rand = config.coarse_n_random
    if n_rand > 0:
        g = torch.Generator(device="cpu").manual_seed(2024)
        rnd = _base_params(n_rand, torch.zeros(n_rand, 3, device=device), device)
        for a in range(6):
            lo, hi = float(bounds[a, 0]), float(bounds[a, 1])
            rnd[:, a] = torch.rand(n_rand, generator=g).to(device) * (hi - lo) + lo
        seeds = torch.cat([seeds, rnd], dim=0)

    # --- batched subsampled evaluation of every seed ---
    matrices = params_to_matrix_batched(seeds)
    B = matrices.shape[0]
    chunk = compute_registration_candidate_batch_size(n_coarse, B, device, bytes_per_point=40)
    costs = []
    for s in range(0, B, chunk):
        with torch.no_grad():
            wb = sample_affine_at_points_batched(
                source_blur, matrices[s : s + chunk], pts_xyz, zero_outside=True, interp=ctx.interp
            )
            val = local_pearson_value_batched(base_pts, wb, weight_c, blokset_c, ctx.ppow)
        costs.append((-val) if is_lpc else val.abs())
    costs = torch.cat(costs)

    # --- polish a few of the best seeds with a short joint subsampled Adam ---
    # The batched eval above already ranks 3000+ seeds, so we only polish a
    # handful past the `tbest` we keep (a couple spares for the dedup) with few
    # iters — and we polish them all in one batched Adam, which keeps the GPU
    # busy (launch-bound otherwise). The fine refinement does the heavy lifting.
    nkeep = min(B, config.tbest + 2)
    top = costs.topk(nkeep).indices.tolist()
    coarse_cost = _batched_sampled_cost(source_blur, pts_xyz, base_pts, weight_c, blokset_c, ctx)
    out_phys, out_costs = _refine_adam_batched(
        [seeds[i].cpu().numpy() for i in top],
        config,
        bounds,
        device,
        coarse_cost,
        verb=0,
        n_iters=40,
        lr=config.adam_lr_2x,
        desc="coarse",
        compile_fwd=config.compile,
    )
    polished = [(float(out_costs[t]), out_phys[t]) for t in range(len(top))]
    polished.sort(key=lambda x: -x[0])

    # --- dedup by parameter distance, keep tbest distinct ---
    trials = [polished[0]]
    for c, p in polished[1:]:
        nj = _normalize(p, bounds)
        if all(
            np.max(np.abs(nj[free_mask] - _normalize(pk, bounds)[free_mask])) >= 0.05
            for _, pk in trials
        ):
            trials.append((c, p))
        if len(trials) >= config.tbest:
            break

    # Insurance seed (AFNI carries the identity/pinit transform into the fine
    # pass): the cmass shift is already baked into align_matrix, so the identity
    # *residual* is the "trust cmass, no rotation" fallback. If the coarse search
    # ever ranks only bad basins, refinement still has this known-decent start to
    # fall back to, and it's nearly free now that trials are batched.
    out = [p for _, p in trials]
    out.append(_identity_physical())

    if verb >= 1:
        p = trials[0][1]
        print(
            f"  Joint coarse: {B} seeds → polished {nkeep} → kept {len(trials)} "
            f"(+identity) (best cost={trials[0][0]:.4f}, "
            f"rot=({p[3]:.1f}°,{p[4]:.1f}°,{p[5]:.1f}°), "
            f"shift=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}))"
        )
    return out


def _base_params(n: int, translations: Tensor, device: torch.device) -> Tensor:
    """(n, 12) identity-scale param rows with the given (n, 3) translations."""
    params = torch.zeros(n, 12, device=device)
    params[:, 0:3] = translations
    params[:, 6:9] = 1.0
    return params


def _translation_candidates(
    center: Tensor, shift_range: Tensor, steps: int, device: torch.device
) -> Tensor:
    """3D grid of translations (zero rotation) over center ± shift_range.

    ``shift_range`` is a per-axis (3,) half-range in voxels; ``steps`` samples
    per axis (forced odd so the center is included).
    """
    if steps < 1:
        steps = 1
    if steps % 2 == 0:
        steps += 1
    axes = []
    for a in range(3):
        r = float(shift_range[a])
        axes.append(
            torch.linspace(-r, r, steps, device=device) if r > 0 else torch.zeros(1, device=device)
        )
    gx, gy, gz = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    offs = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)
    return _base_params(offs.shape[0], center[None, :] + offs, device)


def _rotation_candidates(
    angle_range: float,
    angle_step: float,
    translations: Tensor,
    device: torch.device,
) -> Tensor:
    """Rotation grid (±angle_range) replicated at each given translation."""
    angles = torch.arange(-angle_range, angle_range + angle_step * 0.5, angle_step, device=device)
    if angles.numel() == 0:
        angles = torch.zeros(1, device=device)
    rz, rx, ry = torch.meshgrid(angles, angles, angles, indexing="ij")
    rot = torch.stack([rz.reshape(-1), rx.reshape(-1), ry.reshape(-1)], dim=1)
    nrot = rot.shape[0]
    if translations.ndim == 1:
        translations = translations[None, :]
    blocks = []
    for t in translations:
        p = _base_params(nrot, t[None, :].expand(nrot, -1), device)
        p[:, 3:6] = rot
        blocks.append(p)
    return torch.cat(blocks, dim=0)


def _eval_candidates(
    base: Tensor,
    source: Tensor,
    weight: Tensor | None,
    candidates: Tensor,
    ctx: CostContext,
    voxdims: tuple[float, float, float],
    device: torch.device,
    desc: str,
    verb: int,
) -> Tensor:
    """Cost of every (B, 12) candidate against base (chunked) -> (B,)."""
    matrices = params_to_matrix_batched(candidates)
    B = candidates.shape[0]
    chunk_size = _estimate_chunk_size(base.shape, device, B)
    all_costs = []
    chunks = range(0, B, chunk_size)
    for start in _tqdm_bar(chunks, total=len(range(0, B, chunk_size)), desc=desc, disable=verb < 1):
        end = min(start + chunk_size, B)
        with torch.no_grad():
            # zero_outside (AFNI outval=0): base voxels mapping outside the
            # source read 0, not a replicated border. Border padding smears the
            # brain edge into the shifted-in region and manufactures spurious
            # local correlation, letting the search drift out of overlap; the
            # zero fill dilutes those edge bloks instead (AFNI's implicit
            # overlap mechanism). See mri_genalign_util.c GA_interp_* outval.
            warped = apply_affine_batched(
                source, matrices[start:end], base.shape, zero_outside=True
            )
            costs = _batched_cost(base, warped, weight, ctx, voxdims)
        all_costs.append(costs)
        del warped
    return torch.cat(all_costs)


def _coarse_search(
    base: Tensor,
    source: Tensor,
    weight: Tensor | None,
    config: AffineAlignConfig,
    ctx: CostContext,
    voxdims: tuple[float, float, float],
    init_translation: Tensor,
    device: torch.device,
    verb: int = 1,
) -> list[Tensor]:
    """Two-phase broad coarse search (AFNI-style ranges).

    Phase A sweeps translations over center ± (coarse_shift_frac * grid) on a
    3D grid (rotation = 0). Phase B sweeps the rotation grid (±coarse_range) at
    each of the best translations from phase A. We *evaluate* many positions
    and keep the best ``tbest`` — we do not optimise each candidate.

    Coarse uses the same (signed) cost as the fine pass; ``range_scale`` shrinks
    the ranges for ``-smallrange`` / ``-verysmallrange``.

    Returns the top ``tbest`` parameter vectors (best first).
    """
    nz, ny, nx = base.shape
    rs = config.range_scale
    # Per-axis translation half-range, in this grid's voxels.
    shift_range = torch.tensor(
        [
            config.coarse_shift_frac * rs * (nx - 1),
            config.coarse_shift_frac * rs * (ny - 1),
            config.coarse_shift_frac * rs * (nz - 1),
        ],
        device=device,
    )
    angle_range = config.coarse_range * rs

    # --- Phase A: broad translation sweep (no rotation) ---
    trans_cands = _translation_candidates(
        init_translation, shift_range, config.coarse_shift_steps, device
    )
    if verb >= 1:
        print(
            f"  Coarse phase A (translation): {trans_cands.shape[0]} "
            f"candidates, shift=±({shift_range[0]:.0f},{shift_range[1]:.0f},"
            f"{shift_range[2]:.0f})vox"
        )
    costs_a = _eval_candidates(
        base, source, weight, trans_cands, ctx, voxdims, device, "Coarse-T", verb
    )
    k = min(config.tbest, costs_a.shape[0])
    top_t = costs_a.topk(k).indices
    best_translations = trans_cands[top_t, 0:3]  # (k, 3)

    # --- Phase B: rotation sweep at the best translation(s) ---
    rot_cands = _rotation_candidates(angle_range, config.coarse_step, best_translations, device)
    if verb >= 1:
        print(
            f"  Coarse phase B (rotation): {rot_cands.shape[0]} candidates, "
            f"angle=±{angle_range:.0f}°, step={config.coarse_step}°"
        )
    costs_b = _eval_candidates(
        base, source, weight, rot_cands, ctx, voxdims, device, "Coarse-R", verb
    )

    # Combine both phases and keep the global best tbest.
    all_cands = torch.cat([trans_cands, rot_cands], dim=0)
    all_costs = torch.cat([costs_a, costs_b], dim=0)
    tbest = min(config.tbest, all_cands.shape[0])
    top_costs, top_idx = all_costs.topk(tbest)

    result = [all_cands[top_idx[i]] for i in range(tbest)]
    if verb >= 1:
        p = result[0]
        print(
            f"  Best coarse: cost={top_costs[0].item():.6f}, "
            f"rot=({p[3].item():.1f}°, {p[4].item():.1f}°, {p[5].item():.1f}°), "
            f"shift=({p[0].item():.1f}, {p[1].item():.1f}, {p[2].item():.1f})"
        )
        if tbest > 1:
            print(f"  Keeping top {tbest} candidates for refinement")
    return result


def _estimate_chunk_size(
    vol_shape: tuple[int, ...], device: torch.device, n_candidates: int = 4096
) -> int:
    """Estimate how many candidates we can process at once."""
    voxels = vol_shape[0] * vol_shape[1] * vol_shape[2]
    # Peak memory per candidate: src_coords(16) + gx/gy/gz(12) + grid(12)
    # + grid_sample output(4) + grid_sample internals (~8) ≈ 52 bytes/voxel
    return compute_registration_candidate_batch_size(voxels, n_candidates, device)


# ---------------------------------------------------------------------------
# Refinement: Normalized Adam (GPU-fast)
# ---------------------------------------------------------------------------


def _refine_adam_normalized(
    base: Tensor,
    source: Tensor,
    weight: Tensor | None,
    init_params_phys: np.ndarray,
    config: AffineAlignConfig,
    ctx: CostContext,
    voxdims: tuple[float, float, float],
    bounds: np.ndarray,
    device: torch.device,
    verb: int = 1,
    n_iters: int = 150,
    lr: float = 0.01,
    desc: str = "Adam",
    blokrad_mm: float | None = None,
    cost_fn=None,
) -> tuple[np.ndarray, float]:
    """Refine parameters using Adam on [0,1] normalized params (GPU).

    ``cost_fn``, when given, maps a (4,4) matrix to a scalar cost (higher ==
    better) and replaces the full-grid ``apply_affine`` + ``_compute_cost`` path
    — used for the subsampled blok refinement.

    The key fix over raw Adam: all 12 parameters are normalized to [0,1]
    using AFNI-style bounds, so translations (voxels), rotations (degrees),
    scales (ratios), and shears (ratios) are all on comparable scales.
    Adam's per-parameter adaptive rates then work correctly.

    Returns:
        (params_phys, best_cost): refined params and best cost achieved.
    """
    free_mask_np = _get_free_mask(config.dof)
    free_mask = torch.tensor(free_mask_np, dtype=torch.bool, device=device)
    identity_phys = _identity_physical()

    bmin, span = _bounds_to_torch(bounds, device)
    identity_norm = _normalize_t(
        torch.tensor(identity_phys, dtype=torch.float32, device=device),
        bmin,
        span,
    )

    # Initialize in normalized space
    init_t = torch.tensor(init_params_phys, dtype=torch.float32, device=device)
    params_norm = _normalize_t(init_t, bmin, span).clone().detach()
    params_norm.requires_grad_(True)

    optimizer = torch.optim.Adam([params_norm], lr=lr)

    # Best tracking lives on-device (no per-iter host sync): every step updates
    # best_norm / best_cost_t with torch.where / torch.maximum, which keeps the
    # GPU pipeline running ahead. We only copy a scalar back to the host every
    # `sync_every` steps — purely for the plateau test and the progress bar.
    best_cost_t = torch.full((), -float("inf"), device=device)
    best_norm = params_norm.detach().clone()
    last_best = -float("inf")
    no_improve = 0

    # Plateau detection: count progress only when the *best* cost rose by more
    # than a relative tolerance; stop once flat for `patience` iterations. The
    # generous iteration ceiling then costs nothing when converged.
    rel_tol = 1e-4
    abs_tol = 1e-6
    patience = 40
    sync_every = 15

    pbar = _tqdm_bar(range(n_iters), total=n_iters, desc=desc, disable=verb < 1)
    for it in pbar:
        optimizer.zero_grad()

        # Enforce frozen params and clamp to [0, 1]
        with torch.no_grad():
            params_norm.data[~free_mask] = identity_norm[~free_mask]
            params_norm.data.clamp_(0.0, 1.0)

        # Snapshot the params that produce this cost (before the step).
        cand_norm = params_norm.detach().clone()

        # Denormalize (differentiable) and build matrix
        params_phys = _denormalize_t(params_norm, bmin, span)
        matrix = params_to_matrix(params_phys)
        if cost_fn is not None:
            cost = cost_fn(matrix)
        else:
            # zero_outside (AFNI outval=0) during optimization — see _eval_candidates.
            warped = apply_affine_interp(source, matrix, ctx.interp, base.shape, zero_outside=True)
            cost = _compute_cost(
                base, warped, weight, ctx, voxdims, matrix=matrix, blokrad_mm=blokrad_mm
            )

        loss = -cost
        loss.backward()

        # Zero gradients for frozen params
        if params_norm.grad is not None:
            params_norm.grad.data[~free_mask] = 0.0

        optimizer.step()

        # On-device best tracking — no sync, so the GPU can keep going.
        with torch.no_grad():
            cur = cost.detach()
            improved = cur > best_cost_t
            best_norm = torch.where(improved, cand_norm, best_norm)
            best_cost_t = torch.maximum(best_cost_t, cur)

        if it % sync_every == 0 or it == n_iters - 1:
            bc = best_cost_t.item()  # the only host sync
            # Guard the first sync: with last_best == -inf the relative threshold
            # is -inf + inf == nan, and `bc > nan` is always False, which would
            # (wrongly) trip the plateau counter from iteration 0.
            thr = (
                last_best + max(abs_tol, rel_tol * abs(last_best))
                if math.isfinite(last_best)
                else -float("inf")
            )
            if bc > thr:
                last_best = bc
                no_improve = 0
            else:
                no_improve += sync_every

            if tqdm is not None and verb >= 1:
                pbar.set_postfix_str(f"cost={bc:.6f}")
            elif verb >= 2:
                print(f"    {desc} iter {it}: best={bc:.6f}")

            # Early stopping once the best has been flat (within rel_tol) for
            # `patience` iterations — handles plateaus and oscillation.
            if no_improve >= patience:
                # Only print when there's no live bar; otherwise it interleaves
                # with tqdm's in-place redraw and leaves a stale duplicate line.
                if verb >= 2 and tqdm is None:
                    print(f"    {desc} converged at iter {it} (best={bc:.6f})")
                break

    best_phys = _denormalize_t(best_norm.clamp(0.0, 1.0), bmin, span).detach().cpu().numpy()
    return best_phys, best_cost_t.item()


def _batched_sampled_cost(source_stage, points_xyz, base_pts, weight_s, blokset, ctx: CostContext):
    """Build a batched blok cost: (T,4,4) matrices -> (T,) costs (higher better).

    Scores T independent transforms against one shared base/point/blok set in a
    single set of kernels — the GPU does T× the work per launch, which is what
    lifts utilisation out of the launch-bound regime.
    """
    is_lpc = ctx.name == "lpc"

    def fn(matrices: Tensor) -> Tensor:
        warped = sample_affine_at_points_batched(
            source_stage, matrices, points_xyz, zero_outside=True, interp=ctx.interp
        )  # (T, M)
        val = local_pearson_value_batched(base_pts, warped, weight_s, blokset, ctx.ppow)  # (T,)
        c = (-val) if is_lpc else val.abs()
        if ctx.micho is not None:
            # The histogram terms are per-transform and not batched; T is small
            # (tbest+1), so a loop costs a few kernels, not a rewrite.
            c = c + torch.stack(
                [_micho_terms(ctx, base_pts, warped[t], weight_s) for t in range(warped.shape[0])]
            )
        if ctx.ov_weight > 0.0 and ctx.src_cov is not None:
            pens = torch.stack(
                [
                    _overlap_penalty(ctx, matrices[t], ctx.src_cov.shape)
                    for t in range(matrices.shape[0])
                ]
            )
            c = c - ctx.ov_weight * pens
        return c

    return fn


def _refine_pattern_batched(
    init_params_phys_list: list[np.ndarray],
    config: AffineAlignConfig,
    bounds: np.ndarray,
    device: torch.device,
    batched_cost_fn,
    verb: int = 1,
    n_iters: int = 150,
    desc: str = "Pattern",
    h0: float = 0.04,
    h_min: float = 2e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Derivative-free coordinate pattern search over T trials, one batched cost per step.

    Why this instead of Adam
    ------------------------
    Adam is a high-dimensional stochastic-gradient method. This problem is the
    opposite: 6-12 parameters and one expensive cost. Two measurements decide it.

    1. A batched cost is nearly free in the batch dimension -- B=1 and B=64 take
       the same wall time, because the evaluation is launch-bound rather than
       compute-bound. Evaluations are therefore *not* the scarce resource;
       sequential steps are.
    2. The autograd gradient disagrees with a finite difference of the same
       function (cos ~ -0.44 at h=1e-4, only reaching 0.96 at h=1e-2). The cost
       surface is genuinely jagged below ~1e-2, so a gradient is a poor guide
       exactly where Adam spends most of its iterations.

    So we spend the free batch dimension on *search* rather than on a gradient.
    Each iteration evaluates, in one batched forward, a full coordinate-wise line
    search: every free parameter displaced by +/- h at several scales at once.
    That is a better-informed step than one Adam step, costs a quarter of the
    wall time (no backward: 658 of the 837 kernels in an Adam step are the
    backward pass), and never consults a derivative.

    The step size h contracts on a failed iteration and expands when the winner
    sits at the coarsest scale, which is what gives the method its convergence
    guarantee on continuous functions and its tolerance of small discontinuities.

    Returns:
        (params_phys, costs): (T, 12) refined params and (T,) best costs.
    """
    free_mask = _get_free_mask(config.dof)
    free_idx = np.flatnonzero(free_mask)
    nfree = int(free_idx.size)
    bmin, span = _bounds_to_torch(bounds, device)

    init = torch.tensor(np.stack(init_params_phys_list), dtype=torch.float32, device=device)
    T = init.shape[0]
    x = _normalize_t(init, bmin, span).clamp_(0.0, 1.0)  # (T, 12)

    # Directions are built once: +/- each free axis, plus a set of fixed random
    # directions in the free subspace, each offered at three scales.
    #
    # The random directions are not decoration. A pure coordinate search stalls
    # in a narrow diagonal valley -- and this problem has one, because rotation
    # and translation are strongly correlated for an off-centre object, so the
    # descent direction is rarely axis-aligned. Coordinate-only search converged
    # to a measurably worse cost here. They are free: the batch dimension costs
    # nothing until it is very large, so widening the stencil is pure profit.
    n_rand = nfree
    gen = torch.Generator(device="cpu").manual_seed(0)  # fixed: keep runs reproducible
    rdir = torch.zeros(n_rand, 12)
    rr = torch.randn(n_rand, nfree, generator=gen)
    rr /= rr.norm(dim=1, keepdim=True).clamp_min(1e-12)
    rdir[:, torch.as_tensor(free_idx)] = rr
    rdir = rdir.to(device)

    unit = torch.zeros(2 * nfree, 12, device=device)
    for a, j in enumerate(free_idx):
        unit[2 * a, j] = 1.0
        unit[2 * a + 1, j] = -1.0

    base_dirs = torch.cat([unit, rdir, -rdir], dim=0)  # (D, 12)
    scales = (1.0, 0.3)
    offs = torch.cat([sc * base_dirs for sc in scales], dim=0)  # (K, 12)
    K = offs.shape[0]
    cand_scale = torch.cat(
        [torch.full((base_dirs.shape[0],), sc, device=device) for sc in scales]
    )  # (K,)

    h = torch.full((T,), float(h0), device=device)
    best_x = x.clone()
    best_c = torch.full((T,), -float("inf"), device=device)
    alive = torch.ones(T, dtype=torch.bool, device=device)

    def _costs(flat_x: Tensor) -> Tensor:
        return batched_cost_fn(params_to_matrix_batched(_denormalize_t(flat_x, bmin, span)))

    # Seed the incumbent cost for all trials in one evaluation.
    best_c = _costs(x)
    n_eval = T

    pbar = _tqdm_bar(range(n_iters), total=n_iters, desc=desc, disable=verb < 1)
    momentum = torch.zeros_like(x)  # last accepted displacement, per trial
    mom_scales = (1.0, 2.0, 4.0)  # a pattern move is worth trying long
    stalled = torch.zeros(T, dtype=torch.long, device=device)
    prev_best = best_c.clone()
    patience = 12

    for _it in pbar:
        # (T, K, 12) candidates = incumbent + h * offset, clamped to the box.
        cand = (x[:, None, :] + h[:, None, None] * offs[None, :, :]).clamp_(0.0, 1.0)
        # Hooke-Jeeves pattern move: once a direction has worked, keep going
        # along it. This is what actually escapes the diagonal valley, and it
        # rides in the same batched evaluation as everything else.
        mom = torch.stack([(x + m * momentum).clamp(0.0, 1.0) for m in mom_scales], dim=1)
        cand = torch.cat([cand, mom], dim=1)  # (T, K + 3, 12)
        Kt = cand.shape[1]
        vals = _costs(cand.reshape(T * Kt, 12)).reshape(T, Kt)
        n_eval += T * Kt

        best_k = vals.argmax(dim=1)  # (T,)
        best_val = vals.gather(1, best_k[:, None]).squeeze(1)  # (T,)
        improved = (best_val > best_c) & alive

        # Accept the winner where it beat the incumbent; contract elsewhere.
        win_x = cand.gather(1, best_k[:, None, None].expand(T, 1, 12)).squeeze(1)
        momentum = torch.where(improved[:, None], win_x - x, momentum)
        x = torch.where(improved[:, None], win_x, x)
        best_c = torch.where(improved, best_val, best_c)
        best_x = torch.where(improved[:, None], win_x, best_x)

        # Expand when the winning step was the coarsest offered (the basin is
        # wider than h thinks), contract when nothing beat the incumbent. A
        # momentum win (index >= K) carries no scale, so it leaves h alone.
        is_dir = best_k < K
        won_coarse = improved & is_dir & (cand_scale[best_k.clamp(max=K - 1)] >= scales[0])
        h = torch.where(won_coarse, h * 1.6, h)
        h = torch.where(improved, h, h * 0.5).clamp_(max=0.5)

        # Two ways to be finished: the step has contracted below anything that
        # could matter (h is in normalized units, so h_min ~ 2e-4 is well under a
        # tenth of a voxel), or the best cost has stopped moving.
        stalled = stalled + 1
        stalled = torch.where(best_val > prev_best + 1e-7, torch.zeros_like(stalled), stalled)
        prev_best = torch.maximum(prev_best, best_val)
        alive = alive & (h > h_min) & (stalled < patience)
        if not bool(alive.any()):
            break
        if tqdm is not None and verb >= 1:
            pbar.set_postfix_str(f"best={best_c.max().item():.6f} h={h.max().item():.2e}")

    if verb >= 2:
        print(f"    {desc}: {n_eval} cost evaluations in {_it + 1} batched steps")

    best_phys = _denormalize_t(best_x.clamp(0.0, 1.0), bmin, span).detach().cpu().numpy()
    return best_phys, best_c.detach().cpu().numpy()


def _refine_adam_batched(
    init_params_phys_list: list[np.ndarray],
    config: AffineAlignConfig,
    bounds: np.ndarray,
    device: torch.device,
    batched_cost_fn,
    verb: int = 1,
    n_iters: int = 150,
    lr: float = 0.01,
    desc: str = "Adam",
    compile_fwd: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized Adam over T trials at once (one batched cost per step).

    The T trials are independent (separate parameter rows), so a single
    ``loss = -cost.sum()`` backward produces per-row gradients and Adam steps
    them together. Best tracking and the plateau early-stop are per-trial; we
    stop once *all* trials have plateaued.

    Returns:
        (params_phys, costs): (T, 12) refined params and (T,) best costs.
    """
    free_mask = torch.tensor(_get_free_mask(config.dof), dtype=torch.bool, device=device)
    bmin, span = _bounds_to_torch(bounds, device)
    identity_norm = _normalize_t(
        torch.tensor(_identity_physical(), dtype=torch.float32, device=device), bmin, span
    )

    init = torch.tensor(np.stack(init_params_phys_list), dtype=torch.float32, device=device)
    T = init.shape[0]
    params_norm = _normalize_t(init, bmin, span).clone().detach()
    params_norm.requires_grad_(True)
    optimizer = torch.optim.Adam([params_norm], lr=lr)

    best_cost_t = torch.full((T,), -float("inf"), device=device)
    best_norm = params_norm.detach().clone()
    last_best = np.full(T, -np.inf)
    no_improve = np.zeros(T, dtype=np.int64)
    rel_tol, abs_tol, patience, sync_every = 1e-4, 1e-6, 40, 15

    # Forward = normalized params -> (T,) cost. The per-iter cost is launch-bound
    # (dozens of small kernels: matrix build + sample + blok scatter), so the
    # whole forward is a torch.compile target. Opt-in (FFS_ALLINEATE_COMPILE=1)
    # while it's validated for speed on real GPUs; the sync / early-stop break
    # live outside it, so they don't fragment the compiled graph.
    def _forward(pn: Tensor) -> Tensor:
        return batched_cost_fn(params_to_matrix_batched(_denormalize_t(pn, bmin, span)))

    use_compile = compile_fwd or os.environ.get("FFS_ALLINEATE_COMPILE") == "1"
    forward = torch.compile(_forward) if use_compile else _forward

    pbar = _tqdm_bar(range(n_iters), total=n_iters, desc=desc, disable=verb < 1)
    for it in pbar:
        optimizer.zero_grad()
        with torch.no_grad():
            params_norm.data[:, ~free_mask] = identity_norm[~free_mask]
            params_norm.data.clamp_(0.0, 1.0)
        cand_norm = params_norm.detach().clone()
        cost = forward(params_norm)  # (T,)
        (-cost.sum()).backward()
        if params_norm.grad is not None:
            params_norm.grad.data[:, ~free_mask] = 0.0
        optimizer.step()

        with torch.no_grad():
            cur = cost.detach()
            improved = cur > best_cost_t
            best_norm = torch.where(improved[:, None], cand_norm, best_norm)
            best_cost_t = torch.maximum(best_cost_t, cur)

        if it % sync_every == 0 or it == n_iters - 1:
            bc = best_cost_t.detach().cpu().numpy()
            finite = np.isfinite(last_best)
            lb = np.where(finite, last_best, 0.0)  # avoid -inf+inf -> nan in the unused branch
            thr = np.where(finite, lb + np.maximum(abs_tol, rel_tol * np.abs(lb)), -np.inf)
            imp = bc > thr
            last_best = np.where(imp, bc, last_best)
            no_improve = np.where(imp, 0, no_improve + sync_every)
            if tqdm is not None and verb >= 1:
                pbar.set_postfix_str(f"best={bc.max():.6f}")
            if bool((no_improve >= patience).all()):
                break

    best_phys = _denormalize_t(best_norm.clamp(0.0, 1.0), bmin, span).detach().cpu().numpy()
    return best_phys, best_cost_t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Refinement: Powell (derivative-free, final polish)
# ---------------------------------------------------------------------------


def _make_powell_cost(
    base: Tensor,
    source: Tensor,
    weight: Tensor | None,
    ctx: CostContext,
    voxdims: tuple[float, float, float],
    param_bounds: np.ndarray,
    free_mask: np.ndarray,
    fixed_norm: np.ndarray,
    device: torch.device,
    counter: list | None = None,
    pbar=None,
    matrix_cost_fn=None,
):
    """Create a closure for scipy Powell (avoids kwargs issue).

    ``matrix_cost_fn``, when given, maps a (4,4) matrix to a scalar cost (higher
    == better) and replaces the full-grid path (subsampled blok refinement).
    """

    def cost_fn(x_free_norm: np.ndarray) -> float:
        # Clamp to [0,1] — Powell can overshoot, producing garbage warps
        x_clamped = np.clip(x_free_norm, 0.0, 1.0)

        full_norm = fixed_norm.copy()
        full_norm[free_mask] = x_clamped
        params_phys = _denormalize(full_norm, param_bounds)

        params_t = torch.tensor(params_phys, dtype=torch.float32, device=device)
        matrix = params_to_matrix(params_t)

        with torch.no_grad():
            if matrix_cost_fn is not None:
                cost = matrix_cost_fn(matrix)
            else:
                # zero_outside (AFNI outval=0) during optimization — see _eval_candidates.
                warped = apply_affine_interp(
                    source, matrix, ctx.interp, base.shape, zero_outside=True
                )
                cost = _compute_cost(base, warped, weight, ctx, voxdims, matrix=matrix)

        val = -cost.item()

        if counter is not None:
            counter[0] += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"cost={-val:.6f}")

        return val

    return cost_fn


def _refine_powell(
    base: Tensor,
    source: Tensor,
    weight: Tensor | None,
    init_params_phys: np.ndarray,
    config: AffineAlignConfig,
    ctx: CostContext,
    voxdims: tuple[float, float, float],
    bounds: np.ndarray,
    device: torch.device,
    verb: int = 1,
    initial_step: float = 0.066,
    ftol: float = 1e-4,
    maxfev: int = 500,
    desc: str = "Powell",
    cost_fn=None,
) -> tuple[np.ndarray, float]:
    """Refine parameters using Powell's method (derivative-free).

    GPU-accelerated cost evaluation; only the optimizer loop is CPU.
    Parameters normalized to [0,1] for proper scaling.
    Clamped to [0,1] to prevent wild line search overshoots.

    Returns:
        (params_phys, cost): refined params and final cost.
    """
    free_mask = _get_free_mask(config.dof)
    nfree = int(free_mask.sum())

    identity_norm = _normalize(_identity_physical(), bounds)
    fixed_norm = identity_norm.copy()

    init_norm = _normalize(init_params_phys, bounds)
    x0 = np.clip(init_norm[free_mask], 0.0, 1.0)  # clamp to valid range

    # Set up progress bar and counter
    counter = [0]
    pbar = None
    if tqdm is not None and verb >= 1:
        pbar = tqdm(total=maxfev, desc=desc, file=sys.stderr, leave=True, ncols=80)

    powell_cost = _make_powell_cost(
        base,
        source,
        weight,
        ctx,
        voxdims,
        bounds,
        free_mask,
        fixed_norm,
        device,
        counter=counter,
        pbar=pbar,
        matrix_cost_fn=cost_fn,
    )

    # Powell with bounds to keep params in [0,1]
    param_bounds_01 = [(0.0, 1.0)] * nfree

    start_cost = -powell_cost(x0)  # cost at the incoming (refined) params

    result = minimize(
        powell_cost,
        x0,
        method="Powell",
        bounds=param_bounds_01,
        options={
            "maxfev": maxfev,
            "ftol": ftol,
            "direc": np.eye(nfree) * initial_step,
        },
    )

    if pbar is not None:
        pbar.close()

    final_cost = -result.fun
    # Powell's line search can drift out of a narrow basin and converge to a
    # worse point than it started (seen with the peaky lpc/blok surface). Never
    # return worse than the start — the polish must only ever help.
    if final_cost < start_cost:
        if verb >= 1:
            print(
                f"    {desc}: kept pre-polish (start={start_cost:.6f} "
                f">= powell={final_cost:.6f}, {counter[0]} evals)"
            )
        return init_params_phys.copy(), start_cost

    full_norm = fixed_norm.copy()
    full_norm[free_mask] = np.clip(result.x, 0.0, 1.0)
    params_phys = _denormalize(full_norm, bounds)

    if verb >= 1:
        # Report the gain, not just the endpoint: the polish costs a few hundred
        # cost evaluations, and "accepted" is not the same as "worth it" -- on a
        # same-modality pair the improvement is often in the 6th decimal.
        gain = final_cost - start_cost
        print(
            f"    {desc}: cost={final_cost:.6f} (start={start_cost:.6f}, "
            f"gain={gain:+.2e}, {counter[0]} evals)"
        )

    return params_phys, final_cost


# ---------------------------------------------------------------------------
# Stage 3: Progressive refinement
# ---------------------------------------------------------------------------


def _refine_progressive(
    base: Tensor,
    source: Tensor,
    weight: Tensor | None,
    trial_params_list: list[np.ndarray],
    config: AffineAlignConfig,
    ctx: CostContext,
    voxdims: tuple[float, float, float],
    bounds: np.ndarray,
    device: torch.device,
    verb: int = 1,
    sample: SampleSet | None = None,
) -> tuple[np.ndarray, float]:
    """Progressive multi-resolution refinement: Adam (GPU) + Powell polish.

    Blur pyramid (full resolution throughout — NO decimation):
      - Blurred stage: smooth base+source (σ≈2 vox) → wide basin, all trials
      - Sharp stage: full resolution → Adam, deduplicate to best trial
      - Powell polish on best

    AFNI widens the capture basin by *smoothing* both images and shrinking the
    blur across passes (3dAllineate ``smooth_radius_*``), never by downsampling:
    decimation changes the blok lattice geometry and aliases the cost surface,
    which for the local-Pearson costs drove the optimiser away from the true
    basin. For the blok costs the blok radius is inflated by the blur radius
    (AFNI ``rad=sqrt(blokrad^2+smooth^2)``) so the now-correlated neighbours
    don't dominate the within-blok variance.

    Returns:
        (params, cost): the (12,) best refined parameters (physical units) and
        the best cost achieved (the in-mask sampled cost for blok costs).
    """
    free_mask = _get_free_mask(config.dof)
    trials = [(0.0, p.copy()) for p in trial_params_list]

    vmean = sum(voxdims) / 3.0

    def _stage_blokrad(sigma_vox: float) -> float | None:
        if ctx.blokrad_mm is None or sigma_vox <= 0.0:
            return ctx.blokrad_mm
        return math.sqrt(ctx.blokrad_mm**2 + (sigma_vox * vmean) ** 2)

    # Subsampled blok cost: evaluate lpa/lpc on the fixed point set in ``sample``
    # (point-wise warp + point blok lattice), so each iteration is O(M) instead
    # of O(all voxels). Only for the blok costs; other costs keep the full grid.
    use_sample = sample is not None and ctx.name in _BLOK_COSTS

    def _sampled_cost_fn(base_stage: Tensor, source_stage: Tensor, blokrad_stage, samp: SampleSet):
        base_pts = base_stage.reshape(-1)[samp.idx_flat]
        blokset_s = samp.blokset(blokrad_stage, device)
        is_lpc = ctx.name == "lpc"

        def fn(matrix: Tensor) -> Tensor:
            warped_s = sample_affine_at_points(
                source_stage, matrix, samp.points_xyz, zero_outside=True, interp=ctx.interp
            )
            val = local_pearson_value(base_pts, warped_s, samp.weight_s, blokset_s, ppow=ctx.ppow)
            c = (-val) if is_lpc else val.abs()
            if ctx.micho is not None:
                c = c + _micho_terms(ctx, base_pts, warped_s, samp.weight_s)
            if ctx.ov_weight > 0.0 and ctx.src_cov is not None:
                c = c - ctx.ov_weight * _overlap_penalty(ctx, matrix, ctx.src_cov.shape)
            return c

        return fn

    # Stages: (sigma_vox, n_iters, lr, dedup_after). A blurred basin-widening
    # stage first (only when the volume is big enough for blurring to matter),
    # then the sharp full-resolution stage. Translations stay in full-grid
    # voxels throughout (no bound rescaling), since we never decimate.
    #
    # All stages use the full match-point set: refinement is launch-bound (each
    # Adam iteration is dominated by kernel-launch / autograd overhead, ~constant
    # for ≳300k points), so a per-stage point ramp bought no speed and only made
    # the blur trajectories noisier (changing how many trials survived dedup).
    stages: list[tuple[float, int, float, bool]] = []
    if min(base.shape) > 16:
        stages.append((2.0, config.adam_iters_2x, config.adam_lr_2x, True))
    stages.append((0.0, config.adam_iters_1x, config.adam_lr, False))

    for si, (sigma_vox, n_iters, lr, dedup_after) in enumerate(stages):
        if sigma_vox > 0.0:
            base_s = _separable_smooth_3d(base, sigma_vox)
            source_s = _separable_smooth_3d(source, sigma_vox)
            label = f"Blur σ={sigma_vox:g}vox"
        else:
            base_s, source_s = base, source
            label = "Full resolution"
        blokrad_stage = _stage_blokrad(sigma_vox)

        if verb >= 1:
            npts = f", {sample.idx_flat.numel()} pts" if use_sample else ""
            print(f"  {label} ({base_s.shape}{npts}, {len(trials)} trials):")

        if use_sample:
            # All trials in one batched Adam — T× the work per launch.
            base_pts = base_s.reshape(-1)[sample.idx_flat]
            blokset_s = sample.blokset(blokrad_stage, device)
            bcost = _batched_sampled_cost(
                source_s, sample.points_xyz, base_pts, sample.weight_s, blokset_s, ctx
            )
            if config.optimizer == "pattern":
                out_phys, out_costs = _refine_pattern_batched(
                    [p for _, p in trials],
                    config,
                    bounds,
                    device,
                    bcost,
                    verb=verb,
                    n_iters=n_iters,
                    desc=f"S{si}",
                )
            else:
                out_phys, out_costs = _refine_adam_batched(
                    [p for _, p in trials],
                    config,
                    bounds,
                    device,
                    bcost,
                    verb=verb,
                    n_iters=n_iters,
                    lr=lr,
                    desc=f"S{si}",
                    compile_fwd=config.compile,
                )
            refined = [(float(out_costs[t]), out_phys[t]) for t in range(len(trials))]
        else:
            refined = []
            for i, (_, params) in enumerate(trials):
                params_out, cost = _refine_adam_normalized(
                    base_s,
                    source_s,
                    weight,
                    params,
                    config,
                    ctx,
                    voxdims,
                    bounds,
                    device,
                    verb=verb,
                    n_iters=n_iters,
                    lr=lr,
                    desc=f"S{si} T{i}",
                    blokrad_mm=blokrad_stage,
                )
                refined.append((cost, params_out))
        refined.sort(key=lambda c: -c[0])

        if dedup_after:
            # Keep the best plus any trial not too close to a better one, so the
            # sharp stage still explores distinct basins.
            trials = [refined[0]]
            for j in range(1, len(refined)):
                nj = _normalize(refined[j][1], bounds)
                too_close = any(
                    np.max(np.abs(nj[free_mask] - _normalize(pk, bounds)[free_mask])) < 0.02
                    for _, pk in trials
                )
                if not too_close:
                    trials.append(refined[j])
        else:
            trials = refined

        if sigma_vox > 0.0:
            del base_s, source_s

    best_cost, best_params = trials[0]

    # AFNI's +ZZ: the combination widened the basin and got us here, but the
    # answer we want is the pure lpc/lpa optimum, so drop the extra terms before
    # the final pass. Mutating ctx is deliberate — every cost closure captures ctx
    # and nothing else, so this switches all of them at once.
    #
    # This must stay OUTSIDE the polish block. It used to live inside it, so
    # -fast / -superfast (which set powell_maxfev=0) silently returned the
    # *combined* optimum for a +ZZ cost — the one thing +ZZ exists to avoid.
    if ctx.micho is not None and ctx.micho_zfinal:
        if verb >= 1:
            print(f"    +ZZ: finishing on pure {ctx.name} (combination terms dropped)")
        ctx.micho = None
        if config.powell_maxfev <= 0:
            # Nothing downstream will re-optimize, so the reported cost must at
            # least be the pure one the caller asked for, not the combined one.
            with torch.no_grad():
                _m = params_to_matrix(torch.tensor(best_params, dtype=torch.float32, device=device))
                _w = apply_affine_interp(source, _m, ctx.interp, base.shape, zero_outside=True)
                best_cost = float(_compute_cost(base, _w, weight, ctx, voxdims, matrix=_m))
                del _w

    # --- Powell polish (single pass, tighter convergence) ---
    if config.powell_maxfev > 0:
        if verb >= 1:
            print("  Powell polish (full resolution):")

        polish_cost_fn = (
            _sampled_cost_fn(base, source, ctx.blokrad_mm, sample) if use_sample else None
        )
        best_params, best_cost = _refine_powell(
            base,
            source,
            weight,
            best_params,
            config,
            ctx,
            voxdims,
            bounds,
            device,
            verb=verb,
            initial_step=0.033,
            ftol=1e-4,
            maxfev=config.powell_maxfev,
            desc="Polish",
            cost_fn=polish_cost_fn,
        )

    return best_params, best_cost


# ---------------------------------------------------------------------------
# Grid transform computation
# ---------------------------------------------------------------------------


def _compute_grid_matrix(
    source_affine: np.ndarray,
    base_affine: np.ndarray,
    device: torch.device,
) -> Tensor:
    """Compute the voxel grid transform: base voxel → source voxel."""
    base_ijk2xyz = torch.from_numpy(base_affine.astype(np.float64)).float().to(device)
    source_xyz2ijk = torch.linalg.inv(
        torch.from_numpy(source_affine.astype(np.float64)).float().to(device)
    )
    return source_xyz2ijk @ base_ijk2xyz


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def allineate(
    base: Tensor,
    source: Tensor,
    config: AffineAlignConfig | None = None,
    base_header: dict | None = None,
    source_header: dict | None = None,
    save_automask_path: str | None = None,
    save_cmass_path: str | None = None,
    save_weight_path: str | None = None,
) -> tuple[Tensor, Tensor]:
    """GPU-accelerated affine/rigid alignment.

    Aligns source to base using a 3-stage pipeline:
      1. Center-of-mass pre-alignment
      2. GPU-parallel coarse rotation search
      3. Progressive refinement (Adam GPU + Powell polish)

    Args:
        base: (nz, ny, nx) base/reference image.
        source: (nz, ny, nx) source/moving image.
        config: Alignment configuration.
        base_header: Header info from load_image.
        source_header: Header info from load_image.
        save_automask_path: If set, save the computed automask here.
        save_cmass_path: If set, save the source positioned by the cmass shift
            alone (no rotation/scale), on the base grid — lets you eyeball the
            initial placement and reproduce it with config.cmass_direct.

    Returns:
        (matrix, warped):
            matrix: (4, 4) affine mapping base voxels → source (native) voxels.
            warped: the aligned source image on the base grid.
    """
    if config is None:
        config = AffineAlignConfig()

    if config.device is not None:
        device = torch.device(config.device)
    else:
        device = base.device

    base = base.to(device)
    source_native = source.to(device)
    verb = config.verb

    # Pre-resample the source onto the base grid, folding in the center-of-mass
    # (or manual) shift BEFORE resampling. Computing the shift on the native
    # volumes and baking it into the base→source map is what makes very different
    # FOVs/orientations work: resampling the source at its un-shifted position
    # samples wherever the base grid points, so when the brains are far apart in
    # scanner space the resample captures the *wrong* part of the source and the
    # cost can never recover — it just drifts back to that wrong place. With the
    # shift baked in, the source brain lands in the base FOV up front and the
    # optimiser is left only a small residual. ``align_matrix`` (base voxel →
    # source-native voxel, shift applied first) is carried through to the final
    # transform so the chain stays consistent.
    grid_matrix = None
    source_validity = None
    cross_grid = (
        base_header is not None and source_header is not None and source.shape != base.shape
    )
    if cross_grid:
        if verb >= 1:
            print(f"Resampling source {source.shape} to base grid {base.shape}")
        grid_matrix = _compute_grid_matrix(source_header["affine"], base_header["affine"], device)

    cmass_shift = np.zeros(3)
    if config.cmass_direct is not None:
        cmass_shift = np.asarray(config.cmass_direct, dtype=np.float64)
        if verb >= 1:
            t = cmass_shift
            print(f"  Direct cmass shift: dx={t[0]:.2f}, dy={t[1]:.2f}, dz={t[2]:.2f} voxels")
    elif config.cmass:
        translation = _cmass_translation(base, source_native, grid_matrix=grid_matrix)
        cmass_shift = translation.cpu().numpy()
        if verb >= 1:
            t = cmass_shift
            print(f"  Center-of-mass shift: dx={t[0]:.2f}, dy={t[1]:.2f}, dz={t[2]:.2f} voxels")

    g = grid_matrix if grid_matrix is not None else torch.eye(4, device=device)
    cmass_mat = torch.eye(4, device=device)
    cmass_mat[:3, 3] = torch.as_tensor(cmass_shift, device=device, dtype=cmass_mat.dtype)
    align_matrix = g @ cmass_mat

    if cross_grid or bool(np.any(cmass_shift != 0.0)):
        source_validity = _compute_source_validity_mask(
            source_native.shape, base.shape, align_matrix
        )
        source_on_base = apply_affine(source_native, align_matrix, base.shape, zero_outside=True)
        source_on_base = source_on_base * source_validity.float()
        if verb >= 1 and cross_grid:
            n_valid = int(source_validity.sum().item())
            n_total = source_validity.numel()
            print(
                f"  Source covers {n_valid}/{n_total} base voxels "
                f"({100.0 * n_valid / n_total:.1f}%)"
            )
    else:
        source_on_base = source_native

    # Weight image — AFNI 3dAllineate mri_weightize: histogram clip level, median
    # pre-filter, and the bottom-clip + largest-cluster + erode cleanup that drops
    # the background (otherwise the smoothed weight fills the whole FOV).
    weight = (
        compute_weight_image(
            base,
            edge_fraction=0.05,
            median_radius=2.25,
            clusterize=True,
            hist_cliplevel=True,
        )
        if config.autoweight
        else None
    )

    # Apply source validity mask to weight (exclude base voxels outside source)
    if source_validity is not None and weight is not None:
        weight = weight * source_validity.float()

    # Source automask: restrict the SOURCE to its own object, the way AFNI does
    # (the source automask is applied to the source data, never to the base
    # weight). Folding this binary mask into the weight was wrong twice over: it
    # reshaped the base-space weight into the source's outline, and it replaced
    # the autoweight's smooth, SNR-graded halo with a hard edge — discarding the
    # soft base-edge contribution that lpc leans on. The weight stays purely
    # base-derived (autoweight, optionally FOV-clipped to source coverage).
    if config.source_automask:
        if verb >= 1:
            print("Computing source automask...")
        src_mask = automask(source_on_base, device=device)
        source_on_base = source_on_base * src_mask.float()
        if verb >= 1:
            n = int(src_mask.sum().item())
            print(
                f"  Source automask: {n}/{src_mask.numel()} voxels "
                f"({100.0 * n / src_mask.numel():.1f}%)"
            )

        if save_automask_path is not None:
            from .io import save_image

            save_image(src_mask.float(), save_automask_path, header_info=base_header)
            if verb >= 1:
                print(f"  Saved automask: {save_automask_path}")

    # Diagnostic: dump the exact weight the optimiser sees (autoweight × source
    # validity × source automask), on the full base grid. Compare against AFNI's
    # 3dAllineate -wtprefix to check the weight/mask is emphasising the right
    # voxels (e.g. that a faded brain edge isn't being dropped).
    if save_weight_path is not None:
        from .io import save_image

        w_save = weight if weight is not None else torch.ones_like(base)
        save_image(w_save, save_weight_path, header_info=base_header)
        if verb >= 1:
            print(f"  Saved optimisation weight: {save_weight_path}")

    if verb >= 1:
        print(f"Allineate: {config.dof} alignment, cost={config.cost}")
        print(f"  Base shape: {base.shape}, device: {device}")

    # Free source_native from GPU during optimization — it's only needed
    # for final resampling. Can be large (e.g., 320³ anat = 131 MB).
    if source_native.shape != base.shape:
        source_native = source_native.cpu()

    # Auto-crop: remove zero margins for faster optimization
    # We optimize on the cropped grid, then adjust the final matrix
    crop_offset = None
    _base_full = base
    _source_on_base_full = source_on_base
    _weight_full = weight

    if config.autocrop:
        base_crop, source_crop, weight_crop, offset = _crop_volumes(base, source_on_base, weight)
        if base_crop.shape != base.shape:
            crop_offset = offset
            if verb >= 1:
                savings = 100.0 * (1.0 - (base_crop.numel() / base.numel()))
                print(
                    f"  Auto-crop: {base.shape} → {base_crop.shape} ({savings:.0f}% fewer voxels)"
                )
            # Use cropped volumes for optimization
            base_opt = base_crop
            source_opt = source_crop
            weight_opt = weight_crop
        else:
            base_opt = base
            source_opt = source_on_base
            weight_opt = weight
    else:
        base_opt = base
        source_opt = source_on_base
        weight_opt = weight

    # Build the cost context (blok geometry + histogram clips), reused for
    # every cost evaluation in this run.
    voxdims = _voxdims_from_header(base_header)
    cost_name, micho, micho_zfinal = _parse_cost(config.cost)
    blokrad_mm = config.blokrad
    if cost_name in _BLOK_COSTS and blokrad_mm is None:
        from .cost_blok import auto_blok_radius

        blokrad_mm = auto_blok_radius(voxdims, config.bloktype)
    base_clip = source_clip = None
    # The lpc+/lpa+ combinations lean on the histogram functionals too, so they
    # need the same clip ranges the pure histogram costs get.
    if cost_name in _HIST_COSTS or micho is not None:
        base_clip = clip_range(base_opt)
        source_clip = clip_range(source_opt)

    # Overlap penalty inputs (only when -ov is requested). ``src_cov`` is the
    # source coverage on the optimisation grid (a binary automask — grid_sample's
    # linear interpolation already gives a differentiable boundary, and a blurred
    # mask would leak mass outside ``base_dom`` and read overlap < 1 even at the
    # true alignment, breaking AFNI's 9.95/10 calibration). ``base_dom`` is the
    # base brain domain; ``ov_denom`` mirrors AFNI's MIN(nbsmask, najmask) so a
    # source brain smaller than the base can still reach overlap ~1.
    src_cov = base_dom = None
    ov_denom = 1.0
    if config.ov > 0.0:
        base_dom = automask(base_opt, device=device).float()
        src_cov = automask(source_opt, device=device).float()
        ov_denom = float(min(base_dom.sum().item(), src_cov.sum().item()))

    ctx = CostContext(
        name=cost_name,
        micho=micho,
        micho_zfinal=micho_zfinal,
        sigma=config.lpa_sigma,
        kernel=config.lpa_kernel,
        bloktype=config.bloktype,
        blokrad_mm=blokrad_mm,
        base_voxdims=voxdims,
        ppow=config.ppow,
        base_clip=base_clip,
        source_clip=source_clip,
        ov_weight=config.ov,
        src_cov=src_cov,
        base_dom=base_dom,
        ov_denom=ov_denom,
        interp=config.interp,
    )

    # Stage 1: the optimiser starts from identity — the cmass/manual shift is
    # already folded into ``align_matrix`` (the source was resampled through it
    # above), so only a small residual remains to be found.
    init_params = identity_params(device=device)

    # Parameter bounds centred on identity (the residual after the baked shift).
    bounds = _compute_param_bounds(
        base_opt.shape,
        range_scale=config.range_scale,
        rot_range=config.coarse_range,
        shift_frac=config.coarse_shift_frac,
    )

    # Match-point subsampling for the blok costs: build a fixed random subset of
    # the weight domain once (point-wise sampling), shared by the joint coarse
    # search and the refinement, so both are O(n_match) instead of O(all voxels)
    # and the bloks are populated only by brain points, not the background.
    sample = None
    if cost_name in _BLOK_COSTS:
        sample = _build_sample_set(
            base_opt, weight_opt, voxdims, config.n_match, config.bloktype, device
        )
        if sample is not None and verb >= 1:
            n_dom = int((weight_opt > 0).sum()) if weight_opt is not None else base_opt.numel()
            print(
                f"  Match-point subsampling: {sample.idx_flat.numel()} points (of {n_dom} in domain)"
            )

    # Stage 2: coarse search to seed the refinement.
    if not config.twopass:
        trial_params_list = [init_params.cpu().numpy().copy()]
    elif sample is not None:
        # Blok costs: joint (translation+rotation) subsampled coarse search, at
        # full resolution (no decimation) — see _coarse_search_joint.
        trial_params_list = _coarse_search_joint(
            base_opt, source_opt, sample, ctx, config, bounds, device, verb
        )
    else:
        # Other costs: the split translation→rotation sweep on a downsampled grid.
        min_dim = min(base_opt.shape)
        max_ds = 2 if ctx.name in _HIST_COSTS else 4
        if min_dim >= 128:
            ds_factor = max_ds
        elif min_dim >= 64:
            ds_factor = min(2, max_ds)
        else:
            ds_factor = 1

        if ds_factor > 1:
            base_ds = _downsample_3d(_smooth_to_resolution(base_opt, ds_factor), ds_factor)
            source_ds = _downsample_3d(_smooth_to_resolution(source_opt, ds_factor), ds_factor)
            weight_ds = _downsample_3d(weight_opt, ds_factor) if weight_opt is not None else None

            coarse_init = init_params.clone()
            coarse_init[:3] /= ds_factor

            if verb >= 1:
                print(f"  Coarse resolution ({base_ds.shape}, {ds_factor}x downsample):")

            voxdims_coarse = tuple(v * ds_factor for v in voxdims)
            best_list = _coarse_search(
                base_ds,
                source_ds,
                weight_ds,
                config,
                ctx,
                voxdims_coarse,
                coarse_init[:3],
                device,
                verb,
            )

            trial_params_list = []
            for p in best_list:
                p_np = p.cpu().numpy().copy()
                p_np[:3] *= ds_factor
                trial_params_list.append(p_np)

            del base_ds, source_ds, weight_ds
        else:
            best_list = _coarse_search(
                base_opt,
                source_opt,
                weight_opt,
                config,
                ctx,
                voxdims,
                init_params[:3],
                device,
                verb,
            )
            trial_params_list = [p.cpu().numpy().copy() for p in best_list]

    # Stage 3: Progressive refinement (Adam GPU + Powell polish)
    if verb >= 1:
        print("Refinement phase:")

    best_params_phys, best_refine_cost = _refine_progressive(
        base_opt,
        source_opt,
        weight_opt,
        trial_params_list,
        config,
        ctx,
        voxdims,
        bounds,
        device,
        verb,
        sample=sample,
    )

    # Build final matrix — adjust for crop offset if needed.
    def _residual_to_final(residual: Tensor) -> Tensor:
        """Map a cropped-base→cropped-source residual to the full base→native pull.

        The residual maps cropped-base voxels → cropped-source voxels. Source was
        cropped identically, so undo the crop by conjugating with the offset
        (full_base_voxel = crop_base_voxel + offset → T(+off) @ M @ T(-off)), then
        compose ``align_matrix`` — the base→source map that already carries the
        cmass shift the source was resampled through.
        """
        if crop_offset is not None:
            x_off, y_off, z_off = crop_offset
            T_pos = torch.eye(4, device=device, dtype=torch.float32)
            T_neg = torch.eye(4, device=device, dtype=torch.float32)
            T_pos[0, 3], T_pos[1, 3], T_pos[2, 3] = float(x_off), float(y_off), float(z_off)
            T_neg[0, 3], T_neg[1, 3], T_neg[2, 3] = float(-x_off), float(-y_off), float(-z_off)
            residual = T_pos @ residual @ T_neg
        return align_matrix @ residual

    best_t = torch.tensor(best_params_phys, dtype=torch.float32, device=device)
    final_matrix = _residual_to_final(params_to_matrix(best_t))

    # AFNI-style final-fit parameter report. The final matrix is voxel base->
    # source; convert to AFNI DICOM mm when both affines are known so the numbers
    # are directly comparable to 3dAllineate, otherwise report in voxel space.
    if verb >= 1:
        from .affine import decompose_affine_sdu, format_final_fit_params, voxel_matrix_to_dicom

        if base_header is not None and source_header is not None:
            m_report = voxel_matrix_to_dicom(
                final_matrix, base_header["affine"], source_header["affine"]
            )
            space = "DICOM mm"
        else:
            m_report = final_matrix
            space = "voxel index"
        print(format_final_fit_params(decompose_affine_sdu(m_report), space=space))

    # Final output: single-step resampling from native source
    # Bring source_native back to GPU if it was offloaded
    source_native = source_native.to(device)

    # Optional diagnostic: write the source positioned by the cmass shift alone
    # (no refinement), so the placement can be eyeballed and reproduced or
    # hand-tuned via -cmass_direct. With identity residual this is exactly
    # ``align_matrix`` — the baked-in shift applied to the native source.
    if save_cmass_path is not None:
        from .io import save_image

        cmass_final = _residual_to_final(params_to_matrix(init_params))
        if config.final_interp == "wsinc5":
            cmass_warped = apply_affine_wsinc5(source_native, cmass_final, base.shape)
        else:
            cmass_warped = apply_affine(source_native, cmass_final, base.shape, zero_outside=True)
        save_image(cmass_warped, save_cmass_path, header_info=base_header)
        if verb >= 1:
            print(f"  Saved cmass-shifted source: {save_cmass_path}")
    if config.final_interp == "wsinc5":
        if verb >= 1:
            print("Applying final wsinc5 interpolation...")
        warped = apply_affine_wsinc5(source_native, final_matrix, base.shape)
    else:
        warped = apply_affine(source_native, final_matrix, base.shape, zero_outside=True)

    if verb >= 1:
        # For blok costs, report the in-mask cost the optimiser actually achieved
        # (returned from refinement) — it's computed on the brain match points,
        # not the whole grid, so it isn't diluted by background bloks, and it's
        # free (no extra full-grid blok lattice to build). Other costs print the
        # full-grid value of the final warp.
        if cost_name in _BLOK_COSTS and sample is not None:
            print(f"Final cost ({config.cost}, in-mask): {best_refine_cost:.6f}")
        else:
            final_cost = _compute_cost(base, warped, weight, ctx, voxdims)
            print(f"Final cost: {final_cost.item():.6f}")

    return final_matrix, warped
