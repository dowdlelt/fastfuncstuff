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
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from torch import Tensor

from . import cost_hist
from .affine import (
    apply_affine,
    apply_affine_batched,
    apply_affine_wsinc5,
    identity_params,
    params_to_matrix,
    params_to_matrix_batched,
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

    # Coarse search. Ranges mirror 3dAllineate's defaults: angle ±30°,
    # shift ±32% of grid size, scale ±20%. ``range_scale`` shrinks all of them
    # (-smallrange -> 0.5, -verysmallrange -> 0.25).
    twopass: bool = True
    coarse_range: float = 30.0  # rotation half-range, degrees
    coarse_step: float = 5.0  # rotation step, degrees
    coarse_shift_frac: float = 0.32  # translation half-range (fraction of grid)
    coarse_shift_steps: int = 7  # samples per translation axis (odd, incl. 0)
    coarse_scale_range: float = 0.20  # scale half-range (fraction)
    range_scale: float = 1.0  # global shrink for all coarse ranges + bounds
    tbest: int = 3  # best coarse candidates to carry into refinement

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


# ---------------------------------------------------------------------------
# Cost function dispatch
# ---------------------------------------------------------------------------

# Costs that need the blok lattice (AFNI-faithful local Pearson).
_BLOK_COSTS = ("lpa", "lpc")
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
    _blok_cache: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self._blok_cache is None:
            self._blok_cache = {}

    def blokset(self, shape, voxdims, device) -> BlokSet:
        """Get (or build + cache) the blok assignment for one grid.

        Cached per (shape, rounded voxdims) so the lattice is computed once and
        reused across every optimiser iteration on that grid.
        """
        key = (tuple(shape), tuple(round(v, 4) for v in voxdims))
        bs = self._blok_cache.get(key)
        if bs is None:
            bs = assign_bloks(tuple(shape), voxdims, self.bloktype, self.blokrad_mm, device=device)
            self._blok_cache[key] = bs
        return bs


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


def _compute_cost(
    base: Tensor,
    warped: Tensor,
    weight: Tensor | None,
    ctx: CostContext,
    voxdims: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tensor:
    """Compute alignment cost for one warped volume (higher == better)."""
    name = ctx.name
    if name in ("ls", "pearclp"):
        return clipped_pearson_correlation(
            base.reshape(-1),
            warped.reshape(-1),
            weight.reshape(-1) if weight is not None else None,
        )
    if name == "lps":
        return lpa_correlation(base, warped, weight, sigma=ctx.sigma, kernel_type=ctx.kernel)
    if name == "lpsc":
        return lpc_correlation(base, warped, weight, sigma=ctx.sigma, kernel_type=ctx.kernel)
    if name in _BLOK_COSTS:
        bs = ctx.blokset(base.shape, voxdims, base.device)
        fn = lpa_cost if name == "lpa" else lpc_cost
        return fn(base, warped, weight, bs, ppow=ctx.ppow)
    if name in _HIST_COSTS:
        kw = dict(weight=weight, base_clip=ctx.base_clip, source_clip=ctx.source_clip)
        if name == "mi":
            return cost_hist.mi_cost(base, warped, **kw)
        if name == "nmi":
            return cost_hist.nmi_cost(base, warped, **kw)
        if name == "je":
            return cost_hist.je_cost(base, warped, **kw)
        if name == "hel":
            return cost_hist.hel_cost(base, warped, **kw)
        mode = {"cru": "u", "cra": "a", "crm": "m"}[name]
        return cost_hist.cr_cost(base, warped, mode=mode, **kw)
    raise ValueError(f"Unknown cost function: {name}")


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
    tx, ty, tz = 0.321 * rs * (nx - 1), 0.321 * rs * (ny - 1), 0.321 * rs * (nz - 1)
    bounds[0] = [cmass_shift[0] - tx, cmass_shift[0] + tx]
    bounds[1] = [cmass_shift[1] - ty, cmass_shift[1] + ty]
    bounds[2] = [cmass_shift[2] - tz, cmass_shift[2] + tz]

    bounds[3:6] = [-30.0 * rs, 30.0 * rs]  # rotations (degrees)
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
    chunk_size = _estimate_chunk_size(base.shape, device)
    B = candidates.shape[0]
    all_costs = []
    chunks = range(0, B, chunk_size)
    for start in _tqdm_bar(chunks, total=len(range(0, B, chunk_size)), desc=desc, disable=verb < 1):
        end = min(start + chunk_size, B)
        with torch.no_grad():
            warped = apply_affine_batched(source, matrices[start:end], base.shape)
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


def _estimate_chunk_size(vol_shape: tuple[int, ...], device: torch.device) -> int:
    """Estimate how many candidates we can process at once."""
    voxels = vol_shape[0] * vol_shape[1] * vol_shape[2]
    if device.type == "cuda":
        try:
            free_mem = torch.cuda.mem_get_info(device)[0]
        except Exception:
            free_mem = 4 * 1024**3
    else:
        free_mem = 4 * 1024**3
    # Peak memory per candidate: src_coords(16) + gx/gy/gz(12) + grid(12)
    # + grid_sample output(4) + grid_sample internals (~8) ≈ 52 bytes/voxel
    chunk = max(1, int(free_mem * 0.5 / (voxels * 52)))
    return min(chunk, 4096)


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
) -> tuple[np.ndarray, float]:
    """Refine parameters using Adam on [0,1] normalized params (GPU).

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

    best_cost = -float("inf")
    best_norm = params_norm.detach().clone()
    no_improve = 0

    # Plateau detection: count an iteration as "improved" only if the cost rose
    # by more than a *relative* tolerance (an absolute 1e-7 is meaningless when
    # costs are ~0.02, so the old test never plateaued and just ran to the cap).
    # Stop once it has been flat for `patience` iterations. This lets us set a
    # generous iteration ceiling and still finish quickly when converged.
    rel_tol = 1e-4
    abs_tol = 1e-6
    patience = 40

    # Reading the cost back to the host forces a GPU sync, so do it only every
    # few iterations rather than every step (Adam is smooth enough that the
    # coarser best-tracking / early-stop cadence costs nothing in accuracy).
    sync_every = 5
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
        warped = apply_affine(source, matrix, base.shape)
        cost = _compute_cost(base, warped, weight, ctx, voxdims)

        loss = -cost
        loss.backward()

        # Zero gradients for frozen params
        if params_norm.grad is not None:
            params_norm.grad.data[~free_mask] = 0.0

        optimizer.step()

        if it % sync_every == 0 or it == n_iters - 1:
            cost_val = cost.item()
            threshold = best_cost + max(abs_tol, rel_tol * abs(best_cost))
            if cost_val > threshold:
                best_cost = cost_val
                best_norm = cand_norm
                no_improve = 0
            else:
                # Keep the best params even while plateauing/oscillating.
                if cost_val > best_cost:
                    best_cost = cost_val
                    best_norm = cand_norm
                no_improve += sync_every

            if tqdm is not None and verb >= 1:
                pbar.set_postfix_str(f"cost={cost_val:.6f}")
            elif verb >= 2:
                print(f"    {desc} iter {it}: cost={cost_val:.6f}")

            # Early stopping once the cost has been flat (within rel_tol) for
            # `patience` iterations — handles plateaus and oscillation.
            if no_improve >= patience:
                if verb >= 2:
                    print(f"    {desc} converged at iter {it} (cost={best_cost:.6f})")
                break

    best_phys = _denormalize_t(best_norm.clamp(0.0, 1.0), bmin, span).detach().cpu().numpy()
    return best_phys, best_cost


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
):
    """Create a closure for scipy Powell (avoids kwargs issue)."""

    def cost_fn(x_free_norm: np.ndarray) -> float:
        # Clamp to [0,1] — Powell can overshoot, producing garbage warps
        x_clamped = np.clip(x_free_norm, 0.0, 1.0)

        full_norm = fixed_norm.copy()
        full_norm[free_mask] = x_clamped
        params_phys = _denormalize(full_norm, param_bounds)

        params_t = torch.tensor(params_phys, dtype=torch.float32, device=device)
        matrix = params_to_matrix(params_t)

        with torch.no_grad():
            warped = apply_affine(source, matrix, base.shape)
            cost = _compute_cost(base, warped, weight, ctx, voxdims)

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

    cost_fn = _make_powell_cost(
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
    )

    # Powell with bounds to keep params in [0,1]
    param_bounds_01 = [(0.0, 1.0)] * nfree

    start_cost = -cost_fn(x0)  # cost at the incoming (refined) params

    result = minimize(
        cost_fn,
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
        print(f"    {desc}: cost={final_cost:.6f} ({counter[0]} evals)")

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
) -> np.ndarray:
    """Progressive multi-resolution refinement: Adam (GPU) + Powell polish.

    Resolution pyramid (smooth + downsample for speed):
      - Level 2x: smooth σ=4, downsample 2x → fast Adam, all trials
      - Level 1x: full resolution → Adam, deduplicate to best trial
      - Powell polish on best

    Returns:
        (12,) best refined parameters in physical units.
    """
    free_mask = _get_free_mask(config.dof)
    trials = [(0.0, p.copy()) for p in trial_params_list]

    # --- Level 2x: downsampled (fast, ~8x fewer voxels) ---
    # Skip for the blok / histogram costs: heavy down-blur makes neighbouring
    # voxels correlated, which inflates within-blok |correlation| (and the joint
    # histogram) regardless of alignment, so optimizing the downsampled surface
    # drives *away* from the true full-res basin (verified on EPI->anat lpc).
    # Those costs refine at full resolution, where the basin is stable.
    can_downsample = (
        min(base.shape) > 16 and ctx.name not in _BLOK_COSTS and ctx.name not in _HIST_COSTS
    )
    if can_downsample:
        ds = 2
        base_2x = _downsample_3d(_smooth_to_resolution(base, ds), ds)
        source_2x = _downsample_3d(_smooth_to_resolution(source, ds), ds)
        weight_2x = _downsample_3d(weight, ds) if weight is not None else None

        # Scale translation bounds for downsampled grid
        bounds_2x = bounds.copy()
        bounds_2x[:3] /= ds
        # The 2x grid has voxels ds times larger (in mm), so the blok lattice
        # must be built with scaled voxel sizes to keep a fixed physical blok.
        voxdims_2x = tuple(v * ds for v in voxdims)

        if verb >= 1:
            print(f"  Medium resolution ({base_2x.shape}, {ds}x downsample, {len(trials)} trials):")

        refined = []
        for i, (_, params) in enumerate(trials):
            params_ds = params.copy()
            params_ds[:3] /= ds
            params_out, cost = _refine_adam_normalized(
                base_2x,
                source_2x,
                weight_2x,
                params_ds,
                config,
                ctx,
                voxdims_2x,
                bounds_2x,
                device,
                verb=verb,
                n_iters=config.adam_iters_2x,
                lr=config.adam_lr_2x,
                desc=f"2x T{i}",
            )
            params_out[:3] *= ds  # scale translations back
            refined.append((cost, params_out))
            if verb >= 1:
                print(f"    Trial {i}: cost={cost:.6f}")

        # Sort and deduplicate
        refined.sort(key=lambda c: -c[0])
        trials = [refined[0]]
        for j in range(1, len(refined)):
            nj = _normalize(refined[j][1], bounds)
            too_close = any(
                np.max(np.abs(nj[free_mask] - _normalize(pk, bounds)[free_mask])) < 0.02
                for _, pk in trials
            )
            if not too_close:
                trials.append(refined[j])

        del base_2x, source_2x, weight_2x

    # --- Level 1x: full resolution Adam ---
    if verb >= 1:
        print(f"  Full resolution ({base.shape}, {len(trials)} trials):")

    fine_trials = []
    for i, (_, params) in enumerate(trials):
        params_out, cost = _refine_adam_normalized(
            base,
            source,
            weight,
            params,
            config,
            ctx,
            voxdims,
            bounds,
            device,
            verb=verb,
            n_iters=config.adam_iters_1x,
            lr=config.adam_lr,
            desc=f"Fine T{i}",
        )
        fine_trials.append((cost, params_out))
        if verb >= 1:
            print(f"    Trial {i} Adam: cost={cost:.6f}")

    fine_trials.sort(key=lambda c: -c[0])
    best_params = fine_trials[0][1]

    # --- Powell polish (single pass, tighter convergence) ---
    if config.powell_maxfev > 0:
        if verb >= 1:
            print("  Powell polish (full resolution):")

        best_params, _ = _refine_powell(
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
        )

    return best_params


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
    blokrad_mm = config.blokrad
    if config.cost in _BLOK_COSTS and blokrad_mm is None:
        from .cost_blok import auto_blok_radius

        blokrad_mm = auto_blok_radius(voxdims, config.bloktype)
    base_clip = source_clip = None
    if config.cost in _HIST_COSTS:
        base_clip = clip_range(base_opt)
        source_clip = clip_range(source_opt)
    ctx = CostContext(
        name=config.cost,
        sigma=config.lpa_sigma,
        kernel=config.lpa_kernel,
        bloktype=config.bloktype,
        blokrad_mm=blokrad_mm,
        base_voxdims=voxdims,
        ppow=config.ppow,
        base_clip=base_clip,
        source_clip=source_clip,
    )

    # Stage 1: the optimiser starts from identity — the cmass/manual shift is
    # already folded into ``align_matrix`` (the source was resampled through it
    # above), so only a small residual remains to be found.
    init_params = identity_params(device=device)

    # Parameter bounds centred on identity (the residual after the baked shift).
    bounds = _compute_param_bounds(base_opt.shape, range_scale=config.range_scale)

    # Stage 2: GPU-parallel coarse search (at downsampled resolution)
    if config.twopass:
        min_dim = min(base_opt.shape)
        # Pick the coarse downsample. The blok/histogram costs are distorted by
        # heavy down-blur (neighbouring voxels become correlated), so cap them
        # at 2x; the legacy ls/lps costs tolerate 4x for speed.
        max_ds = 2 if (ctx.name in _BLOK_COSTS or ctx.name in _HIST_COSTS) else 4
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
    else:
        trial_params_list = [init_params.cpu().numpy().copy()]

    # Stage 3: Progressive refinement (Adam GPU + Powell polish)
    if verb >= 1:
        print("Refinement phase:")

    best_params_phys = _refine_progressive(
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
        final_cost = _compute_cost(base, warped, weight, ctx, voxdims)
        print(f"Final cost: {final_cost.item():.6f}")

    return final_matrix, warped
