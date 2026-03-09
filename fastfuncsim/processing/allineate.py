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
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from torch import Tensor

from .affine import (
    apply_affine,
    apply_affine_batched,
    apply_affine_wsinc5,
    identity_params,
    params_to_matrix,
    params_to_matrix_batched,
    resample_to_base_grid,
)
from .cost import (
    _separable_smooth_3d,
    clipped_pearson_correlation,
    lpa_correlation,
    lpc_correlation,
)
from .mask import automask
from .weight import compute_weight_image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _tqdm_bar(iterable, total=None, desc=None, disable=False):
    """Wrap iterable in tqdm if available, otherwise passthrough."""
    if tqdm is not None and not disable:
        return tqdm(iterable, total=total, desc=desc, file=sys.stderr,
                    leave=False, ncols=80)
    return iterable


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AffineAlignConfig:
    """Configuration for affine/rigid alignment."""

    # Degrees of freedom
    dof: str = "affine"  # "rigid" (6), "affine" (12), "epi" (9)

    # Cost function
    cost: str = "lpa"  # "ls" (clipped pearson), "lpa", "lpc"
    lpa_sigma: float = 4.0

    # Coarse search
    twopass: bool = True
    coarse_range: float = 30.0   # degrees
    coarse_step: float = 5.0     # degrees
    tbest: int = 3       # best coarse candidates to carry into refinement

    # Refinement tuning
    adam_iters_2x: int = 150     # Adam iters at 2x downsampled
    adam_iters_1x: int = 200     # Adam iters at full resolution
    powell_maxfev: int = 500     # Powell max function evaluations (0=skip)

    # Center-of-mass
    cmass: bool = True

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

def _compute_cost(
    base: Tensor,
    warped: Tensor,
    weight: Tensor | None,
    cost_name: str,
    sigma: float = 4.0,
) -> Tensor:
    """Compute alignment cost (higher = better)."""
    if cost_name in ("ls", "pearclp"):
        return clipped_pearson_correlation(
            base, warped.reshape(-1),
            weight.reshape(-1) if weight is not None else None,
        )
    elif cost_name == "lpa":
        return lpa_correlation(base, warped, weight, sigma=sigma)
    elif cost_name == "lpc":
        return lpc_correlation(base, warped, weight, sigma=sigma)
    else:
        raise ValueError(f"Unknown cost function: {cost_name}")


def _batched_cost(
    base: Tensor,
    warped_batch: Tensor,
    weight: Tensor | None,
    cost_name: str,
    sigma: float = 4.0,
) -> Tensor:
    """Compute cost for B warped images against a single base.

    Returns:
        (B,) cost values.
    """
    B = warped_batch.shape[0]

    if cost_name in ("ls", "pearclp"):
        base_flat = base.reshape(-1)
        w_flat = weight.reshape(-1) if weight is not None else None
        from .cost import _auto_clip
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

    elif cost_name in ("lpa", "lpc"):
        costs = torch.zeros(B, device=base.device)
        cost_fn = lpa_correlation if cost_name == "lpa" else lpc_correlation
        for i in range(B):
            costs[i] = cost_fn(base, warped_batch[i], weight, sigma=sigma)
        return costs

    else:
        raise ValueError(f"Unknown cost function: {cost_name}")


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

def _compute_nonzero_bbox(vol: Tensor, pad: int = 8) -> tuple[
    tuple[int, int, int], tuple[int, int, int]
]:
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
    coords = torch.stack([ii.reshape(-1), jj.reshape(-1), kk.reshape(-1),
                          torch.ones(N, device=device, dtype=dtype)], dim=0)
    src_coords = grid_matrix @ coords  # (4, N)

    src_x = src_coords[0]
    src_y = src_coords[1]
    src_z = src_coords[2]

    valid = ((src_x >= -0.5) & (src_x <= snx - 0.5) &
             (src_y >= -0.5) & (src_y <= sny - 0.5) &
             (src_z >= -0.5) & (src_z <= snz - 0.5))

    return valid.reshape(onz, ony, onx)


# ---------------------------------------------------------------------------
# Parameter bounds and normalization (AFNI-style)
# ---------------------------------------------------------------------------

def _compute_param_bounds(
    base_shape: tuple[int, int, int],
    cmass_shift: np.ndarray | None = None,
) -> np.ndarray:
    """Compute AFNI-style parameter bounds.

    All units match our internal convention (voxels for translations,
    degrees for rotations, ratios for scales/shears).

    Returns:
        (12, 2) array of [min, max] per parameter.
    """
    nz, ny, nx = base_shape
    if cmass_shift is None:
        cmass_shift = np.zeros(3)

    bounds = np.zeros((12, 2))

    # Translation range: ~1/3 of FOV, centered at cmass
    bounds[0] = [cmass_shift[0] - 0.321 * (nx - 1),
                 cmass_shift[0] + 0.321 * (nx - 1)]
    bounds[1] = [cmass_shift[1] - 0.321 * (ny - 1),
                 cmass_shift[1] + 0.321 * (ny - 1)]
    bounds[2] = [cmass_shift[2] - 0.321 * (nz - 1),
                 cmass_shift[2] + 0.321 * (nz - 1)]

    bounds[3:6] = [-30.0, 30.0]            # rotations (degrees)
    bounds[6:9] = [0.711, 1.0 / 0.711]     # scales
    bounds[9:12] = [-0.1111, 0.1111]        # shears

    return bounds


def _get_free_mask(dof: str) -> np.ndarray:
    """Return boolean mask of which parameters are free."""
    free = np.ones(12, dtype=bool)
    if dof == "rigid":
        free[6:12] = False
    elif dof == "epi":
        free[6] = False    # sx
        free[8] = False    # sz
        free[11] = False   # shzy
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
        return torch.tensor([nx / 2.0, ny / 2.0, nz / 2.0],
                            device=device, dtype=dtype)

    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=dtype, device=device),
        torch.arange(ny, dtype=dtype, device=device),
        torch.arange(nx, dtype=dtype, device=device),
        indexing="ij",
    )
    return torch.stack([(w * ii).sum() / wsum,
                        (w * jj).sum() / wsum,
                        (w * kk).sum() / wsum])


def _cmass_translation(base: Tensor, source: Tensor,
                       weight: Tensor | None = None) -> Tensor:
    """Compute translation to align source center-of-mass to base."""
    return _center_of_mass(source) - _center_of_mass(base, weight)


# ---------------------------------------------------------------------------
# Stage 2: GPU-parallel coarse search
# ---------------------------------------------------------------------------

def _generate_rotation_candidates(
    coarse_range: float,
    coarse_step: float,
    translation: Tensor,
    device: torch.device,
    shift_range: float = 0.0,
) -> Tensor:
    """Generate grid of candidate parameter vectors for coarse search.

    Generates a grid of rotation candidates. If shift_range > 0, also
    tests per-axis translation offsets (±shift on each axis independently,
    7 shifts total instead of 27 for a full 3D grid).

    Returns:
        (B, 12) candidate parameter sets.
    """
    angles = torch.arange(-coarse_range, coarse_range + coarse_step * 0.5,
                          coarse_step, device=device)
    n = angles.shape[0]

    rz_grid, rx_grid, ry_grid = torch.meshgrid(
        angles, angles, angles, indexing="ij")
    B_rot = n * n * n

    # Build shift offsets: identity + ±shift per axis (7 total)
    if shift_range > 0:
        s = float(shift_range)
        shift_offsets = torch.tensor([
            [0, 0, 0],
            [s, 0, 0], [-s, 0, 0],
            [0, s, 0], [0, -s, 0],
            [0, 0, s], [0, 0, -s],
        ], device=device)
    else:
        shift_offsets = torch.zeros(1, 3, device=device)
    n_shifts = shift_offsets.shape[0]

    B = B_rot * n_shifts
    params = torch.zeros(B, 12, device=device)

    for si in range(n_shifts):
        start = si * B_rot
        end = start + B_rot
        params[start:end, 0] = translation[0] + shift_offsets[si, 0]
        params[start:end, 1] = translation[1] + shift_offsets[si, 1]
        params[start:end, 2] = translation[2] + shift_offsets[si, 2]
        params[start:end, 3] = rz_grid.reshape(-1)
        params[start:end, 4] = rx_grid.reshape(-1)
        params[start:end, 5] = ry_grid.reshape(-1)
    params[:, 6:9] = 1.0

    return params


def _coarse_search(
    base: Tensor,
    source: Tensor,
    weight: Tensor | None,
    config: AffineAlignConfig,
    init_translation: Tensor,
    device: torch.device,
    verb: int = 1,
    shift_range: float = 0.0,
) -> list[Tensor]:
    """GPU-parallel coarse rotation + translation search.

    Uses LPA for coarse search when fine cost is LPC (both handle
    cross-modality via absolute correlation). Uses the user's cost
    otherwise.

    Returns top tbest parameter vectors sorted by cost (best first).
    """
    candidates = _generate_rotation_candidates(
        config.coarse_range, config.coarse_step, init_translation, device,
        shift_range=shift_range)
    B = candidates.shape[0]

    if verb >= 1:
        extra = ""
        if shift_range > 0:
            extra = f", shift=±{shift_range:.0f}vox"
        print(f"  Coarse search: {B} candidates "
              f"(range={config.coarse_range}°, step={config.coarse_step}°"
              f"{extra})")

    matrices = params_to_matrix_batched(candidates)
    chunk_size = _estimate_chunk_size(base.shape, device)
    all_costs = []

    # Use LPA for coarse when cost is lpc (both handle cross-modality
    # via absolute correlation). Clipped Pearson fails for inverted
    # contrast because the global correlation is negative.
    coarse_cost = "lpa" if config.cost == "lpc" else config.cost

    chunks = range(0, B, chunk_size)
    for start in _tqdm_bar(chunks, total=len(range(0, B, chunk_size)),
                           desc="Coarse", disable=verb < 1):
        end = min(start + chunk_size, B)
        with torch.no_grad():
            warped_batch = apply_affine_batched(
                source, matrices[start:end], base.shape)
            costs = _batched_cost(base, warped_batch, weight,
                                  coarse_cost, config.lpa_sigma)
        all_costs.append(costs)
        del warped_batch

    all_costs = torch.cat(all_costs)
    tbest = min(config.tbest, B)
    top_costs, top_indices = all_costs.topk(tbest)

    result = []
    for i in range(tbest):
        result.append(candidates[top_indices[i]])
        if verb >= 1 and i == 0:
            p = candidates[top_indices[i]]
            shift_str = ""
            if shift_range > 0:
                shift_str = (f", shift=({p[0].item():.1f}, "
                             f"{p[1].item():.1f}, {p[2].item():.1f})")
            print(f"  Best coarse: cost={top_costs[i].item():.6f}, "
                  f"rot=({p[3].item():.1f}°, {p[4].item():.1f}°, "
                  f"{p[5].item():.1f}°){shift_str}")

    if verb >= 1 and tbest > 1:
        print(f"  Keeping top {tbest} candidates for refinement")

    return result


def _estimate_chunk_size(vol_shape: tuple[int, ...],
                         device: torch.device) -> int:
    """Estimate how many candidates we can process at once."""
    voxels = vol_shape[0] * vol_shape[1] * vol_shape[2]
    if device.type == "cuda":
        try:
            free_mem = torch.cuda.mem_get_info(device)[0]
        except Exception:
            free_mem = 4 * 1024**3
    else:
        free_mem = 4 * 1024**3
    chunk = max(1, int(free_mem * 0.5 / (voxels * 20)))
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
        bmin, span,
    )

    # Initialize in normalized space
    init_t = torch.tensor(init_params_phys, dtype=torch.float32, device=device)
    params_norm = _normalize_t(init_t, bmin, span).clone().detach()
    params_norm.requires_grad_(True)

    optimizer = torch.optim.Adam([params_norm], lr=lr)

    best_cost = -float("inf")
    best_norm = params_norm.detach().clone()
    no_improve = 0

    pbar = _tqdm_bar(range(n_iters), total=n_iters, desc=desc,
                     disable=verb < 1)
    for it in pbar:
        optimizer.zero_grad()

        # Enforce frozen params and clamp to [0, 1]
        with torch.no_grad():
            params_norm.data[~free_mask] = identity_norm[~free_mask]
            params_norm.data.clamp_(0.0, 1.0)

        # Denormalize (differentiable) and build matrix
        params_phys = _denormalize_t(params_norm, bmin, span)
        matrix = params_to_matrix(params_phys)
        warped = apply_affine(source, matrix, base.shape)
        cost = _compute_cost(base, warped, weight, config.cost, config.lpa_sigma)

        loss = -cost
        loss.backward()

        # Zero gradients for frozen params
        if params_norm.grad is not None:
            params_norm.grad.data[~free_mask] = 0.0

        optimizer.step()

        cost_val = cost.item()
        if cost_val > best_cost + 1e-7:
            best_cost = cost_val
            best_norm = params_norm.detach().clone()
            no_improve = 0
        else:
            no_improve += 1

        if tqdm is not None and verb >= 1:
            pbar.set_postfix_str(f"cost={cost_val:.6f}")
        elif verb >= 2 and (it % 50 == 0 or it == n_iters - 1):
            print(f"    {desc} iter {it}: cost={cost_val:.6f}")

        # Early stopping if no improvement for 40 iters
        if no_improve >= 40:
            if verb >= 2:
                print(f"    {desc} early stop at iter {it}")
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
    cost_name: str,
    lpa_sigma: float,
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
            cost = _compute_cost(base, warped, weight, cost_name, lpa_sigma)

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
        pbar = tqdm(total=maxfev, desc=desc, file=sys.stderr,
                    leave=False, ncols=80)

    cost_fn = _make_powell_cost(
        base, source, weight,
        config.cost, config.lpa_sigma,
        bounds, free_mask, fixed_norm, device,
        counter=counter, pbar=pbar,
    )

    # Powell with bounds to keep params in [0,1]
    param_bounds_01 = [(0.0, 1.0)] * nfree

    result = minimize(
        cost_fn, x0, method="Powell",
        bounds=param_bounds_01,
        options={
            "maxfev": maxfev,
            "ftol": ftol,
            "direc": np.eye(nfree) * initial_step,
        },
    )

    if pbar is not None:
        pbar.close()

    full_norm = fixed_norm.copy()
    full_norm[free_mask] = np.clip(result.x, 0.0, 1.0)
    params_phys = _denormalize(full_norm, bounds)
    final_cost = -result.fun

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
    can_downsample = min(base.shape) > 16
    if can_downsample:
        ds = 2
        base_2x = _downsample_3d(_smooth_to_resolution(base, ds), ds)
        source_2x = _downsample_3d(_smooth_to_resolution(source, ds), ds)
        weight_2x = _downsample_3d(weight, ds) if weight is not None else None

        # Scale translation bounds for downsampled grid
        bounds_2x = bounds.copy()
        bounds_2x[:3] /= ds

        if verb >= 1:
            print(f"  Medium resolution ({base_2x.shape}, {ds}x downsample, "
                  f"{len(trials)} trials):")

        refined = []
        for i, (_, params) in enumerate(trials):
            params_ds = params.copy()
            params_ds[:3] /= ds
            params_out, cost = _refine_adam_normalized(
                base_2x, source_2x, weight_2x, params_ds, config,
                bounds_2x, device,
                verb=verb, n_iters=config.adam_iters_2x, lr=0.01,
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
            base, source, weight, params, config, bounds, device,
            verb=verb, n_iters=config.adam_iters_1x, lr=0.005,
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
            print(f"  Powell polish (full resolution):")

        best_params, cost = _refine_powell(
            base, source, weight, best_params, config, bounds, device,
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
    base_ijk2xyz = torch.from_numpy(
        base_affine.astype(np.float64)).float().to(device)
    source_xyz2ijk = torch.linalg.inv(
        torch.from_numpy(
            source_affine.astype(np.float64)).float().to(device))
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

    # Pre-resample source to base grid if needed
    grid_matrix = None
    source_validity = None
    if (base_header is not None and source_header is not None
            and source.shape != base.shape):
        if verb >= 1:
            print(f"Resampling source {source.shape} to base grid {base.shape}")
        grid_matrix = _compute_grid_matrix(
            source_header["affine"], base_header["affine"], device)

        # Compute validity mask: which base voxels are inside the source FOV
        source_validity = _compute_source_validity_mask(
            source_native.shape, base.shape, grid_matrix)

        source_on_base = resample_to_base_grid(
            source_native, base.shape,
            source_header["affine"], base_header["affine"])

        # Zero out base-grid voxels outside the source FOV
        # (resample_to_base_grid uses border padding, which gives fake
        # edge-replicated values for out-of-bounds voxels)
        source_on_base = source_on_base * source_validity.float()

        if verb >= 1:
            n_valid = int(source_validity.sum().item())
            n_total = source_validity.numel()
            print(f"  Source covers {n_valid}/{n_total} base voxels "
                  f"({100.0 * n_valid / n_total:.1f}%)")
    else:
        source_on_base = source_native

    # Weight image
    weight = compute_weight_image(base) if config.autoweight else None

    # Apply source validity mask to weight (exclude base voxels outside source)
    if source_validity is not None and weight is not None:
        weight = weight * source_validity.float()

    # Source automask
    if config.source_automask:
        if verb >= 1:
            print("Computing source automask...")
        src_mask = automask(source_on_base, device=device)
        if weight is not None:
            weight = weight * src_mask.float()
        else:
            weight = src_mask.float()
        if verb >= 1:
            n = int(src_mask.sum().item())
            print(f"  Source automask: {n}/{src_mask.numel()} voxels "
                  f"({100.0 * n / src_mask.numel():.1f}%)")

        if save_automask_path is not None:
            from .io import save_image
            save_image(src_mask.float(), save_automask_path,
                       header_info=base_header)
            if verb >= 1:
                print(f"  Saved automask: {save_automask_path}")

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
    base_full = base
    source_on_base_full = source_on_base
    weight_full = weight

    if config.autocrop:
        base_crop, source_crop, weight_crop, offset = _crop_volumes(
            base, source_on_base, weight)
        if base_crop.shape != base.shape:
            crop_offset = offset
            if verb >= 1:
                savings = 100.0 * (1.0 - (base_crop.numel() / base.numel()))
                print(f"  Auto-crop: {base.shape} → {base_crop.shape} "
                      f"({savings:.0f}% fewer voxels)")
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

    # Stage 1: Center-of-mass initialization
    init_params = identity_params(device=device)
    cmass_shift = np.zeros(3)
    if config.cmass:
        translation = _cmass_translation(base_opt, source_opt, weight_opt)
        cmass_shift = translation.cpu().numpy()
        init_params[:3] = translation
        if verb >= 1:
            t = cmass_shift
            print(f"  Center-of-mass shift: "
                  f"dx={t[0]:.2f}, dy={t[1]:.2f}, dz={t[2]:.2f} voxels")

    # Parameter bounds (based on optimization grid size)
    bounds = _compute_param_bounds(base_opt.shape, cmass_shift)

    # Stage 2: GPU-parallel coarse search (at downsampled resolution)
    if config.twopass:
        min_dim = min(base_opt.shape)
        # Pick largest downsample that keeps min dim ≥ 32
        # (local costs need enough voxels for meaningful neighborhoods)
        if min_dim >= 128:
            ds_factor = 4
        elif min_dim >= 64:
            ds_factor = 2
        else:
            ds_factor = 1

        if ds_factor > 1:
            base_ds = _downsample_3d(
                _smooth_to_resolution(base_opt, ds_factor), ds_factor)
            source_ds = _downsample_3d(
                _smooth_to_resolution(source_opt, ds_factor), ds_factor)
            weight_ds = (_downsample_3d(weight_opt, ds_factor)
                         if weight_opt is not None else None)

            coarse_init = init_params.clone()
            coarse_init[:3] /= ds_factor

            # Coarse translation search: ±3 voxels at downsampled res
            # (= ±6 at 2x, ±12 at 4x full-res voxels)
            coarse_shift_range = 3.0

            if verb >= 1:
                print(f"  Coarse resolution "
                      f"({base_ds.shape}, {ds_factor}x downsample):")

            best_list = _coarse_search(
                base_ds, source_ds, weight_ds, config,
                coarse_init[:3], device, verb,
                shift_range=coarse_shift_range)

            trial_params_list = []
            for p in best_list:
                p_np = p.cpu().numpy().copy()
                p_np[:3] *= ds_factor
                trial_params_list.append(p_np)

            del base_ds, source_ds, weight_ds
        else:
            best_list = _coarse_search(
                base_opt, source_opt, weight_opt, config,
                init_params[:3], device, verb)
            trial_params_list = [p.cpu().numpy().copy() for p in best_list]
    else:
        trial_params_list = [init_params.cpu().numpy().copy()]

    # Stage 3: Progressive refinement (Adam GPU + Powell polish)
    if verb >= 1:
        print("Refinement phase:")

    best_params_phys = _refine_progressive(
        base_opt, source_opt, weight_opt, trial_params_list,
        config, bounds, device, verb)

    # Build final matrix — adjust for crop offset if needed
    best_t = torch.tensor(best_params_phys, dtype=torch.float32, device=device)
    residual_matrix = params_to_matrix(best_t)

    if crop_offset is not None:
        # The residual_matrix maps cropped-base voxels → cropped-source voxels.
        # Since source was cropped identically, we need:
        #   full_base_voxel = crop_base_voxel + offset
        # So: M_full = T(+offset) @ M_crop @ T(-offset)
        # where T(offset) shifts by (x_off, y_off, z_off)
        x_off, y_off, z_off = crop_offset
        T_pos = torch.eye(4, device=device, dtype=torch.float32)
        T_neg = torch.eye(4, device=device, dtype=torch.float32)
        T_pos[0, 3] = float(x_off)
        T_pos[1, 3] = float(y_off)
        T_pos[2, 3] = float(z_off)
        T_neg[0, 3] = float(-x_off)
        T_neg[1, 3] = float(-y_off)
        T_neg[2, 3] = float(-z_off)
        residual_matrix = T_pos @ residual_matrix @ T_neg
    final_matrix = (grid_matrix @ residual_matrix
                    if grid_matrix is not None
                    else residual_matrix)

    # Final output: single-step resampling from native source
    # Bring source_native back to GPU if it was offloaded
    source_native = source_native.to(device)
    if config.final_interp == "wsinc5":
        if verb >= 1:
            print("Applying final wsinc5 interpolation...")
        warped = apply_affine_wsinc5(source_native, final_matrix, base.shape)
    else:
        warped = apply_affine(source_native, final_matrix, base.shape,
                              zero_outside=True)

    if verb >= 1:
        final_cost = _compute_cost(base, warped, weight,
                                   config.cost, config.lpa_sigma)
        print(f"Final cost: {final_cost.item():.6f}")

    return final_matrix, warped
