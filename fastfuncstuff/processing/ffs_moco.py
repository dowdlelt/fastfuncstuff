"""GPU-accelerated motion correction for fMRI/fNIRS timeseries.

Gauss-Newton weighted least squares (GN-WLS) rigid-body registration,
matching AFNI's 3dvolreg algorithm. Optional LPA cost via Powell optimizer.

Produces AFNI-compatible output files (.1D, .aff12.1D).
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from fastfuncstuff.memory import compute_moco_resample_batch_size, make_vram_debugger

# Suppress PyTorch internal deprecation warnings from torch.compile
warnings.filterwarnings("ignore", message=".*torch._prims_common.check.*", category=FutureWarning)

from .affine import (  # noqa: E402
    _build_homo_coords,
    apply_affine_interp,
    apply_affine_interp_batched,
    apply_affine_wsinc5,
    identity_params,
    matrix_to_params,
    params_to_matrix,
    params_to_matrix_batched,
    resample_affine_fast,
    voxel_matrix_to_dicom,
)
from .cost import _separable_smooth_3d, lpa_correlation  # noqa: E402
from .interp import _separable_resample_3d  # noqa: E402
from .shear import shear_resample  # noqa: E402

try:  # Triton kernel is CUDA-only; absence must not break CPU/MPS installs
    from .shear_triton import shear_resample_triton
except Exception:  # pragma: no cover - triton not installed
    shear_resample_triton = None
from .weight import compute_weight_image  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration & result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MocoConfig:
    """Configuration for motion correction."""

    base_index: int = 0  # Reference volume index (0 = first)
    cost: str = "wls"  # "wls", "lpa", or "quad"
    interp: str = "heptic"  # During estimation: linear|cubic|quintic|heptic|wsinc5
    final_interp: str = "wsinc5"  # For output: linear|cubic|quintic|heptic|wsinc5
    max_iter: int = 23  # Per-volume GN iterations (matches 3dvolreg -maxite)
    twopass: bool = False  # Coarse blur + fine pass
    blur_fwhm: float = 0.0  # Pre-blur for estimation (mm)
    dxy_thresh: float = 0.01  # Translation convergence, voxels (3dvolreg -x_thresh)
    dph_thresh: float = 0.02  # Rotation convergence, degrees (3dvolreg -rot_thresh)
    chain_init: bool = (
        False  # Warm-start from previous volume (opt-in; under-detects TR-to-TR motion)
    )
    automask: bool = False  # Use automask for weighting
    weight_automask: bool = False  # automask × continuous weight (best of both)
    lpa_sigma: float = 4.0  # Kernel param for LPA cost (sigma or radius)
    lpa_kernel: str = "gauss"  # "gauss" or "box"
    powell_maxfev: int = 100  # Max function evals for LPA Powell
    fixed_iter: bool = False  # Fast mode: skip convergence, run exactly max_iter
    compile: bool = True  # Use torch.compile for hot path (CUDA only)
    use_shear: bool = True  # Shear-based rigid resample for final pass (AFNI THD_rota_vol)
    skip_resample: bool = False  # Estimate only: skip Pass 2 (no aligned output produced)
    device: str | None = None
    verb: int = 1
    debug_memory: bool = False
    quad_filter_size: int = 7
    quad_center_freq: float = math.pi / 3.0
    quad_bandwidth: float = 2.0

    # -reweight: data-driven weight refinement pre-pass (see moco_reweight.py).
    reweight: bool = False
    reweight_minparams: int = 2  # of 3 displacement axes that must agree to keep a patch
    reweight_rmin: float = 0.1  # per-axis correlation threshold
    reweight_polort: int = -1  # detrend degree; <0 = auto (1 + floor(nt*TR/150))
    reweight_bloktype: str = "rhdd"  # "rhdd", "tohd", or "cube"
    reweight_blokrad: float = 0.0  # blok radius (mm); 0 = auto ~555 voxels
    reweight_maxiter: int = 6  # GN iters for the cheap per-patch estimate
    reweight_min_motion: float = 0.05  # guard: skip if global motion below this
    # When set, use these directly instead of recomputing (used by the recursive
    # global/preweight estimate so it reuses the full estimation path without
    # rebuilding the weight or the expensive derivative images).
    weight_override: Tensor | None = None
    derivs_override: Tensor | None = None


@dataclass
class MocoResult:
    """Result from motion correction."""

    aligned: Tensor  # (nt, nz, ny, nx) corrected timeseries
    params: np.ndarray  # (nt, 6) rigid params per volume [dx,dy,dz,rz,rx,ry]
    matrices_vox: Tensor  # (nt, 4, 4) voxel-space matrices
    matrices_dicom: np.ndarray  # (nt, 4, 4) DICOM-space matrices
    max_displacement: np.ndarray  # (nt,) max displacement in mm
    rms_before: np.ndarray  # (nt,) weighted RMS before alignment
    rms_after: np.ndarray  # (nt,) weighted RMS after alignment
    n_iters: np.ndarray  # (nt,) iterations per volume

    # Populated only when -reweight ran (else None).
    weight_orig: Tensor | None = None  # (nz, ny, nx) original weight
    weight_refined: Tensor | None = None  # (nz, ny, nx) reweighted weight
    patch_labels: Tensor | None = None  # (nz, ny, nx) int patch label map
    params_preweight: np.ndarray | None = None  # (nt, 6) params under original weight
    reweight_applied: bool = False  # False if the low-motion guard skipped it


# ---------------------------------------------------------------------------
# Derivative images (computed once from base)
# ---------------------------------------------------------------------------


def _rigid_params(device, dtype) -> Tensor:
    """Return (12,) identity params for rigid body (scales=1, shear=0)."""
    return identity_params(device=device, dtype=dtype)


def compute_derivative_images(
    base: Tensor,
    device: torch.device,
    verb: int = 0,
) -> Tensor:
    """Compute 6 spatial derivative images of the base via central differences.

    Uses wsinc5 interpolation for accuracy. Computed once and reused for all
    volumes and iterations.

    Args:
        base: (nz, ny, nx) reference volume on device.
        device: torch device.
        verb: verbosity level.

    Returns:
        (6, nz*ny*nx) derivative images flattened.
    """
    nz, ny, nx = base.shape
    dtype = base.dtype

    # Deltas from AFNI mri_3dalign.c
    rot_delta = 2.0 * 1.5 / (nx + ny + nz)  # radians
    rot_delta_deg = math.degrees(rot_delta)
    trans_delta = 0.07  # voxels

    deltas = torch.tensor(
        [
            trans_delta,
            trans_delta,
            trans_delta,
            rot_delta_deg,
            rot_delta_deg,
            rot_delta_deg,
        ],
        device=device,
        dtype=dtype,
    )

    # Build 12 param sets: 6 params × 2 (plus/minus)
    all_params = []
    for k in range(6):
        p_plus = _rigid_params(device, dtype)
        p_plus[k] += deltas[k]
        all_params.append(p_plus)

        p_minus = _rigid_params(device, dtype)
        p_minus[k] -= deltas[k]
        all_params.append(p_minus)

    param_batch = torch.stack(all_params)  # (12, 12)
    matrices = params_to_matrix_batched(param_batch)  # (12, 4, 4)

    # Resample all 12 at once using wsinc5 for accuracy
    if verb >= 2:
        print("  Computing 12 derivative resamples (wsinc5)...")

    derivs = torch.zeros(6, nz * ny * nx, device=device, dtype=dtype)
    for i in range(12):
        resampled = apply_affine_wsinc5(base, matrices[i], base.shape)
        k = i // 2
        if i % 2 == 0:  # plus
            derivs[k] += resampled.reshape(-1)
        else:  # minus
            derivs[k] -= resampled.reshape(-1)

    # Central difference: (I+ - I-) / (2 * delta)
    for k in range(6):
        derivs[k] /= 2.0 * deltas[k]

    return derivs


# ---------------------------------------------------------------------------
# Gauss-Newton WLS solver
# ---------------------------------------------------------------------------


def _weighted_rms(base: Tensor, source: Tensor, weight: Tensor) -> float:
    """Compute weighted RMS difference between two volumes."""
    diff = base - source
    w_sum = weight.sum().clamp(min=1e-10)
    rms = ((weight * diff * diff).sum() / w_sum).sqrt()
    return float(rms.item())


def _unweighted_rms(base: Tensor, source: Tensor) -> float:
    """Compute unweighted RMS difference (matching AFNI 3dvolreg)."""
    diff = base - source
    rms = (diff * diff).mean().sqrt()
    return float(rms.item())


def gauss_newton_rigid(
    base_flat: Tensor,
    source: Tensor,
    weight_flat: Tensor,
    WJ: Tensor,
    JtWJ: Tensor,
    init_params: Tensor,
    config: MocoConfig,
    coords: Tensor | None = None,
    p2m_fn=params_to_matrix,
    resample_fn=resample_affine_fast,
) -> tuple[Tensor, int]:
    """Per-volume Gauss-Newton WLS rigid body registration.

    Args:
        base_flat: (N,) flattened base volume.
        source: (nz, ny, nx) source volume on device.
        weight_flat: (N,) flattened weight image.
        WJ: (6, N) weighted Jacobian.
        JtWJ: (6, 6) normal equation matrix (constant).
        init_params: (12,) initial parameter guess.
        config: MocoConfig.
        coords: (4, N) pre-built homogeneous coordinate grid (optional).
            If provided, avoids rebuilding meshgrid every iteration.
        p2m_fn: Function to convert params to matrix (default: params_to_matrix).
        resample_fn: Function to resample source (default: resample_affine_fast).

    Returns:
        (params, n_iters): optimized (12,) parameters and iteration count.
    """
    device = source.device
    dtype = source.dtype
    params = init_params.clone()
    output_shape = source.shape

    # Build coords once if not provided
    if coords is None:
        coords = _build_homo_coords(output_shape, device, dtype)

    # Regularization
    eps = 1e-6 * JtWJ.diagonal().mean()
    reg = eps * torch.eye(6, device=device, dtype=JtWJ.dtype)
    JtWJ_reg = JtWJ + reg  # float64: see JtWJ construction in moco()

    for it in range(config.max_iter):
        matrix = p2m_fn(params)
        warped = resample_fn(source, matrix, coords, config.interp, output_shape)
        warped_flat = warped.reshape(-1)

        # Weighted residual
        residual = weight_flat * (base_flat - warped_flat)

        # Right-hand side: J^T W r
        JtWr = WJ @ residual  # (6,)

        # Solve 6×6 system (float64; RHS promoted, step cast back to float32)
        dp = torch.linalg.solve(JtWJ_reg, JtWr.to(JtWJ_reg.dtype))

        # Update only rigid params (first 6)
        params[:6] += dp.to(params.dtype)

        # Convergence check
        trans_converged = (dp[:3].abs() < config.dxy_thresh).all()
        rot_converged = (dp[3:].abs() < config.dph_thresh).all()
        if trans_converged and rot_converged:
            return params, it + 1

    return params, config.max_iter


def gauss_newton_rigid_masked(
    base_flat_masked: Tensor,
    source: Tensor,
    weight_flat_masked: Tensor,
    WJ_masked: Tensor,
    JtWJ: Tensor,
    init_params: Tensor,
    config: MocoConfig,
    coords_masked: Tensor,
    p2m_fn=params_to_matrix,
) -> tuple[Tensor, int]:
    """GN solver with convergence check, operating only on masked voxels."""
    device = source.device
    params = init_params.clone()

    eps = 1e-6 * JtWJ.diagonal().mean()
    reg = eps * torch.eye(6, device=device, dtype=JtWJ.dtype)
    JtWJ_reg = JtWJ + reg  # float64: see JtWJ construction in moco()

    for it in range(config.max_iter):
        matrix = p2m_fn(params)
        src_coords = matrix @ coords_masked  # (4, M)
        warped = _separable_resample_3d(
            source, src_coords[0], src_coords[1], src_coords[2], config.interp
        )  # (M,)
        residual = weight_flat_masked * (base_flat_masked - warped)

        JtWr = WJ_masked @ residual
        dp = torch.linalg.solve(JtWJ_reg, JtWr.to(JtWJ_reg.dtype))

        params[:6] += dp.to(params.dtype)

        trans_converged = (dp[:3].abs() < config.dxy_thresh).all()
        rot_converged = (dp[3:].abs() < config.dph_thresh).all()
        if trans_converged and rot_converged:
            return params, it + 1

    return params, config.max_iter


def gauss_newton_rigid_fixed(
    base_flat: Tensor,
    source: Tensor,
    weight_flat: Tensor,
    WJ: Tensor,
    JtWJ: Tensor,
    init_params: Tensor,
    coords: Tensor,
    max_iter: int,
    interp: str,
) -> Tensor:
    device = source.device
    params = init_params.clone()
    output_shape = tuple(source.shape)

    eps = 1e-6 * JtWJ.diagonal().mean()
    reg = eps * torch.eye(6, device=device, dtype=JtWJ.dtype)
    JtWJ_reg = JtWJ + reg  # float64: see JtWJ construction in moco()

    for _ in range(max_iter):
        matrix = params_to_matrix(params)
        warped = resample_affine_fast(source, matrix, coords, interp, output_shape)
        residual = weight_flat * (base_flat - warped.reshape(-1))

        JtWr = WJ @ residual
        dp = torch.linalg.solve(JtWJ_reg, JtWr.to(JtWJ_reg.dtype))

        params[:6] += dp.to(params.dtype)

    return params


def gauss_newton_rigid_fixed_masked(
    base_flat_masked: Tensor,
    source: Tensor,
    weight_flat_masked: Tensor,
    WJ_masked: Tensor,
    JtWJ: Tensor,
    init_params: Tensor,
    coords_masked: Tensor,
    max_iter: int,
    interp: str,
) -> Tensor:
    """GN solver operating only on masked (non-zero weight) voxels.

    coords_masked is (4, M) where M < N. Resamples only M voxels per iteration.
    """
    device = source.device
    params = init_params.clone()

    eps = 1e-6 * JtWJ.diagonal().mean()
    reg = eps * torch.eye(6, device=device, dtype=JtWJ.dtype)
    JtWJ_reg = JtWJ + reg  # float64: see JtWJ construction in moco()

    for _ in range(max_iter):
        matrix = params_to_matrix(params)
        src_coords = matrix @ coords_masked  # (4, M)
        warped = _separable_resample_3d(
            source, src_coords[0], src_coords[1], src_coords[2], interp
        )  # (M,)
        residual = weight_flat_masked * (base_flat_masked - warped)

        JtWr = WJ_masked @ residual  # (6,)
        dp = torch.linalg.solve(JtWJ_reg, JtWr.to(JtWJ_reg.dtype))

        params[:6] += dp.to(params.dtype)

    return params


def batched_gn_estimate(
    sources: Tensor,
    base_flat: Tensor,
    weight_flat_1d: Tensor,
    WJ: Tensor,
    JtWJ: Tensor,
    shape: tuple[int, int, int],
    max_iter: int,
    interp: str,
    dxy_thresh: float,
    dph_thresh: float,
    fixed_iter: bool,
    init_params: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Gauss-Newton WLS for a whole batch of volumes at once (Triton shears).

    Identical normal-equation math to ``gauss_newton_rigid`` — only the resample
    is the fused shear kernel, applied to all volumes per iteration. Volumes
    start from identity (or ``init_params`` for a coarse-to-fine refine pass) and
    converge independently: once a volume's step is below threshold its update is
    frozen, and the loop ends when all have converged.

    Args:
        sources: (B, nz, ny, nx) source volumes on the (CUDA) device.
        base_flat: (N,) flattened base. weight_flat_1d: (N,) weights.
        WJ: (6, N) weighted Jacobian. JtWJ: (6, 6) full normal matrix.
        init_params: optional (B,12) starting params (default: identity).
    Returns:
        (params (B,12), n_iters (B,), invalid (B,)) — ``invalid`` flags volumes
        whose shear decomposition was ever degenerate (caller re-runs those).
    """
    B = sources.shape[0]
    device, dtype = sources.device, sources.dtype
    N = base_flat.numel()

    eps = 1e-6 * JtWJ.diagonal().mean()
    JtWJ_reg = JtWJ + eps * torch.eye(6, device=device, dtype=JtWJ.dtype)  # float64

    if init_params is None:
        params = identity_params(device=device, dtype=dtype)[None].repeat(B, 1)
    else:
        params = init_params.clone()
    active = torch.ones(B, dtype=torch.bool, device=device)
    n_iters = torch.full((B,), max_iter, dtype=torch.int32, device=device)
    invalid = torch.zeros(B, dtype=torch.bool, device=device)

    for it in range(max_iter):
        mats = params_to_matrix_batched(params)  # (B,4,4)
        warped, valid = shear_resample_triton(sources, mats, shape, interp)
        invalid = invalid | ~valid
        residual = weight_flat_1d[None] * (base_flat[None] - warped.reshape(B, N))
        JtWr = residual @ WJ.t()  # (B,6)
        # Solve in float64 (RHS promoted, step cast back); the Gram matrix and
        # solve are where ill-conditioning amplifies float32 rounding.
        dp = torch.linalg.solve(JtWJ_reg, JtWr.t().to(JtWJ_reg.dtype)).t().to(dtype)  # (B,6)

        # only update active, validly-decomposed volumes
        upd = (active & valid).unsqueeze(1).to(dtype)
        params = params.clone()
        params[:, :6] = params[:, :6] + dp * upd

        if not fixed_iter:
            conv = (dp[:, :3].abs() < dxy_thresh).all(1) & (dp[:, 3:].abs() < dph_thresh).all(1)
            newly = active & conv
            n_iters = torch.where(newly, torch.full_like(n_iters, it + 1), n_iters)
            active = active & ~conv
            if not bool(active.any()):
                break

    return params, n_iters, invalid


# ---------------------------------------------------------------------------
# LPA solver (Powell-based)
# ---------------------------------------------------------------------------


def gn_lpa_rigid(
    base: Tensor,
    source: Tensor,
    weight: Tensor,
    init_params: Tensor,
    config: MocoConfig,
) -> tuple[Tensor, int]:
    """Per-volume LPA-based rigid registration using Powell optimizer.

    Args:
        base: (nz, ny, nx) base volume.
        source: (nz, ny, nx) source volume.
        weight: (nz, ny, nx) weight image.
        init_params: (12,) initial parameters.
        config: MocoConfig.

    Returns:
        (params, n_evals): optimized (12,) parameters and eval count.
    """
    from scipy.optimize import minimize

    device = base.device
    dtype = base.dtype
    output_shape = base.shape

    # We optimize only the 6 rigid params, keeping scale=1 and shear=0
    x0 = init_params[:6].detach().cpu().numpy().copy()

    eval_count = [0]

    def cost_fn(x6):
        eval_count[0] += 1
        p = identity_params(device=device, dtype=dtype)
        p[:6] = torch.tensor(x6, device=device, dtype=dtype)
        matrix = params_to_matrix(p)
        warped = apply_affine_interp(source, matrix, config.interp, output_shape)
        # lpa_correlation returns higher = better, so negate for minimization
        cost = -lpa_correlation(
            base, warped, weight, sigma=config.lpa_sigma, kernel_type=config.lpa_kernel
        )
        return float(cost.item())

    result = minimize(
        cost_fn,
        x0,
        method="Powell",
        options={
            "maxfev": config.powell_maxfev,
            "xtol": 1e-4,
            "ftol": 1e-6,
        },
    )

    params = identity_params(device=device, dtype=dtype)
    params[:6] = torch.tensor(result.x, device=device, dtype=dtype)
    return params, eval_count[0]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def compute_max_displacement(
    matrix_vox: Tensor,
    shape: tuple[int, int, int],
    voxel_sizes: np.ndarray,
) -> float:
    """Compute maximum voxel displacement from a rigid transform.

    Evaluates displacement at the 8 corners of the volume.

    Args:
        matrix_vox: (4, 4) voxel-space transformation matrix.
        shape: (nz, ny, nx) volume shape.
        voxel_sizes: (3,) voxel dimensions in mm [dx, dy, dz].

    Returns:
        Maximum displacement in mm.
    """
    nz, ny, nx = shape
    device = matrix_vox.device
    dtype = matrix_vox.dtype

    # 8 corner coordinates (x, y, z, 1)
    corners = torch.tensor(
        [
            [0, 0, 0, 1],
            [nx - 1, 0, 0, 1],
            [0, ny - 1, 0, 1],
            [nx - 1, ny - 1, 0, 1],
            [0, 0, nz - 1, 1],
            [nx - 1, 0, nz - 1, 1],
            [0, ny - 1, nz - 1, 1],
            [nx - 1, ny - 1, nz - 1, 1],
        ],
        device=device,
        dtype=dtype,
    ).T  # (4, 8)

    # Transformed corners
    transformed = matrix_vox @ corners  # (4, 8)

    # Displacement in voxel units
    disp_vox = transformed[:3] - corners[:3]  # (3, 8)

    # Convert to mm
    vs = torch.tensor(voxel_sizes, device=device, dtype=dtype).reshape(3, 1)
    disp_mm = disp_vox * vs

    # Euclidean distance per corner
    dist = (disp_mm**2).sum(dim=0).sqrt()  # (8,)
    return float(dist.max().item())


def _get_voxel_sizes(affine: np.ndarray) -> np.ndarray:
    """Extract voxel sizes from NIfTI affine matrix."""
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def save_moco_1D(
    params_array: np.ndarray,
    path: str,
) -> None:
    """Save 6-column motion parameter file (.1D).

    AFNI 3dvolreg format: roll pitch yaw dS dL dP (degrees, mm).
    Params report the subject's motion (inverse of the correction transform), so
    every column is the negation of the DICOM correction-matrix component:
    - roll  = -rz_DICOM  (rotation about I-S axis)
    - pitch = -rx_DICOM  (rotation about R-L axis)
    - yaw   = -ry_DICOM  (rotation about A-P axis)
    - dS    = -dz_DICOM  (displacement in Superior direction)
    - dL    = -dx_DICOM  (displacement in Left direction)
    - dP    = -dy_DICOM  (displacement in Posterior direction)

    Args:
        params_array: (nt, 6) array of [dx, dy, dz, rz, rx, ry] in DICOM space.
        path: output file path.
    """
    with open(path, "w") as f:
        for t in range(params_array.shape[0]):
            dx, dy, dz = params_array[t, 0], params_array[t, 1], params_array[t, 2]
            rz, rx, ry = params_array[t, 3], params_array[t, 4], params_array[t, 5]
            f.write(f"  {-rz:8.4f}  {-rx:8.4f}  {-ry:8.4f}  {-dz:8.4f}  {-dx:8.4f}  {-dy:8.4f}\n")


def save_moco_dfile(
    params_array: np.ndarray,
    rms_before: np.ndarray,
    rms_after: np.ndarray,
    path: str,
) -> None:
    """Save 9-column diagnostic file.

    Format: vol# roll pitch yaw dI dS dL rms_before rms_after

    Args:
        params_array: (nt, 6) DICOM motion parameters.
        rms_before: (nt,) RMS before alignment.
        rms_after: (nt,) RMS after alignment.
        path: output file path.
    """
    with open(path, "w") as f:
        for t in range(params_array.shape[0]):
            dx, dy, dz = params_array[t, 0], params_array[t, 1], params_array[t, 2]
            rz, rx, ry = params_array[t, 3], params_array[t, 4], params_array[t, 5]
            f.write(
                f"  {t:4d}  {-rz:8.4f}  {-rx:8.4f}  {-ry:8.4f}"
                f"  {-dz:8.4f}  {-dx:8.4f}  {-dy:8.4f}"
                f"  {rms_before[t]:11.4g}  {rms_after[t]:11.4g}\n"
            )


def save_maxdisp_1D(
    max_displacement: np.ndarray,
    path: str,
) -> None:
    """Save max displacement per volume as a 1-column .1D file."""
    with open(path, "w") as f:
        for t in range(max_displacement.shape[0]):
            f.write(f"  {max_displacement[t]:.6f}\n")


# ---------------------------------------------------------------------------
# Blur helper
# ---------------------------------------------------------------------------


def _blur_volume(vol: Tensor, fwhm: float) -> Tensor:
    """Apply Gaussian blur to a volume (fwhm in voxels)."""
    if fwhm <= 0:
        return vol
    sigma = fwhm / 2.355
    return _separable_smooth_3d(vol, sigma)


def _run_batched_estimation(
    timeseries,
    base_flat,
    weight_flat_1d,
    WJ,
    JtWJ,
    vol_shape,
    config,
    device,
    dtype,
    all_params,
    matrices_vox,
    rms_before,
    n_iters,
    base_copy_idx,
    disable_pbar,
    coarse=None,
):
    """Estimate all volumes' rigid params with the whole-batch shear GN solver.

    Fills all_params / matrices_vox / rms_before / n_iters in place. Volumes are
    processed in memory-sized chunks; the base volume (``base_copy_idx``, or -1
    when an external base is used) is left at identity; any volume with a
    degenerate decomposition is re-fit per-volume with the general resampler.

    ``coarse`` (when -twopass): a tuple (bf_coarse, wf1_coarse, WJ_c, JtWJ_c) of
    the coarse-blur normal equations. Each chunk is first solved on
    coarse-blurred sources (large capture range) and that result seeds the
    full-resolution fine pass.
    """
    nt = timeseries.shape[0]
    nz, ny, nx = vol_shape
    coarse_fwhm = max(4.0, config.blur_fwhm) if coarse is not None else 0.0

    identity = identity_params(device=device, dtype=dtype)
    indices = [t for t in range(nt) if t != base_copy_idx]
    if base_copy_idx >= 0:
        all_params[base_copy_idx] = identity.cpu().numpy()
        matrices_vox[base_copy_idx] = params_to_matrix(identity).cpu()
        rms_before[base_copy_idx] = 0.0
        n_iters[base_copy_idx] = 0

    chunk = compute_moco_resample_batch_size(nz, ny, nx, nt, device, interp=config.interp)
    chunk = max(1, chunk)

    pbar = tqdm(
        range(0, len(indices), chunk),
        desc="  Registering",
        disable=disable_pbar,
        unit="batch",
        ncols=80,
    )
    for cstart in pbar:
        idx = indices[cstart : cstart + chunk]
        srcs_raw = torch.stack([timeseries[t].to(device=device, dtype=dtype) for t in idx])
        B = srcs_raw.shape[0]

        # Fine-pass sources (optionally pre-blurred for estimation).
        if config.blur_fwhm > 0:
            srcs = torch.stack([_blur_volume(srcs_raw[i], config.blur_fwhm) for i in range(B)])
        else:
            srcs = srcs_raw

        src_flat = srcs.reshape(B, -1)
        rms_before[idx] = ((base_flat[None] - src_flat) ** 2).mean(dim=1).sqrt().cpu().numpy()

        # Two-pass: coarse-blur solve seeds the fine solve.
        init_params = None
        if coarse is not None:
            bf_c, wf1_c, WJ_c, JtWJ_c = coarse
            srcs_coarse = torch.stack([_blur_volume(srcs_raw[i], coarse_fwhm) for i in range(B)])
            init_params, _, _ = batched_gn_estimate(
                srcs_coarse,
                bf_c,
                wf1_c,
                WJ_c,
                JtWJ_c,
                vol_shape,
                config.max_iter,
                config.interp,
                config.dxy_thresh,
                config.dph_thresh,
                config.fixed_iter,
            )

        params, nit, invalid = batched_gn_estimate(
            srcs,
            base_flat,
            weight_flat_1d,
            WJ,
            JtWJ,
            vol_shape,
            config.max_iter,
            config.interp,
            config.dxy_thresh,
            config.dph_thresh,
            config.fixed_iter,
            init_params=init_params,
        )

        # Re-fit any degenerate-decomposition volumes with the trusted solver.
        if bool(invalid.any()):
            homo = _build_homo_coords(vol_shape, device, dtype)
            for j in invalid.nonzero(as_tuple=True)[0].tolist():
                p, ni = gauss_newton_rigid(
                    base_flat,
                    srcs[j],
                    weight_flat_1d,
                    WJ,
                    JtWJ,
                    identity.clone(),
                    config,
                    coords=homo,
                )
                params[j] = p
                nit[j] = ni

        mats = params_to_matrix_batched(params)  # (B,4,4)
        idx_t = torch.as_tensor(idx)
        all_params[idx] = params.detach().cpu().numpy()
        matrices_vox[idx_t] = mats.cpu()
        n_iters[idx] = nit.cpu().numpy()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def moco(
    timeseries: Tensor,
    config: MocoConfig,
    header_info: dict | None = None,
    base_vol: Tensor | None = None,
) -> MocoResult:
    """Run motion correction on a 4D timeseries.

    Args:
        timeseries: (nt, nz, ny, nx) 4D timeseries (CPU or GPU).
        config: MocoConfig.
        header_info: NIfTI header dict (for coordinate conversions).
        base_vol: optional external base volume. If None, uses
                  timeseries[config.base_index].

    Returns:
        MocoResult with aligned timeseries and motion parameters.
    """
    # Device setup
    if config.device:
        device = torch.device(config.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype = torch.float32
    nt = timeseries.shape[0]
    nz, ny, nx = timeseries.shape[1:]
    vol_shape = (nz, ny, nx)

    if config.verb >= 1:
        print(f"ffs_moco: {nt} volumes, {vol_shape}, device={device}, cost={config.cost}")

    # Get base volume
    if base_vol is not None:
        base = base_vol.to(device=device, dtype=dtype)
    else:
        base = timeseries[config.base_index].to(device=device, dtype=dtype)

    # Compute weight image (or use an injected one for the preweight estimate).
    if config.weight_override is not None:
        weight = config.weight_override.to(device=device, dtype=dtype)
    elif config.weight_automask:
        from .mask import automask as compute_automask

        mask = compute_automask(base, device=device)
        weight = compute_weight_image(base) * mask.float()
    elif config.automask:
        from .mask import automask as compute_automask

        mask = compute_automask(base, device=device)
        weight = mask.float()
    else:
        weight = compute_weight_image(base)

    weight = weight.to(device=device, dtype=dtype)

    # Pre-blur if requested
    base_est = _blur_volume(base, config.blur_fwhm) if config.blur_fwhm > 0 else base

    # Derivative images are shared by the WLS pass and the reweight pre-pass; they
    # depend only on the (blurred) base, not the weight, so compute them at most
    # once. An override (from the recursive global/preweight call) skips the
    # recompute entirely — same base_est, so the derivatives are identical.
    derivs = config.derivs_override

    # ── -reweight pre-pass: learn which patches carry consistent motion and
    #    drop the rest from the weight before building the normal equations. ──
    reweight_out: tuple | None = None  # (weight_orig, refined, labels, applied)
    params_preweight = None
    if config.reweight and config.weight_override is None:
        from dataclasses import replace

        from .moco_reweight import compute_reweight

        t0 = time.time()
        derivs = compute_derivative_images(base_est, device, verb=config.verb)
        if config.verb >= 1:
            print(f"  Derivative images: {time.time() - t0:.2f}s")
        weight0 = weight

        # Global consensus motion = the whole-image fit under the ORIGINAL weight.
        # Reuse the full (fast, batched) estimation path by re-entering moco() with
        # the weight pinned, derivatives handed over, and reweight off (estimate-
        # only — no resampling). This is both the consensus the reweight prediction
        # needs AND the "preweight" motion the user compares against the final params.
        t_global = time.time()
        pre_cfg = replace(
            config,
            reweight=False,
            weight_override=weight0,
            derivs_override=derivs,  # reuse — same base_est, avoids a 2nd wsinc5 build
            skip_resample=True,
            verb=0,
        )
        pre_res = moco(timeseries, pre_cfg, header_info=header_info, base_vol=base_vol)
        global_matrices = pre_res.matrices_vox  # (nt, 4, 4) voxel-space
        params_preweight = pre_res.params
        if config.verb >= 1:
            print(f"  Reweight global fit: {time.time() - t_global:.2f}s")

        affine = header_info["affine"] if header_info else np.eye(4)
        voxdims = tuple(float(v) for v in _get_voxel_sizes(affine))
        tr = 2.0
        try:
            hdr = header_info.get("header") if header_info else None
            if hdr is not None:
                z = hdr.get_zooms()
                if len(z) >= 4 and float(z[3]) > 0:
                    tr = float(z[3])
        except Exception:
            pass

        rw = compute_reweight(
            base_est,
            timeseries,
            weight0,
            derivs,
            global_matrices,
            voxdims=voxdims,
            tr=tr,
            bloktype=config.reweight_bloktype,
            blokrad=config.reweight_blokrad,
            minparams=config.reweight_minparams,
            rmin=config.reweight_rmin,
            polort=config.reweight_polort,
            max_iter=config.reweight_maxiter,
            min_motion=config.reweight_min_motion,
            device=device,
            verb=config.verb,
        )
        weight = rw.weight
        reweight_out = (weight0, rw.weight, rw.patch_labels, rw.applied)
        if config.verb >= 1:
            print(f"  Reweight pre-pass: {time.time() - t0:.2f}s")

    # Setup for WLS
    if config.cost == "wls":
        t0 = time.time()
        if derivs is None:
            derivs = compute_derivative_images(base_est, device, verb=config.verb)
            if config.verb >= 1:
                print(f"  Derivative images: {time.time() - t0:.2f}s")

        # Pre-compute normal equations
        weight_flat = weight.reshape(1, -1)  # (1, N)
        base_flat = base_est.reshape(-1)  # (N,)

        WJ = weight_flat * derivs  # (6, N) — broadcast weight across 6 rows
        # AFNI accumulates the registration normal equations in double
        # (mri_lsqfit.c: "internal calculations are done with doubles"). In
        # float32 the Gram matrix + 6×6 solve lose ~50% of the GN step once JtWJ
        # is ill-conditioned (low tSNR, collinear drift); float64 here drops that
        # to ~0.2%. WJ and the per-iteration RHS stay float32 (negligible loss).
        JtWJ = WJ.double() @ WJ.double().t()  # (6, 6) float64

        weight_flat_1d = weight.reshape(-1)
    elif config.cost == "quad":
        from .quadrature import (
            apply_quadrature_filters_fft,
            design_quadrature_filters,
            precompute_filter_ffts,
        )

        t0 = time.time()
        filters = design_quadrature_filters(
            size=config.quad_filter_size,
            center_freq=config.quad_center_freq,
            bandwidth=config.quad_bandwidth,
            device=device,
            dtype=dtype,
        )
        filter_spectra = precompute_filter_ffts(filters, vol_shape)
        q_base = apply_quadrature_filters_fft(base_est, filter_spectra)
        if config.verb >= 1:
            print(f"  Quadrature filters: {time.time() - t0:.2f}s")

        base_flat = None
        WJ = None
        JtWJ = None
        weight_flat_1d = None
    else:
        base_flat = None
        WJ = None
        JtWJ = None
        weight_flat_1d = None
        filter_spectra = None
        q_base = None

    # Pre-build homogeneous coordinate grid once (reused across all volumes/iterations)
    homo_coords = _build_homo_coords(vol_shape, device, dtype)
    N = homo_coords.shape[1]

    # Whole-batch GN via the fused Triton shear: every volume independent
    # (chain_init off), plain WLS, CUDA + Triton available. Falls back to the
    # per-volume loop for masked/twopass/lpa/quad/CPU/warm-start cases.
    batched_eligible = (
        config.cost == "wls"
        and config.use_shear
        and not config.chain_init
        and device.type == "cuda"
        and shear_resample_triton is not None
    )

    # Compute mask for efficient resampling (skip zero-weight voxels)
    use_masked = False
    if config.cost == "wls" and not batched_eligible:
        mask_idx = (weight.reshape(-1) > 0).nonzero(as_tuple=True)[0]
        M = mask_idx.numel()
        if M < N:
            use_masked = True
            coords_masked = homo_coords[:, mask_idx]  # (4, M)
            base_flat_masked = base_flat[mask_idx]  # (M,)
            weight_flat_masked = weight_flat_1d[mask_idx]  # (M,)
            WJ_masked = WJ[:, mask_idx]  # (6, M)
            JtWJ = WJ_masked.double() @ WJ_masked.double().t()  # (6, 6) float64
            if config.verb >= 1:
                print(f"  Active voxels: {M:,}/{N:,} ({100 * M / N:.1f}%)")

    # Create compiled versions of hot-path functions for CUDA (reduces kernel launch overhead)
    if config.compile and device.type == "cuda" and not batched_eligible:
        torch.set_float32_matmul_precision("high")
        # Enable persistent compile cache so subsequent runs with the same
        # input shape skip recompilation (~/.cache/torch/inductor/).
        import os

        os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
        if config.verb >= 1:
            print("  torch.compile: compiling (first volume will be slow, cached for next run) ...")
        if config.fixed_iter:
            # In fast mode, compile the entire GN loop (not individual functions)
            # This allows the compiler to trace through everything and fuse
            # Masked path uses default mode — "reduce-overhead" (CUDA graphs) is too
            # slow to capture with the nested Z/Y loops (8×8=64 unrolled bodies).
            # Default mode still fuses kernels, just without the graph capture overhead.
            if use_masked:
                _gn_fixed = torch.compile(gauss_newton_rigid_fixed_masked, dynamic=True)
            else:
                _gn_fixed = torch.compile(gauss_newton_rigid_fixed, dynamic=True)
            _p2m = params_to_matrix  # Uncompiled - called from within compiled _gn_fixed
            _resample = resample_affine_fast  # Uncompiled - called from within compiled _gn_fixed
        else:
            # In slow mode, compile individual functions (convergence check breaks graph)
            _p2m = torch.compile(params_to_matrix, dynamic=True)
            _resample = torch.compile(resample_affine_fast, dynamic=True)
            _gn_fixed = gauss_newton_rigid_fixed
        use_cudagraphs = False  # default mode doesn't use CUDA graphs
    else:
        _p2m = params_to_matrix
        _resample = resample_affine_fast
        if use_masked:
            _gn_fixed = gauss_newton_rigid_fixed_masked
        else:
            _gn_fixed = gauss_newton_rigid_fixed
        use_cudagraphs = False

    # Pre-compute coarse-pass normal equations for twopass (done once, not per-volume)
    if config.twopass and config.cost == "wls":
        coarse_fwhm = max(4.0, config.blur_fwhm)
        base_coarse = _blur_volume(base, coarse_fwhm)
        weight_coarse = _blur_volume(weight, coarse_fwhm / 2)
        derivs_coarse = compute_derivative_images(base_coarse, device, verb=config.verb)
        wf_coarse = weight_coarse.reshape(1, -1)
        bf_coarse = base_coarse.reshape(-1)
        WJ_c = wf_coarse * derivs_coarse
        JtWJ_c = WJ_c.double() @ WJ_c.double().t()  # (6, 6) float64
        wf1_coarse = weight_coarse.reshape(-1)
    else:
        base_coarse = weight_coarse = None
        bf_coarse = wf1_coarse = WJ_c = JtWJ_c = None

    # Allocate output arrays
    # Skip the big CPU allocation when no aligned output is requested.
    aligned = timeseries.new_empty(0) if config.skip_resample else torch.zeros_like(timeseries)
    all_params = np.zeros((nt, 12), dtype=np.float64)
    matrices_vox = torch.zeros(nt, 4, 4, dtype=dtype)
    rms_before = np.zeros(nt, dtype=np.float64)
    rms_after = np.zeros(nt, dtype=np.float64)
    n_iters = np.zeros(nt, dtype=np.int32)

    # Identity for base volume and init
    identity = identity_params(device=device, dtype=dtype)

    # Process each volume
    prev_params = identity.clone()
    t_start = time.time()

    # Create progress bar
    disable_pbar = config.verb == 0

    # ── Pass 1 (batched): estimate all volumes at once via fused shears ────
    if batched_eligible:
        _coarse = (
            (bf_coarse, wf1_coarse, WJ_c, JtWJ_c)
            if (config.twopass and config.cost == "wls")
            else None
        )
        _run_batched_estimation(
            timeseries,
            base_flat,
            weight_flat_1d,
            WJ,
            JtWJ,
            vol_shape,
            config,
            device,
            dtype,
            all_params,
            matrices_vox,
            rms_before,
            n_iters,
            config.base_index if base_vol is None else -1,
            disable_pbar,
            coarse=_coarse,
        )

    # ── Pass 1 (per-volume): estimate parameters ─────────────────────────
    pbar = tqdm(
        range(0) if batched_eligible else range(nt),
        desc="  Registering",
        disable=disable_pbar or batched_eligible,
        unit="vol",
        ncols=80,
    )
    _vol_bytes = nz * ny * nx * 4 * 10  # base + source + Jacobian intermediates (rough)
    _reg_dbg = make_vram_debugger(
        device, _vol_bytes, operation="moco_registration", chunk_size=1, enabled=config.debug_memory
    )
    _reg_dbg.__enter__()
    for t in pbar:
        # Mark step begin for CUDA graphs (prevents output tensor reuse issues)
        if use_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()

        if t == config.base_index and base_vol is None:
            # Base volume — identity transform, just copy
            mat = params_to_matrix(identity)
            matrices_vox[t] = mat.cpu()
            all_params[t] = identity.cpu().numpy()
            rms_before[t] = 0.0
            n_iters[t] = 0
            continue

        # Load source volume to GPU
        source = timeseries[t].to(device=device, dtype=dtype)
        source_est = _blur_volume(source, config.blur_fwhm) if config.blur_fwhm > 0 else source

        # Initial parameters
        if config.chain_init and t > 0:
            init_params = prev_params.clone()
        else:
            init_params = identity.clone()

        # RMS before alignment (unweighted, matching AFNI 3dvolreg)
        rms_before[t] = _unweighted_rms(base_est, source_est)

        # Twopass: coarse blur first, then fine
        if config.twopass and config.cost == "wls":
            source_coarse = _blur_volume(source, coarse_fwhm)
            if config.fixed_iter:
                init_params = _gn_fixed(
                    bf_coarse,
                    source_coarse,
                    wf1_coarse,
                    WJ_c,
                    JtWJ_c,
                    init_params,
                    homo_coords,
                    config.max_iter,
                    config.interp,
                )
            else:
                init_params, _ = gauss_newton_rigid(
                    bf_coarse,
                    source_coarse,
                    wf1_coarse,
                    WJ_c,
                    JtWJ_c,
                    init_params,
                    config,
                    coords=homo_coords,
                    p2m_fn=_p2m,
                    resample_fn=_resample,
                )

        # Main alignment
        if config.cost == "wls":
            if config.fixed_iter:
                if use_masked:
                    params = _gn_fixed(
                        base_flat_masked,
                        source_est,
                        weight_flat_masked,
                        WJ_masked,
                        JtWJ,
                        init_params,
                        coords_masked,
                        config.max_iter,
                        config.interp,
                    )
                else:
                    params = _gn_fixed(
                        base_flat,
                        source_est,
                        weight_flat_1d,
                        WJ,
                        JtWJ,
                        init_params,
                        homo_coords,
                        config.max_iter,
                        config.interp,
                    )
                n_iter = config.max_iter
            else:
                if use_masked:
                    params, n_iter = gauss_newton_rigid_masked(
                        base_flat_masked,
                        source_est,
                        weight_flat_masked,
                        WJ_masked,
                        JtWJ,
                        init_params,
                        config,
                        coords_masked=coords_masked,
                        p2m_fn=_p2m,
                    )
                else:
                    params, n_iter = gauss_newton_rigid(
                        base_flat,
                        source_est,
                        weight_flat_1d,
                        WJ,
                        JtWJ,
                        init_params,
                        config,
                        coords=homo_coords,
                        p2m_fn=_p2m,
                        resample_fn=_resample,
                    )
        elif config.cost == "lpa":
            params, n_iter = gn_lpa_rigid(
                base_est,
                source_est,
                weight,
                init_params,
                config,
            )
        elif config.cost == "quad":
            from .quadrature import quadrature_gn_rigid, quadrature_gn_rigid_fixed

            weight_flat = weight.reshape(1, -1) if weight is not None else None
            if config.fixed_iter:
                params = quadrature_gn_rigid_fixed(
                    source_est,
                    q_base,
                    filter_spectra,
                    init_params,
                    homo_coords,
                    vol_shape,
                    config.max_iter,
                    config.interp,
                    weight_flat,
                )
                n_iter = config.max_iter
            else:
                params, n_iter = quadrature_gn_rigid(
                    base_est,
                    source_est,
                    q_base,
                    filter_spectra,
                    init_params,
                    homo_coords,
                    vol_shape,
                    max_iter=config.max_iter,
                    interp=config.interp,
                    weight=weight,
                    dxy_thresh=config.dxy_thresh,
                    dph_thresh=config.dph_thresh,
                )

        # Fallback: if result is worse than identity, retry from identity (skip in fixed_iter mode)
        if not config.fixed_iter:
            mat_result = _p2m(params)
            warped_check = _resample(source_est, mat_result, homo_coords, config.interp, vol_shape)
            rms_result = _weighted_rms(base_est, warped_check, weight)

            rms_identity = _weighted_rms(base_est, source_est, weight)
            if rms_result > rms_identity * 1.05 and not torch.equal(init_params[:6], identity[:6]):
                if config.verb >= 2:
                    print(
                        f"  Vol {t}: fallback to identity init "
                        f"(rms {rms_result:.4f} > {rms_identity:.4f})"
                    )
                if config.cost == "wls":
                    if use_masked:
                        params, n_iter = gauss_newton_rigid_masked(
                            base_flat_masked,
                            source_est,
                            weight_flat_masked,
                            WJ_masked,
                            JtWJ,
                            identity.clone(),
                            config,
                            coords_masked=coords_masked,
                            p2m_fn=_p2m,
                        )
                    else:
                        params, n_iter = gauss_newton_rigid(
                            base_flat,
                            source_est,
                            weight_flat_1d,
                            WJ,
                            JtWJ,
                            identity.clone(),
                            config,
                            coords=homo_coords,
                            p2m_fn=_p2m,
                            resample_fn=_resample,
                        )
                elif config.cost == "lpa":
                    params, n_iter = gn_lpa_rigid(
                        base_est,
                        source_est,
                        weight,
                        identity.clone(),
                        config,
                    )
                elif config.cost == "quad":
                    from .quadrature import quadrature_gn_rigid

                    params, n_iter = quadrature_gn_rigid(
                        base_est,
                        source_est,
                        q_base,
                        filter_spectra,
                        identity.clone(),
                        homo_coords,
                        vol_shape,
                        max_iter=config.max_iter,
                        interp=config.interp,
                        weight=weight,
                        dxy_thresh=config.dxy_thresh,
                        dph_thresh=config.dph_thresh,
                    )

        # Store parameters (no resampling yet)
        prev_params = params.clone()
        mat = _p2m(params)
        matrices_vox[t] = mat.cpu()
        all_params[t] = params.detach().cpu().numpy()
        n_iters[t] = n_iter

        if not disable_pbar:
            pbar.set_postfix(it=n_iter, rms0=f"{rms_before[t]:.1f}", refresh=False)

        if config.verb >= 2:
            pbar.write(f"  Vol {t:4d}: {n_iter:2d} iter, rms_before={rms_before[t]:.4f}")

    if not disable_pbar:
        pbar.close()

    _reg_dbg.__exit__(None, None, None)

    if config.verb >= 1:
        elapsed = time.time() - t_start
        per_vol = elapsed / max(nt - 1, 1)
        print(f"  Estimation: {elapsed:.2f}s ({per_vol:.3f}s/vol)")

    # ── Pass 2: batch resample with final interpolation ────────────────
    t_resample = time.time()

    if config.skip_resample:
        if config.verb >= 1:
            print("  Resampling: skipped (estimate-only; no aligned output requested)")
    else:
        batch_size = compute_moco_resample_batch_size(
            nz, ny, nx, nt, device, interp=config.final_interp
        )
        if config.verb >= 1:
            print(f"  Resampling batch size: {batch_size} volumes")

        # Indices that need actual resampling (base volume is just copied)
        base_copy_idx = config.base_index if base_vol is None else -1

        _resample_dbg = make_vram_debugger(
            device,
            nz * ny * nx * 4 * (3 + batch_size),
            operation="moco_resample",
            chunk_size=batch_size,
            enabled=config.debug_memory,
        )
        _resample_dbg.__enter__()

        resample_pbar = tqdm(
            range(0, nt, batch_size),
            desc="  Resampling",
            disable=disable_pbar,
            unit="batch",
            ncols=80,
        )

        for batch_start in resample_pbar:
            batch_end = min(batch_start + batch_size, nt)

            # Separate base-copy volumes from volumes that need resampling
            resample_indices = [t for t in range(batch_start, batch_end) if t != base_copy_idx]
            for t in range(batch_start, batch_end):
                if t == base_copy_idx:
                    aligned[t] = timeseries[t]
                    rms_after[t] = 0.0

            if not resample_indices:
                continue

            # Load batch to GPU
            sources_batch = torch.stack(
                [timeseries[t].to(device=device, dtype=dtype) for t in resample_indices]
            )
            matrices_batch = matrices_vox[resample_indices].to(device=device, dtype=dtype)

            use_triton = (
                config.use_shear and device.type == "cuda" and shear_resample_triton is not None
            )
            if use_triton:
                # Fused Triton shears (AFNI THD_rota_vol), batched over the whole
                # chunk — ~250x the general resampler. Volumes with a degenerate
                # decomposition (valid=False, essentially never) get the general
                # resampler instead.
                aligned_batch, valid = shear_resample_triton(
                    sources_batch, matrices_batch, vol_shape, config.final_interp
                )
                if not bool(valid.all()):
                    bad = (~valid).nonzero(as_tuple=True)[0].tolist()
                    for j in bad:
                        aligned_batch[j] = resample_affine_fast(
                            sources_batch[j],
                            matrices_batch[j],
                            homo_coords,
                            config.final_interp,
                            vol_shape,
                            zero_outside=True,
                        )
            elif config.use_shear:
                # CPU/MPS shear path (no Triton): per-volume gather shears.
                outs = []
                for j in range(len(resample_indices)):
                    w = shear_resample(
                        sources_batch[j], matrices_batch[j], vol_shape, config.final_interp
                    )
                    if w is None:
                        w = resample_affine_fast(
                            sources_batch[j],
                            matrices_batch[j],
                            homo_coords,
                            config.final_interp,
                            vol_shape,
                            zero_outside=True,
                        )
                    outs.append(w)
                aligned_batch = torch.stack(outs)
            else:
                aligned_batch = apply_affine_interp_batched(
                    sources_batch, matrices_batch, config.final_interp, vol_shape, zero_outside=True
                )

            for i, t in enumerate(resample_indices):
                aligned[t] = aligned_batch[i].cpu()
                rms_after[t] = _unweighted_rms(base_est, aligned_batch[i])

        _resample_dbg.__exit__(None, None, None)

        if not disable_pbar:
            resample_pbar.close()

        if config.verb >= 1:
            elapsed_resample = time.time() - t_resample
            per_vol_resample = elapsed_resample / max(nt - 1, 1)
            print(f"  Resampling: {elapsed_resample:.2f}s ({per_vol_resample:.3f}s/vol)")

    # Convert to DICOM-space matrices and extract params
    affine = header_info["affine"] if header_info else np.eye(4)
    voxel_sizes = _get_voxel_sizes(affine)

    matrices_dicom_np = np.zeros((nt, 4, 4), dtype=np.float64)
    params_dicom = np.zeros((nt, 6), dtype=np.float64)
    max_disp = np.zeros(nt, dtype=np.float64)

    for t in range(nt):
        M_vox = matrices_vox[t]

        # Convert voxel-space matrix to DICOM-space
        M_dicom = voxel_matrix_to_dicom(M_vox, affine, affine)
        matrices_dicom_np[t] = M_dicom.numpy()

        # Extract parameters from the forward transformation
        p12 = matrix_to_params(M_dicom)
        params_dicom[t] = p12[:6].numpy()

        # Max displacement
        max_disp[t] = compute_max_displacement(M_vox, vol_shape, voxel_sizes)

    if reweight_out is not None:
        weight_orig, weight_refined, patch_labels, applied = reweight_out
    else:
        weight_orig = weight_refined = patch_labels = None
        applied = False

    return MocoResult(
        aligned=aligned,
        params=params_dicom,
        matrices_vox=matrices_vox,
        matrices_dicom=matrices_dicom_np,
        max_displacement=max_disp,
        rms_before=rms_before,
        rms_after=rms_after,
        n_iters=n_iters,
        weight_orig=weight_orig,
        weight_refined=weight_refined,
        patch_labels=patch_labels,
        params_preweight=params_preweight,
        reweight_applied=applied,
    )
