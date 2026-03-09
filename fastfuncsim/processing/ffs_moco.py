"""GPU-accelerated motion correction for fMRI/fNIRS timeseries.

Gauss-Newton weighted least squares (GN-WLS) rigid-body registration,
matching AFNI's 3dvolreg algorithm. Optional LPA cost via Powell optimizer.

Produces AFNI-compatible output files (.1D, .aff12.1D).
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

# Suppress PyTorch internal deprecation warnings from torch.compile
warnings.filterwarnings(
    "ignore", message=".*torch._prims_common.check.*", category=FutureWarning
)

from .affine import (
    _build_homo_coords,
    apply_affine,
    apply_affine_interp,
    apply_affine_wsinc5,
    identity_params,
    matrix_to_params,
    params_to_matrix,
    params_to_matrix_batched,
    resample_affine_fast,
    voxel_matrix_to_dicom,
)
from .cost import _separable_smooth_3d, lpa_correlation
from .interp import _separable_resample_3d
from .weight import compute_weight_image


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
    max_iter: int = 5  # Per-volume GN iterations (was 23, now 5 for speed)
    twopass: bool = False  # Coarse blur + fine pass
    blur_fwhm: float = 0.0  # Pre-blur for estimation (mm)
    dxy_thresh: float = 0.07  # Translation convergence (voxels)
    dph_thresh: float = 0.21  # Rotation convergence (degrees)
    chain_init: bool = True  # Init from previous volume
    automask: bool = False  # Use automask for weighting
    weight_automask: bool = False  # automask × continuous weight (best of both)
    lpa_sigma: float = 4.0  # Gaussian sigma for LPA cost
    powell_maxfev: int = 100  # Max function evals for LPA Powell
    fixed_iter: bool = False  # Fast mode: skip convergence, run exactly max_iter
    compile: bool = True  # Use torch.compile for hot path (CUDA only)
    device: str | None = None
    verb: int = 1
    quad_filter_size: int = 7
    quad_center_freq: float = math.pi / 3.0
    quad_bandwidth: float = 2.0


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
    reg = eps * torch.eye(6, device=device, dtype=dtype)
    JtWJ_reg = JtWJ + reg

    for it in range(config.max_iter):
        matrix = p2m_fn(params)
        warped = resample_fn(source, matrix, coords, config.interp, output_shape)
        warped_flat = warped.reshape(-1)

        # Weighted residual
        residual = weight_flat * (base_flat - warped_flat)

        # Right-hand side: J^T W r
        JtWr = WJ @ residual  # (6,)

        # Solve 6×6 system
        dp = torch.linalg.solve(JtWJ_reg, JtWr)

        # Update only rigid params (first 6)
        params[:6] += dp

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
    dtype = source.dtype
    params = init_params.clone()

    eps = 1e-6 * JtWJ.diagonal().mean()
    reg = eps * torch.eye(6, device=device, dtype=dtype)
    JtWJ_reg = JtWJ + reg

    for it in range(config.max_iter):
        matrix = p2m_fn(params)
        src_coords = matrix @ coords_masked  # (4, M)
        warped = _separable_resample_3d(
            source, src_coords[0], src_coords[1], src_coords[2], config.interp
        )  # (M,)
        residual = weight_flat_masked * (base_flat_masked - warped)

        JtWr = WJ_masked @ residual
        dp = torch.linalg.solve(JtWJ_reg, JtWr)

        params[:6] += dp

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
    dtype = source.dtype
    params = init_params.clone()
    output_shape = tuple(source.shape)

    eps = 1e-6 * JtWJ.diagonal().mean()
    reg = eps * torch.eye(6, device=device, dtype=dtype)
    JtWJ_reg = JtWJ + reg

    for _ in range(max_iter):
        matrix = params_to_matrix(params)
        warped = resample_affine_fast(source, matrix, coords, interp, output_shape)
        residual = weight_flat * (base_flat - warped.reshape(-1))

        JtWr = WJ @ residual
        dp = torch.linalg.solve(JtWJ_reg, JtWr)

        params[:6] += dp

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
    dtype = source.dtype
    params = init_params.clone()

    eps = 1e-6 * JtWJ.diagonal().mean()
    reg = eps * torch.eye(6, device=device, dtype=dtype)
    JtWJ_reg = JtWJ + reg

    for _ in range(max_iter):
        matrix = params_to_matrix(params)
        src_coords = matrix @ coords_masked  # (4, M)
        warped = _separable_resample_3d(
            source, src_coords[0], src_coords[1], src_coords[2], interp
        )  # (M,)
        residual = weight_flat_masked * (base_flat_masked - warped)

        JtWr = WJ_masked @ residual  # (6,)
        dp = torch.linalg.solve(JtWJ_reg, JtWr)

        params[:6] += dp

    return params


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
        cost = -lpa_correlation(base, warped, weight, sigma=config.lpa_sigma)
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

    AFNI 3dvolreg format: roll pitch yaw dS dL dP (degrees, mm)
    - roll  = rotation about I-S axis (z in DICOM) = -rz
    - pitch = rotation about R-L axis (x in DICOM) = rx
    - yaw   = rotation about A-P axis (y in DICOM)   ry
    - dS    = displacement in Superior direction (z in DICOM)   -dz
    - dL    = displacement in Left direction (x in DICOM)       dx
    - dP    = displacement in Posterior direction (y in DICOM)  dy

    Args:
        params_array: (nt, 6) array of [dx, dy, dz, rz, rx, ry] in DICOM space.
        path: output file path.
    """
    with open(path, "w") as f:
        for t in range(params_array.shape[0]):
            dx, dy, dz = params_array[t, 0], params_array[t, 1], params_array[t, 2]
            rz, rx, ry = params_array[t, 3], params_array[t, 4], params_array[t, 5]
            # AFNI format: roll pitch yaw dS dL dP
            # Mapping: roll=-rz, pitch=rx, yaw=ry, dS=-dz, dL=dx, dP=dy
            f.write(
                f"  {-rz:8.4f}  {rx:8.4f}  {ry:8.4f}"
                f"  {-dz:8.4f}  {dx:8.4f}  {dy:8.4f}\n"
            )


def save_moco_aff12(
    matrices_dicom: np.ndarray,
    path: str,
) -> None:
    """Save multi-row .aff12.1D file with one row per volume.

    Saves the INVERSE transformation matrix (base←source), which is what AFNI's 3dvolreg outputs.

    Args:
        matrices_dicom: (nt, 4, 4) DICOM-space matrices (forward transform: source→base).
        path: output file path.
    """
    with open(path, "w") as f:
        for t in range(matrices_dicom.shape[0]):
            # Invert the matrix (AFNI saves the inverse transformation)
            M = matrices_dicom[t]
            M_inv = np.linalg.inv(M)

            vals = []
            for i in range(3):
                for j in range(4):
                    vals.append(f"{M_inv[i, j]:.10f}")
            f.write("  ".join(vals) + "\n")


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
            # AFNI format: roll pitch yaw dS dL dP
            # Mapping: roll=-rz, pitch=rx, yaw=ry, dS=-dz, dL=dx, dP=dy
            f.write(
                f"  {t:4d}  {-rz:8.4f}  {rx:8.4f}  {ry:8.4f}"
                f"  {-dz:8.4f}  {dx:8.4f}  {dy:8.4f}"
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
        print(
            f"ffs_moco: {nt} volumes, {vol_shape}, device={device}, cost={config.cost}"
        )

    # Get base volume
    if base_vol is not None:
        base = base_vol.to(device=device, dtype=dtype)
    else:
        base = timeseries[config.base_index].to(device=device, dtype=dtype)

    # Compute weight image
    if config.weight_automask:
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

    # Setup for WLS
    if config.cost == "wls":
        t0 = time.time()
        derivs = compute_derivative_images(base_est, device, verb=config.verb)
        if config.verb >= 1:
            print(f"  Derivative images: {time.time() - t0:.2f}s")

        # Pre-compute normal equations
        weight_flat = weight.reshape(1, -1)  # (1, N)
        base_flat = base_est.reshape(-1)  # (N,)

        WJ = weight_flat * derivs  # (6, N) — broadcast weight across 6 rows
        JtWJ = WJ @ WJ.t()  # (6, 6)

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

    # Compute mask for efficient resampling (skip zero-weight voxels)
    use_masked = False
    if config.cost == "wls":
        mask_idx = (weight.reshape(-1) > 0).nonzero(as_tuple=True)[0]
        M = mask_idx.numel()
        if M < N:
            use_masked = True
            coords_masked = homo_coords[:, mask_idx]  # (4, M)
            base_flat_masked = base_flat[mask_idx]  # (M,)
            weight_flat_masked = weight_flat_1d[mask_idx]  # (M,)
            WJ_masked = WJ[:, mask_idx]  # (6, M)
            JtWJ = WJ_masked @ WJ_masked.t()  # (6, 6) recompute from masked
            if config.verb >= 1:
                print(f"  Active voxels: {M:,}/{N:,} ({100 * M / N:.1f}%)")

    # Create compiled versions of hot-path functions for CUDA (reduces kernel launch overhead)
    if config.compile and device.type == "cuda":
        torch.set_float32_matmul_precision("high")
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
            _p2m = (
                params_to_matrix  # Uncompiled - called from within compiled _gn_fixed
            )
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
        JtWJ_c = WJ_c @ WJ_c.t()
        wf1_coarse = weight_coarse.reshape(-1)
    else:
        base_coarse = weight_coarse = None
        bf_coarse = wf1_coarse = WJ_c = JtWJ_c = None

    # Allocate output arrays
    aligned = torch.zeros_like(timeseries)  # CPU
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
    pbar = tqdm(
        range(nt),
        desc="  Registering",
        disable=disable_pbar,
        unit="vol",
        ncols=80,
    )

    for t in pbar:
        # Mark step begin for CUDA graphs (prevents output tensor reuse issues)
        if use_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()

        if t == config.base_index and base_vol is None:
            # Base volume — identity transform, just copy
            aligned[t] = timeseries[t]
            mat = params_to_matrix(identity)
            matrices_vox[t] = mat.cpu()
            all_params[t] = identity.cpu().numpy()
            rms_before[t] = 0.0
            rms_after[t] = 0.0
            continue

        # Load source volume to GPU
        source = timeseries[t].to(device=device, dtype=dtype)
        source_est = (
            _blur_volume(source, config.blur_fwhm) if config.blur_fwhm > 0 else source
        )

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
            warped_check = _resample(
                source_est, mat_result, homo_coords, config.interp, vol_shape
            )
            rms_result = _weighted_rms(base_est, warped_check, weight)

            rms_identity = _weighted_rms(base_est, source_est, weight)
            if rms_result > rms_identity * 1.05 and not torch.equal(
                init_params[:6], identity[:6]
            ):
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

        # Store results
        prev_params = params.clone()
        mat = _p2m(params)
        matrices_vox[t] = mat.cpu()
        all_params[t] = params.detach().cpu().numpy()
        n_iters[t] = n_iter

        # Final resample with high-quality interpolation
        aligned_vol = apply_affine_interp(
            source, mat, config.final_interp, vol_shape, zero_outside=True
        )

        aligned[t] = aligned_vol.cpu()
        rms_after[t] = _unweighted_rms(base_est, aligned_vol)

        # Update progress bar with current RMS
        if not disable_pbar:
            pbar.set_postfix(
                rms=f"{rms_before[t]:.1f}→{rms_after[t]:.1f}", refresh=False
            )

        if config.verb >= 2:
            pbar.write(
                f"  Vol {t:4d}: {n_iter:2d} iter, "
                f"rms {rms_before[t]:.4f} -> {rms_after[t]:.4f}"
            )

    # Close progress bar
    if not disable_pbar:
        pbar.close()

    if config.verb >= 1:
        elapsed = time.time() - t_start
        per_vol = elapsed / max(nt - 1, 1)
        print(f"  Registration: {elapsed:.2f}s ({per_vol:.3f}s/vol)")

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

    return MocoResult(
        aligned=aligned,
        params=params_dicom,
        matrices_vox=matrices_vox,
        matrices_dicom=matrices_dicom_np,
        max_displacement=max_disp,
        rms_before=rms_before,
        rms_after=rms_after,
        n_iters=n_iters,
    )
