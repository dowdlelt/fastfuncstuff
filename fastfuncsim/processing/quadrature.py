"""Quadrature phase-based rigid body registration.

Implements the quadrature filter approach from BROCCOLI (Eklund et al. 2014)
for intensity-invariant motion correction. Local phase from complex quadrature
filter responses is matched between base and source volumes, providing robust
registration even under contrast changes.

The approach:
1. Apply 3 complex quadrature filters (one per axis) via FFT
2. Phase difference tells how much local structure shifted
3. Certainty auto-downweights unreliable measurements
4. Build normal equations from certainty-weighted phase differences
5. Solve for rigid body parameter update, iterate
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .affine import (
    params_to_matrix,
    resample_affine_fast,
)

# ---------------------------------------------------------------------------
# Filter design
# ---------------------------------------------------------------------------


def design_quadrature_filters(
    size: int = 7,
    center_freq: float = math.pi / 3.0,
    bandwidth: float = 2.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Design 3 complex quadrature filters (one per spatial axis).

    Each filter is a complex-valued 3D kernel that responds to local structure
    (edges, lines) oriented perpendicular to its axis direction.

    Args:
        size: Spatial extent of the filter (default 7).
        center_freq: Center frequency f0 (default pi/3).
        bandwidth: Lognormal bandwidth B (default 2.0).
        device: Torch device.
        dtype: Real dtype (complex will be derived).

    Returns:
        (3, size, size, size) complex tensor of spatial-domain filters.
    """
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    half = size // 2

    # Frequency coordinates for size^3 DFT
    freqs = torch.fft.fftfreq(size, device=device, dtype=dtype) * 2 * math.pi
    fz, fy, fx = torch.meshgrid(freqs, freqs, freqs, indexing="ij")

    # Magnitude of frequency vector
    f_mag = torch.sqrt(fx * fx + fy * fy + fz * fz)
    f_mag_safe = f_mag.clamp(min=1e-10)

    # Lognormal radial profile: R(f) = exp(-B * ln^2(|f| / f0))
    log_ratio = torch.log(f_mag_safe / center_freq)
    radial = torch.exp(-bandwidth * log_ratio * log_ratio)
    # Zero DC
    radial[0, 0, 0] = 0.0

    # Unit frequency vectors
    fx_hat = fx / f_mag_safe
    fy_hat = fy / f_mag_safe
    fz_hat = fz / f_mag_safe

    # Direction vectors for the 3 filters
    directions = [
        (fx_hat, fx),  # X direction: n=[1,0,0], dot = fx_hat, sign from fx
        (fy_hat, fy),  # Y direction: n=[0,1,0], dot = fy_hat, sign from fy
        (fz_hat, fz),  # Z direction: n=[0,0,1], dot = fz_hat, sign from fz
    ]

    filters = torch.zeros(3, size, size, size, device=device, dtype=cdtype)

    for d, (f_dot, f_component) in enumerate(directions):
        # Directional selectivity: D(f) = (f_hat . d_k)^2
        directional = f_dot * f_dot

        # Combined: radial * directional
        magnitude = radial * directional

        # Quadrature: zero negative frequencies along this axis
        # H(f . d_k) = 1 if f_component > 0, 0.5 if == 0, 0 if < 0
        half_plane = (f_component > 0).to(dtype) + 0.5 * (f_component == 0).to(dtype)

        # Full filter in frequency domain
        Q_freq = magnitude * half_plane

        # IFFT to spatial domain
        Q_spatial = torch.fft.ifftn(Q_freq.to(cdtype))

        # Shift so center is at (half, half, half)
        Q_spatial = torch.fft.fftshift(Q_spatial)

        filters[d] = Q_spatial

    return filters


# ---------------------------------------------------------------------------
# Filter application via FFT
# ---------------------------------------------------------------------------


def precompute_filter_ffts(
    filters: Tensor,
    vol_shape: tuple[int, int, int],
) -> Tensor:
    """Pad filters to volume size and pre-compute their FFTs.

    Args:
        filters: (3, fs, fs, fs) complex spatial-domain filters.
        vol_shape: (nz, ny, nx) volume dimensions.

    Returns:
        (3, nz, ny, nx) complex filter spectra.
    """
    nf = filters.shape[0]
    fs = filters.shape[1]
    nz, ny, nx = vol_shape
    device = filters.device
    cdtype = filters.dtype

    # Pad filters into volume-sized arrays (center the filter)
    padded = torch.zeros(nf, nz, ny, nx, device=device, dtype=cdtype)
    half = fs // 2

    # Place filter at origin and wrap around (for circular convolution)
    for d in range(nf):
        # ifftshift to put center at origin for correct circular convolution
        f_shifted = torch.fft.ifftshift(filters[d])
        padded[d, :fs, :fs, :fs] = f_shifted
        # Roll to handle the wrap-around properly
        # The filter is size fs, centered at (half, half, half) after fftshift
        # After ifftshift, center is at (0,0,0), tails wrap to end

    # FFT each filter
    spectra = torch.fft.fftn(padded, dim=(1, 2, 3))
    return spectra


def apply_quadrature_filters_fft(
    volume: Tensor,
    filter_spectra: Tensor,
) -> Tensor:
    """Apply pre-computed quadrature filters to a volume via FFT.

    Args:
        volume: (nz, ny, nx) real-valued volume.
        filter_spectra: (3, nz, ny, nx) complex filter FFTs.

    Returns:
        (3, nz, ny, nx) complex quadrature filter responses.
    """
    # Forward FFT of volume (once)
    vol_fft = torch.fft.fftn(volume)  # (nz, ny, nx) complex

    # Pointwise multiply with each filter spectrum, then inverse FFT
    # Broadcast: (3, nz, ny, nx) * (1, nz, ny, nx)
    product = filter_spectra * vol_fft.unsqueeze(0)

    # Inverse FFT for each direction
    responses = torch.fft.ifftn(product, dim=(1, 2, 3))

    return responses


# ---------------------------------------------------------------------------
# Phase difference and certainty
# ---------------------------------------------------------------------------


def compute_phase_diff_and_certainty(
    q_base: Tensor,
    q_source: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute phase difference and certainty from quadrature responses.

    Args:
        q_base: (3, nz, ny, nx) complex base filter responses.
        q_source: (3, nz, ny, nx) complex source filter responses.

    Returns:
        phase_diff: (3, N) real phase differences (flattened spatial dims).
        certainty: (3, N) real certainty weights (flattened spatial dims).
    """
    # Flatten spatial dimensions
    nf = q_base.shape[0]
    q_b = q_base.reshape(nf, -1)
    q_s = q_source.reshape(nf, -1)

    # Cross-product: q_base * conj(q_source)
    product = q_b * q_s.conj()

    # Phase difference
    phase_diff = product.angle()  # (3, N)

    # Certainty: |product| * cos^2(phase_diff / 2)
    # Maximum when phases agree, zero when opposite
    certainty = product.abs() * torch.cos(phase_diff / 2.0).square()

    return phase_diff, certainty


# ---------------------------------------------------------------------------
# Normal equations for rigid body
# ---------------------------------------------------------------------------


def build_phase_normal_equations(
    phase_diff: Tensor,
    certainty: Tensor,
    coords: Tensor,
    vol_shape: tuple[int, int, int],
) -> tuple[Tensor, Tensor]:
    """Build 6x6 normal equations from phase differences and certainty.

    The rigid body Jacobian for direction d with normal n_d gives coupling:
        t_i^d(x) = n_d . (d(displacement)/d(param_i))

    For X direction (n=[1,0,0]): [1, 0, 0, -(y-cy), 0, (z-cz)]
    For Y direction (n=[0,1,0]): [0, 1, 0, (x-cx), -(z-cz), 0]
    For Z direction (n=[0,0,1]): [0, 0, 1, 0, (y-cy), -(x-cx)]

    Rotation columns are scaled by pi/180 since params are in degrees.

    Args:
        phase_diff: (3, N) phase differences per direction.
        certainty: (3, N) certainty weights per direction.
        coords: (4, N) homogeneous coordinates.
        vol_shape: (nz, ny, nx) for computing center.

    Returns:
        A: (6, 6) normal equation matrix.
        h: (6,) right-hand side vector.
    """
    device = phase_diff.device
    dtype = phase_diff.real.dtype if phase_diff.is_complex() else phase_diff.dtype

    nz, ny, nx = vol_shape
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    cz = (nz - 1) / 2.0

    # Centered coordinates
    x = coords[0] - cx  # (N,)
    y = coords[1] - cy
    z = coords[2] - cz

    deg2rad = math.pi / 180.0

    # Phase diffs and certainties per direction
    dp_x, dp_y, dp_z = phase_diff[0], phase_diff[1], phase_diff[2]
    c_x, c_y, c_z = certainty[0], certainty[1], certainty[2]

    # For X direction: t = [1, 0, 0, -y*s, 0, z*s] where s = deg2rad
    # Non-zero entries: param 0 (=1), param 3 (=-y*s), param 5 (=z*s)
    tx_0 = torch.ones_like(x)          # param 0: 1
    tx_3 = -y * deg2rad                 # param 3: -(y-cy) * deg2rad
    tx_5 = z * deg2rad                  # param 5: (z-cz) * deg2rad

    # For Y direction: t = [0, 1, 0, x*s, 0, 0] -> params 1, 3, (4 = -z*s)
    # Actually: [0, 1, 0, x*s, -z*s, 0]
    ty_1 = torch.ones_like(x)          # param 1: 1
    ty_3 = x * deg2rad                  # param 3: (x-cx) * deg2rad
    ty_4 = -z * deg2rad                 # param 4: -(z-cz) * deg2rad

    # For Z direction: t = [0, 0, 1, 0, y*s, -x*s]
    tz_2 = torch.ones_like(x)          # param 2: 1
    tz_4 = y * deg2rad                  # param 4: (y-cy) * deg2rad
    tz_5 = -x * deg2rad                # param 5: -(x-cx) * deg2rad

    # Build A[i,j] = sum_d sum_x certainty_d * t_i^d * t_j^d
    # Build h[i]   = sum_d sum_x certainty_d * phase_diff_d * t_i^d
    A = torch.zeros(6, 6, device=device, dtype=dtype)
    h = torch.zeros(6, device=device, dtype=dtype)

    # --- X direction contributions ---
    # Non-zero t entries: 0, 3, 5
    c_dp_x = c_x * dp_x  # certainty * phase_diff for X

    # Diagonal
    A[0, 0] += c_x.sum()                              # t0*t0 = 1*1
    A[3, 3] += (c_x * tx_3 * tx_3).sum()              # t3*t3
    A[5, 5] += (c_x * tx_5 * tx_5).sum()              # t5*t5

    # Off-diagonal (symmetric)
    A[0, 3] += (c_x * tx_3).sum()                     # t0*t3
    A[0, 5] += (c_x * tx_5).sum()                     # t0*t5
    A[3, 5] += (c_x * tx_3 * tx_5).sum()              # t3*t5

    # RHS
    h[0] += c_dp_x.sum()                              # t0 * c*dp
    h[3] += (c_dp_x * tx_3).sum()                     # t3 * c*dp
    h[5] += (c_dp_x * tx_5).sum()                     # t5 * c*dp

    # --- Y direction contributions ---
    c_dp_y = c_y * dp_y

    A[1, 1] += c_y.sum()
    A[3, 3] += (c_y * ty_3 * ty_3).sum()
    A[4, 4] += (c_y * ty_4 * ty_4).sum()

    A[1, 3] += (c_y * ty_3).sum()
    A[1, 4] += (c_y * ty_4).sum()
    A[3, 4] += (c_y * ty_3 * ty_4).sum()

    h[1] += c_dp_y.sum()
    h[3] += (c_dp_y * ty_3).sum()
    h[4] += (c_dp_y * ty_4).sum()

    # --- Z direction contributions ---
    c_dp_z = c_z * dp_z

    A[2, 2] += c_z.sum()
    A[4, 4] += (c_z * tz_4 * tz_4).sum()
    A[5, 5] += (c_z * tz_5 * tz_5).sum()

    A[2, 4] += (c_z * tz_4).sum()
    A[2, 5] += (c_z * tz_5).sum()
    A[4, 5] += (c_z * tz_4 * tz_5).sum()

    h[2] += c_dp_z.sum()
    h[4] += (c_dp_z * tz_4).sum()
    h[5] += (c_dp_z * tz_5).sum()

    # Symmetrize
    A = A + A.triu(1).T

    return A, h


# ---------------------------------------------------------------------------
# Gauss-Newton solvers
# ---------------------------------------------------------------------------


def quadrature_gn_rigid(
    base: Tensor,
    source: Tensor,
    q_base: Tensor,
    filter_spectra: Tensor,
    init_params: Tensor,
    coords: Tensor,
    vol_shape: tuple[int, int, int],
    max_iter: int = 5,
    interp: str = "heptic",
    weight: Tensor | None = None,
    dxy_thresh: float = 0.07,
    dph_thresh: float = 0.21,
) -> tuple[Tensor, int]:
    """Quadrature phase-based GN rigid registration with convergence check.

    Args:
        base: (nz, ny, nx) base volume (unused directly, kept for API consistency).
        source: (nz, ny, nx) source volume.
        q_base: (3, nz, ny, nx) complex base quadrature responses.
        filter_spectra: (3, nz, ny, nx) complex filter FFTs.
        init_params: (12,) initial parameters.
        coords: (4, N) homogeneous coordinates.
        vol_shape: (nz, ny, nx).
        max_iter: Maximum iterations.
        interp: Interpolation method.
        weight: Optional (nz, ny, nx) weight mask.
        dxy_thresh: Translation convergence threshold (voxels).
        dph_thresh: Rotation convergence threshold (degrees).

    Returns:
        (params, n_iters): optimized parameters and iteration count.
    """
    device = source.device
    dtype = source.dtype
    params = init_params.clone()

    # Regularization
    reg = 1e-6 * torch.eye(6, device=device, dtype=dtype)

    # Flatten weight if provided
    weight_flat = weight.reshape(1, -1) if weight is not None else None

    for it in range(max_iter):
        matrix = params_to_matrix(params)
        warped = resample_affine_fast(source, matrix, coords, interp, vol_shape)

        q_source = apply_quadrature_filters_fft(warped, filter_spectra)
        phase_diff, certainty = compute_phase_diff_and_certainty(q_base, q_source)

        if weight_flat is not None:
            certainty = certainty * weight_flat

        A, h = build_phase_normal_equations(phase_diff, certainty, coords, vol_shape)
        dp = torch.linalg.solve(A + reg, h)

        params[:6] += dp

        # Convergence check
        trans_converged = (dp[:3].abs() < dxy_thresh).all()
        rot_converged = (dp[3:].abs() < dph_thresh).all()
        if trans_converged and rot_converged:
            return params, it + 1

    return params, max_iter


def quadrature_gn_rigid_fixed(
    source: Tensor,
    q_base: Tensor,
    filter_spectra: Tensor,
    init_params: Tensor,
    coords: Tensor,
    vol_shape: tuple[int, int, int],
    max_iter: int,
    interp: str,
    weight_flat: Tensor | None,
) -> Tensor:
    """Quadrature phase-based GN rigid registration — fixed iterations.

    No convergence check, suitable for torch.compile.

    Args:
        source: (nz, ny, nx) source volume.
        q_base: (3, nz, ny, nx) complex base quadrature responses.
        filter_spectra: (3, nz, ny, nx) complex filter FFTs.
        init_params: (12,) initial parameters.
        coords: (4, N) homogeneous coordinates.
        vol_shape: tuple (nz, ny, nx).
        max_iter: Exact number of iterations.
        interp: Interpolation method.
        weight_flat: Optional (1, N) flattened weight mask.

    Returns:
        (12,) optimized parameters.
    """
    device = source.device
    dtype = source.dtype
    params = init_params.clone()

    reg = 1e-6 * torch.eye(6, device=device, dtype=dtype)

    for _ in range(max_iter):
        matrix = params_to_matrix(params)
        warped = resample_affine_fast(
            source, matrix, coords, interp, tuple(vol_shape)
        )

        q_source = apply_quadrature_filters_fft(warped, filter_spectra)
        phase_diff, certainty = compute_phase_diff_and_certainty(q_base, q_source)

        if weight_flat is not None:
            certainty = certainty * weight_flat

        A, h = build_phase_normal_equations(phase_diff, certainty, coords, vol_shape)
        dp = torch.linalg.solve(A + reg, h)

        params[:6] += dp

    return params


# ---------------------------------------------------------------------------
# Scalar cost for Powell optimizer path
# ---------------------------------------------------------------------------


def quadrature_phase_cost(
    q_base: Tensor,
    q_source: Tensor,
    weight: Tensor | None = None,
) -> Tensor:
    """Compute scalar phase agreement cost (higher = better alignment).

    cost = sum_d sum_x certainty_d(x) * cos(phase_diff_d(x))

    Maximized when all phase differences are zero.

    Args:
        q_base: (3, nz, ny, nx) complex base responses.
        q_source: (3, nz, ny, nx) complex source responses.
        weight: Optional (nz, ny, nx) weight mask.

    Returns:
        Scalar cost tensor.
    """
    nf = q_base.shape[0]
    q_b = q_base.reshape(nf, -1)
    q_s = q_source.reshape(nf, -1)

    product = q_b * q_s.conj()
    phase_diff = product.angle()
    certainty = product.abs() * torch.cos(phase_diff / 2.0).square()

    if weight is not None:
        certainty = certainty * weight.reshape(1, -1)

    cost = (certainty * torch.cos(phase_diff)).sum()
    return cost
