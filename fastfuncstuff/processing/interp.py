"""Trilinear interpolation on GPU, including batched variants.

Provides fast 3D interpolation for warping images, matching the interpolation
used in AFNI's Hwarp_apply() and IW3D_warp_floatim().

Key functions:
  - trilinear_interpolate: single-volume, N sample points
  - warp_image_linear: apply displacement warp to full image
  - batched_trilinear_interpolate: B patches x V voxels, single volume
  - batched_compose_and_interpolate: fused warp composition + source sampling
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _grid_sample_3d(
    input: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> Tensor:
    """grid_sample wrapper with MPS compatibility.

    MPS doesn't support padding_mode='border', so on MPS we clamp the grid
    to [-1, 1] and use padding_mode='zeros' instead (equivalent result since
    all coordinates are in-bounds after clamping).
    """
    if input.device.type == "mps":
        grid = grid.clamp(-1.0, 1.0)
        return F.grid_sample(
            input,
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=align_corners,
        )
    return F.grid_sample(
        input,
        grid,
        mode=mode,
        padding_mode="border",
        align_corners=align_corners,
    )


def trilinear_interpolate(volume: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """Trilinear interpolation of a 3D volume at arbitrary (x, y, z) locations.

    Uses PyTorch's grid_sample for GPU-accelerated interpolation.

    Args:
        volume: (nz, ny, nx) float tensor.
        x, y, z: (N,) float tensors - sample locations in index coordinates.

    Returns:
        (N,) float tensor of interpolated values.
    """
    nz, ny, nx = volume.shape

    # Ensure coordinates match volume dtype (grid_sample requires same dtype)
    x = x.to(dtype=volume.dtype)
    y = y.to(dtype=volume.dtype)
    z = z.to(dtype=volume.dtype)

    gx = 2.0 * x / (nx - 1) - 1.0 if nx > 1 else x * 0.0
    gy = 2.0 * y / (ny - 1) - 1.0 if ny > 1 else y * 0.0
    gz = 2.0 * z / (nz - 1) - 1.0 if nz > 1 else z * 0.0

    vol_5d = volume[None, None, :, :, :]
    grid = torch.stack([gx, gy, gz], dim=-1)[None, None, None, :, :]

    result = _grid_sample_3d(vol_5d, grid)
    return result.reshape(-1)


def warp_image_linear(
    source: Tensor,
    warp_xd: Tensor,
    warp_yd: Tensor,
    warp_zd: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Apply a displacement warp to a source image using trilinear interpolation.

    The warp is defined as displacements: the voxel at index (i,j,k) in the
    OUTPUT (warp grid) maps to location (i + xd, j + yd, k + zd) in the source.
    """
    out_nz, out_ny, out_nx = warp_xd.shape
    src_nz, src_ny, src_nx = source.shape
    device = source.device

    kk, jj, ii = torch.meshgrid(
        torch.arange(out_nz, dtype=torch.float32, device=device),
        torch.arange(out_ny, dtype=torch.float32, device=device),
        torch.arange(out_nx, dtype=torch.float32, device=device),
        indexing="ij",
    )

    x_coords = ii + warp_xd
    y_coords = jj + warp_yd
    z_coords = kk + warp_zd

    # Zero outside source bounds
    out_of_bounds = (
        (x_coords < -0.5) | (x_coords > src_nx - 0.5) |
        (y_coords < -0.5) | (y_coords > src_ny - 0.5) |
        (z_coords < -0.5) | (z_coords > src_nz - 0.5)
    )

    x_coords = x_coords.clamp(-0.499, src_nx - 0.501)
    y_coords = y_coords.clamp(-0.499, src_ny - 0.501)
    z_coords = z_coords.clamp(-0.499, src_nz - 0.501)

    gx = 2.0 * x_coords / (src_nx - 1) - 1.0 if src_nx > 1 else x_coords * 0.0
    gy = 2.0 * y_coords / (src_ny - 1) - 1.0 if src_ny > 1 else y_coords * 0.0
    gz = 2.0 * z_coords / (src_nz - 1) - 1.0 if src_nz > 1 else z_coords * 0.0

    grid = torch.stack([gx, gy, gz], dim=-1)[None, :, :, :, :]
    vol_5d = source[None, None, :, :, :]

    result = _grid_sample_3d(vol_5d, grid)[0, 0]
    result[out_of_bounds] = 0.0

    if mask is not None:
        result = result * mask.float()

    return result


def warp_image_wsinc5(
    source: Tensor,
    warp_xd: Tensor,
    warp_yd: Tensor,
    warp_zd: Tensor,
) -> Tensor:
    """Apply displacement warp using wsinc5 (Hanning-windowed sinc) interpolation.

    The warp is defined as displacements: voxel (i,j,k) in OUTPUT maps to
    (i + xd, j + yd, k + zd) in the source.
    """
    out_nz, out_ny, out_nx = warp_xd.shape
    device = source.device

    kk, jj, ii = torch.meshgrid(
        torch.arange(out_nz, dtype=torch.float32, device=device),
        torch.arange(out_ny, dtype=torch.float32, device=device),
        torch.arange(out_nx, dtype=torch.float32, device=device),
        indexing="ij",
    )

    x_coords = ii + warp_xd
    y_coords = jj + warp_yd
    z_coords = kk + warp_zd

    return wsinc5_resample_3d(source, x_coords, y_coords, z_coords)


# Selectable warp interpolation kernels, in increasing cost order. "linear" is
# the fast grid_sample path; the rest are AFNI's separable Lagrange / windowed-
# sinc interpolants (see _KERNELS below).
WARP_INTERP_MODES = ("nearest", "linear", "cubic", "quintic", "heptic", "wsinc5")

# Accept AFNI's "NN" spelling for nearest-neighbor.
_INTERP_ALIASES = {"nn": "nearest"}


def normalize_interp_mode(mode: str) -> str:
    """Canonicalize an interpolation-mode name (e.g. ``"NN"`` -> ``"nearest"``)."""
    m = mode.lower()
    m = _INTERP_ALIASES.get(m, m)
    if m not in WARP_INTERP_MODES:
        raise ValueError(f"unknown interp mode {mode!r}; choose from {WARP_INTERP_MODES}")
    return m


def nearest_resample_3d(
    source: Tensor, x_coords: Tensor, y_coords: Tensor, z_coords: Tensor
) -> Tensor:
    """Nearest-neighbor resample (AFNI ``-interp NN``).

    No interpolation: each output samples the single closest source voxel. The
    only mode that preserves integer label values exactly, so it is the correct
    choice for atlas / ROI datasets. Out-of-bounds samples are 0, matching the
    other kernels.
    """
    nz, ny, nx = source.shape
    out_shape = x_coords.shape
    xf, yf, zf = x_coords.reshape(-1), y_coords.reshape(-1), z_coords.reshape(-1)
    out_of_bounds = (
        (xf < -0.5) | (xf > nx - 0.5) |
        (yf < -0.5) | (yf > ny - 0.5) |
        (zf < -0.5) | (zf > nz - 0.5)
    )
    xi = xf.round().long().clamp(0, nx - 1)
    yi = yf.round().long().clamp(0, ny - 1)
    zi = zf.round().long().clamp(0, nz - 1)
    result = source[zi, yi, xi]
    result[out_of_bounds] = 0.0
    return result.reshape(out_shape)


def warp_image(
    source: Tensor,
    warp_xd: Tensor,
    warp_yd: Tensor,
    warp_zd: Tensor,
    mode: str = "linear",
    mask: Tensor | None = None,
) -> Tensor:
    """Apply a displacement warp to ``source`` with a selectable interpolation kernel.

    ``mode="linear"`` dispatches to :func:`warp_image_linear` (the fast estimation
    path); ``mode="nearest"`` to :func:`nearest_resample_3d`. Higher-order modes
    use the separable kernels for sharper output, as in AFNI's
    ``-ainterp``/``-final``. Displacement convention matches
    :func:`warp_image_linear`: output voxel (i,j,k) samples source at (i+xd,j+yd,k+zd).
    """
    mode = normalize_interp_mode(mode)
    if mode == "linear":
        return warp_image_linear(source, warp_xd, warp_yd, warp_zd, mask=mask)

    out_nz, out_ny, out_nx = warp_xd.shape
    device = source.device
    kk, jj, ii = torch.meshgrid(
        torch.arange(out_nz, dtype=torch.float32, device=device),
        torch.arange(out_ny, dtype=torch.float32, device=device),
        torch.arange(out_nx, dtype=torch.float32, device=device),
        indexing="ij",
    )
    if mode == "nearest":
        result = nearest_resample_3d(source, ii + warp_xd, jj + warp_yd, kk + warp_zd)
    else:
        result = _separable_resample_3d(
            source, ii + warp_xd, jj + warp_yd, kk + warp_zd, mode
        )
    if mask is not None:
        result = result * mask.float()
    return result


# ---------------------------------------------------------------------------
# Interpolation kernels
# ---------------------------------------------------------------------------

# AFNI wsinc5 constants (GA_interp_wsinc5p, the cubical/separable default).
_WSINC5_IRAD = 5      # window radius -> 10 taps at floor offsets -4..+5
_WSINC5_WRAD = 5.001  # AFNI WRAD = 0.001 + IRAD (float radius for the window arg)
# M3(x) = minimum-sidelobe 3-term window (AFNI default, WFUN=0). Note M3(0)==1,
# so applying it unconditionally matches AFNI's "if d/WRAD > WCUT" guard for the
# default taper cut WCUT=0.
_M3_A, _M3_B, _M3_C = 0.4243801, 0.4973406, 0.0782793


def _wsinc5_kernel(fx: Tensor) -> Tensor:
    """Build AFNI-faithful wsinc5 weights (10 taps, floor convention).

    Mirrors ``GA_interp_wsinc5p`` in AFNI's ``mri_genalign_util.c`` (the cubical
    tensor-product branch, which is the default -- ``AFNI_WSINC5_SPHERICAL`` is
    off by default). For ``fx`` in [0, 1) the taps are at integer offsets
    ``-4 .. +5`` from ``floor(x)`` and each weight is::

        d  = |fx - offset|
        w  = sinc(d) * M3(d / WRAD)

    with ``sinc(t) = sin(pi t)/(pi t)`` and ``M3`` the minimum-sidelobe 3-term
    window. This replaces the earlier Hanning/round-11 approximation so output
    matches ``3dNwarpApply -interp wsinc5``.

    Args:
        fx: (N,) fractional position in [0, 1) past the floor integer.

    Returns:
        (N, 10) kernel weights, normalized to sum to 1.
    """
    offsets = torch.arange(
        -(_WSINC5_IRAD - 1), _WSINC5_IRAD + 1, dtype=fx.dtype, device=fx.device
    )  # -4 .. +5  (10 taps)
    d = (fx[:, None] - offsets[None, :]).abs()  # (N, 10)

    pid = torch.pi * d
    sinc = torch.where(d < 1e-7, torch.ones_like(d), torch.sin(pid) / pid)

    pit = torch.pi * (d / _WSINC5_WRAD)
    m3 = _M3_A + _M3_B * torch.cos(pit) + _M3_C * torch.cos(2.0 * pit)

    w = sinc * m3  # (N, 10)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-10)
    return w


# ---------------------------------------------------------------------------
# Lagrange polynomial interpolation kernels (matching AFNI exactly)
#
# These are all based on floor(x) convention: fx in [0, 1) is the fraction
# past the floor integer.  Taps are at offsets from the floor position.
# ---------------------------------------------------------------------------

_CUBIC_HALF = 2       # 4 taps: floor-1 .. floor+2
_QUINTIC_HALF = 3     # 6 taps: floor-2 .. floor+3
_HEPTIC_HALF = 4      # 8 taps: floor-3 .. floor+4


def _cubic_kernel(fx: Tensor) -> Tensor:
    """4-tap cubic Lagrange kernel (AFNI MRI_CUBIC).

    Args:
        fx: (N,) fractional position in [0, 1).

    Returns:
        (N, 4) weights for taps at offsets [-1, 0, +1, +2] from floor.
    """
    # AFNI uses P_FACTOR=1/216=1/6^3 after all 3 separable passes.
    # For per-axis weights we need 1/6 so each axis sums to 1.
    P_FACTOR_1D = 1.0 / 6.0
    x = fx
    w_m1 = -(x) * (x - 1) * (x - 2)
    w_00 = 3 * (x + 1) * (x - 1) * (x - 2)
    w_p1 = -3 * x * (x + 1) * (x - 2)
    w_p2 = x * (x + 1) * (x - 1)
    w = torch.stack([w_m1, w_00, w_p1, w_p2], dim=1) * P_FACTOR_1D  # (N, 4)
    return w


def _quintic_kernel(fx: Tensor) -> Tensor:
    """6-tap quintic Lagrange kernel (AFNI MRI_QUINTIC).

    Args:
        fx: (N,) fractional position in [0, 1).

    Returns:
        (N, 6) weights for taps at offsets [-2, -1, 0, +1, +2, +3] from floor.
    """
    x = fx
    x2 = x * x
    x2m1 = x2 - 1.0
    x2m4 = x2 - 4.0
    w_m2 = x * x2m1 * (2.0 - x) * (x - 3.0) * 0.008333333
    w_m1 = x * x2m4 * (x - 1.0) * (x - 3.0) * 0.041666667
    w_00 = x2m4 * x2m1 * (3.0 - x) * 0.083333333
    w_p1 = x * x2m4 * (x + 1.0) * (x - 3.0) * 0.083333333
    w_p2 = x * x2m1 * (x + 2.0) * (3.0 - x) * 0.041666667
    w_p3 = x * x2m1 * x2m4 * 0.008333333
    w = torch.stack([w_m2, w_m1, w_00, w_p1, w_p2, w_p3], dim=1)  # (N, 6)
    return w


def _heptic_kernel(fx: Tensor) -> Tensor:
    """8-tap heptic Lagrange kernel (AFNI MRI_HEPTIC).

    Args:
        fx: (N,) fractional position in [0, 1).

    Returns:
        (N, 8) weights for taps at offsets [-3,-2,-1,0,+1,+2,+3,+4] from floor.
    """
    x = fx
    x2 = x * x
    x2m1 = x2 - 1.0
    x2m4 = x2 - 4.0
    x2m9 = x2 - 9.0
    w_m3 = x * x2m1 * x2m4 * (x - 3.0) * (4.0 - x) * 0.0001984126984
    w_m2 = x * x2m1 * (x - 2.0) * x2m9 * (x - 4.0) * 0.001388888889
    w_m1 = x * (x - 1.0) * x2m4 * x2m9 * (4.0 - x) * 0.004166666667
    w_00 = x2m1 * x2m4 * x2m9 * (x - 4.0) * 0.006944444444
    w_p1 = x * (x + 1.0) * x2m4 * x2m9 * (4.0 - x) * 0.006944444444
    w_p2 = x * x2m1 * (x + 2.0) * x2m9 * (x - 4.0) * 0.004166666667
    w_p3 = x * x2m1 * x2m4 * (x + 3.0) * (4.0 - x) * 0.001388888889
    w_p4 = x * x2m1 * x2m4 * x2m9 * 0.0001984126984
    w = torch.stack([w_m3, w_m2, w_m1, w_00, w_p1, w_p2, w_p3, w_p4], dim=1)
    return w


# ---------------------------------------------------------------------------
# Generic separable 3D resampling
# ---------------------------------------------------------------------------

# Kernel registry: name -> (kernel_fn, half_width, uses_floor_convention)
# Floor convention: fx in [0,1), taps offset from floor(x)
# Round convention: frac in [-0.5,0.5], taps offset from round(x)
_KERNELS = {
    "wsinc5":  (_wsinc5_kernel,  5, True),   # floor convention, 10 taps (-4..+5)
    "cubic":   (_cubic_kernel,   2, True),    # floor convention
    "quintic": (_quintic_kernel, 3, True),    # floor convention
    "heptic":  (_heptic_kernel,  4, True),    # floor convention
}


def _resample_chunk_size(n_points: int, ntaps: int, device: torch.device) -> int:
    """Voxels per chunk for the separable resampler.

    The vectorized contraction gathers an ``ntaps**3`` neighborhood per output
    sample (1000 floats for wsinc5), so peak memory scales with chunk size. Size
    the chunk from the memory module's available-memory estimate (which already
    applies the GPU 0.5 safety factor) instead of hardcoding, per the
    [[Memory module]] rule.
    """
    from ..memory import get_available_memory

    # Neighborhood gather + einsum scratch + per-axis weight tables + coords.
    bytes_per_point = (2 * ntaps**3 + 4 * ntaps + 8) * 4
    avail = get_available_memory(device)
    chunk = int(avail // max(1, bytes_per_point))
    return max(1, min(n_points, chunk))


def _separable_resample_3d(
    source: Tensor,
    x_coords: Tensor,
    y_coords: Tensor,
    z_coords: Tensor,
    kernel_name: str,
) -> Tensor:
    """Resample a 3D volume using separable 1D kernel interpolation.

    Applied separably: for each Z-neighbor, for each Y-neighbor,
    interpolate along X, then combine across Y, then across Z.

    Args:
        source: (nz, ny, nx) float tensor.
        x_coords, y_coords, z_coords: output sample locations in index space.
        kernel_name: one of "wsinc5", "cubic", "quintic", "heptic".

    Returns:
        Resampled volume with same shape as coordinate arrays.
    """
    kernel_fn, H, use_floor = _KERNELS[kernel_name]
    nz, ny, nx = source.shape
    out_shape = x_coords.shape
    device = source.device
    dtype = source.dtype
    x_flat = x_coords.reshape(-1)
    y_flat = y_coords.reshape(-1)
    z_flat = z_coords.reshape(-1)
    N = x_flat.numel()

    if use_floor:
        # Floor convention: fx in [0,1), taps offset from floor position.
        # cubic(H=2): 4 taps [-1..+2], quintic(H=3): 6 [-2..+3],
        # heptic(H=4): 8 [-3..+4], wsinc5(H=5): 10 [-4..+5].
        ntaps = H * 2
        offsets = torch.arange(-(H - 1), H + 1, device=device)  # (ntaps,)
    else:
        # Round convention: frac in [-0.5, 0.5], symmetric taps [-H..+H].
        ntaps = 2 * H + 1
        offsets = torch.arange(-H, H + 1, device=device)

    result = torch.empty(N, dtype=dtype, device=device)

    # Chunk over output voxels: the gather materializes an (chunk, ntaps**3)
    # neighborhood, so chunking keeps peak memory bounded (and makes the kernel
    # affordable on MPS/CPU and for the auto-padded grids).
    chunk = _resample_chunk_size(N, ntaps, device)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        xf, yf, zf = x_flat[s:e], y_flat[s:e], z_flat[s:e]

        if use_floor:
            xb, yb, zb = xf.floor(), yf.floor(), zf.floor()
        else:
            xb, yb, zb = xf.round(), yf.round(), zf.round()

        wx = kernel_fn(xf - xb)  # (c, ntaps)
        wy = kernel_fn(yf - yb)
        wz = kernel_fn(zf - zb)

        xi = (xb.long()[:, None] + offsets[None, :]).clamp(0, nx - 1)  # (c, ntaps)
        yi = (yb.long()[:, None] + offsets[None, :]).clamp(0, ny - 1)
        zi = (zb.long()[:, None] + offsets[None, :]).clamp(0, nz - 1)

        # Gather the (c, tz, ty, tx) neighborhood and contract against the three
        # separable weight tables in one einsum -- no Python tap loops.
        neigh = source[
            zi[:, :, None, None], yi[:, None, :, None], xi[:, None, None, :]
        ]  # (c, ntaps, ntaps, ntaps)
        result[s:e] = torch.einsum("ctuv,ct,cu,cv->c", neigh, wz, wy, wx)

    # Zero out-of-bounds (matches AFNI's outval=0 outside [-0.5, n-0.5]).
    out_of_bounds = (
        (x_flat < -0.5) | (x_flat > nx - 0.5) |
        (y_flat < -0.5) | (y_flat > ny - 0.5) |
        (z_flat < -0.5) | (z_flat > nz - 0.5)
    )
    result[out_of_bounds] = 0.0

    return result.reshape(out_shape)


def wsinc5_resample_3d(
    source: Tensor,
    x_coords: Tensor,
    y_coords: Tensor,
    z_coords: Tensor,
) -> Tensor:
    """Resample a 3D volume at arbitrary coordinates using separable wsinc5."""
    return _separable_resample_3d(source, x_coords, y_coords, z_coords, "wsinc5")


def cubic_resample_3d(
    source: Tensor,
    x_coords: Tensor,
    y_coords: Tensor,
    z_coords: Tensor,
) -> Tensor:
    """Resample using cubic (4-tap) Lagrange interpolation (AFNI -cubic)."""
    return _separable_resample_3d(source, x_coords, y_coords, z_coords, "cubic")


def quintic_resample_3d(
    source: Tensor,
    x_coords: Tensor,
    y_coords: Tensor,
    z_coords: Tensor,
) -> Tensor:
    """Resample using quintic (6-tap) Lagrange interpolation (AFNI -quintic)."""
    return _separable_resample_3d(source, x_coords, y_coords, z_coords, "quintic")


def heptic_resample_3d(
    source: Tensor,
    x_coords: Tensor,
    y_coords: Tensor,
    z_coords: Tensor,
) -> Tensor:
    """Resample using heptic (8-tap) Lagrange interpolation (AFNI -heptic)."""
    return _separable_resample_3d(source, x_coords, y_coords, z_coords, "heptic")


# ---------------------------------------------------------------------------
# Batched interpolation for parallel patch processing
# ---------------------------------------------------------------------------


def batched_trilinear_interpolate(
    volume: Tensor,
    x: Tensor,
    y: Tensor,
    z: Tensor,
) -> Tensor:
    """Interpolate a single volume at (B, V) locations.

    Args:
        volume: (nz, ny, nx) single volume.
        x, y, z: each (B, V) - B patches, V voxels each.

    Returns:
        (B, V) interpolated values.
    """
    B, V = x.shape
    nz, ny, nx = volume.shape

    gx = 2.0 * x / (nx - 1) - 1.0 if nx > 1 else x * 0.0
    gy = 2.0 * y / (ny - 1) - 1.0 if ny > 1 else y * 0.0
    gz = 2.0 * z / (nz - 1) - 1.0 if nz > 1 else z * 0.0

    # (B, 1, 1, V, 3) grid for grid_sample
    grid = torch.stack([gx, gy, gz], dim=-1)[:, None, None, :, :]

    # Expand volume to batch: (B, 1, D, H, W) via broadcast
    vol_5d = volume[None, None].expand(B, 1, nz, ny, nx)

    result = _grid_sample_3d(vol_5d, grid)
    return result.reshape(B, V)


def batched_interp_3ch(
    vol_3ch: Tensor,
    x: Tensor,
    y: Tensor,
    z: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Interpolate a 3-channel volume at (B, V) locations in one grid_sample call.

    This fuses the 3 separate warp displacement interpolations into one call.

    Args:
        vol_3ch: (3, nz, ny, nx) - stacked xd, yd, zd fields.
        x, y, z: each (B, V) - sample locations.

    Returns:
        Three tensors of shape (B, V).
    """
    B, V = x.shape
    _, nz, ny, nx = vol_3ch.shape

    gx = 2.0 * x / (nx - 1) - 1.0 if nx > 1 else x * 0.0
    gy = 2.0 * y / (ny - 1) - 1.0 if ny > 1 else y * 0.0
    gz = 2.0 * z / (nz - 1) - 1.0 if nz > 1 else z * 0.0

    grid = torch.stack([gx, gy, gz], dim=-1)[:, None, None, :, :]  # (B, 1, 1, V, 3)

    # (B, 3, D, H, W) via expand
    vol_5d = vol_3ch[None].expand(B, 3, nz, ny, nx)

    result = _grid_sample_3d(vol_5d, grid)
    # result: (B, 3, 1, 1, V) -> (B, 3, V)
    result = result.reshape(B, 3, V)
    return result[:, 0], result[:, 1], result[:, 2]


def batched_compose_and_interpolate(
    source: Tensor,
    global_xd: Tensor,
    global_yd: Tensor,
    global_zd: Tensor,
    patch_xd: Tensor,
    patch_yd: Tensor,
    patch_zd: Tensor,
    ii_p: Tensor,
    jj_p: Tensor,
    kk_p: Tensor,
    ibots: Tensor,
    jbots: Tensor,
    kbots: Tensor,
    nx: int,
    ny: int,
    nz: int,
    global_warp_3ch: Tensor | None = None,
    base_i: Tensor | None = None,
    base_j: Tensor | None = None,
    base_k: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compose patch warps with global warp and interpolate source for B patches.

    This is the fused hot-path operation: one call handles warp composition
    and source sampling for all patches in a checkerboard phase.

    Args:
        source: (nz, ny, nx) source image.
        global_xd/yd/zd: (nz, ny, nx) current global displacement.
        patch_xd/yd/zd: (B, V) local patch displacements.
        ii_p/jj_p/kk_p: (V,) local coordinate grids for a patch.
        ibots/jbots/kbots: (B,) patch origin offsets.
        nx, ny, nz: Full image dimensions.
        global_warp_3ch: Optional pre-stacked (3, nz, ny, nx) tensor with
            [global_xd, global_yd, global_zd]. Avoids re-stacking every
            optimizer iteration (~45 MB saved per call for typical volumes).
        base_i/base_j/base_k: Optional pre-computed (B, V) coordinate offsets
            (ibots[:, None] + ii_p[None, :]). Avoids recomputing every iteration.

    Returns:
        warped_vals: (B, V) interpolated source values.
        ah_xd, ah_yd, ah_zd: (B, V) composed displacement fields.
    """
    # Global coordinates after local patch displacement: (B, V)
    if base_i is not None:
        xq = (base_i + patch_xd).clamp(0, nx - 1)
        yq = (base_j + patch_yd).clamp(0, ny - 1)
        zq = (base_k + patch_zd).clamp(0, nz - 1)
    else:
        xq = (ibots[:, None] + ii_p[None, :] + patch_xd).clamp(0, nx - 1)
        yq = (jbots[:, None] + jj_p[None, :] + patch_yd).clamp(0, ny - 1)
        zq = (kbots[:, None] + kk_p[None, :] + patch_zd).clamp(0, nz - 1)

    # Fused 3-channel global warp interpolation
    if global_warp_3ch is not None:
        warp_3ch = global_warp_3ch
    else:
        warp_3ch = torch.stack([global_xd, global_yd, global_zd], dim=0)
    axd, ayd, azd = batched_interp_3ch(warp_3ch, xq, yq, zq)

    # Composed displacement
    ah_xd = patch_xd + axd
    ah_yd = patch_yd + ayd
    ah_zd = patch_zd + azd

    # Source sample locations
    if base_i is not None:
        src_x = (ah_xd + base_i).clamp(-0.499, nx - 0.501)
        src_y = (ah_yd + base_j).clamp(-0.499, ny - 0.501)
        src_z = (ah_zd + base_k).clamp(-0.499, nz - 0.501)
    else:
        src_x = (ah_xd + ii_p[None, :] + ibots[:, None]).clamp(-0.499, nx - 0.501)
        src_y = (ah_yd + jj_p[None, :] + jbots[:, None]).clamp(-0.499, ny - 0.501)
        src_z = (ah_zd + kk_p[None, :] + kbots[:, None]).clamp(-0.499, nz - 0.501)

    # Batched source interpolation
    warped_vals = batched_trilinear_interpolate(source, src_x, src_y, src_z)

    return warped_vals, ah_xd, ah_yd, ah_zd
