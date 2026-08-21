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

import json
import os
import time
from collections.abc import Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from .interp_triton import separable_resample_3d_triton
except Exception:  # pragma: no cover - Triton is optional and CUDA-only
    separable_resample_3d_triton = None


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


def trilinear_interpolate_multi(volume: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """Trilinear interpolation of a **multi-channel** volume at ``(x, y, z)``.

    Same as :func:`trilinear_interpolate` but samples all channels in ONE ``grid_sample``
    call — the sample locations are shared across channels, so looping per channel just
    multiplies kernel launches (the dominant cost when many small interpolations run in a
    tight fit loop, e.g. sampling every TPM tissue at the warp grid).

    Args:
        volume: ``(C, nz, ny, nx)`` float tensor.
        x, y, z: ``(N,)`` sample locations in index coordinates.

    Returns:
        ``(N, C)`` interpolated values.
    """
    n_chan, nz, ny, nx = volume.shape
    x = x.to(dtype=volume.dtype)
    y = y.to(dtype=volume.dtype)
    z = z.to(dtype=volume.dtype)
    gx = 2.0 * x / (nx - 1) - 1.0 if nx > 1 else x * 0.0
    gy = 2.0 * y / (ny - 1) - 1.0 if ny > 1 else y * 0.0
    gz = 2.0 * z / (nz - 1) - 1.0 if nz > 1 else z * 0.0
    grid = torch.stack([gx, gy, gz], dim=-1)[None, None, None, :, :]  # (1,1,1,N,3)
    result = _grid_sample_3d(volume[None], grid)  # (1, C, 1, 1, N)
    return result.reshape(n_chan, -1).T.contiguous()  # (N, C)


def warp_image_linear(
    source: Tensor,
    warp_xd: Tensor,
    warp_yd: Tensor,
    warp_zd: Tensor,
    mask: Tensor | None = None,
    voxel_grid: tuple[Tensor, Tensor, Tensor] | None = None,
) -> Tensor:
    """Apply a displacement warp to a source image using trilinear interpolation.

    The warp is defined as displacements: the voxel at index (i,j,k) in the
    OUTPUT (warp grid) maps to location (i + xd, j + yd, k + zd) in the source.
    """
    out_nz, out_ny, out_nx = warp_xd.shape
    src_nz, src_ny, src_nx = source.shape
    device = source.device

    if voxel_grid is None:
        kk, jj, ii = torch.meshgrid(
            torch.arange(out_nz, dtype=torch.float32, device=device),
            torch.arange(out_ny, dtype=torch.float32, device=device),
            torch.arange(out_nx, dtype=torch.float32, device=device),
            indexing="ij",
        )
    else:
        kk, jj, ii = voxel_grid

    x_coords = ii + warp_xd
    y_coords = jj + warp_yd
    z_coords = kk + warp_zd

    # Zero outside source bounds
    out_of_bounds = (
        (x_coords < -0.5)
        | (x_coords > src_nx - 0.5)
        | (y_coords < -0.5)
        | (y_coords > src_ny - 0.5)
        | (z_coords < -0.5)
        | (z_coords > src_nz - 0.5)
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
        (xf < -0.5)
        | (xf > nx - 0.5)
        | (yf < -0.5)
        | (yf > ny - 0.5)
        | (zf < -0.5)
        | (zf > nz - 0.5)
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
        result = _separable_resample_3d(source, ii + warp_xd, jj + warp_yd, kk + warp_zd, mode)
    if mask is not None:
        result = result * mask.float()
    return result


def warp_image_multi(
    sources: Sequence[Tensor],
    warp_xd: Tensor,
    warp_yd: Tensor,
    warp_zd: Tensor,
    mode: str = "linear",
) -> list[Tensor]:
    """Warp several co-registered channels through one displacement field.

    The channels share the sample coordinates, so for the separable kernels
    (cubic/quintic/heptic/wsinc5) the OOB mask, grid-node fast path, index tables
    and kernel weights are built once and only the gather+contract repeats per
    channel -- the win when warping mag + real + imag for phase data, or any
    multi-echo stack, at the same pose. ``linear``/``nearest`` fall back to a
    per-channel loop (already cheap; grid_sample / gather dominate, not the setup).
    Returns one warped volume per input channel, in order.
    """
    mode = normalize_interp_mode(mode)
    if not sources:
        return []

    # Treat co-registered inputs as grid_sample channels. This shares both the
    # coordinate grid and the interpolation launch; looping channels is especially
    # expensive on MPS, where launch latency dominates small/medium volumes.
    if mode == "linear":
        stack = torch.stack(tuple(sources), dim=0)
        out_nz, out_ny, out_nx = warp_xd.shape
        src_nz, src_ny, src_nx = stack.shape[-3:]
        device = warp_xd.device
        kk, jj, ii = torch.meshgrid(
            torch.arange(out_nz, dtype=torch.float32, device=device),
            torch.arange(out_ny, dtype=torch.float32, device=device),
            torch.arange(out_nx, dtype=torch.float32, device=device),
            indexing="ij",
        )
        x = ii + warp_xd
        y = jj + warp_yd
        z = kk + warp_zd
        oob = (
            (x < -0.5)
            | (x > src_nx - 0.5)
            | (y < -0.5)
            | (y > src_ny - 0.5)
            | (z < -0.5)
            | (z > src_nz - 0.5)
        )
        x = x.clamp(-0.499, src_nx - 0.501)
        y = y.clamp(-0.499, src_ny - 0.501)
        z = z.clamp(-0.499, src_nz - 0.501)
        gx = 2.0 * x / (src_nx - 1) - 1.0 if src_nx > 1 else x * 0.0
        gy = 2.0 * y / (src_ny - 1) - 1.0 if src_ny > 1 else y * 0.0
        gz = 2.0 * z / (src_nz - 1) - 1.0 if src_nz > 1 else z * 0.0
        grid = torch.stack([gx, gy, gz], dim=-1)[None]
        result = _grid_sample_3d(stack[None], grid)[0]
        result[:, oob] = 0.0
        return list(result.unbind(0))

    if mode == "nearest":
        stack = torch.stack(tuple(sources), dim=0)
        out_nz, out_ny, out_nx = warp_xd.shape
        src_nz, src_ny, src_nx = stack.shape[-3:]
        device = warp_xd.device
        kk, jj, ii = torch.meshgrid(
            torch.arange(out_nz, dtype=torch.float32, device=device),
            torch.arange(out_ny, dtype=torch.float32, device=device),
            torch.arange(out_nx, dtype=torch.float32, device=device),
            indexing="ij",
        )
        x, y, z = ii + warp_xd, jj + warp_yd, kk + warp_zd
        oob = (
            (x < -0.5)
            | (x > src_nx - 0.5)
            | (y < -0.5)
            | (y > src_ny - 0.5)
            | (z < -0.5)
            | (z > src_nz - 0.5)
        )
        result = stack[
            :,
            z.round().long().clamp(0, src_nz - 1),
            y.round().long().clamp(0, src_ny - 1),
            x.round().long().clamp(0, src_nx - 1),
        ]
        result[:, oob] = 0.0
        return list(result.unbind(0))

    out_nz, out_ny, out_nx = warp_xd.shape
    device = warp_xd.device
    kk, jj, ii = torch.meshgrid(
        torch.arange(out_nz, dtype=torch.float32, device=device),
        torch.arange(out_ny, dtype=torch.float32, device=device),
        torch.arange(out_nx, dtype=torch.float32, device=device),
        indexing="ij",
    )
    stack = torch.stack(tuple(sources), dim=0)  # (C, nz, ny, nx)
    out = _separable_resample_3d(stack, ii + warp_xd, jj + warp_yd, kk + warp_zd, mode)
    return list(out.unbind(0))


# ---------------------------------------------------------------------------
# Interpolation kernels
# ---------------------------------------------------------------------------

# M3(x) = minimum-sidelobe 3-term window (AFNI default, WFUN=0).
_M3_A, _M3_B, _M3_C = 0.4243801, 0.4973406, 0.0782793
# HW(x) = Hamming (minimum-sidelobe 2-term) window (AFNI AFNI_WSINC5_TAPERFUN=H).
_HW_A, _HW_B = 0.53836, 0.46164


@lru_cache(maxsize=1)
def _wsinc5_params() -> tuple[int, float, float, bool]:
    """Read the ``AFNI_WSINC5_*`` env vars once, mirroring AFNI ``setup_wsinc5``.

    AFNI reads these at first wsinc5 use (``mri_genalign_util.c:setup_wsinc5``);
    hardcoding the defaults silently diverges from any site that sets them (some
    labs set ``AFNI_WSINC5_RADIUS=9`` globally). We honor the three that keep the
    kernel separable and refuse the one that does not:

      - ``AFNI_WSINC5_RADIUS``   -> IRAD (3..21), tap count = 2*IRAD
      - ``AFNI_WSINC5_TAPERCUT`` -> WCUT (0..0.8), start of the taper region
      - ``AFNI_WSINC5_TAPERFUN`` -> 'H' selects Hamming, else min-sidelobe 3-term
      - ``AFNI_WSINC5_SPHERICAL``-> 'Y' selects AFNI's non-separable spherical
        kernel, which this separable resampler cannot reproduce -> raise rather
        than silently use the cubical/tensor kernel.

    Cached (like AFNI's one-time setup); tests that toggle env must call
    ``_wsinc5_params.cache_clear()``.

    Returns:
        (irad, wrad, wcut, wfun_hamming).
    """
    irad = 5
    eee = os.environ.get("AFNI_WSINC5_RADIUS")
    if eee is not None:
        try:
            val = float(eee)
        except ValueError:
            val = -1.0
        if 3.0 <= val <= 21.9:
            irad = int(val)

    wcut = 0.0
    eee = os.environ.get("AFNI_WSINC5_TAPERCUT")
    if eee is not None:
        try:
            val = float(eee)
        except ValueError:
            val = -1.0
        if 0.0 <= val <= 0.8:
            wcut = val

    eee = os.environ.get("AFNI_WSINC5_TAPERFUN")
    wfun_hamming = eee is not None and eee[:1].upper() == "H"

    eee = os.environ.get("AFNI_WSINC5_SPHERICAL")
    if eee is not None and eee[:1].upper() == "Y":
        raise NotImplementedError(
            "AFNI_WSINC5_SPHERICAL=Y selects AFNI's non-separable spherical "
            "wsinc5 kernel, which this separable resampler cannot reproduce. "
            "Unset it for the default cubical/tensor kernel, or pick -interp "
            "cubic/quintic/heptic."
        )

    wrad = 0.001 + float(irad)
    return irad, wrad, wcut, wfun_hamming


def _wsinc5_kernel(fx: Tensor) -> Tensor:
    """Build AFNI-faithful wsinc5 weights (floor convention).

    Mirrors ``GA_interp_wsinc5p`` in AFNI's ``mri_genalign_util.c`` (the cubical
    tensor-product branch). For ``fx`` in [0, 1) the taps are at integer offsets
    ``-(IRAD-1) .. +IRAD`` from ``floor(x)`` (10 taps at the IRAD=5 default) and
    each weight is::

        d  = |fx - offset|
        xw = d / WRAD
        w  = sinc(d) * (xw > WCUT ? win((xw - WCUT) / (1 - WCUT)) : 1)

    with ``sinc(t) = sin(pi t)/(pi t)`` and ``win`` the min-sidelobe 3-term (or
    Hamming) window. IRAD/WCUT/window follow the ``AFNI_WSINC5_*`` env vars via
    :func:`_wsinc5_params`; the defaults (IRAD=5, WCUT=0, M3) reduce to the
    unconditional-window form and match ``3dNwarpApply -interp wsinc5``.

    Args:
        fx: (N,) fractional position in [0, 1) past the floor integer.

    Returns:
        (N, 2*IRAD) kernel weights, normalized to sum to 1.
    """
    irad, wrad, wcut, wfun_hamming = _wsinc5_params()
    offsets = torch.arange(
        -(irad - 1), irad + 1, dtype=fx.dtype, device=fx.device
    )  # -(IRAD-1) .. +IRAD  (2*IRAD taps)
    d = (fx[:, None] - offsets[None, :]).abs()  # (N, ntaps)

    # Guard the divisor as well as the result: torch.where still backprops the
    # unselected branch, so a bare 0/0 hands autograd a NaN gradient at every
    # tap that lands exactly on a sample (i.e. every point of an identity warp).
    at_node = d < 1e-7
    pid = torch.pi * torch.where(at_node, torch.ones_like(d), d)
    sinc = torch.where(at_node, torch.ones_like(d), torch.sin(pid) / pid)

    # Window argument is remapped so win=1 at the taper start (xw==WCUT) and
    # tapers to the edge (xw==1); no window inside the cut region. WCUT=0 ->
    # arg == xw and the guard is always true (xw>0), i.e. unconditional window.
    xw = d / wrad
    arg = torch.pi * (xw - wcut) / (1.0 - wcut)
    if wfun_hamming:
        win = _HW_A + _HW_B * torch.cos(arg)
    else:
        win = _M3_A + _M3_B * torch.cos(arg) + _M3_C * torch.cos(2.0 * arg)
    win = torch.where(xw > wcut, win, torch.ones_like(win))

    w = sinc * win  # (N, ntaps)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-10)
    return w


# ---------------------------------------------------------------------------
# Lagrange polynomial interpolation kernels (matching AFNI exactly)
#
# These are all based on floor(x) convention: fx in [0, 1) is the fraction
# past the floor integer.  Taps are at offsets from the floor position.
# ---------------------------------------------------------------------------

_CUBIC_HALF = 2  # 4 taps: floor-1 .. floor+2
_QUINTIC_HALF = 3  # 6 taps: floor-2 .. floor+3
_HEPTIC_HALF = 4  # 8 taps: floor-3 .. floor+4


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
    "wsinc5": (_wsinc5_kernel, 5, True),  # floor convention, 10 taps (-4..+5)
    "cubic": (_cubic_kernel, 2, True),  # floor convention
    "quintic": (_quintic_kernel, 3, True),  # floor convention
    "heptic": (_heptic_kernel, 4, True),  # floor convention
}


def _kernel_half_width(name: str) -> int:
    """Half-width (taps = 2*H) for a kernel. wsinc5 follows AFNI_WSINC5_RADIUS,
    so the resampler must read it here rather than the static registry value."""
    if name == "wsinc5":
        return _wsinc5_params()[0]
    return _KERNELS[name][1]


def _resample_chunk_size(n_points: int, ntaps: int, device: torch.device) -> int:
    """Voxels per chunk for the separable resampler.

    The three-pass contraction materializes only an ``ntaps**2`` slab per output
    sample (one z-plane at a time -- 100 floats for wsinc5, not the 1000 of a
    full ``ntaps**3`` neighborhood), so peak memory scales with ``ntaps**2``.
    Size the chunk from the memory module's available-memory estimate (which
    already applies the GPU 0.5 safety factor), per the [[Memory module]] rule.
    """
    from ..memory import get_available_memory

    # Per-point peak of _gather_contract, in float32-equivalent (4-byte) units.
    # The dominant cost is *not* the float slab: advanced indexing
    # ``source[zi[:,t][:,None,None], yi[:,:,None], xi[:,None,:]]`` makes PyTorch
    # materialize three broadcasted (c, ntaps, ntaps) INT64 index grids (8 bytes
    # each) before the gather -- 6*ntaps**2 in 4-byte units, three times the whole
    # float slab. Omitting them (the old 2*ntaps**2 model) undersized the chunk ~3x
    # and OOM'd on large padded grids. Terms:
    #   6*ntaps**2 : 3 broadcasted int64 index grids (the real peak)
    #   2*ntaps**2 : the gathered slab + its (slab * wx) product
    #   6*ntaps    : xi/yi/zi int64 tap tables (3 * ntaps * 2)
    #   3*ntaps    : wx/wy/wz float tap tables
    #   8          : per-point coord/base scratch
    bytes_per_point = (8 * ntaps**2 + 9 * ntaps + 8) * 4
    # empty_cache=False: this runs once per resample (hundreds of times in a 4-D
    # warp), and empty_cache would force a device sync + allocator churn on each.
    # The resulting reserved-inclusive reading only underestimates free memory, so
    # chunks stay safe.
    avail = get_available_memory(device, empty_cache=False)
    chunk = int(avail // max(1, bytes_per_point))
    return max(1, min(n_points, chunk))


def _axis_weights(kernel_fn, frac: Tensor) -> Tensor:
    """Per-axis kernel weights, reusing the kernel eval across repeated offsets.

    S1: a regular output grid commensurate with the warp's grid produces only a
    handful of distinct fractional offsets per axis (one, for an aligned grid),
    so evaluating the kernel on the *unique* fracs and gathering is far cheaper
    than the per-voxel eval AFNI does. A cheap probe (unique count on a 256-point
    prefix) detects low repetition; on arbitrary per-voxel displacements -- the
    common final-apply case -- it shows ~all-distinct and we skip straight to the
    direct eval, so the unique sort never runs on the hot path. The unique match
    is exact (no quantization), so the result is bit-identical to a direct eval.
    """
    # torch.unique has no derivative (_unique2), so the dedup path would break the
    # autograd optimizers (allineate/qwarp refine backward through the resample).
    c = frac.numel()
    if c >= 64 and not frac.requires_grad:
        probe = frac[: min(256, c)]
        if torch.unique(probe).numel() * 4 < probe.numel():
            uniq, inv = torch.unique(frac, return_inverse=True)
            return kernel_fn(uniq)[inv]
    return kernel_fn(frac)


def _gather_contract(
    source: Tensor,
    xi: Tensor,
    yi: Tensor,
    zi: Tensor,
    wx: Tensor,
    wy: Tensor,
    wz: Tensor,
) -> Tensor:
    """Three-pass separable gather+contract for one chunk (AFNI wsinc5p order).

    For each z-tap, gather that ``(c, ty, tx)`` plane, contract X then Y into a
    scalar, accumulate over Z. ``xi/yi/zi`` are clamped ``(c, ntaps)`` index
    tables; ``wx/wy/wz`` the ``(c, ntaps)`` weights. Pulled out as a standalone
    function so :func:`_get_gather_contract` can hand it to ``torch.compile`` --
    inductor fuses the gather with the multiply-reduce and never materializes the
    ntaps planes, ~10x over eager on both CPU and CUDA (measured). Eager and
    compiled share this one body.

    ``source`` may carry a leading channel dim ``(C, nz, ny, nx)``: co-registered
    channels sampled at the *same* coords (e.g. mag + real + imag for phase warps)
    share the index tables and kernel weights, so only the gather+contract is
    repeated per channel. Output is ``(C, c)`` then, else ``(c,)``.
    """
    ntaps = xi.shape[1]
    batched = source.dim() == 4
    if batched:
        # (C, c) accumulator; the channel axis broadcasts through the contraction.
        acc = torch.zeros((source.shape[0], xi.shape[0]), dtype=source.dtype, device=source.device)
        for t in range(ntaps):
            plane = source[
                :, zi[:, t][:, None, None], yi[:, :, None], xi[:, None, :]
            ]  # (C,c,ty,tx)
            sx = (plane * wx[:, None, :]).sum(dim=3)  # (C, c, ty)  contract X
            sxy = (sx * wy).sum(dim=2)  # (C, c)          contract Y
            acc = acc + sxy * wz[:, t]  # accumulate Z (broadcast over C)
        return acc
    acc = torch.zeros(xi.shape[0], dtype=source.dtype, device=source.device)
    for t in range(ntaps):
        plane = source[zi[:, t][:, None, None], yi[:, :, None], xi[:, None, :]]
        sx = (plane * wx[:, None, :]).sum(dim=2)  # (c, ty)  contract X
        sxy = (sx * wy).sum(dim=1)  # (c,)     contract Y
        acc = acc + sxy * wz[:, t]  # accumulate Z
    return acc


# The three-pass eager gather materializes a (C, chunk, ntaps, ntaps) plane ntaps
# times over; torch.compile avoids that -- inductor fuses the gather with the
# multiply-reduce and never materializes the planes: ~10x on BOTH CPU and CUDA
# (measured wsinc5 gather 461->43 ms on an RTX 5070 Ti). A hand-written Triton
# kernel was prototyped and only matched inductor (1.02-1.07x) -- inductor already
# emits near-optimal Triton here -- so it was not worth the CUDA-only complexity.
#
# When to switch from eager to compiled is a payback question, and the two sides
# scale differently: the eager cost grows with the data, while the warmup is a
# near-constant property of the machine and torch build. It is NOT amortized by
# inductor's FX graph cache -- that cache elides Triton codegen, but dynamo
# tracing, fake-tensor propagation, guard construction and even hashing the cache
# key are redone in every process (measured 2.1s against a fully warm 2.9 GB
# cache, of which the cache load was 0.07s). A tool run as N short-lived CLI
# invocations therefore pays it N times, which is how ffs_moco lost ~1.4s/run in
# the benchmark while looking fine under -batch.
#
# So the gate is time, not call counts or voxel counts: accumulate eager seconds
# and compile once they reach what a warmup actually costs *here*, measured on the
# first compile and remembered across runs. Worst case (workload ends right after
# the switch) is ~2x the warmup; best case is the full ~10x on everything after
# it. Disable entirely with FFS_NWARP_NO_COMPILE=1.
#
# dynamic=True on BOTH devices, because the batch dim is a *content-dependent*
# count, not a property of the data: _separable_resample_3d hands us only the
# heavy voxels (in_bounds & ~tiny), and how many survive shifts with the
# transform. Registering a moving series therefore mints a graph per volume.
# CPU was on static shapes for the fusion, but the churn dwarfs it -- across 12
# transforms with a realistically drifting M (480,807 down to 459,000, the range
# one 300-volume run actually produces): eager 151 ms/call, dynamic=False 948
# ms/call, dynamic=True 20 ms/call. Static shapes blow dynamo's recompile cap of
# 8 and fall back to eager for the rest of the process, having paid for eight
# compiles first -- 6x SLOWER than never compiling at all.
_COMPILE_COST_BOOTSTRAP_S = 2.0  # only until the first real measurement lands
_eager_seconds = {"cpu": 0.0, "cuda": 0.0}
_pending_cuda_timings: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
_compiled_gather_contract: dict[str, object] = {}
_compile_cost_cache: dict[str, float] = {}
_compile_pending_measure: set[str] = set()
_no_compile_depth = 0


# Bumped whenever the compile itself changes shape (flags, or what gets traced):
# a cost measured under the old regime does not predict the new one. v2 = the
# move to dynamic=True, whose recorded costs were inflated by recompile churn.
_COMPILE_COST_SCHEMA = "v2"


def _compile_cost_key(dt: str, what: str = "gather") -> str:
    """Warmup cost is per device type and per torch build, so a torch upgrade
    recalibrates rather than inheriting a stale number.

    ``what`` namespaces the entry per compiled function: the gather and the 1D
    shear pass (shear._get_shear_interp) have genuinely different warmups -- the
    shear's cold compile measured 4x the gather's -- so sharing one key makes
    whichever compiled last mis-gate the other.
    """
    return f"{what}:{dt}:{torch.__version__}:{_COMPILE_COST_SCHEMA}"


def _compile_cost_path() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(root) / "fastfuncstuff" / "gather_compile_cost.json"


def _measured_compile_cost(dt: str, what: str = "gather", bootstrap: float | None = None) -> float:
    """Seconds a torch.compile warmup costs on this machine, from the last time we
    paid one. Falls back to a bootstrap prior that the first real measurement
    replaces -- an unfamiliar machine mis-compiles at most once.

    Note the measurement is whatever the *last* compile paid, and a cold inductor
    cache costs several times a warm one (gather ~5s cold vs ~2s warm; the shear
    pass ~20s vs ~3s). So a first-ever compile records a pessimistic number and
    the following run gates conservatively, until a compile on a warm cache
    overwrites it. That self-corrects as long as the workload still reaches the
    higher bar -- pick ``bootstrap`` above the warm cost so it can.
    """
    key = _compile_cost_key(dt, what)
    if key in _compile_cost_cache:
        return _compile_cost_cache[key]
    cost = _COMPILE_COST_BOOTSTRAP_S if bootstrap is None else bootstrap
    try:
        with open(_compile_cost_path()) as f:
            stored = json.load(f).get(key)
        if isinstance(stored, int | float) and stored > 0:
            cost = float(stored)
    except (OSError, ValueError):
        pass  # no calibration yet, or an unreadable cache -- the prior is fine
    _compile_cost_cache[key] = cost
    return cost


def _record_compile_cost(dt: str, seconds: float, what: str = "gather") -> None:
    """Persist what the warmup just cost. Best-effort: a read-only or racing cache
    only means the next process uses the prior again."""
    key = _compile_cost_key(dt, what)
    _compile_cost_cache[key] = seconds
    path = _compile_cost_path()
    try:
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        data[key] = seconds
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)  # atomic, so a concurrent reader never sees a partial file
    except OSError:
        pass


@contextmanager
def no_gather_compile():
    """Keep the resample gather eager for a block of one-shot work.

    Some callers know statically that their resamples run once per process --
    ffs_moco's derivative images, for instance. No accumulated-time heuristic can
    infer that from the inside (six full-volume resamples look exactly like the
    start of a long loop), so the call site declares it. Work inside the block is
    also kept out of the eager budget: it must not push a *later* loop over the
    line on its own.
    """
    global _no_compile_depth
    _no_compile_depth += 1
    try:
        yield
    finally:
        _no_compile_depth -= 1


def _already_compiling() -> bool:
    """True while dynamo is tracing us (i.e. a caller wrapped the whole resample
    in ``torch.compile`` — e.g. ffs_moco's ``torch.compile(resample_affine_fast)``).
    dynamo constant-folds this to True during trace and False otherwise."""
    is_comp = getattr(torch.compiler, "is_compiling", None)
    try:
        return bool(is_comp()) if is_comp is not None else False
    except Exception:
        return False


def _drain_cuda_timings() -> None:
    """Fold finished CUDA event pairs into the eager budget.

    ``query()`` rather than ``synchronize()``: the whole point of the budget is to
    be free, and a pair that isn't ready yet simply counts on the next call.
    """
    if not _pending_cuda_timings:
        return
    unfinished = []
    for start, end in _pending_cuda_timings:
        if end.query():
            _eager_seconds["cuda"] += start.elapsed_time(end) / 1000.0
        else:
            unfinished.append((start, end))
    _pending_cuda_timings[:] = unfinished


def _budget_accounted(device: torch.device) -> bool:
    """Whether this call's runtime should feed the eager budget at all."""
    return (
        device.type in ("cpu", "cuda")
        and not _no_compile_depth
        and not _already_compiling()
        and device.type not in _compiled_gather_contract
        and os.environ.get("FFS_NWARP_NO_COMPILE") != "1"
    )


def _get_gather_contract(device: torch.device):
    """Return the compiled _gather_contract once eager time has covered what a
    warmup costs on this machine, else the eager function.

    If a caller already has us inside a ``torch.compile`` trace, return the EAGER
    function and skip the budget entirely: dynamo inlines it into the caller's
    graph, and — critically — we never touch the module globals while traced. That
    mutation is invisible to eager use but becomes a dynamo guard when traced,
    which recompiles on every call and melts down (the ffs_moco `-cost` resample
    path hit exactly this)."""
    dt = device.type
    if (
        _already_compiling()
        or dt not in ("cpu", "cuda")
        or _no_compile_depth
        or os.environ.get("FFS_NWARP_NO_COMPILE") == "1"
    ):
        return _gather_contract
    existing = _compiled_gather_contract.get(dt)
    if existing is not None:
        return existing
    if dt == "cuda":
        _drain_cuda_timings()
    if _eager_seconds[dt] < _measured_compile_cost(dt):
        return _gather_contract
    try:
        compiled = torch.compile(_gather_contract, dynamic=True)
    except Exception:
        compiled = _gather_contract  # compile unavailable
    _compiled_gather_contract[dt] = compiled
    if compiled is not _gather_contract:
        _compile_pending_measure.add(dt)  # first call through it pays the warmup
    return compiled


def _separable_resample_3d(
    source: Tensor,
    x_coords: Tensor,
    y_coords: Tensor,
    z_coords: Tensor,
    kernel_name: str,
) -> Tensor:
    """Resample a 3D volume using separable 1D kernel interpolation.

    Applied separably in three passes per AFNI's ``GA_interp_wsinc5p``: for each
    z-tap plane, contract along X then Y into a scalar, then accumulate across Z
    weighted by the z kernel. This keeps only an ``(ntaps, ntaps)`` slab live at
    a time instead of the full ``ntaps**3`` neighborhood -- ~10x less peak memory
    for wsinc5, so chunks can be ~10x larger. The result is mathematically
    identical to the one-shot ``einsum("ctuv,ct,cu,cv->c")`` contraction.

    Args:
        source: (nz, ny, nx), or (C, nz, ny, nx) to resample several co-registered
            channels through the *same* coordinates in one pass -- the OOB mask,
            grid-node fast path, index tables and kernel weights are all built once
            and only the gather+contract repeats per channel. Output gains the same
            leading ``C`` axis then.
        x_coords, y_coords, z_coords: output sample locations in index space.
        kernel_name: one of "wsinc5", "cubic", "quintic", "heptic".

    Returns:
        Resampled volume with same shape as coordinate arrays (a leading ``C`` axis
        is prepended when ``source`` is channel-batched).
    """
    needs_grad = torch.is_grad_enabled() and (
        source.requires_grad
        or x_coords.requires_grad
        or y_coords.requires_grad
        or z_coords.requires_grad
    )
    default_wsinc = kernel_name != "wsinc5" or _wsinc5_params() == (5, 5.001, 0.0, False)
    if (
        separable_resample_3d_triton is not None
        and source.device.type == "cuda"
        and source.dtype == torch.float32
        and source.dim() == 3
        and x_coords.numel() >= 65536
        and not needs_grad
        and default_wsinc
        and os.environ.get("FFS_INTERP_NO_TRITON") != "1"
    ):
        return separable_resample_3d_triton(source, x_coords, y_coords, z_coords, kernel_name)

    kernel_fn, _, use_floor = _KERNELS[kernel_name]
    H = _kernel_half_width(kernel_name)  # wsinc5 honors AFNI_WSINC5_RADIUS
    batched = source.dim() == 4
    n_ch = source.shape[0] if batched else 1
    nz, ny, nx = source.shape[-3:]
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

    result = torch.zeros((n_ch, N) if batched else (N,), dtype=dtype, device=device)

    # Per-point bases (cheap elementwise) for the OOB and grid-node masks below.
    if use_floor:
        xb_all, yb_all, zb_all = x_flat.floor(), y_flat.floor(), z_flat.floor()
    else:
        xb_all, yb_all, zb_all = x_flat.round(), y_flat.round(), z_flat.round()

    # S2 (AFNI :630-632): out-of-bounds centers contribute nothing (outval=0).
    # result starts at 0, so we simply never compute them. Bounds match the old
    # end-zeroing window [-0.5, n-0.5], so in-bounds output is unchanged -- this
    # just skips the kernel for voxels that would have been zeroed (often
    # 30-50% of the grid for skull-stripped / cropped-FOV warps).
    in_bounds = (
        (x_flat >= -0.5)
        & (x_flat <= nx - 0.5)
        & (y_flat >= -0.5)
        & (y_flat <= ny - 0.5)
        & (z_flat >= -0.5)
        & (z_flat <= nz - 0.5)
    )

    # S3 (AFNI ISTINY, :638-641): a center sitting on a grid node (fractional
    # part < 1e-4 on every axis) interpolates to that node's value for every
    # kernel here, so take it directly and keep it out of the heavy gather.
    # The shortcut is a plain lookup, so it carries no dependence on the
    # coordinates -- fine for a forward resample, fatal when the caller is
    # differentiating the cost w.r.t. them (allineate's Adam refine). An exact
    # identity start puts *every* point on a node, which made the whole cost
    # grad-free; take the heavy path instead whenever the coords carry grad.
    differentiable = torch.is_grad_enabled() and (
        x_flat.requires_grad or y_flat.requires_grad or z_flat.requires_grad
    )
    tiny = (
        in_bounds
        & ((x_flat - xb_all).abs() < 1e-4)
        & ((y_flat - yb_all).abs() < 1e-4)
        & ((z_flat - zb_all).abs() < 1e-4)
    )
    if differentiable:
        tiny = torch.zeros_like(in_bounds)
    if bool(tiny.any()):
        ti = tiny.nonzero(as_tuple=True)[0]
        node = source[
            ...,
            zb_all[ti].long().clamp(0, nz - 1),
            yb_all[ti].long().clamp(0, ny - 1),
            xb_all[ti].long().clamp(0, nx - 1),
        ]  # (C, n_tiny) when batched, else (n_tiny,)
        result[..., ti] = node

    heavy = in_bounds & ~tiny
    idx = heavy.nonzero(as_tuple=True)[0]
    M = idx.numel()
    if M == 0:
        return result.reshape((n_ch, *out_shape) if batched else out_shape)
    xin, yin, zin = x_flat[idx], y_flat[idx], z_flat[idx]

    # Chunk over the surviving voxels: the gather materializes a
    # (C, chunk, ntaps, ntaps) slab per z-tap, so chunking keeps peak memory
    # bounded (and makes the kernel affordable on MPS/CPU and auto-padded grids).
    # The channel count scales the slab, so shrink the chunk to match.
    chunk = max(1, _resample_chunk_size(M, ntaps, device) // n_ch)
    gather_contract = _get_gather_contract(device)  # compiled once eager time pays for it
    measuring_warmup = device.type in _compile_pending_measure
    accounted = _budget_accounted(device)
    # Wall clock for the warmup (it is host-side work, which CUDA events can't
    # see); CUDA events for the eager budget (the work is async, so wall clock
    # would time the launches, not the kernels).
    t_wall = time.perf_counter() if (measuring_warmup or accounted) else 0.0
    ev_start = ev_end = None
    if accounted and device.type == "cuda":
        ev_start, ev_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        ev_start.record()
    for s in range(0, M, chunk):
        e = min(M, s + chunk)
        xf, yf, zf = xin[s:e], yin[s:e], zin[s:e]

        if use_floor:
            xb, yb, zb = xf.floor(), yf.floor(), zf.floor()
        else:
            xb, yb, zb = xf.round(), yf.round(), zf.round()

        wx = _axis_weights(kernel_fn, xf - xb)  # (c, ntaps)
        wy = _axis_weights(kernel_fn, yf - yb)
        wz = _axis_weights(kernel_fn, zf - zb)

        # Clamped (c, ntaps) index tables; the three-pass gather+contract is in
        # _gather_contract so the CPU path can run it under torch.compile.
        xi = (xb.long()[:, None] + offsets[None, :]).clamp(0, nx - 1)  # (c, ntaps)
        yi = (yb.long()[:, None] + offsets[None, :]).clamp(0, ny - 1)
        zi = (zb.long()[:, None] + offsets[None, :]).clamp(0, nz - 1)

        result[..., idx[s:e]] = gather_contract(source, xi, yi, zi, wx, wy, wz)

    if measuring_warmup:
        # This call ran the compile; almost all of the wall time is it. Remember
        # what it cost so the next process gates on this machine's number.
        _compile_pending_measure.discard(device.type)
        _record_compile_cost(device.type, time.perf_counter() - t_wall)
    elif accounted:
        if ev_end is not None and ev_start is not None:
            ev_end.record()
            _pending_cuda_timings.append((ev_start, ev_end))
        else:
            _eager_seconds[device.type] += time.perf_counter() - t_wall

    return result.reshape((n_ch, *out_shape) if batched else out_shape)


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


# ---------------------------------------------------------------------------
# Source-batched (multi-volume) variants — for batching qwarp over N volumes
# that share one base (and thus one patch lattice). The batch is arranged as
# grid_sample's N dim with (P, V) in the spatial dims, so the N source/warp
# volumes are indexed natively with no per-patch copies. See
# [[Outstanding issues]] "Batched many-to-one nonlinear warp".
# ---------------------------------------------------------------------------


def batched_interp_3ch_multi(
    vol_3ch: Tensor, x: Tensor, y: Tensor, z: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-volume 3-channel interpolation: sample volume n at its own coords.

    Args:
        vol_3ch: (N, 3, nz, ny, nx) - N stacked [xd, yd, zd] fields.
        x, y, z: each (N, P, V) - P patches x V voxels, per volume.

    Returns:
        Three (N, P, V) tensors.
    """
    N, P, V = x.shape
    _, _, nz, ny, nx = vol_3ch.shape
    gx = 2.0 * x / (nx - 1) - 1.0 if nx > 1 else x * 0.0
    gy = 2.0 * y / (ny - 1) - 1.0 if ny > 1 else y * 0.0
    gz = 2.0 * z / (nz - 1) - 1.0 if nz > 1 else z * 0.0
    grid = torch.stack([gx, gy, gz], dim=-1).reshape(N, 1, P, V, 3)
    result = _grid_sample_3d(vol_3ch, grid)  # (N, 3, 1, P, V)
    result = result.reshape(N, 3, P, V)
    return result[:, 0], result[:, 1], result[:, 2]


def batched_trilinear_interpolate_multi(volumes: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """Per-volume trilinear interpolation: sample volume n at its own coords.

    Args:
        volumes: (N, nz, ny, nx) - N source volumes.
        x, y, z: each (N, P, V).

    Returns:
        (N, P, V) interpolated values.
    """
    N, P, V = x.shape
    nz, ny, nx = volumes.shape[1:]
    gx = 2.0 * x / (nx - 1) - 1.0 if nx > 1 else x * 0.0
    gy = 2.0 * y / (ny - 1) - 1.0 if ny > 1 else y * 0.0
    gz = 2.0 * z / (nz - 1) - 1.0 if nz > 1 else z * 0.0
    grid = torch.stack([gx, gy, gz], dim=-1).reshape(N, 1, P, V, 3)
    result = _grid_sample_3d(volumes[:, None], grid)  # (N, 1, 1, P, V)
    return result.reshape(N, P, V)


def batched_compose_and_interpolate_multi(
    source: Tensor,  # (N, nz, ny, nx)
    global_warp_3ch: Tensor,  # (N, 3, nz, ny, nx)
    patch_xd: Tensor,  # (N, P, V)
    patch_yd: Tensor,
    patch_zd: Tensor,
    base_i: Tensor,  # (P, V) shared patch lattice (ibots[:,None] + ii_p[None,:])
    base_j: Tensor,
    base_k: Tensor,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Source-batched analogue of :func:`batched_compose_and_interpolate`.

    N volumes sharing one base (one patch lattice, so ``base_i/j/k`` are (P, V)
    and broadcast over N). For each volume it composes the P patch warps with
    that volume's global warp and samples that volume's source. Equivalent to
    calling the single-volume version once per volume; verified to sub-voxel.

    Returns warped_vals, ah_xd, ah_yd, ah_zd — each (N, P, V).
    """
    xq = (base_i[None] + patch_xd).clamp(0, nx - 1)
    yq = (base_j[None] + patch_yd).clamp(0, ny - 1)
    zq = (base_k[None] + patch_zd).clamp(0, nz - 1)

    axd, ayd, azd = batched_interp_3ch_multi(global_warp_3ch, xq, yq, zq)
    ah_xd = patch_xd + axd
    ah_yd = patch_yd + ayd
    ah_zd = patch_zd + azd

    src_x = (ah_xd + base_i[None]).clamp(-0.499, nx - 0.501)
    src_y = (ah_yd + base_j[None]).clamp(-0.499, ny - 0.501)
    src_z = (ah_zd + base_k[None]).clamp(-0.499, nz - 0.501)

    warped_vals = batched_trilinear_interpolate_multi(source, src_x, src_y, src_z)
    return warped_vals, ah_xd, ah_yd, ah_zd
