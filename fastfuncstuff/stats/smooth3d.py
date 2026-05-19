"""
Batched 3D Gaussian smoothing for [[ffs_perm]]'s ``-vsmooth`` (pseudo-t).

Implements **mask-aware** separable Gaussian smoothing — the standard
``3dBlurInMask``-style correction so out-of-mask zeros don't bleed into
mask voxels.  ``smooth_var_per_perm`` is the public entry point: takes a
``[P, V_in_mask]`` variance matrix, scatters to a 4-D ``[P, X, Y, Z]``
volume, smooths each perm independently with a shared kernel, and
gathers back to ``[P, V_in_mask]``.

Performance: separable 1-D convolutions along each axis, ``torch.conv3d``
treats P as the batch dim (channel = 1).  On GPU at 200 k mask voxels
and 1000 perms, ~ 0.5–1 s including scatter/gather; CPU is ~ 10× slower
but still adds only seconds to a full ffs_perm run.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn.functional import conv3d
from torch.nn.functional import pad as fpad

# FWHM ↔ sigma constant: FWHM = sigma · 2√(2 ln 2)
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def fwhm_mm_to_sigma_vox(
    fwhm_mm: float,
    voxel_size_mm: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert isotropic FWHM in mm to per-axis Gaussian σ in voxels."""
    sx = fwhm_mm * FWHM_TO_SIGMA / voxel_size_mm[0]
    sy = fwhm_mm * FWHM_TO_SIGMA / voxel_size_mm[1]
    sz = fwhm_mm * FWHM_TO_SIGMA / voxel_size_mm[2]
    return (sx, sy, sz)


def _gaussian_kernel_1d(
    sigma: float, dtype: torch.dtype, device: torch.device,
) -> torch.Tensor:
    """Truncated 1-D Gaussian kernel, normalised to sum to 1.

    Truncation radius = 4σ (matches scipy.ndimage default ``truncate=4.0``).
    """
    radius = max(1, int(math.ceil(4.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    return k / k.sum()


def _smooth_along_axis(
    x: torch.Tensor,  # [P, 1, X, Y, Z]
    sigma: float,
    axis: int,
) -> torch.Tensor:
    """Convolve a 5-D tensor with a 1-D Gaussian along spatial axis 0/1/2.

    Padding is reflect so the mask-normalised denominator stays well
    behaved at the brain boundary.
    """
    if sigma <= 0:
        return x
    k = _gaussian_kernel_1d(sigma, x.dtype, x.device)
    r = (k.numel() - 1) // 2
    if axis == 0:
        k3 = k.view(1, 1, -1, 1, 1)
        # F.pad pads spatial dims in reverse order: (W_l, W_r, H_l, H_r, D_l, D_r)
        x = fpad(x, (0, 0, 0, 0, r, r), mode="reflect")
    elif axis == 1:
        k3 = k.view(1, 1, 1, -1, 1)
        x = fpad(x, (0, 0, r, r, 0, 0), mode="reflect")
    else:
        k3 = k.view(1, 1, 1, 1, -1)
        x = fpad(x, (r, r, 0, 0, 0, 0), mode="reflect")
    return conv3d(x, k3)


def gaussian3d_batched(
    vol: torch.Tensor,         # [P, X, Y, Z]
    sigma_vox: tuple[float, float, float],
) -> torch.Tensor:
    """Separable 3-D Gaussian, batched over the leading dim."""
    x = vol.unsqueeze(1)  # [P, 1, X, Y, Z]
    for ax, s in enumerate(sigma_vox):
        x = _smooth_along_axis(x, s, ax)
    return x.squeeze(1)


def smooth_var_per_perm(
    var_pv: torch.Tensor,         # [P, V_in_mask], float32
    mask: np.ndarray,             # [X, Y, Z] bool
    sigma_vox: tuple[float, float, float],
    perm_chunk: int = 256,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Mask-aware Gaussian smoothing of a per-perm variance matrix.

    Mask-aware = divide ``smooth(var · 1_mask)`` by ``smooth(1_mask)``
    so out-of-mask zeros don't dilute voxels near the boundary.  This is
    the same correction used by ``3dBlurInMask``.

    Streams in chunks of ``perm_chunk`` perms so peak memory stays
    bounded regardless of P.
    """
    p, v_mask = var_pv.shape
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    nx, ny, nz = mask.shape
    flat_idx = torch.from_numpy(np.flatnonzero(mask.ravel())).to(dev, dtype=torch.int64)
    assert flat_idx.numel() == v_mask, "mask voxel count must equal V_in_mask"
    n_total = nx * ny * nz

    # Pre-compute mask-smooth denominator once (1-channel, broadcast).
    mask_3d = torch.from_numpy(mask.astype(np.float32)).to(dev).unsqueeze(0)
    mask_smooth = gaussian3d_batched(mask_3d, sigma_vox).clamp_min(1e-10)  # [1, X, Y, Z]

    out = torch.empty_like(var_pv)
    for p0 in range(0, p, perm_chunk):
        p1 = min(p0 + perm_chunk, p)
        pc = p1 - p0
        chunk_cpu = var_pv[p0:p1]
        chunk = chunk_cpu.to(dev, non_blocking=True)
        vol_flat = torch.zeros((pc, n_total), dtype=chunk.dtype, device=dev)
        vol_flat[:, flat_idx] = chunk
        vol = vol_flat.view(pc, nx, ny, nz)
        smoothed = gaussian3d_batched(vol, sigma_vox)         # [Pc, X, Y, Z]
        smoothed = smoothed / mask_smooth                      # mask-aware
        gathered = smoothed.view(pc, -1)[:, flat_idx]
        out[p0:p1] = gathered.cpu() if dev.type != "cpu" else gathered
    return out
