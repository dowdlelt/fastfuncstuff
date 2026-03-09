"""Weight image computation for 3dQwarp.

Implements mri_weightize() from 3dQwarp.c - creates a weight image from the
base image by:
  1. Taking absolute values
  2. Zeroing edges (4% fade on each face)
  3. Clipping large values
  4. Gaussian smoothing
  5. Normalizing to [0, 1]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_weight_image(
    base: Tensor,
    edge_fraction: float = 0.04,
    gauss_fwhm: float = 4.5,
) -> Tensor:
    """Create a weight image from the base image.

    Args:
        base: (nz, ny, nx) base image.
        edge_fraction: Fraction of each dimension to zero at edges.
        gauss_fwhm: Gaussian smoothing FWHM in voxels.

    Returns:
        (nz, ny, nx) weight image in [0, 1].
    """
    nz, ny, nx = base.shape
    device = base.device

    w = base.abs().clone()

    # Zero edges
    xfade = max(2, int(edge_fraction * nx + 2))
    yfade = max(2, int(edge_fraction * ny + 2))
    zfade = max(2, int(edge_fraction * nz + 2))

    if 6 * xfade >= nx:
        xfade = (nx - 1) // 6
    if 6 * yfade >= ny:
        yfade = (ny - 1) // 6
    if 6 * zfade >= nz:
        zfade = (nz - 1) // 6

    # Zero z edges
    if zfade > 0 and nz > 1:
        w[:zfade, :, :] = 0.0
        w[-zfade:, :, :] = 0.0

    # Zero x edges
    if xfade > 0:
        w[:, :, :xfade] = 0.0
        w[:, :, -xfade:] = 0.0

    # Zero y edges
    if yfade > 0:
        w[:, :yfade, :] = 0.0
        w[:, -yfade:, :] = 0.0

    # Clip large values
    clip = 3.0 * _clip_level(w)
    w = w.clamp(max=clip)

    # Gaussian smooth
    if gauss_fwhm > 0 and w.sum() > 0:
        sigma = gauss_fwhm / 2.355  # FWHM to sigma
        w = _gaussian_smooth_3d(w, sigma)

    # Clip small values and normalize to [0, 1]
    w_max = w.max()
    if w_max > 0:
        w = w / w_max
    else:
        w = torch.ones_like(w)

    return w


def _clip_level(vol: Tensor, frac: float = 0.5) -> float:
    """Estimate a clip level for the volume (like THD_cliplevel)."""
    v = vol[vol > 0]
    if v.numel() == 0:
        return 1.0
    sorted_v, _ = v.sort()
    idx = int(frac * sorted_v.numel())
    idx = min(idx, sorted_v.numel() - 1)
    return float(sorted_v[idx].item())


def _gaussian_smooth_3d(vol: Tensor, sigma: float) -> Tensor:
    """Apply 3D Gaussian smoothing using separable convolution."""
    if sigma <= 0:
        return vol

    # Kernel radius
    radius = int(3.0 * sigma + 0.5)
    if radius < 1:
        radius = 1

    # 1D Gaussian kernel
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=vol.device)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Apply separable convolution
    v = vol[None, None, :, :, :]  # (1, 1, D, H, W)

    # Z direction
    if vol.shape[0] > 1:
        k = kernel_1d[None, None, :, None, None]  # (1, 1, K, 1, 1)
        v = F.pad(v, (0, 0, 0, 0, radius, radius), mode='replicate')
        v = F.conv3d(v, k)

    # Y direction
    if vol.shape[1] > 1:
        k = kernel_1d[None, None, None, :, None]
        v = F.pad(v, (0, 0, radius, radius, 0, 0), mode='replicate')
        v = F.conv3d(v, k)

    # X direction
    if vol.shape[2] > 1:
        k = kernel_1d[None, None, None, None, :]
        v = F.pad(v, (radius, radius, 0, 0, 0, 0), mode='replicate')
        v = F.conv3d(v, k)

    return v[0, 0]
