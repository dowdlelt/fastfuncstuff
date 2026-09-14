"""Thin 3-D edge maps, for alignment QC overlays and as a general image transform.

The recipe follows what AFNI's ``@djunct_edgy_align_check`` (the edge overlay in
``@SSwarper`` / ``afni_proc.py`` QC) feeds ``@chauffeur_afni``: a small median filter
to knock out speckle, a gradient-magnitude edge detector, then a weak threshold so
only the clearly-not-noise boundaries remain. AFNI's detector is ``3dedge3``
(Monga-Deriche recursive filtering with non-maximum suppression); this uses the
same structure -- Gaussian derivative, then suppression along the gradient -- which
gives the same one-voxel-thin ridges without the recursive filter.

Thinness is the point of the overlay. A raw gradient magnitude is a band several
voxels wide, and a misalignment of one voxel is invisible inside a band that wide.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from fastfuncstuff.memory import get_available_memory

from .cost import _separable_smooth_3d
from .mask import _quantile


def _median7(vol: Tensor) -> Tensor:
    """Median over the 7-voxel face-neighbour cross (``3dMedianFilter -irad 1.01``)."""
    padded = F.pad(vol[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
    c = padded[1:-1, 1:-1, 1:-1]
    stack = torch.stack(
        (
            c,
            padded[:-2, 1:-1, 1:-1],
            padded[2:, 1:-1, 1:-1],
            padded[1:-1, :-2, 1:-1],
            padded[1:-1, 2:, 1:-1],
            padded[1:-1, 1:-1, :-2],
            padded[1:-1, 1:-1, 2:],
        )
    )
    return stack.median(dim=0).values


def _suppress_non_maxima(mag: Tensor, step: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    """Zero every voxel that is not a maximum of ``mag`` along its own gradient.

    ``step`` is the unit gradient direction in index units, (z, y, x). Streamed in
    z-slabs: the sample grid is three full-volume float tensors per neighbour.
    """
    nz, ny, nx = mag.shape
    device = mag.device
    # Grid (3) + two sampled neighbours + mask, per voxel, all float32.
    bytes_per_voxel = 4 * 8
    slab = max(1, min(nz, get_available_memory(device) // max(1, bytes_per_voxel * ny * nx)))
    kk = torch.arange(ny, device=device, dtype=mag.dtype)
    ii = torch.arange(nx, device=device, dtype=mag.dtype)
    yy, xx = torch.meshgrid(kk, ii, indexing="ij")
    src = mag[None, None]
    out = torch.zeros_like(mag)
    for z0 in range(0, nz, slab):
        z1 = min(nz, z0 + slab)
        zz = torch.arange(z0, z1, device=device, dtype=mag.dtype)[:, None, None]
        dz, dy, dx = (s[z0:z1] for s in step)
        keep = torch.ones(z1 - z0, ny, nx, dtype=torch.bool, device=device)
        for sign in (1.0, -1.0):
            gx = 2.0 * (xx + sign * dx) / max(nx - 1, 1) - 1.0
            gy = 2.0 * (yy + sign * dy) / max(ny - 1, 1) - 1.0
            gz = 2.0 * (zz + sign * dz) / max(nz - 1, 1) - 1.0
            grid = torch.stack((gx, gy, gz), dim=-1)[None]
            neighbour = F.grid_sample(
                src, grid, mode="bilinear", padding_mode="border", align_corners=True
            )[0, 0]
            # >= on one side and > on the other: a plateau two voxels wide keeps one.
            keep &= mag[z0:z1] >= neighbour if sign > 0 else mag[z0:z1] > neighbour
        out[z0:z1] = torch.where(keep, mag[z0:z1], out[z0:z1])
    return out


def edge_map(
    vol: Tensor,
    *,
    sigma: float = 1.0,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    median: bool = True,
    thin: bool = True,
    threshold: float = 0.1,
    mask: Tensor | None = None,
) -> Tensor:
    """Edge strength of a 3-D volume: non-zero only on retained edge voxels.

    Args:
        vol: (nz, ny, nx) image.
        sigma: Gaussian pre-smoothing in voxels before differentiating. Larger keeps
            only coarser boundaries. 0 differentiates the raw (median-filtered) image.
        spacing: Voxel size (dz, dy, dx) in mm. The gradient is taken per mm so an
            anisotropic grid does not favour edges across its thick axis.
        median: Apply the 7-voxel median first, as AFNI's QC does.
        thin: Non-maximum suppression along the gradient -- one-voxel-wide ridges.
        threshold: Drop edges weaker than this fraction of the 99th percentile of
            edge strength. Relative, so it is independent of the image's units.
        mask: Optional (nz, ny, nx) region; edges outside it are zeroed.

    Returns:
        (nz, ny, nx) float tensor of gradient magnitude on the kept edges, 0 elsewhere.
    """
    if vol.ndim != 3:
        raise ValueError(f"edge_map needs a 3-D volume, got shape {tuple(vol.shape)}")
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    v = vol.float()
    if median:
        v = _median7(v)
    if sigma > 0:
        v = _separable_smooth_3d(v, sigma)

    dz, dy, dx = (float(s) or 1.0 for s in spacing)
    gz, gy, gx = torch.gradient(v, spacing=(dz, dy, dx))
    del v
    mag = torch.sqrt(gz * gz + gy * gy + gx * gx)

    if thin:
        # The suppression walks one voxel along the gradient, which is a direction
        # in index space: a per-mm slope converts to per-voxel by multiplying back.
        iz, iy, ix = gz * dz, gy * dy, gx * dx
        del gz, gy, gx
        norm = torch.sqrt(iz * iz + iy * iy + ix * ix).clamp_min(torch.finfo(mag.dtype).tiny)
        mag = _suppress_non_maxima(mag, (iz / norm, iy / norm, ix / norm))
    else:
        del gz, gy, gx

    if mask is not None:
        mag = mag * (mask > 0)

    nonzero = mag[mag > 0]
    if threshold > 0 and nonzero.numel():
        floor = threshold * _quantile(nonzero, 0.99)
        mag = torch.where(mag >= floor, mag, torch.zeros_like(mag))
    return mag
