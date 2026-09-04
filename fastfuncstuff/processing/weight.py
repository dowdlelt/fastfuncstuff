"""Weight image computation for registration (AFNI mri_weightize).

Builds a weight image from the base by: abs → zero a fade margin on each face →
clip super-large values → (optional) median pre-filter → Gaussian smooth →
(optional) bottom-clip + keep largest cluster + erode → normalise to [0,1].

The median + cluster steps and the histogram clip level (``THD_cliplevel``) are
AFNI's; they are opt-in so the simpler 3dQwarp/moco callers keep their existing
(Gaussian-only) behaviour, while ffs_allineate matches 3dAllineate's weight.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_weight_image(
    base: Tensor,
    edge_fraction: float = 0.04,
    gauss_fwhm: float = 4.5,
    median_radius: float = 0.0,
    clusterize: bool = False,
    hist_cliplevel: bool = False,
) -> Tensor:
    """Create a weight image from the base image.

    Args:
        base: (nz, ny, nx) base image.
        edge_fraction: Fraction of each dimension to zero at edges.
        gauss_fwhm: Gaussian smoothing FWHM in voxels.
        median_radius: If > 0, median pre-filter radius in voxels (AFNI 2.25).
        clusterize: If True, bottom-clip then keep the largest connected
            component and erode (AFNI's cleanup that drops the background).
        hist_cliplevel: If True, use AFNI's histogram ``THD_cliplevel`` for the
            clip thresholds instead of a plain quantile.

    Returns:
        (nz, ny, nx) weight image in [0, 1].
    """
    nz, ny, nx = base.shape

    w = base.abs().clone()

    # Edge fade widths (AFNI -edging: a border band that gets zero weight).
    xfade = max(2, int(edge_fraction * nx + 2))
    yfade = max(2, int(edge_fraction * ny + 2))
    zfade = max(2, int(edge_fraction * nz + 2))

    if 6 * xfade >= nx:
        xfade = (nx - 1) // 6
    if 6 * yfade >= ny:
        yfade = (ny - 1) // 6
    if 6 * zfade >= nz:
        zfade = (nz - 1) // 6

    def _zero_edges(vol: Tensor) -> Tensor:
        if zfade > 0 and nz > 1:
            vol[:zfade, :, :] = 0.0
            vol[-zfade:, :, :] = 0.0
        if xfade > 0:
            vol[:, :, :xfade] = 0.0
            vol[:, :, -xfade:] = 0.0
        if yfade > 0:
            vol[:, :yfade, :] = 0.0
            vol[:, -yfade:, :] = 0.0
        return vol

    # NB: the border is zeroed *after* smoothing (below), not before. The base is
    # a single 3D volume with no motion-driven edge artifacts to suppress (those
    # live in the timeseries), and zeroing before the blur would just let the
    # Gaussian bleed interior weight back into the border anyway. AFNI's -edging
    # guarantees the final weight is zero in the border band.

    # Clip super-large values (squash spikes to reasonability).
    cliplev = _thd_cliplevel(w, 0.5) if hist_cliplevel else _clip_level(w)
    w = w.clamp(max=3.0 * cliplev)

    # Median pre-filter: smashes localised spikes before the Gaussian blur.
    if median_radius > 0:
        w = _median_filter_ball(w, median_radius)

    # Gaussian smooth
    if gauss_fwhm > 0 and w.sum() > 0:
        sigma = gauss_fwhm / 2.355  # FWHM to sigma
        w = _gaussian_smooth_3d(w, sigma)

    # Drop small values + isolated background: keep the largest cluster of
    # supra-threshold voxels (AFNI THD_mask_clust → erode → clust). Without this
    # the smoothed background fills the whole FOV (a "square", not a halo).
    if clusterize and w.sum() > 0:
        from .mask import largest_cluster_6conn

        wmax = float(w.max())
        clip2 = 0.33 * _thd_cliplevel(w, 0.33) if hist_cliplevel else 0.33 * _clip_level(w, 0.33)
        clip = max(0.05 * wmax, clip2)
        mask = w >= clip
        mask = largest_cluster_6conn(mask)
        mask = _erode_6conn(mask)
        mask = largest_cluster_6conn(mask)
        w = w * mask.to(w.dtype)

    # Re-zero the border after smoothing/clustering so it is exactly zero in the
    # output (the blur above spreads interior weight into the faded band).
    w = _zero_edges(w)

    # Normalize to [0, 1]
    w_max = w.max()
    if w_max > 0:
        w = w / w_max
    else:
        w = torch.ones_like(w)

    return w


def _clip_level(vol: Tensor, frac: float = 0.5) -> float:
    """Estimate a clip level for the volume (plain quantile of positive voxels)."""
    v = vol[vol > 0]
    if v.numel() == 0:
        return 1.0
    return float(v.quantile(frac).item())


def _thd_cliplevel(vol: Tensor, mfrac: float = 0.5) -> float:
    """AFNI ``THD_cliplevel``: histogram-iterated background/foreground threshold.

    Finds a cut level equal to ``mfrac`` times the median of the values above the
    cut, iterating to convergence. This separates brain from background far
    better than a fixed quantile, so the bottom-clip below actually drops the
    background instead of leaving it bright.
    """
    if mfrac <= 0.0 or mfrac >= 0.99:
        mfrac = 0.5
    v = vol[vol > 0]
    if v.numel() <= 222:
        return 0.0
    vmax = float(v.max())
    if vmax < 1e-30:
        return 0.0

    nhist = 10000
    sfac = nhist / vmax
    kk = torch.floor(sfac * v + 0.499).long()
    kk = kk[kk <= nhist]
    npos = int(kk.numel())
    if npos <= 222:
        return 0.0
    # .cpu() before float64: MPS has no float64; this is a tiny scalar reduction.
    kk_f = kk.cpu().double()
    dsum = float((kk_f * kk_f).sum())
    hist = torch.bincount(kk, minlength=nhist + 1).cpu()
    h = hist.tolist()

    # Initial cut: include the upper ~65% of positive voxels (above a sqrt floor).
    qq = 0.65 * npos
    ib = int(round(0.5 * math.sqrt(dsum / npos)))
    ib = max(0, min(ib, nhist))
    acc = 0
    ii = nhist
    while ii >= ib and acc < qq:
        acc += h[ii]
        ii -= 1
    ncut = ii

    # Median-adjustment iteration.
    nold = -1
    it = 0
    while it < 66 and ncut != nold:
        npos_above = sum(h[ncut:])
        nhalf = npos_above // 2
        acc = 0
        jj = ncut
        while jj < nhist and acc < nhalf:
            acc += h[jj]
            jj += 1
        nold = ncut
        ncut = int(mfrac * jj)
        it += 1

    return ncut / sfac


def _erode_6conn(mask: Tensor) -> Tensor:
    """One-iteration 6-connectivity erosion (drop the boundary layer)."""
    kernel = torch.zeros(1, 1, 3, 3, 3, device=mask.device, dtype=torch.float32)
    for dz, dy, dx in [(1, 1, 1), (0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]:
        kernel[0, 0, dz, dy, dx] = 1.0
    x = mask.float()[None, None]
    cnt = F.conv3d(x, kernel, padding=1)[0, 0]
    return (cnt >= 7.0) & mask  # survives only if it + all 6 neighbours are set


def _median_filter_ball(vol: Tensor, radius: float) -> Tensor:
    """Median filter over a ball of the given radius (AFNI mri_medianfilter).

    Processed in z-slabs so peak memory stays bounded regardless of volume size.
    """
    nz, ny, nx = vol.shape
    r = int(math.ceil(radius))
    r2 = radius * radius
    offs = [
        (dz, dy, dx)
        for dz in range(-r, r + 1)
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
        if dz * dz + dy * dy + dx * dx <= r2
    ]
    vp = F.pad(vol[None, None], (r, r, r, r, r, r), mode="replicate")[0, 0]

    # Bound the stacked-neighbourhood tensor to ~256 MB.
    bytes_per_z = len(offs) * ny * nx * 4
    zc = max(1, int(256 * 1024 * 1024 / max(1, bytes_per_z)))
    out = torch.empty_like(vol)
    for z0 in range(0, nz, zc):
        z1 = min(z0 + zc, nz)
        stack = [
            vp[z0 + r + dz : z1 + r + dz, r + dy : r + dy + ny, r + dx : r + dx + nx]
            for (dz, dy, dx) in offs
        ]
        out[z0:z1] = torch.stack(stack, dim=0).median(dim=0).values
    return out


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
        v = F.pad(v, (0, 0, 0, 0, radius, radius), mode="replicate")
        v = F.conv3d(v, k)

    # Y direction
    if vol.shape[1] > 1:
        k = kernel_1d[None, None, None, :, None]
        v = F.pad(v, (0, 0, radius, radius, 0, 0), mode="replicate")
        v = F.conv3d(v, k)

    # X direction
    if vol.shape[2] > 1:
        k = kernel_1d[None, None, None, None, :]
        v = F.pad(v, (radius, radius, 0, 0, 0, 0), mode="replicate")
        v = F.conv3d(v, k)

    return v[0, 0]
