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
    edge_before_smoothing: bool = False,
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
        edge_before_smoothing: Zero the face bands before filtering, matching
            3dQwarp's ``mri_weightize`` ordering. The default preserves the
            existing moco/allineate behavior and guarantees exactly zero output
            faces after filtering.

    Returns:
        (nz, ny, nx) weight image in [0, 1].
    """
    w = base.abs().clone()
    fades = _edge_fades(base.shape, edge_fraction)

    if edge_before_smoothing:
        w = _zero_edges(w, fades)

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

    if not edge_before_smoothing:
        # The moco/allineate policy guarantees exactly zero output faces. AFNI
        # qwarp instead zeros before filtering and lets clustering remove any
        # insignificant Gaussian bleed, selected by edge_before_smoothing.
        w = _zero_edges(w, fades)

    # Normalize to [0, 1]
    w_max = w.max()
    if w_max > 0:
        w = w / w_max
    else:
        w = torch.ones_like(w)

    return w


def _edge_fades(shape: tuple[int, ...], edge_fraction: float) -> tuple[int, int, int]:
    """Per-axis (z, y, x) widths of the zero-weight border band (AFNI -edging)."""
    fades = []
    for n in shape[-3:]:
        fade = max(2, int(edge_fraction * n + 2))
        if 6 * fade >= n:
            fade = (n - 1) // 6
        fades.append(fade)
    return fades[0], fades[1], fades[2]


def _zero_edges(vol: Tensor, fades: tuple[int, int, int]) -> Tensor:
    zfade, yfade, xfade = fades
    if zfade > 0 and vol.shape[-3] > 1:
        vol[..., :zfade, :, :] = 0.0
        vol[..., -zfade:, :, :] = 0.0
    if xfade > 0:
        vol[..., :, :, :xfade] = 0.0
        vol[..., :, :, -xfade:] = 0.0
    if yfade > 0:
        vol[..., :, :yfade, :] = 0.0
        vol[..., :, -yfade:, :] = 0.0
    return vol


def add_background_band(
    weight: Tensor,
    base: Tensor,
    radius: int,
    level: float = 1.0,
    edge_fraction: float = 0.04,
) -> Tensor:
    """Give the empty background around a masked base real weight.

    Correlation costs cannot see source tissue pushed onto voxels a weight of
    zero excludes, and a skull-stripped template's weight is zero just outside
    the brain -- so a strong optimizer is free to squeeze anatomy past the
    template edge. Flooring the weight in a ``radius``-voxel band around the
    base's nonzero support at ``level`` times the median in-support weight puts
    those template zeros back into the cost: tissue smeared onto them lowers the
    correlation.

    The support is the base's exact-nonzero voxels, so a base with a noisy
    (nonzero) background has no band and the weight is returned unchanged. The
    AFNI border band stays zero.
    """
    support = base != 0
    if radius <= 0 or not support.any() or support.all():
        return weight
    from .mask import _dilate_6conn_once

    # Alternating 6- and 26-connected steps approximate a ball (an octagonal
    # cross-section) rather than the cube that max-pooling alone would grow.
    grown = support.float()[None, None]
    for step in range(radius):
        if step % 2:
            grown = F.max_pool3d(grown, 3, stride=1, padding=1)
        else:
            grown = _dilate_6conn_once(grown)
    band = (grown[0, 0] > 0.5) & ~support
    floor = level * float(weight[support].median())
    out = torch.where(band, weight.clamp(min=floor), weight)
    return _zero_edges(out, _edge_fades(base.shape, edge_fraction))


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
