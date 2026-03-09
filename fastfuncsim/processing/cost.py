"""Cost functions for image matching.

Implements the INCOR (incomplete correlation) framework from AFNI's
thd_incorrelate.c. The key optimization: the correlation is split into
a fixed part (outside the current patch) and a variable part (inside).
Only the variable part needs recomputation during optimization.

Supported cost functions:
  - Clipped Pearson correlation (default in 3dQwarp)
  - Pure Pearson correlation
  - LPA: Local Pearson Absolute correlation (Gaussian-weighted neighborhoods)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def pearson_correlation(
    base: Tensor, source: Tensor, weight: Tensor | None = None
) -> Tensor:
    """Compute weighted Pearson correlation between two images.

    Args:
        base: (N,) base image values.
        source: (N,) source image values (warped).
        weight: (N,) optional weight values. If None, uniform weights.

    Returns:
        Scalar correlation value (higher = better match).
    """
    if weight is not None:
        w = weight
        wsum = w.sum()
        if wsum <= 0:
            return torch.tensor(0.0, device=base.device)
    else:
        w = torch.ones_like(base)
        wsum = torch.tensor(float(base.numel()), device=base.device)

    bm = (w * base).sum() / wsum
    sm = (w * source).sum() / wsum
    bd = base - bm
    sd = source - sm
    bb = (w * bd * bd).sum()
    ss = (w * sd * sd).sum()
    bs = (w * bd * sd).sum()

    denom = torch.sqrt(bb * ss)
    if denom < 1e-10:
        return torch.tensor(0.0, device=base.device)

    return bs / denom


def clipped_pearson_correlation(
    base: Tensor, source: Tensor, weight: Tensor | None = None,
    base_clip: tuple[float, float] | None = None,
    source_clip: tuple[float, float] | None = None,
) -> Tensor:
    """Compute clipped Pearson correlation."""
    if base_clip is None:
        base_clip = _auto_clip(base, weight)
    if source_clip is None:
        source_clip = _auto_clip(source, weight)

    bc = base.clamp(base_clip[0], base_clip[1])
    sc = source.clamp(source_clip[0], source_clip[1])
    return pearson_correlation(bc, sc, weight)


def _auto_clip(data: Tensor, weight: Tensor | None = None) -> tuple[float, float]:
    """Compute automatic clip range (matching AFNI's INCOR_2Dhist_xyclip)."""
    if weight is not None:
        mask = weight > 0
        d = data[mask]
    else:
        d = data

    if d.numel() < 10:
        return (float(d.min()), float(d.max()))

    sorted_d, _ = d.sort()
    n = sorted_d.numel()
    i_lo = max(0, int(0.01 * n))
    i_hi = min(n - 1, int(0.99 * n))
    return (float(sorted_d[i_lo]), float(sorted_d[i_hi]))


# ---------------------------------------------------------------------------
# LPA: Local Pearson Absolute correlation
# ---------------------------------------------------------------------------

def _gauss_kernel_1d(sigma: float, device: torch.device) -> Tensor:
    """Create a 1D Gaussian kernel."""
    radius = int(3.0 * sigma + 0.5)
    if radius < 1:
        radius = 1
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _separable_smooth_3d(vol: Tensor, sigma: float) -> Tensor:
    """Apply 3D Gaussian smoothing using separable convolution.

    Args:
        vol: (1, 1, D, H, W) or (D, H, W) volume.
        sigma: Gaussian sigma in voxels.

    Returns:
        Smoothed volume, same shape as input.
    """
    squeeze = vol.ndim == 3
    if squeeze:
        vol = vol[None, None]

    kernel = _gauss_kernel_1d(sigma, vol.device)
    radius = kernel.shape[0] // 2

    # Z
    if vol.shape[2] > 1:
        k = kernel[None, None, :, None, None]
        vol = F.pad(vol, (0, 0, 0, 0, radius, radius), mode='replicate')
        vol = F.conv3d(vol, k)
    # Y
    if vol.shape[3] > 1:
        k = kernel[None, None, None, :, None]
        vol = F.pad(vol, (0, 0, radius, radius, 0, 0), mode='replicate')
        vol = F.conv3d(vol, k)
    # X
    if vol.shape[4] > 1:
        k = kernel[None, None, None, None, :]
        vol = F.pad(vol, (radius, radius, 0, 0, 0, 0), mode='replicate')
        vol = F.conv3d(vol, k)

    if squeeze:
        vol = vol[0, 0]
    return vol


def lpa_correlation(
    base: Tensor, source: Tensor, weight: Tensor | None = None,
    sigma: float = 4.0,
) -> Tensor:
    """Compute Local Pearson Absolute (LPA) correlation.

    For each voxel, computes the absolute Pearson correlation in a
    Gaussian-weighted local neighborhood. The overall cost is the
    weighted mean of local absolute correlations.

    This is computed efficiently using separable Gaussian convolutions:
      local_mean_x = smooth(w*x) / smooth(w)
      local_var_x  = smooth(w*x*x) / smooth(w) - local_mean_x^2
      local_cov_xy = smooth(w*x*y) / smooth(w) - local_mean_x * local_mean_y
      local_corr   = local_cov_xy / sqrt(local_var_x * local_var_y)

    Args:
        base: (nz, ny, nx) base image.
        source: (nz, ny, nx) source/warped image.
        weight: (nz, ny, nx) optional weight image.
        sigma: Gaussian neighborhood sigma in voxels.

    Returns:
        Scalar: mean absolute local correlation (higher = better).
    """
    if weight is None:
        w = torch.ones_like(base)
    else:
        w = weight

    x = base
    y = source

    # Smoothed weighted statistics (all as 3D volumes)
    sw = _separable_smooth_3d(w, sigma)
    sw = sw.clamp(min=1e-10)

    swx = _separable_smooth_3d(w * x, sigma)
    swy = _separable_smooth_3d(w * y, sigma)
    swxx = _separable_smooth_3d(w * x * x, sigma)
    swyy = _separable_smooth_3d(w * y * y, sigma)
    swxy = _separable_smooth_3d(w * x * y, sigma)

    # Local means
    mx = swx / sw
    my = swy / sw

    # Local variances and covariance
    vxx = (swxx / sw - mx * mx).clamp(min=1e-10)
    vyy = (swyy / sw - my * my).clamp(min=1e-10)
    vxy = swxy / sw - mx * my

    # Local correlation
    local_corr = vxy / (vxx * vyy).sqrt()

    # Take absolute value (LPA = local pearson absolute)
    local_abs_corr = local_corr.abs()

    # Weighted mean
    if weight is not None:
        result = (weight * local_abs_corr).sum() / weight.sum().clamp(min=1e-10)
    else:
        result = local_abs_corr.mean()

    return result


def lpc_correlation(
    base: Tensor, source: Tensor, weight: Tensor | None = None,
    sigma: float = 4.0,
) -> Tensor:
    """Compute Local Pearson Correlation (LPC) for cross-modality alignment.

    Designed for non-similar contrast (e.g., EPI-to-anat) where local
    correlations are typically negative due to contrast inversion.

    Uses Fisher Z (atanh) + z*|z| weighting matching AFNI's aggregation.
    AFNI's lpc = mean(z*|z|) is minimized (more negative = better).
    We return -mean(z*|z|) so that higher = better in our convention.

    For similar-contrast alignment, use LPA instead.

    Args:
        base: (nz, ny, nx) base image.
        source: (nz, ny, nx) source/warped image.
        weight: (nz, ny, nx) optional weight image.
        sigma: Gaussian neighborhood sigma in voxels.

    Returns:
        Scalar: -mean(z*|z|) (higher = better, matching our convention).
    """
    if weight is None:
        w = torch.ones_like(base)
    else:
        w = weight

    x = base
    y = source

    sw = _separable_smooth_3d(w, sigma)
    sw = sw.clamp(min=1e-10)

    swx = _separable_smooth_3d(w * x, sigma)
    swy = _separable_smooth_3d(w * y, sigma)
    swxx = _separable_smooth_3d(w * x * x, sigma)
    swyy = _separable_smooth_3d(w * y * y, sigma)
    swxy = _separable_smooth_3d(w * x * y, sigma)

    mx = swx / sw
    my = swy / sw

    vxx = (swxx / sw - mx * mx).clamp(min=1e-10)
    vyy = (swyy / sw - my * my).clamp(min=1e-10)
    vxy = swxy / sw - mx * my

    local_corr = (vxy / (vxx * vyy).sqrt()).clamp(-0.99, 0.99)

    # Fisher Z transform + z*|z| weighting (AFNI-style)
    z = torch.atanh(local_corr)
    z_weighted = z * z.abs()

    # AFNI minimizes mean(z*|z|); we negate so higher = better
    if weight is not None:
        lpc_afni = (weight * z_weighted).sum() / weight.sum().clamp(min=1e-10)
    else:
        lpc_afni = z_weighted.mean()

    return -lpc_afni


def lpa_cost_patch(
    base_patch: Tensor, source_patch: Tensor, weight_patch: Tensor,
    sigma: float = 2.5,
) -> Tensor:
    """LPA correlation on a single patch (3D). Returns scalar, differentiable."""
    return lpa_correlation(base_patch, source_patch, weight_patch, sigma=sigma)


# ---------------------------------------------------------------------------
# Batched LPA for GPU-parallel patch processing
# ---------------------------------------------------------------------------

def _batched_separable_smooth_3d(vol: Tensor, kernel: Tensor) -> Tensor:
    """Apply separable 3D Gaussian smoothing to batched volumes.

    Uses groups=B so all patches are convolved in a single kernel call per axis.

    Args:
        vol: (B, 1, D, H, W) batched volumes.
        kernel: 1D Gaussian kernel.

    Returns:
        Smoothed (B, 1, D, H, W).
    """
    B = vol.shape[0]
    radius = kernel.shape[0] // 2

    # Reshape from (B, 1, D, H, W) to (1, B, D, H, W) for grouped conv
    vol = vol.permute(1, 0, 2, 3, 4)  # (1, B, D, H, W)

    # Z axis
    if vol.shape[2] > 1:
        k = kernel[None, None, :, None, None].expand(B, 1, -1, 1, 1)  # (B, 1, K, 1, 1)
        vol = F.pad(vol, (0, 0, 0, 0, radius, radius), mode='replicate')
        vol = F.conv3d(vol, k, groups=B)
    # Y axis
    if vol.shape[3] > 1:
        k = kernel[None, None, None, :, None].expand(B, 1, 1, -1, 1)
        vol = F.pad(vol, (0, 0, radius, radius, 0, 0), mode='replicate')
        vol = F.conv3d(vol, k, groups=B)
    # X axis
    if vol.shape[4] > 1:
        k = kernel[None, None, None, None, :].expand(B, 1, 1, 1, -1)
        vol = F.pad(vol, (radius, radius, 0, 0, 0, 0), mode='replicate')
        vol = F.conv3d(vol, k, groups=B)

    # Back to (B, 1, D, H, W)
    return vol.permute(1, 0, 2, 3, 4)


def batched_lpa_cost(
    base_patches: Tensor,
    source_patches: Tensor,
    weight_patches: Tensor,
    nzh: int, nyh: int, nxh: int,
    sigma: float = 4.0,
) -> Tensor:
    """Compute LPA cost for B patches in parallel. Fully differentiable.

    Implements AFNI-style Local Pearson Absolute correlation using
    Gaussian-smoothed local statistics (continuous approximation of
    AFNI's blok-based approach, much faster on GPU).

    Fisher Z-transform (atanh) and z*|z| weighting match AFNI's aggregation.

    Args:
        base_patches: (B, V) base image patch data.
        source_patches: (B, V) warped source patch data.
        weight_patches: (B, V) weight data.
        nzh, nyh, nxh: Patch dimensions (V = nzh * nyh * nxh).
        sigma: Gaussian neighborhood sigma in voxels.

    Returns:
        (B,) correlation values (higher = better match). Differentiable.
    """
    B = base_patches.shape[0]
    device = base_patches.device

    # Reshape flat patches to 3D: (B, 1, D, H, W)
    x = base_patches.reshape(B, 1, nzh, nyh, nxh)
    y = source_patches.reshape(B, 1, nzh, nyh, nxh)
    w = weight_patches.reshape(B, 1, nzh, nyh, nxh)

    # Build kernel once
    kernel = _gauss_kernel_1d(sigma, device)

    # Smoothed weighted statistics (6 convolutions, each batched)
    sw = _batched_separable_smooth_3d(w, kernel).clamp(min=1e-10)
    swx = _batched_separable_smooth_3d(w * x, kernel)
    swy = _batched_separable_smooth_3d(w * y, kernel)
    swxx = _batched_separable_smooth_3d(w * x * x, kernel)
    swyy = _batched_separable_smooth_3d(w * y * y, kernel)
    swxy = _batched_separable_smooth_3d(w * x * y, kernel)

    # Local means
    mx = swx / sw
    my = swy / sw

    # Local variances and covariance
    vxx = (swxx / sw - mx * mx).clamp(min=1e-10)
    vyy = (swyy / sw - my * my).clamp(min=1e-10)
    vxy = swxy / sw - mx * my

    # Local Pearson correlation, clamped for atanh stability
    local_corr = (vxy / (vxx * vyy).sqrt()).clamp(-0.99, 0.99)

    # Fisher Z-transform (atanh), matching AFNI
    z = torch.atanh(local_corr)

    # AFNI aggregation: weighted mean of z * |z| (emphasizes strong correlations)
    z_weighted = z * z.abs()

    # Average over spatial dims, weighted by w
    # Reshape to (B, V) for reduction
    z_flat = z_weighted.reshape(B, -1)
    w_flat = w.reshape(B, -1)

    # Weighted mean per patch → (B,)
    result = (w_flat * z_flat).sum(dim=1) / w_flat.sum(dim=1).clamp(min=1e-10)

    return result


# ---------------------------------------------------------------------------
# Incremental correlation (original serial version, kept for compatibility)
# ---------------------------------------------------------------------------

class IncrementalCorrelation:
    """Incremental correlation calculator matching AFNI's INCOR framework."""

    def __init__(self, method: str = "pearclp"):
        self.method = method
        self._n_fixed: int = 0
        self._sw_fixed: float = 0.0
        self._swx_fixed: float = 0.0
        self._swy_fixed: float = 0.0
        self._swxx_fixed: float = 0.0
        self._swyy_fixed: float = 0.0
        self._swxy_fixed: float = 0.0
        self._base_clip: tuple[float, float] | None = None
        self._source_clip: tuple[float, float] | None = None

    def set_clips(
        self,
        base_clip: tuple[float, float],
        source_clip: tuple[float, float],
    ) -> None:
        self._base_clip = base_clip
        self._source_clip = source_clip

    def add_fixed(
        self, base: Tensor, source: Tensor, weight: Tensor
    ) -> None:
        mask = weight > 0
        if not mask.any():
            return

        w = weight[mask]
        x = base[mask]
        y = source[mask]

        if self._base_clip is not None:
            x = x.clamp(self._base_clip[0], self._base_clip[1])
        if self._source_clip is not None:
            y = y.clamp(self._source_clip[0], self._source_clip[1])

        self._n_fixed = int(mask.sum().item())
        self._sw_fixed = float(w.sum().item())
        self._swx_fixed = float((w * x).sum().item())
        self._swy_fixed = float((w * y).sum().item())
        self._swxx_fixed = float((w * x * x).sum().item())
        self._swyy_fixed = float((w * y * y).sum().item())
        self._swxy_fixed = float((w * x * y).sum().item())

    def evaluate(
        self, base_patch: Tensor, source_patch: Tensor, weight_patch: Tensor
    ) -> float:
        mask = weight_patch > 0
        if not mask.any():
            return 0.0

        w = weight_patch[mask]
        x = base_patch[mask]
        y = source_patch[mask]

        if self._base_clip is not None:
            x = x.clamp(self._base_clip[0], self._base_clip[1])
        if self._source_clip is not None:
            y = y.clamp(self._source_clip[0], self._source_clip[1])

        sw = self._sw_fixed + float(w.sum().item())
        if sw <= 0:
            return 0.0

        swx = self._swx_fixed + float((w * x).sum().item())
        swy = self._swy_fixed + float((w * y).sum().item())
        swxx = self._swxx_fixed + float((w * x * x).sum().item())
        swyy = self._swyy_fixed + float((w * y * y).sum().item())
        swxy = self._swxy_fixed + float((w * x * y).sum().item())

        xbar = swx / sw
        ybar = swy / sw
        vxx = swxx / sw - xbar * xbar
        vyy = swyy / sw - ybar * ybar
        vxy = swxy / sw - xbar * ybar

        denom = vxx * vyy
        if denom <= 0:
            return 0.0

        return vxy / (denom ** 0.5)


# ---------------------------------------------------------------------------
# Batched GPU-native correlation for parallel patch processing
# ---------------------------------------------------------------------------

class BatchedIncrementalCorrelation:
    """Batched incremental correlation for B patches simultaneously.

    All computations stay on GPU as tensors (no .item() calls).
    The result is differentiable for autograd.
    """

    def __init__(
        self,
        method: str = "pearclp",
        base_clip: tuple[float, float] | None = None,
        source_clip: tuple[float, float] | None = None,
    ):
        self.method = method
        self.base_clip = base_clip
        self.source_clip = source_clip

        self._sw_fixed: Tensor | None = None
        self._swx_fixed: Tensor | None = None
        self._swy_fixed: Tensor | None = None
        self._swxx_fixed: Tensor | None = None
        self._swyy_fixed: Tensor | None = None
        self._swxy_fixed: Tensor | None = None

    def precompute_fixed_parts(
        self,
        base: Tensor,
        warped_source: Tensor,
        weight: Tensor,
        patch_slices: list[tuple[int, int, int, int, int, int]],
    ) -> None:
        """Precompute fixed (outside-patch) statistics for all B patches.

        Uses the efficient strategy: compute global sums once, then subtract
        each patch's contribution (avoids B full-image passes).

        Args:
            base, warped_source, weight: (nz, ny, nx) full images.
            patch_slices: List of (ibot, itop, jbot, jtop, kbot, ktop) for B patches.
        """
        B = len(patch_slices)
        device = base.device

        b = base.reshape(-1)
        s = warped_source.reshape(-1)
        w = weight.reshape(-1)

        if self.base_clip:
            b = b.clamp(self.base_clip[0], self.base_clip[1])
        if self.source_clip:
            s = s.clamp(self.source_clip[0], self.source_clip[1])

        mask = w > 0
        wm = w * mask.float()

        # Global sums (once)
        sw_g = wm.sum()
        swx_g = (wm * b).sum()
        swy_g = (wm * s).sum()
        swxx_g = (wm * b * b).sum()
        swyy_g = (wm * s * s).sum()
        swxy_g = (wm * b * s).sum()

        # Subtract each patch's contribution (vectorized: gather all patches at once)
        # Stack patch data as (B, V) tensors
        bp_all = torch.stack([
            base[kbot:ktop+1, jbot:jtop+1, ibot:itop+1].reshape(-1)
            for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
        ])
        sp_all = torch.stack([
            warped_source[kbot:ktop+1, jbot:jtop+1, ibot:itop+1].reshape(-1)
            for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
        ])
        wp_all = torch.stack([
            weight[kbot:ktop+1, jbot:jtop+1, ibot:itop+1].reshape(-1)
            for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
        ])

        if self.base_clip:
            bp_all = bp_all.clamp(self.base_clip[0], self.base_clip[1])
        if self.source_clip:
            sp_all = sp_all.clamp(self.source_clip[0], self.source_clip[1])

        wpm_all = wp_all * (wp_all > 0).float()

        # (B,) sums over voxel dim
        self._sw_fixed = sw_g - wpm_all.sum(dim=1)
        self._swx_fixed = swx_g - (wpm_all * bp_all).sum(dim=1)
        self._swy_fixed = swy_g - (wpm_all * sp_all).sum(dim=1)
        self._swxx_fixed = swxx_g - (wpm_all * bp_all * bp_all).sum(dim=1)
        self._swyy_fixed = swyy_g - (wpm_all * sp_all * sp_all).sum(dim=1)
        self._swxy_fixed = swxy_g - (wpm_all * bp_all * sp_all).sum(dim=1)

    def evaluate(
        self,
        base_patches: Tensor,
        source_patches: Tensor,
        weight_patches: Tensor,
    ) -> Tensor:
        """Compute correlation for B patches. Returns (B,) tensor, differentiable.

        This is the HOT PATH that runs every optimizer iteration.
        No .item() calls, no Python loops, pure tensor ops.
        """
        if self.base_clip:
            base_patches = base_patches.clamp(self.base_clip[0], self.base_clip[1])
        if self.source_clip:
            source_patches = source_patches.clamp(self.source_clip[0], self.source_clip[1])

        w = weight_patches
        x = base_patches
        y = source_patches

        # Sum over voxels (dim=1), keep batch dim → (B,)
        sw = self._sw_fixed + w.sum(dim=1)
        swx = self._swx_fixed + (w * x).sum(dim=1)
        swy = self._swy_fixed + (w * y).sum(dim=1)
        swxx = self._swxx_fixed + (w * x * x).sum(dim=1)
        swyy = self._swyy_fixed + (w * y * y).sum(dim=1)
        swxy = self._swxy_fixed + (w * x * y).sum(dim=1)

        sw_safe = sw.clamp(min=1e-10)
        xbar = swx / sw_safe
        ybar = swy / sw_safe
        vxx = (swxx / sw_safe - xbar * xbar).clamp(min=1e-20)
        vyy = (swyy / sw_safe - ybar * ybar).clamp(min=1e-20)
        vxy = swxy / sw_safe - xbar * ybar

        corr = vxy / (vxx * vyy).sqrt()
        return corr
