"""MP2RAGE joint denoising: INV1 and INV2 smoothed with shared, noise-normalised weights.

UNI = Re(S1*·S2) / (|S1|² + |S2|²) divides out the receive field and, with it, scales the
thermal noise back up wherever that field is weak — so UNI is noisiest exactly where the
coil sees least. Denoising has to happen *before* the division, on the two inversions.

INV1 and INV2 are one anatomy seen twice with independent noise (their pseudo-residuals
correlate at ~0.006 in air), so one set of non-local-means weights is computed from both —
a patch must match in *both* contrasts to contribute — and applied to both. Neither image
is averaged across a boundary the other can see, and since each voxel's two values are
averaged over the same neighbourhood, their relationship is kept.

INV1 is denoised **signed** (polarity from the scanner UNI × magnitude): at TI1 grey matter
sits near its null, where the true signed signal passes smoothly through zero. Averaging
magnitudes there would preserve the Rician floor at the cortex; averaging signed values
cancels it. INV2 (high SNR everywhere in tissue) is averaged in the squared domain with the
2σ² Rician correction.

Quality referee: the residuals ``input - denoised`` of the two channels must stay
uncorrelated (independent noise). Removing real structure puts the same edges into both
residuals and the correlation departs from 0 — see :func:`residual_cross_correlation`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

UNI_SCALE = 4095.0
UNI_OFFSET = 2048.0


def uni_polarity(uni: Tensor, offset: float = UNI_OFFSET) -> Tensor:
    """Sign of ``Re(S1*·S2)`` from a scanner UNI (stored as ``r·4095 + 2048``)."""
    return torch.where(uni >= offset, 1.0, -1.0).to(uni.dtype)


def uni_from_inversions(
    inv1_signed: Tensor, inv2: Tensor, scale_k: float = 1.0, *, eps: float = 1e-6
) -> Tensor:
    """Magnitude-domain UNI ``k·s1·s2 / ((k·s1)² + s2²)`` in scanner units (0..4095).

    ``scale_k`` is the relative scaling of the stored INV1 vs INV2 (scanners scale each
    series independently); :func:`fit_inversion_scale` estimates it. Differs from the
    scanner's complex UNI by the ``cos Δφ`` phase-noise factor, which pulls the scanner's
    values toward the midpoint — compare denoised against this, not against scanner UNI.
    """
    a = scale_k * inv1_signed
    den = a * a + inv2 * inv2
    r = torch.where(den > eps, a * inv2 / den.clamp_min(eps), torch.zeros_like(den))
    return r * UNI_SCALE + UNI_OFFSET


def fit_inversion_scale(inv1: Tensor, inv2: Tensor, uni: Tensor, mask: Tensor) -> float:
    """Relative INV1/INV2 scaling ``k`` that best reproduces the scanner UNI (median-abs)."""
    idx = torch.nonzero(mask.flatten()).squeeze(1)
    idx = idx[:: max(1, idx.numel() // 400_000)]
    a = inv1.flatten()[idx].double()
    b = inv2.flatten()[idx].double()
    u = uni.flatten()[idx].double()
    s = uni_polarity(u)

    def err(k: float) -> float:
        return float((uni_from_inversions(s * a, b, k) - u).abs().median())

    grid = torch.exp(torch.linspace(math.log(0.2), math.log(5.0), 200)).tolist()
    best = min(grid, key=err)
    fine = torch.linspace(best * 0.97, best * 1.03, 120).tolist()
    return min(fine, key=err)


def pseudo_residual(x: Tensor) -> Tensor:
    """``√(6/7)·(x − mean of 6 face neighbours)`` — variance σ² for i.i.d. noise."""
    p = F.pad(x[None, None], (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
    nb = (
        p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1] + p[1:-1, 2:, 1:-1]
        + p[1:-1, :-2, 1:-1] + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]
    ) / 6.0  # fmt: skip
    return math.sqrt(6.0 / 7.0) * (x - nb)


def estimate_noise_sigma(x: Tensor, mask: Tensor, flat_quantile: float = 0.3) -> float:
    """Per-voxel noise σ from lag-1 differences in the flattest part of ``mask``.

    ``E[(x(v) − x(v+e))²]/2 = σ²(1 − ρ_e)`` for lag-1 noise correlation ``ρ_e``. Accelerated 3-D
    acquisitions correlate noise along the phase-encode axes, which makes every local
    estimator (and the 6-neighbour pseudo-residual) read low — and an underestimated σ
    inflates the patch distances until non-local means stops averaging. So the axis giving
    the *largest* value (least correlated, typically readout) is used. Anatomy leaks in at
    edges, hence only the ``flat_quantile`` of voxels with the smallest smoothed gradient.
    Means, not medians: integer-stored images quantise a median into steps of ~1/√2.
    """
    sm = F.avg_pool3d(x[None, None], 3, stride=1, padding=1)[0, 0]
    g = torch.zeros_like(sm)
    for d in range(3):
        g = g + (torch.roll(sm, 1, d) - torch.roll(sm, -1, d)) ** 2
    gm = g[mask]
    thr = torch.quantile(gm[:: max(1, gm.numel() // 2_000_000)], flat_quantile)
    flat = mask & (g <= thr)
    best = 0.0
    for d in range(3):
        diff2 = (x - torch.roll(x, 1, d)) ** 2
        both = flat & torch.roll(flat, 1, d)
        v = diff2[both]
        v = v[v <= torch.quantile(v[:: max(1, v.numel() // 2_000_000)], 0.99)]  # trim spikes
        best = max(best, float(v.mean() / 2.0))
    return math.sqrt(best)


def _shift(x: Tensor, off: tuple[int, int, int]) -> Tensor:
    return torch.roll(x, shifts=off, dims=(-3, -2, -1))


def joint_nlm(
    channels: Tensor,
    sigma: Tensor,
    *,
    rician: tuple[bool, ...],
    search_radius: int = 3,
    patch_radius: int = 1,
    beta: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Multi-channel non-local means with one shared weight per (voxel, offset).

    Patch distance ``D = mean_c pooled((x_c − x_c(·+o))²) / (2σ_c²)`` is ≈1 for two noisy
    copies of the same structure in every channel. ``w = exp(−max(D − 1, 0)/β)``; the centre
    gets the maximum neighbour weight (Buades). Rician channels average ``x²`` and subtract
    ``2σ²`` before the square root; the others (signed INV1) average ``x`` directly.

    Args:
        channels: ``(C, nz, ny, nx)``; crop to the head first — offsets wrap at the edges.
        sigma: ``(C,)`` noise σ per channel.
        rician: per-channel flag.

    Returns:
        ``(denoised (C, …), weight_sum (…))`` — the weight sum is the effective number of
        averaged voxels, a per-voxel diagnostic of how much smoothing happened.
    """
    n_chan = channels.shape[0]
    x = channels.float()
    inv_two_var = (1.0 / (2.0 * sigma.float() ** 2)).to(x.device).view(n_chan, 1, 1, 1)
    sq = torch.tensor(rician, device=x.device).view(n_chan, 1, 1, 1)
    acc = torch.zeros_like(x)
    wsum = torch.zeros_like(x[0])
    wmax = torch.zeros_like(x[0])
    k = 2 * patch_radius + 1
    r = search_radius
    offsets = [
        (dz, dy, dx)
        for dz in range(-r, r + 1)
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
        if (dz, dy, dx) != (0, 0, 0)
    ]
    val = torch.where(sq, x * x, x)
    for off in offsets:
        xs = _shift(x, off)
        d = ((x - xs) ** 2 * inv_two_var).mean(dim=0)
        d = F.avg_pool3d(d[None, None], k, stride=1, padding=patch_radius)[0, 0]
        w = torch.exp(-(d - 1.0).clamp_min(0.0) / beta)
        acc += w * _shift(val, off)
        wsum += w
        wmax = torch.maximum(wmax, w)
    acc += wmax * val
    wsum += wmax
    mean = acc / wsum.clamp_min(1e-12)
    two_var = (2.0 * sigma.float() ** 2).to(x.device).view(n_chan, 1, 1, 1)
    out = torch.where(sq, (mean - two_var).clamp_min(0.0).sqrt(), mean)
    return out, wsum


def residual_cross_correlation(
    raw: Tensor, denoised: Tensor, mask: Tensor, *, window: int = 0
) -> Tensor | float:
    """Correlation of the two channels' residuals ``raw − denoised`` inside ``mask``.

    Independent noise → ≈0. Structure removed by the denoiser appears in *both* residuals
    and pushes this away from 0. ``window > 0`` returns a local map (box window, voxels).
    """
    r1 = raw[0] - denoised[0]
    r2 = raw[1] - denoised[1]
    if window <= 0:
        a, b = r1[mask], r2[mask]
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))
    m = mask.float()

    def box(v: Tensor) -> Tensor:
        return F.avg_pool3d(v[None, None], window, stride=1, padding=window // 2)[0, 0]

    n = box(m).clamp_min(1e-6)
    m1, m2 = box(r1 * m) / n, box(r2 * m) / n
    c12 = box(r1 * r2 * m) / n - m1 * m2
    v1 = (box(r1 * r1 * m) / n - m1 * m1).clamp_min(1e-12)
    v2 = (box(r2 * r2 * m) / n - m2 * m2).clamp_min(1e-12)
    return torch.where(mask, c12 / (v1 * v2).sqrt(), torch.zeros_like(c12))
