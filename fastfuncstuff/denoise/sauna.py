"""SAUNA: Signal-Adaptive Unbiased Noise Attenuation.

Next-generation fMRI denoiser inspired by NORDIC, with three key improvements:

1. **Direct g-factor from noise volumes** — Instead of an expensive MP-PCA
   proxy pass, SAUNA uses trailing noise-only volumes to measure the spatial
   noise pattern directly.  Even with k=3 noise volumes, the spatial pattern
   across hundreds of thousands of voxels is extremely well-determined, and
   g-factor varies smoothly (coil geometry), so spatial smoothing recovers
   it precisely.

2. **Optimal singular value shrinkage** — Instead of hard thresholding
   (keep/discard), SAUNA uses Gavish & Donoho (2017) optimal shrinkage that
   continuously attenuates each component based on its distance from the
   noise floor.  This is provably optimal for Frobenius-norm loss: it
   preserves weak signal that hard thresholding destroys, while still
   removing pure noise components entirely.

3. **Faster** — No g-factor LLR pass needed (saves ~30% of total time).
   The g-factor comes directly from noise volumes, requiring only a voxelwise
   std + Gaussian smooth.

Data flow
---------
1. Load magnitude (+phase), normalize by ABSOLUTE_SCALE
2. Phase stabilization (meanphase, DD_phase — reuses NORDIC functions)
3. G-factor from noise volumes: voxelwise std → smooth → bias-correct
4. Divide by g-factor → approximately homogeneous noise
5. Measure global σ from g-corrected noise volumes
6. Main LLR denoising with optimal Gavish-Donoho shrinkage
7. Restore: multiply by g-factor, undo DD_phase, undo meanphase, rescale
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from fastfuncstuff.io.afni import load_nifti, save_nifti
from fastfuncstuff.memory import (
    estimate_chunk_size,
    estimate_nordic_llr_memory,
    get_available_memory,
)
from fastfuncstuff.utils import get_device, to_tensor

# Reuse phase stabilization and LLR infrastructure from NORDIC
from fastfuncstuff.denoise.nordic import (
    _build_patch_starts,
    _compute_dd_phase,
    _compute_meanphase,
    _dd_phase_multiply_inplace,
    _default_kernel_size_pca,
    _LLRStats,
    _llr_denoise,
    _meanphase_unit,
    _phase_to_radians,
    _remove_meanphase,
    _restore_meanphase,
    _apply_temporal_phase_correction,
)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SaunaConfig:
    """Configuration for SAUNA denoising."""

    temporal_phase: int = 1
    phase_filter_width: float = 10.0
    noise_volume_last: int = 0
    magnitude_only: bool = False
    kernel_size_pca: tuple[int, int, int] | None = None
    patch_overlap: int = 2
    phase_slice_average: bool = False
    save_gfactor_map: bool = False
    save_residual_map: bool = False
    make_complex_nii: bool = False
    write_gzipped_niftis: bool = True
    svd_batch_size: int = 512
    decomp_method: str = "auto"
    verbose: bool = True
    gfactor_smooth_fwhm: float | str = "auto"  # float or "auto" (LOO cross-validated)
    gfactor_fwhm_range: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0)
    gfactor_method: str = "gaussian"  # "gaussian", "polynomial", or "auto"
    gfactor_degree_range: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8)
    shrinkage: str = "optimal"  # "optimal" (Gavish-Donoho) or "hard" (MP-PCA)


@dataclass
class SaunaOutputs:
    """Output paths from a SAUNA run."""

    magnitude_file: Path
    phase_file: Path | None
    gfactor_file: Path | None
    residual_file: Path | None
    metadata_file: Path


# ---------------------------------------------------------------------------
# G-factor estimation from noise volumes
# ---------------------------------------------------------------------------


def _c4_bias_correction(k: int) -> float:
    """Bias correction factor for sample std with k observations.

    The sample standard deviation of k i.i.d. normal observations has
    expectation c4 * σ.  Dividing by c4 gives an unbiased estimate.

    c4 = sqrt(2/(k-1)) * Gamma(k/2) / Gamma((k-1)/2)
    """
    if k < 2:
        return 1.0
    return math.sqrt(2.0 / (k - 1)) * math.gamma(k / 2.0) / math.gamma((k - 1) / 2.0)


def _gaussian_smooth_3d(
    vol: torch.Tensor,
    fwhm: float,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """3D Gaussian smoothing via separable 1D convolutions.

    Parameters
    ----------
    vol : (nx, ny, nz) tensor
    fwhm : full-width at half-maximum in voxel units
    voxel_size : voxel dimensions (for anisotropic data)

    Returns
    -------
    smoothed : same shape as vol
    """
    if fwhm <= 0:
        return vol

    sigma_vox = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    # Kernel radius: 3σ truncation
    radius = max(1, int(math.ceil(3.0 * sigma_vox)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=vol.device)
    kernel_1d = torch.exp(-0.5 * (x / sigma_vox) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Separable: convolve along each axis
    result = vol
    for dim in range(3):
        # Reshape kernel for conv1d along this axis
        k = kernel_1d.reshape(1, 1, -1)
        # Move target axis to last, flatten others into batch
        result = result.movedim(dim, -1)  # (..., axis_len)
        shape = result.shape
        flat = result.reshape(-1, 1, shape[-1])  # (batch, 1, L)
        flat = torch.nn.functional.pad(flat, (radius, radius), mode="replicate")
        flat = torch.nn.functional.conv1d(flat, k)
        result = flat.reshape(shape)
        result = result.movedim(-1, dim)  # restore axis

    return result


def _estimate_gfactor_from_noise(
    noise_vols: torch.Tensor,
    smooth_fwhm: float = 5.0,
    verbose: bool = True,
) -> tuple[torch.Tensor, float]:
    """Estimate g-factor map and global noise σ from noise-only volumes.

    Parameters
    ----------
    noise_vols : (nx, ny, nz, k) complex or real tensor — noise-only volumes
    smooth_fwhm : FWHM for Gaussian smoothing of the noise std map (voxels)
    verbose : print diagnostics

    Returns
    -------
    gfactor : (nx, ny, nz) float — relative noise map (median-normalized to 1)
    global_sigma : float — global noise std (after g-factor correction)
    """
    k = noise_vols.shape[-1]
    if k < 2:
        raise ValueError(f"Need at least 2 noise volumes for g-factor, got {k}")

    # Voxelwise std across noise volumes (complex: use abs for std)
    if noise_vols.is_complex():
        # For complex data, noise std = std of real part * sqrt(2)
        # But more robustly: std(abs(z)) isn't right either.
        # Use: sqrt(var(real) + var(imag)) which equals std of the complex modulus
        # for circular Gaussian noise.
        real_var = torch.var(noise_vols.real, dim=-1)
        imag_var = torch.var(noise_vols.imag, dim=-1)
        std_map = torch.sqrt(real_var + imag_var)
    else:
        std_map = torch.std(noise_vols, dim=-1)

    # Bias correction: E[sample_std] = c4 * σ
    c4 = _c4_bias_correction(k)

    # Smooth the std map — this is where the spatial information shines.
    # Even with k=3 (2 df per voxel), smoothing pools over many voxels,
    # dramatically reducing variance of the estimate.
    std_smooth = _gaussian_smooth_3d(std_map, smooth_fwhm)

    # Bias-correct after smoothing
    std_smooth = std_smooth / c4

    # Normalize to get g-factor (median = 1)
    nonzero = std_smooth[std_smooth > 0]
    if nonzero.numel() > 0:
        median_noise = float(torch.median(nonzero).item())
    else:
        median_noise = 1.0
    median_noise = max(median_noise, 1e-30)

    gfactor = std_smooth / median_noise

    # Clean pathological values
    gfactor = torch.clamp(gfactor, min=0.0)
    bad = (gfactor < 0.1) | ~torch.isfinite(gfactor)
    if bad.any():
        good_vals = gfactor[~bad & (gfactor > 0)]
        fill_val = float(torch.median(good_vals).item()) if good_vals.numel() > 0 else 1.0
        gfactor[bad] = fill_val

    # Global sigma: median noise level (the noise std after g-correction
    # should be approximately uniform at this value)
    global_sigma = median_noise

    if verbose:
        print(f"  G-factor from {k} noise volumes (c4={c4:.4f}, smooth_fwhm={smooth_fwhm})")
        print(f"  G-factor range: [{float(gfactor.min()):.4f}, {float(gfactor.max()):.4f}]")
        print(f"  Global noise σ: {global_sigma:.6g}")

    return gfactor, global_sigma


# ---------------------------------------------------------------------------
# Polynomial g-factor estimation (Legendre basis)
# ---------------------------------------------------------------------------


def _construct_3d_legendre_basis(
    shape: tuple[int, int, int],
    degree: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a 3D Legendre polynomial design matrix up to *total degree*.

    Uses tensor products P_i(x) · P_j(y) · P_k(z) for all (i, j, k) with
    i + j + k <= degree.  Coordinates are mapped to [-1, 1] in each dimension.

    Parameters
    ----------
    shape : (nx, ny, nz)
    degree : maximum **total** polynomial degree
    device : target device

    Returns
    -------
    basis : (N, n_terms) float tensor, where N = nx * ny * nz and
            n_terms = C(degree + 3, 3) = (d+1)(d+2)(d+3) / 6
    """
    nx, ny, nz = shape

    # 1-D Legendre polynomials on [-1, 1] for each axis
    # P_0(t) = 1, P_1(t) = t, P_k(t) via Bonnet recurrence
    def _legendre_1d(n_pts: int, max_order: int) -> torch.Tensor:
        """Return (n_pts, max_order+1) matrix of Legendre P_k(t)."""
        t = torch.linspace(-1.0, 1.0, n_pts, device=device)
        polys = torch.zeros(n_pts, max_order + 1, device=device)
        polys[:, 0] = 1.0
        if max_order >= 1:
            polys[:, 1] = t
        for k in range(2, max_order + 1):
            polys[:, k] = (
                (2 * k - 1) * t * polys[:, k - 1] - (k - 1) * polys[:, k - 2]
            ) / k
        return polys

    Px = _legendre_1d(nx, degree)  # (nx, d+1)
    Py = _legendre_1d(ny, degree)  # (ny, d+1)
    Pz = _legendre_1d(nz, degree)  # (nz, d+1)

    # Enumerate (i, j, k) with i+j+k <= degree
    columns = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            for k in range(degree + 1 - i - j):
                # Outer product: (nx, 1, 1) * (1, ny, 1) * (1, 1, nz) → (nx, ny, nz)
                col = (
                    Px[:, i].unsqueeze(1).unsqueeze(2)
                    * Py[:, j].unsqueeze(0).unsqueeze(2)
                    * Pz[:, k].unsqueeze(0).unsqueeze(1)
                )
                columns.append(col.reshape(-1))

    return torch.stack(columns, dim=1)  # (N, n_terms)


def _fit_polynomial_gfactor(
    noise_vols: torch.Tensor,
    degree: int,
    verbose: bool = False,
) -> tuple[torch.Tensor, float]:
    """Estimate g-factor by fitting a 3D Legendre polynomial to noise std.

    The fit is performed in **log-space**: we model ``log(std_map)`` as a
    polynomial, so ``g = exp(poly)`` — guaranteeing positivity everywhere
    with smooth extrapolation (no clamping, no sharp edges at boundaries).

    Parameters
    ----------
    noise_vols : (nx, ny, nz, k) complex or real
    degree : total polynomial degree (1–10 typical)
    verbose : print diagnostics

    Returns
    -------
    gfactor : (nx, ny, nz) float, median-normalized to 1
    global_sigma : float — global noise std
    """
    nx, ny, nz, k = noise_vols.shape

    # Voxelwise noise std
    if noise_vols.is_complex():
        real_var = torch.var(noise_vols.real, dim=-1)
        imag_var = torch.var(noise_vols.imag, dim=-1)
        std_map = torch.sqrt(real_var + imag_var)
    else:
        std_map = torch.std(noise_vols, dim=-1)

    # Bias-correct
    c4 = _c4_bias_correction(k)
    std_map = std_map / c4

    # Build design matrix on the same device as data
    dev_orig = std_map.device
    X = _construct_3d_legendre_basis((nx, ny, nz), degree, device=dev_orig)
    y = std_map.reshape(-1).float()

    # Mask out zeros / non-finite — log requires strictly positive
    valid = (y > 0) & torch.isfinite(y)
    if valid.sum() < X.shape[1]:
        # Not enough valid voxels — fall back to constant
        if verbose:
            print(f"  Poly fit deg={degree}: only {valid.sum()} valid voxels, falling back")
        median_val = float(torch.median(y[valid]).item()) if valid.any() else 1.0
        gfactor = torch.ones(nx, ny, nz, device=dev_orig)
        return gfactor, max(median_val, 1e-30)

    # Fit in log-space: log(std) = X @ beta  →  std = exp(X @ beta)
    # This guarantees positivity everywhere with smooth extrapolation.
    log_y = torch.log(y[valid])
    beta = torch.linalg.lstsq(X[valid], log_y.unsqueeze(1)).solution.squeeze(1)
    fitted = torch.exp((X @ beta).reshape(nx, ny, nz))

    # Normalize to median=1 using valid-voxel median
    valid_3d = valid.reshape(nx, ny, nz)
    median_noise = float(torch.median(fitted[valid_3d]).item())
    median_noise = max(median_noise, 1e-30)

    gfactor = fitted / median_noise

    # Only clean truly pathological (non-finite) values
    bad = ~torch.isfinite(gfactor)
    if bad.any():
        gfactor[bad] = 1.0

    global_sigma = median_noise

    if verbose:
        n_terms = X.shape[1]
        print(
            f"  G-factor poly deg={degree} ({n_terms} terms, c4={c4:.4f}, log-space fit)"
        )
        print(f"  G-factor range: [{float(gfactor.min()):.4f}, {float(gfactor.max()):.4f}]")
        print(f"  Global noise σ: {global_sigma:.6g}")

    return gfactor, global_sigma


def _heldout_nll(
    noise_vol: torch.Tensor,
    gfactor: torch.Tensor,
    global_sigma: float,
) -> float:
    """Per-voxel negative log-likelihood of held-out noise under the model.

    Model: each voxel i has noise ~ N(0, σ² g_i²).
    For complex data, real and imag parts are iid N(0, σ² g_i² / 2).

    NLL = Σ_i [ x_i² / (2 σ² g_i²) + log(σ g_i) + ½ log(2π) ]

    The constant ½ log(2π) is dropped (doesn't affect optimization).

    Parameters
    ----------
    noise_vol : (nx, ny, nz) or (nx, ny, nz, k) — held-out noise volume(s)
    gfactor : (nx, ny, nz) — g-factor map (median-normalized to 1)
    global_sigma : float — global noise std

    Returns
    -------
    nll : float — mean per-voxel negative log-likelihood (lower is better)
    """
    g = gfactor.clamp(min=1e-8)
    var_model = (global_sigma * g) ** 2  # (nx, ny, nz)

    if noise_vol.ndim == 4:
        # Multiple volumes: average NLL across all voxel-volume pairs
        if noise_vol.is_complex():
            # Real and imag each ~ N(0, var/2), so combined x²/var = (r²+i²)/var
            sq = noise_vol.real ** 2 + noise_vol.imag ** 2  # (nx, ny, nz, k)
        else:
            sq = noise_vol ** 2
        # sum over k volumes at each voxel
        nll_map = sq.sum(dim=-1) / (2.0 * var_model) + noise_vol.shape[-1] * torch.log(var_model) / 2.0
    else:
        if noise_vol.is_complex():
            sq = noise_vol.real ** 2 + noise_vol.imag ** 2
        else:
            sq = noise_vol ** 2
        nll_map = sq / (2.0 * var_model) + torch.log(var_model) / 2.0

    # Only score valid voxels (non-zero g-factor region)
    valid = torch.isfinite(nll_map) & (gfactor > 0.1)
    if valid.any():
        return float(nll_map[valid].mean().item())
    return float("inf")


def _loo_optimize_gfactor_degree(
    noise_vols: torch.Tensor,
    degree_candidates: tuple[int, ...],
    verbose: bool = True,
) -> tuple[int, dict[int, float]]:
    """LOO cross-validation to select polynomial degree for g-factor.

    Uses per-voxel negative log-likelihood as the scoring metric.
    Lower NLL = better model of the held-out noise variance.

    Parameters
    ----------
    noise_vols : (nx, ny, nz, k)
    degree_candidates : tuple of degrees to try (e.g., (1, 2, 3, 4, 5, 6))
    verbose : print progress

    Returns
    -------
    best_degree : int
    scores : dict mapping degree → mean NLL
    """
    k = noise_vols.shape[-1]
    scores: dict[int, float] = {}

    for deg in degree_candidates:
        loo_nlls = []
        for j in range(k):
            train_idx = [i for i in range(k) if i != j]
            train_vols = noise_vols[..., train_idx]

            gf_train, sigma_train = _fit_polynomial_gfactor(
                train_vols, deg, verbose=False
            )

            test_vol = noise_vols[..., j]
            nll = _heldout_nll(test_vol, gf_train, sigma_train)
            loo_nlls.append(nll)

        mean_nll = sum(loo_nlls) / len(loo_nlls)
        scores[deg] = mean_nll

    best_degree = min(scores, key=scores.get)  # type: ignore[arg-type]

    if verbose:
        print("  LOO polynomial degree optimization:")
        for deg in sorted(scores):
            marker = " ◀ best" if deg == best_degree else ""
            print(f"    degree={deg:2d}  NLL={scores[deg]:.4f}{marker}")

    return best_degree, scores


def _patch_variance_cov(
    noise_vol: torch.Tensor,
    gfactor: torch.Tensor,
    kernel_size: tuple[int, int, int],
    patch_overlap: int,
    n_sample_patches: int = 500,
) -> float:
    """Coefficient of variation of per-patch variance after g-correction.

    Used as the objective for LOO FWHM optimization.  Lower = more spatially
    uniform noise after g-correction = better g-factor estimate.

    Parameters
    ----------
    noise_vol : (nx, ny, nz) or (nx, ny, nz, k) — one or more noise volumes
    gfactor : (nx, ny, nz) — g-factor map
    kernel_size : patch dimensions for variance computation
    patch_overlap : overlap divisor
    n_sample_patches : max patches to sample (for speed)

    Returns
    -------
    cov : coefficient of variation of per-patch variance
    """
    if noise_vol.ndim == 3:
        noise_vol = noise_vol.unsqueeze(-1)

    nx, ny, nz = noise_vol.shape[:3]
    wx = min(kernel_size[0], nx)
    wy = min(kernel_size[1], ny)
    wz = min(kernel_size[2], nz)

    sx = max(1, wx // max(1, patch_overlap))
    sy = max(1, wy // max(1, patch_overlap))
    sz = max(1, wz // max(1, patch_overlap))

    xs = _build_patch_starts(nx, wx, sx)
    ys = _build_patch_starts(ny, wy, sy)
    zs = _build_patch_starts(nz, wz, sz)

    gc_noise = noise_vol / gfactor[..., None].clamp(min=1e-8)

    variances = []
    step = max(1, (len(xs) * len(ys) * len(zs)) // n_sample_patches)
    count = 0
    for x0 in xs:
        for y0 in ys:
            for z0 in zs:
                count += 1
                if count % step != 0:
                    continue
                patch = gc_noise[x0 : x0 + wx, y0 : y0 + wy, z0 : z0 + wz, :]
                if patch.is_complex():
                    v = float((torch.var(patch.real) + torch.var(patch.imag)).item())
                else:
                    v = float(torch.var(patch).item())
                if v > 0:
                    variances.append(v)

    if len(variances) < 2:
        return float("inf")

    var_arr = torch.tensor(variances)
    mean_v = float(var_arr.mean().item())
    std_v = float(var_arr.std().item())
    return std_v / max(mean_v, 1e-30)


def _loo_optimize_gfactor_fwhm(
    noise_vols: torch.Tensor,
    fwhm_candidates: tuple[float, ...],
    verbose: bool = True,
) -> tuple[float, dict[float, float]]:
    """Leave-one-out cross-validation to select optimal g-factor smoothing FWHM.

    For each candidate FWHM:
    1. For each noise volume j (held out):
       a. Estimate g-factor from the other k-1 volumes at this FWHM
       b. Score the held-out volume under the model via per-voxel NLL
    2. Average NLL across all held-out volumes
    3. Select FWHM with lowest average NLL

    Under-smoothing → g-factor is noisy → poor variance model → high NLL.
    Over-smoothing → real spatial structure lost → systematic bias → high NLL.

    Parameters
    ----------
    noise_vols : (nx, ny, nz, k) — noise-only volumes
    fwhm_candidates : tuple of FWHM values to try
    verbose : print progress

    Returns
    -------
    best_fwhm : float — optimal FWHM
    scores : dict mapping FWHM → mean NLL score
    """
    k = noise_vols.shape[-1]
    scores: dict[float, float] = {}

    for fwhm in fwhm_candidates:
        loo_nlls = []
        for j in range(k):
            train_idx = [i for i in range(k) if i != j]
            train_vols = noise_vols[..., train_idx]

            gf_train, sigma_train = _estimate_gfactor_from_noise(
                train_vols,
                smooth_fwhm=fwhm,
                verbose=False,
            )

            test_vol = noise_vols[..., j]
            nll = _heldout_nll(test_vol, gf_train, sigma_train)
            loo_nlls.append(nll)

        mean_nll = sum(loo_nlls) / len(loo_nlls)
        scores[fwhm] = mean_nll

    best_fwhm = min(scores, key=scores.get)  # type: ignore[arg-type]

    if verbose:
        print("  LOO FWHM optimization:")
        for fwhm in sorted(scores):
            marker = " ◀ best" if fwhm == best_fwhm else ""
            print(f"    FWHM={fwhm:5.1f}  NLL={scores[fwhm]:.4f}{marker}")

    return best_fwhm, scores


def _calibrate_sigma(
    noise_vols: torch.Tensor,
    gfactor: torch.Tensor,
    global_sigma: float,
    verbose: bool = True,
) -> dict[str, float]:
    """Validate σ calibration using held-out noise statistics.

    After g-correction, each voxel should have variance ≈ σ².  We compute
    the empirical ratio of measured variance to predicted variance, and
    the per-voxel NLL as an overall model quality metric.

    Returns a diagnostics dict with:
    - mean_var_ratio: measured_var / σ² (should be ≈ 1.0)
    - nll: per-voxel negative log-likelihood
    """
    nll = _heldout_nll(noise_vols, gfactor, global_sigma)

    # Global variance ratio
    gc_noise = noise_vols / gfactor[..., None].clamp(min=1e-8)
    if gc_noise.is_complex():
        total_var = float((torch.var(gc_noise.real) + torch.var(gc_noise.imag)).item())
    else:
        total_var = float(torch.var(gc_noise).item())
    expected_var = global_sigma**2
    ratio = total_var / max(expected_var, 1e-30)

    result = {
        "mean_var_ratio": ratio,
        "nll": nll,
        "global_sigma": global_sigma,
        "measured_var": total_var,
        "expected_var": expected_var,
    }

    if verbose:
        print(f"  σ calibration: measured_var/σ²={ratio:.4f} (expect ≈1.0), NLL={nll:.4f}")
        if abs(ratio - 1.0) > 0.3:
            print(
                f"  WARNING: σ calibration off by {abs(ratio - 1.0) * 100:.0f}% — "
                f"g-factor may be {'over' if ratio > 1 else 'under'}-smoothed"
            )

    return result


def _validate_gfactor_uniformity(
    noise_vols: torch.Tensor,
    gfactor: torch.Tensor,
    global_sigma: float,
    verbose: bool = True,
) -> float:
    """Check that g-corrected noise matches the variance model.

    Uses per-voxel NLL as a single scalar quality metric.

    Returns
    -------
    nll : float — per-voxel negative log-likelihood (lower is better)
    """
    nll = _heldout_nll(noise_vols, gfactor, global_sigma)

    if verbose:
        expected_var = global_sigma**2
        gc_noise = noise_vols / gfactor[..., None].clamp(min=1e-8)
        if gc_noise.is_complex():
            measured_var = float((torch.var(gc_noise.real) + torch.var(gc_noise.imag)).item())
        else:
            measured_var = float(torch.var(gc_noise).item())
        print(f"  G-factor validation: NLL={nll:.4f}")
        print(
            f"  Expected variance (σ²): {expected_var:.4f}, "
            f"measured: {measured_var:.4f}, ratio: {measured_var / max(expected_var, 1e-30):.3f}"
        )

    return nll


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_sauna(
    magnitude_file: str,
    phase_file: str | None,
    output_prefix: str,
    config: SaunaConfig | None = None,
    device: torch.device | None = None,
) -> SaunaOutputs:
    """Run SAUNA denoising: noise-volume g-factor + optimal shrinkage.

    Requires ``noise_volume_last >= 2`` — SAUNA needs actual noise volumes
    to estimate the spatial noise pattern directly.
    """
    cfg = config or SaunaConfig()
    dev = (
        device if device is not None else get_device("cuda" if torch.cuda.is_available() else None)
    )

    if cfg.noise_volume_last < 2:
        raise ValueError(
            f"SAUNA requires at least 2 trailing noise volumes "
            f"(noise_volume_last={cfg.noise_volume_last}). "
            f"Use ffs_nordic for data without noise volumes."
        )

    # ------------------------------------------------------------------
    # 1. Load data, form complex, apply ABSOLUTE_SCALE
    # ------------------------------------------------------------------
    mag_img = load_nifti(magnitude_file)
    mag_np = np.abs(mag_img.get_fdata(dtype=np.float32)).astype(np.float32)

    phase_np: np.ndarray | None = None
    phase_in_used = phase_file is not None and not cfg.magnitude_only
    if phase_in_used:
        phase_img = load_nifti(phase_file)
        phase_raw = phase_img.get_fdata(dtype=np.float32).astype(np.float32)
        phase_np = _phase_to_radians(phase_raw)

    if cfg.magnitude_only:
        ii_np = mag_np.astype(np.float32)
    elif phase_np is not None:
        ii_np = (mag_np * np.exp(1j * phase_np)).astype(np.complex64)
    else:
        ii_np = mag_np.astype(np.complex64)

    is_complex = not cfg.magnitude_only and phase_np is not None

    # ABSOLUTE_SCALE normalization (same as NORDIC)
    tempvol = np.abs(ii_np[..., 0])
    nonzero = tempvol[tempvol != 0]
    if nonzero.size > 0:
        absolute_scale = float(np.min(nonzero))
    else:
        absolute_scale = 1.0
    absolute_scale = max(absolute_scale, 1e-30)
    ii_np = ii_np / absolute_scale

    dtype = torch.complex64
    II = to_tensor(ii_np.astype(np.complex64), dtype=dtype, device=dev)
    del ii_np
    nx, ny, nz, nt = II.shape

    # Kernel size
    if cfg.kernel_size_pca is None:
        kernel_pca = _default_kernel_size_pca(nt, n_slices=nz)
    else:
        kernel_pca = cfg.kernel_size_pca

    if cfg.verbose:
        print("\nSAUNA denoising (Signal-Adaptive Unbiased Noise Attenuation)")
        print(f"  Input shape: {tuple(II.shape)}")
        print(f"  Device: {dev}")
        print(f"  Kernel PCA: {kernel_pca}")
        print(f"  Noise volumes: {cfg.noise_volume_last}")
        print(f"  Shrinkage: {cfg.shrinkage}")
        print(f"  ABSOLUTE_SCALE: {absolute_scale:.6g}")

    # ------------------------------------------------------------------
    # 2. Phase stabilization (reuse NORDIC functions)
    # ------------------------------------------------------------------
    effective_tp = cfg.temporal_phase if (phase_in_used and not cfg.magnitude_only) else 0

    mp_unit: torch.Tensor | None = None
    if cfg.phase_slice_average:
        meanphase = _compute_meanphase(II, cfg.noise_volume_last)
        mp_unit = _meanphase_unit(meanphase)
        del meanphase
        _remove_meanphase(II, mp_unit)

    dd_phase: torch.Tensor | None = None
    if effective_tp > 0:
        dd_phase = _compute_dd_phase(II, cfg.phase_filter_width, cfg.verbose)
        if effective_tp == 2:
            dd_phase = _apply_temporal_phase_correction(II, dd_phase, mode=2)
        _dd_phase_multiply_inplace(II, dd_phase)
        dd_phase = dd_phase.cpu()

    # ------------------------------------------------------------------
    # 3. G-factor from noise volumes (the key SAUNA advantage)
    #    Instead of an expensive MP-PCA LLR pass, directly measure the
    #    spatial noise pattern from trailing noise-only volumes.
    # ------------------------------------------------------------------
    n_noise = cfg.noise_volume_last
    noise_vols = II[..., nt - n_noise :].clone()

    gfactor_method = cfg.gfactor_method
    loo_scores: dict | None = None
    smooth_fwhm: float = 0.0
    poly_degree: int = 0

    if gfactor_method == "auto":
        # Run LOO for both gaussian and polynomial, pick the winner
        if n_noise < 3:
            # Can't LOO with < 3 volumes — fall back to gaussian with default
            gfactor_method = "gaussian"
            if cfg.verbose:
                print("  gfactor_method=auto: only 2 noise volumes, using gaussian FWHM=5.0")
        else:
            # Gaussian LOO
            best_fwhm, fwhm_scores = _loo_optimize_gfactor_fwhm(
                noise_vols,
                fwhm_candidates=cfg.gfactor_fwhm_range,
                verbose=cfg.verbose,
            )
            best_gauss_nll = fwhm_scores[best_fwhm]

            # Polynomial LOO
            best_deg, deg_scores = _loo_optimize_gfactor_degree(
                noise_vols,
                degree_candidates=cfg.gfactor_degree_range,
                verbose=cfg.verbose,
            )
            best_poly_nll = deg_scores[best_deg]

            if best_poly_nll <= best_gauss_nll:
                gfactor_method = "polynomial"
                poly_degree = best_deg
                loo_scores = {f"poly_deg_{k}": v for k, v in deg_scores.items()}
                loo_scores["gaussian_best_fwhm"] = best_fwhm
                loo_scores["gaussian_best_nll"] = best_gauss_nll
                if cfg.verbose:
                    print(
                        f"  Auto: polynomial (deg={best_deg}, NLL={best_poly_nll:.4f})"
                        f" beats gaussian (FWHM={best_fwhm}, NLL={best_gauss_nll:.4f})"
                    )
            else:
                gfactor_method = "gaussian"
                smooth_fwhm = best_fwhm
                loo_scores = {str(k): v for k, v in fwhm_scores.items()}
                loo_scores["poly_best_deg"] = best_deg
                loo_scores["poly_best_nll"] = best_poly_nll
                if cfg.verbose:
                    print(
                        f"  Auto: gaussian (FWHM={best_fwhm}, NLL={best_gauss_nll:.4f})"
                        f" beats polynomial (deg={best_deg}, NLL={best_poly_nll:.4f})"
                    )

    if gfactor_method == "polynomial":
        # Polynomial path
        if poly_degree == 0:
            # Not set by auto — need LOO or use middle of range
            if n_noise < 3:
                poly_degree = 3  # sensible default
                if cfg.verbose:
                    print(f"  Poly: only 2 noise volumes, using default degree={poly_degree}")
            else:
                poly_degree, deg_scores = _loo_optimize_gfactor_degree(
                    noise_vols,
                    degree_candidates=cfg.gfactor_degree_range,
                    verbose=cfg.verbose,
                )
                loo_scores = {f"poly_deg_{k}": v for k, v in deg_scores.items()}

        gfactor, global_sigma = _fit_polynomial_gfactor(
            noise_vols,
            degree=poly_degree,
            verbose=cfg.verbose,
        )
    else:
        # Gaussian path (default)
        if smooth_fwhm == 0.0:
            # Not yet determined — run FWHM selection
            if cfg.gfactor_smooth_fwhm == "auto":
                if n_noise < 3:
                    smooth_fwhm = 5.0
                    if cfg.verbose:
                        print("  FWHM auto: only 2 noise volumes, using default FWHM=5.0")
                else:
                    smooth_fwhm, fwhm_scores = _loo_optimize_gfactor_fwhm(
                        noise_vols,
                        fwhm_candidates=cfg.gfactor_fwhm_range,
                        verbose=cfg.verbose,
                    )
                    loo_scores = {str(k): v for k, v in fwhm_scores.items()}
            else:
                smooth_fwhm = float(cfg.gfactor_smooth_fwhm)

        gfactor, global_sigma = _estimate_gfactor_from_noise(
            noise_vols,
            smooth_fwhm=smooth_fwhm,
            verbose=cfg.verbose,
        )

    # Validate g-factor and σ calibration
    sigma_cal: dict[str, float] | None = None
    if cfg.verbose:
        _validate_gfactor_uniformity(
            noise_vols,
            gfactor,
            global_sigma,
            verbose=cfg.verbose,
        )
        sigma_cal = _calibrate_sigma(
            noise_vols,
            gfactor,
            global_sigma,
            verbose=cfg.verbose,
        )
    del noise_vols

    gfactor_file: Path | None = None

    # ------------------------------------------------------------------
    # 4. Divide by g-factor → homogeneous noise
    # ------------------------------------------------------------------
    II /= gfactor[..., None].clamp(min=1e-8)

    # For complex data: σ applies to real and imag separately, so the
    # noise per complex entry has std σ/√2 per component.
    # The SVD operates on the complex matrix, so per-entry noise std = σ.
    # But we measured σ from the real+imag variance, so it's already
    # the right combined measure for the SVD.
    if is_complex:
        noise_sigma = global_sigma / math.sqrt(2.0)
    else:
        noise_sigma = global_sigma

    if cfg.verbose:
        print(
            f"  Noise sigma for shrinkage: {noise_sigma:.6g}"
            f" ({'complex-adjusted' if is_complex else 'real'})"
        )

    # Clean nan/inf
    nan_mask = ~torch.isfinite(II)
    if nan_mask.any():
        II[nan_mask] = 0
    del nan_mask

    KSP2 = II

    # ------------------------------------------------------------------
    # 5. Determine threshold mode
    # ------------------------------------------------------------------
    if cfg.shrinkage == "optimal":
        threshold_mode = "optimal"
        threshold_value = 0.0  # not used for optimal
    else:
        # Fall back to MP-PCA hard threshold
        threshold_mode = "mp"
        threshold_value = 0.0

    # ------------------------------------------------------------------
    # 6. Main LLR denoising with optimal shrinkage
    # ------------------------------------------------------------------
    if dev.type == "cuda":
        mem_est = estimate_nordic_llr_memory(
            shape=KSP2.shape,
            kernel_size=kernel_pca,
            svd_batch_size=cfg.svd_batch_size,
            dtype_bytes=KSP2.element_size(),
            return_recon=True,
        )
        avail = get_available_memory(dev)
        gpu_without_data = mem_est["total"] - mem_est["data"]
        if mem_est["total"] > avail:
            if cfg.verbose:
                print(
                    f"  Memory guard: LLR needs ~{mem_est['total'] / 1024**3:.2f} GiB "
                    f"but only {avail / 1024**3:.2f} GiB available."
                )
                print(
                    f"  Offloading input ({mem_est['data'] / 1024**3:.2f} GiB) to CPU; "
                    f"accumulators + working set = {gpu_without_data / 1024**3:.2f} GiB on GPU."
                )
            KSP2 = KSP2.cpu()
            torch.cuda.empty_cache()

    denoised, llr_stats = _llr_denoise(
        KSP2,
        kernel_size=kernel_pca,
        patch_overlap=max(1, cfg.patch_overlap),
        threshold_mode=threshold_mode,
        threshold_value=threshold_value,
        verbose=cfg.verbose,
        svd_batch_size=cfg.svd_batch_size,
        decomp_method=cfg.decomp_method,
        device=dev,
        noise_sigma=noise_sigma,
    )

    # Compute residual (complex difference in transformed space) before
    # freeing the input.  Reuse KSP2 memory: residual = input - denoised.
    residual: torch.Tensor | None = None
    if cfg.save_residual_map:
        residual = KSP2.to(denoised.device) - denoised
    del KSP2
    II = None  # noqa: F841
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    mean_removed = float(torch.mean(llr_stats.threshold_map).item())
    if cfg.verbose:
        print(f"  Mean components zeroed per patch: {mean_removed:.3f}")
        mean_energy = float(torch.mean(llr_stats.energy_removed).item())
        print(f"  Mean energy removed per patch: {mean_energy:.4f}")

    # ------------------------------------------------------------------
    # 7. Undo transformations: gfactor, DD_phase, meanphase, scale
    # ------------------------------------------------------------------
    denoised *= gfactor[..., None]
    if residual is not None:
        residual *= gfactor.to(residual.device)[..., None]

    if dd_phase is not None:
        _dd_phase_multiply_inplace(denoised, dd_phase, conjugate=True)
        if residual is not None:
            _dd_phase_multiply_inplace(
                residual, dd_phase.to(residual.device), conjugate=True
            )
        del dd_phase
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if mp_unit is not None:
        _restore_meanphase(denoised, mp_unit)
        if residual is not None:
            _restore_meanphase(residual, mp_unit.to(residual.device))
    del mp_unit

    denoised *= absolute_scale
    if residual is not None:
        residual *= absolute_scale

    nan_mask = ~torch.isfinite(denoised)
    if nan_mask.any():
        denoised[nan_mask] = 0
    del nan_mask

    # ------------------------------------------------------------------
    # 8. Write outputs
    # ------------------------------------------------------------------
    out_prefix = Path(output_prefix)
    out_dir = out_prefix.parent if out_prefix.parent != Path("") else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = ".nii.gz" if cfg.write_gzipped_niftis else ".nii"

    if cfg.make_complex_nii:
        magn_path = out_dir / f"{out_prefix.name}_magn{ext}"
        phase_path = out_dir / f"{out_prefix.name}_phase{ext}"
    else:
        magn_path = out_dir / f"{out_prefix.name}{ext}"
        phase_path = out_dir / f"{out_prefix.name}_phase{ext}" if phase_in_used else None

    magn_np_out = torch.abs(denoised).cpu().numpy().astype(np.float32)
    phase_out_np: np.ndarray | None = None
    if phase_path is not None:
        phase_out_np = torch.angle(denoised).cpu().numpy().astype(np.float32)
    del denoised
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    save_nifti(magn_np_out, output_path=magn_path, reference_img=magnitude_file)
    del magn_np_out

    if phase_path is not None and phase_out_np is not None:
        save_nifti(
            phase_out_np,
            output_path=phase_path,
            reference_img=phase_file if phase_file is not None else magnitude_file,
        )
        del phase_out_np

    gfactor_file: Path | None = None
    if cfg.save_gfactor_map:
        gfactor_file = out_dir / f"{out_prefix.name}_gfactor{ext}"
        save_nifti(
            gfactor.detach().cpu().numpy().astype(np.float32),
            output_path=gfactor_file,
            reference_img=magnitude_file,
        )

    residual_file: Path | None = None
    if residual is not None:
        residual_file = out_dir / f"{out_prefix.name}_residual{ext}"
        save_nifti(
            torch.abs(residual).cpu().numpy().astype(np.float32),
            output_path=residual_file,
            reference_img=magnitude_file,
        )
        del residual

    meta = {
        "method": "SAUNA",
        "magnitude_file": str(magnitude_file),
        "phase_file": str(phase_file) if phase_file is not None else None,
        "output_prefix": str(output_prefix),
        "shape": [int(nx), int(ny), int(nz), int(nt)],
        "device": str(dev),
        "absolute_scale": absolute_scale,
        "config": {
            "temporal_phase": cfg.temporal_phase,
            "phase_filter_width": cfg.phase_filter_width,
            "noise_volume_last": cfg.noise_volume_last,
            "magnitude_only": cfg.magnitude_only,
            "kernel_size_pca": list(kernel_pca),
            "patch_overlap": cfg.patch_overlap,
            "phase_slice_average": cfg.phase_slice_average,
            "gfactor_method_requested": cfg.gfactor_method,
            "gfactor_method_used": gfactor_method,
            "gfactor_smooth_fwhm_requested": str(cfg.gfactor_smooth_fwhm),
            "gfactor_smooth_fwhm_used": float(smooth_fwhm) if gfactor_method == "gaussian" else None,
            "gfactor_poly_degree": poly_degree if gfactor_method == "polynomial" else None,
            "shrinkage": cfg.shrinkage,
        },
        "noise_estimation": {
            "n_noise_volumes": cfg.noise_volume_last,
            "global_sigma": float(global_sigma),
            "noise_sigma_for_shrinkage": float(noise_sigma),
            "c4_correction": float(_c4_bias_correction(cfg.noise_volume_last)),
            "loo_scores": {str(k): v for k, v in loo_scores.items()} if loo_scores else None,
            "sigma_calibration": sigma_cal,
        },
        "diagnostics": {
            "mean_components_zeroed": mean_removed,
            "mean_energy_removed": float(torch.mean(llr_stats.energy_removed).item()),
            "mean_snr_weight": float(torch.mean(llr_stats.snr_weight).item()),
        },
        "outputs": {
            "magnitude": str(magn_path),
            "phase": str(phase_path) if phase_path is not None else None,
            "gfactor": str(gfactor_file) if gfactor_file is not None else None,
            "residual": str(residual_file) if residual_file is not None else None,
        },
    }

    meta_file = out_dir / f"{out_prefix.name}_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return SaunaOutputs(
        magnitude_file=magn_path,
        phase_file=phase_path,
        gfactor_file=gfactor_file,
        residual_file=residual_file,
        metadata_file=meta_file,
    )
