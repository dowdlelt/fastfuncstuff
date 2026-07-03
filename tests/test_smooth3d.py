"""Correctness for fastfuncstuff.stats.smooth3d (variance smoothing)."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from fastfuncstuff.stats.smooth3d import (
    FWHM_TO_SIGMA,
    fwhm_mm_to_sigma_vox,
    gaussian3d_batched,
    smooth_var_per_perm,
)


def test_fwhm_to_sigma_constant():
    # FWHM = 2 sqrt(2 ln 2) sigma → sigma = FWHM * 0.4246609...
    assert abs(FWHM_TO_SIGMA - 1.0 / (2 * np.sqrt(2 * np.log(2)))) < 1e-12


def test_fwhm_mm_to_sigma_vox_anisotropic():
    s = fwhm_mm_to_sigma_vox(4.0, (2.0, 2.0, 4.0))
    # 4 mm FWHM ≈ 1.6986 mm σ → divided by per-axis voxel size:
    assert abs(s[0] - 1.6986436 / 2.0) < 1e-4
    assert abs(s[1] - 1.6986436 / 2.0) < 1e-4
    assert abs(s[2] - 1.6986436 / 4.0) < 1e-4


def test_gaussian3d_close_to_scipy_in_interior():
    """Our separable 3-D Gaussian agrees with scipy.ndimage in the volume
    interior.  Tight numerical agreement isn't expected — torch's
    reflect padding skips the edge value while scipy's mirrors it — but
    the result should be close to a few % in the interior.
    """
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(24, 24, 20)).astype(np.float32)
    sigma = (1.5, 1.5, 1.5)
    ours = gaussian3d_batched(torch.from_numpy(vol).unsqueeze(0), sigma).squeeze(0).numpy()
    ref = gaussian_filter(vol, sigma=sigma, mode="reflect", truncate=4.0)
    sl = (slice(6, -6), slice(6, -6), slice(6, -6))
    diff = np.abs(ours[sl] - ref[sl]).max()
    rms = np.sqrt(np.mean((ours[sl] - ref[sl]) ** 2))
    assert diff < 0.02, f"max interior diff {diff} too large"
    assert rms < 0.005, f"rms interior diff {rms} too large"


def test_mask_aware_constant_variance_stays_constant():
    """A constant in-mask variance map should remain constant after mask-aware smoothing."""
    mask = np.zeros((12, 12, 8), dtype=bool)
    mask[2:10, 2:10, 1:7] = True
    V = int(mask.sum())
    var_pv = torch.full((3, V), 4.0, dtype=torch.float32)
    smoothed = smooth_var_per_perm(var_pv, mask, sigma_vox=(1.5, 1.5, 1.5), device="cpu")
    np.testing.assert_allclose(smoothed.numpy(), 4.0, rtol=1e-5, atol=1e-5)


def test_smooth_var_per_perm_preserves_shape():
    rng = np.random.default_rng(1)
    mask = np.zeros((10, 10, 6), dtype=bool)
    mask[1:9, 1:9, 1:5] = True  # noqa: E702
    V = int(mask.sum())
    P = 7
    var_pv = torch.from_numpy(rng.uniform(0.5, 2.0, size=(P, V)).astype(np.float32))
    out = smooth_var_per_perm(var_pv, mask, sigma_vox=(1.0, 1.0, 1.0), device="cpu")
    assert out.shape == var_pv.shape
    # Smoothed variance stays positive
    assert float(out.min()) > 0
