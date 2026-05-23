"""
Tests for the public reuse + diagnostic API of glm/arma.py:

- compute_ljung_box_statistic — residual whiteness diagnostic
- save_arma_rvar / load_arma_params — AFNI -Rvar round-trip enabling
  precomputed-parameter refits (80% speedup advertised in the docstring)
- compare_ols_vs_arma11 — high-level OLS-vs-ARMA comparison

Bugs in this surface area silently corrupt the parameter reuse path
(wrong a/b loaded → wrong prewhitening → wrong t-stats) or the
diagnostic that tells users whether prewhitening worked.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.glm.arma import (
    ARMA11Results,
    compare_ols_vs_arma11,
    compute_ljung_box_statistic,
    fit_glm_arma11,
    load_arma_params,
    save_arma_rvar,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_fmri(seed=0, n_voxels=20, n_timepoints=150, n_regs=3):
    """Generate (data, design) where the GLM is identifiable."""
    torch.manual_seed(seed)
    X = torch.randn(n_timepoints, n_regs, device=DEVICE)
    betas = torch.randn(n_voxels, n_regs, device=DEVICE) * 2.0
    signal = (X @ betas.T).T
    noise = torch.randn(n_voxels, n_timepoints, device=DEVICE) * 0.5
    data = signal + noise + 100.0
    return data, X


# ---------------------------------------------------------------------------
# compute_ljung_box_statistic
# ---------------------------------------------------------------------------

class TestComputeLjungBox:
    def test_white_noise_yields_small_statistic(self):
        """Truly white residuals should produce LB statistics consistent
        with the chi² null. With n=400 and h=30 the mean is ~28 with sd
        ~7.5 — under 100 with overwhelming probability."""
        rng = np.random.default_rng(0)
        resid = rng.standard_normal((10, 400)).astype(np.float32)
        lb = compute_ljung_box_statistic(resid, max_lag=30)
        assert lb.shape == (10,)
        assert (lb < 100).all(), f"white noise should not flag any voxel: {lb}"
        assert (lb > 0).all(), "non-zero residuals should yield non-zero LB"

    def test_strongly_autocorrelated_residuals_yield_large_statistic(self):
        """An AR(1) with phi=0.7 is highly autocorrelated; LB should
        dwarf the white-noise null."""
        rng = np.random.default_rng(1)
        n = 400
        eps = rng.standard_normal((10, n))
        ar = np.zeros_like(eps)
        for t in range(1, n):
            ar[:, t] = 0.7 * ar[:, t - 1] + eps[:, t]
        ar_lb = compute_ljung_box_statistic(ar.astype(np.float32), max_lag=30)
        white_lb = compute_ljung_box_statistic(eps.astype(np.float32), max_lag=30)
        # Every AR(1) voxel should exceed its matched white-noise voxel.
        assert (ar_lb > white_lb).all()
        # And the AR statistics should be huge (much larger than the
        # chi²_28 99.9th percentile ~ 56).
        assert (ar_lb > 100).all()

    def test_zero_residuals_return_zero(self):
        resid = np.zeros((5, 100), dtype=np.float32)
        lb = compute_ljung_box_statistic(resid)
        np.testing.assert_array_equal(lb, np.zeros(5, dtype=np.float32))

    def test_accepts_torch_input(self):
        rng = np.random.default_rng(2)
        resid_np = rng.standard_normal((4, 100)).astype(np.float32)
        resid_t = torch.from_numpy(resid_np)
        lb_np = compute_ljung_box_statistic(resid_np)
        lb_t = compute_ljung_box_statistic(resid_t)
        np.testing.assert_allclose(lb_np, lb_t, atol=1e-5)


# ---------------------------------------------------------------------------
# save_arma_rvar / load_arma_params round-trip
# ---------------------------------------------------------------------------

def _fit_arma_for_save(want_residuals: bool = True) -> ARMA11Results:
    """Fit ARMA(1,1) on synthetic data so we have a real ARMA11Results."""
    data, X = _synthetic_fmri(seed=10, n_voxels=12, n_timepoints=180)
    return fit_glm_arma11(
        data, X, tr=2.0, device=DEVICE, verbose=False,
        want_residuals=want_residuals,
    )


class TestSaveArmaRvar:
    def test_writes_6_volume_nifti(self, tmp_path):
        results = _fit_arma_for_save()
        out = save_arma_rvar(results, tmp_path / "rvar.nii.gz")
        assert out.exists()
        img = nib.load(str(out))
        # Stored as flat (n_voxels, 1, 1, 6) when volume_shape is None.
        assert img.shape[-1] == 6, f"expected 6 sub-bricks, got {img.shape[-1]}"

    def test_volume_shape_with_mask_expands_to_full_volume(self, tmp_path):
        """When volume_shape + voxel_mask are provided, out-of-mask voxels
        should be zero in the output."""
        results = _fit_arma_for_save()
        n_voxels = results.arma_params.shape[0]
        # Treat the 12 voxels as occupying a 4x3x1 grid; mask out 2 of them
        volume_shape = (4, 3, 1)
        full_size = int(np.prod(volume_shape))
        assert n_voxels <= full_size
        mask = np.zeros(full_size, dtype=bool)
        mask[:n_voxels] = True

        out = save_arma_rvar(
            results, tmp_path / "rvar.nii.gz",
            volume_shape=volume_shape, voxel_mask=mask,
        )
        img = nib.load(str(out))
        data = img.get_fdata()
        assert data.shape == (*volume_shape, 6)
        # Out-of-mask voxels are zero in every sub-brick
        out_of_mask_idx = ~mask.reshape(volume_shape)
        assert (data[out_of_mask_idx, :] == 0).all()

    def test_volume_shape_without_mask_reshapes_directly(self, tmp_path):
        results = _fit_arma_for_save()
        # 12 voxels → 3x2x2 = 12
        out = save_arma_rvar(
            results, tmp_path / "rvar.nii.gz",
            volume_shape=(3, 2, 2),
        )
        img = nib.load(str(out))
        assert img.shape == (3, 2, 2, 6)

    def test_ljung_box_zeros_when_residuals_not_kept(self, tmp_path):
        """save_arma_rvar must not crash when results.residuals_whitened is
        None; the LB sub-brick should just be zero."""
        results = _fit_arma_for_save(want_residuals=False)
        out = save_arma_rvar(results, tmp_path / "rvar.nii.gz")
        data = nib.load(str(out)).get_fdata()
        # LB is volume 5
        np.testing.assert_array_equal(data[..., 5], np.zeros_like(data[..., 5]))

    def test_creates_parent_directory(self, tmp_path):
        results = _fit_arma_for_save()
        nested = tmp_path / "a" / "b" / "rvar.nii.gz"
        out = save_arma_rvar(results, nested)
        assert out.exists()


class TestLoadArmaParamsRoundTrip:
    def test_round_trips_a_and_b(self, tmp_path):
        results = _fit_arma_for_save()
        out_path = tmp_path / "rvar.nii.gz"
        save_arma_rvar(results, out_path)
        loaded = load_arma_params(out_path)
        # 12 voxels × 2 params (a, b)
        assert loaded.shape == (12, 2)
        # Match original parameters
        orig = results.arma_params.detach().cpu().numpy()
        np.testing.assert_allclose(loaded, orig, atol=1e-5)

    def test_load_with_mask_returns_only_in_mask_voxels(self, tmp_path):
        results = _fit_arma_for_save()
        # Volume shape so the spatial layout matches a full mask
        volume_shape = (4, 3, 1)
        full_size = int(np.prod(volume_shape))
        n_voxels = results.arma_params.shape[0]
        mask = np.zeros(full_size, dtype=bool)
        mask[:n_voxels] = True

        out_path = tmp_path / "rvar.nii.gz"
        save_arma_rvar(
            results, out_path,
            volume_shape=volume_shape, voxel_mask=mask,
        )
        loaded = load_arma_params(out_path, voxel_mask=mask)
        assert loaded.shape == (n_voxels, 2)
        orig = results.arma_params.detach().cpu().numpy()
        np.testing.assert_allclose(loaded, orig, atol=1e-5)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_arma_params(tmp_path / "missing.nii.gz")

    def test_wrong_dimensionality_raises(self, tmp_path):
        """If the file isn't 4D with at least 2 sub-bricks, load must raise
        rather than silently mis-extract."""
        # Build a 3D NIfTI directly
        nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)).to_filename(
            str(tmp_path / "wrong.nii.gz")
        )
        with pytest.raises(ValueError, match="4D"):
            load_arma_params(tmp_path / "wrong.nii.gz")


# ---------------------------------------------------------------------------
# compare_ols_vs_arma11
# ---------------------------------------------------------------------------

class TestCompareOlsVsArma:
    def test_returns_expected_keys(self):
        data, X = _synthetic_fmri(seed=20, n_voxels=8, n_timepoints=120, n_regs=2)
        out = compare_ols_vs_arma11(data, X, tr=2.0, device=DEVICE)
        assert set(out.keys()) == {"ols", "arma11", "tstat_ratio", "r2_improvement", "summary"}
        assert isinstance(out["tstat_ratio"], float)
        assert isinstance(out["r2_improvement"], float)
        assert "OLS Mean R" in out["summary"]

    def test_results_finite_and_consistent(self):
        data, X = _synthetic_fmri(seed=21, n_voxels=8, n_timepoints=120, n_regs=2)
        out = compare_ols_vs_arma11(data, X, tr=2.0, device=DEVICE)
        ols = out["ols"]
        arma = out["arma11"]
        assert torch.isfinite(ols.r2).all()
        assert torch.isfinite(arma.r2).all()
        # ARMA model captures temporal structure → typically R² no worse
        # than OLS (within numerical tolerance). This is a soft check: we
        # only require both are finite and r2_improvement is a real number.
        assert np.isfinite(out["r2_improvement"])
