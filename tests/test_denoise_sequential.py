"""
Tests for uncovered lines in denoise/sequential.py.

Targets functions/branches not covered by test_denoise_extended.py:
- estimate_noise_component_caps_per_run (lines 240-451)
- select_noise_pool_voxels edge cases (lines 559-573)
- extract_noise_pcs_per_run with component_caps, return_loadings, verbose (lines 643, 701-702, 723-724, 738-750)
- extract_noise_ics_per_run (lines 776-886)
- compute_full_brain_pc_loadings (lines 930-991)
- fit_glm_with_noise_pcs (lines 1050-1108)
- select_optimal_pcs (lines 1670-1724)
- compute_noise_pool_pca_scree_per_run (lines 454-502)
- compute_xval_r2_optimal_full (lines 1755-1943)
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.denoise.sequential import (
    ComponentCountEstimate,
    compute_full_brain_pc_loadings,
    compute_noise_pool_pca_scree_per_run,
    estimate_noise_component_caps_per_run,
    extract_noise_ics_per_run,
    extract_noise_pcs_per_run,
    fit_glm_with_noise_pcs,
    select_noise_pool_voxels,
    select_optimal_pcs,
)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_data():
    """Small (n_voxels=120, n_tp=90) dataset with 3 runs of 30 TRs."""
    torch.manual_seed(7)
    n_voxels, n_tp = 120, 90
    data = torch.randn(n_voxels, n_tp)
    run_starts = [0, 30, 60]
    noise_mask = torch.zeros(n_voxels, dtype=torch.bool)
    noise_mask[:60] = True  # 60 noise voxels
    return data, run_starts, noise_mask


@pytest.fixture
def nuisance_per_run():
    """Simple intercept nuisance per run (3 runs of 30 TRs)."""
    return [torch.ones(30, 1) for _ in range(3)]


# ---------------------------------------------------------------------------
# estimate_noise_component_caps_per_run  (lines 240-451)
# ---------------------------------------------------------------------------


class TestEstimateNoiseComponentCapsPerRun:
    def test_returns_component_count_estimate(self, small_data):
        data, run_starts, noise_mask = small_data
        result = estimate_noise_component_caps_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=10,
            min_components=2,
            device=DEVICE,
        )
        assert isinstance(result, ComponentCountEstimate)
        assert len(result.per_run_caps) == 3
        assert all(c >= 2 for c in result.per_run_caps)

    def test_caps_bounded_by_max_components(self, small_data):
        data, run_starts, noise_mask = small_data
        result = estimate_noise_component_caps_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=5,
            device=DEVICE,
        )
        assert all(c <= 5 for c in result.per_run_caps)

    def test_with_nuisance(self, small_data, nuisance_per_run):
        data, run_starts, noise_mask = small_data
        result = estimate_noise_component_caps_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=8,
            nuisance_per_run=nuisance_per_run,
            device=DEVICE,
        )
        assert len(result.per_run_caps) == 3

    def test_invalid_max_components(self, small_data):
        data, run_starts, noise_mask = small_data
        with pytest.raises(ValueError, match="max_components must be >= 1"):
            estimate_noise_component_caps_per_run(
                data=data,
                run_starts=run_starts,
                noise_pool_mask=noise_mask,
                max_components=0,
                device=DEVICE,
            )

    def test_mp_prior_disabled(self, small_data):
        data, run_starts, noise_mask = small_data
        result = estimate_noise_component_caps_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=8,
            use_mp_prior=False,
            device=DEVICE,
        )
        assert all(r == "disabled" for r in result.mp_reasons)

    def test_details_by_run_populated(self, small_data):
        data, run_starts, noise_mask = small_data
        result = estimate_noise_component_caps_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=8,
            device=DEVICE,
        )
        assert len(result.details_by_run) == 3
        for d in result.details_by_run:
            assert "variance_cap" in d
            assert "effective_rank" in d
            assert "selected_cap" in d


# ---------------------------------------------------------------------------
# compute_noise_pool_pca_scree_per_run  (lines 454-502)
# ---------------------------------------------------------------------------


class TestComputeNoisePoolPcaScreePerRun:
    def test_returns_per_run_scree(self, small_data):
        data, run_starts, noise_mask = small_data
        scree = compute_noise_pool_pca_scree_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=5,
            device=DEVICE,
        )
        assert len(scree) == 3
        for s in scree:
            assert s.shape == (5,)
            assert (s > 0).all()

    def test_with_nuisance(self, small_data, nuisance_per_run):
        data, run_starts, noise_mask = small_data
        scree = compute_noise_pool_pca_scree_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=5,
            nuisance_per_run=nuisance_per_run,
            device=DEVICE,
        )
        assert len(scree) == 3


# ---------------------------------------------------------------------------
# select_noise_pool_voxels edge cases  (lines 559-573)
# ---------------------------------------------------------------------------


class TestSelectNoisePoolEdgeCases:
    def test_too_large_noise_pool_triggers_adjustment(self):
        """When >95% of voxels are in noise pool, threshold should adjust."""
        torch.manual_seed(0)
        n_voxels = 200
        # Almost all voxels have very low R2
        r2 = torch.full((n_voxels,), -0.5)
        r2[:5] = 0.8  # Only 5 criteria voxels
        noise, criteria = select_noise_pool_voxels(r2, threshold=0.5, max_noise_fraction=0.90)
        # Noise pool should be capped at 90%
        assert noise.sum().item() <= int(n_voxels * 0.90) + 1

    def test_no_criteria_voxels_raises(self):
        """Should raise ValueError when no criteria voxels exist.

        Need: enough noise voxels (pass min check), but zero criteria voxels.
        All voxels below threshold -> all noise, zero criteria.
        """
        r2 = torch.full((200,), -1.0)
        with pytest.raises(ValueError, match="No criteria voxels"):
            select_noise_pool_voxels(r2, threshold=-0.5, min_noise_voxels=100)

    def test_too_few_noise_voxels_raises(self):
        """Should raise ValueError when noise pool is too small."""
        r2 = torch.full((200,), 0.9)  # All high R2
        with pytest.raises(ValueError, match="Noise pool has only"):
            select_noise_pool_voxels(r2, threshold=0.5, min_noise_voxels=100)


# ---------------------------------------------------------------------------
# extract_noise_pcs_per_run additional branches (lines 643, 701-750)
# ---------------------------------------------------------------------------


class TestExtractNoisePcsPerRunBranches:
    def test_component_caps_per_run(self, small_data):
        data, run_starts, noise_mask = small_data
        pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=5,
            component_caps_per_run=[2, 3, 2],
            device=DEVICE,
        )
        assert pcs[0].shape[1] == 2
        assert pcs[1].shape[1] == 3
        assert pcs[2].shape[1] == 2

    def test_component_caps_wrong_length_raises(self, small_data):
        data, run_starts, noise_mask = small_data
        with pytest.raises(ValueError, match="component_caps_per_run has"):
            extract_noise_pcs_per_run(
                data=data,
                run_starts=run_starts,
                noise_pool_mask=noise_mask,
                max_components=5,
                component_caps_per_run=[2, 3],
                device=DEVICE,
            )

    def test_return_loadings(self, small_data):
        data, run_starts, noise_mask = small_data
        pcs, loadings = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            return_loadings=True,
            device=DEVICE,
        )
        assert len(loadings) == 3
        n_noise = noise_mask.sum().item()
        for L in loadings:
            assert L.shape[0] == n_noise
            assert L.shape[1] == 3

    def test_verbose_with_nuisance(self, small_data, nuisance_per_run, capsys):
        data, run_starts, noise_mask = small_data
        extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            nuisance_per_run=nuisance_per_run,
            device=DEVICE,
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "PCs" in captured.out
        assert "nuisance projected" in captured.out

    def test_unit_variance_output(self, small_data):
        """PCs should be normalized to approximately unit variance."""
        data, run_starts, noise_mask = small_data
        pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        for pc in pcs:
            stds = pc.std(dim=0)
            assert torch.allclose(stds, torch.ones_like(stds), atol=0.15)


# ---------------------------------------------------------------------------
# extract_noise_ics_per_run  (lines 776-886)
# ---------------------------------------------------------------------------


class TestExtractNoiseIcsPerRun:
    def test_basic_extraction(self, small_data):
        data, run_starts, noise_mask = small_data
        ics = extract_noise_ics_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        assert len(ics) == 3
        for ic in ics:
            assert ic.shape == (30, 3)

    def test_return_loadings(self, small_data):
        data, run_starts, noise_mask = small_data
        ics, loadings = extract_noise_ics_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            return_loadings=True,
            device=DEVICE,
        )
        assert len(loadings) == 3
        n_noise = noise_mask.sum().item()
        for L in loadings:
            assert L.shape == (n_noise, 3)

    def test_return_variance_ratio(self, small_data):
        data, run_starts, noise_mask = small_data
        ics, var_ratio = extract_noise_ics_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            return_variance_ratio=True,
            device=DEVICE,
        )
        assert len(var_ratio) == 3
        for v in var_ratio:
            assert v.shape == (3,)
            # Variance ratios should sum to ~1
            assert abs(v.sum().item() - 1.0) < 0.01

    def test_return_loadings_and_variance_ratio(self, small_data):
        data, run_starts, noise_mask = small_data
        ics, loadings, var_ratio = extract_noise_ics_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            return_loadings=True,
            return_variance_ratio=True,
            device=DEVICE,
        )
        assert len(ics) == 3
        assert len(loadings) == 3
        assert len(var_ratio) == 3

    def test_component_caps(self, small_data):
        data, run_starts, noise_mask = small_data
        ics = extract_noise_ics_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=5,
            component_caps_per_run=[2, 3, 2],
            device=DEVICE,
        )
        assert ics[0].shape[1] == 2
        assert ics[1].shape[1] == 3
        assert ics[2].shape[1] == 2

    def test_component_caps_wrong_length_raises(self, small_data):
        data, run_starts, noise_mask = small_data
        with pytest.raises(ValueError, match="component_caps_per_run has"):
            extract_noise_ics_per_run(
                data=data,
                run_starts=run_starts,
                noise_pool_mask=noise_mask,
                max_components=5,
                component_caps_per_run=[2],
                device=DEVICE,
            )

    def test_unit_variance(self, small_data):
        """IC timecourses should be unit-variance normalized."""
        data, run_starts, noise_mask = small_data
        ics = extract_noise_ics_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        for ic in ics:
            stds = ic.std(dim=0)
            assert torch.allclose(stds, torch.ones_like(stds), atol=0.15)

    def test_with_nuisance(self, small_data, nuisance_per_run):
        data, run_starts, noise_mask = small_data
        ics = extract_noise_ics_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            nuisance_per_run=nuisance_per_run,
            device=DEVICE,
        )
        assert len(ics) == 3

    def test_verbose(self, small_data, capsys):
        data, run_starts, noise_mask = small_data
        extract_noise_ics_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "ICs" in captured.out

    def test_multiple_restarts(self, small_data):
        data, run_starts, noise_mask = small_data
        ics = extract_noise_ics_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            ica_restarts=2,
            device=DEVICE,
        )
        assert len(ics) == 3


# ---------------------------------------------------------------------------
# compute_full_brain_pc_loadings  (lines 930-991)
# ---------------------------------------------------------------------------


class TestComputeFullBrainPcLoadings:
    def test_basic(self, small_data):
        data, run_starts, noise_mask = small_data
        pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        loadings = compute_full_brain_pc_loadings(
            data=data,
            noise_pcs_per_run=pcs,
            run_starts=run_starts,
            device=DEVICE,
        )
        assert len(loadings) == 3
        for L in loadings:
            assert L.shape == (120, 3)

    def test_with_brain_mask(self, small_data):
        data, run_starts, noise_mask = small_data
        pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        brain_mask = torch.zeros(120, dtype=torch.bool)
        brain_mask[:80] = True
        loadings = compute_full_brain_pc_loadings(
            data=data,
            noise_pcs_per_run=pcs,
            run_starts=run_starts,
            brain_mask=brain_mask,
            device=DEVICE,
        )
        # Non-brain voxels should be zero
        for L in loadings:
            assert (L[80:, :] == 0).all()
            assert L[:80, :].abs().sum() > 0

    def test_brain_mask_size_mismatch_raises(self, small_data):
        data, run_starts, noise_mask = small_data
        pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        bad_mask = torch.ones(50, dtype=torch.bool)
        with pytest.raises(ValueError, match="Brain mask size"):
            compute_full_brain_pc_loadings(
                data=data,
                noise_pcs_per_run=pcs,
                run_starts=run_starts,
                brain_mask=bad_mask,
                device=DEVICE,
            )

    def test_verbose(self, small_data, capsys):
        data, run_starts, noise_mask = small_data
        pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        compute_full_brain_pc_loadings(
            data=data,
            noise_pcs_per_run=pcs,
            run_starts=run_starts,
            device=DEVICE,
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "loadings" in captured.out.lower()


# ---------------------------------------------------------------------------
# fit_glm_with_noise_pcs  (lines 1050-1108)
# ---------------------------------------------------------------------------


class TestFitGlmWithNoisePcs:
    def test_basic_fit(self, small_data):
        torch.manual_seed(42)
        data, run_starts, noise_mask = small_data
        n_conds = 2
        design = torch.randn(90, n_conds)
        pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        betas, r2 = fit_glm_with_noise_pcs(
            data=data,
            design_matrix=design,
            noise_pcs=pcs,
            run_starts=run_starts,
            n_pcs_to_use=2,
            tr=2.0,
            device=DEVICE,
        )
        assert betas.shape == (120, n_conds)
        assert r2.shape == (120,)

    def test_zero_pcs(self, small_data):
        torch.manual_seed(42)
        data, run_starts, noise_mask = small_data
        design = torch.randn(90, 2)
        pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        betas, r2 = fit_glm_with_noise_pcs(
            data=data,
            design_matrix=design,
            noise_pcs=pcs,
            run_starts=run_starts,
            n_pcs_to_use=0,
            tr=2.0,
            device=DEVICE,
        )
        assert betas.shape == (120, 2)

    def test_with_eval_mask(self, small_data):
        torch.manual_seed(42)
        data, run_starts, noise_mask = small_data
        design = torch.randn(90, 2)
        pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        eval_mask = torch.zeros(120, dtype=torch.bool)
        eval_mask[:50] = True
        betas, r2 = fit_glm_with_noise_pcs(
            data=data,
            design_matrix=design,
            noise_pcs=pcs,
            run_starts=run_starts,
            n_pcs_to_use=2,
            tr=2.0,
            eval_mask=eval_mask,
            device=DEVICE,
        )
        # Non-eval voxels should be NaN
        assert torch.isnan(r2[50:]).all()
        assert not torch.isnan(r2[:50]).any()

    def test_with_nuisance(self, small_data):
        torch.manual_seed(42)
        data, run_starts, noise_mask = small_data
        design = torch.randn(90, 2)
        nuisance = torch.ones(90, 1)
        pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        betas, r2 = fit_glm_with_noise_pcs(
            data=data,
            design_matrix=design,
            noise_pcs=pcs,
            run_starts=run_starts,
            n_pcs_to_use=2,
            tr=2.0,
            nuisance=nuisance,
            device=DEVICE,
        )
        assert betas.shape == (120, 2)


# ---------------------------------------------------------------------------
# select_optimal_pcs  (lines 1670-1724)
# ---------------------------------------------------------------------------


class TestSelectOptimalPcs:
    def test_basic_selection(self):
        # 50 voxels, 6 PC counts (0-5)
        r2_maps = np.random.RandomState(42).randn(50, 6).astype(np.float32)
        # Make PC=3 clearly best for many voxels
        r2_maps[:, 3] += 2.0
        optimal, criteria = select_optimal_pcs(r2_maps, threshold=0.0)
        assert optimal == 3
        assert criteria.shape == (50,)

    def test_no_criteria_voxels_uses_all(self):
        """When no voxel exceeds threshold, all voxels should be used."""
        r2_maps = np.full((30, 4), -1.0, dtype=np.float32)
        r2_maps[:, 2] = -0.5  # Best but still below 0
        optimal, criteria = select_optimal_pcs(r2_maps, threshold=0.0)
        assert criteria.all()  # All voxels used as fallback

    def test_mean_metric(self):
        r2_maps = np.random.RandomState(0).randn(40, 5).astype(np.float32)
        r2_maps[:, 1] += 3.0
        optimal, criteria = select_optimal_pcs(r2_maps, threshold=-10.0, metric="mean")
        assert optimal == 1

    def test_criteria_mask_correct(self):
        r2_maps = np.zeros((20, 3), dtype=np.float32)
        r2_maps[:10, :] = 0.5  # First 10 voxels have positive R2
        r2_maps[10:, :] = -0.5  # Last 10 voxels negative
        _, criteria = select_optimal_pcs(r2_maps, threshold=0.0)
        assert criteria[:10].all()
        assert not criteria[10:].any()


# ---------------------------------------------------------------------------
# compute_xval_r2_optimal_full  (lines 1755-1943)
# ---------------------------------------------------------------------------


class TestComputeXvalR2OptimalFull:
    """Test the full cross-validated R2 computation at optimal PC count."""

    def test_basic_no_nuisance(self, small_data):
        """Basic test without nuisance regressors."""
        from fastfuncstuff.denoise.sequential import compute_xval_r2_optimal_full

        torch.manual_seed(42)
        data, run_starts, noise_mask = small_data
        design = torch.randn(90, 2)
        pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        r2_all, r2_per_fold = compute_xval_r2_optimal_full(
            data=data,
            design_matrix=design,
            noise_pcs=pcs,
            run_starts=run_starts,
            optimal_n_components=2,
            device=DEVICE,
        )
        assert r2_all.shape == (120,)
        assert r2_per_fold.shape[1] == 120

    def test_with_nuisance(self, small_data, nuisance_per_run):
        """Test with per-run nuisance regressors."""
        from fastfuncstuff.denoise.sequential import compute_xval_r2_optimal_full

        torch.manual_seed(42)
        data, run_starts, noise_mask = small_data
        design = torch.randn(90, 2)
        pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        r2_all, r2_per_fold = compute_xval_r2_optimal_full(
            data=data,
            design_matrix=design,
            noise_pcs=pcs,
            run_starts=run_starts,
            optimal_n_components=2,
            nuisance=nuisance_per_run,
            device=DEVICE,
        )
        assert r2_all.shape == (120,)

    def test_zero_components(self, small_data):
        """Test with 0 optimal components (no denoising)."""
        from fastfuncstuff.denoise.sequential import compute_xval_r2_optimal_full

        torch.manual_seed(42)
        data, run_starts, noise_mask = small_data
        design = torch.randn(90, 2)
        pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        r2_all, _ = compute_xval_r2_optimal_full(
            data=data,
            design_matrix=design,
            noise_pcs=pcs,
            run_starts=run_starts,
            optimal_n_components=0,
            device=DEVICE,
        )
        assert r2_all.shape == (120,)

    def test_with_concat_nuisance(self, small_data):
        """Test with concatenated (non-list) nuisance tensor."""
        from fastfuncstuff.denoise.sequential import compute_xval_r2_optimal_full

        torch.manual_seed(42)
        data, run_starts, noise_mask = small_data
        design = torch.randn(90, 2)
        nuisance = torch.ones(90, 1)
        pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_mask,
            max_components=3,
            device=DEVICE,
        )
        r2_all, _ = compute_xval_r2_optimal_full(
            data=data,
            design_matrix=design,
            noise_pcs=pcs,
            run_starts=run_starts,
            optimal_n_components=2,
            nuisance=nuisance,
            device=DEVICE,
        )
        assert r2_all.shape == (120,)


# ---------------------------------------------------------------------------
# Per-HRF mode with CPU-resident data (large-dataset path)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device to mix")
def test_per_hrf_baseline_r2_with_cpu_resident_data():
    """Per-voxel HRF mode must work when data stays on CPU but device is CUDA.

    Datasets too large for VRAM are loaded to CPU and streamed; the HRF-group
    selection mask lives on the compute device, so indexing has to follow the
    data rather than the compute device.
    """
    from fastfuncstuff.denoise.sequential import fit_denoising_model

    torch.manual_seed(0)
    device = torch.device("cuda")
    n_runs, n_tp_run, n_voxels = 4, 60, 500
    run_starts = [i * n_tp_run for i in range(n_runs)]
    n_tp = n_runs * n_tp_run

    designs_by_hrf = {}
    for hrf_idx in (0, 1):
        design = torch.zeros(n_tp, 2, device=device)
        design[hrf_idx::7, 0] = 1.0
        design[(hrf_idx + 3) :: 11, 1] = 1.0
        designs_by_hrf[hrf_idx] = design

    hrf_indices = torch.zeros(n_voxels, dtype=torch.long, device=device)
    hrf_indices[n_voxels // 2 :] = 1

    data = torch.randn(n_voxels, n_tp)  # deliberately CPU while device is CUDA
    betas = torch.randn(n_voxels, 2) * 3.0
    betas[::3] = 0.0  # leave a genuine noise pool
    for hrf_idx in (0, 1):
        group = (hrf_indices == hrf_idx).cpu()
        data[group] += betas[group] @ designs_by_hrf[hrf_idx].cpu().T

    results = fit_denoising_model(
        data=data,
        designs_by_hrf=designs_by_hrf,
        hrf_indices=hrf_indices,
        run_starts=run_starts,
        tr=1.0,
        max_components=3,
        device=device,
        verbose=False,
    )

    assert results.xval_r2_per_voxel.shape[0] == n_voxels
    assert np.isfinite(results.xval_r2_per_voxel).any()


def test_per_hrf_mode_produces_a_noise_ceiling():
    """The ceiling must be estimated per HRF group and merged, not skipped.

    A voxel fit against its own HRF's design has its own ceiling -- that is what
    a per-voxel ceiling is -- and var(y) is design-free, so the groups land on
    one common scale. This used to print "not yet supported" and write nothing.
    """
    from fastfuncstuff.denoise.sequential import fit_denoising_model

    torch.manual_seed(4)
    device = torch.device("cpu")
    n_runs, n_tp_run, n_voxels = 6, 60, 400
    run_starts = [i * n_tp_run for i in range(n_runs)]
    n_tp = n_runs * n_tp_run

    designs_by_hrf = {}
    for hrf_idx in (0, 1):
        design = torch.zeros(n_tp, 2)
        # Every condition appears in every run, so every fold can split in two.
        for run in range(n_runs):
            base = run * n_tp_run + hrf_idx
            design[base + 2 : base + 30 : 9, 0] = 1.0
            design[base + 5 : base + 50 : 13, 1] = 1.0
        designs_by_hrf[hrf_idx] = design

    hrf_indices = torch.zeros(n_voxels, dtype=torch.long)
    hrf_indices[n_voxels // 2 :] = 1

    data = torch.randn(n_voxels, n_tp) * 0.4
    betas = torch.randn(n_voxels, 2) * 3.0
    betas[1::2] = 0.0  # half the brain is a genuine noise pool
    for hrf_idx in (0, 1):
        group = hrf_indices == hrf_idx
        data[group] += betas[group] @ designs_by_hrf[hrf_idx].T

    results = fit_denoising_model(
        data=data,
        designs_by_hrf=designs_by_hrf,
        hrf_indices=hrf_indices,
        run_starts=run_starts,
        tr=1.0,
        max_components=2,
        compute_noise_ceiling=True,
        device=device,
        verbose=False,
    )

    assert results.noise_ceiling is not None
    assert results.noise_ceiling.shape == (n_voxels,)
    # Both groups contributed: no voxel was left unclaimed by the merge.
    assert not torch.isnan(results.noise_ceiling).all()
    assert torch.isfinite(results.noise_ceiling[betas.abs().sum(1) > 0]).any()
    assert results.explainable_r2 is not None

    # The initial R2 is scored with no PCs removed, so it needs its own ceiling;
    # the optimal one bounds a different denominator.
    assert results.initial_noise_ceiling is not None
    assert results.initial_noise_ceiling.shape == (n_voxels,)
    assert results.initial_explainable_r2 is not None


def test_supplied_initial_r2_gets_no_initial_ceiling():
    """A caller-supplied initial R2 was measured elsewhere; we cannot bound it.

    ffs_hrfop hands over its own xval R2. A ceiling only bounds an R2 with the
    same denominator, and we have no way to know that one matches, so offering a
    ceiling for it would be a bound on nothing.
    """
    from fastfuncstuff.denoise.sequential import fit_denoising_model

    torch.manual_seed(5)
    n_runs, n_tp_run, n_voxels = 6, 60, 400
    run_starts = [i * n_tp_run for i in range(n_runs)]
    n_tp = n_runs * n_tp_run

    design = torch.zeros(n_tp, 2)
    for run in range(n_runs):
        base = run * n_tp_run
        design[base + 2 : base + 30 : 9, 0] = 1.0
        design[base + 5 : base + 50 : 13, 1] = 1.0

    data = torch.randn(n_voxels, n_tp) * 0.4
    betas = torch.randn(n_voxels, 2) * 3.0
    betas[1::2] = 0.0
    data += betas @ design.T

    results = fit_denoising_model(
        data=data,
        design_matrix=design,
        run_starts=run_starts,
        tr=1.0,
        max_components=2,
        initial_r2=torch.rand(n_voxels) * 0.2,  # as if from another tool
        compute_noise_ceiling=True,
        device=torch.device("cpu"),
        verbose=False,
    )

    assert results.noise_ceiling is not None  # the optimal one is still ours
    assert results.initial_noise_ceiling is None
    assert results.initial_explainable_r2 is None


# ---------------------------------------------------------------------------
# Frisch-Waugh-Lovell fast path for cross_validate_noise_pcs
# ---------------------------------------------------------------------------


def _fwl_case(n_runs=4, tp=40, n_vox=90, max_comp=3, ragged=False):
    """Synthetic multi-run dataset with task signal, PC noise and drift."""
    torch.manual_seed(11)
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    n_tp = n_runs * tp
    run_starts = [i * tp for i in range(n_runs)]

    design = torch.zeros(n_tp, 3)
    design[2::9, 0] = 1.0
    design[5::13, 1] = 1.0
    design[1::7, 2] = 1.0

    nuisance = [construct_polynomial_matrix(tp, 3, DEVICE).float() for _ in range(n_runs)]
    # A run with fewer components than requested exercises the ragged branch of
    # the nested projector (a real case: component caps differ per run).
    pcs = [torch.randn(tp, 2 if (ragged and r == 1) else max_comp) for r in range(n_runs)]

    data = torch.randn(n_vox, n_tp) * 0.5 + torch.randn(n_vox, 3) @ design.T
    for r in range(n_runs):
        sl = slice(run_starts[r], run_starts[r] + tp)
        data[:, sl] += torch.randn(n_vox, pcs[r].shape[1]) @ pcs[r].T
        data[:, sl] += torch.randn(n_vox, nuisance[r].shape[1]) @ nuisance[r].T * 2.0

    return dict(
        data=data,
        design_matrix=design,
        noise_pcs=pcs,
        run_starts=run_starts,
        tr=1.0,
        max_components=max_comp,
        nuisance=nuisance,
        device=DEVICE,
    )


def _explicit_full_model_r2(case, cv_strategy=1, n_perms=100, zero_event_strategy="zero"):
    """Reference R² built the slow, obvious way: fit [X | zero-padded PCs] directly.

    This is the formulation the fast path replaced. It exists here so the
    Frisch-Waugh-Lovell identity is checked against something independent
    rather than against a stored snapshot.

    Mirrors the pipeline's "concatenated predictions" semantics: each fold
    writes its test-run predictions into a full-length timeseries (so with
    overlapping test sets the last fold to cover a timepoint wins), and R² is
    computed once at the end.

    Under ``zero_event_strategy="nuisance"`` the held-out data AND the design
    predicting it both lose the span of the conditions that fold cannot fit,
    built here with an explicit pinv projector rather than the SVD basis the
    fast path uses -- so the two agree on the mathematics, not on the code.
    """
    from fastfuncstuff.glm.xval import (
        compute_r2_metric,
        generate_cv_splits,
        project_out_nuisance_per_run,
    )

    data = case["data"]
    design = case["design_matrix"]
    pcs = case["noise_pcs"]
    run_starts = case["run_starts"]
    max_comp = case["max_components"]
    n_tp = data.shape[1]
    n_vox = data.shape[0]
    n_runs = len(run_starts)
    run_ends = [run_starts[i + 1] if i < n_runs - 1 else n_tp for i in range(n_runs)]

    data_p, design_p = project_out_nuisance_per_run(
        data=data,
        design=design,
        nuisance_per_run=case["nuisance"],
        run_starts=run_starts,
        device=DEVICE,
    )

    r2_maps = np.zeros((n_vox, max_comp + 1), dtype=np.float64)
    splits = generate_cv_splits(n_runs, strategy=cv_strategy, n_perms=n_perms)
    # 'nuisance' scores against projected data, so the denominator is rebuilt
    # here fold by fold rather than taken from data_p.
    actual = data_p.double().clone()

    for k in range(max_comp + 1):
        pred = torch.zeros(n_vox, n_tp, dtype=torch.float64)

        for train_runs, test_runs in splits:
            tr_tps = torch.cat([torch.arange(run_starts[r], run_ends[r]) for r in train_runs])
            te_tps = torch.cat([torch.arange(run_starts[r], run_ends[r]) for r in test_runs])

            # Zero-padded block-diagonal PC block over the training runs
            blocks = []
            for pos, r in enumerate(train_runs):
                run_len = run_ends[r] - run_starts[r]
                pad = torch.zeros(run_len, len(train_runs) * k)
                n_use = min(k, pcs[r].shape[1])
                if n_use > 0:
                    pad[:, pos * k : pos * k + n_use] = pcs[r][:, :n_use]
                blocks.append(pad)
            # A condition with no events in any training run is not estimable
            # in this fold; it stays in the design everywhere else and predicts
            # zero here. Identity mask when every condition spans every run.
            fit_mask = design[tr_tps, :].abs().sum(dim=0) > 1e-10
            n_fit = int(fit_mask.sum())

            x_full = design_p[tr_tps, :][:, fit_mask]
            if k > 0:
                x_full = torch.cat([x_full, torch.cat(blocks, dim=0)], dim=1)

            pinv = torch.linalg.pinv(x_full.double(), rcond=1e-6)
            betas = (pinv @ data_p[:, tr_tps].double().T)[:n_fit, :]
            design_te = design_p[te_tps, :][:, fit_mask].double()

            if zero_event_strategy == "nuisance":
                # Per test RUN, since which columns are unpredictable depends on
                # the run being scored, not on the fold.
                for r in test_runs:
                    r_tps = torch.arange(run_starts[r], run_ends[r])
                    local = torch.searchsorted(te_tps, r_tps)
                    unpred = (~fit_mask) & (design[r_tps, :].abs().sum(dim=0) > 1e-10)
                    if not bool(unpred.any()):
                        continue
                    x_u = design_p[r_tps, :][:, unpred].double()
                    proj = x_u @ torch.linalg.pinv(x_u, rcond=1e-6)
                    design_te[local, :] -= proj @ design_te[local, :]
                    actual[:, r_tps] -= (proj @ actual[:, r_tps].T).T

            y_pred = design_te @ betas
            pred[:, te_tps] = y_pred.T

        r2_maps[:, k] = compute_r2_metric(actual, pred, metric="cod").numpy()
        actual = data_p.double().clone()  # rebuilt per PC count

    return r2_maps


@pytest.mark.parametrize("ragged", [False, True])
def test_cross_validate_noise_pcs_matches_explicit_full_model(ragged):
    """The FWL fast path must reproduce an explicit [X | PC] fit, per PC count.

    Task betas of the full model equal those of the PC-residualized reduced
    model; if that identity is ever broken the R² curve silently changes.
    """
    from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

    case = _fwl_case(ragged=ragged)
    fast, _ = cross_validate_noise_pcs(**case, cv_strategy=1)
    reference = _explicit_full_model_r2(case, cv_strategy=1)

    assert np.abs(fast - reference).max() < 1e-4


def test_cross_validate_noise_pcs_drops_conditions_absent_from_training():
    """A condition confined to one run must be dropped from that run's fold only.

    Under LORO the fold holding out run 0 sees an all-zero column for such a
    condition, so the Gram matrix is exactly singular. Excluding the column and
    taking the pseudo-inverse's minimum-norm solution agree here (the zero
    column zeroes its own row of X'y too), which is why this passed before the
    fold-plan rewrite as well -- the point of the test is that the two stay
    agreeing, and that the other conditions' betas are not disturbed by the
    unfittable one. Nothing covered this case at all previously.
    """
    from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

    case = _fwl_case()
    design = case["design_matrix"].clone()
    run_len = case["run_starts"][1] - case["run_starts"][0]
    design[:, 2] = 0.0
    design[1:run_len:7, 2] = 1.0  # events only in run 0
    case["design_matrix"] = design

    fast, _ = cross_validate_noise_pcs(**case, cv_strategy=1)
    reference = _explicit_full_model_r2(case, cv_strategy=1)

    assert np.isfinite(fast).all()
    assert np.abs(fast - reference).max() < 1e-4


def test_invert_fold_gram_matches_pseudo_inverse():
    """The eigh shortcut must equal pinv on the symmetric PSD Gram it is given.

    X'X is symmetric positive semi-definite, so its eigendecomposition is its
    SVD and the truncated inverse is the pseudo-inverse. Worth pinning: eigh was
    chosen for speed (CUDA pinv on a fold-sized batch measured 779 ms/call
    against 27 ms for CPU eigh), and the equivalence is what makes that legal.
    """
    from fastfuncstuff.denoise.sequential import _invert_fold_gram

    torch.manual_seed(3)
    a = torch.randn(5, 24, 60)
    gram = (a @ a.transpose(1, 2)).double()
    gram[:, -4:, :] *= 1e-6  # a near-null tail for the rcond cut to bite on
    gram[:, :, -4:] *= 1e-6

    reference = torch.linalg.pinv(gram, rcond=1e-6).float()
    assert torch.allclose(_invert_fold_gram(gram), reference, atol=1e-5, rtol=1e-4)


def test_cross_validate_noise_pcs_truncates_near_singular_directions():
    """Near-collinear conditions must be truncated, not inverted.

    Dropping the columns a fold cannot fit removes the *exact* singularity but
    not the near-singular directions a wide design produces. A plain inverse
    there is finite and catastrophically wrong -- held-out R2 in the hundreds
    negative -- so the fold's inverse has to keep the rcond truncation that the
    reference uses. Caught exactly this while rewriting the fold plan.
    """
    from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

    case = _fwl_case()
    design = case["design_matrix"]
    run_len = case["run_starts"][1] - case["run_starts"][0]
    # Column 2 duplicates column 0 to within 1e-7, and column 1 lives in run 0
    # alone -- ill-conditioning and unfittability in the same design.
    design[:, 2] = design[:, 0] + 1e-7 * torch.randn(design.shape[0])
    design[:, 1] = 0.0
    design[3:run_len:11, 1] = 1.0

    fast, _ = cross_validate_noise_pcs(**case, cv_strategy=1)
    reference = _explicit_full_model_r2(case, cv_strategy=1)

    assert np.abs(fast - reference).max() < 1e-3


def test_cross_validate_noise_pcs_chunking_is_invariant():
    """Chunk size must not change results: folds combine per-run totals."""
    from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

    case = _fwl_case()
    whole, _ = cross_validate_noise_pcs(**case, cv_strategy=1)
    chunked, _ = cross_validate_noise_pcs(**case, cv_strategy=1, chunk_size=17)

    assert np.abs(whole - chunked).max() < 1e-5


def test_cross_validate_noise_pcs_non_loro_matches_explicit():
    """Leave-two-out uses the full-accumulator path; it must agree too."""
    from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

    case = _fwl_case()
    fast, _ = cross_validate_noise_pcs(**case, cv_strategy=2, n_perms=6)
    reference = _explicit_full_model_r2(case, cv_strategy=2, n_perms=6)

    assert np.abs(fast - reference).max() < 1e-4


# ---------------------------------------------------------------------------
# Output layout: one stack per R2, carrying its own ceiling
# ---------------------------------------------------------------------------


def _minimal_results(with_ceiling: bool, n_voxels: int = 8):
    """A DenoiseResults carrying only what save_denoising_results reads."""
    from fastfuncstuff.denoise.sequential import DenoiseResults

    def ramp(offset):
        return torch.arange(n_voxels, dtype=torch.float32) / n_voxels + offset

    optional = {}
    if with_ceiling:
        optional = dict(
            noise_ceiling=ramp(0.5),
            explainable_r2=ramp(0.6),
            initial_noise_ceiling=ramp(0.7),
            initial_explainable_r2=ramp(0.8),
        )
    return DenoiseResults(
        optimal_n_components=2,
        xval_r2_by_n_components=np.zeros(3, dtype=np.float32),
        xval_r2_median_by_n_components=np.zeros(3, dtype=np.float32),
        xval_r2_per_fold=np.zeros((1, 3), dtype=np.float32),
        xval_r2_per_voxel=None,
        xval_r2_optimal=ramp(0.1),
        xval_r2_optimal_full=ramp(0.2),
        xval_r2_optimal_per_fold=None,
        noise_pool_mask=torch.ones(n_voxels, dtype=torch.bool),
        criteria_mask=torch.ones(n_voxels, dtype=torch.bool),
        pcselection_mask=None,
        valid_voxel_mask=torch.ones(n_voxels, dtype=torch.bool),
        noise_pool_r2=ramp(0.0),
        noise_pcs_per_run=[],
        pc_loadings_per_run=None,
        baseline_r2=0.0,
        optimal_r2=0.0,
        improvement=0.0,
        metadata={},
        **optional,
    )


def _save_and_read(tmp_path, results):
    import nibabel as nib

    from fastfuncstuff.cli.denoise import save_denoising_results

    files = save_denoising_results(
        results=results,
        output_prefix=str(tmp_path / "out"),
        volume_shape=(2, 2, 2),
        affine=np.eye(4),
        run_starts=[0],
        tr=1.0,
        save_pcs_mode="no",
        save_scree_plot=False,
        nii_ext=".nii.gz",
    )
    return files, {k: nib.load(v) for k, v in files.items() if str(v).endswith(".nii.gz")}


def test_r2_outputs_are_one_stack_each_with_their_own_ceiling(tmp_path):
    """Each R2 ships with the ceiling built at its own PC count, in one file.

    A ceiling only bounds an R2 with the same denominator, and the initial (0 PC)
    and optimal R2s are scored on differently-projected data. Keeping each pair
    in one labelled stack makes it impossible to read a map without its ceiling
    or to pair a map with the wrong one.
    """
    files, images = _save_and_read(tmp_path, _minimal_results(with_ceiling=True))

    for key in ("initial_r2", "xval_r2_optimal"):
        assert images[key].shape == (2, 2, 2, 3), f"{key} should be a 3-volume stack"

    # The separate sibling files these replaced must be gone.
    for retired in ("xval_r2_optimal_full", "noise_ceiling", "explainable_r2"):
        assert retired not in files


def test_r2_stacks_stay_3d_without_a_ceiling(tmp_path):
    """-noise_ceiling off must still write a plain R2 map, not a 1-volume stack."""
    _, images = _save_and_read(tmp_path, _minimal_results(with_ceiling=False))

    assert images["initial_r2"].shape == (2, 2, 2)
    assert images["xval_r2_optimal"].shape == (2, 2, 2)


def test_r2_stack_subbriks_are_in_label_order(tmp_path):
    """Sub-brik 0 is the R2, 1 the ceiling, 2 the ratio -- pinned by value."""
    results = _minimal_results(with_ceiling=True)
    _, images = _save_and_read(tmp_path, results)

    stack = images["xval_r2_optimal"].get_fdata()
    for index, expected in enumerate(
        (results.xval_r2_optimal_full, results.noise_ceiling, results.explainable_r2)
    ):
        assert np.allclose(stack[..., index].ravel(), expected.numpy(), atol=1e-6)

    initial = images["initial_r2"].get_fdata()
    for index, expected in enumerate(
        (results.noise_pool_r2, results.initial_noise_ceiling, results.initial_explainable_r2)
    ):
        assert np.allclose(initial[..., index].ravel(), expected.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# zero_event_strategy wiring for the PC-count curve
# ---------------------------------------------------------------------------


def _run_confined_case(amp=6.0, overlap=False, n_runs=4, tp=60, n_vox=90, max_comp=3):
    """A dense, strong condition that fires only in run 0.

    The LORO fold holding out run 0 trains on runs 1-3, where that column is
    all zero, so it has no beta for it. ``amp`` scales only that condition, and
    ``overlap`` slides it onto condition 0's blocks (|r| = 0.66 against 0.02).
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    torch.manual_seed(11)
    n_tp = n_runs * tp
    run_starts = [i * tp for i in range(n_runs)]

    # Boxcars, so the confined condition can be made to genuinely share variance
    # with a predictable one. Stick functions at different offsets have disjoint
    # support and stay near-orthogonal however they are slid, which is exactly
    # the regime where a one-sided projection is invisible.
    design = torch.zeros(n_tp, 3)
    offset = 1 if overlap else 3  # |r| with condition 0: 0.66 against 0.02
    for start in range(0, tp, 16):
        design[start : start + 4, 0] = 1.0
        design[start + 6 : start + 10, 1] = 1.0
        for run in range(n_runs):  # conditions 0 and 1 fire in every run
            base = run * tp + start
            design[base : base + 4, 0] = 1.0
            design[base + 6 : base + 10, 1] = 1.0
        design[start + offset : start + offset + 4, 2] = 1.0  # run 0 only

    nuisance = [construct_polynomial_matrix(tp, 3, DEVICE).float() for _ in range(n_runs)]
    pcs = [torch.randn(tp, max_comp) for _ in range(n_runs)]

    betas = torch.randn(n_vox, 3)
    betas[:, 2] *= amp
    data = torch.randn(n_vox, n_tp) * 0.5 + betas @ design.T
    for r in range(n_runs):
        sl = slice(run_starts[r], run_starts[r] + tp)
        data[:, sl] += torch.randn(n_vox, max_comp) @ pcs[r].T
        data[:, sl] += torch.randn(n_vox, nuisance[r].shape[1]) @ nuisance[r].T * 2.0

    return dict(
        data=data,
        design_matrix=design,
        noise_pcs=pcs,
        run_starts=run_starts,
        tr=1.0,
        max_components=max_comp,
        nuisance=nuisance,
        device=DEVICE,
    )


def test_zero_strategy_curve_is_unchanged_by_the_wiring():
    """'zero' is the default and every published result used it: it must not move.

    R2's denominator moved into the fold loop so it can be measured on the
    projected held-out data, gated on a projection actually happening. This pins
    the gate: with nothing projected the numbers must be bit-identical, not close.
    """
    from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

    case = _run_confined_case()
    explicit, _ = cross_validate_noise_pcs(**case, cv_strategy=1, zero_event_strategy="zero")
    default, _ = cross_validate_noise_pcs(**case, cv_strategy=1)

    np.testing.assert_array_equal(explicit, default)
    # And it still matches the slow reference, which is a 'zero' formulation.
    assert np.abs(default - _explicit_full_model_r2(case, cv_strategy=1)).max() < 1e-4


@pytest.mark.parametrize("overlap", [False, True])
def test_nuisance_curve_ignores_the_unpredictable_condition_entirely(overlap):
    """The whole claim of the strategy, stated as an invariance.

    Scaling a condition no fold can fit changes how much variance 'zero' is
    charged for, and its curve sags accordingly. 'nuisance' removes that
    subspace from the scoring, so its curve must not move at all -- the
    unpredictable condition's amplitude is simply not part of the measurement.
    """
    from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

    weak = _run_confined_case(amp=1.0, overlap=overlap)
    strong = _run_confined_case(amp=6.0, overlap=overlap)

    n_weak, _ = cross_validate_noise_pcs(**weak, cv_strategy=1, zero_event_strategy="nuisance")
    n_strong, _ = cross_validate_noise_pcs(**strong, cv_strategy=1, zero_event_strategy="nuisance")
    z_weak, _ = cross_validate_noise_pcs(**weak, cv_strategy=1, zero_event_strategy="zero")
    z_strong, _ = cross_validate_noise_pcs(**strong, cv_strategy=1, zero_event_strategy="zero")

    assert np.abs(np.median(n_weak, axis=0) - np.median(n_strong, axis=0)).max() < 1e-4
    # ... and that this is a real invariance rather than a curve that never moves.
    assert np.median(z_weak, axis=0).min() - np.median(z_strong, axis=0).min() > 0.005


@pytest.mark.parametrize("overlap", [False, True])
def test_nuisance_curve_matches_the_explicit_projected_model(overlap):
    """The fast path must reproduce an explicit fold-by-fold projection.

    The reference builds its projector with pinv on the design columns, the fast
    path with a truncated SVD basis applied twice, so agreement is about the
    mathematics rather than shared code. This is what pins the projection to
    both sides: projecting the held-out data without the design that predicts it
    passes every cruder check when the columns are near-orthogonal
    (``overlap=False``, |r| = 0.02) and only shows up under real sharing
    (``overlap=True``, |r| = 0.66).
    """
    from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

    case = _run_confined_case(overlap=overlap)
    fast, _ = cross_validate_noise_pcs(**case, cv_strategy=1, zero_event_strategy="nuisance")
    reference = _explicit_full_model_r2(case, cv_strategy=1, zero_event_strategy="nuisance")

    assert np.abs(fast - reference).max() < 1e-4


def test_nuisance_curve_beats_zero_when_the_lost_condition_is_separable(overlap=False):
    """With little sharing, declining to score the subspace is a clear win.

    Stated only for the near-orthogonal case on purpose. Once the unpredictable
    condition shares most of its variance with a real one, removing its subspace
    removes signal that was scorable, and 'nuisance' legitimately falls below
    'zero' -- that is the cost of not inventing error, not a regression.
    """
    from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

    case = _run_confined_case(overlap=overlap)
    zero, _ = cross_validate_noise_pcs(**case, cv_strategy=1, zero_event_strategy="zero")
    nuis, _ = cross_validate_noise_pcs(**case, cv_strategy=1, zero_event_strategy="nuisance")

    assert np.isfinite(nuis).all() and nuis.max() <= 1.0
    assert (np.median(nuis, axis=0) > np.median(zero, axis=0)).all()


def test_nuisance_strategy_is_a_no_op_when_every_condition_spans_runs():
    """No condition is ever unpredictable, so there is nothing to remove."""
    from fastfuncstuff.denoise.sequential import cross_validate_noise_pcs

    case = _fwl_case()  # all three conditions fire in every run
    zero, _ = cross_validate_noise_pcs(**case, cv_strategy=1, zero_event_strategy="zero")
    nuis, _ = cross_validate_noise_pcs(**case, cv_strategy=1, zero_event_strategy="nuisance")

    np.testing.assert_array_equal(zero, nuis)


def test_both_fit_denoising_model_call_sites_pass_the_same_options():
    """The per-HRF and single-HRF branches of ffs_denoise must not drift apart.

    Bug of record: -noise_ceiling worked on the single-HRF branch and did
    nothing on the per-HRF one, because only the single-HRF call passed
    compute_noise_ceiling. Nothing failed and nothing warned -- the ceiling was
    simply absent, and the R2 came out as a plain 3-D map that looked exactly
    like a run without the flag.

    Two long, near-identical keyword call sites are the shape of problem that
    keeps producing this, so the check is on the source: whatever one branch
    forwards, the other must forward too.
    """
    import ast
    import inspect

    from fastfuncstuff.cli import denoise as denoise_cli

    tree = ast.parse(inspect.getsource(denoise_cli))
    call_kwargs = [
        {kw.arg for kw in node.keywords if kw.arg is not None}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fit_denoising_model"
    ]

    assert len(call_kwargs) == 2, f"expected 2 call sites, found {len(call_kwargs)}"
    per_hrf, single_hrf = call_kwargs
    # Only the design arguments legitimately differ between the two branches.
    design_only = {"design_matrix", "designs_by_hrf", "hrf_indices"}
    assert (per_hrf ^ single_hrf) <= design_only, "call sites disagree on: " + ", ".join(
        sorted((per_hrf ^ single_hrf) - design_only)
    )
