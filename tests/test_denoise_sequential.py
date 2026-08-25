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


def _explicit_full_model_r2(case, cv_strategy=1, n_perms=100):
    """Reference R² built the slow, obvious way: fit [X | zero-padded PCs] directly.

    This is the formulation the fast path replaced. It exists here so the
    Frisch-Waugh-Lovell identity is checked against something independent
    rather than against a stored snapshot.

    Mirrors the pipeline's "concatenated predictions" semantics: each fold
    writes its test-run predictions into a full-length timeseries (so with
    overlapping test sets the last fold to cover a timepoint wins), and R² is
    computed once at the end.
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
            x_full = design_p[tr_tps, :]
            if k > 0:
                x_full = torch.cat([x_full, torch.cat(blocks, dim=0)], dim=1)

            pinv = torch.linalg.pinv(x_full.double(), rcond=1e-6)
            betas = (pinv @ data_p[:, tr_tps].double().T)[: design.shape[1], :]
            y_pred = design_p[te_tps, :].double() @ betas

            pred[:, te_tps] = y_pred.T

        r2_maps[:, k] = compute_r2_metric(data_p.double(), pred, metric="cod").numpy()

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
