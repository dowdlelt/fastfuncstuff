"""
Tests for cross-validation functionality.

Tests the core functions in fastfuncsim/xval.py that compute
cross-validated R² metrics.
"""

import numpy as np
import pytest
import torch

from fastfuncsim.xval import (
    compute_r2_metric,
    compute_xval_r2,
    generate_cv_splits,
    project_out_nuisance,
    slice_by_runs,
)


class TestGenerateCVSplits:
    """Test CV split generation"""

    def test_split_halves(self):
        """Test 50/50 split generation"""
        splits = generate_cv_splits(n_runs=4, strategy=0.5, n_perms=100)

        # Check we got splits
        assert len(splits) > 0
        assert len(splits) <= 100  # Max permutations

        # Check each split
        for train, test in splits:
            assert len(train) == 2  # Half of 4
            assert len(test) == 2
            assert set(train + test) == {0, 1, 2, 3}  # All runs covered
            assert set(train).isdisjoint(test)  # No overlap

    def test_loro(self):
        """Test leave-one-run-out"""
        splits = generate_cv_splits(n_runs=4, strategy=1, n_perms=100)

        # Should generate exactly 4 splits (one per run)
        assert len(splits) == 4

        # Check each split
        for train, test in splits:
            assert len(train) == 3
            assert len(test) == 1
            assert set(train + test) == {0, 1, 2, 3}

    def test_leave_two_out(self):
        """Test leave-two-runs-out"""
        splits = generate_cv_splits(n_runs=4, strategy=2, n_perms=100)

        # Should generate C(4,2) = 6 splits
        assert len(splits) == 6

        # Check each split
        for train, test in splits:
            assert len(train) == 2
            assert len(test) == 2
            assert set(train + test) == {0, 1, 2, 3}

    def test_custom_fraction(self):
        """Test custom train fraction"""
        splits = generate_cv_splits(n_runs=5, strategy=0.6, n_perms=100)

        for train, test in splits:
            assert len(train) == 3  # 0.6 * 5 = 3
            assert len(test) == 2
            assert set(train + test) == {0, 1, 2, 3, 4}


class TestSliceByRuns:
    """Test run-based slicing"""

    def test_single_run(self):
        """Test extracting a single run"""
        data = torch.randn(100, 800)
        design = torch.randn(800, 50)
        run_starts = [0, 200, 400, 600]

        data_sub, design_sub, tps = slice_by_runs(
            data, design, run_starts, [1]  # Second run only
        )

        assert data_sub.shape == (100, 200)  # 1 run × 200 TRs
        assert design_sub.shape == (200, 50)
        assert tps == list(range(200, 400))

    def test_multiple_runs(self):
        """Test extracting multiple runs"""
        data = torch.randn(100, 800)
        design = torch.randn(800, 50)
        run_starts = [0, 200, 400, 600]

        data_sub, design_sub, tps = slice_by_runs(
            data, design, run_starts, [0, 2]  # First and third runs
        )

        assert data_sub.shape == (100, 400)  # 2 runs × 200 TRs
        assert design_sub.shape == (400, 50)
        # Should be runs 0 and 2
        assert tps == list(range(0, 200)) + list(range(400, 600))

    def test_all_runs(self):
        """Test extracting all runs returns original"""
        data = torch.randn(100, 800)
        design = torch.randn(800, 50)
        run_starts = [0, 200, 400, 600]

        data_sub, design_sub, tps = slice_by_runs(
            data, design, run_starts, [0, 1, 2, 3]
        )

        assert data_sub.shape == data.shape
        assert design_sub.shape == design.shape
        assert tps == list(range(800))


class TestProjectOutNuisance:
    """Test nuisance projection"""

    def test_simple_projection(self):
        """Test basic nuisance projection"""
        n_voxels, n_timepoints = 100, 200
        data = torch.randn(n_voxels, n_timepoints)
        design = torch.randn(n_timepoints, 50)

        # Last 10 columns are nuisance
        nuisance_indices = list(range(40, 50))

        data_clean, design_clean = project_out_nuisance(
            data, design, nuisance_indices
        )

        # Check shapes
        assert data_clean.shape == data.shape
        assert design_clean.shape == design.shape

        # Check that data has changed (projection did something)
        # Original data and cleaned data should be different
        assert not torch.allclose(data, data_clean)

    def test_zero_columns_removed(self):
        """Test that all-zero columns are handled correctly"""
        n_voxels, n_timepoints = 100, 200
        data = torch.randn(n_voxels, n_timepoints)
        design = torch.randn(n_timepoints, 50)

        # Make some nuisance columns all-zero (simulating run-specific regressors)
        design[:, 45:] = 0

        nuisance_indices = list(range(40, 50))

        # Should not crash despite zero columns
        data_clean, design_clean = project_out_nuisance(
            data, design, nuisance_indices
        )

        assert data_clean.shape == data.shape
        assert design_clean.shape == design.shape

    def test_no_nuisance(self):
        """Test that empty nuisance list returns original"""
        data = torch.randn(100, 200)
        design = torch.randn(200, 50)

        data_clean, design_clean = project_out_nuisance(data, design, [])

        torch.testing.assert_close(data_clean, data)
        torch.testing.assert_close(design_clean, design)


class TestComputeR2Metric:
    """Test R² metric computation"""

    def test_perfect_prediction(self):
        """Test R² = 1 for perfect prediction"""
        y_true = torch.randn(100, 200)
        y_pred = y_true.clone()

        for metric in ["cod", "corr", "corr2"]:
            r2 = compute_r2_metric(y_true, y_pred, metric=metric)
            assert torch.allclose(r2, torch.ones_like(r2), atol=1e-5)

    def test_mean_prediction(self):
        """Test R² = 0 for mean prediction"""
        y_true = torch.randn(100, 200)
        y_pred = y_true.mean(dim=1, keepdim=True).expand_as(y_true)

        # CoD should be ~0 for mean predictor
        r2_cod = compute_r2_metric(y_true, y_pred, metric="cod")
        assert torch.allclose(r2_cod, torch.zeros_like(r2_cod), atol=1e-2)

    def test_random_prediction(self):
        """Test R² < 0 for random prediction (worse than mean)"""
        y_true = torch.randn(100, 200)
        y_pred = torch.randn(100, 200)

        # CoD can be negative for bad predictions
        r2_cod = compute_r2_metric(y_true, y_pred, metric="cod")
        # Most voxels should have negative R²
        assert (r2_cod < 0.1).sum() > 50

    def test_cod_vs_corr2(self):
        """Test that CoD and corr² are similar for good predictions"""
        # Create correlated data
        y_true = torch.randn(100, 200)
        y_pred = y_true + 0.1 * torch.randn(100, 200)  # Add small noise

        r2_cod = compute_r2_metric(y_true, y_pred, metric="cod")
        r2_corr2 = compute_r2_metric(y_true, y_pred, metric="corr2")

        # Should be very similar (within 0.05)
        assert (r2_cod - r2_corr2).abs().mean() < 0.05


class TestComputeXvalR2:
    """Integration tests for cross-validated R²"""

    def test_synthetic_data(self):
        """Test with synthetic data and known ground truth"""
        n_voxels, n_timepoints, n_stim, n_nuisance = 50, 400, 20, 5
        n_runs = 4

        # Create synthetic data
        # Ground truth: data = design_stim @ true_betas + noise
        true_betas = torch.randn(n_stim, n_voxels)
        design_stim = torch.randn(n_timepoints, n_stim)
        design_nuisance = torch.randn(n_timepoints, n_nuisance)
        design_full = torch.cat([design_stim, design_nuisance], dim=1)

        # Generate data
        data = (design_stim @ true_betas).T + 0.1 * torch.randn(n_voxels, n_timepoints)

        # Run starts
        run_starts = [0, 100, 200, 300]
        stim_indices = list(range(n_stim))
        nuisance_indices = list(range(n_stim, n_stim + n_nuisance))

        # Generate splits
        cv_splits = generate_cv_splits(n_runs=4, strategy=1, n_perms=10)  # LORO

        # Compute xval R²
        results = compute_xval_r2(
            data=data,
            design_matrix=design_full.numpy(),
            run_starts=run_starts,
            stim_indices=stim_indices,
            nuisance_indices=nuisance_indices,
            cv_splits=cv_splits,
            metric="cod",
            verbose=False,
        )

        # Check outputs
        assert "r2" in results
        assert "r2_median" in results  # Backward compat - misleading name, actually per-voxel R²
        assert "r2_mean" in results
        assert "r2_std" in results
        assert "r2_min" in results
        assert "r2_max" in results
        assert "n_splits" in results

        # Check shapes
        assert results["r2"].shape == (n_voxels,)
        assert results["r2_median"].shape == (n_voxels,)  # Same as r2 (misleading name)
        # Note: r2_std, r2_min, r2_max are scalars, not per-voxel

        # Sanity checks on values
        # With low noise, R² should be high
        assert results["r2"].mean() > 0.7

        # Cross-validated R² should be positive for good predictions
        assert (results["r2"] > 0).sum() == n_voxels

    def test_different_metrics(self):
        """Test that different metrics give similar results"""
        n_voxels, n_timepoints = 50, 400
        data = torch.randn(n_voxels, n_timepoints)
        design = torch.randn(n_timepoints, 30)

        run_starts = [0, 100, 200, 300]
        stim_indices = list(range(20))
        nuisance_indices = list(range(20, 30))
        cv_splits = generate_cv_splits(n_runs=4, strategy=1, n_perms=10)

        results_cod = compute_xval_r2(
            data, design.numpy(), run_starts, stim_indices, nuisance_indices,
            cv_splits, metric="cod", verbose=False
        )

        results_corr2 = compute_xval_r2(
            data, design.numpy(), run_starts, stim_indices, nuisance_indices,
            cv_splits, metric="corr2", verbose=False
        )

        # CoD and corr² should be reasonably similar
        # (may differ more for poor predictions)
        diff = (results_cod["r2_median"] - results_corr2["r2_median"]).abs()
        assert diff.mean() < 0.3  # Allow some difference


def test_end_to_end_workflow():
    """Test the complete workflow from splits to results"""
    # Setup
    n_voxels, n_runs = 100, 4
    run_length = 100
    n_timepoints = n_runs * run_length

    data = torch.randn(n_voxels, n_timepoints)
    design = torch.randn(n_timepoints, 50)

    run_starts = [i * run_length for i in range(n_runs)]
    stim_indices = list(range(40))
    nuisance_indices = list(range(40, 50))

    # Generate splits
    cv_splits = generate_cv_splits(n_runs=n_runs, strategy=0.5, n_perms=10)

    # Compute xval R²
    results = compute_xval_r2(
        data=data,
        design_matrix=design.numpy(),
        run_starts=run_starts,
        stim_indices=stim_indices,
        nuisance_indices=nuisance_indices,
        cv_splits=cv_splits,
        metric="cod",
        verbose=False,
    )

    # Validate results structure
    assert isinstance(results, dict)
    assert all(key in results for key in ["r2", "r2_median", "r2_mean", "r2_std", "r2_min", "r2_max"])

    # Validate statistics make sense (scalar stats in new GLMdenoise-style API)
    # r2_median is same as r2 (misleading name for backward compat)
    assert results["r2_median"].shape == results["r2"].shape
    assert torch.equal(results["r2_median"], results["r2"])

    # Scalar statistics
    assert results["r2_mean"] <= results["r2_max"] + 1e-5
    assert results["r2_mean"] >= results["r2_min"] - 1e-5
    assert results["r2_std"] >= 0


class TestGenerateCVSplitsEdgeCases:
    """Test edge cases and error conditions for generate_cv_splits"""

    def test_invalid_float_strategy_zero(self):
        """Test that strategy=0.0 raises ValueError"""
        with pytest.raises(ValueError, match="Float strategy must be in"):
            generate_cv_splits(n_runs=4, strategy=0.0)

    def test_invalid_float_strategy_one(self):
        """Test that strategy=1.0 raises ValueError"""
        with pytest.raises(ValueError, match="Float strategy must be in"):
            generate_cv_splits(n_runs=4, strategy=1.0)

    def test_invalid_float_strategy_negative(self):
        """Test that negative float strategy raises ValueError"""
        with pytest.raises(ValueError, match="Float strategy must be in"):
            generate_cv_splits(n_runs=4, strategy=-0.5)

    def test_invalid_float_strategy_greater_than_one(self):
        """Test that strategy > 1.0 raises ValueError"""
        with pytest.raises(ValueError, match="Float strategy must be in"):
            generate_cv_splits(n_runs=4, strategy=1.5)

    def test_fraction_gives_zero_train(self):
        """Test fraction that results in n_train=0 raises ValueError"""
        # With n_runs=3 and strategy=0.1, n_train = int(3 * 0.1) = 0
        with pytest.raises(ValueError, match="results in n_train=0"):
            generate_cv_splits(n_runs=3, strategy=0.1)

    def test_invalid_int_strategy_zero(self):
        """Test that strategy=0 raises ValueError"""
        with pytest.raises(ValueError, match="must be > 0"):
            generate_cv_splits(n_runs=4, strategy=0)

    def test_invalid_int_strategy_negative(self):
        """Test that negative int strategy raises ValueError"""
        with pytest.raises(ValueError, match="must be > 0"):
            generate_cv_splits(n_runs=4, strategy=-1)

    def test_invalid_int_strategy_equals_n_runs(self):
        """Test that strategy >= n_runs raises ValueError"""
        with pytest.raises(ValueError, match="must be > 0 and < n_runs"):
            generate_cv_splits(n_runs=4, strategy=4)

    def test_invalid_strategy_type(self):
        """Test that non-int/float strategy raises ValueError"""
        with pytest.raises(ValueError, match="must be float or int"):
            generate_cv_splits(n_runs=4, strategy="half")

    def test_sampling_float_many_combinations(self):
        """Test that having more combinations than n_perms triggers sampling"""
        # 10 runs with 0.5 split = C(10,5) = 252 combinations
        # But n_perms=5 should sample only 5
        splits = generate_cv_splits(n_runs=10, strategy=0.5, n_perms=5)
        assert len(splits) == 5
        
        # Each split should still be valid
        for train, test in splits:
            assert len(train) == 5
            assert len(test) == 5
            assert set(train + test) == set(range(10))

    def test_sampling_int_many_combinations(self):
        """Test leave-N-out with more combinations than n_perms triggers sampling"""
        # 10 runs with L2O = C(10,2) = 45 combinations
        # But n_perms=5 should sample only 5
        splits = generate_cv_splits(n_runs=10, strategy=2, n_perms=5)
        assert len(splits) == 5
        
        # Each split should be valid
        for train, test in splits:
            assert len(train) == 8
            assert len(test) == 2
            assert set(train + test) == set(range(10))


class TestComputeR2MetricEdgeCases:
    """Test edge cases for R² metric computation"""

    def test_shape_mismatch(self):
        """Test that shape mismatch raises ValueError"""
        y_true = torch.randn(100, 200)
        y_pred = torch.randn(100, 150)  # Different timepoints
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_r2_metric(y_true, y_pred)

    def test_unknown_metric(self):
        """Test that unknown metric raises ValueError"""
        y_true = torch.randn(100, 200)
        y_pred = torch.randn(100, 200)
        
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_r2_metric(y_true, y_pred, metric="unknown")


class TestProjectOutNuisanceEdgeCases:
    """Test edge cases for nuisance projection"""

    def test_all_nuisance_columns_zero(self):
        """Test when ALL nuisance columns are zero (returns original)"""
        data = torch.randn(100, 200)
        design = torch.randn(200, 50)
        
        # Make all nuisance columns zero
        design[:, 40:] = 0
        nuisance_indices = list(range(40, 50))
        
        data_clean, design_clean = project_out_nuisance(data, design, nuisance_indices)
        
        # Should return original since nothing to project
        torch.testing.assert_close(data_clean, data)
        torch.testing.assert_close(design_clean, design)


class TestComputeXvalR2EdgeCases:
    """Test edge cases for cross-validated R² computation"""

    def test_no_nuisance_regressors(self):
        """Test xval R² with empty nuisance_indices"""
        n_voxels, n_timepoints = 50, 400
        data = torch.randn(n_voxels, n_timepoints)
        design = torch.randn(n_timepoints, 20)  # All stimulus, no nuisance

        run_starts = [0, 100, 200, 300]
        stim_indices = list(range(20))
        nuisance_indices = []  # Empty!
        cv_splits = generate_cv_splits(n_runs=4, strategy=1)

        results = compute_xval_r2(
            data, design.numpy(), run_starts, stim_indices, nuisance_indices,
            cv_splits, verbose=False
        )

        assert "r2_median" in results
        assert results["r2_median"].shape == (n_voxels,)

    def test_invalid_zero_event_strategy(self):
        """Test that invalid zero_event_strategy raises ValueError"""
        n_voxels, n_timepoints = 50, 200
        data = torch.randn(n_voxels, n_timepoints)
        design = torch.randn(n_timepoints, 30)
        
        # Make some events missing to trigger the strategy code
        design[:100, 5] = 0  # Event 5 missing in first half
        
        run_starts = [0, 100]
        stim_indices = list(range(20))
        nuisance_indices = list(range(20, 30))
        cv_splits = generate_cv_splits(n_runs=2, strategy=1)

        with pytest.raises(ValueError, match="Unknown zero_event_strategy"):
            compute_xval_r2(
                data, design.numpy(), run_starts, stim_indices, nuisance_indices,
                cv_splits, zero_event_strategy="invalid", verbose=False
            )

    def test_no_overlapping_events_error(self):
        """Test that having no overlapping events raises ValueError"""
        n_voxels, n_timepoints = 50, 200
        data = torch.randn(n_voxels, n_timepoints)
        design = torch.randn(n_timepoints, 30)
        
        # Zero out ALL stim events in run 1
        design[:100, :20] = 0
        # Zero out ALL stim events in run 2 (different columns would cause no overlap)
        # Actually, zero them all in both runs to ensure no overlap
        design[100:, :20] = 0
        
        run_starts = [0, 100]
        stim_indices = list(range(20))
        nuisance_indices = list(range(20, 30))
        cv_splits = generate_cv_splits(n_runs=2, strategy=1)

        with pytest.raises(ValueError, match="No overlapping events"):
            compute_xval_r2(
                data, design.numpy(), run_starts, stim_indices, nuisance_indices,
                cv_splits, verbose=False
            )


class TestQRProjectors:
    """Test QR decomposition for nuisance projection"""

    def test_compute_qr_projectors_basic(self):
        """Test basic QR projector computation"""
        from fastfuncsim.xval import compute_qr_projectors

        n_timepoints = 100
        n_nuisance = 5
        n_runs = 3

        # Create nuisance per run
        nuisance_per_run = []
        run_lengths = [30, 40, 30]

        for length in run_lengths:
            nuisance = torch.randn(length, n_nuisance)
            nuisance_per_run.append(nuisance)

        run_starts = [0, 30, 70]

        # Compute QR projectors
        q_factors = compute_qr_projectors(
            nuisance_per_run=nuisance_per_run,
            run_starts=run_starts,
            device=torch.device("cpu")
        )

        # Should return list of Q factors
        assert len(q_factors) == n_runs
        for i, Q in enumerate(q_factors):
            expected_length = run_lengths[i]
            expected_cols = n_nuisance
            assert Q.shape == (expected_length, expected_cols), \
                f"Q[{i}] should have shape ({expected_length}, {expected_cols}), got {Q.shape}"

        # Q should have orthonormal columns
        for i, Q in enumerate(q_factors):
            if Q is not None and Q.shape[1] > 0:  # Check if non-empty
                # Q.T @ Q should be identity
                identity = Q.T @ Q
                assert torch.allclose(identity, torch.eye(Q.shape[1]), atol=1e-4), \
                    f"Q[{i}] columns should be orthonormal"

    def test_compute_qr_projectors_empty_nuisance(self):
        """Test QR projectors with no nuisance regressors"""
        from fastfuncsim.xval import compute_qr_projectors

        n_timepoints = 100
        n_runs = 3

        # Empty nuisance per run
        nuisance_per_run = []
        run_lengths = [30, 40, 30]

        for length in run_lengths:
            nuisance = torch.zeros(length, 0)  # No nuisance columns
            nuisance_per_run.append(nuisance)

        run_starts = [0, 30, 70]

        q_factors = compute_qr_projectors(
            nuisance_per_run=nuisance_per_run,
            run_starts=run_starts,
            device=torch.device("cpu")
        )

        # Should return list of None (one per run)
        assert len(q_factors) == n_runs
        assert all(q is None for q in q_factors)

    def test_compute_qr_projectors_rank_deficient(self):
        """Test QR projectors with rank-deficient nuisance (fewer timepoints than regressors)"""
        from fastfuncsim.xval import compute_qr_projectors

        # Create nuisance where n_regressors > n_timepoints for one run
        n_timepoints = 10
        n_nuisance = 15

        nuisance_per_run = [
            torch.randn(n_timepoints, n_nuisance),  # Rank-deficient
            torch.zeros(n_timepoints, 5),  # All zeros -> None
        ]
        run_starts = [0, n_timepoints]

        # QR should handle rank-deficient case - Q has at most n_timepoints columns
        q_factors = compute_qr_projectors(
            nuisance_per_run=nuisance_per_run,
            run_starts=run_starts,
            device=torch.device("cpu")
        )

        # First run should have Q with rank ≤ n_timepoints
        assert q_factors[0] is not None
        assert q_factors[0].shape[0] == n_timepoints
        # Q should have at most n_timepoints columns (rank-limited)
        assert q_factors[0].shape[1] <= n_timepoints
        # Second run should be None (all zeros)
        assert q_factors[1] is None


class TestProjectOutNuisancePerRun:
    """Test per-run nuisance projection"""

    def test_project_out_nuisance_per_run_basic(self):
        """Test basic per-run nuisance projection"""
        from fastfuncsim.xval import project_out_nuisance_per_run

        n_voxels = 50
        n_timepoints = 100
        n_runs = 2
        n_nuisance = 5

        # Create data
        data = torch.randn(n_voxels, n_timepoints)
        design = torch.randn(n_timepoints, 10)

        # Create nuisance per run
        nuisance_per_run = []
        run_lengths = [50, 50]

        for length in run_lengths:
            nuisance = torch.randn(length, n_nuisance)
            nuisance_per_run.append(nuisance)

        run_starts = [0, 50]

        # Project out nuisance
        data_clean, design_clean = project_out_nuisance_per_run(
            data=data,
            design=design,
            nuisance_per_run=nuisance_per_run,
            run_starts=run_starts,
            device=torch.device("cpu")
        )

        # Check shapes
        assert data_clean.shape == data.shape
        assert design_clean.shape == design.shape

        # Data should have changed (projection did something)
        assert not torch.allclose(data, data_clean)

    def test_project_out_nuisance_per_run_with_polynomials(self):
        """Test per-run projection with Legendre polynomials"""
        from fastfuncsim.xval import project_out_nuisance_per_run
        from fastfuncsim.glm_core import construct_polynomial_matrix

        n_voxels = 50
        n_timepoints = 100
        n_runs = 2

        # Create data with linear drift
        time = torch.arange(n_timepoints, dtype=torch.float32) / n_timepoints
        data = torch.randn(n_voxels, n_timepoints) + 0.5 * time.unsqueeze(0)

        design = torch.randn(n_timepoints, 10)

        # Create polynomials per run (simulating polort=1)
        nuisance_per_run = []
        run_lengths = [50, 50]

        for length in run_lengths:
            poly = construct_polynomial_matrix(length, max_degree=1, device=torch.device("cpu"))
            nuisance_per_run.append(poly)

        run_starts = [0, 50]

        # Project out polynomials
        data_clean, design_clean = project_out_nuisance_per_run(
            data=data,
            design=design,
            nuisance_per_run=nuisance_per_run,
            run_starts=run_starts,
            device=torch.device("cpu")
        )

        # Check that linear trend was reduced
        # Compute correlation with time before and after
        corr_before = (data * time.unsqueeze(0)).sum() / (data.norm() * time.norm())
        corr_after = (data_clean * time.unsqueeze(0)).sum() / (data_clean.norm() * time.norm())

        assert abs(corr_after) < abs(corr_before), \
            f"Linear trend should be reduced: before={corr_before:.3f}, after={corr_after:.3f}"

    def test_project_out_nuisance_per_run_empty_nuisance(self):
        """Test per-run projection with no nuisance"""
        from fastfuncsim.xval import project_out_nuisance_per_run

        n_voxels = 50
        n_timepoints = 100

        data = torch.randn(n_voxels, n_timepoints)
        design = torch.randn(n_timepoints, 10)

        # No nuisance
        nuisance_per_run = [
            torch.zeros(50, 0),
            torch.zeros(50, 0),
        ]
        run_starts = [0, 50]

        # Should return original data and design
        data_clean, design_clean = project_out_nuisance_per_run(
            data=data,
            design=design,
            nuisance_per_run=nuisance_per_run,
            run_starts=run_starts,
            device=torch.device("cpu")
        )

        # Should be unchanged (no nuisance to project out)
        assert torch.allclose(data, data_clean)
        assert torch.allclose(design, design_clean)


class TestComputeR2MetricAdvanced:
    """Test R² metric computation with different metrics"""

    def test_r2_metric_cod(self):
        """Test CoD (coefficient of determination) metric"""
        from fastfuncsim.xval import compute_r2_metric

        n_voxels = 10
        n_timepoints = 50

        # Create simple test case
        y_true = torch.randn(n_voxels, n_timepoints)
        y_pred = y_true + 0.1 * torch.randn_like(y_true)  # Small error

        # Compute CoD
        r2 = compute_r2_metric(y_true, y_pred, metric="cod")

        # Check shape
        assert r2.shape == (n_voxels,)

        # Most voxels should have positive R² (prediction is close to true)
        assert (r2 > 0.5).sum().item() > n_voxels / 2, \
            "Most voxels should have R² > 0.5 with small noise"

    def test_r2_metric_corr(self):
        """Test correlation metric"""
        from fastfuncsim.xval import compute_r2_metric

        n_voxels = 10
        n_timepoints = 50

        y_true = torch.randn(n_voxels, n_timepoints)
        y_pred = y_true + 0.1 * torch.randn_like(y_true)

        # Compute correlation
        r2 = compute_r2_metric(y_true, y_pred, metric="corr")

        # Correlation should be high (close to 1.0)
        assert r2.shape == (n_voxels,)
        assert (r2 > 0.8).sum().item() > n_voxels / 2, \
            "Most voxels should have high correlation with small noise"

    def test_r2_metric_corr2(self):
        """Test squared correlation metric"""
        from fastfuncsim.xval import compute_r2_metric

        n_voxels = 10
        n_timepoints = 50

        y_true = torch.randn(n_voxels, n_timepoints)
        y_pred = y_true + 0.1 * torch.randn_like(y_true)

        # Compute squared correlation
        r2 = compute_r2_metric(y_true, y_pred, metric="corr2")

        # Squared correlation should be very high
        assert r2.shape == (n_voxels,)
        assert (r2 > 0.6).sum().item() > n_voxels / 2, \
            "Most voxels should have high squared correlation with small noise"

    def test_r2_metric_perfect_prediction(self):
        """Test R² with perfect prediction"""
        from fastfuncsim.xval import compute_r2_metric

        n_voxels = 10
        n_timepoints = 50

        y_true = torch.randn(n_voxels, n_timepoints)
        y_pred = y_true.clone()  # Perfect prediction

        # All metrics should give perfect scores
        r2_cod = compute_r2_metric(y_true, y_pred, metric="cod")
        r2_corr = compute_r2_metric(y_true, y_pred, metric="corr")
        r2_corr2 = compute_r2_metric(y_true, y_pred, metric="corr2")

        # CoD might be slightly different due to floating point, but should be near 1
        assert r2_cod.mean().item() > 0.99, "CoD should be ~1.0 for perfect prediction"
        assert r2_corr.mean().item() > 0.999, "Correlation should be ~1.0 for perfect prediction"
        assert r2_corr2.mean().item() > 0.99, "Squared correlation should be ~1.0"

    def test_r2_metric_worst_prediction(self):
        """Test R² with worst-case prediction (unrelated)"""
        from fastfuncsim.xval import compute_r2_metric

        n_voxels = 10
        n_timepoints = 50

        # Use different random seeds to create unrelated data
        y_true = torch.randn(n_voxels, n_timepoints, generator=torch.Generator().manual_seed(42))
        y_pred = torch.randn(n_voxels, n_timepoints, generator=torch.Generator().manual_seed(123))

        # Compute R²
        r2_cod = compute_r2_metric(y_true, y_pred, metric="cod")

        # CoD can be negative (worse than mean)
        # But we just check it returns something reasonable
        assert r2_cod.shape == (n_voxels,)
        assert torch.all(torch.isfinite(r2_cod))


class TestSingleTrialCVHelper:
    """Tests for single_trial_cv_helper batch beta-series CV."""

    @staticmethod
    def _make_betas(n_voxels=50, n_trials=40, n_conditions=10, n_runs=4, snr=2.0):
        """Create synthetic betas with known condition structure."""
        torch.manual_seed(42)
        trials_per_run = n_trials // n_runs
        trial_condition_ids = torch.arange(n_conditions).repeat(n_trials // n_conditions)
        trial_run_ids = torch.arange(n_runs).repeat_interleave(trials_per_run)

        # Ground truth: each condition has a stable pattern across voxels
        true_patterns = torch.randn(n_voxels, n_conditions)
        betas = true_patterns[:, trial_condition_ids] + (1.0 / snr) * torch.randn(n_voxels, n_trials)
        cv_splits = [(
            [r for r in range(n_runs) if r != held],
            [held],
        ) for held in range(n_runs)]
        return betas, trial_condition_ids, trial_run_ids, cv_splits

    def test_single_variant_equivalence(self):
        """single_trial_cv_helper(unsqueeze) == compute_xval_r2_single_trials."""
        from fastfuncsim.xval import compute_xval_r2_single_trials, single_trial_cv_helper

        betas, cids, rids, splits = self._make_betas()

        old = compute_xval_r2_single_trials(
            betas, cids, rids, splits, device=torch.device("cpu"), verbose=False)

        new = single_trial_cv_helper(
            betas.unsqueeze(0), cids, rids, splits,
            zscore_by_run=False, device=torch.device("cpu"), verbose=False)

        torch.testing.assert_close(old["r2"], new["r2"].squeeze(0), atol=1e-5, rtol=1e-5)
        assert old["n_splits"] == new["n_splits"]
        assert old["n_test_trials_total"] == new["n_test_trials_total"]

    def test_multi_variant_shape(self):
        """Output shape is (n_variants, n_voxels) for multiple variants."""
        from fastfuncsim.xval import single_trial_cv_helper

        betas, cids, rids, splits = self._make_betas()
        n_variants = 5
        # Stack identical betas as multiple variants
        multi = betas.unsqueeze(0).expand(n_variants, -1, -1).clone()

        result = single_trial_cv_helper(
            multi, cids, rids, splits,
            device=torch.device("cpu"), verbose=False)

        assert result["r2"].shape == (n_variants, betas.shape[0])
        assert result["r2_mean"].shape == (n_variants,)

    def test_multi_variant_identical_input_gives_same_r2(self):
        """All variants identical (no zscore) → same R² per variant."""
        from fastfuncsim.xval import single_trial_cv_helper

        betas, cids, rids, splits = self._make_betas()
        n_variants = 3
        multi = betas.unsqueeze(0).expand(n_variants, -1, -1).clone()

        result = single_trial_cv_helper(
            multi, cids, rids, splits,
            zscore_by_run=False, device=torch.device("cpu"), verbose=False)

        # All variants should produce identical R²
        for v in range(1, n_variants):
            torch.testing.assert_close(result["r2"][0], result["r2"][v], atol=1e-5, rtol=1e-5)

    def test_zscore_normalization_math(self):
        """Z-scoring normalizes per-run betas using reference variant stats."""
        from fastfuncsim.xval import single_trial_cv_helper

        betas, cids, rids, splits = self._make_betas(snr=5.0)

        # Variant 0: original betas (reference)
        # Variant 1: scaled betas (2x amplitude) — without z-scoring, R² differs
        scaled = betas * 2.0
        multi = torch.stack([betas, scaled], dim=0)

        # Without z-scoring: both should get same R² (CoD is scale-invariant for
        # condition-average predictions, since both prediction and actual scale)
        result_no_z = single_trial_cv_helper(
            multi, cids, rids, splits,
            zscore_by_run=False, device=torch.device("cpu"), verbose=False)

        # With z-scoring from reference (variant 0): variant 1 gets normalized
        result_z = single_trial_cv_helper(
            multi, cids, rids, splits,
            zscore_by_run=True, reference_variant_idx=0,
            device=torch.device("cpu"), verbose=False)

        # Both variants should have valid R² in both cases
        assert result_no_z["r2"].shape == (2, betas.shape[0])
        assert result_z["r2"].shape == (2, betas.shape[0])
        # Mean R² should be positive (good signal)
        assert result_z["r2_mean"][0] > 0.0
        assert result_z["r2_mean"][1] > 0.0

    def test_zscore_reference_variant(self):
        """Z-scoring with different reference variants gives different results."""
        from fastfuncsim.xval import single_trial_cv_helper

        betas, cids, rids, splits = self._make_betas(snr=3.0)

        # Variant 0: original, Variant 1: add run-specific offset
        shifted = betas.clone()
        for r in range(4):
            mask = rids == r
            shifted[:, mask] += r * 0.5  # different offset per run

        multi = torch.stack([betas, shifted], dim=0)

        r_ref0 = single_trial_cv_helper(
            multi, cids, rids, splits,
            zscore_by_run=True, reference_variant_idx=0,
            device=torch.device("cpu"), verbose=False)

        r_ref1 = single_trial_cv_helper(
            multi, cids, rids, splits,
            zscore_by_run=True, reference_variant_idx=1,
            device=torch.device("cpu"), verbose=False)

        # Results should differ because normalization stats come from different variants
        assert not torch.allclose(r_ref0["r2"], r_ref1["r2"], atol=1e-3)

    def test_single_condition(self):
        """Edge case: all trials belong to the same condition."""
        from fastfuncsim.xval import single_trial_cv_helper

        n_voxels, n_trials, n_runs = 20, 16, 4
        torch.manual_seed(0)
        betas = torch.randn(1, n_voxels, n_trials)
        cids = torch.zeros(n_trials, dtype=torch.long)
        rids = torch.arange(n_runs).repeat_interleave(n_trials // n_runs)
        splits = [([r for r in range(n_runs) if r != h], [h]) for h in range(n_runs)]

        result = single_trial_cv_helper(
            betas, cids, rids, splits,
            device=torch.device("cpu"), verbose=False)

        assert result["r2"].shape == (1, n_voxels)
        # Single condition: prediction = grand mean of train → should be mediocre
        assert torch.all(torch.isfinite(result["r2"]))

    def test_zero_variance_voxels(self):
        """Edge case: voxels with zero variance across trials."""
        from fastfuncsim.xval import single_trial_cv_helper

        betas, cids, rids, splits = self._make_betas()

        # Set first 5 voxels to constant
        betas_mod = betas.clone()
        betas_mod[:5, :] = 1.0
        multi = betas_mod.unsqueeze(0)

        result = single_trial_cv_helper(
            multi, cids, rids, splits,
            device=torch.device("cpu"), verbose=False)

        assert result["r2"].shape == (1, betas.shape[0])
        assert torch.all(torch.isfinite(result["r2"]))

    def test_chunked_matches_unchunked(self):
        """Chunked processing gives same result as unchunked."""
        from fastfuncsim.xval import single_trial_cv_helper

        betas, cids, rids, splits = self._make_betas(n_voxels=100)
        multi = betas.unsqueeze(0)

        r_full = single_trial_cv_helper(
            multi, cids, rids, splits,
            chunk_size=None, device=torch.device("cpu"), verbose=False)

        r_chunked = single_trial_cv_helper(
            multi, cids, rids, splits,
            chunk_size=17, device=torch.device("cpu"), verbose=False)

        torch.testing.assert_close(r_full["r2"], r_chunked["r2"], atol=1e-5, rtol=1e-5)

    def test_multi_variant_chunked_matches_unchunked(self):
        """
        Chunked processing with >1 variant gives same result as unchunked.

        This is a regression test for a bug where fold_pred chunks were
        reshaped with (n_variants*n_chunk) then cat on dim=0, causing
        interleaved variant/voxel ordering incompatible with the final
        (n_variants, n_voxels) reshape.  Fixed by keeping chunks as
        (n_variants, n_chunk, n_test) and catting on dim=1.
        """
        from fastfuncsim.xval import single_trial_cv_helper

        betas, cids, rids, splits = self._make_betas(n_voxels=100)
        n_variants = 5
        # Different variants: scale betas by [0.2, 0.4, 0.6, 0.8, 1.0]
        multi = torch.stack([betas * s for s in [0.2, 0.4, 0.6, 0.8, 1.0]])

        r_full = single_trial_cv_helper(
            multi, cids, rids, splits,
            chunk_size=None, device=torch.device("cpu"), verbose=False)

        r_chunked = single_trial_cv_helper(
            multi, cids, rids, splits,
            chunk_size=17, device=torch.device("cpu"), verbose=False)

        assert r_full["r2"].shape == (n_variants, 100)
        assert r_chunked["r2"].shape == (n_variants, 100)
        torch.testing.assert_close(r_full["r2"], r_chunked["r2"], atol=1e-5, rtol=1e-5)

    def test_multi_variant_test_variant_idx_chunked_matches_unchunked(self):
        """
        test_variant_idx mode (GLMsingle pattern) with chunking gives same
        result as unchunked.  Covers the combined multi-variant + test_variant_idx
        + chunked code path.
        """
        from fastfuncsim.xval import single_trial_cv_helper

        betas, cids, rids, splits = self._make_betas(n_voxels=80)
        n_variants = 4
        multi = torch.stack([betas * s for s in [0.3, 0.6, 0.9, 1.0]])

        # Use last variant (frac=1.0 / OLS) as test target
        r_full = single_trial_cv_helper(
            multi, cids, rids, splits,
            test_variant_idx=n_variants - 1,
            chunk_size=None, device=torch.device("cpu"), verbose=False)

        r_chunked = single_trial_cv_helper(
            multi, cids, rids, splits,
            test_variant_idx=n_variants - 1,
            chunk_size=11, device=torch.device("cpu"), verbose=False)

        torch.testing.assert_close(r_full["r2"], r_chunked["r2"], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

