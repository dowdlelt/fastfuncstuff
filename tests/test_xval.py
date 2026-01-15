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
        assert "r2_median" in results
        assert "r2_std" in results
        assert "r2_min" in results
        assert "r2_max" in results
        assert "r2_splits" in results

        # Check shapes
        assert results["r2_median"].shape == (n_voxels,)
        assert results["r2_std"].shape == (n_voxels,)
        assert results["r2_splits"].shape == (len(cv_splits), n_voxels)

        # Sanity checks on values
        # With low noise, R² should be high
        assert results["r2_median"].mean() > 0.7

        # Cross-validated R² should be positive for good predictions
        assert (results["r2_median"] > 0).sum() == n_voxels

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
    assert all(key in results for key in ["r2_median", "r2_std", "r2_min", "r2_max"])

    # Validate statistics make sense
    # Median should be between min and max
    assert (results["r2_median"] >= results["r2_min"] - 1e-5).all()
    assert (results["r2_median"] <= results["r2_max"] + 1e-5).all()

    # Std should be non-negative
    assert (results["r2_std"] >= 0).all()


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

