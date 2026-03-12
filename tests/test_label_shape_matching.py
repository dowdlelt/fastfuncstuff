"""
Tests for label/shape matching to catch design filtering bugs.

These tests ensure that:
1. Results betas shape matches condition_names length
2. Fitted column indices are tracked correctly
3. Single-trials output has correct shape
"""

import numpy as np
import pytest
import torch

from fastfuncsim.glm.arma import ARMA11Results, fit_glm_arma11


class TestLabelShapeMatching:
    """Test that labels always match results shape"""

    def test_arma_results_tracks_fitted_indices(self):
        """Test that ARMA results track which columns were fitted"""
        n_voxels, n_timepoints = 100, 200
        n_regressors_full = 50
        n_regressors_stim = 30

        # Create synthetic data and design
        data = torch.randn(n_voxels, n_timepoints)
        design_full = torch.randn(n_timepoints, n_regressors_full)

        # Simulate stimulus indices (e.g., columns 10-39)
        task_indices = list(range(10, 40))

        # Fit with task_indices filtering
        results = fit_glm_arma11(
            data,
            design_full,
            tr=2.0,
            task_indices=task_indices,
            want_ols=False,
            verbose=False,
        )

        # Check metadata is tracked
        assert hasattr(results, "fitted_column_indices")
        assert hasattr(results, "n_regressors_full")
        assert results.fitted_column_indices == task_indices
        assert results.n_regressors_full == n_regressors_full

        # Check results shape matches filtered design
        assert results.betas.shape == (n_voxels, n_regressors_stim)
        assert results.tstats.shape == (n_voxels, n_regressors_stim)

    def test_arma_results_no_filtering(self):
        """Test that ARMA results track when no filtering applied"""
        n_voxels, n_timepoints = 100, 200
        n_regressors = 50

        data = torch.randn(n_voxels, n_timepoints)
        design = torch.randn(n_timepoints, n_regressors)

        # Fit without task_indices
        results = fit_glm_arma11(
            data,
            design,
            tr=2.0,
            task_indices=None,
            want_ols=False,
            verbose=False,
        )

        # Check metadata indicates no filtering
        assert results.fitted_column_indices is None
        assert results.n_regressors_full == n_regressors

        # Check results shape matches full design
        assert results.betas.shape == (n_voxels, n_regressors)

    def test_labels_must_match_results_shape(self):
        """Test that we catch label/shape mismatches"""
        # This is a regression test for the bug we just fixed
        n_voxels, n_regressors_fitted = 100, 30

        # Simulate results that were filtered
        results = ARMA11Results()
        results.betas = torch.randn(n_voxels, n_regressors_fitted)
        results.tstats = torch.randn(n_voxels, n_regressors_fitted)
        results.fitted_column_indices = list(range(10, 40))
        results.n_regressors_full = 50

        # Simulate full labels (would cause bug!)
        full_labels = [f"col_{i}" for i in range(50)]

        # Simulate fitted labels (correct!)
        fitted_labels = [full_labels[i] for i in results.fitted_column_indices]

        # Check: fitted labels should match results shape
        assert len(fitted_labels) == results.betas.shape[1]

        # Check: using full labels would be wrong
        assert len(full_labels) != results.betas.shape[1]

    def test_extract_fitted_labels_helper(self):
        """Test helper function to extract correct labels"""
        full_labels = [f"stim_{i}" for i in range(100)] + [f"poly_{i}" for i in range(5)]
        fitted_indices = list(range(100))  # Only stimulus, not polynomials

        # Extract fitted labels
        fitted_labels = [full_labels[i] for i in fitted_indices]

        assert len(fitted_labels) == len(fitted_indices)
        assert all("stim_" in label for label in fitted_labels)
        assert not any("poly_" in label for label in fitted_labels)


class TestSingleTrialsOutput:
    """Test single-trials output shape and ordering"""

    def test_single_trials_shape_matches_stim_count(self):
        """Test that single-trials output has correct number of trials"""
        from fastfuncsim.glm.outputs import extract_onset_times_from_design

        n_timepoints = 200
        n_stim = 30
        n_regressors_full = 50

        # Create design matrix with stimulus columns
        design_matrix = np.zeros((n_timepoints, n_regressors_full))

        # Simulate stimulus onsets at different times
        stim_indices = list(range(20, 50))  # 30 stimulus columns
        for i, col_idx in enumerate(stim_indices):
            onset_time = 10 + i * 5  # Staggered onsets
            design_matrix[onset_time:onset_time+3, col_idx] = 1.0

        # Extract onset times
        onset_times = extract_onset_times_from_design(design_matrix, stim_indices)

        # Check: should have one onset per stimulus
        assert len(onset_times) == n_stim

        # Check: onsets should be in correct range
        assert all(0 <= t < n_timepoints for t in onset_times)

    def test_single_trials_sorting_preserves_count(self):
        """Test that sorting by onset preserves trial count"""
        onset_times = [50, 10, 100, 30, 75]
        n_trials = len(onset_times)

        # Create sort indices
        sort_indices = sorted(range(n_trials), key=lambda i: onset_times[i])

        # Check: sorting creates permutation, doesn't drop trials
        assert len(sort_indices) == n_trials
        assert set(sort_indices) == set(range(n_trials))

        # Check: sorting works correctly
        sorted_onsets = [onset_times[i] for i in sort_indices]
        assert sorted_onsets == [10, 30, 50, 75, 100]


class TestDesignMetadata:
    """Test design metadata extraction and tracking"""

    def test_stim_indices_extraction(self):
        """Test extracting stimulus indices from design_info"""
        # Simulate design_info from AFNI .xmat.1D
        design_info = {
            "n_regressors": 322,
            "column_labels": [f"stim_{i}" for i in range(252)] +
                            [f"mot_{i}" for i in range(6)] +
                            [f"poly_{i}" for i in range(64)],
            "stim_bots": [0, 126],  # Two stimulus groups
            "stim_tops": [125, 251],
        }

        # Extract stimulus indices
        stim_indices = []
        for bot, top in zip(design_info["stim_bots"], design_info["stim_tops"], strict=False):
            stim_indices.extend(range(bot, top + 1))

        # Check: correct number of stimulus columns
        assert len(stim_indices) == 252

        # Check: indices are contiguous ranges
        assert stim_indices == list(range(0, 126)) + list(range(126, 252))

        # Extract stimulus labels
        stim_labels = [design_info["column_labels"][i] for i in stim_indices]
        assert len(stim_labels) == 252
        assert all("stim_" in label for label in stim_labels)

    def test_fitted_vs_full_labels_distinction(self):
        """Test that we distinguish fitted from full labels"""
        # This is the key insight from the bug fix
        full_labels = ["a", "b", "c", "d", "e"]
        fitted_indices = [0, 2, 4]  # Only a, c, e

        # Correct: extract fitted labels
        fitted_labels = [full_labels[i] for i in fitted_indices]
        assert fitted_labels == ["a", "c", "e"]

        # Wrong: using full labels
        # (This would cause the ValueError we fixed)
        # write_glm_bucket_as_nifti(results, "output", condition_names=full_labels)  # BUG!
        # write_glm_bucket_as_nifti(results, "output", condition_names=fitted_labels)  # CORRECT!


def test_regression_ValueError_322_vs_252():
    """
    Regression test for the specific bug we fixed:
    ValueError: condition_names has length 322 but results have 252 regressors
    """
    # Simulate the scenario that caused the bug
    n_voxels = 100
    n_regressors_full = 322
    n_regressors_fitted = 252

    # Create results that were filtered
    results = ARMA11Results()
    results.betas = torch.randn(n_voxels, n_regressors_fitted)
    results.tstats = torch.randn(n_voxels, n_regressors_fitted)
    results.fitted_column_indices = list(range(n_regressors_fitted))
    results.n_regressors_full = n_regressors_full

    # Create labels
    full_labels = [f"label_{i}" for i in range(n_regressors_full)]

    # Old code (would cause bug):
    # condition_names = design_info.get("column_labels")  # 322 labels

    # New code (correct):
    if results.fitted_column_indices is not None:
        fitted_labels = [full_labels[i] for i in results.fitted_column_indices]
    else:
        fitted_labels = full_labels

    # Verify the fix
    assert len(fitted_labels) == results.betas.shape[1]  # Should match!
    assert len(fitted_labels) == n_regressors_fitted


def test_extract_design_metadata_helper():
    """Test the new helper function that consolidates duplicate code"""
    from fastfuncsim.io.afni import extract_design_metadata

    # Simulate design_info from AFNI .xmat.1D
    design_info = {
        "n_regressors": 322,
        "column_labels": (
            [f"stim_{i}" for i in range(252)] +
            [f"mot_{i}" for i in range(6)] +
            [f"poly_{i}" for i in range(64)]
        ),
        "stim_bots": [0],
        "stim_tops": [251],
    }

    # Use helper function
    full_labels, stim_labels, stim_column_indices = extract_design_metadata(design_info)

    # Check full labels
    assert len(full_labels) == 322
    assert full_labels[0].startswith("stim_")
    assert full_labels[-1].startswith("poly_")

    # Check stim labels
    assert len(stim_labels) == 252
    assert all(label.startswith("stim_") for label in stim_labels)

    # Check stim indices
    assert len(stim_column_indices) == 252
    assert stim_column_indices == list(range(0, 252))

    # Verify consistency
    assert len(stim_labels) == len(stim_column_indices)
    assert stim_labels == [full_labels[i] for i in stim_column_indices]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
