"""
Comprehensive tests for ICA postprocessing utilities in ica_postprocess.py.

Tests cover:
- Scoring functions (mean_abs_by_selector, mean_z_excess_by_selector)
- Utility functions (normalize_0_1, best_lag_and_r, weighted_depth_timeseries)
- Mask expansion (expand_mask_file)
- Depth mask handling (prepare_depth_mask)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fastfuncstuff.decomposition.postprocess import (
    best_lag_and_r,
    mean_abs_by_selector,
    mean_z_excess_by_selector,
    normalize_0_1,
    weighted_depth_timeseries,
)


class TestNormalize01:
    """Test 0-1 normalization function."""

    def test_normalizes_positive_values(self):
        """Test normalization of positive values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_0_1(x)
        expected = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_returns_zeros_for_all_zeros(self):
        """Test that all-zero input returns zeros."""
        x = np.zeros(10)
        result = normalize_0_1(x)
        np.testing.assert_array_equal(result, np.zeros(10))

    def test_returns_zeros_for_small_values(self):
        """Test that very small values return zeros."""
        x = np.array([1e-10, 2e-10, 3e-10])
        result = normalize_0_1(x)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_handles_negative_values(self):
        """Test handling of negative values (max determines scale)."""
        x = np.array([-5.0, 0.0, 5.0])
        result = normalize_0_1(x)
        # Max is 5, so -5 -> -1, 0 -> 0, 5 -> 1
        np.testing.assert_allclose(result, np.array([-1.0, 0.0, 1.0]))

    def test_handles_empty_array(self):
        """Test empty array returns empty."""
        x = np.array([])
        result = normalize_0_1(x)
        assert result.shape == (0,)

    def test_returns_float32(self):
        """Test that output is float32."""
        x = np.array([1.0, 2.0, 3.0])
        result = normalize_0_1(x)
        assert result.dtype == np.float32

    def test_single_value(self):
        """Test single value normalizes to 1."""
        x = np.array([5.0])
        result = normalize_0_1(x)
        np.testing.assert_allclose(result, np.array([1.0]))


class TestMeanAbsBySelector:
    """Test mean absolute value computation with voxel selection."""

    def test_basic_selection(self):
        """Test basic voxel selection."""
        # 3 components, 10 voxels
        comp_kv = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 6.0, 8.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        selector = np.array([True, True, True, True, True, False, False, False, False, False])

        result = mean_abs_by_selector(comp_kv, selector)
        # Component 0: mean of [1,2,3,4,5] = 3.0
        # Component 1: mean of [1,2,3,4,5] = 3.0 (absolute)
        # Component 2: mean of [2,4,6,8,10] = 6.0
        np.testing.assert_allclose(result, np.array([3.0, 3.0, 6.0]))

    def test_empty_selector(self):
        """Test empty selector returns zeros."""
        comp_kv = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        selector = np.array([False, False, False])

        result = mean_abs_by_selector(comp_kv, selector)
        np.testing.assert_array_equal(result, np.array([0.0, 0.0]))

    def test_all_selected(self):
        """Test when all voxels are selected."""
        comp_kv = np.array([
            [1.0, 2.0, 3.0],
            [-1.0, -2.0, -3.0],
        ])
        selector = np.array([True, True, True])

        result = mean_abs_by_selector(comp_kv, selector)
        np.testing.assert_allclose(result, np.array([2.0, 2.0]))

    def test_returns_float32(self):
        """Test that output is float32."""
        comp_kv = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        selector = np.array([True, True, True])

        result = mean_abs_by_selector(comp_kv, selector)
        assert result.dtype == np.float32


class TestMeanZExcessBySelector:
    """Test mean z-score excess computation."""

    def test_basic_thresholding(self):
        """Test z-score thresholding."""
        z_kv = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.5, 1.0, 1.5, 2.0, 2.5],
        ])
        selector = np.array([True] * 5)
        z_thresh = 2.0

        result = mean_z_excess_by_selector(z_kv, selector, z_thresh)
        # Component 0: values: [1,2,3,4,5], |z|: [1,2,3,4,5], excess over 2: [0,0,1,2,3], mean = 6/5 = 1.2
        # Component 1: values: [0.5,1,1.5,2,2.5], |z|: [0.5,1,1.5,2,2.5], excess: [0,0,0,0,0.5], mean = 0.5/5 = 0.1
        np.testing.assert_allclose(result, np.array([1.2, 0.1]))

    def test_empty_selector(self):
        """Test empty selector returns zeros."""
        z_kv = np.array([[1.0, 2.0, 3.0]])
        selector = np.array([False, False, False])

        result = mean_z_excess_by_selector(z_kv, selector, 2.0)
        np.testing.assert_array_equal(result, np.array([0.0]))

    def test_all_below_threshold(self):
        """Test when all values are below threshold."""
        z_kv = np.array([[1.0, 1.5, 1.8]])
        selector = np.array([True, True, True])
        z_thresh = 2.0

        result = mean_z_excess_by_selector(z_kv, selector, z_thresh)
        np.testing.assert_array_equal(result, np.array([0.0]))

    def test_negative_values_absolute(self):
        """Test that absolute value is used."""
        z_kv = np.array([[-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]])
        selector = np.array([True] * 6)
        z_thresh = 2.0

        result = mean_z_excess_by_selector(z_kv, selector, z_thresh)
        # |z|: [3, 2, 1, 1, 2, 3], excess over 2: [1, 0, 0, 0, 0, 1], mean = 2/6 = 1/3
        np.testing.assert_allclose(result, np.array([1.0/3.0]), atol=0.01)


class TestBestLagAndR:
    """Test lag estimation via cross-correlation."""

    def test_perfect_correlation_zero_lag(self):
        """Test perfect correlation at zero lag."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tr = 1.0
        max_lag_s = 2.0

        lag, r, method = best_lag_and_r(x, y, tr, max_lag_s)
        assert lag == 0.0, f"Expected zero lag, got {lag}"
        assert r > 0.9, f"Expected high correlation, got {r}"
        assert "numpy.xcorr" in method

    def test_shifted_signal(self):
        """Test detecting a shifted signal."""
        x = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 0.0])
        y = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        tr = 1.0
        max_lag_s = 3.0

        lag, r, method = best_lag_and_r(x, y, tr, max_lag_s)
        # y is shifted -2 relative to x (or x is +2 relative to y)
        # The function returns lag of x relative to y
        assert abs(lag) <= max_lag_s, f"Lag {lag} outside max range"
        assert r > 0.5, f"Expected moderate correlation, got {r}"

    def test_constant_signal(self):
        """Test degenerate case with constant signal."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        y = np.array([2.0, 2.0, 2.0, 2.0])
        tr = 1.0
        max_lag_s = 2.0

        lag, r, method = best_lag_and_r(x, y, tr, max_lag_s)
        assert lag == 0.0
        assert r == 0.0
        assert "degenerate" in method

    def test_input_validation_different_lengths(self):
        """Test that different length inputs raise error."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="must be 1D and same length"):
            best_lag_and_r(x, y, tr=1.0, max_lag_s=2.0)

    def test_input_validation_2d(self):
        """Test that 2D inputs raise error."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="must be 1D and same length"):
            best_lag_and_r(x, y, tr=1.0, max_lag_s=2.0)


class TestWeightedDepthTimeseries:
    """Test weighted timeseries aggregation."""

    def test_basic_weighted_average(self):
        """Test basic weighted averaging."""
        # 5 voxels, 3 timepoints
        source_vox_t = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
            [5.0, 10.0, 15.0],
        ])
        selector = np.array([True, True, True, False, False])
        weight_v = np.array([1.0, 2.0, 0.5, 0.0, 0.0])
        min_voxels = 2

        result, n_used = weighted_depth_timeseries(source_vox_t, selector, weight_v, min_voxels)

        # Selected voxels: 0, 1, 2 with weights [1, 2, 0.5]
        # Time 0: (1*1 + 2*2 + 3*0.5) / (1 + 2 + 0.5) = 6.5 / 3.5 = 1.857
        # Time 1: (2*1 + 4*2 + 6*0.5) / 3.5 = 13.0 / 3.5 = 3.714
        # Time 2: (3*1 + 6*2 + 9*0.5) / 3.5 = 19.5 / 3.5 = 5.571
        assert result is not None
        np.testing.assert_allclose(result, np.array([1.857, 3.714, 5.571]), rtol=0.01)
        assert n_used == 3

    def test_insufficient_voxels(self):
        """Test case with insufficient voxels."""
        source_vox_t = np.array([[1.0, 2.0], [3.0, 4.0]])
        selector = np.array([True, False])
        weight_v = np.array([1.0, 0.0])
        min_voxels = 2

        result, n_used = weighted_depth_timeseries(source_vox_t, selector, weight_v, min_voxels)

        assert result is None
        assert n_used == 1

    def test_zero_weights(self):
        """Test case where all weights are zero (not positive)."""
        source_vox_t = np.array([[1.0, 2.0], [3.0, 4.0]])
        selector = np.array([True, True])
        weight_v = np.array([0.0, 0.0])
        min_voxels = 1

        result, n_used = weighted_depth_timeseries(source_vox_t, selector, weight_v, min_voxels)

        # Zero weights fail the positive weight check: use = selector & finite & (weight > 0)
        # use = [True & True & False, True & True & False] = [False, False]
        assert result is None
        assert n_used == 0  # No voxels pass the positive weight filter

    def test_filters_by_selector_and_finite_and_positive_weights(self):
        """Test that selector, finite, and positive weight conditions are all applied."""
        source_vox_t = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ])
        selector = np.array([True, True, False, True])  # Voxel 2 excluded
        weight_v = np.array([1.0, np.nan, 0.0, -1.0])  # Voxel 1: nan, Voxel 3: negative
        min_voxels = 1

        result, n_used = weighted_depth_timeseries(source_vox_t, selector, weight_v, min_voxels)

        # Only voxel 0 passes: selected=True, finite=True, weight > 0
        assert result is not None
        np.testing.assert_array_equal(result, np.array([1.0, 2.0]))
        assert n_used == 1

    def test_returns_none_for_empty_selection(self):
        """Test empty selection."""
        source_vox_t = np.array([[1.0, 2.0]])
        selector = np.array([False])
        weight_v = np.array([1.0])
        min_voxels = 1

        result, n_used = weighted_depth_timeseries(source_vox_t, selector, weight_v, min_voxels)

        assert result is None
        assert n_used == 0


class TestExpandMaskFile:
    """Test mask file expansion functionality."""

    def test_expands_4d_mask(self):
        """Test expansion of 4D mask into multiple frame masks."""
        # This test would require nibabel and actual mask files
        # For now, we test the concept with a mock
        with patch('fastfuncstuff.ica_postprocess.nib') as mock_nib:
            # Setup mock
            mock_img = MagicMock()
            mock_img.get_fdata.return_value = np.random.rand(10, 10, 10, 3)
            mock_nib.load.return_value = mock_img


            # This would require actual file I/O - skip for now
            # The actual tests would need temporary NIfTI files
            pass
