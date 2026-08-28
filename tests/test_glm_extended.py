"""
Tests for uncovered functions in glm/ modules:
- glm/xval.py: generate_cv_splits, compute_r2_metric (all metrics), metric_higher_is_better
- glm/outputs.py: _ensure_numpy, _resolve_shape, _reshape_parameter_map, write_glm_results_nifti
- glm/ridge.py: edge cases
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.glm.xval import (
    compute_r2_metric,
    generate_cv_splits,
    metric_higher_is_better,
)

DEVICE = torch.device("cpu")


class TestGenerateCvSplits:
    def test_loro_splits(self):
        """LORO with 4 runs should give 4 splits."""
        splits = generate_cv_splits(n_runs=4, strategy=1)
        assert len(splits) == 4
        for train, test in splits:
            assert len(train) == 3
            assert len(test) == 1
            assert set(train + test) == {0, 1, 2, 3}

    def test_leave_2_out(self):
        """Leave-2-out with 4 runs should give 6 splits."""
        splits = generate_cv_splits(n_runs=4, strategy=2)
        assert len(splits) == 6
        for train, test in splits:
            assert len(train) == 2
            assert len(test) == 2

    def test_split_half(self):
        """0.5 fraction = split halves."""
        splits = generate_cv_splits(n_runs=6, strategy=0.5, n_perms=10)
        assert len(splits) > 0
        for train, test in splits:
            assert len(train) == 3
            assert len(test) == 3

    def test_custom_fraction(self):
        """Custom train fraction."""
        splits = generate_cv_splits(n_runs=5, strategy=0.6, n_perms=5)
        assert len(splits) > 0
        for train, test in splits:
            assert len(train) == 3  # 60% of 5 = 3
            assert len(test) == 2

    def test_hundred_run_split_half_samples_directly_with_full_coverage(self):
        """Large split spaces must be sampled, not materialised combinatorially."""
        splits = generate_cv_splits(n_runs=100, strategy=0.5, n_perms=100)
        assert len(splits) == 100
        assert len({tuple(train) for train, _ in splits}) == 100
        assert all(len(train) == 50 and len(test) == 50 for train, test in splits)
        assert set().union(*(set(train) for train, _ in splits)) == set(range(100))
        assert set().union(*(set(test) for _, test in splits)) == set(range(100))

    def test_two_split_halves_are_complementary(self):
        """Two folds should give each run one train and one test appearance."""
        splits = generate_cv_splits(n_runs=100, strategy=0.5, n_perms=2)
        assert len(splits) == 2
        assert set(splits[0][0]) == set(splits[1][1])
        assert set(splits[0][1]) == set(splits[1][0])


class TestComputeR2Metric:
    @pytest.fixture
    def data(self):
        torch.manual_seed(42)
        y_true = torch.randn(50, 200, device=DEVICE)
        y_pred = y_true + 0.1 * torch.randn(50, 200, device=DEVICE)
        return y_true, y_pred

    def test_cod_high_for_good_predictions(self, data):
        y_true, y_pred = data
        r2 = compute_r2_metric(y_true, y_pred, metric="cod")
        assert r2.shape == (50,)
        assert r2.mean() > 0.9

    def test_cod_negative_for_bad_predictions(self):
        y_true = torch.randn(10, 100, device=DEVICE)
        y_pred = -y_true  # Inverted predictions
        r2 = compute_r2_metric(y_true, y_pred, metric="cod")
        assert (r2 < 0).any()

    def test_corr_metric(self, data):
        y_true, y_pred = data
        r = compute_r2_metric(y_true, y_pred, metric="corr")
        assert r.shape == (50,)
        assert (r > 0.9).all()
        assert (r <= 1.0).all()

    def test_corr2_metric(self, data):
        y_true, y_pred = data
        r2 = compute_r2_metric(y_true, y_pred, metric="corr2")
        assert (r2 >= 0).all()
        assert (r2 <= 1.0).all()

    def test_sse_metric(self, data):
        y_true, y_pred = data
        sse = compute_r2_metric(y_true, y_pred, metric="sse")
        assert sse.shape == (50,)
        assert (sse >= 0).all()
        # Better predictions should have lower SSE
        sse_bad = compute_r2_metric(y_true, torch.zeros_like(y_true), metric="sse")
        assert (sse < sse_bad).all()

    def test_unknown_metric_raises(self, data):
        y_true, y_pred = data
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_r2_metric(y_true, y_pred, metric="badmetric")

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_r2_metric(torch.randn(10, 50), torch.randn(10, 60))

    def test_perfect_prediction(self):
        y = torch.randn(5, 100, device=DEVICE)
        r2 = compute_r2_metric(y, y, metric="cod")
        torch.testing.assert_close(r2, torch.ones(5, device=DEVICE), atol=1e-5, rtol=1e-5)


class TestMetricHigherIsBetter:
    def test_cod(self):
        assert metric_higher_is_better("cod") is True

    def test_corr(self):
        assert metric_higher_is_better("corr") is True

    def test_corr2(self):
        assert metric_higher_is_better("corr2") is True

    def test_sse(self):
        assert metric_higher_is_better("sse") is False

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            metric_higher_is_better("badmetric")


class TestEnsureNumpy:
    def test_torch_to_numpy(self):
        from fastfuncstuff.glm.outputs import _ensure_numpy

        t = torch.randn(10, 5)
        arr = _ensure_numpy(t)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32

    def test_numpy_passthrough(self):
        from fastfuncstuff.glm.outputs import _ensure_numpy

        arr = np.ones((5, 3), dtype=np.float32)
        result = _ensure_numpy(arr)
        assert result is arr  # Should be same object

    def test_dtype_conversion(self):
        from fastfuncstuff.glm.outputs import _ensure_numpy

        arr = np.ones((5, 3), dtype=np.float64)
        result = _ensure_numpy(arr)
        assert result.dtype == np.float32


class TestReshapeParameterMap:
    def test_basic_reshape(self):
        from fastfuncstuff.glm.outputs import _reshape_parameter_map

        # 100 voxels, 3 parameters -> (10,10,3) volume
        data = np.random.randn(100, 3).astype(np.float32)
        vol = _reshape_parameter_map(data, volume_shape=(10, 10, 1), voxel_mask=None)
        assert vol.shape == (10, 10, 1, 3)

    def test_with_mask(self):
        from fastfuncstuff.glm.outputs import _reshape_parameter_map

        mask = np.zeros((5, 5, 1), dtype=bool)
        mask[:3, :3, 0] = True  # 9 voxels
        data = np.random.randn(9, 2).astype(np.float32)
        vol = _reshape_parameter_map(data, volume_shape=(5, 5, 1), voxel_mask=mask)
        assert vol.shape == (5, 5, 1, 2)
        # Unmasked voxels should be zero
        assert vol[4, 4, 0, 0] == 0.0

    def test_1d_data(self):
        from fastfuncstuff.glm.outputs import _reshape_parameter_map

        data = np.random.randn(27).astype(np.float32)
        vol = _reshape_parameter_map(data, volume_shape=(3, 3, 3), voxel_mask=None)
        assert vol.shape == (3, 3, 3)
