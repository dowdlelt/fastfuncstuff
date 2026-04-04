import numpy as np
import pytest
import torch

from fastfuncstuff.stats.spatial import (
    consistency_report,
    optimal_matching,
    one_to_many_correlation,
    spatial_correlation,
    spatial_correlation_matrix,
)

DEVICE = torch.device("cpu")
SHAPE = (5, 5, 5)


def _rand_vol(seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(SHAPE, generator=gen, device=DEVICE)


def _rand_4d(n, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, *SHAPE, generator=gen, device=DEVICE)


class TestSpatialCorrelation:
    def test_spatial_correlation_identical(self):
        a = _rand_vol(42)
        r = spatial_correlation(a, a, method="pearson", device=DEVICE)
        assert r == pytest.approx(1.0, abs=0.01)

    def test_spatial_correlation_anticorrelated(self):
        a = _rand_vol(42)
        r = spatial_correlation(a, -a, method="pearson", device=DEVICE)
        assert r == pytest.approx(-1.0, abs=0.01)

    def test_spatial_correlation_spearman_identical(self):
        a = _rand_vol(42)
        r = spatial_correlation(a, a, method="spearman", device=DEVICE)
        assert r == pytest.approx(1.0, abs=0.01)


class TestSpatialCorrelationMatrix:
    def test_spatial_correlation_matrix_shape(self):
        a = _rand_4d(3, seed=0)
        b = _rand_4d(4, seed=1)
        mat = spatial_correlation_matrix(a, b, method="pearson", device=DEVICE)
        assert mat.shape == (3, 4)

    def test_spatial_correlation_matrix_diagonal(self):
        a = _rand_4d(4, seed=7)
        mat = spatial_correlation_matrix(a, a, method="pearson", device=DEVICE)
        for i in range(4):
            assert mat[i, i] == pytest.approx(1.0, abs=0.01)


class TestOneToManyCorrelation:
    def test_one_to_many_correlation(self):
        ref = _rand_vol(10)
        imgs = _rand_4d(3, seed=20)
        r = one_to_many_correlation(ref, imgs, method="pearson", device=DEVICE)
        assert r.shape == (3,)
        full = spatial_correlation_matrix(ref.unsqueeze(0), imgs, method="pearson", device=DEVICE)
        np.testing.assert_allclose(r, full[0], atol=1e-7)


class TestOptimalMatching:
    def test_optimal_matching_identity_matrix(self):
        corr = np.eye(4, dtype=np.float64)
        rows, cols, corrs = optimal_matching(corr)
        pairs = sorted(zip(rows.tolist(), cols.tolist()))
        expected = [(i, i) for i in range(4)]
        assert pairs == expected
        np.testing.assert_allclose(corrs, np.ones(4), atol=1e-12)


class TestConsistencyReport:
    def test_consistency_report_str(self):
        corr = np.eye(3, dtype=np.float64)
        report = consistency_report(corr, method="pearson")
        text = str(report)
        assert "Consistency Report (pearson)" in text
        assert "3 volumes" in text
        assert "Mean matched r:" in text
        assert "Coverage:" in text
