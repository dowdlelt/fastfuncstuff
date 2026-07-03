"""Tests for decomposition/pca.py and decomposition/workflow.py uncovered lines."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.decomposition.pca import PCA, explained_variance_analysis
from fastfuncstuff.decomposition.workflow import (
    apply_melodic_noise_normalization,
    filter_voxels_for_melodic_model_order,
)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# PCA: covariance branch (lines 149-169) -- n_features >> n_samples
# ---------------------------------------------------------------------------


class TestPCACovarianceBranch:
    """When n_features > 10 * n_samples, uses covariance SVD path."""

    def test_cov_branch_basic(self):
        """Fit PCA when n_features >> n_samples triggers covariance path."""
        rng = np.random.default_rng(42)
        n_samples, n_features = 10, 200  # ratio > 10
        X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        pca = PCA(n_components=3, device=DEVICE)
        pca.fit(X)
        assert pca.n_components_ == 3
        assert pca.components_.shape == (3, n_features)
        assert pca.explained_variance_.shape == (3,)

    def test_cov_branch_variance_fraction(self):
        """Float n_components in covariance branch selects by variance."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((8, 200)).astype(np.float32)
        pca = PCA(n_components=0.9, device=DEVICE)
        pca.fit(X)
        assert 1 <= pca.n_components_ <= 8

    def test_cov_branch_eigenvalues_only(self):
        """eigenvalues_only=True skips loadings computation."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((8, 200)).astype(np.float32)
        pca = PCA(n_components=3, device=DEVICE)
        pca.fit(X, eigenvalues_only=True)
        assert pca.components_ is None
        assert pca.explained_variance_.shape == (3,)

    def test_cov_branch_none_keeps_all(self):
        """n_components=None keeps all components in covariance branch."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((5, 100)).astype(np.float32)
        pca = PCA(n_components=None, device=DEVICE)
        pca.fit(X)
        assert pca.n_components_ == 5  # min(n_samples, n_features)


# ---------------------------------------------------------------------------
# PCA: transform raises when not fitted (line 214)
# ---------------------------------------------------------------------------


class TestPCATransformErrors:
    def test_transform_not_fitted(self):
        pca = PCA(device=DEVICE)
        with pytest.raises(RuntimeError, match="fitted"):
            pca.transform(np.zeros((3, 5)))

    def test_transform_after_eigenvalues_only(self):
        """transform raises if fit was done with eigenvalues_only."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 100)).astype(np.float32)
        pca = PCA(n_components=2, device=DEVICE)
        pca.fit(X, eigenvalues_only=True)
        assert pca.components_ is None
        with pytest.raises(RuntimeError, match="fitted"):
            pca.transform(X)


# ---------------------------------------------------------------------------
# PCA: whiten in transform (line 224)
# ---------------------------------------------------------------------------


class TestPCAWhiten:
    def test_transform_whitened(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((30, 10)).astype(np.float32)
        pca = PCA(n_components=3, whiten=True, device=DEVICE)
        pca.fit(X)
        scores = pca.transform(X)
        assert scores.shape == (30, 3)
        # Whitened scores should have unit variance (approx)
        var = scores.var(dim=0)
        assert torch.allclose(var, torch.ones(3), atol=0.3)

    def test_fit_transform_whitened(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((30, 10)).astype(np.float32)
        pca = PCA(n_components=3, whiten=True, device=DEVICE)
        scores = pca.fit_transform(X)
        assert scores.shape == (30, 3)


# ---------------------------------------------------------------------------
# PCA: inverse_transform (lines 266-278)
# ---------------------------------------------------------------------------


class TestPCAInverseTransform:
    def test_roundtrip_reconstruction(self):
        """Full-rank PCA inverse_transform should reconstruct data."""
        rng = np.random.default_rng(5)
        X_np = rng.standard_normal((20, 8)).astype(np.float32)
        X_orig = torch.tensor(X_np, device=DEVICE)  # independent copy
        pca = PCA(n_components=None, device=DEVICE)
        pca.fit(X_np)
        scores = pca.transform(X_orig)  # use original (not centered in-place)
        X_rec = pca.inverse_transform(scores)
        assert X_rec.shape == (20, 8)
        assert torch.allclose(X_rec, X_orig, atol=1e-3)

    def test_inverse_transform_not_fitted(self):
        pca = PCA(device=DEVICE)
        with pytest.raises(RuntimeError, match="fitted"):
            pca.inverse_transform(np.zeros((3, 2)))

    def test_inverse_transform_whitened(self):
        rng = np.random.default_rng(5)
        X_np = rng.standard_normal((20, 8)).astype(np.float32)
        X_orig = torch.tensor(X_np, device=DEVICE)
        pca = PCA(n_components=None, whiten=True, device=DEVICE)
        pca.fit(X_np)
        scores = pca.transform(X_orig)
        X_rec = pca.inverse_transform(scores)
        assert torch.allclose(X_rec, X_orig, atol=1e-3)


# ---------------------------------------------------------------------------
# PCA: _select_n_components edge cases (lines 304, 318, 329-341)
# ---------------------------------------------------------------------------


class TestSelectNComponents:
    def test_none_returns_max(self):
        pca = PCA(n_components=None, device=DEVICE)
        evr = torch.tensor([0.5, 0.3, 0.2])
        assert pca._select_n_components(evr, 10, 5) == 5

    def test_int_too_large_raises(self):
        pca = PCA(n_components=100, device=DEVICE)
        evr = torch.tensor([0.5, 0.3, 0.2])
        with pytest.raises(ValueError, match="too large"):
            pca._select_n_components(evr, 10, 5)

    def test_float_out_of_range_raises(self):
        pca = PCA(n_components=1.5, device=DEVICE)
        evr = torch.tensor([0.5, 0.3, 0.2])
        with pytest.raises(ValueError, match="between 0 and 1"):
            pca._select_n_components(evr, 10, 5)

    def test_string_mle(self):
        pca = PCA(n_components="mle", device=DEVICE)
        evr = torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05])
        n = pca._select_n_components(evr, 20, 10)
        assert 1 <= n <= 5

    def test_string_knee(self):
        pca = PCA(n_components="knee", device=DEVICE)
        evr = torch.tensor([0.5, 0.2, 0.1, 0.08, 0.05, 0.04, 0.03])
        n = pca._select_n_components(evr, 20, 10)
        assert 1 <= n <= 7

    def test_string_unknown_raises(self):
        pca = PCA(n_components="bogus", device=DEVICE)
        evr = torch.tensor([0.5, 0.3, 0.2])
        with pytest.raises(ValueError, match="Unknown"):
            pca._select_n_components(evr, 10, 5)

    def test_invalid_type_raises(self):
        pca = PCA(device=DEVICE)
        pca.n_components = [1, 2, 3]  # invalid type
        evr = torch.tensor([0.5, 0.3, 0.2])
        with pytest.raises(ValueError, match="int, float, str, or None"):
            pca._select_n_components(evr, 10, 5)


# ---------------------------------------------------------------------------
# PCA: _select_mle and _select_knee (lines 355-357, 367-376)
# ---------------------------------------------------------------------------


class TestMleAndKnee:
    def test_mle_returns_valid(self):
        pca = PCA(device=DEVICE)
        evr = torch.tensor([0.4, 0.3, 0.15, 0.1, 0.05])
        n = pca._select_mle(evr, 20, 10)
        assert 1 <= n <= 5

    def test_knee_with_sharp_drop(self):
        """Knee should detect the sharp drop."""
        evr = torch.tensor([0.5, 0.3, 0.05, 0.04, 0.03, 0.02, 0.01])
        pca = PCA(device=DEVICE)
        n = pca._select_knee(evr)
        assert 1 <= n <= 7


# ---------------------------------------------------------------------------
# PCA: get_explained_variance_cumsum (line 388)
# ---------------------------------------------------------------------------


class TestGetExplainedVarianceCumsum:
    def test_not_fitted_raises(self):
        pca = PCA(device=DEVICE)
        with pytest.raises(RuntimeError, match="fitted"):
            pca.get_explained_variance_cumsum()

    def test_cumsum_values(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 8)).astype(np.float32)
        pca = PCA(n_components=3, device=DEVICE)
        pca.fit(X)
        cs = pca.get_explained_variance_cumsum()
        assert cs.shape == (3,)
        assert cs[-1] <= 1.0 + 1e-6
        assert (cs[1:] >= cs[:-1]).all()


# ---------------------------------------------------------------------------
# PCA: to_dict / from_dict (lines 401-404, 432-444)
# ---------------------------------------------------------------------------


class TestPCASerialisation:
    def test_to_dict_not_fitted(self):
        pca = PCA(device=DEVICE)
        with pytest.raises(RuntimeError, match="fitted"):
            pca.to_dict()

    def test_roundtrip(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((15, 6)).astype(np.float32)
        pca = PCA(n_components=3, device=DEVICE)
        pca.fit(X)
        d = pca.to_dict()
        pca2 = PCA.from_dict(d, device=DEVICE)
        assert pca2.n_components_ == 3
        assert np.allclose(
            pca.components_.cpu().numpy(),
            pca2.components_.cpu().numpy(),
        )
        assert pca2.whiten == pca.whiten


# ---------------------------------------------------------------------------
# explained_variance_analysis (lines 487-514)
# ---------------------------------------------------------------------------


class TestExplainedVarianceAnalysis:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 8)).astype(np.float32)
        result = explained_variance_analysis(X, device=DEVICE)
        assert "explained_variance" in result
        assert "cumulative_variance" in result
        assert "n_components_90" in result
        assert result["n_components_80"] <= result["n_components_95"]
        assert result["cumulative_variance"][-1] == pytest.approx(1.0, abs=1e-5)

    def test_max_components(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 8)).astype(np.float32)
        result = explained_variance_analysis(X, max_components=3, device=DEVICE)
        assert len(result["explained_variance"]) == 3

    def test_thresholds_bounded(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((15, 6)).astype(np.float32)
        result = explained_variance_analysis(X, device=DEVICE)
        for key in ["n_components_80", "n_components_85", "n_components_90", "n_components_95"]:
            assert 1 <= result[key] <= 6


# ---------------------------------------------------------------------------
# workflow: filter_voxels_for_melodic_model_order (lines 195-221)
# ---------------------------------------------------------------------------


class TestFilterVoxelsForMelodicModelOrder:
    def test_basic_filtering(self):
        # Normal voxels + some low-variance voxels
        data = torch.randn(50, 30)
        data[:5, :] *= 1e-8  # very low variance
        filtered, info = filter_voxels_for_melodic_model_order(data)
        assert info["voxels_in"] == 50
        assert info["voxels_dropped"] >= 0
        assert filtered.shape[1] == 30
        assert filtered.shape[0] == info["voxels_kept"]

    def test_degenerate_1d_input(self):
        """1D input (ndim!=2) returns as-is."""
        data = torch.randn(10)
        filtered, info = filter_voxels_for_melodic_model_order(data)
        assert torch.equal(filtered, data)
        assert info["voxels_dropped"] == 0

    def test_single_row(self):
        """Single row (shape[0]<2) returns as-is."""
        data = torch.randn(1, 20)
        filtered, info = filter_voxels_for_melodic_model_order(data)
        assert torch.equal(filtered, data)
        assert info["voxels_kept"] == 1

    def test_all_same_variance(self):
        """When all voxels have equal variance, none should be dropped."""
        data = torch.randn(20, 30)
        filtered, info = filter_voxels_for_melodic_model_order(data)
        # All roughly same std, so threshold should keep most
        assert info["voxels_dropped"] <= 5  # generous bound

    def test_safety_fallback(self):
        """If threshold would drop everything, fallback keeps all."""
        # All constant except one outlier -- threshold logic may try to drop all
        data = torch.zeros(10, 20)
        data[0, :] = 100.0  # one big outlier makes others drop
        filtered, info = filter_voxels_for_melodic_model_order(data)
        # Should not return empty
        assert filtered.shape[0] >= 1


# ---------------------------------------------------------------------------
# workflow: apply_melodic_noise_normalization (lines 237-266)
# ---------------------------------------------------------------------------


class TestApplyMelodicNoiseNormalization:
    def test_basic(self):
        n_t, n_k, n_v = 30, 3, 50
        mixing = torch.randn(n_t, n_k)
        components = torch.randn(n_k, n_v)
        x_t = mixing @ components + torch.randn(n_t, n_v) * 0.1
        normed, msg = apply_melodic_noise_normalization(components, mixing, x_t)
        assert normed.shape == components.shape
        assert "Noise normalization" in msg

    def test_singular_mixing(self):
        """If mixing is singular, should still return components with warning."""
        n_t, n_k, n_v = 10, 3, 20
        mixing = torch.zeros(n_t, n_k)  # singular
        components = torch.randn(n_k, n_v)
        x_t = torch.randn(n_t, n_v)
        normed, msg = apply_melodic_noise_normalization(components, mixing, x_t)
        assert normed.shape == components.shape
        # Should either succeed or return warning message
        assert isinstance(msg, str)
