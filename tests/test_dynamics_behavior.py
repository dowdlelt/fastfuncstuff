"""Tests for the brain-behaviour analysis family."""

from __future__ import annotations

import types

import numpy as np

from fastfuncstuff.dynamics.behavior import (
    canonical_correlation,
    cross_validated_prediction,
    loso_classification,
    session_feature_matrix,
)


def _fake_stats(occ, life):
    return types.SimpleNamespace(
        n_states=occ.shape[1],
        subject_occupancy=occ,
        subject_lifetime=life,
    )


def test_feature_matrix_assembles_named_blocks():
    occ = np.arange(6).reshape(3, 2).astype(float)
    life = occ + 10
    x, names = session_feature_matrix(_fake_stats(occ, life))
    assert x.shape == (3, 4)
    assert names == ["occ_S0", "occ_S1", "life_S0", "life_S1"]
    np.testing.assert_array_equal(x[:, :2], occ)


def test_cross_validated_prediction_recovers_planted_signal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 4))
    y = 2.0 * x[:, 0] - 1.5 * x[:, 1] + 0.1 * rng.standard_normal(40)
    res = cross_validated_prediction(x, y, alpha=1.0)
    assert res.r2 > 0.7  # held-out, honest
    assert res.correlation > 0.8
    # the informative features carry the largest weights
    assert np.argmax(np.abs(res.coef)) in (0, 1)


def test_prediction_of_noise_is_not_rewarded():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((30, 5))
    y = rng.standard_normal(30)  # unrelated
    res = cross_validated_prediction(x, y, alpha=1.0)
    assert res.r2 < 0.5  # LOO R^2 near/below zero for pure noise


def test_loso_classification_separable_classes():
    rng = np.random.default_rng(2)
    n = 30
    labels = np.array([0] * n + [1] * n)
    x = np.concatenate(
        [rng.standard_normal((n, 3)), rng.standard_normal((n, 3)) + np.array([4.0, 0, 0])]
    )
    res = loso_classification(x, labels)
    assert res.accuracy > 0.9
    assert res.confusion.shape == (2, 2)


def test_cca_recovers_planted_axis():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((60, 3))
    w = rng.standard_normal(3)
    y0 = x @ w + 0.05 * rng.standard_normal(60)
    y = np.column_stack([y0, rng.standard_normal(60)])
    res = canonical_correlation(x, y, n_components=2)
    assert res.correlations[0] > 0.9
    assert res.correlations[0] >= res.correlations[1]
