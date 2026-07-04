"""Tests for the BSDS ROI-timeseries preprocessing contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.dynamics.preprocess import (
    concat_sessions,
    detrend_session,
    estimate_latent_dim,
    preprocess_sessions,
    standardize_session,
)


def test_detrend_removes_linear_and_constant():
    rng = np.random.default_rng(0)
    n = 200
    t = np.linspace(-1, 1, n)
    # Two ROIs, each = strong linear+constant drift + small signal.
    signal = rng.standard_normal((2, n)) * 0.1
    drift = np.stack([3.0 + 5.0 * t, -2.0 + 4.0 * t])
    y = torch.tensor(signal + drift, dtype=torch.float32)

    out = detrend_session(y, degree=1)
    # Linear+constant gone: near-zero mean and near-zero correlation with t.
    assert out.mean(dim=1).abs().max().item() < 1e-4
    corr = (out * torch.tensor(t, dtype=torch.float32)).mean(dim=1)
    assert corr.abs().max().item() < 1e-3
    # The small signal survives (variance not annihilated).
    assert out.var().item() > 1e-3


def test_detrend_negative_degree_is_noop():
    y = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    out = detrend_session(y, degree=-1)
    assert torch.equal(out, y)


def test_standardize_zscore():
    rng = np.random.default_rng(1)
    y = torch.tensor(rng.standard_normal((5, 300)) * 3 + 7, dtype=torch.float32)
    out = standardize_session(y, "zscore")
    assert out.mean(dim=1).abs().max().item() < 1e-5
    assert (out.std(dim=1, unbiased=False) - 1).abs().max().item() < 1e-5


def test_standardize_constant_roi_no_nan():
    y = torch.ones(3, 50)
    out = standardize_session(y, "zscore")
    assert torch.isfinite(out).all()
    assert out.abs().max().item() < 1e-6  # demeaned constant -> 0, no divide blow-up


def test_preprocess_sessions_shapes_and_validation():
    rng = np.random.default_rng(2)
    sessions = [rng.standard_normal((4, 100)), rng.standard_normal((4, 137))]
    out = preprocess_sessions(sessions, detrend_degree=1, standardize="zscore")
    assert len(out) == 2
    assert out[0].shape == (4, 100)
    assert out[1].shape == (4, 137)
    assert all(isinstance(s, torch.Tensor) for s in out)

    with pytest.raises(ValueError):
        preprocess_sessions([rng.standard_normal((4, 10)), rng.standard_normal((5, 10))])


def test_concat_sessions():
    a = torch.zeros(3, 10)
    b = torch.ones(3, 15)
    cat, lengths = concat_sessions([a, b])
    assert cat.shape == (3, 25)
    assert lengths == [10, 15]


def test_estimate_latent_dim_low_rank():
    rng = np.random.default_rng(0)
    d, n, r = 8, 500, 2
    load = rng.standard_normal((d, r))
    y = load @ rng.standard_normal((r, n)) + rng.standard_normal((d, n)) * 0.01
    dim = estimate_latent_dim([torch.tensor(y)], energy=0.9)
    assert 2 <= dim <= 3  # rank-2 structure -> ~2 components carry 90% variance
    # Capped at D-1 for full-rank noise.
    full = estimate_latent_dim([torch.tensor(rng.standard_normal((5, 400)))], energy=0.99)
    assert full <= 4
