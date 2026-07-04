"""Tests for the dynamic-state backend interface and the CEBRA export bridge."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from fastfuncstuff.dynamics.backends import DynamicStatesResult, fit_dynamic_states
from fastfuncstuff.dynamics.export import (
    behavior_state_correlation,
    frame_aligned_labels,
    prepare_cebra_inputs,
    state_embedding_separation,
)

sys.path.insert(0, "tests")
from test_bsds_model import _simulate  # noqa: E402


def test_bsds_backend_common_schema():
    sessions, _, _, _ = _simulate(k=3, d=6, n_sessions=2, seed=1)
    res = fit_dynamic_states(
        sessions, n_states=3, backend="bsds", max_ldim=3, n_init=2, n_init_iter=10, n_iter=50
    )
    assert isinstance(res, DynamicStatesResult)
    assert res.backend == "bsds"
    assert res.transition.shape == (3, 3)
    assert res.state_covariances.shape == (3, 6, 6)
    assert len(res.state_timecourse) == 2
    assert res.responsibilities[0].shape == (400, 3)


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        fit_dynamic_states([np.zeros((4, 10))], n_states=2, backend="nope")


def test_osl_backend_missing_dependency_message():
    # osl-dynamics is not a core dependency; the adapter must fail cleanly.
    pytest.importorskip  # keep import used
    try:
        import osl_dynamics  # noqa: F401

        pytest.skip("osl-dynamics is installed; skipping missing-dependency test")
    except ImportError:
        pass
    sessions, _, _, _ = _simulate(k=2, d=4, n_sessions=1, seed=0)
    with pytest.raises(ImportError, match="osl-dynamics"):
        fit_dynamic_states(sessions, n_states=2, backend="osl-hmm")


def test_frame_aligned_labels_and_cebra_inputs():
    import torch

    sessions = [torch.randn(5, 30), torch.randn(5, 20)]
    mat, lengths = prepare_cebra_inputs(sessions)
    assert mat.shape == (50, 5)  # (N_total, D), time-major
    assert lengths == [30, 20]

    res = fit_dynamic_states(
        sessions, n_states=2, backend="bsds", max_ldim=2, n_init=1, n_init_iter=5, n_iter=20
    )
    labels, seq_lengths = frame_aligned_labels(res)
    assert labels.shape[0] == 50
    assert seq_lengths == [30, 20]


def test_state_embedding_separation_high_when_clustered():
    rng = np.random.default_rng(0)
    # Three tight clusters far apart -> high CH; scrambled labels -> low CH.
    centers = np.array([[0, 0], [10, 0], [0, 10]], dtype=float)
    labels = rng.integers(0, 3, size=300)
    emb = centers[labels] + rng.standard_normal((300, 2)) * 0.1
    clustered = state_embedding_separation(emb, labels, n_states=3)["ch_score"]
    scrambled = state_embedding_separation(emb, rng.permutation(labels), n_states=3)["ch_score"]
    assert clustered > 50 * scrambled


def test_behavior_state_correlation():
    rng = np.random.default_rng(1)
    feat = rng.standard_normal((20, 3))  # (S, K) occupancy-like
    behavior = feat[:, 0] * 2.0 + 1.0  # perfectly linear in state-0 occupancy
    r = behavior_state_correlation(feat, behavior)
    assert abs(r[0] - 1.0) < 1e-9
    assert abs(r[1]) < 0.6 and abs(r[2]) < 0.6
