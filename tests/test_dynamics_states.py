"""Tests for dynamic-state statistics."""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.dynamics.states import (
    covariance_to_correlation,
    dwell_times,
    empirical_transition_matrix,
    fractional_occupancy,
    mean_lifetime,
)


def test_occupancy_and_lifetime_known_sequence():
    # 0 0 0 | 1 1 | 0 0 | 1  -> state0: 5/8, state1: 3/8
    z = np.array([0, 0, 0, 1, 1, 0, 0, 1])
    occ = fractional_occupancy(z, 2)
    np.testing.assert_allclose(occ, [5 / 8, 3 / 8])
    # state0 visits: {000},{00} -> 2 visits, 5 tps -> lifetime 2.5
    # state1 visits: {11},{1}   -> 2 visits, 3 tps -> lifetime 1.5
    life = mean_lifetime(z, 2)
    np.testing.assert_allclose(life, [2.5, 1.5])


def test_mean_lifetime_scales_with_tr():
    z = np.array([0, 0, 1, 1, 1, 0])
    life = mean_lifetime(z, 2, tr=2.0)
    # state0: 3 tps / 2 visits = 1.5 * TR 2 = 3.0 ; state1: 3/1 * 2 = 6.0
    np.testing.assert_allclose(life, [3.0, 6.0])


def test_dwell_times_distribution():
    z = np.array([0, 0, 0, 1, 1, 0, 0, 1])
    dw = dwell_times(z, 2)
    np.testing.assert_array_equal(np.sort(dw[0]), [2, 3])
    np.testing.assert_array_equal(np.sort(dw[1]), [1, 2])


def test_empirical_transition_rows_sum_to_one():
    rng = np.random.default_rng(0)
    z = rng.integers(0, 3, size=500)
    a = empirical_transition_matrix(z, 3)
    np.testing.assert_allclose(a.sum(axis=1), np.ones(3))


def test_absent_state_has_zero_lifetime():
    z = np.array([0, 0, 0, 0])
    life = mean_lifetime(z, 3)
    assert life[1] == 0 and life[2] == 0


def test_covariance_to_correlation():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((4, 4))
    cov = torch.tensor(a @ a.T + np.eye(4))
    corr = covariance_to_correlation(cov)
    torch.testing.assert_close(torch.diagonal(corr), torch.ones(4, dtype=corr.dtype))
    assert corr.abs().max() <= 1.0 + 1e-9
    # Batched form matches per-matrix form.
    batch = covariance_to_correlation(cov.unsqueeze(0))
    torch.testing.assert_close(batch[0], corr)


def test_compute_state_stats_end_to_end():
    import sys

    sys.path.insert(0, "tests")
    from test_bsds_model import _simulate

    from fastfuncstuff.dynamics.states import compute_state_stats

    sessions, _, _, _ = _simulate(k=3, d=6, n_sessions=2, seed=4)
    from fastfuncstuff.dynamics.bsds.model import fit_bsds

    model = fit_bsds(sessions, n_states=3, max_ldim=3, n_init=2, n_init_iter=10, n_iter=50)
    stats = compute_state_stats(model, tr=0.72)
    assert stats.group_occupancy.shape == (3,)
    np.testing.assert_allclose(stats.group_occupancy.sum(), 1.0)
    assert stats.subject_occupancy.shape == (2, 3)
    assert stats.subject_transition.shape == (2, 3, 3)
    assert stats.state_fc.shape == (3, 6, 6)
    torch.testing.assert_close(
        torch.diagonal(stats.state_fc, dim1=-2, dim2=-1),
        torch.ones(3, 6, dtype=stats.state_fc.dtype),
    )
