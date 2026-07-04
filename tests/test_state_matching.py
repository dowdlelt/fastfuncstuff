"""Tests for cross-fit BSDS state matching."""

from __future__ import annotations

import sys
import types

import numpy as np
import torch

from fastfuncstuff.dynamics.matching import (
    gaussian_symmetric_kl,
    match_states,
    temporal_similarity_matrix,
)


def _fake_model(means, covs):
    return types.SimpleNamespace(
        n_states=means.shape[0],
        state_means=means,
        state_covs=covs,
    )


def _distinct_states(k=3, d=5, seed=0):
    rng = np.random.default_rng(seed)
    means = torch.tensor(rng.standard_normal((k, d)) * 5.0)
    covs = torch.empty(k, d, d, dtype=torch.float64)
    for j in range(k):
        w = torch.tensor(rng.standard_normal((d, d)))
        covs[j] = w @ w.T + torch.eye(d)
    return means, covs


def test_symmetric_kl_zero_for_identical():
    means, covs = _distinct_states()
    kl = gaussian_symmetric_kl(means[0], covs[0], means[0], covs[0])
    assert abs(float(kl)) < 1e-8
    kl2 = gaussian_symmetric_kl(means[0], covs[0], means[1], covs[1])
    assert float(kl2) > 0


def test_state_space_matching_recovers_permutation():
    means, covs = _distinct_states(k=4, d=6, seed=1)
    perm = np.array([2, 0, 3, 1])
    a = _fake_model(means, covs)
    b = _fake_model(means[perm], covs[perm])
    m = match_states(a, b, method="state_space")
    # A-state i corresponds to B-state where perm placed it: perm.index(i).
    inverse = np.argsort(perm)
    np.testing.assert_array_equal(m.col_ind[np.argsort(m.row_ind)], inverse)
    assert m.match_for(0) == int(inverse[0])


def test_temporal_matching_recovers_permutation():
    rng = np.random.default_rng(2)
    resp = torch.tensor(rng.dirichlet(np.ones(3), size=200))  # (T, 3)
    perm = [1, 2, 0]
    resp_b = resp[:, perm]  # B-column j is A-column perm[j]
    # So A-state i best matches B-state argsort(perm)[i] (the inverse permutation).
    expected = np.argsort(perm)
    sim = temporal_similarity_matrix([resp], [resp_b])
    # Diagonal of the permuted similarity should be ~1 at matched pairs.
    m = match_states(
        _fake_model(*_distinct_states(k=3)),  # unused params for temporal
        _fake_model(*_distinct_states(k=3)),
        method="temporal",
        resp_a=[resp],
        resp_b=[resp_b],
    )
    ordered = m.col_ind[np.argsort(m.row_ind)]
    np.testing.assert_array_equal(ordered, expected)
    assert sim.shape == (3, 3)


def test_two_fits_agree_on_parameter_and_temporal_matching():
    sys.path.insert(0, "tests")
    from test_bsds_model import _simulate

    from fastfuncstuff.dynamics.bsds.model import fit_bsds

    sessions, _, _, _ = _simulate(k=3, d=6, n_sessions=2, seed=5)
    a = fit_bsds(sessions, n_states=3, max_ldim=3, n_init=3, n_init_iter=12, n_iter=60, seed=0)
    b = fit_bsds(sessions, n_states=3, max_ldim=3, n_init=3, n_init_iter=12, n_iter=60, seed=11)

    by_param = match_states(a, b, method="state_space")
    by_time = match_states(
        a, b, method="temporal", resp_a=a.responsibilities, resp_b=b.responsibilities
    )
    # Independent criteria should agree on how the two fits' states line up.
    param_map = by_param.col_ind[np.argsort(by_param.row_ind)]
    time_map = by_time.col_ind[np.argsort(by_time.row_ind)]
    np.testing.assert_array_equal(param_map, time_map)
