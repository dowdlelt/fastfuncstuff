"""BSDS hyperparameter selection via held-out log-likelihood."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.dynamics.model_selection import grid_search_bsds, loro_held_out_loglik


def _simulate(k=3, d=5, r=1, t=120, n_sessions=4, stay=0.95, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((k, d)) * 4.0
    chols = []
    for _ in range(k):
        w = rng.standard_normal((d, r)) * 0.5
        chols.append(np.linalg.cholesky(w @ w.T + 0.3 * np.eye(d)))
    trans = np.full((k, k), (1 - stay) / (k - 1))
    np.fill_diagonal(trans, stay)
    sessions = []
    for _ in range(n_sessions):
        z = np.empty(t, dtype=int)
        z[0] = rng.integers(k)
        for i in range(1, t):
            z[i] = rng.choice(k, p=trans[z[i - 1]])
        y = np.stack([means[z[i]] + chols[z[i]] @ rng.standard_normal(d) for i in range(t)], axis=1)
        sessions.append(torch.tensor(y, dtype=torch.float64))
    return sessions


def test_loro_held_out_loglik_runs_and_scales_with_frames():
    sessions = _simulate(n_sessions=3, t=100, seed=0)
    result = loro_held_out_loglik(
        sessions, n_states=3, max_ldim=2, n_folds=3, n_init=2, n_init_iter=5, n_iter=20
    )
    assert result.n_states == 3
    assert result.max_ldim == 2
    assert len(result.fold_logliks) == 3
    assert np.isfinite(result.held_out_loglik)
    assert np.isfinite(result.per_timepoint_loglik)


def test_grid_search_bsds_sorted_best_first():
    sessions = _simulate(n_sessions=3, t=100, seed=1)
    results = grid_search_bsds(
        sessions,
        n_states_grid=[2, 3],
        max_ldim_grid=[1, 2],
        n_folds=3,
        n_init=2,
        n_init_iter=5,
        n_iter=20,
        show_progress=False,
    )
    assert len(results) == 4
    scores = [r.per_timepoint_loglik for r in results]
    assert scores == sorted(scores, reverse=True)
    combos = {(r.n_states, r.max_ldim) for r in results}
    assert combos == {(2, 1), (2, 2), (3, 1), (3, 2)}


@pytest.mark.slow
def test_grid_search_parallel_matches_serial():
    # n_jobs>1 fans grid points across a spawn process pool. Each grid point is
    # deterministic in its seed, so the parallel result set must match the serial
    # one bit-for-bit (only completion order differs, and the list is sorted).
    sessions = _simulate(n_sessions=3, t=100, seed=1)
    kw = dict(
        n_states_grid=[2, 3],
        max_ldim_grid=[1, 2],
        n_folds=3,
        n_init=2,
        n_init_iter=5,
        n_iter=20,
        show_progress=False,
    )
    serial = grid_search_bsds(sessions, n_jobs=1, **kw)
    # worker_threads=1 keeps the 3-worker pool from fanning across all cores — the
    # test verifies parallel==serial correctness, not CPU saturation.
    parallel = grid_search_bsds(sessions, n_jobs=3, worker_threads=1, **kw)
    serial_map = {(r.n_states, r.max_ldim): r.per_timepoint_loglik for r in serial}
    parallel_map = {(r.n_states, r.max_ldim): r.per_timepoint_loglik for r in parallel}
    assert serial_map.keys() == parallel_map.keys()
    for key in serial_map:
        assert serial_map[key] == parallel_map[key], f"parallel diverged at {key}"
