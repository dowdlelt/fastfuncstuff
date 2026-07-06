"""Init-time k-means seeding: per-session clustering must not degenerate at high D."""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.dynamics.bsds.init import init_state


def _simulate_high_d(k=6, d=80, t=200, n_sessions=6, stay=0.95, seed=0):
    """Well-separated states in high-D ROI space (mimics 'lots of ROIs')."""
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((k, d)) * 5.0
    trans = np.full((k, k), (1 - stay) / (k - 1))
    np.fill_diagonal(trans, stay)
    sessions, truth = [], []
    for _ in range(n_sessions):
        z = np.empty(t, dtype=int)
        z[0] = rng.integers(k)
        for i in range(1, t):
            z[i] = rng.choice(k, p=trans[z[i - 1]])
        y = means[z].T + 0.5 * rng.standard_normal((d, t))
        sessions.append(torch.tensor(y, dtype=torch.float64))
        truth.append(z)
    return sessions, truth


def test_per_session_init_populates_all_states_at_high_d():
    k = 6
    sessions, _ = _simulate_high_d(k=k, d=80, n_sessions=6, seed=1)
    lengths = [s.shape[1] for s in sessions]
    y = torch.cat(sessions, dim=1)

    state = init_state(y, lengths, n_states=k, ldim=4, seed=0)
    occ = state.qns.sum(dim=0).numpy()
    # Every requested state should get a meaningful share of initial responsibility —
    # a joint high-D k-means degenerates to 1-2 populated clusters here.
    assert (occ > 0).all(), f"some states got zero initial responsibility: {occ}"
    min_share = occ.min() / occ.sum()
    assert min_share > 0.02, f"initial responsibilities too skewed: {np.round(occ, 1)}"


def test_kmeans_pca_dim_none_matches_raw_space_at_low_d():
    """At low D, projecting onto D PCs is an isometry, so behaviour should be unaffected."""
    k = 3
    sessions, _ = _simulate_high_d(k=k, d=5, n_sessions=3, t=150, seed=2)
    lengths = [s.shape[1] for s in sessions]
    y = torch.cat(sessions, dim=1)

    state_pca = init_state(y, lengths, n_states=k, ldim=2, seed=0, kmeans_pca_dim=20)
    state_raw = init_state(y, lengths, n_states=k, ldim=2, seed=0, kmeans_pca_dim=None)
    occ_pca = state_pca.qns.sum(dim=0).numpy()
    occ_raw = state_raw.qns.sum(dim=0).numpy()
    assert np.allclose(np.sort(occ_pca), np.sort(occ_raw))
