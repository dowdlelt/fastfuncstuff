"""Restart-stability diagnostic: refit and match states by FC across seeds."""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.dynamics.bsds.fc_match import fc_similarity_matrix, hungarian_match
from fastfuncstuff.dynamics.stability import state_stability


def test_fc_similarity_and_hungarian_recover_permutation():
    rng = np.random.default_rng(0)
    d, k = 6, 4
    covs = np.stack([(w := rng.standard_normal((d, 3))) @ w.T + np.eye(d) for _ in range(k)])
    perm = np.array([2, 0, 3, 1])
    sim = fc_similarity_matrix(covs, covs[perm])
    row, col = hungarian_match(sim)
    # state i in A should match the column carrying its own covariance.
    matched = {int(r): int(c) for r, c in zip(row, col, strict=True)}
    for i in range(k):
        assert matched[i] == int(np.where(perm == i)[0][0])


def test_state_stability_on_structured_data():
    # Two well-separated regimes so independent fits should agree strongly.
    rng = np.random.default_rng(1)
    d = 8
    a = rng.standard_normal((d, 2))
    b = rng.standard_normal((d, 2))
    sessions = []
    for _ in range(4):
        blocks = []
        for cov_factor in (a, b, a, b):
            z = rng.standard_normal((2, 40))
            blocks.append(cov_factor @ z + 0.1 * rng.standard_normal((d, 40)))
        sessions.append(torch.tensor(np.concatenate(blocks, axis=1), dtype=torch.float64))

    result = state_stability(
        sessions, n_states=3, max_ldim=2, n_repeats=3, n_init=2, n_iter=25, show_progress=False
    )
    assert result.per_state_fc.shape == (3,)
    assert result.matched_occupancy.shape == (2, 3)
    # The occupied reference states should be recovered with high FC similarity.
    occupied = result.reference_occupancy > 1e-3
    assert np.nanmin(result.per_state_fc[occupied]) > 0.5
    assert -1.0 <= result.mean_fc <= 1.0


def test_repeats_are_genuinely_different_inits():
    # Guards the overlapping-restart-seed bug: with n_init>1, consecutive fit seeds
    # would share restart pools and every repeat could return the identical fit
    # (spurious perfect agreement). On ambiguous data, strided seeds must produce
    # non-identical fits, so matched-FC is not exactly 1.
    rng = np.random.default_rng(2)
    sessions = [torch.tensor(rng.standard_normal((10, 80)), dtype=torch.float64) for _ in range(3)]
    result = state_stability(
        sessions, n_states=4, max_ldim=3, n_repeats=3, n_init=3, n_iter=20, show_progress=False
    )
    assert result.mean_fc < 0.999
