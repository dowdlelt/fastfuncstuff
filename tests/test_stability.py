"""Empirical model order: does it count the components we planted?"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.decomposition.stability import (
    match_components,
    split_half_reproducibility,
    stability_model_order,
)

CPU = torch.device("cpu")


def _sources(k, V, rng):
    """k spatially disjoint, super-Gaussian source maps."""
    S = np.zeros((k, V))
    span = V // k
    for i in range(k):
        S[i, i * span : (i + 1) * span] = rng.laplace(0, 1, span)
    return S


def _dataset(n_runs, T, V, k, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    S = _sources(k, V, rng)
    runs = []
    for _ in range(n_runs):
        A = rng.standard_normal((T, k))
        runs.append((A @ S + noise * rng.standard_normal((T, V))).astype(np.float32))
    return runs, S


def test_seed_stability_saturates_and_does_not_recover_rank():
    """Seed stability is NOT a model-order estimator, and this pins why.

    On rank-6 data run at k_max=12, eleven of twelve clusters clear Iq=0.7 and nine
    exceed 0.97. That is not a threshold that needs tuning: for a *fixed* dataset the
    whitened subspace is fixed too, so FastICA converges to essentially the same solution
    from any initialisation -- including on the noise directions. Restart stability
    measures convergence reproducibility, which is nearly deterministic here, not whether
    a component is real.

    Kept as a test because the temptation to use this curve for model order is strong and
    the failure is silent: the numbers look excellent.
    """
    runs, S = _dataset(n_runs=1, T=100, V=6000, k=6, noise=0.5, seed=1)
    res = stability_model_order(
        torch.tensor(runs[0]), k_max=12, n_runs=12, device=CPU, pca_components=None
    )
    assert res.k > 6, "if this ever tightens to the true rank, revisit the docs"
    assert (res.iq > 0.97).sum() > 6, "noise directions should look just as stable"
    # It remains a usable *diagnostic*: the curve still falls off at the end.
    assert res.iq[0] > res.iq[-1]


def test_stability_result_is_serialisable():
    runs, _ = _dataset(n_runs=1, T=60, V=3000, k=4, seed=2)
    d = stability_model_order(
        torch.tensor(runs[0]), k_max=6, n_runs=6, device=CPU, pca_components=None
    ).as_dict()
    assert set(d) >= {"k", "k_max", "iq", "min_stability", "n_runs"}
    assert isinstance(d["iq"], list)


def test_match_components_is_global_not_greedy():
    """A one-to-one assignment; no map may be claimed twice."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((5, 400))
    # b is a permuted, sign-flipped, noise-added copy of a.
    perm = rng.permutation(5)
    b = -a[perm] + 0.05 * rng.standard_normal((5, 400))

    pairs, r = match_components(a, b)
    assert pairs.shape == (5, 2)
    assert len(set(pairs[:, 0].tolist())) == 5, "a-side index reused"
    assert len(set(pairs[:, 1].tolist())) == 5, "b-side index reused"
    # Sign is arbitrary in ICA, so a sign flip must still match near 1.
    assert r.min() > 0.9
    assert np.all(np.diff(r) <= 1e-12), "not sorted descending"
    # And the recovered permutation is the one we applied.
    recovered = {int(i): int(j) for i, j in pairs}
    assert all(perm[recovered[i]] == i for i in range(5))


def test_split_half_recovers_real_components_and_rejects_noise():
    k = 5
    runs, _ = _dataset(n_runs=6, T=80, V=4000, k=k, noise=0.5, seed=4)
    real = split_half_reproducibility(
        runs, n_components=k, device=CPU, pca_components=None, threshold=0.5
    )
    assert real.n_reproducible >= k - 1, f"only {real.n_reproducible}/{k} reproduced"

    # Pure noise has nothing to reproduce.
    rng = np.random.default_rng(5)
    noise_runs = [rng.standard_normal((80, 4000)).astype(np.float32) for _ in range(6)]
    null = split_half_reproducibility(
        noise_runs, n_components=k, device=CPU, pca_components=None, threshold=0.5
    )
    assert null.n_reproducible <= 1, f"{null.n_reproducible} components 'reproduced' in noise"
    assert real.matched_r.mean() > null.matched_r.mean()


def test_split_half_needs_two_runs():
    runs, _ = _dataset(n_runs=1, T=50, V=1000, k=3, seed=6)
    with pytest.raises(ValueError, match="at least 2 runs"):
        split_half_reproducibility(runs, n_components=3, device=CPU)


def test_split_half_repeated_splits_average():
    runs, _ = _dataset(n_runs=6, T=80, V=3000, k=4, noise=0.5, seed=7)
    res = split_half_reproducibility(
        runs, n_components=4, device=CPU, pca_components=None, n_splits=3
    )
    assert res.n_splits == 3
    assert res.matched_r.size == 4
    assert np.all(np.diff(res.matched_r) <= 1e-12)
    d = res.as_dict()
    assert set(d) >= {"n_reproducible", "matched_r", "threshold", "n_splits"}
