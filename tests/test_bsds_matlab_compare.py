"""Reading a reference-MATLAB v7.3 model and Hungarian-matching it to an ffs fit."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastfuncstuff.dynamics.bsds.matlab_compare import (
    MatlabBSDS,
    compare_to_matlab,
    load_matlab_bsds,
)

h5py = pytest.importorskip("h5py")


def _spd(rng, d, r=2):
    w = rng.standard_normal((d, r))
    return w @ w.T + np.eye(d)


def _write_fake_v73(path, occ, life, covs, means, trans, viterbi):
    k, d = means.shape
    with h5py.File(path, "w") as f:
        m = f.create_group("model")
        m["fractional_occupancy_group_wise"] = occ.reshape(k, 1)
        m["mean_lifetime_group_wise"] = life.reshape(k, 1)
        m["state_transition_probabilities"] = trans.T  # MATLAB stores transposed
        refs = f.create_group("#refs#")

        def cell(name, arrays, shaper):
            col = np.empty((len(arrays), 1), dtype=h5py.ref_dtype)
            for i, a in enumerate(arrays):
                ds = refs.create_dataset(f"{name}{i}", data=shaper(a))
                col[i, 0] = ds.ref
            m.create_dataset(name, data=col)

        cell("estimated_covariance", list(covs), lambda a: a)
        cell("estimated_mean", list(means), lambda a: a.reshape(1, d))  # MATLAB (1, D)
        cell(
            "temporal_evolution_of_states",
            viterbi,
            lambda v: (v.reshape(-1, 1) + 1).astype(float),  # 1-indexed column
        )


def test_load_matlab_bsds_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    k, d = 4, 6
    covs = np.stack([_spd(rng, d) for _ in range(k)])
    means = rng.standard_normal((k, d))
    occ = np.array([0.4, 0.0, 0.35, 0.25])
    life = np.array([10.0, 0.0, 8.0, 5.0])
    trans = rng.random((k, k))
    trans /= trans.sum(1, keepdims=True)
    viterbi = [rng.integers(0, k, size=50), rng.integers(0, k, size=60)]

    path = tmp_path / "model.mat"
    _write_fake_v73(path, occ, life, covs, means, trans, viterbi)

    mat = load_matlab_bsds(str(path))
    assert mat.state_covs.shape == (k, d, d)
    np.testing.assert_allclose(mat.state_covs, covs, atol=1e-10)
    np.testing.assert_allclose(mat.state_means, means, atol=1e-10)
    np.testing.assert_allclose(mat.occupancy, occ)
    np.testing.assert_allclose(mat.transition, trans, atol=1e-10)  # transposed back
    # States round-trip 0-indexed.
    np.testing.assert_array_equal(mat.viterbi_states[0], viterbi[0])
    assert mat.viterbi_states[0].min() >= 0


def test_compare_recovers_permutation():
    rng = np.random.default_rng(1)
    k, d = 4, 6
    covs = np.stack([_spd(rng, d) for _ in range(k)])
    # One run whose blocks define occupancy for the "ffs" fit.
    vit = np.concatenate([np.full(n, s) for s, n in zip(range(k), [40, 30, 20, 10], strict=True)])
    ffs_occ = np.bincount(vit, minlength=k) / vit.size

    # MATLAB fit = a relabelling of the same states by perm (state j = ffs perm[j]).
    perm = np.array([2, 0, 3, 1])
    mat = MatlabBSDS(
        occupancy=ffs_occ[perm],
        lifetime=np.zeros(k),
        state_covs=covs[perm],
        state_means=np.zeros((k, d)),
        transition=np.eye(k),
        viterbi_states=[np.array([int(np.where(perm == s)[0][0]) for s in vit])],
    )
    ffs = SimpleNamespace(state_covs=torch.tensor(covs), viterbi_states=[torch.tensor(vit)])

    res = compare_to_matlab(ffs, mat)
    # ffs state i should match the MATLAB state carrying the same covariance.
    for i in range(k):
        assert res.ffs_to_matlab[i] == int(np.where(perm == i)[0][0])
    assert res.mean_matched_fc > 0.999
    assert res.occupancy_correlation > 0.999
