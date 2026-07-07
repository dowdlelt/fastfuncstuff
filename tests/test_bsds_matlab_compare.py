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

    # A distinctive per-state activation profile and transition matrix so the
    # matched activation/transition correlations are meaningful.
    means = rng.standard_normal((k, d))
    trans = np.full((k, k), 0.1) + 0.6 * np.eye(k)
    trans /= trans.sum(1, keepdims=True)
    lifetime = np.array([4.0, 3.0, 2.0, 1.0])

    # MATLAB fit = a relabelling of the same states by perm (state j = ffs perm[j]).
    perm = np.array([2, 0, 3, 1])
    inv = np.array([int(np.where(perm == s)[0][0]) for s in range(k)])  # ffs state -> mat label
    mat = MatlabBSDS(
        occupancy=ffs_occ[perm],
        lifetime=lifetime[perm],
        state_covs=covs[perm],
        state_means=means[perm],
        transition=trans[np.ix_(perm, perm)],
        viterbi_states=[np.array([inv[s] for s in vit])],
    )
    ffs = SimpleNamespace(
        state_covs=torch.tensor(covs),
        state_means=torch.tensor(means),
        transition=torch.tensor(trans),
        viterbi_states=[torch.tensor(vit)],
    )

    res = compare_to_matlab(ffs, mat)
    # ffs state i should match the MATLAB state carrying the same covariance.
    for i in range(k):
        assert res.ffs_to_matlab[i] == int(np.where(perm == i)[0][0])
    assert res.mean_matched_fc > 0.999
    assert res.occupancy_correlation > 0.999
    # The relabelling is exact, so every structural check should be near-perfect.
    assert res.transition_correlation > 0.999
    assert res.activation_correlation > 0.999
    assert res.n_occupied_ffs == k and res.n_occupied_matlab == k
    # Same runs + exact relabelling -> frame-by-frame MAP agreement is total.
    assert res.temporal_agreement == 1.0
    assert res.temporal_kappa > 0.999
    assert res.temporal_frames == vit.size


def test_temporal_agreement_tolerates_length_equalised_export():
    # The MATLAB export truncates runs to the global min, so ffs paths are often a
    # frame or two longer. The comparison must align on the common leading frames,
    # not bail. Here ffs has one extra trailing frame vs MATLAB.
    rng = np.random.default_rng(5)
    k, d = 3, 5
    covs = np.stack([_spd(rng, d) for _ in range(k)])
    means = rng.standard_normal((k, d))
    trans = np.full((k, k), 0.1) + 0.6 * np.eye(k)
    trans /= trans.sum(1, keepdims=True)
    ffs_vit = np.concatenate([np.full(n, s) for s, n in zip(range(k), [20, 15, 11], strict=True)])
    mat_vit = ffs_vit[:-1].copy()  # MATLAB run is one frame shorter (equalised)
    occ = np.bincount(ffs_vit, minlength=k) / ffs_vit.size

    mat = MatlabBSDS(
        occupancy=occ,
        lifetime=np.arange(k) + 1.0,
        state_covs=covs,
        state_means=means,
        transition=trans,
        viterbi_states=[mat_vit],
    )
    ffs = SimpleNamespace(
        state_covs=torch.tensor(covs),
        state_means=torch.tensor(means),
        transition=torch.tensor(trans),
        viterbi_states=[torch.tensor(ffs_vit)],
    )
    res = compare_to_matlab(ffs, mat)
    assert res.temporal_frames == mat_vit.size  # compared on the shorter (common) length
    assert res.temporal_agreement == 1.0


def test_temporal_diagnostics_flags_run_mispairing():
    from types import SimpleNamespace

    from fastfuncstuff.dynamics.bsds.matlab_compare import (
        ComparisonResult,
        MatlabBSDS,
        temporal_diagnostics,
    )

    rng = np.random.default_rng(6)
    k, d, n_runs = 5, 4, 4
    # Distinct MAP path per run (same states, different sequences).
    ffs_paths = [rng.integers(0, k, size=60) for _ in range(n_runs)]
    run_perm = np.array([2, 3, 0, 1])  # MATLAB runs are the ffs runs, permuted
    mat_paths = [ffs_paths[i].copy() for i in run_perm]

    # Identity state mapping (labels already aligned) so mat2ffs is identity.
    result = ComparisonResult(
        ffs_to_matlab={i: i for i in range(k)},
        fc_similarity=np.ones(k),
        ffs_occ=np.ones(k),
        matlab_occ=np.ones(k),
        ffs_state=np.arange(k),
        matlab_state=np.arange(k),
        similarity_matrix=np.eye(k),
        mean_matched_fc=1.0,
        occupancy_correlation=1.0,
        n_occupied_ffs=k,
        n_occupied_matlab=k,
        transition_correlation=1.0,
        lifetime_correlation=1.0,
        activation_correlation=1.0,
        temporal_agreement=0.0,
        temporal_kappa=0.0,
        temporal_frames=0,
    )
    ffs = SimpleNamespace(viterbi_states=[torch.tensor(p) for p in ffs_paths])
    mat = MatlabBSDS(
        occupancy=np.ones(k),
        lifetime=np.ones(k),
        state_covs=np.stack([np.eye(d) for _ in range(k)]),
        state_means=np.zeros((k, d)),
        transition=np.eye(k),
        viterbi_states=mat_paths,
    )

    diag = temporal_diagnostics(ffs, mat, result)
    # Identity pairing is near chance; re-pairing recovers the permutation perfectly.
    assert diag.identity_agreement < 0.6
    assert diag.run_permuted_agreement > 0.999
    assert list(diag.run_permutation) == list(run_perm)
