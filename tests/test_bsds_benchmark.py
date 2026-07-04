"""Tests for the BSDS-vs-reference comparison harness.

No MATLAB here, so we validate the harness itself by using one native fit as a
stand-in "reference" and checking that a second fit of the same data scores high.
"""

from __future__ import annotations

import sys

import numpy as np

from fastfuncstuff.benchmark.stages.bsds import (
    compare_states_and_covariances,
    compare_to_reference,
    permutation_match,
)
from fastfuncstuff.dynamics.bsds.model import fit_bsds

sys.path.insert(0, "tests")
from test_bsds_model import _simulate  # noqa: E402


def test_permutation_match_recovers_relabeling():
    ref = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    perm = (1, 2, 0)
    pred = np.array([perm.index(r) for r in ref])  # relabel
    acc, found = permutation_match(pred, ref, 3)
    assert acc == 1.0
    # applying `found` to pred must reproduce ref
    assert np.array_equal(np.array([found[p] for p in pred]), ref)


def test_compare_two_fits_scores_high(tmp_path):
    sessions, _, _, _ = _simulate(k=3, d=6, n_sessions=2, seed=1)
    ref_model = fit_bsds(sessions, n_states=3, max_ldim=3, n_init=3, n_init_iter=12, n_iter=70)
    native = fit_bsds(sessions, n_states=3, max_ldim=3, n_init=3, n_init_iter=12, n_iter=70, seed=5)

    ref_states = [v.numpy() for v in ref_model.viterbi_states]
    metrics = compare_states_and_covariances(native, ref_states, ref_model.state_covs.numpy())
    assert metrics["state_recovery_acc"] > 0.85
    assert metrics["covariance_corr"] > 0.8

    # Round-trip through a saved fixture, as compare_to_reference expects.
    fixture = tmp_path / "ref.npz"
    np.savez(
        fixture,
        state_covs=ref_model.state_covs.numpy(),
        viterbi_run_00=ref_states[0],
        viterbi_run_01=ref_states[1],
    )
    result = compare_to_reference(native, fixture)
    assert result["passed"]
    assert set(result["thresholds"]) == {"state_recovery_acc", "covariance_corr"}
