"""BSDS correctness comparison: native ffs_bsds vs the MATLAB reference.

Unlike the other benchmark stages this is **not registered in ``ALL_STAGES``**: it
needs a stored MATLAB reference fixture (there is no MATLAB in CI), so it runs as a
standalone comparison the moment a fixture exists. The comparison logic itself is
covered by ``tests/test_bsds_benchmark.py`` (using one native fit as a stand-in
reference), so the harness is validated even without MATLAB.

Fixture format (``.npz``), produced by running the reference
``BayesianSwitchingDynamicalSystems`` MATLAB on the same preprocessed sessions:

- ``viterbi_run_00 .. viterbi_run_RR`` : ``(T_i,)`` reference MAP state per run
  (``model.temporal_evolution_of_states``), OR a single ``viterbi`` for one run.
- ``state_covs`` : ``(K, D, D)`` reference per-state covariance
  (``model.estimated_covariance``).

VB is initialisation-sensitive, so parity is **distributional, not bitwise**: we
score permutation-matched state-recovery accuracy and covariance correlation.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

name = "bsds"
description = "BSDS native vs MATLAB reference (needs a stored fixture)"

THRESHOLDS = {
    "state_recovery_acc": 0.80,  # permutation-matched Viterbi agreement
    "covariance_corr": 0.80,  # mean per-state covariance correlation
}


def permutation_match(pred: np.ndarray, ref: np.ndarray, n_states: int) -> tuple[float, tuple]:
    """Best label-permutation accuracy between two state sequences (K small)."""
    best_acc, best_perm = 0.0, tuple(range(n_states))
    for perm in itertools.permutations(range(n_states)):
        mapped = np.array([perm[p] for p in pred])
        acc = float((mapped == ref).mean())
        if acc > best_acc:
            best_acc, best_perm = acc, perm
    return best_acc, best_perm


def compare_states_and_covariances(
    native_model,
    ref_states: list[np.ndarray],
    ref_state_covs: np.ndarray,
) -> dict:
    """Score a native :class:`BSDSModel` against reference states + covariances."""
    k = native_model.n_states
    pred = np.concatenate([v.cpu().numpy() for v in native_model.viterbi_states])
    ref = np.concatenate([np.asarray(r, dtype=np.int64) for r in ref_states])
    acc, perm = permutation_match(pred, ref, k)

    # Correlate each native state's covariance with its permutation-matched
    # reference (upper triangle including diagonal).
    d = native_model.state_covs.shape[-1]
    iu = np.triu_indices(d)
    native_covs = native_model.state_covs.cpu().numpy()
    corrs = []
    for native_state, ref_state in enumerate(perm):
        a = native_covs[native_state][iu]
        b = np.asarray(ref_state_covs)[ref_state][iu]
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "state_recovery_acc": acc,
        "covariance_corr": float(np.mean(corrs)),
        "permutation": list(perm),
        "per_state_covariance_corr": corrs,
    }


def compare_to_reference(native_model, fixture_path: str | Path) -> dict:
    """Load a MATLAB reference fixture and score the native fit against it.

    Returns the metrics plus a ``passed`` flag per :data:`THRESHOLDS`.
    """
    npz = np.load(fixture_path)
    ref_covs = npz["state_covs"]
    run_keys = sorted(key for key in npz.files if key.startswith("viterbi_run_"))
    if run_keys:
        ref_states = [npz[k] for k in run_keys]
    elif "viterbi" in npz.files:
        ref_states = [npz["viterbi"]]
    else:
        raise ValueError("fixture has no viterbi_run_* or viterbi state sequence")

    metrics = compare_states_and_covariances(native_model, ref_states, ref_covs)
    metrics["passed"] = all(metrics[key] >= thr for key, thr in THRESHOLDS.items())
    metrics["thresholds"] = dict(THRESHOLDS)
    return metrics
