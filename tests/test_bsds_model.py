"""BSDS end-to-end: recover known states/covariances from simulated data."""

from __future__ import annotations

import itertools

import numpy as np
import torch

from fastfuncstuff.dynamics.bsds.model import fit_bsds


def _simulate(k=3, d=6, r=2, t=400, n_sessions=2, stay=0.96, seed=0):
    """Simulate a switching model: per-state mean + low-rank+diagonal covariance.

    Means are separated enough (relative to the covariance spread) that the
    states are recoverable, while each still carries a distinct low-rank
    covariance so the covariance-recovery assertion is meaningful.
    """
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((k, d)) * 4.0  # well-separated state means
    covs = np.empty((k, d, d))
    chols = []
    for j in range(k):
        w = rng.standard_normal((d, r)) * 0.6
        cov = w @ w.T + 0.3 * np.eye(d)
        covs[j] = cov
        chols.append(np.linalg.cholesky(cov))
    trans = np.full((k, k), (1 - stay) / (k - 1))
    np.fill_diagonal(trans, stay)

    sessions, true_states = [], []
    for _ in range(n_sessions):
        z = np.empty(t, dtype=int)
        z[0] = rng.integers(k)
        for i in range(1, t):
            z[i] = rng.choice(k, p=trans[z[i - 1]])
        y = np.empty((d, t))
        for i in range(t):
            y[:, i] = means[z[i]] + chols[z[i]] @ rng.standard_normal(d)
        sessions.append(torch.tensor(y, dtype=torch.float64))
        true_states.append(z)
    return sessions, true_states, means, covs


def _best_accuracy(pred, true, k):
    """Accuracy under the best label permutation (K small -> brute force)."""
    best = 0.0
    best_perm = None
    for perm in itertools.permutations(range(k)):
        mapped = np.array([perm[p] for p in pred])
        acc = (mapped == true).mean()
        if acc > best:
            best, best_perm = acc, perm
    return best, best_perm


def test_recovers_states_and_covariances():
    k, d = 3, 6
    sessions, true_states, _true_means, true_covs = _simulate(k=k, d=d, seed=1)
    model = fit_bsds(sessions, n_states=k, max_ldim=3, n_init=4, n_init_iter=15, n_iter=80, seed=0)

    pred = torch.cat(model.viterbi_states).numpy()
    true = np.concatenate(true_states)
    acc, perm = _best_accuracy(pred, true, k)
    assert acc > 0.85, f"state recovery accuracy too low: {acc:.3f}"

    # State covariances should track the truth once states are matched.
    iu = np.triu_indices(d)
    corrs = []
    for recovered_state, true_state in enumerate(perm):
        rec = model.state_covs[recovered_state].numpy()[iu]
        tru = true_covs[true_state][iu]
        corrs.append(np.corrcoef(rec, tru)[0, 1])
    assert np.mean(corrs) > 0.6, f"covariance recovery weak: {np.mean(corrs):.3f}"


def test_objective_monotonic_and_converges():
    sessions, _, _, _ = _simulate(k=2, d=5, r=2, t=300, n_sessions=2, seed=2)
    model = fit_bsds(sessions, n_states=2, max_ldim=3, n_init=2, n_init_iter=10, n_iter=100, seed=0)
    hist = np.array(model.objective_history)
    diffs = np.diff(hist)
    scale = max(abs(hist[-1]), 1.0)
    # Free energy must be non-decreasing up to tiny numerical slack.
    assert (diffs > -1e-5 * scale).all(), f"ELBO decreased: min diff {diffs.min():.4g}"
    assert model.converged


def test_init_noise_precision_is_ones():
    # Regression: the init noise precision must be ones, matching the reference
    # (vbhafa.m: psii=ones(p,1)). A data-variance seed (psii=1/var) is
    # scale-fragile — on un-standardised, well-separated data it seeds a tiny
    # precision and systematically steers best-of-N restart selection into a
    # higher-free-energy basin with a spurious extra state, so ffs over-segments
    # relative to the reference. psii=ones is scale-robust (and ~identical to
    # 1/var on standardised input where var~1). Scale is instead carried by the
    # mean-loading prior nu_mcl. This deterministic check pins the fix; the
    # behavioural payoff (recovering the true state count in data-rich regimes)
    # is regime-dependent and lives in the scratch/notebook validation.
    from fastfuncstuff.dynamics.bsds.init import init_state

    d = 6
    rng = np.random.default_rng(3)
    # Deliberately non-unit, heterogeneous per-ROI variance so 1/var != ones.
    y = torch.tensor(rng.standard_normal((d, 400)) * rng.uniform(2, 8, (d, 1)), dtype=torch.float64)
    state = init_state(y, [400], n_states=4, ldim=3, seed=0, kmeans_pca_dim=None)
    torch.testing.assert_close(state.psii, torch.ones(d, dtype=torch.float64))
    # nu_mcl (mean-loading prior) still carries scale as 1/var, per the reference.
    assert (state.nu_mcl < 1.0).all()


def test_criterion_free_energy_is_default():
    # Passing criterion="free_energy" explicitly must reproduce the default fit
    # bit-for-bit (the default path is unchanged by the option).
    sessions, _, _, _ = _simulate(k=2, d=5, r=2, t=300, n_sessions=2, seed=2)
    kw = dict(n_states=2, max_ldim=3, n_init=2, n_init_iter=10, n_iter=60, seed=0)
    a = fit_bsds(sessions, **kw)
    b = fit_bsds(sessions, criterion="free_energy", **kw)
    torch.testing.assert_close(a.state_means, b.state_means)
    torch.testing.assert_close(a.transition, b.transition)
    assert a.objective_history == b.objective_history


def test_criterion_weights_converges_and_selects_on_free_energy():
    # The reference vbhafa.m state-mass criterion: stop when per-state occupancy
    # stops moving. On well-separated states with a loose tol it must fire, and
    # the recorded objective history is still the (monotone) free energy used for
    # restart selection.
    sessions, true_states, _, _ = _simulate(k=3, d=6, t=400, n_sessions=2, seed=1)
    model = fit_bsds(
        sessions,
        n_states=3,
        max_ldim=3,
        n_init=3,
        n_init_iter=12,
        n_iter=200,
        tol=0.5,  # absolute L1 occupancy change (scales with #frames)
        criterion="weights",
        seed=0,
    )
    assert model.converged
    hist = np.array(model.objective_history)
    scale = max(abs(hist[-1]), 1.0)
    # Selection/history is free energy regardless of the stopping criterion.
    assert (np.diff(hist) > -1e-5 * scale).all()
    pred = torch.cat(model.viterbi_states).numpy()
    acc, _ = _best_accuracy(pred, np.concatenate(true_states), 3)
    assert acc > 0.85, f"weights-criterion state recovery too low: {acc:.3f}"


def test_shapes_and_symmetry():
    sessions, _, _, _ = _simulate(k=2, d=4, r=1, t=150, n_sessions=1, seed=3)
    model = fit_bsds(sessions, n_states=2, max_ldim=2, n_init=2, n_init_iter=8, n_iter=40)
    assert model.state_means.shape == (2, 4)
    assert model.state_covs.shape == (2, 4, 4)
    assert model.transition.shape == (2, 2)
    torch.testing.assert_close(model.transition.sum(dim=1), torch.ones(2, dtype=torch.float64))
    for cov in model.state_covs:
        torch.testing.assert_close(cov, cov.T)  # symmetric
        assert torch.linalg.eigvalsh(cov).min() > 0  # positive definite
    assert model.ar_transitions.shape == (2, 2, 2)


def test_subject_fit_stays_in_register_with_group():
    from fastfuncstuff.dynamics.bsds.model import fit_subject

    k = 3
    sessions, true_states, _, _ = _simulate(k=k, d=6, n_sessions=3, seed=7)
    group = fit_bsds(sessions, n_states=k, max_ldim=3, n_init=3, n_init_iter=12, n_iter=60)

    # Re-fit one held-in session under the group posterior; states must match the
    # group labelling (no permutation) since it is warm-started from the group.
    subj = fit_subject(sessions[0], group, n_iter=40)
    group_path = group.viterbi_states[0].numpy()
    subj_path = subj.viterbi_states[0].numpy()
    agree = (group_path == subj_path).mean()
    assert agree > 0.8, f"subject fit drifted from group states: {agree:.3f}"
    assert subj.state_covs.shape == (k, 6, 6)


def test_decode_reproduces_training_and_generalizes():
    from fastfuncstuff.dynamics.bsds.model import decode

    k = 3
    sessions, true_states, _, _ = _simulate(k=k, d=6, n_sessions=3, seed=1)
    model = fit_bsds(sessions[:2], n_states=k, max_ldim=3, n_init=3, n_init_iter=12, n_iter=60)

    # Decoding the training runs (fixed params, E-step only) reproduces the fit.
    dec = decode(model, sessions[:2])
    for fit_path, dec_path in zip(model.viterbi_states, dec.viterbi_states, strict=True):
        assert (fit_path.numpy() == dec_path.numpy()).mean() > 0.99

    # A held-out run: right shapes and it still recovers the ground truth.
    dec_h = decode(model, [sessions[2]])
    assert dec_h.responsibilities[0].shape == (sessions[2].shape[1], k)
    acc, _ = _best_accuracy(dec_h.viterbi_states[0].numpy(), true_states[2], k)
    assert acc > 0.8, f"held-out decode accuracy too low: {acc:.3f}"


def test_fit_bsds_auto_ldim():
    sessions, _, _, _ = _simulate(k=2, d=8, r=2, n_sessions=1, seed=0)
    model = fit_bsds(sessions, n_states=2, max_ldim="auto", n_init=1, n_init_iter=5, n_iter=20)
    assert 1 <= model.ldim <= 7
