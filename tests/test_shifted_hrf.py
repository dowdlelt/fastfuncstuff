"""Tests for per-trial amplitude + latency by exact HRF shifting."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.design.flobs import (
    fit_basis_constrained_ridge,
    generate_spmg_basis,
    spmg_prior,
)
from fastfuncstuff.design.hrf import get_spm_hrf_with_derivatives
from fastfuncstuff.design.hrf_derive import build_pc_basis_design_per_run
from fastfuncstuff.design.shifted_hrf import build_shifted_design_bank, fit_shifted_hrf

TR, NTP, DT = 1.5, 200, 0.1
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def setup():
    h = (
        get_spm_hrf_with_derivatives(
            microtime_dt=DT, hrf_duration=32.0, n_basis=1, device=CPU
        )
        .cpu()
        .numpy()[0]
    )
    h = h / h.max()
    onsets = np.arange(10.0, NTP * TR - 40, 12.0)
    block_onsets = [np.array([o]) for o in onsets]
    polys = np.polynomial.legendre.legvander(np.linspace(-1, 1, NTP), 2)
    return h, onsets, block_onsets, polys


def _simulate(h, block_onsets, tau, amps):
    """Exact-shift simulation on a grid fine enough to be ~continuous."""
    fine = np.arange(-3.0, 3.001, 0.05)
    bank = build_shifted_design_bank(
        block_onsets, h, DT, fine, TR, NTP, device=CPU
    ).numpy()
    nv, nb = tau.shape
    y = np.zeros((nv, NTP))
    for v in range(nv):
        for b in range(nb):
            g = int(np.argmin(np.abs(fine - tau[v, b])))
            y[v] += amps[v, b] * bank[b, g]
    return y


def test_bank_shift_is_exact(setup):
    """C[b, g] must equal the HRF sampled at (t - onset - tau), not a Taylor step."""
    h, onsets, block_onsets, _ = setup
    tau_grid = np.array([-1.7, 0.0, 2.3])
    bank = build_shifted_design_bank(
        block_onsets, h, DT, tau_grid, TR, NTP, device=CPU
    ).numpy()
    t_tr = np.arange(NTP) * TR
    hrf_t = np.arange(h.size) * DT
    for b, o in enumerate(onsets):
        for g, tau in enumerate(tau_grid):
            want = np.interp(t_tr - o - tau, hrf_t, h, left=0.0, right=0.0)
            np.testing.assert_allclose(bank[b, g], want, rtol=1e-10, atol=1e-12)


def test_noiseless_recovery_is_exact(setup):
    """On-grid truth with no noise: A and tau recovered essentially exactly."""
    h, _, block_onsets, polys = setup
    nb = len(block_onsets)
    rng = np.random.default_rng(0)
    # truth placed exactly on the search grid so there is no quantisation floor
    tau = rng.choice(np.arange(-2.0, 2.01, 0.5), size=(20, nb))
    amps = rng.uniform(0.5, 3.0, size=(20, nb))
    y = _simulate(h, block_onsets, tau, amps)
    r = fit_shifted_hrf(
        data=torch.from_numpy(y),
        block_onsets=block_onsets,
        hrf=h,
        hrf_dt=DT,
        tr=TR,
        nuisance=torch.from_numpy(polys),
        tau_max=3.0,
        tau_step=0.25,
        n_sweeps=4,
        delay_prior_sd=None,  # no shrinkage: we want the raw optimum
        device=CPU,
    )
    assert np.median(r.r2) > 0.999, f"noiseless R2 only {np.median(r.r2):.4f}"
    # tau exact on the vast majority of trials, amplitude close everywhere
    assert np.mean(np.abs(r.delays - tau) < 1e-6) > 0.9
    np.testing.assert_allclose(np.median(r.amplitudes / amps), 1.0, rtol=0.02)


def test_delays_never_exceed_the_bound(setup):
    """The box bound is a hard invariant even when the truth is far outside it."""
    h, _, block_onsets, polys = setup
    nb = len(block_onsets)
    y = _simulate(h, block_onsets, np.full((10, nb), 2.9), np.ones((10, nb)))
    r = fit_shifted_hrf(
        data=torch.from_numpy(y),
        block_onsets=block_onsets,
        hrf=h,
        hrf_dt=DT,
        tr=TR,
        nuisance=torch.from_numpy(polys),
        tau_max=1.0,
        tau_step=0.25,
        delay_prior_sd=None,
        device=CPU,
    )
    assert r.delays.max() <= 1.0 + 1e-9
    assert r.delays.min() >= -1.0 - 1e-9


def test_truth_outside_the_bound_oscillates_amplitudes(setup):
    """Known limitation, recorded so a regression would show up as a change.

    When the true latency is far outside the search bound the fit does NOT
    degrade gracefully to "clamped at the edge with a slightly small
    amplitude".  The HRF cannot be placed correctly, so the joint
    amplitude solve compensates using overlapping neighbours, producing
    alternating signed amplitudes across trials -- the same nonsensical
    output the whole approach is meant to prevent, re-entering through the
    amplitude solve rather than the latency parameter.  The practical
    consequence: tau_max must be set wide enough to contain the real
    latency spread, and the tau prior should stay on.
    """
    h, _, block_onsets, polys = setup
    nb = len(block_onsets)
    y = _simulate(h, block_onsets, np.full((10, nb), 2.9), np.ones((10, nb)))
    r = fit_shifted_hrf(
        data=torch.from_numpy(y),
        block_onsets=block_onsets,
        hrf=h,
        hrf_dt=DT,
        tr=TR,
        nuisance=torch.from_numpy(polys),
        tau_max=1.0,
        tau_step=0.25,
        delay_prior_sd=None,
        device=CPU,
    )
    assert np.mean(r.amplitudes < 0) > 0.2, (
        "expected sign oscillation when the truth is outside the bound; if this "
        "now passes cleanly the amplitude solve behaviour has changed"
    )


def test_delays_push_toward_the_bound(setup):
    """A modest out-of-range truth should saturate at the edge, not sit at zero."""
    h, _, block_onsets, polys = setup
    nb = len(block_onsets)
    y = _simulate(h, block_onsets, np.full((30, nb), 1.5), np.ones((30, nb)))
    r = fit_shifted_hrf(
        data=torch.from_numpy(y),
        block_onsets=block_onsets,
        hrf=h,
        hrf_dt=DT,
        tr=TR,
        nuisance=torch.from_numpy(polys),
        tau_max=1.0,
        tau_step=0.25,
        n_sweeps=3,
        delay_prior_sd=None,
        device=CPU,
    )
    assert np.median(r.delays) > 0.5, f"median delay {np.median(r.delays):.2f}"


def test_tau_prior_curbs_winners_curse(setup):
    """Free per-trial latency inflates amplitude at low SNR; the prior fixes it.

    Latency is chosen by maximising fit, so at low SNR the winning tau
    partly fits noise and drags amplitude up with it.  Shrinking tau
    toward the voxel's own mean must pull the amplitude bias back without
    destroying latency recovery.
    """
    h, _, block_onsets, polys = setup
    nb = len(block_onsets)
    rng = np.random.default_rng(1)
    tau = rng.normal(0, 1.0, size=(200, nb)).clip(-3, 3)
    amps = np.ones((200, nb))
    y = _simulate(h, block_onsets, tau, amps) + rng.normal(0, 2.0, size=(200, NTP))
    common = dict(
        data=torch.from_numpy(y),
        block_onsets=block_onsets,
        hrf=h,
        hrf_dt=DT,
        tr=TR,
        nuisance=torch.from_numpy(polys),
        tau_max=3.0,
        tau_step=0.25,
        n_sweeps=3,
        device=CPU,
    )
    free = fit_shifted_hrf(**common, delay_prior_sd=None)
    shrunk = fit_shifted_hrf(**common, delay_prior_sd=1.0)
    bias_free = abs(np.median(free.amplitudes) - 1.0)
    bias_shrunk = abs(np.median(shrunk.amplitudes) - 1.0)
    assert bias_free > 0.3, f"expected the known inflation, got {bias_free:.3f}"
    assert bias_shrunk < 0.5 * bias_free, (
        f"prior should halve the amplitude bias: {bias_free:.3f} -> {bias_shrunk:.3f}"
    )
    rmse_free = np.sqrt(np.mean((free.delays - tau) ** 2))
    rmse_shrunk = np.sqrt(np.mean((shrunk.delays - tau) ** 2))
    assert rmse_shrunk < rmse_free


def test_beats_spmg2_on_latency_recovery(setup):
    """The whole point: SPMG2's derivative ratio carries no latency signal."""
    h, onsets, block_onsets, polys = setup
    nb = len(block_onsets)
    rng = np.random.default_rng(2)
    tau = rng.normal(0, 1.0, size=(200, nb)).clip(-3, 3)
    amps = np.ones((200, nb))
    y = _simulate(h, block_onsets, tau, amps) + rng.normal(0, 0.5, size=(200, NTP))

    shift = fit_shifted_hrf(
        data=torch.from_numpy(y),
        block_onsets=block_onsets,
        hrf=h,
        hrf_dt=DT,
        tr=TR,
        nuisance=torch.from_numpy(polys),
        tau_max=3.0,
        tau_step=0.25,
        n_sweeps=3,
        delay_prior_sd=1.0,
        device=CPU,
    )
    r_shift = np.corrcoef(shift.delays.ravel(), tau.ravel())[0, 1]

    sb = generate_spmg_basis(2)
    Xst = np.concatenate(
        [
            build_pc_basis_design_per_run(
                onsets_per_run=[np.array([o])],
                pcs=sb.basis_functions,
                lag_times=np.arange(sb.basis_functions.shape[1]) * sb.dt,
                tr=TR,
                n_timepoints_per_run=[NTP],
                basis="TENT",
            )[0]
            for o in onsets
        ],
        axis=1,
    )
    m, C = spmg_prior(5.0, 0.3)
    fit = fit_basis_constrained_ridge(
        data=torch.from_numpy(y),
        design_task=torch.from_numpy(Xst),
        basis_functions=sb.basis_functions,
        prior_mean=m,
        prior_cov=C,
        n_blocks=nb,
        nuisance=torch.from_numpy(polys),
        prior_weight="auto",
        device=CPU,
        reconstruct_hrfs=False,
    )
    b2 = fit.betas[:, : 2 * nb].reshape(200, nb, 2)
    ratio = b2[..., 1] / np.where(np.abs(b2[..., 0]) < 1e-9, np.nan, b2[..., 0])
    ok = np.isfinite(ratio)
    r_spmg = abs(np.corrcoef(ratio[ok], tau[ok])[0, 1])

    assert r_shift > 0.5, f"shift model latency r only {r_shift:.3f}"
    assert r_spmg < 0.15, f"SPMG2 ratio unexpectedly informative: r={r_spmg:.3f}"


def test_r2_fixed_is_the_no_latency_baseline(setup):
    """r2_fixed must equal a tau=0-only fit, so r2 - r2_fixed is meaningful."""
    h, _, block_onsets, polys = setup
    nb = len(block_onsets)
    rng = np.random.default_rng(3)
    tau = rng.normal(0, 1.0, size=(50, nb)).clip(-3, 3)
    y = _simulate(h, block_onsets, tau, np.ones((50, nb))) + rng.normal(
        0, 0.5, size=(50, NTP)
    )
    kw = dict(
        data=torch.from_numpy(y),
        block_onsets=block_onsets,
        hrf=h,
        hrf_dt=DT,
        tr=TR,
        nuisance=torch.from_numpy(polys),
        tau_step=0.25,
        device=CPU,
    )
    full = fit_shifted_hrf(**kw, tau_max=3.0, n_sweeps=3, delay_prior_sd=None)
    # tau_max=0 collapses the grid to the single tau=0 point
    pinned = fit_shifted_hrf(**kw, tau_max=0.0, n_sweeps=1, delay_prior_sd=None)
    np.testing.assert_allclose(full.r2_fixed, pinned.r2, rtol=1e-6, atol=1e-8)
    assert np.median(full.r2) > np.median(full.r2_fixed)


# ---------------------------------------------------------------------------
# Held-out validation.  These are the tests that matter most: the validator
# is what tells the user whether a delay map means anything, so it has to
# discriminate real latency structure from none.
# ---------------------------------------------------------------------------


def _multirun(h, n_runs, cond_onsets, tau_by_voxel, jitter_sd, noise_sd, nv, seed):
    """Simulate runs where each voxel has a stable condition-level delay.

    ``tau_by_voxel`` is (nv, n_cond) and repeats across runs -- that is the
    structure the LORO validator should be able to generalise.  Per-trial
    jitter is added on top so the fit still has to average over trials.
    """
    from fastfuncstuff.design.shifted_hrf import build_shifted_design_bank

    rng = np.random.default_rng(seed)
    fine = np.arange(-3.0, 3.001, 0.05)
    runs = []
    per_run_onsets = []
    for _r in range(n_runs):
        per_run_onsets.append([np.asarray(o, dtype=float) for o in cond_onsets])
        blocks, cidx = [], []
        for c, o in enumerate(cond_onsets):
            for t in np.atleast_1d(o):
                blocks.append(np.array([float(t)]))
                cidx.append(c)
        bank = build_shifted_design_bank(blocks, h, DT, fine, TR, NTP, device=CPU).numpy()
        y = np.zeros((nv, NTP))
        for b, c in enumerate(cidx):
            tau = tau_by_voxel[:, c] + rng.normal(0, jitter_sd, size=nv)
            gi = np.argmin(np.abs(fine[None, :] - tau[:, None]), axis=1)
            y += bank[b][gi]
        y += rng.normal(0, noise_sd, size=y.shape)
        runs.append(torch.from_numpy(y))
    return runs, per_run_onsets


def test_xval_detects_real_latency_structure(setup):
    """A stable per-voxel delay must give a positive held-out gap."""
    from fastfuncstuff.design.shifted_hrf import xval_shifted_hrf

    h, _, _, _ = setup
    cond_onsets = [np.arange(10.0, NTP * TR - 40, 12.0)]
    nv = 120
    rng = np.random.default_rng(7)
    # half the voxels early, half late -- a real, run-stable delay
    tau_v = np.where(rng.random((nv, 1)) < 0.5, -1.5, 1.5)
    runs, onsets = _multirun(h, 4, cond_onsets, tau_v, 0.3, 0.5, nv, seed=7)
    r2_shift, r2_tau0 = xval_shifted_hrf(
        per_run_data=runs,
        per_run_condition_onsets=onsets,
        hrf=h,
        hrf_dt=DT,
        tr=TR,
        polort=2,
        single_trials=True,
        tau_max=2.0,
        tau_step=0.25,
        delay_prior_sd=0.75,
        device=CPU,
        verbose=False,
    )
    gap = np.median(r2_shift - r2_tau0)
    assert gap > 0.05, f"validator missed real latency structure: gap {gap:+.4f}"


def test_xval_rejects_absent_latency(setup):
    """With zero true delay the held-out gap must NOT be meaningfully positive.

    This is the counterpart to the in-sample trap: n_blocks free latency
    parameters always buy in-sample fit (+0.14 measured on zero-latency
    data), so a validator that cannot return ~0 here is useless.
    """
    from fastfuncstuff.design.shifted_hrf import xval_shifted_hrf

    h, _, _, _ = setup
    cond_onsets = [np.arange(10.0, NTP * TR - 40, 12.0)]
    nv = 120
    tau_v = np.zeros((nv, 1))
    runs, onsets = _multirun(h, 4, cond_onsets, tau_v, 0.0, 0.5, nv, seed=8)
    r2_shift, r2_tau0 = xval_shifted_hrf(
        per_run_data=runs,
        per_run_condition_onsets=onsets,
        hrf=h,
        hrf_dt=DT,
        tr=TR,
        polort=2,
        single_trials=True,
        tau_max=2.0,
        tau_step=0.25,
        delay_prior_sd=0.75,
        device=CPU,
        verbose=False,
    )
    gap = np.median(r2_shift - r2_tau0)
    assert gap < 0.02, f"validator claims latency where there is none: gap {gap:+.4f}"
