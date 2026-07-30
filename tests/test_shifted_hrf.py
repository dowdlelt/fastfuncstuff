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
        get_spm_hrf_with_derivatives(microtime_dt=DT, hrf_duration=32.0, n_basis=1, device=CPU)
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
    bank = build_shifted_design_bank(block_onsets, h, DT, fine, TR, NTP, device=CPU).numpy()
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
    bank = build_shifted_design_bank(block_onsets, h, DT, tau_grid, TR, NTP, device=CPU).numpy()
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
        refine_delays=False,  # this test is about the GRID search itself
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
    y = _simulate(h, block_onsets, tau, np.ones((50, nb))) + rng.normal(0, 0.5, size=(50, NTP))
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


# ---------------------------------------------------------------------------
# Per-voxel shape selection
# ---------------------------------------------------------------------------


def test_shape_library_curves_are_positive_and_peak_normalised():
    """Every candidate must be peak-1 and positive-going.

    A sign-flipped curve in the library would let a voxel fit a positive
    response with a negative amplitude, which is then indistinguishable
    from genuine negative BOLD downstream.  Peak-normalisation is what
    makes the fitted amplitude mean the same thing across shape groups.
    """
    from fastfuncstuff.design.shifted_hrf import build_shape_library

    for src in ("canonical", "library", "pighs", "flobs"):
        curves, labels = build_shape_library(src, DT, 32.0, n_hrfs=8)
        assert curves.shape[0] == len(labels) >= 1
        for c in curves:
            assert abs(np.max(np.abs(c)) - 1.0) < 1e-9, f"{src} not peak-normalised"
            assert abs(c.max()) >= abs(c.min()), f"{src} has a sign-flipped curve"


def test_shape_library_rejects_unknown_source():
    from fastfuncstuff.design.shifted_hrf import build_shape_library

    with pytest.raises(ValueError, match="unknown shape source"):
        build_shape_library("nope", DT, 32.0)


def test_shape_selection_recovers_the_generating_curve(setup):
    """Voxels simulated from curve k should mostly be assigned curve k."""
    from fastfuncstuff.design.shifted_hrf import (
        build_shape_library,
        select_shape_per_voxel,
    )

    _, _, block_onsets, polys = setup
    curves, _ = build_shape_library("pighs", DT, 32.0, n_hrfs=6)
    # pick two curves that are actually distinguishable
    a, b = 0, curves.shape[0] - 1
    nb = len(block_onsets)
    rng = np.random.default_rng(5)
    ys, truth = [], []
    for k in (a, b):
        y = _simulate(curves[k], block_onsets, np.zeros((40, nb)), np.ones((40, nb)))
        ys.append(y + rng.normal(0, 0.3, size=y.shape))
        truth += [k] * 40
    y_all = np.concatenate(ys, axis=0)
    idx, r2 = select_shape_per_voxel(
        torch.from_numpy(y_all),
        block_onsets,
        curves,
        DT,
        TR,
        nuisance=torch.from_numpy(polys).to(torch.float64),
        device=CPU,
    )
    truth = np.asarray(truth)
    assert r2.shape == (80, curves.shape[0])
    # Neighbouring curves in the library are genuinely similar, so exact
    # index recovery is too strict.  What must hold is that the two groups
    # separate, and in the right direction.
    med_a = float(np.median(idx[truth == a]))
    med_b = float(np.median(idx[truth == b]))
    assert med_a < med_b, f"groups not separated in order: {med_a} vs {med_b}"
    # each group's chosen curve should be nearer its own generator
    assert abs(med_a - a) < abs(med_a - b)
    assert abs(med_b - b) < abs(med_b - a)


def test_per_voxel_shape_beats_a_single_wrong_shape(setup):
    """Selecting per voxel must fit better than forcing one shape on all."""
    from fastfuncstuff.design.shifted_hrf import (
        build_shape_library,
        fit_shifted_hrf_per_voxel_shape,
    )

    _, _, block_onsets, polys = setup
    curves, _ = build_shape_library("pighs", DT, 32.0, n_hrfs=6)
    nb = len(block_onsets)
    rng = np.random.default_rng(6)
    ys = []
    for k in (0, curves.shape[0] - 1):
        y = _simulate(curves[k], block_onsets, np.zeros((30, nb)), np.ones((30, nb)))
        ys.append(y + rng.normal(0, 0.3, size=y.shape))
    y_all = np.concatenate(ys, axis=0)
    Zt = torch.from_numpy(polys).to(torch.float64)

    res, sidx = fit_shifted_hrf_per_voxel_shape(
        data=torch.from_numpy(y_all),
        block_onsets=block_onsets,
        shapes=curves,
        hrf_dt=DT,
        tr=TR,
        nuisance=Zt,
        tau_max=1.0,
        tau_step=0.25,
        n_sweeps=2,
        device=CPU,
    )
    # forcing every voxel onto the first curve
    forced = fit_shifted_hrf(
        data=torch.from_numpy(y_all),
        block_onsets=block_onsets,
        hrf=curves[0],
        hrf_dt=DT,
        tr=TR,
        nuisance=Zt,
        tau_max=1.0,
        tau_step=0.25,
        n_sweeps=2,
        device=CPU,
    )
    assert sidx.shape == (60,)
    assert np.median(res.r2) > np.median(forced.r2), (
        f"per-voxel shape {np.median(res.r2):.4f} did not beat forced {np.median(forced.r2):.4f}"
    )


def test_per_voxel_shape_validates_shape_index_length(setup):
    from fastfuncstuff.design.shifted_hrf import (
        build_shape_library,
        fit_shifted_hrf_per_voxel_shape,
    )

    _, _, block_onsets, polys = setup
    curves, _ = build_shape_library("library", DT, 32.0)
    with pytest.raises(ValueError, match="shape_index must have shape"):
        fit_shifted_hrf_per_voxel_shape(
            data=torch.zeros((5, NTP), dtype=torch.float64),
            block_onsets=block_onsets,
            shapes=curves,
            hrf_dt=DT,
            tr=TR,
            shape_index=np.zeros(4, dtype=np.int64),
            nuisance=torch.from_numpy(polys).to(torch.float64),
            device=CPU,
        )


def test_xval_selects_shape_fold_locally(setup):
    """Shape must be re-selected per fold, and scoring must use that shape.

    Shape choice is a free parameter that buys in-sample fit exactly as
    delay does.  Selecting it once on all the data and then "validating"
    leaks; scoring one shared shape while the fit used per-voxel shapes
    validates a different model than was fitted.  Both bugs show up as an
    xval result that does not respond to the shape library at all, so this
    checks that passing `shapes` genuinely changes the held-out score AND
    that a library containing the true generator beats a single wrong one.
    """
    from fastfuncstuff.design.shifted_hrf import build_shape_library, xval_shifted_hrf

    h, _, _, _ = setup
    curves, _ = build_shape_library("pighs", DT, 32.0, n_hrfs=6)
    cond_onsets = [np.arange(10.0, NTP * TR - 40, 12.0)]
    nv = 60
    rng = np.random.default_rng(11)
    tau_v = np.where(rng.random((nv, 1)) < 0.5, -1.0, 1.0)
    # generate half the voxels from the first curve, half from the last
    runs, onsets = _multirun(curves[0], 4, cond_onsets, tau_v[: nv // 2], 0.3, 0.4, nv // 2, 11)
    runs2, _ = _multirun(curves[-1], 4, cond_onsets, tau_v[nv // 2 :], 0.3, 0.4, nv - nv // 2, 12)
    runs = [torch.cat([a, b], dim=0) for a, b in zip(runs, runs2, strict=True)]

    kw = dict(
        per_run_data=runs,
        per_run_condition_onsets=onsets,
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
    # one shape for everyone, deliberately the wrong one for half the voxels
    single, _ = xval_shifted_hrf(hrf=curves[0], **kw)
    # per-voxel selection from a library containing both generators
    per_vox, _ = xval_shifted_hrf(hrf=curves[0], shapes=curves, **kw)

    # voxels generated from the LAST curve are the ones a single wrong shape
    # hurts, so that half must improve when selection is available
    wrong_half = slice(nv // 2, nv)
    assert np.median(per_vox[wrong_half]) > np.median(single[wrong_half]), (
        f"fold-local shape selection did not help the mismatched voxels: "
        f"{np.median(per_vox[wrong_half]):.4f} vs {np.median(single[wrong_half]):.4f}"
    )


def test_shape_selection_absorbs_delay_confound(setup):
    """Records the shape/delay confound so a change in it is visible.

    A library varying peak time competes with the delay parameter: both
    move the response in time.  This is an identifiability property of the
    model, not a bug, but it decides how the outputs must be read, so it
    is pinned here.  If this test starts failing, the confound structure
    changed and the interpretation guidance needs revisiting.

    The magnitude of the confound depends on the library's peak-time spread
    and on SNR (measured r=0.99 with a FLOBS library over 4 runs, r=0.35
    with 8 PIGHS curves over 1), so asserting a particular correlation
    would be brittle.  What is asserted instead is the consequence that
    governs interpretation: on identical data, letting the shape vary
    shrinks the recovered delay relative to holding the shape fixed.
    """
    from fastfuncstuff.design.shifted_hrf import (
        build_shape_library,
        fit_shifted_hrf_per_voxel_shape,
    )

    h, _, _, _ = setup
    curves, _ = build_shape_library("pighs", DT, 32.0, n_hrfs=12)
    cond_onsets = [np.arange(10.0, NTP * TR - 40, 12.0)]
    nv = 80
    rng = np.random.default_rng(21)
    tau_v = np.where(rng.random((nv, 1)) < 0.5, -1.2, 1.2)
    runs, _ = _multirun(h, 1, cond_onsets, tau_v, 0.2, 0.3, nv, seed=21)
    blocks = [np.array([float(t)]) for t in cond_onsets[0]]
    Zt = torch.from_numpy(np.polynomial.legendre.legvander(np.linspace(-1, 1, NTP), 2)).to(
        torch.float64
    )
    kw = dict(
        block_onsets=blocks,
        hrf_dt=DT,
        tr=TR,
        nuisance=Zt,
        tau_max=2.0,
        tau_step=0.25,
        n_sweeps=4,
        device=CPU,
    )
    with_shapes, _ = fit_shifted_hrf_per_voxel_shape(data=runs[0], shapes=curves, **kw)
    # same data, shape held fixed at the true generator: all the timing
    # information has nowhere to go but the delay parameter
    fixed_shape = fit_shifted_hrf(data=runs[0], hrf=h, **kw)

    d_free = float(np.median(np.abs(with_shapes.delays.mean(axis=1))))
    d_fixed = float(np.median(np.abs(fixed_shape.delays.mean(axis=1))))
    assert d_free < d_fixed, (
        f"expected a varying shape to absorb some delay: |delay| with shapes "
        f"{d_free:.3f} vs fixed shape {d_fixed:.3f}"
    )


def test_band_covers_every_nonzero_gram_entry(setup):
    """The band must be a superset of the projected gram's support.

    Bug of record: banding on raw time overlap alone is WRONG.  Projecting
    the nuisance out of the design bank couples every column through the
    nuisance subspace, so the projected gram is dense even when the raw one
    is banded.  Dropping those terms wrecked the fit (negative R2, runaway
    amplitudes).  This checks the band derivation against a brute-force
    dense gram, which is the thing the fast path must not lose.
    """
    from fastfuncstuff.design.shifted_hrf import _project_out

    h, _, block_onsets, polys = setup
    nb = len(block_onsets)
    tau = np.arange(-2.0, 2.001, 0.25)
    Zt = torch.from_numpy(polys).to(torch.float64)
    raw = build_shifted_design_bank(block_onsets, h, DT, tau, TR, NTP, device=CPU)
    proj = _project_out(raw, Zt)
    flat = proj.reshape(nb * tau.size, NTP)
    dense = (flat @ flat.T).reshape(nb, tau.size, nb, tau.size)
    mag = dense.abs().amax(dim=(1, 3)).numpy()
    nonzero = mag > 1e-12 * mag.max()

    # reproduce the band the solver derives
    raw_support = (raw.abs().amax(dim=1) > 0).to(torch.float64)
    band = ((raw_support @ raw_support.T) > 0).numpy()
    Q, _ = torch.linalg.qr(Zt)
    u = (raw.reshape(nb * tau.size, NTP) @ Q).reshape(nb, tau.size, -1)
    umag = u.abs().amax(dim=1)
    band = band | ((umag @ umag.T) > 0).numpy()

    missed = nonzero & ~band
    assert not missed.any(), (
        f"band drops {int(missed.sum())} non-zero gram pairs; "
        "the fast path would silently lose those interactions"
    )


def test_delay_refinement_beats_the_grid_off_grid(setup):
    """Sub-grid refinement must reduce delay error when truth is off-grid.

    A grid search alone quantises delays to tau_step, which shows up as
    banding in a delay map and puts a floor of step/sqrt(12) on the error.
    The profiled objective is smooth in tau, so a parabola through the
    samples around the argmax should recover most of that.
    """
    h, _, block_onsets, polys = setup
    nb = len(block_onsets)
    rng = np.random.default_rng(31)
    step = 0.5  # deliberately coarse so quantisation dominates
    # truth placed mid-step, the worst case for a grid search
    tau = rng.uniform(-1.5, 1.5, size=(60, nb))
    y = _simulate(h, block_onsets, tau, np.ones((60, nb))) + rng.normal(0, 0.2, size=(60, NTP))
    kw = dict(
        data=torch.from_numpy(y),
        block_onsets=block_onsets,
        hrf=h,
        hrf_dt=DT,
        tr=TR,
        nuisance=torch.from_numpy(polys),
        tau_max=2.0,
        tau_step=step,
        n_sweeps=4,
        delay_prior_sd=None,
        device=CPU,
    )
    grid = fit_shifted_hrf(**kw, refine_delays=False)
    fine = fit_shifted_hrf(**kw, refine_delays=True)
    rmse_grid = float(np.sqrt(np.mean((grid.delays - tau) ** 2)))
    rmse_fine = float(np.sqrt(np.mean((fine.delays - tau) ** 2)))
    assert rmse_fine < rmse_grid, f"refinement did not help: {rmse_grid:.4f} -> {rmse_fine:.4f}"
    # and the refined delays must stop looking quantised
    n_on_grid_grid = np.mean(np.abs(grid.delays / step - np.round(grid.delays / step)) < 1e-6)
    n_on_grid_fine = np.mean(np.abs(fine.delays / step - np.round(fine.delays / step)) < 1e-6)
    assert n_on_grid_grid > 0.99, "grid search should be exactly quantised"
    assert n_on_grid_fine < 0.2, "refined delays should no longer sit on the grid"


def test_r2_is_task_relative_not_drift_inflated(setup):
    """r2 must not credit the nuisance model with explained variance.

    Bug of record: ss_tot came from the RAW data while SSE was measured
    after removing BOTH drift and task, so everything the polynomials
    absorbed counted as "explained".  With realistic drift that inflates
    r2 badly and makes it useless for spotting task-responsive voxels --
    which is exactly what it gets used for.  r2_total keeps the old
    convention, so the two must differ in the presence of drift, and r2
    must be the smaller one.
    """
    h, _, block_onsets, polys = setup
    nb = len(block_onsets)
    rng = np.random.default_rng(41)
    nv = 40
    # weak task response sitting on strong low-frequency drift
    y = 0.3 * _simulate(h, block_onsets, np.zeros((nv, nb)), np.ones((nv, nb)))
    t = np.linspace(-1, 1, NTP)
    drift = np.stack([rng.normal(0, 6.0) * t**k for k in range(1, 4)], axis=0).sum(axis=0)
    y = y + drift[None, :] + rng.normal(0, 0.3, size=(nv, NTP))
    r = fit_shifted_hrf(
        data=torch.from_numpy(y),
        block_onsets=block_onsets,
        hrf=h,
        hrf_dt=DT,
        tr=TR,
        nuisance=torch.from_numpy(polys),
        tau_max=1.0,
        tau_step=0.25,
        n_sweeps=2,
        device=CPU,
    )
    task = float(np.median(r.r2))
    incl = float(np.median(r.r2_total))
    assert incl > task, f"drift-inclusive r2 should be larger: {incl:.3f} vs {task:.3f}"
    # the drift here dominates, so the gap must be substantial
    assert incl - task > 0.2, (
        f"expected drift to inflate r2 substantially, gap only {incl - task:.3f}"
    )
