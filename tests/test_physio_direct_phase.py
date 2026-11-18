"""Tests for the research phase fit, separate from production cardiac detection."""

import numpy as np

from scripts.physio_phase_review.candidate_aliases import complete_alias_offsets
from scripts.physio_phase_review.direct_phase import fit_phase, prediction_score


def test_shared_phase_recovers_jitter_with_free_voxel_signs_and_delays():
    rng = np.random.default_rng(43)
    times = np.arange(140)[:, None] * 1.3 + np.linspace(0, 1.2, 9)[None, :]
    truth = 2 * np.pi * 1.13 * times + 0.35 * np.sin(2 * np.pi * times / 37)
    initial = truth + 0.65 * np.sin(2 * np.pi * 0.73 * times)
    series = []
    held_out = []
    for g in range(times.shape[1]):
        coefficients = rng.normal(size=(50, 2))
        x = np.array([np.cos(truth[:, g]), np.sin(truth[:, g])])
        y = coefficients @ x + 0.3 * rng.normal(size=(50, len(times)))
        series.append(y[:25])
        held_out.append(y[25:])
    phase, diag = fit_phase(series, times, initial, knot_s=3)
    lock = abs(np.mean(np.exp(1j * (phase - truth))))
    baseline_lock = abs(np.mean(np.exp(1j * (initial - truth))))
    assert diag["success"], diag
    assert lock > 0.98 and lock > baseline_lock + 0.07
    assert prediction_score(held_out, phase, 1.3) > prediction_score(held_out, initial, 1.3)


def test_prediction_uses_training_coefficients_instead_of_refitting_test_samples():
    rng = np.random.default_rng(12)
    nt = 160
    phase = (2 * np.pi * 0.13 * np.arange(nt))[:, None]
    y = np.cos(phase[:, 0]) + 0.1 * rng.normal(size=nt)
    # The signal reverses sign in every held-out block. An in-sample score would
    # call both halves perfect; predictive coefficients must fail to transfer.
    fold = (np.arange(nt) // 20) % 2
    y[fold == 1] *= -1
    assert prediction_score([y[None]], phase, 1.0) < -2


def test_free_voxel_coefficients_leave_the_alias_ambiguous():
    # A fit improvement cannot certify an alias: one integer/TR phase rotation
    # is constant within each excitation and its free coefficients absorb it.
    rng = np.random.default_rng(4)
    tr = 1.3
    times = np.arange(120)[:, None] * tr + np.linspace(0, 1.2, 7)[None]
    phase = 2 * np.pi * 1.13 * times + 0.2 * np.sin(times / 17)
    series = [np.cos(p)[None] + 0.1 * rng.normal(size=(6, len(p))) for p in phase.T]
    true_score = prediction_score(series, phase, tr)
    alias_score = prediction_score(series, phase + 2 * np.pi * times / tr, tr)
    np.testing.assert_allclose(true_score, alias_score, atol=1e-10)


def test_smoothing_is_invariant_to_independent_crossed_phase_origins():
    times = np.arange(120)[:, None] + np.linspace(0, 0.9, 9)[None]
    initial = 2 * np.pi * 1.13 * times + 0.3 * np.sin(times / 13)
    series = [np.cos(p)[None] for p in initial.T]
    shifted = initial.copy()
    shifted[:, ::2] += 4 * np.pi
    shifted[60:, 1::2] -= 2 * np.pi
    phase, _ = fit_phase(series, times, initial, maxiter=0)
    other, _ = fit_phase(series, times, shifted, maxiter=0)
    np.testing.assert_allclose(np.exp(1j * phase), np.exp(1j * other), atol=1e-8)


def test_long_tr_alias_enumeration_includes_the_mirrored_cardiac_clock():
    times = np.arange(100)[:, None] * 2 + np.linspace(0, 1.9, 17)[None]
    phase = 2 * np.pi * 0.8 * times
    offsets = complete_alias_offsets(times, phase, 2.0)
    assert (-1, 4) in offsets  # -0.8 + 4/2 = 1.2 Hz; fixed +/-3 misses it.


def test_same_mean_rate_keeps_both_opposite_rate_fluctuations():
    times = np.arange(100)[:, None] * 2 + np.linspace(0, 1.9, 17)[None]
    phase = 2 * np.pi * 0.75 * times + 0.2 * np.sin(np.linspace(0, 2 * np.pi, 100))[:, None]
    offsets = complete_alias_offsets(times, phase, 2.0)
    assert (1, 0) in offsets and (-1, 3) in offsets
