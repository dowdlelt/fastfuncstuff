"""What event timing lets a FIR / TENT response estimate resolve."""

import numpy as np
import pytest

from fastfuncstuff.design.event_timing import (
    aligned_knot_start,
    check_event_timing,
    finest_usable_grid,
    onset_phases,
)

TR = 1.0
RNG = np.random.default_rng(0)
BASE = [np.sort(RNG.choice(np.arange(5, 280), 30, replace=False)).astype(float) for _ in range(2)]


def _report(phase_fn, **kw):
    onsets = [[phase_fn(b) for b in BASE]]
    return check_event_timing(onsets, [300, 300], TR, window=(0.0, 15.0), **kw)


def _status(report):
    return [g.status for g in report.grids]


def test_tr_locked_supports_tr_knots_only():
    r = _report(lambda b: b)
    assert _status(r)[0] == "ok" and r.grids[0].amplification == pytest.approx(1.0)
    assert set(_status(r)[1:]) == {"unidentifiable"}  # nothing is ever seen mid-TR


@pytest.mark.parametrize("phase", [0.3, 0.5])
def test_constant_offset_is_singular_on_tr_knots_but_has_an_aligned_grid(phase):
    """The failure behind up-down TENT averages: one phase, knots off the samples."""
    r = _report(lambda b: b + phase)
    assert _status(r)[0] == "unidentifiable"
    assert aligned_knot_start(r) == pytest.approx((1 - phase) * TR)


def test_offsets_bunched_mid_tr_are_unstable():
    r = _report(lambda b: b + RNG.uniform(0.4, 0.6, b.size))
    assert _status(r)[0] == "unstable"
    assert r.alternation_visibility < 0.05


def test_locked_plus_mid_tr_events_resolve_half_tr_knots():
    r = _report(lambda b: b + RNG.choice([0.0, 0.5], b.size))
    assert finest_usable_grid(r).knot_dt == pytest.approx(TR / 2)
    assert _status(r)[2] == "unidentifiable"  # no events at thirds of a TR


def test_uniform_jitter_is_fine_at_tr_and_half_tr():
    r = _report(lambda b: b + RNG.uniform(0, 1, b.size))
    assert _status(r)[:2] == ["ok", "ok"]
    assert r.phase_concentration < 0.3


def test_precision_hides_logging_noise_and_offset_moves_the_clock():
    noisy = np.array([10.0, 20.0, 30.0]) + np.array([0.0013, -0.0021, 0.0008])
    assert np.allclose(onset_phases(noisy, 2.0, precision=0.05), 0.0)
    assert np.allclose(onset_phases(np.array([11.0, 21.0]), 2.0, microtime_offset=1.0), 0.0)
