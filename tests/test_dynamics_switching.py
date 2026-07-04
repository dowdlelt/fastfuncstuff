"""Tests for state-switching statistics."""

from __future__ import annotations

import types

import numpy as np
import torch

from fastfuncstuff.dynamics.switching import (
    compute_switch_stats,
    switch_indicator,
    switch_path_counts,
    switch_rate,
    switching_probability_over_time,
    visit_sequence,
    windowed_switch_rate,
)

SEQ = [0, 0, 1, 1, 1, 0, 2, 2]  # 3 switches over 7 adjacent pairs


def test_switch_indicator_and_rate():
    ind = switch_indicator(SEQ)
    np.testing.assert_array_equal(ind, [0, 1, 0, 0, 1, 1, 0])
    assert abs(switch_rate(SEQ) - 3 / 7) < 1e-12


def test_visit_sequence_collapses_dwell():
    np.testing.assert_array_equal(visit_sequence(SEQ), [0, 1, 0, 2])


def test_switch_path_counts_order2_and_3():
    c2 = switch_path_counts(SEQ, order=2)
    assert c2[(0, 1)] == 1 and c2[(1, 0)] == 1 and c2[(0, 2)] == 1
    c3 = switch_path_counts(SEQ, order=3)
    assert c3[(0, 1, 0)] == 1 and c3[(1, 0, 2)] == 1


def test_time_locked_probability_averages_sessions():
    a = [0, 1, 1, 0]
    b = [0, 0, 1, 1]
    p = switching_probability_over_time([a, b])
    # switches: a=[1,0,1], b=[0,1,0] -> mean [0.5, 0.5, 0.5]
    np.testing.assert_allclose(p, [0.5, 0.5, 0.5])


def test_time_locked_probability_requires_equal_length():
    try:
        switching_probability_over_time([[0, 1], [0, 1, 0]])
    except ValueError:
        return
    raise AssertionError("expected ValueError for unequal lengths")


def test_windowed_switch_rate_length_and_bounds():
    r = windowed_switch_rate(SEQ, window=3)
    assert r.shape[0] == len(SEQ) - 1
    assert np.all((r >= 0) & (r <= 1))


def test_compute_switch_stats_from_model():
    model = types.SimpleNamespace(
        n_states=3,
        viterbi_states=[torch.tensor(SEQ), torch.tensor([2, 2, 2, 0, 0, 1])],
    )
    ss = compute_switch_stats(model, tr=2.0, path_order=2, top=3)
    assert ss.subject_switch_rate.shape == (2,)
    assert 0 < ss.group_switch_rate < 1
    # tr=2s, so per-minute = rate/2*60
    assert abs(ss.switch_rate_per_minute - ss.group_switch_rate / 2 * 60) < 1e-9
    assert len(ss.top_paths) <= 3
