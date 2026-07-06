"""State<->task alignment from events: recover a planted condition->state mapping."""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.dynamics.task import (
    align_states_to_task,
    events_to_condition_labels,
    events_to_design,
)


def test_condition_labels_persist_until_next_event():
    # Two events in one run: cond 0 at t=0s, cond 1 at t=10s. HRF delay 5s, TR=1s.
    all_onsets = [
        [np.array([0.0])],  # condition 0
        [np.array([10.0])],  # condition 1
    ]
    labels = events_to_condition_labels(
        all_onsets, [20], tr=1.0, hrf_delay=5.0, respect_duration=False
    )[0]
    # Before 5s: baseline; 5..15s: cond 0; 15s..end: cond 1.
    assert (labels[:5] == -1).all()
    assert (labels[5:15] == 0).all()
    assert (labels[15:] == 1).all()


def test_condition_labels_respect_duration_leaves_rest_as_baseline():
    # A 10s task at t=0, then a long gap before the next task at t=40s. With
    # respect_duration the gap (unmarked rest) must be baseline, not the prior task.
    all_onsets = [
        [np.array([0.0])],  # cond 0: on 5..15s (0+delay .. +10s dur)
        [np.array([40.0])],  # cond 1: on 45..55s
    ]
    labels = events_to_condition_labels(
        all_onsets, [60], tr=1.0, hrf_delay=5.0, durations=[10.0, 10.0], respect_duration=True
    )[0]
    assert (labels[:5] == -1).all()  # pre-onset baseline
    assert (labels[5:15] == 0).all()  # task 0 for its 10s duration
    assert (labels[15:45] == -1).all()  # the rest/ITI gap is baseline, NOT task 0
    assert (labels[45:55] == 1).all()  # task 1
    assert (labels[55:] == -1).all()  # trailing rest


def test_condition_labels_respect_duration_requires_durations():
    import pytest

    with pytest.raises(ValueError, match="durations"):
        events_to_condition_labels([[np.array([0.0])]], [10], tr=1.0, respect_duration=True)


def test_design_shapes_and_causality():
    all_onsets = [[np.array([2.0])], [np.array([12.0])]]
    designs = events_to_design(all_onsets, [4.0, 4.0], [30], tr=1.0)
    assert designs[0].shape == (30, 2)
    # Convolved response must be ~0 before its onset frame and rise after.
    d = designs[0].numpy()
    assert abs(d[:2, 0]).max() < 1e-3
    assert d[3:8, 0].max() > 0.3


def _simulate_task_driven(k=3, d=6, t=240, n_runs=3, block=40, tr=1.0, seed=0):
    """States deterministically follow a repeating block schedule of k conditions."""
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((k, d)) * 5.0
    sessions, all_onsets = [], [[[] for _ in range(n_runs)] for _ in range(k)]
    for run in range(n_runs):
        z = np.empty(t, dtype=int)
        onset_s = 0
        cond = 0
        while onset_s < t:
            # record onset for this condition (state will lag by HRF delay ~5 frames)
            all_onsets[cond][run].append(float(onset_s))
            z[onset_s : min(onset_s + block, t)] = cond
            onset_s += block
            cond = (cond + 1) % k
        y = means[z].T + 0.4 * rng.standard_normal((d, t))
        sessions.append(torch.tensor(y, dtype=torch.float64))
    all_onsets = [[np.array(sorted(all_onsets[c][r])) for r in range(n_runs)] for c in range(k)]
    return sessions, all_onsets


def test_alignment_recovers_planted_mapping():
    from fastfuncstuff.dynamics.bsds.model import fit_bsds

    k = 3
    sessions, all_onsets = _simulate_task_driven(k=k, seed=1)
    model = fit_bsds(sessions, n_states=k, max_ldim=3, n_init=4, n_init_iter=12, n_iter=80, seed=0)

    align = align_states_to_task(
        model,
        all_onsets,
        durations=[40.0] * k,
        condition_labels=[f"cond{c}" for c in range(k)],
        tr=1.0,
        hrf_delay=5.0,
    )
    # A clean task-driven fit: each condition should be dominated by a distinct
    # state, so the contingency is close to a permutation matrix -> high NMI.
    assert align.normalized_mutual_info > 0.6, align.normalized_mutual_info
    # Each condition's top state is unique (one-to-one).
    top_state_per_condition = align.contingency.argmax(axis=0)
    assert len(set(top_state_per_condition.tolist())) == k
    # Correlation view: each condition's best-correlated state matches.
    top_corr_state = align.correlation.argmax(axis=0)
    assert len(set(top_corr_state.tolist())) == k
    assert align.state_purity.min() > 0.5


def test_include_rest_adds_condition_and_recovers_rest_state():
    from fastfuncstuff.dynamics.bsds.model import fit_bsds

    # Three task states plus a distinct "rest" regime between blocks. Build data
    # where task blocks are separated by an unmodelled rest with its own pattern.
    rng = np.random.default_rng(4)
    k_task, d, block, gap, n_runs = 3, 6, 20, 20, 3
    means = rng.standard_normal((k_task + 1, d)) * 5.0  # last mean = rest
    sessions, all_onsets = [], [[[] for _ in range(n_runs)] for _ in range(k_task)]
    for run in range(n_runs):
        segs = []
        t_cursor = 0
        cond = 0
        while t_cursor < 240:
            all_onsets[cond][run].append(float(t_cursor))
            segs.append((t_cursor, block, cond))  # task block
            t_cursor += block + gap  # gap = unmarked rest
            cond = (cond + 1) % k_task
        z = np.full(t_cursor if t_cursor <= 260 else 260, k_task, dtype=int)  # default rest
        for onset, dur, c in segs:
            z[onset : onset + dur] = c
        y = means[z].T + 0.4 * rng.standard_normal((len(z), d)).T
        sessions.append(torch.tensor(y, dtype=torch.float64))
    all_onsets = [
        [np.array(sorted(all_onsets[c][r])) for r in range(n_runs)] for c in range(k_task)
    ]

    model = fit_bsds(sessions, n_states=4, max_ldim=3, n_init=4, n_init_iter=12, n_iter=80, seed=0)
    align = align_states_to_task(
        model,
        all_onsets,
        [20.0] * k_task,
        [f"cond{c}" for c in range(k_task)],
        tr=1.0,
        hrf_delay=0.0,
        include_rest=True,
    )
    # "rest" is appended as the last condition; both views gain a column.
    assert align.condition_labels[-1] == "rest"
    assert align.contingency.shape == (4, k_task + 1)
    assert align.correlation.shape == (4, k_task + 1)
    # Some state should be dominated by rest (the between-block regime).
    rest_idx = k_task
    assert (align.dominant_condition == rest_idx).any()
    # The labels now paint rest instead of leaving baseline -1 there.
    assert all((lab >= 0).all() for lab in align.labels)
