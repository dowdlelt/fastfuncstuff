"""State-switching statistics: rate, time-resolved probability, and switch paths.

Beyond the transition *matrix*, the BSDS papers report how switching unfolds in
time ([[Cai 2024]] Fig. 4f/5): how often the brain switches, whether switch
probability is time-locked to the task, and which multi-step **paths** between
states recur. These are pure functions of the decoded MAP sequences.

A "visit sequence" is the run-length-encoded state labels — the order states are
*visited*, with dwell duration collapsed away — which is the right object for
counting switch paths (a path is which states follow which, not how long each is
held).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch


def _as_int_array(states) -> np.ndarray:
    if isinstance(states, torch.Tensor):
        return states.detach().cpu().numpy().astype(np.int64)
    return np.asarray(states, dtype=np.int64)


def switch_indicator(states) -> np.ndarray:
    """Per-frame 0/1 switch indicator (``N-1,``): 1 where the state changes."""
    z = _as_int_array(states)
    if z.size < 2:
        return np.zeros(0)
    return (z[1:] != z[:-1]).astype(np.float64)


def switch_rate(states) -> float:
    """Fraction of adjacent frames at which the state changes (a scalar)."""
    ind = switch_indicator(states)
    return float(ind.mean()) if ind.size else 0.0


def visit_sequence(states) -> np.ndarray:
    """Run-length-encoded state labels — the order states are visited."""
    z = _as_int_array(states)
    if z.size == 0:
        return z
    change = np.concatenate([[True], z[1:] != z[:-1]])
    return z[change]


def switch_path_counts(states, order: int = 2) -> Counter:
    """Count ``order``-length paths over the visit sequence (dwell collapsed).

    ``order=2`` counts direct switches ``(i -> j)``; ``order=3`` counts
    ``(i -> j -> k)`` transitions, etc. Pool several sessions by summing the
    Counters. Keys are integer tuples.
    """
    if order < 2:
        raise ValueError("order must be >= 2")
    visits = visit_sequence(states)
    counts: Counter = Counter()
    for i in range(visits.size - order + 1):
        counts[tuple(int(v) for v in visits[i : i + order])] += 1
    return counts


def switching_probability_over_time(sessions_states) -> np.ndarray:
    """Time-resolved P(switch) across equal-length sessions (``N-1,``).

    Averages the per-frame switch indicator over sessions — the task-locked
    switch-probability curve when runs share a time base (same length/alignment).
    Raises if sessions differ in length; use :func:`windowed_switch_rate` for
    unequal runs.
    """
    seqs = [_as_int_array(s) for s in sessions_states]
    if not seqs:
        return np.zeros(0)
    lengths = {s.size for s in seqs}
    if len(lengths) != 1:
        raise ValueError(
            "sessions must share a length for a time-locked curve; "
            "use windowed_switch_rate for unequal runs"
        )
    return np.stack([switch_indicator(s) for s in seqs]).mean(axis=0)


def windowed_switch_rate(states, window: int) -> np.ndarray:
    """Sliding-window switch density within one run (centered, ``N-1`` valid points).

    Works for a single run of any length: the local rate of switching over a
    ``window``-frame box, revealing bursts of instability vs stable epochs.
    """
    ind = switch_indicator(states)
    if ind.size == 0:
        return ind
    window = max(1, min(window, ind.size))
    kernel = np.ones(window) / window
    return np.convolve(ind, kernel, mode="same")


@dataclass
class SwitchStats:
    """Switching statistics for a set of decoded sessions."""

    n_states: int
    tr: float
    group_switch_rate: float
    subject_switch_rate: np.ndarray  # (S,)
    switch_rate_per_minute: float  # group, if tr given
    path_counts: Counter  # order-2 switch paths, pooled
    top_paths: list[tuple[tuple[int, ...], int]]  # most common, most first


def compute_switch_stats(
    model, tr: float = 1.0, *, path_order: int = 2, top: int = 8
) -> SwitchStats:
    """Derive switching statistics from a fitted :class:`BSDSModel`."""
    seqs = [_as_int_array(s) for s in model.viterbi_states]
    rates = np.array([switch_rate(s) for s in seqs]) if seqs else np.zeros(0)
    group = np.concatenate(seqs) if seqs else np.array([], dtype=np.int64)
    grate = switch_rate(group)
    pooled: Counter = Counter()
    for s in seqs:
        pooled.update(switch_path_counts(s, order=path_order))
    per_min = grate / tr * 60.0 if tr and tr > 0 else float("nan")
    return SwitchStats(
        n_states=model.n_states,
        tr=tr,
        group_switch_rate=grate,
        subject_switch_rate=rates,
        switch_rate_per_minute=per_min,
        path_counts=pooled,
        top_paths=pooled.most_common(top),
    )
