"""Relating BSDS states to task structure from events.tsv.

BSDS finds states *unsupervised* — it never sees the task. A natural question
for a task dataset is then: **do the discovered states line up with the task
conditions?** ([[Taghia 2018]] found brain states are only weakly task-aligned,
so this is a genuine empirical question, not a foregone conclusion.) This module
answers it two complementary ways, both fed by the same BIDS ``events.tsv`` the
GLM tools already parse (:func:`fastfuncstuff.design.bids_events.parse_bids_events`):

1. **Convolved-design correlation** — build the standard HRF-convolved regressor
   for each condition and correlate it with each state's probability time course
   (the responsibilities). A graded, continuous view: how strongly does state
   ``k`` track condition ``c``'s expected BOLD.

2. **Condition-label contingency** — the simpler view the user's block designs
   invite: shift every event onset by an HRF delay (~5 s) and call that condition
   "active" until the next event (also shifted). This yields a piecewise-constant
   condition label per timepoint, which we cross-tabulate against the MAP
   (Viterbi) state sequence — a contingency table plus a single normalized
   mutual-information consistency score. Long-block designs (one condition active
   for many TRs) are exactly where this is clean; heavily overlapping / rapid
   event-related designs are where it blurs.

Everything is a pure function of the fit + events, so the CLI can emit it and the
notebook can consume the same helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fastfuncstuff.design.hrf import get_canonical_hrf
from fastfuncstuff.design.matrices import convolve_hrf

_BASELINE = -1  # label for timepoints before the first event / unmodelled rest


def events_to_condition_labels(
    all_onsets: list[list[np.ndarray]],
    session_lengths: list[int],
    tr: float,
    *,
    hrf_delay: float = 5.0,
    durations: list[float] | None = None,
    respect_duration: bool = True,
) -> list[np.ndarray]:
    """Piecewise-constant condition label per timepoint, one array per run.

    ``all_onsets[cond_idx][run_idx]`` is the onset array (seconds) as returned by
    :func:`~fastfuncstuff.design.bids_events.parse_bids_events`. Every onset is
    shifted by ``hrf_delay`` seconds. Two modes:

    - ``respect_duration=True`` (default): a condition is active only for its own
      ``durations[cond_idx]`` seconds (``onset+delay`` to ``onset+delay+dur``),
      then the run returns to ``-1`` (baseline). This is correct for designs with
      **inter-trial rest**: a 30 s task never bleeds across the following ITI. On
      overlapping windows the later onset wins.
    - ``respect_duration=False``: a condition persists from its shifted onset
      until the **next** event's shifted onset (no baseline gaps). Fine only when
      events tile the run back-to-back with no unmodelled rest.

    Returns a list of ``(T_run,)`` int arrays with values in ``-1..C-1``.
    """
    n_cond = len(all_onsets)
    if respect_duration and durations is None:
        raise ValueError("respect_duration=True requires per-condition durations")
    labels_per_run: list[np.ndarray] = []
    for run_idx, t_run in enumerate(session_lengths):
        labels = np.full(t_run, _BASELINE, dtype=np.int64)
        if respect_duration:
            # Collect (shifted_onset, cond) and paint each event's own window;
            # ascending onset order means a later event overwrites on overlap.
            events = sorted(
                (float(on) + hrf_delay, cidx)
                for cidx in range(n_cond)
                for on in np.asarray(all_onsets[cidx][run_idx], dtype=float).ravel()
            )
            for ts, cidx in events:
                start = int(round(ts / tr))
                end = start + int(round(float(durations[cidx]) / tr))  # type: ignore[index]
                start = max(start, 0)
                end = min(end, t_run)
                if end > start:
                    labels[start:end] = cidx
        else:
            ev_times: list[float] = []
            ev_conds: list[int] = []
            for cidx in range(n_cond):
                for on in np.asarray(all_onsets[cidx][run_idx], dtype=float).ravel():
                    ev_times.append(float(on) + hrf_delay)
                    ev_conds.append(cidx)
            if ev_times:
                order = np.argsort(ev_times, kind="stable")
                sorted_times = np.asarray(ev_times)[order]
                sorted_conds = np.asarray(ev_conds, dtype=np.int64)[order]
                frame_times = np.arange(t_run) * tr
                pos = np.searchsorted(sorted_times, frame_times, side="right") - 1
                active = pos >= 0
                labels[active] = sorted_conds[pos[active]]
        labels_per_run.append(labels)
    return labels_per_run


def events_to_design(
    all_onsets: list[list[np.ndarray]],
    durations: list[float],
    session_lengths: list[int],
    tr: float,
    *,
    hrf_duration: float = 32.0,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    """HRF-convolved design matrix per run: ``(T_run, C)``.

    Each condition gets a canonical double-gamma HRF convolved with a boxcar of
    that condition's ``durations`` value (reusing the GLM design primitives), so
    the regressors match what the rest of the codebase builds.
    """
    n_cond = len(all_onsets)
    designs: list[torch.Tensor] = []
    # One HRF per condition (duration-specific boxcar convolution).
    hrfs = [
        get_canonical_hrf(max(float(durations[c]), 0.0), tr, duration=hrf_duration, device=device)
        for c in range(n_cond)
    ]
    for run_idx, t_run in enumerate(session_lengths):
        design = torch.zeros(t_run, n_cond, dtype=torch.float32)
        for cidx in range(n_cond):
            onsets = np.asarray(all_onsets[cidx][run_idx], dtype=float).ravel()
            if onsets.size == 0:
                continue
            onset_vec = torch.zeros(t_run, dtype=torch.float32)
            frames = np.rint(onsets / tr).astype(int)
            frames = frames[(frames >= 0) & (frames < t_run)]
            onset_vec[frames] = 1.0
            col = convolve_hrf(onset_vec, hrfs[cidx], t_run, device=device)
            design[:, cidx] = col.squeeze().to("cpu")[:t_run]
        designs.append(design)
    return designs


def _pearson_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlation of every column of ``a (N, K)`` with every column of ``b (N, C)`` -> ``(K, C)``."""
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    a_norm = np.linalg.norm(a, axis=0)
    b_norm = np.linalg.norm(b, axis=0)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    return (a.T @ b) / np.outer(a_norm, b_norm)


def _normalized_mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    """NMI between two integer label sequences (``sqrt`` normalisation), in ``[0, 1]``."""
    n = x.size
    if n == 0:
        return 0.0
    xs = np.unique(x)
    ys = np.unique(y)
    if xs.size < 2 or ys.size < 2:
        return 0.0
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    joint = np.zeros((xs.size, ys.size))
    for xv, yv in zip(x, y, strict=True):
        joint[xi[xv], yi[yv]] += 1
    joint /= n
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    def _entropy(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    mi = 0.0
    for i in range(xs.size):
        for j in range(ys.size):
            if joint[i, j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))
    hx, hy = _entropy(px), _entropy(py)
    if hx <= 0 or hy <= 0:
        return 0.0
    return float(mi / np.sqrt(hx * hy))


@dataclass
class TaskStateAlignment:
    """How BSDS states relate to task conditions (from events.tsv)."""

    condition_labels: list[str]  # length C
    n_states: int
    correlation: np.ndarray  # (K, C) state responsibility vs convolved design
    contingency: np.ndarray  # (K, C) P(state=k | condition c active), columns sum to 1
    state_purity: np.ndarray  # (K,) max_c P(condition c | state k) over non-baseline conditions
    dominant_condition: np.ndarray  # (K,) argmax condition per state (or -1 if baseline dominates)
    normalized_mutual_info: float  # scalar state<->condition consistency in [0, 1]
    labels: list[np.ndarray]  # per-run condition label time courses (T_run,)


def _rest_regressor(rest_indicator: np.ndarray, tr: float, device) -> np.ndarray:
    """HRF-convolved regressor for a rest boxcar (1 = rest frame) -> ``(T,)``.

    Uses the canonical impulse HRF convolved with the full rest boxcar, so a
    variable-length rest block gets the correct sustained BOLD shape.
    """
    hrf = get_canonical_hrf(0.0, tr, device=device)  # impulse response
    ind = torch.tensor(rest_indicator.astype(np.float32))
    col = convolve_hrf(ind, hrf, len(rest_indicator), device=device)
    return col.squeeze().to("cpu").numpy()[: len(rest_indicator)]


def align_states_to_task(
    model,
    all_onsets: list[list[np.ndarray]],
    durations: list[float],
    condition_labels: list[str],
    *,
    tr: float,
    hrf_delay: float = 5.0,
    respect_duration: bool = True,
    include_rest: bool = False,
    device: torch.device | None = None,
) -> TaskStateAlignment:
    """Relate a fitted :class:`BSDSModel` to task conditions from parsed events.

    ``all_onsets`` / ``durations`` / ``condition_labels`` come straight from
    :func:`~fastfuncstuff.design.bids_events.parse_bids_events`. The model's
    per-run responsibilities and Viterbi paths must align to the same runs the
    events describe (same order, same lengths).

    ``respect_duration`` (default True) controls the label/contingency view: a
    condition is on only for its ``duration`` (unmarked rest/ITI becomes baseline
    and is excluded from the contingency and NMI). Set False for the
    persist-until-next-event behaviour (only sensible when events tile the run
    with no gaps). The correlation view is always duration-aware (HRF boxcar).

    ``include_rest`` (default False): add a synthetic ``"rest"`` condition for the
    unmodelled null periods (the baseline frames left by ``respect_duration``), so
    both views get a rest column and you can see whether a state owns rest. Only
    meaningful with ``respect_duration=True`` (persist mode leaves no interior
    baseline).
    """
    k = model.n_states
    c = len(condition_labels)
    condition_labels = list(condition_labels)
    session_lengths = [int(r.shape[0]) for r in model.responsibilities]

    # --- Convolved-design correlation view ---
    designs = events_to_design(all_onsets, durations, session_lengths, tr, device=device)

    # --- Condition-label contingency view ---
    labels = events_to_condition_labels(
        all_onsets,
        session_lengths,
        tr,
        hrf_delay=hrf_delay,
        durations=durations,
        respect_duration=respect_duration,
    )

    if include_rest:
        # Promote the leftover baseline frames to a real "rest" condition (index c):
        # colour the rest regressor into the design and relabel the timecourse.
        rest_idx = c
        for run_idx in range(len(labels)):
            rest_ind = labels[run_idx] == _BASELINE
            rest_col = torch.tensor(
                _rest_regressor(rest_ind, tr, device), dtype=designs[run_idx].dtype
            )
            designs[run_idx] = torch.cat([designs[run_idx], rest_col.unsqueeze(1)], dim=1)
            labels[run_idx] = labels[run_idx].copy()
            labels[run_idx][rest_ind] = rest_idx
        condition_labels = condition_labels + ["rest"]
        c = c + 1

    resp = np.concatenate([r.detach().cpu().numpy() for r in model.responsibilities], axis=0)
    des = np.concatenate([d.detach().cpu().numpy() for d in designs], axis=0)
    correlation = _pearson_columns(resp, des)  # (K, C)

    state_seq = np.concatenate(
        [v.detach().cpu().numpy().astype(np.int64) for v in model.viterbi_states]
    )
    label_seq = np.concatenate(labels)

    # counts[k, c] over non-baseline frames.
    counts = np.zeros((k, c))
    valid = label_seq >= 0
    for st, lb in zip(state_seq[valid], label_seq[valid], strict=True):
        counts[st, lb] += 1
    # Contingency: P(state | condition) — normalise within each condition column.
    col_tot = counts.sum(axis=0, keepdims=True)
    contingency = counts / np.where(col_tot > 0, col_tot, 1.0)
    # Purity: for each state, the fraction of its (non-baseline) frames in its
    # most-common condition — how cleanly the state maps to a single condition.
    row_tot = counts.sum(axis=1, keepdims=True)
    p_cond_given_state = counts / np.where(row_tot > 0, row_tot, 1.0)
    state_purity = p_cond_given_state.max(axis=1)
    dominant_condition = np.where(
        row_tot.squeeze(1) > 0, p_cond_given_state.argmax(axis=1), _BASELINE
    ).astype(np.int64)

    nmi = _normalized_mutual_info(state_seq[valid], label_seq[valid])

    return TaskStateAlignment(
        condition_labels=list(condition_labels),
        n_states=k,
        correlation=correlation,
        contingency=contingency,
        state_purity=state_purity,
        dominant_condition=dominant_condition,
        normalized_mutual_info=nmi,
        labels=labels,
    )
