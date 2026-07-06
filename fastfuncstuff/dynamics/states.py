"""Post-hoc dynamic-state statistics for BSDS.

Given the MAP state sequences (Viterbi) and the model's per-state covariances,
these derive the quantities the BSDS papers report: fractional occupancy, mean
lifetime (mean dwell time), dwell-time distributions, empirical transition
matrices, and per-state functional connectivity. Everything is available both
*group-wise* (pooling all sessions) and *subject-wise* (per session), matching
the reference's two reporting modes.

State sequences are integer tensors/arrays in ``0..K-1``; a "session" here is one
decoded run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _as_int_array(states) -> np.ndarray:
    if isinstance(states, torch.Tensor):
        return states.detach().cpu().numpy().astype(np.int64)
    return np.asarray(states, dtype=np.int64)


def fractional_occupancy(states, n_states: int) -> np.ndarray:
    """Fraction of timepoints spent in each state (``K,``)."""
    z = _as_int_array(states)
    counts = np.bincount(z, minlength=n_states)[:n_states]
    return counts / max(len(z), 1)


def _segment_count(mask: np.ndarray) -> int:
    """Number of maximal contiguous runs of True in a boolean sequence."""
    if mask.size == 0:
        return 0
    padded = np.concatenate([[False], mask, [False]])
    return int(((~padded[:-1]) & padded[1:]).sum())


def mean_lifetime(states, n_states: int, tr: float = 1.0) -> np.ndarray:
    """Mean dwell time per state (``K,``), in units of ``tr`` (seconds if TR given).

    ``lifetime_k = (# timepoints in k) / (# contiguous visits to k)``. States that
    never occur get lifetime 0.
    """
    z = _as_int_array(states)
    out = np.zeros(n_states)
    for k in range(n_states):
        mask = z == k
        n_visits = _segment_count(mask)
        if n_visits > 0:
            out[k] = mask.sum() / n_visits * tr
    return out


def dwell_times(states, n_states: int, tr: float = 1.0) -> list[np.ndarray]:
    """Per-state list of individual dwell-segment lengths (the lifetime distribution)."""
    z = _as_int_array(states)
    out: list[list[float]] = [[] for _ in range(n_states)]
    if z.size:
        # Run-length encode the sequence.
        change = np.concatenate([[True], z[1:] != z[:-1]])
        starts = np.flatnonzero(change)
        lengths = np.diff(np.concatenate([starts, [z.size]]))
        for s, ln in zip(starts, lengths, strict=True):
            out[int(z[s])].append(ln * tr)
    return [np.array(v) for v in out]


def empirical_transition_matrix(states, n_states: int) -> np.ndarray:
    """Row-normalised empirical transition matrix from a decoded sequence (``K, K``).

    Complements the model's Dirichlet-posterior transition matrix; useful as a
    consistency check and for reporting per-session dynamics.
    """
    z = _as_int_array(states)
    counts = np.zeros((n_states, n_states))
    for a, b in zip(z[:-1], z[1:], strict=True):
        counts[a, b] += 1
    row = counts.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return counts / row


def effective_state_count(occupancy) -> float:
    """Participation ratio of occupancy: ``exp(entropy)`` = effective # of states used.

    A one-number answer to "how many states are *really* in play". If mass is
    split evenly over ``m`` states it returns ``m``; if one state dominates it
    tends to 1, regardless of how many states were fit or nominally occupied. A
    better headline than the raw occupied count, which weights a 0.001-occupancy
    state the same as a 0.3 one.
    """
    p = np.asarray(occupancy, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def covariance_to_correlation(cov: torch.Tensor) -> torch.Tensor:
    """Normalise a covariance to a correlation matrix (dynamic FC), batched over states."""
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)  # (..., D)
    inv_sd = diag.clamp_min(1e-12).rsqrt()
    return cov * inv_sd.unsqueeze(-1) * inv_sd.unsqueeze(-2)


@dataclass
class StateStats:
    """Group- and subject-wise dynamic-state statistics for a BSDS fit."""

    n_states: int
    tr: float
    effective_state_count: float  # exp(entropy(group_occupancy)) — states really in play
    group_occupancy: np.ndarray  # (K,)
    group_lifetime: np.ndarray  # (K,)
    group_transition: np.ndarray  # (K, K)
    group_dwell_times: list[np.ndarray]  # per state
    subject_occupancy: np.ndarray  # (S, K)
    subject_lifetime: np.ndarray  # (S, K)
    subject_transition: np.ndarray  # (S, K, K)
    state_fc: torch.Tensor  # (K, D, D) per-state correlation matrices


def compute_state_stats(model, tr: float = 1.0) -> StateStats:
    """Derive all reported dynamic-state statistics from a fitted :class:`BSDSModel`."""
    k = model.n_states
    seqs = [_as_int_array(s) for s in model.viterbi_states]
    group = np.concatenate(seqs) if seqs else np.array([], dtype=np.int64)

    subj_occ = np.stack([fractional_occupancy(s, k) for s in seqs])
    subj_life = np.stack([mean_lifetime(s, k, tr) for s in seqs])
    subj_trans = np.stack([empirical_transition_matrix(s, k) for s in seqs])
    group_occ = fractional_occupancy(group, k)

    return StateStats(
        n_states=k,
        tr=tr,
        effective_state_count=effective_state_count(group_occ),
        group_occupancy=group_occ,
        group_lifetime=mean_lifetime(group, k, tr),
        group_transition=empirical_transition_matrix(group, k),
        group_dwell_times=dwell_times(group, k, tr),
        subject_occupancy=subj_occ,
        subject_lifetime=subj_life,
        subject_transition=subj_trans,
        state_fc=covariance_to_correlation(model.state_covs),
    )
