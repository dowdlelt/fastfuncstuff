"""Bridge BSDS states to a CEBRA embedding (kept external — no CEBRA dependency).

BSDS and CEBRA decompose orthogonal axes of the same latent dynamics: BSDS gives
discrete recurring states with Markov switching, CEBRA gives a continuous
nonlinear manifold aligned to behaviour. These helpers support the two low-risk
ways to combine them, taking/returning plain arrays so you run CEBRA yourself and
pass its embedding back:

1. **Overlay** — colour a CEBRA embedding by BSDS state and quantify whether the
   states occupy distinct manifold territories (:func:`state_embedding_separation`).
   Strong separation is convergent evidence the discrete states are real geometric
   structure, not artifacts of the linear model.
2. **Behaviour triangulation** — relate per-session state occupancy/lifetime to a
   behavioural measure (:func:`behavior_state_correlation`), the same axis CEBRA
   can condition its embedding on.

:func:`prepare_cebra_inputs` returns the concatenated ``(N, D)`` matrix (and
optional per-frame state labels) that ``cebra.CEBRA().fit`` expects.
"""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def frame_aligned_labels(model_or_result) -> tuple[np.ndarray, list[int]]:
    """Concatenated per-frame MAP state labels and per-session lengths.

    Accepts a :class:`BSDSModel` (``viterbi_states``) or a
    :class:`DynamicStatesResult` (``state_timecourse``).
    """
    seqs = getattr(model_or_result, "viterbi_states", None)
    if seqs is None:
        seqs = model_or_result.state_timecourse
    seqs = [_to_numpy(s).astype(np.int64) for s in seqs]
    lengths = [int(s.shape[0]) for s in seqs]
    return np.concatenate(seqs) if seqs else np.array([], dtype=np.int64), lengths


def prepare_cebra_inputs(sessions) -> tuple[np.ndarray, list[int]]:
    """Concatenate ``(D, N)`` sessions into the ``(N_total, D)`` matrix CEBRA wants.

    Returns the time-major concatenation and per-session lengths (pass the lengths
    as CEBRA session boundaries if fitting multi-session).
    """
    mats = [_to_numpy(s).T for s in sessions]  # each (N_i, D)
    lengths = [int(m.shape[0]) for m in mats]
    return np.concatenate(mats, axis=0).astype(np.float32), lengths


def state_embedding_separation(
    embedding: np.ndarray,
    labels: np.ndarray,
    n_states: int | None = None,
) -> dict:
    """Quantify how distinctly BSDS states occupy a CEBRA (or any) embedding.

    Uses a Calinski-Harabasz-style ratio (between-state dispersion / within-state
    dispersion) — O(N·E), unlike silhouette — so it scales to fMRI-length data. A
    higher ``ch_score`` means the states sit in more distinct manifold territories.

    Returns a dict with ``ch_score``, ``between``, ``within``, ``centroids``
    ``(K, E)`` and per-state counts.
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if embedding.shape[0] != labels.shape[0]:
        raise ValueError("embedding and labels must have the same number of frames")
    n, e = embedding.shape
    k = n_states if n_states is not None else int(labels.max()) + 1
    grand = embedding.mean(axis=0)
    centroids = np.zeros((k, e))
    counts = np.zeros(k, dtype=np.int64)
    between = 0.0
    within = 0.0
    for j in range(k):
        mask = labels == j
        counts[j] = mask.sum()
        if counts[j] == 0:
            continue
        c = embedding[mask].mean(axis=0)
        centroids[j] = c
        between += counts[j] * np.sum((c - grand) ** 2)
        within += np.sum((embedding[mask] - c) ** 2)
    n_present = int((counts > 0).sum())
    denom = within * max(n_present - 1, 1)
    ch = (between * (n - n_present)) / denom if within > 0 else np.inf
    return {
        "ch_score": float(ch),
        "between": float(between),
        "within": float(within),
        "centroids": centroids,
        "counts": counts,
    }


def behavior_state_correlation(
    feature_per_session: np.ndarray,
    behavior: np.ndarray,
) -> np.ndarray:
    """Pearson correlation of each state's per-session feature with a behaviour.

    ``feature_per_session`` is ``(S, K)`` (e.g. ``stats.subject_occupancy`` or
    ``subject_lifetime``); ``behavior`` is ``(S,)``. Returns ``(K,)`` correlations,
    matching the "does occupancy of state k predict behaviour" analysis in the
    BSDS papers.
    """
    feat = np.asarray(feature_per_session, dtype=np.float64)
    beh = np.asarray(behavior, dtype=np.float64)
    if feat.shape[0] != beh.shape[0]:
        raise ValueError("feature_per_session and behavior must share the session axis")
    beh_c = beh - beh.mean()
    out = np.zeros(feat.shape[1])
    for j in range(feat.shape[1]):
        col = feat[:, j] - feat[:, j].mean()
        denom = np.sqrt((col**2).sum() * (beh_c**2).sum())
        out[j] = (col * beh_c).sum() / denom if denom > 0 else 0.0
    return out
