"""Brain-behaviour analysis from per-session state features.

The BSDS payoff analysis: relate a subject's/session's **state dynamics** to
**behaviour** ([[Cai 2024]] — occupancy of the optimal state predicts task
performance). This module turns a fit into a per-session feature matrix and runs
the cross-validated models that guard against the tiny-``S`` overfitting these
analyses invite:

- :func:`session_feature_matrix` — ``(S, F)`` features (occupancy, lifetime,
  switch rate) with names, the design matrix for everything below.
- :func:`cross_validated_prediction` — leave-one-session-out **ridge regression**
  of a continuous behaviour; reports held-out R² and correlation (the honest
  effect size), never in-sample fit.
- :func:`loso_classification` — leave-one-session-out **nearest-centroid** task
  decoding; reports held-out accuracy + confusion.
- :func:`canonical_correlation` — SVD-based **CCA** between the state features and
  a multivariate behaviour battery.

Dependency-light (numpy only). With ``S`` in the tens these are inherently
low-powered — treat results as descriptive and always cross-validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def session_feature_matrix(
    stats, *, switch_stats=None, include=("occupancy", "lifetime")
) -> tuple[np.ndarray, list[str]]:
    """Assemble a ``(S, F)`` per-session feature matrix with column names.

    ``include`` selects blocks: ``"occupancy"`` and ``"lifetime"`` (each ``K``
    columns from ``stats``), and ``"switch_rate"`` (one column, needs
    ``switch_stats``). Columns are named ``occ_S0``, ``life_S0``, ``switch_rate``.
    """
    k = stats.n_states
    blocks: list[np.ndarray] = []
    names: list[str] = []
    if "occupancy" in include:
        blocks.append(np.asarray(stats.subject_occupancy, dtype=np.float64))
        names += [f"occ_S{s}" for s in range(k)]
    if "lifetime" in include:
        blocks.append(np.asarray(stats.subject_lifetime, dtype=np.float64))
        names += [f"life_S{s}" for s in range(k)]
    if "switch_rate" in include:
        if switch_stats is None:
            raise ValueError("switch_rate requires switch_stats")
        blocks.append(np.asarray(switch_stats.subject_switch_rate, dtype=np.float64)[:, None])
        names.append("switch_rate")
    if not blocks:
        raise ValueError("include selected no feature blocks")
    return np.concatenate(blocks, axis=1), names


def _standardize_train_apply(train: np.ndarray, test: np.ndarray):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd == 0] = 1.0
    return (train - mu) / sd, (test - mu) / sd


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Ridge on centered features; returns (weights, intercept)."""
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    f = x.shape[1]
    w = np.linalg.solve(xc.T @ xc + alpha * np.eye(f), xc.T @ yc)
    intercept = y.mean() - x.mean(axis=0) @ w
    return w, float(intercept)


@dataclass
class PredictionResult:
    """Leave-one-session-out regression of a continuous behaviour."""

    predicted: np.ndarray  # (S,) held-out predictions
    actual: np.ndarray  # (S,)
    r2: float  # held-out R^2
    correlation: float  # held-out Pearson r
    alpha: float
    coef: np.ndarray  # (F,) refit on all sessions (for interpretation)
    feature_names: list[str]


def cross_validated_prediction(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
    feature_names: list[str] | None = None,
) -> PredictionResult:
    """Leave-one-session-out ridge prediction of behaviour ``y`` from features ``x``.

    Standardises features on the training fold only (no leakage). Held-out R² uses
    the total-variance denominator, so a model no better than the mean scores ≤ 0.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    if n != y.shape[0]:
        raise ValueError("x and y must share the session axis")
    if n < 3:
        raise ValueError("need >= 3 sessions for leave-one-out prediction")
    pred = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        xtr, xte = _standardize_train_apply(x[tr], x[i : i + 1])
        w, b = _ridge_fit(xtr, y[tr], alpha)
        pred[i] = float(xte[0] @ w) + b
    resid = y - pred
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if np.std(pred) > 0 and np.std(y) > 0:
        corr = float(np.corrcoef(pred, y)[0, 1])
    else:
        corr = 0.0
    xz = (x - x.mean(axis=0)) / np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
    coef, _ = _ridge_fit(xz, y, alpha)
    return PredictionResult(
        predicted=pred,
        actual=y,
        r2=r2,
        correlation=corr,
        alpha=alpha,
        coef=coef,
        feature_names=feature_names or [f"f{j}" for j in range(x.shape[1])],
    )


@dataclass
class ClassificationResult:
    """Leave-one-session-out nearest-centroid classification (e.g. task decoding)."""

    predicted: np.ndarray  # (S,) predicted labels
    actual: np.ndarray  # (S,)
    accuracy: float  # held-out
    classes: np.ndarray
    confusion: np.ndarray  # (C, C), rows = true


def loso_classification(x: np.ndarray, labels) -> ClassificationResult:
    """Leave-one-session-out nearest-centroid decoding of ``labels`` from ``x``.

    Features are standardised on the training fold; each test session is assigned
    the nearest class centroid (Euclidean on z-scored features). Simple and
    dependency-free — appropriate for the small ``S`` of a dense-sampling design.
    """
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    n = x.shape[0]
    if n != labels.shape[0]:
        raise ValueError("x and labels must share the session axis")
    classes = np.unique(labels)
    pred = np.empty(n, dtype=labels.dtype)
    for i in range(n):
        tr = np.arange(n) != i
        xtr, xte = _standardize_train_apply(x[tr], x[i : i + 1])
        ytr = labels[tr]
        best, best_d = classes[0], np.inf
        for c in classes:
            member = xtr[ytr == c]
            if member.size == 0:
                continue
            dist = np.sum((xte[0] - member.mean(axis=0)) ** 2)
            if dist < best_d:
                best_d, best = dist, c
        pred[i] = best
    acc = float((pred == labels).mean())
    conf = np.zeros((classes.size, classes.size), dtype=np.int64)
    idx = {c: j for j, c in enumerate(classes)}
    for t, p in zip(labels, pred, strict=True):
        conf[idx[t], idx[p]] += 1
    return ClassificationResult(
        predicted=pred, actual=labels, accuracy=acc, classes=classes, confusion=conf
    )


@dataclass
class CCAResult:
    """SVD-based canonical correlation between state features and behaviour."""

    correlations: np.ndarray  # (n_comp,) canonical correlations
    x_weights: np.ndarray  # (Fx, n_comp)
    y_weights: np.ndarray  # (Fy, n_comp)
    x_scores: np.ndarray  # (S, n_comp)
    y_scores: np.ndarray  # (S, n_comp)


def canonical_correlation(
    x: np.ndarray, y: np.ndarray, *, n_components: int | None = None, reg: float = 1e-6
) -> CCAResult:
    """CCA between state features ``x`` ``(S, Fx)`` and behaviour ``y`` ``(S, Fy)``.

    Whitens each block via SVD (ridge-regularised for the small-``S`` case) then
    SVDs the cross-covariance — the standard closed form. Returns canonical
    correlations (descending) and per-block weights/scores.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    if n != y.shape[0]:
        raise ValueError("x and y must share the session axis")
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)

    # Whiten each block via its own SVD: the orthonormal left factors U are the
    # unit-variance canonical bases, so the singular values of Ux^T Uy are exactly
    # the canonical correlations (in [0, 1]) — no (n-1) rescaling.
    ux, sx, vxt = np.linalg.svd(xc, full_matrices=False)
    uy, sy, vyt = np.linalg.svd(yc, full_matrices=False)
    inv_sx = sx / (sx**2 + reg)  # regularised 1/s for the small-S case
    inv_sy = sy / (sy**2 + reg)
    p, sv, qt = np.linalg.svd(ux.T @ uy, full_matrices=False)
    r = sv.size if n_components is None else min(n_components, sv.size)
    corrs = np.clip(sv[:r], 0.0, 1.0)
    x_w = vxt.T @ (inv_sx[:, None] * p[:, :r])  # maps centered X -> canonical scores
    y_w = vyt.T @ (inv_sy[:, None] * qt.T[:, :r])
    return CCAResult(
        correlations=corrs,
        x_weights=x_w,
        y_weights=y_w,
        x_scores=xc @ x_w,
        y_scores=yc @ y_w,
    )
