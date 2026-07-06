"""Match states between two BSDS fits by functional-connectivity pattern.

State labels are arbitrary between two fits (k-means init + non-convex VB), so any
cross-fit comparison — ffs-vs-MATLAB (:mod:`.matlab_compare`) or ffs-vs-ffs
(:mod:`fastfuncstuff.dynamics.stability`) — must first align states. We align on
the **FC pattern**: the upper triangle of each state's correlation matrix, matched
by maximising total Pearson similarity across labels via the Hungarian algorithm.
"""

from __future__ import annotations

import numpy as np


def cov_to_corr_utri(cov: np.ndarray) -> np.ndarray:
    """Upper triangle (excl. diagonal) of the correlation matrix from a covariance."""
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(d, d)
    iu = np.triu_indices(cov.shape[0], k=1)
    return corr[iu]


def fc_similarity_matrix(covs_a: np.ndarray, covs_b: np.ndarray) -> np.ndarray:
    """``(Ka, Kb)`` Pearson correlation between the FC patterns of every state pair."""
    ua = np.stack([cov_to_corr_utri(c) for c in covs_a])  # (Ka, P)
    ub = np.stack([cov_to_corr_utri(c) for c in covs_b])  # (Kb, P)
    ua = ua - ua.mean(axis=1, keepdims=True)
    ub = ub - ub.mean(axis=1, keepdims=True)
    na = np.linalg.norm(ua, axis=1)
    nb = np.linalg.norm(ub, axis=1)
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    return (ua @ ub.T) / np.outer(na, nb)


def hungarian_match(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assignment maximising total FC similarity; returns ``(row_ind, col_ind)``."""
    from scipy.optimize import linear_sum_assignment

    cost = -np.nan_to_num(sim, nan=-1.0)
    return linear_sum_assignment(cost)


def model_state_covs(model) -> np.ndarray:
    """A fit's per-state covariances as a NumPy ``(K, D, D)`` array."""
    import torch

    covs = model.state_covs
    return covs.detach().cpu().numpy() if isinstance(covs, torch.Tensor) else np.asarray(covs)


def occupancy_from_viterbi(viterbi_states, n_states: int) -> np.ndarray:
    """Fractional occupancy pooled over a fit's per-run MAP paths (``K,``)."""
    import torch

    total = np.zeros(n_states)
    n = 0
    for v in viterbi_states:
        z = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        total += np.bincount(z.astype(np.int64), minlength=n_states)[:n_states]
        n += z.size
    return total / max(n, 1)
