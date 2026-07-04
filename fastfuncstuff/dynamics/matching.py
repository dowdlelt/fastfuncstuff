"""Match BSDS states across separate fits (Cai et al. 2024 state matching).

Within one group fit, all sessions already share the same state definitions, so
matching is only needed *between separate fits* — e.g. a reference n-back fit and
a fit of a different task, or two fits of the same subject. Cai et al. use two
complementary criteria:

- **State-space closeness** — match states by the similarity of their generative
  parameters (mean + covariance), via a symmetric KL divergence between the
  per-state Gaussians. Works even when the two fits were run on different data.
- **Temporal closeness** — match states by the similarity of their probability
  time courses, via correlation of the responsibilities. Requires both models to
  have been applied to the *same* runs so the time axes align.

Both return an optimal one-to-one assignment (Hungarian algorithm) plus the score
matrix, so you can also read off a single target state's best match (e.g. the
optimal WM state).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

_DTYPE = torch.float64


def gaussian_symmetric_kl(
    mean_a: torch.Tensor,
    cov_a: torch.Tensor,
    mean_b: torch.Tensor,
    cov_b: torch.Tensor,
) -> torch.Tensor:
    """Symmetric KL divergence between two multivariate Gaussians (scalar)."""
    mean_a = mean_a.to(_DTYPE)
    mean_b = mean_b.to(_DTYPE)
    cov_a = cov_a.to(_DTYPE)
    cov_b = cov_b.to(_DTYPE)
    d = mean_a.shape[0]
    inv_a = torch.linalg.inv(cov_a)
    inv_b = torch.linalg.inv(cov_b)
    diff = (mean_b - mean_a).unsqueeze(1)  # (D, 1)
    kl_ab = 0.5 * (
        torch.trace(inv_b @ cov_a) + (diff.T @ inv_b @ diff).squeeze() - d
    )
    diff2 = (mean_a - mean_b).unsqueeze(1)
    kl_ba = 0.5 * (
        torch.trace(inv_a @ cov_b) + (diff2.T @ inv_a @ diff2).squeeze() - d
    )
    # The log-det terms cancel in the symmetric sum.
    return kl_ab + kl_ba


def state_space_distance_matrix(model_a, model_b) -> torch.Tensor:
    """``(Ka, Kb)`` symmetric-KL distance between every pair of states."""
    ka, kb = model_a.n_states, model_b.n_states
    dist = torch.empty(ka, kb, dtype=_DTYPE)
    for i in range(ka):
        for j in range(kb):
            dist[i, j] = gaussian_symmetric_kl(
                model_a.state_means[i],
                model_a.state_covs[i],
                model_b.state_means[j],
                model_b.state_covs[j],
            )
    return dist


def temporal_similarity_matrix(
    resp_a: list[torch.Tensor],
    resp_b: list[torch.Tensor],
) -> torch.Tensor:
    """``(Ka, Kb)`` correlation between state probability time courses.

    ``resp_a``/``resp_b`` are per-session ``(T_i, K)`` responsibilities for the
    *same* sessions in the same order.
    """
    if len(resp_a) != len(resp_b):
        raise ValueError("resp_a and resp_b must cover the same sessions")
    a = torch.cat([r.to(_DTYPE) for r in resp_a], dim=0)  # (N, Ka)
    b = torch.cat([r.to(_DTYPE) for r in resp_b], dim=0)  # (N, Kb)
    if a.shape[0] != b.shape[0]:
        raise ValueError("responsibility time axes do not align")
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-12)
    b = b / b.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return a.T @ b  # (Ka, Kb) Pearson correlations


@dataclass
class StateMatch:
    """One-to-one matching from model A's states to model B's states."""

    method: str
    mapping: np.ndarray  # (min(Ka,Kb),) pairs; mapping[i] = B-state matched to A-state row_ind[i]
    row_ind: np.ndarray  # A-state indices
    col_ind: np.ndarray  # matched B-state indices
    score_matrix: torch.Tensor  # (Ka, Kb) distance (state_space) or similarity (temporal)

    def match_for(self, a_state: int) -> int | None:
        """B-state matched to a given A-state, or ``None`` if unmatched."""
        hit = np.flatnonzero(self.row_ind == a_state)
        return int(self.col_ind[hit[0]]) if hit.size else None


def match_states(
    model_a,
    model_b,
    *,
    method: str = "state_space",
    resp_a: list[torch.Tensor] | None = None,
    resp_b: list[torch.Tensor] | None = None,
) -> StateMatch:
    """Optimally match A's states to B's states (Hungarian assignment).

    ``method="state_space"`` uses symmetric-KL distance (minimised);
    ``method="temporal"`` uses responsibility correlation (maximised) and requires
    ``resp_a``/``resp_b`` for the same sessions.
    """
    if method == "state_space":
        score = state_space_distance_matrix(model_a, model_b)
        row_ind, col_ind = linear_sum_assignment(score.numpy())
    elif method == "temporal":
        if resp_a is None or resp_b is None:
            raise ValueError("temporal matching requires resp_a and resp_b")
        score = temporal_similarity_matrix(resp_a, resp_b)
        row_ind, col_ind = linear_sum_assignment(-score.numpy())  # maximise
    else:
        raise ValueError(f"unknown method: {method!r}")
    return StateMatch(
        method=method,
        mapping=col_ind.copy(),
        row_ind=row_ind,
        col_ind=col_ind,
        score_matrix=score,
    )
