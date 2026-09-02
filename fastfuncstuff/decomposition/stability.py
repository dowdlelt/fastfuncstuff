"""Empirical model order: how many components actually reproduce?

Paper-derived. See ``../fmri_wiki/notes/FSL clean-room policy.md``.

References
----------
- Himberg, J., Hyvärinen, A. & Esposito, F. (2004). *Validating the independent
  components of neuroimaging time series via clustering and visualization*.
  NeuroImage 22(3):1214-1222.

Why this exists alongside :mod:`fastfuncstuff.decomposition.model_order`
------------------------------------------------------------------------
``model_order`` answers the question from the **eigenspectrum**: Marchenko-Pastur says
how many directions are inconsistent with noise, and Minka's evidence picks within that.
It never runs an ICA. That is cheap and principled, and it is the default.

It is also answering a *PCA* question. Every criterion in that family -- Laplace, AIC,
BIC, MDL -- is a function of the eigenvalues alone, and the ICA rotation is orthogonal
within the whitened subspace, so it leaves the covariance and hence all of them
unchanged. Computing several of them looks like several opinions but is one estimate with
different penalty terms; they fail together, not independently.

This module answers the question **empirically**, at the level of the ICA itself, which
costs many decompositions and is therefore opt-in. Two measurements, which are not the
same thing and fail differently:

- :func:`stability_model_order` measures **optimisation** variability -- run the ICA many
  times from different initialisations and see which components survive as tight clusters.
  **This turns out to be a poor model-order estimator, and the failure is silent.**
  Measured on rank-6 data at ``k_max=12``: eleven clusters clear ``Iq=0.7`` and nine
  exceed 0.97. For a *fixed* dataset the whitened subspace is fixed too, so FastICA
  converges to essentially the same answer from any start -- on the noise directions as
  much as the real ones. The numbers look superb and mean little. Use it as a diagnostic
  of whether the decomposition converged, which is what ICASSO is for, not as a count.
- :func:`split_half_reproducibility` measures **sampling** variability -- decompose two
  disjoint halves of the data independently and count how many components match across
  them. This one works: on the same synthetic data it recovers the planted components and
  returns at most one "reproducible" component from pure noise.

The distinction is the whole point. A component can be perfectly seed-stable and not
reproduce on held-out runs, because seed stability never varies the data. **If you run
only one of these, run the split-half.**
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "ReproducibilityResult",
    "StabilityOrderResult",
    "match_components",
    "split_half_reproducibility",
    "stability_model_order",
]


@dataclass
class StabilityOrderResult:
    """Model order chosen by counting reproducible clusters."""

    k: int
    """Number of components whose cluster is tight enough to keep."""

    k_max: int
    """Generous upper bound the restarts were run at."""

    iq: np.ndarray = field(default_factory=lambda: np.empty(0))
    """Cluster stability index per component, descending. The knee is the diagnostic."""

    min_stability: float = 0.7
    n_runs: int = 0

    def as_dict(self) -> dict:
        return {
            "k": int(self.k),
            "k_max": int(self.k_max),
            "min_stability": float(self.min_stability),
            "n_runs": int(self.n_runs),
            "iq": np.asarray(self.iq, dtype=float).tolist(),
        }


@dataclass
class ReproducibilityResult:
    """How many components survive an independent decomposition of held-out data."""

    n_reproducible: int
    matched_r: np.ndarray = field(default_factory=lambda: np.empty(0))
    """|correlation| of each matched pair, descending."""

    threshold: float = 0.5
    n_splits: int = 1

    def as_dict(self) -> dict:
        return {
            "n_reproducible": int(self.n_reproducible),
            "threshold": float(self.threshold),
            "n_splits": int(self.n_splits),
            "matched_r": np.asarray(self.matched_r, dtype=float).tolist(),
        }


def stability_model_order(
    X: np.ndarray | Tensor,
    k_max: int,
    *,
    n_runs: int = 30,
    min_stability: float = 0.7,
    device: torch.device | None = None,
    base_seed: int = 0,
    pca_components: int | float | str | None = None,
    verbose: bool = False,
) -> StabilityOrderResult:
    """Cluster ``n_runs`` restarts at ``k_max`` and count the tight clusters.

    **Read the module docstring before using ``k`` as a model order.** Restart stability
    saturates: on a fixed dataset the whitened subspace is fixed, FastICA converges to
    nearly the same solution from any start, and noise directions come back as stable as
    real ones. Measured on rank-6 data at ``k_max=12``, this returns 11.

    What it is good for is the ``iq`` curve, which is a genuine convergence diagnostic --
    a component with low Iq did not converge reproducibly and should not be trusted
    whatever else says it is real. For an actual count, use
    :func:`split_half_reproducibility`.
    """
    from .icasso import icasso

    res = icasso(
        X,
        n_components=int(k_max),
        n_runs=int(n_runs),
        pca_components=pca_components,
        min_stability=float(min_stability),
        device=device,
        verbose=verbose,
        base_seed=base_seed,
        mode="randinit",
    )
    iq = np.asarray(res["all_stability"], dtype=float)
    k = int((iq >= min_stability).sum())
    return StabilityOrderResult(
        k=max(1, k),
        k_max=int(k_max),
        iq=iq,
        min_stability=float(min_stability),
        n_runs=int(n_runs),
    )


def match_components(a_kv: np.ndarray, b_kv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match two sets of spatial maps one-to-one by absolute correlation.

    Returns ``(pairs, r)`` where ``pairs`` is ``(n, 2)`` of matched indices and ``r`` the
    matched |correlation| values, both sorted by ``r`` descending.

    Assignment is global (Hungarian), not greedy: greedy matching lets one strong
    component claim a partner that a different pair needed, which inflates the top of the
    list and depresses the tail. ICA components carry an arbitrary sign, so the magnitude
    of the correlation is what is being matched.
    """
    from scipy.optimize import linear_sum_assignment

    a = np.asarray(a_kv, dtype=np.float64)
    b = np.asarray(b_kv, dtype=np.float64)
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-12)
    b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-12)
    corr = np.abs(a @ b.T) / a.shape[1]  # (ka, kb)

    rows, cols = linear_sum_assignment(-corr)
    r = corr[rows, cols]
    order = np.argsort(-r)
    return np.stack([rows[order], cols[order]], axis=1), r[order]


def split_half_reproducibility(
    runs: list[np.ndarray | Tensor],
    n_components: int,
    *,
    threshold: float = 0.5,
    n_splits: int = 1,
    device: torch.device | None = None,
    base_seed: int = 0,
    pca_components: int | float | str | None = None,
    verbose: bool = False,
) -> ReproducibilityResult:
    """Decompose two disjoint halves of the runs and count components that match.

    ``runs`` is a list of ``(T_i, V)`` arrays on a common voxel grid. Halves are formed by
    splitting **runs**, not timepoints: neighbouring timepoints are autocorrelated and
    voxels are spatially smooth, so splitting either of those leaks the same data into
    both halves and the reproducibility comes back flattering and meaningless. Runs are
    the coarsest unit that is plausibly independent.

    With ``n_splits > 1`` the split is repeated with different random partitions and the
    matched correlations are averaged over splits, which matters when the run count is
    small and one particular partition can be unlucky.
    """
    from .ica import FastICA

    if len(runs) < 2:
        raise ValueError(f"need at least 2 runs to split, got {len(runs)}")

    def _decompose(idx: list[int]) -> np.ndarray:
        parts = [torch.as_tensor(np.asarray(runs[i]), dtype=torch.float32) for i in idx]
        data = torch.cat(parts, dim=0)
        ica = FastICA(
            n_components=n_components,
            pca_components=pca_components,
            random_state=base_seed,
            device=device,
        )
        ica.verbose = False  # type: ignore[attr-defined]
        ica.fit(data)
        assert ica.components_ is not None
        return ica.components_.cpu().numpy()

    all_r: list[np.ndarray] = []
    rng = np.random.default_rng(base_seed)
    order = np.arange(len(runs))
    for s in range(int(n_splits)):
        if s > 0:
            rng.shuffle(order)
        half = len(runs) // 2
        a_idx = [int(i) for i in order[:half]]
        b_idx = [int(i) for i in order[half:]]
        if verbose:
            print(f"  split {s + 1}/{n_splits}: runs {a_idx} vs {b_idx}")
        _, r = match_components(_decompose(a_idx), _decompose(b_idx))
        all_r.append(r)

    n_keep = min(len(r) for r in all_r)
    matched = np.mean([r[:n_keep] for r in all_r], axis=0)
    matched = np.sort(matched)[::-1]
    return ReproducibilityResult(
        n_reproducible=int((matched >= threshold).sum()),
        matched_r=matched,
        threshold=float(threshold),
        n_splits=int(n_splits),
    )
