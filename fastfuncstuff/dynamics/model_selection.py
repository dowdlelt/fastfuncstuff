"""Choosing ``n_states`` / ``max_ldim`` for BSDS via held-out log-likelihood.

Training free energy keeps improving as you add states or latent factors — it is
not a model-selection criterion. The empirical referee, matching the
``[[LORO cross-validation]]`` discipline used elsewhere in this codebase, is
held-out log-likelihood: fit on a subset of runs, ``decode`` the held-out run(s)
under the fixed model (E-step only, no refit), and score the fit by how well it
predicts data it never saw. A run is the natural exchangeability unit for BSDS
(never split within a run), the same way a run is the unit for GLM cross-run
nuisance projection.

This is deliberately a *grid* over both hyperparameters jointly: they interact
(a small ``max_ldim`` needs less data per state to be well-conditioned, so it
tolerates more states before starving any of them), so tuning one at a fixed
guess for the other can pick a spuriously bad value for the one held fixed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch

from fastfuncstuff.dynamics.bsds.model import decode, fit_bsds


@dataclass
class GridResult:
    """Held-out log-likelihood for one ``(n_states, max_ldim)`` combination."""

    n_states: int
    max_ldim: int
    held_out_loglik: float  # summed across folds
    per_timepoint_loglik: float  # normalised so configs are comparable
    fold_logliks: list[float]


def loro_held_out_loglik(
    sessions: list[torch.Tensor],
    n_states: int,
    max_ldim: int,
    *,
    n_folds: int | None = None,
    n_init: int = 5,
    n_init_iter: int = 10,
    n_iter: int = 60,
    tol: float = 1e-4,
    seed: int = 0,
    device: torch.device | None = None,
    show_progress: bool = False,
) -> GridResult:
    """Leave-runs-out held-out log-likelihood for one ``(n_states, max_ldim)``.

    Splits ``sessions`` into ``n_folds`` groups of whole runs (default: one run
    per fold, i.e. exact leave-one-run-out). For each fold, fits on the
    remaining runs and decodes the held-out run(s) under the fixed model,
    accumulating their log-likelihood. More folds is a better estimate but
    proportionally more fits.
    """
    n = len(sessions)
    n_folds = n if n_folds is None else min(n_folds, n)
    fold_ids = np.array_split(np.arange(n), n_folds)
    fold_logliks: list[float] = []
    total_frames = 0
    for held_idx in fold_ids:
        held = set(held_idx.tolist())
        train = [s for i, s in enumerate(sessions) if i not in held]
        test = [sessions[i] for i in held_idx]
        model = fit_bsds(
            train,
            n_states=n_states,
            max_ldim=max_ldim,
            n_init=n_init,
            n_init_iter=n_init_iter,
            n_iter=n_iter,
            tol=tol,
            seed=seed,
            device=device,
            show_progress=show_progress,
        )
        dec = decode(model, test, device=device)
        fold_logliks.append(float(dec.loglik))
        total_frames += sum(int(s.shape[1]) for s in test)
    total = float(sum(fold_logliks))
    return GridResult(
        n_states=n_states,
        max_ldim=max_ldim,
        held_out_loglik=total,
        per_timepoint_loglik=total / max(total_frames, 1),
        fold_logliks=fold_logliks,
    )


def grid_search_bsds(
    sessions: list[torch.Tensor],
    n_states_grid: list[int],
    max_ldim_grid: list[int],
    *,
    show_progress: bool = True,
    **kwargs,
) -> list[GridResult]:
    """Held-out log-likelihood over a ``(n_states, max_ldim)`` grid, best first.

    ``**kwargs`` are forwarded to :func:`loro_held_out_loglik` (``n_folds``,
    ``n_init``, ``n_iter``, ``device``, ...). Keep ``n_folds`` and ``n_init``
    modest for the initial coarse pass — this is ``len(grid) * n_folds`` full
    fits — then re-run the top candidates with a larger budget to confirm.
    """
    from tqdm.auto import tqdm

    combos = list(itertools.product(n_states_grid, max_ldim_grid))
    results = []
    for n_states, max_ldim in tqdm(combos, desc="bsds grid search", disable=not show_progress):
        results.append(loro_held_out_loglik(sessions, n_states, max_ldim, **kwargs))
    results.sort(key=lambda r: r.per_timepoint_loglik, reverse=True)
    return results
