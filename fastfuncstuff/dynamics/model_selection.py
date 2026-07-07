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
import os
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


# Set once per worker process (via the pool initializer) so the sessions payload
# is pickled to each worker a single time, not re-shipped for every grid point.
_WORKER_SESSIONS: list[torch.Tensor] | None = None


def _grid_worker_init(sessions: list[torch.Tensor], num_threads: int | None) -> None:
    global _WORKER_SESSIONS
    if num_threads is not None and num_threads >= 1:
        torch.set_num_threads(num_threads)
    _WORKER_SESSIONS = sessions


def _grid_worker(task: tuple[int, int, dict]) -> GridResult:
    n_states, max_ldim, kwargs = task
    assert _WORKER_SESSIONS is not None  # set by _grid_worker_init
    return loro_held_out_loglik(_WORKER_SESSIONS, n_states, max_ldim, **kwargs)


def grid_search_bsds(
    sessions: list[torch.Tensor],
    n_states_grid: list[int],
    max_ldim_grid: list[int],
    *,
    show_progress: bool = True,
    n_jobs: int = 1,
    **kwargs,
) -> list[GridResult]:
    """Held-out log-likelihood over a ``(n_states, max_ldim)`` grid, best first.

    ``**kwargs`` are forwarded to :func:`loro_held_out_loglik` (``n_folds``,
    ``n_init``, ``n_iter``, ``device``, ...). Keep ``n_folds`` and ``n_init``
    modest for the initial coarse pass — this is ``len(grid) * n_folds`` full
    fits — then re-run the top candidates with a larger budget to confirm.

    ``n_jobs > 1`` runs grid points concurrently in a ``spawn`` process pool
    (``fork`` cannot carry a live CUDA context). The grid points are independent
    fits, so this is embarrassingly parallel. On a single GPU the workers
    time-slice the card — worthwhile when one fit leaves it underused; on CPU
    each worker gets ``cpu_count // n_jobs`` threads. Results are identical to
    the sequential path (each grid point is deterministic in its seed); only the
    completion order differs, and the final list is sorted regardless.
    """
    from tqdm.auto import tqdm

    combos = list(itertools.product(n_states_grid, max_ldim_grid))
    n_jobs = max(1, min(n_jobs, len(combos)))

    if n_jobs == 1:
        results = [
            loro_held_out_loglik(sessions, n_states, max_ldim, **kwargs)
            for n_states, max_ldim in tqdm(
                combos, desc="bsds grid search", disable=not show_progress
            )
        ]
        results.sort(key=lambda r: r.per_timepoint_loglik, reverse=True)
        return results

    # Parallel: ship sessions on CPU (CUDA tensors don't pickle across spawn;
    # fit_bsds/decode move to `device` internally), one copy per worker.
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cpu_sessions = [s.detach().to("cpu") for s in sessions]
    device = kwargs.get("device")
    on_gpu = device is not None and str(device) != "cpu"
    # GPU workers: keep host dispatch lean (1 thread) so they don't oversubscribe
    # cores fighting to launch kernels. CPU workers: split the cores evenly.
    num_threads = 1 if on_gpu else max(1, (os.cpu_count() or 1) // n_jobs)

    tasks = [(n_states, max_ldim, kwargs) for n_states, max_ldim in combos]
    results = []
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        mp_context=mp.get_context("spawn"),
        initializer=_grid_worker_init,
        initargs=(cpu_sessions, num_threads),
    ) as pool:
        futures = [pool.submit(_grid_worker, t) for t in tasks]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="bsds grid search",
            disable=not show_progress,
        ):
            results.append(fut.result())
    results.sort(key=lambda r: r.per_timepoint_loglik, reverse=True)
    return results
