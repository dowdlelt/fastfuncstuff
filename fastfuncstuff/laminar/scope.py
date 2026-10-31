"""The laminar model space over many regions at once: what a column reports.

:func:`~fastfuncstuff.laminar.pipeline.fit_roi` for a batch. Every column is
inverted under every layer hypothesis at every vascular resolution K, each
model's posteriors are averaged across K, and the averaged models are compared
by free energy -- the published analysis, run by
:func:`~fastfuncstuff.laminar.batched.variational_laplace_batched` so one model
at one K is one batched call over all columns.

What is kept per column is what the maps need: free energy and probability of
every model, P(depth modulated) marginalised over the model space, the best
model's modulation B with its uncertainty, variance explained, and flags for
fits to distrust (stalled, failed, nesting violations). Deveined time courses
and operators stay per ROI in ``ffs_laminar``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from fastfuncstuff.laminar.batched import BatchedVL, bpa_batched, variational_laplace_batched
from fastfuncstuff.laminar.experiment import faes_priors, layer_model_names, layer_model_targets
from fastfuncstuff.laminar.inversion import unvec
from fastfuncstuff.laminar.io import LaminarData


@dataclass
class ScopeFits:
    """Per column (rows) results of one model space."""

    models: list[str]
    targets: list[tuple[int, ...]]
    ks: list[int]
    F: np.ndarray  # (C, M) averaged models
    per_k_F: np.ndarray  # noqa: N815  (C, M, nK)
    p_model: np.ndarray  # (C, M)
    p_depth: np.ndarray  # (C, N)
    best: np.ndarray  # (C,)
    B_mean: np.ndarray  # (C, N) best model, averaged posterior; 0 where not in the model
    B_sd: np.ndarray  # (C, N)
    B_p: np.ndarray  # (C, N) P(B != 0)
    r2: np.ndarray  # (C, M) first K, fitted rows
    stalled: np.ndarray  # (C, M, nK)
    failed: np.ndarray  # (C, M, nK)
    nesting: np.ndarray  # (C,) worst shortfall of a model below one it contains (nats)
    #: The best model's ascending-vein slope s_d (posterior mean, log-deviation from the
    #: prior): the drainage parameter whose UNDERestimate gives wrong profiles.
    s_d: np.ndarray = field(default_factory=lambda: np.zeros(0))
    timing: dict = field(default_factory=dict)


def _r2(run: BatchedVL, y: torch.Tensor, rows: torch.Tensor) -> np.ndarray:
    y0 = y[:, rows].to(run.y_pred.device, run.y_pred.dtype)
    yp = run.y_pred[:, rows]
    ss_tot = ((y0 - y0.mean((1, 2), keepdim=True)) ** 2).sum((1, 2))
    ss_res = ((y0 - yp) ** 2).sum((1, 2))
    return (1.0 - ss_res / torch.clamp(ss_tot, min=1e-30)).cpu().numpy()


def fit_columns(
    data_for_k: dict[int, LaminarData],
    kernel_for_k: dict[int, torch.Tensor | None] | None = None,
    *,
    targets: list[tuple[int, ...]] | None = None,
    names: list[str] | None = None,
    free_energy: str = "reference",
    max_iter: int = 128,
    hyper_from_rows: bool = False,
    device: torch.device | str | None = None,
    chunk: int | None = None,
    progress: bool = True,
    rescue: int = 0,
) -> ScopeFits:
    """Invert, average across K and compare every column of ``data_for_k``.

    ``data_for_k[K].y`` is ``(C, n_scans, K)`` (from
    :func:`~fastfuncstuff.laminar.io.two_condition_batch`), the same columns in
    the same order at every K. ``kernel_for_k[K]`` is the depth PSF: ``(C, K, K)``
    per column, one shared kernel, or None. ``chunk`` columns go through one
    batched call (None: all of them).
    """
    import time

    ks = sorted(data_for_k)
    first = data_for_k[ks[0]]
    n_cols = int(first.y.shape[0])
    for k in ks[1:]:
        if int(data_for_k[k].y.shape[0]) != n_cols or not torch.equal(
            data_for_k[k].rows, first.rows
        ):
            raise ValueError("every K must hold the same columns and the same design")
    n = first.spec.N
    if targets is None:
        targets = layer_model_targets(n)
    if names is None:
        all_names, all_targets = layer_model_names(n), layer_model_targets(n)
        names = [all_names[all_targets.index(tuple(t))] for t in targets]
    kernels = kernel_for_k or {k: None for k in ks}
    chunk = chunk or n_cols
    n_m, n_k = len(targets), len(ks)

    F = np.full((n_cols, n_m), -np.inf)
    per_k_F = np.full((n_cols, n_m, n_k), -np.inf)
    r2 = np.full((n_cols, n_m), np.nan)
    stalled = np.zeros((n_cols, n_m, n_k), bool)
    failed = np.zeros((n_cols, n_m, n_k), bool)
    b_stats = np.zeros((3, n_m, n_cols, n))  # mean, sd, P per model, picked after
    s_d_all = np.zeros((n_m, n_cols))
    timing: dict[str, float] = {}

    bar = None
    if progress:
        from tqdm.auto import tqdm

        n_chunks = -(-n_cols // chunk)
        bar = tqdm(total=n_m * n_k * n_chunks, desc=f"laminar {n_cols} columns", leave=True,
                   disable=n_m * n_k * n_chunks < 2)  # fmt: skip
    for m, target in enumerate(targets):
        for c0 in range(0, n_cols, chunk):
            c1 = min(n_cols, c0 + chunk)
            runs = []
            for j, k in enumerate(ks):
                d = data_for_k[k]
                kern = kernels.get(k)
                if kern is not None and kern.ndim == 3:
                    kern = kern[c0:c1]
                t0 = time.perf_counter()
                run = variational_laplace_batched(
                    d.spec, faes_priors(d.spec, target), d.u, d.y[c0:c1], rows=d.rows,
                    kernel=kern, max_iter=max_iter, hyper_from_rows=hyper_from_rows,
                    device=device, rescue=rescue, **d.integrate_kwargs,
                )  # fmt: skip
                timing[f"{names[m]}/K{k}"] = (
                    timing.get(f"{names[m]}/K{k}", 0.0) + time.perf_counter() - t0
                )
                runs.append(run)
                stalled[c0:c1, m, j] = run.stalled.cpu().numpy()
                failed[c0:c1, m, j] = run.failed.cpu().numpy()
                if j == 0:
                    r2[c0:c1, m] = _r2(run, d.y[c0:c1], d.rows)
                if bar is not None:
                    bar.update(1)
            bpa = bpa_batched(runs, free_energy=free_energy)
            f = bpa.F.cpu().numpy()
            bad = failed[c0:c1, m].any(1)
            F[c0:c1, m] = np.where(bad, -np.inf, f)
            per_k_F[c0:c1, m] = bpa.per_k_F.cpu().numpy()
            spec = runs[0].spec
            s_d_all[m, c0:c1] = unvec(bpa.Ep_vec, spec)["s_d"].reshape(-1).cpu().numpy()
            for s, vecs in enumerate((bpa.Ep_vec, bpa.Vp_vec, bpa.Pp_vec)):
                B = unvec(vecs, spec)["B"][:, 0]  # (b, N, N)
                v = torch.diagonal(B, dim1=-2, dim2=-1).cpu().numpy()
                b_stats[s, m, c0:c1] = np.sqrt(np.clip(v, 0, None)) if s == 1 else v
    if bar is not None:
        bar.close()

    # Softmax over models, shifted by the max (see posterior_model_probabilities).
    fin = np.isfinite(F)
    shift = np.where(fin.any(1), np.where(fin, F, -np.inf).max(1), 0.0)
    e = np.where(fin, np.exp(np.where(fin, F, 0.0) - shift[:, None]), 0.0)
    tot = e.sum(1, keepdims=True)
    p_model = np.divide(e, tot, out=np.zeros_like(e), where=tot > 0)
    membership = np.asarray(targets, np.float64)  # (M, N)
    p_depth = p_model @ membership
    best = np.where(fin.any(1), np.argmax(np.where(fin, F, -np.inf), axis=1), -1)
    idx = np.arange(n_cols)
    pick = np.clip(best, 0, None)
    b_mean, b_sd, b_p = (b_stats[s, pick, idx] for s in range(3))

    # Nesting: a model can never truly score below a model it contains.
    nesting = np.zeros(n_cols)
    for i, ti in enumerate(targets):
        for j, tj in enumerate(targets):
            if i != j and all(a >= b for a, b in zip(ti, tj, strict=True)):
                gap = np.where(fin[:, i] & fin[:, j], F[:, j] - F[:, i], 0.0)
                nesting = np.maximum(nesting, gap)
    return ScopeFits(
        list(names), [tuple(t) for t in targets], ks, F, per_k_F, p_model, p_depth, best,
        b_mean, b_sd, b_p, r2, stalled, failed, nesting, s_d_all[pick, idx], timing,
    )  # fmt: skip


__all__ = ["ScopeFits", "fit_columns"]
