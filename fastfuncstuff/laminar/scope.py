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

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import torch

from fastfuncstuff.laminar.batched import (
    BatchedVL,
    bpa_batched,
    predict_batched,
    reduce_batched,
    variational_laplace_batched,
)
from fastfuncstuff.laminar.empirical import EmpiricalPrior, batch_priors, empirical_vascular_prior
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
    #: K -> the empirical vascular prior applied (vascular_prior != "none").
    empirical: dict = field(default_factory=dict)


def _r2(run: BatchedVL, y: torch.Tensor, rows: torch.Tensor) -> np.ndarray:
    y0 = y[:, rows].to(run.y_pred.device, run.y_pred.dtype)
    yp = run.y_pred[:, rows]
    ss_tot = ((y0 - y0.mean((1, 2), keepdim=True)) ** 2).sum((1, 2))
    ss_res = ((y0 - yp) ** 2).sum((1, 2))
    return (1.0 - ss_res / torch.clamp(ss_tot, min=1e-30)).cpu().numpy()


def _cat_runs(parts: list[BatchedVL]) -> BatchedVL:
    """Chunks of one model's batched inversion, joined along the dataset axis."""
    if len(parts) == 1:
        return parts[0]
    out = dataclasses.replace(parts[0])
    for f in dataclasses.fields(BatchedVL):
        if f.name not in ("free_idx", "spec"):
            setattr(out, f.name, torch.cat([getattr(p, f.name) for p in parts]))
    return out


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
    model_space: str = "invert",
    vascular_prior: str = "none",
    vascular_radius_mm: float = 10.0,
    centres: np.ndarray | None = None,
    hemi: np.ndarray | None = None,
) -> ScopeFits:
    """Invert, average across K and compare every column of ``data_for_k``.

    ``data_for_k[K].y`` is ``(C, n_scans, K)`` (from
    :func:`~fastfuncstuff.laminar.io.two_condition_batch`), the same columns in
    the same order at every K. ``kernel_for_k[K]`` is the depth PSF: ``(C, K, K)``
    per column, one shared kernel, or None. ``chunk`` columns go through one
    batched call (None: all of them).

    ``model_space``: ``"invert"`` inverts every model separately, as the published
    analysis does. ``"reduce"`` inverts only the full model (every depth modulated)
    and derives the others by Bayesian model reduction
    (:func:`~fastfuncstuff.laminar.batched.reduce_batched`). That is 2^N-fold fewer
    inversions, and nested evidences come out consistent. On the published data it
    gives the same ranking of all seven nested models, with competitive models within
    1.3-1.6 nats of brute force.

    ``vascular_prior`` (reduce only): ``"local"`` gives each column an empirical prior
    on the vascular parameters from the columns within ``vascular_radius_mm`` of its
    centre (``centres`` ``(C, 3)`` mm, ``hemi`` ``(C,)``); ``"global"`` pools a whole
    hemisphere (:mod:`~fastfuncstuff.laminar.empirical`).
    """
    if model_space not in ("invert", "reduce"):
        raise ValueError(f"model_space must be 'invert' or 'reduce', not {model_space!r}")
    if vascular_prior not in ("none", "local", "global"):
        raise ValueError(f"vascular_prior must be none, local or global, not {vascular_prior!r}")
    if vascular_prior != "none" and model_space != "reduce":
        raise ValueError(
            "an empirical vascular prior is applied by reduction: model_space='reduce'"
        )
    empirical: dict[int, EmpiricalPrior] = {}
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
        n_inv = n_k * n_chunks * (1 if model_space == "reduce" else n_m)
        bar = tqdm(total=n_inv, desc=f"laminar {n_cols} columns", leave=True,
                   disable=n_inv < 2)  # fmt: skip
    full_target = tuple([1] * n)

    def invert(target, k, label, c0, c1):
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
        timing[f"{label}/K{k}"] = timing.get(f"{label}/K{k}", 0.0) + time.perf_counter() - t0
        if bar is not None:
            bar.update(1)
        return run

    def record(m, c0, c1, runs):
        for j, (k, run) in enumerate(zip(ks, runs, strict=True)):
            stalled[c0:c1, m, j] = run.stalled.cpu().numpy()
            failed[c0:c1, m, j] = run.failed.cpu().numpy()
            if j == 0:
                r2[c0:c1, m] = _r2(run, data_for_k[k].y[c0:c1], data_for_k[k].rows)
        bpa = bpa_batched(runs, free_energy=free_energy)
        bad = failed[c0:c1, m].any(1)
        F[c0:c1, m] = np.where(bad, -np.inf, bpa.F.cpu().numpy())
        per_k_F[c0:c1, m] = bpa.per_k_F.cpu().numpy()
        spec = runs[0].spec
        s_d_all[m, c0:c1] = unvec(bpa.Ep_vec, spec)["s_d"].reshape(-1).cpu().numpy()
        for s, vecs in enumerate((bpa.Ep_vec, bpa.Vp_vec, bpa.Pp_vec)):
            B = unvec(vecs, spec)["B"][:, 0]  # (b, N, N)
            v = torch.diagonal(B, dim1=-2, dim2=-1).cpu().numpy()
            b_stats[s, m, c0:c1] = np.sqrt(np.clip(v, 0, None)) if s == 1 else v

    chunks = [(c0, min(n_cols, c0 + chunk)) for c0 in range(0, n_cols, chunk)]
    if model_space == "invert":
        for c0, c1 in chunks:
            for m, target in enumerate(targets):
                record(m, c0, c1, [invert(target, k, names[m], c0, c1) for k in ks])
    else:
        full_all = {k: _cat_runs([invert(full_target, k, "full", c0, c1) for c0, c1 in chunks])
                    for k in ks}  # fmt: skip
        overrides: dict[int, dict] = {k: {} for k in ks}
        if vascular_prior != "none":
            if centres is None or hemi is None:
                raise ValueError("an empirical vascular prior needs the column centres")
            for k in ks:
                ep = empirical_vascular_prior(
                    full_all[k], faes_priors(data_for_k[k].spec, full_target), centres, hemi,
                    radius_mm=None if vascular_prior == "global" else vascular_radius_mm,
                )  # fmt: skip
                empirical[k] = ep
                for j, name in enumerate(ep.names):
                    overrides[k][f"pE:{name}"] = torch.as_tensor(ep.mean[:, j])
                    overrides[k][f"pC:{name}"] = ep.priors.pC[name].reshape(-1)
        for m, target in enumerate(targets):
            runs = []
            for j, k in enumerate(ks):
                d = data_for_k[k]
                full = full_all[k]
                if tuple(target) == full_target and not overrides[k]:
                    runs.append(full)
                    continue
                pri = faes_priors(d.spec, target)
                if overrides[k]:
                    pri = batch_priors(pri, n_cols, overrides[k])
                run = reduce_batched(full, faes_priors(d.spec, full_target), pri)
                if j == 0:  # one forward pass, for the variance explained
                    run = dataclasses.replace(run, y_pred=predict_batched(
                        d.spec, run.Ep_vec, d.u, int(d.y.shape[1]), kernel=kernels.get(k),
                        **d.integrate_kwargs))  # fmt: skip
                runs.append(run)
            record(m, 0, n_cols, runs)
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
        b_mean, b_sd, b_p, r2, stalled, failed, nesting, s_d_all[pick, idx], timing, empirical,
    )  # fmt: skip


__all__ = ["ScopeFits", "fit_columns"]
