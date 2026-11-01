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
from fastfuncstuff.laminar.empirical import (
    VASCULAR,
    EmpiricalPrior,
    batch_priors,
    empirical_vascular_prior,
)
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
    #: The best model's averaged posterior over the vascular parameters, name ->
    #: (mean (C,), sd (C,)), for carrying to another fit as a prior (``vascular_given``).
    vascular: dict = field(default_factory=dict)
    #: B averaged over the model space with the model probabilities (Bayesian model
    #: averaging): a model without depth k contributes 0 there. Prefer it to the
    #: winner's B for group tests -- the winner over-selects.
    B_bma: np.ndarray = field(default_factory=lambda: np.zeros(0))
    B_bma_sd: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Global neuronal self-inhibition sigma (Hz), model-averaged; the test condition's
    #: self-inhibition at depth k is sigma - B_bma[:, k] (A_kk = -sigma + u B_k).
    sigma_hz: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: (C, M) models whose reduced posterior was not integrable and were inverted
    #: instead (or withheld, under per-column priors).
    refit: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), bool))
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


def _replace_rows(run: BatchedVL, idx: torch.Tensor, sub: BatchedVL) -> BatchedVL:
    """``run`` with datasets ``idx`` replaced by ``sub`` (same parameter structure)."""
    if not torch.equal(run.free_idx.cpu(), sub.free_idx.cpu()):
        raise ValueError("a refit must estimate the same parameters as the reduction")
    out = dataclasses.replace(run)
    for f in dataclasses.fields(BatchedVL):
        if f.name in ("free_idx", "spec"):
            continue
        whole = getattr(run, f.name).clone()
        whole[idx.to(whole.device)] = getattr(sub, f.name).to(whole.device, whole.dtype)
        setattr(out, f.name, whole)
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
    vascular_given: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
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
    hemisphere (:mod:`~fastfuncstuff.laminar.empirical`). ``vascular_given`` instead
    sets each column's prior on the vascular parameters from another fit's posterior
    (name -> (mean, variance) per column, NaN to keep the original), e.g. a high-SNR
    localiser on the same tissue: the vasculature does not change with the task.
    """
    if model_space not in ("invert", "reduce"):
        raise ValueError(f"model_space must be 'invert' or 'reduce', not {model_space!r}")
    if vascular_prior not in ("none", "local", "global"):
        raise ValueError(f"vascular_prior must be none, local or global, not {vascular_prior!r}")
    if (vascular_prior != "none" or vascular_given) and model_space != "reduce":
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
    refit = np.zeros((n_cols, n_m), bool)
    vasc_all = {name: np.zeros((2, n_m, n_cols)) for name in VASCULAR}
    sigma_all = np.zeros((n_m, n_cols))
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

    def invert_idx(target, k, label, idx):
        d = data_for_k[k]
        kern = kernels.get(k)
        if kern is not None and kern.ndim == 3:
            kern = kern[idx.cpu()]
        t0 = time.perf_counter()
        run = variational_laplace_batched(
            d.spec, faes_priors(d.spec, target), d.u, d.y[idx.cpu()], rows=d.rows,
            kernel=kern, max_iter=max_iter, hyper_from_rows=hyper_from_rows,
            device=device, rescue=rescue, **d.integrate_kwargs,
        )  # fmt: skip
        timing[f"{label}/K{k}"] = timing.get(f"{label}/K{k}", 0.0) + time.perf_counter() - t0
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
        ep_d, vp_d = unvec(bpa.Ep_vec, spec), unvec(bpa.Vp_vec, spec)
        sigma_all[m, c0:c1] = (spec.p0.sigma * torch.exp(ep_d["sigma"].reshape(-1))).cpu().numpy()
        for name in VASCULAR:
            vasc_all[name][0, m, c0:c1] = ep_d[name].reshape(-1).cpu().numpy()
            vasc_all[name][1, m, c0:c1] = np.sqrt(
                np.clip(vp_d[name].reshape(-1).cpu().numpy(), 0, None)
            )
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
        if vascular_given:
            for k in ks:
                pr0 = faes_priors(data_for_k[k].spec, full_target)
                for name, (mean, var) in vascular_given.items():
                    mean, var = np.asarray(mean, np.float64), np.asarray(var, np.float64)
                    ok = np.isfinite(mean) & np.isfinite(var) & (var > 0)
                    # A column the other fit lacks keeps the original prior; the prior
                    # only narrows, which keeps every reduction well posed.
                    pE0, pC0 = float(pr0.pE[name]), float(pr0.pC[name])
                    overrides[k][f"pE:{name}"] = torch.as_tensor(np.where(ok, mean, pE0))
                    overrides[k][f"pC:{name}"] = torch.as_tensor(
                        np.where(ok, np.minimum(var, pC0), pC0)
                    )
        elif vascular_prior != "none":
            if centres is None or hemi is None:
                raise ValueError("an empirical vascular prior needs the column centres")
            for k in ks:
                ep = empirical_vascular_prior(
                    full_all[k], faes_priors(data_for_k[k].spec, full_target), centres, hemi,
                    radius_mm=None if vascular_prior == "global" else vascular_radius_mm,
                )  # fmt: skip
                empirical[k] = ep
                if ep.shrink == 0.0:
                    continue
                for j, name in enumerate(ep.names):
                    overrides[k][f"pE:{name}"] = torch.as_tensor(ep.mean[:, j])
                    overrides[k][f"pC:{name}"] = ep.priors.pC[name].reshape(-1)
        for m, target in enumerate(targets):
            runs = []
            for k in ks:
                d = data_for_k[k]
                full = full_all[k]
                if tuple(target) == full_target and not overrides[k]:
                    runs.append(full)
                    continue
                pri = faes_priors(d.spec, target)
                if overrides[k]:
                    pri = batch_priors(pri, n_cols, overrides[k])
                run = reduce_batched(full, faes_priors(d.spec, full_target), pri)
                # The reduced posterior must at least be integrable: where it is not (a
                # net self-excitation held through a long block grows without bound),
                # the Laplace step from the full posterior has failed, and that model is
                # inverted properly for those columns instead.
                pred = predict_batched(d.spec, run.Ep_vec, d.u, int(d.y.shape[1]),
                                       kernel=kernels.get(k), **d.integrate_kwargs)  # fmt: skip
                bad = ~torch.isfinite(pred).all(-1).all(-1)
                if bool(bad.any()):
                    idx = torch.nonzero(bad).flatten()
                    refit[idx.cpu().numpy(), m] = True
                    if overrides[k]:
                        # Per-column priors cannot go through a shared-prior inversion;
                        # the flag stands and the model's evidence is withheld.
                        run = dataclasses.replace(run, F=torch.where(bad, -torch.inf, run.F))
                    else:
                        sub = invert_idx(target, k, f"{names[m]} (refit)", idx)
                        run = _replace_rows(run, idx, sub)
                        pred = pred.clone()
                        pred[idx] = sub.y_pred.to(pred)
                run = dataclasses.replace(run, y_pred=pred)
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
    # Bayesian model averaging over the space: a mixture of the models' posteriors.
    pm = p_model.T[:, :, None]  # (M, C, 1)
    m1 = b_stats[0]  # (M, C, N); exactly 0 where a model pins the depth
    m2 = b_stats[1] ** 2 + m1**2
    b_bma = (pm * m1).sum(0)
    b_bma_sd = np.sqrt(np.clip((pm * m2).sum(0) - b_bma**2, 0, None))
    sigma_hz = (p_model.T * sigma_all).sum(0)

    # Nesting: a model can never truly score below a model it contains.
    nesting = np.zeros(n_cols)
    for i, ti in enumerate(targets):
        for j, tj in enumerate(targets):
            if i != j and all(a >= b for a, b in zip(ti, tj, strict=True)):
                gap = np.where(fin[:, i] & fin[:, j], F[:, j] - F[:, i], 0.0)
                nesting = np.maximum(nesting, gap)
    return ScopeFits(
        list(names), [tuple(t) for t in targets], ks, F, per_k_F, p_model, p_depth, best,
        b_mean, b_sd, b_p, r2, stalled, failed, nesting, s_d_all[pick, idx],
        {name: (v[0, pick, idx], v[1, pick, idx]) for name, v in vasc_all.items()}, b_bma,
        b_bma_sd, sigma_hz, refit, timing, empirical,
    )  # fmt: skip


LONG_COLUMNS = (
    "subject", "comparison", "region", "name", "hemi", "n_vertices", "area_mm2",
    "in_fov", "baseline_peak", "neuronal_depth", "p_modulated", "B_bma", "B_bma_sd", "best_model", "p_best",
    "B_best", "B_best_sd", "P_B_best", "sigma_hz", "sigma_test_hz", "s_d", "r2_best",
    "stalled", "failed", "nesting_nats", "n_refit",
)  # fmt: skip


def write_long_table(
    path: str,
    fits: ScopeFits,
    *,
    subject: str,
    comparison: str,
    region_ids: np.ndarray,
    names: list[str],
    hemis: list[str],
    n_vertices: np.ndarray,
    area_mm2: np.ndarray,
    depth_names: list[str],
    in_fov: np.ndarray | None = None,
    baseline_peak: np.ndarray | None = None,
) -> str:
    """One line per region x neuronal depth, tidy, for group tests: concatenate the
    subjects' tables and test B_bma (or p_modulated, sigma_test_hz) per region and
    depth. The rows of ``fits`` are the regions, in the order of the other arguments.

    ``in_fov`` is the share of the region's ribbon that was read (a slab can cover
    part of V1), and ``baseline_peak`` the peak of the depth-averaged baseline curve
    (percent): a region with no response has nothing for the model to explain, and
    its B values are noise however well flagged.

    ``sigma_test_hz`` is the self-inhibition during the test condition,
    sigma - B_bma: positive B lowers inhibition (A_kk = -sigma + u B_k), and a
    negative value is net self-excitation while the modulation is on -- what the
    published example ROI's fit has (sigma 0.13 Hz, B 2-5 Hz).
    """
    import csv

    n = len(region_ids)
    in_fov = np.ones(n) if in_fov is None else np.asarray(in_fov, float)
    baseline_peak = (
        np.full(n, np.nan) if baseline_peak is None else np.asarray(baseline_peak, float)
    )
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(LONG_COLUMNS)
        for j in range(len(region_ids)):
            b = int(fits.best[j])
            model = fits.models[b] if b >= 0 else "none"
            for i, d in enumerate(depth_names):
                vals = [
                    subject, comparison, int(region_ids[j]), names[j], hemis[j],
                    int(n_vertices[j]), float(area_mm2[j]), float(in_fov[j]),
                    float(baseline_peak[j]), d, fits.p_depth[j, i],
                    fits.B_bma[j, i], fits.B_bma_sd[j, i], model,
                    fits.p_model[j, max(b, 0)], fits.B_mean[j, i], fits.B_sd[j, i],
                    fits.B_p[j, i], fits.sigma_hz[j], fits.sigma_hz[j] - fits.B_bma[j, i],
                    fits.s_d[j], fits.r2[j, max(b, 0)], int(fits.stalled[j].any()),
                    int(fits.failed[j].any()), fits.nesting[j],
                    int(fits.refit[j].sum()) if fits.refit.size else 0,
                ]  # fmt: skip
                w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in vals])
    return path


__all__ = ["LONG_COLUMNS", "ScopeFits", "fit_columns", "write_long_table"]
