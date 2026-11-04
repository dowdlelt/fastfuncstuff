"""One ROI, two conditions, the model space at several vascular resolutions.

The pipeline of ``notebooks/laminar_dcm_predictive_tones.ipynb`` (Faes et al.'s
analysis, [[Faes 2026]]) as a function: invert every member of the layer-hypothesis
model space at each K, average each model's posteriors across K (Bayesian parameter
averaging), compare the averaged models by free energy, and read off what an analysis
reports -- model and depth-inclusion probabilities, the modulation ``B`` per neuronal
depth with credible intervals, the deveined neuronal time courses and the static
deveining operator.

Two choices differ from the notebook, both on purpose:

* the deveining operator is computed **with** the depth PSF, because it is applied to
  measured depth profiles, which carry it (the notebook's maps the model's vascular
  depths, before the PSF);
* the model space can be a subset (``targets``), for a hypothesis-driven study or a
  quick look.

Nothing here reads or writes files except :func:`write_results`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.laminar.experiment import (
    DEPTH_NAMES,
    depth_inclusion_probabilities,
    fit_model_space,
    layer_model_names,
    layer_model_targets,
    posterior_model_probabilities,
)
from fastfuncstuff.laminar.io import LaminarData
from fastfuncstuff.laminar.params import P0

#: The baselines the reference driver sets on top of ``LBR_param_priors.m``'s defaults
#: (``make_modelspace_oracle.m``; the notebook uses the same).
REFERENCE_P0 = {"V0t": 3.0, "nr": 3.0, "al_v": 0.35, "w_v": 0.5}


def reference_p0() -> P0:
    return P0(**REFERENCE_P0)


def depth_names(n: int) -> list[str]:
    return list(DEPTH_NAMES) if n == 3 else [f"depth{i}" for i in range(n)]


@dataclass
class RoiFit:
    """What one ROI's analysis reports."""

    roi: str
    name: str
    hemi: str
    k_values: list[int]
    models: list[str]
    targets: list[tuple[int, ...]]
    F: np.ndarray  # (n_models,) of the averaged models
    per_k_F: np.ndarray  # noqa: N815 (n_models, n_K); the name BPAResult carries
    p_model: np.ndarray  # (n_models,)
    p_depth: np.ndarray  # (N,) P(depth modulated), marginalised over the models
    best: int
    B_mean: np.ndarray  # (N,) the best model's averaged posterior
    B_sd: np.ndarray
    B_p: np.ndarray  # P(B differs from 0)
    r2: np.ndarray  # (n_models,) variance explained on the fitted rows, first K
    neuronal: np.ndarray  # (n_scans, N) deveined, best model, first K
    W: np.ndarray  # (K0, N) forward mixing, model order (superficial first)
    W_inv: np.ndarray  # (N, K0) deveining operator
    meta: dict = field(default_factory=dict)
    # What the figures need (all for the first K and the best model, model depth order,
    # superficial first):
    y_obs: np.ndarray | None = None  # (n_scans, K0) the padded input, percent change
    y_hat: np.ndarray | None = None  # (n_scans, K0) the fit
    rows: np.ndarray | None = None  # (n_scans,) bool: rows that were fitted
    impulse: np.ndarray | None = None  # (K0, N, n_t) neuronal -> BOLD transfer
    drainage_shift: np.ndarray | None = None  # (K0,) depths deeper than the mapping predicts
    kernels: dict | None = None  # {K: depth PSF (K,K) or (L,) or None}
    stalled: np.ndarray | None = None  # (n_models, n_K) "converged" on rejected steps
    warnings: list = field(default_factory=list)


def fit_roi(
    data_for_k: dict[int, LaminarData],
    kernel_for_k: dict[int, torch.Tensor | None] | None = None,
    *,
    targets: list[tuple[int, ...]] | None = None,
    names: list[str] | None = None,
    free_energy: str = "reference",
    max_iter: int = 128,
    hyper_from_rows: bool = False,
    rescue: int = 0,
    roi: str = "",
    name: str = "",
    hemi: str = "",
) -> RoiFit:
    """Invert, average across K, compare and devein one ROI's two-condition data."""
    from fastfuncstuff.laminar.devein import (
        deveined_timecourses,
        laminar_impulse_response,
        static_deveining_matrix,
    )

    ks = sorted(data_for_k)
    first = data_for_k[ks[0]]
    for k in ks[1:]:
        if not torch.equal(data_for_k[k].rows, first.rows):
            raise ValueError("every K must share the design (the same rows and time points)")
    n = first.spec.N
    if targets is None:
        targets = layer_model_targets(n)
    if names is None:
        all_names = layer_model_names(n)
        all_targets = layer_model_targets(n)
        names = [all_names[all_targets.index(tuple(t))] for t in targets]
    kernels = kernel_for_k or {k: None for k in ks}
    ikw = dict(first.integrate_kwargs)

    names, averaged, per_k = fit_model_space(
        {k: data_for_k[k].spec for k in ks},
        {k: data_for_k[k].y for k in ks},
        {k: data_for_k[k].u for k in ks},
        kernel_for_k={k: kernels.get(k) for k in ks} if any(v is not None for v in kernels.values()) else None,
        rows=first.rows,
        targets=list(targets),
        names=list(names),
        free_energy=free_energy,
        progress=False,
        max_iter=max_iter,
        hyper_from_rows=hyper_from_rows,
        rescue=rescue,
        **ikw,
    )  # fmt: skip
    F = np.array([float(a.F) for a in averaged])
    p_model = posterior_model_probabilities(F).numpy()
    p_depth = depth_inclusion_probabilities(F, list(targets), n).numpy()
    best = int(np.argmax(F))
    bpa = averaged[best]
    b_mean = torch.diagonal(bpa.Ep["B"][0]).numpy().copy()
    b_sd = torch.diagonal(bpa.Vp["B"][0]).clamp(min=0).sqrt().numpy().copy()
    b_p = torch.diagonal(bpa.Pp["B"][0]).numpy().copy()

    rows = first.rows.numpy()
    y0 = first.y.numpy()[rows]
    ss_tot = float(((y0 - y0.mean()) ** 2).sum())
    r2 = np.array(
        [1.0 - float(((y0 - runs[0].y_pred.numpy()[rows]) ** 2).sum()) / max(ss_tot, 1e-30)
         for runs in per_k]
    )  # fmt: skip

    spec0, res0, k0 = first.spec, per_k[best][0], kernels.get(ks[0])
    dev = deveined_timecourses(spec0, res0.Ep, first.u, int(first.y.shape[0]), kernel=k0, **ikw)
    ir = laminar_impulse_response(spec0, res0.Ep, n_scans=20, amplitude=1.0, kernel=k0, **ikw)
    W, W_inv = static_deveining_matrix(ir, mode="auc")
    from fastfuncstuff.laminar.devein import deveining_fidelity

    fid = deveining_fidelity(ir, spec0, res0.Ep, mode="auc")
    stalled = np.array([[bool(r.stalled) for r in runs] for runs in per_k])
    warnings = _warnings(names, list(targets), F, stalled, ks, rescue)
    return RoiFit(
        roi=roi, name=name, hemi=hemi, k_values=ks, models=list(names), targets=list(targets),
        F=F, per_k_F=np.array([[float(f) for f in a.per_k_F] for a in averaged]),
        p_model=p_model, p_depth=p_depth, best=best, B_mean=b_mean, B_sd=b_sd, B_p=b_p,
        r2=r2, neuronal=dev.neuronal.numpy(), W=W.numpy(), W_inv=W_inv.numpy(),
        meta={"TR": first.meta.get("TR"), "block": first.meta.get("block"),
              "gap": first.meta.get("gap"), "converged": [bool(r[0].converged) for r in per_k]},
        y_obs=first.y.numpy().copy(), y_hat=res0.y_pred.numpy().copy(), rows=rows.copy(),
        impulse=ir.numpy().copy(), drainage_shift=fid["drainage_shift"].numpy().copy(),
        kernels={k: (None if (kk := kernels.get(k)) is None else kk.numpy().copy()) for k in ks},
        stalled=stalled, warnings=warnings,
    )  # fmt: skip


def _warnings(names, targets, F, stalled, ks, rescue, tol: float = 10.0) -> list[str]:
    """What to distrust in a fit: runs that "converged" on rejected steps, and nested
    models that scored far worse than the model they contain (they cannot, so the
    optimiser stalled -- see ``VLResult.stalled``)."""
    out = []
    for m, row in enumerate(stalled):
        if row.any():
            where = ", ".join(f"K={ks[j]}" for j in np.flatnonzero(row))
            out.append(f"model '{names[m]}' stalled at {where} (ended on rejected steps)")
    for i, ti in enumerate(targets):
        for j, tj in enumerate(targets):
            if i != j and all(a >= b for a, b in zip(ti, tj, strict=True)) and F[i] < F[j] - tol:
                out.append(
                    f"'{names[i]}' (F {F[i]:.1f}) scores {F[j] - F[i]:.0f} nats below the model "
                    f"it contains, '{names[j]}' (F {F[j]:.1f}): a stalled fit, not evidence"
                )
    if out and not rescue:
        out.append("rerun with -rescue 2 to restart stalled fits with the damping reset")
    return out


def _log_bf(fit: RoiFit) -> float:
    """The best model's evidence over the model space's own ``null``, in nats."""
    if "null" not in fit.models:
        return float("nan")
    return float(np.max(fit.F) - fit.F[fit.models.index("null")])


def null_statistics(fit: RoiFit, nulls: list[RoiFit]) -> dict:
    """Permutation p-values for one ROI: ``{statistic: (real, p, null values)}``.

    ``p = (1 + #{null >= real}) / (1 + n)``, the standard permutation p-value, which
    floors at ``1/(1+n)`` -- one replicate cannot license a rate finer than itself, and
    19 are the fewest that reach 0.05.  The statistics are the ones a reader thresholds:
    the evidence over the null MODEL, each depth's P(modulated), and the size of each
    depth's modulation (``|B|``, unsigned, because a permutation null is symmetric in
    sign and a two-sided question deserves a two-sided statistic).
    """
    out: dict[str, tuple[float, float, list[float]]] = {}

    def add(name, real, draws):
        vals = [v for v in draws if np.isfinite(v)]
        p = (1 + sum(1 for v in vals if v >= real)) / (1 + len(vals)) if vals else float("nan")
        out[name] = (float(real), float(p), [float(v) for v in vals])

    add("logBF_vs_null", _log_bf(fit), [_log_bf(f) for f in nulls])
    for i, d in enumerate(depth_names(len(fit.B_mean))):
        add(f"p_modulated_{d}", fit.p_depth[i], [f.p_depth[i] for f in nulls])
        add(f"abs_B_{d}", abs(fit.B_mean[i]), [abs(f.B_mean[i]) for f in nulls])
    return out


def write_null(prefix: str, fits: list[RoiFit], nulls_by_roi: dict) -> list[str]:
    """``{prefix}_laminar_null.tsv`` (one line per ROI x statistic) and ``_null_draws.tsv``.

    The draws are written out because the summary is three quantiles of them and a reader
    may want a different tail, a different statistic, or to pool ROIs -- and because a
    permutation distribution is the evidence for the p-value beside it.
    """
    paths = [f"{prefix}_laminar_null.tsv", f"{prefix}_laminar_null_draws.tsv"]
    Path(paths[0]).parent.mkdir(parents=True, exist_ok=True)
    with open(paths[0], "w") as fh:
        fh.write("roi\tname\themi\tstatistic\treal\tp_perm\tn_perm\tnull_median"
                 "\tnull_p95\tnull_max\n")  # fmt: skip
        for f in fits:
            stats = null_statistics(f, nulls_by_roi.get((f.roi, f.hemi), []))
            for name, (real, p, draws) in stats.items():
                q = (np.median(draws), np.percentile(draws, 95), np.max(draws)) if draws else (
                    float("nan"), float("nan"), float("nan"))  # fmt: skip
                fh.write(f"{f.roi}\t{f.name}\t{f.hemi}\t{name}\t{real:.6g}\t{p:.6g}\t"
                         f"{len(draws)}\t{q[0]:.6g}\t{q[1]:.6g}\t{q[2]:.6g}\n")  # fmt: skip
    with open(paths[1], "w") as fh:
        fh.write("roi\tname\themi\tpermutation\tstatistic\tvalue\n")
        for f in fits:
            stats = null_statistics(f, nulls_by_roi.get((f.roi, f.hemi), []))
            for name, (_real, _p, draws) in stats.items():
                for j, v in enumerate(draws):
                    fh.write(f"{f.roi}\t{f.name}\t{f.hemi}\t{j}\t{name}\t{v:.6g}\n")
    return paths


def write_results(prefix: str, fits: list[RoiFit], provenance: dict | None = None) -> list[str]:
    """Tidy tables, one line per ROI x (model | neuronal depth | measured depth | time):

    * ``{prefix}_laminar_models.tsv`` -- F of each averaged model, its probability, F at
      every K, variance explained;
    * ``{prefix}_laminar_depths.tsv`` -- per neuronal depth: P(modulated) over the whole
      model space (prefer it to the winner, see ``depth_inclusion_probabilities``), and
      the best model's B with a 90% credible interval;
    * ``{prefix}_laminar_devein.tsv`` -- the static deveining operator, neuronal depth x
      measured depth, **depth 1 deepest** like every other table here: apply it to
      measured amplitudes (betas, FIR peaks, areas) of matching units;
    * ``{prefix}_laminar_neuronal.tsv`` -- the deveined neuronal time course per
      condition and neuronal depth, from event onset;
    * ``{prefix}_laminar.json`` -- how it was run.
    """
    paths = [f"{prefix}_laminar_{s}.tsv" for s in ("models", "depths", "devein", "neuronal")]
    Path(paths[0]).parent.mkdir(parents=True, exist_ok=True)
    z90 = 1.6448536269514722
    with open(paths[0], "w") as fh:
        ks = fits[0].k_values if fits else []
        fh.write(
            "roi\tname\themi\tmodel\tF\tp_model\tbest\tr2\t"
            + "\t".join(f"F_K{k}" for k in ks)
            + "\n"
        )
        for f in fits:
            for m, model in enumerate(f.models):
                fh.write(
                    f"{f.roi}\t{f.name}\t{f.hemi}\t{model}\t{f.F[m]:.6g}\t{f.p_model[m]:.6g}\t"
                    f"{int(m == f.best)}\t{f.r2[m]:.6g}\t"
                    + "\t".join(f"{v:.6g}" for v in f.per_k_F[m]) + "\n"
                )  # fmt: skip
    with open(paths[1], "w") as fh:
        fh.write(
            "roi\tname\themi\tneuronal_depth\tp_modulated\tbest_model\tB\tB_sd\tB_lo90\tB_hi90\tP_B_nonzero\n"
        )
        for f in fits:
            for i, d in enumerate(depth_names(len(f.B_mean))):
                lo, hi = f.B_mean[i] - z90 * f.B_sd[i], f.B_mean[i] + z90 * f.B_sd[i]
                fh.write(
                    f"{f.roi}\t{f.name}\t{f.hemi}\t{d}\t{f.p_depth[i]:.6g}\t{f.models[f.best]}\t"
                    f"{f.B_mean[i]:.6g}\t{f.B_sd[i]:.6g}\t{lo:.6g}\t{hi:.6g}\t{f.B_p[i]:.6g}\n"
                )  # fmt: skip
    with open(paths[2], "w") as fh:
        fh.write("roi\tname\themi\tneuronal_depth\tdepth\tweight\n")
        for f in fits:
            k_n = f.W_inv.shape[1]
            for i, d in enumerate(depth_names(f.W_inv.shape[0])):
                for k in range(k_n):  # model column k (superficial first) is our depth K - k
                    fh.write(f"{f.roi}\t{f.name}\t{f.hemi}\t{d}\t{k_n - k}\t{f.W_inv[i, k]:.6g}\n")
    with open(paths[3], "w") as fh:
        fh.write("roi\tname\themi\tcondition\ttime\tneuronal_depth\tactivity\n")
        for f in fits:
            tr, block, gap = f.meta["TR"], f.meta["block"], f.meta["gap"]
            starts = {"1": 1, "2": block + gap + 1}
            n_t = block - 1
            for cond, s0 in starts.items():
                for t in range(n_t):
                    for i, d in enumerate(depth_names(f.neuronal.shape[1])):
                        fh.write(
                            f"{f.roi}\t{f.name}\t{f.hemi}\t{cond}\t{t * tr:.6g}\t{d}\t"
                            f"{f.neuronal[s0 + t, i]:.6g}\n"
                        )
    doc = {
        "tool": "ffs_laminar",
        "rois": [
            {
                "roi": f.roi,
                "name": f.name,
                "hemi": f.hemi,
                "best": f.models[f.best],
                "converged": f.meta.get("converged"),
            }
            for f in fits
        ],  # fmt: skip
        **(provenance or {}),
    }
    with open(f"{prefix}_laminar.json", "w") as fh:
        json.dump(doc, fh, indent=2, default=str)
    return [*paths, f"{prefix}_laminar.json"]


__all__ = ["REFERENCE_P0", "RoiFit", "depth_names", "fit_roi", "reference_p0", "write_results"]
