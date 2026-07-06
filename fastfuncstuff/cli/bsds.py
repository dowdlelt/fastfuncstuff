#!/usr/bin/env python3
"""ffs_bsds — Bayesian Switching Dynamical Systems on ROI time series.

Drop in fMRI runs (as ROI time series, or as 4-D volumes plus a parcellation) and
fit a switching factor-analysis brain-state model: latent states, their
moment-to-moment probabilities, a transition matrix, per-state functional
connectivity, occupancy and mean lifetime. For a densely-sampled individual, pass
the runs/sessions as multiple ``-input`` files — they become the model's session
list and share one state repertoire.

Inputs
------
- ``-parcellation none`` (default): each ``-input`` is a 2-D ROI time series
  (``.1D``/``.txt``/``.tsv``/``.csv``/``.npy``); axis is auto-detected (fewer
  ROIs than timepoints) or forced with ``-time_axis``.
- ``-parcellation atlas``: each ``-input`` is a 4-D NIfTI; ``-atlas`` labels are
  averaged into ROI time series.
- ``-parcellation ward|voronoi``: data-driven contiguous parcels from the run
  itself (``-mask`` optional, ``-n_parcels`` sets the count). ``ward`` needs
  scikit-learn.

Outputs (under ``-prefix``)
---------------------------
``*_model.npz`` (all arrays), ``*_summary.json``, ``*_transition.txt``,
``*_state_means.txt``, ``*_fc_state-KK.txt``, and per run ``*_run-RR_states.1D``
(MAP path) and ``*_run-RR_stateprob.1D`` (K x T posterior).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from fastfuncstuff.cli_utils import parse_input_files, parse_prefix, print_cli_header
from fastfuncstuff.dynamics.bsds.model import fit_bsds
from fastfuncstuff.dynamics.graph import state_graph_metrics
from fastfuncstuff.dynamics.parcellate import (
    parcellate_atlas,
    parcellate_voronoi,
    parcellate_ward,
)
from fastfuncstuff.dynamics.preprocess import preprocess_sessions
from fastfuncstuff.dynamics.states import compute_state_stats
from fastfuncstuff.dynamics.switching import compute_switch_stats
from fastfuncstuff.utils import get_device


def _ldim(value: str):
    """argparse type for -max_ldim: the string 'auto' or an int bound."""
    return "auto" if value == "auto" else int(value)


def _load_2d_timeseries(path: str, time_axis: str) -> np.ndarray:
    """Load a 2-D ROI time series and orient it to ``(D, N)`` (ROIs by time)."""
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".npz"):
        npz = np.load(path)
        key = "timeseries" if "timeseries" in npz else npz.files[0]
        arr = npz[key]
    else:
        arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise SystemExit(f"expected a 2-D ROI time series in {path}, got shape {arr.shape}")
    if time_axis == "rows":  # rows are time -> transpose to (D, N)
        arr = arr.T
    elif time_axis == "cols":
        pass
    else:  # auto: the smaller dimension is the ROI axis
        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
    return arr


def _load_volume(path: str) -> np.ndarray:
    from fastfuncstuff.io.afni import load_nifti

    return np.asarray(load_nifti(path).get_fdata(), dtype=np.float32)


def _variance_mask(bold4d: np.ndarray) -> np.ndarray:
    return bold4d.std(axis=-1) > 0


def _load_sessions(args) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Load every input into a ``(D, N)`` ROI time series, applying parcellation."""
    paths = parse_input_files(args.input)
    sessions: list[np.ndarray] = []
    labels: np.ndarray | None = None
    atlas3d = None
    mask3d = None
    if args.parcellation == "atlas":
        if not args.atlas:
            raise SystemExit("-parcellation atlas requires -atlas LABEL_IMAGE")
        atlas_vol = _load_volume(args.atlas)
        atlas3d = (atlas_vol[..., 0] if atlas_vol.ndim == 4 else atlas_vol).astype(np.int64)
    if args.mask:
        mask_vol = _load_volume(args.mask)
        mask3d = (mask_vol[..., 0] if mask_vol.ndim == 4 else mask_vol) > 0

    for path in tqdm(paths, desc="load runs", leave=True, disable=len(paths) < 4):
        if args.parcellation == "none":
            sessions.append(_load_2d_timeseries(path, args.time_axis))
            continue
        bold = _load_volume(path)
        if bold.ndim != 4:
            raise SystemExit(f"{path}: expected 4-D volume for parcellation, got {bold.ndim}-D")
        m = mask3d if mask3d is not None else _variance_mask(bold)
        if args.parcellation == "atlas":
            labels, ts = parcellate_atlas(bold, atlas3d, mask3d=m, aggregate=args.aggregate)
        elif args.parcellation == "ward":
            labels, ts = parcellate_ward(bold, m, args.n_parcels, aggregate=args.aggregate)
        elif args.parcellation == "voronoi":
            labels, ts = parcellate_voronoi(
                bold, m, args.n_parcels, seed=args.seed, aggregate=args.aggregate
            )
        else:
            raise SystemExit(f"unknown parcellation: {args.parcellation}")
        sessions.append(ts)
    return sessions, labels


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ffs_bsds", description="Bayesian Switching Dynamical Systems on ROI time series."
    )
    p.add_argument(
        "-input", nargs="+", required=True, help="Run files (ROI time series or 4-D NIfTI)."
    )
    p.add_argument("-prefix", required=True, help="Output prefix.")
    p.add_argument(
        "-n_states",
        "-n-states",
        type=int,
        default=6,
        help="Number of states fit (this many are always returned; ARD only prunes "
        "each state's latent factor count, not the state count itself — read "
        "occupancy/effective_dim to see which states are actually used).",
    )
    p.add_argument(
        "-max_ldim",
        "-max-ldim",
        type=_ldim,
        default=None,
        help="Max latent factors: an int, or 'auto' (PCA-energy heuristic). Default D-1.",
    )
    p.add_argument("-n_init", "-n-init", type=int, default=10, help="Random restarts.")
    p.add_argument("-n_iter", "-n-iter", type=int, default=100, help="Max VB iterations.")
    p.add_argument(
        "-select",
        action="store_true",
        help="Pick n_states/max_ldim by held-out (leave-runs-out) log-likelihood over a "
        "grid before the final fit. Emits *_selection.png and *_selection.json; the "
        "final model is fit at the winning config. Training free energy is not a "
        "selection criterion — this is the empirical referee.",
    )
    p.add_argument(
        "-n_states_grid",
        "-n-states-grid",
        nargs="+",
        type=int,
        default=None,
        help="Candidate n_states for -select (default: a small grid around -n_states).",
    )
    p.add_argument(
        "-max_ldim_grid",
        "-max-ldim-grid",
        nargs="+",
        type=int,
        default=None,
        help="Candidate max_ldim for -select (default: 3 5 8).",
    )
    p.add_argument(
        "-select_folds",
        "-select-folds",
        type=int,
        default=None,
        help="Leave-runs-out folds for -select (default: one run per fold).",
    )
    p.add_argument(
        "-select_n_init",
        "-select-n-init",
        type=int,
        default=4,
        help="Restarts per fit during -select (kept modest; the final fit uses -n_init).",
    )
    p.add_argument(
        "-tol", type=float, default=1e-4, help="Relative free-energy convergence tolerance."
    )
    p.add_argument(
        "-n_kmeans_replicates",
        "-n-kmeans-replicates",
        type=int,
        default=10,
        help="k-means restarts per session for initialisation (kept by inertia).",
    )
    p.add_argument(
        "-kmeans_pca_dim",
        "-kmeans-pca-dim",
        type=int,
        default=20,
        help="Cluster each session's top-N PCs for init instead of raw ROI space. "
        "0 = the reference 'legacy' init (per-run k-means in raw ROI space, exactly "
        "like the MATLAB initPoteriors 'subject' default) — use it to remove the "
        "init as a confound when comparing to MATLAB. The PCA projection (default) "
        "guards against k-means collapse once ROI count is more than a couple dozen.",
    )
    p.add_argument("-tr", type=float, default=1.0, help="TR in seconds (for lifetimes).")
    p.add_argument("-seed", type=int, default=0, help="Random seed.")
    p.add_argument("-device", default=None, help="cpu / cuda / mps (default: auto).")
    p.add_argument(
        "-parcellation",
        choices=["none", "atlas", "ward", "voronoi"],
        default="none",
        help="How to turn inputs into ROI time series.",
    )
    p.add_argument("-atlas", default=None, help="Label image for -parcellation atlas.")
    p.add_argument("-mask", default=None, help="Brain mask for data-driven parcellation.")
    p.add_argument(
        "-n_parcels", "-n-parcels", type=int, default=100, help="Parcel count (ward/voronoi)."
    )
    p.add_argument(
        "-aggregate", choices=["mean", "pca"], default="mean", help="Parcel aggregation."
    )
    p.add_argument(
        "-time_axis",
        choices=["auto", "rows", "cols"],
        default="auto",
        help="ROI-timeseries orientation for -parcellation none.",
    )
    p.add_argument(
        "-detrend_degree", "-detrend-degree", type=int, default=1, help="Legendre drift degree."
    )
    p.add_argument(
        "-standardize",
        choices=["zscore", "varnorm", "demean", "none"],
        default="zscore",
        help="Per-ROI standardisation.",
    )
    p.add_argument(
        "-plots",
        choices=["none", "qc", "all"],
        default="qc",
        help="qc: one multi-panel QC PNG; all: also per-figure publication files.",
    )
    p.add_argument(
        "-plot_format",
        "-plot-format",
        choices=["pdf", "svg", "png"],
        default="pdf",
        help="Vector format for -plots all publication figures.",
    )
    p.add_argument(
        "-events",
        nargs="+",
        default=None,
        help="BIDS *_events.tsv, one per run (same order/count as -input). Enables "
        "state<->task alignment (correlation + contingency + NMI, and a QC figure).",
    )
    p.add_argument(
        "-event_ignore",
        "-event-ignore",
        nargs="+",
        default=None,
        help="trial_type values to exclude from the task alignment (e.g. fixation rest).",
    )
    p.add_argument(
        "-event_cols",
        "-event-cols",
        nargs=3,
        default=None,
        metavar=("ONSET", "DURATION", "TRIAL_TYPE"),
        help="Custom events.tsv column names (default: onset duration trial_type).",
    )
    p.add_argument(
        "-hrf_delay",
        "-hrf-delay",
        type=float,
        default=5.0,
        help="HRF delay (s) for the condition-label contingency view (default 5).",
    )
    p.add_argument(
        "-label_mode",
        "-label-mode",
        choices=["duration", "persist"],
        default="duration",
        help="Condition-label view: 'duration' (on for each event's duration, "
        "unmarked rest becomes baseline — right for designs with ITI/rest) or "
        "'persist' (a condition stays on until the next event).",
    )
    p.add_argument(
        "-include_rest",
        "-include-rest",
        action="store_true",
        help="Add a synthetic 'rest' condition for unmodelled null periods, so the "
        "task alignment shows whether a state owns rest (only with -label_mode duration).",
    )
    return p


def _make_plots(stem: str, model, stats, args, graph_metrics, switch_stats, align=None) -> None:
    """Render QC, analysis, and (optionally) publication figures; matplotlib is core."""
    import matplotlib

    matplotlib.use("Agg")  # headless: write files, never open a window
    from fastfuncstuff.dynamics import plots

    qc_path = f"{stem}_qc.png"
    plots.qc_report(model, stats, qc_path, tr=args.tr)
    analysis_path = f"{stem}_analysis.png"
    plots.analysis_report(model, graph_metrics, switch_stats, analysis_path, tr=args.tr)
    print(f"  wrote {qc_path}, {analysis_path}")
    if align is not None:
        task_path = f"{stem}_task_alignment.png"
        plots.task_alignment_report(model, align, task_path, tr=args.tr)
        print(f"  wrote {task_path}")
    if args.plots == "all":
        written = plots.save_publication_figures(
            model,
            stats,
            stem,
            fmt=args.plot_format,
            tr=args.tr,
            graph_metrics=graph_metrics,
            switch_stats=switch_stats,
        )
        print(f"  wrote {len(written)} publication figures ({args.plot_format} + png)")


def _run_selection(sessions, args, device, stem: str):
    """Grid-search n_states/max_ldim by held-out LL; return the winning (n_states, ldim)."""
    from fastfuncstuff.dynamics import plots
    from fastfuncstuff.dynamics.model_selection import grid_search_bsds

    n_grid = args.n_states_grid or sorted(
        {max(2, args.n_states - 6), args.n_states, args.n_states + 6}
    )
    default_ldim = args.max_ldim if isinstance(args.max_ldim, int) else 5
    l_grid = args.max_ldim_grid or sorted(
        {max(1, default_ldim - 2), default_ldim, default_ldim + 3}
    )
    if len(sessions) < 2:
        raise SystemExit("-select needs at least 2 runs (leave-runs-out cross-validation).")
    print(f"  selection grid: n_states={n_grid} x max_ldim={l_grid} (held-out LL, leave-runs-out)")
    results = grid_search_bsds(
        sessions,
        n_grid,
        l_grid,
        n_folds=args.select_folds,
        n_init=args.select_n_init,
        device=device,
        show_progress=True,
    )
    best = results[0]
    with open(f"{stem}_selection.json", "w") as fh:
        json.dump(
            [
                {
                    "n_states": r.n_states,
                    "max_ldim": r.max_ldim,
                    "held_out_loglik": r.held_out_loglik,
                    "per_timepoint_loglik": r.per_timepoint_loglik,
                }
                for r in results
            ],
            fh,
            indent=2,
        )
    import matplotlib

    matplotlib.use("Agg")
    plots.plot_selection_surface(results, f"{stem}_selection.png")
    print(
        f"  selection: best n_states={best.n_states}, max_ldim={best.max_ldim} "
        f"(held-out LL/TR={best.per_timepoint_loglik:.4f}); wrote {stem}_selection.png/json"
    )
    return best.n_states, best.max_ldim


def _compute_task_alignment(model, args, device):
    """Parse -events and relate the fit to task conditions; None if no -events."""
    if not args.events:
        return None
    from fastfuncstuff.design.bids_events import parse_bids_events
    from fastfuncstuff.dynamics.task import align_states_to_task

    n_runs = len(model.responsibilities)
    if len(args.events) != n_runs:
        raise SystemExit(
            f"-events has {len(args.events)} file(s) but the model has {n_runs} run(s); "
            "pass one events.tsv per -input run, in the same order."
        )
    all_onsets, durations, condition_labels = parse_bids_events(
        args.events,
        event_ignore=args.event_ignore,
        event_cols=tuple(args.event_cols) if args.event_cols else None,
    )
    align = align_states_to_task(
        model,
        all_onsets,
        durations,
        condition_labels,
        tr=args.tr,
        hrf_delay=args.hrf_delay,
        respect_duration=(args.label_mode == "duration"),
        include_rest=args.include_rest,
        device=device,
    )
    print(
        f"  task alignment: {len(condition_labels)} conditions, "
        f"NMI={align.normalized_mutual_info:.3f}, mean state purity={align.state_purity.mean():.2f}"
    )
    return align


def _save_task_alignment(stem: str, align) -> None:
    """Write the state<->task alignment matrices and a JSON summary."""
    np.savetxt(f"{stem}_task_correlation.txt", align.correlation, fmt="%.5f")
    np.savetxt(f"{stem}_task_contingency.txt", align.contingency, fmt="%.5f")
    summary = {
        "condition_labels": align.condition_labels,
        "normalized_mutual_info": align.normalized_mutual_info,
        "state_purity": align.state_purity.tolist(),
        "dominant_condition_idx": align.dominant_condition.tolist(),
        "dominant_condition": [
            align.condition_labels[c] if c >= 0 else "baseline"
            for c in align.dominant_condition.tolist()
        ],
    }
    with open(f"{stem}_task_alignment.json", "w") as fh:
        json.dump(summary, fh, indent=2)


def _save_outputs(
    stem: str, model, stats, args, labels, elapsed: float, directed=None, graph=None, switch=None
) -> None:
    base = Path(stem)
    base.parent.mkdir(parents=True, exist_ok=True)
    k = model.n_states

    np.savez(
        f"{stem}_model.npz",
        state_means=model.state_means.numpy(),
        state_covs=model.state_covs.numpy(),
        state_fc=stats.state_fc.numpy(),
        transition=model.transition.numpy(),
        init_probs=model.init_probs.numpy(),
        loadings=model.loadings.numpy(),
        psii=model.psii.numpy(),
        ar_transitions=model.ar_transitions.numpy(),
        ar_noise_cov=model.ar_noise_cov.numpy(),
        directed_connectivity=(np.array([]) if directed is None else directed.cpu().numpy()),
        effective_dim=model.effective_dim.numpy(),
        ard_precision=model.ard_precision.numpy(),
        session_lengths=np.array(model.session_lengths),
        group_occupancy=stats.group_occupancy,
        group_lifetime=stats.group_lifetime,
        subject_occupancy=stats.subject_occupancy,
        subject_lifetime=stats.subject_lifetime,
        graph_strength=np.array([]) if graph is None else graph.strength,
        graph_clustering=np.array([]) if graph is None else graph.clustering,
        graph_betweenness=np.array([]) if graph is None else graph.betweenness,
        graph_nodal_efficiency=np.array([]) if graph is None else graph.nodal_efficiency,
        graph_global_efficiency=np.array([]) if graph is None else graph.global_efficiency,
        switch_rate_group=np.array([]) if switch is None else np.array(switch.group_switch_rate),
        switch_rate_subject=np.array([]) if switch is None else switch.subject_switch_rate,
        parcel_labels=np.array([]) if labels is None else np.asarray(labels),
    )
    if switch is not None:
        with open(f"{stem}_switch_paths.txt", "w") as fh:
            fh.write("# count\tpath\n")
            for path, count in switch.top_paths:
                fh.write(f"{count}\t{'->'.join(f'S{s}' for s in path)}\n")
    np.savetxt(f"{stem}_transition.txt", model.transition.numpy(), fmt="%.6f")
    np.savetxt(f"{stem}_state_means.txt", model.state_means.numpy(), fmt="%.6f")
    for s in range(k):
        np.savetxt(f"{stem}_fc_state-{s:02d}.txt", stats.state_fc[s].numpy(), fmt="%.6f")
        if directed is not None:
            np.savetxt(f"{stem}_directed_state-{s:02d}.txt", directed[s].cpu().numpy(), fmt="%.6f")
    for r, (path_prob, path_map) in enumerate(
        zip(model.responsibilities, model.viterbi_states, strict=True)
    ):
        np.savetxt(f"{stem}_run-{r:02d}_stateprob.1D", path_prob.numpy(), fmt="%.5f")
        np.savetxt(f"{stem}_run-{r:02d}_states.1D", path_map.numpy(), fmt="%d")

    summary = {
        "n_states": k,
        "ldim": model.ldim,
        "n_sessions": len(model.session_lengths),
        "session_lengths": model.session_lengths,
        "converged": bool(model.converged),
        "n_iter": len(model.objective_history),
        "final_free_energy": model.objective_history[-1] if model.objective_history else None,
        "effective_dim": model.effective_dim.tolist(),
        "tr": args.tr,
        "group_occupancy": stats.group_occupancy.tolist(),
        "group_lifetime_sec": stats.group_lifetime.tolist(),
        "elapsed_sec": round(elapsed, 2),
        "parcellation": args.parcellation,
    }
    if graph is not None:
        summary["global_efficiency"] = graph.global_efficiency.tolist()
        summary["mean_clustering"] = graph.mean_clustering.tolist()
    if switch is not None:
        summary["group_switch_rate"] = switch.group_switch_rate
        summary["switch_rate_per_minute"] = switch.switch_rate_per_minute
        summary["top_switch_paths"] = [
            ["->".join(f"S{s}" for s in path), count] for path, count in switch.top_paths
        ]
    with open(f"{stem}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print_cli_header("ffs_bsds", "Bayesian Switching Dynamical Systems")
    device = get_device(args.device)

    sessions_np, labels = _load_sessions(args)
    print(
        f"  loaded {len(sessions_np)} session(s), "
        f"D={sessions_np[0].shape[0]} ROIs, "
        f"N={[s.shape[1] for s in sessions_np]} timepoints"
    )
    sessions = preprocess_sessions(
        sessions_np,
        detrend_degree=args.detrend_degree,
        standardize=None if args.standardize == "none" else args.standardize,
        device=device,
    )

    pfx = parse_prefix(str(args.prefix))
    Path(pfx.stem).parent.mkdir(parents=True, exist_ok=True)

    n_states, max_ldim = args.n_states, args.max_ldim
    if args.select:
        n_states, max_ldim = _run_selection(sessions, args, device, pfx.stem)

    t0 = time.time()
    model = fit_bsds(
        sessions,
        n_states=n_states,
        max_ldim=max_ldim,
        n_iter=args.n_iter,
        n_init=args.n_init,
        tol=args.tol,
        seed=args.seed,
        device=device,
        show_progress=True,
        n_kmeans_replicates=args.n_kmeans_replicates,
        kmeans_pca_dim=args.kmeans_pca_dim or None,
    )
    elapsed = time.time() - t0
    stats = compute_state_stats(model, tr=args.tr)

    from fastfuncstuff.dynamics.connectivity import per_state_directed_connectivity

    directed, _ = per_state_directed_connectivity(model, sessions)
    graph = state_graph_metrics(stats.state_fc)
    switch = compute_switch_stats(model, tr=args.tr)
    align = _compute_task_alignment(model, args, device)

    _save_outputs(
        pfx.stem,
        model,
        stats,
        args,
        labels,
        elapsed,
        directed=directed,
        graph=graph,
        switch=switch,
    )
    if align is not None:
        _save_task_alignment(pfx.stem, align)
    if args.plots != "none":
        _make_plots(pfx.stem, model, stats, args, graph, switch, align=align)
    print(
        f"  done in {elapsed:.1f}s — {model.n_states} states, "
        f"converged={model.converged}, occupancy={np.round(stats.group_occupancy, 3).tolist()}"
    )
    print(f"  wrote {pfx.stem}_model.npz and companion files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
