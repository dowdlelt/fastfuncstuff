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
from fastfuncstuff.dynamics.parcellate import (
    parcellate_atlas,
    parcellate_voronoi,
    parcellate_ward,
)
from fastfuncstuff.dynamics.preprocess import preprocess_sessions
from fastfuncstuff.dynamics.states import compute_state_stats
from fastfuncstuff.utils import get_device


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
        "-n_states", "-n-states", type=int, default=6, help="Max number of states (ARD prunes)."
    )
    p.add_argument(
        "-max_ldim", "-max-ldim", type=int, default=None, help="Max latent factors (default D-1)."
    )
    p.add_argument("-n_init", "-n-init", type=int, default=10, help="Random restarts.")
    p.add_argument("-n_iter", "-n-iter", type=int, default=100, help="Max VB iterations.")
    p.add_argument(
        "-tol", type=float, default=1e-4, help="Relative free-energy convergence tolerance."
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
    return p


def _make_plots(stem: str, model, stats, args) -> None:
    """Render QC and (optionally) publication figures; matplotlib is a core dep."""
    import matplotlib

    matplotlib.use("Agg")  # headless: write files, never open a window
    from fastfuncstuff.dynamics import plots

    qc_path = f"{stem}_qc.png"
    plots.qc_report(model, stats, qc_path, tr=args.tr)
    print(f"  wrote {qc_path}")
    if args.plots == "all":
        written = plots.save_publication_figures(
            model, stats, stem, fmt=args.plot_format, tr=args.tr
        )
        print(f"  wrote {len(written)} publication figures ({args.plot_format} + png)")


def _save_outputs(stem: str, model, stats, args, labels, elapsed: float) -> None:
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
        effective_dim=model.effective_dim.numpy(),
        session_lengths=np.array(model.session_lengths),
        group_occupancy=stats.group_occupancy,
        group_lifetime=stats.group_lifetime,
        subject_occupancy=stats.subject_occupancy,
        subject_lifetime=stats.subject_lifetime,
        parcel_labels=np.array([]) if labels is None else np.asarray(labels),
    )
    np.savetxt(f"{stem}_transition.txt", model.transition.numpy(), fmt="%.6f")
    np.savetxt(f"{stem}_state_means.txt", model.state_means.numpy(), fmt="%.6f")
    for s in range(k):
        np.savetxt(f"{stem}_fc_state-{s:02d}.txt", stats.state_fc[s].numpy(), fmt="%.6f")
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

    t0 = time.time()
    model = fit_bsds(
        sessions,
        n_states=args.n_states,
        max_ldim=args.max_ldim,
        n_iter=args.n_iter,
        n_init=args.n_init,
        tol=args.tol,
        seed=args.seed,
        device=device,
        show_progress=True,
    )
    elapsed = time.time() - t0
    stats = compute_state_stats(model, tr=args.tr)

    pfx = parse_prefix(str(args.prefix))
    _save_outputs(pfx.stem, model, stats, args, labels, elapsed)
    if args.plots != "none":
        _make_plots(pfx.stem, model, stats, args)
    print(
        f"  done in {elapsed:.1f}s — {model.n_states} states, "
        f"converged={model.converged}, occupancy={np.round(stats.group_occupancy, 3).tolist()}"
    )
    print(f"  wrote {pfx.stem}_model.npz and companion files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
