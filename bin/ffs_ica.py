#!/usr/bin/env python3

"""
ffs_ica.py - Fast run-wise whole-brain ICA sanity-check / demo CLI.

Core goals
----------
- Simple whole-brain ICA workflow with GPU acceleration when available.
- Automatic component estimation, including MELODIC-style Bayesian evidence proxy.
- Optional ICASSO stability analysis at the selected component count.
- Practical preprocessing knobs for fMRI runs:
  - optional spatial blur
  - optional percent-signal scaling
  - optional polynomial detrending
  - optional Fourier high-pass filtering
- Optional task metadata attachment via condition correlations
  (onsets/durations are used for interpretation only, not model fitting).

Notes
-----
- Multiple input runs are processed independently by default.
- Flags for future modes (`-temp_concat`, `-tensor`) are present as placeholders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# fastfuncsim imports
try:
    from fastfuncsim.afni_io import get_tr_from_file, load_afni_mask, load_nifti
    from fastfuncsim.cli_utils import auto_polort, parse_input_files, print_cli_header
    from fastfuncsim.ica import FastICA
    from fastfuncsim.ica_tools import (
        apply_high_pass_fft,
        apply_polort_projection,
        build_task_design_for_run,
        component_condition_correlations,
        estimate_ica_component_count,
        mixture_zscores_signed,
        parse_num_comps_spec,
    )
    from fastfuncsim.icasso import icasso
    from fastfuncsim.utils import (
        gaussian_blur_3d,
        get_device,
        scale_to_percent_signal,
        to_tensor,
    )
except ImportError as e:
    print(f"ERROR: Could not import fastfuncsim: {e}")
    print("Make sure fastfuncsim is installed: pip install -e .")
    sys.exit(1)


def _save_scree_plot(evr: np.ndarray, out_png: Path, title: str):
    if plt is None:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(evr) + 1)
    cum = np.cumsum(evr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(x, evr, lw=1.5)
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    ax2.plot(x, np.clip(cum, 0.0, 1.0), lw=1.5)
    ax2.set_xlabel("Component index")
    ax2.set_ylabel("Cumulative variance")
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_components_4d(
    components_kv: np.ndarray,
    mask3d: np.ndarray,
    shape3d: tuple[int, int, int],
    affine: np.ndarray,
    out_file: Path,
):
    k, n_vox = components_kv.shape
    out = np.zeros((*shape3d, k), dtype=np.float32)
    if mask3d is None:
        if np.prod(shape3d) != n_vox:
            raise ValueError("Component size does not match full volume size")
        for i in range(k):
            out[..., i] = components_kv[i].reshape(shape3d)
    else:
        flat_mask = mask3d.reshape(-1)
        for i in range(k):
            vol = np.zeros(flat_mask.shape[0], dtype=np.float32)
            vol[flat_mask] = components_kv[i]
            out[..., i] = vol.reshape(shape3d)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, affine), str(out_file))


def _run_single_ica(
    run_file: str,
    run_idx: int,
    args,
    device: torch.device,
    shared_mask: np.ndarray | None,
    onsets_files: list[str] | None,
    durations: list[str] | None,
) -> dict:
    img = load_nifti(run_file)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    shape3d = data.shape[:3]
    n_t = data.shape[3]

    tr = float(args.tr) if args.tr is not None else float(get_tr_from_file(run_file))

    if shared_mask is not None and shared_mask.shape != shape3d:
        raise ValueError(
            f"Mask shape {shared_mask.shape} does not match run shape {shape3d} for {run_file}"
        )

    if args.do_blur is not None and args.do_blur > 0:
        voxel_sizes = tuple(float(v) for v in img.header.get_zooms()[:3])
        data = gaussian_blur_3d(
            data=data,
            fwhm_mm=float(args.do_blur),
            voxel_sizes=voxel_sizes,
            device=device,
            verbose=args.verbose,
        )

    if shared_mask is not None:
        data_vox_t_np = data[shared_mask].astype(np.float32)
        mask3d = shared_mask
    else:
        data_vox_t_np = data.reshape(-1, n_t).astype(np.float32)
        mask3d = None

    data_vox_t = to_tensor(data_vox_t_np, device=device)

    if args.do_scale:
        data_vox_t, _, _ = scale_to_percent_signal(data_vox_t, run_starts=[0], verbose=args.verbose)

    if args.polort is None:
        run_duration_sec = tr * n_t
        polort = int(auto_polort(run_duration_sec, formula="afni"))
    else:
        polort = int(args.polort)

    data_vox_t = apply_polort_projection(data_vox_t, polort=polort, device=device)
    data_vox_t = apply_high_pass_fft(data_vox_t, tr=tr, high_pass_hz=args.high_pass)

    num_spec = parse_num_comps_spec(args.num_comps)
    n_components, pca_diag, num_diag = estimate_ica_component_count(
        data_vox_t=data_vox_t,
        method=num_spec,
        max_auto_components=args.max_auto_components,
        auto_min_components=args.auto_min_components,
        auto_var_threshold=args.auto_var_threshold,
        use_mp_prior=not args.auto_no_mp,
        device=device,
    )

    x_t = data_vox_t.T  # (time, vox)

    if args.icasso:
        icasso_res = icasso(
            X=x_t,
            n_components=n_components,
            n_runs=args.icasso_runs,
            pca_components=n_components,
            min_stability=args.icasso_min_stability,
            device=device,
            verbose=args.verbose,
            batch_size=args.icasso_batch_size,
        )
        components = torch.as_tensor(icasso_res["all_centroids"], device=device, dtype=torch.float32)
        mixing = torch.as_tensor(icasso_res["all_mixing"], device=device, dtype=torch.float32)
        stability = np.asarray(icasso_res["all_stability"], dtype=np.float32)
        icasso_meta = {
            "enabled": True,
            "icasso_runs": int(args.icasso_runs),
            "min_stability": float(args.icasso_min_stability),
            "n_stable": int(icasso_res["n_stable"]),
            "stability": stability.tolist(),
        }
    else:
        ica = FastICA(
            n_components=n_components,
            pca_components=n_components,
            max_iter=args.ica_max_iter,
            tol=args.ica_tol,
            random_state=run_idx,
            device=device,
        )
        ica.fit(x_t)
        components = ica.components_.to(device)
        mixing = ica.mixing_.to(device)
        stability = None
        icasso_meta = {"enabled": False}

    mix_var = torch.var(mixing, dim=0, unbiased=False)
    mix_var = torch.clamp(mix_var, min=1e-12)
    sort_idx = torch.argsort(mix_var, descending=True)

    components = components[sort_idx, :]
    mixing = mixing[:, sort_idx]
    mix_var = mix_var[sort_idx]
    var_share = (mix_var / mix_var.sum()).detach().cpu().numpy().astype(np.float32)

    if stability is not None:
        stability = stability[sort_idx.detach().cpu().numpy()]

    if args.var_norm:
        mixing = mixing - mixing.mean(dim=0, keepdim=True)
        mixing_std = torch.clamp(mixing.std(dim=0, keepdim=True), min=1e-8)
        mixing = mixing / mixing_std

    condition_corr = None
    cond_labels = None
    cond_durations = None
    if onsets_files is not None and durations is not None:
        try:
            design_tc, cond_labels, cond_durations = build_task_design_for_run(
                onsets_files=onsets_files,
                durations_arg=durations,
                run_idx=run_idx,
                n_timepoints=n_t,
                tr=tr,
                microtime_dt=args.microtime_dt,
                device=device,
            )
            condition_corr = component_condition_correlations(mixing_tk=mixing, design_tc=design_tc)
        except Exception as e:
            print(f"  Warning: Could not compute condition correlations for run {run_idx + 1}: {e}")

    out_prefix = Path(args.prefix)
    run_tag = f"run{run_idx + 1:02d}"

    comp_np = components.detach().cpu().numpy().astype(np.float32)
    mixing_np = mixing.detach().cpu().numpy().astype(np.float32)

    _save_components_4d(
        components_kv=comp_np,
        mask3d=mask3d,
        shape3d=shape3d,
        affine=affine,
        out_file=Path(f"{out_prefix}_{run_tag}_ica_maps.nii.gz"),
    )

    np.savetxt(
        f"{out_prefix}_{run_tag}_ica_timecourses.1D",
        mixing_np,
        fmt="%.6f",
        delimiter="\t",
    )

    _save_scree_plot(
        evr=np.asarray(pca_diag["scree_ratio"], dtype=np.float64),
        out_png=Path(f"{out_prefix}_{run_tag}_pca_scree.png"),
        title=f"Run {run_idx + 1}: PCA scree",
    )

    mixture_meta = []
    if args.save_mixture_z:
        z_maps = np.zeros_like(comp_np, dtype=np.float32)
        p_maps = np.zeros_like(comp_np, dtype=np.float32)
        for k in range(comp_np.shape[0]):
            z_signed, p_sig, m = mixture_zscores_signed(comp_np[k])
            z_maps[k] = z_signed
            p_maps[k] = p_sig
            mixture_meta.append(m)

        _save_components_4d(
            components_kv=z_maps,
            mask3d=mask3d,
            shape3d=shape3d,
            affine=affine,
            out_file=Path(f"{out_prefix}_{run_tag}_ica_zmaps.nii.gz"),
        )
        _save_components_4d(
            components_kv=p_maps,
            mask3d=mask3d,
            shape3d=shape3d,
            affine=affine,
            out_file=Path(f"{out_prefix}_{run_tag}_ica_signalprob.nii.gz"),
        )

    run_meta = {
        "run_index": int(run_idx + 1),
        "input_file": run_file,
        "tr": float(tr),
        "n_timepoints": int(n_t),
        "n_voxels": int(data_vox_t.shape[0]),
        "polort": int(polort),
        "high_pass_hz": None if args.high_pass is None else float(args.high_pass),
        "num_comps_request": args.num_comps,
        "n_components_selected": int(n_components),
        "num_comps_diagnostics": num_diag,
        "pca_diagnostics": {
            "rank_cap": pca_diag["rank_cap"],
            "n_eigs": pca_diag["n_eigs"],
            "first20_scree_ratio": pca_diag["scree_ratio"][:20],
        },
        "icasso": icasso_meta,
        "var_norm": bool(args.var_norm),
        "component_variance_share": var_share.tolist(),
        "condition_labels": cond_labels,
        "condition_durations": cond_durations,
        "component_condition_corr": None if condition_corr is None else condition_corr.tolist(),
        "mixture_model": mixture_meta if args.save_mixture_z else None,
        "outputs": {
            "ica_maps": f"{out_prefix}_{run_tag}_ica_maps.nii.gz",
            "ica_timecourses": f"{out_prefix}_{run_tag}_ica_timecourses.1D",
            "pca_scree_plot": f"{out_prefix}_{run_tag}_pca_scree.png",
            "ica_zmaps": f"{out_prefix}_{run_tag}_ica_zmaps.nii.gz" if args.save_mixture_z else None,
            "ica_signalprob": f"{out_prefix}_{run_tag}_ica_signalprob.nii.gz" if args.save_mixture_z else None,
        },
    }

    with open(f"{out_prefix}_{run_tag}_ica_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    return run_meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run-wise whole-brain ICA demo / sanity-check pipeline",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    req = parser.add_argument_group("Required")
    req.add_argument(
        "-input",
        nargs="+",
        required=True,
        help="Input fMRI run files (.nii/.nii.gz/.nii.zst). Multiple files = run-wise ICA.",
    )

    basic = parser.add_argument_group("Core")
    basic.add_argument("-prefix", type=str, default="ffs_ica", help="Output prefix (default: ffs_ica)")
    basic.add_argument(
        "-num_comps",
        type=str,
        default="auto",
        help=(
            "Component selection: INT, FLOAT(0-1), or auto/melodic/hybrid/current/erank/mp. "
            "'auto' and 'melodic' use a MELODIC-style Bayesian evidence proxy."
        ),
    )
    basic.add_argument(
        "-max_auto_components",
        type=int,
        default=120,
        help="Upper bound for automatic component selection (default: 120)",
    )
    basic.add_argument(
        "-auto_min_components",
        type=int,
        default=5,
        help="Lower bound for automatic component selection (default: 5)",
    )
    basic.add_argument(
        "-auto_var_threshold",
        type=float,
        default=0.90,
        help="Variance threshold used by hybrid/current estimator (default: 0.90)",
    )
    basic.add_argument(
        "-auto_no_mp",
        action="store_true",
        help="Disable MP prior in hybrid/current estimator",
    )

    proc = parser.add_argument_group("Preprocessing")
    proc.add_argument("-mask", type=str, default=None, help="Optional brain mask")
    proc.add_argument("-tr", type=float, default=None, help="Override TR (seconds), else from NIfTI header")
    proc.add_argument("-do_scale", action="store_true", help="Scale each voxel timeseries to mean=100")
    proc.add_argument("-do_blur", type=float, default=None, help="Spatial blur FWHM in mm")
    proc.add_argument("-polort", type=int, default=None, help="Polynomial detrend order (default: auto AFNI style)")
    proc.add_argument(
        "-high_pass",
        type=float,
        default=None,
        help="Fourier high-pass cutoff in Hz (e.g., 0.01)",
    )
    proc.add_argument("-var_norm", dest="var_norm", action="store_true", default=True,
                      help="Variance-normalize ICA timecourses (default: on)")
    proc.add_argument("-no_var_norm", dest="var_norm", action="store_false",
                      help="Disable variance normalization of ICA timecourses")

    ica_opts = parser.add_argument_group("ICA / ICASSO")
    ica_opts.add_argument("-ica_max_iter", type=int, default=1000, help="FastICA max iterations")
    ica_opts.add_argument("-ica_tol", type=float, default=1e-6, help="FastICA convergence tolerance")
    ica_opts.add_argument("-icasso", action="store_true", help="Run ICASSO stability analysis")
    ica_opts.add_argument("-icasso_runs", type=int, default=50, help="Number of ICA runs for ICASSO")
    ica_opts.add_argument("-icasso_min_stability", type=float, default=0.7,
                          help="Stability threshold for ICASSO")
    ica_opts.add_argument("-icasso_batch_size", type=int, default=None,
                          help="Optional batch size for ICASSO similarity matrix")

    task = parser.add_argument_group("Task annotation (optional)")
    task.add_argument("-onsets", nargs="+", default=None, help="AFNI timing files (one per condition)")
    task.add_argument("-durations", nargs="+", default=None,
                      help="Durations: one value for all or one per condition")
    task.add_argument("-microtime_dt", type=float, default=0.1, help="Microtime resolution for task regressors")

    out = parser.add_argument_group("Output")
    out.add_argument("-save_mixture_z", action="store_true", default=True,
                     help="Save mixture-model z and signal-prob maps (default: on)")
    out.add_argument("-no_mixture_z", dest="save_mixture_z", action="store_false",
                     help="Disable mixture-model z/signal-prob map outputs")

    future = parser.add_argument_group("Future modes (not yet implemented)")
    future.add_argument("-temp_concat", action="store_true", help="Placeholder for future temporal concatenation ICA")
    future.add_argument("-tensor", action="store_true", help="Placeholder for future tensorial ICA")

    misc = parser.add_argument_group("Misc")
    misc.add_argument("-cpu", action="store_true", help="Force CPU")
    misc.add_argument("-verbose", action="store_true", help="Verbose logging")
    misc.add_argument("-help", "--help", action="help")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.temp_concat or args.tensor:
        raise NotImplementedError(
            "-temp_concat and -tensor are placeholders for future versions. "
            "Current implementation supports run-wise ICA only."
        )

    if (args.onsets is None) ^ (args.durations is None):
        raise ValueError("Use -onsets and -durations together for task correlation annotation")

    input_files = parse_input_files(args.input)
    device = torch.device("cpu") if args.cpu else get_device()

    print_cli_header("ffs_ica.py", "Fast run-wise whole-brain ICA")
    print(f"Device: {device}")
    print(f"Runs: {len(input_files)}")

    shared_mask = None
    if args.mask is not None:
        shared_mask = load_afni_mask(args.mask)
        print(f"Mask voxels: {int(shared_mask.sum()):,}")

    all_meta = []
    for run_idx, run_file in enumerate(input_files):
        print(f"\n[{run_idx + 1}/{len(input_files)}] Processing: {run_file}")
        run_meta = _run_single_ica(
            run_file=run_file,
            run_idx=run_idx,
            args=args,
            device=device,
            shared_mask=shared_mask,
            onsets_files=args.onsets,
            durations=args.durations,
        )
        all_meta.append(run_meta)
        print(
            f"  Selected components: {run_meta['n_components_selected']} | "
            f"IC1 share: {run_meta['component_variance_share'][0] * 100:.2f}%"
        )

    summary_path = f"{args.prefix}_ica_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "n_runs": len(input_files),
                "input_files": input_files,
                "num_comps_request": args.num_comps,
                "device": str(device),
                "runs": all_meta,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("✅ ffs_ica complete")
    print(f"Summary: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
