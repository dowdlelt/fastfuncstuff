#!/usr/bin/env python3

"""
ffs_phasereg - Phase regression for macrovascular BOLD suppression

Regresses magnitude fMRI signal on phase to remove the contribution of
oriented macrovasculature (pial veins, cerebral veins).  The corrected
output retains microvascular BOLD and is ready for standard GLM analysis.

Uses Deming regression (errors-in-variables) by default, which accounts
for measurement noise in both magnitude and phase.

Basic usage (no task removal - e.g. after NORDIC, or resting state):
    ffs_phasereg -magnitude epi_mag.nii.gz \\
                 -phase epi_phase.nii.gz \\
                 -prefix epi_pr -tr 2.0

With TENT-based task removal (recommended for task data):
    ffs_phasereg -magnitude run*.nii.gz \\
                 -phase run*_phase.nii.gz \\
                 -prefix epi_pr -tr 2.0 \\
                 -task_removal tent \\
                 -onsets stim_A.1D stim_B.1D \\
                 -tent_window 20

With BIDS events files:
    ffs_phasereg -magnitude run*.nii.gz \\
                 -phase run*_phase.nii.gz \\
                 -prefix epi_pr -tr 2.0 \\
                 -task_removal tent \\
                 -events sub-01_task-finger_run-*_events.tsv

Resting state (FFT noise estimation from high frequencies):
    ffs_phasereg -magnitude rest_mag.nii.gz \\
                 -phase rest_phase.nii.gz \\
                 -prefix rest_pr -tr 2.0 \\
                 -phi_method fft -freq_range 0.1

References
----------
Menon RS (2002). MRM 47:1-9.
Stanley OW et al (2021). NeuroImage 117631.
Knudsen L et al (2023). NeuroImage 271:120011.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_utils import add_verbose_arg


class _HelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    pass


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase regression for macrovascular BOLD suppression",
        add_help=False,
        formatter_class=_HelpFormatter,
    )

    # ── Required ─────────────────────────────────────────────────────────
    req = parser.add_argument_group("Required Arguments")
    req.add_argument(
        "-magnitude", nargs="+", metavar="FILE", required=True,
        help="Magnitude fMRI data files (one per run). Supports wildcards.",
    )
    req.add_argument(
        "-phase", nargs="+", metavar="FILE", required=True,
        help="Phase fMRI data files (radians, unwrapped, one per run). "
             "Must match magnitude files in order and dimensions.",
    )
    req.add_argument(
        "-prefix", required=True, metavar="OUTPUT",
        help="Output file prefix (e.g. results/epi_pr).",
    )

    # ── Task removal ─────────────────────────────────────────────────────
    task = parser.add_argument_group("Task Removal Options")
    task.add_argument(
        "-task_removal", choices=["none", "tent", "canonical"],
        default="none",
        help="Task removal before slope estimation: "
             "'none' (direct regression), "
             "'tent' (TENT/FIR basis - model-free, recommended), "
             "'canonical' (SPM HRF convolution).",
    )
    task.add_argument(
        "-onsets", nargs="+", metavar="FILE",
        help="AFNI-format onset timing files (one per condition, "
             "each with one row per run). Mutually exclusive with -events.",
    )
    task.add_argument(
        "-events", nargs="+", metavar="TSV",
        help="BIDS events TSV files (one per run). "
             "Mutually exclusive with -onsets.",
    )
    task.add_argument(
        "-event_ignore", nargs="+", metavar="CONDITION",
        help="trial_type values to skip when using -events.",
    )
    task.add_argument(
        "-event_cols", nargs=3, metavar=("ONSET", "DUR", "TYPE"),
        help="Custom column names for -events (default: onset, duration, trial_type).",
    )
    task.add_argument(
        "-tent_window", type=float, default=20.0, metavar="SECONDS",
        help="TENT window duration in seconds (for -task_removal tent).",
    )
    task.add_argument(
        "-durations", nargs="+", metavar="SECONDS",
        help="Per-condition stimulus durations for auto window estimation "
             "(with -onsets). With -events, durations come from the TSV.",
    )

    # ── Regression options ───────────────────────────────────────────────
    reg = parser.add_argument_group("Regression Options")
    reg.add_argument(
        "-regression", choices=["deming", "ols"], default="deming",
        help="Regression method. 'deming' corrects for noise in both "
             "magnitude and phase (recommended). 'ols' for comparison.",
    )
    reg.add_argument(
        "-phi", type=float, default=None, metavar="VALUE",
        help="Fixed variance ratio for Deming regression. "
             "If not set, estimated automatically per voxel.",
    )
    reg.add_argument(
        "-phi_method", choices=["fft", "residual"], default="fft",
        help="How to estimate phi: 'fft' uses out-of-band spectral power "
             "(works for task and rest), 'residual' uses temporal variance "
             "of GLM residuals.",
    )
    reg.add_argument(
        "-freq_range", nargs="+", type=float, default=[0.1],
        metavar="HZ",
        help="Frequency range for FFT noise estimation. "
             "One value = lower bound (upper = Nyquist). "
             "Two values = lower and upper bounds.",
    )

    # ── Processing options ───────────────────────────────────────────────
    proc = parser.add_argument_group("Processing Options")
    proc.add_argument(
        "-polort", type=str, default="A", metavar="N",
        help="Polynomial drift order. 'A' = auto (AFNI formula). "
             "Integer for fixed order. -1 for none.",
    )
    proc.add_argument(
        "-tr", type=float, metavar="SECONDS",
        help="Override TR from input file headers.",
    )
    proc.add_argument(
        "-mask", metavar="FILE",
        help="Brain mask (restricts analysis to mask voxels).",
    )
    proc.add_argument(
        "-motion", nargs="+", metavar="FILE",
        help="Motion parameter files (one per run, AFNI dfile format). "
             "Added as nuisance regressors.",
    )

    # ── Hardware ─────────────────────────────────────────────────────────
    hw = parser.add_argument_group("Hardware Options")
    hw.add_argument(
        "-device", type=str, default=None,
        help="PyTorch device: cpu, cuda, mps (default: auto-detect).",
    )

    # ── Output ───────────────────────────────────────────────────────────
    out = parser.add_argument_group("Output Options")
    add_verbose_arg(out, default=0)

    parser.add_argument("-help", action="store_true",
                        help="Show this help message and exit.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for ffs_phasereg."""
    parser = create_parser()

    if "-help" in (argv or sys.argv):
        print(__doc__)
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    # ── Imports ──────────────────────────────────────────────────────────
    try:
        from fastfuncstuff.cli_utils import (
            auto_polort,
            parse_device_arg,
            parse_prefix,
        )
        from fastfuncstuff.io.afni import (
            get_tr_from_file,
            load_afni_mask,
            load_nifti,
            save_nifti,
        )
        from fastfuncstuff.phasereg.core import phase_regress
        from fastfuncstuff.utils import configure_torch_backends, get_device
    except ImportError as e:
        print(f"ERROR: Could not import fastfuncstuff: {e}", file=sys.stderr)
        return 1

    pfx = parse_prefix(args.prefix)
    args.prefix = pfx.stem
    nii_ext = pfx.nifti_ext

    # Device
    device, _, _ = parse_device_arg(args.device)
    configure_torch_backends(device)

    if args.verb >= 1:
        print(f"\n{'=' * 70}")
        print("Phase Regression (ffs_phasereg)")
        print(f"{'=' * 70}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {device}")

    # ── Validate inputs ──────────────────────────────────────────────────
    n_runs = len(args.magnitude)
    if len(args.phase) != n_runs:
        print(
            f"ERROR: Number of magnitude ({n_runs}) and phase "
            f"({len(args.phase)}) files must match.",
            file=sys.stderr,
        )
        return 1

    if args.task_removal != "none" and not args.onsets and not args.events:
        print(
            f"ERROR: -task_removal {args.task_removal} requires -onsets or -events.",
            file=sys.stderr,
        )
        return 1

    if args.onsets and args.events:
        print("ERROR: -onsets and -events are mutually exclusive.", file=sys.stderr)
        return 1

    # ── Parse frequency range ────────────────────────────────────────────
    if len(args.freq_range) == 1:
        freq_range = (args.freq_range[0], None)
    elif len(args.freq_range) == 2:
        freq_range = (args.freq_range[0], args.freq_range[1])
    else:
        print("ERROR: -freq_range takes 1 or 2 values.", file=sys.stderr)
        return 1

    # ── Load data ────────────────────────────────────────────────────────
    if args.verb >= 1:
        print("\nLoading data...")

    mag_list = []
    pha_list = []
    n_tp_per_run = []
    tr_values = []
    nx = ny = nz = None
    ref_img_path = args.magnitude[0]

    for i in range(n_runs):
        mag_path = args.magnitude[i]
        pha_path = args.phase[i]

        for p in (mag_path, pha_path):
            if not Path(p).exists():
                print(f"ERROR: File not found: {p}", file=sys.stderr)
                return 1

        mag_img = load_nifti(mag_path)
        pha_img = load_nifti(pha_path)

        mag_data = mag_img.get_fdata(dtype=np.float32)
        pha_data = pha_img.get_fdata(dtype=np.float32)

        if mag_data.ndim != 4:
            print(f"ERROR: Expected 4D data, got {mag_data.ndim}D: {mag_path}",
                  file=sys.stderr)
            return 1
        if pha_data.shape != mag_data.shape:
            print(f"ERROR: Phase shape {pha_data.shape} != magnitude shape "
                  f"{mag_data.shape}", file=sys.stderr)
            return 1

        if nx is None:
            nx, ny, nz = mag_data.shape[:3]

        n_tp = mag_data.shape[3]
        n_tp_per_run.append(n_tp)

        if args.tr is None:
            tr_values.append(get_tr_from_file(mag_path))

        # Flatten to (n_voxels, n_tp)
        mag_list.append(torch.tensor(mag_data.reshape(-1, n_tp), dtype=torch.float32))
        pha_list.append(torch.tensor(pha_data.reshape(-1, n_tp), dtype=torch.float32))

        if args.verb >= 1:
            print(f"  Run {i + 1}: {mag_path} ({n_tp} TRs)")

    # TR
    if args.tr is None:
        if len(set(tr_values)) > 1:
            print(f"ERROR: Inconsistent TRs: {tr_values}", file=sys.stderr)
            return 1
        tr = tr_values[0]
    else:
        tr = args.tr

    if args.verb >= 1:
        print(f"  TR: {tr}s")
        print(f"  Volume: {nx} x {ny} x {nz}")

    # ── Mask ─────────────────────────────────────────────────────────────
    mask = None
    n_all_voxels = nx * ny * nz
    if args.mask:
        mask = load_afni_mask(args.mask)
        if mask.shape != (nx, ny, nz):
            print(f"ERROR: Mask shape {mask.shape} != data shape {(nx, ny, nz)}",
                  file=sys.stderr)
            return 1
        mask_flat = mask.flatten()
        n_voxels = int(mask_flat.sum())
        mag_list = [m[mask_flat.astype(bool)] for m in mag_list]
        pha_list = [p[mask_flat.astype(bool)] for p in pha_list]
        if args.verb >= 1:
            print(f"  Mask: {n_voxels:,} / {n_all_voxels:,} voxels")
    else:
        mask_flat = None
        n_voxels = n_all_voxels

    # ── Polort ───────────────────────────────────────────────────────────
    polort_str = str(args.polort).strip().upper()
    if polort_str == "A":
        run_dur = min(n_tp_per_run) * tr
        polort = auto_polort(run_dur, formula="afni")
        if args.verb >= 1:
            print(f"  Polort: A -> {polort} (run duration {run_dur:.1f}s)")
    else:
        polort = int(polort_str)

    # ── Parse onsets ─────────────────────────────────────────────────────
    onsets_per_condition = None
    if args.onsets:
        from fastfuncstuff.design.builder import parse_afni_timing_file
        onsets_per_condition = []
        for onset_file in args.onsets:
            onsets_by_run = parse_afni_timing_file(onset_file)
            if len(onsets_by_run) != n_runs:
                print(
                    f"ERROR: {onset_file} has {len(onsets_by_run)} runs, "
                    f"expected {n_runs}",
                    file=sys.stderr,
                )
                return 1
            onsets_per_condition.append(onsets_by_run)
        if args.verb >= 1:
            print(f"  Onsets: {len(onsets_per_condition)} conditions")

    elif args.events:
        from fastfuncstuff.design.bids_events import parse_bids_events
        if len(args.events) != n_runs:
            print(
                f"ERROR: {len(args.events)} events files but {n_runs} runs.",
                file=sys.stderr,
            )
            return 1
        event_cols = tuple(args.event_cols) if args.event_cols else None
        try:
            bids_onsets, bids_durations, bids_labels = parse_bids_events(
                event_files=args.events,
                event_ignore=args.event_ignore,
                event_cols=event_cols,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        onsets_per_condition = bids_onsets
        if args.verb >= 1:
            print(f"  BIDS events: {len(bids_labels)} conditions ({', '.join(bids_labels)})")

    # ── Auto tent window from durations ──────────────────────────────────
    tent_window = args.tent_window
    if args.task_removal == "tent" and args.durations and onsets_per_condition is not None:
        from fastfuncstuff.design.builder import parse_durations
        from fastfuncstuff.design.hrf import compute_windows_from_durations
        n_conds = len(onsets_per_condition)
        cond_labels = [f"cond{i}" for i in range(n_conds)]
        condition_durations = parse_durations(args.durations, n_conds, cond_labels)
        windows = compute_windows_from_durations(condition_durations, tr)
        tent_window = max(top for _, top in windows)
        if args.verb >= 1:
            print(f"  Auto TENT window: {tent_window:.1f}s (from stimulus durations)")

    # ── Load motion nuisance ─────────────────────────────────────────────
    nuisance_per_run = None
    if args.motion:
        if len(args.motion) != n_runs:
            print(
                f"ERROR: {len(args.motion)} motion files but {n_runs} runs.",
                file=sys.stderr,
            )
            return 1
        nuisance_per_run = []
        for mot_file in args.motion:
            mot = np.loadtxt(mot_file, dtype=np.float32)
            if mot.ndim == 1:
                mot = mot[:, np.newaxis]
            nuisance_per_run.append(
                torch.tensor(mot, dtype=torch.float32, device=device)
            )
        if args.verb >= 1:
            print(f"  Motion: {nuisance_per_run[0].shape[1]} parameters per run")

    # ── Run phase regression ─────────────────────────────────────────────
    if args.verb >= 1:
        print()

    result = phase_regress(
        magnitude=mag_list,
        phase=pha_list,
        tr=tr,
        task_removal=args.task_removal,
        onsets_per_condition=onsets_per_condition,
        nuisance_per_run=nuisance_per_run,
        max_poly_degree=polort,
        phi=args.phi,
        phi_method=args.phi_method,
        phi_freq_range=freq_range,
        regression=args.regression,
        tent_window=tent_window,
        device=str(device),
        verbose=args.verb >= 1,
    )

    # ── Save outputs ─────────────────────────────────────────────────────
    if args.verb >= 1:
        print("\nSaving outputs...")

    def _to_volume(data_flat, is_4d=False):
        """Reshape flat data back to volume, respecting mask."""
        if is_4d:
            n_tp = data_flat.shape[1]
            if mask_flat is not None:
                vol = np.zeros((n_all_voxels, n_tp), dtype=np.float32)
                vol[mask_flat.astype(bool)] = data_flat
            else:
                vol = data_flat
            return vol.reshape(nx, ny, nz, n_tp)
        else:
            if mask_flat is not None:
                vol = np.zeros(n_all_voxels, dtype=np.float32)
                vol[mask_flat.astype(bool)] = data_flat
            else:
                vol = data_flat
            return vol.reshape(nx, ny, nz)

    outputs = {}

    # Corrected magnitude (4D)
    corrected_np = result.magnitude_corrected.numpy()
    fname = f"{args.prefix}_corrected{nii_ext}"
    save_nifti(_to_volume(corrected_np, is_4d=True), fname, reference_img=ref_img_path)
    outputs["corrected"] = fname

    # Macrovascular component (4D)
    macro_np = result.macrovascular_component.numpy()
    fname = f"{args.prefix}_macro{nii_ext}"
    save_nifti(_to_volume(macro_np, is_4d=True), fname, reference_img=ref_img_path)
    outputs["macro"] = fname

    # Slope map (3D)
    fname = f"{args.prefix}_slope{nii_ext}"
    save_nifti(_to_volume(result.slope.numpy()), fname, reference_img=ref_img_path)
    outputs["slope"] = fname

    # R2 map (3D)
    fname = f"{args.prefix}_r2{nii_ext}"
    save_nifti(_to_volume(result.r2_phase.numpy()), fname, reference_img=ref_img_path)
    outputs["r2"] = fname

    # Phi map (3D)
    fname = f"{args.prefix}_phi{nii_ext}"
    save_nifti(_to_volume(result.phi.numpy()), fname, reference_img=ref_img_path)
    outputs["phi"] = fname

    if args.verb >= 1:
        for name, path in outputs.items():
            print(f"  {name}: {path}")
        print(f"\n{'=' * 70}")
        print("Phase regression complete!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 70}")
    else:
        print(f"Phase regression complete. Main output: {outputs['corrected']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
