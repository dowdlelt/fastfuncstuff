#!/usr/bin/env python3

"""
ffs_phasereg - Phase regression for macrovascular BOLD suppression

Regresses magnitude fMRI signal on phase to remove the contribution of
oriented macrovasculature (pial veins, cerebral veins).  The corrected
output retains microvascular BOLD.

Uses Deming regression (errors-in-variables) by default — equivalent to
scipy.odr (Menon 2002, Curtis 2014, Stanley 2021); see Liem 2023 /
phaseprep for the canonical reference implementation.

INPUTS REQUIRED UPSTREAM
  - Coil-combination preserving phase (SVD / VRC / COMPOSER).
  - Motion correction in real/imag space (so phase isn't scrambled).
  - Temporal phase unwrap.  (First-volume subtraction is optional —
    polynomial detrending here absorbs the constant shift.)
  - Phase passed in must be radians.

OUTPUT MODES
  Default (analysis endpoint): drift dropped, output re-centered on per-
  voxel mean. Matches phaseprep / Stanley 2021 §2.2.3 exactly.

  -keep_drift (GLM input): drift preserved, slope shrunk to A_eff =
  A/(1 + A²/φ). Use when the corrected mag feeds a downstream GLM.

RECOMMENDED PIPELINES

  Phaseprep / Stanley parity (default for analysis):
    ffs_phasereg -magnitude run_mag.nii.gz -phase run_phase.nii.gz \\
                 -prefix out_pr -polort 1

  Task data, output feeds a GLM (preserves drift + task variance):
    ffs_phasereg -magnitude run*.nii.gz -phase run*_phase.nii.gz \\
                 -prefix epi_pr \\
                 -task_removal tent -onsets stim_A.1D stim_B.1D \\
                 -tent_window 20 -keep_drift

  Resting-state, 7T (noisy phase, regularise via Savitzky-Golay):
    ffs_phasereg -magnitude rest_mag.nii.gz -phase rest_phase.nii.gz \\
                 -prefix rest_pr -polort 1 -phase_filter sgf

  Resting-state, 3T (NORDIC upstream is assumed; see Knudsen 2023):
    ffs_phasereg -magnitude rest_mag.nii.gz -phase rest_phase.nii.gz \\
                 -prefix rest_pr -polort 1

  OLS variant (Chang & Giovanello 2026 approach for 3T):
    ffs_phasereg -magnitude epi.nii.gz -phase epi_phase.nii.gz \\
                 -prefix pr -regression ols

  Knudsen 2023 residual-φ approach (requires task model):
    ffs_phasereg -magnitude epi.nii.gz -phase epi_phase.nii.gz \\
                 -prefix pr -task_removal tent -onsets stim.1D \\
                 -phi_method residual

  Data-driven phase smoothing (Barry & Gore 2014, expensive):
    ffs_phasereg -magnitude epi.nii.gz -phase epi_phase.nii.gz \\
                 -prefix pr -phase_filter explore

OUTPUTS
  prefix_corrected.nii.gz   M_micro time series (4D)
  prefix_macro.nii.gz       what was subtracted (4D)
  prefix_slope.nii.gz       per-voxel A (3D, raw Deming slope)
  prefix_phi.nii.gz         per-voxel variance ratio used (3D)
  prefix_r2.nii.gz          ODR-style R² matching phaseprep (3D) [-r2_mode odr|both]
  prefix_r2_naive.nii.gz    naive R² (voxels most affected) (3D) [-r2_mode naive|both]
  prefix_mask.nii.gz        analysis mask (3D)

  Intermediates (written with -save_intermediates):
  prefix_mag_dt.nii.gz      magnitude after poly detrending (4D)
  prefix_pha_dt.nii.gz      phase after poly detrending (4D)
  prefix_pha_dt_filt.nii.gz phase after poly detrending + SGF filter (4D)
  prefix_mag_res.nii.gz     magnitude after poly + task removal (4D)
  prefix_pha_res_filt.nii.gz phase after poly + task removal + SGF (4D)

References
----------
Menon RS (2002). MRM 47:1-9.
Barry RL & Gore JC (2014). Hum Brain Mapp 35:3832-3840.
Curtis AT et al (2014). NeuroImage 100:51-59.
Stanley OW et al (2021). NeuroImage 117631.
Knudsen L et al (2023). NeuroImage 271:120011.
Chang WT & Giovanello KS (2026). eLife 12:RP92805.
Liem BT (2023). MSc thesis, Western University. (phaseprep)
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
             "'none' (direct regression, OK after NORDIC or for RS), "
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
        help="Regression method. 'deming' is the closed-form Deming/ODR "
             "(equivalent to scipy.odr at fMRI φ scales — see Menon 2002, "
             "Curtis 2014, Stanley 2021, phaseprep). 'ols' regresses mag on "
             "phase without errors-in-variables correction (Chang & "
             "Giovanello 2026 use OLS at 3T after NORDIC).",
    )
    reg.add_argument(
        "-phi", type=float, default=None, metavar="VALUE",
        help="Fixed variance ratio for Deming regression. "
             "If not set, estimated automatically per voxel via -phi_method.",
    )
    reg.add_argument(
        "-phi_method", choices=["fft", "residual"], default="fft",
        help="How to estimate phi (Var(eps_mag)/Var(eps_phase)): "
             "'fft' uses out-of-band spectral power above -freq_range "
             "(Curtis 2014, phaseprep default; works for task and rest). "
             "'residual' uses temporal variance of (poly+task+nuisance)-"
             "projected residuals (Knudsen 2023). "
             "WARNING: 'residual' without -task_removal on task data will "
             "include task variance in the noise estimate and inflate phi. "
             "Pair 'residual' with -task_removal {tent,canonical} for "
             "task data; 'fft' is safe for both.",
    )
    reg.add_argument(
        "-freq_range", nargs="+", type=float, default=[0.15],
        metavar="HZ",
        help="Frequency range for FFT noise estimation. "
             "One value = lower bound (upper = Nyquist). "
             "Two values = lower and upper bounds. "
             "Default 0.15 Hz matches Stanley 2021 §2.2.3 and phaseprep "
             "(noise_lb=0.15). Curtis 2014 used 0.1 Hz; either is reasonable.",
    )

    # ── Phase filtering ──────────────────────────────────────────────────
    pfilt = parser.add_argument_group(
        "Phase Filtering Options (Barry & Gore 2014)"
    )
    pfilt.add_argument(
        "-phase_filter", choices=["none", "sgf", "explore"],
        default="none",
        help="Pre-filter phase time series before regression. "
             "'none': no filtering (Stanley/phaseprep default). "
             "'sgf': Savitzky-Golay with -sgf_window/-sgf_order — same fit "
             "behaviour as 'none' but on a smoother phase, which acts as a "
             "phase-noise regulariser and typically tightens R² by reducing "
             "high-frequency phase contribution to the residual. "
             "'explore': per-voxel data-driven parameter search "
             "optimising |Pearson r(filtered_phase, mag_dt)| (Barry & Gore "
             "2014's phase-magnitude correlation criterion — NOT our ODR-"
             "shrinkage R², which is dominated by 1/(1+A²/φ) and would be a "
             "poor filter-quality metric). Computationally expensive. "
             "Recommended at 7T where phase SNR is low (Hagberg et al 2008 "
             "found k_phi >> k_mag).",
    )
    pfilt.add_argument(
        "-sgf_window", type=int, default=None, metavar="N",
        help="SGF window length in TRs (must be odd; auto-incremented if "
             "even). Default: auto = round(20s / TR), floored at 5. Larger "
             "= more smoothing. Only used with -phase_filter sgf.",
    )
    pfilt.add_argument(
        "-sgf_order", type=int, default=3, metavar="P",
        help="SGF polynomial order (must be < window length). Lower = "
             "more smoothing. Only used with -phase_filter sgf.",
    )

    # ── Processing options ───────────────────────────────────────────────
    proc = parser.add_argument_group("Processing Options")
    proc.add_argument(
        "-polort", type=str, default="A", metavar="N",
        help="Polynomial drift order (Legendre, per run). 'A' = auto "
             "(AFNI formula based on run duration). Integer for fixed "
             "order. -1 for none. Use -polort 1 for exact phaseprep parity "
             "(linear-only detrend, what PreprocessPhase + DetrendMag do). "
             "Higher orders remove more drift before the fit but can absorb "
             "shared low-frequency mag-phase content that you may want to "
             "keep — there's a tradeoff.",
    )
    proc.add_argument(
        "-tr", type=float, metavar="SECONDS",
        help="Repetition time. Default: read from NIfTI pixdim/zooms. "
             "Override here if your header is wrong.",
    )
    proc.add_argument(
        "-mask", metavar="FILE",
        help="Brain mask (restricts analysis to mask voxels). "
             "If not provided, voxels are filtered by -signal_thresh.",
    )
    proc.add_argument(
        "-signal_thresh", type=float, default=0.03, metavar="FRAC",
        help="Minimum mean-signal fraction to include a voxel in "
             "regression (default 0.03 = 3%% of max, matching phaseprep). "
             "Voxels below this have slope=0 and R²=0. "
             "Set to 0 to disable: corrected output stays clean (ODR "
             "shrinkage caps every voxel), but R² map shows bright noise "
             "voxels because ODR-style R² reads unfittable data as "
             "'perfect within noise budget' — visual artifact, not a bug.",
    )
    proc.add_argument(
        "-no_auto_mask", action="store_true",
        help="Disable automatic brain masking when -mask is not provided. "
             "By default, ffs_phasereg computes a brain mask from the mean "
             "magnitude (using AFNI-compatible automask) to exclude air, "
             "skull, and non-brain tissue before regression.",
    )
    proc.add_argument(
        "-keep_drift", action="store_true",
        help="GLM-friendly output mode. Default (off) is phaseprep parity: "
             "corrected = mean + shrunk_residual, drift dropped, ill-"
             "conditioned voxels collapse to per-voxel mean. With this flag "
             "ON, corrected = mag_orig - A_eff·(phase_dt - mean), where "
             "A_eff = A/(1 + A²/φ) is the shrunken slope. Polynomial drift "
             "is preserved (your downstream GLM models it), and A_eff is "
             "naturally bounded by √φ/2 so high-Deming-slope voxels still "
             "produce a clean correction instead of speckle. "
             "Recommended when the corrected mag is the input to a GLM "
             "rather than the analysis endpoint.",
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
    out.add_argument(
        "-r2_mode", choices=["odr", "naive", "both"], default="odr",
        help="Which R² map(s) to write. 'odr' (default): ODR-shrinkage R² "
             "matching phaseprep / Stanley 2021 — the canonical metric. "
             "'naive': 1 - SS_res(observed) / SS_tot without shrinkage — "
             "highlights voxels with the largest raw phase-regression effect, "
             "useful for spotting which voxels were most changed even when "
             "ODR inflation makes the canonical R² appear small. "
             "'both': write prefix_r2.nii.gz (ODR) and prefix_r2_naive.nii.gz.",
    )
    out.add_argument(
        "-save_intermediates", action="store_true",
        help="Write intermediate time-series volumes for step-by-step inspection: "
             "prefix_mag_dt (poly-detrended magnitude), "
             "prefix_pha_dt (poly-detrended phase), "
             "prefix_pha_dt_filt (detrended + SGF-filtered phase, differs from "
             "prefix_pha_dt only when -phase_filter sgf|explore is active), "
             "prefix_mag_res (magnitude after poly + task removal, used for "
             "slope estimation), and "
             "prefix_pha_res_filt (phase after poly + task + SGF, used for phi "
             "estimation and slope estimation).",
    )
    add_verbose_arg(out, default=0)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for ffs_phasereg."""
    parser = create_parser()
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

        mag_list.append(torch.tensor(mag_data.reshape(-1, n_tp), dtype=torch.float32))
        pha_list.append(torch.tensor(pha_data.reshape(-1, n_tp), dtype=torch.float32))

        if args.verb >= 1:
            print(f"  Run {i + 1}: {mag_path} ({n_tp} TRs)")

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

    # ── SGF window: TR-adaptive default (~20s, HRF-duration based) ──────
    if args.sgf_window is None:
        sgf_window = round(20.0 / tr)
        if sgf_window % 2 == 0:
            sgf_window += 1
        sgf_window = max(sgf_window, 5)
        if args.verb >= 1:
            print(f"  SGF window: {sgf_window} (~20s / {tr:.2f}s TR)")
    else:
        sgf_window = args.sgf_window
        if sgf_window % 2 == 0:
            print("ERROR: -sgf_window must be odd.", file=sys.stderr)
            return 1

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
            print(f"  Mask (provided): {n_voxels:,} / {n_all_voxels:,} voxels")
    elif not args.no_auto_mask:
        from fastfuncstuff.processing.mask import automask
        mean_mag = torch.stack([m.mean(dim=1) for m in mag_list]).mean(dim=0)
        mean3d = mean_mag.reshape(nx, ny, nz)
        mask3d = automask(mean3d, dilate_extra=2, device=device,
                          verbose=args.verb >= 1)
        mask_flat = mask3d.flatten().cpu()
        n_voxels = int(mask_flat.sum().item())
        mask_bool = mask_flat.bool()
        mag_list = [m[mask_bool] for m in mag_list]
        pha_list = [p[mask_bool] for p in pha_list]
        if args.verb >= 1:
            print(f"  Mask (auto): {n_voxels:,} / {n_all_voxels:,} voxels")
    else:
        mask_flat = None
        n_voxels = n_all_voxels
        if args.verb >= 1:
            print(f"  Mask: none (using all {n_all_voxels:,} voxels)")

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
        phase_filter=args.phase_filter,
        sgf_window=sgf_window,
        sgf_order=args.sgf_order,
        signal_thresh=args.signal_thresh,
        keep_drift=args.keep_drift,
        device=str(device),
        verbose=args.verb >= 1,
        r2_mode=args.r2_mode,
        save_intermediates=args.save_intermediates,
    )

    # ── Save outputs ─────────────────────────────────────────────────────
    if args.verb >= 1:
        print("\nSaving outputs...")

    def _to_volume(data_flat, is_4d=False):
        if is_4d:
            n_tp = data_flat.shape[1]
            if mask_flat is not None:
                vol = np.zeros((n_all_voxels, n_tp), dtype=np.float32)
                vol[np.asarray(mask_flat, dtype=bool)] = data_flat
            else:
                vol = data_flat
            return vol.reshape(nx, ny, nz, n_tp)
        else:
            if mask_flat is not None:
                vol = np.zeros(n_all_voxels, dtype=np.float32)
                vol[np.asarray(mask_flat, dtype=bool)] = data_flat
            else:
                vol = data_flat
            return vol.reshape(nx, ny, nz)

    outputs = {}

    corrected_np = result.magnitude_corrected.numpy()
    fname = f"{args.prefix}_corrected{nii_ext}"
    save_nifti(_to_volume(corrected_np, is_4d=True), fname, reference_img=ref_img_path)
    outputs["corrected"] = fname

    macro_np = result.macrovascular_component.numpy()
    fname = f"{args.prefix}_macro{nii_ext}"
    save_nifti(_to_volume(macro_np, is_4d=True), fname, reference_img=ref_img_path)
    outputs["macro"] = fname

    fname = f"{args.prefix}_slope{nii_ext}"
    save_nifti(_to_volume(result.slope.numpy()), fname, reference_img=ref_img_path)
    outputs["slope"] = fname

    if args.r2_mode in ("odr", "both"):
        fname = f"{args.prefix}_r2{nii_ext}"
        save_nifti(_to_volume(result.r2_phase.numpy()), fname, reference_img=ref_img_path)
        outputs["r2"] = fname

    if args.r2_mode in ("naive", "both") and result.r2_naive is not None:
        fname = f"{args.prefix}_r2_naive{nii_ext}"
        save_nifti(_to_volume(result.r2_naive.numpy()), fname, reference_img=ref_img_path)
        outputs["r2_naive"] = fname

    fname = f"{args.prefix}_phi{nii_ext}"
    save_nifti(_to_volume(result.phi.numpy()), fname, reference_img=ref_img_path)
    outputs["phi"] = fname

    fname = f"{args.prefix}_mask{nii_ext}"
    save_nifti(_to_volume(result.voxel_mask.numpy().astype(np.float32)), fname, reference_img=ref_img_path)
    outputs["mask"] = fname

    if args.save_intermediates:
        interm_specs = [
            ("mag_dt",       result.mag_detrended,      True,  "poly-detrended magnitude"),
            ("pha_dt",       result.pha_detrended,       True,  "poly-detrended phase"),
            ("pha_dt_filt",  result.pha_detrended_filt,  True,  "detrended + filtered phase"),
            ("mag_res",      result.mag_residual,        True,  "magnitude task-residual"),
            ("pha_res_filt", result.pha_residual_filt,   True,  "phase task-residual + filtered"),
        ]
        for key, data, is_4d, label in interm_specs:
            if data is None:
                continue
            fname = f"{args.prefix}_{key}{nii_ext}"
            save_nifti(_to_volume(data.numpy(), is_4d=is_4d), fname, reference_img=ref_img_path)
            outputs[key] = fname
            if args.verb >= 1:
                print(f"  {key} ({label}): {fname}")

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
