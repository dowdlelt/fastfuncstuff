#!/usr/bin/env python3

"""

ffs_deconvolve.py - Fast fMRI deconvolution analysis using FIR/TENT models

Estimates HRF shapes directly from data using:
- FIR (Finite Impulse Response): For TR-locked onsets (simple diagonal design)
- TENT: Piecewise linear basis for non-TR-locked onsets
- TENTzero: TENT with forced zero start/end (ensures continuous HRF)

The tool automatically detects whether onsets are TR-locked and chooses the
appropriate model, or you can specify explicitly.

Basic usage:
    ffs_deconvolve.py -input run1.nii.gz run2.nii.gz \\
                      -onsets task.txt \\
                      -duration 20 \\
                      -prefix results/GLM

With multiple conditions:
    ffs_deconvolve.py -input run*.nii.gz \\
                      -onsets faces.txt scenes.txt objects.txt \\
                      -labels faces scenes objects \\
                      -duration 20 \\
                      -prefix sub01_deconv

TENT model (non-TR-locked):
    ffs_deconvolve.py -input data.nii.gz \\
                      -onsets task.txt \\
                      -model TENT \\
                      -window 0 20 \\
                      -prefix results/GLM

Per-condition TENT windows:
    ffs_deconvolve.py -input run*.nii.gz \\
                      -onsets faces.txt scenes.txt objects.txt \\
                      -labels faces scenes objects \\
                      -model TENT \\
                      -window 0,15 0,20 0,25 \\
                      -prefix results/GLM

For help:
    ffs_deconvolve.py -help
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from fastfuncstuff.cli_help import FfsHelpFormatter

try:
    import nibabel as nib  # noqa: F401 — availability check
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

# Import fastfuncstuff modules
try:
    from fastfuncstuff.cli_utils import (
        add_cv_metric_arg,
        add_cv_strategy_arg,
        add_device_arg,
        add_load_threads_arg,
        add_ortvec_arguments,
        add_trim_args,
        add_verbose_arg,
        append_nuisance_blocks,
        apply_trim_to_timing,
        auto_polort,
        collect_nuisance_blocks,
        load_and_preprocess_runs,
        parse_cv_strategy,
        parse_input_files,
        parse_prefix,
        parse_timing_spec,
        preflight_check,
        print_cli_header,
        run_lengths_from_starts,
        setup_device,
        spinner,
        trim_spec_from_args,
    )
    from fastfuncstuff.design.builder import (
        legendre_polynomials,
        pack_for_shared_task_glm,
    )
    from fastfuncstuff.design.matrices import (
        is_tr_locked,
        make_csplin_design,
        make_tent_design,
        save_iresp,
    )
    from fastfuncstuff.glm.core import fit_glm
    from fastfuncstuff.glm.xval import compute_r2_metric
    from fastfuncstuff.io.afni import save_nifti
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    print("Make sure fastfuncstuff is installed: pip install -e .")
    sys.exit(1)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Fast fMRI deconvolution with FIR/TENT models",
        formatter_class=FfsHelpFormatter,
    )

    # Required arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-input",
        nargs="+",
        metavar="FILE",
        required=True,
        help="Input fMRI data files (one per run). Can use wildcards: run*.nii.gz",
    )

    required.add_argument(
        "-onsets",
        nargs="+",
        metavar="FILE",
        help=(
            "Onset timing files in AFNI format (one file per condition, "
            "each with one row per run).  Mutually exclusive with -events."
        ),
    )
    required.add_argument(
        "-events",
        nargs="+",
        metavar="TSV",
        help=(
            "BIDS events TSV files, one per run.  Mutually exclusive with -onsets. "
            "Files are sorted by run number automatically (run-1 and run-01 both work). "
            "Conditions come from unique trial_type values; durations are read from the TSV "
            "and used to auto-estimate per-condition HRF windows unless -window is given. "
            "Use -event-ignore to skip conditions; -event-cols for non-standard column names."
        ),
    )
    required.add_argument(
        "-event-ignore",
        nargs="+",
        metavar="CONDITION",
        help="trial_type values to exclude when using -events (e.g. -event-ignore fixation null).",
    )
    required.add_argument(
        "-event-cols",
        nargs=3,
        metavar=("ONSET_COL", "DURATION_COL", "TRIAL_TYPE_COL"),
        help=(
            "Custom column names for -events TSV files, replacing BIDS defaults. "
            "E.g. -event-cols onset_time duration_s condition_name"
        ),
    )

    required.add_argument(
        "-prefix",
        required=True,
        metavar="OUTPUT",
        help="Output file prefix (e.g., results/GLM or sub01_deconv)",
    )

    # Model options
    model_opts = parser.add_argument_group("Deconvolution Model Options")
    model_opts.add_argument(
        "-model",
        choices=["AUTO", "FIR", "TENT", "TENTzero", "CSPLIN", "CSPLINzero", "FLOBS"],
        default="AUTO",
        help=(
            "Deconvolution model: AUTO (auto-detect TR-locking), FIR, TENT, TENTzero, "
            "CSPLIN (cubic spline — smoother than TENT), CSPLINzero, or FLOBS "
            "(constrained K-basis fit with a Gaussian shape prior derived from "
            "half-cosine HRF samples — TR04MW2).  Default: AUTO."
        ),
    )

    # FLOBS-specific options
    flobs_opts = parser.add_argument_group("FLOBS Model Options (only used with -model FLOBS)")
    flobs_opts.add_argument(
        "-flobs-n-basis",
        type=int,
        default=3,
        metavar="K",
        help=(
            "Number of FLOBS basis functions (eigenHRFs) to retain. "
            "TR04MW2 default is 3 — first three look like canonical + "
            "temporal-derivative + dispersion-derivative."
        ),
    )
    flobs_opts.add_argument(
        "-flobs-n-samples",
        type=int,
        default=1000,
        metavar="N",
        help="Number of half-cosine HRF samples used to derive the basis.",
    )
    flobs_opts.add_argument(
        "-flobs-window",
        type=float,
        default=32.0,
        metavar="SECONDS",
        help="FLOBS basis duration (s).  Sampled at -flobs-dt resolution.",
    )
    flobs_opts.add_argument(
        "-flobs-dt",
        type=float,
        default=0.1,
        metavar="SECONDS",
        help="FLOBS basis sample spacing (s).  0.1 matches canonical libraries.",
    )
    flobs_opts.add_argument(
        "-flobs-prior-weight",
        default="auto",
        metavar="VALUE",
        help=(
            "Strength of the FLOBS shape prior.  'auto' uses the "
            "Bayesian-optimal weight σ² (estimated from an OLS pre-pass). "
            "A float overrides as a multiplier on σ² (e.g. 2.0 = twice "
            "as strong).  0 = unconstrained OLS."
        ),
    )
    flobs_opts.add_argument(
        "-flobs-seed",
        type=int,
        default=42,
        help="Seed for the FLOBS half-cosine sampler.",
    )
    flobs_opts.add_argument(
        "-flobs-save-iresp",
        action="store_true",
        default=True,
        help=(
            "Save the reconstructed per-condition HRF as a 4D iresp NIfTI "
            "(time on the last axis at -flobs-dt resolution).  This is the "
            "FLOBS analogue of TENT's iresp output.  Default: on."
        ),
    )
    flobs_opts.add_argument(
        "-flobs-no-iresp",
        action="store_false",
        dest="flobs_save_iresp",
        help="Disable iresp save (only PC weights + amplitude maps emitted).",
    )

    model_opts.add_argument(
        "-duration",
        type=float,
        metavar="SECONDS",
        help=(
            "Legacy fallback: single HRF window length in seconds applied to all conditions. "
            "Prefer -window (explicit) or -durations (auto-estimate from stimulus durations). "
            "If -window or -durations is given, -duration is ignored."
        ),
    )

    model_opts.add_argument(
        "-durations",
        nargs="+",
        metavar="SECONDS",
        help=(
            "Per-condition STIMULUS durations in seconds (one value per condition, or a single "
            "value for all).  Used with -onsets to auto-estimate the per-condition HRF window "
            "via canonical-HRF convolution.  With -events the durations come from the TSV "
            "automatically and this flag is not needed.  Ignored if -window is given."
        ),
    )

    model_opts.add_argument(
        "-window",
        nargs="+",
        metavar="WINDOW",
        help=(
            "Explicit HRF analysis window(s) in seconds after stimulus onset.  "
            "Applies to FIR, TENT, TENTzero, CSPLIN, and CSPLINzero.  "
            "Formats: '0 15' (shared, all conditions), '0,15' (shared), "
            "or '0,15 0,20 0,25' (per-condition, one pair per condition).  "
            "When given, overrides auto-estimation from -durations/-events."
        ),
    )

    model_opts.add_argument(
        "-add-lag",
        nargs="+",
        type=int,
        metavar="TRS",
        help=(
            "Per-condition lag adjustment in TRs applied on top of the estimated or explicit "
            "window.  Positive = more lags, negative = fewer.  "
            "Provide one value (broadcast to all conditions) or one per condition.  "
            "E.g. -add-lag 2 or -add-lag 1 0 -2"
        ),
    )

    model_opts.add_argument(
        "-tent-n-basis",
        type=int,
        metavar="N",
        help="Number of TENT basis functions (knots). Default: auto-calculated for TR spacing.",
    )

    model_opts.add_argument(
        "-xval-tr-range",
        type=int,
        default=0,
        metavar="N",
        help="Cross-validate the TENT window upper bound over ±N TRs around the -window "
        "top, in steps of 1 TR, using leave-one-run-out CV. The top with the highest "
        "mean held-out R² is used for the final fit. "
        "E.g., '-window 0 15 -xval-tr-range 3' tries tops 12s..18s (default: 0 = disabled).",
    )

    model_opts.add_argument(
        "-per-voxel",
        action="store_true",
        help="When combined with -xval-tr-range, select the best window top *per voxel* "
        "rather than a single shared top. Each voxel's HRF is estimated with its "
        "individually optimal window; shorter windows are zero-padded on the right to "
        "the longest candidate window. Saves additional maps: "
        "{prefix}_windowsize.nii.gz (winning top in seconds) and "
        "{prefix}_r2_by_window.nii.gz (LORO R² per candidate, 4D). "
        "Requires -xval-tr-range > 0 and ≥2 runs.",
    )

    model_opts.add_argument(
        "-round-onsets",
        nargs="?",
        const=0.7,
        type=float,
        metavar="THRESHOLD",
        help=(
            "Snap all onset times to the nearest TR boundary. "
            "THRESHOLD (default: 0.7) is the fractional position within a TR above which "
            "an onset rounds up (ceil); below it rounds down (floor). "
            "0.5 = standard nearest-TR rounding.  0.7 = biased toward floor "
            "(only round up if 70%%+ through the TR).  "
            "Reducing TENT/CSPLIN to FIR: use -round-onsets then -model FIR."
        ),
    )

    model_opts.add_argument(
        "-round-durations",
        type=int,
        metavar="PLACES",
        help=(
            "Round stimulus durations to PLACES decimal places before uniquing "
            "and auto-window estimation (0=integer, 1=tenth, etc.).  "
            "Prevents near-identical durations (e.g. 3.0 vs 3.03) from creating "
            "spurious per-condition variation."
        ),
    )

    model_opts.add_argument(
        "-tr-lock-threshold",
        type=float,
        default=0.1,
        metavar="FRAC",
        help="TR-locking detection threshold as fraction of TR (default: 0.1 = 10%%)",
    )

    # Processing options
    proc_opts = parser.add_argument_group("Processing Options")
    proc_opts.add_argument(
        "-mask",
        metavar="FILE",
        help="Brain mask file (restricts analysis to brain voxels)",
    )

    proc_opts.add_argument(
        "-labels",
        nargs="+",
        metavar="LABEL",
        help="Condition labels (e.g., faces scenes objects). Default: cond1, cond2, ...",
    )

    proc_opts.add_argument(
        "-polort",
        type=str,
        default="A",
        metavar="N",
        help="Polynomial drift order for detrending. 'A' (default) = auto (AFNI formula: "
        "1 + floor(run_duration / 150)). Integer N for fixed order. -1 for none.",
    )

    proc_opts.add_argument(
        "-tr",
        type=float,
        metavar="SECONDS",
        help="Override TR from input files (seconds)",
    )

    proc_opts.add_argument(
        "-do_blur",
        "-do-blur",
        dest="do_blur",
        type=float,
        default=None,
        metavar="FWHM",
        help=(
            "3-D Gaussian spatial smoothing, FWHM in mm, applied per run BEFORE "
            "masking so edges do not bleed.  Typical values: 4-8 mm."
        ),
    )

    proc_opts.add_argument(
        "-do_scale",
        "-do-scale",
        dest="do_scale",
        action="store_true",
        help=(
            "Scale each voxel per run to mean=100, so the estimated HRF is in "
            "percent-signal-change units (values clipped at 200)."
        ),
    )

    add_load_threads_arg(proc_opts)
    add_trim_args(proc_opts)

    # External nuisance regressors
    nuis_opts = parser.add_argument_group(
        "External Nuisance Regressors",
        description=(
            "Motion, physio, or denoising components (ffs_denoise /\n"
            "ffs_denoisatorial PC timeseries).  Columns join the per-run\n"
            "polynomial block diagonal, so they stay run-specific."
        ),
    )
    add_ortvec_arguments(nuis_opts)

    # Output options
    out_opts = parser.add_argument_group("Output Options")
    out_opts.add_argument(
        "-save-betas",
        "-save_betas",
        dest="save_betas",
        action="store_true",
        help="Save beta coefficients as 4D NIfTI file",
    )

    out_opts.add_argument(
        "-save-r2",
        "-save_r2",
        dest="save_r2",
        action="store_true",
        help=(
            "Save the in-sample full-model R² map ({prefix}_r2).  Includes the "
            "polynomial / nuisance columns, so it is optimistic by construction "
            "and rises with every basis function added — use -save-xval-r2 to "
            "judge whether a window is actually earning its regressors."
        ),
    )

    out_opts.add_argument(
        "-save-xval-r2",
        "-save_xval_r2",
        dest="save_xval_r2",
        action="store_true",
        help=(
            "Save a cross-validated R² map ({prefix}_xval_r2): fit the shared "
            "HRF on the training runs, predict the held-out run, score the "
            "concatenated predictions.  Polynomials and -ortvec columns are "
            "projected out fold-locally (never from the full dataset), so the "
            "number is the honest referee for 'does this deconvolution "
            "generalise'.  See -cv_strategy / -cv_metric."
        ),
    )

    add_cv_strategy_arg(out_opts)
    add_cv_metric_arg(out_opts, dest="cv_metric", default="cod")

    out_opts.add_argument(
        "-save-design",
        action="store_true",
        help="Save design matrix as .1D file (AFNI format)",
    )

    out_opts.add_argument(
        "-save-design-plot",
        action="store_true",
        help="Save design matrix visualization (PNG image)",
    )

    add_verbose_arg(out_opts, default=0)

    # Hardware options
    hw_opts = parser.add_argument_group("Hardware Options")
    hw_opts.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (default: auto-detect GPU). Deprecated: use -device cpu.",
    )
    add_device_arg(hw_opts, extra="Overrides --cpu.")
    hw_opts.add_argument(
        "-debug-memory",
        action="store_true",
        help="Print VRAM usage vs. prediction after each chunk loop (for memory tuning)",
    )
    hw_opts.add_argument(
        "-debug-design",
        dest="debug_design",
        action="store_true",
        help=(
            "Before the GLM fit, print a design inspection: per-column "
            "L2 norms, near-zero / near-constant columns, X'X rank, "
            "condition number, and the null-space direction for any "
            "rank-deficient combination (so you can see WHICH columns "
            "are degenerate when the fit unexpectedly fails)."
        ),
    )

    return parser


def _labels_from_timing_files(timing_files: list[str]) -> list[str]:
    """Extract unique condition labels from timing filenames.

    Finds the parts of each stem that differ across files.
    E.g. ['onsets.localizer.times.bodies.txt', 'onsets.localizer.times.faces.txt']
    → ['bodies', 'faces']
    """
    stems = [Path(f).stem for f in timing_files]
    if len(stems) == 1:
        return [stems[0]]

    sep = "." if "." in stems[0] else "_"
    parts_list = [s.split(sep) for s in stems]
    min_len = min(len(p) for p in parts_list)

    # Find common prefix length
    common_prefix = 0
    for i in range(min_len):
        if len({p[i] for p in parts_list}) == 1:
            common_prefix += 1
        else:
            break

    # Find common suffix length
    common_suffix = 0
    for i in range(1, min_len - common_prefix + 1):
        if len({p[-i] for p in parts_list}) == 1:
            common_suffix += 1
        else:
            break

    labels = []
    for parts in parts_list:
        end = len(parts) - common_suffix if common_suffix > 0 else len(parts)
        unique = parts[common_prefix:end]
        labels.append(sep.join(unique) if unique else sep.join(parts))
    return labels


def parse_tent_windows(tent_window_args, n_conditions):
    """
    Parse tent_window arguments into per-condition (bot, top) tuples

    Supports multiple formats:
    - Two values: ['0', '15'] → single window for all conditions
    - Comma-separated pair: ['0,15'] → single window for all conditions
    - Multiple pairs: ['0,15', '0,20', '0,25'] → per-condition windows

    Parameters
    ----------
    tent_window_args : list
        Raw arguments from argparse
    n_conditions : int
        Number of conditions

    Returns
    -------
    windows : list of tuple
        [(bot1, top1), (bot2, top2), ...] for each condition

    Raises
    ------
    ValueError
        If format is invalid or number of windows doesn't match conditions
    """
    if tent_window_args is None:
        return None

    # Check if we have comma-separated pairs
    if all("," in arg for arg in tent_window_args):
        # Format: ['0,15', '0,20', ...] (per-condition)
        windows = []
        for arg in tent_window_args:
            parts = arg.split(",")
            if len(parts) != 2:
                raise ValueError(f"Invalid tent_window format: '{arg}'. Expected 'bot,top'")
            try:
                bot = float(parts[0])
                top = float(parts[1])
            except ValueError:
                raise ValueError(
                    f"Invalid tent_window values in '{arg}'. Expected numeric values."
                ) from None
            if bot >= top:
                raise ValueError(f"Invalid tent_window '{arg}': bot ({bot}) must be < top ({top})")
            windows.append((bot, top))

        # Check if we have one or n_conditions windows
        if len(windows) == 1:
            # Single window applies to all conditions
            return windows * n_conditions
        elif len(windows) == n_conditions:
            return windows
        else:
            raise ValueError(
                f"Number of tent_windows ({len(windows)}) must be 1 or match "
                f"number of conditions ({n_conditions})"
            )

    elif len(tent_window_args) == 2:
        # Format: ['0', '15'] (single window for all conditions)
        try:
            bot = float(tent_window_args[0])
            top = float(tent_window_args[1])
        except ValueError:
            raise ValueError("Invalid tent_window values. Expected numeric values.") from None
        if bot >= top:
            raise ValueError(f"Invalid tent_window: bot ({bot}) must be < top ({top})")
        return [(bot, top)] * n_conditions

    else:
        raise ValueError(
            "Invalid tent_window format. Use either: '0 15' (two values), "
            "'0,15' (comma-separated), or '0,15 0,20 ...' (per-condition)"
        )


def _fit_noise_gaussian(r2_vals: np.ndarray) -> tuple[float, float]:
    """
    Estimate the noise component of a LORO R² distribution.

    Noise voxels cluster around a low R² value (often slightly negative).
    We estimate the noise Gaussian by folding the left half of the distribution
    around its median — that left half is almost pure noise, and mirroring it
    gives a symmetric noise distribution to fit.

    Returns (mu_noise, sigma_noise).
    """
    mu = float(np.median(r2_vals))
    left = r2_vals[r2_vals <= mu]
    if len(left) < 10:
        return mu, float(np.std(r2_vals))
    # Mirror the left half around the median to get a symmetric noise estimate
    mirrored = np.concatenate([left, 2.0 * mu - left])
    return mu, float(np.std(mirrored))


def _loro_r2_per_voxel(
    data_clean: list[torch.Tensor],
    designs_clean: list[torch.Tensor],
    n_vox: int,
    device: torch.device,
    chunk: int = 20000,
) -> np.ndarray:
    """
    LORO CV: fit shared HRF on N-1 runs, predict on held-out run.
    Returns per-voxel median R² across folds, shape (n_vox,).
    """
    n_runs = len(data_clean)
    fold_r2s: list[np.ndarray] = []

    for held_out in range(n_runs):
        train_runs = [r for r in range(n_runs) if r != held_out]
        train_design = torch.cat([designs_clean[r] for r in train_runs], dim=0)
        test_design = designs_clean[held_out]

        # Pre-factor X'X once per fold (design is tiny: n_regs × n_regs)
        XtX = train_design.T @ train_design
        try:
            L_fold = torch.linalg.cholesky(XtX)
        except torch.linalg.LinAlgError:
            L_fold = None  # fall back to solve() below

        r2_parts: list[torch.Tensor] = []
        for i in range(0, n_vox, chunk):
            train_c = torch.cat([data_clean[r][i : i + chunk, :] for r in train_runs], dim=1).to(
                device
            )
            XtY = train_design.T @ train_c.T  # (n_regs, chunk)
            if L_fold is not None:
                betas = torch.cholesky_solve(XtY, L_fold)  # (n_regs, chunk)
            else:
                betas = torch.linalg.solve(XtX, XtY)
            test_c = data_clean[held_out][i : i + chunk, :].to(device)
            pred = (test_design @ betas).T
            r2_parts.append(compute_r2_metric(test_c, pred, "cod").cpu())

        fold_r2s.append(torch.cat(r2_parts).numpy())

    # Median across folds per voxel — robust to occasional bad folds
    return np.median(np.stack(fold_r2s, axis=0), axis=0)  # (n_vox,)


def _compute_loro_r2_matrix(
    data_list: list[torch.Tensor],
    onsets_per_condition: list,
    model: str,
    bot: float,
    candidate_tops: list[float],
    tr: float,
    n_conditions: int,
    polort: int,
    device: torch.device,
    verbose: bool,
    max_voxels: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute LORO-CV R² for each candidate window top, for each active voxel.

    Active voxels are those above the bottom-25% variance threshold of
    non-zero voxels (removes background / CSF rim). Polynomial drift is
    projected out via QR decomposition once, independent of window top.

    Parameters
    ----------
    max_voxels : int
        Hard cap on the number of active voxels (random subsample if exceeded).
        Use a large value (e.g. 500_000) to disable subsampling.

    Returns
    -------
    r2_matrix : ndarray, shape (n_candidates, n_vox), float32
        Median LORO R² across folds for each candidate window × active voxel.
    vox_idx : ndarray, shape (n_vox,), int
        Indices into the voxel axis of *data_list* (i.e. loaded/masked voxels,
        not full-volume indices — the caller maps them back).
    """
    n_tp_per_run = [d.shape[1] for d in data_list]
    n_voxels_total = data_list[0].shape[0]

    # ── Coarse background filter ─────────────────────────────────────────────
    rng = np.random.default_rng(42)
    var_proxy = data_list[0].var(dim=1).cpu().numpy()
    nonzero = var_proxy[var_proxy > 0]
    if len(nonzero) == 0:
        active_idx = np.arange(n_voxels_total)
    else:
        var_thresh = np.percentile(nonzero, 25)
        active_idx = np.where(var_proxy >= var_thresh)[0]

    if len(active_idx) > max_voxels:
        vox_idx = np.sort(rng.choice(active_idx, size=max_voxels, replace=False))
    else:
        vox_idx = active_idx
    n_vox = len(vox_idx)
    data_xval = [d[vox_idx, :] for d in data_list]

    if verbose:
        print(
            f"  Active voxels: {n_vox:,} "
            f"({len(active_idx):,} / {n_voxels_total:,} above variance floor)"
        )

    # ── Polynomial QR projection (once; window-independent) ─────────────────
    Q_per_run: list[torch.Tensor | None] = []
    data_clean: list[torch.Tensor] = []

    for run_idx, n_tp in enumerate(n_tp_per_run):
        data_r = data_xval[run_idx].to(device=device, dtype=torch.float32)
        if polort >= 0:
            poly_np = legendre_polynomials(n_tp, polort)
            poly_r = torch.tensor(poly_np, dtype=torch.float32, device=device)
            Q, _ = torch.linalg.qr(poly_r)
            Q_per_run.append(Q)
            data_r = data_r - (Q @ (Q.T @ data_r.T)).T
        else:
            Q_per_run.append(None)
        data_clean.append(data_r.cpu())

    use_csplin = model in ("CSPLIN", "CSPLINzero")
    zero_edges = model in ("TENTzero", "CSPLINzero")

    def _build_designs_clean(top_val: float) -> list[torch.Tensor]:
        """Build poly-projected TENT/CSPLIN designs for one window top."""
        out = []
        for run_idx, n_tp in enumerate(n_tp_per_run):
            cond_parts = []
            for c in range(n_conditions):
                fn = make_csplin_design if use_csplin else make_tent_design
                cond_parts.append(
                    fn(
                        [onsets_per_condition[c][run_idx]],
                        bot,
                        top_val,
                        tr,
                        n_tp,
                        zero_edges=zero_edges,
                        device=device,
                    )
                )
            design_r = torch.cat(cond_parts, dim=1)
            Q = Q_per_run[run_idx]
            if Q is not None:
                design_r = design_r - Q @ (Q.T @ design_r)
            out.append(design_r)
        return out

    # ── LORO R² for every candidate window ───────────────────────────────────
    r2_matrix = np.zeros((len(candidate_tops), n_vox), dtype=np.float32)
    for i, top_k in enumerate(
        tqdm(candidate_tops, desc="  Window candidates", disable=not verbose)
    ):
        designs_k = _build_designs_clean(top_k)
        r2_matrix[i] = _loro_r2_per_voxel(data_clean, designs_k, n_vox, device)

    return r2_matrix, vox_idx


def _xval_tent_top(
    data_list: list[torch.Tensor],
    onsets_per_condition: list,
    model: str,
    bot: float,
    nominal_top: float,
    candidate_tops: list[float],
    tr: float,
    n_conditions: int,
    polort: int,
    device: torch.device,
    verbose: bool,
) -> tuple[float, float]:
    """
    Select the best shared window top via LORO CV.

    Uses the nominal window to identify signal voxels (fold-over Gaussian model),
    then picks the candidate with the highest median R² on those signal voxels.

    Returns (best_top, best_median_r2_on_signal_voxels).
    """
    r2_matrix, vox_idx = _compute_loro_r2_matrix(
        data_list,
        onsets_per_condition,
        model,
        bot,
        candidate_tops,
        tr,
        n_conditions,
        polort,
        device,
        verbose,
        max_voxels=50_000,
    )
    n_vox = len(vox_idx)

    # Nominal window R² → noise model → signal voxels
    nom_idx = min(range(len(candidate_tops)), key=lambda i: abs(candidate_tops[i] - nominal_top))
    r2_nominal = r2_matrix[nom_idx]

    mu_noise, sigma_noise = _fit_noise_gaussian(r2_nominal)
    sig_thresh = mu_noise + sigma_noise
    signal_mask = r2_nominal > sig_thresh
    n_signal = int(signal_mask.sum())

    if n_signal < 200:
        cutoff = int(0.80 * n_vox)
        sig_order = np.argsort(r2_nominal)
        signal_mask = np.zeros(n_vox, dtype=bool)
        signal_mask[sig_order[cutoff:]] = True
        n_signal = int(signal_mask.sum())
        if verbose:
            print(
                f"  Noise model found < 200 signal voxels; "
                f"falling back to top-20% by R² ({n_signal:,} voxels)"
            )
    else:
        if verbose:
            print(
                f"  Noise model: μ={mu_noise:.3f}, σ={sigma_noise:.3f}, "
                f"threshold={sig_thresh:.3f} → {n_signal:,} signal voxels"
            )

    scores = np.array(
        [float(np.median(r2_matrix[i, signal_mask])) for i in range(len(candidate_tops))]
    )
    best_idx = int(np.argmax(scores))
    best_top = candidate_tops[best_idx]
    best_r2 = scores[best_idx]

    if verbose:
        print("  LORO R² on signal voxels (median) by window top:")
        cv_log = sorted(zip(candidate_tops, scores.tolist(), strict=False), key=lambda x: x[0])
        for top_k, r2_k in cv_log:
            marker = " ← selected" if abs(top_k - best_top) < 1e-6 else ""
            print(f"    {bot:.1f}–{top_k:.2f}s  R²={r2_k:.4f}{marker}")

    return best_top, best_r2


def _announce_written(paths: list[str] | str, elapsed: float, verb: int) -> None:
    """One line per file written: the name plus what the write cost.

    Every write here is wrapped in a ``spinner(leave=False)`` so a long save
    still shows motion; this is the completion notice the spinner defers to.
    Leaving the spinner's own line in place as well printed each file twice.
    """
    if verb < 1:
        return
    for p in [paths] if isinstance(paths, str) else paths:
        print(f"  ✓ {p}  ({elapsed:.1f}s)")


def _compute_xval_r2_map(
    *,
    packed,
    run_starts: list[int],
    n_runs: int,
    cv_strategy: str,
    metric: str,
    device: torch.device,
    verbose: bool,
) -> np.ndarray | None:
    """Cross-validated R² per voxel for the packed shared-task design.

    Delegates to :func:`glm.xval.compute_xval_r2`, which is the one place that
    knows how to split by run, project the nuisance **fold-locally**, and score
    the concatenated held-out predictions. Projecting polynomials from the full
    dataset before splitting is the classic way to leak training variance into
    the held-out fold ([[LORO cross-validation]]) — hence the delegation rather
    than a second implementation here.

    Returns ``None`` when there are too few runs to split.
    """
    from fastfuncstuff.glm.xval import compute_xval_r2, generate_cv_splits

    if n_runs < 2:
        return None

    n_task = packed.n_task_cols
    n_cols = packed.design_concat.shape[1]
    cv_splits = generate_cv_splits(n_runs=n_runs, strategy=parse_cv_strategy(cv_strategy))
    if not cv_splits:
        return None

    if verbose:
        print(f"\nCross-validated R² ({cv_strategy}, metric={metric})...")

    result = compute_xval_r2(
        data=packed.data_concat,
        design_matrix=packed.design_concat,
        run_starts=list(run_starts),
        # The packed layout is task-first, so the nuisance is simply the tail.
        # Block-diagonal columns go all-zero once a run is held out; the
        # projector drops them rather than inverting a singular matrix.
        stim_indices=list(range(n_task)),
        nuisance_indices=list(range(n_task, n_cols)),
        cv_splits=cv_splits,
        metric=metric,
        device=device,
        verbose=verbose,
    )
    r2 = result["r2"]
    assert isinstance(r2, torch.Tensor)
    return r2.cpu().numpy().astype(np.float32)


def main():
    """Main CLI entry point"""
    parser = parse_args()
    args = parser.parse_args()

    pfx = parse_prefix(args.prefix)
    args.prefix = pfx.stem  # overwrite with clean stem
    Path(args.prefix).parent.mkdir(parents=True, exist_ok=True)
    _nii_ext = pfx.nifti_ext

    # Setup device — -device takes precedence over legacy --cpu flag
    device_spec = args.device or ("cpu" if args.cpu else None)
    device = setup_device(device_spec)
    if args.verb >= 1:
        print(f"Using device: {device}")

    # Validate design source
    input_files = parse_input_files(args.input)
    n_runs = len(input_files)
    if not args.onsets and not args.events:
        print("ERROR: Must specify -onsets or -events", file=sys.stderr)
        return 1
    if args.onsets and args.events:
        print("ERROR: -onsets and -events are mutually exclusive", file=sys.stderr)
        return 1
    if (args.event_ignore or args.event_cols) and not args.events:
        print("ERROR: -event-ignore and -event-cols require -events", file=sys.stderr)
        return 1

    if args.verb >= 1:
        print_cli_header("ffs_deconvolve", "FIR / TENT / CSPLIN deconvolution")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Compute device: {device}")

    # ── Event timing ─────────────────────────────────────────────────────────
    # One parser for both BIDS -events TSVs and AFNI -onsets timing files, so
    # the one-TSV-broadcast convention and the -input pairing check behave the
    # same here as in every other GLM tool.  Parsed before the load because the
    # condition list validates -window / -add-lag / -labels counts.
    try:
        timing = parse_timing_spec(
            events=args.events,
            onsets=args.onsets,
            durations_arg=args.durations,
            n_runs=n_runs,
            event_ignore=args.event_ignore,
            event_cols=tuple(args.event_cols) if args.event_cols else None,
            round_durations=args.round_durations,
            input_files=input_files,
            verbose=args.verb >= 1,
            allow_missing_durations=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    n_conditions = timing.n_conditions
    # Stimulus durations drive auto-window estimation; None means "not known",
    # in which case -window / -duration must supply the window.
    condition_durations: list[float] | None = timing.durations if timing.durations_given else None

    # User-supplied -labels override BIDS trial_type / timing-file names
    if args.labels:
        if len(args.labels) != n_conditions:
            print(
                f"ERROR: -labels count ({len(args.labels)}) does not match "
                f"number of conditions ({n_conditions}: {timing.condition_labels})",
                file=sys.stderr,
            )
            return 1
        condition_labels = list(args.labels)
    else:
        condition_labels = list(timing.condition_labels)

    preflight_check(
        input_files=input_files,
        onset_files=args.onsets,
        ortvec_files=None,
    )

    # ── Load data ────────────────────────────────────────────────────────────
    # Shared loader: threaded decode (-load_threads), optional per-run blur
    # (-do_blur) before masking, percent-signal scaling (-do_scale) and
    # -drop_first/-drop_last trimming all happen here rather than in a
    # deconvolve-only copy of the load path.
    load_result = load_and_preprocess_runs(
        input_files=input_files,
        tr=args.tr,
        mask_file=args.mask,
        blur_fwhm=args.do_blur,
        do_scale=args.do_scale,
        device=device,
        force_cpu=True,  # per-run views are sliced on CPU, streamed to GPU by fit_glm
        verbose=args.verb >= 1,
        load_threads=args.load_threads,
        drop_first=args.drop_first,
        drop_last=args.drop_last,
    )

    data = load_result.data  # (n_voxels_loaded, n_timepoints), CPU float32
    run_starts = list(load_result.run_starts)
    n_timepoints = load_result.n_timepoints
    tr = load_result.tr
    mask = load_result.mask
    nx, ny, nz = load_result.volume_shape
    n_timepoints_per_run = run_lengths_from_starts(run_starts, n_timepoints)
    n_voxels_loaded = data.shape[0]

    # Timing describes the file on disk; shift it onto the retained window
    # before -round-onsets or any window estimation touches the onsets.
    trim = trim_spec_from_args(args, tr=tr)
    apply_trim_to_timing(
        timing,
        trim,
        run_lengths_tr=n_timepoints_per_run,
        n_runs=n_runs,
        verbose=args.verb >= 1,
    )
    onsets_per_condition = timing.all_onsets

    if args.verb >= 1:
        print(f"  Total timepoints: {n_timepoints} across {n_runs} runs")

    # Resolve polort: "A" → AFNI auto formula based on run duration
    polort_str = str(args.polort).strip().upper()
    if polort_str == "A":
        run_duration_sec = min(n_timepoints_per_run) * tr
        args.polort = auto_polort(run_duration_sec, formula="afni")
        if args.verb >= 1:
            print(f"  Polort: A → {args.polort} (run duration {run_duration_sec:.1f}s)")
    else:
        try:
            args.polort = int(polort_str)
        except ValueError:
            print(
                f"ERROR: -polort must be 'A' or an integer, got: {args.polort!r}", file=sys.stderr
            )
            return 1

    # ── External nuisance (-ortvec family) ───────────────────────────────────
    # Kept per-run on the block diagonal alongside the polynomials: components
    # estimated on one run do not describe the others ([[Block-diagonal
    # nuisance]]).
    try:
        nuisance_blocks = collect_nuisance_blocks(
            args,
            run_starts=run_starts,
            n_timepoints=n_timepoints,
            verbose=args.verb >= 1,
            trim=trim,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    extra_regs_per_run: list[torch.Tensor] | None = None
    if nuisance_blocks:
        extra_regs_per_run = append_nuisance_blocks(
            [torch.zeros((n, 0), dtype=torch.float32) for n in n_timepoints_per_run],
            nuisance_blocks,
            run_starts,
            n_timepoints,
        )
        if args.verb >= 1:
            n_extra = extra_regs_per_run[0].shape[1]
            print(
                f"  External nuisance: {n_extra} column(s) per run "
                f"({n_extra * n_runs} total, block-diagonal)"
            )

    # Per-run views of the loaded (already masked) data.  Slices are views, so
    # this costs nothing until pack_for_shared_task_glm concatenates.
    data_list = [data[:, s : s + n] for s, n in zip(run_starts, n_timepoints_per_run, strict=True)]

    def _to_volume(masked: np.ndarray) -> np.ndarray:
        """Place (n_voxels_loaded, ...) values back into the full 3-D volume."""
        out_shape = (nx, ny, nz) + tuple(masked.shape[1:])
        if mask is not None:
            out = np.zeros(out_shape, dtype=np.float32)
            out[mask, ...] = masked
            return out
        return np.ascontiguousarray(masked, dtype=np.float32).reshape(out_shape)

    # ── Optional onset / duration rounding ──────────────────────────────────
    if args.round_onsets is not None:
        from fastfuncstuff.design.builder import round_onsets

        onsets_per_condition = round_onsets(onsets_per_condition, tr, threshold=args.round_onsets)
        if args.verb >= 1:
            print(f"\nOnsets rounded to TR boundaries (threshold={args.round_onsets:.2f})")

    # For BIDS, round_durations was already applied per-event inside parse_bids_events
    if args.round_durations is not None and condition_durations is not None and not args.events:
        dp = args.round_durations
        condition_durations = [round(d, dp) for d in condition_durations]
        if args.verb >= 1:
            print(
                f"Durations rounded to {dp} decimal place(s): "
                f"{[f'{d:.{dp}f}' for d in condition_durations]}"
            )

    # Flatten all onset times for TR-locking check
    # onsets_per_condition is always set by this point: either from the BIDS
    # branch (line ~896) or the AFNI timing-file loop (line ~1027) above.
    assert onsets_per_condition is not None
    all_onset_times = []
    for cond_onsets in onsets_per_condition:
        for run_onsets in cond_onsets:
            all_onset_times.extend(run_onsets.tolist())

    # Determine model type
    if args.model == "AUTO":
        is_locked = is_tr_locked(all_onset_times, tr, threshold=args.tr_lock_threshold)
        if is_locked:
            model = "FIR"
            if args.verb >= 1:
                print(f"\n✓ Onsets are TR-locked (threshold: {args.tr_lock_threshold * 100:.0f}%)")
                print("  Using FIR model")
        else:
            model = "TENT"
            if args.verb >= 1:
                print(
                    f"\n✗ Onsets are NOT TR-locked (threshold: {args.tr_lock_threshold * 100:.0f}%)"
                )
                print("  Using TENT model")
    else:
        model = args.model
        if args.verb >= 1:
            print(f"\nUsing {model} model")

    # ── FLOBS branch ─────────────────────────────────────────────────────────
    # **DEPRECATED 2026-05-17**: -model FLOBS in ffs_deconvolve is
    # kept working for backwards compatibility, but the canonical
    # home for basis-set / FLOBS fits is now ``ffs_fitbasis``, which
    # additionally supports SPMG1/SPMG2/SPMG3 bases AND single-trial
    # mode (-single-trials) with the same shape prior.  Pipe your
    # arguments to ``ffs_fitbasis -model FLOBS`` and pass any extra
    # SPMG/regularisation flags.  This branch will be removed in a
    # later release.
    if model == "FLOBS":
        import warnings as _warnings

        _warnings.warn(
            "ffs_deconvolve -model FLOBS is deprecated; use ffs_fitbasis "
            "(same FLOBS basis + constrained fit, plus SPMG1/2/3 bases and "
            "-single-trials mode).  ffs_deconvolve will continue to host "
            "FIR/TENT/TENTzero/CSPLIN/CSPLINzero (non-parametric "
            "deconvolution).  See [[Outstanding issues]] for the rationale.",
            DeprecationWarning,
            stacklevel=2,
        )
        if args.verb >= 1:
            print(
                "\n  ⚠️  -model FLOBS in ffs_deconvolve is deprecated.\n"
                "      Prefer: ffs_fitbasis -model FLOBS [-single-trials] …\n"
                "      ffs_fitbasis adds SPMG1/2/3 bases and per-trial fits.\n"
            )
        # NOTE: ``save_nifti`` and ``save_iresp`` are already imported
        # at module scope (lines 65–80) — importing them again HERE
        # would create local-only bindings that shadow the module
        # globals for the *entire* main() function (Python's lexical
        # scope), breaking the FIR/TENT save paths below.  So just
        # import the FLOBS-specific helpers here.
        from fastfuncstuff.design.flobs import (
            fit_flobs_constrained,
            generate_flobs_basis,
        )
        from fastfuncstuff.design.hrf_derive import build_pc_basis_design_per_run

        if args.verb >= 1:
            print(
                f"\nFLOBS basis: K={args.flobs_n_basis} from "
                f"{args.flobs_n_samples} half-cosine samples "
                f"(window {args.flobs_window:.1f}s, dt {args.flobs_dt:.2f}s)"
            )
        basis = generate_flobs_basis(
            n_basis=args.flobs_n_basis,
            n_samples=args.flobs_n_samples,
            duration=args.flobs_window,
            dt=args.flobs_dt,
            seed=args.flobs_seed,
        )
        ev_frac = basis.eigenvalues**2
        ev_frac = ev_frac / max(ev_frac.sum(), 1e-30)
        if args.verb >= 1:
            print(
                f"  Variance explained by top {args.flobs_n_basis}: "
                f"{ev_frac[: args.flobs_n_basis].sum() * 100:.1f}%  "
                f"(PC1 alone {ev_frac[0] * 100:.1f}%)"
            )
            print(f"  Prior MVN(m, C):  m = {basis.m}, σ_diag = {np.sqrt(np.diag(basis.C))}")

        # Per-run task designs: each condition convolved with each
        # basis function.  ``build_pc_basis_design_per_run`` is what
        # ffs_librarian uses for its NSD refit step — same shape as
        # what we want here (one block of K cols per condition).
        n_tp_per_run_list = list(n_timepoints_per_run)
        basis_lag_times = np.arange(basis.basis_functions.shape[1]) * basis.dt
        per_run_designs_per_cond: list[list[np.ndarray]] = []
        for cond_idx in range(n_conditions):
            cond_onsets_per_run = [
                onsets_per_condition[cond_idx][r] for r in range(len(n_tp_per_run_list))
            ]
            cond_designs = build_pc_basis_design_per_run(
                onsets_per_run=cond_onsets_per_run,
                pcs=basis.basis_functions,  # use basis as PCs
                lag_times=basis_lag_times,
                tr=tr,
                n_timepoints_per_run=n_tp_per_run_list,
                basis="FIR" if model == "FIR" else "TENT",
            )
            per_run_designs_per_cond.append(cond_designs)

        # Horizontally concat conditions per run → per-run design
        # (n_tp_run, n_conditions * n_basis) with condition-major order.
        per_run_task_designs = []
        for r in range(len(n_tp_per_run_list)):
            cond_blocks = [per_run_designs_per_cond[c][r] for c in range(n_conditions)]
            per_run_task_designs.append(
                torch.from_numpy(np.concatenate(cond_blocks, axis=1).astype(np.float32))
            )

        # Per-run data list (the loader already applied -mask)
        per_run_data = data_list

        # Canonical shared-task multi-run GLM packing (same helper the
        # FIR/TENT path uses) — task block shared across runs,
        # polynomials block-diagonal per run.
        packed = pack_for_shared_task_glm(
            per_run_data=per_run_data,
            per_run_task_designs=per_run_task_designs,
            polort=args.polort,
            task_column_labels=[
                f"{condition_labels[c]}#PC{b}"
                for c in range(n_conditions)
                for b in range(args.flobs_n_basis)
            ],
            extra_regressors_per_run=extra_regs_per_run,
            drop_empty_nuisance=True,
            device=torch.device("cpu"),
        )
        if args.verb >= 1:
            print(
                f"  Design: {packed.design_concat.shape}  "
                f"({packed.n_task_cols} task + "
                f"{packed.design_concat.shape[1] - packed.n_task_cols} nuisance)"
            )

        # Slice task vs nuisance; fit_flobs_constrained applies the
        # FLOBS prior ONLY to the task block (nuisance is unpenalized).
        task_design = packed.design_concat[:, : packed.n_task_cols]
        nuisance = (
            packed.design_concat[:, packed.n_task_cols :]
            if packed.design_concat.shape[1] > packed.n_task_cols
            else None
        )

        # Parse prior weight
        if str(args.flobs_prior_weight).strip().lower() == "auto":
            pw: float | str = "auto"
        else:
            pw = float(args.flobs_prior_weight)

        if args.verb >= 1:
            print(f"  Fitting (prior_weight={pw!r}) …")
        fit = fit_flobs_constrained(
            data=packed.data_concat,
            design_task=task_design,
            basis=basis,
            n_conditions=n_conditions,
            nuisance=nuisance,
            prior_weight=pw,
            device=device,
        )
        if args.verb >= 1:
            print(f"  ✓ Fit complete.  Mean R² = {fit.r2.mean():.3f}")

        # ── Reshape to volume space ──────────────────────────────────
        # fit.betas       : (n_voxels_masked, n_total_cols)  CONSTRAINED
        # fit.hrfs        : (n_voxels_masked, n_conditions, n_t_basis)  CONSTRAINED
        # fit.r2          : (n_voxels_masked,)  CONSTRAINED
        # fit.betas_ols   : same shape as fit.betas, UNCONSTRAINED (OLS)
        # fit.hrfs_ols    : same shape as fit.hrfs,  UNCONSTRAINED
        # fit.r2_ols      : same shape as fit.r2,    UNCONSTRAINED
        # Save BOTH so the user can SEE where the FLOBS prior reshapes
        # the fit — critical for validation ("does the constraint help
        # or just hide what's there?").
        task_betas = fit.betas[:, : packed.n_task_cols].reshape(
            -1, n_conditions, args.flobs_n_basis
        )
        task_betas_ols = fit.betas_ols[:, : packed.n_task_cols].reshape(
            -1, n_conditions, args.flobs_n_basis
        )

        # Amplitude per (voxel, condition): peak of reconstructed HRF.
        # Most directly interpretable for 2nd-level analyses (it's the
        # "signal change" of the modelled response).  Computed for
        # both constrained and unconstrained fits.
        amplitude = fit.hrfs.max(axis=2)  # (n_vox, n_cond)
        amplitude_ols = fit.hrfs_ols.max(axis=2)  # (n_vox, n_cond)

        if args.verb >= 1:
            print(
                f"  σ² mean = {fit.sigma2_mean:.4g}, "
                f"effective prior weight = {fit.effective_prior_weight:.4g}"
            )
            print(f"  R² mean — OLS: {fit.r2_ols.mean():.3f}   constrained: {fit.r2.mean():.3f}")

        # ── Save FLOBS basis (one TSV, shared across conditions) ────
        basis_path = f"{args.prefix}_flobs_basis.tsv"
        np.savetxt(
            basis_path,
            basis.basis_functions.T,
            fmt="%.10g",
            delimiter="\t",
        )
        if args.verb >= 1:
            print(f"  Wrote {basis_path}  (n_lags×K, dt={basis.dt}s)")

        # ── Save R² volumes ─────────────────────────────────────────
        # Two maps: constrained (the published one) and unconstrained
        # (so the user can see *where the prior changed the fit*).
        for r2_arr, suffix in (
            (fit.r2, ""),
            (fit.r2_ols, "_unconstrained"),
        ):
            r2_path = f"{args.prefix}_flobs_r2{suffix}{_nii_ext}"
            t_write = time.perf_counter()
            with spinner(f"Writing {Path(r2_path).name}", enabled=args.verb >= 1, leave=False):
                save_nifti(
                    _to_volume(r2_arr[:, None]).squeeze(-1),
                    output_path=r2_path,
                    reference_img=input_files[0],
                )
            _announce_written(r2_path, time.perf_counter() - t_write, args.verb)

        # ── Per-condition iresp (reconstructed HRF) — BOTH fits ────
        if args.flobs_save_iresp:
            # Constrained: the "shipping" library.  Unconstrained: the
            # comparison artefact.  Filename convention:
            # <prefix>_flobs_iresp_<cond>.nii.gz                    (constrained)
            # <prefix>_flobs_iresp_<cond>_unconstrained.nii.gz       (OLS)
            for hrfs_arr, fit_suffix in (
                (fit.hrfs, ""),
                (fit.hrfs_ols, "_unconstrained"),
            ):
                iresp_vol = _to_volume(hrfs_arr)
                t_write = time.perf_counter()
                with spinner(
                    f"Writing FLOBS iresp{fit_suffix or ' (constrained)'}",
                    enabled=args.verb >= 1,
                    leave=False,
                ):
                    iresp_files = save_iresp(
                        iresp=iresp_vol,
                        output_prefix=f"{args.prefix}_flobs",
                        condition_labels=[f"{lbl}{fit_suffix}" for lbl in condition_labels],
                        tr=basis.dt,
                        bot=0.0,
                        top=basis.duration - basis.dt,
                        reference_img=input_files[0],
                        nii_ext=_nii_ext,
                    )
                _announce_written(iresp_files, time.perf_counter() - t_write, args.verb)

        # ── Per-condition PC weights + amplitude — BOTH fits ───────
        for cond_idx, label in enumerate(condition_labels):
            for tbetas, amps, suffix in (
                (task_betas, amplitude, ""),
                (task_betas_ols, amplitude_ols, "_unconstrained"),
            ):
                weights_4d = _to_volume(tbetas[:, cond_idx, :])
                weights_path = f"{args.prefix}_flobs_pcweights_{label}{suffix}{_nii_ext}"
                amp_3d = _to_volume(amps[:, cond_idx][:, None]).squeeze(-1)
                amp_path = f"{args.prefix}_flobs_amplitude_{label}{suffix}{_nii_ext}"
                t_write = time.perf_counter()
                with spinner(
                    f"Writing FLOBS maps: {label}{suffix}", enabled=args.verb >= 1, leave=False
                ):
                    save_nifti(weights_4d, output_path=weights_path, reference_img=input_files[0])
                    save_nifti(amp_3d, output_path=amp_path, reference_img=input_files[0])
                _announce_written(
                    [weights_path, amp_path], time.perf_counter() - t_write, args.verb
                )

        if args.verb >= 1:
            print(f"\n{'=' * 70}")
            print("✓ FLOBS deconvolution complete!")
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'=' * 70}")
        return 0

    # ── Unified window determination (FIR and TENT family) ───────────────────
    # Resolve add_lag to a per-condition list
    add_lag_raw = args.add_lag  # list[int] | None
    if add_lag_raw is not None:
        if len(add_lag_raw) == 1:
            add_lag_list = add_lag_raw * n_conditions
        elif len(add_lag_raw) == n_conditions:
            add_lag_list = add_lag_raw
        else:
            print(
                f"ERROR: -add-lag must have 1 or {n_conditions} values (got {len(add_lag_raw)})",
                file=sys.stderr,
            )
            return 1
    else:
        add_lag_list = [0] * n_conditions

    def _apply_add_lag(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Adjust per-condition (bot, top) windows by add_lag (TR units)."""
        result = []
        for (bot, top), lag in zip(windows, add_lag_list, strict=False):
            n_trs = max(1, round(top / tr) + lag)
            result.append((bot, float(n_trs) * tr))
        return result

    # Priority: 1) explicit -window, 2) auto from condition_durations, 3) -duration fallback
    window_source: str
    if args.window is not None:
        try:
            tent_windows = parse_tent_windows(args.window, n_conditions)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        tent_windows = _apply_add_lag(tent_windows)
        window_source = "explicit -window"

    elif condition_durations is not None:
        from fastfuncstuff.design.hrf import compute_windows_from_durations

        tent_windows = compute_windows_from_durations(condition_durations, tr, add_lag=add_lag_list)
        window_source = "auto (HRF convolution)"

    elif args.duration is not None:
        tent_windows = _apply_add_lag([(0.0, args.duration)] * n_conditions)
        window_source = f"legacy -duration ({args.duration}s)"

    else:
        print(
            f"ERROR: No window specified for {model} model.\n"
            "  Use -window (explicit), -durations (auto-estimate per condition),\n"
            "  -events (BIDS, durations from TSV), or -duration (single legacy value).",
            file=sys.stderr,
        )
        return 1

    # For FIR: n_lags per condition = round(top / tr)
    n_lags_per_cond: list[int] = [max(1, round(top / tr)) for _, top in tent_windows]

    # Verbose window summary
    if args.verb >= 1:
        print(f"\nHRF window source: {window_source}")
        if model == "FIR":
            if len(set(n_lags_per_cond)) == 1:
                print(
                    f"  Window (all conditions): {n_lags_per_cond[0]} TRs "
                    f"({tent_windows[0][1]:.1f}s)"
                )
            else:
                print("  Windows (per condition):")
                for lbl, n_l, (_, top) in zip(
                    condition_labels, n_lags_per_cond, tent_windows, strict=False
                ):
                    print(f"    {lbl}: {n_l} TRs ({top:.1f}s)")
        else:
            basis_type = "cubic spline" if "CSPLIN" in model else "tent"
            if len(set(tent_windows)) == 1:
                bot, top = tent_windows[0]
                n_basis_calc = (
                    args.tent_n_basis if args.tent_n_basis else round((top - bot) / tr) + 1
                )
                n_actual = n_basis_calc - 2 if model in ("TENTzero", "CSPLINzero") else n_basis_calc
                print(
                    f"  Window (all conditions): {bot}s–{top}s → {n_basis_calc} {basis_type} knots, {n_actual} regressors"
                )
            else:
                print("  Windows (per condition):")
                for lbl, (bot, top) in zip(condition_labels, tent_windows, strict=False):
                    n_basis_calc = (
                        args.tent_n_basis if args.tent_n_basis else round((top - bot) / tr) + 1
                    )
                    n_actual = (
                        n_basis_calc - 2 if model in ("TENTzero", "CSPLINzero") else n_basis_calc
                    )
                    print(
                        f"    {lbl}: {bot}s–{top}s → {n_basis_calc} {basis_type} knots, {n_actual} regressors"
                    )

    # Cross-validate TENT window upper bound if requested
    # _pv_* variables carry per-voxel mode context into the block below.
    _do_per_voxel = False
    _pv_bot: float = 0.0
    _pv_nom_top: float = 0.0
    _pv_tops: list[float] = []

    if args.xval_tr_range > 0 and model in ("TENT", "TENTzero", "CSPLIN", "CSPLINzero"):
        if n_runs < 2:
            print("WARNING: -xval-tr-range requires ≥2 runs. Skipping.", file=sys.stderr)
        elif tent_windows is None:
            print("WARNING: tent_windows not set. Skipping xval.", file=sys.stderr)
        else:
            unique_tops = {w[1] for w in tent_windows}
            if len(unique_tops) > 1:
                print(
                    "WARNING: -xval-tr-range only supported when all conditions share "
                    "the same window top. Skipping.",
                    file=sys.stderr,
                )
            else:
                nom_top = next(iter(unique_tops))
                nom_bot = tent_windows[0][0]

                # Candidate tops: nom_top ± xval_tr_range TRs, step 1 TR
                n_range = args.xval_tr_range
                candidate_tops = [
                    round(nom_top + k * tr, 6)
                    for k in range(-n_range, n_range + 1)
                    if nom_top + k * tr > nom_bot + tr  # needs at least 2 knots
                ]

                if len(candidate_tops) < 2:
                    print(
                        "WARNING: fewer than 2 valid candidate tops; skipping xval.",
                        file=sys.stderr,
                    )
                elif args.per_voxel:
                    # Per-voxel mode: defer R² computation; keep nominal tent_windows for now
                    _do_per_voxel = True
                    _pv_bot = nom_bot
                    _pv_nom_top = nom_top
                    _pv_tops = candidate_tops
                    if args.verb >= 1:
                        print(f"\nPer-voxel window mode (±{n_range} TRs)...")
                        print(f"  Candidates: {', '.join(f'{t:.2f}s' for t in candidate_tops)}")
                else:
                    if args.verb >= 1:
                        print(f"\nCross-validating window top (±{n_range} TRs)...")
                        print(f"  Candidates: {', '.join(f'{t:.2f}s' for t in candidate_tops)}")

                    best_top, best_r2 = _xval_tent_top(
                        data_list=data_list,
                        onsets_per_condition=onsets_per_condition,
                        model=model,
                        bot=nom_bot,
                        nominal_top=nom_top,
                        candidate_tops=candidate_tops,
                        tr=tr,
                        n_conditions=n_conditions,
                        polort=args.polort,
                        device=device,
                        verbose=args.verb >= 1,
                    )

                    tent_windows = [(nom_bot, best_top)] * n_conditions

                    if args.verb >= 1:
                        print(
                            f"  → Selected: {nom_bot:.1f}s to {best_top:.2f}s  "
                            f"(LORO R²={best_r2:.4f})"
                        )
                    else:
                        print(
                            f"xval: selected window {nom_bot:.1f}s to {best_top:.2f}s  "
                            f"(LORO R²={best_r2:.4f})"
                        )

    # ── Per-voxel window selection ────────────────────────────────────────────
    if _do_per_voxel:
        zero_edges_pv = model in ("TENTzero", "CSPLINzero")
        use_csplin_pv = model in ("CSPLIN", "CSPLINzero")

        # 1. LORO R² for every candidate window, all active voxels (no subsampling cap)
        if args.verb >= 1:
            print("\nComputing per-voxel LORO R² across all candidate windows...")
        r2_matrix, vox_idx = _compute_loro_r2_matrix(
            data_list=data_list,
            onsets_per_condition=onsets_per_condition,
            model=model,
            bot=_pv_bot,
            candidate_tops=_pv_tops,
            tr=tr,
            n_conditions=n_conditions,
            polort=args.polort,
            device=device,
            verbose=args.verb >= 1,
            max_voxels=500_000,
        )
        # r2_matrix: (n_candidates, n_vox); vox_idx indexes the LOADED voxel axis

        # 2. Per-voxel argmax → best candidate index → best top value
        best_cand_per_vox = np.argmax(r2_matrix, axis=0)  # (n_vox,)
        best_top_per_vox = np.array([_pv_tops[i] for i in best_cand_per_vox], dtype=np.float32)

        # 3. Save r2_by_window map (4D: one volume per candidate)
        r2_bw = np.zeros((n_voxels_loaded, len(_pv_tops)), dtype=np.float32)
        r2_bw[vox_idx] = r2_matrix.T
        r2_bw_file = f"{args.prefix}_r2_by_window{_nii_ext}"
        t_write = time.perf_counter()
        with spinner(f"Writing {Path(r2_bw_file).name}", enabled=args.verb >= 1, leave=False):
            save_nifti(_to_volume(r2_bw), r2_bw_file, reference_img=input_files[0])
        _announce_written(r2_bw_file, time.perf_counter() - t_write, args.verb)
        del r2_bw

        # 4. Save windowsize map (3D: winning top in seconds)
        ws = np.zeros((n_voxels_loaded, 1), dtype=np.float32)
        ws[vox_idx, 0] = best_top_per_vox
        ws_file = f"{args.prefix}_windowsize{_nii_ext}"
        t_write = time.perf_counter()
        with spinner(f"Writing {Path(ws_file).name}", enabled=args.verb >= 1, leave=False):
            save_nifti(_to_volume(ws).squeeze(-1), ws_file, reference_img=input_files[0])
        _announce_written(ws_file, time.perf_counter() - t_write, args.verb)
        del ws

        # 4b. The winning candidate's LORO R² IS a cross-validated R² map, so
        # -save-xval-r2 costs nothing extra here. It is optimistically biased
        # (the window was chosen per voxel on this same CV score) — a nested
        # split would be needed to remove that, which this mode does not do.
        if args.save_xval_r2:
            xr2 = np.zeros((n_voxels_loaded, 1), dtype=np.float32)
            xr2[vox_idx, 0] = r2_matrix[best_cand_per_vox, np.arange(len(vox_idx))]
            xr2_file = f"{args.prefix}_xval_r2{_nii_ext}"
            t_write = time.perf_counter()
            with spinner(f"Writing {Path(xr2_file).name}", enabled=args.verb >= 1, leave=False):
                save_nifti(_to_volume(xr2).squeeze(-1), xr2_file, reference_img=input_files[0])
            _announce_written(xr2_file, time.perf_counter() - t_write, args.verb)
            del xr2
            if args.verb >= 1:
                print("      R² of each voxel's winning window")

        # 5. Output dimensions: max window → max_n_knots timepoints per condition
        max_top_pv = max(_pv_tops)
        max_n_knots_pv = round((max_top_pv - _pv_bot) / tr) + 1
        max_n_basis_out = max_n_knots_pv  # includes zero-padded edges for *zero models

        if args.verb >= 1:
            print(
                f"  Output: {max_n_basis_out} timepoints per condition "
                f"(max window {_pv_bot:.1f}–{max_top_pv:.2f}s)"
            )

        # 6. The loaded data, already concatenated across runs by the loader.
        data_tensor_pv = data

        # Map voxels not in vox_idx (below the variance floor) to the nominal
        # top as fallback
        fallback_top = min(_pv_tops, key=lambda t: abs(t - _pv_nom_top))
        full_best_top = np.full(n_voxels_loaded, fallback_top, dtype=np.float32)
        full_best_top[vox_idx] = best_top_per_vox

        # 7. Assemble per-voxel betas: one (n_voxels_loaded, max_n_basis_out) per condition
        assembled_betas = [
            np.zeros((n_voxels_loaded, max_n_basis_out), dtype=np.float32)
            for _ in range(n_conditions)
        ]

        unique_tops_pv = sorted(set(full_best_top.tolist()))
        if args.verb >= 1:
            print(f"\n  Fitting GLMs for {len(unique_tops_pv)} unique winning windows...")

        def _per_run_designs_for_top(top_val: float) -> list[torch.Tensor]:
            """Per-run task-only designs for one candidate window top."""
            out = []
            for run_idx, n_tp in enumerate(n_timepoints_per_run):
                cond_parts = []
                fn = make_csplin_design if use_csplin_pv else make_tent_design
                for cond_idx in range(n_conditions):
                    cond_parts.append(
                        fn(
                            onset_times_list=[onsets_per_condition[cond_idx][run_idx]],
                            bot=_pv_bot,
                            top=top_val,
                            tr=tr,
                            n_timepoints=n_tp,
                            n_basis=args.tent_n_basis,
                            zero_edges=zero_edges_pv,
                            device=torch.device("cpu"),
                        )
                    )
                out.append(torch.cat(cond_parts, dim=1))
            return out

        for top_k in tqdm(unique_tops_pv, desc="  Fitting", disable=not args.verb >= 1):
            vox_this_top = np.where(full_best_top == top_k)[0]
            n_knots_k = round((top_k - _pv_bot) / tr) + 1
            n_regs_k = n_knots_k - 2 if zero_edges_pv else n_knots_k

            data_sub = data_tensor_pv[vox_this_top, :]
            # Same canonical packing as the single-window path: shared task
            # block, per-run polynomials + external nuisance block-diagonal.
            packed_k = pack_for_shared_task_glm(
                per_run_data=[
                    data_sub[:, st : st + n]
                    for st, n in zip(run_starts, n_timepoints_per_run, strict=True)
                ],
                per_run_task_designs=_per_run_designs_for_top(top_k),
                polort=args.polort,
                extra_regressors_per_run=extra_regs_per_run,
                drop_empty_nuisance=True,
                device=torch.device("cpu"),
            )
            results_k = fit_glm(
                data=packed_k.data_concat,
                design=packed_k.design_concat,
                tr=tr,
                max_poly_degree=-1,
                device=device,
                preload_data_to_device=False,
                chunk_size=None,  # auto-estimate via memory module
                verbose=False,
                debug_memory=args.debug_memory,
            )
            betas_stim_k = results_k.betas[:, : packed_k.n_task_cols].cpu().numpy()

            for cond_idx in range(n_conditions):
                cond_start = cond_idx * n_regs_k
                betas_cond_k = betas_stim_k[:, cond_start : cond_start + n_regs_k]

                padded = np.zeros((len(vox_this_top), max_n_basis_out), dtype=np.float32)
                if zero_edges_pv:
                    # slot 0 and slot n_knots_k-1 are forced zeros (edge constraints)
                    # inner betas at slots 1 … n_regs_k
                    padded[:, 1 : 1 + n_regs_k] = betas_cond_k
                else:
                    padded[:, :n_regs_k] = betas_cond_k
                assembled_betas[cond_idx][vox_this_top] = padded

        del data_tensor_pv

        # 8. Save iresp files for each condition (max-window dimensions)
        if args.verb >= 1:
            print("\nSaving per-voxel HRF estimates (iresp files)...")

        output_files_pv = []
        for cond_idx in range(n_conditions):
            betas_4d = _to_volume(assembled_betas[cond_idx])
            iresp_cond = betas_4d[:, :, :, np.newaxis, :]
            t_write = time.perf_counter()
            with spinner(
                f"Writing iresp: {condition_labels[cond_idx]}",
                enabled=args.verb >= 1,
                leave=False,
            ):
                files = save_iresp(
                    iresp=iresp_cond,
                    output_prefix=args.prefix,
                    condition_labels=[condition_labels[cond_idx]],
                    tr=tr,
                    bot=_pv_bot,
                    top=max_top_pv,
                    reference_img=input_files[0],
                    nii_ext=_nii_ext,
                )
            _announce_written(files, time.perf_counter() - t_write, args.verb)
            output_files_pv.extend(files)

        if args.verb >= 1:
            print(f"\n{'=' * 70}")
            print("✓ Per-voxel deconvolution complete!")
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'=' * 70}")
        else:
            print("Created HRF estimate files (per-voxel windows):")
            for f in output_files_pv:
                print(f"  {f}")

        return 0

    # Build per-run task designs via the shared API
    # ----------------------------------------------------------------
    # All FIR/TENT/CSPLIN per-run construction now routes through
    # fastfuncstuff.design.builder.build_per_run_task_designs — the
    # *same* function librarian uses, so any bug here surfaces in
    # both tools at once instead of lurking in one.
    if args.verb >= 1:
        print("\nBuilding design matrices...")

    from fastfuncstuff.design.builder import build_per_run_task_designs

    # The shared builder accepts (bot, top) windows per condition for
    # TENT/CSPLIN, or a scalar `top` for FIR (bot=0 always).
    if model == "FIR":
        # Convert n_lags_per_cond → per-condition top windows (bot=0).
        per_cond_window: list[float] | list[tuple[float, float]] = [
            float(n) * tr for n in n_lags_per_cond
        ]
    elif model in ("TENT", "TENTzero", "CSPLIN", "CSPLINzero"):
        if tent_windows is None:
            raise RuntimeError(f"tent_windows should not be None for {model} model")
        per_cond_window = [(float(b), float(t)) for b, t in tent_windows]
    else:
        # Assumed-HRF path (spmg1/etc.) is handled by a separate branch
        # earlier in deconvolve; this design-build path only fires for
        # FIR/TENT/CSPLIN family models.
        per_cond_window = None  # not used

    design_result = build_per_run_task_designs(
        onsets_per_cond_per_run=onsets_per_condition,
        n_timepoints_per_run=list(n_timepoints_per_run),
        tr=tr,
        basis=model,
        condition_labels=list(condition_labels),
        fir_window_s=per_cond_window,
        tent_n_basis=args.tent_n_basis,
        device=device,
    )

    per_run_designs = design_result.per_run
    n_basis_per_condition_list = list(design_result.n_basis_per_condition)
    column_labels = list(design_result.column_labels)
    n_stimulus_regressors = sum(n_basis_per_condition_list)

    if args.verb >= 1:
        n_task_rows = sum(d.shape[0] for d in per_run_designs)
        print(f"  Task design shape: ({n_task_rows}, {per_run_designs[0].shape[1]})")
        if len(set(n_basis_per_condition_list)) == 1:
            print(
                f"  Stimulus regressors: {n_stimulus_regressors} ({n_basis_per_condition_list[0]} per condition)"
            )
        else:
            print(f"  Stimulus regressors: {n_stimulus_regressors}")
            for i, n_basis in enumerate(n_basis_per_condition_list):
                print(f"    {condition_labels[i]}: {n_basis} regressors")

    # Polynomial nuisance is built by pack_for_shared_task_glm later
    # in the GLM-prep section.  The old code that appended polys here
    # (and the parallel set built inside fit_glm via max_poly_degree)
    # was the historical foot-gun — both can produce the same result
    # but the duplication has caused real bugs.  Now there's exactly
    # one place where polys enter the design: pack_for_shared_task_glm.

    # Prepare data and pack into the canonical shared-task GLM form.
    #
    # The canonical multi-run fMRI GLM is::
    #
    #   [  task block  | run0_poly | run1_poly | ... ]
    #   [   (shared    |   ↑↑↑    |    0      |     ]
    #   [   across     |   run0    |          |     ]
    #   [   runs)      |    0     |   run1   |     ]
    #
    # i.e. ONE set of task betas estimated jointly across all runs +
    # per-run polynomial nuisance on a block diagonal (polys absorb
    # run-specific means / drifts / trends, which are NOT shared).
    # ``fit_glm`` alone does NOT produce this when handed per-run lists
    # — it block-diagonalizes the task block too, estimating per-run
    # betas (1/n_runs of the data per estimate, much noisier).
    #
    # ``pack_for_shared_task_glm`` builds the canonical concatenated
    # form once; the fit then runs with ``max_poly_degree=-1`` to
    # suppress ``fit_glm``'s auto-poly path (polys are already in the
    # packed design — don't double-count them).
    if args.verb >= 1:
        print("\nPreparing data for GLM (shared task + block-diag polys)...")

    per_run_data = data_list  # loader already applied -mask / -do_blur / -do_scale

    packed = pack_for_shared_task_glm(
        per_run_data=per_run_data,
        per_run_task_designs=per_run_designs,
        polort=args.polort,
        task_column_labels=column_labels,
        extra_regressors_per_run=extra_regs_per_run,
        # OLS solve: a per-run nuisance block supplied for only some runs
        # leaves all-zero columns in the others' diagonal slots.
        drop_empty_nuisance=True,
        device=torch.device("cpu"),
    )

    # The save-design / save-plot artefacts below are written from the fully
    # augmented (task + polys + external nuisance) packed form, so what lands
    # on disk is exactly what fit_glm sees.
    design_full = packed.design_concat
    column_labels = packed.column_labels
    n_stimulus_regressors = packed.n_task_cols

    # Save design matrix if requested
    if args.save_design:
        design_file = f"{args.prefix}_design.1D"
        design_np = design_full.cpu().numpy()

        if args.verb >= 1:
            print("\nSaving design matrix...")
            print(f"  {design_file}")

        # Save as AFNI .1D format with header
        with open(design_file, "w") as f:
            # Write metadata header
            f.write("# Design matrix for ffs_deconvolve.py\n")
            f.write(f"# Model: {model}\n")
            f.write(f"# TR: {tr}s\n")
            f.write(f"# Runs: {len(n_timepoints_per_run)}\n")
            f.write(
                f"# Timepoints: {design_np.shape[0]} ({', '.join(map(str, n_timepoints_per_run))})\n"
            )
            f.write(
                f"# Regressors: {design_np.shape[1]} ({n_stimulus_regressors} stimulus + {design_np.shape[1] - n_stimulus_regressors} nuisance)\n"
            )

            if model in ("TENT", "TENTzero", "CSPLIN", "CSPLINzero"):
                basis_type = "Cubic spline" if "CSPLIN" in model else "TENT"
                f.write(f"# {basis_type} windows:\n")
                for cond_idx in range(n_conditions):
                    if tent_windows is not None:
                        bot, top = tent_windows[cond_idx]
                        f.write(
                            f"#   {condition_labels[cond_idx]}: {bot}s to {top}s ({n_basis_per_condition_list[cond_idx]} basis)\n"
                        )
            elif model == "FIR":
                f.write(
                    f"# FIR lags: {n_basis_per_condition_list[0]} (duration: {(n_basis_per_condition_list[0] - 1) * tr}s)\n"
                )

            if args.polort >= 0:
                f.write("#\n")
                f.write("# Polynomial drift (zero-padded per run):\n")
                f.write(f"#   Order: {args.polort}\n")
                f.write(f"#   Regressors per run: {args.polort + 1}\n")
                f.write(
                    f"#   Total polynomial regressors: {len(n_timepoints_per_run) * (args.polort + 1)}\n"
                )

            if extra_regs_per_run is not None:
                n_extra = extra_regs_per_run[0].shape[1]
                f.write("#\n")
                f.write("# External nuisance (zero-padded per run):\n")
                f.write(f"#   Regressors per run: {n_extra}\n")
                f.write(f"#   Total: {n_extra * len(n_timepoints_per_run)}\n")

            f.write("#\n")
            f.write("# Column labels:\n")
            f.write("# " + " ".join(column_labels) + "\n")

            # Write data
            for row in design_np:
                f.write(" ".join(f"{val:.6f}" for val in row) + "\n")

        if args.verb >= 1:
            print(
                f"  ✓ Design matrix saved ({design_np.shape[0]} rows x {design_np.shape[1]} cols)"
            )

    # Save design matrix plot if requested
    if args.save_design_plot:
        # Create figures directory
        figs_dir = f"{args.prefix}_figures"
        Path(figs_dir).mkdir(parents=True, exist_ok=True)
        design_plot_file = f"{figs_dir}/design.png"
        design_np = design_full.cpu().numpy()

        if args.verb >= 1:
            print("\nSaving design matrix plot...")
            print(f"  {design_plot_file}")

        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt

            # Create figure (time on vertical axis)
            _, ax = plt.subplots(figsize=(10, 12))

            # Plot design matrix (no interpolation!)
            # Rows = time, Columns = regressors
            im = ax.imshow(design_np, aspect="auto", interpolation="none", cmap="RdBu_r")

            # Add colorbar
            plt.colorbar(im, ax=ax, label="Design value")

            # Labels
            ax.set_ylabel("Time (TRs)", fontsize=12)
            ax.set_xlabel("Regressor", fontsize=12)
            ax.set_title(
                f"Design Matrix: {model} model ({design_np.shape[0]} TRs x {design_np.shape[1]} regressors)",
                fontsize=14,
            )

            # Add grid lines between stimulus and nuisance
            if n_stimulus_regressors < design_np.shape[1]:
                # Vertical line between stimulus and nuisance
                ax.axvline(
                    n_stimulus_regressors - 0.5,
                    color="black",
                    linewidth=2,
                    linestyle="--",
                    alpha=0.7,
                )

            # Add horizontal lines for run boundaries
            tr_idx = 0
            for _run_idx, n_tp in enumerate(n_timepoints_per_run[:-1]):
                tr_idx += n_tp
                ax.axhline(tr_idx - 0.5, color="yellow", linewidth=1, linestyle="-", alpha=0.5)

            # Add vertical lines between conditions (for TENT with different n_basis)
            if len(set(n_basis_per_condition_list)) > 1:
                basis_idx = 0
                for _cond_idx, n_basis in enumerate(n_basis_per_condition_list[:-1]):
                    basis_idx += n_basis
                    ax.axvline(
                        basis_idx - 0.5, color="green", linewidth=1, linestyle="-", alpha=0.5
                    )

            # Set x-tick labels to show condition names
            # Show labels at the center of each condition's regressors
            x_tick_positions = []
            x_tick_labels = []
            basis_idx = 0
            for cond_idx, n_basis in enumerate(n_basis_per_condition_list):
                x_tick_positions.append(basis_idx + n_basis / 2)
                x_tick_labels.append(condition_labels[cond_idx])
                basis_idx += n_basis

            n_nuisance_cols = design_np.shape[1] - n_stimulus_regressors
            if n_nuisance_cols > 0:
                x_tick_positions.append(n_stimulus_regressors + n_nuisance_cols / 2)
                x_tick_labels.append("nuisance")

            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels(x_tick_labels)

            # Tight layout
            plt.tight_layout()

            # Save
            plt.savefig(design_plot_file, dpi=150, bbox_inches="tight")
            plt.close()

            if args.verb >= 1:
                print("  ✓ Design matrix plot saved")

        except ImportError:
            print("WARNING: matplotlib not available, skipping design plot", file=sys.stderr)

    if args.verb >= 1:
        n_vox_total = packed.data_concat.shape[0]
        n_poly_cols = packed.design_concat.shape[1] - packed.n_task_cols
        print(
            f"  Data: ({n_vox_total}, {packed.design_concat.shape[0]}) "
            f"across {len(per_run_data)} runs"
        )
        n_poly_total = ((args.polort + 1) if args.polort >= 0 else 0) * len(per_run_data)
        print(
            f"  Design: {tuple(packed.design_concat.shape)}  "
            f"({packed.n_task_cols} task + {n_poly_cols} block-diagonal nuisance "
            f"= {n_poly_total} poly + {n_poly_cols - n_poly_total} external)"
        )
        print("\nFitting GLM (chunked for GPU memory)...")

    results = fit_glm(
        data=packed.data_concat,
        design=packed.design_concat,
        tr=tr,
        max_poly_degree=-1,  # polys already packed in
        device=device,
        preload_data_to_device=False,  # stream chunks to GPU
        chunk_size=None,  # auto-estimate
        verbose=args.verb >= 1,
        debug_memory=args.debug_memory,
        debug_design=args.debug_design,
    )

    if args.verb >= 1:
        print("  ✓ GLM fit complete")

    # ── R² maps ──────────────────────────────────────────────────────────────
    if args.save_r2:
        if results.r2 is None:
            print("WARNING: fit_glm returned no R²; skipping -save-r2", file=sys.stderr)
        else:
            r2_file = f"{args.prefix}_r2{_nii_ext}"
            r2_vol = _to_volume(results.r2.cpu().numpy().reshape(-1, 1)).squeeze(-1)
            t_write = time.perf_counter()
            with spinner(f"Writing {Path(r2_file).name}", enabled=args.verb >= 1, leave=False):
                save_nifti(r2_vol, r2_file, reference_img=input_files[0])
            _announce_written(r2_file, time.perf_counter() - t_write, args.verb)
            if args.verb >= 1:
                print(f"      in-sample, mean R² = {float(results.r2.mean()):.4f}")
            del r2_vol

    if args.save_xval_r2:
        xval_r2_map = _compute_xval_r2_map(
            packed=packed,
            run_starts=run_starts,
            n_runs=n_runs,
            cv_strategy=args.cv_strategy,
            metric=args.cv_metric,
            device=device,
            verbose=args.verb >= 1,
        )
        if xval_r2_map is None:
            print(
                "WARNING: -save-xval-r2 needs ≥2 runs; skipping.",
                file=sys.stderr,
            )
        else:
            xr2_file = f"{args.prefix}_xval_r2{_nii_ext}"
            xr2_vol = _to_volume(xval_r2_map.reshape(-1, 1)).squeeze(-1)
            t_write = time.perf_counter()
            with spinner(f"Writing {Path(xr2_file).name}", enabled=args.verb >= 1, leave=False):
                save_nifti(xr2_vol, xr2_file, reference_img=input_files[0])
            _announce_written(xr2_file, time.perf_counter() - t_write, args.verb)
            if args.verb >= 1:
                print(
                    f"      held-out, median R² = {float(np.median(xval_r2_map)):.4f}, "
                    f"max {float(xval_r2_map.max()):.4f}"
                )
            del xr2_vol

    # Extract HRF estimates (only stimulus betas, not polynomials)
    if args.verb >= 1:
        print("\nExtracting HRF estimates...")

    betas_stimulus = results.betas[:, :n_stimulus_regressors].cpu().numpy()

    # Save iresp files (per-condition, since they may have different n_basis)
    if args.verb >= 1:
        print("\nSaving HRF estimates (iresp files)...")

    output_files = []
    beta_col_idx = 0  # Track position in beta matrix

    for cond_idx in range(n_conditions):
        n_basis = n_basis_per_condition_list[cond_idx]

        # Extract betas for this condition
        betas_cond = betas_stimulus[:, beta_col_idx : beta_col_idx + n_basis]
        beta_col_idx += n_basis

        # Reshape to 4D (nx, ny, nz, n_basis)
        betas_4d = _to_volume(betas_cond)

        # For zero-edge models (TENTzero/CSPLINzero) the first and last basis
        # functions were dropped (forced to zero). Pad them back so the saved
        # iresp spans the full window [bot, top] with explicit zeros at the edges.
        if model in ("TENTzero", "CSPLINzero"):
            zeros = np.zeros((nx, ny, nz, 1), dtype=betas_4d.dtype)
            betas_4d = np.concatenate([zeros, betas_4d, zeros], axis=3)

        # Add condition dimension: (nx, ny, nz, 1, n_lags)
        iresp_cond = betas_4d[:, :, :, np.newaxis, :]

        # Determine window for metadata
        if model == "FIR":
            bot_for_save = 0.0
            top_for_save = (n_basis - 1) * tr
        else:
            # TENT/TENTzero
            if tent_windows is None:
                raise RuntimeError("tent_windows should not be None for TENT/TENTzero model")
            bot_for_save, top_for_save = tent_windows[cond_idx]

        # Save this condition
        t_write = time.perf_counter()
        with spinner(
            f"Writing iresp: {condition_labels[cond_idx]}",
            enabled=args.verb >= 1,
            leave=False,
        ):
            files = save_iresp(
                iresp=iresp_cond,
                output_prefix=args.prefix,
                condition_labels=[condition_labels[cond_idx]],
                tr=tr,
                bot=bot_for_save,
                top=top_for_save,
                reference_img=input_files[0],
                nii_ext=_nii_ext,
            )
        _announce_written(files, time.perf_counter() - t_write, args.verb)
        output_files.extend(files)

    if args.verb < 1:
        print("Created HRF estimate files:")
        for f in output_files:
            print(f"  {f}")

    # Save beta coefficients if requested
    if args.save_betas:
        if args.verb >= 1:
            print("\nSaving beta coefficients...")

        # Reshape betas back to 4D
        betas_4d = _to_volume(betas_stimulus)

        # Save as 4D NIfTI
        beta_file = f"{args.prefix}_betas{_nii_ext}"
        t_write = time.perf_counter()
        with spinner(f"Writing {Path(beta_file).name}", enabled=args.verb >= 1, leave=False):
            save_nifti(betas_4d, beta_file, reference_img=input_files[0])
        _announce_written(beta_file, time.perf_counter() - t_write, args.verb)

    # Done
    if args.verb >= 1:
        print(f"\n{'=' * 70}")
        print("✓ Deconvolution complete!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
