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
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    import nibabel as nib  # noqa: F401 — availability check
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

# Import fastfuncstuff modules
try:
    from fastfuncstuff.io.afni import (
        get_tr_from_file,
        load_afni_mask,
        load_nifti,
        onsets_to_tr_matrix,
        save_nifti,
    )
    from fastfuncstuff.cli_utils import add_verbose_arg, auto_polort, parse_device_arg, parse_prefix
    from fastfuncstuff.design.matrices import (
        build_glm_design,
        is_tr_locked,
        make_csplin_design,
        make_tent_design,
        save_iresp,
    )
    from fastfuncstuff.design.builder import (
        legendre_polynomials,
        parse_afni_timing_file,
    )
    from fastfuncstuff.glm.core import fit_glm
    from fastfuncstuff.glm.xval import compute_r2_metric
    from fastfuncstuff.utils import configure_torch_backends, get_device
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    print("Make sure fastfuncstuff is installed: pip install -e .")
    sys.exit(1)


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults while preserving raw description formatting."""


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Fast fMRI deconvolution with FIR/TENT models",
        add_help=False,
        formatter_class=_HelpFormatter,
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
            "Use -event_ignore to skip conditions; -event_cols for non-standard column names."
        ),
    )
    required.add_argument(
        "-event_ignore",
        nargs="+",
        metavar="CONDITION",
        help="trial_type values to exclude when using -events (e.g. -event_ignore fixation null).",
    )
    required.add_argument(
        "-event_cols",
        nargs=3,
        metavar=("ONSET_COL", "DURATION_COL", "TRIAL_TYPE_COL"),
        help=(
            "Custom column names for -events TSV files, replacing BIDS defaults. "
            "E.g. -event_cols onset_time duration_s condition_name"
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
        choices=["AUTO", "FIR", "TENT", "TENTzero", "CSPLIN", "CSPLINzero"],
        default="AUTO",
        help="Deconvolution model: AUTO (auto-detect TR-locking), FIR, TENT, TENTzero, "
             "CSPLIN (cubic spline - smoother than TENT), CSPLINzero (default: AUTO)",
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
        "-add_lag",
        nargs="+",
        type=int,
        metavar="TRS",
        help=(
            "Per-condition lag adjustment in TRs applied on top of the estimated or explicit "
            "window.  Positive = more lags, negative = fewer.  "
            "Provide one value (broadcast to all conditions) or one per condition.  "
            "E.g. -add_lag 2 or -add_lag 1 0 -2"
        ),
    )

    model_opts.add_argument(
        "-tent_n_basis",
        type=int,
        metavar="N",
        help="Number of TENT basis functions (knots). Default: auto-calculated for TR spacing.",
    )

    model_opts.add_argument(
        "-xval_tr_range",
        type=int,
        default=0,
        metavar="N",
        help="Cross-validate the TENT window upper bound over ±N TRs around the -window "
             "top, in steps of 1 TR, using leave-one-run-out CV. The top with the highest "
             "mean held-out R² is used for the final fit. "
             "E.g., '-window 0 15 -xval_tr_range 3' tries tops 12s..18s (default: 0 = disabled).",
    )

    model_opts.add_argument(
        "-per_voxel",
        action="store_true",
        help="When combined with -xval_tr_range, select the best window top *per voxel* "
             "rather than a single shared top. Each voxel's HRF is estimated with its "
             "individually optimal window; shorter windows are zero-padded on the right to "
             "the longest candidate window. Saves additional maps: "
             "{prefix}_windowsize.nii.gz (winning top in seconds) and "
             "{prefix}_r2_by_window.nii.gz (LORO R² per candidate, 4D). "
             "Requires -xval_tr_range > 0 and ≥2 runs.",
    )

    model_opts.add_argument(
        "-round_onsets",
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
            "Reducing TENT/CSPLIN to FIR: use -round_onsets then -model FIR."
        ),
    )

    model_opts.add_argument(
        "-round_durations",
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
        "-tr_lock_threshold",
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

    # Output options
    out_opts = parser.add_argument_group("Output Options")
    out_opts.add_argument(
        "-save_betas",
        action="store_true",
        help="Save beta coefficients as 4D NIfTI file",
    )

    out_opts.add_argument(
        "-save_design",
        action="store_true",
        help="Save design matrix as .1D file (AFNI format)",
    )

    out_opts.add_argument(
        "-save_design_plot",
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
    hw_opts.add_argument(
        "-device",
        type=str,
        default=None,
        help="PyTorch device: cpu, cuda, mps (default: auto-detect). Overrides --cpu.",
    )
    hw_opts.add_argument(
        "-debug_memory",
        action="store_true",
        help="Print VRAM usage vs. prediction after each chunk loop (for memory tuning)",
    )

    # Help
    parser.add_argument(
        "-help",
        action="store_true",
        help="Show this help message and exit",
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
    max_len = max(len(p) for p in parts_list)
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
                raise ValueError(f"Invalid tent_window values in '{arg}'. Expected numeric values.") from None
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


def print_help(parser):
    """Print help message with examples"""
    print(__doc__)
    print("\nCommand-line options:")
    print("=" * 70)
    parser.print_help()


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
            train_c = torch.cat(
                [data_clean[r][i : i + chunk, :] for r in train_runs], dim=1
            ).to(device)
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
    data_list: list[np.ndarray],
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
        Indices into the flattened voxel axis of *data_list* (0 … nx*ny*nz-1).
    """
    n_tp_per_run = [d.shape[1] for d in data_list]
    n_voxels_total = data_list[0].shape[0]

    # ── Coarse background filter ─────────────────────────────────────────────
    rng = np.random.default_rng(42)
    var_proxy = data_list[0].var(axis=1)
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
        data_r = torch.tensor(data_xval[run_idx], dtype=torch.float32, device=device)
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
                    fn([onsets_per_condition[c][run_idx]], bot, top_val, tr, n_tp,
                       zero_edges=zero_edges, device=device)
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
    data_list: list[np.ndarray],
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
        data_list, onsets_per_condition, model, bot, candidate_tops,
        tr, n_conditions, polort, device, verbose,
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
        cv_log = sorted(zip(candidate_tops, scores.tolist()), key=lambda x: x[0])
        for top_k, r2_k in cv_log:
            marker = " ← selected" if abs(top_k - best_top) < 1e-6 else ""
            print(f"    {bot:.1f}–{top_k:.2f}s  R²={r2_k:.4f}{marker}")

    return best_top, best_r2


def main():
    """Main CLI entry point"""
    parser = parse_args()

    # Check for help flag before parsing (to bypass required argument checks)
    if "-help" in sys.argv or "--help" in sys.argv:
        print_help(parser)
        return 0

    args = parser.parse_args()

    pfx = parse_prefix(args.prefix)
    args.prefix = pfx.stem  # overwrite with clean stem
    _nii_ext = pfx.nifti_ext

    # Setup device — -device takes precedence over legacy --cpu flag
    device_spec = args.device or ("cpu" if args.cpu else None)
    device, _, _ = parse_device_arg(device_spec)
    if args.verb >= 1:
        print(f"Using device: {device}")
    configure_torch_backends(device)

    # Validate design source
    n_runs = len(args.input)
    if not args.onsets and not args.events:
        print("ERROR: Must specify -onsets or -events", file=sys.stderr)
        return 1
    if args.onsets and args.events:
        print("ERROR: -onsets and -events are mutually exclusive", file=sys.stderr)
        return 1
    if (args.event_ignore or args.event_cols) and not args.events:
        print("ERROR: -event_ignore and -event_cols require -events", file=sys.stderr)
        return 1

    # ── BIDS events: parse early so we have n_conditions/labels/onsets before data load ──
    condition_durations: list[float] | None = None  # stimulus durations for auto-window
    if args.events:
        if len(args.events) != n_runs:
            print(
                f"ERROR: -events requires one TSV per run: "
                f"got {len(args.events)} events files but {n_runs} input datasets.",
                file=sys.stderr,
            )
            return 1
        from fastfuncstuff.design.bids_events import parse_bids_events, sort_bids_event_files
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
        n_conditions = len(bids_labels)
        # User-supplied -labels override BIDS trial_type names
        if args.labels:
            if len(args.labels) != n_conditions:
                print(
                    f"ERROR: -labels count ({len(args.labels)}) does not match "
                    f"number of BIDS conditions ({n_conditions}: {bids_labels})",
                    file=sys.stderr,
                )
                return 1
            condition_labels = args.labels
        else:
            condition_labels = bids_labels
        onsets_per_condition = bids_onsets
        condition_durations = bids_durations  # used for auto-window estimation

    else:
        # ── AFNI timing files: labels determined now, onsets loaded after data ──
        n_conditions = len(args.onsets)
        if args.labels:
            if len(args.labels) != n_conditions:
                print(
                    f"ERROR: Number of labels ({len(args.labels)}) must match "
                    f"number of onset files ({n_conditions})",
                    file=sys.stderr,
                )
                return 1
            condition_labels = args.labels
        else:
            condition_labels = _labels_from_timing_files(args.onsets)
        onsets_per_condition = None  # loaded below, after data

    if args.verb >= 1:
        print(f"\n{'=' * 70}")
        print("Fast fMRI Deconvolution")
        print(f"{'=' * 70}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nInput:")
        print(f"  Runs: {n_runs}")
        print(f"  Conditions: {n_conditions}")
        print(f"  Condition labels: {', '.join(condition_labels)}")

    # Load data
    if args.verb >= 1:
        print("\nLoading fMRI data...")

    # data_list stores 2D (n_voxels, n_tp) arrays per run (float32)
    data_list = []
    n_timepoints_per_run = []
    tr_values = []
    nx = ny = nz = None

    for i, input_file in enumerate(tqdm(args.input, desc="Loading runs", unit="run")):
        if not Path(input_file).exists():
            print(f"ERROR: Input file not found: {input_file}", file=sys.stderr)
            return 1

        # Load data as float32 (matches reml.py pattern)
        img = load_nifti(input_file)
        data = img.get_fdata(dtype=np.float32)

        if data.ndim != 4:
            print(f"ERROR: Expected 4D data, got {data.ndim}D: {input_file}", file=sys.stderr)
            return 1

        # Get TR
        if args.tr is None:
            tr = get_tr_from_file(input_file)
            tr_values.append(tr)
        else:
            tr = args.tr

        # Store spatial dims from first run
        if nx is None:
            nx, ny, nz = data.shape[:3]

        n_tp = data.shape[3]
        n_timepoints_per_run.append(n_tp)

        # Flatten to 2D immediately to avoid large 4D concatenation later
        data_list.append(data.reshape(-1, n_tp))

    # Check TR consistency
    if args.tr is None:
        if len(set(tr_values)) > 1:
            print(
                f"ERROR: Inconsistent TRs across runs: {tr_values}. Use -tr to override.",
                file=sys.stderr,
            )
            return 1
        tr = tr_values[0]

    if args.verb >= 1:
        print(f"  TR: {tr}s")
        print(f"  Total timepoints: {sum(n_timepoints_per_run)}")

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
            print(f"ERROR: -polort must be 'A' or an integer, got: {args.polort!r}", file=sys.stderr)
            return 1

    if args.verb >= 1:
        print(f"  Data shape: {nx} x {ny} x {nz}")

    # Load mask if provided
    mask = None
    if args.mask:
        if args.verb >= 1:
            print(f"\nLoading mask: {args.mask}")
        mask = load_afni_mask(args.mask)

        # Check mask shape
        if mask.shape != (nx, ny, nz):
            print(
                f"ERROR: Mask shape {mask.shape} doesn't match data shape {(nx, ny, nz)}",
                file=sys.stderr,
            )
            return 1

        n_voxels = np.sum(mask)
        if args.verb >= 1:
            print(
                f"  Mask: {n_voxels} / {nx * ny * nz} voxels ({100 * n_voxels / (nx * ny * nz):.1f}%)"
            )

    # Load onsets
    if args.onsets:
        # ── AFNI timing files ────────────────────────────────────────────────
        if args.verb >= 1:
            print("\nLoading onset timing files...")

        onsets_per_condition = []
        for onset_file in args.onsets:
            if args.verb >= 1:
                print(f"  {onset_file}")

            onsets_by_run = parse_afni_timing_file(onset_file)

            if len(onsets_by_run) != n_runs:
                print(
                    f"ERROR: Timing file {onset_file} has {len(onsets_by_run)} runs, "
                    f"but expected {n_runs} runs",
                    file=sys.stderr,
                )
                return 1

            onsets_per_condition.append(onsets_by_run)

        # Parse per-condition stimulus durations for auto-window estimation
        if args.durations:
            from fastfuncstuff.design.builder import parse_durations
            condition_durations = parse_durations(args.durations, n_conditions, condition_labels)
            if args.verb >= 1:
                if len(set(condition_durations)) == 1:
                    print(f"  Stimulus duration (all conditions): {condition_durations[0]:.3f}s")
                else:
                    for lbl, dur in zip(condition_labels, condition_durations):
                        print(f"  {lbl}: {dur:.3f}s")

    else:
        # BIDS path: onsets_per_condition already set above
        if args.verb >= 1:
            from fastfuncstuff.design.bids_events import sort_bids_event_files
            print("\nBIDS events files (sorted by run):")
            for ep in sort_bids_event_files(args.events):
                print(f"  {ep}")
            print()
            for cidx, lbl in enumerate(condition_labels):
                n_ev = sum(len(onsets_per_condition[cidx][r]) for r in range(n_runs))
                print(f"  {lbl}: {n_ev} events  (duration={condition_durations[cidx]:.3f}s)")  # type: ignore[index]

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
            print(f"Durations rounded to {dp} decimal place(s): "
                  f"{[f'{d:.{dp}f}' for d in condition_durations]}")

    # Flatten all onset times for TR-locking check
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
                f"ERROR: -add_lag must have 1 or {n_conditions} values (got {len(add_lag_raw)})",
                file=sys.stderr,
            )
            return 1
    else:
        add_lag_list = [0] * n_conditions

    def _apply_add_lag(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Adjust per-condition (bot, top) windows by add_lag (TR units)."""
        result = []
        for (bot, top), lag in zip(windows, add_lag_list):
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
        tent_windows = compute_windows_from_durations(
            condition_durations, tr, add_lag=add_lag_list
        )
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
                for lbl, n_l, (_, top) in zip(condition_labels, n_lags_per_cond, tent_windows):
                    print(f"    {lbl}: {n_l} TRs ({top:.1f}s)")
        else:
            basis_type = "cubic spline" if "CSPLIN" in model else "tent"
            if len(set(tent_windows)) == 1:
                bot, top = tent_windows[0]
                n_basis_calc = args.tent_n_basis if args.tent_n_basis else round((top - bot) / tr) + 1
                n_actual = n_basis_calc - 2 if model in ("TENTzero", "CSPLINzero") else n_basis_calc
                print(f"  Window (all conditions): {bot}s–{top}s → {n_basis_calc} {basis_type} knots, {n_actual} regressors")
            else:
                print("  Windows (per condition):")
                for lbl, (bot, top) in zip(condition_labels, tent_windows):
                    n_basis_calc = args.tent_n_basis if args.tent_n_basis else round((top - bot) / tr) + 1
                    n_actual = n_basis_calc - 2 if model in ("TENTzero", "CSPLINzero") else n_basis_calc
                    print(f"    {lbl}: {bot}s–{top}s → {n_basis_calc} {basis_type} knots, {n_actual} regressors")

    # Cross-validate TENT window upper bound if requested
    # _pv_* variables carry per-voxel mode context into the block below.
    _do_per_voxel = False
    _pv_bot: float = 0.0
    _pv_nom_top: float = 0.0
    _pv_tops: list[float] = []

    if args.xval_tr_range > 0 and model in ("TENT", "TENTzero", "CSPLIN", "CSPLINzero"):
        if n_runs < 2:
            print("WARNING: -xval_tr_range requires ≥2 runs. Skipping.", file=sys.stderr)
        elif tent_windows is None:
            print("WARNING: tent_windows not set. Skipping xval.", file=sys.stderr)
        else:
            unique_tops = {w[1] for w in tent_windows}
            if len(unique_tops) > 1:
                print(
                    "WARNING: -xval_tr_range only supported when all conditions share "
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
                        print(
                            f"  Candidates: "
                            f"{', '.join(f'{t:.2f}s' for t in candidate_tops)}"
                        )
                else:
                    if args.verb >= 1:
                        print(f"\nCross-validating window top (±{n_range} TRs)...")
                        print(
                            f"  Candidates: "
                            f"{', '.join(f'{t:.2f}s' for t in candidate_tops)}"
                        )

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
        n_all_voxels = nx * ny * nz

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
        # r2_matrix: (n_candidates, n_vox); vox_idx: flat [0, nx*ny*nz)

        # 2. Per-voxel argmax → best candidate index → best top value
        best_cand_per_vox = np.argmax(r2_matrix, axis=0)  # (n_vox,)
        best_top_per_vox = np.array(
            [_pv_tops[i] for i in best_cand_per_vox], dtype=np.float32
        )

        # 3. Save r2_by_window map (4D: one volume per candidate)
        r2_bw_vol = np.zeros((n_all_voxels, len(_pv_tops)), dtype=np.float32)
        r2_bw_vol[vox_idx] = r2_matrix.T
        r2_bw_vol = r2_bw_vol.reshape(nx, ny, nz, len(_pv_tops))
        r2_bw_file = f"{args.prefix}_r2_by_window{_nii_ext}"
        save_nifti(r2_bw_vol, r2_bw_file, reference_img=args.input[0])
        del r2_bw_vol
        if args.verb >= 1:
            print(f"  Saved: {r2_bw_file}")

        # 4. Save windowsize map (3D: winning top in seconds)
        ws_vol = np.zeros(n_all_voxels, dtype=np.float32)
        ws_vol[vox_idx] = best_top_per_vox
        ws_vol = ws_vol.reshape(nx, ny, nz)
        ws_file = f"{args.prefix}_windowsize{_nii_ext}"
        save_nifti(ws_vol, ws_file, reference_img=args.input[0])
        del ws_vol
        if args.verb >= 1:
            print(f"  Saved: {ws_file}")

        # 5. Output dimensions: max window → max_n_knots timepoints per condition
        max_top_pv = max(_pv_tops)
        max_n_knots_pv = round((max_top_pv - _pv_bot) / tr) + 1
        max_n_basis_out = max_n_knots_pv  # includes zero-padded edges for *zero models

        if args.verb >= 1:
            print(
                f"  Output: {max_n_basis_out} timepoints per condition "
                f"(max window {_pv_bot:.1f}–{max_top_pv:.2f}s)"
            )

        # 6. Assemble full data tensor (unmasked; vox_idx is in full flat space)
        data_full_pv = np.concatenate(data_list, axis=1)  # (n_all_vox, total_tp)
        del data_list
        data_tensor_pv = torch.tensor(data_full_pv, dtype=torch.float32, device="cpu")
        del data_full_pv

        # Map voxels not in vox_idx (background) to the nominal top as fallback
        fallback_top = min(_pv_tops, key=lambda t: abs(t - _pv_nom_top))
        full_best_top = np.full(n_all_voxels, fallback_top, dtype=np.float32)
        full_best_top[vox_idx] = best_top_per_vox

        # 7. Assemble per-voxel betas: one (n_all_voxels, max_n_basis_out) array per condition
        assembled_betas = [
            np.zeros((n_all_voxels, max_n_basis_out), dtype=np.float32)
            for _ in range(n_conditions)
        ]

        unique_tops_pv = sorted(set(full_best_top.tolist()))
        if args.verb >= 1:
            print(f"\n  Fitting GLMs for {len(unique_tops_pv)} unique winning windows...")

        def _build_full_design_for_top(top_val: float) -> torch.Tensor:
            """Build stimulus + poly design for all runs concatenated."""
            stim_parts = []
            for run_idx, n_tp in enumerate(n_timepoints_per_run):
                cond_parts = []
                for cond_idx in range(n_conditions):
                    onset_times = onsets_per_condition[cond_idx][run_idx]
                    if use_csplin_pv:
                        d_c = make_csplin_design(
                            onset_times_list=[onset_times],
                            bot=_pv_bot, top=top_val, tr=tr, n_timepoints=n_tp,
                            n_basis=args.tent_n_basis,
                            zero_edges=(model == "CSPLINzero"), device=device,
                        )
                    else:
                        d_c = make_tent_design(
                            onset_times_list=[onset_times],
                            bot=_pv_bot, top=top_val, tr=tr, n_timepoints=n_tp,
                            n_basis=args.tent_n_basis,
                            zero_edges=(model == "TENTzero"), device=device,
                        )
                    cond_parts.append(d_c)
                stim_parts.append(torch.cat(cond_parts, dim=1))
            design_stim = torch.cat(stim_parts, dim=0)

            if args.polort < 0:
                return design_stim

            n_poly_per_run = args.polort + 1
            total_poly_cols = len(n_timepoints_per_run) * n_poly_per_run
            poly_full = np.zeros((sum(n_timepoints_per_run), total_poly_cols))
            tr_start = col_start = 0
            for run_idx, n_tp in enumerate(n_timepoints_per_run):
                poly_run = legendre_polynomials(n_tp, args.polort)
                poly_full[tr_start:tr_start + n_tp, col_start:col_start + n_poly_per_run] = poly_run
                tr_start += n_tp
                col_start += n_poly_per_run
            poly_tensor = torch.tensor(poly_full, dtype=torch.float32, device=device)
            return torch.cat([design_stim, poly_tensor], dim=1)

        for top_k in tqdm(unique_tops_pv, desc="  Fitting", disable=not args.verb >= 1):
            vox_this_top = np.where(full_best_top == top_k)[0]
            n_knots_k = round((top_k - _pv_bot) / tr) + 1
            n_regs_k = n_knots_k - 2 if zero_edges_pv else n_knots_k
            n_stim_k = n_conditions * n_regs_k

            design_k = _build_full_design_for_top(top_k)

            data_sub = data_tensor_pv[vox_this_top, :]
            results_k = fit_glm(
                data=data_sub,
                design=design_k,
                tr=tr,
                max_poly_degree=-1,
                device=device,
                preload_data_to_device=False,
                chunk_size=None,  # auto-estimate via memory module
                verbose=False,
                debug_memory=args.debug_memory,
            )
            betas_stim_k = results_k.betas[:, :n_stim_k].cpu().numpy()

            for cond_idx in range(n_conditions):
                cond_start = cond_idx * n_regs_k
                betas_cond_k = betas_stim_k[:, cond_start:cond_start + n_regs_k]

                padded = np.zeros((len(vox_this_top), max_n_basis_out), dtype=np.float32)
                if zero_edges_pv:
                    # slot 0 and slot n_knots_k-1 are forced zeros (edge constraints)
                    # inner betas at slots 1 … n_regs_k
                    padded[:, 1:1 + n_regs_k] = betas_cond_k
                else:
                    padded[:, :n_regs_k] = betas_cond_k
                assembled_betas[cond_idx][vox_this_top] = padded

        del data_tensor_pv

        # 8. Save iresp files for each condition (max-window dimensions)
        if args.verb >= 1:
            print("\nSaving per-voxel HRF estimates (iresp files)...")

        output_files_pv = []
        for cond_idx in range(n_conditions):
            betas_4d = assembled_betas[cond_idx].reshape(nx, ny, nz, max_n_basis_out)
            iresp_cond = betas_4d[:, :, :, np.newaxis, :]
            files = save_iresp(
                iresp=iresp_cond,
                output_prefix=args.prefix,
                condition_labels=[condition_labels[cond_idx]],
                tr=tr,
                bot=_pv_bot,
                top=max_top_pv,
                reference_img=args.input[0],
                nii_ext=_nii_ext,
            )
            output_files_pv.extend(files)

        if args.verb >= 1:
            for f in output_files_pv:
                print(f"  ✓ {f}")
            print(f"\n{'=' * 70}")
            print("✓ Per-voxel deconvolution complete!")
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'=' * 70}")
        else:
            print("Created HRF estimate files (per-voxel windows):")
            for f in output_files_pv:
                print(f"  {f}")

        return 0

    # Build design matrices
    if args.verb >= 1:
        print("\nBuilding design matrices...")

    design_list = []
    n_basis_per_condition_list = []  # Track basis functions per condition (can vary with TENT)

    for run_idx, n_tp in enumerate(n_timepoints_per_run):
        # Extract onsets for this run across all conditions
        run_onsets_all_conds = [onsets_per_condition[cond][run_idx] for cond in range(n_conditions)]

        # Convert to TR-resolution binary matrix (for FIR/TENT, no convolution)
        # onsets_to_tr_matrix expects [condition][run] format, so convert:
        # from [cond1_run0, cond2_run0] to [[cond1_run0], [cond2_run0]]
        onsets_binary = onsets_to_tr_matrix(
            [[onset] for onset in run_onsets_all_conds],  # Convert to [condition][run] format
            n_timepoints=n_tp,
            tr=tr,
        )

        # Convert to tensor
        onsets_binary = torch.tensor(onsets_binary, dtype=torch.float32, device=device)

        # Build design matrix
        if model == "FIR":
            # Per-condition FIR (each condition may have a different window)
            cond_designs = []
            for cond_idx in range(n_conditions):
                design_cond = build_glm_design(
                    onsets=onsets_binary[cond_idx : cond_idx + 1, :],
                    mode="fir",
                    n_fir_lags=n_lags_per_cond[cond_idx],
                    tr=tr,
                    device=device,
                )
                cond_designs.append(design_cond)
                if run_idx == 0:
                    n_basis_per_condition_list.append(n_lags_per_cond[cond_idx])
            design = torch.cat(cond_designs, dim=1)

        elif model in ("TENT", "TENTzero", "CSPLIN", "CSPLINzero"):
            # TENT/CSPLIN: use exact onset times (not binary matrix)
            # This produces fractional weights for non-TR-locked onsets
            if tent_windows is None:
                raise RuntimeError(f"tent_windows should not be None for {model} model")

            cond_designs = []
            for cond_idx in range(n_conditions):
                # Get onset times (in seconds) for this condition in this run
                onset_times = run_onsets_all_conds[cond_idx]

                # Get window for this condition
                bot, top = tent_windows[cond_idx]

                # Build design for this condition using exact onset times
                if model in ("TENT", "TENTzero"):
                    design_cond = make_tent_design(
                        onset_times_list=[onset_times],
                        bot=bot,
                        top=top,
                        tr=tr,
                        n_timepoints=n_tp,
                        n_basis=args.tent_n_basis,
                        zero_edges=(model == "TENTzero"),
                        device=device,
                    )
                else:  # CSPLIN or CSPLINzero
                    design_cond = make_csplin_design(
                        onset_times_list=[onset_times],
                        bot=bot,
                        top=top,
                        tr=tr,
                        n_timepoints=n_tp,
                        n_basis=args.tent_n_basis,
                        zero_edges=(model == "CSPLINzero"),
                        device=device,
                    )

                cond_designs.append(design_cond)

                # Track n_basis for this condition
                if run_idx == 0:
                    n_basis_per_condition_list.append(design_cond.shape[1])

            # Concatenate conditions horizontally
            design = torch.cat(cond_designs, dim=1)

        design_list.append(design)

    # Concatenate designs across runs
    design_full = torch.cat(design_list, dim=0)

    n_stimulus_regressors = design_full.shape[1]

    if args.verb >= 1:
        print(f"  Design matrix shape: {design_full.shape}")
        if len(set(n_basis_per_condition_list)) == 1:
            print(
                f"  Stimulus regressors: {n_stimulus_regressors} ({n_basis_per_condition_list[0]} per condition)"
            )
        else:
            print(f"  Stimulus regressors: {n_stimulus_regressors}")
            for i, n_basis in enumerate(n_basis_per_condition_list):
                print(f"    {condition_labels[i]}: {n_basis} regressors")

    # Create column labels for design matrix
    column_labels = []
    for cond_idx in range(n_conditions):
        n_basis = n_basis_per_condition_list[cond_idx]
        for basis_idx in range(n_basis):
            column_labels.append(f"{condition_labels[cond_idx]}#{basis_idx}")

    # Add polynomial drift regressors (zero-padded per run)
    if args.polort >= 0:
        n_poly_per_run = args.polort + 1
        total_poly_cols = len(n_timepoints_per_run) * n_poly_per_run

        # Create zero-padded polynomial matrix
        poly_full = np.zeros((sum(n_timepoints_per_run), total_poly_cols))

        tr_start = 0
        col_start = 0
        for run_idx, n_tp in enumerate(n_timepoints_per_run):
            # Generate polynomials for this run
            poly_run = legendre_polynomials(n_tp, args.polort)

            # Place in correct position (zero-padded)
            poly_full[tr_start:tr_start + n_tp, col_start:col_start + n_poly_per_run] = poly_run

            # Add column labels
            for poly_idx in range(n_poly_per_run):
                column_labels.append(f"run{run_idx+1}_poly{poly_idx}")

            tr_start += n_tp
            col_start += n_poly_per_run

        poly_tensor = torch.tensor(poly_full, dtype=torch.float32, device=device)

        # Append polynomials to design
        design_full = torch.cat([design_full, poly_tensor], dim=1)

        if args.verb >= 1:
            print(f"  Polynomial drift: {total_poly_cols} regressors ({n_poly_per_run} per run x {len(n_timepoints_per_run)} runs)")

    # Save design matrix if requested
    if args.save_design:
        design_file = f"{args.prefix}_design.1D"
        design_np = design_full.cpu().numpy()

        if args.verb >= 1:
            print("\nSaving design matrix...")
            print(f"  {design_file}")

        # Save as AFNI .1D format with header
        with open(design_file, 'w') as f:
            # Write metadata header
            f.write("# Design matrix for ffs_deconvolve.py\n")
            f.write(f"# Model: {model}\n")
            f.write(f"# TR: {tr}s\n")
            f.write(f"# Runs: {len(n_timepoints_per_run)}\n")
            f.write(f"# Timepoints: {design_np.shape[0]} ({', '.join(map(str, n_timepoints_per_run))})\n")
            f.write(f"# Regressors: {design_np.shape[1]} ({n_stimulus_regressors} stimulus + {design_np.shape[1] - n_stimulus_regressors} nuisance)\n")

            if model in ("TENT", "TENTzero", "CSPLIN", "CSPLINzero"):
                basis_type = "Cubic spline" if "CSPLIN" in model else "TENT"
                f.write(f"# {basis_type} windows:\n")
                for cond_idx in range(n_conditions):
                    if tent_windows is not None:
                        bot, top = tent_windows[cond_idx]
                        f.write(f"#   {condition_labels[cond_idx]}: {bot}s to {top}s ({n_basis_per_condition_list[cond_idx]} basis)\n")
            elif model == "FIR":
                f.write(f"# FIR lags: {n_basis_per_condition_list[0]} (duration: {(n_basis_per_condition_list[0]-1)*tr}s)\n")

            if args.polort >= 0:
                f.write("#\n")
                f.write("# Polynomial drift (zero-padded per run):\n")
                f.write(f"#   Order: {args.polort}\n")
                f.write(f"#   Regressors per run: {args.polort + 1}\n")
                f.write(f"#   Total polynomial regressors: {len(n_timepoints_per_run) * (args.polort + 1)}\n")

            f.write("#\n")
            f.write("# Column labels:\n")
            f.write("# " + " ".join(column_labels) + "\n")

            # Write data
            for row in design_np:
                f.write(" ".join(f"{val:.6f}" for val in row) + "\n")

        if args.verb >= 1:
            print(f"  ✓ Design matrix saved ({design_np.shape[0]} rows x {design_np.shape[1]} cols)")

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
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            # Create figure (time on vertical axis)
            _, ax = plt.subplots(figsize=(10, 12))

            # Plot design matrix (no interpolation!)
            # Rows = time, Columns = regressors
            im = ax.imshow(design_np, aspect='auto', interpolation='none', cmap='RdBu_r')

            # Add colorbar
            plt.colorbar(im, ax=ax, label='Design value')

            # Labels
            ax.set_ylabel('Time (TRs)', fontsize=12)
            ax.set_xlabel('Regressor', fontsize=12)
            ax.set_title(f'Design Matrix: {model} model ({design_np.shape[0]} TRs x {design_np.shape[1]} regressors)', fontsize=14)

            # Add grid lines between stimulus and nuisance
            if n_stimulus_regressors < design_np.shape[1]:
                # Vertical line between stimulus and nuisance
                ax.axvline(n_stimulus_regressors - 0.5, color='black', linewidth=2, linestyle='--', alpha=0.7)

            # Add horizontal lines for run boundaries
            tr_idx = 0
            for _run_idx, n_tp in enumerate(n_timepoints_per_run[:-1]):
                tr_idx += n_tp
                ax.axhline(tr_idx - 0.5, color='yellow', linewidth=1, linestyle='-', alpha=0.5)

            # Add vertical lines between conditions (for TENT with different n_basis)
            if len(set(n_basis_per_condition_list)) > 1:
                basis_idx = 0
                for _cond_idx, n_basis in enumerate(n_basis_per_condition_list[:-1]):
                    basis_idx += n_basis
                    ax.axvline(basis_idx - 0.5, color='green', linewidth=1, linestyle='-', alpha=0.5)

            # Set x-tick labels to show condition names
            # Show labels at the center of each condition's regressors
            x_tick_positions = []
            x_tick_labels = []
            basis_idx = 0
            for cond_idx, n_basis in enumerate(n_basis_per_condition_list):
                x_tick_positions.append(basis_idx + n_basis / 2)
                x_tick_labels.append(condition_labels[cond_idx])
                basis_idx += n_basis

            if args.polort >= 0:
                # Add label for polynomials
                poly_n = design_np.shape[1] - n_stimulus_regressors
                x_tick_positions.append(n_stimulus_regressors + poly_n / 2)
                x_tick_labels.append('polort')

            ax.set_xticks(x_tick_positions)
            ax.set_xticklabels(x_tick_labels)

            # Tight layout
            plt.tight_layout()

            # Save
            plt.savefig(design_plot_file, dpi=150, bbox_inches='tight')
            plt.close()

            if args.verb >= 1:
                print("  ✓ Design matrix plot saved")

        except ImportError:
            print("WARNING: matplotlib not available, skipping design plot", file=sys.stderr)

    # Prepare data
    if args.verb >= 1:
        print("\nPreparing data for GLM...")

    # Concatenate runs along time axis (data_list is already 2D per run)
    data_full = np.concatenate(data_list, axis=1)  # (n_all_voxels, total_tp)
    del data_list  # free memory

    # Apply mask if provided; fit_glm expects (n_voxels, n_timepoints)
    if mask is not None:
        data_masked = data_full[mask.flatten(), :]  # Shape: (n_voxels_masked, n_timepoints)
        if args.verb >= 1:
            print(f"  Data shape: {data_masked.shape}")
    else:
        data_masked = data_full  # already (n_all_voxels, n_timepoints)
        if args.verb >= 1:
            print(f"  Data shape: {data_masked.shape} (all voxels)")

    # Convert to CPU tensor (will be chunked to GPU during fitting)
    data_tensor = torch.tensor(data_masked, dtype=torch.float32, device="cpu")
    del data_masked, data_full

    # Fit GLM with chunking
    if args.verb >= 1:
        print("\nFitting GLM (chunked for GPU memory)...")
        print(f"  Design: {design_full.shape}")
        print(f"  Data: {data_tensor.shape}")

    # TODO estimate chunk size based on GPU memroy, n_voxels * timepoints * 4 bytes, and design matrix size

    results = fit_glm(
        data=data_tensor,
        design=design_full,
        tr=tr,
        # Polynomials are already included in design_full (block-diagonal per run).
        # Pass max_poly_degree=-1 so fit_glm does not add a second set.
        max_poly_degree=-1,
        device=device,
        preload_data_to_device=False,  # Stream chunks to GPU
        chunk_size=None,  # auto-estimate via memory module
        verbose=args.verb >= 1,
        debug_memory=args.debug_memory,
    )

    if args.verb >= 1:
        print("  ✓ GLM fit complete")

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
        if mask is not None:
            betas_4d = np.zeros((nx, ny, nz, n_basis))
            betas_4d[mask, :] = betas_cond
        else:
            betas_4d = betas_cond.reshape(nx, ny, nz, n_basis)

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
        files = save_iresp(
            iresp=iresp_cond,
            output_prefix=args.prefix,
            condition_labels=[condition_labels[cond_idx]],
            tr=tr,
            bot=bot_for_save,
            top=top_for_save,
            reference_img=args.input[0],
            nii_ext=_nii_ext,
        )
        output_files.extend(files)

    if args.verb >= 1:
        for f in output_files:
            print(f"  ✓ {f}")
    else:
        print("Created HRF estimate files:")
        for f in output_files:
            print(f"  {f}")

    # Save beta coefficients if requested
    if args.save_betas:
        if args.verb >= 1:
            print("\nSaving beta coefficients...")

        # Reshape betas back to 4D
        betas_4d = np.zeros((nx, ny, nz, n_stimulus_regressors))
        if mask is not None:
            betas_4d[mask, :] = betas_stimulus
        else:
            betas_4d = betas_stimulus.reshape(nx, ny, nz, n_stimulus_regressors)

        # Save as 4D NIfTI
        beta_file = f"{args.prefix}_betas{_nii_ext}"
        save_nifti(betas_4d, beta_file, reference_img=args.input[0])

        if args.verb >= 1:
            print(f"  ✓ {beta_file}")

    # Done
    if args.verb >= 1:
        print(f"\n{'=' * 70}")
        print("✓ Deconvolution complete!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

