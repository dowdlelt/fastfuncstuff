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
                      -tent_window 0 20 \\
                      -prefix results/GLM

Per-condition TENT windows:
    ffs_deconvolve.py -input run*.nii.gz \\
                      -onsets faces.txt scenes.txt objects.txt \\
                      -labels faces scenes objects \\
                      -model TENT \\
                      -tent_window 0,15 0,20 0,25 \\
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

try:
    import nibabel as nib  # noqa: F401 — availability check
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

# Import fastfuncsim modules
try:
    from fastfuncsim.io.afni import (
        get_tr_from_file,
        load_afni_mask,
        load_nifti,
        onsets_to_tr_matrix,
        save_nifti,
    )
    from fastfuncsim.design.matrices import (
        build_glm_design,
        is_tr_locked,
        save_iresp,
    )
    from fastfuncsim.design.builder import (
        legendre_polynomials,
        parse_afni_timing_file,
    )
    from fastfuncsim.glm.core import fit_glm
    from fastfuncsim.utils import get_device
except ImportError as e:
    print(f"ERROR: Could not import fastfuncsim: {e}")
    print("Make sure fastfuncsim is installed: pip install -e .")
    sys.exit(1)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Fast fMRI deconvolution with FIR/TENT models",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        required=True,
        help="Onset timing files in AFNI format (one file per condition, each with one row per run)",
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
        help="HRF duration in seconds (e.g., 20). Required for AUTO/FIR. For TENT, use -tent_window instead.",
    )

    model_opts.add_argument(
        "-tent_window",
        nargs="+",
        metavar="WINDOW",
        help="TENT window(s) in seconds after stimulus onset. "
        "Formats: '0 15' (all conditions), '0,15' (all conditions), "
        "or '0,15 0,20 0,25' (per-condition). For TENT/TENTzero models.",
    )

    model_opts.add_argument(
        "-tent_n_basis",
        type=int,
        metavar="N",
        help="Number of TENT basis functions (knots). Default: auto-calculated for TR spacing.",
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
        type=int,
        default=3,
        metavar="N",
        help="Polynomial drift order for detrending (default: 3, use -1 for none)",
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

    out_opts.add_argument(
        "-verbose",
        action="store_true",
        help="Print detailed progress information",
    )

    # Hardware options
    hw_opts = parser.add_argument_group("Hardware Options")
    hw_opts.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (default: auto-detect GPU)",
    )

    # Help
    parser.add_argument(
        "-help",
        action="store_true",
        help="Show this help message and exit",
    )

    return parser


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


def main():
    """Main CLI entry point"""
    parser = parse_args()

    # Check for help flag before parsing (to bypass required argument checks)
    if "-help" in sys.argv or "--help" in sys.argv:
        print_help(parser)
        return 0

    args = parser.parse_args()

    # Setup device
    if args.cpu:
        device = torch.device("cpu")
        if args.verbose:
            print("Using CPU")
    else:
        device = get_device()
        if args.verbose:
            print(f"Using device: {device}")

    # Validate inputs
    n_runs = len(args.input)
    n_conditions = len(args.onsets)

    if args.verbose:
        print(f"\n{'=' * 70}")
        print("Fast fMRI Deconvolution")
        print(f"{'=' * 70}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nInput:")
        print(f"  Runs: {n_runs}")
        print(f"  Conditions: {n_conditions}")

    # Check condition labels
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
        condition_labels = [f"cond{i + 1}" for i in range(n_conditions)]

    if args.verbose:
        print(f"  Condition labels: {', '.join(condition_labels)}")

    # Load data
    if args.verbose:
        print("\nLoading fMRI data...")

    data_list = []
    n_timepoints_per_run = []
    tr_values = []

    for i, input_file in enumerate(args.input):
        if args.verbose:
            print(f"  Run {i + 1}: {input_file}")

        if not Path(input_file).exists():
            print(f"ERROR: Input file not found: {input_file}", file=sys.stderr)
            return 1

        # Load data
        img = load_nifti(input_file)
        data = np.asarray(img.get_fdata())

        # Get TR
        if args.tr is None:
            tr = get_tr_from_file(input_file)
            tr_values.append(tr)
        else:
            tr = args.tr

        # Get n_timepoints
        n_timepoints = data.shape[3] if len(data.shape) > 3 else 1
        n_timepoints_per_run.append(n_timepoints)

        data_list.append(data)

    # Check TR consistency
    if args.tr is None:
        if len(set(tr_values)) > 1:
            print(
                f"ERROR: Inconsistent TRs across runs: {tr_values}. Use -tr to override.",
                file=sys.stderr,
            )
            return 1
        tr = tr_values[0]

    if args.verbose:
        print(f"  TR: {tr}s")
        print(f"  Total timepoints: {sum(n_timepoints_per_run)}")

    # Get data shape
    nx, ny, nz = data_list[0].shape[:3]

    if args.verbose:
        print(f"  Data shape: {nx} x {ny} x {nz}")

    # Load mask if provided
    mask = None
    if args.mask:
        if args.verbose:
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
        if args.verbose:
            print(
                f"  Mask: {n_voxels} / {nx * ny * nz} voxels ({100 * n_voxels / (nx * ny * nz):.1f}%)"
            )

    # Load onsets
    if args.verbose:
        print("\nLoading onset timing files...")

    onsets_per_condition = []
    for onset_file in args.onsets:
        if args.verbose:
            print(f"  {onset_file}")

        # Parse AFNI timing file
        onsets_by_run = parse_afni_timing_file(onset_file)

        if len(onsets_by_run) != n_runs:
            print(
                f"ERROR: Timing file {onset_file} has {len(onsets_by_run)} runs, "
                f"but expected {n_runs} runs",
                file=sys.stderr,
            )
            return 1

        onsets_per_condition.append(onsets_by_run)

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
            if args.verbose:
                print(f"\n✓ Onsets are TR-locked (threshold: {args.tr_lock_threshold * 100:.0f}%)")
                print("  Using FIR model")
        else:
            model = "TENT"
            if args.verbose:
                print(
                    f"\n✗ Onsets are NOT TR-locked (threshold: {args.tr_lock_threshold * 100:.0f}%)"
                )
                print("  Using TENT model")
    else:
        model = args.model
        if args.verbose:
            print(f"\nUsing {model} model")

    # Determine HRF window parameters
    if model == "FIR":
        if args.duration is None:
            print("ERROR: -duration required for FIR model", file=sys.stderr)
            return 1

        n_lags = int(np.ceil(args.duration / tr))
        tent_windows = None  # Not used for FIR

        if args.verbose:
            print(f"  Duration: {args.duration}s ({n_lags} TRs)")

    elif model in ("TENT", "TENTzero", "CSPLIN", "CSPLINzero"):
        # Parse windows for TENT/CSPLIN models
        try:
            if args.tent_window is not None:
                tent_windows = parse_tent_windows(args.tent_window, n_conditions)
            elif args.duration is not None:
                # Use duration for all conditions
                tent_windows = [(0.0, args.duration)] * n_conditions
            else:
                print(
                    f"ERROR: Either -tent_window or -duration required for {model} model",
                    file=sys.stderr,
                )
                return 1
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        # Print summary
        if args.verbose and tent_windows is not None:
            # Check if all windows are the same
            if len(set(tent_windows)) == 1:
                bot, top = tent_windows[0]
                if args.tent_n_basis is None:
                    n_basis_calc = round((top - bot) / tr) + 1
                else:
                    n_basis_calc = args.tent_n_basis
                n_actual = n_basis_calc - 2 if model in ("TENTzero", "CSPLINzero") else n_basis_calc
                print(f"  Window: {bot}s to {top}s (all conditions)")
                basis_type = "cubic spline" if "CSPLIN" in model else "tent"
                print(
                    f"  Basis functions: {n_basis_calc} {basis_type} knots → {n_actual} regressors per condition"
                )
            else:
                print("  Windows (per condition):")
                basis_type = "cubic spline" if "CSPLIN" in model else "tent"
                for i, (bot, top) in enumerate(tent_windows):
                    if args.tent_n_basis is None:
                        n_basis_calc = round((top - bot) / tr) + 1
                    else:
                        n_basis_calc = args.tent_n_basis
                    n_actual = n_basis_calc - 2 if model in ("TENTzero", "CSPLINzero") else n_basis_calc
                    print(
                        f"    {condition_labels[i]}: {bot}s to {top}s ({n_basis_calc} {basis_type} knots → {n_actual} regressors)"
                    )

    # Build design matrices
    if args.verbose:
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
            # FIR: same n_lags for all conditions
            design = build_glm_design(
                onsets=onsets_binary,
                mode="fir",
                n_fir_lags=n_lags,
                tr=tr,
                device=device,
            )
            if run_idx == 0:
                n_basis_per_condition_list = [n_lags] * n_conditions

        elif model in ("TENT", "TENTzero", "CSPLIN", "CSPLINzero"):
            # TENT/CSPLIN: use exact onset times (not binary matrix)
            # This produces fractional weights for non-TR-locked onsets
            if tent_windows is None:
                raise RuntimeError(f"tent_windows should not be None for {model} model")

            cond_designs = []
            for cond_idx in range(n_conditions):
                # Get onset times (in seconds) for this condition in this run
                onset_times = run_onsets_all_conds[cond_idx]

                # Debug: check if this condition has any onsets
                if args.verbose and run_idx == 0:
                    print(
                        f"    Condition {cond_idx + 1} ({condition_labels[cond_idx]}): {len(onset_times)} onsets in run 1"
                    )

                # Get window for this condition
                bot, top = tent_windows[cond_idx]

                # Build design for this condition using exact onset times
                if model in ("TENT", "TENTzero"):
                    from fastfuncsim.design.matrices import make_tent_design
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
                    from fastfuncsim.design.matrices import make_csplin_design
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
            if args.verbose and run_idx == 0:
                print(f"  DEBUG: Per-condition designs for run {run_idx + 1}:")
                for i, d in enumerate(cond_designs):
                    print(f"    Condition {i + 1}: {d.shape}")

            design = torch.cat(cond_designs, dim=1)

            if args.verbose and run_idx == 0:
                print(f"  DEBUG: Concatenated design for run {run_idx + 1}: {design.shape}")

        design_list.append(design)

    # Concatenate designs across runs
    design_full = torch.cat(design_list, dim=0)

    n_stimulus_regressors = design_full.shape[1]

    if args.verbose:
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

        if args.verbose:
            print(f"  Polynomial drift: {total_poly_cols} regressors ({n_poly_per_run} per run x {len(n_timepoints_per_run)} runs)")

    # Save design matrix if requested
    if args.save_design:
        design_file = f"{args.prefix}_design.1D"
        design_np = design_full.cpu().numpy()

        if args.verbose:
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

        if args.verbose:
            print(f"  ✓ Design matrix saved ({design_np.shape[0]} rows x {design_np.shape[1]} cols)")

    # Save design matrix plot if requested
    if args.save_design_plot:
        # Create figures directory
        figs_dir = f"{args.prefix}_figures"
        Path(figs_dir).mkdir(parents=True, exist_ok=True)
        design_plot_file = f"{figs_dir}/design.png"
        design_np = design_full.cpu().numpy()

        if args.verbose:
            print("\nSaving design matrix plot...")
            print(f"  {design_plot_file}")

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            # Create figure (time on vertical axis)
            fig, ax = plt.subplots(figsize=(10, 12))

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

            if args.verbose:
                print("  ✓ Design matrix plot saved")

        except ImportError:
            print("WARNING: matplotlib not available, skipping design plot", file=sys.stderr)

    # Prepare data
    if args.verbose:
        print("\nPreparing data for GLM...")

    # Concatenate data across runs
    data_full = np.concatenate([d for d in data_list], axis=3)

    # Apply mask if provided
    # fit_glm expects (n_voxels, n_timepoints)
    if mask is not None:
        data_masked = data_full[mask, :]  # Shape: (n_voxels, n_timepoints)
        if args.verbose:
            print(f"  Data shape: {data_masked.shape}")
    else:
        data_masked = data_full.reshape(
            -1, sum(n_timepoints_per_run)
        )  # Shape: (n_voxels, n_timepoints)
        if args.verbose:
            print(f"  Data shape: {data_masked.shape} (all voxels)")

    # Convert to CPU tensor (will be chunked to GPU during fitting)
    data_tensor = torch.tensor(data_masked, dtype=torch.float32, device="cpu")

    # Fit GLM with chunking
    if args.verbose:
        print("\nFitting GLM (chunked for GPU memory)...")
        print(f"  Design: {design_full.shape}")
        print(f"  Data: {data_tensor.shape}")

    # TODO estimate chunk size based on GPU memroy, n_voxels * timepoints * 4 bytes, and design matrix size

    results = fit_glm(
        data=data_tensor,
        design=design_full,
        tr=tr,
        device=device,
        preload_data_to_device=False,  # Stream chunks to GPU
        chunk_size=60000,  # Voxels per chunk
        verbose=args.verbose,
    )

    if args.verbose:
        print("  ✓ GLM fit complete")

    # Extract HRF estimates (only stimulus betas, not polynomials)
    if args.verbose:
        print("\nExtracting HRF estimates...")

    betas_stimulus = results.betas[:, :n_stimulus_regressors].cpu().numpy()

    # Save iresp files (per-condition, since they may have different n_basis)
    if args.verbose:
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
            betas_4d = betas_cond.T.reshape(nx, ny, nz, n_basis)

        # Add condition dimension: (nx, ny, nz, 1, n_basis)
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
        )
        output_files.extend(files)

    if args.verbose:
        for f in output_files:
            print(f"  ✓ {f}")
    else:
        print("Created HRF estimate files:")
        for f in output_files:
            print(f"  {f}")

    # Save beta coefficients if requested
    if args.save_betas:
        if args.verbose:
            print("\nSaving beta coefficients...")

        # Reshape betas back to 4D
        betas_4d = np.zeros((nx, ny, nz, n_stimulus_regressors))
        if mask is not None:
            betas_4d[mask, :] = betas_stimulus
        else:
            betas_4d = betas_stimulus.T.reshape(nx, ny, nz, n_stimulus_regressors)

        # Save as 4D NIfTI
        beta_file = f"{args.prefix}_betas.nii.gz"
        save_nifti(betas_4d, beta_file, reference_img=args.input[0])

        if args.verbose:
            print(f"  ✓ {beta_file}")

    # Done
    if args.verbose:
        print(f"\n{'=' * 70}")
        print("✓ Deconvolution complete!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

