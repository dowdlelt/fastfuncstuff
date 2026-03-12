#!/usr/bin/env python3
"""
3dREMLfast - Fast ARMA(1,1) GLM for fMRI using GPU acceleration

This is a PyTorch/GPU-accelerated implementation of AFNI's 3dREMLfit,
providing 5-50x speedup for ARMA(1,1) prewhitened GLM fitting.

Basic usage:
    3dREMLfast -input func.nii.gz -matrix X.xmat.1D -Rnuisance stats_REML

For help:
    3dREMLfast -help
"""

import argparse
import sys
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
        load_nifti,
        read_afni_design_matrix,  # noqa: F401 — re-imported in sub-function but also used at module scope
        replace_afni_extension,
    )
    from fastfuncsim.analysis import analyze_from_design_matrix
    from fastfuncsim.cli_utils import (
        auto_polort,
        build_nuisance_block_diag,
        compute_run_lengths,
        get_average_run_duration,
        parse_device_arg,
        parse_input_files,
    )
    from fastfuncsim.design.builder import (
        create_onset_matrix_microtime,
        parse_afni_timing_file,
        parse_durations,
    )
    from fastfuncsim.glm.outputs import (
        slice_glm_results,
        write_glm_bucket_as_nifti,
    )
    from fastfuncsim.utils import gaussian_blur_3d, scale_to_percent_signal
except ImportError as e:
    print(f"ERROR: Could not import fastfuncsim: {e}")
    print("Make sure fastfuncsim is installed: pip install -e .")
    sys.exit(1)


def parse_grid_arg(grid_str: str) -> tuple[float, float, int]:
    """Parse grid argument like '0.1,0.9,11' into (start, stop, num_points)"""
    try:
        parts = grid_str.split(",")
        if len(parts) != 3:
            raise ValueError("Grid must have exactly 3 values: start,stop,num_points")
        start = float(parts[0])
        stop = float(parts[1])
        num_points = int(parts[2])
        if num_points < 2:
            raise ValueError("num_points must be >= 2")
        return start, stop, num_points
    except ValueError as e:
        print(f"ERROR parsing grid '{grid_str}': {e}")
        sys.exit(1)



def detect_format(filepath: str) -> str:
    """Detect file format from extension"""
    p = Path(filepath)
    if p.suffix == ".gz" and p.stem.endswith(".nii"):
        return "nii.gz"
    elif p.suffix in [".nii"]:
        return "nii"
    elif "+orig" in p.name or "+tlrc" in p.name:
        return "afni"
    else:
        # Default to nifti
        return "nii.gz"


def extract_onset_times_from_design(
    design_matrix: np.ndarray, column_indices: list
) -> list:
    """
    Extract onset times for stimulus columns from design matrix.

    For each stimulus column, finds the first timepoint where the column becomes non-zero.
    This represents the onset time of that stimulus.

    Parameters
    ----------
    design_matrix : np.ndarray
        Design matrix (n_timepoints, n_regressors)
    column_indices : list of int
        Column indices to extract onset times for

    Returns
    -------
    onset_times : list of int
        Onset timepoint for each column (same length as column_indices)
    """
    onset_times = []

    for col_idx in column_indices:
        column = design_matrix[:, col_idx]

        # Find first non-zero timepoint
        nonzero_indices = np.nonzero(column)[0]

        if len(nonzero_indices) > 0:
            onset_time = int(nonzero_indices[0])
        else:
            # Column is all zeros - use large value to sort to end
            onset_time = len(column) + col_idx  # Add col_idx to maintain stable sort

        onset_times.append(onset_time)

    return onset_times


def write_single_trials_output(
    results,
    output_path: str,
    design_matrix: np.ndarray,
    stim_indices: list,
    stim_labels: list,
):
    """
    Write single-trial betas reordered by presentation time.

    Parameters
    ----------
    results : GLMResults or ARMA11Results
        Results object with betas attribute
    output_path : str
        Output file path (e.g., "ols_single.nii.gz")
    design_matrix : np.ndarray
        Full design matrix (n_timepoints, n_regressors)
    stim_indices : list of int
        Column indices for stimulus regressors
    stim_labels : list of str
        Labels for stimulus regressors
    """
    import nibabel as nib

    from fastfuncsim.glm.outputs import (
        _ensure_numpy,
        _get_voxel_mask,
        _reshape_parameter_map,
        _resolve_shape,
    )

    # Extract onset times for each stimulus column
    onset_times = extract_onset_times_from_design(design_matrix, stim_indices)

    # Create sort order (sorts by onset time, maintaining stable order for ties)
    sort_indices = sorted(range(len(onset_times)), key=lambda i: (onset_times[i], i))

    # Reorder betas by onset time
    betas_np = _ensure_numpy(results.betas)
    betas_reordered = betas_np[:, sort_indices]  # (n_voxels, n_stimuli)

    # Reorder labels
    labels_reordered = [stim_labels[i] for i in sort_indices] if stim_labels else None

    # Reshape to volume
    affine = getattr(results, "affine", np.eye(4))
    volume_shape = _resolve_shape(results, None)
    voxel_mask = _get_voxel_mask(results)
    betas_vol = _reshape_parameter_map(betas_reordered, volume_shape, voxel_mask)

    # Write NIfTI file
    img = nib.Nifti1Image(betas_vol, affine)
    output_path_clean = replace_afni_extension(output_path, ".nii.gz")
    nib.save(img, output_path_clean)

    # Write labels as JSON sidecar
    if labels_reordered:
        import json
        from pathlib import Path

        json_path = Path(output_path_clean).with_suffix(".json")
        with json_path.open("w") as f:
            json.dump(
                {
                    "Description": "Single-trial betas reordered by presentation time (onset order)",
                    "Labels": labels_reordered,
                    "OnsetTimes": [onset_times[i] for i in sort_indices],
                    "OriginalColumnIndices": [stim_indices[i] for i in sort_indices],
                },
                f,
                indent=2,
            )

    return output_path_clean


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults while preserving raw description formatting."""


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="3dREMLfast - Fast GPU-accelerated ARMA(1,1) GLM fitting",
        formatter_class=_HelpFormatter,
        epilog="""
Examples:
  # Basic REML analysis with main bucket output
  3dREMLfast -input func.nii.gz -matrix X.xmat.1D -Rbuck stats_REML
  
  # Multiple runs
  3dREMLfast -input "run1.nii.gz run2.nii.gz run3.nii.gz" \\
             -matrix X.xmat.1D -Rbuck stats_REML
  
  # With all outputs
  3dREMLfast -input func.nii.gz -matrix X.xmat.1D \\
             -Rbuck stats_REML -Rvar params_REML \\
             -Rfitts fitts_REML -Rerrts errts_REML \\
             -Rbeta betas_only_REML -fout -tout -rout
  
  # With OLS comparison
  3dREMLfast -input func.nii.gz -matrix X.xmat.1D \\
             -Rbuck stats_REML -Obuck stats_OLS
  
  # Nuisance regressors only
  3dREMLfast -input func.nii.gz -matrix X.xmat.1D \\
             -Rnuisance nuisance_REML -Onuisance nuisance_OLS

  # Single-trial outputs (reordered by onset time)
  3dREMLfast -input func.nii.gz -matrix X.xmat.1D \\
             -Rbuck stats_REML -single_trials movie
  # Creates: ols_movie_single.nii.gz, reml_movie_single.nii.gz

  # Custom ARMA grid with double precision
  3dREMLfast -input func.nii.gz -matrix X.xmat.1D \\
             -Rbuck stats_REML -use_double \\
             -a_grid 0.0,0.9,10 -b_grid -0.8,0.8,17

  # Manual batch size control (for memory tuning)
  3dREMLfast -input func.nii.gz -matrix X.xmat.1D \\
             -Rbuck stats_REML -batch_size 2000
        """,
    )

    # Required arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-input",
        nargs="+",
        required=True,
        help="Input fMRI dataset(s). Can be single file, multiple files, or glob patterns (e.g., run*.nii.gz)",
    )
    required.add_argument(
        "-matrix",
        required=False,
        help="Design matrix file (X.xmat.1D from 3dDeconvolve). Mutually exclusive with -onsets.",
    )

    # Output arguments - REML
    reml_out = parser.add_argument_group("REML Output Options")
    reml_out.add_argument(
        "-Rvar",
        help="Output REML variance parameters (up to 4 volumes: a, b, lambda, StDev)",
    )
    reml_out.add_argument(
        "-load_Rvar",
        help="Load precomputed ARMA parameters from this file to skip the grid search "
             "(saves ~80%% compute time on re-runs). Must be a -Rvar output from a "
             "previous run. If not specified, the grid search always runs from scratch.",
    )
    reml_out.add_argument(
        "-Rbuck", help="Output REML betas + statistics (main bucket output)"
    )
    reml_out.add_argument("-Rbeta", help="Output REML betas only (no statistics)")
    reml_out.add_argument(
        "-Rnuisance", help="Output REML betas + statistics for NUISANCE regressors only"
    )
    reml_out.add_argument("-Rfitts", help="Output REML fitted model time series")
    reml_out.add_argument("-Rerrts", help="Output REML residuals")
    reml_out.add_argument("-Rwherr", help="Output REML whitened residuals")

    # Output arguments - OLS
    ols_out = parser.add_argument_group("OLS Output Options (for comparison)")
    ols_out.add_argument(
        "-Obuck", help="Output OLS betas + statistics (main bucket output)"
    )
    ols_out.add_argument("-Obeta", help="Output OLS betas only (no statistics)")
    ols_out.add_argument(
        "-Onuisance", help="Output OLS betas + statistics for NUISANCE regressors only"
    )

    # Special output options
    special_out = parser.add_argument_group("Special Output Options")
    special_out.add_argument(
        "-single_trials",
        type=str,
        default=None,
        metavar="LABEL",
        help=(
            "Output single-trial betas reordered by presentation time (onset order). "
            "LABEL will be inserted into filenames: ols_LABEL_single.nii.gz and reml_LABEL_single.nii.gz. "
            "Trials are sorted chronologically by onset time instead of by column order. "
            "Only includes stimulus columns."
        ),
    )

    # Statistics options
    stats_opts = parser.add_argument_group("Statistics Options")
    stats_opts.add_argument(
        "-fout", action="store_true", help="Include F-statistics in output buckets"
    )
    stats_opts.add_argument(
        "-tout", action="store_true", help="Include t-statistics in output buckets"
    )
    stats_opts.add_argument(
        "-rout",
        action="store_true",
        help="Include R² statistics in output buckets (total model R²)",
    )
    stats_opts.add_argument(
        "-rpartial",
        nargs="?",
        const="full",
        choices=["full", "task"],
        help="Include partial R² per condition in output buckets. "
        "'full' (default): partial R² as proportion of total variance. "
        "'task': partial R² as proportion of variance remaining after nuisance regressors (more interpretable for task effects). "
        "NOTE: Partial R² values do NOT sum to total R² (they sum to MORE due to shared variance between regressors).",
    )
    stats_opts.add_argument(
        "-r2semipartial",
        nargs="?",
        const="full",
        choices=["full", "task"],
        help="Include semi-partial R² (squared part correlation) per condition in output buckets. "
        "'full' (default): semi-partial R² as proportion of total variance. "
        "'task': semi-partial R² as proportion of variance remaining after nuisance regressors. "
        "Semi-partial R² shows unique variance contribution and DOES sum to total R² (additive contributions). "
        "Formula: r²_semi = (R²_full - R²_without_regressor)",
    )
    stats_opts.add_argument(
        "-beta_cv",
        action="store_true",
        help="Use beta-space cross-validation (GLMsingle-style). "
        "Fits single-trial model once on all data with ARMA(1,1) prewhitening, "
        "evaluates R² on condition-averaged vs individual trial betas across folds. "
        "Requires -onsets (not -matrix). "
        "NOTE: This is different from -single_trials (which reorders output by onset time).",
    )

    # ARMA grid options
    arma_opts = parser.add_argument_group("ARMA(1,1) Grid Options")
    arma_opts.add_argument(
        "-a_grid", help="AR parameter grid: start,stop,num_points (e.g., 0.0,0.9,10)"
    )
    arma_opts.add_argument(
        "-b_grid", help="MA parameter grid: start,stop,num_points (e.g., -0.8,0.8,17)"
    )
    arma_opts.add_argument(
        "-grid_batching",
        action="store_true",
        help=(
            "Force grid batching mode (low memory, slightly slower). "
            "Processes all voxels for each (a,b) pair instead of precomputing the full grid. "
            "Memory: ~3 GB regardless of grid size. "
            "Default: auto-detect (uses grid batching if grid > 8 GB). "
            "Best for: long timeseries, double precision, limited GPU memory."
        ),
    )
    arma_opts.add_argument(
        "-no_grid_batching",
        action="store_true",
        help=(
            "Force full grid precomputation (AFNI approach, faster but more memory). "
            "Precomputes all Cholesky factorizations once, then reuses for all voxels. "
            "Memory: can be 10+ GB with long timeseries and double precision. "
            "Default: auto-detect (uses full grid if grid ≤ 8 GB). "
            "Best for: short timeseries, float32, abundant GPU memory."
        ),
    )
    arma_opts.add_argument(
        "-quick_estimate",
        action="store_true",
        help=(
            "EXPERIMENTAL: Enable fast grid search with early stopping (GPU only). "
            "Uses smart ordering + batch convergence detection to stop early. "
            "Can be 2-3x faster but may miss true optima for some voxels. "
            "Default: exhaustive search (recommended for publication). "
            "Use this flag ONLY for exploratory analysis or when speed is critical."
        ),
    )

    # Processing options
    proc_opts = parser.add_argument_group("Processing Options")
    proc_opts.add_argument(
        "-use_double",
        action="store_true",
        help="Use double precision (float64) - matches AFNI exactly, ~2x memory, ~1.5x slower",
    )
    proc_opts.add_argument("-mask", help="Mask file to restrict analysis")
    proc_opts.add_argument(
        "-do_scale",
        action="store_true",
        help="Scale each voxel per run to mean=100 (percent signal change units). "
        "Values are clipped to max 200 (100%% increase from mean).",
    )
    proc_opts.add_argument(
        "-do_blur",
        type=float,
        metavar="FWHM",
        default=None,
        help="Apply 3D Gaussian spatial smoothing with FWHM in mm. "
        "Smoothing is applied BEFORE masking to avoid edge effects. "
        "Typical values: 4-8 mm.",
    )
    proc_opts.add_argument(
        "-cache",
        metavar="FILE.h5",
        help="HDF5 cache file for fast data loading. If exists, loads from cache. If not, creates cache from input files.",
    )
    proc_opts.add_argument(
        "-test",
        type=int,
        metavar="N_VOXELS",
        help="Test mode: extract ~N voxels from center of volume (fast iteration for debugging)",
    )
    proc_opts.add_argument(
        "-batch_size",
        type=int,
        help="Number of voxels per batch for ARMA grid search (default: auto-detect). OLS will use 4x this value.",
    )
    proc_opts.add_argument(
        "-force_format",
        choices=["nii", "nii.gz", "afni"],
        help="Force output format (default: match input)",
    )
    proc_opts.add_argument(
        "-device",
        type=str,
        help=(
            "Force device (default: auto-detect GPU). "
            "Format: 'cpu' or 'cuda' for auto-config, "
            "'cpu,N' to use N CPU threads, "
            "'cuda,N' to use GPU device N (e.g., 'cuda,0' for GPU 0)"
        ),
    )
    proc_opts.add_argument(
        "-verbose", action="store_true", help="Print detailed progress information"
    )
    proc_opts.add_argument(
        "-legacy_contrasts",
        action="store_true",
        help="Use legacy loop-based GLT contrast computation (slower, for validation only)",
    )
    proc_opts.add_argument(
        "-debug_memory",
        action="store_true",
        help="Print detailed memory profiling at every step (for debugging)",
    )

    # Onset-based design (alternative to -matrix)
    onset_group = parser.add_argument_group("Onset-Based Design (alternative to -matrix)")
    onset_group.add_argument(
        "-onsets",
        nargs="+",
        help="Onset timing files (AFNI format). One per condition. Mutually exclusive with -matrix.",
    )
    onset_group.add_argument(
        "-durations",
        nargs="+",
        help="Stimulus durations. Single value, per-condition, or 'value,count'.",
    )
    onset_group.add_argument(
        "-microtime_dt",
        type=float,
        default=0.1,
        help="Microtime resolution (seconds, default: 0.1)",
    )
    onset_group.add_argument(
        "-polort",
        type=int,
        default=None,
        help="Polynomial order (default: auto based on run length)",
    )
    onset_group.add_argument(
        "-ortvec",
        action="append",
        nargs=2,
        metavar=("FILE", "LABEL"),
        help="Nuisance regressors (can repeat). FILE LABEL",
    )
    onset_group.add_argument(
        "-canonical",
        type=str,
        default=None,
        help="(DEPRECATED) Use -hrf_model instead. Canonical HRF: 'spmg1' or 'glmsingle'",
    )
    onset_group.add_argument(
        "-hrf_model",
        type=str,
        default="spmg1",
        help="HRF model: 'spmg1' (default), 'spmg2', 'spmg3', 'glmsingle', 'FIR', 'TENT', or 'TENT(bot,top,n)'. "
             "SPMG2 = canonical + temporal derivative. SPMG3 = canonical + time + dispersion derivatives. "
             "FIR/TENT use durations to set window (default: 0 to max(durations)). "
             "Example: 'TENT(0,15,6)' for 6 tent basis functions from 0-15s",
    )

    # Help
    parser.add_argument("-help", action="store_true", help="Show this help message")

    return parser


def print_header(args):
    """Print program header"""
    from datetime import datetime

    print("=" * 70)
    print("3dREMLfast - GPU-Accelerated ARMA(1,1) GLM")
    print("=" * 70)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    if args.use_double:
        print("⚙️  Precision: DOUBLE (float64) - matches AFNI exactly")
    else:
        print("⚙️  Precision: SINGLE (float32) - default, faster")
    print()


def print_output_summary(args):
    """Print summary of requested outputs"""
    print("=" * 70)
    print("📋 Requested Outputs")
    print("=" * 70)

    # ARMA/REML outputs
    arma_outputs = []
    if args.Rbuck:
        arma_outputs.append(f"  • Rbuck (betas + stats): {args.Rbuck}")
    if args.Rbeta:
        arma_outputs.append(f"  • Rbeta (betas only): {args.Rbeta}")
    if args.Rnuisance:
        arma_outputs.append(f"  • Rnuisance (nuisance betas + stats): {args.Rnuisance}")
    if args.Rvar:
        arma_outputs.append(f"  • Rvar (ARMA parameters): {args.Rvar}")
    if args.Rfitts:
        arma_outputs.append(f"  • Rfitts (fitted model): {args.Rfitts}")
    if args.Rerrts:
        arma_outputs.append(f"  • Rerrts (residuals): {args.Rerrts}")
    if args.Rwherr:
        arma_outputs.append(f"  • Rwherr (whitened residuals): {args.Rwherr}")

    if arma_outputs:
        print("ARMA/REML Outputs:")
        for output in arma_outputs:
            print(output)
    else:
        print("ARMA/REML Outputs: None")

    print()

    # OLS outputs
    ols_outputs = []
    if args.Obuck:
        ols_outputs.append(f"  • Obuck (betas + stats): {args.Obuck}")
    if args.Obeta:
        ols_outputs.append(f"  • Obeta (betas only): {args.Obeta}")
    if args.Onuisance:
        ols_outputs.append(f"  • Onuisance (nuisance betas + stats): {args.Onuisance}")

    if ols_outputs:
        print("OLS Outputs:")
        for output in ols_outputs:
            print(output)
    else:
        print("OLS Outputs: None")

    print()

    # Special outputs
    special_outputs = []
    if args.single_trials:
        label = args.single_trials
        special_outputs.append(
            f"  • Single-trial betas (onset order): ols_{label}_single.nii.gz, reml_{label}_single.nii.gz"
        )

    if special_outputs:
        print("Special Outputs:")
        for output in special_outputs:
            print(output)
        print()

    # Statistics flags
    stat_flags = []
    if args.fout:
        stat_flags.append("F-statistics")
    if args.tout:
        stat_flags.append("t-statistics")
    if args.rout:
        stat_flags.append("R² statistics")

    if stat_flags:
        print(f"Statistics: {', '.join(stat_flags)}")
    else:
        print("Statistics: Default (F-statistics only)")

    print("=" * 70)
    print()


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Show help if requested
    if args.help or len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Validate design input: must have either -matrix or -onsets (but not both)
    if args.matrix and args.onsets:
        print("ERROR: Specify either -matrix or -onsets, not both")
        sys.exit(1)
    if not args.matrix and not args.onsets:
        print("ERROR: Must specify either -matrix or -onsets")
        sys.exit(1)

    # Check that at least one output is requested
    outputs = [
        args.Rvar,
        args.Rbuck,
        args.Rbeta,
        args.Rnuisance,
        args.Rfitts,
        args.Rerrts,
        args.Rwherr,
        args.Obuck,
        args.Obeta,
        args.Onuisance,
    ]
    if not any(outputs):
        print("ERROR: At least one output option must be specified")
        print(
            "       Use -Rbuck, -Rbeta, -Rnuisance, -Rvar, -Rfitts, -Rerrts, -Rwherr,"
        )
        print("       -Obuck, -Obeta, or -Onuisance")
        sys.exit(1)

    print_header(args)

    # Parse input files
    input_files = parse_input_files(args.input)
    print(f"📁 Input files: {len(input_files)} file(s)")
    for f in input_files:
        print(f"   • {f}")
    print()

    # Get TR from first file
    tr = get_tr_from_file(input_files[0])
    print(f"⏱️  TR: {tr:.3f} seconds")
    print()

    # Detect input format (for informational purposes only)
    # NOTE: All outputs are written as NIfTI .nii.gz regardless of input format
    if args.force_format:
        output_format = args.force_format  # Keep var name for compatibility
    else:
        output_format = detect_format(input_files[0])
    print(f"📥 Input format detected: {output_format}")
    print(
        "📤 Output format: NIfTI (.nii.gz) - all outputs written as compressed NIfTI"
    )
    print()

    # Print summary of requested outputs
    print_output_summary(args)

    # Parse device specification using shared utility
    import os

    device, cpu_threads_override, cuda_device_id = parse_device_arg(args.device)

    # Configure CPU threading for maximum performance
    if device.type == "cpu":
        try:
            import psutil

            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = os.cpu_count() or 12

            # Determine number of threads to use
            if cpu_threads_override is not None:
                # User explicitly specified thread count
                num_threads = cpu_threads_override
                thread_source = "user-specified"
            else:
                # Auto-detect: use physical cores for compute efficiency
                num_threads = physical_cores or logical_cores
                thread_source = (
                    f"physical cores ({logical_cores} logical with hyperthreading)"
                )

            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
            # Also set environment variables for MKL/OpenMP
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
            os.environ["MKL_NUM_THREADS"] = str(num_threads)

            print(f"🖥️  Device: {device}")
            print(f"⚡ CPU threads: {num_threads} ({thread_source})")
        except ImportError:
            # Fallback if psutil not available
            num_threads = (
                cpu_threads_override
                if cpu_threads_override is not None
                else (os.cpu_count() or 12)
            )
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
            os.environ["MKL_NUM_THREADS"] = str(num_threads)
            print(f"🖥️  Device: {device}")
            print(f"⚡ CPU threads: {num_threads}")
    else:
        print(f"🖥️  Device: {device}")
    print()

    # Parse ARMA grids if provided
    a_grid = None
    b_grid = None
    if args.a_grid:
        start, stop, num = parse_grid_arg(args.a_grid)
        a_grid = torch.linspace(start, stop, num, device=device)
        print(f"🔢 Custom a_grid: [{start}, {stop}] with {num} points")
    if args.b_grid:
        start, stop, num = parse_grid_arg(args.b_grid)
        b_grid = torch.linspace(start, stop, num, device=device)
        print(f"🔢 Custom b_grid: [{start}, {stop}] with {num} points")
    if args.a_grid or args.b_grid:
        print()

    # Print batch size if specified
    if args.batch_size:
        print(f"📦 Batch size: {args.batch_size:,} voxels per batch")
        print()

    # Determine if we need OLS (any OLS output requested)
    want_ols = (
        args.Obuck is not None or args.Obeta is not None or args.Onuisance is not None
    )
    ols_output_path = args.Obuck if args.Obuck else None

    # Load or build design matrix
    from fastfuncsim.io.afni import read_afni_design_matrix

    if args.matrix:
        # Load design matrix from file
        design_info = read_afni_design_matrix(args.matrix)
    else:
        # Build design matrix from onsets
        print()
        print("=" * 70)
        print("Building design matrix from onsets")
        print("=" * 70)
        print()

        # Parse onset files
        print("Parsing onset files...")
        all_onsets = []
        condition_labels = []
        for onset_file in args.onsets:
            condition_label = Path(onset_file).stem
            runs_onsets = parse_afni_timing_file(onset_file)
            all_onsets.append(runs_onsets)
            condition_labels.append(condition_label)
            n_events = sum(len(run_onsets) for run_onsets in runs_onsets)
            print(f"  {condition_label}: {n_events} events across {len(runs_onsets)} runs")

        n_conditions = len(all_onsets)
        n_runs = len(all_onsets[0])  # All conditions must have same number of runs

        # Parse durations
        print()
        print("Parsing durations...")
        durations = parse_durations(args.durations, n_conditions, condition_labels)
        if len(args.durations) == 1 and ',' not in args.durations[0]:
            print(f"  Using {durations[0]}s for all {n_conditions} conditions")
        else:
            print(f"  Matched {len(durations)} durations to {n_conditions} conditions")

        # Parse HRF model arguments
        print()
        from fastfuncsim.cli_utils import parse_hrf_model_args

        hrf_info = parse_hrf_model_args(
            hrf_model_arg=args.hrf_model,
            canonical_arg=args.canonical,
            durations=durations,
            condition_labels=condition_labels,
            tr=args.tr,
        )

        hrf_model_name = hrf_info["hrf_model_name"]
        is_fir_model = hrf_info["is_fir_model"]
        fir_bot = hrf_info["fir_bot"]
        fir_top = hrf_info["fir_top"]
        n_basis = hrf_info["n_basis"]
        condition_labels_full = hrf_info["condition_labels_full"]

        # 3dREMLfast doesn't support -hrf_opt or -single_trial, so no compatibility check needed
        # (FIR is compatible with ARMA modeling)

        # Compute run starts from input files
        print()
        print("Computing run structure...")
        run_starts = [0]
        total_tps = 0
        for f in input_files:
            img_shape = load_nifti(f).shape
            run_len = img_shape[3] if len(img_shape) > 3 else img_shape[0]
            total_tps += run_len
            if f != input_files[-1]:  # Don't add start for after last run
                run_starts.append(total_tps)
        n_timepoints = total_tps

        print(f"  Runs: {n_runs}")
        print(f"  Total timepoints: {n_timepoints}")
        print(f"  Run starts (TRs): {run_starts}")

        # Auto-determine polort if not specified
        if args.polort is None:
            run_lengths = compute_run_lengths(run_starts, n_timepoints)
            avg_run_duration_sec = get_average_run_duration(run_lengths, tr)
            polort = auto_polort(avg_run_duration_sec, formula="afni")
            print(f"  Auto polort: {polort} (based on {avg_run_duration_sec/60:.1f} min avg run)")
        else:
            polort = args.polort
            print(f"  Polort: {polort}")

        # Build microtime onset matrix
        print()
        print("Building onset matrix at microtime resolution...")
        onset_matrix_micro = create_onset_matrix_microtime(
            all_onsets, run_starts, tr, n_timepoints,
            args.microtime_dt, durations, device)
        print(f"  Onset matrix shape: {onset_matrix_micro.shape}")

        # Build task design matrix using refactored function
        print()
        from fastfuncsim.cli_utils import build_task_design_from_args

        task_design, designs_by_hrf = build_task_design_from_args(
            hrf_model_name=hrf_model_name,
            is_fir_model=is_fir_model,
            fir_bot=fir_bot,
            fir_top=fir_top,
            n_basis=n_basis,
            all_onsets=all_onsets,
            onset_matrix_micro=onset_matrix_micro,
            n_conditions=n_conditions,
            n_timepoints=n_timepoints,
            run_starts=run_starts,
            tr=args.tr,
            microtime_dt=args.microtime_dt,
            device=device,
            hrf_opt=None,  # 3dREMLfast doesn't support per-voxel HRFs
            hrf_library=None,
            hrf_indices=None,
            n_voxels=None,
        )

        assert task_design is not None, "3dREMLfast requires single design (no per-voxel HRFs)"
        print(f"  Task design shape: {task_design.shape}")

        # Build nuisance design (polynomials + ortvec)
        print()
        print("Building nuisance regressors...")
        nuisance_design = build_nuisance_block_diag(
            run_starts=run_starts,
            n_timepoints=n_timepoints,
            polort=polort,
            device=device,
            ortvec_files=args.ortvec,
            verbose=True,
        )

        # Build full design: [task | nuisance]
        full_design = torch.cat([task_design, nuisance_design], dim=1)
        print()
        print(f"Full design matrix shape: {full_design.shape}")
        print(f"  Task columns: {task_design.shape[1]}")
        print(f"  Nuisance columns: {nuisance_design.shape[1]}")

        # Create design_info dict to match format from read_afni_design_matrix
        task_indices = list(range(task_design.shape[1]))
        nuisance_indices = list(range(task_design.shape[1], full_design.shape[1]))

        # Build column labels
        column_labels = []
        if is_fir_model:
            # FIR: condition_labels_full already has expanded labels (e.g., "cond1_t0.0s", "cond1_t1.5s", ...)
            # Add #0 suffix to match AFNI format
            for cond_label in condition_labels_full:
                column_labels.append(f"{cond_label}#0")
        else:
            # Canonical: condition_labels_full = condition_labels (no expansion)
            for cond_label in condition_labels_full:
                column_labels.append(f"{cond_label}#0")

        # Add nuisance labels
        _nuisance_label_offset = len(column_labels)
        for run_idx in range(n_runs):
            for p in range(polort + 1):
                column_labels.append(f"Run#{run_idx+1}Pol#{p}")

        if args.ortvec:
            for _, label in args.ortvec:
                column_labels.append(label)

        # Build stim_bots and stim_tops
        if is_fir_model:
            # FIR: each condition has n_basis columns
            stim_bots = []
            stim_tops = []
            for cond_idx in range(n_conditions):
                bot = cond_idx * n_basis
                top = (cond_idx + 1) * n_basis - 1
                stim_bots.append(bot)
                stim_tops.append(top)
        else:
            # Canonical: each condition has 1 column
            stim_bots = list(range(n_conditions))
            stim_tops = list(range(n_conditions))

        design_info = {
            "design_matrix": full_design.cpu().numpy(),
            "column_labels": column_labels,
            "stim_indices": task_indices,
            "nuisance_indices": nuisance_indices,
            "stim_bots": stim_bots,
            "stim_tops": stim_tops,
            "stim_labels": condition_labels,  # Use original (unexpanded) labels
            "run_starts": run_starts,
            "n_runs": n_runs,
        }

        print()
        print("=" * 70)
        print()


    # ========== SINGLE-TRIAL BETA-SPACE CV PATH ==========
    if args.beta_cv:
        from fastfuncsim.glm.outputs import save_single_trial_results
        from fastfuncsim.glm.ridge import create_single_trial_design
        from fastfuncsim.glm.xval import compute_xval_r2_single_trials, generate_cv_splits

        print()
        print("=" * 70)
        print("Using beta-space cross-validation with ARMA(1,1) prewhitening")
        print("=" * 70)
        print()

        # Validate that we're in onset-based mode
        if not args.onsets:
            print("ERROR: -beta_cv requires -onsets (not -matrix)")
            sys.exit(1)

        # 1. Build single-trial design
        print("Building single-trial design...")
        st_design, trial_labels, trial_cond_ids, trial_run_ids, cond_design = create_single_trial_design(
            onsets_by_condition=all_onsets,
            durations=durations,
            run_starts=run_starts,
            tr=args.tr,
            n_timepoints=n_timepoints,
            microtime_dt=args.microtime_dt,
            condition_labels=condition_labels,
            device=device,
        )
        print(f"  Single-trial design: {st_design.shape}")
        n_trials = st_design.shape[1]

        # 2. Build wide design: [single_trial | nuisance]
        # Note: design_info was already created with condition-level design
        # We need to rebuild it with single-trial design
        print("Building nuisance regressors for single-trial design...")
        nuisance_design_st = build_nuisance_block_diag(
            run_starts=run_starts,
            n_timepoints=n_timepoints,
            polort=polort,
            device=device,
            ortvec_files=args.ortvec,
            verbose=True,
        )

        full_design_st = torch.cat([st_design, nuisance_design_st], dim=1)
        task_indices_st = list(range(n_trials))
        nuisance_indices_st = list(range(n_trials, full_design_st.shape[1]))

        print(f"  Full design (wide): {full_design_st.shape}")
        print(f"    Task columns (single-trial): {n_trials}")
        print(f"    Nuisance columns: {len(nuisance_indices_st)}")

        # 3. Run ARMA(1,1) analysis on wide design
        print()
        print("Running ARMA(1,1) analysis on single-trial design...")

        # Create a temporary design_info for single-trial mode
        design_info_st = {
            "design_matrix": full_design_st.cpu().numpy(),
            "column_labels": trial_labels + [f"nuisance_{i}" for i in range(len(nuisance_indices_st))],
            "stim_indices": task_indices_st,
            "nuisance_indices": nuisance_indices_st,
            "run_starts": run_starts,
            "n_runs": n_runs,
        }

        # Load and prepare fMRI data
        # Note: Single-trial mode doesn't use preprocessing caching, just pass input files
        fmri_data_st = input_files

        # Call analyze_from_design_matrix with single-trial design
        # This will fit ARMA(1,1), prewhiten, and return single-trial betas
        results_st, _ = analyze_from_design_matrix(
            fmri_data=fmri_data_st,
            design_matrix=design_info_st["design_matrix"],
            stim_column_indices=task_indices_st,
            nuisance_column_indices=nuisance_indices_st,
            method="arma11",
            arma_a_grid=a_grid,
            arma_b_grid=b_grid,
            device=device,
            mask_file=args.mask,
            voxel_chunk_size=args.batch_size,
            use_double=args.use_double,
            verbose=args.verbose,
        )

        # 4. Extract single-trial betas
        st_betas = results_st.betas  # (n_voxels, n_trials)
        print(f"  Single-trial betas: {st_betas.shape}")

        # 5. Beta-space CV
        print()
        print("Computing beta-space cross-validated R²...")
        cv_splits = generate_cv_splits(n_runs, strategy=1)  # LORO
        xval = compute_xval_r2_single_trials(
            st_betas, trial_cond_ids, trial_run_ids, cv_splits,
            metric="cod", device=device, verbose=True)

        # 6. Save outputs
        print()
        print("Saving single-trial outputs...")

        # Get spatial metadata
        volume_shape = results_st.original_shape
        affine = results_st.affine
        voxel_mask = getattr(results_st, 'voxel_mask', None)

        output_files = save_single_trial_results(
            betas=st_betas,
            xval_r2=xval['r2'],
            trial_labels=trial_labels,
            trial_condition_ids=trial_cond_ids,
            trial_run_ids=trial_run_ids,
            condition_labels=condition_labels,
            output_prefix=args.prefix,
            volume_shape=volume_shape,
            affine=affine,
            voxel_mask=voxel_mask,
        )

        print()
        print("=" * 70)
        print("✅ 3dREMLfast (single-trial mode) Complete!")
        print("=" * 70)
        print(f"  Median beta-space R²: {xval['r2'].median():.4f}")
        print(f"  {xval['n_test_trials_total']} test trials across {xval['n_splits']} folds")
        print()
        for key, path in output_files.items():
            print(f"  {key}: {path}")
        print("=" * 70)

        # Early exit - don't run standard pipeline
        return

    # Setup OLS write callback if any OLS output is requested
    _ols_write_callback = None
    if want_ols:
        # Determine stat flags (default to -fout if none specified)
        _want_fstat = args.fout or (not args.tout and not args.rout)
        _want_tstat = args.tout
        _want_rstat = args.rout
        # Capture single_trials flag and design_info for callback
        want_single_trials = args.single_trials
        callback_design_info = design_info  # Capture in closure

        def write_ols_results(ols_results, original_shape, affine):
            """Write OLS results immediately after computation"""
            print("\n💾 Writing OLS outputs (before ARMA)...")

            # IMPORTANT: When task_indices is passed to fit_glm(), the OLS results
            # already contain ONLY the task regressors (stimulus columns).
            # Extract stimulus labels for proper labeling
            stim_bots = callback_design_info.get("stim_bots", [])
            stim_tops = callback_design_info.get("stim_tops", [])
            stim_indices = []
            if stim_bots and stim_tops:
                for bot, top in zip(stim_bots, stim_tops, strict=False):
                    stim_indices.extend(range(bot, top + 1))

            # Extract labels for stimulus columns only (not all 322 columns!)
            if stim_indices and "column_labels" in callback_design_info:
                stim_labels = [
                    callback_design_info["column_labels"][i] for i in stim_indices
                ]
            else:
                stim_labels = callback_design_info.get("column_labels")

            # Set spatial metadata on OLS results for writing
            ols_results.original_shape = original_shape
            ols_results.affine = affine

            if args.Obuck:
                print(f"  • Writing OLS betas + stats (bucket): {args.Obuck}")
                # Always write NIfTI .nii.gz regardless of input format
                # Results already contain only stimulus columns (252), use stim_labels
                contrast_names = getattr(ols_results, "contrast_labels", None)

                # Build contrast_results dict if we have contrasts
                ols_contrast_results = None
                if (
                    hasattr(ols_results, "contrast_betas")
                    and ols_results.contrast_betas is not None
                ):
                    ols_contrast_results = {
                        "contrast_betas": ols_results.contrast_betas,
                        "contrast_tstats": ols_results.contrast_tstats,
                    }
                    # Add partial R² if available and requested
                    if (
                        hasattr(ols_results, "contrast_r2_partial")
                        and ols_results.contrast_r2_partial is not None
                    ):
                        ols_contrast_results["contrast_r2_partial"] = (
                            ols_results.contrast_r2_partial
                        )
                    # Add semi-partial R² if available and requested
                    if (
                        hasattr(ols_results, "contrast_r2_semipartial")
                        and ols_results.contrast_r2_semipartial is not None
                    ):
                        ols_contrast_results["contrast_r2_semipartial"] = (
                            ols_results.contrast_r2_semipartial
                        )

                write_glm_bucket_as_nifti(
                    ols_results,
                    args.Obuck,
                    condition_names=stim_labels,  # Use stimulus labels, not all labels
                    contrast_names=contrast_names,
                    contrast_results=ols_contrast_results,
                    output_format="nifti_gz",  # Force NIfTI output
                )

            if args.Obeta:
                print(f"  • Writing OLS betas only: {args.Obeta}")
                # Write only betas using the write_glm_results_nifti function correctly
                # Create a temporary results-like object with only betas
                import nibabel as nib

                from fastfuncsim.glm.outputs import (
                    _ensure_numpy,
                    _get_voxel_mask,
                    _reshape_parameter_map,
                    _resolve_shape,
                )

                affine = getattr(ols_results, "affine", np.eye(4))
                volume_shape = _resolve_shape(ols_results, None)
                voxel_mask = _get_voxel_mask(ols_results)

                betas_np = _ensure_numpy(ols_results.betas)
                betas_vol = _reshape_parameter_map(betas_np, volume_shape, voxel_mask)

                beta_img = nib.Nifti1Image(betas_vol, affine)
                # Always write NIfTI .nii.gz regardless of input format
                nib.save(beta_img, replace_afni_extension(args.Obeta, ".nii.gz"))

            if args.Onuisance:
                # NOTE: When task_indices is provided, OLS results contain only stimulus columns.
                # There are no nuisance columns in the OLS results to write out.
                # Nuisance parameters are in the full design matrix but not fitted separately.
                if stim_indices:
                    print(
                        "  ⚠️  Skipping -Onuisance: OLS fit only includes stimulus columns (not nuisance)"
                    )
                    print(
                        "      To get nuisance parameters, fit the full model without StimBots/StimTops filtering"
                    )
                else:
                    # No filtering - all regressors are present
                    print(f"  • Writing OLS nuisance betas + stats: {args.Onuisance}")
                    write_glm_bucket_as_nifti(
                        ols_results,
                        args.Onuisance,
                        condition_names=stim_labels,
                        output_format="nifti_gz",  # Force NIfTI output
                    )

            if want_single_trials:
                if stim_indices:
                    print(
                        "  • Writing OLS single-trial betas (onset order): ols_single.nii.gz"
                    )
                    # Need design matrix to extract onset times
                    # Get it from callback_design_info (key is "matrix" from read_afni_design_matrix)
                    if "matrix" in callback_design_info:
                        write_single_trials_output(
                            ols_results,
                            "ols_single.nii.gz",
                            callback_design_info["matrix"],
                            stim_indices,
                            stim_labels,
                        )
                    else:
                        print(
                            "      ⚠️  Warning: Design matrix not available, cannot determine onset times"
                        )
                else:
                    print(
                        "  ⚠️  Skipping single-trial output: No stimulus columns found (StimBots/StimTops)"
                    )

            # Write partial R² if requested and available
            if (
                args.rpartial
                and hasattr(ols_results, "r2_partial")
                and ols_results.r2_partial is not None
            ):
                # Generate output path by inserting _partialR2 before extension
                if args.Obuck:
                    if args.Obuck.endswith(".nii.gz"):
                        partial_r2_path = args.Obuck.replace(
                            ".nii.gz", "_partialR2.nii.gz"
                        )
                    elif args.Obuck.endswith(".nii"):
                        partial_r2_path = args.Obuck.replace(
                            ".nii", "_partialR2.nii.gz"
                        )
                    else:
                        partial_r2_path = args.Obuck + "_partialR2.nii.gz"
                else:
                    partial_r2_path = "OLS_partialR2.nii.gz"
                print(f"  • Writing OLS partial R² per condition: {partial_r2_path}")

                from fastfuncsim.glm.outputs import (
                    _get_voxel_mask,
                    _resolve_shape,
                    write_partial_r2_with_labels,
                )

                # Get metadata for AFNI stat params
                n_timepoints_ols = callback_design_info.get("n_timepoints")
                n_regressors_ols = callback_design_info.get("n_regressors")

                # Get mode from args (captured in closure)
                r2_mode = args.rpartial if args.rpartial else "full"

                write_partial_r2_with_labels(
                    ols_results.r2_partial,
                    partial_r2_path,
                    condition_labels=stim_labels,
                    volume_shape=_resolve_shape(ols_results, None),
                    voxel_mask=_get_voxel_mask(ols_results),
                    affine=getattr(ols_results, "affine", None),
                    n_timepoints=n_timepoints_ols,
                    n_regressors=n_regressors_ols,
                    apply_afni_metadata=True,
                    mode=r2_mode,  # "full" or "task"
                )

                # Print labels for reference
                suffix = "_partialR2_task" if r2_mode == "task" else "_partialR2"
                print("     Sub-bricks (partial R² with AFNI stat params):")
                for idx, label in enumerate(stim_labels):
                    print(f"       [{idx}] {label}{suffix}")

            print()

        _ols_write_callback = write_ols_results

    # Set environment variable for partial R² mode (so analysis.py callback can access it)
    if args.rpartial:
        import os

        os.environ["FASTFUNCSIM_R2_PARTIAL_MODE"] = args.rpartial

    # Set environment variable for semi-partial R² mode (so analysis.py callback can access it)
    if args.r2semipartial:
        import os

        os.environ["FASTFUNCSIM_R2_SEMIPARTIAL_MODE"] = args.r2semipartial

    # ==========================================================================
    # Preprocessing: Blur and/or Scale if requested
    # ==========================================================================
    preprocessing_applied = args.do_blur is not None or args.do_scale

    if preprocessing_applied:
        print()
        print("📦 Preprocessing data...")

        # Need to load data manually for preprocessing
        from tqdm import tqdm

        # Get header info from first file
        first_img = load_nifti(input_files[0])
        affine = first_img.affine
        volume_shape = first_img.shape[:3]
        voxel_sizes = tuple(np.abs(np.diag(affine)[:3]))

        # Parse design matrix to get run information
        design_info_pre = read_afni_design_matrix(args.matrix)
        run_trs = design_info_pre.get("run_trs", None)

        # Compute run_starts from run_trs
        if run_trs is not None:
            run_starts = [0]
            cumsum = 0
            for rt in run_trs[:-1]:
                cumsum += rt
                run_starts.append(cumsum)
        else:
            # Assume single run
            run_starts = [0]

        # Load and optionally blur each run
        run_data_list = []

        if args.do_blur is not None:
            print(f"  Applying Gaussian blur (FWHM = {args.do_blur} mm)...")

        for run_idx, run_file in enumerate(tqdm(input_files, desc="  Loading runs", unit="run")):
            img = load_nifti(run_file)
            data_4d = img.get_fdata(dtype=np.float32)

            if data_4d.ndim != 4:
                raise ValueError(f"Expected 4D data, got shape {data_4d.shape}")

            # Apply blur if requested (on 4D data)
            if args.do_blur is not None:
                data_4d = gaussian_blur_3d(
                    data_4d,
                    fwhm_mm=args.do_blur,
                    voxel_sizes=voxel_sizes,
                    device=device,
                    verbose=(run_idx == 0),  # Only print details for first run
                )

            # Flatten to 2D (n_voxels, n_timepoints)
            n_tps = data_4d.shape[3]
            data_2d = data_4d.reshape(-1, n_tps)

            run_data_list.append(data_2d)

        # Concatenate all runs
        fmri_data_preprocessed = np.concatenate(run_data_list, axis=1)
        del run_data_list

        # Apply scaling if requested (on concatenated 2D data)
        if args.do_scale:
            print("  Applying scaling (mean=100 per run)...")
            # Convert to torch for scale_to_percent_signal
            data_tensor = torch.from_numpy(fmri_data_preprocessed)
            data_tensor, violations_mask, scale_info = scale_to_percent_signal(
                data=data_tensor,
                run_starts=run_starts,
                max_scale=200.0,
                verbose=True,
            )
            fmri_data_preprocessed = data_tensor.numpy()

            if scale_info["n_violations"] > 0:
                print(f"  ⚠️  {scale_info['n_violations']:,} ceiling violations")

        # Reshape back to 4D for analyze_from_design_matrix
        total_tps = fmri_data_preprocessed.shape[1]
        fmri_data_preprocessed = fmri_data_preprocessed.reshape(*volume_shape, total_tps)

        print(f"  ✓ Preprocessing complete: {volume_shape} × {total_tps} timepoints")
        print()

    print("🚀 Starting GLM analysis...")
    print()

    # Handle HDF5 caching for fast data loading
    fmri_data_to_use = None
    cache_metadata = None

    # If preprocessing was applied, use that data
    if preprocessing_applied:
        fmri_data_to_use = fmri_data_preprocessed

    if args.cache and not preprocessing_applied:
        from fastfuncsim.data_cache import check_cache_valid, load_cache

        cache_valid = check_cache_valid(args.cache, input_files)

        if cache_valid:
            # Load from cache
            cached_data, cache_metadata = load_cache(args.cache, input_files, validate=True)

            # Reshape to 4D if volume_shape available (needed for test mode and output writing)
            if 'volume_shape' in cache_metadata:
                vol_shape = cache_metadata['volume_shape']
                n_timepoints = cached_data.shape[1]
                # Reshape from (n_voxels, n_timepoints) to (x, y, z, n_timepoints)
                cached_data = cached_data.reshape(*vol_shape, n_timepoints)

            fmri_data_to_use = cached_data  # Pass numpy array instead of file list
        else:
            # Will create cache after loading data
            print(f"📝 Cache not found or invalid - will create: {args.cache}")

    # Run analysis
    try:
        # Determine grid batching strategy
        use_grid_batching = None  # Auto-detect by default
        if args.grid_batching:
            use_grid_batching = True
        elif args.no_grid_batching:
            use_grid_batching = False

        # Use cached data if available, otherwise load from files
        if fmri_data_to_use is None:
            fmri_data_to_use = input_files if len(input_files) > 1 else input_files[0]

        # Load precomputed ARMA params only if explicitly requested via -load_Rvar
        precomputed_arma = None
        if args.load_Rvar:
            # Try with automatic extension detection
            rvar_base = Path(args.load_Rvar)
            rvar_path = None
            for candidate in [rvar_base,
                               Path(str(rvar_base) + '.nii.gz'),
                               Path(str(rvar_base) + '.nii')]:
                if candidate.exists():
                    rvar_path = candidate
                    break

            if rvar_path is None:
                print(f"\n❌ ERROR: -load_Rvar file not found: {args.load_Rvar}")
                sys.exit(1)

            print(f"\n📂 Loading precomputed ARMA parameters from: {rvar_path}")
            print("   (Skipping grid search - saves ~80% compute time)")

            try:
                rvar_img = load_nifti(rvar_path)
                rvar_data = rvar_img.get_fdata()  # (x, y, z[, 1], n_params)

                # AFNI bucket files sometimes store sub-briks in the 5th dimension
                # with a singleton 4th dimension (e.g. shape (x, y, z, 1, n)).
                # Squeeze out all size-1 dimensions that sit before the last axis.
                while rvar_data.ndim > 4 and rvar_data.shape[-2] == 1:
                    rvar_data = rvar_data[..., 0, :]  # drop singleton dim

                if rvar_data.ndim != 4:
                    raise ValueError(f"Expected 4D Rvar file, got {rvar_data.ndim}D "
                                     f"(shape {rvar_img.shape}). "
                                     "Cannot reduce to (x, y, z, n_params).")
                if rvar_data.shape[3] < 2:
                    raise ValueError(f"Rvar file must have at least 2 sub-briks (a, b), found {rvar_data.shape[3]}")

                precomputed_arma = rvar_data[..., :2]  # (x, y, z, 2) — only a and b needed
                n_voxels_total = np.prod(precomputed_arma.shape[:3])
                a_range = (precomputed_arma[..., 0].min(), precomputed_arma[..., 0].max())
                b_range = (precomputed_arma[..., 1].min(), precomputed_arma[..., 1].max())
                print(f"   ✓ Loaded ARMA params: {n_voxels_total:,} voxels × 2 params (a, b)")
                print(f"   • a range: [{a_range[0]:.3f}, {a_range[1]:.3f}]")
                print(f"   • b range: [{b_range[0]:.3f}, {b_range[1]:.3f}]")

            except Exception as e:
                print(f"   ⚠️  Failed to load Rvar file: {e}")
                print("   Proceeding with grid search instead")
                precomputed_arma = None

        # Warn if the Rvar output file already exists (user may want -load_Rvar instead)
        if args.Rvar and not args.load_Rvar:
            rvar_out_base = Path(args.Rvar)
            for candidate in [rvar_out_base,
                               Path(str(rvar_out_base) + '.nii.gz'),
                               Path(str(rvar_out_base) + '.nii')]:
                if candidate.exists():
                    print(f"\n⚠️  Note: {candidate} already exists and will be overwritten.")
                    print("   To reuse precomputed ARMA params instead, pass: -load_Rvar {candidate}")
                    break

        results, design_info = analyze_from_design_matrix(
            fmri_data=fmri_data_to_use,
            design_matrix_file=args.matrix,
            method="arma11",  # Always use ARMA for 3dREMLfast
            arma_a_grid=a_grid,
            arma_b_grid=b_grid,
            precomputed_arma_params=precomputed_arma,
            want_ols=want_ols,
            ols_output_path=ols_output_path,
            ols_output_format=output_format,
            device=device,
            mask_file=args.mask,
            cache_file=args.cache if (args.cache and cache_metadata is None) else None,
            cached_metadata=cache_metadata,  # Pass cached header/affine/volume_shape
            test_n_voxels=args.test,
            voxel_chunk_size=args.batch_size,
            use_double=args.use_double,
            debug_memory=args.debug_memory,
            enable_quick_estimate=args.quick_estimate,
            use_grid_batching=use_grid_batching,
            want_r2_partial=bool(args.rpartial),  # True if flag is set (any mode)
            r2_partial_mode=args.rpartial
            if args.rpartial
            else "full",  # "full" or "task"
            want_r2_semipartial=bool(
                args.r2semipartial
            ),  # True if flag is set (any mode)
            r2_semipartial_mode=args.r2semipartial
            if args.r2semipartial
            else "full",  # "full" or "task"
            legacy_contrasts=args.legacy_contrasts,
        )
    except Exception as e:
        print(f"\n❌ ERROR during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("✅ Analysis complete!")
    print()

    # Write outputs
    print("💾 Writing outputs...")
    print()

    # Determine stat flags (default to -fout if none specified)
    _want_fstat = args.fout or (not args.tout and not args.rout and not args.rpartial)
    _want_tstat = args.tout
    _want_rstat = args.rout
    want_r2_partial = args.rpartial

    # Extract design metadata using helper function (clean naming!)
    from fastfuncsim.io.afni import extract_design_metadata

    full_labels, stim_labels, stim_column_indices = extract_design_metadata(design_info)

    # Determine which columns were actually fitted (use metadata from results object)
    fitted_column_indices = getattr(results, "fitted_column_indices", None)

    # Extract labels matching what was actually fitted
    if fitted_column_indices is not None:
        # Results were filtered - use the labels that match fitted columns
        # (In most cases, fitted_column_indices == stim_column_indices)
        fitted_labels = [full_labels[i] for i in fitted_column_indices]
        stim_indices = fitted_column_indices  # For single-trials output
    else:
        # All columns were fitted
        fitted_labels = full_labels
        stim_indices = (
            stim_column_indices
            if stim_column_indices
            else list(range(len(full_labels)))
        )

    # REML outputs
    if args.Rbuck:
        print(f"  • Writing REML betas + stats (bucket): {args.Rbuck}")
        # Rbuck: Betas + stats for fitted regressors + GLT contrasts
        # Use fitted_labels which match results.betas shape
        contrast_names = getattr(results, "contrast_labels", None)

        # Build contrast_results dict if we have contrasts
        contrast_results = None
        if hasattr(results, "contrast_betas") and results.contrast_betas is not None:
            contrast_results = {
                "contrast_betas": results.contrast_betas,
                "contrast_tstats": results.contrast_tstats,
            }
            # Add partial R² if available and requested
            if (
                hasattr(results, "contrast_r2_partial")
                and results.contrast_r2_partial is not None
            ):
                contrast_results["contrast_r2_partial"] = results.contrast_r2_partial
            # Add semi-partial R² if available and requested
            if (
                hasattr(results, "contrast_r2_semipartial")
                and results.contrast_r2_semipartial is not None
            ):
                contrast_results["contrast_r2_semipartial"] = (
                    results.contrast_r2_semipartial
                )

        write_glm_bucket_as_nifti(
            results,
            args.Rbuck,
            condition_names=fitted_labels,
            contrast_names=contrast_names,
            contrast_results=contrast_results,
            output_format="nifti_gz",  # Force NIfTI output
        )

    if args.Rbeta:
        print(f"  • Writing REML betas only: {args.Rbeta}")
        # Rbeta: ALL betas, no stats
        import nibabel as nib

        from fastfuncsim.glm.outputs import (
            _ensure_numpy,
            _get_voxel_mask,
            _reshape_parameter_map,
            _resolve_shape,
        )

        affine = getattr(results, "affine", np.eye(4))
        volume_shape = _resolve_shape(results, None)
        voxel_mask = _get_voxel_mask(results)

        assert results.betas is not None, "Results must have betas"
        betas_np = _ensure_numpy(results.betas)
        betas_vol = _reshape_parameter_map(betas_np, volume_shape, voxel_mask)

        beta_img = nib.Nifti1Image(betas_vol, affine)
        # Always write NIfTI .nii.gz regardless of input format
        nib.save(beta_img, replace_afni_extension(args.Rbeta, ".nii.gz"))

    if args.Rnuisance:
        print(f"  • Writing REML nuisance betas + stats: {args.Rnuisance}")
        # Rnuisance: Extract nuisance regressors (everything NOT in stimulus columns)
        # NOTE: This only works if full design was fitted (not filtered)
        if fitted_column_indices is not None:
            print(
                "  ⚠️  Skipping -Rnuisance: REML fit only includes stimulus columns (not nuisance)"
            )
            print(
                "      To get nuisance parameters, fit the full model without StimBots/StimTops filtering"
            )
        elif stim_indices:
            all_indices = list(range(len(full_labels)))
            nuisance_indices = [i for i in all_indices if i not in stim_indices]
            nuisance_results = slice_glm_results(results, nuisance_indices)
            nuisance_names = (
                [design_info["column_labels"][i] for i in nuisance_indices]
                if "column_labels" in design_info
                else None
            )
        else:
            # No stimulus indices specified, use all regressors
            nuisance_results = results
            nuisance_names = design_info.get("column_labels")

        # Always write NIfTI .nii.gz regardless of input format
        write_glm_bucket_as_nifti(
            nuisance_results,
            args.Rnuisance,
            condition_names=nuisance_names,
            output_format="nifti_gz",  # Force NIfTI output
        )

    if args.Rvar:
        # Ensure Rvar output path has .nii.gz extension
        rvar_output_path = Path(args.Rvar)
        if not (str(rvar_output_path).endswith('.nii.gz') or str(rvar_output_path).endswith('.nii')):
            rvar_output_path = Path(str(rvar_output_path) + '.nii.gz')

        print(f"  • Writing REML variance parameters: {rvar_output_path}")
        # Stack variance parameters: a, b, lambda, StDev (all reliably computed)
        var_stack = []
        var_labels = []

        if results.arma_params is not None:
            var_stack.append(results.arma_params[:, 0])  # a
            var_stack.append(results.arma_params[:, 1])  # b
            var_labels.extend(["a", "b"])

        if results.arma_lambda is not None:
            var_stack.append(results.arma_lambda)  # lambda
            var_labels.append("lambda")

        if results.sigma2 is not None:
            var_stack.append(torch.sqrt(results.sigma2))  # StDev
            var_labels.append("StDev")

        # Stack and write
        var_data = torch.stack(var_stack, dim=1)
        # Write variance parameters directly as 4D NIfTI
        import nibabel as nib

        affine = getattr(results, "affine", np.eye(4))
        volume_shape = getattr(results, "original_shape", None)
        voxel_mask = getattr(results, "voxel_mask", None)

        # Reshape var_data to 4D volume (convert to numpy first!)
        var_data_np = (
            var_data.cpu().numpy() if isinstance(var_data, torch.Tensor) else var_data
        )

        if volume_shape is not None and voxel_mask is not None:
            n_params = var_data_np.shape[1]
            var_vol = np.zeros((*volume_shape, n_params), dtype=np.float32)
            voxel_mask_np = (
                voxel_mask.cpu().numpy()
                if isinstance(voxel_mask, torch.Tensor)
                else voxel_mask
            )
            var_vol[voxel_mask_np.reshape(volume_shape)] = var_data_np
        else:
            # Assume already in volume shape
            var_vol = (
                var_data_np.reshape(*volume_shape, -1) if volume_shape else var_data_np
            )

        var_img = nib.Nifti1Image(var_vol, affine)

        # Save in requested format using helper
        # IMPORTANT: nibabel cannot convert NIfTI headers to AFNI headers
        # So we always save variance files as NIfTI (even if user requested AFNI)
        import shutil
        import subprocess

        from fastfuncsim.glm.outputs import _save_nifti_with_format

        # OPTIMIZATION: Write uncompressed first, then 3drefit, then compress
        # This avoids 3drefit having to decompress/recompress huge files!
        # For 870k voxels, this saves significant time

        # Write uncompressed .nii first
        temp_nii_path = rvar_output_path.with_suffix('.nii') if str(rvar_output_path).endswith('.nii.gz') else rvar_output_path
        _save_nifti_with_format(var_img, temp_nii_path, "nifti")

        # Label sub-briks using AFNI's 3drefit (fast on uncompressed file)
        print("  • Labeling Rvar sub-briks with 3drefit...")

        # Build 3drefit command with all sub-brik labels
        refit_cmd = ["3drefit"]
        for idx, label in enumerate(var_labels):
            refit_cmd.extend(["-sublabel", str(idx), label])
        refit_cmd.append(str(temp_nii_path.absolute()))

        try:
            subprocess.run(refit_cmd, check=True, capture_output=True, text=True)
            print(f"    ✓ Labeled {len(var_labels)} sub-briks: {', '.join(var_labels)}")
        except subprocess.CalledProcessError as e:
            print(f"    ⚠️  3drefit labeling failed: {e.stderr}")
            print("    (File was written successfully, but lacks sub-brik labels)")
        except FileNotFoundError:
            print("    ⚠️  3drefit not found in PATH (AFNI not installed?)")
            print("    (File was written successfully, but lacks sub-brik labels)")

        # Compress with pigz/zstd if output is compressed format
        if str(rvar_output_path).endswith('.nii.gz'):
            from fastfuncsim.io.afni import compress_nifti
            print(f"  • Compressing Rvar output...")
            compress_nifti(temp_nii_path, rvar_output_path, remove_original=True)

    # Write partial R² if requested and available
    if (
        want_r2_partial
        and hasattr(results, "r2_partial")
        and results.r2_partial is not None
    ):
        # Generate output path by inserting _partialR2 before extension
        if args.Rbuck:
            if args.Rbuck.endswith(".nii.gz"):
                partial_r2_path = args.Rbuck.replace(".nii.gz", "_partialR2.nii.gz")
            elif args.Rbuck.endswith(".nii"):
                partial_r2_path = args.Rbuck.replace(".nii", "_partialR2.nii.gz")
            else:
                partial_r2_path = args.Rbuck + "_partialR2.nii.gz"
        else:
            partial_r2_path = "REML_partialR2.nii.gz"
        print(f"  • Writing REML partial R² per condition: {partial_r2_path}")

        from fastfuncsim.glm.outputs import (
            _get_voxel_mask,
            _resolve_shape,
            write_partial_r2_with_labels,
        )

        # Get design info for stat parameters
        n_timepoints_reml = design_info.get("n_timepoints")
        n_regressors_reml = design_info.get("n_regressors")

        # Get mode
        r2_mode = args.rpartial if args.rpartial else "full"

        write_partial_r2_with_labels(
            results.r2_partial,
            partial_r2_path,
            condition_labels=fitted_labels,
            volume_shape=_resolve_shape(results, None),
            voxel_mask=_get_voxel_mask(results),
            affine=getattr(results, "affine", None),
            n_timepoints=n_timepoints_reml,
            n_regressors=n_regressors_reml,
            apply_afni_metadata=True,
            mode=r2_mode,  # "full" or "task"
        )

        suffix = "_partialR2_task" if r2_mode == "task" else "_partialR2"
        print("     Sub-bricks (partial R² with AFNI stat params):")
        for idx, label in enumerate(fitted_labels):
            print(f"       [{idx}] {label}{suffix}")

    # Write nuisance partial R² if available (always "full" mode for nuisance)
    if (
        want_r2_partial
        and hasattr(results, "r2_partial_nuisance")
        and results.r2_partial_nuisance is not None
    ):
        # Generate output path
        if args.Rbuck:
            if args.Rbuck.endswith(".nii.gz"):
                nuisance_r2_path = args.Rbuck.replace(
                    ".nii.gz", "_nuisance_partialR2.nii.gz"
                )
            elif args.Rbuck.endswith(".nii"):
                nuisance_r2_path = args.Rbuck.replace(
                    ".nii", "_nuisance_partialR2.nii.gz"
                )
            else:
                nuisance_r2_path = args.Rbuck + "_nuisance_partialR2.nii.gz"
        else:
            nuisance_r2_path = "REML_nuisance_partialR2.nii.gz"

        print(f"  • Writing REML nuisance partial R² per regressor: {nuisance_r2_path}")

        from fastfuncsim.glm.outputs import (
            _get_voxel_mask,
            _resolve_shape,
            write_partial_r2_with_labels,
        )

        # Get nuisance labels from design_info
        nuisance_labels = design_info.get(
            "nuisance_labels",
            [f"nuisance{i}" for i in range(results.r2_partial_nuisance.shape[1])],
        )

        # Get design info for stat parameters
        n_timepoints_reml = design_info.get("n_timepoints")
        n_regressors_reml = design_info.get("n_regressors")

        write_partial_r2_with_labels(
            results.r2_partial_nuisance,
            nuisance_r2_path,
            condition_labels=nuisance_labels,
            volume_shape=_resolve_shape(results, None),
            voxel_mask=_get_voxel_mask(results),
            affine=getattr(results, "affine", None),
            n_timepoints=n_timepoints_reml,
            n_regressors=n_regressors_reml,
            apply_afni_metadata=True,
            mode="full",  # Always use "full" for nuisance (not rescaled)
        )

        print("     Sub-bricks (nuisance partial R² with AFNI stat params):")
        for idx, label in enumerate(nuisance_labels):
            print(f"       [{idx}] {label}_partialR2")

    # Write semi-partial R² if requested and available
    want_r2_semipartial = args.r2semipartial
    if (
        want_r2_semipartial
        and hasattr(results, "r2_semipartial")
        and results.r2_semipartial is not None
    ):
        # Generate output path by inserting _semipartialR2 before extension
        if args.Rbuck:
            if args.Rbuck.endswith(".nii.gz"):
                semipartial_r2_path = args.Rbuck.replace(
                    ".nii.gz", "_semipartialR2.nii.gz"
                )
            elif args.Rbuck.endswith(".nii"):
                semipartial_r2_path = args.Rbuck.replace(
                    ".nii", "_semipartialR2.nii.gz"
                )
            else:
                semipartial_r2_path = args.Rbuck + "_semipartialR2.nii.gz"
        else:
            semipartial_r2_path = "REML_semipartialR2.nii.gz"
        print(f"  • Writing REML semi-partial R² per condition: {semipartial_r2_path}")

        from fastfuncsim.glm.outputs import (
            _get_voxel_mask,
            _resolve_shape,
            write_partial_r2_with_labels,
        )

        # Get design info for stat parameters
        n_timepoints_reml = design_info.get("n_timepoints")
        n_regressors_reml = design_info.get("n_regressors")

        # Get mode
        r2_semi_mode = args.r2semipartial if args.r2semipartial else "full"

        write_partial_r2_with_labels(
            results.r2_semipartial,
            semipartial_r2_path,
            condition_labels=fitted_labels,
            volume_shape=_resolve_shape(results, None),
            voxel_mask=_get_voxel_mask(results),
            affine=getattr(results, "affine", None),
            n_timepoints=n_timepoints_reml,
            n_regressors=n_regressors_reml,
            apply_afni_metadata=True,
            mode=r2_semi_mode,  # "full" or "task"
        )

        suffix = "_semipartialR2_task" if r2_semi_mode == "task" else "_semipartialR2"
        print("     Sub-bricks (semi-partial R² with AFNI stat params):")
        for idx, label in enumerate(fitted_labels):
            print(f"       [{idx}] {label}{suffix}")

    # Write nuisance semi-partial R² if available (always "full" mode for nuisance)
    if (
        want_r2_semipartial
        and hasattr(results, "r2_semipartial_nuisance")
        and results.r2_semipartial_nuisance is not None
    ):
        # Generate output path
        if args.Rbuck:
            if args.Rbuck.endswith(".nii.gz"):
                nuisance_semi_r2_path = args.Rbuck.replace(
                    ".nii.gz", "_nuisance_semipartialR2.nii.gz"
                )
            elif args.Rbuck.endswith(".nii"):
                nuisance_semi_r2_path = args.Rbuck.replace(
                    ".nii", "_nuisance_semipartialR2.nii.gz"
                )
            else:
                nuisance_semi_r2_path = args.Rbuck + "_nuisance_semipartialR2.nii.gz"
        else:
            nuisance_semi_r2_path = "REML_nuisance_semipartialR2.nii.gz"

        print(
            f"  • Writing REML nuisance semi-partial R² per regressor: {nuisance_semi_r2_path}"
        )

        from fastfuncsim.glm.outputs import (
            _get_voxel_mask,
            _resolve_shape,
            write_partial_r2_with_labels,
        )

        # Get nuisance labels from design_info
        nuisance_labels = design_info.get(
            "nuisance_labels",
            [f"nuisance{i}" for i in range(results.r2_semipartial_nuisance.shape[1])],
        )

        # Get design info for stat parameters
        n_timepoints_reml = design_info.get("n_timepoints")
        n_regressors_reml = design_info.get("n_regressors")

        write_partial_r2_with_labels(
            results.r2_semipartial_nuisance,
            nuisance_semi_r2_path,
            condition_labels=nuisance_labels,
            volume_shape=_resolve_shape(results, None),
            voxel_mask=_get_voxel_mask(results),
            affine=getattr(results, "affine", None),
            n_timepoints=n_timepoints_reml,
            n_regressors=n_regressors_reml,
            apply_afni_metadata=True,
            mode="full",  # Always use "full" for nuisance (not rescaled)
        )

        print("     Sub-bricks (nuisance semi-partial R² with AFNI stat params):")
        for idx, label in enumerate(nuisance_labels):
            print(f"       [{idx}] {label}_semipartialR2")

    if args.Rfitts:
        print(f"  • Writing REML fitted model: {args.Rfitts}")
        if results.predicted is not None:
            import nibabel as nib

            from fastfuncsim.glm.outputs import _ensure_numpy, _get_voxel_mask

            affine = getattr(results, "affine", np.eye(4))
            volume_shape = getattr(results, "original_shape", None)
            voxel_mask = _get_voxel_mask(results)

            # Predicted is (n_timepoints, n_voxels), need (n_voxels, n_timepoints)
            predicted_np = _ensure_numpy(results.predicted.T)

            # Reshape to 4D volume (x, y, z, timepoints)
            if volume_shape is not None and voxel_mask is not None:
                n_timepoints = predicted_np.shape[1]
                predicted_vol = np.zeros(
                    (*volume_shape, n_timepoints), dtype=np.float32
                )
                predicted_vol[voxel_mask.reshape(volume_shape)] = predicted_np
            else:
                predicted_vol = predicted_np

            fitts_img = nib.Nifti1Image(predicted_vol, affine)
            # Always write NIfTI .nii.gz regardless of input format
            nib.save(fitts_img, replace_afni_extension(args.Rfitts, ".nii.gz"))
        else:
            print("    ⚠️  Warning: Fitted values not available (predicted=None)")

    if args.Rerrts:
        print(f"  • Writing REML residuals: {args.Rerrts}")
        if results.residuals is not None:
            import nibabel as nib

            from fastfuncsim.glm.outputs import _ensure_numpy, _get_voxel_mask

            affine = getattr(results, "affine", np.eye(4))
            volume_shape = getattr(results, "original_shape", None)
            voxel_mask = _get_voxel_mask(results)

            # Residuals is (n_timepoints, n_voxels), need (n_voxels, n_timepoints)
            residuals_np = _ensure_numpy(results.residuals.T)

            # Reshape to 4D volume (x, y, z, timepoints)
            if volume_shape is not None and voxel_mask is not None:
                n_timepoints = residuals_np.shape[1]
                residuals_vol = np.zeros(
                    (*volume_shape, n_timepoints), dtype=np.float32
                )
                residuals_vol[voxel_mask.reshape(volume_shape)] = residuals_np
            else:
                residuals_vol = residuals_np

            errts_img = nib.Nifti1Image(residuals_vol, affine)
            # Always write NIfTI .nii.gz regardless of input format
            nib.save(errts_img, replace_afni_extension(args.Rerrts, ".nii.gz"))
        else:
            print("    ⚠️  Warning: Residuals not available")

    if args.Rwherr:
        print(f"  • Writing REML whitened residuals: {args.Rwherr}")
        print("    ⚠️  Warning: Whitened residuals not currently computed")
        # Would need to compute: residuals @ inv(chol(R))

    # Single trials output for ARMA (if requested)
    if args.single_trials and stim_indices:
        label = args.single_trials
        output_filename = f"reml_{label}_single.nii.gz"
        print(f"  • Writing REML single-trial betas (onset order): {output_filename}")
        if "matrix" in design_info:
            from fastfuncsim.glm.outputs import write_single_trials_output

            write_single_trials_output(
                results,
                output_filename,
                design_info["matrix"],  # Full design matrix for onset extraction
                stim_indices,  # Column indices into full design
                fitted_labels,  # Labels matching results.betas shape
            )
        else:
            print(
                "      ⚠️  Warning: Design matrix not available, cannot determine onset times"
            )

    # OLS outputs - already written by callback during analysis!
    # The callback writes OLS results immediately after OLS completion,
    # freeing memory before the ARMA loop starts.

    print()
    print("=" * 70)
    print("✅ 3dREMLfast completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
