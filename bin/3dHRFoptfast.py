#!/usr/bin/env python3
"""
3dHRFoptfast - Fast cross-validated HRF optimization per voxel using GPU acceleration

This tool selects the optimal HRF for each voxel using cross-validation across runs,
then fits a final GLM with the voxel-wise optimal HRFs.

Two HRF library modes:
- library: Double-gamma HRFs with varying time-to-peak and undershoot parameters
- pighs: PIGHS (Parametric Individually Generated HRFs) half-cosine basis with Latin Hypercube sampling

The CV-based HRF selection prevents overfitting that occurs with in-sample selection.

Basic usage:
    3dHRFoptfast -input run1.nii.gz run2.nii.gz run3.nii.gz \\
                 -onsets cond1.txt cond2.txt \\
                 -durations 2.0 5.0 \\
                 -tr 2.0 \\
                 -prefix subject01

For help:
    3dHRFoptfast -help
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

# Import fastfuncsim modules
try:
    from fastfuncsim.cli_utils import (
        auto_polort,
        build_nuisance_per_run,
        compute_run_lengths,
        get_average_run_duration,
        load_and_preprocess_runs,
        parse_device_arg,
        preflight_check,
    )
    from fastfuncsim.design_builder import (
        create_onset_matrix_microtime,
        parse_afni_timing_file,
        parse_durations,
    )
    from fastfuncsim.hrf import get_hrf_library
    from fastfuncsim.hrf_selection import (
        _fit_voxelwise_hrf_canonical,
        _fit_voxelwise_hrf_single_trial,
        fit_glm_hrf_library_with_xval,
        save_hrf_selection_results,
    )
except ImportError as e:
    print(f"ERROR: Could not import fastfuncsim: {e}")
    print("Make sure fastfuncsim is installed: pip install -e .")
    sys.exit(1)


def parse_input_files(input_arg: Union[str, list[str]]) -> list[str]:
    """Parse input files (can be list from nargs='+' or single string)

    Supports:
    - Single file: "/path/to/file.nii.gz"
    - Multiple files: ["/path/run1.nii.gz", "/path/run2.nii.gz"]
    - Glob patterns: ["run*.nii.gz"] or "run*.nii.gz"
    """
    import glob as glob_module

    # Handle both list (from nargs='+') and string
    if isinstance(input_arg, str):
        input_arg = input_arg.strip().strip('"').strip("'")
        input_list = input_arg.split()
    else:
        input_list = input_arg

    # Expand globs and collect files
    files = []
    for pattern in input_list:
        matches = glob_module.glob(pattern)
        if matches:
            files.extend(sorted(matches))
        else:
            files.append(pattern)

    # Validate files exist
    for f in files:
        if not Path(f).exists():
            print(f"ERROR: Input file not found: {f}")
            sys.exit(1)

    return files


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="3dHRFoptfast - Fast GPU-accelerated cross-validated HRF optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,  # We handle -help ourselves to avoid required arg check
        epilog="""
Examples:
  # Basic HRF optimization with library of HRF variants
  3dHRFoptfast -input run1.nii.gz run2.nii.gz run3.nii.gz \\
               -onsets cond1.txt cond2.txt \\
               -durations 2.0 \\
               -tr 2.0 \\
               -prefix subject01_hrf

  # Using PIGHS library with different durations per condition
  3dHRFoptfast -input run*.nii.gz \\
               -onsets face.txt house.txt object.txt \\
               -durations 0.5 2.0 2.0 \\
               -tr 1.5 \\
               -hrf_mode pighs \\
               -n_hrfs 30 \\
               -prefix sub01_pighs

  # Leave-one-run-out CV (default) with verbose output
  3dHRFoptfast -input run*.nii.gz \\
               -onsets stim.txt \\
               -durations 1.0 \\
               -tr 2.0 \\
               -cv_strategy loro \\
               -verbose \\
               -prefix output

  # Split-halves CV with mask
  3dHRFoptfast -input run*.nii.gz \\
               -onsets cond.txt \\
               -durations 2.0 \\
               -tr 2.0 \\
               -cv_strategy 0.5 \\
               -mask brain_mask.nii.gz \\
               -prefix masked_output

  # With motion and physio nuisance regressors
  3dHRFoptfast -input run*.nii.gz \\
               -onsets face.txt house.txt \\
               -durations 2.0 \\
               -tr 2.0 \\
               -ortvec motion_all.1D motion \\
               -ortvec physio_all.txt physio \\
               -prefix sub01_with_nuisance

Outputs:
  {prefix}_hrf_index.nii.gz       - Which HRF (1-N) was selected per voxel
  {prefix}_xval_r2.nii.gz         - Cross-validated R² for selected HRF
  {prefix}_xval_r2_all_hrfs.nii.gz - 4D: CV R² for each HRF (volume per HRF)
  {prefix}_stats.nii.gz           - Final GLM betas and t-stats (AFNI bucket format)
  {prefix}_hrf_library.pt         - HRF library + voxel assignments for ARMA reuse
  {prefix}_metadata.json          - Full metadata for reproducibility

Notes:
  - Durations can be specified as single value (applies to all) or one per condition
  - Nuisance files (-ortvec) must be pre-concatenated across runs (matching total timepoints)
  - Common nuisance files: motion parameters (6 columns), physiological regressors, etc.
  - Future: onset files with 'married' durations (e.g., "1:2 4:5") will be supported
  - HRF library is saved for later ARMA/REML analysis with 3dREMLfast
        """,
    )

    # Required arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-input",
        nargs="+",
        required=True,
        help="Input fMRI dataset(s). Multiple files = multiple runs.",
    )
    required.add_argument(
        "-onsets",
        nargs="+",
        required=True,
        help="Onset timing files (AFNI format). One file per condition.",
    )
    required.add_argument(
        "-durations",
        nargs="+",
        required=True,
        help="Stimulus durations in seconds. Either single value for all conditions, or one per condition.",
    )
    required.add_argument(
        "-tr",
        type=float,
        required=False,
        default=None,
        help="Repetition time (TR) in seconds. If not specified, read from NIfTI header.",
    )
    required.add_argument(
        "-prefix",
        required=True,
        help="Output file prefix (e.g., 'output/subject01')",
    )

    # HRF library options
    hrf_opts = parser.add_argument_group("HRF Library Options")
    hrf_opts.add_argument(
        "-hrf_mode",
        choices=["library", "pighs"],
        default="library",
        help="HRF library type: 'library' (double-gamma variations) or 'pighs' (half-cosine basis). Default: library",
    )
    hrf_opts.add_argument(
        "-n_hrfs",
        type=int,
        default=20,
        help="Number of HRFs in library (default: 20, only affects pighs mode)",
    )

    # PIGHS-specific options
    pighs_opts = parser.add_argument_group("PIGHS Options (only used with -hrf_mode pighs)")
    pighs_opts.add_argument(
        "-pighs_peak_time_range",
        type=float,
        nargs=2,
        default=[3, 10],
        metavar=("MIN", "MAX"),
        help="Range for time-to-peak in seconds. Default: 3 10",
    )
    pighs_opts.add_argument(
        "-pighs_rise_fraction_range",
        type=float,
        nargs=2,
        default=[0.3, 0.9],
        metavar=("MIN", "MAX"),
        help="Range for rise_fraction (fraction of peak_time that is rise vs delay). Default: 0.3 0.9",
    )
    pighs_opts.add_argument(
        "-pighs_fall_range",
        type=float,
        nargs=2,
        default=[3, 10],
        metavar=("MIN", "MAX"),
        help="Range for peak-to-undershoot time (m3) in seconds. Default: 3 10",
    )
    pighs_opts.add_argument(
        "-pighs_recovery_range",
        type=float,
        nargs=2,
        default=[3, 12],
        metavar=("MIN", "MAX"),
        help="Range for undershoot recovery time (m4) in seconds. Default: 3 12",
    )
    pighs_opts.add_argument(
        "-pighs_undershoot_range",
        type=float,
        nargs=2,
        default=[0, 0.35],
        metavar=("MIN", "MAX"),
        help="Range for undershoot magnitude (c2). Default: 0 0.35",
    )

    # Cross-validation options
    cv_opts = parser.add_argument_group("Cross-Validation Options")
    cv_opts.add_argument(
        "-cv_strategy",
        default="loro",
        help=(
            "Cross-validation strategy. Options: "
            "'loro' or '1' for leave-one-run-out (default), "
            "'0.5' for split-halves, "
            "any float (0-1) for that train fraction, "
            "any int > 1 for leave-N-out"
        ),
    )
    cv_opts.add_argument(
        "-n_perms",
        type=int,
        default=100,
        help="Max number of CV permutations for random splits (default: 100)",
    )
    cv_opts.add_argument(
        "-metric",
        choices=["cod", "corr", "corr2"],
        default="cod",
        help="R² metric: 'cod' (coefficient of determination), 'corr', 'corr2' (default: cod)",
    )
    cv_opts.add_argument(
        "-single_trials",
        action="store_true",
        help="Use beta-space cross-validation (GLMsingle-style). "
        "Fits single-trial model once with each HRF, evaluates R² on "
        "condition-averaged vs individual trial betas across folds. "
        "Replaces timeseries CV with beta-space CV for HRF selection.",
    )
    cv_opts.add_argument(
        "-save_single_trial_betas",
        action="store_true",
        help="Save single-trial betas refit with optimal HRF per voxel. "
        "Only works with -single_trials. Processes voxels in HRF groups "
        "with chunking to avoid OOM. Saves to {prefix}_stats_single_trial.nii.gz",
    )
    cv_opts.add_argument(
        "-save_canonical_betas",
        action="store_true",
        help="Also save betas from canonical HRF fit for comparison. "
        "Uses one design matrix for all voxels (chunked). "
        "Saves to {prefix}_canonical_stats.nii.gz",
    )

    # Processing options
    proc_opts = parser.add_argument_group("Processing Options")
    proc_opts.add_argument(
        "-mask",
        help="Mask file to restrict analysis to brain voxels",
    )
    proc_opts.add_argument(
        "-polort",
        type=int,
        default=None,
        help="Polynomial order for drift modeling (default: auto based on run length)",
    )
    proc_opts.add_argument(
        "-ortvec",
        action="append",
        nargs=2,
        metavar=("FILE", "LABEL"),
        help=(
            "Additional nuisance regressors to project out (can be repeated). "
            "FILE is a text file with nuisance columns (AFNI 1D, CSV, or whitespace-separated). "
            "LABEL is a prefix for the column names. "
            "Files must span all runs concatenated (same length as total timepoints). "
            "Example: -ortvec motion_all.1D motion -ortvec physio.txt physio"
        ),
    )
    proc_opts.add_argument(
        "-microtime_dt",
        type=float,
        default=0.1,
        help="Microtime resolution in seconds (default: 0.1). "
        "Onsets and HRFs are represented at this resolution for precise timing.",
    )
    proc_opts.add_argument(
        "-do_scale",
        action="store_true",
        help="Scale each voxel per run to mean=100 (percent signal change units). "
        "Values are clipped to max 200 (100%% increase from mean). "
        "Violation locations are saved to {prefix}_scale_violations.nii.gz",
    )
    proc_opts.add_argument(
        "-do_blur",
        type=float,
        metavar="FWHM",
        default=None,
        help="Apply 3D Gaussian spatial smoothing with FWHM in mm. "
        "Smoothing is applied BEFORE masking to avoid edge effects. "
        "Typical values: 4-8 mm. Uses separable convolutions for speed.",
    )
    proc_opts.add_argument(
        "-canonical",
        type=str,
        default="spmg1",
        metavar="MODE",
        help="Canonical HRF for baseline comparison. Options: "
        "'spmg1' or 'SPMG1' (AFNI's SPMG1, default), "
        "'glmsingle' (GLMsingle/nilearn-style double-gamma). "
        "The baseline comparison shows improvement from HRF optimization.",
    )
    proc_opts.add_argument(
        "-device",
        type=str,
        help="Force device: 'cpu' or 'cuda' (default: auto-detect GPU)",
    )
    proc_opts.add_argument(
        "-keep_on_cpu",
        action="store_true",
        help="Load data to CPU and process voxels in GPU chunks (for large datasets). "
        "Default: auto-detect based on dataset size (>4GB uses CPU chunking).",
    )
    proc_opts.add_argument(
        "-batch_size",
        type=int,
        default=None,
        help="Number of voxels per batch (default: auto)",
    )
    proc_opts.add_argument(
        "-R2method",
        type=str,
        choices=["auto", "fast", "slow"],
        default="auto",
        help="R² computation method. 'fast' uses streaming stats (~3MB vs ~8GB memory), "
        "requires LORO CV. 'slow' stores full timeseries (for non-LORO CV). "
        "'auto' selects based on CV strategy (default: auto).",
    )
    proc_opts.add_argument(
        "-verbose",
        action="store_true",
        help="Print detailed progress information",
    )
    proc_opts.add_argument(
        "-debug",
        action="store_true",
        help="Full diagnostic mode: saves design figures, runs comparison R² paths, "
        "prints per-run statistics. Implies -verbose. Use when R² values look wrong.",
    )
    proc_opts.add_argument(
        "-dry_run",
        action="store_true",
        help="Fast testing mode: load only first run, generate synthetic data for rest. "
        "Results are nonsensical but pipeline runs quickly for testing.",
    )

    # Output options
    out_opts = parser.add_argument_group("Output Options")
    out_opts.add_argument(
        "-fout",
        action="store_true",
        help="Include F-statistics in stats bucket",
    )
    out_opts.add_argument(
        "-tout",
        action="store_true",
        help="Include t-statistics in stats bucket",
    )
    out_opts.add_argument(
        "-rout",
        action="store_true",
        help="Include R² in stats bucket",
    )
    out_opts.add_argument(
        "-save_hrf_designs",
        action="store_true",
        help="Save individual design matrices for each HRF in the library. "
        "Creates a directory {prefix}_hrf_designs/ with xmat files that "
        "can be used for external GLM fitting (e.g., AFNI's 3dREMLfit).",
    )
    out_opts.add_argument(
        "-save_plots",
        action="store_true",
        help="Save design matrix and HRF library plots as PNG images.",
    )

    # Help
    parser.add_argument("-help", action="store_true", help="Show this help message")

    return parser


def parse_cv_strategy(cv_str: str) -> Union[int, float]:
    """Parse CV strategy string into int or float."""
    cv_str = cv_str.lower().strip()

    if cv_str in ["loro", "loo"]:
        return 1  # Leave-one-run-out

    try:
        # Try parsing as float first
        val = float(cv_str)
        if val == int(val) and val > 1:
            return int(val)  # Leave-N-out
        elif 0 < val < 1:
            return val  # Split fraction
        elif val == 1:
            return 1  # LORO
        else:
            print(f"ERROR: Invalid cv_strategy value: {cv_str}")
            print("  Must be 'loro', int > 0, or float in (0, 1)")
            sys.exit(1)
    except ValueError:
        print(f"ERROR: Could not parse cv_strategy: {cv_str}")
        sys.exit(1)


def print_header(args):
    """Print program header"""
    print("=" * 70)
    print("3dHRFoptfast - GPU-Accelerated Cross-Validated HRF Optimization")
    print("=" * 70)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_summary(args, n_runs: int, n_conditions: int, n_voxels: int, condition_labels: list[str]):
    """Print analysis summary"""
    print("=" * 70)
    print("📋 Analysis Summary")
    print("=" * 70)
    print(f"  Input runs: {n_runs}")
    print(f"  Conditions: {n_conditions} - {condition_labels}")
    print(f"  TR: {args.tr}s")
    print(f"  Voxels: {n_voxels:,}")
    print()
    print(f"  HRF mode: {args.hrf_mode}")
    print(f"  HRF candidates: {args.n_hrfs}")
    print(f"  CV strategy: {args.cv_strategy}")
    bins_per_tr = int(round(args.tr / args.microtime_dt))
    print(f"  Microtime: dt={args.microtime_dt}s ({bins_per_tr} bins/TR)")
    print()
    print(f"  Output prefix: {args.prefix}")
    print("=" * 70)
    print()


def main():
    parser = create_parser()

    # Check for help BEFORE parse_args to avoid required argument errors
    if len(sys.argv) == 1 or "-help" in sys.argv or "--help" in sys.argv:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # Debug implies verbose
    if args.debug:
        args.verbose = True

    print_header(args)

    # ==========================================================================
    # 1. Parse and validate inputs
    # ==========================================================================

    # Parse input files
    input_files = parse_input_files(args.input)
    n_runs = len(input_files)

    if n_runs < 2:
        print("ERROR: At least 2 runs required for cross-validation")
        sys.exit(1)

    # Parse onset files
    onset_files = args.onsets
    n_conditions = len(onset_files)
    condition_labels = [Path(f).stem for f in onset_files]

    # Validate onset files exist
    for f in onset_files:
        if not Path(f).exists():
            print(f"ERROR: Onset file not found: {f}")
            sys.exit(1)

    # Parse durations
    durations = parse_durations(args.durations, n_conditions, condition_labels)
    print(f"  Durations: {durations}s")

    # Parse CV strategy
    cv_strategy = parse_cv_strategy(args.cv_strategy)

    # Pre-flight checks (before slow data loading)
    preflight_check(
        input_files=input_files,
        onset_files=onset_files,
        ortvec_files=[(f, label) for f, label in args.ortvec] if args.ortvec else None,
    )

    # ==========================================================================
    # 2. Load and preprocess data
    # ==========================================================================

    # Parse device argument using unified parser
    device, _, _ = parse_device_arg(args.device)
    print(f"  Device: {device}")

    # Load and preprocess data using shared utility
    # This handles: metadata extraction, blur, masking, scaling, device strategy.
    # Always force CPU: data stays in RAM, GPU used only for computation with
    # chunk streaming. This prevents OOM on load and avoids duplicate data on GPU.
    load_result = load_and_preprocess_runs(
        input_files=input_files,
        tr=args.tr,
        mask_file=args.mask,
        blur_fwhm=args.do_blur,
        do_scale=args.do_scale,
        device=device,
        force_cpu=True,  # Data always on CPU; GPU used for compute only
        dry_run=args.dry_run,
        verbose=True,
    )

    # Modify prefix for dry run mode
    if args.dry_run:
        args.prefix = f"dry_run_{args.prefix}"

    # Extract results (these are reference copies, not deep copies — no memory duplication)
    data = load_result.data
    run_starts = load_result.run_starts
    affine = load_result.affine
    volume_shape = load_result.volume_shape
    voxel_sizes = load_result.voxel_sizes
    tr = load_result.tr
    mask = load_result.mask
    mask_flat = load_result.mask_flat
    n_voxels = load_result.n_voxels
    n_timepoints = load_result.n_timepoints
    scale_info = load_result.scale_info
    violations_mask = load_result.violations_mask

    # Update args.tr with extracted value (for consistency with rest of code)
    if args.tr is None:
        args.tr = tr
        print(f"  TR from header: {tr}s")
    else:
        print(f"  TR (specified): {args.tr}s")

    print(f"  Data shape: {data.shape} ({n_voxels:,} voxels x {n_timepoints} timepoints)")
    print(f"  Volume shape: {volume_shape}")
    print(f"  Runs: {n_runs} starting at {run_starts}")

    # ==========================================================================
    # 3. Parse onset files and build onset matrix
    # ==========================================================================

    print()
    print("Building onset matrix...")

    # Parse onset files (AFNI format: one row per run)
    all_onsets = []
    for i, onset_file in enumerate(onset_files):
        onsets_by_run = parse_afni_timing_file(onset_file)

        # Validate number of runs matches
        if len(onsets_by_run) != n_runs:
            print(
                f"ERROR: Onset file '{onset_file}' has {len(onsets_by_run)} runs, "
                f"but {n_runs} input files were provided"
            )
            sys.exit(1)

        all_onsets.append(onsets_by_run)
        if args.verbose:
            n_events = sum(len(r) for r in onsets_by_run)
            print(f"  {condition_labels[i]}: {n_events} events across {n_runs} runs")

    # Build microtime onset matrix
    # This creates a (n_microtime, n_conditions) matrix with boxcar values
    onset_matrix = create_onset_matrix_microtime(
        all_onsets,
        run_starts,
        args.tr,
        n_timepoints,
        args.microtime_dt,
        stim_durations=durations,
        device=device,
    )

    print(f"  Onset matrix shape: {onset_matrix.shape}")

    # ==========================================================================
    # 4. Generate HRF library
    # ==========================================================================

    print()
    print(f"Generating {args.hrf_mode} HRF library ({args.n_hrfs} candidates)...")

    # Build PIGHS kwargs if using pighs mode
    pighs_kwargs = {}
    if args.hrf_mode == "pighs":
        pighs_kwargs = {
            "peak_time_range": tuple(args.pighs_peak_time_range),
            "rise_fraction_range": tuple(args.pighs_rise_fraction_range),
            "fall_time_range": tuple(args.pighs_fall_range),
            "recovery_time_range": tuple(args.pighs_recovery_range),
            "undershoot_range": tuple(args.pighs_undershoot_range),
        }

    # Generate HRF library as IMPULSE RESPONSES (stim_duration=0)
    # Duration-based convolution is handled separately in the design matrix construction
    # This ensures the HRF library represents the actual HRF shapes, not duration-convolved versions
    hrf_library = get_hrf_library(
        mode=args.hrf_mode,
        stim_duration=0.0,  # Impulse response - duration handled in design matrix
        microtime_dt=args.microtime_dt,
        n_hrfs=args.n_hrfs,
        device=device,
        **pighs_kwargs,
    )

    print(f"  HRF library shape: {hrf_library.shape}")
    print(
        f"  HRF length: {hrf_library.shape[1]} samples ({hrf_library.shape[1] * args.microtime_dt}s)"
    )

    # ==========================================================================
    # 5. Run cross-validated HRF selection
    # ==========================================================================

    print_summary(args, n_runs, n_conditions, n_voxels, condition_labels)

    # Convert ortvec argument to list of tuples if provided
    ortvec_files = None
    if args.ortvec:
        ortvec_files = [(f, label) for f, label in args.ortvec]
        if args.verbose:
            for f, label in ortvec_files:
                print(f"  Nuisance: {f} (label={label})")

    if args.single_trials:
        # ========== SINGLE-TRIAL IN-SAMPLE R² PATH (GLMsingle Type-B) ==========
        # Matches GLMsingle's FitHRF step: for each HRF candidate, fit a single-trial
        # OLS model (resampling=0) and use in-sample R² on nuisance-projected data
        # for HRF selection. All HRFs have equal model complexity (one beta per trial),
        # so in-sample R² is a fair comparison metric.
        from tqdm import tqdm

        from fastfuncsim.ridge import create_single_trial_design
        from fastfuncsim.xval import compute_qr_projectors, compute_r2_metric

        print()
        print("=" * 70)
        print("Single-trial HRF fitting (GLMsingle Type-B style)")
        print("=" * 70)
        print()

        # Build nuisance design (polynomials + ortvec, block-diagonal)
        run_lengths = compute_run_lengths(run_starts, n_timepoints)

        # Auto-determine polort if not specified
        if args.polort is None:
            avg_run_duration = get_average_run_duration(run_lengths, args.tr)
            polort = auto_polort(avg_run_duration, formula="afni")
            print(f"  Auto-determined polort: {polort} (avg run duration: {avg_run_duration:.1f}s)")
        else:
            polort = args.polort

        # Build nuisance per run using shared utility
        nuisance_per_run = build_nuisance_per_run(
            run_starts=run_starts,
            n_timepoints=n_timepoints,
            polort=polort,
            device=device,
            ortvec_files=ortvec_files,
            verbose=False,
        )

        # Create block-diagonal nuisance design
        nuisance_design = torch.block_diag(*nuisance_per_run)
        print(f"  Nuisance design shape: {nuisance_design.shape}")

        n_hrfs = hrf_library.shape[0]

        # ---- Project nuisance from data ONCE (reused for all HRFs) ----
        # This matches GLMsingle: R² is computed on nuisance-projected data,
        # so it measures task-related variance only (not drift).
        print("  Projecting nuisance from data (once for all HRFs)...")
        q_factors = compute_qr_projectors(nuisance_per_run, run_starts, device=device)

        projected_data = data.clone()
        for run_idx in range(n_runs):
            start_tp = run_starts[run_idx]
            end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            Q = q_factors[run_idx]
            if Q is not None:
                run_data = projected_data[:, start_tp:end_tp].to(device)
                projected_data[:, start_tp:end_tp] = (run_data - (Q @ (Q.T @ run_data.T)).T).cpu()

        # Storage for R² per HRF (in-sample, on projected data)
        fit_r2_all = torch.zeros(n_voxels, n_hrfs)

        # Chunk size for voxel streaming (avoid GPU OOM)
        chunk_size = args.batch_size or 50_000
        print(f"  Voxel chunk size: {chunk_size:,}")

        # Cache projected designs on CPU for second-pass beta recomputation (~12 MB each)
        projected_designs_cache = []

        # Canonical betas: stored during first pass (lazy-init)
        canonical_betas = None  # Will become (n_voxels, n_trials) on CPU
        n_trials = None

        print(f"  Evaluating {n_hrfs} HRFs (in-sample R² on projected data)...")
        for hrf_idx in tqdm(range(n_hrfs), desc="HRF evaluation"):
            hrf = [hrf_library[hrf_idx]]  # Wrap single HRF in list

            # Build single-trial design with this HRF
            st_design, labels, cond_ids, run_ids, cond_design = create_single_trial_design(
                onsets_by_condition=all_onsets,
                durations=durations,
                run_starts=run_starts,
                tr=args.tr,
                n_timepoints=n_timepoints,
                hrf_library=hrf,
                microtime_dt=args.microtime_dt,
                condition_labels=condition_labels,
                device=device,
            )

            # Project nuisance from single-trial design (per-run, matching data projection)
            projected_st_design = st_design.clone()
            for run_idx in range(n_runs):
                start_tp = run_starts[run_idx]
                end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                Q = q_factors[run_idx]
                if Q is not None:
                    run_design = st_design[start_tp:end_tp].to(device)
                    projected_st_design[start_tp:end_tp] = (
                        run_design - Q @ (Q.T @ run_design)
                    ).cpu()

            # Cache projected design for second-pass beta recomputation
            projected_designs_cache.append(projected_st_design)

            # Lazy-init canonical_betas now that we know n_trials
            if canonical_betas is None:
                n_trials = projected_st_design.shape[1]
                canonical_betas = torch.zeros(n_voxels, n_trials)

            # Fit OLS in voxel chunks (projected data stays on CPU, stream to GPU)
            proj_design_dev = projected_st_design.to(device)

            for c0 in range(0, n_voxels, chunk_size):
                c1 = min(c0 + chunk_size, n_voxels)
                chunk_data = projected_data[c0:c1].to(device)

                chunk_betas = torch.linalg.lstsq(proj_design_dev, chunk_data.T).solution.T
                chunk_pred = (proj_design_dev @ chunk_betas.T).T
                chunk_r2 = compute_r2_metric(chunk_data, chunk_pred, metric="cod")

                fit_r2_all[c0:c1, hrf_idx] = chunk_r2.cpu()

                # Store canonical (idx 0) betas for beta-series CV
                if hrf_idx == 0:
                    canonical_betas[c0:c1] = chunk_betas.cpu()

            if args.verbose and hrf_idx % 5 == 0:
                col_r2 = fit_r2_all[:, hrf_idx]
                print(f"    HRF {hrf_idx}: mean R²={col_r2.mean():.4f}, median R²={col_r2.median():.4f}")

        # Select best HRF per voxel (matches GLMsingle: max over HRFs)
        hrf_index = fit_r2_all.argmax(dim=1).to(device)
        xval_r2_best = fit_r2_all[torch.arange(n_voxels), fit_r2_all.argmax(dim=1)]

        print()
        print("Best HRF selection complete:")
        print(f"  Mean R²: {xval_r2_best.mean():.4f}")
        print(f"  Median R²: {xval_r2_best.median():.4f}")
        print(f"  % positive: {100 * (xval_r2_best > 0).float().mean():.1f}%")

        # ---- Second pass: assemble best-HRF betas per voxel (grouped by HRF) ----
        print()
        print("Assembling best-HRF betas per voxel...")
        best_betas = torch.zeros(n_voxels, n_trials)
        hrf_index_cpu = hrf_index.cpu()

        for hrf_idx in range(n_hrfs):
            vox_mask = (hrf_index_cpu == hrf_idx)
            if not vox_mask.any():
                continue
            vox_indices = vox_mask.nonzero(as_tuple=True)[0]
            proj_design_dev = projected_designs_cache[hrf_idx].to(device)

            for c0 in range(0, len(vox_indices), chunk_size):
                c1 = min(c0 + chunk_size, len(vox_indices))
                idx = vox_indices[c0:c1]
                chunk_data = projected_data[idx].to(device)
                chunk_betas = torch.linalg.lstsq(proj_design_dev, chunk_data.T).solution.T
                best_betas[idx] = chunk_betas.cpu()

        del projected_designs_cache  # Free ~240 MB

        # Create HRFSelectionResults object
        from fastfuncsim.hrf_selection import HRFSelectionResults

        xval_r2_std = torch.zeros_like(xval_r2_best)  # Not meaningful for in-sample R²

        # Compute HRF usage counts for reporting
        hrf_usage_counts = torch.bincount(hrf_index_cpu, minlength=n_hrfs).tolist()

        # Canonical baseline: first HRF in library (index 0)
        xval_r2_canonical = fit_r2_all[:, 0].to(device)

        # ---- Clearly-named R² maps ----
        # In-sample R² (already computed above)
        canonical_full_r2 = fit_r2_all[:, 0]           # canonical HRF, in-sample
        hrfopt_full_r2 = xval_r2_best                  # best-HRF per voxel, in-sample

        # Beta-series CV R² (genuine cross-validated)
        from fastfuncsim.xval import generate_cv_splits, compute_xval_r2_single_trials

        cv_splits = generate_cv_splits(n_runs, strategy=1)  # LORO

        print()
        print("Computing beta-series CV R² (canonical HRF)...")
        canonical_cv = compute_xval_r2_single_trials(
            canonical_betas, cond_ids, run_ids, cv_splits, metric="cod", device=device,
        )
        canonical_xval_r2 = canonical_cv["r2"]

        print("Computing beta-series CV R² (optimal HRF per voxel)...")
        hrfopt_cv = compute_xval_r2_single_trials(
            best_betas, cond_ids, run_ids, cv_splits, metric="cod", device=device,
        )
        hrfopt_xval_r2 = hrfopt_cv["r2"]

        del canonical_betas, best_betas  # Free ~1.4 GB

        print(f"  Canonical in-sample R²: mean={canonical_full_r2.mean():.4f}")
        print(f"  HRFopt   in-sample R²: mean={hrfopt_full_r2.mean():.4f}")
        print(f"  Canonical CV R²:        mean={canonical_xval_r2.mean():.4f}")
        print(f"  HRFopt   CV R²:         mean={hrfopt_xval_r2.mean():.4f}")

        results = HRFSelectionResults(
            hrf_index=hrf_index,
            xval_r2_best=xval_r2_best,
            xval_r2_std=xval_r2_std,
            xval_r2_all_hrfs=fit_r2_all.to(device),
            xval_r2_canonical=xval_r2_canonical,
            canonical_full_r2=canonical_full_r2,
            hrfopt_full_r2=hrfopt_full_r2,
            canonical_xval_r2=canonical_xval_r2,
            hrfopt_xval_r2=hrfopt_xval_r2,
            final_results=None,  # TODO: refit with optimal HRFs
            canonical_results=None,  # TODO: fit with canonical HRF
            hrf_library=hrf_library,
            hrf_metadata={
                "mode": "single_trial_insample_r2",
                "n_hrfs": n_hrfs,
                "hrf_usage_counts": hrf_usage_counts,
            },
        )

        if args.save_canonical_betas:
            print()
            print("Fitting canonical HRF for comparison...")

            # Get canonical HRF (usually first in library)
            # hrf_library is (n_hrfs, n_timepoints), get first HRF as 1D tensor
            canonical_hrf = hrf_library[0] if hrf_library.dim() == 2 else hrf_library

            # Build canonical condition-level design
            canonical_results = _fit_voxelwise_hrf_canonical(
                data=data,
                onsets=onset_matrix,
                canonical_hrf=canonical_hrf,
                nuisance_design=nuisance_design,
                tr=tr,
                microtime_dt=args.microtime_dt,
                microtime_onset=0,  # Default: sample at start of TR
                device=device,
                verbose=args.verbose,
            )

            # Update results
            results.canonical_results = canonical_results
            print("  Canonical fit complete.")

        # ==========================================================================
        # 5b. Optional: Refit with canonical/optimal HRFs for single-trial betas
        # ==========================================================================
        # Note: This section only applies to beta-space CV path (single-trial mode)
        if args.save_single_trial_betas:
            print()
            print("Refitting with optimal HRF per voxel (single-trial betas)...")

            # Convert hrf_library from tensor (n_hrfs, n_timepoints) to list of 1D tensors
            hrf_library_list = [hrf_library[i] for i in range(hrf_library.shape[0])]

            # Refit with optimal HRF per voxel
            final_results = _fit_voxelwise_hrf_single_trial(
                data=data,
                onsets_by_condition=all_onsets,
                hrf_library=hrf_library_list,
                hrf_index=hrf_index,
                nuisance_design=nuisance_design,
                durations=durations,
                run_starts=run_starts,
                tr=tr,
                n_timepoints=n_timepoints,
                microtime_dt=args.microtime_dt,
                condition_labels=condition_labels,
                device=device,
                verbose=args.verbose,
            )

            # Update results
            results.final_results = final_results
            print("  Single-trial refit complete.")

    else:
        # ========== EXISTING Beta @ Design TIMESERIES CV PATH ==========
        results = fit_glm_hrf_library_with_xval(
            data=data,
            onsets=onset_matrix,
            hrf_library=hrf_library,
            tr=args.tr,
            run_starts=run_starts,
            stim_durations=durations,
            cv_strategy=cv_strategy,
            n_perms=args.n_perms,
            metric=args.metric,
            microtime_dt=args.microtime_dt,
            polort=args.polort,
            ortvec_files=ortvec_files,
            canonical_mode=args.canonical,
            device=device,
            verbose=args.verbose,
            chunk_size=args.batch_size,
            r2_method=args.R2method,
            debug=args.debug,
            debug_prefix=args.prefix,
            condition_labels=condition_labels,
        )

    # Update metadata with CLI parameters
    results.hrf_metadata["hrf_mode"] = args.hrf_mode
    results.hrf_metadata["canonical_mode"] = args.canonical
    results.hrf_metadata["condition_labels"] = condition_labels
    results.hrf_metadata["input_files"] = input_files
    results.hrf_metadata["onset_files"] = onset_files
    results.hrf_metadata["durations"] = durations
    if ortvec_files:
        results.hrf_metadata["ortvec_files"] = [(str(f), label) for f, label in ortvec_files]

    # ==========================================================================
    # 6. Save outputs
    # ==========================================================================

    print()
    print("Saving outputs...")

    # Create output directory if needed
    output_prefix = Path(args.prefix)
    output_dir = output_prefix.parent
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Convert mask to voxel_mask tensor if provided
    voxel_mask = None
    if mask is not None:
        voxel_mask = torch.from_numpy(mask.flatten().astype(bool))

    # Save results
    # Note: If we have single-trial betas, we need to save them separately with trial labels
    # So temporarily remove final_results from the results object
    final_results_temp = results.final_results

    if args.save_single_trial_betas and results.final_results is not None:
        results.final_results = (
            None  # Temporarily remove to prevent save_hrf_selection_results from saving it
        )

    output_files = save_hrf_selection_results(
        results=results,
        output_prefix=str(args.prefix),
        volume_shape=volume_shape,
        affine=affine,
        voxel_mask=voxel_mask,
        condition_labels=condition_labels,
        run_starts=run_starts,
        save_all_hrf_designs=args.save_hrf_designs,
        onsets=onset_matrix if args.save_hrf_designs else None,
        save_plots=args.save_plots,
    )

    # Restore final_results for custom saving
    if args.save_single_trial_betas and final_results_temp is not None:
        results.final_results = final_results_temp

    # ==========================================================================
    # 6b. Custom saving for single-trial betas (if requested)
    # ==========================================================================
    # Note: The canonical betas are already saved correctly by save_hrf_selection_results()
    # to {prefix}_canonical_stats.nii.gz. For single-trial betas, we need custom saving
    # to {prefix}_stats_single_trial.nii.gz instead of the default {prefix}_stats.nii.gz
    if args.save_single_trial_betas and results.final_results is not None:
        from fastfuncsim.glm_outputs import write_glm_bucket_as_nifti

        print("  Saving single-trial betas with custom filename...")

        # Set required metadata for saving (same as canonical_results)
        results.final_results.original_shape = volume_shape
        results.final_results.affine = affine
        if voxel_mask is not None:
            results.final_results.voxel_mask = voxel_mask

        # Get trial labels from results (stored during refit)
        trial_labels = results.final_results.trial_labels
        if trial_labels is None:
            print("  WARNING: No trial labels found, using generic names")
            n_trials = results.final_results.betas.shape[1]
            trial_labels = [f"trial_{i:04d}" for i in range(n_trials)]

        # Save with custom filename using trial labels
        single_trial_file = f"{args.prefix}_stats_single_trial.nii.gz"
        write_glm_bucket_as_nifti(
            results.final_results,
            output_path=single_trial_file,
            condition_names=trial_labels,  # Use trial labels, not condition labels!
            volume_shape=volume_shape,
            affine=affine,
            apply_afni_metadata=True,
        )
        output_files["single_trial_betas"] = single_trial_file
        print(f"  Single-trial betas saved: {single_trial_file}")

    # Save scaling violation mask if scaling was performed
    if args.do_scale and violations_mask is not None and scale_info is not None:
        # Sum violations across time to get count per voxel
        violation_counts = violations_mask.cpu().sum(dim=1).numpy()  # (n_voxels,)

        # Reshape back to volume
        if mask is not None:
            violation_vol = np.zeros(np.prod(volume_shape), dtype=np.float32)
            violation_vol[mask_flat] = violation_counts
        else:
            violation_vol = violation_counts
        violation_vol = violation_vol.reshape(volume_shape)

        # Save as NIfTI
        violation_path = f"{args.prefix}_scale_violations.nii.gz"
        violation_img = nib.Nifti1Image(violation_vol, affine)
        nib.save(violation_img, violation_path)
        output_files["scale_violations"] = violation_path

        if scale_info["n_violations"] > 0:
            print(f"  ⚠️  Scale violations saved: {violation_path}")

        # Add scaling info to metadata
        results.hrf_metadata["do_scale"] = True
        results.hrf_metadata["scale_max"] = 200.0
        results.hrf_metadata["scale_n_violations"] = scale_info["n_violations"]
        results.hrf_metadata["scale_n_voxels_with_violations"] = scale_info[
            "n_voxels_with_violations"
        ]

    # Print output summary
    print()
    print("=" * 70)
    print("📁 Output Files")
    print("=" * 70)
    for output_type, filepath in output_files.items():
        print(f"  {output_type}: {filepath}")
    print("=" * 70)

    # Print final summary
    print()
    print("=" * 70)
    print("✅ 3dHRFoptfast Complete!")
    print("=" * 70)
    print(f"  Mean xval R²: {results.xval_r2_best.mean().item():.4f}")
    if results.xval_r2_canonical is not None:
        canonical_r2 = results.xval_r2_canonical.mean().item()
        best_r2 = results.xval_r2_best.mean().item()
        print(f"  Canonical HRF baseline R²: {canonical_r2:.4f}")
        print(f"  Improvement over canonical: {best_r2 - canonical_r2:+.4f}")
    if results.final_results is not None:
        print(f"  Final R² (full data): {results.final_results.r2.mean().item():.4f}")
    print()
    print("  HRF usage distribution:")
    hrf_counts = results.hrf_metadata["hrf_usage_counts"]
    for i, count in enumerate(hrf_counts):
        if count > 0:
            pct = 100 * count / n_voxels
            print(f"    HRF {i}: {count:,} voxels ({pct:.1f}%)")
    print()
    print(f"🕐 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
