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

import sys
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter

try:
    import nibabel as nib  # noqa: F401
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

# Import fastfuncstuff modules
try:
    from fastfuncstuff.cli_utils import (
        add_cv_blur_arg,
        add_cv_metric_arg,
        add_cv_strategy_arg,
        add_device_arg,
        add_load_threads_arg,
        add_noise_ceiling_args,
        add_ortvec_arguments,
        add_single_trial_args,
        add_trim_args,
        add_verbose_arg,
        apply_trim_to_timing,
        auto_polort,
        blur_masked_data,
        build_nuisance_per_run,
        collect_nuisance_blocks,
        compute_run_lengths,
        get_average_run_duration,
        load_and_preprocess_runs,
        parse_cv_strategy,
        parse_input_files,
        parse_prefix,
        preflight_check,
        resolve_cv_design,
        resolve_microtime_dt,
        run_lengths_from_starts,
        save_r2_ceiling_stack,
        setup_device,
        spinner,
        summarize_trial_repeats,
        trim_spec_from_args,
    )
    from fastfuncstuff.design.builder import (
        create_onset_matrix_microtime,
    )
    from fastfuncstuff.design.hrf import get_hrf_library
    from fastfuncstuff.design.hrf_selection import (
        _fit_voxelwise_hrf_canonical,
        _fit_voxelwise_hrf_single_trial,
        fit_glm_hrf_library_with_xval,
        save_hrf_selection_results,
    )
    from fastfuncstuff.design.stim_vec import (
        add_stim_vec_arguments,
        append_stim_vecs_to_single_trial_design,
        collect_stim_vec_blocks,
        stim_vec_bucket_labels,
    )
    from fastfuncstuff.io.afni import save_nifti
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    print("Make sure fastfuncstuff is installed: pip install -e .")
    sys.exit(1)


def create_parser():
    """Create argument parser"""
    parser = FfsArgumentParser(
        description="3dHRFoptfast - Fast GPU-accelerated cross-validated HRF optimization",
        formatter_class=FfsHelpFormatter,
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
  {prefix}_hrf_index.nii.gz       - 2-sub-brick bucket: [0] HRF index (1-N), [1] R² at selected HRF
                                     Use sub-brick [1] to threshold the index map in AFNI
  {prefix}_xval_r2.nii.gz         - Cross-validated R² for selected HRF (standalone copy)
  {prefix}_xval_r2_all_hrfs.nii.gz - 4D: CV R² for each HRF (volume per HRF)
  {prefix}_selected_hrfs.nii.gz   - 4D: the winning HRF shape per voxel (x,y,z,hrf_timepoints at microtime_dt)
  {prefix}_stats.nii.gz           - Final GLM betas and t-stats (AFNI bucket format)
  {prefix}_hrf_library.pt         - HRF library + voxel assignments for ARMA reuse
  {prefix}_metadata.json          - Full metadata for reproducibility

Notes:
  - Durations can be specified as single value (applies to all) or one per condition
  - Nuisance files (-ortvec) must be pre-concatenated across runs (matching total timepoints)
  - Common nuisance files: motion parameters (6 columns), physiological regressors, etc.
  - Future: onset files with 'married' durations (e.g., "1:2 4:5") will be supported
  - HRF library is saved for later ARMA/REML analysis with ffs_reml
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
        required=False,
        default=None,
        help="Onset timing files (AFNI format). One file per condition. "
        "Mutually exclusive with -events.",
    )
    required.add_argument(
        "-durations",
        nargs="+",
        required=False,
        default=None,
        help="Stimulus durations in seconds. Either single value for all conditions, or one per condition. "
        "Required when using -onsets; derived automatically from -events.",
    )

    # BIDS events options
    bids_opts = parser.add_argument_group("BIDS Events (alternative to -onsets/-durations)")
    bids_opts.add_argument(
        "-events",
        nargs="+",
        default=None,
        metavar="TSV",
        help="BIDS *_events.tsv files, one per run. Sorted by run number automatically. "
        "Mutually exclusive with -onsets.",
    )
    bids_opts.add_argument(
        "-event_ignore",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="trial_type values to exclude (e.g. -event_ignore fixation null). "
        "Only valid with -events.",
    )
    bids_opts.add_argument(
        "-event_cols",
        nargs=3,
        default=None,
        metavar=("ONSET_COL", "DURATION_COL", "TRIAL_TYPE_COL"),
        help="Custom column names for onset, duration, trial_type. "
        "Default: onset duration trial_type. Only valid with -events.",
    )
    bids_opts.add_argument(
        "-round_onsets",
        nargs="?",
        const=0.7,
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Round onsets to nearest TR. Fraction-through-TR >= THRESHOLD → ceil, else floor. "
        "Default threshold if flag given without value: 0.7.",
    )
    bids_opts.add_argument(
        "-round_durations",
        type=int,
        default=None,
        metavar="PLACES",
        help="Round stimulus durations to PLACES decimal places before condition uniquing "
        "(prevents 3.03 vs 3.0 being treated as distinct). Applied per-event inside "
        "-events parsing; applied to final list in -onsets mode.",
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
    hrf_opts.add_argument(
        "-hrf-library",
        dest="hrf_library",
        default=None,
        metavar="TSV",
        help=(
            "Path to a custom HRF library TSV (same format as "
            "getcanonicalhrflibrary.tsv, e.g. produced by ffs_librarian). "
            "Used only when -hrf_mode library."
        ),
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
    add_cv_strategy_arg(cv_opts)
    cv_opts.add_argument(
        "-select",
        choices=["xval", "full"],
        default="xval",
        help=(
            "HRF selection criterion. "
            "'xval': cross-validated R² (LORO or split-half, default). "
            "'full': in-sample R² on all data (GLMsingle FITHRF behaviour; "
            "faster, automatically chosen when only one run is present)."
        ),
    )
    cv_opts.add_argument(
        "-n_perms",
        type=int,
        default=100,
        help="Max number of CV permutations for random splits (default: 100)",
    )
    add_cv_metric_arg(cv_opts, dest="metric")
    add_single_trial_args(
        cv_opts,
        emit_help="Refit with the optimal HRF per voxel and save one beta per "
        "trial (GLMsingle-style) to {prefix}_stats_single_trial.nii.gz. Voxels "
        "are processed in HRF groups with chunking to avoid OOM. By default the "
        "HRF is then also selected in single-trial space; see -cv_design to "
        "select it on the condition-level design instead (required when "
        "conditions do not repeat across runs).",
    )
    cv_opts.add_argument(
        "-save_single_trial_betas",
        "-save-single-trial-betas",
        dest="single_trials",
        action="store_true",
        help="Deprecated alias for -single_trials (emitting per-trial betas and "
        "selecting the HRF in single-trial space are now separate: see -cv_design).",
    )
    add_noise_ceiling_args(
        cv_opts,
        stage_note="Available on the beta-space CV path only, and built from the "
        "per-voxel best-HRF betas so it bounds the selected model rather than the "
        "canonical baseline.",
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
    add_ortvec_arguments(proc_opts)
    add_stim_vec_arguments(proc_opts)
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
        help="Apply 3D Gaussian spatial smoothing with FWHM in mm (whole pipeline, "
        "including the saved betas). Smoothing is applied BEFORE masking to avoid "
        "edge effects. Typical values: 4-8 mm. Uses separable convolutions for speed. "
        "See -cv_blur to blur only HRF selection.",
    )
    add_cv_blur_arg(
        proc_opts,
        stage_note=(
            "Applies to HRF selection whether or not it is cross-validated: with "
            "-cv_design single the criterion is in-sample R², and it benefits from "
            "the same de-noising of the search landscape."
        ),
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
    add_device_arg(proc_opts)
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
    add_load_threads_arg(proc_opts)
    add_trim_args(proc_opts)
    add_verbose_arg(proc_opts, default=0)
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
    out_opts.add_argument(
        "-delta_denoise",
        action="store_true",
        help=(
            "Quantify how much the user-supplied nuisance regressors (-ortvec) "
            "changed HRF selection. Re-runs the entire selection pass with "
            "ortvec stripped, then writes a second set of outputs under "
            "{prefix}_nodenoise_* and four delta maps under {prefix}_delta_*: "
            "_delta_xval_r2 (with − without), _delta_hrfopt_r2 (same for in-sample), "
            "_delta_hrf_changed (1 where selected HRF differs), and "
            "_delta_hrf_index (signed shift in HRF index). "
            "Requires at least one -ortvec. Not supported with -cv_design single."
        ),
    )

    return parser


def print_header(args):
    """Print program header"""
    from fastfuncstuff.cli_utils import print_cli_header

    print_cli_header("ffs_hrfopt", "GPU-accelerated cross-validated HRF optimization")


def print_summary(args, n_runs: int, n_conditions: int, n_voxels: int, condition_labels: list[str]):
    """Print analysis summary"""
    from fastfuncstuff.cli_utils import print_cli_section

    print_cli_section("Analysis summary", leading_blank=False)
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
    print()


def main():
    parser = create_parser()

    # Check for help BEFORE parse_args to avoid required argument errors
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    pfx = parse_prefix(args.prefix)
    args.prefix = pfx.stem  # overwrite with clean stem
    _nii_ext = pfx.nifti_ext

    # Debug implies verbose
    if args.debug:
        args.verb = max(args.verb, 1)

    print_header(args)

    # ==========================================================================
    # 0. Validate flag combinations that don't depend on input files
    # ==========================================================================
    any_ortvec = bool(args.ortvec or args.ortvec_run or args.ortvec_glob or args.ortvec_concat)
    if args.delta_denoise:
        if not any_ortvec:
            print(
                "ERROR: -delta_denoise requires at least one -ortvec / "
                "-ortvec_run / -ortvec_glob / -ortvec_concat. It compares HRF "
                "selection with vs. without user-supplied nuisance; without any, "
                "there is nothing to compare."
            )
            sys.exit(1)
        if args.cv_design == "single":
            print(
                "ERROR: -delta_denoise is not yet supported with -cv_design single "
                "(beta-space CV path). Use -cv_design condition to measure the "
                "nuisance effect; -single_trials output is unaffected."
            )
            sys.exit(1)

    # ==========================================================================
    # 1. Parse and validate inputs
    # ==========================================================================

    # Parse input files
    input_files = parse_input_files(args.input)
    n_runs = len(input_files)

    if n_runs < 2:
        print("ERROR: At least 2 runs required for cross-validation")
        sys.exit(1)

    # Validate onset/events mutual exclusivity
    _has_onsets = bool(args.onsets)
    _has_events = bool(args.events)
    if _has_onsets and _has_events:
        print("ERROR: Specify only one of -onsets/-durations or -events")
        sys.exit(1)
    if not _has_onsets and not _has_events:
        print("ERROR: Must specify one of -onsets/-durations or -events")
        sys.exit(1)
    if args.event_ignore and not _has_events:
        print("ERROR: -event_ignore requires -events")
        sys.exit(1)
    if args.event_cols and not _has_events:
        print("ERROR: -event_cols requires -events")
        sys.exit(1)
    if _has_onsets and args.durations is None:
        print("ERROR: -durations is required when using -onsets")
        sys.exit(1)

    # Parse onset files / BIDS events (early pass for condition metadata)
    from fastfuncstuff.cli_utils import parse_timing_spec

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
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    all_onsets = timing.all_onsets
    durations = timing.durations
    condition_labels = timing.condition_labels
    n_conditions = timing.n_conditions
    onset_files = timing.onset_files

    # Parse CV strategy
    cv_strategy = parse_cv_strategy(args.cv_strategy)

    # Resolve -cv_design before any data is loaded: a design with no cross-run
    # condition repeats cannot be scored in single-trial space, and finding that
    # out after the HRF sweep wastes the whole run.
    trial_repeats = summarize_trial_repeats(all_onsets)
    cv_design = resolve_cv_design(
        args.cv_design,
        args.single_trials,
        trial_repeats,
        parameter="HRF",
        # Single-trial HRF selection is in-sample (every candidate HRF costs one
        # beta per trial, so complexity is equal), which makes it the one path
        # that survives an events file with no cross-run repeats at all.
        single_needs_repeats=False,
        manual_hint=(
            "Add -single_trials: its HRF selection is in-sample and needs no repeats. "
            "Otherwise give -events a trial_type column that repeats across runs "
            "(-event_cols)."
        ),
        # Always printed, even at -verb 0: a silent switch between selection
        # designs is exactly the kind of thing that gets misread later.
        verbose=True,
    )

    if args.delta_denoise and cv_design == "single":
        print(
            "ERROR: -delta_denoise is not yet supported with single-trial HRF "
            "selection. Re-run with -cv_design condition (per-trial betas are "
            "still written if -single_trials is set)."
        )
        sys.exit(1)

    # Pre-flight checks (before slow data loading)
    preflight_check(
        input_files=input_files,
        onset_files=onset_files,
        ortvec_files=[(f, label) for f, label in args.ortvec] if args.ortvec else None,
    )

    # ==========================================================================
    # 2. Load and preprocess data
    # ==========================================================================

    # setup_device preserves the shared CPU-thread precedence: an explicit
    # cpu,N wins, then FFS_NUM_THREADS, OMP_NUM_THREADS, and scheduler limits.
    device = setup_device(args.device)
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
        load_threads=args.load_threads,
        drop_first=args.drop_first,
        drop_last=args.drop_last,
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
    args.microtime_dt = resolve_microtime_dt(args.tr, args.microtime_dt)

    print(f"  Data shape: {data.shape} ({n_voxels:,} voxels x {n_timepoints} timepoints)")
    print(f"  Volume shape: {volume_shape}")
    print(f"  Runs: {n_runs} starting at {run_starts}")

    # The timing was parsed before the load (the condition list gates other
    # flags), so the -drop_first shift lands here, once the TR is known.
    trim = trim_spec_from_args(args, tr=tr)
    apply_trim_to_timing(
        timing,
        trim,
        run_lengths_tr=run_lengths_from_starts(run_starts, n_timepoints),
        n_runs=n_runs,
    )
    all_onsets = timing.all_onsets

    # ==========================================================================
    # 3. Parse onset files and build onset matrix
    # ==========================================================================

    print()
    print("Building onset matrix...")

    # Apply onset rounding (after TR is known from data load)
    if args.round_onsets is not None:
        from fastfuncstuff.design.builder import round_onsets as _round_onsets

        all_onsets = _round_onsets(all_onsets, args.tr, threshold=args.round_onsets)

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
        library_path=args.hrf_library,
        **pighs_kwargs,
    )
    if args.hrf_library:
        print(f"  Loaded custom HRF library from {args.hrf_library}")

    print(f"  HRF library shape: {hrf_library.shape}")
    print(
        f"  HRF length: {hrf_library.shape[1]} samples ({hrf_library.shape[1] * args.microtime_dt}s)"
    )

    # ==========================================================================
    # 5. Run cross-validated HRF selection
    # ==========================================================================

    print_summary(args, n_runs, n_conditions, n_voxels, condition_labels)

    # Collect user-supplied nuisance regressors into NuisanceBlock list.
    nuisance_blocks = collect_nuisance_blocks(
        args,
        run_starts,
        n_timepoints,
        verbose=(args.verb >= 1),
        trim=trim,
    )
    # Continuous stimulus vectors join the STIM design, re-convolved with each
    # candidate HRF inside the selection loop. Leaving a strong background in the
    # residual would let it steer which HRF wins.
    stim_vec_blocks = collect_stim_vec_blocks(
        args,
        run_starts,
        n_timepoints,
        trim=trim,
        verbose=(args.verb >= 1),
    )

    # One label per task COLUMN for the output writers. condition_labels itself
    # stays pristine: it is the CONDITION list, and the single-trial builders
    # would read an extra entry as an extra condition.
    task_column_labels = list(condition_labels) + stim_vec_bucket_labels(stim_vec_blocks)

    # Legacy variable retained for back-compat fields (metadata + delta-denoise label).
    ortvec_files = [(f, label) for f, label in args.ortvec] if args.ortvec else None

    results_nodenoise = None  # populated only when -delta_denoise is set

    # -cv_blur: HRF *selection* reads cv_data, the final refit reads `data`. The
    # selected quantity is an index into the HRF library — a property of the
    # voxel's hemodynamics — so it carries over to the unblurred fit unchanged.
    cv_data = data
    if args.cv_blur is not None:
        cv_data = blur_masked_data(
            data,
            fwhm_mm=args.cv_blur,
            volume_shape=volume_shape,
            voxel_sizes=voxel_sizes,
            mask_flat=mask_flat,
            run_starts=run_starts,
            device=device,
            verbose=args.verb >= 1,
        )
        print(f"  HRF selection uses {args.cv_blur} mm blurred data; final fit does not.")

    # Nuisance design is built up front because both the single-trial selection
    # path and the single-trial *refit* (which can now follow condition-level
    # selection) need the same block-diagonal nuisance.
    nuisance_per_run = None
    nuisance_design = None
    # Only the beta-space CV branch can produce a ceiling; bound here so the
    # save block below can test it without caring which branch ran.
    beta_ceiling = None
    if cv_design == "single" or args.single_trials:
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
            blocks=nuisance_blocks,
            verbose=False,
        )

        # Create block-diagonal nuisance design
        nuisance_design = torch.block_diag(*nuisance_per_run)
        print(f"  Nuisance design shape: {nuisance_design.shape}")

    if cv_design == "single":
        # ========== SINGLE-TRIAL IN-SAMPLE R² PATH (GLMsingle Type-B) ==========
        # Matches GLMsingle's FitHRF step: for each HRF candidate, fit a single-trial
        # OLS model (resampling=0) and use in-sample R² on nuisance-projected data
        # for HRF selection. All HRFs have equal model complexity (one beta per trial),
        # so in-sample R² is a fair comparison metric.
        from tqdm import tqdm

        from fastfuncstuff.glm.ridge import create_single_trial_design
        from fastfuncstuff.glm.xval import compute_qr_projectors, compute_r2_metric

        print()
        print("=" * 70)
        print("Single-trial HRF fitting (GLMsingle Type-B style)")
        print("=" * 70)
        print()

        n_hrfs = hrf_library.shape[0]

        # ---- Project nuisance from data ONCE (reused for all HRFs) ----
        # This matches GLMsingle: R² is computed on nuisance-projected data,
        # so it measures task-related variance only (not drift).
        print("  Projecting nuisance from data (once for all HRFs)...")
        q_factors = compute_qr_projectors(nuisance_per_run, run_starts, device=device)

        projected_data = cv_data.clone()
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
        if args.batch_size:
            chunk_size = args.batch_size
        else:
            from fastfuncstuff.memory import estimate_chunk_size

            n_regressors_for_chunk = len(condition_labels) * max(
                1, len(all_onsets[0]) // len(condition_labels)
            )
            chunk_size = estimate_chunk_size(
                n_voxels=n_voxels,
                n_timepoints=n_timepoints,
                n_regressors=n_regressors_for_chunk,
                device=device,
                operation="glm",
                verbose=False,
            )
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

            # Stim vectors ride along, convolved with THIS candidate HRF, but
            # they are not trials: n_trials below stays the trial count and the
            # betas are sliced back to it.
            n_trial_cols_st = int(st_design.shape[1])
            st_design, _ = append_stim_vecs_to_single_trial_design(
                stim_vec_blocks,
                st_design,
                hrf_library=hrf_library[hrf_idx].reshape(1, -1),
                n_timepoints=n_timepoints,
                tr=args.tr,
                microtime_dt=args.microtime_dt,
                run_starts=run_starts,
                device=device,
                verbose=False,
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
                n_trials = n_trial_cols_st
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
                    canonical_betas[c0:c1] = chunk_betas[:, :n_trials].cpu()

            if args.verb >= 1 and hrf_idx % 5 == 0:
                col_r2 = fit_r2_all[:, hrf_idx]
                print(
                    f"    HRF {hrf_idx}: mean R²={col_r2.mean():.4f}, median R²={col_r2.median():.4f}"
                )

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
            vox_mask = hrf_index_cpu == hrf_idx
            if not vox_mask.any():
                continue
            vox_indices = vox_mask.nonzero(as_tuple=True)[0]
            proj_design_dev = projected_designs_cache[hrf_idx].to(device)

            for c0 in range(0, len(vox_indices), chunk_size):
                c1 = min(c0 + chunk_size, len(vox_indices))
                idx = vox_indices[c0:c1]
                chunk_data = projected_data[idx].to(device)
                chunk_betas = torch.linalg.lstsq(proj_design_dev, chunk_data.T).solution.T
                best_betas[idx] = chunk_betas[:, :n_trials].cpu()

        del projected_designs_cache  # Free ~240 MB

        # Create HRFSelectionResults object
        from fastfuncstuff.design.hrf_selection import HRFSelectionResults

        xval_r2_std = torch.zeros_like(xval_r2_best)  # Not meaningful for in-sample R²

        # Compute HRF usage counts for reporting
        hrf_usage_counts = torch.bincount(hrf_index_cpu, minlength=n_hrfs).tolist()

        # Canonical baseline: first HRF in library (index 0)
        xval_r2_canonical = fit_r2_all[:, 0].to(device)

        # ---- Clearly-named R² maps ----
        # In-sample R² (already computed above)
        canonical_full_r2 = fit_r2_all[:, 0]  # canonical HRF, in-sample
        hrfopt_full_r2 = xval_r2_best  # best-HRF per voxel, in-sample

        # Beta-series CV R² (genuine cross-validated)
        from fastfuncstuff.glm.xval import compute_xval_r2_single_trials, generate_cv_splits

        # Honour -cv_strategy here too: this block used to hardcode LORO, so
        # the flag silently did nothing for the beta-series CV while shaping
        # the HRF selection above it.
        cv_splits = generate_cv_splits(n_runs, strategy=cv_strategy)

        print()
        print("Computing beta-series CV R² (canonical HRF)...")
        canonical_cv = compute_xval_r2_single_trials(
            canonical_betas,
            cond_ids,
            run_ids,
            cv_splits,
            metric="cod",
            device=device,
        )
        canonical_xval_r2 = canonical_cv["r2"]
        assert isinstance(canonical_xval_r2, torch.Tensor)  # "r2" key is always a tensor

        print("Computing beta-series CV R² (optimal HRF per voxel)...")
        hrfopt_cv = compute_xval_r2_single_trials(
            best_betas,
            cond_ids,
            run_ids,
            cv_splits,
            metric="cod",
            device=device,
        )
        hrfopt_xval_r2 = hrfopt_cv["r2"]
        assert isinstance(hrfopt_xval_r2, torch.Tensor)  # "r2" key is always a tensor

        # Ceiling on the CV R² just computed. Built from the per-voxel best-HRF
        # betas, matching hrfopt_xval_r2 rather than the canonical baseline --
        # the selected model is the one the explainable fraction is about.
        # compute_xval_r2_single_trials does not z-score, so neither does this.
        beta_ceiling = None
        if args.noise_ceiling in ("auto", "ncsnr"):
            from fastfuncstuff.stats.noise_ceiling import beta_space_ceiling

            beta_ceiling = beta_space_ceiling(
                betas=best_betas.cpu(),
                condition_ids=cond_ids.cpu(),
                run_ids=run_ids.cpu(),
                cv_splits=cv_splits,
                xval_r2=hrfopt_xval_r2.cpu(),
                zscore_by_run=False,
                metric="cod",
            )
            print()
            print(
                f"  Noise ceiling (beta space, m={beta_ceiling.n_train_repeats:.1f} "
                "training trials/condition):"
            )
            print(f"    {beta_ceiling.result.summarize(beta_ceiling.explainable)}")
            for note in beta_ceiling.result.notes:
                print(f"    NOTE: {note}")

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
                verbose=args.verb >= 1,
                stim_vec_blocks=stim_vec_blocks,
                event_onsets=all_onsets,
                stim_durations=durations,
                run_starts=run_starts,
            )

            # Update results
            results.canonical_results = canonical_results
            print("  Canonical fit complete.")

    else:
        # ========== EXISTING Beta @ Design TIMESERIES CV PATH ==========
        results = fit_glm_hrf_library_with_xval(
            data=cv_data,
            final_fit_data=data,
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
            ortvec_files=None,
            nuisance_blocks=nuisance_blocks,
            canonical_mode=args.canonical,
            device=device,
            verbose=args.verb >= 1,
            chunk_size=args.batch_size,
            r2_method=args.R2method,
            select_mode=args.select,
            debug=args.debug,
            debug_prefix=args.prefix,
            condition_labels=condition_labels,
            stim_vec_blocks=stim_vec_blocks,
            event_onsets=all_onsets,
        )

        # ========== -delta_denoise: second pass without ortvec ==========
        # Quantify the effect of the user-supplied nuisance regressors on
        # HRF selection by re-running the entire pass with ortvec stripped.
        # Polort, scaling, library, CV strategy, and select_mode are all
        # held fixed — only ortvec_files differs.
        if args.delta_denoise:
            print()
            print("=" * 70)
            print("Running second pass without -ortvec (for -delta_denoise)")
            print("=" * 70)
            results_nodenoise = fit_glm_hrf_library_with_xval(
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
                ortvec_files=None,
                nuisance_blocks=[],  # the whole point — strip user nuisance
                # Stim vectors are NOT nuisance, so they stay in both passes;
                # -delta_denoise isolates the effect of -ortvec alone.
                stim_vec_blocks=stim_vec_blocks,
                canonical_mode=args.canonical,
                device=device,
                verbose=args.verb >= 1,
                chunk_size=args.batch_size,
                r2_method=args.R2method,
                select_mode=args.select,
                debug=False,  # don't double-write debug artifacts
                debug_prefix=args.prefix,
                condition_labels=condition_labels,
                event_onsets=all_onsets,
            )

    if cv_data is not data:
        del cv_data  # selection is done; the refit below reads unblurred `data`
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ==========================================================================
    # 5b. Optional: refit with the optimal HRF per voxel for single-trial betas
    # ==========================================================================
    # Independent of how the HRF was selected: the HRF index is a property of the
    # voxel, so it transfers to the single-trial design whether it was chosen in
    # beta space (-cv_design single) or on the condition-level design.
    if args.single_trials:
        print()
        print("Refitting with optimal HRF per voxel (single-trial betas)...")

        # Convert hrf_library from tensor (n_hrfs, n_timepoints) to list of 1D tensors
        hrf_library_list = [hrf_library[i] for i in range(hrf_library.shape[0])]

        # Refit with optimal HRF per voxel
        final_results = _fit_voxelwise_hrf_single_trial(
            data=data,
            onsets_by_condition=all_onsets,
            hrf_library=hrf_library_list,
            hrf_index=results.hrf_index,
            nuisance_design=nuisance_design,
            durations=durations,
            run_starts=run_starts,
            tr=tr,
            n_timepoints=n_timepoints,
            microtime_dt=args.microtime_dt,
            condition_labels=condition_labels,
            device=device,
            verbose=args.verb >= 1,
            stim_vec_blocks=stim_vec_blocks,
        )

        results.final_results = final_results
        print("  Single-trial refit complete.")

    # Update metadata with CLI parameters
    results.hrf_metadata["hrf_mode"] = args.hrf_mode
    results.hrf_metadata["canonical_mode"] = args.canonical
    results.hrf_metadata["condition_labels"] = condition_labels
    results.hrf_metadata["input_files"] = input_files
    results.hrf_metadata["onset_files"] = onset_files  # None for BIDS path
    if args.events:
        results.hrf_metadata["event_files"] = args.events
    results.hrf_metadata["durations"] = durations
    results.hrf_metadata["cv_design"] = cv_design
    results.hrf_metadata["cv_blur_fwhm"] = args.cv_blur
    results.hrf_metadata["do_blur_fwhm"] = args.do_blur
    results.hrf_metadata["cv_design_requested"] = args.cv_design
    results.hrf_metadata["single_trials"] = bool(args.single_trials)
    results.hrf_metadata["trial_repeats"] = {
        "n_trials": trial_repeats.n_trials,
        "n_conditions": trial_repeats.n_conditions,
        "n_repeated_conditions": trial_repeats.n_repeated_conditions,
        "predictable_fraction": trial_repeats.predictable_fraction,
        "trials_per_condition": trial_repeats.trials_per_condition,
        "runs_per_condition": trial_repeats.runs_per_condition,
    }
    if ortvec_files:
        results.hrf_metadata["ortvec_files"] = [(str(f), label) for f, label in ortvec_files]
    if nuisance_blocks:
        # Block-level provenance for the JSON sidecar: label, source files per
        # run (None where the block contributed zeros), and column count.
        results.hrf_metadata["nuisance_blocks"] = [
            {
                "label": b.label,
                "n_columns": b.n_columns,
                "source": list(b.source),
                "column_names": b.get_column_names(),
            }
            for b in nuisance_blocks
        ]

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

    if args.single_trials and results.final_results is not None:
        results.final_results = (
            None  # Temporarily remove to prevent save_hrf_selection_results from saving it
        )

    output_files = save_hrf_selection_results(
        results=results,
        output_prefix=str(args.prefix),
        volume_shape=volume_shape,
        affine=affine,
        voxel_mask=voxel_mask,
        condition_labels=task_column_labels,
        run_starts=run_starts,
        save_all_hrf_designs=args.save_hrf_designs,
        onsets=onset_matrix if args.save_hrf_designs else None,
        save_plots=args.save_plots,
        nii_ext=_nii_ext,
    )

    if beta_ceiling is not None and beta_ceiling.result.n_usable:
        # Restacked onto _hrfopt_xval_r2, the beta-space CV map the ceiling was
        # built from -- NOT _xval_r2, which is the in-sample best-HRF R2. Those
        # two live on different scales and their ratio would mean nothing.
        output_files["hrfopt_xval_r2"] = save_r2_ceiling_stack(
            [
                (hrfopt_xval_r2, "hrfopt_xval_R2"),
                (beta_ceiling.result.ceiling, "noise_ceiling"),
                (beta_ceiling.explainable, "explainable_R2"),
                (beta_ceiling.ncsnr_map, "ncsnr"),
            ],
            f"{args.prefix}_hrfopt_xval_r2{_nii_ext}",
            volume_shape,
            affine,
            voxel_mask.numpy() if voxel_mask is not None else None,
        )
        print(f"  {output_files['hrfopt_xval_r2']}")
    if beta_ceiling is not None and beta_ceiling.explainable_withheld_because:
        print(f"  (no explainable_R2: {beta_ceiling.explainable_withheld_because})")

    # Restore final_results for custom saving
    if args.single_trials and final_results_temp is not None:
        results.final_results = final_results_temp

    # ==========================================================================
    # 6a. -delta_denoise: save the nodenoise pass + delta maps
    # ==========================================================================
    if args.delta_denoise and results_nodenoise is not None:
        from fastfuncstuff.design.hrf_selection import _save_volume

        # Mirror the same metadata-stamping the primary pass got, so the
        # nodenoise pt-bundle is self-describing.
        results_nodenoise.hrf_metadata["hrf_mode"] = args.hrf_mode
        results_nodenoise.hrf_metadata["canonical_mode"] = args.canonical
        results_nodenoise.hrf_metadata["condition_labels"] = condition_labels
        results_nodenoise.hrf_metadata["input_files"] = input_files
        results_nodenoise.hrf_metadata["onset_files"] = onset_files
        if args.events:
            results_nodenoise.hrf_metadata["event_files"] = args.events
        results_nodenoise.hrf_metadata["durations"] = durations
        results_nodenoise.hrf_metadata["delta_denoise_pass"] = "nodenoise"

        print()
        print(f"Saving nodenoise outputs ({args.prefix}_nodenoise_*) ...")
        save_hrf_selection_results(
            results=results_nodenoise,
            output_prefix=f"{args.prefix}_nodenoise",
            volume_shape=volume_shape,
            affine=affine,
            voxel_mask=voxel_mask,
            condition_labels=task_column_labels,
            run_starts=run_starts,
            save_all_hrf_designs=False,  # one set of designs is enough
            onsets=None,
            save_plots=False,
            nii_ext=_nii_ext,
        )

        # ---- Delta maps (primary − nodenoise) ----
        # xval R² delta: positive where nuisance helped (CV-honest metric).
        # Compute on CPU so devices match regardless of where each pass lived.
        r2_primary = results.xval_r2_best.detach().cpu()
        r2_nd = results_nodenoise.xval_r2_best.detach().cpu()
        delta_xval = r2_primary - r2_nd
        _save_volume(
            delta_xval,
            f"{args.prefix}_delta_xval_r2{_nii_ext}",
            volume_shape,
            affine,
            voxel_mask,
        )

        # In-sample (HRFopt) R² delta — the headline "fit looks better" number.
        # This field is only present for the time-series path; guard anyway.
        delta_hrfopt = None
        _hrfopt_full_r2 = getattr(results, "hrfopt_full_r2", None)
        _hrfopt_full_r2_nd = getattr(results_nodenoise, "hrfopt_full_r2", None)
        if _hrfopt_full_r2 is not None and _hrfopt_full_r2_nd is not None:
            r2f_primary = _hrfopt_full_r2.detach().cpu()
            r2f_nd = _hrfopt_full_r2_nd.detach().cpu()
            delta_hrfopt = r2f_primary - r2f_nd
            _save_volume(
                delta_hrfopt,
                f"{args.prefix}_delta_hrfopt_r2{_nii_ext}",
                volume_shape,
                affine,
                voxel_mask,
            )

        # HRF index: whether selection changed (uint-style mask) and signed shift.
        idx_primary = results.hrf_index.detach().cpu().long()
        idx_nd = results_nodenoise.hrf_index.detach().cpu().long()
        changed = (idx_primary != idx_nd).to(torch.float32)
        shift = (idx_primary - idx_nd).to(torch.float32)
        _save_volume(
            changed,
            f"{args.prefix}_delta_hrf_changed{_nii_ext}",
            volume_shape,
            affine,
            voxel_mask,
        )
        _save_volume(
            shift,
            f"{args.prefix}_delta_hrf_index{_nii_ext}",
            volume_shape,
            affine,
            voxel_mask,
        )

        # Summary block — keep it the one piece the user actually reads.
        n_voxels = idx_primary.numel()
        n_changed = int(changed.sum().item())
        pct_changed = 100.0 * n_changed / max(n_voxels, 1)
        med_dxval = float(delta_xval.median().item())
        med_dxval_changed = (
            float(delta_xval[changed.bool()].median().item()) if n_changed > 0 else float("nan")
        )
        med_abs_shift_changed = (
            float(shift[changed.bool()].abs().median().item()) if n_changed > 0 else float("nan")
        )

        print()
        print("=" * 70)
        print("Δ summary: primary (with -ortvec) − nodenoise")
        print("=" * 70)
        print(f"  Voxels with changed HRF: {n_changed:,} / {n_voxels:,} ({pct_changed:.2f}%)")
        print(f"  Median Δ xval R² (all voxels):           {med_dxval:+.4f}")
        if n_changed > 0:
            print(f"  Median Δ xval R² (HRF-changed voxels):   {med_dxval_changed:+.4f}")
            print(f"  Median |Δ HRF index| (changed voxels):   {med_abs_shift_changed:.1f}")
        if delta_hrfopt is not None:
            print(
                f"  Median Δ in-sample R²:                   {float(delta_hrfopt.median().item()):+.4f}"
            )
        print()

    # ==========================================================================
    # 6b. Custom saving for single-trial betas (if requested)
    # ==========================================================================
    # Note: The canonical betas are already saved correctly by save_hrf_selection_results()
    # to {prefix}_canonical_stats.nii.gz. For single-trial betas, we need custom saving
    # to {prefix}_stats_single_trial.nii.gz instead of the default {prefix}_stats.nii.gz
    final_results_for_save = results.final_results
    if args.single_trials and final_results_for_save is not None:
        from fastfuncstuff.glm.outputs import write_glm_bucket_as_nifti

        print("  Saving single-trial betas with custom filename...")

        # Set required metadata for saving (same as canonical_results)
        final_results_for_save.original_shape = volume_shape
        final_results_for_save.affine = affine
        if voxel_mask is not None:
            final_results_for_save.voxel_mask = voxel_mask

        # Get trial labels from results (stored during refit)
        trial_labels = final_results_for_save.trial_labels
        if trial_labels is None:
            print("  WARNING: No trial labels found, using generic names")
            assert final_results_for_save.betas is not None
            n_trials = final_results_for_save.betas.shape[1]
            trial_labels = [f"trial_{i:04d}" for i in range(n_trials)]

        # Save with custom filename using trial labels
        single_trial_file = f"{args.prefix}_stats_single_trial{_nii_ext}"
        with spinner(f"Writing {Path(single_trial_file).name}"):
            write_glm_bucket_as_nifti(
                final_results_for_save,
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
        violation_path = f"{args.prefix}_scale_violations{_nii_ext}"
        with spinner(f"Writing {Path(violation_path).name}"):
            save_nifti(violation_vol, output_path=violation_path, affine=affine)
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
    from fastfuncstuff.cli_utils import print_cli_footer, print_cli_section

    print_cli_section("Output files")
    for output_type, filepath in output_files.items():
        print(f"  {output_type}: {filepath}")

    # Print final summary
    print_cli_section("Summary")
    print(f"  Mean xval R²: {results.xval_r2_best.mean().item():.4f}")
    if results.xval_r2_canonical is not None:
        canonical_r2 = results.xval_r2_canonical.mean().item()
        best_r2 = results.xval_r2_best.mean().item()
        print(f"  Canonical HRF baseline R²: {canonical_r2:.4f}")
        print(f"  Improvement over canonical: {best_r2 - canonical_r2:+.4f}")
    _final_results_summary = results.final_results
    if _final_results_summary is not None:
        assert _final_results_summary.r2 is not None
        print(f"  Final R² (full data): {_final_results_summary.r2.mean().item():.4f}")
    print()
    print("  HRF usage distribution:")
    hrf_counts = results.hrf_metadata["hrf_usage_counts"]
    for i, count in enumerate(hrf_counts):
        if count > 0:
            pct = 100 * count / n_voxels
            print(f"    HRF {i}: {count:,} voxels ({pct:.1f}%)")
    print_cli_footer("ffs_hrfopt")


if __name__ == "__main__":
    main()
