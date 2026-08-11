#!/usr/bin/env python3
"""
3dRidgefast - GPU-accelerated ridge regression with single-trial estimation

This tool implements GLMsingle-style ridge regression for fMRI analysis:
- Single-trial beta estimation (one beta per event)
- Cross-validated ridge fraction selection per voxel
- Integration with HRFoptfast (per-voxel HRF shapes)
- Integration with Denoisefast (noise PC regression)
- GPU acceleration with non-TR-locked onsets

Basic usage:
    3dRidgefast -input run1.nii.gz run2.nii.gz run3.nii.gz \\
                -onsets cond1.txt cond2.txt \\
                -durations 2.0 5.0 \\
                -tr 2.0 \\
                -prefix subject01

With HRF optimization:
    3dRidgefast -input run*.nii.gz \\
                -onsets cond1.txt cond2.txt \\
                -durations 2.0 5.0 \\
                -tr 2.0 \\
                -hrf_opt output/hrfopt_prefix \\
                -prefix subject01_ridge

With denoising:
    3dRidgefast -input run*.nii.gz \\
                -onsets cond1.txt cond2.txt \\
                -durations 2.0 5.0 \\
                -tr 2.0 \\
                -denoise output/denoise_prefix \\
                -prefix subject01_ridge

For help:
    3dRidgefast -help
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Import fastfuncstuff modules
try:
    from fastfuncstuff.cli_utils import (
        LoadResult,
        add_load_threads_arg,
        add_noise_ceiling_args,
        add_verbose_arg,
        auto_polort,
        build_nuisance_per_run,
        compute_run_lengths,
        get_average_run_duration,
        load_and_preprocess_runs,
        parse_prefix,
        preflight_check,
        save_4d_nifti,
        save_volume_nifti,
        spinner,
    )
    from fastfuncstuff.design.hrf import get_hrf_library
    from fastfuncstuff.glm.ridge import (
        create_single_trial_design,
        fit_ridge_single_trial,
        load_hrf_indices,
        load_noise_pcs,
    )
    from fastfuncstuff.utils import configure_torch_backends, get_device, scale_to_percent_signal
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    print("Make sure fastfuncstuff is installed: pip install -e .")
    sys.exit(1)


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults while preserving raw description formatting."""


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="3dRidgefast - GPU-accelerated ridge regression with single-trial estimation",
        formatter_class=_HelpFormatter,
        epilog="""
Examples:
  # Basic single-trial ridge regression
  3dRidgefast -input run1.nii.gz run2.nii.gz run3.nii.gz \\
              -onsets faces.txt houses.txt \\
              -durations 2.0 2.0 \\
              -tr 2.0 \\
              -prefix output/subject01

  # With HRF optimization (per-voxel HRFs)
  3dRidgefast -input run*.nii.gz \\
              -onsets stim1.txt stim2.txt \\
              -durations 1.5 3.0 \\
              -tr 2.5 \\
              -hrf_opt output/hrfopt \\
              -prefix output/ridge

  # With denoising (noise PC regression)
  3dRidgefast -input run*.nii.gz \\
              -onsets stim.txt \\
              -durations 2.0 \\
              -tr 2.0 \\
              -denoise output/denoise \\
              -prefix output/ridge

  # Full pipeline (HRF + denoising)
  3dRidgefast -input run*.nii.gz \\
              -onsets stim.txt \\
              -durations 2.0 \\
              -tr 2.0 \\
              -hrf_opt output/hrfopt \\
              -denoise output/denoise \\
              -ridge_fracs 0.05 1.0 0.05 \\
              -prefix output/ridge_full

  # Many conditions with grouped durations (20 conditions @ 3s, 40 @ 5s)
  3dRidgefast -input run*.nii.gz \\
              -onsets cond*.txt \\
              -durations 3,20 5,40 \\
              -tr 2.0 \\
              -prefix output/ridge_many

Notes:
  - Single-trial estimation: Each event gets its own beta estimate
  - Ridge fractions: Fraction of OLS coefficient norm to retain (1=OLS/no regularization, 0=maximum regularization)
  - CV strategy: Leave-one-run-out by default for fraction selection
  - Output: Single-trial beta maps, optimal ridge fraction map, R² maps
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
        help="Stimulus durations in seconds. Single value, one per condition, or use 'value,count' "
        "format (e.g., '3,20 5,40' for 20 conditions with 3s, then 40 with 5s). "
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
        help="trial_type values to exclude. Only valid with -events.",
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
        help="Round stimulus durations to PLACES decimal places.",
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

    # Ridge regression options
    ridge_opts = parser.add_argument_group("Ridge Regression Options")
    ridge_opts.add_argument(
        "-ridge_fracs",
        type=float,
        nargs=3,
        default=[0.05, 1.0, 0.05],
        metavar=("START", "END", "STEP"),
        help="Ridge fractions to test: START END STEP (default: 0.05 1.0 0.05). "
        "Fraction = proportion of OLS coefficient norm to retain (1=no regularization, 0=maximum).",
    )
    ridge_opts.add_argument(
        "-cv_strategy",
        type=str,
        default="loro",
        choices=["loro", "split_half"],
        help="Cross-validation strategy for fraction selection (default: loro)",
    )
    ridge_opts.add_argument(
        "-autoscale",
        action="store_true",
        default=True,
        help="Apply GLMsingle-style post-hoc scaling to undo ridge shrinkage bias (default: True). "
        "Recommended to keep enabled.",
    )
    ridge_opts.add_argument(
        "-no_autoscale",
        action="store_false",
        dest="autoscale",
        help="Disable autoscaling (keep ridge-regularized betas as-is).",
    )
    ridge_opts.add_argument(
        "-single_trials",
        action="store_true",
        help="Use beta-space cross-validation (GLMsingle-style). "
        "Fits single-trial model once on all data, evaluates R² on "
        "condition-averaged vs individual trial betas across folds. "
        "Replaces timeseries CV with beta-space CV.",
    )
    ridge_opts.add_argument(
        "-metric",
        type=str,
        default="sse",
        choices=["sse", "cod", "corr", "corr2"],
        help="CV metric for fraction selection (default: sse). "
        "'sse' = sum of squared errors (GLMsingle-compatible, lower=better). "
        "'cod' = coefficient of determination (higher=better).",
    )
    ridge_opts.add_argument(
        "-zscore_by_run",
        action="store_true",
        default=True,
        help="Z-score betas per run before CV using OLS normalization stats "
        "(GLMsingle default). Only applies with -single_trials.",
    )
    add_noise_ceiling_args(
        ridge_opts,
        stage_note="ffs_ridge scores in beta space, so 'auto' resolves to "
        "'ncsnr' -- which needs conditions that repeat across runs, not "
        "repeated runs.",
    )

    # Integration options
    integ_opts = parser.add_argument_group("Integration Options")
    integ_opts.add_argument(
        "-hrf_opt",
        type=str,
        default=None,
        help="HRFoptfast output prefix. Loads {prefix}_hrf_index.nii.gz for per-voxel HRFs.",
    )
    integ_opts.add_argument(
        "-denoise",
        type=str,
        default=None,
        help="Denoisefast output prefix. Loads {prefix}_noise_pcs.xmat.1D as nuisance regressors.",
    )
    integ_opts.add_argument(
        "-hrf-library",
        dest="hrf_library",
        type=str,
        default=None,
        metavar="TSV",
        help=(
            "Custom HRF library TSV (e.g. from ffs_librarian).  Used "
            "when per-voxel HRF indices are loaded via -hrf_opt."
        ),
    )

    # Processing options
    proc_opts = parser.add_argument_group("Processing Options")
    proc_opts.add_argument(
        "-mask",
        type=str,
        default=None,
        help="Brain mask (AFNI or NIfTI format). If not provided, all voxels are used.",
    )
    proc_opts.add_argument(
        "-polort",
        type=int,
        default=None,
        help="Polynomial order for drift modeling (default: auto based on run length)",
    )
    proc_opts.add_argument(
        "-microtime_dt",
        type=float,
        default=0.1,
        help="Microtime resolution for non-TR-locked onsets (seconds, default: 0.1)",
    )
    proc_opts.add_argument(
        "-hrf_model",
        type=str,
        default="spmg1",
        help="HRF model: 'spmg1' (default), 'spmg2', 'spmg3', 'glmsingle'. "
        "SPMG2 = canonical + temporal derivative (2 basis per trial). "
        "SPMG3 = canonical + time + dispersion derivatives (3 basis per trial). "
        "NOTE: FIR/TENT not supported in single-trial mode - use 3dDenoisefast instead.",
    )
    proc_opts.add_argument(
        "-device",
        type=str,
        help="Force device: 'cpu' or 'cuda' (default: auto-detect GPU)",
    )
    proc_opts.add_argument(
        "-chunk_size",
        type=int,
        default=0,
        help="Voxels per chunk for processing. 0 = auto-detect based on available memory (default)",
    )
    add_load_threads_arg(proc_opts)
    add_verbose_arg(proc_opts, default=0)
    proc_opts.add_argument(
        "-dry_run",
        action="store_true",
        help="Fast testing mode: load only first run, generate synthetic data for rest. "
        "Results are nonsensical but pipeline runs quickly for testing.",
    )
    proc_opts.add_argument(
        "-do_scale",
        action="store_true",
        help="Scale each voxel per run to mean=100 (percent signal change units). "
        "Values are clipped to max 200 (100%% increase from mean). "
        "Violation locations are saved to {prefix}_scale_violations.nii.gz",
    )

    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    pfx = parse_prefix(args.prefix)
    args.prefix = pfx.stem  # overwrite with clean stem
    _nii_ext = pfx.nifti_ext

    print("=" * 70)
    print("3dRidgefast - Ridge regression with single-trial estimation")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ========================================================================
    # 1. Parse onset metadata (fail-fast, before slow data loading)
    # ========================================================================
    input_files = args.input
    n_runs = len(input_files)

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

    from fastfuncstuff.cli_utils import parse_timing_spec

    print("Parsing onset metadata...")
    try:
        timing = parse_timing_spec(
            events=args.events,
            onsets=args.onsets,
            durations_arg=args.durations,
            n_runs=n_runs,
            event_ignore=args.event_ignore,
            event_cols=tuple(args.event_cols) if args.event_cols else None,
            round_durations=args.round_durations,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    all_onsets = timing.all_onsets
    durations = timing.durations
    condition_labels = timing.condition_labels
    n_conditions = timing.n_conditions
    onset_files = timing.onset_files

    # Pre-flight checks (before slow data loading)
    preflight_check(
        input_files=input_files,
        onset_files=onset_files,
        hrf_opt_prefix=args.hrf_opt or None,
        denoise_prefix=args.denoise or None,
    )

    # ========================================================================
    # 2. Load data
    # ========================================================================
    print()
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    configure_torch_backends(device)

    print(f"Device: {device}")
    print()

    # Load and preprocess data using unified utility
    # Note: do_scale=False here - scaling will be applied separately if requested
    # to maintain the existing behavior of showing scaling info separately
    load_result: LoadResult = load_and_preprocess_runs(
        input_files=input_files,
        tr=args.tr,
        mask_file=args.mask,
        blur_fwhm=None,  # No blur in 3dRidgefast
        do_scale=False,
        device=device,
        force_cpu=False,  # 3dRidgefast always loads to GPU
        dry_run=args.dry_run,
        verbose=True,
        load_threads=args.load_threads,
    )

    # Modify prefix for dry run mode
    if args.dry_run:
        args.prefix = f"dry_run_{args.prefix}"

    # Extract loaded data
    data = load_result.data
    run_starts = load_result.run_starts
    mask = load_result.mask
    mask_flat = load_result.mask_flat
    affine = load_result.affine
    volume_shape = load_result.volume_shape
    n_voxels = load_result.n_voxels
    n_timepoints = load_result.n_timepoints
    n_runs = load_result.n_runs

    # Update args.tr from header (consistent with hrfopt/denoise)
    if args.tr is None:
        args.tr = load_result.tr
        print(f"  TR from header: {args.tr}s")
    else:
        print(f"  TR (specified): {args.tr}s")

    # ========================================================================
    # 3. Apply onset rounding (TR now known)
    # ========================================================================
    print()
    if args.round_onsets is not None:
        from fastfuncstuff.design.builder import round_onsets as _round_onsets

        all_onsets = _round_onsets(all_onsets, args.tr, threshold=args.round_onsets)

    # ========================================================================
    # 4. HRF model args and validation
    # ========================================================================
    print()
    from fastfuncstuff.cli_utils import parse_hrf_model_args

    # Note: 3dRidgefast doesn't support FIR (use condition-level tools for that)
    # But it DOES support SPMG2/SPMG3 for single-trial with derivatives
    hrf_info = parse_hrf_model_args(
        hrf_model_arg=args.hrf_model if hasattr(args, "hrf_model") else "spmg1",
        canonical_arg=None,  # 3dRidgefast doesn't have -canonical
        durations=durations,
        condition_labels=condition_labels,
        tr=args.tr,
    )

    hrf_model_name = hrf_info["hrf_model_name"]
    is_fir_model = hrf_info["is_fir_model"]
    n_basis = hrf_info["n_basis"]

    # Validate: FIR is incompatible with single-trial
    if is_fir_model:
        print("ERROR: FIR/TENT models are incompatible with single-trial estimation")
        print("  Use 3dDenoisefast or ffs_reml for FIR/TENT analysis")
        sys.exit(1)

    # SPMG2/SPMG3 with HRF library is incompatible (checked in create_single_trial_design)

    print()
    print("=" * 70)
    print(f"Validated: {len(all_onsets)} conditions, {len(durations)} durations")
    print("=" * 70)

    # Apply scaling if requested (maintaining existing behavior)
    scale_info = None
    violations_mask = None
    if args.do_scale:
        print()
        data, violations_mask, scale_info = scale_to_percent_signal(
            data,
            run_starts=run_starts,
            max_scale=200.0,
            verbose=True,
        )
        print(f"  Scaled {n_voxels:,} voxels to percent signal change")
        if violations_mask is not None:
            n_violations = violations_mask.any(dim=1).sum().item()
            print(f"  Clipping violations: {n_violations:,} voxels")

    print()

    # Load HRF indices if provided
    hrf_library = None
    hrf_indices = None
    if args.hrf_opt:
        print()
        print(f"Loading HRF optimization results from {args.hrf_opt}...")
        hrf_index_file = f"{args.hrf_opt}_hrf_index.nii.gz"
        hrf_indices = load_hrf_indices(hrf_index_file, mask=mask)
        print(f"  Loaded HRF indices: {hrf_indices.shape}")

        # Load HRF library (reconstruct from metadata or use default)
        # For now, use default library
        hrf_library = get_hrf_library(
            mode="library",
            stim_duration=0.0,
            microtime_dt=args.microtime_dt,
            n_hrfs=20,
            library_path=args.hrf_library,
        )
        if args.hrf_library:
            print(f"  Loaded custom HRF library from {args.hrf_library}")
        print(f"  Using HRF library with {len(hrf_library)} HRFs")

    # Load noise PCs if provided
    noise_pcs = None
    if args.denoise:
        print()
        print(f"Loading denoising results from {args.denoise}...")

        # Try to load per-run files first (e.g., prefix_run01_selected_PCs.txt)
        # If that fails, try the Denoisefast format (prefix_noise_pcs.xmat.1D)
        try:
            from fastfuncstuff.utils import load_per_run_nuisance_files

            noise_pcs_np = load_per_run_nuisance_files(
                args.denoise, n_runs=n_runs, suffix="_selected_PCs.txt", verbose=True
            )

            # Convert numpy arrays to torch tensors
            noise_pcs = []
            for pcs_np in noise_pcs_np:
                if pcs_np is not None:
                    noise_pcs.append(torch.from_numpy(pcs_np).float())
                else:
                    noise_pcs.append(torch.zeros((0, 0), dtype=torch.float32))

            print(f"  Loaded per-run noise PCs from {args.denoise}")

        except (RuntimeError, FileNotFoundError):
            # Fall back to Denoisefast format (single file)
            print("  Per-run files not found, trying Denoisefast format...")
            noise_pc_file = f"{args.denoise}_noise_pcs.xmat.1D"
            noise_pcs = load_noise_pcs(noise_pc_file, run_starts, n_timepoints)
            print(f"  Loaded {noise_pcs[0].shape[1]} noise PCs per run")

    # Create single-trial design matrix
    print()
    print("Creating single-trial design matrix...")
    design_matrix, trial_labels, trial_condition_ids, trial_run_ids, condition_design = (
        create_single_trial_design(
            onsets_by_condition=all_onsets,
            durations=durations,
            run_starts=run_starts,
            tr=args.tr,
            n_timepoints=n_timepoints,
            hrf_library=hrf_library,
            hrf_index_per_voxel=hrf_indices,
            microtime_dt=args.microtime_dt,
            condition_labels=condition_labels,
            device=device,
            hrf_model_name=hrf_model_name,
            n_basis=n_basis,
        )
    )

    n_columns = len(trial_labels)
    n_condition_cols = condition_design.shape[1]
    # For SPMG1: n_columns = n_trials, n_condition_cols = n_conditions
    # For SPMG2: n_columns = n_trials * 2, n_condition_cols = n_conditions * 2
    # For SPMG3: n_columns = n_trials * 3, n_condition_cols = n_conditions * 3
    n_trials_actual = n_columns // n_basis
    n_conditions = n_condition_cols // n_basis

    print(f"  Total trials: {n_trials_actual}")
    print(f"  Total columns: {n_columns} ({n_trials_actual} trials × {n_basis} basis)")
    print(
        f"  Condition design: {n_condition_cols} columns ({n_conditions} conditions × {n_basis} basis)"
    )
    print(f"  Conditions: {n_conditions}")
    print(f"  Design shape: {design_matrix.shape}")

    # Parse ridge fractions
    frac_start, frac_end, frac_step = args.ridge_fracs
    fracs = np.arange(frac_start, frac_end + frac_step / 2, frac_step, dtype=np.float32)

    print()
    print("=" * 70)
    print("Ridge regression estimation")
    print("=" * 70)
    print()

    # When per-voxel HRFs are in play (-hrf_opt provided indices),
    # the timeseries-CV path can't process the resulting per-HRF
    # design tensor (shape (n_HRFs, n_t, n_trials)) and the only
    # sane storage avenue is the beta-space CV path (which groups
    # voxels by HRF index, processes each group with its own design
    # matrix, no per-voxel design materialisation).  Auto-enable it
    # rather than crashing on a non-obvious ValueError from
    # fit_ridge_single_trial.
    if hrf_indices is not None and not args.single_trials:
        print(
            "  NOTE: per-voxel HRF library detected (-hrf_opt provided "
            "hrf_indices); auto-enabling beta-space CV (-single_trials)."
        )
        args.single_trials = True

    if args.single_trials:
        # Ridge has no -cv_design escape hatch: the fraction regularizes
        # collinearity between adjacent trial regressors, which only exists in the
        # single-trial design, so it cannot be learned on a condition-level one.
        # All we can do is say out loud when the beta-space score is undefined.
        from fastfuncstuff.cli_utils import summarize_trial_repeats

        _repeats = summarize_trial_repeats(all_onsets)
        print(f"  Event repeat structure: {_repeats.describe()}")
        if _repeats.n_predictable_trials == 0:
            print(
                "  WARNING: no condition appears in more than one run, so every "
                "held-out trial's beta-space score is undefined and the selected "
                "fraction is arbitrary. Pass a single value to -ridge_fracs instead."
            )
        elif _repeats.predictable_fraction < 0.5:
            print(
                f"  WARNING: only {100 * _repeats.predictable_fraction:.0f}% of trials "
                "are predictable across runs; fraction selection rests on a thin subset."
            )

        # ========== SINGLE-TRIAL BETA-SPACE CV PATH ==========
        from fastfuncstuff.glm.outputs import save_single_trial_results
        from fastfuncstuff.glm.ridge import _fit_ridge_multiple_fracs
        from fastfuncstuff.glm.xval import (
            compute_r2_metric,
            generate_cv_splits,
            metric_higher_is_better,
            project_out_nuisance_per_run,
            single_trial_cv_helper,
        )

        print("Using beta-space cross-validation (GLMsingle-style)")
        print()

        # Build nuisance design (polynomials + noise PCs)
        # Auto-determine polort if not specified
        if args.polort is None:
            run_lengths = compute_run_lengths(run_starts, n_timepoints)
            avg_run_duration_sec = get_average_run_duration(run_lengths, args.tr)
            polort = auto_polort(avg_run_duration_sec, formula="afni")
            print(f"Auto polort: {polort} (based on {avg_run_duration_sec / 60:.1f} min avg run)")
        else:
            polort = args.polort

        # Build nuisance per run using shared utility
        nuisance_per_run = build_nuisance_per_run(
            run_starts=run_starts,
            n_timepoints=n_timepoints,
            polort=polort,
            device=device,
            noise_pcs=noise_pcs,
            verbose=False,
        )

        # Generate CV splits
        cv_splits = generate_cv_splits(n_runs, strategy=1)  # strategy=1 is LORO
        cv_metric_token = "sse" if args.metric == "sse" else "r2"
        cv_metric_label = "SSE (lower=better)" if args.metric == "sse" else "R²"

        per_voxel_design = design_matrix.ndim == 3
        _hib = metric_higher_is_better(args.metric)

        if not per_voxel_design:
            # ---- Standard path: single design for all voxels ----
            print("Projecting nuisance from data and design...")
            # data_clean returned on CPU for large datasets (project_out_nuisance_per_run
            # allocates output on "cpu" when chunking is needed, see xval.py:211)
            data_clean, design_clean = project_out_nuisance_per_run(
                data, design_matrix, nuisance_per_run, run_starts, device=device
            )

            # Pre-move design to device once (small: n_timepoints × n_trials)
            design_clean_dev = design_clean.to(device)

            # Allocate output accumulators on CPU
            n_cols = design_matrix.shape[1]
            final_betas = torch.zeros(n_voxels, n_cols)  # CPU
            xval_r2 = torch.zeros(n_voxels)  # CPU
            full_r2 = torch.zeros(n_voxels)  # CPU
            optimal_fracs = torch.zeros(n_voxels)  # CPU
            r2_by_frac = torch.zeros(n_voxels, len(fracs))  # CPU

            # Auto-detect chunk size if not specified
            if args.chunk_size <= 0:
                from fastfuncstuff.memory import estimate_chunk_size

                args.chunk_size = estimate_chunk_size(
                    n_voxels=n_voxels,
                    n_timepoints=n_timepoints,
                    n_regressors=n_cols,
                    device=device,
                    operation="ridge",
                    verbose=args.verb >= 1,
                )

            n_chunks = (n_voxels + args.chunk_size - 1) // args.chunk_size
            print(
                f"Fitting ridge + beta CV in {n_chunks} voxel chunks "
                f"({args.chunk_size:,} voxels/chunk)..."
            )

            fracs_t = torch.from_numpy(fracs)  # CPU tensor for indexing CPU optimal_fracs

            for c0 in range(0, n_voxels, args.chunk_size):
                c1 = min(c0 + args.chunk_size, n_voxels)
                chunk = c1 - c0

                # data_chunk is on CPU; _fit_ridge_multiple_fracs will stream it to device
                data_chunk = data_clean[c0:c1]  # (chunk, n_timepoints)

                # Fit ridge for all fracs for this voxel chunk
                # chunk is already small — use single-pass (no inner chunking)
                chunk_coefs = _fit_ridge_multiple_fracs(
                    design_clean_dev, data_chunk.T, fracs, device, chunk_size=None
                )
                # chunk_coefs: (n_cols, n_fracs, chunk) on device

                # Beta-space CV for this chunk (all fracs in one pass).
                # GLMsingle pattern: score every variant's condition-average
                # train betas against OLS (frac=1, last index) test betas.
                # This gives a meaningful amplitude signal: high-frac betas
                # ≈ OLS betas for high-SNR voxels, shrunken-to-zero for low-SNR.
                # Then exclude frac=1.0 from selection (always regularise a bit).
                chunk_betas_all = chunk_coefs.permute(1, 2, 0)  # (n_fracs, chunk, n_cols)
                n_fracs = len(fracs)
                xval = single_trial_cv_helper(
                    chunk_betas_all,
                    trial_condition_ids,
                    trial_run_ids,
                    cv_splits,
                    metric=args.metric,
                    zscore_by_run=args.zscore_by_run,
                    reference_variant_idx=n_fracs - 1,  # OLS (frac=1.0) sets z-score scale
                    test_variant_idx=n_fracs - 1,
                    device=device,
                    chunk_size=None,
                    verbose=False,
                )
                chunk_r2_frac = xval["r2"].T  # (chunk, n_fracs) on device

                # Select best frac excluding frac=1.0 (last column) — GLMsingle pattern
                # For SSE (GLMsingle "badness"), minimize; for R² metrics, maximize
                _hib = metric_higher_is_better(args.metric)
                if _hib:
                    chunk_xval, chunk_best_idx = chunk_r2_frac[:, :-1].max(dim=1)
                else:
                    chunk_xval, chunk_best_idx = chunk_r2_frac[:, :-1].min(dim=1)
                xval_r2[c0:c1] = chunk_xval.cpu()
                r2_by_frac[c0:c1] = chunk_r2_frac.cpu()
                optimal_fracs[c0:c1] = fracs_t[chunk_best_idx.cpu()]

                # Extract final betas at optimal fraction for this chunk
                vox_idx = torch.arange(chunk, device=device)
                chunk_final = chunk_coefs[:, chunk_best_idx, vox_idx].T  # (chunk, n_cols)
                del chunk_coefs

                # Apply autoscale if requested
                if args.autoscale:
                    data_chunk_dev = data_chunk.to(device)
                    predicted = chunk_final @ design_clean_dev.T
                    numer = (data_chunk_dev * predicted).sum(dim=1)
                    denom = (predicted * predicted).sum(dim=1) + 1e-10
                    chunk_final = chunk_final * (numer / denom).unsqueeze(1)

                final_betas[c0:c1] = chunk_final.cpu()

            # Compute full-model R²: COD of task prediction vs nuisance-projected data.
            # Both data_clean and design_clean_dev are nuisance-projected, so this
            # measures task-explained variance — the same quantity GLMsingle reports.
            print("Computing full-model R²...")
            for c0 in range(0, n_voxels, args.chunk_size):
                c1 = min(c0 + args.chunk_size, n_voxels)
                betas_chunk = final_betas[c0:c1].to(device)  # (chunk, n_cols)
                data_chunk = data_clean[c0:c1].to(device)  # (chunk, n_tp)
                predicted = betas_chunk @ design_clean_dev.T  # (chunk, n_tp)
                full_r2[c0:c1] = compute_r2_metric(data_chunk, predicted, metric="cod").cpu()

            # Move summary stats to device for downstream printing / saving
            xval_r2 = xval_r2.to(device)
            full_r2 = full_r2.to(device)
            optimal_fracs = optimal_fracs.to(device)
            r2_by_frac = r2_by_frac.to(device)
            final_betas = final_betas.to(device)

        else:
            # ---- Per-voxel HRF path: group by unique design ----
            # design_matrix is (n_unique_hrfs, n_timepoints, n_trials)
            assert hrf_indices is not None, "hrf_indices required for per-voxel designs"
            unique_hrfs = torch.unique(hrf_indices).tolist()
            print(f"Per-voxel HRF mode: {len(unique_hrfs)} unique HRFs")

            # Allocate outputs
            final_betas = torch.zeros(n_voxels, n_columns, device=device)
            xval_r2 = torch.zeros(n_voxels, device=device)
            full_r2 = torch.zeros(n_voxels, device=device)
            optimal_fracs = torch.zeros(n_voxels, device=device)
            r2_by_frac = torch.zeros(n_voxels, len(fracs), device=device)

            # Auto-detect chunk size if not specified
            if args.chunk_size <= 0:
                from fastfuncstuff.memory import estimate_chunk_size

                args.chunk_size = estimate_chunk_size(
                    n_voxels=n_voxels,
                    n_timepoints=n_timepoints,
                    n_regressors=n_columns,
                    device=device,
                    operation="ridge",
                    verbose=args.verb >= 1,
                )

            # Project nuisance from data once (shared across HRF groups)
            # Use a dummy 2D design just to get projected data
            dummy_design = design_matrix[0]  # (n_timepoints, n_trials)
            data_clean, _ = project_out_nuisance_per_run(
                data, dummy_design, nuisance_per_run, run_starts, device=device
            )

            fracs_dev = torch.from_numpy(fracs).to(device)  # for GPU indexing in per-voxel path

            for hrf_idx in unique_hrfs:
                voxel_mask_cpu = hrf_indices == hrf_idx  # CPU mask for indexing CPU data_clean
                group_voxel_indices = torch.where(voxel_mask_cpu)[
                    0
                ]  # Linear indices of voxels in this group
                n_group = len(group_voxel_indices)
                print(f"  HRF {hrf_idx}: {n_group:,} voxels")

                # Get this group's 2D design and project nuisance
                group_design_2d = design_matrix[hrf_idx]  # (n_timepoints, n_trials)
                _, design_clean_group = project_out_nuisance_per_run(
                    data[:1],  # minimal data, we only need the projected design
                    group_design_2d,
                    nuisance_per_run,
                    run_starts,
                    device=device,
                )

                # Process this HRF group in chunks to avoid OOM on large groups
                for gc0 in range(0, n_group, args.chunk_size):
                    gc1 = min(gc0 + args.chunk_size, n_group)
                    chunk_size_actual = gc1 - gc0

                    # Linear voxel indices for this chunk — use directly, no 167k-element
                    # bool mask clone/zero/scatter/transfer needed
                    chunk_voxel_idx = group_voxel_indices[gc0:gc1]  # CPU, (chunk,)
                    chunk_voxel_idx_dev = chunk_voxel_idx.to(device)  # GPU, small

                    # Extract data using integer indexing
                    chunk_data_clean = data_clean[chunk_voxel_idx]  # (chunk, n_timepoints)

                    # Fit ridge for all fractions
                    chunk_coefs = _fit_ridge_multiple_fracs(
                        design_clean_group, chunk_data_clean.T, fracs, device
                    )
                    # chunk_coefs: (n_trials, n_fracs, chunk)

                    # Batch beta-space CV across all fractions.
                    # GLMsingle pattern: score against OLS (frac=1, last index)
                    # test betas, then exclude frac=1.0 from selection.
                    all_chunk_betas = chunk_coefs.permute(1, 2, 0)  # (n_fracs, chunk, n_trials)
                    n_fracs_pv = len(fracs)
                    xval = single_trial_cv_helper(
                        all_chunk_betas,
                        trial_condition_ids,
                        trial_run_ids,
                        cv_splits,
                        metric=args.metric,
                        zscore_by_run=args.zscore_by_run,
                        reference_variant_idx=n_fracs_pv - 1,
                        test_variant_idx=n_fracs_pv - 1,
                        device=device,
                        verbose=False,
                    )
                    r2_chunk = xval["r2"].T.to(device)  # (chunk, n_fracs)
                    r2_by_frac[chunk_voxel_idx_dev] = r2_chunk

                    # Select best frac excluding frac=1.0 (last column)
                    # Use r2_chunk directly — avoids reading back from r2_by_frac
                    # SSE: minimize; R² metrics: maximize
                    if _hib:
                        chunk_r2, chunk_best_idx = r2_chunk[:, :-1].max(dim=1)
                    else:
                        chunk_r2, chunk_best_idx = r2_chunk[:, :-1].min(dim=1)
                    xval_r2[chunk_voxel_idx_dev] = chunk_r2
                    optimal_fracs[chunk_voxel_idx_dev] = fracs_dev[chunk_best_idx]

                    # Extract final betas at optimal fraction
                    chunk_voxel_range = torch.arange(chunk_size_actual, device=device)
                    chunk_final_betas = chunk_coefs[:, chunk_best_idx, chunk_voxel_range].T

                    # Autoscale if requested
                    if args.autoscale:
                        predicted = chunk_final_betas @ design_clean_group.T
                        numer = (chunk_data_clean.to(device) * predicted).sum(dim=1)
                        denom = (predicted * predicted).sum(dim=1) + 1e-10
                        scale_factors = numer / denom
                        chunk_final_betas *= scale_factors.unsqueeze(1)

                    final_betas[chunk_voxel_idx_dev] = chunk_final_betas

                    # Full-model R²: task variance explained after nuisance projection
                    predicted_full = chunk_final_betas @ design_clean_group.T  # (chunk, n_tp)
                    full_r2[chunk_voxel_idx_dev] = compute_r2_metric(
                        chunk_data_clean.to(device), predicted_full, metric="cod"
                    )
                    del predicted_full

                    # Cleanup
                    del chunk_coefs, chunk_final_betas, chunk_data_clean

        print()
        print(
            f"Beta-space CV {cv_metric_label}:  mean={xval_r2.mean():.4f}, median={xval_r2.median():.4f}"
        )
        print(f"Full-model R²:     mean={full_r2.mean():.4f}, median={full_r2.median():.4f}")

    else:
        # ========== EXISTING TIMESERIES CV PATH (unchanged) ==========
        results = fit_ridge_single_trial(
            data=data,
            design_matrix=design_matrix,
            run_starts=run_starts,
            tr=args.tr,
            trial_condition_ids=trial_condition_ids,
            condition_design=condition_design,
            fracs=fracs,
            nuisance=noise_pcs,
            polort=args.polort,
            trial_labels=trial_labels,
            autoscale=args.autoscale,
            chunk_size=args.chunk_size,
            device=device,
            verbose=args.verb >= 1,
        )

    # Save outputs
    print()
    print("Saving outputs...")
    output_prefix = Path(args.prefix)
    output_dir = output_prefix.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get volume shape and affine for saving
    vol_shape = volume_shape
    # affine is already extracted from load_result

    # Companion table describing every single-trial volume (both output paths below
    # write one volume per trial, in the design's chronological order).
    from fastfuncstuff.design.trial_table import write_single_trial_event_table

    write_single_trial_event_table(
        args.prefix,
        args.events,
        run_starts,
        args.tr,
        event_ignore=args.event_ignore,
        event_cols=tuple(args.event_cols) if args.event_cols else None,
        n_runs=n_runs,
        n_basis=n_basis,
    )

    if args.single_trials:
        # ========== SINGLE-TRIAL OUTPUT MODE ==========
        # Use save_single_trial_results for single-trial mode
        voxel_mask_tensor = torch.from_numpy(mask_flat) if mask is not None else None

        with spinner("Writing single-trial results"):
            save_single_trial_results(
                betas=final_betas,
                xval_r2=xval_r2,
                trial_labels=trial_labels,
                trial_condition_ids=trial_condition_ids,
                trial_run_ids=trial_run_ids,
                condition_labels=condition_labels,
                output_prefix=str(args.prefix),
                volume_shape=vol_shape,
                affine=affine,
                voxel_mask=voxel_mask_tensor,
                xval_metric_name=cv_metric_token,
            )

        # Also save ridge-specific outputs
        voxel_mask_np = mask_flat if mask is not None else None
        save_volume_nifti(
            full_r2,
            f"{args.prefix}_single_trial_full_r2{_nii_ext}",
            vol_shape,
            affine,
            voxel_mask_np,
        )
        print(f"  {args.prefix}_single_trial_full_r2{_nii_ext}")
        save_volume_nifti(
            optimal_fracs, f"{args.prefix}_optimal_frac{_nii_ext}", vol_shape, affine, voxel_mask_np
        )
        print(f"  {args.prefix}_optimal_frac{_nii_ext}")
        save_4d_nifti(
            r2_by_frac,
            f"{args.prefix}_{cv_metric_token}_by_frac{_nii_ext}",
            vol_shape,
            affine,
            voxel_mask_np,
        )
        print(f"  {args.prefix}_{cv_metric_token}_by_frac{_nii_ext}")

        if args.noise_ceiling in ("auto", "ncsnr"):
            from fastfuncstuff.stats.noise_ceiling import (
                mean_train_repeats,
                ncsnr,
                ncsnr_noise_ceiling,
                zscore_betas_by_run,
            )

            # The ceiling must see the betas the CV actually scored. With
            # -zscore_by_run (the default) the CV strips each run's mean and
            # scale first, so a ceiling from the raw betas bounds a different
            # quantity and the explainable fraction runs past 1.
            ceiling_betas = final_betas.cpu()
            if args.zscore_by_run and args.metric != "sse":
                ceiling_betas = zscore_betas_by_run(ceiling_betas, trial_run_ids.cpu())

            # The CV predicts each held-out trial from an average of m training
            # trials, so the divisor has to account for that predictor's own
            # noise; the published NSD form assumes a noiseless one and would
            # read as though the model fell short of a ceiling it could not
            # reach. Both are saved -- ncsnr for comparison with NSD maps, the
            # fold-matched ceiling for the ratio.
            repeats = mean_train_repeats(trial_condition_ids, trial_run_ids, cv_splits)
            ceiling = ncsnr_noise_ceiling(
                ceiling_betas, trial_condition_ids.cpu(), n_train_repeats=repeats
            )
            print()
            print(f"Noise ceiling (beta space, m={repeats:.1f} training trials/condition):")
            print(f"  {ceiling.summarize()}")
            for note in ceiling.notes:
                print(f"  NOTE: {note}")

            if ceiling.n_usable:
                save_volume_nifti(
                    ncsnr(ceiling_betas, trial_condition_ids.cpu()),
                    f"{args.prefix}_ncsnr{_nii_ext}",
                    vol_shape,
                    affine,
                    voxel_mask_np,
                )
                print(f"  {args.prefix}_ncsnr{_nii_ext}")
                save_volume_nifti(
                    ceiling.ceiling,
                    f"{args.prefix}_noise_ceiling{_nii_ext}",
                    vol_shape,
                    affine,
                    voxel_mask_np,
                )
                print(f"  {args.prefix}_noise_ceiling{_nii_ext}")
                # Only meaningful against a coefficient-of-determination CV.
                # Test args.metric, not cv_metric_token: the token collapses
                # cod/corr/corr2 to "r2", and a squared correlation is not the
                # variance fraction the ceiling is expressed in.
                if args.metric == "cod":
                    save_volume_nifti(
                        ceiling.explainable_r2(xval_r2.cpu()),
                        f"{args.prefix}_explainable_r2{_nii_ext}",
                        vol_shape,
                        affine,
                        voxel_mask_np,
                    )
                    print(f"  {args.prefix}_explainable_r2{_nii_ext}")
                else:
                    print(
                        f"  (no explainable_r2: -metric {args.metric} is not on the "
                        "variance-fraction scale the ceiling uses, so the ratio would "
                        "not mean anything; use -metric cod)"
                    )

    else:
        # ========== EXISTING OUTPUT MODE ==========
        voxel_mask_np = mask_flat if mask is not None else None

        # Save R² maps and optimal fractions
        save_volume_nifti(
            results.r2_initial,
            f"{args.prefix}_r2_initial{_nii_ext}",
            vol_shape,
            affine,
            voxel_mask_np,
        )
        print(f"  {args.prefix}_r2_initial{_nii_ext}")
        save_volume_nifti(
            results.r2, f"{args.prefix}_r2{_nii_ext}", vol_shape, affine, voxel_mask_np
        )
        print(f"  {args.prefix}_r2{_nii_ext}")
        save_volume_nifti(
            results.xval_r2, f"{args.prefix}_xval_r2{_nii_ext}", vol_shape, affine, voxel_mask_np
        )
        print(f"  {args.prefix}_xval_r2{_nii_ext}")
        save_volume_nifti(
            results.optimal_fracs,
            f"{args.prefix}_optimal_frac{_nii_ext}",
            vol_shape,
            affine,
            voxel_mask_np,
        )
        print(f"  {args.prefix}_optimal_frac{_nii_ext}")

        # Save single-trial betas (4D file)
        save_4d_nifti(
            results.betas_single_trial,
            f"{args.prefix}_betas_single_trial{_nii_ext}",
            vol_shape,
            affine,
            voxel_mask_np,
        )
        print(f"  {args.prefix}_betas_single_trial{_nii_ext}")

        # Save trial labels text file
        trial_labels_file = f"{args.prefix}_trial_labels.txt"
        with open(trial_labels_file, "w") as f:
            for label in results.trial_labels:
                f.write(f"{label}\n")
        print(f"  {trial_labels_file}")

        # Compute and save mean condition betas (average single-trial betas within each condition)
        print()
        print("Computing mean condition betas...")
        n_conditions = trial_condition_ids.max().item() + 1
        condition_betas = torch.zeros(n_voxels, n_conditions, device="cpu")
        for cond_idx in range(n_conditions):
            cond_mask = trial_condition_ids.cpu() == cond_idx
            if cond_mask.sum() > 0:
                condition_betas[:, cond_idx] = results.betas_single_trial[:, cond_mask].mean(dim=1)

        save_4d_nifti(
            condition_betas,
            f"{args.prefix}_betas_condition{_nii_ext}",
            vol_shape,
            affine,
            voxel_mask_np,
        )
        print(f"  {args.prefix}_betas_condition{_nii_ext}")

        # Save condition labels text file
        condition_labels_file = f"{args.prefix}_condition_labels.txt"
        with open(condition_labels_file, "w") as f:
            for label in condition_labels:
                f.write(f"{label}\n")
        print(f"  {condition_labels_file}")

        # Save R² per ridge fraction (4D file with one volume per fraction)
        save_4d_nifti(
            results.r2_by_frac,
            f"{args.prefix}_r2_by_frac{_nii_ext}",
            vol_shape,
            affine,
            voxel_mask_np,
        )
        print(f"  {args.prefix}_r2_by_frac{_nii_ext}")

        # Save fractions text file for reference
        fracs_file = f"{args.prefix}_ridge_fracs.txt"
        with open(fracs_file, "w") as f:
            for frac in fracs:
                f.write(f"{frac:.4f}\n")
        print(f"  {fracs_file}")

        # Save scaling violation mask if scaling was performed
        if args.do_scale and violations_mask is not None and scale_info is not None:
            # Sum violations across time to get count per voxel
            violation_counts = violations_mask.cpu().sum(dim=1).numpy()  # (n_voxels,)
            save_volume_nifti(
                torch.from_numpy(violation_counts),
                f"{args.prefix}_scale_violations{_nii_ext}",
                vol_shape,
                affine,
                voxel_mask_np,
            )
            print(f"  {args.prefix}_scale_violations{_nii_ext}")

        print()
        print("Output files created:")
        print(f"  {args.prefix}_r2_initial{_nii_ext} - Initial R² (minimal ridge, ~OLS)")
        print(f"  {args.prefix}_r2{_nii_ext} - Final R² (in-sample, at optimal ridge)")
        print(f"  {args.prefix}_xval_r2{_nii_ext} - Cross-validated R²")
        print(f"  {args.prefix}_optimal_frac{_nii_ext} - Optimal ridge fraction per voxel")
        print(
            f"  {args.prefix}_betas_single_trial{_nii_ext} - Single-trial betas (4D, {n_columns} volumes)"
        )
        print(f"  {args.prefix}_trial_labels.txt - Trial labels for single-trial betas")
        print(
            f"  {args.prefix}_betas_condition{_nii_ext} - Mean condition betas (4D, {n_conditions} volumes)"
        )
        print(f"  {args.prefix}_condition_labels.txt - Condition labels for condition betas")
        print(
            f"  {args.prefix}_r2_by_frac{_nii_ext} - CV R² per ridge fraction (4D, {len(fracs)} volumes)"
        )
        print(f"  {args.prefix}_ridge_fracs.txt - Ridge fraction values")
        if args.do_scale and violations_mask is not None:
            print(
                f"  {args.prefix}_scale_violations{_nii_ext} - Scaling violation counts per voxel"
            )
        print()
        print("Summary statistics:")
        print(f"  Median R² (initial, OLS): {results.r2_initial.median():.4f}")
        print(f"  Median R² (final, ridge): {results.r2.median():.4f}")
        print(f"  Median R² (xval): {results.xval_r2.median():.4f}")
        print(f"  Median optimal ridge fraction: {results.optimal_fracs.median():.4f}")

        print()
        print("=" * 70)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)


if __name__ == "__main__":
    main()
