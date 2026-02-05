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

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

# Import fastfuncsim modules
try:
    from fastfuncsim.afni_io import load_afni_mask, load_nifti
    from fastfuncsim.cli_utils import (
        auto_polort,
        build_nuisance_per_run,
        compute_run_lengths,
        get_average_run_duration,
        load_and_preprocess_runs,
        LoadResult,
        save_4d_nifti,
        save_volume_nifti,
    )
    from fastfuncsim.design_builder import parse_afni_timing_file, parse_durations
    from fastfuncsim.hrf import get_hrf_library
    from fastfuncsim.ridge import (
        create_single_trial_design,
        fit_ridge_single_trial,
        load_hrf_indices,
        load_noise_pcs,
    )
    from fastfuncsim.utils import get_device, scale_to_percent_signal
except ImportError as e:
    print(f"ERROR: Could not import fastfuncsim: {e}")
    print("Make sure fastfuncsim is installed: pip install -e .")
    sys.exit(1)


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="3dRidgefast - GPU-accelerated ridge regression with single-trial estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
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
        required=True,
        help="Onset timing files (AFNI format). One file per condition.",
    )
    required.add_argument(
        "-durations",
        nargs="+",
        required=True,
        help="Stimulus durations in seconds. Single value, one per condition, or use 'value,count' "
             "format (e.g., '3,20 5,40' for 20 conditions with 3s, then 40 with 5s).",
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
        "-device",
        type=str,
        help="Force device: 'cpu' or 'cuda' (default: auto-detect GPU)",
    )
    proc_opts.add_argument(
        "-chunk_size",
        type=int,
        default=10000,
        help="Voxels per chunk for processing (default: 10000)",
    )
    proc_opts.add_argument(
        "-verbose",
        action="store_true",
        help="Print detailed progress information",
    )
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

    # Help
    help_group = parser.add_argument_group("Help")
    help_group.add_argument(
        "-help",
        action="store_true",
        help="Show this help message and exit",
    )

    return parser


def main():
    """Main entry point"""
    parser = create_parser()

    # Handle -help flag
    if "-help" in sys.argv:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    print("=" * 70)
    print("3dRidgefast - Ridge regression with single-trial estimation")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ========================================================================
    # Parse onsets and durations (BEFORE loading data - fail fast)
    # ========================================================================
    print("Parsing onset files...")
    all_onsets = []
    condition_labels = []
    input_files = args.input
    n_runs = len(input_files)

    for onset_file in args.onsets:
        condition_label = Path(onset_file).stem
        runs_onsets = parse_afni_timing_file(onset_file)
        all_onsets.append(runs_onsets)
        condition_labels.append(condition_label)
        n_events = sum(len(run_onsets) for run_onsets in runs_onsets)
        print(f"  {condition_label}: {n_events} events across {len(runs_onsets)} runs")

    # Parse durations (supports formats: "2.0", "3,20" for 20 copies of 3)
    print()
    print("Parsing durations...")
    durations = parse_durations(args.durations, len(all_onsets), condition_labels)

    # Print summary based on input
    if len(args.durations) == 1 and ',' not in args.durations[0]:
        print(f"  Using {durations[0]}s for all {len(all_onsets)} conditions")
    else:
        print(f"  Matched {len(durations)} durations to {len(all_onsets)} conditions")

    print()
    print("=" * 70)
    print(f"Validated: {len(all_onsets)} conditions, {len(durations)} durations")
    print("=" * 70)

    # ========================================================================
    # Set device and load data
    # ========================================================================
    print()
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f"Device: {device}")
    print()

    # Load data
    print("Loading data...")

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

    # Update args.tr with loaded value (for later use)
    if args.tr is None:
        args.tr = load_result.tr

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
        hrf_library = get_hrf_library(mode="library", tr=args.tr, n_hrfs=20)
        print(f"  Using HRF library with {len(hrf_library)} HRFs")

    # Load noise PCs if provided
    noise_pcs = None
    if args.denoise:
        print()
        print(f"Loading denoising results from {args.denoise}...")
        noise_pc_file = f"{args.denoise}_noise_pcs.xmat.1D"
        noise_pcs = load_noise_pcs(noise_pc_file, run_starts, n_timepoints)
        n_pcs = noise_pcs[0].shape[1]
        print(f"  Loaded {n_pcs} noise PCs per run")

    # Create single-trial design matrix
    print()
    print("Creating single-trial design matrix...")
    design_matrix, trial_labels, trial_condition_ids, trial_run_ids, condition_design = create_single_trial_design(
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
    )

    n_trials = len(trial_labels)
    n_conditions = condition_design.shape[1]
    print(f"  Total trials: {n_trials}")
    print(f"  Conditions: {n_conditions}")
    print(f"  Design shape: {design_matrix.shape}")

    # Parse ridge fractions
    frac_start, frac_end, frac_step = args.ridge_fracs
    fracs = np.arange(frac_start, frac_end + frac_step/2, frac_step)

    print()
    print("=" * 70)
    print("Ridge regression estimation")
    print("=" * 70)
    print()

    if args.single_trials:
        # ========== SINGLE-TRIAL BETA-SPACE CV PATH ==========
        from fastfuncsim.glm_core import construct_polynomial_matrix
        from fastfuncsim.glm_outputs import save_single_trial_results
        from fastfuncsim.ridge import _fit_ridge_multiple_fracs
        from fastfuncsim.xval import (
            compute_xval_r2_single_trials,
            generate_cv_splits,
            project_out_nuisance_per_run,
        )

        print("Using beta-space cross-validation (GLMsingle-style)")
        print()

        # Build nuisance design (polynomials + noise PCs)
        # Auto-determine polort if not specified
        if args.polort is None:
            run_lengths = compute_run_lengths(run_starts, n_timepoints)
            avg_run_duration_sec = get_average_run_duration(run_lengths, args.tr)
            polort = auto_polort(avg_run_duration_sec, formula="afni")
            print(f"Auto polort: {polort} (based on {avg_run_duration_sec/60:.1f} min avg run)")
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

        per_voxel_design = design_matrix.ndim == 3

        if not per_voxel_design:
            # ---- Standard path: single design for all voxels ----
            print("Projecting nuisance from data and design...")
            data_clean, design_clean = project_out_nuisance_per_run(
                data, design_matrix, nuisance_per_run, run_starts, device=device)

            print(f"Fitting ridge for {len(fracs)} fractions on full data...")
            coefs = _fit_ridge_multiple_fracs(design_clean, data_clean.T, fracs, device)
            # coefs: (n_trials, n_fracs, n_voxels)

            # For each fraction: compute beta-space R²
            print("Computing beta-space CV R² for each fraction...")
            r2_by_frac = torch.zeros(n_voxels, len(fracs), device=device)
            for frac_idx in range(len(fracs)):
                frac_betas = coefs[:, frac_idx, :].T  # (n_voxels, n_trials)
                xval = compute_xval_r2_single_trials(
                    frac_betas, trial_condition_ids, trial_run_ids, cv_splits,
                    device=device, verbose=False)
                r2_by_frac[:, frac_idx] = xval['r2']

            # Select optimal fraction per voxel
            xval_r2, best_frac_idx = r2_by_frac.max(dim=1)
            optimal_fracs = torch.from_numpy(fracs[best_frac_idx.cpu().numpy()]).to(device)

            # Extract final betas at optimal fraction
            voxel_indices = torch.arange(n_voxels, device=device)
            final_betas = coefs[:, best_frac_idx, voxel_indices].T  # (n_voxels, n_trials)

            # Apply autoscale if requested
            if args.autoscale:
                print("Applying GLMsingle-style autoscaling...")
                predicted = final_betas @ design_clean.T  # (n_voxels, n_timepoints)
                numer = (data_clean * predicted).sum(dim=1)
                denom = (predicted * predicted).sum(dim=1) + 1e-10
                scale_factors = numer / denom
                final_betas *= scale_factors.unsqueeze(1)

        else:
            # ---- Per-voxel HRF path: group by unique design ----
            # design_matrix is (n_unique_hrfs, n_timepoints, n_trials)
            assert hrf_indices is not None, "hrf_indices required for per-voxel designs"
            hrf_indices_device = hrf_indices.to(device)
            unique_hrfs = torch.unique(hrf_indices_device).tolist()
            print(f"Per-voxel HRF mode: {len(unique_hrfs)} unique HRFs")

            # Allocate outputs
            final_betas = torch.zeros(n_voxels, n_trials, device=device)
            xval_r2 = torch.zeros(n_voxels, device=device)
            optimal_fracs = torch.zeros(n_voxels, device=device)
            r2_by_frac = torch.zeros(n_voxels, len(fracs), device=device)

            # Project nuisance from data once (shared across HRF groups)
            # Use a dummy 2D design just to get projected data
            dummy_design = design_matrix[0]  # (n_timepoints, n_trials)
            data_clean, _ = project_out_nuisance_per_run(
                data, dummy_design, nuisance_per_run, run_starts, device=device)

            for hrf_idx in unique_hrfs:
                voxel_mask = hrf_indices_device == hrf_idx
                n_group = voxel_mask.sum().item()
                print(f"  HRF {hrf_idx}: {n_group:,} voxels")

                # Get this group's 2D design and project nuisance
                group_design_2d = design_matrix[hrf_idx]  # (n_timepoints, n_trials)
                _, design_clean_group = project_out_nuisance_per_run(
                    data[:1],  # minimal data, we only need the projected design
                    group_design_2d, nuisance_per_run, run_starts, device=device)

                group_data_clean = data_clean[voxel_mask]  # (n_group, n_timepoints)

                # Fit ridge for all fractions
                group_coefs = _fit_ridge_multiple_fracs(
                    design_clean_group, group_data_clean.T, fracs, device)
                # group_coefs: (n_trials, n_fracs, n_group)

                # Beta-space CV for each fraction
                for frac_idx in range(len(fracs)):
                    frac_betas = group_coefs[:, frac_idx, :].T  # (n_group, n_trials)
                    xval = compute_xval_r2_single_trials(
                        frac_betas, trial_condition_ids, trial_run_ids, cv_splits,
                        device=device, verbose=False)
                    r2_by_frac[voxel_mask, frac_idx] = xval['r2'].to(device)

                # Select optimal fraction per voxel in this group
                group_r2, group_best_idx = r2_by_frac[voxel_mask].max(dim=1)
                xval_r2[voxel_mask] = group_r2
                optimal_fracs[voxel_mask] = torch.from_numpy(
                    fracs[group_best_idx.cpu().numpy()]).to(device)

                # Extract final betas at optimal fraction
                group_voxel_indices = torch.arange(n_group, device=device)
                group_final_betas = group_coefs[:, group_best_idx, group_voxel_indices].T

                # Autoscale if requested
                if args.autoscale:
                    predicted = group_final_betas @ design_clean_group.T
                    numer = (group_data_clean * predicted).sum(dim=1)
                    denom = (predicted * predicted).sum(dim=1) + 1e-10
                    scale_factors = numer / denom
                    group_final_betas *= scale_factors.unsqueeze(1)

                final_betas[voxel_mask] = group_final_betas

        print()
        print(f"Beta-space CV R²: mean={xval_r2.mean():.4f}, median={xval_r2.median():.4f}")

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
            verbose=args.verbose,
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

    if args.single_trials:
        # ========== SINGLE-TRIAL OUTPUT MODE ==========
        # Use save_single_trial_results for single-trial mode
        voxel_mask_tensor = torch.from_numpy(mask_flat) if mask is not None else None

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
        )

        # Also save ridge-specific outputs
        # Save optimal fractions and R² per fraction
        voxel_mask_np = mask_flat if mask is not None else None
        save_volume_nifti(optimal_fracs, f"{args.prefix}_optimal_frac.nii.gz", vol_shape, affine, voxel_mask_np)
        print(f"  {args.prefix}_optimal_frac.nii.gz")
        save_4d_nifti(r2_by_frac, f"{args.prefix}_r2_by_frac.nii.gz", vol_shape, affine, voxel_mask_np)
        print(f"  {args.prefix}_r2_by_frac.nii.gz")

    else:
        # ========== EXISTING OUTPUT MODE ==========
        voxel_mask_np = mask_flat if mask is not None else None

        # Save R² maps and optimal fractions
        save_volume_nifti(results.r2_initial, f"{args.prefix}_r2_initial.nii.gz", vol_shape, affine, voxel_mask_np)
        print(f"  {args.prefix}_r2_initial.nii.gz")
        save_volume_nifti(results.r2, f"{args.prefix}_r2.nii.gz", vol_shape, affine, voxel_mask_np)
        print(f"  {args.prefix}_r2.nii.gz")
        save_volume_nifti(results.xval_r2, f"{args.prefix}_xval_r2.nii.gz", vol_shape, affine, voxel_mask_np)
        print(f"  {args.prefix}_xval_r2.nii.gz")
        save_volume_nifti(results.optimal_fracs, f"{args.prefix}_optimal_frac.nii.gz", vol_shape, affine, voxel_mask_np)
        print(f"  {args.prefix}_optimal_frac.nii.gz")

        # Save single-trial betas (4D file)
        save_4d_nifti(results.betas_single_trial, f"{args.prefix}_betas_single_trial.nii.gz", vol_shape, affine, voxel_mask_np)
        print(f"  {args.prefix}_betas_single_trial.nii.gz")

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

        save_4d_nifti(condition_betas, f"{args.prefix}_betas_condition.nii.gz", vol_shape, affine, voxel_mask_np)
        print(f"  {args.prefix}_betas_condition.nii.gz")

        # Save condition labels text file
        condition_labels_file = f"{args.prefix}_condition_labels.txt"
        with open(condition_labels_file, "w") as f:
            for label in condition_labels:
                f.write(f"{label}\n")
        print(f"  {condition_labels_file}")

        # Save R² per ridge fraction (4D file with one volume per fraction)
        save_4d_nifti(results.r2_by_frac, f"{args.prefix}_r2_by_frac.nii.gz", vol_shape, affine, voxel_mask_np)
        print(f"  {args.prefix}_r2_by_frac.nii.gz")

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
            save_volume_nifti(torch.from_numpy(violation_counts), f"{args.prefix}_scale_violations.nii.gz", vol_shape, affine, voxel_mask_np)
            print(f"  {args.prefix}_scale_violations.nii.gz")

        print()
        print("Output files created:")
        print(f"  {args.prefix}_r2_initial.nii.gz - Initial R² (minimal ridge, ~OLS)")
        print(f"  {args.prefix}_r2.nii.gz - Final R² (in-sample, at optimal ridge)")
        print(f"  {args.prefix}_xval_r2.nii.gz - Cross-validated R²")
        print(f"  {args.prefix}_optimal_frac.nii.gz - Optimal ridge fraction per voxel")
        print(f"  {args.prefix}_betas_single_trial.nii.gz - Single-trial betas (4D, {n_trials} volumes)")
        print(f"  {args.prefix}_trial_labels.txt - Trial labels for single-trial betas")
        print(f"  {args.prefix}_betas_condition.nii.gz - Mean condition betas (4D, {n_conditions} volumes)")
        print(f"  {args.prefix}_condition_labels.txt - Condition labels for condition betas")
        print(f"  {args.prefix}_r2_by_frac.nii.gz - CV R² per ridge fraction (4D, {len(fracs)} volumes)")
        print(f"  {args.prefix}_ridge_fracs.txt - Ridge fraction values")
        if args.do_scale and violations_mask is not None:
            print(f"  {args.prefix}_scale_violations.nii.gz - Scaling violation counts per voxel")
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
