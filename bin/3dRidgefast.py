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
    from fastfuncsim.afni_io import load_and_concatenate_runs, load_afni_mask, load_nifti
    from fastfuncsim.design_builder import parse_afni_timing_file
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
    durations_parsed = []
    for d in args.durations:
        if ',' in d:
            # Parse "value,count" format (e.g., "3,20" -> [3, 3, 3, ...])
            try:
                value_str, count_str = d.split(',')
                value = float(value_str)
                count = int(count_str)
                durations_parsed.extend([value] * count)
            except (ValueError, IndexError):
                print(f"ERROR: Invalid duration format '{d}'. Use 'value' or 'value,count' (e.g., '3,20')")
                sys.exit(1)
        else:
            # Single value
            durations_parsed.append(float(d))

    # Check if parsed durations match conditions
    if len(durations_parsed) == 1:
        durations = durations_parsed * len(all_onsets)
        print(f"  Using {durations_parsed[0]}s for all {len(all_onsets)} conditions")
    elif len(durations_parsed) == len(all_onsets):
        durations = durations_parsed
        print(f"  Matched {len(durations)} durations to {len(all_onsets)} conditions")
    else:
        print()
        print(f"ERROR: Number of durations ({len(durations_parsed)}) must be 1 or match number of conditions ({len(all_onsets)})")
        print(f"  Got: {len(durations_parsed)} durations for {len(all_onsets)} conditions")
        print(f"  Onset files: {len(args.onsets)}")
        print(f"  Duration spec: {args.durations}")
        sys.exit(1)

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
    first_img = load_nifti(input_files[0])

    # Load mask if provided
    mask = None
    if args.mask:
        mask = load_afni_mask(args.mask)
        n_voxels_masked = mask.sum()
        print(f"  Mask: {args.mask} ({n_voxels_masked:,} voxels)")
    else:
        print("  No mask specified - using all voxels")

    # Load fMRI data
    mask_flat = mask.flatten().astype(bool) if mask is not None else None
    data, run_starts = load_and_concatenate_runs(
        [Path(f) for f in input_files],
        device=device,
        keep_on_cpu=False,
        mask_flat=mask_flat,
    )

    n_voxels, n_timepoints = data.shape
    n_runs = len(run_starts)

    print(f"  Data shape: {data.shape} ({n_voxels:,} voxels × {n_timepoints} timepoints)")
    print(f"  Runs: {n_runs}")

    # Get TR
    if args.tr is None:
        zooms = first_img.header.get_zooms()
        if len(zooms) > 3 and zooms[3] > 0:
            args.tr = float(zooms[3])
            print(f"  TR (from header): {args.tr}s")
        else:
            print("ERROR: Could not determine TR from header. Use -tr flag.")
            sys.exit(1)
    else:
        print(f"  TR (specified): {args.tr}s")

    # Apply scaling if requested
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
    design_matrix, trial_labels, trial_condition_ids, condition_design = create_single_trial_design(
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

    # Fit ridge regression
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
    vol_shape = first_img.shape[:3]
    affine = first_img.affine

    # Helper to reshape and save 3D volumes
    def save_result_volume(data_tensor, filename):
        if mask is not None:
            # Unmask to full volume
            full_data = np.zeros(mask.size, dtype=np.float32)
            full_data[mask_flat] = data_tensor.cpu().numpy()
            data_3d = full_data.reshape(vol_shape)
        else:
            # No mask - reshape to original volume
            data_3d = data_tensor.cpu().numpy().reshape(vol_shape)

        img = nib.Nifti1Image(data_3d, affine=affine)
        nib.save(img, filename)
        print(f"  {filename}")

    # Helper to reshape and save 4D volumes (e.g., betas, R² per frac)
    def save_result_4d(data_tensor, filename):
        """Save 4D volume: data_tensor shape (n_voxels, n_volumes)"""
        n_vols = data_tensor.shape[1]
        if mask is not None:
            # Unmask to full volume: (n_voxels, n_volumes) -> (x, y, z, n_volumes)
            full_data = np.zeros((mask.size, n_vols), dtype=np.float32)
            full_data[mask_flat, :] = data_tensor.cpu().numpy()
            data_4d = full_data.reshape((*vol_shape, n_vols))
        else:
            # No mask
            data_4d = data_tensor.cpu().numpy().reshape((*vol_shape, n_vols))

        img = nib.Nifti1Image(data_4d, affine=affine)
        nib.save(img, filename)
        print(f"  {filename}")

    # Save R² maps and optimal fractions
    save_result_volume(results.r2_initial, f"{args.prefix}_r2_initial.nii.gz")
    save_result_volume(results.r2, f"{args.prefix}_r2.nii.gz")
    save_result_volume(results.xval_r2, f"{args.prefix}_xval_r2.nii.gz")
    save_result_volume(results.optimal_fracs, f"{args.prefix}_optimal_frac.nii.gz")

    # Save single-trial betas (4D file)
    save_result_4d(results.betas_single_trial, f"{args.prefix}_betas_single_trial.nii.gz")

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

    save_result_4d(condition_betas, f"{args.prefix}_betas_condition.nii.gz")

    # Save condition labels text file
    condition_labels_file = f"{args.prefix}_condition_labels.txt"
    with open(condition_labels_file, "w") as f:
        for label in condition_labels:
            f.write(f"{label}\n")
    print(f"  {condition_labels_file}")

    # Save R² per ridge fraction (4D file with one volume per fraction)
    save_result_4d(results.r2_by_frac, f"{args.prefix}_r2_by_frac.nii.gz")

    # Save fractions text file for reference
    fracs_file = f"{args.prefix}_ridge_fracs.txt"
    with open(fracs_file, "w") as f:
        for frac in fracs:
            f.write(f"{frac:.4f}\n")
    print(f"  {fracs_file}")

    # Save scaling violation mask if scaling was performed
    if args.do_scale and violations_mask is not None and scale_info is not None:
        # Sum violations across time to get count per voxel
        violation_counts = violations_mask.sum(dim=1).cpu().numpy()  # (n_voxels,)
        save_result_volume(torch.from_numpy(violation_counts), f"{args.prefix}_scale_violations.nii.gz")

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
