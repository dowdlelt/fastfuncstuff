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
    from fastfuncsim.afni_io import load_and_concatenate_runs, load_nifti
    from fastfuncsim.design_builder import parse_afni_timing_file
    from fastfuncsim.hrf import get_hrf_library
    from fastfuncsim.ridge import (
        create_single_trial_design,
        fit_ridge_single_trial,
        load_hrf_indices,
        load_noise_pcs,
    )
    from fastfuncsim.utils import get_device
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

Notes:
  - Single-trial estimation: Each event gets its own beta estimate
  - Ridge fractions: Controls regularization strength (0=OLS, 1=max regularization)
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
        help="Stimulus durations in seconds. Either single value or one per condition.",
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
        help="Ridge fractions to test: START END STEP (default: 0.05 1.0 0.05)",
    )
    ridge_opts.add_argument(
        "-cv_strategy",
        type=str,
        default="loro",
        choices=["loro", "split_half"],
        help="Cross-validation strategy for fraction selection (default: loro)",
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

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f"Device: {device}")
    print()

    # Load data
    print("Loading data...")
    input_files = args.input
    first_img = load_nifti(input_files[0])

    # Load mask if provided
    mask = None
    if args.mask:
        from fastfuncsim.afni_io import load_afni_mask
        mask = load_afni_mask(args.mask, first_img)
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

    # Parse onsets
    print()
    print("Parsing onset files...")
    all_onsets = []
    condition_labels = []
    for onset_file in args.onsets:
        condition_label = Path(onset_file).stem
        runs_onsets = parse_afni_timing_file(onset_file, n_runs=n_runs)
        all_onsets.append(runs_onsets)
        condition_labels.append(condition_label)
        n_events = sum(len(run_onsets) for run_onsets in runs_onsets)
        print(f"  {condition_label}: {n_events} events across {n_runs} runs")

    # Parse durations
    durations_parsed = [float(d) for d in args.durations]
    if len(durations_parsed) == 1:
        durations = durations_parsed * len(all_onsets)
    elif len(durations_parsed) == len(all_onsets):
        durations = durations_parsed
    else:
        print(f"ERROR: Number of durations ({len(durations_parsed)}) must be 1 or match number of conditions ({len(all_onsets)})")
        sys.exit(1)

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
    design_matrix, trial_labels = create_single_trial_design(
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
    print(f"  Total trials: {n_trials}")
    print(f"  Design shape: {design_matrix.shape}")

    # Parse ridge fractions
    frac_start, frac_end, frac_step = args.ridge_fracs
    fracs = np.arange(frac_start, frac_end + frac_step/2, frac_step)

    print()
    print("=" * 70)
    print("Ridge regression estimation")
    print("=" * 70)
    print()

    # Fit ridge regression (placeholder - to be implemented)
    print("NOTE: Ridge regression fitting is not yet fully implemented.")
    print("The infrastructure is in place for:")
    print("  - Single-trial design matrix generation ✓")
    print("  - Per-voxel HRF integration ✓")
    print("  - Noise PC loading ✓")
    print("  - Cross-validated ridge fraction selection (TODO)")
    print("  - Beta estimation and saving (TODO)")
    print()
    print("Next steps:")
    print("  1. Implement fracridge CV loop")
    print("  2. Add beta and R² map saving")
    print("  3. Add trial-averaged condition betas")
    print("  4. Add visualization and QC outputs")

    # results = fit_ridge_single_trial(
    #     data=data,
    #     design_matrix=design_matrix,
    #     run_starts=run_starts,
    #     fracs=fracs,
    #     nuisance=noise_pcs,
    #     trial_labels=trial_labels,
    #     chunk_size=args.chunk_size,
    #     device=device,
    #     verbose=args.verbose,
    # )

    print()
    print("=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
