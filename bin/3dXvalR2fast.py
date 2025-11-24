#!/usr/bin/env python
"""
3dXvalR2fast - Fast cross-validated R² computation for fMRI GLM

Computes out-of-sample prediction accuracy using run-based cross-validation.
This provides a more reliable estimate of model generalization than in-sample R².

Main use cases:
- Testing denoising methods
- Model selection (e.g., HRF choice)
- Evaluating preprocessing pipelines
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch

try:
    from fastfuncsim.analysis import load_and_concatenate_runs
    from fastfuncsim.afni_io import extract_design_metadata, read_afni_design_matrix
    from fastfuncsim.utils import get_device
    from fastfuncsim.xval import compute_xval_r2, generate_cv_splits
except ImportError as e:
    print(f"ERROR: Could not import fastfuncsim: {e}")
    print("Make sure fastfuncsim is installed: pip install -e .")
    sys.exit(1)


def parse_input_files(input_str: str) -> List[str]:
    """Parse input file string (space-separated or single file)"""
    # Remove quotes if present
    input_str = input_str.strip().strip('"').strip("'")
    # Split on spaces
    files = input_str.split()
    # Validate files exist
    for f in files:
        if not Path(f).exists():
            print(f"ERROR: Input file not found: {f}")
            sys.exit(1)
    return files


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="3dXvalR2fast - Fast cross-validated R² for fMRI GLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split halves (default, 50/50 train/test)
  3dXvalR2fast -input func.nii.gz -matrix X.xmat.1D -output xvalR2

  # Multiple runs (space-separated, quoted)
  3dXvalR2fast -input "run1.nii.gz run2.nii.gz run3.nii.gz run4.nii.gz" \\
               -matrix X.xmat.1D -output xvalR2

  # Leave-one-run-out (LORO)
  3dXvalR2fast -input func.nii.gz -matrix X.xmat.1D \\
               -output xvalR2 -cv_strategy 1

  # Leave-two-runs-out
  3dXvalR2fast -input func.nii.gz -matrix X.xmat.1D \\
               -output xvalR2 -cv_strategy 2

  # Custom train fraction (60% train, 40% test)
  3dXvalR2fast -input func.nii.gz -matrix X.xmat.1D \\
               -output xvalR2 -cv_strategy 0.6

  # More permutations for better stability
  3dXvalR2fast -input func.nii.gz -matrix X.xmat.1D \\
               -output xvalR2 -nperms 200

  # Different R² metrics
  3dXvalR2fast -input func.nii.gz -matrix X.xmat.1D \\
               -output xvalR2 -metric cod    # Coefficient of determination (default)
  3dXvalR2fast ... -metric corr   # Pearson correlation
  3dXvalR2fast ... -metric corr2  # Pearson correlation squared

  # Save all split R² values (for inspection)
  3dXvalR2fast -input func.nii.gz -matrix X.xmat.1D \\
               -output xvalR2 -save_splits

Notes:
  - Requires multiple runs for cross-validation
  - Projects out nuisance regressors (motion, polynomials) before fitting
  - Returns median R² across splits (robust to outliers)
  - Cross-validated R² is typically lower than in-sample R²
  - Negative R² values indicate predictions worse than mean
        """,
    )

    # Required arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-input",
        type=str,
        required=True,
        help='Input fMRI data. Single file or space-separated list in quotes: "run1.nii.gz run2.nii.gz"',
    )
    required.add_argument(
        "-matrix",
        type=str,
        required=True,
        help="Design matrix file (AFNI .xmat.1D format)",
    )
    required.add_argument(
        "-output",
        type=str,
        required=True,
        help="Output prefix (will create PREFIX_median.nii.gz, PREFIX_std.nii.gz, etc.)",
    )

    # Cross-validation options
    cv_opts = parser.add_argument_group("Cross-Validation Options")
    cv_opts.add_argument(
        "-cv_strategy",
        type=str,
        default="0.5",
        metavar="STRATEGY",
        help=(
            "CV strategy: float (0-1) for train fraction (e.g., 0.5 = split halves), "
            "or int for leave-N-out (e.g., 1 = LORO). Default: 0.5"
        ),
    )
    cv_opts.add_argument(
        "-nperms",
        type=int,
        default=100,
        metavar="N",
        help="Number of permutations/splits to run. Default: 100",
    )
    cv_opts.add_argument(
        "-metric",
        type=str,
        default="cod",
        choices=["cod", "corr", "corr2"],
        help=(
            "R² metric: 'cod' = coefficient of determination (traditional R²), "
            "'corr' = Pearson correlation, 'corr2' = Pearson r². Default: cod"
        ),
    )

    # Output options
    out_opts = parser.add_argument_group("Output Options")
    out_opts.add_argument(
        "-save_splits",
        action="store_true",
        help="Save R² for each split (creates large 4D file: n_splits × spatial)",
    )

    # Compute options
    comp_opts = parser.add_argument_group("Compute Options")
    comp_opts.add_argument(
        "-mask",
        type=str,
        default=None,
        metavar="FILE",
        help="Brain mask (optional, speeds up computation)",
    )
    comp_opts.add_argument(
        "-mask_threshold",
        type=float,
        default=0.5,
        metavar="THRESH",
        help="Mask threshold (default: 0.5). Voxels > threshold are included.",
    )
    comp_opts.add_argument(
        "-batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Voxels per batch for projection (auto-detected if not specified)",
    )
    comp_opts.add_argument(
        "-data_chunk_size",
        type=int,
        default=None,
        metavar="N",
        help="Voxels to load to GPU at once (auto-detected if not specified). For very large datasets that don't fit on GPU.",
    )
    comp_opts.add_argument(
        "-device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Compute device (auto-detected if not specified)",
    )

    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Print header
    print("=" * 70)
    print("3dXvalR2fast - Cross-Validated R² Computation")
    print("=" * 70)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Parse input files
    input_files = parse_input_files(args.input)
    print(f"Input:")
    if len(input_files) == 1:
        print(f"  • Single file: {input_files[0]}")
    else:
        print(f"  • Multiple runs: {len(input_files)} files")
        for f in input_files:
            print(f"    - {f}")
    print()

    # Load design matrix
    print(f"Design matrix: {args.matrix}")
    design_info = read_afni_design_matrix(args.matrix)
    print(f"  • Timepoints: {design_info['n_timepoints']}")
    print(f"  • Regressors: {design_info['n_regressors']}")
    print(f"  • Runs: {len(design_info.get('run_starts', []))}")
    print()

    # Extract metadata
    full_labels, stim_labels, stim_indices = extract_design_metadata(design_info)
    nuisance_indices = [i for i in range(len(full_labels)) if i not in stim_indices]

    print(f"Regressor breakdown:")
    print(f"  • Stimulus: {len(stim_indices)} columns")
    print(f"  • Nuisance: {len(nuisance_indices)} columns")
    print()

    # Validate we have multiple runs
    run_starts = design_info.get("run_starts", [])
    if not run_starts or len(run_starts) < 2:
        print("ERROR: Cross-validation requires multiple runs!")
        print("Design matrix must have RunStart metadata with at least 2 runs.")
        sys.exit(1)

    # Parse CV strategy
    try:
        if "." in args.cv_strategy:
            cv_strategy = float(args.cv_strategy)
        else:
            cv_strategy = int(args.cv_strategy)
    except ValueError:
        print(f"ERROR: Invalid cv_strategy '{args.cv_strategy}'")
        print("Must be float (e.g., 0.5) or int (e.g., 1)")
        sys.exit(1)

    # Generate CV splits
    print(f"Cross-validation:")
    print(f"  • Strategy: {cv_strategy}")
    print(f"  • Max permutations: {args.nperms}")

    cv_splits = generate_cv_splits(
        n_runs=len(run_starts),
        strategy=cv_strategy,
        n_perms=args.nperms,
    )
    print(f"  • Generated splits: {len(cv_splits)}")
    print()

    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Compute device: {device}")
    print()

    # Load data
    print("📂 Loading data...")
    if len(input_files) == 1:
        import nibabel as nib

        img = nib.load(input_files[0])
        data_np = img.get_fdata()  # type: ignore[attr-defined]
        volume_shape = data_np.shape[:3]
        affine = img.affine  # type: ignore[attr-defined]

        # Reshape to (n_voxels, n_timepoints)
        data_np = data_np.reshape(-1, data_np.shape[-1])
        data = torch.from_numpy(data_np).float()
    else:
        # Multiple runs - use existing loader
        data, actual_run_starts = load_and_concatenate_runs(
            input_files, torch.device("cpu")  # type: ignore[arg-type]
        )

        # Get spatial info from first file
        import nibabel as nib

        img = nib.load(input_files[0])
        volume_shape = img.shape[:3]  # type: ignore[attr-defined]
        affine = img.affine  # type: ignore[attr-defined]

    print(f"  • Shape: {data.shape} (voxels × timepoints)")
    print(f"  • Volume: {volume_shape}")
    print()

    # Apply mask if provided
    if args.mask:
        print(f"Applying mask: {args.mask}")
        from fastfuncsim.afni_io import load_afni_mask

        mask_volume = load_afni_mask(args.mask, threshold=args.mask_threshold)

        # Verify mask shape matches data
        if mask_volume.shape != volume_shape:
            print(f"ERROR: Mask shape {mask_volume.shape} doesn't match data shape {volume_shape}")
            sys.exit(1)

        mask = mask_volume.flatten()
        n_total_voxels = mask.size
        data = data[mask]
        print(f"  • Masked voxels: {data.shape[0]:,} / {n_total_voxels:,} ({100*data.shape[0]/n_total_voxels:.1f}%)")
        print()
    else:
        mask = None

    # Compute cross-validated R²
    print("🚀 Computing cross-validated R²...")
    print()

    xval_results = compute_xval_r2(
        data=data,
        design_matrix=design_info["matrix"],
        run_starts=run_starts,
        stim_indices=stim_indices,
        nuisance_indices=nuisance_indices,
        cv_splits=cv_splits,
        metric=args.metric,
        device=device,
        batch_size=args.batch_size,
        data_chunk_size=args.data_chunk_size,
        verbose=True,
    )

    print("✅ Cross-validation complete!")
    print()

    # Print summary statistics
    print("📊 Results Summary:")
    print(f"  • Median R²: {xval_results['r2_median'].mean():.4f} ± {xval_results['r2_median'].std():.4f}")
    print(f"  • Mean std across voxels: {xval_results['r2_std'].mean():.4f}")
    print(f"  • Min R²: {xval_results['r2_min'].min():.4f}")
    print(f"  • Max R²: {xval_results['r2_max'].max():.4f}")
    print()

    # Write outputs
    print("💾 Writing outputs...")

    import nibabel as nib

    # Helper to reshape and save
    def save_volume(data_1d, filename, description):
        """Save 1D data as NIfTI volume"""
        if mask is not None:
            # Unmask
            data_full = np.zeros(mask.shape, dtype=np.float32)
            data_full[mask] = data_1d.cpu().numpy()
            data_vol = data_full.reshape(volume_shape)
        else:
            data_vol = data_1d.cpu().numpy().reshape(volume_shape)

        img = nib.Nifti1Image(data_vol, affine)
        nib.save(img, filename)
        print(f"  • {description}: {filename}")

    # Save median, std, min, max
    save_volume(xval_results["r2_median"], f"{args.output}_median.nii.gz", "Median R²")
    save_volume(xval_results["r2_std"], f"{args.output}_std.nii.gz", "Std R²")
    save_volume(xval_results["r2_min"], f"{args.output}_min.nii.gz", "Min R²")
    save_volume(xval_results["r2_max"], f"{args.output}_max.nii.gz", "Max R²")

    # Optionally save all splits
    if args.save_splits:
        print()
        print("  Saving all splits (this may take a moment)...")
        r2_splits = xval_results["r2_splits"]  # (n_splits, n_voxels)

        if mask is not None:
            # Unmask each split
            r2_splits_full = np.zeros((len(cv_splits), *volume_shape), dtype=np.float32)
            for split_idx in range(len(cv_splits)):
                data_full = np.zeros(mask.shape, dtype=np.float32)
                data_full[mask] = r2_splits[split_idx].cpu().numpy()
                r2_splits_full[split_idx] = data_full.reshape(volume_shape)
        else:
            r2_splits_full = r2_splits.cpu().numpy().reshape(len(cv_splits), *volume_shape)

        # Save as 4D volume (splits as 4th dimension)
        img = nib.Nifti1Image(r2_splits_full.transpose(1, 2, 3, 0), affine)
        nib.save(img, f"{args.output}_splits.nii.gz")
        print(f"  • All splits: {args.output}_splits.nii.gz")

    print()
    print("=" * 70)
    print("✓ Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
