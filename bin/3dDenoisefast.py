#!/usr/bin/env python3
"""
3dDenoisefast - Fast cross-validated data-driven denoising using GPU acceleration

This tool implements adaptive denoising via noise pool PCA:
1. Identify noise pool voxels (low task R²) and criteria voxels (high task R²)
2. Extract PCs from noise pool as candidate nuisance regressors
3. Cross-validate to select optimal number of PCs that maximizes prediction
4. Train on denoised data but test on raw data to prevent overfitting

The key anti-overfitting strategy: we denoise training data but predict non-denoised
test data, ensuring we're improving signal recovery rather than just fitting noise removal.

Basic usage:
    3dDenoisefast -input run1.nii.gz run2.nii.gz run3.nii.gz \\
                  -onsets cond1.txt cond2.txt \\
                  -durations 2.0 5.0 \\
                  -tr 2.0 \\
                  -prefix subject01_denoised

For help:
    3dDenoisefast -help
"""

import argparse
import glob as glob_module
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

# Import fastfuncsim modules
try:
    from fastfuncsim.afni_io import (
        load_afni_mask,
        load_and_concatenate_runs,
    )
    from fastfuncsim.design_builder import (
        parse_afni_timing_file,
        create_design_matrix_from_onsets,
    )
    from fastfuncsim.denoise import (
        fit_denoising_model,
        DenoiseResults,
    )
    from fastfuncsim.glm_core import fit_glm_torch
    from fastfuncsim.utils import get_device, scale_to_percent_signal, gaussian_blur_3d
except ImportError as e:
    print(f"ERROR: Could not import fastfuncsim: {e}")
    print("Make sure fastfuncsim is installed: pip install -e .")
    sys.exit(1)


def parse_input_files(input_arg: Union[str, List[str]]) -> List[str]:
    """Parse input files (can be list from nargs='+' or single string)"""
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


def parse_durations(
    durations_arg: List[str],
    n_conditions: int,
    condition_labels: List[str],
) -> List[float]:
    """Parse durations argument."""
    if len(durations_arg) == 1:
        # Single duration for all conditions
        try:
            dur = float(durations_arg[0])
            return [dur] * n_conditions
        except ValueError:
            print(f"ERROR: Could not parse duration '{durations_arg[0]}' as float")
            sys.exit(1)
    elif len(durations_arg) == n_conditions:
        # One duration per condition
        try:
            return [float(d) for d in durations_arg]
        except ValueError as e:
            print(f"ERROR: Could not parse durations: {e}")
            sys.exit(1)
    else:
        print(
            f"ERROR: Number of durations ({len(durations_arg)}) must be 1 or match "
            f"number of conditions ({n_conditions})"
        )
        print(f"  Conditions: {condition_labels}")
        sys.exit(1)


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="3dDenoisefast - Fast GPU-accelerated cross-validated denoising",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,  # We handle -help ourselves
        epilog="""
Examples:
  # Basic denoising with automatic R² threshold
  3dDenoisefast -input run1.nii.gz run2.nii.gz run3.nii.gz \\
                -onsets cond1.txt cond2.txt \\
                -durations 2.0 \\
                -tr 2.0 \\
                -prefix subject01_denoised

  # Custom R² threshold for noise pool selection
  3dDenoisefast -input run*.nii.gz \\
                -onsets face.txt house.txt \\
                -durations 2.0 \\
                -tr 1.5 \\
                -r2_threshold 0.15 \\
                -prefix sub01_r2_015

  # With mask and more PCs
  3dDenoisefast -input run*.nii.gz \\
                -onsets stim.txt \\
                -durations 1.0 \\
                -tr 2.0 \\
                -mask brain_mask.nii.gz \\
                -max_pcs 30 \\
                -prefix masked_denoised

  # With motion nuisance regressors
  3dDenoisefast -input run*.nii.gz \\
                -onsets face.txt house.txt \\
                -durations 2.0 \\
                -tr 2.0 \\
                -ortvec motion_all.1D motion \\
                -prefix sub01_motion_denoised

Outputs:
  {prefix}_noise_pool_mask.nii.gz    - Noise pool voxels (low task R²)
  {prefix}_criteria_mask.nii.gz      - Criteria voxels (high task R²)
  {prefix}_initial_r2.nii.gz         - Initial R² before denoising
  {prefix}_xval_r2_by_npcs.npy       - CV R² for each number of PCs
  {prefix}_noise_pcs.pt              - Noise PCs for optimal denoising
  {prefix}_metadata.json             - Full metadata for reproducibility
  {prefix}_denoising_report.png      - Visualization of results

Workflow:
  1. Fit initial GLM to compute R² for each voxel
  2. Select noise pool (R² < threshold) and criteria voxels (R² >= threshold)
  3. Extract PCs from noise pool voxels per run
  4. Cross-validate: train on denoised data, test on raw data
  5. Select optimal number of PCs that maximizes CV R²
  6. Save results and optimal denoising parameters

Notes:
  - At least 2 runs required for cross-validation
  - Noise pool must have sufficient voxels (default: min 100)
  - Training on denoised but testing on raw prevents overfitting
  - Polynomial drift is always included (auto-determined or via -polort)
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

    # Denoising options
    denoise_opts = parser.add_argument_group("Denoising Options")
    denoise_opts.add_argument(
        "-r2_threshold",
        type=float,
        default=0.1,
        help="R² threshold for noise pool selection (default: 0.1). "
        "Voxels with R² < threshold are noise pool, >= threshold are criteria.",
    )
    denoise_opts.add_argument(
        "-max_pcs",
        type=int,
        default=20,
        help="Maximum number of PCs to test (default: 20)",
    )
    denoise_opts.add_argument(
        "-min_noise_voxels",
        type=int,
        default=100,
        help="Minimum voxels required in noise pool (default: 100)",
    )
    denoise_opts.add_argument(
        "-max_noise_fraction",
        type=float,
        default=0.5,
        help="Maximum fraction of voxels in noise pool (default: 0.5)",
    )
    denoise_opts.add_argument(
        "-variance_threshold",
        type=float,
        default=0.95,
        help="Cumulative variance threshold for PC extraction (default: 0.95)",
    )
    denoise_opts.add_argument(
        "-cv_metric",
        choices=["mean", "median"],
        default="median",
        help="Aggregation metric across CV folds (default: median)",
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
            "Additional nuisance regressors (can be repeated). "
            "FILE: text file with nuisance columns. "
            "LABEL: prefix for column names. "
            "Must span all runs concatenated. "
            "Example: -ortvec motion_all.1D motion"
        ),
    )
    proc_opts.add_argument(
        "-microtime_dt",
        type=float,
        default=0.1,
        help="Microtime resolution in seconds (default: 0.1)",
    )
    proc_opts.add_argument(
        "-canonical",
        type=str,
        default="spmg1",
        help="Canonical HRF mode: 'spmg1' (default) or 'glmsingle'",
    )
    proc_opts.add_argument(
        "-do_scale",
        action="store_true",
        help="Scale each voxel per run to mean=100 (percent signal change)",
    )
    proc_opts.add_argument(
        "-do_blur",
        type=float,
        metavar="FWHM",
        default=None,
        help="Apply 3D Gaussian spatial smoothing with FWHM in mm",
    )
    proc_opts.add_argument(
        "-device",
        type=str,
        help="Force device: 'cpu' or 'cuda' (default: auto-detect GPU)",
    )
    proc_opts.add_argument(
        "-keep_on_cpu",
        action="store_true",
        help="Load data to CPU and process in GPU chunks (for large datasets)",
    )
    proc_opts.add_argument(
        "-verbose",
        action="store_true",
        help="Print detailed progress information",
    )

    # Output options
    out_opts = parser.add_argument_group("Output Options")
    out_opts.add_argument(
        "-save_plots",
        action="store_true",
        help="Save denoising performance plots as PNG",
    )
    out_opts.add_argument(
        "-no_save_pcs",
        action="store_true",
        help="Don't save noise PCs (reduces output file size)",
    )

    # Help
    parser.add_argument("-help", action="store_true", help="Show this help message")

    return parser


def print_header(args):
    """Print program header"""
    print("=" * 70)
    print("3dDenoisefast - GPU-Accelerated Cross-Validated Denoising")
    print("=" * 70)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def save_denoising_results(
    results: DenoiseResults,
    output_prefix: str,
    volume_shape: tuple,
    affine: np.ndarray,
    voxel_mask: Optional[torch.Tensor] = None,
    save_plots: bool = False,
    save_pcs: bool = True,
):
    """
    Save denoising results to disk
    
    Parameters
    ----------
    results : DenoiseResults
        Denoising results
    output_prefix : str
        Output file prefix
    volume_shape : tuple
        Shape of 3D volume
    affine : np.ndarray
        Affine matrix for NIfTI files
    voxel_mask : torch.Tensor, optional
        Voxel mask (if brain mask was used)
    save_plots : bool
        Save visualization plots
    save_pcs : bool
        Save noise PCs
    
    Returns
    -------
    output_files : dict
        Dictionary of output file paths
    """
    output_files = {}
    
    # Helper to reshape flat data to volume
    def to_volume(flat_data):
        if voxel_mask is not None:
            vol = np.zeros(voxel_mask.shape[0], dtype=flat_data.dtype)
            vol[voxel_mask.cpu().numpy()] = flat_data
        else:
            vol = flat_data
        return vol.reshape(volume_shape)
    
    # 1. Noise pool mask
    noise_pool_vol = to_volume(results.noise_pool_mask.cpu().numpy().astype(np.float32))
    noise_pool_img = nib.Nifti1Image(noise_pool_vol, affine)
    noise_pool_path = f"{output_prefix}_noise_pool_mask.nii.gz"
    nib.save(noise_pool_img, noise_pool_path)
    output_files['noise_pool_mask'] = noise_pool_path
    
    # 2. Criteria mask
    criteria_vol = to_volume(results.criteria_mask.cpu().numpy().astype(np.float32))
    criteria_img = nib.Nifti1Image(criteria_vol, affine)
    criteria_path = f"{output_prefix}_criteria_mask.nii.gz"
    nib.save(criteria_img, criteria_path)
    output_files['criteria_mask'] = criteria_path
    
    # 3. Initial R²
    initial_r2_vol = to_volume(results.noise_pool_r2.cpu().numpy().astype(np.float32))
    initial_r2_img = nib.Nifti1Image(initial_r2_vol, affine)
    initial_r2_path = f"{output_prefix}_initial_r2.nii.gz"
    nib.save(initial_r2_img, initial_r2_path)
    output_files['initial_r2'] = initial_r2_path
    
    # 4. CV R² by number of PCs
    xval_r2_path = f"{output_prefix}_xval_r2_by_npcs.npy"
    np.save(xval_r2_path, results.xval_r2_by_n_components)
    output_files['xval_r2_by_npcs'] = xval_r2_path
    
    # 5. CV R² per fold
    xval_r2_folds_path = f"{output_prefix}_xval_r2_per_fold.npy"
    np.save(xval_r2_folds_path, results.xval_r2_per_fold)
    output_files['xval_r2_per_fold'] = xval_r2_folds_path
    
    # 6. Noise PCs (optional - can be large)
    if save_pcs:
        pcs_path = f"{output_prefix}_noise_pcs.pt"
        torch.save({
            'noise_pcs_per_run': results.noise_pcs_per_run,
            'optimal_n_components': results.optimal_n_components,
        }, pcs_path)
        output_files['noise_pcs'] = pcs_path
    
    # 7. Metadata
    metadata = {
        **results.metadata,
        'optimal_n_components': results.optimal_n_components,
        'baseline_r2': results.baseline_r2,
        'optimal_r2': results.optimal_r2,
        'improvement': results.improvement,
        'volume_shape': volume_shape,
    }
    
    metadata_path = f"{output_prefix}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    output_files['metadata'] = metadata_path
    
    # 8. Plots (optional)
    if save_plots:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # CV R² by number of PCs
        ax = axes[0, 0]
        ax.plot(results.xval_r2_by_n_components, 'o-', label='Mean')
        ax.plot(results.xval_r2_median_by_n_components, 's--', alpha=0.7, label='Median')
        ax.axvline(results.optimal_n_components, color='r', linestyle='--', alpha=0.5,
                   label=f'Optimal ({results.optimal_n_components} PCs)')
        ax.set_xlabel('Number of PCs')
        ax.set_ylabel('Cross-validated R²')
        ax.set_title('Denoising Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # R² per fold heatmap
        ax = axes[0, 1]
        im = ax.imshow(results.xval_r2_per_fold, aspect='auto', cmap='viridis')
        ax.set_xlabel('Number of PCs')
        ax.set_ylabel('CV Fold (run)')
        ax.set_title('R² per CV Fold')
        plt.colorbar(im, ax=ax, label='R²')
        
        # Initial R² distribution
        ax = axes[1, 0]
        r2_cpu = results.noise_pool_r2.cpu().numpy()
        ax.hist(r2_cpu, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(results.metadata['r2_threshold'], color='r', linestyle='--',
                   label=f"Threshold = {results.metadata['r2_threshold']:.2f}")
        ax.set_xlabel('Initial R²')
        ax.set_ylabel('Number of voxels')
        ax.set_title('R² Distribution (Noise Pool Selection)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary text
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
Denoising Results
{'='*40}

Voxel Selection:
  Noise pool: {results.metadata['n_noise_voxels']:,} voxels
  Criteria: {results.metadata['n_criteria_voxels']:,} voxels
  Threshold: R² < {results.metadata['r2_threshold']:.2f}

Cross-Validation:
  Strategy: Leave-one-run-out
  Runs: {results.metadata['n_runs']}
  Max PCs: {results.metadata['max_components']}

Performance:
  Baseline R²: {results.baseline_r2:.4f}
  Optimal R²: {results.optimal_r2:.4f}
  Improvement: {results.improvement:+.4f}
  Optimal PCs: {results.optimal_n_components}
        """
        ax.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plot_path = f"{output_prefix}_denoising_report.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        output_files['denoising_report'] = plot_path
    
    return output_files


def main():
    parser = create_parser()

    # Check for help
    if len(sys.argv) == 1 or "-help" in sys.argv or "--help" in sys.argv:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    print_header(args)

    # Parse and validate inputs
    input_files = parse_input_files(args.input)
    n_runs = len(input_files)

    if n_runs < 2:
        print("ERROR: At least 2 runs required for cross-validation")
        sys.exit(1)

    # Parse onset files
    onset_files = args.onsets
    n_conditions = len(onset_files)
    condition_labels = [Path(f).stem for f in onset_files]

    # Validate onset files
    for f in onset_files:
        if not Path(f).exists():
            print(f"ERROR: Onset file not found: {f}")
            sys.exit(1)

    # Parse durations
    durations = parse_durations(args.durations, n_conditions, condition_labels)
    print(f"  Durations: {durations}s")

    # Setup device
    if args.device:
        device = torch.device(args.device if args.device.lower() != "cpu" else "cpu")
    else:
        device = get_device()
    print(f"  Device: {device}")

    # ==========================================================================
    # Load data
    # ==========================================================================

    print()
    print("Loading data...")

    # Load mask if provided
    mask = None
    if args.mask:
        mask = load_afni_mask(args.mask)
        print(f"  Mask: {args.mask} ({mask.sum():,} voxels)")

    # Load first file for metadata
    first_img = nib.load(input_files[0])
    affine = np.array(first_img.affine) if hasattr(first_img, "affine") else np.eye(4)
    volume_shape = tuple(first_img.shape[:3]) if hasattr(first_img, "shape") else (0, 0, 0)
    voxel_sizes = tuple(np.abs(np.diag(affine)[:3]))

    # Determine memory strategy
    if hasattr(first_img, 'shape'):
        n_voxels_per_run = first_img.shape[0] * first_img.shape[1] * first_img.shape[2]
        n_timepoints_per_run = first_img.shape[3] if len(first_img.shape) > 3 else first_img.shape[-1]
    else:
        n_voxels_per_run = 10000
        n_timepoints_per_run = 200

    if mask is not None:
        n_voxels_per_run = int(mask.sum())

    total_timepoints = n_timepoints_per_run * n_runs
    data_size_gb = (n_voxels_per_run * total_timepoints * 4) / (1024**3)
    gpu_memory_threshold_gb = 4.0

    if args.keep_on_cpu:
        keep_on_cpu = True
        print(f"\n  Loading to CPU (user-specified)")
    elif device.type == "cuda" and data_size_gb > gpu_memory_threshold_gb:
        keep_on_cpu = True
        print(f"\n  ⚠️  Large dataset ({data_size_gb:.2f} GB)")
        print(f"     Loading to CPU and processing in GPU chunks")
    else:
        keep_on_cpu = False

    # Load and optionally blur data
    if args.do_blur is not None:
        print(f"\nApplying Gaussian blur (FWHM = {args.do_blur} mm)...")
        
        from tqdm import tqdm
        run_data_list = []
        run_starts = [0]
        current_timepoint = 0

        for run_idx, run_file in enumerate(tqdm(input_files, desc="  Loading & blurring", unit="run")):
            img = nib.load(run_file)
            data_4d = img.get_fdata(dtype=np.float32)

            if data_4d.ndim != 4:
                print(f"ERROR: Expected 4D data, got {data_4d.ndim}D")
                sys.exit(1)

            data_4d_blurred = gaussian_blur_3d(
                data_4d,
                fwhm_mm=args.do_blur,
                voxel_sizes=voxel_sizes,
                device=device,
                verbose=(run_idx == 0),
            )

            n_tps = data_4d_blurred.shape[3]
            data_2d = data_4d_blurred.reshape(-1, n_tps)

            if mask is not None:
                mask_flat = mask.flatten().astype(bool)
                data_2d = data_2d[mask_flat, :]

            if keep_on_cpu:
                data_2d = torch.from_numpy(data_2d).to(torch.float32)
            else:
                data_2d = torch.from_numpy(data_2d).to(device=device, dtype=torch.float32)

            run_data_list.append(data_2d)
            current_timepoint += n_tps
            if run_idx < len(input_files) - 1:
                run_starts.append(current_timepoint)

        data = torch.cat(run_data_list, dim=1)
        del run_data_list
        print(f"  ✓ Blurred and concatenated {n_runs} runs")
    else:
        # Standard loading
        data, run_starts = load_and_concatenate_runs(
            [Path(f) for f in input_files],
            device=device,
            keep_on_cpu=keep_on_cpu,
        )

        if mask is not None:
            mask_flat = mask.flatten().astype(bool)
            data = data[mask_flat, :]

    # Get TR
    if args.tr is None:
        zooms = first_img.header.get_zooms()
        if len(zooms) > 3 and zooms[3] > 0:
            args.tr = float(zooms[3])
            print(f"  TR (from header): {args.tr}s")
        else:
            print("ERROR: Could not determine TR from header. Please specify with -tr")
            sys.exit(1)
    else:
        print(f"  TR (specified): {args.tr}s")

    n_voxels, n_timepoints = data.shape

    # Optional scaling
    if args.do_scale:
        print()
        data, _, scale_info = scale_to_percent_signal(
            data=data,
            run_starts=run_starts,
            max_scale=200.0,
            verbose=True,
        )

    print(f"  Data shape: {data.shape} ({n_voxels:,} voxels × {n_timepoints} timepoints)")
    print(f"  Volume shape: {volume_shape}")
    print(f"  Runs: {n_runs} starting at {run_starts}")

    # ==========================================================================
    # Build design matrix
    # ==========================================================================

    print()
    print("Building design matrix...")

    # Parse onset files
    all_onsets = []
    for onset_file in onset_files:
        onsets_by_run = parse_afni_timing_file(onset_file)
        if len(onsets_by_run) != n_runs:
            print(f"ERROR: Onset file {onset_file} has {len(onsets_by_run)} runs, expected {n_runs}")
            sys.exit(1)
        all_onsets.append(onsets_by_run)

    # Build design matrix
    design_result = create_design_matrix_from_onsets(
        onsets=all_onsets,
        stim_durations=durations,
        tr=args.tr,
        n_timepoints=n_timepoints,
        run_starts=run_starts,
        polort=args.polort,
        ortvec_files=[(f, label) for f, label in args.ortvec] if args.ortvec else None,
        canonical_mode=args.canonical,
        microtime_dt=args.microtime_dt,
        device=device,
    )

    task_design = design_result['stimulus']
    nuisance = design_result['nuisance']

    print(f"  Task predictors: {task_design.shape[1]}")
    print(f"  Nuisance predictors: {nuisance.shape[1]}")
    print(f"  Condition labels: {condition_labels}")

    # ==========================================================================
    # Fit denoising model
    # ==========================================================================

    print()
    print("=" * 70)
    print("Fitting cross-validated denoising model...")
    print("=" * 70)

    results = fit_denoising_model(
        data=data,
        design_matrix=task_design,
        run_starts=run_starts,
        r2_threshold=args.r2_threshold,
        max_components=args.max_pcs,
        variance_threshold=args.variance_threshold,
        nuisance=nuisance,
        metric=args.cv_metric,
        min_noise_voxels=args.min_noise_voxels,
        max_noise_fraction=args.max_noise_fraction,
        device=device,
        verbose=args.verbose,
    )

    # ==========================================================================
    # Save outputs
    # ==========================================================================

    print()
    print("Saving outputs...")

    output_dir = Path(args.prefix).parent
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    voxel_mask = None
    if mask is not None:
        voxel_mask = torch.from_numpy(mask.flatten().astype(bool))

    output_files = save_denoising_results(
        results=results,
        output_prefix=args.prefix,
        volume_shape=volume_shape,
        affine=affine,
        voxel_mask=voxel_mask,
        save_plots=args.save_plots,
        save_pcs=not args.no_save_pcs,
    )

    print()
    print("=" * 70)
    print("📁 Output Files")
    print("=" * 70)
    for output_type, filepath in output_files.items():
        print(f"  {output_type}: {filepath}")
    print("=" * 70)

    # Print summary
    print()
    print("=" * 70)
    print("✅ 3dDenoisefast Complete!")
    print("=" * 70)
    print(f"  Noise pool: {results.metadata['n_noise_voxels']:,} voxels")
    print(f"  Criteria: {results.metadata['n_criteria_voxels']:,} voxels")
    print(f"  Baseline R²: {results.baseline_r2:.4f}")
    print(f"  Optimal R²: {results.optimal_r2:.4f}")
    print(f"  Improvement: {results.improvement:+.4f}")
    print(f"  Optimal PCs: {results.optimal_n_components}")
    print()
    print(f"🕐 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
