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
from typing import Optional, Union

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
    from fastfuncsim.denoise import (
        DenoiseResults,
        fit_denoising_model,
    )
    from fastfuncsim.design import convolve_hrf_microtime
    from fastfuncsim.design_builder import parse_afni_timing_file
    from fastfuncsim.glm_core import construct_polynomial_matrix
    from fastfuncsim.hrf import get_spmg1_hrf
    from fastfuncsim.hrf_selection import load_nuisance_file
    from fastfuncsim.utils import gaussian_blur_3d, get_device, scale_to_percent_signal, to_tensor
except ImportError as e:
    print(f"ERROR: Could not import fastfuncsim: {e}")
    print("Make sure fastfuncsim is installed: pip install -e .")
    sys.exit(1)


def parse_input_files(input_arg: Union[str, list[str]]) -> list[str]:
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
    durations_arg: list[str],
    n_conditions: int,
    condition_labels: list[str],
) -> list[float]:
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

  # With full diagnostic plots and model fit outputs
  3dDenoisefast -input run*.nii.gz \\
                -onsets face.txt house.txt \\
                -durations 2.0 \\
                -tr 2.0 \\
                -plots full \\
                -save_model_fit \\
                -prefix sub01_full_diagnostics

  # With mask and spatial PC weight maps
  3dDenoisefast -input run*.nii.gz \\
                -onsets stim.txt \\
                -durations 1.0 \\
                -tr 2.0 \\
                -mask brain_mask.nii.gz \\
                -max_pcs 30 \\
                -save_pcs both \\
                -prefix masked_denoised

  # With motion nuisance regressors
  3dDenoisefast -input run*.nii.gz \\
                -onsets face.txt house.txt \\
                -durations 2.0 \\
                -tr 2.0 \\
                -ortvec motion_all.1D motion \\
                -prefix sub01_motion_denoised

Outputs:
    Core outputs (always saved):
        {prefix}_noise_pool_mask.nii.gz       - Noise pool voxels (low task R²)
        {prefix}_criteria_mask.nii.gz         - Criteria voxels (high task R²)
        {prefix}_initial_r2.nii.gz            - Initial xval R² (task-only)
        {prefix}_xval_r2_optimal.nii.gz       - Xval R² at optimal PC count (criteria voxels)
        {prefix}_xval_r2_optimal_full.nii.gz  - Xval R² at optimal PC count (all voxels)
        {prefix}_xval_r2_optimal_per_fold.nii.gz - Per-fold xval R² at optimal PCs (4D)
        {prefix}_xval_r2_by_npcs.npy          - CV R² for each number of PCs
        {prefix}_metadata.json                - Full metadata for reproducibility

  With -save_pcs timecourse/both:
    {prefix}_noise_pcs.pt              - PC timecourses (.pt PyTorch file)

  With -save_pcs spatial/both:
    {prefix}_run01_pc_weights.nii.gz   - Spatial PC weights per run (4D NIfTI)

  With -plots yes/full:
    {prefix}_denoising_summary.png     - CV performance summary
    {prefix}_pc_diagnostics_PC01.png   - Per-PC timecourse plots (full mode)

    With -save_model_fit:
        {prefix}_initial_betas.nii.gz      - Initial model betas (4D)
        {prefix}_denoised_betas.nii.gz     - Denoised model betas (4D)

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
        default=0.05,
        help="R² threshold for noise pool selection (default: 0.05). "
        "Voxels with R² < threshold are noise pool, >= threshold are criteria.",
    )
    denoise_opts.add_argument(
        "-max_pcs",
        type=int,
        default=20,
        help="Maximum number of PCs to test (default: 20)",
    )
    denoise_opts.add_argument(
        "-pcstop",
        type=float,
        default=1.05,
        help="PC selection stopping threshold (default: 1.05, GLMdenoise-style). "
        ">=1: Stop when R² is within (pcstop-1)*100%% of max (e.g., 1.05 = 5%%). "
        "<0: Use exactly abs(pcstop) PCs (user override). "
        "=1: Pure argmax (pick maximum R²).",
    )
    denoise_opts.add_argument(
        "-pcR2cutoff",
        type=float,
        default=None,
        help="R² cutoff for PC selection (GLMdenoise default: 0.05). "
        "If set, only voxels with max R² > cutoff across any PC count are used "
        "to compute the selection curve. More robust to noisy voxels.",
    )
    denoise_opts.add_argument(
        "-brainthresh",
        nargs=2,
        type=float,
        metavar=("PERCENTILE", "FRACTION"),
        default=None,
        help="Signal intensity threshold for noise pool selection (GLMdenoise-style). "
        "PERCENTILE: percentile of mean intensity (e.g., 99). "
        "FRACTION: fraction of that percentile (e.g., 0.5). "
        "Voxels with mean intensity < PERCENTILE * FRACTION are excluded from noise pool. "
        "Applied BEFORE scaling. Example: -brainthresh 99 0.5",
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
        default=0.95,
        help="Maximum fraction of voxels in noise pool (default: 0.5)",
    )
    denoise_opts.add_argument(
        "-variance_threshold",
        type=float,
        default=0.95,
        help="Cumulative variance threshold for PC extraction (default: 0.95)",
    )
    denoise_opts.add_argument(
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
    denoise_opts.add_argument(
        "-n_perms",
        type=int,
        default=100,
        help="Max number of CV permutations for random splits (default: 100)",
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
        "-plots",
        type=str,
        choices=["no", "yes", "full"],
        default="no",
        help="Save diagnostic plots: 'no' (none), 'yes' (summary), 'full' (summary + per-PC plots)",
    )
    out_opts.add_argument(
        "-plot_ax",
        type=str,
        choices=["x", "y", "z"],
        default="x",
        help="Slice axis for PC spatial maps: 'x' (sagittal), 'y' (coronal), 'z' (axial)",
    )
    out_opts.add_argument(
        "-save_pcs",
        type=str,
        choices=["no", "timecourse", "spatial", "both"],
        default="timecourse",
        help="Save noise PCs: 'no', 'timecourse' (default: .pt file), 'spatial' (NIfTI weight maps), 'both'",
    )
    out_opts.add_argument(
        "-save_model_fit",
        action="store_true",
        help="Save initial and final (denoised) model fit outputs (betas, tstats) as NIfTI",
    )
    out_opts.add_argument(
        "-snr",
        action="store_true",
        help="Compute and save SNR (signal-to-noise ratio) outputs. "
        "Creates SNR volumes before/after denoising and scatter plot comparison. "
        "Computes both residual-based SNR and bootstrap-based SNR (if -numboots > 0).",
    )
    out_opts.add_argument(
        "-numboots",
        type=int,
        default=0,
        help="Number of bootstrap iterations for SE estimation (default: 0 = no bootstrapping). "
        "Recommended: 100-1000 for robust SE. Enables bootstrap-based SNR if -snr is also set.",
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
    print("3dDenoisefast - GPU-Accelerated Cross-Validated Denoising")
    print("=" * 70)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def save_denoising_results(
    results: DenoiseResults,
    output_prefix: str,
    volume_shape: tuple,
    affine: np.ndarray,
    run_starts: list[int],
    tr: float,
    voxel_mask: Optional[torch.Tensor] = None,
    plots_mode: str = "no",
    slice_axis: str = "x",
    save_pcs_mode: str = "timecourse",
    condition_labels: Optional[list[str]] = None,
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
    run_starts : list of int
        Starting timepoint for each run
    tr : float
        Repetition time in seconds
    voxel_mask : torch.Tensor, optional
        Voxel mask (if brain mask was used)
    plots_mode : str
        'no', 'yes' (summary only), or 'full' (summary + per-PC)
    save_pcs_mode : str
        'no', 'timecourse', 'spatial', or 'both'
    condition_labels : list of str, optional
        Labels for task conditions

    Returns
    -------
    output_files : dict
        Dictionary of output file paths
    """
    output_files = {}
    voxel_mask_np = voxel_mask.cpu().numpy() if voxel_mask is not None else None

    # Note: Results already have extreme R² voxels excluded via valid_voxel_mask
    # All results tensors are in the same space as the input data (no reallocation was done)
    # So we can use voxel_mask directly without modification

    # Helper to reshape flat data to volume
    def to_volume(flat_data):
        if voxel_mask_np is not None:
            vol = np.zeros(voxel_mask_np.shape[0], dtype=flat_data.dtype)
            vol[voxel_mask_np] = flat_data
        else:
            vol = flat_data
        return vol.reshape(volume_shape)

    # 1. Noise pool mask
    noise_pool_vol = to_volume(results.noise_pool_mask.cpu().numpy().astype(np.float32))
    noise_pool_img = nib.Nifti1Image(noise_pool_vol, affine)
    noise_pool_path = f"{output_prefix}_noise_pool_mask.nii.gz"
    nib.save(noise_pool_img, noise_pool_path)
    output_files["noise_pool_mask"] = noise_pool_path

    # 2. Criteria mask
    criteria_vol = to_volume(results.criteria_mask.cpu().numpy().astype(np.float32))
    criteria_img = nib.Nifti1Image(criteria_vol, affine)
    criteria_path = f"{output_prefix}_criteria_mask.nii.gz"
    nib.save(criteria_img, criteria_path)
    output_files["criteria_mask"] = criteria_path

    # 3. Initial R²
    initial_r2_vol = to_volume(results.noise_pool_r2.cpu().numpy().astype(np.float32))
    initial_r2_img = nib.Nifti1Image(initial_r2_vol, affine)
    initial_r2_path = f"{output_prefix}_initial_r2.nii.gz"
    nib.save(initial_r2_img, initial_r2_path)
    output_files["initial_r2"] = initial_r2_path

    # 3b. Xval R² at optimal PC count (criteria voxels only)
    if results.xval_r2_optimal is not None:
        xval_opt_vol = to_volume(results.xval_r2_optimal.cpu().numpy().astype(np.float32))
        xval_opt_img = nib.Nifti1Image(xval_opt_vol, affine)
        xval_opt_path = f"{output_prefix}_xval_r2_optimal.nii.gz"
        nib.save(xval_opt_img, xval_opt_path)
        output_files["xval_r2_optimal"] = xval_opt_path

    # 3c. Xval R² at optimal PC count (all voxels)
    if results.xval_r2_optimal_full is not None:
        xval_opt_full_vol = to_volume(results.xval_r2_optimal_full.cpu().numpy().astype(np.float32))
        xval_opt_full_img = nib.Nifti1Image(xval_opt_full_vol, affine)
        xval_opt_full_path = f"{output_prefix}_xval_r2_optimal_full.nii.gz"
        nib.save(xval_opt_full_img, xval_opt_full_path)
        output_files["xval_r2_optimal_full"] = xval_opt_full_path

    # 3d. Per-fold xval R² at optimal PCs (4D)
    if results.xval_r2_optimal_per_fold is not None:
        fold_vols = []
        for fold_idx in range(results.xval_r2_optimal_per_fold.shape[0]):
            fold_vol = to_volume(results.xval_r2_optimal_per_fold[fold_idx].astype(np.float32))
            fold_vols.append(fold_vol)

        fold_4d = np.stack(fold_vols, axis=-1)
        fold_img = nib.Nifti1Image(fold_4d, affine)
        fold_path = f"{output_prefix}_xval_r2_optimal_per_fold.nii.gz"
        nib.save(fold_img, fold_path)
        output_files["xval_r2_optimal_per_fold"] = fold_path

    # 4. CV R² arrays
    xval_r2_path = f"{output_prefix}_xval_r2_by_npcs.npy"
    np.save(xval_r2_path, results.xval_r2_by_n_components)
    output_files["xval_r2_by_npcs"] = xval_r2_path

    xval_r2_folds_path = f"{output_prefix}_xval_r2_per_fold.npy"
    np.save(xval_r2_folds_path, results.xval_r2_per_fold)
    output_files["xval_r2_per_fold"] = xval_r2_folds_path

    # 5. Metadata
    metadata = {
        **results.metadata,
        "optimal_n_components": results.optimal_n_components,
        "baseline_r2": results.baseline_r2,
        "optimal_r2": results.optimal_r2,
        "improvement": results.improvement,
        "volume_shape": list(volume_shape),
        "tr": tr,
        "run_starts": run_starts,
    }
    if condition_labels:
        metadata["condition_labels"] = condition_labels

    metadata_path = f"{output_prefix}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    output_files["metadata"] = metadata_path

    # 6. Noise PCs (based on save_pcs_mode)
    if save_pcs_mode in ["timecourse", "both"]:
        pcs_path = f"{output_prefix}_noise_pcs.pt"
        torch.save(
            {
                "noise_pcs_per_run": results.noise_pcs_per_run,
                "optimal_n_components": results.optimal_n_components,
                "run_starts": run_starts,
            },
            pcs_path,
        )
        output_files["noise_pcs_timecourse"] = pcs_path

    if save_pcs_mode in ["spatial", "both"]:
        # Save PC spatial weights as NIfTI files (per run, per PC)
        if results.pc_loadings_per_run is not None:
            # Helper to reshape noise pool loadings to full volume
            noise_pool_np = results.noise_pool_mask.cpu().numpy()

            def loadings_to_volume(loadings_flat):
                """Map noise pool loadings back to full volume (zeros outside noise pool)"""
                if voxel_mask_np is not None:
                    # Two-level mask: voxel_mask (full volume) and noise_pool (within masked voxels)
                    # Map noise_pool indices into full-volume indices via voxel_mask
                    brain_indices = np.where(voxel_mask_np)[0]
                    noise_pool_indices = brain_indices[noise_pool_np]
                    vol = np.zeros(np.prod(volume_shape), dtype=loadings_flat.dtype)
                    vol[noise_pool_indices] = loadings_flat
                else:
                    # No brain mask, noise_pool is directly in volume space
                    vol = np.zeros(np.prod(volume_shape), dtype=loadings_flat.dtype)
                    vol[noise_pool_np] = loadings_flat
                return vol.reshape(volume_shape)

            n_runs = len(results.pc_loadings_per_run)
            for run_idx, loadings in enumerate(results.pc_loadings_per_run):
                loadings_np = loadings.cpu().numpy() if torch.is_tensor(loadings) else loadings
                n_pcs = loadings_np.shape[1]

                # Save each PC as a separate volume (or combine into 4D)
                pc_vols = []
                for pc_idx in range(
                    min(n_pcs, results.optimal_n_components + 3)
                ):  # Save optimal + a few more
                    pc_vol = loadings_to_volume(loadings_np[:, pc_idx])
                    pc_vols.append(pc_vol)

                # Stack into 4D and save
                pc_4d = np.stack(pc_vols, axis=-1)
                pc_img = nib.Nifti1Image(pc_4d, affine)
                pc_path = f"{output_prefix}_run{run_idx + 1:02d}_pc_weights.nii.gz"
                nib.save(pc_img, pc_path)
                output_files[f"run{run_idx + 1}_pc_weights"] = pc_path

            print(f"  Saved PC spatial weights for {n_runs} runs")
        else:
            print("  Warning: PC loadings not available (run with return_loadings=True)")

    # 6b. Save selected PCs as text files (one per run)
    # These are the PC timecourses for the optimal number of components
    n_runs = len(results.noise_pcs_per_run)
    for run_idx, pcs in enumerate(results.noise_pcs_per_run):
        pcs_np = pcs.cpu().numpy() if torch.is_tensor(pcs) else pcs
        # Take only the selected (optimal) number of PCs
        n_selected = min(results.optimal_n_components, pcs_np.shape[1])
        selected_pcs = pcs_np[:, :n_selected]

        pc_txt_path = f"{output_prefix}_run{run_idx + 1:02d}_selected_PCs.txt"
        # Save with header
        with open(pc_txt_path, "w") as f:
            f.write(f"# Selected noise PCs for run {run_idx + 1}\n")
            f.write(f"# n_components: {n_selected}\n")
            f.write(f"# Shape: {selected_pcs.shape[0]} timepoints x {selected_pcs.shape[1]} PCs\n")
            f.write(f"# Columns: PC1, PC2, ..., PC{n_selected}\n")
            np.savetxt(f, selected_pcs, fmt="%.6f", delimiter="\t")
        output_files[f"run{run_idx + 1}_selected_pcs_txt"] = pc_txt_path

    print(
        f"  Saved selected PCs ({results.optimal_n_components} PCs) as text files for {n_runs} runs"
    )

    # 7. Plots (based on plots_mode)
    if plots_mode in ["yes", "full"]:
        try:
            from fastfuncsim.visualization import plot_denoising_summary, plot_denoising_pcs

            # Summary plot
            r2_cpu = results.noise_pool_r2.cpu().numpy()
            summary_fig = plot_denoising_summary(
                xval_r2_by_n_components=results.xval_r2_by_n_components,
                xval_r2_per_fold=results.xval_r2_per_fold,
                optimal_n_components=results.optimal_n_components,
                initial_r2_distribution=r2_cpu,
                r2_threshold=results.metadata["r2_threshold"],
                n_noise_voxels=results.metadata["n_noise_voxels"],
                n_criteria_voxels=results.metadata["n_criteria_voxels"],
                output_path=f"{output_prefix}_denoising_summary.png",
            )
            output_files["denoising_summary_plot"] = f"{output_prefix}_denoising_summary.png"
            import matplotlib.pyplot as plt

            plt.close(summary_fig)

            # Per-PC plots (only for "full" mode)
            if plots_mode == "full":
                # Convert PC tensors to CPU for plotting
                pcs_cpu = (
                    [pc.cpu() for pc in results.noise_pcs_per_run]
                    if results.noise_pcs_per_run
                    else None
                )
                loadings_cpu = (
                    [ld.cpu() for ld in results.pc_loadings_per_run]
                    if results.pc_loadings_per_run
                    else None
                )

                # Create combined mask: voxel_mask (brain) AND noise_pool_mask (low R²)
                # This maps noise pool indices to full volume space
                noise_pool_mask_np = results.noise_pool_mask.cpu().numpy()

                pc_figs = plot_denoising_pcs(
                    noise_pcs_per_run=pcs_cpu,
                    run_starts=run_starts,
                    pc_weights_per_run=loadings_cpu,
                    volume_shape=volume_shape,
                    voxel_mask=voxel_mask_np,
                    noise_pool_mask=noise_pool_mask_np,
                    n_pcs_to_show=results.metadata.get("max_components", 0),
                    n_slices=5,
                    slice_axis=slice_axis,
                    tr=tr,
                    optimal_n_pcs=results.optimal_n_components,
                    output_prefix=f"{output_prefix}_pc_diagnostics",
                )
                output_files["pc_diagnostic_plots"] = f"{output_prefix}_pc_diagnostics_PC*.png"
                for fig in pc_figs:
                    plt.close(fig)

        except ImportError as e:
            print(f"  Warning: Could not import visualization module: {e}")
        except Exception as e:
            print(f"  Warning: Error creating plots: {e}")

    return output_files


def compute_bootstrap_se(
    data: torch.Tensor,
    design: torch.Tensor,
    n_task: int,
    n_boots: int = 100,
    chunk_size: int = 5000,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute bootstrap standard errors for beta coefficients.

    Uses residual bootstrap: resample residuals, add to fitted values, refit.

    Parameters
    ----------
    data : torch.Tensor
        (n_voxels, n_timepoints) fMRI data
    design : torch.Tensor
        (n_timepoints, n_regressors) full design matrix
    n_task : int
        Number of task regressors (first n columns)
    n_boots : int
        Number of bootstrap iterations
    chunk_size : int
        Voxels per batch
    device : torch.device
        Compute device
    verbose : bool
        Print progress

    Returns
    -------
    bootstrap_se : np.ndarray
        (n_voxels, n_task) standard error for each task beta
    """
    if device is None:
        device = get_device()

    n_voxels, n_timepoints = data.shape
    n_regressors = design.shape[1]

    # Storage for bootstrap betas
    boot_betas = np.zeros((n_boots, n_voxels, n_task), dtype=np.float32)

    # Compute original fit once
    design_gpu = design.to(device)
    XtX_inv = torch.linalg.inv(
        design_gpu.T @ design_gpu + 1e-6 * torch.eye(n_regressors, device=device)
    )

    if verbose:
        from tqdm import tqdm

        boot_iter = tqdm(range(n_boots), desc="  Bootstrap iterations")
    else:
        boot_iter = range(n_boots)

    for boot_idx in boot_iter:
        # Resample timepoints with replacement
        resample_idx = torch.randint(0, n_timepoints, (n_timepoints,), device=device)

        design_boot = design_gpu[resample_idx, :]
        XtX_inv_boot = torch.linalg.inv(
            design_boot.T @ design_boot + 1e-6 * torch.eye(n_regressors, device=device)
        )

        # Process in chunks
        for chunk_start in range(0, n_voxels, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_voxels)

            data_chunk = data[chunk_start:chunk_end, :].to(device)
            data_boot = data_chunk[:, resample_idx]

            # OLS fit
            betas_boot = (XtX_inv_boot @ design_boot.T @ data_boot.T).T  # (chunk, n_regressors)
            boot_betas[boot_idx, chunk_start:chunk_end, :] = betas_boot[:, :n_task].cpu().numpy()

    # Compute SE as std across bootstrap samples
    bootstrap_se = np.std(boot_betas, axis=0)  # (n_voxels, n_task)

    return bootstrap_se


def compute_snr(
    betas: np.ndarray,
    residual_std: Optional[np.ndarray] = None,
    bootstrap_se: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute SNR metrics from betas and noise estimates.

    Parameters
    ----------
    betas : np.ndarray
        (n_voxels, n_task) beta coefficients
    residual_std : np.ndarray, optional
        (n_voxels,) residual standard deviation
    bootstrap_se : np.ndarray, optional
        (n_voxels, n_task) bootstrap standard errors

    Returns
    -------
    snr_dict : dict
        'snr_residual': (n_voxels,) max|beta| / residual_std
        'snr_bootstrap': (n_voxels,) max(|beta| / bootstrap_se)
    """
    result = {}

    # Max absolute beta across conditions (the "signal")
    max_abs_beta = np.max(np.abs(betas), axis=1)  # (n_voxels,)

    # Residual-based SNR: signal / noise_floor
    if residual_std is not None:
        snr_residual = max_abs_beta / (residual_std + 1e-10)
        result["snr_residual"] = snr_residual

    # Bootstrap-based SNR: max of per-condition SNR
    if bootstrap_se is not None:
        # Per-condition SNR
        per_cond_snr = np.abs(betas) / (bootstrap_se + 1e-10)  # (n_voxels, n_task)
        snr_bootstrap = np.max(per_cond_snr, axis=1)  # (n_voxels,)
        result["snr_bootstrap"] = snr_bootstrap

    return result


def save_snr_outputs(
    snr_initial: dict,
    snr_denoised: dict,
    output_prefix: str,
    volume_shape: tuple,
    affine: np.ndarray,
    voxel_mask: Optional[torch.Tensor] = None,
    create_plots: bool = True,
) -> dict:
    """
    Save SNR volumes and create before/after comparison plots.
    """
    output_files = {}
    voxel_mask_np = voxel_mask.cpu().numpy() if voxel_mask is not None else None

    def to_volume(flat_data):
        if voxel_mask_np is not None:
            vol = np.zeros(voxel_mask_np.shape[0], dtype=flat_data.dtype)
            vol[voxel_mask_np] = flat_data
        else:
            vol = flat_data
        return vol.reshape(volume_shape)

    # Save residual-based SNR volumes
    if "snr_residual" in snr_initial:
        snr_vol = to_volume(snr_initial["snr_residual"].astype(np.float32))
        snr_img = nib.Nifti1Image(snr_vol, affine)
        snr_path = f"{output_prefix}_snr_residual_initial.nii.gz"
        nib.save(snr_img, snr_path)
        output_files["snr_residual_initial"] = snr_path

    if "snr_residual" in snr_denoised:
        snr_vol = to_volume(snr_denoised["snr_residual"].astype(np.float32))
        snr_img = nib.Nifti1Image(snr_vol, affine)
        snr_path = f"{output_prefix}_snr_residual_denoised.nii.gz"
        nib.save(snr_img, snr_path)
        output_files["snr_residual_denoised"] = snr_path

    # Save bootstrap-based SNR volumes
    if "snr_bootstrap" in snr_initial:
        snr_vol = to_volume(snr_initial["snr_bootstrap"].astype(np.float32))
        snr_img = nib.Nifti1Image(snr_vol, affine)
        snr_path = f"{output_prefix}_snr_bootstrap_initial.nii.gz"
        nib.save(snr_img, snr_path)
        output_files["snr_bootstrap_initial"] = snr_path

    if "snr_bootstrap" in snr_denoised:
        snr_vol = to_volume(snr_denoised["snr_bootstrap"].astype(np.float32))
        snr_img = nib.Nifti1Image(snr_vol, affine)
        snr_path = f"{output_prefix}_snr_bootstrap_denoised.nii.gz"
        nib.save(snr_img, snr_path)
        output_files["snr_bootstrap_denoised"] = snr_path

    # Create scatter plots
    if create_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Residual-based SNR scatter
            if "snr_residual" in snr_initial and "snr_residual" in snr_denoised:
                ax = axes[0]
                x = snr_initial["snr_residual"]
                y = snr_denoised["snr_residual"]

                # Subsample for plotting if too many points
                if len(x) > 10000:
                    idx = np.random.choice(len(x), 10000, replace=False)
                    x_plot, y_plot = x[idx], y[idx]
                else:
                    x_plot, y_plot = x, y

                ax.scatter(x_plot, y_plot, alpha=0.3, s=1, c="steelblue")
                max_val = max(np.percentile(x, 99), np.percentile(y, 99))
                ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="unity")
                ax.set_xlabel("SNR (initial)")
                ax.set_ylabel("SNR (denoised)")
                ax.set_title(f"Residual-based SNR\nMean: {x.mean():.2f} → {y.mean():.2f}")
                ax.set_xlim(0, max_val)
                ax.set_ylim(0, max_val)
                ax.legend()

            # Bootstrap-based SNR scatter
            if "snr_bootstrap" in snr_initial and "snr_bootstrap" in snr_denoised:
                ax = axes[1]
                x = snr_initial["snr_bootstrap"]
                y = snr_denoised["snr_bootstrap"]

                if len(x) > 10000:
                    idx = np.random.choice(len(x), 10000, replace=False)
                    x_plot, y_plot = x[idx], y[idx]
                else:
                    x_plot, y_plot = x, y

                ax.scatter(x_plot, y_plot, alpha=0.3, s=1, c="darkorange")
                max_val = max(np.percentile(x, 99), np.percentile(y, 99))
                ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="unity")
                ax.set_xlabel("SNR (initial)")
                ax.set_ylabel("SNR (denoised)")
                ax.set_title(f"Bootstrap-based SNR\nMean: {x.mean():.2f} → {y.mean():.2f}")
                ax.set_xlim(0, max_val)
                ax.set_ylim(0, max_val)
                ax.legend()
            else:
                axes[1].text(
                    0.5,
                    0.5,
                    "Bootstrap SNR\nnot computed\n(use -numboots)",
                    ha="center",
                    va="center",
                    transform=axes[1].transAxes,
                )
                axes[1].set_title("Bootstrap-based SNR")

            plt.tight_layout()
            plot_path = f"{output_prefix}_snr_comparison.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            output_files["snr_plot"] = plot_path

        except Exception as e:
            print(f"  Warning: Could not create SNR plots: {e}")

    return output_files


def save_model_fit_outputs(
    results,  # GLMResults
    output_prefix: str,
    volume_shape: tuple,
    affine: np.ndarray,
    model_type: str,  # "initial" or "denoised"
    condition_labels: Optional[list[str]] = None,
    voxel_mask: Optional[torch.Tensor] = None,
    n_timepoints: Optional[int] = None,
    n_regressors: Optional[int] = None,
    bootstrap_se: Optional[np.ndarray] = None,
):
    """
    Save GLM model fit outputs (betas, tstats) as NIfTI files with AFNI labeling

    AFNI-style output: betas and tstats are interleaved in a single 4D file
    with sub-bricks ordered as: [beta1, tstat1, beta2, tstat2, ...]

    Uses 3drefit to add proper AFNI sub-brick labels and DOF.

    Parameters
    ----------
    results : GLMResults
        GLM results from fit_glm
    output_prefix : str
        Output file prefix
    volume_shape : tuple
        Shape of 3D volume
    affine : np.ndarray
        Affine matrix for NIfTI files
    model_type : str
        "initial" or "denoised"
    condition_labels : list of str, optional
        Labels for task conditions
    voxel_mask : torch.Tensor, optional
        Voxel mask (if brain mask was used)
    n_timepoints : int, optional
        Number of timepoints (for DOF calculation)
    n_regressors : int, optional
        Total number of regressors in model (for DOF calculation)
    bootstrap_se : np.ndarray, optional
        Bootstrap standard errors (n_voxels, n_task) if available

    Returns
    -------
    output_files : dict
        Dictionary of output file paths
    """
    import subprocess
    import shutil

    output_files = {}
    voxel_mask_np = voxel_mask.cpu().numpy() if voxel_mask is not None else None

    def to_volume(flat_data):
        if voxel_mask_np is not None:
            vol = np.zeros(voxel_mask_np.shape[0], dtype=flat_data.dtype)
            vol[voxel_mask_np] = flat_data
        else:
            vol = flat_data
        return vol.reshape(volume_shape)

    # Get number of task regressors (first n columns, rest are nuisance)
    betas = results.betas.cpu().numpy() if torch.is_tensor(results.betas) else results.betas
    n_total_regs = betas.shape[1]
    n_task = len(condition_labels) if condition_labels else n_total_regs

    # Get tstats if available
    has_tstats = results.tstats is not None
    if has_tstats:
        tstats = results.tstats.cpu().numpy() if torch.is_tensor(results.tstats) else results.tstats

    # Calculate DOF for t-statistics
    dof = None
    if n_timepoints is not None and n_regressors is not None:
        dof = n_timepoints - n_regressors

    # Build AFNI-style bucket file: interleaved betas and tstats
    # Sub-brick order: [beta1, tstat1, beta2, tstat2, ...]
    bucket_vols = []
    sub_brick_labels = []
    sub_brick_types = []  # 'coef' or 'tstat' for 3drefit

    for reg_idx in range(n_task):
        label = condition_labels[reg_idx] if condition_labels else f"reg{reg_idx}"

        # Add beta
        beta_vol = to_volume(betas[:, reg_idx].astype(np.float32))
        bucket_vols.append(beta_vol)
        sub_brick_labels.append(f"{label}#0_Coef")
        sub_brick_types.append("coef")

        # Add tstat (if available)
        if has_tstats:
            tstat_vol = to_volume(tstats[:, reg_idx].astype(np.float32))
            bucket_vols.append(tstat_vol)
            sub_brick_labels.append(f"{label}#0_Tstat")
            sub_brick_types.append("tstat")

    # Add bootstrap SE sub-bricks if available
    if bootstrap_se is not None:
        for reg_idx in range(n_task):
            label = condition_labels[reg_idx] if condition_labels else f"reg{reg_idx}"
            se_vol = to_volume(bootstrap_se[:, reg_idx].astype(np.float32))
            bucket_vols.append(se_vol)
            sub_brick_labels.append(f"{label}#0_SE")
            sub_brick_types.append("se")

    # Stack into 4D
    bucket_4d = np.stack(bucket_vols, axis=-1)
    bucket_img = nib.Nifti1Image(bucket_4d, affine)
    bucket_path = f"{output_prefix}_{model_type}_bucket.nii.gz"
    nib.save(bucket_img, bucket_path)
    output_files[f"{model_type}_bucket"] = bucket_path

    # Use 3drefit to add proper AFNI labels and DOF
    has_3drefit = shutil.which("3drefit") is not None
    if has_3drefit:
        try:
            refit_cmd = ["3drefit"]

            # Add sub-brick labels
            for i, label in enumerate(sub_brick_labels):
                refit_cmd.extend(["-sublabel", str(i), label])

            # Add DOF for t-stat sub-bricks
            if dof is not None:
                for i, sbtype in enumerate(sub_brick_types):
                    if sbtype == "tstat":
                        refit_cmd.extend(["-substatpar", str(i), "fitt", str(dof)])

            refit_cmd.append(bucket_path)

            subprocess.run(refit_cmd, check=True, capture_output=True)
            print(f"  ✓ Applied AFNI labels to {bucket_path}")
        except subprocess.CalledProcessError as e:
            print(f"  Warning: 3drefit failed: {e}")
    else:
        # Save sub-brick labels as text file fallback
        labels_path = f"{output_prefix}_{model_type}_labels.txt"
        with open(labels_path, "w") as f:
            for i, label in enumerate(sub_brick_labels):
                f.write(f"{i}\t{label}\n")
        output_files[f"{model_type}_labels"] = labels_path

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

    # Parse CV strategy
    cv_strategy = parse_cv_strategy(args.cv_strategy)
    if args.verbose:
        print(f"  CV strategy: {cv_strategy}")

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
    if hasattr(first_img, "shape"):
        n_voxels_per_run = first_img.shape[0] * first_img.shape[1] * first_img.shape[2]
        n_timepoints_per_run = (
            first_img.shape[3] if len(first_img.shape) > 3 else first_img.shape[-1]
        )
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
        print("\n  Loading to CPU (user-specified)")
    elif device.type == "cuda" and data_size_gb > gpu_memory_threshold_gb:
        keep_on_cpu = True
        print(f"\n  ⚠️  Large dataset ({data_size_gb:.2f} GB)")
        print("     Loading to CPU and processing in GPU chunks")
    else:
        keep_on_cpu = False

    # Load and optionally blur data
    if args.do_blur is not None:
        print(f"\nApplying Gaussian blur (FWHM = {args.do_blur} mm)...")

        from tqdm import tqdm

        run_data_list = []
        run_starts = [0]
        current_timepoint = 0

        for run_idx, run_file in enumerate(
            tqdm(input_files, desc="  Loading & blurring", unit="run")
        ):
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
        mask_flat = mask.flatten().astype(bool) if mask is not None else None
        data, run_starts = load_and_concatenate_runs(
            [Path(f) for f in input_files],
            device=device,
            keep_on_cpu=keep_on_cpu,
            mask_flat=mask_flat,
        )

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

    # Compute brainthresh intensity mask BEFORE scaling
    # This excludes low-intensity voxels from the noise pool
    brainthresh_mask = None
    if args.brainthresh is not None:
        percentile, fraction = args.brainthresh
        print()
        print(f"Computing intensity threshold (brainthresh={percentile}, {fraction})...")

        # Compute mean intensity per voxel (across time)
        mean_intensity = data.mean(dim=1)  # Shape: (n_voxels,)

        # Get the percentile value
        percentile_value = torch.quantile(mean_intensity, percentile / 100.0)
        threshold = percentile_value * fraction

        # Create mask: True for voxels ABOVE threshold (valid voxels)
        brainthresh_mask = mean_intensity > threshold
        n_above = brainthresh_mask.sum().item()

        print(f"  {percentile:.0f}th percentile intensity: {percentile_value:.2f}")
        print(f"  Threshold ({fraction:.2f} × {percentile_value:.2f}): {threshold:.2f}")
        print(
            f"  Voxels above threshold: {n_above:,} of {n_voxels:,} ({n_above / n_voxels * 100:.1f}%)"
        )

    # Filter out voxels with zero/low variance in ANY run (edge artifacts)
    # This prevents extreme negative R² values from runs with all-zero data
    print()
    print("Filtering voxels with invalid data in any run...")
    valid_per_run_mask = torch.ones(n_voxels, dtype=torch.bool, device=data.device)

    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
        run_data = data[:, start_tp:end_tp]

        # Check for runs with zero variance (constant or all-zero)
        run_std = run_data.std(dim=1)
        run_valid = run_std > 1e-6  # Require non-zero variance

        valid_per_run_mask &= run_valid

    n_valid = valid_per_run_mask.sum().item()
    n_invalid = n_voxels - n_valid

    if n_invalid > 0:
        print(f"  Removed {n_invalid:,} voxels with zero/constant values in any run")
        print(f"  Valid voxels: {n_valid:,} ({n_valid / n_voxels * 100:.1f}%)")

        # Combine with brainthresh mask if it exists
        if brainthresh_mask is not None:
            brainthresh_mask = brainthresh_mask & valid_per_run_mask
        else:
            brainthresh_mask = valid_per_run_mask

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

    # Design matrix structure:
    # - TASK regressors: Shared across all runs (e.g., ring_01, ring_02, ...)
    #   Shape: (n_total_timepoints, n_task_predictors)
    #   No padding needed - same columns used by all runs
    #
    # - NUISANCE regressors: Run-specific (polynomial drift per run)
    #   Stored as list: nuisance_per_run[i] = (n_timepoints_run_i, n_nuisance_cols)
    #   Column padding needed: All runs padded to max # of nuisance columns
    #
    # Total model: Y = X_task @ beta_task + X_nuisance @ beta_nuisance + error
    # Columns: n_task (shared) + n_nuisance_padded (run-specific, column-padded)

    print()
    print("Building design matrix...")

    # Parse onset files
    all_onsets = []
    for onset_file in onset_files:
        onsets_by_run = parse_afni_timing_file(onset_file)
        if len(onsets_by_run) != n_runs:
            print(
                f"ERROR: Onset file {onset_file} has {len(onsets_by_run)} runs, expected {n_runs}"
            )
            sys.exit(1)
        all_onsets.append(onsets_by_run)

    # Build onset matrix at microtime resolution
    bins_per_tr = int(np.round(args.tr / args.microtime_dt))
    n_microtime = n_timepoints * bins_per_tr
    onset_matrix_micro = torch.zeros((n_microtime, n_conditions), device=device)

    for cond_idx in range(n_conditions):
        duration_bins = max(1, int(np.round(durations[cond_idx] / args.microtime_dt)))

        for run_idx in range(n_runs):
            onsets = all_onsets[cond_idx][run_idx]
            run_start_tr = run_starts[run_idx]
            run_start_micro = run_start_tr * bins_per_tr

            for onset_time in onsets:
                onset_bin = run_start_micro + int(np.round(onset_time / args.microtime_dt))
                if onset_bin < n_microtime:
                    onset_matrix_micro[
                        onset_bin : min(onset_bin + duration_bins, n_microtime),
                        cond_idx,
                    ] = 1.0

    # Get canonical HRF at microtime resolution (SPMG1 - matches 3dHRFoptfast)
    hrf = get_spmg1_hrf(
        microtime_dt=args.microtime_dt,  # Sample at microtime resolution
        stim_duration=0.0,  # Impulse response (duration handled in onset matrix)
        hrf_duration=32.0,
        normalize_peak=True,
        device=device,
    )

    # Convolve and downsample to TR resolution
    task_design = convolve_hrf_microtime(
        onsets_microtime=onset_matrix_micro,
        hrf=hrf,
        n_timepoints=n_timepoints,
        tr=args.tr,
        microtime_dt=args.microtime_dt,
        device=device,
    )
    # Type assertion for pyright (return_single_trials=False returns Tensor, not tuple)
    assert isinstance(task_design, torch.Tensor), (
        "task_design should be Tensor when return_single_trials=False"
    )
    # Build polynomial nuisance regressors PER RUN
    # -------------------------------------------
    # CRITICAL: Nuisance regressors are RUN-SPECIFIC (each run has its own drift)
    # Different runs can have different # of columns (different polort based on duration)
    # We'll pad all to max # columns so they can be concatenated during CV

    nuisance_per_run = []
    max_nuisance_cols = 0  # Track max columns across all runs

    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
        run_length = end_tp - start_tp

        # Auto-determine polort if not specified
        if args.polort is None:
            run_duration = run_length * args.tr
            polort = int(np.floor(1 + run_duration / 150.0))
        else:
            polort = args.polort

        if polort >= 0:
            poly = construct_polynomial_matrix(run_length, polort, device=device)
        else:
            poly = torch.zeros((run_length, 0), device=device)

        nuisance_per_run.append(poly)
        max_nuisance_cols = max(max_nuisance_cols, poly.shape[1])

    # Add ortvec files if provided (split by run)
    if args.ortvec:
        # Load and concatenate all ortvec files
        ortvec_all = []
        for ortvec_file, label in args.ortvec:
            ortvec_data = load_nuisance_file(ortvec_file)
            ortvec_data = to_tensor(ortvec_data, device=device)
            if ortvec_data.shape[0] != n_timepoints:
                print(
                    f"ERROR: ortvec file {ortvec_file} has {ortvec_data.shape[0]} rows, expected {n_timepoints}"
                )
                sys.exit(1)
            ortvec_all.append(ortvec_data)

        ortvec_concat = torch.cat(ortvec_all, dim=1) if ortvec_all else None

        # Split ortvec by run and concatenate with polynomials
        if ortvec_concat is not None:
            for run_idx in range(n_runs):
                start_tp = run_starts[run_idx]
                end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                ortvec_run = ortvec_concat[start_tp:end_tp, :]
                nuisance_per_run[run_idx] = torch.cat(
                    [nuisance_per_run[run_idx], ortvec_run], dim=1
                )

            max_nuisance_cols = nuisance_per_run[0].shape[1]  # All now have same # after ortvec

    # Pad all runs to have same number of columns (for CV concatenation compatibility)
    # -------------------------------------------
    # PADDING STRUCTURE:
    # - Task columns: NO padding (shared across runs)
    # - Nuisance columns: YES padding (run-specific, must match for concatenation)
    #
    # Example: Run 0 has 3 poly cols, Run 1 has 4 poly cols → pad Run 0 to 4
    # This allows clean concatenation during CV: concat([run1_nuisance, run2_nuisance])

    for run_idx in range(n_runs):
        n_cols = nuisance_per_run[run_idx].shape[1]
        if n_cols < max_nuisance_cols:
            # Pad with zeros on the right (extra polynomial terms this run doesn't need)
            padding = torch.zeros(
                (nuisance_per_run[run_idx].shape[0], max_nuisance_cols - n_cols), device=device
            )
            nuisance_per_run[run_idx] = torch.cat([nuisance_per_run[run_idx], padding], dim=1)

    # Summary
    print(f"  Task predictors: {task_design.shape[1]} ({', '.join(condition_labels)})")
    print(f"  Nuisance predictors per run: {nuisance_per_run[0].shape[1]} (polynomial drift)")
    print(
        f"  Total columns per run: {task_design.shape[1]} task + {nuisance_per_run[0].shape[1]} nuisance = {task_design.shape[1] + nuisance_per_run[0].shape[1]}"
    )

    # ==========================================================================
    # Fit denoising model
    # ==========================================================================

    print()
    print("=" * 70)
    print("Fitting cross-validated denoising model...")
    print("=" * 70)

    # Memory strategy:
    # - PCA needs noise pool voxels loaded (subset of data - can't chunk)
    # - GLM fitting chunks voxels automatically via chunk_size
    # - PCs are cached timecourses (tiny memory footprint)
    # - For 16GB GPU: chunk_size=None (auto) works for most datasets
    # - When keep_on_cpu=True: set preload_data_to_device=False to avoid GPU OOM

    # Determine chunk_size based on device:
    # - CPU: process all voxels at once (no GPU memory limit)
    # - GPU: auto-detect based on available memory
    if device.type == "cpu":
        chunk_size = n_voxels  # Process all voxels at once on CPU
        if args.verbose:
            print(f"  CPU mode: chunk_size = {n_voxels:,} (all voxels)")
    else:
        chunk_size = None  # Auto-detect for GPU

    results = fit_denoising_model(
        data=data,
        design_matrix=task_design,
        run_starts=run_starts,
        r2_threshold=args.r2_threshold,
        intensity_mask=brainthresh_mask,  # Intensity threshold for noise pool
        max_components=args.max_pcs,
        variance_threshold=args.variance_threshold,
        nuisance=nuisance_per_run,  # Pass as list per run (cleaner bookkeeping!)
        tr=args.tr,
        polort=args.polort,
        metric=args.cv_metric,
        min_noise_voxels=args.min_noise_voxels,
        max_noise_fraction=args.max_noise_fraction,
        pcstop=args.pcstop,  # GLMdenoise-style early stopping
        pcR2cutoff=args.pcR2cutoff,  # R² cutoff for PC selection voxels
        cv_strategy=cv_strategy,  # CV split strategy (1=LORO, float=split fraction, int>1=leave-N-out)
        n_perms=args.n_perms,  # Max CV permutations for random splits
        chunk_size=chunk_size,  # CPU: all voxels, GPU: auto-detect
        preload_data_to_device=not keep_on_cpu,  # Don't preload if using CPU chunking
        return_loadings=(
            args.save_pcs in ["spatial", "both"] or args.plots == "full"
        ),  # Get loadings for spatial output/plots
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
        run_starts=run_starts,
        tr=args.tr,
        voxel_mask=voxel_mask,
        plots_mode=args.plots,
        slice_axis=args.plot_ax,
        save_pcs_mode=args.save_pcs,
        condition_labels=condition_labels,
    )

    # ==========================================================================
    # Save initial and final model fits (if requested or needed for SNR)
    # ==========================================================================

    # We need model fits if either save_model_fit or snr is requested
    need_model_fits = args.save_model_fit or args.snr

    # Clear GPU memory after saving denoising results
    # The results object and plotting may have left tensors on GPU
    if device.type == "cuda":
        # Move results tensors to CPU to free GPU memory
        if hasattr(results, "noise_pcs_per_run") and results.noise_pcs_per_run is not None:
            results.noise_pcs_per_run = [
                pc.cpu() if torch.is_tensor(pc) else pc for pc in results.noise_pcs_per_run
            ]
        if hasattr(results, "pc_loadings_per_run") and results.pc_loadings_per_run is not None:
            results.pc_loadings_per_run = [
                ld.cpu() if torch.is_tensor(ld) else ld for ld in results.pc_loadings_per_run
            ]

        # CRITICAL: Move main data tensor to CPU if it's on GPU
        # This frees up the largest allocation before model fitting
        if torch.is_tensor(data) and data.device.type == "cuda":
            data = data.cpu()
            if args.verbose:
                print("  Moved data tensor to CPU to free GPU memory")

        torch.cuda.empty_cache()
        if args.verbose:
            print("  Cleared GPU cache before model fitting")

    initial_results = None
    final_results = None
    initial_bootstrap_se = None
    final_bootstrap_se = None

    if need_model_fits:
        print()
        print("Fitting initial model (no denoising)...")

        from fastfuncsim.glm_core import fit_glm

        # Build zero-padded nuisance for concatenated fit
        n_total_timepoints = data.shape[1]
        nuisance_padded_list = []
        current_tp = 0
        for run_nuisance in nuisance_per_run:
            run_length = run_nuisance.shape[0]
            n_cols = run_nuisance.shape[1]
            padded = torch.zeros((n_total_timepoints, n_cols), device=device)
            padded[current_tp : current_tp + run_length, :] = run_nuisance
            nuisance_padded_list.append(padded)
            current_tp += run_length
        nuisance_concat = torch.cat(nuisance_padded_list, dim=1)

        # Full design for initial fit
        full_design_initial = torch.cat([task_design, nuisance_concat], dim=1)
        n_task_cols = task_design.shape[1]
        n_total_regs_initial = full_design_initial.shape[1]

        initial_results = fit_glm(
            data=data,
            design=task_design,
            tr=args.tr,
            extra_regressors=nuisance_concat,
            want_residuals=True,  # Need residuals for SNR
            chunk_size=chunk_size,  # CPU: all voxels, GPU: auto-detect
            preload_data_to_device=False,  # ALWAYS stream from CPU for model fits (safer)
            device=device,
            verbose=False,
        )

        # Bootstrap SE for initial model
        if args.numboots > 0:
            print(f"  Computing bootstrap SE ({args.numboots} iterations)...")
            initial_bootstrap_se = compute_bootstrap_se(
                data=data,
                design=full_design_initial,
                n_task=n_task_cols,
                n_boots=args.numboots,
                device=device,
                verbose=args.verbose,
            )

        if args.save_model_fit:
            initial_files = save_model_fit_outputs(
                results=initial_results,
                output_prefix=args.prefix,
                volume_shape=volume_shape,
                affine=affine,
                model_type="initial",
                condition_labels=condition_labels,
                voxel_mask=voxel_mask,
                n_timepoints=n_timepoints,
                n_regressors=n_total_regs_initial,
                bootstrap_se=initial_bootstrap_se,
            )
            output_files.update(initial_files)

        # Final fit (with optimal denoising)
        print(f"Fitting final model (with {results.optimal_n_components} noise PCs)...")

        # Build combined nuisance with noise PCs for final fit
        n_pcs_optimal = results.optimal_n_components
        if n_pcs_optimal > 0:
            pc_padded_blocks = []
            for run_idx in range(n_runs):
                start_tp = run_starts[run_idx]
                end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
                run_length = end_tp - start_tp
                pcs_run = results.noise_pcs_per_run[run_idx][:, :n_pcs_optimal]

                padded = torch.zeros((n_timepoints, n_runs * n_pcs_optimal), device=device)
                start_col = run_idx * n_pcs_optimal
                end_col = start_col + n_pcs_optimal
                padded[start_tp:end_tp, start_col:end_col] = pcs_run

                pc_padded_blocks.append(padded)

            pc_concat = sum(pc_padded_blocks)

            if nuisance_concat is not None:
                nuisance_with_pcs = torch.cat([nuisance_concat, pc_concat], dim=1)
            else:
                nuisance_with_pcs = pc_concat
        else:
            nuisance_with_pcs = nuisance_concat

        full_design_final = torch.cat([task_design, nuisance_with_pcs], dim=1)
        n_total_regs_final = full_design_final.shape[1]

        final_results = fit_glm(
            data=data,
            design=task_design,
            tr=args.tr,
            extra_regressors=nuisance_with_pcs,
            want_residuals=True,  # Need residuals for SNR
            chunk_size=chunk_size,  # CPU: all voxels, GPU: auto-detect
            preload_data_to_device=False,  # ALWAYS stream from CPU for model fits (safer)
            device=device,
            verbose=False,
        )

        # Bootstrap SE for final model
        if args.numboots > 0:
            print(f"  Computing bootstrap SE for denoised model ({args.numboots} iterations)...")
            final_bootstrap_se = compute_bootstrap_se(
                data=data,
                design=full_design_final,
                n_task=n_task_cols,
                n_boots=args.numboots,
                device=device,
                verbose=args.verbose,
            )

        if args.save_model_fit:
            final_files = save_model_fit_outputs(
                results=final_results,
                output_prefix=args.prefix,
                volume_shape=volume_shape,
                affine=affine,
                model_type="denoised",
                condition_labels=condition_labels,
                voxel_mask=voxel_mask,
                n_timepoints=n_timepoints,
                n_regressors=n_total_regs_final,
                bootstrap_se=final_bootstrap_se,
            )
            output_files.update(final_files)

    # ==========================================================================
    # Compute and save SNR (if requested)
    # ==========================================================================

    if args.snr and initial_results is not None and final_results is not None:
        print()
        print("Computing SNR metrics...")

        # Get betas and residuals
        initial_betas = initial_results.betas[:, :n_conditions].cpu().numpy()
        final_betas = final_results.betas[:, :n_conditions].cpu().numpy()

        # Residual std from MSE
        initial_residual_std = (
            torch.sqrt(initial_results.mse).cpu().numpy()
            if hasattr(initial_results, "mse") and initial_results.mse is not None
            else None
        )
        final_residual_std = (
            torch.sqrt(final_results.mse).cpu().numpy()
            if hasattr(final_results, "mse") and final_results.mse is not None
            else None
        )

        # If MSE not available, compute from residuals
        if initial_residual_std is None and initial_results.residuals is not None:
            initial_residual_std = initial_results.residuals.std(dim=1).cpu().numpy()
        if final_residual_std is None and final_results.residuals is not None:
            final_residual_std = final_results.residuals.std(dim=1).cpu().numpy()

        # Compute SNR
        snr_initial = compute_snr(
            betas=initial_betas,
            residual_std=initial_residual_std,
            bootstrap_se=initial_bootstrap_se,
        )
        snr_denoised = compute_snr(
            betas=final_betas,
            residual_std=final_residual_std,
            bootstrap_se=final_bootstrap_se,
        )

        # Report improvement
        if "snr_residual" in snr_initial and "snr_residual" in snr_denoised:
            mean_initial = snr_initial["snr_residual"].mean()
            mean_denoised = snr_denoised["snr_residual"].mean()
            print(
                f"  Residual-based SNR: {mean_initial:.2f} → {mean_denoised:.2f} ({(mean_denoised / mean_initial - 1) * 100:+.1f}%)"
            )

        if "snr_bootstrap" in snr_initial and "snr_bootstrap" in snr_denoised:
            mean_initial = snr_initial["snr_bootstrap"].mean()
            mean_denoised = snr_denoised["snr_bootstrap"].mean()
            print(
                f"  Bootstrap-based SNR: {mean_initial:.2f} → {mean_denoised:.2f} ({(mean_denoised / mean_initial - 1) * 100:+.1f}%)"
            )

        # Save SNR outputs
        snr_files = save_snr_outputs(
            snr_initial=snr_initial,
            snr_denoised=snr_denoised,
            output_prefix=args.prefix,
            volume_shape=volume_shape,
            affine=affine,
            voxel_mask=voxel_mask,
            create_plots=True,
        )
        output_files.update(snr_files)

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
