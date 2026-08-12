#!/usr/bin/env python3
"""
ffs_pathfinder - Joint HRF + Denoising Optimization Pipeline

This tool combines HRF optimization (from 3dHRFoptfast) with adaptive denoising
(from 3dDenoisefast) into a single joint optimization loop:

1. For each HRF candidate in the library:
   a. Build task design matrix with this HRF
   b. Compute initial cross-validated R² (task-only) for noise pool selection
   c. Extract noise PCs from noise pool voxels
   d. Cross-validate denoising: test 0 to N PCs, find optimal PC count
   e. Track the best denoised CV R² for this HRF

2. Select best HRF per voxel based on DENOISED CV R² (not raw R²)

3. Final refit with voxel-wise optimal HRFs and optimal PC counts

This is similar to GLMsingle's approach where HRF selection and denoising are
jointly optimized, finding the combination that best predicts held-out data.

Chunking Strategy:
- Outer loop: HRFs (process all HRFs, chunking voxels within each)
- Inner loop: Voxels (GLM fitting and CV are chunked automatically)

Basic usage:
    ffs_pathfinder -input run1.nii.gz run2.nii.gz run3.nii.gz \\
                   -onsets cond1.txt cond2.txt \\
                   -durations 2.0 5.0 \\
                   -tr 2.0 \\
                   -prefix subject01_pathfinder

For help:
    ffs_pathfinder -help
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    import nibabel as nib  # noqa: F401
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

# Import fastfuncstuff modules
try:
    from fastfuncstuff.cli_utils import (
        add_ortvec_arguments,
        add_verbose_arg,
        auto_polort,
        collect_nuisance_blocks,
        compute_run_lengths,
        get_average_run_duration,
        parse_cv_strategy,
        parse_input_files,
        parse_prefix,
        setup_device,
        spinner,
    )
    from fastfuncstuff.denoise.sequential import (
        extract_noise_pcs_per_run,
        select_noise_pool_voxels,
    )
    from fastfuncstuff.design.builder import parse_afni_timing_file, parse_durations
    from fastfuncstuff.design.hrf import get_hrf_library, get_spmg1_hrf
    from fastfuncstuff.design.hrf_selection import load_nuisance_file  # noqa: F401
    from fastfuncstuff.design.matrices import convolve_hrf_microtime
    from fastfuncstuff.glm.core import GLMResults, construct_polynomial_matrix, fit_glm
    from fastfuncstuff.glm.outputs import write_glm_bucket_as_nifti
    from fastfuncstuff.glm.xval import (
        compute_xval_r2,
        generate_cv_splits,
        project_out_nuisance_per_run,
    )
    from fastfuncstuff.io.afni import (
        load_afni_mask,
        load_and_concatenate_runs,
        load_nifti,
        save_nifti,
    )
    from fastfuncstuff.utils import (
        gaussian_blur_3d,
        get_device,
        scale_to_percent_signal,
    )
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    print("Make sure fastfuncstuff is installed: pip install -e .")
    sys.exit(1)


@dataclass
class PathfinderResults:
    """Results from joint HRF + denoising optimization.

    Attributes
    ----------
    hrf_index : torch.Tensor
        (n_voxels,) Index of best HRF for each voxel (0 to n_hrfs-1)
    optimal_n_pcs : int
        Global optimal number of noise PCs selected
    xval_r2_best : torch.Tensor
        (n_voxels,) Denoised cross-validated R² for selected HRF
    xval_r2_all_hrfs : torch.Tensor
        (n_voxels, n_hrfs) Denoised CV R² for each HRF at optimal PCs
    xval_r2_by_hrf_and_pcs : np.ndarray
        (n_hrfs, max_pcs+1) Mean CV R² for each HRF and PC count
    noise_pool_mask : torch.Tensor
        (n_voxels,) Boolean mask for noise pool voxels
    criteria_mask : torch.Tensor
        (n_voxels,) Boolean mask for criteria voxels
    noise_pcs_per_run : List[torch.Tensor]
        Extracted noise PCs per run (from final HRF-based R²)
    initial_results : GLMResults
        Initial GLM fit with canonical HRF, no denoising (baseline)
    initial_xval_r2 : torch.Tensor
        (n_voxels,) Initial cross-validated R² (canonical HRF, no denoising)
    final_results : GLMResults
        Final GLM fit with optimal HRFs and denoising
    hrf_library : torch.Tensor
        (n_hrfs, n_hrf_samples) The HRF library used
    metadata : dict
        Additional metadata
    """

    hrf_index: torch.Tensor = None
    optimal_n_pcs: int = 0
    xval_r2_best: torch.Tensor = None
    xval_r2_all_hrfs: torch.Tensor = None
    xval_r2_by_hrf_and_pcs: np.ndarray = None
    noise_pool_mask: torch.Tensor = None
    criteria_mask: torch.Tensor = None
    noise_pcs_per_run: list[torch.Tensor] = field(default_factory=list)
    initial_results: GLMResults = None
    initial_xval_r2: torch.Tensor = None
    final_results: GLMResults = None
    hrf_library: torch.Tensor = None
    metadata: dict = field(default_factory=dict)


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults while preserving raw description formatting."""


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="ffs_pathfinder - Joint HRF + Denoising Optimization",
        formatter_class=_HelpFormatter,
        epilog="""
Examples:
  # Basic joint optimization
  ffs_pathfinder -input run1.nii.gz run2.nii.gz run3.nii.gz \\
                 -onsets cond1.txt cond2.txt \\
                 -durations 2.0 \\
                 -tr 2.0 \\
                 -prefix subject01_pathfinder

  # With PIGHS HRF library and custom PC range
  ffs_pathfinder -input run*.nii.gz \\
                 -onsets face.txt house.txt \\
                 -durations 2.0 \\
                 -tr 2.0 \\
                 -hrf_mode pighs \\
                 -n_hrfs 20 \\
                 -max_pcs 30 \\
                 -prefix sub01_pighs_denoised

  # With mask and motion regressors
  ffs_pathfinder -input run*.nii.gz \\
                 -onsets stim.txt \\
                 -durations 1.0 \\
                 -tr 2.0 \\
                 -mask brain_mask.nii.gz \\
                 -ortvec motion_all.1D motion \\
                 -verbose \\
                 -prefix masked_pathfinder

Outputs:
  INITIAL (canonical HRF, no denoising):
    {prefix}_initial_xval_r2.nii.gz   - Initial cross-validated R²
    {prefix}_initial_r2.nii.gz        - Initial full-fit R²
    {prefix}_initial_stats.nii.gz     - Initial betas and t-stats (AFNI bucket)

  FINAL (optimal HRFs + denoising):
    {prefix}_final_xval_r2.nii.gz     - Final denoised cross-validated R²
    {prefix}_final_r2.nii.gz          - Final full-fit R²
    {prefix}_final_stats.nii.gz       - Final betas and t-stats (AFNI bucket)

  HRF/DENOISING:
    {prefix}_hrf_index.nii.gz         - Best HRF per voxel (1-indexed)
    {prefix}_xval_r2_all_hrfs.nii.gz  - 4D: Denoised CV R² for each HRF
    {prefix}_noise_pool_mask.nii.gz   - Noise pool voxels
    {prefix}_criteria_mask.nii.gz     - Criteria voxels
    {prefix}_noise_pcs.pt             - Extracted noise PCs
    {prefix}_optimization_curve.npy   - HRF x PC optimization surface
    {prefix}_metadata.json            - Full metadata

Workflow:
  1. For each HRF candidate:
     a. Build task design matrix with this HRF
     b. Compute xval R² to define noise pool
     c. Extract noise PCs from noise pool
     d. Cross-validate 0 to N PCs to find optimal count
     e. Store denoised CV R² for this HRF

  2. Select best HRF per voxel based on denoised CV R²

  3. Final refit with voxel-wise optimal HRFs + global optimal PCs

Notes:
  - At least 2 runs required for cross-validation
  - Joint optimization finds the HRF that works best WITH denoising
  - This replicates core aspects of GLMsingle's optimization loop
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
        help="TR in seconds. If not specified, read from NIfTI header.",
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
        help="HRF library type: 'library' (double-gamma) or 'pighs'. Default: library",
    )
    hrf_opts.add_argument(
        "-n_hrfs",
        type=int,
        default=20,
        help="Number of HRFs in library (default: 20)",
    )

    # Denoising options
    denoise_opts = parser.add_argument_group("Denoising Options")
    denoise_opts.add_argument(
        "-r2_threshold",
        type=float,
        default=0.05,
        help="R² threshold for noise pool selection (default: 0.05)",
    )
    denoise_opts.add_argument(
        "-max_pcs",
        type=int,
        default=20,
        help="Maximum number of noise PCs to test (default: 20)",
    )
    denoise_opts.add_argument(
        "-pcstop",
        type=float,
        default=1.05,
        help="PC selection stopping threshold (default: 1.05 = within 5%% of max)",
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

    # Cross-validation options
    cv_opts = parser.add_argument_group("Cross-Validation Options")
    cv_opts.add_argument(
        "-cv_strategy",
        default="loro",
        help="CV strategy: 'loro' (leave-one-run-out), float for train fraction",
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
        help="R² metric: 'cod', 'corr', or 'corr2' (default: cod)",
    )
    cv_opts.add_argument(
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
        help="Polynomial order for drift modeling (default: auto)",
    )
    add_ortvec_arguments(proc_opts)
    proc_opts.add_argument(
        "-microtime_dt",
        type=float,
        default=0.1,
        help="Microtime resolution in seconds (default: 0.1)",
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
        help="Force device: 'cpu' or 'cuda' (default: auto-detect)",
    )
    proc_opts.add_argument(
        "-keep_on_cpu",
        action="store_true",
        help="Load data to CPU and process in GPU chunks",
    )
    proc_opts.add_argument(
        "-batch_size",
        type=int,
        default=None,
        help="Number of voxels per batch (default: auto)",
    )
    add_verbose_arg(proc_opts, default=0)

    # Output options
    out_opts = parser.add_argument_group("Output Options")
    out_opts.add_argument(
        "-plots",
        action="store_true",
        help="Save diagnostic plots (optimization surface, etc.)",
    )
    out_opts.add_argument(
        "-save_all_hrfs",
        action="store_true",
        help="Save denoised xval R² for ALL HRFs (4D volume)",
    )

    return parser


def print_header():
    """Print program header"""
    print("=" * 70)
    print("ffs_pathfinder - Joint HRF + Denoising Optimization")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def cross_validate_denoising_for_hrf(
    data: torch.Tensor,
    design_matrix: torch.Tensor,
    noise_pcs: list[torch.Tensor],
    run_starts: list[int],
    criteria_mask: torch.Tensor,
    nuisance_per_run: list[torch.Tensor],
    max_pcs: int,
    cv_splits: list[tuple[list[int], list[int]]],
    device: torch.device,
    chunk_size: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-validate noise PC denoising for a single HRF.

    Returns
    -------
    r2_by_n_pcs : np.ndarray
        (max_pcs+1,) Median CV R² for each number of PCs
    r2_per_voxel : np.ndarray
        (n_criteria, max_pcs+1) Per-voxel CV R² at each PC count
    """
    proj_device = data.device
    n_runs = len(run_starts)
    n_timepoints = data.shape[1]
    n_task_regs = design_matrix.shape[1]
    n_splits = len(cv_splits)
    n_criteria = criteria_mask.sum().item()

    # Storage for CV results
    r2_per_fold = np.zeros((n_splits, max_pcs + 1))
    r2_per_voxel_accum = np.zeros((n_criteria, max_pcs + 1))

    for fold_idx, (train_runs, test_runs) in enumerate(cv_splits):
        # Build train/test indices
        train_tps = []
        for run_idx in train_runs:
            start_tp = run_starts[run_idx]
            end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            train_tps.extend(range(start_tp, end_tp))

        test_tps = []
        for run_idx in test_runs:
            start_tp = run_starts[run_idx]
            end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            test_tps.extend(range(start_tp, end_tp))

        train_tps_t = torch.tensor(train_tps, device=data.device)
        test_tps_t = torch.tensor(test_tps, device=data.device)

        data_train = data[:, train_tps_t]
        data_test = data[:, test_tps_t]
        design_train = design_matrix[train_tps_t, :]
        design_test = design_matrix[test_tps_t, :]

        # Project out nuisance per run (project-first approach)
        train_data_proj_runs = []
        train_design_proj_runs = []

        for run_idx in train_runs:
            run_start_in_train = 0
            for prev_run in train_runs:
                if prev_run == run_idx:
                    break
                prev_start = run_starts[prev_run]
                prev_end = run_starts[prev_run + 1] if prev_run < n_runs - 1 else n_timepoints
                run_start_in_train += prev_end - prev_start

            run_start_global = run_starts[run_idx]
            run_end_global = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            run_length = run_end_global - run_start_global
            run_end_in_train = run_start_in_train + run_length

            run_data = data_train[:, run_start_in_train:run_end_in_train].to(proj_device)
            run_design = design_train[run_start_in_train:run_end_in_train, :].to(proj_device)
            run_nuisance = nuisance_per_run[run_idx].to(proj_device)

            if run_nuisance.shape[1] > 0:
                XtX = run_nuisance.T @ run_nuisance
                XtX_inv = torch.linalg.inv(XtX + 1e-6 * torch.eye(XtX.shape[0], device=proj_device))
                P_nuisance = run_nuisance @ XtX_inv @ run_nuisance.T
                projection = torch.eye(run_length, device=proj_device) - P_nuisance
                run_data_proj = (projection @ run_data.T).T
                run_design_proj = projection @ run_design
            else:
                run_data_proj = run_data
                run_design_proj = run_design

            train_data_proj_runs.append(run_data_proj)
            train_design_proj_runs.append(run_design_proj)

        data_train_projected = torch.cat(train_data_proj_runs, dim=1)
        design_train_projected = torch.cat(train_design_proj_runs, dim=0)

        # Project test data per run
        test_data_proj_runs = []
        test_design_proj_runs = []

        for run_idx in test_runs:
            run_start_global = run_starts[run_idx]
            run_end_global = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            run_length = run_end_global - run_start_global

            run_start_in_test = 0
            for prev_run in test_runs:
                if prev_run == run_idx:
                    break
                prev_start = run_starts[prev_run]
                prev_end = run_starts[prev_run + 1] if prev_run < n_runs - 1 else n_timepoints
                run_start_in_test += prev_end - prev_start

            run_end_in_test = run_start_in_test + run_length
            run_test_data = data_test[:, run_start_in_test:run_end_in_test].to(proj_device)
            run_test_design = design_test[run_start_in_test:run_end_in_test, :].to(proj_device)
            run_nuisance = nuisance_per_run[run_idx].to(proj_device)

            if run_nuisance.shape[1] > 0:
                XtX_test = run_nuisance.T @ run_nuisance
                XtX_test_inv = torch.linalg.inv(
                    XtX_test + 1e-6 * torch.eye(XtX_test.shape[0], device=proj_device)
                )
                P_nuisance_test = run_nuisance @ XtX_test_inv @ run_nuisance.T
                projection_test = torch.eye(run_length, device=proj_device) - P_nuisance_test
                run_test_data_proj = (projection_test @ run_test_data.T).T
                run_test_design_proj = projection_test @ run_test_design
            else:
                run_test_data_proj = run_test_data
                run_test_design_proj = run_test_design

            test_data_proj_runs.append(run_test_data_proj)
            test_design_proj_runs.append(run_test_design_proj)

        data_test_projected = torch.cat(test_data_proj_runs, dim=1)
        design_test_projected = torch.cat(test_design_proj_runs, dim=0)

        # Project nuisance from PCs for training runs
        train_noise_pcs_projected = []
        for run_idx in train_runs:
            run_start_global = run_starts[run_idx]
            run_end_global = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            run_length = run_end_global - run_start_global

            pcs_this_run = noise_pcs[run_idx]
            run_nuisance = nuisance_per_run[run_idx].to(device)

            if run_nuisance.shape[1] > 0:
                XtX = run_nuisance.T @ run_nuisance
                XtX_inv = torch.linalg.inv(XtX + 1e-6 * torch.eye(XtX.shape[0], device=device))
                P_nuisance = run_nuisance @ XtX_inv @ run_nuisance.T
                projection = torch.eye(run_length, device=device) - P_nuisance
                pcs_projected = projection @ pcs_this_run.to(device)
            else:
                pcs_projected = pcs_this_run.to(device)

            train_noise_pcs_projected.append(pcs_projected)

        # Pre-compute pseudo-inverses for each PC count
        pinv_task_gpu = []
        n_train_runs = len(train_noise_pcs_projected)

        for n_pcs in range(max_pcs + 1):
            components = [design_train_projected.to(device)]

            if n_pcs > 0:
                pc_padded_blocks = []
                for block_idx, pcs_run in enumerate(train_noise_pcs_projected):
                    run_length = pcs_run.shape[0]
                    n_available = pcs_run.shape[1]
                    n_use = min(n_pcs, n_available)

                    padded = torch.zeros((run_length, n_train_runs * n_pcs), device=device)
                    start_col = block_idx * n_pcs
                    end_col = start_col + n_use
                    padded[:, start_col:end_col] = pcs_run[:, :n_use]
                    pc_padded_blocks.append(padded)

                pc_combined = torch.cat(pc_padded_blocks, dim=0)
                components.append(pc_combined)

            x_full = torch.cat(components, dim=1)
            xtx = x_full.T @ x_full
            xtx_inv = torch.linalg.pinv(xtx)
            pinv_full = xtx_inv @ x_full.T
            pinv_task = pinv_full[:n_task_regs, :]
            pinv_task_gpu.append(pinv_task)

        design_test_gpu = design_test_projected.to(device)
        pinv_stack = torch.stack(pinv_task_gpu, dim=0)

        # Chunk-based evaluation
        criteria_mask_device = criteria_mask.to(data.device)
        criteria_indices = torch.where(criteria_mask_device)[0]
        auto_chunk = chunk_size or 50000
        n_chunks = (n_criteria + auto_chunk - 1) // auto_chunk

        r2_accum = [[] for _ in range(max_pcs + 1)]

        for chunk_idx in range(n_chunks):
            start_v = chunk_idx * auto_chunk
            end_v = min(start_v + auto_chunk, n_criteria)
            chunk_indices = criteria_indices[start_v:end_v]

            chunk_train = data_train_projected[chunk_indices, :].to(device)
            chunk_test = data_test_projected[chunk_indices, :].to(device)

            # Batched computation for ALL PC counts
            betas_all = torch.einsum("prt,tc->prc", pinv_stack, chunk_train.T)
            y_pred_all = torch.einsum("tr,prc->pct", design_test_gpu, betas_all)

            residuals = chunk_test.unsqueeze(0) - y_pred_all
            ss_res_all = (residuals**2).sum(dim=2)
            ss_tot = ((chunk_test - chunk_test.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
            r2_all = 1 - (ss_res_all / (ss_tot.unsqueeze(0) + 1e-10))

            for n_pcs in range(max_pcs + 1):
                r2_accum[n_pcs].append(r2_all[n_pcs].cpu())

        # Aggregate
        for n_pcs in range(max_pcs + 1):
            all_r2 = torch.cat(r2_accum[n_pcs])
            r2_per_fold[fold_idx, n_pcs] = all_r2.median().item()
            r2_per_voxel_accum[:, n_pcs] += all_r2.numpy()

    r2_by_n_pcs = np.median(r2_per_fold, axis=0)
    r2_per_voxel = r2_per_voxel_accum / n_splits

    return r2_by_n_pcs, r2_per_voxel


def fit_pathfinder(
    data: torch.Tensor,
    onset_matrix: torch.Tensor,
    hrf_library: torch.Tensor,
    tr: float,
    run_starts: list[int],
    nuisance_per_run: list[torch.Tensor],
    cv_splits: list[tuple[list[int], list[int]]],
    r2_threshold: float = 0.05,
    max_pcs: int = 20,
    variance_threshold: float = 0.95,
    min_noise_voxels: int = 100,
    max_noise_fraction: float = 0.5,
    pcstop: float = 1.05,
    microtime_dt: float = 0.1,
    metric: str = "cod",
    device: torch.device = None,
    chunk_size: int | None = None,
    verbose: bool = True,
) -> PathfinderResults:
    """
    Main pathfinder optimization loop.

    For each HRF:
    1. Build task design matrix
    2. Compute initial xval R² to identify noise pool
    3. Extract noise PCs
    4. Cross-validate denoising (0 to max_pcs)
    5. Track best denoised R² for this HRF

    Then select best HRF per voxel based on denoised R².
    """
    if device is None:
        device = get_device()

    n_voxels, n_timepoints = data.shape
    n_hrfs = hrf_library.shape[0]
    n_runs = len(run_starts)
    _bins_per_tr = int(round(tr / microtime_dt))
    n_conditions = onset_matrix.shape[1]

    if verbose:
        print("=" * 70)
        print("JOINT HRF + DENOISING OPTIMIZATION")
        print("=" * 70)
        print(f"  Voxels: {n_voxels:,}")
        print(f"  Timepoints: {n_timepoints}")
        print(f"  Runs: {n_runs}")
        print(f"  HRF candidates: {n_hrfs}")
        print(f"  Max PCs to test: {max_pcs}")
        print(f"  CV folds: {len(cv_splits)}")
        print()

    # =========================================================================
    # Compute INITIAL baseline: canonical HRF, no denoising
    # This is the "what if we did nothing fancy" comparison
    # =========================================================================
    if verbose:
        print("Computing initial baseline (canonical HRF, no denoising)...")

    # Get canonical HRF (SPMG1)
    canonical_hrf = get_spmg1_hrf(
        microtime_dt=microtime_dt,
        stim_duration=0.0,  # Impulse response
        normalize_peak=True,
        device=device,
    )

    # Build canonical design matrix
    canonical_design = convolve_hrf_microtime(
        onset_matrix,
        canonical_hrf,
        n_timepoints,
        tr=tr,
        microtime_dt=microtime_dt,
        device=device,
    )

    # Project out nuisance (for CV)
    projected_data_init, projected_design_init = project_out_nuisance_per_run(
        data=data,
        design=canonical_design,
        nuisance_per_run=nuisance_per_run,
        run_starts=run_starts,
        device=device,
    )

    # Compute initial xval R² (canonical HRF, no denoising)
    initial_xval_results = compute_xval_r2(
        data=projected_data_init,
        design_matrix=projected_design_init,
        run_starts=run_starts,
        stim_indices=list(range(n_conditions)),
        nuisance_indices=[],
        cv_splits=cv_splits,
        metric=metric,
        zero_event_strategy="zero",
        device=device,
        batch_size=chunk_size,
        verbose=False,
    )
    _initial_r2_median = initial_xval_results["r2_median"]
    assert isinstance(_initial_r2_median, torch.Tensor)  # "r2_median" key is always a tensor
    initial_xval_r2 = _initial_r2_median.cpu()

    # Full GLM fit with canonical HRF for betas/tstats
    nuisance_block_diag = torch.block_diag(*nuisance_per_run)
    full_canonical_design = torch.cat([canonical_design, nuisance_block_diag], dim=1)

    initial_glm_results = fit_glm(
        data=data,
        design=full_canonical_design,
        tr=tr,
        max_poly_degree=0,  # Already in design
        device=device,
        verbose=False,
        task_indices=list(range(n_conditions)),
        preload_data_to_device=(data.device == device),
    )

    assert initial_glm_results.r2 is not None  # set by fit_glm above
    if verbose:
        print(f"  Initial xval R² (canonical, no denoise): {initial_xval_r2.mean().item():.4f}")
        print(f"  Initial full-fit R²: {initial_glm_results.r2.mean().item():.4f}")
        print()

    # Storage for results across all HRFs
    # (n_voxels, n_hrfs) - best denoised CV R² per HRF per voxel
    xval_r2_all_hrfs = torch.zeros(n_voxels, n_hrfs, device=device)
    # (n_hrfs, max_pcs+1) - mean CV R² curve for each HRF
    xval_r2_by_hrf_and_pcs = np.zeros((n_hrfs, max_pcs + 1))
    # Track optimal PCs per HRF
    optimal_pcs_per_hrf = np.zeros(n_hrfs, dtype=int)

    # We'll use the best HRF's noise pool for final extraction
    best_noise_pool_mask = None
    best_criteria_mask = None
    best_noise_pcs = None

    # Main loop: evaluate each HRF
    hrf_iterator = tqdm(range(n_hrfs), desc="Evaluating HRF + Denoising")

    for hrf_idx in hrf_iterator:
        hrf = hrf_library[hrf_idx]

        # 1. Build task design matrix for this HRF
        stim_design = convolve_hrf_microtime(
            onset_matrix,
            hrf,
            n_timepoints,
            tr=tr,
            microtime_dt=microtime_dt,
            device=device,
        )
        assert isinstance(stim_design, torch.Tensor)  # return_single_trials defaults False
        n_stim_cols = stim_design.shape[1]

        # 2. Compute initial xval R² (task-only) to identify noise pool
        # Project out nuisance first (GLMdenoise style)
        projected_data, projected_design = project_out_nuisance_per_run(
            data=data,
            design=stim_design,
            nuisance_per_run=nuisance_per_run,
            run_starts=run_starts,
            device=device,
        )

        # Run CV on projected data
        xval_results = compute_xval_r2(
            data=projected_data,
            design_matrix=projected_design,
            run_starts=run_starts,
            stim_indices=list(range(n_stim_cols)),
            nuisance_indices=[],
            cv_splits=cv_splits,
            metric=metric,
            zero_event_strategy="zero",
            device=device,
            batch_size=chunk_size,
            verbose=False,
        )

        _r2_median = xval_results["r2_median"]
        assert isinstance(_r2_median, torch.Tensor)  # "r2_median" key is always a tensor
        initial_r2 = _r2_median.to(device)

        # 3. Select noise pool and criteria voxels
        try:
            noise_pool_mask, criteria_mask = select_noise_pool_voxels(
                r2=initial_r2,
                threshold=r2_threshold,
                min_noise_voxels=min_noise_voxels,
                max_noise_fraction=max_noise_fraction,
            )
        except ValueError as e:
            if verbose:
                print(f"  HRF {hrf_idx}: Skipping - {e}")
            continue

        # 4. Extract noise PCs
        noise_pcs = extract_noise_pcs_per_run(
            data=data,
            run_starts=run_starts,
            noise_pool_mask=noise_pool_mask,
            max_components=max_pcs,
            variance_threshold=variance_threshold,
            return_loadings=False,
            device=device,
            verbose=False,
        )

        # 5. Cross-validate denoising for this HRF
        r2_by_n_pcs, r2_per_voxel = cross_validate_denoising_for_hrf(
            data=data,
            design_matrix=stim_design,
            noise_pcs=noise_pcs,
            run_starts=run_starts,
            criteria_mask=criteria_mask,
            nuisance_per_run=nuisance_per_run,
            max_pcs=max_pcs,
            cv_splits=cv_splits,
            device=device,
            chunk_size=chunk_size,
            verbose=False,
        )

        # Select optimal PC count for this HRF (GLMdenoise-style early stopping)
        curve = r2_by_n_pcs - r2_by_n_pcs[0]
        max_improvement = curve.max()

        if pcstop < 0:
            optimal_pcs = int(abs(pcstop))
        elif pcstop == 1.0:
            optimal_pcs = int(np.argmax(r2_by_n_pcs))
        else:
            threshold = max_improvement / pcstop
            optimal_pcs = 0
            best_so_far = -np.inf
            for n_pcs in range(len(curve)):
                if curve[n_pcs] > best_so_far:
                    optimal_pcs = n_pcs
                    best_so_far = curve[n_pcs]
                    if best_so_far >= threshold:
                        break

        optimal_pcs_per_hrf[hrf_idx] = optimal_pcs
        xval_r2_by_hrf_and_pcs[hrf_idx, :] = r2_by_n_pcs

        # Store per-voxel R² at optimal PCs for this HRF
        # Map criteria voxels back to full volume
        optimal_r2_per_voxel = r2_per_voxel[:, optimal_pcs]
        criteria_indices = torch.where(criteria_mask)[0]
        xval_r2_all_hrfs[criteria_indices, hrf_idx] = torch.from_numpy(
            optimal_r2_per_voxel.astype(np.float32)
        ).to(device)

        # Track best HRF's noise pool/PCs for final use
        mean_r2_this_hrf = r2_by_n_pcs[optimal_pcs]
        if (
            best_noise_pool_mask is None
            or mean_r2_this_hrf > xval_r2_by_hrf_and_pcs[:hrf_idx, :].max()
        ):
            best_noise_pool_mask = noise_pool_mask
            best_criteria_mask = criteria_mask
            best_noise_pcs = noise_pcs

        hrf_iterator.set_postfix(
            {
                "HRF": hrf_idx,
                "opt_PCs": optimal_pcs,
                "R²": f"{mean_r2_this_hrf:.4f}",
            }
        )

    # Select best HRF per voxel based on denoised CV R²
    hrf_index = xval_r2_all_hrfs.argmax(dim=1)
    xval_r2_best = xval_r2_all_hrfs[torch.arange(n_voxels, device=device), hrf_index]

    # Determine global optimal PC count (mode across HRFs weighted by voxel count)
    hrf_counts = torch.bincount(hrf_index, minlength=n_hrfs).cpu().numpy()
    weighted_pcs = (hrf_counts * optimal_pcs_per_hrf).sum() / max(hrf_counts.sum(), 1)
    global_optimal_pcs = int(round(weighted_pcs))

    if verbose:
        print()
        print("=" * 70)
        print("HRF + Denoising Selection Summary")
        print("=" * 70)
        print(f"  Global optimal PCs: {global_optimal_pcs}")
        print(f"  HRF usage distribution: {hrf_counts.tolist()}")
        print(f"  Mean denoised xval R²: {xval_r2_best.mean().item():.4f}")
        print(f"  Median denoised xval R²: {xval_r2_best.median().item():.4f}")
        print()

    # =========================================================================
    # Compute FINAL fit: voxel-wise optimal HRFs + optimal noise PCs
    # =========================================================================
    if verbose:
        print("Computing final fit with optimal HRFs and denoising...")

    # Build voxel-wise optimal design by grouping voxels by HRF
    unique_hrfs = torch.unique(hrf_index)

    # Storage for final results
    final_betas = torch.zeros(n_voxels, n_conditions, device=device)
    final_r2 = torch.zeros(n_voxels, device=device)
    final_tstats = torch.zeros(n_voxels, n_conditions, device=device)
    final_sigma2 = torch.zeros(n_voxels, device=device)

    # Build noise PC design (block-diagonal, same for all HRFs)
    if global_optimal_pcs > 0 and best_noise_pcs is not None:
        pc_blocks = []
        for run_idx in range(n_runs):
            pcs_run = best_noise_pcs[run_idx]
            assert isinstance(
                pcs_run, torch.Tensor
            )  # extract_noise_pcs_per_run(return_loadings=False)
            n_use = min(global_optimal_pcs, pcs_run.shape[1])
            pc_blocks.append(pcs_run[:, :n_use].to(device))
        noise_pc_design = torch.block_diag(*pc_blocks)
    else:
        noise_pc_design = None

    # Fit each HRF group
    hrf_group_iterator = (
        tqdm(unique_hrfs, desc="Final fit per HRF group") if verbose else unique_hrfs
    )

    for hrf_idx_t in hrf_group_iterator:
        hrf_idx_int = hrf_idx_t.item()
        voxel_mask_group = hrf_index == hrf_idx_t
        voxel_indices = torch.where(voxel_mask_group)[0]
        n_group_voxels = len(voxel_indices)

        if n_group_voxels == 0:
            continue

        # Get data for this group
        if data.device.type == "cpu" and voxel_indices.device.type != "cpu":
            voxel_indices_cpu = voxel_indices.cpu()
            group_data = data[voxel_indices_cpu, :]
        else:
            group_data = data[voxel_indices, :]

        # Build design with this HRF
        hrf = hrf_library[hrf_idx_int]
        stim_design = convolve_hrf_microtime(
            onset_matrix,
            hrf,
            n_timepoints,
            tr=tr,
            microtime_dt=microtime_dt,
            device=device,
        )

        # Combine: [task | nuisance | noise_pcs]
        components = [stim_design, nuisance_block_diag]
        if noise_pc_design is not None:
            components.append(noise_pc_design)
        full_design = torch.cat(components, dim=1)

        # Fit GLM
        group_results = fit_glm(
            group_data,
            full_design,
            tr=tr,
            max_poly_degree=0,
            device=device,
            verbose=False,
            task_indices=list(range(n_conditions)),
            preload_data_to_device=(group_data.device == device),
        )

        # Store results
        if group_results.betas is not None:
            betas_gpu = (
                group_results.betas.to(device)
                if group_results.betas.device != device
                else group_results.betas
            )
            final_betas[voxel_indices, :] = betas_gpu
        if group_results.r2 is not None:
            r2_gpu = (
                group_results.r2.to(device)
                if group_results.r2.device != device
                else group_results.r2
            )
            final_r2[voxel_indices] = r2_gpu
        if group_results.tstats is not None:
            tstats_gpu = (
                group_results.tstats.to(device)
                if group_results.tstats.device != device
                else group_results.tstats
            )
            final_tstats[voxel_indices, :] = tstats_gpu
        if group_results.sigma2 is not None:
            sigma2_gpu = (
                group_results.sigma2.to(device)
                if group_results.sigma2.device != device
                else group_results.sigma2
            )
            final_sigma2[voxel_indices] = sigma2_gpu

    # Build final GLMResults
    final_glm_results = GLMResults()
    final_glm_results.betas = final_betas.cpu()
    final_glm_results.r2 = final_r2.cpu()
    final_glm_results.tstats = final_tstats.cpu()
    final_glm_results.sigma2 = final_sigma2.cpu()
    final_glm_results.meanvol = data.mean(dim=1).cpu()
    final_glm_results.dof = n_timepoints - full_design.shape[1]  # Approximate

    if verbose:
        print(f"  Final full-fit R²: {final_glm_results.r2.mean().item():.4f}")
        r2_improvement = final_glm_results.r2.mean().item() - initial_glm_results.r2.mean().item()
        print(f"  R² improvement over initial: {r2_improvement:+.4f}")
        print()

    # Build metadata
    metadata = {
        "n_hrfs": n_hrfs,
        "max_pcs": max_pcs,
        "global_optimal_pcs": global_optimal_pcs,
        "optimal_pcs_per_hrf": optimal_pcs_per_hrf.tolist(),
        "r2_threshold": r2_threshold,
        "pcstop": pcstop,
        "tr": tr,
        "n_voxels": n_voxels,
        "n_timepoints": n_timepoints,
        "n_runs": n_runs,
        "n_cv_splits": len(cv_splits),
        "n_conditions": n_conditions,
        "hrf_usage_counts": hrf_counts.tolist(),
        "initial_xval_r2_mean": float(initial_xval_r2.mean().item()),
        "final_xval_r2_mean": float(xval_r2_best.mean().item()),
        "initial_r2_mean": float(initial_glm_results.r2.mean().item()),
        "final_r2_mean": float(final_glm_results.r2.mean().item()),
    }

    return PathfinderResults(
        hrf_index=hrf_index.cpu(),
        optimal_n_pcs=global_optimal_pcs,
        xval_r2_best=xval_r2_best.cpu(),
        xval_r2_all_hrfs=xval_r2_all_hrfs.cpu(),
        xval_r2_by_hrf_and_pcs=xval_r2_by_hrf_and_pcs,
        noise_pool_mask=best_noise_pool_mask.cpu() if best_noise_pool_mask is not None else None,
        criteria_mask=best_criteria_mask.cpu() if best_criteria_mask is not None else None,
        noise_pcs_per_run=best_noise_pcs if best_noise_pcs is not None else [],
        initial_results=initial_glm_results,
        initial_xval_r2=initial_xval_r2,
        final_results=final_glm_results,
        hrf_library=hrf_library.cpu(),
        metadata=metadata,
    )


def save_pathfinder_results(
    results: PathfinderResults,
    output_prefix: str,
    volume_shape: tuple[int, int, int],
    affine: np.ndarray,
    voxel_mask: torch.Tensor | None = None,
    condition_labels: list[str] | None = None,
    save_all_hrfs: bool = False,
    save_plots: bool = False,
    nii_ext: str = ".nii.gz",
) -> dict[str, str]:
    """Save pathfinder results to disk."""
    output_files = {}
    voxel_mask_np = voxel_mask.cpu().numpy() if voxel_mask is not None else None

    def to_volume(flat_data):
        if isinstance(flat_data, torch.Tensor):
            flat_data = flat_data.cpu().numpy()
        if voxel_mask_np is not None:
            vol = np.zeros(voxel_mask_np.shape[0], dtype=flat_data.dtype)
            vol[voxel_mask_np] = flat_data
        else:
            vol = flat_data
        return vol.reshape(volume_shape)

    # Generate condition labels if not provided
    n_conditions = results.metadata.get("n_conditions", 1)
    if condition_labels is None:
        condition_labels = [f"cond{i + 1:02d}" for i in range(n_conditions)]

    # =========================================================================
    # INITIAL RESULTS (canonical HRF, no denoising)
    # =========================================================================

    # Initial xval R² (canonical HRF, no denoising)
    if results.initial_xval_r2 is not None:
        initial_xval_r2_vol = to_volume(results.initial_xval_r2.numpy())
        initial_xval_r2_path = f"{output_prefix}_initial_xval_r2{nii_ext}"
        save_nifti(
            initial_xval_r2_vol.astype(np.float32), output_path=initial_xval_r2_path, affine=affine
        )
        output_files["initial_xval_r2"] = initial_xval_r2_path

    # Initial stats (betas, t-stats) with AFNI-style labels
    initial_results_obj = results.initial_results
    if initial_results_obj is not None:
        initial_results_obj.original_shape = volume_shape
        initial_results_obj.affine = affine
        if voxel_mask is not None:
            initial_results_obj.voxel_mask = voxel_mask

        initial_stats_path = f"{output_prefix}_initial_stats{nii_ext}"
        with spinner(f"Writing {Path(initial_stats_path).name}"):
            write_glm_bucket_as_nifti(
                initial_results_obj,
                initial_stats_path,
                condition_names=condition_labels,
                volume_shape=volume_shape,
                affine=affine,
                apply_afni_metadata=True,
            )
        output_files["initial_stats"] = initial_stats_path

        # Also save initial R² (full-fit, not xval)
        assert initial_results_obj.r2 is not None
        initial_r2_vol = to_volume(initial_results_obj.r2.numpy())
        initial_r2_path = f"{output_prefix}_initial_r2{nii_ext}"
        save_nifti(initial_r2_vol.astype(np.float32), output_path=initial_r2_path, affine=affine)
        output_files["initial_r2"] = initial_r2_path

    # =========================================================================
    # FINAL RESULTS (optimal HRFs + denoising)
    # =========================================================================

    # Final stats (betas, t-stats) with AFNI-style labels
    final_results_obj = results.final_results
    if final_results_obj is not None:
        final_results_obj.original_shape = volume_shape
        final_results_obj.affine = affine
        if voxel_mask is not None:
            final_results_obj.voxel_mask = voxel_mask

        final_stats_path = f"{output_prefix}_final_stats{nii_ext}"
        with spinner(f"Writing {Path(final_stats_path).name}"):
            write_glm_bucket_as_nifti(
                final_results_obj,
                final_stats_path,
                condition_names=condition_labels,
                volume_shape=volume_shape,
                affine=affine,
                apply_afni_metadata=True,
            )
        output_files["final_stats"] = final_stats_path

        # Also save final R² (full-fit, not xval)
        assert final_results_obj.r2 is not None
        final_r2_vol = to_volume(final_results_obj.r2.numpy())
        final_r2_path = f"{output_prefix}_final_r2{nii_ext}"
        save_nifti(final_r2_vol.astype(np.float32), output_path=final_r2_path, affine=affine)
        output_files["final_r2"] = final_r2_path

    # =========================================================================
    # HRF SELECTION AND DENOISING OUTPUTS
    # =========================================================================

    # 1. HRF index (1-indexed for AFNI)
    hrf_index_vol = to_volume((results.hrf_index.float() + 1.0).numpy())
    hrf_index_path = f"{output_prefix}_hrf_index{nii_ext}"
    save_nifti(hrf_index_vol.astype(np.float32), output_path=hrf_index_path, affine=affine)
    output_files["hrf_index"] = hrf_index_path

    # 2. Best denoised xval R² (final, optimized)
    xval_r2_vol = to_volume(results.xval_r2_best.numpy())
    xval_r2_path = f"{output_prefix}_final_xval_r2{nii_ext}"
    save_nifti(xval_r2_vol.astype(np.float32), output_path=xval_r2_path, affine=affine)
    output_files["final_xval_r2"] = xval_r2_path

    # 3. Noise pool mask
    if results.noise_pool_mask is not None:
        noise_pool_vol = to_volume(results.noise_pool_mask.numpy().astype(np.float32))
        noise_pool_path = f"{output_prefix}_noise_pool_mask{nii_ext}"
        save_nifti(noise_pool_vol, output_path=noise_pool_path, affine=affine)
        output_files["noise_pool_mask"] = noise_pool_path

    # 4. Criteria mask
    if results.criteria_mask is not None:
        criteria_vol = to_volume(results.criteria_mask.numpy().astype(np.float32))
        criteria_path = f"{output_prefix}_criteria_mask{nii_ext}"
        save_nifti(criteria_vol, output_path=criteria_path, affine=affine)
        output_files["criteria_mask"] = criteria_path

    # 5. All HRFs denoised R² (4D)
    if save_all_hrfs:
        n_hrfs = results.xval_r2_all_hrfs.shape[1]
        all_hrfs_vols = []
        for hrf_idx in range(n_hrfs):
            vol = to_volume(results.xval_r2_all_hrfs[:, hrf_idx].numpy())
            all_hrfs_vols.append(vol)
        all_hrfs_4d = np.stack(all_hrfs_vols, axis=-1)
        all_hrfs_path = f"{output_prefix}_xval_r2_all_hrfs{nii_ext}"
        with spinner(f"Writing {Path(all_hrfs_path).name}"):
            save_nifti(all_hrfs_4d.astype(np.float32), output_path=all_hrfs_path, affine=affine)
        output_files["xval_r2_all_hrfs"] = all_hrfs_path

    # 6. Noise PCs
    if results.noise_pcs_per_run:
        pcs_path = f"{output_prefix}_noise_pcs.pt"
        with spinner(f"Writing {Path(pcs_path).name}"):
            torch.save(
                {
                    "noise_pcs_per_run": results.noise_pcs_per_run,
                    "optimal_n_pcs": results.optimal_n_pcs,
                },
                pcs_path,
            )
        output_files["noise_pcs"] = pcs_path

    # 7. Optimization curve (HRF x PC)
    curve_path = f"{output_prefix}_optimization_curve.npy"
    np.save(curve_path, results.xval_r2_by_hrf_and_pcs)
    output_files["optimization_curve"] = curve_path

    # 8. Plots
    if save_plots:
        # Create figures directory
        figs_dir = f"{output_prefix}_figures"
        Path(figs_dir).mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Optimization surface plot
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(
                results.xval_r2_by_hrf_and_pcs,
                aspect="auto",
                cmap="viridis",
                origin="lower",
            )
            ax.set_xlabel("Number of Noise PCs")
            ax.set_ylabel("HRF Index")
            ax.set_title("Joint HRF + Denoising Optimization Surface\n(Mean Denoised CV R²)")
            plt.colorbar(im, ax=ax, label="Mean CV R²")

            # Mark optimal points per HRF
            for hrf_idx, opt_pcs in enumerate(results.metadata["optimal_pcs_per_hrf"]):
                ax.scatter(opt_pcs, hrf_idx, marker="x", color="red", s=50)

            plt.tight_layout()
            surface_path = f"{figs_dir}/optimization_surface.png"
            fig.savefig(surface_path, dpi=150)
            plt.close(fig)
            output_files["optimization_surface_plot"] = surface_path

            # HRF usage histogram
            fig, ax = plt.subplots(figsize=(8, 5))
            hrf_counts = results.metadata["hrf_usage_counts"]
            ax.bar(range(len(hrf_counts)), hrf_counts)
            ax.set_xlabel("HRF Index")
            ax.set_ylabel("Voxel Count")
            ax.set_title("HRF Selection Distribution")
            plt.tight_layout()
            hist_path = f"{figs_dir}/hrf_histogram.png"
            fig.savefig(hist_path, dpi=150)
            plt.close(fig)
            output_files["hrf_histogram_plot"] = hist_path

        except Exception as e:
            print(f"  Warning: Could not create plots: {e}")

    # 9. Metadata JSON
    metadata_path = f"{output_prefix}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(results.metadata, f, indent=2)
    output_files["metadata"] = metadata_path

    return output_files


def main():
    parser = create_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    pfx = parse_prefix(args.prefix)
    args.prefix = pfx.stem  # overwrite with clean stem
    _nii_ext = pfx.nifti_ext

    print_header()

    # Parse inputs
    input_files = parse_input_files(args.input)
    n_runs = len(input_files)

    if n_runs < 2:
        print("ERROR: At least 2 runs required for cross-validation")
        sys.exit(1)

    onset_files = args.onsets
    n_conditions = len(onset_files)
    from fastfuncstuff.cli_utils import clean_condition_labels

    condition_labels = clean_condition_labels([Path(f).stem for f in onset_files])

    for f in onset_files:
        if not Path(f).exists():
            print(f"ERROR: Onset file not found: {f}")
            sys.exit(1)

    durations = parse_durations(args.durations, n_conditions, condition_labels)
    print(f"  Durations: {durations}s")

    cv_strategy = parse_cv_strategy(args.cv_strategy)

    # Setup device
    device = setup_device(args.device)
    print(f"  Device: {device}")

    # Load data
    print()
    print("Loading data...")

    mask = None
    if args.mask:
        mask = load_afni_mask(args.mask)
        print(f"  Mask: {args.mask} ({mask.sum():,} voxels)")

    first_img = load_nifti(input_files[0])
    affine = np.array(first_img.affine) if hasattr(first_img, "affine") else np.eye(4)
    volume_shape = tuple(first_img.shape[:3])
    # Voxel sizes from pixdim (get_zooms), orientation-independent. The affine
    # diagonal is wrong for permuted/oblique grids (e.g. RSP/LIA) where the size
    # sits off-diagonal; fall back to affine column norms if pixdim is unset.
    if hasattr(first_img, "header"):
        voxel_sizes = tuple(float(z) for z in first_img.header.get_zooms()[:3])
    else:
        voxel_sizes = (0.0, 0.0, 0.0)
    if any(v == 0 for v in voxel_sizes):
        voxel_sizes = tuple(np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))

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
        print(f"\n  Large dataset ({data_size_gb:.2f} GB), loading to CPU")
    else:
        keep_on_cpu = False

    # Load and optionally blur. The blur rides inside the shared loader as a
    # per-run callback rather than forking into a second, serial load path.
    mask_flat = mask.flatten().astype(bool) if mask is not None else None

    per_run_fn = None
    if args.do_blur is not None:
        print(f"\nApplying Gaussian blur (FWHM = {args.do_blur} mm)...")

        def per_run_fn(run_data, _run_idx):
            # (n_voxels, n_tps) in C order and pre-mask, so the view back to
            # (x, y, z, t) is free and covers the whole volume.
            n_tps = run_data.shape[1]
            blurred = gaussian_blur_3d(
                run_data.cpu().numpy().reshape(*volume_shape, n_tps),
                fwhm_mm=args.do_blur,
                voxel_sizes=voxel_sizes,
                device=device,
                verbose=False,
            )
            return torch.from_numpy(blurred.reshape(-1, n_tps)).to(run_data.device)

    data, run_starts = load_and_concatenate_runs(
        [Path(f) for f in input_files],
        device=device,
        keep_on_cpu=keep_on_cpu,
        mask_flat=mask_flat,
        per_run_fn=per_run_fn,
    )

    # Get TR
    if args.tr is None:
        zooms = first_img.header.get_zooms()
        if len(zooms) > 3 and zooms[3] > 0:
            args.tr = float(zooms[3])
            print(f"  TR (from header): {args.tr}s")
        else:
            print("ERROR: Could not determine TR. Please specify with -tr")
            sys.exit(1)
    else:
        print(f"  TR (specified): {args.tr}s")

    n_voxels, n_timepoints = data.shape

    # Optional scaling
    if args.do_scale:
        print()
        data, _, _ = scale_to_percent_signal(
            data=data, run_starts=run_starts, max_scale=200.0, verbose=True
        )

    print(f"  Data shape: {data.shape}")
    print(f"  Volume shape: {volume_shape}")
    print(f"  Runs: {n_runs} starting at {run_starts}")

    # Build onset matrix
    print()
    print("Building onset matrix...")

    all_onsets = []
    for onset_file in onset_files:
        onsets_by_run = parse_afni_timing_file(onset_file)
        if len(onsets_by_run) != n_runs:
            print(
                f"ERROR: Onset file {onset_file} has {len(onsets_by_run)} runs, expected {n_runs}"
            )
            sys.exit(1)
        all_onsets.append(onsets_by_run)

    bins_per_tr = int(round(args.tr / args.microtime_dt))
    n_microtime = n_timepoints * bins_per_tr
    onset_matrix = torch.zeros((n_microtime, n_conditions), device=device)

    for cond_idx in range(n_conditions):
        duration_bins = max(1, int(np.round(durations[cond_idx] / args.microtime_dt)))
        for run_idx in range(n_runs):
            onsets = all_onsets[cond_idx][run_idx]
            run_start_tr = run_starts[run_idx]
            run_start_micro = run_start_tr * bins_per_tr
            for onset_time in onsets:
                onset_bin = run_start_micro + int(np.round(onset_time / args.microtime_dt))
                if onset_bin < n_microtime:
                    onset_matrix[
                        onset_bin : min(onset_bin + duration_bins, n_microtime), cond_idx
                    ] = 1.0

    print(f"  Onset matrix shape: {onset_matrix.shape}")

    # Build nuisance regressors per run
    print()
    print("Building nuisance regressors...")

    if args.polort is None:
        run_lengths = compute_run_lengths(run_starts, n_timepoints)
        avg_run_duration_sec = get_average_run_duration(run_lengths, args.tr)
        polort = auto_polort(avg_run_duration_sec, formula="afni")
    else:
        polort = args.polort

    nuisance_per_run = []
    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
        run_length = end_tp - start_tp
        poly_block = construct_polynomial_matrix(run_length, polort, device)
        nuisance_per_run.append(poly_block)

    # Add user nuisance blocks (-ortvec / -ortvec_run / -ortvec_glob).
    user_blocks = collect_nuisance_blocks(args, run_starts, n_timepoints, verbose=True)
    for block in user_blocks:
        if block.n_columns == 0:
            continue
        for run_idx in range(n_runs):
            start_tp = run_starts[run_idx]
            end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            run_length = end_tp - start_tp
            m = block.get_run(run_idx, run_length).copy()
            col_mean = m.mean(axis=0, keepdims=True)
            if np.max(np.abs(col_mean)) > 1e-4:
                m = m - col_mean
            run_nuisance = torch.from_numpy(m).to(
                device=device,
                dtype=nuisance_per_run[run_idx].dtype,
            )
            nuisance_per_run[run_idx] = torch.cat([nuisance_per_run[run_idx], run_nuisance], dim=1)

    print(f"  Nuisance per run: {nuisance_per_run[0].shape[1]} columns (polort={polort})")

    # Generate HRF library
    print()
    print(f"Generating {args.hrf_mode} HRF library ({args.n_hrfs} candidates)...")

    hrf_library = get_hrf_library(
        mode=args.hrf_mode,
        stim_duration=0.0,  # Impulse response
        microtime_dt=args.microtime_dt,
        n_hrfs=args.n_hrfs,
        device=device,
    )

    print(f"  HRF library shape: {hrf_library.shape}")

    # Generate CV splits
    cv_splits = generate_cv_splits(n_runs, strategy=cv_strategy, n_perms=args.n_perms)

    # Run joint optimization
    print()
    results = fit_pathfinder(
        data=data,
        onset_matrix=onset_matrix,
        hrf_library=hrf_library,
        tr=args.tr,
        run_starts=run_starts,
        nuisance_per_run=nuisance_per_run,
        cv_splits=cv_splits,
        r2_threshold=args.r2_threshold,
        max_pcs=args.max_pcs,
        variance_threshold=args.variance_threshold,
        min_noise_voxels=args.min_noise_voxels,
        max_noise_fraction=args.max_noise_fraction,
        pcstop=args.pcstop,
        microtime_dt=args.microtime_dt,
        metric=args.metric,
        device=device,
        chunk_size=args.batch_size,
        verbose=args.verb >= 1,
    )

    # Save outputs
    print()
    print("Saving outputs...")

    output_dir = Path(args.prefix).parent
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    voxel_mask = None
    if mask is not None:
        voxel_mask = torch.from_numpy(mask.flatten().astype(bool))

    output_files = save_pathfinder_results(
        results=results,
        output_prefix=args.prefix,
        volume_shape=volume_shape,
        affine=affine,
        voxel_mask=voxel_mask,
        condition_labels=condition_labels,
        save_all_hrfs=args.save_all_hrfs,
        save_plots=args.plots,
        nii_ext=_nii_ext,
    )

    print()
    print("=" * 70)
    print("Output Files")
    print("=" * 70)
    for output_type, filepath in output_files.items():
        print(f"  {output_type}: {filepath}")
    print("=" * 70)

    print()
    print("=" * 70)
    print("ffs_pathfinder Complete!")
    print("=" * 70)
    print(f"  Global optimal PCs: {results.optimal_n_pcs}")
    print(f"  Mean denoised xval R²: {results.xval_r2_best.mean().item():.4f}")
    print(f"  Median denoised xval R²: {results.xval_r2_best.median().item():.4f}")
    print()
    print("  HRF usage distribution:")
    for i, count in enumerate(results.metadata["hrf_usage_counts"]):
        if count > 0:
            pct = 100 * count / n_voxels
            print(f"    HRF {i + 1}: {count:,} voxels ({pct:.1f}%)")
    print()
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
