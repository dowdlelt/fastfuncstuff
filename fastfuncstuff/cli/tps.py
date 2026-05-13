#!/usr/bin/env python3

"""

Thin Plate Splines (TPS) for fMRI HRF Estimation with Cross-Validation

Estimates HRF using penalized splines with automatic smoothness selection.
Uses cross-validation (LORO or split-half) to choose optimal smoothness.

Key features:
- Penalized cubic splines (P-splines) for smooth HRF estimation
- Cross-validated smoothness parameter (λ) selection
- Global optimization (one λ for all voxels) or per-voxel optimization
- Adapts to local SNR: high SNR → less smoothing, low SNR → more smoothing
- Supports per-condition windows for different stimulus durations
- GPU-accelerated with chunking for memory efficiency

References:
- Chen, Taylor, Reynolds, Leibenluft, Pine, Brotman, Pagliaccio & Haller (2023):
  "BOLD response is more than just magnitude: improving detection sensitivity
  through capturing hemodynamic profiles." NeuroImage 277:120224.
  https://www.sciencedirect.com/science/article/pii/S1053811923003750
- AFNI's 3dMSS.R for group-level TPS

Usage:
    ffs_tps.py -input func.nii.gz \\
               -stim_times face.txt house.txt \\
               -stim_labels face house \\
               -tps_window 0,15 0,20 \\
               -n_knots 10 \\
               -optimize_level global \\
               -output_prefix tps_results

Examples:
    # Global λ optimization (fast, good baseline)
    ffs_tps.py -input func.nii.gz -stim_times onsets.txt \\
               -tps_window 0,20 -n_knots 15 -optimize_level global

    # Per-voxel λ optimization (adaptive to local SNR)
    ffs_tps.py -input func.nii.gz -stim_times onsets.txt \\
               -tps_window 0,20 -n_knots 15 -optimize_level per_voxel

    # Custom λ search grid
    ffs_tps.py -input func.nii.gz -stim_times onsets.txt \\
               -tps_window 0,20 -lambda_values 0.01 0.1 1 10 100

Author: Logan Thomas
Date: 2026-01-27
"""
from __future__ import annotations

import argparse

import os
import sys

import numpy as np
import torch

from fastfuncstuff.cli_utils import add_verbose_arg, parse_prefix
from fastfuncstuff.io.afni import (
    load_nifti,
    save_nifti,
)
from fastfuncstuff.design.matrices import (
    fit_penalized_glm,
    fit_penalized_glm_cv,
    make_penalty_matrix,
    make_tps_design,
)
from fastfuncstuff.design.builder import legendre_polynomials, parse_afni_timing_file
from fastfuncstuff.utils import configure_torch_backends, get_device


def parse_tps_windows(tps_window_args, n_conditions):
    """
    Parse TPS window specifications

    Supports:
    - Single window: '0 15' or '0,15'
    - Per-condition: '0,15 0,20 0,25'

    Parameters
    ----------
    tps_window_args : list of str
        Command-line arguments for -tps_window
    n_conditions : int
        Number of stimulus conditions

    Returns
    -------
    windows : list of tuple
        [(bot1, top1), (bot2, top2), ...] for each condition

    Examples
    --------
    >>> parse_tps_windows(['0', '15'], 3)
    [(0.0, 15.0), (0.0, 15.0), (0.0, 15.0)]

    >>> parse_tps_windows(['0,15', '0,20'], 2)
    [(0.0, 15.0), (0.0, 20.0)]
    """
    if not tps_window_args:
        raise ValueError("Must specify -tps_window")

    # Check if single window or per-condition
    # Join args in case user did: -tps_window 0 15 (splits into ['0', '15'])
    window_str = ' '.join(tps_window_args)

    # Check if comma-separated (per-condition format)
    if ',' in window_str:
        # Per-condition format: '0,15 0,20' or '0,15'
        pairs = window_str.split()
        if len(pairs) == 1:
            # Single window, replicate for all conditions
            bot, top = pairs[0].split(',')
            window = (float(bot), float(top))
            return [window] * n_conditions
        elif len(pairs) == n_conditions:
            # Per-condition windows
            windows = []
            for pair in pairs:
                bot, top = pair.split(',')
                windows.append((float(bot), float(top)))
            return windows
        else:
            raise ValueError(
                f"Got {len(pairs)} window specifications but {n_conditions} conditions. "
                "Use one window for all conditions or one per condition."
            )
    else:
        # Space-separated format: '0 15'
        parts = window_str.split()
        if len(parts) == 2:
            # Single window
            bot, top = float(parts[0]), float(parts[1])
            return [(bot, top)] * n_conditions
        else:
            raise ValueError(
                f"Invalid window format: '{window_str}'. "
                "Use '0 15', '0,15', or '0,15 0,20' (per-condition)"
            )


def parse_n_knots(n_knots_args, n_conditions):
    """
    Parse number of knots (basis functions) per condition

    Supports:
    - Single value: applies to all conditions
    - Per-condition: one value per condition

    Parameters
    ----------
    n_knots_args : int or list of int
        Command-line arguments for -n_knots
    n_conditions : int
        Number of stimulus conditions

    Returns
    -------
    n_knots_list : list of int
        Number of knots for each condition
    """
    if isinstance(n_knots_args, int):
        # Single value
        return [n_knots_args] * n_conditions
    elif len(n_knots_args) == 1:
        # Single value passed as list
        return [n_knots_args[0]] * n_conditions
    elif len(n_knots_args) == n_conditions:
        # Per-condition
        return n_knots_args
    else:
        raise ValueError(
            f"Got {len(n_knots_args)} n_knots values but {n_conditions} conditions. "
            "Use one value for all conditions or one per condition."
        )


def main():
    class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        """Show defaults while preserving raw description formatting."""

    parser = argparse.ArgumentParser(
        description='TPS HRF estimation with cross-validated smoothness selection',
        formatter_class=_HelpFormatter,
        epilog=__doc__
    )

    # Input/output
    parser.add_argument('-input', required=True, help='Input fMRI data (4D NIfTI)')
    parser.add_argument('-mask', help='Brain mask (3D NIfTI), auto-generated if not provided')
    parser.add_argument('-output_prefix', required=True, help='Output file prefix')

    # Stimulus timing
    parser.add_argument('-stim_times', nargs='+', required=True,
                        help='Onset time files (one per condition)')
    parser.add_argument('-stim_labels', nargs='+', required=True,
                        help='Condition labels (must match number of -stim_times files)')

    # TPS parameters
    parser.add_argument('-tps_window', nargs='+', required=True,
                        help='Estimation window: "0 15", "0,15", or "0,15 0,20" (per-condition)')
    parser.add_argument('-n_knots', type=int, default=None,
                        help='Number of basis functions (knots). Default: auto (window_duration / TR)')
    parser.add_argument('-force_zero_edges', action='store_true',
                        help='Force HRF to be zero at window start/end')

    # Cross-validation
    parser.add_argument('-optimize_level', choices=['global', 'per_voxel'], default='global',
                        help='Smoothness optimization: global (one λ) or per_voxel (adaptive). Default: global')
    parser.add_argument('-lambda_values', nargs='+', type=float, default=None,
                        help='Lambda search grid (e.g., "0.01 0.1 1 10 100"). Default: auto (logarithmic grid)')
    parser.add_argument('-cv_method', choices=['loro', 'split_half'], default='loro',
                        help='Cross-validation method. Default: loro (leave-one-run-out)')

    # Nuisance regressors
    parser.add_argument('-polort', type=int, default=3,
                        help='Polynomial detrending order (-1 = none). Default: 3')

    # Computational
    parser.add_argument('-device', choices=['cpu', 'cuda', 'auto'], default='auto',
                        help='Computation device. Default: auto')
    parser.add_argument('-chunk_size', type=int, default=60000,
                        help='Voxels per chunk for GPU memory management. Default: 60000')

    # Output options
    parser.add_argument('-save_design', action='store_true',
                        help='Save design matrix as .1D file')
    parser.add_argument('-save_lambda_map', action='store_true',
                        help='Save optimal lambda values as NIfTI (per_voxel mode only)')
    add_verbose_arg(parser, default=0)

    args = parser.parse_args()

    pfx = parse_prefix(args.output_prefix)
    args.output_prefix = pfx.stem
    _nii_ext = pfx.nifti_ext

    # ========================================================================
    # Validate inputs
    # ========================================================================

    if len(args.stim_times) != len(args.stim_labels):
        parser.error(f"Number of -stim_times ({len(args.stim_times)}) must match "
                     f"number of -stim_labels ({len(args.stim_labels)})")

    n_conditions = len(args.stim_times)

    # ========================================================================
    # Setup
    # ========================================================================

    if args.verb >= 1:
        print("=" * 70)
        print("TPS HRF Estimation with Cross-Validated Smoothness Selection")
        print("=" * 70)

    # Device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    configure_torch_backends(device)

    if args.verb >= 1:
        print(f"\nDevice: {device}")

    # ========================================================================
    # Load data
    # ========================================================================

    if args.verb >= 1:
        print(f"\nLoading data: {args.input}")

    # Load NIfTI data
    img = load_nifti(args.input)
    data_full = np.array(img.dataobj)
    affine = img.affine
    header = img.header

    nx, ny, nz, n_timepoints_total = data_full.shape

    if args.verb >= 1:
        print(f"  Shape: {data_full.shape}")
        print(f"  Voxels: {nx} x {ny} x {nz} = {nx*ny*nz:,}")
        print(f"  Timepoints: {n_timepoints_total}")

    # Get TR from header
    tr = float(header.get_zooms()[3])
    if args.verb >= 1:
        print(f"  TR: {tr}s")

    # ========================================================================
    # Create or load mask
    # ========================================================================

    if args.mask:
        if args.verb >= 1:
            print(f"\nLoading mask: {args.mask}")
        mask_3d = load_nifti(args.mask)
        if mask_3d.shape != (nx, ny, nz):
            raise ValueError(f"Mask shape {mask_3d.shape} doesn't match data {(nx, ny, nz)}")
        mask = mask_3d.astype(bool)
    else:
        if args.verb >= 1:
            print("\nGenerating brain mask (non-zero voxels)...")
        # Simple mask: non-zero in at least 10% of timepoints
        nonzero_count = np.sum(data_full != 0, axis=3)
        mask = nonzero_count > (0.1 * n_timepoints_total)

    n_voxels = int(np.sum(mask))
    if args.verb >= 1:
        print(f"  Brain voxels: {n_voxels:,} ({100*n_voxels/(nx*ny*nz):.1f}%)")

    # ========================================================================
    # Parse stimulus timing
    # ========================================================================

    if args.verb >= 1:
        print(f"\nParsing stimulus timing ({n_conditions} conditions)...")

    # Parse onset files
    onsets_per_condition = []  # [condition][run] -> onset times
    n_runs = None

    for cond_idx, onset_file in enumerate(args.stim_times):
        label = args.stim_labels[cond_idx]
        if args.verb >= 1:
            print(f"  {label}: {onset_file}")

        onsets = parse_afni_timing_file(onset_file)
        onsets_per_condition.append(onsets)

        if n_runs is None:
            n_runs = len(onsets)
        elif len(onsets) != n_runs:
            raise ValueError(
                f"Condition '{label}' has {len(onsets)} runs, "
                f"but first condition has {n_runs} runs"
            )

        n_events = sum(len(run_onsets) for run_onsets in onsets)
        if args.verb >= 1:
            print(f"    {n_runs} runs, {n_events} total events")

    # ========================================================================
    # Parse TPS windows
    # ========================================================================

    windows = parse_tps_windows(args.tps_window, n_conditions)

    if args.verb >= 1:
        print("\nTPS estimation windows:")
        for cond_idx, (bot, top) in enumerate(windows):
            label = args.stim_labels[cond_idx]
            duration = top - bot
            print(f"  {label}: [{bot}s, {top}s] (duration: {duration}s)")

    # ========================================================================
    # Determine number of knots per condition
    # ========================================================================

    if args.n_knots is None:
        # Auto: roughly one knot per TR in the window
        n_knots_list = []
        for bot, top in windows:
            duration = top - bot
            n_knots = max(5, int(np.ceil(duration / tr)))  # At least 5 knots
            n_knots_list.append(n_knots)

        if args.verb >= 1:
            print("\nAuto-selecting number of knots (~ 1 per TR):")
            for cond_idx, n_knots in enumerate(n_knots_list):
                label = args.stim_labels[cond_idx]
                print(f"  {label}: {n_knots} knots")
    else:
        n_knots_list = [args.n_knots] * n_conditions
        if args.verb >= 1:
            print(f"\nNumber of knots: {args.n_knots} (all conditions)")

    # ========================================================================
    # Build TPS design matrices (one per condition)
    # ========================================================================

    if args.verb >= 1:
        print("\nBuilding TPS design matrices...")

    design_matrices = []
    n_basis_per_condition = []

    for cond_idx in range(n_conditions):
        label = args.stim_labels[cond_idx]
        bot, top = windows[cond_idx]
        n_knots = n_knots_list[cond_idx]

        if args.verb >= 1:
            print(f"  {label}: {n_knots} knots, window [{bot}s, {top}s]")

        # Concatenate onsets across runs
        onset_times_all_runs = np.concatenate(onsets_per_condition[cond_idx])

        # Create TPS design
        design_cond = make_tps_design(
            onset_times_list=[onset_times_all_runs],
            bot=bot,
            top=top,
            n_knots=n_knots,
            tr=tr,
            n_timepoints=n_timepoints_total,
            force_zero_edges=args.force_zero_edges,
            device=device,
        )

        n_basis = design_cond.shape[1]
        n_basis_per_condition.append(n_basis)

        design_matrices.append(design_cond)

        if args.verb >= 1:
            print(f"    Design shape: {design_cond.shape}")

    # Concatenate all conditions
    design_stimulus = torch.cat(design_matrices, dim=1)  # (n_timepoints, n_basis_total)
    n_stimulus_regressors = design_stimulus.shape[1]

    if args.verb >= 1:
        print(f"\n  Total stimulus regressors: {n_stimulus_regressors}")

    # ========================================================================
    # Add polynomial regressors (zero-padded per run)
    # ========================================================================

    if args.verb >= 1:
        print(f"\nAdding polynomial regressors (order {args.polort})...")

    # Infer run boundaries (assume equal length for now)
    # TODO: support variable-length runs via -num_stimts or similar
    n_timepoints_per_run = n_timepoints_total // n_runs
    remainder = n_timepoints_total % n_runs

    if remainder != 0:
        raise NotImplementedError(
            f"Unequal run lengths not yet supported. "
            f"Total TRs ({n_timepoints_total}) not divisible by n_runs ({n_runs}). "
            f"Please specify run lengths explicitly or concatenate equal-length runs."
        )

    run_boundaries = []
    for run_idx in range(n_runs):
        start_tr = run_idx * n_timepoints_per_run
        end_tr = start_tr + n_timepoints_per_run
        run_boundaries.append((start_tr, end_tr))

    if args.verb >= 1:
        print(f"  Detected {n_runs} runs of {n_timepoints_per_run} TRs each")

    # Create zero-padded polynomials
    if args.polort >= 0:
        n_poly_per_run = args.polort + 1
        total_poly_cols = n_runs * n_poly_per_run

        poly_full = np.zeros((n_timepoints_total, total_poly_cols))

        for run_idx in range(n_runs):
            start_tr, end_tr = run_boundaries[run_idx]
            n_tp = end_tr - start_tr

            # Generate polynomials for this run
            poly_run = legendre_polynomials(n_tp, args.polort)

            # Insert into zero-padded matrix
            col_start = run_idx * n_poly_per_run
            poly_full[start_tr:end_tr, col_start:col_start+n_poly_per_run] = poly_run

        poly_tensor = torch.from_numpy(poly_full).float().to(device)

        if args.verb >= 1:
            print(f"  Polynomial regressors: {total_poly_cols} ({n_poly_per_run} per run x {n_runs} runs)")
    else:
        poly_tensor = torch.zeros((n_timepoints_total, 0), device=device)
        total_poly_cols = 0

    # Concatenate stimulus and polynomial regressors
    design_full = torch.cat([design_stimulus, poly_tensor], dim=1)

    if args.verb >= 1:
        print(f"\nFull design matrix: {design_full.shape}")
        print(f"  Stimulus: {n_stimulus_regressors} regressors")
        print(f"  Polynomials: {total_poly_cols} regressors")

    # ========================================================================
    # Save design matrix (optional)
    # ========================================================================

    if args.save_design:
        design_file = f"{args.output_prefix}_design_TPS.1D"
        if args.verb >= 1:
            print(f"\nSaving design matrix: {design_file}")

        design_np = design_full.cpu().numpy()

        # Add header with metadata
        header_lines = [
            "# TPS design matrix",
            f"# Shape: {design_np.shape[0]} timepoints x {design_np.shape[1]} regressors",
            f"# TR: {tr}s",
            f"# Stimulus regressors: columns 0-{n_stimulus_regressors-1}",
            f"# Polynomial regressors: columns {n_stimulus_regressors}-{design_np.shape[1]-1}",
            "#",
        ]

        # Add per-condition metadata
        col_idx = 0
        for cond_idx in range(n_conditions):
            label = args.stim_labels[cond_idx]
            n_basis = n_basis_per_condition[cond_idx]
            bot, top = windows[cond_idx]
            header_lines.append(
                f"# {label}: columns {col_idx}-{col_idx+n_basis-1} "
                f"({n_basis} knots, window [{bot}s, {top}s])"
            )
            col_idx += n_basis

        header = '\n'.join(header_lines)

        np.savetxt(design_file, design_np, fmt='%.6f', header=header, comments='')

        if args.verb >= 1:
            print(f"  ✓ Saved: {design_file}")

    # ========================================================================
    # Prepare data for fitting
    # ========================================================================

    if args.verb >= 1:
        print("\nPreparing data for GLM fitting...")

    # Extract masked voxels: (n_voxels, n_timepoints)
    data_masked = data_full[mask, :]

    if args.verb >= 1:
        print(f"  Data shape: {data_masked.shape}")

    # Convert to tensor (keep on CPU for chunking)
    data_tensor = torch.from_numpy(data_masked).float()

    # ========================================================================
    # Lambda grid for cross-validation
    # ========================================================================

    if args.lambda_values is None:
        # Auto: logarithmic grid from 1e-3 to 1e3
        lambda_values = np.logspace(-3, 3, 13).tolist()
    else:
        lambda_values = args.lambda_values

    if args.verb >= 1:
        print(f"\nLambda search grid ({len(lambda_values)} values):")
        print(f"  {', '.join([f'{lam:.2e}' for lam in lambda_values])}")

    # ========================================================================
    # Create penalty matrix (block diagonal for multiple conditions)
    # ========================================================================

    if args.verb >= 1:
        print("\nCreating penalty matrix (2nd order differences)...")

    # Create block-diagonal penalty matrix for all conditions
    # Each condition gets its own penalty block
    penalty_blocks = []
    for n_basis in n_basis_per_condition:
        D_cond = make_penalty_matrix(n_basis, order=2)
        penalty_blocks.append(D_cond)

    # Stack into block diagonal
    # For simplicity, create full block diagonal matrix
    n_rows_total = sum(D.shape[0] for D in penalty_blocks)
    penalty_matrix = np.zeros((n_rows_total, n_stimulus_regressors))

    row_offset = 0
    col_offset = 0
    for D_cond in penalty_blocks:
        n_rows, n_cols = D_cond.shape
        penalty_matrix[row_offset:row_offset+n_rows, col_offset:col_offset+n_cols] = D_cond
        row_offset += n_rows
        col_offset += n_cols

    if args.verb >= 1:
        print(f"  Penalty matrix: {penalty_matrix.shape}")
        print(f"  Penalizes {n_stimulus_regressors} stimulus regressors across {n_conditions} conditions")

    # ========================================================================
    # Cross-validation for λ selection
    # ========================================================================

    if args.optimize_level == 'global':
        if args.verb >= 1:
            print("\n" + "="*70)
            print("Global λ Optimization (LORO Cross-Validation)")
            print("="*70)

        # Use only stimulus regressors for CV (polynomials are nuisance)
        design_cv = design_stimulus

        # Fit CV on subset of voxels for speed (e.g., 10k random voxels)
        n_voxels_cv = min(10000, n_voxels)
        if n_voxels_cv < n_voxels:
            if args.verb >= 1:
                print(f"\nUsing {n_voxels_cv:,} random voxels for CV (for speed)")
            cv_indices = np.random.choice(n_voxels, size=n_voxels_cv, replace=False)
            data_cv = data_tensor[cv_indices, :]
        else:
            data_cv = data_tensor

        best_lambda, cv_errors = fit_penalized_glm_cv(
            data=data_cv,
            design=design_cv,
            penalty_matrix=penalty_matrix,
            lambda_values=lambda_values,
            run_boundaries=run_boundaries,
            device=device,
            verbose=args.verb >= 1,
        )

        # Use this lambda for all voxels
        lambda_map = np.full(n_voxels, best_lambda)

        if args.verb >= 1:
            print(f"\n✓ Global λ = {best_lambda:.3e}")

    elif args.optimize_level == 'per_voxel':
        if args.verb >= 1:
            print("\n" + "="*70)
            print("Per-Voxel λ Optimization (Adaptive Smoothness)")
            print("="*70)

        # Fit each voxel independently
        # This is SLOW but maximally adaptive
        lambda_map = np.zeros(n_voxels)

        if args.verb >= 1:
            print(f"\nOptimizing λ for {n_voxels:,} voxels...")
            print("  (This may take a while...)")

        # Process in chunks
        chunk_size_cv = 1000  # Voxels per chunk for CV
        n_chunks = int(np.ceil(n_voxels / chunk_size_cv))

        design_cv = design_stimulus

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size_cv
            end_idx = min(start_idx + chunk_size_cv, n_voxels)

            if args.verb >= 1 and chunk_idx % max(1, n_chunks // 10) == 0:
                print(f"  Chunk {chunk_idx+1}/{n_chunks}: voxels {start_idx:,}-{end_idx:,}")

            # Optimize λ for each voxel in this chunk
            for voxel_idx in range(start_idx, end_idx):
                data_voxel = data_tensor[voxel_idx:voxel_idx+1, :]  # (1, n_timepoints)

                best_lambda_voxel, _ = fit_penalized_glm_cv(
                    data=data_voxel,
                    design=design_cv,
                    penalty_matrix=penalty_matrix,
                    lambda_values=lambda_values,
                    run_boundaries=run_boundaries,
                    device=device,
                    verbose=False,
                )

                lambda_map[voxel_idx] = best_lambda_voxel

        if args.verb >= 1:
            print("\n✓ Per-voxel λ optimization complete")
            print(f"  λ range: [{lambda_map.min():.3e}, {lambda_map.max():.3e}]")
            print(f"  λ median: {np.median(lambda_map):.3e}")

    # ========================================================================
    # Save lambda map (per_voxel only)
    # ========================================================================

    if args.save_lambda_map and args.optimize_level == 'per_voxel':
        if args.verb >= 1:
            print("\nSaving lambda map...")

        # Create 3D volume
        lambda_volume = np.zeros((nx, ny, nz))
        lambda_volume[mask] = lambda_map

        lambda_file = f"{args.output_prefix}_lambda{_nii_ext}"
        save_nifti(lambda_volume, lambda_file, affine=affine)

        if args.verb >= 1:
            print(f"  ✓ Saved: {lambda_file}")

    # ========================================================================
    # Fit final TPS model with optimal λ
    # ========================================================================

    if args.verb >= 1:
        print("\n" + "="*70)
        print("Fitting Final TPS Model")
        print("="*70)

    # Fit penalized GLM for stimulus regressors
    # Then fit polynomial regressors on residuals

    if args.verb >= 1:
        print("\n  Step 1: Fit penalized TPS model (stimulus regressors)")

    betas_stimulus_tensor = fit_penalized_glm(
        data=data_tensor,
        design=design_stimulus,
        penalty_matrix=penalty_matrix,
        lambda_values=lambda_map if args.optimize_level == 'per_voxel' else best_lambda,
        device=device,
        chunk_size=args.chunk_size,
        verbose=args.verb >= 1,
    )

    betas_stimulus = betas_stimulus_tensor.cpu().numpy()

    # Fit polynomial regressors on residuals (if present)
    if total_poly_cols > 0:
        if args.verb >= 1:
            print("\n  Step 2: Fit polynomial regressors (nuisance, no penalty)")

        # Compute residuals after removing stimulus effects
        pred_stimulus = (design_stimulus @ betas_stimulus_tensor.T.to(device)).T  # (n_voxels, n_timepoints)
        residuals = data_tensor.to(device) - pred_stimulus

        # Fit polynomials to residuals using standard least squares
        # β_poly = (X'X)^-1 X'y
        poly_XTX = poly_tensor.T @ poly_tensor  # (n_poly, n_poly)
        poly_XTy = poly_tensor.T @ residuals.T.to(device)  # (n_poly, n_voxels)

        try:
            betas_poly = torch.linalg.solve(poly_XTX, poly_XTy)  # (n_poly, n_voxels)
        except Exception:
            betas_poly = torch.linalg.lstsq(poly_XTX, poly_XTy).solution

        betas_poly = betas_poly.T.cpu().numpy()  # (n_voxels, n_poly)

        if args.verb >= 1:
            print("  ✓ Polynomial fit complete")
    else:
        betas_poly = np.zeros((n_voxels, 0))

    if args.verb >= 1:
        print("\n  ✓ TPS model fit complete")

    # ========================================================================
    # Extract and save HRF estimates
    # ========================================================================

    if args.verb >= 1:
        print("\nExtracting HRF estimates...")

    if args.verb >= 1:
        print("\nSaving HRF estimates (iresp files)...")

    output_files = []
    beta_col_idx = 0

    for cond_idx in range(n_conditions):
        label = args.stim_labels[cond_idx]
        n_basis = n_basis_per_condition[cond_idx]
        bot, top = windows[cond_idx]

        # Extract betas for this condition
        betas_cond = betas_stimulus[:, beta_col_idx:beta_col_idx+n_basis]
        beta_col_idx += n_basis

        # Create 4D volume: (nx, ny, nz, n_basis)
        hrf_volume = np.zeros((nx, ny, nz, n_basis))
        hrf_volume[mask, :] = betas_cond

        # Save as iresp file
        iresp_file = f"{args.output_prefix}_iresp_{label}{_nii_ext}"

        # Save with TR spacing in header
        # The "TR" here is the spacing between HRF samples (knot spacing)
        knot_spacing = (top - bot) / (n_basis - 1) if n_basis > 1 else tr

        save_nifti(hrf_volume, iresp_file, affine=affine)

        output_files.append(iresp_file)

        if args.verb >= 1:
            print(f"  {label}: {iresp_file}")
            print(f"    Shape: {hrf_volume.shape}")
            print(f"    Knot spacing: {knot_spacing:.3f}s")

    # ========================================================================
    # Summary
    # ========================================================================

    if args.verb >= 1:
        print("\n" + "="*70)
        print("TPS HRF Estimation Complete")
        print("="*70)
        print("\nOutput files:")
        for f in output_files:
            print(f"  {f}")
        if args.save_design:
            print(f"  {args.output_prefix}_design_TPS.1D")
        if args.save_lambda_map and args.optimize_level == 'per_voxel':
            print(f"  {args.output_prefix}_lambda{_nii_ext}")
        print()


if __name__ == '__main__':
    main()

