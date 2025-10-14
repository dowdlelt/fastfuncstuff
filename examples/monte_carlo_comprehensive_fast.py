#!/usr/bin/env python
"""
ULTRA-FAST Comprehensive Monte Carlo ISI Study

Key optimizations:
1. Test ALL 20 HRFs in parallel (not sequential)
2. Process multiple patterns simultaneously
3. Pre-generate all noise efficiently
4. Minimize CPU↔GPU transfers

Settings:
- ISI means: 2.0 to 5.0 in 0.25 steps (13 conditions)
- 1000 noise realizations per condition
- 20 activation patterns × 20 HRFs = 400 combinations per ISI
- Total: 5.2 million simulations

GPU-batched for MAXIMUM speed.
"""

import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "/Users/logan/local_bin/fastfuncsim")
from simulate_isi_sweep import (
    build_alternating_design,
    convolve_design_with_hrf,
    create_polynomial_regressors,
    generate_poisson_isis,
    load_cnvlab_hrf_library,
)

import fastfuncsim as ffs
from fastfuncsim.noise import generate_fmri_noise


def create_fir_design(design_unconvolved: np.ndarray, n_fir_bins: int) -> np.ndarray:
    """Create FIR design matrix."""
    n_timepoints, n_conds = design_unconvolved.shape
    X_fir = np.zeros((n_timepoints, n_conds * n_fir_bins), dtype=np.float32)

    for cond_idx in range(n_conds):
        onsets = np.where(design_unconvolved[:, cond_idx] > 0)[0]
        for bin_idx in range(n_fir_bins):
            col_idx = cond_idx * n_fir_bins + bin_idx
            for onset in onsets:
                if onset + bin_idx < n_timepoints:
                    X_fir[onset + bin_idx, col_idx] += 1.0

    return X_fir


def run_ultra_fast_monte_carlo(
    activation_patterns: np.ndarray,  # (n_patterns, n_conds)
    hrfs: np.ndarray,  # (n_hrfs, hrf_length)
    design_unconvolved: np.ndarray,
    X_fir_full_torch: torch.Tensor,
    tr: float,
    n_noise_realizations: int,
    n_fir_bins: int,
    noise_std: float,
    baseline: float,
    device: torch.device,
    verbose: bool = False,
    collect_noise_sample: bool = False,
) -> Dict:
    """
    Ultra-fast Monte Carlo testing ALL patterns × ALL HRFs in parallel.

    Returns: results[pattern_idx][true_hrf_idx] = {...metrics...}
    """
    n_timepoints = design_unconvolved.shape[0]
    n_conds = design_unconvolved.shape[1]
    n_patterns = activation_patterns.shape[0]
    n_hrfs = hrfs.shape[0]

    if verbose:
        print(
            f"  Ultra-fast batch: {n_patterns} patterns × {n_hrfs} HRFs × {n_noise_realizations} realizations"
        )
        print(
            f"                  = {n_patterns * n_hrfs * n_noise_realizations:,} simulations"
        )

    # =========================================================================
    # 1. Pre-generate ALL noise realizations efficiently
    # =========================================================================
    if verbose:
        print(f"  Generating {n_noise_realizations} noise realizations...")

    noise_realizations = []
    noise_pbar = tqdm(
        range(n_noise_realizations), desc="  Noise", leave=False, disable=not verbose
    )
    for i in noise_pbar:
        # Randomize physiological parameters
        resp_freq_i = np.random.uniform(0.25, 0.35)
        resp_width_i = np.random.uniform(0.08, 0.12)
        resp_strength_i = np.random.uniform(2.5, 3.5)
        cardiac_freq_i = np.random.uniform(0.9, 1.1)
        cardiac_width_i = np.random.uniform(0.04, 0.06)
        cardiac_strength_i = np.random.uniform(4.0, 6.0)
        pink_exp_i = np.random.uniform(0.9, 1.1)

        noise_i = generate_fmri_noise(
            tr=tr,
            duration_s=n_timepoints * tr,
            matrix_size=(1, 1),
            resp_freq=resp_freq_i,
            resp_width=resp_width_i,
            resp_strength=resp_strength_i,
            cardiac_freq=cardiac_freq_i,
            cardiac_width=cardiac_width_i,
            cardiac_strength=cardiac_strength_i,
            pink_exp=pink_exp_i,
            normalize=True,
            device=device,
        ).squeeze()
        noise_realizations.append(noise_i)

    noise_all = (
        torch.stack(noise_realizations, dim=1) * noise_std
    )  # (n_timepoints, n_realizations)

    # Save noise sample if requested
    noise_sample = noise_all.cpu().numpy() if collect_noise_sample else None

    # =========================================================================
    # 2. Pre-build ALL design matrices for all HRFs (on GPU)
    # =========================================================================
    if verbose:
        print(f"  Building {n_hrfs} design matrices...")

    X_all_hrfs = []
    design_pbar = tqdm(hrfs, desc="  Design matrices", leave=False, disable=not verbose)
    for hrf in design_pbar:
        X = convolve_design_with_hrf(design_unconvolved, hrf)
        poly = create_polynomial_regressors(X.shape[0], max_order=3)
        X_full = np.hstack([X, poly]).astype(np.float32)
        X_all_hrfs.append(X_full)

    # Stack: (n_hrfs, n_timepoints, n_params)
    X_all_hrfs = np.array(X_all_hrfs)
    X_all_hrfs_torch = torch.from_numpy(X_all_hrfs).float().to(device)

    n_params = X_all_hrfs.shape[2]

    # =========================================================================
    # 3. Process ALL patterns × ALL HRFs in mega-batches
    # =========================================================================
    if verbose:
        print(f"  Running {n_patterns} patterns × {n_hrfs} HRFs...")

    results = {}

    pattern_pbar = tqdm(
        enumerate(activation_patterns),
        total=n_patterns,
        desc="  Patterns",
        leave=False,
        disable=not verbose,
    )
    for pattern_idx, pattern in pattern_pbar:
        results[pattern_idx] = {}

        # Generate true signals for ALL HRFs at once
        # true_signals: (n_hrfs, n_timepoints)
        onset_vector = np.zeros(n_timepoints)
        for cond_idx in range(n_conds):
            onset_times = np.where(design_unconvolved[:, cond_idx] > 0)[0]
            onset_vector[onset_times] = pattern[cond_idx]

        true_signals = []
        for hrf in hrfs:
            true_signal = np.convolve(onset_vector, hrf, mode="full")[:n_timepoints]
            true_signals.append(true_signal + baseline)

        true_signals_torch = (
            torch.from_numpy(np.array(true_signals)).float().to(device)
        )  # (n_hrfs, n_timepoints)

        # Add noise: (n_hrfs, n_timepoints, n_realizations)
        data_all_hrfs = true_signals_torch[:, :, None] + noise_all[None, :, :]

        # =====================================================================
        # Parametric HRF fitting: Test ALL HRFs against ALL true HRF scenarios
        # =====================================================================
        # For each true_hrf_idx, test all candidate HRFs
        # Shape: (n_candidate_hrfs, n_true_hrfs, n_realizations)

        r2_matrix = torch.zeros((n_hrfs, n_hrfs, n_noise_realizations), device=device)
        betas_matrix = torch.zeros(
            (n_hrfs, n_hrfs, n_conds, n_noise_realizations), device=device
        )

        hrf_pbar = tqdm(
            range(n_hrfs),
            desc=f"    HRFs (Pattern {pattern_idx})",
            leave=False,
            disable=not verbose,
        )
        for candidate_hrf_idx in hrf_pbar:
            X_torch = X_all_hrfs_torch[candidate_hrf_idx]  # (n_timepoints, n_params)

            # Fit to all true HRF scenarios at once
            # data_all_hrfs: (n_hrfs, n_timepoints, n_realizations)
            # Reshape for batched solve: (n_true_hrfs * n_realizations, n_timepoints)
            data_reshaped = data_all_hrfs.reshape(
                -1, n_timepoints
            ).T  # (n_timepoints, n_true_hrfs * n_realizations)

            # Batched OLS
            XTX = X_torch.T @ X_torch
            XTY = X_torch.T @ data_reshaped
            ridge = 1e-6
            XTX_reg = XTX + ridge * torch.eye(XTX.shape[0], device=device)
            beta = torch.linalg.solve(
                XTX_reg, XTY
            )  # (n_params, n_true_hrfs * n_realizations)

            # R²
            predicted = X_torch @ beta
            residuals = data_reshaped - predicted
            ss_res = torch.sum(residuals**2, dim=0)
            ss_tot = torch.sum(
                (data_reshaped - data_reshaped.mean(dim=0, keepdim=True)) ** 2, dim=0
            )
            r2 = 1.0 - ss_res / (ss_tot + 1e-10)

            # Reshape back: (n_true_hrfs, n_realizations)
            r2_reshaped = r2.reshape(n_hrfs, n_noise_realizations)
            beta_reshaped = beta.reshape(n_params, n_hrfs, n_noise_realizations)

            r2_matrix[candidate_hrf_idx] = r2_reshaped
            betas_matrix[candidate_hrf_idx] = beta_reshaped[:n_conds].permute(
                1, 0, 2
            )  # (n_true_hrfs, n_conds, n_realizations)

        # =====================================================================
        # Extract results for each true HRF
        # =====================================================================
        true_hrf_pbar = tqdm(
            range(n_hrfs),
            desc=f"    Extracting (Pattern {pattern_idx})",
            leave=False,
            disable=not verbose,
        )
        for true_hrf_idx in true_hrf_pbar:
            # R² for all candidate HRFs: (n_candidate_hrfs, n_realizations)
            r2_per_candidate = r2_matrix[:, true_hrf_idx, :]  # (n_hrfs, n_realizations)
            betas_per_candidate = betas_matrix[
                :, true_hrf_idx, :, :
            ]  # (n_hrfs, n_conds, n_realizations)

            # Find best HRF per realization
            best_hrf_idx_per_realization = torch.argmax(
                r2_per_candidate, dim=0
            )  # (n_realizations,)
            hrf_correct = (best_hrf_idx_per_realization == true_hrf_idx).float()

            # Get R² from best HRF
            r2_parametric = r2_per_candidate[
                best_hrf_idx_per_realization, torch.arange(n_noise_realizations)
            ]

            # Beta estimates from best HRF
            beta_estimates_A = torch.stack(
                [
                    betas_per_candidate[best_hrf_idx_per_realization[i], 0, i]
                    for i in range(n_noise_realizations)
                ]
            )
            beta_estimates_B = torch.stack(
                [
                    betas_per_candidate[best_hrf_idx_per_realization[i], 1, i]
                    for i in range(n_noise_realizations)
                ]
            )

            beta_error_A = torch.abs(beta_estimates_A - pattern[0])
            beta_error_B = torch.abs(beta_estimates_B - pattern[1])

            # Move to CPU for storage
            r2_parametric_cpu = r2_parametric.cpu().numpy()
            beta_error_A_cpu = beta_error_A.cpu().numpy()
            beta_error_B_cpu = beta_error_B.cpu().numpy()
            hrf_correct_cpu = hrf_correct.cpu().numpy()
            best_hrf_idx_cpu = best_hrf_idx_per_realization.cpu().numpy()

            # =================================================================
            # FIR model fitting (same for all true HRFs - fits the data)
            # =================================================================
            data_true = data_all_hrfs[true_hrf_idx]  # (n_timepoints, n_realizations)

            XTX = X_fir_full_torch.T @ X_fir_full_torch
            XTY = X_fir_full_torch.T @ data_true
            ridge = 1e-6
            XTX_reg = XTX + ridge * torch.eye(XTX.shape[0], device=device)
            beta_fir_full = torch.linalg.solve(XTX_reg, XTY)

            # R² for FIR
            predicted_fir = X_fir_full_torch @ beta_fir_full
            residuals_fir = data_true - predicted_fir
            ss_res_fir = torch.sum(residuals_fir**2, dim=0)
            ss_tot_fir = torch.sum(
                (data_true - data_true.mean(dim=0, keepdim=True)) ** 2, dim=0
            )
            r2_fir = 1.0 - ss_res_fir / (ss_tot_fir + 1e-10)

            # Extract FIR betas
            n_fir_params = n_conds * n_fir_bins
            beta_fir = beta_fir_full[:n_fir_params]

            # FIR HRF shape recovery
            true_hrf = hrfs[true_hrf_idx]
            if len(true_hrf) > n_fir_bins:
                true_hrf_matched = true_hrf[:n_fir_bins]
            else:
                true_hrf_matched = np.pad(
                    true_hrf, (0, n_fir_bins - len(true_hrf)), mode="constant"
                )

            true_hrf_torch = torch.from_numpy(true_hrf_matched).float().to(device)

            # Condition A
            beta_A_fir = beta_fir[:n_fir_bins]
            if np.abs(pattern[0]) > 1e-10:
                beta_A_norm = beta_A_fir / pattern[0]
                corr_A = []
                for real_idx in range(n_noise_realizations):
                    hrf_est = beta_A_norm[:, real_idx]
                    corr = torch.corrcoef(torch.stack([hrf_est, true_hrf_torch]))[0, 1]
                    corr_A.append(corr.item() if not torch.isnan(corr) else 0.0)
                corr_A = np.array(corr_A)
            else:
                corr_A = np.full(n_noise_realizations, np.nan)

            # Condition B
            beta_B_fir = beta_fir[n_fir_bins : 2 * n_fir_bins]
            if np.abs(pattern[1]) > 1e-10:
                beta_B_norm = beta_B_fir / pattern[1]
                corr_B = []
                for real_idx in range(n_noise_realizations):
                    hrf_est = beta_B_norm[:, real_idx]
                    corr = torch.corrcoef(torch.stack([hrf_est, true_hrf_torch]))[0, 1]
                    corr_B.append(corr.item() if not torch.isnan(corr) else 0.0)
                corr_B = np.array(corr_B)
            else:
                corr_B = np.full(n_noise_realizations, np.nan)

            r2_fir_cpu = r2_fir.cpu().numpy()

            # Store results
            results[pattern_idx][true_hrf_idx] = {
                "hrf_recovery_rate": hrf_correct_cpu.mean(),
                "true_hrf_idx": true_hrf_idx,
                "best_hrf_idx_all": best_hrf_idx_cpu,
                "r2_parametric_median": np.median(r2_parametric_cpu),
                "r2_parametric_iqr": [
                    np.percentile(r2_parametric_cpu, 25),
                    np.percentile(r2_parametric_cpu, 75),
                ],
                "r2_parametric_all": r2_parametric_cpu,
                "r2_fir_median": np.median(r2_fir_cpu),
                "r2_fir_iqr": [
                    np.percentile(r2_fir_cpu, 25),
                    np.percentile(r2_fir_cpu, 75),
                ],
                "r2_fir_all": r2_fir_cpu,
                "beta_error_A_median": np.median(beta_error_A_cpu),
                "beta_error_A_iqr": [
                    np.percentile(beta_error_A_cpu, 25),
                    np.percentile(beta_error_A_cpu, 75),
                ],
                "beta_error_A_all": beta_error_A_cpu,
                "beta_error_B_median": np.median(beta_error_B_cpu),
                "beta_error_B_iqr": [
                    np.percentile(beta_error_B_cpu, 25),
                    np.percentile(beta_error_B_cpu, 75),
                ],
                "beta_error_B_all": beta_error_B_cpu,
                "hrf_corr_A_median": np.nanmedian(corr_A)
                if not np.all(np.isnan(corr_A))
                else 0.0,
                "hrf_corr_A_iqr": [
                    np.nanpercentile(corr_A, 25),
                    np.nanpercentile(corr_A, 75),
                ]
                if not np.all(np.isnan(corr_A))
                else [0.0, 0.0],
                "hrf_corr_A_all": corr_A,
                "hrf_corr_B_median": np.nanmedian(corr_B)
                if not np.all(np.isnan(corr_B))
                else 0.0,
                "hrf_corr_B_iqr": [
                    np.nanpercentile(corr_B, 25),
                    np.nanpercentile(corr_B, 75),
                ]
                if not np.all(np.isnan(corr_B))
                else [0.0, 0.0],
                "hrf_corr_B_all": corr_B,
            }

            # Add noise sample to first result only
            if collect_noise_sample and pattern_idx == 0 and true_hrf_idx == 0:
                results[pattern_idx][true_hrf_idx]["noise_samples"] = noise_sample

    return results


def run_full_study_fast(
    isi_means: List[float],
    activation_patterns: np.ndarray,
    hrf_library_name: str = "cnvlab",
    n_noise_realizations: int = 100,
    n_fir_bins: int = 30,
    noise_std: float = 2.0,
    tr: float = 1.0,
    total_duration: float = 290.0,
    stim_duration: float = 5.0,
    baseline: float = 100.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict:
    """Run ultra-fast comprehensive study."""
    if device is None:
        device = ffs.get_device()

    if verbose:
        print(f"{'=' * 70}")
        print("ULTRA-FAST COMPREHENSIVE MONTE CARLO STUDY")
        print(f"{'=' * 70}\n")
        print(f"Device: {device}")
        print(
            f"ISI means: {len(isi_means)} conditions ({min(isi_means)}-{max(isi_means)}s)"
        )
        print(f"Patterns: {len(activation_patterns)}")
        print(f"Realizations: {n_noise_realizations}")
        print(f"FIR bins: {n_fir_bins}")
        print(
            f"\nTotal simulations: {len(isi_means)} × {len(activation_patterns)} × 20 HRFs × {n_noise_realizations}"
        )
        print(
            f"                 = {len(isi_means) * len(activation_patterns) * 20 * n_noise_realizations:,}\n"
        )

    # Load HRF library
    if hrf_library_name == "cnvlab":
        hrfs = load_cnvlab_hrf_library(duration=stim_duration, tr=tr)
    else:
        raise ValueError(f"Unknown library: {hrf_library_name}")

    n_hrfs = hrfs.shape[0]
    if verbose:
        print(f"Loaded {n_hrfs} HRFs\n")

    total_tr = int(total_duration / tr)
    stim_tr = int(stim_duration / tr)
    padding_tr = int(10 / tr)

    results = {}
    start_time = time.time()

    isi_pbar = tqdm(
        enumerate(isi_means),
        total=len(isi_means),
        desc="ISI Conditions",
        disable=not verbose,
    )
    for isi_idx, isi_mean in isi_pbar:
        if verbose:
            isi_pbar.set_description(f"ISI {isi_mean:.2f}s")
            print(f"\n{'=' * 70}")
            print(f"ISI: {isi_mean}s ({isi_idx + 1}/{len(isi_means)})")
            print(f"{'=' * 70}\n")

        # Generate design
        block_average = isi_mean + stim_duration
        n_trials = int((total_duration - 2 * padding_tr * tr) / block_average)
        n_trials = n_trials - (n_trials % 2)

        isis_sec = generate_poisson_isis(
            target_mean=isi_mean,
            n_isis=n_trials,
            lower_limit=2.0,
            upper_limit=8.0,
            seed=42 + isi_idx,
        )
        isis_tr = np.round(isis_sec / tr).astype(int)

        design_unconvolved = build_alternating_design(
            isis_tr=isis_tr,
            stim_dur_tr=stim_tr,
            n_conds=2,
            total_tr=total_tr,
            padding_tr=padding_tr,
        )

        # Pre-compute FIR design
        X_fir = create_fir_design(design_unconvolved, n_fir_bins)
        poly = create_polynomial_regressors(X_fir.shape[0], max_order=3)
        X_fir_full = np.hstack([X_fir, poly]).astype(np.float32)
        X_fir_full_torch = torch.from_numpy(X_fir_full).float().to(device)

        if verbose:
            print(
                f"  Design: {np.sum(design_unconvolved[:, 0]):.0f} + {np.sum(design_unconvolved[:, 1]):.0f} trials\n"
            )

        # Run ultra-fast mega-batch for this ISI
        collect_noise = isi_idx == 0
        isi_start = time.time()

        results[isi_mean] = run_ultra_fast_monte_carlo(
            activation_patterns=activation_patterns,
            hrfs=hrfs,
            design_unconvolved=design_unconvolved,
            X_fir_full_torch=X_fir_full_torch,
            tr=tr,
            n_noise_realizations=n_noise_realizations,
            n_fir_bins=n_fir_bins,
            noise_std=noise_std,
            baseline=baseline,
            device=device,
            verbose=verbose,
            collect_noise_sample=collect_noise,
        )

        isi_elapsed = time.time() - isi_start
        sims_per_sec = (
            len(activation_patterns) * n_hrfs * n_noise_realizations
        ) / isi_elapsed
        if verbose:
            print(
                f"  ISI completed in {isi_elapsed:.1f}s ({sims_per_sec:,.0f} sims/sec)"
            )

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n{'=' * 70}")
        print("COMPLETE!")
        print(f"{'=' * 70}\n")
        print(f"Total time: {elapsed / 60:.1f} minutes")
        print(f"Time per ISI: {elapsed / len(isi_means):.1f}s")
        print(
            f"Simulations per second: {len(isi_means) * len(activation_patterns) * n_hrfs * n_noise_realizations / elapsed:,.0f}"
        )

    return results


if __name__ == "__main__":
    device = ffs.get_device()
    output_dir = Path("monte_carlo_comprehensive_results")
    output_dir.mkdir(exist_ok=True)

    # Import plotting functions from original script
    from monte_carlo_comprehensive import (
        plot_comprehensive_results,
        plot_noise_spectrum,
    )

    # Finer ISI resolution: 2.0 to 5.0 in 0.25 steps
    isi_means = np.arange(2.0, 5.25, 0.25).tolist()

    activation_patterns = np.array(
        [
            [5, 1],
            [5, 2],
            [5, 3],
            [5, 4],
            [5, 5],
            [4, 5],
            [3, 5],
            [2, 5],
            [1, 5],
            [1, 3],
            [3, 3],
            [3, 1],
            [1, 1],
            [0, 2],
            [2, 0],
            [-1, -1],
            [-3, -3],
            [-3, 4],
            [4, -3],
            [0, 0],
        ],
        dtype=np.float32,
    )

    # Run with 1000 realizations - ULTRA FAST!
    results = run_full_study_fast(
        isi_means=isi_means,
        activation_patterns=activation_patterns,
        hrf_library_name="cnvlab",
        n_noise_realizations=100,
        n_fir_bins=30,
        noise_std=2.0,
        tr=1.0,
        total_duration=290.0,
        stim_duration=5.0,
        baseline=100.0,
        device=device,
        verbose=True,
    )

    # Save
    with open(output_dir / "comprehensive_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved: {output_dir / 'comprehensive_results.pkl'}")

    # Plot noise spectrum (if we collected noise samples)
    first_isi = isi_means[0]
    if "noise_samples" in results[first_isi][0][0]:
        print("\nCreating noise spectrum plot...")
        noise_samples = results[first_isi][0][0]["noise_samples"]
        plot_noise_spectrum(noise_samples, 1.0, 2.0, 100.0, output_dir)

    # Plot
    print("\nCreating performance plots...")
    plot_comprehensive_results(results, activation_patterns, isi_means, output_dir)

    print(f"\n{'=' * 70}")
    print("ALL DONE!")
    print(f"{'=' * 70}")
    print(f"\nResults: {output_dir}/")
