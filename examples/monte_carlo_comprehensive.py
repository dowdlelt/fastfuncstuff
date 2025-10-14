#!/usr/bin/env python
"""
Comprehensive Monte Carlo ISI Study

Combines:
1. Parametric HRF library recovery
2. Beta estimation accuracy
3. FIR-based HRF shape recovery
4. All metrics with error bars (IQR shading)

Settings:
- ISI means: 2.0 to 5.0 in 0.25 steps (13 conditions)
- 500 noise realizations per condition
- 20 activation patterns
- Total: 130,000 simulations

GPU-batched for speed.
"""

import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

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


def run_comprehensive_monte_carlo(
    activation_pattern: np.ndarray,
    true_hrf_idx: int,
    hrfs: np.ndarray,
    design_unconvolved: np.ndarray,
    X_fir_full_torch: torch.Tensor,
    tr: float,
    n_noise_realizations: int,
    n_fir_bins: int,
    noise_std: float,
    baseline: float,
    device: torch.device,
    verbose: bool = False,
    collect_noise: bool = False,
) -> Dict:
    """
    Run comprehensive Monte Carlo with all metrics (batched).

    Returns metrics:
    - Parametric HRF recovery rate
    - R² (parametric and FIR) with IQR
    - Beta estimation error with IQR
    - FIR HRF shape correlation with IQR
    """
    n_timepoints = design_unconvolved.shape[0]
    n_conds = design_unconvolved.shape[1]

    if verbose:
        print(
            f"    Pattern {activation_pattern}: {n_noise_realizations} realizations (varying physiological params)..."
        )

    # Generate true signal
    true_hrf = hrfs[true_hrf_idx]
    onset_vector = np.zeros(n_timepoints)
    for cond_idx in range(n_conds):
        onset_times = np.where(design_unconvolved[:, cond_idx] > 0)[0]
        onset_vector[onset_times] = activation_pattern[cond_idx]

    true_signal = np.convolve(onset_vector, true_hrf, mode="full")[:n_timepoints]
    true_signal += baseline

    # Generate all noise realizations with varying physiological parameters
    # Each realization gets slightly different respiratory/cardiac characteristics
    # to better capture real-world variability
    noise_realizations = []
    for i in range(n_noise_realizations):
        # Randomize physiological parameters within realistic bounds
        # Respiratory: ~0.2-0.4 Hz (12-24 breaths/min)
        resp_freq_i = np.random.uniform(0.25, 0.35)
        resp_width_i = np.random.uniform(0.08, 0.12)  # Width of respiratory peak
        resp_strength_i = np.random.uniform(2.5, 3.5)

        # Cardiac: ~0.8-1.2 Hz (48-72 bpm)
        cardiac_freq_i = np.random.uniform(0.9, 1.1)
        cardiac_width_i = np.random.uniform(0.04, 0.06)  # Width of cardiac peak
        cardiac_strength_i = np.random.uniform(4.0, 6.0)

        # Pink noise exponent: slight variation
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

    true_signal_torch = torch.from_numpy(true_signal).float().to(device)[:, None]
    data_all = true_signal_torch + noise_all  # (n_timepoints, n_realizations)

    # =========================================================================
    # 1. Parametric HRF fitting
    # =========================================================================
    n_hrfs = hrfs.shape[0]
    r2_per_hrf = []
    betas_per_hrf = []  # Store betas for best HRF

    for hrf_idx, hrf in enumerate(hrfs):
        X = convolve_design_with_hrf(design_unconvolved, hrf)
        poly = create_polynomial_regressors(X.shape[0], max_order=3)
        X_full = np.hstack([X, poly]).astype(np.float32)
        X_torch = torch.from_numpy(X_full).float().to(device)

        # Batched OLS
        XTX = X_torch.T @ X_torch
        XTY = X_torch.T @ data_all
        ridge = 1e-6
        XTX_reg = XTX + ridge * torch.eye(XTX.shape[0], device=device)
        beta = torch.linalg.solve(XTX_reg, XTY)  # (n_params, n_realizations)

        # R²
        predicted = X_torch @ beta
        residuals = data_all - predicted
        ss_res = torch.sum(residuals**2, dim=0)
        ss_tot = torch.sum((data_all - data_all.mean(dim=0, keepdim=True)) ** 2, dim=0)
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)

        r2_per_hrf.append(r2.cpu().numpy())

        # Store betas (task regressors only)
        betas_per_hrf.append(beta[:n_conds].cpu().numpy())  # (n_conds, n_realizations)

    r2_per_hrf = np.array(r2_per_hrf)  # (n_hrfs, n_realizations)
    betas_per_hrf = np.array(betas_per_hrf)  # (n_hrfs, n_conds, n_realizations)

    # Find best HRF per realization
    best_hrf_idx_per_realization = np.argmax(r2_per_hrf, axis=0)  # (n_realizations,)
    hrf_correct = (best_hrf_idx_per_realization == true_hrf_idx).astype(float)

    # Get R² and betas from best HRF
    r2_parametric = r2_per_hrf[
        best_hrf_idx_per_realization, np.arange(n_noise_realizations)
    ]

    # Beta estimates from best HRF per realization
    beta_estimates_A = np.array(
        [
            betas_per_hrf[best_hrf_idx_per_realization[i], 0, i]
            for i in range(n_noise_realizations)
        ]
    )
    beta_estimates_B = np.array(
        [
            betas_per_hrf[best_hrf_idx_per_realization[i], 1, i]
            for i in range(n_noise_realizations)
        ]
    )

    # Beta estimation errors
    beta_error_A = np.abs(beta_estimates_A - activation_pattern[0])
    beta_error_B = np.abs(beta_estimates_B - activation_pattern[1])

    # =========================================================================
    # 2. FIR model fitting (batched)
    # =========================================================================
    XTX = X_fir_full_torch.T @ X_fir_full_torch
    XTY = X_fir_full_torch.T @ data_all
    ridge = 1e-6
    XTX_reg = XTX + ridge * torch.eye(XTX.shape[0], device=device)
    beta_fir_full = torch.linalg.solve(XTX_reg, XTY)

    # R² for FIR
    predicted_fir = X_fir_full_torch @ beta_fir_full
    residuals_fir = data_all - predicted_fir
    ss_res_fir = torch.sum(residuals_fir**2, dim=0)
    ss_tot_fir = torch.sum((data_all - data_all.mean(dim=0, keepdim=True)) ** 2, dim=0)
    r2_fir = 1.0 - ss_res_fir / (ss_tot_fir + 1e-10)
    r2_fir = r2_fir.cpu().numpy()

    # Extract FIR betas
    n_fir_params = n_conds * n_fir_bins
    beta_fir = beta_fir_full[:n_fir_params]  # (n_fir_params, n_realizations)

    # =========================================================================
    # 3. FIR HRF shape recovery
    # =========================================================================
    # Truncate true HRF
    if len(true_hrf) > n_fir_bins:
        true_hrf_matched = true_hrf[:n_fir_bins]
    else:
        true_hrf_matched = np.pad(
            true_hrf, (0, n_fir_bins - len(true_hrf)), mode="constant"
        )

    true_hrf_torch = torch.from_numpy(true_hrf_matched).float().to(device)

    # Condition A
    beta_A_fir = beta_fir[:n_fir_bins]  # (n_fir_bins, n_realizations)
    if np.abs(activation_pattern[0]) > 1e-10:
        beta_A_norm = beta_A_fir / activation_pattern[0]
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
    if np.abs(activation_pattern[1]) > 1e-10:
        beta_B_norm = beta_B_fir / activation_pattern[1]
        corr_B = []
        for real_idx in range(n_noise_realizations):
            hrf_est = beta_B_norm[:, real_idx]
            corr = torch.corrcoef(torch.stack([hrf_est, true_hrf_torch]))[0, 1]
            corr_B.append(corr.item() if not torch.isnan(corr) else 0.0)
        corr_B = np.array(corr_B)
    else:
        corr_B = np.full(n_noise_realizations, np.nan)

    # =========================================================================
    # Compute statistics
    # =========================================================================
    results = {
        # Parametric HRF recovery
        "hrf_recovery_rate": hrf_correct.mean(),
        "true_hrf_idx": true_hrf_idx,  # The HRF that was actually used
        "best_hrf_idx_all": best_hrf_idx_per_realization,  # Selected HRF for each realization
        # R² (parametric)
        "r2_parametric_median": np.median(r2_parametric),
        "r2_parametric_iqr": [
            np.percentile(r2_parametric, 25),
            np.percentile(r2_parametric, 75),
        ],
        "r2_parametric_all": r2_parametric,
        # R² (FIR)
        "r2_fir_median": np.median(r2_fir),
        "r2_fir_iqr": [np.percentile(r2_fir, 25), np.percentile(r2_fir, 75)],
        "r2_fir_all": r2_fir,
        # Beta estimation error
        "beta_error_A_median": np.median(beta_error_A),
        "beta_error_A_iqr": [
            np.percentile(beta_error_A, 25),
            np.percentile(beta_error_A, 75),
        ],
        "beta_error_A_all": beta_error_A,
        "beta_error_B_median": np.median(beta_error_B),
        "beta_error_B_iqr": [
            np.percentile(beta_error_B, 25),
            np.percentile(beta_error_B, 75),
        ],
        "beta_error_B_all": beta_error_B,
        # FIR HRF shape recovery (handle all-NaN case for zero activation)
        "hrf_corr_A_median": np.nanmedian(corr_A)
        if not np.all(np.isnan(corr_A))
        else 0.0,
        "hrf_corr_A_iqr": [np.nanpercentile(corr_A, 25), np.nanpercentile(corr_A, 75)]
        if not np.all(np.isnan(corr_A))
        else [0.0, 0.0],
        "hrf_corr_A_all": corr_A,
        "hrf_corr_B_median": np.nanmedian(corr_B)
        if not np.all(np.isnan(corr_B))
        else 0.0,
        "hrf_corr_B_iqr": [np.nanpercentile(corr_B, 25), np.nanpercentile(corr_B, 75)]
        if not np.all(np.isnan(corr_B))
        else [0.0, 0.0],
        "hrf_corr_B_all": corr_B,
    }

    # Save noise samples if requested
    if collect_noise:
        results["noise_samples"] = noise_all.cpu().numpy()

    return results


def run_full_study(
    isi_means: List[float],
    activation_patterns: np.ndarray,
    hrf_library_name: str = "cnvlab",
    n_noise_realizations: int = 500,
    n_fir_bins: int = 30,
    noise_std: float = 2.0,
    tr: float = 1.0,
    total_duration: float = 290.0,
    stim_duration: float = 5.0,
    baseline: float = 100.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict:
    """Run comprehensive study."""
    if device is None:
        device = ffs.get_device()

    if verbose:
        print(f"{'=' * 70}")
        print("COMPREHENSIVE MONTE CARLO STUDY")
        print(f"{'=' * 70}\n")
        print(f"Device: {device}")
        print(
            f"ISI means: {len(isi_means)} conditions ({min(isi_means)}-{max(isi_means)}s)"
        )
        print(f"Patterns: {len(activation_patterns)}")
        print(f"Realizations: {n_noise_realizations}")
        print(f"FIR bins: {n_fir_bins}")
        print(f"\nNoise characteristics (varying per realization):")
        print(
            f"  Noise SD: {noise_std:.1f} ({100 * noise_std / baseline:.1f}% of baseline)"
        )
        print(f"  Respiratory freq: 0.25-0.35 Hz")
        print(f"  Cardiac freq: 0.9-1.1 Hz")
        print(f"  Pink noise exponent: 0.9-1.1")
        print(
            f"\nTotal simulations: {len(isi_means)} × {len(activation_patterns)} × {n_noise_realizations}"
        )
        print(
            f"                 = {len(isi_means) * len(activation_patterns) * n_noise_realizations:,}\n"
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

    for isi_idx, isi_mean in enumerate(isi_means):
        if verbose:
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

        results[isi_mean] = {}

        for pattern_idx, pattern in enumerate(activation_patterns):
            results[isi_mean][pattern_idx] = {}

            # Test this activation pattern with ALL HRFs
            for true_hrf_idx in range(n_hrfs):
                # Collect noise samples from first ISI, first pattern, first HRF only
                collect_noise = isi_idx == 0 and pattern_idx == 0 and true_hrf_idx == 0

                if verbose and true_hrf_idx == 0:  # Only print once per pattern
                    print(f"    Pattern {pattern}: testing all {n_hrfs} HRFs...")

                pattern_results = run_comprehensive_monte_carlo(
                    activation_pattern=pattern,
                    true_hrf_idx=true_hrf_idx,
                    hrfs=hrfs,
                    design_unconvolved=design_unconvolved,
                    X_fir_full_torch=X_fir_full_torch,
                    tr=tr,
                    n_noise_realizations=n_noise_realizations,
                    n_fir_bins=n_fir_bins,
                    noise_std=noise_std,
                    baseline=baseline,
                    device=device,
                    verbose=False,  # Don't print for each HRF
                    collect_noise=collect_noise,
                )

                results[isi_mean][pattern_idx][true_hrf_idx] = pattern_results

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n{'=' * 70}")
        print("COMPLETE!")
        print(f"{'=' * 70}\n")
        print(f"Total time: {elapsed / 60:.1f} minutes")
        print(f"Time per ISI: {elapsed / len(isi_means):.1f}s")

    return results


def plot_comprehensive_results(results, activation_patterns, isi_means, output_dir):
    """
    Create all plots with error shading.

    Metrics are averaged across all HRFs to show overall pattern performance.
    HRF recovery is shown as heatmaps (pattern × true HRF) for each ISI.
    """
    n_patterns = len(activation_patterns)
    n_rows, n_cols = 4, 5

    # Get number of HRFs from first result
    first_isi = isi_means[0]
    n_hrfs = len(results[first_isi][0])  # Number of HRFs tested

    # Helper function: Average metric across all HRFs
    def plot_metric_with_error(
        ax,
        isi_means,
        results,
        pattern_idx,
        metric_key,
        color="blue",
        ylabel="",
        ylim=None,
    ):
        medians, lower, upper = [], [], []
        for isi in isi_means:
            # Collect metric from all HRFs for this pattern/ISI
            values_across_hrfs = []
            for hrf_idx in range(n_hrfs):
                values_across_hrfs.append(
                    results[isi][pattern_idx][hrf_idx][f"{metric_key}_median"]
                )

            # Average across HRFs
            medians.append(np.mean(values_across_hrfs))

            # For IQR, average the lower and upper bounds across HRFs
            lower_vals = [
                results[isi][pattern_idx][hrf_idx][f"{metric_key}_iqr"][0]
                for hrf_idx in range(n_hrfs)
            ]
            upper_vals = [
                results[isi][pattern_idx][hrf_idx][f"{metric_key}_iqr"][1]
                for hrf_idx in range(n_hrfs)
            ]
            lower.append(np.mean(lower_vals))
            upper.append(np.mean(upper_vals))

        ax.plot(isi_means, medians, "o-", linewidth=2, markersize=6, color=color)
        ax.fill_between(isi_means, lower, upper, alpha=0.25, color=color)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlabel("ISI (s)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.3)

    # Figure 1: Parametric R² (averaged across HRFs)
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Parametric Model R² (Averaged across all HRFs)", fontsize=18, fontweight="bold"
    )

    for idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        pattern = activation_patterns[idx]
        plot_metric_with_error(
            ax,
            isi_means,
            results,
            idx,
            "r2_parametric",
            color="blue",
            ylabel="R²",
            ylim=[0, 1.05],
        )
        ax.set_title(
            f"P{idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "r2_parametric.png", dpi=150)
    print(f"Saved: {output_dir / 'r2_parametric.png'}")
    plt.close()

    # Figure 2: Beta Error Condition A (averaged across HRFs)
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Beta Estimation Error: Condition A (Averaged across all HRFs)",
        fontsize=18,
        fontweight="bold",
    )

    for idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        pattern = activation_patterns[idx]
        plot_metric_with_error(
            ax,
            isi_means,
            results,
            idx,
            "beta_error_A",
            color="green",
            ylabel="|β̂ - β_true|",
        )
        ax.set_title(
            f"P{idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "beta_error_A.png", dpi=150)
    print(f"Saved: {output_dir / 'beta_error_A.png'}")
    plt.close()

    # Figure 3: Beta Error Condition B (averaged across HRFs)
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Beta Estimation Error: Condition B (Averaged across all HRFs)",
        fontsize=18,
        fontweight="bold",
    )

    for idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        pattern = activation_patterns[idx]
        plot_metric_with_error(
            ax,
            isi_means,
            results,
            idx,
            "beta_error_B",
            color="orange",
            ylabel="|β̂ - β_true|",
        )
        ax.set_title(
            f"P{idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "beta_error_B.png", dpi=150)
    print(f"Saved: {output_dir / 'beta_error_B.png'}")
    plt.close()

    # Figure 4: FIR HRF Recovery Condition A (averaged across HRFs)
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "FIR HRF Shape Recovery: Condition A (Averaged across all HRFs)",
        fontsize=18,
        fontweight="bold",
    )

    for idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        pattern = activation_patterns[idx]
        plot_metric_with_error(
            ax,
            isi_means,
            results,
            idx,
            "hrf_corr_A",
            color="blue",
            ylabel="Correlation",
            ylim=[-0.1, 1.05],
        )
        ax.axhline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(
            f"P{idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "fir_hrf_recovery_A.png", dpi=150)
    print(f"Saved: {output_dir / 'fir_hrf_recovery_A.png'}")
    plt.close()

    # Figure 5: FIR HRF Recovery Condition B (averaged across HRFs)
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "FIR HRF Shape Recovery: Condition B (Averaged across all HRFs)",
        fontsize=18,
        fontweight="bold",
    )

    for idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        pattern = activation_patterns[idx]
        plot_metric_with_error(
            ax,
            isi_means,
            results,
            idx,
            "hrf_corr_B",
            color="orange",
            ylabel="Correlation",
            ylim=[-0.1, 1.05],
        )
        ax.axhline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_title(
            f"P{idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "fir_hrf_recovery_B.png", dpi=150)
    print(f"Saved: {output_dir / 'fir_hrf_recovery_B.png'}")
    plt.close()

    # =========================================================================
    # HRF RECOVERY HEATMAPS: One per ISI showing (pattern × true HRF)
    # =========================================================================
    print(f"\nCreating HRF recovery heatmaps (one per ISI)...")

    for isi_idx, isi in enumerate(isi_means):
        # Build recovery rate matrix: (n_patterns × n_hrfs)
        recovery_matrix = np.zeros((n_patterns, n_hrfs))

        for pattern_idx in range(n_patterns):
            for true_hrf_idx in range(n_hrfs):
                recovery_rate = results[isi][pattern_idx][true_hrf_idx][
                    "hrf_recovery_rate"
                ]
                recovery_matrix[pattern_idx, true_hrf_idx] = recovery_rate

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 10))

        im = ax.imshow(
            recovery_matrix.T,
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            origin="lower",
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("HRF Recovery Rate", fontsize=12, fontweight="bold")

        # Axes labels
        ax.set_xlabel("Activation Pattern", fontsize=13, fontweight="bold")
        ax.set_ylabel("True HRF Index", fontsize=13, fontweight="bold")
        ax.set_title(
            f"HRF Recovery Rate by Pattern and True HRF (ISI = {isi:.2f}s)",
            fontsize=15,
            fontweight="bold",
            pad=15,
        )

        # Pattern labels on x-axis
        ax.set_xticks(range(n_patterns))
        pattern_labels = [
            f"P{i}\n[{activation_patterns[i][0]:.0f},{activation_patterns[i][1]:.0f}]"
            for i in range(n_patterns)
        ]
        ax.set_xticklabels(pattern_labels, fontsize=8, rotation=0)

        # HRF labels on y-axis
        ax.set_yticks(range(n_hrfs))
        ax.set_yticklabels([f"HRF {i}" for i in range(n_hrfs)], fontsize=8)

        # Grid
        ax.set_xticks(np.arange(n_patterns) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_hrfs) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

        # Add text annotations for values
        for pattern_idx in range(n_patterns):
            for hrf_idx in range(n_hrfs):
                value = recovery_matrix[pattern_idx, hrf_idx]
                # White text for dark cells, black for light cells
                text_color = "white" if value < 0.5 else "black"
                ax.text(
                    pattern_idx,
                    hrf_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                )

        plt.tight_layout()
        filename = output_dir / f"hrf_recovery_heatmap_isi_{isi:.2f}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close()

    print(f"\nCreated {len(isi_means)} HRF recovery heatmaps!")

    # =========================================================================
    # HRF SELECTION HEATMAPS: Show median selected HRF index (pattern × true HRF)
    # =========================================================================
    print(f"\nCreating HRF selection heatmaps (median selected HRF index)...")

    for isi_idx, isi in enumerate(isi_means):
        # Build selection matrix: (n_patterns × n_hrfs)
        # Shows the MEDIAN selected HRF index across realizations
        selection_matrix = np.zeros((n_patterns, n_hrfs))

        for pattern_idx in range(n_patterns):
            for true_hrf_idx in range(n_hrfs):
                best_indices = results[isi][pattern_idx][true_hrf_idx][
                    "best_hrf_idx_all"
                ]
                selection_matrix[pattern_idx, true_hrf_idx] = np.median(best_indices)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 10))

        # Use a diverging colormap centered on the diagonal
        im = ax.imshow(
            selection_matrix.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=0,
            vmax=n_hrfs - 1,
            origin="lower",
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Median Selected HRF Index", fontsize=12, fontweight="bold")

        # Axes labels
        ax.set_xlabel("Activation Pattern", fontsize=13, fontweight="bold")
        ax.set_ylabel("True HRF Index", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Median Selected HRF Index by Pattern and True HRF (ISI = {isi:.2f}s)",
            fontsize=15,
            fontweight="bold",
            pad=15,
        )

        # Pattern labels on x-axis
        ax.set_xticks(range(n_patterns))
        pattern_labels = [
            f"P{i}\n[{activation_patterns[i][0]:.0f},{activation_patterns[i][1]:.0f}]"
            for i in range(n_patterns)
        ]
        ax.set_xticklabels(pattern_labels, fontsize=8, rotation=0)

        # HRF labels on y-axis
        ax.set_yticks(range(n_hrfs))
        ax.set_yticklabels([f"HRF {i}" for i in range(n_hrfs)], fontsize=8)

        # Grid
        ax.set_xticks(np.arange(n_patterns) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_hrfs) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

        # Add diagonal line (perfect recovery)
        ax.plot(
            [-0.5, n_patterns - 0.5],
            [-0.5, n_hrfs - 0.5],
            "g--",
            linewidth=2,
            alpha=0.5,
            label="Perfect Recovery",
        )

        # Add text annotations showing median selected HRF
        for pattern_idx in range(n_patterns):
            for hrf_idx in range(n_hrfs):
                value = selection_matrix[pattern_idx, hrf_idx]
                error = abs(value - hrf_idx)  # Distance from true HRF

                # Color code: green for correct, yellow for close, red for far
                if error < 0.5:
                    text_color = "white"
                    bg_color = "green"
                elif error < 2:
                    text_color = "black"
                    bg_color = "yellow"
                else:
                    text_color = "white"
                    bg_color = "red"

                # Show selected index and error
                ax.text(
                    pattern_idx,
                    hrf_idx,
                    f"{int(value)}\n(Δ{error:.1f})",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor=bg_color,
                        alpha=0.3,
                        edgecolor="none",
                    ),
                )

        plt.tight_layout()
        filename = output_dir / f"hrf_selection_heatmap_isi_{isi:.2f}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close()

    print(f"\nCreated {len(isi_means)} HRF selection heatmaps!")


def plot_noise_spectrum(
    noise_samples: np.ndarray,
    tr: float,
    noise_std: float,
    baseline: float,
    output_dir,
    resp_freq: float = 0.3,
    cardiac_freq: float = 1.0,
):
    """
    Plot average noise power spectrum.

    Note: Each noise realization has different physiological parameters
    (frequency, width, strength vary within realistic bounds), so this
    plot shows the average spectrum across that variability.

    Parameters:
    -----------
    noise_samples : np.ndarray
        Shape (n_timepoints, n_realizations)
    tr : float
        Repetition time in seconds
    noise_std : float
        Standard deviation of noise
    baseline : float
        Baseline signal level
    output_dir : Path
        Directory to save plot
    resp_freq : float
        Nominal respiratory frequency (center of range)
    cardiac_freq : float
        Nominal cardiac frequency (center of range)
    """
    n_timepoints, n_realizations = noise_samples.shape

    # Compute FFT for each realization
    freqs = np.fft.rfftfreq(n_timepoints, d=tr)  # One-sided frequencies

    # Average power spectrum across realizations
    power_spectra = []
    for i in range(n_realizations):
        fft_vals = np.fft.rfft(noise_samples[:, i])
        power = np.abs(fft_vals) ** 2
        power_spectra.append(power)

    power_avg = np.mean(power_spectra, axis=0)
    power_std = np.std(power_spectra, axis=0)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Linear scale
    ax = axes[0]
    ax.plot(freqs, power_avg, "b-", linewidth=2, label="Mean power")
    ax.fill_between(
        freqs,
        power_avg - power_std,
        power_avg + power_std,
        alpha=0.3,
        color="blue",
        label="±1 SD",
    )

    # Mark physiological frequencies (nominal values - actual values vary per realization)
    nyquist = 0.5 / tr
    ax.axvline(
        resp_freq,
        color="green",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label=f"Respiratory\n(~{resp_freq} Hz nominal)",
    )

    # Cardiac aliases: For cardiac_freq=1.0 Hz at TR=1s (Nyquist=0.5 Hz),
    # the 1.0 Hz component is at exactly the sampling frequency and aliases to DC (0 Hz)
    fs_sample = 1.0 / tr
    cardiac_aliased = abs(cardiac_freq - round(cardiac_freq / fs_sample) * fs_sample)
    ax.axvline(
        cardiac_aliased,
        color="red",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label=f"Cardiac aliased\n(~{cardiac_freq:.1f} Hz → {cardiac_aliased:.2f} Hz nominal)",
    )

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Power", fontsize=12)
    ax.set_title(
        "Noise Power Spectrum (Linear Scale)\nAveraged across variable physiological parameters",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim([0, nyquist])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 2: Log-log scale
    ax = axes[1]
    # Skip DC component (freq=0) for log-log plot
    idx_nonzero = freqs > 0
    ax.loglog(
        freqs[idx_nonzero],
        power_avg[idx_nonzero],
        "b-",
        linewidth=2,
        label="Mean power",
    )
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Power", fontsize=12)
    ax.set_title("Noise Power Spectrum (Log-Log Scale)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, which="both")

    # Add reference lines for 1/f behavior
    f_ref = np.logspace(
        np.log10(freqs[idx_nonzero][0]), np.log10(freqs[idx_nonzero][-1]), 100
    )
    # Fit 1/f to data to get scaling
    p = np.polyfit(np.log10(freqs[idx_nonzero]), np.log10(power_avg[idx_nonzero]), 1)
    slope = p[0]
    intercept = p[1]
    fitted_power = 10 ** (intercept + slope * np.log10(f_ref))
    ax.plot(
        f_ref,
        fitted_power,
        "r--",
        linewidth=2,
        alpha=0.7,
        label=f"1/f^{-slope:.2f} fit",
    )
    ax.legend()

    # Add text info
    info_text = (
        f"Noise SD: {noise_std:.2f}\n"
        f"Baseline: {baseline:.1f}\n"
        f"Effective noise: {100 * noise_std / baseline:.1f}%\n"
        f"TR: {tr:.2f}s\n"
        f"Nyquist: {0.5 / tr:.2f} Hz\n"
        f"Spectral slope: {slope:.2f}\n"
        f"\nPhysiological (nominal):\n"
        f"  Resp: ~{resp_freq:.1f} Hz\n"
        f"    (varies 0.25-0.35 Hz)\n"
        f"  Cardiac: ~{cardiac_freq:.1f} Hz\n"
        f"    (varies 0.9-1.1 Hz)\n"
        f"    → aliases to ~{cardiac_aliased:.1f} Hz"
    )
    fig.text(
        0.98,
        0.98,
        info_text,
        transform=fig.transFigure,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "noise_spectrum.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'noise_spectrum.png'}")
    plt.close()

    # Print summary statistics
    print(f"\nNoise Spectrum Summary:")
    print(
        f"  Note: Averaged across {n_realizations} realizations with varying physiology"
    )
    print(f"  Spectral slope (log-log): {slope:.3f}")
    print(f"  Expected for 1/f noise: ~-1.0")
    print(
        f"  Mean power (0-0.1 Hz): {np.mean(power_avg[(freqs > 0) & (freqs < 0.1)]):.2e}"
    )
    print(
        f"  Mean power (0.1-0.5 Hz): {np.mean(power_avg[(freqs > 0.1) & (freqs < 0.5)]):.2e}"
    )
    print(f"  Physiological variability smooths peaks across realizations")


if __name__ == "__main__":
    device = ffs.get_device()
    output_dir = Path("monte_carlo_comprehensive_results")
    output_dir.mkdir(exist_ok=True)

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

    # Run with 1000 realizations
    results = run_full_study(
        isi_means=isi_means,
        activation_patterns=activation_patterns,
        hrf_library_name="cnvlab",
        n_noise_realizations=1000,
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
    if "noise_samples" in results[first_isi][0]:
        print(f"\nCreating noise spectrum plot...")
        noise_samples = results[first_isi][0]["noise_samples"]
        plot_noise_spectrum(noise_samples, 1.0, 2.0, 100.0, output_dir)

    # Plot
    print(f"\nCreating performance plots...")
    plot_comprehensive_results(results, activation_patterns, isi_means, output_dir)

    print(f"\n{'=' * 70}")
    print("ALL DONE!")
    print(f"{'=' * 70}")
    print(f"\nResults: {output_dir}/")
