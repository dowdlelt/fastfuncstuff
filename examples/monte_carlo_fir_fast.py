#!/usr/bin/env python
"""
FAST Monte Carlo FIR Study - GPU Batched

Optimizations:
1. Batch all noise realizations together (fit 100 datasets simultaneously)
2. Pre-compute design matrices once per ISI
3. GPU-batched OLS solving
4. Minimal CPU/GPU transfers

Should be ~100x faster than sequential version.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.insert(0, '/Users/logan/local_bin/fastfuncsim')
import fastfuncsim as ffs
from fastfuncsim.noise import generate_fmri_noise

from simulate_isi_sweep import (
    load_cnvlab_hrf_library,
    generate_poisson_isis,
    build_alternating_design,
    convolve_design_with_hrf,
    create_polynomial_regressors
)


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


def run_monte_carlo_batched(
    activation_pattern: np.ndarray,
    true_hrf_idx: int,
    hrfs: np.ndarray,
    design_unconvolved: np.ndarray,
    X_fir_full_torch: torch.Tensor,  # Pre-computed FIR design
    tr: float,
    n_noise_realizations: int,
    n_fir_bins: int,
    noise_std: float,
    baseline: float,
    device: torch.device,
    verbose: bool = False
) -> Dict:
    """
    Run Monte Carlo with BATCHED FIR fitting (all realizations at once).

    Key optimization: Generate all noise realizations, stack them, fit once.
    """
    n_timepoints = design_unconvolved.shape[0]
    n_conds = design_unconvolved.shape[1]

    if verbose:
        print(f"    Pattern {activation_pattern}: {n_noise_realizations} realizations (batched)...")

    # =========================================================================
    # Generate true signal
    # =========================================================================
    true_hrf = hrfs[true_hrf_idx]
    onset_vector = np.zeros(n_timepoints)
    for cond_idx in range(n_conds):
        onset_times = np.where(design_unconvolved[:, cond_idx] > 0)[0]
        onset_vector[onset_times] = activation_pattern[cond_idx]

    true_signal = np.convolve(onset_vector, true_hrf, mode='full')[:n_timepoints]
    true_signal += baseline

    # =========================================================================
    # Generate ALL noise realizations at once (batched)
    # =========================================================================
    noise_all = generate_fmri_noise(
        tr=tr,
        duration_s=n_timepoints * tr,
        matrix_size=(1, n_noise_realizations),
        normalize=True,
        device=device
    ).squeeze() * noise_std  # (n_timepoints, n_realizations)

    # Create data: signal + noise
    true_signal_torch = torch.from_numpy(true_signal).float().to(device)[:, None]  # (n_timepoints, 1)
    data_all = true_signal_torch + noise_all  # (n_timepoints, n_realizations)

    # =========================================================================
    # Parametric HRF fitting (for library selection)
    # =========================================================================
    n_hrfs = hrfs.shape[0]
    r2_per_hrf = []

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

        # R² per realization
        predicted = X_torch @ beta
        residuals = data_all - predicted
        ss_res = torch.sum(residuals ** 2, dim=0)
        ss_tot = torch.sum((data_all - data_all.mean(dim=0, keepdim=True)) ** 2, dim=0)
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)  # (n_realizations,)

        r2_per_hrf.append(r2.cpu().numpy())

    r2_per_hrf = np.array(r2_per_hrf)  # (n_hrfs, n_realizations)
    best_hrf_idx_per_realization = np.argmax(r2_per_hrf, axis=0)  # (n_realizations,)
    hrf_correct = (best_hrf_idx_per_realization == true_hrf_idx).astype(float)
    r2_parametric = r2_per_hrf[best_hrf_idx_per_realization, np.arange(n_noise_realizations)]

    # =========================================================================
    # FIR model fitting (BATCHED across all realizations)
    # =========================================================================
    XTX = X_fir_full_torch.T @ X_fir_full_torch
    XTY = X_fir_full_torch.T @ data_all
    ridge = 1e-6
    XTX_reg = XTX + ridge * torch.eye(XTX.shape[0], device=device)
    beta_fir_full = torch.linalg.solve(XTX_reg, XTY)  # (n_params, n_realizations)

    # R² for FIR model
    predicted_fir = X_fir_full_torch @ beta_fir_full
    residuals_fir = data_all - predicted_fir
    ss_res_fir = torch.sum(residuals_fir ** 2, dim=0)
    ss_tot_fir = torch.sum((data_all - data_all.mean(dim=0, keepdim=True)) ** 2, dim=0)
    r2_fir = 1.0 - ss_res_fir / (ss_tot_fir + 1e-10)
    r2_fir = r2_fir.cpu().numpy()  # (n_realizations,)

    # Extract FIR betas (exclude polynomial regressors)
    n_fir_params = n_conds * n_fir_bins
    beta_fir = beta_fir_full[:n_fir_params]  # (n_fir_params, n_realizations)

    # =========================================================================
    # Normalize HRFs by true betas and compute correlations
    # =========================================================================
    # Truncate true HRF to match FIR bins
    if len(true_hrf) > n_fir_bins:
        true_hrf_matched = true_hrf[:n_fir_bins]
    else:
        true_hrf_matched = np.pad(true_hrf, (0, n_fir_bins - len(true_hrf)), mode='constant')

    true_hrf_torch = torch.from_numpy(true_hrf_matched).float().to(device)

    # Condition A
    beta_A = beta_fir[:n_fir_bins]  # (n_fir_bins, n_realizations)
    if np.abs(activation_pattern[0]) > 1e-10:
        beta_A_norm = beta_A / activation_pattern[0]
        # Correlation per realization
        corr_A = []
        for real_idx in range(n_noise_realizations):
            hrf_est = beta_A_norm[:, real_idx]
            corr = torch.corrcoef(torch.stack([hrf_est, true_hrf_torch]))[0, 1]
            corr_A.append(corr.item() if not torch.isnan(corr) else 0.0)
        corr_A = np.array(corr_A)
    else:
        corr_A = np.full(n_noise_realizations, np.nan)

    # Condition B
    beta_B = beta_fir[n_fir_bins:2*n_fir_bins]  # (n_fir_bins, n_realizations)
    if np.abs(activation_pattern[1]) > 1e-10:
        beta_B_norm = beta_B / activation_pattern[1]
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
        'hrf_recovery_rate': hrf_correct.mean(),
        'r2_parametric_median': np.median(r2_parametric),
        'r2_parametric_iqr': [np.percentile(r2_parametric, 25), np.percentile(r2_parametric, 75)],

        # FIR model fit quality
        'r2_fir_median': np.median(r2_fir),
        'r2_fir_iqr': [np.percentile(r2_fir, 25), np.percentile(r2_fir, 75)],

        # FIR HRF shape recovery
        'hrf_corr_A_median': np.nanmedian(corr_A),
        'hrf_corr_A_iqr': [np.nanpercentile(corr_A, 25), np.nanpercentile(corr_A, 75)],
        'hrf_corr_B_median': np.nanmedian(corr_B),
        'hrf_corr_B_iqr': [np.nanpercentile(corr_B, 25), np.nanpercentile(corr_B, 75)],

        # Raw data
        'hrf_corr_A_all': corr_A,
        'hrf_corr_B_all': corr_B,
        'r2_fir_all': r2_fir,
    }

    return results


def run_full_study(
    isi_means: List[float],
    activation_patterns: np.ndarray,
    hrf_library_name: str = 'cnvlab',
    n_noise_realizations: int = 100,
    n_fir_bins: int = 30,
    noise_std: float = 2.0,
    tr: float = 1.0,
    total_duration: float = 290.0,
    stim_duration: float = 5.0,
    baseline: float = 100.0,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict:
    """Run full study with batched fitting."""
    if device is None:
        device = ffs.get_device()

    if verbose:
        print(f"{'='*70}")
        print("FAST BATCHED MONTE CARLO FIR STUDY")
        print(f"{'='*70}\n")
        print(f"Device: {device}")
        print(f"ISI means: {isi_means}")
        print(f"Patterns: {len(activation_patterns)}")
        print(f"Realizations: {n_noise_realizations}")
        print(f"FIR bins: {n_fir_bins}\n")

    # Load HRF library
    if hrf_library_name == 'cnvlab':
        hrfs = load_cnvlab_hrf_library(duration=stim_duration, tr=tr)
    else:
        raise ValueError(f"Unknown HRF library: {hrf_library_name}")

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
            print(f"\n{'='*70}")
            print(f"ISI: {isi_mean}s ({isi_idx+1}/{len(isi_means)})")
            print(f"{'='*70}\n")

        # Generate design
        block_average = isi_mean + stim_duration
        n_trials = int((total_duration - 2 * padding_tr * tr) / block_average)
        n_trials = n_trials - (n_trials % 2)

        isis_sec = generate_poisson_isis(
            target_mean=isi_mean,
            n_isis=n_trials,
            lower_limit=2.0,
            upper_limit=8.0,
            seed=42 + isi_idx
        )
        isis_tr = np.round(isis_sec / tr).astype(int)

        design_unconvolved = build_alternating_design(
            isis_tr=isis_tr,
            stim_dur_tr=stim_tr,
            n_conds=2,
            total_tr=total_tr,
            padding_tr=padding_tr
        )

        # Pre-compute FIR design (once per ISI)
        X_fir = create_fir_design(design_unconvolved, n_fir_bins)
        poly = create_polynomial_regressors(X_fir.shape[0], max_order=3)
        X_fir_full = np.hstack([X_fir, poly]).astype(np.float32)
        X_fir_full_torch = torch.from_numpy(X_fir_full).float().to(device)

        if verbose:
            print(f"  Design: {np.sum(design_unconvolved[:, 0]):.0f} + {np.sum(design_unconvolved[:, 1]):.0f} trials\n")

        results[isi_mean] = {}

        for pattern_idx, pattern in enumerate(activation_patterns):
            true_hrf_idx = pattern_idx % n_hrfs

            pattern_results = run_monte_carlo_batched(
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
                verbose=verbose
            )

            results[isi_mean][pattern_idx] = pattern_results

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n{'='*70}")
        print("COMPLETE!")
        print(f"{'='*70}\n")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Time per ISI: {elapsed/len(isi_means):.1f}s")

    return results


def plot_results(results, activation_patterns, isi_means, output_dir):
    """Plot with error shading."""
    n_patterns = len(activation_patterns)
    n_rows, n_cols = 4, 5

    # FIR HRF Recovery - Condition A
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('FIR HRF Shape Recovery: Condition A', fontsize=18, fontweight='bold', y=0.995)

    for idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        pattern = activation_patterns[idx]

        medians, lower, upper = [], [], []
        for isi in isi_means:
            medians.append(results[isi][idx]['hrf_corr_A_median'])
            iqr = results[isi][idx]['hrf_corr_A_iqr']
            lower.append(iqr[0])
            upper.append(iqr[1])

        ax.plot(isi_means, medians, 'o-', linewidth=2, markersize=8, color='blue')
        ax.fill_between(isi_means, lower, upper, alpha=0.3, color='blue')
        ax.set_ylim([-0.1, 1.05])
        ax.set_title(f'P{idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]', fontsize=10, fontweight='bold')
        ax.set_xlabel('ISI (s)', fontsize=9)
        ax.set_ylabel('Correlation', fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'fir_recovery_condA.png', dpi=150)
    print(f"Saved: {output_dir / 'fir_recovery_condA.png'}")
    plt.close()

    # FIR HRF Recovery - Condition B
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('FIR HRF Shape Recovery: Condition B', fontsize=18, fontweight='bold', y=0.995)

    for idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        pattern = activation_patterns[idx]

        medians, lower, upper = [], [], []
        for isi in isi_means:
            medians.append(results[isi][idx]['hrf_corr_B_median'])
            iqr = results[isi][idx]['hrf_corr_B_iqr']
            lower.append(iqr[0])
            upper.append(iqr[1])

        ax.plot(isi_means, medians, 'o-', linewidth=2, markersize=8, color='orange')
        ax.fill_between(isi_means, lower, upper, alpha=0.3, color='orange')
        ax.set_ylim([-0.1, 1.05])
        ax.set_title(f'P{idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]', fontsize=10, fontweight='bold')
        ax.set_xlabel('ISI (s)', fontsize=9)
        ax.set_ylabel('Correlation', fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'fir_recovery_condB.png', dpi=150)
    print(f"Saved: {output_dir / 'fir_recovery_condB.png'}")
    plt.close()


if __name__ == "__main__":
    device = ffs.get_device()
    output_dir = Path("monte_carlo_fir_results")
    output_dir.mkdir(exist_ok=True)

    isi_means = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    activation_patterns = np.array([
        [5, 1], [5, 2], [5, 3], [5, 4], [5, 5],
        [4, 5], [3, 5], [2, 5], [1, 5],
        [1, 3], [3, 3], [3, 1], [1, 1],
        [0, 2], [2, 0],
        [-1, -1], [-3, -3],
        [-3, 4], [4, -3],
        [0, 0]
    ], dtype=np.float32)

    results = run_full_study(
        isi_means=isi_means,
        activation_patterns=activation_patterns,
        hrf_library_name='cnvlab',
        n_noise_realizations=100,
        n_fir_bins=30,
        noise_std=2.0,
        tr=1.0,
        total_duration=290.0,
        stim_duration=5.0,
        baseline=100.0,
        device=device,
        verbose=True
    )

    with open(output_dir / 'fir_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {output_dir / 'fir_results.pkl'}")

    plot_results(results, activation_patterns, isi_means, output_dir)
    print("\nDONE!")
