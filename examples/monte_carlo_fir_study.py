#!/usr/bin/env python
"""
Monte Carlo ISI Study with FIR HRF Recovery

Extends parametric HRF fitting with FIR (Finite Impulse Response) model to directly
test HRF shape recovery quality:

1. Fit FIR model (free-form HRF estimation, no parametric assumption)
2. Extract separate HRF estimates for conditions A and B
3. Normalize by true beta: HRF_est_A / β_true_A, HRF_est_B / β_true_B
4. Should recover original HRF shape (height=1) if estimation is perfect
5. Correlate normalized estimates with true HRF
6. This R² reflects noise impact on shape recovery across 100 realizations

With error shading on all plots!
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle

import sys
sys.path.insert(0, '/Users/logan/local_bin/fastfuncsim')
import fastfuncsim as ffs
from fastfuncsim.noise import generate_fmri_noise

# Import functions from simulate_isi_sweep
from simulate_isi_sweep import (
    load_cnvlab_hrf_library,
    generate_poisson_isis,
    build_alternating_design,
    convolve_design_with_hrf,
    create_polynomial_regressors
)


# =============================================================================
# FIR Model Functions
# =============================================================================

def create_fir_design(design_unconvolved: np.ndarray, n_fir_bins: int) -> np.ndarray:
    """
    Create FIR design matrix with separate time bins for each condition.

    Parameters
    ----------
    design_unconvolved : ndarray
        Onset matrix, shape (n_timepoints, n_conds)
    n_fir_bins : int
        Number of time bins for FIR model (e.g., 30 for 30-TR HRF)

    Returns
    -------
    X_fir : ndarray
        FIR design matrix, shape (n_timepoints, n_conds * n_fir_bins)
    """
    n_timepoints, n_conds = design_unconvolved.shape
    X_fir = np.zeros((n_timepoints, n_conds * n_fir_bins), dtype=np.float32)

    for cond_idx in range(n_conds):
        # Get onset times for this condition
        onsets = np.where(design_unconvolved[:, cond_idx] > 0)[0]

        # Create FIR regressors
        for bin_idx in range(n_fir_bins):
            col_idx = cond_idx * n_fir_bins + bin_idx

            # Each onset contributes to this time bin
            for onset in onsets:
                if onset + bin_idx < n_timepoints:
                    X_fir[onset + bin_idx, col_idx] += 1.0

    return X_fir


def fit_fir_model(
    data: torch.Tensor,
    design_unconvolved: np.ndarray,
    n_fir_bins: int,
    device: torch.device
) -> Tuple[torch.Tensor, float]:
    """
    Fit FIR model with polynomial nuisance regressors.

    Parameters
    ----------
    data : torch.Tensor
        fMRI data, shape (n_timepoints, n_voxels)
    design_unconvolved : ndarray
        Onset matrix
    n_fir_bins : int
        Number of FIR time bins
    device : torch.device
        Computation device

    Returns
    -------
    beta_fir : torch.Tensor
        FIR beta estimates, shape (n_conds * n_fir_bins, n_voxels)
    r2 : float
        Model R²
    """
    # Create FIR design
    X_fir = create_fir_design(design_unconvolved, n_fir_bins)

    # Add polynomial nuisance regressors
    poly = create_polynomial_regressors(X_fir.shape[0], max_order=3)
    X_full = np.hstack([X_fir, poly]).astype(np.float32)

    X_torch = torch.from_numpy(X_full).float().to(device)

    # OLS fitting
    XTX = X_torch.T @ X_torch
    XTY = X_torch.T @ data
    ridge = 1e-6
    XTX_reg = XTX + ridge * torch.eye(XTX.shape[0], device=device)
    beta = torch.linalg.solve(XTX_reg, XTY)

    # R² calculation
    predicted = X_torch @ beta
    residuals = data - predicted
    ss_res = torch.sum(residuals ** 2, dim=0)
    ss_tot = torch.sum((data - data.mean(dim=0, keepdim=True)) ** 2, dim=0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    # Extract FIR betas only (exclude polynomial regressors)
    n_fir_params = X_fir.shape[1]
    beta_fir = beta[:n_fir_params]

    return beta_fir, r2.mean().item()


def extract_condition_hrfs(
    beta_fir: torch.Tensor,
    n_conds: int,
    n_fir_bins: int
) -> List[np.ndarray]:
    """
    Extract HRF estimates for each condition from FIR betas.

    Parameters
    ----------
    beta_fir : torch.Tensor
        FIR betas, shape (n_conds * n_fir_bins, n_voxels)
    n_conds : int
        Number of conditions
    n_fir_bins : int
        Number of FIR time bins

    Returns
    -------
    hrfs : list of ndarray
        List of HRF estimates, each shape (n_fir_bins, n_voxels)
    """
    beta_np = beta_fir.cpu().numpy()
    hrfs = []

    for cond_idx in range(n_conds):
        start_idx = cond_idx * n_fir_bins
        end_idx = start_idx + n_fir_bins
        hrf_cond = beta_np[start_idx:end_idx, :]  # (n_fir_bins, n_voxels)
        hrfs.append(hrf_cond)

    return hrfs


def normalize_hrf_by_beta(hrf_est: np.ndarray, true_beta: float) -> np.ndarray:
    """
    Normalize HRF estimate by true beta magnitude.

    If true_beta is 0, return nans (cannot normalize).

    Parameters
    ----------
    hrf_est : ndarray
        HRF estimate, shape (n_fir_bins, n_voxels)
    true_beta : float
        True activation magnitude

    Returns
    -------
    hrf_norm : ndarray
        Normalized HRF (should match input HRF shape with peak~1)
    """
    if np.abs(true_beta) < 1e-10:
        # Cannot normalize by zero
        return np.full_like(hrf_est, np.nan)

    return hrf_est / true_beta


def compute_hrf_correlation(
    hrf_est_norm: np.ndarray,
    hrf_true: np.ndarray
) -> np.ndarray:
    """
    Compute correlation between normalized HRF estimate and true HRF.

    Parameters
    ----------
    hrf_est_norm : ndarray
        Normalized HRF estimate, shape (n_fir_bins, n_voxels)
    hrf_true : ndarray
        True HRF, shape (hrf_length,)

    Returns
    -------
    correlations : ndarray
        Correlation per voxel, shape (n_voxels,)
    """
    n_fir_bins, n_voxels = hrf_est_norm.shape

    # Truncate or pad true HRF to match FIR bins
    if len(hrf_true) > n_fir_bins:
        hrf_true_matched = hrf_true[:n_fir_bins]
    else:
        hrf_true_matched = np.pad(hrf_true, (0, n_fir_bins - len(hrf_true)), mode='constant')

    # Compute correlation for each voxel
    correlations = np.zeros(n_voxels, dtype=np.float32)
    for vox_idx in range(n_voxels):
        hrf_est_vox = hrf_est_norm[:, vox_idx]

        # Skip if any nans
        if np.any(np.isnan(hrf_est_vox)):
            correlations[vox_idx] = np.nan
            continue

        # Pearson correlation
        corr = np.corrcoef(hrf_est_vox, hrf_true_matched)[0, 1]
        correlations[vox_idx] = corr if not np.isnan(corr) else 0.0

    return correlations


# =============================================================================
# Monte Carlo Simulation with FIR
# =============================================================================

def run_monte_carlo_with_fir(
    activation_pattern: np.ndarray,
    true_hrf_idx: int,
    hrfs: np.ndarray,
    design_unconvolved: np.ndarray,
    tr: float,
    n_voxels: int,
    n_noise_realizations: int,
    n_fir_bins: int,
    noise_std: float,
    baseline: float,
    device: torch.device,
    verbose: bool = False
) -> Dict:
    """
    Run Monte Carlo simulation with both parametric and FIR HRF fitting.

    Returns
    -------
    results : dict
        Contains:
        - Parametric HRF recovery stats
        - FIR HRF shape recovery correlations (per condition)
        - R² from FIR model
    """
    n_timepoints = design_unconvolved.shape[0]
    n_conds = design_unconvolved.shape[1]
    n_hrfs = hrfs.shape[0]

    if verbose:
        print(f"    Pattern {activation_pattern}: {n_noise_realizations} realizations...")

    # Generate true signal
    true_hrf = hrfs[true_hrf_idx]
    onset_vector = np.zeros(n_timepoints)
    for cond_idx in range(n_conds):
        onset_times = np.where(design_unconvolved[:, cond_idx] > 0)[0]
        onset_vector[onset_times] = activation_pattern[cond_idx]

    true_signal = np.convolve(onset_vector, true_hrf, mode='full')[:n_timepoints]
    true_signal_voxels = np.tile(true_signal, (n_voxels, 1)).T + baseline

    # Storage for results
    hrf_correct = []
    r2_parametric = []
    r2_fir = []
    hrf_corr_A_all = []  # Correlations for condition A
    hrf_corr_B_all = []  # Correlations for condition B

    # Run Monte Carlo realizations
    for realization_idx in range(n_noise_realizations):
        # Generate noise
        noise_2d = generate_fmri_noise(
            tr=tr,
            duration_s=n_timepoints * tr,
            matrix_size=(1, n_voxels),
            normalize=True,
            device=device
        )
        noise_np = noise_2d.cpu().numpy().squeeze() * noise_std

        # Create data
        data = true_signal_voxels + noise_np
        Y_torch = torch.from_numpy(data).float().to(device)

        # ===================================================================
        # 1. Parametric HRF fitting (for library selection)
        # ===================================================================
        r2_per_hrf = []
        for hrf_idx, hrf in enumerate(hrfs):
            X = convolve_design_with_hrf(design_unconvolved, hrf)
            poly = create_polynomial_regressors(X.shape[0], max_order=3)
            X_full = np.hstack([X, poly]).astype(np.float32)
            X_torch = torch.from_numpy(X_full).float().to(device)

            XTX = X_torch.T @ X_torch
            XTY = X_torch.T @ Y_torch
            ridge = 1e-6
            XTX_reg = XTX + ridge * torch.eye(XTX.shape[0], device=device)
            beta = torch.linalg.solve(XTX_reg, XTY)

            predicted = X_torch @ beta
            residuals = Y_torch - predicted
            ss_res = torch.sum(residuals ** 2, dim=0)
            ss_tot = torch.sum((Y_torch - Y_torch.mean(dim=0, keepdim=True)) ** 2, dim=0)
            r2 = 1.0 - ss_res / (ss_tot + 1e-10)

            r2_per_hrf.append(r2.mean().item())

        best_hrf_idx = np.argmax(r2_per_hrf)
        hrf_correct.append(best_hrf_idx == true_hrf_idx)
        r2_parametric.append(r2_per_hrf[best_hrf_idx])

        # ===================================================================
        # 2. FIR model fitting
        # ===================================================================
        beta_fir, r2_fir_val = fit_fir_model(
            data=Y_torch,
            design_unconvolved=design_unconvolved,
            n_fir_bins=n_fir_bins,
            device=device
        )
        r2_fir.append(r2_fir_val)

        # Extract HRF estimates for each condition
        hrf_estimates = extract_condition_hrfs(beta_fir, n_conds, n_fir_bins)

        # Normalize by true betas and compute correlations
        # Condition A
        hrf_est_A = hrf_estimates[0]  # (n_fir_bins, n_voxels)
        hrf_est_A_norm = normalize_hrf_by_beta(hrf_est_A, activation_pattern[0])
        corr_A = compute_hrf_correlation(hrf_est_A_norm, true_hrf)
        hrf_corr_A_all.append(corr_A)  # (n_voxels,)

        # Condition B
        hrf_est_B = hrf_estimates[1]
        hrf_est_B_norm = normalize_hrf_by_beta(hrf_est_B, activation_pattern[1])
        corr_B = compute_hrf_correlation(hrf_est_B_norm, true_hrf)
        hrf_corr_B_all.append(corr_B)

    # Compute statistics across realizations
    hrf_correct = np.array(hrf_correct)
    r2_parametric = np.array(r2_parametric)
    r2_fir = np.array(r2_fir)
    hrf_corr_A_all = np.array(hrf_corr_A_all)  # (n_realizations, n_voxels)
    hrf_corr_B_all = np.array(hrf_corr_B_all)

    # Aggregate correlation across voxels (median per realization)
    # Then compute statistics across realizations
    hrf_corr_A_median_per_realization = np.nanmedian(hrf_corr_A_all, axis=1)
    hrf_corr_B_median_per_realization = np.nanmedian(hrf_corr_B_all, axis=1)

    results = {
        # Parametric HRF recovery
        'hrf_recovery_rate': hrf_correct.mean(),
        'r2_parametric_median': np.median(r2_parametric),
        'r2_parametric_iqr': [np.percentile(r2_parametric, 25), np.percentile(r2_parametric, 75)],

        # FIR model fit quality
        'r2_fir_median': np.median(r2_fir),
        'r2_fir_iqr': [np.percentile(r2_fir, 25), np.percentile(r2_fir, 75)],

        # FIR HRF shape recovery (correlation with true HRF after normalization)
        'hrf_corr_A_median': np.nanmedian(hrf_corr_A_median_per_realization),
        'hrf_corr_A_iqr': [
            np.nanpercentile(hrf_corr_A_median_per_realization, 25),
            np.nanpercentile(hrf_corr_A_median_per_realization, 75)
        ],
        'hrf_corr_B_median': np.nanmedian(hrf_corr_B_median_per_realization),
        'hrf_corr_B_iqr': [
            np.nanpercentile(hrf_corr_B_median_per_realization, 25),
            np.nanpercentile(hrf_corr_B_median_per_realization, 75)
        ],

        # Raw data for plotting
        'hrf_corr_A_all': hrf_corr_A_median_per_realization,
        'hrf_corr_B_all': hrf_corr_B_median_per_realization,
        'r2_fir_all': r2_fir,
    }

    return results


def run_full_study(
    isi_means: List[float],
    activation_patterns: np.ndarray,
    hrf_library_name: str = 'cnvlab',
    n_noise_realizations: int = 100,
    n_voxels_per_realization: int = 1000,
    n_fir_bins: int = 30,
    noise_std: float = 2.0,
    tr: float = 1.0,
    total_duration: float = 290.0,
    stim_duration: float = 5.0,
    baseline: float = 100.0,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict:
    """Run full Monte Carlo study with FIR analysis."""
    if device is None:
        device = ffs.get_device()

    if verbose:
        print(f"{'='*70}")
        print("MONTE CARLO ISI STUDY WITH FIR HRF RECOVERY")
        print(f"{'='*70}\n")
        print(f"Device: {device}")
        print(f"ISI means: {isi_means}")
        print(f"Activation patterns: {len(activation_patterns)}")
        print(f"Noise realizations: {n_noise_realizations}")
        print(f"FIR bins: {n_fir_bins}")
        print(f"Noise std: {noise_std}\n")

    # Load HRF library
    if hrf_library_name == 'cnvlab':
        hrfs = load_cnvlab_hrf_library(duration=stim_duration, tr=tr)
    else:
        raise ValueError(f"Unknown HRF library: {hrf_library_name}")

    n_hrfs = hrfs.shape[0]
    if verbose:
        print(f"Loaded {n_hrfs} HRFs from {hrf_library_name} library\n")

    # Calculate durations
    total_tr = int(total_duration / tr)
    stim_tr = int(stim_duration / tr)
    padding_tr = int(10 / tr)

    results = {}
    start_time = time.time()

    # Loop over ISI means
    for isi_idx, isi_mean in enumerate(isi_means):
        if verbose:
            print(f"\n{'='*70}")
            print(f"ISI MEAN: {isi_mean}s ({isi_idx+1}/{len(isi_means)})")
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

        if verbose:
            print(f"  Design: {np.sum(design_unconvolved[:, 0]):.0f} + {np.sum(design_unconvolved[:, 1]):.0f} trials\n")

        results[isi_mean] = {}

        # Loop over patterns
        for pattern_idx, pattern in enumerate(activation_patterns):
            true_hrf_idx = pattern_idx % n_hrfs

            pattern_results = run_monte_carlo_with_fir(
                activation_pattern=pattern,
                true_hrf_idx=true_hrf_idx,
                hrfs=hrfs,
                design_unconvolved=design_unconvolved,
                tr=tr,
                n_voxels=n_voxels_per_realization,
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
        print("STUDY COMPLETE!")
        print(f"{'='*70}\n")
        print(f"Total time: {elapsed/60:.1f} minutes")

    return results


# =============================================================================
# Visualization with Error Shading
# =============================================================================

def plot_fir_results(
    results: Dict,
    activation_patterns: np.ndarray,
    isi_means: List[float],
    output_dir: Path
):
    """Create plots with error shading (IQR bands)."""
    n_patterns = len(activation_patterns)
    n_rows = 4
    n_cols = 5

    # =========================================================================
    # Figure 1: FIR HRF Shape Recovery (Condition A)
    # =========================================================================
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('FIR HRF Shape Recovery: Condition A (Correlation with True HRF)',
                 fontsize=18, fontweight='bold', y=0.995)

    for pattern_idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, pattern_idx + 1)
        pattern = activation_patterns[pattern_idx]

        medians = []
        lower_iqr = []
        upper_iqr = []

        for isi in isi_means:
            med = results[isi][pattern_idx]['hrf_corr_A_median']
            iqr = results[isi][pattern_idx]['hrf_corr_A_iqr']
            medians.append(med)
            lower_iqr.append(iqr[0])
            upper_iqr.append(iqr[1])

        ax.plot(isi_means, medians, 'o-', linewidth=2, markersize=8, color='blue')
        ax.fill_between(isi_means, lower_iqr, upper_iqr, alpha=0.3, color='blue')
        ax.set_ylim([-0.1, 1.05])
        ax.set_title(f'Pattern {pattern_idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('ISI (s)', fontsize=9)
        ax.set_ylabel('Correlation', fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_dir / 'fir_hrf_recovery_condA.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fir_hrf_recovery_condA.png'}")
    plt.close()

    # =========================================================================
    # Figure 2: FIR HRF Shape Recovery (Condition B)
    # =========================================================================
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('FIR HRF Shape Recovery: Condition B (Correlation with True HRF)',
                 fontsize=18, fontweight='bold', y=0.995)

    for pattern_idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, pattern_idx + 1)
        pattern = activation_patterns[pattern_idx]

        medians = []
        lower_iqr = []
        upper_iqr = []

        for isi in isi_means:
            med = results[isi][pattern_idx]['hrf_corr_B_median']
            iqr = results[isi][pattern_idx]['hrf_corr_B_iqr']
            medians.append(med)
            lower_iqr.append(iqr[0])
            upper_iqr.append(iqr[1])

        ax.plot(isi_means, medians, 'o-', linewidth=2, markersize=8, color='orange')
        ax.fill_between(isi_means, lower_iqr, upper_iqr, alpha=0.3, color='orange')
        ax.set_ylim([-0.1, 1.05])
        ax.set_title(f'Pattern {pattern_idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('ISI (s)', fontsize=9)
        ax.set_ylabel('Correlation', fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_dir / 'fir_hrf_recovery_condB.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fir_hrf_recovery_condB.png'}")
    plt.close()

    # =========================================================================
    # Figure 3: Library HRF Recovery Rate
    # =========================================================================
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Parametric HRF Recovery Rate (Library Selection)',
                 fontsize=18, fontweight='bold', y=0.995)

    for pattern_idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, pattern_idx + 1)
        pattern = activation_patterns[pattern_idx]

        recovery_rates = [results[isi][pattern_idx]['hrf_recovery_rate']
                          for isi in isi_means]

        ax.plot(isi_means, recovery_rates, 'o-', linewidth=2, markersize=8, color='green')
        ax.set_ylim([0, 1.05])
        ax.set_title(f'Pattern {pattern_idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('ISI (s)', fontsize=9)
        ax.set_ylabel('Recovery Rate', fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_dir / 'parametric_hrf_recovery.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'parametric_hrf_recovery.png'}")
    plt.close()

    # =========================================================================
    # Figure 4: FIR R² (with IQR)
    # =========================================================================
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('FIR Model R² (Median with IQR)', fontsize=18, fontweight='bold', y=0.995)

    for pattern_idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, pattern_idx + 1)
        pattern = activation_patterns[pattern_idx]

        medians = []
        lower_iqr = []
        upper_iqr = []

        for isi in isi_means:
            med = results[isi][pattern_idx]['r2_fir_median']
            iqr = results[isi][pattern_idx]['r2_fir_iqr']
            medians.append(med)
            lower_iqr.append(iqr[0])
            upper_iqr.append(iqr[1])

        ax.plot(isi_means, medians, 'o-', linewidth=2, markersize=8, color='purple')
        ax.fill_between(isi_means, lower_iqr, upper_iqr, alpha=0.3, color='purple')
        ax.set_ylim([0, 1.05])
        ax.set_title(f'Pattern {pattern_idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('ISI (s)', fontsize=9)
        ax.set_ylabel('R² (FIR)', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fir_r2.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fir_r2.png'}")
    plt.close()


# =============================================================================
# Main Execution
# =============================================================================

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

    # Run study
    results = run_full_study(
        isi_means=isi_means,
        activation_patterns=activation_patterns,
        hrf_library_name='cnvlab',
        n_noise_realizations=100,
        n_voxels_per_realization=1000,
        n_fir_bins=30,  # 30 TRs = 30s HRF estimate
        noise_std=2.0,
        tr=1.0,
        total_duration=290.0,
        stim_duration=5.0,
        baseline=100.0,
        device=device,
        verbose=True
    )

    # Save results
    results_file = output_dir / 'fir_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {results_file}")

    # Plot
    print(f"\nCreating plots...")
    plot_fir_results(results, activation_patterns, isi_means, output_dir)

    print(f"\n{'='*70}")
    print("ALL DONE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}/")
