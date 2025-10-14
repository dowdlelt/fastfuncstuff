#!/usr/bin/env python
"""
Monte Carlo ISI Simulation Study

Systematic evaluation of HRF recovery and detection power across:
- ISI means: [2.0, 2.5, 3.0, 3.5, 4.0, 4.5] seconds
- Activation patterns: 20 different [A, B] magnitude combinations
- 100 independent noise realizations per condition

Focus on per-pattern statistics (NOT averaged across patterns):
- HRF recovery rate
- R² (median, IQR)
- Beta estimation error
- Detection power

GPU-accelerated for speed. No NIfTI output (statistics only).
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
# Fast Monte Carlo Simulation
# =============================================================================

def run_monte_carlo_simulation(
    activation_pattern: np.ndarray,
    true_hrf_idx: int,
    hrfs: np.ndarray,
    design_unconvolved: np.ndarray,
    tr: float,
    n_voxels: int,
    n_noise_realizations: int,
    noise_std: float,
    baseline: float,
    device: torch.device,
    verbose: bool = False
) -> Dict:
    """
    Run Monte Carlo simulation for ONE activation pattern.

    Parameters
    ----------
    activation_pattern : ndarray
        [beta_A, beta_B] true activation magnitudes
    true_hrf_idx : int
        Index of true HRF used to generate data
    hrfs : ndarray
        HRF library, shape (n_hrfs, hrf_length)
    design_unconvolved : ndarray
        Unconvolved design matrix, shape (n_timepoints, n_conds)
    tr : float
        Repetition time
    n_voxels : int
        Number of voxels per realization
    n_noise_realizations : int
        Number of independent noise realizations
    noise_std : float
        Noise standard deviation
    baseline : float
        Baseline signal level
    device : torch.device
        Device for computation
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Statistics across Monte Carlo realizations
    """
    n_timepoints = design_unconvolved.shape[0]
    n_conds = design_unconvolved.shape[1]
    n_hrfs = hrfs.shape[0]

    if verbose:
        print(f"    Pattern {activation_pattern}: {n_noise_realizations} realizations...")

    # Generate true signal (same for all realizations)
    true_hrf = hrfs[true_hrf_idx]

    # Create onset vector with activations
    onset_vector = np.zeros(n_timepoints)
    for cond_idx in range(n_conds):
        onset_times = np.where(design_unconvolved[:, cond_idx] > 0)[0]
        onset_vector[onset_times] = activation_pattern[cond_idx]

    # Convolve with true HRF
    true_signal = np.convolve(onset_vector, true_hrf, mode='full')[:n_timepoints]

    # Replicate for all voxels (all voxels same signal)
    true_signal_voxels = np.tile(true_signal, (n_voxels, 1)).T  # (n_timepoints, n_voxels)

    # Add baseline
    true_signal_voxels = true_signal_voxels + baseline

    # Storage for results across realizations
    hrf_correct = []
    r2_values = []
    beta_estimates_A = []
    beta_estimates_B = []
    beta_std_A = []
    beta_std_B = []

    # Run Monte Carlo realizations
    for realization_idx in range(n_noise_realizations):
        # Generate realistic fMRI noise for this realization
        noise_2d = generate_fmri_noise(
            tr=tr,
            duration_s=n_timepoints * tr,
            matrix_size=(1, n_voxels),
            normalize=True,
            device=device
        )

        # Convert to numpy and scale
        noise_np = noise_2d.cpu().numpy().squeeze() * noise_std  # (n_timepoints, n_voxels)

        # Create data: signal + noise
        data = true_signal_voxels + noise_np

        # Convert to torch
        Y_torch = torch.from_numpy(data).float().to(device)

        # Fit OLS with each HRF in library
        r2_per_hrf = []
        betas_per_hrf = []

        for hrf_idx, hrf in enumerate(hrfs):
            # Convolve design with this HRF
            X = convolve_design_with_hrf(design_unconvolved, hrf)

            # Add polynomial regressors
            poly = create_polynomial_regressors(X.shape[0], max_order=3)
            X_full = np.hstack([X, poly]).astype(np.float32)

            X_torch = torch.from_numpy(X_full).float().to(device)

            # Fit OLS
            XTX = X_torch.T @ X_torch
            XTY = X_torch.T @ Y_torch
            ridge = 1e-6
            XTX_reg = XTX + ridge * torch.eye(XTX.shape[0], device=device)
            beta = torch.linalg.solve(XTX_reg, XTY)

            # Predicted and R²
            predicted = X_torch @ beta
            residuals = Y_torch - predicted
            ss_res = torch.sum(residuals ** 2, dim=0)
            ss_tot = torch.sum((Y_torch - Y_torch.mean(dim=0, keepdim=True)) ** 2, dim=0)
            r2 = 1.0 - ss_res / (ss_tot + 1e-10)

            r2_per_hrf.append(r2.mean().item())

            # Store betas for task regressors only (first n_conds)
            betas_per_hrf.append(beta[:n_conds].cpu().numpy())

        # Find best HRF
        best_hrf_idx = np.argmax(r2_per_hrf)
        hrf_correct.append(best_hrf_idx == true_hrf_idx)
        r2_values.append(r2_per_hrf[best_hrf_idx])

        # Get beta estimates from best HRF
        best_betas = betas_per_hrf[best_hrf_idx]  # (n_conds, n_voxels)

        # Average across voxels
        beta_estimates_A.append(best_betas[0].mean())
        beta_estimates_B.append(best_betas[1].mean())
        beta_std_A.append(best_betas[0].std())
        beta_std_B.append(best_betas[1].std())

    # Compute statistics across realizations
    hrf_correct = np.array(hrf_correct)
    r2_values = np.array(r2_values)
    beta_estimates_A = np.array(beta_estimates_A)
    beta_estimates_B = np.array(beta_estimates_B)

    results = {
        # HRF recovery
        'hrf_recovery_rate': hrf_correct.mean(),

        # R² statistics
        'r2_median': np.median(r2_values),
        'r2_iqr': [np.percentile(r2_values, 25), np.percentile(r2_values, 75)],
        'r2_mean': r2_values.mean(),

        # Beta estimation error
        'beta_A_error_median': np.median(np.abs(beta_estimates_A - activation_pattern[0])),
        'beta_B_error_median': np.median(np.abs(beta_estimates_B - activation_pattern[1])),
        'beta_A_bias': np.median(beta_estimates_A) - activation_pattern[0],
        'beta_B_bias': np.median(beta_estimates_B) - activation_pattern[1],

        # Detection power (proportion where we recovered HRF correctly)
        'detection_power': hrf_correct.mean(),

        # Raw data for further analysis
        'r2_all': r2_values,
        'beta_A_all': beta_estimates_A,
        'beta_B_all': beta_estimates_B,
    }

    return results


def run_full_monte_carlo_study(
    isi_means: List[float],
    activation_patterns: np.ndarray,
    hrf_library_name: str = 'cnvlab',
    n_noise_realizations: int = 100,
    n_voxels_per_realization: int = 1000,
    noise_std: float = 2.0,
    tr: float = 1.0,
    total_duration: float = 290.0,
    stim_duration: float = 5.0,
    baseline: float = 100.0,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict:
    """
    Run full Monte Carlo study across ISI means and activation patterns.

    Parameters
    ----------
    isi_means : list
        List of ISI means to test
    activation_patterns : ndarray
        Array of activation patterns, shape (n_patterns, n_conds)
    hrf_library_name : str
        'cnvlab' or 'canonical'
    n_noise_realizations : int
        Number of Monte Carlo realizations per condition
    n_voxels_per_realization : int
        Number of voxels per realization
    noise_std : float
        Noise standard deviation
    tr : float
        Repetition time
    total_duration : float
        Scan duration
    stim_duration : float
        Stimulus duration
    baseline : float
        Baseline signal level
    device : torch.device, optional
        Device for computation
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Nested dictionary: results[isi_mean][pattern_idx] = stats_dict
    """
    if device is None:
        device = ffs.get_device()

    if verbose:
        print(f"{'='*70}")
        print("MONTE CARLO ISI STUDY")
        print(f"{'='*70}\n")
        print(f"Device: {device}")
        print(f"ISI means: {isi_means}")
        print(f"Activation patterns: {len(activation_patterns)}")
        print(f"Noise realizations per condition: {n_noise_realizations}")
        print(f"Voxels per realization: {n_voxels_per_realization}")
        print(f"Noise std: {noise_std}")
        print(f"\nTotal simulations: {len(isi_means)} × {len(activation_patterns)} × {n_noise_realizations}")
        print(f"                 = {len(isi_means) * len(activation_patterns) * n_noise_realizations:,}\n")

    # Load HRF library
    if hrf_library_name == 'cnvlab':
        hrfs = load_cnvlab_hrf_library(duration=stim_duration, tr=tr)
    else:
        raise ValueError(f"Unknown HRF library: {hrf_library_name}")

    n_hrfs = hrfs.shape[0]
    if verbose:
        print(f"Loaded {n_hrfs} HRFs from {hrf_library_name} library\n")

    # Calculate durations in TRs
    total_tr = int(total_duration / tr)
    stim_tr = int(stim_duration / tr)
    padding_tr = int(10 / tr)

    # Results storage
    results = {}

    start_time = time.time()

    # Loop over ISI means
    for isi_idx, isi_mean in enumerate(isi_means):
        if verbose:
            print(f"\n{'='*70}")
            print(f"ISI MEAN: {isi_mean}s ({isi_idx+1}/{len(isi_means)})")
            print(f"{'='*70}\n")

        # Generate Poisson ISIs
        block_average = isi_mean + stim_duration
        n_trials = int((total_duration - 2 * padding_tr * tr) / block_average)
        n_trials = n_trials - (n_trials % 2)  # Make even for 2 conditions

        isis_sec = generate_poisson_isis(
            target_mean=isi_mean,
            n_isis=n_trials,
            lower_limit=2.0,
            upper_limit=8.0,
            seed=42 + isi_idx
        )
        isis_tr = np.round(isis_sec / tr).astype(int)

        # Build design matrix
        design_unconvolved = build_alternating_design(
            isis_tr=isis_tr,
            stim_dur_tr=stim_tr,
            n_conds=2,
            total_tr=total_tr,
            padding_tr=padding_tr
        )

        if verbose:
            print(f"  Design: {np.sum(design_unconvolved[:, 0]):.0f} + {np.sum(design_unconvolved[:, 1]):.0f} trials\n")

        # Results for this ISI
        results[isi_mean] = {}

        # Loop over activation patterns
        for pattern_idx, pattern in enumerate(activation_patterns):
            # Use HRF index based on pattern (cycle through library)
            true_hrf_idx = pattern_idx % n_hrfs

            # Run Monte Carlo for this pattern
            pattern_results = run_monte_carlo_simulation(
                activation_pattern=pattern,
                true_hrf_idx=true_hrf_idx,
                hrfs=hrfs,
                design_unconvolved=design_unconvolved,
                tr=tr,
                n_voxels=n_voxels_per_realization,
                n_noise_realizations=n_noise_realizations,
                noise_std=noise_std,
                baseline=baseline,
                device=device,
                verbose=verbose
            )

            results[isi_mean][pattern_idx] = pattern_results

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n{'='*70}")
        print("MONTE CARLO STUDY COMPLETE!")
        print(f"{'='*70}\n")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Time per ISI mean: {elapsed/len(isi_means):.1f}s")
        print(f"Time per pattern: {elapsed/(len(isi_means)*len(activation_patterns)):.2f}s")

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_monte_carlo_results(
    results: Dict,
    activation_patterns: np.ndarray,
    isi_means: List[float],
    output_dir: Path
):
    """
    Create comprehensive plots from Monte Carlo results.

    4 rows × 5 cols = 20 activation patterns per metric.
    """
    n_patterns = len(activation_patterns)
    n_rows = 4
    n_cols = 5

    # =========================================================================
    # Figure 1: HRF Recovery Rate
    # =========================================================================
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('HRF Recovery Rate vs ISI Mean', fontsize=18, fontweight='bold', y=0.995)

    for pattern_idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, pattern_idx + 1)

        pattern = activation_patterns[pattern_idx]
        recovery_rates = [results[isi][pattern_idx]['hrf_recovery_rate']
                          for isi in isi_means]

        ax.plot(isi_means, recovery_rates, 'o-', linewidth=2, markersize=8)
        ax.set_ylim([0, 1.05])
        ax.set_title(f'Pattern {pattern_idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('ISI (s)', fontsize=9)
        ax.set_ylabel('Recovery Rate', fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_dir / 'monte_carlo_hrf_recovery.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'monte_carlo_hrf_recovery.png'}")
    plt.close()

    # =========================================================================
    # Figure 2: R² (median)
    # =========================================================================
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Median R² vs ISI Mean', fontsize=18, fontweight='bold', y=0.995)

    for pattern_idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, pattern_idx + 1)

        pattern = activation_patterns[pattern_idx]
        r2_medians = [results[isi][pattern_idx]['r2_median']
                      for isi in isi_means]
        r2_iqrs = [results[isi][pattern_idx]['r2_iqr']
                   for isi in isi_means]

        # Plot median with IQR
        r2_lower = [iqr[0] for iqr in r2_iqrs]
        r2_upper = [iqr[1] for iqr in r2_iqrs]

        ax.plot(isi_means, r2_medians, 'o-', linewidth=2, markersize=8, color='blue')
        ax.fill_between(isi_means, r2_lower, r2_upper, alpha=0.3, color='blue')
        ax.set_ylim([0, 1.05])
        ax.set_title(f'Pattern {pattern_idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('ISI (s)', fontsize=9)
        ax.set_ylabel('R² (median)', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'monte_carlo_r2.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'monte_carlo_r2.png'}")
    plt.close()

    # =========================================================================
    # Figure 3: Beta Estimation Error (Condition A)
    # =========================================================================
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Beta Estimation Error (Condition A) vs ISI Mean', fontsize=18, fontweight='bold', y=0.995)

    for pattern_idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, pattern_idx + 1)

        pattern = activation_patterns[pattern_idx]
        errors_A = [results[isi][pattern_idx]['beta_A_error_median']
                    for isi in isi_means]

        ax.plot(isi_means, errors_A, 'o-', linewidth=2, markersize=8, color='green')
        ax.set_title(f'Pattern {pattern_idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('ISI (s)', fontsize=9)
        ax.set_ylabel('|β̂ - β_true|', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'monte_carlo_beta_error_A.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'monte_carlo_beta_error_A.png'}")
    plt.close()

    # =========================================================================
    # Figure 4: Beta Estimation Error (Condition B)
    # =========================================================================
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Beta Estimation Error (Condition B) vs ISI Mean', fontsize=18, fontweight='bold', y=0.995)

    for pattern_idx in range(n_patterns):
        ax = plt.subplot(n_rows, n_cols, pattern_idx + 1)

        pattern = activation_patterns[pattern_idx]
        errors_B = [results[isi][pattern_idx]['beta_B_error_median']
                    for isi in isi_means]

        ax.plot(isi_means, errors_B, 'o-', linewidth=2, markersize=8, color='orange')
        ax.set_title(f'Pattern {pattern_idx}: [{pattern[0]:.0f}, {pattern[1]:.0f}]',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('ISI (s)', fontsize=9)
        ax.set_ylabel('|β̂ - β_true|', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'monte_carlo_beta_error_B.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'monte_carlo_beta_error_B.png'}")
    plt.close()

    # =========================================================================
    # Figure 5: Heatmaps (Summary)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Monte Carlo Summary Heatmaps (rows=patterns, cols=ISI)',
                 fontsize=16, fontweight='bold')

    # Prepare data matrices
    hrf_recovery_matrix = np.zeros((n_patterns, len(isi_means)))
    r2_matrix = np.zeros((n_patterns, len(isi_means)))
    beta_A_error_matrix = np.zeros((n_patterns, len(isi_means)))
    beta_B_error_matrix = np.zeros((n_patterns, len(isi_means)))

    for pattern_idx in range(n_patterns):
        for isi_idx, isi in enumerate(isi_means):
            hrf_recovery_matrix[pattern_idx, isi_idx] = results[isi][pattern_idx]['hrf_recovery_rate']
            r2_matrix[pattern_idx, isi_idx] = results[isi][pattern_idx]['r2_median']
            beta_A_error_matrix[pattern_idx, isi_idx] = results[isi][pattern_idx]['beta_A_error_median']
            beta_B_error_matrix[pattern_idx, isi_idx] = results[isi][pattern_idx]['beta_B_error_median']

    # Plot heatmaps
    im1 = axes[0, 0].imshow(hrf_recovery_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[0, 0].set_title('HRF Recovery Rate', fontweight='bold')
    axes[0, 0].set_ylabel('Pattern Index')
    axes[0, 0].set_xticks(range(len(isi_means)))
    axes[0, 0].set_xticklabels([f'{isi:.1f}' for isi in isi_means])
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(r2_matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    axes[0, 1].set_title('Median R²', fontweight='bold')
    axes[0, 1].set_ylabel('Pattern Index')
    axes[0, 1].set_xticks(range(len(isi_means)))
    axes[0, 1].set_xticklabels([f'{isi:.1f}' for isi in isi_means])
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(beta_A_error_matrix, cmap='plasma', aspect='auto')
    axes[0, 2].set_title('Beta Error (Cond A)', fontweight='bold')
    axes[0, 2].set_ylabel('Pattern Index')
    axes[0, 2].set_xticks(range(len(isi_means)))
    axes[0, 2].set_xticklabels([f'{isi:.1f}' for isi in isi_means])
    plt.colorbar(im3, ax=axes[0, 2])

    im4 = axes[1, 0].imshow(beta_B_error_matrix, cmap='plasma', aspect='auto')
    axes[1, 0].set_title('Beta Error (Cond B)', fontweight='bold')
    axes[1, 0].set_xlabel('ISI Mean (s)')
    axes[1, 0].set_ylabel('Pattern Index')
    axes[1, 0].set_xticks(range(len(isi_means)))
    axes[1, 0].set_xticklabels([f'{isi:.1f}' for isi in isi_means])
    plt.colorbar(im4, ax=axes[1, 0])

    # Pattern magnitude plot
    pattern_mags = np.linalg.norm(activation_patterns, axis=1)
    axes[1, 1].barh(range(n_patterns), pattern_mags, color='steelblue')
    axes[1, 1].set_xlabel('||[A, B]||')
    axes[1, 1].set_ylabel('Pattern Index')
    axes[1, 1].set_title('Pattern Magnitudes', fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='x')

    # Pattern scatter
    axes[1, 2].scatter(activation_patterns[:, 0], activation_patterns[:, 1],
                       s=100, alpha=0.6, c=range(n_patterns), cmap='tab20')
    axes[1, 2].set_xlabel('Condition A')
    axes[1, 2].set_ylabel('Condition B')
    axes[1, 2].set_title('Activation Patterns', fontweight='bold')
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].axhline(0, color='k', linewidth=0.5)
    axes[1, 2].axvline(0, color='k', linewidth=0.5)

    # Add pattern labels
    for idx, (a, b) in enumerate(activation_patterns):
        axes[1, 2].annotate(str(idx), (a, b), fontsize=8, ha='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'monte_carlo_heatmaps.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'monte_carlo_heatmaps.png'}")
    plt.close()


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Setup
    device = ffs.get_device()

    # Output directory
    output_dir = Path("monte_carlo_results")
    output_dir.mkdir(exist_ok=True)

    # Simulation parameters
    isi_means = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

    # Activation patterns (from simulate_movietasks.m)
    activation_patterns = np.array([
        [5, 1], [5, 2], [5, 3], [5, 4], [5, 5],
        [4, 5], [3, 5], [2, 5], [1, 5],
        [1, 3], [3, 3], [3, 1], [1, 1],
        [0, 2], [2, 0],
        [-1, -1], [-3, -3],
        [-3, 4], [4, -3],
        [0, 0]
    ], dtype=np.float32)

    # Run Monte Carlo study
    results = run_full_monte_carlo_study(
        isi_means=isi_means,
        activation_patterns=activation_patterns,
        hrf_library_name='cnvlab',
        n_noise_realizations=100,
        n_voxels_per_realization=1000,
        noise_std=2.0,
        tr=1.0,
        total_duration=290.0,
        stim_duration=5.0,
        baseline=100.0,
        device=device,
        verbose=True
    )

    # Save results
    results_file = output_dir / 'monte_carlo_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results: {results_file}")

    # Create plots
    print(f"\nCreating plots...")
    plot_monte_carlo_results(results, activation_patterns, isi_means, output_dir)

    print(f"\n{'='*70}")
    print("ALL DONE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - monte_carlo_results.pkl (raw data)")
    print(f"  - monte_carlo_hrf_recovery.png")
    print(f"  - monte_carlo_r2.png")
    print(f"  - monte_carlo_beta_error_A.png")
    print(f"  - monte_carlo_beta_error_B.png")
    print(f"  - monte_carlo_heatmaps.png")
