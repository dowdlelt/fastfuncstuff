#!/usr/bin/env python3
"""
Example: Cross-Validated Denoising with Noise Pool PCA

This example demonstrates the adaptive denoising approach:
1. Identify noise pool voxels (low task R²) and criteria voxels (high task R²)
2. Extract PCs from noise pool as candidate nuisance regressors
3. Cross-validate to select optimal number of PCs
4. Train on denoised data but test on raw data to prevent overfitting

The key innovation: we denoise training data but predict non-denoised test data,
ensuring we're improving signal recovery rather than just fitting the denoising.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from fastfuncsim import (
    # GLM and design
    fit_glm_torch,
    build_glm_design,
    
    # Denoising
    fit_denoising_model,
    DenoiseResults,
    
    # Simulation and noise
    simulate_fmri_data,
    generate_ar1_noise,
    
    # HRF
    get_hrf_library,
    
    # Utils
    get_device,
    to_tensor,
)


def example_basic_denoising():
    """
    Basic denoising example with simulated data
    """
    print("="*70)
    print("Example 1: Basic Denoising with Simulated Data")
    print("="*70)
    
    device = get_device()
    
    # Simulation parameters
    n_voxels = 1000
    n_runs = 3
    n_tps_per_run = 200
    tr = 2.0
    
    # Create simple block design (same across runs)
    onsets = [10, 50, 90, 130, 170]  # seconds
    onsets = [np.array(onsets) for _ in range(n_runs)]  # replicate per run
    
    run_starts = [i * n_tps_per_run for i in range(n_runs)]
    n_timepoints = n_tps_per_run * n_runs
    
    # Build design matrix
    from fastfuncsim.design_builder import create_design_matrix_from_onsets
    
    design_matrix = create_design_matrix_from_onsets(
        onsets=[onsets],  # one condition
        stim_durations=[2.0],
        tr=tr,
        n_timepoints=n_timepoints,
        run_starts=run_starts,
        polort=2,
        microtime_dt=0.1,
        device=device,
    )
    
    # Separate task and nuisance
    task_design = design_matrix['stimulus']
    nuisance = design_matrix['nuisance']
    
    print(f"\nSimulation setup:")
    print(f"  Voxels: {n_voxels}")
    print(f"  Runs: {n_runs} × {n_tps_per_run} TRs")
    print(f"  TR: {tr}s")
    print(f"  Task predictors: {task_design.shape[1]}")
    print(f"  Nuisance predictors: {nuisance.shape[1]}")
    
    # Simulate data
    # - 20% of voxels have task signal (criteria voxels)
    # - 80% have no task signal (noise pool)
    n_task_voxels = 200
    n_noise_voxels = n_voxels - n_task_voxels
    
    # Task voxels: signal + noise
    task_betas = torch.randn(n_task_voxels, task_design.shape[1], device=device) * 2.0
    task_signal = task_design @ task_betas.T  # (n_timepoints, n_task_voxels)
    task_signal = task_signal.T  # (n_task_voxels, n_timepoints)
    
    # Noise voxels: pure noise (no task signal)
    noise_signal = torch.zeros(n_noise_voxels, n_timepoints, device=device)
    
    # Add AR(1) noise to all voxels
    ar1_noise_task = generate_ar1_noise(
        n_voxels=n_task_voxels,
        n_timepoints=n_timepoints,
        rho=0.3,
        sigma=1.0,
        device=device,
    )
    
    ar1_noise_pool = generate_ar1_noise(
        n_voxels=n_noise_voxels,
        n_timepoints=n_timepoints,
        rho=0.3,
        sigma=1.0,
        device=device,
    )
    
    # Combine
    data_task = task_signal + ar1_noise_task
    data_noise = noise_signal + ar1_noise_pool
    
    # Concatenate (noise voxels first, task voxels last)
    data = torch.cat([data_noise, data_task], dim=0)
    
    print(f"\nData simulation:")
    print(f"  Task voxels: {n_task_voxels} (with signal)")
    print(f"  Noise voxels: {n_noise_voxels} (no signal)")
    
    # Fit denoising model
    print(f"\n{'='*70}")
    print("Fitting cross-validated denoising model...")
    print(f"{'='*70}")
    
    results = fit_denoising_model(
        data=data,
        design_matrix=task_design,
        run_starts=run_starts,
        r2_threshold=0.1,
        max_components=15,
        nuisance=nuisance,
        device=device,
        verbose=True,
    )
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. CV R² by number of PCs
    ax = axes[0, 0]
    ax.plot(results.xval_r2_by_n_components, 'o-', label='Mean across folds')
    ax.plot(results.xval_r2_median_by_n_components, 's--', alpha=0.7, label='Median across folds')
    ax.axvline(results.optimal_n_components, color='r', linestyle='--', alpha=0.5, label=f'Optimal ({results.optimal_n_components} PCs)')
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('Cross-validated R²')
    ax.set_title('Denoising Performance vs Number of PCs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. R² per fold (heatmap)
    ax = axes[0, 1]
    im = ax.imshow(results.xval_r2_per_fold, aspect='auto', cmap='viridis')
    ax.set_xlabel('Number of PCs')
    ax.set_ylabel('CV Fold (run)')
    ax.set_title('R² per CV Fold')
    plt.colorbar(im, ax=ax, label='R²')
    
    # 3. Initial R² distribution (noise pool selection)
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
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    Denoising Results Summary
    {'='*40}
    
    Voxel Selection:
      • Noise pool: {results.metadata['n_noise_voxels']:,} voxels
      • Criteria: {results.metadata['n_criteria_voxels']:,} voxels
      • Threshold: R² < {results.metadata['r2_threshold']:.2f}
    
    Cross-Validation:
      • Strategy: Leave-one-run-out
      • Runs: {results.metadata['n_runs']}
      • Max PCs tested: {results.metadata['max_components']}
    
    Performance:
      • Baseline R²: {results.baseline_r2:.4f}
      • Optimal R²: {results.optimal_r2:.4f}
      • Improvement: {results.improvement:+.4f}
      • Optimal PCs: {results.optimal_n_components}
    
    Key Insight:
      Training on denoised data while testing
      on raw data prevents overfitting and
      ensures true signal improvement.
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('denoising_example_basic.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: denoising_example_basic.png")
    plt.show()
    
    return results


def example_with_hrf_optimization():
    """
    Combined HRF optimization + denoising
    
    This shows how to use R² from HRF optimization for noise pool selection,
    then apply denoising with the optimized HRFs.
    """
    print("\n" + "="*70)
    print("Example 2: HRF Optimization + Denoising")
    print("="*70)
    
    # TODO: Implement once we integrate with hrf_selection module
    print("\n⚠️  This example requires integration with HRF selection.")
    print("    Coming in next iteration!")


if __name__ == "__main__":
    # Run examples
    print("\n" + "="*70)
    print("Cross-Validated Denoising Examples")
    print("="*70 + "\n")
    
    # Example 1: Basic denoising
    results = example_basic_denoising()
    
    # Example 2: With HRF optimization (TODO)
    # example_with_hrf_optimization()
    
    print("\n" + "="*70)
    print("✅ Examples complete!")
    print("="*70 + "\n")
