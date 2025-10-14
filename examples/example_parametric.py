"""
Example: Parametric Exploration Across Three Axes

Demonstrates the flexible 3-axis exploration framework:
1. Event magnitude combinations (A vs B effect sizes)
2. Variable HRFs (different HRF shapes)
3. Noise levels (different SNR)

This example shows:
- How well the model performs across parameter space
- Single-case deep dive for individual simulations
- Batch summary for statistical power
- Flexible across any number of conditions
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from time import time
import fastfuncsim as ffs

def run_parametric_simulation(
    beta_cond1: float,
    beta_cond2: float,
    hrf_index: int,
    noise_level: float,
    hrf_library: torch.Tensor,
    config: dict,
    device: torch.device
) -> dict:
    """
    Run a single simulation with specific parameters

    Returns dict with:
    - r2_mean, r2_median, r2_std
    - beta_error_mae, beta_error_rmse
    - betas_estimated
    - data, design, results (for deep dive)
    """
    # Generate onsets
    onsets = ffs.generate_random_onsets(
        n_timepoints=config['n_timepoints'],
        n_conditions=config['n_conditions'],
        isi_mean=config['isi_mean'],
        tr=config['tr'],
        alternate_conditions=True,
        device=device
    )

    # Select HRF
    hrf = hrf_library[hrf_index, :]

    # Simulate data
    betas_true = [beta_cond1, beta_cond2]
    data = ffs.simulate_fmri_run(
        onsets=onsets,
        betas=betas_true,
        hrf=hrf,
        tr=config['tr'],
        n_timepoints=config['n_timepoints'],
        matrix_size=config['matrix_size'],
        noise_level=noise_level,
        baseline=100.0,
        device=device
    )

    # Fit GLM with assumed HRF
    design = ffs.build_glm_design(onsets, hrf, config['n_timepoints'],
                                   mode='assumed', device=device)
    results = ffs.fit_glm(
        data=data,
        design=design,
        tr=config['tr'],
        want_residuals=False,
        want_predicted=False,
        want_r2_run=False,
        device=device,
        verbose=False
    )

    # Compute metrics
    betas_est = results.betas.cpu().numpy()  # (n_voxels, n_conditions)
    betas_true_np = np.array(betas_true)

    # Beta error
    beta_error = np.abs(betas_est[:, :len(betas_true)] - betas_true_np[np.newaxis, :])
    beta_error_mae = np.mean(beta_error)
    beta_error_rmse = np.sqrt(np.mean(beta_error**2))

    return {
        'r2_mean': results.r2.mean().item(),
        'r2_median': results.r2.median().item(),
        'r2_std': results.r2.std().item(),
        'r2_max': results.r2.max().item(),
        'beta_error_mae': beta_error_mae,
        'beta_error_rmse': beta_error_rmse,
        'betas_true': betas_true,
        'beta_cond1': beta_cond1,
        'beta_cond2': beta_cond2,
        'hrf_index': hrf_index,
        'noise_level': noise_level,
        # Store for deep dive
        'data': data,
        'design': design,
        'results': results,
        'onsets': onsets,
        'hrf': hrf,
    }


def main():
    print("=" * 70)
    print("FastFuncSim - Parametric Exploration Example")
    print("3-Axis: (Magnitudes × HRFs × Noise)")
    print("=" * 70)

    # Setup
    device = ffs.get_device()
    ffs.print_device_info(device)

    # Base configuration
    config = {
        'tr': 1.0,
        'n_timepoints': 290,
        'n_conditions': 2,
        'isi_mean': 4.0,
        'matrix_size': (20, 20, 3),  # Small for speed: 1,200 voxels
    }

    print(f"\nExperiment Configuration:")
    print(f"  TR: {config['tr']}s")
    print(f"  Duration: {config['n_timepoints']} TRs")
    print(f"  Conditions: {config['n_conditions']}")
    print(f"  Matrix size: {config['matrix_size']}")

    # Define parameter space
    # Axis 1: Event magnitude combinations (A vs B)
    beta_ratios = [
        (1.0, 1.0),  # Equal
        (2.0, 1.0),  # A twice B
        (3.0, 1.0),  # A three times B
        (1.0, 2.0),  # B twice A
    ]

    # Axis 2: HRF variations (create small library)
    n_hrfs = 5
    print(f"\nCreating HRF library with {n_hrfs} variants...")
    hrf_library = ffs.get_canonical_hrf_library(
        stim_duration=5.0,
        tr=config['tr'],
        n_hrfs=n_hrfs,
        device=device
    )
    print(f"  HRF library shape: {hrf_library.shape}")

    # Axis 3: Noise levels
    noise_levels = [0.5, 1.0, 2.0]

    print(f"\nParameter Space:")
    print(f"  Beta ratios: {len(beta_ratios)} combinations")
    print(f"  HRFs: {n_hrfs} variants")
    print(f"  Noise levels: {len(noise_levels)}")
    print(f"  Total combinations: {len(beta_ratios) * n_hrfs * len(noise_levels)}")

    # Storage for results
    results_grid = {}  # {noise_level: {hrf_idx: {beta_ratio: result}}}
    results_list = []  # Flat list for batch summary

    # Run parametric exploration
    print("\nRunning parametric simulations...")
    start_time = time()

    total_sims = 0
    for noise_level in noise_levels:
        results_grid[noise_level] = {}
        print(f"\n  Noise level {noise_level}:")

        for hrf_idx in range(n_hrfs):
            results_grid[noise_level][hrf_idx] = {}

            for beta_idx, (beta1, beta2) in enumerate(beta_ratios):
                if (total_sims + 1) % 10 == 0:
                    print(f"    Simulation {total_sims+1}...", end='\r')

                result = run_parametric_simulation(
                    beta_cond1=beta1,
                    beta_cond2=beta2,
                    hrf_index=hrf_idx,
                    noise_level=noise_level,
                    hrf_library=hrf_library,
                    config=config,
                    device=device
                )

                # Store in grid for parametric visualization
                results_grid[noise_level][hrf_idx][beta_idx] = result

                # Store in flat list for batch summary
                results_list.append(result)

                total_sims += 1

        print(f"    Simulation {total_sims}... Done!")

    elapsed = time() - start_time
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Time per simulation: {elapsed/total_sims:.3f}s")
    print(f"  Simulations per minute: {60*total_sims/elapsed:.1f}")

    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================

    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)

    # 1. Single-case deep dive (best R² from middle noise level)
    print("\n1. Single-case deep dive (best performing simulation)...")

    # Find best simulation
    best_result = max(results_list, key=lambda r: r['r2_mean'])
    print(f"   Best: β=({best_result['beta_cond1']}, {best_result['beta_cond2']}), "
          f"HRF={best_result['hrf_index']}, Noise={best_result['noise_level']}")
    print(f"   R²={best_result['r2_mean']:.3f}")

    fig_deep = ffs.plot_simulation_deep_dive(
        data=best_result['data'],
        design=best_result['design'],
        results=best_result['results'],
        onsets=best_result['onsets'],
        betas_true=torch.tensor(best_result['betas_true']),
        hrf_true=best_result['hrf'],
        voxel_selection='best',
        n_voxels=4,
        tr=config['tr'],
        save_path='parametric_deep_dive.png'
    )
    plt.close(fig_deep)

    # 2. Parametric exploration heatmaps
    print("\n2. Parametric exploration (3-axis heatmaps)...")

    # Convert to format expected by plot_parametric_exploration
    # Grid structure: {z_val: {y_val: {x_val: metrics}}}
    metrics_grid = {}
    for noise_level in noise_levels:
        metrics_grid[noise_level] = {}
        for hrf_idx in range(n_hrfs):
            metrics_grid[noise_level][hrf_idx] = {}
            for beta_idx in range(len(beta_ratios)):
                result = results_grid[noise_level][hrf_idx][beta_idx]
                # Store just the metrics (not the full data)
                metrics_grid[noise_level][hrf_idx][beta_idx] = {
                    'r2_mean': result['r2_mean'],
                    'beta_error_mae': result['beta_error_mae'],
                }

    fig_param = ffs.plot_parametric_exploration(
        results_grid=metrics_grid,
        x_var='Beta Ratio Index',
        y_var='HRF Index',
        z_var='Noise Level',
        metric='r2_mean',
        save_path='parametric_exploration_r2.png'
    )
    plt.close(fig_param)

    # Beta error parametric plot
    fig_param_beta = ffs.plot_parametric_exploration(
        results_grid=metrics_grid,
        x_var='Beta Ratio Index',
        y_var='HRF Index',
        z_var='Noise Level',
        metric='beta_error_mae',
        save_path='parametric_exploration_beta_error.png'
    )
    plt.close(fig_param_beta)

    # 3. Batch summary
    print("\n3. Batch summary statistics...")

    fig_batch = ffs.plot_batch_summary(
        results_list=results_list,
        metrics=['r2', 'beta_error'],
        group_by='noise_level',
        save_path='parametric_batch_summary.png'
    )
    plt.close(fig_batch)

    # 4. HRF-specific analysis (FIR estimation for one case)
    print("\n4. HRF recovery analysis (FIR estimation)...")

    # Take a representative case
    repr_result = results_grid[noise_levels[1]][n_hrfs//2][0]  # Middle noise, middle HRF, first beta ratio

    # Run FIR estimation
    design_fir = ffs.build_glm_design(
        repr_result['onsets'],
        mode='fir',
        n_fir_lags=30,
        n_timepoints=config['n_timepoints'],
        device=device
    )

    results_fir = ffs.fit_glm(
        data=repr_result['data'],
        design=design_fir,
        tr=config['tr'],
        device=device,
        verbose=False
    )

    # Extract FIR estimates for condition 1
    n_fir_lags = 30
    n_voxels = results_fir.betas.shape[0]
    n_conditions = config['n_conditions']

    # Reshape: (n_voxels, n_conditions * n_lags) -> (n_voxels * n_conditions, n_lags)
    # For simplicity, just take condition 1
    fir_estimates = results_fir.betas[:, :n_fir_lags]  # Condition 1

    # Compare to true HRF
    true_hrf = repr_result['hrf'].cpu().numpy()
    # Pad if needed
    if len(true_hrf) < n_fir_lags:
        true_hrf = np.pad(true_hrf, (0, n_fir_lags - len(true_hrf)))
    else:
        true_hrf = true_hrf[:n_fir_lags]

    fig_hrf = ffs.plot_hrf_recovery(
        hrf_estimated=fir_estimates,
        hrf_true=true_hrf,
        tr=config['tr'],
        voxel_selection='best',
        n_voxels=6,
        save_path='parametric_hrf_recovery.png'
    )
    plt.close(fig_hrf)

    # 5. Design comparison
    print("\n5. Design matrix comparison...")

    # Show designs for different beta ratios (same HRF, same noise)
    sample_hrf = hrf_library[n_hrfs//2, :]
    designs_to_compare = {}

    for beta_idx, (beta1, beta2) in enumerate(beta_ratios[:3]):  # Show first 3
        # Generate onsets
        onsets = ffs.generate_random_onsets(
            n_timepoints=config['n_timepoints'],
            n_conditions=config['n_conditions'],
            isi_mean=config['isi_mean'],
            tr=config['tr'],
            alternate_conditions=True,
            device=device
        )

        design = ffs.build_glm_design(onsets, sample_hrf, config['n_timepoints'],
                                       mode='assumed', device=device)
        designs_to_compare[f'β=({beta1:.1f}, {beta2:.1f})'] = design

    fig_design = ffs.plot_design_comparison(
        designs=designs_to_compare,
        labels=['Condition 1', 'Condition 2'],
        tr=config['tr'],
        save_path='parametric_design_comparison.png'
    )
    plt.close(fig_design)

    # 6. Interactive HTML summary
    print("\n6. Interactive HTML summary...")
    html_path = ffs.create_interactive_summary_html(
        results_list=results_list,
        output_path='parametric_summary.html'
    )

    # ========================================================================
    # PRINT SUMMARY STATISTICS
    # ========================================================================

    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    # Overall
    r2_means = [r['r2_mean'] for r in results_list]
    beta_errors = [r['beta_error_mae'] for r in results_list]

    print(f"\nOverall (n={len(results_list)} simulations):")
    print(f"  R² - Mean: {np.mean(r2_means):.3f}, Std: {np.std(r2_means):.3f}")
    print(f"  Beta Error (MAE) - Mean: {np.mean(beta_errors):.3f}, Std: {np.std(beta_errors):.3f}")

    # By noise level
    print(f"\nBy Noise Level:")
    for noise_level in noise_levels:
        noise_results = [r for r in results_list if r['noise_level'] == noise_level]
        r2_mean = np.mean([r['r2_mean'] for r in noise_results])
        beta_error_mean = np.mean([r['beta_error_mae'] for r in noise_results])
        print(f"  Noise={noise_level}: R²={r2_mean:.3f}, Beta Error={beta_error_mean:.3f}")

    # By HRF
    print(f"\nBy HRF Index:")
    for hrf_idx in range(min(5, n_hrfs)):  # Show first 5
        hrf_results = [r for r in results_list if r['hrf_index'] == hrf_idx]
        r2_mean = np.mean([r['r2_mean'] for r in hrf_results])
        beta_error_mean = np.mean([r['beta_error_mae'] for r in hrf_results])
        print(f"  HRF={hrf_idx}: R²={r2_mean:.3f}, Beta Error={beta_error_mean:.3f}")

    # By beta ratio
    print(f"\nBy Beta Ratio:")
    for beta_idx, (beta1, beta2) in enumerate(beta_ratios):
        beta_results = [r for r in results_list if r['beta_cond1'] == beta1 and r['beta_cond2'] == beta2]
        r2_mean = np.mean([r['r2_mean'] for r in beta_results])
        beta_error_mean = np.mean([r['beta_error_mae'] for r in beta_results])
        print(f"  β=({beta1}, {beta2}): R²={r2_mean:.3f}, Beta Error={beta_error_mean:.3f}")

    print("\n" + "=" * 70)
    print("Files Created:")
    print("=" * 70)
    print("  parametric_deep_dive.png           - Single simulation deep dive")
    print("  parametric_exploration_r2.png      - R² across parameter space")
    print("  parametric_exploration_beta_error.png - Beta error across parameter space")
    print("  parametric_batch_summary.png       - Batch statistics")
    print("  parametric_hrf_recovery.png        - HRF recovery analysis")
    print("  parametric_design_comparison.png   - Design matrix comparison")
    print("  parametric_summary.html            - Interactive summary")

    print("\n" + "=" * 70)
    print("Parametric exploration complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
