"""
Example: Batch Simulation for Statistical Power Analysis
Demonstrates fast batch processing for thousands of simulations
"""

from time import time

import matplotlib.pyplot as plt
import numpy as np

import fastfuncstuff as ffs


def run_single_simulation(config, device):
    """Run a single simulation and GLM fit"""
    # Generate onsets
    onsets = ffs.generate_random_onsets(
        n_timepoints=config['n_timepoints'],
        n_conditions=config['n_conditions'],
        isi_mean=config['isi_mean'],
        tr=config['tr'],
        alternate_conditions=True,
        device=device
    )

    # Simulate data
    data = ffs.simulate_fmri_run(
        onsets=onsets,
        betas=config['betas'],
        hrf=config['hrf'],
        tr=config['tr'],
        n_timepoints=config['n_timepoints'],
        matrix_size=config['matrix_size'],
        noise_level=config['noise_level'],
        baseline=100.0,
        device=device
    )

    # Fit GLM
    design = ffs.build_glm_design(onsets, config['hrf'], config['n_timepoints'],
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

    return results


def main():
    print("=" * 60)
    print("FastFuncSim - Batch Simulation Example")
    print("Statistical Power Analysis")
    print("=" * 60)

    # Setup
    device = ffs.get_device()
    ffs.print_device_info(device)

    # Batch configuration
    n_simulations = 100  # Increase to 1000+ for real power analysis
    print(f"\nRunning {n_simulations} simulations...")

    # Base configuration
    base_config = {
        'tr': 1.0,
        'n_timepoints': 290,
        'n_conditions': 2,
        'isi_mean': 4.0,
        'matrix_size': (20, 20, 5),  # Small for speed (2,000 voxels)
        'noise_level': 1.0,
    }

    # Test different effect sizes
    effect_sizes = [0.5, 1.0, 2.0, 3.0, 5.0]
    print(f"Testing effect sizes: {effect_sizes}")

    # Pre-generate HRF (shared across simulations)
    hrf = ffs.get_canonical_hrf(stim_duration=5.0, tr=base_config['tr'], device=device)
    base_config['hrf'] = hrf

    # Storage for results
    results_by_effect = {es: [] for es in effect_sizes}

    # Run batch simulations
    print("\nRunning simulations...")
    start_time = time()

    for es_idx, effect_size in enumerate(effect_sizes):
        print(f"\n  Effect size {effect_size} ({es_idx+1}/{len(effect_sizes)})")

        # Update config for this effect size
        config = base_config.copy()
        config['betas'] = [effect_size, effect_size]

        # Run simulations
        for sim_idx in range(n_simulations):
            if (sim_idx + 1) % 20 == 0:
                print(f"    Simulation {sim_idx+1}/{n_simulations}...", end='\r')

            results = run_single_simulation(config, device)

            # Store summary stats
            results_by_effect[effect_size].append({
                'r2_mean': results.r2.mean().item(),
                'mean_r2': results.r2.mean().item(),  # Legacy name
                'r2_median': results.r2.median().item(),
                'median_r2': results.r2.median().item(),  # Legacy name
                'max_r2': results.r2.max().item(),
                'mean_beta_cond1': results.betas[:, 0].mean().item(),
                'std_beta_cond1': results.betas[:, 0].std().item(),
                'effect_size': effect_size,  # For grouping
            })

        print(f"    Simulation {n_simulations}/{n_simulations}... Done!")

    elapsed = time() - start_time
    total_sims = n_simulations * len(effect_sizes)
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Simulations: {total_sims}")
    print(f"  Time per simulation: {elapsed/total_sims:.3f}s")
    print(f"  Simulations per minute: {60*total_sims/elapsed:.1f}")

    # Analyze results
    print("\n" + "=" * 60)
    print("Statistical Power Analysis Results")
    print("=" * 60)

    # Use new visualization module
    print("\nCreating batch summary visualization...")
    fig_batch = ffs.plot_batch_summary(
        results_list=sum(results_by_effect.values(), []),  # Flatten all results
        metrics=['r2', 'beta_error', 'power'],
        group_by='effect_size',
        save_path='batch_summary_new.png'
    )
    plt.close(fig_batch)

    # Legacy detailed plots
    plt.figure(figsize=(15, 10))

    # Plot 1: Mean R² vs Effect Size
    plt.subplot(2, 3, 1)
    mean_r2_by_es = []
    std_r2_by_es = []
    for es in effect_sizes:
        r2_values = [r['mean_r2'] for r in results_by_effect[es]]
        mean_r2_by_es.append(np.mean(r2_values))
        std_r2_by_es.append(np.std(r2_values))

    plt.errorbar(effect_sizes, mean_r2_by_es, yerr=std_r2_by_es,
                marker='o', capsize=5, linewidth=2)
    plt.xlabel('Effect Size')
    plt.ylabel('Mean R²')
    plt.title('Detection Power vs Effect Size')
    plt.grid(True)

    # Plot 2: R² distributions
    plt.subplot(2, 3, 2)
    for es in effect_sizes:
        r2_values = [r['mean_r2'] for r in results_by_effect[es]]
        plt.hist(r2_values, bins=20, alpha=0.5, label=f'ES={es}')
    plt.xlabel('Mean R²')
    plt.ylabel('Count')
    plt.title('R² Distribution by Effect Size')
    plt.legend()
    plt.grid(True)

    # Plot 3: Beta estimation accuracy
    plt.subplot(2, 3, 3)
    mean_beta_by_es = []
    std_beta_by_es = []
    for es in effect_sizes:
        beta_values = [r['mean_beta_cond1'] for r in results_by_effect[es]]
        mean_beta_by_es.append(np.mean(beta_values))
        std_beta_by_es.append(np.std(beta_values))

    plt.errorbar(effect_sizes, mean_beta_by_es, yerr=std_beta_by_es,
                marker='o', capsize=5, linewidth=2, label='Estimated')
    plt.plot(effect_sizes, effect_sizes, 'k--', linewidth=2, label='True')
    plt.xlabel('True Effect Size')
    plt.ylabel('Estimated Beta')
    plt.title('Beta Estimation Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot 4: Statistical power (proportion R² > threshold)
    plt.subplot(2, 3, 4)
    r2_thresholds = [0.01, 0.05, 0.10, 0.20]
    for threshold in r2_thresholds:
        power = []
        for es in effect_sizes:
            r2_values = [r['mean_r2'] for r in results_by_effect[es]]
            prop_significant = np.mean([r > threshold for r in r2_values])
            power.append(prop_significant)
        plt.plot(effect_sizes, power, marker='o', label=f'R²>{threshold}')

    plt.axhline(0.8, color='k', linestyle='--', label='80% power')
    plt.xlabel('Effect Size')
    plt.ylabel('Statistical Power')
    plt.title('Power Curves')
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 1])

    # Plot 5: Beta variance vs effect size
    plt.subplot(2, 3, 5)
    for es in effect_sizes:
        beta_stds = [r['std_beta_cond1'] for r in results_by_effect[es]]
        plt.scatter([es] * len(beta_stds), beta_stds, alpha=0.3)

    plt.xlabel('Effect Size')
    plt.ylabel('Beta Std Dev')
    plt.title('Beta Estimation Variance')
    plt.grid(True)

    # Plot 6: Summary statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')

    # Calculate summary stats
    summary_text = "Summary Statistics:\n\n"
    for es in effect_sizes:
        r2_values = [r['mean_r2'] for r in results_by_effect[es]]
        mean_r2 = np.mean(r2_values)
        power_5pct = np.mean([r > 0.05 for r in r2_values])
        summary_text += f"Effect Size {es}:\n"
        summary_text += f"  Mean R²: {mean_r2:.3f}\n"
        summary_text += f"  Power (R²>0.05): {power_5pct:.1%}\n\n"

    plt.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    plt.savefig('batch_power_analysis.png', dpi=150)
    print("\nSaved: batch_power_analysis.png")

    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 60)
    for es in effect_sizes:
        r2_values = [r['mean_r2'] for r in results_by_effect[es]]
        beta_values = [r['mean_beta_cond1'] for r in results_by_effect[es]]
        power_1pct = np.mean([r > 0.01 for r in r2_values])
        power_5pct = np.mean([r > 0.05 for r in r2_values])

        print(f"\nEffect Size: {es}")
        print(f"  R² - Mean: {np.mean(r2_values):.3f}, SD: {np.std(r2_values):.3f}")
        print(f"  Beta - Mean: {np.mean(beta_values):.3f}, SD: {np.std(beta_values):.3f}")
        print(f"  Power (R²>0.01): {power_1pct:.1%}")
        print(f"  Power (R²>0.05): {power_5pct:.1%}")

    print("\n" + "=" * 60)
    print("Batch simulation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
