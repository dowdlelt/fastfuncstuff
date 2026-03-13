"""
Example: Single Interactive Simulation
Demonstrates fast interactive workflow for exploring fMRI simulations
"""

import matplotlib.pyplot as plt
import torch

import fastfuncstuff as ffs


def main():
    print("=" * 60)
    print("FastFuncSim - Interactive Single Simulation Example")
    print("=" * 60)

    # Setup
    device = ffs.get_device()
    ffs.print_device_info(device)

    # Experiment parameters (matching MATLAB simulate_movietasks.m)
    tr = 1.0
    total_duration_s = 290
    stim_duration = 5.0
    n_conditions = 2
    n_runs = 4
    n_timepoints = int(total_duration_s / tr)

    # Small test size for interactive speed
    matrix_size = (50, 50, 5)  # 12,500 voxels

    print("\nExperiment Setup:")
    print(f"  TR: {tr}s")
    print(f"  Duration: {total_duration_s}s ({n_timepoints} TRs)")
    print(f"  Conditions: {n_conditions}")
    print(f"  Runs: {n_runs}")
    print(f"  Matrix size: {matrix_size} ({matrix_size[0]*matrix_size[1]*matrix_size[2]:,} voxels)")

    # Generate HRF
    print("\n1. Generating HRF...")
    hrf = ffs.get_canonical_hrf(stim_duration=stim_duration, tr=tr, device=device)
    print(f"   HRF shape: {hrf.shape}")

    # Plot HRF
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(torch.arange(len(hrf)).cpu() * tr, hrf.cpu())
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Canonical HRF')
    plt.grid(True)

    # Generate task design
    print("\n2. Generating task design...")
    onsets_list = []
    for run in range(n_runs):
        onsets = ffs.generate_random_onsets(
            n_timepoints=n_timepoints,
            n_conditions=n_conditions,
            isi_mean=4.0,  # Mean ISI in seconds
            isi_range=(2, 8),
            tr=tr,
            alternate_conditions=True,
            device=device
        )
        onsets_list.append(onsets)
        print(f"   Run {run+1}: {onsets.sum().item():.0f} trials")

    # Plot example design matrix
    plt.subplot(1, 2, 2)
    design = ffs.build_glm_design(onsets_list[0], hrf, n_timepoints, mode='assumed', device=device)
    plt.plot(design.cpu())
    plt.xlabel('Time (TRs)')
    plt.ylabel('Amplitude')
    plt.title('Example Design Matrix')
    plt.legend([f'Condition {i+1}' for i in range(n_conditions)])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('example_design.png', dpi=150)
    print("   Saved: example_design.png")

    # Simulate data
    print("\n3. Simulating fMRI data...")
    betas = [5.0, 3.0]  # Different betas for each condition
    data_list = []

    for run_idx in range(n_runs):
        print(f"   Simulating run {run_idx+1}/{n_runs}...", end=' ')
        data = ffs.simulate_fmri_run(
            onsets=onsets_list[run_idx],
            betas=betas,
            hrf=hrf,
            tr=tr,
            n_timepoints=n_timepoints,
            matrix_size=matrix_size,
            noise_level=1.0,
            baseline=100.0,
            add_scanner_drift=True,
            device=device
        )
        data_list.append(data)
        print(f"shape: {data.shape}, range: [{data.min().item():.1f}, {data.max().item():.1f}]")

    # Fit GLM with assumed HRF
    print("\n4. Fitting GLM (Assumed HRF)...")
    results_assumed = ffs.fit_glm(
        data=data_list,
        design=[ffs.build_glm_design(o, hrf, n_timepoints, mode='assumed', device=device)
                for o in onsets_list],
        tr=tr,
        want_residuals=False,
        want_predicted=False,
        want_r2_run=True,
        device=device,
        verbose=True
    )

    print("\n   Results:")
    print(f"   Mean R²: {results_assumed.r2.mean().item():.3f}")
    print(f"   Median R²: {results_assumed.r2.median().item():.3f}")
    print(f"   Max R²: {results_assumed.r2.max().item():.3f}")
    print(f"   Betas shape: {results_assumed.betas.shape}")

    # Fit GLM with FIR (no HRF assumption)
    print("\n5. Fitting GLM (FIR - no HRF assumption)...")
    n_fir_lags = 30  # 30 TRs ≈ 30s response
    results_fir = ffs.fit_glm(
        data=data_list,
        design=[ffs.build_glm_design(o, None, n_timepoints, mode='fir',
                                     n_fir_lags=n_fir_lags, device=device)
                for o in onsets_list],
        tr=tr,
        want_r2_run=False,
        device=device,
        verbose=True
    )

    print("\n   Results:")
    print(f"   Mean R²: {results_fir.r2.mean().item():.3f}")
    print(f"   Betas shape: {results_fir.betas.shape}")

    # Visualizations using new visualization module
    print("\n6. Creating visualizations...")

    # Deep dive for assumed HRF results
    print("   Creating deep dive visualization...")
    fig_deep = ffs.plot_simulation_deep_dive(
        data=data_list[0],  # Show first run
        design=ffs.build_glm_design(onsets_list[0], hrf, n_timepoints, mode='assumed', device=device),
        results=results_assumed,
        onsets=onsets_list[0],
        betas_true=torch.tensor(betas),
        hrf_true=hrf,
        voxel_selection='best',
        n_voxels=4,
        tr=tr,
        save_path='example_deep_dive.png'
    )
    plt.close(fig_deep)

    # HRF recovery visualization
    print("   Creating HRF recovery visualization...")
    # Extract FIR estimates for condition 1
    fir_estimates = results_fir.betas[:, :n_fir_lags]
    true_hrf_padded = torch.cat([hrf, torch.zeros(n_fir_lags - len(hrf), device=device)])[:n_fir_lags]

    fig_hrf = ffs.plot_hrf_recovery(
        hrf_estimated=fir_estimates,
        hrf_true=true_hrf_padded,
        tr=tr,
        voxel_selection='best',
        n_voxels=6,
        save_path='example_hrf_recovery.png'
    )
    plt.close(fig_hrf)

    # Design comparison
    print("   Creating design comparison...")
    designs_dict = {
        'Assumed HRF': ffs.build_glm_design(onsets_list[0], hrf, n_timepoints, mode='assumed', device=device),
        'FIR (30 lags)': ffs.build_glm_design(onsets_list[0], None, n_timepoints, mode='fir',
                                               n_fir_lags=n_fir_lags, device=device)[:, :2*n_fir_lags],  # Show first 2 conditions
    }
    fig_design = ffs.plot_design_comparison(
        designs=designs_dict,
        labels=['Condition 1', 'Condition 2'],
        tr=tr,
        save_path='example_design_comparison.png'
    )
    plt.close(fig_design)

    # Legacy visualization (compare estimated vs true HRF)
    plt.figure(figsize=(12, 5))

    # Select a high-R² voxel
    best_voxel = torch.argmax(results_fir.r2).item()
    print(f"\n   Best voxel: {best_voxel} (R²={results_fir.r2[best_voxel].item():.3f})")

    # Plot FIR estimates for each condition
    plt.subplot(1, 2, 1)
    for cond in range(n_conditions):
        fir_betas = results_fir.betas[best_voxel, cond*n_fir_lags:(cond+1)*n_fir_lags].cpu()
        fir_betas = fir_betas / fir_betas.max()  # Normalize
        plt.plot(torch.arange(n_fir_lags).numpy() * tr, fir_betas.numpy(),
                label=f'Condition {cond+1} (FIR)')

    # Plot true HRF
    true_hrf = hrf.cpu() * betas[0]  # Scale by beta
    true_hrf = true_hrf / true_hrf.max()
    t_hrf = torch.arange(len(true_hrf)).numpy() * tr
    plt.plot(t_hrf, true_hrf.numpy(), 'k--', linewidth=2, label='True HRF')

    plt.xlabel('Time (s)')
    plt.ylabel('Response (normalized)')
    plt.title(f'FIR Estimate vs True HRF (Voxel {best_voxel})')
    plt.legend()
    plt.grid(True)

    # Plot R² histogram
    plt.subplot(1, 2, 2)
    plt.hist(results_assumed.r2.cpu().numpy(), bins=50, alpha=0.5, label='Assumed HRF')
    plt.hist(results_fir.r2.cpu().numpy(), bins=50, alpha=0.5, label='FIR')
    plt.xlabel('R²')
    plt.ylabel('Count')
    plt.title('R² Distribution')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('example_results.png', dpi=150)
    print("   Saved: example_results.png")

    print("\n" + "=" * 60)
    print("Interactive simulation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
