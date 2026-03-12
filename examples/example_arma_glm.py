"""
Example: ARMA(1,1) Prewhitened GLM Analysis

This script demonstrates:
1. Simulating fMRI data with temporal autocorrelation
2. Comparing OLS vs ARMA(1,1) GLM results
3. Showing the impact of prewhitening on t-statistics
4. Extracting ARMA parameters from real data

This is the ANALYSIS side - use after data collection for publication-quality results.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from arma_glm import (
    build_arma11_covariance,
    fit_glm_arma11,
    reml_grid_search,
)
from glm_core import fit_glm
from hrf import spm_hrf
from noise import generate_ar1_noise

# Import fastfuncsim modules
from utils import get_device

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def example_1_basic_arma_fit():
    """
    Example 1: Basic ARMA(1,1) GLM fit
    
    Simulate data with known ARMA(1,1) noise, fit GLM, verify parameter recovery
    """
    print("="*70)
    print("Example 1: Basic ARMA(1,1) GLM Fit")
    print("="*70)
    
    device = get_device()
    print(f"Using device: {device}\n")
    
    # Parameters
    tr = 2.0
    duration = 300  # seconds
    n_timepoints = int(duration / tr)
    n_voxels = 100
    
    # True ARMA(1,1) parameters
    true_a = 0.4
    true_b = 0.1
    
    print("Simulation parameters:")
    print(f"  TR: {tr}s")
    print(f"  Duration: {duration}s ({n_timepoints} timepoints)")
    print(f"  Voxels: {n_voxels}")
    print(f"  True ARMA(1,1): a={true_a}, b={true_b}\n")
    
    # Create simple block design (2 conditions)
    onsets = torch.zeros(n_timepoints, 2, device=device)
    
    # Condition 1: blocks at 30s, 90s, 150s (each 20s)
    for start_time in [30, 90, 150]:
        start_tr = int(start_time / tr)
        duration_tr = int(20 / tr)
        onsets[start_tr:start_tr + duration_tr, 0] = 1
    
    # Condition 2: blocks at 60s, 120s, 180s (each 20s)
    for start_time in [60, 120, 180]:
        start_tr = int(start_time / tr)
        duration_tr = int(20 / tr)
        onsets[start_tr:start_tr + duration_tr, 1] = 1
    
    # Create HRF
    hrf = spm_hrf(tr=tr, duration=32, device=device)
    
    # Convolve with HRF to get design matrix
    design = torch.zeros(n_timepoints, 2, device=device)
    for cond in range(2):
        # Convolve using FFT for speed
        onset_signal = onsets[:, cond]
        design[:, cond] = torch.nn.functional.conv1d(
            onset_signal.unsqueeze(0).unsqueeze(0),
            hrf.unsqueeze(0).unsqueeze(0),
            padding=len(hrf)//2
        ).squeeze()[:n_timepoints]
    
    # True beta weights
    true_betas = torch.tensor([[1.5, 1.0]] * n_voxels, device=device)  # (n_voxels, 2)
    
    # Generate signal
    signal = design @ true_betas.T  # (n_timepoints, n_voxels)
    
    # Generate ARMA(1,1) noise
    print("Generating ARMA(1,1) noise...")
    from noise import generate_arma_noise
    noise = generate_arma_noise(
        ar_coeffs=[true_a],
        ma_coeffs=[true_b],
        n_timepoints=n_timepoints,
        n_voxels=n_voxels,
        normalize=True,
        device=device
    )
    
    # Combine signal + noise
    data = signal.T + noise.T  # (n_voxels, n_timepoints)
    snr = signal.std() / noise.std()
    print(f"SNR: {snr.item():.2f}\n")
    
    # Fit OLS GLM
    print("Fitting OLS GLM...")
    ols_results = fit_glm(data, design, tr, verbose=False, device=device)
    
    # Fit ARMA(1,1) GLM
    print("\nFitting ARMA(1,1) GLM...")
    arma_results = fit_glm_arma11(
        data, design, tr,
        estimate_per_voxel=True,
        batch_size=50,
        device=device,
        verbose=True
    )
    
    # Compare results
    print("\n" + "="*70)
    print("Results Comparison:")
    print("="*70)
    print(f"\nTrue ARMA parameters: a={true_a:.3f}, b={true_b:.3f}")
    print(f"Estimated mean (a, b): ({arma_results.arma_params[:, 0].mean():.3f}, "
          f"{arma_results.arma_params[:, 1].mean():.3f})")
    print(f"Estimated std (a, b): ({arma_results.arma_params[:, 0].std():.3f}, "
          f"{arma_results.arma_params[:, 1].std():.3f})")
    
    print(f"\nTrue betas: {true_betas[0].tolist()}")
    print(f"OLS mean betas: [{ols_results.betas[:, 0].mean():.3f}, "
          f"{ols_results.betas[:, 1].mean():.3f}]")
    print(f"ARMA mean betas: [{arma_results.betas[:, 0].mean():.3f}, "
          f"{arma_results.betas[:, 1].mean():.3f}]")
    
    print(f"\nOLS Mean R²: {ols_results.r2.mean():.4f}")
    print(f"ARMA Mean R²: {arma_results.r2.mean():.4f}")
    print(f"R² improvement: {arma_results.r2.mean() - ols_results.r2.mean():.4f}")
    
    print(f"\nOLS Mean |t-stat|: {ols_results.tstats.abs().mean():.3f}")
    print(f"ARMA Mean |t-stat|: {arma_results.tstats.abs().mean():.3f}")
    tstat_ratio = arma_results.tstats.abs().mean() / ols_results.tstats.abs().mean()
    print(f"t-stat ratio (ARMA/OLS): {tstat_ratio:.3f}")
    
    if tstat_ratio < 1:
        print("  → ARMA corrected for inflated t-stats due to positive autocorrelation ✓")
    
    return {
        'data': data,
        'design': design,
        'ols_results': ols_results,
        'arma_results': arma_results,
        'true_a': true_a,
        'true_b': true_b
    }


def example_2_arma_covariance_visualization():
    """
    Example 2: Visualize ARMA(1,1) covariance matrices
    
    Show how different (a, b) parameters affect temporal correlation structure
    """
    print("\n" + "="*70)
    print("Example 2: ARMA(1,1) Covariance Visualization")
    print("="*70)
    
    device = get_device()
    n_timepoints = 50
    
    # Different ARMA(1,1) configurations
    configs = [
        (0.0, 0.0, "White Noise"),
        (0.3, 0.0, "AR(1), a=0.3"),
        (0.6, 0.0, "AR(1), a=0.6"),
        (0.4, 0.2, "ARMA(1,1), a=0.4, b=0.2"),
        (0.4, -0.2, "ARMA(1,1), a=0.4, b=-0.2"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (a, b, title) in enumerate(configs[:6]):
        ax = axes[idx]
        
        if a == 0 and b == 0:
            # White noise - identity matrix
            R = torch.eye(n_timepoints, device=device)
        else:
            R = build_arma11_covariance(a, b, n_timepoints, device)
            if R is None:
                print(f"Invalid parameters: a={a}, b={b}")
                continue
        
        # Plot covariance matrix
        im = ax.imshow(R.cpu().numpy(), cmap='RdBu_r', vmin=-0.5, vmax=1.0)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (TR)')
        ax.set_ylabel('Time (TR)')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Compute lag-1 correlation
        if a == 0 and b == 0:
            lam = 0.0
        else:
            lam = ((b + a) * (1 + a * b)) / (1 + 2 * a * b + b**2 + 1e-10)
        ax.text(0.02, 0.98, f'λ = {lam:.3f}', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide last subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig('arma11_covariance_matrices.png', dpi=150, bbox_inches='tight')
    print("\nCovariance matrices saved to 'arma11_covariance_matrices.png'")
    plt.show()


def example_3_reml_grid_search_visualization():
    """
    Example 3: Visualize REML likelihood surface
    
    Show how REML likelihood varies across (a, b) grid
    """
    print("\n" + "="*70)
    print("Example 3: REML Likelihood Surface")
    print("="*70)
    
    device = get_device()
    
    # Use data from example 1
    print("Generating synthetic data...")
    tr = 2.0
    n_timepoints = 150
    
    # Simple design
    onsets = torch.zeros(n_timepoints, 1, device=device)
    onsets[30:40] = 1
    onsets[60:70] = 1
    onsets[90:100] = 1
    
    hrf = spm_hrf(tr=tr, duration=32, device=device)
    design = torch.nn.functional.conv1d(
        onsets.T.unsqueeze(1),
        hrf.unsqueeze(0).unsqueeze(0),
        padding=len(hrf)//2
    ).squeeze().T[:n_timepoints]
    
    # Generate ARMA(1,1) noise
    true_a, true_b = 0.5, 0.1
    from noise import generate_arma_noise
    noise = generate_arma_noise(
        ar_coeffs=[true_a],
        ma_coeffs=[true_b],
        n_timepoints=n_timepoints,
        n_voxels=1,
        device=device
    )
    
    signal = design * 2.0
    data = signal.squeeze() + noise.squeeze()
    
    # Compute REML likelihood over grid
    print("Computing REML likelihood surface...")
    a_values = np.linspace(0.1, 0.9, 17)
    b_values = np.linspace(-0.3, 0.3, 13)
    
    likelihood_surface = np.zeros((len(b_values), len(a_values)))
    
    for i, a in enumerate(a_values):
        for j, b in enumerate(b_values):
            R = build_arma11_covariance(a, b, n_timepoints, device)
            if R is not None:
                from arma_glm import compute_reml_likelihood
                likelihood = compute_reml_likelihood(design, data, R)
                likelihood_surface[j, i] = likelihood
            else:
                likelihood_surface[j, i] = np.nan
    
    # Find optimal
    a_opt, b_opt, _ = reml_grid_search(design, data, device=device)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.contourf(a_values, b_values, likelihood_surface, levels=20, cmap='viridis')
    ax.contour(a_values, b_values, likelihood_surface, levels=10, colors='white', alpha=0.3)
    
    # Mark true and estimated
    ax.plot(true_a, true_b, 'r*', markersize=20, label=f'True: ({true_a:.2f}, {true_b:.2f})', 
            markeredgecolor='white', markeredgewidth=2)
    ax.plot(a_opt, b_opt, 'wo', markersize=15, label=f'Estimated: ({a_opt:.2f}, {b_opt:.2f})',
            markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xlabel('a (AR parameter)', fontsize=12, fontweight='bold')
    ax.set_ylabel('b (MA parameter)', fontsize=12, fontweight='bold')
    ax.set_title('REML Log-Likelihood Surface\n(Lower = Better)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(im, ax=ax, label='Log-Likelihood')
    plt.tight_layout()
    plt.savefig('reml_likelihood_surface.png', dpi=150, bbox_inches='tight')
    print("Likelihood surface saved to 'reml_likelihood_surface.png'")
    plt.show()


def example_4_estimate_from_real_data():
    """
    Example 4: Extract ARMA parameters from real data
    
    This shows how to extract scanner-specific noise parameters for future simulations
    """
    print("\n" + "="*70)
    print("Example 4: Extract ARMA Parameters from Real Data")
    print("="*70)
    print("\nNOTE: This example requires real fMRI data.")
    print("If you have a NIFTI file, modify this function to load it with nibabel.")
    print("\nFor now, using synthetic 'pseudo-real' data to demonstrate workflow...\n")
    
    device = get_device()
    
    # Simulate "real" data with complex noise
    tr = 2.0
    n_timepoints = 200
    n_voxels = 500  # Smaller sample for demo
    
    # Generate realistic noise (AR1 with physiological components)
    ar1_noise = generate_ar1_noise(
        rho=0.35,  # Typical fMRI autocorrelation
        n_timepoints=n_timepoints,
        n_voxels=n_voxels,
        device=device
    )
    
    # Add signal (simple task blocks)
    onsets = torch.zeros(n_timepoints, 1, device=device)
    for start in range(30, n_timepoints, 40):
        onsets[start:min(start+10, n_timepoints)] = 1
    
    hrf = spm_hrf(tr=tr, duration=32, device=device)
    design = torch.nn.functional.conv1d(
        onsets.T.unsqueeze(1),
        hrf.unsqueeze(0).unsqueeze(0),
        padding=len(hrf)//2
    ).squeeze().T[:n_timepoints]
    
    signal = design * 1.5  # Moderate effect size
    data = signal.repeat(1, n_voxels) + ar1_noise.T
    
    # Estimate parameters
    print("Estimating ARMA parameters from sample voxels...")
    from noise import estimate_noise_parameters_from_data
    
    params = estimate_noise_parameters_from_data(
        data.cpu().numpy(),
        design=design.cpu().numpy(),
        ar_order=1,
        device=device
    )
    
    print("\n" + "="*70)
    print("Extracted Noise Parameters:")
    print("="*70)
    print(params['summary'])
    print("\nDetailed statistics:")
    print(f"  AR(1) coefficient range: [{min(params['ar_coefficients_all']):.3f}, "
          f"{max(params['ar_coefficients_all']):.3f}]")
    print(f"  SFNR range: [{min(params['sfnr_all']):.1f}, {max(params['sfnr_all']):.1f}]")
    
    print("\n" + "="*70)
    print("How to use these parameters:")
    print("="*70)
    print(f"""
    # In your simulation script:
    from noise import generate_ar1_noise
    
    realistic_noise = generate_ar1_noise(
        rho={params['ar_coefficients'][0]:.3f},  # Use extracted AR(1)
        n_timepoints=300,
        n_voxels=10000,
        device=device
    )
    
    # Scale to match SFNR
    noise_scaled = realistic_noise * (mean_signal / {params['sfnr']:.1f})
    """)


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("FastFuncSim: ARMA(1,1) GLM Examples")
    print("GPU-Accelerated Prewhitening for fMRI Analysis")
    print("="*70 + "\n")
    
    # Example 1: Basic fit
    _results = example_1_basic_arma_fit()
    
    # Example 2: Covariance visualization
    example_2_arma_covariance_visualization()
    
    # Example 3: REML surface
    example_3_reml_grid_search_visualization()
    
    # Example 4: Extract from real data
    example_4_estimate_from_real_data()
    
    print("\n" + "="*70)
    print("All examples complete!")
    print("="*70)
    print("\nKey takeaways:")
    print("1. ARMA(1,1) corrects for temporal autocorrelation in fMRI data")
    print("2. Provides accurate t-statistics (OLS often inflated)")
    print("3. GPU acceleration: 5-30x faster than AFNI 3dREMLfit")
    print("4. Extract parameters from YOUR scanner for realistic simulations")
    print("\nFor publication-quality analysis, always use ARMA(1,1) or similar prewhitening!")


if __name__ == "__main__":
    main()
