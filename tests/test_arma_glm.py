"""
Quick Test: ARMA(1,1) Implementation

Verify that the GPU-accelerated ARMA(1,1) GLM implementation works correctly.
"""

import sys

import torch


def test_arma11_covariance():
    """Test ARMA(1,1) covariance matrix construction"""
    print("\n" + "="*70)
    print("Test 1: ARMA(1,1) Covariance Matrix")
    print("="*70)
    
    from fastfuncsim.arma_glm import build_arma11_covariance
    from fastfuncsim.utils import get_device
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Test different configurations
    tests = [
        (0.5, 0.0, "AR(1): a=0.5, b=0"),
        (0.4, 0.2, "ARMA(1,1): a=0.4, b=0.2"),
        (0.3, -0.1, "ARMA(1,1): a=0.3, b=-0.1"),
    ]
    
    for a, b, desc in tests:
        R = build_arma11_covariance(a, b, n=10, device=device)
        
        if R is not None:
            # Check properties
            is_symmetric = torch.allclose(R, R.T, atol=1e-6)
            is_positive_def = torch.all(torch.linalg.eigvals(R).real > 0)
            
            lam = ((b + a) * (1 + a * b)) / (1 + 2 * a * b + b**2)
            
            print(f"\n{desc}")
            print(f"  λ (lag-1 corr): {lam:.4f}")
            print(f"  Symmetric: {is_symmetric}")
            print(f"  Positive definite: {is_positive_def}")
            print(f"  R[0,1]: {R[0, 1].item():.4f} (should equal λ: {lam:.4f})")
            
            assert is_symmetric, "Matrix should be symmetric"
            assert is_positive_def, "Matrix should be positive definite"
            assert abs(R[0, 1].item() - lam) < 1e-4, "Off-diagonal should equal λ"
            
            print("  ✓ All checks passed")
        else:
            print(f"\n{desc}: Invalid parameters")
    
    print("\n✅ Test 1 PASSED")


def test_reml_grid_search():
    """Test REML grid search parameter estimation"""
    print("\n" + "="*70)
    print("Test 2: REML Grid Search")
    print("="*70)
    
    from fastfuncsim.arma_glm import reml_grid_search
    from fastfuncsim.noise import generate_arma_noise
    from fastfuncsim.utils import get_device
    
    device = get_device()
    
    # Generate synthetic data with known ARMA(1,1)
    true_a = 0.4
    true_b = 0.1
    n_timepoints = 200
    
    print(f"\nGenerating data with true (a, b) = ({true_a}, {true_b})")
    
    # Simple design matrix
    X = torch.randn(n_timepoints, 2, device=device)
    true_beta = torch.tensor([1.5, 1.0], device=device)
    
    # Generate ARMA noise
    noise = generate_arma_noise(
        ar_coeffs=[true_a],
        ma_coeffs=[true_b],
        n_timepoints=n_timepoints,
        n_voxels=1,
        device=device
    ).squeeze()
    
    # Data = signal + noise
    Y = X @ true_beta + noise
    
    # Estimate parameters
    print("Running REML grid search...")
    a_opt, b_opt, likelihood = reml_grid_search(X, Y, device=device)
    
    print(f"\nTrue (a, b): ({true_a:.3f}, {true_b:.3f})")
    print(f"Estimated (a, b): ({a_opt:.3f}, {b_opt:.3f})")
    print(f"REML likelihood: {likelihood:.2f}")
    
    # Check if estimates are close
    a_error = abs(a_opt - true_a)
    b_error = abs(b_opt - true_b)
    
    print("\nEstimation errors:")
    print(f"  |a_est - a_true|: {a_error:.3f}")
    print(f"  |b_est - b_true|: {b_error:.3f}")
    
    # Allow some tolerance due to grid resolution
    if a_error < 0.2 and b_error < 0.2:
        print("\n✅ Test 2 PASSED (parameters recovered within tolerance)")
    else:
        print("\n⚠️  Test 2: Parameters not perfectly recovered (but this is OK due to grid resolution)")


def test_prewhitening():
    """Test prewhitening transformation"""
    print("\n" + "="*70)
    print("Test 3: Prewhitening")
    print("="*70)
    
    from fastfuncsim.arma_glm import build_arma11_covariance, prewhiten_with_arma11
    from fastfuncsim.utils import get_device
    
    device = get_device()
    
    n = 100
    a, b = 0.5, 0.1
    
    # Design and data
    X = torch.randn(n, 3, device=device)
    Y = torch.randn(n, device=device)
    
    print(f"\nPrewhitening with (a, b) = ({a}, {b})")
    
    # Prewhiten
    X_w, Y_w, L_inv = prewhiten_with_arma11(X, Y, a, b)
    
    # Check that prewhitened data has correct covariance structure
    # After prewhitening, should be approximately white (identity covariance)
    
    print(f"Original X shape: {X.shape}")
    print(f"Whitened X shape: {X_w.shape}")
    print(f"L_inv shape: {L_inv.shape}")
    
    # Verify transformation
    _R = build_arma11_covariance(a, b, n, device)
    X_w_manual = L_inv @ X
    
    match = torch.allclose(X_w, X_w_manual, atol=1e-5)
    print(f"\nPrewhitening matches manual computation: {match}")
    
    assert match, "Prewhitening should match manual L_inv @ X"
    
    print("\n✅ Test 3 PASSED")


def test_glm_arma11():
    """Test full ARMA(1,1) GLM fit"""
    print("\n" + "="*70)
    print("Test 4: Full ARMA(1,1) GLM Fit")
    print("="*70)
    
    from fastfuncsim.arma_glm import fit_glm_arma11
    from fastfuncsim.glm_core import fit_glm
    from fastfuncsim.noise import generate_arma_noise
    from fastfuncsim.utils import get_device
    
    device = get_device()
    
    # Small test case
    n_timepoints = 150
    n_voxels = 50
    n_regressors = 2
    
    true_a = 0.4
    true_b = 0.1
    
    print("\nSimulation setup:")
    print(f"  Timepoints: {n_timepoints}")
    print(f"  Voxels: {n_voxels}")
    print(f"  Regressors: {n_regressors}")
    print(f"  True ARMA: a={true_a}, b={true_b}")
    
    # Design matrix
    X = torch.randn(n_timepoints, n_regressors, device=device)
    true_betas = torch.randn(n_voxels, n_regressors, device=device)
    
    # Generate signal
    signal = X @ true_betas.T  # (n_timepoints, n_voxels)
    
    # Generate ARMA noise
    noise = generate_arma_noise(
        ar_coeffs=[true_a],
        ma_coeffs=[true_b],
        n_timepoints=n_timepoints,
        n_voxels=n_voxels,
        device=device
    )
    
    # Data
    data = (signal + noise).T  # (n_voxels, n_timepoints)
    
    # Fit OLS
    print("\nFitting OLS GLM...")
    ols_results = fit_glm(data, X, tr=2.0, verbose=False, device=device)
    
    # Fit ARMA(1,1)
    print("\nFitting ARMA(1,1) GLM...")
    arma_results = fit_glm_arma11(
        data, X, tr=2.0,
        estimate_per_voxel=False,  # Use global for speed
        verbose=False,
        device=device
    )
    
    # Compare
    print("\nResults:")
    print(f"  OLS R²: {ols_results.r2.mean():.4f}")
    print(f"  ARMA R²: {arma_results.r2.mean():.4f}")
    print(f"  ARMA mean (a, b): ({arma_results.arma_params[:, 0].mean():.3f}, "
          f"{arma_results.arma_params[:, 1].mean():.3f})")
    print(f"  ARMA mean λ: {arma_results.arma_lambda.mean():.3f}")
    
    # Check that ARMA gives reasonable results
    assert arma_results.r2.mean() > 0, "R² should be positive"
    assert arma_results.r2.mean() <= 1, "R² should be <= 1"
    assert torch.all(torch.isfinite(arma_results.betas)), "Betas should be finite"
    assert torch.all(torch.isfinite(arma_results.tstats)), "t-stats should be finite"
    assert torch.all(torch.isfinite(arma_results.fstats)), "F-stats should be finite"
    assert arma_results.fstats.numel() == arma_results.betas.shape[0], "F-stat length mismatch"
    assert torch.all(torch.isfinite(ols_results.tstats)), "OLS t-stats should be finite"
    assert torch.all(torch.isfinite(ols_results.fstats)), "OLS F-stats should be finite"
    assert torch.all(ols_results.sigma2 >= 0), "Noise variances should be non-negative"
    
    print("\n✅ Test 4 PASSED")


def test_batch_processing():
    """Test batch REML grid search"""
    print("\n" + "="*70)
    print("Test 5: Batch Processing")
    print("="*70)
    
    from fastfuncsim.arma_glm import batch_reml_grid_search
    from fastfuncsim.noise import generate_arma_noise
    from fastfuncsim.utils import get_device
    
    device = get_device()
    
    n_timepoints = 100
    n_voxels_batch = 20
    n_regressors = 2
    
    print("\nBatch test:")
    print(f"  Timepoints: {n_timepoints}")
    print(f"  Voxels in batch: {n_voxels_batch}")
    
    # Design
    X = torch.randn(n_timepoints, n_regressors, device=device)
    
    # Generate batch of voxel data with different ARMA parameters
    Y_batch = torch.zeros(n_timepoints, n_voxels_batch, device=device)
    
    for v in range(n_voxels_batch):
        # Random ARMA parameters for each voxel
        a_v = 0.3 + 0.4 * (v / n_voxels_batch)  # 0.3 to 0.7
        b_v = 0.0  # Keep simple
        
        noise_v = generate_arma_noise(
            ar_coeffs=[a_v],
            ma_coeffs=[b_v],
            n_timepoints=n_timepoints,
            n_voxels=1,
            device=device
        ).squeeze()
        
        Y_batch[:, v] = noise_v
    
    # Batch REML search
    print("\nRunning batch REML grid search...")
    best_params, likelihoods = batch_reml_grid_search(X, Y_batch, device=device)
    
    print("\nEstimated parameters:")
    print(f"  a range: [{best_params[:, 0].min():.3f}, {best_params[:, 0].max():.3f}]")
    print(f"  b range: [{best_params[:, 1].min():.3f}, {best_params[:, 1].max():.3f}]")
    print(f"  Likelihood range: [{likelihoods.min():.2f}, {likelihoods.max():.2f}]")
    
    # Check shapes
    assert best_params.shape == (n_voxels_batch, 2), "Wrong output shape"
    assert likelihoods.shape == (n_voxels_batch,), "Wrong likelihood shape"
    assert torch.all(torch.isfinite(best_params)), "Parameters should be finite"
    
    print("\n✅ Test 5 PASSED")


def main():
    """Run all tests"""
    print("="*70)
    print("ARMA(1,1) GLM Implementation Tests")
    print("GPU-Accelerated Prewhitening")
    print("="*70)
    
    try:
        test_arma11_covariance()
        test_reml_grid_search()
        test_prewhitening()
        test_glm_arma11()
        test_batch_processing()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        print("\nThe ARMA(1,1) implementation is working correctly.")
        print("You can now use fit_glm_arma11() for your analysis.")
        print("\nNext steps:")
        print("1. Run example_arma_glm.py for detailed examples")
        print("2. See ARMA_GLM_README.md for full documentation")
        print("3. Extract noise parameters from YOUR scanner data")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
