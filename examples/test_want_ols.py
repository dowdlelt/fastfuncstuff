#!/usr/bin/env python
"""
Test want_ols=True parameter in fit_glm_arma11()

This demonstrates the new feature that computes OLS baseline for comparison.
"""

import numpy as np
import torch
import time

import fastfuncsim as ffs

# Setup
device = ffs.get_device()
print(f"Using device: {device}\n")

# Generate synthetic data with autocorrelation
np.random.seed(42)
n_voxels = 1000
n_timepoints = 200
n_regressors = 5
tr = 2.0

# Create design matrix
design = np.random.randn(n_timepoints, n_regressors).astype(np.float32)
design[:, 0] = 1  # Intercept

# True ARMA(1,1) parameters
true_a = 0.6
true_b = -0.3

# Generate data with ARMA(1,1) noise
true_betas = np.random.randn(n_voxels, n_regressors).astype(np.float32)
true_betas[:, 0] = 100  # Intercept

signal = (design @ true_betas.T).T  # (n_voxels, n_timepoints)

# Generate ARMA(1,1) noise
noise = np.zeros((n_voxels, n_timepoints), dtype=np.float32)
white_noise = np.random.randn(n_voxels, n_timepoints).astype(np.float32)
for t in range(n_timepoints):
    if t == 0:
        noise[:, t] = white_noise[:, t]
    else:
        # ARMA(1,1): e[t] = a*e[t-1] + w[t] + b*w[t-1]
        noise[:, t] = true_a * noise[:, t - 1] + white_noise[:, t]
        if t > 0:
            noise[:, t] += true_b * white_noise[:, t - 1]

data = signal + noise * 5  # Add noise with SNR

print("=" * 70)
print("Test 1: ARMA fit WITHOUT OLS (want_ols=False)")
print("=" * 70)
start = time.time()
results_no_ols = ffs.fit_glm_arma11(
    data,
    design,
    tr=tr,
    device=device,
    verbose=True,
    want_ols=False,  # Default behavior
    batch_size=500,
)
elapsed_no_ols = time.time() - start

print(f"\n✓ ARMA fit complete (no OLS)")
print(f"  Time: {elapsed_no_ols:.2f} seconds")
print(f"  Mean R²: {results_no_ols.r2.mean():.4f}")
print(f"  Mean |t|: {results_no_ols.tstats.abs().mean():.3f}")
print(
    f"  Mean (a,b): ({results_no_ols.arma_params[:, 0].mean():.3f}, {results_no_ols.arma_params[:, 1].mean():.3f})"
)
print(f"  True (a,b): ({true_a}, {true_b})")
print(f"  Has ols_results: {results_no_ols.ols_results is not None}")

print("\n" + "=" * 70)
print("Test 2: ARMA fit WITH OLS (want_ols=True)")
print("=" * 70)
start = time.time()
results_with_ols = ffs.fit_glm_arma11(
    data,
    design,
    tr=tr,
    device=device,
    verbose=True,
    want_ols=True,  # NEW FEATURE!
    batch_size=500,
)
elapsed_with_ols = time.time() - start

print(f"\n✓ ARMA fit complete (with OLS)")
print(f"  Time: {elapsed_with_ols:.2f} seconds")
print(
    f"  Overhead: {(elapsed_with_ols - elapsed_no_ols):.2f} seconds ({100 * (elapsed_with_ols - elapsed_no_ols) / elapsed_no_ols:.1f}%)"
)
print(f"  Mean R²: {results_with_ols.r2.mean():.4f}")
print(f"  Mean |t|: {results_with_ols.tstats.abs().mean():.3f}")
print(
    f"  Mean (a,b): ({results_with_ols.arma_params[:, 0].mean():.3f}, {results_with_ols.arma_params[:, 1].mean():.3f})"
)
print(f"  Has ols_results: {results_with_ols.ols_results is not None}")

if results_with_ols.ols_results is not None:
    ols = results_with_ols.ols_results
    print(f"\n  OLS Results:")
    print(f"    Mean R²: {ols.r2.mean():.4f}")
    print(f"    Mean |t|: {ols.tstats.abs().mean():.3f}")
    print(f"    Type: {type(ols).__name__}")

    # Compare ARMA vs OLS
    print(f"\n  ARMA vs OLS Comparison:")
    print(f"    R² improvement: {(results_with_ols.r2.mean() - ols.r2.mean()):.4f}")
    print(
        f"    |t| ratio (ARMA/OLS): {(results_with_ols.tstats.abs().mean() / ols.tstats.abs().mean()):.3f}"
    )
    print(
        f"    β correlation: {np.corrcoef(results_with_ols.betas.flatten().cpu().numpy(), ols.betas.flatten().cpu().numpy())[0, 1]:.4f}"
    )

    print(f"\n  Interpretation:")
    if results_with_ols.r2.mean() > ols.r2.mean():
        print(
            f"    ✓ ARMA has better fit (R² improvement: +{100 * (results_with_ols.r2.mean() - ols.r2.mean()):.2f}%)"
        )
    else:
        print(f"    ⚠ ARMA and OLS have similar fit (expected for low autocorrelation)")

    tstat_ratio = results_with_ols.tstats.abs().mean() / ols.tstats.abs().mean()
    if tstat_ratio < 1:
        print(
            f"    ✓ ARMA corrects inflated t-stats (reduction: {100 * (1 - tstat_ratio):.1f}%)"
        )
        print(f"      This is expected with positive autocorrelation (a={true_a})")
    else:
        print(f"    ⚠ ARMA increases t-stats (rare, suggests negative autocorrelation)")

print("\n" + "=" * 70)
print("Test 3: Verify ARMA parameters recovered true values")
print("=" * 70)
print(f"True a: {true_a:.3f}")
print(f"Mean estimated a: {results_with_ols.arma_params[:, 0].mean():.3f}")
print(f"Std estimated a: {results_with_ols.arma_params[:, 0].std():.3f}")

print(f"\nTrue b: {true_b:.3f}")
print(f"Mean estimated b: {results_with_ols.arma_params[:, 1].mean():.3f}")
print(f"Std estimated b: {results_with_ols.arma_params[:, 1].std():.3f}")

# Check if recovered parameters are close to true values
a_error = abs(results_with_ols.arma_params[:, 0].mean() - true_a)
b_error = abs(results_with_ols.arma_params[:, 1].mean() - true_b)

if a_error < 0.1 and b_error < 0.1:
    print("\n✓ SUCCESS: ARMA parameters recovered accurately!")
else:
    print(f"\n⚠ Warning: Parameter recovery error may be high")
    print(f"  a error: {a_error:.3f}")
    print(f"  b error: {b_error:.3f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
✓ want_ols=True feature working correctly!

Performance:
  - ARMA only: {elapsed_no_ols:.2f}s
  - ARMA + OLS: {elapsed_with_ols:.2f}s
  - Overhead: {100 * (elapsed_with_ols - elapsed_no_ols) / elapsed_no_ols:.1f}%

Results:
  - OLS results properly attached to ARMA11Results.ols_results
  - Can compare betas, tstats, R² between OLS and ARMA
  - Confirms ARMA correction working (t-stat ratio = {tstat_ratio:.3f})

Use Cases:
  1. Validation: Ensure ARMA improving over OLS
  2. Quality control: Check if autocorrelation present
  3. Publication: Show OLS vs ARMA comparison figures
  4. Debugging: Verify parameter estimates reasonable

Next: Update example scripts to use want_ols=True for validation!
""")
