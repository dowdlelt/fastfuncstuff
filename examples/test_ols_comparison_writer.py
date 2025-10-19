#!/usr/bin/env python
"""
Test write_ols_arma_comparison() function

Demonstrates creating side-by-side OLS vs ARMA bucket files for validation.
"""

import numpy as np
import torch
from pathlib import Path

import fastfuncsim as ffs

# Setup
device = ffs.get_device()
print(f"Using device: {device}\n")

# Output directory
output_dir = Path("test_ols_outputs")
output_dir.mkdir(exist_ok=True)

# Generate synthetic data with autocorrelation
np.random.seed(42)
n_voxels = 1000
n_timepoints = 200
n_regressors = 5
tr = 2.0

# Create design matrix
design = np.random.randn(n_timepoints, n_regressors).astype(np.float32)
design[:, 0] = 1  # Intercept
design[:, 1] = np.linspace(-1, 1, n_timepoints)  # Linear trend
design[:, 2] = np.sin(2 * np.pi * np.arange(n_timepoints) / 50)  # Task 1
design[:, 3] = np.cos(2 * np.pi * np.arange(n_timepoints) / 50)  # Task 2
design[:, 4] = np.random.randn(n_timepoints)  # Nuisance

# True ARMA(1,1) parameters
true_a = 0.6
true_b = -0.3

# Generate data with ARMA(1,1) noise
true_betas = np.random.randn(n_voxels, n_regressors).astype(np.float32)
true_betas[:, 0] = 100  # Intercept
true_betas[:, 1] = 2  # Trend effect
true_betas[:, 2] = 5  # Task 1 effect
true_betas[:, 3] = 3  # Task 2 effect

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

data = signal + noise * 5  # Add noise

# Reshape to 3D volume for testing (10x10x10)
volume_shape = (10, 10, 10)
data_3d = data.reshape(*volume_shape, n_timepoints)

print("=" * 70)
print("Fitting GLM with OLS comparison (want_ols=True)")
print("=" * 70)
results = ffs.fit_glm_arma11(
    data,
    design,
    tr=tr,
    device=device,
    verbose=True,
    want_ols=True,  # Enable OLS comparison
    batch_size=500,
)

# Store volume shape in results for writing
results.original_shape = volume_shape
results.ols_results.original_shape = volume_shape

print(f"\n✓ ARMA fit complete with OLS comparison")
print(f"  Mean R² (ARMA): {results.r2.mean():.4f}")
print(f"  Mean R² (OLS):  {results.ols_results.r2.mean():.4f}")
print(f"  R² improvement: {(results.r2.mean() - results.ols_results.r2.mean()):.4f}")

# Compute contrasts for both OLS and ARMA
print("\n" + "=" * 70)
print("Computing contrasts for OLS and ARMA")
print("=" * 70)

contrasts = {
    "Task1": np.array([0, 0, 1, 0, 0]),
    "Task2": np.array([0, 0, 0, 1, 0]),
    "Task1 > Task2": np.array([0, 0, 1, -1, 0]),
}

arma_contrasts = ffs.compute_contrasts(
    results,
    contrasts,
    device=device,
)

ols_contrasts = ffs.compute_contrasts(
    results.ols_results,
    contrasts,
    device=device,
)

print(f"✓ Contrasts computed for both methods")
print(f"  Task1 t-stat (ARMA): {arma_contrasts['Task1']['tstat'].mean():.3f}")
print(f"  Task1 t-stat (OLS):  {ols_contrasts['Task1']['tstat'].mean():.3f}")

# Write side-by-side comparison
print("\n" + "=" * 70)
print("Writing OLS vs ARMA comparison files")
print("=" * 70)

output_files = ffs.write_ols_arma_comparison(
    results,
    output_dir / "test_analysis",
    condition_names=["Intercept", "Trend", "Task1", "Task2", "Nuisance"],
    contrast_names=list(contrasts.keys()),
    contrast_results_arma=arma_contrasts,
    contrast_results_ols=ols_contrasts,
    volume_shape=volume_shape,
    voxel_size=(2.0, 2.0, 2.0),
    apply_afni_metadata=True,
    compress_output=True,
)

print("\n" + "=" * 70)
print("SUCCESS! Files created:")
print("=" * 70)
print(f"  OLS bucket:  {output_files['ols']}")
print(f"  ARMA bucket: {output_files['arma']}")
print(f"  Summary:     {output_files['comparison_summary']}")

# Show summary statistics
import json

with open(output_files["comparison_summary"]) as f:
    summary = json.load(f)

print("\n" + "=" * 70)
print("Quantitative Comparison")
print("=" * 70)
print(f"\nModel Fit:")
print(f"  OLS R²:  {summary['ols']['mean_r2']:.4f}")
print(f"  ARMA R²: {summary['arma']['mean_r2']:.4f}")
print(f"  Improvement: +{100 * summary['comparison']['r2_improvement']:.2f}%")

print(f"\nStatistical Inference:")
print(f"  OLS |t|:  {summary['ols']['mean_abs_tstat']:.3f}")
print(f"  ARMA |t|: {summary['arma']['mean_abs_tstat']:.3f}")
print(f"  Ratio: {summary['comparison']['tstat_ratio']:.3f}")

print(f"\nARMA Parameters:")
print(f"  Mean a: {summary['arma']['mean_a']:.3f} (true: {true_a})")
print(f"  Mean b: {summary['arma']['mean_b']:.3f} (true: {true_b})")

print(f"\nInterpretation:")
print(f"  • {summary['interpretation']['fit_quality']}")
print(f"  • {summary['interpretation']['tstat_correction']}")

print("\n" + "=" * 70)
print("Next steps:")
print("=" * 70)
print("""
1. View in AFNI:
   afni test_ols_outputs/

2. Compare side-by-side:
   afni -com "OPEN_WINDOW A.axialimage" \\
        -com "SWITCH_UNDERLAY test_analysis_OLS.nii.gz" \\
        -com "SWITCH_OVERLAY test_analysis_ARMA.nii.gz"

3. Check the JSON summary:
   cat test_ols_outputs/test_analysis_comparison_summary.json

4. Use in your analysis scripts:
   - Set want_ols=True when fitting
   - Compute contrasts for both OLS and ARMA
   - Call write_ols_arma_comparison() to save

This validation ensures ARMA is correcting for autocorrelation properly!
""")
