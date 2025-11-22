"""Test adaptive batching path for ARMA grid search."""
import torch
from pathlib import Path

from fastfuncsim.analysis import analyze_from_design_matrix

# AFNI reference data paths
AFNI_DATA_DIR = Path.home() / "Dropbox/Data/small_validation_afni_data"
DESIGN_MATRIX = AFNI_DATA_DIR / "X.xmat.1D"
INPUT_FILES = [
    AFNI_DATA_DIR / "small_test_r01.nii.gz",
    AFNI_DATA_DIR / "small_test_r02.nii.gz",
]

print("Testing ADAPTIVE BATCHING path (use_grid_batching=True)...")
print("="*80)

results_batch, design_info = analyze_from_design_matrix(
    fmri_data=INPUT_FILES,
    design_matrix_file=DESIGN_MATRIX,
    method="arma11",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_grid_batching=True,  # Force adaptive batching
)

print("\nResults:")
print(f"  ARMA params shape: {results_batch.arma_params.shape}")
print(f"  Mean (a, b): ({results_batch.arma_params[:, 0].mean():.3f}, {results_batch.arma_params[:, 1].mean():.3f})")
print(f"  F-stats shape: {results_batch.fstats.shape}")
print(f"  Mean F-stat: {results_batch.fstats.mean():.3f}")

print("\n" + "="*80)
print("Testing PRECOMPUTED GRID path (use_grid_batching=False)...")
print("="*80)

results_precomp, _ = analyze_from_design_matrix(
    fmri_data=INPUT_FILES,
    design_matrix_file=DESIGN_MATRIX,
    method="arma11",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    use_grid_batching=False,  # Force precomputed grid
)

print("\nResults:")
print(f"  ARMA params shape: {results_precomp.arma_params.shape}")
print(f"  Mean (a, b): ({results_precomp.arma_params[:, 0].mean():.3f}, {results_precomp.arma_params[:, 1].mean():.3f})")
print(f"  F-stats shape: {results_precomp.fstats.shape}")
print(f"  Mean F-stat: {results_precomp.fstats.mean():.3f}")

# Compare the two methods
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

import numpy as np

a_diff = np.abs(results_batch.arma_params[:, 0] - results_precomp.arma_params[:, 0])
b_diff = np.abs(results_batch.arma_params[:, 1] - results_precomp.arma_params[:, 1])
fstat_diff = np.abs(results_batch.fstats - results_precomp.fstats)

print(f"\nParameter differences:")
print(f"  a: max={a_diff.max():.4f}, mean={a_diff.mean():.4f}")
print(f"  b: max={b_diff.max():.4f}, mean={b_diff.mean():.4f}")
print(f"  Voxels with a diff > 0.05: {(a_diff > 0.05).sum()}")
print(f"  Voxels with b diff > 0.05: {(b_diff > 0.05).sum()}")

print(f"\nF-stat differences:")
print(f"  max={fstat_diff.max():.4f}, mean={fstat_diff.mean():.4f}")

# Correlation
corr = np.corrcoef(results_batch.fstats, results_precomp.fstats)[0, 1]
print(f"  Correlation: {corr:.6f}")

if corr > 0.999:
    print("\n✓ Both paths produce highly consistent results!")
else:
    print(f"\n⚠ Warning: correlation {corr:.6f} suggests paths may differ")
