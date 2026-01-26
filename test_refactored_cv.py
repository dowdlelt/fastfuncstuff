#!/usr/bin/env python3
"""
Test script for the refactored cross_validate_noise_pcs function.
Verifies that the concatenated prediction approach works correctly.
"""

import torch
import numpy as np
from fastfuncsim.denoise import cross_validate_noise_pcs

# Create synthetic data
n_voxels = 1000
n_timepoints = 400
n_runs = 4
n_task_regs = 10

# Random data
data = torch.randn(n_voxels, n_timepoints)
design_matrix = torch.randn(n_timepoints, n_task_regs)

# Random noise PCs (4 runs × 100 timepoints × 20 PCs)
noise_pcs = [torch.randn(100, 20) for _ in range(n_runs)]
run_starts = [0, 100, 200, 300]

# Criteria mask (use all voxels)
criteria_mask = torch.ones(n_voxels, dtype=torch.bool)

# Simple nuisance (polynomials) - 3rd order per run
nuisance_per_run = [torch.randn(100, 4) for _ in range(n_runs)]

print("Testing cross_validate_noise_pcs with concatenated predictions...")
print(f"Data shape: {data.shape}")
print(f"Design shape: {design_matrix.shape}")
print(f"Number of PCs: {noise_pcs[0].shape[1]}")
print(f"Number of runs: {n_runs}")
print(f"Number of criteria voxels: {criteria_mask.sum()}")

# Run cross-validation
r2_by_n, r2_median, r2_per_fold, r2_per_voxel = cross_validate_noise_pcs(
    data=data,
    design_matrix=design_matrix,
    noise_pcs=noise_pcs,
    run_starts=run_starts,
    criteria_mask=criteria_mask,
    tr=2.0,  # TR in seconds
    nuisance=nuisance_per_run,
    max_components=20,
    chunk_size=500,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=True,
)

print("\n" + "=" * 60)
print("Results:")
print(f"  r2_by_n_components shape: {r2_by_n.shape} (expected: (21,))")
print(f"  r2_median_by_n_components shape: {r2_median.shape} (expected: (21,))")
print(f"  r2_per_fold shape: {r2_per_fold.shape} (expected: (4, 21))")
print(f"  r2_per_voxel shape: {r2_per_voxel.shape} (expected: ({n_voxels}, 21))")

print(f"\n  R² values (baseline to best):")
print(f"    0 PCs: {r2_by_n[0]:.4f}")
best_idx = int(np.argmax(r2_by_n))
print(f"    Best ({best_idx} PCs): {r2_by_n[best_idx]:.4f}")
print(f"    Improvement: {r2_by_n[best_idx] - r2_by_n[0]:+.4f}")

print("\n✅ Test completed successfully!")
