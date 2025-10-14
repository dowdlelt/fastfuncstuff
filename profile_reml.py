#!/usr/bin/env python
"""
Profile REML grid search to find bottlenecks
"""

import time

import torch

from fastfuncsim import get_device
from fastfuncsim.arma_glm import batch_reml_grid_search, precompute_reml_grid

device = get_device()
print(f"Device: {device}\n")

# Simulate realistic problem size
n_timepoints = 1200
n_regressors = 48
batch_size = 10000

# Create fake data
design = torch.randn(n_timepoints, n_regressors, device=device)
Y_batch = torch.randn(n_timepoints, batch_size, device=device)

# Grids - match what we're actually using
a_grid = torch.linspace(0.0, 0.8, 7, device=device)
b_grid = torch.linspace(-0.8, 0.8, 7, device=device)

print(f"Problem size:")
print(f"  Timepoints: {n_timepoints}")
print(f"  Regressors: {n_regressors}")
print(f"  Batch size: {batch_size}")
print(f"  Grid: {len(a_grid)} × {len(b_grid)} = {len(a_grid) * len(b_grid)} points\n")

# Profile precomputation
print("=" * 70)
print("PHASE 1: Precomputing REML grid")
print("=" * 70)
start = time.time()
precomputed = precompute_reml_grid(
    design, n_timepoints, a_grid, b_grid, device, verbose=True
)
precomp_time = time.time() - start
print(f"\n✓ Precomputation: {precomp_time:.2f}s")
print(f"  Valid (a,b) pairs: {len(precomputed)}/{len(a_grid) * len(b_grid)}")

# Profile batch search
print("\n" + "=" * 70)
print("PHASE 2: Batch REML grid search (using precomputed)")
print("=" * 70)
print(f"Processing {batch_size:,} voxels...")

# Warm-up
_ = batch_reml_grid_search(
    design, Y_batch[:, :100], a_grid, b_grid, device, precomputed=precomputed
)

# Actual timing
torch.cuda.synchronize() if device.type == "cuda" else None
start = time.time()
batch_params, batch_likelihoods = batch_reml_grid_search(
    design, Y_batch, a_grid, b_grid, device, precomputed=precomputed
)
torch.cuda.synchronize() if device.type == "cuda" else None
search_time = time.time() - start

print(f"\n✓ Batch search: {search_time:.2f}s")
print(f"  Time per voxel: {search_time / batch_size * 1000:.2f}ms")
print(f"  Throughput: {batch_size / search_time:.0f} voxels/sec")

# Expected time for full analysis
n_voxels_total = 392240
n_batches = (n_voxels_total + batch_size - 1) // batch_size
estimated_time = precomp_time + n_batches * search_time

print("\n" + "=" * 70)
print("ESTIMATED FULL ANALYSIS TIME")
print("=" * 70)
print(f"  Total voxels: {n_voxels_total:,}")
print(f"  Batches: {n_batches}")
print(f"  Precomputation: {precomp_time:.1f}s (once)")
print(f"  Per batch: {search_time:.1f}s")
print(f"  Total estimated: {estimated_time / 60:.1f} minutes")

print("\n" + "=" * 70)
print("BREAKDOWN")
print("=" * 70)
print(f"  Grid evaluation: {search_time:.2f}s for {len(precomputed)} (a,b) pairs")
print(f"  Per (a,b) pair: {search_time / len(precomputed) * 1000:.1f}ms")
print(
    f"  Per voxel per (a,b): {search_time / (batch_size * len(precomputed)) * 1e6:.1f}µs"
)

# Check what's in each iteration
print("\n" + "=" * 70)
print("DETAILED TIMING: One (a,b) evaluation")
print("=" * 70)
(a_val, b_val), cached = next(iter(precomputed.items()))
print(f"Testing (a,b) = ({a_val:.2f}, {b_val:.2f})")

L_inv = cached["L_inv"]
X_w = cached["X_w"]
XwTXw_reg = cached["XwTXw_reg"]

# Time each operation
torch.cuda.synchronize() if device.type == "cuda" else None

# 1. Prewhiten data
start = time.time()
Y_w = L_inv @ Y_batch
torch.cuda.synchronize() if device.type == "cuda" else None
t1 = time.time() - start
print(f"  1. Y_w = L_inv @ Y_batch: {t1 * 1000:.2f}ms")

# 2. Solve
start = time.time()
beta_w = torch.linalg.solve(XwTXw_reg, X_w.T @ Y_w)
torch.cuda.synchronize() if device.type == "cuda" else None
t2 = time.time() - start
print(f"  2. Solve for betas: {t2 * 1000:.2f}ms")

# 3. Residuals
start = time.time()
residuals_w = Y_w - X_w @ beta_w
torch.cuda.synchronize() if device.type == "cuda" else None
t3 = time.time() - start
print(f"  3. Compute residuals: {t3 * 1000:.2f}ms")

# 4. RSS
start = time.time()
rss = torch.sum(residuals_w**2, dim=0)
torch.cuda.synchronize() if device.type == "cuda" else None
t4 = time.time() - start
print(f"  4. Compute RSS: {t4 * 1000:.2f}ms")

# 5. Likelihood
start = time.time()
term1 = cached["logdet_R"]
term2 = cached["logdet_XwTXw"]
term3 = (n_timepoints - n_regressors) * torch.log(rss + 1e-10)
likelihoods = term1 + term2 + term3
torch.cuda.synchronize() if device.type == "cuda" else None
t5 = time.time() - start
print(f"  5. Compute likelihood: {t5 * 1000:.2f}ms")

total_per_ab = t1 + t2 + t3 + t4 + t5
print(f"\n  Total per (a,b): {total_per_ab * 1000:.2f}ms")
print(
    f"  Expected for {len(precomputed)} pairs: {total_per_ab * len(precomputed):.2f}s"
)

if search_time > total_per_ab * len(precomputed) * 2:
    print(
        f"\n⚠️  WARNING: Actual time ({search_time:.2f}s) >> expected ({total_per_ab * len(precomputed):.2f}s)"
    )
    print("  Something else is taking time!")
