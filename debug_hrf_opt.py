#!/usr/bin/env python3
"""
Diagnostic script to find why 3dHRFoptfast produces garbage R² and fits
while 3dDenoisefast works perfectly with the same data.

This creates a minimal synthetic test case and traces through both code paths.
"""
import torch
import numpy as np

# Setup
device = torch.device("cpu")
torch.manual_seed(42)

# Simulate: 4 runs of 100 TRs each, 1 condition, 500 voxels
n_runs = 4
n_tps_per_run = 100
n_timepoints = n_runs * n_tps_per_run
n_voxels = 500
n_conditions = 1
tr = 2.0
microtime_dt = 0.1
polort = 2
bins_per_tr = int(round(tr / microtime_dt))
n_microtime = n_timepoints * bins_per_tr

run_starts = [i * n_tps_per_run for i in range(n_runs)]

# Create onset matrix with events at regular intervals
onset_matrix = torch.zeros((n_microtime, n_conditions), device=device)
duration_s = 3.0
duration_bins = int(round(duration_s / microtime_dt))

for run_idx in range(n_runs):
    run_start_micro = run_starts[run_idx] * bins_per_tr
    # Place events every 20 TRs (40s), starting at TR 5
    for event_onset_tr in range(5, n_tps_per_run, 20):
        event_onset_micro = run_start_micro + event_onset_tr * bins_per_tr
        if event_onset_micro + duration_bins < n_microtime:
            onset_matrix[event_onset_micro:event_onset_micro + duration_bins, 0] = 1.0

print(f"Onset matrix: {onset_matrix.shape}, sum={onset_matrix.sum():.0f}")

# Create HRF library
from fastfuncsim.hrf import get_hrf_library, get_spmg1_hrf
from fastfuncsim.design import convolve_hrf_microtime

hrf_library = get_hrf_library(mode="library", stim_duration=0.0, microtime_dt=microtime_dt, device=device)
print(f"HRF library: {hrf_library.shape}")

# Create canonical SPMG1 design
canonical_hrf = get_spmg1_hrf(microtime_dt=microtime_dt, stim_duration=0.0, normalize_peak=True, device=device)
canonical_design = convolve_hrf_microtime(
    onset_matrix, canonical_hrf, n_timepoints, tr=tr, microtime_dt=microtime_dt, device=device
)
print(f"Canonical design: {canonical_design.shape}")
print(f"  Max: {canonical_design.max():.4f}, Min: {canonical_design.min():.4f}")
print(f"  Sum abs: {canonical_design.abs().sum():.2f}")

# Make library design for HRF 0
lib_hrf_0 = hrf_library[0]
lib_design_0 = convolve_hrf_microtime(
    onset_matrix, lib_hrf_0, n_timepoints, tr=tr, microtime_dt=microtime_dt, device=device
)
print(f"Library HRF 0 design: {lib_design_0.shape}")
print(f"  Max: {lib_design_0.max():.4f}, Min: {lib_design_0.min():.4f}")

# Simulate data: true response is canonical HRF + noise
noise = torch.randn(n_voxels, n_timepoints) * 1.0

# Add polynomial drift per run
from fastfuncsim.glm_core import construct_polynomial_matrix
drift = torch.zeros(n_voxels, n_timepoints)
for run_idx in range(n_runs):
    start = run_starts[run_idx]
    end = start + n_tps_per_run
    t = torch.linspace(-1, 1, n_tps_per_run)
    drift[:, start:end] = 50 * t.unsqueeze(0)  # Linear drift

# True signal = canonical design * beta + drift + noise
true_beta = 5.0  # Strong signal for easy detection
signal = true_beta * canonical_design[:, 0].unsqueeze(0).expand(n_voxels, -1)
data = signal + drift + noise + 1000  # Add mean offset

print(f"\nData: {data.shape}")
print(f"  Mean: {data.mean():.2f}, Std: {data.std():.2f}")
print(f"  Signal range: [{signal.min():.2f}, {signal.max():.2f}]")

# ========== PATH 1: 3dDenoisefast approach ==========
# Project out nuisance from data and design, then xval
print("\n" + "="*70)
print("PATH 1: 3dDenoisefast approach (project-first, then xval)")
print("="*70)

from fastfuncsim.xval import project_out_nuisance_per_run, compute_xval_r2, generate_cv_splits

# Build nuisance blocks per run
nuisance_per_run = []
for run_idx in range(n_runs):
    run_len = n_tps_per_run
    poly = construct_polynomial_matrix(run_len, polort, device)
    nuisance_per_run.append(poly)
    if run_idx == 0:
        print(f"  Nuisance per run: {poly.shape}")

# Project out nuisance
proj_data_denoise, proj_design_denoise = project_out_nuisance_per_run(
    data=data,
    design=canonical_design,
    nuisance_per_run=nuisance_per_run,
    run_starts=run_starts,
    device=device,
)
print(f"  Projected data: {proj_data_denoise.shape}")
print(f"  Projected design: {proj_design_denoise.shape}")
print(f"  Projected design max: {proj_design_denoise.max():.4f}")

# CV splits
cv_splits = generate_cv_splits(n_runs, strategy=1, n_perms=100)

# Compute xval R²
xval_denoise = compute_xval_r2(
    data=proj_data_denoise,
    design_matrix=proj_design_denoise,
    run_starts=run_starts,
    stim_indices=list(range(n_conditions)),
    nuisance_indices=[],
    cv_splits=cv_splits,
    metric="cod",
    device=device,
    verbose=False,
)
print(f"\n  3dDenoisefast xval R²: mean={xval_denoise['r2'].mean():.4f}, "
      f"median={xval_denoise['r2'].median():.4f}")

# ========== PATH 2: 3dHRFoptfast approach ==========
print("\n" + "="*70)
print("PATH 2: 3dHRFoptfast approach (fit_glm_hrf_library_with_xval)")
print("="*70)

from fastfuncsim.hrf_selection import fit_glm_hrf_library_with_xval

results = fit_glm_hrf_library_with_xval(
    data=data,
    onsets=onset_matrix,
    hrf_library=hrf_library,
    tr=tr,
    run_starts=run_starts,
    stim_durations=[duration_s],
    cv_strategy=1,
    metric="cod",
    microtime_dt=microtime_dt,
    polort=polort,
    canonical_mode="spmg1",
    device=device,
    verbose=True,
)

print(f"\n  3dHRFoptfast xval R² (best): mean={results.xval_r2_best.mean():.4f}, "
      f"median={results.xval_r2_best.median():.4f}")
print(f"  3dHRFoptfast xval R² (canonical): mean={results.xval_r2_canonical.mean():.4f}, "
      f"median={results.xval_r2_canonical.median():.4f}")
print(f"  3dHRFoptfast final GLM R²: mean={results.final_results.r2.mean():.4f}, "
      f"median={results.final_results.r2.median():.4f}")
if results.canonical_results is not None:
    print(f"  3dHRFoptfast canonical GLM R²: mean={results.canonical_results.r2.mean():.4f}")

# ========== COMPARISON ==========
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
r2_denoise = xval_denoise['r2'].mean().item()
r2_hrfopt_canonical = results.xval_r2_canonical.mean().item()
r2_hrfopt_best = results.xval_r2_best.mean().item()
print(f"  3dDenoisefast xval R² (SPMG1):  {r2_denoise:.4f}")
print(f"  3dHRFoptfast canonical R² (SPMG1): {r2_hrfopt_canonical:.4f}")
print(f"  3dHRFoptfast best HRF R²:         {r2_hrfopt_best:.4f}")
diff = abs(r2_denoise - r2_hrfopt_canonical)
print(f"\n  Difference (should be ~0): {diff:.6f}")
if diff > 0.01:
    print("  *** BIG DISCREPANCY DETECTED! ***")
    print("  The canonical R² from hrf_selection should match 3dDenoisefast!")
else:
    print("  OK - canonical R² matches between tools")
