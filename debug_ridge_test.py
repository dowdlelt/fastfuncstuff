"""Debug test for ridge prediction accumulation bug"""

import torch
import numpy as np
from fastfuncsim.utils import get_device
from fastfuncsim.ridge import create_single_trial_design
from fastfuncsim.glm_core import construct_polynomial_matrix, fit_glm
from fastfuncsim.hrf import get_canonical_hrf
from fastfuncsim.simulation import simulate_fmri_run
from fastfuncsim.xval import generate_cv_splits, project_out_nuisance_per_run

device = get_device()
tr = 2.0
n_timepoints = 120
n_runs = 2

# Create minimal data
onsets_by_condition = []
for cond_idx in range(2):
    cond_onsets = []
    for run_idx in range(n_runs):
        events = np.array([10, 30, 50, 70, 90, 110])  # Fixed events
        cond_onsets.append(events * tr)
    onsets_by_condition.append(cond_onsets)

durations = [0.0, 0.0]
run_starts = [0, n_timepoints]

print("Creating design...")
design_matrix, trial_labels, trial_condition_ids, trial_run_ids, condition_design = \
    create_single_trial_design(
        onsets_by_condition=onsets_by_condition,
        durations=durations,
        run_starts=run_starts,
        tr=tr,
        n_timepoints=n_timepoints * n_runs,
        device=device,
    )

print(f"design_matrix.shape: {design_matrix.shape}")
print(f"condition_design.shape: {condition_design.shape}")
print(f"trial_condition_ids: {trial_condition_ids}")

# Create simple data
print("\nCreating data...")
n_voxels = 10
data_all = torch.randn(n_voxels, n_timepoints * n_runs, device=device) * 10 + 100

# Build polynomials
poly = construct_polynomial_matrix(n_timepoints * n_runs, max_degree=2, device=device)
poly_per_run = []
for i in range(n_runs):
    poly_run = torch.zeros(n_timepoints, poly.shape[1], device=device)
    start = i * n_timepoints
    end = start + n_timepoints
    poly_run[:, :] = poly[start:end, :]
    poly_per_run.append(poly_run)

print(f"poly_per_run[0].shape: {poly_per_run[0].shape}")
print(f"poly_per_run[1].shape: {poly_per_run[1].shape}")

# Create CV splits
print("\nCreating CV splits...")
cv_splits = generate_cv_splits(n_runs=n_runs, strategy=1, n_perms=1)
print(f"cv_splits: {cv_splits}")

# Test projection on first fold
print("\nTesting projection...")
train_runs, test_runs = cv_splits[0]
print(f"train_runs: {train_runs}, test_runs: {test_runs}")

# Prepare for projection
train_tps = []
train_run_starts_local = [0]
for run_idx in train_runs:
    start_tp = run_starts[run_idx]
    run_length = n_timepoints  # Fixed for this simple test
    train_tps.extend(range(start_tp, start_tp + run_length))
    train_run_starts_local.append(len(train_tps))
train_run_starts_local = train_run_starts_local[:-1]

print(f"train_tps (first 10): {train_tps[:10]}")
print(f"train_run_starts_local: {train_run_starts_local}")

train_data = data_all[:, train_tps]
train_design_raw = design_matrix[train_tps, :]
train_nuisance = [poly_per_run[i] for i in train_runs]

print(f"train_data.shape: {train_data.shape}")
print(f"train_design_raw.shape: {train_design_raw.shape}")
print(f"train_nuisance[0].shape: {train_nuisance[0].shape}")

# Try projection
print("\nCalling project_out_nuisance_per_run...")
try:
    train_data_clean, train_design_clean = project_out_nuisance_per_run(
        train_data,
        train_design_raw,
        train_nuisance,
        train_run_starts_local,
        device=device,
    )
    print(f"SUCCESS!")
    print(f"train_data_clean.shape: {train_data_clean.shape}")
    print(f"train_design_clean.shape: {train_design_clean.shape}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
