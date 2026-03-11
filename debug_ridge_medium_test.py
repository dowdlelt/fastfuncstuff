"""Debug test for Medium test failure - trace data flow"""

import numpy as np
import torch

from fastfuncsim.glm_core import construct_polynomial_matrix
from fastfuncsim.hrf import get_canonical_hrf
from fastfuncsim.ridge import create_single_trial_design, fit_ridge_single_trial
from fastfuncsim.simulation import simulate_fmri_run
from fastfuncsim.utils import get_device
from fastfuncsim.xval import generate_cv_splits

device = get_device()

# Reproduce exact test setup
tr = 2.0
n_timepoints = 120
n_runs = 2
matrix_size = (8, 8, 4)

print("="*70)
print("STEP 1: Create onsets")
print("="*70)

onsets_by_condition = []
for cond_idx in range(2):
    cond_onsets = []
    for run_idx in range(n_runs):
        events = np.sort(np.random.choice(n_timepoints, size=6, replace=False))
        cond_onsets.append(events * tr)
    onsets_by_condition.append(cond_onsets)

durations = [0.0, 0.0]
run_starts = [0, n_timepoints]

print(f"run_starts: {run_starts}")
print(f"n_timepoints: {n_timepoints}, n_runs: {n_runs}")
print(f"Total timepoints: {n_timepoints * n_runs}")

print("\n" + "="*70)
print("STEP 2: Create single-trial design")
print("="*70)

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
print(f"trial_condition_ids.shape: {trial_condition_ids.shape}")
print(f"n_trials (design_matrix.shape[1]): {design_matrix.shape[1]}")
print(f"n_conditions (condition_design.shape[1]): {condition_design.shape[1]}")

print("\n" + "="*70)
print("STEP 3: Simulate data")
print("="*70)

n_voxels = 8 * 8 * 4
true_betas = torch.tensor([3.0, 5.0], device=device)

# Create condition-wise onsets for simulation
onsets_list = []
for run_idx in range(n_runs):
    onsets = torch.zeros(n_timepoints, 2, device=device)
    for cond_idx in range(2):
        events = torch.randperm(n_timepoints, device=device)[:6]
        onsets[events, cond_idx] = 1.0
    onsets_list.append(onsets)

hrf = get_canonical_hrf(stim_duration=0.0, tr=tr, duration=30.0, device=device)

data_list = []
for run_idx, onsets in enumerate(onsets_list):
    data = simulate_fmri_run(
        onsets=onsets,
        betas=true_betas.tolist(),
        hrf=hrf,
        tr=tr,
        n_timepoints=n_timepoints,
        matrix_size=matrix_size,
        noise_level=1.0,
        baseline=100.0,
        device=device
    )
    data_list.append(data.reshape(-1, n_timepoints))

data_all = torch.cat(data_list, dim=1)  # Concatenate along time dimension
print(f"data_all.shape: {data_all.shape}")

print("\n" + "="*70)
print("STEP 4: Build nuisance (polynomials)")
print("="*70)

poly = construct_polynomial_matrix(n_timepoints * n_runs, max_degree=2, device=device)
print(f"poly.shape: {poly.shape}")

poly_per_run = []
for i in range(n_runs):
    poly_run = torch.zeros(n_timepoints, poly.shape[1], device=device)
    start = i * n_timepoints
    end = start + n_timepoints
    poly_run[:, :] = poly[start:end, :]
    poly_per_run.append(poly_run)
    print(f"poly_per_run[{i}].shape: {poly_per_run[i].shape}")

print("\n" + "="*70)
print("STEP 5: Create CV splits")
print("="*70)

cv_splits = generate_cv_splits(n_runs=n_runs, strategy=1, n_perms=1)
print(f"cv_splits: {cv_splits}")

print("\n" + "="*70)
print("STEP 6: Call fit_ridge_single_trial")
print("="*70)

fracs = np.array([0.0, 0.3, 0.7, 1.0])

# Add some debug output to understand the flow
import sys
from io import StringIO

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

try:
    results = fit_ridge_single_trial(
        data=data_all,
        design_matrix=design_matrix,
        run_starts=run_starts,
        tr=tr,
        trial_condition_ids=trial_condition_ids,
        condition_design=condition_design,
        fracs=fracs,
        nuisance=poly_per_run,
        polort=None,  # Already provided nuisance
        cv_splits=cv_splits,
        autoscale=True,
        device=device,
        verbose=True,  # Enable verbose to see what's happening
    )
finally:
    sys.stdout = old_stdout
    output = mystdout.getvalue()
    print(output)
    print("SUCCESS! Test passed.")
