#!/usr/bin/env python
"""
Quick test to trace the all-zeros bug
"""

import torch

from fastfuncsim.design import build_glm_design, generate_random_onsets
from fastfuncsim.hrf import get_canonical_hrf
from fastfuncsim.simulation import simulate_fmri_run
from fastfuncsim.utils import get_device

# Setup
device = get_device()
print(f"Using device: {device}")

# Simulation parameters
n_timepoints = 200
tr = 2.0
n_conditions = 2
matrix_size = (20, 20, 5)

# Generate HRF
print("\n1. Generating HRF...")
hrf = get_canonical_hrf(stim_duration=2.0, tr=tr, device=device)
print(f"   HRF shape: {hrf.shape}")
print(
    f"   HRF min: {hrf.min().item():.6f}, max: {hrf.max().item():.6f}, sum: {hrf.sum().item():.6f}"
)
print(f"   HRF first 5 values: {hrf[:5]}")

# Generate onsets
print("\n2. Generating onsets...")
torch.manual_seed(42)
onsets = generate_random_onsets(
    n_timepoints=n_timepoints,
    n_conditions=n_conditions,
    isi_mean=6,
    tr=tr,
    device=device,
)
print(f"   Onsets shape: {onsets.shape}")
print(f"   Number of events per condition: {onsets.sum(dim=0)}")
print(f"   First 10 timepoints:\n{onsets[:10]}")

# Build design matrix
print("\n3. Building design matrix...")
design = build_glm_design(onsets, hrf, n_timepoints, mode="assumed", device=device)
print(f"   Design shape: {design.shape}")
print(f"   Design min: {design.min().item():.6f}, max: {design.max().item():.6f}")
print(f"   Design mean: {design.mean().item():.6f}, std: {design.std().item():.6f}")

# Simulate
print("\n4. Simulating fMRI run...")
betas = [3.0, 4.0]
data = simulate_fmri_run(
    onsets=onsets,
    betas=betas,
    hrf=hrf,
    tr=tr,
    n_timepoints=n_timepoints,
    matrix_size=matrix_size,
    noise_level=1.0,
    baseline=100.0,
    add_scanner_drift=True,
    device=device,
)

print(f"   Data shape: {data.shape}")
print(f"   Data min: {data.min().item():.3f}, max: {data.max().item():.3f}")
print(f"   Data mean: {data.mean().item():.3f}, std: {data.std().item():.3f}")

# Check a single voxel timecourse
voxel_timecourse = data[10, 10, 2, :]
print("\n5. Example voxel [10,10,2] timecourse:")
print(
    f"   Min: {voxel_timecourse.min().item():.3f}, Max: {voxel_timecourse.max().item():.3f}"
)
print(f"   First 10 timepoints: {voxel_timecourse[:10]}")

# Check if it's all zeros
if data.max().item() == 0 and data.min().item() == 0:
    print("\n⚠️ ERROR: Data is all zeros!")
else:
    print("\n✓ Data looks good (not all zeros)")
