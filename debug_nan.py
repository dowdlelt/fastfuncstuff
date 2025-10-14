#!/usr/bin/env python
"""Debug NaN issue in simulation"""

import fastfuncsim as ffs
import torch

# Setup
device = ffs.get_device()
print(f"Using device: {device}\n")

# Parameters
n_timepoints = 200
tr = 2.0
n_conditions = 2
matrix_size = (20, 20, 5)

# 1. Generate HRF
print("1. Generating HRF...")
hrf = ffs.get_canonical_hrf(stim_duration=2.0, tr=tr, device=device)
print(f"   Shape: {hrf.shape}")
print(f"   Has NaN: {torch.isnan(hrf).any()}")
print(f"   Has Inf: {torch.isinf(hrf).any()}")
print(f"   Min/Max: {hrf.min():.6f} / {hrf.max():.6f}")
print(f"   First 5 values: {hrf[:5]}\n")

# 2. Generate onsets
print("2. Generating onsets...")
torch.manual_seed(42)
onsets = ffs.generate_random_onsets(
    n_timepoints=n_timepoints,
    n_conditions=n_conditions,
    isi_mean=6,
    tr=tr,
    device=device
)
print(f"   Shape: {onsets.shape}")
print(f"   Has NaN: {torch.isnan(onsets).any()}")
print(f"   Events per condition: {onsets.sum(dim=0)}\n")

# 3. Build design
print("3. Building design matrix...")
design = ffs.build_glm_design(onsets, hrf, n_timepoints, mode='assumed', device=device)
print(f"   Shape: {design.shape}")
print(f"   Has NaN: {torch.isnan(design).any()}")
print(f"   Has Inf: {torch.isinf(design).any()}")
print(f"   Min/Max: {design.min():.6f} / {design.max():.6f}\n")

# 4. Create signal
print("4. Creating signal...")
betas = torch.tensor([3.0, 4.0], device=device)
n_voxels = matrix_size[0] * matrix_size[1] * matrix_size[2]
betas_expanded = betas.unsqueeze(0).expand(n_voxels, n_conditions)
signal = design @ betas_expanded.T  # (n_timepoints, n_voxels)
signal = signal.T  # (n_voxels, n_timepoints)
print(f"   Signal shape: {signal.shape}")
print(f"   Has NaN: {torch.isnan(signal).any()}")
print(f"   Min/Max: {signal.min():.6f} / {signal.max():.6f}\n")

# 5. Add baseline
print("5. Adding baseline...")
baseline = 100.0
data = baseline + signal
print(f"   Data shape: {data.shape}")
print(f"   Has NaN: {torch.isnan(data).any()}")
print(f"   Min/Max: {data.min():.3f} / {data.max():.3f}\n")

# 6. Generate noise for one slice
print("6. Generating noise...")
nx, ny, nz = matrix_size
try:
    slice_noise = ffs.generate_fmri_noise(
        tr, n_timepoints * tr,
        matrix_size=(nx, ny),
        normalize=True,
        device=device
    )
    print(f"   Noise shape: {slice_noise.shape}")
    print(f"   Has NaN: {torch.isnan(slice_noise).any()}")
    print(f"   Has Inf: {torch.isinf(slice_noise).any()}")
    print(f"   Min/Max: {slice_noise.min():.6f} / {slice_noise.max():.6f}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
