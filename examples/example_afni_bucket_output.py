#!/usr/bin/env python
"""
Example: Complete ARMA(1,1) analysis with AFNI-compatible bucket output

Demonstrates:
1. Analyzing data with ARMA(1,1) prewhitening
2. Computing custom contrasts
3. Writing AFNI-style bucket file with:
   - Overall F-statistic
   - Beta and t-stat for each condition
   - Beta and t-stat for each contrast
"""

import torch

import fastfuncstuff as ffs

# Setup
device = ffs.get_device()
print(f"Using device: {device}\n")

# =============================================================================
# Step 1: Simulate data with multiple conditions
# =============================================================================
print("=" * 70)
print("Step 1: Simulating multi-condition experiment")
print("=" * 70)

tr = 2.0
n_timepoints = 200
n_conditions = 4  # Let's say: faces, places, objects, scrambled
matrix_size = (20, 20, 5)

# Generate onsets
onsets = ffs.generate_random_onsets(
    n_timepoints=n_timepoints,
    n_conditions=n_conditions,
    isi_mean=6,
    tr=tr,
    device=device
)

# Simulate with different betas for each condition
hrf = ffs.get_canonical_hrf(stim_duration=2.0, tr=tr, device=device)
betas_true = [3.0, 4.0, 2.5, 1.5]  # faces, places, objects, scrambled

data = ffs.simulate_fmri_run(
    onsets=onsets,
    betas=betas_true,
    hrf=hrf,
    tr=tr,
    n_timepoints=n_timepoints,
    matrix_size=matrix_size,
    noise_level=1.0,
    baseline=100.0,
    device=device
)

print(f"Simulated data shape: {data.shape}")
print(f"True betas: {betas_true}")
print(f"Data range: [{data.min():.1f}, {data.max():.1f}]\n")

# =============================================================================
# Step 2: Fit GLM with ARMA(1,1) prewhitening
# =============================================================================
print("=" * 70)
print("Step 2: Fitting GLM with ARMA(1,1) prewhitening")
print("=" * 70)

# Build design matrix
design = ffs.build_glm_design(onsets, hrf, n_timepoints, mode='assumed', device=device)

# Reshape data
data_reshaped = data.reshape(-1, n_timepoints)

# Fit ARMA(1,1) GLM
a_grid = torch.linspace(-0.9, 0.9, 19, device=device)
b_grid = torch.linspace(-0.9, 0.9, 19, device=device)

results = ffs.fit_glm_arma11(
    data_reshaped,
    design,
    tr=tr,
    a_grid=a_grid,
    b_grid=b_grid,
    device=device,
    verbose=False
)

# Store original shape for output writing
results.original_shape = matrix_size

print(f"Mean R²: {results.r2.mean():.3f}")
print("Estimated betas:")
for i in range(n_conditions):
    print(f"  Condition {i+1}: {results.betas[:, i].mean():.3f} (true: {betas_true[i]:.1f})")
print(f"Mean ARMA a: {results.arma_params[:, 0].mean():.3f}")
print(f"Mean ARMA b: {results.arma_params[:, 1].mean():.3f}\n")

# =============================================================================
# Step 3: Compute contrasts
# =============================================================================
print("=" * 70)
print("Step 3: Computing contrasts")
print("=" * 70)

# Define contrasts
# faces vs places
# objects vs scrambled
# all stimuli vs baseline (not possible with this design, need baseline regressor)

contrasts = torch.tensor([
    [1, -1, 0, 0],   # faces > places
    [0, 0, 1, -1],   # objects > scrambled
    [1, 1, -1, -1],  # (faces+places) > (objects+scrambled)
], device=device, dtype=torch.float32)

contrast_results = ffs.compute_contrasts(results, contrasts, device=device)

print("Contrast results:")
print(f"  Faces > Places: mean t-stat = {contrast_results['contrast_tstats'][:, 0].mean():.3f}")
print(f"  Objects > Scrambled: mean t-stat = {contrast_results['contrast_tstats'][:, 1].mean():.3f}")
print(f"  (F+P) > (O+S): mean t-stat = {contrast_results['contrast_tstats'][:, 2].mean():.3f}\n")

# =============================================================================
# Step 4: Write AFNI-style bucket file
# =============================================================================
print("=" * 70)
print("Step 4: Writing AFNI-style bucket file")
print("=" * 70)

condition_names = ['faces', 'places', 'objects', 'scrambled']
contrast_names = ['faces_vs_places', 'objects_vs_scrambled', 'category_vs_control']

output_path = ffs.write_glm_bucket_as_nifti(
    results,
    'glm_bucket.nii.gz',
    condition_names=condition_names,
    contrast_names=contrast_names,
    contrast_results=contrast_results,
)

print(f"✓ Wrote AFNI bucket to: {output_path}")
print(f"✓ Wrote sub-brick labels to: {output_path.with_suffix('.json')}\n")

# =============================================================================
# Show what's in the bucket
# =============================================================================
print("=" * 70)
print("Bucket contents (sub-brick order):")
print("=" * 70)

# Read the JSON to show what's in the bucket
import json

with open(output_path.with_suffix('.json')) as f:
    bucket_info = json.load(f)

for idx, label in enumerate(bucket_info['SubBricks']):
    print(f"  [{idx:2d}] {label}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
AFNI-compatible bucket file created with:
✓ Overall F-statistic (tests all conditions jointly)
✓ Beta + T-stat for each condition (4 conditions = 8 sub-bricks)
✓ Beta + T-stat for each contrast (3 contrasts = 6 sub-bricks)
✓ Total: 1 + 8 + 6 = 15 sub-bricks

This matches AFNI's 3dDeconvolve output format!

You can view it with AFNI:
  afni -niml &
  # Open glm_bucket.nii.gz
  # Use the "Define Overlay" panel to select sub-bricks

Or with Python/nibabel:
  import nibabel as nib
  img = nib.load('glm_bucket.nii.gz')
  data = img.get_fdata()
  print(f"Shape: {data.shape}")  # (20, 20, 5, 15)

The JSON sidecar provides human-readable sub-brick labels.
""")

print("\n✓ Complete AFNI workflow demonstrated!")
