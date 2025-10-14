#!/usr/bin/env python
"""
Fast test case: Analyze SINGLE run with ARMA(1,1)

Use this for rapid iteration while optimizing the ARMA GLM code.
Loads only 1 run (~150 timepoints) instead of 4 runs (~600 timepoints).

Expected timing:
- Full 4 runs: ~5-10 minutes
- Single run: ~1-2 minutes (5x faster testing!)
"""

import time
from pathlib import Path

import numpy as np
import torch

import fastfuncsim as ffs

# Setup
device = ffs.get_device()
print(f"Using device: {device}\n")

# Data paths
data_dir = Path(
    "/Users/logan/local_data/MindsEye_Pilot_Project/derivatives/models/"
    "preproc_sub-ME486585_ses-03WB_imagery.results"
)

# SINGLE RUN ONLY (for fast testing)
run_files = [
    data_dir / "run01.nii.gz",  # Just run 1!
]

design_matrix_file = data_dir / "X.xmat.1D"
mask_file = data_dir / "mask.nii.gz"

print("=" * 70)
print("FAST TEST: Single Run ARMA(1,1) Analysis")
print("=" * 70)
print(f"\nRun: {run_files[0].name}")
print(f"Design matrix: {design_matrix_file.name}")
print("(Will extract run 1 design from full 4-run matrix)\n")

# =============================================================================
# Load data and design manually for single-run extraction
# =============================================================================

print("Loading data and design...")

# Load run 1 data
import nibabel as nib
img = nib.load(run_files[0])
data = img.get_fdata(dtype=np.float32)
affine = img.affine
original_shape = data.shape[:3]
n_timepoints_run1 = data.shape[-1]

print(f"  Data shape: {data.shape}")
print(f"  Timepoints in run 1: {n_timepoints_run1}")

# Reshape to voxels
data = data.reshape(-1, n_timepoints_run1)  # (n_voxels, n_time)

# Load and apply mask
if mask_file.exists():
    mask = ffs.load_afni_mask(mask_file, threshold=0.5)
    mask_flat = mask.reshape(-1)
    data = data[mask_flat]
    print(f"  Masked to {data.shape[0]:,} voxels")
else:
    mask = None
    print("  No mask found, using all voxels")

# Load full design matrix
design_full = np.loadtxt(design_matrix_file)
print(f"  Full design shape: {design_full.shape}")

# Extract run 1 design (assumes AFNI format with RunStart markers)
# Parse design matrix to find run boundaries
with open(design_matrix_file, 'r') as f:
    lines = f.readlines()

# Find RunStart line
run_starts = []
for line in lines:
    if 'RunStart' in line:
        # Parse: # RunStart = 0 151 302 453 or "0,300,600,900"
        starts_str = line.split('=')[1].strip()
        # Remove quotes if present
        starts_str = starts_str.strip('"\'')
        # Split by either comma or space
        if ',' in starts_str:
            run_starts = [int(x.strip()) for x in starts_str.split(',')]
        else:
            run_starts = [int(x) for x in starts_str.split()]
        break

if not run_starts:
    raise ValueError("Could not find RunStart in design matrix")

print(f"  RunStart indices: {run_starts}")

# Extract run 1 design (from run_starts[0] to run_starts[1])
start_idx = run_starts[0]
end_idx = run_starts[1]
design = design_full[start_idx:end_idx]

print(f"  Run 1 design shape: {design.shape}")
print(f"  Run 1 timepoints: {design.shape[0]}")

# Verify match
if design.shape[0] != n_timepoints_run1:
    raise ValueError(
        f"Design timepoints ({design.shape[0]}) != data timepoints ({n_timepoints_run1})"
    )

print("✓ Data and design loaded\n")

# =============================================================================
# Run ARMA(1,1) GLM
# =============================================================================

print("=" * 70)
print("Running ARMA(1,1) GLM (single run, fast!)")
print("=" * 70)

# Convert to tensors
data_tensor = torch.from_numpy(data).float()
design_tensor = torch.from_numpy(design).float()

# TR from design matrix
tr = 2.0  # MindsEye TR

start_time = time.time()

results = ffs.fit_glm_arma11(
    data_tensor,
    design_tensor,
    tr=tr,
    estimate_per_voxel=True,
    batch_size=10000,  # Adjust based on GPU memory
    want_residuals=False,
    want_predicted=False,
    device=device,
    verbose=True,
)

elapsed = time.time() - start_time

print(f"\n✓ Analysis complete in {elapsed:.1f} seconds!")
print(f"  Mean R²: {results.r2.mean():.3f}")
print(f"  Mean ARMA a: {results.arma_params[:, 0].mean():.3f}")
print(f"  Mean ARMA b: {results.arma_params[:, 1].mean():.3f}")
print(f"  Mean λ: {results.arma_lambda.mean():.3f}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
✓ Fast single-run test complete!

Data:
  - 1 run: {n_timepoints_run1} timepoints
  - {data.shape[0]:,} voxels (masked)
  - {design.shape[1]} regressors
  - TR = {tr}s

Method:
  - ARMA(1,1) prewhitened GLS
  - GPU: {device.type.upper()}

Performance:
  - Time: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)
  - Throughput: {data.shape[0] / elapsed:.0f} voxels/sec

Results:
  - Mean R²: {results.r2.mean():.3f}
  - Mean ARMA(a,b): ({results.arma_params[:, 0].mean():.3f}, {results.arma_params[:, 1].mean():.3f})
  - Mean λ (lag-1): {results.arma_lambda.mean():.3f}

Use this test case for rapid iteration while optimizing!

To test optimizations:
1. Edit arma_glm.py
2. Run: python examples/analyze_real_data_single_run.py
3. Compare timing to baseline ({elapsed:.1f}s)
4. Verify results haven't changed (R², ARMA params)

Target: Get this under 30 seconds with optimizations!
""")
