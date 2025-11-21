"""Debug script to find ARMA outliers."""
import numpy as np
import nibabel as nib
import torch
from pathlib import Path

from fastfuncsim.analysis import analyze_from_design_matrix
from fastfuncsim.glm_outputs import write_afni_bucket

# AFNI reference data paths
AFNI_DATA_DIR = Path.home() / "Dropbox/Data/small_validation_afni_data"
DESIGN_MATRIX = AFNI_DATA_DIR / "X.xmat.1D"
INPUT_FILES = [
    AFNI_DATA_DIR / "small_test_r01.nii.gz",
    AFNI_DATA_DIR / "small_test_r02.nii.gz",
]
AFNI_REML_BUCKET = AFNI_DATA_DIR / "afni_REML.nii.gz"
AFNI_RVAR = AFNI_DATA_DIR / "afni_REMLvar.nii.gz"

print("Running ARMA(1,1) analysis...")
results, design_info = analyze_from_design_matrix(
    fmri_data=INPUT_FILES,
    design_matrix_file=DESIGN_MATRIX,
    method="arma11",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# Load AFNI outputs
afni_reml = nib.load(AFNI_REML_BUCKET)
afni_rvar = nib.load(AFNI_RVAR)

# Get F-stats
afni_fstat = afni_reml.get_fdata()[..., 0, 0].ravel()
our_fstat = results.fstats.reshape(10, 10, 10).ravel()

# Get ARMA parameters
afni_a = afni_rvar.get_fdata()[..., 0, 0].ravel()
afni_b = afni_rvar.get_fdata()[..., 0, 1].ravel()
our_a = results.arma_params[:, 0].numpy()
our_b = results.arma_params[:, 1].numpy()

# Calculate differences
fstat_diff = np.abs(our_fstat - afni_fstat)
fstat_rel_diff = fstat_diff / (np.abs(afni_fstat) + 1e-10)

# Filter out NaNs and Infs
valid_mask = np.isfinite(fstat_rel_diff)
print(f"Valid voxels: {valid_mask.sum()}/1000")

# Find worst outliers (sort ascending, take last 10, reverse to descending)
valid_indices = np.where(valid_mask)[0]
valid_rel_diff = fstat_rel_diff[valid_mask]
sorted_idx = np.argsort(valid_rel_diff)
worst_valid_idx = sorted_idx[-10:]
worst_indices = valid_indices[worst_valid_idx][::-1]

print("\n" + "="*80)
print("TOP 10 WORST F-STAT OUTLIERS")
print("="*80)
print(f"{'Idx':<6} {'AFNI_F':<10} {'Our_F':<10} {'AbsDiff':<10} {'RelDiff':<10} {'AFNI(a,b)':<15} {'Our(a,b)'}")
print("-"*80)

for idx in worst_indices:
    i, j, k = np.unravel_index(idx, (10, 10, 10))
    print(f"{idx:<6} {afni_fstat[idx]:<10.4f} {our_fstat[idx]:<10.4f} "
          f"{fstat_diff[idx]:<10.4f} {fstat_rel_diff[idx]:<10.2%} "
          f"({afni_a[idx]:.1f},{afni_b[idx]:.1f}){'':<5} "
          f"({our_a[idx]:.1f},{our_b[idx]:.1f})")

# Summary stats
print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)

# Count parameter mismatches
param_mismatch = (np.abs(our_a - afni_a) > 0.05) | (np.abs(our_b - afni_b) > 0.05)
n_mismatch = param_mismatch.sum()
print(f"Voxels with parameter mismatch (>0.05): {n_mismatch}/1000 ({n_mismatch/10:.1f}%)")

# Check if outliers have parameter mismatches
outliers_with_mismatch = param_mismatch[worst_indices].sum()
print(f"Outliers with parameter mismatch: {outliers_with_mismatch}/10")

# F-stat correlation by parameter match
match_mask = ~param_mismatch
if match_mask.sum() > 0:
    corr_match = np.corrcoef(afni_fstat[match_mask], our_fstat[match_mask])[0, 1]
    print(f"\nF-stat correlation (matching params): {corr_match:.6f}")
if param_mismatch.sum() > 0:
    corr_mismatch = np.corrcoef(afni_fstat[param_mismatch], our_fstat[param_mismatch])[0, 1]
    print(f"F-stat correlation (mismatched params): {corr_mismatch:.6f}")

# Distribution of ARMA parameters
print("\n" + "="*80)
print("ARMA PARAMETER DISTRIBUTIONS")
print("="*80)
print("AFNI:")
print(f"  a: mean={afni_a.mean():.3f}, median={np.median(afni_a):.3f}, range=[{afni_a.min():.1f}, {afni_a.max():.1f}]")
print(f"  b: mean={afni_b.mean():.3f}, median={np.median(afni_b):.3f}, range=[{afni_b.min():.1f}, {afni_b.max():.1f}]")
print("\nOurs:")
print(f"  a: mean={our_a.mean():.3f}, median={np.median(our_a):.3f}, range=[{our_a.min():.1f}, {our_a.max():.1f}]")
print(f"  b: mean={our_b.mean():.3f}, median={np.median(our_b):.3f}, range=[{our_b.min():.1f}, {our_b.max():.1f}]")

# Show unique (a,b) pairs
afni_unique = set(zip(afni_a.round(1), afni_b.round(1)))
our_unique = set(zip(our_a.round(1), our_b.round(1)))
print(f"\nUnique (a,b) pairs (rounded to 0.1):")
print(f"  AFNI: {len(afni_unique)} pairs")
print(f"  Ours: {len(our_unique)} pairs")
print(f"  Intersection: {len(afni_unique & our_unique)} pairs")
print(f"  AFNI only: {len(afni_unique - our_unique)} pairs")
print(f"  Ours only: {len(our_unique - afni_unique)} pairs")
