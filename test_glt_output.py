"""Test that GLTs are written to bucket file."""
import torch
import nibabel as nib
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

print("Running OLS analysis with GLTs...")
results, design_info = analyze_from_design_matrix(
    fmri_data=INPUT_FILES,
    design_matrix_file=DESIGN_MATRIX,
    method="ols",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

print(f"\nResults attributes:")
print(f"  Has contrast_betas: {hasattr(results, 'contrast_betas')}")
print(f"  Has contrast_tstats: {hasattr(results, 'contrast_tstats')}")
print(f"  Has contrast_labels: {hasattr(results, 'contrast_labels')}")

if hasattr(results, "contrast_betas"):
    print(f"  contrast_betas shape: {results.contrast_betas.shape}")
    print(f"  contrast_tstats shape: {results.contrast_tstats.shape}")
    print(f"  contrast_labels: {results.contrast_labels}")

# Get labels from design info
stim_labels = design_info.get("stim_labels", [])
glt_labels = design_info.get("glt_labels", [])

print(f"\nDesign info:")
print(f"  stim_labels: {stim_labels}")
print(f"  glt_labels: {glt_labels}")

# Write bucket file
output_path = Path("/tmp/test_glt_bucket.nii.gz")
write_afni_bucket(
    results,
    output_path,
    condition_names=stim_labels,
    # Don't pass contrast_names - let it use results.contrast_labels
    contrast_results=None,  # Should use results.contrast_betas/tstats
    output_format="nifti_gz",
    apply_afni_metadata=False,  # Skip AFNI metadata for this test
)

# Load and check the bucket file
bucket = nib.load(output_path)
data = bucket.get_fdata()

print(f"\nBucket file:")
print(f"  Shape: {data.shape}")
print(f"  Expected sub-briks: 1 (Full_Fstat) + 2*{len(stim_labels)} (stims) + 2*{len(glt_labels)} (GLTs) = {1 + 2*len(stim_labels) + 2*len(glt_labels)}")
print(f"  Actual sub-briks: {data.shape[-1]}")

# Check sub-brik order
expected_order = ["Full_Fstat"]
for label in stim_labels:
    expected_order.append(f"{label}#0_Coef")
    expected_order.append(f"{label}#0_Tstat")
if hasattr(results, "contrast_labels"):
    for label in results.contrast_labels:
        expected_order.append(f"{label}#0_Coef")
        expected_order.append(f"{label}#0_Tstat")

print(f"\nExpected sub-brik order:")
for i, label in enumerate(expected_order):
    print(f"  [{i}] {label}")

if data.shape[-1] == len(expected_order):
    print(f"\n✓ SUCCESS: Bucket file has correct number of sub-briks including GLTs!")
else:
    print(f"\n✗ FAIL: Expected {len(expected_order)} sub-briks, got {data.shape[-1]}")

# Clean up
output_path.unlink()
print(f"\nCleaned up test file: {output_path}")
