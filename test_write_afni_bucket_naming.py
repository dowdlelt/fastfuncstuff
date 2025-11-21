#!/usr/bin/env python3
"""
Test write_afni_bucket output naming
"""
from pathlib import Path
from fastfuncsim.glm_outputs import _normalize_output_path, _strip_imaging_extension

# Simulate what write_afni_bucket does when output_format="nifti_gz" (always now)

test_cases = [
    # (user_input, output_format, expected_final_output)
    ("stats", "nifti_gz", "stats.nii.gz"),
    ("stats.nii", "nifti_gz", "stats.nii.gz"),
    ("stats.nii.gz", "nifti_gz", "stats.nii.gz"),
    ("stats+orig.HEAD", "nifti_gz", "stats.nii.gz"),
    ("stats.blur.2mm", "nifti_gz", "stats.blur.2mm.nii.gz"),
]

print("=" * 70)
print("Testing write_afni_bucket naming logic")
print("=" * 70)
print()

all_passed = True
for user_input, output_format_arg, expected in test_cases:
    # Simulate write_afni_bucket logic (lines 744-756)
    output_path = Path(user_input)
    base_path, detected_format = _normalize_output_path(output_path)

    # Line 746-747: if output_format is not None, override
    if output_format_arg is not None:
        detected_format = output_format_arg

    # Lines 752-756: Convert AFNI to NIfTI
    if detected_format == "afni":
        detected_format = "nifti_gz"
        base_name = _strip_imaging_extension(str(base_path))
        base_path = Path(base_name + ".nii.gz")

    # Lines 760-765: Create temp_path for uncompressed write
    if str(base_path).endswith(".nii.gz"):
        temp_path = base_path.parent / (base_path.name[:-7] + ".nii")
    elif str(base_path).endswith(".nii"):
        temp_path = base_path
    else:
        temp_path = base_path.with_suffix(".nii")

    # Lines 857-858: Compress
    base_name_compress = _strip_imaging_extension(str(temp_path))
    final_path = Path(base_name_compress + ".nii.gz")

    result = str(final_path)
    status = "✅ PASS" if result == expected else "❌ FAIL"

    if result != expected:
        all_passed = False
        print(f"{status:8s}  {user_input:30s} → {result:30s}")
        print(f"         {'':30s}    (expected: {expected})")
        print(f"         Intermediate: base_path={base_path}, temp_path={temp_path}")
    else:
        print(f"{status:8s}  {user_input:30s} → {result:30s}")

print()
if all_passed:
    print("=" * 70)
    print("✅ ALL TESTS PASSED - No .nii.nii.gz files!")
    print("=" * 70)
else:
    print("=" * 70)
    print("❌ SOME TESTS FAILED!")
    print("=" * 70)
