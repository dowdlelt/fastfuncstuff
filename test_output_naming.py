#!/usr/bin/env python3
"""
Test that output file naming works correctly, avoiding .nii.nii.gz
"""
import sys
sys.path.insert(0, '/home/logan/Dropbox/Resources/code/fastfuncsim/bin')

from pathlib import Path

# Import the function from 3dREMLfast
exec(open('/home/logan/Dropbox/Resources/code/fastfuncsim/bin/3dREMLfast.py').read().split('def replace_afni_extension')[0])
exec('def replace_afni_extension' + open('/home/logan/Dropbox/Resources/code/fastfuncsim/bin/3dREMLfast.py').read().split('def replace_afni_extension')[1].split('\n\ndef ')[0])

# Test cases
test_cases = [
    ("stats", "stats.nii.gz"),
    ("stats.nii", "stats.nii.gz"),
    ("stats.nii.gz", "stats.nii.gz"),
    ("stats+orig.HEAD", "stats.nii.gz"),
    ("stats.blur.2mm", "stats.blur.2mm.nii.gz"),
    ("stats.blur.2mm.nii", "stats.blur.2mm.nii.gz"),
    ("stats.blur.2mm.nii.gz", "stats.blur.2mm.nii.gz"),
    ("errts.sub-01+orig.BRIK", "errts.sub-01.nii.gz"),
]

print("=" * 70)
print("Testing ensure_nifti_gz_extension() function")
print("=" * 70)
print()

all_passed = True
for input_path, expected in test_cases:
    result = replace_afni_extension(input_path, '.nii.gz')
    status = "✅ PASS" if result == expected else "❌ FAIL"

    if result != expected:
        all_passed = False

    print(f"{status:8s}  {input_path:30s} → {result:30s}  (expected: {expected})")

print()
if all_passed:
    print("=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
else:
    print("=" * 70)
    print("❌ SOME TESTS FAILED!")
    print("=" * 70)
    sys.exit(1)
