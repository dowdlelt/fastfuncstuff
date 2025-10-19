#!/usr/bin/env python3
"""
Diagnostic script to compare OLS results between AFNI and fastfuncsim

Usage:
    python debug_ols_diff.py <afni_bucket> <fastfuncsim_bucket>
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path


def load_subbrick(img, idx):
    """Load a specific sub-brick from 4D image"""
    data = img.get_fdata()
    if data.ndim == 4:
        return data[:, :, :, idx]
    return data


def compare_buckets(afni_file, ffs_file):
    """Compare AFNI and fastfuncsim bucket files"""

    print("=" * 80)
    print("OLS Comparison: AFNI vs fastfuncsim")
    print("=" * 80)

    afni_img = nib.load(afni_file)
    ffs_img = nib.load(ffs_file)

    afni_data = afni_img.get_fdata()
    ffs_data = ffs_img.get_fdata()

    print(f"\nAFNI file: {afni_file}")
    print(f"  Shape: {afni_data.shape}")
    print(f"  Dtype: {afni_data.dtype}")

    print(f"\nfastfuncsim file: {ffs_file}")
    print(f"  Shape: {ffs_data.shape}")
    print(f"  Dtype: {ffs_data.dtype}")

    # Compare sub-bricks
    n_bricks = min(afni_data.shape[3], ffs_data.shape[3])

    print(
        f"\n{'Brick':<6} {'Type':<15} {'AFNI Mean':<15} {'FFS Mean':<15} {'Diff %':<12} {'Correlation':<12}"
    )
    print("-" * 80)

    for i in range(n_bricks):
        afni_brick = afni_data[:, :, :, i].flatten()
        ffs_brick = ffs_data[:, :, :, i].flatten()

        # Remove zeros/NaNs
        mask = (
            (afni_brick != 0)
            & (ffs_brick != 0)
            & (~np.isnan(afni_brick))
            & (~np.isnan(ffs_brick))
        )
        afni_brick = afni_brick[mask]
        ffs_brick = ffs_brick[mask]

        if len(afni_brick) == 0:
            continue

        # Determine brick type
        if i == 0:
            brick_type = "F-stat"
        elif i % 2 == 1:
            brick_type = f"Beta_{(i - 1) // 2}"
        else:
            brick_type = f"T-stat_{(i - 2) // 2}"

        # Compute statistics
        afni_mean = np.mean(afni_brick)
        ffs_mean = np.mean(ffs_brick)
        diff_pct = 100 * (ffs_mean - afni_mean) / (abs(afni_mean) + 1e-10)
        corr = np.corrcoef(afni_brick, ffs_brick)[0, 1]

        print(
            f"{i:<6} {brick_type:<15} {afni_mean:>14.6f} {ffs_mean:>14.6f} {diff_pct:>11.3f}% {corr:>11.6f}"
        )

    # Detailed comparison for first beta
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS: First Beta Coefficient")
    print("=" * 80)

    if afni_data.shape[3] > 1 and ffs_data.shape[3] > 1:
        afni_beta1 = afni_data[:, :, :, 1].flatten()
        ffs_beta1 = ffs_data[:, :, :, 1].flatten()

        mask = (
            (afni_beta1 != 0)
            & (ffs_beta1 != 0)
            & (~np.isnan(afni_beta1))
            & (~np.isnan(ffs_beta1))
        )
        afni_beta1 = afni_beta1[mask]
        ffs_beta1 = ffs_beta1[mask]

        diff = ffs_beta1 - afni_beta1
        rel_diff = diff / (np.abs(afni_beta1) + 1e-10)

        print(f"\nAbsolute Difference Statistics:")
        print(f"  Mean diff:     {np.mean(diff):.8f}")
        print(f"  Median diff:   {np.median(diff):.8f}")
        print(f"  Std diff:      {np.std(diff):.8f}")
        print(f"  Max abs diff:  {np.max(np.abs(diff)):.8f}")

        print(f"\nRelative Difference Statistics (%):")
        print(f"  Mean rel diff:   {100 * np.mean(rel_diff):.4f}%")
        print(f"  Median rel diff: {100 * np.median(rel_diff):.4f}%")
        print(f"  Std rel diff:    {100 * np.std(rel_diff):.4f}%")
        print(f"  Max rel diff:    {100 * np.max(np.abs(rel_diff)):.4f}%")

        print(f"\nCorrelation: {np.corrcoef(afni_beta1, ffs_beta1)[0, 1]:.10f}")

        # Check if differences are systematic or random
        print(f"\n{'Percentile':<12} {'AFNI':<15} {'FFS':<15} {'Diff':<15}")
        print("-" * 60)
        for p in [1, 5, 25, 50, 75, 95, 99]:
            afni_p = np.percentile(afni_beta1, p)
            ffs_p = np.percentile(ffs_beta1, p)
            diff_p = ffs_p - afni_p
            print(f"{p}%{'':<9} {afni_p:>14.6f} {ffs_p:>14.6f} {diff_p:>14.6f}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)
    print("""
Possible causes of differences:
1. **Float32 vs Float64 precision**: fastfuncsim uses float32, AFNI may use float64
   - Expected: 0.001% to 0.1% differences
   - Check: If differences are < 0.1%, this is likely the cause

2. **Ridge regularization**: fastfuncsim adds 1e-6*I to X'X
   - Expected: Small systematic bias (~0.01% to 1%)
   - Check: If FFS betas are consistently slightly smaller, this is the cause

3. **Numerical instability**: Different matrix inversion methods
   - Expected: Larger random differences
   - Check: If differences are inconsistent/random, investigate matrix condition

4. **Design matrix mismatch**: Different preprocessing
   - Expected: Large systematic differences
   - Check: Verify design matrices are identical

RECOMMENDATION:
- If correlation > 0.9999 and mean diff < 1%: Precision issue (acceptable)
- If correlation > 0.999 and mean diff < 5%: Ridge regularization (can disable)
- If correlation < 0.99: Investigate design matrix or numerical issues
""")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    compare_buckets(sys.argv[1], sys.argv[2])
