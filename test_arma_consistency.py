#!/usr/bin/env python3
"""
Test consistency between scalar and batch ARMA(1,1) covariance functions.

This ensures both implementations use the same mathematical formula
through the single source of truth: _compute_arma11_lambda()
"""

import torch
import numpy as np
from fastfuncsim.arma_glm import (
    build_arma11_covariance,
    build_arma11_covariance_batch,
    compute_arma_lambda,
)


def test_lambda_consistency():
    """Test that lambda computation is consistent across all functions"""

    test_cases = [
        (0.5, 0.2),
        (0.7, -0.1),
        (0.3, 0.0),
        (0.0, 0.3),
        (0.8, 0.3),
        (0.4, -0.3),
    ]

    device = torch.device("cpu")
    n = 100

    print("=" * 70)
    print("ARMA(1,1) Lambda Consistency Test")
    print("=" * 70)
    print("\nTesting that scalar, batch, and helper functions produce same λ:")
    print()

    all_passed = True

    for a, b in test_cases:
        # Method 1: Public helper function
        lambda_helper = compute_arma_lambda(a, b)

        # Method 2: Scalar covariance function (extract from matrix)
        R_scalar = build_arma11_covariance(a, b, n, device)
        if R_scalar is None:
            print(f"  a={a:4.1f}, b={b:5.1f} → INVALID (filtered)")
            continue
        lambda_scalar = R_scalar[0, 1].item()  # R[0,1] = lambda

        # Method 3: Batch covariance function
        a_grid = torch.tensor([a], device=device)
        b_grid = torch.tensor([b], device=device)
        R_batch, params, keys = build_arma11_covariance_batch(a_grid, b_grid, n, device)
        if len(R_batch) == 0:
            print(f"  a={a:4.1f}, b={b:5.1f} → INVALID (filtered by batch)")
            continue
        lambda_batch = R_batch[0, 0, 1].item()  # First matrix, R[0,1]

        # Check consistency
        diff_helper = abs(lambda_helper - lambda_scalar)
        diff_batch = abs(lambda_batch - lambda_scalar)
        passed = (diff_helper < 1e-6) and (diff_batch < 1e-6)

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  a={a:4.1f}, b={b:5.1f} → λ={lambda_scalar:7.4f}  {status}")

        if not passed:
            print(f"      Helper: {lambda_helper:.6f} (diff: {diff_helper:.2e})")
            print(f"      Batch:  {lambda_batch:.6f} (diff: {diff_batch:.2e})")
            all_passed = False

    print()
    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Math is consistent!")
    else:
        print("✗ SOME TESTS FAILED - Check implementation!")
    print("=" * 70)

    return all_passed


def test_matrix_consistency():
    """Test that entire covariance matrices match"""

    print("\n" + "=" * 70)
    print("Full Covariance Matrix Consistency Test")
    print("=" * 70)

    device = torch.device("cpu")
    n = 50

    # Test on a grid
    a_vals = [0.3, 0.5, 0.7]
    b_vals = [-0.2, 0.0, 0.2]

    print(f"\nComparing {len(a_vals) * len(b_vals)} matrices (n={n})...")
    print()

    all_passed = True

    for a in a_vals:
        for b in b_vals:
            # Build with scalar function
            R_scalar = build_arma11_covariance(a, b, n, device)
            if R_scalar is None:
                continue

            # Build with batch function
            a_grid = torch.tensor([a], device=device)
            b_grid = torch.tensor([b], device=device)
            R_batch, _, _ = build_arma11_covariance_batch(a_grid, b_grid, n, device)
            if len(R_batch) == 0:
                continue

            R_batch_single = R_batch[0]  # Extract first matrix

            # Compare
            max_diff = torch.max(torch.abs(R_scalar - R_batch_single)).item()
            passed = max_diff < 1e-5

            status = "✓" if passed else "✗"
            print(f"  {status} a={a:3.1f}, b={b:4.1f}  max_diff={max_diff:.2e}")

            if not passed:
                all_passed = False

    print()
    print("=" * 70)
    if all_passed:
        print("✓ ALL MATRICES MATCH - Implementations are consistent!")
    else:
        print("✗ MATRICES DIFFER - Check implementation!")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    passed1 = test_lambda_consistency()
    passed2 = test_matrix_consistency()

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    if passed1 and passed2:
        print("✓✓✓ ALL CONSISTENCY CHECKS PASSED ✓✓✓")
        print("\nBoth scalar and batch implementations use the same math!")
        print("Single source of truth: _compute_arma11_lambda()")
    else:
        print("✗✗✗ CONSISTENCY CHECKS FAILED ✗✗✗")
        print("\nImplementations are producing different results!")
    print("=" * 70)
