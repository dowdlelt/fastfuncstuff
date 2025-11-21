"""Test grid search for a single problematic voxel."""
import numpy as np
import nibabel as nib
import torch
from pathlib import Path

# AFNI reference data paths
AFNI_DATA_DIR = Path.home() / "Dropbox/Data/small_validation_afni_data"
INPUT_FILES = [
    AFNI_DATA_DIR / "small_test_r01.nii.gz",
    AFNI_DATA_DIR / "small_test_r02.nii.gz",
]
DESIGN_MATRIX = AFNI_DATA_DIR / "X.xmat.1D"
AFNI_RVAR = AFNI_DATA_DIR / "afni_REMLvar.nii.gz"

# Load data
X = np.loadtxt(DESIGN_MATRIX)
print(f"Design matrix: {X.shape}")

# Load both runs and concatenate
img1 = nib.load(INPUT_FILES[0])
img2 = nib.load(INPUT_FILES[1])
data1 = img1.get_fdata()
data2 = img2.get_fdata()
data = np.concatenate([data1, data2], axis=3)  # (10, 10, 10, 720)
print(f"Data shape: {data.shape}")

# Load AFNI ARMA parameters
afni_rvar = nib.load(AFNI_RVAR)
afni_a = afni_rvar.get_fdata()[..., 0, 0]  # (10, 10, 10)
afni_b = afni_rvar.get_fdata()[..., 0, 1]

# Pick a problematic voxel - idx 957 from outliers (9,5,7)
voxel_idx = 957
i, j, k = np.unravel_index(voxel_idx, (10, 10, 10))
print(f"\nAnalyzing voxel [{i},{j},{k}] (linear index {voxel_idx})")
print(f"AFNI selected: a={afni_a[i,j,k]:.1f}, b={afni_b[i,j,k]:.1f}")

# Get voxel timeseries
y = data[i, j, k, :]  # (720,)
print(f"Timeseries shape: {y.shape}")

# Convert to torch
X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32)

# Import ARMA functions
from fastfuncsim.arma_glm import prewhiten_with_arma11

# Test a few key (a,b) values
test_params = [
    (0.0, 0.0, "White noise (AFNI choice)"),
    (0.2, -0.1, "AFNI actual choice"),
    (0.9, -0.8, "Our choice (problematic)"),
    (0.1, 0.1, "Mild correlation"),
]

print("\n" + "="*80)
print("LIKELIHOOD COMPARISON")
print("="*80)
print(f"{'(a,b)':<15} {'LogDet(R)':<15} {'LogDet(XtX)':<15} {'RSS':<15} {'Likelihood':<15} {'Description'}")
print("-"*80)

n_timepoints = 720
n_regressors = 10

for a, b, desc in test_params:
    # Prewhiten
    X_w, Y_w, L_inv = prewhiten_with_arma11(X_torch, y_torch, a, b)

    # Fit betas: (X_w' X_w)^-1 X_w' Y_w
    XwTXw = X_w.T @ X_w
    XwTYw = X_w.T @ Y_w

    # Add small regularization for numerical stability
    XwTXw_reg = XwTXw + 1e-8 * torch.eye(n_regressors)
    beta = torch.linalg.solve(XwTXw_reg, XwTYw)

    # Compute RSS
    pred = X_w @ beta
    resid = Y_w - pred
    rss = torch.sum(resid ** 2).item()

    # Compute likelihood terms
    # Term 1: log det of correlation matrix = 2 * sum(log(diag(L)))
    L = torch.linalg.inv(L_inv)
    logdet_R = 2 * torch.sum(torch.log(torch.diag(L) + 1e-10)).item()

    # Term 2: log det of X'X
    sign, logdet_XwTXw = torch.linalg.slogdet(XwTXw)
    if sign > 0:
        logdet_XwTXw = logdet_XwTXw.item()
    else:
        logdet_XwTXw = 1e10

    # Term 3: (n-m) * log(RSS)
    term3 = (n_timepoints - n_regressors) * np.log(rss + 1e-10)

    # Total likelihood
    likelihood = logdet_R + logdet_XwTXw + term3

    print(f"({a:.1f},{b:>5.1f}){'':<5} {logdet_R:<15.2f} {logdet_XwTXw:<15.2f} {rss:<15.4f} {likelihood:<15.2f} {desc}")

# Now do a grid search to find the minimum
print("\n" + "="*80)
print("GRID SEARCH RESULTS")
print("="*80)

from fastfuncsim.arma_glm import get_default_arma_grids

device = torch.device("cpu")
a_grid, b_grid = get_default_arma_grids(device)

print(f"Grid size: {len(a_grid)} a values × {len(b_grid)} b values")

best_likelihood = float('inf')
best_a, best_b = None, None

likelihoods = []

for a in a_grid:
    for b in b_grid:
        a_val = a.item()
        b_val = b.item()

        # Check stability (skip if lambda <= 0)
        lam = ((b_val + a_val) * (1 + a_val * b_val) /
               (1 + 2 * a_val * b_val + b_val**2 + 1e-10))
        if lam <= 0:
            continue

        # Prewhiten
        X_w, Y_w, L_inv = prewhiten_with_arma11(X_torch, y_torch, a_val, b_val)

        # Fit
        XwTXw = X_w.T @ X_w
        XwTYw = X_w.T @ Y_w
        XwTXw_reg = XwTXw + 1e-8 * torch.eye(n_regressors)
        beta = torch.linalg.solve(XwTXw_reg, XwTYw)

        # RSS
        resid = Y_w - X_w @ beta
        rss = torch.sum(resid ** 2).item()

        # Likelihood
        L = torch.linalg.inv(L_inv)
        logdet_R = 2 * torch.sum(torch.log(torch.diag(L) + 1e-10)).item()
        sign, logdet_XwTXw = torch.linalg.slogdet(XwTXw)
        logdet_XwTXw = logdet_XwTXw.item() if sign > 0 else 1e10
        term3 = (n_timepoints - n_regressors) * np.log(rss + 1e-10)
        likelihood = logdet_R + logdet_XwTXw + term3

        likelihoods.append((a_val, b_val, likelihood))

        if likelihood < best_likelihood:
            best_likelihood = likelihood
            best_a, best_b = a_val, b_val

print(f"\nBest parameters found: a={best_a:.1f}, b={best_b:.1f}")
print(f"Best likelihood: {best_likelihood:.2f}")
print(f"\nAFNI selected: a={afni_a[i,j,k]:.1f}, b={afni_b[i,j,k]:.1f}")

# Show top 10 best and worst
likelihoods.sort(key=lambda x: x[2])
print("\nTop 10 best (lowest likelihood):")
for idx, (a, b, lik) in enumerate(likelihoods[:10]):
    marker = " <-- AFNI" if (abs(a - afni_a[i,j,k]) < 0.05 and abs(b - afni_b[i,j,k]) < 0.05) else ""
    marker += " <-- BEST" if idx == 0 else ""
    print(f"  ({a:.1f}, {b:>5.1f}): {lik:.2f}{marker}")

print("\nTop 10 worst (highest likelihood):")
for a, b, lik in likelihoods[-10:][::-1]:
    print(f"  ({a:.1f}, {b:>5.1f}): {lik:.2f}")
