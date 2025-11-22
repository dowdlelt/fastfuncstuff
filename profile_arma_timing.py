"""Profile ARMA grid search to identify bottlenecks."""
import torch
import time
from pathlib import Path

from fastfuncsim.analysis import analyze_from_design_matrix

# AFNI reference data paths
AFNI_DATA_DIR = Path.home() / "Dropbox/Data/small_validation_afni_data"
DESIGN_MATRIX = AFNI_DATA_DIR / "X.xmat.1D"
INPUT_FILES = [
    AFNI_DATA_DIR / "small_test_r01.nii.gz",
    AFNI_DATA_DIR / "small_test_r02.nii.gz",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("ARMA GRID SEARCH TIMING PROFILE")
print("="*80)
print(f"Device: {device}")
print(f"Dataset: 1000 voxels, 720 timepoints, 10 regressors")
print(f"Grid: ~151 valid (a,b) pairs")
print()

# Test both paths
for use_batching in [False, True]:
    mode = "ADAPTIVE BATCHING" if use_batching else "PRECOMPUTED GRID"
    print(f"\n{'='*80}")
    print(f"{mode} MODE")
    print(f"{'='*80}")

    start = time.time()
    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="arma11",
        device=device,
        use_grid_batching=use_batching,
    )
    total_time = time.time() - start

    print(f"\n{'='*80}")
    print(f"{mode} - TOTAL TIME: {total_time:.2f}s")
    print(f"{'='*80}")

print("\n" + "="*80)
print("TIMING BREAKDOWN NEEDED")
print("="*80)
print("""
To get more detailed timing, we need to instrument the code with:

1. Grid precomputation phases:
   - Covariance matrix construction (batched)
   - Cholesky decomposition (batched)
   - Design matrix prewhitening (batched)
   - QR/X'X factorization
   - Likelihood term precomputation

2. Per-voxel grid search:
   - Data prewhitening (per grid point)
   - Beta fitting (per grid point)
   - Residual computation (per grid point)
   - Likelihood evaluation (per grid point)
   - Parameter selection

3. Voxel grouping and final GLM:
   - Grouping by (a,b)
   - Prewhitening per group
   - GLM fitting per group
   - Statistics computation

Let me add detailed timing instrumentation to the code...
""")
