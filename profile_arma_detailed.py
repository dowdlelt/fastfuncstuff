"""Detailed timing profile of ARMA grid search."""
import torch
from pathlib import Path

# Import before running to enable timing
from fastfuncsim.timing_utils import get_profiler, reset_profiler

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
print("DETAILED ARMA GRID SEARCH TIMING PROFILE")
print("="*80)
print(f"Device: {device}")
print(f"Dataset: 1000 voxels, 720 timepoints, 10 regressors")
print(f"Grid: ~151 valid (a,b) pairs")
print()

# Get profiler
profiler = get_profiler(enabled=True)

# For now, we'll manually instrument the code
# Let me first check what's already timed and add more if needed

print("\nRunning ARMA analysis (precomputed grid mode)...")
print("This will take ~2-3 seconds...\n")

# Import and patch to enable timing
import fastfuncsim.arma_glm as arma_module

# Save original function
original_search = arma_module.search_voxels_precomputed_grid

# Wrap with timing enabled
def search_with_timing(*args, **kwargs):
    kwargs['enable_timing'] = True
    return original_search(*args, **kwargs)

# Monkey patch
arma_module.search_voxels_precomputed_grid = search_with_timing

try:
    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="arma11",
        device=device,
        use_grid_batching=False,  # Use precomputed grid path
    )

    print("\n" + profiler.get_report())

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Restore original
    arma_module.search_voxels_precomputed_grid = original_search

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("""
The timing breakdown shows where time is spent during grid search:

1. get_grid_data: Fetching precomputed matrices from dict
2. prewhiten_data: Y @ L_inv.T matrix multiplication
3. compute_XwTYw: X_w.T @ Y_w matrix multiplication
4. solve_beta: Linear solve for beta coefficients
5. compute_residuals: Computing RSS for likelihood
6. compute_likelihood: REML likelihood calculation
7. update_best: Updating best parameters

Each operation is called 151 times (once per grid point).
Total calls shown will be 151 * operation_count.
""")
