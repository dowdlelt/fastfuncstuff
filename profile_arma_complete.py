"""Complete timing profile including ALL phases of ARMA analysis."""
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
print("COMPLETE ARMA ANALYSIS TIMING BREAKDOWN")
print("="*80)
print(f"Device: {device}")
print(f"Dataset: 1000 voxels, 720 timepoints, 10 regressors\n")

# Phase timing
phase_times = {}

def time_phase(name):
    """Decorator to time a phase."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            result = func(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start
            phase_times[name] = elapsed
            return result
        return wrapper
    return decorator

# Monkey-patch key functions to add timing
import fastfuncsim.arma_glm as arma_module

# Save originals
orig_precompute = arma_module.precompute_reml_grid
orig_search = arma_module.search_voxels_precomputed_grid

# Wrap with timing
@time_phase("1_precompute_grid")
def precompute_with_timing(*args, **kwargs):
    return orig_precompute(*args, **kwargs)

@time_phase("2_grid_search")
def search_with_timing(*args, **kwargs):
    kwargs['enable_timing'] = False  # Disable inner timing for overall view
    return orig_search(*args, **kwargs)

# Patch
arma_module.precompute_reml_grid = precompute_with_timing
arma_module.search_voxels_precomputed_grid = search_with_timing

try:
    total_start = time.time()

    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="arma11",
        device=device,
        use_grid_batching=False,
    )

    total_time = time.time() - total_start

    print("\n" + "="*80)
    print("TIMING BREAKDOWN BY PHASE")
    print("="*80)
    print(f"{'Phase':<40} {'Time':>10} {'% Total':>10}")
    print("-"*80)

    for name, t in sorted(phase_times.items()):
        pct = 100 * t / total_time
        print(f"{name:<40} {t:>9.3f}s {pct:>9.1f}%")

    # Calculate unaccounted time
    accounted = sum(phase_times.values())
    unaccounted = total_time - accounted

    print("-"*80)
    print(f"{'Accounted time':<40} {accounted:>9.3f}s {100*accounted/total_time:>9.1f}%")
    print(f"{'Other (data loading, overhead)':<40} {unaccounted:>9.3f}s {100*unaccounted/total_time:>9.1f}%")
    print("-"*80)
    print(f"{'TOTAL':<40} {total_time:>9.3f}s {100.0:>9.1f}%")
    print("="*80)

    print("\nPHASE DESCRIPTIONS:")
    print("  1_precompute_grid: Build & factorize correlation matrices for all (a,b)")
    print("  2_grid_search: Evaluate 1000 voxels x 151 grid points")
    print("  Other: Final GLM (group by a,b), data loading, overhead")

finally:
    # Restore originals
    arma_module.precompute_reml_grid = orig_precompute
    arma_module.search_voxels_precomputed_grid = orig_search
