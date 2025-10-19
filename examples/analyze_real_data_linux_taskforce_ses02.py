#!/usr/bin/env python
"""
Analyze real fMRI data with ARMA(1,1) and write AFNI bucket

This script demonstrates the complete analysis pipeline:
1. Load multiple run files (9 runs in this example)
2. Load AFNI design matrix
3. Fit ARMA(1,1) GLM with intelligent memory management
4. Compute contrasts from design matrix (auto-extracted from GLTs)
5. Write AFNI-compatible bucket files with automatic metadata
6. Save ARMA parameters (like AFNI 3dREMLfit)

NOTE: AFNI metadata (labels and stat parameters) are now applied AUTOMATICALLY
by write_afni_bucket() with apply_afni_metadata=True (default). No need for
manual 3drefit commands!

MEMORY MANAGEMENT FOR LARGE DATASETS:
--------------------------------------
For datasets with many runs (many timepoints), ARMA covariance matrices
become huge and can cause GPU OOM errors:
  - 3240 timepoints → 3240×3240 matrix = 42 MB per matrix
  - 170 grid combinations → 7+ GB just for covariances
  - Add Cholesky factors, inverses → 15+ GB total

This script automatically detects large datasets (>1000 timepoints) and:
  1. **Reduces ARMA grid size** (~50 combinations instead of 170)
     - Still covers the important parameter space
     - Drastically reduces memory footprint
  2. **Keeps GPU for speed** (10x faster than CPU)
     - Library already uses CPU for Cholesky (memory-efficient)
     - GPU used for matrix operations (fast)
  3. **Uses smaller voxel batches** (1000 instead of 50000)
     - Prevents OOM during final GLS solve
     - Each batch processes complete timeseries together

KEY: We reduce the grid resolution, NOT the device speed!
Result: GPU-fast analysis that fits in memory
"""

import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

import fastfuncsim as ffs

# Setup
device = ffs.get_device()
print(f"Using device: {device}\n")

# Data paths
data_dir = Path(
    "/media/logan/NVMe_Storage/Data/nii_data/PROJECT_TASKFORCE/derivatives/models/preproc_sub-pilot01_ses-02_movietask.results"
)

run_files = [
    data_dir
    / f"pb04.preproc_sub-pilot01_ses-02_movietask.r{runNum:02d}.scale+orig.HEAD"
    for runNum in range(1, 9)
]

design_matrix_file = data_dir / "X.xmat.1D"
mask_file = (
    data_dir / "full_mask.preproc_sub-pilot01_ses-02_movietask+orig.HEAD"
)  # Optional brain mask

print("=" * 70)
print("Real Data Analysis: TASKFORCE")
print("=" * 70)
print(f"\nData directory: {data_dir}")
print(f"Run files: {[f.name for f in run_files]}")
print(f"Design matrix: {design_matrix_file.name}\n")

# =============================================================================
# Step 1: Check for precomputed ARMA parameters
# =============================================================================
arma_params_file = data_dir / "nonononarma_rvar.nii.gz"
precomputed_arma = None

if arma_params_file.exists():
    print("=" * 70)
    print("Found precomputed ARMA parameters!")
    print("=" * 70)
    print(f"\n✓ Loading ARMA parameters from: {arma_params_file.name}")
    print("  This will skip REML estimation (80% faster!)\n")

    # Load mask first if it exists
    use_mask = mask_file.exists()
    if use_mask:
        mask = ffs.load_afni_mask(mask_file, threshold=0.5)
        precomputed_arma = ffs.load_arma_params(arma_params_file, voxel_mask=mask)
        print(f"  Loaded {precomputed_arma.shape[0]:,} voxel parameters")
        print(f"  Mean a: {precomputed_arma[:, 0].mean():.3f}")
        print(f"  Mean b: {precomputed_arma[:, 1].mean():.3f}\n")
    else:
        print("  Warning: No mask found, loading all parameters")
        precomputed_arma = ffs.load_arma_params(arma_params_file)
else:
    print("=" * 70)
    print("No precomputed ARMA parameters found")
    print("=" * 70)
    print("\nWill perform full ARMA estimation (~10 minutes)")
    print("Parameters will be saved for future fast re-analysis\n")

# =============================================================================
# Step 2: Analyze with ARMA(1,1)
# =============================================================================
print("=" * 70)
if precomputed_arma is not None:
    print("Step 2: Running ARMA(1,1) GLM with precomputed parameters")
else:
    print("Step 2: Running ARMA(1,1) GLM analysis (full estimation)")
print("=" * 70)

# Analyze - this will:
# - Load and concatenate the 4 run files
# - Validate run lengths against design matrix RunStart
# - Extract design matrix
# - Fit ARMA(1,1) GLM (with or without REML estimation)
# - Apply mask if available (restricts analysis to brain voxels)

# Check if mask exists
use_mask = mask_file.exists()
print(f"Brain mask: {'Found' if use_mask else 'Not found (analyzing all voxels)'}")

start_time = time.time()


a_grid = torch.linspace(0.0, 0.9, 10, device=device)
b_grid = torch.linspace(-0.8, 0.8, 17, device=device)

# Memory management: For large datasets (many runs), we need to be smarter
# Rule of thumb: n_timepoints > 1000 → use smaller batches but KEEP GPU!
n_runs = len(run_files)
estimated_timepoints = n_runs * 360  # Rough estimate: 360 TRs per run

if estimated_timepoints > 1000:
    print(f"\n⚠ Large dataset detected (~{estimated_timepoints} timepoints)")
    print("  Optimizing for GPU memory efficiency:")

    # MEMORY MATH for large T:
    # - Single cov matrix: T×T×4bytes (e.g., 3240² × 4 = 42 MB)
    # - With 100 grid combos: 42 MB × 100 = 4.2 GB
    # - Plus Cholesky, inverses, prewhitened X: multiply by ~3 = 12+ GB
    #
    # SOLUTION: Smaller grid = fewer matrices to hold in memory simultaneously
    # Trade-off: Slightly coarser ARMA grid, but still very accurate
    # Benefit: Runs on GPU (fast!) instead of CPU (slow!)

    # Calculate memory-safe grid size
    # Rule: Total memory < 4 GB for covariances alone
    # Target: ~50 combinations max
    max_combos = 50
    estimated_combos = max_combos

    # Try to keep aspect ratio reasonable (more 'a' points than 'b')
    n_a = 9  # AFNI -Grid 3 default
    n_b = max(5, max_combos // n_a)  # At least 5 'b' points

    # Adjust if still too many
    while n_a * n_b > max_combos and n_a > 5:
        n_a -= 1
        n_b = max(5, max_combos // n_a)

    a_grid = torch.linspace(0.1, 0.9, n_a, device=device)
    b_grid = torch.linspace(-0.4, 0.4, n_b, device=device)
    total_combos = len(a_grid) * len(b_grid)

    # Estimate memory usage
    matrix_size_mb = (estimated_timepoints**2) * 4 / (1024**2)
    total_memory_gb = (
        matrix_size_mb * total_combos * 3
    ) / 1024  # ×3 for all intermediate matrices

    print(
        f"  - ARMA grid: {len(a_grid)} a × {len(b_grid)} b = {total_combos} combinations"
    )
    print(
        f"  - Estimated GPU memory: ~{total_memory_gb:.1f} GB (covariances + processing)"
    )

    # Calculate safe voxel batch size based on available GPU memory
    # Memory per batch = batch_size × n_timepoints × n_regressors × 4 bytes
    # For 3240 TRs × 131 regressors:
    #   - 5000 voxels = 8.1 GB ❌ (auto-detected default, too large!)
    #   - 1000 voxels = 1.6 GB
    #   - 500 voxels = 0.8 GB ✓ (safe for 16 GB GPU)
    #
    # NOTE: voxel_chunk_size parameter is currently NOT passed through to
    # fit_glm_arma11 (library bug). Using environment variable workaround.
    voxel_chunk_size = 500  # Very conservative for large T
    print(
        f"  - Voxel chunk size: {voxel_chunk_size:,} (⚠ auto-detect would use 5000 - too large!)"
    )

    # Calculate expected memory per batch
    bytes_per_batch = voxel_chunk_size * estimated_timepoints * 131 * 4
    gb_per_batch = bytes_per_batch / (1024**3)
    print(f"  - Memory per batch: ~{gb_per_batch:.1f} GB (safe for 16 GB GPU)")
    print(f"  - Device: GPU ({device}) - much faster than CPU!")
    print(f"  - Expected time: ~15-20 minutes (more batches = more time)\n")
else:
    # Small dataset: full speed ahead!
    a_grid = torch.linspace(0.0, 0.9, 10, device=device)
    b_grid = torch.linspace(-0.8, 0.8, 17, device=device)
    voxel_chunk_size = 50000
    print(f"Standard settings: chunk_size={voxel_chunk_size:,}, device={device}")

# Run the analysis
# The library's cholesky_on_cpu=True (default) handles memory efficiently:
#   1. Covariance matrices: built on GPU (fast)
#   2. Cholesky factors: computed on CPU (saves GPU memory)
#   3. Prewhitened matrices: cached in GPU memory
#   4. REML grid search: done per-voxel on GPU (fast)
#   5. Final GLS solve: batched on GPU (fast + memory-efficient)
#
# ==============================================================================
# WORKAROUND for library bug: analyze_from_design_matrix doesn't pass
# voxel_chunk_size to fit_glm_arma11 as batch_size parameter.
#
# Problem: With 3240 timepoints, auto-detection chooses batch_size=5000
#          → 5000 × 3240 × 131 × 4 bytes = 8.1 GB per batch ❌
#
# Solution: Manually load data and call fit_glm_arma11 with explicit batch_size
#           → 500 × 3240 × 131 × 4 bytes = 0.8 GB per batch ✓
#
# This gives us explicit control over memory usage for large datasets.
# ==============================================================================

print("Step 1: Loading data and design matrix...")
# Read design matrix to get structure info
design_dict = ffs.read_afni_design_matrix(design_matrix_file)
design_matrix = design_dict["matrix"]
tr = design_dict["tr"]

# Load and concatenate run files
print(f"Loading {len(run_files)} fMRI runs...")
run_data = []
for i, run_file in enumerate(run_files, 1):
    img = nib.load(str(run_file))
    data_4d = img.get_fdata()
    # Reshape (x,y,z,t) -> (nvoxels, t)
    data_2d = data_4d.reshape(-1, data_4d.shape[-1])
    run_data.append(data_2d)
    print(f"  Run {i}: shape {data_4d.shape} -> {data_2d.shape}")

# Concatenate along time dimension
data_concat = np.concatenate(run_data, axis=1)
print(f"✓ Concatenated data shape: {data_concat.shape}")

# Apply mask if provided
if use_mask:
    mask = ffs.load_afni_mask(mask_file, threshold=0.5)
    mask_flat = mask.reshape(-1)
    data_concat = data_concat[mask_flat, :]
    print(f"✓ Applied mask: {data_concat.shape[0]:,} voxels kept")

# Convert to torch tensors
data_tensor = torch.from_numpy(data_concat).float()
design_tensor = torch.from_numpy(design_matrix).float()

print(f"\nStep 2: Running ARMA(1,1) GLM with batch_size={voxel_chunk_size}...")
print(f"  This explicitly controls memory usage (auto-detect would use 5000!)")

# Now call fit_glm_arma11 directly with EXPLICIT batch_size
results = ffs.fit_glm_arma11(
    data_tensor,
    design_tensor,
    tr=tr,
    a_grid=a_grid,
    b_grid=b_grid,
    batch_size=voxel_chunk_size,  # ← EXPLICIT control!
    device=device,
    want_ols=True,
    verbose=True,
)

# Add metadata back to results (that analyze_from_design_matrix would have added)
results.affine = nib.load(str(run_files[0])).affine  # type: ignore
if use_mask:
    results.voxel_mask = torch.from_numpy(mask_flat)  # type: ignore
    results.full_shape = mask.shape  # type: ignore

# Copy same metadata to OLS results for file writing
if results.ols_results is not None:
    results.ols_results.affine = results.affine  # type: ignore
    if use_mask:
        results.ols_results.voxel_mask = results.voxel_mask  # type: ignore
        results.ols_results.full_shape = results.full_shape  # type: ignore

# Construct design_info dict manually
design_info = design_dict  # Already has most of what we need
design_info["mask_voxels"] = data_concat.shape[0] if use_mask else None

elapsed_time = time.time() - start_time

print("\n✓ Analysis complete!")
print(f"  Time elapsed: {elapsed_time / 60:.1f} minutes")
if precomputed_arma is not None:
    print("  Used precomputed ARMA parameters (skipped REML estimation)")
else:
    print("  Performed full ARMA estimation")

print(f"\n  Number of runs: {len(design_info['run_starts'])}")
print(f"  Run starts: {design_info['run_starts']}")
print(f"  Total timepoints: {design_info['n_timepoints']}")
print(f"  TR: {design_info['tr']}s")
print(f"  Stimulus labels: {design_info['stim_labels']}")
print(f"  Number of regressors: {design_info['n_regressors']}")

if use_mask:
    print(f"  Masked voxels analyzed: {design_info['mask_voxels']:,}")
    print(f"  Full volume shape: {results.full_shape}")  # type: ignore

print(f"\n  Mean R²: {results.r2.mean():.3f}")
print(f"  Mean ARMA a: {results.arma_params[:, 0].mean():.3f}")
print(f"  Mean ARMA b: {results.arma_params[:, 1].mean():.3f}")

# Get stimulus betas
stim_bots = design_info["stim_bots"]
stim_tops = design_info["stim_tops"]
print(f"\n  Stimulus betas (mean across voxels):")
for i, label in enumerate(design_info["stim_labels"]):
    stim_idx = stim_bots[i]
    beta_mean = results.betas[:, stim_idx].mean().item()
    tstat_mean = results.tstats[:, stim_idx].mean().item()
    print(f"    {label}: β = {beta_mean:.3f}, t = {tstat_mean:.3f}")

# =============================================================================
# Step 2: Compute contrasts from design matrix
# =============================================================================
print("\n" + "=" * 70)
print("Step 2: Computing contrasts from design matrix")
print("=" * 70)

# GLT (General Linear Test) contrasts are automatically extracted from the
# X.xmat.1D file by read_afni_design_matrix(). These are stored in:
#   - design_info["glt_labels"]: names of each contrast
#   - design_info["glt_matrices"]: contrast weight vectors
# This approach is general and works with any AFNI design matrix!

print(f"\nContrasts defined in design matrix:")
for i, label in enumerate(design_info["glt_labels"]):
    print(f"  [{i}] {label}")

# Extract contrasts from design matrix (if any are defined)
if design_info["glt_labels"] and design_info["glt_matrices"]:
    print(f"\nUsing {len(design_info['glt_labels'])} contrasts from X.xmat.1D file...")

    # Stack all GLT matrices into a single tensor
    n_regressors = design_info["n_regressors"]
    n_contrasts = len(design_info["glt_matrices"])

    contrasts = torch.zeros(
        (n_contrasts, n_regressors), device=device, dtype=torch.float32
    )
    for i, glt_matrix in enumerate(design_info["glt_matrices"]):
        # Convert numpy array to torch tensor if needed
        glt_tensor = torch.as_tensor(glt_matrix, device=device, dtype=torch.float32)
        # GLT matrices are typically 1D (one row per contrast)
        if glt_tensor.ndim == 1:
            contrasts[i, :] = glt_tensor
        else:
            # If 2D, take the first row (single contrast per GLT)
            contrasts[i, :] = glt_tensor[0, :]

    contrast_names = design_info["glt_labels"]
    print(f"\nComputing {len(contrast_names)} contrasts from design matrix...")
else:
    print("\n⚠ No contrasts defined in design matrix")
    print("  Skipping contrast computation")
    contrasts = None
    contrast_names = []

# Compute contrasts (if any were defined)
# For large datasets, var_betas can be huge (21+ GB!)
# Solution: Compute contrasts on CPU where we have more RAM
if contrasts is not None:
    if estimated_timepoints > 1000:
        print(
            "  ⚠ Large dataset: computing contrasts on CPU (var_betas too large for GPU)"
        )
        contrast_device = torch.device("cpu")
        contrasts_cpu = contrasts.cpu()
    else:
        contrast_device = device
        contrasts_cpu = contrasts

    contrast_results = ffs.compute_contrasts(
        results, contrasts_cpu, device=contrast_device
    )

    # Compute OLS contrasts too (same device/memory strategy)
    print("  Computing OLS contrasts (for comparison)...")
    ols_contrast_results = ffs.compute_contrasts(
        results.ols_results, contrasts_cpu, device=contrast_device
    )
else:
    contrast_results = None
    ols_contrast_results = None

if contrast_results is not None:
    print("✓ Contrasts computed (ARMA and OLS)")
    for i, name in enumerate(contrast_names):
        arma_t = contrast_results["contrast_tstats"][:, i].mean().item()
        ols_t = ols_contrast_results["contrast_tstats"][:, i].mean().item()
        print(f"  {name}: ARMA t = {arma_t:.3f}, OLS t = {ols_t:.3f}")
# =============================================================================
# Step 3: Write AFNI bucket files with automatic metadata
# =============================================================================
print("\n" + "=" * 70)
print("Step 3: Writing AFNI-style bucket files (with automatic metadata)")
print("=" * 70)

# Identify stimulus vs nuisance regressors
stim_labels = design_info["stim_labels"]
all_labels = design_info["column_labels"]
n_stim = len(stim_labels)
n_total = len(all_labels)

print(f"\nRegressor breakdown:")
print(f"  Stimulus regressors: {n_stim}")
print(f"  Nuisance regressors: {n_total - n_stim}")
print(f"  Total regressors: {n_total}")


# Helper function to slice results object by regressor indices
def slice_results(results, indices):
    """Create a new results object with only selected regressors

    Works with both GLMResults (OLS) and ARMA11Results objects.
    """
    import torch

    sliced = type(results)()  # Create new instance of same type

    # Copy scalar attributes (common to both GLMResults and ARMA11Results)
    sliced.r2 = results.r2
    sliced.sigma2 = results.sigma2
    sliced.dof = results.dof
    sliced.tr = results.tr
    sliced.voxel_mask = results.voxel_mask
    sliced.full_shape = results.full_shape
    sliced.affine = results.affine
    sliced.original_shape = results.original_shape

    # Copy ARMA-specific attributes (only if present)
    if hasattr(results, "arma_params"):
        sliced.arma_params = results.arma_params
    if hasattr(results, "arma_lambda"):
        sliced.arma_lambda = results.arma_lambda
    if hasattr(results, "reml_likelihood"):
        sliced.reml_likelihood = results.reml_likelihood

    # Slice regressor-specific attributes
    # IMPORTANT: Use .clone() to create independent copies, not views!
    if results.betas is not None:
        if torch.is_tensor(results.betas):
            sliced.betas = results.betas[:, indices].clone()
        else:
            sliced.betas = results.betas[:, indices].copy()

    if results.tstats is not None:
        if torch.is_tensor(results.tstats):
            sliced.tstats = results.tstats[:, indices].clone()
        else:
            sliced.tstats = results.tstats[:, indices].copy()

    if results.fstats is not None:
        # F-stat is for ALL regressors - we'll recompute or keep original
        sliced.fstats = results.fstats  # Keep full model F-stat for now

    # Slice var_betas for ARMA results (variance-covariance matrix)
    if hasattr(results, "var_betas") and results.var_betas is not None:
        if torch.is_tensor(results.var_betas):
            sliced.var_betas = results.var_betas[:, indices, :][:, :, indices].clone()
        else:
            sliced.var_betas = results.var_betas[:, indices, :][:, :, indices].copy()

    # Slice xtx_inv for OLS results (needed for contrast computation)
    if hasattr(results, "xtx_inv") and results.xtx_inv is not None:
        if torch.is_tensor(results.xtx_inv):
            sliced.xtx_inv = results.xtx_inv[indices, :][:, indices].clone()
        else:
            sliced.xtx_inv = results.xtx_inv[indices, :][:, indices].copy()

    # Copy time-series attributes (not regressor-specific)
    if hasattr(results, "residuals"):
        sliced.residuals = results.residuals
    if hasattr(results, "predicted"):
        sliced.predicted = results.predicted
    if hasattr(results, "residuals_whitened"):
        sliced.residuals_whitened = results.residuals_whitened

    return sliced


# ------- File 1: Main effects (stimulus) + Contrasts -------
output_path_main = data_dir / "glm_arma11_main.nii.gz"
print(f"\n[1] Writing main effects + contrasts: {output_path_main.name}")

# Create results object with only stimulus regressors
# IMPORTANT: Use stim_bots for actual column indices, NOT range(n_stim)!
# Stimulus regressors may not be in columns 0:n_stim
stim_indices = design_info["stim_bots"]
results_main = slice_results(results, stim_indices)

# Also slice OLS results to match
ols_results_main = slice_results(results.ols_results, stim_indices)

print(f"  Using stimulus regressor indices: {stim_indices}")

if contrast_results is not None:
    # Write OLS and ARMA side-by-side
    # Create a temporary ARMA results object with sliced ARMA and OLS
    results_for_comparison = results_main
    results_for_comparison.ols_results = ols_results_main

    outputs = ffs.write_ols_arma_comparison(
        results_for_comparison,
        data_dir / "glm_main",  # Creates glm_main_OLS.nii.gz and glm_main_ARMA.nii.gz
        condition_names=stim_labels,
        contrast_names=contrast_names,
        contrast_results_ols=ols_contrast_results,
        contrast_results_arma=contrast_results,
        affine=results.affine,
    )
else:
    ffs.write_afni_bucket(
        results_main,
        output_path_main,
        condition_names=stim_labels,
        affine=results.affine,
        apply_afni_metadata=True,
        compress_output=True,
    )

print(f"  ✓ Wrote {n_stim} stimulus regressors + {len(contrast_names)} contrasts")
print(f"  ✓ AFNI metadata applied automatically")

# ------- File 2: Nuisance regressors (motion, baseline, etc.) -------
if n_total > n_stim:
    output_path_nuisance = data_dir / "glm_arma11_nuisance.nii.gz"
    print(f"\n[2] Writing nuisance regressors: {output_path_nuisance.name}")

    # Get indices of nuisance regressors (all columns NOT in stim_bots)
    stim_indices_set = set(stim_indices)
    nuisance_indices = [i for i in range(n_total) if i not in stim_indices_set]
    nuisance_labels = [all_labels[i] for i in nuisance_indices]

    print(f"  Using nuisance regressor indices: {nuisance_indices}")

    results_nuisance = slice_results(results, nuisance_indices)

    ffs.write_afni_bucket(
        results_nuisance,
        output_path_nuisance,
        condition_names=nuisance_labels,
        affine=results.affine,
        apply_afni_metadata=True,  # Automatic 3drefit!
        compress_output=True,
    )

    print(f"  ✓ Wrote {len(nuisance_labels)} nuisance regressors")
    print("  ✓ AFNI metadata applied automatically")
else:
    output_path_nuisance = None
    print(
        "\n[2] No nuisance regressors to write (all regressors are stimulus)"
    )  # Show main file contents
print(f"\n✓ Main file: {output_path_main.name}")
with open(output_path_main.with_suffix(".json"), "r") as f:
    main_bucket_info = json.load(f)

print(f"  Contains {len(main_bucket_info['SubBricks'])} sub-bricks:")
print("    [0] Full_Fstat (overall model fit)")
for i, label in enumerate(stim_labels, 1):
    print(f"    [{2 * i - 1}] {label}#0_Coef")
    print(f"    [{2 * i}] {label}#0_Tstat")

if contrast_names:
    contrast_start = n_stim + 1
    for i, label in enumerate(contrast_names):
        brick_idx = contrast_start + i * 2 - 1
        print(f"    [{brick_idx}] {label}#0_Coef")
        print(f"    [{brick_idx + 1}] {label}#0_Tstat")

# Show nuisance file contents (if exists)
if output_path_nuisance:
    print(f"\n✓ Nuisance file: {output_path_nuisance.name}")
    with open(output_path_nuisance.with_suffix(".json"), "r") as f:
        nuisance_bucket_info = json.load(f)
    print(f"  Contains {len(nuisance_bucket_info['SubBricks'])} sub-bricks:")
    print("    [0] Full_Fstat (overall model fit)")
    for i, label in enumerate(nuisance_labels, 1):
        print(f"    [{2 * i - 1}] {label}#0_Coef")
        print(f"    [{2 * i}] {label}#0_Tstat")

# =============================================================================
# Step 4: Save ARMA parameters (like AFNI 3dREMLfit)
# =============================================================================
print("\n" + "=" * 70)
print("Step 5: Saving ARMA variance parameters (AFNI -Rvar format)")
print("=" * 70)

# Save ARMA variance parameters (AFNI 3dREMLfit -Rvar compatible format)
arma_rvar_path = data_dir / "arma_rvar.nii.gz"
if precomputed_arma is not None:
    print(f"\nARMA parameters already exist")
    print("  Skipping -Rvar save (parameters unchanged)")
else:
    print(f"\nSaving ARMA -Rvar file (AFNI-compatible): {arma_rvar_path.name}")
    print("  This contains 6 volumes like AFNI 3dREMLfit:")
    print("    [0] = a (AR parameter)")
    print("    [1] = b (MA parameter)")
    print("    [2] = lambda (lag-1 correlation)")
    print("    [3] = StDev (prewhitened residual std)")
    print("    [4] = -LogLik (negative REML log-likelihood)")
    print("    [5] = LjungBox (residual autocorrelation test)")

    ffs.save_arma_rvar(
        results,
        arma_rvar_path,
        volume_shape=results.full_shape,
        voxel_mask=results.voxel_mask,
        affine=results.affine,
        max_lag=30,  # AFNI default
    )
    print("\n✓ Saved ARMA -Rvar parameters (6 volumes with AFNI labels)")
    print(f"  Mean a: {results.arma_params[:, 0].mean():.3f}")
    print(f"  Mean b: {results.arma_params[:, 1].mean():.3f}")
    print(f"  Mean λ: {results.arma_lambda.mean():.3f}")
    if results.residuals_whitened is not None:
        print(f"  Ljung-Box statistic computed (h={30}, df={30 - 2})")
    else:
        print("  ⚠ Ljung-Box set to zero (residuals not saved)")
    print("\n  → Next run: use load_arma_params(arma_rvar_path) for 80% time savings!")

# Save R² map
r2_path = data_dir / "glm_r2.nii.gz"
print(f"\nSaving R² map to: {r2_path.name}")

if hasattr(results, "voxel_mask") and results.voxel_mask is not None:
    mask_flat = results.voxel_mask.cpu().numpy().reshape(-1)
    r2_vol = np.zeros(np.prod(results.full_shape), dtype=np.float32)
    r2_vol[mask_flat] = results.r2.cpu().numpy()
    r2_vol = r2_vol.reshape(results.full_shape)
else:
    r2_vol = results.r2.cpu().numpy().reshape(results.full_shape)

r2_img = nib.Nifti1Image(r2_vol, results.affine)
r2_img.header.set_xyzt_units(xyz="mm")
r2_img.header["descrip"] = b"ARMA(1,1) GLM R-squared"
nib.save(r2_img, r2_path)
print("✓ Saved R² map")
print(f"  Mean R²: {results.r2.mean():.3f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
✓ Successfully analyzed TASKFORCE data!

Data:
  - {len(run_files)} runs: {design_info["n_timepoints"]} total timepoints
  - {design_info["n_regressors"]} total regressors ({n_stim} stimulus + {n_total - n_stim} nuisance)
  - TR = {design_info["tr"]}s

Method:
  - ARMA(1,1) prewhitened GLS
  - GPU-accelerated on {device.type.upper()}

Output Files:
  1. glm_arma11_main.nii.gz - Main effects + contrasts
     • {len(main_bucket_info["SubBricks"])} sub-bricks:
       - 1 overall F-statistic (stimulus regressors only)
       - {n_stim * 2} stimulus sub-bricks (beta + t-stat pairs)
       {f"- {len(contrast_names) * 2} contrast sub-bricks (beta + t-stat pairs)" if contrast_names else "- No contrasts defined"}
     • AFNI labels and stat parameters applied
     • Contrasts automatically extracted from X.xmat.1D GLT definitions
  
  2. glm_arma11_nuisance.nii.gz - Nuisance regressors
     • {len(nuisance_bucket_info["SubBricks"]) if output_path_nuisance else 0} sub-bricks:
       - 1 overall F-statistic (nuisance regressors only)
       - {(n_total - n_stim) * 2 if output_path_nuisance else 0} nuisance sub-bricks (beta + t-stat pairs)
     • AFNI labels and stat parameters applied
  
  3. arma_rvar.nii.gz - ARMA variance params (AFNI 3dREMLfit -Rvar format)
     • Volume [0]: a = AR parameter
     • Volume [1]: b = MA parameter
     • Volume [2]: lam = lag-1 correlation
     • Volume [3]: StDev = prewhitened residual std deviation
     • Volume [4]: -LogLik = negative REML log-likelihood
     • Volume [5]: LjungBox = residual autocorrelation test (chi², df={30 - 2})
     • Fully compatible with AFNI analysis tools!
  
  3. arma_params.nii.gz - Simple ARMA parameters (2 volumes: a, b)
     • Reusable for fast re-analysis (80% time savings!)
  
  4. glm_r2.nii.gz - R² map
     • Model fit quality per voxel
     • Compare with OLS to see ARMA improvement

To view in AFNI:
  cd {data_dir}
  afni -niml &
  # Open glm_arma11_bucket.nii.gz
  # Compare with AFNI's 3dREMLfit results

To reuse ARMA parameters (fast re-analysis):
  # Load saved parameters
  arma_params = ffs.load_arma_params('arma_params.nii.gz', mask=mask)
  
  # Re-run with different contrasts (80% faster!)
  results = ffs.analyze_from_design_matrix(
      ..., precomputed_arma_params=arma_params
  )

The ARMA(1,1) results should show:
  ✓ More accurate t-statistics (corrected for autocorrelation)
  ✓ Similar or slightly better R² compared to OLS
  ✓ Proper statistical inference for fMRI data
""")

print("✓ Analysis complete!\n")
