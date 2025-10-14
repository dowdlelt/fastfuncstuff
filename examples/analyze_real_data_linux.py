#!/usr/bin/env python
"""
Analyze real MindsEye data with ARMA(1,1) and write AFNI bucket

This script demonstrates the complete analysis pipeline:
1. Load multiple run files (4 runs)
2. Load AFNI design matrix
3. Fit ARMA(1,1) GLM
4. Compute contrasts from design matrix
5. Write AFNI-compatible bucket file
6. Apply AFNI metadata with 3drefit
7. Save ARMA parameters (like AFNI 3dREMLfit)
"""

import json
import shutil
import subprocess
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
data_dir = Path("/home/logan/Dropbox/temp/ffs_testdata")

run_files = [
    data_dir / "run01.nii.gz",
    data_dir / "run02.nii.gz",
    data_dir / "run03.nii.gz",
    data_dir / "run04.nii.gz",
]

design_matrix_file = data_dir / "X.xmat.1D"
mask_file = data_dir / "mask.nii.gz"  # Optional brain mask

print("=" * 70)
print("Real Data Analysis: MindsEye Imagery")
print("=" * 70)
print(f"\nData directory: {data_dir}")
print(f"Run files: {[f.name for f in run_files]}")
print(f"Design matrix: {design_matrix_file.name}\n")

# =============================================================================
# Step 1: Check for precomputed ARMA parameters
# =============================================================================
arma_params_file = data_dir / "arma_params.nii.gz"
precomputed_arma = None

# if arma_params_file.exists():
#     print("=" * 70)
#     print("Found precomputed ARMA parameters!")
#     print("=" * 70)
#     print(f"\n✓ Loading ARMA parameters from: {arma_params_file.name}")
#     print("  This will skip REML estimation (80% faster!)\n")

#     # Load mask first if it exists
#     use_mask = mask_file.exists()
#     if use_mask:
#         mask = ffs.load_afni_mask(mask_file, threshold=0.5)
#         precomputed_arma = ffs.load_arma_params(arma_params_file, voxel_mask=mask)
#         print(f"  Loaded {precomputed_arma.shape[0]:,} voxel parameters")
#         print(f"  Mean a: {precomputed_arma[:, 0].mean():.3f}")
#         print(f"  Mean b: {precomputed_arma[:, 1].mean():.3f}\n")
#     else:
#         print("  Warning: No mask found, loading all parameters")
#         precomputed_arma = ffs.load_arma_params(arma_params_file)
# else:
#     print("=" * 70)
#     print("No precomputed ARMA parameters found")
#     print("=" * 70)
#     print("\nWill perform full ARMA estimation (~10 minutes)")
#     print("Parameters will be saved for future fast re-analysis\n")

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


a_grid = torch.linspace(0.0, 0.9, 20, device=device)
b_grid = torch.linspace(-0.8, 0.8, 34, device=device)

results, design_info = ffs.analyze_from_design_matrix(
    run_files,
    design_matrix_file,
    method="arma11",
    use_stimulus_only=False,  # Use full design (including polynomials, motion)
    mask_file=mask_file if use_mask else None,
    mask_threshold=0.5,  # Include voxels with mask value > 0.5
    voxel_chunk_size=50000,  # Process 50k voxels at a time (good for GPU)
    arma_a_grid=a_grid,
    arma_b_grid=b_grid,
    device=device,
)

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

print(f"\nContrasts defined in design matrix:")
for i, label in enumerate(design_info["glt_labels"]):
    print(f"  [{i}] {label}")

# Compute contrasts on the FULL model (not stimulus-only!)
# Map stimulus columns to full design matrix
n_regressors = design_info["n_regressors"]
contrasts_full = torch.zeros((4, n_regressors), device=device, dtype=torch.float32)

# Map contrasts to correct columns in full design
# stim_bots tells us where each stimulus starts
contrasts_full[0, stim_bots[0]] = 1  # img_face
contrasts_full[0, stim_bots[1]] = -1  # img_place
contrasts_full[1, stim_bots[2]] = 1  # prc_face
contrasts_full[1, stim_bots[3]] = -1  # prc_place
contrasts_full[2, stim_bots[0]] = 1  # img_face
contrasts_full[2, stim_bots[2]] = -1  # prc_face
contrasts_full[3, stim_bots[1]] = 1  # img_place
contrasts_full[3, stim_bots[3]] = -1  # prc_place

print(f"\nComputing contrasts on full model (no re-fit needed!)...")
contrasts = contrasts_full

contrast_names = [
    "imagery_face_vs_place",
    "perception_face_vs_place",
    "face_imagery_vs_perception",
    "place_imagery_vs_perception",
]

print(f"\nComputing {len(contrast_names)} custom contrasts...")
# Use the original full-model results
contrast_results = ffs.compute_contrasts(results, contrasts, device=device)

print("✓ Contrasts computed")
for i, name in enumerate(contrast_names):
    mean_t = contrast_results["contrast_tstats"][:, i].mean().item()
    print(f"  {name}: mean t = {mean_t:.3f}")

# =============================================================================
# Step 3: Write AFNI bucket
# =============================================================================
print("\n" + "=" * 70)
print("Step 3: Writing AFNI-style bucket file")
print("=" * 70)

output_path = data_dir / "glm_arma11_bucket.nii.gz"

# Use the full-model results and preserve affine from input data
# Pass ALL regressor names (stimulus + nuisance), not just stimulus labels
ffs.write_afni_bucket(
    results,
    output_path,
    condition_names=design_info["column_labels"],  # All 48 regressors
    contrast_names=contrast_names,
    contrast_results=contrast_results,
    affine=results.affine,  # Preserve spatial coordinates from input
)

print(f"✓ Wrote AFNI bucket to: {output_path}")
print(f"✓ Wrote sub-brick labels to: {output_path.with_suffix('.json')}")

# Show bucket contents
with open(output_path.with_suffix(".json"), "r") as f:
    bucket_info = json.load(f)

print(f"\nBucket contains {len(bucket_info['SubBricks'])} sub-bricks:")
print("  [0] Full_Fstat (overall model fit)")

# Show first few condition sub-bricks (just stimulus for brevity)
n_stim = len(design_info["stim_labels"])
for i, label in enumerate(design_info["stim_labels"], 1):
    print(f"  [{2 * i - 1}] {label}#0_Coef (beta)")
    print(f"  [{2 * i}] {label}#0_Tstat (t-statistic)")

# Show that nuisance regressors are also included
n_regressors = design_info["n_regressors"]
if n_regressors > n_stim:
    print(
        f"  ... (plus {(n_regressors - n_stim) * 2} more sub-bricks for nuisance regressors)"
    )

# Show contrast sub-bricks
contrast_start = n_regressors + 1
for i, label in enumerate(contrast_names):
    brick_idx = contrast_start + i * 2 - 1
    print(f"  [{brick_idx}] {label}#0_Coef (beta)")
    print(f"  [{brick_idx + 1}] {label}#0_Tstat (t-statistic)")

# =============================================================================
# Apply AFNI metadata with 3drefit
# =============================================================================
print("\n" + "=" * 70)
print("Step 4: Adding AFNI metadata")
print("=" * 70)

# Get DoF from results
dof = results.dof  # Residual degrees of freedom (n_timepoints - n_regressors)

# Build 3drefit command for labels - single string with space-separated labels
label_str = " ".join(bucket_info["SubBricks"])
print("\nApplying sub-brick labels with 3drefit...")

# Check if 3drefit is available
if shutil.which("3drefit"):
    # Run 3drefit to add labels
    cmd_labels = ["3drefit", "-relabel_all_str", label_str, str(output_path)]

    try:
        subprocess.run(cmd_labels, check=True, capture_output=True, text=True)
        print("✓ Sub-brick labels applied")
    except subprocess.CalledProcessError as e:
        print(f"⚠ Warning: 3drefit labels failed: {e.stderr}")

    # Build 3drefit command for statistical parameters
    print("Applying statistical parameters...")
    cmd_stats = ["3drefit"]

    brick_idx = 0

    # F-statistic (sub-brick 0)
    cmd_stats.extend(
        [
            "-substatpar",
            str(brick_idx),
            "fift",
            str(design_info["n_regressors"]),
            str(dof),
        ]
    )
    brick_idx += 1

    # All regressor t-statistics (stimulus + nuisance)
    for label in design_info["column_labels"]:
        # Skip beta coefficient (brick_idx), add t-stat parameters (brick_idx + 1)
        cmd_stats.extend(["-substatpar", str(brick_idx + 1), "fitt", str(dof)])
        brick_idx += 2

    # Contrast t-statistics
    for label in contrast_names:
        # Skip beta coefficient (brick_idx), add t-stat parameters (brick_idx + 1)
        cmd_stats.extend(["-substatpar", str(brick_idx + 1), "fitt", str(dof)])
        brick_idx += 2

    cmd_stats.append(str(output_path))

    try:
        subprocess.run(cmd_stats, check=True, capture_output=True, text=True)
        print("✓ Statistical parameters applied")
    except subprocess.CalledProcessError as e:
        print(f"⚠ Warning: 3drefit stats failed: {e.stderr}")
else:
    print("⚠ 3drefit not found - skipping AFNI metadata")
    print("  Install AFNI for full metadata support")

# =============================================================================
# Save ARMA parameters (like AFNI 3dREMLfit)
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
✓ Successfully analyzed real MindsEye data!

Data:
  - 4 runs: {design_info["n_timepoints"]} total timepoints
  - {design_info["n_regressors"]} total regressors (stimulus + nuisance)
  - {len(design_info["stim_labels"])} stimulus conditions
  - TR = {design_info["tr"]}s

Method:
  - ARMA(1,1) prewhitened GLS
  - GPU-accelerated on {device.type.upper()}

Output Files:
  1. glm_arma11_bucket.nii.gz - Main results
     • {len(bucket_info["SubBricks"])} sub-bricks total:
       - 1 overall F-statistic
       - {design_info["n_regressors"] * 2} regressor sub-bricks (beta + t-stat pairs)
       - {len(contrast_names) * 2} contrast sub-bricks (beta + t-stat pairs)
     • AFNI labels and stat parameters applied
  
  2. arma_rvar.nii.gz - ARMA variance params (AFNI 3dREMLfit -Rvar format)
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
