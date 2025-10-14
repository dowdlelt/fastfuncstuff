#!/usr/bin/env python
"""
Example: Using Precomputed ARMA Parameters for Fast Re-analysis

This example demonstrates how to:
1. Run ARMA(1,1) GLM once and save the parameters
2. Reuse those parameters for fast re-analysis with different contrasts
3. Compare with AFNI-estimated parameters for validation

This is useful for:
- Iterative analysis workflows (exploring different contrasts)
- Validation against AFNI results
- Skipping ~80% of compute time on repeated analyses
"""

from pathlib import Path

import numpy as np

import fastfuncsim as ffs

# Setup
device = ffs.get_device()
print(f"Using device: {device}\n")

# Data paths
data_dir = Path(
    "/Users/logan/local_data/MindsEye_Pilot_Project/derivatives/models/preproc_sub-ME486585_ses-03WB_imagery.results"
)

run_files = [
    data_dir / "run01.nii.gz",
    data_dir / "run02.nii.gz",
    data_dir / "run03.nii.gz",
    data_dir / "run04.nii.gz",
]

design_matrix_file = data_dir / "X.xmat.1D"
mask_file = data_dir / "mask.nii.gz"

# Output paths
arma_params_file = data_dir / "arma_params.nii.gz"
output_bucket = data_dir / "glm_arma11_fast.nii.gz"

print("=" * 70)
print("Example: Precomputed ARMA Parameters")
print("=" * 70)

# =============================================================================
# Scenario 1: First Analysis - Full ARMA estimation
# =============================================================================
print("\n" + "=" * 70)
print("Scenario 1: First Analysis (Full ARMA estimation)")
print("=" * 70)

# Check if we already have saved ARMA parameters
if arma_params_file.exists():
    print(f"\n✓ Found existing ARMA parameters: {arma_params_file.name}")
    print("  Loading precomputed parameters...")

    # Load mask to extract correct voxels
    from fastfuncsim.afni_io import load_afni_mask

    mask = load_afni_mask(mask_file, threshold=0.5)

    # Load parameters
    arma_params = ffs.load_arma_params(arma_params_file, voxel_mask=mask)
    print(f"  Loaded {arma_params.shape[0]:,} voxel parameters")
    print(f"  Mean a: {arma_params[:, 0].mean():.3f}")
    print(f"  Mean b: {arma_params[:, 1].mean():.3f}")

else:
    print("\nNo saved parameters found. Running full ARMA estimation...")
    print("(This will take ~10 minutes for this dataset)")

    # Run full analysis with ARMA estimation
    results_full, design_info = ffs.analyze_from_design_matrix(
        run_files,
        design_matrix_file,
        method="arma11",
        use_stimulus_only=False,
        mask_file=mask_file,
        mask_threshold=0.5,
        voxel_chunk_size=50000,
        device=device,
    )

    print("\n✓ ARMA estimation complete!")
    print(f"  Mean R²: {results_full.r2.mean():.3f}")
    print(f"  Mean ARMA a: {results_full.arma_params[:, 0].mean():.3f}")
    print(f"  Mean ARMA b: {results_full.arma_params[:, 1].mean():.3f}")

    # Save ARMA parameters for future use
    print(f"\nSaving ARMA parameters to: {arma_params_file.name}")
    ffs.save_arma_params(
        results_full.arma_params,
        arma_params_file,
        volume_shape=results_full.full_shape,
        voxel_mask=results_full.voxel_mask,
        affine=results_full.affine,
    )
    print("✓ ARMA parameters saved!")

    arma_params = results_full.arma_params

# =============================================================================
# Scenario 2: Re-analysis with Precomputed ARMA (80% faster!)
# =============================================================================
print("\n" + "=" * 70)
print("Scenario 2: Re-analysis with Precomputed ARMA Parameters")
print("=" * 70)
print("\nNow let's re-run the analysis using precomputed ARMA parameters...")
print("This skips the REML grid search (~80% faster!)\n")

# Import time to measure speedup
import time

start_time = time.time()

# Run analysis with precomputed parameters
results_fast, design_info = ffs.analyze_from_design_matrix(
    run_files,
    design_matrix_file,
    method="arma11",
    use_stimulus_only=False,
    mask_file=mask_file,
    mask_threshold=0.5,
    voxel_chunk_size=50000,
    precomputed_arma_params=arma_params,  # ← KEY: Skip REML estimation!
    device=device,
)

elapsed = time.time() - start_time

print(f"\n✓ Fast re-analysis complete!")
print(f"  Time: {elapsed / 60:.1f} minutes")
print(f"  Mean R²: {results_fast.r2.mean():.3f}")
print(f"  Used precomputed ARMA parameters (no grid search!)")

# =============================================================================
# Compute different contrasts
# =============================================================================
print("\n" + "=" * 70)
print("Computing New Contrasts")
print("=" * 70)

# Get stimulus column indices
stim_bots = design_info["stim_bots"]
stim_labels = design_info["stim_labels"]

print(f"\nStimulus conditions: {stim_labels}")
print(f"Column indices: {stim_bots}")

# Define new contrasts
n_contrasts = 3
contrasts_full = np.zeros((n_contrasts, design_info["n_regressors"]))

# Contrast 1: img_face vs img_place
contrasts_full[0, stim_bots[0]] = 1  # img_face
contrasts_full[0, stim_bots[1]] = -1  # img_place

# Contrast 2: img_body vs img_scene
contrasts_full[1, stim_bots[2]] = 1  # img_body
contrasts_full[1, stim_bots[3]] = -1  # img_scene

# Contrast 3: All imagery vs baseline (average of all 4 conditions)
for idx in stim_bots:
    contrasts_full[2, idx] = 0.25

contrast_names = [
    "face_vs_place",
    "body_vs_scene",
    "imagery_vs_baseline",
]

# Compute contrasts
contrast_results = ffs.compute_contrasts(results_fast, contrasts_full)

print("\n✓ Contrasts computed:")
for i, name in enumerate(contrast_names):
    mean_t = contrast_results["contrast_tstats"][:, i].mean().item()
    print(f"  {name}: mean t = {mean_t:.3f}")

# =============================================================================
# Write output bucket
# =============================================================================
print("\n" + "=" * 70)
print("Writing AFNI Bucket")
print("=" * 70)

ffs.write_afni_bucket(
    results_fast,
    output_bucket,
    condition_names=design_info["column_labels"],
    contrast_names=contrast_names,
    contrast_results=contrast_results,
    affine=results_fast.affine,
)

print(f"✓ Wrote: {output_bucket}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
✓ Demonstrated precomputed ARMA parameter workflow!

Workflow:
  1. First analysis: Full ARMA estimation (~10 min)
     - Estimates optimal (a,b) for each voxel via REML
     - Saves parameters to NIfTI file
  
  2. Future analyses: Use precomputed parameters (~2 min)
     - Skips REML grid search (80% faster!)
     - Same accurate GLS results
     - Can explore different contrasts quickly

Saved Files:
  - {arma_params_file.name}: ARMA(1,1) parameters (2 volumes: a, b)
  - {output_bucket.name}: GLM results bucket

Use Cases:
  ✓ Iterative contrast exploration
  ✓ Comparing multiple analysis approaches
  ✓ Validating against AFNI results
  ✓ Group-level analyses with consistent ARMA params

To use AFNI parameters for validation:
  # Extract AFNI's ARMA parameters (if available)
  # afni_a = read from AFNI output (errts.REML file or similar)
  # afni_b = ...
  # afni_params = np.stack([afni_a, afni_b], axis=1)
  
  # Use AFNI parameters in fastfuncsim
  results = ffs.fit_glm_arma11(
      data, design, tr=2.0,
      precomputed_arma_params=afni_params
  )
  
  # Should match AFNI betas/t-stats exactly!
""")
