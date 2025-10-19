#!/usr/bin/env python
"""
Modular ARMA(1,1) Analysis - Clean Composition Pattern

This example demonstrates the new modular approach:
- Uses existing library functions
- Each function does ONE thing well
- Compose them for complete workflow
- No 700-line scripts with helper functions!

The key insight: DON'T create mega-functions. Instead, use small,
well-defined functions from the library and compose them.
"""

from pathlib import Path
import fastfuncsim as ffs

# =============================================================================
# Configuration
# =============================================================================
data_dir = Path(
    "/media/logan/NVMe_Storage/Data/nii_data/PROJECT_TASKFORCE/derivatives/models/"
    "preproc_sub-pilot01_ses-02_movietask.results"
)

run_files = [
    data_dir / f"pb04.preproc_sub-pilot01_ses-02_movietask.r{i:02d}.scale+orig.HEAD"
    for i in range(1, 9)
]

design_matrix_file = data_dir / "X.xmat.1D"
mask_file = data_dir / "full_mask.preproc_sub-pilot01_ses-02_movietask+orig.HEAD"
output_prefix = data_dir / "glm_arma11"  # Will create multiple files from this

# =============================================================================
# Step 1: Fit GLM with ARMA(1,1) - Use existing function, no changes needed!
# =============================================================================
print("=" * 70)
print("Step 1: Fitting ARMA(1,1) GLM")
print("=" * 70)

results, design_info = ffs.analyze_from_design_matrix(
    run_files,
    design_matrix_file,
    method="arma11",
    mask_file=mask_file,
    device="cuda",
    # NEW: Get OLS baseline for validation
    want_ols=True,  # This parameter was added to fit_glm_arma11()
)

print(
    f"✓ Fit complete: {design_info['n_regressors']} regressors, "
    f"{design_info['n_timepoints']} timepoints"
)
print(f"  Mean R²: {results.r2.mean():.3f}")
print(
    f"  Mean ARMA(a,b): ({results.arma_params[:, 0].mean():.3f}, "
    f"{results.arma_params[:, 1].mean():.3f})"
)

# =============================================================================
# Step 2: Compute contrasts - NEW modular function!
# =============================================================================
print("\n" + "=" * 70)
print("Step 2: Computing contrasts from design matrix")
print("=" * 70)

# NEW: Auto-extract GLTs from design with CPU fallback
contrast_results_arma = ffs.compute_contrasts_from_design(
    results,
    design_info,
    auto_cpu_fallback=True,  # Handles large datasets automatically!
)

# Also compute for OLS if available
if hasattr(results, "ols_results") and results.ols_results is not None:
    contrast_results_ols = ffs.compute_contrasts_from_design(
        results.ols_results,
        design_info,
        auto_cpu_fallback=True,
    )
else:
    contrast_results_ols = None

if contrast_results_arma:
    print(f"✓ Computed {len(design_info['glt_labels'])} contrasts")
    for name, tstat in zip(
        design_info["glt_labels"], contrast_results_arma["contrast_tstats"].T
    ):
        print(f"  {name}: mean t = {tstat.mean():.3f}")

# =============================================================================
# Step 3: Slice results by regressor type - NEW modular function!
# =============================================================================
print("\n" + "=" * 70)
print("Step 3: Separating stimulus vs nuisance regressors")
print("=" * 70)

# NEW: Use library function instead of 57-line helper function!
stim_indices = design_info["stim_bots"]
results_stim = ffs.slice_glm_results(results, stim_indices)

# Also slice OLS if available
if hasattr(results, "ols_results") and results.ols_results is not None:
    ols_stim = ffs.slice_glm_results(results.ols_results, stim_indices)
    # Attach to main results for comparison writer
    results_stim.ols_results = ols_stim

print(f"✓ Extracted {len(stim_indices)} stimulus regressors")
print(f"  Stimulus: {design_info['stim_labels']}")

# Nuisance regressors (everything NOT in stim_bots)
all_indices = set(range(design_info["n_regressors"]))
stim_set = set(stim_indices)
nuisance_indices = sorted(all_indices - stim_set)

if nuisance_indices:
    results_nuisance = ffs.slice_glm_results(results, nuisance_indices)
    nuisance_labels = [design_info["column_labels"][i] for i in nuisance_indices]
    print(f"✓ Extracted {len(nuisance_indices)} nuisance regressors")

# =============================================================================
# Step 4: Write output files - Use existing functions with new prefix arg!
# =============================================================================
print("\n" + "=" * 70)
print("Step 4: Writing output files")
print("=" * 70)

# Write stimulus regressors + contrasts (OLS vs ARMA comparison)
if hasattr(results_stim, "ols_results") and results_stim.ols_results is not None:
    print("\n[1] Writing OLS vs ARMA comparison (stimulus + contrasts)...")

    outputs = ffs.write_ols_arma_comparison(
        results_stim,
        output_prefix / "main",  # Creates main_OLS.nii.gz and main_ARMA.nii.gz
        condition_names=design_info["stim_labels"],
        contrast_names=design_info.get("glt_labels", []),
        contrast_results_arma=contrast_results_arma,
        contrast_results_ols=contrast_results_ols,
        apply_afni_metadata=True,
        compress_output=True,
    )

    print(f"  ✓ {outputs['ols']}")
    print(f"  ✓ {outputs['arma']}")
    print(f"  ✓ {outputs['comparison_summary']}")
else:
    print("\n[1] Writing ARMA results (stimulus + contrasts)...")

    ffs.write_afni_bucket(
        results_stim,
        output_prefix / "main_ARMA.nii.gz",
        condition_names=design_info["stim_labels"],
        contrast_names=design_info.get("glt_labels", []),
        contrast_results=contrast_results_arma,
        apply_afni_metadata=True,
        compress_output=True,
    )

    print(f"  ✓ {output_prefix / 'main_ARMA.nii.gz'}")

# Write nuisance regressors (ARMA only)
if nuisance_indices:
    print("\n[2] Writing nuisance regressors (motion, baseline, etc.)...")

    ffs.write_afni_bucket(
        results_nuisance,
        output_prefix / "nuisance.nii.gz",
        condition_names=nuisance_labels,
        apply_afni_metadata=True,
        compress_output=True,
    )

    print(f"  ✓ {output_prefix / 'nuisance.nii.gz'}")

# Save ARMA parameters (for fast re-analysis)
print("\n[3] Saving ARMA parameters...")

ffs.save_arma_rvar(
    results,
    output_prefix / "arma_rvar.nii.gz",
    volume_shape=results.full_shape,
    voxel_mask=results.voxel_mask,
    affine=results.affine,
    max_lag=30,
)

print(f"  ✓ {output_prefix / 'arma_rvar.nii.gz'} (6 volumes, AFNI-compatible)")
print("  → Reuse with: ffs.load_arma_params('arma_rvar.nii.gz') for 80% speedup!")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"""
✓ Successfully analyzed {len(run_files)} runs!

Method: ARMA(1,1) prewhitened GLS
Device: GPU-accelerated
Regressors: {len(stim_indices)} stimulus + {len(nuisance_indices)} nuisance

Output Files:
  • main_ARMA.nii.gz - Stimulus regressors + contrasts
  • main_OLS.nii.gz - OLS baseline for comparison
  • nuisance.nii.gz - Motion, baseline, etc.
  • arma_rvar.nii.gz - ARMA parameters (reusable!)
  • main_comparison_summary.json - Quantitative OLS vs ARMA stats

Key Insight:
This script is CLEAN because it uses small, modular library functions:
  • ffs.analyze_from_design_matrix() - Existing function (no changes)
  • ffs.compute_contrasts_from_design() - NEW: Auto-extract + CPU fallback
  • ffs.slice_glm_results() - NEW: Slice by regressor indices
  • ffs.write_ols_arma_comparison() - Existing function (no changes)
  • ffs.write_afni_bucket() - Existing function (no changes)

No 700-line scripts. No helper functions. Just composition!
""")
