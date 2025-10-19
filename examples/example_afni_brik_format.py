# #!/usr/bin/env python
# """
# Example: Writing AFNI BRIK/HEAD format files

# Demonstrates the new AFNI BRIK format support in write_afni_bucket()
# """

# import fastfuncsim as ffs
# import torch
# from pathlib import Path

# # Simulate some results
# device = ffs.get_device()

# # ... (assume we have results from fit_glm_arma11) ...

# # =============================================================================
# # Example 1: Auto-detect format from extension
# # =============================================================================

# # NIfTI format (default) - auto-detected from .nii.gz extension
# ffs.write_afni_bucket(
#     results,
#     'glm_bucket.nii.gz',
#     condition_names=['faces', 'scenes', 'objects'],
#     apply_afni_metadata=True,
# )
# # Creates: glm_bucket.nii.gz

# # AFNI format - auto-detected from .HEAD extension
# ffs.write_afni_bucket(
#     results,
#     'glm_bucket+tlrc.HEAD',
#     condition_names=['faces', 'scenes', 'objects'],
#     apply_afni_metadata=True,
# )
# # Creates: glm_bucket+tlrc.HEAD and glm_bucket+tlrc.BRIK.gz

# # =============================================================================
# # Example 2: Explicit format specification
# # =============================================================================

# # Force AFNI format (even without extension)
# ffs.write_afni_bucket(
#     results,
#     'glm_bucket',
#     condition_names=['faces', 'scenes', 'objects'],
#     output_format='afni',
#     compress_output=True,
# )
# # Creates: glm_bucket.HEAD and glm_bucket.BRIK.gz

# # Force NIfTI format
# ffs.write_afni_bucket(
#     results,
#     'glm_bucket',
#     condition_names=['faces', 'scenes', 'objects'],
#     output_format='nifti_gz',
# )
# # Creates: glm_bucket.nii.gz

# # =============================================================================
# # Example 3: Uncompressed output
# # =============================================================================

# # Uncompressed AFNI
# ffs.write_afni_bucket(
#     results,
#     'glm_bucket+orig.HEAD',
#     condition_names=['faces', 'scenes', 'objects'],
#     compress_output=False,  # Uncompressed
# )
# # Creates: glm_bucket+orig.HEAD and glm_bucket+orig.BRIK

# # Uncompressed NIfTI
# ffs.write_afni_bucket(
#     results,
#     'glm_bucket.nii',
#     condition_names=['faces', 'scenes', 'objects'],
#     compress_output=False,
# )
# # Creates: glm_bucket.nii

# # =============================================================================
# # Example 4: Loading AFNI files
# # =============================================================================

# # Load AFNI BRIK files (works automatically!)
# run_files = [
#     'run01+orig.HEAD',
#     'run02+orig.HEAD',
#     'run03+orig.HEAD',
# ]

# data, run_starts = ffs.load_and_concatenate_runs(run_files)
# print(f"Loaded {len(run_files)} runs")
# print(f"Total timepoints: {data.shape[1]}")
# print(f"Run starts: {run_starts}")

# # =============================================================================
# # Example 5: Mixed format workflow
# # =============================================================================

# # Load AFNI format
# afni_files = ['func01+orig.HEAD', 'func02+orig.HEAD']
# data, _ = ffs.load_and_concatenate_runs(afni_files)

# # Analyze
# results = ffs.fit_glm_arma11(data, design, tr=2.0)

# # Save as NIfTI for portability
# ffs.write_afni_bucket(
#     results,
#     'results_portable.nii.gz',
#     condition_names=labels,
# )

# # Also save as AFNI for AFNI tools
# ffs.write_afni_bucket(
#     results,
#     'results_afni+tlrc.HEAD',
#     condition_names=labels,
# )

# print("✓ Saved in both formats!")
# print("  NIfTI: results_portable.nii.gz")
# print("  AFNI:  results_afni+tlrc.HEAD + results_afni+tlrc.BRIK.gz")

# # =============================================================================
# # Example 6: AFNI naming conventions
# # =============================================================================

# # Original space
# ffs.write_afni_bucket(results, 'stats+orig.HEAD', ...)
# # Creates: stats+orig.HEAD, stats+orig.BRIK.gz

# # Talairach space
# ffs.write_afni_bucket(results, 'stats+tlrc.HEAD', ...)
# # Creates: stats+tlrc.HEAD, stats+tlrc.BRIK.gz

# # AC-PC aligned
# ffs.write_afni_bucket(results, 'stats+acpc.HEAD', ...)
# # Creates: stats+acpc.HEAD, stats+acpc.BRIK.gz

# # =============================================================================
# # Benefits
# # =============================================================================

# print("""
# ✓ No conversion needed!
# ✓ Works seamlessly with AFNI tools
# ✓ Preserves AFNI metadata
# ✓ Same API for both formats
# ✓ Auto-detection from extension
# ✓ Compression supported for both
# """)
