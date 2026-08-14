"""Command-line tools for fastfuncstuff.

All CLI entry points live here. Library code stays in its respective module
(glm/, design/, denoise/, processing/, etc.).

Analysis tools:
  ffs_build_design  - Build design matrices from onset files
  ffs_decompose     - ICA decomposition with stability analysis
  ffs_deconvolve    - Event-related deconvolution (FIR estimation)
  ffs_denoise       - Cross-validated noise PC denoising (GLMdenoise/GLMsingle)
  ffs_denoisatorial - Combinatorial denoising (exhaustive PC subset evaluation)
  ffs_hrfopt        - Cross-validated HRF optimization per voxel
  ffs_ica           - MELODIC-style ICA with auto component selection
  ffs_pathfinder    - Joint HRF + denoising optimization
  ffs_reml          - ARMA(1,1) prewhitened GLM (like AFNI 3dREMLfit)
  ffs_ridge         - Fractional ridge regression with per-voxel regularization
  ffs_tps           - Thin-plate spline HRF estimation
  ffs_xval_r2       - Cross-validated R-squared computation

Image processing tools:
  ffs_allineate     - Affine alignment
  ffs_moco          - Motion correction
  ffs_motsim        - Motion-simulation nuisance regressors
  ffs_nwarp         - Non-linear warp application
  ffs_qwarp         - Qwarp-style nonlinear registration
  ffs_slicetime     - Slice timing correction
  ffs_util_automask - Automatic brain masking
  ffs_util_pcwarp   - PC-based warp analysis

Stats / utility tools:
  ffs_info          - Dataset header report / 3dinfo-compatible value flags
  ffs_spatial_xcorr - Spatial cross-correlation

Benchmark:
  ffs_benchmark     - Performance benchmarking suite
"""
