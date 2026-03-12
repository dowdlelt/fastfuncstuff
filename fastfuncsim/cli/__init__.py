"""Command-line tools for fastfuncsim.

Analysis tools:
  ffs_denoise      - Cross-validated noise PC denoising (GLMdenoise/GLMsingle)
  ffs_hrfopt       - Cross-validated HRF optimization per voxel
  ffs_reml         - ARMA(1,1) prewhitened GLM (like AFNI 3dREMLfit)
  ffs_ridge        - Fractional ridge regression with per-voxel regularization
  ffs_xval_r2      - Cross-validated R-squared computation
  ffs_build_design - Build design matrices from onset files
  ffs_decompose    - ICA decomposition with stability analysis
  ffs_deconvolve   - Event-related deconvolution (FIR estimation)
  ffs_denoisatorial - Combinatorial denoising (exhaustive PC subset evaluation)
  ffs_ica          - MELODIC-style ICA with auto component selection
  ffs_pathfinder   - Joint HRF + denoising optimization
  ffs_tps          - Thin-plate spline warping

Image processing tools (via fastfuncsim.processing):
  ffs_moco         - Motion correction
  ffs_allineate    - Affine alignment
  ffs_nwarp        - Non-linear warping
  ffs_qwarp        - Qwarp-style warping
  ffs_motsim       - Motion artifact simulation
  ffs_automask     - Automatic brain masking
  ffs_util_pcwarp  - PC-based warping
"""
