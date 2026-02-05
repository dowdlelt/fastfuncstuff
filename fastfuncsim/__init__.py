"""
FastFuncSim - Fast GPU-accelerated functional MRI simulation and GLM fitting

A Python package for fast simulation of fMRI experiments and GLM-based analysis.
Designed for both interactive exploration and large-scale batch simulations.

Core Philosophy:
- GLM is the engine: Everything feeds into fast GPU-accelerated GLM solver
- Dual-mode: Interactive single simulations + batch thousands
- Flexible: FIR, assumed HRF, HRF library - same GLM, different designs
- Fast: GPU acceleration with MPS/CUDA/CPU fallback

Main Components:
- glm_core: Ultra-fast GLM fitting engine
- design: Design matrix construction (FIR, HRF convolution, etc.)
- hrf: HRF generation (canonical, PIGHS, libraries)
- noise: Realistic fMRI noise generation
- simulation: Simulation pipeline (single and batch)
- utils: Device management and helpers
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Logan Grosenick (converted from MATLAB)"

# Core GLM functionality
# AFNI file I/O
from .afni_io import (
    extract_nuisance_columns,
    extract_stimulus_columns,
    get_contrast_matrix,
    get_run_lengths,
    load_afni_mask,
    load_and_concatenate_runs,
    onsets_to_binary_matrix,
    parse_afni_matrix_notation,
    read_afni_design_matrix,
    read_afni_onset_file,
    read_afni_onset_files,
)

# High-level analysis workflows
from .analysis import (
    analyze_from_design_matrix,
    analyze_from_onsets,
    compute_contrasts,
    compute_contrasts_from_design,
)

# ARMA(1,1) prewhitening for GLM analysis
from .arma_glm import (
    ARMA11Results,
    batch_reml_grid_search,
    build_arma11_covariance,
    compare_ols_vs_arma11,
    compute_arma_lambda,
    fit_glm_arma11,
    load_arma_params,
    prewhiten_with_arma11,
    reml_grid_search,
    save_arma_rvar,
)

# Cross-validated denoising
from .denoise import (
    DenoiseResults,
    compute_full_brain_pc_loadings,
    cross_validate_noise_pcs,
    extract_noise_pcs_per_run,
    fit_denoising_model,
    select_noise_pool_voxels,
)

# Design matrix construction
from .design import (
    build_glm_design,
    convolve_hrf,
    generate_random_onsets,
    make_fir_design,
    make_singletrialdesign,
)

# Design optimization (Liu & Frank metrics)
from .design_optimization import (
    DesignCandidate,
    ISIConstraints,
    compare_designs_summary,
    create_onset_matrix,
    evaluate_design_candidates,
    find_optimal_designs,
    generate_event_sequence,
    generate_isi_sequence,
    plot_fitness_landscape,
    plot_isi_range_by_target_mean,
    plot_isi_range_optimization,
    plot_pareto_frontier,
    sample_design_space,
)
from .design_optimization import (
    plot_hrf_recovery as plot_design_hrf_recovery,
)
from .glm_core import GLMResults, fit_glm, fit_glm_hrf_library, percent_bold_change

# GLM output utilities
from .glm_outputs import (
    slice_glm_results,
    write_afni_bucket,
    write_glm_bucket_as_nifti,
    write_glm_results_nifti,
    write_ols_arma_comparison,
)

# HRF generation
from .hrf import (
    create_flobs_library,  # Backwards compatibility alias
    create_pighs_library,
    flobs_halfcos,  # Backwards compatibility alias
    get_canonical_hrf,
    get_canonical_hrf_library,
    get_hrf_library,
    pighs_halfcos,
)

# HRF selection with cross-validation
from .hrf_selection import (
    HRFSelectionResults,
    fit_glm_hrf_library_with_xval,
    load_hrf_selection_for_arma,
    save_hrf_selection_results,
)

# Empirical metrics (GLS with AR(1) correction)
from .metrics_empirical import (
    build_ar1_covariance_matrix,
    compute_detection_power_empirical,
    compute_estimation_efficiency_empirical,
    estimate_ar1_coefficient,
    evaluate_design_empirical,
    gls_fit,
)

# Noise generation
from .noise import (
    add_drift,
    add_motion_artifacts,
    estimate_noise_parameters_from_data,
    estimate_sfnr,
    generate_ar1_noise,
    generate_ar_noise,
    generate_arma_noise,
    generate_fmri_noise,
    generate_fmri_noise_batch,
)

# Simulation
from .simulation import (
    create_parametric_voxels,
    save_simulation_outputs,
    simulate_batch_experiments,
    simulate_fmri_experiment,
    simulate_fmri_run,
    write_afni_onset_files,
    write_nifti_files,
)

# Utilities
from .utils import (
    get_device,
    print_device_info,
    to_tensor,
)

# Visualization
from .visualization import (
    create_interactive_summary_html,
    plot_batch_summary,
    plot_design_comparison,
    plot_hrf_recovery,
    plot_parametric_exploration,
    plot_simulation_deep_dive,
)

# Convenience imports
__all__ = [
    # GLM
    "fit_glm",
    "fit_glm_hrf_library",
    "percent_bold_change",
    "GLMResults",
    "write_glm_results_nifti",
    "write_glm_bucket_as_nifti",
    "write_ols_arma_comparison",
    # Design
    "build_glm_design",
    "convolve_hrf",
    "make_fir_design",
    "make_singletrialdesign",
    "generate_random_onsets",
    # HRF
    "get_canonical_hrf",
    "get_canonical_hrf_library",
    "get_hrf_library",
    "pighs_halfcos",
    "create_pighs_library",
    "flobs_halfcos",  # Backwards compatibility
    "create_flobs_library",  # Backwards compatibility
    # Noise
    "generate_fmri_noise",
    "generate_fmri_noise_batch",
    "add_drift",
    "add_motion_artifacts",
    "generate_ar1_noise",
    "generate_ar_noise",
    "generate_arma_noise",
    "estimate_noise_parameters_from_data",
    "estimate_sfnr",
    # Simulation
    "simulate_fmri_run",
    "simulate_fmri_experiment",
    "create_parametric_voxels",
    "simulate_batch_experiments",
    "write_afni_onset_files",
    "write_nifti_files",
    "save_simulation_outputs",
    # Utils
    "get_device",
    "print_device_info",
    "to_tensor",
    # Visualization
    "plot_simulation_deep_dive",
    "plot_batch_summary",
    "plot_parametric_exploration",
    "plot_hrf_recovery",
    "plot_design_comparison",
    "create_interactive_summary_html",
    # Design Optimization
    "ISIConstraints",
    "DesignCandidate",
    "generate_event_sequence",
    "generate_isi_sequence",
    "create_onset_matrix",
    "sample_design_space",
    "evaluate_design_candidates",
    "find_optimal_designs",
    "compare_designs_summary",
    "plot_fitness_landscape",
    "plot_pareto_frontier",
    "plot_isi_range_optimization",
    "plot_isi_range_by_target_mean",
    "plot_design_hrf_recovery",
    # Empirical Metrics
    "estimate_ar1_coefficient",
    "build_ar1_covariance_matrix",
    "gls_fit",
    "compute_detection_power_empirical",
    "compute_estimation_efficiency_empirical",
    "evaluate_design_empirical",
    # ARMA(1,1) GLM Analysis
    "fit_glm_arma11",
    "compare_ols_vs_arma11",
    "build_arma11_covariance",
    "reml_grid_search",
    "batch_reml_grid_search",
    "prewhiten_with_arma11",
    "compute_arma_lambda",
    "save_arma_rvar",
    "load_arma_params",
    "ARMA11Results",
    # Cross-validated Denoising
    "DenoiseResults",
    "compute_full_brain_pc_loadings",
    "cross_validate_noise_pcs",
    "extract_noise_pcs_per_run",
    "fit_denoising_model",
    "select_noise_pool_voxels",
    # AFNI File I/O
    "read_afni_onset_file",
    "read_afni_onset_files",
    "onsets_to_binary_matrix",
    "read_afni_design_matrix",
    "extract_stimulus_columns",
    "extract_nuisance_columns",
    "get_contrast_matrix",
    "parse_afni_matrix_notation",
    "load_and_concatenate_runs",
    "get_run_lengths",
    "load_afni_mask",
    # High-level Analysis Workflows
    "analyze_from_onsets",
    "analyze_from_design_matrix",
    "compute_contrasts",
    # HRF Selection with Cross-Validation
    "fit_glm_hrf_library_with_xval",
    "HRFSelectionResults",
    "save_hrf_selection_results",
    "load_hrf_selection_for_arma",
]


# Quick start guide
def print_quickstart():
    """Print quick start guide"""
    print("""
FastFuncSim Quick Start
=======================

1. Interactive Single Simulation:
    import fastfuncsim as ffs
    import torch

    # Setup
    device = ffs.get_device()  # Auto-detect MPS/CUDA/CPU
    hrf = ffs.get_canonical_hrf(stim_duration=5.0, tr=1.0, device=device)
    onsets = ffs.generate_random_onsets(n_timepoints=290, n_conditions=2,
                                        isi_mean=4, tr=1.0, device=device)

    # Simulate
    data = ffs.simulate_fmri_run(onsets, betas=[5, 5], hrf=hrf, tr=1.0,
                                 n_timepoints=290, matrix_size=(50, 50, 5))

    # Fit GLM
    results = ffs.fit_glm(data, onsets, tr=1.0, mode='assumed')
    print(f"Mean R² = {results.r2.mean():.3f}")

2. FIR Estimation (no HRF assumption):
    design_fir = ffs.build_glm_design(onsets, mode='fir', n_fir_lags=30)
    results_fir = ffs.fit_glm(data, design_fir, tr=1.0)

3. HRF Library (try 20 HRFs, pick best per voxel):
    hrf_library = ffs.get_hrf_library('canonical', stim_duration=5.0, tr=1.0, n_hrfs=20)
    results, hrf_idx, r2_all = ffs.fit_glm_hrf_library(data, onsets, hrf_library, tr=1.0)

4. Batch Simulations (thousands of experiments):
    for i in range(1000):
        data = ffs.simulate_fmri_run(...)
        results = ffs.fit_glm(...)
        # Accumulate statistics

For more examples, see examples/ directory.
For documentation, see README.md
    """)


# Print info on import
def _print_import_info():
    """Print brief info when package is imported"""
    import sys

    if not sys.flags.quiet:
        device = get_device()
        print(f"FastFuncSim v{__version__}")
        print(f"Device: {device.type.upper()}", end="")
        if device.type == "cuda":
            import torch

            print(f" ({torch.cuda.get_device_name(device)})")
        elif device.type == "mps":
            print(" (Apple Metal)")
        else:
            print()
        print("Type ffs.print_quickstart() for examples")


# Optionally print on import (disabled by default)
# _print_import_info()
