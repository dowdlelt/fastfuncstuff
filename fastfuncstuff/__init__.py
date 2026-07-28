"""FastFuncSim -- GPU-accelerated fMRI analysis toolkit.

Subpackages
-----------
glm            GLM fitting, cross-validation, ridge regression, ARMA prewhitening
design         Design matrix construction, HRF generation, HRF selection
denoise        Cross-validated noise PC denoising
decomposition  PCA, FastICA, ICASSO stability analysis
simulation     Noise generation, fMRI simulation, design metrics
io             AFNI format support, NIfTI I/O
processing     Motion correction, alignment, non-linear warping
cli            Command-line tools
"""

from __future__ import annotations

__version__ = "0.1.0"

# Apply the central torch.compile policy (disable the fragile precompiled-header
# cache) BEFORE any submodule defines/calls a compiled kernel. See _compile.py.
from fastfuncstuff import _compile as _compile  # noqa: F401

# ---------------------------------------------------------------------------
# Analysis workflows
# ---------------------------------------------------------------------------
from fastfuncstuff.analysis import (
    analyze_from_design_matrix,
    analyze_from_onsets,
    compute_contrasts,
    compute_contrasts_from_design,
)

# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------
from fastfuncstuff.denoise.sequential import (
    DenoiseResults,
    compute_full_brain_pc_loadings,
    cross_validate_noise_pcs,
    extract_noise_pcs_per_run,
    fit_denoising_model,
    select_noise_pool_voxels,
)
from fastfuncstuff.design.hrf import (
    create_flobs_library,
    create_pighs_library,
    flobs_halfcos,
    get_canonical_hrf,
    get_canonical_hrf_library,
    get_hrf_library,
    pighs_halfcos,
)
from fastfuncstuff.design.hrf_selection import (
    HRFSelectionResults,
    fit_glm_hrf_library_with_xval,
    load_hrf_selection_for_arma,
    save_hrf_selection_results,
)

# ---------------------------------------------------------------------------
# Design matrices and HRF
# ---------------------------------------------------------------------------
from fastfuncstuff.design.matrices import (
    build_glm_design,
    convolve_hrf,
    generate_random_onsets,
    make_fir_design,
    make_singletrialdesign,
)
from fastfuncstuff.design.optimization import (
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
from fastfuncstuff.design.optimization import (
    plot_hrf_index_recovery as plot_design_hrf_recovery,
)

# ---------------------------------------------------------------------------
# ARMA(1,1)
# ---------------------------------------------------------------------------
from fastfuncstuff.glm.arma import (
    ARMA11Results,
    batch_reml_grid_search,
    build_arma11_covariance,
    build_ljung_box_tau,
    compare_ols_vs_arma11,
    compute_arma_lambda,
    compute_ljung_box_statistic,
    fit_glm_arma11,
    ljung_box_max_lag,
    load_arma_params,
    prewhiten_with_arma11,
    reml_grid_search,
    save_arma_rvar,
)

# ---------------------------------------------------------------------------
# Core GLM
# ---------------------------------------------------------------------------
from fastfuncstuff.glm.core import GLMResults, fit_glm, fit_glm_hrf_library, percent_bold_change
from fastfuncstuff.glm.outputs import (
    slice_glm_results,
    write_afni_bucket,
    write_glm_bucket_as_nifti,
    write_glm_results_nifti,
    write_ols_arma_comparison,
)

# ---------------------------------------------------------------------------
# AFNI I/O
# ---------------------------------------------------------------------------
from fastfuncstuff.io.afni import (
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
from fastfuncstuff.simulation.core import (
    create_parametric_voxels,
    save_simulation_outputs,
    simulate_batch_experiments,
    simulate_fmri_experiment,
    simulate_fmri_run,
    write_afni_onset_files,
    write_nifti_files,
)

# ---------------------------------------------------------------------------
# Empirical metrics
# ---------------------------------------------------------------------------
from fastfuncstuff.simulation.metrics_empirical import (
    build_ar1_covariance_matrix,
    compute_detection_power_empirical,
    compute_estimation_efficiency_empirical,
    estimate_ar1_coefficient,
    evaluate_design_empirical,
    gls_fit,
)

# ---------------------------------------------------------------------------
# Simulation and noise
# ---------------------------------------------------------------------------
from fastfuncstuff.simulation.noise import (
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

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
from fastfuncstuff.utils import (
    accum_dtype,
    compute_power_spectra,
    compute_power_spectrum,
    get_device,
    linalg_device,
    print_device_info,
    to_linalg_f64,
    to_tensor,
)

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
from fastfuncstuff.visualization import (
    create_interactive_summary_html,
    plot_batch_summary,
    plot_design_comparison,
    plot_hrf_recovery,
    plot_parametric_exploration,
    plot_simulation_deep_dive,
)

__all__ = [
    # GLM
    "fit_glm",
    "fit_glm_hrf_library",
    "percent_bold_change",
    "GLMResults",
    "write_glm_results_nifti",
    "write_glm_bucket_as_nifti",
    "write_ols_arma_comparison",
    "slice_glm_results",
    "write_afni_bucket",
    # ARMA
    "fit_glm_arma11",
    "compare_ols_vs_arma11",
    "build_arma11_covariance",
    "reml_grid_search",
    "batch_reml_grid_search",
    "prewhiten_with_arma11",
    "compute_arma_lambda",
    "save_arma_rvar",
    "build_ljung_box_tau",
    "compute_ljung_box_statistic",
    "ljung_box_max_lag",
    "load_arma_params",
    "ARMA11Results",
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
    "flobs_halfcos",
    "create_flobs_library",
    # HRF selection
    "fit_glm_hrf_library_with_xval",
    "HRFSelectionResults",
    "save_hrf_selection_results",
    "load_hrf_selection_for_arma",
    # Design optimization
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
    # Denoising
    "DenoiseResults",
    "compute_full_brain_pc_loadings",
    "cross_validate_noise_pcs",
    "extract_noise_pcs_per_run",
    "fit_denoising_model",
    "select_noise_pool_voxels",
    # AFNI I/O
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
    # Analysis
    "analyze_from_onsets",
    "analyze_from_design_matrix",
    "compute_contrasts",
    "compute_contrasts_from_design",
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
    # Empirical metrics
    "estimate_ar1_coefficient",
    "build_ar1_covariance_matrix",
    "gls_fit",
    "compute_detection_power_empirical",
    "compute_estimation_efficiency_empirical",
    "evaluate_design_empirical",
    # Utils
    "accum_dtype",
    "compute_power_spectra",
    "compute_power_spectrum",
    "get_device",
    "linalg_device",
    "print_device_info",
    "to_linalg_f64",
    "to_tensor",
    # Visualization
    "plot_simulation_deep_dive",
    "plot_batch_summary",
    "plot_parametric_exploration",
    "plot_hrf_recovery",
    "plot_design_comparison",
    "create_interactive_summary_html",
]
