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

Copyright (C) 2026 Logan Dowdle and contributors.

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, see <https://www.gnu.org/licenses/>.

Portions are derived from AFNI and SPM, both GPL-2-or-later; see PROVENANCE.md
for the full attribution."""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Every public name, mapped to the (module, attribute) it lives in. Importing
# them eagerly meant that `import fastfuncstuff.<anything>` -- a header read, a
# CLI --help -- paid ~4 s to bring up torch, scipy, sympy and matplotlib. They
# resolve on first attribute access instead (PEP 562), so `from fastfuncstuff
# import fit_glm` still works and costs what it always did *when it is used*.
#
# The torch.compile policy (fastfuncstuff._compile) used to be applied here for
# the same "before any kernel runs" reason; it now rides on fastfuncstuff.utils,
# which every torch-using module imports. See _compile.configure_inductor.
_LAZY: dict[str, tuple[str, str]] = {
    "ARMA11Results": ("fastfuncstuff.glm.arma", "ARMA11Results"),
    "DenoiseResults": ("fastfuncstuff.denoise.sequential", "DenoiseResults"),
    "DesignCandidate": ("fastfuncstuff.design.optimization", "DesignCandidate"),
    "GLMResults": ("fastfuncstuff.glm.core", "GLMResults"),
    "HRFSelectionResults": ("fastfuncstuff.design.hrf_selection", "HRFSelectionResults"),
    "ISIConstraints": ("fastfuncstuff.design.optimization", "ISIConstraints"),
    "accum_dtype": ("fastfuncstuff.utils", "accum_dtype"),
    "add_drift": ("fastfuncstuff.simulation.noise", "add_drift"),
    "add_motion_artifacts": ("fastfuncstuff.simulation.noise", "add_motion_artifacts"),
    "analyze_from_design_matrix": ("fastfuncstuff.analysis", "analyze_from_design_matrix"),
    "analyze_from_onsets": ("fastfuncstuff.analysis", "analyze_from_onsets"),
    "batch_reml_grid_search": ("fastfuncstuff.glm.arma", "batch_reml_grid_search"),
    "build_ar1_covariance_matrix": (
        "fastfuncstuff.simulation.metrics_empirical",
        "build_ar1_covariance_matrix",
    ),
    "build_arma11_covariance": ("fastfuncstuff.glm.arma", "build_arma11_covariance"),
    "build_glm_design": ("fastfuncstuff.design.matrices", "build_glm_design"),
    "build_ljung_box_tau": ("fastfuncstuff.glm.arma", "build_ljung_box_tau"),
    "compare_designs_summary": ("fastfuncstuff.design.optimization", "compare_designs_summary"),
    "compare_ols_vs_arma11": ("fastfuncstuff.glm.arma", "compare_ols_vs_arma11"),
    "compute_arma_lambda": ("fastfuncstuff.glm.arma", "compute_arma_lambda"),
    "compute_contrasts": ("fastfuncstuff.analysis", "compute_contrasts"),
    "compute_contrasts_from_design": ("fastfuncstuff.analysis", "compute_contrasts_from_design"),
    "compute_detection_power_empirical": (
        "fastfuncstuff.simulation.metrics_empirical",
        "compute_detection_power_empirical",
    ),
    "compute_estimation_efficiency_empirical": (
        "fastfuncstuff.simulation.metrics_empirical",
        "compute_estimation_efficiency_empirical",
    ),
    "compute_full_brain_pc_loadings": (
        "fastfuncstuff.denoise.sequential",
        "compute_full_brain_pc_loadings",
    ),
    "compute_ljung_box_statistic": ("fastfuncstuff.glm.arma", "compute_ljung_box_statistic"),
    "compute_power_spectra": ("fastfuncstuff.utils", "compute_power_spectra"),
    "compute_power_spectrum": ("fastfuncstuff.utils", "compute_power_spectrum"),
    "convolve_hrf": ("fastfuncstuff.design.matrices", "convolve_hrf"),
    "create_flobs_library": ("fastfuncstuff.design.hrf", "create_flobs_library"),
    "create_interactive_summary_html": (
        "fastfuncstuff.visualization",
        "create_interactive_summary_html",
    ),
    "create_onset_matrix": ("fastfuncstuff.design.optimization", "create_onset_matrix"),
    "create_parametric_voxels": ("fastfuncstuff.simulation.core", "create_parametric_voxels"),
    "create_pighs_library": ("fastfuncstuff.design.hrf", "create_pighs_library"),
    "cross_validate_noise_pcs": ("fastfuncstuff.denoise.sequential", "cross_validate_noise_pcs"),
    "estimate_ar1_coefficient": (
        "fastfuncstuff.simulation.metrics_empirical",
        "estimate_ar1_coefficient",
    ),
    "estimate_noise_parameters_from_data": (
        "fastfuncstuff.simulation.noise",
        "estimate_noise_parameters_from_data",
    ),
    "estimate_sfnr": ("fastfuncstuff.simulation.noise", "estimate_sfnr"),
    "evaluate_design_candidates": (
        "fastfuncstuff.design.optimization",
        "evaluate_design_candidates",
    ),
    "evaluate_design_empirical": (
        "fastfuncstuff.simulation.metrics_empirical",
        "evaluate_design_empirical",
    ),
    "extract_noise_pcs_per_run": ("fastfuncstuff.denoise.sequential", "extract_noise_pcs_per_run"),
    "extract_nuisance_columns": ("fastfuncstuff.io.afni", "extract_nuisance_columns"),
    "extract_stimulus_columns": ("fastfuncstuff.io.afni", "extract_stimulus_columns"),
    "find_optimal_designs": ("fastfuncstuff.design.optimization", "find_optimal_designs"),
    "fit_denoising_model": ("fastfuncstuff.denoise.sequential", "fit_denoising_model"),
    "fit_glm": ("fastfuncstuff.glm.core", "fit_glm"),
    "fit_glm_arma11": ("fastfuncstuff.glm.arma", "fit_glm_arma11"),
    "fit_glm_hrf_library": ("fastfuncstuff.glm.core", "fit_glm_hrf_library"),
    "fit_glm_hrf_library_with_xval": (
        "fastfuncstuff.design.hrf_selection",
        "fit_glm_hrf_library_with_xval",
    ),
    "flobs_halfcos": ("fastfuncstuff.design.hrf", "flobs_halfcos"),
    "generate_ar1_noise": ("fastfuncstuff.simulation.noise", "generate_ar1_noise"),
    "generate_ar_noise": ("fastfuncstuff.simulation.noise", "generate_ar_noise"),
    "generate_arma_noise": ("fastfuncstuff.simulation.noise", "generate_arma_noise"),
    "generate_event_sequence": ("fastfuncstuff.design.optimization", "generate_event_sequence"),
    "generate_fmri_noise": ("fastfuncstuff.simulation.noise", "generate_fmri_noise"),
    "generate_fmri_noise_batch": ("fastfuncstuff.simulation.noise", "generate_fmri_noise_batch"),
    "generate_isi_sequence": ("fastfuncstuff.design.optimization", "generate_isi_sequence"),
    "generate_random_onsets": ("fastfuncstuff.design.matrices", "generate_random_onsets"),
    "get_canonical_hrf": ("fastfuncstuff.design.hrf", "get_canonical_hrf"),
    "get_canonical_hrf_library": ("fastfuncstuff.design.hrf", "get_canonical_hrf_library"),
    "get_contrast_matrix": ("fastfuncstuff.io.afni", "get_contrast_matrix"),
    "get_device": ("fastfuncstuff.utils", "get_device"),
    "get_hrf_library": ("fastfuncstuff.design.hrf", "get_hrf_library"),
    "get_run_lengths": ("fastfuncstuff.io.afni", "get_run_lengths"),
    "gls_fit": ("fastfuncstuff.simulation.metrics_empirical", "gls_fit"),
    "linalg_device": ("fastfuncstuff.utils", "linalg_device"),
    "ljung_box_max_lag": ("fastfuncstuff.glm.arma", "ljung_box_max_lag"),
    "load_afni_mask": ("fastfuncstuff.io.afni", "load_afni_mask"),
    "load_and_concatenate_runs": ("fastfuncstuff.io.afni", "load_and_concatenate_runs"),
    "load_arma_params": ("fastfuncstuff.glm.arma", "load_arma_params"),
    "load_hrf_selection_for_arma": (
        "fastfuncstuff.design.hrf_selection",
        "load_hrf_selection_for_arma",
    ),
    "make_fir_design": ("fastfuncstuff.design.matrices", "make_fir_design"),
    "make_singletrialdesign": ("fastfuncstuff.design.matrices", "make_singletrialdesign"),
    "onsets_to_binary_matrix": ("fastfuncstuff.io.afni", "onsets_to_binary_matrix"),
    "parse_afni_matrix_notation": ("fastfuncstuff.io.afni", "parse_afni_matrix_notation"),
    "percent_bold_change": ("fastfuncstuff.glm.core", "percent_bold_change"),
    "pighs_halfcos": ("fastfuncstuff.design.hrf", "pighs_halfcos"),
    "plot_batch_summary": ("fastfuncstuff.visualization", "plot_batch_summary"),
    "plot_design_comparison": ("fastfuncstuff.visualization", "plot_design_comparison"),
    "plot_design_hrf_recovery": ("fastfuncstuff.design.optimization", "plot_hrf_index_recovery"),
    "plot_fitness_landscape": ("fastfuncstuff.design.optimization", "plot_fitness_landscape"),
    "plot_hrf_recovery": ("fastfuncstuff.visualization", "plot_hrf_recovery"),
    "plot_isi_range_by_target_mean": (
        "fastfuncstuff.design.optimization",
        "plot_isi_range_by_target_mean",
    ),
    "plot_isi_range_optimization": (
        "fastfuncstuff.design.optimization",
        "plot_isi_range_optimization",
    ),
    "plot_parametric_exploration": ("fastfuncstuff.visualization", "plot_parametric_exploration"),
    "plot_pareto_frontier": ("fastfuncstuff.design.optimization", "plot_pareto_frontier"),
    "plot_simulation_deep_dive": ("fastfuncstuff.visualization", "plot_simulation_deep_dive"),
    "prewhiten_with_arma11": ("fastfuncstuff.glm.arma", "prewhiten_with_arma11"),
    "print_device_info": ("fastfuncstuff.utils", "print_device_info"),
    "read_afni_design_matrix": ("fastfuncstuff.io.afni", "read_afni_design_matrix"),
    "read_afni_onset_file": ("fastfuncstuff.io.afni", "read_afni_onset_file"),
    "read_afni_onset_files": ("fastfuncstuff.io.afni", "read_afni_onset_files"),
    "reml_grid_search": ("fastfuncstuff.glm.arma", "reml_grid_search"),
    "sample_design_space": ("fastfuncstuff.design.optimization", "sample_design_space"),
    "save_arma_rvar": ("fastfuncstuff.glm.arma", "save_arma_rvar"),
    "save_hrf_selection_results": (
        "fastfuncstuff.design.hrf_selection",
        "save_hrf_selection_results",
    ),
    "save_simulation_outputs": ("fastfuncstuff.simulation.core", "save_simulation_outputs"),
    "select_noise_pool_voxels": ("fastfuncstuff.denoise.sequential", "select_noise_pool_voxels"),
    "simulate_batch_experiments": ("fastfuncstuff.simulation.core", "simulate_batch_experiments"),
    "simulate_fmri_experiment": ("fastfuncstuff.simulation.core", "simulate_fmri_experiment"),
    "simulate_fmri_run": ("fastfuncstuff.simulation.core", "simulate_fmri_run"),
    "slice_glm_results": ("fastfuncstuff.glm.outputs", "slice_glm_results"),
    "to_factor_f64": ("fastfuncstuff.utils", "to_factor_f64"),
    "to_linalg_f64": ("fastfuncstuff.utils", "to_linalg_f64"),
    "to_tensor": ("fastfuncstuff.utils", "to_tensor"),
    "write_afni_bucket": ("fastfuncstuff.glm.outputs", "write_afni_bucket"),
    "write_afni_onset_files": ("fastfuncstuff.simulation.core", "write_afni_onset_files"),
    "write_glm_bucket_as_nifti": ("fastfuncstuff.glm.outputs", "write_glm_bucket_as_nifti"),
    "write_glm_results_nifti": ("fastfuncstuff.glm.outputs", "write_glm_results_nifti"),
    "write_nifti_files": ("fastfuncstuff.simulation.core", "write_nifti_files"),
    "write_ols_arma_comparison": ("fastfuncstuff.glm.outputs", "write_ols_arma_comparison"),
}


def __getattr__(name: str):
    """Resolve a public name to its module on first access (PEP 562)."""
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    value = getattr(import_module(module), attr)
    globals()[name] = value  # later lookups skip __getattr__ entirely
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:  # type checkers and IDEs want the real bindings
    from fastfuncstuff.analysis import (
        analyze_from_design_matrix,
        analyze_from_onsets,
        compute_contrasts,
        compute_contrasts_from_design,
    )
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
    from fastfuncstuff.glm.core import GLMResults, fit_glm, fit_glm_hrf_library, percent_bold_change
    from fastfuncstuff.glm.outputs import (
        slice_glm_results,
        write_afni_bucket,
        write_glm_bucket_as_nifti,
        write_glm_results_nifti,
        write_ols_arma_comparison,
    )
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
    from fastfuncstuff.simulation.metrics_empirical import (
        build_ar1_covariance_matrix,
        compute_detection_power_empirical,
        compute_estimation_efficiency_empirical,
        estimate_ar1_coefficient,
        evaluate_design_empirical,
        gls_fit,
    )
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
    from fastfuncstuff.utils import (
        accum_dtype,
        compute_power_spectra,
        compute_power_spectrum,
        get_device,
        linalg_device,
        print_device_info,
        to_factor_f64,
        to_linalg_f64,
        to_tensor,
    )
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
    "to_factor_f64",
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
