"""GLM engine: fitting, cross-validation, ridge regression, ARMA prewhitening, output."""

from fastfuncsim.glm.arma import (
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
from fastfuncsim.glm.core import GLMResults, fit_glm, fit_glm_hrf_library, percent_bold_change
from fastfuncsim.glm.outputs import (
    slice_glm_results,
    write_afni_bucket,
    write_glm_bucket_as_nifti,
    write_glm_results_nifti,
    write_ols_arma_comparison,
)
from fastfuncsim.glm.ridge import (
    RidgeResults,
    create_single_trial_design,
    fit_ridge_single_trial,
)

__all__ = [
    "GLMResults",
    "fit_glm",
    "fit_glm_hrf_library",
    "percent_bold_change",
    "slice_glm_results",
    "write_afni_bucket",
    "write_glm_bucket_as_nifti",
    "write_glm_results_nifti",
    "write_ols_arma_comparison",
    "RidgeResults",
    "create_single_trial_design",
    "fit_ridge_single_trial",
    "ARMA11Results",
    "batch_reml_grid_search",
    "build_arma11_covariance",
    "compare_ols_vs_arma11",
    "compute_arma_lambda",
    "fit_glm_arma11",
    "load_arma_params",
    "prewhiten_with_arma11",
    "reml_grid_search",
    "save_arma_rvar",
]
