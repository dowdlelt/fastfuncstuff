# Tests

## Running

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_glm_core.py -v

# Specific test
pytest tests/test_glm_core.py::TestBasicGLM::test_simple_glm_fit -v

# With output
pytest tests/ -v -s
```

## Organization

Tests mirror the library subpackage structure:

| Test file | Covers |
|-----------|--------|
| `test_glm_core.py`, `test_glm_core_extended.py` | `glm.core` -- GLM fitting, polynomials |
| `test_glm_outputs.py`, `test_write_functions.py` | `glm.outputs` -- NIfTI/AFNI writing |
| `test_xval.py`, `test_xval_missing_events.py` | `glm.xval` -- cross-validation utilities |
| `test_ridge_comprehensive.py`, `test_fracridge_glmsingle.py` | `glm.ridge` -- fractional ridge |
| `test_arma_core.py`, `test_arma_extended.py`, `test_arma_glm_comprehensive.py` | `glm.arma` -- ARMA prewhitening |
| `test_design_builder.py` | `design.builder` -- onset matrices, polynomials |
| `test_design_opt.py` | `design.optimization` -- design efficiency |
| `test_hrf_selection.py` | `design.hrf_selection` -- per-voxel HRF |
| `test_denoise_comprehensive.py`, `test_denoise_combinatorial.py` | `denoise` -- noise PC selection |
| `test_pca_ica_gpu_verification.py` | `decomposition.pca`, `decomposition.ica` |
| `test_icasso_verification.py` | `decomposition.icasso` |
| `test_ica_workflow.py`, `test_ica_tools.py`, `test_ica_postprocess.py` | `decomposition` tools |
| `test_simulation_core.py`, `test_simulation_recovery.py` | `simulation` |
| `test_metrics.py`, `test_metrics_empirical.py` | `simulation.metrics` |
| `test_afni_io.py` | `io.afni` |
| `test_e2e_cli_workflows.py` | end-to-end CLI integration |
| `test_high_level_confirmation.py` | high-level API |
