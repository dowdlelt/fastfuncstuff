# FastFuncSim Testing Guide

## Test Data

Small synthetic dataset in `/home/logan/Dropbox/Data/`:
- **Runs**: 2 runs (360 TRs each, TR=1s)
- **Shape**: 10x10x10 voxels (1000 voxels total)
- **Design**: 10 regressors (8 polynomials + 2 stimuli)
  - Task: `movie`, `prompt`
  - 1 GLT: `movieVprompt-tout` (movie - prompt)
- **Reference outputs**: OLS betas/tstats, REML betas/tstats, Rvar file

## Running Tests

```bash
# All tests
pytest tests/test_basic_workflow.py -v

# Specific test
pytest tests/test_basic_workflow.py::test_ols_basic -v

# Show output
pytest tests/test_basic_workflow.py -v -s

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## Test Coverage

### `test_basic_workflow.py`
- ✅ `test_data_dimensions` - Verify test data shape
- ✅ `test_ols_basic` - OLS fitting, shape checks, metadata
- ⚠️  `test_arma_basic` - ARMA(1,1) fitting with small grid
- ⚠️  `test_cache_save_load` - Cache with header preservation
- ⚠️  `test_contrasts_glt` - GLT contrast handling
- ⚠️  `test_bucket_output` - Bucket file with header
- ⚠️  `test_rvar_output` - Rvar file creation
- ⚠️  `test_masking` - Mask file support
- ⚠️  `test_test_mode` - Subset fitting
- ⚠️  `test_batch_size_reasonable` - Memory optimizations

## Bugs Found by Tests

### 1. **UnboundLocalError: `stim_indices`** (FIXED)
- **File**: `fastfuncsim/analysis.py:659`
- **Issue**: `stim_indices` used in OLS before being defined
- **Fix**: Extract metadata before method check (both OLS and ARMA need it)

### 2. **Shape assumptions**
- Tests revealed that results only contain task regressors (2), not all regressors (10)
- This is correct behavior - nuisance regressors are regressed out

## Type Checking (Pyright)

```bash
# Full project
pyright fastfuncsim/*.py bin/3dREMLfast.py

# Specific module
pyright fastfuncsim/arma_glm.py
```

**Current status**:
- **88 errors, 172 warnings** across all modules
- Most errors are h5py types, optional handling, dynamic attributes
- Focus on fixing high-impact errors (could catch real bugs)

## Adding New Tests

1. Add test function to `test_basic_workflow.py`
2. Use existing fixtures: `test_data_dir`, `temp_output_dir`
3. Use small test data for speed
4. Mark slow tests: `@pytest.mark.slow`
5. Mark GPU tests: `@pytest.mark.gpu`

Example:
```python
def test_new_feature(test_data_dir, temp_output_dir):
    """Test description."""
    results, design_info = analyze_from_design_matrix(
        fmri_data=INPUT_FILES,
        design_matrix_file=DESIGN_MATRIX,
        method="ols",
        device=torch.device("cpu"),  # Use CPU for CI
    )

    assert results.some_output.shape == (1000, 2)
```

## Future: Regression Tests vs AFNI

When reference outputs are ready:
```python
def test_ols_matches_afni(test_data_dir):
    """Compare OLS results with AFNI reference."""
    # Load AFNI reference
    afni_betas = nib.load(test_data_dir / "Decon_OLS_betas.nii.gz").get_fdata()

    # Run our version
    results, _ = analyze_from_design_matrix(...)

    # Compare (with tolerance for numerical differences)
    np.testing.assert_allclose(results.betas, afni_betas, rtol=1e-5)
```

## CI Integration

Add to `.github/workflows/test.yml`:
```yaml
- name: Run tests
  run: |
    pytest tests/test_basic_workflow.py -v --cov=fastfuncsim
```
