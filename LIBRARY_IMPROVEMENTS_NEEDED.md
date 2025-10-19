# Library Improvements for OLS/ARMA Comparison and AFNI Integration

## Summary
Based on user feedback, the following improvements should be made to the fastfuncsim library to make it more user-friendly and reduce code duplication in example scripts.

## 1. Add `want_ols` Parameter to `fit_glm_arma11()`

### Current State
- `fit_glm_arma11()` only returns ARMA(1,1) corrected results
- Users cannot easily compare OLS vs ARMA performance
- OLS computation is fast (especially on GPU) but not exposed

### Proposed Change
```python
def fit_glm_arma11(
    data: Union[torch.Tensor, np.ndarray],
    design: Union[torch.Tensor, np.ndarray],
    tr: float,
    # ... existing parameters ...
    want_ols: bool = True,  # NEW PARAMETER
    # ... rest of parameters ...
) -> ARMA11Results:
```

### Implementation Details
- **When `want_ols=True` (default)**:
  - Compute OLS fit FIRST (fast, GPU-accelerated)
  - Store in `results.ols_betas`, `results.ols_tstats`, `results.ols_fstats`
  - Then continue with ARMA(1,1) as normal
  - Return single ARMA11Results object with both sets of results

- **Benefits**:
  - Users can compare OLS vs ARMA easily
  - OLS residuals already computed for REML estimation
  - Minimal overhead (OLS is ~10% of total time)
  - Standard workflow: always have OLS baseline

### Modified ARMA11Results Class
```python
class ARMA11Results:
    """Container for ARMA(1,1) GLM results"""
    
    def __init__(self):
        # ARMA results (corrected)
        self.betas = None          # GLS betas
        self.tstats = None         # Corrected t-stats
        self.fstats = None         # Corrected F-stats
        
        # OLS results (uncorrected, for comparison)
        self.ols_betas = None      # OLS betas
        self.ols_tstats = None     # Uncorrected t-stats  
        self.ols_fstats = None     # Uncorrected F-stats
        
        # ARMA parameters
        self.arma_params = None
        self.arma_lambda = None
        
        # ... rest unchanged ...
```

## 2. Move 3drefit Logic to Library (`write_afni_bucket()`)

### Current State
- Example scripts contain 50+ lines of 3drefit command building
- Duplicated across all example scripts
- Error-prone (easy to get DoF calculations wrong)
- Not maintained centrally

### Proposed Change
Add `apply_afni_metadata=True` parameter to `write_afni_bucket()`:

```python
def write_afni_bucket(
    results: ResultsLike,
    output_path: Union[str, Path],
    condition_names: Optional[Sequence[str]] = None,
    contrast_names: Optional[Sequence[str]] = None,
    contrast_results: Optional[dict] = None,
    volume_shape: Optional[Sequence[int]] = None,
    affine: Optional[np.ndarray] = None,
    voxel_size: Sequence[float] = (2.0, 2.0, 2.0),
    dtype: Union[np.dtype, str] = np.float32,
    apply_afni_metadata: bool = True,  # NEW: Apply 3drefit automatically
) -> Path:
```

### Implementation Details
- **When `apply_afni_metadata=True` (default)**:
  - Check if `3drefit` is available with `shutil.which("3drefit")`
  - Build and run `-relabel_all_str` command from JSON labels
  - Build and run `-substatpar` commands for F-stats and t-stats
  - Use `results.dof` for degrees of freedom
  - Handle errors gracefully (warn if 3drefit fails)
  
- **Benefits**:
  - One-line bucket writing with full AFNI metadata
  - Centralized logic (easier to maintain/fix)
  - Example scripts become much shorter
  - Still works without AFNI installed (just warns)

### Example Usage (MUCH SIMPLER!)
```python
# Old way (50+ lines of 3drefit code)
ffs.write_afni_bucket(results, output_path, ...)
# Then 50 lines of subprocess.run(["3drefit", ...])

# New way (ONE LINE!)
ffs.write_afni_bucket(
    results, 
    output_path, 
    condition_names=stim_labels,
    contrast_names=contrast_names,
    contrast_results=contrast_results,
    apply_afni_metadata=True  # Default, applies 3drefit automatically
)
```

## 3. Add `write_ols_arma_comparison()` Helper Function

### Motivation
Users want to write both OLS and ARMA results for comparison. Make this easy!

### Proposed Function
```python
def write_ols_arma_comparison(
    results: ARMA11Results,
    output_dir: Union[str, Path],
    prefix: str = "glm",
    condition_names: Optional[Sequence[str]] = None,
    contrast_names: Optional[Sequence[str]] = None,
    contrast_results: Optional[dict] = None,
    affine: Optional[np.ndarray] = None,
    apply_afni_metadata: bool = True,
) -> Dict[str, Path]:
    """
    Write both OLS and ARMA(1,1) results for comparison
    
    Creates two bucket files:
    - {prefix}_ols.nii.gz: Uncorrected OLS results
    - {prefix}_arma11.nii.gz: ARMA(1,1) corrected results
    
    Parameters
    ----------
    results : ARMA11Results
        Must have been computed with want_ols=True
    output_dir : Path
        Directory to write files
    prefix : str
        Filename prefix (default: "glm")
    ... other params same as write_afni_bucket ...
    
    Returns
    -------
    paths : dict
        {"ols": Path, "arma11": Path} with output file paths
    """
```

### Implementation
```python
def write_ols_arma_comparison(results, output_dir, prefix="glm", **kwargs):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if results.ols_betas is None:
        raise ValueError("OLS results not available. Run fit_glm_arma11 with want_ols=True")
    
    # Write OLS results
    ols_path = output_dir / f"{prefix}_ols.nii.gz"
    ols_results = _extract_ols_results(results)  # Helper to create OLS-only results object
    write_afni_bucket(ols_results, ols_path, **kwargs)
    
    # Write ARMA results  
    arma_path = output_dir / f"{prefix}_arma11.nii.gz"
    write_afni_bucket(results, arma_path, **kwargs)
    
    # Write comparison report
    comparison_path = output_dir / f"{prefix}_comparison.json"
    _write_comparison_stats(results, comparison_path)
    
    return {"ols": ols_path, "arma11": arma_path, "comparison": comparison_path}
```

## 4. Example Script Simplification

### Before (Current)
```python
# 50+ lines of manual 3drefit commands
# Separate code for OLS (if wanted)
# Manual DoF calculations
# Error handling scattered
```

### After (With Improvements)
```python
# Fit ARMA(1,1) with OLS baseline
results = ffs.fit_glm_arma11(
    data_tensor, 
    design_tensor, 
    tr=tr,
    want_ols=True,  # Get OLS for comparison
    batch_size=500,
)

# Write both OLS and ARMA with full AFNI metadata (ONE FUNCTION CALL!)
output_paths = ffs.write_ols_arma_comparison(
    results,
    output_dir=data_dir,
    prefix="glm",
    condition_names=stim_labels,
    contrast_names=contrast_names,
    contrast_results=contrast_results,
    affine=results.affine,
    apply_afni_metadata=True,  # Automatic 3drefit
)

print(f"✓ OLS results: {output_paths['ols']}")
print(f"✓ ARMA results: {output_paths['arma11']}")
print(f"✓ Comparison: {output_paths['comparison']}")
```

## 5. Implementation Priority

1. **HIGH PRIORITY**: Move 3drefit to library (biggest impact, reduces duplication)
2. **MEDIUM PRIORITY**: Add want_ols parameter (useful for comparisons)
3. **LOW PRIORITY**: Add comparison helper (nice-to-have convenience)

## 6. Backward Compatibility

- All new parameters have sensible defaults
- Existing code continues to work unchanged
- New features opt-in (want_ols=True, apply_afni_metadata=True)

## 7. Testing Needed

- Test 3drefit with/without AFNI installed
- Test OLS computation doesn't slow down ARMA
- Test comparison function with various result types
- Test with masked/unmasked data
- Test with/without contrasts

## 8. Documentation Updates Needed

- Update fit_glm_arma11 docstring
- Update write_afni_bucket docstring
- Add tutorial for OLS vs ARMA comparison
- Update all example scripts to use new features
- Add FAQ: "When do I need OLS comparison?"
