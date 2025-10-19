# Before & After: OLS Comparison Feature

## Memory Management Fix

### Before (OOM on large datasets)
```python
# In fit_glm_arma11() when want_ols=True
ols_results = fit_glm(
    ols_data,
    design,
    tr=tr,
    device=device,
    verbose=False,
    want_residuals=False,
    want_predicted=False,
)
# ❌ No batching! Auto-detects chunk_size
# ❌ May preload all data to GPU
# ❌ OOM on 335k voxels × 3240 TRs
```

### After (Memory efficient)
```python
# In fit_glm_arma11() when want_ols=True
ols_results = fit_glm(
    ols_data,
    design,
    tr=tr,
    device=device,
    verbose=False,
    want_residuals=False,
    want_predicted=False,
    chunk_size=batch_size,  # ✅ Same as ARMA!
    preload_data_to_device=False,  # ✅ Stream data
)
# ✅ Uses same batch_size as ARMA (e.g., 500)
# ✅ Streams data chunk by chunk
# ✅ No OOM even on huge datasets
```

## File Writing Simplification

### Before (~240 lines, duplicated logic)
```python
def write_ols_arma_comparison(
    arma_results: ARMA11Results,
    output_prefix: Union[str, Path],
    condition_names: Optional[Sequence[str]] = None,
    contrast_names: Optional[Sequence[str]] = None,
    contrast_results_arma: Optional[dict] = None,
    contrast_results_ols: Optional[dict] = None,
    volume_shape: Optional[Sequence[int]] = None,
    affine: Optional[np.ndarray] = None,
    voxel_size: Sequence[float] = (2.0, 2.0, 2.0),
    dtype: Union[np.dtype, str] = np.float32,
    apply_afni_metadata: bool = True,
    compress_output: bool = True,
    output_format: Optional[str] = None,
) -> dict:
    # ... 240 lines of duplicated write logic ...
    # Manually handling paths, formats, compression, etc.
    # Copy-pasted from write_afni_bucket() ❌
```

### After (~120 lines, thin wrapper)
```python
def write_ols_arma_comparison(
    arma_results: ARMA11Results,
    output_prefix: Union[str, Path],
    **kwargs,  # ✅ Pass everything through!
) -> dict:
    """Thin wrapper: just calls write_afni_bucket() twice"""
    
    # Extract OLS/ARMA specific contrast results
    contrast_results_ols = kwargs.pop('contrast_results_ols', ...)
    contrast_results_arma = kwargs.pop('contrast_results_arma', ...)
    
    # Construct filenames with _OLS and _ARMA suffixes
    ols_path = f"{prefix}_OLS.nii.gz"
    arma_path = f"{prefix}_ARMA.nii.gz"
    
    # ✅ Call write_afni_bucket() twice (reuse existing code!)
    ols_file = write_afni_bucket(
        ols_results, ols_path,
        contrast_results=contrast_results_ols,
        **kwargs,  # All other args passed through
    )
    
    arma_file = write_afni_bucket(
        arma_results, arma_path,
        contrast_results=contrast_results_arma,
        **kwargs,
    )
    
    # Generate JSON comparison summary
    summary = {...}
    
    return {'ols': ols_file, 'arma': arma_file, 'comparison_summary': ...}
```

## Usage Comparison

### Option 1: Use the wrapper (convenient)
```python
# Wrapper handles filenames and creates JSON summary
outputs = ffs.write_ols_arma_comparison(
    results,
    'outputs/analysis',
    condition_names=['Task', 'Rest'],
    contrast_results_ols=ols_contrasts,
    contrast_results_arma=arma_contrasts,
)

# Creates:
# - outputs/analysis_OLS.nii.gz
# - outputs/analysis_ARMA.nii.gz
# - outputs/analysis_comparison_summary.json
```

### Option 2: Manual (now obvious it's not custom!)
```python
# Just call write_afni_bucket() yourself!
ffs.write_afni_bucket(
    results.ols_results,
    'outputs/analysis_OLS.nii.gz',
    condition_names=['Task', 'Rest'],
    contrast_results=ols_contrasts,
)

ffs.write_afni_bucket(
    results,
    'outputs/analysis_ARMA.nii.gz',
    condition_names=['Task', 'Rest'],
    contrast_results=arma_contrasts,
)

# No JSON summary, but you get full control
```

## Architecture Comparison

### Before: Custom implementation
```
User calls write_ols_arma_comparison()
  ↓
  Custom file writing logic (240 lines)
  ├─ Handle NIfTI format
  ├─ Handle AFNI BRIK format
  ├─ Build bucket structure
  ├─ Apply 3drefit metadata
  ├─ Handle compression
  └─ Write both files
  ↓
Two output files + JSON summary
```
**Problems**: 
- Duplicates all logic from `write_afni_bucket()`
- Hard to maintain (changes needed in 2 places)
- Gives impression file writing is "brittle"

### After: Thin wrapper
```
User calls write_ols_arma_comparison()
  ↓
  Thin wrapper (~50 lines)
  ├─ Construct filenames (_OLS, _ARMA)
  ├─ Call write_afni_bucket() for OLS ───→ Existing robust code
  ├─ Call write_afni_bucket() for ARMA ──→ Existing robust code  
  └─ Generate JSON summary
  ↓
Two output files + JSON summary
```
**Benefits**:
- Reuses robust existing code
- Easy to maintain (1 place to change)
- Shows file writing is NOT brittle
- Obvious what it does

## Type Safety Comparison

### write_afni_bucket() already handles both types!
```python
def write_afni_bucket(
    results: ResultsLike,  # Union[GLMResults, ARMA11Results]
    output_path: str,
    ...
) -> Path:
    # Works with both types - not brittle!
    betas = results.betas  # Both have this
    tstats = results.tstats  # Both have this
    r2 = results.r2  # Both have this
    
    # ARMA-specific (optional)
    if hasattr(results, 'arma_params'):
        arma_params = results.arma_params
    
    # Same code path for both types ✅
```

## Code Reduction

**Before**: ~240 lines in `write_ols_arma_comparison()`
**After**: ~120 lines (50% reduction)
**Eliminated**: All duplicated file writing logic

## What We Learned

1. **File writing is NOT brittle** ✅
   - `write_afni_bucket()` works with both GLMResults and ARMA11Results
   - Type checking with `isinstance()` and `hasattr()`
   - Same interface, optional ARMA-specific fields

2. **Wrappers should be thin** ✅
   - Don't duplicate logic
   - Just modify inputs/outputs and call existing code
   - ~50 lines, not ~200+ lines

3. **Memory management must be explicit** ✅
   - Can't rely on auto-detection for large datasets
   - Pass `chunk_size` explicitly
   - Use `preload_data_to_device=False` for streaming

4. **DRY principle matters** ✅
   - Don't Repeat Yourself
   - Reuse robust existing code
   - Changes propagate automatically
