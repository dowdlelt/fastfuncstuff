# FastFuncSim - Modular Analysis Quick Reference

## New Library Functions (Refactoring Update)

### 1. `slice_glm_results(results, indices)`
**Extract subset of regressors from GLM results**

```python
import fastfuncsim as ffs

# Slice to stimulus regressors only
design_info = ffs.read_afni_design_matrix('X.xmat.1D')
stim_indices = design_info['stim_bots']
results_stim = ffs.slice_glm_results(results, stim_indices)

# Works with both OLS and ARMA
ols_stim = ffs.slice_glm_results(results.ols_results, stim_indices)

# Or extract specific regressors by index
face_results = ffs.slice_glm_results(results, [0, 1, 2])  # First 3 regressors
```

**What it does:**
- Creates new results object with only selected regressors
- Works with `GLMResults` (OLS) and `ARMA11Results`
- Properly handles covariance matrices (slices both dimensions)
- Copies spatial metadata (affine, mask, shape)
- Recursively slices embedded OLS results

---

### 2. `compute_contrasts_from_design(results, design_info, ...)`
**Auto-extract and compute GLT contrasts from design matrix**

```python
import fastfuncsim as ffs

# Read design matrix
results, design_info = ffs.analyze_from_design_matrix(
    run_files, 'X.xmat.1D', method='arma11', want_ols=True
)

# Compute contrasts (auto-extracted from design matrix)
contrast_results_arma = ffs.compute_contrasts_from_design(
    results, design_info,
    auto_cpu_fallback=True,  # Smart: uses CPU for large datasets
)

# Also for OLS
contrast_results_ols = ffs.compute_contrasts_from_design(
    results.ols_results, design_info
)

# Check what was computed
if contrast_results_arma:
    print(f"Computed {len(design_info['glt_labels'])} contrasts:")
    for name in design_info['glt_labels']:
        print(f"  - {name}")
```

**What it does:**
- Extracts GLT matrices from `design_info['glt_matrices']`
- Returns `None` if no contrasts defined (graceful)
- Automatically uses CPU for large datasets (>1000 timepoints)
- Works with both OLS (xtx_inv) and ARMA (var_betas)
- Customizable memory threshold

---

## Complete Workflow Examples

### Example 1: Basic ARMA Analysis
```python
import fastfuncsim as ffs
from pathlib import Path

# 1. Fit model
results, design_info = ffs.analyze_from_design_matrix(
    run_files=['run01.nii.gz', 'run02.nii.gz'],
    design_matrix_file='X.xmat.1D',
    method='arma11',
    mask_file='mask.nii.gz',
)

# 2. Compute contrasts
contrasts = ffs.compute_contrasts_from_design(results, design_info)

# 3. Write output
ffs.write_afni_bucket(
    results,
    'glm_results.nii.gz',
    condition_names=design_info['column_labels'],
    contrast_names=design_info['glt_labels'],
    contrast_results=contrasts,
)

print("Done!")
```

---

### Example 2: OLS vs ARMA Comparison
```python
import fastfuncsim as ffs

# 1. Fit both models
results, design_info = ffs.analyze_from_design_matrix(
    run_files, 'X.xmat.1D',
    method='arma11',
    want_ols=True,  # Get OLS baseline
)

# 2. Compute contrasts for both
contrasts_arma = ffs.compute_contrasts_from_design(results, design_info)
contrasts_ols = ffs.compute_contrasts_from_design(results.ols_results, design_info)

# 3. Write side-by-side comparison
outputs = ffs.write_ols_arma_comparison(
    results,
    'glm_comparison',  # Creates glm_comparison_OLS.nii.gz and glm_comparison_ARMA.nii.gz
    condition_names=design_info['column_labels'],
    contrast_names=design_info['glt_labels'],
    contrast_results_ols=contrasts_ols,
    contrast_results_arma=contrasts_arma,
)

print(f"OLS:  {outputs['ols']}")
print(f"ARMA: {outputs['arma']}")
print(f"Summary: {outputs['comparison_summary']}")
```

---

### Example 3: Separate Stimulus vs Nuisance
```python
import fastfuncsim as ffs

# 1. Fit model
results, design_info = ffs.analyze_from_design_matrix(
    run_files, 'X.xmat.1D', method='arma11'
)

# 2. Slice by regressor type
stim_indices = design_info['stim_bots']
results_stim = ffs.slice_glm_results(results, stim_indices)

all_indices = set(range(design_info['n_regressors']))
nuisance_indices = sorted(all_indices - set(stim_indices))
results_nuisance = ffs.slice_glm_results(results, nuisance_indices)

# 3. Compute contrasts (stimulus only)
contrasts = ffs.compute_contrasts_from_design(results_stim, design_info)

# 4. Write separate files
ffs.write_afni_bucket(
    results_stim, 'glm_stimulus.nii.gz',
    condition_names=design_info['stim_labels'],
    contrast_names=design_info['glt_labels'],
    contrast_results=contrasts,
)

nuisance_labels = [design_info['column_labels'][i] for i in nuisance_indices]
ffs.write_afni_bucket(
    results_nuisance, 'glm_nuisance.nii.gz',
    condition_names=nuisance_labels,
)
```

---

### Example 4: Fast Re-Analysis with Cached ARMA Parameters
```python
import fastfuncsim as ffs

# First run (slow - estimates ARMA parameters)
results, design_info = ffs.analyze_from_design_matrix(
    run_files, 'X.xmat.1D',
    method='arma11',
    mask_file='mask.nii.gz',
)

# Save ARMA parameters
ffs.save_arma_rvar(
    results, 'arma_params.nii.gz',
    volume_shape=results.full_shape,
    voxel_mask=results.voxel_mask,
    affine=results.affine,
)

# Second run (fast - 80% speedup!)
mask = ffs.load_afni_mask('mask.nii.gz')
arma_params = ffs.load_arma_params('arma_params.nii.gz', voxel_mask=mask)

results_fast, design_info = ffs.analyze_from_design_matrix(
    run_files, 'X.xmat.1D',
    method='arma11',
    precomputed_arma_params=arma_params,  # Skip REML estimation!
)
```

---

## Function Parameters Reference

### `slice_glm_results(results, indices)`
- `results`: GLMResults or ARMA11Results
- `indices`: list/array of regressor indices to keep (0-indexed)
- **Returns**: New results object with only selected regressors

### `compute_contrasts_from_design(results, design_info, **kwargs)`
- `results`: GLMResults or ARMA11Results
- `design_info`: dict from `read_afni_design_matrix()`
- `device`: torch.device (optional)
- `auto_cpu_fallback`: bool, default=True
- `memory_threshold_timepoints`: int, default=1000
- **Returns**: dict with 'contrast_betas', 'contrast_tstats', 'contrast_stderr', or None

---

## Common Patterns

### Pattern 1: Load → Fit → Contrast → Write
```python
# Load
results, design_info = ffs.analyze_from_design_matrix(...)

# Contrast
contrasts = ffs.compute_contrasts_from_design(results, design_info)

# Write
ffs.write_afni_bucket(results, 'output.nii.gz', contrast_results=contrasts)
```

### Pattern 2: Slice → Process → Write
```python
# Slice
results_subset = ffs.slice_glm_results(results, indices)

# Process
contrasts = ffs.compute_contrasts_from_design(results_subset, design_info)

# Write
ffs.write_afni_bucket(results_subset, 'output.nii.gz', ...)
```

### Pattern 3: Fit OLS+ARMA → Compare
```python
# Fit both
results, design_info = ffs.analyze_from_design_matrix(..., want_ols=True)

# Contrast both
c_arma = ffs.compute_contrasts_from_design(results, design_info)
c_ols = ffs.compute_contrasts_from_design(results.ols_results, design_info)

# Compare
ffs.write_ols_arma_comparison(results, 'compare', 
                               contrast_results_arma=c_arma,
                               contrast_results_ols=c_ols)
```

---

## Tips & Best Practices

### 1. **Always use library functions**
❌ Don't write helper functions in user scripts  
✅ Use `ffs.slice_glm_results()` instead

### 2. **Let library handle complexity**
❌ Don't manually check dataset size for CPU fallback  
✅ Use `auto_cpu_fallback=True` (default)

### 3. **Cache ARMA parameters**
❌ Don't re-estimate ARMA parameters every run  
✅ Save with `ffs.save_arma_rvar()`, load with `ffs.load_arma_params()`

### 4. **Check for contrasts gracefully**
```python
contrasts = ffs.compute_contrasts_from_design(results, design_info)
if contrasts:
    # Use contrasts
else:
    # No contrasts defined in design matrix
```

### 5. **Compose functions**
```python
# Each line does ONE thing
results, design_info = ffs.analyze_from_design_matrix(...)
results_stim = ffs.slice_glm_results(results, design_info['stim_bots'])
contrasts = ffs.compute_contrasts_from_design(results_stim, design_info)
ffs.write_afni_bucket(results_stim, 'output.nii.gz', contrast_results=contrasts)
```

---

## Migration from Old Scripts

### Old Way (729 lines):
```python
# Define 57-line helper function
def slice_results(results, indices):
    # ... 57 lines of copying attributes ...

# Manually extract contrasts
contrasts = torch.zeros(...)
for i, glt in enumerate(design_info['glt_matrices']):
    # ... manual parsing ...

# Manual CPU fallback
if n_timepoints > 1000:
    device = 'cpu'
    # ... manual device management ...

# Slice manually
results_stim = slice_results(results, stim_indices)
```

### New Way (~200 lines):
```python
# Use library functions
results_stim = ffs.slice_glm_results(results, stim_indices)
contrasts = ffs.compute_contrasts_from_design(results, design_info, auto_cpu_fallback=True)
```

**Benefits:**
- 73% less code
- No helper functions
- Automatic optimization
- Tested and reusable

---

## Getting Help

### Check function documentation:
```python
help(ffs.slice_glm_results)
help(ffs.compute_contrasts_from_design)
```

### See examples:
- `examples/analyze_taskforce_ses02_clean.py` - Complete workflow
- `examples/analyze_real_data_modular.py` - General template
- `examples/REFACTORING_SUMMARY.md` - Before/after comparison

### Philosophy:
**"Code once, use a billion times"**
- Write general functions in library
- Compose them in user scripts
- Each function does ONE thing well
