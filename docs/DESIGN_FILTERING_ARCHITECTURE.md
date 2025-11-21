# Design Matrix Filtering Architecture

## Overview

This document describes the clean architecture for handling filtered vs. full design matrices in fastfuncsim. This was implemented to fix the "322 vs 252" bug where ARMA results had filtered betas but full labels.

## The Problem We Solved

### Before (Buggy):
```python
# In 3dREMLfast.py
results, design_info = analyze_from_design_matrix(...)

# Results were filtered to 252 stimulus columns
assert results.betas.shape[1] == 252

# But we used full labels (322 columns)
write_afni_bucket(results, "output",
                  condition_names=design_info["column_labels"])  # 322 labels! ❌

# ERROR: ValueError: condition_names has length 322 but results have 252 regressors
```

### After (Fixed):
```python
# Extract metadata with clear names
full_labels, stim_labels, stim_indices = extract_design_metadata(design_info)

# Check what was actually fitted
fitted_indices = getattr(results, "fitted_column_indices", None)

if fitted_indices is not None:
    # Use labels matching the fitted columns
    fitted_labels = [full_labels[i] for i in fitted_indices]
else:
    fitted_labels = full_labels

write_afni_bucket(results, "output", condition_names=fitted_labels)  # ✅
```

---

## Clean Naming Convention

### File Paths (always end with `_file`):
- `design_file` - Path to .xmat.1D file

### Full Design (from file, unmodified):
- `full_design_matrix` - np.ndarray shape (n_timepoints, n_regressors_full)
- `full_labels` - List[str] of ALL column labels (length = n_regressors_full)
- `design_info` - Dict with ALL metadata from file

### Stimulus-Only (filtered subset):
- `stim_column_indices` - List[int], indices into full design (e.g., [0, 1, ..., 251])
- `stim_design_matrix` - np.ndarray shape (n_timepoints, n_stim)
- `stim_labels` - List[str] of stimulus column labels only (length = n_stim)

### Fitted Results (what was actually computed):
- `results.betas` - Shape (n_voxels, n_fitted) where n_fitted depends on filtering
- `results.fitted_column_indices` - List[int] or None (which columns from full design were fitted)
- `results.n_regressors_full` - int (original design width before any filtering)
- `fitted_labels` - List[str] matching `results.betas.shape[1]`

---

## Helper Function

Use `extract_design_metadata()` instead of duplicating extraction logic:

```python
from fastfuncsim.afni_io import extract_design_metadata

# One function call replaces ~10 lines of duplicate code
full_labels, stim_labels, stim_column_indices = extract_design_metadata(design_info)

# full_labels: ALL 322 labels from design matrix
# stim_labels: ONLY 252 stimulus labels
# stim_column_indices: [0, 1, ..., 251] indices for stimulus columns
```

---

## Metadata Tracking in Results

### ARMA11Results and GLMResults now track:

```python
class ARMA11Results:
    def __init__(self):
        # ... existing fields ...

        # Design filtering metadata (NEW!)
        self.fitted_column_indices: Optional[List[int]] = None
        # None = all columns fitted
        # List[int] = specific columns fitted (e.g., stimulus only)

        self.n_regressors_full: Optional[int] = None
        # Total columns in original design before filtering
```

### Set in `fit_glm_arma11()`:

```python
def fit_glm_arma11(data, design, tr, task_indices=None, ...):
    # Track original size
    n_regressors_full = design.shape[1]

    # Filter if requested
    if task_indices is not None:
        design = design[:, task_indices]
        fitted_column_indices = task_indices
    else:
        fitted_column_indices = None

    # Store in results
    results.fitted_column_indices = fitted_column_indices
    results.n_regressors_full = n_regressors_full
```

---

## Usage Pattern in Output Functions

### Pattern 1: In `3dREMLfast.py`

```python
# Extract metadata
full_labels, stim_labels, stim_indices = extract_design_metadata(design_info)

# Check what was fitted
fitted_indices = getattr(results, "fitted_column_indices", None)

if fitted_indices is not None:
    # Results were filtered - extract corresponding labels
    fitted_labels = [full_labels[i] for i in fitted_indices]
else:
    # All columns fitted
    fitted_labels = full_labels

# Now write outputs with correct labels
write_afni_bucket(results, output_path, condition_names=fitted_labels)
```

### Pattern 2: In `analysis.py` (OLS callback)

```python
# Extract metadata once
full_labels, stim_labels, stim_indices = extract_design_metadata(design_info)

# OLS callback already receives filtered results
# (task_indices was passed to fit_glm())
def write_ols(ols_results, original_shape, affine):
    write_afni_bucket(
        ols_results,
        output_path,
        condition_names=stim_labels,  # Use stim_labels (already filtered)
    )
```

---

## Testing

### Pytest Suite (`tests/test_label_shape_matching.py`)

Tests ensure:
1. ✅ Results track `fitted_column_indices` correctly
2. ✅ Labels always match `results.betas.shape[1]`
3. ✅ Helper function returns consistent metadata
4. ✅ Single-trials output has correct shape
5. ✅ Regression test for "322 vs 252" bug

Run tests:
```bash
pytest tests/test_label_shape_matching.py -v
```

---

## Before/After Comparison

### Before (Inconsistent):
```python
# In 3dREMLfast.py
stim_indices = []
for bot, top in zip(design_info["stim_bots"], design_info["stim_tops"]):
    stim_indices.extend(range(bot, top + 1))

# In analysis.py (DUPLICATE CODE!)
stim_indices = []
for bot, top in zip(design_info["stim_bots"], design_info["stim_tops"]):
    stim_indices.extend(range(bot, top + 1))

# Results don't track what was fitted
# Have to guess/recompute everywhere
```

### After (Clean):
```python
# ONE helper function
from fastfuncsim.afni_io import extract_design_metadata
full_labels, stim_labels, stim_indices = extract_design_metadata(design_info)

# Results track what was fitted
assert results.fitted_column_indices == stim_indices
assert len(stim_labels) == results.betas.shape[1]
```

---

## Key Files Modified

1. **`fastfuncsim/afni_io.py`**:
   - Added `extract_design_metadata()` helper function

2. **`fastfuncsim/arma_glm.py`**:
   - Filter design early when `task_indices` provided
   - Track metadata in `ARMA11Results`

3. **`bin/3dREMLfast.py`**:
   - Use helper function
   - Extract `fitted_labels` from results metadata
   - Pass correct labels to all output functions

4. **`fastfuncsim/analysis.py`**:
   - Use helper function
   - Simplified OLS callback

5. **`tests/test_label_shape_matching.py`**:
   - Comprehensive test suite for label/shape matching

---

## Benefits

1. **No More Label Mismatches**: Labels always match results shape
2. **No Duplicate Code**: One helper replaces ~20 lines of duplicate logic
3. **Clear Intent**: Variable names make it obvious what they contain
4. **Easy Debugging**: Metadata tracks exactly what was fitted
5. **Type Safe**: Type hints and tests prevent future bugs

---

## Migration Guide

### If you have code that extracts stimulus indices:

**Before:**
```python
stim_indices = []
for bot, top in zip(design_info["stim_bots"], design_info["stim_tops"]):
    stim_indices.extend(range(bot, top + 1))
stim_labels = [design_info["column_labels"][i] for i in stim_indices]
```

**After:**
```python
from fastfuncsim.afni_io import extract_design_metadata
full_labels, stim_labels, stim_indices = extract_design_metadata(design_info)
```

### If you write results to files:

**Before:**
```python
# RISKY - might use wrong labels
write_afni_bucket(results, path, condition_names=design_info["column_labels"])
```

**After:**
```python
# SAFE - uses metadata from results
full_labels, _, _ = extract_design_metadata(design_info)
fitted_indices = getattr(results, "fitted_column_indices", None)
fitted_labels = [full_labels[i] for i in fitted_indices] if fitted_indices else full_labels
write_afni_bucket(results, path, condition_names=fitted_labels)
```

---

## Future Improvements

### Short Term:
- ✅ Add pytest for label/shape matching (DONE)
- ✅ Create helper function (DONE)
- ✅ Consolidate duplicate code (DONE)

### Medium Term:
- Add `results.get_fitted_labels(design_info)` method
- Create `DesignMetadata` dataclass
- Add validation in `write_afni_bucket()` to catch mismatches early

### Long Term:
- Refactor `design_info` dict → typed dataclass
- Add comprehensive type hints throughout
- Consider making `fitted_column_indices` required (not Optional)

---

## Questions?

This architecture was implemented to fix the "ValueError: condition_names has length 322 but results have 252 regressors" bug. If you encounter similar issues:

1. Check that you're using `extract_design_metadata()`
2. Verify `results.fitted_column_indices` matches your expectations
3. Ensure labels match `results.betas.shape[1]`
4. Run the test suite: `pytest tests/test_label_shape_matching.py -v`
