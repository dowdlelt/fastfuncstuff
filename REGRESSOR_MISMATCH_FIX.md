# Fix: Regressor Count Mismatch in OLS/ARMA Comparison

## Problem
```
ValueError: condition_names has length 41 but results have 121 regressors
```

## Root Cause
- ARMA results were sliced to 41 stimulus regressors: `results_main = slice_results(results, stim_indices)`
- OLS results still had all 121 regressors (not sliced)
- Passed only 41 condition names (`stim_labels`)
- Mismatch: OLS has 121 regressors but only 41 names provided

## Solution
Slice OLS results too, before passing to comparison writer:

```python
# Slice ARMA results to stimulus only
stim_indices = design_info["stim_bots"]  # [32, 33, ..., 72] (41 indices)
results_main = slice_results(results, stim_indices)

# Slice OLS results to match
ols_results_main = slice_results(results.ols_results, stim_indices)

# Create comparison object with sliced results
results_for_comparison = results_main
results_for_comparison.ols_results = ols_results_main

# Now both have 41 regressors!
ffs.write_ols_arma_comparison(
    results_for_comparison,
    'glm_main',
    condition_names=stim_labels,  # 41 names for 41 regressors ✓
    ...
)
```

## Why This Happened
You have two separate output files:
1. **Main effects** (`glm_main_*.nii.gz`) - 41 stimulus regressors
2. **Nuisance** (`glm_nuisance_*.nii.gz`) - 80 nuisance regressors

The comparison writer needs to write BOTH OLS and ARMA for the same subset of regressors.

## What Gets Written Now

### `glm_main_OLS.nii.gz` (41 stimulus regressors only)
- Sub-brick [0]: Full F-stat
- Sub-bricks [1-82]: 41 beta/tstat pairs for stimulus regressors
- Sub-bricks [83+]: GLT contrast beta/tstat pairs

### `glm_main_ARMA.nii.gz` (41 stimulus regressors only)  
- Sub-brick [0]: Full F-stat
- Sub-bricks [1-82]: 41 beta/tstat pairs for stimulus regressors
- Sub-bricks [83+]: GLT contrast beta/tstat pairs

### `glm_main_comparison_summary.json`
- Statistics comparing OLS vs ARMA for the 41 stimulus regressors
- R² improvement, t-stat ratio, beta correlation
- ARMA parameter recovery (a, b, λ)

## Alternative Approach (If You Want Full Results)

If you wanted to write ALL 121 regressors in the comparison:

```python
# Don't slice - use full results
outputs = ffs.write_ols_arma_comparison(
    results,  # Full 121 regressors
    'glm_full',
    condition_names=all_labels,  # All 121 names
    ...
)
```

But typically you want to separate stimulus and nuisance, so the current approach is correct!

## Summary

✅ **Fixed**: Both OLS and ARMA results now sliced to same 41 stimulus regressors
✅ **Matches**: 41 condition names for 41 regressors in both OLS and ARMA
✅ **Correct**: Comparing apples-to-apples (same regressors in both methods)

Your script should now run successfully! 🎉
