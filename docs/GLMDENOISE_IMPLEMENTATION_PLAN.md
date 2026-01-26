# GLMdenoise Implementation Plan

## Status Overview

| Task | Status |
|------|--------|
| Process ALL voxels (not just criteria) | 🔄 IN PROGRESS |
| Determine criteria AFTER getting full R² maps | TODO |
| Remove duplicate PC nuisance projection | ✅ DONE |
| GLMdenoise-style concatenated predictions | ✅ DONE |
| Select optimal PC from criteria voxels | TODO |

---

## Algorithm Flow (from GLMdenoise paper)

### Step 5: Calculate noise regressors using PCA
> For each run, we extract the time-series of the voxels in the noise pool, **project out the polynomial regressors** from each time-series, **normalize each time-series to unit length**, and perform PCA.

✅ **Our status:** This is already correct in `extract_noise_pcs`. PCs are derived from nuisance-projected, unit-normalized noise pool timeseries. **No need to re-project during CV.**

### Step 6: Enter noise regressors into model; evaluate using cross-validation
> We refit the model to the data, systematically varying the number of noise regressors included in the model. Fitting is performed using leave-one-run-out cross-validation.

✅ **Our status:** We do leave-one-run-out CV with varying PC counts.

> Note that the noise regressors will, in general, have **some correlation with the task-related regressors**. Indeed, the only way that beta weight estimates will change (thereby producing changes in cross-validation performance) **is if there is some correlation between the noise and task regressors**.

🔑 **Key insight:** PCs must have correlation with task design to affect betas. If they're orthogonal, adding PCs won't change task betas (or R²).

### Step 3/6: Cross-validated R² computation
> Model predictions are **concatenated across the left-out runs** and then compared to the data using R².

✅ **Our status:** We accumulate predictions across folds and compute R² once.

### Step 7: Select number of noise regressors
> We first identify voxels that are likely to be related to the experiment. This is done by selecting **all voxels that achieve a cross-validated R² greater than 0% under any of the numbers of noise regressors**.

❌ **Our current bug:** We process only criteria voxels during CV. But criteria should be determined AFTER we have full R² maps for all voxels × all PC counts.

---

## The Critical Fix: Process ALL Voxels

### Current (Wrong) Flow:
```
1. Receive criteria_mask as input
2. Process only criteria voxels in CV
3. Return R² for criteria voxels only
```

### Correct Flow:
```
1. Process ALL voxels in CV (chunked for VRAM efficiency)
2. Get full R² maps: (n_voxels, max_components+1)
3. Determine criteria: any(r2_maps > threshold, axis=1)
4. Select optimal PC count using median R² of criteria voxels
5. Return: full R² maps, criteria mask, optimal_n_pcs
```

---

## Implementation Changes

### Change 1: Modify `cross_validate_noise_pcs` to process ALL voxels

**Before:**
```python
# Use criteria voxels only for memory efficiency during CV
n_criteria = criteria_mask.sum().item()
criteria_indices = torch.where(criteria_mask.to(proj_device))[0]
pred_by_pc = [torch.zeros(n_criteria, n_timepoints, ...) ...]
```

**After:**
```python
# Process ALL voxels (chunked for memory efficiency)
# Criteria will be determined AFTER from the full R² maps
n_process = n_voxels  # ALL voxels
pred_by_pc = [torch.zeros(n_voxels, n_timepoints, ...) ...]
```

### Change 2: Remove duplicate PC projection

PCs already have nuisance projected out during PC extraction step.

### Change 3: Memory-efficient loop structure

Instead of storing all predictions (too large), restructure loops:
- Outer loop: voxel chunks
- Inner loop: folds
- Compute R² per chunk, discard predictions, keep only R² maps

---

## Memory Considerations

### Challenge
- Full predictions: 21 PC counts × 170k voxels × 1800 tps × 4 bytes = ~25 GB (too large!)

### Solution: Chunk-wise R² computation
- R² maps only: 21 × 170k × 4 bytes = ~14 MB (manageable!)

```python
for chunk in voxel_chunks:
    pred_chunk = allocate predictions for this chunk
    for fold in folds:
        for n_pcs in range(max_components + 1):
            predict and accumulate into pred_chunk
    r2_maps[chunk] = compute_r2(pred_chunk, actual_chunk)
    del pred_chunk  # Free memory
```

---

## Files to Modify

1. **`fastfuncsim/denoise.py`**
   - `cross_validate_noise_pcs`: Major refactor for all-voxel processing
   - Add `select_optimal_pcs` function
   - Remove duplicate PC projection code

---

## Previous Analysis (Reference)

### GLMdenoise Algorithm Details

From `GLMestimatemodel.m`:

```matlab
% 1. Construct projection matrix for nuisance + extraregressors
combinedmatrix{p} = projectionmatrix(cat(2,pmatrix,opt.extraregressors{p}));

% 2. Project out nuisance from DATA
data2{p} = combinedmatrix{p}*squish(data{p},dimdata)';

% 3. Project out nuisance from DESIGN
design{p} = combinedmatrix{p}*design{p};

% 4. Fit model
f = mtimescell(olsmatrix2(cat(1,design{:})),data2);

% 5. Predict using fitted model
modelfit(p) = GLMpredictresponses(results{p},{design{p}},tr,...);

% 6. Project polynomials from predictions before R²
modelfit = polymatrix * modelfit;

% 7. Compute R²
results.R2 = calccodcell(modelfit,data,1)';
```

### Our Approach (Correct in Principle)
1. Project nuisance from train data, train design
2. PCs already clean from extraction (no re-projection needed)
3. Fit: X = [task_proj | PCs], Y = data_train_proj
4. Get task betas only (PCs absorb noise variance)
5. Project nuisance from test data, test design
6. Predict: y_pred = X_test_proj @ betas_task
7. R² from concatenated predictions vs actual

### Why R² Might Be Flat
> The only way that beta weight estimates will change is if there is **some correlation between the noise and task regressors**.

If PCs are orthogonal to task design after projection, task betas remain unchanged.

---

## Verification Checklist

- [ ] All voxels processed (not just criteria)
- [ ] Criteria determined from `any(r2 > 0)` across PC counts  
- [ ] PCs not re-projected (already clean from extraction)
- [ ] R² computed from concatenated predictions
- [ ] Memory usage bounded by chunk size
- [ ] Output shapes: `r2_maps` = (n_voxels, 21)

---

## Notes on R² Formula

GLMdenoise uses:
```matlab
R² = 100 * (1 - sum((data-model)^2) / sum(data^2))
```

We use standard R²:
```python
R² = 1 - SSres / SStot
   = 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)
```

These differ but relative ranking should be preserved.

