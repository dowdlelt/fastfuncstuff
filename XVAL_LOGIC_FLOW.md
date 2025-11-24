# Cross-Validation Logic and Flow

## Problem: Extremely Negative R² Values

Getting R² values around -500 to -1800, which suggests model predictions are **massively** worse than just using the mean. This document explains the logic to help identify the issue.

---

## High-Level Flow (`compute_xval_r2` in `fastfuncsim/xval.py`)

```
For each CV split (train_runs, test_runs):
    1. Slice data and design by runs (train and test)
    2. Project out nuisance from train data & design
    3. Project out nuisance from test data & design
    4. Extract stimulus design (after projection)
    5. Fit OLS on cleaned train data with cleaned train stimulus design
    6. Predict cleaned test data using train betas and cleaned test stimulus design
    7. Compute R² between cleaned test data and predictions

Aggregate across splits: median, std, min, max
```

---

## Detailed Step-by-Step (from `fastfuncsim/xval.py:372-597`)

### Setup (lines 457-466)
```python
device = get_device()
design_matrix = to_tensor(design_matrix, device=device, dtype=torch.float32)  # Convert to tensor
n_voxels = data.shape[0]
n_splits = len(cv_splits)
```

### Main Loop (lines 511-597)

For each split:

#### 1. Slice by runs (lines 517-522)
```python
train_data, train_design, _ = slice_by_runs(data_chunk, design_matrix, run_starts, train_runs)
test_data, test_design, _ = slice_by_runs(data_chunk, design_matrix, run_starts, test_runs)
```

**What `slice_by_runs` does** (lines 110-170):
- Takes timepoint indices for selected runs
- Returns: `data[:, timepoints]` and `design[timepoints, :]`

#### 2. Compute projection matrices (lines 525-526)
```python
train_P = _compute_projection_matrix(train_design, nuisance_indices)
test_P = _compute_projection_matrix(test_design, nuisance_indices)
```

**What `_compute_projection_matrix` does** (lines 241-296):
- If no nuisance indices, returns `None`
- Otherwise: `P = X_nuisance @ (X_nuisance.T @ X_nuisance)^-1 @ X_nuisance.T`
- This is the **projection onto nuisance space**

#### 3. Project design matrices (lines 529-540)
```python
if train_P is not None:
    train_design_proj = train_design - train_P @ train_design  # Residualize
    train_design_proj = project_out_nuisance(train_design_proj, ...)  # Remove zero columns

if test_P is not None:
    test_design_proj = test_design - test_P @ test_design
    test_design_proj = project_out_nuisance(test_design_proj, ...)
```

**What `project_out_nuisance` does** (lines 173-238):
- Removes columns that are now zero after projection
- Returns clean design matrix with only non-zero columns

#### 4. Extract stimulus design (lines 543-555)
```python
# After projection, figure out which columns are stimulus
stimulus_mask = ...  # Complex logic to identify stimulus columns after projection
train_stim_design = train_design_proj[:, stimulus_mask]
test_stim_design = test_design_proj[:, stimulus_mask]
```

**KEY ISSUE LOCATION?** This logic is complex (lines 543-555)

#### 5. Fit OLS on training data (lines 558-581)
```python
# Process voxels in batches
for batch in batches:
    if train_P is not None:
        train_data_batch = train_data_batch - train_P @ train_data_batch  # Residualize

    # Fit OLS: beta = (X'X)^-1 X'Y
    betas = fit_glm_batch(train_data_batch, train_stim_design)
```

#### 6. Predict test data (lines 583-589)
```python
for batch in batches:
    if test_P is not None:
        test_data_batch = test_data_batch - test_P @ test_data_batch  # Residualize

    # Predict: Y_pred = X_test @ beta_train
    predictions = test_stim_design @ betas
```

#### 7. Compute R² (lines 591-594)
```python
r2 = compute_r2_metric(
    y_true=test_data_batch,  # Actual cleaned test data
    y_pred=predictions,       # Predictions from train model
    metric=metric            # "cod" = coefficient of determination
)
```

**What `compute_r2_metric` does** (lines 299-369) for CoD:
```python
# CoD (coefficient of determination): 1 - SS_res / SS_tot
ss_res = ((y_true - y_pred) ** 2).sum(dim=1)  # Residual sum of squares
ss_tot = ((y_true - y_true.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)  # Total sum of squares
r2 = 1 - ss_res / ss_tot

# If ss_tot is 0 (flat signal), return 0
# Otherwise return r2
```

---

## Potential Issues to Check

### 1. **Data Scaling/Centering**
- Is the data centered? If not, predictions might be way off scale
- Check: Are train and test data on similar scales?
- Location: Lines 558-570, 583-589

### 2. **Nuisance Projection Issue**
- Are we accidentally projecting out ALL variance?
- Is `nuisance_indices` accidentally including stimulus indices?
- Check: What are `stim_indices` and `nuisance_indices` in the test?
- Location: Lines 1062-1072 in `analysis.py`

### 3. **Stimulus Column Identification After Projection**
- After projecting out nuisance, some columns become zero
- The logic to identify which columns are still stimulus might be wrong
- Location: Lines 543-555 in `xval.py`

### 4. **Train/Test Design Matrix Mismatch**
- Are we using the correct test design matrix for prediction?
- Should we be using test_stim_design or something else?
- Location: Lines 586-589

### 5. **R² Calculation**
- Is `y_true` actually centered before computing ss_tot?
- Location: Lines 299-369

---

## Debug Strategy

### Quick Check 1: What are the regressor indices?
```python
# In analyze_with_cross_validation, add:
print(f"stim_indices: {stim_indices}")
print(f"nuisance_indices: {nuisance_indices}")
print(f"Design matrix shape: {design_matrix.shape}")
```

### Quick Check 2: What happens in one split?
```python
# Add debugging in compute_xval_r2 around line 558:
print(f"Train data shape: {train_data.shape}")
print(f"Train stim design shape: {train_stim_design.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Test stim design shape: {test_stim_design.shape}")
print(f"Train data mean: {train_data.mean():.4f}, std: {train_data.std():.4f}")
print(f"Test data mean: {test_data.mean():.4f}, std: {test_data.std():.4f}")
```

### Quick Check 3: Are predictions reasonable?
```python
# Add debugging after line 589:
print(f"Predictions mean: {predictions.mean():.4f}, std: {predictions.std():.4f}")
print(f"Test data mean: {test_data_batch.mean():.4f}, std: {test_data_batch.std():.4f}")
print(f"Prediction error mean: {(test_data_batch - predictions).abs().mean():.4f}")
```

---

## Key Code Locations

1. **Main CV function**: `fastfuncsim/xval.py:372-597` (`compute_xval_r2`)
2. **Projection logic**: `fastfuncsim/xval.py:241-296` (`_compute_projection_matrix`)
3. **Nuisance removal**: `fastfuncsim/xval.py:173-238` (`project_out_nuisance`)
4. **R² computation**: `fastfuncsim/xval.py:299-369` (`compute_r2_metric`)
5. **High-level API**: `fastfuncsim/analysis.py:919-1181` (`analyze_with_cross_validation`)

---

## Current Behavior

With `vis_small_test_r01.nii.gz` + `vis_small_test_r02.nii.gz`:
- Mean R²: **-1853.7**
- Min R²: **-8655.3**
- Max R²: **-1.3**

This is **catastrophically bad** - the model is predicting values that are completely wrong scale/sign.

**Most likely culprits (in order of probability):**
1. Using wrong regressor indices (accidentally using all as nuisance)
2. Not centering/scaling data properly
3. Bug in stimulus column identification after projection
4. Using wrong test design matrix for prediction

---

## **FOUND THE ISSUE!**

The problem is with **run-specific regressors** (polynomials) when `use_stimulus_only=False`.

### The Design Matrix:
- Columns 0-7: `Run#1Pol#0-3`, `Run#2Pol#0-3` (run-specific polynomials)
- Columns 8-9: `movie#0`, `prompt#0` (stimulus)

### What Happens with `use_stimulus_only=False` (default):
```python
stim_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # ALL columns
nuisance_indices = []  # EMPTY - no projection!
```

### The Problem:
1. Train on Run 1, Test on Run 2:
   - Train design has: `Run#1Pol#0-3` (non-zero), `Run#2Pol#0-3` (ZERO!)
   - Test design has: `Run#1Pol#0-3` (ZERO!), `Run#2Pol#0-3` (non-zero)

2. The model learns betas for `Run#1Pol#0-3` from training data

3. When predicting on test data, it tries to use `Run#1Pol#0-3` which are **ALL ZERO**!

4. Result: Predictions are completely wrong → R² = -500

### The Fix:
**Use `use_stimulus_only=True`** to project out run-specific polynomials!

This will:
1. Set `nuisance_indices = [0,1,2,3,4,5,6,7]` (the polynomials)
2. Set `stim_indices = [8,9]` (movie, prompt)
3. Project out nuisance before fitting
4. Only use stimulus columns for prediction

This way, run-specific polynomials are removed, and only cross-run stimulus is used!
