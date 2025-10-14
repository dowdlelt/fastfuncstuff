# Multiple Run Handling in FastFuncSim

## Overview

FastFuncSim fully supports multiple-run fMRI experiments. The `RunStart` parameter from AFNI design matrices is properly interpreted and used to handle runs of equal or unequal length.

## The RunStart Parameter

In AFNI's X.xmat.1D files, the `RunStart` parameter indicates the starting timepoint index for each run:

```
# RunStart = "0,300,600,900"
```

This means:
- Run 1: starts at TR 0 (timepoints 0-299)
- Run 2: starts at TR 300 (timepoints 300-599)
- Run 3: starts at TR 600 (timepoints 600-899)
- Run 4: starts at TR 900 (timepoints 900-1199)

## Two Interpretations

### 1. Single Concatenated File

When you have a single fMRI file with all runs concatenated:

```python
# Data file: func_all_runs.nii.gz (1200 timepoints)
# The file already contains all 4 runs concatenated

results, design_info = ffs.analyze_from_design_matrix(
    'func_all_runs.nii.gz',  # Single file
    'X.xmat.1D',
    method='ols'
)

# RunStart tells us where each run begins in the single file
print(design_info['run_starts'])  # [0, 300, 600, 900]
```

The design matrix is already built correctly for the concatenated data - you just fit the GLM directly.

### 2. Multiple Separate Files

When you have separate files for each run:

```python
# Data files: run01.nii.gz (300 TRs)
#             run02.nii.gz (300 TRs)
#             run03.nii.gz (300 TRs)
#             run04.nii.gz (300 TRs)

results, design_info = ffs.analyze_from_design_matrix(
    ['run01.nii.gz', 'run02.nii.gz', 'run03.nii.gz', 'run04.nii.gz'],
    'X.xmat.1D',
    method='ols'
)

# Files are automatically concatenated
# RunStart from design matrix validates the alignment
```

FastFuncSim will:
1. Load each run file
2. Concatenate them along the time dimension
3. Compute `run_starts` from the actual file lengths: `[0, 300, 600, 900]`
4. Validate these match the `RunStart` from the design matrix
5. Throw an error if there's a mismatch

## Usage Examples

### Example 1: AFNI Onsets with Single Concatenated File

```python
import fastfuncsim as ffs

# Equal-length runs (default behavior)
results = ffs.analyze_from_onsets(
    'func_all_runs.nii.gz',  # 1200 timepoints = 4 runs × 300 TRs
    ['onsets_cond1.txt', 'onsets_cond2.txt'],
    tr=1.5,
    hrf_mode='canonical'
)
# Assumes equal runs: [0, 300, 600, 900]
```

### Example 2: AFNI Onsets with Unequal Runs

```python
# Unequal-length runs - explicitly provide run_starts
results = ffs.analyze_from_onsets(
    'func_all_runs.nii.gz',  # 1150 timepoints
    ['onsets_cond1.txt', 'onsets_cond2.txt'],
    tr=1.5,
    run_starts=[0, 300, 550, 850],  # Runs of length 300, 250, 300, 300
    hrf_mode='canonical'
)
```

### Example 3: AFNI Onsets with Multiple Run Files

```python
# Multiple files - run_starts inferred automatically
results = ffs.analyze_from_onsets(
    ['run01.nii.gz', 'run02.nii.gz', 'run03.nii.gz'],  # Auto-concatenated
    ['onsets_cond1.txt', 'onsets_cond2.txt'],
    tr=1.5,
    hrf_mode='canonical'
)
# run_starts computed from file lengths
```

### Example 4: AFNI Design Matrix with Single File

```python
# Design matrix already has RunStart parameter
results, design_info = ffs.analyze_from_design_matrix(
    'func_all_runs.nii.gz',
    'X.xmat.1D',
    method='ols'
)

print(f"Runs: {len(design_info['run_starts'])}")
print(f"Run starts: {design_info['run_starts']}")
print(f"Run lengths: {ffs.get_run_lengths(design_info['run_starts'], design_info['n_timepoints'])}")
```

### Example 5: AFNI Design Matrix with Multiple Files

```python
# Multiple files - validates against design matrix
results, design_info = ffs.analyze_from_design_matrix(
    ['run01.nii.gz', 'run02.nii.gz', 'run03.nii.gz', 'run04.nii.gz'],
    'X.xmat.1D',
    method='arma11'
)

# Automatic validation:
# - Number of files must match number of runs in design matrix
# - Run lengths from files must match run lengths from RunStart
# - Raises ValueError if mismatch detected
```

## Onset File Format

AFNI onset files have **one row per run**:

```
# onsets_condition1.txt
0.00 6.00 10.00 16.00 28.00 ...    # Run 1 onsets (in seconds)
0.00 10.00 18.00 34.00 42.00 ...   # Run 2 onsets (in seconds)
0.00 14.00 26.00 40.00 48.00 ...   # Run 3 onsets (in seconds)
```

When converting to binary onset matrix:
- If `run_starts` is provided → use those indices
- If `run_starts` is `None` → assume equal-length runs

## Internal Functions

### `onsets_to_binary_matrix()`

```python
onsets = ffs.onsets_to_binary_matrix(
    onsets_per_condition,  # List[List[np.ndarray]]
    n_timepoints=1200,
    tr=1.5,
    run_starts=[0, 300, 600, 900],  # Optional
    device=device
)
```

Converts onset times (in seconds) to binary matrix with proper run alignment.

### `load_and_concatenate_runs()`

```python
data, run_starts = ffs.load_and_concatenate_runs(
    ['run01.nii.gz', 'run02.nii.gz', 'run03.nii.gz']
)
# Returns:
#   data: (n_voxels, total_timepoints)
#   run_starts: [0, 300, 600]  # Inferred from file lengths
```

Loads multiple NIfTI files and concatenates them, computing `run_starts` automatically.

### `get_run_lengths()`

```python
run_starts = [0, 300, 600, 900]
run_lengths = ffs.get_run_lengths(run_starts, n_timepoints=1200)
# Returns: [300, 300, 300, 300]
```

Computes the length of each run from the start indices.

### `read_afni_design_matrix()`

```python
design_info = ffs.read_afni_design_matrix('X.xmat.1D')

print(design_info['run_starts'])    # [0, 300, 600, 900]
print(design_info['n_timepoints'])  # 1200
print(design_info['tr'])            # 1.5
```

Parses AFNI design matrix including the `RunStart` parameter.

## Error Handling

### Mismatched Number of Runs

```python
# Error: 3 files but design matrix has 4 runs
results, design_info = ffs.analyze_from_design_matrix(
    ['run01.nii.gz', 'run02.nii.gz', 'run03.nii.gz'],  # 3 files
    'X.xmat.1D',  # RunStart = "0,300,600,900" (4 runs)
    method='ols'
)
# Raises: ValueError("Number of run files (3) doesn't match number of runs in design matrix (4)")
```

### Mismatched Run Lengths

```python
# Error: Files have different lengths than design matrix
results, design_info = ffs.analyze_from_design_matrix(
    ['run01.nii.gz', 'run02.nii.gz', 'run03.nii.gz', 'run04.nii.gz'],
    # run01.nii.gz has 250 TRs, but design matrix expects 300
    'X.xmat.1D',  # RunStart = "0,300,600,900" → lengths [300, 300, 300, 300]
    method='ols'
)
# Raises: ValueError("Run lengths from files [250, 300, 300, 300] don't match ...")
```

## Best Practices

1. **Single Concatenated File**: Simplest approach if you already have concatenated data
   - Design matrix `RunStart` is used for bookkeeping only
   - No validation needed

2. **Multiple Run Files**: Use when you have separate run files
   - Safer - automatic validation against design matrix
   - Catches alignment errors early

3. **Unequal Runs**: Always provide explicit `run_starts` parameter
   - Don't rely on equal-length assumption
   - Prevents misalignment errors

4. **Check Your Data**: Use `get_run_lengths()` to verify alignment
   ```python
   design_info = ffs.read_afni_design_matrix('X.xmat.1D')
   expected_lengths = ffs.get_run_lengths(design_info['run_starts'], design_info['n_timepoints'])
   print(f"Expected run lengths: {expected_lengths}")
   ```

## Simulation → Analysis Workflow

When simulating multi-run experiments:

```python
# Simulate 4 runs
data_list = []
for run_idx in range(4):
    data = ffs.simulate_fmri_run(...)
    data_list.append(data)

# Save as separate files
for i, data in enumerate(data_list):
    ffs.write_nifti_files(data, f'run{i+1:02d}.nii.gz')

# Analyze (files will be concatenated automatically)
results = ffs.analyze_from_onsets(
    ['run01.nii.gz', 'run02.nii.gz', 'run03.nii.gz', 'run04.nii.gz'],
    ['onsets_cond1.txt', 'onsets_cond2.txt'],
    tr=2.0,
    hrf_mode='canonical'
)
```

## Summary

✅ **RunStart is properly handled**:
- Single file: Used to understand run boundaries
- Multiple files: Used to validate file lengths match expected

✅ **Both equal and unequal length runs supported**:
- Equal: Infer from total timepoints / number of runs
- Unequal: Explicitly provide `run_starts` parameter

✅ **Automatic validation**:
- File count vs design matrix run count
- File lengths vs design matrix run lengths
- Clear error messages if mismatch

✅ **Flexible input**:
- Single concatenated file (most common with AFNI)
- List of separate run files (safer, auto-validated)
- torch.Tensor (for simulation workflows)
