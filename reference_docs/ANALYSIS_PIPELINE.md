# Analysis Pipeline - Complete Documentation

## Overview

The package is organized around a **core GLM engine** that works with data from any source. Both simulated and real data flow through the same analysis functions, ensuring consistency and GPU acceleration throughout.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Sources                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Simulation                2. AFNI Files    3. Other         │
│     └─ simulate_fmri_run()   └─ afni_io.py    └─ Any format    │
│        Returns torch.Tensor     Returns data     (numpy, etc)  │
│                                                                 │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│               Design Matrix Construction                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  • build_glm_design()    - FIR, assumed HRF, library           │
│  • get_canonical_hrf()   - SPM canonical HRF                   │
│  • get_hrf_library()     - HRF libraries (canonical, FLOBS)    │
│  • read_afni_onset_file() - Convert AFNI onsets to matrix      │
│  • read_afni_design_matrix() - Use pre-built design            │
│                                                                 │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Core GLM Engine                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  • fit_glm()               - Fast OLS GLM (GPU-accelerated)    │
│  • fit_glm_hrf_library()   - Try multiple HRFs per voxel       │
│  • fit_glm_arma11()        - ARMA(1,1) prewhitened GLS         │
│                                                                 │
│  ALL accept torch.Tensor, np.ndarray, or file paths            │
│                                                                 │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Results                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  • GLMResults        - Betas, t-stats, F-stats, R², etc.       │
│  • ARMA11Results     - GLS results with ARMA parameters         │
│  • compute_contrasts() - Custom contrasts on results           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Modules

### Core Analysis (`glm_core.py`, `arma_glm.py`)
- **`fit_glm()`** - Ordinary least squares GLM
- **`fit_glm_arma11()`** - ARMA(1,1) prewhitened generalized least squares
- Both return complete statistical results (betas, t-stats, F-stats, R²)
- Both accept torch.Tensor data directly (GPU-accelerated)

### AFNI I/O (`afni_io.py`)
- **`read_afni_onset_file()`** - Read AFNI timing files
- **`onsets_to_binary_matrix()`** - Convert onset times to binary matrix
- **`read_afni_design_matrix()`** - Parse X.xmat.1D files with full metadata
- **`extract_stimulus_columns()`** - Extract task regressors only
- **`extract_nuisance_columns()`** - Extract polynomials + motion
- **`get_contrast_matrix()`** - Get GLT contrast matrices

### High-Level Workflows (`analysis.py`)
- **`analyze_from_onsets()`** - Complete pipeline: onsets → HRF → design → GLM
- **`analyze_from_design_matrix()`** - Complete pipeline: AFNI matrix → GLM
- **`compute_contrasts()`** - Compute custom contrasts on results

### Design Construction (`design.py`, `hrf.py`)
- **`build_glm_design()`** - Build design matrix (FIR, assumed HRF, library)
- **`get_canonical_hrf()`** - SPM-style double-gamma HRF
- **`get_hrf_library()`** - Generate HRF libraries (canonical, FLOBS)

### Simulation (`simulation.py`, `noise.py`)
- **`simulate_fmri_run()`** - Simulate single run
- **`simulate_fmri_experiment()`** - Simulate full experiment
- **`generate_fmri_noise()`** - Realistic spatiotemporal noise

## Usage Patterns

### Pattern 1: Simulation → Analysis (Direct)

```python
import fastfuncsim as ffs

# Simulate data
hrf = ffs.get_canonical_hrf(stim_duration=2.0, tr=2.0)
onsets = ffs.generate_random_onsets(n_timepoints=200, n_conditions=2, isi_mean=6, tr=2.0)
data = ffs.simulate_fmri_run(onsets, betas=[3.0, 4.0], hrf=hrf, tr=2.0,
                              n_timepoints=200, matrix_size=(20, 20, 5))

# Analyze (data is already torch.Tensor - direct to GLM)
design = ffs.build_glm_design(onsets, hrf, n_timepoints=200, mode='assumed')
data_reshaped = data.reshape(-1, 200)
results = ffs.fit_glm(data_reshaped, design, tr=2.0)
```

**Key insight**: Simulation produces `torch.Tensor` → direct GPU-accelerated GLM

### Pattern 2: AFNI Onsets → Analysis (High-level)

```python
import fastfuncsim as ffs

# Read AFNI onset files and analyze
results = ffs.analyze_from_onsets(
    fmri_data='func.nii.gz',
    onset_files=['onsets_cond1.txt', 'onsets_cond2.txt'],
    tr=2.0,
    hrf_mode='canonical',
    method='ols'
)

print(f"R² = {results.r2.mean():.3f}")
print(f"Beta[0] = {results.betas[:, 0].mean():.3f}")
```

**Key insight**: Same analysis function works with file paths or torch.Tensor

### Pattern 3: AFNI Design Matrix → Analysis

```python
import fastfuncsim as ffs

# Read pre-built AFNI design matrix
results, design_info = ffs.analyze_from_design_matrix(
    fmri_data='func.nii.gz',
    design_matrix_file='X.xmat.1D',
    method='arma11'
)

print(f"Conditions: {design_info['stim_labels']}")
print(f"R² = {results.r2.mean():.3f}")
print(f"Mean ARMA a: {results.arma_params[:, 0].mean():.3f}")
```

**Key insight**: Can use AFNI-generated design matrices directly

### Pattern 4: Simulation → AFNI Files → Analysis (Testing)

```python
import fastfuncsim as ffs

# Simulate and save
data, metadata = ffs.simulate_fmri_experiment(...)
ffs.save_simulation_outputs(data, metadata, output_dir='sim_output')

# Later: analyze as if real data
results, design_info = ffs.analyze_from_design_matrix(
    fmri_data='sim_output/run_01.nii.gz',
    design_matrix_file='sim_output/X.xmat.1D',
    method='arma11'
)
```

**Key insight**: Simulation can write AFNI-compatible files for testing analysis pipelines

## Input Flexibility

All analysis functions accept multiple input types:

```python
# All of these work:
results = ffs.fit_glm(data, design, tr=2.0)

# Where data can be:
data = torch.Tensor(...)           # Direct (fastest)
data = np.ndarray(...)             # Converted to tensor
data = 'func.nii.gz'               # Loaded from file
```

This means:
- ✅ Simulation data (torch.Tensor) → direct to GLM
- ✅ Numpy arrays → converted to tensor → GLM
- ✅ File paths → loaded → converted → GLM

## Function Separation and Modularity

### Low-Level (Maximum control)
```python
# Build each component separately
hrf = ffs.get_canonical_hrf(...)
onsets = ffs.generate_random_onsets(...)
design = ffs.build_glm_design(onsets, hrf, ...)
data = ffs.simulate_fmri_run(...)
results = ffs.fit_glm(data, design, ...)
```

### High-Level (Convenience)
```python
# One function does it all
results = ffs.analyze_from_onsets(
    fmri_data=data,  # Can be tensor, array, or path
    onset_files=['cond1.txt', 'cond2.txt'],
    tr=2.0,
    hrf_mode='canonical',
    method='ols'
)
```

## Organization Benefits

1. **Modularity**: Swap HRF, noise, design matrix independently
2. **GPU Acceleration**: torch.Tensor throughout = fast
3. **Consistency**: Same GLM engine for simulation and real data
4. **Flexibility**: Use low-level or high-level functions as needed
5. **Testing**: Simulate → Save → Analyze (validates whole pipeline)
6. **AFNI Compatibility**: Read/write AFNI formats directly

## Complete Example

See `examples/example_analysis_workflow.py` for a complete demonstration showing:
1. Simulation data → analysis
2. AFNI onset files → analysis
3. AFNI design matrix → analysis

```bash
python examples/example_analysis_workflow.py
```

## Multiple Run Support

FastFuncSim fully supports multiple-run experiments:

- ✅ **Single concatenated file** - All runs in one NIfTI (most common with AFNI)
- ✅ **Multiple run files** - Separate NIfTI per run (auto-concatenated)
- ✅ **RunStart parameter** - Properly interprets AFNI's RunStart for run boundaries
- ✅ **Equal or unequal runs** - Handles both with explicit `run_starts` parameter
- ✅ **Automatic validation** - Checks file lengths match design matrix expectations

See [MULTIPLE_RUNS.md](MULTIPLE_RUNS.md) for complete documentation on multi-run handling.

## Summary

**The organization is smart because:**

- ✅ GLM engine is the core - everything flows through `fit_glm()` or `fit_glm_arma11()`
- ✅ Simulation and real data use the same analysis functions
- ✅ Functions are well-separated (HRF, design, noise, GLM are independent)
- ✅ GPU-accelerated throughout (torch.Tensor from start to finish)
- ✅ Can use low-level functions for control or high-level for convenience
- ✅ AFNI compatibility for real-world workflows
- ✅ Full multi-run support with RunStart validation

**You can push simulation data through the analysis pipeline by:**
1. Using the low-level `fit_glm()` directly with simulation tensors
2. Using the high-level `analyze_from_onsets()` with simulation tensors
3. Saving simulation to AFNI files, then analyzing as "real" data

All three approaches work seamlessly!
