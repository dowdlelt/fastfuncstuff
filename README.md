# FastFuncSim

**Fast GPU-Accelerated Functional MRI Simulation and GLM Fitting**

Convert from MATLAB (`simulate_movietasks.m` and GLMsingle) to Python with GPU acceleration for both interactive exploration and large-scale batch simulations.

## Key Features

- **🚀 Ultra-Fast GLM Solver**: GPU-accelerated core engine handles FIR, assumed HRF, and HRF library approaches
- **🔄 Dual Mode**: Interactive single simulations (<10s) AND batch thousands of simulations
- **🎯 Flexible Design**: FIR (no HRF assumption), assumed HRF, or HRF library selection
- **💻 Device Agnostic**: Auto-detects and uses MPS (Apple Silicon) → CUDA → CPU
- **📊 Realistic Noise**: 1/f spectrum with physiological components (cardiac, respiratory)
- **📈 Comprehensive Visualization**: Single-case deep dives and batch summaries with parametric exploration
- **🎨 Complete Pipeline**: From simulation to GLM fitting to visualization

## Installation

```bash
cd /Users/logan/local_bin/fastfuncsim
pip install -e .
```

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0 (with MPS/CUDA support recommended)
- NumPy, SciPy
- Matplotlib (for examples)

## Quick Start

### 1. Interactive Single Simulation

```python
import fastfuncsim as ffs

# Auto-detect best device (MPS/CUDA/CPU)
device = ffs.get_device()

# Generate HRF
hrf = ffs.get_canonical_hrf(stim_duration=5.0, tr=1.0, device=device)

# Generate task design
onsets = ffs.generate_random_onsets(
    n_timepoints=290, n_conditions=2,
    isi_mean=4, tr=1.0, device=device
)

# Simulate fMRI data
data = ffs.simulate_fmri_run(
    onsets=onsets,
    betas=[5.0, 3.0],  # Effect sizes for each condition
    hrf=hrf,
    tr=1.0,
    n_timepoints=290,
    matrix_size=(50, 50, 5),  # nx, ny, nz
    device=device
)

# Fit GLM with assumed HRF
design = ffs.build_glm_design(onsets, hrf, 290, mode='assumed', device=device)
results = ffs.fit_glm(data, design, tr=1.0, device=device)

print(f"Mean R² = {results.r2.mean():.3f}")
```

### 2. FIR Estimation (No HRF Assumption)

```python
# FIR design: 30 time lags per condition
design_fir = ffs.build_glm_design(onsets, mode='fir', n_fir_lags=30,
                                   n_timepoints=290, device=device)

results_fir = ffs.fit_glm(data, design_fir, tr=1.0, device=device)

# Extract estimated HRF for condition 1, best voxel
best_voxel = torch.argmax(results_fir.r2)
fir_hrf = results_fir.betas[best_voxel, :30]  # First 30 lags
```

### 3. HRF Library (GLMsingle Approach)

```python
# Create library of 20 HRF candidates
hrf_library = ffs.get_hrf_library('canonical', stim_duration=5.0,
                                   tr=1.0, n_hrfs=20, device=device)

# Fit GLM with each HRF, select best per voxel
results, hrf_index, r2_all = ffs.fit_glm_hrf_library(
    data, onsets, hrf_library, tr=1.0, device=device
)

# hrf_index[voxel] = index of best HRF for that voxel
# r2_all[voxel, hrf] = R² for each HRF candidate
```

### 4. Batch Simulations (Power Analysis)

```python
# Run 1000 simulations to estimate statistical power
results_batch = []

for i in range(1000):
    # Generate new onsets and noise each iteration
    onsets = ffs.generate_random_onsets(...)
    data = ffs.simulate_fmri_run(onsets, betas=[effect_size], ...)
    design = ffs.build_glm_design(onsets, hrf, ...)
    results = ffs.fit_glm(data, design, tr=1.0, verbose=False)

    results_batch.append({
        'mean_r2': results.r2.mean().item(),
        'mean_beta': results.betas.mean().item(),
    })

# Analyze power across simulations
mean_r2 = np.mean([r['mean_r2'] for r in results_batch])
power = np.mean([r['mean_r2'] > threshold for r in results_batch])
```

## Architecture

```
fastfuncsim/
├── glm_core.py      # Core GLM engine (THE KEY COMPONENT)
├── glm_outputs.py  # NIfTI export utilities for GLM results
├── design.py        # Design matrix construction (FIR, HRF convolution)
├── hrf.py           # HRF generation (canonical, FLOBS, libraries)
├── noise.py         # Realistic fMRI noise generation
├── simulation.py    # Simulation pipeline
├── utils.py         # Device management, helpers
├── examples/        # Ready-to-run analysis and simulation scripts
└── __init__.py      # Package interface
```

### Core Philosophy

**Everything feeds into the fast GLM solver.** Whether you're using:
- **FIR**: Design matrix is shifted impulses
- **Assumed HRF**: Design matrix is onsets ⊗ HRF
- **HRF Library**: Run GLM with each HRF, pick best R²

All use the same GPU-accelerated solver: `β = (X'X)⁻¹X'Y`

## Examples

Run the example scripts:

```bash
# Interactive single simulation
python examples/example_single.py

# Batch power analysis
python examples/example_batch.py
```

## GLM Fitting Details

The core GLM solver (`fit_glm`) supports:

- **Polynomial detrending**: Auto-computed or specified per run
- **Extra regressors**: Motion parameters, GLMdenoise PCs, etc.
- **Multi-run data**: Concatenates runs, handles run-specific nuisance
- **Chunked processing**: Automatically chunks voxels to fit GPU memory
- **Per-run R²**: Computes R² separately for each run
- **Percent BOLD**: Converts betas to % signal change

### GLM Options

```python
results = ffs.fit_glm(
    data=data_list,           # List of runs or single array
    design=design_list,       # Design matrices per run
    tr=1.0,
    max_poly_degree=3,        # Polynomial detrending (or list per run)
    extra_regressors=None,    # Additional nuisance regressors
    want_residuals=False,     # Return residuals (memory intensive)
    want_predicted=False,     # Return predicted timecourses
    want_r2_run=True,         # Compute per-run R²
    device=device,
    chunk_size=None,          # Auto-compute optimal chunk size
    verbose=True
)

# Results object contains:
# - betas: (n_voxels, n_regressors) or spatial shape
# - r2: (n_voxels,) R² values
# - r2_run: (n_voxels, n_runs) per-run R²
# - meanvol: Mean signal per voxel
# - tstats: t-statistics per regressor (correct for OLS or ARMA)
# - fstats: Omnibus F-statistic across task regressors
# - sigma2: Residual variance estimates per voxel
# - residuals, predicted: Optional (original data space)
# - residuals_whitened: Optional (GLS-prewhitened residuals for diagnostics)
```

### Exporting GLM Outputs

FastFuncSim now writes publication-ready statistical maps in AFNI-style stacks:

```python
results = ffs.fit_glm(data, design, tr=1.0, want_residuals=True)

ffs.write_glm_results_nifti(
    results,
    output_dir="./glm_outputs",
    prefix="subject01",
    condition_names=["stim_A", "stim_B"],
    include_beta=True,
    include_tstat=True,
    include_fstat=True,   # omnibus across all task regressors
    include_r2=True,
    include_mean=True,
    include_sigma=False,
    write_residuals=False,
)
```

The resulting `subject01_stats.nii.gz` packs **Beta** and **t-stat** pairs per condition along
the 4th dimension (AFNI bucket style), while `subject01_fstat.nii.gz` stores the overall
model F-statistic. Optional sidecar maps include R², mean signal, and residual standard
deviation. A JSON manifest (`subject01_stats.json`) records the ordering of volumes.

## HRF Generation

### Canonical HRF (SPM-style double gamma)
```python
hrf = ffs.get_canonical_hrf(stim_duration=5.0, tr=1.0)
```

### Canonical HRF Library (parameter variations)
```python
library = ffs.get_canonical_hrf_library(stim_duration=5.0, tr=1.0, n_hrfs=20)
```

### FLOBS (Flexible half-cosine basis)
```python
library, params = ffs.create_flobs_library(
    n_hrfs=20,
    m1_range=(0, 2),      # Delay before rise
    m2_range=(3, 8),      # Time to peak
    m3_range=(3, 10),     # Time to undershoot
    m4_range=(3, 12),     # Recovery time
    c2_range=(0, 0.35),   # Undershoot magnitude
    tr=1.0
)
```

## Noise Generation

Realistic fMRI noise with:
- **1/f (pink) spectrum**: Matches real fMRI temporal correlations
- **Respiratory component**: ~0.3 Hz peak
- **Cardiac component**: ~1.0 Hz peak
- **Independent per voxel**: Each voxel has unique noise realization

```python
noise = ffs.generate_fmri_noise(
    tr=1.0,
    duration_s=290,
    matrix_size=(100, 100),  # 2D slice
    resp_freq=0.35,
    cardiac_freq=1.0,
    pink_exp=1.0,           # 1/f exponent
    normalize=True,
    device=device
)
```

## Performance

Approximate timings on M1 Max (MPS):
- **Single simulation** (50×50×5, 290 TRs, 4 runs): ~5-10s
- **GLM fit** (same data, assumed HRF): ~2-3s
- **GLM fit** (FIR, 30 lags): ~5-8s
- **HRF library** (20 HRFs): ~30-60s
- **Batch (100 sims)**: ~10-15 min

On CUDA GPU, expect 2-5x faster for large batches.

## Comparison to MATLAB

| Feature | MATLAB | FastFuncSim |
|---------|--------|-------------|
| HRF Library | ✓ (GLMsingle) | ✓ (canonical + FLOBS) |
| FIR Estimation | ✓ (GLMdenoise) | ✓ |
| Single Trial | ✓ | ✓ |
| GPU Acceleration | ✗ | ✓ (MPS/CUDA) |
| Batch Mode | Manual loops | Optimized |
| Memory Efficient | Chunking | Auto-chunking |
| Speed (single) | ~30-60s | ~5-10s |
| Speed (batch 100) | ~1-2 hrs | ~10-15 min |

## Advanced Usage

### Custom HRF
```python
# Define your own HRF
custom_hrf = torch.tensor([0, 0.2, 0.5, 1.0, 0.8, 0.3, -0.1, -0.05, 0])
design = ffs.build_glm_design(onsets, custom_hrf, n_timepoints, mode='assumed')
```

### Multi-run with Different Designs
```python
# Different onsets per run
onsets_list = [generate_design_run1(), generate_design_run2(), ...]
designs = [ffs.build_glm_design(o, hrf, n_tp, mode='assumed')
           for o, n_tp in zip(onsets_list, n_timepoints_list)]

results = ffs.fit_glm(data_list, designs, tr=1.0)
```

### Real Data (not simulated)
```python
# Load your real fMRI data
data = load_nifti('sub-01_task-rest_bold.nii.gz')  # (nx, ny, nz, nt)

# Create design from your paradigm
onsets = load_onsets_from_tsv('events.tsv')
hrf = ffs.get_canonical_hrf(stim_duration=2.0, tr=2.0)
design = ffs.build_glm_design(onsets, hrf, data.shape[-1], mode='assumed')

# Fit GLM
results = ffs.fit_glm(data, design, tr=2.0, device='cuda')
```

## Visualization

FastFuncSim provides comprehensive visualization tools for both **single-case exploration** and **batch summaries**.

### Single-Case Deep Dive
```python
# Deep exploration of individual simulation
fig = ffs.plot_simulation_deep_dive(
    data=data,
    design=design,
    results=results,
    betas_true=true_betas,
    hrf_true=true_hrf,
    voxel_selection='best',  # or 'worst', 'median', 'random'
    n_voxels=4,
    tr=1.0,
    save_path='deep_dive.png'
)
```

Shows:
- Observed vs predicted timecourses
- Residuals
- Beta estimates vs true values
- R² distribution
- Summary statistics

### Batch Summary
```python
# Statistical summary across many simulations
fig = ffs.plot_batch_summary(
    results_list=results_list,  # List of dicts with metrics
    metrics=['r2', 'beta_error', 'hrf_recovery', 'power'],
    group_by='effect_size',  # Group by any variable
    save_path='batch_summary.png'
)
```

### Parametric Exploration (3-Axis)
```python
# Explore parameter space: magnitudes × HRFs × noise
fig = ffs.plot_parametric_exploration(
    results_grid=results_grid,  # {z_val: {y_val: {x_val: metrics}}}
    x_var='beta_ratio',
    y_var='hrf_index',
    z_var='noise_level',
    metric='r2_mean',
    save_path='parametric.png'
)
```

Flexible across **any number of conditions**, **any ordering**, **any magnitude** of effects.

### HRF Recovery Analysis
```python
# Evaluate FIR estimation quality
fig = ffs.plot_hrf_recovery(
    hrf_estimated=fir_betas,
    hrf_true=true_hrf,
    tr=1.0,
    save_path='hrf_recovery.png'
)
```

### More Visualization Tools
- `plot_design_comparison()` - Compare multiple design matrices
- `create_interactive_summary_html()` - Generate interactive HTML report

**See `VISUALIZATION_GUIDE.md` for complete documentation and examples.**

## Contributing

This is a research tool converted from MATLAB. Contributions welcome!

- Report issues: GitHub Issues
- Feature requests: Start a discussion
- Pull requests: Always welcome

## Citation

If you use this in research, please cite:
- Original GLMsingle: [Prince et al., 2022, Nature Neuroscience]
- FLOBS method: [FSL Documentation]
- This implementation: [Your paper/GitHub]

## License

MIT License (same as GLMsingle)

## Contact

Logan Grosenick
Converted from MATLAB `simulate_movietasks.m` and GLMsingle

---

**FastFuncSim**: Because science needs speed 🚀
