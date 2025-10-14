# Simulation Output Formats

FastFuncSim now supports saving simulation outputs in standard neuroimaging formats compatible with AFNI, FSL, SPM, and other analysis tools.

## Important: Saving is OPTIONAL

**By default, simulations DO NOT save any files**. All simulation functions (`simulate_fmri_run`, `simulate_fmri_experiment`, `simulate_batch_experiments`) only compute and return data in memory.

You must **explicitly** call `save_simulation_outputs()` when you want to save results. This is crucial for batch simulations where you might run thousands of experiments and only want to save selected results or summary statistics.

## Overview

Three new functions added to `simulation.py`:

1. **`write_afni_onset_files()`** - Write AFNI-compatible onset timing files
2. **`write_nifti_files()`** - Write NIfTI (nii.gz) files using nibabel
3. **`save_simulation_outputs()`** - Main function that saves everything to organized folders (call explicitly)

## Quick Start

```python
import fastfuncsim as ffs

# Run multi-run simulation
data_list = ffs.simulate_fmri_experiment(
    n_runs=3,  # Multiple runs for cross-validation
    onsets=onsets_list,
    betas=[3.0, 4.0],
    hrf=hrf,
    tr=2.0,
    n_timepoints=200,
    matrix_size=(64, 64, 30)
)

# Save everything
output_info = ffs.save_simulation_outputs(
    data_list=data_list,
    onsets_list=onsets_list,
    tr=2.0,
    output_dir='./my_simulations',
    label='experiment_001',
    metadata={'betas': [3.0, 4.0], 'noise_level': 1.0},
    voxel_size=(2.0, 2.0, 2.0)
)
```

## Output Structure

Creates folder: `output_dir/simulation_{label}/`

```
simulation_experiment_001/
├── onsets_condition1.txt    # AFNI onset timing (one per condition)
├── onsets_condition2.txt
├── run01.nii.gz              # NIfTI files (one per run)
├── run02.nii.gz
├── run03.nii.gz
└── metadata.txt              # Simulation parameters
```

## File Formats

### AFNI Onset Timing Files

**Format**: Space-separated onset times in seconds, one row per run

**Example** (`onsets_condition1.txt`):
```
0.00 4.00 14.00 24.00 38.00 42.00 48.00 54.00 68.00
0.00 10.00 26.00 34.00 48.00 58.00 72.00 80.00 92.00
0.00 12.00 16.00 26.00 32.00 42.00 58.00 70.00 78.00
```

- **Row 1** = Run 1 onsets
- **Row 2** = Run 2 onsets
- **Row 3** = Run 3 onsets

**Special case**: If a condition has no events in a run, uses `*` (AFNI convention)

### NIfTI Files

**Format**: Standard NIfTI-1 format (`.nii.gz`)

**Header information**:
- **Shape**: (nx, ny, nz, n_timepoints)
- **Voxel size**: Configurable (default: 2.0 x 2.0 x 2.0 mm)
- **TR**: Set in pixdim[4]
- **Units**: mm (spatial), sec (temporal)
- **Data type**: float32
- **Affine**: Simple diagonal matrix (can customize)

### Metadata File

**Format**: Plain text with simulation parameters

**Example** (`metadata.txt`):
```
Simulation Label: experiment_001
TR: 2.0 sec
Number of runs: 3
Voxel size: 2.0 x 2.0 x 2.0 mm
Data shape per run: torch.Size([64, 64, 30, 200])
Number of timepoints: 200
Matrix size: 64 x 64 x 30

Additional Parameters:
  betas: [3.0, 4.0]
  n_conditions: 2
  noise_level: 1.0
  hrf_duration: 32.0
  scanner_drift: True
  isi_mean: 6
```

## Usage with Analysis Tools

---

## Analysis Output Formats (NEW)

FastFuncSim GLM results (`fit_glm`, `fit_glm_arma11`) can now be exported directly to
NIfTI using `write_glm_results_nifti`. The exporter mirrors AFNI's bucket layout:

- The **4th dimension** contains interleaved Beta / t-stat volumes for each condition.
- An **omnibus F-statistic** is written to a companion file (one value per voxel).
- Optional maps include R², mean signal, and residual sigma.
- A JSON manifest documents the ordering of volumes for transparency.

```python
import fastfuncsim as ffs

results = ffs.fit_glm(
    data, design, tr=1.0,
    want_residuals=False,
    want_predicted=False,
    verbose=False,
)

ffs.write_glm_results_nifti(
    results,
    output_dir="./glm_exports",
    prefix="sub-01",
    condition_names=["stimA", "stimB"],
    include_beta=True,
    include_tstat=True,
    include_fstat=True,
    include_r2=True,
    include_mean=True,
)
```

### Exported Files

```
glm_exports/
├── sub-01_stats.nii.gz    # 4D volume: beta/t-stat pairs per condition
├── sub-01_stats.json      # Volume manifest (condition + metric)
├── sub-01_fstat.nii.gz    # Omnibus F-statistic map
├── sub-01_r2.nii.gz       # Voxelwise R²
└── sub-01_mean.nii.gz     # Mean signal
```

### Volume Ordering

For 2 conditions with both betas and t-stats enabled the 4th dimension follows:

1. `stimA` Beta
2. `stimA` t-stat
3. `stimB` Beta
4. `stimB` t-stat

The JSON manifest mirrors this structure and is intended for downstream pipelines
that need deterministic ordering (AFNI's bucket history, FSL FEAT custom contrasts,
etc.). Residual timeseries and model predictions can be exported on demand by
enabling `write_residuals=True` / `write_predictions=True`.

### AFNI

```bash
cd simulation_experiment_001/

# GLM analysis with 3dDeconvolve
3dDeconvolve \
  -input run*.nii.gz \
  -polort 3 \
  -num_stimts 2 \
  -stim_times 1 onsets_condition1.txt 'BLOCK(2,1)' \
  -stim_times 2 onsets_condition2.txt 'BLOCK(2,1)' \
  -stim_label 1 condition1 \
  -stim_label 2 condition2 \
  -bucket stats.nii.gz

# Or with REML prewhitening
3dREMLfit \
  -input run*.nii.gz \
  -matrix X.xmat.1D \
  -Rbuck stats_REML.nii.gz
```

### FSL

```bash
# Convert onset files to FSL format (3-column: onset duration amplitude)
# Then use FEAT GUI or command-line

# Quick conversion for event-related (assuming 2s stimulus duration)
awk '{for(i=1;i<=NF;i++) print $i, 2, 1}' onsets_condition1.txt > ev1.txt
```

### SPM (MATLAB)

```matlab
% Load NIfTI files
V = spm_vol('run01.nii.gz');
Y = spm_read_vols(V);

% Parse onset files
onsets_cond1 = load('onsets_condition1.txt');
% Each row is a run
```

### Python (nilearn, nibabel)

```python
import nibabel as nib
import numpy as np

# Load data
img = nib.load('simulation_experiment_001/run01.nii.gz')
data = img.get_fdata()  # (nx, ny, nz, n_timepoints)

# Load onsets
onsets_cond1 = np.loadtxt('simulation_experiment_001/onsets_condition1.txt')
# onsets_cond1[0] = run 1 onsets
# onsets_cond1[1] = run 2 onsets

# Use with nilearn GLM
from nilearn.glm.first_level import FirstLevelModel
fmri_glm = FirstLevelModel(t_r=2.0)
# ... construct design matrix from onsets
```

## Advanced Usage

### Custom Affine Matrix

```python
import numpy as np

# Create custom affine (e.g., oblique slice acquisition)
affine = np.array([
    [2.0,  0.1, 0.0, -64.0],
    [0.1,  2.0, 0.0, -64.0],
    [0.0,  0.0, 2.5, -30.0],
    [0.0,  0.0, 0.0,   1.0]
])

output_info = ffs.save_simulation_outputs(
    data_list=data_list,
    onsets_list=onsets_list,
    tr=2.0,
    output_dir='./my_simulations',
    label='oblique_slices',
    affine=affine  # Custom affine
)
```

### Different Voxel Sizes

```python
# Anisotropic voxels (2 x 2 x 3 mm)
output_info = ffs.save_simulation_outputs(
    data_list=data_list,
    onsets_list=onsets_list,
    tr=2.0,
    output_dir='./my_simulations',
    label='anisotropic',
    voxel_size=(2.0, 2.0, 3.0)
)
```

### Rich Metadata

```python
metadata = {
    'betas': [3.0, 4.0],
    'n_conditions': 2,
    'noise_level': 1.5,
    'hrf_type': 'canonical',
    'hrf_duration': 32.0,
    'scanner_drift': True,
    'drift_amplitude': 0.5,
    'isi_mean': 6,
    'isi_distribution': 'exponential',
    'experiment_date': '2025-10-12',
    'notes': 'High SNR test for design optimization'
}

output_info = ffs.save_simulation_outputs(
    data_list=data_list,
    onsets_list=onsets_list,
    tr=2.0,
    output_dir='./my_simulations',
    label='high_snr_test',
    metadata=metadata
)
```

## Individual Function Usage

### Write Only Onset Files

```python
onset_files = ffs.write_afni_onset_files(
    onsets_list=onsets_list,
    tr=2.0,
    output_dir='./my_onsets',
    prefix='onsets'
)
# Creates: onsets_condition1.txt, onsets_condition2.txt, ...
```

### Write Only NIfTI Files

```python
nifti_files = ffs.write_nifti_files(
    data_list=data_list,
    tr=2.0,
    output_dir='./my_data',
    prefix='run',
    voxel_size=(2.0, 2.0, 2.0)
)
# Creates: run01.nii.gz, run02.nii.gz, ...
```

## Cross-Validation Workflow

```python
import fastfuncsim as ffs

# Setup
n_runs = 4  # For 4-fold cross-validation
device = ffs.get_device()
tr = 2.0
n_timepoints = 250

# Generate different onsets per run
onsets_list = []
for run_idx in range(n_runs):
    torch.manual_seed(run_idx + 100)  # Different seed per run
    onsets = ffs.generate_random_onsets(
        n_timepoints=n_timepoints,
        n_conditions=3,
        isi_mean=5,
        tr=tr,
        device=device
    )
    onsets_list.append(onsets)

# Simulate
data_list = ffs.simulate_fmri_experiment(
    n_runs=n_runs,
    onsets=onsets_list,
    betas=[2.5, 3.0, 3.5],
    hrf=ffs.get_canonical_hrf(stim_duration=2.0, tr=tr, device=device),
    tr=tr,
    n_timepoints=n_timepoints,
    matrix_size=(64, 64, 30)
)

# Save
output_info = ffs.save_simulation_outputs(
    data_list=data_list,
    onsets_list=onsets_list,
    tr=tr,
    output_dir='./cross_validation',
    label='4fold_cv',
    metadata={'n_folds': 4, 'betas': [2.5, 3.0, 3.5]}
)

# Now you can:
# - Train on runs 1,2,3 → test on run 4
# - Train on runs 1,2,4 → test on run 3
# - etc.
```

## Return Values

All functions return file path information for downstream processing:

```python
output_info = ffs.save_simulation_outputs(...)

print(output_info['output_dir'])      # Path to simulation folder
print(output_info['onset_files'])     # List of onset file paths
print(output_info['nifti_files'])     # List of nifti file paths
print(output_info['metadata_file'])   # Path to metadata file

# Use for batch processing
for nifti_file in output_info['nifti_files']:
    # Process each run
    pass
```

## Notes

1. **Multi-run support**: All functions support multiple runs for cross-validation
2. **AFNI compatibility**: Onset files follow AFNI's space-separated format
3. **Nibabel integration**: Uses standard nibabel library for maximum compatibility
4. **Folder organization**: Each simulation gets its own folder with all outputs
5. **Metadata tracking**: Automatically saves simulation parameters for reproducibility
6. **Flexible metadata**: Can save arbitrary parameters (betas, HRF, noise, etc.)
7. **GPU → CPU conversion**: Automatically converts PyTorch tensors to numpy arrays

## Batch Simulations (1000s of runs)

**Key principle**: Only save what you need!

```python
import fastfuncsim as ffs
import numpy as np

# Simulation parameters
n_simulations = 1000
device = ffs.get_device()

# Storage for summary statistics (lightweight!)
power_estimates = []
efficiency_estimates = []

for sim_idx in range(n_simulations):
    # Simulate (NO automatic saving!)
    data = ffs.simulate_fmri_run(
        onsets=onsets,
        betas=[3.0, 4.0],
        hrf=hrf,
        tr=2.0,
        n_timepoints=200,
        matrix_size=(64, 64, 30),
        device=device
    )

    # Fit GLM
    results = ffs.fit_glm(data, onsets, tr=2.0)

    # Extract summary statistics (small!)
    power_estimates.append(results.t_stats.mean().item())
    efficiency_estimates.append(results.r2.mean().item())

    # Data is automatically freed when out of scope
    # NO files written to disk!

# Save only summary statistics
np.savetxt('batch_power_estimates.txt', power_estimates)
np.savetxt('batch_efficiency_estimates.txt', efficiency_estimates)

print(f"Simulated {n_simulations} experiments")
print(f"Saved only 2 small text files (not {n_simulations} × ~500MB nifti files!)")
```

### Selective Saving (Save Only Interesting Cases)

```python
# Run many simulations, save only outliers
for sim_idx in range(1000):
    data = ffs.simulate_fmri_run(...)
    results = ffs.fit_glm(...)

    # Only save if something interesting happens
    mean_power = results.t_stats.mean().item()

    if mean_power > threshold:  # Outlier!
        # Save this specific case for inspection
        ffs.save_simulation_outputs(
            data_list=[data],
            onsets_list=onsets,
            tr=2.0,
            output_dir='./outliers',
            label=f'high_power_sim{sim_idx:04d}',
            metadata={'power': mean_power, 'sim_id': sim_idx}
        )
```

### Save Only First/Last for Verification

```python
# Verify first and last simulation, skip the rest
for sim_idx in range(1000):
    data = ffs.simulate_fmri_run(...)

    # Save only sim #0 and sim #999
    if sim_idx == 0 or sim_idx == 999:
        ffs.save_simulation_outputs(
            data_list=[data],
            onsets_list=onsets,
            tr=2.0,
            output_dir='./verification',
            label=f'sim{sim_idx:04d}'
        )

    # Process all simulations normally
    results = ffs.fit_glm(...)
    # ... accumulate statistics
```

## See Also

- `example_save_simulation.py` - Complete working example
- AFNI documentation: https://afni.nimh.nih.gov/pub/dist/doc/program_help/1deval.html
- nibabel documentation: https://nipy.org/nibabel/
