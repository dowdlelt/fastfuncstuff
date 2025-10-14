# Masking and Chunking Guide

## Overview

FastFuncSim now supports **brain masking** and **memory-efficient chunking** for analyzing real fMRI data with millions of voxels.

## Brain Masking

### Why Use Masks?

- **Speed**: Analyze only brain voxels (skip background/skull)
- **Memory**: Reduce memory footprint by 50-80%
- **Quality**: Focus statistics on relevant tissue

### Using Masks

```python
import fastfuncsim as ffs

# With AFNI mask
results, design_info = ffs.analyze_from_design_matrix(
    run_files=['run01.nii.gz', 'run02.nii.gz', 'run03.nii.gz', 'run04.nii.gz'],
    design_matrix_file='X.xmat.1D',
    method='arma11',
    mask_file='mask.nii.gz',        # Brain mask
    mask_threshold=0.5,              # Include voxels > 0.5
    device=device
)

# Check mask info
print(f"Masked voxels: {design_info['mask_voxels']:,}")
print(f"Full volume shape: {results.full_shape}")
```

### Mask Format

- **NIfTI format** (`.nii` or `.nii.gz`)
- **Same spatial dimensions** as functional data
- **Values**: Typically 0 (exclude) or 1 (include), but any threshold works
- **Common masks**:
  - `mask.nii.gz` from AFNI preprocessing
  - `mask_epi_anat.nii.gz` from `align_epi_anat.py`
  - Custom masks from segmentation

### Output with Masks

Masked analysis produces **sparse** results (only brain voxels), but exports automatically reconstruct **full volumes**:

```python
# Masked voxels only
print(results.betas.shape)  # (n_masked_voxels, n_regressors)

# Export reconstructs full volume (zeros outside mask)
ffs.write_afni_bucket(
    results,
    'output_bucket.nii.gz',
    condition_names=['face', 'place'],
)
# Output has full spatial dimensions with mask applied
```

## Chunking for Large Datasets

### Why Chunk?

- **GPU Memory**: Millions of voxels don't fit in GPU VRAM
- **Efficiency**: Process large blocks at once (not one-by-one)

### Automatic Chunking

FastFuncSim automatically chooses chunk sizes based on:
- Available GPU memory
- Number of timepoints
- Number of regressors

```python
# Automatic chunking (recommended)
results, _ = ffs.analyze_from_design_matrix(
    data,
    design_matrix,
    method='ols',
    device=device
)
# Will print: "Processing in chunks of 50000 voxels"
```

### Manual Chunk Size

Override for specific hardware:

```python
# Explicit chunk size
results, _ = ffs.analyze_from_design_matrix(
    data,
    design_matrix,
    method='ols',
    voxel_chunk_size=50000,  # Process 50k voxels at a time
    device=device
)
```

### Recommended Chunk Sizes

| GPU VRAM | Timepoints | Chunk Size |
|----------|------------|------------|
| 4 GB     | 300-500    | 10,000     |
| 8 GB     | 300-500    | 50,000     |
| 16 GB    | 300-500    | 100,000    |
| 24+ GB   | 300-500    | 200,000+   |

**Note**: Larger chunks = faster overall, but need more memory.

### ARMA(1,1) Batch Size

ARMA(1,1) uses **batch processing** for per-voxel parameter estimation:

```python
results = ffs.fit_glm_arma11(
    data,
    design,
    tr=2.0,
    batch_size=10000,  # REML grid search batch
    device=device
)
```

Default `batch_size=10000` works well for most GPUs. Increase for:
- Larger GPU memory
- Fewer timepoints
- Fewer ARMA grid points

## CPU/GPU Memory Strategy

### Data Storage

- **Data on CPU**: Voxel timeseries stored in RAM
- **Chunks to GPU**: Small batches copied to GPU for processing
- **Results to CPU**: Statistics moved back to RAM

This allows analyzing **unlimited voxels** with limited GPU memory.

### Example: 3M Voxels

```python
# Analyze 3 million voxels on 8GB GPU
results, design_info = ffs.analyze_from_design_matrix(
    ['run01.nii.gz', 'run02.nii.gz', 'run03.nii.gz', 'run04.nii.gz'],
    'X.xmat.1D',
    method='arma11',
    mask_file='mask.nii.gz',        # Reduces to ~300k brain voxels
    voxel_chunk_size=50000,          # Process 50k at a time = 6 chunks
    device=device
)

# Expected:
# - Chunk 1: 50,000 voxels -> GPU -> process -> CPU
# - Chunk 2: 50,000 voxels -> GPU -> process -> CPU
# - ...
# - Chunk 6: 50,000 voxels -> GPU -> process -> CPU
# Total time: ~10-15 minutes (vs hours without chunking/masking)
```

## Performance Tips

### 1. Always Use a Mask

```python
# ✓ GOOD: With mask (300k voxels)
results = ffs.analyze_from_design_matrix(
    data, design, method='arma11',
    mask_file='mask.nii.gz'
)
# Fast: 10-15 minutes

# ✗ BAD: No mask (3M voxels)
results = ffs.analyze_from_design_matrix(
    data, design, method='arma11'
)
# Slow: 1-2 hours
```

### 2. Optimize Chunk Size

```python
# Too small: 1000 voxels
# - Many chunks = overhead
# - Time: 30 minutes

# Optimal: 50,000 voxels
# - Few chunks, good GPU utilization
# - Time: 12 minutes

# Too large: 500,000 voxels (if exceeds GPU memory)
# - OOM error or disk swapping
```

### 3. Use Global ARMA for Exploration

```python
# Per-voxel (accurate but slow)
results = ffs.fit_glm_arma11(
    data, design, tr=2.0,
    estimate_per_voxel=True,  # 15 minutes
    batch_size=10000
)

# Global (fast approximation)
results = ffs.fit_glm_arma11(
    data, design, tr=2.0,
    estimate_per_voxel=False,  # 2 minutes
)
```

Use global for initial exploration, per-voxel for final analysis.

## Troubleshooting

### Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce chunk size
```python
results = ffs.analyze_from_design_matrix(
    data, design,
    voxel_chunk_size=10000,  # Smaller chunks
    device=device
)
```

### Slow Analysis

**Symptom**: Taking hours for large dataset

**Solutions**:
1. **Add mask** (biggest speedup)
2. **Increase chunk size** (if memory allows)
3. **Use global ARMA** (for exploration)

### Mask Shape Mismatch

**Symptom**: `ValueError: Mask shape does not match data volume shape`

**Solution**: Ensure mask has same spatial dimensions
```python
# Check shapes
import nibabel as nib
data_img = nib.load('run01.nii.gz')
mask_img = nib.load('mask.nii.gz')

print(f"Data shape: {data_img.shape}")  # (64, 64, 35, 300)
print(f"Mask shape: {mask_img.shape}")  # Should be (64, 64, 35)
```

Resample mask if needed using AFNI's `3dresample`.

## Example Workflow

Complete real-data analysis with masking and chunking:

```python
import fastfuncsim as ffs
from pathlib import Path

# Setup
device = ffs.get_device()
data_dir = Path('/path/to/data')

# Files
run_files = [
    data_dir / 'run01.nii.gz',
    data_dir / 'run02.nii.gz',
    data_dir / 'run03.nii.gz',
    data_dir / 'run04.nii.gz',
]
design_matrix = data_dir / 'X.xmat.1D'
mask_file = data_dir / 'mask.nii.gz'

# Analyze with mask and optimal chunking
print("Starting analysis...")
results, design_info = ffs.analyze_from_design_matrix(
    run_files,
    design_matrix,
    method='arma11',
    mask_file=mask_file,
    mask_threshold=0.5,
    voxel_chunk_size=50000,  # Adjust for your GPU
    device=device
)

print(f"✓ Complete! Analyzed {design_info['mask_voxels']:,} voxels")
print(f"  Mean R²: {results.r2.mean():.3f}")
print(f"  Mean ARMA(a,b): ({results.arma_params[:, 0].mean():.3f}, "
      f"{results.arma_params[:, 1].mean():.3f})")

# Write results (automatically handles masking)
ffs.write_afni_bucket(
    results,
    data_dir / 'glm_arma11_bucket.nii.gz',
    condition_names=design_info['stim_labels']
)

print("✓ Results saved!")
```

## Summary

| Feature | Benefit | Usage |
|---------|---------|-------|
| **Masking** | 50-80% speedup | `mask_file='mask.nii.gz'` |
| **Chunking** | Unlimited voxels | Automatic or `voxel_chunk_size=50000` |
| **CPU storage** | GPU memory efficiency | Automatic |
| **Batch ARMA** | Fast parameter estimation | `batch_size=10000` |

With masking + chunking, FastFuncSim can analyze **whole-brain** datasets on **consumer GPUs** in **minutes**, not hours.
