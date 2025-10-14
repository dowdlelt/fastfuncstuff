# Precomputed ARMA Parameters

## Overview

FastFuncSim now supports using **precomputed ARMA(1,1) parameters** for ultra-fast re-analysis. This feature allows you to:

1. **Save ~80% of compute time** on repeated analyses
2. **Validate against AFNI** by using AFNI-estimated parameters
3. **Explore different contrasts** without re-estimating ARMA parameters
4. **Ensure consistency** across related analyses

## Key Benefits

### Time Savings

The ARMA(1,1) GLM process has two main steps:

1. **REML Grid Search** (~80% of time)
   - Finds optimal (a, b) parameters for each voxel
   - Evaluates 63 (a,b) combinations per voxel
   - ~8-10 minutes for 392k voxels

2. **GLS Fit** (~20% of time)
   - Prewhitens data using ARMA covariance
   - Solves weighted least squares
   - ~2 minutes for 392k voxels

**With precomputed parameters:** Skip step 1 entirely! → **~2 minutes total**

### Use Cases

#### 1. Iterative Contrast Exploration
```python
# First run: Full ARMA estimation
results_v1 = ffs.analyze_from_design_matrix(
    data, design_matrix, method="arma11"
)

# Save ARMA parameters
ffs.save_arma_params(
    results_v1.arma_params,
    "arma_params.nii.gz",
    volume_shape=results_v1.full_shape,
    voxel_mask=results_v1.voxel_mask,
    affine=results_v1.affine,
)

# Later: Try different contrasts (80% faster!)
arma_params = ffs.load_arma_params("arma_params.nii.gz", mask)
results_v2 = ffs.analyze_from_design_matrix(
    data, design_matrix, method="arma11",
    precomputed_arma_params=arma_params  # ← Skip REML!
)
```

#### 2. AFNI Validation
```python
# Extract AFNI's ARMA parameters from their output
# (This requires reading AFNI's errts.REML file or similar)
afni_params = load_afni_arma_parameters("errts.REML+tlrc")

# Use AFNI parameters in fastfuncsim
results = ffs.fit_glm_arma11(
    data, design, tr=2.0,
    precomputed_arma_params=afni_params
)

# Betas and t-stats should match AFNI exactly!
```

#### 3. Group Analysis Consistency
```python
# Use average ARMA parameters across subjects
all_params = []
for subject in subjects:
    results = ffs.analyze_from_design_matrix(...)
    all_params.append(results.arma_params)

mean_params = np.mean(all_params, axis=0)

# Apply to all subjects for consistency
for subject in subjects:
    results = ffs.analyze_from_design_matrix(
        ..., precomputed_arma_params=mean_params
    )
```

## API Reference

### Saving ARMA Parameters

```python
ffs.save_arma_params(
    arma_params: Union[torch.Tensor, np.ndarray],
    output_path: Union[str, Path],
    volume_shape: Optional[Tuple[int, int, int]] = None,
    voxel_mask: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
) -> Path
```

Creates a 4D NIfTI file with 2 volumes:
- Volume 0: 'a' parameter (AR component)
- Volume 1: 'b' parameter (MA component)

**Parameters:**
- `arma_params`: (n_voxels, 2) array from `results.arma_params`
- `output_path`: Where to save (e.g., 'arma_params.nii.gz')
- `volume_shape`: Optional (nx, ny, nz) for reshaping
- `voxel_mask`: Optional mask if data was masked
- `affine`: Optional 4x4 affine matrix for spatial registration

**Example:**
```python
results = ffs.fit_glm_arma11(data, design, tr=2.0)

ffs.save_arma_params(
    results.arma_params,
    'arma_params.nii.gz',
    volume_shape=results.full_shape,
    voxel_mask=results.voxel_mask,
    affine=results.affine
)
```

### Loading ARMA Parameters

```python
ffs.load_arma_params(
    filepath: Union[str, Path],
    voxel_mask: Optional[np.ndarray] = None,
) -> np.ndarray
```

Loads precomputed ARMA parameters from NIfTI file.

**Parameters:**
- `filepath`: Path to saved parameters file
- `voxel_mask`: Optional mask to extract only masked voxels

**Returns:**
- (n_voxels, 2) array ready for `precomputed_arma_params`

**Example:**
```python
# Load with mask
mask = ffs.load_afni_mask('mask.nii.gz', threshold=0.5)
arma_params = ffs.load_arma_params('arma_params.nii.gz', voxel_mask=mask)

# Use in analysis
results = ffs.analyze_from_design_matrix(
    data, design_matrix, method='arma11',
    precomputed_arma_params=arma_params
)
```

### Using in Analysis

#### Option 1: High-level API
```python
results, design_info = ffs.analyze_from_design_matrix(
    fmri_data='run01.nii.gz',
    design_matrix_file='X.xmat.1D',
    method='arma11',
    precomputed_arma_params=arma_params,  # ← Add this!
    mask_file='mask.nii.gz',
    device=device
)
```

#### Option 2: Low-level API
```python
results = ffs.fit_glm_arma11(
    data,
    design,
    tr=2.0,
    precomputed_arma_params=arma_params,  # ← Add this!
    device=device
)
```

## File Format

The ARMA parameters NIfTI file has:
- **Shape**: (nx, ny, nz, 2)
- **Volume 0**: AR parameter 'a' (range: 0.1-0.9 typically)
- **Volume 1**: MA parameter 'b' (range: -0.3 to 0.3 typically)
- **Header**: Preserves affine from input data
- **Description**: "ARMA(1,1) parameters: vol0=a, vol1=b"

### Viewing in AFNI
```bash
# View a-parameter map
3dinfo arma_params.nii.gz
afni -niml &
# Load arma_params.nii.gz
# Select sub-brick 0 (a) or 1 (b)
```

### Viewing in FSLeyes
```bash
fsleyes arma_params.nii.gz &
# Use volume slider to switch between a and b
```

## Implementation Details

### How It Works

When you provide `precomputed_arma_params`:

1. **Validation**: Checks shape is (n_voxels, 2)
2. **Skip REML**: No grid search performed
3. **Build Cache**: Creates Cholesky factorizations for unique (a,b) pairs
4. **GLS Fit**: Proceeds with normal prewhitening and parameter estimation

### Performance

#### With REML Estimation (default)
```
Step 1: REML grid search
  - Evaluate 63 (a,b) combinations per voxel
  - Time: ~8-10 min for 392k voxels
  
Step 2: GLS fit  
  - Prewhiten and solve
  - Time: ~2 min

Total: ~10-12 minutes
```

#### With Precomputed Parameters
```
Step 1: Load parameters
  - Read NIfTI file
  - Time: < 1 second

Step 2: Build Cholesky cache
  - Only for unique (a,b) pairs
  - Time: ~10-30 seconds
  
Step 3: GLS fit
  - Prewhiten and solve
  - Time: ~2 min

Total: ~2-3 minutes (80% faster!)
```

### Memory Usage

The precomputed parameters require minimal memory:
- **Storage**: (n_voxels × 2 × 4 bytes) compressed
- **Example**: 392k voxels → ~3 MB compressed

## AFNI Compatibility

### Reading AFNI Parameters

AFNI stores ARMA parameters in various ways:
1. **errts.REML file**: Contains residuals + parameters
2. **Bucket sub-bricks**: Sometimes includes ARMA params
3. **FITTS file**: May contain prewhitening info

**To extract (example):**
```python
import nibabel as nib

# This is dataset-specific - check AFNI documentation
img = nib.load('errts.REML+tlrc.BRIK')
data = img.get_fdata()

# Extract a and b from appropriate sub-bricks
# (exact method depends on AFNI output format)
a_params = data[..., a_subbrick]
b_params = data[..., b_subbrick]

arma_params = np.stack([a_params.flatten(), b_params.flatten()], axis=1)
```

### Validation Against AFNI

To verify fastfuncsim matches AFNI:

```python
# 1. Run AFNI analysis
# 3dREMLfit -matrix X.xmat.1D -input pb04.$subj.r$run.scale+tlrc ...

# 2. Extract AFNI's ARMA parameters
afni_params = extract_afni_arma_params(...)  # Your extraction code

# 3. Run fastfuncsim with same parameters
results = ffs.fit_glm_arma11(
    data, design, tr=2.0,
    precomputed_arma_params=afni_params
)

# 4. Compare betas and t-stats
# Should match within numerical precision!
afni_betas = load_afni_betas(...)
np.allclose(results.betas, afni_betas, rtol=1e-5)
```

## Best Practices

### When to Save Parameters

✅ **DO save** when:
- Running multiple analyses on same data
- Exploring different contrast combinations
- Validating methods across tools
- Doing group-level analyses

❌ **DON'T save** when:
- One-off analysis
- Data preprocessing changes between runs
- Different masks are used

### File Organization

```
project/
├── derivatives/
│   └── sub-01/
│       ├── func/
│       │   ├── run01.nii.gz
│       │   └── run02.nii.gz
│       ├── mask.nii.gz
│       ├── X.xmat.1D
│       ├── arma_params.nii.gz         ← Save here
│       ├── glm_v1_bucket.nii.gz       ← First analysis
│       └── glm_v2_bucket.nii.gz       ← Reanalysis (fast!)
```

### Verification

Always verify parameters are reasonable:
```python
arma_params = ffs.load_arma_params('arma_params.nii.gz', mask)

print(f"Mean a: {arma_params[:, 0].mean():.3f}")  # Should be ~0.2-0.6
print(f"Mean b: {arma_params[:, 1].mean():.3f}")  # Should be ~-0.1-0.1

# Check for outliers
assert arma_params[:, 0].min() >= 0.0
assert arma_params[:, 0].max() <= 1.0
assert abs(arma_params[:, 1]).max() <= 0.8
```

## Complete Example Workflow

See `examples/example_precomputed_arma.py` for a full working example:

```python
# Step 1: First analysis (full ARMA estimation)
results_v1, design_info = ffs.analyze_from_design_matrix(
    run_files, design_matrix_file, method="arma11"
)

# Step 2: Save parameters
ffs.save_arma_params(
    results_v1.arma_params, 
    "arma_params.nii.gz",
    volume_shape=results_v1.full_shape,
    voxel_mask=results_v1.voxel_mask,
    affine=results_v1.affine
)

# Step 3: Later - fast reanalysis
mask = ffs.load_afni_mask(mask_file, threshold=0.5)
arma_params = ffs.load_arma_params("arma_params.nii.gz", voxel_mask=mask)

results_v2, design_info = ffs.analyze_from_design_matrix(
    run_files,
    design_matrix_file,
    method="arma11",
    precomputed_arma_params=arma_params  # ← 80% faster!
)

# Step 4: New contrasts
contrasts = np.array([[1, -1, 0, 0]])  # Different contrasts
contrast_results = ffs.compute_contrasts(results_v2, contrasts)
```

## Troubleshooting

### Shape Mismatch Error
```
ValueError: precomputed_arma_params must have shape (392240, 2), got (100, 2)
```
**Solution**: Make sure the mask used during saving matches the mask used during loading.

### Invalid Parameter Values
```
Warning: Some ARMA parameters are outside expected range
```
**Solution**: Check if the saved parameters are from the correct analysis. Verify a ∈ [0, 1] and |b| < 1.

### Performance Not Improved
```
Still taking ~10 minutes even with precomputed params
```
**Solution**: Verify the parameters are actually being passed to the function. Check for typos in parameter name.

## See Also

- `examples/example_precomputed_arma.py`: Complete working example
- `examples/analyze_real_data.py`: Real dataset analysis
- `fastfuncsim/arma_glm.py`: Core ARMA implementation
- AFNI 3dREMLfit documentation: https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/statistics/remlfit.html
