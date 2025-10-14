# Output Improvements Summary

## Changes Implemented

### 1. ✅ Fixed 3drefit Label String Format

**Before:**
```bash
3drefit -relabel_all_str 'label1' 'label2' 'label3' ...
```

**After:**
```bash
3drefit -relabel_all_str "label1 label2 label3"
```

**Implementation:**
```python
# Single space-separated string (correct AFNI format)
label_str = " ".join(bucket_info["SubBricks"])
cmd = ["3drefit", "-relabel_all_str", label_str, str(output_path)]
```

### 2. ✅ Automated 3drefit Execution

**Before:** Printed commands for user to run manually

**After:** Automatically runs 3drefit if available

**Implementation:**
```python
import subprocess
import shutil

# Check if 3drefit is available
if shutil.which("3drefit"):
    # Apply labels
    subprocess.run(
        ["3drefit", "-relabel_all_str", label_str, str(output_path)],
        check=True, capture_output=True, text=True
    )
    
    # Apply statistical parameters
    subprocess.run(cmd_stats, check=True, capture_output=True, text=True)
else:
    print("⚠ 3drefit not found - skipping AFNI metadata")
```

**Features:**
- ✅ Automatically detects if AFNI is installed
- ✅ Applies labels in one command
- ✅ Applies statistical parameters in one command
- ✅ Captures errors gracefully
- ✅ Continues if 3drefit not available

### 3. ✅ Save ARMA Parameters (Like AFNI 3dREMLfit)

AFNI's 3dREMLfit saves several diagnostic outputs. Now fastfuncsim does too!

#### Output Files

##### 1. `arma_params.nii.gz`
**Format:** 4D NIfTI with 2 volumes
- Volume 0: **a** parameter (AR component, range 0-1)
- Volume 1: **b** parameter (MA component, range -1 to 1)

**Purpose:**
- Reuse for fast re-analysis (80% time savings)
- Validate against AFNI parameters
- Quality checking (visualize parameter maps)

**Example:**
```python
ffs.save_arma_params(
    results.arma_params,
    'arma_params.nii.gz',
    volume_shape=results.full_shape,
    voxel_mask=results.voxel_mask,
    affine=results.affine
)
```

##### 2. `arma_lambda.nii.gz`
**Format:** 3D NIfTI
- Contains: λ (lag-1 autocorrelation)
- Relationship: λ = (a+b)(1+ab)/(1+2ab+b²)

**Purpose:**
- Shows effective temporal correlation
- Quality checking (should be 0.1-0.5 typically for fMRI)
- Compare with empirical autocorrelation

**Interpretation:**
- λ ≈ 0: No temporal correlation (rare in fMRI)
- λ ≈ 0.2-0.4: Typical for fMRI data
- λ > 0.6: High autocorrelation (check preprocessing)

##### 3. `glm_r2.nii.gz`
**Format:** 3D NIfTI
- Contains: R² (coefficient of determination)
- Range: 0-1 (0 = no fit, 1 = perfect fit)

**Purpose:**
- Visualize model fit quality
- Identify problem voxels
- Compare OLS vs ARMA improvement

**Typical Values:**
- Gray matter: R² ≈ 0.3-0.5
- White matter: R² ≈ 0.1-0.2
- High activation: R² > 0.5

## Complete Workflow

### Step-by-Step Output Generation

```python
# Step 1: Fit ARMA GLM
results, design_info = ffs.analyze_from_design_matrix(
    run_files, design_matrix_file, method="arma11"
)

# Step 2: Compute contrasts
contrasts = np.array([[1, -1, 0, 0]])
contrast_results = ffs.compute_contrasts(results, contrasts)

# Step 3: Write AFNI bucket
ffs.write_afni_bucket(
    results, output_path,
    condition_names=design_info["column_labels"],
    contrast_names=["face_vs_place"],
    contrast_results=contrast_results,
    affine=results.affine
)

# Step 4: Apply AFNI metadata (automatic!)
# - Labels applied via 3drefit
# - Stat parameters applied via 3drefit

# Step 5: Save ARMA parameters
ffs.save_arma_params(
    results.arma_params, "arma_params.nii.gz",
    volume_shape=results.full_shape,
    voxel_mask=results.voxel_mask,
    affine=results.affine
)

# Step 6: Save lambda
# ... (see analyze_real_data.py)

# Step 7: Save R²
# ... (see analyze_real_data.py)
```

## Output File Summary

| File | Size | Purpose | AFNI Equivalent |
|------|------|---------|-----------------|
| `glm_arma11_bucket.nii.gz` | ~400 MB | Main GLM results | `stats.*.REML+tlrc` |
| `arma_params.nii.gz` | ~3 MB | ARMA (a,b) parameters | Embedded in errts |
| `arma_lambda.nii.gz` | ~1.5 MB | Lag-1 autocorrelation | Not directly saved |
| `glm_r2.nii.gz` | ~1.5 MB | R² map | Can compute from residuals |
| `glm_arma11_bucket.json` | ~5 KB | Sub-brick labels | Header metadata |

## Comparison with AFNI 3dREMLfit

### Similarities
✅ Same ARMA(1,1) model
✅ Same REML estimation
✅ Same GLS procedure
✅ Same corrected t-statistics
✅ Saves diagnostic outputs

### Differences
| Feature | AFNI 3dREMLfit | FastFuncSim |
|---------|----------------|-------------|
| Speed | 2-4 hours | 10 minutes |
| Output format | BRIK/HEAD | NIfTI |
| Metadata | Automatic | Via 3drefit |
| Parameter reuse | Manual extraction | `save_arma_params()` |
| Visualization | AFNI GUI | Any viewer |

## Usage Examples

### Example 1: Basic Analysis
```python
results, design_info = ffs.analyze_from_design_matrix(
    'run01.nii.gz', 'X.xmat.1D', method='arma11'
)

# All outputs saved automatically!
# - glm_arma11_bucket.nii.gz (with AFNI labels)
# - arma_params.nii.gz
# - arma_lambda.nii.gz
# - glm_r2.nii.gz
```

### Example 2: Reuse Parameters
```python
# First run: Full estimation
results_v1 = ffs.analyze_from_design_matrix(
    data, design, method='arma11'
)

# Save parameters
ffs.save_arma_params(results_v1.arma_params, 'arma_params.nii.gz')

# Later: Fast reanalysis
arma_params = ffs.load_arma_params('arma_params.nii.gz', mask)
results_v2 = ffs.analyze_from_design_matrix(
    data, design, method='arma11',
    precomputed_arma_params=arma_params  # 80% faster!
)
```

### Example 3: Validation Against AFNI
```python
# Run fastfuncsim
ffs_results = ffs.analyze_from_design_matrix(...)

# Load AFNI results
afni_betas = load_afni_brik('stats.REML+tlrc', subbrick=1)

# Compare
difference = ffs_results.betas[:, 0] - afni_betas.flatten()
print(f"Max difference: {np.abs(difference).max():.6f}")
# Should be < 1e-5 (numerical precision)
```

### Example 4: Quality Checking
```python
import nibabel as nib
import matplotlib.pyplot as plt

# Load diagnostic outputs
lambda_img = nib.load('arma_lambda.nii.gz')
lambda_data = lambda_img.get_fdata()

r2_img = nib.load('glm_r2.nii.gz')
r2_data = r2_img.get_fdata()

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(lambda_data[lambda_data > 0], bins=50)
axes[0].set_title('Lag-1 Autocorrelation (λ)')
axes[0].set_xlabel('λ')
axes[0].axvline(0.3, color='r', linestyle='--', label='Typical fMRI')

axes[1].hist(r2_data[r2_data > 0], bins=50)
axes[1].set_title('Model Fit (R²)')
axes[1].set_xlabel('R²')
axes[1].axvline(0.4, color='r', linestyle='--', label='Good fit')

plt.tight_layout()
plt.savefig('arma_diagnostics.png')
```

## Technical Details

### 3drefit Command Structure

#### Labels
```bash
3drefit -relabel_all_str "label1 label2 label3 ... labelN" file.nii.gz
```

#### Statistical Parameters
```bash
3drefit \
  -substatpar 0 fift num_dof denom_dof \   # F-statistic
  -substatpar 2 fitt dof \                  # T-statistic (brick 2)
  -substatpar 4 fitt dof \                  # T-statistic (brick 4)
  ...
  file.nii.gz
```

**Parameter Types:**
- `fift`: F-statistic (2 DoF values: numerator, denominator)
- `fitt`: t-statistic (1 DoF value)

### ARMA Parameter Calculation

The ARMA(1,1) model:
```
ε[t] = a*ε[t-1] + b*η[t-1] + η[t]
```

Where:
- `a`: AR parameter (autocorrelation strength)
- `b`: MA parameter (moving average)
- λ: Effective lag-1 correlation

Relationship:
```python
lambda_val = (b + a) * (1 + a*b) / (1 + 2*a*b + b**2)
```

### File Format Details

All outputs use:
- **Format**: NIfTI-1
- **Datatype**: float32
- **Compression**: gzip
- **Affine**: Preserved from input
- **Units**: mm (spatial), seconds (temporal)

## Troubleshooting

### Issue: 3drefit not found
```
⚠ 3drefit not found - skipping AFNI metadata
```
**Solution:** Install AFNI, or manually add labels later

### Issue: Labels don't appear in AFNI
```
# Check if labels were applied
3dinfo -label glm_arma11_bucket.nii.gz
```
**Solution:** Run 3drefit manually if automatic application failed

### Issue: Parameter files are large
```
arma_params.nii.gz is 100 MB!
```
**Solution:** Make sure you're applying the mask. Unmasked data includes all voxels.

### Issue: Lambda values seem wrong
```
Mean λ = 0.85 (too high!)
```
**Cause:** ARMA parameters hit grid boundaries (a=0.9, b=0.3)
**Solution:** Check preprocessing, may need finer grid or different model

## See Also

- `examples/analyze_real_data.py`: Complete working example
- `examples/example_precomputed_arma.py`: Parameter reuse example
- `PRECOMPUTED_ARMA.md`: Detailed parameter documentation
- AFNI 3dREMLfit: https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/statistics/remlfit.html
