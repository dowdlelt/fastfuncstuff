# ARMA(1,1) Quick Reference Card

## 🚀 Quick Start (Copy-Paste Ready)

```python
import fastfuncsim as ffs
import torch

# Load your data
data = torch.load('fmri_data.pt')      # (n_voxels, n_timepoints)
design = torch.load('design_matrix.pt') # (n_timepoints, n_regressors)

# Fit ARMA(1,1) GLM
results = ffs.fit_glm_arma11(data, design, tr=2.0)

# View results
print(f"Mean (a,b): ({results.arma_params[:, 0].mean():.3f}, "
      f"{results.arma_params[:, 1].mean():.3f})")
print(f"Mean R²: {results.r2.mean():.3f}")
```

---

## 📊 Common Use Cases

### 1. Compare OLS vs ARMA
```python
comparison = ffs.compare_ols_vs_arma11(data, design, tr=2.0)
```

### 2. Extract Scanner Parameters
```python
from noise import estimate_noise_parameters_from_data

params = estimate_noise_parameters_from_data(
    data='pilot_scan.nii.gz',
    ar_order=1
)
print(params['summary'])
```

### 3. Generate Realistic Noise
```python
noise = ffs.generate_ar1_noise(
    rho=0.35,  # From your scanner
    n_timepoints=300,
    n_voxels=10000
)
```

### 4. Custom Grid Search
```python
results = ffs.fit_glm_arma11(
    data, design, tr=2.0,
    a_grid=torch.linspace(0.1, 0.9, 17),  # Finer grid
    b_grid=torch.linspace(-0.3, 0.3, 13),
    batch_size=100
)
```

---

## 📝 Results Object

```python
results.betas           # (n_voxels, n_regressors)
results.tstats          # t-statistics (corrected)
results.r2              # R² values
results.arma_params     # (n_voxels, 2) - (a, b)
results.arma_lambda     # Lag-1 correlation
results.sigma2          # Noise variance
results.residuals       # If want_residuals=True
results.predicted       # If want_predicted=True
```

---

## ⚙️ Key Parameters

```python
fit_glm_arma11(
    data,                          # (n_voxels, n_timepoints)
    design,                        # (n_timepoints, n_regressors)
    tr=2.0,                       # Repetition time
    estimate_per_voxel=True,      # True: per-voxel (accurate)
                                  # False: global (fast)
    batch_size=100,               # Voxels per batch
    a_grid=None,                  # Default: linspace(0.1, 0.9, 9)
    b_grid=None,                  # Default: linspace(-0.3, 0.3, 7)
    want_residuals=False,         # Return residuals?
    want_predicted=False,         # Return predictions?
    device=None,                  # Auto-detect MPS/CUDA/CPU
    verbose=True                  # Print progress
)
```

---

## 🎯 When to Use

| Scenario | OLS | AR(1) | ARMA(1,1) |
|----------|-----|-------|-----------|
| Quick exploration | ✅ | ❌ | ❌ |
| Design optimization | ❌ | ✅ | ❌ |
| Final analysis | ❌ | ⚠️ | ✅ |
| Publication | ❌ | ⚠️ | ✅ |
| Meta-analysis (3dMEMA) | ❌ | ⚠️ | ✅ |

---

## ⏱️ Performance Guide

### Batch Size Selection
```
n_voxels     batch_size    GPU Memory    Time
---------    ----------    ----------    ----
1,000        100           ~1 GB         12s
10,000       100           ~2-4 GB       75s
10,000       500           ~6-8 GB       45s (faster!)
100,000      100           ~10-20 GB     750s
```

**Rule of thumb**: Larger batch_size = faster, but needs more GPU memory

### Speed Estimates
- **Per voxel**: ~7-10ms (with batching)
- **1,000 voxels**: ~10-15 seconds
- **10,000 voxels**: ~1-2 minutes
- **100,000 voxels**: ~10-15 minutes

---

## 🔧 Troubleshooting

### Problem: "Cholesky failed"
```python
# Solution: Use coarser grid or AR(1) only
results = fit_glm_arma11(..., b_grid=torch.tensor([0.0]))
```

### Problem: "Out of memory"
```python
# Solution: Reduce batch size
results = fit_glm_arma11(..., batch_size=50)
```

### Problem: "Negative parameters"
```python
# Solution: Restrict grid
results = fit_glm_arma11(
    ...,
    a_grid=torch.linspace(0.1, 0.9, 9),  # Positive only
    b_grid=torch.linspace(0.0, 0.3, 4)   # Positive only
)
```

### Problem: "Too slow"
```python
# Solution 1: Use global estimation
results = fit_glm_arma11(..., estimate_per_voxel=False)

# Solution 2: Coarser grid
results = fit_glm_arma11(
    ...,
    a_grid=torch.linspace(0.1, 0.9, 5),
    b_grid=torch.linspace(-0.3, 0.3, 5)
)
```

---

## 📚 Files Reference

| File | Purpose |
|------|---------|
| `arma_glm.py` | Core implementation |
| `example_arma_glm.py` | Usage examples |
| `test_arma_glm.py` | Unit tests |
| `ARMA_GLM_README.md` | Full documentation |
| `ARMA_GLM_NOTES.md` | Implementation notes |
| `ARMA_IMPLEMENTATION_SUMMARY.md` | This guide |

---

## 🧪 Testing

```bash
# Quick test
python test_arma_glm.py

# Expected: All tests pass ✅

# Run examples
python example_arma_glm.py
```

---

## 💡 Pro Tips

1. **Extract scanner parameters first**:
   ```python
   params = estimate_noise_parameters_from_data('pilot_scan.nii.gz')
   # Use these for all future simulations
   ```

2. **Compare methods**:
   ```python
   comparison = compare_ols_vs_arma11(data, design, tr=2.0)
   # See how much ARMA corrects your t-stats
   ```

3. **Start with global estimation**:
   ```python
   # Fast preliminary analysis
   results_global = fit_glm_arma11(..., estimate_per_voxel=False)
   
   # Then per-voxel for ROIs only
   roi_data = data[roi_mask]
   results_roi = fit_glm_arma11(roi_data, ..., estimate_per_voxel=True)
   ```

4. **Use appropriate grid**:
   ```python
   # Quick: 5×5 = 25 points (~3x faster)
   # Default: 9×7 = 63 points (good balance)
   # Fine: 17×13 = 221 points (best accuracy, slower)
   ```

---

## 🎓 Theory Recap

**ARMA(1,1) correlation**:
```
r(k) = λ * a^(k-1)
λ = (b+a)(1+ab)/(1+2ab+b²)
```

**Why it matters**:
- OLS assumes independence → **inflated t-stats**
- ARMA accounts for correlation → **correct inference**

**Special cases**:
- b=0: AR(1) only
- a=0: MA(1) only
- a=b=0: White noise (OLS equivalent)

---

## 📞 Help & Support

1. **Read documentation**: `ARMA_GLM_README.md`
2. **Run tests**: `python test_arma_glm.py`
3. **Try examples**: `python example_arma_glm.py`
4. **Check code**: `arma_glm.py` (well-commented)

---

## ✅ Checklist

Before first use:
- [ ] Run `test_arma_glm.py` (verify installation)
- [ ] Run `example_arma_glm.py` (see examples)
- [ ] Read `ARMA_GLM_README.md` (understand theory)
- [ ] Extract your scanner parameters
- [ ] Compare OLS vs ARMA on your data

For each analysis:
- [ ] Choose appropriate grid resolution
- [ ] Select batch_size for your GPU
- [ ] Decide per-voxel vs global estimation
- [ ] Validate results (R² reasonable, parameters in range)
- [ ] Compare to OLS (sanity check)

---

**Version**: 0.1.0  
**Last Updated**: October 2025  
**Status**: Production Ready ✅

---

Print this card and keep it handy! 📋
