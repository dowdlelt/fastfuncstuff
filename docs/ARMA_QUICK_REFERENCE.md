# ARMA(1,1) Quick Reference

## Basic Usage

```python
import fastfuncstuff as ffs

results = ffs.fit_glm_arma11(data, design, tr=2.0)

print(f"Mean (a,b): ({results.arma_params[:, 0].mean():.3f}, "
      f"{results.arma_params[:, 1].mean():.3f})")
print(f"Mean R2: {results.r2.mean():.3f}")
```


## Parameters

```python
ffs.fit_glm_arma11(
    data,                          # (n_voxels, n_timepoints)
    design,                        # (n_timepoints, n_regressors)
    tr=2.0,                        # repetition time in seconds
    estimate_per_voxel=True,       # True: per-voxel params (accurate)
                                   # False: global params (fast)
    batch_size=100,                # voxels per GPU batch
    a_grid=None,                   # default: linspace(0.1, 0.9, 9)
    b_grid=None,                   # default: linspace(-0.3, 0.3, 7)
    want_residuals=False,
    want_predicted=False,
    device=None,                   # auto-detect MPS/CUDA/CPU
    verbose=True,
)
```


## Results Object

```python
results.betas           # (n_voxels, n_regressors) -- GLS coefficients
results.tstats          # t-statistics (corrected for autocorrelation)
results.r2              # R-squared values
results.arma_params     # (n_voxels, 2) -- (a, b) parameters
results.arma_lambda     # lag-1 autocorrelation
results.sigma2          # noise variance estimates
results.residuals       # if want_residuals=True
results.predicted       # if want_predicted=True
```


## When to Use What

| Scenario | OLS | ARMA(1,1) |
|----------|-----|-----------|
| Quick exploration / prototyping | good | unnecessary |
| Design optimization | good | unnecessary |
| Final analysis for publication | inadequate | recommended |
| Group analysis (e.g., 3dMEMA) | inadequate | recommended |

OLS assumes temporal independence, which inflates t-statistics. ARMA(1,1)
models the autocorrelation structure of fMRI residuals and produces
correct standard errors.


## Performance

Larger batch_size is faster but uses more GPU memory:

```
n_voxels     batch_size    approx. time
---------    ----------    ------------
1,000        100           ~12s
10,000       100           ~75s
10,000       500           ~45s
100,000      100           ~750s
```


## Troubleshooting

**Cholesky failed**: Use a coarser grid or restrict to AR(1) only:
```python
results = ffs.fit_glm_arma11(..., b_grid=torch.tensor([0.0]))
```

**Out of memory**: Reduce batch size:
```python
results = ffs.fit_glm_arma11(..., batch_size=50)
```

**Too slow**: Use global estimation first, then per-voxel on ROIs:
```python
results_global = ffs.fit_glm_arma11(..., estimate_per_voxel=False)
```


## Theory

ARMA(1,1) autocorrelation at lag k:
```
r(k) = lambda * a^(k-1)
lambda = (b + a)(1 + ab) / (1 + 2ab + b^2)
```

Special cases: b=0 gives AR(1), a=0 gives MA(1), a=b=0 gives white noise.


## Related Files

- Library: `fastfuncstuff/glm/arma.py`
- CLI: `ffs_reml` (installed command)
- Tests: `tests/test_arma_glm.py`, `tests/test_arma_glm_comprehensive.py`
- Example: `examples/example_arma_glm.py`
