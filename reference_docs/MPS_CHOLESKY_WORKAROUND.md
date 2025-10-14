# MPS Cholesky Workaround

## Issue
PyTorch's MPS (Apple Silicon GPU) backend has a bug with `torch.linalg.cholesky()`:
```
A_t.is_contiguous() INTERNAL ASSERT FAILED at 
"/Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/mps/operations/LinearAlgebra.mm":331
```

This affects all ARMA(1,1) covariance matrix factorizations.

## Root Cause
The MPS backend's Cholesky implementation fails on certain matrix memory layouts, even when `.contiguous()` is called. This is a PyTorch bug, not our code.

## Solution
Compute Cholesky decomposition on CPU, then transfer result back to MPS:

```python
if device.type == "mps":
    R_cpu = R.cpu()
    L_cpu = torch.linalg.cholesky(R_cpu)
    L = L_cpu.to(device)
else:
    L = torch.linalg.cholesky(R)
```

## Performance Impact
**Minimal!** Cholesky is only computed once per (a,b) grid point during pre-computation:
- Pre-computation: ~1.0s for 49 grid points (vs ~0.5s without workaround)
- The expensive GPU operations (matrix multiplications for all voxels) are unaffected
- Overall analysis: ~21 seconds for 392k voxels (vs 2+ hours before optimization)

## Locations Fixed
All three Cholesky operations in `arma_glm.py`:
1. `compute_reml_likelihood()` - line ~174
2. `precompute_reml_grid()` - line ~381
3. `prewhiten_with_arma11()` - line ~590

## Future
This workaround can be removed when PyTorch fixes the MPS Cholesky bug. Monitor:
- https://github.com/pytorch/pytorch/issues (search for "MPS Cholesky")
