# ARMA(1,1) Prewhitening for GLM Analysis

**Purpose**: Final data analysis with improved statistical accuracy
**Priority**: HIGH for publication-quality results
**Status**: Not yet implemented (Phase 3.3)

---

## Overview

ARMA(1,1) prewhitening corrects for temporal autocorrelation in fMRI data to produce:
- **More accurate t-statistics**: Proper accounting of temporal dependencies
- **Better beta estimates**: Reduced variance through GLS
- **Correct p-values**: Prevents inflated false positives from autocorrelation

This is the **ANALYSIS** side (not simulation). Use this after data collection for final GLM fits.

---

## What is ARMA(1,1)?

From AFNI 3dREMLfit documentation (see ARMA_notes.txt):

**Model**: Noise correlation r(k) = λ * a^(k-1) where:
- `a` = AR parameter (decay factor, typically 0.2-0.9)
- `b` = MA parameter (typically -0.3 to +0.3)
- `λ = (b+a)(1+ab)/(1+2ab+b²)` = lag-1 correlation

**Special cases**:
- AR(1): b=0, r(k) = a^k
- MA(1): a=0, r(k) = b for k=1, else 0
- ARMA(1,1): Full flexibility

---

## AFNI 3dREMLfit Method

**Source**: https://github.com/afni/afni/blob/master/src/3dREMLfit.c

### Algorithm Overview:

1. **Initial OLS Fit**: Get residuals from ordinary least squares
2. **Grid Search for (a,b)**:
   - Try discrete grid of (a,b) values (default: 11×11 = 121 points)
   - For each (a,b):
     - Build ARMA(1,1) covariance matrix R(a,b)
     - Compute REML likelihood L(a,b)
   - Select (a,b) that minimizes L(a,b)
3. **GLS Fit with Optimal (a,b)**:
   - Prewhiten: Y* = L^(-1) Y, X* = L^(-1) X (where R = L L^T)
   - Solve: β = (X*^T X*)^(-1) X*^T Y*
4. **Compute Statistics**:
   - Variance: Var(β) = σ² (X*^T X*)^(-1)
   - t-statistics: t = β / sqrt(Var(β))

### REML Likelihood Function:

```
L(a,b) = log(det(R)) + log(det(X^T R^(-1) X)) + (n-m)log(Y^T P Y)
```

where:
- R(a,b) = ARMA(1,1) covariance matrix (Toeplitz)
- P(a,b) = prewhitening projection matrix
- n = number of timepoints
- m = number of regressors

---

## GPU Implementation Strategy

### Step 1: ARMA(1,1) Covariance Matrix (GPU)

```python
def build_arma11_covariance(a: float, b: float, n: int, device) -> torch.Tensor:
    """
    Build ARMA(1,1) covariance matrix R

    R[i,j] = λ * a^|i-j| where λ = (b+a)(1+ab)/(1+2ab+b²)

    This is a Toeplitz matrix (constant along diagonals)
    """
    lam = (b + a) * (1 + a*b) / (1 + 2*a*b + b**2 + 1e-10)

    # Power sequence: [1, a, a², a³, ..., a^(n-1)]
    powers = torch.pow(a, torch.arange(n, device=device))

    # Toeplitz matrix from powers
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
    R = lam * torch.pow(a, torch.abs(i - j).float())

    return R
```

**GPU Speed**: Matrix construction is O(n²) but very fast (~1ms for n=300)

### Step 2: Cholesky Decomposition (GPU)

```python
def prewhiten_with_arma11(X: torch.Tensor, Y: torch.Tensor,
                         a: float, b: float) -> Tuple:
    """
    Prewhiten design and data using ARMA(1,1) covariance

    Returns: X*, Y* where * means prewhitened
    """
    n = X.shape[0]
    R = build_arma11_covariance(a, b, n, X.device)

    # Cholesky: R = L L^T
    L = torch.linalg.cholesky(R)
    L_inv = torch.linalg.inv(L)

    # Prewhiten
    X_white = L_inv @ X
    Y_white = L_inv @ Y

    return X_white, Y_white
```

**GPU Speed**: Cholesky is O(n³) but PyTorch is highly optimized (~10ms for n=300)

### Step 3: REML Grid Search (GPU)

```python
def reml_grid_search(X: torch.Tensor, Y: torch.Tensor,
                     a_grid: torch.Tensor = None,
                     b_grid: torch.Tensor = None,
                     device=None) -> Tuple[float, float]:
    """
    Find optimal (a,b) via REML grid search

    Default grid: a in [0.1, 0.2, ..., 0.9], b in [-0.3, -0.2, ..., 0.3]
    """
    if a_grid is None:
        a_grid = torch.linspace(0.1, 0.9, 9, device=device)
    if b_grid is None:
        b_grid = torch.linspace(-0.3, 0.3, 7, device=device)

    n, m = X.shape
    best_likelihood = float('inf')
    best_a, best_b = 0.5, 0.0

    # Grid search (can parallelize across voxels)
    for a in a_grid:
        for b in b_grid:
            # Skip invalid combinations (ensure λ > 0)
            lam = (b + a) * (1 + a*b) / (1 + 2*a*b + b**2 + 1e-10)
            if lam <= 0:
                continue

            # Build covariance
            R = build_arma11_covariance(a.item(), b.item(), n, device)

            # Compute REML likelihood
            L_val = compute_reml_likelihood(X, Y, R)

            if L_val < best_likelihood:
                best_likelihood = L_val
                best_a, best_b = a.item(), b.item()

    return best_a, best_b
```

**GPU Speed**:
- Sequential: 9×7 = 63 grid points × 10ms = 630ms per voxel
- Parallel across voxels: Can evaluate all voxels simultaneously (huge speedup!)

### Step 4: REML Likelihood Computation (GPU)

```python
def compute_reml_likelihood(X: torch.Tensor, Y: torch.Tensor,
                           R: torch.Tensor) -> float:
    """
    Compute REML log-likelihood

    L = log(det(R)) + log(det(X^T R^(-1) X)) + (n-m)log(Y^T P Y)
    """
    n, m = X.shape

    # Prewhiten
    L_chol = torch.linalg.cholesky(R)
    L_inv = torch.linalg.inv(L_chol)
    X_w = L_inv @ X
    Y_w = L_inv @ Y

    # Term 1: log(det(R)) = 2 * sum(log(diag(L)))
    term1 = 2 * torch.sum(torch.log(torch.diag(L_chol)))

    # Term 2: log(det(X^T R^(-1) X)) = log(det(X_w^T X_w))
    XwTXw = X_w.T @ X_w
    term2 = torch.log(torch.linalg.det(XwTXw + 1e-6 * torch.eye(m, device=X.device)))

    # Term 3: (n-m) log(Y^T P Y)
    beta_w = torch.linalg.solve(XwTXw, X_w.T @ Y_w)
    residuals_w = Y_w - X_w @ beta_w
    term3 = (n - m) * torch.log(torch.sum(residuals_w**2) + 1e-10)

    return term1 + term2 + term3
```

**GPU Speed**: All matrix ops native PyTorch, very fast

### Step 5: Complete ARMA(1,1) GLM Fit (GPU)

```python
def fit_glm_arma11(data: torch.Tensor,
                   design: torch.Tensor,
                   tr: float,
                   a_grid: torch.Tensor = None,
                   b_grid: torch.Tensor = None,
                   estimate_per_voxel: bool = True,
                   device=None) -> GLMResults:
    """
    Fit GLM with ARMA(1,1) prewhitening (AFNI 3dREMLfit style)

    Parameters
    ----------
    data : (n_timepoints, n_voxels)
    design : (n_timepoints, n_regressors)
    estimate_per_voxel : bool
        If True: Estimate (a,b) separately per voxel
        If False: Estimate (a,b) once from mean timeseries (faster)

    Returns
    -------
    GLMResults with:
        betas : (n_voxels, n_regressors)
        tstats : t-statistics (properly corrected)
        arma_params : (n_voxels, 2) - (a, b) per voxel
    """
    n_timepoints, n_voxels = data.shape
    n_regressors = design.shape[1]

    betas = torch.zeros(n_voxels, n_regressors, device=device)
    tstats = torch.zeros(n_voxels, n_regressors, device=device)
    arma_params = torch.zeros(n_voxels, 2, device=device)

    if estimate_per_voxel:
        # Per-voxel ARMA estimation (most accurate)
        for v in range(n_voxels):
            y_v = data[:, v]

            # REML grid search for this voxel
            a_opt, b_opt = reml_grid_search(design, y_v, a_grid, b_grid, device)
            arma_params[v] = torch.tensor([a_opt, b_opt], device=device)

            # Prewhiten
            X_w, y_w = prewhiten_with_arma11(design, y_v, a_opt, b_opt)

            # GLS fit
            XwTXw = X_w.T @ X_w
            beta_v = torch.linalg.solve(XwTXw, X_w.T @ y_w)
            betas[v] = beta_v

            # t-statistics
            residuals_w = y_w - X_w @ beta_v
            sigma2 = torch.sum(residuals_w**2) / (n_timepoints - n_regressors)
            var_beta = sigma2 * torch.linalg.inv(XwTXw)
            tstats[v] = beta_v / torch.sqrt(torch.diag(var_beta))

    else:
        # Global ARMA estimation (faster, less accurate)
        y_mean = data.mean(dim=1)
        a_opt, b_opt = reml_grid_search(design, y_mean, a_grid, b_grid, device)
        arma_params[:] = torch.tensor([a_opt, b_opt], device=device)

        # Prewhiten once
        X_w, _ = prewhiten_with_arma11(design, y_mean, a_opt, b_opt)

        # Batch GLS for all voxels
        for v in range(n_voxels):
            y_w = prewhiten_with_arma11(design, data[:, v], a_opt, b_opt)[1]
            XwTXw = X_w.T @ X_w
            beta_v = torch.linalg.solve(XwTXw, X_w.T @ y_w)
            betas[v] = beta_v

            # ... (compute tstats as above)

    return GLMResults(betas=betas, tstats=tstats, arma_params=arma_params, ...)
```

---

## GPU Acceleration Strategy

### Key Insight: Parallelize Across Voxels

**Sequential** (1 CPU core):
- 10,000 voxels × 630ms = 6,300 seconds = 105 minutes ❌

**Parallel** (GPU with batching):
- Evaluate 100 voxels simultaneously
- 10,000 voxels / 100 = 100 batches × 630ms = 63 seconds ✓

### Implementation:

```python
def batch_reml_grid_search(X: torch.Tensor, Y_batch: torch.Tensor,
                          a_grid: torch.Tensor, b_grid: torch.Tensor):
    """
    Parallel REML grid search across voxels

    Y_batch : (n_timepoints, n_voxels_batch) - e.g., 100 voxels

    Returns: (n_voxels_batch, 2) - (a, b) for each voxel
    """
    n_timepoints, n_voxels_batch = Y_batch.shape
    best_params = torch.zeros(n_voxels_batch, 2, device=Y_batch.device)

    # Vectorize across (a,b) grid AND voxels
    # This is where GPU shines!
    for a in a_grid:
        for b in b_grid:
            R = build_arma11_covariance(a.item(), b.item(), n_timepoints, Y_batch.device)

            # Compute likelihood for ALL voxels at once
            likelihoods = torch.stack([
                compute_reml_likelihood(X, Y_batch[:, v], R)
                for v in range(n_voxels_batch)
            ])

            # Update best for each voxel independently
            # ... (track best_params per voxel)

    return best_params
```

---

## When to Use ARMA(1,1) vs AR(1)?

### AR(1) Only (Simpler):
- **Pros**: Faster (closed-form estimation), one parameter
- **Cons**: Less flexible, may underfit complex autocorrelation
- **Use when**: Quick analysis, exploratory work

### ARMA(1,1) (Better):
- **Pros**: More accurate, captures richer autocorrelation patterns
- **Cons**: Slower (grid search), two parameters
- **Use when**: Final analysis, publication, maximum accuracy

### From AFNI Documentation:
> "ARMA(1,1) is hard-coded into 3dREMLfit -- there is no way to use a more elaborate model"

So ARMA(1,1) is the "gold standard" compromise between:
- Simplicity: Only 2 parameters
- Flexibility: Captures most fMRI autocorrelation
- Speed: Fast enough with grid search

---

## Comparison to Existing metrics_empirical.py

**Current Implementation** (metrics_empirical.py):
- Uses AR(1) only (ρ parameter)
- Estimates ρ via lag-1 autocorrelation
- Simpler, faster for design optimization

**New ARMA(1,1) Implementation**:
- Uses ARMA(1,1) (a, b parameters)
- Estimates via REML grid search
- More accurate for final data analysis

**Relationship**:
- AR(1) = ARMA(1,1) with b=0
- Current code is a special case of new code
- Can keep both: AR(1) for fast design evaluation, ARMA(1,1) for final analysis

---

## Implementation Checklist

### Core Functions:
- [ ] `build_arma11_covariance(a, b, n, device)` - Build Toeplitz R matrix
- [ ] `prewhiten_with_arma11(X, Y, a, b)` - Cholesky prewhitening
- [ ] `compute_reml_likelihood(X, Y, R)` - REML objective function
- [ ] `reml_grid_search(X, Y, a_grid, b_grid)` - Find optimal (a,b)
- [ ] `batch_reml_grid_search(X, Y_batch, a_grid, b_grid)` - Parallel version
- [ ] `fit_glm_arma11(data, design, tr, ...)` - Complete GLM with ARMA(1,1)

### Integration:
- [ ] Add to `glm_core.py` as `fit_glm(..., prewhiten='arma11')`
- [ ] Add to `__init__.py` exports
- [ ] Create example script `example_arma_glm.py`
- [ ] Add tests comparing to AFNI 3dREMLfit outputs

### Documentation:
- [ ] Add section to README about ARMA(1,1) prewhitening
- [ ] Document (a,b) interpretation
- [ ] Explain when to use vs AR(1)
- [ ] Performance benchmarks (GPU speedup)

---

## Performance Targets

**Goal**: <5 minutes for 10,000 voxels on GPU

**Estimated**:
- Grid search: 63 seconds (with batching)
- GLS fits: 10 seconds
- Total: ~75 seconds ✓

**Comparison to AFNI 3dREMLfit**:
- AFNI (CPU, 1 core): ~10-30 minutes for whole brain
- FastFuncSim (GPU, MPS/CUDA): ~1-2 minutes
- **Speedup: 5-30x** 🚀

---

## Testing Strategy

### Validation Against AFNI:
1. Run AFNI 3dREMLfit on test dataset
2. Run FastFuncSim ARMA(1,1) on same data
3. Compare:
   - Estimated (a,b) parameters
   - Beta estimates
   - t-statistics
   - Residuals

### Synthetic Data Tests:
1. Generate data with known ARMA(1,1) parameters
2. Fit with `fit_glm_arma11()`
3. Check if estimated (a,b) matches true values
4. Verify t-statistics are properly calibrated

---

## References

1. **AFNI 3dREMLfit**: https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/statistics/remlfit.html
2. **Source code**: https://github.com/afni/afni/blob/master/src/3dREMLfit.c
3. **Math notes**: https://afni.nimh.nih.gov/pub/dist/doc/misc/3dREMLfit/3dREMLfit_mathnotes.pdf
4. **Woolrich et al. (2001)**: "Temporal autocorrelation in univariate linear modeling of FMRI data"
5. **Worsley & Friston (1995)**: "Analysis of fMRI time-series revisited—again"

---

## Summary

ARMA(1,1) prewhitening is the **gold standard** for final fMRI GLM analysis:
- Corrects for temporal autocorrelation
- Produces accurate t-statistics
- GPU-acceleratable (5-30x speedup over AFNI)
- Hard but feasible to implement

**Next Step**: Implement core functions, test on synthetic data, validate against AFNI outputs.
