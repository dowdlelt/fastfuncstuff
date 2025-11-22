# ARMA Grid Search Performance Analysis

## Executive Summary

Total ARMA(1,1) analysis time for 1000 voxels: **~3.6 seconds**

### Time Breakdown by Phase

| Phase | Time | % Total | Description |
|-------|------|---------|-------------|
| **Grid Precomputation** | 2.3s | 64.7% | Build & factorize 151 correlation matrices |
| **Grid Search** | 0.3s | 7.6% | Evaluate 1000 voxels × 151 grid points |
| **Final GLM + Overhead** | 1.0s | 27.7% | Group by (a,b), compute stats, data loading |

### Grid Search Loop Breakdown (0.27s total)

Within the grid search loop (151 iterations, 1000 voxels per iteration):

| Operation | Time | % of Loop | Calls | Description |
|-----------|------|-----------|-------|-------------|
| Prewhiten data | 0.109s | 38.4% | 151 | Y @ L_inv.T matrix multiplication |
| Update best | 0.082s | 29.1% | 151 | Comparison + parameter assignment |
| Solve beta | 0.047s | 16.6% | 151 | Linear solve for coefficients |
| Compute likelihood | 0.019s | 6.6% | 151 | REML likelihood calculation |
| Compute residuals | 0.016s | 5.7% | 151 | RSS computation |
| Compute X_w'Y_w | 0.009s | 3.3% | 151 | Matrix multiplication |
| Get grid data | 0.001s | 0.5% | 151 | Dictionary lookups |

## Key Findings

### 1. Grid Precomputation is the Bottleneck (65%)

The Cholesky decomposition of 151 correlation matrices takes **~2.1s of the 2.3s** precomputation time.

**Why this is already optimized:**
- Uses batched Cholesky decomposition (all 151 matrices at once)
- Computed on CPU with optimized BLAS (MKL)
- Results cached and reused for all voxels
- This is the AFNI approach - unavoidable overhead

**Could we optimize further?**
- ❌ **Reduce grid size**: Would hurt accuracy (already at AFNI's default Grid 3)
- ❌ **Skip precomputation**: Would require 1000× recomputation (much slower)
- ⚠️ **GPU Cholesky**: Already using CPU (optimal for this size)
- ✅ **Reuse across runs**: If analyzing multiple runs, precompute once and reuse

### 2. Grid Search Loop is Highly Efficient (8%)

Only 0.27s to evaluate 151,000 likelihood evaluations (1000 voxels × 151 grid points).

**Already optimized:**
- Vectorized operations (all 1000 voxels simultaneously)
- Precomputed matrices reused
- GPU-accelerated matrix operations
- Minimal Python overhead

**Top consumer: Prewhitening (38%)**
- `Y @ L_inv.T`: (1000, 720) @ (720, 720).T matrix multiplication
- This is an O(n²m) operation, unavoidable for grid search
- Already using optimized cuBLAS on GPU

**Second: Update best (29%)**
- Comparison + conditional assignment across 1000 voxels
- Already minimal overhead
- Could potentially fuse with likelihood computation, but gains would be negligible

### 3. Final GLM Phase is Reasonable (28%)

Includes:
- Grouping 1000 voxels by their optimal (a,b) parameters (36 unique groups)
- Prewhitening and GLM fitting for each group
- Statistics computation (F-stats, t-stats)
- GLT computation
- Data loading and design matrix parsing

**Already optimized:**
- Voxels grouped by (a,b) to eliminate redundant Cholesky decompositions
- Recent 25% speedup from F-stat formula optimization
- Batch processing within groups

## Comparison to OLS

- **OLS**: ~0.5s for 1000 voxels
- **ARMA**: ~3.6s for 1000 voxels
- **Overhead**: 7.2× slower (expected given grid search)

This is **excellent** for ARMA analysis. The grid search evaluates 151 different noise models per voxel, so a 7× slowdown is very reasonable.

## Potential Optimizations

### High Impact (if applicable)

1. **Reuse precomputed grid across multiple runs** (✅ **RECOMMENDED**)
   - If analyzing multiple runs with same design matrix
   - Precompute once, save to disk, reload
   - Would reduce time from 3.6s → 1.3s (64% speedup)
   - **Implementation**: Add caching for precomputed_grid keyed by (design_hash, grid_params)

2. **Early stopping in grid search** (⚠️ **RISKY**)
   - AFNI uses "power-of-2 descent" hierarchical search
   - Could reduce grid evaluations from 151 → ~30
   - BUT: May miss global optimum (we currently find better params than AFNI in some cases)
   - Would save ~0.2s (6% total speedup)
   - **Not recommended**: Exhaustive search is more accurate

### Medium Impact

3. **Fuse likelihood computation with update** (⚠️ **COMPLEX**)
   - Combine likelihood calculation + parameter update in single kernel
   - Could eliminate some intermediate allocations
   - Potential savings: ~0.05s (1-2% speedup)
   - **Not recommended**: Code complexity vs minimal gain

4. **Use float16 for grid precomputation** (⚠️ **ACCURACY RISK**)
   - Cholesky decomposition in half precision
   - Potential 2× speedup on modern GPUs with tensor cores
   - BUT: May have numerical stability issues
   - **Not recommended** without extensive validation

### Low Impact

5. **Optimize dict lookup in grid search** (minimal)
   - Currently 0.5% of loop time
   - Could use list indexing instead
   - Potential savings: <0.001s (negligible)

6. **Profile final GLM phase** (informational)
   - Could add detailed timing to grouping + stats computation
   - Unlikely to find major optimizations (already optimized)

## Recommendations

### For Current Dataset (1000 voxels)

**Status**: Already well-optimized, no action needed.

The 3.6s for ARMA analysis is reasonable given:
- 151-point exhaustive grid search
- 1000 voxels
- 720 timepoints
- Full GLM with statistics

### For Large Datasets (>100k voxels)

**Consider**: Precomputed grid caching
- When analyzing multiple runs with same design
- One-time 2.3s cost, reuse for all subsequent runs
- Saves 64% of time per additional run

**Already implemented**: Adaptive batching strategy
- Automatically switches to batched grid search for large datasets
- Prevents GPU OOM
- Maintains performance

### For Real-Time Analysis

**Not applicable**: ARMA analysis is inherently expensive
- Exhaustive grid search is necessary for accuracy
- If speed is critical, use OLS (7× faster)
- ARMA is for final, publication-quality analysis

## Conclusion

The ARMA grid search is **already highly optimized**:

✅ Batched Cholesky decomposition (unavoidable, already optimal)
✅ Vectorized grid search (all voxels simultaneously)
✅ GPU-accelerated matrix operations
✅ Minimal Python overhead
✅ Voxel grouping to eliminate redundant computations
✅ Optimized F-statistic formula (recent 25% speedup)

**Major bottleneck**: Grid precomputation (65%)
- This is unavoidable for ARMA(1,1) analysis
- Already using optimal implementation (batched, CPU BLAS)
- Could cache and reuse across runs (64% savings)

**Grid search loop**: Highly efficient (8%)
- 151,000 likelihood evaluations in 0.27s
- No significant optimization opportunities without sacrificing accuracy

**Overall**: Performance is excellent for ARMA analysis. Focus optimization efforts elsewhere if needed.
