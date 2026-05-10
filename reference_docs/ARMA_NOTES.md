# ARMA(1,1) REML Implementation Notes

## Overview

`ffs_reml` (CLI) / `fit_glm_arma11()` (API) implements AFNI-style ARMA(1,1)
prewhitened GLM using REML estimation of temporal autocorrelation parameters.
The core math follows AFNI's `3dREMLfit.c` and the accompanying tech note
(`afni.nimh.nih.gov/pub/dist/doc/misc/3dREMLfit/3dREMLfit.pdf`, RW Cox).

---

## 1. ARMA(1,1) Covariance Model

For an AR parameter `a` and MA parameter `b`, the lag-k correlation is:

```
γ(0) = (1 + b² + 2ab) / (1 - a²)          # variance
γ(1) = (a + b)(1 + ab) / (1 + b² + 2ab)   # lag-1 autocorrelation = λ
γ(k) = λ · aᵏ⁻¹    for k ≥ 2              # geometric decay
```

The correlation matrix R is Toeplitz with diagonal=1 and off-diagonals from γ.

**Validity constraints (same as AFNI):**

- `|a| < 1` and `|b| < 1` (stationarity)
- `γ(0) > 0`
- `λ = γ(1)/γ(0) ≥ 0` (AFNI default; `-NEGcor` removes this)

The λ ≥ 0 constraint filters out physically implausible anticorrelated noise
and is the reason many high-b, low-a grid points are excluded.

---

## 2. REML Likelihood (What Is Being Minimized)

The restricted log-likelihood for ARMA parameters θ = (a, b) is:

```
-2 logL_REML(θ) ∝  log|R| + log|X'R⁻¹X| + (N-p) log(σ̂²)
```

where:
- R = ARMA correlation matrix (Toeplitz, depends on θ)
- X = design matrix, p = number of regressors, N = number of timepoints
- σ̂² = RSS_w / (N-p) = whitened residual variance

Since `(N-p) log(RSS_w / (N-p)) = (N-p) log(RSS_w) − const`, and constants
don't affect the argmin, our implemented criterion is:

```python
likelihood = logdet_Rcorr + logdet_XwTXw + (N - p) * log(RSS_w)
```

We **minimize** this quantity. `best_likelihoods` is initialized to `+inf` and
updated when a smaller value is found.

**Why this matches AFNI:** 3dREMLfit.c minimizes the same criterion. The key
insight is that `log|Σ| = log|σ²R| = N log σ² + log|R|`, and σ² gets absorbed
into the RSS term, so only the correlation matrix appears in the logdets.

---

## 3. Default Grid (AFNI "Grid 3" — Medium Resolution)

```python
a ∈ [0.0, 0.9]   in steps of 0.1  → 10 values
b ∈ [-0.9, 0.9]  in steps of 0.1  → 19 values
```

After filtering by validity constraints, typically ~117–130 grid points survive.
The point `(a=0, b=0)` (white noise) is always guaranteed to be in the grid.

**AFNI's actual grid:** 3dREMLfit uses an identical grid by default (its "Grid 3").
The grid can be extended with `-Grid <n>` in AFNI (denser), or via `--a-grid`
and `--b-grid` in `ffs_reml`.

---

## 4. Pre-Computation Phase (One-Time, All Grid Points)

`precompute_reml_grid()` runs once before processing any voxels. It converts
the raw (a, b) grid into cached matrices that are reused for every voxel.

### Phase 1: Build All Covariance Matrices (Vectorized)

`build_arma11_covariance_batch()` constructs all ~130 Toeplitz matrices as a
batched tensor `R_batch: (n_valid, N, N)` using broadcasting — no Python loop.

### Phase 2: Batch Cholesky

```python
L_batch = torch.linalg.cholesky(R_batch)  # (n_valid, N, N), lower triangular
```

Done on CPU (default) to avoid GPU OOM on large N. Still fast (~1–5s for
N=1340, n_valid=151).

### Phase 3: Log|R| from Diagonal of L

```python
logdet_Rcorr = 2 * sum(log(diag(L) + 1e-10))   # (n_valid,)
```

Computed from the Cholesky diagonal before L is freed. The `+1e-10` guard
prevents log(0) on near-singular cases.

### Phase 4: Compute L_inv via Batched Triangular Solve

**Key recent change.** Previously stored L directly and called
`solve_triangular(L, Y)` per voxel in the hot path. This caused Intel oneMKL
`DLASWP` errors (stride-layout rejection for float64 batched strides).

New approach: compute L_inv once, store it, then whitening is a plain GEMM:

```python
eye_n = torch.eye(N, ...)
L_inv_batch = torch.linalg.solve_triangular(
    L_batch, eye_n.unsqueeze(0).expand(n_valid, -1, -1), upper=False
)   # (n_valid, N, N)
del L_batch  # L no longer needed
```

Per-voxel whitening becomes: `Y_w = L_inv @ Y` (GEMM, no triangular solve).

**Memory cost:** For N=1340, n_valid=151, float64 → L_inv_batch ≈ 2.7 GB.
This is a one-time allocation amortized over all voxels.

### Phase 5: Prewhiten Design (GEMM)

```python
X_w_batch = bmm(L_inv_batch, X.expand(n_valid, -1, -1))  # (n_valid, N, p)
```

### Phase 6: QR Decomposition — Always (New)

**Key recent change.** Previously had two paths: QR (slower, stable) and
X'X+slogdet (faster, but called DGETRF/DLASWP internally → MKL crashes).

Now QR always:

```python
Q_batch, R_qr_batch = torch.linalg.qr(X_w_batch)
logdet_XwTXw = 2 * sum(log(|diag(R_qr)| + 1e-10))   # (n_valid,)
del R_qr_batch   # Not stored — GLS path recomputes fresh QR
```

**Why QR is always better:**
1. Avoids squaring the condition number (QR vs X'X solve)
2. Never internally calls DGETRF/DLASWP (eliminated MKL crash)
3. logdet from R-diagonal is the same computation AFNI does
4. Negligible extra cost — batched QR is fast on modern BLAS

### Stored Per Grid Point

```python
precomputed[(a, b)] = {
    "L_inv":          (N, N)    # for GEMM whitening
    "X_w":            (N, p)    # for GLS fitting path
    "Q":              (N, p)    # for Pythagorean RSS
    "logdet_Rcorr":   scalar
    "logdet_XwTXw":   scalar
}
```

---

## 5. Grid Search Phase (Per Voxel, Device-Dependent)

### 5a. GPU Path — Exhaustive Batched Search

`batch_reml_grid_search()` processes thousands of voxels in parallel. All
voxels must evaluate the same grid points (GPUs can't diverge per-thread),
so the full grid is evaluated exhaustively.

**Per chunk of grid points:**

```python
# Stack L_inv and Q for chunk (memory-adaptive chunking)
Y_w_all = bmm(L_inv_stack, Y_batch_expanded)     # (n_chunk, N, n_vox)
Qt_Yw   = bmm(Q_stack.T, Y_w_all)                # (n_chunk, p, n_vox)

# Pythagorean RSS — no betas computed in search!
rss_all = Y_w_all.pow(2).sum(dim=1) - Qt_Yw.pow(2).sum(dim=1)  # (n_chunk, n_vox)

likelihood = logdet_R + logdet_XwTXw + (N-p) * log(rss_all + 1e-10)
```

Smart grid ordering evaluates the empirically most-common (a,b) pairs first
(e.g. `(0.0, 0.3)` covers ~30% of voxels). Optional early stopping at 80%
convergence is available but off by default.

### 5b. CPU Path — Hierarchical Search (AFNI-Style)

`_cpu_hierarchical_reml_search()` implements AFNI's power-of-2 descent:

```
step = 8 → 4 → 2 → 1
```

Each voxel processed sequentially with an independent search window that
narrows around the current best. Evaluates ~40–50 grid points vs. ~130
(2–3× speedup vs. exhaustive). Works because CPUs can diverge per-voxel.

**Evaluated via `_evaluate_single_param()`:**

```python
Y_w  = L_inv @ Y_voxel          # GEMM
Qt_Yw = Q.T @ Y_w
rss  = Y_w.pow(2).sum() - Qt_Yw.pow(2).sum()   # Pythagorean RSS
likelihood = logdet_R + logdet_XwTXw + (N-p) * log(max(rss, 1e-10))
```

### 5c. `search_voxels_precomputed_grid()` — Legacy/Alternative

Sequential over grid points, vectorized over voxels. Used when the precomputed
grid is passed in directly (non-batched mode). Same Pythagorean RSS math.

---

## 6. Pythagorean RSS — Key Mathematical Identity

**Old approach:** Compute betas, then residuals, then RSS (3 ops, needs betas).

**New approach:**

```
X_w = Q R  (QR decomp, Q orthonormal, R upper triangular)
β̂  = R⁻¹ Q' Y_w
Ŷ_w = X_w β̂ = Q Q' Y_w
e_w = (I - QQ') Y_w

RSS = ‖e_w‖² = ‖Y_w‖² - ‖Q'Y_w‖²     (Pythagoras, since ‖Qv‖ = ‖v‖)
```

No betas needed during grid search. Same result AFNI gets because AFNI also
uses QR internally for 3dREMLfit. Saves ~30% of search time on GPU.

---

## 7. GLS Fitting Phase (After ARMA Estimation)

Once optimal (a*, b*) is determined per voxel, voxels are grouped by shared
(a*, b*) pair and processed together (typically a handful of dominant pairs
cover most voxels).

**For each (a*, b*) group:**

1. Retrieve `L_inv` and `X_w` from precomputed cache
2. Whiten data: `Y_w = (L_inv @ Y_batch_dev).T`   ← GEMM if cache hit
   OR: `Y_w = solve_triangular(L, Y_batch_dev)` ← if computed on-demand
3. Solve GLS via QR (`use_qr=True`) or X'X (`use_qr=False`):
   - **QR path:** `Q'Y_w → solve R β = Q'Y_w → betas`
   - **X'X path:** `XwTXw_inv @ (X_w.T @ Y_w) → betas`
4. Compute σ̂² = RSS_w / (N-p) with **float64 accumulation** (eliminates
   catastrophic cancellation: `resid_w.to(float64).pow(2).sum()`)
5. Compute t-stats: `t_i = β_i / (σ̂ sqrt((X'X)⁻¹_ii))`
6. Compute R², partial R², semi-partial R² if requested

**`_using_l_inv` flag** distinguishes the whitening path:
- `True` (cache hit): `y_w = (L_inv @ Y_batch_dev).T`   — GEMM
- `False` (on-demand): `y_w = solve_triangular(L, Y_batch_dev).T`

---

## 8. Outputs

### Rbuck (main stats bucket)

Per-regressor: `[beta_0, tstat_0, beta_1, tstat_1, ...]`
+ Full-model F-stat (if applicable)

Written by `write_glm_bucket_as_nifti()` in `glm/outputs.py`.

### Rvar (ARMA parameter estimates)

Sub-briks (in order):
1. `a` — AR parameter per voxel
2. `b` — MA parameter per voxel
3. `lam` — λ = lag-1 autocorrelation per voxel
4. `StDev` — sqrt(σ̂²) per voxel
5. `-LogLik` (if saved) — minimum REML likelihood per voxel
6. `LjungBox` — χ² statistic (if computed)

Both Rbuck and Rvar now inherit the AFNI SCENE_DATA[0]/TEMPLATE_SPACE from
the source EPI so they open in the correct AFNI view (tlrc/mni/orig). Both
have SCENE_DATA[1]=11 (fbuc) and TYPESTRING=3DIM_HEAD_FUNC set by
`set_afni_func_type()`.

---

## 9. Comparison with AFNI 3dREMLfit.c

| Feature | 3dREMLfit | ffs_reml |
|---------|-----------|----------|
| Grid | Grid 3: a∈[0,.9]×10, b∈[-.9,.9]×19 | Same default; configurable |
| Grid search | Hierarchical descent per voxel (sequential) | GPU: exhaustive batch; CPU: hierarchical |
| Likelihood | log\|R\| + log\|X'R⁻¹X\| + (N-p)log(RSS_w) | Identical |
| Whitening | Cholesky solve (DPOTRS-style) | L_inv GEMM (equivalent, avoids DLASWP) |
| λ constraint | λ ≥ 0 by default | Same |
| logdet computation | QR diagonal | QR diagonal (was X'X+slogdet, now always QR) |
| RSS computation | Forms residuals | Pythagorean: ‖Y_w‖²−‖Q'Y_w‖² |
| Float precision | float64 throughout | Configurable; float32 default, float64 via `-use_double` |
| GLS solve | Sequential per voxel | Batched by (a,b) group |
| σ̂² accumulation | float64 | float64 (explicit cast before `.pow(2).sum()`) |

### Potential Sources of Divergence from AFNI

1. **Float32 default:** `ffs_reml` uses float32 by default. The logdet
   computations and RSS can differ from AFNI's float64. Use `-use_double`
   for closest match. The `+1e-10` guards in logdet are float32-scale.

2. **Grid search path (CPU):** The hierarchical descent uses window narrowing
   that may miss the true global minimum on non-smooth likelihood surfaces.
   The GPU exhaustive search avoids this, but if running on CPU you may get
   a slightly different (a,b) per voxel than AFNI even with the same grid.

3. **λ ≥ 0 filtering:** Both apply this, but floating-point equality on the
   boundary may differ slightly. Check if any voxels have λ values near 0.

4. **The `+1e-10` logdet guard:** If L has near-zero diagonal elements (near-
   singular R), our logdet estimate is biased slightly. AFNI uses different
   regularization. This should not matter for valid (a,b) pairs.

5. **Covariance matrix construction:** We build the Toeplitz matrix explicitly
   as a dense matrix. AFNI uses the same structure but may handle edge cases
   (short runs, boundary effects) slightly differently.

6. **Design matrix handling:** AFNI normalizes the design matrix before the
   REML search. Check whether your design matrix is scaled comparably.

7. **Polort handling:** AFNI generates polynomial regressors directly as
   part of its design. `ffs_reml` reads an external `.xmat.1D` file. Verify
   that the polynomial basis in the file matches what AFNI would generate.

---

## 10. Likelihood Surface Output (-Rlklhd)

**Motivation:** The grid search evaluates the full REML likelihood surface
L(a, b) for every voxel. Inspecting it per voxel can reveal:
- Whether the optimum is well-defined or flat (parameter unidentifiability)
- Why ffs and AFNI disagree (different grid point wins?)
- The shape of the joint (a,b) surface — neither axis alone tells you this

**Output format:**

`-Rlklhd <prefix>`: 4D NIfTI with **one sub-brik per valid (a,b) grid point**
(~117–130 sub-briks). Sub-brik k = `L(a_k, b_k)` per voxel.

Sub-briks are labeled `a=0.00_b=0.30`, `a=0.10_b=-0.20`, etc. via 3drefit,
so you can identify which (a,b) pair each sub-brik corresponds to in AFNI.

The argmin across all sub-briks identifies the selected (a*, b*) for each
voxel — this is exactly the joint argmin, not a marginal.

**Memory:** `n_voxels × n_pairs × 4 bytes` ≈ `167k × 117 × 4 ≈ 78 MB`.
Accumulated in CPU RAM during the grid search, then written once.

---

## 11. Key Functions (Call Graph)

```
ffs_reml (cli/reml.py)
└── fit_glm_arma11() (glm/arma.py)
    ├── precompute_reml_grid()        ← Phase 1-6: L_inv, X_w, Q, logdets
    ├── batch_reml_grid_search()      ← GPU exhaustive or CPU hierarchical
    │   └── _cpu_hierarchical_reml_search() → _evaluate_single_param()
    ├── search_voxels_precomputed_grid()   ← alternative: sequential per grid point
    └── [GLS fitting loop, grouped by (a*,b*)]
        ├── prewhiten: L_inv @ Y (GEMM) or solve_triangular(L, Y)
        ├── QR solve → betas
        ├── float64 RSS → sigma2
        └── t-stats, R², partial R²

write_glm_bucket_as_nifti()   (glm/outputs.py)    → Rbuck
[Rvar writing block]          (cli/reml.py ~1750)  → Rvar
```

---

## 12. Commit History — REML-Related Changes (Bisect Guide)

Ordered newest → oldest. The question is: **at which commit did ffs_reml
results start diverging from AFNI 3dREMLfit?** Step back through these
commits to isolate the change that broke numerical equivalence.

To test a specific commit: `git stash && git checkout <hash>` then rerun
`ffs_reml` and compare to the AFNI reference. Key metric: per-voxel (a,b)
maps and t-stat distributions.

---

### `8d5f155` — `perf(reml): L_inv+Q precomputation, Pythagorean RSS, float32 accuracy`

**What changed in the REML path:**

- **L_inv replaces L in the cache.** Previously `ddfae62` stored L and used
  `solve_triangular(L, Y)` per voxel. Now L_inv is precomputed once
  (`solve_triangular(L, eye)`) and whitening is a plain GEMM `L_inv @ Y`.
  Mathematically identical (L_inv @ Y = solve(L, Y)), but eliminates the
  per-voxel triangular solve and the MKL DLASWP error.

- **QR is now always used** (was gated behind `use_qr` flag before). The
  `use_qr=False` path (X'X → `slogdet`) is completely removed. Both logdets
  now come from the QR R-diagonal: `2 * sum(log(|diag(R)| + 1e-10))`.

- **Pythagorean RSS** replaces the old beta→residual→RSS pipeline during grid
  search. `RSS = ‖Y_w‖² − ‖Q'Y_w‖²`. Betas are no longer computed at all
  during the grid search phase.

- **Float32 y-normalization**: data is divided by `data.std(dim=1)` before
  the grid search to prevent float32 squaring overflow. Scale-invariant
  outputs (t-stats, R²) are unaffected; betas/sigma2 are unscaled afterward.

- **Float64 RSS accumulation** in GLS fitting: `resid.to(float64).pow(2).sum()`
  instead of float32, reducing catastrophic cancellation in sigma2.

**Risk for divergence from AFNI:**
The `+1e-10` guard in `logdet = 2*sum(log(|diag(R)| + 1e-10))` biases the
logdet for any near-zero R-diagonal elements. The old QR path had the same
guard (`abs(diag_R) + 1e-10`), so this is likely not new. The Pythagorean
RSS is mathematically exact (Pythagoras theorem), not an approximation.
Low risk for this commit.

---

### `ddfae62` — `Add benchmark framework, REML optimization, and multi-arch timing plots`

**What changed in the REML path:**

- **L_inv → L (solve_triangular).** This was the commit that *removed*
  `L_inv` and switched to storing L + per-voxel `solve_triangular(L, Y)`.
  Previously (before this commit) whitening used `L_inv @ Y` (GEMM).
  (`8d5f155` later reverted back to L_inv.)

- **`use_qr` flag introduced** with two paths: QR path and X'X+slogdet path.
  The `slogdet` path was the source of the MKL DLASWP error later seen.

- **Phase 6 vectorization:** the `best_params` update loop in
  `batch_reml_grid_search` was vectorized (removed Python for-loop over
  voxels, used tensor indexing).

- **`_evaluate_single_param`** switched from `L_inv @ Y` to
  `solve_triangular(L, Y)`.

**Risk for divergence from AFNI:**
The switch from GEMM to triangular solve is numerically equivalent. The
introduction of the `slogdet` path (X'X) *could* have affected results if
the default was `use_qr=False`, but the default was `use_qr=False` for
the slogdet path which calls `slogdet(X'X)` — this forms X'X first (squaring
the condition number) and then computes logdet, which is less accurate than
QR for ill-conditioned designs. **This is a plausible divergence point** if
`use_qr` was False by default.

---

### `930d4f7` — test coverage + bug fix: `solve_triangular requires 2D input`

**What changed in the REML path:**

- **Bug fix in `reml_grid_search` (single-voxel path):** `solve_triangular`
  was passed a 1D Y tensor, causing it to silently fail (all likelihoods
  identical). Fixed with `Y_2d = Y.unsqueeze(1) if Y.ndim == 1 else Y`.

**Risk for divergence from AFNI:**
If the single-voxel path (`reml_grid_search`) was being called somewhere,
this bug would cause all voxels to get identical ARMA params. But the main
batch path (`batch_reml_grid_search`) was unaffected. Low risk for the
primary ffs_reml pipeline.

---

### `72052cf` — `Update CLI, GLM, and analysis workflows`

**What changed in the REML path:**

- Removed hand-rolled AFNI extension from Rvar output; switched to using
  `save_nifti()` helper from `io/afni.py`.
- Added `@torch.inference_mode()` decorator to key REML functions.
- Minor cleanups; no changes to the likelihood computation or grid search.

**Risk for divergence from AFNI:** None for numerical results.

---

### `a3e9e6f` — `chore(cli_cleanup): consolidate CLI entry points, misc fixes`

**What changed in the REML path:**

- `cli/reml.py`: minor argument help text changes.
- `glm/outputs.py`: minor formatting fix.
- `io/afni.py`: handle missing volume metadata gracefully.
- No changes to `glm/arma.py`.

**Risk for divergence from AFNI:** None.

---

### `21351e3` — `fix(io): inherit AFNI space/type metadata in GLM outputs`

**What changed in the REML path:**

- Rvar now deep-copies `results.nifti_header` so SCENE_DATA[0] (view=tlrc)
  and TEMPLATE_SPACE carry forward from the source EPI.
- `set_afni_func_type()` added to `io/afni.py`; called on both Rbuck and
  Rvar to set SCENE_DATA[1]=11 (fbuc) and TYPESTRING=3DIM_HEAD_FUNC.

**Risk for divergence from AFNI:** Zero — purely metadata, no change to
the numerical results.

---

### Summary: Where to Bisect

The most likely suspects for numerical divergence, in order of suspicion:

1. **`ddfae62`** — introduced the `use_qr` flag and X'X+slogdet path. If
   the default was `use_qr=False` when your reference comparison was run,
   the slogdet path (via X'X) gives slightly different logdet than the QR
   path due to squaring the condition number. Also: this commit vectorized
   the `batch_reml_grid_search` update loop — verify the argmin indexing
   is still correct.

2. **Anything before `ddfae62`** — the pre-benchmark codebase. If results
   matched AFNI before the benchmark refactor, that commit is the break point.

3. **`8d5f155`** — Pythagorean RSS + float32 normalization. Very low risk
   (math is exact), but worth testing since it rewrote the core inner loop.

**Suggested bisect procedure:**
```bash
# Test AFNI reference (already have this output)
# Test each checkpoint:
git checkout ddfae62^  # just BEFORE the benchmark/REML refactor
pip install -e . -q
ffs_reml [your command] -Rvar test_ddfae62_pre.nii.gz
# compare a/b maps with AFNI reference

git checkout ddfae62
pip install -e . -q  
ffs_reml [your command] -Rvar test_ddfae62_post.nii.gz
# if these differ → ddfae62 is the commit that broke it

git checkout 8d5f155
pip install -e . -q
ffs_reml [your command] -Rvar test_8d5f155.nii.gz
# compare to AFNI
```
