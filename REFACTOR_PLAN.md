# Refactoring Plan: Next Steps

## Scope

The 4 CLI tools (`3dREMLfast.py`, `3dDenoisefast.py`, `3dHRFoptfast.py`, `3dRidgefast.py`) plus their shared library code in `fastfuncsim/`. This plan covers remaining work after Phases A (P0 bugs), B (deduplication), and C (GPU optimization) are substantially complete.

**Current test status**: 6 failures, 317 passed, 1 skipped (down from 34 failures + 15 errors).

**Lint policy**: Ruff/pyright warnings are informational only. Do NOT spend time chasing lint issues unless they reveal actual bugs. The N806/N803 suppressions for linear algebra naming are already configured.

---

## COMPLETED WORK (Do Not Repeat)

### Phase A: Critical Bug Fixes (P0) — ALL 7/7 DONE
- A1: `arma_glm.py` missing `profile_section` import → created context manager in `timing_utils.py`
- A2: `xval.py` return dict API mismatch → added backward-compat keys
- A3: `xval.py` device mismatch on CPU-only path → all masks moved to CPU
- A4: `3dRidgefast.py` indentation error → lines 655-714 inside `else` block
- A5: `hrf_selection.py` wrong polort formula → uses `auto_polort()` now
- A6: `3dREMLfast.py` `load_nifti(f)[0].shape` → `load_nifti(f).shape`
- A7: `denoise.py` inline polort formula → uses `auto_polort()`

### Phase B: Deduplication & Harmonization — 90% DONE
- B1: `3dHRFoptfast.py` uses `load_and_preprocess_runs()` (~150 lines eliminated)
- B2: `3dRidgefast.py` save helpers consolidated with `cli_utils` (~60 lines eliminated)
- B4: Device argument parsing unified via `parse_device_arg()` in all CLIs
- B5: ARMA chunking integrated with `memory.py` (API consistency)
- B3: `3dREMLfast.py` — DEFERRED (see D3 below)

### Phase C: GPU & Performance Optimization — ALL 3/3 DONE
- C1: Vectorized autoscale in `3dRidgefast.py` and `ridge.py` (matmul + broadcasting)
- C2: QR-based nuisance projection in `xval.py project_out_nuisance_per_run()`
- C3: `torch.cuda.empty_cache()` reduced to strategic boundaries only

### Previously Completed Infrastructure
- `cli_utils.py`: `load_and_preprocess_runs()`, `parse_device_arg()`, `auto_polort()`, `build_nuisance_per_run()`, `save_volume_nifti()`, `save_4d_nifti()`, `LoadResult` dataclass
- `memory.py`: `estimate_chunk_size()` with per-operation models (glm, xval, ridge, denoise, arma)
- `timing_utils.py`: `profile_section()` context manager
- All ruff auto-fixes applied (F541, E722, B006, I001, UP035, F401)

---

## PHASE D: Remaining Bugs & Fixes

### D1: `denoise.py` Undefined Variables in Verbose Print — CRASH

**File**: `fastfuncsim/denoise.py` line 1581
**Problem**: When `verbose=True` and `polort=None`, the print statement references undefined variables `median_run_length` and `run_duration`:
```python
print(f"\nAuto-determined polort={polort} (median run: {median_run_length} TRs = {run_duration:.1f}s)")
```
Neither `median_run_length` nor `run_duration` are defined. The available variables are `run_lengths` (line 1576) and `avg_run_duration_sec` (line 1577).

**Fix**: Replace line 1581 with:
```python
median_run_length = int(torch.tensor(run_lengths).median().item()) if len(run_lengths) > 1 else run_lengths[0]
print(f"\nAuto-determined polort={polort} (median run: {median_run_length} TRs = {avg_run_duration_sec:.1f}s)")
```
Or simpler, just use the available variables directly in the f-string.

**Verify**: Call `fit_denoising_model()` with `verbose=True, polort=None`.

---

### D2: Fix 6 Remaining Test Failures

**Current failures** (from `pytest tests/ -q`):

#### D2a: Polynomial orthogonality tests (2 failures)
- `tests/test_glm_core_extended.py::TestPolynomialMatrix::test_construct_polynomial_degree_2`
- `tests/test_glm_core_extended.py::TestPolynomialMatrix::test_construct_polynomial_degree_3`

**Investigate**: Read the test expectations and compare against `construct_polynomial_matrix()` in `glm_core.py`. The Legendre polynomial implementation may have changed shape/normalization. Check if tests expect raw monomials vs Legendre, or if a normalization convention changed.

#### D2b: HRF library shape mismatch (1 failure)
- `test_fit_glm_hrf_library_logic` — `RuntimeError: mat1 and mat2 shapes cannot be multiplied (6x10 and 200x100)`

**Investigate**: The error is at `glm_core.py:214`. The design matrix dimensions don't match data dimensions. This likely means the test creates a design matrix with wrong dimensions for the data it provides, OR the function changed its expectation about input layout.

#### D2c: Renamed keyword argument (2 failures)
- `test_microtime_resolution` — `TypeError: onsets_to_binary_matrix() got an unexpected keyword argument 'microtime_resolution'`
- `test_microtime_vs_tr_locked` — same error

**Root cause**: `onsets_to_binary_matrix()` in `afni_io.py:272` was refactored from `microtime_resolution: int` (an integer multiplier of TR, e.g. `20` → dt = TR/20) to `microtime_dt: float = 0.1` (direct time step in seconds). The microtime functionality is fully intact — only the parameter name and semantics changed.

**Fix**: Update test calls in `tests/test_high_level_confirmation.py` to use the new parameter:
- `microtime_resolution=1` (TR-locked) → `microtime_dt=tr` (i.e., step size equals TR)
- `microtime_resolution=N` → `microtime_dt=tr/N` (e.g., `microtime_resolution=20, tr=1.0` → `microtime_dt=0.05`)

#### D2d: Cross-validation metric tolerance (1 failure)
- `tests/test_xval_real_data.py::test_xval_r2_different_metrics` — Pearson r² values slightly below -0.05 tolerance

**Fix**: Either widen the tolerance in the assertion (e.g., `-0.1` instead of `-0.05`) since Pearson correlation R² on noise can go slightly negative, or investigate if the test data is degenerate.

**Verify**: `pytest tests/ -q` → target 0 failures.

---

### D3: `3dREMLfast.py` Nuisance & Loading Deduplication (was B3)

**File**: `bin/3dREMLfast.py`
**Problem**: Three areas of inline duplication:

1. **Lines 82-119**: Local `parse_input_files()` duplicates `cli_utils.parse_input_files()`. The import already exists (line 42) but the local copy is called instead.

2. **Lines 888-909 and 990-996**: Nuisance building duplicated at two locations. Both manually compute run lengths and construct polynomial blocks with `construct_polynomial_matrix()` + `torch.block_diag()`.

3. **Lines 1358-1388**: Manual per-run data loading loop instead of `load_and_preprocess_runs()`.

**Architectural blockers** (documented for context, not necessarily blocking):
- `-matrix` mode uses precomputed design from `read_afni_design_matrix()`, different from onset-based flow
- Uses 4D numpy arrays for `analyze_from_design_matrix` compatibility
- `preprocessing_applied` flag determines data flow

**Recommended approach** (incremental, low risk):
1. Delete local `parse_input_files()`, use the imported `cli_utils` version
2. Replace both nuisance-building blocks with `build_nuisance_per_run()`. The function returns a list of per-run blocks; if `3dREMLfast` needs `torch.block_diag(*blocks)` it can call that on the result
3. Leave data loading as-is for now (the `-matrix` mode has genuinely different requirements)

**Verify**: Run any available REML tests, check that onset-path and matrix-path both work.

---

## PHASE E: Deduplication & Numerical Harmonization

### E1: Unify Projection Functions in `xval.py`

**File**: `fastfuncsim/xval.py`
**Problem**: Three projection implementations coexist with different algorithms:

| Function | Lines | Algorithm | Used by |
|----------|-------|-----------|---------|
| `project_out_nuisance_per_run()` | 19-214 | QR decomposition (stable, memory-efficient) | `3dRidgefast.py`, `denoise.py` (initial R²) |
| `project_out_nuisance()` | 369-457 | Explicit `(X'X)^{-1}` projection matrix | Various internal callers |
| `_compute_projection_matrix()` | 536-571 | Same as above (helper) | `compute_xval_r2()` |

The per-run QR version is numerically superior. The other two use explicit `(X'X)^{-1}` with ridge regularization, which is less stable and creates a full `(n, n)` matrix.

**Fix**:
1. Upgrade `project_out_nuisance()` to use QR internally (same API, better numerics):
   ```python
   Q, _ = torch.linalg.qr(X_nuis)
   data_cleaned = data - (Q @ (Q.T @ data.T)).T
   design_cleaned = design_matrix - Q @ (Q.T @ design_matrix)
   ```
2. Upgrade `_compute_projection_matrix()` to return a `Q` factor instead of a full `P` matrix, or refactor callers to use the per-run function directly.
3. Alternatively, if `_compute_projection_matrix()` is only used inside `compute_xval_r2()`, inline it or replace with a QR-based helper.

**Caution**: `project_out_nuisance()` takes `nuisance_indices` (column indices into the full design matrix), while `project_out_nuisance_per_run()` takes pre-extracted per-run nuisance matrices. They serve different call sites. Unify the algorithm, but the two API shapes may both be needed.

**Verify**: Run `pytest tests/test_xval.py` and compare numerical outputs.

---

### E2: Unify `denoise.py` Inline Projection with `xval.py`

**File**: `fastfuncsim/denoise.py`
**Problem**: The `cross_validate_noise_pcs()` function (lines 787-896) and `compute_xval_r2_optimal_full()` (lines 1230-1316) both implement their own per-run nuisance projection inline, using the less stable `(X'X)^{-1}` approach. Meanwhile, `fit_denoising_model()` (lines 1643, 1712) correctly calls `project_out_nuisance_per_run()` from `xval.py`.

This means the SAME file uses two different projection algorithms for the same mathematical operation — the inline version is less stable and harder to maintain.

**Current inline pattern** (appears twice, ~100 lines each):
```python
for run_idx in train_runs:
    XtX = run_nuisance.T @ run_nuisance
    XtX_inv = torch.linalg.inv(XtX + 1e-6 * torch.eye(...))
    P_nuisance = run_nuisance @ XtX_inv @ run_nuisance.T
    projection = torch.eye(run_length, ...) - P_nuisance
    run_data_proj = (projection @ run_data.T).T
```

**Fix**: Replace both inline projection loops with calls to `project_out_nuisance_per_run()` from `xval.py`. The function already handles:
- Per-run QR decomposition (more stable)
- Zero-column detection
- Memory-aware chunking
- Device management

**Specifics**:
- In `cross_validate_noise_pcs()` (lines 787-896): Extract train/test data+design slices, call `project_out_nuisance_per_run()` for each split. Need to construct local `run_starts` for the train/test subsets.
- In `compute_xval_r2_optimal_full()` (lines 1230-1316): Same pattern.
- The function expects `(data, design, nuisance_per_run, run_starts)` → returns `(data_clean, design_clean)`.

**Complication**: The denoise CV loop processes chunks of voxels and projects per-chunk. `project_out_nuisance_per_run()` handles this internally with its own chunking. You'll need to either:
- Call it on the full data and let it chunk internally, OR
- Pass the chunk through and let it project the subset

The first approach is cleaner. If memory is a concern, `project_out_nuisance_per_run()` already has a `chunk_size` parameter.

**Impact**: ~200 lines of duplicated projection code removed, numerical consistency across the codebase.

**Also fix** (while in this area):
- Lines 841, 896: Remove `torch.cuda.empty_cache()` calls inside the CV loop. `project_out_nuisance_per_run()` handles cache management internally.

**Verify**: `pytest tests/` and compare R² outputs to ensure numerical equivalence.

---

### E3: `denoise.py` Hardcoded Chunk Sizes → `memory.py`

**File**: `fastfuncsim/denoise.py` lines 660-691
**Problem**: Three hardcoded chunk-size policies with magic numbers:

```python
# LORO path (line 667):
target_chunk_memory_gb = 0.3  # magic
voxel_chunk_size = min(n_voxels, max(max_voxels_from_memory, 10000), 42000)  # magic bounds

# Split-half GPU path (line 681):
target_chunk_memory_gb = 0.8  # magic
voxel_chunk_size = min(n_voxels, max(max_voxels_from_memory, 10000), 50000)  # magic bounds

# CPU path (line 688):
target_chunk_memory_gb = 0.5  # magic
voxel_chunk_size = min(n_voxels, max(max_voxels_from_memory, 5000), 20000)  # magic bounds
```

**Fix**: Replace with `estimate_chunk_size(operation="denoise", ...)` from `memory.py`. The memory module already has `bytes_per_voxel_denoise()` (line 174). If the LORO vs split-half distinction matters, add a `cv_mode` parameter to `bytes_per_voxel_denoise()`.

**Verify**: Run denoising on representative data, check memory usage doesn't regress.

---

## PHASE F: Performance Optimizations

### F1: Vectorize CV Prediction Scattering in `ridge.py`

**File**: `fastfuncsim/ridge.py` lines 597-605
**Problem**: Double nested Python loop scattering predictions to output accumulators:
```python
for frac_idx in range(n_fracs):
    y_pred_frac = y_pred_all[:, frac_idx, :].T
    for i, tp in enumerate(test_tps):
        predictions_by_frac[frac_idx][:, tp] = y_pred_frac[:, i]

for i, tp in enumerate(test_tps):
    actual_test_clean[:, tp] = test_data_clean[:, i]
```

**Fix**: Use fancy indexing to vectorize:
```python
test_tps_t = torch.tensor(test_tps, device=device)
for frac_idx in range(n_fracs):
    predictions_by_frac[frac_idx][:, test_tps_t] = y_pred_all[:, frac_idx, :].T

actual_test_clean[:, test_tps_t] = test_data_clean
```
The outer loop over `n_fracs` (typically 10-20) is fine. The inner loop over `test_tps` (hundreds) should be vectorized.

---

### F2: Vectorize Result Scattering in `ridge.py`

**File**: `fastfuncsim/ridge.py` lines 782-788
**Problem**: Per-voxel Python loop scattering group results back to global arrays:
```python
for i, vox_idx in enumerate(voxel_indices):
    betas_all[vox_idx, :] = group_results["betas"][i, :]
    r2_initial_all[vox_idx] = group_results["r2_initial"][i]
    # ... 4 more assignments
```

**Fix**: Use fancy indexing:
```python
idx = torch.tensor(voxel_indices, device=betas_all.device)
betas_all[idx] = group_results["betas"]
r2_initial_all[idx] = group_results["r2_initial"]
r2_final_all[idx] = group_results["r2_final"]
xval_r2_all[idx] = group_results["xval_r2"]
optimal_fracs_all[idx] = group_results["optimal_fracs"]
r2_by_frac_all[idx] = group_results["r2_by_frac"]
```

---

### F3: Vectorize Design Hashing in `ridge.py`

**File**: `fastfuncsim/ridge.py` lines 741-745
**Problem**: Per-voxel loop computing design hashes with individual `.sum().item()` calls (each triggers GPU→CPU sync):
```python
for vox_idx in range(chunk_voxels):
    design_hash = float(design_per_voxel[vox_idx].sum().item())
```

**Fix**: If designs are stacked as a 3D tensor:
```python
design_hashes = design_per_voxel.sum(dim=(1, 2)).tolist()  # single GPU→CPU transfer
```
If they're a list, stack first or compute hashes on GPU before transferring.

---

## PHASE G: Architecture & Extensibility

These are higher-level improvements to make the codebase more modular, so that new analysis methods (new denoising strategies, new GLM variants, new CV metrics) can be added without duplicating infrastructure.

### G1: Make `xval.py` a General-Purpose CV Engine

**File**: `fastfuncsim/xval.py`
**Current state**: The module is tightly coupled to specific GLM/ridge workflows. It handles nuisance projection, prediction, R² computation, and split management all in monolithic functions (`compute_xval_r2` is ~600 lines).

**Goal**: `xval.py` should be the single entry point for ANY cross-validation in the codebase — ridge, ARMA, denoising, HRF selection, future methods. Different analysis methods plug in their own fitting/prediction logic; `xval.py` owns the CV loop, split management, nuisance projection, and metric computation.

**Concrete steps**:

1. **Extract a `CVSplitter` or split-management utility**. Currently every module (`denoise.py`, `ridge.py`, `hrf_selection.py`) computes its own train/test run indices and local `run_starts`. Factor out:
   ```python
   def make_cv_splits(run_starts, n_timepoints, strategy="loro"):
       """Generate train/test splits as lists of run indices."""
       ...

   def extract_split_data(data, design, run_starts, train_runs, test_runs):
       """Slice data/design by run membership, return with local run_starts."""
       ...
   ```
   This is the most impactful step — every analysis path reimplements run slicing.

2. **Define a `FitPredictFn` protocol** (or just a callable convention) for pluggable model fitting:
   ```python
   # Convention: fit_fn(train_data, train_design) -> betas
   #             predict_fn(betas, test_design) -> predictions
   ```
   Then `compute_xval_r2` becomes:
   ```python
   def compute_xval_r2(data, design, nuisance_per_run, run_starts, cv_splits,
                        fit_fn=ols_fit, predict_fn=linear_predict, metric="cod", ...):
   ```
   This lets `denoise.py` pass in its GLM+noise-PC fit function, `ridge.py` pass in fractional ridge, and future methods plug in without touching `xval.py`.

3. **Generalize metric computation**. Currently `_compute_r2()` handles `"cod"`, `"corr"`, `"corr2"`. Make metrics pluggable:
   ```python
   def compute_metric(actual, predicted, metric="cod"):
       """Compute per-voxel metric. Extensible to new metrics."""
   ```
   Move this to a standalone function (it already mostly is at lines 460-533). Ensure `denoise.py` and `ridge.py` use the same function instead of inline R² computation.

4. **Unify nuisance projection as a preprocessing step** rather than interleaved with CV logic. The current `project_out_nuisance_per_run()` is already a good building block. Make it the single path for all callers (see E2 above).

**Impact**: New analysis methods (e.g., a Bayesian GLM, a different denoising approach, MVPA cross-validation) can reuse the entire CV infrastructure by providing a fit/predict function and a metric.

---

### G2: Make `denoise.py` Extensible for New Denoising Methods

**File**: `fastfuncsim/denoise.py`
**Current state**: The denoising module is built around the GLMdenoise approach (noise pool → PCA → augmented GLM). The CV loop for selecting optimal PC count is tightly integrated.

**Goal**: New denoising strategies (e.g., tCompCor, aCompCor, ICA-based, wavelet-based) should be addable without rewriting the CV infrastructure.

**Concrete steps**:

1. **Extract the "noise regressor generator" as a pluggable component**. Currently `extract_noise_pcs_per_run()` (line 233) generates noise PCs. Generalize to a callable:
   ```python
   # Protocol: noise_gen_fn(data, design, run_starts, ...) -> List[torch.Tensor]
   # Each tensor is (run_length, n_components) for one run
   ```
   Then `cross_validate_noise_pcs()` becomes `cross_validate_noise_components()` and accepts any generator.

2. **Separate "component selection" from "component generation"**. The CV loop that tests 0..max_components should work with any noise regressors, not just PCs from a noise pool. This is mostly a naming/API change — the math is already general.

3. **Use `xval.py` for the inner CV** (see G1). Currently `cross_validate_noise_pcs()` reimplements the entire CV loop (800+ lines). If `xval.py` provides a general CV engine with pluggable fit/predict, `denoise.py` just provides the fit function (OLS + noise PCs of various counts) and calls `compute_xval_r2()`.

**Impact**: Adding a new denoising method becomes: (1) write a noise-component generator function, (2) plug it into the existing CV infrastructure.

---

### G3: Harmonize R² Computation Across the Codebase

**Problem**: R² / cross-validation metrics are computed in at least 5 different places:

| Location | Lines | Method |
|----------|-------|--------|
| `xval.py _compute_r2()` | 460-533 | CoD, Pearson corr, corr² (general, handles metrics) |
| `xval.py compute_xval_r2()` | 1140-1155 | Streaming SS_res/SS_tot accumulation |
| `ridge.py _fit_ridge_chunk()` | 610-614 | Inline `1 - ss_res / (ss_tot + 1e-10)` |
| `ridge.py` (final R²) | 691-697 | Same inline formula |
| `denoise.py cross_validate_noise_pcs()` | 1012-1026 | Inline R² with streaming accumulators |
| `denoise.py compute_xval_r2_optimal_full()` | ~1400 | Similar inline R² |

All compute the same thing with slight variations (epsilon values, streaming vs batch, metric choice).

**Fix**: Promote `_compute_r2()` from `xval.py` to a public function (rename to `compute_r2()` or `compute_metric()`). Have all callers use it. For streaming accumulation contexts where you can't pass full arrays, provide a companion:
```python
def compute_r2_from_stats(ss_res, ss_tot, metric="cod"):
    """Compute R² from pre-accumulated sufficient statistics."""
```

**Impact**: Single source of truth for R² computation, consistent epsilon handling, easier to add new metrics.

---

### G4: Clean Up `xval.py` Return Dict API

**File**: `fastfuncsim/xval.py` lines 1192-1202
**Problem**: The return dict has misleading backward-compat keys:
```python
"r2_median": r2_final,  # Actually per-voxel tensor, NOT a median
```
This was kept for `ffs_pathfinder` compatibility.

**Fix**:
1. Update `ffs_pathfinder.py` to use the `"r2"` key (the correct one)
2. Remove `"r2_median"` alias (or make it actually compute `r2_final.median()`)
3. Same for `compute_xval_r2_single_trials()` at line 1332

**Impact**: Eliminates confusing API surface. Low risk since the fix is just updating the one downstream caller.

---

## PHASE H: Challenging Structural Fixes

### H1: `3dREMLfast.py` Matrix Mode vs Onset Mode Unification

**File**: `bin/3dREMLfast.py`
**Problem**: The CLI has two fundamentally different data flows:
- `-matrix` mode: Reads precomputed AFNI design matrix, optionally preprocesses data
- `-onset_files` mode: Builds design from scratch (onsets → convolution → design matrix)

These paths share nuisance building, data loading, and output logic but are currently separate code paths with ~200 lines of duplication.

**Approach**: Don't try to fully unify. Instead:
1. Apply `build_nuisance_per_run()` to both paths (see D3)
2. Create a `prepare_reml_data()` function that both paths call after their respective setup, handling the common data validation / device placement / chunking
3. Keep the design-matrix construction separate (it's genuinely different between modes)

**This is the most architecturally complex remaining task.** Handle incrementally.

---

### H2: Per-Voxel Design Matrix Support in Ridge

**File**: `fastfuncsim/ridge.py`
**Problem**: `_fit_ridge_chunk_with_per_voxel_designs()` (lines 715-797) groups voxels by design hash, then calls `_fit_ridge_chunk()` for each group. The hash is a simple `.sum()` (line 744), which is collision-prone.

**Issues**:
1. Hash collisions: two different designs could have the same sum
2. Per-group loop with result scattering (see F2)
3. The grouping overhead may outweigh savings for small groups

**Fix** (if per-voxel designs are rare):
- Improve hash to use multiple statistics: `(sum, sum_of_squares, max, min)` tuple
- Or compute a proper content hash via `torch.hash` or checksumming
- Vectorize scattering (F2)

**Fix** (if per-voxel designs are common):
- Consider a batched SVD approach where all designs are stacked and solved simultaneously
- This is a bigger architectural change and should be profiled first

---

## Execution Priority

| Task | Severity | Impact | Priority |
|------|----------|--------|----------|
| D1: denoise.py undefined vars | **CRASH** | Verbose output broken | **P0** |
| D2: Fix 6 test failures | **TESTS** | CI/trust | **P0** |
| D3: 3dREMLfast parse/nuisance dedup | Low | ~50 lines + consistency | **P1** |
| E1: Unify xval.py projection functions | Medium | Numerical consistency | **P1** |
| E2: denoise.py → xval.py projection | **HIGH** | ~200 lines dedup + stability | **P1** |
| E3: denoise.py chunk sizes → memory.py | Low | Consistency | **P2** |
| F1: ridge.py CV scatter vectorization | Low | Performance | **P2** |
| F2: ridge.py result scatter vectorization | Low | Performance | **P2** |
| F3: ridge.py design hash vectorization | Low | Performance (GPU sync) | **P2** |
| G1: xval.py general CV engine | **ARCH** | Extensibility | **P2** |
| G2: denoise.py extensible denoising | **ARCH** | Extensibility | **P2** |
| G3: Harmonize R² computation | Medium | Consistency + extensibility | **P2** |
| G4: Clean up xval.py return dict | Low | API clarity | **P3** |
| H1: 3dREMLfast mode unification | **ARCH** | Maintainability | **P3** |
| H2: Ridge per-voxel design hashing | Low | Correctness (edge case) | **P3** |

**Recommended execution order**:
1. D1 + D2 (fix crashes and tests first)
2. E1 + E2 (biggest remaining dedup win, improves numerical consistency)
3. D3 (quick dedup win in 3dREMLfast)
4. F1-F3 (vectorization, can be done independently)
5. G1-G3 (architectural — plan carefully before implementing)
6. G4, H1, H2 (low priority, do when convenient)

---

## Testing Strategy

After each fix:
1. Run `pytest tests/` — target: 0 failures, 0 errors
2. For numerical changes (E1, E2, G3), compare R² outputs before/after to verify equivalence
3. For CLI changes, verify the tool runs on test data if available
4. For GPU changes, verify memory doesn't regress on representative data

**Key invariant**: Output files should be bit-identical before and after deduplication/optimization phases. Only numerical harmonization (E1/E2 switching from `(X'X)^{-1}` to QR) may produce small floating-point differences.

---

## What NOT to Change

- The CLAUDE.md design principles (Legendre polynomials, CV patterns, etc.)
- Output file formats or naming conventions
- argparse argument names (these are user-facing APIs)
- Core algorithm logic in library modules (only change interfaces/projection method)
- Don't chase ruff/pyright warnings unless they reveal real bugs
- ARMA's specialized chunking in `arma_glm.py` (it has good reasons to differ from `memory.py`)
