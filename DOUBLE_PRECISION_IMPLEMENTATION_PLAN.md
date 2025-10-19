# Double Precision (Float64) Support - Implementation Plan

## Executive Summary

**Goal**: Add optional float64 precision to match AFNI's OLS/ARMA results exactly while maintaining current float32 speed as default.

**Performance Impact**: ~2x memory usage, ~1.3-1.5x slower on GPU (tensor core acceleration still works)

**Benefit**: Exact numerical agreement with AFNI (down to machine precision), eliminating all systematic differences.

---

## Architecture Overview

### Current State (Float32 Only)
```python
# Data conversion (hardcoded)
data = data.to(torch.float32)

# Memory calculation (hardcoded)
mem_per_voxel = n_timepoints * n_regressors * 4  # 4 bytes per float32

# Tensor creation (hardcoded)
corr = torch.zeros(n, device=device, dtype=torch.float32)
```

### Proposed State (Configurable Precision)
```python
# Dtype determined by parameter
dtype = torch.float64 if use_double else torch.float32

# Data conversion (configurable)
data = data.to(dtype)

# Memory calculation (precision-aware)
bytes_per_element = 8 if use_double else 4
mem_per_voxel = n_timepoints * n_regressors * bytes_per_element

# Tensor creation (dtype-aware)
corr = torch.zeros(n, device=device, dtype=dtype)
```

---

## Implementation Strategy

### Phase 1: Core Infrastructure (Low Risk)
**Goal**: Thread dtype through the codebase without changing default behavior

#### 1.1 Add dtype Parameter to All GLM Functions
```python
# glm_core.py
def fit_glm(
    data, design, tr,
    use_double: bool = False,  # NEW PARAMETER
    ...
) -> GLMResults:
    dtype = torch.float64 if use_double else torch.float32
    # ... use dtype everywhere

# arma_glm.py  
def fit_glm_arma11(
    data, design, tr,
    use_double: bool = False,  # NEW PARAMETER
    ...
) -> ARMA11Results:
    dtype = torch.float64 if use_double else torch.float32
    # ... use dtype everywhere
```

**Files to modify**:
- `fastfuncsim/glm_core.py` - OLS fitting
- `fastfuncsim/arma_glm.py` - ARMA fitting
- `fastfuncsim/analysis.py` - High-level API

**Risk**: LOW - Adding parameter with default value is backward compatible

---

#### 1.2 Update Batch Size Calculation
**Critical**: Memory doubles with float64, so batch sizes must halve

```python
def get_adaptive_batch_size(
    device: torch.device,
    n_timepoints: int,
    n_regressors: int,
    use_double: bool = False  # NEW
) -> int:
    # Precision-aware memory calculation
    bytes_per_element = 8 if use_double else 4
    
    mem_per_voxel = (
        n_timepoints * n_regressors * bytes_per_element  # X_w_batch
        + n_timepoints * bytes_per_element               # y_w_batch
        + n_regressors * n_regressors * bytes_per_element  # XtX
        + n_regressors * bytes_per_element               # betas
    )
    
    # Rest of logic unchanged
    ...
```

**Files to modify**:
- `fastfuncsim/arma_glm.py::get_adaptive_batch_size()`

**Risk**: LOW - Simple multiplication factor

---

### Phase 2: Data Type Conversion (Medium Risk)

#### 2.1 Replace All Hardcoded `torch.float32`

**Search pattern**: `torch.float32` → Replace with `dtype` variable

**Locations** (11 total):
```python
# glm_core.py (2 occurrences)
Line 301: d_tensor = d_tensor.to(torch.float32)
          → d_tensor = d_tensor.to(dtype)

# arma_glm.py (9 occurrences)  
Line 394:  corr = torch.zeros(n, device=device, dtype=torch.float32)
           → corr = torch.zeros(n, device=device, dtype=dtype)

Line 401:  powers = torch.full((n - 2,), a, device=device, dtype=torch.float32)
           → powers = torch.full((n - 2,), a, device=device, dtype=dtype)

Line 516:  corr = torch.zeros(n_valid, n, device=device, dtype=torch.float32)
           → corr = torch.zeros(n_valid, n, device=device, dtype=dtype)

Line 527:  torch.linspace(1, n - 1, device=device, dtype=torch.float32)
           → torch.linspace(1, n - 1, device=device, dtype=dtype)

Line 1362: data = to_tensor(data, device=None, dtype=torch.float32)
           → data = to_tensor(data, device=None, dtype=dtype)
```

**Risk**: MEDIUM - Requires careful review to ensure dtype is in scope

---

#### 2.2 Update Helper Functions

```python
# arma_glm.py - Add dtype parameter
def arma_autocorrelation_vectorized(
    a: torch.Tensor, 
    b: torch.Tensor, 
    n: int, 
    device: torch.device,
    dtype: torch.dtype = torch.float32  # NEW
) -> torch.Tensor:
    corr = torch.zeros(n_valid, n, device=device, dtype=dtype)  # Use dtype
    ...

def arma_autocorrelation(
    a: float, 
    b: float, 
    n: int, 
    device: torch.device,
    dtype: torch.dtype = torch.float32  # NEW
) -> torch.Tensor:
    corr = torch.zeros(n, device=device, dtype=dtype)  # Use dtype
    ...
```

**Files to modify**:
- `fastfuncsim/arma_glm.py::arma_autocorrelation()`
- `fastfuncsim/arma_glm.py::arma_autocorrelation_vectorized()`

**Risk**: LOW - Localized changes

---

### Phase 3: API Integration (Low Risk)

#### 3.1 Add to User-Facing API

```python
# analysis.py
def analyze_from_design_matrix(
    fmri_data,
    design_matrix_file,
    method: str = "ols",
    use_double: bool = False,  # NEW - Match AFNI precision exactly
    ...
) -> Tuple[Union[GLMResults, ARMA11Results], Dict]:
    """
    ...
    use_double : bool, default=False
        If True, use float64 precision (matches AFNI exactly, ~2x memory, ~1.5x slower)
        If False, use float32 precision (faster, tiny differences from AFNI)
    ...
    """
    
    # Pass to underlying functions
    if method == "ols":
        results = fit_glm(data, design, tr=tr, use_double=use_double, ...)
    elif method == "arma11":
        results = fit_glm_arma11(data, design, tr=tr, use_double=use_double, ...)
```

**Files to modify**:
- `fastfuncsim/analysis.py::analyze_from_design_matrix()`

**Risk**: LOW - Simple parameter pass-through

---

### Phase 4: Testing & Validation (Critical)

#### 4.1 Unit Tests
```python
# tests/test_precision.py (NEW FILE)
def test_float32_vs_float64_agreement():
    """Verify float64 is more accurate than float32"""
    data = torch.randn(100, 50)
    design = torch.randn(50, 3)
    
    results_f32 = fit_glm(data, design, tr=2.0, use_double=False)
    results_f64 = fit_glm(data, design, tr=2.0, use_double=True)
    
    # Float64 should be numerically closer to "truth"
    # (Use higher precision reference or analytical solution)
    ...

def test_batch_size_scales_correctly():
    """Verify batch size halves for float64"""
    batch_f32 = get_adaptive_batch_size(device, 300, 48, use_double=False)
    batch_f64 = get_adaptive_batch_size(device, 300, 48, use_double=True)
    
    assert batch_f64 <= batch_f32 / 1.8  # Account for some headroom
    assert batch_f64 >= batch_f32 / 2.2
```

#### 4.2 AFNI Comparison Test
```python
def test_afni_exact_match():
    """Verify float64 matches AFNI OLS exactly"""
    # Load same data/design that AFNI used
    # Compare betas, t-stats, F-stats
    # Differences should be < 1e-10 (machine precision)
    ...
```

**New files**:
- `tests/test_precision.py`

**Risk**: MEDIUM - Need access to AFNI ground truth data

---

## Code Changes Summary

### Files to Modify (6 total)

1. **`fastfuncsim/glm_core.py`** (~50 lines)
   - Add `use_double` parameter to `fit_glm()`
   - Replace 2 hardcoded `torch.float32` → `dtype`
   - Thread dtype through helper functions

2. **`fastfuncsim/arma_glm.py`** (~80 lines)
   - Add `use_double` parameter to `fit_glm_arma11()`
   - Replace 9 hardcoded `torch.float32` → `dtype`
   - Update `get_adaptive_batch_size()` for 8-byte elements
   - Update autocorrelation functions with dtype

3. **`fastfuncsim/analysis.py`** (~15 lines)
   - Add `use_double` parameter to `analyze_from_design_matrix()`
   - Pass through to GLM functions
   - Update docstring

4. **`tests/test_precision.py`** (NEW, ~100 lines)
   - Unit tests for precision
   - Batch size scaling tests
   - AFNI comparison tests

5. **`tests/test_write_functions.py`** (~5 lines)
   - Add precision tests to existing suite

6. **`README.md` / `docs/`** (~20 lines)
   - Document `use_double` parameter
   - Performance implications
   - When to use it

**Total LOC**: ~270 lines (mostly parameter threading, low risk)

---

## Implementation Checklist

### Preparation
- [ ] Create feature branch: `feat/double-precision`
- [ ] Backup current test results (float32 baseline)
- [ ] Identify AFNI test dataset for validation

### Core Changes (1-2 hours)
- [ ] Add `use_double` parameter to `fit_glm()` signature
- [ ] Add `use_double` parameter to `fit_glm_arma11()` signature  
- [ ] Update `get_adaptive_batch_size()` memory calculation
- [ ] Replace all 11 hardcoded `torch.float32` with `dtype` variable
- [ ] Thread dtype through autocorrelation functions
- [ ] Add dtype parameter to `analyze_from_design_matrix()`

### Testing (2-3 hours)
- [ ] Run existing test suite (should pass with use_double=False)
- [ ] Add unit tests for float32 vs float64
- [ ] Add batch size scaling tests
- [ ] Test with small dataset (100 voxels, both precisions)
- [ ] Test with medium dataset (10k voxels, both precisions)
- [ ] Compare float64 results with AFNI OLS output

### Validation (1-2 hours)
- [ ] Profile performance (memory & speed) for both precisions
- [ ] Document actual slowdown factor (target: <1.5x)
- [ ] Verify GPU batch sizes scale correctly
- [ ] Test on CPU, CUDA, and MPS (if available)

### Documentation (1 hour)
- [ ] Update docstrings with `use_double` parameter
- [ ] Add usage examples to README
- [ ] Create performance comparison table
- [ ] Document when to use float64 vs float32

### Deployment
- [ ] Code review
- [ ] Merge to main branch
- [ ] Tag release (e.g., v1.1.0)
- [ ] Update user scripts with option

**Total estimated time**: 5-8 hours (1 working day)

---

## Performance Expectations

### Memory Usage
| Precision | Bytes/Element | 10k voxels, 300 TRs, 48 regs | Batch Size (16GB GPU) |
|-----------|---------------|------------------------------|----------------------|
| float32   | 4             | ~576 MB                      | ~5,000 voxels        |
| float64   | 8             | ~1.15 GB                     | ~2,500 voxels        |

**Impact**: Exactly 2x memory, batch size halves

### Computation Speed
| Operation | float32 | float64 | Slowdown |
|-----------|---------|---------|----------|
| Matrix Multiply (GPU) | 1.0x | 1.3-1.5x | Tensor cores still accelerate |
| Matrix Inversion | 1.0x | 1.4-1.6x | Double precision more stable |
| Memory Transfer | 1.0x | 2.0x | Twice the data |

**Overall**: Expect 1.3-1.5x slower end-to-end (GPU), 1.1-1.2x slower (CPU)

### Numerical Accuracy
| Metric | float32 | float64 |
|--------|---------|---------|
| AFNI Beta Difference | ~0.01% - 0.1% | < 1e-10 (machine precision) |
| AFNI T-stat Difference | ~0.01% - 0.1% | < 1e-10 (machine precision) |
| Matrix Condition Tolerance | ~1e-6 | ~1e-14 |

**Benefit**: Perfect agreement with AFNI, no systematic bias

---

## Risk Mitigation

### High Risk Items
1. **Dtype variable scope**: Ensure dtype is defined before use in all code paths
   - Mitigation: Comprehensive linting, unit tests
   
2. **Mixed precision bugs**: Accidentally mixing float32 and float64 tensors
   - Mitigation: Explicit `.to(dtype)` conversions, type checking

3. **GPU OOM with double**: Batch size calculation could be off
   - Mitigation: Conservative scaling (halve batch size), runtime memory checks

### Medium Risk Items
1. **Performance regression**: Slower than expected
   - Mitigation: Profile early, optimize bottlenecks

2. **Backward compatibility**: Breaking existing user code
   - Mitigation: Default to float32, gradual rollout

### Low Risk Items
1. **Documentation drift**: Docs not updated
   - Mitigation: Update docs in same PR

2. **Test coverage**: Missing edge cases
   - Mitigation: Comprehensive test plan above

---

## Success Criteria

✅ **Must Have** (Release Blockers):
1. All existing tests pass with `use_double=False` (backward compatible)
2. OLS results match AFNI within 1e-9 with `use_double=True`
3. ARMA results match AFNI within 1e-9 with `use_double=True`
4. GPU batch size automatically adjusts for float64
5. Performance slowdown < 2x on GPU, < 1.5x on CPU
6. No memory leaks or OOM crashes

✅ **Should Have**:
1. Documentation with usage examples
2. Performance comparison table
3. Unit tests for precision differences
4. AFNI validation test with real data

✅ **Nice to Have**:
1. Automatic precision selection based on matrix condition number
2. Mixed precision support (float32 data, float64 computation)
3. Benchmark suite for performance tracking

---

## Alternative Approaches Considered

### Option A: Always Use Float64 (Rejected)
- ❌ 2x slower for most users who don't need exact AFNI match
- ❌ 2x memory usage limits batch sizes
- ✅ Simpler code (no dtype parameter)

### Option B: Mixed Precision (Future Work)
- ✅ Float32 for data storage, float64 for critical operations (X'X inversion)
- ✅ Best of both worlds: speed + accuracy
- ❌ Much more complex to implement correctly
- 💡 Could be Phase 2 after basic float64 support

### Option C: Adaptive Precision (Future Work)
- ✅ Automatically switch to float64 if matrix is ill-conditioned
- ✅ Optimal user experience
- ❌ Requires condition number estimation (overhead)
- 💡 Could build on top of basic float64 support

**Decision**: Implement basic configurable precision first (this plan), consider advanced options later

---

## Next Steps

1. **Review this plan** - Get feedback, adjust if needed
2. **Implement Phase 1** - Core infrastructure (1-2 hours)
3. **Validate Phase 1** - Run tests, check backward compatibility
4. **Implement Phases 2-3** - Data conversion, API (2-3 hours)
5. **Comprehensive testing** - AFNI comparison, performance profiling
6. **Documentation & Release** - Update docs, merge to main

**Ready to proceed?** Let me know if you want to adjust the approach or start implementation!
