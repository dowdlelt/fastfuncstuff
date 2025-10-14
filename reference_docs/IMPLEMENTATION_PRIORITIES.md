# Implementation Priorities - Quick Reference

Based on analysis of BrainIAK fmrisim, Liu & Frank (2004), GLMsingle, and current gaps.

---

## Phase 1: Realistic Noise (v0.2.0) - CRITICAL

### 1.1 Temporal Autocorrelation ⭐⭐⭐⭐⭐

**Why First**:
- Real fMRI has ρ ≈ 0.2-0.4 autocorrelation
- Current implementation assumes independence (WRONG)
- Affects ALL downstream analyses

**What to Implement**:
```python
# noise.py additions
def generate_ar1_noise_gpu(rho, n_timepoints, matrix_size, device):
    """AR(1): y_t = ρ*y_{t-1} + ε_t"""

def generate_ar_noise_gpu(rho_coeffs, n_timepoints, matrix_size, device):
    """AR(p): General autoregressive"""

def generate_arma_noise_gpu(ar_coeffs, ma_coeffs, n_timepoints, matrix_size, device):
    """ARMA(p,q): Most flexible"""
```

**GPU Implementation**: ✅ Confirmed feasible
- Sequential generation: ~1ms per voxel
- Vectorize across voxels: ~100ms for 10K voxels
- Alternative: Conv1D approach for parallelization

**Effort**: 1-2 days
**Impact**: HIGH - Fundamental for realism

---

### 1.2 Noise Parameter Estimation ⭐⭐⭐⭐⭐

**Why Critical**:
- Match YOUR scanner's noise
- Extract AR coefficients from real data
- Quantitative validation

**What to Implement**:
```python
# noise.py additions
def estimate_ar_from_residuals(residuals, ar_order='auto', max_order=5):
    """
    Fit AR model to GLM residuals
    Uses: Yule-Walker equations (fast, GPU-compatible)
    Returns: AR coefficients, noise variance
    """

def estimate_noise_parameters(real_data, design, tr, mask):
    """
    Complete noise characterization
    Returns: {
        'ar_coefficients': [ρ_1, ρ_2, ...],
        'sfnr': 150.0,
        'power_spectrum': ...,
        'physio_freqs': {'respiratory': 0.35, 'cardiac': 1.02}
    }
    """

def simulate_with_real_noise_model(noise_params, onsets, betas, hrf, ...):
    """Use extracted parameters for simulation"""
```

**GPU Implementation**: ✅ Yule-Walker is closed-form
- Matrix operations: `torch.linalg.solve()`
- Already have FFT for power spectrum

**Effort**: 2-3 days
**Impact**: HIGH - Enables realistic scanner-specific simulations

---

### 1.3 Spatial Correlation ⭐⭐⭐

**Why Important**:
- Real fMRI smoothed (scanner PSF, physiology)
- Typical FWHM: 2-5mm

**What to Implement**:
```python
# noise.py additions
def generate_spatially_correlated_noise(
    matrix_size, fwhm_mm, voxel_size, temporal_noise, device
):
    """
    Apply 3D Gaussian smoothing
    Uses: torch.nn.functional.conv3d
    """

def estimate_spatial_fwhm(residuals, voxel_size):
    """Estimate smoothness from real data"""
```

**GPU Implementation**: ✅ Conv3D native
- ~10ms for 100×100×50 volume

**Effort**: 1 day
**Impact**: MEDIUM - Nice for realism, not critical

---

### 1.4 SNR/SFNR Targeting ⭐⭐⭐

**Why Useful**:
- Quantitative noise levels
- "SFNR=150" vs "noise_level=1.0"

**What to Implement**:
```python
# simulation.py modifications
def simulate_fmri_run(
    ...,
    target_sfnr=None,  # If specified, scale noise to achieve this
    target_snr=None,   # Alternative: spatial SNR
):
    """Auto-scale noise to match target SFNR/SNR"""
```

**Effort**: 0.5 days
**Impact**: MEDIUM - Better specification

---

## Phase 2: Design Optimization (v0.2.1) - HIGH VALUE

### 2.1 Liu & Frank Metrics ⭐⭐⭐⭐⭐

**Why Essential**:
- Quantify efficiency vs power trade-off
- Design optimization foundation
- Theory-driven experimental design

**What to Implement**:
```python
# NEW: metrics.py
def compute_estimation_efficiency(design, n_conditions, hrf_length, tr):
    """
    ε_k = Tr[A_k^(-1)]^(-1)
    Higher = better HRF shape estimation
    """

def compute_detection_power(design, hrf_assumed, effect_size):
    """
    R_k = (h_0^T A_k h_0) / (h_0^T h_0)
    Higher = better activation detection
    """

def compute_conditional_entropy(onsets, n_conditions):
    """
    H_r ≈ log₂(Q·ε_norm + 1)
    Higher = more randomness = fewer confounds
    """

def compute_efficiency_power_tradeoff(hrf_length, n_conditions, alpha_range):
    """
    Theoretical trade-off curves
    Returns: (efficiency_curve, power_curve, alpha_values)
    """
```

**GPU Implementation**: ✅ Pure matrix math
- Matrix multiplication, inversion, eigenvalues
- All torch.linalg operations

**Effort**: 2-3 days
**Impact**: HIGH - Core theory implementation

---

### 2.2 Design Optimization Tools ⭐⭐⭐⭐

**Why Important**:
- Automated design generation
- Better than manual tuning
- Multi-objective optimization

**What to Implement**:
```python
# NEW: design_opt.py
def optimize_isi_distribution(
    n_conditions, n_trials, duration, tr,
    objective='balanced',  # or 'power', 'efficiency', 'entropy'
    constraints={'min_isi': 2, 'max_isi': 12}
):
    """
    Search for optimal ISI distribution
    Methods: Grid search or genetic algorithm
    """

def optimize_event_ordering(
    n_conditions, n_trials_per_condition,
    objective='efficiency',
    randomness_constraint=0.7
):
    """
    Find optimal event sequence
    Options: block, random, permuted-block, m-sequence
    """

def pareto_optimize_design(
    n_conditions, n_trials, duration, tr,
    objectives=['efficiency', 'power', 'entropy']
):
    """
    Multi-objective optimization
    Returns: Pareto frontier of designs
    """
```

**Effort**: 3-4 days
**Impact**: HIGH - Practical design tool

---

### 2.3 Design Comparison & Validation ⭐⭐⭐

**What to Implement**:
```python
# metrics.py additions
def compare_designs(designs_dict, hrf_assumed, effect_size):
    """
    Compare multiple designs on efficiency/power/entropy
    Returns: DataFrame with metrics for each design
    """

def validate_design_via_simulation(
    design, hrf_assumed, effect_sizes, noise_levels, n_sims=100
):
    """
    Empirical validation of design performance
    Returns: Statistical power, estimation variance, etc.
    """
```

**Effort**: 1-2 days
**Impact**: MEDIUM - Helpful validation

---

## Phase 3: Advanced GLM (v0.2.2) - MEDIUM PRIORITY

### 3.1 Ridge Regression ⭐⭐⭐

**Why Useful**:
- Stabilizes FIR estimates
- Better for correlated regressors
- Single-trial improvement

**What to Implement**:
```python
# glm_core.py additions
def fit_glm_ridge(data, design, tr, alphas='auto', cv_folds=5, **kwargs):
    """
    Ridge regression with cross-validated lambda
    β = (X'X + λI)^(-1) X'Y
    """

def fractional_ridge(data, design, tr, fracs=[0.2, 0.5, 0.8], **kwargs):
    """
    Fractional ridge (Rokem & Kay 2020)
    Interpolates between OLS and full ridge
    """
```

**Effort**: 1-2 days
**Impact**: MEDIUM - Improves estimates

---

### 3.2 GLMdenoise Cross-Validation ⭐⭐⭐

**Why Important**:
- Automatic PC selection
- Prevents overfitting
- GLMsingle Step 3 completion

**What to Implement**:
```python
# glm_core.py additions
def fit_glm_denoise_cv(
    data, design, tr,
    max_pcs=10, cv_folds=5,
    noise_pool='auto'  # or provide mask
):
    """
    Cross-validated GLMdenoise
    1. Extract PCs from noise pool
    2. CV to select optimal # of PCs
    3. Refit with optimal PCs
    """
```

**Effort**: 2-3 days
**Impact**: MEDIUM - Better denoising

---

### 3.3 LSS (Least-Squares-Separate) ⭐⭐

**Why Nice-to-Have**:
- Better single-trial estimates
- Critical for MVPA/RSA

**What to Implement**:
```python
# glm_core.py additions
def fit_glm_lss(data, onsets_per_trial, hrf, tr, **kwargs):
    """
    Fit separate GLM for each trial
    Returns: (n_trials, n_voxels) beta matrix
    """
```

**Effort**: 1 day
**Impact**: LOW-MEDIUM - Niche use case

---

## Phase 4: Integration & Workflow (v0.3.0) - FUTURE

### 4.1 Real Data Integration ⭐⭐⭐

```python
# NEW: io.py
def load_fmri(path):
    """Load NIFTI/GIFTI/CIFTI"""

def save_fmri(data, path, affine=None):
    """Save results"""

def load_design_from_bids(events_tsv, tr, hrf):
    """Load BIDS events.tsv"""
```

**Effort**: 1-2 days
**Impact**: MEDIUM - Practical integration

---

### 4.2 Complete Design Workflow ⭐⭐⭐⭐

```python
# NEW: workflow.py
class ExperimentalDesigner:
    """
    End-to-end experimental design workflow

    Usage:
    designer = ExperimentalDesigner(n_conditions=2, duration=300, tr=1.0)
    designer.extract_noise_from_data('pilot_data.nii.gz')
    designer.optimize_design(objective='balanced')
    designer.validate_design(n_sims=100)
    designer.export_design('experiment_design.tsv')
    designer.export_report('design_report.pdf')
    """
```

**Effort**: 3-5 days
**Impact**: HIGH - Complete tool

---

## Quick Decision Tree

### "I need realistic noise NOW"
→ **Phase 1.1 + 1.2**: AR noise + parameter estimation (3-5 days)

### "I need to optimize my experimental design"
→ **Phase 2.1 + 2.2**: Liu & Frank metrics + design optimization (5-7 days)

### "I need better GLM estimates"
→ **Phase 3.1 + 3.2**: Ridge regression + GLMdenoise CV (3-5 days)

### "I need a complete pipeline"
→ **All phases**: ~3-4 weeks full-time

---

## Recommended Order

1. **Week 1**: AR/ARMA noise (1.1) + parameter estimation (1.2)
   - Biggest gap in current implementation
   - Foundational for everything else

2. **Week 2**: Liu & Frank metrics (2.1)
   - Theoretical foundation
   - Needed for optimization

3. **Week 3**: Design optimization (2.2) + SFNR targeting (1.4)
   - Practical tools
   - Uses metrics from Week 2

4. **Week 4**: Spatial correlation (1.3) + Ridge regression (3.1)
   - Polish realism
   - Improve estimates

5. **Future**: Workflow integration (4.2) + Real data I/O (4.1)

---

## Critical Dependencies

```
Phase 1.2 (Parameter Estimation)
    ↓
Phase 1.1 (AR/ARMA Noise) ← Foundation for realistic simulations
    ↓
Phase 2.1 (Metrics) ← Theory
    ↓
Phase 2.2 (Optimization) ← Practical application of theory
    ↓
Phase 2.3 (Validation) ← Combine simulation + theory
    ↓
Phase 4.2 (Workflow) ← Integration
```

---

## GPU Acceleration Summary

✅ **Confirmed GPU-Compatible**:
- AR(p) noise generation: Sequential but vectorizable
- ARMA(p,q): Same as AR
- Yule-Walker AR estimation: Closed-form
- Spatial smoothing: Conv3D native
- Liu & Frank metrics: Matrix operations
- Ridge regression: Trivial addition
- Cross-validation: Parallel GLM fits

⚠️ **Requires Care**:
- Design optimization: Use vectorized operations, avoid Python loops
- Parameter estimation: Batch process voxels

❌ **Not GPU**:
- Genetic algorithms: Mostly CPU, but objective function on GPU
- HTML generation: CPU only (but negligible time)

---

## Effort vs Impact Matrix

```
                    HIGH IMPACT               LOW IMPACT
            ┌─────────────────────────┬─────────────────────────┐
            │                         │                         │
HIGH        │  AR/ARMA Noise (1.1)    │  Spatial Corr (1.3)     │
EFFORT      │  Param Estimation (1.2) │  LSS (3.3)              │
            │  Liu & Frank (2.1)      │  Multi-session (low)    │
            │  Design Opt (2.2)       │                         │
            │  Workflow (4.2)         │                         │
            │                         │                         │
            ├─────────────────────────┼─────────────────────────┤
            │                         │                         │
LOW         │  Ridge Regression (3.1) │  Basis Functions (low)  │
EFFORT      │  SNR/SFNR (1.4)         │  ROI Tools (low)        │
            │  GLMdenoise CV (3.2)    │                         │
            │                         │                         │
            │                         │                         │
            └─────────────────────────┴─────────────────────────┘

🎯 Focus on top-left quadrant first!
```

---

## Bottom Line

**Most Critical Missing Features**:
1. ⭐⭐⭐⭐⭐ **AR/ARMA temporal autocorrelation** - Fundamental realism gap
2. ⭐⭐⭐⭐⭐ **Noise parameter estimation** - Match real data
3. ⭐⭐⭐⭐⭐ **Liu & Frank metrics** - Design theory foundation
4. ⭐⭐⭐⭐ **Design optimization** - Practical application

**Quick Wins** (high impact, low effort):
- Ridge regression (1 day)
- SNR/SFNR targeting (0.5 day)

**Foundation for Everything**:
- AR/ARMA noise + parameter estimation (3-5 days)
- Without this, simulations are unrealistic

**Next Big Step**:
- Implement Phase 1 (Realistic Noise) → Week 1
- Then decide: Design optimization (Phase 2) OR Better estimates (Phase 3)
