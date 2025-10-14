# Missing Features & Potential Improvements

Comprehensive list of features mentioned in reference materials but not yet implemented.

---

## 1. Temporal Autocorrelation (High Priority)

### What's Missing

**Current Implementation**:
- 1/f (pink) noise via FFT
- Respiratory (~0.3 Hz) and cardiac (~1.0 Hz) peaks
- **NO temporal autocorrelation structure**
- Each timepoint is independent given the frequency spectrum

**What BrainIAK fmrisim Has**:

#### AR(1) - First-order autoregressive
```python
# y_t = ρ * y_{t-1} + ε_t
# ρ ≈ 0.2-0.4 for fMRI (typical)
```
- Simplest temporal correlation
- One parameter (ρ)
- Common in fMRI prewhitening

#### AR(p) - Higher-order autoregressive
```python
# y_t = ρ_1 * y_{t-1} + ρ_2 * y_{t-2} + ... + ρ_p * y_{t-p} + ε_t
# AR(2): ρ_1 ≈ 0.6, ρ_2 ≈ 0.2 (typical)
```
- Better fit to real fMRI noise
- AR(2) or AR(3) usually sufficient
- Captures "sluggish" hemodynamic response artifacts

#### ARMA(p,q) - Autoregressive Moving Average
```python
# AR part: ρ_1*y_{t-1} + ... + ρ_p*y_{t-p}
# MA part: θ_1*ε_{t-1} + ... + θ_q*ε_{t-q}
# ARMA(1,1) is common: ρ ≈ 0.3, θ ≈ 0.2
```
- More flexible than pure AR
- Better for complex autocorrelation patterns
- ARMA(1,1) often sufficient for fMRI

### Can We Do This on GPU?

**YES** - All are implementable on GPU via PyTorch:

#### AR(p) Generation on GPU
```python
# Recursive formulation (sequential, but fast on GPU)
def generate_ar_gpu(rho, n_timepoints, device):
    """
    Generate AR(p) process on GPU
    rho: tensor of AR coefficients [ρ_1, ρ_2, ..., ρ_p]
    """
    p = len(rho)
    y = torch.zeros(n_timepoints, device=device)
    epsilon = torch.randn(n_timepoints, device=device)

    for t in range(p, n_timepoints):
        # Vectorized dot product of past values
        y[t] = torch.dot(rho, y[t-p:t].flip(0)) + epsilon[t]

    return y

# For multiple voxels: vectorize across voxel dimension
# Shape: (n_voxels, n_timepoints)
```

**Speed**: Sequential loop is slower than FFT, but still fast
- ~1ms per voxel on GPU for 300 TRs
- Vectorize across voxels: ~100ms for 10,000 voxels
- Acceptable for batch simulations

#### ARMA(p,q) Generation on GPU
```python
def generate_arma_gpu(rho, theta, n_timepoints, device):
    """
    Generate ARMA(p,q) process on GPU
    rho: AR coefficients
    theta: MA coefficients
    """
    p, q = len(rho), len(theta)
    y = torch.zeros(n_timepoints, device=device)
    epsilon = torch.randn(n_timepoints, device=device)

    for t in range(max(p, q), n_timepoints):
        # AR part
        ar_part = torch.dot(rho, y[t-p:t].flip(0))
        # MA part
        ma_part = torch.dot(theta, epsilon[t-q:t].flip(0))
        y[t] = ar_part + ma_part + epsilon[t]

    return y
```

**Alternative**: Use `torch.nn.Conv1d` for convolution-based implementation (fully parallel)

### Why This Matters

1. **Realistic noise**: Real fMRI has temporal autocorrelation (ρ ≈ 0.2-0.4)
2. **GLM efficiency**: Autocorrelated noise reduces efficiency of estimates
3. **Statistical inference**: t-tests assume independence; autocorrelation inflates false positives
4. **Prewhitening validation**: Need autocorrelated noise to test prewhitening methods

### Implementation Priority: **HIGH**

---

## 2. Noise Parameter Estimation from Real Data (High Priority)

### What's Missing

**Current**: User manually specifies noise parameters
- `noise_level = 1.0` (arbitrary scale)
- `pink_exp = 1.0` (fixed 1/f exponent)
- `resp_freq = 0.35, cardiac_freq = 1.0` (fixed)

**What BrainIAK Has**:

#### Estimate AR Parameters from Real Data
```python
def estimate_ar_from_data(real_fmri_data, mask, ar_order=2):
    """
    Fit AR model to real fMRI residuals

    Returns:
    - ar_coefficients: [ρ_1, ρ_2, ..., ρ_p]
    - noise_variance: σ²
    """
    # 1. Mask data
    # 2. Detrend (remove mean, polynomial trends)
    # 3. Fit AR model using:
    #    - Yule-Walker equations (fast, biased)
    #    - Burg method (better, unbiased)
    #    - Maximum likelihood (slow, optimal)
    # 4. Return coefficients per voxel or mean across ROI
```

**Tools**:
- `statsmodels.tsa.ar_model.AutoReg`
- `nitime.algorithms.autoregressive`
- Custom implementation via Yule-Walker (closed-form solution)

#### Estimate SFNR (Signal Fluctuation to Noise Ratio)
```python
def estimate_sfnr(real_fmri_data, mask):
    """
    SFNR = mean_signal / std_signal (across time)

    Typical values:
    - Good 3T data: SFNR = 150-200
    - Poor data: SFNR = 50-100
    - 7T data: SFNR = 100-150
    """
    mean_signal = data.mean(axis=-1)
    std_signal = data.std(axis=-1)
    sfnr = mean_signal / std_signal
    return sfnr.mean()  # Average across voxels
```

#### Estimate Spatial Smoothness (FWHM)
```python
def estimate_fwhm(residuals, voxel_size):
    """
    Estimate Full Width at Half Maximum (FWHM) of spatial smoothness

    Returns: FWHM in mm (e.g., 3.5mm)

    Used to generate spatially correlated noise
    """
    # SPM/FSL methods:
    # 1. Compute spatial ACF of residuals
    # 2. Fit Gaussian to ACF
    # 3. Extract FWHM from Gaussian width
```

#### Full Noise Model Extraction
```python
def extract_noise_model(real_fmri_data, design, mask):
    """
    Complete noise characterization

    Returns:
    - ar_coefficients: temporal autocorrelation
    - sfnr: signal-to-noise ratio
    - fwhm: spatial smoothness
    - power_spectrum: frequency content
    - physiological_freqs: respiratory/cardiac peaks
    """
    # 1. Fit GLM to get residuals
    # 2. Estimate AR from residuals
    # 3. Compute SFNR
    # 4. Estimate spatial smoothness
    # 5. FFT of residuals for power spectrum
    # 6. Peak detection for physiological frequencies
```

### Why This Matters

1. **Realistic simulations**: Match noise properties of actual scanner/subject
2. **Scanner-specific**: Different scanners have different noise profiles
3. **Subject-specific**: Motion, physiology vary by subject
4. **Validation**: Test if methods work on YOUR data's noise structure

### Implementation: GPU-Compatible

- Yule-Walker: Closed-form solution, easily on GPU
- FFT for power spectrum: Already have FFT on GPU
- SFNR: Simple mean/std operations

### Implementation Priority: **HIGH**

---

## 3. Spatial Autocorrelation (Medium Priority)

### What's Missing

**Current**: Each voxel has **independent** noise realization
- No spatial correlation between neighbors
- Unrealistic for real fMRI

**What's Needed**:

#### Spatial Smoothing with FWHM
```python
def generate_spatially_correlated_noise(
    matrix_size, fwhm_mm, voxel_size, device
):
    """
    Generate noise with spatial correlation

    fwhm_mm: Full Width at Half Maximum in mm (e.g., 3.5)

    Two approaches:
    1. Generate white noise → smooth with Gaussian kernel
    2. Generate correlated noise directly via covariance
    """
    # Approach 1: Smoothing (simpler, faster)
    noise = torch.randn(matrix_size, device=device)
    sigma_voxels = fwhm_mm / voxel_size / 2.355  # FWHM to sigma
    noise_smooth = gaussian_filter_3d_gpu(noise, sigma=sigma_voxels)
    return noise_smooth
```

#### GPU Gaussian Smoothing
```python
def gaussian_filter_3d_gpu(data, sigma, device):
    """
    3D Gaussian smoothing on GPU

    Options:
    1. torch.nn.functional.conv3d with Gaussian kernel
    2. FFT-based convolution (faster for large kernels)
    3. Separable convolution (x, y, z independently)
    """
    # Create Gaussian kernel
    kernel_size = int(6 * sigma + 1)  # 6σ covers 99.7%
    kernel = create_gaussian_kernel_3d(kernel_size, sigma, device)

    # Convolve
    data_padded = F.pad(data, pad=kernel_size//2, mode='reflect')
    smoothed = F.conv3d(data_padded[None, None], kernel[None, None])[0, 0]
    return smoothed
```

**GPU Speed**: 3D convolution on GPU is very fast
- ~10ms for 100×100×50 volume on GPU
- Negligible compared to simulation time

### Why This Matters

1. **Realism**: Real fMRI has spatial smoothness (physiological, scanner PSF)
2. **Cluster inference**: Spatial correlation affects cluster-based statistics
3. **Searchlight analysis**: Spatial patterns matter

### Implementation Priority: **MEDIUM**

---

## 4. SNR/SFNR Targeting (Medium Priority)

### What's Missing

**Current**: Noise level is relative, arbitrary scale
- `noise_level = 1.0` → what does this mean?
- Cannot match specific SNR or SFNR values

**What's Needed**:

```python
def simulate_fmri_run_with_sfnr(
    onsets, betas, hrf,
    target_sfnr=150,  # Typical 3T value
    baseline=100.0,
    ...
):
    """
    Generate data with specific SFNR

    SFNR = mean_signal / std_noise

    Algorithm:
    1. Generate signal (task + baseline)
    2. Compute mean signal level
    3. Generate noise with std = mean_signal / target_sfnr
    4. Add signal + noise
    """
    # Signal
    signal = simulate_signal(onsets, betas, hrf, baseline)
    mean_signal = signal.mean()

    # Noise scaled to achieve target SFNR
    noise_std = mean_signal / target_sfnr
    noise = generate_noise(...) * noise_std

    data = signal + noise

    # Verify SFNR
    achieved_sfnr = data.mean() / data.std()
    assert abs(achieved_sfnr - target_sfnr) < 1

    return data
```

**SNR** (spatial) vs **SFNR** (temporal):
- SNR = signal / noise (spatial contrast)
- SFNR = mean / std (temporal stability)
- fMRI typically reports SFNR

### Why This Matters

1. **Quantitative**: "SFNR = 150" is meaningful, "noise_level = 1.0" is not
2. **Scanner comparison**: Compare 3T vs 7T using SFNR
3. **Power analysis**: "How much power do I have at SFNR = 120?"

### Implementation Priority: **MEDIUM**

---

## 5. Liu & Frank (2004) Metrics (High Priority for Design)

### What's Missing

All the theoretical design optimization metrics:

#### Estimation Efficiency
```python
def compute_estimation_efficiency(design, n_conditions, hrf_length):
    """
    Efficiency for estimating HRF shape

    ε_k = Tr[A_k^(-1)]^(-1)

    where A_k = (X_k^T X_k) / N
    X_k = FIR design for condition k

    Higher = better HRF estimation
    """
    # Extract FIR design for each condition
    # Compute Fisher information A_k = X_k^T X_k
    # Invert and take trace
    # Efficiency = 1 / Tr[A_k^(-1)]
```

#### Detection Power
```python
def compute_detection_power(design, hrf_assumed, effect_size):
    """
    Power to detect activation with assumed HRF

    R_k = (h_0^T A_k h_0) / (h_0^T h_0)

    where h_0 = assumed HRF

    Higher = better detection power
    """
    # Convolve design with assumed HRF
    # Compute contrast variance
    # Power = 1 / variance
```

#### Conditional Entropy
```python
def compute_conditional_entropy(onsets, isi_distribution):
    """
    Randomness of design

    H_r ≈ log₂(Q·ε_norm + 1)

    Higher = more random = fewer confounds
    But trades off with power/efficiency
    """
    # Compute ISI distribution
    # Calculate entropy from probabilities
```

#### Trade-off Curves
```python
def compute_efficiency_power_tradeoff(hrf_length, n_conditions):
    """
    Theoretical trade-off between efficiency and power

    As efficiency increases, power must decrease (and vice versa)

    Returns curve: efficiency vs power for different designs
    """
    # Based on eigenvalue distribution of A_k
    # See Liu & Frank (2004) Figure 1
```

### Why This Matters

1. **Design optimization**: Choose ISI, ordering to maximize efficiency OR power
2. **Theory-driven**: Not just empirical - mathematical foundations
3. **Trade-offs**: Understand what you sacrifice for what you gain
4. **Publication**: Justify design choices quantitatively

### Implementation: Pure Math (Easy)

All these are matrix operations - trivial on GPU:
- Matrix multiplication: `X.T @ X`
- Matrix inversion: `torch.linalg.inv()`
- Eigenvalues: `torch.linalg.eigvals()`
- Trace: `torch.trace()`

### Implementation Priority: **HIGH** (for design optimization)

---

## 6. Design Optimization (High Priority for Design)

### What's Missing

**Current**: User manually specifies:
- ISI distribution
- Event ordering
- Trial counts

**What's Needed**:

#### ISI Optimization
```python
def optimize_isi_distribution(
    n_conditions, n_trials, duration,
    objective='power',  # or 'efficiency' or 'entropy'
    constraints={'min_isi': 2, 'max_isi': 12}
):
    """
    Find optimal ISI distribution

    Methods:
    1. Grid search (slow but complete)
    2. Genetic algorithm (fast, approximate)
    3. Gradient descent (if differentiable)
    """
    # Search over ISI means, ranges
    # Compute efficiency/power/entropy for each
    # Return best design
```

#### Event Ordering Optimization
```python
def optimize_event_ordering(
    n_conditions, n_trials,
    objective='efficiency',
    randomness_constraint=0.7  # Entropy must be > 0.7
):
    """
    Find optimal ordering of events

    Options:
    - Block design (high power, low efficiency, low entropy)
    - Random (low power, high efficiency, high entropy)
    - Permuted block (compromise)
    - m-sequence (maximum efficiency)
    """
    # Generate candidate orderings
    # Score each by efficiency/power/entropy
    # Filter by constraints
    # Return Pareto optimal set
```

#### Multi-Objective Optimization
```python
def pareto_optimize_design(
    objectives=['efficiency', 'power', 'entropy'],
    constraints={...}
):
    """
    Find Pareto frontier of designs

    Returns set of designs where no design dominates another

    Example: Design A has higher efficiency but lower power than B
             Both are Pareto optimal (neither dominates)
    """
    # Genetic algorithm or NSGA-II
    # Return Pareto set
    # User chooses from frontier based on priorities
```

### Why This Matters

1. **Better designs**: Automated optimization beats manual tuning
2. **Principled trade-offs**: See entire Pareto frontier
3. **Scanner time**: Optimize within time/budget constraints

### Implementation: Moderate Complexity

- Grid search: Easy, slow
- Genetic algorithms: `deap` library, GPU-compatible
- Gradient-based: Hard (discrete ordering)

### Implementation Priority: **HIGH** (for experimental design)

---

## 7. GLMdenoise Step 3 - Cross-Validation (Medium Priority)

### What's Missing

**Current**: PC extraction exists, but no cross-validation for selecting number of PCs

**What GLMdenoise Does**:

```python
def glmdenoise_cv(data, design, max_pcs=10, n_folds=5):
    """
    Cross-validated PC selection

    Algorithm:
    1. Extract noise PCs from "noise pool" (non-task voxels)
    2. For each PC count (0, 1, 2, ..., max_pcs):
       a. Split data into train/test folds
       b. Fit GLM with PCs as nuisance regressors on train
       c. Predict test data
       d. Compute test R²
    3. Select PC count that maximizes mean test R²
    4. Refit on all data with optimal PC count

    Returns: GLM results with denoised estimates
    """
```

**Key Challenge**: Must be fast for cross-validation
- 5 folds × 10 PC counts = 50 GLM fits
- With GPU: Still fast (~seconds for 50 fits)

### Why This Matters

1. **Prevents overfitting**: Too many PCs hurt generalization
2. **Automatic**: No manual tuning
3. **Data-driven**: Adapts to noise structure

### Implementation: Moderate

- Already have GLM solver
- Need cross-validation loop
- Need "noise pool" voxel selection

### Implementation Priority: **MEDIUM**

---

## 8. Ridge Regression (Step 4) (Medium Priority)

### What's Missing

**Current**: OLS only (β = (X'X)^(-1) X'Y)

**What Ridge Regression Does**:

```python
def ridge_regression_gpu(X, Y, alpha, device):
    """
    Regularized regression: minimize ||Y - Xβ||² + α||β||²

    Solution: β = (X'X + αI)^(-1) X'Y

    Benefits:
    - Prevents overfitting when X'X is near-singular
    - Stabilizes estimates when regressors are correlated
    - Reduces variance of estimates (increases bias)
    """
    XtX = X.T @ X
    XtX_ridge = XtX + alpha * torch.eye(XtX.shape[0], device=device)
    XtY = X.T @ Y
    beta = torch.linalg.solve(XtX_ridge, XtY)
    return beta
```

**Cross-Validated Lambda**:
```python
def cv_ridge_regression(X, Y, alphas, n_folds=5):
    """
    Select regularization strength (alpha/lambda) via CV

    Typical alphas: [0.001, 0.01, 0.1, 1, 10, 100]
    """
    # For each alpha
    #   For each fold
    #     Fit on train, predict on test
    #   Compute mean test error
    # Select alpha with min test error
```

**Fractional Ridge** (advanced):
```python
def fractional_ridge(X, Y, frac=0.5):
    """
    Fractional regularization: interpolates between OLS and Ridge

    frac=0: OLS (no regularization)
    frac=1: Full ridge
    frac=0.5: Compromise

    See: Rokem & Kay (2020)
    """
```

### Why This Matters

1. **Stability**: Prevents overfitting in FIR estimation (many regressors)
2. **Correlated regressors**: HRF library designs are correlated
3. **Single-trial**: Ridge improves single-trial estimates (low power)

### Implementation: Easy (Already Have Solver)

Just add `+ alpha * I` to `XtX` before inversion.

### Implementation Priority: **MEDIUM**

---

## 9. LSS (Least-Squares-Separate) (Low Priority)

### What's Missing

**Current**: Single-trial estimation via `make_singletrialdesign()`
- One regressor per trial
- All trials fitted simultaneously

**What LSS Does**:

```python
def lss_estimation(data, onsets_per_trial, hrf, tr):
    """
    Fit separate GLM for each trial

    For each trial i:
      Design has TWO regressors:
      1. Trial i (target)
      2. All other trials (nuisance)

    Better than single GLM with all trials because:
    - Reduces collinearity between trials
    - More accurate single-trial estimates
    - But: n_trials times slower
    """
    betas_lss = []
    for i in range(n_trials):
        # Design: [trial_i, all_other_trials]
        design = make_lss_design(onsets_per_trial, i, hrf)
        results = fit_glm(data, design, tr)
        betas_lss.append(results.betas[:, 0])  # First regressor

    return torch.stack(betas_lss)
```

### Why This Matters

1. **Accuracy**: Better single-trial estimates (less bias)
2. **MVPA**: Critical for representational similarity analysis
3. **Connectivity**: Trial-by-trial coupling analyses

### Implementation: Moderate (Need Loop)

- Must fit n_trials separate GLMs
- With GPU: ~10ms × 100 trials = 1 second (acceptable)

### Implementation Priority: **LOW** (nice to have)

---

## 10. Basis Function Expansions (Low Priority)

### What's Missing

**Current**: FIR uses discrete time bins (shifted impulses)
- Many parameters (one per time lag)
- Noisy estimates

**Alternative**: Smooth basis functions

```python
def make_basis_function_design(
    onsets, basis='fourier', n_basis=5, duration=30
):
    """
    Use smooth basis functions instead of FIR

    Basis options:
    1. Fourier: sin/cos at different frequencies
    2. Wavelets: Multi-resolution time-frequency
    3. Gamma functions: Physiologically motivated
    4. Cubic splines: Smooth interpolation

    Benefits:
    - Fewer parameters than FIR
    - Smoother estimates
    - Less noise

    Trade-offs:
    - Assumes smooth HRF (may miss sharp features)
    - Harder to interpret
    """
```

**Example: Fourier Basis**
```python
def fourier_basis(n_timepoints, n_basis, tr):
    """
    Create Fourier basis functions

    Basis k: sin(2πk·t/T), cos(2πk·t/T)
    """
    t = torch.arange(n_timepoints) * tr
    T = n_timepoints * tr

    basis = []
    for k in range(1, n_basis + 1):
        basis.append(torch.sin(2 * np.pi * k * t / T))
        basis.append(torch.cos(2 * np.pi * k * t / T))

    return torch.stack(basis, dim=1)  # (n_timepoints, 2*n_basis)
```

### Why This Matters

1. **Parameter reduction**: 5 basis functions vs 30 FIR lags
2. **Regularization**: Smoothness constraint improves estimates
3. **Interpretation**: Frequency components meaningful

### Implementation Priority: **LOW** (advanced feature)

---

## 11. Spatial Masking & ROI Analysis (Low Priority)

### What's Missing

**Current**: Basic support, not comprehensive

**What's Needed**:

```python
def fit_glm_with_mask(data, design, mask, tr):
    """
    Only fit GLM in masked voxels

    Benefits:
    - Faster (fewer voxels)
    - Excludes non-brain (skull, CSF, air)
    """
    masked_data = data[mask]
    results = fit_glm(masked_data, design, tr)

    # Map back to full volume
    results_full = GLMResults(...)
    results_full.betas[mask] = results.betas
    return results_full
```

**ROI-Based Analysis**:
```python
def fit_glm_per_roi(data, design, rois, tr):
    """
    Fit GLM separately for each ROI

    Returns: dict of {roi_name: GLMResults}
    """
    results_per_roi = {}
    for roi_name, roi_mask in rois.items():
        roi_data = data[roi_mask]
        results_per_roi[roi_name] = fit_glm(roi_data, design, tr)

    return results_per_roi
```

### Implementation Priority: **LOW**

---

## 12. Advanced Noise Models

### Multi-Component Noise Mixing (from BrainIAK)

```python
def generate_multi_component_noise(
    tr, duration, matrix_size,
    components={
        'ar': {'weight': 0.5, 'ar_coefficients': [0.6, 0.2]},
        'physiological': {'weight': 0.3, 'resp_freq': 0.35},
        'drift': {'weight': 0.1, 'n_modes': 3},
        'task_related': {'weight': 0.05, 'design': design_matrix},
        'system': {'weight': 0.05, 'sigma': 0.1}
    },
    device='cuda'
):
    """
    Mix multiple noise components with specified weights

    More realistic than single noise source
    """
    total_noise = torch.zeros(matrix_size + (duration,), device=device)

    for component, params in components.items():
        weight = params['weight']
        if component == 'ar':
            noise = generate_ar_noise(params['ar_coefficients'], ...)
        elif component == 'physiological':
            noise = generate_physiological_noise(params['resp_freq'], ...)
        # ... etc

        total_noise += weight * noise

    return total_noise
```

### Implementation Priority: **MEDIUM**

---

## 13. Multi-Session Support (Low Priority)

### What's Missing

**Current**: Multi-run within session

**What's Needed**: Multi-session with different:
- Scanners
- Days
- HRF variability
- Baseline shifts

```python
def simulate_multi_session_experiment(
    n_sessions, n_runs_per_session,
    session_variability={
        'hrf_shift': 0.5,  # HRF varies ±0.5s across sessions
        'baseline_shift': 5,  # Baseline varies ±5 units
        'noise_scaling': 0.1  # Noise varies ±10%
    }
):
    """
    Simulate longitudinal study
    """
```

### Implementation Priority: **LOW**

---

## 14. Real Data Integration (Medium Priority)

### Loading Real fMRI Data

**Current**: No NIFTI support

**Needed**:
```python
import nibabel as nib

def load_nifti(path):
    """Load NIFTI file"""
    img = nib.load(path)
    data = torch.tensor(img.get_fdata())
    return data, img.affine

def save_nifti(data, affine, path):
    """Save as NIFTI"""
    img = nib.Nifti1Image(data.cpu().numpy(), affine)
    nib.save(img, path)
```

### Implementation Priority: **MEDIUM**

---

## Summary: Implementation Priorities

### HIGH Priority (Critical for Realism & Design)
1. ✅ **AR(p) temporal autocorrelation** - Essential for realistic noise
2. ✅ **ARMA(p,q) noise** - More flexible autocorrelation
3. ✅ **Noise parameter estimation from real data** - Match actual scanner noise
4. ✅ **Liu & Frank metrics** (efficiency, power, entropy) - Design optimization
5. ✅ **Design optimization tools** (ISI, ordering) - Better experiments

### MEDIUM Priority (Useful Enhancements)
6. ⚠️ Spatial autocorrelation (FWHM smoothing)
7. ⚠️ SNR/SFNR targeting
8. ⚠️ GLMdenoise cross-validation
9. ⚠️ Ridge regression
10. ⚠️ Multi-component noise mixing
11. ⚠️ Real data integration (NIFTI I/O)

### LOW Priority (Nice to Have)
12. ⏸️ LSS estimation
13. ⏸️ Basis function expansions
14. ⏸️ Multi-session support
15. ⏸️ Comprehensive ROI tools

---

## GPU Feasibility Summary

| Feature | GPU-Compatible? | Speed Notes |
|---------|----------------|-------------|
| AR(1) | ✅ YES | Sequential but fast (~1ms/voxel) |
| AR(p) | ✅ YES | Vectorize across voxels |
| ARMA(p,q) | ✅ YES | Same as AR(p) |
| Spatial smoothing | ✅ YES | Conv3D is very fast on GPU |
| Yule-Walker AR estimation | ✅ YES | Closed-form matrix ops |
| Liu & Frank metrics | ✅ YES | Pure matrix math |
| Ridge regression | ✅ YES | Just add λI to XtX |
| Cross-validation | ✅ YES | Multiple GLM fits in parallel |
| Fourier basis | ✅ YES | Matrix operations |

**Conclusion**: Everything can be done efficiently on GPU! 🚀

---

## References for Implementation

1. **BrainIAK fmrisim**: https://brainiak.org/docs/brainiak.utils.html#module-brainiak.utils.fmrisim
2. **Liu & Frank (2004)**: "Efficiency, power, and entropy in event-related fMRI"
3. **GLMsingle**: https://github.com/cvnlab/GLMsingle
4. **nitime**: Timeseries analysis, AR fitting
5. **statsmodels**: ARMA model fitting
6. **Fractional Ridge**: Rokem & Kay (2020)

---

**Next Step**: Prioritize based on your research needs!
- Need realistic noise → AR(p) + parameter estimation
- Need design optimization → Liu & Frank metrics
- Need better estimates → Ridge regression + GLMdenoise CV
