# FastFuncSim: Comprehensive Implementation Plan
**Version 0.2 Roadmap - Efficiency, Estimation, and Real Noise**

## Executive Summary

This plan integrates three critical streams of work:
1. **Liu & Frank (2004)**: Efficiency vs Power trade-offs in experimental design
2. **BrainIAK fmrisim**: Realistic noise with AR(n) processes and parameter estimation
3. **Design Optimization**: Tools for creating experiments that balance detection and estimation

**Core Philosophy**: FastFuncSim must support both *detection* (finding activation) and *estimation* (recovering true HRF shapes) while respecting real fMRI noise properties.

---

## Part 1: Theoretical Foundation (Liu & Frank 2004)

### Key Concepts from Paper

#### 1. Estimation Efficiency (ε)
**Definition**: Inverse of variance in HRF shape estimates
- Measures: "How well can I estimate the SHAPE of the HRF?"
- Formula: `ε_tot ≈ N·f(p,Q) / Tr[A_k^(-1)]`
- Maximized when: Eigenvalues of A_k are equally distributed (random/m-sequence designs)

#### 2. Detection Power (R)
**Definition**: Inverse of variance in amplitude estimates (with assumed HRF)
- Measures: "How well can I detect THAT something happened?"
- Formula: `R_tot ≈ N·f(p,Q) · (h_0^T A_k h_0)/(h_0^T h_0)`
- Maximized when: Single large eigenvalue of A_k (block designs)

#### 3. The Fundamental Trade-Off
**Cannot maximize both simultaneously!**

Characterized by parameter α (eigenvalue distribution):
- α = 1/k → max efficiency, min power (random designs)
- α = 1 → min efficiency, max power (block designs)
- Intermediate α → balanced designs (semi-random, mixed)

Trade-off equation:
```
R_tot ≈ N·f(p,Q)·R(α,θ)
ε_tot ≈ N·f(p,Q)·ξ(α)

where R(α,θ) = α·cos²θ + ((1-α)/(k-1))·sin²θ
      ξ(α) = α(1-α)M / (1 + α(k²-2k))
```

#### 4. Conditional Entropy (H_r)
**Definition**: Average uncertainty in next trial type given r previous trials
- Measures: "How random/unpredictable is the design?"
- Critical for: Avoiding habituation, anticipation
- Empirical relation: `H_r ≈ log₂(Q·ε_norm + 1)`
- Finding: Entropy increases with estimation efficiency!

#### 5. Optimal Frequency of Occurrence
**Key Result**: For Q trial types, optimal p = 1/(Q+1)
- Equalizes efficiency for individual events and pairwise contrasts
- Incorporates null events explicitly

---

## Part 2: Implementation Priorities

### Phase 1: Metrics & Analysis Tools (v0.2.0)

#### Module: `fastfuncsim/metrics.py`

**A. Estimation Efficiency**
```python
def compute_estimation_efficiency(design_matrix: torch.Tensor,
                                  hrf_length: int,
                                  n_conditions: int,
                                  device: torch.device) -> Dict:
    """
    Compute estimation efficiency metrics for design

    Returns
    -------
    dict with:
        epsilon_tot: Overall estimation efficiency
        epsilon_per_condition: Per-condition efficiency
        epsilon_min: Minimum efficiency across contrasts
        A_k: Autocorrelation matrix
        eigenvalues: Eigenvalue distribution of A_k
        alpha: Eigenvalue concentration parameter
    """
```

**B. Detection Power**
```python
def compute_detection_power(design_matrix: torch.Tensor,
                           assumed_hrf: torch.Tensor,
                           device: torch.device) -> Dict:
    """
    Compute detection power metrics

    Returns
    -------
    dict with:
        R_tot: Overall detection power
        R_per_condition: Per-condition power
        R_min: Minimum power across contrasts
        noncentrality: Non-centrality parameters
    """
```

**C. Trade-Off Analysis**
```python
def compute_efficiency_power_tradeoff(design_matrix: torch.Tensor,
                                     hrf_length: int,
                                     assumed_hrf: torch.Tensor,
                                     device: torch.device) -> Dict:
    """
    Complete efficiency-power trade-off analysis

    Returns
    -------
    dict with:
        epsilon: Estimation efficiency
        R: Detection power
        alpha: Eigenvalue parameter
        theta: Angle between HRF and dominant eigenvector
        theoretical_curve: Expected trade-off from theory
        actual_point: Where this design falls
    """
```

**D. Conditional Entropy**
```python
def compute_conditional_entropy(onset_sequence: torch.Tensor,
                               order: int = 2,
                               n_conditions: int = None) -> Dict:
    """
    Compute conditional entropy of design

    Parameters
    ----------
    order : int
        Order of conditional entropy (1, 2, or 3)

    Returns
    -------
    dict with:
        H_r: Conditional entropy (bits)
        predictability: 1/2^H_r
        randomness: 2^H_r (linear measure)
    """
```

**E. Design Quality Metrics**
```python
def evaluate_design_quality(onsets: torch.Tensor,
                           tr: float,
                           n_timepoints: int,
                           assumed_hrf: torch.Tensor,
                           hrf_length: int = 30,
                           device: torch.device = None) -> Dict:
    """
    Complete design quality assessment

    Returns all metrics:
    - Estimation efficiency
    - Detection power
    - Conditional entropy
    - Trade-off position
    - Optimal frequency check
    - Recommendations
    """
```

---

### Phase 2: Design Optimization Tools (v0.2.1)

#### Module: `fastfuncsim/design_opt.py`

**A. ISI Optimization**
```python
def optimize_isis(n_trials: int,
                 n_conditions: int,
                 target_mean_isi: float,
                 isi_range: Tuple[float, float] = (2, 8),
                 tr: float = 1.0,
                 criterion: str = 'efficiency',  # or 'power' or 'balanced'
                 device: torch.device = None) -> torch.Tensor:
    """
    Generate optimal ISI sequence using truncated Poisson

    Based on Liu & Frank Eq. 17: f(p,Q) optimization

    Parameters
    ----------
    criterion : str
        'efficiency': Maximize estimation efficiency
        'power': Maximize detection power
        'balanced': Optimize for intermediate trade-off
        'entropy': Maximize conditional entropy
    """
```

**B. Event Ordering Optimization**
```python
def optimize_event_order(isis: torch.Tensor,
                        n_conditions: int,
                        method: str = 'm_sequence',  # or 'permuted_block' or 'genetic'
                        target_efficiency: float = None,
                        target_power: float = None,
                        target_entropy: float = None,
                        device: torch.device = None) -> torch.Tensor:
    """
    Optimize ordering of events to achieve target metrics

    Methods:
    - 'm_sequence': Use prime m-sequences (high efficiency, high entropy)
    - 'permuted_block': Start with block, permute to target
    - 'clustered_m_seq': Start with m-seq, cluster to increase power
    - 'genetic': Genetic algorithm optimization (Wager & Nichols 2003)
    - 'mixed': Concatenate block + m-sequence
    """
```

**C. Design Search**
```python
def search_design_space(n_timepoints: int,
                       n_conditions: int,
                       target_metrics: Dict,
                       constraints: Dict,
                       search_method: str = 'genetic',
                       n_iterations: int = 1000,
                       device: torch.device = None) -> List[Dict]:
    """
    Search design space to find optimal designs

    Parameters
    ----------
    target_metrics : dict
        {'efficiency': 0.8, 'power': 0.6, 'entropy': 2.5}
    constraints : dict
        {'min_isi': 2, 'max_isi': 8, 'alternate_conditions': True}

    Returns
    -------
    List of designs ranked by distance to target metrics
    """
```

**D. Multi-Objective Optimization**
```python
def pareto_optimal_designs(n_timepoints: int,
                          n_conditions: int,
                          objectives: List[str],  # ['efficiency', 'power', 'entropy']
                          n_designs: int = 100,
                          device: torch.device = None) -> List[Dict]:
    """
    Find Pareto frontier of designs

    Returns designs that are non-dominated (no other design is better
    on all objectives)
    """
```

---

### Phase 3: Advanced Noise Modeling (v0.2.2)

#### Module: `fastfuncsim/noise_advanced.py`

**Integrate BrainIAK fmrisim concepts**

**A. AR(n) Noise Generation**
```python
def generate_ar_noise(tr: float,
                     duration_s: float,
                     ar_order: int = 2,
                     ar_coefs: Optional[torch.Tensor] = None,
                     ma_order: int = 0,
                     ma_coefs: Optional[torch.Tensor] = None,
                     matrix_size: Tuple[int, int] = (1, 1),
                     normalize: bool = True,
                     device: torch.device = None) -> torch.Tensor:
    """
    Generate ARMA(p,q) noise process

    If ar_coefs/ma_coefs not provided, estimate from empirical fMRI
    Default: AR(2) with typical fMRI coefficients
    """
```

**B. Multi-Component Noise Mixing**
```python
def generate_mixed_noise(tr: float,
                        duration_s: float,
                        components: Dict,
                        matrix_size: Tuple[int, int] = (1, 1),
                        device: torch.device = None) -> torch.Tensor:
    """
    Mix multiple noise components

    Parameters
    ----------
    components : dict
        {
            'ar': {'weight': 0.5, 'ar_coefs': [0.6, 0.2]},
            'physiological': {'weight': 0.3, 'resp_freq': 0.35},
            'drift': {'weight': 0.1, 'n_modes': 3},
            'task': {'weight': 0.05, 'design': design_matrix},
            'system': {'weight': 0.05}
        }
    """
```

**C. Noise Parameter Estimation from Real Data**
```python
def estimate_noise_params(real_data: torch.Tensor,
                         tr: float,
                         mask: Optional[torch.Tensor] = None,
                         fit_ar: bool = True,
                         max_ar_order: int = 5,
                         estimate_spatial: bool = True,
                         device: torch.device = None) -> Dict:
    """
    Estimate noise parameters from real fMRI data

    Returns
    -------
    dict with:
        ar_order: Selected AR order (via AIC/BIC)
        ar_coefs: AR coefficients
        ma_order: MA order
        ma_coefs: MA coefficients
        sfnr: Signal fluctuation to noise ratio
        snr: Spatial signal to noise ratio
        fwhm: Spatial smoothness (mm)
        resp_freq: Detected respiratory frequency
        cardiac_freq: Detected cardiac frequency
        noise_model: Callable to generate matched noise
    """
```

**D. SNR/SFNR Control**
```python
def generate_noise_with_snr(tr: float,
                           duration_s: float,
                           target_snr: float = None,
                           target_sfnr: float = None,
                           noise_type: str = 'realistic',  # 'white', 'ar', 'realistic'
                           **kwargs) -> torch.Tensor:
    """
    Generate noise targeting specific SNR or SFNR

    SNR = spatial signal-to-noise (mean/spatial_std)
    SFNR = temporal signal-to-noise (mean/temporal_std)
    """
```

**E. Spatial Correlation**
```python
def add_spatial_correlation(noise: torch.Tensor,
                           fwhm: float,  # in mm or voxels
                           voxel_size: Tuple[float, float, float] = (2, 2, 2),
                           correlation_type: str = 'gaussian',
                           device: torch.device = None) -> torch.Tensor:
    """
    Add spatial correlation to noise

    Uses GPU-accelerated 3D Gaussian smoothing
    """
```

---

### Phase 4: HRF Estimation Validation (v0.2.3)

#### Module: `fastfuncsim/estimation_quality.py`

**Validate ability to recover true HRF**

**A. HRF Recovery Metrics**
```python
def assess_hrf_recovery(estimated_hrf: torch.Tensor,
                       true_hrf: torch.Tensor,
                       device: torch.device = None) -> Dict:
    """
    Compute metrics for HRF estimation quality

    Returns
    -------
    dict with:
        correlation: Correlation between true and estimated
        normalized_rmse: RMSE normalized by signal amplitude
        peak_time_error: Error in time-to-peak (seconds)
        peak_amplitude_error: Error in peak amplitude
        undershoot_error: Error in undershoot amplitude
        auc_error: Error in area under curve
        derivative_correlation: Match in temporal derivatives
    """
```

**B. Simulation-Based Power Analysis**
```python
def estimate_hrf_recovery_power(design_config: Dict,
                               hrf_library: torch.Tensor,
                               noise_params: Dict,
                               n_simulations: int = 1000,
                               device: torch.device = None) -> Dict:
    """
    Estimate power to correctly identify HRF via simulations

    For each HRF in library:
    1. Simulate data with that HRF
    2. Fit model with HRF library selection
    3. Check if correct HRF selected

    Returns
    -------
    dict with:
        selection_accuracy: P(correct HRF selected)
        confusion_matrix: Which HRFs confused with which
        recovery_metrics: HRF recovery quality when correct
        power_curves: Accuracy vs effect size
    """
```

**C. Efficiency vs Estimation Trade-Off**
```python
def efficiency_estimation_tradeoff(design_config: Dict,
                                  hrf_library: torch.Tensor,
                                  noise_params: Dict,
                                  n_simulations: int = 100,
                                  device: torch.device = None) -> Dict:
    """
    Characterize trade-off between detection efficiency and
    HRF estimation quality

    Key question: Does high estimation efficiency (from Liu & Frank)
    actually lead to better HRF recovery?

    Returns
    -------
    Plots and metrics showing:
    - Estimation efficiency vs HRF recovery correlation
    - Detection power vs HRF recovery correlation
    - Optimal designs for different goals
    """
```

---

### Phase 5: Complete Experimental Design Workflow (v0.3.0)

#### Module: `fastfuncsim/design_workflow.py`

**End-to-end design optimization**

```python
class ExperimentalDesigner:
    """
    Complete workflow for designing optimal fMRI experiments

    Usage:
    ------
    designer = ExperimentalDesigner(
        n_conditions=3,
        total_duration_s=300,
        tr=1.0,
        objectives={'efficiency': 1.0, 'power': 0.5, 'entropy': 0.3},
        constraints={'min_isi': 2, 'max_isi': 8}
    )

    # Estimate noise from pilot data
    designer.estimate_noise_from_data(pilot_data)

    # Generate candidate designs
    candidates = designer.generate_candidates(n_designs=100)

    # Evaluate designs
    evaluations = designer.evaluate_candidates(candidates)

    # Select best
    best_design = designer.select_best(evaluations)

    # Validate with simulation
    validation = designer.validate_design(best_design, n_sims=1000)

    # Export for experiment
    designer.export_design(best_design, format='psychopy')
    """

    def __init__(self, ...):
        pass

    def estimate_noise_from_data(self, pilot_data: torch.Tensor):
        """Estimate noise parameters from pilot/existing data"""

    def generate_candidates(self, n_designs: int) -> List[Design]:
        """Generate candidate designs using various methods"""

    def evaluate_candidates(self, designs: List[Design]) -> List[Evaluation]:
        """Evaluate all candidates on all metrics"""

    def select_best(self, evaluations: List[Evaluation]) -> Design:
        """Select best design based on objectives"""

    def validate_design(self, design: Design, n_sims: int) -> Validation:
        """Validate via simulation"""

    def visualize_design(self, design: Design):
        """Create comprehensive visualization"""

    def export_design(self, design: Design, format: str):
        """Export to PsychoPy, E-Prime, AFNI, SPM, etc."""
```

---

## Part 3: Integration with Existing Code

### Updates to Existing Modules

**A. `glm_core.py`**
- Add `compute_design_matrix_properties()` - compute A_k, eigenvalues
- Add `compute_efficiency_metrics()` - estimation efficiency from GLM results
- Store Fisher information matrix for efficiency calculations

**B. `design.py`**
- Add efficiency/power computation to `build_glm_design()`
- Add `evaluate_design()` function that wraps metrics
- Add helper for converting between onset formats

**C. `simulation.py`**
- Add `simulate_with_quality_metrics()` that tracks efficiency/power
- Add `batch_design_comparison()` for comparing multiple designs
- Integrate noise parameter estimation

**D. `hrf.py`**
- Add `evaluate_hrf_library_separability()` - how distinguishable are HRFs?
- Add HRF parameter perturbation for sensitivity analysis

---

## Part 4: Example Usage Patterns

### Example 1: Design a High-Efficiency Experiment
```python
import fastfuncsim as ffs

# Goal: Estimate HRF shape accurately for different conditions
designer = ffs.ExperimentalDesigner(
    n_conditions=3,
    total_duration_s=300,
    tr=1.0,
    primary_goal='estimation',  # Prioritize HRF estimation
    device=ffs.get_device()
)

# Estimate noise from pilot data
designer.estimate_noise_from_data(pilot_fmri_data)

# Generate and evaluate designs
designs = designer.generate_candidates(method='m_sequence', n=10)
best = designer.select_best(designs)

# Validate
validation = designer.validate_design(best, n_simulations=1000)
print(f"Expected HRF correlation: {validation['mean_hrf_correlation']:.3f}")
print(f"HRF selection accuracy: {validation['selection_accuracy']:.3f}")
```

### Example 2: Balanced Design for Detection + Estimation
```python
designer = ffs.ExperimentalDesigner(
    n_conditions=2,
    total_duration_s=290,
    tr=1.0,
    objectives={'efficiency': 0.5, 'power': 0.5},  # Equal weight
    device=device
)

# Multi-objective optimization finds Pareto frontier
pareto_designs = designer.find_pareto_optimal(n_designs=100)

# User selects preferred trade-off point
for i, d in enumerate(pareto_designs[:5]):
    print(f"{i}: Efficiency={d.efficiency:.3f}, Power={d.power:.3f}, "
          f"Entropy={d.entropy:.2f} bits")

selected = pareto_designs[2]  # User choice
```

### Example 3: Replicate Liu & Frank Figure 1
```python
# Reproduce efficiency-power trade-off curves
from fastfuncsim.metrics import plot_efficiency_power_tradeoff

designs = {
    'random': ffs.generate_random_onsets(...),
    'm_sequence': ffs.generate_m_sequence_design(...),
    'block_1': ffs.generate_block_design(n_blocks=1, ...),
    'block_30': ffs.generate_block_design(n_blocks=30, ...),
    'mixed': ffs.generate_mixed_design(...),
}

plot_efficiency_power_tradeoff(
    designs=designs,
    hrf_length=30,
    assumed_hrf=canonical_hrf,
    Q=2,
    show_theory=True  # Overlay theoretical curves
)
```

### Example 4: Estimate Noise from Real Data
```python
# Load real fMRI data
real_data = load_nifti('sub-01_task-rest_bold.nii.gz')

# Estimate noise properties
noise_params = ffs.estimate_noise_params(
    real_data,
    tr=2.0,
    fit_ar=True,
    max_ar_order=5,
    estimate_spatial=True
)

print(f"AR order: {noise_params['ar_order']}")
print(f"AR coefficients: {noise_params['ar_coefs']}")
print(f"SFNR: {noise_params['sfnr']:.1f}")
print(f"Spatial FWHM: {noise_params['fwhm']:.1f} mm")

# Use matched noise in simulations
noise = ffs.generate_noise_from_params(
    tr=2.0,
    duration_s=300,
    params=noise_params,
    device=device
)
```

---

## Part 5: Testing & Validation Strategy

### Unit Tests
1. Verify efficiency calculations match Liu & Frank equations
2. Verify power calculations match Liu & Frank equations
3. Test entropy calculations on known sequences
4. Validate AR noise generation against statsmodels
5. Test parameter estimation on synthetic data

### Integration Tests
1. Reproduce key figures from Liu & Frank (2004)
2. Reproduce BrainIAK fmrisim noise examples
3. Compare designs against published m-sequences
4. Validate against GLMsingle MATLAB outputs

### Validation Studies
1. Run simulations comparing designs at different efficiency-power trade-offs
2. Validate HRF recovery under different noise conditions
3. Test on real experimental data where ground truth known
4. Compare to published experimental designs

---

## Part 6: Documentation & Examples

### New Example Scripts

**`example_design_optimization.py`**
- Complete design optimization workflow
- Comparison of different design types
- Efficiency-power-entropy trade-offs

**`example_noise_modeling.py`**
- AR noise generation
- Parameter estimation from real data
- SNR/SFNR control

**`example_hrf_recovery.py`**
- HRF library selection validation
- Recovery quality metrics
- Power analysis for HRF estimation

**`example_liu_frank_replication.py`**
- Replicate all figures from Liu & Frank (2004)
- Serves as validation of implementation

---

## Part 7: Performance Considerations

### GPU Optimization
- Batch evaluation of multiple designs on GPU
- Parallel simulation for power analysis
- Efficient matrix operations for efficiency/power computation

### Memory Management
- Streaming for large design space searches
- Chunked processing for parameter estimation
- Caching of frequently-used HRF libraries

### Speed Targets
- Design evaluation: <100ms per design
- Parameter estimation: <5s for typical 4D dataset
- Power analysis (1000 sims): <5 min on GPU
- Design search (1000 candidates): <10 min on GPU

---

## Part 8: Future Enhancements (v0.4+)

1. **GLMdenoise Integration** (Step 3 of GLMsingle)
   - PC extraction from noise pool
   - Cross-validation for PC selection
   - Full GLMdenoise pipeline

2. **Ridge Regression** (Step 4 of GLMsingle)
   - Fractional ridge regression
   - Per-voxel regularization
   - Cross-validated lambda selection

3. **Nonlinear Models**
   - Volterra kernels for nonlinear effects
   - HRF adaptation/saturation models

4. **Multi-Session Support**
   - Session-wise normalization
   - Cross-session design optimization
   - Longitudinal studies

5. **Advanced Spatial Models**
   - ROI-based analysis
   - Searchlight analysis
   - Spatial basis functions

6. **Real-Time Applications**
   - Online design optimization
   - Adaptive designs based on interim data
   - Real-time GLM fitting

---

## Part 9: Key References to Implement

**Core Theory**
- Liu & Frank (2004) Part I: Theory - THIS PAPER
- Liu (2004) Part II: Design of Experiments
- Dale (1999): Optimal experimental design for event-related fMRI

**Noise Modeling**
- Ellis et al. (2020): BrainIAK fmrisim validation
- Burock & Dale (2000): Temporally correlated noise
- Worsley & Friston (1995): AR(1) + drift models

**Design Optimization**
- Buracas & Boynton (2002): M-sequences for fMRI
- Wager & Nichols (2003): Genetic algorithm optimization
- Friston et al. (1999): Stochastic designs

**HRF Estimation**
- Lindquist et al. (2009): HRF estimation via basis functions
- Pedregosa et al. (2015): HRF estimation methods comparison
- Glover (1999): Deconvolution of impulse response

---

## Implementation Timeline

### Sprint 1 (Weeks 1-2): Core Metrics
- [ ] Implement estimation efficiency calculations
- [ ] Implement detection power calculations
- [ ] Implement conditional entropy
- [ ] Unit tests for all metrics
- [ ] Replicate Liu & Frank Figure 1

### Sprint 2 (Weeks 3-4): Design Optimization
- [ ] ISI optimization with truncated Poisson
- [ ] M-sequence generation
- [ ] Permuted block designs
- [ ] Mixed designs
- [ ] Design evaluation workflow

### Sprint 3 (Weeks 5-6): Advanced Noise
- [ ] AR(n) noise generation (GPU-accelerated)
- [ ] Multi-component noise mixing
- [ ] Parameter estimation from real data
- [ ] SNR/SFNR control
- [ ] Spatial correlation

### Sprint 4 (Weeks 7-8): HRF Recovery
- [ ] HRF recovery metrics
- [ ] Simulation-based power analysis
- [ ] Efficiency vs estimation trade-off study
- [ ] Validation with ground truth data

### Sprint 5 (Weeks 9-10): Integration
- [ ] Complete ExperimentalDesigner class
- [ ] Integration with existing modules
- [ ] Example scripts
- [ ] Documentation
- [ ] Performance optimization

---

## Success Criteria

### Technical
- [ ] Replicate Liu & Frank (2004) Figures 1-4 quantitatively
- [ ] Match BrainIAK fmrisim noise characteristics (SFNR, AR, FWHM)
- [ ] Demonstrate HRF recovery >0.8 correlation with ground truth
- [ ] Design evaluation <100ms per design on GPU
- [ ] Pass all unit and integration tests

### Scientific
- [ ] Tool helps users make informed design choices
- [ ] Clear visualization of efficiency-power-entropy trade-offs
- [ ] Realistic noise models match empirical data
- [ ] Validated on published experimental designs
- [ ] Demonstrate superior designs vs naive approaches

### Usability
- [ ] Clean API matching existing fastfuncsim style
- [ ] Comprehensive examples for common use cases
- [ ] Clear documentation with equations from papers
- [ ] Helpful error messages and warnings
- [ ] Export to common experiment software

---

## Notes for Future Me

### Key Insights to Remember

1. **Efficiency ≠ Power**: You cannot maximize both. Understand user's goal.

2. **Entropy Matters**: High entropy prevents habituation but often trades off with power.

3. **Noise Structure**: Real fMRI noise is AR + physiological + drift. Model all components.

4. **Design Space is Huge**: Need smart search, not exhaustive. Use theoretical guidance.

5. **Ground Truth Validation**: Simulate → Estimate → Compare. This is how we know it works.

6. **Balance is Key**: Most experiments need intermediate trade-offs, not extremes.

### Implementation Priorities

**Must Have (v0.2)**:
- Efficiency and power calculations (Liu & Frank core)
- AR(n) noise (BrainIAK style)
- M-sequence designs
- Basic design evaluation

**Should Have (v0.3)**:
- Full design optimization workflow
- Noise parameter estimation from real data
- HRF recovery validation
- Pareto optimization

**Nice to Have (v0.4+)**:
- GLMdenoise + ridge regression
- Nonlinear models
- Real-time adaptation
- Advanced spatial models

### Questions to Resolve

1. **How to balance speed vs accuracy in design search?**
   - Use theoretical bounds to prune search space
   - GPU-accelerate candidate evaluation
   - Multi-stage coarse-to-fine search

2. **How to handle overlapping stimuli?**
   - Current model assumes non-overlapping
   - Need extension for overlapping/semi-overlapping
   - Important for rapid event-related designs

3. **How to incorporate basis functions into efficiency?**
   - Gain factor: (k/s)² theoretical, but often less in practice
   - Need empirical validation
   - Depends on how well basis spans true HRF space

4. **How much do AR parameters vary across subjects/regions?**
   - High variance could limit utility of group-level estimates
   - May need conservative "worst case" assumptions
   - Or: design robust across parameter ranges

---

## Getting Started (Right Now)

### Immediate Next Steps

1. **Create metrics.py stub with key functions**
2. **Implement estimation efficiency calculation** (start simple, no basis functions)
3. **Test on synthetic design matrices** (block vs random)
4. **Verify matches Liu & Frank equations**
5. **Move to detection power**

### First Validation

Replicate Liu & Frank Figure 1 for Q=2:
- Block designs (1, 2, 4-block, etc.)
- Random designs
- M-sequence design
- Plot efficiency vs power with theoretical curves

This will prove the core math is right before building more.

---

**Remember**: This is a balancing act between efficiency (estimate HRF shape), power (detect activation), and entropy (avoid confounds). The tool must help users navigate this space intelligently, respecting real noise properties and practical constraints.

**Let's build something that actually helps design better experiments!** 🚀

