# Design Optimization Implementation - Status & Next Steps

**Date**: 2025-10-11
**Status**: Implementation complete, ready for testing

---

## What We Just Built

### 1. Core Modules Created

#### `metrics_empirical.py` ✅
**Purpose**: Real-world design metrics using GLS with AR(1) correction (Das et al. 2023 approach)

**Key Functions**:
- `estimate_ar1_coefficient()`: Extract temporal autocorrelation from residuals
- `build_ar1_covariance_matrix()`: Build Toeplitz covariance Σ[i,j] = ρ^|i-j|
- `gls_fit()`: Generalized Least Squares with Cholesky decomposition
- `compute_detection_power_empirical()`: Fd = 1/trace(C * Var(β) * C')
- `compute_estimation_efficiency_empirical()`: Fe = 1/trace((C⊗I) * Var(β_FIR) * (C⊗I)')
- `evaluate_design_empirical()`: Complete design evaluation

**Why Important**: This uses the ACTUAL implementation from your deconv-master code, not just Liu & Frank theory. It accounts for AR(1) autocorrelation in the metrics themselves.

#### `design_optimization.py` ✅
**Purpose**: Flexible design space exploration tools

**Key Innovation**: Separates WHAT (event ordering) from WHEN (ISI timing)

**Key Functions**:
- `generate_event_sequence()`: Create event ordering
  - 'random': Fully randomized
  - 'alternating': A-B-A-B... (NOT hardcoded to 2 conditions!)
  - 'blocked': AAAA-BBBB...
  - 'permuted_block': Randomized mini-blocks
- `generate_isi_sequence()`: Generate ISI timings
  - 'exponential': Most common for event-related
  - 'poisson': Discrete intervals
  - 'uniform': Uniform random
  - 'fixed': Constant ISI
- `create_onset_matrix()`: Combine event sequence + ISIs → onset matrix
- `sample_design_space()`: Generate candidate designs
- `evaluate_design_candidates()`: Compute metrics for all candidates
- `find_optimal_designs()`: Rank by power/efficiency/balanced
- `compare_designs_summary()`: Text summary of top designs
- `plot_fitness_landscape()`: Power vs Efficiency scatter
- `plot_pareto_frontier()`: Pareto optimal designs

**Flexibility**: Works with ANY number of conditions, ANY design structure, ANY ISI distribution

#### `example_design_optimization.py` ✅
**Purpose**: Complete demonstration workflow

**What It Does**:
1. Define ISI constraints (min=2s, max=10s, mean=5s)
2. Sample 60 candidate designs (3 orderings × 2 distributions × 10 samples)
3. Evaluate each with empirical metrics (AR(1)-corrected)
4. Identify optimal designs for different objectives
5. Visualize fitness landscape and Pareto frontier
6. Export best design to BIDS-format TSV

#### `test_design_opt.py` ✅
**Purpose**: Quick verification that imports work

---

## Package Structure Updates

### Updated Files:
- `__init__.py`: Added exports for design_optimization and metrics_empirical modules
- Both new modules integrated into package API

---

## Current State: READY FOR TESTING

### What's Complete:
✅ Empirical metrics with AR(1) correction (GLS approach)
✅ Flexible event sequence generation (any ordering, any conditions)
✅ ISI distribution sampling (exponential, Poisson, uniform, fixed)
✅ Design space exploration tools
✅ Fitness landscape visualization
✅ Pareto frontier analysis
✅ Complete example script
✅ Package integration

### What's NOT Done Yet:
❌ Package not installed in conda environment
❌ Haven't run tests to verify it works
❌ No setup.py/pyproject.toml for installation

---

## How to Continue (When You Return)

### Step 1: Activate Environment
```bash
conda activate py312_movies_mac
cd /Users/logan/local_bin/fastfuncsim
```

### Step 2: Quick Test (No Installation)
Run test directly by adding parent directory to Python path:
```bash
cd /Users/logan/local_bin/fastfuncsim
PYTHONPATH=/Users/logan/local_bin:$PYTHONPATH python test_design_opt.py
```

Or test imports directly:
```bash
cd /Users/logan/local_bin
python -c "import fastfuncsim; print('Import successful!')"
```

### Step 3: Run Full Example
```bash
cd /Users/logan/local_bin/fastfuncsim
PYTHONPATH=/Users/logan/local_bin:$PYTHONPATH python example_design_optimization.py
```

This will:
- Generate 60 candidate designs
- Evaluate with empirical metrics
- Create visualizations in `design_optimization_results/`
- Export best design to TSV

### Step 4 (Optional): Install Package Properly
If you want to install it properly, we can create a `setup.py`:
```bash
pip install -e /Users/logan/local_bin/fastfuncsim
```

---

## Key Architectural Decisions

### 1. Separation of Event Ordering vs ISI Timing
**Old approach** (too rigid):
- Each condition has its own ISI sequence
- Assumes independent timing per condition
- Hard to implement alternating designs

**New approach** (flexible):
- `generate_event_sequence()`: Decides WHAT happens [0,1,0,1,0,2,...]
- `generate_isi_sequence()`: Decides WHEN (intervals between ALL events)
- `create_onset_matrix()`: Combines them

**Benefits**:
- Works with any design structure (random, alternating, blocked, etc.)
- Any number of conditions (not hardcoded to A-B alternation)
- Matches how real experiments are designed

### 2. Empirical vs Theoretical Metrics
**Theoretical** (`metrics.py` - we created but less important):
- Pure Liu & Frank 2004 formulas
- Assumes independent errors (WRONG for fMRI)
- Fast to compute, good for understanding theory

**Empirical** (`metrics_empirical.py` - MAIN IMPLEMENTATION):
- Uses GLS with AR(1) prewhitening (Das et al. 2023)
- Accounts for temporal autocorrelation
- Matches your deconv-master code
- More realistic for actual fMRI

**Decision**: Use empirical approach as default, keep theoretical for comparison.

### 3. GPU Compatibility
All functions use PyTorch and are GPU-compatible:
- AR(1) estimation: Closed-form (fast)
- GLS fitting: Cholesky decomposition (native PyTorch)
- Design convolution: Already GPU-accelerated
- Evaluation: Batch process across voxels

---

## Your Original Workflow Request

You said:
> "I'm choosing a min and a max ISI, setting a mean, and letting iterations find a poisson or exponential that fits that - then I could sample that parameter space, look at metrics, and decide."

**Implementation**:
```python
# Define constraints
isi_constraints = ISIConstraints(
    min_isi=2.0,
    max_isi=10.0,
    mean_isi=5.0,
    tr=1.0
)

# Sample parameter space
candidates = sample_design_space(
    n_conditions=2,
    n_trials_per_condition=30,
    duration=300.0,
    isi_constraints=isi_constraints,
    n_samples=100,  # 100 samples
    event_orderings=['random', 'alternating'],
    isi_distributions=['exponential', 'poisson']
)

# Evaluate
candidates = evaluate_design_candidates(candidates)

# Look at metrics and decide
summary = compare_designs_summary(candidates, objective='balanced')
print(summary)

# Visualize
plot_fitness_landscape(candidates)
plot_pareto_frontier(candidates)
```

**This is exactly what you asked for!**

---

## Next Development Steps (After Testing)

Once we verify this works:

1. **Add More Metrics** (from Liu & Frank theory):
   - Conditional entropy H_r
   - Efficiency-power trade-off curves
   - Design matrix condition number

2. **Parameter Grid Search**:
   - Systematic exploration of ISI mean/min/max space
   - Multi-dimensional optimization

3. **Real Data Integration**:
   - Extract noise parameters from pilot data
   - Use real scanner AR(1) coefficients
   - Match specific scanner characteristics

4. **Design Constraints**:
   - Maximum run duration
   - Minimum trials per condition
   - Counterbalancing requirements
   - Block randomization within runs

5. **Advanced Optimization**:
   - Genetic algorithms for design search
   - Multi-objective Pareto optimization
   - Constraint satisfaction solvers

---

## Testing Checklist (For When You Return)

- [ ] Verify imports work
- [ ] Test event sequence generation (random, alternating, blocked)
- [ ] Test ISI generation (exponential, Poisson, uniform)
- [ ] Test onset matrix creation
- [ ] Test AR(1) estimation
- [ ] Test GLS fitting
- [ ] Test detection power computation
- [ ] Test estimation efficiency computation
- [ ] Run full example_design_optimization.py
- [ ] Verify output visualizations created
- [ ] Check BIDS TSV export format
- [ ] Test with different numbers of conditions (2, 3, 4, 5)
- [ ] Test with unequal trials per condition

---

## Questions to Answer (After Testing)

1. **Performance**: How long does it take to evaluate 100 designs?
2. **Accuracy**: Do the AR(1) estimates match expectations (ρ ≈ 0.2-0.4)?
3. **Pareto Frontier**: How much trade-off between power and efficiency?
4. **Design Types**: Which ordering (random/alternating/blocked) performs best?
5. **ISI Distributions**: Exponential vs Poisson vs uniform - which wins?

---

## Files to Review When You Return

1. `design_optimization.py` - Main implementation (600+ lines)
2. `metrics_empirical.py` - GLS metrics (400+ lines)
3. `example_design_optimization.py` - Complete demo (300+ lines)
4. `test_design_opt.py` - Quick verification

---

## Contact Points with Existing Code

### Uses from existing modules:
- `design.convolve_hrf()`: Convolve onsets with HRF
- `hrf.canonical_hrf()`: Generate canonical HRF
- `utils.get_device()`: Auto-detect GPU
- `simulation.simulate_fmri_run()`: (optional) for generating test data

### Integrates with:
- `visualization.py`: Could add design comparison plots
- `simulation.py`: Use optimal designs for simulations
- Future `noise.py` AR/ARMA: Use real noise parameters

---

## Summary

**What we built**: Complete flexible design optimization system with empirical metrics that accounts for temporal autocorrelation, supports any design structure, and provides visualization/analysis tools.

**Why it matters**: Enables your exact workflow: set ISI constraints → sample parameter space → evaluate metrics → choose optimal design.

**What's next**: Test it, verify it works, then potentially add more advanced features (real data integration, automated optimization, constraint solvers).

---

**When you return, just say "continue with design optimization testing" and I'll pick up from here!**
