# FastFuncSim Visualization Guide

Comprehensive guide to visualization tools for both **single-case deep exploration** and **batch simulation summaries**.

## Philosophy

FastFuncSim provides two complementary visualization approaches:

1. **Single-Case Deep Dive**: Detailed exploration of individual simulations to understand model performance, residuals, and parameter estimates
2. **Batch Summary**: Statistical summaries across hundreds/thousands of simulations for power analysis and design optimization

## Core Visualization Functions

### 1. `plot_simulation_deep_dive()`

**Purpose**: Deep exploration of a single simulation

**When to use**:
- Understanding how well the model fits individual voxels
- Diagnosing issues with design or noise
- Comparing estimated vs true parameters
- Visualizing residuals and model quality

**What it shows**:
- Observed vs predicted timecourses for selected voxels
- Residuals over time
- Beta estimates vs true values (if provided)
- Design matrix heatmap
- R² distribution across all voxels
- Summary statistics

**Example**:
```python
import fastfuncsim as ffs

# After running a simulation
fig = ffs.plot_simulation_deep_dive(
    data=data,
    design=design,
    results=results,
    betas_true=true_betas,
    hrf_true=true_hrf,
    voxel_selection='best',  # or 'worst', 'median', 'random'
    n_voxels=4,
    tr=1.0,
    save_path='deep_dive.png'
)
```

**Voxel Selection Options**:
- `'best'`: Highest R² voxels (default) - shows where model performs best
- `'worst'`: Lowest R² voxels - useful for diagnosing problems
- `'median'`: Middle R² voxels - shows typical performance
- `'random'`: Random selection - unbiased sample

---

### 2. `plot_batch_summary()`

**Purpose**: Statistical summary across multiple simulations

**When to use**:
- Power analysis across different effect sizes
- Comparing design efficiency
- Understanding variability in estimates
- Statistical inference across parameter space

**What it shows**:
- R² distributions (overall or grouped)
- Beta estimation error (MAE, RMSE)
- HRF recovery quality (correlation with true HRF)
- Statistical power curves
- Beta estimation variance

**Example**:
```python
# After running 100+ simulations
results_list = [
    {'r2_mean': 0.45, 'beta_error_mae': 0.12, 'effect_size': 2.0, ...},
    {'r2_mean': 0.52, 'beta_error_mae': 0.10, 'effect_size': 3.0, ...},
    # ... more simulations
]

fig = ffs.plot_batch_summary(
    results_list=results_list,
    metrics=['r2', 'beta_error', 'hrf_recovery', 'power'],
    group_by='effect_size',  # or 'noise_level', 'hrf_type', etc.
    save_path='batch_summary.png'
)
```

**Metrics Available**:
- `'r2'`: R² distribution and statistics
- `'beta_error'`: MAE and RMSE of beta estimates
- `'hrf_recovery'`: Correlation with true HRF
- `'power'`: Statistical power curves (if effect_size in results)

**Grouping**: Any variable in result dicts can be used for grouping (e.g., `effect_size`, `noise_level`, `design_type`)

---

### 3. `plot_parametric_exploration()`

**Purpose**: Visualize results across 3-axis parameter space

**When to use**:
- Testing combinations of experimental parameters
- Understanding interactions between variables
- Systematic exploration of design space

**3-Axis Structure**:
1. **X-axis**: Often beta ratios or effect sizes (condition A vs B)
2. **Y-axis**: Often HRF variations (different HRF shapes)
3. **Z-axis** (subplots): Often noise levels (different SNR)

**What it shows**:
- Heatmaps of any metric across parameter space
- One subplot per Z-axis value
- Color-coded performance
- Numerical annotations

**Example**:
```python
# results_grid structure: {z_val: {y_val: {x_val: result_dict}}}
results_grid = {
    0.5: {  # noise_level = 0.5
        0: {0: {'r2_mean': 0.65}, 1: {'r2_mean': 0.62}, ...},  # hrf_index = 0
        1: {0: {'r2_mean': 0.58}, 1: {'r2_mean': 0.60}, ...},  # hrf_index = 1
        # ... more HRFs
    },
    1.0: {  # noise_level = 1.0
        # ... same structure
    },
    # ... more noise levels
}

fig = ffs.plot_parametric_exploration(
    results_grid=results_grid,
    x_var='beta_ratio',
    y_var='hrf_index',
    z_var='noise_level',
    metric='r2_mean',
    save_path='parametric_r2.png'
)
```

**Flexibility**: Works with **any number of conditions**, **any ordering**, **any magnitude** of effects. Just organize results into the nested dictionary structure.

---

### 4. `plot_hrf_recovery()`

**Purpose**: Evaluate HRF estimation quality from FIR models

**When to use**:
- Testing FIR estimation accuracy
- Comparing estimated HRF shapes to ground truth
- Understanding HRF recovery across noise/design variations

**What it shows**:
- Individual voxel HRF estimates vs true HRF
- Correlation with true HRF per voxel
- RMSE between estimated and true
- Mean estimated HRF ± std across voxels
- Peak timing errors
- Summary statistics

**Example**:
```python
# After FIR estimation
design_fir = ffs.build_glm_design(onsets, mode='fir', n_fir_lags=30)
results_fir = ffs.fit_glm(data, design_fir, tr=1.0)

# Extract FIR estimates for condition 1
fir_estimates = results_fir.betas[:, :30]  # First 30 lags

fig = ffs.plot_hrf_recovery(
    hrf_estimated=fir_estimates,
    hrf_true=true_hrf,
    tr=1.0,
    voxel_selection='best',
    n_voxels=6,
    save_path='hrf_recovery.png'
)
```

**HRF Recovery Metrics**:
- **Correlation**: Shape similarity (-1 to 1, higher better)
- **RMSE**: Amplitude error (lower better)
- **Peak timing error**: Difference in time-to-peak (seconds)

---

### 5. `plot_design_comparison()`

**Purpose**: Compare multiple design matrices side-by-side

**When to use**:
- Comparing different experimental designs
- Visualizing FIR vs assumed HRF designs
- Understanding design matrix structure

**What it shows**:
- Timecourses for each regressor
- Heatmaps of design matrices
- Side-by-side comparison

**Example**:
```python
designs = {
    'Block Design': design_block,
    'Event-Related': design_er,
    'Mixed Design': design_mixed,
}

fig = ffs.plot_design_comparison(
    designs=designs,
    labels=['Condition A', 'Condition B'],
    tr=1.0,
    save_path='design_comparison.png'
)
```

---

### 6. `create_interactive_summary_html()`

**Purpose**: Generate interactive HTML report for batch simulations

**When to use**:
- Sharing results with collaborators
- Exploring large numbers of simulations
- Creating permanent records of analyses

**What it generates**:
- Interactive HTML file with sortable table
- Summary statistics
- All simulation results in searchable format

**Example**:
```python
html_path = ffs.create_interactive_summary_html(
    results_list=results_list,
    output_path='simulation_summary.html'
)
# Open simulation_summary.html in web browser
```

---

## Complete Workflow Examples

### Example 1: Single Simulation Exploration

```python
import fastfuncsim as ffs
import torch

# Setup
device = ffs.get_device()
hrf = ffs.get_canonical_hrf(stim_duration=5.0, tr=1.0, device=device)
onsets = ffs.generate_random_onsets(290, n_conditions=2, isi_mean=4, tr=1.0)

# Simulate
betas_true = [5.0, 3.0]
data = ffs.simulate_fmri_run(onsets, betas=betas_true, hrf=hrf, tr=1.0,
                             n_timepoints=290, matrix_size=(50, 50, 5))

# Fit GLM
design = ffs.build_glm_design(onsets, hrf, 290, mode='assumed')
results = ffs.fit_glm(data, design, tr=1.0)

# Deep dive visualization
fig = ffs.plot_simulation_deep_dive(
    data=data, design=design, results=results,
    betas_true=torch.tensor(betas_true), hrf_true=hrf,
    voxel_selection='best', n_voxels=4, tr=1.0,
    save_path='single_sim_deep_dive.png'
)

print(f"Mean R² = {results.r2.mean():.3f}")
```

---

### Example 2: Batch Power Analysis

```python
# Run 100 simulations across effect sizes
effect_sizes = [0.5, 1.0, 2.0, 3.0, 5.0]
results_list = []

for es in effect_sizes:
    for sim in range(100):
        # Simulate and fit
        data = ffs.simulate_fmri_run(onsets, betas=[es, es], ...)
        results = ffs.fit_glm(data, design, tr=1.0)

        # Store results
        results_list.append({
            'r2_mean': results.r2.mean().item(),
            'r2_median': results.r2.median().item(),
            'effect_size': es,
        })

# Batch summary
fig = ffs.plot_batch_summary(
    results_list=results_list,
    metrics=['r2', 'power'],
    group_by='effect_size',
    save_path='power_analysis.png'
)

# Interactive HTML
ffs.create_interactive_summary_html(results_list, 'power_analysis.html')
```

---

### Example 3: 3-Axis Parametric Exploration

```python
# Define parameter space
beta_ratios = [(1, 1), (2, 1), (3, 1)]  # Condition A vs B
hrf_library = ffs.get_canonical_hrf_library(stim_duration=5.0, tr=1.0, n_hrfs=5)
noise_levels = [0.5, 1.0, 2.0]

# Storage
results_grid = {}
results_list = []

# Run all combinations
for noise in noise_levels:
    results_grid[noise] = {}
    for hrf_idx in range(len(hrf_library)):
        results_grid[noise][hrf_idx] = {}
        for beta_idx, (beta1, beta2) in enumerate(beta_ratios):
            # Simulate
            data = ffs.simulate_fmri_run(
                onsets, betas=[beta1, beta2], hrf=hrf_library[hrf_idx],
                noise_level=noise, ...
            )
            results = ffs.fit_glm(data, design, tr=1.0)

            # Store
            result_dict = {
                'r2_mean': results.r2.mean().item(),
                'beta_error_mae': compute_beta_error(results, [beta1, beta2]),
                'noise_level': noise,
                'hrf_index': hrf_idx,
                'beta_ratio': beta_idx,
            }
            results_grid[noise][hrf_idx][beta_idx] = result_dict
            results_list.append(result_dict)

# Parametric visualization
fig_param = ffs.plot_parametric_exploration(
    results_grid=results_grid,
    x_var='beta_ratio',
    y_var='hrf_index',
    z_var='noise_level',
    metric='r2_mean',
    save_path='parametric_r2.png'
)

# Batch summary
fig_batch = ffs.plot_batch_summary(
    results_list=results_list,
    metrics=['r2', 'beta_error'],
    group_by='noise_level',
    save_path='batch_summary.png'
)
```

---

### Example 4: HRF Recovery Analysis

```python
# Simulate with known HRF
true_hrf = ffs.get_canonical_hrf(stim_duration=5.0, tr=1.0)
data = ffs.simulate_fmri_run(onsets, betas=[5, 5], hrf=true_hrf, ...)

# Estimate HRF using FIR
design_fir = ffs.build_glm_design(onsets, mode='fir', n_fir_lags=30)
results_fir = ffs.fit_glm(data, design_fir, tr=1.0)

# Extract FIR estimates
fir_estimates = results_fir.betas[:, :30]  # Condition 1

# Visualize recovery
fig = ffs.plot_hrf_recovery(
    hrf_estimated=fir_estimates,
    hrf_true=true_hrf[:30],
    tr=1.0,
    voxel_selection='best',
    n_voxels=6,
    save_path='hrf_recovery.png'
)
```

---

## Tips and Best Practices

### Single-Case Visualization

1. **Start with 'best' voxels** to see optimal performance, then check 'worst' to diagnose issues
2. **Look at residuals** - should be white noise. Patterns indicate model misspecification
3. **Compare estimated vs true betas** - systematic bias indicates design problems
4. **Check R² distribution** - bimodal distributions may indicate distinct voxel populations

### Batch Visualization

1. **Use grouping** to separate effects of different parameters
2. **Check power curves** - need 80% power to detect effects reliably
3. **Examine beta estimation variance** - high variance suggests need for more trials
4. **Compare across noise levels** - ensures robustness to real-world SNR

### Parametric Exploration

1. **Start with coarse grid**, then refine around interesting regions
2. **Use heatmaps** to quickly identify parameter combinations that work well
3. **Look for interactions** - e.g., HRF misspecification worse at high noise
4. **Document parameter space** - save the grid structure for reproducibility

### HRF Recovery

1. **Correlation > 0.8** indicates good shape recovery
2. **Peak timing error < 1 TR** is excellent
3. **Mean ± std plot** shows consistency across voxels
4. **Worst voxels** reveal where FIR estimation fails (low SNR, poor design)

---

## Customization

All visualization functions accept matplotlib-compatible parameters:

```python
fig = ffs.plot_simulation_deep_dive(
    ...,
    figsize=(20, 15),     # Larger figure
    save_path='custom.png'
)

# Further customize
fig.axes[0].set_title('Custom Title')
fig.savefig('custom_modified.png', dpi=300, bbox_inches='tight')
```

---

## Troubleshooting

**Q: Plots look cluttered with many voxels/conditions**
A: Reduce `n_voxels` parameter or increase `figsize`

**Q: Can't see differences in parametric heatmaps**
A: Adjust colormap limits or use different metrics (R² vs beta error)

**Q: Batch summary shows no grouping effect**
A: Check that grouping variable is in result dicts and varies across simulations

**Q: HRF recovery shows poor correlation despite good simulation**
A: Ensure FIR design has enough lags (>30) and sufficient trials per condition

**Q: HTML summary is huge**
A: Sample results list or create separate HTMLs per parameter group

---

## Advanced: Custom Visualizations

All plot functions return matplotlib Figure objects, enabling custom modifications:

```python
import matplotlib.pyplot as plt

# Create base visualization
fig = ffs.plot_simulation_deep_dive(...)

# Add custom annotations
axes = fig.get_axes()
axes[0].axvline(x=100, color='red', linestyle='--', label='Event onset')
axes[0].legend()

# Adjust layout
fig.tight_layout()
fig.savefig('custom_deep_dive.png', dpi=300)
```

---

## See Also

- `examples/example_single.py` - Single simulation with deep dive visualization
- `examples/example_batch.py` - Batch power analysis with summaries
- `example_parametric.py` - Full 3-axis parametric exploration
- `IMPLEMENTATION_PLAN.md` - Future metrics and efficiency visualizations

---

**FastFuncSim Visualization**: From single voxels to thousands of simulations 🚀
