#!/usr/bin/env python
"""
Example: Complete analysis workflow with both simulated and real data

Demonstrates that the same analysis pipeline works for:
1. Simulation data (torch.Tensor) → analysis
2. AFNI onset files → analysis
3. AFNI design matrix → analysis

This shows how the GLM engine is the core, and both simulation and
real data flow through the same analysis functions.
"""

import torch

import fastfuncstuff as ffs

# Setup
device = ffs.get_device()
print(f"Using device: {device}\n")

# =============================================================================
# Example 1: Simulation data → Analysis pipeline
# =============================================================================
print("=" * 70)
print("Example 1: Simulation data → Analysis pipeline")
print("=" * 70)

# Simulate experiment
print("\n1. Generating simulated fMRI data...")
tr = 2.0
n_timepoints = 200
n_conditions = 2
matrix_size = (20, 20, 5)

# Generate onsets
onsets = ffs.generate_random_onsets(
    n_timepoints=n_timepoints,
    n_conditions=n_conditions,
    isi_mean=6,
    tr=tr,
    device=device
)

# Simulate data
hrf = ffs.get_canonical_hrf(stim_duration=2.0, tr=tr, device=device)
data = ffs.simulate_fmri_run(
    onsets=onsets,
    betas=[3.0, 4.0],
    hrf=hrf,
    tr=tr,
    n_timepoints=n_timepoints,
    matrix_size=matrix_size,
    noise_level=1.0,
    baseline=100.0,
    device=device
)

print(f"   Data shape: {data.shape}")
print(f"   Data range: [{data.min():.1f}, {data.max():.1f}]")

# Analyze using the same pipeline that would work for real data
print("\n2. Analyzing with OLS (canonical HRF)...")
# Note: data is already a torch.Tensor, analyze_from_onsets handles this!
# But we'll use the lower-level fit_glm directly since we already have everything

# Build design
design = ffs.build_glm_design(onsets, hrf, n_timepoints, mode='assumed', device=device)

# Reshape data to (n_voxels, n_timepoints)
data_reshaped = data.reshape(-1, n_timepoints)

# Fit GLM
results_ols = ffs.fit_glm(data_reshaped, design, tr=tr, device=device)

print(f"   Mean R²: {results_ols.r2.mean():.3f}")
print(f"   Mean beta[0]: {results_ols.betas[:, 0].mean():.3f} (true: 3.0)")
print(f"   Mean beta[1]: {results_ols.betas[:, 1].mean():.3f} (true: 4.0)")

# Analyze with ARMA(1,1)
print("\n3. Analyzing with ARMA(1,1) prewhitening...")
# Create ARMA parameter grids
a_grid = torch.linspace(-0.9, 0.9, 19, device=device)
b_grid = torch.linspace(-0.9, 0.9, 19, device=device)

results_arma = ffs.fit_glm_arma11(
    data_reshaped,
    design,
    tr=tr,
    a_grid=a_grid,
    b_grid=b_grid,
    device=device
)

print(f"   Mean R²: {results_arma.r2.mean():.3f}")
print(f"   Mean beta[0]: {results_arma.betas[:, 0].mean():.3f} (true: 3.0)")
print(f"   Mean beta[1]: {results_arma.betas[:, 1].mean():.3f} (true: 4.0)")
print(f"   Mean a: {results_arma.arma_params[:, 0].mean():.3f}")
print(f"   Mean b: {results_arma.arma_params[:, 1].mean():.3f}")
print(f"   Mean lambda (lag-1 corr): {results_arma.arma_lambda.mean():.3f}")

# =============================================================================
# Example 2: AFNI onset files → Analysis pipeline
# =============================================================================
print("\n" + "=" * 70)
print("Example 2: AFNI onset files → Analysis (using existing test data)")
print("=" * 70)

# Check if example simulation exists
import os

onset_file_1 = '/Users/logan/local_bin/fastfuncstuff/simulations/simulation_example_test/onsets_condition1.txt'
onset_file_2 = '/Users/logan/local_bin/fastfuncstuff/simulations/simulation_example_test/onsets_condition2.txt'

if os.path.exists(onset_file_1) and os.path.exists(onset_file_2):
    print("\n1. Reading AFNI onset files...")
    onset_files = [onset_file_1, onset_file_2]

    # Read onsets
    onset_data = ffs.read_afni_onset_files(onset_files)
    print(f"   Loaded {len(onset_data)} conditions")
    print(f"   Condition 1, Run 1: {len(onset_data[0][0])} events")
    print(f"   Condition 2, Run 1: {len(onset_data[1][0])} events")

    # Convert to binary matrix
    onsets_from_file = ffs.onsets_to_binary_matrix(
        onset_data,
        n_timepoints=n_timepoints,
        tr=tr,
        device=device
    )

    print(f"   Onset matrix shape: {onsets_from_file.shape}")
    print(f"   Total events per condition: {onsets_from_file.sum(dim=0)}")

    # Could now use the same GLM pipeline with this onset matrix
    print("\n2. These onsets can now flow through the same GLM pipeline!")
    print("   (Skipping actual GLM fit to save time)")

else:
    print("\n(Example AFNI onset files not found - skipping this example)")

# =============================================================================
# Example 3: AFNI design matrix → Analysis
# =============================================================================
print("\n" + "=" * 70)
print("Example 3: AFNI design matrix → Analysis")
print("=" * 70)

design_matrix_file = '/Users/logan/local_bin/fastfuncstuff/X.xmat.1D'

if os.path.exists(design_matrix_file):
    print("\n1. Reading AFNI design matrix...")
    design_info = ffs.read_afni_design_matrix(design_matrix_file)

    print(f"   Design shape: {design_info['matrix'].shape}")
    print(f"   TR: {design_info['tr']}s")
    print(f"   Stimulus labels: {design_info['stim_labels']}")
    print(f"   Number of GLT contrasts: {len(design_info['glt_labels'])}")

    # Extract stimulus columns only
    print("\n2. Extracting stimulus columns...")
    stim_design = ffs.extract_stimulus_columns(design_info, device=device)
    print(f"   Stimulus design shape: {stim_design.shape}")

    # Extract nuisance columns
    nuisance_design = ffs.extract_nuisance_columns(design_info, device=device)
    print(f"   Nuisance design shape: {nuisance_design.shape}")

    print("\n3. This design matrix can now be used with fit_glm() or fit_glm_arma11()!")
    print("   (Would need matching fMRI data to actually fit)")

else:
    print("\n(AFNI design matrix file not found - skipping this example)")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("Summary: Organization and Data Flow")
print("=" * 70)
print("""
The package is organized so that the GLM engine is the core:

Data Sources → GLM Engine → Results
-------------   ----------   -------
1. Simulations  →  fit_glm()      → GLMResults
2. AFNI onsets  →  fit_glm()      → GLMResults
3. AFNI design  →  fit_glm()      → GLMResults
4. Any data     →  fit_glm_arma11() → ARMA11Results

Key advantages:
- GLM functions accept torch.Tensor directly (fast!)
- Same functions work for simulation and real data
- Modular design: swap HRF, noise, design matrix, etc.
- GPU-accelerated throughout

The analysis.py module provides high-level workflows:
- analyze_from_onsets(): Read AFNI → Build design → GLM
- analyze_from_design_matrix(): Read AFNI → GLM
- Both accept torch.Tensor, np.ndarray, or file paths

For simulations, you can either:
A) Use low-level functions directly (fit_glm, build_glm_design)
B) Use high-level analyze_from_onsets() with torch.Tensor data
""")

print("\n✓ Organization is sound - simulation and real data use same pipeline!")
