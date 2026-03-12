"""
Example: Design Space Exploration with Liu & Frank Metrics

Demonstrates the flexible design optimization workflow:
1. Define ISI constraints (min, max, mean)
2. Sample parameter space with different orderings and distributions
3. Evaluate designs using empirical metrics (AR(1)-corrected)
4. Identify optimal designs
5. Visualize fitness landscape and Pareto frontier

This workflow works with ANY design structure:
- Random event-related
- Alternating (A-B-A-B...)
- Blocked (AAAA-BBBB...)
- Permuted block (randomized mini-blocks)

And ANY ISI distribution:
- Exponential (most common)
- Poisson
- Uniform
- Fixed

Author: FastFuncSim
Date: 2024
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Import FastFuncSim modules
from fastfuncsim.design.optimization import (
    ISIConstraints,
    compare_designs_summary,
    evaluate_design_candidates,
    find_optimal_designs,
    plot_fitness_landscape,
    plot_pareto_frontier,
    sample_design_space,
)

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Check device
device = torch.device('mps' if torch.backends.mps.is_available() else
                     'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print("=" * 80)

# ============================================================================
# 1. Define Experimental Design Parameters
# ============================================================================

print("\n1. DEFINING EXPERIMENTAL PARAMETERS")
print("-" * 80)

# Basic parameters
n_conditions = 2  # e.g., A vs B
n_trials_per_condition = 30  # 30 trials each
duration = 300.0  # 5-minute scan
tr = 1.0  # 1-second TR

# ISI constraints
# These define the allowable inter-stimulus intervals
isi_constraints = ISIConstraints(
    min_isi=2.0,   # Minimum 2 seconds between events
    max_isi=10.0,  # Maximum 10 seconds between events
    mean_isi=5.0,  # Target mean of 5 seconds
    tr=tr
)

print(f"N conditions: {n_conditions}")
print(f"N trials per condition: {n_trials_per_condition}")
print(f"Scan duration: {duration} s")
print(f"TR: {tr} s")
print(f"ISI constraints: min={isi_constraints.min_isi}s, "
      f"max={isi_constraints.max_isi}s, mean={isi_constraints.mean_isi}s")

# ============================================================================
# 2. Sample Design Space
# ============================================================================

print("\n2. SAMPLING DESIGN SPACE")
print("-" * 80)

# Explore multiple orderings and distributions
# This will create: 3 orderings × 2 distributions × 10 samples = 60 candidate designs
candidates = sample_design_space(
    n_conditions=n_conditions,
    n_trials_per_condition=n_trials_per_condition,
    duration=duration,
    isi_constraints=isi_constraints,
    n_samples=10,  # 10 samples per combination
    event_orderings=['random', 'alternating', 'permuted_block'],  # Try different orderings
    isi_distributions=['exponential', 'uniform'],  # Try different ISI distributions
    hrf_type='spm',  # Use SPM canonical HRF
    device=device,
    seed=SEED
)

print(f"\nGenerated {len(candidates)} candidate designs")

# Quick check: show first candidate
first_candidate = candidates[0]
print("\nExample candidate (first design):")
print(f"  Ordering: {first_candidate.metadata['ordering']}")
print(f"  Distribution: {first_candidate.metadata['distribution']}")
print(f"  Total events: {first_candidate.metadata['total_events']}")
print(f"  ISI mean: {first_candidate.metadata['isi_mean_actual']:.2f} s")
print(f"  ISI std: {first_candidate.metadata['isi_std']:.2f} s")
print(f"  Onset matrix shape: {first_candidate.onsets.shape}")
print(f"  Design matrix shape: {first_candidate.design_matrix.shape}")

# ============================================================================
# 3. Evaluate Designs with Empirical Metrics
# ============================================================================

print("\n3. EVALUATING DESIGNS WITH EMPIRICAL METRICS")
print("-" * 80)
print("Computing detection power and estimation efficiency...")
print("(This uses GLS with AR(1) correction as in Das et al. 2023)")

# Evaluate all candidates
# Since we don't have real data, we'll simulate simple data for each design
candidates = evaluate_design_candidates(
    candidates=candidates,
    data=None,  # Will simulate data
    hrf_length=30,  # 30 TRs for FIR estimation
    effect_sizes=[1.0, 1.0],  # Equal effect sizes for both conditions
    noise_level=1.0,  # Moderate noise
    n_voxels=100,  # 100 voxels for evaluation
    device=device,
    verbose=True
)

print("\nEvaluation complete!")

# Check results
n_valid = sum(1 for c in candidates if c.metrics is not None and
              not np.isnan(c.metrics.get('detection_power', np.nan)))
print(f"Successfully evaluated {n_valid}/{len(candidates)} designs")

# ============================================================================
# 4. Identify Optimal Designs
# ============================================================================

print("\n4. IDENTIFYING OPTIMAL DESIGNS")
print("-" * 80)

# Find top designs for different objectives
print("\nTop designs for DETECTION POWER:")
summary_power = compare_designs_summary(
    candidates=candidates,
    top_k=5,
    objective='power',
    alpha=1.0  # 100% weight on power
)
print(summary_power)

print("\nTop designs for ESTIMATION EFFICIENCY:")
summary_efficiency = compare_designs_summary(
    candidates=candidates,
    top_k=5,
    objective='efficiency',
    alpha=0.0  # 100% weight on efficiency
)
print(summary_efficiency)

print("\nTop designs for BALANCED (50/50):")
summary_balanced = compare_designs_summary(
    candidates=candidates,
    top_k=5,
    objective='balanced',
    alpha=0.5  # Equal weight
)
print(summary_balanced)

# ============================================================================
# 5. Visualize Fitness Landscape
# ============================================================================

print("\n5. VISUALIZING FITNESS LANDSCAPE")
print("-" * 80)

# Create output directory
output_dir = Path("design_optimization_results")
output_dir.mkdir(exist_ok=True)

# Plot fitness landscape
print("Creating fitness landscape plot...")
fig_landscape = plot_fitness_landscape(
    candidates=candidates,
    figsize=(14, 6),
    save_path=output_dir / "fitness_landscape.png"
)
plt.close(fig_landscape)

# Plot Pareto frontier
print("Creating Pareto frontier plot...")
fig_pareto, pareto_indices = plot_pareto_frontier(
    candidates=candidates,
    figsize=(10, 7),
    save_path=output_dir / "pareto_frontier.png"
)
plt.close(fig_pareto)

# ============================================================================
# 6. Analyze Pareto Optimal Designs
# ============================================================================

print("\n6. ANALYZING PARETO OPTIMAL DESIGNS")
print("-" * 80)

pareto_designs = [candidates[idx] for idx in pareto_indices]

print(f"\nFound {len(pareto_designs)} Pareto optimal designs:")
for i, design in enumerate(pareto_designs, 1):
    print(f"\n  Pareto Design {i}:")
    print(f"    Ordering: {design.metadata['ordering']}")
    print(f"    ISI Distribution: {design.metadata['distribution']}")
    print(f"    Detection Power: {design.metrics['detection_power']:.4f}")
    print(f"    Estimation Efficiency: {design.metrics['estimation_efficiency']:.4f}")
    print(f"    AR(1) coefficient: {design.metrics.get('rho', np.nan):.3f}")

# ============================================================================
# 7. Compare Specific Design Types
# ============================================================================

print("\n7. COMPARING DESIGN TYPES")
print("-" * 80)

# Group by ordering
orderings = {}
for candidate in candidates:
    if candidate.metrics is None:
        continue
    ordering = candidate.metadata['ordering']
    if ordering not in orderings:
        orderings[ordering] = {'power': [], 'efficiency': []}

    orderings[ordering]['power'].append(candidate.metrics['detection_power'])
    orderings[ordering]['efficiency'].append(candidate.metrics['estimation_efficiency'])

print("\nAverage metrics by ordering:")
for ordering, metrics in orderings.items():
    mean_power = np.mean(metrics['power'])
    mean_eff = np.mean(metrics['efficiency'])
    print(f"  {ordering:20s}: Power={mean_power:.4f}, Efficiency={mean_eff:.4f}")

# Group by ISI distribution
distributions = {}
for candidate in candidates:
    if candidate.metrics is None:
        continue
    dist = candidate.metadata['distribution']
    if dist not in distributions:
        distributions[dist] = {'power': [], 'efficiency': []}

    distributions[dist]['power'].append(candidate.metrics['detection_power'])
    distributions[dist]['efficiency'].append(candidate.metrics['estimation_efficiency'])

print("\nAverage metrics by ISI distribution:")
for dist, metrics in distributions.items():
    mean_power = np.mean(metrics['power'])
    mean_eff = np.mean(metrics['efficiency'])
    print(f"  {dist:20s}: Power={mean_power:.4f}, Efficiency={mean_eff:.4f}")

# ============================================================================
# 8. Export Best Design for Use
# ============================================================================

print("\n8. EXPORTING BEST DESIGN")
print("-" * 80)

# Get best balanced design
best_designs = find_optimal_designs(
    candidates=candidates,
    objective='balanced',
    alpha=0.5,
    top_k=1
)

if best_designs:
    best_idx, best_candidate, best_score = best_designs[0]

    print(f"\nBest design (index {best_idx}):")
    print(f"  Ordering: {best_candidate.metadata['ordering']}")
    print(f"  ISI Distribution: {best_candidate.metadata['distribution']}")
    print(f"  Detection Power: {best_candidate.metrics['detection_power']:.4f}")
    print(f"  Estimation Efficiency: {best_candidate.metrics['estimation_efficiency']:.4f}")
    print(f"  Score: {best_score:.4f}")

    # Extract onset times for export
    onsets = best_candidate.onsets.cpu().numpy()
    n_timepoints, n_conditions = onsets.shape

    # Find onset times for each condition
    onset_times = {}
    for cond in range(n_conditions):
        onset_trs = np.where(onsets[:, cond] > 0.5)[0]
        onset_times[cond] = onset_trs * tr

    # Save to TSV (BIDS format)
    output_file = output_dir / "best_design_onsets.tsv"
    with open(output_file, 'w') as f:
        f.write("onset\tduration\ttrial_type\n")
        for cond in range(n_conditions):
            for onset in onset_times[cond]:
                f.write(f"{onset:.2f}\t0\tcondition_{cond}\n")

    print(f"\nSaved onset times to: {output_file}")

    # Visualize best design
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot onset matrix
    ax = axes[0]
    im = ax.imshow(onsets.T, aspect='auto', cmap='binary', interpolation='nearest')
    ax.set_xlabel('Time (TRs)', fontsize=12)
    ax.set_ylabel('Condition', fontsize=12)
    ax.set_title(f'Best Design: {best_candidate.metadata["ordering"]} + '
                f'{best_candidate.metadata["distribution"]} ISI', fontsize=14)
    ax.set_yticks(range(n_conditions))
    ax.set_yticklabels([f'Condition {i}' for i in range(n_conditions)])

    # Plot design matrix (first 150 TRs for visibility)
    ax = axes[1]
    design_matrix = best_candidate.design_matrix.cpu().numpy()
    n_show = min(150, n_timepoints)
    ax.plot(np.arange(n_show) * tr, design_matrix[:n_show, :])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('BOLD Response (AU)', fontsize=12)
    ax.set_title('Design Matrix (HRF-Convolved)', fontsize=14)
    ax.legend([f'Condition {i}' for i in range(n_conditions)])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "best_design_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to: {output_dir / 'best_design_visualization.png'}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("DESIGN OPTIMIZATION COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {output_dir}/")
print("\nGenerated files:")
print("  - fitness_landscape.png: Power vs Efficiency scatter plots")
print("  - pareto_frontier.png: Pareto optimal designs")
print("  - best_design_onsets.tsv: Onset times for best design (BIDS format)")
print("  - best_design_visualization.png: Visualization of best design")
print("\nNext steps:")
print("  1. Review fitness landscape to understand trade-offs")
print("  2. Choose design based on your experimental priorities:")
print("     - High power → better for detecting small effects")
print("     - High efficiency → better for estimating HRF shape")
print("     - Balanced → compromise between both")
print("  3. Use best_design_onsets.tsv for your experiment")
print("  4. Consider running with real pilot data for more accurate estimates")
print("\n" + "=" * 80)
