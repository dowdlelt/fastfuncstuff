#!/usr/bin/env python
"""
Simple HRF selection visualization.

Shows mean selected HRF index as a gradient across patterns and ISI.
Since the CVN library is ordered (by time-to-peak), the index is meaningful.
"""
import sys
sys.path.insert(0, '/Users/logan/local_bin/fastfuncsim')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


def plot_hrf_selection_gradient(results_file, output_dir):
    """
    Create simple gradient plot showing mean selected HRF index.
    """
    print("Loading results...")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    isi_means = sorted(results.keys())
    first_isi = isi_means[0]
    pattern_indices = sorted(results[first_isi].keys())

    n_isis = len(isi_means)
    n_patterns = len(pattern_indices)

    # Check if we have selection data
    if 'best_hrf_idx_all' not in results[first_isi][pattern_indices[0]]:
        print("\nERROR: HRF selection data not saved.")
        print("Rerun monte_carlo_comprehensive.py with updated code.")
        return

    print(f"Found {n_isis} ISI conditions and {n_patterns} patterns")

    # Extract mean selected HRF index and true HRF index
    mean_selected = np.zeros((n_isis, n_patterns))
    true_hrf = np.zeros(n_patterns, dtype=int)

    for i, isi in enumerate(isi_means):
        for j, pattern_idx in enumerate(pattern_indices):
            pattern_results = results[isi][pattern_idx]

            # True HRF (same across all ISI for a given pattern)
            if i == 0:
                true_hrf[j] = pattern_results['true_hrf_idx']

            # Mean selected HRF
            selected_hrfs = pattern_results['best_hrf_idx_all']
            mean_selected[i, j] = np.mean(selected_hrfs)

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Panel 1: Mean selected HRF index
    ax = axes[0]
    im = ax.imshow(mean_selected.T, aspect='auto', cmap='turbo',
                   vmin=0, vmax=19, origin='lower')

    ax.set_xticks(np.arange(n_isis))
    ax.set_xticklabels([f"{isi:.2f}" for isi in isi_means])
    ax.set_yticks(np.arange(n_patterns))
    ax.set_yticklabels([f"P{idx}" for idx in pattern_indices])

    ax.set_xlabel('ISI (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Activation Pattern', fontsize=14, fontweight='bold')
    ax.set_title('Mean Selected HRF Index\n(averaged across 1000 realizations)',
                 fontsize=15, fontweight='bold', pad=15)

    # Add text annotations
    for i in range(n_isis):
        for j in range(n_patterns):
            text = ax.text(i, j, f'{mean_selected[i, j]:.1f}',
                          ha="center", va="center",
                          color="white" if mean_selected[i, j] < 10 else "black",
                          fontsize=7)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('HRF Index\n(0=earliest peak, 19=latest peak)',
                   rotation=270, labelpad=25, fontsize=11, fontweight='bold')

    # Panel 2: Difference from true HRF
    ax = axes[1]

    # Compute difference (selected - true)
    # mean_selected is (n_isis, n_patterns), true_hrf is (n_patterns,)
    true_hrf_grid = np.tile(true_hrf, (n_isis, 1))  # (n_isis, n_patterns)
    difference = mean_selected - true_hrf_grid  # (n_isis, n_patterns)

    im = ax.imshow(difference.T, aspect='auto', cmap='RdBu_r',
                   vmin=-10, vmax=10, origin='lower')

    ax.set_xticks(np.arange(n_isis))
    ax.set_xticklabels([f"{isi:.2f}" for isi in isi_means])
    ax.set_yticks(np.arange(n_patterns))
    ax.set_yticklabels([f"P{idx} (true={true_hrf[idx]})" for idx, pidx in enumerate(pattern_indices)])

    ax.set_xlabel('ISI (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Activation Pattern (true HRF index shown)', fontsize=14, fontweight='bold')
    ax.set_title('Selection Error\n(mean selected - true HRF index)',
                 fontsize=15, fontweight='bold', pad=15)

    # Add text annotations (i is ISI, j is pattern)
    for i in range(n_isis):
        for j in range(n_patterns):
            color = "white" if abs(difference[i, j]) > 5 else "black"
            text = ax.text(i, j, f'{difference[i, j]:+.1f}',
                          ha="center", va="center", color=color,
                          fontsize=7)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Error (index units)\n(negative=earlier, positive=later)',
                   rotation=270, labelpad=25, fontsize=11, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / 'hrf_selection_gradient.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "="*70)
    print("HRF SELECTION SUMMARY")
    print("="*70)
    print(f"\nMean absolute error: {np.abs(difference).mean():.2f} index units")
    print(f"Median absolute error: {np.median(np.abs(difference)):.2f} index units")
    print(f"Std of error: {difference.std():.2f} index units")

    print(f"\nBias (mean error): {difference.mean():+.2f} index units")
    if difference.mean() > 0:
        print("  → Tends to select HRFs with LATER peaks")
    elif difference.mean() < 0:
        print("  → Tends to select HRFs with EARLIER peaks")
    else:
        print("  → No systematic bias")

    # Best and worst patterns
    pattern_errors = np.abs(difference).mean(axis=0)
    best_idx = pattern_indices[np.argmin(pattern_errors)]
    worst_idx = pattern_indices[np.argmax(pattern_errors)]

    print(f"\nBest pattern: P{best_idx} (mean error: {pattern_errors.min():.2f})")
    print(f"Worst pattern: P{worst_idx} (mean error: {pattern_errors.max():.2f})")


if __name__ == "__main__":
    results_file = Path("monte_carlo_comprehensive_results/comprehensive_results.pkl")
    output_dir = Path("monte_carlo_comprehensive_results")

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Run monte_carlo_comprehensive.py first!")
        sys.exit(1)

    plot_hrf_selection_gradient(results_file, output_dir)
