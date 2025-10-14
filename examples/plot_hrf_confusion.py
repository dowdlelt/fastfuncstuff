#!/usr/bin/env python
"""
Comprehensive HRF Selection Visualization

Shows confusion matrix of true HRF vs selected HRF across all simulations.
"""
import sys
sys.path.insert(0, '/Users/logan/local_bin/fastfuncsim')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from matplotlib.colors import LogNorm


def plot_hrf_confusion_comprehensive(results_file, output_dir):
    """
    Create comprehensive visualization of HRF selection.

    Shows for each pattern and ISI: which HRF was chosen when each true HRF was used.
    """
    print("Loading results...")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    # Get ISI conditions and patterns
    isi_means = sorted(results.keys())
    n_isis = len(isi_means)

    # Get activation patterns from first ISI
    first_isi = isi_means[0]
    pattern_indices = sorted(results[first_isi].keys())
    n_patterns = len(pattern_indices)

    print(f"Found {n_isis} ISI conditions and {n_patterns} patterns")

    # Get HRF info from first pattern
    sample_pattern_idx = pattern_indices[0]
    sample_results = results[first_isi][sample_pattern_idx]

    # Infer n_hrfs from recovery rate (it's a proportion)
    # We need to look at the actual data structure
    # Let's check what's in the results

    # First, let's create an overall confusion matrix across all conditions
    print("\nCreating overall confusion matrix...")

    # We need to extract the HRF selection data
    # From the code, we stored: best_hrf_idx_per_realization and compared to true_hrf_idx
    # But we didn't save the raw selection data, only the recovery rate!

    # Let me check if we can reconstruct or if we need to recompute
    print("\nAvailable keys in results:")
    print(list(sample_results.keys()))

    # Check if we have the raw data
    if 'best_hrf_idx_all' not in sample_results:
        print("\nWarning: Raw HRF selection data not saved.")
        print("We only have recovery rate (percentage correct).")
        print("\nTo create confusion matrices, we need to rerun with saved selection data.")
        print("For now, plotting recovery rates across patterns and ISI...")

        plot_recovery_rate_grid(results, isi_means, pattern_indices, output_dir)
        return

    print("Found raw HRF selection data!")

    # Get number of HRFs
    n_hrfs = 20  # From the code, we load 20 HRFs

    # Create overall confusion matrix across all conditions
    overall_confusion = np.zeros((n_hrfs, n_hrfs), dtype=int)

    for isi in isi_means:
        for pattern_idx in pattern_indices:
            pattern_results = results[isi][pattern_idx]
            true_hrf = pattern_results['true_hrf_idx']
            selected_hrfs = pattern_results['best_hrf_idx_all']

            # Update confusion matrix
            for sel_hrf in selected_hrfs:
                overall_confusion[true_hrf, sel_hrf] += 1

    # Plot overall confusion matrix
    plot_confusion_matrix(overall_confusion, output_dir, 'overall')

    # Plot per-ISI confusion matrices
    plot_confusion_per_isi(results, isi_means, pattern_indices, n_hrfs, output_dir)

    # Also plot the recovery rate grid
    plot_recovery_rate_grid(results, isi_means, pattern_indices, output_dir)


def plot_confusion_matrix(confusion_matrix, output_dir, suffix='overall'):
    """Plot a single confusion matrix."""
    n_hrfs = confusion_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(14, 12))

    # Normalize by row (true HRF) to show proportions
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    confusion_normalized = confusion_matrix / (row_sums + 1e-10)

    im = ax.imshow(confusion_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(np.arange(n_hrfs))
    ax.set_yticks(np.arange(n_hrfs))
    ax.set_xticklabels(np.arange(n_hrfs))
    ax.set_yticklabels(np.arange(n_hrfs))

    ax.set_xlabel('Selected HRF Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('True HRF Index', fontsize=14, fontweight='bold')
    ax.set_title(f'HRF Confusion Matrix ({suffix})\n' +
                 'Showing proportion selected for each true HRF',
                 fontsize=16, fontweight='bold', pad=20)

    # Add text annotations for diagonal and high values
    for i in range(n_hrfs):
        for j in range(n_hrfs):
            if confusion_normalized[i, j] > 0.1 or i == j:  # Show if >10% or diagonal
                text = ax.text(j, i, f'{confusion_normalized[i, j]:.2f}',
                              ha="center", va="center",
                              color="white" if confusion_normalized[i, j] > 0.5 else "black",
                              fontsize=8)

    # Highlight diagonal
    for i in range(n_hrfs):
        rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                             edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion Selected', rotation=270, labelpad=20,
                   fontsize=12, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / f'hrf_confusion_{suffix}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print summary
    print(f"\n{suffix.upper()} Confusion Matrix Summary:")
    print(f"  Diagonal accuracy: {np.diag(confusion_normalized).mean():.3f}")
    print(f"  Total selections: {confusion_matrix.sum():,}")


def plot_confusion_per_isi(results, isi_means, pattern_indices, n_hrfs, output_dir):
    """Create confusion matrix for each ISI condition."""
    n_isis = len(isi_means)

    # Create subplot grid
    n_cols = 4
    n_rows = int(np.ceil(n_isis / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()

    for idx, isi in enumerate(isi_means):
        ax = axes[idx]

        # Build confusion matrix for this ISI
        confusion = np.zeros((n_hrfs, n_hrfs), dtype=int)

        for pattern_idx in pattern_indices:
            pattern_results = results[isi][pattern_idx]
            true_hrf = pattern_results['true_hrf_idx']
            selected_hrfs = pattern_results['best_hrf_idx_all']

            for sel_hrf in selected_hrfs:
                confusion[true_hrf, sel_hrf] += 1

        # Normalize
        row_sums = confusion.sum(axis=1, keepdims=True)
        confusion_norm = confusion / (row_sums + 1e-10)

        # Plot
        im = ax.imshow(confusion_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'ISI={isi:.2f}s\n(Acc={np.diag(confusion_norm).mean():.2f})',
                     fontsize=11, fontweight='bold')

        if idx % n_cols == 0:  # First column
            ax.set_ylabel('True HRF', fontsize=10)
        if idx >= (n_rows-1)*n_cols:  # Bottom row
            ax.set_xlabel('Selected HRF', fontsize=10)

        # Highlight diagonal
        for i in range(n_hrfs):
            rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                edgecolor='red', linewidth=1, alpha=0.5)
            ax.add_patch(rect)

        ax.set_xticks(np.arange(0, n_hrfs, 5))
        ax.set_yticks(np.arange(0, n_hrfs, 5))

    # Hide unused subplots
    for idx in range(n_isis, len(axes)):
        axes[idx].axis('off')

    # Add overall colorbar
    fig.colorbar(im, ax=axes, fraction=0.01, pad=0.02,
                label='Proportion Selected')

    plt.suptitle('HRF Confusion Matrices by ISI Condition', fontsize=18,
                 fontweight='bold', y=1.00)
    plt.tight_layout()

    output_path = output_dir / 'hrf_confusion_by_isi.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def plot_recovery_rate_grid(results, isi_means, pattern_indices, output_dir):
    """
    Plot HRF recovery rate as a heatmap across ISI and patterns.
    """
    n_isis = len(isi_means)
    n_patterns = len(pattern_indices)

    # Extract recovery rates
    recovery_grid = np.zeros((n_isis, n_patterns))

    for i, isi in enumerate(isi_means):
        for j, pattern_idx in enumerate(pattern_indices):
            recovery_grid[i, j] = results[isi][pattern_idx]['hrf_recovery_rate']

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    im = ax.imshow(recovery_grid.T, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=1, origin='lower')

    # Set ticks
    ax.set_xticks(np.arange(n_isis))
    ax.set_xticklabels([f"{isi:.2f}" for isi in isi_means])
    ax.set_yticks(np.arange(n_patterns))
    ax.set_yticklabels([f"P{idx}" for idx in pattern_indices])

    ax.set_xlabel('ISI (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Activation Pattern', fontsize=14, fontweight='bold')
    ax.set_title('HRF Library Selection Accuracy\n(Proportion Correct across 1000 realizations)',
                 fontsize=16, fontweight='bold', pad=20)

    # Add text annotations
    for i in range(n_isis):
        for j in range(n_patterns):
            text = ax.text(i, j, f'{recovery_grid[i, j]:.2f}',
                          ha="center", va="center", color="black",
                          fontsize=7)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('HRF Recovery Rate', rotation=270, labelpad=20,
                   fontsize=12, fontweight='bold')

    # Add reference line at 0.5 (chance level if HRFs were very different)
    # Actually chance is 1/20 = 0.05 with 20 HRFs
    ax.axhline(y=-0.5, color='white', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(-0.5, -0.5, 'Chance=1/20=0.05', fontsize=10, ha='right',
            va='center', color='red', fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / 'hrf_recovery_heatmap.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "="*70)
    print("HRF RECOVERY SUMMARY")
    print("="*70)
    print(f"Overall mean recovery rate: {recovery_grid.mean():.3f}")
    print(f"Overall median recovery rate: {np.median(recovery_grid):.3f}")
    print(f"Min recovery rate: {recovery_grid.min():.3f}")
    print(f"Max recovery rate: {recovery_grid.max():.3f}")
    print(f"\nChance level: {1/20:.3f} (1/20 HRFs)")
    print(f"Mean improvement over chance: {recovery_grid.mean() - 1/20:.3f}")

    # Best and worst patterns
    pattern_means = recovery_grid.mean(axis=0)
    best_pattern_idx = pattern_indices[np.argmax(pattern_means)]
    worst_pattern_idx = pattern_indices[np.argmin(pattern_means)]

    print(f"\nBest pattern: P{best_pattern_idx} (mean recovery: {pattern_means.max():.3f})")
    print(f"Worst pattern: P{worst_pattern_idx} (mean recovery: {pattern_means.min():.3f})")

    # Best and worst ISI
    isi_means_recovery = recovery_grid.mean(axis=1)
    best_isi = isi_means[np.argmax(isi_means_recovery)]
    worst_isi = isi_means[np.argmin(isi_means_recovery)]

    print(f"\nBest ISI: {best_isi:.2f}s (mean recovery: {isi_means_recovery.max():.3f})")
    print(f"Worst ISI: {worst_isi:.2f}s (mean recovery: {isi_means_recovery.min():.3f})")


if __name__ == "__main__":
    results_file = Path("monte_carlo_comprehensive_results/comprehensive_results.pkl")
    output_dir = Path("monte_carlo_comprehensive_results")

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        print("Run monte_carlo_comprehensive.py first!")
        sys.exit(1)

    plot_hrf_confusion_comprehensive(results_file, output_dir)
