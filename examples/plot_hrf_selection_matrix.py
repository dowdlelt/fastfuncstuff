#!/usr/bin/env python
"""
Full HRF selection matrix visualization.

Shows which HRFs were selected for each activation pattern and true HRF.
"""
import sys
sys.path.insert(0, '/Users/logan/local_bin/fastfuncsim')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


def plot_hrf_selection_matrix_per_isi(results_file, output_dir):
    """
    Create selection matrix for each ISI showing:
    - X-axis: Activation patterns (labeled with beta values)
    - Y-axis: HRF index (0-19, representing different HRF shapes)
    - Color: Mean selected HRF for that pattern
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
        return

    # Get activation patterns (beta values for conditions A and B)
    # These are hardcoded in monte_carlo_comprehensive.py
    activation_patterns = np.array([
        [5, 1], [5, 2], [5, 3], [5, 4], [5, 5],
        [4, 5], [3, 5], [2, 5], [1, 5],
        [1, 3], [3, 3], [3, 1], [1, 1],
        [0, 2], [2, 0],
        [-1, -1], [-3, -3],
        [-3, 4], [4, -3],
        [0, 0]
    ], dtype=np.float32)

    # Create figure with subplots for each ISI
    n_cols = 3
    n_rows = int(np.ceil(n_isis / n_cols))

    fig = plt.figure(figsize=(22, 6*n_rows))

    for isi_idx, isi in enumerate(isi_means):
        ax = plt.subplot(n_rows, n_cols, isi_idx + 1)

        # Build matrix: (n_patterns, 20) showing which HRF was selected
        # For each pattern, we have 1000 selections, we'll show the distribution
        selection_heatmap = np.zeros((n_patterns, 20))

        for j, pattern_idx in enumerate(pattern_indices):
            pattern_results = results[isi][pattern_idx]
            true_hrf = pattern_results['true_hrf_idx']
            selected_hrfs = pattern_results['best_hrf_idx_all']

            # Count selections for each HRF
            for sel_hrf in selected_hrfs:
                selection_heatmap[j, sel_hrf] += 1

            # Normalize to proportions
            selection_heatmap[j, :] /= len(selected_hrfs)

        # Plot heatmap
        im = ax.imshow(selection_heatmap.T, aspect='auto', cmap='turbo',
                      vmin=0, vmax=1, origin='lower')

        # Set ticks
        ax.set_xticks(np.arange(n_patterns))
        pattern_labels = [f"P{i}\n[{activation_patterns[i,0]:.0f},{activation_patterns[i,1]:.0f}]\nH{i%20}"
                         for i in range(n_patterns)]
        ax.set_xticklabels(pattern_labels, fontsize=7, rotation=0)

        ax.set_yticks(np.arange(0, 20, 2))
        ax.set_yticklabels(np.arange(0, 20, 2))

        ax.set_xlabel('Pattern [A,B] / True HRF', fontsize=10, fontweight='bold')
        ax.set_ylabel('Selected HRF Index', fontsize=10, fontweight='bold')
        ax.set_title(f'ISI={isi:.2f}s\nSelection Distribution',
                    fontsize=11, fontweight='bold')

        # Add diagonal line to show where true HRF is
        true_hrf_indices = [pattern_idx % 20 for pattern_idx in pattern_indices]
        ax.plot(np.arange(n_patterns), true_hrf_indices, 'r--',
               linewidth=2, alpha=0.7, label='True HRF')

        # Mark highest selections with dots
        for j in range(n_patterns):
            best_sel = np.argmax(selection_heatmap[j, :])
            if selection_heatmap[j, best_sel] > 0.1:  # Only if >10% selected this
                ax.plot(j, best_sel, 'ko', markersize=4, alpha=0.5)

        if isi_idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Add overall colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Proportion Selected\n(across 1000 realizations)',
                  rotation=270, labelpad=25, fontsize=12, fontweight='bold')

    plt.suptitle('HRF Selection Matrix by ISI Condition\n' +
                'Red dashed line = True HRF | Black dots = Most selected HRF',
                fontsize=16, fontweight='bold', y=0.98)

    output_path = output_dir / 'hrf_selection_matrix_by_isi.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


def plot_mean_selected_per_pattern(results_file, output_dir):
    """
    Simpler view: just show mean selected HRF for each pattern at each ISI.
    """
    print("\nCreating simplified mean selection plot...")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    isi_means = sorted(results.keys())
    first_isi = isi_means[0]
    pattern_indices = sorted(results[first_isi].keys())

    n_isis = len(isi_means)
    n_patterns = len(pattern_indices)

    # Get activation patterns
    activation_patterns = np.array([
        [5, 1], [5, 2], [5, 3], [5, 4], [5, 5],
        [4, 5], [3, 5], [2, 5], [1, 5],
        [1, 3], [3, 3], [3, 1], [1, 1],
        [0, 2], [2, 0],
        [-1, -1], [-3, -3],
        [-3, 4], [4, -3],
        [0, 0]
    ], dtype=np.float32)

    # Create single large heatmap: ISI x Pattern
    mean_selected = np.zeros((n_isis, n_patterns))
    true_hrf_per_pattern = np.zeros(n_patterns, dtype=int)

    for i, isi in enumerate(isi_means):
        for j, pattern_idx in enumerate(pattern_indices):
            pattern_results = results[isi][pattern_idx]

            if i == 0:
                true_hrf_per_pattern[j] = pattern_results['true_hrf_idx']

            selected_hrfs = pattern_results['best_hrf_idx_all']
            mean_selected[i, j] = np.mean(selected_hrfs)

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 8))

    im = ax.imshow(mean_selected.T, aspect='auto', cmap='turbo',
                   vmin=0, vmax=19, origin='lower')

    # Set ticks
    ax.set_xticks(np.arange(n_isis))
    ax.set_xticklabels([f"{isi:.2f}" for isi in isi_means], fontsize=10)

    ax.set_yticks(np.arange(n_patterns))
    pattern_labels = [f"P{i}: [{activation_patterns[i,0]:.0f},{activation_patterns[i,1]:.0f}] (true HRF={true_hrf_per_pattern[i]})"
                     for i in range(n_patterns)]
    ax.set_yticklabels(pattern_labels, fontsize=9)

    ax.set_xlabel('ISI (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Activation Pattern [A, B] (True HRF shown)', fontsize=14, fontweight='bold')
    ax.set_title('Mean Selected HRF Index\n(Color shows average selected HRF across 1000 realizations)',
                 fontsize=16, fontweight='bold', pad=15)

    # Add text annotations showing the value
    for i in range(n_isis):
        for j in range(n_patterns):
            # Color text based on background
            bg_val = mean_selected[i, j]
            # Turbo colormap: darker in middle, lighter at edges
            text_color = "white" if 5 < bg_val < 15 else "black"

            # Show mean selected with true in parentheses
            text = ax.text(i, j,
                          f'{mean_selected[i, j]:.1f}\n({true_hrf_per_pattern[j]})',
                          ha="center", va="center",
                          color=text_color, fontsize=6)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Selected HRF Index\n(0=earliest peak, 19=latest peak)',
                   rotation=270, labelpad=25, fontsize=12, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / 'hrf_mean_selected_full.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    results_file = Path("monte_carlo_comprehensive_results/comprehensive_results.pkl")
    output_dir = Path("monte_carlo_comprehensive_results")

    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)

    plot_hrf_selection_matrix_per_isi(results_file, output_dir)
    plot_mean_selected_per_pattern(results_file, output_dir)

    print("\nDone!")
