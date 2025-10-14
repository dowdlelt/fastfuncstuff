#!/usr/bin/env python
"""
Regenerate plots from existing Monte Carlo results with fixed plotting code.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/logan/local_bin/fastfuncsim")

from monte_carlo_comprehensive import plot_comprehensive_results

# Load existing results
output_dir = Path("monte_carlo_comprehensive_results")
results_file = output_dir / "comprehensive_results.pkl"

print(f"Loading results from: {results_file}")
with open(results_file, "rb") as f:
    results = pickle.load(f)

# Extract ISI means from results keys
isi_means = sorted(results.keys())
print(
    f"Found {len(isi_means)} ISI conditions: {isi_means[0]:.2f} to {isi_means[-1]:.2f}"
)

# Get number of patterns from results
n_patterns = len(results[isi_means[0]])
print(f"Found {n_patterns} activation patterns")

# Get number of HRFs from results
n_hrfs = len(results[isi_means[0]][0])
print(f"Found {n_hrfs} HRFs tested per pattern")

# Reconstruct activation patterns (same as original)
activation_patterns = np.array(
    [
        [5, 1],
        [5, 2],
        [5, 3],
        [5, 4],
        [5, 5],
        [4, 5],
        [3, 5],
        [2, 5],
        [1, 5],
        [1, 3],
        [3, 3],
        [3, 1],
        [1, 1],
        [0, 2],
        [2, 0],
        [-1, -1],
        [-3, -3],
        [-3, 4],
        [4, -3],
        [0, 0],
    ],
    dtype=np.float32,
)

print(f"\nRegenerating plots with fixed code...")
plot_comprehensive_results(results, activation_patterns, isi_means, output_dir)

print(f"\n{'=' * 70}")
print("DONE! Check the output directory for updated plots:")
print(f"  - r2_parametric.png")
print(f"  - beta_error_A.png")
print(f"  - beta_error_B.png")
print(f"  - fir_hrf_recovery_A.png")
print(f"  - fir_hrf_recovery_B.png")
print(f"  - hrf_recovery_heatmap_isi_*.png (one per ISI)")
print(f"{'=' * 70}")
