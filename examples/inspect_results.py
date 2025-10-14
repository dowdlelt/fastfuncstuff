#!/usr/bin/env python
"""
Inspect the structure of the Monte Carlo results.
"""

import pickle
from pathlib import Path

# Load existing results
output_dir = Path("monte_carlo_comprehensive_results")
results_file = output_dir / "comprehensive_results.pkl"

print(f"Loading results from: {results_file}")
with open(results_file, "rb") as f:
    results = pickle.load(f)

# Extract ISI means from results keys
isi_means = sorted(results.keys())
print(f"\nISI conditions: {len(isi_means)}")
print(f"  First ISI: {isi_means[0]}")

# Check structure
first_isi = isi_means[0]
print(f"\nStructure at results[{first_isi}]:")
print(f"  Type: {type(results[first_isi])}")
print(
    f"  Keys/Length: {len(results[first_isi]) if isinstance(results[first_isi], dict) else 'N/A'}"
)

if isinstance(results[first_isi], dict):
    pattern_keys = list(results[first_isi].keys())
    print(f"  Pattern keys (first 5): {pattern_keys[:5]}")

    first_pattern = pattern_keys[0]
    print(f"\nStructure at results[{first_isi}][{first_pattern}]:")
    print(f"  Type: {type(results[first_isi][first_pattern])}")

    if isinstance(results[first_isi][first_pattern], dict):
        hrf_keys = list(results[first_isi][first_pattern].keys())
        print(f"  HRF keys (first 5): {hrf_keys[:5]}")
        print(f"  Total HRFs: {len(hrf_keys)}")

        if hrf_keys:
            first_hrf = hrf_keys[0]
            print(f"\nStructure at results[{first_isi}][{first_pattern}][{first_hrf}]:")
            print(f"  Type: {type(results[first_isi][first_pattern][first_hrf])}")

            if isinstance(results[first_isi][first_pattern][first_hrf], dict):
                metric_keys = list(results[first_isi][first_pattern][first_hrf].keys())
                print(f"  Metric keys: {metric_keys}")
    else:
        # Maybe it's directly the results dict?
        if isinstance(results[first_isi][first_pattern], dict):
            metric_keys = list(results[first_isi][first_pattern].keys())
            print(f"  Metric keys: {metric_keys}")
