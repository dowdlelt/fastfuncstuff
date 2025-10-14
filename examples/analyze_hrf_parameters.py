#!/usr/bin/env python
"""
Analyze HRF library parameters and create enhanced confusion matrices.
"""
import sys
sys.path.insert(0, '/Users/logan/local_bin/fastfuncsim')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
from simulate_isi_sweep import load_cnvlab_hrf_library


def analyze_hrf_parameters(hrfs, tr=1.0):
    """
    Extract key parameters from each HRF.

    Returns dict with:
    - time_to_peak: seconds to maximum positive response
    - peak_amplitude: maximum value
    - time_to_undershoot: seconds to minimum (undershoot)
    - undershoot_amplitude: minimum value
    - fwhm: full width at half maximum
    """
    n_hrfs = hrfs.shape[0]
    params = {
        'time_to_peak': [],
        'peak_amplitude': [],
        'time_to_undershoot': [],
        'undershoot_amplitude': [],
        'fwhm': [],
        'integral': []
    }

    for i in range(n_hrfs):
        hrf = hrfs[i]
        time = np.arange(len(hrf)) * tr

        # Time to peak
        peak_idx = np.argmax(hrf)
        params['time_to_peak'].append(time[peak_idx])
        params['peak_amplitude'].append(hrf[peak_idx])

        # Time to undershoot (minimum after peak)
        undershoot_idx = peak_idx + np.argmin(hrf[peak_idx:])
        params['time_to_undershoot'].append(time[undershoot_idx])
        params['undershoot_amplitude'].append(hrf[undershoot_idx])

        # FWHM (full width at half maximum)
        half_max = hrf[peak_idx] / 2
        above_half = hrf > half_max
        if above_half.sum() > 1:
            first_idx = np.where(above_half)[0][0]
            last_idx = np.where(above_half)[0][-1]
            fwhm = (last_idx - first_idx) * tr
        else:
            fwhm = tr
        params['fwhm'].append(fwhm)

        # Integral (area under curve)
        params['integral'].append(np.trapz(hrf, dx=tr))

    return {k: np.array(v) for k, v in params.items()}


def plot_hrf_library_grid(hrfs, params, output_dir):
    """Plot all HRFs in a grid with parameters labeled."""
    n_hrfs = hrfs.shape[0]
    n_cols = 5
    n_rows = int(np.ceil(n_hrfs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()

    tr = 1.0
    time = np.arange(hrfs.shape[1]) * tr

    for i in range(n_hrfs):
        ax = axes[i]
        ax.plot(time, hrfs[i], 'b-', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(params['time_to_peak'][i], color='r', linestyle='--', alpha=0.5)

        ax.set_title(f'HRF {i}\n' +
                    f'Peak: {params["time_to_peak"][i]:.1f}s\n' +
                    f'FWHM: {params["fwhm"][i]:.1f}s',
                    fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 30])

    # Hide unused subplots
    for i in range(n_hrfs, len(axes)):
        axes[i].axis('off')

    plt.suptitle('HRF Library (20 Canonical HRFs)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'hrf_library_grid.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_hrf_parameter_space(params, output_dir):
    """Plot HRFs in parameter space."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Time to peak vs FWHM
    ax = axes[0, 0]
    scatter = ax.scatter(params['time_to_peak'], params['fwhm'],
                        c=np.arange(len(params['time_to_peak'])),
                        cmap='viridis', s=150, alpha=0.7, edgecolors='black')
    for i in range(len(params['time_to_peak'])):
        ax.text(params['time_to_peak'][i], params['fwhm'][i], str(i),
               ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set_xlabel('Time to Peak (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('FWHM (s)', fontsize=12, fontweight='bold')
    ax.set_title('HRF Parameter Space', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='HRF Index')

    # Plot 2: Peak amplitude vs undershoot
    ax = axes[0, 1]
    scatter = ax.scatter(params['peak_amplitude'], params['undershoot_amplitude'],
                        c=np.arange(len(params['peak_amplitude'])),
                        cmap='viridis', s=150, alpha=0.7, edgecolors='black')
    for i in range(len(params['peak_amplitude'])):
        ax.text(params['peak_amplitude'][i], params['undershoot_amplitude'][i], str(i),
               ha='center', va='center', fontsize=8, fontweight='bold')
    ax.set_xlabel('Peak Amplitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Undershoot Amplitude', fontsize=12, fontweight='bold')
    ax.set_title('Amplitude Characteristics', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='HRF Index')

    # Plot 3: Time to peak distribution
    ax = axes[1, 0]
    ax.hist(params['time_to_peak'], bins=15, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Time to Peak (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Time to Peak', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Plot 4: Parameter summary table
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = "HRF Library Summary\n" + "="*40 + "\n\n"
    summary_text += f"Number of HRFs: {len(params['time_to_peak'])}\n\n"
    summary_text += f"Time to Peak:\n"
    summary_text += f"  Range: {params['time_to_peak'].min():.2f} - {params['time_to_peak'].max():.2f}s\n"
    summary_text += f"  Mean: {params['time_to_peak'].mean():.2f}s\n"
    summary_text += f"  Std: {params['time_to_peak'].std():.2f}s\n\n"
    summary_text += f"FWHM:\n"
    summary_text += f"  Range: {params['fwhm'].min():.2f} - {params['fwhm'].max():.2f}s\n"
    summary_text += f"  Mean: {params['fwhm'].mean():.2f}s\n\n"
    summary_text += f"Peak Amplitude:\n"
    summary_text += f"  Range: {params['peak_amplitude'].min():.3f} - {params['peak_amplitude'].max():.3f}\n\n"
    summary_text += f"Undershoot:\n"
    summary_text += f"  Range: {params['undershoot_amplitude'].min():.3f} - {params['undershoot_amplitude'].max():.3f}\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = output_dir / 'hrf_parameter_space.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_parameter_error_analysis(results_file, params, output_dir):
    """
    Analyze parameter errors: show distance in parameter space between
    true and selected HRF.
    """
    print("\nLoading results for parameter error analysis...")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    isi_means = sorted(results.keys())
    first_isi = isi_means[0]
    pattern_indices = sorted(results[first_isi].keys())

    # Check if we have selection data
    if 'best_hrf_idx_all' not in results[first_isi][pattern_indices[0]]:
        print("Warning: Need HRF selection data. Results saved before modification.")
        print("Skipping parameter error analysis.")
        return

    # Collect parameter errors
    time_to_peak_errors = []
    fwhm_errors = []

    for isi in isi_means:
        for pattern_idx in pattern_indices:
            pattern_results = results[isi][pattern_idx]
            true_hrf = pattern_results['true_hrf_idx']
            selected_hrfs = pattern_results['best_hrf_idx_all']

            for sel_hrf in selected_hrfs:
                # Compute parameter differences
                ttp_error = abs(params['time_to_peak'][sel_hrf] - params['time_to_peak'][true_hrf])
                fwhm_error = abs(params['fwhm'][sel_hrf] - params['fwhm'][true_hrf])

                time_to_peak_errors.append(ttp_error)
                fwhm_errors.append(fwhm_error)

    # Plot error distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(time_to_peak_errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='green', linestyle='--', linewidth=2, label='Perfect')
    ax.axvline(np.median(time_to_peak_errors), color='red', linestyle='--', linewidth=2,
              label=f'Median: {np.median(time_to_peak_errors):.2f}s')
    ax.set_xlabel('|Time-to-Peak Error| (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('HRF Time-to-Peak Selection Error', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.hist(fwhm_errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='green', linestyle='--', linewidth=2, label='Perfect')
    ax.axvline(np.median(fwhm_errors), color='red', linestyle='--', linewidth=2,
              label=f'Median: {np.median(fwhm_errors):.2f}s')
    ax.set_xlabel('|FWHM Error| (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('HRF Width Selection Error', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = output_dir / 'hrf_parameter_errors.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Print summary
    print("\n" + "="*70)
    print("HRF PARAMETER ERROR SUMMARY")
    print("="*70)
    print(f"Time-to-Peak Error:")
    print(f"  Median: {np.median(time_to_peak_errors):.3f}s")
    print(f"  Mean: {np.mean(time_to_peak_errors):.3f}s")
    print(f"  75th percentile: {np.percentile(time_to_peak_errors, 75):.3f}s")
    print(f"\nFWHM Error:")
    print(f"  Median: {np.median(fwhm_errors):.3f}s")
    print(f"  Mean: {np.mean(fwhm_errors):.3f}s")
    print(f"  75th percentile: {np.percentile(fwhm_errors, 75):.3f}s")


if __name__ == "__main__":
    output_dir = Path("monte_carlo_comprehensive_results")
    results_file = output_dir / "comprehensive_results.pkl"

    print("Loading HRF library...")
    hrfs = load_cnvlab_hrf_library(duration=5.0, tr=1.0)
    print(f"Loaded {hrfs.shape[0]} HRFs, {hrfs.shape[1]} timepoints each")

    print("\nExtracting HRF parameters...")
    params = analyze_hrf_parameters(hrfs, tr=1.0)

    print("\nCreating visualizations...")
    plot_hrf_library_grid(hrfs, params, output_dir)
    plot_hrf_parameter_space(params, output_dir)

    if results_file.exists():
        plot_parameter_error_analysis(results_file, params, output_dir)
    else:
        print(f"\nResults file not found: {results_file}")
        print("Run monte_carlo_comprehensive.py first to analyze selection errors.")

    print("\nDone!")
