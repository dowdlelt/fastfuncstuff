#!/usr/bin/env python
"""
Quick test of noise spectrum with physiological components
"""
import sys
sys.path.insert(0, '/Users/logan/local_bin/fastfuncsim')

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import fastfuncsim as ffs
from fastfuncsim.noise import generate_fmri_noise


def plot_noise_spectrum_test():
    """Generate and plot noise spectrum to verify physiological peaks."""
    device = ffs.get_device()
    print(f"Using device: {device}")

    # Parameters matching monte_carlo_comprehensive.py
    tr = 1.0
    duration_s = 290.0
    n_realizations = 500
    noise_std = 2.0
    baseline = 100.0

    # Generate noise with physiological components
    print(f"\nGenerating {n_realizations} noise realizations...")
    print(f"  TR: {tr}s")
    print(f"  Duration: {duration_s}s")
    print(f"  Noise SD: {noise_std}")
    print(f"  Baseline: {baseline}")
    print(f"\nPhysiological components:")
    print(f"  Respiratory: 0.3 Hz (strength: 30)")
    print(f"  Cardiac: 1.5 Hz (strength: 25)")
    print(f"  → Cardiac will alias to 0.5 Hz at TR=1s")

    noise_all = generate_fmri_noise(
        tr=tr,
        duration_s=duration_s,
        matrix_size=(1, n_realizations),
        resp_freq=0.3,
        resp_strength=30.0,  # Strong respiratory
        cardiac_freq=1.5,     # Will alias to 0.5 Hz
        cardiac_strength=25.0,  # Strong cardiac
        pink_exp=1.0,
        normalize=True,
        device=device
    ).squeeze() * noise_std

    noise_samples = noise_all.cpu().numpy()  # (n_timepoints, n_realizations)
    n_timepoints = noise_samples.shape[0]

    print(f"  Generated: {noise_samples.shape}")
    print(f"  Mean: {noise_samples.mean():.3f}")
    print(f"  Std: {noise_samples.std():.3f}")

    # Compute FFT for each realization
    print("\nComputing power spectrum...")
    freqs = np.fft.rfftfreq(n_timepoints, d=tr)

    power_spectra = []
    for i in range(n_realizations):
        fft_vals = np.fft.rfft(noise_samples[:, i])
        power = np.abs(fft_vals) ** 2
        power_spectra.append(power)

    power_avg = np.mean(power_spectra, axis=0)
    power_std = np.std(power_spectra, axis=0)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Linear scale (full range)
    ax = axes[0, 0]
    ax.plot(freqs, power_avg, 'b-', linewidth=2)
    ax.fill_between(freqs, power_avg - power_std, power_avg + power_std,
                     alpha=0.3, color='blue')

    nyquist = 0.5 / tr
    resp_freq = 0.3
    cardiac_freq = 1.5  # Pre-alias frequency
    fs_sample = 1.0 / tr
    cardiac_aliased = abs(cardiac_freq - round(cardiac_freq / fs_sample) * fs_sample)

    ax.axvline(resp_freq, color='green', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Respiratory ({resp_freq} Hz)')
    ax.axvline(cardiac_aliased, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Cardiac aliased\n({cardiac_freq} Hz → {cardiac_aliased:.1f} Hz)')

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power', fontsize=12)
    ax.set_title('Full Spectrum (Linear)', fontsize=13, fontweight='bold')
    ax.set_xlim([0, nyquist])
    ax.grid(alpha=0.3)
    ax.legend()

    # Panel 2: Zoomed to low frequencies
    ax = axes[0, 1]
    max_freq_zoom = 0.15
    idx_zoom = freqs <= max_freq_zoom
    ax.plot(freqs[idx_zoom], power_avg[idx_zoom], 'b-', linewidth=2)
    ax.fill_between(freqs[idx_zoom],
                     power_avg[idx_zoom] - power_std[idx_zoom],
                     power_avg[idx_zoom] + power_std[idx_zoom],
                     alpha=0.3, color='blue')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power', fontsize=12)
    ax.set_title(f'Low Frequencies (< {max_freq_zoom} Hz)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # Panel 3: Respiratory band
    ax = axes[1, 0]
    resp_band = (freqs > 0.2) & (freqs < 0.4)
    ax.plot(freqs[resp_band], power_avg[resp_band], 'g-', linewidth=3)
    ax.fill_between(freqs[resp_band],
                     power_avg[resp_band] - power_std[resp_band],
                     power_avg[resp_band] + power_std[resp_band],
                     alpha=0.3, color='green')
    ax.axvline(resp_freq, color='darkgreen', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power', fontsize=12)
    ax.set_title(f'Respiratory Band (0.2-0.4 Hz)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # Panel 4: Log-log
    ax = axes[1, 1]
    idx_nonzero = freqs > 0
    ax.loglog(freqs[idx_nonzero], power_avg[idx_nonzero], 'b-', linewidth=2)

    # Fit slope
    p = np.polyfit(np.log10(freqs[idx_nonzero]), np.log10(power_avg[idx_nonzero]), 1)
    slope = p[0]
    intercept = p[1]
    f_ref = np.logspace(np.log10(freqs[idx_nonzero][0]),
                        np.log10(freqs[idx_nonzero][-1]), 100)
    fitted_power = 10**(intercept + slope * np.log10(f_ref))
    ax.plot(f_ref, fitted_power, 'r--', linewidth=2, alpha=0.7,
            label=f'Slope: {slope:.2f}')

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power', fontsize=12)
    ax.set_title('Log-Log Scale', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, which='both')
    ax.legend()

    plt.tight_layout()

    output_path = Path('noise_spectrum_test.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("NOISE SPECTRUM SUMMARY")
    print("="*60)
    print(f"Spectral slope (log-log): {slope:.3f}")
    print(f"Expected for 1/f noise: ~-1.0")
    print(f"\nPower at key frequencies:")

    # Find power at specific frequencies
    def get_power_at_freq(target_freq, width=0.02):
        idx = (freqs >= target_freq - width) & (freqs <= target_freq + width)
        if idx.any():
            return power_avg[idx].mean()
        return np.nan

    dc_power = power_avg[0]
    resp_power = get_power_at_freq(0.3, width=0.05)
    cardiac_power = get_power_at_freq(cardiac_aliased, width=0.05)
    max_power = power_avg.max()

    print(f"  DC (0 Hz): {dc_power:.2e} ({100*dc_power/max_power:.1f}% of max)")
    print(f"  Low freq (0.01-0.1 Hz): {np.mean(power_avg[(freqs > 0.01) & (freqs < 0.1)]):.2e}")
    print(f"  Respiratory (~0.3 Hz): {resp_power:.2e} ({100*resp_power/max_power:.1f}% of max)")
    print(f"  Cardiac aliased (~{cardiac_aliased:.1f} Hz): {cardiac_power:.2e} ({100*cardiac_power/max_power:.1f}% of max)")
    print(f"  High freq (0.4-0.5 Hz): {np.mean(power_avg[(freqs > 0.4) & (freqs < 0.5)]):.2e}")

    # Check if peaks are visible
    low_freq_power = np.mean(power_avg[(freqs > 0.05) & (freqs < 0.15)])
    resp_peak_ratio = resp_power / low_freq_power
    cardiac_peak_ratio = cardiac_power / low_freq_power

    print(f"\nPeak prominence (relative to 0.05-0.15 Hz baseline):")
    print(f"  Respiratory peak: {resp_peak_ratio:.2f}x")
    print(f"  Cardiac peak: {cardiac_peak_ratio:.2f}x")

    if resp_peak_ratio > 1.5:
        print("  ✓ Respiratory peak is visible")
    else:
        print("  ⚠ Respiratory peak may be weak")

    if cardiac_peak_ratio > 1.2:
        print("  ✓ Cardiac aliased peak is visible")
    else:
        print("  ⚠ Cardiac peak may be weak (expected for aliasing)")


if __name__ == "__main__":
    plot_noise_spectrum_test()
