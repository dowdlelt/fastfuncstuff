#!/usr/bin/env python
"""Quick diagnostic to check SNR and noise characteristics in the simulation."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, '/Users/logan/local_bin/fastfuncsim')
import fastfuncsim as ffs
from fastfuncsim.noise import generate_fmri_noise

from simulate_isi_sweep import (
    load_cnvlab_hrf_library,
    generate_poisson_isis,
    build_alternating_design,
    convolve_design_with_hrf,
)

# Parameters
tr = 1.0
total_duration = 290.0
stim_duration = 5.0
noise_std = 2.0
baseline = 100.0

# Generate a single design
total_tr = int(total_duration / tr)
stim_tr = int(stim_duration / tr)
padding_tr = int(10 / tr)

# Generate ISIs
isi_mean = 3.0
block_average = isi_mean + stim_duration
n_trials = int((total_duration - 2 * padding_tr * tr) / block_average)
n_trials = n_trials - (n_trials % 2)

isis_sec = generate_poisson_isis(
    target_mean=isi_mean,
    n_isis=n_trials,
    lower_limit=2.0,
    upper_limit=8.0,
    seed=42
)
isis_tr = np.round(isis_sec / tr).astype(int)

# Build design
design_unconvolved = build_alternating_design(
    isis_tr=isis_tr,
    stim_dur_tr=stim_tr,
    n_conds=2,
    total_tr=total_tr,
    padding_tr=padding_tr
)

# Load HRF library
hrfs = load_cnvlab_hrf_library(duration=stim_duration, tr=tr)
n_hrfs = hrfs.shape[0]

print(f"Loaded {n_hrfs} HRFs from CNVLab library")
print(f"Design: {np.sum(design_unconvolved[:, 0]):.0f} + {np.sum(design_unconvolved[:, 1]):.0f} trials")
print()

# Test strong activation pattern [5, 1]
activation_pattern = np.array([5.0, 1.0])
true_hrf = hrfs[0]

# Create signal
onset_vector = np.zeros(total_tr)
for cond_idx in range(2):
    onset_times = np.where(design_unconvolved[:, cond_idx] > 0)[0]
    onset_vector[onset_times] = activation_pattern[cond_idx]

true_signal = np.convolve(onset_vector, true_hrf, mode='full')[:total_tr]

# Generate noise
device = ffs.get_device()
noise_2d = generate_fmri_noise(
    tr=tr,
    duration_s=total_tr * tr,
    matrix_size=(1, 100),
    normalize=True,
    device=device
)
noise_np = noise_2d.cpu().numpy().squeeze()[:, 0] * noise_std  # Just one voxel

# Compute statistics
signal_mean = true_signal.mean()
signal_std = true_signal.std()
signal_peak = np.abs(true_signal - signal_mean).max()

noise_mean = noise_np.mean()
noise_std_actual = noise_np.std()

print("SIGNAL STATISTICS (pattern [5, 1]):")
print(f"  Mean: {signal_mean:.2f}")
print(f"  Std:  {signal_std:.2f}")
print(f"  Peak (from mean): {signal_peak:.2f}")
print()

print("NOISE STATISTICS:")
print(f"  Mean: {noise_mean:.2f}")
print(f"  Std:  {noise_std_actual:.2f}")
print(f"  Expected std: {noise_std:.2f}")
print()

# SNR
snr = signal_std / noise_std_actual
print(f"SNR (signal_std / noise_std): {snr:.2f}")
print(f"Peak SNR (signal_peak / noise_std): {signal_peak / noise_std_actual:.2f}")
print()

# Check HRF diversity
print("HRF DIVERSITY:")
print(f"  Library size: {n_hrfs}")
print(f"  HRF length: {hrfs.shape[1]} timepoints")

# Compute pairwise correlations
correlations = []
for i in range(n_hrfs):
    for j in range(i+1, n_hrfs):
        corr = np.corrcoef(hrfs[i], hrfs[j])[0, 1]
        correlations.append(corr)

correlations = np.array(correlations)
print(f"  Mean correlation: {correlations.mean():.3f}")
print(f"  Min correlation:  {correlations.min():.3f}")
print(f"  Max correlation:  {correlations.max():.3f}")
print()

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Signal
axes[0, 0].plot(true_signal, linewidth=1)
axes[0, 0].axhline(signal_mean, color='red', linestyle='--', alpha=0.5, label='Mean')
axes[0, 0].set_title(f'Signal (pattern [5, 1])', fontweight='bold')
axes[0, 0].set_xlabel('Time (TR)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Noise
axes[0, 1].plot(noise_np, linewidth=0.5, alpha=0.8)
axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[0, 1].set_title(f'Noise (std={noise_std})', fontweight='bold')
axes[0, 1].set_xlabel('Time (TR)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].grid(alpha=0.3)

# Signal + Noise
data = true_signal + noise_np
axes[1, 0].plot(data, linewidth=1, alpha=0.8, label='Signal + Noise')
axes[1, 0].plot(true_signal, linewidth=1, alpha=0.5, linestyle='--', label='True Signal')
axes[1, 0].set_title(f'Signal + Noise (SNR={snr:.2f})', fontweight='bold')
axes[1, 0].set_xlabel('Time (TR)')
axes[1, 0].set_ylabel('Amplitude')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# HRF correlation histogram
axes[1, 1].hist(correlations, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].set_title(f'HRF Pairwise Correlations (n={len(correlations)})', fontweight='bold')
axes[1, 1].set_xlabel('Correlation')
axes[1, 1].set_ylabel('Count')
axes[1, 1].axvline(correlations.mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {correlations.mean():.3f}')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('snr_diagnostic.png', dpi=150, bbox_inches='tight')
print(f"Saved: snr_diagnostic.png")
plt.close()

# Plot all HRFs
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(n_hrfs):
    ax.plot(hrfs[i], alpha=0.5, linewidth=1)
ax.set_title(f'CNVLab HRF Library (n={n_hrfs})', fontweight='bold')
ax.set_xlabel('Time (TR)')
ax.set_ylabel('Amplitude')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('hrf_library.png', dpi=150, bbox_inches='tight')
print(f"Saved: hrf_library.png")
plt.close()

print("\nDone!")
