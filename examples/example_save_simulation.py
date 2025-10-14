#!/usr/bin/env python
"""
Example: Save simulation outputs with nibabel

Demonstrates:
- Multi-run simulation
- AFNI-compatible onset timing files
- NIfTI output with nibabel
- Organized folder structure
"""

import fastfuncsim as ffs
import torch

# Setup
device = ffs.get_device()
print(f"Using device: {device}")

# Simulation parameters
n_runs = 3  # For cross-validation
n_conditions = 2
n_timepoints = 200
tr = 2.0
duration = n_timepoints * tr
matrix_size = (20, 20, 5)  # Small for quick test

# Generate HRF
hrf = ffs.get_canonical_hrf(stim_duration=2.0, tr=tr, device=device)

# Generate onsets for each run (different per run for cross-validation)
onsets_list = []
for run_idx in range(n_runs):
    # Set different seed for each run to get different onsets
    torch.manual_seed(run_idx + 42)
    onsets = ffs.generate_random_onsets(
        n_timepoints=n_timepoints,
        n_conditions=n_conditions,
        isi_mean=6,
        tr=tr,
        device=device
    )
    onsets_list.append(onsets)

# Define true betas
betas = [3.0, 4.0]  # Different signal for each condition

print(f"\nSimulating {n_runs} fMRI runs...")
print(f"  Conditions: {n_conditions}")
print(f"  Timepoints per run: {n_timepoints}")
print(f"  TR: {tr}s")
print(f"  Matrix size: {matrix_size}")

# Simulate multi-run experiment
data_list = ffs.simulate_fmri_experiment(
    n_runs=n_runs,
    onsets=onsets_list,
    betas=betas,
    hrf=hrf,
    tr=tr,
    n_timepoints=n_timepoints,
    matrix_size=matrix_size,
    noise_level=1.0,
    add_scanner_drift=True,
    device=device,
    verbose=True
)

# Save all outputs
metadata = {
    'betas': betas,
    'n_conditions': n_conditions,
    'noise_level': 1.0,
    'hrf_duration': len(hrf) * tr,
    'scanner_drift': True,
    'isi_mean': 6,
}

output_info = ffs.save_simulation_outputs(
    data_list=data_list,
    onsets_list=onsets_list,
    tr=tr,
    output_dir='./simulations',
    label='example_test',
    metadata=metadata,
    voxel_size=(2.0, 2.0, 2.0),
    verbose=True
)

print("\nOutput files:")
print(f"  Simulation folder: {output_info['output_dir']}")
print(f"  Onset files: {len(output_info['onset_files'])}")
for f in output_info['onset_files']:
    print(f"    - {f.name}")
print(f"  NIfTI files: {len(output_info['nifti_files'])}")
for f in output_info['nifti_files']:
    print(f"    - {f.name}")
print(f"  Metadata: {output_info['metadata_file'].name}")

print("\n✓ Done! You can now use these files with AFNI, FSL, SPM, etc.")
print(f"\nExample AFNI command:")
print(f"  cd {output_info['output_dir']}")
print(f"  3dDeconvolve -input run*.nii.gz \\")
print(f"    -polort 3 \\")
print(f"    -num_stimts {n_conditions} \\")
print(f"    -stim_times 1 onsets_condition1.txt 'BLOCK(2,1)' \\")
print(f"    -stim_times 2 onsets_condition2.txt 'BLOCK(2,1)' \\")
print(f"    -stim_label 1 condition1 \\")
print(f"    -stim_label 2 condition2 \\")
print(f"    -bucket stats_example.nii.gz")
