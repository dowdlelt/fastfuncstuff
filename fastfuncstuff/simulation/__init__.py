"""Simulation: noise generation, fMRI experiment simulation, design metrics."""

from fastfuncstuff.simulation.core import (
    create_parametric_voxels,
    save_simulation_outputs,
    simulate_batch_experiments,
    simulate_fmri_experiment,
    simulate_fmri_run,
    write_afni_onset_files,
    write_nifti_files,
)
from fastfuncstuff.simulation.noise import (
    add_drift,
    add_motion_artifacts,
    estimate_noise_parameters_from_data,
    estimate_sfnr,
    generate_ar1_noise,
    generate_ar_noise,
    generate_arma_noise,
    generate_fmri_noise,
    generate_fmri_noise_batch,
)

__all__ = [
    "simulate_fmri_run",
    "simulate_fmri_experiment",
    "create_parametric_voxels",
    "simulate_batch_experiments",
    "save_simulation_outputs",
    "write_afni_onset_files",
    "write_nifti_files",
    "generate_fmri_noise",
    "generate_fmri_noise_batch",
    "generate_ar1_noise",
    "generate_ar_noise",
    "generate_arma_noise",
    "add_drift",
    "add_motion_artifacts",
    "estimate_noise_parameters_from_data",
    "estimate_sfnr",
]
