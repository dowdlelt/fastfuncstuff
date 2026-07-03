"""
fMRI simulation pipeline
Single and batch simulation modes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastfuncstuff.design.matrices import build_glm_design
from fastfuncstuff.io.afni import save_nifti
from fastfuncstuff.utils import get_device, print_device_info, to_tensor

from .noise import add_drift, generate_fmri_noise


def simulate_fmri_run(
    onsets: torch.Tensor,
    betas: torch.Tensor | list[float],
    hrf: torch.Tensor,
    tr: float,
    n_timepoints: int,
    matrix_size: tuple[int, int, int] = (100, 100, 10),
    noise_level: float = 1.0,
    baseline: float = 100.0,
    add_scanner_drift: bool = True,
    drift_amplitude: float = 0.5,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Simulate a single fMRI run

    Parameters
    ----------
    onsets : torch.Tensor
        (n_timepoints, n_conditions) binary onset matrix
    betas : torch.Tensor or list
        Beta coefficients for each condition. Can be:
        - (n_conditions,) same betas for all voxels
        - (n_voxels, n_conditions) different betas per voxel
        - list of scalars for simple case
    hrf : torch.Tensor
        (n_hrf_timepoints,) HRF to convolve with
    tr : float
        TR in seconds
    n_timepoints : int
        Total number of timepoints
    matrix_size : tuple
        (nx, ny, nz) spatial dimensions
    noise_level : float
        Noise amplitude (default: 1.0)
    baseline : float
        Baseline signal level (default: 100)
    add_scanner_drift : bool
        Add low-frequency scanner drift (default: True)
    drift_amplitude : float
        Amplitude of drift (default: 0.5)
    device : torch.device, optional
        Device for computation

    Returns
    -------
    data : torch.Tensor
        (nx, ny, nz, n_timepoints) simulated fMRI data
    """
    if device is None:
        device = get_device()

    onsets = to_tensor(onsets, device=device)
    hrf = to_tensor(hrf, device=device)

    nx, ny, nz = matrix_size
    n_voxels = nx * ny * nz
    n_conditions = onsets.shape[1] if onsets.ndim > 1 else 1

    # Convert betas to tensor
    if isinstance(betas, (list, tuple)):
        betas = torch.tensor(betas, device=device).float()

    if betas.ndim == 1:
        # Broadcast to all voxels
        betas = betas.unsqueeze(0).expand(n_voxels, n_conditions)
    elif betas.shape[0] != n_voxels:
        raise ValueError(f"Betas shape {betas.shape} doesn't match n_voxels {n_voxels}")

    # Build design matrix
    design = build_glm_design(onsets, hrf, n_timepoints, mode="assumed", device=device)

    # Generate signal: data = design @ betas.T
    # design: (n_timepoints, n_conditions)
    # betas: (n_voxels, n_conditions)
    signal = design @ betas.T  # (n_timepoints, n_voxels)
    signal = signal.T  # (n_voxels, n_timepoints)

    # Add baseline
    data = baseline + signal

    # Generate noise
    noise = torch.zeros(n_voxels, n_timepoints, device=device)

    # Generate noise per slice (more efficient than per voxel)
    for slice_idx in range(nz):
        slice_noise = generate_fmri_noise(
            tr, n_timepoints * tr, matrix_size=(nx, ny), normalize=True, device=device
        )
        # Reshape and assign
        slice_start = slice_idx * nx * ny
        slice_end = (slice_idx + 1) * nx * ny
        noise[slice_start:slice_end, :] = slice_noise.reshape(-1, n_timepoints) * noise_level

    data = data + noise

    # Add scanner drift if requested
    if add_scanner_drift:
        data = add_drift(data.T, amplitude=drift_amplitude, device=device).T

    # Reshape to 4D
    data = data.reshape(nx, ny, nz, n_timepoints)

    return data


def simulate_fmri_experiment(
    n_runs: int,
    onsets: torch.Tensor | list[torch.Tensor],
    betas: torch.Tensor | list[float],
    hrf: torch.Tensor,
    tr: float,
    n_timepoints: int | list[int],
    matrix_size: tuple[int, int, int] = (100, 100, 10),
    device: torch.device | None = None,
    verbose: bool = True,
    **kwargs,
) -> list[torch.Tensor]:
    """
    Simulate a multi-run fMRI experiment

    Parameters
    ----------
    n_runs : int
        Number of runs
    onsets : torch.Tensor or list of torch.Tensor
        Onsets for each run. If single tensor, same onsets used for all runs.
    betas : torch.Tensor or list
        Beta coefficients
    hrf : torch.Tensor
        HRF
    tr : float
        TR in seconds
    n_timepoints : int or list of int
        Number of timepoints per run
    matrix_size : tuple
        Spatial dimensions
    device : torch.device, optional
        Device for computation
    verbose : bool
        Print progress
    **kwargs : dict
        Additional arguments passed to simulate_fmri_run

    Returns
    -------
    data : list of torch.Tensor
        List of data tensors, one per run
    """
    if device is None:
        device = get_device()

    if verbose:
        print(f"Simulating {n_runs} fMRI runs...")
        print_device_info(device)

    # Handle single vs multiple onsets
    if not isinstance(onsets, list):
        onsets = [onsets] * n_runs

    if isinstance(n_timepoints, int):
        n_timepoints = [n_timepoints] * n_runs

    data_list = []

    for run_idx in range(n_runs):
        if verbose:
            print(f"  Run {run_idx + 1}/{n_runs}...")

        data = simulate_fmri_run(
            onsets[run_idx],
            betas,
            hrf,
            tr,
            n_timepoints[run_idx],
            matrix_size=matrix_size,
            device=device,
            **kwargs,
        )

        data_list.append(data)

    if verbose:
        print("Simulation complete!")

    return data_list


def create_parametric_voxels(
    matrix_size: tuple[int, int, int],
    n_conditions: int,
    hrf_library: torch.Tensor | None = None,
    beta_ranges: list[tuple[float, float]] | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create spatially organized voxels with varying betas and HRFs

    This mimics the MATLAB simulate_movietasks.m approach where:
    - X dimension: Different HRFs (e.g., 20 HRFs across 100 voxels)
    - Y dimension: Different beta ratios
    - Z dimension: Different noise levels

    Parameters
    ----------
    matrix_size : tuple
        (nx, ny, nz) spatial dimensions
    n_conditions : int
        Number of experimental conditions
    hrf_library : torch.Tensor, optional
        (n_hrfs, n_timepoints) HRF library
        If None, use canonical HRF for all voxels
    beta_ranges : list of tuple, optional
        List of (min, max) beta ranges for each condition
        If None, use default ranges
    device : torch.device, optional
        Device for computation

    Returns
    -------
    betas : torch.Tensor
        (n_voxels, n_conditions) beta coefficients
    hrf_indices : torch.Tensor
        (n_voxels,) HRF index for each voxel
    noise_levels : torch.Tensor
        (n_voxels,) noise level for each voxel
    """
    if device is None:
        device = get_device()

    nx, ny, nz = matrix_size
    n_voxels = nx * ny * nz

    # Default beta ranges
    if beta_ranges is None:
        beta_ranges = [(0, 5) for _ in range(n_conditions)]

    # Create spatial organization
    betas = torch.zeros(n_voxels, n_conditions, device=device)
    hrf_indices = torch.zeros(n_voxels, dtype=torch.long, device=device)
    noise_levels = torch.zeros(n_voxels, device=device)

    voxel_idx = 0

    # Z dimension: noise levels
    noise_steps = torch.linspace(0.5, 2.0, nz, device=device)

    # X dimension: HRFs
    n_hrfs = hrf_library.shape[0] if hrf_library is not None else 1
    hrf_voxels_per_block = nx // n_hrfs if n_hrfs > 0 else nx

    # Y dimension: beta patterns
    n_beta_patterns = 20  # Can be made configurable
    beta_voxels_per_block = ny // n_beta_patterns

    for z in range(nz):
        for x in range(nx):
            # Determine HRF for this X position
            hrf_idx = min(x // hrf_voxels_per_block, n_hrfs - 1) if n_hrfs > 1 else 0

            for y in range(ny):
                # Determine beta pattern for this Y position
                beta_pattern_idx = min(y // beta_voxels_per_block, n_beta_patterns - 1)

                # Create beta pattern (can be customized)
                for cond_idx in range(n_conditions):
                    beta_min, beta_max = beta_ranges[cond_idx]
                    # Vary betas across Y dimension
                    beta_val = beta_min + (beta_max - beta_min) * (
                        beta_pattern_idx / n_beta_patterns
                    )
                    betas[voxel_idx, cond_idx] = beta_val

                hrf_indices[voxel_idx] = hrf_idx
                noise_levels[voxel_idx] = noise_steps[z]

                voxel_idx += 1

    return betas, hrf_indices, noise_levels


def simulate_batch_experiments(
    n_experiments: int, sim_config: dict, device: torch.device | None = None, verbose: bool = True
) -> list[dict]:
    """
    Simulate multiple experiments in batch (for statistical power analysis, etc.)

    Parameters
    ----------
    n_experiments : int
        Number of independent experiments to simulate
    sim_config : dict
        Configuration dictionary containing simulation parameters:
        - n_runs, tr, n_timepoints, matrix_size, n_conditions, etc.
    device : torch.device, optional
        Device for computation
    verbose : bool
        Print progress

    Returns
    -------
    experiments : list of dict
        List of experiment dictionaries, each containing:
        - 'data': List of data tensors (one per run)
        - 'onsets': Onsets used
        - 'betas': True betas
        - 'hrf': HRF used
    """
    if device is None:
        device = get_device()

    if verbose:
        print(f"Simulating {n_experiments} experiments in batch...")
        print_device_info(device)

    experiments = []

    for exp_idx in range(n_experiments):
        if verbose and exp_idx % max(1, n_experiments // 10) == 0:
            print(f"  Experiment {exp_idx + 1}/{n_experiments}...")

        # Generate this experiment
        # (Implementation would extract from sim_config and call simulate_fmri_experiment)
        # This is a template - full implementation depends on specific needs

        exp_data = {
            "id": exp_idx,
            # Add simulated data here
        }

        experiments.append(exp_data)

    if verbose:
        print("Batch simulation complete!")

    return experiments


def write_afni_onset_files(
    onsets_list: list[torch.Tensor] | torch.Tensor,
    tr: float,
    output_dir: Path,
    prefix: str = "onsets",
) -> list[Path]:
    """
    Write AFNI-compatible onset timing files

    AFNI format: Space-separated onset times in seconds, one row per run
    One file per condition: onsets_condition1.txt, onsets_condition2.txt, etc.

    Parameters
    ----------
    onsets_list : list of torch.Tensor or torch.Tensor
        Either:
        - List of onset matrices (one per run): [(n_timepoints, n_conditions), ...]
        - Single onset matrix: (n_timepoints, n_conditions)
    tr : float
        TR in seconds (to convert timepoints to seconds)
    output_dir : Path
        Directory to save onset files
    prefix : str
        Prefix for onset files (default: "onsets")

    Returns
    -------
    onset_files : list of Path
        Paths to created onset files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle single onset matrix vs list
    if isinstance(onsets_list, torch.Tensor):
        onsets_list = [onsets_list]

    # Convert to numpy
    onsets_np_list = [o.cpu().numpy() if isinstance(o, torch.Tensor) else o for o in onsets_list]

    n_runs = len(onsets_np_list)
    n_conditions = onsets_np_list[0].shape[1] if onsets_np_list[0].ndim > 1 else 1

    onset_files = []

    # Write one file per condition
    for cond_idx in range(n_conditions):
        filename = output_dir / f"{prefix}_condition{cond_idx + 1}.txt"

        with open(filename, "w") as f:
            for run_idx, onsets in enumerate(onsets_np_list):
                # Extract onsets for this condition (binary onset matrix)
                if onsets.ndim > 1:
                    onset_timepoints = np.where(onsets[:, cond_idx] > 0)[0]
                else:
                    onset_timepoints = np.where(onsets > 0)[0]

                # Convert to seconds
                onset_seconds = onset_timepoints * tr

                # Write space-separated
                if len(onset_seconds) > 0:
                    onset_str = " ".join([f"{t:.2f}" for t in onset_seconds])
                else:
                    onset_str = "*"  # AFNI convention for no events

                f.write(onset_str)
                if run_idx < n_runs - 1:
                    f.write("\n")

        onset_files.append(filename)

    return onset_files


def write_nifti_files(
    data_list: list[torch.Tensor],
    tr: float,
    output_dir: Path,
    prefix: str = "run",
    affine: np.ndarray | None = None,
    voxel_size: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> list[Path]:
    """
    Write fMRI data as nii.gz files using nibabel

    Parameters
    ----------
    data_list : list of torch.Tensor
        List of data tensors (one per run): [(nx, ny, nz, n_timepoints), ...]
    tr : float
        TR in seconds
    output_dir : Path
        Directory to save nifti files
    prefix : str
        Prefix for run files (default: "run")
    affine : np.ndarray, optional
        4x4 affine matrix for nifti header. If None, creates simple affine.
    voxel_size : tuple
        (x, y, z) voxel size in mm (default: 2.0 x 2.0 x 2.0)

    Returns
    -------
    nifti_files : list of Path
        Paths to created nifti files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create default affine if not provided
    if affine is None:
        affine = np.eye(4)
        affine[0, 0] = voxel_size[0]
        affine[1, 1] = voxel_size[1]
        affine[2, 2] = voxel_size[2]

    nifti_files = []

    for run_idx, data in enumerate(data_list):
        # Convert to numpy
        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data

        # Save nifti with TR in header
        filename = output_dir / f"{prefix}{run_idx + 1:02d}.nii.gz"
        save_nifti(data_np.astype(np.float32), output_path=filename, affine=affine, tr=tr)
        nifti_files.append(filename)

    return nifti_files


def save_simulation_outputs(
    data_list: list[torch.Tensor],
    onsets_list: list[torch.Tensor] | torch.Tensor,
    tr: float,
    output_dir: str | Path,
    label: str,
    metadata: dict[str, Any] | None = None,
    affine: np.ndarray | None = None,
    voxel_size: tuple[float, float, float] = (2.0, 2.0, 2.0),
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Save all simulation outputs to organized folder structure

    Creates folder: output_dir/simulation_{label}/
    Contains:
    - onset files (AFNI format)
    - nifti files (one per run)
    - metadata.txt

    Parameters
    ----------
    data_list : list of torch.Tensor
        List of data tensors (one per run)
    onsets_list : list of torch.Tensor or torch.Tensor
        Onset matrices (one per run or single)
    tr : float
        TR in seconds
    output_dir : str or Path
        Base output directory
    label : str
        Label for this simulation (used in folder name)
    metadata : dict, optional
        Additional metadata to save (betas, HRF params, noise params, etc.)
    affine : np.ndarray, optional
        Affine matrix for nifti files
    voxel_size : tuple
        Voxel size in mm
    verbose : bool
        Print progress

    Returns
    -------
    output_info : dict
        Dictionary containing:
        - 'output_dir': Path to simulation folder
        - 'onset_files': List of onset file paths
        - 'nifti_files': List of nifti file paths
        - 'metadata_file': Path to metadata file
    """
    # Create simulation folder
    output_dir = Path(output_dir)
    sim_dir = output_dir / f"simulation_{label}"
    sim_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSaving simulation outputs to: {sim_dir}")

    # Write onset files
    if verbose:
        print("  Writing AFNI onset timing files...")
    onset_files = write_afni_onset_files(onsets_list, tr, sim_dir, prefix="onsets")

    # Write nifti files
    if verbose:
        print("  Writing nifti files...")
    nifti_files = write_nifti_files(
        data_list, tr, sim_dir, prefix="run", affine=affine, voxel_size=voxel_size
    )

    # Write metadata
    metadata_file = sim_dir / "metadata.txt"
    if verbose:
        print("  Writing metadata...")

    with open(metadata_file, "w") as f:
        f.write(f"Simulation Label: {label}\n")
        f.write(f"TR: {tr} sec\n")
        f.write(f"Number of runs: {len(data_list)}\n")
        f.write(f"Voxel size: {voxel_size[0]} x {voxel_size[1]} x {voxel_size[2]} mm\n")

        if len(data_list) > 0:
            data_shape = data_list[0].shape
            f.write(f"Data shape per run: {data_shape}\n")
            f.write(f"Number of timepoints: {data_shape[-1]}\n")
            f.write(f"Matrix size: {data_shape[0]} x {data_shape[1]} x {data_shape[2]}\n")

        if metadata is not None:
            f.write("\nAdditional Parameters:\n")
            for key, value in metadata.items():
                # Handle tensors
                if isinstance(value, torch.Tensor):
                    if value.numel() < 20:  # Small tensors
                        value = value.cpu().numpy().tolist()
                    else:  # Large tensors
                        value = f"Tensor{tuple(value.shape)}"
                f.write(f"  {key}: {value}\n")

    if verbose:
        print(f"  ✓ {len(onset_files)} onset files")
        print(f"  ✓ {len(nifti_files)} nifti files")
        print("  ✓ metadata file")
        print("\nSimulation outputs saved successfully!")

    return {
        "output_dir": sim_dir,
        "onset_files": onset_files,
        "nifti_files": nifti_files,
        "metadata_file": metadata_file,
    }
