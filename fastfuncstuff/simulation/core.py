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

from .noise import add_drift, generate_thermal_physio_noise


def simulate_fmri_run(
    onsets: torch.Tensor,
    betas: torch.Tensor | list[float],
    hrf: torch.Tensor,
    tr: float,
    n_timepoints: int,
    matrix_size: tuple[int, int, int] = (100, 100, 10),
    noise_level: float | torch.Tensor = 1.0,
    baseline: float = 100.0,
    add_scanner_drift: bool = True,
    drift_amplitude: float = 0.5,
    device: torch.device | None = None,
    phys_fraction: float | torch.Tensor = 0.5,
    tau: float | torch.Tensor = 6.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Simulate a single fMRI run on a TR-grid onset matrix

    For events in seconds (sub-TR onsets, durations) use :func:`simulate_bold`,
    which shares this noise model. Both draw noise from
    :func:`~.noise.generate_thermal_physio_noise` with tSNR = baseline /
    noise_level.

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
    noise_level : float or tensor
        Noise std, scalar or one per voxel (default 1.0): tSNR = baseline / noise_level
    baseline : float
        Baseline signal level (default: 100)
    add_scanner_drift : bool
        Add low-frequency scanner drift (default: True)
    drift_amplitude : float
        Drift std as a fraction of each voxel's noise std (default: 0.5)
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

    # noise_level may be a scalar or one value per voxel, so that the per-voxel
    # levels create_parametric_voxels returns are actually usable.
    scale_per_voxel = to_tensor(noise_level, device=device).flatten().double()
    if scale_per_voxel.numel() == 1:
        scale_per_voxel = scale_per_voxel.expand(n_voxels)
    elif scale_per_voxel.numel() != n_voxels:
        raise ValueError(
            f"noise_level must be a scalar or have one value per voxel "
            f"({n_voxels}); got {scale_per_voxel.numel()}"
        )
    # One noise model for every simulation path. This used to call the 1/f
    # spectral generator slice by slice, a second model with no white floor and
    # no tSNR parameter. Voxels come out in the (nx, ny, nz) order of the final
    # reshape, which the old per-slice loop had to reconstruct by hand.
    noise = generate_thermal_physio_noise(
        n_timepoints,
        tr,
        baseline / scale_per_voxel,
        phys_fraction,
        tau,
        baseline=baseline,
        n_voxels=n_voxels,
        device=device,
        generator=generator,
    ).T  # (n_voxels, n_timepoints)

    # Drift is scaled to the noise, not to the data. add_drift sizes it from the
    # std of whatever it is handed, and handed the data that std includes the
    # task signal -- so a strongly active voxel got proportionally more drift
    # (residual SD 1.23 vs 1.04 at beta=5 vs 0 under a cubic detrend), coupling a
    # nuisance to the very effect being simulated.
    if add_scanner_drift:
        noise = add_drift(noise.T, amplitude=drift_amplitude, device=device, generator=generator).T

    data = data + noise

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
    n_beta_patterns: int | None = None,
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

    # Z dimension: noise levels
    noise_steps = torch.linspace(0.5, 2.0, nz, device=device)

    # X dimension: HRFs
    n_hrfs = hrf_library.shape[0] if hrf_library is not None else 1

    # Y dimension: beta patterns. Capped at ny so that a volume with fewer than
    # n_beta_patterns rows still gets one pattern per row instead of dividing by
    # a zero block size -- ny // 20 is 0 for any ny < 20, which made every test
    # volume (a 4x4x2, say) raise ZeroDivisionError here.
    if n_beta_patterns is None:
        n_beta_patterns = min(20, ny)
    n_beta_patterns = max(1, min(n_beta_patterns, ny))

    # Spread the blocks across the axis by proportion rather than by an integer
    # block size, so n_hrfs > nx degrades to "as many distinct HRFs as fit"
    # instead of dividing by zero.
    hrf_of_x = (torch.arange(nx, device=device) * n_hrfs // max(nx, 1)).clamp(max=n_hrfs - 1)
    pattern_of_y = (torch.arange(ny, device=device) * n_beta_patterns // max(ny, 1)).clamp(
        max=n_beta_patterns - 1
    )

    # Voxel index must match the (nx, ny, nz) reshape that simulate_fmri_run
    # applies to its output: x*ny*nz + y*nz + z. The original loop walked z, x, y
    # and wrote sequentially, laying voxels out z-major -- so the "HRF varies
    # along X, noise along Z" organisation this function documents did not
    # survive the round trip into a volume.
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                voxel_idx = x * ny * nz + y * nz + z
                for cond_idx in range(n_conditions):
                    beta_min, beta_max = beta_ranges[cond_idx]
                    beta_val = beta_min + (beta_max - beta_min) * (
                        int(pattern_of_y[y]) / n_beta_patterns
                    )
                    betas[voxel_idx, cond_idx] = beta_val

                hrf_indices[voxel_idx] = int(hrf_of_x[x]) if n_hrfs > 1 else 0
                noise_levels[voxel_idx] = noise_steps[z]

    return betas, hrf_indices, noise_levels


def write_timing_files(
    onsets: list[list[np.ndarray | list[float]]],
    conditions: list[str],
    output_dir: str | Path,
    prefix: str = "",
) -> list[Path]:
    """AFNI timing files: one per condition, one row per run, onsets in seconds.

    ``onsets[condition][run]``. An empty run is written as ``*``. These read
    back with io.afni.read_afni_onset_files, e.g. as ffs_simulate -events.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cond, runs in zip(conditions, onsets, strict=True):
        path = output_dir / f"{prefix}{cond}.txt"
        with open(path, "w") as f:
            for run in runs:
                run = np.asarray(run, dtype=float)
                f.write((" ".join(f"{t:.3f}" for t in run) if run.size else "*") + "\n")
        paths.append(path)
    return paths


def write_afni_onset_files(
    onsets_list: list[torch.Tensor] | torch.Tensor,
    tr: float,
    output_dir: Path,
    prefix: str = "onsets",
) -> list[Path]:
    """AFNI timing files from binary TR-grid onset matrices (one per run).

    Converts each ``(n_timepoints, n_conditions)`` matrix to onset times and
    writes ``{prefix}_condition{k}.txt`` through :func:`write_timing_files`.
    """
    if isinstance(onsets_list, torch.Tensor):
        onsets_list = [onsets_list]
    mats = [np.asarray(o.cpu() if isinstance(o, torch.Tensor) else o) for o in onsets_list]
    mats = [m[:, None] if m.ndim == 1 else m for m in mats]
    n_cond = mats[0].shape[1]
    per_cond = [[np.flatnonzero(m[:, k] > 0) * tr for m in mats] for k in range(n_cond)]
    names = [f"condition{k + 1}" for k in range(n_cond)]
    return write_timing_files(per_cond, names, output_dir, prefix=f"{prefix}_")


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


def default_microtime_dt(tr: float) -> float:
    """Largest step <= 0.05 s dividing the TR: onsets land within 25 ms of their request."""
    from fastfuncstuff.design.matrices import commensurate_microtime_dt

    return commensurate_microtime_dt(tr, 0.05)


def hrfs_from_spec(
    spec: str, microtime_dt: float, device: torch.device | None = None
) -> list[tuple[str, torch.Tensor]]:
    """``spmg1``, ``lib:K`` (one HRF of the 20-HRF library) or ``lib:all``.

    Returns (label, (1, n_microtime) response) pairs. The library is the
    GLMsingle-style set in design/getcanonicalhrflibrary.tsv (peaks ~2.7-5.7 s).
    """
    from fastfuncstuff.design.hrf import get_spmg1_hrf, load_canonical_hrf_library

    text = spec.strip().lower()
    if text == "spmg1":
        return [("spmg1", get_spmg1_hrf(microtime_dt=microtime_dt, device=device).reshape(1, -1))]
    if text.startswith("lib:"):
        lib = load_canonical_hrf_library(microtime_dt=microtime_dt, device=device)
        which = text[4:]
        idx = range(lib.shape[0]) if which == "all" else [int(which)]
        if any(not 0 <= i < lib.shape[0] for i in idx):
            raise ValueError(f"{spec!r}: the library has HRFs 0-{lib.shape[0] - 1}")
        return [(f"lib:{i}", lib[i].reshape(1, -1)) for i in idx]
    raise ValueError(f"cannot parse HRF {spec!r}: use spmg1, lib:K or lib:all")


def build_task_design(
    onsets: list[list[np.ndarray | list[float]]],
    durations: list[float],
    tr: float,
    n_timepoints_per_run: list[int],
    hrf_bases: torch.Tensor | None = None,
    microtime_dt: float | None = None,
    delay: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Unit-peak task regressors from events in seconds, via the GLM tools' builder.

    ``delay`` shifts every onset (seconds) -- the truth for an HRF-latency
    mismatch, fitted with the nominal onsets.
    """
    from fastfuncstuff.design.hrf import get_spmg1_hrf
    from fastfuncstuff.design.matrices import build_event_design_microtime

    if device is None:
        device = get_device()
    if microtime_dt is None:
        microtime_dt = default_microtime_dt(tr)
    if hrf_bases is None:
        hrf_bases = get_spmg1_hrf(microtime_dt=microtime_dt, device=device)
    design = build_event_design_microtime(
        all_onsets=[[np.asarray(r, dtype=np.float64) + delay for r in cond] for cond in onsets],
        durations=list(durations),
        hrf_bases=hrf_bases,
        n_timepoints_per_run=list(n_timepoints_per_run),
        tr=tr,
        microtime_dt=microtime_dt,
        device=device,
    )
    assert isinstance(design, torch.Tensor)
    return design.to(torch.float32)


def simulate_bold(
    onsets: list[list[np.ndarray | list[float]]],
    durations: list[float],
    tr: float,
    n_timepoints_per_run: list[int] | int,
    amplitude_psc: float | list[float] | torch.Tensor | np.ndarray,
    tsnr: float | torch.Tensor | np.ndarray,
    phys_fraction: float | torch.Tensor | np.ndarray = 0.5,
    tau: float | torch.Tensor | np.ndarray = 6.0,
    n_voxels: int | None = None,
    baseline: float = 100.0,
    hrf_bases: torch.Tensor | None = None,
    microtime_dt: float | None = None,
    drift_amplitude: float = 0.0,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
    arma: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Simulate BOLD timeseries from events in seconds, at any TR, with a known noise model.

    Timing is not tied to the TR grid: events at whole seconds sampled at
    TR = 1.25, say, go through the same microtime design builder the GLM tools
    fit with (:func:`build_event_design_microtime`), so the simulated response
    and the analysis model are built identically.

    Signal: ``baseline * amplitude_psc / 100 * regressor``, with regressors at
    AFNI's unit-peak convention, so ``amplitude_psc`` is the peak percent signal
    change of an isolated event of that condition. Noise:
    :func:`generate_thermal_physio_noise`, generated independently per run.

    Parameters
    ----------
    onsets : list (per condition) of lists (per run) of onset times in seconds
    durations : list of float, one per condition (0 = impulse)
    tr : float
    n_timepoints_per_run : list of int, or one int for a single run
    amplitude_psc : (n_conditions,) or (n_voxels, n_conditions)
    tsnr, phys_fraction, tau : scalar or (n_voxels,); tau in seconds
    hrf_bases : (n_bases, n_microtime) response at ``microtime_dt``; default SPMG1.
        With several bases, amplitude_psc must give one value per column.
    microtime_dt : float, optional
        Defaults to the largest step <= 0.05 s that divides the TR, so onsets
        land within 25 ms of where they were asked for.
    drift_amplitude : float
        Drift std as a fraction of each voxel's noise std (0 = none).
    arma : (a, b), optional
        Generate AFNI-form ARMA(1,1) noise directly instead of white + OU; valid
        only at the TR it was measured at (see generate_thermal_physio_noise).

    Returns
    -------
    dict with 'data' (n_voxels, n_timepoints), 'signal', 'noise' (same shape),
    'design' (n_timepoints, n_columns), 'run_starts', 'microtime_dt', and the
    per-voxel 'tsnr', 'phys_fraction', 'tau', 'arma_a', 'arma_b'.
    """
    from .noise import generate_thermal_physio_noise, ou_to_arma11

    if device is None:
        device = get_device()
    if isinstance(n_timepoints_per_run, int):
        n_timepoints_per_run = [n_timepoints_per_run]
    if microtime_dt is None:
        microtime_dt = default_microtime_dt(tr)
    design = build_task_design(
        onsets, durations, tr, n_timepoints_per_run, hrf_bases, microtime_dt, device=device
    )
    n_columns = design.shape[1]

    amps = torch.as_tensor(amplitude_psc, dtype=torch.float32, device=device)
    if amps.ndim == 0:
        amps = amps.expand(n_columns)
    if n_voxels is None:
        sizes = [
            torch.as_tensor(v).numel()
            for v in (tsnr, phys_fraction, tau)
            if np.ndim(v) or torch.is_tensor(v)
        ]
        n_voxels = max(sizes + ([amps.shape[0]] if amps.ndim == 2 else []), default=1)
    if amps.ndim == 1:
        amps = amps.unsqueeze(0).expand(n_voxels, -1)
    if amps.shape != (n_voxels, n_columns):
        raise ValueError(
            f"amplitude_psc must be ({n_columns},) or ({n_voxels}, {n_columns}); got "
            f"{tuple(amps.shape)}"
        )

    signal = (baseline / 100.0) * (amps @ design.T)  # (n_voxels, n_timepoints)

    noise_runs = []
    for n_run in n_timepoints_per_run:
        run_noise = generate_thermal_physio_noise(
            n_run,
            tr,
            tsnr,
            phys_fraction,
            tau,
            baseline=baseline,
            n_voxels=n_voxels,
            device=device,
            generator=generator,
            arma=arma,
        )
        if drift_amplitude > 0:
            run_noise = add_drift(
                run_noise, amplitude=drift_amplitude, device=device, generator=generator
            )
        noise_runs.append(run_noise)
    noise = torch.cat(noise_runs, dim=0).T

    run_starts = np.concatenate([[0], np.cumsum(n_timepoints_per_run)[:-1]]).astype(int)
    if arma is not None:
        a = torch.full((n_voxels,), float(arma[0]), dtype=torch.float64)
        b = torch.full((n_voxels,), float(arma[1]), dtype=torch.float64)
    else:
        a, b = ou_to_arma11(tr, torch.as_tensor(tau).expand(n_voxels), phys_fraction)
    return {
        "data": baseline + signal + noise,
        "signal": signal,
        "noise": noise,
        "design": design,
        "run_starts": run_starts.tolist(),
        "microtime_dt": microtime_dt,
        "tsnr": torch.as_tensor(tsnr).expand(n_voxels),
        "phys_fraction": torch.as_tensor(phys_fraction).expand(n_voxels),
        "tau": torch.as_tensor(tau).expand(n_voxels),
        "arma_a": a,
        "arma_b": b,
    }
