"""
Shared utility functions for CLI tools.

This module contains common utilities used across multiple CLI tools to avoid
code duplication. Functions include argument parsing, CV strategy parsing,
data loading, and output formatting.
"""
from __future__ import annotations

import glob as glob_module
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def parse_input_files(input_arg: Union[str, list[str]]) -> list[str]:
    """
    Parse input files from command line arguments.

    Supports multiple input formats for backwards compatibility:
    - Single file: "/path/to/file.nii.gz"
    - Multiple files (list): ["/path/run1.nii.gz", "/path/run2.nii.gz"]
    - Multiple files (string): "/path/run1.nii.gz /path/run2.nii.gz"
    - Glob patterns: ["run*.nii.gz"] or "run*.nii.gz"

    Parameters
    ----------
    input_arg : str or list of str
        Input file argument from argparse nargs='+'

    Returns
    -------
    list of str
        Expanded and validated list of file paths

    Notes
    -----
    - Glob patterns are expanded and sorted for consistent ordering
    - Non-glob patterns are passed through as-is
    - Exits with error if any file doesn't exist
    """
    # Handle both list (from nargs='+') and string (old behavior)
    if isinstance(input_arg, str):
        # Old behavior: space-separated string in quotes
        input_arg = input_arg.strip().strip('"').strip("'")
        input_list = input_arg.split()
    else:
        # New behavior: list from nargs='+'
        input_list = input_arg

    # Expand globs and collect files
    files = []
    for pattern in input_list:
        # Try glob expansion
        matches = glob_module.glob(pattern)
        if matches:
            # Sort for consistent ordering
            files.extend(sorted(matches))
        else:
            # Not a glob pattern, use as-is
            files.append(pattern)

    # Validate files exist
    for f in files:
        if not Path(f).exists():
            print(f"ERROR: Input file not found: {f}")
            sys.exit(1)

    return files


def parse_cv_strategy(cv_str: str) -> Union[int, float]:
    """
    Parse cross-validation strategy string into int or float.

    Supported formats:
    - 'loro' or 'loo' or '1' → Leave-one-run-out (returns 1)
    - Float in (0, 1) → Split fraction (e.g., '0.5' for 50% train)
    - Int > 1 → Leave-N-out (e.g., '2' for leave-two-out)

    Parameters
    ----------
    cv_str : str
        CV strategy string from command line

    Returns
    -------
    int or float
        Parsed CV strategy:
        - 1 for leave-one-run-out
        - float in (0, 1) for split fraction
        - int > 1 for leave-N-out

    Notes
    -----
    Exits with error if the string cannot be parsed.
    """
    cv_str = cv_str.lower().strip()

    if cv_str in ["loro", "loo"]:
        return 1  # Leave-one-run-out

    try:
        # Try parsing as float first
        val = float(cv_str)
        if val == int(val) and val > 1:
            return int(val)  # Leave-N-out
        elif 0 < val < 1:
            return val  # Split fraction
        elif val == 1:
            return 1  # LORO
        else:
            print(f"ERROR: Invalid cv_strategy value: {cv_str}")
            print("  Must be 'loro', int > 0, or float in (0, 1)")
            sys.exit(1)
    except ValueError:
        print(f"ERROR: Could not parse cv_strategy: {cv_str}")
        sys.exit(1)


def print_cli_header(tool_name: str, subtitle: str = ""):
    """
    Print standardized CLI header with timestamp.

    Parameters
    ----------
    tool_name : str
        Name of the CLI tool (e.g., "3dDenoisefast")
    subtitle : str, optional
        Additional subtitle text to display
    """
    print("=" * 70)
    print(f"{tool_name}")
    if subtitle:
        print(subtitle)
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def estimate_device_strategy(
    n_voxels: int,
    n_timepoints_total: int,
    device: torch.device,
    force_cpu: bool = False,
    gpu_threshold_gb: float = 4.0,
) -> bool:
    """
    Estimate whether to keep data on CPU based on dataset size and GPU memory.

    This is a convenience wrapper that delegates to memory.estimate_keep_on_cpu.
    The memory module contains the canonical implementation.

    Parameters
    ----------
    n_voxels : int
        Number of voxels (after masking, if applicable)
    n_timepoints_total : int
        Total number of timepoints across all runs
    device : torch.device
        Target compute device
    force_cpu : bool, default=False
        Force CPU storage regardless of data size
    gpu_threshold_gb : float, default=4.0
        Size threshold in GB for GPU memory

    Returns
    -------
    keep_on_cpu : bool
        True if data should be kept on CPU, False if direct GPU loading is safe
    """
    from fastfuncsim.memory import estimate_keep_on_cpu

    return estimate_keep_on_cpu(
        n_voxels=n_voxels,
        n_timepoints_total=n_timepoints_total,
        device=device,
        force_cpu=force_cpu,
        data_threshold_gb=gpu_threshold_gb,
    )


@dataclass
class LoadResult:
    """Result from load_and_preprocess_runs()"""
    data: torch.Tensor              # (n_voxels, n_timepoints) fMRI data
    run_starts: list[int]           # Starting timepoint for each run
    affine: np.ndarray              # Affine matrix for NIfTI files
    volume_shape: tuple             # Shape of 3D volume
    voxel_sizes: tuple             # Voxel dimensions in mm
    tr: float                       # Repetition time in seconds
    mask: Optional[np.ndarray]      # Brain mask (original 3D)
    mask_flat: Optional[np.ndarray]  # Flattened mask (1D bool)
    n_voxels: int                   # Number of voxels
    n_timepoints: int               # Total timepoints
    n_runs: int                     # Number of runs
    keep_on_cpu: bool               # Whether data is stored on CPU
    scale_info: Optional[dict]      # Scaling info if do_scale was True
    violations_mask: Optional[torch.Tensor]  # Scaling violations if do_scale


def load_and_preprocess_runs(
    input_files: list[str],
    tr: Optional[float] = None,
    mask_file: Optional[str] = None,
    blur_fwhm: Optional[float] = None,
    do_scale: bool = False,
    device: torch.device = torch.device("cpu"),
    force_cpu: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> LoadResult:
    """
    Load and preprocess fMRI data from multiple runs.

    This is a unified data loading pipeline that handles:
    - Metadata extraction (affine, volume_shape, voxel_sizes, TR)
    - Optional Gaussian blur (applied per-run before concatenation)
    - Optional masking
    - Optional scaling to percent signal change
    - Automatic device strategy (CPU vs GPU) based on data size

    Parameters
    ----------
    input_files : list of str
        List of input NIfTI file paths
    tr : float, optional
        Repetition time in seconds. If None, attempts to read from first file header.
    mask_file : str, optional
        Path to brain mask NIfTI file
    blur_fwhm : float, optional
        FWHM of Gaussian blur in mm. If None, no blur applied.
    do_scale : bool, default=False
        Whether to scale data to percent signal change
    device : torch.device, default=torch.device("cpu")
        Target compute device
    force_cpu : bool, default=False
        Force data to stay on CPU regardless of size
    dry_run : bool, default=False
        If True, load only first run and generate synthetic data for remaining runs.
        Much faster for testing pipelines. Results will be nonsensical but useful for
        verifying code logic and performance.
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    LoadResult
        Dataclass containing:
        - data: torch.Tensor (n_voxels, n_timepoints) preprocessed data
        - run_starts: list[int] starting timepoint for each run
        - affine: np.ndarray affine matrix
        - volume_shape: tuple 3D volume dimensions
        - voxel_sizes: tuple voxel dimensions in mm
        - tr: float repetition time
        - mask: np.ndarray or None original 3D mask
        - mask_flat: np.ndarray or None flattened 1D boolean mask
        - n_voxels: int number of voxels
        - n_timepoints: int total timepoints
        - n_runs: int number of runs
        - keep_on_cpu: bool whether data is on CPU
        - scale_info: dict or None scaling information if do_scale
        - violations_mask: torch.Tensor or None scaling violations if do_scale

    Notes
    -----
    - Blur is applied per-run before concatenation for memory efficiency
    - Scaling is applied after concatenation for proper percent signal calculation
    - Device strategy is automatically determined based on data size vs GPU memory
    """
    try:
        from tqdm import tqdm

        from fastfuncsim.afni_io import load_afni_mask, load_and_concatenate_runs, load_nifti
        from fastfuncsim.utils import gaussian_blur_3d, scale_to_percent_signal
    except ImportError as e:
        print(f"ERROR: Could not import required modules: {e}")
        sys.exit(1)

    if verbose:
        print("\n" + "=" * 70)
        print("📂 Loading fMRI Data")
        print("=" * 70)

    # Load metadata from first file
    first_img = load_nifti(input_files[0])
    affine = np.array(first_img.affine) if hasattr(first_img, "affine") else np.eye(4)
    volume_shape = tuple(first_img.shape[:3]) if hasattr(first_img, "shape") else (0, 0, 0)

    # Get voxel sizes and TR from header (using get_zooms() which reads pixdim correctly)
    zooms = first_img.header.get_zooms()
    voxel_sizes = tuple(float(z) for z in zooms[:3])  # First 3 are spatial dimensions in mm

    # Get TR
    if tr is None:
        if len(zooms) > 3 and zooms[3] > 0:
            tr = float(zooms[3])
        else:
            print("ERROR: Could not determine TR from header. Please specify with -tr")
            sys.exit(1)

    if verbose:
        print(f"  Volume shape: {volume_shape}")
        print(f"  Voxel sizes: {voxel_sizes} mm")
        print(f"  TR: {tr} s")
        print(f"  Runs: {len(input_files)}")

    # Load mask if provided
    mask = None
    mask_flat = None
    if mask_file:
        mask = load_afni_mask(mask_file)
        mask_flat = mask.flatten().astype(bool)
        if verbose:
            print(f"  Mask: {mask_file} ({mask.sum():,} voxels)")

    # Determine number of voxels for memory estimation
    n_voxels = mask_flat.sum() if mask_flat is not None else int(np.prod(volume_shape))

    # ======================================================================
    # DRY RUN MODE: Skip loading data, just read header and generate synthetic
    # ======================================================================
    if dry_run:
        from fastfuncsim.utils import generate_synthetic_runs

        if verbose:
            print("\n" + "=" * 70)
            print("🎭 DRY RUN MODE - Fast Pipeline Testing")
            print("=" * 70)
            print("  Reading header only, generating synthetic data...")

        # Read only the header (no data loading) using nibabel
        import nibabel as nib
        first_img = nib.load(input_files[0])
        run_length = first_img.shape[3] if len(first_img.shape) > 3 else first_img.shape[0]

        if verbose:
            print(f"  Run length: {run_length} TRs")
            print(f"  Generating {len(input_files)} runs of synthetic data...")

        # Generate synthetic data for ALL runs (including first one)
        # No need to load any real data - just use the shape info
        data = generate_synthetic_runs(
            first_run_data=None,  # No real data needed
            n_runs_total=len(input_files),
            run_length=run_length,
            n_voxels=n_voxels,
            verbose=verbose,
        )

        # Build run_starts
        run_starts = list(range(0, len(input_files) * run_length, run_length))

        # Always keep on CPU for dry run (fastest)
        keep_on_cpu = True

        # Scale if requested (apply to synthetic data too)
        scale_info = None
        violations_mask = None
        if do_scale:
            if verbose:
                print("\n  Scaling to percent signal change...")
            data, violations_mask, scale_info = scale_to_percent_signal(
                data=data,
                run_starts=run_starts,
                max_scale=200.0,
                verbose=verbose,
            )

        # Create result
        result = LoadResult(
            data=data,
            run_starts=run_starts,
            affine=affine,
            volume_shape=volume_shape,
            voxel_sizes=voxel_sizes,
            tr=tr,
            mask=mask,
            mask_flat=mask_flat,
            n_voxels=n_voxels,
            n_timepoints=data.shape[1],
            n_runs=len(input_files),
            keep_on_cpu=keep_on_cpu,
            scale_info=scale_info,
            violations_mask=violations_mask,
        )

        if verbose:
            print(f"\n  Data shape: {data.shape} (n_voxels × n_timepoints)")
            print(f"  Device: CPU (dry run)")
            print("  ⚠️  Results are from synthetic data - for testing only!")
            print("=" * 70 + "\n")

        return result

    # Estimate total timepoints
    total_timepoints = 0
    for f in input_files:
        img = load_nifti(f)
        if len(img.shape) > 3:
            total_timepoints += img.shape[3]
        else:
            total_timepoints += img.shape[0]

    # Determine device strategy
    keep_on_cpu = estimate_device_strategy(
        n_voxels=n_voxels,
        n_timepoints_total=total_timepoints,
        device=device,
        force_cpu=force_cpu,
    )

    if verbose and keep_on_cpu:
        data_size_gb = (n_voxels * total_timepoints * 4) / (1024**3)
        print(f"  ⚠️  Large dataset ({data_size_gb:.2f} GB)")
        print("     Loading to CPU and processing in GPU chunks")

    # Load data with blur if needed
    if blur_fwhm is not None:
        if verbose:
            print(f"\n  Applying Gaussian blur (FWHM = {blur_fwhm} mm)...")

        run_data_list = []
        run_starts = [0]
        current_timepoint = 0

        for run_file in tqdm(input_files, desc="    Loading & blurring", unit="run", disable=not verbose):
            img = load_nifti(run_file)
            data_4d = img.get_fdata(dtype=np.float32)

            # Apply blur on 4D data
            data_4d_blurred = gaussian_blur_3d(
                data_4d,
                fwhm_mm=blur_fwhm,
                voxel_sizes=voxel_sizes,
                device=device,
                verbose=False,
            )

            # Flatten and mask
            n_tps = data_4d_blurred.shape[3]
            data_2d = data_4d_blurred.reshape(-1, n_tps)

            if mask_flat is not None:
                data_2d = data_2d[mask_flat, :]

            # Convert to tensor
            data_tensor = torch.from_numpy(data_2d.T).contiguous()  # (n_tps, n_voxels)
            if not keep_on_cpu:
                data_tensor = data_tensor.to(device)

            run_data_list.append(data_tensor)
            current_timepoint += n_tps
            run_starts.append(current_timepoint)

        # Concatenate runs
        data = torch.cat(run_data_list, dim=0).T  # (n_voxels, n_timepoints)
    else:
        # Use optimized loading function
        data, run_starts = load_and_concatenate_runs(
            [Path(f) for f in input_files],
            device=device,
            keep_on_cpu=keep_on_cpu,
            mask_flat=mask_flat,
        )

    # Remove duplicate last run_start
    if len(run_starts) > len(input_files):
        run_starts = run_starts[:len(input_files)]

    # Scale to percent signal change if requested
    scale_info = None
    violations_mask = None

    if do_scale:
        if verbose:
            print("\n  Scaling to percent signal change...")

        data, violations_mask, scale_info = scale_to_percent_signal(
            data=data,
            run_starts=run_starts,
            max_scale=200.0,
            verbose=verbose,
        )

        if verbose and scale_info["n_violations"] > 0:
            print(f"    ⚠️  {scale_info['n_violations']} voxels had scaling violations")

    # Create result
    result = LoadResult(
        data=data,
        run_starts=run_starts,
        affine=affine,
        volume_shape=volume_shape,
        voxel_sizes=voxel_sizes,
        tr=tr,
        mask=mask,
        mask_flat=mask_flat,
        n_voxels=n_voxels,
        n_timepoints=data.shape[1],
        n_runs=len(input_files),
        keep_on_cpu=keep_on_cpu,
        scale_info=scale_info,
        violations_mask=violations_mask,
    )

    if verbose:
        print(f"\n  Data shape: {data.shape} (n_voxels × n_timepoints)")
        print(f"  Device: {'CPU' if keep_on_cpu else str(device)}")
        print("=" * 70 + "\n")

    return result


def save_volume_nifti(
    data_flat: torch.Tensor | np.ndarray,
    filename: str,
    volume_shape: tuple,
    affine: np.ndarray,
    mask_flat: Optional[np.ndarray] = None,
):
    """
    Reshape flat data to 3D volume and save as NIfTI file.

    Parameters
    ----------
    data_flat : torch.Tensor or np.ndarray
        Flat data (n_voxels,) to reshape to 3D
    filename : str
        Output NIfTI filename
    volume_shape : tuple
        Shape of 3D volume (x, y, z)
    affine : np.ndarray
        Affine matrix for NIfTI file
    mask_flat : np.ndarray, optional
        Flattened brain mask. If provided, unmasks data before saving.
    """
    try:
        import nibabel as nib
    except ImportError:
        print("ERROR: nibabel is required. Install with: pip install nibabel")
        sys.exit(1)

    # Convert to numpy if tensor
    if torch.is_tensor(data_flat):
        data_np = data_flat.cpu().numpy()
    else:
        data_np = data_flat

    # Unmask if needed
    if mask_flat is not None:
        full_data = np.zeros(mask_flat.size, dtype=data_np.dtype)
        full_data[mask_flat] = data_np
        data_3d = full_data.reshape(volume_shape)
    else:
        data_3d = data_np.reshape(volume_shape)

    img = nib.Nifti1Image(data_3d, affine)
    nib.save(img, filename)


def save_4d_nifti(
    data_flat: torch.Tensor | np.ndarray,
    filename: str,
    volume_shape: tuple,
    affine: np.ndarray,
    mask_flat: Optional[np.ndarray] = None,
):
    """
    Reshape (n_voxels, n_volumes) flat data to 4D and save as NIfTI file.

    Parameters
    ----------
    data_flat : torch.Tensor or np.ndarray
        Flat data (n_voxels, n_volumes) to reshape to 4D
    filename : str
        Output NIfTI filename
    volume_shape : tuple
        Shape of 3D volume (x, y, z)
    affine : np.ndarray
        Affine matrix for NIfTI file
    mask_flat : np.ndarray, optional
        Flattened brain mask. If provided, unmasks data before saving.
    """
    try:
        import nibabel as nib
    except ImportError:
        print("ERROR: nibabel is required. Install with: pip install nibabel")
        sys.exit(1)

    # Convert to numpy if tensor
    if torch.is_tensor(data_flat):
        data_np = data_flat.cpu().numpy()
    else:
        data_np = data_flat

    n_vols = data_np.shape[1]

    # Unmask if needed
    if mask_flat is not None:
        full_data = np.zeros((mask_flat.size, n_vols), dtype=data_np.dtype)
        full_data[mask_flat, :] = data_np
        data_4d = full_data.reshape((*volume_shape, n_vols))
    else:
        data_4d = data_np.reshape((*volume_shape, n_vols))

    img = nib.Nifti1Image(data_4d, affine)
    nib.save(img, filename)


# ============================================================================
# Design-Building Utilities
# ============================================================================

@dataclass
class DesignResult:
    """Result from build_design_from_onsets() or related design building functions."""
    task_design: Optional[torch.Tensor]          # (n_timepoints, n_conditions) or None for per-HRF
    nuisance_per_run: list[torch.Tensor]         # Per-run nuisance blocks
    polort: int
    condition_labels: list[str]
    n_timepoints: int
    n_runs: int
    task_indices: Optional[list[int]] = None     # Indices of task regressors
    nuisance_indices: Optional[list[int]] = None # Indices of nuisance regressors
    ortvec_labels: Optional[list[str]] = None    # Labels for ortvec regressors


def auto_polort(
    run_duration_sec: float,
    formula: str = "afni",
) -> int:
    """
    Auto-determine polynomial order from run duration.

    Parameters
    ----------
    run_duration_sec : float
        Duration of a single run in seconds
    formula : str, default="afni"
        Formula to use:
        - "afni": polort = 1 + floor(duration / 150) (AFNI/GLMdenoise convention)
        - "conservative": polort = max(1, round(duration / 120))

    Returns
    -------
    int
        Recommended polynomial order

    Notes
    -----
    The AFNI formula (1 + floor(duration/150)) is the standard convention:
    - 150s run → polort 2
    - 300s run → polort 3
    - 450s run → polort 4

    This provides one polynomial coefficient per ~2.5 minutes of data.
    """
    if formula == "afni":
        return int(1 + np.floor(run_duration_sec / 150.0))
    elif formula == "conservative":
        return max(1, int(round(run_duration_sec / 120.0)))
    else:
        raise ValueError(f"Unknown polort formula: {formula}")


def build_nuisance_per_run(
    run_starts: list[int],
    n_timepoints: int,
    polort: int,
    device: torch.device,
    ortvec_files: Optional[list[tuple[str, str]]] = None,
    ortvec_data: Optional[torch.Tensor] = None,
    noise_pcs: Optional[list[torch.Tensor]] = None,
    verbose: bool = False,
) -> list[torch.Tensor]:
    """
    Build per-run nuisance regressors (polynomials + ortvec + noise PCs).

    This is a unified function that builds block-diagonal nuisance matrices
    for cross-validation. Each run gets its own set of nuisance regressors,
    which are zero-padded so they don't affect other runs.

    Parameters
    ----------
    run_starts : list of int
        Starting timepoint for each run
    n_timepoints : int
        Total number of timepoints (all runs concatenated)
    polort : int
        Polynomial order for drift modeling (use -1 for no polynomials)
    device : torch.device
        Device to create tensors on
    ortvec_files : list of tuple, optional
        List of (filepath, label) tuples for nuisance regressor files
        Either ortvec_files or ortvec_data should be provided, not both
    ortvec_data : torch.Tensor, optional
        Pre-loaded nuisance data (n_timepoints, n_ortvec)
        Either ortvec_files or ortvec_data should be provided, not both
    noise_pcs : list of torch.Tensor, optional
        Per-run noise PCs to add as nuisance regressors
        Each element is (n_timepoints_run, n_pcs)

    Returns
    -------
    list of torch.Tensor
        Per-run nuisance matrices. Each run has its own nuisance regressors,
        potentially with different numbers of columns (polort can vary by run).

    Notes
    -----
    Nuisance regressors are RUN-SPECIFIC (each run has its own drift).
    Different runs can have different # of columns based on duration.
    For cross-validation, these should be zero-padded to max columns.

    Examples
    --------
    >>> nuisance_per_run = build_nuisance_per_run(
    ...     run_starts=[0, 200, 400],
    ...     n_timepoints=600,
    ...     polort=2,
    ...     device=torch.device("cuda"),
    ... )
    """
    from fastfuncsim.glm_core import construct_polynomial_matrix

    n_runs = len(run_starts)
    nuisance_per_run = []

    # Build polynomial blocks
    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
        run_length = end_tp - start_tp

        if polort >= 0:
            poly = construct_polynomial_matrix(run_length, polort, device=device)
        else:
            poly = torch.zeros((run_length, 0), device=device)

        nuisance_per_run.append(poly)

    # Load ortvec files if provided
    if ortvec_files is not None:
        try:
            from fastfuncsim.hrf_selection import load_nuisance_file
            from fastfuncsim.utils import to_tensor

            ortvec_all = []
            ortvec_labels = []

            for filepath, label in ortvec_files:
                ortvec = load_nuisance_file(filepath)
                ortvec = to_tensor(ortvec, device=device)

                if ortvec.shape[0] != n_timepoints:
                    print(f"ERROR: ortvec file {filepath} has {ortvec.shape[0]} rows, expected {n_timepoints}")
                    sys.exit(1)

                ortvec_all.append(ortvec)
                ortvec_labels.append(label)

                if verbose:
                    print(f"  Loaded ortvec: {filepath} (label={label}, shape={ortvec.shape})")

            ortvec_data = torch.cat(ortvec_all, dim=1) if ortvec_all else None

        except ImportError:
            print("ERROR: Could not import load_nuisance_file. Is fastfuncsim.hrf_selection available?")
            sys.exit(1)

    # Add ortvec to each run if provided
    if ortvec_data is not None:
        for run_idx in range(n_runs):
            start_tp = run_starts[run_idx]
            end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            ortvec_run = ortvec_data[start_tp:end_tp, :]

            # Concatenate polynomials + ortvec for this run
            nuisance_per_run[run_idx] = torch.cat([nuisance_per_run[run_idx], ortvec_run], dim=1)

    # Add noise PCs if provided
    if noise_pcs is not None:
        for run_idx, pcs in enumerate(noise_pcs):
            if pcs is not None and pcs.shape[1] > 0:
                # Ensure noise PCs are on the same device as nuisance
                pcs = pcs.to(device)
                # Concatenate existing nuisance + noise PCs
                nuisance_per_run[run_idx] = torch.cat([nuisance_per_run[run_idx], pcs], dim=1)

    if verbose:
        for run_idx, nuisance in enumerate(nuisance_per_run):
            print(f"  Run {run_idx}: nuisance shape = {nuisance.shape}")

    return nuisance_per_run


def build_nuisance_block_diag(
    run_starts: list[int],
    n_timepoints: int,
    polort: int,
    device: torch.device,
    ortvec_files: Optional[list[tuple[str, str]]] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Build a single block-diagonal nuisance matrix (for REML and other uses).

    This creates a block-diagonal matrix where each run has its own polynomial
    regressors, and any ortvec files are concatenated globally (not per-run).
    This is the format needed by REML and other full-design operations.

    Parameters
    ----------
    run_starts : list of int
        Starting timepoint for each run
    n_timepoints : int
        Total number of timepoints (all runs concatenated)
    polort : int
        Polynomial order for drift modeling (use -1 for no polynomials)
    device : torch.device
        Device to create tensors on
    ortvec_files : list of tuple, optional
        List of (filepath, label) tuples for nuisance regressor files
    verbose : bool
        Print progress messages

    Returns
    -------
    torch.Tensor
        Block-diagonal nuisance matrix (n_timepoints, total_nuisance_columns)
        Format: [Poly_run1  0         0        | Ortvec]
                [0         Poly_run2  0        | Ortvec]
                [0         0         Poly_run3 | Ortvec]

    Notes
    -----
    Unlike build_nuisance_per_run() which returns per-run matrices for CV,
    this returns a single matrix with block-diagonal structure.

    Ortrovec regressors are concatenated GLOBALLY (affecting all runs),
    not per-run like in build_nuisance_per_run().

    Examples
    --------
    >>> nuisance = build_nuisance_block_diag(
    ...     run_starts=[0, 200, 400],
    ...     n_timepoints=600,
    ...     polort=2,
    ...     device=torch.device("cuda"),
    ... )
    """
    from fastfuncsim.glm_core import construct_polynomial_matrix

    n_runs = len(run_starts)
    run_lengths = []

    # Compute run lengths
    for i in range(n_runs):
        if i < n_runs - 1:
            run_lengths.append(run_starts[i + 1] - run_starts[i])
        else:
            run_lengths.append(n_timepoints - run_starts[i])

    # Build polynomial blocks
    poly_blocks = []
    for run_len in run_lengths:
        if polort >= 0:
            poly_blocks.append(construct_polynomial_matrix(run_len, polort, device))
        else:
            poly_blocks.append(torch.zeros((run_len, 0), device=device))

    # Create block-diagonal matrix
    nuisance_design = torch.block_diag(*poly_blocks)

    # Load and add ortvec files if provided
    if ortvec_files is not None:
        try:
            from fastfuncsim.hrf_selection import load_nuisance_file
            from fastfuncsim.utils import to_tensor

            if verbose:
                print(f"  Loading {len(ortvec_files)} ortvec file(s)...")

            for filepath, label in ortvec_files:
                nuisance_data = load_nuisance_file(filepath, expected_rows=n_timepoints)
                ortvec_tensor = to_tensor(nuisance_data, device=device)
                nuisance_design = torch.cat([nuisance_design, ortvec_tensor], dim=1)

                if verbose:
                    print(f"    {label}: {ortvec_tensor.shape[1]} regressor(s)")

        except ImportError:
            print("ERROR: Could not import load_nuisance_file. Is fastfuncsim.hrf_selection available?")
            sys.exit(1)

    if verbose:
        print(f"  Nuisance design shape: {nuisance_design.shape}")

    return nuisance_design


def compute_run_lengths(
    run_starts: list[int],
    n_timepoints: int,
) -> list[int]:
    """
    Compute run lengths from run_starts.

    Parameters
    ----------
    run_starts : list of int
        Starting timepoint for each run
    n_timepoints : int
        Total number of timepoints

    Returns
    -------
    list of int
        Length of each run in timepoints
    """
    n_runs = len(run_starts)
    run_lengths = []

    for i in range(n_runs):
        if i < n_runs - 1:
            run_lengths.append(run_starts[i + 1] - run_starts[i])
        else:
            run_lengths.append(n_timepoints - run_starts[i])

    return run_lengths


def get_average_run_duration(
    run_lengths: list[int],
    tr: float,
) -> float:
    """
    Get average run duration in seconds.

    Parameters
    ----------
    run_lengths : list of int
        Length of each run in timepoints
    tr : float
        Repetition time in seconds

    Returns
    -------
    float
        Average run duration in seconds
    """
    avg_run_len = sum(run_lengths) / len(run_lengths)
    return avg_run_len * tr



def parse_device_arg(device_spec: Optional[str]) -> tuple[torch.device, Optional[int], Optional[int]]:
    """
    Parse device argument specification.

    Handles device specification in the format:
    - None or "auto": Auto-detect best available device
    - "cpu": Use CPU
    - "cuda": Use auto-selected CUDA device
    - "cpu,N": Use CPU with N threads (returns N in cpu_threads_override)
    - "cuda,N": Use specific CUDA device N (returns N in cuda_device_id)

    Parameters
    ----------
    device_spec : str or None
        Device specification string

    Returns
    -------
    device : torch.device
        PyTorch device object
    cpu_threads_override : int or None
        Number of CPU threads if specified (for 'cpu,N' format)
    cuda_device_id : int or None
        CUDA device ID if specified (for 'cuda,N' format)

    Examples
    --------
    >>> device, threads, cuda_id = parse_device_arg("cpu")
    >>> device, threads, cuda_id = parse_device_arg("cpu,8")  # threads=8
    >>> device, threads, cuda_id = parse_device_arg("cuda,0")  # cuda:0
    >>> device, threads, cuda_id = parse_device_arg(None)  # auto

    Notes
    -----
    This is the canonical device parser used by all CLI tools.
    """
    if not device_spec or device_spec.lower() == "auto":
        from .utils import get_device
        return get_device(), None, None

    # Parse device specification: "cpu", "cuda", "cpu,12", "cuda,0"
    parts = device_spec.split(",")
    device_type = parts[0].strip().lower()
    cpu_threads_override = None
    cuda_device_id = None

    if len(parts) > 1:
        # User specified threads or device ID
        try:
            device_param = int(parts[1].strip())
            if device_type == "cpu":
                cpu_threads_override = device_param
            elif device_type == "cuda":
                cuda_device_id = device_param
        except ValueError:
            raise ValueError(
                f"Invalid device specification: {device_spec}. "
                "Expected format: 'cpu', 'cuda', 'cpu,N', or 'cuda,N'"
            )

    # Create device
    if device_type == "cpu":
        device = torch.device("cpu")
    elif device_type == "cuda":
        if cuda_device_id is not None:
            device = torch.device(f"cuda:{cuda_device_id}")
        else:
            device = torch.device("cuda")
    else:
        raise ValueError(f"Unknown device type: {device_type}. Use 'cpu' or 'cuda'.")

    return device, cpu_threads_override, cuda_device_id


def parse_hrf_model_args(
    hrf_model_arg: str,
    canonical_arg: str | None,
    durations: list[float],
    condition_labels: list[str],
    tr: float,
) -> dict:
    """
    Parse HRF model arguments and expand labels for FIR.

    Handles backwards compatibility with -canonical flag and creates
    expanded condition labels for FIR/TENT models.

    Parameters
    ----------
    hrf_model_arg : str
        HRF model string from -hrf_model argument
    canonical_arg : str | None
        Deprecated -canonical argument (for backwards compatibility)
    durations : list[float]
        Durations for each condition (used to determine FIR window)
    condition_labels : list[str]
        Condition labels
    tr : float
        Repetition time in seconds

    Returns
    -------
    dict
        Dictionary with:
        - hrf_model_name : str
        - hrf_params : dict
        - is_fir_model : bool
        - fir_bot : float (if FIR)
        - fir_top : float (if FIR)
        - n_basis : int (if FIR)
        - condition_labels_full : list[str]

    Raises
    ------
    ValueError
        If HRF model string is invalid
    SystemExit
        If HRF model is invalid (prints error and exits)
    """
    from .design_builder import parse_hrf_model

    # Handle backwards compatibility with -canonical
    if canonical_arg is not None:
        print("  WARNING: -canonical is deprecated, use -hrf_model instead")
        hrf_model_str = canonical_arg
    else:
        hrf_model_str = hrf_model_arg

    # Parse HRF model string
    # Support both simple names (spmg1, glmsingle) and AFNI format (SPMG1(5), TENT(0,15,6))
    if "(" in hrf_model_str:
        # AFNI format with parameters: SPMG1(5), TENT(0,15,6)
        try:
            hrf_model_name, hrf_params = parse_hrf_model(hrf_model_str)
        except ValueError as e:
            print(f"ERROR: Invalid HRF model '{hrf_model_str}': {e}")
            import sys
            sys.exit(1)
    else:
        # Simple model name: spmg1, spmg2, spmg3, glmsingle
        hrf_model_name = hrf_model_str.upper()
        hrf_params = {}  # No parameters, will use durations from -durations flag

    # Determine if FIR/TENT or canonical with derivatives
    is_fir_model = hrf_model_name in ("FIR", "TENT", "TENTZERO")
    is_spm_deriv = hrf_model_name in ("SPMG2", "SPMG3")  # SPM with derivatives

    if is_fir_model:
        # For FIR/TENT, determine window from durations or params
        if "bot" in hrf_params and "top" in hrf_params:
            # Explicit window specified (e.g., TENT(0,15,6))
            fir_bot = hrf_params["bot"]
            fir_top = hrf_params["top"]
            if "n_basis" in hrf_params:
                n_basis = hrf_params["n_basis"]
            else:
                # Default: 1 basis per TR
                n_basis = int(np.ceil((fir_top - fir_bot) / tr))
        else:
            # Use durations to set window: 0 to max(durations)
            fir_bot = 0.0
            fir_top = max(durations)
            # Default: 1 basis per TR
            n_basis = int(np.ceil(fir_top / tr))

        print(f"  HRF model: {hrf_model_name} (window: {fir_bot:.1f}-{fir_top:.1f}s, {n_basis} basis functions)")

        # Expand condition labels for FIR: cond1_t0.0s, cond1_t1.5s, ..., cond2_t0.0s, ...
        fir_condition_labels = []
        for cond_label in condition_labels:
            for lag_idx in range(n_basis):
                lag_time = fir_bot + lag_idx * (fir_top - fir_bot) / max(1, n_basis - 1)
                fir_condition_labels.append(f"{cond_label}_t{lag_time:.1f}s")
        condition_labels_full = fir_condition_labels
    elif is_spm_deriv:
        # SPMG2/SPMG3: Canonical HRF with derivatives
        # Each condition gets multiple basis functions
        if hrf_model_name == "SPMG2":
            n_basis = 2  # Canonical + temporal derivative
            basis_suffixes = ["_canonical", "_timederiv"]
            print(f"  HRF model: {hrf_model_name} (canonical + temporal derivative, 2 basis functions per condition)")
        elif hrf_model_name == "SPMG3":
            n_basis = 3  # Canonical + temporal + dispersion derivatives
            basis_suffixes = ["_canonical", "_timederiv", "_dispderiv"]
            print(f"  HRF model: {hrf_model_name} (canonical + time + dispersion derivatives, 3 basis functions per condition)")

        # Expand condition labels: cond1_canonical, cond1_timederiv, ..., cond2_canonical, ...
        deriv_condition_labels = []
        for cond_label in condition_labels:
            for suffix in basis_suffixes:
                deriv_condition_labels.append(f"{cond_label}{suffix}")
        condition_labels_full = deriv_condition_labels

        fir_bot = None
        fir_top = None
    else:
        # Simple canonical: spmg1, glmsingle
        print(f"  HRF model: {hrf_model_name} (canonical)")
        # Canonical: condition labels stay as-is
        condition_labels_full = condition_labels
        fir_bot = None
        fir_top = None
        n_basis = 1  # Single basis function per condition

    return {
        "hrf_model_name": hrf_model_name,
        "hrf_params": hrf_params,
        "is_fir_model": is_fir_model,
        "is_spm_deriv": is_spm_deriv,
        "fir_bot": fir_bot,
        "fir_top": fir_top,
        "n_basis": n_basis,
        "condition_labels_full": condition_labels_full,
    }


def validate_hrf_compatibility(
    is_fir_model: bool,
    single_trial: bool = False,
    hrf_opt: str | None = None,
) -> None:
    """
    Validate HRF model compatibility with other options.

    FIR/TENT models are incompatible with:
    - Single-trial refitting (FIR already provides time-resolved estimates)
    - Per-voxel HRF optimization (FIR uses data-driven basis functions)

    Parameters
    ----------
    is_fir_model : bool
        Whether using FIR/TENT model
    single_trial : bool, optional
        Whether single-trial refitting is enabled
    hrf_opt : str | None, optional
        Per-voxel HRF optimization mode

    Raises
    ------
    SystemExit
        If incompatible options are detected (prints error and exits)
    """
    import sys

    if is_fir_model:
        if single_trial:
            print("ERROR: FIR/TENT models are incompatible with -single_trial")
            print("  FIR already provides time-resolved estimates; single-trial refitting is redundant")
            sys.exit(1)
        if hrf_opt:
            print("ERROR: FIR/TENT models are incompatible with -hrf_opt (per-voxel HRFs)")
            print("  FIR/TENT use data-driven basis functions, not assumed HRF shapes")
            sys.exit(1)


def build_task_design_from_args(
    hrf_model_name: str,
    is_fir_model: bool,
    fir_bot: float | None,
    fir_top: float | None,
    n_basis: int | None,
    all_onsets: list,
    onset_matrix_micro: torch.Tensor,
    n_conditions: int,
    n_timepoints: int,
    run_starts: list[int],
    tr: float,
    microtime_dt: float,
    device: torch.device,
    hrf_opt: str | None = None,
    hrf_library: torch.Tensor | None = None,
    hrf_indices: torch.Tensor | None = None,
    n_voxels: int | None = None,
) -> tuple[torch.Tensor | None, dict | None]:
    """
    Build task design matrix based on HRF model type.

    Handles three modes:
    1. FIR/TENT models: Use basis functions (no convolution)
    2. Canonical HRF: Convolve with assumed shape (spmg1 or glmsingle)
    3. Per-voxel HRF: Build designs_by_hrf dict for each unique HRF

    Parameters
    ----------
    hrf_model_name : str
        HRF model name (e.g., "FIR", "TENT", "spmg1", "glmsingle")
    is_fir_model : bool
        Whether using FIR/TENT model
    fir_bot : float | None
        FIR window bottom (seconds), required if is_fir_model
    fir_top : float | None
        FIR window top (seconds), required if is_fir_model
    n_basis : int | None
        Number of basis functions, required if is_fir_model
    all_onsets : list
        List of [condition][run] onset arrays
    onset_matrix_micro : torch.Tensor
        Onset matrix at microtime resolution (n_microtime, n_conditions)
    n_conditions : int
        Number of conditions
    n_timepoints : int
        Total number of TR timepoints
    run_starts : list[int]
        Starting timepoint for each run
    tr : float
        Repetition time in seconds
    microtime_dt : float
        Microtime resolution in seconds
    device : torch.device
        Device for computation
    hrf_opt : str | None, optional
        Per-voxel HRF optimization mode (if enabled)
    hrf_library : torch.Tensor | None, optional
        HRF library for per-voxel HRFs (n_hrfs, hrf_length)
    hrf_indices : torch.Tensor | None, optional
        HRF indices per voxel (n_voxels,)
    n_voxels : int | None, optional
        Number of voxels (for reporting with per-voxel HRFs)

    Returns
    -------
    task_design : torch.Tensor | None
        Task design matrix (n_timepoints, n_task_regressors) or None if per-voxel HRF
    designs_by_hrf : dict | None
        Dictionary mapping HRF index to design matrix, or None if single design

    Raises
    ------
    SystemExit
        If required parameters are missing or invalid
    """
    import sys
    from .design import make_fir_design, make_tent_design, convolve_hrf_microtime
    from .hrf import get_spmg1_hrf

    if hrf_opt:
        # Per-voxel HRF mode: build designs_by_hrf dict
        print(f"  Convolving design with {len(hrf_library)} HRFs from library...")

        # Find unique HRF indices
        unique_hrf_indices = torch.unique(hrf_indices).tolist()
        n_unique = len(unique_hrf_indices)
        if n_voxels:
            print(f"  Processing {n_unique} unique HRFs across {n_voxels:,} voxels")
        else:
            print(f"  Processing {n_unique} unique HRFs")

        # Build per-HRF design matrices
        designs_by_hrf = {}
        for hrf_idx_val in unique_hrf_indices:
            hrf_kernel = hrf_library[hrf_idx_val]
            design_for_hrf = convolve_hrf_microtime(
                onsets_microtime=onset_matrix_micro,
                hrf=hrf_kernel,
                n_timepoints=n_timepoints,
                tr=tr,
                microtime_dt=microtime_dt,
                device=device,
            )
            assert isinstance(design_for_hrf, torch.Tensor)
            designs_by_hrf[hrf_idx_val] = design_for_hrf

        print(f"  Created {len(designs_by_hrf)} HRF-specific design matrices")
        print(f"  Task predictors: {n_conditions} conditions")
        return None, designs_by_hrf

    else:
        # Single HRF model for all voxels
        if is_fir_model:
            # FIR/TENT: Use basis functions (no convolution)
            print(f"  Building {hrf_model_name} design matrix ({n_basis} basis functions per condition)")

            if hrf_model_name == "FIR":
                task_design = make_fir_design(
                    onsets=all_onsets,
                    n_lags=n_basis,
                    tr=tr,
                    n_timepoints=n_timepoints,
                    run_starts=run_starts,
                    device=device,
                )
            elif hrf_model_name in ("TENT", "TENTZERO"):
                task_design = make_tent_design(
                    onsets=all_onsets,
                    bot=fir_bot,
                    top=fir_top,
                    n_basis=n_basis,
                    tr=tr,
                    n_timepoints=n_timepoints,
                    run_starts=run_starts,
                    zero_edges=(hrf_model_name == "TENTZERO"),
                    device=device,
                )
            else:
                print(f"ERROR: Unknown FIR model: {hrf_model_name}")
                sys.exit(1)

            print(f"  Design shape: {task_design.shape[0]} timepoints × {task_design.shape[1]} regressors")
            print(f"    ({n_conditions} conditions × {n_basis} basis functions)")

        else:
            # Canonical HRF: Convolve with assumed shape (with optional derivatives)
            if hrf_model_name in ("SPMG2", "SPMG3"):
                # SPM canonical with derivatives
                print(f"  Using SPM canonical HRF with derivatives ({hrf_model_name})")
                from .hrf import get_spm_hrf_with_derivatives

                # Get HRF set (canonical + derivatives)
                # Shape: (n_basis, hrf_length) where n_basis = 2 for SPMG2, 3 for SPMG3
                hrf_set = get_spm_hrf_with_derivatives(
                    microtime_dt=microtime_dt,
                    hrf_duration=32.0,
                    n_basis=n_basis,
                    device=device,
                )

                # Convolve each basis function with onset matrix
                # Result: list of (n_timepoints, n_conditions) matrices
                design_per_basis = []
                for basis_idx in range(n_basis):
                    hrf_basis = hrf_set[basis_idx]  # Single HRF from the set
                    design_basis = convolve_hrf_microtime(
                        onsets_microtime=onset_matrix_micro,
                        hrf=hrf_basis,
                        n_timepoints=n_timepoints,
                        tr=tr,
                        microtime_dt=microtime_dt,
                        device=device,
                    )
                    design_per_basis.append(design_basis)

                # Interleave columns: cond1_canonical, cond1_timederiv, ..., cond2_canonical, ...
                # For each condition, stack all basis functions
                task_columns = []
                for cond_idx in range(n_conditions):
                    for basis_idx in range(n_basis):
                        # Extract this condition's column from this basis
                        task_columns.append(design_per_basis[basis_idx][:, cond_idx:cond_idx+1])

                # Concatenate all columns
                task_design = torch.cat(task_columns, dim=1)
                print(f"  Design shape: {task_design.shape[0]} timepoints × {task_design.shape[1]} regressors")
                print(f"    ({n_conditions} conditions × {n_basis} basis functions)")

            elif hrf_model_name == "spmg1":
                print("  Using canonical SPMG1 HRF")
                hrf = get_spmg1_hrf(
                    microtime_dt=microtime_dt,
                    stim_duration=0.0,
                    hrf_duration=32.0,
                    normalize_peak=True,
                    device=device,
                )
                task_design = convolve_hrf_microtime(
                    onsets_microtime=onset_matrix_micro,
                    hrf=hrf,
                    n_timepoints=n_timepoints,
                    tr=tr,
                    microtime_dt=microtime_dt,
                    device=device,
                )
            elif hrf_model_name == "glmsingle":
                print("  Using canonical GLMsingle HRF")
                from .hrf import get_glmsingle_hrf
                hrf = get_glmsingle_hrf(
                    microtime_dt=microtime_dt,
                    device=device,
                )
                task_design = convolve_hrf_microtime(
                    onsets_microtime=onset_matrix_micro,
                    hrf=hrf,
                    n_timepoints=n_timepoints,
                    tr=tr,
                    microtime_dt=microtime_dt,
                    device=device,
                )
            else:
                print(f"ERROR: Unknown canonical HRF: {hrf_model_name}")
                sys.exit(1)

        assert isinstance(task_design, torch.Tensor), "Expected Tensor for single HRF"
        return task_design, None
