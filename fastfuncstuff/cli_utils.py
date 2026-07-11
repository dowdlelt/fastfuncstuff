"""
Shared utility functions for CLI tools.

This module contains common utilities used across multiple CLI tools to avoid
code duplication. Functions include argument parsing, CV strategy parsing,
data loading, and output formatting.
"""

from __future__ import annotations

import contextlib
import glob as glob_module
import itertools
import re
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_NIFTI_EXTENSIONS = (".nii.zst", ".nii.gz", ".nii")


@contextlib.contextmanager
def spinner(message: str, stream=None, interval: float = 0.1) -> Iterator[None]:
    """Show a braille spinner next to ``message`` while the ``with`` block runs.

    For opaque single-shot work a progress bar can't measure — reading one big
    run off disk, a slow decompress — where the only honest signal is "still
    going". Animates on a background thread so the wrapped call is unblocked, and
    only when ``stream`` is a real terminal: piped/redirected output gets nothing
    (no ``\\r`` spam in logs). The line is cleared on exit, so the caller's own
    result line (shape, timing) reads as the completion notice.
    """
    stream = stream or sys.stderr
    if not getattr(stream, "isatty", lambda: False)():
        yield  # non-interactive: stay silent, let the caller's prints speak
        return

    stop = threading.Event()

    def _spin() -> None:
        for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            if stop.is_set():
                break
            stream.write(f"\r{ch} {message}")
            stream.flush()
            time.sleep(interval)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        stream.write("\r\033[K")  # carriage return + clear-to-end-of-line
        stream.flush()


@dataclass
class PrefixInfo:
    """Parsed prefix with NIfTI extension separated from the stem.

    Attributes
    ----------
    stem : str
        Prefix with any NIfTI extension stripped.  Safe for use as a
        directory name or as a base for constructing output paths.
    nifti_ext : str
        NIfTI extension to use for volume outputs (e.g. ``".nii.gz"``).
        Defaults to ``".nii.gz"`` when no extension was specified.
    """

    stem: str
    nifti_ext: str

    def with_suffix(self, descriptor: str) -> str:
        """Build an output path: ``{stem}_{descriptor}{nifti_ext}``.

        Parameters
        ----------
        descriptor : str
            File descriptor (e.g. ``"r2"``, ``"betas"``).
        """
        return f"{self.stem}_{descriptor}{self.nifti_ext}"

    def as_file(self) -> str:
        """Return ``{stem}{nifti_ext}`` — the prefix used as a single output file."""
        return f"{self.stem}{self.nifti_ext}"


def parse_prefix(prefix: str, default_ext: str = ".nii.gz") -> PrefixInfo:
    """Parse a CLI ``-prefix`` value into a stem and NIfTI extension.

    The user can signal the desired output compression via the prefix
    extension::

        -prefix out           →  stem="out",       ext=".nii.gz"  (default)
        -prefix out.nii.gz    →  stem="out",       ext=".nii.gz"
        -prefix out.nii       →  stem="out",       ext=".nii"
        -prefix out.nii.zst   →  stem="out",       ext=".nii.zst"
        -prefix dir/sub01     →  stem="dir/sub01", ext=".nii.gz"

    Parameters
    ----------
    prefix : str
        Raw ``-prefix`` value from argparse.
    default_ext : str
        Extension to use when the prefix has none (default ``".nii.gz"``).

    Returns
    -------
    PrefixInfo
    """
    for ext in _NIFTI_EXTENSIONS:
        if prefix.endswith(ext):
            return PrefixInfo(stem=prefix[: -len(ext)], nifti_ext=ext)
    return PrefixInfo(stem=prefix, nifti_ext=default_ext)


def clean_condition_labels(raw_labels: list[str]) -> list[str]:
    """Strip common prefix and suffix from condition labels.

    Onset filenames like ``onsets.localizer.times.faces`` share the prefix
    ``onsets.localizer.times.`` — stripping it yields just ``faces``.
    Works on dot-, underscore-, or hyphen-separated names.

    Examples
    --------
    >>> clean_condition_labels(["onsets.localizer.times.faces",
    ...     "onsets.localizer.times.bodies", "onsets.localizer.times.scenes"])
    ['faces', 'bodies', 'scenes']
    """
    import os

    if len(raw_labels) <= 1:
        return raw_labels

    # Find longest common prefix (character-level), trim to last separator
    prefix = os.path.commonprefix(raw_labels)
    for sep in (".", "_", "-"):
        idx = prefix.rfind(sep)
        if idx >= 0:
            prefix = prefix[: idx + 1]
            break
    else:
        prefix = ""

    # Find longest common suffix (character-level), trim to first separator
    reversed_labels = [lab[::-1] for lab in raw_labels]
    suffix_rev = os.path.commonprefix(reversed_labels)
    suffix = suffix_rev[::-1]
    if suffix:
        for sep in (".", "_", "-"):
            idx = suffix.find(sep)
            if idx >= 0:
                suffix = suffix[idx:]
                break
        else:
            suffix = ""

    # Strip prefix and suffix
    cleaned = []
    for lab in raw_labels:
        s = lab
        if prefix:
            s = s[len(prefix) :]
        if suffix:
            s = s[: -len(suffix)]
        cleaned.append(s if s else lab)

    return cleaned


def parse_input_files(input_arg: str | list[str]) -> list[str]:
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

    # Validate files exist (strip AFNI sub-brick selectors before checking)
    from fastfuncstuff.io.afni import parse_subbrick_selector

    for f in files:
        clean_path, _ = parse_subbrick_selector(f)
        if not Path(clean_path).exists():
            print(f"ERROR: Input file not found: {f}")
            sys.exit(1)

    return files


def parse_cv_strategy(cv_str: str) -> int | float:
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
    from fastfuncstuff.memory import estimate_keep_on_cpu

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

    data: torch.Tensor  # (n_voxels, n_timepoints) fMRI data
    run_starts: list[int]  # Starting timepoint for each run
    affine: np.ndarray  # Affine matrix for NIfTI files
    volume_shape: tuple  # Shape of 3D volume
    voxel_sizes: tuple  # Voxel dimensions in mm
    tr: float  # Repetition time in seconds
    mask: np.ndarray | None  # Brain mask (original 3D)
    mask_flat: np.ndarray | None  # Flattened mask (1D bool)
    n_voxels: int  # Number of voxels
    n_timepoints: int  # Total timepoints
    n_runs: int  # Number of runs
    keep_on_cpu: bool  # Whether data is stored on CPU
    scale_info: dict | None  # Scaling info if do_scale was True
    violations_mask: torch.Tensor | None = None  # Scaling violations if do_scale
    nifti_header: object | None = None  # NIfTI header (preserves AFNI extension/space info)


def load_and_preprocess_runs(
    input_files: list[str],
    tr: float | None = None,
    mask_file: str | None = None,
    blur_fwhm: float | None = None,
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

        from fastfuncstuff.io.afni import load_afni_mask, load_and_concatenate_runs, load_nifti
        from fastfuncstuff.utils import gaussian_blur_3d, scale_to_percent_signal
    except ImportError as e:
        print(f"ERROR: Could not import required modules: {e}")
        sys.exit(1)

    if verbose:
        print("\n" + "=" * 70)
        print("📂 Loading fMRI Data")
        print("=" * 70)

    # Load metadata from first file (keep header for AFNI space/view info)
    first_img = load_nifti(input_files[0])
    affine = np.array(first_img.affine) if hasattr(first_img, "affine") else np.eye(4)
    nifti_header = first_img.header.copy() if hasattr(first_img, "header") else None
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
        print(f"  Voxel sizes: {tuple(f'{v:.3f}' for v in voxel_sizes)} mm")
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
        from fastfuncstuff.utils import generate_synthetic_runs

        if verbose:
            print("\n" + "=" * 70)
            print("🎭 DRY RUN MODE - Fast Pipeline Testing")
            print("=" * 70)
            print("  Reading header only, generating synthetic data...")

        # Read only the header (no data loading)
        from fastfuncstuff.io.afni import load_nifti

        first_img = load_nifti(input_files[0])
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
            nifti_header=nifti_header,
        )

        if verbose:
            print(f"\n  Data shape: {data.shape} (n_voxels × n_timepoints)")
            print("  Device: CPU (dry run)")
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

        for run_file in tqdm(
            input_files, desc="    Loading & blurring", unit="run", disable=not verbose
        ):
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
        run_starts = run_starts[: len(input_files)]

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
        nifti_header=nifti_header,
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
    mask_flat: np.ndarray | None = None,
    header: object | None = None,
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
    header : nibabel header, optional
        NIfTI header to preserve (AFNI space/view info, etc.).
    """
    from fastfuncstuff.io.afni import save_nifti

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

    save_nifti(data_3d, output_path=filename, affine=affine, header=header)


def save_4d_nifti(
    data_flat: torch.Tensor | np.ndarray,
    filename: str,
    volume_shape: tuple,
    affine: np.ndarray,
    mask_flat: np.ndarray | None = None,
    header: object | None = None,
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
    header : nibabel header, optional
        NIfTI header to preserve (AFNI space/view info, etc.).
    """
    from fastfuncstuff.io.afni import save_nifti

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

    save_nifti(data_4d, output_path=filename, affine=affine, header=header)


# ============================================================================
# Design-Building Utilities
# ============================================================================


@dataclass
class DesignResult:
    """Result from build_design_from_onsets() or related design building functions."""

    task_design: torch.Tensor | None  # (n_timepoints, n_conditions) or None for per-HRF
    nuisance_per_run: list[torch.Tensor]  # Per-run nuisance blocks
    polort: int
    condition_labels: list[str]
    n_timepoints: int
    n_runs: int
    task_indices: list[int] | None = None  # Indices of task regressors
    nuisance_indices: list[int] | None = None  # Indices of nuisance regressors
    ortvec_labels: list[str] | None = None  # Labels for ortvec regressors


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


# ---------------------------------------------------------------------------
# Nuisance harmonisation: NuisanceBlock + factories + assembler.
#
# Three CLI input modes funnel into the same internal representation
# (a list[NuisanceBlock]). Downstream consumers see one structure.
# ---------------------------------------------------------------------------


@dataclass
class NuisanceBlock:
    """A labelled set of nuisance regressors with explicit per-run mapping.

    ``per_run[i]`` is either an ``(n_timepoints_i, n_cols_block)`` array of
    regressor values for run ``i`` (any dtype; coerced to float32 on use)
    or ``None`` meaning "this block contributes nothing to run ``i``"
    (zero columns get padded in at assembly time).

    Runs with fewer columns than the block's widest run are zero-padded
    on the right during ``get_run``. This is what makes variable-PC-count-
    per-run paths (e.g. after Kay-style PC selection) representable
    without forcing the user to pre-pad.

    Note: data is stored raw. Per-run demeaning happens at assembly,
    not here, so the original input remains inspectable.
    """

    label: str
    per_run: list[np.ndarray | None]
    column_names: list[str] | None = None
    # Provenance: filename that fed each run (None where the block is empty).
    source: list[str | None] = field(default_factory=list)
    # True → per-run regressors that must be expanded BLOCK-DIAGONALLY (each run
    # gets its own columns, zero outside that run), like the polynomials — this is
    # the -ortvec_run / -ortvec_glob case (distinct per-run files). False → a single
    # full-length regressor SHARED across runs (AFNI -ortvec semantics).
    block_diagonal: bool = False

    def __post_init__(self):
        if not self.source:
            self.source = [None] * len(self.per_run)
        elif len(self.source) != len(self.per_run):
            raise ValueError(
                f"NuisanceBlock {self.label!r}: source has "
                f"{len(self.source)} entries but per_run has {len(self.per_run)}"
            )

    @property
    def n_columns(self) -> int:
        """Max columns across runs — the assembled block's width."""
        return max((m.shape[1] for m in self.per_run if m is not None), default=0)

    def get_run(self, run_idx: int, run_length: int) -> np.ndarray:
        """Return ``(run_length, n_columns)`` for ``run_idx``, zero-padded.

        Empty runs (per_run[run_idx] is None) → all zeros.
        Short-column runs → right-padded with zero columns.
        """
        ncols = self.n_columns
        m = self.per_run[run_idx]
        if m is None:
            return np.zeros((run_length, ncols), dtype=np.float32)
        if m.shape[0] != run_length:
            raise ValueError(
                f"NuisanceBlock {self.label!r}: run {run_idx} has "
                f"{m.shape[0]} rows, design expects {run_length}"
            )
        m = m.astype(np.float32, copy=False)
        if m.shape[1] < ncols:
            pad = np.zeros((run_length, ncols - m.shape[1]), dtype=np.float32)
            return np.hstack([m, pad])
        return m

    def get_column_names(self) -> list[str]:
        ncols = self.n_columns
        if self.column_names is not None and len(self.column_names) == ncols:
            return list(self.column_names)
        return [f"{self.label}_{i:02d}" for i in range(ncols)]


@dataclass
class NuisanceAssembly:
    """Result of assembling polort + nuisance blocks (+ optional noise PCs)
    into per-run nuisance design matrices.

    Attributes
    ----------
    per_run : list[torch.Tensor]
        One tensor per run, shape ``(run_length_i, n_total_cols_i)``.
    per_run_column_names : list[list[str]]
        Names for each column of each run's matrix. Lengths match
        ``per_run[i].shape[1]``.
    blocks : list[NuisanceBlock]
        The blocks consumed (after factory-time loading), for downstream
        provenance / sidecar emission.
    """

    per_run: list[torch.Tensor]
    per_run_column_names: list[list[str]]
    blocks: list[NuisanceBlock]


# Regex priority list for `-ortvec_glob` filename → run-index inference.
# Try most-specific (BIDS) first; fall through to broad / trailing-number
# heuristics. Each must yield an all-files-matched, unique, in-bounds
# assignment to succeed. If none does, the caller gets a list of what
# was tried so it can pick a different input mode.
_RUN_INDEX_PATTERNS = (
    (r"_run-(\d+)_", "BIDS _run-N_"),
    (r"_run-(\d+)\b", "BIDS _run-N"),
    (r"_run(\d+)_", "_runN_"),
    (r"_run(\d+)\b", "_runN"),
    (r"[._]run[-_]?(\d+)", "broad .run-N / _run_N"),
    (r"(\d+)\.[^.]+$", "trailing N before extension"),
    (r"(\d+)\D*$", "trailing N"),
)


def _infer_run_indices_from_filenames(
    names: list[str],
    n_runs: int,
) -> list[int]:
    """Try patterns in priority order; return 0-indexed run for each name.

    A pattern succeeds when it (a) matches every name, (b) yields unique
    run numbers, and (c) all numbers fall in ``[1, n_runs]``. The first
    success wins. If none succeeds we raise ``ValueError`` with the
    failures from each attempt, so the user can see what's ambiguous.
    """
    failures: list[str] = []
    for pat, desc in _RUN_INDEX_PATTERNS:
        matches = [re.search(pat, n, re.IGNORECASE) for n in names]
        if not all(matches):
            unmatched = [n for n, m in zip(names, matches, strict=False) if not m]
            failures.append(f"{desc}: did not match {unmatched[:3]}")
            continue
        nums = [int(m.group(1)) for m in matches]
        dup_counts = Counter(nums)
        dups = [n for n, c in dup_counts.items() if c > 1]
        if dups:
            failures.append(f"{desc}: duplicate run numbers {sorted(dups)}")
            continue
        if any(n < 1 or n > n_runs for n in nums):
            failures.append(f"{desc}: out-of-range run numbers {sorted(nums)} for n_runs={n_runs}")
            continue
        return [n - 1 for n in nums]
    raise ValueError(
        "Could not infer per-run indices from filenames. Patterns tried:\n  "
        + "\n  ".join(failures)
        + f"\n  Files: {names}\n"
        "Use -ortvec_run FILE LABEL RUN to assign explicitly, or rename "
        "files to a BIDS-style _run-NN_ pattern."
    )


def _slice_full_length_per_run(
    arr: np.ndarray,
    run_starts: list[int],
    n_timepoints: int,
) -> list[np.ndarray]:
    n_runs = len(run_starts)
    out = []
    for i in range(n_runs):
        end = run_starts[i + 1] if i < n_runs - 1 else n_timepoints
        out.append(arr[run_starts[i] : end])
    return out


def make_nuisance_block_from_full_length(
    path: str | Path,
    label: str,
    run_starts: list[int],
    n_timepoints: int,
) -> NuisanceBlock:
    """Mode 1: one file with rows for *all* runs concatenated (pre-padded)."""
    from fastfuncstuff.design.hrf_selection import load_nuisance_file

    arr = load_nuisance_file(path)
    if arr.shape[0] != n_timepoints:
        raise ValueError(
            f"{path}: has {arr.shape[0]} rows, design has {n_timepoints} total timepoints"
        )
    per_run = _slice_full_length_per_run(arr, run_starts, n_timepoints)
    return NuisanceBlock(
        label=label,
        per_run=per_run,
        source=[str(path)] * len(run_starts),
    )


def make_nuisance_block_from_per_run_file(
    path: str | Path,
    label: str,
    run_idx_1based: int,
    run_starts: list[int],
    n_timepoints: int,
) -> NuisanceBlock:
    """Mode 2: one file that covers exactly one run; other runs get zeros."""
    from fastfuncstuff.design.hrf_selection import load_nuisance_file

    n_runs = len(run_starts)
    run_idx = run_idx_1based - 1
    if not (0 <= run_idx < n_runs):
        raise ValueError(f"run index {run_idx_1based} out of range [1, {n_runs}] for {path}")
    expected = (run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints) - run_starts[
        run_idx
    ]
    arr = load_nuisance_file(path)
    if arr.shape[0] != expected:
        raise ValueError(
            f"{path}: has {arr.shape[0]} rows, run {run_idx_1based} has {expected} timepoints"
        )
    per_run: list[np.ndarray | None] = [None] * n_runs
    per_run[run_idx] = arr
    source: list[str | None] = [None] * n_runs
    source[run_idx] = str(path)
    return NuisanceBlock(label=label, per_run=per_run, source=source, block_diagonal=True)


def make_nuisance_block_from_glob(
    pattern: str,
    label: str,
    run_starts: list[int],
    n_timepoints: int,
) -> NuisanceBlock:
    """Mode 3: glob matches N files; infer per-file run index from filename.

    Runs absent from the glob are zero-padded into the block.
    """
    from fastfuncstuff.design.hrf_selection import load_nuisance_file

    matched = sorted(Path(p) for p in glob_module.glob(pattern))
    if not matched:
        raise ValueError(f"-ortvec_glob {pattern!r}: matched no files")

    n_runs = len(run_starts)
    run_indices_0 = _infer_run_indices_from_filenames(
        [p.name for p in matched],
        n_runs=n_runs,
    )

    per_run: list[np.ndarray | None] = [None] * n_runs
    source: list[str | None] = [None] * n_runs
    for path, run_idx in zip(matched, run_indices_0, strict=False):
        expected = (run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints) - run_starts[
            run_idx
        ]
        arr = load_nuisance_file(path)
        if arr.shape[0] != expected:
            raise ValueError(
                f"{path}: has {arr.shape[0]} rows, run {run_idx + 1} has {expected} timepoints"
            )
        per_run[run_idx] = arr
        source[run_idx] = str(path)

    return NuisanceBlock(label=label, per_run=per_run, source=source, block_diagonal=True)


def add_ortvec_arguments(parser_or_group, include_legacy: bool = True) -> None:
    """Register `-ortvec`, `-ortvec_run`, `-ortvec_glob` on a parser/group.

    All three are repeatable. The CLI then funnels them through
    `collect_nuisance_blocks(args, ...)` to get a `list[NuisanceBlock]`.
    """
    if include_legacy:
        parser_or_group.add_argument(
            "-ortvec",
            action="append",
            nargs=2,
            metavar=("FILE", "LABEL"),
            help=(
                "Full-length nuisance regressors (pre-concatenated across all runs). "
                "Repeatable: -ortvec motion.1D motion -ortvec physio.1D physio"
            ),
        )
    parser_or_group.add_argument(
        "-ortvec_run",
        "-ortvec-run",
        action="append",
        nargs=3,
        metavar=("FILE", "LABEL", "RUN"),
        help=(
            "Per-run nuisance regressors; RUN is 1-based run index. Other runs are zero-padded. "
            "Repeatable: -ortvec_run motion_r1.1D motion 1 -ortvec_run motion_r2.1D motion 2"
        ),
    )
    parser_or_group.add_argument(
        "-ortvec_glob",
        "-ortvec-glob",
        action="append",
        nargs=2,
        metavar=("PATTERN", "LABEL"),
        help=(
            "Glob matching per-run nuisance files; run index inferred from filename "
            "(BIDS-style `_run-N_` preferred). Missing runs zero-padded. "
            "Repeatable: -ortvec_glob 'motion_run-*.1D' motion"
        ),
    )
    parser_or_group.add_argument(
        "-ortvec_concat",
        "-ortvec-concat",
        action="append",
        nargs=2,
        metavar=("PATTERN", "LABEL"),
        help=(
            "Glob matching N already-full-length per-run files (e.g. AFNI "
            "mot_demean.r0N.1D — each spans every run with zeros outside its own). "
            "Expanded into N -ortvec entries labelled LABEL01, LABEL02, … "
            "(width auto-padded from n_runs). Repeatable: "
            "-ortvec_concat 'mot_demean.r0*.1D' motion"
        ),
    )


def expand_ortvec_concat(
    pattern: str,
    label: str,
    n_runs: int,
) -> list[tuple[Path, str]]:
    """Expand a ``-ortvec_concat PATTERN LABEL`` invocation into a list of
    ``(file, suffixed_label)`` pairs, ordered by inferred run index.

    Each matched file is expected to be already full-length (the AFNI
    ``mot_demean.r0N.1D`` shape), so the caller should treat the result as
    a sequence of ``-ortvec`` (mode 1) entries. Labels are zero-padded to
    the width of ``n_runs``: ``motion01``, ``motion02``, ….
    """
    matched = sorted(Path(p) for p in glob_module.glob(pattern))
    if not matched:
        raise ValueError(f"-ortvec_concat {pattern!r}: matched no files")
    run_indices_1 = [
        i + 1
        for i in _infer_run_indices_from_filenames(
            [p.name for p in matched],
            n_runs=n_runs,
        )
    ]
    width = max(2, len(str(n_runs)))
    return sorted(
        (
            (path, f"{label}{idx:0{width}d}")
            for path, idx in zip(matched, run_indices_1, strict=True)
        ),
        key=lambda t: t[1],
    )


def collect_nuisance_blocks(
    args,
    run_starts: list[int],
    n_timepoints: int,
    verbose: bool = False,
) -> list[NuisanceBlock]:
    """Translate argparse Namespace fields into a list of NuisanceBlock.

    Recognises ``args.ortvec``, ``args.ortvec_run``, ``args.ortvec_glob``,
    ``args.ortvec_concat`` — any combination, all repeatable, all optional.
    Designed to plug into any CLI that called ``add_ortvec_arguments`` on
    its parser.
    """
    blocks: list[NuisanceBlock] = []
    for path, label in getattr(args, "ortvec", None) or []:
        blocks.append(
            make_nuisance_block_from_full_length(
                path,
                label,
                run_starts,
                n_timepoints,
            )
        )
        if verbose:
            print(f"  -ortvec: {path} (label={label})")
    for path, label, run_str in getattr(args, "ortvec_run", None) or []:
        try:
            run_idx = int(run_str)
        except ValueError:
            print(f"ERROR: -ortvec_run RUN must be a 1-based integer, got {run_str!r}")
            sys.exit(1)
        blocks.append(
            make_nuisance_block_from_per_run_file(
                path,
                label,
                run_idx,
                run_starts,
                n_timepoints,
            )
        )
        if verbose:
            print(f"  -ortvec_run: {path} (label={label}, run={run_idx})")
    for pattern, label in getattr(args, "ortvec_glob", None) or []:
        block = make_nuisance_block_from_glob(
            pattern,
            label,
            run_starts,
            n_timepoints,
        )
        blocks.append(block)
        if verbose:
            matched = [s for s in block.source if s]
            print(f"  -ortvec_glob: {pattern} (label={label}) → {len(matched)} run(s) assigned")
    # -ortvec_concat: glob over already-full-length per-run files; each becomes
    # its own full-length NuisanceBlock with an auto-suffixed label. Equivalent
    # to writing N -ortvec calls.
    for pattern, label in getattr(args, "ortvec_concat", None) or []:
        n_runs = len(run_starts)
        for path, suffixed_label in expand_ortvec_concat(pattern, label, n_runs):
            blocks.append(
                make_nuisance_block_from_full_length(
                    path,
                    suffixed_label,
                    run_starts,
                    n_timepoints,
                )
            )
            if verbose:
                print(f"  -ortvec_concat: {path} (label={suffixed_label})")
    return blocks


def assemble_per_run_nuisance(
    blocks: list[NuisanceBlock],
    run_starts: list[int],
    n_timepoints: int,
    polort: int,
    device: torch.device,
    noise_pcs: list[torch.Tensor] | None = None,
    verbose: bool = False,
) -> NuisanceAssembly:
    """Combine polynomials + nuisance blocks + optional noise PCs per run.

    Each block is demeaned per-run before concatenation: if any column's
    per-run mean exceeds 1e-4 in absolute value the column is centred
    (in-memory; the original NuisanceBlock is untouched). This prevents
    a non-zero-mean ortvec column from fighting the polort intercept and
    making the design rank-degenerate.
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    n_runs = len(run_starts)
    per_run_t: list[torch.Tensor] = []
    per_run_names: list[list[str]] = []

    # Polynomials first (block-diagonal: each run's own columns).
    for run_idx in range(n_runs):
        end = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
        run_length = end - run_starts[run_idx]
        if polort >= 0:
            poly = construct_polynomial_matrix(run_length, polort, device=device)
            names = [f"r{run_idx + 1:02d}_poly{p}" for p in range(polort + 1)]
        else:
            poly = torch.zeros((run_length, 0), device=device)
            names = []
        per_run_t.append(poly)
        per_run_names.append(names)

    # Nuisance blocks.
    for block in blocks:
        ncols = block.n_columns
        if ncols == 0:
            continue
        block_demeaned_any_run = False
        block_col_names = block.get_column_names()
        for run_idx in range(n_runs):
            end = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            run_length = end - run_starts[run_idx]
            m = block.get_run(run_idx, run_length).copy()
            col_mean = m.mean(axis=0, keepdims=True)
            if np.max(np.abs(col_mean)) > 1e-4:
                block_demeaned_any_run = True
                m = m - col_mean
            m_t = torch.from_numpy(m).to(device=device, dtype=per_run_t[run_idx].dtype)
            per_run_t[run_idx] = torch.cat([per_run_t[run_idx], m_t], dim=1)
            per_run_names[run_idx].extend(block_col_names)
        if block_demeaned_any_run and verbose:
            print(
                f"  Demeaned nuisance block {block.label!r} per-run (was not zero-mean as supplied)"
            )

    # Optional already-loaded per-run noise PCs (legacy path, e.g. ffs_denoise).
    if noise_pcs is not None:
        for run_idx, pcs in enumerate(noise_pcs):
            if pcs is None or pcs.shape[1] == 0:
                continue
            pcs = pcs.to(device=device, dtype=per_run_t[run_idx].dtype)
            per_run_t[run_idx] = torch.cat([per_run_t[run_idx], pcs], dim=1)
            per_run_names[run_idx].extend(f"noisepc_{c:02d}" for c in range(pcs.shape[1]))

    if verbose:
        for run_idx, m in enumerate(per_run_t):
            print(f"  Run {run_idx + 1}: nuisance shape = {tuple(m.shape)}")

    return NuisanceAssembly(
        per_run=per_run_t,
        per_run_column_names=per_run_names,
        blocks=list(blocks),
    )


def build_nuisance_per_run(
    run_starts: list[int],
    n_timepoints: int,
    polort: int,
    device: torch.device,
    ortvec_files: list[tuple[str, str]] | None = None,
    ortvec_data: torch.Tensor | None = None,
    noise_pcs: list[torch.Tensor] | None = None,
    blocks: list[NuisanceBlock] | None = None,
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

    Internal: this is now a thin shim over ``assemble_per_run_nuisance``,
    which is the canonical entry point taking ``list[NuisanceBlock]``
    directly. Existing callers passing ``ortvec_files`` / ``ortvec_data``
    keep working unchanged; the conversion is silent.
    """
    # Funnel legacy inputs into NuisanceBlocks; pre-built blocks pass through.
    all_blocks: list[NuisanceBlock] = list(blocks) if blocks else []
    if ortvec_files is not None:
        for filepath, label in ortvec_files:
            all_blocks.append(
                make_nuisance_block_from_full_length(
                    filepath,
                    label,
                    run_starts,
                    n_timepoints,
                )
            )
            if verbose:
                print(f"  Loaded ortvec: {filepath} (label={label})")
    if ortvec_data is not None:
        # Pre-loaded tensor: slice + wrap as a single block called "ortvec".
        arr = (
            ortvec_data.detach().cpu().numpy()
            if isinstance(ortvec_data, torch.Tensor)
            else np.asarray(ortvec_data)
        )
        if arr.shape[0] != n_timepoints:
            print(f"ERROR: ortvec_data has {arr.shape[0]} rows, expected {n_timepoints}")
            sys.exit(1)
        all_blocks.append(
            NuisanceBlock(
                label="ortvec",
                per_run=_slice_full_length_per_run(arr, run_starts, n_timepoints),
                source=[None] * len(run_starts),
            )
        )

    assembly = assemble_per_run_nuisance(
        blocks=all_blocks,
        run_starts=run_starts,
        n_timepoints=n_timepoints,
        polort=polort,
        device=device,
        noise_pcs=noise_pcs,
        verbose=verbose,
    )
    return assembly.per_run


def build_nuisance_block_diag(
    run_starts: list[int],
    n_timepoints: int,
    polort: int,
    device: torch.device,
    ortvec_files: list[tuple[str, str]] | None = None,
    blocks: list[NuisanceBlock] | None = None,
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

    Ortvec handling depends on the block's ``block_diagonal`` flag:
      * ``block_diagonal=False`` (full-length ``-ortvec``, AFNI semantics) —
        SHARED across runs: the block's vertical concatenation is appended on
        the right as one set of columns, globally demeaned.
      * ``block_diagonal=True`` (``-ortvec_run`` / ``-ortvec_glob``, distinct
        per-run files) — each populated run gets its OWN columns (zero outside
        that run), per-run demeaned. So 6 runs × 3 PCs → 18 block-diagonal
        columns, not 3 shared ones.
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    n_runs = len(run_starts)
    run_lengths = [
        run_starts[i + 1] - run_starts[i] if i < n_runs - 1 else n_timepoints - run_starts[i]
        for i in range(n_runs)
    ]

    # Polynomial block-diag.
    poly_blocks = []
    for run_len in run_lengths:
        if polort >= 0:
            poly_blocks.append(construct_polynomial_matrix(run_len, polort, device))
        else:
            poly_blocks.append(torch.zeros((run_len, 0), device=device))
    nuisance_design = torch.block_diag(*poly_blocks)

    # Merge legacy ortvec_files into blocks.
    all_blocks: list[NuisanceBlock] = list(blocks) if blocks else []
    if ortvec_files:
        if verbose:
            print(f"  Loading {len(ortvec_files)} ortvec file(s)...")
        for filepath, label in ortvec_files:
            all_blocks.append(
                make_nuisance_block_from_full_length(
                    filepath,
                    label,
                    run_starts,
                    n_timepoints,
                )
            )

    if not all_blocks:
        if verbose:
            print(f"  Nuisance design shape: {nuisance_design.shape}")
        return nuisance_design

    for block in all_blocks:
        ncols = block.n_columns
        if ncols == 0:
            continue

        if block.block_diagonal:
            # Per-run regressors (-ortvec_run / -ortvec_glob): each populated run
            # gets its OWN ncols columns, zero outside that run — block-diagonal,
            # like the polynomials. So 6 runs × 3 PCs -> 18 columns, not 3 shared.
            # Demean each run's block independently (per-run, so it doesn't fight
            # that run's polort intercept).
            col_groups: list[np.ndarray] = []
            demeaned = False
            for i in range(n_runs):
                m = block.per_run[i]
                if m is None:
                    continue
                m = np.asarray(m, dtype=np.float32).copy()
                col_mean = m.mean(axis=0, keepdims=True)
                if np.max(np.abs(col_mean)) > 1e-4:
                    m = m - col_mean
                    demeaned = True
                full = np.zeros((n_timepoints, m.shape[1]), dtype=np.float32)
                full[run_starts[i] : run_starts[i] + run_lengths[i]] = m
                col_groups.append(full)
            if not col_groups:
                continue
            stacked = np.concatenate(col_groups, axis=1)
            ortvec_tensor = torch.from_numpy(stacked).to(device=device, dtype=nuisance_design.dtype)
            nuisance_design = torch.cat([nuisance_design, ortvec_tensor], dim=1)
            if verbose:
                if demeaned:
                    print(f"    Demeaned ortvec {block.label!r} per-run (was not zero-mean)")
                print(
                    f"    {block.label}: {ortvec_tensor.shape[1]} regressor(s) "
                    f"(block-diagonal, {len(col_groups)} run(s) × {ncols})"
                )
            continue

        # Shared full-length regressor (AFNI -ortvec): stack per-run matrices into
        # one set of columns spanning all runs. Missing runs come out as zeros.
        full = np.vstack([block.get_run(i, run_lengths[i]) for i in range(n_runs)]).astype(
            np.float32
        )
        col_mean = full.mean(axis=0, keepdims=True)
        if np.max(np.abs(col_mean)) > 1e-4:
            full = full - col_mean
            if verbose:
                print(f"    Demeaned ortvec {block.label!r} (was not zero-mean)")
        ortvec_tensor = torch.from_numpy(full).to(
            device=device,
            dtype=nuisance_design.dtype,
        )
        nuisance_design = torch.cat([nuisance_design, ortvec_tensor], dim=1)
        if verbose:
            print(f"    {block.label}: {ortvec_tensor.shape[1]} regressor(s)")

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


def add_verbose_arg(parser_or_group, default: int = 1, dest: str = "verb"):
    """Register canonical verbosity flags on a parser or argument group.

    Canonical flag is ``-verb {0,1,2}`` (integer). ``-verbose`` and ``-quiet``
    are silent aliases that set the same dest — ``-verbose`` to 2, ``-quiet``
    to 0. All three are grouped under a "Verbosity" heading in --help so the
    aliases look intentional rather than accidental.

    Parameters
    ----------
    parser_or_group : argparse.ArgumentParser or argparse._ArgumentGroup
        Where to add the flags. Pass a parser to create a new "Verbosity"
        group; pass an existing group to add flags to it directly.
    default : int, default=1
        Default verbosity level when none of the flags are given.
    dest : str, default="verb"
        Destination attribute name on the parsed args. Leave as "verb"
        unless a tool needs a different name for backwards compatibility.

    Examples
    --------
    >>> parser = argparse.ArgumentParser()
    >>> add_verbose_arg(parser)
    >>> args = parser.parse_args(["-verbose"])
    >>> args.verb
    2
    """
    import argparse

    if isinstance(parser_or_group, argparse.ArgumentParser):
        group = parser_or_group.add_argument_group("Verbosity")
    else:
        group = parser_or_group

    group.add_argument(
        "-verb",
        type=int,
        choices=[0, 1, 2],
        default=default,
        metavar="LEVEL",
        dest=dest,
        help=f"Verbosity: 0=silent, 1=normal, 2=debug (default: {default}).",
    )
    group.add_argument(
        "-verbose",
        dest=dest,
        action="store_const",
        const=2,
        help="Alias for -verb 2.",
    )
    group.add_argument(
        "-quiet",
        dest=dest,
        action="store_const",
        const=0,
        help="Alias for -verb 0.",
    )


def parse_device_arg(
    device_spec: str | None,
) -> tuple[torch.device, int | None, int | None]:
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
        except ValueError as err:
            raise ValueError(
                f"Invalid device specification: {device_spec}. "
                "Expected format: 'cpu', 'cuda', 'cpu,N', 'cuda,N', or 'mps'"
            ) from err

    # Create device
    if device_type == "cpu":
        device = torch.device("cpu")
    elif device_type == "cuda":
        if cuda_device_id is not None:
            device = torch.device(f"cuda:{cuda_device_id}")
        else:
            device = torch.device("cuda")
    elif device_type == "mps":
        device = torch.device("mps")
    else:
        raise ValueError(f"Unknown device type: {device_type}. Use 'cpu', 'cuda', or 'mps'.")

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
    from fastfuncstuff.design.builder import parse_hrf_model

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

        print(
            f"  HRF model: {hrf_model_name} (window: {fir_bot:.1f}-{fir_top:.1f}s, {n_basis} basis functions)"
        )

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
            print(
                f"  HRF model: {hrf_model_name} (canonical + temporal derivative, 2 basis functions per condition)"
            )
        elif hrf_model_name == "SPMG3":
            n_basis = 3  # Canonical + temporal + dispersion derivatives
            basis_suffixes = ["_canonical", "_timederiv", "_dispderiv"]
            print(
                f"  HRF model: {hrf_model_name} (canonical + time + dispersion derivatives, 3 basis functions per condition)"
            )

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
            print(
                "  FIR already provides time-resolved estimates; single-trial refitting is redundant"
            )
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

    from fastfuncstuff.design.hrf import get_hrf_library, get_spmg1_hrf
    from fastfuncstuff.design.matrices import (
        convolve_hrf_microtime,
        make_fir_design,
        make_tent_design,
    )

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
            print(
                f"  Building {hrf_model_name} design matrix ({n_basis} basis functions per condition)"
            )

            if hrf_model_name == "FIR":
                bins_per_tr = int(round(tr / microtime_dt))
                onset_matrix_tr = onset_matrix_micro[::bins_per_tr, :]
                onset_matrix_tr = onset_matrix_tr[:n_timepoints, :]
                task_design = make_fir_design(
                    onsets=onset_matrix_tr,
                    n_lags=n_basis,
                    n_timepoints=n_timepoints,
                    device=device,
                )
            elif hrf_model_name in ("TENT", "TENTZERO"):
                onset_times_list = []
                for cond_idx in range(n_conditions):
                    cond_all_runs = []
                    for run_idx, run_onsets in enumerate(all_onsets[cond_idx]):
                        if len(run_onsets) > 0:
                            cond_all_runs.append(run_onsets + run_starts[run_idx] * tr)
                    if len(cond_all_runs) > 0:
                        onset_times_list.append(np.concatenate(cond_all_runs))
                    else:
                        onset_times_list.append(np.array([], dtype=float))

                task_design = make_tent_design(
                    onset_times_list=onset_times_list,
                    bot=fir_bot,
                    top=fir_top,
                    n_basis=n_basis,
                    tr=tr,
                    n_timepoints=n_timepoints,
                    zero_edges=(hrf_model_name == "TENTZERO"),
                    device=device,
                )
            else:
                print(f"ERROR: Unknown FIR model: {hrf_model_name}")
                sys.exit(1)

            print(
                f"  Design shape: {task_design.shape[0]} timepoints × {task_design.shape[1]} regressors"
            )
            print(f"    ({n_conditions} conditions × {n_basis} basis functions)")

        else:
            # Canonical HRF: Convolve with assumed shape (with optional derivatives)
            if hrf_model_name in ("SPMG2", "SPMG3"):
                # SPM canonical with derivatives
                print(f"  Using SPM canonical HRF with derivatives ({hrf_model_name})")
                from fastfuncstuff.design.hrf import get_spm_hrf_with_derivatives

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
                        task_columns.append(design_per_basis[basis_idx][:, cond_idx : cond_idx + 1])

                # Concatenate all columns
                task_design = torch.cat(task_columns, dim=1)
                print(
                    f"  Design shape: {task_design.shape[0]} timepoints × {task_design.shape[1]} regressors"
                )
                print(f"    ({n_conditions} conditions × {n_basis} basis functions)")

            elif hrf_model_name == "SPMG1":
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
            elif hrf_model_name == "GLMSINGLE":
                print("  Using canonical GLMsingle HRF")
                hrf = get_hrf_library(
                    mode="glmsingle",
                    microtime_dt=microtime_dt,
                    stim_duration=0.0,
                    hrf_duration=32.0,
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


def preflight_check(
    input_files: list[str],
    onset_files: list[str] | None = None,
    ortvec_files: list[tuple[str, str]] | None = None,
    hrf_opt_prefix: str | None = None,
    denoise_prefix: str | None = None,
) -> None:
    """
    Run pre-flight input checks before slow data loading.

    Validates file existence and basic consistency (onset row counts, ortvec
    row counts) using only fast header reads and line counts. Collects ALL
    errors before exiting so users can fix everything in one pass.

    Parameters
    ----------
    input_files : list of str
        Resolved list of input NIfTI run files (already validated to exist).
    onset_files : list of str, optional
        AFNI timing files — one per condition. Each must have exactly n_runs rows.
    ortvec_files : list of (path, label) tuples, optional
        Nuisance regressor files. Each must exist and have exactly total_timepoints rows.
    hrf_opt_prefix : str, optional
        Prefix for HRFoptfast outputs. Checks `{prefix}_hrf_index.nii.gz` exists.
    denoise_prefix : str, optional
        Prefix for 3dDenoisefast outputs. Checks `{prefix}_noise_pcs.xmat.1D` exists.
    """
    from fastfuncstuff.io.afni import load_nifti

    errors: list[str] = []
    n_runs = len(input_files)

    # ------------------------------------------------------------------
    # 1. Onset files: each file must have exactly n_runs non-empty rows
    # ------------------------------------------------------------------
    if onset_files:
        for onset_file in onset_files:
            try:
                with open(onset_file) as fh:
                    rows = [
                        ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")
                    ]
                if len(rows) != n_runs:
                    errors.append(
                        f"  Onset file '{onset_file}': {len(rows)} rows "
                        f"but {n_runs} runs were specified"
                    )
            except OSError as exc:
                errors.append(f"  Cannot read onset file '{onset_file}': {exc}")

    # ------------------------------------------------------------------
    # 2. HRFopt output file
    # ------------------------------------------------------------------
    if hrf_opt_prefix:
        hrf_index_file = f"{hrf_opt_prefix}_hrf_index.nii.gz"
        if not Path(hrf_index_file).exists():
            errors.append(f"  -hrf_opt file not found: {hrf_index_file}")

    # ------------------------------------------------------------------
    # 3. Denoise output file(s)
    # ------------------------------------------------------------------
    if denoise_prefix:
        # Check for per-run files first (e.g., prefix_run01_selected_PCs.txt)
        # If not found, check for Denoisefast format (prefix_noise_pcs.xmat.1D)
        per_run_files_exist = False
        per_run_missing = []

        for run_idx in range(1, n_runs + 1):
            run_file = Path(f"{denoise_prefix}_run{run_idx:02d}_selected_PCs.txt")
            if run_file.exists():
                per_run_files_exist = True
            else:
                per_run_missing.append(run_file.name)

        if per_run_files_exist:
            # At least one per-run file exists - warn about missing ones but don't error
            # (empty/missing files are allowed for per-run format)
            if per_run_missing:
                print("  Note: Some per-run denoise files not found (will use no regressors):")
                for name in per_run_missing[:3]:
                    print(f"    {name}")
                if len(per_run_missing) > 3:
                    print(f"    ... and {len(per_run_missing) - 3} more")
        else:
            # No per-run files found - check for Denoisefast format
            noise_pc_file = f"{denoise_prefix}_noise_pcs.xmat.1D"
            if not Path(noise_pc_file).exists():
                errors.append(
                    f"  -denoise files not found:\n"
                    f"    Neither per-run files ({denoise_prefix}_run01_selected_PCs.txt, etc.)\n"
                    f"    nor Denoisefast format ({noise_pc_file})"
                )

    # ------------------------------------------------------------------
    # 4. Ortvec files: must exist and have correct row count
    #    (requires reading NIfTI headers to get total_timepoints)
    # ------------------------------------------------------------------
    if ortvec_files:
        # Fast header-only reads to get total timepoints
        total_timepoints = 0
        header_ok = True
        for nii_file in input_files:
            try:
                shape = load_nifti(nii_file).shape
                total_timepoints += shape[3] if len(shape) >= 4 else 1
            except Exception as exc:
                errors.append(f"  Cannot read NIfTI header '{nii_file}': {exc}")
                header_ok = False

        if header_ok:
            for ortvec_file, label in ortvec_files:
                if not Path(ortvec_file).exists():
                    errors.append(f"  -ortvec file not found: {ortvec_file} (label={label})")
                    continue
                try:
                    with open(ortvec_file) as fh:
                        rows = [ln for ln in fh if ln.strip() and not ln.strip().startswith("#")]
                    if len(rows) != total_timepoints:
                        errors.append(
                            f"  -ortvec file '{ortvec_file}' (label={label}): "
                            f"{len(rows)} rows but expected {total_timepoints} "
                            f"(sum of all run lengths)"
                        )
                except OSError as exc:
                    errors.append(f"  Cannot read ortvec file '{ortvec_file}': {exc}")

    # ------------------------------------------------------------------
    # Report and exit
    # ------------------------------------------------------------------
    if errors:
        print()
        print("=" * 60)
        print("PRE-FLIGHT CHECK FAILED — fix these before re-running:")
        print("=" * 60)
        for err in errors:
            print(err)
        print()
        sys.exit(1)
