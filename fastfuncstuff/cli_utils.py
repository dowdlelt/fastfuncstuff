"""
Shared utility functions for CLI tools.

This module contains common utilities used across multiple CLI tools to avoid
code duplication. Functions include argument parsing, CV strategy parsing,
data loading, and output formatting.
"""

from __future__ import annotations

import argparse
import contextlib
import glob as glob_module
import itertools
import re
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Re-exported: cli_help is the torch-free home for these, but cli_utils was the
# original import site for the formatter and is the natural one for suggest().
from fastfuncstuff.cli_help import (  # noqa: F401
    FfsArgumentParser,
    FfsHelpFormatter,
    ScannableHelpFormatter,
    canonical_option_strings,
    suggest,
)
from fastfuncstuff.design.trim import TrimSpec, TrimTimingReport, shift_onsets_for_trim
from fastfuncstuff.utils import suppress_io_progress

_NIFTI_EXTENSIONS = (".nii.zst", ".nii.gz", ".nii")


@contextlib.contextmanager
def spinner(
    message: str,
    stream=None,
    interval: float = 0.1,
    *,
    enabled: bool = True,
    leave: bool = True,
) -> Iterator[None]:
    """Show a braille spinner next to ``message`` while the ``with`` block runs.

    For opaque single-shot work a progress bar can't measure — reading one big
    run off disk, a slow decompress — where the only honest signal is "still
    going". Animates on a background thread so the wrapped call is unblocked, and
    only when ``stream`` is a real terminal. On exit the line is rewritten as
    ``message done (1.2s)`` and left in place, so the transcript keeps a record of
    every step and its cost; pass ``leave=False`` to clear it instead (for callers
    whose own result line is the completion notice).

    Non-interactive output gets no ``\\r`` spam: just the single final
    ``message ... done (1.2s)`` line once the block finishes.

    Parameters
    ----------
    enabled : bool
        When false, run the block with no output at all (quiet/verbosity gates).
    """
    if not enabled:
        yield
        return

    stream = stream or sys.stderr
    tty = getattr(stream, "isatty", lambda: False)()
    t0 = time.perf_counter()

    if not tty:
        try:
            with suppress_io_progress():
                yield
        finally:
            if leave:
                stream.write(f"{message} ... done ({time.perf_counter() - t0:.1f}s)\n")
                stream.flush()
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
        with suppress_io_progress():
            yield
    finally:
        stop.set()
        thread.join()
        stream.write("\r\033[K")  # carriage return + clear-to-end-of-line
        if leave:
            stream.write(f"✔ {message} done ({time.perf_counter() - t0:.1f}s)\n")
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


CLI_RULE_WIDTH = 70


def print_cli_section(title: str, *, leading_blank: bool = True) -> None:
    """Print a consistently spaced section heading for interactive CLI output."""
    if leading_blank:
        print()
    print(title)
    print("-" * CLI_RULE_WIDTH)


def print_cli_footer(tool_name: str, *, elapsed_seconds: float | None = None) -> None:
    """Print a standard completion block with an optional elapsed time."""
    print()
    print("=" * CLI_RULE_WIDTH)
    print(f"{tool_name} complete")
    if elapsed_seconds is not None:
        print(f"Elapsed: {elapsed_seconds:.1f}s")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * CLI_RULE_WIDTH)


def print_cli_header(tool_name: str, subtitle: str = "") -> None:
    """
    Print standardized CLI header with timestamp.

    Parameters
    ----------
    tool_name : str
        Name of the CLI tool (e.g., "3dDenoisefast")
    subtitle : str, optional
        Additional subtitle text to display
    """
    print("=" * CLI_RULE_WIDTH)
    print(tool_name)
    if subtitle:
        print(subtitle)
    print("=" * CLI_RULE_WIDTH)
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
    mask_array: np.ndarray | None = None,
    blur_fwhm: float | None = None,
    do_scale: bool = False,
    device: torch.device = torch.device("cpu"),
    force_cpu: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
    load_threads: int | None = None,
    drop_first: int = 0,
    drop_last: int = 0,
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
    mask_array : np.ndarray, optional
        Already-computed mask (3D volume or flat boolean), used instead of
        *mask_file*. For callers that load more voxels than they will analyse --
        a union of a fit mask and a noise-pool mask, say -- and so cannot name
        the loaded region with a single file.
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
    load_threads : int, optional
        Runs to decode concurrently (``-load_threads``). Default: auto from
        CPU count and free RAM; 1 disables threading.
    drop_first, drop_last : int, default=0
        TRs dropped from each end of every run (``-drop_first``/``-drop_last``).
        Applied during the load, so ``run_starts``, ``n_timepoints`` and the
        percent-signal run means all describe the retained window. Event timing
        must be shifted to match -- pass the same :class:`~fastfuncstuff.design.trim.TrimSpec`
        to :func:`parse_timing_spec`.

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
        from fastfuncstuff.io.afni import (
            _peek_run_length,
            load_afni_mask,
            load_and_concatenate_runs,
            load_nifti,
        )
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
    if mask_array is not None:
        mask_flat = np.asarray(mask_array).reshape(-1).astype(bool)
        mask = mask_flat.reshape(volume_shape)
        if verbose:
            print(f"  Mask: supplied array ({mask_flat.sum():,} voxels)")
    elif mask_file:
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
        # Synthesise the trimmed length, so a dry run exercises the same shapes
        # (and the same design alignment) the real run would see.
        run_length = max(1, run_length - drop_first - drop_last)

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

    # Estimate total timepoints. Header-only peek: this number only feeds the
    # CPU-vs-GPU device decision below, and the runs are about to be decoded for
    # real by load_and_concatenate_runs. Fully decoding each run here just to read
    # its 4th dimension doubled startup on multi-run datasets, and for .nii.zst it
    # meant a whole extra decompress-to-disk pass per run.
    total_timepoints = 0
    for f in input_files:
        n_tp = _peek_run_length(f)
        if n_tp is None:
            img = load_nifti(f)
            n_tp = img.shape[3] if len(img.shape) > 3 else 1
        total_timepoints += max(0, n_tp - drop_first - drop_last)

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

    # Blur is per-run work, so it rides along inside the shared loader rather
    # than justifying a second, slower load path: the old fork here decoded
    # serially, did the slow volume-major reshape, and then torch.cat'd a list
    # of runs (~2x peak). The loader threads the decode, fills one preallocated
    # buffer, and hands the callback each run while it is the only copy alive.
    per_run_fn = None
    if blur_fwhm is not None:
        if verbose:
            print(f"\n  Applying Gaussian blur (FWHM = {blur_fwhm} mm)...")

        def per_run_fn(run_data, _run_idx):
            # The loader hands us (n_voxels, n_tps) in C order and, because a
            # blur needs whole volumes, before any mask -- so the view back to
            # (x, y, z, t) is free and complete.
            n_tps = run_data.shape[1]
            blurred = gaussian_blur_3d(
                run_data.cpu().numpy().reshape(*volume_shape, n_tps),
                fwhm_mm=blur_fwhm,
                voxel_sizes=voxel_sizes,
                device=device,
                verbose=False,
            )
            out = torch.from_numpy(blurred.reshape(-1, n_tps))
            return out.to(run_data.device)

    data, run_starts = load_and_concatenate_runs(
        [Path(f) for f in input_files],
        device=device,
        keep_on_cpu=keep_on_cpu,
        mask_flat=mask_flat,
        load_threads=load_threads,
        per_run_fn=per_run_fn,
        drop_first=drop_first,
        drop_last=drop_last,
    )

    if verbose and (drop_first or drop_last):
        print(
            f"  Dropped {TrimSpec(drop_first, drop_last).describe()} from each of "
            f"{len(input_files)} run(s) → {data.shape[1]} timepoints retained"
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


def restrict_voxels(
    data: torch.Tensor,
    keep: torch.Tensor,
    volume_shape: tuple,
    mask_flat: np.ndarray | None,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, int]:
    """
    Drop voxels from a (n_voxels, n_timepoints) matrix and update its masks.

    *keep* is a boolean tensor over the CURRENT rows of *data* (i.e. already
    within *mask_flat* if one is active).  The returned ``mask``/``mask_flat``
    are full-volume, so every downstream unmask-and-save keeps working and the
    dropped voxels come back as zeros in the output volumes.

    Returns ``(data, mask, mask_flat, n_voxels)``.
    """
    keep_np = keep.detach().cpu().numpy().astype(bool)
    n_full = int(np.prod(volume_shape))

    new_flat = np.zeros(n_full, dtype=bool)
    if mask_flat is not None:
        new_flat[np.flatnonzero(mask_flat)[keep_np]] = True
    else:
        new_flat[keep_np] = True

    data = data[torch.as_tensor(keep_np, device=data.device)]
    new_mask = new_flat.reshape(volume_shape)
    return data, new_mask, new_flat, int(new_flat.sum())


def find_constant_voxels(
    data: torch.Tensor,
    run_starts: list[int],
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Boolean mask (True = usable) of voxels with real variance in EVERY run.

    Voxels that are flat in any run — out-of-FoV background, the empty corners
    left by an oblique rotation, dead slices — cannot be fit or cross-validated.
    Worse, they score R²=1: SS_res and SS_tot are both 0, and the 1e-10 floor on
    SS_tot turns ``1 - 0/0`` into a perfect fit. Drop them, don't model them.
    """
    n_voxels, n_timepoints = data.shape
    valid = torch.ones(n_voxels, dtype=torch.bool, device=data.device)
    for run_idx in range(len(run_starts)):
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx + 1 < len(run_starts) else n_timepoints
        valid &= data[:, start_tp:end_tp].std(dim=1) > tol
    return valid


def compute_automask_from_data(
    data: torch.Tensor,
    run_starts: list[int],
    volume_shape: tuple,
    mask_flat: np.ndarray | None = None,
    dilate_extra: int = 4,
    verbose: bool = True,
) -> np.ndarray:
    """
    AFNI-style automask (full-volume flat bool) from the mean of the first run.

    The mean is taken over run 1 only — enough signal to find the brain, and it
    avoids materialising the whole timeseries mean for long concatenations.
    """
    from fastfuncstuff.processing.mask import automask as _automask

    end_tp = run_starts[1] if len(run_starts) > 1 else data.shape[1]
    mean_1d = data[:, :end_tp].mean(dim=1).cpu()

    if mask_flat is not None:
        mean_full = torch.zeros(int(np.prod(volume_shape)), dtype=mean_1d.dtype)
        mean_full[torch.from_numpy(mask_flat)] = mean_1d
        mean_3d = mean_full.reshape(volume_shape)
    else:
        mean_3d = mean_1d.reshape(volume_shape)

    auto_mask_3d = _automask(mean_3d, dilate_extra=dilate_extra, verbose=verbose)
    auto_flat = auto_mask_3d.numpy().flatten().astype(bool)
    if mask_flat is not None:
        auto_flat = auto_flat & mask_flat
    return auto_flat


def apply_automask(
    data: torch.Tensor,
    run_starts: list[int],
    volume_shape: tuple,
    mask_flat: np.ndarray | None,
    dilate_extra: int = 4,
    verbose: bool = True,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, int]:
    """
    Compute an automask and restrict *data* to it.

    Returns ``(data, mask, mask_flat, n_voxels)`` like :func:`restrict_voxels`.
    """
    auto_flat = compute_automask_from_data(
        data, run_starts, volume_shape, mask_flat, dilate_extra, verbose
    )
    if verbose:
        n_full = int(np.prod(volume_shape))
        print(
            f"  Automask: {auto_flat.sum():,} / {n_full:,} voxels "
            f"({100 * auto_flat.sum() / n_full:.1f}%)"
        )

    # auto_flat is full-volume; convert to a keep-mask over the current rows.
    keep_full = torch.from_numpy(auto_flat)
    keep = keep_full[torch.from_numpy(mask_flat)] if mask_flat is not None else keep_full
    return restrict_voxels(data, keep, volume_shape, mask_flat)


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


def save_r2_ceiling_stack(
    layers: Sequence[tuple[torch.Tensor | np.ndarray | None, str]],
    filename: str,
    volume_shape: tuple,
    affine: np.ndarray,
    mask_flat: np.ndarray | None = None,
    header: object | None = None,
) -> str:
    """Save an R2 map together with the ceiling that interprets it, as one stack.

    A held-out R2 of 0.08 means nothing on its own -- it is excellent where the
    reproducible signal tops out at 0.09 and poor where the ceiling is 0.4 -- so
    the ceiling and the ``explainable_R2`` ratio belong in the same file as the
    map they qualify, as labelled sub-briks. Loose sibling files invite two
    mistakes this prevents: reading an R2 without its ceiling, and dividing one
    R2 by another R2's ceiling. The second is the dangerous one, because the
    result looks perfectly plausible.

    ``layers`` is ordered ``[(r2, label), (ceiling, ...), (explainable, ...)]``;
    entries whose map is ``None`` are dropped, so a run without a ceiling writes
    a plain 3-D map rather than a one-volume stack. **The caller is responsible
    for passing the R2 the ceiling was actually built from** -- for several tools
    that is not the map with the most obvious filename.

    Used by ffs_denoise, ffs_ridge, ffs_hrfopt, ffs_reml and ffs_denoisatorial so
    the family's outputs can be read the same way and compared to each other.
    """
    from fastfuncstuff.io.afni import save_nifti

    volumes: list[np.ndarray] = []
    labels: list[str] = []
    for values, label in layers:
        if values is None:
            continue
        flat = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
        flat = flat.astype(np.float32, copy=False)
        if mask_flat is not None:
            full = np.zeros(mask_flat.size, dtype=np.float32)
            full[mask_flat] = flat
            flat = full
        volumes.append(flat.reshape(volume_shape))
        labels.append(label)

    if not volumes:
        raise ValueError("save_r2_ceiling_stack needs at least one non-None map")

    stacked = np.stack(volumes, axis=-1) if len(volumes) > 1 else volumes[0]
    save_nifti(
        stacked,
        output_path=filename,
        affine=affine,
        header=header,
        brick_labels=labels if len(labels) > 1 else None,
    )
    return filename


def save_4d_nifti(
    data_flat: torch.Tensor | np.ndarray,
    filename: str,
    volume_shape: tuple,
    affine: np.ndarray,
    mask_flat: np.ndarray | None = None,
    header: object | None = None,
    brick_labels: list[str] | None = None,
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
    brick_labels : list of str, optional
        Per-volume labels. Worth passing whenever the fourth axis is not time --
        an unlabelled stack of parameter maps is unreadable in a viewer.
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

    if brick_labels is not None and len(brick_labels) != n_vols:
        raise ValueError(f"brick_labels has {len(brick_labels)} entries for {n_vols} volumes")

    save_nifti(
        data_4d,
        output_path=filename,
        affine=affine,
        header=header,
        brick_labels=brick_labels,
    )


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

# Per-run transforms a nuisance block may declare. Adding one here (plus a
# branch in apply_nuisance_transform) makes it available to every input mode
# and to design.toml's `transform =` field at once.
NUISANCE_TRANSFORMS = ("none", "deriv", "deriv_back", "deriv_fwd")


def apply_nuisance_transform(arr: np.ndarray, transform: str) -> np.ndarray:
    """Apply a per-run transform to ONE run's regressor block.

    Derivatives follow ``1d_tool.py`` exactly (afni_util.derivative), including
    the length: the output has as many rows as the input, with the edge row the
    difference cannot reach set to zero.

    - ``deriv`` / ``deriv_back`` — backward difference (``1d_tool.py
      -derivative`` / ``-backward_diff``, the afni_proc.py default):
      ``d[t] = v[t] - v[t-1]``, ``d[0] = 0``.
    - ``deriv_fwd`` — forward difference (``-forward_diff``):
      ``d[t] = v[t+1] - v[t]``, ``d[-1] = 0``. Same regressor shifted one TR;
      which one you want depends on where you think the artefact sits relative
      to the motion.

    Per run is not an optimisation: differencing across a run boundary turns the
    between-run offset into a spike in a regressor that then eats real signal.
    """
    if transform in (None, "", "none"):
        return arr
    a = np.asarray(arr, dtype=np.float64)
    out = np.zeros_like(a)
    if transform in ("deriv", "deriv_back"):
        out[1:] = a[1:] - a[:-1]
    elif transform == "deriv_fwd":
        out[:-1] = a[1:] - a[:-1]
    else:
        raise ValueError(f"unknown nuisance transform {transform!r} (known: {NUISANCE_TRANSFORMS})")
    return out


def split_label_transform(label: str) -> tuple[str, str]:
    """Split an ortvec LABEL into ``(label, transform)``.

    ``motion:deriv`` → ``("motion", "deriv")``; a bare label → transform "none".
    One modifier syntax for all four ``-ortvec*`` flags beats four parallel
    flag families, and it round-trips to design.toml's `transform =` field.
    """
    if ":" not in label:
        return label, "none"
    base, _, tf = label.rpartition(":")
    if tf not in NUISANCE_TRANSFORMS:
        raise ValueError(
            f"unknown transform {tf!r} in ortvec label {label!r} "
            f"(known: {', '.join(NUISANCE_TRANSFORMS)})"
        )
    return base, tf


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
    # Per-run transform applied in get_run (see apply_nuisance_transform). Stored
    # rather than baked in so per_run stays the file's raw content, inspectable.
    transform: str = "none"

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
        # Transform before padding: the zero columns of a short run must stay
        # zero, and the derivative is defined on this run's rows alone.
        m = apply_nuisance_transform(m, self.transform).astype(np.float32, copy=False)
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
    # AFNI-style short run token: `.r001_`, `_r6.`, etc. X may be zero-padded.
    # Requires a . or _ before the `r` so it doesn't fire inside ordinary words.
    (r"[._]r(\d+)_", "rN_ (e.g. .r001_)"),
    (r"[._]r(\d+)\b", "rN (e.g. _r06)"),
    (r"(\d+)\.[^.]+$", "trailing N before extension"),
    (r"(\d+)\D*$", "trailing N"),
)


def _infer_run_indices_from_filenames(
    names: list[str],
    n_runs: int,
    allow_sequential_fallback: bool = False,
) -> list[int]:
    """Try patterns in priority order; return 0-indexed run for each name.

    A pattern succeeds when it (a) matches every name, (b) yields unique
    run numbers, and (c) all numbers fall in ``[1, n_runs]``. The first
    success wins.

    Token patterns come first because they stay correct even when the files
    would sort in the wrong order (``r1, r10, r2``). Only if every pattern
    fails do we consider the count: when ``allow_sequential_fallback`` is set
    and the number of files exactly equals ``n_runs``, the caller-sorted list
    is trusted 1:1 (file *i* → run *i*). This is what lets un-tokenised but
    complete file sets work without any BIDS-style naming. If neither route
    resolves, raise ``ValueError`` listing what each pattern tried.
    """
    failures: list[str] = []
    for pat, desc in _RUN_INDEX_PATTERNS:
        matches = [re.search(pat, n, re.IGNORECASE) for n in names]
        if not all(matches):
            unmatched = [n for n, m in zip(names, matches, strict=False) if not m]
            failures.append(f"{desc}: did not match {unmatched[:3]}")
            continue
        nums = [int(m.group(1)) for m in matches if m is not None]
        dup_counts = Counter(nums)
        dups = [n for n, c in dup_counts.items() if c > 1]
        if dups:
            failures.append(f"{desc}: duplicate run numbers {sorted(dups)}")
            continue
        if any(n < 1 or n > n_runs for n in nums):
            failures.append(f"{desc}: out-of-range run numbers {sorted(nums)} for n_runs={n_runs}")
            continue
        return [n - 1 for n in nums]

    # No run token parsed. If the glob is already complete (one file per run),
    # sorted order is unambiguous — assign sequentially rather than erroring.
    if allow_sequential_fallback and len(names) == n_runs:
        return list(range(n_runs))

    raise ValueError(
        "Could not infer per-run indices from filenames. Patterns tried:\n  "
        + "\n  ".join(failures)
        + f"\n  Files: {names}\n"
        + (
            f"  Sequential fallback not used: {len(names)} files vs {n_runs} runs "
            "(counts must match to assign by sorted order).\n"
            if allow_sequential_fallback
            else ""
        )
        + "Use -ortvec_run FILE LABEL RUN to assign explicitly, or rename "
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


def run_lengths_from_starts(run_starts: list[int], n_timepoints: int) -> list[int]:
    """Per-run TR counts implied by ``run_starts`` and the concatenated length.

    Re-exported from :mod:`fastfuncstuff.design.matrices` so CLI and library
    code cannot drift apart on how runs are split.
    """
    from fastfuncstuff.design.matrices import run_lengths_from_starts as _impl

    return _impl(run_starts, n_timepoints)


def make_nuisance_block_from_full_length(
    path: str | Path,
    label: str,
    run_starts: list[int],
    n_timepoints: int,
    transform: str = "none",
    trim: TrimSpec | None = None,
) -> NuisanceBlock:
    """Mode 1: one file with rows for *all* runs concatenated (pre-padded).

    Under ``-drop_first``/``-drop_last`` the file may be either already trimmed
    (matching the loaded data) or at the original acquired length, in which case
    each run's block is trimmed here. Regressors like motion parameters are
    produced from the untrimmed run, so demanding a pre-trimmed file would make
    the flag unusable with the files people actually have.
    """
    from fastfuncstuff.design.hrf_selection import load_nuisance_file
    from fastfuncstuff.design.trim import trim_run_series

    arr = load_nuisance_file(path)
    trimmed_lengths = run_lengths_from_starts(run_starts, n_timepoints)

    if trim is not None and trim.active and arr.shape[0] != n_timepoints:
        # Walk the file on its own (untrimmed) run grid and trim each block.
        untrimmed = [n + trim.total for n in trimmed_lengths]
        if arr.shape[0] != sum(untrimmed):
            raise ValueError(
                f"{path}: has {arr.shape[0]} rows, but the design has {n_timepoints} "
                f"timepoints after dropping {trim.describe()} per run "
                f"(an untrimmed file would have {sum(untrimmed)} rows)."
            )
        blocks = []
        off = 0
        for i, n_un in enumerate(untrimmed):
            blocks.append(trim_run_series(arr[off : off + n_un], trimmed_lengths[i], trim, path))
            off += n_un
        per_run = blocks
        return NuisanceBlock(
            label=label,
            per_run=per_run,
            source=[str(path)] * len(run_starts),
            transform=transform,
        )

    if arr.shape[0] != n_timepoints:
        raise ValueError(
            f"{path}: has {arr.shape[0]} rows, design has {n_timepoints} total timepoints"
        )
    per_run = _slice_full_length_per_run(arr, run_starts, n_timepoints)
    return NuisanceBlock(
        label=label,
        per_run=per_run,
        source=[str(path)] * len(run_starts),
        transform=transform,
    )


def make_nuisance_block_from_per_run_file(
    path: str | Path,
    label: str,
    run_idx_1based: int,
    run_starts: list[int],
    n_timepoints: int,
    transform: str = "none",
    trim: TrimSpec | None = None,
) -> NuisanceBlock:
    """Mode 2: one file that covers exactly one run; other runs get zeros.

    Accepts a trimmed or an untrimmed file (see
    :func:`make_nuisance_block_from_full_length`).
    """
    from fastfuncstuff.design.hrf_selection import load_nuisance_file
    from fastfuncstuff.design.trim import trim_run_series

    n_runs = len(run_starts)
    run_idx = run_idx_1based - 1
    if not (0 <= run_idx < n_runs):
        raise ValueError(f"run index {run_idx_1based} out of range [1, {n_runs}] for {path}")
    expected = (run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints) - run_starts[
        run_idx
    ]
    arr = load_nuisance_file(path)
    if trim is not None and trim.active:
        arr = trim_run_series(arr, expected, trim, path)
    elif arr.shape[0] != expected:
        raise ValueError(
            f"{path}: has {arr.shape[0]} rows, run {run_idx_1based} has {expected} timepoints"
        )
    per_run: list[np.ndarray | None] = [None] * n_runs
    per_run[run_idx] = arr
    source: list[str | None] = [None] * n_runs
    source[run_idx] = str(path)
    return NuisanceBlock(
        label=label, per_run=per_run, source=source, block_diagonal=True, transform=transform
    )


def make_nuisance_block_from_glob(
    pattern: str,
    label: str,
    run_starts: list[int],
    n_timepoints: int,
    transform: str = "none",
    trim: TrimSpec | None = None,
) -> NuisanceBlock:
    """Mode 3: glob matches N files; infer per-file run index from filename.

    Runs absent from the glob are zero-padded into the block. Each matched file
    may be trimmed or untrimmed (see :func:`make_nuisance_block_from_full_length`).
    """
    from fastfuncstuff.design.hrf_selection import load_nuisance_file
    from fastfuncstuff.design.trim import trim_run_series

    matched = sorted(Path(p) for p in glob_module.glob(pattern))
    if not matched:
        raise ValueError(f"-ortvec_glob {pattern!r}: matched no files")

    n_runs = len(run_starts)
    run_indices_0 = _infer_run_indices_from_filenames(
        [p.name for p in matched],
        n_runs=n_runs,
        allow_sequential_fallback=True,
    )

    per_run: list[np.ndarray | None] = [None] * n_runs
    source: list[str | None] = [None] * n_runs
    for path, run_idx in zip(matched, run_indices_0, strict=False):
        expected = (run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints) - run_starts[
            run_idx
        ]
        arr = load_nuisance_file(path)
        if trim is not None and trim.active:
            arr = trim_run_series(arr, expected, trim, path)
        elif arr.shape[0] != expected:
            raise ValueError(
                f"{path}: has {arr.shape[0]} rows, run {run_idx + 1} has {expected} timepoints"
            )
        per_run[run_idx] = arr
        source[run_idx] = str(path)

    return NuisanceBlock(
        label=label, per_run=per_run, source=source, block_diagonal=True, transform=transform
    )


def add_ortvec_arguments(parser_or_group, include_legacy: bool = True, prefix: str = "") -> None:
    """Register `-ortvec`, `-ortvec_run`, `-ortvec_glob`, `-ortvec_concat` on a parser/group.

    All four are repeatable. The CLI then funnels them through
    `collect_nuisance_blocks(args, ...)` to get a `list[NuisanceBlock]`.
    Any per-CLI guard in front of that call must test all four flags
    (a subset guard silently drops the omitted mode).

    `prefix` registers a second, independent family (e.g. `prefix="test_"` gives
    `-test_ortvec` …) so one CLI can carry nuisance for two different datasets.
    Pass the same prefix to `collect_nuisance_blocks`.

    Every LABEL accepts a `:transform` modifier (see NUISANCE_TRANSFORMS), so
    the same file can enter twice — once raw, once differenced — without a
    parallel flag family per transform.
    """
    transform_note = (
        " LABEL may carry a transform modifier: LABEL:deriv (per-run backward "
        "difference, as 1d_tool.py -derivative), LABEL:deriv_fwd (forward "
        "difference), LABEL:deriv_back (explicit synonym of :deriv)."
    )
    # argparse would derive these dests itself, but the prefixed family relies
    # on them matching what collect_nuisance_blocks(prefix=...) looks up.
    p = prefix
    dash = prefix.replace("_", "-")
    if include_legacy:
        parser_or_group.add_argument(
            f"-{p}ortvec",
            dest=f"{p}ortvec",
            action="append",
            nargs=2,
            metavar=("FILE", "LABEL"),
            help=(
                "Full-length nuisance regressors (pre-concatenated across all runs). "
                f"Repeatable: -{p}ortvec motion.1D motion -{p}ortvec physio.1D physio"
                + transform_note
            ),
        )
    parser_or_group.add_argument(
        f"-{p}ortvec_run",
        f"-{dash}ortvec-run",
        dest=f"{p}ortvec_run",
        action="append",
        nargs=3,
        metavar=("FILE", "LABEL", "RUN"),
        help=(
            "Per-run nuisance regressors; RUN is 1-based run index. Other runs are zero-padded. "
            f"Repeatable: -{p}ortvec_run motion_r1.1D motion 1 "
            f"-{p}ortvec_run motion_r2.1D motion 2" + transform_note
        ),
    )
    parser_or_group.add_argument(
        f"-{p}ortvec_glob",
        f"-{dash}ortvec-glob",
        dest=f"{p}ortvec_glob",
        action="append",
        nargs=2,
        metavar=("PATTERN", "LABEL"),
        help=(
            "Glob matching per-run nuisance files; run index inferred from filename "
            "(BIDS-style `_run-N_` preferred). Missing runs zero-padded. "
            f"Repeatable: -{p}ortvec_glob 'motion_run-*.1D' motion" + transform_note
        ),
    )
    parser_or_group.add_argument(
        f"-{p}ortvec_concat",
        f"-{dash}ortvec-concat",
        dest=f"{p}ortvec_concat",
        action="append",
        nargs=2,
        metavar=("PATTERN", "LABEL"),
        help=(
            "Glob matching N already-full-length per-run files (e.g. AFNI "
            "mot_demean.r0N.1D — each spans every run with zeros outside its own). "
            f"Expanded into N -{p}ortvec entries labelled LABEL01, LABEL02, … "
            "(width auto-padded from n_runs). Repeatable: "
            f"-{p}ortvec_concat 'mot_demean.r0*.1D' motion"
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
    prefix: str = "",
    trim: TrimSpec | None = None,
) -> list[NuisanceBlock]:
    """Translate argparse Namespace fields into a list of NuisanceBlock.

    Recognises ``args.ortvec``, ``args.ortvec_run``, ``args.ortvec_glob``,
    ``args.ortvec_concat`` — any combination, all repeatable, all optional.
    Designed to plug into any CLI that called ``add_ortvec_arguments`` on
    its parser.

    ``prefix`` must match the one given to ``add_ortvec_arguments``; it selects
    the prefixed flag family (``-test_ortvec`` …) instead of the plain one, and
    ``run_starts``/``n_timepoints`` then describe that family's dataset.

    ``trim`` lets each file be supplied either already trimmed or at its
    original acquired length; without it, an untrimmed motion file against
    ``-drop_first`` data is a hard row-count error.
    """
    p = prefix
    blocks: list[NuisanceBlock] = []
    for path, raw_label in getattr(args, f"{p}ortvec", None) or []:
        label, tf = split_label_transform(raw_label)
        blocks.append(
            make_nuisance_block_from_full_length(
                path,
                label,
                run_starts,
                n_timepoints,
                transform=tf,
                trim=trim,
            )
        )
        if verbose:
            print(f"  -{p}ortvec: {path} (label={label}, transform={tf})")
    for path, raw_label, run_str in getattr(args, f"{p}ortvec_run", None) or []:
        label, tf = split_label_transform(raw_label)
        try:
            run_idx = int(run_str)
        except ValueError:
            print(f"ERROR: -{p}ortvec_run RUN must be a 1-based integer, got {run_str!r}")
            sys.exit(1)
        blocks.append(
            make_nuisance_block_from_per_run_file(
                path,
                label,
                run_idx,
                run_starts,
                n_timepoints,
                transform=tf,
                trim=trim,
            )
        )
        if verbose:
            print(f"  -{p}ortvec_run: {path} (label={label}, run={run_idx}, transform={tf})")
    for pattern, raw_label in getattr(args, f"{p}ortvec_glob", None) or []:
        label, tf = split_label_transform(raw_label)
        # An unquoted PATTERN is expanded by the shell before argparse sees it.
        # With 3+ matches argparse rejects the leftovers, but with exactly 2 the
        # call parses cleanly and the LABEL silently becomes the second
        # filename -- the glob then matches one run and every other run is
        # zero-padded, i.e. a plausible-looking fit with the nuisance missing
        # from most of the data. A label that names an existing file is the
        # tell; a real label never does.
        if Path(raw_label).exists():
            # Best-effort reconstruction of what the user meant to type: put the
            # wildcard back where the run number is.
            suggestion = re.sub(r"(?<=run[-_])\d+", "*", pattern) or pattern
            raise ValueError(
                f"-{p}ortvec_glob LABEL {raw_label!r} is an existing file, which means the "
                f"shell expanded PATTERN before argparse saw it. Quote the pattern, e.g. "
                f"-{p}ortvec_glob '{suggestion}' <label>"
            )
        block = make_nuisance_block_from_glob(
            pattern,
            label,
            run_starts,
            n_timepoints,
            transform=tf,
            trim=trim,
        )
        blocks.append(block)
        if verbose:
            matched = [s for s in block.source if s]
            print(
                f"  -{p}ortvec_glob: {pattern} (label={label}, transform={tf})"
                f" → {len(matched)} run(s) assigned"
            )
    # -ortvec_concat: glob over already-full-length per-run files; each becomes
    # its own full-length NuisanceBlock with an auto-suffixed label. Equivalent
    # to writing N -ortvec calls.
    for pattern, raw_label in getattr(args, f"{p}ortvec_concat", None) or []:
        label, tf = split_label_transform(raw_label)
        n_runs = len(run_starts)
        for path, suffixed_label in expand_ortvec_concat(pattern, label, n_runs):
            blocks.append(
                make_nuisance_block_from_full_length(
                    path,
                    suffixed_label,
                    run_starts,
                    n_timepoints,
                    transform=tf,
                    trim=trim,
                )
            )
            if verbose:
                print(f"  -{p}ortvec_concat: {path} (label={suffixed_label})")
    return blocks


def append_nuisance_blocks(
    nuisance_per_run: list[torch.Tensor],
    blocks: list[NuisanceBlock],
    run_starts: list[int],
    n_timepoints: int,
) -> list[torch.Tensor]:
    """Concatenate NuisanceBlock columns onto per-run nuisance, then pad to width.

    Blocks are demeaned per run before they join the polynomials, which already
    span the constant. Runs end up with a uniform column count because
    ``compute_qr_projectors`` drops the all-zero padding anyway, and the
    block-diagonal builders downstream assume a rectangular stack.
    """
    n_runs = len(run_starts)
    for run_idx in range(n_runs):
        start_tp = run_starts[run_idx]
        end_tp = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
        run_length = end_tp - start_tp
        for block in blocks:
            if block.n_columns == 0:
                continue
            m = block.get_run(run_idx, run_length).copy()
            col_mean = m.mean(axis=0, keepdims=True)
            if np.max(np.abs(col_mean)) > 1e-4:
                m = m - col_mean
            columns = torch.from_numpy(m).to(
                device=nuisance_per_run[run_idx].device,
                dtype=nuisance_per_run[run_idx].dtype,
            )
            nuisance_per_run[run_idx] = torch.cat([nuisance_per_run[run_idx], columns], dim=1)

    max_cols = max(n.shape[1] for n in nuisance_per_run)
    for run_idx in range(n_runs):
        n_cols = nuisance_per_run[run_idx].shape[1]
        if n_cols < max_cols:
            padding = torch.zeros(
                (nuisance_per_run[run_idx].shape[0], max_cols - n_cols),
                device=nuisance_per_run[run_idx].device,
                dtype=nuisance_per_run[run_idx].dtype,
            )
            nuisance_per_run[run_idx] = torch.cat([nuisance_per_run[run_idx], padding], dim=1)
    return nuisance_per_run


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


def add_trim_args(parser_or_group) -> None:
    """Register ``-drop_first`` / ``-drop_last`` on a parser or argument group.

    ``-skip_first``/``-skip_last`` are accepted as aliases because ``ffs_moco``
    established that spelling for the same operation on a single series.

    Every tool that fits a model to timing must resolve these through
    :func:`trim_spec_from_args` and hand the result to both the loader and
    :func:`parse_timing_spec`; the timing shift is not optional.
    """
    parser_or_group.add_argument(
        "-drop_first",
        "-drop-first",
        "-skip_first",
        "-skip-first",
        dest="drop_first",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Drop the first N TRs of every run (steady-state volumes, say). "
            "Event timing is shifted back by N*TR automatically, and the shift "
            "is reported. An event that began before the retained window but is "
            "still ongoing at its start is kept, with a truncated response."
        ),
    )
    parser_or_group.add_argument(
        "-drop_last",
        "-drop-last",
        "-skip_last",
        "-skip-last",
        dest="drop_last",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Drop the last N TRs of every run. Needs no timing shift (the run's "
            "time origin does not move), but events left past the new run end "
            "are dropped and reported."
        ),
    )


def trim_spec_from_args(args, tr: float | None = None) -> TrimSpec:
    """Build a :class:`TrimSpec` from parsed ``-drop_first``/``-drop_last`` args.

    *tr* is usually unknown until the data is loaded, hence
    :meth:`TrimSpec.with_tr` -- but the spec is needed *before* the load to size
    the buffers, so it is built TR-less first and completed afterwards.
    """
    return TrimSpec(
        drop_first=int(getattr(args, "drop_first", 0) or 0),
        drop_last=int(getattr(args, "drop_last", 0) or 0),
        tr=tr,
    )


def add_load_threads_arg(parser_or_group) -> None:
    """Register ``-load_threads`` on a parser or argument group.

    Decoding runs is thread-parallel (zstd, nibabel and the copy out of the
    file's volume-major layout all release the GIL), worth ~3-4x on multi-run
    loads. The default adapts to CPU count and free RAM; this is the escape
    hatch for a shared machine or a memory-tight host.
    """
    parser_or_group.add_argument(
        "-load_threads",
        "-load-threads",
        dest="load_threads",
        type=int,
        default=None,
        metavar="N",
        help="Runs to decode concurrently when loading (default: auto from CPU "
        "count and free RAM; 1 disables threading). Also settable via "
        "FFS_LOAD_THREADS.",
    )


def add_device_arg(
    parser_or_group,
    *,
    default: str | None = None,
    extra: str = "",
) -> None:
    """Register ``-device`` with the spec the canonical parser actually accepts.

    The help text advertises ``cuda,N`` and ``cpu,N`` deliberately: pinning a
    card matters on a shared GPU, and users only find the form if the tool
    names it. Pair with :func:`setup_device`, never with ``torch.device(...)``
    directly — bare torch rejects every comma form.
    """
    suggest(
        parser_or_group.add_argument(
            "-device",
            default=default,
            metavar="SPEC",
            help="Compute device: auto, cpu, cuda, mps. Append ',N' to pin a CUDA "
            "device (cuda,0) or cap CPU threads (cpu,8)." + (f" {extra}" if extra else ""),
        ),
        # Not choices=: the comma forms are open-ended, which is exactly the
        # case suggest() exists for.
        ("auto", "cpu", "cuda", "mps"),
    )


def resolve_microtime_dt(tr: float, requested_dt: float, verbose: bool = True) -> float:
    """Snap a requested microtime step onto a grid that divides ``tr`` exactly.

    ``-microtime_dt`` goes through here, the way ``-device`` goes through
    :func:`setup_device`. A nominal 0.1 s step is not commensurate with most
    real TRs: TR=1.75 gives ``round(17.5) = 18`` bins, an effective 1.8 s TR,
    and stimulus columns slide ~3.5 s early by TR 60. The snapped step divides
    any TR exactly -- 3.542 s becomes 35 bins of 0.101200 s -- so the decimal
    value is cosmetic and exact TR boundaries are what we keep.
    """
    from fastfuncstuff.design.matrices import commensurate_microtime_dt

    dt = commensurate_microtime_dt(tr, requested_dt)
    if verbose and abs(dt - requested_dt) > 1e-9:
        print(
            f"  Microtime: dt {requested_dt}s -> {dt:.6f}s "
            f"({round(tr / dt)} bins/TR) to divide TR={tr}s exactly"
        )
    return dt


def setup_device(
    device_spec: str | None,
    *,
    tf32: bool = True,
) -> torch.device:
    """Resolve ``-device`` and configure the torch backends in one step.

    The one-call form exists because the two halves were drifting apart: a tool
    that parsed the device by hand also tended to drop the ``cpu,N`` thread
    override on the floor. Registration tools pass ``tf32=REGISTRATION_TF32``.
    """
    from .utils import configure_torch_backends

    device, cpu_threads_override, _cuda_device_id = parse_device_arg(device_spec)
    configure_torch_backends(device, n_threads=cpu_threads_override, tf32=tf32)
    return device


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


@dataclass
class TimingSpec:
    """Event timing parsed from either BIDS ``*_events.tsv`` or AFNI timing files.

    ``all_onsets[condition_idx][run_idx]`` is an ndarray of onset times in
    seconds — the shape every design builder in the repo expects, regardless of
    which input format the user supplied.
    """

    all_onsets: list[list[np.ndarray]]
    durations: list[float]
    condition_labels: list[str]
    from_events: bool
    onset_files: list[str] | None = None
    durations_given: bool = True
    """False when ``-onsets`` was used without ``-durations`` and the caller
    allowed it (see ``allow_missing_durations``); ``durations`` is then all
    zeros and must not be read as a stimulus length."""

    @property
    def n_conditions(self) -> int:
        return len(self.condition_labels)


def parse_timing_spec(
    *,
    events: list[str] | None,
    onsets: list[str] | None,
    durations_arg: list[str] | str | None,
    n_runs: int,
    event_ignore: list[str] | None = None,
    event_cols: tuple[str, str, str] | None = None,
    round_durations: int | None = None,
    input_files: list[str] | None = None,
    verbose: bool = True,
    trim: TrimSpec | None = None,
    run_lengths_tr: list[int] | None = None,
    allow_missing_durations: bool = False,
) -> TimingSpec:
    """Parse the timing spec of an ``ffs_*`` GLM tool into a common structure.

    Single source of truth for both timing paths: BIDS ``-events`` TSVs (with
    the one-file-broadcast-across-runs convention) and AFNI ``-onsets`` timing
    files (one file per condition, one line per run). Every caller used to
    hand-roll this, which is how ``-events`` broadcasting ended up supported in
    some tools and not others.

    Exactly one of *events* / *onsets* must be non-empty. Raises ``ValueError``
    (or ``FileNotFoundError``) with a user-facing message on any problem; the
    CLI is responsible for printing it and exiting.

    When *trim* is given (and carries a TR), every onset is shifted back by
    ``trim.shift_sec`` so the timing describes the *retained* window rather than
    the file on disk -- see :mod:`fastfuncstuff.design.trim`. This is the single
    place that shift happens, so no CLI can wire ``-drop_first`` into its loader
    and forget the timing. *run_lengths_tr* (post-trim, per run) lets events
    stranded past the new run end be dropped as well; without it only the
    start-of-run side is checked.
    """
    from fastfuncstuff.design.bids_events import check_events_pairing, parse_bids_events
    from fastfuncstuff.design.builder import parse_afni_timing_file, parse_durations

    if bool(events) == bool(onsets):
        raise ValueError("Specify exactly one of -onsets/-durations or -events")

    if events:
        # One TSV per run, or a single shared TSV broadcast to every run
        # (a valid BIDS pattern when all runs share the same stimulus timing).
        if len(events) not in (1, n_runs):
            raise ValueError(
                f"-events requires one TSV per run or a single shared TSV: "
                f"got {len(events)} events files but {n_runs} input datasets."
            )

        # Events pair with -input by position. When both sides carry sub/ses/task/run
        # entities, verify the pairing rather than trusting it: a mispaired timing file
        # produces a plausible-looking design that is simply wrong about which run is
        # which, and nothing downstream necessarily notices. The pairing is printed in
        # pairing order either way -- this listing used to be sorted independently of
        # the parse, which made a mispairing look like a display quirk.
        check_events_pairing(input_files, events, n_runs=n_runs, verbose=verbose)

        all_onsets, durations, condition_labels = parse_bids_events(
            event_files=events,
            event_ignore=event_ignore,
            event_cols=event_cols,
            round_durations=round_durations,
            n_runs=n_runs,
        )
        spec = TimingSpec(
            all_onsets=all_onsets,
            durations=durations,
            condition_labels=condition_labels,
            from_events=True,
        )
    else:
        assert onsets is not None
        missing = [f for f in onsets if not Path(f).exists()]
        if missing:
            raise FileNotFoundError(f"Onset file not found: {missing[0]}")

        condition_labels = clean_condition_labels([Path(f).stem for f in onsets])
        # ffs_deconvolve can size its HRF window from -window/-duration instead,
        # so it is the one caller allowed to omit -durations entirely; every
        # other tool guards on it before calling and gets the hard error.
        durations_given = bool(durations_arg)
        if not durations_given:
            if not allow_missing_durations:
                raise ValueError("-durations is required with -onsets")
            durations = [0.0] * len(onsets)
        else:
            durations = parse_durations(durations_arg, len(onsets), condition_labels)
            if round_durations is not None:
                durations = [round(d, round_durations) for d in durations]

        all_onsets = []
        for onset_file in onsets:
            onsets_by_run = parse_afni_timing_file(onset_file)
            if len(onsets_by_run) != n_runs:
                raise ValueError(
                    f"Onset file {onset_file} has {len(onsets_by_run)} runs, "
                    f"but {n_runs} input runs were given."
                )
            all_onsets.append(onsets_by_run)

        spec = TimingSpec(
            all_onsets=all_onsets,
            durations=durations,
            condition_labels=condition_labels,
            from_events=False,
            onset_files=list(onsets),
            durations_given=durations_given,
        )

    # Shift AFTER parsing, so both timing formats get it and the numbers printed
    # below are the ones the design is actually built from.
    if trim is not None:
        apply_trim_to_timing(spec, trim, run_lengths_tr, n_runs=n_runs, verbose=verbose)

    if verbose:
        print(f"  Conditions: {spec.n_conditions} ({', '.join(spec.condition_labels)})")
        for cidx, label in enumerate(spec.condition_labels):
            n_events = sum(len(spec.all_onsets[cidx][r]) for r in range(n_runs))
            dur_note = f" (duration={spec.durations[cidx]:.3f}s)" if spec.durations_given else ""
            print(f"    {label}: {n_events} events across {n_runs} runs{dur_note}")

    return spec


def apply_trim_to_timing(
    spec: TimingSpec,
    trim: TrimSpec,
    run_lengths_tr: list[int] | None = None,
    n_runs: int | None = None,
    verbose: bool = True,
) -> TrimTimingReport | None:
    """Shift a :class:`TimingSpec` in place to match ``-drop_first`` trimmed data.

    Separate from :func:`parse_timing_spec` because most CLIs parse their timing
    before loading any data -- they need the condition list to validate other
    flags -- and so do not know the TR or the run lengths until later. Those
    call this once the load is done; CLIs that already know the TR up front get
    it for free by passing ``trim=`` to :func:`parse_timing_spec`.

    Returns the report (``None`` when the spec is inactive or TR-less), having
    already printed it when *verbose*.
    """
    if not trim.active or trim.tr is None:
        return None

    n_runs = n_runs if n_runs is not None else len(spec.all_onsets[0]) if spec.all_onsets else 0
    if run_lengths_tr is not None:
        run_lengths_sec = [n * trim.tr for n in run_lengths_tr]
    else:
        # Without run lengths only the start-of-run side can be checked; late
        # events are left for the caller's own late-event guard.
        run_lengths_sec = [float("inf")] * n_runs
    if len(run_lengths_sec) == 1 and n_runs > 1:
        run_lengths_sec = run_lengths_sec * n_runs

    spec.all_onsets, report = shift_onsets_for_trim(
        spec.all_onsets,
        spec.durations,
        spec.condition_labels,
        run_lengths_sec,
        trim,
    )
    if verbose:
        for line in report.lines():
            print(line)
    return report


def parse_hrf_model_args(
    hrf_model_arg: str,
    canonical_arg: str | None,
    durations: list[float],
    condition_labels: list[str],
    tr: float,
    fir_window_s: float | None = None,
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
    fir_window_s : float or None, optional
        Explicit FIR/TENT window length in seconds (``-fir_duration`` /
        ``-tent_duration``).  Overrides the window derived from the stimulus
        durations.  Ignored for canonical models.

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
        window_source = "durations"
        if "bot" in hrf_params and "top" in hrf_params:
            # Explicit window specified (e.g., TENT(0,15,6))
            fir_bot = hrf_params["bot"]
            fir_top = hrf_params["top"]
            window_source = f"{hrf_model_str}"
            if "n_basis" in hrf_params:
                n_basis = hrf_params["n_basis"]
            else:
                # Default: 1 basis per TR
                n_basis = int(np.ceil((fir_top - fir_bot) / tr))
        elif fir_window_s is not None:
            fir_bot = 0.0
            fir_top = float(fir_window_s)
            n_basis = max(1, int(np.ceil(fir_top / tr)))
            window_source = "-fir_duration"
        else:
            # The response to a D-second block does not end at D seconds — the
            # HRF keeps rising for ~5 s and decays for ~20 s after that. Taking
            # the window as max(durations) (the old behaviour) truncated the
            # entire post-stimulus response. Estimate it instead: convolve the
            # canonical HRF with the boxcar and find where it returns to
            # baseline. See design/hrf.py:estimate_hrf_window.
            from fastfuncstuff.design.hrf import estimate_hrf_window

            fir_bot = 0.0
            n_basis = max(1, estimate_hrf_window(max(durations), tr))
            fir_top = n_basis * tr
            window_source = "estimated from durations"

        print(
            f"  HRF model: {hrf_model_name} (window: {fir_bot:.1f}-{fir_top:.1f}s, "
            f"{n_basis} basis functions; window {window_source})"
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


# Minimum fraction of trials that must be predictable from other runs before
# beta-space (single-trial) cross-validation is trustworthy.  Below this, the
# criterion is scoring mostly-undefined predictions.
CV_DESIGN_MIN_PREDICTABLE_FRACTION = 0.5


@dataclass
class TrialRepeatSummary:
    """How repeatable the event design is, from the caller's ``[cond][run]`` onsets.

    Beta-space cross-validation predicts a held-out trial from the *other runs'*
    trials of the same condition.  A trial whose condition never appears in
    another run has no such target, so its score is undefined no matter how good
    the denoising is.  ``predictable_fraction`` is the share of trials that do
    have one.
    """

    n_trials: int
    n_runs: int
    trials_per_condition: list[int]
    runs_per_condition: list[int]
    n_predictable_trials: int

    @property
    def predictable_fraction(self) -> float:
        return self.n_predictable_trials / self.n_trials if self.n_trials else 0.0

    @property
    def n_repeated_conditions(self) -> int:
        return sum(1 for r in self.runs_per_condition if r >= 2)

    @property
    def n_conditions(self) -> int:
        return len(self.trials_per_condition)

    def describe(self) -> str:
        return (
            f"{self.n_trials} trials, {self.n_conditions} conditions, "
            f"{self.n_repeated_conditions} of them appearing in >=2 runs "
            f"({100 * self.predictable_fraction:.0f}% of trials predictable across runs)"
        )


def summarize_trial_repeats(
    onsets_by_condition: list[list[np.ndarray]],
) -> TrialRepeatSummary:
    """Count cross-run repeats in ``all_onsets[condition][run]`` onset lists.

    Parameters
    ----------
    onsets_by_condition : list of list of np.ndarray
        ``[condition][run] -> onset times``, i.e. :class:`TimingSpec.all_onsets`.
    """
    trials_per_condition: list[int] = []
    runs_per_condition: list[int] = []
    n_predictable = 0

    n_runs = max((len(per_run) for per_run in onsets_by_condition), default=0)

    for per_run in onsets_by_condition:
        counts = [len(np.atleast_1d(np.asarray(onsets))) for onsets in per_run]
        n_cond_trials = int(sum(counts))
        n_cond_runs = int(sum(1 for c in counts if c > 0))
        trials_per_condition.append(n_cond_trials)
        runs_per_condition.append(n_cond_runs)
        if n_cond_runs >= 2:
            n_predictable += n_cond_trials

    return TrialRepeatSummary(
        n_trials=int(sum(trials_per_condition)),
        n_runs=n_runs,
        trials_per_condition=trials_per_condition,
        runs_per_condition=runs_per_condition,
        n_predictable_trials=n_predictable,
    )


def add_cv_blur_arg(group, *, stage_note: str = "") -> None:
    """Add ``-cv_blur``: blur applied only to the parameter-selection stage.

    Distinct from ``-do_blur``, which blurs the whole pipeline including the
    final fit and the saved betas.
    """
    group.add_argument(
        "-cv_blur",
        "-cv-blur",
        dest="cv_blur",
        type=float,
        default=None,
        metavar="FWHM",
        help="Gaussian FWHM in mm applied ONLY to the parameter-selection stage; "
        "the final model is fit on the unblurred data. In thermal-noise-dominated "
        "data the structure the search needs is buried, so noise components come "
        "out mixed with thermal junk and HRF fits are unstable — a few mm of blur "
        "lets the search see the structure without smoothing the output. "
        "Everything the selection touches is blurred consistently (noise pool, "
        "component extraction, and scoring). Contrast -do_blur, which blurs the "
        "whole pipeline including the saved betas. " + stage_note,
    )


def blur_masked_data(
    data: torch.Tensor,
    *,
    fwhm_mm: float,
    volume_shape: tuple,
    voxel_sizes: tuple,
    mask_flat: np.ndarray | torch.Tensor | None,
    run_starts: list[int],
    device: torch.device | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Blur a masked ``(n_voxels, n_timepoints)`` timeseries back through its volume.

    The data has already been reduced to in-mask voxels, so blurring means
    scattering to the volume, convolving, and gathering back. The convolution is
    **normalized** (divided by the identically-blurred mask), because
    ``gaussian_blur_3d`` zero-pads: without it, every voxel near the mask edge or
    the FOV boundary gets pulled toward zero, which looks exactly like signal
    dropout to whatever criterion is reading the result.

    Returns a new tensor on the same device and dtype as ``data``; the input is
    untouched, since the caller still needs it for the final fit.
    """
    from fastfuncstuff.utils import gaussian_blur_3d

    n_voxels, n_timepoints = data.shape
    n_vol_voxels = int(np.prod(volume_shape))

    if mask_flat is None:
        if n_voxels != n_vol_voxels:
            raise ValueError(
                f"cannot blur: {n_voxels} data voxels do not fill the "
                f"{volume_shape} volume and no mask was supplied to place them"
            )
        mask_idx = None
    else:
        mask_np = mask_flat.cpu().numpy() if torch.is_tensor(mask_flat) else np.asarray(mask_flat)
        mask_np = mask_np.astype(bool).reshape(-1)
        if int(mask_np.sum()) != n_voxels:
            raise ValueError(
                f"cannot blur: mask selects {int(mask_np.sum())} voxels but data has {n_voxels}"
            )
        mask_idx = np.flatnonzero(mask_np)

    if verbose:
        print(f"  Blurring selection-stage data (FWHM = {fwhm_mm} mm)...")

    # Normalizer: the mask itself, blurred with the same kernel.  Computed once.
    weight_vol = np.zeros(n_vol_voxels, dtype=np.float32)
    if mask_idx is None:
        weight_vol[:] = 1.0
    else:
        weight_vol[mask_idx] = 1.0
    weight = gaussian_blur_3d(
        weight_vol.reshape(*volume_shape, 1),
        fwhm_mm=fwhm_mm,
        voxel_sizes=voxel_sizes,
        device=device,
        # Silent: this is the one-volume mask normalizer, and its "Blurred 1
        # volumes" progress bar reads as if the whole dataset were one volume
        # (the per-run data blurs below are the real work, and are quiet).
        verbose=False,
    ).reshape(n_vol_voxels)
    # Below this the normalizer is all edge and the quotient is noise amplification.
    weight = np.where(weight > 1e-3, weight, np.inf)

    out = torch.empty_like(data)
    run_bounds = list(run_starts) + [n_timepoints]
    for run_idx in range(len(run_starts)):
        start_tp, end_tp = run_bounds[run_idx], run_bounds[run_idx + 1]
        n_tps = end_tp - start_tp

        vol = np.zeros((n_vol_voxels, n_tps), dtype=np.float32)
        run_data = data[:, start_tp:end_tp].detach().cpu().numpy().astype(np.float32, copy=False)
        if mask_idx is None:
            vol[:] = run_data
        else:
            vol[mask_idx] = run_data

        blurred = gaussian_blur_3d(
            vol.reshape(*volume_shape, n_tps),
            fwhm_mm=fwhm_mm,
            voxel_sizes=voxel_sizes,
            device=device,
            verbose=False,
        ).reshape(n_vol_voxels, n_tps)
        blurred /= weight[:, None]

        gathered = blurred if mask_idx is None else blurred[mask_idx]
        out[:, start_tp:end_tp] = torch.from_numpy(gathered).to(
            device=data.device, dtype=data.dtype
        )
        del vol, blurred

    return out


def add_single_trial_args(group, *, emit_help: str) -> None:
    """Add the harmonized ``-single_trials`` / ``-cv_design`` pair.

    The two flags are orthogonal on purpose: ``-single_trials`` controls the
    *output* (per-trial betas), ``-cv_design`` controls which design the tool's
    hyperparameter search is scored against.  Keeping them separate is what lets
    a design with no repeated conditions still produce single-trial betas — the
    parameter is learned where the data has leverage, then applied per trial.
    """
    group.add_argument(
        "-single_trials",
        "-single-trials",
        dest="single_trials",
        action="store_true",
        help=emit_help,
    )
    group.add_argument(
        "-cv_design",
        "-cv-design",
        dest="cv_design",
        choices=["auto", "condition", "single"],
        default="auto",
        help="Design used to score the cross-validated parameter search. "
        "'single' = beta-space CV (held-out trial betas vs. same-condition "
        "training-run betas; needs repeated conditions across runs). "
        "'condition' = timeseries CV on the condition-level design (works with "
        "no repeats at all). 'auto' (default) = 'single' when -single_trials is "
        "set and the events support it, otherwise 'condition'. Selection and "
        "output are independent: -single_trials -cv_design condition learns the "
        "parameter from condition structure and still writes per-trial betas.",
    )


def add_cv_strategy_arg(group, *, dest: str = "cv_strategy", default: str = "loro") -> None:
    """The one spelling of ``-cv_strategy``, so four tools cannot drift apart.

    Every tool accepted this flag already but with slightly different help and
    only some with the dashed alias, which is precisely the sort of drift that
    makes the CV surface feel like five tools instead of one.
    """
    group.add_argument(
        "-cv_strategy",
        "-cv-strategy",
        dest=dest,
        default=default,
        help="Cross-validation strategy: 'loro' or '1' for leave-one-run-out "
        "(default), '0.5' for split-halves, any float in (0,1) for that training "
        "fraction, any int > 1 for leave-N-out.",
    )


def add_cv_metric_arg(
    group,
    *,
    dest: str = "cv_metric",
    default: str = "cod",
    choices: Sequence[str] = ("cod", "corr", "corr2", "sse"),
) -> None:
    """``-cv_metric`` with ``-metric`` kept as an alias.

    ffs_denoise called it ``-cv_metric`` while ffs_ridge and ffs_hrfopt called
    it ``-metric``. Both spellings now work everywhere; ``-cv_metric`` is the
    one documented, since a bare ``-metric`` reads like it might control the
    reported statistic rather than the cross-validation's scoring rule.

    ``dest`` stays whatever each tool already used: renaming it would churn
    every internal reference for no user-visible gain.
    """
    group.add_argument(
        "-cv_metric",
        "-cv-metric",
        "-metric",
        dest=dest,
        choices=list(choices),
        default=default,
        help="Scoring rule for the cross-validated search: 'cod' (coefficient of "
        "determination), 'corr' (Pearson r), 'corr2' (r squared), 'sse' (sum of "
        "squared errors, GLMsingle-compatible, lower is better). Only 'cod' is on "
        "the variance-fraction scale a noise ceiling uses, so -noise_ceiling "
        "writes an explainable-R2 map only under 'cod'.",
    )


def add_noise_ceiling_args(group, *, stage_note: str = "") -> None:
    """Add ``-noise_ceiling``: the R2 ceiling and the explainable-R2 map.

    Deliberately one flag rather than a family. The estimator is not really the
    user's choice -- it is dictated by which space the cross-validation scored
    in, and picking the wrong one produces a ratio of two incommensurate
    numbers -- so ``auto`` follows ``-cv_design`` and the explicit values exist
    only to force a cross-check.
    """
    group.add_argument(
        "-noise_ceiling",
        "-noise-ceiling",
        dest="noise_ceiling",
        nargs="?",
        const="auto",
        default="off",
        choices=["off", "auto", "loro", "df", "ncsnr", "repeat"],
        help="Estimate the per-voxel ceiling on the cross-validated R2 and save it alongside an"
        " explainable-R2 map: xval_r2 / ceiling, the fraction of the ACHIEVABLE variance the"
        " model captured. NaN where the ceiling is too near zero for the fraction to mean"
        " anything.\n"
        "  auto    the value used when the flag is given bare. Follows -cv_design: 'ncsnr' for"
        " beta-space CV, 'loro' for condition-level timeseries CV, falling back to 'df' when"
        " there are too few runs to split. Never picks 'repeat'.\n"
        "  loro    split each fold's training runs in two and take the covariance of their two"
        " predictions of the held-out run. Same units, same timepoints as the R2 it bounds."
        " Wants 4+ runs.\n"
        "  df      correct the in-sample fit for degrees of freedom. Needs no repeats of any"
        " kind, and is the fallback when nothing else applies.\n"
        "  ncsnr   the NSD/GLMsingle beta-space ceiling. Needs CONDITIONS that repeat across"
        " runs, not repeated runs.\n"
        "  repeat  for whole RUNS that repeat -- identical stimulus timing, as in a block design"
        " or localiser replayed verbatim. Uses the run-to-run correlation directly.\n"
        "The first four bound what THIS DESIGN can predict, not what any model could, so a design"
        " missing a real condition gets a low ceiling and a flattering explainable R2. 'repeat'"
        " alone never fits the design and so bounds any model: far below 1 under 'repeat' but"
        " near 1 under 'loro' means the design is the limit, not the noise.\n" + stage_note,
    )


def resolve_cv_design(
    requested: str,
    single_trials: bool,
    repeats: TrialRepeatSummary,
    *,
    parameter: str,
    manual_hint: str,
    single_needs_repeats: bool = True,
    verbose: bool = True,
) -> str:
    """Turn ``-cv_design {auto,condition,single}`` into 'condition' or 'single'.

    Both selection designs are cross-validated across runs *except* single-trial
    HRF selection, which is in-sample (all HRF candidates carry one beta per
    trial, so model complexity is equal and in-sample R² is a fair comparison).
    That is what ``single_needs_repeats`` encodes: with no cross-run condition
    overlap, condition-level CV is always undefined, but in-sample single-trial
    selection still works.

    Parameters
    ----------
    parameter : str
        What is being chosen ("PC count", "HRF"), for the printed rationale.
    manual_hint : str
        Tool-specific advice printed when no selection design is usable.
    single_needs_repeats : bool
        False when the tool's single-trial selection is in-sample rather than
        cross-validated.

    Raises
    ------
    SystemExit
        If the requested design cannot be scored on this event structure, or if
        beta-space selection was requested without -single_trials.
    """
    import sys

    if requested == "single" and not single_trials:
        print("ERROR: -cv_design single requires -single_trials.")
        print("  Beta-space selection scores single-trial betas; without")
        print("  -single_trials there are none. Use -cv_design condition instead.")
        sys.exit(1)

    frac = repeats.predictable_fraction
    has_repeats = repeats.n_predictable_trials > 0
    # Condition-level CV predicts held-out runs from the others, so it needs the
    # same cross-run condition overlap that beta-space CV needs.  A design where
    # every condition is confined to one run cannot be cross-validated at all.
    viable_condition = has_repeats
    viable_single = single_trials and (has_repeats or not single_needs_repeats)

    if verbose:
        print()
        print(f"Event repeat structure: {repeats.describe()}")

    def _no_cross_run_structure() -> None:
        print()
        print(f"ERROR: no condition appears in more than one run, so {parameter}")
        print("  selection cannot be cross-validated: a held-out run shares no")
        print("  condition with the runs it would be predicted from.")
        print(f"  {repeats.describe()}")
        print(f"  {manual_hint}")
        sys.exit(1)

    if requested == "condition":
        if not viable_condition:
            _no_cross_run_structure()
        resolved = "condition"
    elif requested == "single":
        if not viable_single:
            print()
            print("ERROR: -cv_design single needs conditions that repeat across runs.")
            print(f"  {repeats.describe()}")
            print("  Every trial's condition is confined to a single run, so a held-out")
            print("  trial has no same-condition training trial to be scored against.")
            if viable_condition:
                print("  Use -cv_design condition; -single_trials still writes per-trial betas.")
            else:
                print(f"  {manual_hint}")
            sys.exit(1)
        if single_needs_repeats and frac < CV_DESIGN_MIN_PREDICTABLE_FRACTION:
            print(
                f"  WARNING: only {100 * frac:.0f}% of trials are predictable across runs; "
                f"beta-space {parameter} selection rests on a thin subset."
            )
        resolved = "single"
    else:  # auto
        if viable_single and (frac >= CV_DESIGN_MIN_PREDICTABLE_FRACTION or not viable_condition):
            resolved = "single"
        elif viable_condition:
            resolved = "condition"
        elif viable_single:
            resolved = "single"
        else:
            _no_cross_run_structure()
            raise AssertionError("unreachable")  # _no_cross_run_structure exits

    if verbose:
        if resolved == "single":
            how = (
                "beta-space CV on single-trial betas"
                if single_needs_repeats
                else "in-sample R² on single-trial betas"
            )
            why = ""
            if requested == "auto" and not viable_condition:
                why = " — the only design with a usable criterion here"
            print(f"  {parameter} selection: {how} (-cv_design single){why}")
        else:
            why = ""
            if requested == "auto" and single_trials:
                why = (
                    f" — auto fell back: {100 * frac:.0f}% of trials predictable "
                    f"across runs, below {100 * CV_DESIGN_MIN_PREDICTABLE_FRACTION:.0f}%"
                )
            print(f"  {parameter} selection: timeseries CV on the condition-level design{why}")
            if single_trials:
                print("  Single-trial betas are still estimated and saved.")

    return resolved


def build_task_design_from_args(
    hrf_model_name: str,
    is_fir_model: bool,
    fir_bot: float | None,
    fir_top: float | None,
    n_basis: int | None,
    all_onsets: list,
    stim_durations: list[float],
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
        build_task_design,
        make_fir_design,
        make_tent_design,
        onsets_to_tr_matrix,
    )

    def _event_design(hrf_bases: torch.Tensor) -> torch.Tensor:
        return build_task_design(
            hrf_bases,
            n_timepoints,
            run_starts,
            tr=tr,
            microtime_dt=microtime_dt,
            event_onsets=all_onsets,
            durations=stim_durations,
            device=device,
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
            design_for_hrf = _event_design(hrf_kernel)
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
                onset_matrix_tr, max_shift = onsets_to_tr_matrix(
                    all_onsets=all_onsets,
                    run_starts=run_starts,
                    n_timepoints=n_timepoints,
                    tr=tr,
                    durations=stim_durations,
                    device=device,
                )
                if max_shift > 1e-6:
                    print(
                        f"  FIR lags are whole TRs: onsets rounded to the TR grid "
                        f"(largest shift {max_shift:.3f}s of a {tr:.3f}s TR). "
                        f"Use TENT/TENTZERO to keep sub-TR onset timing."
                    )
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

                task_design = _event_design(hrf_set)
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
                task_design = _event_design(hrf)
            elif hrf_model_name == "GLMSINGLE":
                print("  Using canonical GLMsingle HRF")
                hrf = get_hrf_library(
                    mode="glmsingle",
                    microtime_dt=microtime_dt,
                    stim_duration=0.0,
                    hrf_duration=32.0,
                    device=device,
                )
                task_design = _event_design(hrf)
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
    from fastfuncstuff.io.afni import nifti_shape

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
                # nifti_shape, not load_nifti(...).shape -- the comment above
                # already claimed "header-only", but load_nifti decompressed the
                # entire payload of every run just to reach dim 4.
                shape = nifti_shape(nii_file)
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


# ---------------------------------------------------------------------------
# Shared batch runner
#
# ffs_moco, ffs_nwarp, ffs_allineate and ffs_formwarp all process many datasets
# per session, so one process per dataset pays the Python/CUDA/torch.compile
# startup for every run.
# A batch mode amortizes those fixed costs across many self-contained runs in a
# single process. The collection and loop mechanics are identical between tools;
# only the parse/validate/dispatch/expected-outputs callbacks differ, so they
# live here and each CLI supplies the callbacks.
# ---------------------------------------------------------------------------


def add_batch_args(
    group,
    *,
    tool: str,
    what: str,
    example: str,
    skip_note: str,
) -> None:
    """Register the canonical ``-batch`` / ``-batch_run`` / ``-batch_skip`` trio.

    Every batchable tool takes the same three flags with the same semantics; only
    the noun and the example differ, so the help text is generated here rather
    than re-typed per CLI.

    Parameters
    ----------
    group : argparse parser or argument group
        Where the flags land (normally the tool's Input/Output group).
    tool : str
        The command name, e.g. ``"ffs_allineate"``.
    what : str
        Plural noun for one unit of work, e.g. ``"affine alignments"``.
    example : str
        A representative argument string for one run (no leading tool name).
    skip_note : str
        Which flags ``-batch_skip`` derives a job's outputs from, e.g.
        ``"-prefix / -1Dmatrix_save"``.
    """
    group.add_argument(
        "-batch",
        default=None,
        metavar="FILE",
        help=f"Run many self-contained {what} in one process, amortizing "
        "Python/CUDA startup and torch.compile warmup. FILE is a manifest with "
        "one run per line; each line is exactly the arguments you would pass "
        f"after `{tool}` for that run (e.g. `{example}`). Blank lines and lines "
        "starting with # are ignored. The device is chosen once for the whole "
        "batch; a per-line -device is ignored. One failing run is reported and "
        "skipped without sinking the rest; the batch exits nonzero if any run "
        "failed.",
    )
    group.add_argument(
        "-batch_run",
        "-batch-run",
        dest="batch_run",
        action="append",
        metavar="ARGS",
        help="Inline alternative to -batch for self-contained scripts: one run "
        f'given as a single quoted argument string (e.g. -batch_run "{example}"), '
        f"exactly as you would type it after `{tool}`. Repeatable — pass it once "
        "per run. Same semantics as -batch (device chosen once, failures "
        "isolated); may be combined with -batch, in which case manifest-file "
        "runs come first.",
    )
    group.add_argument(
        "-batch_skip",
        "-batch-skip",
        dest="batch_skip",
        action="store_true",
        help="In a -batch / -batch_run run, skip any job whose requested outputs "
        f"all already exist on disk (checked from that job's {skip_note}). Lets "
        "you re-run a manifest and only pay for the jobs that still need work. A "
        "job missing any one of its outputs is run in full.",
    )


def collect_batch_jobs(
    batch_file: str | None,
    batch_run: list[str] | None,
) -> list[tuple[str, str]]:
    """Gather ``(label, argv-string)`` runs from a ``-batch`` manifest and/or
    repeated ``-batch_run`` inline arguments.

    Each argv-string is exactly the arguments one would pass after the tool for
    that run. Manifest-file runs come first (so labels stay stable regardless of
    which source is used), then inline runs. Blank lines and ``#`` comments in
    the manifest are ignored. Exits(1) if a requested source yields nothing.
    """
    jobs: list[tuple[str, str]] = []

    if batch_file is not None:
        manifest = Path(batch_file)
        if not manifest.is_file():
            print(f"Error: -batch file not found: {batch_file}", file=sys.stderr)
            sys.exit(1)
        n_before = len(jobs)
        for lineno, raw in enumerate(manifest.read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            jobs.append((f"line {lineno}", line))
        if len(jobs) == n_before:
            print(f"Error: -batch file has no runs: {batch_file}", file=sys.stderr)
            sys.exit(1)

    for i, raw in enumerate(batch_run or [], 1):
        line = raw.strip()
        if not line:
            print(f"Error: -batch_run #{i} is empty.", file=sys.stderr)
            sys.exit(1)
        jobs.append((f"run {i}", line))

    return jobs


# Outer settings that describe the BATCH rather than a run, so a run must not
# inherit them: -batch/-batch_run would trip the no-nesting guard on every line,
# and -batch_skip/-device are decided once for the whole batch by the runner.
_BATCH_ONLY_ARGS = frozenset({"batch", "batch_run", "batch_skip", "device"})


def _run_defaults(defaults) -> argparse.Namespace | None:
    """A fresh copy of the outer args to seed one run's parse with.

    Fresh per run because argparse writes into the namespace it is given, so a
    shared one would leak the first run's settings into the second.
    """
    if defaults is None:
        return None
    return argparse.Namespace(
        **{k: v for k, v in vars(defaults).items() if k not in _BATCH_ONLY_ARGS}
    )


def run_batch_jobs(
    *,
    tool: str,
    jobs: list[tuple[str, str]],
    device: torch.device,
    parse_line,
    dispatch,
    validate=None,
    is_nested=None,
    expected_outputs=None,
    skip_existing: bool = False,
    verb: int = 1,
    defaults=None,
) -> None:
    """Run every collected batch job in this one process, isolating failures.

    The wins over a shell loop are all fixed costs: the Python interpreter, the
    torch/CUDA import, the CUDA context, and — the big one — torch.compile's
    kernel warmup are paid once and reused across runs. The per-file work is
    unchanged. Runs stay isolated: one failure is reported and skipped, and the
    batch exits nonzero if any run failed.

    Parameters
    ----------
    tool : str
        Tool name for messages (e.g. ``"ffs_moco"``).
    jobs : list of (label, argv-string)
        From :func:`collect_batch_jobs`.
    device : torch.device
        Chosen once for the whole batch; reused by every run.
    parse_line : callable(str) -> Namespace
        Parse one argv-string into args (typically ``lambda s: parse_args(shlex.split(s))``).
    dispatch : callable(Namespace, torch.device) -> None
        Execute one run.
    validate : callable(Namespace) -> None, optional
        Extra per-run validation; raise/``SystemExit`` on a bad request.
    is_nested : callable(Namespace) -> bool, optional
        True when a run line itself re-requests ``-batch`` (rejected to avoid
        nesting).
    expected_outputs : callable(Namespace) -> list[str], optional
        Concrete output paths the run would write; required for ``skip_existing``.
    skip_existing : bool
        When True, a run whose ``expected_outputs`` all already exist is skipped.
    defaults : Namespace, optional
        The outer command's own args. Every setting on it becomes the DEFAULT for
        each run, which the run's own line overrides flag by flag.

        Bug of record: without this, a flag outside -batch was silently dropped.
        `ffs_optiwarp -force lk -batch runs.txt` ran demons -- the default -- on
        every pair, and nothing said so; the same held for every batchable tool
        and every flag. Shared settings are exactly what one wants to state once,
        and a manifest is already per-run, so "outer = default, line = override"
        is the only reading that does not make a whole run wrong in silence.
    """
    import os

    from tqdm import tqdm

    if verb >= 1:
        print(f"{tool} batch: {len(jobs)} runs on device={device}")

    failures: list[tuple[str, str]] = []
    skipped = 0
    t0 = time.time()
    desc = f"{tool.removeprefix('ffs_')} batch"
    bar = tqdm(jobs, desc=desc, unit="run", leave=True, disable=len(jobs) == 1)
    for label, line in bar:
        try:
            run_args = parse_line(line, _run_defaults(defaults))
            if is_nested is not None and is_nested(run_args):
                raise ValueError("-batch/-batch_run cannot be nested inside a run")
            if validate is not None:
                validate(run_args)
            if skip_existing and expected_outputs is not None:
                outs = expected_outputs(run_args)
                if outs and all(os.path.exists(p) for p in outs):
                    skipped += 1
                    if verb >= 1:
                        tqdm.write(f"[{label}] skip: all {len(outs)} output(s) exist")
                    continue
            dispatch(run_args, device)
        except KeyboardInterrupt:
            raise
        except (Exception, SystemExit) as exc:
            # A bad run must not sink the batch. The underlying error has usually
            # already gone to stderr; record the run so the summary is actionable.
            reason = str(exc) if str(exc) not in ("", "1") else exc.__class__.__name__
            failures.append((label, reason))
            tqdm.write(f"[{label}] FAILED ({reason}): {line}")
        finally:
            # Don't let this run's peak strand VRAM for the next one — the memory
            # module sizes chunks off *free* VRAM, so stale caches shrink them.
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if verb >= 1:
        ok = len(jobs) - len(failures) - skipped
        tail = f", {skipped} skipped" if skipped else ""
        print(f"{tool} batch done: {ok}/{len(jobs)} succeeded{tail} in {time.time() - t0:.2f}s")
    if failures:
        print(f"Error: {len(failures)} batch run(s) failed:", file=sys.stderr)
        for label, reason in failures:
            print(f"  {label}: {reason}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Data coverage / brain masking, shared by the registration CLIs
# ---------------------------------------------------------------------------


def add_coverage_args(parser, automask_help: str | None = None) -> None:
    """Register the shared ``-automask*`` / data-coverage flags on a registration CLI.

    ffs_formwarp, ffs_optiwarp and ffs_qwarp all face the same problem -- a source
    resampled onto the base grid carries an empty wedge where its FoV was clipped, and
    a metric that can see that wedge's edge will warp real tissue into it -- so they
    share one set of flags and one meaning for each.
    """
    g = parser.add_argument_group("Masking / data coverage")
    g.add_argument(
        "-automask",
        action="store_true",
        help=automask_help
        or "Restrict the metric to the automask of BOTH images (unioned; see "
        "-automask_intersect). Independent of the data-coverage restriction below.",
    )
    g.add_argument(
        "-automask_base", "-automask-base", action="store_true", help="Automask the base only."
    )
    g.add_argument(
        "-automask_source",
        "-automask-source",
        action="store_true",
        help="Automask the source only.",
    )
    g.add_argument(
        "-automask_intersect",
        "-automask-intersect",
        action="store_true",
        help="Intersect the two automasks instead of unioning them. Stricter, but one "
        "image's automask failure (common where a FoV fades in over several slices) "
        "then vetoes the other image's good data.",
    )
    g.add_argument(
        "-nocoverage",
        "-no_coverage",
        "-no-coverage",
        action="store_true",
        help="Do not treat non-finite/zero voxels as missing data. By default each "
        "image's empty region is excluded from the metric AND filled from the other "
        "image, so a clipped FoV presents no artificial edge for the warp to chase. "
        "Use this when a zero background is a REAL edge you want registered -- a "
        "skull-stripped anatomical, say -- rather than data loss.",
    )
    g.add_argument(
        "-void_guard",
        "-void-guard",
        type=float,
        default=1.0,
        help="Strength (0..1) of the no-data-boundary guard: near a coverage boundary, "
        "remove this fraction of the update that points along the boundary's normal "
        "(i.e. into the void), leaving both tangential directions free. This is the "
        "general form of -noZdis for a void that happens to lie below the brain: it "
        "blocks only the direction that reaches into missing data, so an in-plane "
        "shift along that same edge still happens. 0 disables.",
    )
    g.add_argument(
        "-coverage_erode",
        "-coverage-erode",
        type=int,
        default=1,
        help="Voxels to peel off the data-coverage boundary (default 1), removing the "
        "partial-value ramp resampling leaves at a clipped FoV edge. 0 keeps the raw "
        "nonzero region. The peel also trims the volume's own outer faces, so keep it "
        "small.",
    )


def sanitize_volume(vol: torch.Tensor, label: str, verb: int) -> torch.Tensor:
    """Replace non-finite voxels with 0, loudly.

    Must happen before anything else touches the volume. ``automask``'s clip level
    comes out NaN on a NaN-bearing image and it returns an *empty* mask; ``!= 0`` is
    True for NaN so a coverage test passes the whole void through as valid data; and
    NaN survives every smoothing kernel, so one bad slab makes a whole metric NaN. A
    NaN rim is common in the wild -- anything that divides by the data turns the
    exact-zero background into NaN, leaving a volume with an empty slab and not one
    zero in it -- and every one of those failures is silent.
    """
    n_bad = int((~torch.isfinite(vol)).sum().item())
    if n_bad:
        if verb >= 1:
            print(
                f"WARNING: {label} has {n_bad:,} non-finite voxels "
                f"({100.0 * n_bad / vol.numel():.1f}%) -- treating as no-data (0)"
            )
        vol = torch.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    return vol


def image_support(
    vol: torch.Tensor,
    args,
    device: torch.device,
    want_automask: bool,
    label: str,
    verb: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """One image's ``(brain, cover)`` masks: where is the object, where is the data.

    Two different questions with two different jobs. ``cover`` (finite and nonzero) is
    both excluded from the metric and cross-filled, because a missing-data boundary is
    an artificial cliff that must not exist as far as the metric is concerned.
    ``brain`` only ever damps the metric -- an automask boundary is real anatomy, and
    filling across it would splice one image's skull into the other's.
    """
    from fastfuncstuff.processing.mask import automask, data_coverage_mask

    v = vol.float().to(device)
    brain = None
    if want_automask:
        brain = automask(v, device=device)
        if verb >= 1:
            print(f"Automask ({label}): {100.0 * brain.float().mean().item():.1f}% of voxels")
    cover = None
    if not args.nocoverage:
        cover = data_coverage_mask(v, erode=args.coverage_erode, device=device)
        if verb >= 1:
            frac = 100.0 * cover.float().mean().item()
            if frac < 99.95:
                print(f"Coverage ({label}): {frac:.1f}% of voxels hold data")
    return brain, cover


def combine_brain_masks(
    base_brain: torch.Tensor | None,
    src_brain: torch.Tensor | None,
    intersect: bool = False,
) -> torch.Tensor | None:
    """Union (default) the two brain masks, or intersect them if asked.

    Union is the default because automask fails asymmetrically: on a partially covered
    slice (a FoV that fades in over several slices rather than cutting cleanly) the
    clip-level test and the peel erosion reject real tissue, and intersecting lets one
    image's failure veto the other's perfectly good data. Measured on a 9.4T pair whose
    source coverage ramps in over ~10 slices, intersecting deleted 40% of the evaluable
    voxels in that band and bought nothing (edge-band correlation 0.9067 intersected vs
    0.9094 unioned). Data coverage is the constraint to trust; a brain mask is a soft
    preference, so let either image vouch for a voxel.
    """
    if base_brain is None or src_brain is None:
        return base_brain if src_brain is None else src_brain
    return base_brain & src_brain if intersect else base_brain | src_brain


# ---------------------------------------------------------------------------
# Tuned presets (-type)
# ---------------------------------------------------------------------------


def add_recipe_arg(parser, backend: str) -> None:
    """Register ``-type RECIPE``, which applies settings a tuning run measured.

    The registration engines have a lot of knobs and no single right setting for
    all data, so the honest default is conservative and the *good* setting is
    whatever ``ffs_tunewarp`` found for data of that kind. This is how that finding
    reaches a user who is not going to run a search themselves.
    """
    from fastfuncstuff.processing.tunespec import RECIPES, describe_presets

    parser.add_argument(
        "-type",
        dest="recipe",
        choices=sorted(RECIPES),
        default=None,
        help="Apply the settings ffs_tunewarp measured for this kind of "
        "registration. Any flag you pass explicitly still wins, so -type sets a "
        "starting point rather than overriding you. Presets available here:\n"
        + describe_presets(backend),
    )


def apply_recipe_preset(args, backend: str, argv: list[str] | None = None, verb: int = 1) -> None:
    """Push a ``-type`` preset onto ``args``, without overriding explicit flags.

    Explicit-wins is the whole contract, and it cannot be decided by comparing a
    value against its default: a user who deliberately passes the default value
    would be silently overridden by the preset. So it is decided by whether the
    flag appears in argv, which is the only thing that actually distinguishes
    "asked for" from "not mentioned".
    """
    import sys

    from fastfuncstuff.processing.tunespec import preset_config_for_cli, preset_for

    recipe = getattr(args, "recipe", None)
    if not recipe:
        return
    preset = preset_for(recipe, backend)
    if preset is None:
        if verb >= 1:
            print(
                f"-type {recipe}: no tuned preset exists for {backend} yet; "
                "using the built-in defaults.",
                file=sys.stderr,
            )
        return

    typed = {tok.lstrip("-").replace("-", "_") for tok in (sys.argv[1:] if argv is None else argv)}
    applied, skipped = {}, []
    for dest, value in preset_config_for_cli(recipe, backend).items():
        if dest in typed:
            skipped.append(dest)
            continue
        setattr(args, dest, value)
        applied[dest] = value

    if verb >= 1 and applied:
        shown = " ".join(f"-{k} {v}" for k, v in sorted(applied.items()))
        print(f"-type {recipe}: applied {shown}", file=sys.stderr)
        if skipped:
            print(
                f"  (kept your explicit {', '.join('-' + s for s in sorted(skipped))})",
                file=sys.stderr,
            )
        print(f"  measured on: {preset.provenance}", file=sys.stderr)
        if preset.caveat:
            print(f"  caveat: {preset.caveat}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Reproducibility (-deterministic)
# ---------------------------------------------------------------------------

CUBLAS_DETERMINISM_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_DETERMINISM_VALUE = ":4096:8"


def add_deterministic_arg(parser) -> None:
    """Register ``-deterministic``: same input, same output, bit for bit."""
    parser.add_argument(
        "-deterministic",
        action="store_true",
        # NB: literal percent signs must be doubled -- argparse runs help text
        # through %-formatting, so a bare "40%" is read as a conversion and blows up
        # -help for the whole tool.
        help="Produce bit-identical output for identical input. OFF BY DEFAULT: "
        "repeated runs of the same command differ by ~0.4 voxels of displacement on "
        "average (p99 ~2.9), so a warp is not exactly reproducible, though "
        "similarity scores move only ~0.001 and rankings are unaffected. Speed is "
        "the priority here and determinism costs roughly 40%% of the runtime. This "
        "is not a random seed -- there is no RNG in the solver; the variation comes "
        "from cuBLAS picking GEMM strategies run to run.",
    )


def enable_determinism(verb: int = 1) -> None:
    """Make CUDA results reproducible, re-executing once if that is what it takes.

    Two things are required and only one of them can be set from inside a running
    process. ``torch.use_deterministic_algorithms`` can be switched on at any point,
    but cuBLAS reads ``CUBLAS_WORKSPACE_CONFIG`` when it initialises, so setting it
    after import does nothing. Measured on a 193^3 qwarp fit: the environment
    variable alone leaves runs differing by 9.6 voxels, the flag alone leaves them
    differing too (and warns), and only both together give bit-identical output.

    So if the variable is missing this re-executes the same command with it set. The
    alternative -- carrying on and reporting success -- would claim a reproducibility
    the run does not have, which is the failure mode this whole option exists to
    remove.
    """
    import os
    import sys

    import torch

    if os.environ.get(CUBLAS_DETERMINISM_ENV) != CUBLAS_DETERMINISM_VALUE:
        os.environ[CUBLAS_DETERMINISM_ENV] = CUBLAS_DETERMINISM_VALUE
        if verb >= 1:
            print(
                f"-deterministic: re-executing with {CUBLAS_DETERMINISM_ENV}="
                f"{CUBLAS_DETERMINISM_VALUE} (cuBLAS reads it at start-up, so it "
                "cannot be set from here).",
                file=sys.stderr,
                flush=True,
            )
        os.execv(sys.executable, [sys.executable, *sys.argv])

    torch.use_deterministic_algorithms(True, warn_only=True)
    if verb >= 1:
        print(
            "-deterministic: on. Expect roughly 40% more runtime.",
            file=sys.stderr,
            flush=True,
        )
