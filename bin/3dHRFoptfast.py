#!/usr/bin/env python3
"""
3dHRFoptfast - Fast cross-validated HRF optimization per voxel using GPU acceleration

This tool selects the optimal HRF for each voxel using cross-validation across runs,
then fits a final GLM with the voxel-wise optimal HRFs.

Two HRF library modes:
- library: Double-gamma HRFs with varying time-to-peak and undershoot parameters
- pighs: PIGHS (Parametric Individually Generated HRFs) half-cosine basis with Latin Hypercube sampling

The CV-based HRF selection prevents overfitting that occurs with in-sample selection.

Basic usage:
    3dHRFoptfast -input run1.nii.gz run2.nii.gz run3.nii.gz \\
                 -onsets cond1.txt cond2.txt \\
                 -durations 2.0 5.0 \\
                 -tr 2.0 \\
                 -prefix subject01

For help:
    3dHRFoptfast -help
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

# Import fastfuncsim modules
try:
    from fastfuncsim.afni_io import (
        load_afni_mask,
        load_and_concatenate_runs,
    )
    from fastfuncsim.design_builder import (
        parse_afni_timing_file,
    )
    from fastfuncsim.hrf import get_hrf_library
    from fastfuncsim.hrf_selection import (
        fit_glm_hrf_library_with_xval,
        save_hrf_selection_results,
    )
    from fastfuncsim.utils import get_device
except ImportError as e:
    print(f"ERROR: Could not import fastfuncsim: {e}")
    print("Make sure fastfuncsim is installed: pip install -e .")
    sys.exit(1)


def create_onset_matrix_microtime(
    all_onsets: List[List[np.ndarray]],
    run_starts: List[int],
    tr: float,
    n_timepoints: int,
    microtime_resolution: int,
    stim_durations: List[float],
    device: torch.device,
) -> torch.Tensor:
    """
    Create binary onset matrix at microtime resolution.

    Parameters
    ----------
    all_onsets : list of list of np.ndarray
        Onsets organized as [condition][run] -> np.ndarray of onset times
    run_starts : list of int
        Starting timepoint for each run
    tr : float
        Repetition time in seconds
    n_timepoints : int
        Total number of TR timepoints
    microtime_resolution : int
        Sub-TR resolution (e.g., 16 = 16 bins per TR)
    stim_durations : list of float
        Duration in seconds for each condition
    device : torch.device
        Device for output tensor

    Returns
    -------
    onset_matrix : torch.Tensor
        (n_microtime, n_conditions) binary matrix
    """
    n_microtime = n_timepoints * microtime_resolution
    n_conditions = len(all_onsets)
    n_runs = len(run_starts)

    # Compute run lengths in TRs
    run_lengths = []
    for i in range(n_runs):
        if i < n_runs - 1:
            run_lengths.append(run_starts[i + 1] - run_starts[i])
        else:
            run_lengths.append(n_timepoints - run_starts[i])

    # Initialize onset matrix
    onset_matrix = torch.zeros(
        (n_microtime, n_conditions), dtype=torch.float32, device=device
    )

    # Microtime bin size
    microtime_dt = tr / microtime_resolution

    for cond_idx in range(n_conditions):
        duration = stim_durations[cond_idx]
        duration_bins = max(1, int(np.round(duration / microtime_dt)))

        # Boxcar value is 1.0 (AFNI convention)
        # The convolution function scales by dt, so the integral is properly computed.
        # Result: A 3s event produces ~3x larger response than a 1s event (block scaling)
        boxcar_value = 1.0

        for run_idx in range(n_runs):
            onsets = all_onsets[cond_idx][run_idx]
            run_start_tr = run_starts[run_idx]

            for onset_time in onsets:
                # Convert onset time (seconds) to microtime bin
                # onset_time is relative to run start
                global_time = run_start_tr * tr + onset_time
                microtime_bin = int(np.round(global_time / microtime_dt))

                if 0 <= microtime_bin < n_microtime:
                    # Place boxcar
                    end_bin = min(microtime_bin + duration_bins, n_microtime)
                    onset_matrix[microtime_bin:end_bin, cond_idx] = boxcar_value

    return onset_matrix


def parse_input_files(input_arg: Union[str, List[str]]) -> List[str]:
    """Parse input files (can be list from nargs='+' or single string)

    Supports:
    - Single file: "/path/to/file.nii.gz"
    - Multiple files: ["/path/run1.nii.gz", "/path/run2.nii.gz"]
    - Glob patterns: ["run*.nii.gz"] or "run*.nii.gz"
    """
    import glob as glob_module

    # Handle both list (from nargs='+') and string
    if isinstance(input_arg, str):
        input_arg = input_arg.strip().strip('"').strip("'")
        input_list = input_arg.split()
    else:
        input_list = input_arg

    # Expand globs and collect files
    files = []
    for pattern in input_list:
        matches = glob_module.glob(pattern)
        if matches:
            files.extend(sorted(matches))
        else:
            files.append(pattern)

    # Validate files exist
    for f in files:
        if not Path(f).exists():
            print(f"ERROR: Input file not found: {f}")
            sys.exit(1)

    return files


def parse_durations(
    durations_arg: List[str],
    n_conditions: int,
    condition_labels: List[str],
) -> List[float]:
    """Parse durations argument.

    Supports:
    - Single value: applies to all conditions
    - Multiple values: one per condition (in same order as onsets)
    """
    if len(durations_arg) == 1:
        # Single duration for all conditions
        try:
            dur = float(durations_arg[0])
            return [dur] * n_conditions
        except ValueError:
            print(f"ERROR: Could not parse duration '{durations_arg[0]}' as float")
            sys.exit(1)
    elif len(durations_arg) == n_conditions:
        # One duration per condition
        try:
            return [float(d) for d in durations_arg]
        except ValueError as e:
            print(f"ERROR: Could not parse durations: {e}")
            sys.exit(1)
    else:
        print(
            f"ERROR: Number of durations ({len(durations_arg)}) must be 1 or match "
            f"number of conditions ({n_conditions})"
        )
        print(f"  Conditions: {condition_labels}")
        sys.exit(1)


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="3dHRFoptfast - Fast GPU-accelerated cross-validated HRF optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,  # We handle -help ourselves to avoid required arg check
        epilog="""
Examples:
  # Basic HRF optimization with library of HRF variants
  3dHRFoptfast -input run1.nii.gz run2.nii.gz run3.nii.gz \\
               -onsets cond1.txt cond2.txt \\
               -durations 2.0 \\
               -tr 2.0 \\
               -prefix subject01_hrf

  # Using PIGHS library with different durations per condition
  3dHRFoptfast -input run*.nii.gz \\
               -onsets face.txt house.txt object.txt \\
               -durations 0.5 2.0 2.0 \\
               -tr 1.5 \\
               -hrf_mode pighs \\
               -n_hrfs 30 \\
               -prefix sub01_pighs

  # Leave-one-run-out CV (default) with verbose output
  3dHRFoptfast -input run*.nii.gz \\
               -onsets stim.txt \\
               -durations 1.0 \\
               -tr 2.0 \\
               -cv_strategy loro \\
               -verbose \\
               -prefix output

  # Split-halves CV with mask
  3dHRFoptfast -input run*.nii.gz \\
               -onsets cond.txt \\
               -durations 2.0 \\
               -tr 2.0 \\
               -cv_strategy 0.5 \\
               -mask brain_mask.nii.gz \\
               -prefix masked_output

  # With motion and physio nuisance regressors
  3dHRFoptfast -input run*.nii.gz \\
               -onsets face.txt house.txt \\
               -durations 2.0 \\
               -tr 2.0 \\
               -ortvec motion_all.1D motion \\
               -ortvec physio_all.txt physio \\
               -prefix sub01_with_nuisance

Outputs:
  {prefix}_hrf_index.nii.gz       - Which HRF (1-N) was selected per voxel
  {prefix}_xval_r2.nii.gz         - Cross-validated R² for selected HRF
  {prefix}_xval_r2_all_hrfs.nii.gz - 4D: CV R² for each HRF (volume per HRF)
  {prefix}_stats.nii.gz           - Final GLM betas and t-stats (AFNI bucket format)
  {prefix}_hrf_library.pt         - HRF library + voxel assignments for ARMA reuse
  {prefix}_metadata.json          - Full metadata for reproducibility

Notes:
  - Durations can be specified as single value (applies to all) or one per condition
  - Nuisance files (-ortvec) must be pre-concatenated across runs (matching total timepoints)
  - Common nuisance files: motion parameters (6 columns), physiological regressors, etc.
  - Future: onset files with 'married' durations (e.g., "1:2 4:5") will be supported
  - HRF library is saved for later ARMA/REML analysis with 3dREMLfast
        """,
    )

    # Required arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-input",
        nargs="+",
        required=True,
        help="Input fMRI dataset(s). Multiple files = multiple runs.",
    )
    required.add_argument(
        "-onsets",
        nargs="+",
        required=True,
        help="Onset timing files (AFNI format). One file per condition.",
    )
    required.add_argument(
        "-durations",
        nargs="+",
        required=True,
        help="Stimulus durations in seconds. Either single value for all conditions, or one per condition.",
    )
    required.add_argument(
        "-tr",
        type=float,
        required=False,
        default=None,
        help="Repetition time (TR) in seconds. If not specified, read from NIfTI header.",
    )
    required.add_argument(
        "-prefix",
        required=True,
        help="Output file prefix (e.g., 'output/subject01')",
    )

    # HRF library options
    hrf_opts = parser.add_argument_group("HRF Library Options")
    hrf_opts.add_argument(
        "-hrf_mode",
        choices=["library", "pighs"],
        default="library",
        help="HRF library type: 'library' (double-gamma variations) or 'pighs' (half-cosine basis). Default: library",
    )
    hrf_opts.add_argument(
        "-n_hrfs",
        type=int,
        default=20,
        help="Number of HRFs in library (default: 20, only affects pighs mode)",
    )

    # PIGHS-specific options
    pighs_opts = parser.add_argument_group(
        "PIGHS Options (only used with -hrf_mode pighs)"
    )
    pighs_opts.add_argument(
        "-pighs_peak_time_range",
        type=float,
        nargs=2,
        default=[3, 10],
        metavar=("MIN", "MAX"),
        help="Range for time-to-peak in seconds. Default: 3 10",
    )
    pighs_opts.add_argument(
        "-pighs_rise_fraction_range",
        type=float,
        nargs=2,
        default=[0.3, 0.9],
        metavar=("MIN", "MAX"),
        help="Range for rise_fraction (fraction of peak_time that is rise vs delay). Default: 0.3 0.9",
    )
    pighs_opts.add_argument(
        "-pighs_fall_range",
        type=float,
        nargs=2,
        default=[3, 10],
        metavar=("MIN", "MAX"),
        help="Range for peak-to-undershoot time (m3) in seconds. Default: 3 10",
    )
    pighs_opts.add_argument(
        "-pighs_recovery_range",
        type=float,
        nargs=2,
        default=[3, 12],
        metavar=("MIN", "MAX"),
        help="Range for undershoot recovery time (m4) in seconds. Default: 3 12",
    )
    pighs_opts.add_argument(
        "-pighs_undershoot_range",
        type=float,
        nargs=2,
        default=[0, 0.35],
        metavar=("MIN", "MAX"),
        help="Range for undershoot magnitude (c2). Default: 0 0.35",
    )

    # Cross-validation options
    cv_opts = parser.add_argument_group("Cross-Validation Options")
    cv_opts.add_argument(
        "-cv_strategy",
        default="loro",
        help=(
            "Cross-validation strategy. Options: "
            "'loro' or '1' for leave-one-run-out (default), "
            "'0.5' for split-halves, "
            "any float (0-1) for that train fraction, "
            "any int > 1 for leave-N-out"
        ),
    )
    cv_opts.add_argument(
        "-n_perms",
        type=int,
        default=100,
        help="Max number of CV permutations for random splits (default: 100)",
    )
    cv_opts.add_argument(
        "-metric",
        choices=["cod", "corr", "corr2"],
        default="cod",
        help="R² metric: 'cod' (coefficient of determination), 'corr', 'corr2' (default: cod)",
    )

    # Processing options
    proc_opts = parser.add_argument_group("Processing Options")
    proc_opts.add_argument(
        "-mask",
        help="Mask file to restrict analysis to brain voxels",
    )
    proc_opts.add_argument(
        "-polort",
        type=int,
        default=None,
        help="Polynomial order for drift modeling (default: auto based on run length)",
    )
    proc_opts.add_argument(
        "-ortvec",
        action="append",
        nargs=2,
        metavar=("FILE", "LABEL"),
        help=(
            "Additional nuisance regressors to project out (can be repeated). "
            "FILE is a text file with nuisance columns (AFNI 1D, CSV, or whitespace-separated). "
            "LABEL is a prefix for the column names. "
            "Files must span all runs concatenated (same length as total timepoints). "
            "Example: -ortvec motion_all.1D motion -ortvec physio.txt physio"
        ),
    )
    proc_opts.add_argument(
        "-microtime_resolution",
        type=int,
        default=20,
        help="Sub-TR resolution for onset timing (default: 20)",
    )
    proc_opts.add_argument(
        "-device",
        type=str,
        help="Force device: 'cpu' or 'cuda' (default: auto-detect GPU)",
    )
    proc_opts.add_argument(
        "-batch_size",
        type=int,
        default=None,
        help="Number of voxels per batch (default: auto)",
    )
    proc_opts.add_argument(
        "-verbose",
        action="store_true",
        help="Print detailed progress information",
    )

    # Output options
    out_opts = parser.add_argument_group("Output Options")
    out_opts.add_argument(
        "-fout",
        action="store_true",
        help="Include F-statistics in stats bucket",
    )
    out_opts.add_argument(
        "-tout",
        action="store_true",
        help="Include t-statistics in stats bucket",
    )
    out_opts.add_argument(
        "-rout",
        action="store_true",
        help="Include R² in stats bucket",
    )
    out_opts.add_argument(
        "-save_hrf_designs",
        action="store_true",
        help="Save individual design matrices for each HRF in the library. "
        "Creates a directory {prefix}_hrf_designs/ with xmat files that "
        "can be used for external GLM fitting (e.g., AFNI's 3dREMLfit).",
    )
    out_opts.add_argument(
        "-save_plots",
        action="store_true",
        help="Save design matrix and HRF library plots as PNG images.",
    )

    # Help
    parser.add_argument("-help", action="store_true", help="Show this help message")

    return parser


def parse_cv_strategy(cv_str: str) -> Union[int, float]:
    """Parse CV strategy string into int or float."""
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


def print_header(args):
    """Print program header"""
    print("=" * 70)
    print("3dHRFoptfast - GPU-Accelerated Cross-Validated HRF Optimization")
    print("=" * 70)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_summary(
    args, n_runs: int, n_conditions: int, n_voxels: int, condition_labels: List[str]
):
    """Print analysis summary"""
    print("=" * 70)
    print("📋 Analysis Summary")
    print("=" * 70)
    print(f"  Input runs: {n_runs}")
    print(f"  Conditions: {n_conditions} - {condition_labels}")
    print(f"  TR: {args.tr}s")
    print(f"  Voxels: {n_voxels:,}")
    print()
    print(f"  HRF mode: {args.hrf_mode}")
    print(f"  HRF candidates: {args.n_hrfs}")
    print(f"  CV strategy: {args.cv_strategy}")
    print(f"  Microtime resolution: {args.microtime_resolution}x")
    print()
    print(f"  Output prefix: {args.prefix}")
    print("=" * 70)
    print()


def main():
    parser = create_parser()

    # Check for help BEFORE parse_args to avoid required argument errors
    if len(sys.argv) == 1 or "-help" in sys.argv or "--help" in sys.argv:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    print_header(args)

    # ==========================================================================
    # 1. Parse and validate inputs
    # ==========================================================================

    # Parse input files
    input_files = parse_input_files(args.input)
    n_runs = len(input_files)

    if n_runs < 2:
        print("ERROR: At least 2 runs required for cross-validation")
        sys.exit(1)

    # Parse onset files
    onset_files = args.onsets
    n_conditions = len(onset_files)
    condition_labels = [Path(f).stem for f in onset_files]

    # Validate onset files exist
    for f in onset_files:
        if not Path(f).exists():
            print(f"ERROR: Onset file not found: {f}")
            sys.exit(1)

    # Parse durations
    durations = parse_durations(args.durations, n_conditions, condition_labels)
    print(f"  Durations: {durations}s")

    # Parse CV strategy
    cv_strategy = parse_cv_strategy(args.cv_strategy)

    # Setup device
    if args.device:
        if args.device.lower() == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
    else:
        device = get_device()
    print(f"  Device: {device}")

    # ==========================================================================
    # 2. Load data
    # ==========================================================================

    print()
    print("Loading data...")

    # Load mask if provided
    mask = None
    if args.mask:
        mask = load_afni_mask(args.mask)
        print(f"  Mask: {args.mask} ({mask.sum():,} voxels)")

    # Load and concatenate runs
    # Note: load_and_concatenate_runs returns (data, run_starts)
    from pathlib import Path as PathLib  # avoid shadowing
    from typing import cast

    run_paths: list[Union[str, PathLib]] = [PathLib(f) for f in input_files]
    data, run_starts = load_and_concatenate_runs(
        cast(list[Union[str, PathLib]], run_paths),
        device=device,
    )

    # Get affine and TR from first input
    first_img = nib.load(input_files[0])
    affine = np.array(first_img.affine) if hasattr(first_img, "affine") else np.eye(4)
    volume_shape = (
        tuple(first_img.shape[:3]) if hasattr(first_img, "shape") else (0, 0, 0)
    )

    # Get TR from header if not provided
    if args.tr is None:
        # TR is in zooms[3] for 4D NIfTI (pixdim[4] in raw header)
        zooms = first_img.header.get_zooms()
        if len(zooms) > 3 and zooms[3] > 0:
            args.tr = float(zooms[3])
            print(f"  TR from header: {args.tr}s")
        else:
            print("ERROR: Could not determine TR from NIfTI header.")
            print("       Please specify TR with -tr option.")
            sys.exit(1)
    else:
        print(f"  TR (specified): {args.tr}s")

    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
        data = data[mask_flat, :]

    n_voxels, n_timepoints = data.shape

    print(
        f"  Data shape: {data.shape} ({n_voxels:,} voxels × {n_timepoints} timepoints)"
    )
    print(f"  Volume shape: {volume_shape}")
    print(f"  Runs: {n_runs} starting at {run_starts}")

    # ==========================================================================
    # 3. Parse onset files and build onset matrix
    # ==========================================================================

    print()
    print("Building onset matrix...")

    # Parse onset files (AFNI format: one row per run)
    all_onsets = []
    for i, onset_file in enumerate(onset_files):
        onsets_by_run = parse_afni_timing_file(onset_file)

        # Validate number of runs matches
        if len(onsets_by_run) != n_runs:
            print(
                f"ERROR: Onset file '{onset_file}' has {len(onsets_by_run)} runs, "
                f"but {n_runs} input files were provided"
            )
            sys.exit(1)

        all_onsets.append(onsets_by_run)
        if args.verbose:
            n_events = sum(len(r) for r in onsets_by_run)
            print(f"  {condition_labels[i]}: {n_events} events across {n_runs} runs")

    # Build microtime onset matrix
    # This creates a (n_microtime, n_conditions) binary matrix
    onset_matrix = create_onset_matrix_microtime(
        all_onsets,
        run_starts,
        args.tr,
        n_timepoints,
        args.microtime_resolution,
        stim_durations=durations,
        device=device,
    )

    print(f"  Onset matrix shape: {onset_matrix.shape}")

    # ==========================================================================
    # 4. Generate HRF library
    # ==========================================================================

    print()
    print(f"Generating {args.hrf_mode} HRF library ({args.n_hrfs} candidates)...")

    # Build PIGHS kwargs if using pighs mode
    pighs_kwargs = {}
    if args.hrf_mode == "pighs":
        pighs_kwargs = {
            "peak_time_range": tuple(args.pighs_peak_time_range),
            "rise_fraction_range": tuple(args.pighs_rise_fraction_range),
            "fall_time_range": tuple(args.pighs_fall_range),
            "recovery_time_range": tuple(args.pighs_recovery_range),
            "undershoot_range": tuple(args.pighs_undershoot_range),
        }

    # Generate HRF library as IMPULSE RESPONSES (stim_duration=0)
    # Duration-based convolution is handled separately in the design matrix construction
    # This ensures the HRF library represents the actual HRF shapes, not duration-convolved versions
    hrf_library = get_hrf_library(
        mode=args.hrf_mode,
        stim_duration=0.0,  # Impulse response - duration handled in design matrix
        tr=args.tr,
        n_hrfs=args.n_hrfs,
        device=device,
        **pighs_kwargs,
    )

    print(f"  HRF library shape: {hrf_library.shape}")
    print(
        f"  HRF length: {hrf_library.shape[1]} TRs ({hrf_library.shape[1] * args.tr}s)"
    )

    # ==========================================================================
    # 5. Run cross-validated HRF selection
    # ==========================================================================

    print_summary(args, n_runs, n_conditions, n_voxels, condition_labels)

    # Convert ortvec argument to list of tuples if provided
    ortvec_files = None
    if args.ortvec:
        ortvec_files = [(f, label) for f, label in args.ortvec]
        if args.verbose:
            for f, label in ortvec_files:
                print(f"  Nuisance: {f} (label={label})")

    results = fit_glm_hrf_library_with_xval(
        data=data,
        onsets=onset_matrix,
        hrf_library=hrf_library,
        tr=args.tr,
        run_starts=run_starts,
        stim_durations=durations,
        cv_strategy=cv_strategy,
        n_perms=args.n_perms,
        metric=args.metric,
        microtime_resolution=args.microtime_resolution,
        polort=args.polort,
        ortvec_files=ortvec_files,
        device=device,
        verbose=args.verbose,
        chunk_size=args.batch_size,
    )

    # Update metadata with CLI parameters
    results.hrf_metadata["hrf_mode"] = args.hrf_mode
    results.hrf_metadata["condition_labels"] = condition_labels
    results.hrf_metadata["input_files"] = input_files
    results.hrf_metadata["onset_files"] = onset_files
    results.hrf_metadata["durations"] = durations
    if ortvec_files:
        results.hrf_metadata["ortvec_files"] = [(str(f), label) for f, label in ortvec_files]

    # ==========================================================================
    # 6. Save outputs
    # ==========================================================================

    print()
    print("Saving outputs...")

    # Create output directory if needed
    output_prefix = Path(args.prefix)
    output_dir = output_prefix.parent
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Convert mask to voxel_mask tensor if provided
    voxel_mask = None
    if mask is not None:
        voxel_mask = torch.from_numpy(mask.flatten().astype(bool))

    # Save results
    output_files = save_hrf_selection_results(
        results=results,
        output_prefix=str(args.prefix),
        volume_shape=volume_shape,
        affine=affine,
        voxel_mask=voxel_mask,
        condition_labels=condition_labels,
        run_starts=run_starts,
        save_all_hrf_designs=args.save_hrf_designs,
        onsets=onset_matrix if args.save_hrf_designs else None,
        save_plots=args.save_plots,
    )

    # Print output summary
    print()
    print("=" * 70)
    print("📁 Output Files")
    print("=" * 70)
    for output_type, filepath in output_files.items():
        print(f"  {output_type}: {filepath}")
    print("=" * 70)

    # Print final summary
    print()
    print("=" * 70)
    print("✅ 3dHRFoptfast Complete!")
    print("=" * 70)
    print(f"  Mean xval R²: {results.xval_r2_best.mean().item():.4f}")
    if results.xval_r2_canonical is not None:
        canonical_r2 = results.xval_r2_canonical.mean().item()
        best_r2 = results.xval_r2_best.mean().item()
        print(f"  Canonical HRF baseline R²: {canonical_r2:.4f}")
        print(f"  Improvement over canonical: {best_r2 - canonical_r2:+.4f}")
    if results.final_results is not None:
        print(f"  Final R² (full data): {results.final_results.r2.mean().item():.4f}")
    print()
    print("  HRF usage distribution:")
    hrf_counts = results.hrf_metadata["hrf_usage_counts"]
    for i, count in enumerate(hrf_counts):
        if count > 0:
            pct = 100 * count / n_voxels
            print(f"    HRF {i}: {count:,} voxels ({pct:.1f}%)")
    print()
    print(f"🕐 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
