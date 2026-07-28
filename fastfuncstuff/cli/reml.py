#!/usr/bin/env python3
"""
ffs_reml - Fast ARMA(1,1) GLM for fMRI using GPU acceleration

This is a PyTorch/GPU-accelerated implementation of AFNI's 3dREMLfit,
providing 5-50x speedup for ARMA(1,1) prewhitened GLM fitting.

Basic usage:
    ffs_reml -input func.nii.gz -matrix X.xmat.1D -Rnuisance stats_REML

For help:
    ffs_reml -help
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

# Reduce CUDA fragmentation: groups allocate/free n_time×n_time Cholesky tensors
# in a loop while a multi-GB grid stays resident — exactly the workload
# expandable_segments was built for. Must be set before `import torch`.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

try:
    import nibabel as nib  # noqa: F401 — availability check
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

# Import fastfuncstuff modules
try:
    from fastfuncstuff.analysis import analyze_from_design_matrix
    from fastfuncstuff.cli_utils import (
        add_ortvec_arguments,
        add_verbose_arg,
        auto_polort,
        build_nuisance_block_diag,
        collect_nuisance_blocks,
        compute_run_lengths,
        get_average_run_duration,
        parse_device_arg,
        parse_input_files,
        parse_prefix,  # noqa: F401 — TODO: apply parse_prefix to individual output flags
    )
    from fastfuncstuff.design.builder import (
        create_onset_matrix_microtime,
        parse_afni_timing_file,
        parse_durations,
    )
    from fastfuncstuff.glm.outputs import (
        slice_glm_results,
        write_glm_bucket_as_nifti,
        write_single_trials_output,
    )
    from fastfuncstuff.io.afni import (
        get_tr_from_file,
        load_nifti,
        nifti_shape,
        read_afni_design_matrix,  # noqa: F401 — re-imported in sub-function but also used at module scope
        replace_afni_extension,
        save_nifti,
    )
    from fastfuncstuff.utils import (
        configure_torch_backends,
        gaussian_blur_3d,
        scale_to_percent_signal,
    )
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    print("Make sure fastfuncstuff is installed: pip install -e .")
    sys.exit(1)


def parse_grid_arg(grid_str: str) -> torch.Tensor:
    """
    Parse a grid argument into a 1D tensor of float values.

    Accepted formats:
      start:stop:num   Linspace range, e.g.  0.025:0.9:32  or  -0.9:0.9:65
                       (colon separator avoids argparse negative-number issues)
      v1,v2,v3,...     Explicit list, e.g.   0.1,0.2,0.3,0.5,0.7,0.9
      start,stop,num   Legacy linspace (3 comma-sep values, last is integer ≥ 2)
    """
    grid_str = grid_str.strip()

    # ── Colon-separated linspace: start:stop:num ─────────────────────────────
    if ":" in grid_str:
        parts = grid_str.split(":")
        if len(parts) != 3:
            print(f"ERROR: colon range must be start:stop:num, got '{grid_str}'")
            sys.exit(1)
        try:
            start, stop, num = float(parts[0]), float(parts[1]), int(parts[2])
        except ValueError:
            print(f"ERROR parsing range '{grid_str}'")
            sys.exit(1)
        if num < 2:
            print(f"ERROR: num_points must be >= 2 in '{grid_str}'")
            sys.exit(1)
        return torch.linspace(start, stop, num)

    # ── Comma-separated: explicit list or legacy start,stop,num ──────────────
    parts = grid_str.split(",")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        print(f"ERROR: could not parse grid values '{grid_str}'")
        sys.exit(1)

    # Legacy detection: exactly 3 values where the last is a whole number ≥ 2
    if len(values) == 3 and values[2] == int(values[2]) and int(values[2]) >= 2:
        start, stop, num = values[0], values[1], int(values[2])
        return torch.linspace(start, stop, num)

    # Explicit list
    if len(values) < 1:
        print(f"ERROR: grid must have at least one value: '{grid_str}'")
        sys.exit(1)
    return torch.tensor(values)


def _extract_grid_args(argv: list[str]) -> tuple[list[str], str | None, str | None]:
    """
    Pull -a_grid / -b_grid values out of argv before argparse sees them.

    argparse cannot handle option values that start with '-' unless they look
    like bare negative numbers (e.g. '-0.9' is fine, but '-0.9,0.9,65' is not).
    Extracting these manually sidesteps the issue entirely.
    """
    a_grid_str: str | None = None
    b_grid_str: str | None = None
    new_argv: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("-a_grid", "--a_grid") and i + 1 < len(argv):
            a_grid_str = argv[i + 1]
            i += 2
            continue
        if tok in ("-b_grid", "--b_grid") and i + 1 < len(argv):
            b_grid_str = argv[i + 1]
            i += 2
            continue
        if tok.startswith(("-a_grid=", "--a_grid=")):
            a_grid_str = tok.split("=", 1)[1]
            i += 1
            continue
        if tok.startswith(("-b_grid=", "--b_grid=")):
            b_grid_str = tok.split("=", 1)[1]
            i += 1
            continue
        new_argv.append(tok)
        i += 1
    return new_argv, a_grid_str, b_grid_str


def detect_format(filepath: str) -> str:
    """Detect file format from extension"""
    p = Path(filepath)
    if p.suffix == ".gz" and p.stem.endswith(".nii"):
        return "nii.gz"
    elif p.suffix in [".nii"]:
        return "nii"
    elif "+orig" in p.name or "+tlrc" in p.name:
        return "afni"
    else:
        # Default to nifti
        return "nii.gz"


def extract_onset_times_from_design(design_matrix: np.ndarray, column_indices: list) -> list:
    """
    Extract onset times for stimulus columns from design matrix.

    For each stimulus column, finds the first timepoint where the column becomes non-zero.
    This represents the onset time of that stimulus.

    Parameters
    ----------
    design_matrix : np.ndarray
        Design matrix (n_timepoints, n_regressors)
    column_indices : list of int
        Column indices to extract onset times for

    Returns
    -------
    onset_times : list of int
        Onset timepoint for each column (same length as column_indices)
    """
    onset_times = []

    for col_idx in column_indices:
        column = design_matrix[:, col_idx]

        # Find first non-zero timepoint
        nonzero_indices = np.nonzero(column)[0]

        if len(nonzero_indices) > 0:
            onset_time = int(nonzero_indices[0])
        else:
            # Column is all zeros - use large value to sort to end
            onset_time = len(column) + col_idx  # Add col_idx to maintain stable sort

        onset_times.append(onset_time)

    return onset_times


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults while preserving raw description formatting."""


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="ffs_reml - Fast GPU-accelerated ARMA(1,1) GLM fitting",
        formatter_class=_HelpFormatter,
        epilog="""
Examples:
  # Basic REML analysis with main bucket output
  ffs_reml -input func.nii.gz -matrix X.xmat.1D -Rbuck stats_REML
  
  # Multiple runs
  ffs_reml -input "run1.nii.gz run2.nii.gz run3.nii.gz" \\
             -matrix X.xmat.1D -Rbuck stats_REML
  
  # With all outputs
  ffs_reml -input func.nii.gz -matrix X.xmat.1D \\
             -Rbuck stats_REML -Rvar params_REML \\
             -Rfitts fitts_REML -Rerrts errts_REML \\
             -Rbeta betas_only_REML -fout -tout -rout
  
  # With OLS comparison
  ffs_reml -input func.nii.gz -matrix X.xmat.1D \\
             -Rbuck stats_REML -Obuck stats_OLS
  
  # Nuisance regressors only
  ffs_reml -input func.nii.gz -matrix X.xmat.1D \\
             -Rnuisance nuisance_REML -Onuisance nuisance_OLS

  # Single-trial outputs (reordered by onset time)
  ffs_reml -input func.nii.gz -matrix X.xmat.1D \\
             -Rbuck stats_REML -single_trials movie
  # Creates: ols_movie_single.nii.gz, reml_movie_single.nii.gz

  # Wider ARMA grid (default matches AFNI: a in [0,0.8], b in [-0.8,0.8])
  ffs_reml -input func.nii.gz -matrix X.xmat.1D \\
             -Rbuck stats_REML -use_double \\
             -a_grid 0:0.9:10 -b_grid -0.9:0.9:19

  # Manual batch size control (for memory tuning)
  ffs_reml -input func.nii.gz -matrix X.xmat.1D \\
             -Rbuck stats_REML -batch_size 2000
        """,
    )

    # Required arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "-input",
        nargs="+",
        required=True,
        help="Input fMRI dataset(s). Can be single file, multiple files, or glob patterns (e.g., run*.nii.gz)",
    )
    required.add_argument(
        "-matrix",
        required=False,
        help="Design matrix file (X.xmat.1D from 3dDeconvolve). Mutually exclusive with -onsets / -events / -spec.",
    )
    required.add_argument(
        "-spec",
        required=False,
        help="ffs_design_spec TOML file. Compiled to <specname>.xmat.1D before "
        "the GLM runs (so REML errors out fast on a bad spec rather than "
        "after a long data load). Mutually exclusive with -matrix / -onsets / -events.",
    )
    required.add_argument(
        "-xmat",
        required=False,
        help="When using -spec, override the auto-derived output xmat path "
        "(default: <specname>.xmat.1D in the spec's directory). Ignored "
        "without -spec.",
    )
    required.add_argument(
        "-overwrite",
        action="store_true",
        help="Allow -spec compile to overwrite an existing -xmat output.",
    )

    # Output arguments - REML
    reml_out = parser.add_argument_group("REML Output Options")
    reml_out.add_argument(
        "-Rvar",
        help="Output REML variance parameters (up to 5 volumes: a, b, lambda, StDev, -LogLik)",
    )
    reml_out.add_argument(
        "-Rlklhd",
        dest="Rlklhd",
        metavar="PREFIX",
        help="Output full REML likelihood surface: 4D NIfTI with one sub-brik per valid "
        "(a,b) grid point (~117 sub-briks). Sub-brik k = L(a_k, b_k) per voxel. "
        "Sub-briks are labeled 'a=0.00_b=0.30' etc. Argmin sub-brik identifies "
        "the selected (a,b) pair for each voxel.",
    )
    reml_out.add_argument(
        "-load_Rvar",
        help="Load precomputed ARMA parameters from this file to skip the grid search "
        "(saves ~80%% compute time on re-runs). Must be a -Rvar output from a "
        "previous run. If not specified, the grid search always runs from scratch.",
    )
    reml_out.add_argument("-Rbuck", help="Output REML betas + statistics (main bucket output)")
    reml_out.add_argument("-Rbeta", help="Output REML betas only (no statistics)")
    reml_out.add_argument(
        "-Rnuisance", help="Output REML betas + statistics for NUISANCE regressors only"
    )
    reml_out.add_argument("-Rfitts", help="Output REML fitted model time series")
    reml_out.add_argument("-Rerrts", help="Output REML residuals")
    reml_out.add_argument("-Rwherr", help="Output REML whitened residuals")
    reml_out.add_argument(
        "-save_clean",
        "-save-clean",
        dest="save_clean",
        metavar="PATH",
        help="Save a nuisance-removed ('clean') timeseries: the raw data minus the "
        "fitted nuisance signal (everything NOT a stimulus column — polynomials, "
        "motion, etc.), with the per-voxel temporal mean added back so the series "
        "sits at the original signal level. Equals task_fit + residuals + mean. "
        "Because the GLM fits task and nuisance jointly, the task keeps variance a "
        "plain projection would leak into nuisance. Off by default; requires the full "
        "design (not StimBots/StimTops-filtered).",
    )
    reml_out.add_argument(
        "-save_nuisance",
        "-save-nuisance",
        dest="save_nuisance",
        metavar="PATH",
        help="Save the reconstructed nuisance-only timeseries (fitted nuisance "
        "regressors × their betas) — the signal -save_clean removes. Off by default.",
    )
    reml_out.add_argument(
        "-save_taskfit",
        "-save-taskfit",
        dest="save_taskfit",
        metavar="PATH",
        help="Save the pure task fit (stimulus columns × their betas), with NO "
        "residual and NO mean. This is the smooth model signal; -save_clean is the "
        "same task fit PLUS the residual (and baseline). Comparing the two shows what "
        "the model explains vs. what it leaves as noise. Off by default. Together with "
        "-save_nuisance and -Rerrts it gives the exact partition data = taskfit + "
        "nuisance + residual.",
    )
    reml_out.add_argument(
        "-save_per_run_polort",
        "-save-per-run-polort",
        dest="save_per_run_polort",
        action="store_true",
        help="Modifier for -save_clean: instead of adding back one per-voxel grand "
        "mean, keep each run's fitted polort-0 baseline (the per-run means). Preserves "
        "run-to-run offsets rather than collapsing them to a common level. No effect "
        "without -save_clean.",
    )

    # Whole-dataset diagnostics: cheap maps that fall out of data already resident
    # during the fit (the one point the entire timeseries is in RAM). See
    # glm/reml_diagnostics.py. Any of these forces the manual load path.
    diag_out = parser.add_argument_group("Whole-dataset Diagnostics")
    diag_out.add_argument(
        "-save_grandmean",
        "-save-grandmean",
        dest="save_grandmean",
        metavar="PATH",
        help="Save the grand mean (per-voxel temporal mean, BEFORE scaling) as NIfTI.",
    )
    diag_out.add_argument(
        "-save_tsnr",
        "-save-tsnr",
        dest="save_tsnr",
        metavar="PREFIX",
        help="Save temporal-SNR maps: PREFIX.raw_tsnr (mean/std of the scaled data; "
        "requires -do_scale) and, when residuals are available, PREFIX.resid_tsnr_ols "
        "/ PREFIX.resid_tsnr_reml (scaled mean / residual std).",
    )
    diag_out.add_argument(
        "-save_acf",
        "-save-acf",
        dest="save_acf",
        metavar="PREFIX",
        help="Save per-run spatial ACF / effective FWHM of the residuals (3dFWHMx-style, "
        "a*exp(-r^2/2b^2)+(1-a)*exp(-r/c)) as PREFIX.fwhmx_ols.txt / .fwhmx_reml.txt.",
    )
    diag_out.add_argument(
        "-save_mask",
        "-save-mask",
        dest="save_mask",
        metavar="PATH",
        help="Save the diagnostics brain mask as NIfTI. When no -mask is given, this "
        "is the automask computed from the grand mean (the same mask -save_acf / "
        "-save_tsnr use). No-op if -mask was supplied (that mask is already on disk).",
    )
    diag_out.add_argument(
        "-adjust_dof",
        "-adjust-dof",
        dest="adjust_dof",
        metavar="MAP|N",
        default=None,
        help="After writing the stat buckets, correct their degrees of freedom "
        "(e.g. for NORDIC component removal): subtract N (a number) or a per-voxel "
        "map of lost dof, convert each t/F sub-brick to a z-score at the new dof, "
        "and insert it (Coef,Tstat -> Coef,Tstat,Zstat). Applies to -Obuck/-Rbuck. "
        "Voxels with new dof<=0 are flagged in a *_invalid_dof map. Same as "
        "running ffs_util_updatedof on the outputs.",
    )

    # Output arguments - OLS
    ols_out = parser.add_argument_group("OLS Output Options (for comparison)")
    ols_out.add_argument("-Obuck", help="Output OLS betas + statistics (main bucket output)")
    ols_out.add_argument("-Obeta", help="Output OLS betas only (no statistics)")
    ols_out.add_argument(
        "-Onuisance", help="Output OLS betas + statistics for NUISANCE regressors only"
    )
    ols_out.add_argument("-Oerrts", help="Output OLS residuals (the OLS-baseline errts)")
    ols_out.add_argument("-Ofitts", help="Output OLS fitted model time series")

    # FDR options
    fdr_group = parser.add_argument_group("FDR Options")
    fdr_group.add_argument(
        "-add_fdr",
        action="store_true",
        help=(
            "Compute and store AFNI FDRCURVE attributes for every stat sub-brick "
            "in -Rbuck / -Obuck / -Rnuisance / -Onuisance outputs (matches AFNI's "
            "3drefit -addFDR behavior). The GUI / fdrval can then read q from a "
            "threshold without re-running BH."
        ),
    )

    # Special output options
    special_out = parser.add_argument_group("Special Output Options")
    special_out.add_argument(
        "-single_trials",
        "-single-trials",
        dest="single_trials",
        type=str,
        default=None,
        metavar="LABEL",
        help=(
            "Single-trial mode. Rebuilds the design so each event onset is its own "
            "regressor (GLMsingle-style) and fits trial-specific betas, then writes "
            "them ordered chronologically. LABEL is inserted into filenames: "
            "ols_LABEL_single.nii.gz, reml_LABEL_single.nii.gz. Requires -events or "
            "-onsets/-durations; not compatible with -matrix, -hrfopt_prefix, or FIR/TENT."
        ),
    )

    # Statistics options
    stats_opts = parser.add_argument_group("Statistics Options")
    stats_opts.add_argument(
        "-fout", action="store_true", help="Include F-statistics in output buckets"
    )
    stats_opts.add_argument(
        "-tout", action="store_true", help="Include t-statistics in output buckets"
    )
    stats_opts.add_argument(
        "-rout",
        action="store_true",
        help="Include R² statistics in output buckets (total model R²)",
    )
    stats_opts.add_argument(
        "-rpartial",
        nargs="?",
        const="full",
        choices=["full", "task"],
        help="Include partial R² per condition in output buckets. "
        "'full' (default): partial R² as proportion of total variance. "
        "'task': partial R² as proportion of variance remaining after nuisance regressors (more interpretable for task effects). "
        "NOTE: Partial R² values do NOT sum to total R² (they sum to MORE due to shared variance between regressors).",
    )
    stats_opts.add_argument(
        "-r2semipartial",
        nargs="?",
        const="full",
        choices=["full", "task"],
        help="Include semi-partial R² (squared part correlation) per condition in output buckets. "
        "'full' (default): semi-partial R² as proportion of total variance. "
        "'task': semi-partial R² as proportion of variance remaining after nuisance regressors. "
        "Semi-partial R² shows unique variance contribution and DOES sum to total R² (additive contributions). "
        "Formula: r²_semi = (R²_full - R²_without_regressor)",
    )
    stats_opts.add_argument(
        "-beta_cv",
        action="store_true",
        help="Use beta-space cross-validation (GLMsingle-style). "
        "Fits single-trial model once on all data with ARMA(1,1) prewhitening, "
        "evaluates R² on condition-averaged vs individual trial betas across folds. "
        "Requires -onsets (not -matrix). "
        "NOTE: This is different from -single_trials (which reorders output by onset time).",
    )

    # ARMA grid options
    arma_opts = parser.add_argument_group("ARMA(1,1) Grid Options")
    arma_opts.add_argument(
        "-a_grid",
        help=(
            "AR parameter grid. Default matches AFNI 3dREMLfit:\n"
            "  0:0.8:9  (i.e. [0.0, 0.1, ..., 0.8], step=0.1, MAXa=0.8)\n"
            "Three formats accepted:\n"
            "  start:stop:num  — linspace, e.g.  0:0.9:10  (widen MAXa to 0.9)\n"
            "  v1,v2,...       — explicit list, e.g.  0.1,0.3,0.5,0.7,0.9\n"
            "  start,stop,num  — legacy linspace, e.g.  0,0.9,10\n"
            "Absolute upper bound is 0.9 (ARMA(1,1) is degenerate above)."
        ),
    )
    arma_opts.add_argument(
        "-b_grid",
        help=(
            "MA parameter grid. Default matches AFNI 3dREMLfit:\n"
            "  -0.8:0.8:17  (i.e. [-0.8, -0.7, ..., 0.7, 0.8], step=0.1, MAXb=0.8)\n"
            "Three formats accepted:\n"
            "  start:stop:num  — linspace, e.g.  -0.9:0.9:19  (widen MAXb to 0.9)\n"
            "  v1,v2,...       — explicit list, e.g.  -0.5,-0.2,0.0,0.2,0.5\n"
            "  start,stop,num  — legacy linspace, e.g.  -0.9,0.9,19\n"
            "Absolute upper bound is 0.9 (ARMA(1,1) is degenerate above)."
        ),
    )
    arma_opts.add_argument(
        "-afni_mode",
        "-afni-mode",
        action="store_true",
        help=(
            "Match AFNI 3dREMLfit byte-for-byte on the small (a,b) divergences "
            "instead of FFS's more-accurate defaults. Switches three things to "
            "AFNI's choices: banded covariance (corcut=1e-4 truncation) instead "
            "of the exact dense Toeplitz; the coarser hierarchical search top "
            "level; and dropping near-white grid points (0<lam<corcut). "
            "Default OFF — FFS is at least as accurate. Use only for AFNI parity."
        ),
    )
    arma_opts.add_argument(
        "-grid_batching",
        action="store_true",
        help=(
            "Force grid batching mode (low memory, slightly slower). "
            "Processes all voxels for each (a,b) pair instead of precomputing the full grid. "
            "Memory: ~3 GB regardless of grid size. "
            "Default: auto-detect (uses grid batching if grid > 8 GB). "
            "Best for: long timeseries, double precision, limited GPU memory."
        ),
    )
    arma_opts.add_argument(
        "-no_grid_batching",
        action="store_true",
        help=(
            "Force full grid precomputation (AFNI approach, faster but more memory). "
            "Precomputes all Cholesky factorizations once, then reuses for all voxels. "
            "Memory: can be 10+ GB with long timeseries and double precision. "
            "Default: auto-detect (uses full grid if grid ≤ 8 GB). "
            "Best for: short timeseries, float32, abundant GPU memory."
        ),
    )
    arma_opts.add_argument(
        "-quick_estimate",
        action="store_true",
        help=(
            "EXPERIMENTAL: Enable fast grid search with early stopping (GPU only). "
            "Uses smart ordering + batch convergence detection to stop early. "
            "Can be 2-3x faster but may miss true optima for some voxels. "
            "Default: exhaustive search (recommended for publication). "
            "Use this flag ONLY for exploratory analysis or when speed is critical."
        ),
    )
    arma_opts.add_argument(
        "-exhaustive",
        action="store_true",
        help=(
            "Force exhaustive 117-point (a,b) grid search on CPU. Default on "
            "CPU is AFNI-style hierarchical descent (matches 3dREMLfit, ~3x "
            "fewer pair evaluations). GPU always uses exhaustive. Required "
            "when -Rlklhd is set since the likelihood surface needs every "
            "grid point evaluated."
        ),
    )

    # Processing options
    proc_opts = parser.add_argument_group("Processing Options")
    proc_opts.add_argument(
        "-use_double",
        action="store_true",
        help="Use double precision (float64) - matches AFNI exactly, ~2x memory, ~1.5x slower",
    )
    proc_opts.add_argument("-mask", help="Mask file to restrict analysis")
    proc_opts.add_argument(
        "-censor",
        metavar="FILE.1D",
        help=(
            "Censor file: one value per concatenated TR (1=keep, 0=censor), in "
            "order across runs (e.g. 600 rows for 3×200-TR runs). Censored TRs "
            "are dropped from data and design; the ARMA noise model steps across "
            "each gap via tau (lag respects the true time distance), and run "
            "boundaries stay a hard cut. Use with -onsets/-events; for -matrix "
            "the xmat GoodList is honoured automatically (and -censor overrides)."
        ),
    )
    proc_opts.add_argument(
        "-do_scale",
        action="store_true",
        help="Scale each voxel per run to mean=100 (percent signal change units). "
        "Values are clipped to max 200 (100%% increase from mean).",
    )
    proc_opts.add_argument(
        "-do_blur",
        type=float,
        metavar="FWHM",
        default=None,
        help="Apply 3D Gaussian spatial smoothing with FWHM in mm. "
        "Smoothing is applied BEFORE masking to avoid edge effects. "
        "Typical values: 4-8 mm.",
    )
    proc_opts.add_argument(
        "-cache",
        metavar="FILE.h5",
        help="HDF5 cache file for fast data loading. If exists, loads from cache. If not, creates cache from input files.",
    )
    proc_opts.add_argument(
        "-test",
        type=int,
        metavar="N_VOXELS",
        help="Test mode: extract ~N voxels from center of volume (fast iteration for debugging)",
    )
    proc_opts.add_argument(
        "-batch_size",
        type=int,
        help="Number of voxels per batch for ARMA grid search (default: auto-detect). OLS will use 4x this value.",
    )
    proc_opts.add_argument(
        "-force_format",
        choices=["nii", "nii.gz", "afni"],
        help="Force output format (default: match input)",
    )
    proc_opts.add_argument(
        "-device",
        type=str,
        help=(
            "Force device (default: auto-detect GPU). "
            "Format: 'cpu' or 'cuda' for auto-config, "
            "'cpu,N' to use N CPU threads, "
            "'cuda,N' to use GPU device N (e.g., 'cuda,0' for GPU 0)"
        ),
    )
    add_verbose_arg(proc_opts, default=0)
    proc_opts.add_argument(
        "-legacy_contrasts",
        action="store_true",
        help="Use legacy loop-based GLT contrast computation (slower, for validation only)",
    )
    proc_opts.add_argument(
        "-debug_memory",
        action="store_true",
        help="Print detailed memory profiling at every step (for debugging)",
    )

    # Voxel-wise regressors (ANATICOR-style)
    vox_opts = parser.add_argument_group("Voxel-wise Regressors (ANATICOR)")
    vox_opts.add_argument(
        "-dsort",
        action="append",
        nargs="+",
        metavar="DSET",
        default=None,
        help=(
            "Voxel-wise baseline regressor(s) (like AFNI 3dREMLfit -dsort), a 4D "
            "dataset giving a different baseline for every voxel. Pass ONE OR MORE "
            "files after -dsort; they are concatenated in time to the (concatenated) "
            "input length. The regressor is fit PER RUN (block-diagonal, one column "
            "per run, like polynomials/motion). Repeatable: each -dsort is a separate "
            "block-diagonal set. The ARMA (a,b) is estimated WITHOUT dsort, then the "
            "final GLS is redone per voxel WITH it. See -dsort_betas to output betas."
        ),
    )
    vox_opts.add_argument(
        "-dsort_betas",
        "-dsort-betas",
        dest="dsort_betas",
        choices=("yes", "no"),
        default="no",
        help=(
            "Write the per-run -dsort coefficients (and t-stats) as trailing "
            "sub-bricks in -Rbuck / -Rbeta. Default 'no' (dsort is fit as a nuisance "
            "but its betas are not reported)."
        ),
    )
    vox_opts.add_argument(
        "-dsort_nods",
        "-dsort-nods",
        dest="dsort_nods",
        action="store_true",
        help=(
            "Also write the no-dsort results (base design only) to parallel outputs "
            "with '_nods' appended to each prefix, for comparison."
        ),
    )
    vox_opts.add_argument(
        "-slibase",
        action="append",
        metavar="FILE.1D",
        default=None,
        help=(
            "Slicewise baseline regressors (AFNI 3dREMLfit -slibase), e.g. physiological "
            "noise. The .1D file has n_slices*m columns in slice-MINOR (cyclic) order: "
            "bb[0]→slice0, bb[1]→slice1, ..., bb[n_slices]→slice0. Each slice gets its "
            "own [design | slice_block] design, folded into the REML (a,b) estimate. "
            "Columns are nuisance (surface via -Rnuisance). Repeatable. Use -slibase_sm "
            "if your columns are in slice-MAJOR order."
        ),
    )
    vox_opts.add_argument(
        "-slibase_sm",
        "-slibase-sm",
        dest="slibase_sm",
        action="append",
        metavar="FILE.1D",
        default=None,
        help=(
            "Like -slibase but columns are in slice-MAJOR order (all of slice 0's "
            "regressors first, then slice 1's, ...). Repeatable."
        ),
    )

    # Onset-based design (alternative to -matrix)
    onset_group = parser.add_argument_group("Onset-Based Design (alternative to -matrix)")
    onset_group.add_argument(
        "-onsets",
        nargs="+",
        help="Onset timing files (AFNI format). One per condition. Mutually exclusive with -matrix and -events.",
    )
    onset_group.add_argument(
        "-durations",
        nargs="+",
        help="Stimulus durations. Single value, per-condition, or 'value,count'. Required with -onsets.",
    )
    onset_group.add_argument(
        "-events",
        nargs="+",
        metavar="TSV",
        help=(
            "BIDS events TSV files, one per run (e.g. sub-01_task-loc_run-01_events.tsv). "
            "Mutually exclusive with -matrix and -onsets. Files are sorted by run number "
            "automatically (handles both run-1 and run-01 zero-padding). "
            "The TSV must contain 'onset', 'duration', and 'trial_type' columns "
            "(override with -event_cols). "
            "Conditions are derived from unique trial_type values (sorted). "
            "Durations are derived from the TSV data."
        ),
    )
    onset_group.add_argument(
        "-event_ignore",
        nargs="+",
        metavar="CONDITION",
        help=(
            "Condition names (trial_type values) to exclude from modeling when using -events. "
            "E.g. -event_ignore fixation null rest"
        ),
    )
    onset_group.add_argument(
        "-event_cols",
        nargs=3,
        metavar=("ONSET_COL", "DURATION_COL", "TRIAL_TYPE_COL"),
        help=(
            "Custom column names for -events TSV files, replacing the BIDS defaults. "
            "Provide exactly 3 names: the columns mapped to onset, duration, and trial_type. "
            "E.g. -event_cols onset_time duration_s condition_name"
        ),
    )
    onset_group.add_argument(
        "-round_onsets",
        nargs="?",
        const=0.7,
        type=float,
        metavar="THRESHOLD",
        help=(
            "Snap all onset times to the nearest TR boundary. "
            "THRESHOLD (default: 0.7) is the fractional position within a TR above which "
            "an onset rounds up; below it rounds down. "
            "0.5 = standard nearest-TR rounding."
        ),
    )
    onset_group.add_argument(
        "-round_durations",
        type=int,
        metavar="PLACES",
        help=(
            "Round stimulus durations to PLACES decimal places before uniquing "
            "and design matrix construction (0=integer, 1=tenth, etc.)."
        ),
    )
    onset_group.add_argument(
        "-microtime_dt",
        type=float,
        default=0.1,
        help="Microtime resolution (seconds, default: 0.1)",
    )
    onset_group.add_argument(
        "-tr",
        "-TR",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Override TR from input files (seconds). Default: read from the header.",
    )
    onset_group.add_argument(
        "-polort",
        type=int,
        default=None,
        help="Polynomial order (default: auto based on run length)",
    )
    add_ortvec_arguments(onset_group)
    onset_group.add_argument(
        "-canonical",
        type=str,
        default=None,
        help="(DEPRECATED) Use -hrf_model instead. Canonical HRF: 'spmg1' or 'glmsingle'",
    )
    onset_group.add_argument(
        "-hrf_model",
        type=str,
        default="spmg1",
        help="HRF model: 'spmg1' (default), 'spmg2', 'spmg3', 'glmsingle', 'FIR', 'TENT', or 'TENT(bot,top,n)'. "
        "SPMG2 = canonical + temporal derivative. SPMG3 = canonical + time + dispersion derivatives. "
        "FIR/TENT use durations to set window (default: 0 to max(durations)). "
        "Example: 'TENT(0,15,6)' for 6 tent basis functions from 0-15s",
    )
    onset_group.add_argument(
        "-hrfopt_prefix",
        "-hrf-opt",
        "-hrf_opt",
        dest="hrfopt_prefix",
        type=str,
        default=None,
        help=(
            "Per-voxel HRF mode. PREFIX from ffs_hrfopt — loads {PREFIX}_hrf_index.nii.gz "
            "and {PREFIX}_hrf_library.pt. Each voxel is fit with its assigned HRF. "
            "Requires -events or -onsets; mutually exclusive with -matrix."
        ),
    )

    return parser


def _voxels_to_4d_volume(data_2d, volume_shape, voxel_mask) -> np.ndarray:
    """Reshape a ``(n_voxels, n_timepoints)`` array to a 4D ``(nx, ny, nz, T)`` volume.

    ``results.residuals`` / ``results.predicted`` are stored ``(n_voxels, n_time)``
    (see fit_glm_arma11). With a ``voxel_mask`` the fitted rows are scattered onto
    the full grid; without one (whole-brain fit) every voxel is present and the rows
    reshape directly. ``volume_shape`` is ``(nx, ny, nz)`` matching the C-order voxel
    flattening used by the fitter. Shared by -Rfitts / -Rerrts / -Oerrts so the
    no-mask case can't fall through to a 2D array (which nibabel rejects).
    """
    arr = np.ascontiguousarray(data_2d)
    n_time = arr.shape[1]
    vol = np.zeros((*volume_shape, n_time), dtype=np.float32)
    if voxel_mask is not None:
        vol[np.asarray(voxel_mask).reshape(volume_shape)] = arr
    else:
        vol[...] = arr.reshape(*volume_shape, n_time)
    return vol


def _derive_rvar_path(rbuck_path: str) -> str:
    """Append ``_ffsremlvar`` to the Rbuck stem, preserving its multi-suffix
    extension (``.nii.gz`` / ``.nii.zst`` / ``.nii``). Mirrors AFNI's
    ``<prefix>_REMLvar+orig`` convention but with the FFS-specific suffix so
    AFNI's REMLvar file in the same directory never gets clobbered.

    Examples:
        ``stats.nii.gz``         → ``stats_ffsremlvar.nii.gz``
        ``stats_bucket.nii.zst`` → ``stats_bucket_ffsremlvar.nii.zst``
        ``stats.nii``            → ``stats_ffsremlvar.nii``
    """
    p = Path(rbuck_path)
    name = p.name
    for suffix in (".nii.gz", ".nii.zst", ".nii"):
        if name.endswith(suffix):
            return str(p.with_name(name[: -len(suffix)] + "_ffsremlvar" + suffix))
    # Unknown extension: drop the single suffix and re-add it.
    return str(p.with_name(p.stem + "_ffsremlvar" + p.suffix))


def _insert_path_suffix(path: str, suffix: str) -> str:
    """Insert ``suffix`` before the (possibly multi-part) extension of ``path``.

    Used for ``-dsort_nods`` parallel outputs: ``stats.nii.gz`` → ``stats_nods.nii.gz``.
    """
    p = Path(path)
    name = p.name
    for ext in (".nii.gz", ".nii.zst", ".nii"):
        if name.endswith(ext):
            return str(p.with_name(name[: -len(ext)] + suffix + ext))
    return str(p.with_name(p.stem + suffix + p.suffix))


def print_header(args):
    """Print program header"""
    from datetime import datetime

    print("=" * 70)
    print("ffs_reml - GPU-Accelerated ARMA(1,1) GLM")
    print("=" * 70)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    if args.use_double:
        print("⚙️  Precision: DOUBLE (float64) - matches AFNI exactly")
    else:
        print("⚙️  Precision: SINGLE (float32) - default, faster")
    print()


def _report_loaded_design(design_info: dict, source: str) -> None:
    """Echo a run-structure / column summary for a pre-built design matrix.

    Mirrors the reporting the -events/-onsets path prints while building the
    design, so a user who passed -matrix or -spec still sees run count, polort,
    stim classes, nuisance width and contrasts before the GLM runs. Purely
    cosmetic — the numbers come straight out of the already-parsed xmat.
    """
    matrix = design_info.get("matrix")
    n_rows = matrix.shape[0] if matrix is not None else design_info.get("n_timepoints")
    n_cols = matrix.shape[1] if matrix is not None else design_info.get("n_regressors")
    run_starts = design_info.get("run_starts") or [0]
    groups = design_info.get("column_groups")
    stim_labels = design_info.get("stim_labels") or []
    glt_labels = design_info.get("glt_labels") or []

    print()
    print("Design matrix summary (loaded, not rebuilt):")
    print(f"  Source: {source}")
    if design_info.get("tr") is not None:
        print(f"  TR: {design_info['tr']:.3f}s")
    print(f"  Runs: {len(run_starts)}")
    print(f"  Total timepoints: {n_rows}")
    print(f"  Run starts (TRs): {run_starts}")
    print(f"  Regressors: {n_cols}")

    # AFNI ColumnGroups convention: -1 = polort drift, 0 = nuisance, ≥1 = stim.
    if groups:
        n_polort = sum(1 for g in groups if g == -1)
        n_stim_cols = sum(1 for g in groups if g >= 1)
        n_nuis = sum(1 for g in groups if g == 0)
        n_runs = len(run_starts)
        polort_note = ""
        if n_runs and n_polort % n_runs == 0:
            polort_note = f" (order {n_polort // n_runs - 1} per run × {n_runs} runs)"
        print(f"  Polort drift columns: {n_polort}{polort_note}")
        print(f"  Stim columns: {n_stim_cols}")
        print(f"  Other nuisance columns (ortvec/motion/etc): {n_nuis}")
    if stim_labels:
        print(f"  Stim classes ({len(stim_labels)}): {list(stim_labels)}")
    if glt_labels:
        print(f"  Contrasts ({len(glt_labels)}): {list(glt_labels)}")


def print_output_summary(args):
    """Print summary of requested outputs"""
    print("=" * 70)
    print("📋 Requested Outputs")
    print("=" * 70)

    # ARMA/REML outputs
    arma_outputs = []
    if args.Rbuck:
        arma_outputs.append(f"  • Rbuck (betas + stats): {args.Rbuck}")
    if args.Rbeta:
        arma_outputs.append(f"  • Rbeta (betas only): {args.Rbeta}")
    if args.Rnuisance:
        arma_outputs.append(f"  • Rnuisance (nuisance betas + stats): {args.Rnuisance}")
    if args.Rvar:
        arma_outputs.append(f"  • Rvar (ARMA parameters): {args.Rvar}")
    if args.Rfitts:
        arma_outputs.append(f"  • Rfitts (fitted model): {args.Rfitts}")
    if args.Rerrts:
        arma_outputs.append(f"  • Rerrts (residuals): {args.Rerrts}")
    if args.Rwherr:
        arma_outputs.append(f"  • Rwherr (whitened residuals): {args.Rwherr}")
    if args.save_clean:
        arma_outputs.append(f"  • save_clean (nuisance-removed timeseries): {args.save_clean}")
    if args.save_nuisance:
        arma_outputs.append(f"  • save_nuisance (nuisance-only timeseries): {args.save_nuisance}")
    if args.save_taskfit:
        arma_outputs.append(f"  • save_taskfit (task-only fit): {args.save_taskfit}")
    if getattr(args, "Rlklhd", None):
        arma_outputs.append(f"  • Rlklhd (full likelihood surface): {args.Rlklhd}")

    if arma_outputs:
        print("ARMA/REML Outputs:")
        for output in arma_outputs:
            print(output)
    else:
        print("ARMA/REML Outputs: None")

    print()

    # OLS outputs
    ols_outputs = []
    if args.Obuck:
        ols_outputs.append(f"  • Obuck (betas + stats): {args.Obuck}")
    if args.Obeta:
        ols_outputs.append(f"  • Obeta (betas only): {args.Obeta}")
    if args.Onuisance:
        ols_outputs.append(f"  • Onuisance (nuisance betas + stats): {args.Onuisance}")

    if ols_outputs:
        print("OLS Outputs:")
        for output in ols_outputs:
            print(output)
    else:
        print("OLS Outputs: None")

    print()

    # Special outputs
    special_outputs = []
    if args.single_trials:
        label = args.single_trials
        special_outputs.append(
            f"  • Single-trial mode (one regressor per event, chronological output): "
            f"ols_{label}_single.nii.gz, reml_{label}_single.nii.gz"
        )

    if special_outputs:
        print("Special Outputs:")
        for output in special_outputs:
            print(output)
        print()

    # Statistics flags
    stat_flags = []
    if args.fout:
        stat_flags.append("F-statistics")
    if args.tout:
        stat_flags.append("t-statistics")
    if args.rout:
        stat_flags.append("R² statistics")

    if stat_flags:
        print(f"Statistics: {', '.join(stat_flags)}")
    else:
        print("Statistics: Default (F-statistics only)")

    print("=" * 70)
    print()


def main():
    # Pre-extract -a_grid / -b_grid before argparse sees them.
    # argparse cannot handle values like "-0.9,0.9,65" (starts with '-' but not a
    # bare number), so we pull those out manually before parse_args().
    raw_argv = sys.argv[1:]

    # No-args shows help (argparse handles -h/--help itself)
    if not raw_argv:
        parser = create_parser()
        parser.print_help()
        sys.exit(0)

    filtered_argv, a_grid_str, b_grid_str = _extract_grid_args(raw_argv)

    parser = create_parser()
    # Temporarily replace sys.argv so parse_args() sees the filtered list
    _orig_argv = sys.argv
    sys.argv = [sys.argv[0]] + filtered_argv
    try:
        args = parser.parse_args()
    finally:
        sys.argv = _orig_argv

    # Show help if no arguments provided (argparse handles -h/-help itself)
    if not filtered_argv:
        parser.print_help()
        sys.exit(0)

    # -afni_mode: switch the REML noise model to AFNI-faithful behaviour for the
    # whole run (banded R, AFNI ltop, corcut grid filter). Default is FFS-accurate.
    if getattr(args, "afni_mode", False):
        from fastfuncstuff.glm.arma import set_afni_mode

        set_afni_mode(True)

    # Validate design input: exactly one of -matrix, -onsets, -events, -spec.
    _design_sources = [
        bool(args.matrix),
        bool(args.onsets),
        bool(args.events),
        bool(args.spec),
    ]
    if sum(_design_sources) > 1:
        print("ERROR: Specify only one of -matrix, -onsets, -events, or -spec")
        sys.exit(1)
    if not any(_design_sources):
        print("ERROR: Must specify one of -matrix, -onsets, -events, or -spec")
        sys.exit(1)

    # ── -spec handling: compile to xmat *before* loading any data ────────
    # We do this early so a syntax error or missing event file fails fast
    # instead of after the slow input load. The compiled xmat becomes the
    # input for the rest of the pipeline (treated as -matrix from here on).
    if args.spec:
        from fastfuncstuff.cli.design_spec import (
            _do_compile as _design_spec_compile,
        )
        from fastfuncstuff.cli.design_spec import (
            _resolve_spec_path as _design_spec_resolve,
        )

        spec_path = _design_spec_resolve(args.spec)
        if args.xmat:
            compiled_xmat = Path(args.xmat)
        else:
            # Default: <specname>.xmat.1D next to the spec. Avoids clobbering
            # AFNI's conventional X.xmat.1D in the same directory.
            stem = spec_path.stem
            compiled_xmat = spec_path.with_name(f"{stem}.xmat.1D")

        print(f"📐 Compiling spec {spec_path} → {compiled_xmat}", flush=True)
        compile_args = argparse.Namespace(
            spec=str(spec_path),
            xmat=str(compiled_xmat),
            verb=0,  # quiet — our header above + the REML banner are enough
            overwrite=args.overwrite,
        )
        rc = _design_spec_compile(compile_args)
        if rc != 0:
            sys.exit(rc)
        args.matrix = str(compiled_xmat)
        args.spec = None  # downstream branches key off args.matrix from here

    # Auto-write Rvar alongside Rbuck so future ffs_concalc has the per-voxel
    # ARMA (a, b) it needs to rebuild (X̃ᵀX̃)⁻¹ on demand. AFNI does the same
    # with its `_REMLvar+orig` companion. Skip when the user explicitly set
    # -Rvar (they already chose a name).
    if args.Rbuck and not args.Rvar:
        args.Rvar = _derive_rvar_path(args.Rbuck)
        print(
            f"📝 Auto-saving Rvar for concalc: {args.Rvar}\n"
            "   (sub-bricks 0 and 1 are ARMA `a` and `b`; "
            "pass -Rvar to override the path)",
            flush=True,
        )

    # -event_ignore / -event_cols are only meaningful with -events
    if args.event_ignore and not args.events:
        print("ERROR: -event_ignore requires -events")
        sys.exit(1)
    if args.event_cols and not args.events:
        print("ERROR: -event_cols requires -events")
        sys.exit(1)

    # Per-voxel HRF needs the design built from onsets/events — a pre-built
    # AFNI .xmat.1D file fixed the HRF at build time, so per-voxel HRFs are
    # mathematically impossible from that input.
    if args.hrfopt_prefix and not (args.events or args.onsets):
        print(
            "ERROR: -hrfopt_prefix requires -events or -onsets/-durations "
            "(design must be built from onsets, not a pre-built -matrix)."
        )
        sys.exit(1)

    # Single-trial design needs onset times; -matrix lacks per-event structure.
    if args.single_trials and not (args.events or args.onsets):
        print(
            "ERROR: -single_trials requires -events or -onsets/-durations "
            "(per-event regressors must be built from onsets, not a pre-built -matrix)."
        )
        sys.exit(1)

    # Check that at least one output is requested
    outputs = [
        args.Rvar,
        args.Rbuck,
        args.Rbeta,
        args.Rnuisance,
        args.Rfitts,
        args.Rerrts,
        args.Rwherr,
        args.save_clean,
        args.save_nuisance,
        args.save_taskfit,
        args.Obuck,
        args.Obeta,
        args.Onuisance,
        getattr(args, "Rlklhd", None),
    ]
    if args.save_per_run_polort and not args.save_clean:
        print("⚠️  -save_per_run_polort has no effect without -save_clean; ignoring.")

    if not any(outputs):
        print("ERROR: At least one output option must be specified")
        print("       Use -Rbuck, -Rbeta, -Rnuisance, -Rvar, -Rfitts, -Rerrts, -Rwherr,")
        print("       -Obuck, -Obeta, or -Onuisance")
        sys.exit(1)

    print_header(args)

    # Parse input files
    input_files = parse_input_files(args.input)
    print(f"📁 Input files: {len(input_files)} file(s)")
    for f in input_files:
        print(f"   • {f}")
    print()

    # Fail fast on a mask/data grid mismatch. The full check lives in
    # analyze_from_design_matrix, but by then we've paid for a full volume load
    # (minutes for large 4D inputs). Headers are cheap — nibabel reads shape
    # without touching the data — so compare grids up front.
    if args.mask:
        data_shape = nifti_shape(input_files[0])[:3]
        mask_shape = nifti_shape(args.mask)[:3]
        if mask_shape != data_shape:
            raise SystemExit(
                f"❌ Mask/data grid mismatch: mask '{args.mask}' is {mask_shape} "
                f"but input '{input_files[0]}' is {data_shape}. "
                "Resample the mask onto the data grid (or pick the right mask) "
                "before running."
            )

    # Get TR: an explicit -tr overrides whatever is in the header (headers get
    # mangled by upstream tools, so being explicit at the modeling stage is safer).
    if args.tr is not None:
        tr = args.tr
        print(f"⏱️  TR: {tr:.3f} seconds (specified)")
    else:
        tr = get_tr_from_file(input_files[0])
        args.tr = tr
        print(f"⏱️  TR: {tr:.3f} seconds (from header)")
    print()

    # Detect input format (for informational purposes only)
    # NOTE: All outputs are written as NIfTI .nii.gz regardless of input format
    if args.force_format:
        output_format = args.force_format  # Keep var name for compatibility
    else:
        output_format = detect_format(input_files[0])
    print(f"📥 Input format detected: {output_format}")
    print("📤 Output format: NIfTI (.nii.gz) - all outputs written as compressed NIfTI")
    print()

    # Print summary of requested outputs
    print_output_summary(args)

    # Parse device specification using shared utility
    import os

    device, cpu_threads_override, cuda_device_id = parse_device_arg(args.device)

    # MPS does not support float64. ARMA(1,1) requires float64 for numerical
    # stability (see CLAUDE.md §8), so fall back to CPU automatically.
    if device.type == "mps" and args.use_double:
        import warnings

        warnings.warn(
            "MPS does not support float64. -use_double requires CPU; "
            "switching to device=cpu. Use -device cpu to suppress this warning.",
            stacklevel=1,
        )
        device = torch.device("cpu")

    configure_torch_backends(device)

    # Configure CPU threading for maximum performance
    if device.type == "cpu":
        try:
            import psutil

            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = os.cpu_count() or 12

            # Determine number of threads to use
            if cpu_threads_override is not None:
                # User explicitly specified thread count
                num_threads = cpu_threads_override
                thread_source = "user-specified"
            else:
                # Auto-detect: use physical cores for compute efficiency
                num_threads = physical_cores or logical_cores
                thread_source = f"physical cores ({logical_cores} logical with hyperthreading)"

            torch.set_num_threads(num_threads)
            # set_num_interop_threads is one-shot and must run before any
            # parallel work has started; under repeated entry points (tests,
            # subprocess relaunches, ipython reloads) it may already be set,
            # so don't fail the run if torch refuses.
            try:
                torch.set_num_interop_threads(num_threads)
            except RuntimeError:
                pass
            # Also set environment variables for MKL/OpenMP
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
            os.environ["MKL_NUM_THREADS"] = str(num_threads)

            print(f"🖥️  Device: {device}")
            print(f"⚡ CPU threads: {num_threads} ({thread_source})")
        except ImportError:
            # Fallback if psutil not available
            num_threads = (
                cpu_threads_override if cpu_threads_override is not None else (os.cpu_count() or 12)
            )
            torch.set_num_threads(num_threads)
            # One-shot and must precede any parallel work; configure_torch_backends
            # above may already have set it, so don't fail the run if torch refuses.
            try:
                torch.set_num_interop_threads(num_threads)
            except RuntimeError:
                pass
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
            os.environ["MKL_NUM_THREADS"] = str(num_threads)
            print(f"🖥️  Device: {device}")
            print(f"⚡ CPU threads: {num_threads}")
    else:
        print(f"🖥️  Device: {device}")
    print()

    # Parse ARMA grids if provided (values pre-extracted from argv by _extract_grid_args)
    a_grid = None
    b_grid = None
    if a_grid_str:
        a_grid = parse_grid_arg(a_grid_str).to(device)
        print(
            f"🔢 Custom a_grid: {a_grid.numel()} points [{a_grid[0].item():.4g}, {a_grid[-1].item():.4g}]"
        )
    elif args.a_grid:
        # Fallback: value didn't need pre-extraction (no leading '-')
        a_grid = parse_grid_arg(args.a_grid).to(device)
        print(
            f"🔢 Custom a_grid: {a_grid.numel()} points [{a_grid[0].item():.4g}, {a_grid[-1].item():.4g}]"
        )
    if b_grid_str:
        b_grid = parse_grid_arg(b_grid_str).to(device)
        print(
            f"🔢 Custom b_grid: {b_grid.numel()} points [{b_grid[0].item():.4g}, {b_grid[-1].item():.4g}]"
        )
    elif args.b_grid:
        b_grid = parse_grid_arg(args.b_grid).to(device)
        print(
            f"🔢 Custom b_grid: {b_grid.numel()} points [{b_grid[0].item():.4g}, {b_grid[-1].item():.4g}]"
        )
    if a_grid_str or b_grid_str or args.a_grid or args.b_grid:
        print()

    # Print batch size if specified
    if args.batch_size:
        print(f"📦 Batch size: {args.batch_size:,} voxels per batch")
        print()

    # Determine requested output families
    want_ols = (
        args.Obuck is not None
        or args.Obeta is not None
        or args.Onuisance is not None
        or args.Oerrts is not None
        or args.Ofitts is not None
    )
    want_reml = any(
        x is not None
        for x in [
            args.Rvar,
            args.Rbuck,
            args.Rbeta,
            args.Rnuisance,
            args.Rfitts,
            args.Rerrts,
            args.Rwherr,
            args.save_clean,
            args.save_nuisance,
            args.save_taskfit,
        ]
    )
    ols_output_path = args.Obuck if args.Obuck else None

    # Load or build design matrix
    from fastfuncstuff.io.afni import read_afni_design_matrix

    if args.matrix:
        # Load design matrix from file
        design_info = read_afni_design_matrix(args.matrix)
        # Echo a summary so -matrix / -spec runs confirm run structure, polort
        # and nuisance width just like the onset-building path does.
        _report_loaded_design(design_info, source=str(args.matrix))
    else:
        # Build design matrix from onsets
        print()
        print("=" * 70)
        print("Building design matrix from onsets")
        print("=" * 70)
        print()

        if args.events:
            # ── BIDS events TSV path ─────────────────────────────────────────
            from fastfuncstuff.design.bids_events import parse_bids_events, sort_bids_event_files

            # Validate: one TSV per input run, OR a single shared TSV to
            # broadcast across all runs (identical timing every run).
            if len(args.events) not in (1, len(input_files)):
                print(
                    f"ERROR: -events requires one TSV per run or a single shared TSV: "
                    f"got {len(args.events)} events files but {len(input_files)} input datasets."
                )
                sys.exit(1)

            # Custom column mapping
            event_cols = tuple(args.event_cols) if args.event_cols else None

            if len(args.events) == 1 and len(input_files) > 1:
                print(
                    f"Parsing BIDS events file (broadcasting 1 events file across "
                    f"{len(input_files)} runs)..."
                )
            else:
                print("Parsing BIDS events files...")
            # Show the sorted order so the user can confirm alignment
            sorted_event_paths = sort_bids_event_files(args.events)
            for ep in sorted_event_paths:
                print(f"  {ep}")

            try:
                all_onsets, durations, condition_labels = parse_bids_events(
                    event_files=args.events,
                    event_ignore=args.event_ignore,
                    event_cols=event_cols,
                    round_durations=args.round_durations,
                    n_runs=len(input_files),
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)

            n_conditions = len(condition_labels)
            n_runs = len(input_files)

            print()
            for cidx, label in enumerate(condition_labels):
                n_events = sum(len(all_onsets[cidx][r]) for r in range(n_runs))
                print(
                    f"  {label}: {n_events} events across {n_runs} runs  (duration={durations[cidx]:.3f}s)"
                )

            if len(set(durations)) == 1:
                print(f"\n  Single duration for all conditions: {durations[0]:.3f}s")
            else:
                print(
                    f"\n  Per-condition durations: {dict(zip(condition_labels, durations, strict=False))}"
                )

        else:
            # ── AFNI timing files path ───────────────────────────────────────
            print("Parsing onset files...")
            all_onsets = []
            condition_labels = []
            for onset_file in args.onsets:
                condition_label = Path(onset_file).stem
                runs_onsets = parse_afni_timing_file(onset_file)
                all_onsets.append(runs_onsets)
                condition_labels.append(condition_label)
                n_events = sum(len(run_onsets) for run_onsets in runs_onsets)
                print(f"  {condition_label}: {n_events} events across {len(runs_onsets)} runs")

            n_conditions = len(all_onsets)
            n_runs = len(all_onsets[0])  # All conditions must have same number of runs

            # Parse durations
            print()
            print("Parsing durations...")
            durations = parse_durations(args.durations, n_conditions, condition_labels)
            if len(args.durations) == 1 and "," not in args.durations[0]:
                print(f"  Using {durations[0]}s for all {n_conditions} conditions")
            else:
                print(f"  Matched {len(durations)} durations to {n_conditions} conditions")

        # ── Optional onset / duration rounding ──────────────────────────
        if args.round_onsets is not None:
            from fastfuncstuff.design.builder import round_onsets

            all_onsets = round_onsets(all_onsets, tr, threshold=args.round_onsets)
            print(f"\nOnsets rounded to TR boundaries (threshold={args.round_onsets:.2f})")

        # For BIDS, round_durations was already applied per-event inside parse_bids_events
        if args.round_durations is not None and not args.events:
            dp = args.round_durations
            durations = [round(d, dp) for d in durations]
            print(f"Durations rounded to {dp} decimal place(s): {durations}")

        # Parse HRF model arguments
        print()
        from fastfuncstuff.cli_utils import parse_hrf_model_args

        hrf_info = parse_hrf_model_args(
            hrf_model_arg=args.hrf_model,
            canonical_arg=args.canonical,
            durations=durations,
            condition_labels=condition_labels,
            tr=args.tr,
        )

        hrf_model_name = hrf_info["hrf_model_name"]
        is_fir_model = hrf_info["is_fir_model"]
        fir_bot = hrf_info["fir_bot"]
        fir_top = hrf_info["fir_top"]
        n_basis = hrf_info["n_basis"]
        condition_labels_full = hrf_info["condition_labels_full"]

        # ffs_reml doesn't support -hrf_opt or -single_trial, so no compatibility check needed
        # (FIR is compatible with ARMA modeling)

        # Compute run starts from input files
        print()
        print("Computing run structure...")
        run_starts = [0]
        total_tps = 0
        for f in input_files:
            img_shape = nifti_shape(f)
            run_len = img_shape[3] if len(img_shape) > 3 else img_shape[0]
            total_tps += run_len
            if f != input_files[-1]:  # Don't add start for after last run
                run_starts.append(total_tps)
        n_timepoints = total_tps

        print(f"  Runs: {n_runs}")
        print(f"  Total timepoints: {n_timepoints}")
        print(f"  Run starts (TRs): {run_starts}")

        # Auto-determine polort if not specified
        if args.polort is None:
            run_lengths = compute_run_lengths(run_starts, n_timepoints)
            avg_run_duration_sec = get_average_run_duration(run_lengths, tr)
            polort = auto_polort(avg_run_duration_sec, formula="afni")
            print(f"  Auto polort: {polort} (based on {avg_run_duration_sec / 60:.1f} min avg run)")
        else:
            polort = args.polort
            print(f"  Polort: {polort}")

        # Build microtime onset matrix
        print()
        print("Building onset matrix at microtime resolution...")
        onset_matrix_micro = create_onset_matrix_microtime(
            all_onsets, run_starts, tr, n_timepoints, args.microtime_dt, durations, device
        )
        print(f"  Onset matrix shape: {onset_matrix_micro.shape}")

        # Per-voxel HRF: load assignments + library if -hrfopt_prefix set
        hrf_library_obj = None
        hrf_indices_obj = None
        if args.hrfopt_prefix:
            from fastfuncstuff.glm.ridge import load_hrf_indices

            hrf_index_file = f"{args.hrfopt_prefix}_hrf_index.nii.gz"
            hrf_lib_file = f"{args.hrfopt_prefix}_hrf_library.pt"
            if not Path(hrf_index_file).exists():
                print(f"ERROR: HRF index file not found: {hrf_index_file}")
                print(f"  Expected ffs_hrfopt output with prefix: {args.hrfopt_prefix}")
                sys.exit(1)
            if not Path(hrf_lib_file).exists():
                print(f"ERROR: HRF library file not found: {hrf_lib_file}")
                sys.exit(1)

            print()
            print(f"Loading per-voxel HRF assignments from {args.hrfopt_prefix}...")
            # Volume-aligned (no mask) — analyze_from_design_matrix masks alongside data
            hrf_indices_obj = load_hrf_indices(hrf_index_file, mask=None)
            from fastfuncstuff.cli_utils import spinner

            with spinner(f"Loading {Path(hrf_lib_file).name}"):
                hrf_lib_data = torch.load(hrf_lib_file, weights_only=False)
            hrf_library_obj = hrf_lib_data["hrf_library"].to(device)
            unique_hrfs, counts = torch.unique(hrf_indices_obj, return_counts=True)
            print(f"  HRF library shape: {tuple(hrf_library_obj.shape)}")
            print(
                f"  {len(unique_hrfs)} unique HRFs across {hrf_indices_obj.numel():,} volume voxels"
            )

        # Build task design matrix using refactored function
        print()
        from fastfuncstuff.cli_utils import build_task_design_from_args

        task_design, designs_by_hrf = build_task_design_from_args(
            hrf_model_name=hrf_model_name,
            is_fir_model=is_fir_model,
            fir_bot=fir_bot,
            fir_top=fir_top,
            n_basis=n_basis,
            all_onsets=all_onsets,
            onset_matrix_micro=onset_matrix_micro,
            n_conditions=n_conditions,
            n_timepoints=n_timepoints,
            run_starts=run_starts,
            tr=args.tr,
            microtime_dt=args.microtime_dt,
            device=device,
            hrf_opt=args.hrfopt_prefix,
            hrf_library=hrf_library_obj,
            hrf_indices=hrf_indices_obj,
            n_voxels=hrf_indices_obj.numel() if hrf_indices_obj is not None else None,
        )

        per_voxel_hrf_mode = designs_by_hrf is not None
        if per_voxel_hrf_mode:
            # In per-HRF mode all designs share column count — use first as the
            # canonical "task_design" for shape/label bookkeeping below.
            rep_hrf_idx = next(iter(designs_by_hrf.keys()))
            task_design = designs_by_hrf[rep_hrf_idx]
            print(
                f"  Per-voxel HRF: {len(designs_by_hrf)} task designs, "
                f"each {tuple(task_design.shape)}"
            )
        else:
            assert task_design is not None
            print(f"  Task design shape: {task_design.shape}")

        # --- Single-trial design rewrite ---
        # `-single_trials LABEL` converts the condition-based task design into
        # one regressor per event (GLMsingle-style). Trials are sorted by
        # absolute onset time, so output betas read chronologically.
        is_single_trial = False
        trial_cond_ids = None
        trial_run_ids = None
        if args.single_trials:
            if is_fir_model:
                print("ERROR: -single_trials is incompatible with FIR/TENT models.")
                sys.exit(1)
            if per_voxel_hrf_mode and (n_basis or 1) > 1:
                print(
                    "ERROR: -single_trials with -hrfopt_prefix requires n_basis=1 "
                    "(SPMG1 / glmsingle). SPM derivatives (SPMG2/3) cannot be combined "
                    "with per-voxel HRF libraries."
                )
                sys.exit(1)

            from fastfuncstuff.glm.ridge import create_single_trial_design

            print()
            print(
                f"🎯 Single-trial mode (label: {args.single_trials}) — "
                "rebuilding design with one regressor per event"
                + (" × per-voxel HRF" if per_voxel_hrf_mode else "")
            )

            st_design_out, trial_labels, trial_cond_ids, trial_run_ids, _cond_design = (
                create_single_trial_design(
                    onsets_by_condition=all_onsets,
                    durations=durations,
                    run_starts=run_starts,
                    tr=args.tr,
                    n_timepoints=n_timepoints,
                    microtime_dt=args.microtime_dt,
                    condition_labels=condition_labels,
                    device=device,
                    hrf_model_name=hrf_model_name,
                    n_basis=n_basis if n_basis else 1,
                    hrf_library=hrf_library_obj if per_voxel_hrf_mode else None,
                    hrf_index_per_voxel=hrf_indices_obj if per_voxel_hrf_mode else None,
                )
            )

            if per_voxel_hrf_mode:
                # create_single_trial_design returned (n_hrfs, n_t, n_trials).
                # Replace the condition-based designs_by_hrf with per-HRF single-trial.
                assert st_design_out.dim() == 3, (
                    f"Expected 3D stacked design, got {st_design_out.shape}"
                )
                designs_by_hrf = {int(i): st_design_out[i] for i in range(st_design_out.shape[0])}
                task_design = designs_by_hrf[next(iter(designs_by_hrf.keys()))]
                print(
                    f"  Single-trial × HRF designs: {len(designs_by_hrf)} HRFs, "
                    f"each {tuple(task_design.shape)}"
                )
            else:
                task_design = st_design_out
                print(
                    f"  Single-trial design: {tuple(task_design.shape)} "
                    f"({len(trial_labels)} columns)"
                )

            # Override label/count bookkeeping so downstream design_info uses
            # per-trial columns (each trial is its own "condition" for output).
            condition_labels_full = trial_labels
            condition_labels = trial_labels
            n_conditions = len(trial_labels)
            is_single_trial = True

        # Build nuisance design (polynomials + ortvec)
        print()
        print("Building nuisance regressors...")
        nuisance_blocks = collect_nuisance_blocks(
            args,
            run_starts,
            n_timepoints,
            verbose=True,
        )
        nuisance_design = build_nuisance_block_diag(
            run_starts=run_starts,
            n_timepoints=n_timepoints,
            polort=polort,
            device=device,
            blocks=nuisance_blocks,
            verbose=True,
        )

        # Build full design: [task | nuisance]
        full_design = torch.cat([task_design, nuisance_design], dim=1)
        # Per-voxel HRF: build one full design per HRF (same nuisance, different task).
        full_designs_by_hrf = None
        if per_voxel_hrf_mode:
            full_designs_by_hrf = {
                int(h): torch.cat([designs_by_hrf[h], nuisance_design], dim=1)
                for h in designs_by_hrf
            }
        print()
        print(f"Full design matrix shape: {full_design.shape}")
        print(f"  Task columns: {task_design.shape[1]}")
        print(f"  Nuisance columns: {nuisance_design.shape[1]}")
        if per_voxel_hrf_mode:
            print(f"  Per-HRF designs: {len(full_designs_by_hrf)}")

        # Create design_info dict to match format from read_afni_design_matrix
        task_indices = list(range(task_design.shape[1]))
        nuisance_indices = list(range(task_design.shape[1], full_design.shape[1]))

        # Build column labels
        column_labels = []
        if is_fir_model:
            # FIR: condition_labels_full already has expanded labels (e.g., "cond1_t0.0s", "cond1_t1.5s", ...)
            # Add #0 suffix to match AFNI format
            for cond_label in condition_labels_full:
                column_labels.append(f"{cond_label}#0")
        else:
            # Canonical: condition_labels_full = condition_labels (no expansion)
            for cond_label in condition_labels_full:
                column_labels.append(f"{cond_label}#0")

        # Add nuisance labels
        _nuisance_label_offset = len(column_labels)
        for run_idx in range(n_runs):
            for p in range(polort + 1):
                column_labels.append(f"Run#{run_idx + 1}Pol#{p}")

        # Nuisance block labels (one entry per column; widest column count
        # per block when per-run widths differ).
        for block in nuisance_blocks:
            column_labels.extend(block.get_column_names())

        # Build stim_bots and stim_tops
        if is_fir_model:
            # FIR: each condition has n_basis columns
            stim_bots = []
            stim_tops = []
            for cond_idx in range(n_conditions):
                bot = cond_idx * n_basis
                top = (cond_idx + 1) * n_basis - 1
                stim_bots.append(bot)
                stim_tops.append(top)
        else:
            # Canonical: each condition has 1 column
            stim_bots = list(range(n_conditions))
            stim_tops = list(range(n_conditions))

        full_design_np = full_design.cpu().numpy()
        design_info = {
            "matrix": full_design_np,
            "design_matrix": full_design_np,
            "tr": args.tr,
            "n_timepoints": full_design_np.shape[0],
            "n_regressors": full_design_np.shape[1],
            "column_labels": column_labels,
            "stim_indices": task_indices,
            "nuisance_indices": nuisance_indices,
            "stim_bots": stim_bots,
            "stim_tops": stim_tops,
            "stim_labels": condition_labels,  # Use original (unexpanded) labels
            "run_starts": run_starts,
            "n_runs": n_runs,
        }

        print()
        print("=" * 70)
        print()

    # ========== SINGLE-TRIAL BETA-SPACE CV PATH ==========
    if args.beta_cv:
        from fastfuncstuff.glm.outputs import save_single_trial_results
        from fastfuncstuff.glm.ridge import create_single_trial_design
        from fastfuncstuff.glm.xval import compute_xval_r2_single_trials, generate_cv_splits

        print()
        print("=" * 70)
        print("Using beta-space cross-validation with ARMA(1,1) prewhitening")
        print("=" * 70)
        print()

        # Validate that we're in onset-based mode
        if not args.onsets:
            print("ERROR: -beta_cv requires -onsets (not -matrix)")
            sys.exit(1)

        # 1. Build single-trial design
        print("Building single-trial design...")
        st_design, trial_labels, trial_cond_ids, trial_run_ids, cond_design = (
            create_single_trial_design(
                onsets_by_condition=all_onsets,
                durations=durations,
                run_starts=run_starts,
                tr=args.tr,
                n_timepoints=n_timepoints,
                microtime_dt=args.microtime_dt,
                condition_labels=condition_labels,
                device=device,
            )
        )
        print(f"  Single-trial design: {st_design.shape}")
        n_trials = st_design.shape[1]

        # 2. Build wide design: [single_trial | nuisance]
        # Note: design_info was already created with condition-level design
        # We need to rebuild it with single-trial design
        print("Building nuisance regressors for single-trial design...")
        nuisance_blocks_st = collect_nuisance_blocks(
            args,
            run_starts,
            n_timepoints,
            verbose=False,
        )
        nuisance_design_st = build_nuisance_block_diag(
            run_starts=run_starts,
            n_timepoints=n_timepoints,
            polort=polort,
            device=device,
            blocks=nuisance_blocks_st,
            verbose=True,
        )

        full_design_st = torch.cat([st_design, nuisance_design_st], dim=1)
        task_indices_st = list(range(n_trials))
        nuisance_indices_st = list(range(n_trials, full_design_st.shape[1]))

        print(f"  Full design (wide): {full_design_st.shape}")
        print(f"    Task columns (single-trial): {n_trials}")
        print(f"    Nuisance columns: {len(nuisance_indices_st)}")

        # 3. Run ARMA(1,1) analysis on wide design
        print()
        print("Running ARMA(1,1) analysis on single-trial design...")

        # Build nuisance labels: per-run poly + per-block column names.
        nuisance_labels_st: list[str] = []
        for run_idx in range(n_runs):
            for p in range(polort + 1):
                nuisance_labels_st.append(f"r{run_idx + 1:02d}_poly{p}")
        for block in nuisance_blocks_st:
            nuisance_labels_st.extend(block.get_column_names())
        # Pad if rounding/widening mismatch (shouldn't happen, but be safe).
        while len(nuisance_labels_st) < len(nuisance_indices_st):
            nuisance_labels_st.append(f"nuisance_{len(nuisance_labels_st)}")
        nuisance_labels_st = nuisance_labels_st[: len(nuisance_indices_st)]

        # Create a temporary design_info for single-trial mode
        design_info_st = {
            "design_matrix": full_design_st.cpu().numpy(),
            "column_labels": trial_labels + nuisance_labels_st,
            "stim_indices": task_indices_st,
            "nuisance_indices": nuisance_indices_st,
            "run_starts": run_starts,
            "n_runs": n_runs,
        }

        # Load and prepare fMRI data
        # Note: Single-trial mode doesn't use preprocessing caching, just pass input files
        fmri_data_st = input_files

        # Call analyze_from_design_matrix with single-trial design
        # This will fit ARMA(1,1), prewhiten, and return single-trial betas
        results_st, _ = analyze_from_design_matrix(
            fmri_data=fmri_data_st,
            design_matrix=design_info_st["design_matrix"],
            stim_column_indices=task_indices_st,
            nuisance_column_indices=nuisance_indices_st,
            method="arma11",
            arma_a_grid=a_grid,
            arma_b_grid=b_grid,
            device=device,
            mask_file=args.mask,
            voxel_chunk_size=args.batch_size,
            use_double=args.use_double,
            verbose=args.verb >= 1,
        )

        # 4. Extract single-trial betas
        st_betas = results_st.betas  # (n_voxels, n_trials)
        print(f"  Single-trial betas: {st_betas.shape}")

        # 5. Beta-space CV
        print()
        print("Computing beta-space cross-validated R²...")
        cv_splits = generate_cv_splits(n_runs, strategy=1)  # LORO
        xval = compute_xval_r2_single_trials(
            st_betas,
            trial_cond_ids,
            trial_run_ids,
            cv_splits,
            metric="cod",
            device=device,
            verbose=True,
        )

        # 6. Save outputs
        print()
        print("Saving single-trial outputs...")

        # Get spatial metadata
        volume_shape = results_st.original_shape
        affine = results_st.affine
        voxel_mask = getattr(results_st, "voxel_mask", None)

        output_files = save_single_trial_results(
            betas=st_betas,
            xval_r2=xval["r2"],
            trial_labels=trial_labels,
            trial_condition_ids=trial_cond_ids,
            trial_run_ids=trial_run_ids,
            condition_labels=condition_labels,
            output_prefix=args.prefix,
            volume_shape=volume_shape,
            affine=affine,
            voxel_mask=voxel_mask,
        )

        print()
        print("=" * 70)
        print("✅ ffs_reml (single-trial mode) Complete!")
        print("=" * 70)
        print(f"  Median beta-space R²: {xval['r2'].median():.4f}")
        print(f"  {xval['n_test_trials_total']} test trials across {xval['n_splits']} folds")
        print()
        for key, path in output_files.items():
            print(f"  {key}: {path}")
        print("=" * 70)

        # Early exit - don't run standard pipeline
        return

    # Setup OLS write callback if any OLS output is requested
    _ols_write_callback = None
    if want_ols:
        # Determine stat flags (default to -fout if none specified)
        _want_fstat = args.fout or (not args.tout and not args.rout)
        _want_tstat = args.tout
        _want_rstat = args.rout
        # Capture single_trials flag and design_info for callback
        want_single_trials = args.single_trials
        callback_design_info = design_info  # Capture in closure

        def write_ols_results(ols_results, original_shape, affine):
            """Write OLS results immediately after computation"""
            print("\n💾 Writing OLS outputs (before ARMA)...")

            # OLS residuals / fitted series (full-model), if requested. Written
            # here while ols_results is still in memory (it's cleared afterward).
            if args.Oerrts or args.Ofitts:
                from fastfuncstuff.glm.outputs import (
                    _ensure_numpy,
                    _get_voxel_mask,
                    _resolve_shape,
                )

                _ols_vm = _get_voxel_mask(ols_results)
                _ols_shape = _resolve_shape(ols_results, original_shape)
                _ols_aff = affine if affine is not None else np.eye(4)
                if args.Oerrts and getattr(ols_results, "residuals", None) is not None:
                    print(f"  • Writing OLS residuals: {args.Oerrts}")
                    save_nifti(
                        _voxels_to_4d_volume(
                            _ensure_numpy(ols_results.residuals), _ols_shape, _ols_vm
                        ),
                        output_path=replace_afni_extension(args.Oerrts, ".nii.gz"),
                        affine=_ols_aff,
                    )
                if args.Ofitts and getattr(ols_results, "predicted", None) is not None:
                    print(f"  • Writing OLS fitted model: {args.Ofitts}")
                    save_nifti(
                        _voxels_to_4d_volume(
                            _ensure_numpy(ols_results.predicted), _ols_shape, _ols_vm
                        ),
                        output_path=replace_afni_extension(args.Ofitts, ".nii.gz"),
                        affine=_ols_aff,
                    )

            # IMPORTANT: When task_indices is passed to fit_glm(), the OLS results
            # already contain ONLY the task regressors (stimulus columns).
            # Extract stimulus labels for proper labeling
            stim_bots: list = callback_design_info.get("stim_bots", [])
            stim_tops: list = callback_design_info.get("stim_tops", [])
            stim_indices = []
            if stim_bots and stim_tops:
                for bot, top in zip(stim_bots, stim_tops, strict=False):
                    stim_indices.extend(range(bot, top + 1))

            # Extract labels for stimulus columns only (not all 322 columns!)
            if stim_indices and "column_labels" in callback_design_info:
                stim_labels = [callback_design_info["column_labels"][i] for i in stim_indices]
            else:
                stim_labels = callback_design_info.get("column_labels")

            # Set spatial metadata on OLS results for writing
            ols_results.original_shape = original_shape
            ols_results.affine = affine

            # Voxel-wise (-dsort) coefficients: appended as trailing beta/t-stat
            # sub-bricks only on -dsort_betas yes (default: dsort is a nuisance whose
            # per-run betas aren't reported). Mirrors the REML -Rbuck path.
            _want_dsort_betas = getattr(args, "dsort_betas", "no") == "yes"
            _ols_dsort_betas = getattr(ols_results, "dsort_betas", None)

            @contextlib.contextmanager
            def _spliced_ols_dsort(base_labels):
                if _ols_dsort_betas is None or not _want_dsort_betas:
                    yield list(base_labels) if base_labels is not None else base_labels
                    return
                d_labels = getattr(ols_results, "dsort_labels", None) or [
                    f"dsort#{k}" for k in range(_ols_dsort_betas.shape[1])
                ]
                saved_b, saved_t = ols_results.betas, ols_results.tstats
                try:
                    ols_results.betas = torch.cat([saved_b, _ols_dsort_betas], dim=1)
                    if (
                        saved_t is not None
                        and getattr(ols_results, "dsort_tstats", None) is not None
                    ):
                        ols_results.tstats = torch.cat([saved_t, ols_results.dsort_tstats], dim=1)
                    yield (list(base_labels) if base_labels is not None else []) + list(d_labels)
                finally:
                    ols_results.betas, ols_results.tstats = saved_b, saved_t

            if args.Obuck:
                print(f"  • Writing OLS betas + stats (bucket): {args.Obuck}")
                # Always write NIfTI .nii.gz regardless of input format
                # Results already contain only stimulus columns (252), use stim_labels
                contrast_names = getattr(ols_results, "contrast_labels", None)

                # Build contrast_results dict if we have contrasts
                ols_contrast_results = None
                if (
                    hasattr(ols_results, "contrast_betas")
                    and ols_results.contrast_betas is not None
                ):
                    ols_contrast_results = {
                        "contrast_betas": ols_results.contrast_betas,
                        "contrast_tstats": ols_results.contrast_tstats,
                    }
                    # Add partial R² if available and requested
                    if (
                        hasattr(ols_results, "contrast_r2_partial")
                        and ols_results.contrast_r2_partial is not None
                    ):
                        ols_contrast_results["contrast_r2_partial"] = (
                            ols_results.contrast_r2_partial
                        )
                    # Add semi-partial R² if available and requested
                    if (
                        hasattr(ols_results, "contrast_r2_semipartial")
                        and ols_results.contrast_r2_semipartial is not None
                    ):
                        ols_contrast_results["contrast_r2_semipartial"] = (
                            ols_results.contrast_r2_semipartial
                        )

                with _spliced_ols_dsort(stim_labels) as _obuck_labels:
                    write_glm_bucket_as_nifti(
                        ols_results,
                        args.Obuck,
                        condition_names=_obuck_labels,  # stim labels (+ dsort on -dsort_betas yes)
                        contrast_names=contrast_names,
                        contrast_results=ols_contrast_results,
                        add_fdr=args.add_fdr,
                    )

            if args.Obeta:
                print(f"  • Writing OLS betas only: {args.Obeta}")
                # Write only betas using the write_glm_results_nifti function correctly
                # Create a temporary results-like object with only betas

                from fastfuncstuff.glm.outputs import (
                    _ensure_numpy,
                    _get_voxel_mask,
                    _reshape_parameter_map,
                    _resolve_shape,
                )

                affine = getattr(ols_results, "affine", np.eye(4))
                volume_shape = _resolve_shape(ols_results, None)
                voxel_mask = _get_voxel_mask(ols_results)

                # Append dsort coefficients last only on -dsort_betas yes.
                _obeta_betas = ols_results.betas
                if _want_dsort_betas and _ols_dsort_betas is not None:
                    _obeta_betas = torch.cat([_obeta_betas, _ols_dsort_betas], dim=1)
                betas_np = _ensure_numpy(_obeta_betas)
                betas_vol = _reshape_parameter_map(betas_np, volume_shape, voxel_mask)

                # Always write NIfTI .nii.gz regardless of input format
                from fastfuncstuff.cli_utils import spinner

                _obeta_path = replace_afni_extension(args.Obeta, ".nii.gz")
                with spinner(f"Writing {Path(_obeta_path).name}"):
                    save_nifti(
                        betas_vol,
                        output_path=_obeta_path,
                        affine=affine,
                    )

            if args.Onuisance:
                # NOTE: When task_indices is provided, OLS results contain only stimulus columns.
                # There are no nuisance columns in the OLS results to write out.
                # Nuisance parameters are in the full design matrix but not fitted separately.
                if stim_indices:
                    print(
                        "  ⚠️  Skipping -Onuisance: OLS fit only includes stimulus columns (not nuisance)"
                    )
                    print(
                        "      To get nuisance parameters, fit the full model without StimBots/StimTops filtering"
                    )
                else:
                    # No filtering - all regressors are present
                    print(f"  • Writing OLS nuisance betas + stats: {args.Onuisance}")
                    write_glm_bucket_as_nifti(
                        ols_results,
                        args.Onuisance,
                        condition_names=stim_labels,
                        add_fdr=args.add_fdr,
                    )

            if want_single_trials:
                if stim_indices:
                    ols_single_path = f"ols_{want_single_trials}_single.nii.gz"
                    print(f"  • Writing OLS single-trial betas (onset order): {ols_single_path}")
                    if "matrix" in callback_design_info:
                        write_single_trials_output(
                            ols_results,
                            ols_single_path,
                            callback_design_info["matrix"],
                            stim_indices,
                            stim_labels,
                        )
                    else:
                        print(
                            "      ⚠️  Warning: Design matrix not available, cannot determine onset times"
                        )
                else:
                    print(
                        "  ⚠️  Skipping single-trial output: No stimulus columns found (StimBots/StimTops)"
                    )

            # Write partial R² if requested and available
            if (
                args.rpartial
                and hasattr(ols_results, "r2_partial")
                and ols_results.r2_partial is not None
            ):
                # Generate output path by inserting _partialR2 before extension
                if args.Obuck:
                    if args.Obuck.endswith(".nii.gz"):
                        partial_r2_path = args.Obuck.replace(".nii.gz", "_partialR2.nii.gz")
                    elif args.Obuck.endswith(".nii"):
                        partial_r2_path = args.Obuck.replace(".nii", "_partialR2.nii.gz")
                    else:
                        partial_r2_path = args.Obuck + "_partialR2.nii.gz"
                else:
                    partial_r2_path = "OLS_partialR2.nii.gz"
                print(f"  • Writing OLS partial R² per condition: {partial_r2_path}")

                from fastfuncstuff.glm.outputs import (
                    _get_voxel_mask,
                    _resolve_shape,
                    write_partial_r2_with_labels,
                )

                # Get metadata for AFNI stat params
                n_timepoints_ols = callback_design_info.get("n_timepoints")
                n_regressors_ols = callback_design_info.get("n_regressors")

                # Get mode from args (captured in closure)
                r2_mode = args.rpartial if args.rpartial else "full"

                write_partial_r2_with_labels(
                    ols_results.r2_partial,
                    partial_r2_path,
                    condition_labels=stim_labels,
                    volume_shape=_resolve_shape(ols_results, None),
                    voxel_mask=_get_voxel_mask(ols_results),
                    affine=getattr(ols_results, "affine", None),
                    n_timepoints=n_timepoints_ols,
                    n_regressors=n_regressors_ols,
                    apply_afni_metadata=True,
                    mode=r2_mode,  # "full" or "task"
                )

                # Print labels for reference
                suffix = "_partialR2_task" if r2_mode == "task" else "_partialR2"
                print("     Sub-bricks (partial R² with AFNI stat params):")
                for idx, label in enumerate(stim_labels):
                    print(f"       [{idx}] {label}{suffix}")

            print()

        _ols_write_callback = write_ols_results

    # Set environment variable for partial R² mode (so analysis.py callback can access it)
    if args.rpartial:
        import os

        os.environ["FASTFUNCSIM_R2_PARTIAL_MODE"] = args.rpartial

    # Set environment variable for semi-partial R² mode (so analysis.py callback can access it)
    if args.r2semipartial:
        import os

        os.environ["FASTFUNCSIM_R2_SEMIPARTIAL_MODE"] = args.r2semipartial

    # ==========================================================================
    # Preprocessing: Blur and/or Scale if requested
    # ==========================================================================
    # Whole-dataset diagnostics need the manual load path (data resident in RAM).
    want_diag = bool(args.save_grandmean or args.save_tsnr or args.save_acf or args.save_mask)
    preprocessing_applied = args.do_blur is not None or args.do_scale or want_diag
    preproc_cached_metadata = None
    diag = None

    if preprocessing_applied:
        print()
        print("📦 Preprocessing data...")

        # Need to load data manually for preprocessing
        from tqdm import tqdm

        # Get header info from first file
        first_img = load_nifti(input_files[0])
        affine = first_img.affine
        nifti_header = first_img.header.copy()
        volume_shape = first_img.shape[:3]
        # Voxel sizes from pixdim (get_zooms), which is orientation-independent.
        # The affine diagonal is wrong for permuted/oblique grids (e.g. an RSP/LIA
        # FreeSurfer master): the size sits off-diagonal, so np.diag reads 0 mm and
        # divide-by-zeros the blur. Fall back to the affine column norms (never the
        # diagonal) if pixdim is unset.
        voxel_sizes = tuple(float(z) for z in nifti_header.get_zooms()[:3])
        if any(v == 0 for v in voxel_sizes):
            voxel_sizes = tuple(np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))

        # Preserve geometry/header metadata so ndarray-based analysis keeps
        # the same spatial orientation and voxel sizes as the original input.
        preproc_cached_metadata = {
            "affine": affine,
            "volume_shape": volume_shape,
            "nifti_header": nifti_header,
        }

        # Get run structure for preprocessing.
        # The -onsets path sets run_starts earlier (line ~1152). The -matrix
        # path skips that block, so we have to read RunStart out of the xmat
        # here. The reader stores it under the key "run_starts" — the older
        # "run_trs" lookup never worked and left run_starts unbound.
        if args.matrix is not None and not args.onsets:
            design_info_pre = read_afni_design_matrix(args.matrix)
            run_starts_from_xmat = design_info_pre.get("run_starts")
            if run_starts_from_xmat:
                run_starts = [int(rs) for rs in run_starts_from_xmat]
            else:
                # No RunStart header → single concatenated block.
                run_starts = [0]

        # Load each run straight into a single preallocated buffer. The old
        # path appended every run to a list and then np.concatenate'd it, which
        # keeps the full list AND the new concatenated array alive at once
        # (~2x peak). At 100-run / 160 GB scale that doubling OOM-kills the
        # host right after load (the concatenate memcpy is also what pins a
        # few cores). The design's n_timepoints is the authoritative total
        # length the analysis uses downstream, so allocate to it up front and
        # fill run-by-run — peak stays ~one extra run instead of a full copy.
        n_voxels = int(np.prod(volume_shape))
        total_tps = int(design_info["n_timepoints"])
        fmri_data_preprocessed = np.empty((n_voxels, total_tps), dtype=np.float32)

        if args.do_blur is not None:
            print(f"  Applying Gaussian blur (FWHM = {args.do_blur} mm)...")

        col = 0
        for run_idx, run_file in enumerate(tqdm(input_files, desc="  Loading runs", unit="run")):
            img = load_nifti(run_file)
            data_4d = img.get_fdata(dtype=np.float32)

            if data_4d.ndim != 4:
                raise ValueError(f"Expected 4D data, got shape {data_4d.shape}")

            # Apply blur if requested (on 4D data)
            if args.do_blur is not None:
                data_4d = gaussian_blur_3d(
                    data_4d,
                    fwhm_mm=args.do_blur,
                    voxel_sizes=voxel_sizes,
                    device=device,
                    verbose=(run_idx == 0),  # Only print details for first run
                )

            # Flatten to 2D (n_voxels, n_timepoints) and copy into the buffer,
            # freeing this run before the next load so we never hold two copies.
            n_tps = data_4d.shape[3]
            if col + n_tps > total_tps:
                raise ValueError(
                    f"Run data exceeds design length: loaded {col + n_tps} "
                    f"timepoints, design expects {total_tps}"
                )
            fmri_data_preprocessed[:, col : col + n_tps] = data_4d.reshape(n_voxels, n_tps)
            col += n_tps
            del img, data_4d

        if col != total_tps:
            raise ValueError(
                f"Loaded {col} timepoints but design expects {total_tps}; "
                "check that the input runs match the design matrix."
            )

        # Diagnostics hook 1: grand mean, computed on the un-scaled data.
        if want_diag:
            from fastfuncstuff.glm.reml_diagnostics import DatasetDiagnostics

            diag = DatasetDiagnostics(
                volume_shape=tuple(volume_shape),
                run_starts=[int(rs) for rs in run_starts],
                voxdims=tuple(float(v) for v in voxel_sizes),
                device=device,
            )
            diag.observe_raw(torch.from_numpy(fmri_data_preprocessed))

        # Apply scaling if requested (on concatenated 2D data)
        if args.do_scale:
            print("  Applying scaling (mean=100 per run)...")
            # Convert to torch for scale_to_percent_signal
            data_tensor = torch.from_numpy(fmri_data_preprocessed)
            # reml only consumes the violation counts in scale_info, so skip the
            # full (n_voxels, n_timepoints) mask -- it would add tens of GB on a
            # whole-dataset run.
            data_tensor, _violations_mask, scale_info = scale_to_percent_signal(
                data=data_tensor,
                run_starts=run_starts,
                max_scale=200.0,
                verbose=True,
                track_violations=False,
            )
            fmri_data_preprocessed = data_tensor.numpy()

            if scale_info["n_violations"] > 0:
                print(f"  ⚠️  {scale_info['n_violations']:,} ceiling violations")

            # Diagnostics hook 2: raw tSNR on the scaled data.
            if diag is not None:
                diag.observe_scaled(torch.from_numpy(fmri_data_preprocessed))
        elif want_diag and args.save_tsnr and diag is not None:
            # tSNR = mean/std is invariant to per-voxel scaling (the mean just
            # isn't ~100), so unscaled data gives the same tSNR -- don't block it
            # on -do_scale.
            diag.observe_scaled(torch.from_numpy(fmri_data_preprocessed))

        # Reshape back to 4D for analyze_from_design_matrix
        total_tps = fmri_data_preprocessed.shape[1]
        fmri_data_preprocessed = fmri_data_preprocessed.reshape(*volume_shape, total_tps)

        print(f"  ✓ Preprocessing complete: {volume_shape} × {total_tps} timepoints")
        print()

    print("🚀 Starting GLM analysis...")
    print()

    # Handle HDF5 caching for fast data loading
    fmri_data_to_use = None
    cache_metadata = None

    # If preprocessing was applied, use that data
    if preprocessing_applied:
        fmri_data_to_use = fmri_data_preprocessed
        cache_metadata = preproc_cached_metadata

    if args.cache and not preprocessing_applied:
        from fastfuncstuff.data_cache import check_cache_valid, load_cache

        cache_valid = check_cache_valid(args.cache, input_files)

        if cache_valid:
            # Load from cache
            cached_data, cache_metadata = load_cache(args.cache, input_files, validate=True)

            # Reshape to 4D if volume_shape available (needed for test mode and output writing)
            if "volume_shape" in cache_metadata:
                vol_shape = cache_metadata["volume_shape"]
                n_timepoints = cached_data.shape[1]
                # Reshape from (n_voxels, n_timepoints) to (x, y, z, n_timepoints)
                cached_data = cached_data.reshape(*vol_shape, n_timepoints)

            fmri_data_to_use = cached_data  # Pass numpy array instead of file list
        else:
            # Will create cache after loading data
            print(f"📝 Cache not found or invalid - will create: {args.cache}")

    # Run analysis
    try:
        # Determine analysis method from requested outputs
        analysis_method = "arma11" if want_reml else "ols"
        if analysis_method == "ols":
            print("ℹ️  No -R* outputs requested: running OLS only (skipping ARMA grid search)")

            # ARMA-specific options are ignored in OLS-only mode
            if any(
                [
                    a_grid_str or args.a_grid,
                    b_grid_str or args.b_grid,
                    args.grid_batching,
                    args.no_grid_batching,
                    args.quick_estimate,
                    args.load_Rvar,
                    args.Rvar,
                ]
            ):
                print("   Note: ARMA-specific options are ignored in OLS-only mode")

        # Determine grid batching strategy
        use_grid_batching = None  # Auto-detect by default
        if analysis_method == "arma11" and args.grid_batching:
            use_grid_batching = True
        elif analysis_method == "arma11" and args.no_grid_batching:
            use_grid_batching = False

        # Use cached data if available, otherwise load from files
        if fmri_data_to_use is None:
            fmri_data_to_use = input_files if len(input_files) > 1 else input_files[0]

        # Load precomputed ARMA params only if explicitly requested via -load_Rvar
        precomputed_arma = None
        if analysis_method == "arma11" and args.load_Rvar:
            # Try with automatic extension detection
            rvar_base = Path(args.load_Rvar)
            rvar_path = None
            for candidate in [
                rvar_base,
                Path(str(rvar_base) + ".nii.gz"),
                Path(str(rvar_base) + ".nii"),
            ]:
                if candidate.exists():
                    rvar_path = candidate
                    break

            if rvar_path is None:
                print(f"\n❌ ERROR: -load_Rvar file not found: {args.load_Rvar}")
                sys.exit(1)

            print(f"\n📂 Loading precomputed ARMA parameters from: {rvar_path}")
            print("   (Skipping grid search - saves ~80% compute time)")

            try:
                from fastfuncstuff.cli_utils import spinner

                with spinner(f"Loading {Path(rvar_path).name}"):
                    rvar_img = load_nifti(rvar_path)
                    rvar_data = rvar_img.get_fdata()  # (x, y, z[, 1], n_params)

                # AFNI bucket files sometimes store sub-briks in the 5th dimension
                # with a singleton 4th dimension (e.g. shape (x, y, z, 1, n)).
                # Squeeze out all size-1 dimensions that sit before the last axis.
                while rvar_data.ndim > 4 and rvar_data.shape[-2] == 1:
                    rvar_data = rvar_data[..., 0, :]  # drop singleton dim

                if rvar_data.ndim != 4:
                    raise ValueError(
                        f"Expected 4D Rvar file, got {rvar_data.ndim}D "
                        f"(shape {rvar_img.shape}). "
                        "Cannot reduce to (x, y, z, n_params)."
                    )
                if rvar_data.shape[3] < 2:
                    raise ValueError(
                        f"Rvar file must have at least 2 sub-briks (a, b), found {rvar_data.shape[3]}"
                    )

                precomputed_arma = rvar_data[..., :2]  # (x, y, z, 2) — only a and b needed
                n_voxels_total = np.prod(precomputed_arma.shape[:3])
                a_range = (precomputed_arma[..., 0].min(), precomputed_arma[..., 0].max())
                b_range = (precomputed_arma[..., 1].min(), precomputed_arma[..., 1].max())
                print(f"   ✓ Loaded ARMA params: {n_voxels_total:,} voxels × 2 params (a, b)")
                print(f"   • a range: [{a_range[0]:.3f}, {a_range[1]:.3f}]")
                print(f"   • b range: [{b_range[0]:.3f}, {b_range[1]:.3f}]")

            except Exception as e:
                print(f"   ⚠️  Failed to load Rvar file: {e}")
                print("   Proceeding with grid search instead")
                precomputed_arma = None

        # Warn if the Rvar output file already exists (user may want -load_Rvar instead)
        if analysis_method == "arma11" and args.Rvar and not args.load_Rvar:
            rvar_out_base = Path(args.Rvar)
            for candidate in [
                rvar_out_base,
                Path(str(rvar_out_base) + ".nii.gz"),
                Path(str(rvar_out_base) + ".nii"),
            ]:
                if candidate.exists():
                    print(f"\n⚠️  Note: {candidate} already exists and will be overwritten.")
                    print(
                        "   To reuse precomputed ARMA params instead, pass: -load_Rvar {candidate}"
                    )
                    break

        _per_hrf = locals().get("full_designs_by_hrf") is not None

        results, design_info = analyze_from_design_matrix(
            fmri_data=fmri_data_to_use,
            design_matrix_file=args.matrix,
            design_info=design_info if (args.onsets or args.events) else None,
            method=analysis_method,
            arma_a_grid=a_grid,
            arma_b_grid=b_grid,
            precomputed_arma_params=precomputed_arma,
            want_ols=(want_ols and analysis_method == "arma11"),
            want_ols_residuals=bool(args.Oerrts),
            want_ols_predicted=bool(args.Ofitts),
            ols_output_path=ols_output_path,
            ols_output_format=output_format,
            device=device,
            mask_file=args.mask,
            cache_file=args.cache if (args.cache and cache_metadata is None) else None,
            cached_metadata=cache_metadata,  # Pass cached header/affine/volume_shape
            test_n_voxels=args.test,
            voxel_chunk_size=args.batch_size,
            use_double=args.use_double,
            debug_memory=args.debug_memory,
            enable_quick_estimate=args.quick_estimate,
            force_exhaustive_search=args.exhaustive,
            use_grid_batching=use_grid_batching,
            want_r2_partial=bool(args.rpartial),  # True if flag is set (any mode)
            r2_partial_mode=args.rpartial if args.rpartial else "full",  # "full" or "task"
            want_r2_semipartial=bool(args.r2semipartial),  # True if flag is set (any mode)
            r2_semipartial_mode=args.r2semipartial
            if args.r2semipartial
            else "full",  # "full" or "task"
            legacy_contrasts=args.legacy_contrasts,
            save_profile_likelihoods=bool(getattr(args, "Rlklhd", None)) and not _per_hrf,
            designs_by_hrf=locals().get("full_designs_by_hrf"),
            hrf_indices=locals().get("hrf_indices_obj"),
            dsort_files=args.dsort,
            want_dsort_nods=args.dsort_nods,
            slibase_files=args.slibase,
            slibase_files_sm=args.slibase_sm,
            censor_file=args.censor,
            want_residuals=bool(args.Rerrts)
            or bool(args.save_clean)
            or (want_diag and bool(args.save_tsnr or args.save_acf)),
            want_ljung_box=bool(args.Rvar),
        )

        # In OLS-only mode, write requested OLS outputs here (ARMA path writes via callback).
        if analysis_method == "ols" and want_ols and _ols_write_callback is not None:
            _ols_write_callback(
                results,
                getattr(results, "original_shape", None),
                getattr(results, "affine", None),
            )
    except Exception as e:
        print(f"\n❌ ERROR during analysis: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("✅ Analysis complete!")
    print()

    # ── Whole-dataset diagnostics (grand mean / tSNR / FWHMx) ───────────────
    # Bonus maps from data that was resident for the fit; never fatal to the
    # main analysis (wrapped so a diagnostics bug can't lose the GLM output).
    if diag is not None:
        try:
            _diag_meta = preproc_cached_metadata or {}
            _diag_affine = _diag_meta.get("affine")
            _diag_header = _diag_meta.get("nifti_header")
            print("🧪 Writing whole-dataset diagnostics...")

            # Residual-derived maps: label by the model that ran (OLS residuals
            # in OLS mode, REML/GLS residuals under -reml). residuals are
            # (n_fit_voxels, n_time); place them onto the full grid, then extract
            # at the diagnostics mask (the -mask, else an automask of the grand
            # mean) for resid tSNR + per-run FWHMx.
            _resid = getattr(results, "residuals", None)
            _label = "ols" if analysis_method == "ols" else "reml"
            _want_resid_obs = _resid is not None and (args.save_tsnr or args.save_acf)
            if _want_resid_obs or args.save_mask:
                from fastfuncstuff.glm.reml_diagnostics import resolve_mask

                _gm = torch.from_numpy(diag.maps["grandmean"])
                _user_mask = None
                if args.mask:
                    from fastfuncstuff.cli_utils import spinner

                    with spinner(f"Loading {Path(args.mask).name}"):
                        _user_mask = torch.from_numpy(
                            (load_nifti(args.mask).get_fdata() > 0).astype("float32")
                        )
                # Run the automask on the GPU (device=device); without it the
                # 3dAutomask dilation/fill ran on the CPU grand mean (millions of
                # voxels) — a big CPU spike right before the GPU ACF.
                _dmask = resolve_mask(_user_mask, _gm, device=device)

                # -save_mask: persist the diagnostics automask (only when we made
                # one — a user-supplied -mask is already on disk). Route through
                # diag.save_map so it gets the same grid/header handling as the
                # grand-mean map.
                if args.save_mask and _user_mask is None:
                    mask_path = replace_afni_extension(args.save_mask, ".nii.gz")
                    diag.maps["automask"] = _dmask.cpu().numpy().astype(np.float32)
                    diag.save_map("automask", mask_path, _diag_affine, _diag_header)
                    print(f"  • diagnostics automask: {mask_path}")

                if _want_resid_obs:
                    _nvox = int(np.prod(diag.volume_shape))
                    _rt = (
                        _resid.detach().cpu().float()
                        if hasattr(_resid, "detach")
                        else torch.as_tensor(np.asarray(_resid), dtype=torch.float32)
                    )
                    _nt = _rt.shape[1]
                    _fit_vm = getattr(results, "voxel_mask", None)
                    if _fit_vm is not None:
                        _vmf = torch.as_tensor(np.asarray(_fit_vm)).reshape(-1).bool()
                        _full = torch.zeros(_nvox, _nt)
                        _full[_vmf] = _rt
                    else:
                        _full = _rt  # fit used all voxels
                    # _full is CPU (built from CPU residuals); _dmask lives on the
                    # compute device. Index with a CPU mask to avoid a CPU/CUDA
                    # index mismatch. observe_residuals re-homes _dmask itself.
                    _dmf = _dmask.reshape(-1).bool().cpu()
                    diag.observe_residuals(
                        {_label: _full[_dmf]},
                        _dmask,
                        want_tsnr=bool(args.save_tsnr),
                        want_fwhmx=bool(args.save_acf),
                    )

            if args.save_grandmean:
                diag.save_map(
                    "grandmean",
                    replace_afni_extension(args.save_grandmean, ".nii.gz"),
                    _diag_affine,
                    _diag_header,
                )
            if args.save_tsnr:
                diag.save_map(
                    "raw_tsnr", f"{args.save_tsnr}.raw_tsnr.nii.gz", _diag_affine, _diag_header
                )
                diag.save_map(
                    f"resid_tsnr_{_label}",
                    f"{args.save_tsnr}.resid_tsnr_{_label}.nii.gz",
                    _diag_affine,
                    _diag_header,
                )
            if args.save_acf:
                diag.save_table(f"fwhmx_{_label}", f"{args.save_acf}.fwhmx_{_label}.txt")
                diag.save_table(f"blur_est_{_label}", f"{args.save_acf}.blur_est_{_label}.1D")
        except Exception as _diag_err:  # diagnostics must never break the fit output
            import traceback as _tb

            print(f"  ⚠️  diagnostics failed (continuing): {_diag_err}")
            _tb.print_exc()
        print()

    # Write outputs
    print("💾 Writing outputs...")
    print()

    # Determine stat flags (default to -fout if none specified)
    _want_fstat = args.fout or (not args.tout and not args.rout and not args.rpartial)
    _want_tstat = args.tout
    _want_rstat = args.rout
    want_r2_partial = args.rpartial

    # Extract design metadata using helper function (clean naming!)
    from fastfuncstuff.io.afni import extract_design_metadata

    full_labels, stim_labels, stim_column_indices = extract_design_metadata(design_info)

    # Determine which columns were actually fitted (use metadata from results object)
    fitted_column_indices = getattr(results, "fitted_column_indices", None)

    # Extract labels matching what was actually fitted
    if fitted_column_indices is not None:
        # Results were filtered - use the labels that match fitted columns
        # (In most cases, fitted_column_indices == stim_column_indices)
        fitted_labels = [full_labels[i] for i in fitted_column_indices]
        stim_indices = fitted_column_indices  # For single-trials output
    else:
        # All columns were fitted
        fitted_labels = full_labels
        stim_indices = stim_column_indices if stim_column_indices else list(range(len(full_labels)))

    # Voxel-wise (-dsort) coefficients are appended as trailing beta/t-stat
    # sub-bricks (AFNI places them last). The bucket/beta writers read
    # results.betas/tstats directly, so we temporarily splice the dsort columns
    # in around each write and restore afterwards (keeps r2_partial etc. aligned
    # to the task columns).
    _want_dsort_betas = getattr(args, "dsort_betas", "no") == "yes"

    @contextlib.contextmanager
    def _spliced_dsort(res, base_labels):
        d_betas = getattr(res, "dsort_betas", None)
        # dsort is always fit as a nuisance; only report its betas on -dsort_betas yes.
        if d_betas is None or not _want_dsort_betas:
            yield list(base_labels)
            return
        d_labels = getattr(res, "dsort_labels", None) or [
            f"dsort#{k}" for k in range(d_betas.shape[1])
        ]
        saved_b, saved_t = res.betas, res.tstats
        try:
            res.betas = torch.cat([saved_b, d_betas], dim=1)
            if saved_t is not None and getattr(res, "dsort_tstats", None) is not None:
                res.tstats = torch.cat([saved_t, res.dsort_tstats], dim=1)
            yield list(base_labels) + list(d_labels)
        finally:
            res.betas, res.tstats = saved_b, saved_t

    # REML outputs
    if args.Rbuck:
        print(f"  • Writing REML betas + stats (bucket): {args.Rbuck}")
        # Rbuck: Betas + stats for fitted regressors + GLT contrasts
        # Use fitted_labels which match results.betas shape
        contrast_names = getattr(results, "contrast_labels", None)

        # Build contrast_results dict if we have contrasts
        contrast_results = None
        if hasattr(results, "contrast_betas") and results.contrast_betas is not None:
            contrast_results = {
                "contrast_betas": results.contrast_betas,
                "contrast_tstats": results.contrast_tstats,
            }
            # Add partial R² if available and requested
            if hasattr(results, "contrast_r2_partial") and results.contrast_r2_partial is not None:
                contrast_results["contrast_r2_partial"] = results.contrast_r2_partial
            # Add semi-partial R² if available and requested
            if (
                hasattr(results, "contrast_r2_semipartial")
                and results.contrast_r2_semipartial is not None
            ):
                contrast_results["contrast_r2_semipartial"] = results.contrast_r2_semipartial

        with _spliced_dsort(results, fitted_labels) as _rbuck_labels:
            write_glm_bucket_as_nifti(
                results,
                args.Rbuck,
                condition_names=_rbuck_labels,
                contrast_names=contrast_names,
                contrast_results=contrast_results,
                add_fdr=args.add_fdr,
            )

        # -dsort_nods: parallel no-dsort bucket for comparison.
        if getattr(results, "nods_results", None) is not None:
            nods_path = _insert_path_suffix(args.Rbuck, "_nods")
            print(f"  • Writing no-dsort REML bucket: {nods_path}")
            nods = results.nods_results
            nods_contrasts = None
            if getattr(nods, "contrast_betas", None) is not None:
                nods_contrasts = {
                    "contrast_betas": nods.contrast_betas,
                    "contrast_tstats": nods.contrast_tstats,
                }
            write_glm_bucket_as_nifti(
                nods,
                nods_path,
                condition_names=fitted_labels,
                contrast_names=getattr(nods, "contrast_labels", None),
                contrast_results=nods_contrasts,
                output_format="nifti_gz",
                add_fdr=args.add_fdr,
            )

    # Single-trial REML output: chronologically ordered per-trial betas.
    if args.single_trials and getattr(results, "betas", None) is not None:
        stim_idx = design_info.get("stim_indices") or []
        stim_lbls = (
            [design_info["column_labels"][i] for i in stim_idx]
            if "column_labels" in design_info and stim_idx
            else None
        )
        if stim_idx:
            reml_single_path = f"reml_{args.single_trials}_single.nii.gz"
            print(f"  • Writing REML single-trial betas (onset order): {reml_single_path}")
            write_single_trials_output(
                results,
                reml_single_path,
                design_info["matrix"],
                stim_idx,
                stim_lbls,
            )
        else:
            print("  ⚠️  Skipping REML single-trial output: no stimulus columns in design")

    if args.Rbeta:
        print(f"  • Writing REML betas only: {args.Rbeta}")
        # Rbeta: ALL betas, no stats
        from fastfuncstuff.glm.outputs import (
            _ensure_numpy,
            _get_voxel_mask,
            _reshape_parameter_map,
            _resolve_shape,
        )

        affine = getattr(results, "affine", np.eye(4))
        volume_shape = _resolve_shape(results, None)
        voxel_mask = _get_voxel_mask(results)

        assert results.betas is not None, "Results must have betas"
        # Append dsort coefficients last, only on -dsort_betas yes (else dsort is a
        # nuisance whose per-run betas aren't reported).
        _rbeta_betas = results.betas
        if _want_dsort_betas and getattr(results, "dsort_betas", None) is not None:
            _rbeta_betas = torch.cat([_rbeta_betas, results.dsort_betas], dim=1)
        betas_np = _ensure_numpy(_rbeta_betas)
        betas_vol = _reshape_parameter_map(betas_np, volume_shape, voxel_mask)

        # Always write NIfTI .nii.gz regardless of input format
        from fastfuncstuff.cli_utils import spinner

        _rbeta_path = replace_afni_extension(args.Rbeta, ".nii.gz")
        with spinner(f"Writing {Path(_rbeta_path).name}"):
            save_nifti(betas_vol, output_path=_rbeta_path, affine=affine)

        # -dsort_nods: parallel no-dsort betas.
        if getattr(results, "nods_results", None) is not None:
            nods_beta_path = _insert_path_suffix(args.Rbeta, "_nods")
            print(f"  • Writing no-dsort REML betas only: {nods_beta_path}")
            nods_betas_np = _ensure_numpy(results.nods_results.betas)
            nods_betas_vol = _reshape_parameter_map(nods_betas_np, volume_shape, voxel_mask)
            _nods_beta_path = replace_afni_extension(nods_beta_path, ".nii.gz")
            with spinner(f"Writing {Path(_nods_beta_path).name}"):
                save_nifti(
                    nods_betas_vol,
                    output_path=_nods_beta_path,
                    affine=affine,
                )

    if args.Rnuisance:
        print(f"  • Writing REML nuisance betas + stats: {args.Rnuisance}")
        # Rnuisance: Extract nuisance regressors (everything NOT in stimulus columns)
        # NOTE: This only works if full design was fitted (not filtered)
        if fitted_column_indices is not None:
            print(
                "  ⚠️  Skipping -Rnuisance: REML fit only includes stimulus columns (not nuisance)"
            )
            print(
                "      To get nuisance parameters, fit the full model without StimBots/StimTops filtering"
            )
        elif stim_indices:
            all_indices = list(range(len(full_labels)))
            nuisance_indices = [i for i in all_indices if i not in stim_indices]
            nuisance_results = slice_glm_results(results, nuisance_indices)
            nuisance_names = (
                [design_info["column_labels"][i] for i in nuisance_indices]
                if "column_labels" in design_info
                else None
            )
        else:
            # No stimulus indices specified, use all regressors
            nuisance_results = results
            nuisance_names = design_info.get("column_labels")

        # Always write NIfTI .nii.gz regardless of input format
        write_glm_bucket_as_nifti(
            nuisance_results,
            args.Rnuisance,
            condition_names=nuisance_names,
            add_fdr=args.add_fdr,
        )

    if args.Rvar:
        # Respect the extension the user gave (.nii/.nii.gz/.nii.zst); default
        # to compressed .nii.gz only when no NIfTI extension is present.
        rvar_output_path = Path(args.Rvar)
        if not str(rvar_output_path).endswith((".nii.gz", ".nii", ".nii.zst")):
            rvar_output_path = Path(str(rvar_output_path) + ".nii.gz")

        print(f"  • Writing REML variance parameters: {rvar_output_path}")
        # Stack variance parameters: a, b, lambda, StDev (all reliably computed)
        var_stack = []
        var_labels = []

        if results.arma_params is not None:
            var_stack.append(results.arma_params[:, 0])  # a
            var_stack.append(results.arma_params[:, 1])  # b
            var_labels.extend(["a", "b"])

        if results.arma_lambda is not None:
            var_stack.append(results.arma_lambda)  # lambda
            var_labels.append("lambda")

        if results.sigma2 is not None:
            var_stack.append(torch.sqrt(results.sigma2))  # StDev
            var_labels.append("StDev")

        # -LogLik: AFNI's Rvar[4] = the minimized REML criterion
        # (n-m)log(RSS_w) + logdet(R) + logdet(X'R^-1 X) — the same value (and
        # sign) we already track per voxel as reml_likelihood. Matches
        # 3dREMLfit's "-LogLik" subbrik (remla.c:1014).
        if getattr(results, "reml_likelihood", None) is not None:
            var_stack.append(results.reml_likelihood)
            var_labels.append("-LogLik")

        # -Rvar[5] = LjungBox: whiteness of the prewhitened residuals ("did the
        # ARMA(1,1) actually remove the autocorrelation?"). AFNI writes it for
        # every -Rvar (3dREMLfit.c:3780) — not only under -Rwherr — so it is
        # computed inside the GLS loop rather than from retained residuals.
        rvar_stataux = None
        if getattr(results, "ljung_box", None) is not None:
            var_stack.append(results.ljung_box)
            var_labels.append("LjungBox")
            if results.ljung_box_dof:
                from fastfuncstuff.io.afni import stat_type_to_stataux

                rvar_stataux = {
                    len(var_labels) - 1: stat_type_to_stataux(
                        "fict", (float(results.ljung_box_dof),)
                    )
                }

        # Stack and write
        var_data = torch.stack(var_stack, dim=1)
        # Write variance parameters directly as 4D NIfTI
        affine = getattr(results, "affine", np.eye(4))
        volume_shape = getattr(results, "original_shape", None)
        voxel_mask = getattr(results, "voxel_mask", None)

        # Reshape var_data to 4D volume (convert to numpy first!)
        var_data_np = var_data.cpu().numpy() if isinstance(var_data, torch.Tensor) else var_data

        if volume_shape is not None and voxel_mask is not None:
            n_params = var_data_np.shape[1]
            var_vol = np.zeros((*volume_shape, n_params), dtype=np.float32)
            voxel_mask_np = (
                voxel_mask.cpu().numpy() if isinstance(voxel_mask, torch.Tensor) else voxel_mask
            )
            var_vol[voxel_mask_np.reshape(volume_shape)] = var_data_np
        else:
            # Assume already in volume shape
            var_vol = var_data_np.reshape(*volume_shape, -1) if volume_shape else var_data_np

        # Inherit AFNI header from source data so SCENE_DATA[0] (view) and
        # TEMPLATE_SPACE carry forward correctly (e.g. TLRC + MNI_2009c_asym).
        # Then set SCENE_DATA[1] and TYPESTRING to fbuc/3DIM_HEAD_FUNC since
        # GLM outputs are stat buckets, not EPI timeseries.
        nifti_header_rvar = getattr(results, "nifti_header", None)
        var_header = None
        if nifti_header_rvar is not None:
            import copy

            var_header = copy.deepcopy(nifti_header_rvar)
            var_header.set_data_shape(var_vol.shape)
            var_header.set_data_dtype(
                var_vol.dtype
            )  # don't quantize float stats to an int source dtype
        from fastfuncstuff.io.afni import set_afni_func_type

        if var_header is not None:
            set_afni_func_type(var_header, func_code=11)  # fbuc / 3DIM_HEAD_FUNC

        # Sub-brick labels are written into the AFNI extension in-script by
        # save_nifti (no 3drefit round-trip); compression is chosen from the
        # output extension in the same single pass.
        from fastfuncstuff.cli_utils import spinner

        with spinner(f"Writing {rvar_output_path.name}"):
            save_nifti(
                var_vol,
                output_path=rvar_output_path,
                affine=affine,
                header=var_header,
                brick_labels=var_labels,
                brick_stataux=rvar_stataux,
            )
        print(f"    ✓ Labeled {len(var_labels)} sub-briks: {', '.join(var_labels)}")

    # Write full REML likelihood surface if requested (-Rlklhd)
    _rlklhd = getattr(args, "Rlklhd", None)
    _has_surface = (
        hasattr(results, "reml_lklhd_surface") and results.reml_lklhd_surface is not None  # type: ignore[union-attr]
    )
    if _rlklhd and _has_surface:
        import copy

        from fastfuncstuff.io.afni import set_afni_func_type

        print(f"  • Writing REML likelihood surface: {_rlklhd}")
        surface_np = results.reml_lklhd_surface.cpu().float().numpy()  # type: ignore[union-attr]
        surf_params = results.reml_surface_params  # type: ignore[union-attr]
        n_pairs = surface_np.shape[1]
        print(f"    {n_pairs} valid (a,b) grid points → {n_pairs} sub-briks")

        affine_lk = getattr(results, "affine", np.eye(4))
        volume_shape_lk = getattr(results, "original_shape", None)
        voxel_mask_lk = getattr(results, "voxel_mask", None)
        nifti_header_lk = getattr(results, "nifti_header", None)

        if volume_shape_lk is not None and voxel_mask_lk is not None:
            voxel_mask_np = (
                voxel_mask_lk.cpu().numpy() if hasattr(voxel_mask_lk, "cpu") else voxel_mask_lk
            )
            vol_4d = np.zeros((*volume_shape_lk, n_pairs), dtype=np.float32)
            vol_4d[voxel_mask_np.reshape(volume_shape_lk)] = surface_np
        elif volume_shape_lk is not None:
            vol_4d = surface_np.reshape(*volume_shape_lk, n_pairs)
        else:
            vol_4d = surface_np

        lklhd_header = None
        if nifti_header_lk is not None:
            lklhd_header = copy.deepcopy(nifti_header_lk)
            lklhd_header.set_data_shape(vol_4d.shape)
            # don't quantize float stats to an int source dtype
            lklhd_header.set_data_dtype(vol_4d.dtype)
            set_afni_func_type(lklhd_header, func_code=11)

        # Normalise output path, respecting the user's extension
        # (.nii/.nii.gz/.nii.zst); default to .nii.gz only when none is given.
        lklhd_out = _rlklhd
        if not lklhd_out.endswith((".nii.gz", ".nii", ".nii.zst")):
            lklhd_out = lklhd_out + ".nii.gz"
        lklhd_path = Path(lklhd_out)

        # Label each sub-brik with its (a, b) pair, in-script via save_nifti.
        lklhd_labels = [f"a={a_k:.2f}_b={b_k:.2f}" for (a_k, b_k) in surf_params]
        from fastfuncstuff.cli_utils import spinner

        with spinner(f"Writing {lklhd_path.name}"):
            save_nifti(
                vol_4d,
                output_path=lklhd_path,
                affine=affine_lk,
                header=lklhd_header,
                brick_labels=lklhd_labels,
            )
        print(f"    ✓ Labeled {n_pairs} sub-briks (a=X.XX_b=Y.YY format)")

    elif _rlklhd and not _has_surface:
        print(
            "  ⚠️  Likelihood surface requested but not available (OLS mode or precomputed ARMA params?)"
        )

    # Write partial R² if requested and available
    if want_r2_partial and hasattr(results, "r2_partial") and results.r2_partial is not None:
        # Generate output path by inserting _partialR2 before extension
        if args.Rbuck:
            if args.Rbuck.endswith(".nii.gz"):
                partial_r2_path = args.Rbuck.replace(".nii.gz", "_partialR2.nii.gz")
            elif args.Rbuck.endswith(".nii"):
                partial_r2_path = args.Rbuck.replace(".nii", "_partialR2.nii.gz")
            else:
                partial_r2_path = args.Rbuck + "_partialR2.nii.gz"
        else:
            partial_r2_path = "REML_partialR2.nii.gz"
        print(f"  • Writing REML partial R² per condition: {partial_r2_path}")

        from fastfuncstuff.glm.outputs import (
            _get_voxel_mask,
            _resolve_shape,
            write_partial_r2_with_labels,
        )

        # Get design info for stat parameters
        n_timepoints_reml = design_info.get("n_timepoints")
        n_regressors_reml = design_info.get("n_regressors")

        # Get mode
        r2_mode = args.rpartial if args.rpartial else "full"

        write_partial_r2_with_labels(
            results.r2_partial,
            partial_r2_path,
            condition_labels=fitted_labels,
            volume_shape=_resolve_shape(results, None),
            voxel_mask=_get_voxel_mask(results),
            affine=getattr(results, "affine", None),
            n_timepoints=n_timepoints_reml,
            n_regressors=n_regressors_reml,
            apply_afni_metadata=True,
            mode=r2_mode,  # "full" or "task"
        )

        suffix = "_partialR2_task" if r2_mode == "task" else "_partialR2"
        print("     Sub-bricks (partial R² with AFNI stat params):")
        for idx, label in enumerate(fitted_labels):
            print(f"       [{idx}] {label}{suffix}")

    # Write nuisance partial R² if available (always "full" mode for nuisance)
    if (
        want_r2_partial
        and hasattr(results, "r2_partial_nuisance")
        and results.r2_partial_nuisance is not None
    ):
        # Generate output path
        if args.Rbuck:
            if args.Rbuck.endswith(".nii.gz"):
                nuisance_r2_path = args.Rbuck.replace(".nii.gz", "_nuisance_partialR2.nii.gz")
            elif args.Rbuck.endswith(".nii"):
                nuisance_r2_path = args.Rbuck.replace(".nii", "_nuisance_partialR2.nii.gz")
            else:
                nuisance_r2_path = args.Rbuck + "_nuisance_partialR2.nii.gz"
        else:
            nuisance_r2_path = "REML_nuisance_partialR2.nii.gz"

        print(f"  • Writing REML nuisance partial R² per regressor: {nuisance_r2_path}")

        from fastfuncstuff.glm.outputs import (
            _get_voxel_mask,
            _resolve_shape,
            write_partial_r2_with_labels,
        )

        # Get nuisance labels from design_info
        nuisance_labels = design_info.get(
            "nuisance_labels",
            [f"nuisance{i}" for i in range(results.r2_partial_nuisance.shape[1])],
        )

        # Get design info for stat parameters
        n_timepoints_reml = design_info.get("n_timepoints")
        n_regressors_reml = design_info.get("n_regressors")

        write_partial_r2_with_labels(
            results.r2_partial_nuisance,
            nuisance_r2_path,
            condition_labels=nuisance_labels,
            volume_shape=_resolve_shape(results, None),
            voxel_mask=_get_voxel_mask(results),
            affine=getattr(results, "affine", None),
            n_timepoints=n_timepoints_reml,
            n_regressors=n_regressors_reml,
            apply_afni_metadata=True,
            mode="full",  # Always use "full" for nuisance (not rescaled)
        )

        print("     Sub-bricks (nuisance partial R² with AFNI stat params):")
        for idx, label in enumerate(nuisance_labels):
            print(f"       [{idx}] {label}_partialR2")

    # Write semi-partial R² if requested and available
    want_r2_semipartial = args.r2semipartial
    if (
        want_r2_semipartial
        and hasattr(results, "r2_semipartial")
        and results.r2_semipartial is not None
    ):
        # Generate output path by inserting _semipartialR2 before extension
        if args.Rbuck:
            if args.Rbuck.endswith(".nii.gz"):
                semipartial_r2_path = args.Rbuck.replace(".nii.gz", "_semipartialR2.nii.gz")
            elif args.Rbuck.endswith(".nii"):
                semipartial_r2_path = args.Rbuck.replace(".nii", "_semipartialR2.nii.gz")
            else:
                semipartial_r2_path = args.Rbuck + "_semipartialR2.nii.gz"
        else:
            semipartial_r2_path = "REML_semipartialR2.nii.gz"
        print(f"  • Writing REML semi-partial R² per condition: {semipartial_r2_path}")

        from fastfuncstuff.glm.outputs import (
            _get_voxel_mask,
            _resolve_shape,
            write_partial_r2_with_labels,
        )

        # Get design info for stat parameters
        n_timepoints_reml = design_info.get("n_timepoints")
        n_regressors_reml = design_info.get("n_regressors")

        # Get mode
        r2_semi_mode = args.r2semipartial if args.r2semipartial else "full"

        write_partial_r2_with_labels(
            results.r2_semipartial,
            semipartial_r2_path,
            condition_labels=fitted_labels,
            volume_shape=_resolve_shape(results, None),
            voxel_mask=_get_voxel_mask(results),
            affine=getattr(results, "affine", None),
            n_timepoints=n_timepoints_reml,
            n_regressors=n_regressors_reml,
            apply_afni_metadata=True,
            mode=r2_semi_mode,  # "full" or "task"
        )

        suffix = "_semipartialR2_task" if r2_semi_mode == "task" else "_semipartialR2"
        print("     Sub-bricks (semi-partial R² with AFNI stat params):")
        for idx, label in enumerate(fitted_labels):
            print(f"       [{idx}] {label}{suffix}")

    # Write nuisance semi-partial R² if available (always "full" mode for nuisance)
    if (
        want_r2_semipartial
        and hasattr(results, "r2_semipartial_nuisance")
        and results.r2_semipartial_nuisance is not None
    ):
        # Generate output path
        if args.Rbuck:
            if args.Rbuck.endswith(".nii.gz"):
                nuisance_semi_r2_path = args.Rbuck.replace(
                    ".nii.gz", "_nuisance_semipartialR2.nii.gz"
                )
            elif args.Rbuck.endswith(".nii"):
                nuisance_semi_r2_path = args.Rbuck.replace(".nii", "_nuisance_semipartialR2.nii.gz")
            else:
                nuisance_semi_r2_path = args.Rbuck + "_nuisance_semipartialR2.nii.gz"
        else:
            nuisance_semi_r2_path = "REML_nuisance_semipartialR2.nii.gz"

        print(f"  • Writing REML nuisance semi-partial R² per regressor: {nuisance_semi_r2_path}")

        from fastfuncstuff.glm.outputs import (
            _get_voxel_mask,
            _resolve_shape,
            write_partial_r2_with_labels,
        )

        # Get nuisance labels from design_info
        nuisance_labels = design_info.get(
            "nuisance_labels",
            [f"nuisance{i}" for i in range(results.r2_semipartial_nuisance.shape[1])],
        )

        # Get design info for stat parameters
        n_timepoints_reml = design_info.get("n_timepoints")
        n_regressors_reml = design_info.get("n_regressors")

        write_partial_r2_with_labels(
            results.r2_semipartial_nuisance,
            nuisance_semi_r2_path,
            condition_labels=nuisance_labels,
            volume_shape=_resolve_shape(results, None),
            voxel_mask=_get_voxel_mask(results),
            affine=getattr(results, "affine", None),
            n_timepoints=n_timepoints_reml,
            n_regressors=n_regressors_reml,
            apply_afni_metadata=True,
            mode="full",  # Always use "full" for nuisance (not rescaled)
        )

        print("     Sub-bricks (nuisance semi-partial R² with AFNI stat params):")
        for idx, label in enumerate(nuisance_labels):
            print(f"       [{idx}] {label}_semipartialR2")

    if args.Rfitts:
        print(f"  • Writing REML fitted model: {args.Rfitts}")
        if results.predicted is not None:
            from fastfuncstuff.glm.outputs import _ensure_numpy, _get_voxel_mask, _resolve_shape

            affine = getattr(results, "affine", np.eye(4))
            volume_shape = _resolve_shape(results, None)
            voxel_mask = _get_voxel_mask(results)
            # predicted is (n_voxels, n_timepoints); scatter/reshape to 4D.
            predicted_vol = _voxels_to_4d_volume(
                _ensure_numpy(results.predicted), volume_shape, voxel_mask
            )
            save_nifti(
                predicted_vol,
                output_path=replace_afni_extension(args.Rfitts, ".nii.gz"),
                affine=affine,
            )
        else:
            print("    ⚠️  Warning: Fitted values not available (predicted=None)")

    if args.Rerrts:
        print(f"  • Writing REML residuals: {args.Rerrts}")
        if results.residuals is not None:
            from fastfuncstuff.glm.outputs import _ensure_numpy, _get_voxel_mask, _resolve_shape

            affine = getattr(results, "affine", np.eye(4))
            volume_shape = _resolve_shape(results, None)
            voxel_mask = _get_voxel_mask(results)
            # residuals is (n_voxels, n_timepoints); scatter/reshape to 4D.
            residuals_vol = _voxels_to_4d_volume(
                _ensure_numpy(results.residuals), volume_shape, voxel_mask
            )
            # Always write NIfTI .nii.gz regardless of input format
            save_nifti(
                residuals_vol,
                output_path=replace_afni_extension(args.Rerrts, ".nii.gz"),
                affine=affine,
            )
        else:
            print("    ⚠️  Warning: Residuals not available")

    if args.save_clean or args.save_nuisance or args.save_taskfit:
        from fastfuncstuff.glm.outputs import (
            _ensure_numpy,
            _get_voxel_mask,
            _resolve_shape,
            find_baseline_columns,
            reconstruct_partial_timeseries,
        )

        # All three reconstruct from the full-design betas. StimBots/StimTops
        # filtering keeps stimulus columns only, so results.betas no longer aligns
        # with the full design's columns and there are no nuisance betas — bail like
        # -Rnuisance.
        design_mat = design_info.get("matrix", design_info.get("design_matrix"))
        if fitted_column_indices is not None or design_mat is None:
            print(
                "  ⚠️  Skipping -save_clean/-save_nuisance/-save_taskfit: needs the full "
                "design (fit was filtered to stimulus columns, or design matrix unavailable)."
            )
        else:
            all_indices = list(range(len(full_labels)))
            task_idx = list(stim_indices) if stim_indices else []
            nuisance_idx = [i for i in all_indices if i not in task_idx]
            affine = getattr(results, "affine", np.eye(4))
            volume_shape = _resolve_shape(results, None)
            voxel_mask = _get_voxel_mask(results)
            betas_np = _ensure_numpy(results.betas)
            design_np = _ensure_numpy(design_mat)

            if args.save_nuisance:
                print(f"  • Writing REML nuisance-only timeseries: {args.save_nuisance}")
                if not nuisance_idx:
                    print("    ⚠️  Warning: no nuisance columns in design; output is all zeros")
                nuis_ts = reconstruct_partial_timeseries(betas_np, design_np, nuisance_idx)
                save_nifti(
                    _voxels_to_4d_volume(nuis_ts, volume_shape, voxel_mask),
                    output_path=replace_afni_extension(args.save_nuisance, ".nii.gz"),
                    affine=affine,
                )
                del nuis_ts

            if args.save_taskfit:
                # Pure task fit: X_task·β_task, no residual, no mean. With -save_nuisance
                # and -Rerrts this is the exact partition data = taskfit + nuisance + resid.
                print(f"  • Writing REML task-only fit: {args.save_taskfit}")
                if not task_idx:
                    print("    ⚠️  Warning: no stimulus columns identified; output is all zeros")
                task_only = reconstruct_partial_timeseries(betas_np, design_np, task_idx)
                save_nifti(
                    _voxels_to_4d_volume(task_only, volume_shape, voxel_mask),
                    output_path=replace_afni_extension(args.save_taskfit, ".nii.gz"),
                    affine=affine,
                )
                del task_only

            if args.save_clean:
                print(f"  • Writing REML nuisance-removed (clean) timeseries: {args.save_clean}")
                if results.residuals is None:
                    print("    ⚠️  Warning: residuals unavailable; cannot build clean timeseries")
                else:
                    resid_np = _ensure_numpy(results.residuals)
                    # clean = data - removed_nuisance_fit + baseline
                    #       = task_fit + residuals + baseline.
                    # Default baseline is one per-voxel grand mean; -save_per_run_polort
                    # instead keeps each run's fitted polort-0 column so run offsets
                    # survive. mean_t(data) = betas @ mean_t(X) + mean_t(residuals), so
                    # the raw data need not be resident — every term comes from the fit.
                    # keep_idx = task columns that stay in the clean signal (task, plus
                    # the per-run baselines when -save_per_run_polort).
                    keep_idx = list(task_idx)
                    baseline = None
                    if args.save_per_run_polort:
                        pol0_idx = find_baseline_columns(design_np, nuisance_idx)
                        if pol0_idx:
                            keep_idx = keep_idx + pol0_idx
                        else:
                            print(
                                "    ⚠️  -save_per_run_polort: no polort-0 columns found; "
                                "falling back to grand mean"
                            )
                            baseline = betas_np @ design_np.mean(axis=0) + resid_np.mean(axis=1)
                    else:
                        baseline = betas_np @ design_np.mean(axis=0) + resid_np.mean(axis=1)
                    # Build in place (keep_fit → clean) to avoid a third (n_vox, n_time) copy.
                    clean_ts = reconstruct_partial_timeseries(betas_np, design_np, keep_idx)
                    clean_ts += resid_np
                    if baseline is not None:
                        clean_ts += baseline[:, None]
                    save_nifti(
                        _voxels_to_4d_volume(clean_ts, volume_shape, voxel_mask),
                        output_path=replace_afni_extension(args.save_clean, ".nii.gz"),
                        affine=affine,
                    )
                    del clean_ts

    if args.Rwherr:
        print(f"  • Writing REML whitened residuals: {args.Rwherr}")
        print("    ⚠️  Warning: Whitened residuals not currently computed")
        # Would need to compute: residuals @ inv(chol(R))

    # Single trials output for ARMA (if requested)
    if args.single_trials and stim_indices:
        label = args.single_trials
        output_filename = f"reml_{label}_single.nii.gz"
        print(f"  • Writing REML single-trial betas (onset order): {output_filename}")
        if "matrix" in design_info:
            write_single_trials_output(
                results,
                output_filename,
                design_info["matrix"],  # Full design matrix for onset extraction
                stim_indices,  # Column indices into full design
                fitted_labels,  # Labels matching results.betas shape
            )
        else:
            print("      ⚠️  Warning: Design matrix not available, cannot determine onset times")

    # OLS outputs - already written by callback during analysis!
    # The callback writes OLS results immediately after OLS completion,
    # freeing memory before the ARMA loop starts.

    # DoF adjustment (e.g. post-NORDIC): rewrite the stat buckets with a reduced
    # dof and inserted z-scores. Done as a post-pass on the finished files so it
    # is identical to running ffs_util_updatedof, regardless of which internal
    # write path produced them.
    if args.adjust_dof:
        from fastfuncstuff.stats.dof_adjust import (
            resolve_dof_adjust_arg,
            update_dof_in_file,
        )

        print()
        print("=" * 70)
        print("🔧 Adjusting statistics for lost degrees of freedom (-adjust_dof)")
        print("=" * 70)
        adjust = resolve_dof_adjust_arg(args.adjust_dof)
        for buck in (args.Obuck, args.Rbuck):
            if not buck:
                continue
            # write_glm_bucket_as_nifti compresses, so a requested ".nii" lands
            # as ".nii.gz"; resolve to whatever was actually written.
            actual = buck if Path(buck).exists() else None
            if actual is None:
                stem = buck
                for ext in (".nii.gz", ".nii.zst", ".nii"):
                    if stem.endswith(ext):
                        stem = stem[: -len(ext)]
                        break
                if Path(stem + ".nii.gz").exists():
                    actual = stem + ".nii.gz"
            if actual is None:
                continue
            print(f"  • {actual}")
            try:
                update_dof_in_file(actual, adjust, actual)
            except ValueError as e:
                # Most likely: no AFNI stat metadata (needs 3drefit at write time).
                print(f"  ⚠️  skipped {actual}: {e}")

    print()
    print("=" * 70)
    print("✅ ffs_reml completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
