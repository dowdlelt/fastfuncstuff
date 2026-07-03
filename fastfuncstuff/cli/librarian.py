#!/usr/bin/env python3
"""
ffs_librarian — derive a custom HRF library from a subject's own data.

This is an NSD-style HRF library builder.  Given fMRI runs + event
onsets, it pools events into one or more groups, fits a per-voxel
FIR/TENT impulse-response, then runs SVD → spherical density manifold →
double-gamma fit to emit a 20-HRF TSV library that drops into the same
slot as ``getcanonicalhrflibrary.tsv`` for downstream tools
(``ffs_hrfopt``, ``ffs_denoise``, ``ffs_ridge``).

The full mathematical pipeline lives in
:mod:`fastfuncstuff.design.hrf_derive`.  This module is the CLI glue:
parse args, load data, fit the GLM, slice betas per group, write
outputs.

Basic usage
-----------

::

    ffs_librarian -input run*.nii.gz \\
                  -events sub-01_run-*_events.tsv \\
                  -prefix sub01_lib \\
                  -tr 2.0

Output (in this example):

- ``sub01_lib_hrflibrary.tsv`` — fitted double-gamma library, 20 HRFs
  × 321 samples at 0.1 s, drop-in replacement for the canonical TSV.
- ``sub01_lib_hrfraw.tsv`` — raw cubic-spline reconstruction (no
  parametric fit) for diagnostic comparison.
- ``sub01_lib_fir_r2.nii.gz`` — pooled FIR R² map.
- ``sub01_lib_pcs.tsv`` — the temporal PCs (3 × n_lags) at TR resolution.
- ``sub01_lib_metadata.json`` — full provenance: TR, FIR window,
  groups, event durations, voxel count, SVD eigenvalues, gamma params.

The ``-split`` flag enables per-group libraries — see the flag's help.

CLI conventions follow the AFNI-style single-dash flags used elsewhere
in ``ffs_*`` (e.g. ``-input`` not ``--input``).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    from fastfuncstuff.cli_utils import (
        add_verbose_arg,
        load_and_preprocess_runs,
        parse_device_arg,
        parse_input_files,
        parse_prefix,
        preflight_check,
    )
    from fastfuncstuff.design.builder import (
        create_onset_matrix_microtime,  # noqa: F401
        parse_afni_timing_file,
        parse_durations,
    )
    from fastfuncstuff.design.hrf import (
        compute_windows_from_durations,  # noqa: F401
        estimate_hrf_window,  # noqa: F401
    )
    from fastfuncstuff.design.hrf_derive import (
        build_pc_basis_design_per_run,
        derive_library,
        svd_decompose,
    )
    from fastfuncstuff.design.matrices import (
        is_tr_locked,  # noqa: F401
        make_fir_design,
        make_tent_design,
    )
    from fastfuncstuff.glm.core import fit_glm
    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.utils import configure_torch_backends
except ImportError as e:
    print(f"ERROR: Could not import fastfuncstuff: {e}")
    print("Make sure fastfuncstuff is installed: pip install -e .")
    sys.exit(1)


class _HelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """Match the formatter used by other ffs_* CLIs."""


# ----------------------------------------------------------------------------
# CLI parser
# ----------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Build the ``ffs_librarian`` argument parser.

    All long flags use AFNI-style single-dash prefixes
    (``-input``, ``-hrf-library``…) to match the rest of the
    ``ffs_*`` family.  Flag values use kebab-case where applicable.
    """
    parser = argparse.ArgumentParser(
        prog="ffs_librarian",
        description=(
            "Build a custom HRF library from your own data (NSD-style "
            "FIR -> SVD -> manifold -> double-gamma)."
        ),
        formatter_class=_HelpFormatter,
        epilog="""
Examples:
  # Single library across all events (BIDS)
  ffs_librarian -input run*.nii.gz -events sub-01_run-*_events.tsv \\
                -prefix sub01_lib -tr 2.0

  # Per-condition-group library (groups by index vector)
  # Suppose 4 conditions [face, house, voice, music]; group visual/audio:
  ffs_librarian -input run*.nii.gz -onsets face.txt house.txt voice.txt music.txt \\
                -durations 2 -tr 2.0 \\
                -split 1,1,2,2 -prefix sub01_lib

  # Per-events-column group (BIDS only) — one library per value of `modality`
  ffs_librarian -input run*.nii.gz -events events*.tsv \\
                -split modality -prefix sub01_lib

  # Force TENT basis (sub-TR onsets)
  ffs_librarian -input run*.nii.gz -events events*.tsv \\
                -basis TENT -prefix sub01_lib

Outputs:
  {prefix}[_group<G>]_hrflibrary.tsv  Double-gamma fit library (drop-in)
  {prefix}[_group<G>]_hrfraw.tsv      Raw cubic-spline reconstruction
  {prefix}[_group<G>]_fir_r2.nii.gz   Per-voxel FIR R² (for the pooled fit)
  {prefix}[_group<G>]_pcs.tsv         Temporal PCs at TR resolution
  {prefix}_metadata.json              Full provenance + per-group details

Notes:
  - The library represents the response *to events of the observed duration*,
    not the impulse response.  For block designs (>~4 s) downstream consumers
    will double-convolve.  A per-group deconvolution step is planned; see
    fastfuncstuff.design.hrf_derive.deconvolve_event_duration.
  - At least 2 runs are recommended so the FIR fit's polynomial detrending
    has enough degrees of freedom; the tool will run on a single run but
    print a warning.
""",
    )

    req = parser.add_argument_group("Required Arguments")
    req.add_argument(
        "-input",
        nargs="+",
        required=True,
        help="Input fMRI run files (one or more, NIfTI).",
    )
    req.add_argument(
        "-prefix",
        required=True,
        help="Output prefix (e.g. 'out/sub01_lib').",
    )

    # Onsets: either AFNI -onsets/-durations OR BIDS -events.
    onset_grp = parser.add_argument_group("Event timing (choose one)")
    onset_grp.add_argument(
        "-onsets",
        nargs="+",
        default=None,
        help="AFNI-format onset files, one per condition.",
    )
    onset_grp.add_argument(
        "-durations",
        nargs="+",
        default=None,
        help="Stimulus durations (s).  Single value or one per condition.",
    )
    onset_grp.add_argument(
        "-events",
        nargs="+",
        default=None,
        metavar="TSV",
        help="BIDS *_events.tsv files (one per run).",
    )
    onset_grp.add_argument(
        "-event-ignore",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="trial_type values to drop from BIDS events.",
    )
    onset_grp.add_argument(
        "-event-cols",
        nargs=3,
        default=None,
        metavar=("ONSET_COL", "DURATION_COL", "TRIAL_TYPE_COL"),
        help=(
            "Override BIDS column names for onset / duration / "
            "trial_type.  Use when the events TSVs use non-standard "
            "names like 'event_onset' instead of 'onset'.  Only valid "
            "with -events."
        ),
    )
    onset_grp.add_argument(
        "-round-onsets",
        nargs="?",
        const=0.7,
        type=float,
        default=None,
        metavar="THRESHOLD",
        help=(
            "Round onsets to nearest TR.  Fraction-through-TR >= "
            "THRESHOLD → ceil, else floor.  If supplied without a "
            "value, defaults to 0.7."
        ),
    )
    onset_grp.add_argument(
        "-round-durations",
        type=int,
        default=None,
        metavar="PLACES",
        help=(
            "Round event durations to PLACES decimal places before "
            "grouping unique durations.  Prevents floating-point "
            "noise (3.03 vs 3.0) from spawning separate conditions."
        ),
    )

    # FIR/TENT options
    fir_grp = parser.add_argument_group("FIR / TENT fit")
    fir_grp.add_argument(
        "-basis",
        choices=["FIR", "TENT", "TENTzero", "auto"],
        default="auto",
        help=(
            "Basis for the pooled impulse-response fit.  ``auto`` "
            "(default) inspects the onset times: if every onset lands "
            "within 10%% of a TR boundary it uses FIR (the fastest, "
            "tightest design); otherwise it uses TENT, which evaluates "
            "the basis at the exact (sub-TR) onset times and is "
            "essential when events are at fractional-TR offsets."
        ),
    )
    fir_grp.add_argument(
        "-fir-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Override the FIR/TENT window length (s).  Default: "
            "max(30 s, estimate_hrf_window across groups).  The 30 s "
            "floor matches NSD's 0-30 s FIR; short estimates from "
            "estimate_hrf_window have produced too-coarse libraries in "
            "testing."
        ),
    )
    fir_grp.add_argument(
        "-fir-duration-min",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help=(
            "Floor on the auto-estimated FIR window (s).  Ignored when "
            "-fir-duration is given.  Defaults to 30 s to match NSD."
        ),
    )
    fir_grp.add_argument(
        "-tr",
        type=float,
        default=None,
        help="Repetition time (s).  Read from NIfTI header if omitted.",
    )
    fir_grp.add_argument(
        "-mask",
        default=None,
        help="Brain mask NIfTI (optional).",
    )
    fir_grp.add_argument(
        "-microtime-dt",
        type=float,
        default=0.1,
        help="Microtime resolution for onset matrices (s).",
    )

    # Splitting
    split_grp = parser.add_argument_group("Condition splitting")
    split_grp.add_argument(
        "-split",
        default=None,
        metavar="VECTOR_OR_COLUMN",
        help=(
            "Group conditions for shape analysis.  Either a "
            "comma-separated integer vector with length = #conditions "
            "(e.g. '1,1,2,2') mapping condition index to group, OR a "
            "BIDS events column name (use that column's unique values as "
            "groups).  Default: all events in one group.\n\n"
            "By default, when -split produces multiple groups they are "
            "STACKED into a single library — per-group FIR betas are "
            "concatenated voxel-wise before the SVD so one set of PCs "
            "covers all groups (between-group HRF shape differences "
            "are captured as extra variance in the manifold).  Pass "
            "-split-separate to emit one library per group instead."
        ),
    )
    split_grp.add_argument(
        "-split-separate",
        action="store_true",
        help=(
            "Force per-group libraries instead of the default stacked "
            "single library.  Required when groups have different event "
            "durations (the duration-deconvolution step needs a single "
            "duration per library)."
        ),
    )

    # Derivation params
    derive_grp = parser.add_argument_group("Library derivation")
    derive_grp.add_argument(
        "-r2-threshold",
        type=float,
        default=0.10,
        help="Minimum FIR R² to include a voxel in the SVD.",
    )
    derive_grp.add_argument(
        "-max-voxels",
        type=int,
        default=20_000,
        help="Random-sample to this many voxels if more survive R² gate.",
    )
    derive_grp.add_argument(
        "-seed",
        type=int,
        default=42,
        help="Random seed for voxel subsampling.",
    )
    derive_grp.add_argument(
        "-n-pcs",
        type=int,
        default=3,
        help="Number of temporal PCs to retain (NSD used 3).",
    )
    derive_grp.add_argument(
        "-n-hrfs",
        type=int,
        default=20,
        help="Target library size (manifold sample count).",
    )
    derive_grp.add_argument(
        "-angular-step",
        type=float,
        default=6.0,
        metavar="DEG",
        help="Angular spacing between manifold samples (auto mode, NSD=6).",
    )
    derive_grp.add_argument(
        "-bandwidth",
        type=float,
        default=8.0,
        metavar="DEG",
        help="Spherical KDE bandwidth for manifold density (auto mode).",
    )
    derive_grp.add_argument(
        "-manifold",
        choices=["auto", "grid", "points"],
        default="auto",
        help="Manifold tracing strategy.",
    )
    derive_grp.add_argument(
        "-manifold-points",
        default=None,
        metavar="JSON",
        help=(
            "Path to a JSON list of K-D points (one per HRF) for "
            "-manifold=points override.  Each entry is a length-K list."
        ),
    )
    derive_grp.add_argument(
        "-fit-gamma",
        choices=["double", "none"],
        default="double",
        help=(
            "Final parametric smoothing of each manifold HRF.  'double' "
            "= SPM double-gamma fit (drop-in replacement); 'none' = emit "
            "only the raw cubic reconstruction."
        ),
    )
    derive_grp.add_argument(
        "-deconvolve-duration",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Wiener-deconvolve the event-duration boxcar from the "
            "library so it represents the impulse response.  "
            "Downstream consumers re-convolve with the event boxcar at "
            "modelling time, so without this step the design is "
            "doubly-convolved for any non-impulse event.  ``auto`` "
            "(default) enables deconvolution when the group's median "
            "event duration exceeds 1.5 * target_dt (i.e. > ~0.15 s "
            "with default 0.1 s sampling); ``on`` forces it; ``off`` "
            "disables it (library entries remain duration-convolved)."
        ),
    )
    derive_grp.add_argument(
        "-deconv-snr",
        type=float,
        default=100.0,
        metavar="SNR",
        help=(
            "Wiener-filter SNR.  Higher = sharper deconvolution but "
            "noisier; lower = smoother but more biased.  100 = a 1%% "
            "noise floor, generally appropriate for the library "
            "waveforms emerging from SVD + manifold + PCHIP."
        ),
    )
    derive_grp.add_argument(
        "-refit-pcs",
        choices=["on", "off"],
        default="on",
        help=(
            "NSD refinement: after the SVD, refit the data using the "
            "top-K PCs convolved with onsets as a fresh design (3 task "
            "regressors per group), and use those per-voxel "
            "coefficients to place voxels on the unit sphere.  This is "
            "what NSD's hrf_constructmanifold.m does and cleans up the "
            "density manifold considerably vs projecting noisy FIR "
            "betas onto the PCs.  Costs one extra GLM fit pass."
        ),
    )

    # Processing
    proc_grp = parser.add_argument_group("Processing")
    proc_grp.add_argument(
        "-device",
        default="auto",
        help="Compute device: 'auto', 'cpu', 'cuda', 'cuda:0', 'mps'.",
    )
    proc_grp.add_argument(
        "-max-poly-degree",
        type=int,
        default=None,
        help="Polynomial detrend degree (per run).  Auto if omitted.",
    )
    proc_grp.add_argument(
        "-debug-design",
        action="store_true",
        help=(
            "Before the GLM fit, print a design-matrix inspection: per-"
            "column L2 norms, near-zero / near-constant columns, X'X "
            "rank, condition number, and the null-space direction for "
            "any rank-deficient combination (so you can see which "
            "columns are degenerate).  Cheap; useful for diagnosing "
            "'should-not-be-singular' failures."
        ),
    )
    add_verbose_arg(proc_grp, default=1)

    return parser


# ----------------------------------------------------------------------------
# Split-flag resolution
# ----------------------------------------------------------------------------


def resolve_split(
    split_arg: str | None,
    n_conditions: int,
    condition_labels: list[str],
    events_split_column: list[list[str]] | None = None,
) -> tuple[list[int], list[str]]:
    """Map ``-split`` flag → per-condition group assignment.

    Parameters
    ----------
    split_arg : str or None
        Raw flag value.  ``None`` -> all in group 0.  Comma-separated
        integers -> vector form, one entry per condition.  Anything
        else is treated as a BIDS events column name.
    n_conditions : int
        Number of conditions in ``condition_labels``.
    condition_labels : list[str]
        Condition labels (from -onsets stems or BIDS trial_type uniques).
    events_split_column : list[list[str]] or None
        Per-condition list of column values (only relevant for BIDS
        column-mode splits).  ``events_split_column[i]`` = the value of
        the named column for condition ``i``.  Unused for the MVP, which
        only supports column-mode when the column name *replaces*
        trial_type at parse time — see the CLI implementation.

    Returns
    -------
    group_per_cond : list[int]
        Group index assigned to each condition (length n_conditions).
    group_labels : list[str]
        Display name for each group (used in output filenames).

    Notes
    -----
    BIDS column-mode (``-split <name>``) is implemented in the CLI by
    re-parsing events with ``event_cols=('onset', 'duration', <name>)``,
    so by the time this function is called each "condition" is already
    a value of the chosen column.  We therefore just return one group
    per condition in that case.
    """
    del events_split_column  # reserved for future BIDS column-mode wiring
    if split_arg is None:
        return [0] * n_conditions, ["all"]

    # Vector form: e.g. "1,1,2,2"
    if all(p.strip().lstrip("-").isdigit() for p in split_arg.split(",")):
        parts = [int(p.strip()) for p in split_arg.split(",")]
        if len(parts) != n_conditions:
            raise ValueError(
                f"-split vector has {len(parts)} entries but there are {n_conditions} conditions."
            )
        unique_groups = sorted(set(parts))
        remap = {g: i for i, g in enumerate(unique_groups)}
        group_per_cond = [remap[g] for g in parts]
        group_labels = [f"g{g}" for g in unique_groups]
        return group_per_cond, group_labels

    # Column-mode: handled at events-parse time; here each cond = its own group.
    group_per_cond = list(range(n_conditions))
    # Sanitize labels for filenames.
    safe = [
        "".join(c if c.isalnum() or c in "-_" else "_" for c in lbl) for lbl in condition_labels
    ]
    return group_per_cond, safe


# ----------------------------------------------------------------------------
# Pooled-onset construction
# ----------------------------------------------------------------------------


def pool_onsets_by_group(
    all_onsets: list[list[np.ndarray]],
    durations: list[float],
    group_per_cond: list[int],
    n_groups: int,
) -> tuple[list[list[np.ndarray]], list[float]]:
    """Merge same-group conditions into one pooled "condition" per group.

    Used to set up the NSD-style FIR fit: each group is treated as a
    single experimental condition pooling all events of all conditions
    in that group.  Within-run onset arrays are concatenated and
    sorted; per-group duration is the median across pooled events
    (groups with heterogeneous durations get logged via the sidecar so
    a future deconvolution step can do something smarter).

    Parameters
    ----------
    all_onsets : list[list[ndarray]]
        ``all_onsets[cond][run]`` -> onset times in seconds.
    durations : list[float]
        One duration per condition.
    group_per_cond : list[int]
        Group id (0..n_groups-1) for each condition.
    n_groups : int
        Total number of groups.

    Returns
    -------
    pooled_onsets : list[list[ndarray]]
        ``pooled_onsets[group][run]`` -> sorted onset times for that
        group within that run.
    pooled_durations : list[float]
        Median event duration per group (seconds).
    """
    if len(all_onsets) == 0:
        raise ValueError("No conditions in all_onsets.")
    n_runs = len(all_onsets[0])
    pooled_onsets: list[list[np.ndarray]] = []
    pooled_durations: list[float] = []
    for g in range(n_groups):
        cond_idx = [i for i, gi in enumerate(group_per_cond) if gi == g]
        per_run: list[np.ndarray] = []
        for r in range(n_runs):
            chunks = [all_onsets[i][r] for i in cond_idx if all_onsets[i][r].size > 0]
            if chunks:
                merged = np.sort(np.concatenate(chunks))
            else:
                merged = np.array([], dtype=float)
            per_run.append(merged)
        pooled_onsets.append(per_run)
        dur_pool = [durations[i] for i in cond_idx]
        pooled_durations.append(float(np.median(dur_pool)) if dur_pool else 0.0)
    return pooled_onsets, pooled_durations


# ----------------------------------------------------------------------------
# Per-run FIR / TENT design builder
# ----------------------------------------------------------------------------


def build_per_run_designs(
    pooled_onsets: list[list[np.ndarray]],
    pooled_durations: list[float],
    run_starts: list[int],
    n_timepoints: int,
    tr: float,
    basis: str,
    fir_window_s: float,
    device: torch.device,
) -> tuple[list[torch.Tensor], int, np.ndarray]:
    # DEPRECATED — kept only so any external script importing this name
    # keeps working.  All ffs_* CLIs route through
    # fastfuncstuff.design.builder.build_per_run_task_designs now.
    import warnings as _warnings

    _warnings.warn(
        "cli.librarian.build_per_run_designs is deprecated; use "
        "fastfuncstuff.design.builder.build_per_run_task_designs instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    """Construct one design matrix per run containing all groups' lag blocks.

    The design has ``n_groups * n_lags`` task columns.  Groups are
    stacked along the regressor axis, so each group's betas can be
    extracted with a simple slice after fitting.

    Parameters
    ----------
    pooled_onsets, pooled_durations
        Output of :func:`pool_onsets_by_group`.
    run_starts : list[int]
        Run boundary start indices (in TRs).
    n_timepoints : int
        Total TRs across all runs.
    tr : float
        Repetition time in seconds.
    basis : {"FIR", "TENT", "TENTzero"}
        Basis selection.
    fir_window_s : float
        Length of the impulse response window in seconds.
    device : torch.device
        Where to materialize the design tensors.

    Returns
    -------
    per_run_designs : list[torch.Tensor]
        One ``(n_run_tp, n_groups * n_lags)`` tensor per run.
    n_lags : int
        Number of basis functions per group.
    lag_times : np.ndarray, shape (n_lags,)
        Lag time grid in seconds.  For FIR these are TR multiples; for
        TENT/TENTzero they are knot positions in [0, fir_window_s].
    """
    del pooled_durations  # consumed at the CLI orchestration layer for metadata
    n_groups = len(pooled_onsets)
    n_runs = len(run_starts)
    run_starts_ext = list(run_starts) + [n_timepoints]

    if basis == "FIR":
        n_lags = max(1, int(np.ceil(fir_window_s / tr)))
        lag_times = np.arange(n_lags) * tr
    else:
        # TENT/TENTzero: one knot per TR, plus 1 at the right edge.
        n_lags = max(2, int(round(fir_window_s / tr)) + 1)
        # ``make_tent_design`` zero_edges drops the first and last basis
        # so the *user-visible* lag time grid still spans [0, window].
        lag_times = np.linspace(0.0, fir_window_s, n_lags)

    per_run_designs: list[torch.Tensor] = []

    for r in range(n_runs):
        r_start, r_end = run_starts_ext[r], run_starts_ext[r + 1]
        n_run_tp = r_end - r_start

        group_blocks: list[torch.Tensor] = []
        for g in range(n_groups):
            group_onsets_run = pooled_onsets[g][r]
            if basis == "FIR":
                # FIR expects TR-locked, binary onsets: build a column vector.
                onset_vec = torch.zeros(n_run_tp, 1, device=device)
                if group_onsets_run.size > 0:
                    idx = np.round(group_onsets_run / tr).astype(int)
                    idx = idx[(idx >= 0) & (idx < n_run_tp)]
                    onset_vec[idx, 0] = 1.0
                block = make_fir_design(onset_vec, n_lags, n_run_tp, device=device)
            else:
                zero_edges = basis == "TENTzero"
                # make_tent_design wants a list of per-condition onset arrays.
                block = make_tent_design(
                    [group_onsets_run.astype(np.float64)],
                    bot=0.0,
                    top=float(fir_window_s),
                    tr=tr,
                    n_timepoints=n_run_tp,
                    n_basis=n_lags,
                    zero_edges=zero_edges,
                    device=device,
                )
            group_blocks.append(block)

        # Concatenate group blocks along regressor axis.
        per_run_designs.append(torch.cat(group_blocks, dim=1))

    # For TENTzero the actual n_lags is n_basis - 2 (edges dropped).
    if basis == "TENTzero":
        n_lags_actual = n_lags - 2
        lag_times = np.linspace(
            (fir_window_s) / (n_lags - 1),
            fir_window_s - (fir_window_s) / (n_lags - 1),
            n_lags_actual,
        )
    else:
        n_lags_actual = n_lags

    return per_run_designs, n_lags_actual, lag_times


# ----------------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------------


def write_tsv(path: Path, library: np.ndarray) -> None:
    """Write an HRF library array as TSV in the canonical format.

    The canonical file ``getcanonicalhrflibrary.tsv`` stores the library
    as ``(n_timepoints, n_hrfs)`` (columns are HRFs).  We transpose
    accordingly before writing.

    Parameters
    ----------
    path : Path
        Output path.
    library : np.ndarray, shape (n_hrfs, n_timepoints)
        HRF rows.  Will be transposed on write.
    """
    np.savetxt(path, library.T, fmt="%.10g", delimiter="\t")


def write_r2_volume(
    r2: np.ndarray,
    mask_flat: np.ndarray,
    volume_shape: tuple[int, int, int],
    path: Path,
    affine: np.ndarray,
) -> None:
    """Save a (n_voxels,) R² array as a 3D NIfTI using the run mask."""
    vol = np.zeros(volume_shape, dtype=np.float32)
    flat = vol.reshape(-1)
    flat[mask_flat] = r2.astype(np.float32)
    save_nifti(vol.reshape(volume_shape), output_path=str(path), affine=affine)


def write_qc_artifacts(
    lib,  # LibraryResult from hrf_derive
    lag_times: np.ndarray,
    group_label: str,
    prefix: str,
    gtag: str,
    verbose: int = 1,
) -> dict:
    """Emit the QC artifacts that let a human sanity-check the library.

    Three TSVs and (if matplotlib is available) three PNGs:

    - ``{prefix}{gtag}_qc_mean_fir.tsv`` — pooled FIR HRF estimate
      averaged across the selected voxels.  This is the closest thing
      to a "what does the average task HRF look like in this dataset?"
      check; should be visibly HRF-shaped (rise to peak around 4-6 s,
      undershoot, decay).  Two columns: lag time (s), beta.
    - ``{prefix}{gtag}_qc_eigvals.tsv`` — full singular-value spectrum
      from the SVD plus the variance-explained percentage for each.
      Look for an elbow around index 3 (NSD-validated K=3).
    - ``{prefix}{gtag}_qc_sphere_hist.tsv`` — 2D histogram of voxel
      loadings on (PC2, PC3) at NSD's 0.02 bin spacing.  This is the
      "unit-circle heatmap" referenced in ``hrf_constructmanifold.m``
      that visualizes the HRF-shape manifold; saved as a square TSV
      grid (rows = PC2 bins, cols = PC3 bins).

    PNGs (only if matplotlib is importable):

    - ``{prefix}{gtag}_qc_mean_fir.png`` — single curve.
    - ``{prefix}{gtag}_qc_sphere_hist.png`` — heatmap of the 2D
      histogram with a unit-circle overlay; mirrors NSD's plot.
    - ``{prefix}{gtag}_qc_library.png`` — overlay of every library HRF
      (raw + gamma-fit if available) so the user can confirm the
      manifold sampling covered a sensible HRF shape range.

    Returns a dict mapping artifact name → filesystem path.
    """
    artifacts: dict[str, str] = {}

    mean_path = Path(f"{prefix}{gtag}_qc_mean_fir.tsv")
    arr = np.column_stack([lag_times, lib.mean_fir_hrf])
    np.savetxt(
        mean_path,
        arr,
        fmt="%.10g",
        delimiter="\t",
        header="lag_time_s\tmean_fir_beta",
        comments="",
    )
    artifacts["mean_fir_tsv"] = str(mean_path)

    eig_path = Path(f"{prefix}{gtag}_qc_eigvals.tsv")
    s = lib.svd.eigvals
    var = (s**2) / max((s**2).sum(), 1e-30)
    np.savetxt(
        eig_path,
        np.column_stack([np.arange(s.size), s, var, np.cumsum(var)]),
        fmt=["%d", "%.10g", "%.10g", "%.10g"],
        delimiter="\t",
        header="index\tsingular_value\tvariance_fraction\tcumulative_variance",
        comments="",
    )
    artifacts["eigvals_tsv"] = str(eig_path)

    if lib.sphere_hist2d is not None:
        hist_path = Path(f"{prefix}{gtag}_qc_sphere_hist.tsv")
        np.savetxt(hist_path, lib.sphere_hist2d, fmt="%d", delimiter="\t")
        artifacts["sphere_hist_tsv"] = str(hist_path)

    # PNGs — best-effort; if matplotlib is missing, just skip without
    # failing the run (TSV artifacts already provide the raw data).
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if verbose >= 1:
            print("    (matplotlib not installed — skipping QC PNGs)")
        return artifacts

    # Mean FIR HRF plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lag_times, lib.mean_fir_hrf, "o-", color="C0")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("mean FIR beta (across selected voxels)")
    ax.set_title(f"Pooled task HRF — group '{group_label}'")
    fig.tight_layout()
    mean_png = Path(f"{prefix}{gtag}_qc_mean_fir.png")
    fig.savefig(mean_png, dpi=120)
    plt.close(fig)
    artifacts["mean_fir_png"] = str(mean_png)

    # Sphere heatmap (PC2 vs PC3)
    if lib.sphere_hist2d is not None and lib.sphere_hist_edges is not None:
        edges = lib.sphere_hist_edges
        fig, ax = plt.subplots(figsize=(5.5, 5))
        # extent matches the bin edges; transpose to put PC2 on X and
        # PC3 on Y (NSD convention: xlabel "loading on PC2",
        # ylabel "loading on PC3").
        im = ax.imshow(
            lib.sphere_hist2d.T,
            extent=(edges[0], edges[-1], edges[0], edges[-1]),
            origin="lower",
            cmap="hot",
            aspect="equal",
        )
        # Overlay the manifold points (those used to build the library)
        # in cyan; also draw the unit circle to make orientation obvious.
        if lib.manifold.shape[1] == 3:
            ax.scatter(
                lib.manifold[:, 1],
                lib.manifold[:, 2],
                marker="o",
                facecolors="none",
                edgecolors="cyan",
                s=40,
                label="library samples",
            )
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="white", lw=0.8)
        ax.set_xlabel("loading on PC2")
        ax.set_ylabel("loading on PC3")
        ax.set_title(f"Unit-sphere density — group '{group_label}'")
        fig.colorbar(im, ax=ax, label="voxel count")
        fig.tight_layout()
        sphere_png = Path(f"{prefix}{gtag}_qc_sphere_hist.png")
        fig.savefig(sphere_png, dpi=120)
        plt.close(fig)
        artifacts["sphere_hist_png"] = str(sphere_png)

    # Library overlay — one subplot per curve type so different
    # transformations don't pile on top of each other.  Use the turbo
    # colormap (high luminance gradient across the full HRF index
    # range, much easier to distinguish than viridis when the curves
    # almost coincide as they often do).  We always emit a 4-row figure
    # whose unused rows are blank when deconvolution / gamma fitting
    # were skipped, so a user comparing runs gets the same layout.
    n = lib.raw.shape[0]
    cmap = plt.get_cmap("turbo")
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    panels = [
        ("raw cubic — duration-convolved", lib.raw, "-"),
        ("gamma fit — duration-convolved", lib.fitted, "--"),
        ("raw cubic — impulse (deconvolved)", lib.raw_deconvolved, "-"),
        ("gamma fit — impulse (final library)", lib.fitted_deconvolved, "--"),
    ]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(8, 1.9 * len(panels)),
        sharex=True,
        sharey=True,
    )
    for ax, (title, data, linestyle) in zip(axes, panels, strict=False):
        ax.axhline(0, color="0.6", lw=0.5)
        if data is None:
            ax.text(
                0.5,
                0.5,
                "not computed",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="0.6",
                fontsize=10,
            )
            ax.set_title(title, fontsize=10, loc="left")
            continue
        for i in range(data.shape[0]):
            ax.plot(
                lib.target_times,
                data[i],
                color=colors[i],
                lw=1.1,
                linestyle=linestyle,
                alpha=0.85,
            )
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("peak=1")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        f"Library HRFs — group '{group_label}' ({n} samples)  ·  turbo colormap by HRF index",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    overlay_png = Path(f"{prefix}{gtag}_qc_library.png")
    fig.savefig(overlay_png, dpi=120)
    plt.close(fig)
    artifacts["library_overlay_png"] = str(overlay_png)

    return artifacts


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    """Entry point — orchestrates the full ffs_librarian pipeline."""
    parser = create_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    # Resolve prefix / extension.
    pfx = parse_prefix(args.prefix)
    args.prefix = pfx.stem
    nii_ext = pfx.nifti_ext

    print("=" * 72)
    print(" ffs_librarian — data-derived HRF library")
    print("=" * 72)
    print(f"  Started: {datetime.now().isoformat(timespec='seconds')}")
    print(f"  Prefix:  {args.prefix}")

    # --- Validate event input -------------------------------------------------
    has_onsets = bool(args.onsets)
    has_events = bool(args.events)
    if has_onsets == has_events:
        print("ERROR: Specify exactly one of -onsets/-durations or -events.")
        sys.exit(1)
    if has_onsets and args.durations is None:
        print("ERROR: -durations is required with -onsets.")
        sys.exit(1)
    if args.event_cols and not has_events:
        print("ERROR: -event-cols requires -events.")
        sys.exit(1)
    if args.event_ignore and not has_events:
        print("ERROR: -event-ignore requires -events.")
        sys.exit(1)

    # --- Parse inputs ---------------------------------------------------------
    input_files = parse_input_files(args.input)
    n_runs = len(input_files)
    if n_runs < 1:
        print("ERROR: At least 1 input run required.")
        sys.exit(1)
    if n_runs < 2:
        print(
            "WARNING: only 1 run provided; polynomial detrending will eat "
            "most degrees of freedom.  Multi-run data strongly recommended."
        )

    # -split COLUMN mode requires BIDS events — detect it here so we can
    # adjust the events-parse to make COLUMN the trial_type.
    split_arg = args.split
    split_column_mode = split_arg is not None and not all(
        p.strip().lstrip("-").isdigit() for p in split_arg.split(",")
    )

    if has_events:
        from fastfuncstuff.cli_utils import clean_condition_labels
        from fastfuncstuff.design.bids_events import parse_bids_events

        if len(args.events) != n_runs:
            print(f"ERROR: -events: {len(args.events)} files but {n_runs} runs.")
            sys.exit(1)
        # Resolve event_cols: -event-cols takes precedence; split-column
        # mode overlays the split column as the trial_type axis.  Falls
        # back to None (parse_bids_events defaults: onset/duration/trial_type).
        event_cols: tuple[str, str, str] | None = None
        if args.event_cols:
            event_cols = tuple(args.event_cols)  # type: ignore[assignment]
        if split_column_mode:
            # Reuse the user-supplied onset/duration column names when
            # provided, else fall back to BIDS defaults; replace the
            # trial_type column with the split column.
            on_col = event_cols[0] if event_cols else "onset"
            dur_col = event_cols[1] if event_cols else "duration"
            event_cols = (on_col, dur_col, split_arg)
            print(f"  -split column mode: grouping by events column '{split_arg}'")
        all_onsets, durations, condition_labels = parse_bids_events(
            event_files=args.events,
            event_ignore=args.event_ignore,
            event_cols=event_cols,
            round_durations=args.round_durations,
        )
        n_conditions = len(condition_labels)
        print(f"  {n_conditions} conditions from BIDS events: {condition_labels}")
        # Column-mode -> each condition IS already a group; clear arg so
        # resolve_split doesn't try to vectorize the column name.
        split_for_resolver = None if split_column_mode else split_arg
    else:
        from fastfuncstuff.cli_utils import clean_condition_labels

        onset_files = args.onsets
        n_conditions = len(onset_files)
        condition_labels = clean_condition_labels([Path(f).stem for f in onset_files])
        for f in onset_files:
            if not Path(f).exists():
                print(f"ERROR: Onset file not found: {f}")
                sys.exit(1)
        durations = parse_durations(args.durations, n_conditions, condition_labels)
        if args.round_durations is not None:
            durations = [round(d, args.round_durations) for d in durations]
        all_onsets = [parse_afni_timing_file(f) for f in onset_files]
        for i, cond_runs in enumerate(all_onsets):
            if len(cond_runs) != n_runs:
                print(
                    f"ERROR: Onset file {onset_files[i]} has "
                    f"{len(cond_runs)} runs, but {n_runs} input runs."
                )
                sys.exit(1)
        if split_column_mode:
            print(
                "ERROR: -split as a column name requires -events (BIDS); "
                "with -onsets, -split must be a comma vector."
            )
            sys.exit(1)
        split_for_resolver = split_arg

    # --- Resolve grouping -----------------------------------------------------
    group_per_cond, group_labels = resolve_split(split_for_resolver, n_conditions, condition_labels)
    n_groups = len(group_labels)
    print(f"  Groups ({n_groups}): {group_labels}")
    if args.verb >= 1:
        for i, lbl in enumerate(condition_labels):
            print(f"    {lbl} -> {group_labels[group_per_cond[i]]}")

    preflight_check(
        input_files=input_files,
        onset_files=args.onsets if has_onsets else None,
        ortvec_files=None,
    )

    # --- Load data ------------------------------------------------------------
    device, _, _ = parse_device_arg(args.device)
    configure_torch_backends(device)
    # Two "Device:" lines may follow — they mean different things:
    # the one here is the *compute* device (where GLM chunks are
    # processed), the one printed by load_and_preprocess_runs below
    # is the *storage* device (where the loaded data lives; forced to
    # CPU to keep GPU memory free for compute chunks).
    print(f"  Compute device: {device}  (storage device printed below by loader)")

    load_result = load_and_preprocess_runs(
        input_files=input_files,
        tr=args.tr,
        mask_file=args.mask,
        blur_fwhm=None,
        do_scale=True,
        device=device,
        force_cpu=True,
        dry_run=False,
        verbose=True,
    )
    data = load_result.data
    run_starts = load_result.run_starts
    affine = load_result.affine
    volume_shape = load_result.volume_shape
    tr = load_result.tr
    mask_flat = load_result.mask_flat
    n_voxels = load_result.n_voxels
    n_timepoints = load_result.n_timepoints
    if args.tr is None:
        args.tr = tr
    print(f"  Data: {n_voxels:,} voxels × {n_timepoints} TR ({n_runs} runs, TR={tr}s)")

    # Snap onsets to TR boundaries (now that TR is known).  Applies to
    # both BIDS and AFNI paths.  Helpful when events are jittered around
    # nominal TR boundaries but the analyst wants the FIR path (rather
    # than auto-falling back to TENT).
    if args.round_onsets is not None:
        from fastfuncstuff.design.builder import round_onsets as _round_onsets

        all_onsets = _round_onsets(all_onsets, tr, threshold=args.round_onsets)
        print(f"  Rounded onsets to nearest TR (threshold={args.round_onsets:.2f} of TR).")

    # --- Pool onsets per group ------------------------------------------------
    pooled_onsets, pooled_durations = pool_onsets_by_group(
        all_onsets, durations, group_per_cond, n_groups
    )
    for g in range(n_groups):
        n_ev = sum(o.size for o in pooled_onsets[g])
        print(
            f"  Group '{group_labels[g]}': {n_ev} pooled events, "
            f"median duration {pooled_durations[g]:.2f}s"
        )

    # --- Build per-run task designs via the shared API -----------------------
    # All FIR/TENT/CSPLIN design construction (auto-basis resolution,
    # auto-window from durations, per-condition basis counts) lives in
    # fastfuncstuff.design.builder.build_per_run_task_designs.  We pool
    # events per group into a single pseudo-condition first (NSD does
    # this — "treat all stimuli as one condition" — to make the SVD
    # input clean); then call the shared builder with one "condition"
    # per group.  This means the bug-prone bits (axis flips, polynomial
    # double-counting, off-by-one in TENTzero) are the same code path
    # used by ffs_deconvolve and any other CLI doing FIR/TENT modelling.
    from fastfuncstuff.design.builder import build_per_run_task_designs

    run_starts_ext = list(run_starts) + [n_timepoints]
    n_tp_per_run = [run_starts_ext[r + 1] - run_starts_ext[r] for r in range(n_runs)]

    design_result = build_per_run_task_designs(
        onsets_per_cond_per_run=pooled_onsets,
        n_timepoints_per_run=n_tp_per_run,
        tr=tr,
        basis=args.basis,
        condition_labels=group_labels,
        durations_per_condition=pooled_durations,
        fir_window_s=(float(args.fir_duration) if args.fir_duration is not None else None),
        fir_window_min_s=float(args.fir_duration_min),
        tr_locked_threshold=0.1,
        device=device,
    )
    for note in design_result.notes:
        print(f"  {note}")
    args.basis = design_result.basis_resolved
    per_run_designs = design_result.per_run
    # Librarian pools each group into one pseudo-condition, so all
    # n_basis_per_condition entries are equal: take the first as the
    # canonical n_lags for downstream slicing/labelling.
    n_lags = design_result.n_basis_per_condition[0]
    lag_times = design_result.lag_times_s[0] if design_result.lag_times_s else np.array([])
    fir_window_s = design_result.fir_window_s[0][1] if design_result.fir_window_s else 0.0
    n_task_cols = n_groups * n_lags
    print(f"  Basis: {args.basis}, window: {fir_window_s:.2f}s ({n_lags} {args.basis} basis fns)")
    print(f"  Design: {n_task_cols} task cols ({n_groups} groups × {n_lags} lags)")

    # --- Split data per run, then pack into canonical shared-task GLM ------
    # Shared-task across runs (one set of FIR betas per voxel, fit
    # JOINTLY using all data) + per-run block-diagonal polynomials.
    # See fastfuncstuff.design.builder.pack_for_shared_task_glm — the
    # naive per-run-list call to fit_glm block-diagonalizes the task
    # block too, giving per-run task betas (1/n_runs of the data per
    # estimate, much noisier).  We want the shared form.
    from fastfuncstuff.design.builder import pack_for_shared_task_glm

    per_run_data = [data[:, run_starts_ext[r] : run_starts_ext[r + 1]] for r in range(n_runs)]
    # Resolve polort: None (auto) → ~run_duration_min/2 per run.  fit_glm
    # used to do this automatically when handed a per-run list; we
    # replicate it here so the packed-design path matches.
    if args.max_poly_degree is None:
        run_duration_min = per_run_data[0].shape[1] * tr / 60.0 if per_run_data else 1.0
        polort_resolved = max(0, round(run_duration_min / 2))
        print(f"  Polort auto: {polort_resolved} (run duration ≈ {run_duration_min:.1f} min)")
    else:
        polort_resolved = int(args.max_poly_degree)

    packed = pack_for_shared_task_glm(
        per_run_data=per_run_data,
        per_run_task_designs=per_run_designs,
        polort=polort_resolved,
        task_column_labels=design_result.column_labels,
        device=torch.device("cpu"),
    )
    print(
        f"  Fitting pooled FIR/TENT GLM "
        f"(design {packed.design_concat.shape}: "
        f"{packed.n_task_cols} task + "
        f"{packed.design_concat.shape[1] - packed.n_task_cols} nuisance "
        f"across {len(per_run_data)} runs)"
    )
    results = fit_glm(
        data=packed.data_concat,
        design=packed.design_concat,
        tr=tr,
        max_poly_degree=-1,
        device=device,
        verbose=args.verb >= 1,
        want_r2_run=False,
        debug_design=args.debug_design,
    )
    # ``betas`` is (n_voxels, n_task_cols + n_nuisance); take the first
    # n_task_cols (these correspond to our group blocks, in order).
    betas_full = results.betas
    if isinstance(betas_full, torch.Tensor):
        betas_full = betas_full.detach().cpu().numpy()
    r2 = results.r2
    if isinstance(r2, torch.Tensor):
        r2 = r2.detach().cpu().numpy()

    # Auto-mask: zero out R² for voxels with no FIR signal.  Without
    # a brain mask the loader keeps air/background voxels, and the
    # polynomial detrending fits their constant signal to R²=1; those
    # voxels then dominate the voxel-selection step and produce a
    # noise-only SVD.  Compute the L2-norm of the task block of betas
    # per voxel; voxels at exactly zero are non-brain.  We zero their
    # R² so the output map is interpretable and so they don't pollute
    # the SVD.
    task_block = betas_full[:, :n_task_cols]
    task_norm = np.linalg.norm(task_block, axis=1)
    dead = task_norm <= 1e-10
    n_dead = int(dead.sum())
    if n_dead > 0:
        r2[dead] = 0.0
        if n_dead > 0.2 * r2.size and args.mask is None:
            print(
                f"  WARNING: {n_dead:,}/{r2.size:,} voxels ({100 * n_dead / r2.size:.1f}%) "
                f"had zero FIR-beta norm (likely air/background; their R² was set to 0). "
                f"Pass -mask <brain.nii.gz> for cleaner outputs."
            )
        elif args.verb >= 1:
            print(f"  Auto-masked {n_dead:,} zero-FIR voxels (R² set to 0).")

    # --- Per-group library derivation -----------------------------------------
    optional_manifold_points = None
    if args.manifold == "points":
        if args.manifold_points is None:
            print("ERROR: -manifold points requires -manifold-points <json>.")
            sys.exit(1)
        with open(args.manifold_points) as fh:
            optional_manifold_points = np.asarray(json.load(fh), dtype=float)

    # ------------------------------------------------------------------
    # Decide flow: stacked single library (default for n_groups > 1) vs
    # per-group separate libraries (-split-separate, or n_groups == 1).
    # ------------------------------------------------------------------
    use_stacking = (n_groups > 1) and (not args.split_separate)

    # Check duration consistency for stacked mode w/ deconvolution.
    uniform_duration = True
    if n_groups > 1:
        ref_dur = round(float(pooled_durations[0]), 6)
        uniform_duration = all(round(float(d), 6) == ref_dur for d in pooled_durations)

    if use_stacking and not uniform_duration:
        if args.deconvolve_duration == "on":
            print(
                "ERROR: -deconvolve-duration on requires a single duration "
                "across all -split groups, but pooled durations differ "
                f"({pooled_durations}).  Either pass -split-separate (one "
                "library per group, each deconvolved at its own duration), "
                "or set -deconvolve-duration off."
            )
            sys.exit(1)
        elif args.deconvolve_duration == "auto":
            print(
                "  WARNING: stacked mode but groups have different event "
                f"durations {pooled_durations}.  Skipping duration "
                "deconvolution (use -split-separate for per-group deconv)."
            )

    if use_stacking:
        print(
            f"\n  Stacking mode: combining {n_groups} groups into one "
            "shared library (-split-separate to disable)."
        )
    elif n_groups > 1:
        print(f"\n  Per-group mode: one library per group ({n_groups} libraries, -split-separate).")

    metadata: dict = {
        "tool": "ffs_librarian",
        "started": datetime.now().isoformat(timespec="seconds"),
        "tr": float(tr),
        "basis": args.basis,
        "fir_window_s": float(fir_window_s),
        "n_lags": int(n_lags),
        "lag_times_s": lag_times.tolist(),
        "n_voxels": int(n_voxels),
        "n_timepoints": int(n_timepoints),
        "n_runs": int(n_runs),
        "condition_labels": condition_labels,
        "group_labels": group_labels,
        "group_per_condition": group_per_cond,
        "split_mode": "stacked" if use_stacking else ("single" if n_groups == 1 else "separate"),
        # Per-group ``duration_convolved`` flags live under each entry in
        # ``groups`` (the global flag here is the AND of all groups, kept
        # for backwards-compat with code that scans top-level only).
        "duration_convolved": True,  # overwritten after the loop
        "groups": [],
    }

    # ------------------------------------------------------------------
    # Helper: compute NSD-refit PC loadings for a given (betas, onsets)
    # pair using a *given* set of PCs.  Used by both branches — separate
    # mode passes per-group SVD's PCs; stacked mode passes the shared
    # PCs derived from the stacked-betas SVD.
    # ------------------------------------------------------------------
    def _compute_refit_weights(group_idx: int, shared_pcs: np.ndarray) -> np.ndarray:
        onsets_per_run_g = pooled_onsets[group_idx]
        n_tp_per_run_local = [run_starts_ext[r + 1] - run_starts_ext[r] for r in range(n_runs)]
        pc_designs_np = build_pc_basis_design_per_run(
            onsets_per_run=onsets_per_run_g,
            pcs=shared_pcs,
            lag_times=lag_times,
            tr=tr,
            n_timepoints_per_run=n_tp_per_run_local,
            basis=args.basis,
        )
        pc_designs_torch = [
            torch.from_numpy(d.astype(np.float32)).to(device) for d in pc_designs_np
        ]
        refit_packed = pack_for_shared_task_glm(
            per_run_data=per_run_data,
            per_run_task_designs=pc_designs_torch,
            polort=polort_resolved,
            task_column_labels=[f"PC{i}" for i in range(args.n_pcs)],
            device=torch.device("cpu"),
        )
        refit_result = fit_glm(
            data=refit_packed.data_concat,
            design=refit_packed.design_concat,
            tr=tr,
            max_poly_degree=-1,
            device=device,
            verbose=False,
            want_r2_run=False,
            debug_design=args.debug_design,
        )
        rb = refit_result.betas
        if isinstance(rb, torch.Tensor):
            rb = rb.detach().cpu().numpy()
        return rb[:, : refit_packed.n_task_cols]

    # ------------------------------------------------------------------
    # Helper: take prepared (betas, r2, refit_weights, deconv_duration)
    # for one library — call derive_library, write the TSVs / PNGs /
    # metadata, return the LibraryResult.
    # ------------------------------------------------------------------
    def _derive_and_write_one_library(
        betas_in: np.ndarray,
        r2_in: np.ndarray,
        refit_weights_in: np.ndarray | None,
        deconv_duration: float | None,
        gtag: str,
        label_for_output: str,
        event_durations_arr: np.ndarray,
    ):
        if deconv_duration is not None:
            print(
                f"    Wiener deconvolving {deconv_duration:.2f}s boxcar (SNR={args.deconv_snr:g})"
            )
        lib = derive_library(
            betas_in,
            r2_in,
            lag_times,
            n_pcs=args.n_pcs,
            n_hrfs=args.n_hrfs,
            r2_threshold=args.r2_threshold,
            max_voxels=args.max_voxels,
            angular_step_deg=args.angular_step,
            bandwidth_deg=args.bandwidth,
            manifold_mode=args.manifold,
            manifold_points=optional_manifold_points,
            fit_gamma=(args.fit_gamma == "double"),
            seed=args.seed,
            event_durations=event_durations_arr,
            refit_weights=refit_weights_in,
            deconvolve_duration=deconv_duration,
            deconv_snr=args.deconv_snr,
        )

        raw_path = Path(f"{args.prefix}{gtag}_hrfraw.tsv")
        pcs_path = Path(f"{args.prefix}{gtag}_pcs.tsv")
        r2_path = Path(f"{args.prefix}{gtag}_fir_r2{nii_ext}")

        write_tsv(raw_path, lib.raw)
        print(f"    Wrote {raw_path}   [duration-convolved cubic recon]")
        np.savetxt(pcs_path, lib.svd.pcs.T, fmt="%.10g", delimiter="\t")
        print(f"    Wrote {pcs_path}")

        if lib.raw_deconvolved is not None:
            raw_imp_path = Path(f"{args.prefix}{gtag}_hrfraw_imp.tsv")
            write_tsv(raw_imp_path, lib.raw_deconvolved)
            print(f"    Wrote {raw_imp_path}   [impulse-response cubic recon]")

        final_library, final_label = lib.raw, "raw (duration-convolved)"
        if lib.fitted_deconvolved is not None:
            final_library, final_label = lib.fitted_deconvolved, "deconv+gamma"
        elif lib.fitted is not None:
            final_library, final_label = lib.fitted, "gamma (duration-convolved)"
        elif lib.raw_deconvolved is not None:
            final_library, final_label = lib.raw_deconvolved, "deconv raw"
        lib_path = Path(f"{args.prefix}{gtag}_hrflibrary.tsv")
        write_tsv(lib_path, final_library)
        print(f"    Wrote {lib_path}   [final library — {final_label}]")

        # R² volume is written once per call.  For stacked mode the
        # underlying r2 is the full-model R² across all groups (the
        # tiled stack collapses back to one map per voxel).
        write_r2_volume(r2, mask_flat, volume_shape, r2_path, affine)
        print(f"    Wrote {r2_path}")

        qc_artifacts = write_qc_artifacts(
            lib, lag_times, label_for_output, args.prefix, gtag, verbose=args.verb
        )
        for kind, path in qc_artifacts.items():
            print(f"    Wrote {path}   [QC: {kind}]")

        metadata["groups"].append(
            {
                "label": label_for_output,
                "n_lags": int(n_lags),
                "median_duration_s": (
                    float(event_durations_arr[0])
                    if event_durations_arr.size == 1
                    else event_durations_arr.tolist()
                ),
                "n_selected_voxels": int(lib.selected_voxels.size),
                "variance_explained": lib.svd.variance_explained.tolist(),
                "eigvals_top10": lib.svd.eigvals[:10].tolist(),
                "gamma_params": lib.gamma_params,
                "gamma_params_deconvolved": lib.gamma_params_deconvolved,
                "duration_convolved": bool(lib.duration_convolved),
                "deconvolution": (
                    {
                        "method": "wiener",
                        "duration_s": float(deconv_duration),
                        "snr": float(args.deconv_snr),
                    }
                    if deconv_duration is not None
                    else None
                ),
                "qc_artifacts": qc_artifacts,
            }
        )
        return lib

    # ------------------------------------------------------------------
    # Branch A: stacked mode — single library across all groups.
    # ------------------------------------------------------------------
    if use_stacking:
        # Reshape and row-stack per-group task betas.
        # betas_full[:, :n_groups*n_lags] is the task block; reshape to
        # (n_voxels, n_groups, n_lags) and stack groups along the voxel
        # axis → (n_voxels * n_groups, n_lags).
        task_betas_3d = betas_full[:, : n_groups * n_lags].reshape(n_voxels, n_groups, n_lags)
        # Order: all of group 0 first, then group 1, ... — matches the
        # per-group order so refit_weights stacks line up.
        stacked_betas = np.concatenate([task_betas_3d[:, g, :] for g in range(n_groups)], axis=0)
        stacked_r2 = np.tile(r2, n_groups)

        # Compute shared PCs (one SVD on the stacked betas, same voxel
        # selection + zero-norm filter as derive_library will do).
        refit_weights_stacked = None
        if args.refit_pcs == "on":
            from fastfuncstuff.design.hrf_derive import select_voxels as _sel_fn

            sel_s = _sel_fn(
                stacked_r2,
                threshold=args.r2_threshold,
                max_voxels=args.max_voxels,
                seed=args.seed,
            )
            sel_betas_s = stacked_betas[sel_s]
            sel_norms_s = np.linalg.norm(sel_betas_s, axis=1)
            sel_alive_s = sel_norms_s > 1e-8 * max(
                float(np.median(sel_norms_s[sel_norms_s > 0])) if (sel_norms_s > 0).any() else 1.0,
                1.0,
            )
            sel_s = sel_s[sel_alive_s]
            sel_betas_s = stacked_betas[sel_s]

            svd_shared = svd_decompose(
                sel_betas_s,
                n_pcs=args.n_pcs,
                unit_normalize=True,
                sign_align=True,
            )
            print(
                f"\n  --- stacked SVD: {sel_s.size} voxel-group rows; "
                f"PC variance {svd_shared.variance_explained[:3] * 100} ---"
            )

            # Per-group refit using SHARED PCs; concatenate the resulting
            # per-voxel × K weight matrices in the same group order as
            # stacked_betas so the rows align.
            per_group_refit = []
            for g in range(n_groups):
                rw_g = _compute_refit_weights(g, svd_shared.pcs)
                per_group_refit.append(rw_g)
            refit_weights_stacked = np.concatenate(per_group_refit, axis=0)

        # Single library duration: uniform → that value; else None
        # (deconvolution gets skipped, see check above).
        stacked_duration = float(pooled_durations[0]) if uniform_duration else None
        if args.deconvolve_duration == "on":
            deconv_for_stack = (
                stacked_duration if stacked_duration and stacked_duration > 0 else None
            )
        elif args.deconvolve_duration == "off":
            deconv_for_stack = None
        else:  # "auto"
            deconv_for_stack = (
                stacked_duration if (stacked_duration and stacked_duration > 1.5 * 0.1) else None
            )

        print(
            f"\n  --- stacked library across {n_groups} groups "
            f"({stacked_betas.shape[0]:,} voxel-group rows) ---"
        )
        _derive_and_write_one_library(
            betas_in=stacked_betas,
            r2_in=stacked_r2,
            refit_weights_in=refit_weights_stacked,
            deconv_duration=deconv_for_stack,
            gtag="",
            label_for_output="stacked",
            event_durations_arr=np.asarray(pooled_durations, dtype=float),
        )

    # ------------------------------------------------------------------
    # Branch B: separate libraries (n_groups == 1 OR -split-separate).
    # ------------------------------------------------------------------
    else:
        for g in range(n_groups):
            col_lo, col_hi = g * n_lags, (g + 1) * n_lags
            betas_g = betas_full[:, col_lo:col_hi]
            print(f"\n  --- group '{group_labels[g]}' ({col_lo}-{col_hi - 1}) ---")

            # NSD refinement using PCs derived FROM THIS GROUP only.
            refit_weights_g = None
            if args.refit_pcs == "on":
                from fastfuncstuff.design.hrf_derive import select_voxels as _sel_fn

                sel_g = _sel_fn(
                    r2,
                    threshold=args.r2_threshold,
                    max_voxels=args.max_voxels,
                    seed=args.seed,
                )
                sel_betas = betas_g[sel_g]
                sel_norms = np.linalg.norm(sel_betas, axis=1)
                sel_alive = sel_norms > 1e-8 * max(
                    float(np.median(sel_norms[sel_norms > 0])) if (sel_norms > 0).any() else 1.0,
                    1.0,
                )
                sel_g = sel_g[sel_alive]
                sel_betas = betas_g[sel_g]
                svd_pre = svd_decompose(
                    sel_betas,
                    n_pcs=args.n_pcs,
                    unit_normalize=True,
                    sign_align=True,
                )
                print(
                    f"    NSD refinement: refitting with {args.n_pcs} PC-basis "
                    f"regressors (shared across {len(per_run_data)} runs + "
                    f"per-run polort={polort_resolved})"
                )
                refit_weights_g = _compute_refit_weights(g, svd_pre.pcs)

            # Per-group duration deconvolution: each group uses its own
            # event duration.  This is the only valid path for groups
            # with different durations (stacking can't handle that).
            group_duration = float(pooled_durations[g])
            if args.deconvolve_duration == "on":
                deconv_for_g = group_duration if group_duration > 0 else None
            elif args.deconvolve_duration == "off":
                deconv_for_g = None
            else:  # "auto"
                deconv_for_g = group_duration if group_duration > 1.5 * 0.1 else None

            gtag = "" if n_groups == 1 else f"_group_{group_labels[g]}"
            _derive_and_write_one_library(
                betas_in=betas_g,
                r2_in=r2,
                refit_weights_in=refit_weights_g,
                deconv_duration=deconv_for_g,
                gtag=gtag,
                label_for_output=group_labels[g],
                event_durations_arr=np.asarray([group_duration], dtype=float),
            )

    # Roll up the per-group duration_convolved flag: top-level True only
    # if EVERY group is still duration-convolved (i.e. deconvolution was
    # skipped everywhere).  Any group with deconvolved output flips this.
    metadata["duration_convolved"] = all(
        bool(g.get("duration_convolved", True)) for g in metadata["groups"]
    )

    meta_path = Path(f"{args.prefix}_metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\n  Wrote {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
