"""CLI for GPU-accelerated motion correction (ffs_moco).

Command: ffs_moco (registered as entry point in pyproject.toml)

Usage:
    ffs_moco -input epi.nii.gz -prefix epi_mc.nii.gz -1Dfile motion.1D
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_utils import (
    add_batch_args,
    add_device_arg,
    add_verbose_arg,
    collect_batch_jobs,
    parse_prefix,
    run_batch_jobs,
    setup_device,
    spinner,
)
from fastfuncstuff.processing.affine import (
    matrix_to_params,
    save_matrix_1D,
    voxel_matrix_to_dicom,
)
from fastfuncstuff.processing.ffs_moco import (
    MocoConfig,
    _blur_volume,
    moco,
    moco_spacetime,
    resample_timeseries,
    save_maxdisp_1D,
    save_moco_1D,
    save_moco_dfile,
)
from fastfuncstuff.processing.io import (
    derive_prefixed_output_path,
    load_image,
    save_first_last,
    save_image,
    save_tsnr,
)
from fastfuncstuff.processing.locomoco import normalize_axis_argv, resolve_pe_axis
from fastfuncstuff.processing.shiftcorr import (
    apply_shift,
    estimate_shifts,
    fold_shift_into_matrices,
    save_shift_tables,
    shift_table_paths,
)
from fastfuncstuff.utils import REGISTRATION_TF32

# Sentinel for `-save_mean` / `-save_max` / `-save_min` given with no value:
# derive the path from -prefix.
_MEAN_FROM_PREFIX = "\x00from_prefix"

# Temporal reductions of the corrected series, in the order they are written.
# max/min exist for coverage, not contrast: motion carries edge voxels out of the
# FoV, where the resampler writes 0, so the max over time is the union of what
# was ever imaged (the most complete alignment target) and the min is the
# intersection (0 wherever ANY volume lost the voxel — an exact "analysable
# everywhere" mask). Both compose down a chain of maxes/mins; a mean does not.
_TEMPORAL_REDUCTIONS: tuple[tuple[str, str], ...] = (
    ("save_mean", "mean"),
    ("save_max", "max"),
    ("save_min", "min"),
)

# Sentinel for `-save_weight` given with no value: derive the paths from -prefix.
_WEIGHT_FROM_PREFIX = "\x00weight_from_prefix"


def _sibling(path: str, prefix: str) -> str:
    """Return ``path`` with ``prefix`` prepended to its basename (dir preserved)."""
    import os

    d, base = os.path.split(path)
    return os.path.join(d, prefix + base)


def _run_estimation(args, data, config, header_info, base_vol, input_file, verb):
    """Dispatch to slice-timing-aware moco (space-time) or plain moco.

    When -tpattern is given, resolve the TR (flag > JSON RepetitionTime > header)
    and run the alternating joint slice-timing + motion estimator; otherwise the
    standard estimator.
    """
    if config.slice_times is None:
        return moco(data, config, header_info=header_info, base_vol=base_vol)

    tr = config.st_tr
    if tr is None and args.tpattern.endswith(".json"):
        import json

        jdata = json.loads(Path(args.tpattern).read_text())
        if "RepetitionTime" in jdata:
            tr = float(jdata["RepetitionTime"])
    if tr is None:
        from fastfuncstuff.io.afni import get_tr_from_file

        hdr_tr = get_tr_from_file(input_file)
        if hdr_tr and hdr_tr > 0:
            tr = hdr_tr
    if tr is None:
        print(
            "Error: -tpattern needs a TR; none in JSON/header, pass -TR.",
            file=sys.stderr,
        )
        sys.exit(1)

    from dataclasses import replace as _replace

    if config.st_tr != tr:
        config = _replace(config, st_tr=tr)
    if verb >= 1:
        print(f"Slice-timing-aware moco: {len(config.slice_times)} slices, TR={tr:.4f}s")
    return moco_spacetime(data, config, header_info=header_info, base_vol=base_vol)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="ffs_moco",
        description="GPU-accelerated motion correction for fMRI/fNIRS timeseries "
        "(inspired by 3dvolreg)",
        epilog="""Examples:
  # Standard motion correction:
  ffs_moco -input epi.nii.gz -prefix epi_mc.nii.gz -1Dfile motion.1D

  # With affine matrix output:
  ffs_moco -input epi.nii.gz -prefix epi_mc.nii.gz -1Dmatrix_save mat.aff12.1D

  # Use LPA cost function:
  ffs_moco -input epi.nii.gz -prefix epi_mc.nii.gz -cost lpa

  # Two-pass with base volume 10:
  ffs_moco -input epi.nii.gz -prefix epi_mc.nii.gz -base 10 -twopass

  # Multi-echo: estimate from echo 1, apply to all echoes (writes e1_/e2_/e3_):
  ffs_moco -input e1.nii.gz e2.nii.gz e3.nii.gz -reg_echo 1 -prefix mc.nii.gz

  # Estimate from the cross-echo mean instead:
  ffs_moco -input e?.nii.gz -reg_echo mean -prefix mc.nii.gz -1Dfile motion.1D

  # Multi-echo 3-D EPI: strip the TE-dependent partition-axis shift first, then
  # motion-correct, in one resample; save the per-echo TOTAL transforms:
  ffs_moco -input e1.nii.gz e2.nii.gz e3.nii.gz -reg_echo 1 -prefix mc.nii.gz \\
      -me_3depi -axis IS -echo_times 7.61 21.71 35.81 -shift_ordering ascending \\
      -1Dfile motion.1D -1Dfile_shiftcorr motion_shiftcorr.1D \\
      -1Dmatrix_shiftcorr mat_shiftcorr.aff12.1D -save_shifts mc

  # Drop the first 4 and last 2 volumes of every input:
  ffs_moco -input epi.nii.gz -prefix epi_mc.nii.gz -skip_first 4 -skip_last 2
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Input/Output ---
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "-input",
        nargs="+",
        dest="input_file",
        metavar="FILE",
        help="4D timeseries (.nii/.nii.gz) [required unless -batch]. Pass several "
        "files (one per echo, in echo order) together with -reg_echo for "
        "multi-echo registration: motion is estimated once and applied to "
        "every echo.",
    )
    add_batch_args(
        io_group,
        tool="ffs_moco",
        what="motion corrections",
        example="-input a.nii -base 0 -prefix a_mc.nii -dfile a.1D",
        skip_note="-prefix / -save_mean / -1Dfile / -1Dmatrix_save / -dfile / "
        "-maxdisp1D / -iterfile / QC flags",
    )
    io_group.add_argument(
        "-reg_echo",
        "-reg-echo",
        dest="reg_echo",
        default=None,
        metavar="N|mean",
        help="Multi-echo: which echo drives the estimation. An integer N "
        "(1-based) estimates motion from that echo; 'mean' estimates from the "
        "per-timepoint mean across echoes. The transforms are then applied to "
        "every echo. Required when more than one -input file is given.",
    )
    io_group.add_argument(
        "-prefix",
        default=None,
        help="Output prefix for the corrected timeseries. Omit to skip writing "
        "it (e.g. when you only want motion params or -save_mean). With "
        "multiple echoes each output is prefixed with eN_ (e1_, e2_, ...).",
    )
    io_group.add_argument(
        "-base",
        default="0",
        help="Base volume index (default: 0), or path to external 3D file. "
        "The index is into the series after -skip_first/-skip_last trimming.",
    )
    io_group.add_argument(
        "-skip_first",
        "-skip-first",
        dest="skip_first",
        type=int,
        default=0,
        help="Drop this many volumes from the start of every input before "
        "registration (default: 0). Alternative to a [n..$] sub-brick selector "
        "for globbed multi-echo inputs.",
    )
    io_group.add_argument(
        "-skip_last",
        "-skip-last",
        dest="skip_last",
        type=int,
        default=0,
        help="Drop this many volumes from the end of every input (default: 0).",
    )

    # --- Method ---
    method_group = parser.add_argument_group("Method")
    method_group.add_argument(
        "-cost",
        choices=["wls", "lpa", "quad"],
        default="wls",
        help="Cost function: wls=weighted least squares "
        "(default), lpa=local Pearson absolute, quad=quadrature phase",
    )
    method_group.add_argument(
        "-maxiter",
        type=int,
        default=23,
        help="Max GN iterations per volume (default: 23, matches 3dvolreg)",
    )
    method_group.add_argument(
        "-dxy_thresh",
        type=float,
        default=0.01,
        help="Translation convergence threshold in voxels (default: 0.01, matches 3dvolreg -x_thresh)",
    )
    method_group.add_argument(
        "-dph_thresh",
        type=float,
        default=0.02,
        help="Rotation convergence threshold in degrees (default: 0.02, matches 3dvolreg -rot_thresh)",
    )
    method_group.add_argument("-twopass", action="store_true", help="Coarse blur + fine pass")
    method_group.add_argument(
        "-chain_init",
        dest="chain_init",
        action="store_true",
        help="Warm-start each volume's estimate from the previous volume. "
        "Faster but tends to under-detect TR-to-TR motion; off by default so "
        "every volume is estimated independently (matches 3dvolreg sensitivity).",
    )
    method_group.add_argument(
        "-chain-init",
        dest="chain_init",
        action="store_true",
        help="Alias for -chain_init.",
    )
    # Deprecated no-ops: chaining is now off by default.
    method_group.add_argument(
        "-no_chain", dest="no_chain", action="store_true", help=argparse.SUPPRESS
    )
    method_group.add_argument(
        "-nochain", dest="no_chain", action="store_true", help=argparse.SUPPRESS
    )
    method_group.add_argument("-automask", action="store_true", help="Use automask for weighting")
    method_group.add_argument(
        "-weight_automask",
        action="store_true",
        help="Use automask × continuous weight (tight mask + quality weighting)",
    )
    method_group.add_argument(
        "-blur",
        type=float,
        default=0.0,
        help="Pre-blur FWHM in mm for estimation (default: 0)",
    )
    method_group.add_argument(
        "-fast",
        action="store_true",
        help="Fast mode: fixed iterations, no convergence check (runs exactly -maxiter)",
    )
    method_group.add_argument(
        "-workhard",
        action="store_true",
        help="Spend the speed on accuracy: 5x stricter convergence thresholds "
        "and double the max iterations. Useful for high-motion or demanding runs.",
    )
    method_group.add_argument(
        "-no_compile",
        dest="no_compile",
        action="store_true",
        help="Disable torch.compile for hot path (default: compile on CUDA)",
    )
    method_group.add_argument(
        "-no-compile",
        dest="no_compile",
        action="store_true",
        help="Alias for -no_compile.",
    )

    # --- Reweight (data-driven weight refinement) ---
    rw_group = parser.add_argument_group("Reweight")
    rw_group.add_argument(
        "-reweight",
        action="store_true",
        help="Pre-pass that softly downweights regions whose residual improves less "
        "than the brain-wide trend under an initial whole-head alignment. Like "
        "-twopass, it looks at the data first, then reruns estimation with the "
        "refined weight.",
    )
    rw_group.add_argument(
        "-reweight_tolerance",
        "-reweight-tolerance",
        dest="reweight_tolerance",
        type=float,
        default=1.1,
        help="Leave a region unchanged when its post/pre residual ratio is no "
        "more than this multiple of the brain-wide ratio; larger values are "
        "more conservative (default: 1.1; must be >= 1).",
    )
    rw_group.add_argument(
        "-save_weight",
        "-save-weight",
        dest="save_weight",
        nargs="?",
        const=_WEIGHT_FROM_PREFIX,
        default=None,
        metavar="PREFIX",
        help="Save the original weight, the soft reweighted weight, and a binary "
        "map of downweighted voxels. With no value, derives the paths "
        "from -prefix.",
    )

    # --- Interpolation ---
    interp_group = parser.add_argument_group("Interpolation")
    interp_group.add_argument(
        "-interp",
        choices=["linear", "cubic", "quintic", "heptic", "wsinc5"],
        default="heptic",
        help="During estimation (default: heptic)",
    )
    interp_group.add_argument(
        "-final",
        choices=["linear", "cubic", "quintic", "heptic", "wsinc5"],
        default="wsinc5",
        dest="final_interp",
        help="For output (default: wsinc5)",
    )
    interp_group.add_argument(
        "-no_shear",
        dest="no_shear",
        action="store_true",
        help="Disable shear-based rigid resampling for the final pass "
        "(AFNI THD_rota_vol method); use the general affine resampler instead.",
    )
    interp_group.add_argument(
        "-no-shear", dest="no_shear", action="store_true", help=argparse.SUPPRESS
    )

    st_group = parser.add_argument_group("Slice timing (space-time realignment)")
    st_group.add_argument(
        "-tpattern",
        default=None,
        help="Per-slice acquisition timing (text: one time/line in seconds, or "
        "BIDS JSON with SliceTiming). Enables slice-timing-AWARE motion "
        "correction: motion is estimated on data with the slice-timing-vs-BOLD "
        "confound removed (reduces stimulus-correlated motion), and the aligned "
        "output applies motion+slice-timing in one interpolation. Requires a TR.",
    )
    st_group.add_argument(
        "-TR",
        type=float,
        default=None,
        help="Repetition time (seconds) for -tpattern; read from the NIfTI "
        "header (or JSON RepetitionTime) when omitted.",
    )
    st_group.add_argument(
        "-tzero",
        type=float,
        default=None,
        help="Reference time within the TR to align slices to (seconds). "
        "Default: mean of slice times.",
    )
    st_group.add_argument(
        "-tinterp",
        choices=["linear", "cubic", "wsinc5", "wsinc9"],
        default="cubic",
        help="Temporal interpolation kernel for -tpattern (default cubic).",
    )
    st_group.add_argument(
        "-st_iters",
        "-st-iters",
        type=int,
        default=2,
        dest="st_iters",
        help="Space-time outer refinement iterations (default 2; 1 == "
        "tshift-then-moco, each extra iter re-estimates motion on freshly "
        "joint-corrected data).",
    )

    sc_group = parser.add_argument_group(
        "Multi-echo 3-D EPI partition-axis shift correction (-me_3depi)"
    )
    sc_group.add_argument(
        "-me_3depi",
        "-me-3depi",
        dest="me_3depi",
        action="store_true",
        help="Before estimating motion, remove the TE-dependent apparent shift along "
        "the partition (slow phase-encode) axis that a frequency drift over the 3-D "
        "shot produces. It is a DIFFERENT shift for every echo, which is exactly what "
        "one-pose-for-all-echoes multi-echo moco cannot represent, and a bulk "
        "translation that would confuse ffs_locomoco downstream. Estimated per volume "
        "by whole-volume inter-echo cross-correlation, then FOLDED into each echo's "
        "motion matrix so the data is still resampled only once. Needs -axis and "
        "several -input echoes.",
    )
    sc_group.add_argument(
        "-axis",
        default=None,
        help="Partition (slow phase-encode) axis for -me_3depi: a direction code "
        "AP/PA/LR/RL/IS/SI or an axis letter x/y/z (i/j/k).",
    )
    sc_group.add_argument(
        "-echo_times",
        "-echo-times",
        dest="echo_times",
        nargs="+",
        type=float,
        default=None,
        help="Echo times in MILLISECONDS, one per -input echo. Enables the TE "
        "regression: the shift is fit as a line in TE per volume and applied through "
        "the origin, so echo 1 is corrected too and only the echo-COMMON offset (the "
        "fit's intercept) is left for rigid motion correction. Without it, the raw "
        "cumulative shifts are applied and echo 1 is the fixed reference.",
    )
    sc_group.add_argument(
        "-shift_ordering",
        "-shift-ordering",
        dest="shift_ordering",
        choices=["ascending", "descending", "unknown"],
        default="unknown",
        help="Partition view ordering. A known ordering fixes the sign of the drift "
        "and halves the search — but over a timeseries respiration swings the frequency "
        "BOTH ways, and a constrained range pins every volume that wanted the other sign "
        "at exactly zero. Leave this at 'unknown' unless you know the drift is "
        "one-directional.",
    )
    sc_group.add_argument(
        "-shift_max",
        "-shift-max",
        dest="shift_max",
        type=float,
        default=5.0,
        help="Shift search half-range in voxels.",
    )
    sc_group.add_argument(
        "-corr_extent",
        "-corr-extent",
        dest="corr_extent",
        choices=["full", "inner_half"],
        default="full",
        help="Which voxels the inter-echo correlation sees. 'full' uses the whole "
        "volume, tapering only the outermost few partitions a trial shift fills with "
        "replicated content; 'inner_half' is the reference script's central-half crop "
        "(which assumes the anatomy is centred on the partition axis).",
    )
    sc_group.add_argument(
        "-shift_weight",
        "-shift-weight",
        dest="shift_weight",
        choices=["none", "signal"],
        default="none",
        help="Voxel weighting for the correlation. 'none' is a plain Pearson r over "
        "the whole volume and is enough here because the patch IS the volume; "
        "'signal' softly weights by mean echo-1 intensity.",
    )
    sc_group.add_argument(
        "-save_shifts",
        "-save-shifts",
        dest="save_shifts",
        default=None,
        help="Stem for the -me_3depi shift/QC tables: {stem}_shifts_xcorr.1D, "
        "_shifts_applied.1D, _corr.1D and (with -echo_times) _te_fit.1D.",
    )

    # --- Output files ---
    out_group = parser.add_argument_group("Output files")
    out_group.add_argument("-1Dfile", default=None, help="Save 6-column motion parameters (.1D)")
    out_group.add_argument("-1Dmatrix_save", default=None, help="Save affine matrices (.aff12.1D)")
    out_group.add_argument(
        "-1Dfile_shiftcorr",
        "-1Dfile-shiftcorr",
        dest="onedfile_shiftcorr",
        default=None,
        help="-me_3depi: save the TOTAL per-echo motion parameters — rigid motion "
        "WITH that echo's partition-axis shift folded in — one eN_ prefixed 6-column "
        ".1D per echo. Unlike -1Dfile (one shared rigid pose) these differ across "
        "echoes, which is the whole point.",
    )
    out_group.add_argument(
        "-1Dmatrix_shiftcorr",
        "-1Dmatrix-shiftcorr",
        dest="onedmatrix_shiftcorr",
        default=None,
        help="-me_3depi: the same total per-echo transforms as affine matrices, one "
        "eN_ prefixed .aff12.1D per echo. These are exactly the matrices the "
        "resampler used.",
    )
    out_group.add_argument(
        "-dfile",
        default=None,
        help="Save 9-column diagnostic file. With -reg_echo mean the final RMS "
        "column is 0 (no single echo to measure against); with -reg_echo N it "
        "reports echo N.",
    )
    out_group.add_argument("-maxdisp1D", default=None, help="Save max displacement per volume")
    out_group.add_argument("-iterfile", default=None, help="Save iterations per volume (.1D)")
    out_group.add_argument(
        "-save_mean",
        nargs="?",
        const=_MEAN_FROM_PREFIX,
        default=None,
        metavar="PREFIX",
        help="Save the temporal mean of the corrected series. With no value, "
        "derives mean_{prefix} from -prefix (legacy behavior); give a PREFIX to "
        "write it there (and you may then omit -prefix to skip the full series).",
    )
    out_group.add_argument(
        "-save_max",
        "-save-max",
        dest="save_max",
        nargs="?",
        const=_MEAN_FROM_PREFIX,
        default=None,
        metavar="PREFIX",
        help="Save the temporal MAX of the corrected series (max_{prefix} with no "
        "value). Voxels that motion carried outside the FoV are 0 in the volumes "
        "that lost them, so the max is the union of everything ever imaged — a "
        "fuller alignment target than the mean, which dims those edges.",
    )
    out_group.add_argument(
        "-save_min",
        "-save-min",
        dest="save_min",
        nargs="?",
        const=_MEAN_FROM_PREFIX,
        default=None,
        metavar="PREFIX",
        help="Save the temporal MIN of the corrected series (min_{prefix} with no "
        "value): 0 wherever ANY volume lost the voxel, so >0 is exactly the region "
        "with complete data for every timepoint.",
    )
    out_group.add_argument(
        "-save_first_last",
        "-save-first-last",
        dest="save_first_last",
        action="store_true",
        help="Save the first & last corrected volumes as one switchable file "
        "firstlast_{prefix} — flip between them in a viewer to see how well the "
        "correction worked. Requires -prefix (used to name the file).",
    )
    out_group.add_argument(
        "-save_first_last_diff",
        "-save-first-last-diff",
        dest="save_first_last_diff",
        action="store_true",
        help="Like -save_first_last but the file also carries a third volume, the "
        "difference (last - first): firstlastdiff_{prefix}. Written separately so "
        "the signed difference keeps its own scale.",
    )
    out_group.add_argument(
        "-save_tsnr",
        "-save-tsnr",
        dest="save_tsnr",
        action="store_true",
        help="Save a temporal-SNR map (temporal mean / temporal std) of the "
        "corrected series as tsnr_{prefix} — a QC map of where the correction "
        "left clean signal. Requires -prefix (used to name the file).",
    )
    out_group.add_argument(
        "-save_initial",
        "-save-initial",
        dest="save_initial",
        action="store_true",
        help="Also emit the requested QC files (first/last, diff, tSNR) for the "
        "ORIGINAL uncorrected data (e.g. firstlast_initial_{prefix}, "
        "tsnr_initial_{prefix}), for a before/after comparison. Defaults to plain "
        "first/last when no other QC flag is given.",
    )

    # --- Hardware ---
    hw_group = parser.add_argument_group("Hardware")
    add_device_arg(
        hw_group,
        extra="On Apple Silicon, MPS is recommended for typical full-size brain volumes; CPU may win on small jobs.",
    )
    add_verbose_arg(hw_group, default=1)
    hw_group.add_argument(
        "-debug_memory",
        action="store_true",
        help="Print VRAM usage vs. prediction after registration and resampling loops",
    )

    # `-axis -k` looks like an option flag to argparse; rewrite the axis token.
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(normalize_axis_argv(list(argv), {"-axis"}))
    return args


def _select_device(arg_device: str | None) -> torch.device:
    """Resolve the -device flag (or auto-detect)."""
    return setup_device(arg_device, tf32=REGISTRATION_TF32)


def _parse_base(args: argparse.Namespace, verb: int) -> tuple[torch.Tensor | None, int]:
    """Resolve -base into (external base volume or None, base index)."""
    try:
        return None, int(args.base)
    except ValueError:
        if verb >= 1:
            print(f"Loading external base: {args.base}")
        with spinner(f"Loading {Path(args.base).name}"):
            base_vol, _ = load_image(args.base)
        if base_vol.ndim == 4:
            base_vol = base_vol[0]
        return base_vol, 0


def _load_trimmed(path: str, skip_first: int, skip_last: int, verb: int):
    """Load a 4D series and drop -skip_first / -skip_last volumes from the ends."""
    with spinner(f"Loading {Path(path).name}"):
        data, header_info = load_image(path)
    if data.ndim != 4:
        print(f"Error: input must be 4D, got {data.ndim}D ({path})", file=sys.stderr)
        sys.exit(1)
    nt = data.shape[0]
    if skip_first < 0 or skip_last < 0:
        print("Error: -skip_first and -skip_last must be non-negative.", file=sys.stderr)
        sys.exit(1)
    if skip_first + skip_last >= nt:
        print(
            f"Error: -skip_first={skip_first} + -skip_last={skip_last} removes all "
            f"{nt} volumes of {path}.",
            file=sys.stderr,
        )
        sys.exit(1)
    if skip_first or skip_last:
        data = data[skip_first : nt - skip_last]
        if verb >= 1:
            print(
                f"  Trimmed {path}: {nt} -> {data.shape[0]} volumes "
                f"(-skip_first {skip_first}, -skip_last {skip_last})"
            )
    return data, header_info


def _load_echo_mean(
    input_files: list[str],
    skip_first: int,
    skip_last: int,
    verb: int,
    est=None,
    axis: int | None = None,
):
    """Per-timepoint mean across echoes, accumulated one echo at a time.

    Loads echoes sequentially and sums in place so peak memory stays at roughly
    two 4D volumes rather than N — the running accumulator plus the current echo.
    With an ``est`` from -me_3depi, each echo is shift-corrected BEFORE it enters
    the sum: averaging echoes that sit at different partition offsets would blur
    the very structure the rigid estimate needs.
    """
    if verb >= 1:
        print(f"Building reg series: mean of {len(input_files)} echoes")
    acc = None
    header_info = None
    for i, path in enumerate(input_files):
        echo, hdr = _load_trimmed(path, skip_first, skip_last, verb)
        if est is not None:
            echo = apply_shift(echo, est.applied[:, i], axis)
        if acc is None:
            acc = echo.float()
            header_info = hdr
        else:
            if echo.shape != acc.shape:
                print(
                    f"Error: echo {path} shape {tuple(echo.shape)} does not match "
                    f"the first echo {tuple(acc.shape)}.",
                    file=sys.stderr,
                )
                sys.exit(1)
            acc += echo.float()
        del echo
    acc /= len(input_files)
    return acc, header_info


def _build_config(
    args: argparse.Namespace,
    device: torch.device,
    verb: int,
    skip_resample: bool,
    base_index: int,
) -> MocoConfig:
    """Assemble a MocoConfig from parsed args (shared by single/multi-echo)."""
    # -workhard: trade the speed headroom for accuracy — stricter convergence
    # and twice the iteration budget.
    max_iter = args.maxiter * 2 if args.workhard else args.maxiter
    dxy_thresh = args.dxy_thresh * 0.2 if args.workhard else args.dxy_thresh
    dph_thresh = args.dph_thresh * 0.2 if args.workhard else args.dph_thresh
    if args.workhard and verb >= 1:
        print(f"  -workhard: max_iter={max_iter}, dxy={dxy_thresh:g}, dph={dph_thresh:g}")

    slice_times = None
    if getattr(args, "tpattern", None) is not None:
        from fastfuncstuff.processing.slicetime import load_slice_timing

        slice_times = load_slice_timing(args.tpattern)

    return MocoConfig(
        skip_resample=skip_resample,
        base_index=base_index,
        cost=args.cost,
        slice_times=slice_times,
        st_tr=args.TR,
        st_tzero=args.tzero,
        st_tinterp=args.tinterp,
        st_iters=args.st_iters,
        interp=args.interp,
        final_interp=args.final_interp,
        max_iter=max_iter,
        twopass=args.twopass,
        blur_fwhm=args.blur,
        chain_init=args.chain_init,
        use_shear=not args.no_shear,
        automask=args.automask,
        weight_automask=args.weight_automask,
        dxy_thresh=dxy_thresh,
        dph_thresh=dph_thresh,
        fixed_iter=args.fast,
        compile=not args.no_compile,
        device=str(device),
        verb=verb,
        debug_memory=args.debug_memory,
        reweight=args.reweight,
        reweight_tolerance=args.reweight_tolerance,
    )


def _want_qc(args) -> bool:
    """True if any QC-output flag (first/last, diff, tSNR, initial) was requested."""
    return bool(
        args.save_first_last or args.save_first_last_diff or args.save_tsnr or args.save_initial
    )


def _want_corrected_qc(args) -> bool:
    """True if a QC output needs the resampled corrected series (not just the raw)."""
    return bool(args.save_first_last or args.save_first_last_diff or args.save_tsnr)


def _write_qc(args, corrected, original, base_path, header_info, verb) -> None:
    """Write the requested QC files (first/last, difference, tSNR) for one series.

    ``corrected`` is the aligned 4-D series (named files use it), ``original`` is
    the matching pre-correction series (for -save_initial). ``base_path`` is the
    corrected-series output path the files are named after.
    """
    want_plain = args.save_first_last
    want_diff = args.save_first_last_diff

    if want_plain:
        save_first_last(corrected, base_path, header_info, include_diff=False, verb=verb)
    if want_diff:
        save_first_last(corrected, base_path, header_info, include_diff=True, verb=verb)
    if args.save_tsnr:
        save_tsnr(corrected, base_path, header_info, verb=verb)

    if args.save_initial:
        # -save_initial mirrors whichever QC output(s) are active onto the raw
        # data; alone it means plain first/last on the raw data.
        if want_plain or not (want_diff or args.save_tsnr):
            save_first_last(
                original, base_path, header_info, include_diff=False, initial=True, verb=verb
            )
        if want_diff:
            save_first_last(
                original, base_path, header_info, include_diff=True, initial=True, verb=verb
            )
        if args.save_tsnr:
            save_tsnr(original, base_path, header_info, initial=True, verb=verb)


def _save_estimation_outputs(args, result, header_info, verb) -> None:
    """Save the single-instance outputs (one per run, independent of echo).

    Covers the motion parameters, affine matrices, dfile, max-displacement,
    iteration counts, and reweight weight/diagnostic images. The corrected series
    and its mean are handled per echo by the caller.
    """
    # Motion parameters
    onedfile = getattr(args, "1Dfile", None)
    if onedfile is not None:
        save_moco_1D(result.params, onedfile)
        if verb >= 1:
            print(f"Saved 1Dfile: {onedfile}")

        # Reweight diagnostic: the pre-reweight (consensus) motion estimated with
        # the original weight, in the same AFNI 6-column format, for comparison
        # against the final post-reweight params above.
        if args.reweight and result.params_preweight is not None:
            pre_path = _sibling(onedfile, "preweight_")
            save_moco_1D(result.params_preweight, pre_path)
            if verb >= 1:
                print(f"Saved preweight params: {pre_path}")

    # Affine matrices
    matrix_save = getattr(args, "1Dmatrix_save", None)
    if matrix_save is not None:
        save_matrix_1D(
            result.matrices_dicom,
            matrix_save,
            header="ffs_moco matrices (DICOM-to-DICOM, row-by-row):",
        )
        if verb >= 1:
            print(f"Saved matrices: {matrix_save}")

    # Diagnostic file
    if args.dfile is not None:
        save_moco_dfile(result.params, result.rms_before, result.rms_after, args.dfile)
        if verb >= 1:
            print(f"Saved dfile: {args.dfile}")

    # Max displacement
    if args.maxdisp1D is not None:
        save_maxdisp_1D(result.max_displacement, args.maxdisp1D)
        if verb >= 1:
            print(f"Saved maxdisp: {args.maxdisp1D}")

    # Iterations per volume
    if args.iterfile is not None:
        with open(args.iterfile, "w") as f:
            for it in result.n_iters:
                f.write(f"{it}\n")
        if verb >= 1:
            print(f"Saved iterfile: {args.iterfile}")

    # Reweight weight images + downweighted-voxel map (single, estimated once).
    if args.save_weight is not None:
        if not args.reweight:
            print(
                "Warning: -save_weight given without -reweight; nothing to save.",
                file=sys.stderr,
            )
        elif result.weight_refined is None or result.patch_labels is None:
            print(
                "Warning: reweight did not run (empty weight / low-motion guard); "
                "skipping -save_weight.",
                file=sys.stderr,
            )
        else:
            if args.save_weight is _WEIGHT_FROM_PREFIX:
                if args.prefix is None:
                    print(
                        "Error: -save_weight with no value needs -prefix to derive "
                        "the paths; pass -save_weight PREFIX instead.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                pfx = parse_prefix(args.prefix)
            else:
                pfx = parse_prefix(args.save_weight)
            w_orig = pfx.with_suffix("weight_orig")
            w_new = pfx.with_suffix("weight_reweight")
            w_patch = pfx.with_suffix("patches")
            with spinner(f"Writing reweight maps ({pfx.stem})", enabled=verb >= 1):
                save_image(result.weight_orig, w_orig, header_info=header_info)
                save_image(result.weight_refined, w_new, header_info=header_info)
                save_image(result.patch_labels.float(), w_patch, header_info=header_info)


def _parse_reg_echo(reg_echo: str | None, n_echoes: int) -> tuple[bool, int]:
    """Resolve -reg_echo into (use_mean, zero_based_echo_index).

    A value of 'mean' selects the cross-echo mean; an integer is 1-based. With a
    single input, only 1 or 'mean' are valid (both reduce to that one echo).
    """
    if reg_echo is None:
        return False, 0
    if str(reg_echo).lower() == "mean":
        return True, 0
    try:
        r = int(reg_echo)
    except ValueError:
        print(
            f"Error: -reg_echo must be an integer (1-based) or 'mean', got {reg_echo!r}.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not (1 <= r <= n_echoes):
        print(f"Error: -reg_echo {r} out of range for {n_echoes} echo(es).", file=sys.stderr)
        sys.exit(1)
    return False, r - 1


def _want_reductions(args: argparse.Namespace) -> bool:
    """True when any temporal reduction was requested (each needs the resample)."""
    return any(getattr(args, dest, None) is not None for dest, _ in _TEMPORAL_REDUCTIONS)


def _reduce_time(series: torch.Tensor, which: str) -> torch.Tensor:
    """One temporal reduction of a (nt, nz, ny, nx) corrected series."""
    if which == "max":
        return series.amax(dim=0)
    if which == "min":
        return series.amin(dim=0)
    return series.mean(dim=0)


def _reduction_path(value: str, which: str, prefix: str | None, flag: str) -> str:
    """Output path for a temporal reduction: the given PREFIX, or ``{which}_`` on
    the front of -prefix when the flag was passed bare. Exits when neither exists."""
    if value is not _MEAN_FROM_PREFIX:
        return parse_prefix(value).as_file()
    if prefix is None:
        print(
            f"Error: -{flag} with no value needs -prefix to derive the "
            f"{which} path; pass -{flag} PREFIX instead.",
            file=sys.stderr,
        )
        sys.exit(1)
    return derive_prefixed_output_path(parse_prefix(prefix).as_file(), which)


def _validate_run_args(args: argparse.Namespace) -> None:
    """Validate the output/QC request for one run; exit(1) on an empty request.

    Shared by the standalone path and every -batch line so a manifest run is
    validated exactly like the equivalent solo invocation.
    """
    if args.reweight_tolerance < 1:
        print("Error: -reweight_tolerance must be >= 1.", file=sys.stderr)
        sys.exit(1)

    # Guard against a run that would produce nothing.
    _any_output = any(
        getattr(args, name, None) is not None
        for name in (
            "prefix",
            "save_mean",
            "save_max",
            "save_min",
            "1Dfile",
            "1Dmatrix_save",
            "dfile",
            "maxdisp1D",
            "iterfile",
            "save_weight",
            "save_shifts",
            "onedfile_shiftcorr",
            "onedmatrix_shiftcorr",
        )
    ) or _want_qc(args)
    if not _any_output:
        print(
            "Error: no outputs requested. Give at least one of -prefix, "
            "-save_mean, -save_max, -save_min, -1Dfile, -1Dmatrix_save, -dfile, "
            "-maxdisp1D, -iterfile.",
            file=sys.stderr,
        )
        sys.exit(1)

    _validate_shiftcorr_args(args)

    # A bare -save_mean/-save_max/-save_min/-save_weight derives its path from
    # -prefix, and without one the run cannot write what it was asked for. Catch
    # that here rather than where the path is built: those call sites sit after
    # the correction, so the user paid for a full 300-volume registration before
    # being told the invocation was unusable (bug of record).
    _bare = [
        f"-{flag}"
        for flag, _which in _TEMPORAL_REDUCTIONS
        if getattr(args, flag, None) is _MEAN_FROM_PREFIX
    ]
    if getattr(args, "save_weight", None) is _WEIGHT_FROM_PREFIX:
        _bare.append("-save_weight")
    if _bare and args.prefix is None:
        _plural = len(_bare) > 1
        print(
            f"Error: {', '.join(_bare)} given with no value, so "
            f"{'their paths are' if _plural else 'its path is'} derived from "
            f"-prefix. Pass -prefix, or give {'each flag' if _plural else 'the flag'} "
            "an explicit PREFIX.",
            file=sys.stderr,
        )
        sys.exit(1)

    # The QC files are named after -prefix; require it when requested.
    if _want_qc(args) and args.prefix is None:
        print(
            "Error: -save_first_last / -save_first_last_diff / -save_tsnr / "
            "-save_initial name their files after -prefix; pass -prefix.",
            file=sys.stderr,
        )
        sys.exit(1)


def _validate_shiftcorr_args(args: argparse.Namespace) -> None:
    """Validate the -me_3depi request (and the flags that only mean anything with it)."""
    dependent = {
        "save_shifts": "-save_shifts",
        "onedfile_shiftcorr": "-1Dfile_shiftcorr",
        "onedmatrix_shiftcorr": "-1Dmatrix_shiftcorr",
        "echo_times": "-echo_times",
    }
    if not args.me_3depi:
        for name, flag in dependent.items():
            if getattr(args, name, None) is not None:
                print(f"Error: {flag} only applies with -me_3depi.", file=sys.stderr)
                sys.exit(1)
        return

    n_echoes = len(args.input_file or [])
    if n_echoes < 2:
        print(
            "Error: -me_3depi estimates the shift BETWEEN echoes; give at least two "
            "-input files (one per echo).",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.axis is None:
        print("Error: -me_3depi requires -axis (the partition direction).", file=sys.stderr)
        sys.exit(1)
    try:
        resolve_pe_axis(args.axis)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    if args.echo_times is not None and len(args.echo_times) != n_echoes:
        print(
            f"Error: -echo_times has {len(args.echo_times)} values but -input has "
            f"{n_echoes} echoes.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.tpattern is not None:
        print("Error: -me_3depi and -tpattern cannot be combined yet.", file=sys.stderr)
        sys.exit(1)
    if args.shift_max <= 0:
        print("Error: -shift_max must be positive.", file=sys.stderr)
        sys.exit(1)


def _estimate_shiftcorr(args, input_files: list[str], device: torch.device, verb: int):
    """Run the -me_3depi inter-echo estimate, streaming one echo at a time.

    Only two echoes are ever resident: the estimator consumes the generator
    lazily, so a 5-echo series costs the same RAM as a 2-echo one.
    """
    axis = resolve_pe_axis(args.axis)
    tes = np.asarray(args.echo_times, dtype=np.float64) if args.echo_times else None
    if verb >= 1:
        mode = "TE regression through the origin" if tes is not None else "raw cumulative"
        print(
            f"-me_3depi: partition axis {args.axis} (voxel axis {axis}), "
            f"{args.shift_ordering} ordering, {mode}"
        )

    def _echo_stream():
        for path in input_files:
            data, _ = _load_trimmed(path, args.skip_first, args.skip_last, verb)
            yield data

    est = estimate_shifts(
        _echo_stream(),
        axis,
        tes=tes,
        ordering=args.shift_ordering,
        max_shift=args.shift_max,
        weight=None if args.shift_weight == "none" else args.shift_weight,
        extent=args.corr_extent,
        device=device,
        verb=verb,
    )
    if args.save_shifts is not None:
        save_shift_tables(est, args.save_shifts, tes, verb)
    return est, axis


def _save_shiftcorr_params(args, result, est, axis: int, header_info, verb: int) -> None:
    """Write the per-echo TOTAL transforms (rigid motion + that echo's shift).

    -1Dfile / -1Dmatrix_save report the one rigid pose shared by every echo;
    these report what each echo's voxels actually did, which is the shared pose
    plus a translation that differs per echo and per volume.
    """
    if args.onedfile_shiftcorr is None and args.onedmatrix_shiftcorr is None:
        return
    affine = header_info["affine"] if header_info else np.eye(4)
    for i in range(est.applied.shape[1]):
        tag = f"e{i + 1}_"
        mats = fold_shift_into_matrices(result.matrices_vox, est.applied[:, i], axis)
        dicom = np.stack(
            [voxel_matrix_to_dicom(mats[t], affine, affine).numpy() for t in range(mats.shape[0])]
        )
        if args.onedmatrix_shiftcorr is not None:
            path = _sibling(args.onedmatrix_shiftcorr, tag)
            save_matrix_1D(
                dicom,
                path,
                header=f"ffs_moco echo {i + 1} matrices, motion + -me_3depi shift "
                "(DICOM-to-DICOM, row-by-row):",
            )
            if verb >= 1:
                print(f"Saved echo {i + 1} matrices: {path}")
        if args.onedfile_shiftcorr is not None:
            params = np.stack(
                [
                    matrix_to_params(torch.from_numpy(dicom[t]))[:6].numpy()
                    for t in range(len(dicom))
                ]
            )
            path = _sibling(args.onedfile_shiftcorr, tag)
            save_moco_1D(params, path)
            if verb >= 1:
                print(f"Saved echo {i + 1} params: {path}")


def _dispatch_run(args: argparse.Namespace, device: torch.device, verb: int) -> None:
    """Estimate + resample one self-contained run (single- or multi-echo).

    This is the entire per-file body; both the standalone path and the batch
    loop go through it, so a manifest line reproduces a solo run bit-for-bit.
    """
    input_files = args.input_file
    n_echoes = len(input_files)
    if n_echoes > 1 and args.reg_echo is None:
        print(
            "Error: multiple -input files require -reg_echo N|mean to choose the "
            "echo (or 'mean') that drives the estimation.",
            file=sys.stderr,
        )
        sys.exit(1)
    reg_mean, reg_index = _parse_reg_echo(args.reg_echo, n_echoes)

    t0 = time.time()
    if n_echoes == 1:
        _run_single_echo(args, input_files[0], device, verb)
    else:
        _run_multi_echo(args, input_files, device, verb, reg_mean, reg_index)

    if verb >= 1:
        print(f"Total time: {time.time() - t0:.2f}s")


def _expected_outputs(args: argparse.Namespace) -> list[str]:
    """Concrete output file paths a solo run of ``args`` would write.

    Mirrors the naming in `_run_single_echo` / `_run_multi_echo` and
    `_save_estimation_outputs` so `-batch_skip` checks exactly what the run
    produces. Multi-echo per-echo files get their eN_ prefix. Conditional QC /
    reweight maps are listed on intent: if a runtime guard prevents one (tSNR
    needs >=2 volumes; reweight maps need a live reweight), the run simply isn't
    skipped next time — safe, since re-running is cheaper than a wrong skip.
    """
    outs: list[str] = []
    n_echoes = len(args.input_file or [])
    echo_tags = [f"e{i + 1}_" for i in range(n_echoes)] if n_echoes > 1 else [""]

    def _pfx(value: str, tag: str) -> str:
        return _sibling(value, tag) if tag else value

    for tag in echo_tags:
        # Corrected series.
        if args.prefix is not None:
            outs.append(parse_prefix(_pfx(args.prefix, tag)).as_file())
        # Temporal reductions of the corrected series.
        for dest, which in _TEMPORAL_REDUCTIONS:
            value = getattr(args, dest, None)
            if value is None:
                continue
            if value is _MEAN_FROM_PREFIX:
                if args.prefix is not None:
                    outs.append(
                        derive_prefixed_output_path(
                            parse_prefix(_pfx(args.prefix, tag)).as_file(), which
                        )
                    )
            else:
                outs.append(parse_prefix(_pfx(value, tag)).as_file())
        # QC files — all named after the (per-echo) prefix.
        if _want_qc(args) and args.prefix is not None:
            base_file = parse_prefix(_pfx(args.prefix, tag)).as_file()
            want_plain = bool(args.save_first_last)
            want_diff = bool(args.save_first_last_diff)
            if want_plain:
                outs.append(derive_prefixed_output_path(base_file, "firstlast"))
            if want_diff:
                outs.append(derive_prefixed_output_path(base_file, "firstlastdiff"))
            if args.save_tsnr:
                outs.append(derive_prefixed_output_path(base_file, "tsnr"))
            if args.save_initial:
                if want_plain or not (want_diff or args.save_tsnr):
                    outs.append(derive_prefixed_output_path(base_file, "firstlast_initial"))
                if want_diff:
                    outs.append(derive_prefixed_output_path(base_file, "firstlastdiff_initial"))
                if args.save_tsnr:
                    outs.append(derive_prefixed_output_path(base_file, "tsnr_initial"))

    # Single-instance outputs (once per run, echo-independent).
    for name in ("1Dfile", "1Dmatrix_save", "dfile", "maxdisp1D", "iterfile"):
        val = getattr(args, name, None)
        if val is not None:
            outs.append(val)

    # -me_3depi: the QC tables, plus the per-echo total-transform files.
    if args.me_3depi:
        if args.save_shifts is not None:
            outs.extend(shift_table_paths(args.save_shifts, args.echo_times is not None))
        for name in ("onedfile_shiftcorr", "onedmatrix_shiftcorr"):
            val = getattr(args, name, None)
            if val is not None:
                outs.extend(_sibling(val, tag) for tag in echo_tags)

    # Reweight weight/diagnostic maps — only written when reweight actually runs.
    if args.save_weight is not None and args.reweight:
        wv = args.prefix if args.save_weight is _WEIGHT_FROM_PREFIX else args.save_weight
        if wv is not None:
            pfx = parse_prefix(wv)
            outs.append(pfx.with_suffix("weight_orig"))
            outs.append(pfx.with_suffix("weight_reweight"))
            outs.append(pfx.with_suffix("patches"))

    return outs


def _validate_batch_run(run_args: argparse.Namespace) -> None:
    """Per-run validation for a batch job: needs -input, then the usual checks."""
    if not run_args.input_file:
        raise ValueError("run has no -input")
    _validate_run_args(run_args)


def _run_batch(args: argparse.Namespace) -> None:
    """Run every -batch / -batch_run job in this one process via the shared runner.

    The wins over a shell loop are all fixed costs: the Python interpreter, the
    torch/CUDA import, the CUDA context, and — the big one — torch.compile's
    kernel warmup are paid once and reused across runs. The per-file load /
    estimate / write cost is unchanged. With -batch_skip, jobs whose outputs all
    exist are skipped.
    """
    device = _select_device(args.device)  # chosen once; reused for every run
    jobs = collect_batch_jobs(args.batch, args.batch_run)
    run_batch_jobs(
        tool="ffs_moco",
        jobs=jobs,
        device=device,
        parse_line=lambda line: parse_args(shlex.split(line)),
        dispatch=lambda run_args, dev: _dispatch_run(run_args, dev, run_args.verb),
        validate=_validate_batch_run,
        is_nested=lambda ra: ra.batch is not None or ra.batch_run is not None,
        expected_outputs=_expected_outputs,
        skip_existing=args.batch_skip,
        verb=args.verb,
    )


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point for ffs_moco."""
    args = parse_args(argv)

    if args.batch is not None or args.batch_run:
        _run_batch(args)
        return

    if not args.input_file:
        print(
            "Error: -input is required (or use -batch FILE / -batch_run ARGS).",
            file=sys.stderr,
        )
        sys.exit(1)

    _validate_run_args(args)

    device = _select_device(args.device)
    verb = args.verb
    if verb >= 1:
        print(f"ffs_moco\n  device: {device}")
        print(
            "  interpolation kernels:\n"
            f"    motion estimation: {args.interp}\n"
            f"    final data resampling: {args.final_interp}"
        )
        if args.tpattern is not None:
            print(f"    temporal resampling: {args.tinterp}")
        print()

    _dispatch_run(args, device, verb)


def _run_single_echo(args, input_file: str, device: torch.device, verb: int) -> None:
    """Classic single-input motion correction: estimate and resample one series."""
    t0 = time.time()
    data, header_info = _load_trimmed(input_file, args.skip_first, args.skip_last, verb)
    if verb >= 1:
        print(f"Input: {input_file} {tuple(data.shape)} ({data.shape[0]} volumes)")
        print(f"Load time: {time.time() - t0:.2f}s")

    base_vol, base_index = _parse_base(args, verb)

    # Resampling is only needed if we will emit a corrected series, its mean, or a
    # corrected-series QC map (-save_initial reads the raw data only).
    need_aligned = args.prefix is not None or _want_reductions(args) or _want_corrected_qc(args)
    config = _build_config(
        args, device, verb, skip_resample=not need_aligned, base_index=base_index
    )

    t1 = time.time()
    result = _run_estimation(args, data, config, header_info, base_vol, input_file, verb)
    if verb >= 1:
        print(f"Total registration: {time.time() - t1:.2f}s")

    # Corrected timeseries (skipped when -prefix is omitted).
    if args.prefix is not None:
        out_path = parse_prefix(args.prefix).as_file()
        with spinner(f"Writing {out_path}", enabled=verb >= 1):
            save_image(result.aligned, out_path, header_info=header_info)

    # Temporal reductions of the corrected series (mean / max / min).
    for dest, which in _TEMPORAL_REDUCTIONS:
        value = getattr(args, dest)
        if value is None:
            continue
        out = _reduction_path(value, which, args.prefix, dest)
        with spinner(f"Writing {which} {out}", enabled=verb >= 1):
            save_image(_reduce_time(result.aligned, which), out, header_info=header_info)

    if _want_qc(args):
        base_path = parse_prefix(args.prefix).as_file()
        _write_qc(args, result.aligned, data, base_path, header_info, verb)

    _save_estimation_outputs(args, result, header_info, verb)


def _run_multi_echo(
    args,
    input_files: list[str],
    device: torch.device,
    verb: int,
    reg_mean: bool,
    reg_index: int,
) -> None:
    """Multi-echo: estimate motion from one echo (or the cross-echo mean), then
    apply the same transforms to every echo, writing eN_ prefixed outputs."""
    if getattr(args, "tpattern", None) is not None:
        print(
            "Error: -tpattern (slice-timing-aware moco) is single-echo only for now.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_echoes = len(input_files)
    if verb >= 1:
        which = "mean" if reg_mean else f"echo {reg_index + 1}"
        print(f"Multi-echo: {n_echoes} echoes, estimating motion from {which}")

    base_vol, base_index = _parse_base(args, verb)

    # -me_3depi: take the TE-dependent partition shift out FIRST, so the rigid
    # estimate below sees geometry that is genuinely common to all echoes.
    est = shift_axis = None
    if args.me_3depi:
        est, shift_axis = _estimate_shiftcorr(args, input_files, device, verb)

    # Build the estimation source (one echo, or the per-timepoint mean).
    if reg_mean:
        reg_data, header_info = _load_echo_mean(
            input_files, args.skip_first, args.skip_last, verb, est=est, axis=shift_axis
        )
    else:
        reg_data, header_info = _load_trimmed(
            input_files[reg_index], args.skip_first, args.skip_last, verb
        )
        if est is not None:
            reg_data = apply_shift(reg_data, est.applied[:, reg_index], shift_axis, device=device)

    # Estimate once — matrices only. Each echo (including the reg echo) is
    # resampled separately below from its own data, so skip Pass 2 here.
    config = _build_config(args, device, verb, skip_resample=True, base_index=base_index)
    t1 = time.time()
    result = moco(reg_data, config, header_info=header_info, base_vol=base_vol)
    if verb >= 1:
        print(f"Total registration: {time.time() - t1:.2f}s")
    # The dfile's post-alignment RMS must be measured against the SHIFT-CORRECTED
    # base, since that is the geometry the matrices were estimated in.
    reg_base_shifted = reg_data[base_index].clone() if est is not None else None
    del reg_data  # free the estimation series before loading echoes for resampling

    write_series = args.prefix is not None
    write_reductions = _want_reductions(args)
    want_qc = _want_qc(args)

    # Resample each echo with the shared matrices and write eN_ outputs.
    if write_series or write_reductions or want_qc:
        # Normally the base volume is copied through verbatim to spare it an
        # interpolation. With -me_3depi it must NOT be: its own shift correction
        # is nonzero and lives in the matrix we are about to fold.
        base_copy_idx = -1 if est is not None else (base_index if base_vol is None else -1)
        dtype = torch.float32
        for i, path in enumerate(input_files):
            echo_num = i + 1
            echo, echo_hdr = _load_trimmed(path, args.skip_first, args.skip_last, verb)

            # Fold this echo's partition shift into the shared rigid matrices, so
            # motion and shift reach the output through ONE interpolation.
            matrices = result.matrices_vox
            if est is not None:
                matrices = fold_shift_into_matrices(matrices, est.applied[:, i], shift_axis)

            # Post-alignment RMS is only wired into the dfile, which reports the
            # reg echo's motion — compute it just for that echo when -dfile is set.
            base_est = None
            if not reg_mean and i == reg_index and args.dfile is not None:
                if base_vol is not None:
                    bsrc = base_vol
                elif reg_base_shifted is not None:
                    bsrc = reg_base_shifted
                else:
                    bsrc = echo[base_index]
                base_est = _blur_volume(bsrc.to(device=device, dtype=dtype), args.blur)

            if verb >= 1:
                print(f"Resampling echo {echo_num} with {config.final_interp}: {path}")
            aligned, rms_after = resample_timeseries(
                echo,
                matrices,
                config,
                device,
                base_copy_idx=base_copy_idx,
                base_est=base_est,
                disable_pbar=verb == 0,
            )
            if base_est is not None:
                result.rms_after = rms_after  # feed the dfile written after the loop

            if write_series:
                out_path = parse_prefix(_sibling(args.prefix, f"e{echo_num}_")).as_file()
                with spinner(f"Writing {out_path}", enabled=verb >= 1):
                    save_image(aligned, out_path, header_info=echo_hdr)

            for dest, which in _TEMPORAL_REDUCTIONS:
                value = getattr(args, dest)
                if value is None:
                    continue
                echo_prefix = _sibling(args.prefix, f"e{echo_num}_") if args.prefix else None
                out_path = _reduction_path(
                    value if value is _MEAN_FROM_PREFIX else _sibling(value, f"e{echo_num}_"),
                    which,
                    echo_prefix,
                    dest,
                )
                with spinner(f"Writing {which} {out_path}", enabled=verb >= 1):
                    save_image(_reduce_time(aligned, which), out_path, header_info=echo_hdr)

            if want_qc:
                base_path = parse_prefix(_sibling(args.prefix, f"e{echo_num}_")).as_file()
                _write_qc(args, aligned, echo, base_path, echo_hdr, verb)

            del echo, aligned
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _save_estimation_outputs(args, result, header_info, verb)
    if est is not None:
        _save_shiftcorr_params(args, result, est, shift_axis, header_info, verb)


if __name__ == "__main__":
    main()
