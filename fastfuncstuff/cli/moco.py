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

import torch

from fastfuncstuff.cli_utils import add_verbose_arg, parse_prefix, spinner
from fastfuncstuff.processing.affine import save_matrix_1D
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
    derive_mean_output_path,
    load_image,
    save_first_last,
    save_image,
    save_tsnr,
)

# Sentinel for `-save_mean` given with no value: derive the path from -prefix.
_MEAN_FROM_PREFIX = "\x00from_prefix"

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
    io_group.add_argument(
        "-batch",
        default=None,
        metavar="FILE",
        help="Run many self-contained motion corrections in one process, "
        "amortizing Python/CUDA startup and torch.compile warmup. FILE is a "
        "manifest with one run per line; each line is exactly the arguments "
        "you would pass after `ffs_moco` for that run (e.g. `-input a.nii "
        "-base 0 -prefix a_mc.nii -dfile a.1D`). Blank lines and lines "
        "starting with # are ignored. The device is chosen once for the whole "
        "batch; a per-line -device is ignored. One failing run is reported and "
        "skipped without sinking the rest; the batch exits nonzero if any "
        "run failed.",
    )
    io_group.add_argument(
        "-batch_run",
        "-batch-run",
        dest="batch_run",
        action="append",
        metavar="ARGS",
        help="Inline alternative to -batch for self-contained scripts: one run "
        'given as a single quoted argument string (e.g. -batch_run "-input '
        'a.nii -base 0 -prefix a_mc.nii -dfile a.1D"), exactly as you would '
        "type it after `ffs_moco`. Repeatable — pass it once per run. Same "
        "semantics as -batch (device chosen once, failures isolated); may be "
        "combined with -batch, in which case manifest-file runs come first.",
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
        help="Pre-pass that drops weight regions whose local displacement doesn't "
        "match what the global head motion predicts there (removes bright "
        "artifacts/ghosts that mislead alignment). Like -twopass, it looks at the "
        "data first, then runs the normal estimation with the refined weight.",
    )
    rw_group.add_argument(
        "-reweight_minparams",
        "-reweight-minparams",
        dest="reweight_minparams",
        type=int,
        default=2,
        help="Keep a patch if its displacement agrees with the global-motion "
        "prediction on at least this many of the 3 axes (default: 2).",
    )
    rw_group.add_argument(
        "-reweight_rmin",
        "-reweight-rmin",
        dest="reweight_rmin",
        type=float,
        default=0.1,
        help="Per-axis correlation threshold for 'agrees' (default: 0.1).",
    )
    rw_group.add_argument(
        "-reweight_polort",
        "-reweight-polort",
        dest="reweight_polort",
        type=int,
        default=-1,
        help="Detrend degree for the per-patch time-courses before correlating "
        "(default: -1 = auto, 1 + floor(nt*TR/150)).",
    )
    rw_group.add_argument(
        "-reweight_bloktype",
        "-reweight-bloktype",
        dest="reweight_bloktype",
        choices=["rhdd", "tohd", "cube"],
        default="rhdd",
        help="Space-filling patch shape (default: rhdd, AFNI's LPC default).",
    )
    rw_group.add_argument(
        "-reweight_blokrad",
        "-reweight-blokrad",
        dest="reweight_blokrad",
        type=float,
        default=0.0,
        help="Patch radius in mm (default: 0 = auto, ~555 voxels/patch).",
    )
    rw_group.add_argument(
        "-reweight_maxiter",
        "-reweight-maxiter",
        dest="reweight_maxiter",
        type=int,
        default=6,
        help="Gauss-Newton iterations for the cheap per-patch estimate (default: 6).",
    )
    rw_group.add_argument(
        "-save_weight",
        "-save-weight",
        dest="save_weight",
        nargs="?",
        const=_WEIGHT_FROM_PREFIX,
        default=None,
        metavar="PREFIX",
        help="Save the original weight, the reweighted weight, and the patch "
        "label map (random id per kept patch). With no value, derives the paths "
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

    # --- Output files ---
    out_group = parser.add_argument_group("Output files")
    out_group.add_argument("-1Dfile", default=None, help="Save 6-column motion parameters (.1D)")
    out_group.add_argument("-1Dmatrix_save", default=None, help="Save affine matrices (.aff12.1D)")
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
    hw_group.add_argument(
        "-device", default=None, help="PyTorch device: cuda, mps, cpu (auto-detected)"
    )
    add_verbose_arg(hw_group, default=1)
    hw_group.add_argument(
        "-debug_memory",
        action="store_true",
        help="Print VRAM usage vs. prediction after registration and resampling loops",
    )

    args = parser.parse_args(argv)
    return args


def _select_device(arg_device: str | None) -> torch.device:
    """Resolve the -device flag (or auto-detect)."""
    if arg_device:
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def _load_echo_mean(input_files: list[str], skip_first: int, skip_last: int, verb: int):
    """Per-timepoint mean across echoes, accumulated one echo at a time.

    Loads echoes sequentially and sums in place so peak memory stays at roughly
    two 4D volumes rather than N — the running accumulator plus the current echo.
    """
    if verb >= 1:
        print(f"Building reg series: mean of {len(input_files)} echoes")
    acc = None
    header_info = None
    for path in input_files:
        echo, hdr = _load_trimmed(path, skip_first, skip_last, verb)
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
        reweight_minparams=args.reweight_minparams,
        reweight_rmin=args.reweight_rmin,
        reweight_polort=args.reweight_polort,
        reweight_bloktype=args.reweight_bloktype,
        reweight_blokrad=args.reweight_blokrad,
        reweight_maxiter=args.reweight_maxiter,
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
    iteration counts, and reweight weight/patch images. The corrected series
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

    # Reweight weight images + patch label map (single, estimated once).
    if args.save_weight is not None:
        if not args.reweight:
            print(
                "Warning: -save_weight given without -reweight; nothing to save.",
                file=sys.stderr,
            )
        elif result.weight_refined is None or result.patch_labels is None:
            print(
                "Warning: reweight did not run (no patches / guard); skipping -save_weight.",
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
            with spinner("Writing weight/patch maps"):
                save_image(result.weight_orig, w_orig, header_info=header_info)
                save_image(result.weight_refined, w_new, header_info=header_info)
                save_image(result.patch_labels.float(), w_patch, header_info=header_info)
            if verb >= 1:
                print(f"Saved weights: {w_orig}, {w_new}, {w_patch}")


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


def _validate_run_args(args: argparse.Namespace) -> None:
    """Validate the output/QC request for one run; exit(1) on an empty request.

    Shared by the standalone path and every -batch line so a manifest run is
    validated exactly like the equivalent solo invocation.
    """
    # Guard against a run that would produce nothing.
    _any_output = any(
        getattr(args, name, None) is not None
        for name in (
            "prefix",
            "save_mean",
            "1Dfile",
            "1Dmatrix_save",
            "dfile",
            "maxdisp1D",
            "iterfile",
            "save_weight",
        )
    ) or _want_qc(args)
    if not _any_output:
        print(
            "Error: no outputs requested. Give at least one of -prefix, "
            "-save_mean, -1Dfile, -1Dmatrix_save, -dfile, -maxdisp1D, -iterfile.",
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


def _collect_batch_jobs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Gather (label, argv-string) runs from -batch file and/or -batch_run inline.

    File runs come first, then inline runs, so labels stay stable regardless of
    which source(s) are used. Exits(1) if a requested source yields nothing.
    """
    jobs: list[tuple[str, str]] = []

    if args.batch is not None:
        manifest = Path(args.batch)
        if not manifest.is_file():
            print(f"Error: -batch file not found: {args.batch}", file=sys.stderr)
            sys.exit(1)
        n_before = len(jobs)
        for lineno, raw in enumerate(manifest.read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            jobs.append((f"line {lineno}", line))
        if len(jobs) == n_before:
            print(f"Error: -batch file has no runs: {args.batch}", file=sys.stderr)
            sys.exit(1)

    for i, raw in enumerate(args.batch_run or [], 1):
        line = raw.strip()
        if not line:
            print(f"Error: -batch_run #{i} is empty.", file=sys.stderr)
            sys.exit(1)
        jobs.append((f"run {i}", line))

    return jobs


def _run_batch(args: argparse.Namespace) -> None:
    """Run every -batch / -batch_run job in this one process.

    The wins over a shell loop are all fixed costs: the Python interpreter, the
    torch/CUDA import, the CUDA context, and — the big one — torch.compile's
    kernel warmup are paid once and reused across runs. The per-file load /
    estimate / write cost is unchanged. Runs stay isolated: one failure is
    reported and skipped, and the batch exits nonzero if any run failed.
    """
    from tqdm import tqdm

    verb = args.verb
    device = _select_device(args.device)  # chosen once; reused for every run

    jobs = _collect_batch_jobs(args)

    if verb >= 1:
        print(f"ffs_moco batch: {len(jobs)} runs on device={device}")

    failures: list[tuple[str, str]] = []
    t0 = time.time()
    bar = tqdm(jobs, desc="moco batch", unit="run", leave=True, disable=len(jobs) == 1)
    for label, line in bar:
        try:
            run_args = parse_args(shlex.split(line))
            if run_args.batch is not None or run_args.batch_run is not None:
                raise ValueError("-batch/-batch_run cannot be nested inside a run")
            if not run_args.input_file:
                raise ValueError("run has no -input")
            if run_args.device and verb >= 1:
                tqdm.write(
                    f"[{label}] ignoring per-run -device {run_args.device}; batch uses {device}"
                )
            _validate_run_args(run_args)
            _dispatch_run(run_args, device, run_args.verb)
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
        ok = len(jobs) - len(failures)
        print(f"ffs_moco batch done: {ok}/{len(jobs)} succeeded in {time.time() - t0:.2f}s")
    if failures:
        print(f"Error: {len(failures)} batch run(s) failed:", file=sys.stderr)
        for label, reason in failures:
            print(f"  {label}: {reason}", file=sys.stderr)
        sys.exit(1)


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
        print(f"ffs_moco: device={device}")

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
    need_aligned = args.prefix is not None or args.save_mean is not None or _want_corrected_qc(args)
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
        with spinner(f"Writing {Path(out_path).name}"):
            save_image(result.aligned, out_path, header_info=header_info)
        if verb >= 1:
            print(f"Saved: {out_path}")

    # Temporal mean of the corrected series.
    if args.save_mean is not None:
        if args.save_mean is _MEAN_FROM_PREFIX:
            if args.prefix is None:
                print(
                    "Error: -save_mean with no value needs -prefix to derive the "
                    "mean path; pass -save_mean PREFIX instead.",
                    file=sys.stderr,
                )
                sys.exit(1)
            mean_path = derive_mean_output_path(parse_prefix(args.prefix).as_file())
        else:
            mean_path = parse_prefix(args.save_mean).as_file()
        with spinner(f"Writing {Path(mean_path).name}"):
            save_image(result.aligned.mean(dim=0), mean_path, header_info=header_info)
        if verb >= 1:
            print(f"Saved mean: {mean_path}")

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

    # Build the estimation source (one echo, or the per-timepoint mean).
    if reg_mean:
        reg_data, header_info = _load_echo_mean(input_files, args.skip_first, args.skip_last, verb)
    else:
        reg_data, header_info = _load_trimmed(
            input_files[reg_index], args.skip_first, args.skip_last, verb
        )

    # Estimate once — matrices only. Each echo (including the reg echo) is
    # resampled separately below from its own data, so skip Pass 2 here.
    config = _build_config(args, device, verb, skip_resample=True, base_index=base_index)
    t1 = time.time()
    result = moco(reg_data, config, header_info=header_info, base_vol=base_vol)
    if verb >= 1:
        print(f"Total registration: {time.time() - t1:.2f}s")
    del reg_data  # free the estimation series before loading echoes for resampling

    write_series = args.prefix is not None
    write_mean = args.save_mean is not None
    want_qc = _want_qc(args)

    # Resample each echo with the shared matrices and write eN_ outputs.
    if write_series or write_mean or want_qc:
        base_copy_idx = base_index if base_vol is None else -1
        dtype = torch.float32
        for i, path in enumerate(input_files):
            echo_num = i + 1
            echo, echo_hdr = _load_trimmed(path, args.skip_first, args.skip_last, verb)

            # Post-alignment RMS is only wired into the dfile, which reports the
            # reg echo's motion — compute it just for that echo when -dfile is set.
            base_est = None
            if not reg_mean and i == reg_index and args.dfile is not None:
                bsrc = base_vol if base_vol is not None else echo[base_index]
                base_est = _blur_volume(bsrc.to(device=device, dtype=dtype), args.blur)

            if verb >= 1:
                print(f"Resampling echo {echo_num}: {path}")
            aligned, rms_after = resample_timeseries(
                echo,
                result.matrices_vox,
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
                save_image(aligned, out_path, header_info=echo_hdr)
                if verb >= 1:
                    print(f"  Saved: {out_path}")

            if write_mean:
                if args.save_mean is _MEAN_FROM_PREFIX:
                    if args.prefix is None:
                        print(
                            "Error: -save_mean with no value needs -prefix to derive "
                            "the mean path; pass -save_mean PREFIX instead.",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                    base_pfx = parse_prefix(_sibling(args.prefix, f"e{echo_num}_")).as_file()
                    mean_path = derive_mean_output_path(base_pfx)
                else:
                    mean_path = parse_prefix(_sibling(args.save_mean, f"e{echo_num}_")).as_file()
                save_image(aligned.mean(dim=0), mean_path, header_info=echo_hdr)
                if verb >= 1:
                    print(f"  Saved mean: {mean_path}")

            if want_qc:
                base_path = parse_prefix(_sibling(args.prefix, f"e{echo_num}_")).as_file()
                _write_qc(args, aligned, echo, base_path, echo_hdr, verb)

            del echo, aligned
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _save_estimation_outputs(args, result, header_info, verb)


if __name__ == "__main__":
    main()
