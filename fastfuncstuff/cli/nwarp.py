"""CLI for GPU-accelerated multi-warp composition and application.

Command: nwarpforge (registered as entry point in pyproject.toml)

Usage:
    nwarpforge -source input.nii -nwarp 'warp1.nii mat.1D warp2.nii' -master template.nii -prefix output.nii

Equivalent to AFNI's 3dNwarpApply but with GPU acceleration and wsinc5 interpolation.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time

import torch

from fastfuncstuff.cli_help import FfsHelpFormatter
from fastfuncstuff.cli_utils import (
    add_batch_args,
    add_device_arg,
    add_verbose_arg,
    collect_batch_jobs,
    print_cli_footer,
    print_cli_header,
    print_cli_section,
    run_batch_jobs,
    setup_device,
    spinner,
)
from fastfuncstuff.processing.io import derive_prefixed_output_path
from fastfuncstuff.processing.nwarpforge import (
    derive_phase_output_path,
    nwarpforge,
    parse_nwarp_string,
)
from fastfuncstuff.utils import REGISTRATION_TF32


def parse_args(
    argv: list[str] | None = None, namespace: argparse.Namespace | None = None
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffs_nwarp",
        description="GPU-accelerated multi-warp composition and application",
        epilog="""Composition order:
  For -nwarp 'A B C', the result is C(B(A(x)))
  where x is a coordinate in output space.

  This is a 'pull' or 'backward' mapping convention:
  the composed warp tells where to sample FROM in source space
  for each output location.

Examples:
  # Apply single warp:
  nwarpforge -source subj.nii -nwarp warp.nii -prefix out.nii

  # Compose warp + affine matrix:
  nwarpforge -source epi.nii -nwarp 'warp.nii matrix.aff12.1D' -master template.nii -prefix out.nii

  # Full preprocessing chain (motion + warp to template):
  nwarpforge -source epi.nii \\
      -nwarp 'template_warp.nii motion_correct.aff12.1D fieldmap_warp.nii' \\
      -master template.nii \\
      -interp wsinc5 \\
      -prefix epi_in_template.nii

  # 4D time series with per-volume motion matrices:
  nwarpforge -source epi_4d.nii \\
      -nwarp 'template_warp.nii motion_matrices.aff12.1D' \\
      -master template.nii \\
      -prefix epi_4d_warped.nii
""",
        formatter_class=FfsHelpFormatter,
    )

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "-source",
        default=None,
        help="Source (magnitude) dataset to warp (3D or 4D NIfTI) [required unless -batch]",
    )
    add_batch_args(
        io_group,
        tool="ffs_nwarp",
        what="warp applies",
        example="-source a.nii -nwarp warp.nii -master m.nii -prefix a_w.nii",
        skip_note="-prefix / -phase_prefix / -save_mean / -save_max / -save_min / -save_first_last",
    )
    io_group.add_argument(
        "-phase",
        default=None,
        help="Phase dataset (any range — automatically scaled to radians). "
        "When provided, magnitude+phase are converted to real+imaginary, "
        "each component is warped separately, then converted back to "
        "magnitude+phase for output.",
    )
    io_group.add_argument(
        "-nwarp",
        default=None,
        help="Warp chain (quoted, space-separated). E.g., 'warp1.nii matrix.1D warp2.nii' "
        "[required unless -batch]",
    )
    io_group.add_argument(
        "-prefix",
        default=None,
        help="Output path for magnitude (or only output) [required unless -batch]",
    )
    io_group.add_argument(
        "-phase_prefix",
        default=None,
        help="Output path for warped phase. Auto-derived from -prefix when not given "
        "(e.g. out.nii.gz -> out_phase.nii.gz). Only used with -phase.",
    )
    io_group.add_argument(
        "-phase_units",
        choices=["raw", "rad"],
        default="raw",
        help="Units of the input phase data (only used with -phase). "
        "'raw' (default): scanner units (e.g. -4096..4095). Automatically "
        "scaled to radians. 'rad': already in radians (e.g. unwrapped). "
        "No scaling applied.",
    )
    io_group.add_argument(
        "-phase_warp",
        choices=["complex", "split", "direct", "circular"],
        default="complex",
        help="How to warp phase data (only used with -phase). "
        "'complex' (default): convert mag+phase to real/imag, warp each, "
        "convert back. Magnitude is derived from warped real/imag. "
        "'split': warp magnitude directly (clean), then warp real/imag "
        "and extract phase only. Magnitude never touched by phase. "
        "'direct': warp magnitude and phase independently. Assumes phase "
        "is already unwrapped. Fastest option. "
        "'circular': warp cos(phase) and sin(phase) separately (unit "
        "circle interpolation), then atan2. Handles wraps without "
        "magnitude corruption.",
    )
    io_group.add_argument(
        "-save_mean",
        action="store_true",
        help="If output is 4D, save mean as mean_{prefix_basename}{ext}",
    )
    io_group.add_argument(
        "-save_max",
        "-save-max",
        dest="save_max",
        action="store_true",
        help="If output is 4D, save the temporal MAX as max_{prefix_basename}{ext} "
        "— the union of every voxel imaged in any volume (motion/warp carry edge "
        "voxels out of the FoV, where the mean dims them). Computed on the OUTPUT "
        "grid, so it is exact for the final space.",
    )
    io_group.add_argument(
        "-save_min",
        "-save-min",
        dest="save_min",
        action="store_true",
        help="If output is 4D, save the temporal MIN as min_{prefix_basename}{ext} "
        "— 0 wherever any volume lost the voxel, i.e. >0 is the region with "
        "complete data at every timepoint (an analysis mask).",
    )
    io_group.add_argument(
        "-save_first_last",
        "-save-first-last",
        dest="save_first_last",
        action="store_true",
        help="If output is 4D, save the first and last volumes as a single "
        "switchable file firstlast_{prefix_basename}{ext} — flip between them in "
        "a viewer to eyeball how much the warp moved the data.",
    )
    io_group.add_argument(
        "-master",
        default=None,
        help="Master dataset defining output grid. "
        "If not specified, uses source grid. Use '-master WARP' (or NWARP) to "
        "use the first nonlinear warp's grid as the output master.",
    )

    io_group.add_argument(
        "-dxyz",
        type=float,
        default=None,
        help="Force isotropic output voxel size (mm). "
        "Recomputes output grid dimensions to cover the same FOV at the new resolution. "
        "E.g., -dxyz 3.0 for 3mm isotropic output.",
    )

    io_group.add_argument(
        "-no_autopad",
        "-no-autopad",
        dest="auto_pad",
        action="store_false",
        help="Disable automatic output-grid padding. By default (master-less "
        "case only) the grid grows when a warp would clip real source signal off "
        "the edge (overlap + clipped-mass test, not raw displacement). An explicit "
        "-master always fixes the output grid exactly, regardless of this flag.",
    )
    io_group.add_argument(
        "-expad",
        type=int,
        default=0,
        help="Extra padding (voxels) added on every side, on top of the auto "
        "estimate. Forces padding even with -no_autopad.",
    )

    interp_group = parser.add_argument_group("Interpolation")
    interp_group.add_argument(
        "-interp",
        choices=["NN", "nearest", "linear", "cubic", "quintic", "heptic", "wsinc5"],
        default="wsinc5",
        help="Final interpolation for source data (default: wsinc5). Use NN for "
        "atlas/label data (only mode that preserves integer labels).",
    )
    interp_group.add_argument(
        "-ainterp",
        choices=["linear", "cubic", "quintic", "heptic", "wsinc5"],
        default=None,
        help="Kernel for warp-field interpolation during composition (default: "
        "cubic). Warp fields are smooth, so cubic matches wsinc5 to negligible "
        "error at ~10x less cost; this only affects the warp, not the data "
        "(-interp). Higher order reduces the smoothing each composition step adds "
        "to the warp; 'linear' is fastest.",
    )
    interp_group.add_argument(
        "-no_neg",
        "-no-neg",
        dest="no_neg",
        action="store_true",
        help="Clamp warped output at 0 to suppress wsinc5/cubic negative ringing "
        "on non-negative data (magnitude, masks, probability maps).",
    )
    interp_group.add_argument(
        "-jac",
        dest="jac",
        default=None,
        metavar="AXIS[:FIELDMAP]",
        help="Apply phase-encode Jacobian intensity modulation (1 + d(disp)/d(AXIS)) "
        "to the warped output, so a geometry-only distortion warp (fieldmap / MEDIC / "
        "locomoco) is intensity-corrected like FSL applytopup --method=jac. AXIS is the "
        "phase-encode direction: i/j/k, x/y/z or LR/AP/IS (equivalent spellings). For a "
        "multi-transform chain, name the fieldmap with AXIS:FIELDMAP (a filename or unique "
        "substring, e.g. 'j:fmap'): its Jacobian is computed on its own grid and transported "
        "through the downstream transforms to the output -- exact even with upstream per-frame "
        "motion and -tpattern. Without :FIELDMAP it auto-uses the lone static single-axis warp. "
        "An affine-mixed / 3-D chain is left unmodulated.",
    )

    st_group = parser.add_argument_group("Slice timing (joint space-time)")
    st_group.add_argument(
        "-tpattern",
        default=None,
        help="Per-slice acquisition timing: text file (one time per line, "
        "seconds) or BIDS JSON with SliceTiming. When given, slice-timing "
        "correction is folded into the SAME resample as the warp chain "
        "(Roche 2011 joint space-time) -- the data is interpolated once, and "
        "the temporal shift follows the scanner slice each voxel lands in "
        "after motion. Requires a 4-D -source and a TR. Works with -phase: the "
        "complex channels go through the same space-time resample.",
    )
    st_group.add_argument(
        "-TR",
        type=float,
        default=None,
        help="Repetition time (seconds) for -tpattern. Read from the NIfTI "
        "header (or JSON RepetitionTime) when omitted.",
    )
    st_group.add_argument(
        "-tzero",
        type=float,
        default=None,
        help="Reference time within the TR to align all slices to (seconds). "
        "Default: mean of slice times (matches 3dTshift).",
    )
    st_group.add_argument(
        "-tinterp",
        choices=["linear", "cubic", "quintic", "heptic", "wsinc5", "wsinc9"],
        default="heptic",
        help="Temporal interpolation kernel for -tpattern (default heptic). "
        "Fourier is unavailable on the joint path (the per-voxel shift is not "
        "a single per-slice phase rotation).",
    )
    st_group.add_argument(
        "-tfollow",
        "-follow_tissue",
        "-follow-tissue",
        dest="follow_tissue",
        action="store_true",
        default=True,
        help="(default) Tissue-following joint resample: sample each temporal "
        "neighbour at its own frame's pose instead of freezing the output frame's "
        "pose (the slow-motion assumption). Recovers the right signal when motion "
        "sweeps tissue between scanner locations frame to frame (e.g. a brain edge "
        "moving in and out of a voxel), and corrects the non-uniform tap spacing that "
        "through-plane motion creates. Measured ~20-25%% slower than -frozen.",
    )
    st_group.add_argument(
        "-frozen",
        "-no_tfollow",
        "-no-tfollow",
        dest="follow_tissue",
        action="store_false",
        help="Force the frozen-pose (slow-motion-assumption) joint path instead of "
        "the default tissue-following resample. Matches the pre-2026 behaviour and "
        "a static 3dTshift-then-motion two-step.",
    )

    hw_group = parser.add_argument_group("Hardware")
    add_device_arg(
        hw_group,
        extra="On Apple Silicon, auto uses CPU; pass mps explicitly to experiment with Metal.",
    )
    add_verbose_arg(hw_group, default=1)

    debug_group = parser.add_argument_group("Debug")
    debug_group.add_argument(
        "-time_range",
        default=None,
        help="Process only specified time points (e.g., '0,5' for first 6 volumes). "
        "Useful for debugging.",
    )
    debug_group.add_argument(
        "-debug",
        action="store_true",
        help="Print detailed debug info (matrices, warp stats)",
    )

    return parser.parse_args(argv, namespace or argparse.Namespace())


def _select_device(device_arg: str | None) -> torch.device:
    """Resolve -device (explicit wins, else CUDA → CPU)."""
    return setup_device(device_arg, tf32=REGISTRATION_TF32)


def _expected_outputs(args: argparse.Namespace) -> list[str]:
    """Concrete output file paths a solo run of ``args`` would write, for
    -batch_skip. -prefix is used verbatim (nwarp does not run it through
    parse_prefix). The phase output, mean, and first/last files are listed on
    intent; a mean/first-last that a 3-D output skips just means the run isn't
    skipped next time (safe)."""
    outs: list[str] = [args.prefix]
    if args.phase:
        outs.append(args.phase_prefix or derive_phase_output_path(args.prefix))
    for want, which in ((args.save_mean, "mean"), (args.save_max, "max"), (args.save_min, "min")):
        if want:
            outs.append(derive_prefixed_output_path(args.prefix, which))
    if args.save_first_last:
        outs.append(derive_prefixed_output_path(args.prefix, "firstlast"))
    return outs


def _validate_batch_run(run_args: argparse.Namespace) -> None:
    """Per-run validation for a batch job: needs -source/-nwarp/-prefix."""
    missing = [f for f in ("source", "nwarp", "prefix") if getattr(run_args, f, None) is None]
    if missing:
        raise ValueError("run is missing " + ", ".join("-" + m for m in missing))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.batch is not None or args.batch_run:
        device = _select_device(args.device)
        jobs = collect_batch_jobs(args.batch, args.batch_run)
        run_batch_jobs(
            tool="ffs_nwarp",
            jobs=jobs,
            device=device,
            parse_line=lambda line, base: parse_args(shlex.split(line), base),
            defaults=args,
            dispatch=_dispatch_run,
            validate=_validate_batch_run,
            is_nested=lambda ra: ra.batch is not None or ra.batch_run is not None,
            expected_outputs=_expected_outputs,
            skip_existing=args.batch_skip,
            verb=args.verb,
        )
        return

    missing = [f for f in ("source", "nwarp", "prefix") if getattr(args, f, None) is None]
    if missing:
        print(
            "Error: " + ", ".join("-" + m for m in missing) + " required "
            "(or use -batch FILE / -batch_run ARGS).",
            file=sys.stderr,
        )
        sys.exit(1)

    _dispatch_run(args, _select_device(args.device))


def _dispatch_run(args: argparse.Namespace, device: torch.device) -> None:
    """Apply one self-contained warp chain (the entire per-run body).

    Both the standalone path and every batch job go through here, so a manifest
    line reproduces a solo invocation bit-for-bit."""
    verb = args.verb
    if verb >= 1:
        print_cli_header("ffs_nwarp", "Compose and apply spatial transforms")
        print_cli_section("Inputs", leading_blank=False)
        print(f"  Source: {args.source}")
        print(f"  Master: {args.master or 'source grid'}")
        print(f"  Output: {args.prefix}")
        if args.phase:
            print(f"  Phase:  {args.phase}")
        print_cli_section("Configuration")
        print(f"  Device: {device}")
        print(f"  Data interpolation: {args.interp}")
        if args.phase:
            print(f"  Phase interpolation: {args.phase_warp} ({args.phase_units})")

    # Parse time_range if provided
    time_range = None
    if args.time_range:
        parts = args.time_range.split(",")
        if len(parts) == 2:
            time_range = (int(parts[0]), int(parts[1]))
        else:
            time_range = (0, int(parts[0]))

    # -ainterp resamples the *warp fields* during composition, not the data. A
    # displacement field is smooth/band-limited, so cubic reproduces it to
    # negligible error while wsinc5 costs ~10x more per composition step (measured
    # on a 5-transform chain: composition 69s -> 7s, whole job 1.67x faster, data
    # sampling unchanged). We therefore default to cubic instead of inheriting
    # -interp the way AFNI 3dNwarpApply does -- the extra sinc accuracy is wasted
    # on a warp. The final *data* sampling still honours -interp (e.g. wsinc5).
    # Override with an explicit -ainterp when maximum warp fidelity is wanted.
    ainterp = args.ainterp
    if ainterp is None:
        ainterp = "cubic"
    if verb >= 1:
        print(f"  Warp interpolation: {ainterp}")
        print(f"  Time range: {time_range or 'all volumes'}")
        if args.tpattern is not None:
            print(f"  Temporal interpolation: {args.tinterp}")

    jac_axis = None
    jac_match = None
    if args.jac is not None:
        from fastfuncstuff.processing.nwarpforge import parse_pe_axis

        axis_str, _, jac_match = args.jac.partition(":")
        jac_match = jac_match.strip() or None
        jac_axis = parse_pe_axis(axis_str)
        if verb >= 1:
            tail = f", fieldmap '{jac_match}'" if jac_match else ""
            print(f"  Jacobian modulation: PE {axis_str} (axis {jac_axis}){tail}")

    nwarp_specs = parse_nwarp_string(args.nwarp)
    if verb >= 1:
        print_cli_section(f"Transform chain ({len(nwarp_specs)} transforms)")
        print("  Applied top to bottom; each transform feeds the next.")
        for i, spec in enumerate(nwarp_specs):
            print(f"  {i + 1:>2}. {spec}")

    # Slice timing: load per-slice offsets and resolve TR (header when not given).
    slice_times = None
    tr = args.TR
    if args.tpattern is not None:
        from fastfuncstuff.io.afni import get_tr_from_file
        from fastfuncstuff.processing.slicetime import load_slice_timing

        slice_times = load_slice_timing(args.tpattern)
        if tr is None:
            if args.tpattern.endswith(".json"):
                import json
                from pathlib import Path as _Path

                data = json.loads(_Path(args.tpattern).read_text())
                if "RepetitionTime" in data:
                    tr = float(data["RepetitionTime"])
            if tr is None:
                hdr_tr = get_tr_from_file(args.source)
                if hdr_tr and hdr_tr > 0:
                    tr = hdr_tr
        if tr is None:
            raise SystemExit("ffs_nwarp: -tpattern needs a TR; none in header, pass -TR.")
        if verb >= 1:
            print(
                f"  Slice timing: {len(slice_times)} slices, TR={tr:.4f}s, pattern={args.tpattern}"
            )

    t0 = time.time()
    if verb >= 1:
        print_cli_section("Applying transforms")

    nwarpforge(
        source_path=args.source,
        nwarp_specs=nwarp_specs,
        prefix=args.prefix,
        phase_path=args.phase,
        phase_prefix=args.phase_prefix,
        master_path=args.master,
        interp=args.interp,
        phase_warp=args.phase_warp,
        phase_units=args.phase_units,
        device=device,
        verb=verb,
        time_range=time_range,
        debug=args.debug,
        save_mean=args.save_mean,
        save_max=args.save_max,
        save_min=args.save_min,
        save_first_last_flag=args.save_first_last,
        dxyz=args.dxyz,
        no_neg=args.no_neg,
        auto_pad=args.auto_pad,
        expad=args.expad,
        ainterp=ainterp,
        slice_times=slice_times,
        tr=tr,
        tzero=args.tzero,
        tinterp=args.tinterp,
        follow_tissue=args.follow_tissue,
        jac_axis=jac_axis,
        jac_match=jac_match,
        progress=lambda message: spinner(message, enabled=verb >= 1),
    )

    if verb >= 1:
        print_cli_footer("ffs_nwarp", elapsed_seconds=time.time() - t0)


if __name__ == "__main__":
    main()
