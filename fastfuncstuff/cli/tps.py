#!/usr/bin/env python3
"""ffs_tps -- smooth spline HRF estimation (a preset of ffs_deconvolve).

A cubic-spline (CSPLIN) response whose knot values carry a smoothness
penalty -- the discrete 1-D thin-plate / cubic smoothing spline -- with the
strength, and optionally the penalty itself, chosen per voxel.  Everything
is ffs_deconvolve's: the shared loader and timing, the per-run design
builder, the canonical multi-run GLM packing and glm/smooth_basis.  This
file only translates ffs_tps's vocabulary and sets smoothing on; any other
ffs_deconvolve flag passes straight through.

The first ffs_tps had its own design builder (knots off the TR grid), a
CV-only global lambda, equal-division run boundaries and a hardcoded chunk
size.  See the wiki note "Smooth FIR".
"""

from __future__ import annotations

import argparse
import sys

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter

EPILOG = """\
Examples:
  ffs_tps -input run1.nii.gz run2.nii.gz -stim-times face.1D house.1D \\
          -tps-window 0 16 -output-prefix tps
  # lambda by REML, the penalty (curvature vs GP prior) by held-out runs:
  ffs_tps -input run*.nii.gz -events run*_events.tsv -output-prefix tps \\
          -smooth reml -penalty diff2,gp:4 -save-xval-r2
  # the average event response, knots every 0.335 s at TR 0.67:
  ffs_tps -input run*.nii.gz -events run*_events.tsv -output-prefix all \\
          -pool-conditions -tps-window 0 16.08 -n-knots 49

Any ffs_deconvolve flag is accepted (-mask, -polort, -microtime_offset,
-save-betas, -do_scale, ...); see ffs_deconvolve -help.
"""


def create_parser() -> argparse.ArgumentParser:
    parser = FfsArgumentParser(
        prog="ffs_tps",
        description="Smooth spline HRF estimation: CSPLIN knots with a per-voxel "
        "smoothness penalty (a preset of ffs_deconvolve -model CSPLIN -tent-smooth).",
        epilog=EPILOG,
        formatter_class=FfsHelpFormatter,
        # Unknown flags pass through to ffs_deconvolve; prefix matching could
        # otherwise claim one of them for an ffs_tps flag.
        allow_abbrev=False,
    )
    parser.add_argument("-input", nargs="+", required=True, help="fMRI runs (one file per run).")
    parser.add_argument("-output-prefix", required=True, help="Output prefix.")
    parser.add_argument("-stim-times", nargs="+", help="AFNI timing files, one per condition.")
    parser.add_argument("-stim-labels", nargs="+", help="Condition labels for -stim-times.")
    parser.add_argument(
        "-tps-window",
        nargs="+",
        help='Response window in seconds: "0 16", "0,16" or per condition "0,16 0,20".',
    )
    parser.add_argument(
        "-n-knots", type=int, help="Knots per condition (default: one per TR over the window)."
    )
    parser.add_argument(
        "-force-zero-edges",
        action="store_true",
        help="Pin the response to zero at both window edges (CSPLINzero).",
    )
    parser.add_argument(
        "-smooth",
        default="reml",
        metavar="RULE",
        help="How the smoothing strength is chosen, per voxel: reml (default), gcv, "
        "loro (held-out runs), or a fixed relative lambda.",
    )
    parser.add_argument(
        "-penalty",
        default="diff2",
        metavar="SPEC[,SPEC...]",
        help="Penalty shape(s): diff2 (curvature: the cubic smoothing / thin-plate "
        "spline), diff1, diff3, gp:SEC. Several are chosen per voxel by held-out runs.",
    )
    return parser


def translate(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    """ffs_tps options -> the equivalent ffs_deconvolve argv."""
    argv = ["-input", *args.input, "-prefix", args.output_prefix]
    argv += ["-model", "CSPLINzero" if args.force_zero_edges else "CSPLIN"]
    argv += ["-tent-smooth", args.smooth, "-smooth-penalty", args.penalty]
    if args.stim_times:
        argv += ["-onsets", *args.stim_times]
    if args.stim_labels:
        argv += ["-labels", *args.stim_labels]
    if args.tps_window:
        argv += ["-window", *args.tps_window]
    if args.n_knots is not None:
        argv += ["-tent-n-basis", str(args.n_knots)]
    return argv + passthrough


def main(argv: list[str] | None = None) -> int:
    args, passthrough = create_parser().parse_known_args(argv)
    from fastfuncstuff.cli import deconvolve

    return deconvolve.main(translate(args, passthrough))


if __name__ == "__main__":
    sys.exit(main())
