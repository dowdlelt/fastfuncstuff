"""CLI for per-TR outlier fractions (AFNI 3dToutcount) and outlier censoring.

Command: ffs_util_outcount (registered as entry point in pyproject.toml)

Computes what afni_proc.py's outcount block does with ``3dToutcount -automask
-fraction -polort P -legendre``: per run, each automask voxel is detrended by an
L1 polynomial fit, and a sample is an outlier when it lies more than
``qginv(0.001/nt) * sqrt(pi/2) * MAD`` from the trend. The output is the fraction
of masked voxels that are outliers at each TR, one row per TR, runs concatenated
in the order given (afni_proc's ``outcount_rall.1D``).

``ffs_moco`` computes the same thing while it has the raw data loaded
(``-outcount``); this tool is the standalone form, and the one to diff against
3dToutcount.

Usage:
    ffs_util_outcount -input run1.nii.gz run2.nii.gz -prefix outcount_rall.1D
    ffs_util_outcount -input run1.nii.gz -censor_outliers -censor out_censor.1D
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import add_device_arg, add_verbose_arg, setup_device, spinner
from fastfuncstuff.processing import censor as C


def add_outlier_censor_args(parser_or_group) -> None:
    """``-censor_outliers [F]`` / ``-skip_first_outliers N``, shared with ffs_moco."""
    parser_or_group.add_argument(
        "-censor_outliers",
        nargs="?",
        const=C.DEFAULT_OUTLIER_LIMIT,
        type=float,
        default=None,
        metavar="F",
        help="Censor TRs whose outlier fraction is strictly greater than F "
        f"(afni_proc -regress_censor_outliers). Bare flag = {C.DEFAULT_OUTLIER_LIMIT}.",
    )
    parser_or_group.add_argument(
        "-skip_first_outliers",
        type=int,
        default=0,
        metavar="N",
        help="Never outlier-censor the first N TRs of each run (afni_proc "
        "-regress_skip_first_outliers) -- for leading volumes dropped another way.",
    )
    parser_or_group.add_argument(
        "-outlier_polort",
        type=int,
        default=None,
        metavar="P",
        help="Legendre detrend order for outlier counting (default per run: "
        "1 + floor(TR*nt/150), afni_proc's and 3dDeconvolve's rule).",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = FfsArgumentParser(
        formatter_class=FfsHelpFormatter,
        prog="ffs_util_outcount",
        description="Per-TR fraction of outlier voxels (AFNI 3dToutcount -automask "
        "-fraction -legendre), GPU, with optional outlier censor file.",
    )
    parser.add_argument(
        "-input",
        required=True,
        nargs="+",
        help="One 4D dataset per run; each run is counted separately (own mask, "
        "own polort) and the results are concatenated in the order given.",
    )
    parser.add_argument(
        "-prefix",
        default=None,
        metavar="FILE.1D",
        help="Write the fractions here (default: stdout, as 3dToutcount).",
    )
    parser.add_argument(
        "-mask",
        default=None,
        help="Count only voxels in this mask (default: AFNI automask of each run, "
        "afni_proc's -automask).",
    )
    parser.add_argument(
        "-count",
        action="store_true",
        help="Write outlier voxel counts instead of fractions.",
    )
    parser.add_argument(
        "-qthr",
        type=float,
        default=C.DEFAULT_QTHR,
        help="Tail probability q in alpha = qginv(q/nt).",
    )
    add_outlier_censor_args(parser)
    parser.add_argument(
        "-censor",
        default=None,
        metavar="FILE.1D",
        help="Write the keep mask (1=keep, 0=censor) here. Requires -censor_outliers.",
    )
    add_device_arg(parser)
    add_verbose_arg(parser, default=1)
    args = parser.parse_args(argv)
    if args.censor and args.censor_outliers is None:
        parser.error("-censor needs -censor_outliers [F]")
    if args.qthr <= 0.0 or args.qthr >= 0.999:
        parser.error("-qthr must be in (0, 0.999)")
    return args


def main(argv: list[str] | None = None) -> None:
    from fastfuncstuff.io.afni import get_tr_from_file
    from fastfuncstuff.processing.io import load_image

    args = parse_args(argv)
    device = setup_device(args.device)
    verb = args.verb
    t0 = time.time()

    mask = None
    if args.mask is not None:
        m, _ = load_image(args.mask)
        mask = (m[0] if m.ndim == 4 else m) != 0

    fractions, counts, run_lengths = [], [], []
    for path in args.input:
        with spinner(f"Loading {Path(path).name}", enabled=verb >= 1):
            data, _ = load_image(path, device=device)
        if data.ndim != 4:
            raise SystemExit(f"ffs_util_outcount: {path} is not 4D (shape {tuple(data.shape)})")
        if mask is not None and tuple(mask.shape) != tuple(data.shape[1:]):
            raise SystemExit(
                f"ffs_util_outcount: mask grid {tuple(mask.shape)} != {path} grid {tuple(data.shape[1:])}"
            )
        tr = get_tr_from_file(path) if args.outlier_polort is None else None
        frac, nvox = C.outlier_fraction_4d(
            data.float(),
            tr=tr,
            polort=args.outlier_polort,
            mask=mask,
            qthr=args.qthr,
            progress=verb >= 1,
        )
        fractions.append(frac)
        counts.append(np.rint(frac * nvox[0]).astype(int))
        run_lengths.append(data.shape[0])
        if verb >= 1:
            p = args.outlier_polort
            if p is None:
                p = C.default_outlier_polort(tr or 0.0, data.shape[0])
            print(
                f"  {Path(path).name}: {data.shape[0]} TRs, {nvox[0]:,} voxels, polort {p}, "
                f"max fraction {frac.max():.4f}",
                file=sys.stderr,
            )
            if frac[0] > C.PRE_STEADY_STATE_LIMIT:
                print(
                    f"  ** TR #0 outliers ({frac[0]:.2f}): possible pre-steady state TRs",
                    file=sys.stderr,
                )
        del data
    frac_all = np.concatenate(fractions)

    if args.count:
        lines = [f"{c:6d}" for c in np.concatenate(counts)]
    else:
        lines = [f"{f:0.5f}" for f in frac_all]
    if args.prefix:
        Path(args.prefix).write_text("\n".join(lines) + "\n")
    else:
        sys.stdout.write("\n".join(lines) + "\n")

    if args.censor_outliers is not None:
        keep = C.censor_from_outliers(
            frac_all, args.censor_outliers, run_lengths, skip_first=args.skip_first_outliers
        )
        if args.censor:
            C.write_1d(args.censor, keep, fmt="%d")
        if verb >= 1:
            print(
                f"  censored {int((keep == 0).sum())} of {keep.size} TRs "
                f"(outlier fraction > {args.censor_outliers:g})",
                file=sys.stderr,
            )
    if verb >= 1:
        print(f"  done in {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
