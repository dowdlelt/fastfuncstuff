"""CLI: one cross-subject QC table from many QC folders, with flags.

Command: ffs_qc_table (registered as entry point in pyproject.toml)

The figures are for understanding one subject; this is for comparing many. Each
row is one (subject, run); each column a tool.metric from the folders' records.
Rows are flagged when a metric crosses its tool's warning limit, a limit you give
with -flag, or sits beyond a robust z-score among its peers.

Writes PREFIX.tsv (the table), PREFIX_flags.tsv (what was flagged and why) and
PREFIX.png (one strip plot per metric, flagged runs labelled).

Usage:
    ffs_qc_table sub-*/qc -prefix group_qc
    ffs_qc_table sub-*/qc -prefix group_qc -flag "moco.censor_frac > 0.1" -flag "moco.tsnr_median_corrected < 40"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastfuncstuff.cli_help import FfsArgumentParser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = FfsArgumentParser(
        prog="ffs_qc_table",
        description="Gather QC records from many folders into one table and flag outliers "
        "(the cross-subject companion to ffs_qc_report).",
    )
    parser.add_argument(
        "qc_dirs",
        nargs="+",
        metavar="DIR",
        help="QC folders. A folder named qc is named for its parent (sub-01/qc -> sub-01).",
    )
    parser.add_argument("-prefix", default="group_qc", help="Output stem (default: group_qc).")
    parser.add_argument(
        "-flag",
        action="append",
        default=[],
        metavar="RULE",
        help="Extra limit, e.g. 'moco.censor_frac > 0.1' (repeatable; > >= < <=).",
    )
    parser.add_argument(
        "-z",
        type=float,
        default=3.5,
        help="Flag values beyond this robust z-score (median/MAD) among all rows of a "
        "column; needs >= 5 rows. 0 disables.",
    )
    parser.add_argument("-no_png", action="store_true", help="Skip the strip-plot overview.")
    parser.add_argument("-verb", type=int, default=1, help="Verbosity (0 = quiet).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from fastfuncstuff.qc.report import fmt_value
    from fastfuncstuff.qc.table import build_table

    args = parse_args(argv)
    dirs = [d for d in args.qc_dirs if Path(d).is_dir()]
    if not dirs:
        print("Error: none of the given QC folders exist.", file=sys.stderr)
        return 1
    try:
        out = build_table(dirs, args.prefix, args.flag, args.z, png=not args.no_png)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.verb >= 1:
        n_subj = len({r["subject"] for r in out["rows"]})
        print(
            f"{n_subj} subjects, {len(out['rows'])} rows, {len(out['columns'])} metrics -> {out['tsv']}"
        )
        if out.get("png"):
            print(f"overview -> {out['png']}")
        if out["flags"]:
            print(f"{len(out['flags'])} flags -> {out['flags_tsv']}")
            for fl in out["flags"][:40]:
                print(
                    f"  {fl.subject:<14s} {fl.label:<28s} {fl.column:<30s} {fmt_value(fl.value):>10s}  {fl.reason}"
                )
            if len(out["flags"]) > 40:
                print(f"  ... {len(out['flags']) - 40} more in {out['flags_tsv']}")
        else:
            print("no flags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
