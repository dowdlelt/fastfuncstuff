"""CLI: turn a QC folder into a subject table and one HTML page of its figures.

Command: ffs_qc_report (registered as entry point in pyproject.toml)

Every tool given ``-qc DIR`` drops self-explanatory PNGs and a metrics record into
DIR. This writes, beside them:

    qc_review.tsv   one row per run, one column per tool.metric
    qc_review.txt   the same numbers as key : value blocks, warnings marked
    index.html      the PNGs in folder order under a table of the numbers

Usage:
    ffs_qc_report sub-01/qc
    ffs_qc_report sub-*/qc -no_html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastfuncstuff.cli_help import FfsArgumentParser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = FfsArgumentParser(
        prog="ffs_qc_report",
        description="Build the subject table and HTML page for one or more QC folders "
        "(written by any ffs tool's -qc DIR).",
        epilog="Compare subjects with ffs_qc_table once each folder has its report.",
    )
    parser.add_argument("qc_dirs", nargs="+", metavar="DIR", help="QC folder(s).")
    parser.add_argument(
        "-title", default="", help="Page title (default: the folder's parent name)."
    )
    parser.add_argument("-no_html", action="store_true", help="Write only the tables.")
    parser.add_argument("-verb", type=int, default=1, help="Verbosity (0 = quiet).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from fastfuncstuff.qc.report import build_report

    args = parse_args(argv)
    status = 0
    for d in args.qc_dirs:
        if not Path(d).is_dir():
            print(f"Error: not a directory: {d}", file=sys.stderr)
            status = 1
            continue
        out = build_report(d, title=args.title, html_page=not args.no_html)
        if args.verb >= 1:
            page = f", {out['html']}" if "html" in out else ""
            print(f"{d}: {out['records']} records -> {out['tsv']}{page}")
    return status


if __name__ == "__main__":
    sys.exit(main())
