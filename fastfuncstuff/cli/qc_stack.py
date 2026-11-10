"""CLI: render image stacks that should line up as QC figures.

Command: ffs_qc_stack (registered as entry point in pyproject.toml)

A stack is a 4-D file whose volumes are images that a step claims to have put
in one space: every run's mean after cross-run alignment, each session after
cross-session alignment, the fieldmaps. autoproc writes these as stageNN.QC.*;
any 4-D file works. Per stack this writes, into -qc DIR:

    a ranking   every image's spatial correlation with the stack's median
    a flicker   an animated PNG stepping through the images under the median's edges

Usage:
    ffs_qc_stack -qc qc stage06.QC.xrun.nii.gz stage10.QC.final.nii.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastfuncstuff.cli_help import FfsArgumentParser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = FfsArgumentParser(
        prog="ffs_qc_stack",
        description="Ranking + flicker QC figures for 4-D stacks of images that should line up.",
    )
    parser.add_argument("stacks", nargs="+", metavar="STACK", help="4-D NIfTI stack(s).")
    parser.add_argument("-qc", required=True, metavar="DIR", help="QC folder to write into.")
    parser.add_argument(
        "-overwrite",
        action="store_true",
        help="Re-render a stack whose figures are already newer than it.",
    )
    parser.add_argument("-verb", type=int, default=1, help="Verbosity (0 = quiet).")
    return parser.parse_args(argv)


def render_stack(path: Path, qc_dir: Path, overwrite: bool, verb: int) -> bool:
    import nibabel as nib
    import numpy as np

    from fastfuncstuff.io.headers import read_brick_labels
    from fastfuncstuff.qc import checks
    from fastfuncstuff.qc.record import QCRecord, label_from_path

    label = label_from_path(path)
    rec = QCRecord(qc_dir, "stack", label, order=55)
    out = rec.figure_path(2, "flicker")
    if not overwrite and out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return False
    img = nib.load(str(path))
    if len(img.shape) != 4 or img.shape[3] < 2:
        if verb >= 1:
            print(f"  {path.name}: not a 4-D stack of >= 2 images, skipped")
        return False
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    frames = np.ascontiguousarray(data.transpose(3, 2, 1, 0))
    labels = read_brick_labels(img)
    stage = label.split(".QC.")[-1] if ".QC." in label else label
    rec.info["stack"] = str(path)
    checks.stack_check(rec, 1, frames, img.affine, labels, what=stage)
    rec.write(verb)
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    qc_dir = Path(args.qc)
    qc_dir.mkdir(parents=True, exist_ok=True)
    status = 0
    for s in args.stacks:
        p = Path(s)
        if not p.is_file():
            print(f"  {s}: missing, skipped", file=sys.stderr)
            status = 1
            continue
        render_stack(p, qc_dir, args.overwrite, args.verb)
    return status


if __name__ == "__main__":
    sys.exit(main())
