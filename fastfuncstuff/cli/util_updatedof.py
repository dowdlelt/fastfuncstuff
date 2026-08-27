"""ffs_util_updatedof — adjust the degrees of freedom of a statistical bucket.

After NORDIC (or any component-removal denoising) the residual degrees of
freedom are lower than the GLM's model dof, so the t/F p-values AFNI reports are
too optimistic. This tool rewrites the statistics at the corrected dof: for each
t/F sub-brick it computes ``new_dof = dof - adjustment`` and converts the
statistic to a z-score at that dof (AFNI-faithful), inserting the z-score after
each stat sub-brick and tagging it so AFNI reports the right p-value.

``-adjust_dof`` is either a number (subtract that many dof everywhere) or a
per-voxel NIfTI map of dof lost (float, rounded to int) — e.g. the summed count
of NORDIC components removed per voxel. It may be given **more than once**: lost
dof is additive, so every source (denoising, missing-data censoring, …) is just
summed.

``-adjust_dof_set PERRUN INCLUDE`` handles the case where the dof cost is
per-run *and* the voxel did not survive every run. ``PERRUN`` is a 4-D
``(X, Y, Z, n_runs)`` map of dof lost per run (or a scalar applied to each run),
``INCLUDE`` is the run-inclusion mask from ``ffs_reml -save_runmask``. Only the
runs that actually contributed to a voxel are charged to it. Also repeatable,
and freely mixed with ``-adjust_dof``.

If the input already carries z-score sub-bricks (a previous run), they are
recomputed in place from each stat brick's model dof — so re-running updates
rather than duplicates.

Examples
--------
    ffs_util_updatedof -input reml.stats.nii.gz -adjust_dof lost_dof.nii.gz \\
        -prefix reml.stats.dofadj.nii.gz
    ffs_util_updatedof -input ols.stats.nii.gz -adjust_dof 5 -prefix ols.dofadj.nii.gz

    # NORDIC cost per run, charged only for the runs each voxel survived,
    # plus a flat 3 dof from something else:
    ffs_util_updatedof -input reml.stats.nii.gz \\
        -adjust_dof_set nordic_dof_per_run.nii.gz reml.runmask.nii.gz \\
        -adjust_dof 3 -prefix reml.stats.dofadj.nii.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.stats.dof_adjust import (
    combine_dof_adjustments,
    resolve_dof_adjust_arg,
    resolve_dof_adjust_set,
    update_dof_in_file,
)


def _build_parser() -> argparse.ArgumentParser:
    p = FfsArgumentParser(
        prog="ffs_util_updatedof",
        description=__doc__,
        formatter_class=FfsHelpFormatter,
    )
    p.add_argument(
        "-input",
        required=True,
        metavar="FILE",
        help="Statistical bucket (NIfTI) written by ffs_reml (must carry AFNI "
        "BRICK_STATAUX so its t/F sub-bricks and dof are known).",
    )
    p.add_argument(
        "-adjust_dof",
        "-adjust-dof",
        action="append",
        default=None,
        metavar="MAP|N",
        dest="adjust_dof",
        help="Degrees of freedom to REMOVE: a number (e.g. 5) subtracted "
        "everywhere, or a per-voxel NIfTI map of dof lost (rounded to int). "
        "May be repeated; all adjustments are summed.",
    )
    p.add_argument(
        "-adjust_dof_set",
        "-adjust-dof-set",
        action="append",
        nargs=2,
        default=None,
        metavar=("PERRUN", "INCLUDE"),
        dest="adjust_dof_set",
        help="Per-run dof loss charged only for the runs a voxel survived. "
        "PERRUN is a 4D (X,Y,Z,n_runs) map of dof lost per run, or a scalar "
        "applied to every run; INCLUDE is the 4D run-inclusion mask from "
        "ffs_reml -save_runmask. May be repeated and mixed with -adjust_dof.",
    )
    p.add_argument(
        "-prefix",
        required=True,
        metavar="FILE",
        help="Output bucket path (.nii/.nii.gz). Gets the inserted z-score "
        "sub-bricks and updated AFNI stat metadata.",
    )
    p.add_argument(
        "-save_invalid",
        "-save-invalid",
        metavar="FILE",
        dest="save_invalid",
        default=None,
        help="Where to write the mask of voxels whose new dof <= 0 (invalid "
        "statistics, clamped to dof=1). Default: <prefix>_invalid_dof.nii.gz.",
    )
    p.add_argument(
        "-overwrite",
        action="store_true",
        help="Allow overwriting an existing -prefix.",
    )
    p.add_argument("-quiet", action="store_true", help="Suppress progress output.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.adjust_dof and not args.adjust_dof_set:
        print(
            "ERROR: give at least one of -adjust_dof or -adjust_dof_set.",
            file=sys.stderr,
        )
        return 1

    if Path(args.prefix).exists() and not args.overwrite:
        print(f"ERROR: output {args.prefix} exists (use -overwrite).", file=sys.stderr)
        return 1

    verbose = not args.quiet
    try:
        pieces: list = []
        for a in args.adjust_dof or []:
            pieces.append(resolve_dof_adjust_arg(a))
        for per_run, include in args.adjust_dof_set or []:
            pieces.append(resolve_dof_adjust_set(per_run, include))
        adjust = combine_dof_adjustments(pieces)
        if verbose and len(pieces) > 1:
            print(f"  Summed {len(pieces)} dof adjustments")
        result = update_dof_in_file(
            args.input,
            adjust,
            args.prefix,
            invalid_path=args.save_invalid,
            verbose=verbose,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if result.n_stat_bricks == 0:
        print("WARNING: no t/F statistic sub-bricks found to adjust.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
