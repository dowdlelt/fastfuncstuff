"""CLI for deformation-regularity QC of a nonlinear warp.

Command: ffs_util_warpqc (registered as entry point in pyproject.toml)

Reports whether a warp field is anatomically plausible: how much of it folds
(det(J) <= 0), how far the Jacobian strays, how far voxels move, and how bendy
the field is. The companion to ``ffs_util_cost`` — that one asks whether two
images ended up looking alike, this one asks whether the deformation that got
them there is one you would believe.

You need both. A warp with enough degrees of freedom drives any single
similarity metric to its optimum by folding tissue around, so a warp that wins
on similarity and fails here has not won.

Usage:
    ffs_util_warpqc -warp out_WARP.nii.gz -mask brain.nii.gz
    ffs_util_warpqc -warp a_WARP.nii.gz b_WARP.nii.gz -mask brain.nii.gz -json qc.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastfuncstuff.cli_help import FfsHelpFormatter
from fastfuncstuff.cli_utils import add_device_arg, add_verbose_arg, setup_device
from fastfuncstuff.processing.io import load_image, load_warp_field
from fastfuncstuff.processing.nwarpforge import _nifti_mm_to_voxels, compute_cardinal_affine
from fastfuncstuff.processing.warpqc import (
    DEFAULT_MARGINAL_NEG_FRAC,
    DEFAULT_MAX_JAC,
    DEFAULT_MAX_NEG_FRAC,
    DEFAULT_MAX_NEG_VOXELS,
    DEFAULT_MIN_JAC,
    FAIL,
    format_warpqc,
    pad_mask_to_field,
    regularity_cautions,
    regularity_verdict,
    remedies,
    warp_regularity,
)
from fastfuncstuff.utils import REGISTRATION_TF32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffs_util_warpqc",
        description="Deformation-regularity QC for a nonlinear warp field "
        "(folding, Jacobian range, displacement, bending energy).",
        formatter_class=FfsHelpFormatter,
        epilog="""Reading the output:
  det(J) < 0     the map folded there — tissue turned inside out. A handful of
                 voxels grazing zero is stencil noise; a region of them is not.
  det(J) 1%/99%  how much the warp compresses/expands typical tissue. Far from
                 1.0 means large volume change, plausible for subject-to-template
                 and suspicious within subject.
  |disp|         how far voxels moved, in mm.
  bending        how wiggly the field is; compare between candidates, not
                 against an absolute number -- and only between candidates on
                 the same grid, since it averages over the whole field and
                 padding differs with each base's support box.
""",
    )
    parser.add_argument(
        "-warp",
        required=True,
        nargs="+",
        metavar="WARP",
        help="Displacement field(s) as saved by -save_warp (4D, 3 components)",
    )
    parser.add_argument(
        "-mask",
        default=None,
        help="Restrict statistics to this mask (nonzero = in). Strongly "
        "recommended: the field outside the brain is unconstrained, and letting "
        "it into the percentiles buries real folding in noise. Aligned from the "
        "mask and warp affines when the warp sits on a padded grid.",
    )
    parser.add_argument(
        "-units",
        "--units",
        choices=["mm", "voxels"],
        default="mm",
        help="Displacement units on disk. Every ffs tool that saves a warp for "
        "AFNI interop writes DICOM mm (the default); ffs_util_pcwarp writes raw "
        "voxel displacements, which need -units voxels to be read correctly.",
    )

    gate = parser.add_argument_group("Pass/fail thresholds")
    gate.add_argument(
        "-max_neg_voxels",
        type=int,
        default=DEFAULT_MAX_NEG_VOXELS,
        help=f"Folded voxels tolerated outright (default: {DEFAULT_MAX_NEG_VOXELS})",
    )
    gate.add_argument(
        "-max_neg_frac",
        type=float,
        default=DEFAULT_MAX_NEG_FRAC,
        help=f"...or this fraction of in-mask voxels, whichever is larger "
        f"(default: {DEFAULT_MAX_NEG_FRAC})",
    )
    gate.add_argument(
        "-min_jac",
        type=float,
        default=DEFAULT_MIN_JAC,
        help=f"Fail if the 1st-percentile det(J) is below this (default: {DEFAULT_MIN_JAC})",
    )
    gate.add_argument(
        "-max_jac",
        type=float,
        default=DEFAULT_MAX_JAC,
        help=f"Fail if the 99th-percentile det(J) is above this (default: {DEFAULT_MAX_JAC})",
    )

    gate.add_argument(
        "-marginal_neg_frac",
        type=float,
        default=DEFAULT_MARGINAL_NEG_FRAC,
        help=f"Folding above the budget but below this fraction grades MARGINAL "
        f"(recoverable with more regularization) rather than FAIL "
        f"(default: {DEFAULT_MARGINAL_NEG_FRAC})",
    )

    parser.add_argument("-json", dest="out_json", default=None, help="Write the metrics as JSON")
    add_device_arg(parser)
    add_verbose_arg(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = setup_device(args.device, tf32=REGISTRATION_TF32)

    mask = None
    mask_info = None
    if args.mask:
        m, mask_info = load_image(args.mask, device=device)
        mask = (m > 0).float()

    results, worst_ok = {}, True
    for path in args.warp:
        xd, yd, zd, header = load_warp_field(path, device=device)
        cardinal = compute_cardinal_affine(header["affine"])
        voxdims = tuple(float(abs(cardinal[i, i])) or 1.0 for i in range(3))
        if args.units == "mm":
            # save_warp_field writes AFNI DICOM-mm. Convert DICOM -> NIfTI/RAS mm,
            # then into the padded field grid's voxel coordinates before derivatives.
            xd, yd, zd = _nifti_mm_to_voxels(-xd, -yd, zd, cardinal)
        m = (
            None
            if mask is None
            else pad_mask_to_field(
                mask,
                tuple(xd.shape),
                mask_affine=mask_info["affine"],
                field_affine=header["affine"],
            )
        )
        qc = warp_regularity(xd, yd, zd, mask=m, voxdims=voxdims)
        grade, reasons = regularity_verdict(
            qc,
            args.max_neg_voxels,
            args.max_neg_frac,
            args.marginal_neg_frac,
        )
        # Reported, never graded on: an extreme Jacobian is unusual anatomy, not
        # broken anatomy. See regularity_cautions.
        cautions = regularity_cautions(qc, args.min_jac, args.max_jac)
        worst_ok = worst_ok and grade != FAIL
        results[Path(path).name] = {
            **qc.as_dict(),
            "grade": grade,
            "reasons": reasons,
            "cautions": cautions,
            "remedies": remedies(reasons + cautions),
        }
        if args.verb >= 1:
            print(format_warpqc(qc, Path(path).name))
            print()
        del xd, yd, zd

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2))
        if args.verb >= 1:
            print(f"Wrote {args.out_json}")

    # Non-zero exit only on an outright FAIL, so a pipeline can gate on it while
    # still letting a marginal-but-usable warp through.
    return 0 if worst_ok else 1


if __name__ == "__main__":
    sys.exit(main())
