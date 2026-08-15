"""CLI for reporting every alignment cost functional (AFNI 3dAllineate -allcostX).

Command: ffs_util_cost (registered as entry point in pyproject.toml)

Evaluates all fourteen cost functionals for a base/source pair *as they sit* —
no optimisation — so you can see what each functional thinks of an alignment
you already have. Every value is in AFNI's convention: **lower is better**.

Give more than one source and it also ranks them by consensus across the
functionals, which is the honest way to compare two candidate alignments: the
cost that was optimised is guaranteed to like its own answer, the other
thirteen are not.

Usage:
    # What do all the costs say about this pair, where they currently sit?
    ffs_util_cost -base mni.nii.gz -source anat.nii.gz

    # Which of these three alignments is actually best?
    ffs_util_cost -base mni.nii.gz -source qwarp.nii.gz optiwarp.nii.gz formwarp.nii.gz

    # Evaluate at a transform, rather than where the source sits
    ffs_util_cost -base mni.nii.gz -source anat.nii.gz -1Dmatrix_apply mat.aff12.1D
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from fastfuncstuff.cli_utils import add_device_arg, add_verbose_arg, setup_device
from fastfuncstuff.processing.affine import apply_affine, load_matrix_1D
from fastfuncstuff.processing.allcost import (
    ALL_COSTS,
    COST_INFO,
    build_cost_inputs,
    consensus_rank,
    evaluate_all_costs,
    format_cost_table,
)
from fastfuncstuff.processing.allineate import (
    _compute_grid_matrix,
    _compute_source_validity_mask,
    _voxdims_from_header,
)
from fastfuncstuff.processing.io import load_image
from fastfuncstuff.processing.mask import automask
from fastfuncstuff.processing.weight import compute_weight_image
from fastfuncstuff.utils import REGISTRATION_TF32


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffs_util_cost",
        description="Report all alignment cost functionals for an image pair "
        "(AFNI 3dAllineate -allcostX). Lower is better for every value.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(),
    )
    parser.add_argument("-base", required=True, help="Base/reference dataset")
    parser.add_argument(
        "-source",
        required=True,
        nargs="+",
        metavar="DSET",
        help="Source dataset(s). More than one is scored and ranked by consensus "
        "across the functionals.",
    )
    parser.add_argument(
        "-1Dmatrix_apply",
        "-1Dmatrix-apply",
        dest="matrix_apply",
        default=None,
        metavar="FILE",
        help="Apply this aff12 matrix (base voxel -> source voxel) to the source "
        "before evaluating, instead of scoring it where it sits.",
    )

    cost_group = parser.add_argument_group("Cost selection")
    cost_group.add_argument(
        "-cost",
        nargs="+",
        default=None,
        metavar="NAME",
        choices=ALL_COSTS,
        help=f"Evaluate only these functionals (default: all). Choices: {', '.join(ALL_COSTS)}",
    )
    cost_group.add_argument(
        "-describe",
        action="store_true",
        help="Print what each reported number actually is, alongside the value",
    )
    cost_group.add_argument(
        "-blok",
        "-bloktype",
        dest="bloktype",
        choices=["tohd", "rhdd", "cube"],
        default="tohd",
        help="Blok shape for the lpc/lpa family (default: tohd, like 3dAllineate)",
    )
    cost_group.add_argument(
        "-blokrad", type=float, default=None, help="Blok radius in mm (default: auto)"
    )
    cost_group.add_argument(
        "-nbin",
        type=int,
        default=None,
        help="2D histogram bins for the mi/nmi/je/hel/cr family (default: auto)",
    )

    mask_group = parser.add_argument_group("Masking / weighting")
    mask_group.add_argument(
        "-mask", default=None, help="Restrict evaluation to this mask (nonzero = in)"
    )
    mask_group.add_argument(
        "-autoweight",
        action="store_true",
        default=True,
        help="Weight by base intensity, as 3dAllineate does (default: on)",
    )
    mask_group.add_argument(
        "-noautoweight",
        action="store_true",
        help="No weighting — every in-mask voxel counts equally",
    )
    mask_group.add_argument(
        "-automask",
        dest="source_automask",
        action="store_true",
        help="Automask the source to exclude its background before evaluating",
    )
    mask_group.add_argument(
        "-whole_volume",
        "-whole-volume",
        dest="whole_volume",
        action="store_true",
        help="Score every voxel including background, the way 3dAllineate "
        "-allcostX does. Use it only to compare against AFNI: agreement in the "
        "background (both images are zero out there) inflates every functional "
        "and squashes the difference between good and bad alignments.",
    )
    mask_group.add_argument(
        "-nmatch",
        "-n_match",
        dest="n_match",
        type=float,
        default=1.0,
        help="Match points to score, as in ffs_allineate: <=1.0 is a FRACTION of "
        "the in-mask voxels (default 1.0 = all of them), >1.0 an absolute count. "
        "Only lower it if a big grid is slow — there is no optimiser loop here.",
    )

    out_group = parser.add_argument_group("Output")
    out_group.add_argument(
        "-1D",
        dest="out_1D",
        default=None,
        metavar="FILE",
        help="Write the values as a 1D table (one row per source, one column per cost)",
    )
    out_group.add_argument(
        "-json", dest="out_json", default=None, metavar="FILE", help="Write the values as JSON"
    )

    hw_group = parser.add_argument_group("Hardware")
    add_device_arg(hw_group)
    add_verbose_arg(hw_group)
    return parser.parse_args(argv)


def _epilog() -> str:
    lines = [
        "Cost functionals (all reported so that LOWER == better aligned):",
        "",
    ]
    for name, (long, desc) in COST_INFO.items():
        lines.append(f"  {name:<5s} {long:<20s} {desc}")
    lines += [
        "",
        "Notes:",
        "  * lpc/lpa are the robust workhorses (lpc for cross-modal, lpa for",
        "    like-to-like); mi/nmi/je/hel/cr* are mostly for comparison.",
        "  * A single functional can be fooled. When you are choosing between",
        "    alignments, look at the consensus rank, not one row.",
        "  * The values match 3dAllineate -allcostX in convention, not to the",
        "    last decimal: the weight image and histogram binning are ours.",
    ]
    return "\n".join(lines)


def _prepare_weight(
    base: torch.Tensor,
    mask_path: str | None,
    autoweight: bool,
    validity: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Build the evaluation weight the same way allineate builds its own."""
    weight = None
    if autoweight:
        weight = compute_weight_image(
            base,
            edge_fraction=0.05,
            median_radius=2.25,
            clusterize=True,
            hist_cliplevel=True,
        )
    if mask_path is not None:
        mask, _ = load_image(mask_path, device=device)
        m = (mask > 0).float()
        weight = m if weight is None else weight * m
    if validity is not None:
        v = validity.float()
        weight = v if weight is None else weight * v
    return weight


def _source_on_base(
    source: torch.Tensor,
    source_header: dict,
    base: torch.Tensor,
    base_header: dict,
    matrix_apply: str | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, float | None]:
    """Put the source on the base grid; return (volume, validity mask, overlap)."""
    eye = torch.eye(4, device=device)
    align = eye
    if matrix_apply is not None:
        # allineate saves the transform in DICOM mm; converting it here with both
        # affines is what turns it back into the base-voxel -> source-voxel map
        # that apply_affine wants.
        align = load_matrix_1D(matrix_apply, base_header["affine"], source_header["affine"]).to(
            device
        )
    elif source_header is not None and base_header is not None:
        align = _compute_grid_matrix(source_header["affine"], base_header["affine"], device)

    if torch.allclose(align, eye):
        return source, None, None

    validity = _compute_source_validity_mask(source.shape, base.shape, align)
    on_base = apply_affine(source, align, base.shape, zero_outside=True) * validity.float()

    # AFNI's overlap fraction: the shared voxels over the smaller of the two
    # object masks, which is what the lpc+/lpa+ 'ov' term penalises.
    base_obj = base > 0
    src_obj = validity & (on_base != 0)
    denom = min(int(base_obj.sum()), int(src_obj.sum()))
    overlap = float((base_obj & src_obj).sum()) / denom if denom > 0 else None
    return on_base, validity, overlap


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = setup_device(args.device, tf32=REGISTRATION_TF32)
    verb = args.verb
    autoweight = args.autoweight and not args.noautoweight

    t0 = time.time()
    base, base_header = load_image(args.base, device=device)
    if base.ndim == 4:
        base = base[..., 0]
    voxdims = _voxdims_from_header(base_header)

    results: dict[str, dict[str, float]] = {}
    for path in args.source:
        source, source_header = load_image(path, device=device)
        if source.ndim == 4:
            source = source[..., 0]

        on_base, validity, overlap = _source_on_base(
            source, source_header, base, base_header, args.matrix_apply, device
        )
        if args.source_automask:
            on_base = on_base * automask(on_base, device=device).float()

        weight = _prepare_weight(base, args.mask, autoweight, validity, device)
        inputs = build_cost_inputs(
            base,
            on_base,
            weight,
            voxdims,
            args.n_match,
            args.bloktype,
            overlap,
            args.whole_volume,
        )
        vals = evaluate_all_costs(
            inputs,
            costs=args.cost,
            bloktype=args.bloktype,
            blokrad=args.blokrad,
            nbin=args.nbin,
        )
        results[Path(path).name] = vals

        if verb >= 1:
            print(f"\n{Path(path).name}" + (f"  (overlap {overlap:.3f})" if overlap else ""))
            print(format_cost_table(vals, describe=args.describe))

        del source, on_base, validity, weight, inputs

    if len(results) > 1 and verb >= 1:
        print("\nConsensus rank (mean rank across functionals, lower == better):")
        for i, (name, score) in enumerate(consensus_rank(results), start=1):
            print(f"  {i}. {name:<40s} {score:6.2f}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2))
        if verb >= 1:
            print(f"\nWrote {args.out_json}")

    if args.out_1D:
        names = list(next(iter(results.values())))
        lines = ["# " + "  ".join(f"{n:>12s}" for n in names)]
        for src, vals in results.items():
            lines.append("  ".join(f"{vals[n]:12.6f}" for n in names) + f"  # {src}")
        Path(args.out_1D).write_text("\n".join(lines) + "\n")
        if verb >= 1:
            print(f"Wrote {args.out_1D}")

    if verb >= 1:
        print(f"\nTotal time: {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
