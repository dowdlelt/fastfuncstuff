"""CLI for auto-cropping a dataset to its non-zero bounding box (AFNI 3dAutobox).

Command: ffs_util_autobox (registered as entry point in pyproject.toml)

Finds the box that holds the data and, with ``-prefix``, writes the dataset
cropped to it. Nothing moves in space — the matrix shrinks and the header origin
walks to the new corner, so the output overlays the input exactly.

Usage:
    ffs_util_autobox -input anat.nii.gz -prefix anat_box.nii.gz
    ffs_util_autobox -input epi.nii.gz -npad 4 -prefix epi_box.nii.gz
    ffs_util_autobox -input anat.nii.gz -extent_ijk
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from fastfuncstuff.cli_help import FfsHelpFormatter
from fastfuncstuff.cli_utils import add_verbose_arg, parse_device_arg, spinner
from fastfuncstuff.processing.autobox import (
    DEFAULT_CLFRAC,
    DEFAULT_PEELCOUNT,
    autobox_bounds,
    crop_to_bounds,
    pad_bounds,
)
from fastfuncstuff.processing.grid import afni_orient_code, grid_extent_rai
from fastfuncstuff.processing.io import load_image, save_image
from fastfuncstuff.utils import REGISTRATION_TF32, configure_torch_backends


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=FfsHelpFormatter,
        prog="ffs_util_autobox",
        description="Crop a dataset to the box around its non-zero data, "
        "preserving the header geometry (GPU, matching AFNI 3dAutobox).",
    )
    parser.add_argument("-input", "-inset", dest="input", help="Input dataset (.nii/.nii.gz)")
    parser.add_argument(
        "-prefix",
        default=None,
        help="Write the cropped dataset here. Without it, only the extents are reported.",
    )
    parser.add_argument(
        "-noclust",
        action="store_true",
        help="Skip clip-level thresholding and clustering: keep every non-zero voxel. "
        "Use for masks and label volumes, where any voxel matters.",
    )
    parser.add_argument(
        "-npad",
        type=int,
        default=0,
        metavar="NNN",
        help="Pad the box by NNN voxels a side. May grow past the input matrix "
        "(zero-filled); negative crops tighter.",
    )
    parser.add_argument(
        "-npad_safety_on",
        "-npad-safety-on",
        action="store_true",
        help="Clamp the padded box inside the input matrix, so the output is never larger.",
    )

    ext = parser.add_argument_group("extent reporting")
    ext.add_argument("-extent", action="store_true", help="Print the spatial extent of the box")
    ext.add_argument(
        "-extent_xyz_quiet",
        "-extent-xyz-quiet",
        action="store_true",
        help="The -extent numbers alone, RLAPIS order, no labels",
    )
    ext.add_argument(
        "-extent_ijk",
        "-extent-ijk",
        action="store_true",
        help="Print imin imax jmin jmax kmin kmax",
    )
    ext.add_argument(
        "-extent_ijk_to_file", "-extent-ijk-to-file", metavar="FF", help="Write those 6 ijk to FF"
    )
    ext.add_argument(
        "-extent_ijk_midslice",
        "-extent-ijk-midslice",
        action="store_true",
        help="Print the 3 ijk midslices of the box",
    )
    ext.add_argument(
        "-extent_ijkord",
        "-extent-ijkord",
        action="store_true",
        help="Print the ijk extents as a 3x3 table ordered by x/y/z axis (see NOTES)",
    )
    ext.add_argument(
        "-extent_ijkord_to_file",
        "-extent-ijkord-to-file",
        metavar="FF",
        help="Write that table to FF",
    )
    ext.add_argument(
        "-extent_xyz_to_file",
        "-extent-xyz-to-file",
        metavar="GG",
        help="Write the 6 xyz extents to GG",
    )
    ext.add_argument(
        "-extent_xyz_midslice",
        "-extent-xyz-midslice",
        action="store_true",
        help="Print the 3 xyz midslices of the box",
    )

    tune = parser.add_argument_group("tuning (beyond 3dAutobox)")
    tune.add_argument(
        "-clfrac",
        type=float,
        default=DEFAULT_CLFRAC,
        help=f"Clip-level fraction for the data/background split (default: {DEFAULT_CLFRAC})",
    )
    tune.add_argument(
        "-peelcount",
        type=int,
        default=DEFAULT_PEELCOUNT,
        help=f"Peel/re-dilate layers when clustering (default: {DEFAULT_PEELCOUNT})",
    )
    parser.add_argument("-device", default=None, help="PyTorch device (cuda, mps, cpu)")
    add_verbose_arg(parser, default=1)
    parser.add_argument("dataset", nargs="?", help=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    if args.input is None:
        args.input = args.dataset
    if args.input is None:
        parser.error("no input dataset (use -input DSET, or pass it as the last argument)")
    return args


def _ijkord_rows(orient: str, bounds: tuple[int, ...]) -> list[tuple[str, int, int]]:
    """Reorder the ijk extents so row 0/1/2 is the x/y/z axis, as ``-extent_ijkord``.

    Resampling can move which index runs along which anatomical axis, so a script
    that hardcodes "k is the slice index" breaks silently. This table names the
    index letter alongside its range instead.
    """
    axis_of = {"L": 0, "R": 0, "A": 1, "P": 1, "I": 2, "S": 2}
    rows: list[tuple[str, int, int] | None] = [None, None, None]
    for index_axis, letter in enumerate(orient):
        rows[axis_of[letter]] = (
            "ijk"[index_axis],
            bounds[2 * index_axis],
            bounds[2 * index_axis + 1],
        )
    return [r for r in rows if r is not None]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device, _, _ = parse_device_arg(args.device)
    configure_torch_backends(device, tf32=REGISTRATION_TF32)
    verb = args.verb

    t0 = time.time()
    with spinner(f"Loading {Path(args.input).name}"):
        vol, header = load_image(args.input, device=device)
    affine = header["affine"]
    orient = afni_orient_code(affine)

    bounds = autobox_bounds(
        vol,
        clust=not args.noclust,
        clip=not args.noclust,
        clfrac=args.clfrac,
        peelcount=args.peelcount,
    )
    if args.npad:
        bounds = pad_bounds(
            bounds, args.npad, tuple(vol.shape[-3:]) if args.npad_safety_on else None
        )

    if verb >= 1:
        print(
            f"++ Auto bbox: x={bounds[0]}..{bounds[1]}  "
            f"y={bounds[2]}..{bounds[3]}  z={bounds[4]}..{bounds[5]}"
        )

    # Field widths below reproduce 3dAutobox's output byte for byte, so scripts
    # that parse it with awk/sed keep working across the swap.
    ijk_text = " ".join(f"{b:8d}" for b in bounds)
    if args.extent_ijk:
        print(ijk_text)
    if args.extent_ijk_to_file:
        Path(args.extent_ijk_to_file).write_text(ijk_text + "\n")
    if args.extent_ijk_midslice:
        mids = (
            (bounds[0] + bounds[1]) // 2,
            (bounds[2] + bounds[3]) // 2,
            (bounds[4] + bounds[5]) // 2,
        )
        print(" ".join(f"{m:8d}" for m in mids))

    ijkord_text = "".join(f"{c} {lo:8d} {hi:8d}\n" for c, lo, hi in _ijkord_rows(orient, bounds))
    if args.extent_ijkord:
        sys.stdout.write(ijkord_text)
    if args.extent_ijkord_to_file:
        Path(args.extent_ijkord_to_file).write_text(ijkord_text)

    # The xyz extents describe the *cropped* grid, so they follow -npad.
    cropped, new_affine = crop_to_bounds(vol, affine, bounds)
    r, l, a, p, i, s = grid_extent_rai(tuple(cropped.shape[-3:]), new_affine)

    if args.extent:
        print(f"Extent auto bbox: R={r:f} L={l:f}  A={a:f} P={p:f}  I={i:f} S={s:f}")
    if args.extent_xyz_quiet:
        print(f"{r:f} {l:f}  {a:f} {p:f}  {i:f} {s:f}")
    if args.extent_xyz_to_file:
        Path(args.extent_xyz_to_file).write_text(f"{r:f}  {l:f}  {a:f}  {p:f}  {i:f}  {s:f}\n")
    if args.extent_xyz_midslice:
        print(f"{(r + l) / 2.0:10.5f} {(a + p) / 2.0:10.5f} {(i + s) / 2.0:10.5f}")

    if args.prefix:
        with spinner(f"Writing {Path(args.prefix).name}"):
            save_image(cropped, args.prefix, header_info=header, affine=new_affine)
        if verb >= 1:
            old = "x".join(str(d) for d in reversed(tuple(vol.shape[-3:])))
            new = "x".join(str(d) for d in reversed(tuple(cropped.shape[-3:])))
            frac = cropped[..., :, :, :].numel() / max(1, vol.numel())
            print(f"++ output dataset = {args.prefix}  ({old} -> {new}, {frac:.1%} of the voxels)")

    if verb >= 1:
        print(f"++ done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
