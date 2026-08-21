"""CLI for reorienting and/or resampling a dataset (AFNI 3dresample).

Command: ffs_util_resample (registered as entry point in pyproject.toml)

Changes a dataset's grid — axis order, voxel size, or both — without moving it in
space. Reorienting alone is a pure relabelling and stays bit-exact; changing the
voxel size interpolates.

Usage:
    ffs_util_resample -orient asl -rmode NN -prefix asl.nii.gz -input in.nii.gz
    ffs_util_resample -dxyz 1.0 1.0 0.9 -prefix 119.nii.gz -input in.nii.gz
    ffs_util_resample -master mast.nii.gz -prefix new.nii.gz -input old.nii.gz

Use ``-master`` rather than ``-dxyz`` when several datasets must end up on the
*same* grid: -dxyz fixes the voxel size but leaves the field of view free to
differ between inputs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_utils import add_verbose_arg, parse_device_arg, spinner
from fastfuncstuff.io.afni import load_nifti
from fastfuncstuff.processing.grid import (
    BOUND_TYPES,
    RMODES,
    afni_orient_code,
    reorient_grid,
    resample_grid,
    resample_to_grid,
    validate_orient,
)
from fastfuncstuff.processing.io import load_image, save_image
from fastfuncstuff.utils import REGISTRATION_TF32, configure_torch_backends


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ffs_util_resample",
        description="Reorient and/or resample a dataset onto a new grid "
        "(GPU, matching AFNI 3dresample).",
    )
    parser.add_argument("-input", "-inset", dest="input", required=True, help="Input dataset")
    parser.add_argument("-prefix", required=True, help="Output dataset")
    parser.add_argument(
        "-orient",
        default=None,
        metavar="OR_CODE",
        help="Reorient to this 3-letter axis order, e.g. 'asl' or 'lpi'. "
        "Characters come from {A,P}, {I,S}, {L,R}. Ignored when -master is given.",
    )
    parser.add_argument(
        "-dxyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Resample to these voxel sizes, in the order of the axes as they "
        "stand after any reorientation. Overrides the master's voxel size.",
    )
    parser.add_argument(
        "-master",
        default=None,
        metavar="MAST_DSET",
        help="Take the output grid (orientation, voxel size, field of view) from "
        "this dataset. The way to guarantee several outputs share one grid.",
    )
    parser.add_argument(
        "-rmode",
        default="wsinc5",
        metavar="RESAM",
        help="Interpolation: NN, Li, Cu, Bk (3dresample's set) or quintic, "
        "heptic, wsinc5 (sharper, ours). Default wsinc5; pass NN for atlas/label "
        "data. Ignored when the grids differ only by axis order or whole voxels — "
        "that path is exact.",
    )
    parser.add_argument(
        "-bound_type",
        "-bound-type",
        default="FOV",
        metavar="TYPE",
        help="What -dxyz preserves: FOV (default), SLAB (outer voxel centres), "
        "CENT / CENT_ORIG (voxel centres, so integer factors land on old centres). "
        "Default becomes CENT for -upsample/-downsample/-delta_scale.",
    )

    scale = parser.add_argument_group("voxel-size shorthands (imply -bound_type CENT)")
    scale.add_argument(
        "-upsample", type=float, default=None, metavar="FAC", help="Shrink voxels by FAC (>= 1)"
    )
    scale.add_argument(
        "-downsample", type=float, default=None, metavar="FAC", help="Grow voxels by FAC (>= 1)"
    )
    scale.add_argument(
        "-delta_scale",
        "-delta-scale",
        type=float,
        default=None,
        metavar="FAC",
        help="Scale voxel sizes by FAC (> 0)",
    )

    parser.add_argument("-device", default=None, help="PyTorch device (cuda, mps, cpu)")
    add_verbose_arg(parser, default=1)
    parser.add_argument("-debug", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _resolve_dxyz(args: argparse.Namespace, src_vox: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Turn -dxyz / -upsample / -downsample / -delta_scale into one dxyz triple.

    The scale shorthands multiply the *input's own* voxel sizes and default to
    ``-bound_type CENT``, so an integer factor keeps the old voxel centres.
    """
    bound = args.bound_type
    scales = [
        (args.upsample, "upsample"),
        (args.downsample, "downsample"),
        (args.delta_scale, "delta_scale"),
    ]
    given = [(v, name) for v, name in scales if v is not None]
    if len(given) > 1:
        raise SystemExit("** only one of -upsample / -downsample / -delta_scale may be used")
    if given:
        if args.dxyz is not None:
            raise SystemExit("** cannot use -dxyz with -upsample/-downsample/-delta_scale")
        value, name = given[0]
        if name in ("upsample", "downsample") and value < 1.0:
            raise SystemExit(f"** {name} factor must be >= 1.0")
        if name == "delta_scale" and value <= 0.0:
            raise SystemExit("** delta_scale factor must be > 0.0")
        factor = 1.0 / value if name == "upsample" else value
        # Argparse can't see whether -bound_type was typed or defaulted; CENT is
        # the documented default for these, and an explicit flag still wins.
        if args.bound_type == "FOV":
            bound = "CENT"
        return src_vox * factor, bound

    if args.dxyz is None:
        return None, bound
    return np.asarray(args.dxyz, dtype=np.float64), bound


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device, _, _ = parse_device_arg(args.device)
    configure_torch_backends(device, tf32=REGISTRATION_TF32)
    verb = args.debug if args.debug is not None else args.verb

    if args.bound_type.upper() not in BOUND_TYPES:
        raise SystemExit(
            f"** illegal bound_type '{args.bound_type}'; choose from {sorted(BOUND_TYPES)}"
        )
    interp = RMODES.get(args.rmode.lower())
    if interp is None:
        raise SystemExit(f"** invalid resample mode <{args.rmode}>")

    t0 = time.time()
    with spinner(f"Loading {Path(args.input).name}"):
        vol, header = load_image(args.input, device=device)
    src_affine = header["affine"]
    src_shape = tuple(vol.shape[-3:])
    src_vox = np.linalg.norm(src_affine[:3, :3], axis=0)

    dxyz, bound_type = _resolve_dxyz(args, src_vox)

    if args.master is not None:
        # Only the master's geometry is needed, so read the header and skip the
        # voxels — a master is often a full 4D run.
        with spinner(f"Reading master {Path(args.master).name}"):
            master = load_nifti(args.master)
        mshape = master.shape[:3]
        out_shape = (int(mshape[2]), int(mshape[1]), int(mshape[0]))
        out_affine = np.asarray(master.affine, dtype=np.float64)
        if args.orient is not None and verb >= 1:
            print("++ -orient ignored: the orientation comes from -master")
        if dxyz is not None:
            # AFNI starts from the master's grid and re-derives it at the new
            # voxel size, so -dxyz overrides only the resolution, not the FOV.
            out_shape, out_affine = resample_grid(out_shape, out_affine, dxyz, bound_type)
    else:
        out_shape, out_affine = src_shape, src_affine
        if args.orient is not None:
            out_shape, out_affine = reorient_grid(
                out_shape, out_affine, validate_orient(args.orient)
            )
        if dxyz is not None:
            out_shape, out_affine = resample_grid(out_shape, out_affine, dxyz, bound_type)
        if args.orient is None and dxyz is None:
            raise SystemExit("** nothing to do: give -orient, -dxyz, or -master")

    if verb >= 1:
        src_dims = "x".join(str(d) for d in reversed(src_shape))
        out_dims = "x".join(str(d) for d in reversed(out_shape))
        out_vox = np.linalg.norm(out_affine[:3, :3], axis=0)
        print(
            f"++ {afni_orient_code(src_affine)} {src_dims} "
            f"({', '.join(f'{v:g}' for v in src_vox)} mm)  ->  "
            f"{afni_orient_code(out_affine)} {out_dims} "
            f"({', '.join(f'{v:g}' for v in out_vox)} mm)"
        )

    out = resample_to_grid(vol, src_affine, out_shape, out_affine, interp=interp, verbose=verb)

    with spinner(f"Writing {Path(args.prefix).name}"):
        save_image(out, args.prefix, header_info=header, affine=out_affine)
    if verb >= 1:
        print(f"++ output dataset = {args.prefix}")
        print(f"++ done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
