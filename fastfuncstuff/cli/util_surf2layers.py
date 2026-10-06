"""CLI: FreeSurfer white/pial -> LayNii-compatible depth volumes, at any resolution.

Command: ffs_util_surf2layers

Usage:
    ffs_util_surf2layers -fs_subj $SUBJECTS_DIR/sub01 -dxyz 0.2 -prefix layers/sub01
    ffs_util_surf2layers -fs_subj sub01 -master epi_al.nii.gz -dxyz 0.25 -nr_layers 5 -prefix p
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser
from fastfuncstuff.cli_utils import add_device_arg, parse_prefix, setup_device

_DESCRIPTION = """\
LN2_LAYERS's outputs computed directly from a FreeSurfer subject's white and pial
surfaces, on any grid -- no rim file, no MapIcosahedron/3dSurf2Vol, no flood fill.

GM is the solid between the two closed meshes (a winding-number fill: hole-free at
any resolution); depth uses exact point-to-mesh distances; equivolume uses the
volume quantile of each voxel within its cortical column -- volume preserved by
construction, as LN2_LAYERS aims for, with no reliance on white/pial vertex pairing. The medial wall (label/?h.cortex.label) is excluded.

The grid is -master (default: the subject's mri/rawavg.mgz, i.e. the original
anatomical) at voxel size -dxyz (default: the master's own), cropped to the cortex
unless -no_autobox. A functional image aligned to the anatomical, upsampled with
-dxyz, gives layers on exactly the part of the brain it covers.

Outputs (prefix P), named as LN2_LAYERS names them:
  P_rim              1 = CSF border, 2 = WM border, 3 = GM  (an LN2_* -rim input)
  P_metric_equidist  P_layers_equidist  P_midGM_equidist
  P_metric_equivol   P_layers_equivol   P_midGM_equivol
  P_thickness        d_white + d_pial, mm
Metric is 0 at WM, 1 at CSF; layer 1 is deepest.

GM thicker than -thick_warn (default 6 mm) is reported with the scanner-RAS
centroids of its clusters -- almost always pial on dura or a sinus, worth a look.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = FfsArgumentParser(prog="ffs_util_surf2layers", description=_DESCRIPTION)
    req = p.add_argument_group("required")
    req.add_argument("-fs_subj", required=True, help="FreeSurfer subject directory")
    req.add_argument("-prefix", required=True, help="Output prefix")

    g = p.add_argument_group("grid")
    g.add_argument(
        "-master",
        default=None,
        help="Image whose grid (orientation, field of view) to use; default mri/rawavg.mgz",
    )
    g.add_argument(
        "-dxyz",
        type=float,
        nargs="+",
        default=None,
        metavar="MM",
        help="Voxel size: 1 value (isotropic) or 3 (master's i j k axes), as 3dresample "
        "-dxyz. Default: the master's own",
    )
    g.add_argument("-no_autobox", action="store_true", help="Keep the full master field of view")
    g.add_argument(
        "-autobox_pad", type=float, default=1.0, metavar="MM", help="Margin around pial (1 mm)"
    )

    s = p.add_argument_group("surfaces and layers")
    s.add_argument("-hemi", default="both", choices=("both", "lh", "rh"))
    s.add_argument("-white", default="white", help="White surface name in surf/ (white)")
    s.add_argument("-pial", default="pial", help="Pial surface name in surf/ (pial)")
    s.add_argument("-nr_layers", type=int, default=3, help="Number of layers (3)")
    s.add_argument(
        "-column_voxels",
        type=float,
        default=64.0,
        metavar="N",
        help="Equivolume: voxels per cortical column whose volume is split into equal "
        "layers; columns are widened along the surface until they hold this many (64)",
    )
    s.add_argument(
        "-thick_warn", type=float, default=6.0, metavar="MM", help="Warn above this thickness (6)"
    )

    o = p.add_argument_group("execution")
    add_device_arg(o)
    o.add_argument("-quiet", action="store_true")
    return p.parse_args(sys.argv[1:] if argv is None else argv)


def master_grid(path: str | Path) -> tuple[np.ndarray, tuple[int, int, int]]:
    """``(affine, (X, Y, Z))`` of an image, from its header alone."""
    path = str(path)
    if path.endswith((".mgz", ".mgh")):
        import nibabel as nib

        img = nib.load(path)
        affine, shape = img.affine, img.shape  # ty: ignore[unresolved-attribute]
    else:
        from fastfuncstuff.io.headers import read_nifti_header

        hdr = read_nifti_header(path)
        affine, shape = hdr.get_best_affine(), hdr.get_data_shape()
    nx, ny, nz = (int(n) for n in shape[:3])
    return np.asarray(affine, np.float64), (nx, ny, nz)


def main(argv: list[str] | None = None) -> int:
    from fastfuncstuff.io.freesurfer import load_hemisphere
    from fastfuncstuff.surface.volume_depth import (
        RibbonSurfaces,
        export_depth_volumes,
        thick_report,
    )

    args = parse_args(argv)
    device = setup_device(args.device)
    verbose = not args.quiet
    subj = Path(args.fs_subj)
    if not (subj / "surf").is_dir():
        raise SystemExit(f"{subj} has no surf/ directory; is it a FreeSurfer subject?")
    if args.nr_layers < 1:
        raise SystemExit("-nr_layers must be at least 1")
    master = args.master
    if master is None:
        for name in ("rawavg.mgz", "orig.mgz"):
            if (subj / "mri" / name).exists():
                master = subj / "mri" / name
                break
        else:
            raise SystemExit(f"no -master given and {subj}/mri has no rawavg.mgz or orig.mgz")
    pinfo = parse_prefix(args.prefix)
    stem, ext = pinfo.stem, pinfo.nifti_ext
    t0 = time.time()

    hemis = ("lh", "rh") if args.hemi == "both" else (args.hemi,)
    surfaces = []
    for hemi in hemis:
        # load_hemisphere requires white and pial whichever pair is asked for.
        states = tuple(dict.fromkeys(("white", "pial", args.white, args.pial)))
        h = load_hemisphere(subj, hemi, states=states, morph=(), patches=False)
        if args.white not in h.states:
            raise SystemExit(f"{subj}/surf/{hemi}.{args.white} not found")
        if args.pial not in h.states:
            raise SystemExit(f"{subj}/surf/{hemi}.{args.pial} not found")
        if h.cortex is None and verbose:
            print(f"  {hemi}: no label/{hemi}.cortex.label -- medial wall kept")
        surfaces.append(RibbonSurfaces.from_hemisphere(h, args.white, args.pial))

    if args.dxyz is not None and len(args.dxyz) not in (1, 3):
        raise SystemExit("-dxyz takes 1 or 3 values")
    if verbose:
        print(f"ffs_util_surf2layers: master {master}")
    affine, shape = master_grid(master)
    try:
        out, affine, _ = export_depth_volumes(
            surfaces,
            affine,
            shape,
            stem,
            ext,
            dxyz=args.dxyz,
            autobox=not args.no_autobox,
            pad_mm=args.autobox_pad,
            n_layers=args.nr_layers,
            column_voxels=args.column_voxels,
            thick_limit=args.thick_warn,
            device=device,
            verbose=verbose,
        )
    except ValueError as err:
        raise SystemExit(f"ffs_util_surf2layers: {err}") from err
    if verbose:
        print(
            f"  GM voxels {out.n_gm:,}  medial wall dropped {out.n_medial:,}"
            + (f"  hemisphere overlap {out.n_overlap:,}" if out.n_overlap else "")
        )
    for line in thick_report(out, affine):
        print(line)
    if verbose:
        print(f"ffs_util_surf2layers done in {time.time() - t0:.0f}s -> {stem}_*{ext}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
