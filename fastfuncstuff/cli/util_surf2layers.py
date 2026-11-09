"""CLI: FreeSurfer white/pial -> LayNii-compatible depth volumes, at any resolution.

Command: ffs_util_surf2layers

Usage:
    ffs_util_surf2layers -fs_subj $SUBJECTS_DIR/sub01 -dxyz 0.2 -prefix layers/sub01
    ffs_util_surf2layers -fs_subj sub01 -master epi_al.nii.gz -dxyz 0.25 -nr_layers 5 -prefix p
    ffs_util_surf2layers -white_mesh lh.white rh.white -pial_mesh lh.pial.ffsedit rh.pial.ffsedit \
        -master T1.nii.gz -dxyz 0.3 -prefix p
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser
from fastfuncstuff.cli_utils import add_device_arg, parse_prefix, setup_device
from fastfuncstuff.surface.ribbon_args import add_ribbon_args, add_ribbon_frac_arg, ribbon_frac

_DESCRIPTION = """\
LN2_LAYERS's outputs computed directly from a FreeSurfer subject's white and pial
surfaces, on any grid -- no rim file, no MapIcosahedron/3dSurf2Vol, no flood fill.

GM is the solid between the two closed meshes (a winding-number fill: hole-free at
any resolution); depth uses exact point-to-mesh distances; equivolume uses the
volume quantile of each voxel within its cortical column -- volume preserved by
construction, as LN2_LAYERS aims for, with no reliance on white/pial vertex pairing. The medial wall (label/?h.cortex.label) is excluded.

Surfaces come from -fs_subj (surf/?h.<-white>, surf/?h.<-pial>), or straight from
mesh files with -white_mesh/-pial_mesh -- any FreeSurfer-format surfaces, e.g.
edited copies saved beside the originals, so layers come from the edits without
installing them over the subject's own. Each file is placed in scanner space by
its own volume geometry. With -fs_subj as well, its cortex label and default
master still apply; without it, give -master, and -cortex_label to drop the
medial wall.

The grid is -master (default: the subject's mri/rawavg.mgz, i.e. the original
anatomical) at voxel size -dxyz (default: the master's own), cropped to the cortex
unless -no_autobox. A functional image aligned to the anatomical, upsampled with
-dxyz, gives layers on exactly the part of the brain it covers.

With -roi, only the cortex a region covers is layered, cropped to that patch and
written per region. A volume region (a thresholded statistic, say) is turned into
whole cortical columns, so a patchy blob becomes a depth-spanning ROI; cortex
around it is computed to get the equivolume depths right and then dropped.

Outputs (prefix P, or P_<region> with -roi; -roi also writes P_<region>_roi,
the region as a mask), named as LN2_LAYERS names them:
  P_rim              1 = CSF border, 2 = WM border, 3 = GM  (an LN2_* -rim input)
  P_metric_equidist  P_layers_equidist  P_midGM_equidist
  P_metric_equivol   P_layers_equivol   P_midGM_equivol
  P_thickness        d_white + d_pial, mm
  P_gm_frac          each GM voxel's share inside the ribbon (-ribbon_frac, on by default)
Metric is 0 at WM, 1 at CSF; layer 1 is deepest. A voxel joins the ribbon when at
least -ribbon_frac (0.1) of it lies in GM, so the ribbon's edge voxels are not lost;
its depth is measured at the GM part of the voxel.

GM thicker than -thick_warn (default 6 mm) is reported with the scanner-RAS
centroids of its clusters -- almost always pial on dura or a sinus, worth a look.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = FfsArgumentParser(prog="ffs_util_surf2layers", description=_DESCRIPTION)
    req = p.add_argument_group("required")
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
    add_ribbon_args(s)
    add_ribbon_frac_arg(
        s,
        "Every output then covers these voxels (rim, metric, layers, midGM, thickness), "
        "and P_gm_frac holds each one's share. Measured whole-brain at 0.3 mm: 82 s and "
        "4.7 GB against 12 s and 2.6 GB. ",
    )
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
        "-column_rounds",
        type=int,
        default=None,
        metavar="N",
        help="Equivolume: rounds of 1-ring pooling, instead of the number -column_voxels "
        "asks for at this resolution.  Pass a whole-hemisphere run's reported rounds to "
        "reproduce its depths exactly inside a -roi",
    )
    s.add_argument(
        "-thick_warn", type=float, default=6.0, metavar="MM", help="Warn above this thickness (6)"
    )

    r = p.add_argument_group(
        "regions",
        "Layers for part of the cortex only, each region on its own cropped grid.  "
        "Everything is still measured from the whole surfaces, so the depths are the "
        "ones a whole-hemisphere run gives -- only the grid is small.",
    )
    r.add_argument(
        "-roi",
        "-rois",
        dest="roi",
        nargs="+",
        default=None,
        metavar="ROI",
        help="Region(s) to layer, as surface or volume files, in any mix.  Surface: a "
        "FreeSurfer .label (an ROI drawn in ffs_viewer), .annot (one region per name) or "
        "a .label.gii.  Volume: a 3-D integer label map (one region per value), a 3-D "
        "binary mask (one region, named after the file) or a 4-D stack (one per volume), "
        "on any grid in register with the surfaces -- no resampling needed.  A volume "
        "region claims the cortical columns it covers at any depth (-roi_claim), so a "
        "thresholded statistic that only reaches the pial half still gives whole-depth "
        "columns.  Outputs are PREFIX_<region>_<tag>, plus PREFIX_<region>_roi, the "
        "region's own voxels as a mask",
    )
    r.add_argument(
        "-roi_fill",
        type=float,
        default=2.0,
        metavar="MM",
        help="Close holes and notches narrower than twice this far along the surface (2 "
        "mm), so the specks a threshold leaves inside a region become part of it.  0 "
        "takes the region exactly as given",
    )
    r.add_argument(
        "-roi_surround",
        type=float,
        default=None,
        metavar="MM",
        help="Cortex computed around each region along the surface, and then dropped.  "
        "Equivolume pools each column's volume with its neighbours', so the columns at a "
        "region's edge need cortex beyond it or they pool against nothing.  The default "
        "is worked out per region from the voxel size and the meshes, which is what it "
        "depends on -- a 0.8 mm grid pools several times further than a 0.2 mm one.  "
        "Either way the run reports how far it pooled and says so if the surround was "
        "too small.  Costs grid, not accuracy: more surround is a bigger box to measure "
        "and throw away",
    )
    r.add_argument(
        "-roi_hemi",
        default="auto",
        choices=("auto", "split", "major", "both"),
        help="A region covering BOTH hemispheres (a bilateral V1, say) would otherwise "
        "be layered on one grid spanning both, which is mostly air between the banks.  "
        "'auto' (default) writes one region per hemisphere when the smaller side holds "
        "at least a third of the columns, and otherwise keeps the larger side and warns "
        "-- a few columns over the midline is threshold spill or a millimetre of "
        "misalignment, not a region.  'split' always writes both, 'major' always keeps "
        "the larger, 'both' leaves the region on one grid.  Every region's name carries "
        "its hemisphere (NAME_lh), so -hemi lh and -hemi rh runs do not collide",
    )
    r.add_argument(
        "-roi_claim",
        default="any",
        choices=("any", "mid"),
        help="Which columns a VOLUME region claims: 'any' = the region covers the ribbon "
        "anywhere along the column's depth (default -- what a thresholded map needs), "
        "'mid' = at its midthickness point only (stricter, and what the ROI depth "
        "sampling in ffs_reml/ffs_nwarp uses)",
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
    from fastfuncstuff.surface.volume_depth import (
        export_depth_volumes,
        resolve_ribbon_surfaces,
        thick_report,
    )

    args = parse_args(argv)
    device = setup_device(args.device)
    verbose = not args.quiet
    subj = Path(args.fs_subj) if args.fs_subj is not None else None
    if args.nr_layers < 1:
        raise SystemExit("-nr_layers must be at least 1")
    if args.roi_fill < 0 or (args.roi_surround is not None and args.roi_surround < 0):
        raise SystemExit("-roi_fill and -roi_surround are distances along the surface, mm >= 0")
    if args.column_rounds is not None and args.column_rounds < 0:
        raise SystemExit("-column_rounds must be 0 or more")
    try:
        surfaces = resolve_ribbon_surfaces(
            args.fs_subj, args.white, args.pial, args.white_mesh, args.pial_mesh,
            args.cortex_label, args.hemi, log=print if verbose else None,
        )  # fmt: skip
    except (OSError, ValueError) as err:
        raise SystemExit(f"ffs_util_surf2layers: {err}") from err
    master = args.master
    if master is None:
        if subj is None:
            raise SystemExit("-master is required without -fs_subj")
        for name in ("rawavg.mgz", "orig.mgz"):
            if (subj / "mri" / name).exists():
                master = subj / "mri" / name
                break
        else:
            raise SystemExit(f"no -master given and {subj}/mri has no rawavg.mgz or orig.mgz")
    pinfo = parse_prefix(args.prefix)
    stem, ext = pinfo.stem, pinfo.nifti_ext
    t0 = time.time()

    if args.dxyz is not None and len(args.dxyz) not in (1, 3):
        raise SystemExit("-dxyz takes 1 or 3 values")
    if verbose:
        print(f"ffs_util_surf2layers: master {master}")
    affine, shape = master_grid(master)
    if args.roi:
        return _roi_main(args, surfaces, affine, shape, stem, ext, device, verbose, t0)
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
            column_rounds=args.column_rounds,
            thick_limit=args.thick_warn,
            device=device,
            verbose=verbose,
            min_gm_fraction=ribbon_frac(args),
        )
    except ValueError as err:
        raise SystemExit(f"ffs_util_surf2layers: {err}") from err
    if verbose:
        print(
            f"  GM voxels {out.n_gm:,}  medial wall dropped {out.n_medial:,}"
            + (f"  hemisphere overlap {out.n_overlap:,}" if out.n_overlap else "")
            + f"  equivolume pooling {'/'.join(str(r) for _, r in out.smooth_rounds)} rounds"
        )
    for line in thick_report(out, affine):
        print(line)
    if verbose:
        print(f"ffs_util_surf2layers done in {time.time() - t0:.0f}s -> {stem}_*{ext}")
    return 0


def _roi_main(args, surfaces, affine, shape, stem, ext, device, verbose, t0) -> int:
    """``-roi``: one region at a time, each on its own crop."""
    from fastfuncstuff.processing.roi_depth import CLAIM_ANY, CLAIM_MID, read_depth_rois
    from fastfuncstuff.surface.volume_depth import (
        export_roi_depth_volumes,
        roi_report,
        thick_report,
    )

    notes: list[str] = []
    try:
        rois = read_depth_rois(
            args.roi, surfaces, notes, CLAIM_MID if args.roi_claim == "mid" else CLAIM_ANY
        )
        if verbose:
            for note in notes:
                print(f"  {note}")
        results = export_roi_depth_volumes(
            surfaces,
            affine,
            shape,
            stem,
            ext,
            rois,
            dxyz=args.dxyz,
            autobox=not args.no_autobox,
            pad_mm=args.autobox_pad,
            fill_mm=args.roi_fill,
            surround_mm=args.roi_surround,
            hemi_mode=args.roi_hemi,
            n_layers=args.nr_layers,
            column_voxels=args.column_voxels,
            column_rounds=args.column_rounds,
            thick_limit=args.thick_warn,
            device=device,
            verbose=verbose,
            min_gm_fraction=ribbon_frac(args),
        )
    except (OSError, ValueError) as err:
        raise SystemExit(f"ffs_util_surf2layers: {err}") from err
    for res in results:
        if verbose:
            for line in roi_report(res):
                print(line)
        elif not res.enough_surround:
            print(roi_report(res)[-1])
        for line in thick_report(res.out, res.affine):
            print(line)
    if verbose:
        print(
            f"ffs_util_surf2layers done in {time.time() - t0:.0f}s -> "
            f"{len(results)} region(s), {stem}_<region>_*{ext}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
