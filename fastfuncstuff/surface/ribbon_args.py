"""The flags that name a cortical ribbon, and how ROI depth sampling cuts it.

Kept free of torch and the mesh code so a tool's parser (and ``-help``) stays fast;
:func:`surface.volume_depth.resolve_ribbon_surfaces` reads the ribbon flags and
:mod:`processing.roi_depth` the depth ones.
"""

from __future__ import annotations


def add_ribbon_args(group, *, hemi: bool = True) -> None:
    """Register ``-fs_subj -white -pial -white_mesh -pial_mesh -cortex_label`` (+ ``-hemi``)."""
    group.add_argument(
        "-fs_subj",
        default=None,
        help="FreeSurfer subject directory (or give -white_mesh and -pial_mesh)",
    )
    if hemi:
        group.add_argument("-hemi", default="both", choices=("both", "lh", "rh"))
    group.add_argument("-white", default="white", help="White surface name in surf/ (white)")
    group.add_argument("-pial", default="pial", help="Pial surface name in surf/ (pial)")
    group.add_argument(
        "-white_mesh",
        nargs="+",
        default=None,
        metavar="FILE",
        help="White surface file(s), one per hemisphere, instead of surf/?h.<-white>. "
        "Hemisphere read from the name (lh./rh.); else the order is lh, rh. One path "
        "without lh/rh that does not exist (surf/smoothwm) means lh./rh. of it",
    )
    group.add_argument(
        "-pial_mesh",
        nargs="+",
        default=None,
        metavar="FILE",
        help="Pial surface file(s) paired with -white_mesh by hemisphere",
    )
    group.add_argument(
        "-cortex_label",
        nargs="+",
        default=None,
        metavar="FILE",
        help="?h.cortex.label file(s) for -white_mesh/-pial_mesh (default with "
        "-fs_subj: its label/?h.cortex.label; otherwise the medial wall is kept)",
    )


def add_depth_args(group, *, interp: bool = True) -> None:
    """Ribbon flags plus ``-depths -equidistant/-equivolume -depth_dxyz -depth_smooth -depth_interp``.

    Giving surfaces is what switches depth sampling on: the ROI means are then read
    along the cortical ribbon, one per ROI per depth, instead of over ROI voxels.
    """
    add_ribbon_args(group, hemi=False)
    group.add_argument(
        "-depths",
        type=int,
        nargs="+",
        default=[1],
        metavar="N",
        help="With surfaces: N cortical depth bins per ROI, 1 = deepest (next to white "
        "matter) .. N = by pial.  1 averages the whole ribbon inside each ROI.  Several "
        "counts (-depths 7 9 11) sample every one from the same read of the data, for "
        "models that average over the number of depths: ffs_nwarp writes one file per "
        "count (PREFIX.K7.roi.*), ffs_reml / ffs_deconvolve one table with a K column",
    )
    how = group.add_mutually_exclusive_group()
    how.add_argument(
        "-equidistant",
        dest="depth_method",
        action="store_const",
        const="equidistant",
        help="Depth bins of equal thickness (default)",
    )
    how.add_argument(
        "-equivolume",
        dest="depth_method",
        action="store_const",
        const="equivolume",
        help="Depth bins of equal volume within each cortical column (Bok): thinner "
        "where the cortex bends away from them",
    )
    group.set_defaults(depth_method="equidistant")
    group.add_argument(
        "-depth_dxyz",
        type=float,
        default=None,
        metavar="MM",
        help="Read spacing along the surface, mm.  The ribbon is cut into cells this "
        "wide, and at most half this thick through the depth with at least 2 per depth "
        "bin.  Each cell counts by its exact volume.  Default: the data's voxel size",
    )
    group.add_argument(
        "-depth_smooth",
        type=int,
        default=1,
        metavar="N",
        help="With -equivolume: rounds of 1-ring mesh smoothing of each cortical column's "
        "volume profile.  0 keeps every vertex column on its own (exactly equal-volume bins "
        "per column, but noisier depths); more pools across neighbours, which cuts noise but "
        "blurs across abrupt folding (outer bin ~5%% light at 2 rounds on real cortex)",
    )
    group.add_argument(
        "-roi_claim",
        choices=("mid", "any"),
        default="mid",
        help="How a volume ROI claims a cortical column: mid = the ROI holds the column's "
        "midthickness point; any = it covers the column at any depth, so a blob in the "
        "outer third still claims it. Match ffs_hrftdv -atlas (default any) when fitting "
        "its pairs",
    )
    if not interp:  # the tool reads through its own transform chain, with its own -interp
        return
    group.add_argument(
        "-depth_interp",
        default="cubic",
        choices=("linear", "cubic", "quintic", "heptic", "wsinc5"),
        help="How each ribbon read interpolates the data grid.  The data were already "
        "resampled once; linear blurs across depth, wsinc5 is sharpest but reads +-5 "
        "voxels",
    )


def depth_requested(args) -> bool:
    return bool(getattr(args, "fs_subj", None) or getattr(args, "white_mesh", None))
