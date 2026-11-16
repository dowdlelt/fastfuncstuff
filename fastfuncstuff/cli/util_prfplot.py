"""CLI: retinotopic maps from an ffs_pyrf bucket, on an equal-area view of the sphere.

Command: ffs_util_prfplot

Usage:
    ffs_util_prfplot -input prf.lh.func.gii -fs_subj /fs/sub-01 -prefix prf_maps.png
    ffs_util_prfplot -input prf.lh.func.gii -fs_subj /fs/sub-01 -center medial \
        -radius 60 -maps angle eccentricity rfsize -units deg -prefix medial.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser

_DESCRIPTION = """\
Draws polar angle, eccentricity, R2 (or any sub-brick) of an ffs_pyrf surface
bucket on the subject's sphere, flattened by an equal-area (Lambert) projection:
nothing is cut, neighbours stay neighbours, and area is not distorted. One row per
map, one column per hemisphere; the .lh / .rh partner of -input is found beside it.

The view looks at the sphere from outside, dorsal up, so every subject and both
hemispheres come out the same way round. Where it looks:
  -center auto        (default) the R2-weighted centre of the vertices that pass
                      -r2_min -- the responsive cortex, found without labels
  -center posterior | anterior | dorsal | ventral | lateral | medial
  -center rois        the centre of the -rois labels
  -center_xyz X Y Z   any direction on the sphere (FreeSurfer: x right, y anterior,
                      z superior)
-radius is how far the view reaches, in degrees of arc on the sphere (the zoom).
-align rois turns the view so the labelled region's long axis is horizontal (with
V1 alone in -rois, the calcarine runs left-right); -rotate then turns it further.

Visual-area labels are optional: -rois draws their borders (FreeSurfer .mgz,
.annot, .label, or GIfTI), one file per hemisphere.

Polar angle uses a colour wheel (0 = right horizontal meridian, 90 = up), coloured
flat per face, because smooth shading would sweep the whole wheel between 350 and
10 degrees. Spatial values are in the bucket's units: degrees when ffs_pyrf had
-screen_extent, aperture pixels otherwise, as the bucket records.
"""

_MAP_STYLE = {
    # name: (colormap, log scale, cyclic period)
    "angle": ("hsv", False, 360.0),
    "eccentricity": ("viridis", False, None),
    "r2": ("magma", False, None),
    "rfsize": ("plasma", True, None),
    "sigma": ("plasma", True, None),
    "exponent": ("cividis", True, None),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = FfsArgumentParser(prog="ffs_util_prfplot", description=_DESCRIPTION)
    p.add_argument("-input", nargs="+", required=True, metavar="PRF.gii",
                   help="ffs_pyrf surface bucket(s). One hemisphere's file is enough when its "
                   ".lh/.rh partner sits beside it.")  # fmt: skip
    p.add_argument("-prefix", required=True, metavar="OUT.png",
                   help="Figure to write; the extension picks the format (.png, .pdf, .svg).")  # fmt: skip
    surf = p.add_argument_group("surfaces")
    surf.add_argument("-fs_subj", metavar="DIR", default=None,
                      help="FreeSurfer subject: surf/?h.sphere and surf/?h.sulc are used "
                      "unless -sphere / -underlay name others.")  # fmt: skip
    surf.add_argument("-sphere", nargs="+", metavar="SPHERE", default=None,
                      help="Sphere per hemisphere, in -input's hemisphere order (FreeSurfer "
                      "or .surf.gii), on the bucket's mesh.")  # fmt: skip
    surf.add_argument("-underlay", nargs="+", metavar="MORPH", default=None,
                      help="Sulcal depth or curvature per hemisphere (FreeSurfer morph or "
                      ".shape.gii), drawn grey underneath. 'none' to omit.")  # fmt: skip
    surf.add_argument("-rois", nargs="+", metavar="LABELS", default=None,
                      help="Area labels per hemisphere (.mgz, .annot, .label, .gii); their "
                      "borders are drawn.")  # fmt: skip
    maps = p.add_argument_group("maps")
    maps.add_argument("-maps", nargs="+", default=["angle", "eccentricity", "r2"], metavar="MAP",
                      help="Sub-bricks to draw, one row each: angle, eccentricity, r2, rfsize, "
                      "sigma, exponent, or any label in the bucket.")  # fmt: skip
    maps.add_argument("-r2_min", type=float, default=10.0, metavar="PCT",
                      help="Colour only vertices with R2 at or above this (percent).")  # fmt: skip
    maps.add_argument("-ecc_max", type=float, default=None, metavar="E",
                      help="Top of the eccentricity scale. Default: the screen's edge, which "
                      "ffs_pyrf records in the bucket (older buckets: the 95th percentile of "
                      "the best-fit quarter).")  # fmt: skip
    maps.add_argument("-units", default="", metavar="TEXT",
                      help="Label for spatial values. Default: what the bucket records (deg "
                      "or px).")  # fmt: skip
    view = p.add_argument_group("view")
    view.add_argument("-center", default="auto",
                      choices=["auto", "rois", "posterior", "anterior", "dorsal", "ventral",
                               "lateral", "medial"],
                      help="Where the view looks (see above).")  # fmt: skip
    view.add_argument("-center_xyz", nargs=3, type=float, metavar=("X", "Y", "Z"), default=None,
                      help="Look along this direction instead (overrides -center).")  # fmt: skip
    view.add_argument("-radius", type=float, default=42.0, metavar="DEG",
                      help="How far the view reaches from its centre, in degrees of arc.")  # fmt: skip
    view.add_argument("-align", choices=["none", "rois"], default="none",
                      help="rois: turn the view so the labelled region's long axis is "
                      "horizontal.")  # fmt: skip
    view.add_argument("-roi_subset", nargs="+", type=int, default=None, metavar="LABEL",
                      help="Label values -center rois and -align rois use (e.g. 1 for V1 in a "
                      "V1-V4 file). Default: every nonzero label.")  # fmt: skip
    view.add_argument("-rotate", type=float, default=0.0, metavar="DEG",
                      help="Turn the view counter-clockwise by this much.")  # fmt: skip
    out = p.add_argument_group("figure")
    out.add_argument("-panel_size", type=float, default=3.2, metavar="IN",
                     help="Width of one panel, inches.")  # fmt: skip
    out.add_argument("-dpi", type=int, default=150, help="Resolution of raster output.")
    out.add_argument("-title", default=None, help="Figure title. Default: the input's name.")
    return p.parse_args(argv)


def _load_surface(path: str) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    from fastfuncstuff.io.gifti import is_gifti, load_gifti_surface

    if is_gifti(path):
        vertices, faces, _ = load_gifti_surface(path)
        return vertices, faces
    vertices, faces = nib.freesurfer.read_geometry(path)
    return np.asarray(vertices, np.float64), np.asarray(faces, np.int64)


def _load_per_vertex(path: str, n_vertices: int) -> np.ndarray:
    """A morph or label file as one value per vertex."""
    import nibabel as nib

    from fastfuncstuff.io.gifti import is_gifti, load_gifti_data

    lower = path.lower()
    if is_gifti(path):
        values = np.asarray(load_gifti_data(path)[0])
        values = values[:, 0] if values.ndim == 2 else values
    elif lower.endswith((".mgz", ".mgh", ".nii", ".nii.gz")):
        values = np.asarray(nib.load(path).dataobj).reshape(-1)  # ty: ignore[unresolved-attribute]
    elif lower.endswith(".annot"):
        values = nib.freesurfer.read_annot(path)[0]
        values = np.where(values < 0, 0, values + 1)  # unlabelled -> 0
    elif lower.endswith(".label"):
        values = np.zeros(n_vertices)
        values[nib.freesurfer.read_label(path)] = 1
    else:
        values = nib.freesurfer.read_morph_data(path)
    values = np.asarray(values)
    if values.shape[0] != n_vertices:
        raise ValueError(f"{path} has {values.shape[0]} values, the mesh {n_vertices} vertices")
    return values


def _load_bucket(path: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.io.gifti import surface_meta

    image = load_nifti(path)
    meta = surface_meta(image.header) or {}
    labels = meta.get("brick_labels", "").split("\t")
    data = np.asarray(image.dataobj, np.float64).reshape(image.shape[0], -1)
    if len(labels) != data.shape[1] or not all(labels):
        raise ValueError(f"{path} has no sub-brick labels; is it an ffs_pyrf bucket?")
    return dict(zip(labels, data.T, strict=True)), meta


def _pair_inputs(inputs: list[str]) -> list[tuple[str, str]]:
    """``[(hemi, path), ...]`` in lh, rh order."""
    from fastfuncstuff.processing.ribbon_sample import group_hemisphere_files, hemisphere_of

    if len(inputs) == 1:
        try:
            grouped = group_hemisphere_files(inputs)
        except FileNotFoundError:
            grouped = None
        if grouped is not None:
            return list(zip(grouped[1], grouped[0][0], strict=True))
    pairs = []
    for path in inputs:
        found = hemisphere_of(path)
        if found is None:
            raise ValueError(
                f"cannot tell which hemisphere {path} is: name it ...lh... or ...rh..."
            )
        pairs.append((found[0], path))
    return sorted(pairs, key=lambda pair: pair[0])


def _per_hemi(option: list[str] | None, name: str, hemis: list[str]) -> list[str | None]:
    if option is None:
        return [None] * len(hemis)
    if len(option) != len(hemis):
        raise ValueError(f"-{name} needs one file per hemisphere ({len(hemis)}), got {len(option)}")
    return list(option)


def _style(name: str, values: np.ndarray, shown: np.ndarray, r2: np.ndarray, args) -> tuple:
    """``(cmap, norm, period, colorbar label)`` for one map."""
    from matplotlib.colors import LogNorm, Normalize

    cmap, log, period = _MAP_STYLE.get(name, ("viridis", False, None))
    units = f" ({args.units})" if args.units else ""
    finite = values[shown & np.isfinite(values)]
    if name == "angle":
        return cmap, Normalize(0, 360), period, "polar angle (deg)"
    if name == "r2":
        top = float(np.nanpercentile(finite, 99)) if finite.size else 100.0
        return cmap, Normalize(args.r2_min, max(top, args.r2_min + 1)), None, "R² (%)"
    if name == "eccentricity":
        # From the best-fit quarter: off-screen fits (centres a screen width out,
        # weak R2) otherwise set the top and crush the on-screen gradient.
        top = args.ecc_max
        if top is None and finite.size:
            best = shown & np.isfinite(values) & (r2 >= np.nanpercentile(r2[shown], 75))
            top = float(np.nanpercentile(values[best], 95))
        return cmap, Normalize(0, top or 1.0), None, f"eccentricity{units}"
    positive = finite[finite > 0] if log else finite
    if positive.size == 0:
        return cmap, Normalize(0, 1), None, name
    low, high = np.nanpercentile(positive, [5, 95])
    high = high if high > low else low * 1.01 + 1e-12
    label = f"{name}{units}" if name in ("rfsize", "sigma", "x", "y") else name
    return cmap, (LogNorm(low, high) if log else Normalize(low, high)), None, label


def _angle_wheel(ax) -> None:
    """The visual field as a colour wheel: the legend for polar angle."""
    edges = np.linspace(0, 2 * np.pi, 361)
    t, r = np.meshgrid(edges, [0.45, 1.0])
    centres = np.rad2deg(edges[:-1] + np.pi / 360)[None, :]
    ax.pcolormesh(r * np.cos(t), r * np.sin(t), centres, cmap="hsv", vmin=0, vmax=360,
                  shading="flat")  # fmt: skip
    for degrees, text in ((0, "R"), (90, "up"), (180, "L"), (270, "down")):
        angle = np.deg2rad(degrees)
        ax.text(1.3 * np.cos(angle), 1.3 * np.sin(angle), text, ha="center", va="center",
                fontsize=7)  # fmt: skip
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_axis_off()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fastfuncstuff.surface.flatmap import center_direction, draw_flat_map, sphere_view

    pairs = _pair_inputs(args.input)
    hemis = [hemi for hemi, _ in pairs]
    spheres = _per_hemi(args.sphere, "sphere", hemis)
    underlays = _per_hemi(None if args.underlay == ["none"] else args.underlay, "underlay", hemis)
    rois = _per_hemi(args.rois, "rois", hemis)
    if (args.center == "rois" or args.align == "rois") and args.rois is None:
        raise SystemExit("ffs_util_prfplot: -center rois and -align rois need -rois")

    panels = []
    panel_r2: list[np.ndarray] = []
    for (hemi, path), sphere, underlay, roi in zip(pairs, spheres, underlays, rois, strict=True):
        fs_surf = Path(args.fs_subj, "surf") if args.fs_subj else None
        sphere = sphere or (str(fs_surf / f"{hemi}.sphere") if fs_surf else None)
        if sphere is None:
            raise SystemExit("ffs_util_prfplot: give -fs_subj or -sphere")
        if underlay is None and fs_surf is not None and args.underlay != ["none"]:
            candidate = fs_surf / f"{hemi}.sulc"
            underlay = str(candidate) if candidate.is_file() else None
        vertices, faces = _load_surface(sphere)
        bucket, bucket_meta = _load_bucket(path)
        if not args.units:
            args.units = bucket_meta.get("prf_units", "")
        if args.ecc_max is None and bucket_meta.get("prf_stimulus_radius"):
            args.ecc_max = float(bucket_meta["prf_stimulus_radius"])
        n_vertices = len(vertices)
        if len(next(iter(bucket.values()))) != n_vertices:
            raise SystemExit(
                f"ffs_util_prfplot: {path} has {len(next(iter(bucket.values())))} vertices, "
                f"{sphere} {n_vertices}: not the same mesh"
            )
        missing = [name for name in args.maps if name not in bucket]
        if missing:
            raise SystemExit(f"ffs_util_prfplot: {path} has no {', '.join(missing)}; it has "
                             f"{', '.join(bucket)}")  # fmt: skip
        labels = None if roi is None else _load_per_vertex(roi, n_vertices).astype(int)
        chosen = None
        if labels is not None:
            chosen = labels > 0 if args.roi_subset is None else np.isin(labels, args.roi_subset)
            if (args.center == "rois" or args.align == "rois") and not chosen.any():
                raise SystemExit(f"ffs_util_prfplot: {roi} has none of labels {args.roi_subset}")
        r2 = 100.0 * bucket["r2"] if "r2" in bucket else np.full(n_vertices, 100.0)
        shown = np.isfinite(r2) & (r2 >= args.r2_min)
        if args.center_xyz is not None:
            center = np.asarray(args.center_xyz, float)
        elif args.center == "rois":
            assert chosen is not None
            unit = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
            center = unit[chosen].mean(axis=0)
        elif args.center == "auto":
            unit = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
            weights = np.where(shown, r2, 0.0)
            center = (
                (unit * weights[:, None]).sum(axis=0)
                if shown.sum() >= 10
                else center_direction("posterior")
            )
        else:
            center = center_direction(args.center, hemi)
        view = sphere_view(
            vertices,
            faces,
            center,
            radius_deg=args.radius,
            rotate_deg=args.rotate,
            align=None if args.align == "none" else chosen,
        )
        underlay_values = None if underlay is None else _load_per_vertex(underlay, n_vertices)
        panels.append((hemi, view, bucket, shown, underlay_values, labels))
        panel_r2.append(r2)

    n_rows, n_cols = len(args.maps), len(panels)
    size = args.panel_size
    # A narrow last column holds each row's colorbar or angle wheel, so every map
    # panel keeps the same size.
    fig = plt.figure(figsize=(size * (n_cols + 0.35), size * n_rows + 0.5))
    grid = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1.0] * n_cols + [0.35])
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(n_cols)] for r in range(n_rows)])
    for row, name in enumerate(args.maps):
        values_all = [
            100.0 * bucket[name] if name == "r2" else bucket[name]
            for _, _, bucket, _, _, _ in panels
        ]
        pooled = np.concatenate(values_all)
        pooled_shown = np.concatenate([shown for _, _, _, shown, _, _ in panels])
        pooled_r2 = np.concatenate([r2 for r2 in panel_r2])
        cmap, norm, period, label = _style(name, pooled, pooled_shown, pooled_r2, args)
        mappable = None
        for col, (hemi, view, _, shown, underlay_values, labels) in enumerate(panels):
            drawn = draw_flat_map(
                axes[row, col],
                view,
                values_all[col],
                cmap=cmap,
                norm=norm,
                mask=shown,
                underlay=underlay_values,
                borders=labels,
                period=period,
                title=f"{label} · {hemi}",
            )
            mappable = mappable or drawn
        side = fig.add_subplot(grid[row, n_cols])
        if name == "angle":
            _angle_wheel(side.inset_axes((0.0, 0.25, 1.0, 0.5)))
            side.set_axis_off()
        elif mappable is not None:
            fig.colorbar(mappable, cax=side.inset_axes((0.25, 0.15, 0.18, 0.7)), label=label)
            side.set_axis_off()
        else:
            side.set_axis_off()
    title = args.title if args.title is not None else os.path.basename(pairs[0][1])
    fig.suptitle(f"{title}   (R² ≥ {args.r2_min:g}%)", fontsize=10)
    fig.savefig(args.prefix, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
