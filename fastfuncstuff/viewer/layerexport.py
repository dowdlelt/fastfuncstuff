"""Export LayNii depth volumes from the surfaces as they are on screen.

The point of doing this from the viewer rather than ffs_util_surf2layers is the
edits: the meshes in the session -- dragged, snapped, unsaved -- are what get
voxelised, so fixing a pial surface and seeing the layers it gives is one loop
instead of save, rerun, reload. The work is the CLI's own
(:func:`~fastfuncstuff.surface.volume_depth.export_depth_volumes`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fastfuncstuff.viewer.commands import Aspect
from fastfuncstuff.viewer.modes.base import (
    BoolControl,
    ChoiceControl,
    DialogSpec,
    FloatControl,
    IntControl,
    PathControl,
)
from fastfuncstuff.viewer.vocab import Load

#: Where the grid comes from: the image the slices are drawn on (an EPI slab
#: shows layers on exactly what it covers), or the subject's own anatomical.
GRIDS = ("underlay", "anatomical")


def _anatomical(subject: Path) -> Path | None:
    for name in ("rawavg.mgz", "orig.mgz"):
        if (subject / "mri" / name).exists():
            return subject / "mri" / name
    return None


def export_dialog(session) -> DialogSpec:
    surfaces = session.surfaces
    subject = surfaces.subject
    base = session.state.layers.base
    blocked = ""
    if not surfaces.hemis or subject is None:
        blocked = "load a FreeSurfer subject's surfaces first"
    edited = sorted(f"{h}.{s}" for h, s in surfaces.edited)
    prefix = str(subject / "laynii" / subject.name) if subject is not None else ""
    controls = (
        PathControl(
            name="prefix",
            label="prefix",
            default=prefix,
            filter="NIfTI (*.nii.gz *.nii)",
            help="Writes PREFIX_rim, _metric_equidist/_equivol, _layers_*, _midGM_*, _thickness.",
        ),
        ChoiceControl(
            name="grid",
            label="grid",
            choices=GRIDS,
            default="underlay" if base is not None else "anatomical",
            style="radio",
            help="underlay: the image on screen (its field of view, upsampled). "
            "anatomical: the subject's mri/rawavg.mgz.",
        ),
        FloatControl(
            name="dxyz",
            label="voxel",
            lo=0.1,
            hi=1.5,
            default=0.3,
            step=0.05,
            unit="mm",
            help="Output voxel size. LayNii's equivolume wants <= 0.3 mm; these "
            "metrics come from the meshes, so coarser is fine when it is all you need.",
        ),
        IntControl(name="n_layers", label="layers", lo=1, hi=20, default=3),
        BoolControl(
            name="autobox",
            label="crop to cortex",
            default=True,
            help="Crop the grid to the pial surfaces (+1 mm).",
        ),
        BoolControl(
            name="show",
            label="show layers",
            default=False,
            help="Load the equivolume layers as a layer when done.",
        ),
    )
    report: dict[str, str] = {"done": "written"}

    def run(params: dict[str, Any], progress) -> Load | None:
        from fastfuncstuff.surface.volume_depth import (
            RibbonSurfaces,
            export_depth_volumes,
            thick_report,
        )

        # Copied first: edits move vertices in place on the GUI thread, and
        # the job must voxelise one consistent surface, not one mid-drag.
        ribbons = [
            RibbonSurfaces(
                h.name,
                np.array(h.states["white"], copy=True),
                np.array(h.states["pial"], copy=True),
                h.faces.copy(),
                None if h.cortex is None else h.cortex.copy(),
            )
            for h in surfaces.hemis.values()
        ]
        if params["grid"] == "underlay":
            if base is None:
                raise ValueError("no underlay to take the grid from")
            affine, shape = base.affine, base.shape
        else:
            from fastfuncstuff.cli.util_surf2layers import master_grid

            anat = _anatomical(subject)
            if anat is None:
                raise ValueError(f"{subject}/mri has no rawavg.mgz or orig.mgz")
            affine, shape = master_grid(anat)
        if progress is not None:
            progress(0.0, f"voxelising {', '.join(r.name for r in ribbons)}")
        stem = str(params["prefix"])
        for ext in (".nii.gz", ".nii"):
            stem = stem.removesuffix(ext)
        out, grid, written = export_depth_volumes(
            ribbons,
            affine,
            shape,
            stem,
            dxyz=float(params["dxyz"]),
            autobox=bool(params["autobox"]),
            n_layers=int(params["n_layers"]),
            device=session.store.device,
            verbose=True,
        )
        warning = thick_report(out, grid)
        for line in warning:
            print(line)
        report["done"] = f"wrote {len(written)} files to {Path(stem).parent}" + (
            f"; {out.n_thick:,} GM voxels > {out.thick_limit:g} mm (see details)" if warning else ""
        )
        return Load(f"{stem}_layers_equivol.nii.gz") if params["show"] else None

    def install(result: Load | None) -> Aspect:
        return Aspect.NOTHING if result is None else session.do(result)

    blurb = "LayNii rim, depth metrics, layers and thickness from the surfaces as shown"
    blurb += f" (edited: {', '.join(edited)})." if edited else "."
    return DialogSpec(
        name="surf2layers",
        title="LayNii layers from surfaces",
        blurb=blurb,
        controls=controls,
        params={c.name: c.default for c in controls},
        run=run,
        install=install,
        run_label="export",
        blocked=blocked,
        done=lambda: report["done"],
    )


__all__ = ["GRIDS", "export_dialog"]
