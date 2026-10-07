"""ffs_viewer — the interactive data explorer.

Thin by design: parse, resolve the device, hand off to the UI. Everything the
window can do is reachable as a command, so a session recorded here replays
through ``-script`` without the GUI in the loop.
"""

from __future__ import annotations

import sys

from fastfuncstuff.cli_help import FfsArgumentParser
from fastfuncstuff.cli_utils import add_device_arg

EPILOG = """\
the core is the data selector: READ a directory, pick an UNDERLAY and an
OVERLAY, then +1 to stack another. MODE changes where the overlay comes from
(View / InstaCorr / InstaGLM / InstaPCA / ICA / Denoise / Preproc). the main window is a
controller -- every image and every graph is a companion window you open,
arrange and close.

DERIVE projects a design's nuisance out of a run and keeps the result as a new
layer just above it, so raw and denoised sit next to each other in the stack --
graphable together, and one keypress apart under solo.

INSTAGLM fits one run against one events file and then lets you take the model
apart: step the drift order, add a motion file and its derivatives, add PCs off
the noise pool, move the HRF's peak. Every column is pickable as a map of its
own -- a condition's percent signal change, a motion parameter's, the t or the
variance only that one regressor explains -- and each graph window draws the
measurement, the same measurement with the nuisance taken out, the fit and the
residual at every voxel it shows. Changing the design refits; changing which map
you are looking at does not.

INSTAPCA decomposes the input run on the spot -- drift out, every voxel scaled to
unit length, as ffs_denoise extracts its noise PCs -- so each map is the
correlation of a voxel's series with the component. Step through with the
arrows in a trace window, label with s / n, and SAVE NOISE writes the noise
components' time courses as an ortvec 1D file. The mask defaults to an
automask; give a mask file on the run's grid to use your own.

A CARPET window (grayplot) draws every voxel of one run at once, automasked and
row-sorted -- by correlation with the dominant component, with the seed voxel,
or with the mean of what an overlay picked out. Point it at a DERIVE'd layer to
see a cleaned carpet; the overlay is drawn as a band beside the rows.

keys (main window)
  n  N  C              new image / graph / carpet window
  f  F                 tile / stagger every window      r  raise them all
  d                    dark / light palette
  arrows / PgUp PgDn   move the crosshair
  , .                  step time            v  play / pause
  [ ]                  select layer       space  show / hide layer
  { }  u  Del          lower / raise layer, make underlay, remove
  t T                  threshold down / up
  a  alpha mode        s  sign mode       b  boxed      c  colormap
  D                    denoise the selected run into a new layer
  ctrl+O  open         ctrl+S  save session script       h  this list
  O                    cycle surface outlines (white+pial / white / pial / off)
  ctrl+shift+S         save edited surfaces as ?h.<surf>.<suffix> copies
  ctrl+shift+I         install edits over the originals (asks; keeps backups)
  V                    3-D surface window (data sampled between white and pial)
  P                    ribbon profile column: every vertex, back to front, flagged
  L                    depth profiles of the overlay around the crosshair (laminar)

keys (image window)
  1 2 3                axial / sagittal / coronal
  o                    solo the selected layer (flip between layers with [ ])
  l                    follow the crosshair, or unlock to park a slice
  + -  0               zoom in / out, fit the whole plane
  right-drag           pan            ctrl+click  set the InstaCorr seed
  w                    close

surface editing (image window, with -surfaces loaded)
  g                    edit mode: drag a white/pial outline toward where it
                       belongs; it snaps to the anatomy's edge over a 3-D brush
  G                    draw mode: press on an outline, draw where it should
                       run, release on the same outline; the stretch moves onto
                       the line and the surface around follows
  p                    point mode: select a vertex; Delete removes it, i splits
                       its longest edge, I all its edges (every surface + file)
  ( )                  brush radius         m  snap / follow the hand
  ctrl+Z               undo                 Esc  cancel the drag

keys (graph window)
  + -                  more / fewer voxels    s  shared scale    w  close

keys (carpet window)
  o                    next row order         r  rebuild         w  close

examples
  ffs_viewer -read results.subj01/
  ffs_viewer anat.nii.gz stats.nii.gz
  ffs_viewer -device cpu bold.nii.gz
  ffs_viewer -script session.ffs
  ffs_viewer -surfaces $SUBJECTS_DIR/subj subj/SUMA/brain.nii.gz stats.nii.gz
  ffs_viewer -mesh lh.white -mesh lh.pial.ffsedit T1.nii.gz
"""


def build_parser() -> FfsArgumentParser:
    p = FfsArgumentParser(
        prog="ffs_viewer",
        description="Interactive GPU-first viewer for volumetric data.",
        epilog=EPILOG,
    )
    p.add_argument(
        "datasets",
        nargs="*",
        help="Datasets to load, bottom layer first (anatomy, then overlays).",
    )
    p.add_argument(
        "-read",
        metavar="DIR",
        help="Read this directory into the underlay/overlay pickers on startup.",
    )
    p.add_argument(
        "-script",
        metavar="FILE",
        help="Replay a recorded session script before showing the window.",
    )
    p.add_argument(
        "-no_window",
        action="store_true",
        help="Run the script and exit without opening a window (for testing "
        "and for regenerating figures headlessly).",
    )
    p.add_argument(
        "-surfaces",
        metavar="SUBJ_DIR",
        help="FreeSurfer subject directory: outline its white and pial surfaces on "
        "the slices. Placed in scanner space from the surface files themselves, so "
        "they line up with orig.mgz, the SUMA SurfVol, or anything aligned to them.",
    )
    p.add_argument(
        "-mesh",
        metavar="FILE",
        action="append",
        default=[],
        help="A FreeSurfer surface file to outline, no subject directory needed; "
        "repeat for more. Hemisphere and white/pial are read from the name "
        "(lh.pial, rh.smoothwm...). With only one of white and pial, depth "
        "sampling collapses onto that surface; load the other to get the ribbon "
        "back. After -surfaces, files are added to the subject's mesh list.",
    )
    p.add_argument(
        "-surf_data",
        metavar="FILE",
        action="append",
        default=[],
        help="A per-vertex result (.func.gii from ffs_reml on a surface) loaded as a "
        "layer once the meshes are in: painted into the cortical ribbon on the slices, "
        "drawn on its own vertices in a surface window, and set by the same controls as "
        "any layer (threshold, p, sub-bricks, clusters). It must be on a loaded mesh "
        "(-surfaces, or -mesh PREFIX.SPACE.lh.white.surf.gii and pial), matched by "
        "fingerprint; the other hemisphere's file beside it (.lh. / .rh.) joins the "
        "same layer. Repeat for more.",
    )
    add_device_arg(p, default="auto")
    return p


def _surf_data_commands(paths: list[str]):
    """Plain LOADs, issued after the meshes: a surface result is a layer like any
    other, it just needs the meshes it was made on to be there first."""
    from fastfuncstuff.viewer.vocab import AddLayer

    return [AddLayer(path) for path in paths]


def _mesh_commands(paths: list[str]):
    from fastfuncstuff.viewer.meshlist import infer_label
    from fastfuncstuff.viewer.vocab import LoadMesh

    out = []
    for path in paths:
        hemi, kind = infer_label(path)
        if hemi is None or kind is None:
            raise SystemExit(
                f"ffs_viewer: -mesh {path}: can't tell "
                f"{'the hemisphere (lh/rh)' if hemi is None else 'white or pial'} from the "
                "name; rename it, or load it from the MESH window, which asks"
            )
        out.append(LoadMesh(path, hemi, kind))
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.no_window:
        # Headless replay shares the session and command path the GUI uses, so
        # a script that works here works there.
        from fastfuncstuff.cli_utils import setup_device
        from fastfuncstuff.viewer.session import ViewerSession

        session = ViewerSession(device=setup_device(args.device))
        try:
            if args.read:
                session.read_directory(args.read)
            for path in args.datasets:
                session.load(path)
            if args.surfaces:
                from fastfuncstuff.viewer.vocab import LoadSurfaces

                session.do(LoadSurfaces(args.surfaces))
            for cmd in [*_mesh_commands(args.mesh), *_surf_data_commands(args.surf_data)]:
                session.do(cmd)
            if args.script:
                session.run_script(open(args.script).read())
            print(session.to_script(header="replayed"), end="")
        finally:
            session.close()
        return 0

    try:
        from fastfuncstuff.viewer.ui.window import launch
    except ImportError as exc:  # PySide6 is an extra, not a core dependency
        print(
            f"ffs_viewer needs the GUI extra: pip install 'fastfuncstuff[viewer]'\n  ({exc})",
            file=sys.stderr,
        )
        return 1

    return launch(
        args.datasets,
        device=args.device,
        script=args.script,
        directory=args.read,
        surfaces=args.surfaces,
        meshes=args.mesh,
        surf_data=_surf_data_commands(args.surf_data),
    )


if __name__ == "__main__":
    raise SystemExit(main())
