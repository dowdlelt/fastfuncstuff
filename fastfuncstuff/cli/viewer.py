"""ffs_viewer — the interactive data explorer.

Thin by design: parse, resolve the device, hand off to the UI. Everything the
window can do is reachable as a command, so a session recorded here replays
through ``-script`` without the GUI in the loop.
"""

from __future__ import annotations

import io
import sys
import time

from fastfuncstuff.cli_help import FfsArgumentParser

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
  , .                  step time            v  play / pause (a in a 3-D window)
                       (a selected layer that does not follow time steps its own
                       sub-bricks; with THR ON +1 / +2, one Coef group at a time)
  [ ]                  select layer       o  show / hide the overlay (space too)
  O                    flip the overlay against the layer under it (again: back)
  { }  u  Del          lower / raise layer, make underlay, remove
  t T                  threshold down / up
  a  alpha mode        s  sign mode       b  boxed      c  colormap
  D                    denoise the selected run into a new layer
  ctrl+O  open         ctrl+S  save session script       h  this list
  ctrl+P               print (and copy) the ffs_viewer command that reopens
                       what is loaded -- files, directory, surfaces; not edits
  alt+o                cycle surface outlines (white+pial / white / pial / off)
  ctrl+shift+S         save edited surfaces as ?h.<surf>.<suffix> copies
  ctrl+shift+I         install edits over the originals (asks; keeps backups)
  V                    3-D surface window (data sampled between white and pial)
  P                    ribbon profile column: every vertex, back to front, flagged
  L                    depth profiles of the overlay around the crosshair (laminar)

keys (image window)
  1 2 3                axial / sagittal / coronal
  o  O                 show / hide the overlay, or flip it against the one below
  S                    solo the selected layer (flip between layers with [ ])
  l                    lock: Lock (follow the crosshair) -> LOCK (also share zoom
                       with every LOCK window, centred on one point) -> off (park)
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
  ffs_viewer -read results.subj01/ results.subj02/
  ffs_viewer anat.nii.gz stats.nii.gz        (those two, loaded; nothing else listed)
  ffs_viewer results.subj01/
  ffs_viewer -device cpu bold.nii.gz
  ffs_viewer -script session.ffs
  ffs_viewer -surfaces $SUBJECTS_DIR/subj subj/SUMA/brain.nii.gz stats.nii.gz
  ffs_viewer -mesh lh.white -mesh lh.pial.ffsedit T1.nii.gz
"""


def _say(text: str) -> None:
    """One line of what startup is doing. stderr, so ``-no_window``'s script stays clean."""
    print(f"ffs_viewer: {text}", file=sys.stderr, flush=True)


def build_parser() -> FfsArgumentParser:
    # Here and not at the top of the module: cli_utils imports torch, and the
    # console script imports this module before main() can say anything.
    from fastfuncstuff.cli_utils import add_device_arg

    p = FfsArgumentParser(
        prog="ffs_viewer",
        description="Interactive GPU-first viewer for volumetric data.",
        epilog=EPILOG,
    )
    p.add_argument(
        "datasets",
        nargs="*",
        metavar="PATH",
        help="Same as -read: files load as layers, bottom first (anatomy, then "
        "overlays), and with no directory given they are all the pickers list. A "
        "directory is read.",
    )
    p.add_argument(
        "-read",
        metavar="PATH",
        nargs="+",
        help="What to open on startup. A directory fills the underlay/overlay pickers. "
        "Several directories all join the directory list, with only the first read until "
        "another is picked. Files load as layers, and with no directory given they are "
        "all the pickers list; their directories are never scanned, the safe choice on a "
        "synced folder (Dropbox), where a scan can fetch every file. Takes every path "
        "after it.",
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
        "-surf_space",
        metavar="MESH",
        default=None,
        help="With -surfaces: show the subject on a template mesh (onavg-ico64, "
        "fsaverage... by name or path, as ffs_nwarp -surf_mesh finds them), its "
        "vertices placed in this subject through sphere.reg -- white, pial, inflated, "
        "curv/sulc carried along. Results made on that template then load with "
        "-surf_data directly. View-only: edits are saved from the native mesh.",
    )
    p.add_argument(
        "-mesh",
        metavar="FILE",
        action="append",
        default=[],
        help="A FreeSurfer surface file to outline, no subject directory needed; "
        "repeat for more. Hemisphere and white/pial are read from the name "
        "(lh.pial, rh.smoothwm...); a midthickness stands in for both. With only "
        "one of white and pial, depth "
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
        "fingerprint -- or, for a hemisphere with nothing loaded, the mesh the file "
        "names in its own 'geometry' metadata (ffs_group and ffs_reml write it) is "
        "loaded for it. The other hemisphere's file beside it (.lh. / .rh.) joins the "
        "same layer. Repeat for more.",
    )
    add_device_arg(p, default="auto")
    p.add_argument(
        "-vram_budget",
        metavar="SIZE",
        default=None,
        help="Most GPU memory the viewer holds between interactions: 6G, 512M, or a share "
        "of the card such as 40%%. Default 25%% of the card's total. Everything held also "
        "has a RAM copy, so a smaller budget costs re-uploads, not results; the viewer "
        "also hands memory back whenever the card runs short.",
    )
    return p


def _surf_data_commands(paths: list[str]):
    """Plain LOADs, issued after the meshes: a surface result is a layer like any
    other, it just needs the meshes it was made on to be there first."""
    from fastfuncstuff.viewer.vocab import Load

    # Load, not AddLayer: a result over an anatomy gets the overlay defaults (a
    # signed map in red-blue on a symmetric range, a starting threshold).
    return [Load(path) for path in paths]


def _mesh_commands(paths: list[str]):
    from pathlib import Path

    from fastfuncstuff.viewer.meshlist import infer_label
    from fastfuncstuff.viewer.vocab import LoadMesh

    out = []
    for path in paths:
        hemi, kind = infer_label(path)
        if kind is None and "midthick" in Path(path).name.lower():
            # A midthickness (fs_LR, group templates) stands in for both white and
            # pial: the hemisphere is made from it and the ribbon collapses onto it.
            kind = "white"
        if hemi is None or kind is None:
            raise SystemExit(
                f"ffs_viewer: -mesh {path}: can't tell "
                f"{'the hemisphere (lh/rh)' if hemi is None else 'white or pial'} from the "
                "name; rename it, or load it from the MESH window, which asks"
            )
        out.append(LoadMesh(path, hemi, kind))
    return out


def _read_targets(args) -> tuple[list[str], list[str], list[str]]:
    """The paths sorted out: directories to offer, files to list, datasets to load.

    A file someone named is a file they want to see, so every one loads, however
    it was given. With no directory they are also all the picker lists.
    """
    from pathlib import Path

    from fastfuncstuff.viewer.catalog import split_read_targets

    try:
        directories, files = split_read_targets(args.read or [])
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ffs_viewer: -read {exc}") from exc
    # Only directories are picked out here. Anything else goes to the loader as
    # always, which knows more dataset spellings than the picker's suffix list.
    bare_dirs = [p for p in args.datasets if Path(p).is_dir()]
    bare = [p for p in args.datasets if p not in bare_dirs]
    dirs = [*bare_dirs, *(str(d) for d in directories)]
    named = list(dict.fromkeys([*bare, *(str(f) for f in files)]))
    return dirs, [] if dirs else named, named


def _enable_stack_dumps() -> None:
    """``kill -USR1 <pid>`` prints every thread's Python stack to stderr.

    A GUI freeze is the one bug a user cannot report usefully: by the time it is
    noticed the only evidence is that the window stopped painting, and "not
    responding" says nothing about which call is holding the main thread. This
    makes a frozen viewer answer for itself, on a live process, with no restart
    and no debugger attached -- the dump goes to the terminal it was started
    from. Costs nothing until the signal arrives.

    SIGUSR1 because Qt uses none of the standard signals and this one survives a
    blocked event loop: faulthandler writes from the signal handler itself rather
    than queueing work the stuck thread would have to come back to run.
    """
    import faulthandler
    import signal

    handler = getattr(signal, "SIGUSR1", None)
    if handler is None:  # not POSIX
        return
    # chain=False deliberately: SIGUSR1's default action is to terminate, so
    # chaining would dump the stack and then kill the window the dump was meant
    # to diagnose -- which defeats the point of asking a LIVE process what it is
    # doing. Measured: with chain=True the process died on the signal.
    try:
        faulthandler.register(handler, all_threads=True, chain=False)
    except (OSError, RuntimeError, ValueError, io.UnsupportedOperation):
        # faulthandler writes to a real file descriptor, and stderr does not
        # always have one -- captured output, or a GUI launched with no console.
        # This is a diagnostic: it never gets to be the reason startup fails.
        pass


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    _enable_stack_dumps()
    if not {"-h", "-help", "--help"} & set(sys.argv[1:] if argv is None else argv):
        _say("starting, importing torch...")
    args = build_parser().parse_args(argv)
    directories, files, datasets = _read_targets(args)

    from fastfuncstuff.viewer.catalog import opening

    if args.no_window:
        # Headless replay shares the session and command path the GUI uses, so
        # a script that works here works there.
        from fastfuncstuff.cli_utils import setup_device
        from fastfuncstuff.viewer.residency import apply_vram_budget
        from fastfuncstuff.viewer.session import ViewerSession

        session = ViewerSession(device=setup_device(args.device))
        apply_vram_budget(args.vram_budget, session.store.device)
        _say(f"device {session.store.device}")
        try:
            if directories:
                _say(f"reading {directories[0]}")
                session.read_directory(directories[0])
                session.offer_directories(directories[1:])
                _say(f"found {len(session.catalog)} datasets")
            for path in opening(datasets):
                session.load(path)
            if files:
                session.read_files(files)
            if args.surfaces:
                from fastfuncstuff.viewer.vocab import LoadSurfaces

                _say(f"loading surfaces from {args.surfaces}")
                session.do(LoadSurfaces(args.surfaces, space=args.surf_space or ""))
            for cmd in [*_mesh_commands(args.mesh), *_surf_data_commands(args.surf_data)]:
                session.do(cmd)
            if args.script:
                _say(f"replaying {args.script}")
                session.run_script(open(args.script).read())
            _say(f"done in {time.perf_counter() - started:.1f} s")
            print(session.to_script(header="replayed"), end="")
        finally:
            session.close()
        return 0

    _say("importing the viewer window...")
    try:
        from fastfuncstuff.viewer.ui.window import launch
    except ImportError as exc:  # a broken or missing Qt install
        print(
            f"ffs_viewer could not import Qt (PySide6); reinstall with: pip install 'PySide6>=6.7'\n  ({exc})",
            file=sys.stderr,
        )
        return 1

    return launch(
        datasets,
        device=args.device,
        script=args.script,
        directories=directories,
        files=files,
        surfaces=args.surfaces,
        surf_space=args.surf_space,
        meshes=args.mesh,
        surf_data=_surf_data_commands(args.surf_data),
        vram_budget=args.vram_budget,
        say=_say,
        started=started,
    )


if __name__ == "__main__":
    raise SystemExit(main())
