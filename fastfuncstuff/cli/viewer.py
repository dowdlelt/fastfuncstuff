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
(View / InstaCorr / ICA). the main window is a controller -- every image and
every graph is a companion window you open, arrange and close.

keys (main window)
  n  N                 new image / new graph window
  f  F                 tile / stagger every window      r  raise them all
  d                    dark / light palette
  arrows / PgUp PgDn   move the crosshair
  , .                  step time            v  play / pause
  [ ]                  select layer       space  show / hide layer
  t T                  threshold down / up
  a  alpha mode        s  sign mode       b  boxed      c  colormap
  ctrl+O  open         ctrl+S  save session script       h  this list

keys (image window)
  1 2 3                axial / sagittal / coronal
  o                    solo the selected layer (flip between layers with [ ])
  l                    follow the crosshair, or unlock to park a slice
  ctrl+click           set the InstaCorr seed             w  close

keys (graph window)
  + -                  more / fewer voxels    s  shared scale    w  close

examples
  ffs_viewer -read results.subj01/
  ffs_viewer anat.nii.gz stats.nii.gz
  ffs_viewer -device cpu bold.nii.gz
  ffs_viewer -script session.ffs
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
    add_device_arg(p, default="auto")
    return p


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

    return launch(args.datasets, device=args.device, script=args.script, directory=args.read)


if __name__ == "__main__":
    raise SystemExit(main())
