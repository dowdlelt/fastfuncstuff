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
keys
  arrows / PgUp PgDn   move the crosshair
  , .                  step time            v  play / pause
  [ ]                  select layer       space  show / hide layer
  t T                  threshold down / up
  a  alpha mode        s  sign mode       b  boxed      c  colormap
  ctrl+click           set the InstaCorr seed
  ctrl+O  open         ctrl+S  save session script

examples
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

    return launch(args.datasets, device=args.device, script=args.script)


if __name__ == "__main__":
    raise SystemExit(main())
