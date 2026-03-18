"""Backward-compatible shim for qwarp CLI.

The canonical implementation now lives in `qwarp_cli.py`.
This module is kept so existing imports like
`fastfuncstuff.processing.cli` continue to work.
"""

from .qwarp_cli import *  # noqa: F401,F403
