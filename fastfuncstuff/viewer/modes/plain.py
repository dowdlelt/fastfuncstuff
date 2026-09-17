"""Plain mode: the overlay is whatever file you picked.

The default, and the one mode that computes nothing. It exists so that "no
mode" is still a mode -- the UI asks the active mode what controls to show and
what the slider means, and having a real object answer those keeps every other
code path free of ``if mode is None``.
"""

from __future__ import annotations

from fastfuncstuff.viewer.modes.base import ComputedOverlay, Mode, OverlayKind, mode


@mode
class PlainMode(Mode):
    name = "plain"
    label = "View"
    overlay_kind = OverlayKind.VALUE
    produces_overlay = False

    def compute(self) -> ComputedOverlay | None:
        return None

    def status(self) -> str:
        return ""
