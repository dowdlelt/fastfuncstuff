"""The window manager: real windows reconciled against the viewport list.

One rule, and it is the reason this file exists: **the viewport list is the
truth and the windows are a projection of it.** Opening a window is dispatching
OPEN_VIEW; closing one is dispatching CLOSE_VIEW; the manager then makes the
screen match. That is what makes a click, a keystroke, a replayed script and a
restored layout the same code path instead of four that drift.

The alternative -- widgets that create each other and a separate save-layout
routine that walks them afterwards -- is how AFNI ended up able to record only
the actions someone remembered to instrument.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.ui.carpetwindow import CarpetWindow
from fastfuncstuff.viewer.ui.gridgraph import GraphWindow
from fastfuncstuff.viewer.ui.imagewindow import ImageWindow
from fastfuncstuff.viewer.viewports import ViewKind, Viewport
from fastfuncstuff.viewer.vocab import CloseView, SetViewGeometry

#: Gap left between tiled windows, and around the edge of the work area.
TILE_GAP = 6


class WindowManager(QtCore.QObject):
    """Owns the companion windows and keeps them matching the viewports."""

    #: A carpet needs rebuilding on the worker; the controller owns the runner.
    rebuild_requested = QtCore.Signal(str)

    def __init__(
        self,
        session,
        dispatch: Callable[[Command], None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self._dispatch = dispatch
        self._parent = parent
        self.windows: dict[str, ImageWindow | GraphWindow | CarpetWindow] = {}
        #: Set while the manager is placing windows, so the geometry it writes
        #: back does not read as the user having dragged them.
        self._placing = False

    # -- reconciliation -------------------------------------------------
    def sync(self) -> None:
        """Create, destroy and update windows so they match the viewports."""
        wanted = {v.id: v for v in self.session.state.viewports}
        for vid in [k for k in self.windows if k not in wanted]:
            win = self.windows.pop(vid)
            win.blockSignals(True)  # its closeEvent must not re-dispatch
            win.close()
            win.deleteLater()
        for vid, viewport in wanted.items():
            win = self.windows.get(vid)
            if win is None:
                win = self._build(viewport)
                self.windows[vid] = win
                if viewport.geometry is not None:
                    win.setGeometry(QtCore.QRect(*viewport.geometry))
                win.show()
            win.apply(viewport)

    def _build(self, viewport: Viewport) -> ImageWindow | GraphWindow | CarpetWindow:
        if viewport.is_image:
            win = ImageWindow(viewport.id, self.session, self._dispatch, self._parent)
        elif viewport.is_carpet:
            win = CarpetWindow(viewport.id, self.session, self._dispatch, self._parent)
            win.scrubbed.connect(self._on_scrubbed)
            win.rebuild_requested.connect(self.rebuild_requested)
        else:
            win = GraphWindow(viewport.id, self.session, self._dispatch, self._parent)
            win.scrubbed.connect(self._on_scrubbed)
        win.closed.connect(self._on_closed)
        return win

    def _on_closed(self, vid: str) -> None:
        """A window closed by its own title bar still goes through the bus."""
        if vid in self.windows:
            self.windows.pop(vid, None)
            self._dispatch(CloseView(vid))

    def _on_scrubbed(self, index: int) -> None:
        from fastfuncstuff.viewer.vocab import SetIndex

        self._dispatch(SetIndex(int(index)))

    # -- drawing ---------------------------------------------------------
    def redraw(self, dirty: Aspect) -> None:
        """Push whatever changed into whichever windows care about it."""
        # VIEWPORTS belongs in the image set, not only in sync(): an image
        # draws the footprint of every *graph* on its plane, so opening a graph
        # or stepping its grid changes what an image has to show. This is the
        # same hazard as the crosshair redraw that did not listen for
        # CROSSHAIR -- a consumer whose input is wider than it looks.
        images = dirty & (
            Aspect.SLICES
            | Aspect.COLORMAP
            | Aspect.THRESHOLD
            | Aspect.TIME
            | Aspect.GRID
            | Aspect.CROSSHAIR
            | Aspect.LAYERS
            | Aspect.VIEWPORTS
        )
        graphs = dirty & (Aspect.CROSSHAIR | Aspect.GRAPH | Aspect.TIME | Aspect.LAYERS)
        for win in list(self.windows.values()):
            if isinstance(win, ImageWindow):
                if images:
                    win.redraw()
            elif graphs:
                # A carpet's refresh only moves its time cursor; the picture
                # itself is seconds of work and is rebuilt deliberately.
                win.refresh()

    def restyle(self) -> None:
        """Re-read the palette in every companion window."""
        for win in list(self.windows.values()):
            win.restyle()

    # -- opening ---------------------------------------------------------
    def open(self, kind: ViewKind, plane: Plane = Plane.AXIAL) -> str:
        return self.session.open_view(kind, plane)

    def carpets(self) -> list[CarpetWindow]:
        return [w for w in self.windows.values() if isinstance(w, CarpetWindow)]

    def mark_carpets_stale(self) -> None:
        for window in self.carpets():
            window.mark_stale()

    def focused(self) -> ImageWindow | GraphWindow | CarpetWindow | None:
        """Whichever companion window has focus, if any."""
        active = QtWidgets.QApplication.activeWindow()
        for win in self.windows.values():
            if win is active:
                return win
        return None

    def raise_all(self) -> None:
        for win in self.windows.values():
            win.raise_()

    # -- arrangement -----------------------------------------------------
    def tile(self, anchor: QtWidgets.QWidget | None = None) -> None:
        """Lay every window out on a grid over the free part of the screen.

        The arrangement is computed here and written back as one
        SET_VIEW_GEOMETRY per window, rather than being an arrangement command
        of its own. That way a recorded session replays the rectangles it
        actually had, on a screen that may be a different size, instead of
        re-running a tiling algorithm against different inputs.
        """
        windows = [w for w in self.windows.values() if w.isVisible()]
        if not windows:
            return
        area = self._work_area(anchor)
        cols = max(1, int(len(windows) ** 0.5 + 0.999))
        rows = max(1, -(-len(windows) // cols))
        cw = (area.width() - TILE_GAP * (cols + 1)) // cols
        ch = (area.height() - TILE_GAP * (rows + 1)) // rows
        self._placing = True
        try:
            for i, win in enumerate(windows):
                r, c = divmod(i, cols)
                x = area.x() + TILE_GAP + c * (cw + TILE_GAP)
                y = area.y() + TILE_GAP + r * (ch + TILE_GAP)
                win.setGeometry(x, y, cw, ch)
                self._dispatch(SetViewGeometry(win.vid, x, y, cw, ch))
        finally:
            self._placing = False

    def cascade(self, anchor: QtWidgets.QWidget | None = None) -> None:
        """Stagger the windows so every title bar is reachable."""
        windows = [w for w in self.windows.values() if w.isVisible()]
        if not windows:
            return
        area = self._work_area(anchor)
        step = 30
        w = min(area.width() - step * len(windows), max(520, area.width() // 2))
        h = min(area.height() - step * len(windows), max(420, area.height() // 2))
        self._placing = True
        try:
            for i, win in enumerate(windows):
                x, y = area.x() + TILE_GAP + i * step, area.y() + TILE_GAP + i * step
                win.setGeometry(x, y, w, h)
                self._dispatch(SetViewGeometry(win.vid, x, y, w, h))
                win.raise_()
        finally:
            self._placing = False

    def _work_area(self, anchor: QtWidgets.QWidget | None) -> QtCore.QRect:
        """The screen, minus whatever the controller window is occupying.

        Tiling companion windows *over* the controller would hide the very
        panel they are driven from, which is the one window that must stay
        reachable.
        """
        screen = (
            anchor.screen() if anchor is not None else None
        ) or QtGui.QGuiApplication.primaryScreen()
        area = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1280, 800)
        if anchor is None or not anchor.isVisible():
            return area
        frame = anchor.frameGeometry()
        right = area.right() - frame.right()
        below = area.bottom() - frame.bottom()
        if right >= below:
            return QtCore.QRect(frame.right() + TILE_GAP, area.y(), max(right, 200), area.height())
        return QtCore.QRect(area.x(), frame.bottom() + TILE_GAP, area.width(), max(below, 200))

    def close_all(self) -> None:
        for win in list(self.windows.values()):
            win.blockSignals(True)
            win.close()
        self.windows.clear()


__all__ = ["TILE_GAP", "WindowManager"]
