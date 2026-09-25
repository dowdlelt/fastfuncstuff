"""A floating image window: one viewport, one plane, its own settings.

Image windows are top-level and independent because the things people want two
of are images. Two axial views soloed on an EPI and an anat, flipped between,
is how you see what registration did; a reference slice parked while you
navigate elsewhere is how you compare. Neither is expressible when the number
of images is fixed at three and their identity is their plane.

The window owns no state. It reads a :class:`~viewer.viewports.Viewport` and
dispatches commands, so what it shows is reproducible from a recorded script
rather than from whatever the widgets happen to be set to.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.commands import Command
from fastfuncstuff.viewer.compose import plane_view, render_viewport
from fastfuncstuff.viewer.slicing import plane_layout
from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.panes import ImagePane
from fastfuncstuff.viewer.ui.shortcuts import Binding, ShortcutHelp, keep_keys_for_shortcuts
from fastfuncstuff.viewer.viewports import Viewport
from fastfuncstuff.viewer.vocab import (
    SetEdges,
    SetIJK,
    SetLayerOpacity,
    SetPan,
    SetSeed,
    SetViewLocked,
    SetViewPlane,
    SetViewPosition,
    SetViewSolo,
    SetZoom,
)

PLANE_KEYS = {Plane.AXIAL: "1", Plane.SAGITTAL: "2", Plane.CORONAL: "3"}

#: Header widths. Below the first the buttons keep only their key, below the
#: second the header goes away entirely -- the keys still work, so nothing is
#: lost but the reminder, and a nine-window wall of small images is worth more
#: than a row of labels on each.
COMPACT_WIDTH = 330
BARE_WIDTH = 170


class ImageWindow(QtWidgets.QWidget):
    """One image viewport as a top-level window."""

    closed = QtCore.Signal(str)

    def __init__(
        self,
        vid: str,
        session,
        dispatch: Callable[[Command], None],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.vid = vid
        self.session = session
        self._dispatch = dispatch
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.setStyleSheet(theme.stylesheet())

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self.header = QtWidgets.QWidget()
        bar = QtWidgets.QHBoxLayout(self.header)
        bar.setContentsMargins(6, 4, 6, 4)
        bar.setSpacing(4)
        #: (button, full label, key-only label), walked on every resize.
        self._labels: list[tuple[QtWidgets.QPushButton, str, str]] = []
        self._plane_buttons: dict[Plane, QtWidgets.QPushButton] = {}
        for plane, key in PLANE_KEYS.items():
            b = self._button(plane.value[:3].upper(), key, f"Show the {plane.value} plane")
            b.clicked.connect(
                lambda _=False, p=plane: self._dispatch(SetViewPlane(self.vid, str(p)))
            )
            bar.addWidget(b)
            self._plane_buttons[plane] = b
        bar.addSpacing(6)

        self.solo_button = self._button(
            "SOLO",
            "o",
            "Draw only the selected layer. With [ and ] this flips between "
            "neighbouring layers in place, which is how alignment is checked.",
        )
        self.solo_button.clicked.connect(lambda on: self._dispatch(SetViewSolo(self.vid, bool(on))))
        bar.addWidget(self.solo_button)

        self.lock_button = self._button(
            "LOCK", "l", "Follow the shared crosshair. Unlock to park a slice."
        )
        self.lock_button.clicked.connect(self._toggle_lock)
        bar.addWidget(self.lock_button)

        bar.addStretch(1)
        self.slice_label = QtWidgets.QLabel("")
        self.slice_label.setObjectName("value")
        bar.addWidget(self.slice_label)
        v.addWidget(self.header)

        # Opacity of the selected layer, one key away from the image it changes.
        # A row of its own rather than a header button, so it survives the
        # header being shed on a narrow window.
        self.opacity_bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.opacity_bar)
        row.setContentsMargins(6, 0, 6, 2)
        row.setSpacing(4)
        self.opacity_name = QtWidgets.QLabel("")
        self.opacity_name.setObjectName("value")
        row.addWidget(self.opacity_name)
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.opacity_slider.valueChanged.connect(self._opacity_moved)
        row.addWidget(self.opacity_slider, 1)
        self.opacity_value = QtWidgets.QLabel("")
        self.opacity_value.setObjectName("value")
        row.addWidget(self.opacity_value)
        self.opacity_bar.setVisible(False)
        v.addWidget(self.opacity_bar)

        self.pane = ImagePane(Plane.AXIAL)
        self.pane.picked.connect(lambda r, c: self._pick(r, c, seed=False))
        self.pane.seeded.connect(lambda r, c: self._pick(r, c, seed=True))
        self.pane.stepped.connect(self._step)
        self.pane.panned.connect(self._pan_by)
        v.addWidget(self.pane, 1)
        # Two thirds of what it used to be. An EPI slice is 64 to 100 voxels
        # across, so a 420-pixel window was showing it at four times its own
        # resolution and spending most of the screen on interpolation -- and
        # three of them left nowhere to put a graph or an HRF curve.
        self.resize(280, 280)
        self.setMinimumSize(64, 64)

        self.help = ShortcutHelp(self, f"image · {vid}")
        self.help.apply(
            [
                *[
                    Binding(k, f"{p.value} plane", lambda p=p: self._set_plane(p), group="plane")
                    for p, k in PLANE_KEYS.items()
                ],
                Binding(
                    "Left", "crosshair left", lambda: self._nudge_in_plane(0, -1), group="navigate"
                ),
                Binding(
                    "Right", "crosshair right", lambda: self._nudge_in_plane(0, 1), group="navigate"
                ),
                Binding(
                    "Up", "crosshair up", lambda: self._nudge_in_plane(-1, 0), group="navigate"
                ),
                Binding(
                    "Down", "crosshair down", lambda: self._nudge_in_plane(1, 0), group="navigate"
                ),
                Binding("PgUp", "next slice", lambda: self._step(1), group="navigate"),
                Binding("PgDn", "previous slice", lambda: self._step(-1), group="navigate"),
                Binding("o", "solo the selected layer", self.solo_button.click, group="view"),
                Binding("+", "zoom in", lambda: self._zoom_by(1.25), group="view", aliases=("=",)),
                Binding("-", "zoom out", lambda: self._zoom_by(1 / 1.25), group="view"),
                Binding("0", "fit the whole plane", self._reset_view, group="view"),
                Binding("right-drag", "pan", None, group="view"),
                Binding("l", "follow the crosshair", self.lock_button.click, group="view"),
                Binding("scroll", "step through slices", None, group="view"),
                Binding(".", "next volume", lambda: self._step_time(1), group="time"),
                Binding(",", "previous volume", lambda: self._step_time(-1), group="time"),
                Binding("click", "move the crosshair", None, group="view"),
                Binding("ctrl+click", "set the InstaCorr seed", None, group="view"),
                Binding("e", "draw the selected layer as edges", self._toggle_edges, group="layer"),
                Binding(
                    "6",
                    "opacity slider for the selected layer",
                    self._toggle_opacity,
                    group="layer",
                ),
                Binding("h", "this list", self.help.toggle, group="window"),
                Binding("w", "close this window", self.close, group="window"),
            ]
        )
        keep_keys_for_shortcuts(self)

    def _button(self, text: str, key: str, tip: str) -> QtWidgets.QPushButton:
        """A header button that knows how to say itself in less space."""
        full, bare = theme.key_label(text, key), f"[{key}]"
        b = QtWidgets.QPushButton(full)
        b.setCheckable(True)
        b.setToolTip(f"{tip}  ({key})")
        b.setStyleSheet(f"QPushButton {{ font-size: {theme.FONT_SMALL}px; padding: 2px 5px; }}")
        self._labels.append((b, full, bare))
        return b

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 (Qt)
        """Shed the header as the window narrows, rather than refusing to."""
        width = event.size().width()
        self.header.setVisible(width >= BARE_WIDTH)
        compact = width < COMPACT_WIDTH
        for button, full, bare in self._labels:
            button.setText(bare if compact else full)
        self.slice_label.setVisible(not compact)
        super().resizeEvent(event)

    # -- input ---------------------------------------------------------
    def _set_plane(self, plane: Plane) -> None:
        self._dispatch(SetViewPlane(self.vid, str(plane)))

    def _toggle_lock(self, on: bool) -> None:
        # Unlocking parks the window on the slice it is showing. Without that
        # it would keep following until something else moved, which reads as
        # the button not working.
        if not on:
            pane_pos = self.pane.position
            if pane_pos is not None:
                self._dispatch(SetViewPosition(self.vid, int(pane_pos)))
        self._dispatch(SetViewLocked(self.vid, bool(on)))

    def _selected(self):
        return self.session.state.selected_layer()

    def _toggle_edges(self) -> None:
        layer = self._selected()
        if layer is not None:
            self._dispatch(SetEdges(layer.key, not layer.edges))

    def _toggle_opacity(self) -> None:
        """Show or hide the slider; opening it on an opaque layer halves it.

        Halved so the key does something visible on its own -- a slider that
        appears over a layer still drawn at 100% has changed nothing yet.
        """
        showing = not self.opacity_bar.isVisible()
        self.opacity_bar.setVisible(showing)
        layer = self._selected()
        if showing and layer is not None and layer.opacity >= 1.0:
            self._dispatch(SetLayerOpacity(layer.key, 0.5))
        self._sync_opacity()

    def _opacity_moved(self, value: int) -> None:
        layer = self._selected()
        if layer is not None:
            self.opacity_value.setText(f"{value}%")
            self._dispatch(SetLayerOpacity(layer.key, value / 100.0))

    def _sync_opacity(self) -> None:
        if not self.opacity_bar.isVisible():
            return
        layer = self._selected()
        self.opacity_slider.setEnabled(layer is not None)
        pct = 0 if layer is None else int(round(layer.opacity * 100))
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(pct)
        self.opacity_slider.blockSignals(False)
        self.opacity_name.setText("" if layer is None else layer.name)
        self.opacity_value.setText(f"{pct}%")

    def _viewport(self) -> Viewport | None:
        return self.session.state.viewports.find(self.vid)

    def _pick(self, row: int, col: int, *, seed: bool) -> None:
        state = self.session.state
        vp = self._viewport()
        view = None if vp is None else plane_view(state, vp)
        if state.grid is None or vp is None or view is None:
            return
        # Through the view, not the layout: when the pane is showing a crop,
        # image pixel (0, 0) is not grid voxel 0 and a click would land
        # wherever the offset happened not to be applied.
        ijk = view.to_ijk(row, col, state.crosshair)
        # A click in an unlocked window still reports where it was clicked --
        # it just does not take its own slice from the crosshair afterwards.
        self._dispatch(SetIJK(*ijk))
        if seed:
            self._dispatch(SetSeed(*ijk))

    def _nudge_in_plane(self, drow: int, dcol: int) -> None:
        """Move the crosshair one voxel, in the direction the key points on screen.

        Screen-relative rather than volume-relative, which is the difference
        between this and the controller's arrow keys. The controller has no
        picture, so ``Up`` there can only mean "+y"; here there is a picture,
        and ``Up`` has to mean up in it whatever axis that turns out to be.
        Which axis, and which way along it, is what the plane's layout knows.

        The step is applied to the volume axis rather than to the image row or
        column, because converting an out-of-range row back through a flipped
        axis lands at the *far* edge: pressing Up at the top of a flipped plane
        would jump the crosshair to the bottom. The layout's flip is the whole
        of the difference, so applying it here is equivalent and cannot wrap.
        """
        state = self.session.state
        vp = self._viewport()
        if state.grid is None or vp is None:
            return
        layout = plane_layout(state.grid.affine, vp.plane)
        axis = layout.col if dcol else layout.row
        flipped = layout.col_flip if dcol else layout.row_flip
        step = dcol or drow
        ijk = list(state.crosshair)
        # SetIJK clamps to the grid, so the edge stops rather than wrapping.
        ijk[axis] += -step if flipped else step
        self._dispatch(SetIJK(*ijk))

    def _step_time(self, delta: int) -> None:
        """Step the shared volume index, wrapping -- the controller's , and . here too."""
        from fastfuncstuff.viewer.vocab import SetIndex

        st = self.session.state
        hi = st.max_time_index()
        if hi > 0:
            self._dispatch(SetIndex((st.time_index + delta) % (hi + 1)))

    def _zoom_by(self, factor: float) -> None:
        vp = self._viewport()
        if vp is not None:
            self._dispatch(SetZoom(self.vid, max(1.0, min(vp.zoom * factor, 16.0))))

    def _reset_view(self) -> None:
        """Back to the whole plane. One key, because a lost view is a dead end."""
        if self._viewport() is not None:
            self._dispatch(SetZoom(self.vid, 1.0))
            self._dispatch(SetPan(self.vid, 0.0, 0.0))

    def _pan_by(self, d_row: float, d_col: float) -> None:
        vp = self._viewport()
        if vp is not None:
            self._dispatch(SetPan(self.vid, vp.pan[0] + d_row, vp.pan[1] + d_col))

    def _step(self, delta: int) -> None:
        state = self.session.state
        vp = self._viewport()
        if state.grid is None or vp is None:
            return
        axis = plane_layout(state.grid.affine, vp.plane).fixed
        if vp.locked:
            ijk = list(state.crosshair)
            ijk[axis] += delta
            self._dispatch(SetIJK(*ijk))
            return
        here = vp.position if vp.position is not None else state.crosshair[axis]
        limit = state.grid.shape[axis] - 1
        self._dispatch(SetViewPosition(self.vid, max(0, min(here + delta, limit))))

    # -- output --------------------------------------------------------
    def apply(self, viewport: Viewport) -> None:
        """Push the viewport's settings into the widgets."""
        self.setWindowTitle(viewport.title)
        for plane, button in self._plane_buttons.items():
            button.setChecked(plane is viewport.plane)
        self.solo_button.setChecked(viewport.solo)
        self.lock_button.setChecked(viewport.locked)
        self.pane.plane = viewport.plane

    def restyle(self) -> None:
        """Re-read the palette after a theme switch."""
        self.setStyleSheet(theme.stylesheet())
        self.pane.update()

    def redraw(self) -> None:
        vp = self._viewport()
        if vp is None:
            return
        self.pane.set_pane(render_viewport(self.session, vp))
        self._sync_opacity()
        state = self.session.state
        if state.grid is not None:
            layout = plane_layout(state.grid.affine, vp.plane)
            extent = state.grid.shape[layout.fixed]
            pos = self.pane.position
            follow = "" if vp.locked else " parked"
            self.slice_label.setText(f"{'--' if pos is None else pos}/{extent - 1}{follow}")
        self.redraw_crosshair()

    def redraw_crosshair(self) -> None:
        state = self.session.state
        vp = self._viewport()
        if state.grid is None or vp is None:
            return
        layout = plane_layout(state.grid.affine, vp.plane)
        self.pane.set_layout(layout)
        view = plane_view(state, vp)
        if view is None:
            return
        row, col = view.to_image(state.crosshair)
        self.pane.set_crosshair(row, col)
        self.pane.set_zoomed(not view.is_identity)
        self.pane.set_coverage(self._graph_coverage(vp.plane, row, col))
        self.pane.set_readout(self.session.overlay_readout())

    def _graph_coverage(self, plane: Plane, row: int, col: int) -> list[tuple[int, int, int, int]]:
        """Footprints of the graphs reading this plane, in image indices.

        Only graphs on the *same* plane. An axial graph's 5x5 block is three
        voxels of one slice as far as a sagittal view is concerned, and drawing
        that as a box on sagittal would claim a coverage the graph does not
        have. The block walks from ``-half`` exactly as the graph does, so the
        square on screen is the voxels the cells are showing rather than an
        approximation of them.
        """
        out: list[tuple[int, int, int, int]] = []
        for graph in self.session.state.viewports.graphs:
            if graph.plane is not plane:
                continue
            half = graph.grid_n // 2
            out.append((row - half, col - half, graph.grid_n, graph.grid_n))
        return out

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self.closed.emit(self.vid)
        super().closeEvent(event)


__all__ = ["ImageWindow"]
