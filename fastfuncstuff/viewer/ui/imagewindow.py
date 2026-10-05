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

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.commands import Command
from fastfuncstuff.viewer.compose import plane_view, render_viewport
from fastfuncstuff.viewer.slicing import plane_layout
from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.surfaces import Grab
from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.panes import ImagePane
from fastfuncstuff.viewer.ui.shortcuts import Binding, ShortcutHelp, keep_keys_for_shortcuts
from fastfuncstuff.viewer.viewports import Viewport
from fastfuncstuff.viewer.vocab import (
    DeleteSurfaceVertex,
    EditSurface,
    EditSurfaceStroke,
    SelectSurfaceVertex,
    SetEdges,
    SetIJK,
    SetLayerOpacity,
    SetPan,
    SetSeed,
    SetSurfaceBrush,
    SetSurfaceEditing,
    SetSurfaceTool,
    SetViewLocked,
    SetViewPlane,
    SetViewPosition,
    SetViewSolo,
    SetXYZ,
    SetZoom,
    SplitSurfaceEdge,
    UndoSurfaceEdit,
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
    #: A review key asked the mode to do something: the action's name. Ignored
    #: by the controller in a mode that does not declare that action.
    action_requested = QtCore.Signal(str)
    #: A surface drag moved vertices without touching any image; every image
    #: window should redraw its outlines (and nothing else).
    surfaces_previewed = QtCore.Signal()

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

        self.edit_button = self._button(
            "EDIT",
            "g",
            "Edit surfaces: drag a white or pial outline toward where it should be\n"
            "and it snaps to the edge in the anatomy, over a brush in 3-D.\n"
            "( ) brush radius, ctrl+Z undo, Esc cancel a drag.",
        )
        self.edit_button.clicked.connect(self._toggle_grab)
        bar.addWidget(self.edit_button)

        self.draw_button = self._button(
            "DRAW",
            "G",
            "Redraw surfaces: press on a white or pial outline, draw where it\n"
            "should run, release on the same outline. The stretch between moves\n"
            "onto the line (m: snapped to the edge, or exactly as drawn) and the\n"
            "surface around follows, over the brush radius in 3-D.",
        )
        self.draw_button.clicked.connect(self._toggle_draw)
        bar.addWidget(self.draw_button)

        self.point_button = self._button(
            "POINT",
            "p",
            "Select a surface vertex (press near an outline): Delete removes it\n"
            "from every surface and file of the hemisphere, i splits its longest\n"
            "edge, shift+I splits all its edges -- for more triangles where a\n"
            "fold needs them.",
        )
        self.point_button.clicked.connect(self._toggle_point)
        bar.addWidget(self.point_button)

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
        self.pane.zoomed.connect(self._zoom_by)
        self.pane.slid.connect(self._slide)
        self.pane.turned.connect(self._turn)
        self.pane.edit_pressed.connect(self._edit_press)
        self.pane.edit_dragged.connect(self._edit_drag)
        self.pane.edit_released.connect(self._edit_release)
        #: A stroke being drawn: the outline it started on, its points in mm,
        #: and the same points in image pixels for drawing it.
        self._stroke: tuple[Grab, list[np.ndarray], list[tuple[float, float]]] | None = None
        #: (grab, press mm) of the drag in progress, if this window started one.
        self._edit: tuple[Grab, np.ndarray] | None = None
        self._edit_drag_mm: np.ndarray | None = None
        v.addWidget(self.pane, 1)
        # Two thirds of what it used to be. An EPI slice is 64 to 100 voxels
        # across, so a 420-pixel window was showing it at four times its own
        # resolution and spending most of the screen on interpolation -- and
        # three of them left nowhere to put a graph or an HRF curve.
        self.resize(280, 280)
        self.setMinimumSize(64, 64)

        def ask(name: str) -> Callable[[], None]:
            return lambda: self.action_requested.emit(name)

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
                Binding("c", "centre the view on the crosshair", self._centre_view, group="view"),
                Binding(
                    "t",
                    "tilt the slice to cut the cortex square-on here",
                    self._tilt_to_cortex,
                    group="view",
                ),
                Binding("shift+t", "untilt the slice", self._untilt, group="view"),
                Binding(
                    "alt+Up",
                    "tilt the slice 5 deg (top away)",
                    lambda: self._tilt_by(0, 5.0),
                    group="view",
                ),
                Binding(
                    "alt+Down",
                    "tilt the slice 5 deg (top toward)",
                    lambda: self._tilt_by(0, -5.0),
                    group="view",
                ),
                Binding(
                    "alt+Left",
                    "tilt the slice 5 deg (left away)",
                    lambda: self._tilt_by(1, -5.0),
                    group="view",
                ),
                Binding(
                    "alt+Right",
                    "tilt the slice 5 deg (right away)",
                    lambda: self._tilt_by(1, 5.0),
                    group="view",
                ),
                Binding(
                    "shift+o",
                    "surface outlines: both, white, pial, off",
                    self._cycle_outlines,
                    group="surface",
                ),
                Binding(
                    "[",
                    "thinner surface outlines",
                    lambda: self._outline_width_by(1 / 1.4),
                    group="surface",
                ),
                Binding(
                    "]",
                    "thicker surface outlines",
                    lambda: self._outline_width_by(1.4),
                    group="surface",
                ),
                Binding("right-drag", "zoom (up = in)", None, group="view"),
                Binding("middle-drag", "pan (or shift+drag)", None, group="view"),
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
                Binding(
                    "g", "edit surfaces: grab an outline", self.edit_button.click, group="surface"
                ),
                Binding(
                    "shift+g",
                    "edit surfaces: draw a stretch of outline anew",
                    self.draw_button.click,
                    group="surface",
                ),
                Binding(
                    "p", "edit surfaces: select a vertex", self.point_button.click, group="surface"
                ),
                Binding(
                    "Delete",
                    "delete the selected vertex",
                    self._delete_selected,
                    group="surface",
                    aliases=("Backspace",),
                ),
                Binding(
                    "i",
                    "split the selected vertex's longest edge",
                    lambda: self._split_selected(False),
                    group="surface",
                ),
                Binding(
                    "shift+i",
                    "split all the selected vertex's edges",
                    lambda: self._split_selected(True),
                    group="surface",
                ),
                Binding("(", "smaller brush", lambda: self._scale_brush(1 / 1.25), group="surface"),
                Binding(")", "larger brush", lambda: self._scale_brush(1.25), group="surface"),
                Binding(
                    "m",
                    "snap -> edge (strongest, ungated) -> hand",
                    self._toggle_snap,
                    group="surface",
                ),
                Binding("ctrl+z", "undo the last surface edit", self._undo_edit, group="surface"),
                Binding("Escape", "cancel the drag", self._cancel_edit, group="surface"),
                Binding("drag the ring", "turn the moving image", None, group="align mode"),
                Binding(
                    "drag the centre",
                    "slide it (or shift-drag anywhere)",
                    None,
                    group="align mode",
                ),
                # The review loop (ICA, InstaPCA) is judged on the map, so its
                # keys belong here as well as in the trace windows. Left/Right
                # move the crosshair here, hence < and > for stepping.
                Binding(">", "next component", ask("next"), group="review"),
                Binding("<", "previous component", ask("prev"), group="review"),
                Binding("s", "label signal, then next", ask("signal"), group="review"),
                Binding("n", "label noise, then next", ask("noise"), group="review"),
                Binding("u", "clear the label", ask("unlabel"), group="review"),
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

    # -- align mode ----------------------------------------------------
    def _align_mode(self):
        """The active align mode, when it has an image to move."""
        mode = self.session.mode
        if mode.name != "align" or mode.moving() is None:
            return None
        return mode

    def _screen_axes(self) -> tuple[np.ndarray, np.ndarray] | None:
        """World millimetres per image pixel down the rows and along the columns.

        Through the plane's layout, flips included, so a drag moves the image
        the way the hand moved on screen whichever way the grid is stored.
        """
        state = self.session.state
        vp = self._viewport()
        if state.grid is None or vp is None:
            return None
        layout = plane_layout(state.grid.affine, vp.plane)
        linear = np.asarray(state.grid.affine, dtype=float)[:3, :3]
        down = linear[:, layout.row] * (-1.0 if layout.row_flip else 1.0)
        right = linear[:, layout.col] * (-1.0 if layout.col_flip else 1.0)
        return down, right

    def _slide(self, d_row: float, d_col: float) -> None:
        mode, axes = self._align_mode(), self._screen_axes()
        if mode is None or axes is None:
            return
        down, right = axes
        command = mode.move_to(mode.shifted(d_row * down + d_col * right))
        if command is not None:
            self._dispatch(command)

    def _turn(self, degrees: float) -> None:
        """Clockwise on screen: about ``right x down``, which points at the viewer."""
        mode, axes = self._align_mode(), self._screen_axes()
        if mode is None or axes is None:
            return
        down, right = axes
        command = mode.move_to(mode.turned(np.cross(right, down), degrees))
        if command is not None:
            self._dispatch(command)

    def _handle_position(self) -> tuple[float, float] | None:
        """The pivot, projected into this window's image, in (row, col) pixels."""
        mode = self._align_mode()
        state = self.session.state
        vp = self._viewport()
        if mode is None or state.grid is None or vp is None:
            return None
        centre = mode.pivot_mm()
        view = plane_view(state, vp)
        if centre is None or view is None:
            return None
        ijk = np.linalg.inv(state.grid.affine) @ np.append(centre, 1.0)
        layout = view.layout
        row, col = float(ijk[layout.row]), float(ijk[layout.col])
        if layout.row_flip:
            row = state.grid.shape[layout.row] - 1 - row
        if layout.col_flip:
            col = state.grid.shape[layout.col] - 1 - col
        r0, c0 = view.origin
        return (row - r0, col - c0)

    def _viewport(self) -> Viewport | None:
        return self.session.state.viewports.find(self.vid)

    def _grid(self):
        """The grid this window samples through -- tilted when the window is oblique."""
        from fastfuncstuff.viewer.compose import view_grid

        return view_grid(self.session.state, self._viewport())

    def _tilted(self) -> bool:
        grid = self._grid()
        return grid is not None and grid.layout_affine is not None

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
        grid = self._grid()
        if grid is not None and grid.layout_affine is not None:
            # Oblique: the tilted voxel is somewhere in the shared grid, not at
            # these indices; go through mm and take the voxel it lands in.
            self._dispatch(SetXYZ(*grid.ijk_to_mm(tuple(float(v) for v in ijk))))
            if seed:
                self._dispatch(SetSeed(*self.session.state.crosshair))
            return
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

    def _cycle_outlines(self) -> None:
        from fastfuncstuff.viewer.surfaces import next_outlines
        from fastfuncstuff.viewer.vocab import ShowSurfaces

        nxt = next_outlines(self.session.state.surfaces_shown)
        self._dispatch(ShowSurfaces(nxt))
        self.pane.show_toast(f"outlines: {nxt or 'off'}")

    def _outline_width_by(self, factor: float) -> None:
        from fastfuncstuff.viewer.vocab import SetOutlineWidth

        self._dispatch(SetOutlineWidth(self.session.state.surface_outline_width * factor))

    def _tilt_to_cortex(self) -> None:
        """Oblique window: turn the slice to contain the cortex's normal at the crosshair.

        Through a sulcal wall that runs at a slant to the slice, the ribbon is
        a smear and dragging an outline moves the surface mostly out of the
        slice. Cut square-on, the boundary is sharp and the drag means what it
        shows. Pressed again after moving, it re-aims for the new spot.
        """
        from fastfuncstuff.viewer.compose import section_tilt, tilt_matrix
        from fastfuncstuff.viewer.vocab import SetViewTilt

        state = self.session.state
        vp = self._viewport()
        mm = state.crosshair_mm
        if vp is None or state.grid is None or mm is None:
            return
        n = self.session.surfaces.cortex_normal(mm)
        if n is None:
            self.pane.show_toast("no cortex within 6 mm of the crosshair to tilt to")
            return
        fixed = plane_layout(state.grid.affine, vp.plane).fixed
        plane_normal = np.linalg.inv(state.grid.affine)[fixed, :3]
        now = tilt_matrix(vp)
        a = now @ (plane_normal / np.linalg.norm(plane_normal))
        if abs(float(a @ n)) < 0.05:
            self.pane.show_toast("this slice already cuts the cortex square-on here")
            return
        r = section_tilt(now, plane_normal, n)
        if np.allclose(r, now):
            self.pane.show_toast("the cortex lies in this slice here: try another plane")
            return
        self._dispatch(SetViewTilt(self.vid, tuple(float(x) for x in r.ravel())))
        angle = np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
        self.pane.show_toast(f"tilted {angle:.0f} deg to cut the cortex square-on (T: untilt)")

    def _tilt_by(self, about: int, degrees: float) -> None:
        """Tilt the slice about the pane's horizontal (0) or vertical (1) axis, by hand."""
        from fastfuncstuff.viewer.align import axis_rotation
        from fastfuncstuff.viewer.compose import tilt_matrix
        from fastfuncstuff.viewer.vocab import SetViewTilt

        state = self.session.state
        vp = self._viewport()
        grid = self._grid()
        if vp is None or grid is None or state.grid is None:
            return
        layout = plane_layout(state.grid.affine, vp.plane)
        # The pane's horizontal runs along its columns, its vertical along rows.
        axis = grid.affine[:3, layout.col if about == 0 else layout.row]
        r = axis_rotation(axis, degrees) @ tilt_matrix(vp)
        self._dispatch(SetViewTilt(self.vid, tuple(float(x) for x in r.ravel())))

    def _untilt(self) -> None:
        from fastfuncstuff.viewer.vocab import SetViewTilt

        self._dispatch(SetViewTilt(self.vid))

    def _centre_view(self) -> None:
        state = self.session.state
        vp = self._viewport()
        view = None if vp is None else plane_view(state, vp)
        if view is not None:
            self._dispatch(SetPan(self.vid, *view.pan_centring(state.crosshair)))

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
        grid = self._grid()
        if grid is not None and grid.layout_affine is not None:
            # Oblique: step along the tilted slice's own normal.
            mm = np.asarray(state.crosshair_mm) + delta * grid.affine[:3, axis]
            self._dispatch(SetXYZ(float(mm[0]), float(mm[1]), float(mm[2])))
            return
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
        self.edit_button.setChecked(self.session.state.surface_editing)
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
        self._redraw_outlines(vp)
        self.edit_button.setChecked(state.surface_editing and state.surface_tool == "grab")
        self.draw_button.setChecked(state.surface_editing and state.surface_tool == "draw")
        self.point_button.setChecked(state.surface_editing and state.surface_tool == "point")
        self.pane.set_marks(self._selected_marks())
        self._sync_brush()
        if state.grid is not None:
            layout = plane_layout(state.grid.affine, vp.plane)
            extent = state.grid.shape[layout.fixed]
            pos = self.pane.position
            follow = "" if vp.locked else " parked"
            self.slice_label.setText(f"{'--' if pos is None else pos}/{extent - 1}{follow}")
        self.redraw_crosshair()

    # -- surface editing ----------------------------------------------
    def _brush_px(self) -> float | None:
        """Brush radius in image pixels, or ``None`` when not editing."""
        state = self.session.state
        if not state.surface_editing or state.grid is None or not self.session.surfaces.hemis:
            return None
        vp = self._viewport()
        if vp is None:
            return None
        layout = plane_layout(state.grid.affine, vp.plane)
        # In-plane voxel size: the pane draws one display voxel per pixel.
        cols = np.linalg.norm(state.grid.affine[:3, [layout.row, layout.col]], axis=0)
        return float(state.surface_brush[0] / cols.mean())

    def _scale_brush(self, factor: float) -> None:
        r, snap, smooth, search, sign = self.session.state.surface_brush
        r = float(np.clip(r * factor, 0.5, 30.0))
        self._dispatch(SetSurfaceBrush(round(r, 2), snap, smooth, search, sign))
        self._sync_brush()

    def _toggle_snap(self) -> None:
        """Cycle snap -> edge -> hand.

        Steps rather than a slider because the uses are distinct: snap when
        the image shows the boundary at the intensity it should be; edge when
        it shows a boundary the tissue estimate misjudges (pial lying in
        dura, a stripped brain's outer edge) -- the strongest edge wins; hand
        when it does not show one (a vessel, a dura fold) and the eye knows
        better than the gradient.
        """
        from fastfuncstuff.viewer.vocab import SetSurfaceSnapGate

        state = self.session.state
        r, snap, smooth, search, sign = state.surface_brush
        if snap > 0 and state.surface_snap_gate:
            self._dispatch(SetSurfaceSnapGate(False))
        elif snap > 0:
            self._dispatch(SetSurfaceBrush(r, 0.0, smooth, search, sign))
        else:
            self._dispatch(SetSurfaceBrush(r, 1.0, smooth, search, sign))
            self._dispatch(SetSurfaceSnapGate(True))
        self._sync_brush()

    def _sync_brush(self, note: str = "") -> None:
        state = self.session.state
        r, snap, *_ = state.surface_brush
        mode = "snap" if snap >= 1 else ("hand" if snap <= 0 else f"snap {snap:.0%}")
        if snap > 0 and not state.surface_snap_gate:
            mode = mode.replace("snap", "edge")
        tool = {"draw": "DRAW", "point": "POINT"}.get(state.surface_tool, "EDIT")
        if state.surface_tool == "point" and state.surface_selected is not None:
            hemi, v = state.surface_selected
            tool += f" {hemi} #{v}"
        self.pane.set_brush(self._brush_px(), f"{tool}  r={r:g} mm  {mode}")
        if note:
            self.pane.show_toast(note)

    def _toggle_grab(self) -> None:
        state = self.session.state
        on = not (state.surface_editing and state.surface_tool == "grab")
        self._dispatch(SetSurfaceTool("grab"))
        self._dispatch(SetSurfaceEditing(on))

    def _toggle_point(self) -> None:
        state = self.session.state
        on = not (state.surface_editing and state.surface_tool == "point")
        self._dispatch(SetSurfaceTool("point" if on else "grab"))
        self._dispatch(SetSurfaceEditing(on))

    def _topology(self, command) -> None:
        try:
            self._dispatch(command)
        except ValueError as exc:
            self._sync_brush(str(exc))
            return
        self._goto_selected()

    def _delete_selected(self) -> None:
        sel = self.session.state.surface_selected
        if sel is not None:
            self._topology(DeleteSurfaceVertex(sel[0], sel[1]))

    def _split_selected(self, all_edges: bool) -> None:
        sel = self.session.state.surface_selected
        if sel is None:
            return
        hemi, v = sel
        surfaces = self.session.surfaces
        if not all_edges:
            self._topology(SplitSurfaceEdge(hemi, *surfaces.longest_edge(hemi, v)))
            return
        # Splits append vertices, so v and its old neighbours keep their numbers.
        for u in surfaces.neighbours(hemi, v):
            self._topology(SplitSurfaceEdge(hemi, v, int(u)))
        self._dispatch(SelectSurfaceVertex(hemi, v))

    def _goto_selected(self) -> None:
        """Put the crosshair on the selected vertex, so every view finds it."""
        sel = self.session.state.surface_selected
        if sel is None:
            return
        h = self.session.surfaces.hemis[sel[0]]
        p = 0.5 * (h.states["white"][sel[1]] + h.states["pial"][sel[1]])
        self._dispatch(SetXYZ(float(p[0]), float(p[1]), float(p[2])))

    def _selected_marks(self) -> list[tuple[float, float, str]]:
        """Where the selected vertex sits on this slice, per surface, if it is close."""
        state = self.session.state
        sel = state.surface_selected
        vp = self._viewport()
        view = None if vp is None else plane_view(state, vp)
        pos = self.pane.position
        if sel is None or view is None or pos is None or state.grid is None:
            return []
        h = self.session.surfaces.hemis.get(sel[0])
        if h is None or sel[1] >= h.n_vertices:
            return []
        grid = self._grid()
        assert grid is not None
        inv = np.linalg.inv(grid.affine)
        marks = []
        for surface in ("white", "pial"):
            ijk = inv[:3, :3] @ h.states[surface][sel[1]] + inv[:3, 3]
            if abs(ijk[view.layout.fixed] - pos) <= 1.5:
                row, col = view.points_to_image(ijk)
                marks.append((float(row), float(col), surface))
        return marks

    def _toggle_draw(self) -> None:
        state = self.session.state
        on = not (state.surface_editing and state.surface_tool == "draw")
        self._dispatch(SetSurfaceTool("draw" if on else "grab"))
        self._dispatch(SetSurfaceEditing(on))

    def _press_point_mm(self, row: float, col: float) -> np.ndarray | None:
        state = self.session.state
        vp = self._viewport()
        view = None if vp is None else plane_view(state, vp)
        pos = self.pane.position
        if view is None or pos is None or state.grid is None:
            return None
        ijk = view.image_to_points(row, col, pos)
        affine = self._grid().affine
        return affine[:3, :3] @ ijk + affine[:3, 3]

    def _edit_press(self, row: float, col: float) -> None:
        from fastfuncstuff.surface.edit import SnapParams

        state = self.session.state
        surfaces = self.session.surfaces
        vp = self._viewport()
        view = None if vp is None else plane_view(state, vp)
        pos = self.pane.position
        if view is None or pos is None or state.grid is None:
            return
        # Tolerance in image pixels from a screen distance, so grabbing feels
        # the same at every zoom.
        tolerance = 8.0 / self.pane._image_scale()
        grab = surfaces.grab(
            self._grid().affine, view, pos, state.surfaces_shown, row, col, tolerance
        )
        if grab is None:
            # Not near an outline: the press still means "look here".
            self._pick(int(round(row)), int(round(col)), seed=False)
            return
        if state.surface_tool == "point":
            self._dispatch(SelectSurfaceVertex(grab.hemi, grab.vertex))
            self._goto_selected()
            return
        if state.surface_tool == "draw":
            # A stroke starts on the outline it was pressed on.
            start = np.asarray(grab.at_mm, np.float64)
            self._stroke = (grab, [start], [(row, col)])
            self.pane.set_stroke([(row, col)])
            self._sync_brush()
            return
        r, snap, smooth, search, sign = state.surface_brush
        params = SnapParams(
            radius=r,
            snap=snap,
            smooth=smooth,
            search=search,
            edge_sign=sign,
            gate=state.surface_snap_gate,
        )
        try:
            sampler = self.session.surface_sampler(state.surface_snap_key)
        except (ValueError, KeyError) as exc:
            self.pane.show_toast(str(exc))
            return
        surfaces.begin(grab, sampler, params)
        self._edit = (grab, np.asarray(grab.at_mm))
        self._edit_drag_mm = np.zeros(3)

    def _edit_drag(self, row: float, col: float) -> None:
        if self._stroke is not None:
            here = self._press_point_mm(row, col)
            if here is not None:
                self._stroke[1].append(here)
                self._stroke[2].append((row, col))
                self.pane.set_stroke(self._stroke[2])
            return
        if self._edit is None:
            return
        here = self._press_point_mm(row, col)
        if here is None:
            return
        self._edit_drag_mm = here - self._edit[1]
        res = self.session.surfaces.preview(self._edit_drag_mm)
        self.surfaces_previewed.emit()
        if res is not None:
            from fastfuncstuff.surface.edit import explain

            # Said while dragging, so the hand can respond: let go and
            # switch mode, or move the other surface first.
            note = explain(res)
            if note:
                self.pane.show_toast(note)

    def _edit_release(self) -> None:
        if self._stroke is not None:
            self._finish_stroke()
            return
        if self._edit is None:
            return
        grab, _ = self._edit
        drag = self._edit_drag_mm if self._edit_drag_mm is not None else np.zeros(3)
        self._edit = None
        self._edit_drag_mm = None
        if not np.any(drag):
            self.session.surfaces.cancel()
            return
        r, snap, smooth, search, sign = self.session.state.surface_brush
        self._dispatch(
            EditSurface(
                grab.hemi,
                grab.surface,
                grab.vertex,
                grab.at_mm,
                (float(drag[0]), float(drag[1]), float(drag[2])),
                r,
                snap,
                smooth,
                search,
                sign,
                self.session.state.surface_snap_key or "",
                self.session.state.surface_snap_gate,
            )
        )

    def _finish_stroke(self) -> None:
        """End a stroke: it must land on the outline it started on."""
        assert self._stroke is not None
        start, points, pixels = self._stroke
        self._stroke = None
        self.pane.set_stroke([])
        state = self.session.state
        vp = self._viewport()
        view = None if vp is None else plane_view(state, vp)
        pos = self.pane.position
        if view is None or pos is None or state.grid is None or len(points) < 3:
            return
        row, col = pixels[-1]
        end = self.session.surfaces.grab(
            self._grid().affine,
            view,
            pos,
            state.surfaces_shown,
            row,
            col,
            8.0 / self.pane._image_scale(),
        )
        if end is None or (end.hemi, end.surface) != (start.hemi, start.surface):
            self._sync_brush(f"end the stroke on the {start.surface} outline it began on")
            return
        # Anchor both ends on the outline itself, so the redrawn stretch meets
        # the untouched surface rather than wherever the hand let go.
        line = [np.asarray(start.at_mm), *points[1:-1], np.asarray(end.at_mm)]
        r, snap, smooth, search, sign = state.surface_brush
        try:
            self._dispatch(
                EditSurfaceStroke(
                    start.hemi,
                    start.surface,
                    int(view.layout.fixed),
                    float(pos),
                    EditSurfaceStroke.encode(line),
                    r,
                    snap,
                    smooth,
                    search,
                    sign,
                    state.surface_snap_key or "",
                    state.surface_snap_gate,
                    EditSurfaceStroke.encode_grid(self._grid().affine) if self._tilted() else "",
                )
            )
        except ValueError as exc:
            self._sync_brush(str(exc))
            return
        from fastfuncstuff.surface.edit import explain

        res = getattr(self.session.surfaces, "last_result", None)
        self._sync_brush(explain(res) if res is not None else "")

    def _cancel_edit(self) -> None:
        if self._stroke is not None:
            self._stroke = None
            self.pane.set_stroke([])
        if self._edit is not None:
            self._edit = None
            self._edit_drag_mm = None
            self.session.surfaces.cancel()
            self.surfaces_previewed.emit()

    def _undo_edit(self) -> None:
        self._dispatch(UndoSurfaceEdit())

    def redraw_outlines(self) -> None:
        """Only the outlines a drag can move -- the image is unchanged."""
        vp = self._viewport()
        if vp is not None:
            self._redraw_outlines(vp, only=self.session.surfaces.editing_keys)

    def _redraw_outlines(self, vp: Viewport, only: set[tuple[str, str]] | None = None) -> None:
        state = self.session.state
        surfaces = self.session.surfaces
        view = plane_view(state, vp)
        pos = self.pane.position
        self.pane.set_outline_width(state.surface_outline_width)
        if not surfaces.hemis or not state.surfaces_shown or view is None or pos is None:
            self.pane.set_outlines([])
            return
        assert state.grid is not None
        self.pane.set_outlines(
            surfaces.outlines(self._grid().affine, view, pos, state.surfaces_shown, only=only),
            only=only,
        )

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
        from fastfuncstuff.viewer.compose import tilt_matrix

        r = tilt_matrix(vp)
        self.pane.set_tilt(float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))))
        self.pane.set_coverage(self._graph_coverage(vp.plane, row, col))
        self.pane.set_readout(self.session.overlay_readout())
        self.pane.set_handle(self._handle_position())

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
