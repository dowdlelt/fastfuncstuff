"""Viewer state: one display grid, a layer stack, and where you are looking.

Dropping AFNI's ``+orig``/``+acq``/``+tlrc`` view spaces means there is exactly
one grid on screen and every layer carries its own affine into it. Resampling a
slice costs about 0.08 ms on GPU, so the thing AFNI spent warp-on-demand and
three resample-mode menus to avoid is now cheaper than the menu would be.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fastfuncstuff.viewer.layers import LayerStack

# Plane lives in viewports.py because a viewport takes one as a field
# default and this module holds the viewports; re-exported here because
# `from viewer.state import Plane` is how the rest of the viewer says it.
from fastfuncstuff.viewer.viewports import Plane, ViewportSet


@dataclass(frozen=True)
class DisplayGrid:
    """The grid every layer is resampled into for drawing.

    ``affine`` maps display voxel indices to scanner millimetres (RAS), so a
    crosshair is a real anatomical location rather than an index into whichever
    dataset happened to load first.
    """

    shape: tuple[int, int, int]
    affine: np.ndarray
    #: For a window's tilted copy of the grid (see ``compose.view_grid``): the
    #: untilted affine, which decides which voxel axis a pane shows and which
    #: way it runs. Without it a 50-degree tilt could swap a pane's axes.
    layout_affine: np.ndarray | None = None

    @property
    def frame(self) -> np.ndarray:
        """The affine plane layouts are read from."""
        return self.affine if self.layout_affine is None else self.layout_affine

    @classmethod
    def from_layer(cls, shape: tuple[int, int, int], affine: np.ndarray) -> DisplayGrid:
        return cls(shape=tuple(int(s) for s in shape), affine=np.asarray(affine, float))

    def ijk_to_mm(self, ijk: tuple[float, float, float]) -> tuple[float, float, float]:
        v = self.affine @ np.array([*ijk, 1.0], dtype=float)
        return (float(v[0]), float(v[1]), float(v[2]))

    def mm_to_ijk(self, mm: tuple[float, float, float]) -> tuple[float, float, float]:
        v = np.linalg.inv(self.affine) @ np.array([*mm, 1.0], dtype=float)
        return (float(v[0]), float(v[1]), float(v[2]))

    def clamp(self, ijk: tuple[int, int, int]) -> tuple[int, int, int]:
        i, j, k = (int(max(0, min(int(v), n - 1))) for v, n in zip(ijk, self.shape, strict=True))
        return (i, j, k)

    def contains(self, ijk: tuple[int, int, int]) -> bool:
        return all(0 <= v < n for v, n in zip(ijk, self.shape, strict=True))


#: Longest axis a resized display grid may have. A plane is drawn per
#: frame, so this bounds pixels per pane, not memory: 2048 is already a
#: 0.125 mm grid across a 256 mm head.
MAX_GRID_AXIS = 2048


def resize_grid(
    shape: tuple[int, int, int], affine: np.ndarray, voxel_mm: float
) -> tuple[tuple[int, int, int], np.ndarray]:
    """The same field of view and orientation, with voxels of about ``voxel_mm``.

    Either direction: finer to draw a high-resolution overlay on its own
    voxels, coarser to see the anatomy the way a low-resolution run sees it.
    ``voxel_mm <= 0`` returns the grid unchanged. Each axis gets a whole number
    of voxels, so the size is matched to within one voxel's rounding, and the
    outer *edges* stay put (not the outer centres): the new grid covers
    exactly what the underlay covered and never straddles its border.
    """
    shape = tuple(int(n) for n in shape)
    affine = np.asarray(affine, dtype=float)
    if voxel_mm <= 0:
        return (shape[0], shape[1], shape[2]), affine
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    out_shape = [
        int(min(MAX_GRID_AXIS, max(1, round(n * s / voxel_mm))))
        for n, s in zip(shape, spacing, strict=True)
    ]
    scale = np.asarray(shape, dtype=float) / np.asarray(out_shape, dtype=float)
    out = affine.copy()
    out[:3, :3] = affine[:3, :3] * scale
    # Voxel centre 0 of the new grid sits half a new voxel inside the old edge.
    out[:3, 3] = affine[:3, 3] + affine[:3, :3] @ (0.5 * scale - 0.5)
    return (out_shape[0], out_shape[1], out_shape[2]), out


#: Round display voxel sizes offered after "underlay" and the layers' own.
GRID_SIZES = (3.0, 2.0, 1.5, 1.0, 0.8, 0.5, 0.4, 0.25)


def voxel_mm(affine: np.ndarray) -> float:
    """A grid's finest voxel edge, mm -- the size a slab's in-plane voxels are."""
    return float(np.linalg.norm(np.asarray(affine, dtype=float)[:3, :3], axis=0).min())


def grid_res_choices(state: ViewerState) -> list[tuple[str, float]]:
    """``(label, mm)`` for the GRID picker; ``mm == 0`` is the underlay's own grid.

    The underlay, then every other layer's own size by name (which of the
    two is finer swaps from dataset to dataset), then round sizes. The
    current setting is always present, so a replayed script's odd value is
    shown rather than silently mislabelled.
    """
    items: list[tuple[str, float]] = []
    seen: list[float] = []

    def offer(label: str, mm: float) -> None:
        if not any(abs(mm - have) <= 0.01 * max(mm, have) for have in seen):
            seen.append(mm)
            items.append((label, mm))

    base = state.layers.base
    if base is None:
        items.append(("underlay", 0.0))
    else:
        under = voxel_mm(base.affine)
        items.append((f"underlay ({under:.3g} mm)", 0.0))
        seen.append(under)
        for layer in list(state.layers)[1:]:
            mm = round(voxel_mm(layer.affine), 4)
            offer(f"match {layer.name} ({mm:.3g} mm)", mm)
        for mm in GRID_SIZES:
            offer(f"{mm:g} mm", mm)
    if all(abs(mm - state.grid_mm) > 1e-6 for _, mm in items):
        items.append((f"{state.grid_mm:g} mm", state.grid_mm))
    return items


@dataclass
class ViewerState:
    """Everything the UI draws from.

    Mutated only by command handlers -- see :mod:`fastfuncstuff.viewer.vocab`.
    """

    layers: LayerStack = field(default_factory=LayerStack)
    grid: DisplayGrid | None = None
    crosshair: tuple[int, int, int] = (0, 0, 0)
    time_index: int = 0
    #: The open image and graph windows. Zoom, pan and what a window is locked
    #: to live here rather than on the state, because none of them mean
    #: anything once there is more than one window.
    viewports: ViewportSet = field(default_factory=ViewportSet)
    #: Which layer the controls act on, and the one a soloed viewport draws.
    #: In state rather than in a list widget so that `[` and `]` are commands
    #: a script can replay, and so solo has something to be solo *of*.
    selected: str | None = None
    #: Which layer a mode reads. Deliberately its own field: what a mode
    #: computes *from* and what the screen shows are different questions, and
    #: tying them together is what forced the run to stay visible under its own
    #: correlation map. A layer can be unticked, buried at the bottom or
    #: unselected and still be the input -- it only has to be loaded.
    #: ``None`` means "whichever loaded layer the mode can use", resolved by
    #: :meth:`ViewerSession.input_layer` and never written back, so the answer
    #: keeps following the stack until someone names one.
    input_key: str | None = None
    #: Display voxel size, mm. 0 (the default) draws on the underlay's own
    #: voxels. Anything else resizes that grid, keeping its field of view and
    #: orientation: finer to show a high-resolution overlay on its own voxels
    #: with the underlay interpolated, coarser to see the anatomy at a run's
    #: resolution. Each layer's DRAW mode still decides how it is painted
    #: into whatever grid this gives. See :func:`resize_grid`.
    grid_mm: float = 0.0
    #: Seed voxel for InstaCorr, in display-grid indices. ``None`` until set.
    seed: tuple[int, int, int] | None = None
    #: Interface palette. State rather than a widget setting so a recorded
    #: session comes back looking the way it was recorded -- a screenshot from
    #: a replay should match the one that prompted it.
    theme: str = "light"
    #: FreeSurfer subject whose surfaces are loaded (the meshes themselves are
    #: on the session), and which of them are drawn as slice outlines.
    surface_subject: str | None = None
    surfaces_shown: tuple[str, ...] = ("white", "pial")
    #: Outline width in screen pixels. Fixed on screen, so a small window
    #: (few pixels per voxel) wants it thinner than a large one.
    surface_outline_width: float = 1.25
    #: Whether a press near an outline grabs it (instead of moving the
    #: crosshair), and the brush it is dragged with: radius mm, snap 0-1,
    #: smoothing, search mm either side, and the outward edge sign (-1 = T1).
    surface_editing: bool = False
    #: ``grab`` drags an outline; ``draw`` redraws a stretch of it.
    surface_tool: str = "grab"
    #: The vertex the point tool selected, as (hemi, vertex); ``None`` when none.
    #: Topology edits act on it and leave it on the vertex they produce.
    surface_selected: tuple[str, int] | None = None
    surface_brush: tuple[float, float, float, float, int] = (4.0, 1.0, 0.2, 1.5, -1)
    #: Snap only to edges at the boundary's expected intensity (see
    #: ``surface.edit.SnapParams.gate``); off snaps to the strongest edge.
    surface_snap_gate: bool = True
    #: Hand moves follow the drag, not the normals (``SnapParams.free``).
    surface_free: bool = False
    #: How far one nudge, or one press moving the marked cortex, goes, mm.
    surface_step: float = 0.25
    #: Layer an edit snaps to; ``None`` means the bottom of the stack, which is
    #: the anatomy the surfaces are being checked against.
    surface_snap_key: str | None = None
    #: Which FreeSurfer parcellation (``label/?h.<name>.annot``) and which
    #: label volume (``mri/<name>.mgz``) the readout names regions from.
    #: Empty turns that line off.
    surface_annot: str = "aparc"
    volume_atlas: str = "aparc+aseg"

    @property
    def crosshair_mm(self) -> tuple[float, float, float] | None:
        if self.grid is None:
            return None
        return self.grid.ijk_to_mm(self.crosshair)

    def selected_layer(self):
        """The layer the controls act on, defaulting to the top of the stack.

        The default is *adopted*, not merely reported. A selection that
        re-resolves to "whatever is on top" is not a selection: it moves when
        the stack is reordered, so two presses of the same key act on two
        different layers -- the second one lowers whatever the first one
        promoted past. Writing it down the first time it is asked for makes it
        stick, and it stays sticky until the layer is removed.
        """
        if self.selected is not None:
            found = self.layers.find(self.selected)
            if found is not None:
                return found
        top = self.layers.layers[-1] if len(self.layers) else None
        self.selected = top.key if top is not None else None
        return top

    def max_time_index(self) -> int:
        """Longest 4-D extent in the stack, so scrubbing spans everything."""
        if not len(self.layers):
            return 0
        return max(ly.n_volumes for ly in self.layers) - 1

    def adopt_grid(self, shape: tuple[int, int, int], affine: np.ndarray) -> None:
        """Set the display grid and centre the crosshair in it.

        Called when the first layer arrives. Later layers resample into this
        grid rather than replacing it, so loading a functional dataset over an
        anatomical one does not throw away the anatomy's resolution. The
        layer's grid is resized to :attr:`grid_mm` when that is set.
        """
        self.grid = DisplayGrid.from_layer(*resize_grid(shape, affine, self.grid_mm))
        i, j, k = (s // 2 for s in self.grid.shape)
        self.crosshair = (i, j, k)


__all__ = ["DisplayGrid", "Plane", "ViewerState", "grid_res_choices", "resize_grid", "voxel_mm"]
