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
    #: Seed voxel for InstaCorr, in display-grid indices. ``None`` until set.
    seed: tuple[int, int, int] | None = None
    #: Interface palette. State rather than a widget setting so a recorded
    #: session comes back looking the way it was recorded -- a screenshot from
    #: a replay should match the one that prompted it.
    theme: str = "light"

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
        anatomical one does not throw away the anatomy's resolution.
        """
        self.grid = DisplayGrid.from_layer(shape, affine)
        i, j, k = (s // 2 for s in self.grid.shape)
        self.crosshair = (i, j, k)


__all__ = ["DisplayGrid", "Plane", "ViewerState"]
