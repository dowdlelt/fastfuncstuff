"""Viewer state: one display grid, a layer stack, and where you are looking.

Dropping AFNI's ``+orig``/``+acq``/``+tlrc`` view spaces means there is exactly
one grid on screen and every layer carries its own affine into it. Resampling a
slice costs about 0.08 ms on GPU, so the thing AFNI spent warp-on-demand and
three resample-mode menus to avoid is now cheaper than the menu would be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from fastfuncstuff.viewer.layers import LayerStack


class Plane(StrEnum):
    """The three cardinal display planes.

    Oblique views are a display grid with a rotated affine rather than a fourth
    plane, so this stays closed.
    """

    AXIAL = "axial"
    SAGITTAL = "sagittal"
    CORONAL = "coronal"


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
class Locks:
    """What stays synchronised across panes.

    This replaces AFNI's cross-controller Lock system. In a single-window
    layout the interesting locks are between panes rather than between
    top-level windows, which makes them cheap to reason about.
    """

    crosshair: bool = True
    zoom: bool = True
    time: bool = True
    threshold: bool = False


@dataclass
class ViewerState:
    """Everything the UI draws from.

    Mutated only by command handlers -- see :mod:`fastfuncstuff.viewer.vocab`.
    """

    layers: LayerStack = field(default_factory=LayerStack)
    grid: DisplayGrid | None = None
    crosshair: tuple[int, int, int] = (0, 0, 0)
    time_index: int = 0
    zoom: float = 1.0
    pan: tuple[float, float] = (0.0, 0.0)
    locks: Locks = field(default_factory=Locks)
    #: Seed voxel for InstaCorr, in display-grid indices. ``None`` until set.
    seed: tuple[int, int, int] | None = None

    @property
    def crosshair_mm(self) -> tuple[float, float, float] | None:
        if self.grid is None:
            return None
        return self.grid.ijk_to_mm(self.crosshair)

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
