"""Cortical surfaces in the viewer: meshes held beside the layer stack.

Surfaces are not layers. A layer is voxels on a grid with an affine; a surface
is a mesh whose vertices are already in scanner millimetres, and drawing it on
a slice is a geometric intersection rather than a resample. So the meshes live
here, on the session, and the state carries only the scalars a script needs
to rebuild them (which subject, which surfaces are drawn) -- the same split as
layers, whose voxels live in the store while the state holds the record.

Nothing here imports Qt. Outlines come out in the drawn image's fractional
(row, col), through :meth:`PlaneView.points_to_image`, so the flip and crop
arithmetic stays in the one place every other overlay already uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fastfuncstuff.io.freesurfer import Hemisphere, load_subject
from fastfuncstuff.surface.geometry import SliceIndex, apply_affine
from fastfuncstuff.viewer.slicing import PlaneView

#: Outline colours (RGB, 0-1). Yellow white and red pial follow freeview, which
#: is what anyone checking a recon has been looking at for years.
OUTLINE_RGB: dict[str, tuple[float, float, float]] = {
    "white": (1.0, 0.92, 0.0),
    "pial": (1.0, 0.15, 0.15),
    "smoothwm": (0.2, 0.85, 1.0),
}

#: The surfaces that are anatomically placed, and so can be drawn on a slice.
#: Inflated and sphere positions have no meaning in the scanner.
ANATOMICAL = ("white", "pial", "smoothwm")


@dataclass(frozen=True)
class Outline:
    hemi: str
    surface: str
    rgb: tuple[float, float, float]
    #: ``(N, 2, 2)`` segment endpoints as fractional (row, col) image pixels,
    #: pixel ``r`` centred at ``r + 0.5``.
    segments: np.ndarray


class SurfaceStore:
    """The loaded hemispheres, and slice indices cached per display grid."""

    def __init__(self) -> None:
        self.subject: Path | None = None
        self.hemis: dict[str, Hemisphere] = {}
        self._grid_key: bytes | None = None
        self._index: dict[tuple[str, str], SliceIndex] = {}

    def load(self, subject_dir: str | Path, hemis: tuple[str, ...] = ("lh", "rh")) -> None:
        loaded = load_subject(subject_dir, hemis)
        if not loaded:
            raise FileNotFoundError(f"no ?h.white surfaces under {subject_dir}/surf")
        self.subject = Path(subject_dir)
        self.hemis = loaded
        self._index.clear()

    def clear(self) -> None:
        self.subject = None
        self.hemis = {}
        self._index.clear()

    def _slice_index(self, hemi: str, surface: str, grid_affine: np.ndarray) -> SliceIndex | None:
        key = np.asarray(grid_affine, np.float64).tobytes()
        if key != self._grid_key:
            # A new display grid re-expresses every vertex; indices built in
            # the old one would cut the wrong slice.
            self._index.clear()
            self._grid_key = key
        found = self._index.get((hemi, surface))
        if found is None:
            h = self.hemis[hemi]
            if surface not in h.states:
                return None
            ijk = apply_affine(np.linalg.inv(grid_affine), h.states[surface])
            found = self._index[(hemi, surface)] = SliceIndex(ijk, h.faces)
        return found

    def moved(self, hemi: str, surface: str) -> None:
        """Say a surface's vertices changed, so its index is rebuilt on next use."""
        self._index.pop((hemi, surface), None)

    def outlines(
        self,
        grid_affine: np.ndarray,
        view: PlaneView,
        position: int,
        shown: tuple[str, ...],
    ) -> list[Outline]:
        """Where each shown surface crosses the slice at ``position``."""
        out: list[Outline] = []
        axis = view.layout.fixed
        for hemi in self.hemis:
            for surface in shown:
                if surface not in ANATOMICAL:
                    continue
                index = self._slice_index(hemi, surface, grid_affine)
                if index is None:
                    continue
                seg = index.segments(axis, float(position))
                if not len(seg):
                    continue
                rgb = OUTLINE_RGB.get(surface, (1.0, 1.0, 1.0))
                out.append(Outline(hemi, surface, rgb, view.points_to_image(seg)))
        return out


__all__ = ["ANATOMICAL", "OUTLINE_RGB", "Outline", "SurfaceStore"]
