"""Orientation-correct display planes through a 3-D grid.

A plane is described by the *grid* coordinates of each of its pixels rather than by a
slice of an array, so the same plane can be sampled out of anything defined on that
grid -- the image, an edge map, or a source image pulled through a warp -- without
ever reorienting a volume.

Display convention matches :func:`fastfuncstuff.termvis.orthoview` (and AFNI's
default): radiological, so subject right is on the left of the axial and coronal
panels; anterior is up on the axial, superior is up on the other two, and anterior
is to the right on the sagittal. Oblique grids are snapped to their nearest cardinal
axes, exactly as ``to_ras`` does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_VIEW_ALIASES = {
    "ax": "ax",
    "axi": "ax",
    "axial": "ax",
    "sag": "sag",
    "sagittal": "sag",
    "cor": "cor",
    "coronal": "cor",
}
VIEW_CHOICES = ("ax", "sag", "cor")


def parse_views(spec: str | Sequence[str]) -> tuple[str, ...]:
    """``"ax,sag,cor"`` (or a list) -> canonical view names, order kept, no repeats."""
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    views: list[str] = []
    for raw in items:
        key = raw.strip().lower()
        if not key:
            continue
        if key not in _VIEW_ALIASES:
            raise ValueError(f"unknown view {raw!r}; choose from {', '.join(VIEW_CHOICES)}")
        name = _VIEW_ALIASES[key]
        if name not in views:
            views.append(name)
    if not views:
        raise ValueError("at least one view is required")
    return tuple(views)


@dataclass(frozen=True)
class PlaneView:
    """One display panel: its pixel grid, physical pixel size and where it cuts."""

    name: str
    shape: tuple[int, int]
    """(rows, cols) in display order."""
    mm: tuple[float, float]
    """(mm per row, mm per col)."""
    ras_index: int
    """Index of the cut along the plane's normal, in RAS-ordered voxel indices."""
    left_letter: str
    """Anatomical direction at the panel's left edge (for a corner label)."""


@dataclass(frozen=True)
class SlicePlanes:
    """Display planes and the grid coordinates of every pixel in them."""

    views: tuple[PlaneView, ...]
    points: np.ndarray
    """(N, 3) float32 (z, y, x) array-index coordinates, views concatenated in order."""
    grid_shape: tuple[int, int, int]
    """(nz, ny, nx) of the grid the points index."""

    def split(self, values: np.ndarray) -> list[np.ndarray]:
        """(N, ...) per-point values -> one (rows, cols, ...) array per view."""
        out, start = [], 0
        for v in self.views:
            n = v.shape[0] * v.shape[1]
            out.append(values[start : start + n].reshape(*v.shape, *values.shape[1:]))
            start += n
        return out

    def flat_indices(self) -> np.ndarray:
        """(N,) int64 raveled indices into a (nz, ny, nx) array (points are integral)."""
        _, ny, nx = self.grid_shape
        p = np.rint(self.points).astype(np.int64)
        return (p[:, 0] * ny + p[:, 1]) * nx + p[:, 2]


def _orientation(affine: np.ndarray) -> np.ndarray:
    import nibabel as nib

    return nib.orientations.io_orientation(np.asarray(affine, dtype=np.float64))


def ras_shape_and_zooms(
    grid_shape: tuple[int, int, int], affine: np.ndarray
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    """Grid size and voxel size in RAS axis order, for a (nz, ny, nx) grid."""
    nz, ny, nx = grid_shape
    n_ijk = (nx, ny, nz)
    ornt = _orientation(affine)
    zooms = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    n_ras, z_ras = [0, 0, 0], [1.0, 1.0, 1.0]
    for a, (out, _) in enumerate(ornt):
        n_ras[int(out)] = n_ijk[a]
        z_ras[int(out)] = float(zooms[a]) or 1.0
    return (n_ras[0], n_ras[1], n_ras[2]), (z_ras[0], z_ras[1], z_ras[2])


def ijk_to_ras_index(ijk: Sequence[float], grid_shape: tuple[int, int, int], affine) -> np.ndarray:
    """Voxel (i, j, k) -> RAS-ordered voxel index (the frame ``ffs_info -vis_slice`` uses)."""
    nz, ny, nx = grid_shape
    n_ijk = (nx, ny, nz)
    out = np.zeros(3)
    for a, (o, flip) in enumerate(_orientation(affine)):
        out[int(o)] = ijk[a] if flip > 0 else n_ijk[a] - 1 - ijk[a]
    return out


def center_of_mass_ras(vol: np.ndarray, affine: np.ndarray) -> tuple[int, int, int]:
    """Intensity-weighted centre of a (nz, ny, nx) volume, as RAS voxel indices.

    Negative values are ignored, so a signed image still centres on its bright part.
    Falls back to the grid centre for an empty volume.
    """
    w = np.clip(np.nan_to_num(np.asarray(vol, dtype=np.float64)), 0, None)
    nz, ny, nx = w.shape
    total = w.sum()
    if total <= 0:
        kji = ((nz - 1) / 2, (ny - 1) / 2, (nx - 1) / 2)
    else:
        kji = tuple(
            float((w.sum(axis=tuple(d for d in range(3) if d != ax)) * np.arange(n)).sum() / total)
            for ax, n in enumerate(w.shape)
        )
    ras = ijk_to_ras_index((kji[2], kji[1], kji[0]), (nz, ny, nx), affine)
    return int(round(ras[0])), int(round(ras[1])), int(round(ras[2]))


def build_slice_planes(
    grid_shape: tuple[int, int, int],
    affine: np.ndarray,
    views: Sequence[str] = VIEW_CHOICES,
    center_ras: Sequence[int] | None = None,
) -> SlicePlanes:
    """Planes through a (nz, ny, nx) grid, cut at ``center_ras`` (RAS voxel indices).

    ``center_ras`` defaults to the grid centre; pass :func:`center_of_mass_ras` of the
    image to centre on the brain in a padded field of view.
    """
    nz, ny, nx = grid_shape
    n_ijk = (nx, ny, nz)
    ornt = _orientation(affine)
    (n_r, n_a, n_s), (z_r, z_a, z_s) = ras_shape_and_zooms(grid_shape, affine)
    if center_ras is None:
        center_ras = ((n_r - 1) // 2, (n_a - 1) // 2, (n_s - 1) // 2)
    r0, a0, s0 = (
        int(np.clip(c, 0, n - 1)) for c, n in zip(center_ras, (n_r, n_a, n_s), strict=True)
    )

    plane_views: list[PlaneView] = []
    chunks: list[np.ndarray] = []
    for name in parse_views(views):
        # Each branch lays out RAS index grids (R, A, S) in display order: rows top
        # to bottom, columns left to right.
        if name == "ax":
            rows, cols = np.meshgrid(np.arange(n_a), np.arange(n_r), indexing="ij")
            R, A, S = n_r - 1 - cols, n_a - 1 - rows, np.full_like(rows, s0)
            view = PlaneView(name, (n_a, n_r), (z_a, z_r), s0, "R")
        elif name == "cor":
            rows, cols = np.meshgrid(np.arange(n_s), np.arange(n_r), indexing="ij")
            R, A, S = n_r - 1 - cols, np.full_like(rows, a0), n_s - 1 - rows
            view = PlaneView(name, (n_s, n_r), (z_s, z_r), a0, "R")
        else:  # sag
            rows, cols = np.meshgrid(np.arange(n_s), np.arange(n_a), indexing="ij")
            R, A, S = np.full_like(rows, r0), cols, n_s - 1 - rows
            view = PlaneView(name, (n_s, n_a), (z_s, z_a), r0, "P")
        ras = (R.ravel(), A.ravel(), S.ravel())
        ijk = [np.zeros(R.size, dtype=np.float32) for _ in range(3)]
        for a, (o, flip) in enumerate(ornt):
            v = ras[int(o)]
            ijk[a] = (v if flip > 0 else n_ijk[a] - 1 - v).astype(np.float32)
        chunks.append(np.stack((ijk[2], ijk[1], ijk[0]), axis=1))
        plane_views.append(view)

    return SlicePlanes(tuple(plane_views), np.concatenate(chunks, axis=0), (nz, ny, nx))
