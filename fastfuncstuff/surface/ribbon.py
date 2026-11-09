"""The voxel <-> vertex map: which vertex owns each voxel of the cortical ribbon.

The projection reads a volume at points it assigns to vertices (footprints). This is
the same assignment run the other way, on a grid: every voxel between white and pial
belongs to the vertex whose footprint holds it. The owner is the heaviest corner of
the voxel's closest point on the midthickness, which is the rule
:func:`surface.projection.build_sampling` gives each read. One map then answers both
directions:

* **vertex values -> a volume** (:meth:`RibbonMap.paint`): a surface result drawn on
  slices, thresholded and clustered by the same controls as any volume;
* **a vertex set -> voxels** (:meth:`RibbonMap.voxels_of`): a surface cluster as a
  volume ROI, whose voxels carry a cortical depth (:attr:`RibbonMap.depth`, white 0 to
  pial 1), so it can feed a depth or laminar analysis.

Ribbon membership uses winding numbers (inside pial, not inside white), as
``ffs_util_surf2layers`` does, so the ribbon is hole-free where the surfaces are. By
default a voxel is in the ribbon when >= 10% of it is (``min_gm_fraction``, the same
fractional ribbon as surf2layers / laminar_tdv), its owner and depth measured at its
grey-matter part; None keeps the voxel-centre rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RibbonMap", "build_ribbon_map"]


@dataclass
class RibbonMap:
    """Ribbon voxels of one hemisphere on one grid, with their owner vertex and depth."""

    shape: tuple[int, int, int]
    affine: np.ndarray
    flat: np.ndarray  # (N,) int64 C-order indices into shape
    vertex: np.ndarray  # (N,) int64 owner vertex
    depth: np.ndarray  # (N,) float32, white 0 .. pial 1
    n_vertices: int
    gm_frac: np.ndarray | None = None  # (N,) float32 share in the ribbon (fractional rule)

    def paint(self, values: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Per-vertex ``values`` (V,) as an (X, Y, Z) float32 volume, 0 off the ribbon."""
        vol = np.zeros(self.shape, np.float32) if out is None else out
        vol.reshape(-1)[self.flat] = np.asarray(values, np.float32)[self.vertex]
        return vol

    def voxels_of(self, vertices: np.ndarray) -> np.ndarray:
        """Flat indices of the voxels owned by a vertex set (bool mask or ids)."""
        sel = np.asarray(vertices)
        if sel.dtype != bool:
            m = np.zeros(self.n_vertices, bool)
            m[sel] = True
            sel = m
        return self.flat[sel[self.vertex]]

    def vertex_counts(self) -> np.ndarray:
        """How many voxels each vertex owns (0 for a vertex finer than the grid)."""
        return np.bincount(self.vertex, minlength=self.n_vertices)


def build_ribbon_map(
    white: np.ndarray,
    pial: np.ndarray,
    faces: np.ndarray,
    affine: np.ndarray,
    shape: tuple[int, int, int],
    device=None,
    pad: int = 2,
    min_gm_fraction: float | None = 0.1,
) -> RibbonMap:
    """The ribbon map of one hemisphere on the grid ``(shape, affine)``.

    Work happens on the mesh's bounding box only (plus ``pad`` voxels): a hemisphere
    covers well under half of a whole-head grid.
    """
    from fastfuncstuff.surface.voxelize import MeshDistance, winding_number

    shape = tuple(int(s) for s in shape)
    aff = np.asarray(affine, np.float64)
    w = np.asarray(white, np.float64)
    p = np.asarray(pial, np.float64)
    inv = np.linalg.inv(aff)
    ijk = np.r_[w, p] @ inv[:3, :3].T + inv[:3, 3]
    lo = np.maximum(np.floor(ijk.min(0)).astype(int) - pad, 0)
    hi = np.minimum(np.ceil(ijk.max(0)).astype(int) + pad + 1, np.asarray(shape))
    if np.any(hi <= lo):  # the mesh is not on this grid at all
        empty = np.zeros(0, np.int64)
        return RibbonMap(shape, aff, empty, empty, np.zeros(0, np.float32), len(w))
    sub_shape = tuple(int(x) for x in hi - lo)
    sub_aff = aff.copy()
    sub_aff[:3, 3] = aff[:3, :3] @ lo + aff[:3, 3]
    f = np.asarray(faces, np.int64)
    gm_frac = None
    if min_gm_fraction is None:
        inside = winding_number(p, f, sub_aff, sub_shape, device) > 0
        inside &= ~(winding_number(w, f, sub_aff, sub_shape, device) > 0)
        local = np.argwhere(inside)
        at = local.astype(np.float64)
    else:
        import torch

        from fastfuncstuff.surface.volume_depth import RibbonSurfaces, _fractional_ribbon

        _, _, (kept, gm_frac, centroid), _ = _fractional_ribbon(
            [RibbonSurfaces("h", w, p, f)], sub_aff, sub_shape, min_gm_fraction, 4,
            device or torch.device("cpu"),
        )  # fmt: skip
        local = np.stack(np.unravel_index(kept, sub_shape), 1)
        at = local + centroid  # owner and depth at the voxel's grey-matter part
    pts = at @ sub_aff[:3, :3].T + sub_aff[:3, 3]
    # ~4 voxels a side, as volume_depth sizes it: the CUDA brick search, ignored elsewhere.
    brick_mm = 4.0 * float(np.abs(np.linalg.det(aff[:3, :3])) ** (1 / 3))
    mid = 0.5 * (w + p)
    near = MeshDistance(mid, f)(pts, device, brick_mm=brick_mm)
    owner = f[near.face, np.argmax(near.bary, axis=1)]
    dw = MeshDistance(w, f)(pts, device, brick_mm=brick_mm).distance
    dp = MeshDistance(p, f)(pts, device, brick_mm=brick_mm).distance
    depth = (dw / np.maximum(dw + dp, 1e-6)).astype(np.float32)
    g = local + lo
    flat = np.ravel_multi_index((g[:, 0], g[:, 1], g[:, 2]), shape).astype(np.int64)
    return RibbonMap(shape, aff, flat, owner.astype(np.int64), depth, len(w), gm_frac)
