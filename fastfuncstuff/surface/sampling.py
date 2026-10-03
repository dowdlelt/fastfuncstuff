"""Reading a volume at points in scanner millimetres.

Profiles along a vertex normal, depth samples between white and pial, the
intensity a snapping edit searches -- all are "the image at these mm
coordinates", so they share one sampler that owns the affine inversion.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


class VolumeSampler:
    """Trilinear reads of a 3-D volume at scanner-RAS points."""

    def __init__(self, data: np.ndarray, affine: np.ndarray) -> None:
        data = np.asarray(data)
        if data.ndim == 4:
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f"expected a 3-D volume, got shape {data.shape}")
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.affine = np.asarray(affine, np.float64)
        self._inverse = np.linalg.inv(self.affine)
        #: Mean voxel edge, mm: the natural unit for search steps and blur.
        self.voxel_mm = float(abs(np.linalg.det(self.affine[:3, :3])) ** (1 / 3))

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Values at ``(..., 3)`` mm points; edge-clamped outside the volume."""
        pts = np.asarray(points, np.float64)
        flat = pts.reshape(-1, 3)
        ijk = flat @ self._inverse[:3, :3].T + self._inverse[:3, 3]
        vals = map_coordinates(self.data, ijk.T, order=1, mode="nearest")
        return vals.reshape(pts.shape[:-1])


__all__ = ["VolumeSampler"]
