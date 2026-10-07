"""Reading a volume at points in scanner millimetres.

Profiles along a vertex normal, depth samples between white and pial, the
intensity a snapping edit searches -- all are "the image at these mm
coordinates", so they share one sampler that owns the affine inversion.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

#: How a sampler reads between voxels. ``linear`` is what edits snap with.
MODES = ("nearest", "linear", "cubic")


class VolumeSampler:
    """Reads of a 3-D volume at scanner-RAS points: nearest, trilinear or tricubic."""

    def __init__(self, data: np.ndarray, affine: np.ndarray, mode: str = "linear") -> None:
        data = np.asarray(data)
        if data.ndim == 4:
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f"expected a 3-D volume, got shape {data.shape}")
        if mode not in MODES:
            raise ValueError(f"sampling is one of {', '.join(MODES)}, not {mode!r}")
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.affine = np.asarray(affine, np.float64)
        self.mode = mode
        self._inverse = np.linalg.inv(self.affine)
        #: Mean voxel edge, mm: the natural unit for search steps and blur.
        self.voxel_mm = float(abs(np.linalg.det(self.affine[:3, :3])) ** (1 / 3))

    def with_mode(self, mode: str) -> VolumeSampler:
        """The same volume read another way; shares the voxels, copies nothing."""
        if mode == self.mode:
            return self
        if mode not in MODES:
            raise ValueError(f"sampling is one of {', '.join(MODES)}, not {mode!r}")
        other = object.__new__(VolumeSampler)
        other.__dict__.update(self.__dict__)
        other.mode = mode
        return other

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Values at ``(..., 3)`` mm points; edge-clamped outside the volume."""
        pts = np.asarray(points, np.float64)
        flat = pts.reshape(-1, 3)
        ijk = flat @ self._inverse[:3, :3].T + self._inverse[:3, 3]
        if self.mode == "cubic":
            # Catmull-Rom, the viewer's own cubic: it interpolates, so a voxel
            # centre reads that voxel in every mode and only the edges change.
            import torch

            from fastfuncstuff.viewer.slicing import _sample_cubic

            hi = np.asarray(self.data.shape) - 1
            clamped = np.clip(ijk, 0, hi)
            vals = _sample_cubic(torch.from_numpy(self.data), torch.from_numpy(clamped)).numpy()
        else:
            order = 0 if self.mode == "nearest" else 1
            vals = map_coordinates(self.data, ijk.T, order=order, mode="nearest")
        return vals.reshape(pts.shape[:-1])


__all__ = ["MODES", "VolumeSampler"]
