"""Sampling a display plane out of a layer, on GPU.

Every layer carries its own affine and is resampled into the shared display grid
at draw time. That is what let the go/no-go delete view spaces, warp-on-demand
and the three resample-mode menus: a slice resample measures about 0.08 ms, so
doing it every frame is cheaper than the machinery for avoiding it.

The same routine serves oblique views. An oblique plane is a display grid with a
rotated affine, not a fourth kind of plane, so nothing here needs to know the
difference -- which is why the viewer can do arbitrary-plane reslicing that AFNI
effectively cannot.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff.viewer.state import DisplayGrid, Plane

#: Which display axis each plane holds fixed, and which two it spans. Ordered
#: (fixed, rows, cols) in display i/j/k indices.
_PLANE_AXES: dict[Plane, tuple[int, int, int]] = {
    Plane.SAGITTAL: (0, 1, 2),
    Plane.CORONAL: (1, 0, 2),
    Plane.AXIAL: (2, 0, 1),
}


def plane_axes(plane: Plane) -> tuple[int, int, int]:
    """``(fixed, row, col)`` display axes for a plane."""
    return _PLANE_AXES[plane]


def plane_shape(grid: DisplayGrid, plane: Plane) -> tuple[int, int]:
    """Pixel dimensions of one display plane."""
    _, row, col = plane_axes(plane)
    return (grid.shape[row], grid.shape[col])


def plane_indices(
    grid: DisplayGrid,
    plane: Plane,
    position: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``(H, W, 3)`` display-grid indices covering one plane.

    Built on-device so a redraw never round-trips index arithmetic through the
    host.
    """
    fixed, row, col = plane_axes(plane)
    h, w = grid.shape[row], grid.shape[col]
    rr = torch.arange(h, device=device, dtype=dtype).unsqueeze(1).expand(h, w)
    cc = torch.arange(w, device=device, dtype=dtype).unsqueeze(0).expand(h, w)
    ff = torch.full((h, w), float(position), device=device, dtype=dtype)
    out = torch.empty((h, w, 3), device=device, dtype=dtype)
    out[..., fixed] = ff
    out[..., row] = rr
    out[..., col] = cc
    return out


def display_to_layer(
    display_ijk: Tensor, grid_affine: np.ndarray, layer_affine: np.ndarray
) -> Tensor:
    """Map display-grid indices to a layer's voxel indices.

    Composes ``inv(layer_affine) @ grid_affine`` once and applies it as a single
    matrix, rather than going through millimetres per voxel.
    """
    m = np.linalg.inv(np.asarray(layer_affine, float)) @ np.asarray(grid_affine, float)
    mat = torch.as_tensor(m, dtype=display_ijk.dtype, device=display_ijk.device)
    rot, shift = mat[:3, :3], mat[:3, 3]
    return display_ijk @ rot.T + shift


def sample_volume(
    volume: Tensor,
    layer_ijk: Tensor,
    *,
    mode: str = "bilinear",
    fill: float = 0.0,
) -> Tensor:
    """Sample a 3-D volume at fractional voxel indices.

    ``volume`` is ``(nx, ny, nz)`` and ``layer_ijk`` is ``(..., 3)`` in that same
    index order. Samples outside the volume come back as ``fill`` rather than
    edge-clamped, so a layer with a smaller field of view shows its actual
    extent instead of smearing its border across the display.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {tuple(volume.shape)}")
    nx, ny, nz = volume.shape
    out_shape = layer_ijk.shape[:-1]

    # grid_sample wants normalized [-1, 1] coordinates whose last axis is
    # ordered (w, h, d) -- the reverse of the volume's (nx, ny, nz). Getting
    # this backwards silently transposes the image, so it is spelled out.
    sizes = torch.tensor([nx, ny, nz], dtype=layer_ijk.dtype, device=layer_ijk.device)
    norm = 2.0 * layer_ijk / (sizes - 1).clamp(min=1) - 1.0
    grid = norm.flip(-1).reshape(1, 1, 1, -1, 3)

    src = volume.unsqueeze(0).unsqueeze(0)
    sampled = torch.nn.functional.grid_sample(
        src, grid, mode=mode, padding_mode="zeros", align_corners=True
    ).reshape(out_shape)

    if fill != 0.0:
        inside = ((layer_ijk >= 0) & (layer_ijk <= (sizes - 1))).all(dim=-1)
        sampled = torch.where(inside, sampled, torch.full_like(sampled, fill))
    return sampled


def extract_plane(
    volume: Tensor,
    grid: DisplayGrid,
    layer_affine: np.ndarray,
    plane: Plane,
    position: int,
    *,
    mode: str = "bilinear",
    fill: float = 0.0,
) -> Tensor:
    """One display plane of a layer, resampled into the display grid.

    Returns ``(H, W)``. When the layer already sits on the display grid this
    still goes through ``grid_sample``; at 0.08 ms the special case would cost
    more in divergent code paths than it saves.
    """
    ijk = plane_indices(grid, plane, position, device=volume.device, dtype=volume.dtype)
    layer_ijk = display_to_layer(ijk, grid.affine, layer_affine)
    return sample_volume(volume, layer_ijk, mode=mode, fill=fill)


def voxel_value(
    volume: Tensor, grid: DisplayGrid, layer_affine: np.ndarray, ijk: tuple[int, int, int]
) -> float | None:
    """The layer's value under a display-grid voxel, or ``None`` if outside it.

    Nearest-neighbour on purpose: a readout should report a voxel that exists in
    the data, not an interpolated value that no voxel holds.
    """
    point = torch.tensor([[float(v) for v in ijk]], dtype=torch.float32)
    layer_ijk = display_to_layer(point, grid.affine, layer_affine)
    idx = layer_ijk.round().to(torch.long)[0]
    nx, ny, nz = volume.shape
    i, j, k = (int(v) for v in idx)
    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
        return None
    return float(volume[i, j, k])
