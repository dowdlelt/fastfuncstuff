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

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff.viewer.state import DisplayGrid, Plane

#: Which RAS direction each plane is normal to, and how the two in-plane
#: directions should be laid out. ``up`` is the anatomical direction that must
#: point to the top of the image, ``right`` the one that must point to its
#: right edge.
#:
#: Axial and coronal both put +R to the right of the image, which is the
#: *neurological* convention -- the subject's left appears on the viewer's
#: left. Sagittal puts anterior to the left, so the face looks left.
_PLANE_CONVENTION: dict[Plane, tuple[str, str, str]] = {
    # plane: (normal, up, right)
    Plane.AXIAL: ("S", "A", "R"),
    Plane.CORONAL: ("A", "S", "R"),
    Plane.SAGITTAL: ("R", "S", "P"),
}

_RAS_INDEX = {"R": 0, "A": 1, "S": 2}
_OPPOSITE = {"R": "L", "A": "P", "S": "I", "L": "R", "P": "A", "I": "S"}


@dataclass(frozen=True)
class PlaneLayout:
    """How one plane maps onto the screen, anatomy included.

    ``row``/``col`` are display-grid axes and ``row_flip``/``col_flip`` say
    whether each runs backwards. Without this the panes show whatever order the
    array happened to be stored in, which for a coronal slice means superior
    can end up at the bottom -- and an upside-down brain still looks like a
    brain, so nothing about the image says it is wrong.
    """

    fixed: int
    row: int
    col: int
    row_flip: bool
    col_flip: bool
    #: Edge labels, clockwise from the top: (top, right, bottom, left).
    labels: tuple[str, str, str, str]

    @property
    def axes(self) -> tuple[int, int, int]:
        return (self.fixed, self.row, self.col)

    # The flip arithmetic lives here and nowhere else. It is needed when
    # drawing, when hit-testing a click and when placing the crosshair, and
    # three hand-written copies is how one of them ends up mirrored.

    def to_image(self, ijk: tuple[int, int, int], shape: tuple[int, int, int]) -> tuple[int, int]:
        """Display-grid indices to (row, col) within the drawn plane."""
        row, col = ijk[self.row], ijk[self.col]
        if self.row_flip:
            row = shape[self.row] - 1 - row
        if self.col_flip:
            col = shape[self.col] - 1 - col
        return (int(row), int(col))

    def to_ijk(
        self,
        row: int,
        col: int,
        current: tuple[int, int, int],
        shape: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """(row, col) in the drawn plane back to display-grid indices."""
        if self.row_flip:
            row = shape[self.row] - 1 - row
        if self.col_flip:
            col = shape[self.col] - 1 - col
        out = list(current)
        out[self.row] = int(row)
        out[self.col] = int(col)
        return (out[0], out[1], out[2])


@dataclass(frozen=True)
class PlaneView:
    """Which part of a plane is drawn, and at what magnification.

    Zoom **crops** rather than interpolates: the sampled region shrinks and the
    pane's existing scale-to-fit blows it up with smoothing off. That keeps the
    rule the pane already states -- a viewer must not invent voxels that are
    not there -- and it is cheaper, because magnifying costs fewer samples
    rather than more.

    It composes with :class:`PlaneLayout` rather than duplicating it. The flip
    arithmetic lives in exactly one place for the reason written there, and an
    offset that only *some* of the four callers applied would land in the same
    family of bug: a crosshair drawn where the click did not happen.
    """

    layout: PlaneLayout
    shape: tuple[int, int, int]
    zoom: float = 1.0
    #: Pan in display-grid voxels along the plane's own (row, col) axes. Voxels
    #: rather than a fraction of the window, so the meaning of a drag does not
    #: change as you zoom.
    pan: tuple[float, float] = (0.0, 0.0)

    @property
    def extent(self) -> tuple[int, int]:
        """Full plane size in display-grid voxels, (rows, cols)."""
        return (self.shape[self.layout.row], self.shape[self.layout.col])

    @property
    def span(self) -> tuple[int, int]:
        """Size of the drawn window, which is also the image's pixel size."""
        h, w = self.extent
        z = max(float(self.zoom), 1e-3)
        return (max(1, min(h, round(h / z))), max(1, min(w, round(w / z))))

    @property
    def origin(self) -> tuple[int, int]:
        """Top-left of the drawn window, clamped inside the plane.

        Clamped so panning cannot walk the view off the data and leave a blank
        pane with no indication of which way to come back.
        """
        h, w = self.extent
        sh, sw = self.span
        r0 = round((h - sh) / 2.0 + float(self.pan[0]))
        c0 = round((w - sw) / 2.0 + float(self.pan[1]))
        return (max(0, min(r0, h - sh)), max(0, min(c0, w - sw)))

    @property
    def is_identity(self) -> bool:
        return self.span == self.extent and self.origin == (0, 0)

    def to_image(self, ijk: tuple[int, int, int]) -> tuple[int, int]:
        """Display-grid indices to (row, col) in the drawn image.

        May fall outside the image when the voxel is off-view; callers that
        draw a crosshair want that, so it is not clamped.
        """
        row, col = self.layout.to_image(ijk, self.shape)
        r0, c0 = self.origin
        return (row - r0, col - c0)

    def to_ijk(self, row: int, col: int, current: tuple[int, int, int]) -> tuple[int, int, int]:
        """(row, col) in the drawn image back to display-grid indices."""
        r0, c0 = self.origin
        return self.layout.to_ijk(int(row) + r0, int(col) + c0, current, self.shape)


def ras_axes(affine: np.ndarray) -> dict[str, tuple[int, int]]:
    """For each of R/A/S, which display axis carries it and in which direction.

    Read off the affine's columns: column ``d`` is where display axis ``d``
    points in scanner space, so the dominant row of that column names the
    anatomical direction it most nearly follows.
    """
    mat = np.asarray(affine, dtype=float)[:3, :3]
    out: dict[str, tuple[int, int]] = {}
    used: set[int] = set()
    # Strongest pairings first, so a near-tie cannot steal an axis that another
    # direction matches far better.
    order = sorted(((abs(mat[r, d]), r, d) for r in range(3) for d in range(3)), reverse=True)
    for _, r, d in order:
        letter = "RAS"[r]
        if letter in out or d in used:
            continue
        out[letter] = (d, 1 if mat[r, d] >= 0 else -1)
        used.add(d)
    return out


def plane_layout(affine: np.ndarray, plane: Plane) -> PlaneLayout:
    """Lay a plane out so anatomy is where a radiologist expects it."""
    axes = ras_axes(affine)
    normal, up, right = _PLANE_CONVENTION[plane]

    def resolve(direction: str) -> tuple[int, int]:
        """Display axis and sign for an anatomical direction like 'P'."""
        if direction in _RAS_INDEX:
            return axes[direction]
        axis, sign = axes[_OPPOSITE[direction]]
        return axis, -sign

    fixed, _ = axes[normal]
    up_axis, up_sign = resolve(up)
    right_axis, right_sign = resolve(right)
    return PlaneLayout(
        fixed=fixed,
        row=up_axis,
        col=right_axis,
        # Rows run down the screen, so "up" means the row index must decrease
        # as the anatomical direction increases.
        row_flip=up_sign > 0,
        col_flip=right_sign < 0,
        labels=(up, right, _OPPOSITE[up], _OPPOSITE[right]),
    )


def plane_axes(plane: Plane, affine: np.ndarray | None = None) -> tuple[int, int, int]:
    """``(fixed, row, col)`` display axes for a plane."""
    if affine is None:
        # Identity fallback: only for callers with no grid yet.
        return {Plane.SAGITTAL: (0, 1, 2), Plane.CORONAL: (1, 0, 2), Plane.AXIAL: (2, 0, 1)}[plane]
    return plane_layout(affine, plane).axes


def plane_shape(grid: DisplayGrid, plane: Plane) -> tuple[int, int]:
    """Pixel dimensions of one display plane."""
    layout = plane_layout(grid.affine, plane)
    return (grid.shape[layout.row], grid.shape[layout.col])


def plane_indices(
    grid: DisplayGrid,
    plane: Plane,
    position: int,
    *,
    view: PlaneView | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``(H, W, 3)`` display-grid indices covering one plane.

    ``view`` crops and magnifies; without one the whole plane is covered. Built
    on-device so a redraw never round-trips index arithmetic through the host.
    """
    layout = plane_layout(grid.affine, plane)
    if view is None:
        view = PlaneView(layout=layout, shape=grid.shape)
    full_h, full_w = view.extent
    h, w = view.span
    r0, c0 = view.origin
    rows = torch.arange(h, device=device, dtype=dtype) + float(r0)
    cols = torch.arange(w, device=device, dtype=dtype) + float(c0)
    if layout.row_flip:
        rows = (full_h - 1) - rows
    if layout.col_flip:
        cols = (full_w - 1) - cols
    out = torch.empty((h, w, 3), device=device, dtype=dtype)
    out[..., layout.fixed] = float(position)
    out[..., layout.row] = rows.unsqueeze(1).expand(h, w)
    out[..., layout.col] = cols.unsqueeze(0).expand(h, w)
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
    view: PlaneView | None = None,
    mode: str = "bilinear",
    fill: float = 0.0,
) -> Tensor:
    """One display plane of a layer, resampled into the display grid.

    Returns ``(H, W)``. When the layer already sits on the display grid this
    still goes through ``grid_sample``; at 0.08 ms the special case would cost
    more in divergent code paths than it saves.
    """
    ijk = plane_indices(
        grid, plane, position, view=view, device=volume.device, dtype=volume.dtype
    )
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
