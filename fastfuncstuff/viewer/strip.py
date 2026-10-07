"""The neighbour strip: small cropped slices either side of an image window's own.

Editing a surface in one slice moves it in the slices around it, over the
brush, and a single view cannot show that. A montage can, but at the cost of
every slice being small. The strip keeps the main view as it is and adds a row
of crops centred on the crosshair -- the place being edited -- from the slices
on either side.

While a surface is being edited the strip's default spacing follows the brush:
the outermost cells sit at the brush's reach and the rest divide it evenly, so
the strip shows where the edit fades out as well as how it does so. Outside
editing the default is the adjacent slices.

Kept out of the UI for the reason :mod:`viewer.compose` is: the cells come
from one call that a test, a screenshot and the screen all share.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastfuncstuff.viewer.compose import PaneImage, plane_view, render_plane, view_grid
from fastfuncstuff.viewer.slicing import PlaneView, plane_normal, ras_axes


@dataclass(frozen=True)
class StripCell:
    """One slice of the strip."""

    #: Signed slice offset from the main view, in display-grid indices.
    offset: int
    #: The slice drawn, or ``None`` when the offset runs off the grid.
    position: int | None
    #: Signed distance along the plane's anatomical normal (S, A or R), mm.
    #: Positive cells are drawn right of centre.
    mm: float
    view: PlaneView
    image: PaneImage | None
    #: The anatomical side the cell lies on (S/I, A/P or R/L).
    side: str

    @property
    def label(self) -> str:
        """``S 4.0`` -- which side and how far, which is what the eye needs."""
        return f"{self.side} {abs(self.mm):.1f}"


def strip_offsets(count: int, step: int, reach: float) -> list[int]:
    """Positive offsets for one half of the strip, nearest first.

    ``step`` > 0 spaces them evenly by that many slices. ``step`` 0 (auto)
    divides ``reach`` slices evenly so the last lands on it, but never closer
    than one slice apart -- a reach smaller than the half-count just gives
    the adjacent slices.
    """
    half = count // 2
    if half <= 0:
        return []
    if step > 0:
        return [k * step for k in range(1, half + 1)]
    out: list[int] = []
    for k in range(1, half + 1):
        nxt = int(round(k * reach / half))
        out.append(max(nxt, out[-1] + 1 if out else 1))
    return out


def slice_mm(grid, fixed: int) -> float:
    """Slice thickness: the length of the fixed axis's voxel edge, in mm."""
    return float(np.linalg.norm(np.asarray(grid.affine, float)[:3, fixed]))


def auto_reach(session, viewport) -> float:
    """Slices the auto step spans: the brush's reach while editing, else 1 per cell.

    Returns 0 when not editing, which :func:`strip_offsets` turns into the
    adjacent slices.
    """
    state = session.state
    if state.grid is None or not state.surface_editing or not session.surfaces.hemis:
        return 0.0
    view = plane_view(state, viewport)
    assert view is not None
    return float(state.surface_brush[0]) / max(slice_mm(state.grid, view.layout.fixed), 1e-6)


def effective_step(session, viewport) -> int:
    """The spacing the strip is using, in slices -- what auto resolved to."""
    if viewport.strip_step > 0:
        return int(viewport.strip_step)
    offs = strip_offsets(max(viewport.strip, 2), 0, auto_reach(session, viewport))
    return offs[0] if offs else 1


def strip_view(state, viewport, centre=None) -> PlaneView | None:
    """The crop every cell shares: the strip's zoom, centred on ``centre`` (grid
    indices; the crosshair when ``None``)."""
    view = plane_view(state, viewport)
    if view is None:
        return None
    at = state.crosshair if centre is None else tuple(int(round(float(v))) for v in centre)
    centre = view.pan_centring(at)
    return PlaneView(
        layout=view.layout, shape=view.shape, zoom=float(viewport.strip_zoom), pan=centre
    )


def base_position(state, viewport) -> int:
    """The main view's slice: its parked one, or the crosshair's."""
    view = plane_view(state, viewport)
    assert view is not None
    if not viewport.locked and viewport.position is not None:
        return int(viewport.position)
    return int(state.crosshair[view.layout.fixed])


def normal_sign(state, viewport) -> tuple[int, tuple[str, str]]:
    """(+1 or -1, letters): which way an increasing slice index goes, anatomically.

    The plane's normal letter (S for axial, A coronal, R sagittal) is the side
    drawn on the right, so the strip reads the same way whatever order the
    grid stores its slices in.
    """
    view = plane_view(state, viewport)
    assert view is not None and state.grid is not None
    letters = plane_normal(viewport.plane)
    # Through the untilted layout: a tilted grid keeps the same index order.
    axis, sign = ras_axes(np.asarray(state.grid.affine, float))[letters[0]]
    assert axis == view.layout.fixed
    return sign, letters


def strip_cells(session, viewport, *, centre=None, render: bool = True) -> list[StripCell]:
    """The strip's cells, left to right, with their pixels when ``render``.

    ``centre`` moves the crop off the crosshair -- to where a surface was
    grabbed, which is not where the crosshair is unless it was clicked there.

    Each cell is drawn the way the main view is -- same grid (tilted when the
    window is oblique), same solo -- so the strip cannot disagree with it about
    anything but which slice it shows.
    """
    state = session.state
    count = int(viewport.strip)
    if count <= 0 or state.grid is None:
        return []
    view = strip_view(state, viewport, centre)
    if view is None:
        return []
    grid = view_grid(state, viewport)
    assert grid is not None
    fixed = view.layout.fixed
    extent = state.grid.shape[fixed]
    thick = slice_mm(grid, fixed)
    reach = auto_reach(session, viewport) if viewport.strip_step <= 0 else 0.0
    half = strip_offsets(count, int(viewport.strip_step), reach)
    sign, (pos_side, neg_side) = normal_sign(state, viewport)
    here = base_position(state, viewport)

    solo = None
    if viewport.solo:
        layer = state.selected_layer()
        solo = layer.key if layer is not None else None

    cells: list[StripCell] = []
    # Anatomically negative side first (left), nearest the centre last.
    for k in [*(-sign * o for o in reversed(half)), *(sign * o for o in half)]:
        pos = here + k
        inside = 0 <= pos < extent
        image = None
        # A soloed window with nothing selected draws nothing, as the main view does.
        if render and inside and (solo is not None or not viewport.solo):
            image = render_plane(
                session, viewport.plane, position=pos, solo_key=solo, view=view, grid=grid
            )
        cells.append(
            StripCell(
                offset=int(k),
                position=pos if inside else None,
                mm=float(k * sign * thick),
                view=view,
                image=image,
                side=pos_side if k * sign >= 0 else neg_side,
            )
        )
    return cells


__all__ = [
    "StripCell",
    "auto_reach",
    "base_position",
    "effective_step",
    "normal_sign",
    "slice_mm",
    "strip_cells",
    "strip_offsets",
    "strip_view",
]
