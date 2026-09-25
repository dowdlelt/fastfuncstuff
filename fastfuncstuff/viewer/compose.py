"""Turning session state into pixels.

Kept out of the UI so a pane image can be produced, tested and saved with no
window open -- which is also what makes screenshots and montages fall out of the
same code path the screen uses, rather than a parallel one that drifts.

The whole composition measures well under a millisecond for three planes, so
nothing here caches rendered output. Only the two things that would be wasteful
to rebuild per frame are cached: colour LUTs and the on-device copy of the
currently displayed sub-brick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from fastfuncstuff.viewer.colormap import (
    apply_colormap,
    apply_label_colors,
    build_lut,
    composite,
    label_edges,
    suprathreshold_edges,
    threshold_alpha,
    to_rgba8,
)
from fastfuncstuff.viewer.layers import Layer
from fastfuncstuff.viewer.slicing import (
    PlaneView,
    extract_plane,
    plane_coverage,
    plane_layout,
    plane_shape,
)
from fastfuncstuff.viewer.state import Plane, ViewerState

#: Colour drawn around suprathreshold voxels in boxed mode. Near-white reads
#: against every scale in the palette, warm or cool.
BOX_RGB = (0.96, 0.96, 0.92)

_LUT_CACHE: dict[tuple[str, str, int, bool], Tensor] = {}


def cached_lut(
    name: str, device: torch.device, size: int = 256, *, reverse: bool = False
) -> Tensor:
    key = (name, str(device), size, reverse)
    lut = _LUT_CACHE.get(key)
    if lut is None:
        lut = build_lut(name, size, device=device, reverse=reverse)
        _LUT_CACHE[key] = lut
    return lut


@dataclass(frozen=True)
class PaneImage:
    """One rendered plane, ready to blit."""

    rgba: Tensor  # (H, W, 4) uint8
    plane: Plane
    position: int

    @property
    def size(self) -> tuple[int, int]:
        return (int(self.rgba.shape[0]), int(self.rgba.shape[1]))


def plane_position(state: ViewerState, plane: Plane) -> int:
    """Where the crosshair puts this plane."""
    if state.grid is None:
        return 0
    return int(state.crosshair[plane_layout(state.grid.affine, plane).fixed])


def _layer_alpha(layer: Layer, values: Tensor, stat: Tensor) -> tuple[Tensor, Tensor | None]:
    """Opacity for one layer, plus its boxed edge mask when enabled."""
    alpha = threshold_alpha(stat, layer.threshold, mode=layer.alpha_mode, sign_mode=layer.sign_mode)
    alpha = alpha * float(layer.opacity)
    edges = None
    if layer.boxed and layer.threshold > 0.0:
        edges = suprathreshold_edges(stat, layer.threshold, sign_mode=layer.sign_mode)
    return alpha, edges


def _voxel_mm(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(np.asarray(affine, dtype=float)[:3, :3])) ** (1.0 / 3.0))


def edge_colors(values: Tensor, inside: Tensor, sigma: float) -> tuple[Tensor, Tensor]:
    """A plane drawn as its thin edges: ``(H, W, 3)`` colour and ``(H, W)`` alpha.

    The same detector and colour ramp as the warp movies' edge overlay, so an
    edge on screen and an edge in a QC movie mean the same thing. Found in 2-D on
    the displayed plane: a 3-D edge map cut by a slice shows filled patches
    wherever a surface runs parallel to the cut.

    ``sigma`` is in display pixels. ``inside`` marks where the layer has data;
    the step from data to the zero outside its field of view is an edge of the
    sampling, not of the anatomy, and is dropped.
    """
    from fastfuncstuff.processing.edges import edge_map
    from fastfuncstuff.viz.compose import _EDGE_RAMP

    rgb = torch.zeros((*values.shape, 3), dtype=values.dtype, device=values.device)
    alpha = torch.zeros_like(values)
    if min(values.shape) < 3:
        return rgb, alpha
    strength = edge_map(values, sigma=sigma, threshold=0.1)
    if not bool(inside.all()):
        reach = int(np.ceil(2.0 * sigma)) + 1
        outside = (~inside).to(values.dtype)[None, None]
        near = F.max_pool2d(outside, 2 * reach + 1, stride=1, padding=reach)[0, 0] > 0
        strength = torch.where(near, torch.zeros_like(strength), strength)
    kept = strength[strength > 0]
    if kept.numel() == 0:
        return rgb, alpha
    # Saturate at the 33rd percentile of the kept edges, as AFNI's edge QC does:
    # most edges draw at full colour and only the weakest third fade to red.
    vmax = torch.quantile(kept, 0.33).clamp_min(torch.finfo(values.dtype).tiny)
    ramp = torch.as_tensor(_EDGE_RAMP, dtype=values.dtype, device=values.device)
    t = (strength / vmax).clamp(0.0, 1.0) * (len(ramp) - 1)
    lo = t.floor().long()
    hi = (lo + 1).clamp(max=len(ramp) - 1)
    frac = (t - lo.to(t.dtype)).unsqueeze(-1)
    rgb = ramp[lo] * (1.0 - frac) + ramp[hi] * frac
    return rgb, (strength > 0).to(values.dtype)


#: The viewer's words for resampling, and what ``grid_sample`` calls them.
#: "linear" rather than "bilinear" in the interface, because the sampling is
#: three-dimensional and the ``bi`` is an artefact of torch's 2-D naming.
_GRID_SAMPLE = {"nearest": "nearest", "linear": "bilinear"}


def render_plane(
    session,
    plane: Plane,
    *,
    position: int | None = None,
    solo_key: str | None = None,
    view: PlaneView | None = None,
) -> PaneImage | None:
    """Composite every visible layer for one display plane.

    With ``solo_key`` only that layer is drawn, and its visibility flag is
    ignored -- soloing a hidden layer that then stays hidden would make the
    flip-between-two gesture fail silently on every other press.

    Returns ``None`` when there is nothing to draw, so callers can distinguish
    an empty session from a black image.
    """
    state: ViewerState = session.state
    grid = state.grid
    if grid is None:
        return None
    if solo_key is not None:
        only = state.layers.find(solo_key)
        visible = [only] if only is not None else []
    else:
        visible = state.layers.visible_layers()
    if not visible:
        return None

    layout = plane_layout(grid.affine, plane)
    pos = plane_position(state, plane) if position is None else position
    pos = max(0, min(pos, grid.shape[layout.fixed] - 1))

    stacked: list[tuple[Tensor, Tensor]] = []
    box_overlays: list[Tensor] = []

    for layer in visible:
        volume = session.display_volume(layer.key)
        if volume is None:
            continue
        # Display only: how the voxels are painted into the grid, never what
        # they are. session.resample_mode is the single place that is decided.
        how = _GRID_SAMPLE[session.resample_mode(layer)]
        if layer.edges:
            # Edges of a nearest-sampled coarse layer would trace the voxel
            # staircase rather than the anatomy.
            how = "bilinear"
        values = extract_plane(volume, grid, layer.affine, plane, pos, view=view, mode=how)

        if layer.edges:
            inside = plane_coverage(
                layer.shape, grid, layer.affine, plane, pos, view=view, device=values.device
            )
            # One native voxel of smoothing, however far the grid magnifies it.
            sigma = max(1.0, _voxel_mm(layer.affine) / max(_voxel_mm(grid.affine), 1e-6))
            rgb, alpha = edge_colors(values, inside, sigma)
            stacked.append((rgb, alpha * float(layer.opacity)))
            continue

        # The threshold statistic may live in a different sub-brick than the one
        # being displayed -- that is the normal case for a stats dataset, where
        # you colour by effect size and threshold on a t.
        if layer.threshold_index is None:
            stat = values
        else:
            stat_vol = session.display_volume(layer.key, index=layer.threshold_index)
            stat = (
                values
                if stat_vol is None
                else extract_plane(stat_vol, grid, layer.affine, plane, pos, view=view, mode=how)
            )

        # A label layer is coloured by identity rather than by magnitude, and
        # nothing about a range, a threshold or an alpha ramp applies to it --
        # "half of region 12" is not a thing. Boxed, though, means something
        # better here than it does on a stat map: the borders between regions.
        palette = session.roi_palette(layer.key, values.device) if layer.roi else None
        if palette is not None:
            rgb, alpha = apply_label_colors(values, palette)
            stacked.append((rgb, alpha * float(layer.opacity)))
            if layer.boxed:
                box_overlays.append(label_edges(values))
            continue

        lo = layer.range_lo if layer.range_lo is not None else 0.0
        hi = layer.range_hi if layer.range_hi is not None else 1.0
        rgb = apply_colormap(
            values,
            lut=cached_lut(layer.colormap, values.device, reverse=layer.colormap_reversed),
            lo=float(lo),
            hi=float(hi),
            sign_mode=layer.sign_mode,
            n_panes=layer.n_panes,
        )
        # Zero the stat as well as the alpha: boxed edges are traced on the
        # stat, and a cluster cut by the mask should be outlined where it is cut.
        keep = session.display_mask(layer.key)
        if keep is not None:
            inside = extract_plane(keep, grid, layer.affine, plane, pos, view=view, mode="nearest")
            inside = inside > 0.5
            stat = torch.where(inside, stat, torch.zeros_like(stat))
        alpha, edges = _layer_alpha(layer, values, stat)
        if keep is not None:
            alpha = alpha * inside
        stacked.append((rgb, alpha))
        if edges is not None:
            box_overlays.append(edges)

    if not stacked:
        return None

    rgb = composite(stacked)

    # Boxes go on last so an outline is never hidden by a layer above it.
    for edges in box_overlays:
        box = torch.tensor(BOX_RGB, dtype=rgb.dtype, device=rgb.device)
        rgb = torch.where(edges.unsqueeze(-1), box, rgb)

    return PaneImage(rgba=to_rgba8(rgb), plane=plane, position=pos)


def plane_view(state: ViewerState, viewport) -> PlaneView | None:
    """The crop-and-magnify a viewport asks for, or ``None`` for the whole plane.

    One function so the renderer, the crosshair and the hit test all read the
    viewport the same way. Four separate readings is the shape of the bug the
    flip arithmetic already taught us about.
    """
    if state.grid is None:
        return None
    return PlaneView(
        layout=plane_layout(state.grid.affine, viewport.plane),
        shape=state.grid.shape,
        zoom=float(viewport.zoom),
        pan=(float(viewport.pan[0]), float(viewport.pan[1])),
    )


def render_viewport(session, viewport) -> PaneImage | None:
    """Render what one image window shows.

    The window's own settings -- which plane, and whether it is soloed -- are
    read here rather than by the widget, so a screenshot and the screen come
    from the same call.
    """
    solo = None
    if viewport.solo:
        layer = session.state.selected_layer()
        solo = layer.key if layer is not None else None
        if solo is None:
            return None
    # An unlocked window stays on the slice it was parked on; a locked one has
    # no slice of its own and takes the crosshair's.
    position = None if viewport.locked else viewport.position
    return render_plane(
        session,
        viewport.plane,
        position=position,
        solo_key=solo,
        view=plane_view(session.state, viewport),
    )


def render_all(session) -> dict[Plane, PaneImage]:
    """Every plane at the current crosshair."""
    out: dict[Plane, PaneImage] = {}
    for plane in Plane:
        img = render_plane(session, plane)
        if img is not None:
            out[plane] = img
    return out


def empty_pane(
    state: ViewerState, plane: Plane, *, device: torch.device | None = None
) -> PaneImage:
    """A black pane of the right size, for when a layer is not resident yet."""
    if state.grid is None:
        shape = (1, 1)
    else:
        shape = plane_shape(state.grid, plane)
    rgba = torch.zeros((*shape, 4), dtype=torch.uint8, device=device)
    rgba[..., 3] = 255
    return PaneImage(rgba=rgba, plane=plane, position=0)


__all__ = [
    "BOX_RGB",
    "PaneImage",
    "cached_lut",
    "edge_colors",
    "empty_pane",
    "plane_position",
    "plane_view",
    "render_all",
    "render_plane",
    "render_viewport",
]
