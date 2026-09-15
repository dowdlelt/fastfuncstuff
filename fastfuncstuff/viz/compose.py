"""Frame composition: greyscale panels on one physical scale, edge overlays, captions.

Pure numpy + Pillow on the CPU. Nothing here runs per optimizer iteration -- a
recorder stores sampled planes and all of this happens once, after the fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from .slices import PlaneView

# Edge colour ramp, weak -> strong: deep red through orange to yellow. The same
# family as the positive half of AFNI's Reds_and_Blues_Inv that the @SSwarper edge
# QC uses, and it stays readable over both dark and bright tissue.
_EDGE_RAMP = np.array(
    [
        [0.75, 0.05, 0.05],
        [1.00, 0.35, 0.00],
        [1.00, 0.75, 0.05],
        [1.00, 1.00, 0.35],
    ],
    dtype=np.float32,
)

_BG = 0
_GAP = 4


def intensity_window(
    values: np.ndarray, clip: tuple[float, float] = (1.0, 99.0)
) -> tuple[float, float]:
    """Display range from the non-zero, finite values (zeros are usually padding)."""
    v = values[np.isfinite(values) & (values != 0)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = (float(x) for x in np.percentile(v, clip))
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    return lo, hi if hi > lo else lo + 1.0


def edge_range(edges: np.ndarray, pct: float = 33.0) -> float:
    """Strength at which the edge colour saturates.

    AFNI's QC saturates at the 33rd percentile of the non-zero edges, so most edges
    draw at full colour and only the weakest third fade toward red.
    """
    nz = edges[edges > 0]
    return float(np.percentile(nz, pct)) if nz.size else 1.0


def colorize_edges(strength: np.ndarray, vmax: float) -> tuple[np.ndarray, np.ndarray]:
    """(H, W) edge strength -> (H, W, 3) float RGB in [0, 1] and an (H, W) bool mask."""
    t = np.clip(strength / max(vmax, 1e-12), 0.0, 1.0) * (len(_EDGE_RAMP) - 1)
    lo = np.floor(t).astype(np.int64)
    hi = np.minimum(lo + 1, len(_EDGE_RAMP) - 1)
    frac = (t - lo)[..., None]
    rgb = _EDGE_RAMP[lo] * (1 - frac) + _EDGE_RAMP[hi] * frac
    return rgb, strength > 0


def display_edges(
    plane: np.ndarray, size: tuple[int, int], sigma: float = 1.0, threshold: float = 0.1
) -> np.ndarray:
    """Thin edges of one displayed plane, found at its display resolution.

    Edges are computed in 2-D on the plane *after* upscaling to ``size``, for two
    reasons. A 3-D edge map cut by a slice shows filled patches wherever a surface
    runs parallel to the cut. And edges found on the native grid and then enlarged
    are as thick as the enlargement -- a 2 mm grid shown at 2.7x draws 5-pixel lines
    that bury the anatomy the overlay is meant to be judged against.

    Args:
        plane: (rows, cols) reference image on the native grid.
        size: (h, w) display pixels (from :func:`panel_sizes`).
        sigma: Smoothing in *native* voxels, so the edges found do not depend on the
            display size.
        threshold: As in :func:`fastfuncstuff.processing.edges.edge_map`.
    """
    import torch

    from fastfuncstuff.processing.edges import _median_cross, edge_map

    native = torch.as_tensor(np.nan_to_num(plane), dtype=torch.float32)
    if min(native.shape) >= 3:
        native = _median_cross(native)
    big = torch.tensor(_resize(native.numpy(), size, True), dtype=torch.float32)
    mag = min(size[0] / plane.shape[0], size[1] / plane.shape[1])
    return edge_map(big, sigma=sigma * mag, median=False, threshold=threshold).numpy()


def panel_sizes(views: Sequence[PlaneView], height: int) -> list[tuple[int, int]]:
    """(h, w) pixels per panel, one mm-per-pixel scale shared by every panel.

    The physically tallest panel gets ``height`` rows; the others keep their true
    size relative to it, so a coronal and a sagittal cut of the same head line up.
    """
    tallest = max(v.shape[0] * v.mm[0] for v in views)
    scale = height / tallest
    return [
        (max(1, round(v.shape[0] * v.mm[0] * scale)), max(1, round(v.shape[1] * v.mm[1] * scale)))
        for v in views
    ]


def _resize(img: np.ndarray, size: tuple[int, int], smooth: bool) -> np.ndarray:
    from PIL import Image

    h, w = size
    if img.shape[:2] == (h, w):
        return img
    mode = Image.Resampling.BILINEAR if smooth else Image.Resampling.NEAREST
    if img.dtype == np.uint8:
        return np.asarray(Image.fromarray(img).resize((w, h), mode))
    return np.asarray(Image.fromarray(img.astype(np.float32)).resize((w, h), mode))


@lru_cache(maxsize=8)
def _font(px: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=px)
    except TypeError:  # Pillow < 10.1 has only the fixed bitmap font
        return ImageFont.load_default()


def compose_frame(
    rows: Sequence[Sequence[np.ndarray]],
    views: Sequence[PlaneView],
    height: int,
    windows: Sequence[tuple[float, float]],
    *,
    edges: Sequence[np.ndarray] | None = None,
    edge_vmax: float = 1.0,
    edge_opacity: float = 1.0,
    label: str = "",
    row_labels: Sequence[str] | None = None,
    progress: float | None = None,
) -> np.ndarray:
    """One (H, W, 3) uint8 movie frame: a caption strip over rows of side-by-side panels.

    Args:
        rows: Per row (one image being warped), one (rows, cols) float image per view.
        views: The matching :class:`PlaneView` descriptions.
        height: Pixel height of the physically tallest panel.
        windows: Per row, the (lo, hi) greyscale display range -- fixed for the whole
            movie so brightness does not flicker between frames.
        edges: Optional per-view edge strength drawn over every row, at panel
            resolution (:func:`display_edges`) or at plane resolution (upscaled nearest).
        edge_vmax: Strength at which edge colour saturates (see :func:`edge_range`).
        edge_opacity: 0..1 blend of the edge colour over the greyscale.
        label: Caption text.
        row_labels: Optional name drawn at the start of each row.
        progress: Optional 0..1 fraction drawn as a thin bar under the caption.
    """
    from PIL import Image, ImageDraw

    if len(windows) != len(rows):
        raise ValueError("need one display window per row")
    sizes = panel_sizes(views, height)
    font_px = max(10, height // 18)
    strip = font_px + 8
    bar = 3 if progress is not None else 0
    width = sum(w for _, w in sizes) + _GAP * (len(sizes) - 1)
    top = strip + bar
    canvas = np.full((top + len(rows) * (height + _GAP) - _GAP, width, 3), _BG, np.uint8)

    for r, (panels, (lo, hi)) in enumerate(zip(rows, windows, strict=True)):
        x = 0
        for i, (img, (h, w)) in enumerate(zip(panels, sizes, strict=True)):
            grey = np.clip((np.nan_to_num(img) - lo) / (hi - lo), 0.0, 1.0)
            rgb = np.repeat(_resize((grey * 255).astype(np.uint8), (h, w), True)[..., None], 3, -1)
            rgb = rgb.astype(np.float32) / 255.0
            if edges is not None:
                # Nearest-neighbour: blurring a one-pixel ridge smears it back into a band.
                colour, on = colorize_edges(_resize(edges[i], (h, w), False), edge_vmax)
                rgb[on] = rgb[on] * (1 - edge_opacity) + colour[on] * edge_opacity
            y = top + r * (height + _GAP) + (height - h) // 2
            canvas[y : y + h, x : x + w] = (rgb * 255).astype(np.uint8)
            x += w + _GAP

    if progress is not None:
        canvas[strip : strip + bar, : int(round(np.clip(progress, 0, 1) * width))] = (90, 150, 220)

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    if label:
        draw.text((4, 4), label, fill=(235, 235, 235), font=_font(font_px))
    small = _font(max(9, font_px - 3))
    for r in range(len(rows)):
        x = 0
        for i, (v, (h, w)) in enumerate(zip(views, sizes, strict=True)):
            y = top + r * (height + _GAP) + (height - h) // 2
            text = v.left_letter
            if i == 0 and row_labels:
                text = f"{v.left_letter}  {row_labels[r]}"
            draw.text((x + 3, y + 2), text, fill=(120, 200, 255), font=small)
            x += w + _GAP
    return np.asarray(pil)
