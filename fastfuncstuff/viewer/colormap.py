"""Colour mapping, thresholding and compositing, on whatever device holds the data.

Everything here is torch and shape-agnostic, so the same code paths a test drives
on a 2-D slice run unchanged on a montage or a whole volume. A slice colour-map
measures about 0.08 ms on GPU, which is why thresholding is recomputed on every
change rather than cached.

Two AFNI behaviours are reproduced deliberately, because they are the honest way
to draw a statistic map:

* **Alpha** fades sub-threshold voxels instead of hiding them, so a voxel just
  under threshold reads as faint rather than absent.
* **Boxed** outlines the voxels that did pass, so fading never leaves you
  guessing which ones survived.

Value parity with AFNI is the goal for thresholds and readouts; the colours
themselves are ours.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fastfuncstuff.viewer.layers import AlphaMode, SignMode

#: Colour scales as control points, interpolated into a LUT on demand. Held as
#: stops rather than baked tables so a scale stays readable and editable here,
#: and so any resolution can be generated without shipping a big array.
_SCALES: dict[str, tuple[tuple[float, float, float], ...]] = {
    "gray": ((0, 0, 0), (1, 1, 1)),
    "hot": ((0, 0, 0), (0.7, 0, 0), (1, 0.6, 0), (1, 1, 0.85)),
    "cool": ((0, 0, 0), (0, 0.35, 0.7), (0, 0.75, 0.9), (0.85, 1, 1)),
    # Diverging, for signed statistics: cool for negative, warm for positive,
    # with a neutral mid so zero does not read as a value.
    "redblue": (
        (0.15, 0.45, 0.85),
        (0.45, 0.7, 0.95),
        (0.92, 0.92, 0.92),
        (0.98, 0.65, 0.35),
        (0.85, 0.2, 0.12),
    ),
    "viridis": (
        (0.267, 0.005, 0.329),
        (0.229, 0.322, 0.545),
        (0.128, 0.567, 0.551),
        (0.369, 0.789, 0.383),
        (0.993, 0.906, 0.144),
    ),
    "spectrum": (
        (0.6, 0.0, 0.7),
        (0.0, 0.0, 1.0),
        (0.0, 0.8, 0.8),
        (0.0, 0.8, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.4, 0.0),
        (1.0, 0.0, 0.0),
    ),
}

DEFAULT_LUT_SIZE = 256


def available_colormaps() -> list[str]:
    return sorted(_SCALES)


def build_lut(
    name: str,
    size: int = DEFAULT_LUT_SIZE,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """A ``(size, 3)`` RGB lookup table for one colour scale."""
    try:
        stops = _SCALES[name]
    except KeyError:
        raise KeyError(
            f"unknown colormap {name!r}; have {', '.join(available_colormaps())}"
        ) from None
    anchors = torch.tensor(stops, dtype=dtype, device=device)
    n = anchors.shape[0]
    if size < 2:
        raise ValueError("LUT size must be at least 2")
    pos = torch.linspace(0.0, 1.0, size, dtype=dtype, device=device) * (n - 1)
    lo = pos.floor().clamp(max=n - 2).long()
    frac = (pos - lo.to(dtype)).unsqueeze(1)
    return anchors[lo] * (1 - frac) + anchors[lo + 1] * frac


def normalize(
    values: Tensor, lo: float, hi: float, *, sign_mode: SignMode = SignMode.BOTH
) -> Tensor:
    """Map values into ``[0, 1]`` for LUT lookup, honouring the sign mode.

    ``BOTH`` spreads ``[lo, hi]`` across the whole scale, which puts zero at the
    midpoint of a symmetric range -- what a diverging scale needs. ``POS`` and
    ``NEG`` use only the relevant half of the data but still the whole scale, so
    a one-sided map keeps its full colour resolution.
    """
    if sign_mode is SignMode.POS:
        top = max(abs(hi), abs(lo))
        return (values.clamp(min=0.0) / top).clamp(0.0, 1.0) if top else torch.zeros_like(values)
    if sign_mode is SignMode.NEG:
        top = max(abs(hi), abs(lo))
        return ((-values).clamp(min=0.0) / top).clamp(0.0, 1.0) if top else torch.zeros_like(values)
    span = hi - lo
    if span <= 0:
        return torch.zeros_like(values)
    return ((values - lo) / span).clamp(0.0, 1.0)


def quantize(unit: Tensor, n_panes: int) -> Tensor:
    """Collapse a continuous ``[0, 1]`` field into ``n_panes`` discrete bands.

    This is AFNI's classic panelled colour bar. It is not a degraded continuous
    scale: banding is what makes it possible to read a value off a map by eye
    instead of estimating it from a gradient.
    """
    if n_panes <= 0:
        return unit
    idx = (unit * n_panes).floor().clamp(0, n_panes - 1)
    return (idx + 0.5) / n_panes


def apply_colormap(
    values: Tensor,
    *,
    lut: Tensor,
    lo: float,
    hi: float,
    sign_mode: SignMode = SignMode.BOTH,
    n_panes: int = 0,
) -> Tensor:
    """Colour a field, returning ``(..., 3)`` RGB in ``[0, 1]``."""
    unit = quantize(normalize(values, lo, hi, sign_mode=sign_mode), n_panes)
    idx = (unit * (lut.shape[0] - 1)).round().long().clamp(0, lut.shape[0] - 1)
    return lut[idx]


def apply_label_colors(values: Tensor, palette: Tensor) -> tuple[Tensor, Tensor]:
    """Colour a label field by identity: ``(rgb, alpha)``, no scale involved.

    A label volume has no range to normalise against -- region 40 is not twice
    region 20 -- so this indexes a palette by the value itself and makes
    everything outside every region transparent. ``palette`` is ``(N, 3)`` in
    ``[0, 1]``, indexed by label value, with row 0 unused.
    """
    idx = torch.nan_to_num(values).round().long().clamp(0, palette.shape[0] - 1)
    return palette[idx], (idx > 0).to(palette.dtype)


def label_edges(values: Tensor) -> Tensor:
    """Boolean map of voxels on a boundary between two different labels.

    Not :func:`suprathreshold_edges` on the union: an atlas outlined as one
    blob is a picture of the brain's convex hull. What makes outlines worth
    having on a parcellation is the borders *between* regions, so the test is
    "my neighbour has a different label", and the background counts as a label
    for that purpose so the outer rim is drawn too.
    """
    if values.ndim != 2:
        raise ValueError("edges are computed on a 2-D slice")
    labels = torch.nan_to_num(values).round()
    interior = torch.ones_like(labels, dtype=torch.bool)
    interior[:-1, :] &= labels[1:, :] == labels[:-1, :]
    interior[1:, :] &= labels[:-1, :] == labels[1:, :]
    interior[:, :-1] &= labels[:, 1:] == labels[:, :-1]
    interior[:, 1:] &= labels[:, :-1] == labels[:, 1:]
    return (labels > 0) & ~interior


def threshold_alpha(
    stat: Tensor,
    threshold: float,
    *,
    mode: AlphaMode = AlphaMode.OFF,
    sign_mode: SignMode = SignMode.BOTH,
) -> Tensor:
    """Per-voxel opacity from a statistic.

    With ``AlphaMode.OFF`` this is a hard mask -- the classic behaviour where a
    voxel is either drawn or not. The ramps fade what lies below threshold, and
    reach zero only at zero, so nothing is hidden that was merely close.
    """
    if sign_mode is SignMode.POS:
        magnitude = stat.clamp(min=0.0)
    elif sign_mode is SignMode.NEG:
        magnitude = (-stat).clamp(min=0.0)
    else:
        magnitude = stat.abs()

    thr = abs(float(threshold))
    if thr <= 0.0:
        # No threshold set: everything is fully opaque, but a one-sided mode
        # still hides the side it does not show.
        if sign_mode is SignMode.BOTH:
            return torch.ones_like(magnitude)
        return (magnitude > 0).to(magnitude.dtype)

    passed = magnitude >= thr
    if mode is AlphaMode.OFF:
        return passed.to(magnitude.dtype)
    ramp = (magnitude / thr).clamp(0.0, 1.0)
    if mode is AlphaMode.QUADRATIC:
        ramp = ramp * ramp
    return torch.where(passed, torch.ones_like(ramp), ramp)


def suprathreshold_edges(
    stat: Tensor, threshold: float, *, sign_mode: SignMode = SignMode.BOTH
) -> Tensor:
    """Boolean edge map of the region that passed threshold.

    A voxel is an edge if it passed and any 4-neighbour did not, which is what
    AFNI's boxed mode outlines. Works on a 2-D slice; the caller slices first.
    """
    if stat.ndim != 2:
        raise ValueError("edges are computed on a 2-D slice")
    if sign_mode is SignMode.POS:
        magnitude = stat.clamp(min=0.0)
    elif sign_mode is SignMode.NEG:
        magnitude = (-stat).clamp(min=0.0)
    else:
        magnitude = stat.abs()
    passed = magnitude >= abs(float(threshold))

    interior = torch.ones_like(passed)
    # Shifted comparisons rather than a convolution: this stays exact for a
    # boolean field and costs one pass per direction.
    interior[:-1, :] &= passed[1:, :]
    interior[1:, :] &= passed[:-1, :]
    interior[:, :-1] &= passed[:, 1:]
    interior[:, 1:] &= passed[:, :-1]
    return passed & ~interior


def composite(layers: list[tuple[Tensor, Tensor]], *, background: float = 0.0) -> Tensor:
    """Source-over blend of ``(rgb, alpha)`` pairs, bottom layer first.

    Returns ``(..., 3)`` RGB. Standard premultiplied source-over, so a stack of
    partially transparent statistic maps composites the way a painter would
    expect rather than the way a max-blend would.
    """
    if not layers:
        raise ValueError("nothing to composite")
    rgb0, _ = layers[0]
    out = torch.full_like(rgb0, float(background))
    for rgb, alpha in layers:
        a = alpha.unsqueeze(-1) if alpha.ndim == rgb.ndim - 1 else alpha
        out = rgb * a + out * (1.0 - a)
    return out.clamp(0.0, 1.0)


def to_rgba8(rgb: Tensor, alpha: Tensor | None = None) -> Tensor:
    """Pack float RGB (and optional alpha) into ``uint8`` RGBA for upload."""
    rgb8 = (rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    if alpha is None:
        a8 = torch.full(rgb8.shape[:-1] + (1,), 255, dtype=torch.uint8, device=rgb8.device)
    else:
        a8 = (alpha.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).unsqueeze(-1)
    return torch.cat([rgb8, a8], dim=-1)
