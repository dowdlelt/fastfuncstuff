"""Brain montages, edge overlays and flicker PNGs, drawn with numpy + PIL.

Slice montages never go through matplotlib: a montage is an array — window each
panel, look up a colour table, composite, write — and doing that directly is
~10 ms per image against a few hundred for a matplotlib figure. Matplotlib is
kept for traces and curves (:mod:`fastfuncstuff.qc.plots`); both end in
:func:`compose`, which adds the same header and caption band to every PNG.

Volumes arrive in the toolbox's array order ``(z, y, x)`` with their NIfTI
affine and are put in RAS order first (:func:`ras_volume`), so orientation comes
from the header and never from storage order.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import style

# ---------------------------------------------------------------------------
# Planes on a RAS-ordered volume
# ---------------------------------------------------------------------------

# Multi-planar cut geometry for QC montages and the denoising PC figures, in RAS+ index order
# (axis 0 = R, 1 = A, 2 = S), which is what `termvis.to_ras` hands back. Volumes
# must be reoriented through the affine first: a dataset written LPI and one
# written RAI are the same brain, and only the header says which is which -- the
# old index-space specs drew whichever of upside-down, mirrored or rotated the
# input happened to imply.
#
# Display convention is AFNI's, and matches `viz/slices.py`: radiological, so
# subject right is on the left of the axial and coronal panels; superior is up on
# the sagittal and coronal, anterior up on the axial and to the right on the
# sagittal. Panels are drawn with origin="upper".
#
# Fields: (name, cut axis, row axis, col axis, flip rows, flip cols, left letter).
PLANE_SPECS = (
    ("Sagittal", 0, 2, 1, True, False, "P"),
    ("Coronal", 1, 2, 0, True, True, "R"),
    ("Axial", 2, 1, 0, True, True, "R"),
)
PLANES = {spec[0].lower(): spec for spec in PLANE_SPECS}


def plane_slice(
    vol: np.ndarray, cut_ax: int, idx: int, row_ax: int, flip_rows: bool, flip_cols: bool
) -> np.ndarray:
    """One 2-D cut of a RAS-ordered volume, laid out for display."""
    cut = np.take(vol, idx, axis=cut_ax)
    # np.take drops cut_ax, so the surviving axes keep their relative order.
    remaining = [a for a in range(3) if a != cut_ax]
    img = cut if remaining[0] == row_ax else cut.T
    return img[:: -1 if flip_rows else 1, :: -1 if flip_cols else 1]


def cut_indices(mask_vol: np.ndarray | None, cut_ax: int, n_cut: int, extent: int) -> np.ndarray:
    """Evenly spaced cuts inside the mask's extent along `cut_ax`.

    Fractions of the mask extent rather than a fixed inset from each end, so a
    thin slab (a 12-slice slab has no interior left after a 5-voxel inset) and a
    whole brain both land on interior cuts.
    """
    lo, hi = 0, extent - 1
    if mask_vol is not None:
        present = np.where(mask_vol.any(axis=tuple(a for a in range(3) if a != cut_ax)))[0]
        if present.size:
            lo, hi = int(present.min()), int(present.max())

    span = hi - lo
    fracs = (np.arange(n_cut) + 1.0) / (n_cut + 1.0)
    return np.clip(np.round(lo + fracs * span).astype(int), 0, extent - 1)


def mask_bbox(mask_vol: np.ndarray, pad: int = 2) -> tuple[slice, slice, slice]:
    """Tightest box containing the mask, padded by `pad` voxels per side.

    Without this a montage draws the whole acquisition box, and on a slab the
    empty surround is most of the picture.
    """
    box = []
    for ax in range(3):
        present = np.where(mask_vol.any(axis=tuple(a for a in range(3) if a != ax)))[0]
        if present.size:
            lo = max(0, int(present.min()) - pad)
            hi = min(mask_vol.shape[ax], int(present.max()) + 1 + pad)
        else:
            lo, hi = 0, mask_vol.shape[ax]
        box.append(slice(lo, hi))
    return box[0], box[1], box[2]


def montage_2d(cuts: list[np.ndarray], pad: int = 2) -> np.ndarray:
    """Lay cuts of one plane side by side with a NaN gutter between them."""
    if len(cuts) == 1:
        return cuts[0]
    gutter = np.full((cuts[0].shape[0], pad), np.nan)
    pieces: list[np.ndarray] = []
    for i, cut in enumerate(cuts):
        if i:
            pieces.append(gutter)
        pieces.append(cut)
    return np.hstack(pieces)


def ras_volume(vol_zyx, affine) -> tuple[np.ndarray, tuple[float, float, float]]:
    """A ``(z, y, x)`` array (numpy or torch) in RAS index order, with its zooms."""
    from fastfuncstuff.termvis import to_ras

    arr = vol_zyx.detach().cpu().numpy() if hasattr(vol_zyx, "detach") else np.asarray(vol_zyx)
    arr = np.asarray(arr, dtype=np.float32).transpose(2, 1, 0)
    aff = np.eye(4) if affine is None else np.asarray(affine, dtype=float)
    out, zooms = to_ras(arr, aff)
    return np.ascontiguousarray(out), zooms


class Grid:
    """Where to cut: one set of slice positions shared by every image on a grid.

    Built once from a mask (or the first image), then applied to each volume so a
    mean, its tSNR and an overlay are cut at exactly the same places.
    """

    def __init__(self, ras_ref: np.ndarray, zooms, mask: np.ndarray | None = None, n_cut: int = 7):
        self.zooms = tuple(float(z) for z in zooms)
        self.mask = mask if mask is not None and mask.any() else None
        self.crop = (
            mask_bbox(self.mask)
            if self.mask is not None
            else tuple(slice(0, n) for n in ras_ref.shape)
        )
        shape = tuple(s.stop - s.start for s in self.crop)
        cropped_mask = self.mask[self.crop] if self.mask is not None else None
        self.cuts = {
            name: cut_indices(cropped_mask, spec[1], n_cut, shape[spec[1]])
            for name, spec in PLANES.items()
        }
        self.n_cut = n_cut

    def slices(self, ras_vol: np.ndarray, plane: str) -> list[np.ndarray]:
        _, cut_ax, row_ax, _, fr, fc, _ = PLANES[plane]
        v = ras_vol[self.crop]
        return [plane_slice(v, cut_ax, int(i), row_ax, fr, fc) for i in self.cuts[plane]]

    def aspect(self, plane: str) -> tuple[float, float]:
        """mm per display row and per display column for a plane."""
        _, _, row_ax, col_ax, _, _, _ = PLANES[plane]
        return self.zooms[row_ax], self.zooms[col_ax]

    def side_letter(self, plane: str) -> str:
        return PLANES[plane][6]


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def window(vol: np.ndarray, mask: np.ndarray | None = None, lo=1.0, hi=99.5) -> tuple[float, float]:
    """Robust display range from the in-mask (or nonzero) values."""
    v = vol[mask] if mask is not None and mask.any() else vol[np.isfinite(vol) & (vol != 0)]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(v, [lo, hi])
    if b <= a:
        b = a + 1.0
    return float(a), float(b)


@lru_cache(maxsize=16)
def lut(name: str) -> np.ndarray:
    """A 256x3 uint8 colour table from a matplotlib colormap name, or 'gray'."""
    if name == "gray":
        g = np.arange(256, dtype=np.uint8)
        return np.stack([g, g, g], axis=1)
    import matplotlib

    cmap = matplotlib.colormaps[name]
    return (cmap(np.linspace(0, 1, 256))[:, :3] * 255).round().astype(np.uint8)


def colorize(img: np.ndarray, lo: float, hi: float, cmap: str = "gray") -> np.ndarray:
    """Window + colour table -> (H, W, 3) uint8. NaN draws as the surface colour."""
    finite = np.isfinite(img)
    t = np.clip((np.where(finite, img, lo) - lo) / (hi - lo), 0, 1)
    rgb = lut(cmap)[(t * 255).astype(np.uint8)]
    if not finite.all():
        rgb[~finite] = style.hex_rgb(style.SURFACE)
    return rgb


def blend(base: np.ndarray, over: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    return (base.astype(np.float32) * (1 - a) + over.astype(np.float32) * a).astype(np.uint8)


def edges_2d(img: np.ndarray, aspect: tuple[float, float], threshold: float = 0.15) -> np.ndarray:
    """Thin edges of one displayed slice (boolean), from the shared edge detector.

    Edges are found in the displayed plane, not cut from a 3-D edge volume: a
    surface running parallel to the slice would otherwise show as a filled patch.
    """
    import torch

    from fastfuncstuff.processing.edges import edge_map

    t = torch.from_numpy(np.nan_to_num(np.ascontiguousarray(img), nan=0.0).astype(np.float32))
    if t.shape[0] < 4 or t.shape[1] < 4:
        return np.zeros(img.shape, dtype=bool)
    e = edge_map(t, sigma=1.0, spacing=aspect, threshold=threshold)
    return e.numpy() > 0


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def to_image(rgb: np.ndarray):
    from PIL import Image

    return Image.fromarray(np.ascontiguousarray(rgb), "RGB")


def panel(rgb: np.ndarray, aspect: tuple[float, float], height: int, smooth: bool = True):
    """Resize a slice to `height` pixels at its true physical aspect ratio."""
    from PIL import Image

    rows, cols = rgb.shape[:2]
    mm_h, mm_w = rows * aspect[0], cols * aspect[1]
    width = max(1, round(height * mm_w / mm_h))
    resample = Image.Resampling.BILINEAR if smooth else Image.Resampling.NEAREST
    return to_image(rgb).resize((width, height), resample)


def _text(draw, xy, s, size, fill=style.INK, bold=False, anchor="la"):
    draw.text(xy, s, font=style.pil_font(size, bold), fill=fill, anchor=anchor)


def montage(
    rows: Sequence[tuple[str, Sequence]],
    width: int = style.WIDTH,
    gutter: int = 4,
    label_w: int = 120,
    side_letters: Sequence[str] | None = None,
):
    """Stack labelled rows of panels into one image, every row scaled to fill `width`.

    ``rows`` is ``[(row label, [PIL panels]), ...]``. Panels in a row share a height
    (as :func:`panel` makes them); the row is scaled as a unit so physical aspect
    survives.
    """
    from PIL import Image, ImageDraw

    avail = width - label_w - 16
    scaled = []
    for label, panels in rows:
        total = sum(p.width for p in panels) + gutter * (len(panels) - 1)
        scale = avail / max(total, 1)
        h = max(1, round(panels[0].height * scale))
        # floor, not round: rounding every panel up can push a row past the canvas edge.
        ps = [
            p.resize((max(1, int(p.width * scale)), h), Image.Resampling.BILINEAR) for p in panels
        ]
        scaled.append((label, ps, h))
    height = sum(h for _, _, h in scaled) + gutter * (len(scaled) + 1)
    canvas = Image.new("RGB", (width, height), style.SURFACE)
    draw = ImageDraw.Draw(canvas)
    y = gutter
    for r, (label, ps, h) in enumerate(scaled):
        lines = label.split("\n")
        ly = y + h // 2 - 9 * len(lines)
        for i, ln in enumerate(lines):
            _text(
                draw,
                (16, ly + 18 * i),
                ln,
                14 if i == 0 else 12,
                style.INK if i == 0 else style.MUTED,
                bold=i == 0,
            )
        x = label_w
        for p in ps:
            canvas.paste(p, (x, y))
            x += p.width + gutter
        if side_letters is not None and side_letters[r]:
            _text(draw, (label_w + 4, y + 4), side_letters[r], 12, style.MUTED)
        y += h + gutter
    return canvas


def compose(
    body,
    title: str,
    subtitle: str = "",
    stats: Sequence[tuple[str, str | None]] = (),
    caption: str = "",
    width: int = style.WIDTH,
):
    """Header band (title, run, headline numbers) + body + 'look for' caption.

    ``stats`` is ``[(text, status)]`` with status in ``{None, 'ok', 'warn', 'fail'}``;
    a status draws a coloured dot beside the text, never colour alone.
    """
    from PIL import Image, ImageDraw

    if body.width != width:
        h = round(body.height * width / body.width)
        body = body.resize((width, h), Image.Resampling.LANCZOS)
    cap_lines = _wrap(caption, width - 48, 13) if caption else []
    cap_h = (18 * len(cap_lines) + 2 * style.FOOTER_PAD) if cap_lines else style.FOOTER_PAD
    canvas = Image.new("RGB", (width, style.HEADER_H + body.height + cap_h), style.SURFACE)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, style.HEADER_H - 1), fill=style.PANEL)
    _text(draw, (24, 16), title, 24, bold=True)
    if subtitle:
        _text(draw, (24, 52), subtitle, 14, style.INK2)
    x = width - 24
    for text, status in reversed(list(stats)):
        f = style.pil_font(15, True)
        tw = draw.textlength(text, font=f)
        x -= tw
        draw.text((x, 24), text, font=f, fill=style.INK)
        if status:
            draw.ellipse((x - 18, 28, x - 6, 40), fill=style.STATUS[status])
            x -= 18
        x -= 28
    canvas.paste(body, (0, style.HEADER_H))
    y = style.HEADER_H + body.height + style.FOOTER_PAD
    for i, ln in enumerate(cap_lines):
        _text(draw, (24, y + 18 * i), ln, 13, style.INK2 if i else style.INK2)
    return canvas


def _wrap(text: str, max_w: int, size: int) -> list[str]:
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    f = style.pil_font(size)
    out: list[str] = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            trial = f"{line} {word}".strip()
            if draw.textlength(trial, font=f) <= max_w or not line:
                line = trial
            else:
                out.append(line)
                line = word
        out.append(line)
    return out


def figure_image(fig):
    """A matplotlib Figure (Agg canvas, no pyplot) as a PIL image."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return Image.fromarray(rgba[..., :3].copy(), "RGB")


def save_png(img, path: str | Path) -> Path:
    path = Path(path)
    # compress_level 3: ~3x faster than PIL's default 6 for a few percent more bytes.
    img.save(path, format="PNG", compress_level=3)
    return path


def save_apng(frames: Sequence, path: str | Path, duration_ms: int | Sequence[int] = 700) -> Path:
    """An animated PNG. Browsers and most viewers animate it; the rest show frame 1,
    which is therefore always the reference image."""
    path = Path(path)
    frames[0].save(
        path,
        format="PNG",
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration_ms,
        loop=0,
        compress_level=3,
        disposal=0,
        blend=0,
    )
    return path


def stat_status(flagged: bool) -> str:
    return "warn" if flagged else "ok"
