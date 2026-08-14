"""Render a volume as greyscale images *in the terminal*.

Two half-height pixels stack into one character cell (``▀`` with a foreground
colour for the top pixel and a background colour for the bottom), which is what
makes terminal pixels roughly square and the anatomy recognisable. Falls back to
an ASCII ramp when the terminal can't do 24-bit colour.

This is a sanity-check view — "did I just write a brain, and is it the right way
up" — not a viewer. It exists so a header dump can also answer the question the
header can't.
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np

# Dark → light. Trailing '@' reads as solid in most terminal themes.
_ASCII_RAMP = " .:-=+*#%@"

_UPPER_HALF = "▀"


def supports_truecolor(stream=None) -> bool:
    """True when it's worth emitting 24-bit colour escapes."""
    stream = stream or sys.stdout
    if not getattr(stream, "isatty", lambda: False)():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return True
    return os.environ.get("TERM", "") not in ("", "dumb")


def _normalize(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip to [lo, hi] and scale to 0–255 uint8."""
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    out = (np.clip(img, lo, hi) - lo) / (hi - lo)
    return (out * 255.0 + 0.5).astype(np.uint8)


def _resize_nn(img: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Nearest-neighbour resample — no SciPy, and blockiness is honest here."""
    rows, cols = max(rows, 1), max(cols, 1)
    r = np.minimum((np.arange(rows) * img.shape[0]) // rows, img.shape[0] - 1)
    c = np.minimum((np.arange(cols) * img.shape[1]) // cols, img.shape[1] - 1)
    return img[np.ix_(r, c)]


def _panel_lines(img8: np.ndarray, color: bool) -> list[str]:
    """One panel of an already-sized uint8 image → terminal lines.

    Rows are consumed in pairs; an odd final row pairs with black. In colour mode
    each cell is ``▀`` with fg = upper pixel and bg = lower pixel, so a cell
    carries two pixels and the aspect stays square.
    """
    h, w = img8.shape
    if h % 2:
        img8 = np.vstack([img8, np.zeros((1, w), dtype=np.uint8)])
        h += 1

    lines: list[str] = []
    if not color:
        # One character per *pair* of rows: average them so nothing is dropped.
        pairs = img8.reshape(h // 2, 2, w).mean(axis=1)
        idx = (pairs / 256.0 * len(_ASCII_RAMP)).astype(int).clip(0, len(_ASCII_RAMP) - 1)
        return ["".join(_ASCII_RAMP[i] for i in row) for row in idx]

    for r in range(0, h, 2):
        top, bot = img8[r], img8[r + 1]
        parts = [
            f"\x1b[38;2;{t};{t};{t}m\x1b[48;2;{b};{b};{b}m{_UPPER_HALF}"
            for t, b in zip(top.tolist(), bot.tolist(), strict=True)
        ]
        lines.append("".join(parts) + "\x1b[0m")
    return lines


def _pad(lines: list[str], rows: int, width: int, color: bool) -> list[str]:
    blank = ("\x1b[0m" + " " * width) if color else " " * width
    return list(lines) + [blank] * (rows - len(lines))


def orthoview(
    vol: np.ndarray,
    zooms: tuple[float, float, float],
    *,
    slices: tuple[int, int, int] | None = None,
    width: int | None = None,
    color: bool = True,
    clip: tuple[float, float] = (1.0, 99.0),
    labels: bool = True,
) -> str:
    """Three orthogonal greyscale planes, side by side, as printable text.

    ``vol`` must be in RAS+ index order (``i``→Right, ``j``→Anterior,
    ``k``→Superior); :func:`to_ras` gets you there from an arbitrary affine. The
    panels are scaled by ``zooms`` so a 1×1×4 mm acquisition looks like one.

    Radiological-free convention: subject left is on the **right** of each
    panel, anterior is up on the axial, superior is up on the other two. Corner
    letters say so explicitly, because every neuroimager has been burned once.
    """
    vol = np.asarray(vol, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"orthoview needs a 3D volume, got shape {vol.shape}")

    nx, ny, nz = vol.shape
    i0, j0, k0 = slices or (nx // 2, ny // 2, nz // 2)
    i0, j0, k0 = (
        int(np.clip(i0, 0, nx - 1)),
        int(np.clip(j0, 0, ny - 1)),
        int(np.clip(k0, 0, nz - 1)),
    )

    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return "(no finite voxels to display)"
    lo, hi = (float(v) for v in np.percentile(finite, clip))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())

    dx, dy, dz = (float(z) or 1.0 for z in zooms)

    # Each panel: (image[row, col], mm per row/col, caption).
    # Rows are built top-down, so the "up" direction is flipped out of index order,
    # and columns are flipped where subject-left must land on the right.
    panels = [
        (vol[i0, :, :].T[::-1, :], (dz, dy), f"sag i={i0} ↑S →A"),
        (vol[:, j0, :].T[::-1, ::-1], (dz, dx), f"cor j={j0} ↑S →L"),
        (vol[:, :, k0].T[::-1, ::-1], (dy, dx), f"axi k={k0} ↑A →L"),
    ]

    # An explicit width is honoured as given; otherwise fit the terminal.
    term = shutil.get_terminal_size((100, 30)).columns
    total = max(30, width or (term - 2))
    gap = 2
    avail = total - gap * (len(panels) - 1)

    # Split the available columns by physical width so the three panels share a
    # single mm-per-character scale; rows then follow from the same scale, with
    # the ×2 because a character cell holds two stacked pixels.
    phys_w = [img.shape[1] * mm_col for img, (_, mm_col), _ in panels]
    scale = avail / sum(phys_w)
    cols = [max(8, int(round(w * scale))) for w in phys_w]
    rows = [
        max(4, int(round(img.shape[0] * mm_row * scale / 2.0))) for img, (mm_row, _), _ in panels
    ]
    n_rows = max(rows)

    rendered: list[list[str]] = []
    for (img, _, caption), c, r in zip(panels, cols, rows, strict=True):
        img8 = _normalize(_resize_nn(img, r * 2, c), lo, hi)
        # Pad to the tallest panel *before* the caption so all captions line up.
        lines = _pad(_panel_lines(img8, color), n_rows, c, color)
        if labels:
            text = caption if len(caption) <= c else caption[:c]
            lines.append((f"\x1b[2m{text.ljust(c)}\x1b[0m") if color else text.ljust(c))
        rendered.append(lines)

    sep = " " * gap
    return "\n".join(sep.join(parts) for parts in zip(*rendered, strict=True))


def to_ras(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Reorient array+affine to RAS+ index order and return it with its zooms.

    Display code should never guess at storage order: a dataset written LPI and
    one written RAI are the same brain, and only the affine says which is which.
    """
    import nibabel as nib

    ornt = nib.orientations.io_orientation(affine)
    out = nib.orientations.apply_orientation(data, ornt)
    zooms = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    # io_orientation's first column maps each *input* axis to the output slot it
    # lands in, so the zooms are scattered, not gathered — reading it the other
    # way around silently draws anisotropic data at the wrong aspect ratio.
    zoom_out = [0.0, 0.0, 0.0]
    for in_axis, out_slot in enumerate(ornt[:, 0].astype(int)):
        zoom_out[out_slot] = float(zooms[in_axis])
    return out, (zoom_out[0], zoom_out[1], zoom_out[2])
