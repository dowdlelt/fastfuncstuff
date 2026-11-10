"""One visual system for every QC image: dark surface, fixed series order.

Brain images read best on a dark surface, and a single surface for traces too
keeps a scrolled folder from flashing between themes. Colours are the dark-mode
steps of the validated reference palette (categorical order is the CVD-safety
mechanism: assign in order, never cycle); status colours are reserved for
pass/warn/fail and always carry a word beside them.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

SURFACE = "#1a1a19"
PANEL = "#20201f"
GRID = "#383835"
INK = "#f0efec"
INK2 = "#c3c2b7"
MUTED = "#8f8e88"

# Dark categorical steps, in their validated order.
SERIES = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES

STATUS = {"ok": "#0ca30c", "warn": "#fab219", "fail": "#d03b3b"}

# Shading for excluded time points (censored TRs) — a recessive band, not a colour
# that competes with the traces.
CENSOR_FILL = "#5a2a2a"

# Pixel geometry shared by montages and traces, so every PNG in a folder has the
# same width and the header bands line up when scrolling.
WIDTH = 1600
HEADER_H = 92
FOOTER_PAD = 14


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@lru_cache(maxsize=4)
def font_path(bold: bool = False) -> str | None:
    """DejaVu Sans from matplotlib's own data directory: always installed with it,
    and importing the top-level package is cheap (no pyplot, no backend)."""
    try:
        import matplotlib

        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
        return str(p) if p.exists() else None
    except Exception:
        return None


@lru_cache(maxsize=32)
def pil_font(size: int, bold: bool = False):
    from PIL import ImageFont

    path = font_path(bold)
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(path, size)


_MPL_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": None,  # filled in below
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.color": INK,
    "font.family": "DejaVu Sans",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": INK2,
    "lines.linewidth": 1.4,
    "lines.solid_capstyle": "round",
}


@contextmanager
def mpl_style():
    """Apply the QC theme to matplotlib for the duration of one figure."""
    import matplotlib as mpl
    from cycler import cycler

    rc = dict(_MPL_RC)
    rc["axes.prop_cycle"] = cycler(color=list(SERIES))
    with mpl.rc_context(rc):
        yield
