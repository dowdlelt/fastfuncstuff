"""``ffs_fc -plot``: the correlation matrix, seriated, and the co-fluctuation trace.

Unlike a brain-map diagnostic, a correlation matrix's colours *are* a measurement:
the scale is fixed at -1..1 with a colour bar so two subjects' figures compare.
The order is the same seriation the viewer's matrix window uses
(:func:`connectivity.order.seriate`), so the figure and the window show one picture.
Palette and axis style come from :mod:`fastfuncstuff.dynamics.plots`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

#: Above this many ROIs the row labels are unreadable; the module lines carry it.
MAX_LABELLED = 60


def _major_boundaries(blocks: Sequence[int], n: int, min_size: int) -> list[int]:
    """Module boundaries with a module of at least ``min_size`` on both sides.

    The dendrogram cut leaves outlier ROIs as one-member modules, and drawing a line
    round each buries the real networks (360 Glasser ROIs drew ~60 lines). Small
    modules are folded into their neighbours for drawing only.
    """
    edges = [0, *blocks, n]
    starts = [edges[i] for i in range(len(edges) - 1) if edges[i + 1] - edges[i] >= min_size]
    # A line at the start of every major module after the first; the small modules
    # between two major ones fold into the earlier.
    return starts[1:]


def plot_fc(
    path: str | Path,
    r: np.ndarray,
    names: Sequence[str],
    *,
    rss: np.ndarray | None = None,
    times: np.ndarray | None = None,
    title: str = "",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from fastfuncstuff.connectivity.order import seriate
    from fastfuncstuff.dynamics.plots import _INK, _INK2, _SURFACE, DIVERGING, _style_axes

    order, all_blocks = seriate(r)
    shown = np.asarray(r)[order][:, order]
    n = shown.shape[0]
    blocks = _major_boundaries(all_blocks, n, max(3, n // 50))
    tall = 7.0 if rss is None else 9.4
    fig = plt.figure(figsize=(7.6, tall), facecolor=_SURFACE)
    if rss is None:
        ax = fig.add_axes((0.12, 0.1, 0.72, 0.8))
    else:
        ax = fig.add_axes((0.12, 0.3, 0.72, 0.62))
    im = ax.imshow(shown, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
    for b in blocks:
        ax.axhline(b - 0.5, color=_INK, lw=0.6)
        ax.axvline(b - 0.5, color=_INK, lw=0.6)
    if n <= MAX_LABELLED:
        labels = [str(names[i]) for i in order]
        ax.set_xticks(range(n), labels, rotation=90, fontsize=6)
        ax.set_yticks(range(n), labels, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{n} ROIs, clustered order", color=_INK2)
    ax.set_title(title or f"Correlation, {n} ROIs ({len(blocks) + 1} major modules)", color=_INK)
    cax = fig.add_axes((0.86, 0.3 if rss is not None else 0.1, 0.025, 0.2))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("r", color=_INK2)
    cb.ax.tick_params(colors=_INK2, labelsize=7)

    if rss is not None:
        ax2 = fig.add_axes((0.12, 0.06, 0.72, 0.16))
        t = np.arange(rss.size) if times is None else np.asarray(times)
        ax2.plot(t, rss, color="#2a78d6", lw=0.8)
        ax2.set_ylabel("RSS", fontsize=8)
        ax2.set_xlabel("time (s)" if times is not None else "kept TR", fontsize=8)
        ax2.set_title("Edge co-fluctuation amplitude", fontsize=9, color=_INK)
        _style_axes(ax2)
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    return str(path)
