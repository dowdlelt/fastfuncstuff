"""Time-series QC figures (motion, outliers, carpet), drawn with matplotlib Agg.

Figures are built on ``matplotlib.figure.Figure`` directly — no pyplot, so no
global figure registry and no GUI backend — and handed back as PIL images for
:func:`fastfuncstuff.qc.render.compose` to frame. Every panel shares one time
axis; measures of different units get their own panel, never a second y-axis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import style
from .render import figure_image

DPI = 100


@dataclass
class Trace:
    """One line on a shared time axis."""

    name: str
    values: np.ndarray
    color: str = style.BLUE
    limit: float | None = None  # a dashed reference line (a censor limit)


def _time_axis(n: int, tr: float | None) -> tuple[np.ndarray, str]:
    if tr and tr > 0:
        return np.arange(n) * tr, "time (s)"
    return np.arange(n, dtype=float), "volume"


def _shade_censored(ax, t: np.ndarray, keep: np.ndarray | None) -> None:
    """Shade each run of censored time points as one band."""
    if keep is None:
        return
    bad = np.asarray(keep) == 0
    if not bad.any():
        return
    dt = t[1] - t[0] if t.size > 1 else 1.0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], bad.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2], strict=True):
        ax.axvspan(t[a] - dt / 2, t[b - 1] + dt / 2, color=style.CENSOR_FILL, lw=0, zorder=0)


def _panel(ax, t, traces: Sequence[Trace], ylabel: str, keep) -> None:
    _shade_censored(ax, t, keep)
    for tr in traces:
        ax.plot(t, tr.values, color=tr.color, lw=1.4, label=tr.name)
        if tr.limit is not None:
            ax.axhline(tr.limit, color=style.INK2, lw=1.0, ls=(0, (4, 3)))
            ax.annotate(
                f"limit {tr.limit:g}",
                xy=(1.0, tr.limit),
                xycoords=("axes fraction", "data"),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=style.INK2,
            )
    ax.set_ylabel(ylabel)
    ax.set_xlim(t[0], t[-1] if t.size > 1 else t[0] + 1)
    if len(traces) > 1:
        # Direct labels at the right end (<= 4 series) plus the legend.
        ax.legend(loc="upper left", ncol=len(traces), handlelength=1.6, borderaxespad=0.2)


def stacked_traces(
    panels: Sequence[tuple[str, Sequence[Trace]]],
    tr: float | None = None,
    keep: np.ndarray | None = None,
    panel_h: float = 1.55,
    width_px: int = style.WIDTH,
):
    """A column of panels on one time axis; ``panels`` is ``[(ylabel, traces)]``."""
    from matplotlib.figure import Figure

    n = len(panels[0][1][0].values)
    t, xlabel = _time_axis(n, tr)
    with style.mpl_style():
        fig = Figure(figsize=(width_px / DPI, panel_h * len(panels) + 0.6), dpi=DPI)
        axes = fig.subplots(len(panels), 1, sharex=True, squeeze=False)[:, 0]
        for ax, (ylabel, traces) in zip(axes, panels, strict=True):
            _panel(ax, t, traces, ylabel, keep)
        axes[-1].set_xlabel(xlabel)
        fig.subplots_adjust(
            left=0.075,
            right=0.93,
            top=0.97,
            bottom=0.42 / (panel_h * len(panels) + 0.6) + 0.05,
            hspace=0.18,
        )
        return figure_image(fig)


def carpet(
    rows: np.ndarray,
    groups: Sequence[tuple[str, int, int]],
    traces: Sequence[Trace],
    tr: float | None = None,
    keep: np.ndarray | None = None,
    width_px: int = style.WIDTH,
    vlim: float = 2.0,
):
    """Carpet plot: voxels (rows, z-scored) x time, under traces on the same axis.

    ``groups`` names contiguous row blocks ``(label, start, stop)`` — tissue
    classes, or slices — drawn as a labelled band down the left edge.
    """
    from matplotlib.figure import Figure

    n = rows.shape[1]
    t, xlabel = _time_axis(n, tr)
    nt = len(traces)
    with style.mpl_style():
        fig = Figure(figsize=(width_px / DPI, 1.15 * nt + 5.2), dpi=DPI)
        gs = fig.add_gridspec(nt + 1, 1, height_ratios=[1.0] * nt + [4.6], hspace=0.12)
        ax_c = fig.add_subplot(gs[nt])
        axes = [fig.add_subplot(gs[i], sharex=ax_c) for i in range(nt)]
        for ax, trc in zip(axes, traces, strict=True):
            _panel(ax, t, [trc], trc.name, keep)
            ax.tick_params(labelbottom=False)
        dt = t[1] - t[0] if n > 1 else 1.0
        ax_c.imshow(
            rows,
            aspect="auto",
            cmap="gray",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
            extent=(t[0] - dt / 2, t[-1] + dt / 2, rows.shape[0], 0),
        )
        ax_c.grid(False)
        ax_c.set_yticks([])
        ax_c.set_xlabel(xlabel)
        for i, (label, a, b) in enumerate(groups):
            color = style.SERIES[i % len(style.SERIES)]
            ax_c.plot(
                [-0.008, -0.008],
                [a, b],
                transform=ax_c.get_yaxis_transform(),
                color=color,
                lw=6,
                clip_on=False,
                solid_capstyle="butt",
            )
            ax_c.annotate(
                label,
                xy=(-0.014, (a + b) / 2),
                xycoords=("axes fraction", "data"),
                ha="right",
                va="center",
                fontsize=10,
                color=style.INK2,
            )
            if i:
                ax_c.axhline(a, color=style.SURFACE, lw=1.5)
        if keep is not None and (np.asarray(keep) == 0).any():
            bad = np.flatnonzero(np.asarray(keep) == 0)
            ax_c.plot(
                t[bad],
                np.full(bad.size, -0.5),
                "|",
                color=style.STATUS["fail"],
                ms=6,
                clip_on=False,
                transform=ax_c.transData,
            )
        fig.subplots_adjust(left=0.09, right=0.93, top=0.98, bottom=0.07)
        return figure_image(fig)


def _zscore_detrended(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    tt = np.linspace(-1, 1, n, dtype=np.float32)
    basis = np.stack([np.ones_like(tt), tt, 1.5 * tt**2 - 0.5], axis=1)
    coef, *_ = np.linalg.lstsq(basis, x, rcond=None)
    x = x - basis @ coef
    sd = x.std(axis=0)
    return x / np.where(sd > 0, sd, 1.0)


def carpet_rows(
    series_tv: np.ndarray,
    groups: Sequence[tuple[str, np.ndarray]],
    max_rows: int = 700,
) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Carpet rows from ``(time, voxels)`` data and named voxel groups.

    Each voxel is detrended (linear + quadratic) and z-scored, or slow drift
    dominates the grey scale and hides the bands (spikes, slice dropouts,
    respiration) a carpet exists to show. Groups share ``max_rows`` in proportion
    to their size; within a group, neighbouring voxels (in the given order) are
    averaged into one row and re-z-scored, so a 100k-voxel brain draws as a few
    hundred legible rows without dropping any voxel.

    Returns the ``(rows, time)`` image and ``(label, start, stop)`` row blocks.
    """
    x = np.asarray(series_tv, dtype=np.float32)
    total = sum(len(idx) for _, idx in groups)
    out, blocks, start = [], [], 0
    for label, idx in groups:
        if len(idx) == 0:
            continue
        z = _zscore_detrended(x[:, idx])
        n_rows = max(1, min(len(idx), round(max_rows * len(idx) / max(total, 1))))
        bounds = np.linspace(0, len(idx), n_rows + 1).astype(int)
        rows = np.add.reduceat(z, bounds[:-1], axis=1) / np.diff(bounds)
        sd = rows.std(axis=0)
        rows = (rows - rows.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
        out.append(rows.T)
        blocks.append((label, start, start + n_rows))
        start += n_rows
    return np.concatenate(out, axis=0), blocks
