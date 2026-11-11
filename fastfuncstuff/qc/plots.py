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
    rows: np.ndarray | Sequence[tuple[str, np.ndarray]],
    groups: Sequence[tuple[str, int, int]],
    traces: Sequence[Trace],
    tr: float | None = None,
    keep: np.ndarray | None = None,
    width_px: int = style.WIDTH,
    vlim: float = 2.0,
    run_starts: Sequence[int] | None = None,
):
    """Carpet plot: voxels (rows, z-scored) x time, under traces on the same axis.

    ``groups`` names contiguous row blocks ``(label, start, stop)`` — tissue
    classes, or slices — drawn as a labelled band down the left edge. ``rows``
    may be several named carpets ``[(name, rows)]`` with the same row blocks:
    they stack on one time axis and ONE grey scale, which is what lets a quiet
    carpet read as quieter than its neighbour. ``run_starts`` draws a line at
    each run boundary of a concatenated series.
    """
    from matplotlib.figure import Figure

    stack = [("", rows)] if isinstance(rows, np.ndarray) else list(rows)
    n = stack[0][1].shape[1]
    t, xlabel = _time_axis(n, tr)
    nt = len(traces)
    nc = len(stack)
    carpet_h = 4.6 if nc == 1 else 2.7
    with style.mpl_style():
        fig = Figure(figsize=(width_px / DPI, 1.15 * nt + carpet_h * nc + 0.6), dpi=DPI)
        gs = fig.add_gridspec(
            nt + nc, 1, height_ratios=[1.0] * nt + [carpet_h] * nc, hspace=0.12 if nc == 1 else 0.1
        )
        carpets = [fig.add_subplot(gs[nt])]
        carpets += [fig.add_subplot(gs[nt + k], sharex=carpets[0]) for k in range(1, nc)]
        axes = [fig.add_subplot(gs[i], sharex=carpets[0]) for i in range(nt)]
        for ax, trc in zip(axes, traces, strict=True):
            _panel(ax, t, [trc], trc.name, keep)
            ax.tick_params(labelbottom=False)
        dt = t[1] - t[0] if n > 1 else 1.0
        bounds = [t[int(s)] - dt / 2 for s in (run_starts or [])[1:] if 0 < int(s) < n]
        for ax in axes:
            for b in bounds:
                ax.axvline(b, color=style.GRID, lw=1.0)
        for k, (ax_c, (name, img)) in enumerate(zip(carpets, stack, strict=True)):
            ax_c.imshow(
                img,
                aspect="auto",
                cmap="gray",
                vmin=-vlim,
                vmax=vlim,
                interpolation="nearest",
                extent=(t[0] - dt / 2, t[-1] + dt / 2, img.shape[0], 0),
            )
            ax_c.grid(False)
            ax_c.set_yticks([])
            for b in bounds:
                ax_c.axvline(b, color=style.YELLOW, lw=0.9, alpha=0.8)
            if k < nc - 1:
                ax_c.tick_params(labelbottom=False)
            else:
                ax_c.set_xlabel(xlabel)
            if name:
                ax_c.annotate(
                    name,
                    xy=(1.006, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="left",
                    va="center",
                    fontsize=10,
                    color=style.INK,
                )
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
            if k == 0 and keep is not None and (np.asarray(keep) == 0).any():
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
        total_h = 1.15 * nt + carpet_h * nc + 0.6
        fig.subplots_adjust(left=0.09, right=0.93, top=0.98, bottom=0.42 / total_h)
        return figure_image(fig)


def density_scatter(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    lim: tuple[float, float] | None = None,
    markers: Sequence[tuple[str, float, float, str]] = (),
    bins: int = 220,
    width_px: int = style.WIDTH,
):
    """Two per-voxel values against each other as a density, on shared log axes.

    A 2-D histogram drawn as an image, with counts on a log colour scale: a
    million voxels cost one ``histogram2d``, where a scatter of points would
    cost seconds and saturate into a blob. The dashed diagonal is "no change";
    ``markers`` are ``(label, x, y, colour)`` points, typically group medians.
    ``lim`` fixes both axes; by default they zoom to the pooled 0.5-99.5th
    percentiles. The right panel is the distribution of ``y / x``.
    """
    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if lim is None:
        # Zoom to the data: the question is the offset from the diagonal, and on
        # an axis wide enough for every scanner a real cloud is a smudge.
        both = np.concatenate([x[ok], y[ok]]) if ok.any() else np.array([1.0, 10.0])
        lo, hi = np.percentile(both, [0.5, 99.5])
        lo, hi = float(lo) / 1.25, float(hi) * 1.25
    else:
        lo, hi = lim
    edges = np.geomspace(lo, hi, bins + 1)
    h, _, _ = np.histogram2d(np.clip(x[ok], lo, hi), np.clip(y[ok], lo, hi), bins=(edges, edges))
    with style.mpl_style():
        from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

        fig = Figure(figsize=(width_px / DPI, 7.6), dpi=DPI)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.72], wspace=0.22)
        ax = fig.add_subplot(gs[0])
        ax.pcolormesh(
            edges,
            edges,
            np.where(h.T > 0, h.T, np.nan),
            cmap="inferno",
            norm=LogNorm(vmin=1, vmax=max(float(h.max()), 2.0)),
            rasterized=True,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_aspect("equal")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())
        ax.plot([lo, hi], [lo, hi], color=style.INK2, lw=1.0, ls=(0, (4, 3)))
        ax.annotate(
            "no change",
            xy=(hi, hi),
            xytext=(-6, -14),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=style.INK2,
        )
        for label, mx, my, color in markers:
            if mx > 0 and my > 0:
                ax.plot(mx, my, "o", ms=9, mfc=color, mec=style.INK, mew=1.2)
                ax.annotate(
                    label,
                    xy=(mx, my),
                    xytext=(9, -4),
                    textcoords="offset points",
                    fontsize=10,
                    color=style.INK,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{int(ok.sum()):,} voxels (colour: count, log scale)")
        ax2 = fig.add_subplot(gs[1])
        ratio = y[ok] / x[ok]
        p_lo, p_hi = np.percentile(ratio, [0.5, 99.5]) if ratio.size else (0.9, 1.3)
        r_edges = np.linspace(min(0.9, float(p_lo)), max(1.3, float(p_hi)), 121)
        ax2.hist(np.clip(ratio, r_edges[0], r_edges[-1]), bins=r_edges, color=style.BLUE, lw=0)
        ax2.axvline(1.0, color=style.INK2, lw=1.0, ls=(0, (4, 3)))
        for label, mx, my, color in markers:
            if mx > 0 and my > 0:
                ax2.axvline(my / mx, color=color, lw=1.8, label=f"{label} {my / mx:.2f}x")
        if markers:
            ax2.legend(loc="upper right")
        ax2.set_xlabel("ratio (vertical / horizontal)")
        ax2.set_ylabel("voxels")
        ax2.set_title("gain per voxel (ends hold everything beyond them)")
        fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.09)
        return figure_image(fig)


def _zscore_detrended(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    tt = np.linspace(-1, 1, n, dtype=np.float32)
    basis = np.stack([np.ones_like(tt), tt, 1.5 * tt**2 - 0.5], axis=1)
    coef, *_ = np.linalg.lstsq(basis, x, rcond=None)
    x = x - basis @ coef
    sd = x.std(axis=0)
    return x / np.where(sd > 0, sd, 1.0)


CARPET_ORDERS = ("pc", "gs", "none")


@dataclass
class Carpet:
    """A carpet image and what was learned building it."""

    rows: np.ndarray  # (rows, time), in units of the image's robust SD
    blocks: list[tuple[str, int, int]]  # (label, first row, end row)
    pc1: np.ndarray | None  # first principal time course (unit norm), or None
    pc1_frac: float  # variance fraction PC1 explains across the voxels used
    gcor: float  # AFNI GCOR of the same voxels
    scale: float = 1.0  # what one unit of ``rows`` is, in the input's units


def _first_pc(z: np.ndarray) -> tuple[np.ndarray, float]:
    """Top time course of ``z`` (time, voxels) via the time x time Gram matrix.

    The Gram route costs O(T^2 V) and a T x T eigh, where an SVD of the wide
    matrix would cost O(T V^2)-ish memory traffic; T is hundreds, V tens of
    thousands.
    """
    g = (z @ z.T).astype(np.float64)
    w, v = np.linalg.eigh(g)
    tr = float(np.trace(g))
    return v[:, -1].astype(np.float32), float(w[-1] / tr) if tr > 0 else 0.0


def carpet_rows(
    series_tv: np.ndarray,
    groups: Sequence[tuple[str, np.ndarray]],
    max_rows: int = 700,
    order: str = "pc",
    *,
    standardize: bool = True,
    score: np.ndarray | None = None,
    scale: float | None = None,
) -> Carpet:
    """Carpet rows from ``(time, voxels)`` data and named voxel groups.

    Each voxel is detrended (linear + quadratic) and z-scored, or slow drift and
    bright voxels dominate. Then two choices decide whether structure is visible:

    - **Order.** Within each group, voxels are sorted by their loading on the
      first principal time course (``pc``, default) or their correlation with the
      global mean (``gs``), so voxels that move together sit together and a shared
      fluctuation reads as one coherent band, not scattered speckle.
    - **Scale.** Neighbouring voxels (in that order) are averaged into rows, and
      the WHOLE image is divided by one robust SD. Averaging shrinks independent
      noise and keeps what the voxels share, so shared structure stands above a
      dim noise floor. Re-z-scoring each row instead would inflate pure-noise rows
      to the same contrast as structured ones and erase exactly that difference.

    Groups share ``max_rows`` in proportion to their size.

    Three switches serve carpets that must be compared with each other (a
    model's data, fit and residual):

    - ``standardize=False`` keeps the input's own units (already detrended and
      scaled by the caller, e.g. percent signal). z-scoring would give white
      matter and grey matter the same contrast and a residual the same contrast
      as the data it came from.
    - ``score`` (one value per voxel, in group order) replaces the sort key, so
      several carpets share one row order. Highest first.
    - ``scale`` fixes the grey-scale unit instead of taking this image's own
      robust SD; the unit used comes back as ``Carpet.scale``.
    """
    if order not in CARPET_ORDERS:
        raise ValueError(f"order must be one of {CARPET_ORDERS}, got {order!r}")
    x = np.asarray(series_tv, dtype=np.float32)
    n = x.shape[0]
    allidx = (
        np.concatenate([np.asarray(idx, int) for _, idx in groups]) if groups else np.zeros(0, int)
    )
    if not allidx.size:
        z_all = np.zeros((n, 0), np.float32)
    elif standardize:
        z_all = _zscore_detrended(x[:, allidx])
    else:
        z_all = x[:, allidx]
    gmean = z_all.mean(axis=1) if allidx.size else np.zeros(n, np.float32)
    if standardize:
        gcor = float(np.sum((gmean / np.sqrt(max(n, 1))) ** 2))
    else:  # GCOR is defined on unit-variance series
        sd = z_all.std(axis=0)
        zm = (z_all / np.where(sd > 0, sd, 1.0)).mean(axis=1) if allidx.size else gmean
        gcor = float(np.sum((zm / np.sqrt(max(n, 1))) ** 2))
    pc1, frac = _first_pc(z_all) if allidx.size and n > 1 else (None, 0.0)
    if score is not None:
        score = np.asarray(score, np.float32)
        if pc1 is not None and (z_all.T @ pc1).sum() < 0:
            pc1 = -pc1
    elif order == "pc" and pc1 is not None:
        score = z_all.T @ pc1
        if score.sum() < 0:  # orient so most voxels load positively: top = strongest
            score, pc1 = -score, -pc1
    elif order == "gs":
        score = z_all.T @ gmean
    else:
        score = None
    total = allidx.size
    out, blocks, start, pos = [], [], 0, 0
    for label, idx in groups:
        k = len(idx)
        if k == 0:
            continue
        z = z_all[:, pos : pos + k]
        if score is not None:
            z = z[:, np.argsort(-score[pos : pos + k], kind="stable")]
        pos += k
        n_rows = max(1, min(k, round(max_rows * k / max(total, 1))))
        bounds = np.linspace(0, k, n_rows + 1).astype(int)
        out.append((np.add.reduceat(z, bounds[:-1], axis=1) / np.diff(bounds)).T)
        blocks.append((label, start, start + n_rows))
        start += n_rows
    if not out:
        return Carpet(np.zeros((1, n), np.float32), [("empty", 0, 1)], None, 0.0, float("nan"))
    rows = np.concatenate(out, axis=0)
    rows = rows - np.median(rows, axis=1, keepdims=True)
    if scale is None:
        scale = 1.4826 * float(np.median(np.abs(rows)))
    scale = scale if scale > 0 else 1.0
    return Carpet(rows / scale, blocks, pc1, frac, gcor, scale)


def _kde(v: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE at Silverman's bandwidth (numpy only; a few hundred points)."""
    v = v[np.isfinite(v)]
    if v.size < 2 or np.ptp(v) == 0:
        return np.zeros_like(grid)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    sd = min(v.std(), iqr / 1.349) if iqr > 0 else v.std()
    bw = 0.9 * sd * v.size ** (-0.2) or 1e-3
    d = (grid[:, None] - v[None, :]) / bw
    return np.exp(-0.5 * d * d).sum(axis=1) / (v.size * bw * np.sqrt(2 * np.pi))


@dataclass
class RunValues:
    """One run's per-TR values for :func:`raincloud`."""

    label: str
    values: np.ndarray
    censored: np.ndarray | None = None  # boolean, True = censored TR


def raincloud(
    panels: Sequence[tuple[str, Sequence[RunValues], float | None]],
    width_px: int = style.WIDTH,
    seed: int = 0,
):
    """Per-run distributions side by side on ONE y scale: half violin, box, points.

    ``panels`` is ``[(ylabel, runs, limit)]``, one row per measure. The y range is
    shared by every run in a panel so a run that moved more stands out; the top is
    the pooled 99.5th percentile (a single huge spike would otherwise flatten every
    other run into the floor), and values above it are drawn as triangles on the
    top edge with their count.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    rng = np.random.default_rng(seed)
    n_runs = max(len(runs) for _, runs, _ in panels)
    panel_h = 3.0
    with style.mpl_style():
        fig = Figure(figsize=(width_px / DPI, panel_h * len(panels) + 0.9), dpi=DPI)
        axes = fig.subplots(len(panels), 1, squeeze=False)[:, 0]
        for ax, (ylabel, runs, limit) in zip(axes, panels, strict=True):
            pooled = np.concatenate([r.values[np.isfinite(r.values)] for r in runs] or [[0.0]])
            top = float(np.percentile(pooled, 99.5)) if pooled.size else 1.0
            if limit is not None:
                top = max(top, 1.15 * limit)
            top = top * 1.08 if top > 0 else 1.0
            grid = np.linspace(0, top, 200)
            for i, r in enumerate(runs):
                v = np.asarray(r.values, float)
                fin = np.isfinite(v)
                dens = _kde(np.minimum(v[fin], top), grid)
                if dens.max() > 0:
                    dens = 0.36 * dens / dens.max()
                    ax.fill_betweenx(
                        grid, i - 0.06 - dens, i - 0.06, color=style.BLUE, alpha=0.55, lw=0
                    )
                if fin.any():
                    q1, med, q3 = np.percentile(v[fin], [25, 50, 75])
                    ax.add_patch(
                        Rectangle((i - 0.045, q1), 0.09, q3 - q1, fc=style.PANEL, ec=style.INK2)
                    )
                    ax.plot([i - 0.045, i + 0.045], [med, med], color=style.INK, lw=1.6)
                    ax.plot(i, float(np.mean(v[fin])), "o", color=style.YELLOW, ms=4, zorder=4)
                jit = i + 0.1 + 0.22 * rng.random(v.size)
                cens = (
                    np.asarray(r.censored, bool)
                    if r.censored is not None and len(r.censored) == v.size
                    else np.zeros(v.size, bool)
                )
                over = fin & (v > top)
                ok = fin & ~over
                ax.scatter(jit[ok & ~cens], v[ok & ~cens], s=5, color=style.AQUA, alpha=0.6, lw=0)
                ax.scatter(
                    jit[ok & cens], v[ok & cens], s=7, color=style.STATUS["fail"], alpha=0.9, lw=0
                )
                if over.any():
                    ax.plot(
                        jit[over], np.full(over.sum(), top), "^", color=style.STATUS["fail"], ms=5
                    )
                    ax.annotate(
                        f"{int(over.sum())} > {top:.2g}",
                        (i + 0.2, top),
                        xytext=(0, -12),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                        color=style.INK2,
                    )
            if limit is not None:
                ax.axhline(limit, color=style.INK2, lw=1.0, ls=(0, (4, 3)))
                ax.annotate(
                    f"censor limit {limit:g}",
                    xy=(1.0, limit),
                    xycoords=("axes fraction", "data"),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    color=style.INK2,
                )
            ax.set_ylim(0, top)
            ax.set_xlim(-0.6, n_runs - 0.4)
            ax.set_ylabel(ylabel)
            ax.grid(axis="x", visible=False)
            ticks = [
                f"{r.label}\nmean {np.nanmean(r.values):.3f}"
                if np.isfinite(r.values).any()
                else r.label
                for r in runs
            ]
            ax.set_xticks(range(len(runs)), ticks, fontsize=8 if n_runs > 8 else 9)
            if n_runs > 8:
                for t in ax.get_xticklabels():
                    t.set_rotation(30)
                    t.set_ha("right")
        fig.subplots_adjust(
            left=0.07, right=0.92, top=0.97, bottom=0.16 if n_runs > 8 else 0.11, hspace=0.45
        )
        return figure_image(fig)
