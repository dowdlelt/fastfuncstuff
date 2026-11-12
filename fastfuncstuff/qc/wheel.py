"""The session wheel: every run of a subject on one page.

Data quality is mostly tSNR and model fit, and those follow from motion and
noise, so the wheel reads from the outside in, cause to consequence::

    motion  ->  per-tool checks  ->  tSNR (input -> residual)  ->  held-out fit
            ->  task significance  ->  residual tSNR of the brain (centre)

One sector per run, runs grouped by task (the F statistic is a task's, pooled
over its runs). **Nothing autoscales.** Every axis, every reference circle and
every slot is fixed, so flipping through subjects shows differences in the data
and never differences in the drawing: a quiet session is the same quiet shape
each time and a bad run breaks it. That is also why the motion ring holds
distributions and not traces: a distribution has the same width for a 100-volume
run and a 1000-volume one.

The motion ring is two distributions side by side, the outlier fraction
(3dToutcount) and enorm, each on its own log axis and each coloured by its
value. Width alone would hide a mostly still run that lurched a few times (five
frames of 316 are a sliver), so every value that occurs at all gets a minimum
width: the lurch shows as a spot of hot colour out at the rim. The tSNR ring is
a box per run for the GLM's input and its residual, filled through the same
colour scale as the centre brain, so a run whose lower quartile sits low is a
box that starts darker and further in than its neighbours'.

Length carries what is compared by size (motion, tSNR, fit, significance);
the checks ring's colour carries only pass / over-a-limit.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from . import render, style

# Fixed axes. fMRI tSNR spans ~6 (thin-slab 7T) to a few hundred (smoothed 3T);
# enorm from a still head (~0.01 mm) to a lurch of millimetres.
TSNR_AXIS = (5.0, 300.0)
MOTION_AXIS = (0.01, 3.0)
# 3dToutcount fraction: a still run sits at 1e-4 to 1e-2, afni_proc censors at 0.05.
OUTLIER_AXIS = (0.0001, 0.3)  # 0 is clipped into the first bin
OUTLIER_BINS = 24
OUTLIER_FULL = 0.5  # a bin holding this share of a run's volumes spans its half
ENORM_DENS_FULL = 2.5  # enorm density (share per decade) that spans its half
ENORM_BW = 0.06  # smoothing bandwidth of the enorm curve, decades
SEEN_WIDTH = 0.14  # minimum width (share of a half) of any value that occurs at all
ENORM_CMAP = "turbo"
OUTLIER_CMAP = "hot"
OUTLIER_CMAP_FROM = 0.3  # hot starts at black, which vanishes on the dark panel
R2_TOP = 0.5
FRAC_TOP = 0.5

R_MOTION = (0.80, 1.00)
R_STATUS = (0.752, 0.778)
R_TSNR = (0.50, 0.72)
R_FIT = (0.36, 0.47)
R_TASK = (0.29, 0.34)
R_CENTRE = 0.265

GAP_TOP = 34.0  # degrees left open at 12 o'clock for the axis numbers
GAP_TASK = 6.0


@dataclass
class WheelRun:
    label: str
    motion: np.ndarray | None = None  # per-volume enorm (or FD), mm
    motion_limit: float | None = None
    outlier: np.ndarray | None = None  # per-volume outlier fraction
    outlier_limit: float | None = None
    status: dict[str, str] = field(default_factory=dict)  # slot -> 'ok' | 'warn'
    tsnr_in: float | None = None
    tsnr_res: float | None = None
    r2: float | None = None
    tsnr_in_q: tuple[float, ...] | None = None  # 5, 25, 50, 75, 95th percentiles
    tsnr_res_q: tuple[float, ...] | None = None


@dataclass
class WheelTask:
    name: str
    runs: list[WheelRun]
    frac_sig: float | None = None


def short_run_label(label: str, index: int) -> str:
    """``ses-02 ... run-03`` -> ``s02 r3``; a label without a run entity gets its
    position in the task."""
    ses = re.search(r"ses-([A-Za-z0-9]+)", label)
    run = re.search(r"run-?0*(\d+)", label)
    r = f"r{run.group(1)}" if run else f"r{index + 1}"
    return f"s{ses.group(1)} {r}" if ses else r


def _log_t(v, axis: tuple[float, float]):
    lo, hi = axis
    return (np.log(np.clip(v, lo, hi)) - np.log(lo)) / (np.log(hi) - np.log(lo))


def _log_radius(v: float, axis: tuple[float, float], ring: tuple[float, float]) -> float:
    return float(ring[0] + _log_t(v, axis) * (ring[1] - ring[0]))


def _cmap(name: str, t: float, start: float = 0.0) -> tuple[float, float, float]:
    """Colour at ``t`` in [0, 1] of a matplotlib map, its bottom ``start`` skipped."""
    table = render.lut(name)
    k = int(np.clip(start + (1 - start) * t, 0, 1) * 255)
    r, g, b = (float(c) / 255 for c in table[k])
    return r, g, b


def outlier_hist(frac: np.ndarray) -> np.ndarray:
    """Share of volumes in each of :data:`OUTLIER_BINS` log bins of :data:`OUTLIER_AXIS`."""
    v = np.asarray(frac, float)
    v = np.clip(v[np.isfinite(v)], *OUTLIER_AXIS)
    edges = np.geomspace(*OUTLIER_AXIS, OUTLIER_BINS + 1)
    return np.histogram(v, edges)[0] / max(v.size, 1)


def enorm_density(motion: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian density of log10 enorm on ``grid`` (log10 mm), in share per decade,
    and whether any volume lies within reach of each grid point (one volume's
    kernel above a quarter of its own peak)."""
    v = np.asarray(motion, float)
    v = np.log10(np.clip(v[np.isfinite(v)], *MOTION_AXIS))
    if not v.size:
        return np.zeros_like(grid), np.zeros(grid.shape, bool)
    k = np.exp(-0.5 * ((grid[:, None] - v[None]) / ENORM_BW) ** 2)
    dens = k.sum(1) / (v.size * ENORM_BW * np.sqrt(2 * np.pi))
    return dens, k.max(1) > 0.25


def _xy(r: float, a: float) -> tuple[float, float]:
    """Clock angle (0 = up, clockwise, degrees) to Cartesian."""
    t = np.deg2rad(a)
    return float(r * np.sin(t)), float(r * np.cos(t))


def layout(tasks: Sequence[WheelTask]) -> list[list[tuple[float, float]]]:
    """Clock-angle span ``(a0, a1)`` of every run, per task. Runs are equal width."""
    n_runs = sum(len(t.runs) for t in tasks)
    n_tasks = sum(1 for t in tasks if t.runs)
    usable = 360.0 - GAP_TOP - GAP_TASK * max(n_tasks - 1, 0)
    w = usable / max(n_runs, 1)
    a = GAP_TOP / 2
    out = []
    for t in tasks:
        spans = []
        for _ in t.runs:
            spans.append((a, a + w))
            a += w
        if t.runs:
            a += GAP_TASK
        out.append(spans)
    return out


def tsnr_rgb(img: np.ndarray) -> np.ndarray:
    """A tSNR slice through inferno on the fixed log axis; 0 (outside) is the surface."""
    lo, hi = TSNR_AXIS
    img = np.asarray(img, np.float32)
    t = (np.log(np.clip(img, lo, hi)) - np.log(lo)) / (np.log(hi) - np.log(lo))
    rgb = render.lut("inferno")[(np.clip(t, 0, 1) * 255).astype(np.uint8)]
    rgb[img <= 0] = style.hex_rgb(style.SURFACE)
    return rgb


def wheel(
    tasks: Sequence[WheelTask],
    slots: Sequence[str] = (),
    centre: np.ndarray | None = None,
    centre_aspect: tuple[float, float] = (1.0, 1.0),
    headline: Sequence[str] = (),
    flags: Sequence[str] = (),
    width_px: int = style.WIDTH,
):
    """Draw the wheel; returns a PIL image for :func:`render.compose`.

    ``slots`` names the cells of the checks ring, in order, the same for every
    run. ``centre`` is one residual-tSNR slice (display rows x columns) with its
    mm-per-row / mm-per-column ``centre_aspect``.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Arc, Circle, Wedge

    spans = layout(tasks)
    h_in = 10.6
    with style.mpl_style():
        fig = Figure(figsize=(width_px / 100, h_in), dpi=100)
        side = h_in / (width_px / 100)
        ax = fig.add_axes((0.0, 0.0, side, 1.0))
        lim = 1.22
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axis("off")

        def wedge(a0, a1, r0, r1, color, **kw):
            ax.add_patch(Wedge((0, 0), r1, 90 - a1, 90 - a0, width=r1 - r0, fc=color, lw=0, **kw))

        def arc(a0, a1, r, color, lw=1.0, ls="-"):
            ax.add_patch(
                Arc((0, 0), 2 * r, 2 * r, theta1=90 - a1, theta2=90 - a0, color=color, lw=lw, ls=ls)
            )

        # Reference circles: the same radii on every wheel.
        for v in (10, 30, 100):
            arc(GAP_TOP / 2, 360 - GAP_TOP / 2, _log_radius(v, TSNR_AXIS, R_TSNR), style.GRID, 0.8)
            ax.text(
                0,
                _log_radius(v, TSNR_AXIS, R_TSNR),
                f"{v}",
                ha="center",
                va="center",
                fontsize=8,
                color=style.MUTED,
            )
        # Motion: outlier fraction on the left of the gap, enorm on the right,
        # each number in its own map's colour.
        for v in (0.001, 0.01, 0.1):
            ax.text(
                -0.075,
                _log_radius(v, OUTLIER_AXIS, R_MOTION),
                f"{v:g}",
                ha="center",
                va="center",
                fontsize=8,
                color=_cmap(OUTLIER_CMAP, float(_log_t(v, OUTLIER_AXIS)), OUTLIER_CMAP_FROM),
            )
        for v in (0.03, 0.3, 3):
            ax.text(
                0.075,
                _log_radius(v, MOTION_AXIS, R_MOTION) - (0.012 if v == 3 else 0),
                f"{v:g}",
                ha="center",
                va="center",
                fontsize=8,
                color=_cmap(ENORM_CMAP, float(_log_t(v, MOTION_AXIS))),
            )
        ax.text(
            0,
            R_FIT[1] - 0.012,
            f"{R2_TOP:g}",
            ha="center",
            va="center",
            fontsize=8,
            color=style.MUTED,
        )
        ax.text(0, R_FIT[0] + 0.012, "0", ha="center", va="center", fontsize=8, color=style.MUTED)
        for x, name, r in (
            (-0.075, "outl.", 1.045),
            (0.075, "mm", 1.045),
            (0, "tSNR", R_TSNR[1] + 0.022),
            (0, "R²", R_FIT[1] + 0.022),
        ):
            ax.text(x, r, name, ha="center", va="center", fontsize=8, color=style.INK2)

        n_slots = max(len(slots), 1)
        for task, tspans in zip(tasks, spans, strict=True):
            if not task.runs:
                continue
            for i, (run, (a0, a1)) in enumerate(zip(task.runs, tspans, strict=True)):
                w = a1 - a0
                pad = min(0.9, 0.08 * w)
                b0, b1 = a0 + pad, a1 - pad
                mid = (a0 + a1) / 2
                # --- motion: outlier fraction | enorm, each on its own axis --
                wedge(b0, b1, *R_MOTION, style.PANEL)
                gap = min(0.5, 0.04 * w)
                halves = ((b0, mid - gap), (mid + gap, b1))
                _draw_outliers(wedge, arc, run, *halves[0])
                _draw_enorm(wedge, arc, run, *halves[1])
                # --- checks: one cell per tool --------------------------------
                cw = (b1 - b0) / n_slots
                for k, slot in enumerate(slots or [""]):
                    state = run.status.get(slot)
                    color = style.STATUS[state] if state else style.GRID
                    wedge(b0 + k * cw + 0.06 * cw, b0 + (k + 1) * cw - 0.06 * cw, *R_STATUS, color)
                # --- tSNR: a box for the input, a box for the residual -------
                wedge(b0, b1, *R_TSNR, style.PANEL, alpha=0.55)
                q = 0.1 * (b1 - b0)
                for (c0, c1), quant, med, alpha in (
                    ((b0 + q, mid - gap), run.tsnr_in_q, run.tsnr_in, 0.5),
                    ((mid + gap, b1 - q), run.tsnr_res_q, run.tsnr_res, 1.0),
                ):
                    if quant is None and med:
                        quant = (med,) * 5
                    if quant is not None:
                        _draw_box(ax, wedge, arc, c0, c1, quant, alpha)
                # --- fit: held-out R2 as a bar ---------------------------------
                wedge(b0, b1, *R_FIT, style.PANEL)
                if run.r2 is not None:
                    if run.r2 > 0:
                        top = R_FIT[0] + (R_FIT[1] - R_FIT[0]) * min(run.r2, R2_TOP) / R2_TOP
                        q = 0.2 * (b1 - b0)
                        wedge(b0 + q, b1 - q, R_FIT[0], top, style.VIOLET)
                    else:
                        ax.plot(
                            *_xy(R_FIT[0] + 0.02, mid),
                            "x",
                            ms=6,
                            mew=1.8,
                            color=style.STATUS["warn"],
                        )
                # --- run label -------------------------------------------------
                rot = 90 - mid if mid <= 180 else 270 - mid
                ax.text(
                    *_xy(1.015, mid),
                    short_run_label(run.label, i),
                    rotation=rot,
                    rotation_mode="anchor",
                    ha="left" if mid <= 180 else "right",
                    va="center",
                    fontsize=9 if w > 7 else 7.5,
                    color=style.INK2,
                )
            # --- task: an arc under its runs, thickness = significance --------
            a0, a1 = tspans[0][0], tspans[-1][1]
            wedge(a0, a1, *R_TASK, style.PANEL)
            if task.frac_sig is not None:
                thick = (R_TASK[1] - R_TASK[0]) * np.sqrt(min(task.frac_sig, FRAC_TOP) / FRAC_TOP)
                wedge(a0, a1, R_TASK[0], R_TASK[0] + max(float(thick), 0.003), style.YELLOW)
            mid = (a0 + a1) / 2
            sig = "" if task.frac_sig is None else f"   {100 * task.frac_sig:.1f}% p<.001"
            # Along the rim, turned so it always reads left to right.
            ax.text(
                *_xy(1.135, mid),
                task.name + sig,
                rotation=-mid if not 90 < mid < 270 else 180 - mid,
                rotation_mode="anchor",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=style.INK,
            )

        # --- centre: residual tSNR of the brain, on the ring's own scale ------
        ax.add_patch(Circle((0, 0), R_CENTRE, fc="#000000", ec=style.GRID, lw=1.0))
        if centre is not None and centre.size:
            hh = centre.shape[0] * centre_aspect[0] / 2
            hw = centre.shape[1] * centre_aspect[1] / 2
            s = 0.94 * R_CENTRE / max(hh, hw)
            im = ax.imshow(
                tsnr_rgb(centre),
                extent=(-hw * s, hw * s, -hh * s, hh * s),
                interpolation="bilinear",
                zorder=2,
            )
            im.set_clip_path(Circle((0, 0), R_CENTRE - 0.004, transform=ax.transData))
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

        _key(fig, side, slots, headline, flags)
        return render.figure_image(fig)


def _draw_outliers(wedge, arc, run: WheelRun, a0: float, a1: float) -> None:
    """Outlier-fraction histogram in ``[a0, a1]``: baseline at ``a0``, bars grow
    clockwise, each bin in the heat colour of its own value."""
    wedge(a0, a0 + 0.15, *R_MOTION, style.GRID)
    if run.outlier is None or not np.isfinite(run.outlier).any():
        return
    frac = outlier_hist(run.outlier)
    edges = np.geomspace(*OUTLIER_AXIS, OUTLIER_BINS + 1)
    for k in np.flatnonzero(frac > 0):
        share = max(min(1.0, frac[k] / OUTLIER_FULL), SEEN_WIDTH)
        t = float(_log_t(np.sqrt(edges[k] * edges[k + 1]), OUTLIER_AXIS))
        wedge(
            a0,
            a0 + share * (a1 - a0),
            _log_radius(edges[k], OUTLIER_AXIS, R_MOTION),
            _log_radius(edges[k + 1], OUTLIER_AXIS, R_MOTION),
            _cmap(OUTLIER_CMAP, t, OUTLIER_CMAP_FROM),
        )
    if run.outlier_limit:
        r = _log_radius(run.outlier_limit, OUTLIER_AXIS, R_MOTION)
        arc(a0, a1, r, style.INK2, 1.0, (0, (3, 2)))


def _draw_enorm(wedge, arc, run: WheelRun, a0: float, a1: float, n: int = 96) -> None:
    """Smoothed enorm density in ``[a0, a1]``, coloured through turbo by radius."""
    wedge(a0, a0 + 0.15, *R_MOTION, style.GRID)
    if run.motion is None or not np.isfinite(run.motion).any():
        return
    lo, hi = np.log10(MOTION_AXIS)
    bounds = np.linspace(lo, hi, n + 1)
    dens, seen = enorm_density(run.motion, 0.5 * (bounds[1:] + bounds[:-1]))
    r_edge = R_MOTION[0] + (bounds - lo) / (hi - lo) * (R_MOTION[1] - R_MOTION[0])
    for k in range(n):
        share = min(1.0, dens[k] / ENORM_DENS_FULL)
        if seen[k]:
            share = max(share, SEEN_WIDTH)
        if share <= 0.004:
            continue
        wedge(
            a0,
            a0 + share * (a1 - a0),
            r_edge[k],
            r_edge[k + 1],
            _cmap(ENORM_CMAP, (k + 0.5) / n),
        )
    if run.motion_limit:
        r = _log_radius(run.motion_limit, MOTION_AXIS, R_MOTION)
        arc(a0, a1, r, style.INK2, 1.0, (0, (3, 2)))


def _draw_box(ax, wedge, arc, a0: float, a1: float, quant, alpha: float, n: int = 24) -> None:
    """A radial box plot of voxel tSNR: whisker 5-95%, box 25-75% filled through
    inferno by radius (the centre's scale), median as a bright arc."""
    p5, p25, p50, p75, p95 = quant
    mid = 0.5 * (a0 + a1)
    r5, r95 = (_log_radius(v, TSNR_AXIS, R_TSNR) for v in (p5, p95))
    (x0, y0), (x1, y1) = _xy(r5, mid), _xy(r95, mid)
    ax.plot([x0, x1], [y0, y1], color=style.MUTED, lw=1.1, alpha=max(alpha, 0.7))
    cap = 0.25 * (a1 - a0)
    for r in (r5, r95):
        arc(mid - cap, mid + cap, r, style.MUTED, 1.1)
    t25, t75 = (float(_log_t(v, TSNR_AXIS)) for v in (p25, p75))
    edges = np.linspace(t25, t75, n + 1)
    for k in range(n):
        if edges[k + 1] <= edges[k]:
            continue
        wedge(
            a0,
            a1,
            R_TSNR[0] + edges[k] * (R_TSNR[1] - R_TSNR[0]),
            R_TSNR[0] + edges[k + 1] * (R_TSNR[1] - R_TSNR[0]),
            _cmap("inferno", 0.5 * (edges[k] + edges[k + 1])),
            alpha=alpha,
        )
    arc(a0, a1, _log_radius(p50, TSNR_AXIS, R_TSNR), style.INK, 2.0)


def _key(fig, left: float, slots, headline, flags) -> None:
    """The right-hand panel: headline numbers, how to read each ring, what to check."""
    from matplotlib.patches import Rectangle

    ax = fig.add_axes((left + 0.005, 0.03, 1.0 - left - 0.02, 0.94))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.985
    for i, line in enumerate(headline):
        ax.text(
            0,
            y,
            line,
            fontsize=17 if i == 0 else 11,
            fontweight="bold" if i == 0 else "normal",
            color=style.INK if i == 0 else style.INK2,
            va="top",
        )
        y -= 0.05 if i == 0 else 0.032
    y -= 0.02
    ax.text(0, y, "Outside in", fontsize=11, fontweight="bold", color=style.INK, va="top")
    y -= 0.04
    checks = ", ".join(slots) if slots else "none reported"
    rows = [
        (
            _cmap(ENORM_CMAP, 0.75),
            "motion",
            f"left: outlier fraction ({OUTLIER_AXIS[0]:g}-{OUTLIER_AXIS[1]:g}, heat); right: "
            f"enorm ({MOTION_AXIS[0]:g}-{MOTION_AXIS[1]:g} mm, turbo). Radius = value (log), "
            "width = share of volumes, but any value that occurs gets a minimum width: a "
            "few big frames are a spot of hot colour at the rim. Dashed: censor limit.",
        ),
        (
            style.STATUS["ok"],
            "checks",
            f"one cell per tool, in order: {checks}. Green: within its limits. "
            "Amber: over one (listed below). Grey: not reported.",
        ),
        (
            _cmap("inferno", 0.6),
            "tSNR",
            f"voxel tSNR, {TSNR_AXIS[0]:g}-{TSNR_AXIS[1]:g}, log, in the centre's colours. "
            "Faint left box: the GLM's input; right: its residual. Box 25-75%, whiskers "
            "5-95%, white arc the median.",
        ),
        (
            style.VIOLET,
            "fit",
            f"held-out R² (0 to {R2_TOP:g}): the other runs' task betas predicting this "
            "run. Amber x: zero or below.",
        ),
        (
            style.YELLOW,
            "task",
            f"arc thickness: share of fitted voxels with F at p < .001 (square-root "
            f"scale, full at {100 * FRAC_TOP:.0f}%).",
        ),
    ]
    for color, name, text in rows:
        ax.add_patch(Rectangle((0, y - 0.022), 0.03, 0.02, fc=color, lw=0))
        ax.text(0.05, y, name, fontsize=10.5, fontweight="bold", color=style.INK, va="top")
        wrapped = _wrap_chars(text, 62)
        ax.text(0.05, y - 0.028, wrapped, fontsize=9.5, color=style.INK2, va="top", linespacing=1.4)
        y -= 0.04 + 0.027 * (wrapped.count("\n") + 1)
    # centre + its colour bar
    ax.text(0.05, y, "centre", fontsize=10.5, fontweight="bold", color=style.INK, va="top")
    ax.text(
        0.05,
        y - 0.028,
        "residual tSNR of the brain (middle axial cut), same axis:",
        fontsize=9.5,
        color=style.INK2,
        va="top",
    )
    y -= 0.075
    grad = render.lut("inferno")[None, :, :]
    ax.imshow(grad, extent=(0.05, 0.75, y - 0.018, y), aspect="auto", zorder=2)
    lo, hi = TSNR_AXIS
    for v in (5, 10, 30, 100, 300):
        x = 0.05 + 0.7 * (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))
        ax.text(x, y - 0.024, f"{v}", fontsize=8.5, color=style.MUTED, ha="center", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y -= 0.085
    ax.text(
        0,
        y,
        "Check" if flags else "No limit crossed",
        fontsize=11,
        fontweight="bold",
        color=style.INK,
        va="top",
    )
    y -= 0.036
    room = max(int((y - 0.01) / 0.027), 1)
    shown = list(flags[:room])
    if len(flags) > room:
        shown[-1] = f"... and {len(flags) - room + 1} more (see qc_review.txt)"
    for line in shown:
        ax.add_patch(Rectangle((0, y - 0.018), 0.018, 0.014, fc=style.STATUS["warn"], lw=0))
        ax.text(0.035, y, line[:74], fontsize=9.5, color=style.INK2, va="top")
        y -= 0.027


def _wrap_chars(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width))
