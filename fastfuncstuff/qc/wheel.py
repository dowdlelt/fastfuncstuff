"""The session wheel: every run of a subject on one page.

Data quality is mostly tSNR and model fit, and those follow from motion and
noise, so the wheel reads from the outside in, cause to consequence::

    motion  ->  per-tool checks  ->  tSNR (input -> residual)  ->  held-out fit
            ->  task significance  ->  residual tSNR of the brain (centre)

One sector per run, runs grouped by task (the F statistic is a task's, pooled
over its runs). **Nothing autoscales.** Every axis, every reference circle and
every slot is fixed, so flipping through subjects shows differences in the data
and never differences in the drawing: a quiet session is the same quiet shape
each time and a bad run breaks it. That is also why the motion ring is a
histogram and not a trace: a distribution has the same width for a 100-volume
run and a 1000-volume one.

Length carries what is compared by size (motion, tSNR, fit, significance);
colour carries only pass / over-a-limit, and the key says so in words.
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
MOTION_BINS = 16
MOTION_FULL = 0.35  # a bin holding this share of a run's volumes spans the sector
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
    status: dict[str, str] = field(default_factory=dict)  # slot -> 'ok' | 'warn'
    tsnr_in: float | None = None
    tsnr_res: float | None = None
    r2: float | None = None


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


def _log_radius(v: float, axis: tuple[float, float], ring: tuple[float, float]) -> float:
    lo, hi = axis
    t = (np.log(np.clip(v, lo, hi)) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return float(ring[0] + t * (ring[1] - ring[0]))


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
        for v in (0.03, 0.3, 3):
            ax.text(
                0,
                _log_radius(v, MOTION_AXIS, R_MOTION) - (0.012 if v == 3 else 0),
                f"{v:g}",
                ha="center",
                va="center",
                fontsize=8,
                color=style.MUTED,
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
        for name, r in (("mm", 1.045), ("tSNR", R_TSNR[1] + 0.022), ("R²", R_FIT[1] + 0.022)):
            ax.text(0, r, name, ha="center", va="center", fontsize=8, color=style.INK2)

        n_slots = max(len(slots), 1)
        edges = np.geomspace(*MOTION_AXIS, MOTION_BINS + 1)
        for task, tspans in zip(tasks, spans, strict=True):
            if not task.runs:
                continue
            for i, (run, (a0, a1)) in enumerate(zip(task.runs, tspans, strict=True)):
                w = a1 - a0
                pad = min(0.9, 0.08 * w)
                b0, b1 = a0 + pad, a1 - pad
                mid = (a0 + a1) / 2
                # --- motion: a histogram along the radius ---------------------
                wedge(b0, b1, *R_MOTION, style.PANEL)
                if run.motion is not None and np.isfinite(run.motion).any():
                    v = np.clip(run.motion[np.isfinite(run.motion)], *MOTION_AXIS)
                    frac = np.histogram(v, edges)[0] / max(v.size, 1)
                    for k in np.flatnonzero(frac > 0):
                        half = 0.5 * (b1 - b0) * min(1.0, frac[k] / MOTION_FULL)
                        over = run.motion_limit is not None and edges[k] >= run.motion_limit
                        wedge(
                            mid - half,
                            mid + half,
                            _log_radius(edges[k], MOTION_AXIS, R_MOTION),
                            _log_radius(edges[k + 1], MOTION_AXIS, R_MOTION),
                            style.STATUS["fail"] if over else style.BLUE,
                        )
                    if run.motion_limit:
                        arc(
                            b0,
                            b1,
                            _log_radius(run.motion_limit, MOTION_AXIS, R_MOTION),
                            style.INK2,
                            1.0,
                            (0, (3, 2)),
                        )
                # --- checks: one cell per tool --------------------------------
                cw = (b1 - b0) / n_slots
                for k, slot in enumerate(slots or [""]):
                    state = run.status.get(slot)
                    color = style.STATUS[state] if state else style.GRID
                    wedge(b0 + k * cw + 0.06 * cw, b0 + (k + 1) * cw - 0.06 * cw, *R_STATUS, color)
                # --- tSNR: input (hollow) to residual (filled) ----------------
                wedge(b0, b1, *R_TSNR, style.PANEL, alpha=0.55)
                if run.tsnr_in and run.tsnr_res:
                    r_in = _log_radius(run.tsnr_in, TSNR_AXIS, R_TSNR)
                    r_res = _log_radius(run.tsnr_res, TSNR_AXIS, R_TSNR)
                    color = style.AQUA if run.tsnr_res >= run.tsnr_in else style.ORANGE
                    (x0, y0), (x1, y1) = _xy(r_in, mid), _xy(r_res, mid)
                    ms = float(np.clip(1.1 * w, 4.5, 9.0))
                    ax.plot([x0, x1], [y0, y1], color=color, lw=2.2, solid_capstyle="butt")
                    ax.plot(x0, y0, "o", ms=ms, mfc=style.SURFACE, mec=color, mew=1.6)
                    ax.plot(x1, y1, "o", ms=ms, mfc=color, mec=color)
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
            style.BLUE,
            "motion",
            f"share of volumes at each enorm, {MOTION_AXIS[0]:g} (inner) to {MOTION_AXIS[1]:g} mm "
            "(outer), log. Dashed arc: censor limit; red: beyond it.",
        ),
        (
            style.STATUS["ok"],
            "checks",
            f"one cell per tool, in order: {checks}. Green: within its limits. "
            "Amber: over one (listed below). Grey: not reported.",
        ),
        (
            style.AQUA,
            "tSNR",
            f"hollow: the GLM's input; filled: its residual. {TSNR_AXIS[0]:g} to "
            f"{TSNR_AXIS[1]:g}, log. The bar between them is what the model removed.",
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
