"""Figures for design-power results, shared by ffs_simulate and notebooks.

Colour follows the job: tSNR levels are ordered, so they take a single-hue
sequential ramp; conditions are identities, so they take categorical slots in
fixed order; correlation is signed, so it takes a two-hue diverging map with a
neutral midpoint. Curves are direct-labelled only up to four series -- beyond
that the legend (always present) carries identity.

Each function returns the figure and, given ``path``, also saves and closes it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
# Sequential blue, steps 200 -> 700 of the reference ramp.
BLUES = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ramp(n: int) -> list[str]:
    """``n`` ordered colours from the sequential ramp, light (worst) to dark (best)."""
    if n <= 1:
        return [BLUES[3]]
    if n > len(BLUES):
        import matplotlib.colors as mcolors

        cmap = mcolors.LinearSegmentedColormap.from_list("seq", [BLUES[0], BLUES[-1]])
        return [mcolors.to_hex(cmap(x)) for x in np.linspace(0, 1, n)]
    return [BLUES[i] for i in np.linspace(0, len(BLUES) - 1, n).round().astype(int)]


def _finish(fig, path: str | Path | None):
    import textwrap

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    footer_top = 0.0
    for text in fig.texts:
        if not text.get_text():
            continue
        x, y = text.get_position()
        width_fraction = 2 * min(x, 1 - x) if text.get_ha() == "center" else 1 - x
        width_pixels = max(0.5, width_fraction) * fig.bbox.width - 24
        char_pixels = (
            renderer.get_text_width_height_descent(
                "abcdefghijklmnopqrstuvwxyz", text.get_fontproperties(), False
            )[0]
            / 26
        )
        wrapper = textwrap.TextWrapper(
            width=max(20, int(width_pixels / char_pixels)),
            break_long_words=False,
            break_on_hyphens=False,
        )
        text.set_text("\n".join(wrapper.fill(line) for line in text.get_text().splitlines()))
        text.set_wrap(False)
        if y < 0.1:
            footer_top = max(footer_top, text.get_window_extent(renderer).y1 / fig.bbox.height)
    # Power's manually positioned scorecard has its own height calculation.
    # Other figures can reserve caption space through their subplot layout.
    if all(ax.get_subplotspec() is not None for ax in fig.axes):
        bottom = max(0.02, footer_top + 0.02)
        engine = fig.get_layout_engine()
        if engine is not None and type(engine).__name__ == "ConstrainedLayoutEngine":
            engine.set(rect=(0.015, bottom, 0.97, 0.985 - bottom))
        else:
            fig.tight_layout(rect=(0.015, bottom, 0.985, 0.985), pad=1.2)
    if path is not None:
        import matplotlib.pyplot as plt

        fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.18)
        plt.close(fig)
    return fig


def _effective(contrasts: dict[str, Any], pattern) -> list[str]:
    """Contrasts with a true effect to detect -- the ones power applies to."""
    from .power import has_true_effect

    return [c for c in contrasts if has_true_effect(contrasts[c], pattern)]


def plot_power(
    result: dict[str, Any],
    noise_labels: list[str],
    contrasts: dict[str, Any],
    pattern,
    alpha: float = 0.001,
    effect: float | None = None,
    path: str | Path | None = None,
    title: str | None = None,
    max_panels: int = 6,
    summary: dict[str, Any] | None = None,
):
    """Power vs amplitude, one panel per contrast, one line per noise level.

    Line: analytic power (median over realizations/true HRFs), band: its range,
    dots: Monte Carlo. Dashed: 80%; dotted: ``effect``.

    ``summary`` adds a text panel underneath, so the figure stands alone: a
    block of ``facts`` ((key, value) pairs) and ``notes`` on the left, the
    answer table (``header``, ``rows``) and its ``footer`` on the right.
    """
    import matplotlib.pyplot as plt

    names = _effective(contrasts, pattern)[:max_panels]
    if not names:
        raise ValueError("no contrast has a true effect under this pattern")
    rows = result["table"]
    reml = any(r.get("estimator") == "reml" for r in rows)
    colors = ramp(len(noise_labels))
    width = max(4.6 * len(names), 13.0 if summary else 0.0)
    text_h = _summary_height(summary, width) if summary else 0.0
    fig, axes = plt.subplots(1, len(names), figsize=(width, 4.2 + text_h), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, c in zip(axes[0], names, strict=True):
        _style(ax)
        ax.axhline(0.8, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        for label, col in zip(noise_labels, colors, strict=True):
            sel = [r for r in rows if r["noise"] == label and r["contrast"] == c]
            sweep = sorted({r["amplitude"] for r in sel})
            by = {a: [r for r in sel if r["amplitude"] == a] for a in sweep}
            # Plotted against the contrast's true effect, as every table reports
            # it -- the sweep differs from it under -pattern.
            amps = [abs(by[a][0]["true_effect"]) for a in sweep]
            # nan-aware: a plain median over one nan (a realization REML once
            # withheld) blanked the whole line, and min/max drew an arbitrary band.
            pred = np.array(
                [[r["power" if reml else "power_predicted"] for r in by[a]] for a in sweep]
            )
            ok = np.isfinite(pred).any(axis=1)
            if not ok.any():
                ax.plot([], [], color=col, linewidth=2, label=label)
                continue
            keep = np.flatnonzero(ok)
            pred, amps, sweep = pred[keep], [amps[i] for i in keep], [sweep[i] for i in keep]
            med = np.nanmedian(pred, axis=1)
            ax.fill_between(
                amps,
                np.nanmin(pred, axis=1),
                np.nanmax(pred, axis=1),
                color=col,
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(amps, med, color=col, linewidth=2, label=label)
            ax.plot(
                amps,
                [np.mean([r["power"] for r in by[a]]) for a in sweep],
                "o",
                color=col,
                markersize=4.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1,
            )
            if len(noise_labels) <= 4:
                # At the 50% crossing: saturated curves all end at 1.0, where
                # right-edge labels land on top of each other.
                k = int(np.argmin(np.abs(med - 0.5)))
                ax.annotate(
                    label.split(" (")[0],
                    (amps[k], med[k]),
                    xytext=(6, -2),
                    textcoords="offset points",
                    fontsize=8,
                    color=INK2,
                    va="top",
                )
        at = [r for r in rows if r["contrast"] == c and abs(r["amplitude"] - (effect or -1)) < 1e-6]
        if effect is not None and at:
            ax.axvline(abs(at[0]["true_effect"]), color=INK2, linewidth=1, linestyle=":")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(c, color=INK, fontsize=11)
        swept = next((r.get("swept", "amplitude") for r in rows if r["contrast"] == c), "amplitude")
        if swept == "difference":
            shared = next(r.get("shared", 0.0) for r in rows if r["contrast"] == c)
            on = f" on {shared:g}% shared" if shared else ""
            ax.set_xlabel(f"{c} difference (% signal change{on})", color=INK2, fontsize=9)
        elif swept == "contrast":
            ax.set_xlabel(f"{c} (% signal change; others at -responses)", color=INK2, fontsize=9)
        else:
            ax.set_xlabel(f"{c} effect (% signal change)", color=INK2, fontsize=9)
    axes[0][0].set_ylabel("power", color=INK2, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    total_h = 4.2 + text_h
    legend_y = (text_h + 0.05) / total_h  # the strip between the curves and the text
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        labelcolor=INK2,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=min(len(labels), 5),
    )
    from .power import has_mismatch

    note = (
        " (HRF mismatch: the analytic line is approximate, the dots decide)"
        if has_mismatch(rows) and not reml
        else ""
    )
    fig.suptitle(
        title
        or f"Power at two-tailed p < {alpha:g} -- line: {'fitted REML' if reml else 'analytic'} (band: range over "
        f"realizations), dots: Monte Carlo; dashed: 80%{note}",
        color=INK,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, (text_h + 0.32) / total_h, 1, 1))
    if summary:
        ax_t = fig.add_axes((0.02, 0.01, 0.96, (text_h - 0.05) / total_h))
        _draw_summary(ax_t, summary, width)
    return _finish(fig, path)


_LINE_IN = 0.19  # inches per summary text line at fontsize 8.5


def _wrap_facts(summary: dict[str, Any], width: float) -> list[tuple[str, str]]:
    """Facts and notes as (key, line) pairs, wrapped to the left block's width."""
    import textwrap

    chars = max(40, int(width * 0.46 * 13))  # ~13 chars per inch at 8.5 pt
    out: list[tuple[str, str]] = []
    items = list(summary.get("facts", [])) + [("note", n) for n in summary.get("notes", [])]
    for key, value in items:
        for k, line in enumerate(textwrap.wrap(str(value), chars - 12) or [""]):
            out.append((key if k == 0 else "", line))
    return out


def _summary_height(summary: dict[str, Any], width: float) -> float:
    left = len(_wrap_facts(summary, width))
    right = 2 + len(summary.get("rows", [])) + 2 * len(summary.get("footer", []))
    return (max(left, right) + 1.5) * _LINE_IN


def _draw_summary(ax, summary: dict[str, Any], width: float) -> None:
    """The text panel: facts on the left, the answer table on the right."""
    import textwrap

    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axhline(1.0, color=GRID, linewidth=1)
    n_lines = _summary_height(summary, width) / _LINE_IN
    dy = 1.0 / n_lines
    y0 = 1.0 - 0.9 * dy
    for k, (key, line) in enumerate(_wrap_facts(summary, width)):
        y = y0 - k * dy
        ax.text(0.0, y, key, fontsize=8.5, color=INK2, va="top", fontweight="bold")
        ax.text(0.105, y, line, fontsize=8.5, color=INK, va="top")
    header, rows = summary.get("header", []), summary.get("rows", [])
    if not header:
        return
    # Right block: first column left-aligned, the rest right-aligned, each column
    # as wide as its longest entry (even slots let a long header run into the next).
    x_left, x_right = 0.5, 1.0
    chars = [max(len(str(c)) for c in [h, *(r[k] for r in rows)]) + 3 for k, h in enumerate(header)]
    edges = x_left + (x_right - x_left) * np.cumsum(chars) / sum(chars)
    slots = edges[1:]
    ax.text(x_left, y0, header[0], fontsize=8.5, color=INK2, va="top", fontweight="bold")
    for x, h in zip(slots, header[1:], strict=True):
        ax.text(x, y0, h, fontsize=8.5, color=INK2, va="top", ha="right", fontweight="bold")
    ax.plot([x_left, x_right], [y0 - 1.05 * dy] * 2, color=GRID, linewidth=0.8)
    for r, cells in enumerate(rows):
        y = y0 - (r + 1.3) * dy
        ax.text(x_left, y, cells[0], fontsize=8.5, color=INK, va="top")
        for x, cell in zip(slots, cells[1:], strict=True):
            ax.text(x, y, cell, fontsize=8.5, color=INK, va="top", ha="right")
    y = y0 - (len(rows) + 1.6) * dy
    for foot in summary.get("footer", []):
        for line in textwrap.wrap(foot, int(width * 0.5 * 13)):
            ax.text(x_left, y, line, fontsize=7.5, color=INK2, va="top")
            y -= dy


def plot_design(
    result: dict[str, Any],
    realization,
    tr: float,
    path: str | Path | None = None,
    title: str | None = None,
    corr: np.ndarray | None = None,
    needed: np.ndarray | None = None,
    needed_label: str = "",
):
    """Events and regressors of a realization's first run, plus how separable they are.

    ``corr``: regressor correlation after drift removal (from
    :func:`~.power.design_quality`); the raw correlation of ``result`` is shown
    when it is not given. ``needed``: the effect-for-80%-power matrix -- the
    diagonal is each condition against baseline, off-diagonal cells the
    difference, so a dark cell is a hard contrast.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    real = realization
    X = result["designs"][0]["X"].numpy()
    n0 = real.run_lengths[0]
    t = np.arange(n0) * tr
    n_mat = 1 + (needed is not None)
    fig = plt.figure(figsize=(9.5 + 3.6 * n_mat, 5.8))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 1 + n_mat, width_ratios=[3.2] + [1.25] * n_mat, height_ratios=[1, 1.4])
    ax_ev, ax_x = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    for ax in (ax_ev, ax_x):
        _style(ax)
    for i, (cond, dur) in enumerate(zip(real.conditions, real.durations, strict=True)):
        col = CATEGORICAL[i % len(CATEGORICAL)]
        for on in real.onsets[i][0]:
            ax_ev.broken_barh([(on, max(dur, tr / 4))], (i + 0.15, 0.7), color=col)
        ax_x.plot(t, X[:n0, i], color=col, linewidth=2, label=cond)
    ax_ev.set_yticks(np.arange(len(real.conditions)) + 0.5, real.conditions)
    ax_ev.set_xlim(0, t[-1] + tr)
    ax_ev.set_title(title or "Events, run 1", color=INK, fontsize=10, loc="left")
    ax_x.set_xlim(0, t[-1] + tr)
    ax_x.set_xlabel("time (s)", color=INK2, fontsize=9)
    ax_x.set_title("Regressors (unit peak)", color=INK, fontsize=10, loc="left")
    # Above the axes, beside the title: inside, it sat on top of the curves.
    ax_x.legend(
        frameon=False,
        fontsize=8,
        labelcolor=INK2,
        ncol=min(8, len(real.conditions)),
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.2,
    )

    names = real.conditions
    if corr is None:
        corr = np.corrcoef(X.T) if X.shape[1] > 1 else np.ones((1, 1))
        corr_title = "Regressor correlation"
    else:
        corr_title = "Regressor correlation\n(after drift removal)"
    div = LinearSegmentedColormap.from_list("div", [CATEGORICAL[0], "#f0efec", CATEGORICAL[1]])
    ax_c = fig.add_subplot(gs[:, 1])
    if len(names) > 1:
        _matrix(ax_c, *_below_diagonal(corr, names), div, -1, 1, "{:.2f}", corr_title)
    if needed is not None:
        ax_n = fig.add_subplot(gs[:, 2])
        seq = LinearSegmentedColormap.from_list("seq", ["#f0efec", BLUES[-1]])
        lower = np.where(np.tril(np.ones_like(needed, dtype=bool)), needed, np.nan)
        finite = lower[np.isfinite(lower)]
        # Scaled to its own range: from 0, a 2.46-2.82 spread was one flat dark block.
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        im = _matrix(
            ax_n,
            lower,
            names,
            seq,
            lo - 0.15 * (hi - lo + 1e-9),
            hi,
            "{:.2f}",
            f"% signal for 80% power, {needed_label}\ndiagonal: vs baseline; below: difference",
        )
        # The colour range is the matrix's own, so a small spread looks large:
        # the bar keeps the scale honest.
        cb = fig.colorbar(im, ax=ax_n, shrink=0.6, pad=0.03)
        for spine in cb.ax.spines.values():
            spine.set_visible(False)
        cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    return _finish(fig, path)


def _below_diagonal(m: np.ndarray, names: list[str]) -> tuple[np.ndarray, tuple[list, list]]:
    """The strictly-lower triangle as its own block: rows 1.., columns ..-1.

    A correlation's diagonal is 1 and its upper half a mirror; dropping them
    also drops the always-empty first row and last column.
    """
    block = np.where(np.tril(np.ones_like(m, dtype=bool), k=-1), m, np.nan)[1:, :-1]
    return block, (list(names[1:]), list(names[:-1]))


def _matrix(ax, m, names, cmap, vmin, vmax, fmt, title):
    """A labelled heat-map matrix with values printed in the cells (nan cells blank).

    ``names`` labels both axes, or is a (row names, column names) pair.
    """
    rows, cols = names if isinstance(names, tuple) else (names, names)
    im = ax.imshow(np.ma.masked_invalid(m), cmap=cmap, vmin=vmin, vmax=vmax)
    mid = (vmin + vmax) / 2
    for (i, j), v in np.ndenumerate(m):
        if np.isfinite(v):
            dark = v > mid if vmin >= 0 else abs(v) > 0.6
            ax.text(
                j,
                i,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=8,
                color=SURFACE if dark else INK,
            )
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(colors=INK2, labelsize=9)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, color=INK, fontsize=10)
    return im


def plot_design_spread(
    quality: list[dict[str, Any]],
    realizations: list[Any],
    noise_label: str,
    path: str | Path | None = None,
    title: str | None = None,
    score: np.ndarray | None = None,
    score_label: str | None = None,
):
    """How much the sampled realizations of one design differ, and what a bad one looks like.

    Columns: typical (mean correlation, median effect needed), the best and
    the worst realization. Rows: the order of events in every run, regressor
    correlation after drift removal, and the effect for 80% power at
    ``noise_label`` (one colour scale across the row). A realization's score
    is the mean of its effect-needed matrix -- every condition against
    baseline and every pairwise difference -- or ``score`` (one per
    realization, lower better; ``score_label`` names it), e.g. from -rank_by.
    ``quality`` is :func:`~.power.realizations_design_quality` output.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    names = realizations[0].conditions
    n = len(names)
    tri = np.tril(np.ones((n, n), dtype=bool))
    needed = np.array([q["needed"][noise_label] for q in quality])
    if score is None:
        score = np.array([m[tri].mean() for m in needed])
        score_label = "mean effect needed (%)"
    score = np.asarray(score, dtype=float)
    best, worst = int(np.argmin(score)), int(np.argmax(score))
    corr = np.array([q["corr"] for q in quality])

    div = LinearSegmentedColormap.from_list("div", [CATEGORICAL[0], "#f0efec", CATEGORICAL[1]])
    seq = LinearSegmentedColormap.from_list("seq", ["#f0efec", BLUES[-1]])
    lower = np.where(tri, 1.0, np.nan)
    mats = [np.median(needed, axis=0), needed[best], needed[worst]]
    lo, hi = (
        min(float(np.nanmin(m * lower)) for m in mats),
        max(float(np.nanmax(m * lower)) for m in mats),
    )

    cell = max(2.6, 0.55 * n + 1.2)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(3 * cell + 1.2, cell * 2 + 2.4),
        height_ratios=[0.8, 1, 1],
        layout="constrained",
    )
    fig.patch.set_facecolor(SURFACE)

    # Row 0: score spread, then the event order of best and worst.
    ax = axes[0, 0]
    _style(ax)
    order = np.argsort(score)
    ax.plot(range(len(score)), score[order], "o", color=BLUES[2], markersize=4)
    for k, col, lab in ((best, CATEGORICAL[2], "best"), (worst, CATEGORICAL[7], "worst")):
        x = int(np.nonzero(order == k)[0][0])
        ax.plot(x, score[k], "o", color=col, markersize=7)
        ax.annotate(
            lab, (x, score[k]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK2
        )
    ax.set_xlabel("realizations, sorted", color=INK2, fontsize=8)
    ax.set_ylabel(score_label, color=INK2, fontsize=8)
    ax.set_title(f"{len(score)} realizations", color=INK, fontsize=10, loc="left")
    for ax, k, lab in ((axes[0, 1], best, "best"), (axes[0, 2], worst, "worst")):
        _style(ax)
        real = realizations[k]
        for run in range(len(real.run_lengths)):
            for i, _cond in enumerate(real.conditions):
                for on in real.onsets[i][run]:
                    ax.broken_barh(
                        [(on, max(real.durations[i], 1.0))],
                        (run + 0.15, 0.7),
                        color=CATEGORICAL[i % len(CATEGORICAL)],
                    )
        ax.set_yticks(
            np.arange(len(real.run_lengths)) + 0.5,
            [f"run {r + 1}" for r in range(len(real.run_lengths))],
        )
        ax.invert_yaxis()
        ax.set_xlabel("time (s)", color=INK2, fontsize=8)
        ax.set_title(f"{lab}: realization {k} ({score[k]:.3g})", color=INK, fontsize=10, loc="left")

    # Row 1: correlation after drift removal.
    for ax, m, lab in (
        (axes[1, 0], corr.mean(axis=0), "mean"),
        (axes[1, 1], corr[best], "best"),
        (axes[1, 2], corr[worst], "worst"),
    ):
        if n > 1:
            _matrix(ax, *_below_diagonal(m, names), div, -1, 1, "{:.2f}", f"correlation, {lab}")
        else:
            ax.set_axis_off()

    # Row 2: effect for 80% power, one scale across the row.
    pad = 0.15 * (hi - lo + 1e-9)
    for ax, m, lab in zip(axes[2], mats, ("median", "best", "worst"), strict=True):
        im = _matrix(ax, m * lower, names, seq, lo - pad, hi, "{:.2f}", f"% for 80% power, {lab}")
    cb = fig.colorbar(im, ax=axes[2].tolist(), shrink=0.8, pad=0.02)
    for spine in cb.ax.spines.values():
        spine.set_visible(False)
    cb.ax.tick_params(colors=INK2, labelsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORICAL[i % len(CATEGORICAL)]) for i in range(n)
    ]
    fig.legend(
        handles,
        names,
        frameon=False,
        fontsize=8,
        labelcolor=INK2,
        loc="upper right",
        ncol=min(n, 8),
    )
    fig.suptitle(
        title
        or f"Across realizations, {noise_label}: effect matrices -- diagonal vs baseline, "
        "below it the difference",
        color=INK,
        fontsize=10,
        x=0.02,
        ha="left",
    )
    return _finish(fig, path)


def _plain_log_ticks(ax, axis: str) -> None:
    """Plain-number labels on a log axis (1, 2, 5 steps) instead of 6x10^0."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    target.set_minor_formatter(NullFormatter())


def plot_scan_time(
    sweep: dict[str, Any],
    noise_labels: list[str],
    contrasts: list[str],
    current_minutes: float | None = None,
    path: str | Path | None = None,
    title: str | None = None,
    max_panels: int = 6,
):
    """How long to scan: the effect needed as total scan time grows, per contrast.

    Top: effect for 80% power against total minutes (log-log), median over
    realizations with their range as a band, one line per noise level; the
    dashed guide is 1/sqrt(T) through the longest scan of each level.
    Bottom: effect x sqrt(minutes) -- flat where the design scales ideally,
    so it compares scan lengths (and designs) per unit of scan time, and a
    short scan's fixed overhead (fixation, drift terms) shows as a rise at
    the left. ``sweep`` is :func:`~.power.scan_time_sweep` output.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    rows = sweep["rows"]
    names = [
        c for c in contrasts if any(np.isfinite(r["needed"]) for r in rows if r["contrast"] == c)
    ][:max_panels]
    if not names:
        raise ValueError("no contrast has a true effect to sweep")
    colors = ramp(len(noise_labels))
    fig, axes = plt.subplots(
        2,
        len(names),
        figsize=(4.4 * len(names) + 0.6, 7.0),
        squeeze=False,
        sharex=True,
        layout="constrained",
    )
    fig.patch.set_facecolor(SURFACE)
    for col, c in enumerate(names):
        top, bot = axes[0, col], axes[1, col]
        for ax in (top, bot):
            _style(ax)
            ax.set_xscale("log")
            if current_minutes is not None:
                ax.axvline(current_minutes, color=INK2, linewidth=1, linestyle=":")
        top.set_yscale("log")
        for label, color in zip(noise_labels, colors, strict=True):
            sel = [r for r in rows if r["contrast"] == c and r["noise"] == label]
            mins = sorted({r["minutes"] for r in sel})
            by = {m: [r for r in sel if r["minutes"] == m] for m in mins}
            for key, ax in (("needed", top), ("per_minute", bot)):
                vals = [np.array([r[key] for r in by[m]]) for m in mins]
                med = [float(np.nanmedian(v)) for v in vals]
                ax.fill_between(
                    mins,
                    [float(np.nanmin(v)) for v in vals],
                    [float(np.nanmax(v)) for v in vals],
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )
                ax.plot(mins, med, "o-", color=color, linewidth=2, markersize=4, label=label)
                if key == "needed":
                    m0, v0 = mins[-1], med[-1]
                    grid = np.geomspace(mins[0], mins[-1], 20)
                    ax.plot(
                        grid,
                        v0 * np.sqrt(m0 / grid),
                        color=color,
                        linewidth=1,
                        linestyle=(0, (4, 3)),
                    )
        top.set_title(c, color=INK, fontsize=11)
        bot.set_xlabel("total scan time (minutes, all runs)", color=INK2, fontsize=9)
        _plain_log_ticks(top, "y")
    lo = min(r["minutes"] for r in rows)
    hi = max(r["minutes"] for r in rows)
    ticks = [
        t for t in (1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180) if lo * 0.9 <= t <= hi * 1.1
    ]
    for ax in axes.ravel():
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].set_ylabel("% signal for 80% power", color=INK2, fontsize=9)
    axes[1, 0].set_ylabel("% x sqrt(minutes)  (lower = more per minute)", color=INK2, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        labelcolor=INK2,
        loc="outside lower center",
        ncol=min(len(labels), 6),
    )
    fig.suptitle(
        title
        or "How long to scan -- analytic, fitted HRF assumed right. Dashed: 1/sqrt(T); "
        "band: range over realizations"
        + ("; dotted: this design" if current_minutes else ""),
        color=INK,
        fontsize=10,
    )
    return _finish(fig, path)


def plot_design_comparison(
    results: dict[str, dict[str, Any]],
    contrast: str,
    noise_labels: list[str],
    target: float = 0.8,
    path: str | Path | None = None,
    title: str | None = None,
):
    """Amplitude each design needs for ``target`` power: one row per design, dots per noise level.

    Dot: median over realizations (and true HRFs); whisker: range. Designs that
    never reach ``target`` within the sweep are drawn as an arrow at its edge.
    Lower is better -- read across a row for how much tSNR buys.

    When every design's scan time is known, a second panel shows the effect x
    sqrt(total minutes): a longer scan wins the first panel partly by having
    more data, and for a fixed design the effect falls as 1/sqrt(T), so this
    compares designs per unit of scan time.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .power import effect_needed, scan_seconds

    names = list(results)
    minutes = {n: scan_seconds(results[n]) for n in names}
    per_min = all(m is not None for m in minutes.values())
    need = {n: effect_needed(results[n], target) for n in names}
    top = max(r["amplitude"] for res in results.values() for r in res["table"])
    fig, axes = plt.subplots(
        1, 1 + per_min, figsize=(9 + 5 * per_min, 0.7 * len(names) + 1.6), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    labels = [f"{n}\n{minutes[n] / 60:.1f} min" if minutes[n] is not None else n for n in names]
    panels = [(axes[0, 0], {n: 1.0 for n in names}, top, "amplitude")]
    if per_min:
        scale = {n: float(np.sqrt(minutes[n] / 60)) for n in names}  # type: ignore[operator]
        panels.append((axes[0, 1], scale, top * max(scale.values()), "per minute"))
    colors = ramp(len(noise_labels))
    for ax, scale, edge, kind in panels:
        _comparison_panel(ax, names, need, scale, contrast, noise_labels, colors, edge)
        ax.set_yticks(range(len(names)), labels if ax is axes[0, 0] else [""] * len(names))
        if kind == "amplitude":
            ax.set_xlabel(
                f"amplitude for {target:.0%} power (% signal change)", color=INK2, fontsize=9
            )
            ax.set_title(
                title or f"{contrast}: effect needed (lower is better)",
                color=INK,
                fontsize=10,
                loc="left",
            )
        else:
            ax.set_xlabel("effect x sqrt(total minutes)", color=INK2, fontsize=9)
            ax.set_title(
                "per unit of scan time (lower = more per minute)",
                color=INK,
                fontsize=10,
                loc="left",
            )
    handles = [
        Line2D([], [], marker="o", color=col, linewidth=2, markersize=7, label=label)
        for label, col in zip(noise_labels, colors, strict=True)
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=INK2,
            marker=r"$\rightarrow$",
            linestyle="",
            markersize=12,
            label=f"not reached by {top:g}%",
        )
    )
    axes[0, -1].legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        labelcolor=INK2,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    fig.tight_layout()
    return _finish(fig, path)


def _comparison_panel(ax, names, need, scale, contrast, noise_labels, colors, edge) -> None:
    """One design-comparison panel: median dot and range whisker per design x noise level."""
    _style(ax)
    offsets = np.linspace(-0.22, 0.22, len(noise_labels)) if len(noise_labels) > 1 else [0.0]
    for row, name in enumerate(names):
        for label, col, dy in zip(noise_labels, colors, offsets, strict=True):
            v = need[name].get((label, contrast), np.full(1, np.nan)) * scale[name]
            y = row + dy
            if np.all(np.isnan(v)):
                ax.annotate(
                    "",
                    (edge * 1.05, y),
                    (edge * 0.9, y),
                    arrowprops={"arrowstyle": "->", "color": col, "lw": 2},
                )
                continue
            ax.plot([np.nanmin(v), np.nanmax(v)], [y, y], color=col, linewidth=2)
            ax.plot(
                np.nanmedian(v),
                y,
                "o",
                color=col,
                markersize=7,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
            )
    ax.set_ylim(len(names) - 0.5, -0.5)  # first design on top, every offset inside
    # Sized to the data: to the sweep's edge only when an arrow has to sit there,
    # otherwise the dots crowd into the left of an axis running to the sweep's top.
    vals = [
        need[n].get((lab, contrast), np.full(1, np.nan)) * scale[n]
        for n in names
        for lab in noise_labels
    ]
    finite = np.concatenate([v[np.isfinite(v)] for v in vals])
    unreached = any(np.all(np.isnan(v)) for v in vals)
    right = edge if unreached or not finite.size else float(finite.max())
    ax.set_xlim(0, right * (1.08 if unreached else 1.15))


def plot_detection_estimation(
    results: dict[str, dict[str, Any]],
    contrast: str,
    noise_label: str,
    target: float = 0.8,
    path: str | Path | None = None,
    title: str | None = None,
):
    """Detection against shape estimation, one colour per design: the Liu (2001) trade-off.

    x: effect ``contrast`` needs for ``target`` power; y: SD of the response
    shape's FIR estimate per bin (mean over conditions). Both in % signal at
    ``noise_label``, both lower-is-better, so the best designs sit bottom-left.
    Small dots are realizations, the large one their median. Needs the
    ``quality`` block ffs_simulate writes to each _spec.json.
    """
    import matplotlib.pyplot as plt

    from .power import effect_needed

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for i, (name, res) in enumerate(results.items()):
        q = (res.get("spec") or {}).get("quality")
        if not q or noise_label not in q.get("shape_sd", {}):
            continue
        col = CATEGORICAL[i % len(CATEGORICAL)]
        y = np.array([np.mean(v) for v in q["shape_sd"][noise_label]], dtype=float)
        x = effect_needed(res, target).get((noise_label, contrast), np.full(1, np.nan))
        if not np.isfinite(x).any() or not np.isfinite(y).any():
            continue
        if len(x) == len(y):  # one per realization, in order
            ax.scatter(x, y, s=14, color=col, alpha=0.35, linewidths=0)
        mx, my = float(np.nanmedian(x)), float(np.nanmedian(y[np.isfinite(y)]))
        ax.scatter(
            [mx], [my], s=90, color=col, edgecolors=SURFACE, linewidths=1.5, label=name, zorder=3
        )
        ax.annotate(
            name, (mx, my), xytext=(7, 5), textcoords="offset points", fontsize=8, color=INK2
        )
    ax.set_xlabel(
        f"{contrast}: % signal for {target:.0%} power (detection; lower is better)",
        color=INK2,
        fontsize=9,
    )
    ax.set_ylabel(
        "response shape: SD per FIR bin, % (estimation; lower is better)", color=INK2, fontsize=9
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # Every design is labelled at its median; a legend only covered points.
    ax.set_title(
        title
        or f"Detection vs estimation at {noise_label} -- best designs sit bottom-left; "
        "dots: realizations",
        color=INK,
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    return _finish(fig, path)


def plot_exploration(
    axes_spec: list[Any],
    configs: list[dict[str, str]],
    x: np.ndarray,
    y: np.ndarray,
    front: np.ndarray,
    keep: list[int],
    contrast: str,
    noise_label: str,
    path: str | Path | None = None,
    max_axes: int = 6,
    y_label: str | None = None,
):
    """What a design space looks like: every design, its Pareto front, and what each axis does.

    Left: detection (``x``: % signal ``contrast`` needs for 80% power) against
    response-shape estimation (``y``: SD per FIR bin), one dot per design;
    the front is the line, the shortlist numbered. Right: each explored axis
    against detection (top) and shape (bottom) -- a range as a scatter, a
    choice as one box per option. Lower is better throughout.
    """
    import matplotlib.pyplot as plt

    shown = axes_spec[:max_axes]
    ok = np.isfinite(x) & np.isfinite(y)
    fig = plt.figure(figsize=(6.5 + 2.6 * len(shown), 6.2), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 1 + len(shown), width_ratios=[2.6] + [1] * len(shown))
    ax = fig.add_subplot(gs[:, 0])
    _style(ax)
    ax.scatter(x[ok & ~front], y[ok & ~front], s=10, color=BLUES[0], alpha=0.6, linewidths=0)
    order = np.flatnonzero(front)[np.argsort(x[front])]
    ax.plot(
        x[order], y[order], "o-", color=BLUES[4], markersize=4, linewidth=1.5, label="Pareto front"
    )
    for rank, k in enumerate(keep, start=1):
        ax.scatter([x[k]], [y[k]], s=150, color=CATEGORICAL[1], edgecolors=SURFACE, zorder=3)
        ax.annotate(
            str(rank),
            (x[k], y[k]),
            ha="center",
            va="center",
            fontsize=8,
            color=SURFACE,
            fontweight="bold",
            zorder=4,
        )
    ax.set_xlabel(f"{contrast}: % signal for 80% power (detection)", color=INK2, fontsize=9)
    ax.set_ylabel(
        y_label or "response shape: SD per FIR bin, % (estimation)", color=INK2, fontsize=9
    )
    ax.set_title(
        f"{int(ok.sum())} designs at {noise_label}; front and shortlist (numbered)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    for col, a in enumerate(shown, start=1):
        vals = [a.numeric(c[a.label]) for c in configs]
        y_short = (
            "1 - reliability"
            if y_label and "reliab" in y_label
            else "steps"
            if y_label and "steps" in y_label
            else "combined"
            if y_label and "combined" in y_label
            else "shape SD %"
        )
        for row, (metric, name) in enumerate(((x, "detection %"), (y, y_short))):
            sub = fig.add_subplot(gs[row, col])
            _style(sub)
            if a.is_choice:
                groups = [metric[ok & np.array([v == ch for v in vals])] for ch in a.choices]
                sub.boxplot(
                    groups,
                    tick_labels=a.choices,
                    widths=0.6,
                    showfliers=False,
                    medianprops={"color": BLUES[4]},
                    boxprops={"color": INK2},
                    whiskerprops={"color": INK2},
                    capprops={"color": INK2},
                )
                sub.tick_params(axis="x", labelrotation=20, labelsize=7)
            else:
                v = np.array(vals, dtype=float)
                sub.scatter(v[ok], metric[ok], s=6, color=BLUES[2], alpha=0.5, linewidths=0)
                sub.scatter(v[front], metric[front], s=12, color=BLUES[4], linewidths=0)
            if row == 0:
                sub.set_title(a.label, color=INK, fontsize=9)
            if col == 1:
                sub.set_ylabel(name, color=INK2, fontsize=8)
    fig.suptitle(
        "Design space -- lower is better on every axis; analytic, fitted HRF assumed right",
        color=INK,
        fontsize=10,
    )
    return _finish(fig, path)


def plot_optimize(result: dict[str, Any], label: str, path: str | Path | None = None):
    """Search progress: the evolved best against best-of-N random at equal evaluations.

    ``result`` is :func:`~.optimize.evolve` output. The dashed line is a median
    random draw -- what an unoptimized realization gives.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ev = result["evaluations"]
    ax.plot(ev, result["history"], color=CATEGORICAL[1], linewidth=2.2, label="evolved (best)")
    ax.plot(
        ev, result["random_best"], color=BLUES[3], linewidth=2.2, label="best of N random draws"
    )
    ax.axhline(
        result["random_median"],
        color=INK2,
        linewidth=1,
        linestyle=(0, (4, 3)),
        label="median random draw",
    )
    ax.set_xlabel("realizations scored", color=INK2, fontsize=9)
    ax.set_ylabel(label + " (lower is better)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_title(
        "Design search: evolution against random sampling at the same budget",
        color=INK,
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    return _finish(fig, path)


def plot_tent(
    result: dict[str, Any],
    conditions: list[str],
    noise_labels: list[str],
    amplitudes: list[float] | np.ndarray,
    path: str | Path | None = None,
    title: str | None = None,
    max_conditions: int = 4,
):
    """The response shape one voxel would give: TENT deconvolution against the truth.

    One row per noise level, one column per condition (each its own window).
    The ink line is the
    true response to one event (the true HRF, duration included); the dots are
    the TENT estimate from one simulated voxel, the band its 95% confidence
    interval (+-1.96 SE under the noise ARMA). A band as wide as the response
    means the shape cannot be read off one voxel, however well the amplitude is
    detected. ``result`` is :func:`~.power.tent_estimate`.
    """
    import matplotlib.pyplot as plt

    names = conditions[:max_conditions]
    rows = len(noise_labels)
    fig, axes = plt.subplots(
        rows, len(names), figsize=(3.4 * len(names) + 0.8, 2.3 * rows + 1.0),
        squeeze=False, sharex="col", sharey="row", layout="constrained",
    )  # fmt: skip
    fig.patch.set_facecolor(SURFACE)
    for i, label in enumerate(noise_labels):
        for q, cond in enumerate(names):
            ax = axes[i, q]
            kn, ft = result["knots"][q], result["fine_t"][q]
            _style(ax)
            col = CATEGORICAL[q % len(CATEGORICAL)]
            est, se = result["est"][label][q], result["se"][label][q]
            ax.axhline(0, color=GRID, linewidth=1)
            ax.fill_between(kn, est - 1.96 * se, est + 1.96 * se, color=col, alpha=0.18,
                            linewidth=0)  # fmt: skip
            ax.plot(ft, result["truth"][q], color=INK, linewidth=1.8, label="true response")
            ax.plot(kn, est, "o-", color=col, linewidth=1.2, markersize=4,
                    markeredgecolor=SURFACE, label="TENT estimate, one voxel")  # fmt: skip
            if i == 0:
                ax.set_title(f"{cond} ({float(amplitudes[q]):g}%)", color=INK, fontsize=10)
            if q == 0:
                ax.set_ylabel(f"{label}\n% signal", color=INK2, fontsize=8)
            if i == rows - 1:
                ax.set_xlabel("time after onset (s)", color=INK2, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK2,
               loc="outside lower center", ncol=2)  # fmt: skip
    fig.suptitle(
        title
        or "Response shape from one voxel: TENT deconvolution (knots every "
        f"{kn[1] - kn[0]:g} s to each condition's duration + 20 s, not TR-locked); band: 95% "
        "confidence interval",
        color=INK, fontsize=10,
    )  # fmt: skip
    return _finish(fig, path)


def plot_single_trials(
    result: dict[str, Any],
    conditions: list[str],
    trial_sd: float,
    path: str | Path | None = None,
    title: str | None = None,
):
    """Single trials, estimated three ways, against the truth -- and how variable they must be.

    Panels 1-3: each trial's true deviation from its condition mean (x)
    against its estimate's (y), for LSS, LSA and single-trial ridge, coloured
    by condition; r is this voxel's, 'expected' the analytic reliability.
    Ridge shrinks its estimates, so its slope is below one; a correlation does
    not mind. Panel 4: expected reliability against trial-to-trial SD -- where
    the curve crosses 0.5, trials start to be told apart. ``result`` is
    :func:`~.power.single_trial_example`.
    """
    import matplotlib.pyplot as plt

    cond = result["cond"]

    def centred(v):
        v = np.asarray(v, dtype=float).copy()
        for q in np.unique(cond):
            if np.isfinite(v[cond == q]).any():  # LSA can be all-nan: not estimable
                v[cond == q] -= np.nanmean(v[cond == q])
        return v

    true = centred(result["true"])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    names = {"lss": "LSS", "lsa": "LSA", "ridge": "ridge"}
    for ax, m in zip(axes[:3], ("lss", "lsa", "ridge"), strict=True):
        _style(ax)
        est = centred(result[m])
        ok = np.isfinite(est)
        for q, cname in enumerate(conditions):
            sel = ok & (cond == q)
            ax.scatter(true[sel], est[sel], s=10, color=CATEGORICAL[q % len(CATEGORICAL)],
                       alpha=0.6, linewidths=0, label=cname)  # fmt: skip
        r = float(np.corrcoef(true[ok], est[ok])[0, 1]) if ok.sum() > 2 else float("nan")
        extra = f", fraction {result['ridge_frac']:.2f}" if m == "ridge" else ""
        ax.set_title(f"{names[m]}: r {r:.2f} (expected {result['expected'][m]:.2f}{extra})",
                     color=INK, fontsize=10, loc="left")  # fmt: skip
        ax.set_xlabel("true trial deviation (%)", color=INK2, fontsize=9)
        if m == "lss":
            ax.set_ylabel("estimated trial deviation (%)", color=INK2, fontsize=9)
    ax = axes[3]
    _style(ax)
    for m, col in zip(("lss", "lsa", "ridge"), (CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2]),
                      strict=True):  # fmt: skip
        ax.plot(result["sd_grid"], result["curve"][m], "o-", color=col, linewidth=2,
                markersize=4, label=names[m])  # fmt: skip
    ax.axvline(trial_sd, color=INK2, linewidth=1, linestyle=":")
    ax.axhline(0.5, color=GRID, linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("trial-to-trial SD (% signal)", color=INK2, fontsize=9)
    ax.set_ylabel("expected reliability", color=INK2, fontsize=9)
    ax.set_title("how variable trials must be", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    if len(conditions) > 1:
        axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.suptitle(
        title
        or f"Single trials at {result['noise']}: estimated vs true trial-to-trial deviations "
        f"(trial SD {trial_sd:g}%; dotted: this SD)",
        color=INK, fontsize=10,
    )  # fmt: skip
    return _finish(fig, path)


def plot_tstats(
    result: dict[str, Any],
    contrast: str,
    noise_labels: list[str],
    alpha: float,
    path: str | Path | None = None,
    title: str | None = None,
    max_panels: int = 5,
):
    """The t values behind the power: the null and an effect, corrected and naive.

    One panel per noise level. Grey: t under the null (amplitude 0), filled
    for the ARMA-corrected t, outlined for naive OLS; the ink curve is the t
    density the corrected t should follow. Colour: the corrected t at the
    effect. Dashed: the corrected +-critical t, dotted: the naive one. A naive
    null wider than the curve is why naive OLS is anticonservative -- its
    false positives are the grey mass past the dotted lines.
    ``result`` is :func:`~.power.t_example`.
    """
    import matplotlib.pyplot as plt
    from scipy import stats as st

    labels = noise_labels[:max_panels]
    colors = ramp(len(noise_labels))
    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels) + 0.6, 3.9),
                             squeeze=False, layout="constrained")  # fmt: skip
    fig.patch.set_facecolor(SURFACE)
    method = "REML" if result.get("estimator") == "reml" else "corrected"
    for ax, label, col in zip(axes[0], labels, colors, strict=False):
        _style(ax)
        t = result["t"][label]
        (c_crit, n_crit), (c_dof, _) = result["crit"][label], result["dof"][label]
        null_c, null_n = t["null"]
        eff_c = t["effect"][0]
        lo = min(np.percentile(null_n, 0.5), -c_crit * 1.4)
        hi = max(np.percentile(eff_c, 99.5), c_crit * 1.4)
        bins = np.linspace(lo, hi, 70)
        ax.hist(
            null_c, bins=bins, density=True, color="#bdbcb6", alpha=0.7, label=f"null, {method}"
        )
        ax.hist(null_n, bins=bins, density=True, histtype="step", color=INK2, linewidth=1.2,
                label="null, naive OLS")  # fmt: skip
        ax.hist(eff_c, bins=bins, density=True, color=col, alpha=0.55,
                label=f"at the effect, {method}")  # fmt: skip
        xs = np.linspace(lo, hi, 400)
        ax.plot(xs, st.t.pdf(xs, c_dof), color=INK, linewidth=1.3, label=f"t({c_dof:.0f})")
        for x in (-c_crit, c_crit):
            ax.axvline(x, color=INK, linewidth=1, linestyle=(0, (4, 3)))
        for x in (-n_crit, n_crit):
            ax.axvline(x, color=INK2, linewidth=1, linestyle=":")
        fp_c = float(np.mean(np.abs(null_c) > c_crit))
        fp_n = float(np.mean(np.abs(null_n) > n_crit))
        pw = float(np.mean(np.abs(eff_c) > c_crit))
        ax.set_title(
            f"{label}: effect {result['effects'][label]:.2f}%\n"
            f"false pos. {fp_c:.4f} {method}, {fp_n:.4f} naive; power {pw:.2f}",
            color=INK, fontsize=9, loc="left",
        )  # fmt: skip
        ax.set_xlabel(f"t ({contrast})", color=INK2, fontsize=9)
        ax.set_yticks([])
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, frameon=False, fontsize=8, labelcolor=INK2,
               loc="outside lower center", ncol=4)  # fmt: skip
    fig.suptitle(
        title
        or f"t under the null and at the effect, two-tailed p < {alpha:g} (dashed: corrected "
        "critical t; dotted: naive)",
        color=INK, fontsize=10,
    )  # fmt: skip
    return _finish(fig, path)


def plot_spectrum(
    result: dict[str, Any],
    contrasts: list[str],
    path: str | Path | None = None,
    title: str | None = None,
    max_panels: int = 4,
):
    """Why a contrast is efficient or not: its power spectrum against noise and drift.

    Per contrast: the contrast regressor's power spectrum (filled, relative),
    the noise's (grey, relative) and the fraction of each frequency the drift
    polynomials remove (dashed). Power the drift removes is lost; power where
    the noise is loud is expensive. ``result`` is :func:`~.power.design_spectrum`.
    """
    import matplotlib.pyplot as plt

    names = [c for c in contrasts if c in result["power"]][:max_panels]
    f = result["freq"]
    keep = f <= min(f.max(), 0.3)  # the HRF passes little above ~0.25 Hz
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names) + 0.6, 3.8),
                             squeeze=False, layout="constrained")  # fmt: skip
    fig.patch.set_facecolor(SURFACE)
    for q, (ax, c) in enumerate(zip(axes[0], names, strict=True)):
        _style(ax)
        col = CATEGORICAL[q % len(CATEGORICAL)]
        ax.fill_between(f[keep], result["power"][c][keep], color=col, alpha=0.45, linewidth=0,
                        label="contrast power")  # fmt: skip
        ax.plot(f[keep], result["noise_psd"][keep], color=INK2, linewidth=1.4,
                label=f"noise ({result['noise_label']})")  # fmt: skip
        ax.plot(f[keep], result["removed"][keep], color=INK, linewidth=1.2,
                linestyle=(0, (4, 3)), label="removed by the drift")  # fmt: skip
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("frequency (Hz)", color=INK2, fontsize=9)
        ax.set_title(f"{c}: {100 * result['drift_share'][c]:.0f}% of its power in the drift",
                     color=INK, fontsize=10, loc="left")  # fmt: skip
    axes[0, 0].set_ylabel("relative power / fraction", color=INK2, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK2,
               loc="outside lower center", ncol=3)  # fmt: skip
    fig.suptitle(
        title or "Where each contrast's information sits (first run): lost to the drift at "
        "low frequencies, expensive where the noise is loud",
        color=INK, fontsize=10,
    )  # fmt: skip
    return _finish(fig, path)


def plot_shape_steps(
    result: dict[str, Any],
    conditions: list[str],
    noise_labels: list[str],
    amplitudes: list[float] | np.ndarray,
    target: float = 0.8,
    path: str | Path | None = None,
    title: str | None = None,
    max_conditions: int = 4,
):
    """Power to tell two response shapes apart, against how far apart they are.

    Per condition, one line per noise level: the power to detect that the true
    shape is not the modelled one, when they are s steps apart in the ordered
    20-HRF library (about 0.16 s of peak latency per step, the width growing
    with it). Where a line crosses the target is the shape resolution of the
    design -- also how different two conditions' shapes must be to be told
    apart. ``result`` is :func:`~.power.shape_steps`.
    """
    import matplotlib.pyplot as plt

    names = conditions[:max_conditions]
    steps = np.arange(1, result["max_step"] + 1)
    colors = ramp(len(noise_labels))
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names) + 0.6, 3.8),
                             squeeze=False, layout="constrained")  # fmt: skip
    fig.patch.set_facecolor(SURFACE)
    for q, (ax, cond) in enumerate(zip(axes[0], names, strict=True)):
        _style(ax)
        ax.axhline(target, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        for label, col in zip(noise_labels, colors, strict=True):
            ax.plot(steps, result["power"][label][q], "o-", color=col, linewidth=2,
                    markersize=4, label=label)  # fmt: skip
        s80 = result["steps"][noise_labels[len(noise_labels) // 2]][q]
        tail = (
            f"; {s80:.1f} steps at {noise_labels[len(noise_labels) // 2]}"
            if np.isfinite(s80)
            else ""
        )
        ax.set_title(f"{cond} ({float(amplitudes[q]):g}%){tail}", color=INK, fontsize=10,
                     loc="left")  # fmt: skip
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("library steps apart (~0.16 s of peak latency each)", color=INK2,
                      fontsize=9)  # fmt: skip
    axes[0, 0].set_ylabel("power to tell the shapes apart", color=INK2, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK2,
               loc="outside lower center", ncol=min(len(labels), 5))  # fmt: skip
    fig.suptitle(
        title or "Shape resolution: how far apart two response shapes must be to be told apart "
        f"(dashed: {target:.0%} power)",
        color=INK, fontsize=10,
    )  # fmt: skip
    return _finish(fig, path)


def plot_soa(
    result: dict[str, Any],
    contrasts: dict[str, Any],
    this: list[dict[str, Any]] | None = None,
    target: float = 0.8,
    path: str | Path | None = None,
    title: str | None = None,
):
    """Efficiency against SOA: the classic curves, with this design marked.

    Panels: the first condition contrast, the first difference contrast (if
    any), and the response shape. Lines: a fixed SOA, jittered gaps and
    jittered with a third blank (median over realizations; band: range). The
    star is this design, at its mean SOA. Fixed SOAs are good only very short
    or long; jitter keeps short SOAs efficient; blanks help the main effect.
    ``result`` is :func:`~.power.soa_sweep`; ``this`` its scores for this design.
    """
    import matplotlib.pyplot as plt

    from .power import SOA_FAMILIES, is_difference

    ref = result["ref"]
    panels = []
    first = next((c for c, w in contrasts.items() if not is_difference(w)), None)
    diff = next((c for c, w in contrasts.items() if is_difference(w)), None)
    for c, what in ((first, "main effect"), (diff, "difference")):
        if c is not None:
            panels.append((f"{c} ({what}): % signal for {target:.0%} power",
                           lambda sc, c=c: sc["needed"][(ref, c)]))  # fmt: skip
    panels.append(("response shape: SD per FIR bin (%)", lambda sc: sc["shape_sd"][ref]))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels) + 0.6, 4.0),
                             squeeze=False, layout="constrained")  # fmt: skip
    fig.patch.set_facecolor(SURFACE)
    soa = result["soa"]
    for ax, (label, get) in zip(axes[0], panels, strict=True):
        _style(ax)
        ax.set_xscale("log")
        for fam, col in zip(SOA_FAMILIES, CATEGORICAL, strict=False):
            vals = [[get(sc) for sc in per] for per in result["families"][fam]]
            med = [float(np.median(v)) if v else np.nan for v in vals]
            ax.fill_between(soa, [min(v) if v else np.nan for v in vals],
                            [max(v) if v else np.nan for v in vals], color=col, alpha=0.15,
                            linewidth=0)  # fmt: skip
            ax.plot(soa, med, "o-", color=col, linewidth=2, markersize=3.5, label=fam)
        if this and np.isfinite(result["this_soa"]):
            v = float(np.median([get(sc) for sc in this]))
            ax.plot([result["this_soa"]], [v], "*", color=INK, markersize=14, label="this design")
        ax.set_xlabel("mean SOA, onset to onset (s)", color=INK2, fontsize=9)
        ax.set_title(label, color=INK, fontsize=9, loc="left")
        _plain_log_ticks(ax, "x")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK2,
               loc="outside lower center", ncol=len(labels))  # fmt: skip
    fig.suptitle(
        title or f"Efficiency against SOA at {ref}: the same trials, rearranged (lower is "
        "better; same run length)",
        color=INK, fontsize=10,
    )  # fmt: skip
    return _finish(fig, path)


def plot_liu(
    points: dict[str, list[tuple[float, float]]],
    k: int,
    n_conditions: int,
    path: str | Path | None = None,
    title: str | None = None,
    scatter: bool = False,
):
    """Designs on Liu et al. (2001)'s plane: estimation efficiency against detection power.

    Both axes are fractions of their theoretical bounds (Liu & Frank 2004,
    Eqs. 26-27). The curves are the theoretical trade-off -- no design sits
    outside the outer one -- for the angle theta between the HRF and the
    design's dominant eigenvector (0 deg: the outer bound). Random designs sit
    at high efficiency, blocks at high power; nothing gets both. Liu's axes
    assume white noise and a constant baseline: the effect-needed numbers
    elsewhere include the ARMA noise and the per-run drift, which cost slow,
    block-like designs most -- so a design chosen on those can sit at low R here
    (the footnote says so). ``points``:
    {label: [(xi, R), ...]} -- families and this design's realizations; with
    ``scatter`` each label is a cloud (designs, realizations), not a curve.
    """
    import matplotlib.pyplot as plt

    from .metrics import compute_efficiency_power_tradeoff

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for th, ls in ((0.0, "-"), (45.0, (0, (4, 3))), (70.0, ":")):
        curve = compute_efficiency_power_tradeoff(k, n_conditions, theta_deg=th)
        ax.plot(curve["efficiency"], curve["power"], color=INK2, linewidth=1.2, linestyle=ls,
                label=f"theoretical, theta {th:g} deg")  # fmt: skip
    cols = iter(CATEGORICAL)
    for label, pts in points.items():
        if not pts:
            continue
        xi, r = np.array(pts, dtype=float).T
        if label == "this design":
            ax.scatter(xi, r, s=90, marker="*", color=INK, zorder=4, label=label)
            continue
        col = next(cols)
        if scatter:
            big = len(pts) == 1
            ax.scatter(xi, r, s=70 if big else 12, color=col, alpha=0.9 if big else 0.45,
                       edgecolors=SURFACE if big else "none", zorder=3 if big else 2,
                       label=label)  # fmt: skip
            continue
        order = np.argsort(xi)
        ax.plot(xi[order], r[order], "o-", color=col, linewidth=1.2, markersize=4, alpha=0.9,
                label=label)  # fmt: skip
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("estimation efficiency (fraction of its bound)", color=INK2, fontsize=9)
    ax.set_ylabel("detection power (fraction of its bound)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper right")
    fig.text(
        0.01, 0.005,
        "Liu's axes assume white noise and a constant baseline. The effect needed elsewhere "
        "includes autocorrelation and drift, which cost slow designs most.",
        fontsize=7.5, color=INK2, ha="left", va="bottom",
    )  # fmt: skip
    ax.set_title(
        title or "Liu et al. (2001): estimation against detection -- random designs right, "
        "blocks up; the bound caps both",
        color=INK, fontsize=9.5, loc="left",
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _finish(fig, path)


def plot_robustness(
    result: dict[str, Any],
    contrasts: list[str],
    fit_hrf: str,
    noise_label: str,
    target: float = 0.8,
    path: str | Path | None = None,
    title: str | None = None,
    max_panels: int = 4,
):
    """What a wrong HRF costs: every library HRF as the truth, the model fitting ``fit_hrf``.

    Top: the effect each contrast needs for ``target`` power, against the true
    HRF's peak latency; dotted, the same when the model fits that HRF too (the
    shape alone, no mismatch: a slow response needs more); the dashed line is the cost when the fitted HRF is
    right; a cross at the top is a truth under which no amplitude gets there
    (power levels off, the annotation says where). Bottom: the fraction of the
    true amplitude the estimate recovers. Blocks are robust (a boxcar smooths
    shapes alike); brief events detect well only near the fitted shape.
    ``result`` is :func:`~.power.hrf_robustness`.
    """
    import matplotlib.pyplot as plt

    names = [c for c in contrasts if c in result["contrasts"]][:max_panels]
    # The library index on x (two shapes can share a peak latency; the index keeps
    # them apart), the peaks in the tick labels.
    peaks = np.arange(len(result["peaks"]))
    ticks = peaks[::3]
    tick_labels = [f"{k}\n{result['peaks'][k]:.1f} s" for k in ticks]
    fig, axes = plt.subplots(2, len(names), figsize=(4.6 * len(names) + 0.6, 6.4),
                             squeeze=False, sharex=True, height_ratios=[1.3, 1],
                             layout="constrained")  # fmt: skip
    fig.patch.set_facecolor(SURFACE)
    for q, c in enumerate(names):
        r = result["contrasts"][c]
        col = CATEGORICAL[q % len(CATEGORICAL)]
        top, bot = axes[0, q], axes[1, q]
        for ax in (top, bot):
            _style(ax)
        need = np.asarray(r["needed"], dtype=float)
        ok = np.isfinite(need)
        matched = np.asarray(r.get("matched", np.full(len(need), np.nan)), dtype=float)
        finite = np.concatenate([need[ok], matched, [r["fitted"]]])
        finite = finite[np.isfinite(finite)]
        ymax = max(float(finite.max()) * 1.25, 0.01) if finite.size else 1.0
        top.plot(peaks, matched, "s:", color=col, alpha=0.55, linewidth=1.6, markersize=3.5,
                 label="truth and fit both HRF k")  # fmt: skip
        top.plot(peaks[ok], need[ok], "o-", color=col, linewidth=2, markersize=4,
                 label=f"truth HRF k, fitting {fit_hrf}")  # fmt: skip
        top.axhline(r["fitted"], color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        for i in np.flatnonzero(~ok):
            top.plot([peaks[i]], [ymax * 0.97], "x", color=CATEGORICAL[7], markersize=8,
                     markeredgewidth=2)  # fmt: skip
            top.annotate(f"{r['ceiling'][i]:.2f}", (peaks[i], ymax * 0.97), xytext=(0, -12),
                         textcoords="offset points", ha="center", fontsize=6.5, color=INK2)  # fmt: skip
        top.set_ylim(0, ymax)
        top.set_title(f"{c}: % signal for {target:.0%} power", color=INK, fontsize=10,
                      loc="left")  # fmt: skip
        if q == 0:
            top.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper center")
        rec = np.asarray(r["recovered"], dtype=float)
        bot.plot(peaks, rec, "o-", color=col, linewidth=2, markersize=4)
        bot.axhline(1.0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        bot.set_ylim(min(0.0, float(np.nanmin(rec)) - 0.05), max(1.1, float(np.nanmax(rec)) + 0.05))
        bot.set_xticks(ticks, tick_labels)
        bot.set_xlabel("true HRF: library index / peak latency", color=INK2, fontsize=9)
        if q == 0:
            top.set_ylabel("effect needed (% signal)", color=INK2, fontsize=9)
            bot.set_ylabel("fraction recovered", color=INK2, fontsize=9)
    import textwrap

    heading = title or (
        f"The response is library HRF k, at {noise_label}: fitting {fit_hrf} (solid) or HRF k "
        "itself (dotted: no mismatch, only what that shape does to the design)\n"
        f"Dashed: the response is {fit_hrf}; ×: target not reached, number = maximum power "
        "(analytic)"
    )
    heading = "\n".join(
        textwrap.fill(line, width=int(fig.get_figwidth() * 11)) for line in heading.splitlines()
    )
    fig.suptitle(
        heading,
        color=INK, fontsize=9.5,
    )  # fmt: skip
    return _finish(fig, path)


def plot_tsnr(
    curves: dict[str, np.ndarray],
    tsnr: np.ndarray,
    points: dict[str, list[tuple[float, float]]] | None = None,
    effects: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
    target: float = 0.8,
    path: str | Path | None = None,
    title: str | None = None,
):
    """What tSNR a design needs: the effect needed against tSNR, per contrast.

    Lines: analytic, median over realizations (for a fixed noise ARMA the effect
    needed scales exactly with the noise SD, 100/tSNR). Dots: the Monte Carlo
    at the simulated tSNR levels. The grey lines are effects of 0.25-2%%: where a
    contrast's line crosses one is the tSNR that effect needs -- read off below.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    notes = []
    for q, (c, need) in enumerate(curves.items()):
        col = CATEGORICAL[q % len(CATEGORICAL)]
        ax.plot(tsnr, need, color=col, linewidth=2.2, label=c)
        if points and points.get(c):
            px, py = np.array(points[c], dtype=float).T
            ax.plot(px, py, "o", color=col, markersize=6, markeredgecolor=SURFACE)
        cross = []
        for e in effects:
            # need falls with tSNR; the tSNR where it reaches e (log-linear interpolation)
            if need.min() <= e <= need.max():
                cross.append(
                    f"{e:g}% at tSNR {np.exp(np.interp(np.log(e), np.log(need[::-1]), np.log(tsnr[::-1]))):.0f}"
                )
        if cross:
            notes.append(f"{c}: " + ", ".join(cross))
    for e in effects:
        ax.axhline(e, color=GRID, linewidth=1.2, zorder=0)
        ax.annotate(f"{e:g}%", (tsnr[-1], e), xytext=(3, 0), textcoords="offset points",
                    fontsize=7.5, color=INK2, va="center")  # fmt: skip
    _plain_log_ticks(ax, "x")
    _plain_log_ticks(ax, "y")
    ax.set_xlabel("tSNR", color=INK2, fontsize=9)
    ax.set_ylabel(f"% signal for {target:.0%} power", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    ax.set_title(title or "What tSNR this design needs -- lines analytic, dots Monte Carlo",
                 color=INK, fontsize=10, loc="left")  # fmt: skip
    if notes:
        fig.text(0.01, 0.01, "tSNR needed:  " + ";   ".join(notes), fontsize=7.5, color=INK2,
                 ha="left", va="bottom", wrap=True)  # fmt: skip
    fig.tight_layout(rect=(0, 0.06 if notes else 0, 1, 1))
    return _finish(fig, path)


def plot_design_matrix(
    task: np.ndarray,
    conditions: list[str],
    run_lengths: list[int],
    poly_degree: int,
    tr: float,
    path: str | Path | None = None,
    title: str | None = None,
):
    """The model as SPM draws it: every column of the fit, scans down, one image.

    The task regressors (the fitted HRF), then each run's Legendre drift
    polynomials -- the full X the fit uses. Each column scaled to its own
    range (dark low, light high); run boundaries in the task columns as lines.
    """
    import matplotlib.pyplot as plt

    from .power import _nuisance

    D = _nuisance(list(run_lengths), poly_degree).numpy()
    X = np.concatenate([np.asarray(task, dtype=float), D], axis=1)
    lo, hi = X.min(axis=0), X.max(axis=0)
    Xs = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
    labels = list(conditions) + [
        f"run {r + 1} P{k}" for r in range(len(run_lengths)) for k in range(poly_degree + 1)
    ]
    n_cols = X.shape[1]
    fig, ax = plt.subplots(figsize=(max(5.0, 0.32 * n_cols + 2.0), 7.5))
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(Xs, aspect="auto", cmap="gray", interpolation="nearest")
    starts = np.cumsum([0, *run_lengths])[1:-1]
    for b in starts:
        ax.axhline(b - 0.5, color=CATEGORICAL[1], linewidth=1)
    ax.axvline(len(conditions) - 0.5, color=CATEGORICAL[0], linewidth=1.5)
    ax.set_xticks(range(n_cols), labels, rotation=90, fontsize=7 if n_cols > 12 else 8)
    yt = np.linspace(0, X.shape[0] - 1, 6).astype(int)
    ax.set_yticks(yt, [f"{int(v)}\n{v * tr:.0f} s" for v in yt], fontsize=7.5)
    ax.tick_params(colors=INK2)
    ax.set_ylabel("scan", color=INK2, fontsize=9)
    ax.set_title(
        title or f"Design matrix: {len(conditions)} task column(s), then drift\n(polort "
        f"{poly_degree} per run); each column scaled to its range",
        color=INK, fontsize=9.5, loc="left",
    )  # fmt: skip
    fig.tight_layout()
    return _finish(fig, path)


def plot_hrf_recovery(
    result: dict[str, Any],
    tr: float,
    contrast: str | None = None,
    path: str | Path | None = None,
    title: str | None = None,
):
    """What an HRF mismatch costs: the true shapes, and the amplitude the fit recovers.

    Left: each true HRF (ordered by peak time, sequential ramp) against the
    fitted one (ink, dashed). Right: recovered fraction of the true amplitude,
    E[estimate] / truth, per true HRF, on the same colours -- 1.0 is unbiased.
    It is a property of the design and the two shapes, not of the noise.
    """
    import matplotlib.pyplot as plt
    import torch

    from .core import default_microtime_dt, hrfs_from_spec

    rows = [r for r in result["table"] if abs(r["true_effect"]) > 0]
    if contrast is None:
        contrast = rows[0]["contrast"]
    rows = [r for r in rows if r["contrast"] == contrast]
    top = max(r["amplitude"] for r in rows)
    truths = list(dict.fromkeys(r["true_hrf"] for r in rows))
    frac = {
        th: float(
            np.mean(
                [
                    r["expected_est"] / r["true_effect"]
                    for r in rows
                    if r["true_hrf"] == th and r["amplitude"] == top
                ]
            )
        )
        for th in truths
    }
    dt = default_microtime_dt(tr)
    cpu = torch.device("cpu")
    shapes = {th: hrfs_from_spec(th, dt, cpu)[0][1].ravel().numpy() for th in truths}
    fit = hrfs_from_spec(result.get("hrf", "spmg1"), dt, cpu)[0][1].ravel().numpy()
    peak = {th: float(np.argmax(h)) * dt for th, h in shapes.items()}
    order = sorted(truths, key=lambda th: peak[th])
    colors = dict(zip(order, ramp(len(order)), strict=True))

    fig, (ax_h, ax_r) = plt.subplots(1, 2, figsize=(12, 4.3), width_ratios=[1.1, 1])
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_h, ax_r):
        _style(ax)
    for th in order:
        h = shapes[th] / np.abs(shapes[th]).max()
        t = np.arange(h.size) * dt
        ax_h.plot(t, h, color=colors[th], linewidth=1)
        # Mark each peak: twenty unit-peak curves peaking 2.7-5.7 s apart overlap
        # at 1.0 into what reads as one flat-topped response.
        ax_h.plot(
            peak[th],
            1.0,
            "o",
            color=colors[th],
            markersize=4.5,
            markeredgecolor=SURFACE,
            markeredgewidth=0.8,
            zorder=3,
        )
    t = np.arange(fit.size) * dt
    ax_h.plot(
        t,
        fit / np.abs(fit).max(),
        color=INK,
        linewidth=2,
        linestyle=(0, (5, 3)),
        label=f"fitted ({result.get('hrf', 'spmg1')})",
    )
    ax_h.axhline(0, color=INK2, linewidth=0.8)
    ax_h.set_xlim(0, 25)
    ax_h.set_xlabel("time (s)", color=INK2, fontsize=9)
    ax_h.set_ylabel("response (unit peak)", color=INK2, fontsize=9)
    ax_h.set_title("True HRFs (dot = peak), light = earliest", color=INK, fontsize=10, loc="left")
    ax_h.legend(frameon=False, fontsize=8, labelcolor=INK2)

    x = np.arange(len(order))
    ax_r.axhline(1.0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax_r.vlines(x, 0, [frac[th] for th in order], colors=[colors[th] for th in order], linewidth=2)
    ax_r.scatter(
        x,
        [frac[th] for th in order],
        c=[colors[th] for th in order],
        s=40,
        edgecolors=SURFACE,
        linewidths=1,
        zorder=3,
    )
    ax_r.set_xticks(x, [f"{peak[th]:.1f}" for th in order], fontsize=8)
    ax_r.set_xlabel("true HRF peak (s)", color=INK2, fontsize=9)
    ax_r.set_ylabel("recovered fraction of the amplitude", color=INK2, fontsize=9)
    ax_r.set_ylim(0, max(1.15, max(frac.values()) * 1.08))
    worst = min(frac, key=frac.get)
    ax_r.set_title(
        f"{contrast}: median {np.median(list(frac.values())):.2f}, "
        f"worst {frac[worst]:.2f} ({worst})",
        color=INK,
        fontsize=10,
        loc="left",
    )
    fig.suptitle(title or "HRF mismatch: what the fitted HRF recovers", color=INK, fontsize=10)
    fig.tight_layout()
    return _finish(fig, path)


def plot_example_voxels(
    realization,
    tr: float,
    noise: list[dict[str, Any]],
    amplitude: float | list[float] = 1.0,
    true_hrf: str = "spmg1",
    run: int = 0,
    basis: str | None = None,
    notes: list[str] | None = None,
    seed: int = 0,
    path: str | Path | None = None,
    title: str | None = None,
):
    """What the data would look like: an active and a silent voxel at each noise level.

    One row per noise level (``noise`` as for the power engine, with labels).
    Each row shows, in percent signal change over one run: a voxel responding
    to every condition equally at ``amplitude`` (the level's colour), a voxel
    that does not respond at all (grey), and the noiseless response (ink).
    Event onsets are marked along the top in condition colours. A picture of
    what a tSNR level means before any statistics.

    ``amplitude`` is one value, or one per row -- e.g. the effect each level
    needs for 80% power, so every row shows a just-detectable response.
    ``basis`` says in the title where the amplitudes came from; ``notes``
    adds one remark per row (e.g. that it never reached 80%). Each row has
    its own y-scale: a shared one is set by the noisiest level and flattens
    every quieter row.
    """
    import matplotlib.pyplot as plt
    import torch

    from .core import build_task_design, default_microtime_dt, hrfs_from_spec
    from .noise import generate_thermal_physio_noise

    real = realization
    cpu = torch.device("cpu")
    dt = default_microtime_dt(tr)
    bases = hrfs_from_spec(true_hrf, dt, cpu)[0][1]
    X = build_task_design(
        real.onsets, real.durations, tr, real.run_lengths, bases, dt, device=cpu
    ).numpy()
    start = int(sum(real.run_lengths[:run]))
    n_t = real.run_lengths[run]
    t = np.arange(n_t) * tr
    unit = X[start : start + n_t].sum(axis=1)  # every condition equal, unit peak
    amps = np.atleast_1d(np.asarray(amplitude, dtype=float))
    if amps.size == 1:
        amps = np.full(len(noise), amps[0])
    if len(amps) != len(noise):
        raise ValueError(f"{len(amps)} amplitudes for {len(noise)} noise levels")

    gen = torch.Generator().manual_seed(seed)
    colors = ramp(len(noise))
    fig, axes = plt.subplots(
        len(noise), 1, figsize=(13, 1.9 * len(noise) + 0.9), sharex=True, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    notes = notes or [""] * len(noise)
    for ax, cond, col, amp, note in zip(axes[:, 0], noise, colors, amps, notes, strict=True):
        signal = amp * unit  # PSC
        _style(ax)
        kw = {k: v for k, v in cond.items() if k != "label"}
        # Two voxels, same noise statistics: column 0 responds, column 1 does not.
        n = generate_thermal_physio_noise(
            n_t, tr, baseline=100.0, n_voxels=2, device=cpu, generator=gen, **kw
        ).numpy()  # already in PSC
        ax.plot(t, n[:, 1], color="#a9a8a2", linewidth=1, label="silent voxel")
        ax.plot(t, signal + n[:, 0], color=col, linewidth=1.3, label="active voxel")
        ax.plot(t, signal, color=INK, linewidth=1.6, label="true response")
        lo = min(n.min(), (signal + n[:, 0]).min())
        hi = max(n.max(), (signal + n[:, 0]).max())
        sd = 100.0 / float(kw["tsnr"])
        ax.set_ylabel("% signal", color=INK2, fontsize=9)
        ax.set_title(
            f"{cond.get('label', '')}  (noise SD {sd:.2g}%, response {amp:.2g}%{note})",
            loc="left",
            fontsize=9,
            color=INK,
            pad=3,
        )
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + 3 * pad)
        for i, cond_name in enumerate(real.conditions):
            for on in real.onsets[i][run]:
                ax.plot(
                    [on, on],
                    [hi + 1.6 * pad, hi + 2.6 * pad],
                    color=CATEGORICAL[i % len(CATEGORICAL)],
                    linewidth=1.5,
                    label=cond_name if on == real.onsets[i][run][0] else None,
                )
    axes[-1, 0].set_xlabel("time (s)", color=INK2, fontsize=9)
    axes[-1, 0].set_xlim(0, t[-1])
    from matplotlib.lines import Line2D

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # The first row's active trace wears the lightest ramp step; the legend
    # stands for every row, so it gets a mid step and says what colour means.
    k = labels.index("active voxel")
    handles[k] = Line2D([], [], color=BLUES[3], linewidth=1.3)
    labels[k] = "active voxel (colour = noise level)"
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        labelcolor=INK2,
        loc="lower center",
        ncol=min(len(labels), 8),
    )
    fig.suptitle(
        title
        or f"Example voxels, run {run + 1}: every condition at "
        f"{basis or ', '.join(f'{a:.2g}%' for a in dict.fromkeys(amps))} "
        f"({true_hrf} response); y-scale per row",
        color=INK,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _finish(fig, path)
