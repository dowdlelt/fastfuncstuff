"""Figures for BSDS fits — QC and publication.

Depends only on matplotlib (a core dependency), so plotting works without the
``[dynamics]`` extra. The palette and rules follow the project's data-viz method:
a fixed-order categorical palette for **states** (never cycled), a single-hue blue
**sequential** ramp for probabilities/transitions, and a blue↔red **diverging**
map with a true gray zero for functional connectivity and state activation
profiles. Axes are recessive (no top/right spines, light grid), identity is never
color-alone (legends + labels), and text wears ink colors, not the series color.

Two entry points:

- :func:`qc_report` — one multi-panel PNG answering "did it fit, does it make
  sense?" (convergence, occupancy, lifetime, transition matrix, effective
  dimensionality, and the state probability time course + MAP ribbon for a run).
- :func:`save_publication_figures` — each key figure saved individually as a
  vector file (PDF/SVG) plus PNG, paper-ready.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# Fixed-order categorical palette for states (validated; assign by index, never
# cycle). Beyond 8 states we extend with a warning rather than recolor silently.
STATE_COLORS = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
_INK = "#0b0b0b"
_INK2 = "#52514e"
_GRID = "#e6e5e1"
_SURFACE = "#fcfcfb"

# Single-hue blue sequential ramp (near-zero recedes toward the surface).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "ffs_blue", ["#eef5fe", "#9ec5f4", "#3987e5", "#256abf", "#0d366b"]
)
# Blue<->red diverging with a neutral gray midpoint (zero reads as "nothing").
DIVERGING = LinearSegmentedColormap.from_list(
    "ffs_bwr",
    ["#184f95", "#2a78d6", "#86b6ef", "#f0efec", "#f09a99", "#e34948", "#9c1f1f"],
)


def golden_palette(n: int, *, start: float = 0.055) -> list[str]:
    """``n`` maximally-spread categorical hex colors via the golden angle.

    Stepping hue by the golden-ratio conjugate (~137.5° per index) keeps *any*
    number of successive colors well separated with no cycling — the standard
    trick for large categorical sets. Saturation and value are dithered over a
    few levels so same-family hues (the unavoidable blues/greens once you have
    dozens) still separate by lightness. Colors stay in a mid-saturation,
    mid-value band so every entry reads on the near-white surface. ``start``
    shifts the whole ramp so distinct categorical sets (states vs task
    conditions) don't come out looking identical.
    """
    import colorsys

    golden = 0.618033988749895  # golden-ratio conjugate; hue step in [0, 1)
    sats = (0.82, 0.58, 0.70)
    vals = (0.78, 0.62, 0.90)
    colors: list[str] = []
    h = start % 1.0  # fixed start -> deterministic palette
    for i in range(n):
        h = (h + golden) % 1.0
        s = sats[i % len(sats)]
        v = vals[(i // len(sats)) % len(vals)]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return colors


def state_colors(n_states: int) -> list[str]:
    """Categorical colors for ``n_states`` states, in fixed order.

    Up to 8 states use the curated, validated palette. Beyond that (e.g. BSDS
    fit with a generous ``n_states``), switch to a golden-angle palette that
    yields ``n_states`` distinct hues rather than recycling the eight — so a
    26-state ribbon is legible instead of aliasing state 0 with state 8.
    """
    if n_states <= len(STATE_COLORS):
        return STATE_COLORS[:n_states]
    return golden_palette(n_states)


def condition_colors(n_conditions: int) -> list[str]:
    """Categorical colors for task conditions (distinct ramp from the states).

    Uses the curated palette for small sets and a golden-angle palette shifted
    to a different starting hue for large ones — so a 48-condition strip is
    legible and doesn't read as the same colors as the state ribbon above it.
    """
    if n_conditions <= len(STATE_COLORS):
        return STATE_COLORS[:n_conditions]
    return golden_palette(n_conditions, start=0.5)


def _style_axes(ax) -> None:
    """Recessive axes: drop top/right spines, mute the rest, light grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_INK2, labelsize=8, length=3, width=0.8)
    ax.title.set_color(_INK)
    ax.xaxis.label.set_color(_INK2)
    ax.yaxis.label.set_color(_INK2)


def _thinned_state_labels(k: int) -> tuple[list[int], list[str]]:
    """Tick positions + ``S#`` labels, thinned so dozens of states stay readable."""
    if k <= 12:
        idx = list(range(k))
    else:
        step = 2 if k <= 26 else max(1, k // 13)
        idx = list(range(0, k, step))
    return idx, [f"S{s}" for s in idx]


def _set_state_xticks(ax, k: int) -> None:
    """State labels on a per-state bar/heatmap axis, thinned + rotated for large K.

    With a generous BSDS ``n_states`` (dozens), labelling every state overlaps
    into an unreadable smear — so beyond 12 states we show every other (or
    sparser) label, rotated and smaller.
    """
    idx, lab = _thinned_state_labels(k)
    if k <= 12:
        ax.set_xticks(idx, lab, fontsize=8)
    else:
        ax.set_xticks(idx, lab, fontsize=6, rotation=90)


def _run_time_axis(n: int, tr: float) -> tuple[np.ndarray, str]:
    if tr and tr != 1.0:
        return np.arange(n) * tr, "time (s)"
    return np.arange(n), "TR"


# --------------------------------------------------------------------------- #
# Individual panels (each takes an Axes so they compose into the QC report).   #
# --------------------------------------------------------------------------- #


def plot_convergence(model, ax) -> None:
    """Free-energy trace over VB iterations (a single series — no legend)."""
    hist = np.asarray(model.objective_history, dtype=float)
    ax.plot(np.arange(1, len(hist) + 1), hist, color="#0d366b", linewidth=2)
    ax.set_xlabel("VB iteration")
    ax.set_ylabel("free energy F")
    status = "converged" if getattr(model, "converged", False) else "max-iter"
    ax.set_title(f"Convergence ({status}, {len(hist)} iters)")
    _style_axes(ax)


def plot_state_timecourses(model, run_idx, ax, tr: float = 1.0) -> None:
    """Posterior state probabilities over time for one run (the core QC view)."""
    resp = model.responsibilities[run_idx]
    resp = resp.cpu().numpy() if hasattr(resp, "cpu") else np.asarray(resp)
    t, xlabel = _run_time_axis(resp.shape[0], tr)
    colors = state_colors(model.n_states)
    for s in range(model.n_states):
        ax.plot(t, resp[:, s], color=colors[s], linewidth=1.3, label=f"S{s}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(state)")
    ax.set_title(f"State probabilities — run {run_idx}")
    ax.legend(
        loc="upper right",
        fontsize=7,
        ncol=min(model.n_states, 4),
        frameon=False,
        labelcolor=_INK2,
        handlelength=1.2,
    )
    _style_axes(ax)


def plot_state_ribbon(model, run_idx, ax, tr: float = 1.0) -> None:
    """MAP (Viterbi) state as a colored ribbon over time."""
    path = model.viterbi_states[run_idx]
    path = path.cpu().numpy() if hasattr(path, "cpu") else np.asarray(path)
    cmap = ListedColormap(state_colors(model.n_states))
    extent = (0, len(path) * (tr if tr else 1.0), 0, 1)
    ax.imshow(
        path[np.newaxis, :],
        aspect="auto",
        cmap=cmap,
        extent=extent,
        vmin=-0.5,
        vmax=model.n_states - 0.5,
        interpolation="nearest",
    )
    ax.set_yticks([])
    ax.set_xlabel("time (s)" if tr and tr != 1.0 else "TR")
    ax.set_title(f"MAP state — run {run_idx}")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_INK2)
    ax.tick_params(colors=_INK2, labelsize=8, length=3)
    ax.title.set_color(_INK)


def plot_transition_matrix(model, ax) -> None:
    """Transition-probability heatmap (sequential; diagonal is dominant)."""
    trans = model.transition
    trans = trans.cpu().numpy() if hasattr(trans, "cpu") else np.asarray(trans)
    k = trans.shape[0]
    im = ax.imshow(trans, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="equal")
    # Per-cell probability text only when the grid is small enough to read; at
    # dozens of states it turns into unreadable overlap, so let the heatmap speak.
    if k <= 12:
        for i in range(k):
            for j in range(k):
                ax.text(
                    j,
                    i,
                    f"{trans[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=_INK if trans[i, j] < 0.6 else _SURFACE,
                )
    _set_state_xticks(ax, k)
    ax.set_yticks(*_thinned_state_labels(k))
    ax.set_xlabel("to state")
    ax.set_ylabel("from state")
    ax.set_title("Transition matrix")
    ax.title.set_color(_INK)
    ax.tick_params(colors=_INK2, labelsize=8, length=0)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors=_INK2)


def plot_occupancy(stats, ax) -> None:
    """Group fractional occupancy per state, with per-session spread."""
    occ = stats.group_occupancy
    k = len(occ)
    colors = state_colors(k)
    ax.bar(range(k), occ, color=colors, width=0.72)
    subj = np.asarray(stats.subject_occupancy)
    if subj.shape[0] > 1:  # scatter per-session values to show variability
        for s in range(k):
            ax.scatter(
                np.full(subj.shape[0], s),
                subj[:, s],
                s=10,
                color=_INK2,
                alpha=0.5,
                zorder=3,
                linewidths=0,
            )
    _set_state_xticks(ax, k)
    ax.set_ylabel("fractional occupancy")
    ax.set_title("Occupancy")
    _style_axes(ax)


def plot_lifetime(stats, ax) -> None:
    """Group mean lifetime (dwell time) per state."""
    life = stats.group_lifetime
    k = len(life)
    colors = state_colors(k)
    ax.bar(range(k), life, color=colors, width=0.72)
    unit = "s" if stats.tr and stats.tr != 1.0 else "TR"
    _set_state_xticks(ax, k)
    ax.set_ylabel(f"mean lifetime ({unit})")
    ax.set_title("Mean lifetime")
    _style_axes(ax)


def plot_effective_dim(model, ax) -> None:
    """ARD effective factor count per state (a complexity diagnostic)."""
    eff = model.effective_dim
    eff = eff.cpu().numpy() if hasattr(eff, "cpu") else np.asarray(eff)
    k = len(eff)
    ax.bar(range(k), eff, color=state_colors(k), width=0.72)
    _set_state_xticks(ax, k)
    ax.set_ylabel("active factors")
    ax.set_title("ARD effective dim")
    _style_axes(ax)


def plot_state_means(model, ax) -> None:
    """State activation profiles across ROIs (K x D, diverging around zero)."""
    means = model.state_means
    means = means.cpu().numpy() if hasattr(means, "cpu") else np.asarray(means)
    vmax = float(np.abs(means).max()) or 1.0
    im = ax.imshow(means, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(means.shape[0]), [f"S{s}" for s in range(means.shape[0])])
    ax.set_xlabel("ROI")
    ax.set_title("State means (activation)")
    ax.title.set_color(_INK)
    ax.tick_params(colors=_INK2, labelsize=8, length=0)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors=_INK2)


def plot_state_fc(stats, fig=None):
    """Per-state functional-connectivity matrices (diverging, shared scale)."""
    fc = stats.state_fc
    fc = fc.cpu().numpy() if hasattr(fc, "cpu") else np.asarray(fc)
    k = fc.shape[0]
    ncol = min(k, 4)
    nrow = (k + ncol - 1) // ncol
    if fig is None:
        fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))
    else:
        axes = fig.subplots(nrow, ncol)
    axes = np.atleast_1d(axes).ravel()
    colors = state_colors(k)
    im = None
    for s in range(k):
        ax = axes[s]
        im = ax.imshow(fc[s], cmap=DIVERGING, vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"S{s}", color=colors[s], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for s in range(k, len(axes)):
        axes[s].axis("off")
    if im is not None:
        cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors=_INK2)
        cb.set_label("correlation", color=_INK2, fontsize=8)
    fig.suptitle("Per-state functional connectivity", color=_INK, fontsize=11)
    return fig


def plot_graph_metrics(gm, fig=None):
    """Per-state graph-theoretic summary: integration, segregation, node strength."""
    k = gm.n_states
    colors = state_colors(k)
    if fig is None:
        fig = plt.figure(figsize=(10, 3.2), facecolor=_SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.4], wspace=0.42)
    a0, a1, a2 = (fig.add_subplot(gs[0, i]) for i in range(3))
    a0.bar(range(k), gm.global_efficiency, color=colors, width=0.72)
    _set_state_xticks(a0, k)
    a0.set_ylabel("global efficiency")
    a0.set_title("Integration")
    _style_axes(a0)
    a1.bar(range(k), gm.mean_clustering, color=colors, width=0.72)
    _set_state_xticks(a1, k)
    a1.set_ylabel("mean clustering")
    a1.set_title("Segregation")
    _style_axes(a1)
    im = a2.imshow(gm.strength, cmap=SEQUENTIAL, aspect="auto")
    a2.set_yticks(range(k), [f"S{s}" for s in range(k)])
    a2.set_xlabel("ROI")
    a2.set_title("Node strength")
    a2.title.set_color(_INK)
    a2.tick_params(colors=_INK2, labelsize=8, length=0)
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors=_INK2)
    fig.suptitle("Per-state network metrics", color=_INK, fontsize=11)
    return fig


def plot_switch_rate_over_time(model, ax, run_idx: int = 0, window: int = 20, tr: float = 1.0):
    """Windowed switch density within one run — bursts vs stable epochs."""
    from fastfuncstuff.dynamics.switching import windowed_switch_rate

    rate = windowed_switch_rate(model.viterbi_states[run_idx], window)
    t, xlabel = _run_time_axis(rate.shape[0], tr)
    ax.plot(t, rate, color="#256abf", linewidth=1.6)
    ax.fill_between(t, rate, color="#256abf", alpha=0.12, linewidth=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("switch rate")
    ax.set_title(f"Switching over time — run {run_idx} ({window}-frame window)")
    _style_axes(ax)


def plot_switch_paths(switch_stats, ax, top: int = 8):
    """Most frequent multi-step switch paths as a horizontal bar chart."""
    paths = switch_stats.top_paths[:top]
    labels = ["→".join(f"S{s}" for s in p) for p, _ in paths]
    counts = [c for _, c in paths]
    y = np.arange(len(paths))[::-1]  # most frequent on top
    ax.barh(y, counts, color="#256abf", height=0.7)
    ax.set_yticks(y, labels)
    ax.set_xlabel("count")
    ax.set_title("Most frequent switch paths")
    _style_axes(ax)


def plot_prediction(pred, ax):
    """Held-out predicted vs actual behaviour, with the identity line and CV score."""
    actual, predicted = np.asarray(pred.actual), np.asarray(pred.predicted)
    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    pad = 0.05 * (hi - lo or 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=_INK2, linewidth=1, linestyle="--")
    ax.scatter(actual, predicted, s=28, color="#256abf", zorder=3, linewidths=0)
    ax.set_xlabel("actual behaviour")
    ax.set_ylabel("predicted (held-out)")
    ax.set_title("LOSO prediction")
    ax.text(
        0.04,
        0.96,
        f"R²={pred.r2:.2f}\nr={pred.correlation:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=_INK,
    )
    _style_axes(ax)


def plot_behavior_correlation(corrs, ax):
    """Per-state correlation of a state feature (e.g. occupancy) with behaviour."""
    corrs = np.asarray(corrs)
    k = len(corrs)
    ax.axhline(0, color=_INK2, linewidth=0.8)
    ax.bar(range(k), corrs, color=state_colors(k), width=0.72)
    _set_state_xticks(ax, k)
    ax.set_ylabel("corr with behaviour")
    ax.set_title("State feature ↔ behaviour")
    _style_axes(ax)


# --------------------------------------------------------------------------- #
# Composite entry points.                                                      #
# --------------------------------------------------------------------------- #


def qc_report(model, stats, path: str | Path, run_idx: int = 0, tr: float = 1.0):
    """One multi-panel QC figure: did it fit, do the states make sense?"""
    fig = plt.figure(figsize=(13, 9), facecolor=_SURFACE)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.45, wspace=0.32)
    plot_convergence(model, fig.add_subplot(gs[0, 0]))
    plot_occupancy(stats, fig.add_subplot(gs[0, 1]))
    plot_lifetime(stats, fig.add_subplot(gs[0, 2]))
    plot_transition_matrix(model, fig.add_subplot(gs[1, 0]))
    plot_state_means(model, fig.add_subplot(gs[1, 1]))
    plot_effective_dim(model, fig.add_subplot(gs[1, 2]))
    plot_state_timecourses(model, run_idx, fig.add_subplot(gs[2, :]), tr=tr)
    fig.suptitle(
        f"ffs_bsds QC — {model.n_states} states, {len(model.session_lengths)} session(s)",
        color=_INK,
        fontsize=13,
        y=0.995,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return path


def analysis_report(
    model, graph_metrics, switch_stats, path: str | Path, run_idx: int = 0, tr: float = 1.0
):
    """One figure for the post-hoc analyses: network metrics + switching dynamics."""
    fig = plt.figure(figsize=(13, 8), facecolor=_SURFACE)
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.42)
    top = outer[0, 0].subgridspec(1, 3, width_ratios=[1, 1, 1.4], wspace=0.42)
    a0 = fig.add_subplot(top[0, 0])
    a1 = fig.add_subplot(top[0, 1])
    a2 = fig.add_subplot(top[0, 2])
    k = model.n_states
    colors = state_colors(k)
    a0.bar(range(k), graph_metrics.global_efficiency, color=colors, width=0.72)
    _set_state_xticks(a0, k)
    a0.set_ylabel("global efficiency")
    a0.set_title("Integration")
    _style_axes(a0)
    a1.bar(range(k), graph_metrics.mean_clustering, color=colors, width=0.72)
    _set_state_xticks(a1, k)
    a1.set_ylabel("mean clustering")
    a1.set_title("Segregation")
    _style_axes(a1)
    im = a2.imshow(graph_metrics.strength, cmap=SEQUENTIAL, aspect="auto")
    a2.set_yticks(range(k), [f"S{s}" for s in range(k)])
    a2.set_xlabel("ROI")
    a2.set_title("Node strength")
    a2.title.set_color(_INK)
    a2.tick_params(colors=_INK2, labelsize=8, length=0)
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors=_INK2)
    bot = outer[1, 0].subgridspec(1, 2, width_ratios=[1.6, 1], wspace=0.3)
    plot_switch_rate_over_time(model, fig.add_subplot(bot[0, 0]), run_idx=run_idx, tr=tr)
    plot_switch_paths(switch_stats, fig.add_subplot(bot[0, 1]))
    fig.suptitle("ffs_bsds analysis — network metrics & switching", color=_INK, fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return path


def save_publication_figures(
    model,
    stats,
    stem: str | Path,
    *,
    fmt: str = "pdf",
    tr: float = 1.0,
    graph_metrics=None,
    switch_stats=None,
) -> list[str]:
    """Save each key figure individually (vector ``fmt`` + PNG), paper-ready."""
    stem = str(stem)
    written: list[str] = []

    def _save(fig, name):
        for ext in (fmt, "png"):
            out = f"{stem}_{name}.{ext}"
            fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=_SURFACE)
            written.append(out)
        plt.close(fig)

    # State probability time course (per run).
    for r in range(len(model.responsibilities)):
        fig, ax = plt.subplots(figsize=(9, 2.6), facecolor=_SURFACE)
        plot_state_timecourses(model, r, ax, tr=tr)
        _save(fig, f"timecourse_run-{r:02d}")

    fig, ax = plt.subplots(figsize=(4.5, 4), facecolor=_SURFACE)
    plot_transition_matrix(model, ax)
    _save(fig, "transition")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(8, 3.2), facecolor=_SURFACE)
    plot_occupancy(stats, a0)
    plot_lifetime(stats, a1)
    _save(fig, "occupancy_lifetime")

    fig, ax = plt.subplots(figsize=(7, 2.8), facecolor=_SURFACE)
    plot_state_means(model, ax)
    _save(fig, "state_means")

    _save(plot_state_fc(stats), "state_fc")

    if graph_metrics is not None:
        _save(plot_graph_metrics(graph_metrics), "graph_metrics")
    if switch_stats is not None:
        fig, (a0, a1) = plt.subplots(
            1, 2, figsize=(10, 3), facecolor=_SURFACE, width_ratios=[1.6, 1]
        )
        plot_switch_rate_over_time(model, a0, tr=tr)
        plot_switch_paths(switch_stats, a1)
        fig.tight_layout()
        _save(fig, "switching")
    return written


def _condition_ticks(labels: list[str]) -> list[str]:
    """Truncate long condition names for axis ticks."""
    return [ln if len(ln) <= 12 else ln[:11] + "…" for ln in labels]


def _cat_fontsize(n: int) -> float:
    """Axis-tick font size that shrinks as the number of categories grows."""
    if n <= 12:
        return 7.0
    if n <= 24:
        return 5.5
    if n <= 40:
        return 4.5
    return 3.5


def plot_state_condition_correlation(align, ax) -> None:
    """Heatmap: correlation of each state's probability with each condition's design."""
    corr = np.asarray(align.correlation)  # (K, C)
    k, c = corr.shape
    vmax = float(np.abs(corr).max()) if corr.size else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(k), [f"S{s}" for s in range(k)], fontsize=_cat_fontsize(k))
    ax.set_xticks(
        range(c),
        _condition_ticks(align.condition_labels),
        rotation=90,
        fontsize=_cat_fontsize(c),
    )
    ax.set_title("State × condition correlation")
    ax.title.set_color(_INK)
    ax.tick_params(colors=_INK2, length=0)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors=_INK2)


def plot_state_condition_contingency(align, ax) -> None:
    """Heatmap: P(state active | condition), from the label-contingency view."""
    cont = np.asarray(align.contingency)  # (K, C)
    k, c = cont.shape
    im = ax.imshow(cont, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(k), [f"S{s}" for s in range(k)], fontsize=_cat_fontsize(k))
    ax.set_xticks(
        range(c),
        _condition_ticks(align.condition_labels),
        rotation=90,
        fontsize=_cat_fontsize(c),
    )
    ax.set_title(f"P(state | condition)   NMI={align.normalized_mutual_info:.2f}")
    ax.title.set_color(_INK)
    ax.tick_params(colors=_INK2, length=0)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7, colors=_INK2)


def plot_task_state_overlay(model, align, ax, run_idx: int = 0, tr: float = 1.0) -> None:
    """State ribbon with the condition-label timecourse drawn beneath it."""
    seq = model.viterbi_states[run_idx]
    seq = seq.cpu().numpy() if hasattr(seq, "cpu") else np.asarray(seq)
    labels = np.asarray(align.labels[run_idx])
    t, unit = _run_time_axis(len(seq), tr)
    state_cmap = ListedColormap(state_colors(model.n_states))
    ax.imshow(
        seq[np.newaxis, :],
        aspect="auto",
        cmap=state_cmap,
        vmin=0,
        vmax=model.n_states - 1,
        extent=(float(t[0]), float(t[-1]), 0.55, 1.45),
        interpolation="nearest",
    )
    # Condition label strip (baseline -1 rendered as blank). Golden-angle
    # condition palette so 48 conditions stay distinct (tab20 would alias).
    c = len(align.condition_labels)
    disp = np.where(labels < 0, np.nan, labels).astype(float)
    cond_cmap = ListedColormap(condition_colors(max(c, 1)))
    ax.imshow(
        disp[np.newaxis, :],
        aspect="auto",
        cmap=cond_cmap,
        vmin=0,
        vmax=max(c - 1, 1),
        extent=(float(t[0]), float(t[-1]), -0.45, 0.45),
        interpolation="nearest",
    )
    ax.set_yticks([1.0, 0.0], ["state", "condition"])
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel(unit)
    ax.set_title(f"State vs task — run {run_idx}")
    ax.title.set_color(_INK)
    ax.tick_params(colors=_INK2, labelsize=8, length=0)


def task_alignment_report(model, align, path: str | Path, run_idx: int = 0, tr: float = 1.0):
    """One figure: correlation + contingency heatmaps and a state-vs-task overlay."""
    fig = plt.figure(figsize=(13, 6.5), facecolor=_SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8], hspace=0.55, wspace=0.3)
    plot_state_condition_correlation(align, fig.add_subplot(gs[0, 0]))
    plot_state_condition_contingency(align, fig.add_subplot(gs[0, 1]))
    plot_task_state_overlay(model, align, fig.add_subplot(gs[1, :]), run_idx=run_idx, tr=tr)
    fig.suptitle(
        f"ffs_bsds task alignment — {len(align.condition_labels)} conditions, "
        f"NMI={align.normalized_mutual_info:.2f}",
        color=_INK,
        fontsize=13,
        y=0.99,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return path
