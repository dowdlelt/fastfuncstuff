"""The tunewarp results table as one picture: the accuracy/smoothness frontier.

`format_results_table` already prints the frontier, and a printed frontier is a
list of rows that the reader has to hold in their head to see the *shape* of the
trade. The shape is the finding: whether the score keeps buying detail as the
field roughens, or flattens out and leaves the extra bending unpaid for. That is
a curve, and a curve belongs on axes.

The mapping is chosen so nothing has to be looked up twice:

* **x = bending energy** (log). Roughness spans orders of magnitude across a
  search -- the whole reason the table warns about "a 10x jump in bend" -- so a
  linear axis would pile every sane config into one column.
* **y = consensus score, axis inverted** so better is *up*. The axis is the same
  quantity `_mark_pareto` runs on: plotting the raw headline metric instead would
  put points off the staircase they are marked as being on, which reads as a bug
  in the marking rather than a difference of units.
* **symbol = backend**, because "which tool" is the one property you never want
  to read off a legend colour while comparing two nearby points.
* **area = seconds per fit**, on a square-root scale (perceived size goes with
  area, not radius) and floored large enough to hold the config id *inside* the
  marker. The id is what `-reproduce N` takes, so it has to be readable without
  a leader line.
* **colour = the grade band**, the same bands the table prints, since a config
  that scores well by sitting on the fold guard is exactly the point a picture
  invites you to trust by position alone.

The baseline is a horizontal line rather than a point: it has no bending energy
at all, so it has no home on a log x-axis, and it is a level to compare against
rather than a candidate to pick.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .tunestore import BASELINE, FAIL, MARGINAL, NARROW, PASS, PINNED, headline_metric

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tunestore import ConfigResult

# Marker per backend. The three force models share a family shape (triangles) so
# that "which optical-flow variant" reads as a distinction *within* a tool rather
# than as three unrelated tools.
BACKEND_MARKERS: dict[str, str] = {
    "qwarp": "o",
    "formwarp": "s",
    "optiwarp_demons": "^",
    "optiwarp_lk": "<",
    "optiwarp_hs": ">",
}
_FALLBACK_MARKER = "P"

BAND_COLORS: dict[str, str] = {
    PASS: "#2f7d32",
    NARROW: "#b58900",
    PINNED: "#c46210",
    MARGINAL: "#d84315",
}
_FAIL_COLOR = "#9e9e9e"

# Where a label may go when the centre of its marker is taken, in units of the
# label's own font size. Ordered outward: the first free slot wins, so a label
# only travels as far as it has to.
_LABEL_OFFSETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 1.6),
    (0.0, -1.6),
    (1.4, 0.0),
    (-1.4, 0.0),
    (1.3, 1.4),
    (-1.3, 1.4),
    (1.3, -1.4),
    (-1.3, -1.4),
    (0.0, 3.0),
    (0.0, -3.0),
    (2.6, 0.0),
    (-2.6, 0.0),
)

# Marker areas in points^2. The floor is set by the label, not by aesthetics: a
# three-digit config id at 7pt needs roughly this much room to sit inside the
# shape without touching its edge.
_AREA_MIN = 420.0
_AREA_MAX = 2600.0


def _areas(seconds: Sequence[float]) -> list[float]:
    """Fit times mapped to marker areas, sqrt-scaled and floored.

    Square root because a fit that takes four times as long should look twice as
    wide, not four times; and because a search that spans 2 s to 600 s otherwise
    renders every cheap config as an unreadable dot.
    """
    lo = min(seconds, default=0.0)
    hi = max(seconds, default=0.0)
    if hi <= lo:
        return [_AREA_MIN for _ in seconds]
    span = hi - lo
    return [_AREA_MIN + (_AREA_MAX - _AREA_MIN) * ((s - lo) / span) ** 0.5 for s in seconds]


def _band_color(r: ConfigResult) -> str:
    return BAND_COLORS.get(r.band, _FAIL_COLOR)


def _label_size(area: float) -> float:
    """Point size for the id drawn inside a marker of this area."""
    return max(5.5, min(11.0, area**0.5 / 3.4))


def plot_frontier(
    results: Sequence[ConfigResult],
    path: str | Path,
    *,
    recipe: str = "",
    include_failed: bool = True,
    figsize: tuple[float, float] = (13.0, 9.0),
    dpi: int = 160,
) -> str | None:
    """Write the accuracy/smoothness scatter to ``path``. Returns the path, or None.

    None means there was nothing worth drawing (no config carried a warp), not an
    error: a `-list` on a table of failures is a legitimate thing to ask for.
    """
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patheffects import withStroke

    plottable = [
        r
        for r in results
        if not r.is_baseline
        and r.backend != BASELINE
        and (include_failed or r.grade != FAIL)
        and r.bending_mean > 0
    ]
    if not plottable:
        return None

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    areas = dict(
        zip(
            [r.config_id for r in plottable],
            _areas([r.seconds_mean for r in plottable]),
            strict=True,
        )
    )

    # Biggest first, so a slow config can never bury a fast one it happens to sit
    # next to. Overlap is unavoidable in a dense search; which label survives it
    # should not be.
    for r in sorted(plottable, key=lambda r: -areas[r.config_id]):
        area = areas[r.config_id]
        failed = r.band not in BAND_COLORS
        ax.scatter(
            r.bending_mean,
            r.score_mean,
            s=area,
            marker=BACKEND_MARKERS.get(r.backend, _FALLBACK_MARKER),
            facecolor=_band_color(r),
            edgecolor="black" if r.pareto else "none",
            linewidth=1.6 if r.pareto else 0.0,
            alpha=0.3 if failed else 0.85,
            zorder=3 if r.pareto else 2,
        )

    _draw_labels(fig, ax, plottable, areas, withStroke)

    # The staircase, not a smooth line through the points: between two frontier
    # configs there is no measured setting, and a diagonal would draw one.
    front = sorted((r for r in plottable if r.pareto), key=lambda r: r.bending_mean)
    if len(front) > 1:
        ax.step(
            [r.bending_mean for r in front],
            [r.score_mean for r in front],
            where="post",
            color="black",
            linewidth=1.0,
            alpha=0.45,
            zorder=1,
        )

    base = next((r for r in results if r.is_baseline), None)
    if base is not None:
        ax.axhline(base.score_mean, color="#444444", linestyle="--", linewidth=1.0, zorder=1)
        ax.annotate(
            "no warp at all (baseline)",
            (0.005, base.score_mean),
            xycoords=("axes fraction", "data"),
            ha="left",
            va="bottom",
            fontsize=8,
            color="#444444",
        )

    ax.set_xscale("log")
    ax.invert_yaxis()  # better similarity is up
    ax.set_xlabel("bending energy  (mean squared 2nd derivative of displacement)  →  rougher")
    ax.set_ylabel("consensus score  (rank within this run)  →  better")
    metric = headline_metric(list(results))
    title = "Accuracy / smoothness frontier"
    if recipe:
        title += f"  —  {recipe}"
    if metric:
        title += f"  (jury headline: {metric})"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.18)
    ax.set_axisbelow(True)

    _add_legends(fig, ax, plottable, areas, Line2D, plt)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def _draw_labels(
    fig: Any, ax: Any, plottable: Sequence[ConfigResult], areas: dict[int, float], stroke: Any
) -> None:
    """Write each config id on its marker, nudging a label that would land on another.

    Ids are the point of the picture -- `-reproduce N` takes one -- so two configs
    at nearly the same score and roughness must not produce one unreadable smear.
    Big markers claim the centre first because their label has room to sit inside;
    the small ones move, and only far enough to be read (a nudged label still lands
    within its own marker's neighbourhood, so it needs no leader line).
    """
    fig.canvas.draw()  # transData is only meaningful once the limits are final
    placed: list[tuple[float, float, float, float]] = []
    for r in sorted(plottable, key=lambda r: -areas[r.config_id]):
        size = _label_size(areas[r.config_id])
        text = str(r.config_id)
        px, py = ax.transData.transform((r.bending_mean, r.score_mean))
        px, py = px * 72.0 / fig.dpi, py * 72.0 / fig.dpi  # display px -> points
        half_w, half_h = 0.34 * size * len(text), 0.6 * size
        dx = dy = 0.0
        for cand_x, cand_y in _LABEL_OFFSETS:
            dx, dy = cand_x * size, cand_y * size
            box = (px + dx - half_w, px + dx + half_w, py + dy - half_h, py + dy + half_h)
            if not any(
                box[0] < o[1] and o[0] < box[1] and box[2] < o[3] and o[2] < box[3] for o in placed
            ):
                break
        placed.append((px + dx - half_w, px + dx + half_w, py + dy - half_h, py + dy + half_h))
        failed = r.band not in BAND_COLORS
        ax.annotate(
            text,
            (r.bending_mean, r.score_mean),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=size,
            color="#555555" if failed else "white",
            zorder=4,
            path_effects=None
            if failed
            else [stroke(linewidth=1.4, foreground="black", alpha=0.35)],
        )


def _add_legends(
    fig: Any,
    ax: Any,
    plottable: Sequence[ConfigResult],
    areas: dict[int, float],
    line2d: Any,
    plt: Any,
) -> None:
    """Three legends, because three properties are encoded and none is optional.

    Kept separate rather than merged into one strip: a reader looking up "which
    tool is the square" should not have to skip past five time bubbles to find it.
    """
    backends = sorted({r.backend for r in plottable})
    handles = [
        line2d(
            [],
            [],
            linestyle="none",
            marker=BACKEND_MARKERS.get(b, _FALLBACK_MARKER),
            color="#555555",
            markersize=9,
            label=b,
        )
        for b in backends
    ]
    first = ax.legend(
        handles=handles, title="backend", loc="upper left", fontsize=8, framealpha=0.9
    )
    ax.add_artist(first)

    bands = [b for b in (PASS, NARROW, PINNED, MARGINAL) if any(r.band == b for r in plottable)]
    band_handles = [
        line2d([], [], linestyle="none", marker="o", color=BAND_COLORS[b], markersize=9, label=b)
        for b in bands
    ]
    if any(r.band not in BAND_COLORS for r in plottable):
        band_handles.append(
            line2d(
                [], [], linestyle="none", marker="o", color=_FAIL_COLOR, markersize=9, label=FAIL
            )
        )
    band_handles.append(
        line2d(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            color="none",
            markersize=11,
            label="on the frontier",
        )
    )
    second = ax.legend(
        handles=band_handles, title="grade", loc="upper right", fontsize=8, framealpha=0.9
    )
    ax.add_artist(second)

    # Size legend: the actual extremes rather than round numbers, so the reader
    # calibrates against fits this run really produced.
    times = sorted(r.seconds_mean for r in plottable)
    picks = {times[0], times[len(times) // 2], times[-1]}
    size_handles = [
        line2d(
            [],
            [],
            linestyle="none",
            marker="o",
            color="#777777",
            markersize=max(
                4.0, areas[next(r.config_id for r in plottable if r.seconds_mean == t)] ** 0.5 / 3.2
            ),
            label=f"{t:.0f} s",
        )
        for t in sorted(picks)
    ]
    ax.legend(
        handles=size_handles,
        title="seconds per fit",
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        labelspacing=1.4,
        borderpad=1.0,
    )
    fig.subplots_adjust(right=0.98)
    plt.setp(ax.get_xticklabels(), fontsize=8)
