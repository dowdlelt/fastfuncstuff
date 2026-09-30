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
    if path is not None:
        import matplotlib.pyplot as plt

        fig.savefig(path, dpi=130, facecolor=SURFACE)
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
):
    """Power vs amplitude, one panel per contrast, one line per noise level.

    Line: analytic power (median over realizations/true HRFs), band: its range,
    dots: Monte Carlo. Dashed: 80%; dotted: ``effect``.
    """
    import matplotlib.pyplot as plt

    names = _effective(contrasts, pattern)[:max_panels]
    if not names:
        raise ValueError("no contrast has a true effect under this pattern")
    rows = result["table"]
    colors = ramp(len(noise_labels))
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, c in zip(axes[0], names, strict=True):
        _style(ax)
        ax.axhline(0.8, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        for label, col in zip(noise_labels, colors, strict=True):
            sel = [r for r in rows if r["noise"] == label and r["contrast"] == c]
            amps = sorted({r["amplitude"] for r in sel})
            by = {a: [r for r in sel if r["amplitude"] == a] for a in amps}
            pred = [[r["power_predicted"] for r in by[a]] for a in amps]
            med = [float(np.median(p)) for p in pred]
            ax.fill_between(
                amps,
                [min(p) for p in pred],
                [max(p) for p in pred],
                color=col,
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(amps, med, color=col, linewidth=2, label=label)
            ax.plot(
                amps,
                [np.mean([r["power"] for r in by[a]]) for a in amps],
                "o",
                color=col,
                markersize=4.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1,
            )
            if len(noise_labels) <= 4:
                # At the 50% crossing: saturated curves all end at 1.0, where
                # right-edge labels land on top of each other.
                k = int(np.argmin(np.abs(np.asarray(med) - 0.5)))
                ax.annotate(
                    label.split(" (")[0],
                    (amps[k], med[k]),
                    xytext=(6, -2),
                    textcoords="offset points",
                    fontsize=8,
                    color=INK2,
                    va="top",
                )
        if effect is not None:
            ax.axvline(effect, color=INK2, linewidth=1, linestyle=":")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(c, color=INK, fontsize=11)
        swept = next((r.get("swept", "amplitude") for r in rows if r["contrast"] == c), "amplitude")
        if swept == "difference":
            shared = next(r.get("shared", 0.0) for r in rows if r["contrast"] == c)
            on = f" on {shared:g}% shared" if shared else ""
            ax.set_xlabel(f"{c} difference (% signal change{on})", color=INK2, fontsize=9)
        else:
            ax.set_xlabel(f"{c} amplitude (% signal change)", color=INK2, fontsize=9)
    axes[0][0].set_ylabel("power", color=INK2, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        labelcolor=INK2,
        loc="lower center",
        ncol=min(len(labels), 5),
    )
    from .power import has_mismatch

    note = (
        " (HRF mismatch: the analytic line is approximate, the dots decide)"
        if has_mismatch(rows)
        else ""
    )
    fig.suptitle(
        title
        or f"Power at two-tailed p < {alpha:g} -- line: analytic (band: range over "
        f"realizations), dots: Monte Carlo; dashed: 80%{note}",
        color=INK,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _finish(fig, path)


def plot_design(
    result: dict[str, Any],
    realization,
    tr: float,
    path: str | Path | None = None,
    title: str | None = None,
):
    """Events and regressors of the first run of a realization, and regressor correlation."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    real = realization
    X = result["designs"][0]["X"].numpy()
    n0 = real.run_lengths[0]
    t = np.arange(n0) * tr
    fig = plt.figure(figsize=(13, 5.6))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], height_ratios=[1, 1.4])
    ax_ev, ax_x, ax_c = (
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[:, 1]),
    )
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
    ax_x.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=min(4, len(real.conditions)))

    corr = np.corrcoef(X.T) if X.shape[1] > 1 else np.ones((1, 1))
    cmap = LinearSegmentedColormap.from_list("div", [CATEGORICAL[0], "#f0efec", CATEGORICAL[1]])
    ax_c.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    for (i, j), v in np.ndenumerate(corr):
        ax_c.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=INK)
    ax_c.set_xticks(range(len(real.conditions)), real.conditions)
    ax_c.set_yticks(range(len(real.conditions)), real.conditions)
    ax_c.tick_params(colors=INK2, labelsize=9)
    ax_c.set_title("Regressor correlation", color=INK, fontsize=10)
    fig.tight_layout()
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
    """
    import matplotlib.pyplot as plt

    from .power import effect_needed

    names = list(results)
    colors = ramp(len(noise_labels))
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(names) + 1.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    top = max(r["amplitude"] for res in results.values() for r in res["table"])
    offsets = np.linspace(-0.22, 0.22, len(noise_labels)) if len(noise_labels) > 1 else [0.0]
    for row, name in enumerate(names):
        need = effect_needed(results[name], target)
        for label, col, dy in zip(noise_labels, colors, offsets, strict=True):
            v = need.get((label, contrast), np.full(1, np.nan))
            y = row + dy
            if np.all(np.isnan(v)):
                ax.annotate(
                    "",
                    (top * 1.05, y),
                    (top * 0.9, y),
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
    from matplotlib.lines import Line2D

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
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(len(names) - 0.5, -0.5)  # first design on top, every offset inside
    ax.set_xlim(0, top * 1.08)
    ax.set_xlabel(f"amplitude for {target:.0%} power (% signal change)", color=INK2, fontsize=9)
    ax.set_title(
        title or f"{contrast}: amplitude needed (lower is better)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=8,
        labelcolor=INK2,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
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
