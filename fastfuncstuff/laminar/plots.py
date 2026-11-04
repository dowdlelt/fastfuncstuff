"""Figures for a laminar fit: one page per ROI, the notebook's plots as functions.

Every panel is a function drawing onto an axes it is given, so a figure can be rebuilt
from any subset; :func:`plot_roi` composes the page ``ffs_laminar -plots`` writes.

Depth is drawn the way the model orders it -- superficial (pial side) first -- with the
ends named rather than numbered, so nobody has to remember which way a convention runs:
the tables say "depth 1 = deepest", the model says "k = 0 = surface", and a figure that
shows only the integer would be wrong half the time. Colour runs from superficial
(yellow) to deep (purple) in every panel that has depth.

matplotlib is imported inside the functions, on the Agg backend: a CLI must not need a
display.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fastfuncstuff.laminar.pipeline import RoiFit, depth_names, profile_pair

BEST = "#c44e52"
OTHER = "#4c72b0"
GOOD = "#55a868"


def _req(a, what: str):
    """A figure input of the fit, or a plain error when the fit was made without it."""
    if a is None:
        raise ValueError(f"this RoiFit has no {what}; refit it with fit_roi to plot it")
    return a


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True}
    )
    return plt


def _depth_colours(k: int):
    cmap = _plt().get_cmap("viridis_r")  # superficial (index 0) bright, deep dark
    return [cmap(i / max(k - 1, 1)) for i in range(k)]


def _blocks(fit: RoiFit) -> list[slice]:
    block, gap = int(fit.meta["block"]), int(fit.meta["gap"])
    return [slice(1, block), slice(block + gap + 1, block + gap + block)]


def _ends(ax, axis: str = "y") -> None:
    """Label the two ends of a depth axis instead of numbering it."""
    ticks, labels = [0, 1], ["superficial", "deep"]
    if axis == "y":
        ax.set_yticks(ticks, labels)
    else:
        ax.set_xticks(ticks, labels)


def panel_input(ax, fit: RoiFit) -> None:
    """What the model was given: both responses, the padding between, and what was masked."""
    y = _req(fit.y_obs, "data")
    t = np.arange(len(y))
    for k, c in enumerate(_depth_colours(y.shape[1])):
        ax.plot(t, y[:, k], color=c, lw=1.2)
    ax.fill_between(t, np.nanmin(y) - 0.2, np.nanmax(y) + 0.2, where=~_req(fit.rows, "row mask"), color="0.85",
                    zorder=0, label="masked out of the fit")  # fmt: skip
    ax.set(xlabel="sample (concatenated)", ylabel="signal change (%)", title="Model input")
    ax.legend(fontsize=7, loc="upper right")


def panel_free_energy(ax, fit: RoiFit) -> None:
    """F of every model at every vascular resolution, relative to the best model at that K."""
    rel = fit.per_k_F - fit.per_k_F.max(axis=0, keepdims=True)
    im = ax.imshow(rel, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(fit.k_values)), [f"K={k}" for k in fit.k_values])
    ax.set_yticks(range(len(fit.models)), fit.models, fontsize=7)
    ax.grid(False)
    ax.set_title("Free energy relative to the best model at each K")
    ax.figure.colorbar(im, ax=ax, label="dF (nats)", fraction=0.046)


def panel_model_probability(ax, fit: RoiFit) -> None:
    colours = [BEST if i == fit.best else OTHER for i in range(len(fit.models))]
    ax.bar(range(len(fit.models)), fit.p_model, color=colours)
    ax.set_xticks(range(len(fit.models)), fit.models, rotation=40, ha="right", fontsize=7)
    ax.set(ylabel="posterior probability", title="Model comparison (averaged over K)")


def panel_modulation(ax, fit: RoiFit) -> None:
    """The best model's B per neuronal depth with 90% credible intervals, and the
    probability each depth is modulated at all (marginalised over the model space)."""
    names = depth_names(len(fit.B_mean))
    x = np.arange(len(names))
    ax.bar(x, fit.B_mean, yerr=1.6448536 * fit.B_sd, capsize=4, color=GOOD, alpha=0.85)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x, [f"{n}\nP={p:.2f}" for n, p in zip(names, fit.p_depth, strict=True)])
    ax.set(ylabel="B (log change in excitability)",
           title=f"Modulation, '{fit.models[fit.best]}' (P = P(depth modulated), all models)")  # fmt: skip


def panel_fit(ax, fit: RoiFit, conditions: tuple[str, str]) -> None:
    """Data (dotted) against the model (solid) at every depth, both conditions."""
    y_obs, y_hat = _req(fit.y_obs, "data"), _req(fit.y_hat, "fit")
    cols = _depth_colours(y_obs.shape[1])
    offset = 0
    for name, sl in zip(conditions, _blocks(fit), strict=True):
        x = np.arange(sl.stop - sl.start) + offset
        for k, c in enumerate(cols):
            ax.plot(x, y_obs[sl, k], ":o", color=c, ms=3, lw=1, alpha=0.8)
            ax.plot(x, y_hat[sl, k], "-", color=c, lw=1.8)
        ax.text(x.mean(), ax.get_ylim()[1] * 0.92, name, ha="center", fontsize=8)
        offset = x[-1] + 3
    r2 = fit.r2[fit.best]
    ax.set(xlabel="sample", ylabel="signal change (%)",
           title=f"Data (dotted) and fit (solid); variance explained {r2:.3f}")  # fmt: skip


def _era(fit: RoiFit, which: int) -> tuple[np.ndarray, slice]:
    sl = _blocks(fit)[which]
    return np.arange(sl.stop - sl.start) * float(fit.meta["TR"]), sl


def panel_measured(ax, fit: RoiFit, conditions: tuple[str, str]) -> None:
    t, sl = _era(fit, 1)
    y = _req(fit.y_obs, "data")
    for k, c in enumerate(_depth_colours(y.shape[1])):
        ax.plot(t, y[sl, k], color=c, lw=1.8)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set(xlabel="time (s)", ylabel="signal change (%)",
           title=f"Measured BOLD, {conditions[1]}")  # fmt: skip


def panel_neuronal(ax, fit: RoiFit, conditions: tuple[str, str]) -> None:
    """The deveined result: the model's neuronal states at each neuronal depth."""
    t, sl = _era(fit, 1)
    for i, name in enumerate(depth_names(fit.neuronal.shape[1])):
        ax.plot(t, fit.neuronal[sl, i], lw=2, label=name)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.legend(fontsize=7)
    ax.set(xlabel="time (s)", ylabel="excitatory activity (a.u.)",
           title=f"Deveined neuronal response, {conditions[1]}")  # fmt: skip


def _pial_deep(profile: np.ndarray) -> float:
    """Pial / deep peak, the scale-free summary of the bias. Index 0 is superficial."""
    return float("nan") if abs(profile[-1]) < 1e-12 else float(profile[0] / profile[-1])


def panel_profiles(ax, fit: RoiFit, conditions: tuple[str, str] = ("cond 1", "cond 2")) -> None:
    """Peak response by depth, measured and deveined, BOTH in percent on one axis.

    The measured profile ramps toward the pial surface; the deveined one is what each
    depth's own neurons made, so the gap between them at the pial end IS the draining.
    The deepest depth draws almost nothing from below, so its two values should nearly
    coincide -- a deep end that moves a lot is a fit problem, not a result.
    """
    for which, (label, style) in enumerate(zip(conditions, ("--", "-"), strict=True)):
        meas, dev = profile_pair(fit, which)
        x = np.linspace(1, 0, len(meas))  # index 0 is superficial -> plot pial at x=1
        ax.plot(x, meas, style, color=BEST, lw=2.2, marker="o", ms=4,
                label=f"{label}: measured  ({_pial_deep(meas):.2f})")  # fmt: skip
        ax.plot(x, dev, style, color=GOOD, lw=2.2, marker="s", ms=5,
                label=f"{label}: deveined  ({_pial_deep(dev):.2f})")  # fmt: skip
    ax.axhline(0, color="0.75", lw=0.8)
    ax.set_xticks([0, 0.5, 1], ["WM", "mid", "pial"])
    ax.set(xlabel="cortical depth", ylabel="peak signal change (%)",
           title="Depth profile: measured vs deveined (pial/deep)")  # fmt: skip
    ax.legend(fontsize=6.5, loc="best")


def panel_operator(ax, fit: RoiFit) -> None:
    """The forward mixing, rows normalised: strictly triangular -- the deepest depth is its own
    neurons, the superficial one draws much of its signal from below. A per-depth scalar
    cannot express that."""
    w = np.abs(fit.W)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
    im = ax.imshow(w, cmap="magma", aspect="auto", vmin=0, vmax=1)
    names = depth_names(w.shape[1])
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right", fontsize=7)
    _ends_rows(ax, w.shape[0])
    ax.grid(False)
    ax.set(
        xlabel="neuronal depth", ylabel="measured depth", title="Forward mixing W (row-normalised)"
    )
    ax.figure.colorbar(im, ax=ax, fraction=0.046)


def _ends_rows(ax, n: int) -> None:
    ax.set_yticks([0, n - 1], ["superficial", "deep"])


def panel_drainage(ax, fit: RoiFit) -> None:
    """How much deeper each depth sources its signal than the depth mapping alone gives --
    only a draining vein can cause it; zero at the deepest depth."""
    shift = _req(fit.drainage_shift, "drainage shift")
    k = len(shift)
    ax.plot(shift, np.linspace(0, 1, k), "o-", color=OTHER, lw=2)
    ax.axvline(0, color="0.7", lw=0.8)
    ax.invert_yaxis()
    _ends(ax)
    ax.set(
        xlabel="neuronal depths deeper than the mapping", title="Vein-attributable contamination"
    )


def panel_transit(ax, fit: RoiFit) -> None:
    """The transit delay and smearing a static operator discards: BOLD at three depths in
    response to the same neuronal drive, summed over neuronal depths."""
    ir = _req(fit.impulse, "impulse response")
    tr = float(fit.meta["TR"])
    t = np.arange(ir.shape[2]) * tr
    k = ir.shape[0]
    for idx, name in ((0, "superficial"), (k // 2, "middle"), (k - 1, "deep")):
        ax.plot(t, ir[idx].sum(axis=0), lw=2, label=name)
    ax.legend(fontsize=7)
    ax.set(xlabel="time after drive (s)", ylabel="BOLD (a.u.)", title="Transit delay and smearing")


def panel_psf(ax, fit: RoiFit) -> None:
    """The depth point-spread function at each vascular resolution: a 1-D kernel as the
    reference builds it, or the central row of a measured sampling matrix."""
    drawn = False
    for k, kern in (fit.kernels or {}).items():
        if kern is None:
            continue
        kern = np.asarray(kern)
        y = kern[kern.shape[0] // 2] if kern.ndim == 2 else kern
        ax.plot(np.linspace(0, 1, 2 * len(y) + 1)[1::2], y, "o-", lw=1.6, label=f"K={k}")
        drawn = True
    if drawn:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no depth PSF", ha="center", va="center", transform=ax.transAxes)
    ax.set(
        xlabel="depth (superficial to deep)", ylabel="weight", title="Depth point-spread function"
    )


def _label(f: RoiFit) -> str:
    """ROI label for a summary axis; '!' marks a fit with warnings (stalled, inconsistent)."""
    return f"{f.name} ({f.hemi})" + (" !" if f.warnings else "")


def plot_summary(
    fits: list[RoiFit], path: str | Path, conditions: tuple[str, str] = ("cond 1", "cond 2")
) -> str:
    """Where the effect is, across ROIs: one row per ROI, one column per neuronal depth.

    Left, the probability each depth is modulated (marginalised over the whole model
    space -- prefer it to the winning model); middle, the best model's B, starred where its
    90% interval excludes zero; right, the winner and how sure the model comparison is.
    ROIs whose fit stalled or contradicts nesting are marked '!' -- look at their pages.
    """
    plt = _plt()
    n = len(fits)
    names = depth_names(len(fits[0].B_mean))
    labels = [_label(f) for f in fits]
    p = np.array([f.p_depth for f in fits])
    b = np.array([f.B_mean for f in fits])
    sd = np.array([f.B_sd for f in fits])
    fig, axes = plt.subplots(
        1, 3, figsize=(15, max(3.5, 0.45 * n + 2.5)), gridspec_kw={"width_ratios": [1, 1, 1.15]}
    )  # fmt: skip

    ax = axes[0]
    im = ax.imshow(p, vmin=0, vmax=1, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
    ax.set_yticks(range(n), labels)
    for i in range(n):
        for j in range(len(names)):
            ax.text(j, i, f"{p[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if p[i, j] < 0.6 else "black")  # fmt: skip
    ax.grid(False)
    ax.set_title(f"P(depth modulated: {conditions[1]} vs {conditions[0]})")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    lim = max(float(np.nanmax(np.abs(b))), 1e-6)
    im = ax.imshow(b, vmin=-lim, vmax=lim, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
    ax.set_yticks(range(n), [""] * n)
    for i in range(n):
        for j in range(len(names)):
            if sd[i, j] > 0 and abs(b[i, j]) > 1.6448536 * sd[i, j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=14)
    ax.grid(False)
    ax.set_title("B of the best model (* = 90% interval excludes 0)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[2]
    pm = np.array([f.p_model[f.best] for f in fits])
    ax.barh(range(n), pm, color=[BEST if f.warnings else OTHER for f in fits])
    ax.set_yticks(range(n), [""] * n)
    ax.invert_yaxis()
    for i, f in enumerate(fits):
        ax.text(min(pm[i], 0.98) + 0.01, i, f.models[f.best], va="center", fontsize=8)
    ax.set(xlim=(0, 1.5), xlabel="posterior probability of the winner",
           title="Winning model per ROI")  # fmt: skip
    fig.suptitle(f"Laminar model, {n} ROI(s): where the effect is", y=0.995)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def plot_roi(
    fit: RoiFit, path: str | Path, conditions: tuple[str, str] = ("cond 1", "cond 2")
) -> str:
    """One page for one ROI: input, evidence, modulation, fit, deveining. Returns the path."""
    plt = _plt()
    fig, axes = plt.subplots(4, 3, figsize=(16, 15))
    a = axes.ravel()
    panel_input(a[0], fit)
    panel_free_energy(a[1], fit)
    panel_model_probability(a[2], fit)
    panel_modulation(a[3], fit)
    panel_fit(a[4], fit, conditions)
    panel_profiles(a[5], fit, conditions)
    panel_measured(a[6], fit, conditions)
    panel_neuronal(a[7], fit, conditions)
    panel_operator(a[8], fit)
    panel_drainage(a[9], fit)
    panel_transit(a[10], fit)
    panel_psf(a[11], fit)
    fig.suptitle(
        f"{fit.name} ({fit.hemi}):  best model '{fit.models[fit.best]}' "
        f"(p = {fit.p_model[fit.best]:.3f}),  K = {fit.k_values}"
        + ("   [WARNING: stalled or inconsistent fit -- see the table]" if fit.warnings else ""),
        y=0.995, fontsize=12,
    )  # fmt: skip
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


__all__ = [
    "panel_drainage",
    "panel_fit",
    "panel_free_energy",
    "panel_input",
    "panel_measured",
    "panel_model_probability",
    "panel_modulation",
    "panel_neuronal",
    "panel_operator",
    "panel_profiles",
    "panel_psf",
    "panel_transit",
    "plot_roi",
    "plot_summary",
]


def plot_null(fit: RoiFit, nulls: list[RoiFit], path: str | Path) -> str:
    """One page per ROI: every statistic against its permutation distribution.

    The point of the figure is that the reader sees the null, not a p-value standing in
    for it.  A permutation distribution with a long right tail, or one whose median sits
    where the real value sits, says something no single number does.
    """
    from fastfuncstuff.laminar.pipeline import null_statistics

    plt = _plt()
    stats = null_statistics(fit, nulls)
    n = len(stats)
    cols = min(4, n)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.1 * rows), squeeze=False)
    for ax, (name, (real, p, draws)) in zip(axes.ravel(), stats.items(), strict=False):
        if draws:
            ax.hist(draws, bins=min(30, max(8, len(draws) // 4)), color="0.72",
                    edgecolor="0.45", linewidth=0.5)  # fmt: skip
            ax.axvline(float(np.percentile(draws, 95)), color="0.35", ls=":", lw=1.2,
                       label="null 95th")  # fmt: skip
        ax.axvline(real, color="crimson", lw=2, label="observed")
        ax.set_title(f"{name}\np_perm = {p:.3g}  (n = {len(draws)})", fontsize=9)
        ax.set_xlabel("statistic")
        ax.set_ylabel("permutations")
        ax.legend(fontsize=7, frameon=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    floor = 1 / (1 + len(nulls)) if nulls else float("nan")
    fig.suptitle(
        f"{fit.name} ({fit.hemi}): permutation null, {len(nulls)} replicate(s) "
        f"-- no p below {floor:.3g} is expressible"
        + ("   [the conditions were permuted; nothing here differs by design]"),
        y=0.995, fontsize=11,
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def panel_null_profile(ax, fit: RoiFit, nulls: list[RoiFit]) -> None:
    """P(modulated) per depth with the permutation null's 5-95% band behind it."""
    names = _depth_names(len(fit.p_depth))
    x = np.arange(len(names))
    if nulls:
        draws = np.array([f.p_depth for f in nulls])  # (n_perm, N)
        lo, hi = np.percentile(draws, 5, axis=0), np.percentile(draws, 95, axis=0)
        ax.fill_between(x, lo, hi, color="0.8", label="null 5-95%")
        ax.plot(x, np.median(draws, 0), color="0.45", ls="--", lw=1, label="null median")
    ax.plot(x, fit.p_depth, "o-", color="crimson", lw=2, label="observed")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(depth modulated)")
    ax.set_title("depth probability against the null")
    ax.legend(fontsize=7, frameon=False)


def _depth_names(n):
    from fastfuncstuff.laminar.pipeline import depth_names

    return depth_names(n)
