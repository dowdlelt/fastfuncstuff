"""Figures and numbers for docs/prf_validation.md.

    compare.py DATA WORK OUTDIR

Reads the ffs_pyrf buckets and analyzePRF results run.sh produced, scores both
on the independent float64 evaluator in common.py, and writes PNG figures plus
numbers.json (every number the page quotes) to OUTDIR.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import (  # noqa: E402
    HEMIS,
    SES,
    SUB,
    ang_diff,
    circ_corr,
    css_predict,
    load_apr,
    load_bucket,
    load_map,
    load_mgz_pair,
    load_runs,
    occipital_view,
    pearson,
    project_runs,
    r2_percent,
)
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

FFS = "#2a78d6"
APR = "#eb6834"
VISTA = "#1baf7a"
FOURTH = "#6250d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
R2_MIN = 20.0  # percent; the vertex set agreement statistics are reported on

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


# ----------------------------------------------------------------------------- helpers


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name)
    plt.close(fig)
    print(f"  wrote {name}")


def read_time(path: Path) -> float | None:
    try:
        return float(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def log_value(path: Path, pattern: str) -> str | None:
    try:
        m = re.search(pattern, path.read_text())
    except OSError:
        return None
    return m.group(1) if m else None


def hexbin_compare(ax, a, b, label, unit="", log=False, lim=None):
    """Density scatter of b (ffs) against a (analyzePRF) with the identity line."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if log:
        ok &= (a > 0) & (b > 0)
    a, b = a[ok], b[ok]
    if lim is None:
        lo = np.quantile(np.concatenate([a, b]), 0.005)
        hi = np.quantile(np.concatenate([a, b]), 0.995)
        lim = (lo, hi)
    ax.hexbin(
        a,
        b,
        gridsize=60,
        bins="log",
        cmap="Blues",
        mincnt=1,
        linewidths=0,
        xscale="log" if log else "linear",
        yscale="log" if log else "linear",
        extent=(np.log10(lim[0]), np.log10(lim[1])) * 2 if log else lim * 2,
    )
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"analyzePRF {label}{unit}")
    ax.set_ylabel(f"ffs_pyrf {label}{unit}")
    if log:
        r = pearson(np.log(a), np.log(b))
        txt = f"r(log) = {r:.3f}\nmedian ratio {np.median(b / a):.3f}"
    else:
        r = pearson(a, b)
        txt = f"r = {r:.3f}\nmedian |Δ| {np.median(np.abs(b - a)):.3g}{unit}"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=8, color=INK)
    ax.set_title(label, fontsize=10, color=INK, loc="left")
    return r


def surface_panel(ax, view, values, sulc, roi, cmap, norm, mask=None, title=""):
    """Sphere-projected occipital map: sulcal gray underneath, data on top, ROI borders."""
    xy = view.xy
    tri = Triangulation(xy[:, 0], xy[:, 1], view.faces)
    ax.tripcolor(tri, np.where(sulc > 0, 0.55, 0.8), cmap="gray", vmin=0, vmax=1, shading="gouraud")
    vals = np.asarray(values, float).copy()
    show = np.isfinite(vals) if mask is None else (mask & np.isfinite(vals))
    face_ok = show[view.faces].all(1)
    if face_ok.any():
        tri2 = Triangulation(xy[:, 0], xy[:, 1], view.faces[face_ok])
        vals[~show] = 0
        ax.tripcolor(tri2, vals, cmap=cmap, norm=norm, shading="gouraud")
    for label in (1, 2, 3, 4):
        ind = (roi == label).astype(float)
        if ind[view.keep].any():
            ax.tricontour(tri, ind, levels=[0.5], colors="k", linewidths=0.6)
    lim = np.abs(xy[view.keep]).max() * 1.02
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9, color=INK)


def angle_wheel(ax, cmap):
    """Legend for polar angle: the visual field as a colour wheel."""
    th = np.linspace(0, 2 * np.pi, 361)
    r = np.linspace(0.45, 1, 2)
    T, R = np.meshgrid(th, r)
    ax.pcolormesh(
        R * np.cos(T), R * np.sin(T), np.rad2deg(T), cmap=cmap, vmin=0, vmax=360, shading="auto"
    )
    for deg, lab in ((0, "R"), (90, "up"), (180, "L"), (270, "down")):
        ax.text(
            1.28 * np.cos(np.deg2rad(deg)),
            1.28 * np.sin(np.deg2rad(deg)),
            lab,
            ha="center",
            va="center",
            fontsize=7,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_axis_off()


# ----------------------------------------------------------------------------- main


def main() -> None:
    data_root, work, out = (Path(p) for p in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((work / "meta.json").read_text())
    n_lh = meta["n_vertices"]["lh"]
    aperture = np.load(work / "aperture.npy")
    T = meta["n_frames"]
    numbers: dict = {
        "subject": SUB,
        "runs": meta["runs"],
        "n_vertices": n_lh + meta["n_vertices"]["rh"],
    }

    der = data_root / "derivatives"
    roi = load_mgz_pair(str(der / f"ROIs/{SUB}/{{hemi}}.ROIs_V1-4.mgz"))
    vista = {
        m: load_mgz_pair(str(der / f"prfanalyze-vista/{SUB}/{SES}/{{hemi}}.{m}.mgz"))
        for m in ("angle", "eccen", "sigma", "vexpl", "x", "y")
    }
    surf = der / f"freesurfer/{SUB}/surf"
    views, sulcs, rois = {}, {}, {}
    offset = {"lh": 0, "rh": n_lh}
    for hemi in HEMIS:
        n = meta["n_vertices"][hemi]
        sl = slice(offset[hemi], offset[hemi] + n)
        rois[hemi] = roi[sl]
        views[hemi] = occipital_view(surf / f"{hemi}.sphere", roi[sl])
        import nibabel as nib

        sulcs[hemi] = nib.freesurfer.read_morph_data(str(surf / f"{hemi}.sulc"))

    def split(values):
        return {h: np.asarray(values)[offset[h] : offset[h] + meta["n_vertices"][h]] for h in HEMIS}

    # ---------------------------------------------------------------- load fits
    tag = sys.argv[4] if len(sys.argv) > 4 else ""  # "probe": test against the 300-vertex probe
    apr = load_apr(
        work / f"apr{'_' + tag if tag else ''}_input.mat",
        work / f"apr{'_' + tag if tag else ''}_output.mat",
        n_lh,
    )
    par = load_bucket(work / "ffs_parity")
    screen = load_map(work / "ffs_cortex_gpu_screen")
    idx = apr.index
    numbers["n_screened"] = int(idx.size)

    # analyzePRF's raw params -> the same derived quantities, in pixels.
    pa = apr.params.copy()
    pa[:, 2] = np.abs(pa[:, 2])
    pa[:, 4] = np.maximum(pa[:, 4], 1e-3)
    center = (1 + 100) / 2
    apr_x = pa[:, 1] - center
    apr_y = center - pa[:, 0]
    apr_ang = np.rad2deg(np.arctan2(apr_y, apr_x)) % 360
    apr_ecc = np.hypot(apr_x, apr_y)
    apr_rf = pa[:, 2] / np.sqrt(pa[:, 4])
    ffs = {
        k: par[k][idx]
        for k in ("x", "y", "angle", "eccentricity", "rfsize", "exponent", "sigma", "r2")
    }
    ffs["r2"] = 100 * ffs["r2"]
    # analyzePRF's own derived outputs must agree with ours from its params.
    numbers["apr_derived_check_max_ang_deg"] = float(np.nanmax(np.abs(ang_diff(apr_ang, apr.ang))))

    # ---------------------------------------------------------------- independent evaluator
    print("scoring both tools with the independent evaluator ...")
    runs = load_runs(work, idx)
    R = len(runs)
    data = np.concatenate(runs, 1)
    proj = project_runs(data, T, 2)
    p_ffs = np.stack([center - ffs["y"], ffs["x"] + center, ffs["sigma"], ffs["exponent"]], 1)
    p_apr = pa[:, [0, 1, 2, 4]]
    pred_ffs = project_runs(css_predict(p_ffs, aperture, apr.hrf, R), T, 2)
    pred_apr = project_runs(css_predict(p_apr, aperture, apr.hrf, R), T, 2)
    ev_ffs, g_ffs = r2_percent(proj, pred_ffs)
    ev_apr, g_apr = r2_percent(proj, pred_apr)
    ev_apr_own, _ = r2_percent(proj, pred_apr, gain=np.maximum(pa[:, 3], 0))
    d_ev = ev_ffs - ev_apr
    numbers["evaluator"] = {
        "reproduces_apr_r2_median_abs": float(np.median(np.abs(ev_apr_own - apr.r2))),
        "reproduces_apr_r2_p99_abs": float(np.quantile(np.abs(ev_apr_own - apr.r2), 0.99)),
        "reproduces_ffs_r2_median_abs": float(np.median(np.abs(ev_ffs - ffs["r2"]))),
        "median_ffs_minus_apr": float(np.median(d_ev)),
        "frac_ffs_better_0p5": float(np.mean(d_ev > 0.5)),
        "frac_ffs_worse_0p5": float(np.mean(d_ev < -0.5)),
        "frac_within_0p5": float(np.mean(np.abs(d_ev) <= 0.5)),
    }

    # ---------------------------------------------------------------- parity statistics
    good = (apr.r2 > R2_MIN) & (ffs["r2"] > R2_MIN)
    numbers["parity"] = {"r2_min_percent": R2_MIN, "n": int(good.sum())}
    ecc_ok = good & (apr_ecc > 2) & (ffs["eccentricity"] > 2)  # angle is undefined at the centre
    pq = numbers["parity"]
    pq["x_r"] = pearson(apr_x[good], ffs["x"][good])
    pq["y_r"] = pearson(apr_y[good], ffs["y"][good])
    pq["angle_circ_r"] = circ_corr(apr_ang[ecc_ok], ffs["angle"][ecc_ok])
    pq["angle_median_abs_deg"] = float(
        np.median(np.abs(ang_diff(ffs["angle"][ecc_ok], apr_ang[ecc_ok])))
    )
    pq["angle_p90_abs_deg"] = float(
        np.quantile(np.abs(ang_diff(ffs["angle"][ecc_ok], apr_ang[ecc_ok])), 0.9)
    )
    pq["ecc_r"] = pearson(apr_ecc[good], ffs["eccentricity"][good])
    pq["ecc_median_abs_px"] = float(np.median(np.abs(apr_ecc[good] - ffs["eccentricity"][good])))
    pq["position_median_px"] = float(
        np.median(np.hypot(apr_x[good] - ffs["x"][good], apr_y[good] - ffs["y"][good]))
    )
    pq["rfsize_r_log"] = pearson(np.log(apr_rf[good]), np.log(ffs["rfsize"][good]))
    pq["rfsize_median_ratio"] = float(np.median(ffs["rfsize"][good] / apr_rf[good]))
    pq["exponent_r_log"] = pearson(np.log(pa[good, 4]), np.log(ffs["exponent"][good]))
    pq["r2_r"] = pearson(apr.r2[good], ffs["r2"][good])
    small_n = good & ((pa[:, 4] < 0.01) | (ffs["exponent"] < 0.01))
    pq["frac_small_exponent"] = float(small_n.sum() / good.sum())
    wellposed = good & ~small_n
    pq["wellposed_n"] = int(wellposed.sum())
    pq["wellposed_rfsize_r_log"] = pearson(
        np.log(apr_rf[wellposed]), np.log(ffs["rfsize"][wellposed])
    )
    pq["wellposed_exponent_r_log"] = pearson(
        np.log(pa[wellposed, 4]), np.log(ffs["exponent"][wellposed])
    )
    pq["wellposed_ev_median_diff"] = float(np.median(d_ev[wellposed]))
    pq["smalln_ev_median_diff"] = float(np.median(d_ev[small_n]))
    # Agreement as the R2 bar rises.
    pq["by_threshold"] = []
    for thr in (10, 20, 30, 50):
        m = (apr.r2 > thr) & (ffs["r2"] > thr)
        e = m & (apr_ecc > 2)
        pq["by_threshold"].append(
            {
                "r2_min": thr,
                "n": int(m.sum()),
                "angle_median_abs_deg": float(
                    np.median(np.abs(ang_diff(ffs["angle"][e], apr_ang[e])))
                ),
                "position_median_px": float(
                    np.median(np.hypot(apr_x[m] - ffs["x"][m], apr_y[m] - ffs["y"][m]))
                ),
                "rfsize_r_log": pearson(np.log(apr_rf[m]), np.log(ffs["rfsize"][m])),
            }
        )

    # ---------------------------------------------------------------- fig 1: stimulus + HRF
    fig = plt.figure(figsize=(9, 3.2))
    gs = fig.add_gridspec(2, 8, height_ratios=[1, 0.9], hspace=0.5)
    on = aperture.reshape(-1, T).mean(0)
    starts = [0] + [t for t in range(1, T) if on[t] > 0 and on[t - 1] == 0]
    picks = [s + 12 for s in starts[:4]] + [s + 30 for s in starts[:4]]
    for i, t in enumerate(picks[:8]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(aperture[:, :, t], cmap="gray_r", vmin=0, vmax=1)
        ax.set_axis_off()
        ax.set_title(f"TR {t + 1}", fontsize=7, color=MUTED)
    ax = fig.add_subplot(gs[1, :5])
    ax.plot(np.arange(1, T + 1), on * 100, color=INK, lw=1.2)
    ax.set_xlabel("TR (1 s)")
    ax.set_ylabel("% aperture on")
    ax.set_xlim(1, T)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax = fig.add_subplot(gs[1, 5:])
    from fastfuncstuff.design.hrf import load_canonical_hrf_basic

    h_ffs = load_canonical_hrf_basic(
        microtime_dt=1.0, hrf_duration=32.0, stim_duration=1.0, device=torch.device("cpu")
    ).numpy()
    ax.plot(np.arange(apr.hrf.size), apr.hrf, color=APR, lw=2.2, label="analyzePRF")
    ax.plot(np.arange(h_ffs.size), h_ffs, color=FFS, lw=1.2, ls="--", label="ffs_pyrf")
    ax.set_xlabel("s")
    ax.set_title("canonical HRF", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7)
    n = min(h_ffs.size, apr.hrf.size)
    numbers["hrf_max_abs_diff"] = float(np.max(np.abs(h_ffs[:n] - apr.hrf[:n])))
    numbers["hrf_truncated_tail_max"] = (
        float(np.max(np.abs(apr.hrf[n:]))) if apr.hrf.size > n else 0.0
    )
    save(fig, out, "fig1_stimulus.png")

    # ---------------------------------------------------------------- fig 2: screening
    in_roi = (roi >= 1) & (roi <= 3)
    kept = np.isfinite(screen) & (screen > 0.1)
    numbers["screen"] = {
        "threshold": 0.1,
        "kept": int(kept.sum()),
        "kept_frac": float(kept.mean()),
        "v1v3_kept_frac": float(kept[in_roi].mean()),
        "outside_kept_frac": float(kept[~in_roi & (roi != 4)].mean()),
        "kept_inside_v1v4_frac": float(((roi >= 1) & kept).sum() / kept.sum()),
    }
    fig, axes = plt.subplots(
        1, 3, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.2], "wspace": 0.35}
    )
    norm = Normalize(0, 0.6)
    s = split(screen)
    for ax, hemi in zip(axes[:2], HEMIS, strict=False):
        surface_panel(
            ax,
            views[hemi],
            s[hemi],
            sulcs[hemi],
            rois[hemi],
            "magma",
            norm,
            mask=s[hemi] > 0.1,
            title=f"{hemi}: screening R² (kept > 0.1)",
        )
    sm = plt.cm.ScalarMappable(norm=norm, cmap="magma")
    fig.colorbar(sm, ax=axes[1], fraction=0.05, pad=0.02, label="cross-validated R²")
    ax = axes[2]
    bins = np.linspace(-0.2, 0.8, 60)
    ax.hist(
        screen[in_roi & np.isfinite(screen)],
        bins=bins,
        color=FFS,
        alpha=0.85,
        label="V1–V3 (dataset ROIs)",
        density=True,
    )
    ax.hist(
        screen[~in_roi & (roi != 4) & np.isfinite(screen)],
        bins=bins,
        color=MUTED,
        alpha=0.5,
        label="rest of cortex",
        density=True,
    )
    ax.axvline(0.1, color=INK, lw=0.8, ls="--")
    ax.set_xlabel("screening R² (normalised histogram)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(
        f"{100 * numbers['screen']['v1v3_kept_frac']:.0f}% of V1–V3 kept, "
        f"{100 * numbers['screen']['outside_kept_frac']:.0f}% elsewhere",
        fontsize=9,
        loc="left",
    )
    save(fig, out, "fig2_screen.png")

    # ---------------------------------------------------------------- fig 3: parity scatter
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    g = good
    hexbin_compare(axes[0, 0], apr_x[g], ffs["x"][g], "x", " px", lim=(-50, 50))
    hexbin_compare(axes[0, 1], apr_y[g], ffs["y"][g], "y", " px", lim=(-50, 50))
    ax = axes[0, 2]
    e = ecc_ok
    ax.hexbin(
        apr_ang[e],
        ffs["angle"][e],
        gridsize=60,
        bins="log",
        cmap="Blues",
        mincnt=1,
        extent=(0, 360, 0, 360),
        linewidths=0,
    )
    ax.plot([0, 360], [0, 360], color=MUTED, lw=0.8, ls="--")
    ax.set_aspect("equal")
    ax.set_xlabel("analyzePRF angle (deg)")
    ax.set_ylabel("ffs_pyrf angle (deg)")
    ax.set_title("polar angle (ecc > 2 px)", fontsize=10, loc="left")
    ax.text(
        0.04,
        0.96,
        f"circular r = {pq['angle_circ_r']:.3f}\nmedian |Δ| {pq['angle_median_abs_deg']:.2f}°",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    hexbin_compare(
        axes[0, 3], apr_ecc[g], ffs["eccentricity"][g], "eccentricity", " px", lim=(0, 70)
    )
    hexbin_compare(
        axes[1, 0], apr_rf[g], ffs["rfsize"][g], "rfsize", " px", log=True, lim=(0.3, 100)
    )
    hexbin_compare(axes[1, 1], pa[g, 4], ffs["exponent"][g], "exponent", log=True, lim=(1e-3, 3))
    hexbin_compare(axes[1, 2], apr.r2[g], ffs["r2"][g], "R²", " %", lim=(R2_MIN, 95))
    ax = axes[1, 3]
    ax.hist(np.clip(d_ev, -3, 3), bins=np.linspace(-3, 3, 61), color=FFS)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("ΔR² ffs − analyzePRF (points)")
    ax.set_ylabel("vertices")
    ax.set_title("which found the better optimum", fontsize=10, loc="left")
    ev = numbers["evaluator"]
    ax.text(
        0.98,
        0.96,
        f"ffs better > 0.5: {100 * ev['frac_ffs_better_0p5']:.1f}%\n"
        f"ffs worse > 0.5: {100 * ev['frac_ffs_worse_0p5']:.1f}%\nmedian {ev['median_ffs_minus_apr']:+.3f}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
    )
    fig.suptitle(
        f"{good.sum():,} vertices with R² > {R2_MIN:.0f}% in both fits",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    save(fig, out, "fig3_parity.png")

    # ---------------------------------------------------------------- fig 4: evaluator check
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    hexbin_compare(axes[0], apr.r2, ev_apr_own, "R² (reported vs re-evaluated)", " %", lim=(0, 95))
    axes[0].set_xlabel("analyzePRF reported R² %")
    axes[0].set_ylabel("evaluator R² %, analyzePRF params")
    axes[0].set_title("evaluator reproduces analyzePRF", fontsize=9, loc="left")
    hexbin_compare(axes[1], ffs["r2"], ev_ffs, "R²", " %", lim=(0, 95))
    axes[1].set_xlabel("ffs_pyrf reported R² %")
    axes[1].set_ylabel("evaluator R² %, ffs params")
    axes[1].set_title("…and ffs_pyrf", fontsize=9, loc="left")
    ax = axes[2]
    bins = np.logspace(-3, np.log10(3), 50)
    ax.hist(pa[good, 4], bins=bins, color=APR, alpha=0.7, label="analyzePRF")
    ax.hist(ffs["exponent"][good], bins=bins, color=FFS, alpha=0.6, label="ffs_pyrf")
    ax.set_xscale("log")
    ax.axvline(0.01, color=INK, lw=0.8, ls="--")
    ax.set_xlabel("CSS exponent n")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title(
        f"{100 * pq['frac_small_exponent']:.0f}% of vertices at n < 0.01", fontsize=9, loc="left"
    )
    fig.tight_layout()
    save(fig, out, "fig4_evaluator.png")

    # ---------------------------------------------------------------- fig 5: maps
    # vistasoft's y points down (its slope against ours is negative), so its angle
    # is mirrored before comparing. Degrees per pixel are estimated from the
    # eccentricity ratio, not a regression slope (noise in the predictor biases
    # a slope low); the dataset does not state the screen extent.
    vista_ang_up = np.rad2deg(np.arctan2(-vista["y"], vista["x"])) % 360
    deg_per_px = None
    vgood = in_roi[idx] & good & (vista["vexpl"][idx] > 0.3) & (ffs["eccentricity"] > 2)
    if vgood.sum() > 100:
        deg_per_px = float(np.median(vista["eccen"][idx][vgood] / ffs["eccentricity"][vgood]))
        numbers["vista"] = {
            "deg_per_px_estimate": deg_per_px,
            "x_slope_deg_per_px": float(np.polyfit(ffs["x"][vgood], vista["x"][idx][vgood], 1)[0]),
            "y_slope_deg_per_px": float(np.polyfit(ffs["y"][vgood], vista["y"][idx][vgood], 1)[0]),
            "n": int(vgood.sum()),
            "angle_circ_r_ffs": circ_corr(vista_ang_up[idx][vgood], ffs["angle"][vgood]),
            "angle_circ_r_apr": circ_corr(vista_ang_up[idx][vgood], apr_ang[vgood]),
            "angle_median_abs_ffs": float(
                np.median(np.abs(ang_diff(vista_ang_up[idx][vgood], ffs["angle"][vgood])))
            ),
            "angle_median_abs_apr": float(
                np.median(np.abs(ang_diff(vista_ang_up[idx][vgood], apr_ang[vgood])))
            ),
            "ecc_r_ffs": pearson(vista["eccen"][idx][vgood], ffs["eccentricity"][vgood]),
            "ecc_r_apr": pearson(vista["eccen"][idx][vgood], apr_ecc[vgood]),
        }
    full = {
        k: np.full(roi.size, np.nan)
        for k in ("ffs_ang", "ffs_ecc", "ffs_r2", "apr_ang", "apr_ecc", "apr_r2")
    }
    full["ffs_ang"][idx], full["ffs_ecc"][idx], full["ffs_r2"][idx] = (
        ffs["angle"],
        ffs["eccentricity"],
        ffs["r2"],
    )
    full["apr_ang"][idx], full["apr_ecc"][idx], full["apr_r2"][idx] = apr_ang, apr_ecc, apr.r2
    v_ang = vista_ang_up
    v_ecc = vista["eccen"] / (deg_per_px or 1.0)
    rows = [
        ("ffs_pyrf", full["ffs_ang"], full["ffs_ecc"], full["ffs_r2"]),
        ("analyzePRF", full["apr_ang"], full["apr_ecc"], full["apr_r2"]),
        ("vistasoft (dataset)", v_ang, v_ecc, vista["vexpl"] * 100),
    ]
    cyc = "twilight_shifted"
    fig, axes = plt.subplots(3, 6, figsize=(15, 8.2))
    for r, (name, ang, ecc, r2) in enumerate(rows):
        for h, hemi in enumerate(HEMIS):
            sa, se, sr = split(ang), split(ecc), split(r2)
            m = sr[hemi] > R2_MIN
            surface_panel(
                axes[r, h],
                views[hemi],
                sa[hemi],
                sulcs[hemi],
                rois[hemi],
                cyc,
                Normalize(0, 360),
                m,
                f"{name}\nangle · {hemi}",
            )
            surface_panel(
                axes[r, 2 + h],
                views[hemi],
                se[hemi],
                sulcs[hemi],
                rois[hemi],
                "viridis",
                Normalize(0, 50),
                m,
                f"eccentricity · {hemi}",
            )
            surface_panel(
                axes[r, 4 + h],
                views[hemi],
                sr[hemi],
                sulcs[hemi],
                rois[hemi],
                "magma",
                Normalize(R2_MIN, 90),
                m,
                f"R² % · {hemi}",
            )
    fig.tight_layout()
    cax = fig.add_axes([0.005, 0.02, 0.06, 0.12])
    angle_wheel(cax, cyc)
    save(fig, out, "fig5_maps.png")

    # ---------------------------------------------------------------- fig 6: timecourses
    order = np.argsort(apr.r2)
    choose = []
    for q in (0.97, 0.75, 0.5):
        cand = order[int(q * (order.size - 1))]
        choose.append(cand)
    # and the vertex where the tools disagree most on R2 among well-fit ones
    cand = np.flatnonzero(good)
    choose.append(cand[np.argmax(np.abs(d_ev[cand]))])
    fig, axes = plt.subplots(
        len(choose), 2, figsize=(11, 2.1 * len(choose)), gridspec_kw={"width_ratios": [5, 1]}
    )
    t = np.arange(1, T + 1)
    for row, v in enumerate(choose):
        mean_data = proj[v].reshape(R, T).mean(0)
        ax = axes[row, 0]
        ax.plot(t, mean_data, color=MUTED, lw=0.9, label="data (mean of runs)")
        ax.plot(
            t,
            (g_apr[v] * pred_apr[v]).reshape(R, T).mean(0),
            color=APR,
            lw=2.0,
            label=f"analyzePRF  R² {ev_apr[v]:.1f}",
        )
        ax.plot(
            t,
            (g_ffs[v] * pred_ffs[v]).reshape(R, T).mean(0),
            color=FFS,
            lw=1.2,
            ls="--",
            label=f"ffs_pyrf  R² {ev_ffs[v]:.1f}",
        )
        ax.set_xlim(1, T)
        ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=3)
        ax.set_ylabel("signal")
        if row == len(choose) - 1:
            ax.set_xlabel("TR")
        ax = axes[row, 1]
        ax.imshow(
            aperture.max(2) > 0, cmap="Greys", vmin=0, vmax=4, extent=(0.5, 100.5, 100.5, 0.5)
        )
        half = 0
        for p, col in ((p_apr[v], APR), (p_ffs[v], FFS)):
            rf = p[2] / np.sqrt(p[3])
            half = max(half, abs(p[0] - 50.5) + rf, abs(p[1] - 50.5) + rf)
            ax.add_patch(plt.Circle((p[1], p[0]), rf, fill=False, color=col, lw=1.4))
            ax.plot(p[1], p[0], "+", color=col, ms=6)
        half = min(50, max(8, 1.15 * half))
        ax.set_xlim(50.5 - half, 50.5 + half)
        ax.set_ylim(50.5 + half, 50.5 - half)
        ax.plot([50.5], [50.5], "k.", ms=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"pRF ±1 rfsize ({2 * half:.0f} px wide)", fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, out, "fig6_timecourses.png")

    # ---------------------------------------------------------------- fig 7: speed
    t_cortex = read_time(work / "ffs_cortex_gpu.time")
    t_par = read_time(work / "ffs_parity.time")
    n_fit = log_value(work / "ffs_cortex_gpu.log", r"Fitting ([\d,]+) voxels")
    numbers["speed"] = {
        "analyzeprf_s": apr.elapsed,
        "analyzeprf_workers": apr.nworkers,
        "analyzeprf_vertices": int(idx.size),
        "ffs_parity_s": t_par,
        "ffs_cortex_s": t_cortex,
        "ffs_cortex_vertices_loaded": numbers["n_vertices"],
        "ffs_cortex_vertices_fit": int(n_fit.replace(",", "")) if n_fit else None,
    }
    sp = numbers["speed"]
    fig, ax = plt.subplots(figsize=(8, 2.3))
    bars = [
        (f"analyzePRF, {idx.size:,} vertices\n(CPU, {apr.nworkers} workers)", apr.elapsed, APR),
        (f"ffs_pyrf, same {idx.size:,} vertices\n(GPU, whole tool incl. I/O)", t_par, FFS),
        (f"ffs_pyrf, whole cortex {numbers['n_vertices']:,}\n(screen + fit, GPU)", t_cortex, FFS),
    ]
    bars = [b for b in bars if b[1]]
    y = np.arange(len(bars))[::-1]
    ax.barh(y, [b[1] for b in bars], color=[b[2] for b in bars], height=0.55)
    for yy, b in zip(y, bars, strict=False):
        lab = f"{b[1] / 3600:.2f} h" if b[1] > 3600 else f"{b[1]:.0f} s"
        ax.text(b[1] * 1.15, yy, lab, va="center", fontsize=8)
    ax.set_yticks(y, [b[0] for b in bars], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("wall time (s, log scale)")
    if t_par:
        sp["speedup_same_vertices"] = apr.elapsed / t_par
        ax.set_title(
            f"{sp['speedup_same_vertices']:.0f}× faster on identical vertices",
            fontsize=9,
            loc="left",
        )
    save(fig, out, "fig7_speed.png")

    (out / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
