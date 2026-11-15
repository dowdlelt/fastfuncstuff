"""What denoising and the HRF library change: figures + numbers for the docs page.

    variants.py DATA WORK OUTDIR

Four ffs_pyrf fits on the same screened vertices, each with -xval halves:
canonical HRF, HRF library, canonical + -denoise, library + -denoise. Held-out
R2 (each half predicts the other) is the referee; split-half reliability says
whether the parameters got more stable, not just the predictions better.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import (  # noqa: E402
    HEMIS,
    SUB,
    ang_diff,
    css_predict,
    load_bucket,
    load_mgz_pair,
    load_runs,
    occipital_view,
    pearson,
    project_runs,
    r2_percent,
)
from compare import APR, FFS, FOURTH, GRID, INK, MUTED, VISTA, save, surface_panel  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402

VARIANTS = [
    ("ffs_canon", "canonical HRF", FFS),
    ("ffs_lib", "HRF library", APR),
    ("ffs_canon_dn", "canonical + denoise", VISTA),
    ("ffs_lib_dn", "library + denoise", FOURTH),
]


def read_reliability(path: Path) -> dict[str, np.ndarray]:
    rows = [line.rstrip("\n").split("\t") for line in path.read_text().splitlines()]
    head = rows[0]
    arr = np.array([[float(v) for v in r] for r in rows[1:]])
    return {h: arr[:, i] for i, h in enumerate(head)}


def read_pcs(path: Path, n_runs: int, run_len: int) -> list[np.ndarray] | None:
    if not path.exists():
        return None
    pcs = np.loadtxt(path, ndmin=2)
    return [pcs[r * run_len : (r + 1) * run_len] for r in range(n_runs)]


def main() -> None:
    data_root, work, out = (Path(p) for p in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((work / "meta.json").read_text())
    n_lh = meta["n_vertices"]["lh"]
    T = meta["n_frames"]
    R = len(meta["runs"])
    aperture = np.load(work / "aperture.npy")
    der = data_root / "derivatives"
    roi = load_mgz_pair(str(der / f"ROIs/{SUB}/{{hemi}}.ROIs_V1-4.mgz"))
    surf = der / f"freesurfer/{SUB}/surf"
    offset = {"lh": 0, "rh": n_lh}

    def split(values):
        return {h: np.asarray(values)[offset[h] : offset[h] + meta["n_vertices"][h]] for h in HEMIS}

    views = {h: occipital_view(surf / f"{h}.sphere", split(roi)[h]) for h in HEMIS}
    sulcs = {h: nib.freesurfer.read_morph_data(str(surf / f"{h}.sulc")) for h in HEMIS}
    rois = split(roi)

    fits = {name: load_bucket(work / name) for name, _, _ in VARIANTS}
    base = fits["ffs_canon"]
    fit_mask = np.isfinite(base["xval_r2"])
    numbers: dict = {"n_fit": int(fit_mask.sum())}

    # ---------------------------------------------------------------- denoise details
    for name in ("ffs_canon_dn", "ffs_lib_dn"):
        log = (work / f"{name}.log").read_text()
        kept = re.findall(r"[Kk]eep(?:ing)?\s+(\d+)\s+(?:noise )?component", log)
        pool = re.findall(r"[Nn]oise pool[^\d]*([\d,]+)", log)
        numbers[name] = {
            "components": int(kept[-1]) if kept else None,
            "pool": int(pool[0].replace(",", "")) if pool else None,
        }
        pool_path = work / f"{name}_noisepool.lh.func.gii"
        if pool_path.exists():
            pm = (
                np.concatenate(
                    [
                        np.asarray(
                            nib.load(work / f"{name}_noisepool.{h}.func.gii").darrays[0].data
                        )
                        for h in HEMIS
                    ]
                )
                > 0
            )
            numbers[name]["pool"] = int(pm.sum())
            numbers[name]["pool_in_v1v4_frac"] = float(((roi >= 1) & pm).sum() / max(pm.sum(), 1))
            numbers[name]["v1v4_in_pool_frac"] = float(pm[roi >= 1].mean())

    # ---------------------------------------------------------------- held-out R2
    x0 = 100 * base["xval_r2"]
    ceiling = 100 * base["noise_ceiling"] if "noise_ceiling" in base.maps else None
    numbers["xval"] = {}
    resp = fit_mask & (x0 > 5)  # vertices with any held-out signal under the baseline
    numbers["n_responsive"] = int(resp.sum())
    for name, label, _ in VARIANTS:
        xv = 100 * fits[name]["xval_r2"]
        d = xv - x0
        e = {
            "median_xval_r2_responsive": float(np.nanmedian(xv[resp])),
            "median_xval_r2_all": float(np.nanmedian(xv[fit_mask])),
        }
        if name != "ffs_canon":
            e.update(
                {
                    "median_delta_responsive": float(np.nanmedian(d[resp])),
                    "frac_improved_responsive": float(np.nanmean(d[resp] > 0)),
                    "median_delta_v1v3": float(np.nanmedian(d[resp & (roi >= 1) & (roi <= 3)])),
                }
            )
        if ceiling is not None:
            e["median_frac_of_ceiling_responsive"] = float(
                np.nanmedian((xv / ceiling)[resp & (ceiling > 5)])
            )
        numbers["xval"][label] = e

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    bins = np.linspace(0, 80, 9)
    centers = 0.5 * (bins[1:] + bins[:-1])
    for name, label, col in VARIANTS[1:]:
        d = 100 * fits[name]["xval_r2"] - x0
        med = [
            np.nanmedian(d[resp & (x0 >= lo) & (x0 < hi)])
            for lo, hi in zip(bins[:-1], bins[1:], strict=False)
        ]
        q1 = [
            np.nanquantile(d[resp & (x0 >= lo) & (x0 < hi)], 0.25)
            for lo, hi in zip(bins[:-1], bins[1:], strict=False)
        ]
        q3 = [
            np.nanquantile(d[resp & (x0 >= lo) & (x0 < hi)], 0.75)
            for lo, hi in zip(bins[:-1], bins[1:], strict=False)
        ]
        ax.fill_between(centers, q1, q3, color=col, alpha=0.15, linewidth=0)
        ax.plot(centers, med, color=col, lw=2, marker="o", ms=4, label=label)
    ax.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("held-out R² with canonical HRF (%)")
    ax.set_ylabel("Δ held-out R² vs canonical (points)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("gain by vertex quality (median, IQR)", fontsize=9, loc="left")
    ax = axes[1]
    for name, label, col in VARIANTS:
        xv = np.sort(100 * fits[name]["xval_r2"][resp])
        ax.plot(xv, 1 - np.arange(xv.size) / xv.size, color=col, lw=1.8, label=label)
    ax.set_xlabel("held-out R² (%)")
    ax.set_ylabel("fraction of vertices ≥ x")
    ax.set_xlim(0, 90)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(color=GRID, lw=0.6)
    ax.set_title(f"{resp.sum():,} vertices with baseline held-out R² > 5%", fontsize=9, loc="left")
    ax = axes[2]
    if ceiling is not None:
        for name, label, col in VARIANTS:
            frac = (100 * fits[name]["xval_r2"] / ceiling)[resp & (ceiling > 5)]
            ax.hist(
                np.clip(frac, 0, 1.3),
                bins=np.linspace(0, 1.3, 53),
                histtype="step",
                lw=1.6,
                color=col,
                label=label,
            )
        ax.set_xlabel("held-out R² / noise ceiling")
        ax.set_ylabel("vertices")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("fraction of the explainable variance", fontsize=9, loc="left")
    fig.tight_layout()
    save(fig, out, "fig10_xval.png")

    # ---------------------------------------------------------------- reliability
    params = ["x", "y", "eccentricity", "angle", "rfsize"]
    fig, axes = plt.subplots(1, len(params), figsize=(13, 2.8), sharey=True)
    numbers["reliability"] = {}
    for name, label, col in VARIANTS:
        path = work / f"{name}_reliability.tsv"
        if not path.exists():
            continue
        rel = read_reliability(path)
        at = int(np.argmin(np.abs(rel["threshold"] - 0.2)))
        numbers["reliability"][label] = {
            p: float(rel[f"{p}_sb"][at]) for p in params if f"{p}_sb" in rel
        }
        numbers["reliability"][label]["threshold"] = float(rel["threshold"][at])
        numbers["reliability"][label]["n"] = int(rel["n_voxels"][at])
        for ax, p in zip(axes, params, strict=False):
            if f"{p}_sb" in rel:
                ax.plot(rel["threshold"], rel[f"{p}_sb"], color=col, lw=1.6, label=label)
    for ax, p in zip(axes, params, strict=False):
        ax.set_title(p, fontsize=9, loc="left")
        ax.set_xlabel("full-fit R² threshold")
        ax.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel("split-half reliability\n(Spearman-Brown)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig11_reliability.png")

    # ---------------------------------------------------------------- maps
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.4))
    norm = Normalize(0, 70)
    for c, (name, label, _) in enumerate(VARIANTS):
        s = split(100 * fits[name]["xval_r2"])
        for h, hemi in enumerate(HEMIS):
            surface_panel(
                axes[h, c],
                views[hemi],
                s[hemi],
                sulcs[hemi],
                rois[hemi],
                "magma",
                norm,
                s[hemi] > 5,
                f"{label} · {hemi}",
            )
    d = split(100 * fits["ffs_lib_dn"]["xval_r2"] - x0)
    for h, hemi in enumerate(HEMIS):
        surface_panel(
            axes[h, 4],
            views[hemi],
            d[hemi],
            sulcs[hemi],
            rois[hemi],
            "RdBu_r",
            TwoSlopeNorm(0, -20, 20),
            split(resp)[hemi],
            f"Δ both − canonical · {hemi}",
        )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="magma"),
        ax=axes[:, :4],
        orientation="horizontal",
        fraction=0.04,
        pad=0.03,
        aspect=50,
        label="held-out R² %",
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=TwoSlopeNorm(0, -20, 20), cmap="RdBu_r"),
        ax=axes[:, 4],
        orientation="horizontal",
        fraction=0.04,
        pad=0.03,
        label="Δ held-out R² (points)",
    )
    save(fig, out, "fig12_xval_maps.png")

    # ---------------------------------------------------------------- parameter changes
    both = fits["ffs_lib_dn"]
    good = fit_mask & (100 * base["r2"] > 20) & (100 * both["r2"] > 20)
    e = good & (base["eccentricity"] > 2)
    numbers["param_change"] = {
        "n": int(good.sum()),
        "angle_median_abs_deg": float(
            np.median(np.abs(ang_diff(both["angle"][e], base["angle"][e])))
        ),
        "ecc_r": pearson(base["eccentricity"][good], both["eccentricity"][good]),
        "rfsize_median_ratio": float(np.median(both["rfsize"][good] / base["rfsize"][good])),
        "rfsize_r_log": pearson(np.log(base["rfsize"][good]), np.log(both["rfsize"][good])),
    }
    # A pRF centred outside the aperture (radius 50 px) is driven only by the
    # aperture edge, and its position flips with small model changes; it dominates
    # any Pearson r. Report it, and the shifts robustly, for each variant.
    numbers["outside_aperture_frac"] = {
        label: float(np.mean(fits[name]["eccentricity"][good] > 50)) for name, label, _ in VARIANTS
    }
    areas = (
        ("V1", roi == 1),
        ("V2", roi == 2),
        ("V3", roi == 3),
        ("hV4", roi == 4),
        ("beyond V1-hV4", roi == 0),
    )
    numbers["outside_aperture_by_area"] = {
        label: {
            area: float(
                np.mean(
                    fits[name]["eccentricity"][
                        m & np.isfinite(fits[name]["r2"]) & (fits[name]["r2"] > 0.2)
                    ]
                    > 50
                )
            )
            for area, m in areas
        }
        for name, label, _ in VARIANTS
    }
    numbers["param_change_by_variant"] = {}
    for name, label, _ in VARIANTS[1:]:
        b = fits[name]
        ok = good & np.isfinite(b["r2"]) & (b["r2"] > 0.2)
        inside = ok & (base["eccentricity"] < 50) & (b["eccentricity"] < 50)
        shift = np.hypot(b["x"] - base["x"], b["y"] - base["y"])
        numbers["param_change_by_variant"][label] = {
            "n": int(ok.sum()),
            "inside_n": int(inside.sum()),
            "position_shift_median_px": float(np.median(shift[inside])),
            "position_shift_p90_px": float(np.quantile(shift[inside], 0.9)),
            "ecc_signed_median_px": float(
                np.median((b["eccentricity"] - base["eccentricity"])[inside])
            ),
            "ecc_r_inside": pearson(base["eccentricity"][inside], b["eccentricity"][inside]),
            "angle_median_abs_deg": float(
                np.median(
                    np.abs(
                        ang_diff(
                            b["angle"][inside & (base["eccentricity"] > 2)],
                            base["angle"][inside & (base["eccentricity"] > 2)],
                        )
                    )
                )
            ),
            "rfsize_median_ratio": float(np.median((b["rfsize"] / base["rfsize"])[inside])),
            "flips_across_aperture_edge": float(
                np.mean((base["eccentricity"][ok] > 50) != (b["eccentricity"][ok] > 50))
            ),
        }
    lib_hrf = fits["ffs_lib"]["hrf_index"]
    numbers["hrf_index"] = {
        "median": float(np.nanmedian(lib_hrf[good])),
        "iqr": [
            float(np.nanquantile(lib_hrf[good], 0.25)),
            float(np.nanquantile(lib_hrf[good], 0.75)),
        ],
    }
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, key, lim, log in (
        (axes[0], "eccentricity", (0, 50), False),
        (axes[1], "rfsize", (0.3, 100), True),
    ):
        a, b = base[key][good], both[key][good]
        ok = (a > 0) & (b > 0) if log else np.isfinite(a) & np.isfinite(b)
        ax.hexbin(
            a[ok],
            b[ok],
            gridsize=60,
            bins="log",
            cmap="Purples",
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
        ax.set_xlabel(f"canonical {key} (px)")
        ax.set_ylabel(f"library + denoise {key} (px)")
    ax = axes[2]
    ax.hist(
        ang_diff(both["angle"][e], base["angle"][e]), bins=np.linspace(-30, 30, 61), color=FOURTH
    )
    ax.set_xlabel("Δ polar angle (deg)")
    ax.set_title(
        f"median |Δ| {numbers['param_change']['angle_median_abs_deg']:.1f}°", fontsize=9, loc="left"
    )
    ax = axes[3]
    ax.hist(lib_hrf[good], bins=np.arange(0.5, np.nanmax(lib_hrf[good]) + 1.5), color=APR)
    ax.set_xlabel("selected HRF (library index, early → late)")
    ax.set_ylabel("vertices")
    fig.tight_layout()
    save(fig, out, "fig13_params.png")

    # ---------------------------------------------------------------- timecourses
    from fastfuncstuff.design.hrf import load_canonical_hrf_basic, load_canonical_hrf_library

    canon = load_canonical_hrf_basic(
        microtime_dt=1.0, hrf_duration=32.0, stim_duration=1.0, device=torch.device("cpu")
    ).numpy()
    library = load_canonical_hrf_library(
        microtime_dt=1.0, hrf_duration=32.0, stim_duration=1.0, device=torch.device("cpu")
    ).numpy()
    dgain = 100 * both["xval_r2"] - x0
    cand = np.flatnonzero(resp & (roi >= 1) & (roi <= 3) & np.isfinite(dgain))
    pick = [cand[np.argsort(dgain[cand])[int(q * (cand.size - 1))]] for q in (0.9, 0.5)]
    runs = load_runs(work, np.array(pick))
    raw = np.concatenate(runs, 1)
    raw = np.concatenate([100 * r / r.mean(1, keepdims=True) for r in np.split(raw, R, axis=1)], 1)
    fig, axes = plt.subplots(len(pick), 1, figsize=(14, 2.6 * len(pick)))
    t = np.arange(1, T + 1)
    for row, v in enumerate(pick):
        ax = axes[row]
        for name, label, col in VARIANTS:
            f = fits[name]
            pcs = read_pcs(work / f"{name}_noisepcs.1D", R, T)
            d = project_runs(raw[row : row + 1], T, 2, pcs)
            p = np.array([[50.5 - f["y"][v], f["x"][v] + 50.5, f["sigma"][v], f["exponent"][v]]])
            h = canon if "canon" in name else library[int(f["hrf_index"][v]) - 1]
            pred = project_runs(css_predict(p, aperture, h, R), T, 2, pcs)
            r2, g = r2_percent(d, pred)
            if name == "ffs_canon":
                ax.plot(
                    t, d[0].reshape(R, T).mean(0), color=MUTED, lw=0.9, label="data, drift removed"
                )
            if name == "ffs_lib_dn":
                ax.plot(
                    t,
                    d[0].reshape(R, T).mean(0),
                    color=INK,
                    lw=0.9,
                    alpha=0.6,
                    label="data, drift + noise PCs removed",
                )
            ax.plot(
                t,
                (g[0] * pred[0]).reshape(R, T).mean(0),
                color=col,
                lw=1.6,
                label=f"{label}: R² {r2[0]:.0f}%, held-out {100 * f['xval_r2'][v]:.0f}%",
            )
        ax.set_xlim(1, T)
        ax.set_ylabel("% signal")
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.set_title(
            f"vertex {v} ({'lh' if v < n_lh else 'rh'}, V{int(roi[v])})", fontsize=9, loc="left"
        )
    axes[-1].set_xlabel("TR (mean of runs)")
    fig.tight_layout()
    save(fig, out, "fig14_variant_timecourses.png")

    (out / "numbers_variants.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
