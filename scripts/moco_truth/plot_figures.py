"""Figures for docs/moco.md from a truth testbed and estimator outputs.

    python plot_figures.py manifest.json OUTDIR

The manifest names the testbeds (gen_truth.py prefixes), the estimate files
for each estimator, and the real runs for the bias-signature figure; see
plot_manifest_example.json. Static PNGs: every number shown is also in the
tables of docs/moco.md, which serve as the table view.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402

FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])

# Categorical slots 1-4 of the validated reference palette, fixed order; each
# also carries a line style and marker so identity is never colour alone.
EST = {
    "fourier": dict(label="ffs fourier (default)", color="#2a78d6", ls="-", marker="o"),
    "heptic": dict(label="ffs heptic", color="#eb6834", ls="--", marker="s"),
    "vrheptic": dict(label="3dvolreg -heptic", color="#1baf7a", ls="-.", marker="^"),
    "vrFourier": dict(label="3dvolreg -Fourier", color="#eda100", ls=":", marker="D"),
}
INK, INK2, GRID, SURF = "#2b2b2a", "#6b6a64", "#e6e5e0", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
    }
)


def mm_axis(ax, which="y", ticks=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)):
    """Log axis labelled in plain millimetres (powers of ten collide and read badly)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    a = ax.yaxis if which == "y" else ax.xaxis
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def load_aff12(path):
    m = np.loadtxt(path, comments="#").reshape(-1, 12)
    out = np.tile(np.eye(4), (len(m), 1, 1))
    out[:, :3, :] = m.reshape(-1, 3, 4)
    return out


class Testbed:
    def __init__(self, prefix):
        self.meta = json.load(open(f"{prefix}_meta.json"))
        self.tags = self.meta["tags"]
        mimg = nb.load(f"{prefix}_mask.nii.gz")
        self.mask = np.asarray(mimg.dataobj) > 0
        self.affine = mimg.affine
        ijk = np.c_[np.argwhere(self.mask), np.ones(self.mask.sum())].T
        self.xyz = FLIP @ mimg.affine @ ijk
        self.truth = load_aff12(f"{prefix}_truth.aff12.1D")
        self.groups = defaultdict(list)
        for i, t in enumerate(self.tags):
            self.groups[re.sub(r"_[-\d.]+$", "", t)].append(i)

    def disp(self, mats):
        """Mean displacement (mm) over the brain of each matrix from identity."""
        return np.linalg.norm(((mats - np.eye(4)) @ self.xyz)[:, :3], axis=1).mean(1)

    def err(self, est):
        """Per-volume mean displacement-field error (mm) against the truth."""
        return np.linalg.norm(((est - self.truth) @ self.xyz)[:, :3], axis=1).mean(1)


def fig_real_traces(m, out):
    tb = Testbed(m["real_trace"]["truth"])
    traces = m["real_trace"]["groups"]  # {group tag: title}
    fig, axes = plt.subplots(
        2,
        len(traces),
        figsize=(6.2 * len(traces), 6.4),
        sharex="col",
        gridspec_kw={"height_ratios": [1, 1.4]},
    )
    for col, (g, title) in enumerate(traces.items()):
        idx = np.array(tb.groups[g])
        t = np.arange(1, len(idx) + 1)
        ax = axes[0, col]
        ax.plot(t, tb.disp(tb.truth[idx]), color=INK, lw=1.6)
        ax.set_title(f"{title}: true motion")
        ax.set_ylabel("mean brain displacement (mm)")
        ax = axes[1, col]
        for key, st in EST.items():
            e = tb.err(load_aff12(m["real_trace"]["est"][key]))[idx]
            ax.plot(t, e, color=st["color"], ls=st["ls"], lw=1.4, label=st["label"])
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 0.1)
        mm_axis(ax)
        ax.set_title("error of each estimate")
        ax.set_ylabel("displacement error (mm)")
        ax.set_xlabel("volume")
    axes[1, 0].legend(loc="upper left", ncol=2, fontsize=9)
    fig.suptitle(
        "Recovering real motion (0.35 mm anatomy, 2.1 mm 2D, 2% noise)",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_subvoxel_bias(m, out):
    tb = Testbed(m["battery"]["truth"])
    est = {k: load_aff12(p) for k, p in m["battery"]["est_noisefree"].items()}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, (axis, name) in zip(axes, [(0, "in-plane (x)"), (2, "slice axis (z)")], strict=True):
        idx = tb.groups[f"sweep_t{'xyz'[axis]}"]
        sgn = -1 if axis < 2 else 1
        vox = tb.meta["voxel_mm"][axis]
        true = ((tb.truth[idx] - np.eye(4)) @ tb.xyz)[:, axis].mean(1) * sgn / vox
        ax.axhline(0, color=INK2, lw=1)
        for key, st in EST.items():
            e = ((est[key][idx] - np.eye(4)) @ tb.xyz)[:, axis].mean(1) * sgn / vox
            ax.plot(
                true,
                100 * (e - true),
                color=st["color"],
                ls=st["ls"],
                marker=st["marker"],
                ms=3.5,
                label=st["label"],
            )
        ax.set_title(f"translation along the {name}")
        ax.set_xlabel("true shift (voxels)")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("estimate − truth (% of a voxel)")
    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle(
        "Sub-voxel bias: polynomial kernels pull toward the half voxel (noise-free)",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_conditions(m, out):
    rows = m["conditions"]  # list of [label, truth prefix, group, {est: path}]
    fig, ax = plt.subplots(figsize=(9.5, 0.48 * len(rows) + 1.4))
    offs = np.linspace(0.27, -0.27, len(EST))  # first series on top, as in the legend
    cache = {}
    for r, (_label, prefix, group, ests) in enumerate(rows):
        tb = cache.setdefault(prefix, Testbed(prefix))
        idx = tb.groups[group]
        y = len(rows) - 1 - r
        for o, (key, st) in zip(offs, EST.items(), strict=True):
            e = tb.err(load_aff12(ests[key]))[idx]
            ax.plot([e.mean(), e.max()], [y + o, y + o], color=st["color"], lw=1, alpha=0.5)
            ax.plot(
                e.mean(),
                y + o,
                marker=st["marker"],
                ms=7,
                color=st["color"],
                ls="none",
                markeredgecolor=SURF,
                markeredgewidth=1.5,
                label=st["label"] if r == 0 else None,
            )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row[0] for row in rows][::-1])
    ax.set_xscale("log")
    mm_axis(ax, "x")
    ax.set_xlabel("displacement error (mm): marker = mean, line runs to the max")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=9)
    fig.suptitle("Error across conditions (2% noise)", x=0.01, ha="left", fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_noise(m, out):
    tb = Testbed(m["noise"]["truth"])
    idx = tb.groups[m["noise"]["group"]]
    levels = m["noise"]["levels"]  # {"0": {est: path}, "2": ..., "5": ...}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [float(k) for k in levels]
    for key, st in EST.items():
        ys = [tb.err(load_aff12(levels[k][key]))[idx].mean() for k in levels]
        ax.plot(
            xs, ys, color=st["color"], ls=st["ls"], marker=st["marker"], ms=7, label=st["label"]
        )
        ax.annotate(
            st["label"],
            (xs[-1], ys[-1]),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_yscale("log")
    mm_axis(ax)
    ax.set_xlabel("noise (% of the brain mean)")
    ax.set_ylabel("mean displacement error (mm)")
    ax.set_xticks(xs)
    ax.set_xlim(-0.3, xs[-1] + 4.0)
    ax.set_title(m["noise"]["title"])
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_gain(m, out):
    tb = Testbed(m["battery"]["truth"])
    est = {k: load_aff12(p) for k, p in m["battery"]["est_noisefree"].items()}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, axis in zip(axes, "xyz", strict=True):
        idx = tb.groups[f"sweep_r{axis}"]
        dt = tb.disp(tb.truth[idx])
        deg = np.array([float(tb.tags[i].split("_")[-1]) for i in idx])
        ax.axhline(1, color=INK2, lw=1)
        for key, st in EST.items():
            ax.plot(
                deg,
                tb.disp(est[key][idx]) / dt,
                color=st["color"],
                ls=st["ls"],
                marker=st["marker"],
                ms=5,
                label=st["label"],
            )
        ax.set_xscale("log")
        ax.set_title(f"rotation about {axis}")
        ax.set_xlabel("true rotation (deg)")
    axes[0].set_ylabel("estimated / true size")
    axes[0].legend(fontsize=9)
    fig.suptitle(
        "Small rotations: estimated vs true size (noise-free)", x=0.01, ha="left", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_real_signature(m, out):
    pts = {0: [], 1: []}
    for run in m["signature"]["runs"]:
        img = nb.load(run["bold"])
        v0 = np.asarray(img.dataobj[..., 0], dtype=np.float32)
        mask = v0 > 0.3 * np.percentile(v0, 98)
        xyz = FLIP @ img.affine @ np.c_[np.argwhere(mask), np.ones(mask.sum())].T
        to_vox = np.linalg.inv(FLIP @ img.affine)[:3, :3]
        d = {}
        for k in ("heptic", "fourier"):
            mats = load_aff12(run[k])
            d[k] = (((mats - np.eye(4)) @ xyz)[:, :3].mean(-1)) @ to_vox.T
        for a in (0, 1):
            pts[a].append(np.c_[d["fourier"][:, a], d["heptic"][:, a] - d["fourier"][:, a]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, a in zip(axes, (0, 1), strict=True):
        p = np.concatenate(pts[a])
        f, dd = p[:, 0], 100 * p[:, 1]
        ax.scatter(
            f, dd, s=6, color=EST["heptic"]["color"], alpha=0.25, linewidths=0, label="one volume"
        )
        X = np.c_[np.sin(2 * np.pi * f), np.ones_like(f)]
        coef, *_ = np.linalg.lstsq(X, dd, rcond=None)
        r = np.corrcoef(np.sin(2 * np.pi * f), dd)[0, 1]
        gx = np.linspace(f.min(), f.max(), 200)
        ax.plot(
            gx,
            coef[0] * np.sin(2 * np.pi * gx) + coef[1],
            color=INK,
            lw=1.8,
            label=f"{coef[0]:.2f}% · sin(2π·position), r = {r:.2f}",
        )
        ax.axhline(0, color=INK2, lw=1)
        ax.set_title(f"voxel axis {'ij'[a]}")
        ax.set_xlabel("position relative to the base (voxels)")
        ax.legend(fontsize=9, loc="upper left")
    axes[0].set_ylabel("heptic − fourier (% of a voxel)")
    fig.suptitle(m["signature"]["title"], x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_resample_maps(m, out):
    import torch

    from fastfuncstuff.processing.shear import shear_resample
    from fastfuncstuff.processing.shear_triton import shear_resample_triton

    tb = Testbed(m["resample"]["truth"])
    idx = tb.groups[m["resample"]["group"]]
    img = nb.load(m["resample"]["series"])
    data = np.asarray(img.dataobj, dtype=np.float32)
    A = FLIP @ img.affine
    vox = np.linalg.inv(A) @ tb.truth[idx] @ A
    src = torch.tensor(np.ascontiguousarray(data[..., idx].transpose(3, 2, 1, 0)), device="cuda")
    shape = tuple(src.shape[1:])
    base = data[..., 0]
    bm = float(base[tb.mask].mean())
    maps = {}
    for mode in m["resample"]["modes"]:
        w, valid = shear_resample_triton(
            src, torch.tensor(vox, device="cuda", dtype=torch.float32), shape, mode
        )
        for j in (~valid).nonzero(as_tuple=True)[0].tolist():
            w[j] = shear_resample(
                src[j], torch.tensor(vox[j], device="cuda", dtype=torch.float32), shape, mode
            )
        res = w.cpu().numpy().transpose(3, 2, 1, 0) - base[..., None]
        maps[mode] = 100 * res.std(-1) / bm
    # orient through the affine: closest canonical (RAS) view
    ornt = nb.orientations.io_orientation(img.affine)
    to_ras = nb.orientations.ornt_transform(ornt, nb.orientations.axcodes2ornt("RAS"))
    z = None
    vmax = np.percentile(np.concatenate([v[tb.mask] for v in maps.values()]), 99)
    labels = {"heptic": "heptic", "wsinc5": "wsinc5 (old -final default)", "fourier": "fourier"}
    fig, axes = plt.subplots(1, len(maps), figsize=(4.0 * len(maps), 4.4))
    for ax, (mode, mp) in zip(axes, maps.items(), strict=True):
        ras = nb.orientations.apply_orientation(mp, to_ras)
        msk = nb.orientations.apply_orientation(tb.mask, to_ras)
        if z is None:  # 60% up the brain: above the orbits and temporal poles
            zs = np.where(msk.any((0, 1)))[0]
            z = int(zs[0] + 0.6 * (zs[-1] - zs[0]))
        sl = np.where(msk[:, :, z], ras[:, :, z], np.nan)
        im = ax.imshow(sl.T, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{labels.get(mode, mode)}\nmedian {np.median(mp[tb.mask]):.2f}%")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.text(0.02, 0.02, "R", transform=ax.transAxes, color=INK2, fontsize=9)
        ax.text(0.95, 0.02, "L", transform=ax.transAxes, color=INK2, fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("injected temporal std (% of brain mean)")
    fig.suptitle(m["resample"]["title"], x=0.01, ha="left", fontsize=12)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    m = json.load(open(sys.argv[1]))
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("real_trace_recovery", fig_real_traces),
        ("subvoxel_bias", fig_subvoxel_bias),
        ("conditions", fig_conditions),
        ("noise", fig_noise),
        ("small_rotation_gain", fig_gain),
        ("real_data_signature", fig_real_signature),
        ("resample_maps", fig_resample_maps),
    ]:
        if name.split("_")[0] in m.get("skip", []):
            continue
        fn(m, outdir / f"{name}.png")
        print("wrote", outdir / f"{name}.png")


if __name__ == "__main__":
    main()
