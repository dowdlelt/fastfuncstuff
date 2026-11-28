"""Benchmark report: accuracy and timing figures, rebuilt from the worklist on every call.

``work -report DIR`` writes ``DIR/index.html`` with one collapsible section per check,
and, per figure, a vector SVG (for the page), a PDF (for a paper) and the CSV it was drawn
from. Everything is derived from the main worklist plus every ingested host set: new
units, datasets, hosts and architectures appear without a change here. A check opts in
with a ``report`` spec (``report_spec.ReportSpec``) that names its headline metrics.

Accuracy reads one row per unit: when a unit ran on several hosts, the latest finished run
counts (latest code), and the code versions behind each section are printed with it, so a
section mixing old and new code says so. Timing reads every run, since each host is its own
architecture.
"""

from __future__ import annotations

import ast
import html
import re
import sqlite3
import subprocess
import time
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from .report_spec import Metric, ReportSpec

PROV_KEYS = {
    "host": ("cpu_model", "gpu", "n_physical_cores"),
    "worker": ("threads", "hostname"),
    "ffs": ("ffs_sha",),
    "scorer": ("check_hash",),  # recorded from 2026-11-28; older units show "?"
}


# ---------------------------------------------------------------- loading


def short_cpu(model: str | None) -> str:
    if not model:
        return "CPU"
    s = re.sub(r"\((R|TM)\)", "", model)
    s = re.sub(r"\s+CPU\s*@.*$|\s+\d+-Core Processor.*$|^(Intel|AMD)\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def short_gpu(name: str | None) -> str:
    if not name:
        return "GPU"
    return re.sub(r"^(NVIDIA\s+)?(GeForce\s+)?", "", name).strip()


def arch_label(device: str, cpu_model: str | None, gpu: str | None) -> str:
    if device == "cuda":
        return f"{short_gpu(gpu)} (CUDA)"
    if device == "mps":
        return f"{short_gpu(gpu) if gpu else 'Apple GPU'} (MPS)"
    return f"{short_cpu(cpu_model)} (CPU)"


def _read_db(path: Path, check: str, host: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one wide row per finished unit, timings long) from one worklist file."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        units = pd.read_sql(
            "SELECT id AS unit_id, dataset, unit_key, finished_at FROM units"
            " WHERE check_name=? AND status='done'",
            db,
            params=(check,),
        )
        if units.empty:
            return pd.DataFrame(), pd.DataFrame()
        q = "SELECT unit_id, {cols} FROM {t} WHERE unit_id IN (SELECT id FROM units WHERE check_name=? AND status='done')"
        met = pd.read_sql(
            q.format(cols="judge || '.' || metric AS key, value", t="metrics"), db, params=(check,)
        )
        tim = pd.read_sql(
            q.format(cols="side, device, seconds, threads", t="timings"), db, params=(check,)
        )
        prov = pd.read_sql(q.format(cols="side, key, value", t="provenance"), db, params=(check,))
    finally:
        db.close()
    keep = [(s, k) for s, ks in PROV_KEYS.items() for k in ks]
    prov = prov[[(s, k) in keep for s, k in zip(prov.side, prov.key, strict=False)]]
    prov = prov.assign(col=prov.side + "." + prov.key).pivot_table(
        index="unit_id", columns="col", values="value", aggfunc="first"
    )
    wide = met.pivot_table(index="unit_id", columns="key", values="value", aggfunc="first")
    units = units.set_index("unit_id").join(prov).join(wide).reset_index()
    units.insert(0, "host", host)
    for c in ("host.cpu_model", "host.gpu", "ffs.ffs_sha", "worker.threads", "scorer.check_hash"):
        if c not in units:
            units[c] = None
    tim = tim.merge(
        units[["unit_id", "dataset", "host.cpu_model", "host.gpu", "worker.threads"]],
        on="unit_id",
    )
    tim.insert(0, "host", host)
    tim["arch"] = [
        arch_label(d, c, g)
        for d, c, g in zip(tim.device, tim["host.cpu_model"], tim["host.gpu"], strict=False)
    ]
    return units, tim


def worklist_files(root: Path) -> list[tuple[str, Path]]:
    """('local', main worklist) then every ingested host set, by its label."""
    root = Path(root)
    out = [("local", root / "worklist.sqlite")]
    db = sqlite3.connect(f"file:{root / 'worklist.sqlite'}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT label, path FROM hostsets ORDER BY ingested_at").fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        db.close()
    out += [(label, Path(p) / "worklist.sqlite") for label, p in rows]
    return [(h, p) for h, p in out if p.exists()]


REPO = Path(__file__).resolve().parents[2]


def _module_file(mod: str) -> Path | None:
    p = REPO / Path(*mod.split("."))
    for f in (p.with_suffix(".py"), p / "__init__.py"):
        if f.is_file():
            return f
    return None


@cache
def code_files(entries: tuple[str, ...]) -> tuple[str, ...]:
    """Repo files the entry modules import. Every import of an entry module counts (CLIs
    import lazily inside main() to start fast); below them only module-level imports, since
    following every lazy import reaches half the package and flags every unrelated commit."""
    seen: set[Path] = set()
    stack = [(m, True) for m in entries]
    while stack:
        mod, entry = stack.pop()
        f = _module_file(mod)
        if f is None or f in seen:
            continue
        seen.add(f)
        pkg = mod if f.name == "__init__.py" else mod.rsplit(".", 1)[0]
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree) if entry else tree.body:
            if isinstance(n, ast.ImportFrom):
                base = n.module or ""
                if n.level:
                    parts = pkg.split(".")[: len(pkg.split(".")) - (n.level - 1)]
                    base = ".".join([*parts, base] if base else parts)
                if base.startswith("fastfuncstuff"):
                    stack += [(base, False)] + [(f"{base}.{a.name}", False) for a in n.names]
            elif isinstance(n, ast.Import):
                stack += [(a.name, False) for a in n.names if a.name.startswith("fastfuncstuff")]
    return tuple(sorted(str(p.relative_to(REPO)) for p in seen))


@cache
def commit_time(sha: str) -> float:
    """Commit date, so the newest code wins, not the last run to finish (hosts lag)."""
    r = subprocess.run(
        ["git", "-C", str(REPO), "log", "-1", "--format=%ct", sha], capture_output=True, text=True
    )
    return float(r.stdout.strip() or 0)


@cache
def code_changed(old: str | None, new: str | None, files: tuple[str, ...]) -> bool:
    """Whether any of ``files`` differs between two commits; unknown commits count as changed."""
    if old == new:
        return False
    if not old or not new or not files or not (REPO / ".git").exists():
        return True
    r = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", old, new, "--", *files],
        capture_output=True,
    )
    return r.returncode != 0


def load_check(
    root: Path, check: str, code: tuple[str, ...] = ()
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(all runs wide, one row per unit (latest run), timings long over all runs)."""
    runs, tims = [], []
    for host, path in worklist_files(root):
        u, t = _read_db(path, check, host)
        if not u.empty:
            runs.append(u)
            tims.append(t)
    if not runs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    allruns = pd.concat(runs, ignore_index=True)
    latest = allruns.sort_values("finished_at").drop_duplicates("unit_id", keep="last")
    # A unit whose latest run predates a change to the tool's code is drawn hollow.
    shas = [s for s in latest["ffs.ffs_sha"].dropna().unique() if isinstance(s, str)]
    # commit date first; a commit git does not know falls back to when its runs finished
    last = latest.groupby("ffs.ffs_sha").finished_at.max()
    head = max(shas, key=lambda s: (commit_time(s), last[s])) if shas else None
    files = code_files(tuple(code)) if code else ()
    latest["stale"] = [
        code_changed(s if isinstance(s, str) else None, head, files) if files else s != head
        for s in latest["ffs.ffs_sha"]
    ]
    return allruns, latest.sort_values("unit_id").reset_index(drop=True), pd.concat(tims)


def metric_long(df: pd.DataFrame, m: Metric, spec: ReportSpec) -> pd.DataFrame:
    """unit_id, dataset, tool, value for one headline metric (tool '' for a pair metric)."""
    if "{tool}" not in m.key:
        if m.key not in df:
            return pd.DataFrame(columns=["unit_id", "dataset", "tool", "value", "suffix"])
        d = df[["unit_id", "dataset", "stale"]].assign(tool="", suffix="", value=df[m.key])
        return d.dropna(subset=["value"])
    parts = [
        df[["unit_id", "dataset", "stale"]].assign(
            tool=name, suffix=suf, value=df[m.key.format(tool=suf)]
        )
        for suf, name in spec.tools.items()
        if m.key.format(tool=suf) in df
    ]
    if not parts:
        return pd.DataFrame(columns=["unit_id", "dataset", "tool", "value", "suffix"])
    return pd.concat(parts, ignore_index=True).dropna(subset=["value"])


def failures(d: pd.DataFrame, m: Metric) -> pd.DataFrame:
    if m.kind == "rate":
        return d[d.value < 0.5]
    if m.line is None or m.better == "none":
        return d.iloc[:0]
    return d[d.value > m.line] if m.better == "lower" else d[d.value < m.line]


def speedups(tim: pd.DataFrame, spec: ReportSpec) -> pd.DataFrame:
    """Per unit and host: reference seconds / ffs seconds, for each spec pair and ffs device."""
    rows = []
    for ffs_sides, ref_sides in spec.speedup:
        fs, rs = ffs_sides.split("+"), ref_sides.split("+")
        for (host, uid), g in tim.groupby(["host", "unit_id"]):
            ref = g[g.side.isin(rs)]
            if set(ref.side) != set(rs):
                continue
            ref_s = ref.groupby("side").seconds.min().sum()
            for _dev, gd in g[g.side.isin(fs)].groupby("device"):
                if set(gd.side) != set(fs):
                    continue
                rows.append({
                    "host": host, "unit_id": uid, "dataset": g.dataset.iloc[0],
                    "pair": f"{' + '.join(spec.timing.get(s, s) for s in fs)} vs "
                            f"{' + '.join(spec.timing.get(s, s) for s in rs)}",
                    "ffs_arch": gd.arch.iloc[0], "ref_arch": ref.arch.iloc[0],
                    "ffs_s": gd.seconds.sum(), "ref_s": ref_s,
                    "speedup": ref_s / gd.seconds.sum(),
                })  # fmt: skip
    return pd.DataFrame(rows)


def agreement(allruns: pd.DataFrame, spec: ReportSpec) -> pd.DataFrame:
    """Units run on more than one host: how far their headline metrics differ."""
    multi = allruns[allruns.unit_id.duplicated(keep=False)]
    if multi.empty:
        return pd.DataFrame()
    rows = []
    for m in spec.metrics:
        keys = [m.key.format(tool=s) for s in spec.tools] if "{tool}" in m.key else [m.key]
        for k in keys:
            if k not in multi:
                continue
            g = multi.groupby("unit_id")
            spread = (g[k].max() - g[k].min()).dropna()
            shas = g["ffs.ffs_sha"].nunique()
            if spread.empty:
                continue
            rows.append({
                "metric": k, "units": len(spread),
                "median |diff|": spread.median(), "max |diff|": spread.max(),
                "worst unit": int(spread.idxmax()),
                "units on different code": int((shas.loc[spread.index] > 1).sum()),
            })  # fmt: skip
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import seaborn as sns

    sns.set_theme(context="paper", style="ticks", font_scale=1.0)
    matplotlib.rcParams.update({
        "svg.fonttype": "none", "pdf.fonttype": 42,  # text stays text, for editing in a paper
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 150,
    })  # fmt: skip


def _save(fig, out: Path, stem: str, data: pd.DataFrame) -> str:
    import matplotlib.pyplot as plt

    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    data.to_csv(out / f"{stem}.csv", index=False)
    return stem


def _palette(datasets) -> dict[str, tuple]:
    import seaborn as sns

    ds = sorted(set(datasets))
    base = sns.color_palette("tab10" if len(ds) <= 10 else "husl", len(ds))
    return dict(zip(ds, base, strict=False))


def _wrap(label: str, width: int = 34) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(label, width))


def _dots(ax, d: pd.DataFrame, x: str, order, pal, by_dataset: bool = False) -> None:
    """One dot per unit, jittered by a fixed seed (figures do not change between builds);
    units whose latest run predates a change to the code under test are hollow. On a
    per-dataset axis the colour would repeat the axis, so ``by_dataset`` drops it."""
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(0)
    pos = d[x].map({k: i for i, k in enumerate(order)}).to_numpy(float)
    xs = pos + rng.uniform(-0.22, 0.22, len(d))
    cols = ["#3b6ea8" if by_dataset else pal[k] for k in d.dataset]
    stale = d["stale"].fillna(False).to_numpy(bool) if "stale" in d else np.zeros(len(d), bool)
    ax.scatter(xs[~stale], d.value.to_numpy()[~stale], s=11, c=[c for c, st in zip(cols, stale, strict=False) if not st],
               lw=0, alpha=0.85, zorder=3)  # fmt: skip
    if stale.any():
        ax.scatter(xs[stale], d.value.to_numpy()[stale], s=11, facecolors="none",
                   edgecolors=[c for c, st in zip(cols, stale, strict=False) if st], lw=0.8, zorder=3)  # fmt: skip
    handles = [] if by_dataset else [
        Line2D([], [], ls="", marker="o", ms=3.5, color=pal[k], label=k)
        for k in sorted(d.dataset.unique())
    ]  # fmt: skip
    if stale.any():
        handles.append(Line2D([], [], ls="", marker="o", ms=3.5, mfc="none", color="0.3",
                              label="run before a code change"))  # fmt: skip
    if by_dataset and handles:  # one entry: inside the axes, clear of the threshold label
        ax.legend(handles=handles, loc="lower right", fontsize=6, handletextpad=0.2)
    elif len(handles) > 1:
        ax.legend(handles=handles, title="dataset", bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=6, title_fontsize=7, ncol=2 if len(handles) > 14 else 1,
                  handletextpad=0.2)  # fmt: skip


def _dataset_legend(ax, n):
    if not ax.get_legend_handles_labels()[0] or n < 2:
        return
    ax.legend(
        title="dataset", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6,
        title_fontsize=7, ncol=2 if n > 14 else 1, markerscale=0.8, handletextpad=0.2,
    )  # fmt: skip


def _threshold(ax, m: Metric, horizontal=True):
    if m.line is None:
        return
    (ax.axhline if horizontal else ax.axvline)(m.line, ls="--", lw=0.8, c="0.45", zorder=0)
    if m.line_label and horizontal:
        ax.annotate(
            m.line_label, (1, m.line), xycoords=("axes fraction", "data"), xytext=(2, 0),
            textcoords="offset points", fontsize=6, color="0.35", va="center",
        )  # fmt: skip


def fig_metric(d: pd.DataFrame, m: Metric, spec: ReportSpec, pal, out: Path, stem: str) -> str:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if m.log:
        d = d[d.value > 0]
    pair = (d.tool == "").all()
    x = "dataset" if pair else "tool"
    order = (
        sorted(d.dataset.unique()) if pair else [t for t in spec.tools.values() if t in set(d.tool)]
    )
    if m.kind == "rate":
        share = d.groupby(x).value.agg(["mean", "sum", "count"]).reindex(order).reset_index()
        fig, ax = plt.subplots(figsize=(max(2.6, 0.75 * len(order) + 1), 2.8))
        sns.barplot(data=share, x=x, y="mean", order=order, color="0.75", ax=ax)
        for i, r in share.iterrows():
            ax.text(
                i, r["mean"] + 0.02, f"{int(r['sum'])}/{int(r['count'])}", ha="center", fontsize=7
            )
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(_wrap(f"{m.label} (share of units)"))
        _shade_refs(ax, order, spec)
    else:
        w = min(7.2, 0.32 * len(order) + 1.5) if pair else max(3.0, 0.8 * len(order) + 1.2)
        fig, ax = plt.subplots(figsize=(max(3.0, w), 3.0))
        sns.boxplot(data=d, x=x, y="value", order=order, color="white", showfliers=False,
                    width=0.55, linecolor="0.3", linewidth=0.9, ax=ax)  # fmt: skip
        _dots(ax, d, x, order, pal, by_dataset=pair)
        if pair and len(order) > 12:
            ax.tick_params(axis="x", labelsize=6)
        if m.log:
            ax.set_yscale("log")
        _threshold(ax, m)
        bad = failures(d, m)
        if not bad.empty:  # name the worst units, so a failure leads straight to its folder
            worst = bad.sort_values("value", ascending=m.better == "higher")
            pos = {k: i for i, k in enumerate(order)}
            for _, r in worst.groupby(x).head(4).iterrows():
                ax.annotate(str(int(r.unit_id)), (pos[r[x]] + 0.28, r.value), fontsize=5.5,
                            color="0.25", va="center")  # fmt: skip
        ax.set_ylabel(_wrap(m.label))
        if not pair:
            _shade_refs(ax, order, spec)
    ax.set_xlabel("")
    if pair or max(len(str(o)) for o in order) > 10:
        ax.tick_params(axis="x", rotation=40)
        for t in ax.get_xticklabels():
            t.set_ha("right")
    return _save(fig, out, stem, d)


def _shade_refs(ax, order, spec: ReportSpec):
    refs = {spec.tools.get(s, spec.timing.get(s)) for s in spec.references}
    for i, t in enumerate(order):
        if t in refs:
            ax.axvspan(i - 0.5, i + 0.5, color="0.93", zorder=-1, lw=0)


def fig_difficulty(df, m: Metric, spec, pal, out: Path, stem: str) -> str | None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    key, xlabel = spec.difficulty  # type: ignore[misc]
    d = metric_long(df, m, spec)
    if d.empty or key not in df:
        return None
    d = d.merge(df[["unit_id", key]], on="unit_id").rename(columns={key: "difficulty"})
    if m.log:
        d = d[d.value > 0]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    pair = (d.tool == "").all()
    hue_order = (
        sorted(d.dataset.unique()) if pair else [t for t in spec.tools.values() if t in set(d.tool)]
    )
    sns.scatterplot(data=d, x="difficulty", y="value", hue="dataset" if pair else "tool",
                    hue_order=hue_order,
                    palette=pal if pair else "dark", style=None if pair else "tool",
                    s=16, alpha=0.85, linewidth=0, ax=ax)  # fmt: skip
    if m.log:
        ax.set_yscale("log")
    _threshold(ax, m)
    ax.set(xlabel=xlabel, ylabel=_wrap(m.label))
    if pair:
        _dataset_legend(ax, d.dataset.nunique())
    else:
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    return _save(fig, out, stem, d)


def fig_tradeoff(df, ykey: str, xkey: str, spec: ReportSpec, out: Path, stem: str) -> str | None:
    """Each tool's median y against its median x (interquartile bars), units faint behind."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    by_key = {m.key: m for m in spec.metrics}
    ym = by_key.get(ykey, Metric(ykey, ykey))
    xm = by_key.get(xkey, Metric(xkey, xkey))
    d = metric_long(df, ym, spec).merge(
        metric_long(df, xm, spec)[["unit_id", "tool", "value"]].rename(columns={"value": "x"}),
        on=["unit_id", "tool"],
    )
    if d.empty or d.tool.nunique() < 2:
        return None
    order = [t for t in spec.tools.values() if t in set(d.tool)]
    pal = dict(zip(order, sns.color_palette("dark", len(order)), strict=False))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for t in order:
        g = d[d.tool == t]
        ax.scatter(g.x, g.value, s=6, color=pal[t], alpha=0.18, lw=0)
        qx, qy = g.x.quantile([0.25, 0.5, 0.75]), g.value.quantile([0.25, 0.5, 0.75])
        ax.errorbar(qx[0.5], qy[0.5], xerr=[[qx[0.5] - qx[0.25]], [qx[0.75] - qx[0.5]]],
                    yerr=[[qy[0.5] - qy[0.25]], [qy[0.75] - qy[0.5]]], fmt="o", ms=6,
                    color=pal[t], capsize=2, lw=1, label=t)  # fmt: skip
    ax.set(xlabel=_wrap(xm.label, 44), ylabel=_wrap(ym.label))
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    return _save(fig, out, stem, d)


def fig_timing(tim: pd.DataFrame, spec: ReportSpec, out: Path, stem: str) -> str | None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    t = tim[tim.side.isin(spec.timing)].copy()
    if t.empty:
        return None
    t["tool"] = t.side.map(spec.timing)
    order = [v for k, v in spec.timing.items() if k in set(t.side)]
    archs = sorted(t.arch.unique(), key=lambda a: ("CPU" not in a, a))
    fig, ax = plt.subplots(figsize=(max(3.2, 0.9 * len(order) + 1.6), 3.0))
    sns.boxplot(data=t, x="tool", y="seconds", hue="arch", hue_order=archs, order=order,
                showfliers=False, width=0.7, linewidth=0.8, fill=False, legend=False,
                palette="dark", ax=ax)  # fmt: skip
    sns.stripplot(data=t, x="tool", y="seconds", hue="arch", hue_order=archs, order=order,
                  dodge=True, size=2.6, alpha=0.7, jitter=0.15, palette="dark", ax=ax)  # fmt: skip
    ax.set_yscale("log")
    ax.set(xlabel="", ylabel="wall time per unit (s)")
    _shade_refs(ax, order, spec)
    ax.legend(title="architecture", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    if max(len(o) for o in order) > 10:
        ax.tick_params(axis="x", rotation=30)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
    return _save(fig, out, stem, t.drop(columns=["host.cpu_model", "host.gpu"]))


def fig_speedup(sp: pd.DataFrame, out: Path, stem: str) -> str | None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if sp.empty:
        return None
    sp = sp.assign(row=sp.pair + "\n" + sp.ffs_arch + " vs " + sp.ref_arch)
    pairs = list(dict.fromkeys(sp.pair))  # the spec's order: its first pair is the headline
    sp = sp.assign(_k=sp.pair.map(pairs.index))
    rows = list(dict.fromkeys(sp.sort_values(["_k", "ffs_arch"], kind="stable").row))
    fig, ax = plt.subplots(figsize=(5.2, 0.55 * len(rows) + 0.9))
    sns.stripplot(data=sp, y="row", x="speedup", order=rows, color="0.6", size=2.5,
                  alpha=0.6, jitter=0.18, ax=ax)  # fmt: skip
    sns.pointplot(data=sp, y="row", x="speedup", order=rows, estimator=np.median,
                  errorbar=("ci", 95), linestyle="none", color="k", markersize=4,
                  err_kws={"linewidth": 1.2}, ax=ax)  # fmt: skip
    ax.set_xscale("log", base=2)
    ax.axvline(1, ls="--", lw=0.8, c="0.45", zorder=0)
    for i, r in enumerate(rows):
        med = sp.loc[sp.row == r, "speedup"].median()
        ax.annotate(f"{med:.1f}×", (1, i), xycoords=("axes fraction", "data"), xytext=(4, 0),
                    textcoords="offset points", fontsize=7, va="center")  # fmt: skip
    ax.set(xlabel="speed-up (reference time / ffs time, per unit)", ylabel="")
    ax.tick_params(axis="y", labelsize=6.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
    return _save(fig, out, stem, sp.drop(columns=["row", "_k"]))


def fig_overview(sp_all: pd.DataFrame, out: Path, stem: str) -> str | None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if sp_all.empty:
        return None
    first = sp_all.groupby("check").pair.transform("first") == sp_all.pair
    med = (
        sp_all[first]
        .groupby(["check", "ffs_arch"]).speedup.median().unstack()
    )  # fmt: skip
    fig, ax = plt.subplots(figsize=(1.3 * med.shape[1] + 2.2, 0.45 * med.shape[0] + 1.0))
    sns.heatmap(np.log2(med), annot=med.map(lambda v: f"{v:.1f}×" if v == v else ""),
                fmt="", cmap="RdBu", center=0, vmin=-4, vmax=4, cbar=False,
                linewidths=0.6, linecolor="white", ax=ax)  # fmt: skip
    ax.set(xlabel="ffs runs on", ylabel="")
    ax.tick_params(axis="x", rotation=0)
    return _save(fig, out, stem, med.reset_index())


# ---------------------------------------------------------------- captions and page


def _code_line(df: pd.DataFrame) -> tuple[str, bool]:
    shas = df["ffs.ffs_sha"].fillna("?").str[:8].value_counts()
    text = ", ".join(f"{s} ({n})" for s, n in shas.items())
    # Hosts score on site, so the judges can differ as well as the code they judge.
    sc = df["scorer.check_hash"].fillna("?").str[:8].value_counts()
    scored = ", ".join(f"{s} ({n})" for s, n in sc.items())
    return f"ffs code: {text}; scorer: {scored}", len(shas) > 1 or len(sc.drop("?", errors="ignore")) > 1


def _who(df: pd.DataFrame) -> str:
    n, k = len(df), df.dataset.nunique()
    return f"{n} unit{'s' * (n != 1)} from {k} dataset{'s' * (k != 1)}"


def _arch_caption(t: pd.DataFrame) -> str:
    parts = []
    for a, g in t.groupby("arch"):
        th = pd.to_numeric(g.threads, errors="coerce").dropna()
        thr = f", {int(th.mode().iloc[0])} threads" if len(th) and "CPU" in a else ""
        hosts = "/".join(sorted(g.host.unique()))
        parts.append(f"{a}{thr} [{hosts}]")
    return "; ".join(parts)


def caption_metric(m: Metric, d: pd.DataFrame, df: pd.DataFrame, spec: ReportSpec) -> str:
    pair = (d.tool == "").all()
    s = [f"<b>{html.escape(m.label)}.</b>"]
    if m.note:
        s.append(html.escape(m.note))
    s.append(f"{_who(df)}; one dot per unit, coloured by dataset." if m.kind != "rate"
             else f"{_who(df)}; bars are the share of units, counts above.")  # fmt: skip
    if "stale" in d and d["stale"].fillna(False).any():
        s.append(
            f"Hollow dots are the {d.loc[d['stale'].fillna(False), 'unit_id'].nunique()} units "
            "whose latest run predates a change to the code under test."
        )
    if pair:
        s.append("Drawn per dataset (the metric already compares ffs with the reference).")
    elif spec.references:
        s.append("Shaded columns are reference tools.")
    if m.better != "none":
        s.append(f"{m.better.capitalize()} is better.")
    if m.line is not None:
        s.append(
            f"Dashed line: {m.line:g}{' (' + html.escape(m.line_label) + ')' if m.line_label else ''}."
        )
    bad = failures(d, m)
    if len(bad) and m.kind != "rate":
        s.append(
            f"{len(bad)} point{'s' * (len(bad) != 1)} past the line, worst labelled by unit id."
        )
    return " ".join(s)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ffs benchmark</title><style>
:root{--fg:#1d1d1f;--mut:#666;--bg:#fff;--line:#e3e3e3;--warn:#a15c00;--card:#fafafa}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--fg:#e8e8e8;--mut:#a0a0a0;--bg:#16161a;--line:#33333a;--warn:#f0a040;--card:#1e1e24}
 :root:not([data-theme="light"]) img{background:#fff;border-radius:4px}}
body{font:15px/1.5 system-ui,sans-serif;color:var(--fg);background:var(--bg);max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:1.6em;margin:.2em 0}h2{display:inline;font-size:1.25em}h3{font-size:1.05em;margin:1.2em 0 .3em}
details{border-top:1px solid var(--line);padding:.6em 0}summary{cursor:pointer}
.mut{color:var(--mut)}.warn{color:var(--warn)}
figure{margin:.8em 0 1.4em;background:var(--card);padding:10px;border-radius:6px}
figure img{max-width:100%;height:auto}figcaption{font-size:.88em;color:var(--mut);margin-top:.4em}
figcaption a{color:inherit}table{border-collapse:collapse;font-size:.82em;margin:.5em 0;display:block;overflow-x:auto}
td,th{border-bottom:1px solid var(--line);padding:2px 8px;text-align:right}td:first-child,th:first-child{text-align:left}
</style></head><body>
<h1>ffs benchmark</h1>
<p class="mut">Built {{ built }} from {{ sources }}. Every figure links its PDF and the CSV it was drawn from.</p>
{% if overview %}<figure><img src="{{ overview }}.svg" alt="speed-up overview">
<figcaption><b>Speed-up overview.</b> Median reference time / ffs time per unit, for each check's
first comparison, by the architecture ffs ran on (the reference runs on that host's CPU).
<a href="{{ overview }}.pdf">pdf</a> · <a href="{{ overview }}.csv">csv</a></figcaption></figure>{% endif %}
{{ summary_table }}
{% for s in sections %}
<details {% if loop.first %}open{% endif %}><summary><h2>{{ s.name }}</h2>
<span class="mut"> — {{ s.who }}</span></summary>
<p>{{ s.summary }}</p>
<p class="{{ 'warn' if s.mixed else 'mut' }}">{{ s.code }}{% if s.mixed %} — mixed code versions: accuracy counts each unit's latest run{% endif %}</p>
<details open><summary><h3 style="display:inline">Accuracy</h3></summary>
{% for f in s.acc %}<figure><img src="{{ f.stem }}.svg" alt="{{ f.alt }}">
<figcaption>{{ f.caption }} <a href="{{ f.stem }}.pdf">pdf</a> · <a href="{{ f.stem }}.csv">csv</a></figcaption>
{% if f.fail %}<details><summary class="mut">units past the line ({{ f.nfail }})</summary>{{ f.fail }}</details>{% endif %}
</figure>{% endfor %}
</details>
<details open><summary><h3 style="display:inline">Timing</h3></summary>
{% for f in s.tim %}<figure><img src="{{ f.stem }}.svg" alt="{{ f.alt }}">
<figcaption>{{ f.caption }} <a href="{{ f.stem }}.pdf">pdf</a> · <a href="{{ f.stem }}.csv">csv</a></figcaption></figure>{% endfor %}
</details>
{% if s.agree %}<details><summary><h3 style="display:inline">Agreement across hosts</h3></summary>
<p class="mut">Units run on more than one host; differences on different code are code changes, not hosts.</p>{{ s.agree }}</details>{% endif %}
<details><summary><h3 style="display:inline">All metrics</h3></summary>{{ s.all }}</details>
</details>
{% endfor %}
</body></html>"""


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, float_format=lambda v: f"{v:.4g}", na_rep="")


def build_section(root: Path, check: str, spec: ReportSpec, out: Path) -> dict | None:
    allruns, df, tim = load_check(root, check, spec.code)
    if df.empty:
        return None
    pal = _palette(df.dataset)
    code, mixed = _code_line(df)
    acc = []
    for i, m in enumerate(spec.metrics):
        d = metric_long(df, m, spec)
        if d.empty:
            continue
        stem = f"{check}_acc{i:02d}_{re.sub(r'[^a-z0-9]+', '_', m.key.replace('{tool}', '').lower()).strip('_')}"
        fig_metric(d, m, spec, pal, out, stem)
        bad = failures(d, m)
        acc.append({
            "stem": stem, "alt": m.label, "caption": caption_metric(m, d, df, spec),
            "fail": _table(bad.sort_values(["tool", "value"])[["tool", "unit_id", "dataset", "value"]])
            if len(bad) else "", "nfail": len(bad),
        })  # fmt: skip
    if spec.difficulty:
        first = next((m for m in spec.metrics if m.kind != "rate"), None)
        if first is not None:
            stem = f"{check}_difficulty"
            if fig_difficulty(df, first, spec, pal, out, stem):
                acc.append({
                    "stem": stem, "alt": "difficulty", "fail": "", "nfail": 0,
                    "caption": f"<b>{html.escape(first.label)} against {html.escape(spec.difficulty[1])}.</b> "
                               f"{_who(df)}. Read failures against how hard the data was.",
                })  # fmt: skip
    for i, (ykey, xkey) in enumerate(spec.tradeoff):
        stem = f"{check}_tradeoff{i:02d}"
        if fig_tradeoff(df, ykey, xkey, spec, out, stem):
            better = next((m.better for m in spec.metrics if m.key == ykey), "none")
            acc.append({
                "stem": stem, "alt": "trade-off", "fail": "", "nfail": 0,
                "caption": f"<b>Trade-off.</b> {_who(df)}. Each tool's median (interquartile "
                           f"bars; units faint) of the score ({better} is better) against how hard "
                           "its warp deforms. A tool that wins only by moving right bought the "
                           "score with deformation; a fair win is up/down at the same x.",
            })  # fmt: skip
    tims = []
    if fig_timing(tim, spec, out, f"{check}_timing"):
        tims.append({
            "stem": f"{check}_timing", "alt": "timing",
            "caption": f"<b>Wall time per unit.</b> Every run on every host (log scale); "
                       f"architectures: {html.escape(_arch_caption(tim[tim.side.isin(spec.timing)]))}. "
                       "Shaded columns are reference tools, which run on the CPU.",
        })  # fmt: skip
    sp = speedups(tim, spec)
    if fig_speedup(sp, out, f"{check}_speedup"):
        tims.append({
            "stem": f"{check}_speedup", "alt": "speed-up",
            "caption": "<b>Speed-up.</b> Reference time / ffs time on the same unit and host; grey "
                       "dots are units, black the median with its 95% bootstrap interval, the median "
                       "printed at the right. Right of the dashed line, ffs is faster. A chain "
                       "(a + b) is timed in full against the reference that does the same job.",
        })  # fmt: skip
    ag = agreement(allruns, spec)
    metric_cols = [c for c in df.columns if "." in c and c.split(".")[0] not in PROV_KEYS]
    allm = (
        df[metric_cols].apply(pd.to_numeric, errors="coerce").describe(percentiles=[0.1, 0.5, 0.9])
    )
    allm = allm.T[["count", "10%", "50%", "90%"]].reset_index().rename(columns={"index": "metric"})
    return {
        "name": check, "who": _who(df) + f", hosts: {', '.join(sorted(allruns.host.unique()))}",
        "summary": html.escape(spec.summary), "code": code, "mixed": mixed,
        "acc": acc, "tim": tims, "agree": _table(ag) if len(ag) else "", "all": _table(allm),
        "speedups": sp.assign(check=check) if len(sp) else sp,
        "row": {"check": check, "units": len(df), "datasets": df.dataset.nunique(),
                "hosts": ", ".join(sorted(allruns.host.unique())),
                "ffs code versions": df["ffs.ffs_sha"].nunique(),
                "scorer versions": df["scorer.check_hash"].fillna("?").nunique()},
    }  # fmt: skip


def build_report(root: Path, out: Path, checks: list[str] | None = None) -> Path:
    import jinja2

    from .checks import CHECKS

    _style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = checks or [n for n, c in CHECKS.items() if getattr(c, "report", None)]
    sections = []
    for n in names:
        spec = getattr(CHECKS[n], "report", None)
        if spec is None:
            print(f"{n}: no report spec, skipped")
            continue
        s = build_section(root, n, spec, out)
        if s is None:
            print(f"{n}: no finished units")
            continue
        print(f"{n}: {len(s['acc'])} accuracy + {len(s['tim'])} timing figures")
        sections.append(s)
    sp = [s["speedups"] for s in sections if len(s["speedups"])]
    overview = fig_overview(pd.concat(sp), out, "overview_speedup") if sp else None
    page = jinja2.Template(PAGE).render(
        built=time.strftime("%Y-%m-%d %H:%M"),
        sources=html.escape(", ".join(h for h, _ in worklist_files(root))),
        overview=overview,
        summary_table=_table(pd.DataFrame([s["row"] for s in sections])) if sections else "",
        sections=sections,
    )
    (out / "index.html").write_text(page)
    return out / "index.html"
