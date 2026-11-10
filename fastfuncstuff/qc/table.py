"""Many QC folders -> one cross-subject table, flags, and a strip-plot overview.

``ffs_qc_report`` writes a table per subject; this joins them, the way AFNI's
``gen_ss_review_table.py`` joins ``out.ss_review`` files. Two kinds of flag:

- **limits** — a metric crossing the warning limit its tool recorded, or one
  given on the command line (``-flag "moco.censor_frac > 0.1"``);
- **outliers** — a robust z-score (median / 1.4826·MAD) beyond ``-z`` among the
  rows of the same column. This is the flag that needs no prior about the
  acquisition: a run whose tSNR is far below its peers stands out whatever the
  voxel size.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .record import load_records, metric_flagged, natural_key, warn_text

_FLAG_RE = re.compile(r"^\s*([\w.\-]+)\s*(>=|<=|>|<)\s*([-+0-9.eE]+)\s*$")
_FLAG_OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}


@dataclass
class Flag:
    subject: str
    label: str
    column: str
    value: float
    reason: str


def subject_name(qc_dir: Path) -> str:
    """A folder named ``qc`` is named for its parent (``sub-01/qc`` -> ``sub-01``)."""
    d = qc_dir.resolve()
    return d.parent.name if d.name.lower() in ("qc", "qc_report") else d.name


def parse_flag(spec: str) -> tuple[str, str, float]:
    m = _FLAG_RE.match(spec)
    if not m:
        raise ValueError(f"cannot read flag {spec!r}; expected e.g. 'moco.censor_frac > 0.1'")
    return m.group(1), m.group(2), float(m.group(3))


def gather(qc_dirs: list[Path]) -> tuple[list[str], list[dict], dict]:
    """Rows (subject, label, metric values) plus per-column metadata."""
    columns: list[str] = []
    meta: dict[str, dict] = {}
    rows: dict[tuple[str, str], dict] = {}
    for d in qc_dirs:
        subj = subject_name(d)
        for rec in load_records(d):
            row = rows.setdefault((subj, rec["label"]), {"subject": subj, "label": rec["label"]})
            for name, m in rec["metrics"].items():
                col = f"{rec['tool']}.{name}"
                if col not in columns:
                    columns.append(col)
                    meta[col] = {k: m.get(k) for k in ("unit", "desc")}
                row[col] = m
    order = sorted(rows, key=lambda k: (natural_key(k[0]), natural_key(k[1])))
    return columns, [rows[k] for k in order], meta


def numeric(rows: list[dict], col: str) -> np.ndarray:
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        m = r.get(col)
        v = m.get("value") if m else None
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            out[i] = float(v)
    return out


def robust_z(x: np.ndarray, min_n: int = 5) -> np.ndarray:
    z = np.full_like(x, np.nan)
    ok = np.isfinite(x)
    if ok.sum() < min_n:
        return z
    med = np.median(x[ok])
    mad = 1.4826 * np.median(np.abs(x[ok] - med))
    if mad <= 0:
        return z
    z[ok] = (x[ok] - med) / mad
    return z


def find_flags(
    columns: list[str], rows: list[dict], user_flags: list[str] = (), z_limit: float = 3.5
) -> list[Flag]:
    flags: list[Flag] = []
    for r in rows:
        for c in columns:
            m = r.get(c)
            if m and metric_flagged(m):
                flags.append(
                    Flag(
                        r["subject"], r["label"], c, m["value"], f"limit {warn_text(m.get('warn'))}"
                    )
                )
    for spec in user_flags:
        col, op, lim = parse_flag(spec)
        x = numeric(rows, col)
        hit = np.isfinite(x) & _FLAG_OPS[op](x, lim)
        for i in np.flatnonzero(hit):
            flags.append(
                Flag(rows[i]["subject"], rows[i]["label"], col, float(x[i]), f"{op} {lim:g}")
            )
    if z_limit and z_limit > 0:
        for c in columns:
            x = numeric(rows, c)
            z = robust_z(x)
            for i in np.flatnonzero(np.abs(np.nan_to_num(z)) > z_limit):
                flags.append(
                    Flag(
                        rows[i]["subject"],
                        rows[i]["label"],
                        c,
                        float(x[i]),
                        f"robust z {z[i]:+.1f}",
                    )
                )
    return flags


def write_table(path: Path, columns: list[str], rows: list[dict]) -> None:
    from .report import fmt_value

    with open(path, "w") as f:
        f.write("\t".join(["subject", "label", *columns]) + "\n")
        for r in rows:
            vals = [fmt_value(r[c]["value"]) if c in r else "NA" for c in columns]
            f.write("\t".join([r["subject"], r["label"], *vals]) + "\n")


def write_flags(path: Path, flags: list[Flag]) -> None:
    from .report import fmt_value

    with open(path, "w") as f:
        f.write("subject\tlabel\tmetric\tvalue\treason\n")
        for fl in flags:
            f.write(f"{fl.subject}\t{fl.label}\t{fl.column}\t{fmt_value(fl.value)}\t{fl.reason}\n")


def strip_plot(path: Path, columns: list[str], rows: list[dict], meta: dict, flags: list[Flag]):
    """One small panel per numeric metric: every row a dot, flagged rows labelled."""
    from matplotlib.figure import Figure

    from . import render, style

    cols = [c for c in columns if np.isfinite(numeric(rows, c)).sum() >= 2]
    if not cols:
        return None
    flagged = {(f.subject, f.label, f.column) for f in flags}
    ncol = 4
    nrow = math.ceil(len(cols) / ncol)
    rng = np.random.default_rng(0)
    with style.mpl_style():
        fig = Figure(figsize=(style.WIDTH / 100, 2.3 * nrow + 0.3), dpi=100)
        axes = fig.subplots(nrow, ncol, squeeze=False).ravel()
        for ax, c in zip(axes, cols, strict=False):
            x = numeric(rows, c)
            ok = np.isfinite(x)
            jit = rng.uniform(-0.28, 0.28, x.size)
            hit = np.array([(r["subject"], r["label"], c) in flagged for r in rows])
            ax.scatter(
                jit[ok & ~hit],
                x[ok & ~hit],
                s=22,
                color=style.BLUE,
                alpha=0.8,
                edgecolors=style.SURFACE,
                linewidths=0.6,
            )
            if (ok & hit).any():
                ax.scatter(
                    jit[ok & hit],
                    x[ok & hit],
                    s=34,
                    color=style.STATUS["warn"],
                    edgecolors=style.SURFACE,
                    linewidths=0.6,
                    zorder=3,
                )
                for i in np.flatnonzero(ok & hit)[:6]:
                    ax.annotate(
                        rows[i]["subject"],
                        (jit[i], x[i]),
                        xytext=(5, 0),
                        textcoords="offset points",
                        fontsize=8,
                        color=style.INK2,
                        va="center",
                    )
            med = np.nanmedian(x)
            ax.axhline(med, color=style.INK2, lw=0.8, ls=(0, (3, 3)))
            unit = (meta.get(c) or {}).get("unit") or ""
            ax.set_title(c + (f" ({unit})" if unit else ""), fontsize=10)
            ax.set_xticks([])
            ax.set_xlim(-0.6, 0.6)
            ax.grid(axis="x", visible=False)
        for ax in axes[len(cols) :]:
            ax.set_visible(False)
        fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.03, hspace=0.45, wspace=0.3)
        body = render.figure_image(fig)
    n_subj = len({r["subject"] for r in rows})
    img = render.compose(
        body,
        "Cross-subject QC",
        f"{n_subj} subjects · {len(rows)} rows · {len(cols)} metrics",
        [(f"{len({(f.subject, f.label) for f in flags})} rows flagged", "warn" if flags else "ok")],
        "Each dot is one run; dashed line = median. Yellow dots crossed a limit or sit beyond the "
        "robust z cut among their peers; the table and flags TSV say which and why.",
    )
    return render.save_png(img, path)


def build_table(
    qc_dirs: list[str | os.PathLike],
    prefix: str,
    user_flags: list[str] = (),
    z_limit: float = 3.5,
    png: bool = True,
) -> dict:
    dirs = [Path(d) for d in qc_dirs]
    columns, rows, meta = gather(dirs)
    flags = find_flags(columns, rows, user_flags, z_limit)
    stem = Path(prefix)
    stem.parent.mkdir(parents=True, exist_ok=True)
    tsv = stem.with_name(stem.name + ".tsv")
    ftsv = stem.with_name(stem.name + "_flags.tsv")
    write_table(tsv, columns, rows)
    write_flags(ftsv, flags)
    out = {"rows": rows, "columns": columns, "flags": flags, "tsv": tsv, "flags_tsv": ftsv}
    if png:
        out["png"] = strip_plot(stem.with_name(stem.name + ".png"), columns, rows, meta, flags)
    return out
