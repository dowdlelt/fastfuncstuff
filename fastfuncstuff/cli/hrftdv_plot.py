"""ffs_hrftdv_plot — depth profiles of ffs_hrftdv -atlas -roi_lag outputs, one PNG per ROI.

    ffs_hrftdv_plot -prefix tdv/sub01                     # the best-fit ROI
    ffs_hrftdv_plot -prefix tdv/sub01 -top 5
    ffs_hrftdv_plot -prefix tdv/sub01 -rois FFA_rh OFA_rh -reml stats_roi.tsv

Panels: lag through depth against the ROI's early/late pair (error bars: the spread
of the same fit run by run); the column, early, late and per-depth curves; and,
given ffs_reml's ROI x depth table (-reml), the early and late betas through depth
per condition (error bars: beta / t).  Depth runs white (bottom) to pial (top).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter


def create_parser() -> FfsArgumentParser:
    parser = FfsArgumentParser(
        prog="ffs_hrftdv_plot", description=__doc__, formatter_class=FfsHelpFormatter
    )
    parser.add_argument("-prefix", required=True, help="The ffs_hrftdv -prefix")
    parser.add_argument(
        "-reml",
        default=None,
        metavar="TSV",
        help="ffs_reml -atlas ... -depths N -hrf_opt PREFIX's {Rbuck stem}_roi.tsv: adds "
        "the early/late betas through depth",
    )
    which = parser.add_mutually_exclusive_group()
    which.add_argument(
        "-rois", nargs="+", default=None, metavar="NAME", help="ROIs to plot, by name"
    )
    which.add_argument(
        "-top", type=int, default=1, help="Plot the N ROIs whose depth rows fit best (mean R²)"
    )
    parser.add_argument(
        "-out_dir", default=None, help="Where the PNGs go (default: next to the prefix)"
    )
    return parser


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True}
    )
    return plt


def _rows(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _f(row: dict, key: str) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", "nan") else np.nan


def _key(row: dict) -> tuple[str, str]:
    return row["roi"], row.get("hemi", "")


def _label(rows: list[dict]) -> str:
    r = rows[0]
    return f"{r['name']}_{r['hemi']}" if r.get("hemi") else r["name"]


def rank_rois(table: list[dict]) -> list[tuple[str, str]]:
    """ROI keys, best mean depth-row R² first."""
    by: dict[tuple[str, str], list[float]] = {}
    for r in table:
        by.setdefault(_key(r), []).append(_f(r, "r2"))
    return sorted(by, key=lambda k: -np.nanmean(by[k]))


def _select(table: list[dict], rois) -> list[tuple[str, str]]:
    keys = list(dict.fromkeys(_key(r) for r in table))
    out = []
    for want in rois:
        if isinstance(want, tuple):
            want = (str(want[0]), str(want[1]))
            hit = [k for k in keys if k == want]
        else:
            hit = [
                k for k in keys
                if want in (_label([r for r in table if _key(r) == k]),
                            next(r["name"] for r in table if _key(r) == k))
            ]  # fmt: skip
        if not hit:
            raise ValueError(f"no ROI {want!r} in the depth table")
        out += hit
    return list(dict.fromkeys(out))


def plot_roi(rows: list[dict], shapes: np.ndarray, dt: float, reml_rows: list[dict] | None,
             path: str) -> str:  # fmt: skip
    """One ROI's figure from its depth-table rows (any order)."""
    from fastfuncstuff.processing.hrftdv import stretch_shift

    plt = _plt()
    rows = sorted(rows, key=lambda r: int(r["depth"]))
    frac = np.array([_f(r, "depth_frac") for r in rows])
    lag = np.array([_f(r, "lag") for r in rows])
    sd = np.array([_f(r, "lag_run_sd") for r in rows])
    n_panels = 3 if reml_rows else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.8), layout="constrained")
    cmap = plt.get_cmap("viridis")
    colours = [cmap(i / max(len(rows) - 1, 1)) for i in range(len(rows))]

    ax = axes[0]
    ax.errorbar(lag, frac, xerr=sd, color="k", marker="o", ms=4, capsize=2, lw=1)
    for key, colour, name in (("pair_early_lag", "tab:blue", "early"), ("pair_late_lag", "tab:red", "late")):  # fmt: skip
        ax.axvline(_f(rows[0], key), color=colour, ls="--", lw=1, label=f"pair {name}")
    ax.set_xlabel("lag vs column curve (s)")
    ax.set_ylabel("depth (0 white, 1 pial)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="best")
    ax.set_title("lag through depth")

    ax = axes[1]
    s = int(_f(rows[0], "shape"))
    h = shapes[s]
    t = np.arange(h.size) * dt
    for row, colour in zip(rows, colours, strict=True):
        if np.isfinite(_f(row, "lag")):
            c = stretch_shift(h, dt, _f(row, "lag"), _f(row, "width"))
            ax.plot(t, c, color=colour, lw=0.8, alpha=0.8)
    ax.plot(t, h, color="k", lw=1.5, label="column")
    for kind, colour in (("early", "tab:blue"), ("late", "tab:red")):
        c = stretch_shift(h, dt, _f(rows[0], f"pair_{kind}_lag"), _f(rows[0], f"pair_{kind}_width"))
        ax.plot(t, c, color=colour, lw=1.5, ls="--", label=kind)
    ax.set_xlim(0, min(t[-1], 20.0))
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("curves (thin: per depth, dark = white)")

    if reml_rows:
        ax = axes[2]
        rr = sorted(reml_rows, key=lambda r: int(r["depth"]))
        rf = np.array([_f(r, "depth_frac") for r in rr])
        conds = sorted({k[: -len("_early#0_Coef")] for k in rr[0] if k.endswith("_early#0_Coef")})
        styles = ["-", ":", "-.", (0, (5, 1))]
        for ci, cond in enumerate(conds):
            for kind, colour in (("early", "tab:blue"), ("late", "tab:red")):
                b = np.array([_f(r, f"{cond}_{kind}#0_Coef") for r in rr])
                tt = np.array([_f(r, f"{cond}_{kind}#0_Tstat") for r in rr])
                se = np.abs(np.divide(b, tt, out=np.full_like(b, np.nan), where=tt != 0))
                ax.errorbar(b, rf, xerr=se, color=colour, ls=styles[ci % len(styles)],
                            marker="o", ms=3, capsize=2, lw=1, label=f"{cond} {kind}")  # fmt: skip
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_xlabel("beta")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="best")
        ax.set_title("early / late betas")
    fig.suptitle(f"{_label(rows)}  (mean R² {np.nanmean([_f(r, 'r2') for r in rows]):.2f})")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_from_files(
    prefix: str,
    rois: list | None = None,
    *,
    top: int = 1,
    reml: str | None = None,
    out_dir: str | None = None,
) -> list[str]:
    """PNGs for ``rois`` (names or ``(roi id, hemi)``), else the ``top`` best-fit ones."""
    import torch

    table = _rows(f"{prefix}_roi_depth.tsv")
    lib = torch.load(f"{prefix}_hrf_library.pt", weights_only=False)
    if "unit_shapes" not in lib:
        raise ValueError(f"{prefix}_hrf_library.pt predates -roi_lag plots: rerun ffs_hrftdv")
    shapes = np.asarray(lib["unit_shapes"])
    dt = float(lib["metadata"]["microtime_dt"])
    keys = _select(table, rois) if rois else rank_rois(table)[:top]
    reml_rows = _rows(reml) if reml else None
    where = Path(out_dir) if out_dir else Path(prefix).parent
    where.mkdir(parents=True, exist_ok=True)
    out = []
    for k in keys:
        rows = [r for r in table if _key(r) == k]
        rr = [r for r in reml_rows if _key(r) == k] if reml_rows else None
        if reml_rows is not None and not rr:
            print(f"  ⚠️  {_label(rows)}: not in {reml}; plotting without betas")
        name = f"{Path(prefix).name}_roi_depth_{_label(rows)}.png"
        out.append(plot_roi(rows, shapes, dt, rr or None, str(where / name)))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = create_parser()
    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args(argv)
    try:
        paths = plot_from_files(
            args.prefix, args.rois, top=args.top, reml=args.reml, out_dir=args.out_dir
        )
    except (OSError, ValueError, KeyError) as exc:
        sys.exit(f"ERROR: {exc}")
    for p in paths:
        print(f"  Wrote {p}")


if __name__ == "__main__":
    main()
