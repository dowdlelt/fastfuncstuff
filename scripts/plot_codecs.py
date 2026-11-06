#!/usr/bin/env python
"""Plot the CSV from ``bench_codecs.sh``: one row per file, compress left, decompress right.

    python scripts/plot_codecs.py codecs.csv            # writes codecs.png beside it
    python scripts/plot_codecs.py codecs.csv -o fig.png
    python scripts/plot_codecs.py codecs.csv -split 4,4,3   # codecs_1.png .. codecs_3.png

Bars are medians over the repetitions, fastest at the top, coloured by tool in the
same colours as ``bench_nifti_compression.py``'s figures. A decoder that runs on one
core is drawn faded: for gzip that is not a setting anyone chose but the format's
limit, and the fade keeps it from reading as a competitor that merely lost.
Each panel has its own linear seconds axis, so compare lengths within a panel and
read the printed numbers across panels.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from bench_nifti_compression import (  # noqa: E402
    BLUE,
    GRID,
    INK,
    MUTED,
    ORANGE,
    TOOL_COLOR,
    _mb_label,
    _style,
)

ROW_IN = 0.24  # figure inches per bar


def tool_of(name: str) -> str:
    return re.match(r"[a-z]+", name).group()  # type: ignore[union-attr]


def threads_of(name: str) -> int:
    m = re.search(r"\.[pPT](\d+)$", name)
    return int(m.group(1)) if m else 1


def encoder_label(name: str) -> str:
    """igzip-2.T6 -> 'igzip -2 (ISA-L), 6 thr'."""
    tool, rest = name.split("-", 1) if "-" in name else (name, "")
    level = rest.split(".")[0]
    if tool == "igzip" and level.startswith("py-"):  # the single-thread Python fallback
        return f"python -m isal.igzip -{level[3:]}, 1 thr"
    isal = " (ISA-L)" if tool == "igzip" else ""
    return f"{tool} -{level}{isal}, {threads_of(name)} thr"


def decoder_label(method: str) -> str:
    """'pzstd-1.p6>pzstd.p6' -> 'pzstd -1 file → pzstd -d, 6 thr'."""
    writer, reader = method.split(">")
    w_tool, rest = writer.split("-", 1)
    w_level = rest.split(".")[0].removeprefix("py-")
    r_tool = tool_of(reader)
    n = threads_of(reader)
    # pigz -d always inflates on one core; -p only buys it helper threads for I/O and CRC.
    note = "1 thr" if r_tool == "pigz" else f"{n} thr"
    return f"{w_tool} -{w_level} file → {r_tool}, {note}"


def single_core(reader: str) -> bool:
    return tool_of(reader) == "pigz" or threads_of(reader) == 1


def load(path: Path) -> tuple[dict, dict, list[str]]:
    times: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    sizes: dict[tuple[str, str, str], float] = {}
    plain: dict[str, int] = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (r["file"], r["op"], r["method"])
            times[key].append(float(r["seconds"]))
            sizes[key] = 100 * int(r["bytes"]) / int(r["plain_bytes"])
            plain[r["file"]] = int(r["plain_bytes"])
    med = {k: statistics.median(v) for k, v in times.items()}
    files = sorted(plain, key=plain.get)  # type: ignore[arg-type]
    return med, {"size": sizes, "plain": plain}, files


def meta_line(csv_path: Path) -> str:
    meta = csv_path.with_name(csv_path.name + ".meta")
    if not meta.exists():
        return ""
    first = meta.read_text().splitlines()[0].lstrip("# ")
    return first


def short(name: str, width: int = 44) -> str:
    """Keep both ends of a long BIDS name; the middle is the part everyone shares."""
    return name if len(name) <= width else f"{name[: width // 2 - 1]}…{name[-(width // 2 - 1) :]}"


def bar_panel(ax, rows: list[tuple[str, float, str, bool, str]], title: str, n_slots: int) -> None:
    """rows: (label, seconds, colour, faded, annotation), drawn fastest at the top.

    Both panels of a file get the same number of slots, so a bar is equally thick in
    each; the shorter list leaves blank space at the bottom instead of fat bars.
    """
    rows = sorted(rows, key=lambda r: r[1])
    ys = [n_slots - 1 - i for i in range(len(rows))]
    for y, (_, s, colour, faded, _) in zip(ys, rows, strict=True):
        ax.barh(y, s, height=0.7, color=colour, alpha=0.38 if faded else 1.0, lw=0)
    top = max(r[1] for r in rows)
    for y, (_, s, _, _, note) in zip(ys, rows, strict=True):
        ax.text(s + top * 0.012, y, f"{s:.3f} s{note}", va="center", fontsize=8, color=INK)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_xlim(0, top * 1.3)
    ax.set_ylim(-0.6, n_slots - 0.4)
    _style(ax)
    ax.grid(False, axis="y")
    ax.set_xlabel("seconds (shorter is less waiting)", color=MUTED, fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("csv", type=Path)
    p.add_argument("-o", "--out", type=Path)
    p.add_argument(
        "-split",
        help="files per figure, smallest first, e.g. 4,4,3; writes <out>_1.png, _2.png, ...",
    )
    args = p.parse_args()
    out = args.out or args.csv.with_suffix(".png")
    med, extra, files = load(args.csv)

    panels = []
    for f in files:
        enc = [
            (
                encoder_label(m),
                s,
                TOOL_COLOR.get(tool_of(m).replace("igzip", "isal"), BLUE),
                False,
                f"   {extra['size'][(f, op, m)]:.1f}% of original",
            )
            for (ff, op, m), s in med.items()
            if ff == f and op == "compress"
        ]
        dec = [
            (
                decoder_label(m),
                s,
                BLUE
                if tool_of(m.split(">")[1]) == "igzip"
                else TOOL_COLOR.get(tool_of(m.split(">")[1]), ORANGE),
                single_core(m.split(">")[1]),
                "",
            )
            for (ff, op, m), s in med.items()
            if ff == f and op == "decompress"
        ]
        panels.append((f, enc, dec))

    groups = [int(g) for g in args.split.split(",")] if args.split else [len(panels)]
    if sum(groups) != len(panels):
        p.error(f"-split adds to {sum(groups)}, but the CSV has {len(panels)} files")
    start = 0
    for i, n in enumerate(groups):
        part = out if len(groups) == 1 else out.with_name(f"{out.stem}_{i + 1}{out.suffix}")
        draw(panels[start : start + n], extra, meta_line(args.csv), part)
        start += n
        print(part)


def draw(panels: list, extra: dict, meta: str, out: Path) -> None:
    slots = [max(len(e), len(d)) for _, e, d in panels]
    gap = 3.4  # rows of space for the axis labels below a panel and the title above the next
    fig_h = (sum(slots) + gap * len(slots)) * ROW_IN + 1.0
    fig, axes = plt.subplots(
        len(panels),
        2,
        figsize=(13, fig_h),
        squeeze=False,
        gridspec_kw={"height_ratios": slots},
    )
    for (f, enc, dec), (ax_c, ax_d), n in zip(panels, axes, slots, strict=True):
        size = _mb_label(extra["plain"][f] / 1e6)
        bar_panel(ax_c, enc, f"{short(f)}  {size}\ncompress", n)
        bar_panel(ax_d, dec, f"{short(f)}  {size}\ndecompress to /dev/null", n)
    fig.subplots_adjust(
        top=1 - 1.25 / fig_h,
        bottom=0.45 / fig_h,
        left=0.17,
        right=0.985,
        wspace=0.62,
        hspace=gap / (sum(slots) / len(slots)),
    )
    fig.text(
        0.012,
        1 - 0.18 / fig_h,
        "gzip and zstd tools, timed from the shell",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.012,
        1 - 0.5 / fig_h,
        f"{meta}.\nRAM disk to RAM disk, median of shuffled repetitions. "
        "Blue ISA-L, green zstd/pzstd, orange pigz, yellow rapidgzip; faded = one core.",
        fontsize=9,
        color=MUTED,
        va="top",
    )
    for ax in axes.flat:
        ax.spines["left"].set_color(GRID)
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
