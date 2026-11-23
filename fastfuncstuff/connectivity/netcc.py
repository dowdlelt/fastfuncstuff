"""AFNI ``.netcc`` matrices: 3dNetCorr's text layout, read and written.

Two header lines (ROI count, matrix count), ``# WITH_ROI_LABELS`` and a row of
labels, a row of integer ROI ids, then each matrix as ``# NAME`` followed by N rows
of ``%12.4f`` values separated by tabs. Writing it means ``fat_mat2d_plot.py``,
``fat_mat_sel.py`` and 3dNetCorr-reading scripts take our matrices as they are.
Four decimals is all the format carries; the ``.npz`` written beside it holds the
full precision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def _label(name: str) -> str:
    # The layout is whitespace-delimited: a label with spaces would split a column.
    return "_".join(str(name).split()) or "?"


def write_netcc(
    path: str | Path,
    matrices: Mapping[str, np.ndarray],
    ids: Sequence[int],
    names: Sequence[str],
) -> str:
    n = len(ids)
    if len(names) != n:
        raise ValueError(f"{len(names)} names for {n} ROIs")
    lines = [
        f"# {n}  # Number of network ROIs",
        f"# {len(matrices)}  # Number of netcc matrices",
        "# WITH_ROI_LABELS",
        "\t".join(f" {_label(s):>10s} " for s in names[:-1]) + f"\t  {_label(names[-1]):>10s}"
        if n > 1
        else f"  {_label(names[0]):>10s}",
        "\t".join(f" {int(i):10d} " for i in ids[:-1]) + f"\t  {int(ids[-1]):10d}"
        if n > 1
        else f"  {int(ids[0]):10d}",
    ]
    for key, mat in matrices.items():
        m = np.asarray(mat, dtype=np.float64)
        if m.shape != (n, n):
            raise ValueError(f"matrix {key} is {m.shape}, expected {(n, n)}")
        lines.append(f"# {key}")
        lines.extend("\t".join(f"{v:12.4f}" for v in row) for row in m)
    Path(path).write_text("\n".join(lines) + "\n")
    return str(path)


def read_netcc(path: str | Path) -> tuple[dict[str, np.ndarray], list[int], list[str]]:
    """``(matrices, ids, labels)`` from a ``.netcc`` file (ours or 3dNetCorr's)."""
    raw = Path(path).read_text().splitlines()
    n = int(raw[0].lstrip("#").split()[0])
    k = 3
    labels: list[str] = []
    if raw[2].strip() == "# WITH_ROI_LABELS":
        labels = raw[3].split()
        k = 4
    ids = [int(float(v)) for v in raw[k].split()]
    if not labels:
        labels = [str(i) for i in ids]
    mats: dict[str, np.ndarray] = {}
    i = k + 1
    while i < len(raw):
        line = raw[i].strip()
        if line.startswith("#"):
            key = line.lstrip("#").strip()
            rows = [[float(v) for v in raw[i + 1 + r].split()] for r in range(n)]
            mats[key] = np.asarray(rows)
            i += n + 1
        else:
            i += 1
    return mats, ids, labels
