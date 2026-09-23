"""Ortvec entries: a regressor file plus what was done to it.

An InstaGLM ortvec list holds *entries*, not bare paths, and an entry is a path
followed by the transforms that turn the file into regressors::

    motion.1D                         the six columns as written
    motion.1D:deriv                   their backward differences
    motion.1D:deriv:deriv             the second difference
    motion.1D:band=0-0.01             each column, only its slowest part
    motion.1D:cols=2,4:band=0.01-0.08 two columns, one band of them

The modifier syntax is the CLI's -- ``-ortvec FILE motion:deriv`` --
extended, so an entry reads the same in the viewer as on a command line.
Transforms apply left to right.

Why entries and not a checkbox: a derived regressor is a *different* regressor,
and the whole point of the list is that each one can be unticked on its own to
see what it was buying. "Motion with derivatives" as one switch answers one
question; motion and its derivative as two ticks answer three.

Bands are split with a DCT rather than an FFT. A DCT treats the column as
mirrored at its ends instead of wrapped, so a motion trace that ends somewhere
other than where it started does not ring -- and the bands of a split sum back
to the column exactly, which is what lets a split *replace* its source in the
model rather than sit beside it collinearly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Separates a path from its transforms and the transforms from each other.
OP_SEP = ":"
#: Written for a band with no upper edge. Spelled out rather than left empty so
#: ``band=0.08-`` is not mistaken for a truncated entry.
NYQUIST = "nyq"


@dataclass(frozen=True)
class Band:
    lo: float
    #: ``None`` runs to Nyquist.
    hi: float | None

    def tag(self) -> str:
        """``<0.01Hz``, ``0.01-0.08Hz`` or ``>0.08Hz``: short enough for a column picker."""
        if self.lo <= 0 and self.hi is not None:
            return f"<{self.hi:g}Hz"
        if self.hi is None:
            return f">{self.lo:g}Hz"
        return f"{self.lo:g}-{self.hi:g}Hz"

    def spec(self) -> str:
        return f"band={_num(self.lo)}-{NYQUIST if self.hi is None else _num(self.hi)}"


def _num(value: float) -> str:
    """Four significant figures, never in exponent form -- ``1e-05`` has a dash in it."""
    return np.format_float_positional(float(value), precision=4, fractional=False, trim="-")


def _parse_op(text: str) -> str | int | tuple[int, ...] | Band | None:
    """One transform, or ``None`` if the text is not one (it is part of the path)."""
    from fastfuncstuff.cli_utils import NUISANCE_TRANSFORMS

    if text in NUISANCE_TRANSFORMS and text != "none":
        return text
    key, eq, value = text.partition("=")
    if not eq:
        return None
    try:
        if key == "cols":
            return tuple(int(c) for c in value.split(",") if c.strip())
        if key == "band":
            lo, dash, hi = value.partition("-")
            if not dash:
                return None
            return Band(float(lo), None if hi.strip() in (NYQUIST, "") else float(hi))
    except ValueError:
        return None
    return None


def parse_entry(entry: str) -> tuple[str, list]:
    """``"a.1D:cols=0:deriv"`` to ``("a.1D", [(0,), "deriv"])``.

    Peeled from the right, and only while what is peeled is a transform, so a
    colon that belongs to the path -- ``C:\\data`` -- stays in the path.
    """
    path, ops = entry, []
    while True:
        head, sep, tail = path.rpartition(OP_SEP)
        if not sep:
            break
        op = _parse_op(tail)
        if op is None:
            break
        ops.insert(0, op)
        path = head
    return path, ops


def with_op(entry: str, op: str) -> str:
    return f"{entry}{OP_SEP}{op}"


def describe_entry(entry: str) -> str:
    """What a list row shows: the file's name and, after it, what was done to it."""
    path, ops = parse_entry(entry)
    parts = [path.replace("\\", "/").rsplit("/", 1)[-1]]
    n_deriv = 0
    for op in ops:
        if isinstance(op, str):
            n_deriv += 1
            continue
        if n_deriv:
            parts.append(_deriv_word(n_deriv))
            n_deriv = 0
        if isinstance(op, Band):
            parts.append(op.tag())
        elif isinstance(op, tuple):
            parts.append(f"[{','.join(str(c) for c in op)}]")
    if n_deriv:
        parts.append(_deriv_word(n_deriv))
    return "  ".join(parts)


def _deriv_word(n: int) -> str:
    return "d/dt" if n == 1 else f"d{_SUPER.get(n, n)}/dt{_SUPER.get(n, n)}"


_SUPER = {2: "²", 3: "³"}


# -- the arithmetic -----------------------------------------------------------
def dct_frequencies(n_time: int, tr: float) -> np.ndarray:
    """The frequency, in Hz, each DCT-II coefficient of an n-sample column sits at."""
    return np.arange(n_time, dtype=np.float64) / (2.0 * n_time * tr)


def band_columns(columns: np.ndarray, tr: float, band: Band) -> np.ndarray:
    """The part of each column whose DCT coefficients fall in ``[lo, hi)``.

    Half-open so that adjacent bands of a split share no coefficient and miss
    none; the top band is closed at Nyquist. The mean sits at 0 Hz and so goes
    with the lowest band, which is what makes a split sum to its source.
    """
    from scipy.fft import dct, idct

    columns = np.asarray(columns, dtype=np.float64)
    freqs = dct_frequencies(columns.shape[0], tr)
    keep = freqs >= band.lo
    if band.hi is not None:
        keep &= freqs < band.hi
    coef = dct(columns, type=2, norm="ortho", axis=0)
    coef[~keep] = 0.0
    return idct(coef, type=2, norm="ortho", axis=0)


def column_spectra(columns: np.ndarray, tr: float) -> tuple[np.ndarray, np.ndarray]:
    """``(freqs, power)`` of each column, power as a fraction of its variance.

    The DCT's, so the spectrum shown is exactly the one :func:`band_columns`
    cuts: a band's share of the area under a curve is the share of that
    column's variance the band regressor will carry. Normalised per column
    because a file mixes millimetres and degrees, and a spectrum drawn in raw
    power would be a picture of the units. The mean is left out -- it is the
    baseline's, not the column's.
    """
    from scipy.fft import dct

    columns = np.asarray(columns, dtype=np.float64)
    coef = dct(columns, type=2, norm="ortho", axis=0)[1:]
    power = coef**2
    total = power.sum(axis=0, keepdims=True)
    power = np.divide(power, total, out=np.zeros_like(power), where=total > 0)
    return dct_frequencies(columns.shape[0], tr)[1:], power


def apply_ops(
    columns: np.ndarray, labels: Sequence[str], ops: Sequence, tr: float
) -> tuple[np.ndarray, list[str]]:
    """Run an entry's transforms over the columns its file contributed."""
    from fastfuncstuff.cli_utils import apply_nuisance_transform

    out, names = np.asarray(columns, dtype=np.float64), list(labels)
    for op in ops:
        if isinstance(op, tuple):
            bad = [c for c in op if not 0 <= c < out.shape[1]]
            if bad:
                raise IndexError(f"cols={bad} but the file has {out.shape[1]} columns")
            out, names = out[:, list(op)], [names[c] for c in op]
        elif isinstance(op, Band):
            if tr <= 0:
                raise ValueError("a band split is in Hz and this run has no TR")
            out, names = band_columns(out, tr, op), [f"{n} {op.tag()}" for n in names]
        else:
            out = apply_nuisance_transform(out, op)
            names = [f"{n}'" if op in ("deriv", "deriv_back") else f"{n}:{op}" for n in names]
    return out, names


def read_columns(path: str) -> tuple[np.ndarray, list[str]]:
    """A regressor file's columns, by the rule DERIVE applies.

    An xmat contributes its ColumnGroups nuisance and a plain 1D file all of
    itself -- one opinion about what counts as a regressor, not two.
    """
    from fastfuncstuff.viewer.derive import _read_matrix

    columns, labels, _kind = _read_matrix(Path(path))
    return np.asarray(columns, dtype=np.float64), list(labels)


# -- editing a list ----------------------------------------------------------
def split_entries(
    entry: str, cutoffs: Sequence[float], *, n_columns: int, columns: Sequence[int] | None = None
) -> list[str]:
    """The entries that replace ``entry`` once it is cut at ``cutoffs`` Hz.

    One entry per band, so each can be unticked on its own. When only some
    columns are split, the rest come along as an entry of their own: the
    source has to be switched off (its bands sum to it), and switching it off
    must not quietly drop the columns nobody asked to split.
    """
    edges = sorted({float(c) for c in cutoffs if c > 0})
    chosen = sorted(set(range(n_columns) if columns is None else columns))
    subset = 0 < len(chosen) < n_columns
    base = with_op(entry, f"cols={','.join(map(str, chosen))}") if subset else entry
    bounds = [0.0, *edges]
    out = [
        with_op(base, Band(lo, edges[i] if i < len(edges) else None).spec())
        for i, lo in enumerate(bounds)
    ]
    if subset:
        rest = [c for c in range(n_columns) if c not in chosen]
        out.append(with_op(entry, f"cols={','.join(map(str, rest))}"))
    return out


__all__ = [
    "Band",
    "apply_ops",
    "band_columns",
    "column_spectra",
    "dct_frequencies",
    "describe_entry",
    "parse_entry",
    "read_columns",
    "split_entries",
    "with_op",
]
