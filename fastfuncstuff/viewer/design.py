"""Design regressors in a graph: the model's own timing, drawn beside the data.

A stats bucket already states how it was made. Its HISTORY_NOTE holds the GLM
command line, which names the runs that went in and the design matrix that was
fit, and the xmat's RunStart says where each run's rows begin. So "which
regressor goes with this underlay" has an answer on disk. Nothing here has to
be guessed from a filename, and nothing is drawn when that chain breaks,
because a regressor drawn against the wrong run looks exactly as plausible as
the right one.

Nothing here imports Qt: arrays and paths in, arrays out.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Sub-brick label suffixes that belong to one design column. A GLT, a
#: per-stimulus F and a Full_Fstat span several columns or none, and choosing
#: one of them to draw would misdescribe the statistic.
COLUMN_SUFFIXES = ("_Coef", "_Tstat")

#: GLM programs whose history line names a design matrix, and the flag that
#: names it. ffs_reml's -spec is handled on its own: it names a TOML, and the
#: xmat is derived from it.
MATRIX_FLAGS = {
    "ffs_reml": "-matrix",
    "3dREMLfit": "-matrix",
    "3dDeconvolve": "-x1D",
}


@dataclass(frozen=True)
class Design:
    """A design matrix and where its runs begin."""

    path: str
    matrix: np.ndarray  # (T, k)
    labels: tuple[str, ...]
    #: ColumnGroups, when the file has them: -1 drift, 0 nuisance, >0 stimulus.
    groups: tuple[int, ...]
    run_starts: tuple[int, ...]

    @property
    def n_runs(self) -> int:
        return len(self.run_starts)

    @property
    def n_columns(self) -> int:
        return int(self.matrix.shape[1])

    def run_rows(self, run: int) -> slice:
        """Rows of one run, 1-based as AFNI numbers them in ``Run#1Pol#0``."""
        if not 1 <= run <= self.n_runs:
            raise IndexError(f"{Path(self.path).name} has {self.n_runs} run(s), not run {run}")
        start = self.run_starts[run - 1]
        stop = self.run_starts[run] if run < self.n_runs else self.matrix.shape[0]
        return slice(start, stop)

    def run_length(self, run: int) -> int:
        rows = self.run_rows(run)
        return rows.stop - rows.start

    def column(self, run: int, col: int) -> np.ndarray:
        if not 0 <= col < self.n_columns:
            raise IndexError(f"{Path(self.path).name} has {self.n_columns} columns, not {col}")
        return np.asarray(self.matrix[self.run_rows(run), col], dtype=np.float32)

    def label(self, col: int) -> str:
        return self.labels[col] if 0 <= col < len(self.labels) else f"col{col}"

    def display_order(self) -> list[int]:
        """Stimulus columns first, then nuisance, then drift.

        The stimulus columns are what a timing check is about, and in an xmat
        they sit after thirty drift columns.
        """
        if not self.groups:
            return list(range(self.n_columns))
        stim = [i for i, g in enumerate(self.groups) if g > 0]
        nuisance = [i for i, g in enumerate(self.groups) if g == 0]
        drift = [i for i, g in enumerate(self.groups) if g < 0]
        return stim + nuisance + drift

    def column_for_brick(self, brick_label: str) -> int | None:
        """The one column a ``Faces#0_Coef`` / ``Faces#0_Tstat`` sub-brick describes."""
        for suffix in COLUMN_SUFFIXES:
            if brick_label.endswith(suffix):
                name = brick_label.removesuffix(suffix)
                matches = [i for i, lab in enumerate(self.labels) if lab == name]
                # Labels repeat in an xmat -- motion[0] once per run -- so a
                # name that is not unique does not identify a column.
                return matches[0] if len(matches) == 1 else None
        return None


def load_design(path: str | Path) -> Design:
    """Read an xmat, or a plain 1D file as a design with one run."""
    p = Path(path)
    return _load_design(str(p.resolve()), p.stat().st_mtime_ns)


@lru_cache(maxsize=16)
def _load_design(path: str, _mtime: int) -> Design:
    from fastfuncstuff.io.afni import read_afni_design_matrix

    try:
        design = read_afni_design_matrix(path)
    except (ValueError, KeyError, IndexError, UnicodeDecodeError):
        design = None

    if design is not None and design.get("column_labels"):
        matrix = np.asarray(design["matrix"], dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        starts = [int(s) for s in (design.get("run_starts") or [0])]
        good = design.get("good_list")
        # A censored xmat keeps only the good rows, while RunStart counts the
        # full timeline -- slicing one with the other would shift every run
        # after the first censored TR. GoodList skipping an index is the tell.
        if good is not None and [int(g) for g in good] != list(range(len(good))):
            raise ValueError(
                f"{Path(path).name} is censored ({len(good)} of {int(max(good)) + 1}+ rows "
                "kept); its runs cannot be sliced back out"
            )
        return Design(
            path=path,
            matrix=matrix,
            labels=tuple(design["column_labels"]),
            groups=tuple(int(g) for g in (design.get("column_groups") or ())),
            run_starts=tuple(starts),
        )

    from fastfuncstuff.design.hrf_selection import load_nuisance_file

    matrix = np.asarray(load_nuisance_file(Path(path)), dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    stem = Path(path).name.removesuffix(".1D").removesuffix(".txt")
    return Design(
        path=path,
        matrix=matrix,
        labels=tuple(f"{stem}#{i}" for i in range(matrix.shape[1])),
        groups=(),
        run_starts=(0,),
    )


# ---------------------------------------------------------------------------
# provenance: which design and which runs a stats bucket came from
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """What a GLM history line says went into a bucket."""

    program: str
    inputs: tuple[str, ...]
    design: str


def _program(token: str) -> str:
    return os.path.basename(token)


def _flag_values(tokens: list[str], flag: str) -> list[str]:
    """Every value after ``flag`` up to the next flag.

    Both spellings are accepted, because FfsArgumentParser takes
    ``-foo_bar`` and ``-foo-bar`` alike and a history records what was typed.
    """
    body = flag.lstrip("-")
    want = {flag, "-" + body.replace("_", "-"), "-" + body.replace("-", "_")}
    out: list[str] = []
    for i, tok in enumerate(tokens):
        if tok in want:
            for value in tokens[i + 1 :]:
                if value.startswith("-") and not _is_number(value):
                    break
                out.append(value)
    return out


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _history_lines(history: str) -> list[str]:
    """Commands, with the ``[user@host: date]`` prefix taken off."""
    out = []
    for line in history.splitlines():
        line = re.sub(r"^\s*\[[^\]]*\]\s*", "", line).strip()
        if line:
            out.append(line)
    return out


def parse_provenance(
    history: str, *, stats_name: str = "", base: Path | None = None
) -> Provenance | None:
    """The GLM that wrote a bucket, read from its HISTORY_NOTE.

    The last GLM line that names this file wins, because a history accumulates
    every step that touched the dataset. When no line names it (a renamed
    file) the last GLM line is used. Relative paths are taken against ``base``,
    the bucket's directory: the history does not record a working directory,
    and a pipeline writes its stats next to what it read.
    """
    base = base or Path(".")
    candidates: list[tuple[bool, Provenance]] = []
    for line in _history_lines(history):
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if not tokens:
            continue
        program = _program(tokens[0])
        if program not in MATRIX_FLAGS:
            continue
        inputs = _expand_inputs(_flag_values(tokens, "-input"), base)
        design = _design_path(program, tokens, base)
        if not inputs or design is None:
            continue
        names_it = bool(stats_name) and any(
            os.path.basename(t).startswith(_prefix_stem(stats_name)) for t in tokens[1:]
        )
        candidates.append((names_it, Provenance(program, tuple(inputs), str(design))))
    if not candidates:
        return None
    named = [p for hit, p in candidates if hit]
    return (named or [p for _, p in candidates])[-1]


def _prefix_stem(name: str) -> str:
    """``stats.nii.gz`` and ``stats+orig.HEAD`` are both written as ``stats``."""
    for ext in (".nii.gz", ".nii.zst", ".nii", ".HEAD", ".BRIK", ".BRIK.gz"):
        name = name.removesuffix(ext)
    return re.sub(r"\+(orig|tlrc|acpc)$", "", name)


def _expand_inputs(values: list[str], base: Path) -> list[str]:
    """3dDeconvolve takes ``-input 'a b c'`` as one quoted word; ffs takes a list.

    AFNI selectors (``run1+orig[2..$]``) are cut off: they select volumes, and
    a trimmed run is caught by the length check rather than by parsing them.
    """
    out = []
    for value in values:
        for part in value.split():
            part = re.sub(r"\[.*\]$", "", part)
            out.append(str(_resolve(part, base)))
    return out


def _design_path(program: str, tokens: list[str], base: Path) -> Path | None:
    if program == "ffs_reml":
        spec = _flag_values(tokens, "-spec")
        if spec:
            xmat = _flag_values(tokens, "-xmat")
            if xmat:
                return _resolve(xmat[0], base)
            # ffs_reml's rule: <specname>.xmat.1D next to the spec.
            spec_path = _resolve(spec[0], base)
            return spec_path.with_name(f"{spec_path.stem}.xmat.1D")
    values = _flag_values(tokens, MATRIX_FLAGS[program])
    return _resolve(values[0], base) if values else None


def _resolve(path: str, base: Path) -> Path:
    p = Path(os.path.expanduser(path))
    return p if p.is_absolute() else base / p


def run_of(provenance: Provenance, path: str | Path) -> int | None:
    """Which run (1-based) a dataset was in the fit, or ``None``.

    The full path decides first. The name alone is accepted only when exactly
    one input has it, since run directories often repeat file names.
    """
    target = _canonical(Path(path))
    inputs = [_canonical(Path(p)) for p in provenance.inputs]
    for i, p in enumerate(inputs):
        if p == target:
            return i + 1
    same_name = [i for i, p in enumerate(inputs) if p.name == target.name]
    return same_name[0] + 1 if len(same_name) == 1 else None


def _canonical(p: Path) -> Path:
    try:
        return p.resolve()
    except OSError:
        return p.absolute()


# ---------------------------------------------------------------------------
# pinned regressors: a scalar spec a command can carry
# ---------------------------------------------------------------------------


#: The one line that follows the stats sub-brick rather than a pin.
AUTO_IDENT = "design:auto"


@dataclass(frozen=True)
class Pin:
    """One design column for one run, as a graph window remembers it."""

    path: str
    run: int
    col: int

    def encode(self) -> str:
        # Path last, so a colon inside it survives the split.
        return f"{self.run}:{self.col}:{self.path}"

    @classmethod
    def decode(cls, text: str) -> Pin:
        run, col, path = text.split(":", 2)
        return cls(path=path, run=int(run), col=int(col))

    @property
    def ident(self) -> str:
        return f"design:{self.encode()}"


@dataclass(frozen=True)
class RegressorTrace:
    """A design column as a graph draws it."""

    ident: str
    legend: str
    full: str
    values: np.ndarray
    pin: Pin


def pin_label(design: Design, run: int, col: int) -> str:
    """``r3·active_DI#0`` -- the run first, since the column name repeats across runs."""
    return f"r{run}·{design.label(col)}"


def fit_length(values: np.ndarray, n_time: int) -> np.ndarray:
    """Pad with zeros or cut to ``n_time``.

    A graph stretches every line across the cell by its own length, so a
    460-row regressor beside a 450-volume run would be drawn with its events
    progressively late. Zero is a convolved regressor's own baseline.
    """
    if n_time <= 0 or values.size == n_time:
        return values
    if values.size > n_time:
        return values[:n_time]
    out = np.zeros(n_time, dtype=values.dtype)
    out[: values.size] = values
    return out


__all__ = [
    "AUTO_IDENT",
    "Design",
    "Pin",
    "Provenance",
    "RegressorTrace",
    "fit_length",
    "load_design",
    "parse_provenance",
    "pin_label",
    "run_of",
]
