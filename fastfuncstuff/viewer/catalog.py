"""Reading a directory into a pickable list of datasets.

This is what sits behind **Read**. Everything else in the viewer is modular, but
the data selector is not: underlay, overlay and every mode draw from the same
catalog, so it has to be cheap enough to run the moment a directory is named.

It is: every field here comes from the header, which for a ``.nii.gz`` is a few
KB off the front of the file and for a ``.nii.zst`` is one frame. Scanning a
results directory is milliseconds, not seconds, so the picker can be populated
eagerly rather than on demand.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from fastfuncstuff.io.dsetinfo import DatasetInfo, read_info

#: Extensions the viewer can open. Ordered longest-first so ``.nii.gz`` is
#: matched before ``.gz``.
SUFFIXES = (".nii.gz", ".nii.zst", ".nii", ".HEAD", ".BRIK.gz", ".BRIK")


class Kind(StrEnum):
    """What a dataset is *for*, which is what the picker sorts by.

    A guess from the header, not a guarantee -- but it is the difference between
    a picker that shows a useful default underlay and one that shows an
    alphabetical list where ``all_runs`` sorts above ``anat``.
    """

    ANAT = "anat"
    FUNC = "func"
    STATS = "stats"
    MASK = "mask"
    OTHER = "other"


@dataclass(frozen=True)
class CatalogEntry:
    """One openable dataset, described from its header alone."""

    path: Path
    name: str
    shape: tuple[int, int, int]
    n_volumes: int
    kind: Kind
    tr: float = 0.0
    labels: tuple[str, ...] = ()
    file_bytes: int = 0

    @property
    def is_4d(self) -> bool:
        return self.n_volumes > 1

    @property
    def summary(self) -> str:
        dims = "x".join(str(s) for s in self.shape)
        vols = f" x{self.n_volumes}" if self.is_4d else ""
        return f"{dims}{vols}"


def classify(info: DatasetInfo) -> Kind:
    """Guess a dataset's role from its header and name.

    Sub-brick labels are the one strong signal -- 3dDeconvolve and ffs write
    them on stats output and nothing else does. The rest is naming convention,
    which is weak but is what people actually rely on when scanning a directory.
    """
    if info.labels:
        return Kind.STATS
    stem = info.path.name.lower()
    if any(t in stem for t in ("mask", "automask", "brainmask")):
        return Kind.MASK
    if any(t in stem for t in ("stat", "tstat", "zscore", "coef", "fitts", "bucket")):
        return Kind.STATS
    if info.n_volumes > 1:
        return Kind.FUNC
    if any(t in stem for t in ("anat", "t1", "t2", "mprage", "spgr", "template")):
        return Kind.ANAT
    return Kind.ANAT if info.n_volumes == 1 else Kind.OTHER


def _looks_openable(name: str) -> bool:
    lowered = name.lower()
    # A BRIK without its HEAD is not openable, and listing both would show the
    # same dataset twice; the HEAD is the one to keep.
    if lowered.endswith((".brik", ".brik.gz")):
        return False
    return any(lowered.endswith(s.lower()) for s in SUFFIXES)


def _walk_fd(directory: Path, recursive: bool) -> list[Path] | None:
    """File discovery via ``fd`` when it is installed.

    ``fd`` is the right tool here rather than ``rg``: this is a filename query,
    not a content search, and on a big results tree it is markedly faster than
    walking in Python. Returns ``None`` when it is unavailable so the caller
    falls back rather than failing.
    """
    exe = shutil.which("fd") or shutil.which("fdfind")
    if exe is None:
        return None
    cmd = [exe, "--type", "f", "--absolute-path"]
    if not recursive:
        cmd += ["--max-depth", "1"]
    for suffix in (".nii.gz", ".nii.zst", ".nii", ".HEAD"):
        cmd += ["-e", suffix.lstrip(".")]
    cmd += [".", str(directory)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return [Path(line) for line in out.stdout.splitlines() if line]


def _walk_python(directory: Path, recursive: bool) -> list[Path]:
    if not recursive:
        return [
            Path(e.path) for e in os.scandir(directory) if e.is_file() and _looks_openable(e.name)
        ]
    found: list[Path] = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        found.extend(Path(root) / f for f in files if _looks_openable(f))
    return found


def discover(directory: str | Path, *, recursive: bool = False) -> list[Path]:
    """Openable dataset paths under a directory, sorted."""
    d = Path(directory).expanduser()
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {d}")
    paths = _walk_fd(d, recursive)
    if paths is None:
        paths = _walk_python(d, recursive)
    return sorted({p for p in paths if _looks_openable(p.name)})


def describe(paths: Iterable[str | Path]) -> Iterator[CatalogEntry]:
    """Header-read each path, skipping anything that will not open.

    Unreadable files are dropped rather than raised on: a results directory
    routinely contains a truncated or half-written dataset, and one bad file
    must not stop the picker from listing the other forty.
    """
    for p in paths:
        path = Path(p)
        try:
            info = read_info(path)
        except Exception:
            continue
        if not info.exists or info.shape[0] == 0:
            continue
        nx, ny, nz, nv = info.shape
        yield CatalogEntry(
            path=path,
            name=path.name,
            shape=(int(nx), int(ny), int(nz)),
            n_volumes=max(int(nv), 1),
            kind=classify(info),
            tr=float(info.tr),
            labels=tuple(info.labels),
            file_bytes=int(info.file_bytes),
        )


def natural_key(text: str) -> tuple[tuple[int, int | str], ...]:
    """Sort key that reads digit runs as numbers: ``run-2`` before ``run-10``."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", text.lower())
        if part
    )


def scan(directory: str | Path, *, recursive: bool = False) -> list[CatalogEntry]:
    """Read a directory into a pickable catalog, in natural name order.

    By name and not by guessed kind. Pipelines number their outputs --
    ``stage02.moco``, ``stage12.stats`` -- and that order is the one a person
    navigates by; grouping by kind first scattered each stage across four
    blocks, because a stage's mean volume guesses "anat" and its runs "func".
    Relative to the directory, so a recursive scan keeps subfolders together.
    """
    root = Path(directory)
    entries = list(describe(discover(directory, recursive=recursive)))

    def key(entry: CatalogEntry):
        try:
            rel = entry.path.relative_to(root)
        except ValueError:
            rel = entry.path
        return natural_key(rel.as_posix())

    entries.sort(key=key)
    return entries


def suggest_underlay(entries: list[CatalogEntry]) -> CatalogEntry | None:
    """The most plausible base image in a directory.

    Prefers a real anatomical; falls back to the highest-resolution 3-D volume,
    which in a results directory is usually a template or a mean EPI.
    """
    anats = [e for e in entries if e.kind is Kind.ANAT]
    pool = anats or [e for e in entries if not e.is_4d]
    if not pool:
        return entries[0] if entries else None
    return max(pool, key=lambda e: e.shape[0] * e.shape[1] * e.shape[2])
