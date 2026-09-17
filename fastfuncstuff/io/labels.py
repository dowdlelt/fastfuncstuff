"""Value label tables: what an integer in a label volume means.

A label volume -- an atlas, a segmentation, the output of a clusterize -- is a
picture whose values are names. The names live somewhere else, and "somewhere
else" is four different places depending on who wrote the file:

* **AFNI** puts them in the header, as a ``VALUE_LABEL_DTABLE`` (what
  ``3drefit -labeltable`` writes) or, for a real atlas, an
  ``ATLAS_LABEL_TABLE`` of ``ATLAS_POINT`` elements.
* **BIDS** puts them in a ``*_dseg.tsv`` beside the image, with an ``index``
  column and a ``name`` column.
* **FreeSurfer** puts them in a LUT: ``index name R G B A``, one per line.
* **Nobody** puts them anywhere, and the value is its own name.

All four collapse to ``{value: LabelEntry}``, which is the only shape anything
downstream should have to know about. Colours ride along because two of the
formats carry them and a segmentation's conventional colours are worth keeping
-- a reader that dropped them would force the viewer to invent colours for
structures that already have famous ones.

Cheap to import on purpose, like :mod:`fastfuncstuff.io.headers`: numpy-free,
torch-free, so asking "what are this atlas's regions called" costs a file read.
"""

from __future__ import annotations

import csv
import html
import re
from dataclasses import dataclass
from pathlib import Path

#: Sidecar extensions, in the order they are tried. A ``.niml.lt`` is AFNI's
#: own external label table and is the most specific, so it wins over a
#: same-stem ``.txt`` that might be anything.
SIDECAR_SUFFIXES = (".niml.lt", ".niml", ".tsv", ".csv", ".lut", ".txt")

_ATR_RE = r'atr_name\s*=\s*"{}"[^>]*>(.*?)</AFNI_atr>'
#: Pairs of quoted strings in a NIML dtable: ``"10" "Left-Thalamus"``.
_PAIR_RE = re.compile(r'"\s*(-?\d+)\s*"\s+"([^"]*)"')
#: One ``<ATLAS_POINT ... STRUCT="..." ... VAL="..." >`` element's attributes.
_POINT_RE = re.compile(r"<ATLAS_POINT\b(.*?)/?>", re.S)
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


@dataclass(frozen=True)
class LabelEntry:
    """One row of a label table."""

    value: int
    name: str
    #: 0-255 RGB when the table carried one. ``None`` means "choose a colour",
    #: which is a different statement from "this region is black".
    color: tuple[int, int, int] | None = None


def _entries(pairs) -> dict[int, LabelEntry]:
    out: dict[int, LabelEntry] = {}
    for value, name, color in pairs:
        if value == 0:  # background is not a region
            continue
        text = str(name).strip().strip('"')
        if text:
            out[int(value)] = LabelEntry(int(value), text, color)
    return out


# -- headers -----------------------------------------------------------
def _afni_atr(text: str, name: str) -> str | None:
    m = re.search(_ATR_RE.format(re.escape(name)), text, re.S)
    return m.group(1) if m else None


def parse_dtable(text: str) -> dict[int, LabelEntry]:
    """Parse a NIML ``VALUE_LABEL_DTABLE`` body.

    Tolerant of escaping because the same table is stored escaped inside a
    NIfTI extension and unescaped in a ``.niml.lt`` file, and a reader that
    handled only one of those would work on exactly half the datasets that
    carry one.
    """
    return _entries((int(v), n, None) for v, n in _PAIR_RE.findall(html.unescape(text)))


def parse_atlas_points(text: str) -> dict[int, LabelEntry]:
    """Parse ``ATLAS_POINT`` elements -- AFNI's atlas label table."""
    rows = []
    for body in _POINT_RE.findall(html.unescape(text)):
        attrs = dict(_ATTR_RE.findall(body))
        val, name = attrs.get("VAL"), attrs.get("STRUCT")
        if val is None or name is None:
            continue
        try:
            rows.append((int(float(val)), name, None))
        except ValueError:
            continue
    return _entries(rows)


def read_value_labels(img) -> dict[int, LabelEntry]:
    """Label table from a NIfTI image's AFNI extension (``{}`` when absent)."""
    from fastfuncstuff.io.headers import _afni_ext_text

    text = _afni_ext_text(img)
    if not text:
        return {}
    body = _afni_atr(text, "ATLAS_LABEL_TABLE")
    if body:
        found = parse_atlas_points(body)
        if found:
            return found
    body = _afni_atr(text, "VALUE_LABEL_DTABLE")
    return parse_dtable(body) if body else {}


# -- sidecars ----------------------------------------------------------
def _parse_delimited(path: Path) -> dict[int, LabelEntry]:
    """A BIDS ``_dseg.tsv``/``.csv``: a header row naming ``index`` and ``name``."""
    with path.open(newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delimiter = "\t" if "\t" in sample.splitlines()[0] else ","
        rows = list(csv.reader(fh, delimiter=delimiter))
    if not rows:
        return {}
    head = [c.strip().lower() for c in rows[0]]
    if "index" not in head:
        return {}
    i_index = head.index("index")
    i_name = next((head.index(c) for c in ("name", "label", "abbreviation") if c in head), None)
    i_color = next((head.index(c) for c in ("color", "rgb") if c in head), None)
    out = []
    for row in rows[1:]:
        if len(row) <= i_index or not row[i_index].strip():
            continue
        try:
            value = int(float(row[i_index]))
        except ValueError:
            continue
        name = row[i_name].strip() if i_name is not None and len(row) > i_name else str(value)
        out.append((value, name, _parse_color(row[i_color]) if i_color is not None else None))
    return _entries(out)


def _parse_color(text: str) -> tuple[int, int, int] | None:
    text = text.strip().lstrip("#")
    if len(text) == 6 and all(c in "0123456789abcdefABCDEF" for c in text):
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    parts = re.split(r"[\s,]+", text)
    if len(parts) >= 3:
        try:
            r, g, b = (int(float(p)) for p in parts[:3])
        except ValueError:
            return None
        return (r, g, b)
    return None


def _parse_lut(path: Path) -> dict[int, LabelEntry]:
    """A FreeSurfer-style LUT: ``index name R G B A``, ``#`` comments."""
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        color = None
        if len(parts) >= 5:
            try:
                color = (int(parts[2]), int(parts[3]), int(parts[4]))
            except ValueError:
                color = None
        rows.append((value, parts[1], color))
    return _entries(rows)


def read_label_file(path: str | Path) -> dict[int, LabelEntry]:
    """Read a label table from a sidecar, sniffing the format by content."""
    p = Path(path)
    text = p.read_text(errors="ignore")
    if "ATLAS_POINT" in text:
        return parse_atlas_points(text)
    if "VALUE_LABEL_DTABLE" in text or _PAIR_RE.search(text):
        found = parse_dtable(text)
        if found:
            return found
    if p.suffix.lower() in (".tsv", ".csv"):
        return _parse_delimited(p)
    return _parse_lut(p)


def _stem(path: Path) -> str:
    """The name a sidecar would share with the image.

    ``sub-01_dseg.nii.gz`` should find ``sub-01_dseg.tsv``, and an atlas called
    ``Schaefer400.nii.gz`` should find ``Schaefer400.txt``. Both are the file
    name with the image suffixes taken off, which is one rule -- and one that
    ``Path.stem`` gets wrong on every double extension in neuroimaging.
    """
    name = path.name
    for suffix in (".nii.gz", ".nii.zst", ".nii", ".HEAD", ".BRIK.gz", ".BRIK"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def find_label_file(path: str | Path) -> Path | None:
    """The sidecar label table beside a label volume, if there is one."""
    p = Path(path)
    stem = _stem(p)
    for suffix in SIDECAR_SUFFIXES:
        candidate = p.parent / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def label_table(path: str | Path, img=None) -> dict[int, LabelEntry]:
    """Every label table route for one dataset, header first then sidecar.

    The header wins because it travels with the data: a sidecar left behind by
    a copy is exactly the case where the two disagree, and the one inside the
    file is the one that describes *this* file.
    """
    if img is not None:
        found = read_value_labels(img)
        if found:
            return found
    sidecar = find_label_file(path)
    if sidecar is None:
        return {}
    try:
        return read_label_file(sidecar)
    except (OSError, UnicodeDecodeError):
        return {}


__all__ = [
    "SIDECAR_SUFFIXES",
    "LabelEntry",
    "find_label_file",
    "label_table",
    "parse_atlas_points",
    "parse_dtable",
    "read_label_file",
    "read_value_labels",
]
