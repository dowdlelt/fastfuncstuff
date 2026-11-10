"""The QC record a tool writes under ``-qc DIR``: named metrics plus its figures.

One JSON file per (tool, label) under ``DIR/records/``. Metrics are flat and
named ``<tool>.<metric>`` once gathered, the way AFNI's ``ss_review`` keys are, so
a table across subjects is a plain join. Each metric carries its unit, a short
description, and optionally a warning limit; the limit travels with the number
so ``ffs_qc_report`` / ``ffs_qc_table`` flag it without a separate config.

Figure names sort check-first, run-second::

    20.moco.01-motion.run-01.png
    20.moco.01-motion.run-02.png
    20.moco.02-tsnr.run-01.png

``20`` is the tool's place in a pipeline (:data:`QC_ORDER`), ``01`` is the check's
place in that tool's story. A step number is fixed per check, never counted, so
a check skipped on one run cannot shift every later figure out of alignment.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Where each tool's figures sit in a folder listing: pipeline order, with gaps so
# a new tool slots in without renumbering.
QC_ORDER: dict[str, int] = {
    "nordic": 10,
    "tshift": 15,
    "moco": 20,
    "locomoco": 25,
    "blipflip": 30,
    "segment": 40,
    "allineate": 50,
    "nwarp": 60,
    "denoise": 70,
    "reml": 80,
}
DEFAULT_ORDER = 90

RECORDS_DIR = "records"

_OPS = {
    "gt": (lambda v, x: v > x, ">"),
    "ge": (lambda v, x: v >= x, ">="),
    "lt": (lambda v, x: v < x, "<"),
    "le": (lambda v, x: v <= x, "<="),
}

_IMAGE_EXTS = (".nii.gz", ".nii.zst", ".nii", ".HEAD", ".BRIK.gz", ".BRIK", ".1D", ".txt")


def label_from_path(path: str | os.PathLike) -> str:
    """A run label from a file name: directory and imaging extension removed,
    AFNI view suffix (``+orig``) dropped."""
    name = Path(path).name
    for ext in _IMAGE_EXTS:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return re.sub(r"\+(orig|tlrc|acpc)$", "", name)


def clean_label(label: str) -> str:
    """Make a label safe inside a file name (no separators, no spaces)."""
    return re.sub(r"[\s/\\:]+", "_", label.strip()) or "run"


@dataclass
class Metric:
    value: float | int | str | None
    unit: str = ""
    desc: str = ""
    warn: dict | None = None  # {"op": "gt", "limit": 0.1}

    def flagged(self) -> bool:
        return metric_flagged(asdict_metric(self))


def asdict_metric(m: Metric) -> dict:
    d: dict[str, Any] = {"value": m.value}
    if m.unit:
        d["unit"] = m.unit
    if m.desc:
        d["desc"] = m.desc
    if m.warn:
        d["warn"] = m.warn
    return d


def metric_flagged(m: dict) -> bool:
    """True when a metric's value crosses its own warning limit."""
    warn = m.get("warn")
    v = m.get("value")
    if not warn or not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    if not math.isfinite(float(v)):
        return True
    fn, _ = _OPS[warn["op"]]
    return bool(fn(float(v), float(warn["limit"])))


def warn_text(warn: dict | None) -> str:
    if not warn:
        return ""
    return f"{_OPS[warn['op']][1]} {warn['limit']:g}"


def _jsonable(v):
    if v is None or isinstance(v, (str, bool)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if isinstance(v, int) or (hasattr(v, "dtype") and getattr(v.dtype, "kind", "") in "iu"):
        return int(v)
    return f if math.isfinite(f) else None


@dataclass
class QCRecord:
    """Collects one tool's QC for one label, then writes it.

    Build it with :meth:`for_args` from a tool's parsed ``-qc`` / ``-qc_label``.
    """

    qc_dir: Path
    tool: str
    label: str
    order: int = DEFAULT_ORDER
    metrics: dict[str, Metric] = field(default_factory=dict)
    figures: list[dict] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_args(cls, args, tool: str, fallback_label: str | os.PathLike) -> QCRecord | None:
        """The record for a run, or None when ``-qc`` was not given."""
        qc_dir = getattr(args, "qc", None)
        if not qc_dir:
            return None
        label = getattr(args, "qc_label", None) or label_from_path(fallback_label)
        d = Path(qc_dir)
        d.mkdir(parents=True, exist_ok=True)
        return cls(d, tool, clean_label(label), QC_ORDER.get(tool, DEFAULT_ORDER))

    def metric(
        self,
        name: str,
        value,
        unit: str = "",
        desc: str = "",
        *,
        warn_gt: float | None = None,
        warn_lt: float | None = None,
    ) -> None:
        warn = None
        if warn_gt is not None:
            warn = {"op": "gt", "limit": float(warn_gt)}
        elif warn_lt is not None:
            warn = {"op": "lt", "limit": float(warn_lt)}
        self.metrics[name] = Metric(_jsonable(value), unit, desc, warn)

    def figure_path(self, step: int, check: str, ext: str = ".png") -> Path:
        return self.qc_dir / f"{self.order:02d}.{self.tool}.{step:02d}-{check}.{self.label}{ext}"

    def add_figure(self, step: int, check: str, title: str, path: Path) -> None:
        self.figures.append({"step": step, "check": check, "title": title, "file": path.name})

    def flagged(self) -> list[str]:
        return [k for k, m in self.metrics.items() if m.flagged()]

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "label": self.label,
            "order": self.order,
            "info": {k: _jsonable(v) for k, v in self.info.items()},
            "metrics": {k: asdict_metric(m) for k, m in self.metrics.items()},
            "figures": sorted(self.figures, key=lambda f: f["step"]),
        }

    def write(self, verb: int = 1) -> Path:
        out = self.qc_dir / RECORDS_DIR / f"{self.order:02d}.{self.tool}.{self.label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1))
        os.replace(tmp, out)
        if verb >= 1:
            n = len(self.figures)
            flags = self.flagged()
            msg = f"  QC: {n} figure{'s' if n != 1 else ''} + metrics -> {self.qc_dir}/"
            if flags:
                msg += f"  (check: {', '.join(flags)})"
            print(msg)
        return out


def load_records(qc_dir: str | os.PathLike) -> list[dict]:
    """Every record in a QC folder, in pipeline-then-label order."""
    rdir = Path(qc_dir) / RECORDS_DIR
    if not rdir.is_dir():
        return []
    recs = []
    for p in sorted(rdir.glob("*.json"), key=lambda p: natural_key(p.name)):
        try:
            recs.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return recs


def natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def add_qc_args(parser_or_group) -> None:
    """Register ``-qc DIR`` and ``-qc_label LABEL`` on a tool's parser."""
    g = parser_or_group
    if hasattr(parser_or_group, "add_argument_group"):
        g = parser_or_group.add_argument_group("Quality control")
    g.add_argument(
        "-qc",
        default=None,
        metavar="DIR",
        help="Write QC figures (PNG) and a metrics record into DIR. Figures sort "
        "check-first, run-second, so a folder of several runs reads in order in any "
        "image viewer. ffs_qc_report DIR stitches them into a page and a table; "
        "ffs_qc_table compares subjects.",
    )
    g.add_argument(
        "-qc_label",
        default=None,
        metavar="LABEL",
        help="Name this run in QC file names and tables (default: from the input or "
        "-prefix file name).",
    )
