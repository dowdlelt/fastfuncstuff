"""Output-file naming for ffs_autoproc-generated scripts.

Every intermediate/final file follows a single, greppable scheme so that a
human reading the working directory can tell at a glance which stage, session,
fieldmap, task and run produced it::

    stageNN.<label>[.ses-<S>][.task-<T>][.fmap-<F>][.run-<R>][.part-phase]<ext>

Consistent BIDS-style ``key-value`` tokens throughout, with the run entity kept
raw (``run-010``, no re-padding). Fieldmap groups carry an explicit ``fmap-<F>``
so multi-fieldmap sessions stay unambiguous. Tokens are omitted where they don't
apply (an anat matrix has none; a per-session grandmean has only ``ses-<S>``).

Tool-specific suffixes (``.aff12.1D``, ``_warp``, ``.1D``) are appended by the
tools; this module owns only the *prefix* stem. ``coord()`` returns just the
coordinate fragment (no ``stageNN.label``) so the emitter can build the same
filenames in its bash loops as the Python data table references — one source of
truth, no drift.
"""

from __future__ import annotations

from dataclasses import dataclass

# Stage numbers are fixed so the on-disk sort order matches processing order,
# and so a partially-run directory is self-describing. Gaps are intentional
# (room to insert without renumbering everything).
STAGE_NUMBERS: dict[str, int] = {
    "nordic": 0,
    "tshift": 1,
    "moco": 2,
    "nlmoco": 3,
    "blip": 4,
    "xfmap": 5,
    "xrun": 6,
    "grandmean": 7,
    "xses": 8,
    "xref": 9,  # align this data's grandmean to an external -grand_reference
    "anat": 9,
    "nlanat": 9,
    "final": 10,
    "scale": 11,
    "stats": 12,
}


@dataclass(frozen=True)
class NameKey:
    """The addressable coordinates of a file within the plan.

    Any of ``session``/``task``/``fmap``/``run`` may be ``None`` when the file is
    not specific to that level (a session grandmean has only ``session``; a blip
    fieldmap has ``session``+``fmap`` but no run; an anat matrix has none).
    """

    label: str
    session: str | None = None
    task: str | None = None
    fmap: str | None = None
    run: str | None = None
    part: str | None = None  # e.g. "phase"


def coord(key: NameKey) -> str:
    """The coordinate fragment (``ses-06.task-X.fmap-Y.run-010``), no stage prefix.

    Examples
    --------
    >>> coord(NameKey("moco", session="06", task="HalfFovNoTask", run="010"))
    'ses-06.task-HalfFovNoTask.run-010'
    >>> coord(NameKey("blip", session="WB", fmap="PA01"))
    'ses-WB.fmap-PA01'
    """
    parts: list[str] = []
    if key.session is not None:
        parts.append(f"ses-{key.session}")
    if key.task is not None:
        parts.append(f"task-{key.task}")
    if key.fmap is not None:
        parts.append(f"fmap-{key.fmap}")
    if key.run is not None:
        parts.append(f"run-{key.run}")  # raw run entity, not re-padded
    if key.part is not None:
        parts.append(f"part-{key.part}")
    return ".".join(parts)


def stem(key: NameKey) -> str:
    """Build the prefix stem (no extension) for a file at ``key``.

    Examples
    --------
    >>> stem(NameKey("moco", session="06", task="HalfFovNoTask", run="010"))
    'stage02.moco.ses-06.task-HalfFovNoTask.run-010'
    >>> stem(NameKey("grandmean", session="SM"))
    'stage07.grandmean.ses-SM'
    >>> stem(NameKey("anat"))
    'stage09.anat'
    """
    if key.label not in STAGE_NUMBERS:
        raise KeyError(f"unknown stage label {key.label!r}")
    prefix = f"stage{STAGE_NUMBERS[key.label]:02d}.{key.label}"
    c = coord(key)
    return f"{prefix}.{c}" if c else prefix
