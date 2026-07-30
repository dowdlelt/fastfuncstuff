"""Lightweight BIDS scanner for ffs_autoproc.

Deliberately *not* pybids: we walk the tree with pathlib, parse entities from
filenames with a regex, and read JSON sidecars ourselves. This keeps the
dependency surface at zero and — more importantly — lets us encode explicit
fallbacks for the real-world non-compliance we hit in practice (see the
``# quirk:`` notes below), which a strict validator-driven indexer chokes on.

Scope for this milestone: magnitude BOLD, optional per-run phase, magnitude
SBRef, reverse-PE fieldmaps (either conventional ``fmap/*_epi`` or task-tagged
``fmap/*_bold``/``*_sbref``), and a single anat T1w. Everything is returned as
plain dataclasses; reference/warp logic lives in ``plan.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# BIDS entity = value tokens: ``key-value`` separated by underscores. Value is
# alphanumeric (BIDS forbids separators inside a value).
_ENTITY_RE = re.compile(r"(?:^|_)(?P<key>[a-zA-Z0-9]+)-(?P<val>[a-zA-Z0-9]+)")
# The suffix is the final ``_<suffix>`` before the extension (bold, sbref, epi,
# T1w, ...). ``part``/``dir``/etc. are entities, not suffixes.
_SUFFIX_RE = re.compile(r"_(?P<suffix>[a-zA-Z0-9]+)\.(?:nii\.gz|nii|nii\.zst)$")
_NIFTI_RE = re.compile(r"\.(?:nii\.gz|nii|nii\.zst)$")


def parse_entities(filename: str) -> dict[str, str]:
    """Return the BIDS ``key-value`` entities present in ``filename``."""
    return {m.group("key"): m.group("val") for m in _ENTITY_RE.finditer(filename)}


def parse_suffix(filename: str) -> str | None:
    m = _SUFFIX_RE.search(filename)
    return m.group("suffix") if m else None


def sidecar_path(nifti: Path) -> Path:
    """The ``.json`` sidecar path for a NIfTI (``.nii[.gz|.zst]`` → ``.json``)."""
    return nifti.parent / (_NIFTI_RE.sub("", nifti.name) + ".json")


# ---------------------------------------------------------------------------
# JSON sidecar loading with BIDS inheritance
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def load_sidecar(nifti: Path, bids_root: Path) -> dict:
    """Load a sidecar with BIDS inheritance (dataset-root → subject → session).

    BIDS lets metadata live at coarser levels (e.g. a dataset-root
    ``task-foo_bold.json`` that applies to every ``task-foo`` run — this is how
    ds001555 stores TR/SliceTiming). More-specific files override less-specific
    ones, so we merge from the root down and let the local sidecar win.
    """
    ents = parse_entities(nifti.name)
    suffix = parse_suffix(nifti.name)
    merged: dict = {}
    # Inheritance search dirs, coarse → fine.
    search_dirs = [bids_root]
    sub = ents.get("sub")
    ses = ents.get("ses")
    if sub:
        search_dirs.append(bids_root / f"sub-{sub}")
        if ses:
            search_dirs.append(bids_root / f"sub-{sub}" / f"ses-{ses}")
    search_dirs.append(nifti.parent)
    for d in search_dirs:
        if not d.is_dir():
            continue
        for jf in sorted(d.glob("*.json")):
            jents = parse_entities(jf.name)
            jsuffix = parse_suffix(jf.name + ".nii")  # suffix regex needs an ext
            if jsuffix is None:
                jsuffix = jf.stem.rsplit("_", 1)[-1] if "_" in jf.stem else None
            # An inherited sidecar applies only if its entities are a subset of
            # ours and the suffix matches (BIDS inheritance principle).
            if jsuffix is not None and suffix is not None and jsuffix != suffix:
                continue
            if any(ents.get(k) != v for k, v in jents.items()):
                continue
            merged.update(_load_json(jf))
    # The exact sidecar (same-name) always wins last.
    exact = sidecar_path(nifti)
    if exact.is_file():
        merged.update(_load_json(exact))
    return merged


# ---------------------------------------------------------------------------
# Events lookup
# ---------------------------------------------------------------------------

# Entities an ``_events.tsv`` may carry, in BIDS filename order. Everything else
# on the BOLD name (part, echo, inv, ...) describes the *image*, not the task, so
# it must be dropped: a `part-mag_bold.nii.gz` run pairs with a `..._events.tsv`
# that has no `part-` at all. Naively swapping `_bold`→`_events` misses it.
_EVENTS_ENTITIES = ("sub", "ses", "task", "acq", "ce", "dir", "rec", "run")
# Order in which entities are dropped to widen the search, finest first. `task`
# is never dropped — an events file without it belongs to a different task.
_EVENTS_DROP_ORDER = ("rec", "ce", "dir", "acq", "run", "ses", "sub")


def find_events(bold_path: Path, bids_root: Path | str | None = None) -> Path | None:
    """The events TSV for a BOLD run, following BIDS inheritance.

    Searches the run's own directory first, then each parent up to ``bids_root``
    (session → subject → dataset root), trying progressively fewer entities so a
    shared ``task-<T>_events.tsv`` at the root is found after the per-run file.
    Returns None when the task has no events anywhere (a resting-state run, or a
    dataset where the timing simply was not shared).
    """
    ents = parse_entities(bold_path.name)
    keys = [k for k in _EVENTS_ENTITIES if k in ents]
    if "task" not in keys:
        return None

    stems: list[str] = []
    for drop in (0, *range(1, len(_EVENTS_DROP_ORDER) + 1)):
        dropped = set(_EVENTS_DROP_ORDER[:drop])
        kept = [k for k in keys if k not in dropped]
        s = "_".join(f"{k}-{ents[k]}" for k in kept)
        if s not in stems:
            stems.append(s)

    # Walk up in the *given* form (relative stays relative, matching how the rest
    # of the scanner records paths); resolve only to test against bids_root.
    root = Path(bids_root).resolve() if bids_root else None
    dirs = [bold_path.parent]
    for d in bold_path.parent.parents:
        dirs.append(d)
        if root is not None and d.resolve() == root:
            break
        if root is None and len(dirs) > 4:  # func/ → ses → sub → root
            break

    for directory in dirs:
        for s in stems:
            cand = directory / f"{s}_events.tsv"
            if cand.is_file():
                return cand
    return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


def _acq_seconds(value) -> float | None:
    """Parse a BIDS ``AcquisitionTime`` into seconds-of-day for ordering.

    Accepts ``"HH:MM:SS[.ffffff]"`` or a full ``...THH:MM:SS`` datetime. Within a
    session (one visit/day) seconds-of-day orders acquisitions correctly.
    """
    if not value:
        return None
    s = str(value)
    if "T" in s:  # datetime → keep the time part
        s = s.split("T", 1)[1]
    try:
        h, m, sec = s.split(":")[:3]
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except (ValueError, TypeError):
        return None


@dataclass
class BoldRun:
    subject: str
    session: str | None
    task: str
    run: str  # BIDS run entity as-found (may be non-contiguous); "" if absent
    mag_path: Path
    json: dict
    phase_path: Path | None = None
    sbref_path: Path | None = None  # magnitude SBRef, if present
    fmap_id: str | None = None  # assigned in plan.py

    @property
    def tr(self) -> float | None:
        v = self.json.get("RepetitionTime")
        return float(v) if v is not None else None

    @property
    def pe_dir(self) -> str | None:
        return self.json.get("PhaseEncodingDirection")

    @property
    def has_slice_timing(self) -> bool:
        return bool(self.json.get("SliceTiming"))

    @property
    def task_name(self) -> str:
        # quirk: TaskName is absent from every func sidecar in some
        # data; derive it from the entity so downstream (BIDS validator, GLM)
        # still has one.
        return self.json.get("TaskName") or self.task

    @property
    def rep(self) -> Path:
        """Representative 3D image for registration: SBRef if present, else the
        4D BOLD (the emitter takes a mean/base from it)."""
        return self.sbref_path or self.mag_path

    @property
    def acq_time(self) -> float | None:
        return _acq_seconds(self.json.get("AcquisitionTime"))


@dataclass
class FmapGroup:
    """A reverse-PE fieldmap and the runs it corrects.

    ``reverse_path`` is the blip-down (opposite PE) image. The forward (blip-up)
    image is ``forward_path`` when ``fmap/`` supplies its own matched-PE mate
    (the AP/PA pair case — the pair is self-contained, so the correction never
    borrows a data run); otherwise it is None and the emitter uses the first
    intended run's SBRef/mean. Runs are associated via ``IntendedFor`` when
    present, else by task+session.

    ``json`` is the sidecar of the image whose distorted space the estimated
    field lives in: the forward image for a pair, the lone reverse image
    otherwise.
    """

    session: str | None
    fmap_id: str  # human tag: task name, ``acq`` value, or ``dir-run`` fallback
    reverse_path: Path
    json: dict
    # (task, run) pairs — run numbers repeat across tasks, so the pair is the
    # unique run identity this fmap corrects.
    intended_runs: list[tuple[str, str]] = field(default_factory=list)
    forward_path: Path | None = None

    @property
    def run_ids(self) -> list[str]:
        """Just the run entities (for display), de-duplicated in order."""
        seen: list[str] = []
        for _task, run in self.intended_runs:
            if run not in seen:
                seen.append(run)
        return seen

    @property
    def readout(self) -> float | None:
        v = self.json.get("TotalReadoutTime")
        return float(v) if v is not None else None

    @property
    def pe_dir(self) -> str | None:
        return self.json.get("PhaseEncodingDirection")

    @property
    def acq_time(self) -> float | None:
        return _acq_seconds(self.json.get("AcquisitionTime"))


@dataclass
class Session:
    session: str | None
    bold_runs: list[BoldRun] = field(default_factory=list)
    fmaps: list[FmapGroup] = field(default_factory=list)
    anat: Path | None = None

    @property
    def tasks(self) -> list[str]:
        seen: list[str] = []
        for r in self.bold_runs:
            if r.task not in seen:
                seen.append(r.task)
        return seen


@dataclass
class Subject:
    subject: str
    sessions: list[Session] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def _is_phase(nifti: Path, ents: dict[str, str], bids_root: Path) -> bool:
    """True if this file holds phase data.

    quirk: some acquisitions mislabel phase as ``part-sbref_bold``; the only
    reliable signal is ``ImageType`` containing ``PHASE`` in the sidecar.
    """
    if ents.get("part") == "phase":
        return True
    image_type = load_sidecar(nifti, bids_root).get("ImageType") or []
    return any("PHASE" in str(t).upper() for t in image_type)


def _norm_id(value: str, prefix: str) -> str:
    """Normalize a user-supplied subject/session id to the bare BIDS value.

    Accepts ``1`` → ``1`` (label ``sub-1``), ``sub-01`` → ``01``, ``SM`` → ``SM``.
    """
    value = str(value)
    return value[len(prefix) :] if value.startswith(prefix) else value


def scan_subject(
    bids_root: str | Path,
    subject: str,
    sessions: list[str] | None = None,
    tasks: list[str] | None = None,
    fmap_pe_dir: str | None = None,
) -> Subject:
    """Scan one subject into a :class:`Subject` tree.

    ``sessions``/``tasks`` optionally restrict the scan (values may be bare
    labels or full ``ses-``/``task-`` forms). Returns sessions and runs in
    directory-sorted order; run numbers are taken as-found (never assumed
    contiguous).

    ``fmap_pe_dir`` names the ``dir`` label whose polarity matches the BOLD runs
    (e.g. ``AP``) — needed to pair opposite-PE fieldmaps when the sidecars omit
    ``PhaseEncodingDirection``. See :func:`_pick_pe_pair`.
    """
    bids_root = Path(bids_root)
    sub_label = _norm_id(subject, "sub-")
    sub_dir = bids_root / f"sub-{sub_label}"
    if not sub_dir.is_dir():
        raise FileNotFoundError(f"subject dir not found: {sub_dir}")

    want_ses = {_norm_id(s, "ses-") for s in sessions} if sessions else None
    want_task = {_norm_id(t, "task-") for t in tasks} if tasks else None

    session_dirs = sorted(sub_dir.glob("ses-*"))
    if not session_dirs:
        session_dirs = [sub_dir]  # sessionless layout

    out = Subject(subject=sub_label)
    for sdir in session_dirs:
        ses_label = _norm_id(sdir.name, "ses-") if sdir.name.startswith("ses-") else None
        if want_ses is not None and ses_label not in want_ses:
            continue
        sess = _scan_session(sdir, bids_root, sub_label, ses_label, want_task, fmap_pe_dir)
        if sess.bold_runs:
            out.sessions.append(sess)
    return out


def _scan_session(
    sdir: Path,
    bids_root: Path,
    sub_label: str,
    ses_label: str | None,
    want_task: set[str] | None,
    fmap_pe_dir: str | None = None,
) -> Session:
    sess = Session(session=ses_label)

    # ---- func: group by (task, run) so mag/phase/sbref land on one BoldRun ----
    func_dir = sdir / "func"
    grouped: dict[tuple[str, str], dict] = {}
    if func_dir.is_dir():
        for nf in sorted(func_dir.glob("*.nii*")):
            if not _NIFTI_RE.search(nf.name):
                continue
            ents = parse_entities(nf.name)
            suffix = parse_suffix(nf.name)
            task = ents.get("task")
            if task is None:
                continue
            if want_task is not None and task not in want_task:
                continue
            run = ents.get("run", "")
            slot = grouped.setdefault((task, run), {})
            if suffix == "sbref":
                if _is_phase(nf, ents, bids_root):
                    slot.setdefault("sbref_phase", nf)  # quirk file; unused now
                else:
                    slot["sbref"] = nf
            elif suffix == "bold":
                # The run's phase timeseries is an *explicit* part-phase bold.
                # A file that merely has PHASE ImageType but isn't part-phase
                # (e.g. the mislabeled part-sbref_bold) is a phase SBRef, not the
                # 4D phase — set it aside so it can't masquerade as the phase.
                if ents.get("part") == "phase":
                    slot["phase"] = nf
                elif _is_phase(nf, ents, bids_root):
                    slot.setdefault("phase_quirk", nf)
                else:
                    slot["mag"] = nf

    for (task, run), slot in sorted(grouped.items()):
        mag = slot.get("mag")
        if mag is None:
            continue  # no magnitude BOLD → nothing to process
        sess.bold_runs.append(
            BoldRun(
                subject=sub_label,
                session=ses_label,
                task=task,
                run=run,
                mag_path=mag,
                json=load_sidecar(mag, bids_root),
                phase_path=slot.get("phase"),
                sbref_path=slot.get("sbref"),
            )
        )

    # ---- fmap: reverse-PE images (conventional epi OR task-tagged bold/sbref) ----
    fmap_dir = sdir / "fmap"
    if fmap_dir.is_dir():
        sess.fmaps = _scan_fmaps(fmap_dir, bids_root, sess.bold_runs, fmap_pe_dir)

    # ---- anat: prefer acq-uni T1w (MP2RAGE), else first T1w / MPRAGE ----
    sess.anat = _pick_anat(sdir)
    return sess


def _scan_fmaps(
    fmap_dir: Path,
    bids_root: Path,
    bold_runs: list[BoldRun],
    fmap_pe_dir: str | None = None,
) -> list[FmapGroup]:
    """Build FmapGroups from the reverse-PE images in ``fmap/``.

    Preference order for the representative image within a group: SBRef > EPI >
    BOLD (SBRef is cleanest). Grouping key is the ``task`` entity when present
    (Fieldmaps could betask-tagged), else ``run`` (conventional epi). ``acq`` is
    *not* a group key: ``acq-bold``/``acq-sbref`` are two forms of one fieldmap,
    not two fieldmaps.

    ``dir`` is not a group key either, for the same reason one step out: when
    ``fmap/`` holds both PE polarities of the same acquisition (dir-AP + dir-PA),
    that opposite-PE *pair* is one fieldmap — a self-contained correction that no
    data run contributes to. Only when a polarity stands alone does ``dir`` split
    groups (the fmap is then the reverse image and a data run is the forward).
    """
    # tag -> dir (or "" when absent) -> form -> (path, sidecar)
    candidates: dict[str, dict[str, dict[str, tuple[Path, dict]]]] = {}
    for nf in sorted(fmap_dir.glob("*.nii*")):
        if not _NIFTI_RE.search(nf.name):
            continue
        ents = parse_entities(nf.name)
        if _is_phase(nf, ents, bids_root):
            continue  # phase fmap unused this milestone
        suffix = parse_suffix(nf.name)
        # One group per DISTINCT fieldmap = task and/or run, but NOT acq — those
        # are two forms of one fieldmap. A task-tagged fmap with a run (SKILLED:
        # task-skilled_dir-PA_run-1/2/3) is several fmaps, so the run must be in
        # the key or they'd collapse into one.
        task, run, d = ents.get("task"), ents.get("run"), ents.get("dir")
        tag = (task or "") + (f"-run{run}" if run else "")
        # Form within the group: acq (bold/sbref) refines the plain suffix so the
        # SBRef form wins the preference below even for conventional epi fmaps.
        form = ents.get("acq") or suffix or "epi"
        candidates.setdefault(tag, {}).setdefault(d or "", {})[form] = (
            nf,
            load_sidecar(nf, bids_root),
        )

    groups: list[FmapGroup] = []
    for tag, by_dir in candidates.items():
        picks = {d: p for d, slot in by_dir.items() if (p := _pick_form(slot)) is not None}
        pair = _pick_pe_pair(picks, bold_runs, fmap_pe_dir)
        if pair is not None:
            fwd_dir, rev_dir = pair
            fwd_nf, fwd_js = picks[fwd_dir]
            rev_nf, _ = picks[rev_dir]
            gid = tag.lstrip("-") or f"{fwd_dir}-{rev_dir}"
            groups.append(
                FmapGroup(
                    session=bold_runs[0].session if bold_runs else None,
                    fmap_id=gid,
                    reverse_path=rev_nf,
                    json=fwd_js,
                    forward_path=fwd_nf,
                    intended_runs=_resolve_intended(fwd_js, gid, bold_runs),
                )
            )
            continue
        for d, (nf, js) in picks.items():
            # Unpaired: the dir carries real information (which polarity this is),
            # so it goes back into the id — unless a task already names the group.
            gid = tag if tag and not tag.startswith("-") else f"{d or 'x'}{tag}"
            groups.append(
                FmapGroup(
                    session=bold_runs[0].session if bold_runs else None,
                    fmap_id=gid,
                    reverse_path=nf,
                    json=js,
                    intended_runs=_resolve_intended(js, gid, bold_runs),
                )
            )

    # Keep only (task, run) pairs that are actually in scope. Because the pair
    # carries the task, this drops an out-of-scope fmap (e.g. a task-floc fmap
    # when only -task primary was scanned) AND is immune to run-number collisions
    # across tasks (rest/run-01 vs skilled/run-01) — no task_tags bookkeeping.
    present = {(r.task, r.run) for r in bold_runs}
    for g in groups:
        g.intended_runs = [p for p in g.intended_runs if p in present]
    served = [g for g in groups if g.intended_runs]
    if served:
        return served
    # No explicit assignment (no IntendedFor, not task-tagged). Assign by
    # AcquisitionTime when available (conventional multi-fmap sessions), else the
    # single-geometry fallback.
    if groups and bold_runs and _assign_by_time(groups, bold_runs):
        return [g for g in groups if g.intended_runs]
    if len(groups) == 1 and bold_runs:
        groups[0].intended_runs = [(r.task, r.run) for r in bold_runs]
        return groups
    return served


def _pick_form(slot: dict[str, tuple[Path, dict]]) -> tuple[Path, dict] | None:
    """Representative image for one fieldmap acquisition: SBRef > EPI > BOLD."""
    for key in ("sbref", "epi", "bold"):
        if key in slot:
            return slot[key]
    return None


def _pe_axis_sign(js: dict) -> tuple[str, int] | None:
    """``PhaseEncodingDirection`` as (axis letter, +1/-1), or None if absent."""
    pe = js.get("PhaseEncodingDirection")
    if not pe:
        return None
    pe = str(pe)
    return pe[0], (-1 if pe.endswith("-") else 1)


# Conventional opposite ``dir`` labels, used only to *recognise* a pair when the
# sidecars omit PhaseEncodingDirection. These names are convention, not BIDS.
_OPPOSITE_DIRS = {"ap": "pa", "pa": "ap", "lr": "rl", "rl": "lr", "is": "si", "si": "is"}


def _pick_pe_pair(
    picks: dict[str, tuple[Path, dict]],
    bold_runs: list[BoldRun],
    fmap_pe_dir: str | None = None,
) -> tuple[str, str] | None:
    """Find the opposite-PE pair among one tag's ``dir`` variants, as
    ``(forward_dir, reverse_dir)``; None when there is no usable pair.

    Forward is the polarity that matches the BOLD runs. This is not cosmetic:
    ffs_blipflip estimates the field in the *blip_up* image's distorted space, so
    naming the wrong side forward would apply the correction backwards and double
    the distortion instead of removing it. Three ways to know, in order:

    1. ``PhaseEncodingDirection`` on the fmap sidecars vs the runs' — the real answer.
    2. the runs' own ``dir`` entity (present on some datasets that omit the PE field).
    3. ``fmap_pe_dir`` (``-fmap_pe_dir``), the user naming the runs' polarity.

    quirk: some conversions write only ``InPlanePhaseEncodingDirectionDICOM``
    (axis, no sign) on *both* the fmaps and the runs — nothing there identifies a
    polarity. Such a pair is recognisable by its ``dir`` labels but not
    orientable, so it stays unmerged (one group per polarity, the pre-pairing
    behaviour) and ``pair_undetermined`` flags it for the caller to warn about.
    """
    if len(picks) < 2:
        return None
    axes = {d: _pe_axis_sign(js) for d, (_nf, js) in picks.items()}
    run_pe = next((pe for r in bold_runs if (pe := _pe_axis_sign(r.json)) is not None), None)
    if run_pe is not None and all(v is not None for v in axes.values()):
        fwd = [d for d, v in axes.items() if v == run_pe]
        rev = [d for d, v in axes.items() if v[0] == run_pe[0] and v[1] != run_pe[1]]
        return (fwd[0], rev[0]) if fwd and rev else None
    # No usable PE signs. Fall back to the dir labels, which only work as a pair.
    forward = fmap_pe_dir or next(
        (d for r in bold_runs if (d := parse_entities(r.mag_path.name).get("dir"))), None
    )
    if forward is None:
        return None
    fwd = [d for d in picks if d.lower() == forward.lower()]
    rev = [d for d in picks if d.lower() == _OPPOSITE_DIRS.get(forward.lower())]
    return (fwd[0], rev[0]) if fwd and rev else None


def pair_undetermined(session: Session) -> list[str]:
    """``dir`` labels that look like an opposite-PE pair the scanner could not
    orient — the sidecars carry no polarity and nothing named the runs'. Each such
    fieldmap fell back to borrowing a data run as its forward image, which works
    but wastes the acquired mate; ``-fmap_pe_dir`` resolves it.
    """
    labels = [f.fmap_id for f in session.fmaps if f.forward_path is None]
    bare = {lab.split("-")[0].lower() for lab in labels}
    return sorted(lab for lab in labels if _OPPOSITE_DIRS.get(lab.split("-")[0].lower()) in bare)


def _assign_by_time(groups: list[FmapGroup], bold_runs: list[BoldRun]) -> bool:
    """Assign each run to a fieldmap by AcquisitionTime, in place. Returns True if
    the time data was sufficient to assign anything.

    Fieldmaps are normally acquired *before* the runs they cover, so a run takes
    the most-recent fieldmap acquired at-or-before it; a run acquired before any
    fieldmap falls back to the nearest fieldmap in time.
    """
    timed = [(g, g.acq_time) for g in groups if g.acq_time is not None]
    if not timed or all(r.acq_time is None for r in bold_runs):
        return False
    for g in groups:
        g.intended_runs = []
    assigned = False
    for r in bold_runs:
        rt = r.acq_time
        if rt is None:
            continue
        preceding = [(g, t) for g, t in timed if t <= rt]
        g = (
            max(preceding, key=lambda gt: gt[1])[0]
            if preceding
            else min(timed, key=lambda gt: abs(gt[1] - rt))[0]
        )
        g.intended_runs.append((r.task, r.run))
        assigned = True
    return assigned


def _resolve_intended(fmap_json: dict, tag: str, bold_runs: list[BoldRun]) -> list[tuple[str, str]]:
    """Which BOLD runs does this fmap correct, as ``(task, run)`` pairs (run
    numbers repeat across tasks, so the pair is the unique run identity).

    ``IntendedFor`` when present; else task-match (the fmap tag equals a task
    name). No blanket "serves everything" here — that fallback is applied in
    ``_scan_fmaps`` only when a single candidate survives. quirk: ``IntendedFor``
    is present on only *some* fmaps in some datasets.
    """
    intended = fmap_json.get("IntendedFor")
    if intended:
        if isinstance(intended, str):
            intended = [intended]
        pairs: list[tuple[str, str]] = []
        for entry in intended:
            e = str(entry)
            # BIDS entities are alphanumeric (no separators) — don't let \w
            # swallow the trailing _part-mag etc. (would yield '01_part').
            rt = re.search(r"task-([A-Za-z0-9]+)", e)
            rr = re.search(r"run-([A-Za-z0-9]+)", e)
            if rr:
                pairs.append((rt.group(1) if rt else "", rr.group(1)))
        if pairs:
            return pairs
    # fmap tag names a task → that task's runs.
    return [(r.task, r.run) for r in bold_runs if r.task == tag]


def _pick_anat(sdir: Path) -> Path | None:
    anat_dir = sdir / "anat"
    if not anat_dir.is_dir():
        return None
    t1s = [
        p
        for p in sorted(anat_dir.glob("*_T1w.nii*"))
        # quirk: skip non-BIDS derivatives (denoised/border/ROI) that lack sidecars.
        if not any(tok in p.name for tok in ("_denoised", "_border", "_ROI"))
    ]
    if not t1s:
        return None
    for p in t1s:  # prefer the MP2RAGE UNI image
        if parse_entities(p.name).get("acq") == "uni":
            return p
    return t1s[0]
