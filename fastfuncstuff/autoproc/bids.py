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
# Data model
# ---------------------------------------------------------------------------


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
        # quirk: TaskName is absent from every func sidecar in the MindsEye
        # data; derive it from the entity so downstream (BIDS validator, GLM)
        # still has one.
        return self.json.get("TaskName") or self.task

    @property
    def rep(self) -> Path:
        """Representative 3D image for registration: SBRef if present, else the
        4D BOLD (the emitter takes a mean/base from it)."""
        return self.sbref_path or self.mag_path


@dataclass
class FmapGroup:
    """A reverse-PE fieldmap and the runs it corrects.

    ``reverse_path`` is the blip-down (opposite PE) image; the forward (blip-up)
    reference is the intended runs' SBRef/mean, filled in by the emitter. Runs
    are associated via ``IntendedFor`` when present, else by task+session.
    """

    session: str | None
    fmap_id: str  # human tag: task name, ``acq`` value, or ``dir-run`` fallback
    reverse_path: Path
    json: dict
    intended_runs: list[str] = field(default_factory=list)  # BIDS run ids

    @property
    def readout(self) -> float | None:
        v = self.json.get("TotalReadoutTime")
        return float(v) if v is not None else None

    @property
    def pe_dir(self) -> str | None:
        return self.json.get("PhaseEncodingDirection")


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
) -> Subject:
    """Scan one subject into a :class:`Subject` tree.

    ``sessions``/``tasks`` optionally restrict the scan (values may be bare
    labels or full ``ses-``/``task-`` forms). Returns sessions and runs in
    directory-sorted order; run numbers are taken as-found (never assumed
    contiguous).
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
        sess = _scan_session(sdir, bids_root, sub_label, ses_label, want_task)
        if sess.bold_runs:
            out.sessions.append(sess)
    return out


def _scan_session(
    sdir: Path,
    bids_root: Path,
    sub_label: str,
    ses_label: str | None,
    want_task: set[str] | None,
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
        sess.fmaps = _scan_fmaps(fmap_dir, bids_root, sess.bold_runs)

    # ---- anat: prefer acq-uni T1w (MP2RAGE), else first T1w / MPRAGE ----
    sess.anat = _pick_anat(sdir)
    return sess


def _scan_fmaps(fmap_dir: Path, bids_root: Path, bold_runs: list[BoldRun]) -> list[FmapGroup]:
    """Build FmapGroups from the reverse-PE images in ``fmap/``.

    Preference order for the representative reverse image within a group:
    SBRef > EPI > BOLD (SBRef is cleanest). Grouping key is the ``task`` entity
    when present (MindsEye task-tagged fmaps), else ``dir``+``run`` (conventional
    epi). ``acq`` is *not* a group key: ``acq-bold``/``acq-sbref`` are two forms
    of one fieldmap, not two fieldmaps.
    """
    candidates: dict[str, dict] = {}
    for nf in sorted(fmap_dir.glob("*.nii*")):
        if not _NIFTI_RE.search(nf.name):
            continue
        ents = parse_entities(nf.name)
        if _is_phase(nf, ents, bids_root):
            continue  # phase fmap unused this milestone
        suffix = parse_suffix(nf.name)
        tag = ents.get("task") or f"{ents.get('dir', 'x')}{ents.get('run', '')}"
        # Form within the group: acq (bold/sbref) refines the plain suffix so the
        # SBRef form wins the preference below even for conventional epi fmaps.
        form = ents.get("acq") or suffix or "epi"
        slot = candidates.setdefault(tag, {})
        slot[form] = (nf, load_sidecar(nf, bids_root))

    groups: list[FmapGroup] = []
    for tag, slot in candidates.items():
        for key in ("sbref", "epi", "bold"):
            if key in slot:
                nf, js = slot[key]
                break
        else:
            continue
        groups.append(
            FmapGroup(
                session=bold_runs[0].session if bold_runs else None,
                fmap_id=tag,
                reverse_path=nf,
                json=js,
                intended_runs=_resolve_intended(js, tag, bold_runs),
            )
        )

    # Restrict each group's intended runs to what's actually in scope, then drop
    # groups that serve nothing (e.g. a task-floc fmap when only -task primary was
    # requested). If that leaves no group but we did find exactly one candidate,
    # fall back to "it serves every in-scope run" (single-geometry session).
    in_scope = {r.run for r in bold_runs}
    for g in groups:
        g.intended_runs = [r for r in g.intended_runs if r in in_scope]
    served = [g for g in groups if g.intended_runs]
    if not served and len(groups) == 1 and bold_runs:
        groups[0].intended_runs = [r.run for r in bold_runs]
        return groups
    return served


def _resolve_intended(fmap_json: dict, tag: str, bold_runs: list[BoldRun]) -> list[str]:
    """Which BOLD runs does this fmap correct (before in-scope filtering)?

    ``IntendedFor`` when present; else task-match (the fmap tag equals a task
    name). No blanket "serves everything" here — that fallback is applied in
    ``_scan_fmaps`` only when a single candidate survives. quirk: ``IntendedFor``
    is present on only *some* fmaps in the MindsEye data.
    """
    intended = fmap_json.get("IntendedFor")
    if intended:
        if isinstance(intended, str):
            intended = [intended]
        runs: list[str] = []
        for entry in intended:
            m = re.search(r"run-(\w+)", str(entry))
            if m:
                runs.append(m.group(1))
        if runs:
            return runs
    # fmap tag names a task → that task's runs.
    return [r.run for r in bold_runs if r.task == tag]


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
