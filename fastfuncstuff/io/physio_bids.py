"""Recorded physiology to BIDS ``_physio.tsv.gz``: CMRR (Siemens) logs.

CMRR's multiband sequence writes ``Physio_<date>_<uuid>_{Info,PULS,RESP,ECG}.log``.
Every sample carries its own tick, and a tick is 2.5 ms **since midnight**, so the Info
log's first acquisition tick times 0.0025 equals the BOLD sidecar's ``AcquisitionTime``.
Runs are matched to logs by that clock, never by folder or run names, which drift.

Each channel goes to its own file (``recording-cardiac`` / ``recording-respiratory``),
because PULS and RESP are sampled at different rates and BIDS wants one uniform rate
per file. ``StartTime`` is the first sample relative to the first volume (negative when
the recording started earlier, as it usually does).
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

TICK_S = 0.0025
_CHANNELS = {"PULS": "cardiac", "RESP": "respiratory", "ECG1": "cardiac", "ECG": "cardiac"}


def read_cmrr_log(path) -> tuple[np.ndarray, np.ndarray]:
    """(times in s, values) from a CMRR physio log (RESP / PULS / ECG channel 1),
    placed by each sample's own 2.5 ms tick -- gaps cost nothing."""
    t, v = [], []
    for line in open(path):
        p = line.split()
        if len(p) >= 3 and p[0].isdigit() and p[1] in _CHANNELS:
            t.append(int(p[0]))
            v.append(float(p[2]))
    return np.asarray(t, dtype=float) * TICK_S, np.asarray(v, dtype=float)


def cmrr_volume_zero(info_path) -> float:
    """Start of the first acquisition (s, same clock as the logs) from a CMRR Info log."""
    first = None
    for line in open(info_path):
        p = line.split()
        if len(p) == 5 and all(x.isdigit() for x in p):
            t = int(p[2]) * TICK_S
            first = t if first is None else min(first, t)
    if first is None:
        raise ValueError(f"no acquisition rows in {info_path}")
    return first


def cmrr_volume_starts(info_path) -> np.ndarray:
    """First-slice start of every logged volume (s since midnight), in volume order."""
    first: dict[int, int] = {}
    for line in open(info_path):
        p = line.split()
        if len(p) == 5 and all(x.isdigit() for x in p):
            v, tick = int(p[0]), int(p[2])
            if v not in first or tick < first[v]:
                first[v] = tick
    return np.array([first[v] * TICK_S for v in sorted(first)])


def cmrr_num_volumes(info_path) -> int | None:
    for line in open(info_path):
        m = re.match(r"\s*NumVolumes\s*=\s*(\d+)", line)
        if m:
            return int(m.group(1))
    return None


@dataclass
class CmrrRecording:
    info: Path
    channels: dict[str, Path]  # 'PULS' / 'RESP' / 'ECG1' -> log
    t0: float  # first image volume, seconds since midnight (set per match)
    n_vols: int | None
    vol_starts: np.ndarray  # every logged volume, dummies included when the sequence logs them
    skipped: int = 0  # logged volumes before the image's first (dummies, discarded volumes)


def find_cmrr_recordings(root: Path) -> list[CmrrRecording]:
    """Every ``*_Info.log`` under ``root`` with its sibling channel logs (same UUID)."""
    out = []
    for info in sorted(Path(root).rglob("*_Info.log")):
        stem = info.name[: -len("_Info.log")]
        chans = {}
        for ch in _CHANNELS:
            p = info.with_name(f"{stem}_{ch}.log")
            if p.exists():
                chans[ch] = p
        if not chans:
            continue
        starts = cmrr_volume_starts(info)
        if not len(starts):
            continue
        out.append(CmrrRecording(info, chans, float(starts[0]), cmrr_num_volumes(info), starts))
    return out


def acquisition_seconds(sidecar: dict) -> float | None:
    """BIDS/dcm2niix ``AcquisitionTime`` ('HH:MM:SS.ffffff') as seconds since midnight."""
    at = sidecar.get("AcquisitionTime")
    if not isinstance(at, str):
        return None
    h, m, s = at.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def match_runs(
    recordings: list[CmrrRecording], bolds: list[tuple[Path, dict, int]], tol_s: float = 0.5
) -> list[tuple[Path, CmrrRecording | None, str]]:
    """Pair each BOLD (path, sidecar, n_vols) with the logged volume whose start is its
    AcquisitionTime (|dt| <= tol_s). Logged volumes before it (dummies, volumes dropped at
    conversion) are skipped; the volumes from there on must number exactly the image's."""
    out = []
    for path, meta, nv in bolds:
        t = acquisition_seconds(meta)
        if t is None:
            out.append((path, None, "no AcquisitionTime in sidecar"))
            continue
        best, k, dt = None, 0, np.inf
        for r in recordings:
            i = int(np.argmin(np.abs(r.vol_starts - t)))
            d = float(r.vol_starts[i] - t)
            if abs(d) < abs(dt):
                best, k, dt = r, i, d
        if best is None or abs(dt) > tol_s:
            near = "" if best is None else f" (nearest {dt:+.1f} s)"
            out.append((path, None, f"no logged volume at {meta['AcquisitionTime']}{near}"))
            continue
        remaining = len(best.vol_starts) - k
        if remaining != nv:
            out.append((path, None, f"{remaining} logged volumes from the match vs {nv} in image"))
            continue
        matched = replace(best, t0=float(best.vol_starts[k]), skipped=k)
        note = f", {k} leading logged volumes skipped" if k else ""
        out.append((path, matched, f"dt {dt:+.3f} s{note}"))
    return out


def _uniform(t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, float]:
    """Resample onto the channel's own median spacing (CMRR drops the odd sample)."""
    dt = float(np.median(np.diff(t)))
    grid = t[0] + dt * np.arange(int(round((t[-1] - t[0]) / dt)) + 1)
    return np.interp(grid, t, v), 1.0 / dt


def physio_stem(bold: Path) -> str:
    """BIDS stem for the run's physio: the BOLD name without echo/part/suffix."""
    name = bold.name.split(".", 1)[0]
    parts = [p for p in name.split("_")[:-1] if not p.startswith(("echo-", "part-"))]
    return "_".join(parts)


def write_bids_physio(rec: CmrrRecording, bold: Path, overwrite: bool = False) -> list[Path]:
    """Write one ``recording-<kind>_physio.tsv.gz`` + ``.json`` per channel next to ``bold``."""
    written = []
    stem = physio_stem(bold)
    used: set[str] = set()
    for ch, log in rec.channels.items():
        kind = _CHANNELS[ch]
        if kind in used:  # PULS and ECG both cardiac: keep the first (PULS)
            continue
        used.add(kind)
        t, v = read_cmrr_log(log)
        if len(t) < 2:
            continue
        vals, fs = _uniform(t, v)
        tsv = bold.with_name(f"{stem}_recording-{kind}_physio.tsv.gz")
        side = tsv.with_name(tsv.name.replace(".tsv.gz", ".json"))
        if not overwrite and (tsv.exists() or side.exists()):
            raise FileExistsError(tsv)
        with gzip.open(tsv, "wt") as f:
            f.write("\n".join(f"{x:g}" for x in vals) + "\n")
        side.write_text(
            json.dumps(
                {
                    "SamplingFrequency": round(fs, 6),
                    "StartTime": round(float(t[0] - rec.t0), 4),
                    "Columns": [kind],
                    "Source": f"CMRR {log.name}",
                },
                indent=2,
            )
        )
        written += [tsv, side]
    return written
