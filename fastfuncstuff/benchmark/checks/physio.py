"""physio: ffs_physio's data-derived cycle phases against the recorded physiology.

No reference tool: the judge is truth, the run's own pulse (and belt) recording in BIDS
``_physio.tsv.gz`` (``ffs_util_physio2bids`` writes them from CMRR logs). The recording
plays no part in the run; ffs_physio sees only the raw magnitude (and phase).

Scored as in docs/physio.md, *Cardiac detection, against recordings*: the cardiac phase
ffs evaluates at every slice's acquisition time (slibase ``card.c1/s1``) against the
RETROICOR phase of the recorded pulse at the same times. Phase-locking value (PLV)
= |mean exp(i(phi_ffs - phi_rec))|. **right** = found and PLV > 0.5, **wrong** = found
and not locked, otherwise not found. Respiration is scored the same way (no verdict).
"""

from __future__ import annotations

import gzip
import json

import numpy as np

from .base import Check, UnitResult, ffs_version, timed

PLV_RIGHT = 0.5


def _recording(tsv: dict, meta: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """kind -> (times from volume 0, samples) for each cardiac / respiratory column."""
    fs = float(meta["SamplingFrequency"])
    t0 = float(meta.get("StartTime", 0.0))
    out = {}
    for name, col in tsv.items():
        low = name.lower()
        kind = (
            "cardiac"
            if any(k in low for k in ("cardiac", "pulse", "ppg", "ecg", "ekg", "puls"))
            else "respiratory"
            if "resp" in low or "belt" in low  # ds004499: belt, EKG
            else None
        )
        if kind and kind not in out:
            out[kind] = (t0 + np.arange(col.size) / fs, col)
    return out


def _read_physio(path, meta: dict) -> dict[str, np.ndarray]:
    cols = meta.get("Columns") or []
    with gzip.open(path, "rt") as f:
        data = np.loadtxt(f, ndmin=2, dtype=float)
    return {str(c): data[:, i] for i, c in enumerate(cols) if i < data.shape[1]}


def _slibase_phase(path, kind: str) -> np.ndarray:
    """(nt, nslices) phase of the first-order ``kind`` (card|resp) RETROICOR pair."""
    labels: list[str] = []
    for ln in open(path):
        if ln.startswith("# ColumnLabels"):
            labels = [s.strip() for s in ln.split('"')[1].split(";")]
            break
    X = np.loadtxt(path, comments="#", ndmin=2)
    c = [i for i, lab in enumerate(labels) if lab.endswith(f".{kind}.c1")]
    s = [i for i, lab in enumerate(labels) if lab.endswith(f".{kind}.s1")]
    return np.arctan2(X[:, s], X[:, c])


def plv(phi_a: np.ndarray, phi_b: np.ndarray) -> float:
    return float(np.abs(np.mean(np.exp(1j * (phi_a - phi_b)))))


class PhysioCheck(Check):
    name = "physio"
    description = "Physio from the data: ffs_physio cycle phases vs the recorded pulse/belt"
    needs = [["epi_2d", "slice_timing", "physio_cardiac"]]
    sampling = "breadth"
    reference = "recorded physiology"

    def unit_inputs(self, files, index):
        mags = [f for f in files if f.ent.get("part") != "phase" and f.suffix != "phase"]
        first = sorted(mags, key=lambda f: int(f.ent.get("echo", "1")))[0]
        phase = [
            f
            for f in files
            if (f.ent.get("part") == "phase" or f.suffix == "phase")
            and f.ent.get("echo", "1") == first.ent.get("echo", "1")
        ]
        own = {k: v for k, v in first.entities if k not in {"echo", "part"}}
        physio = [
            p
            for p in index.names
            if p.suffix == "physio"
            and p.ext == ".tsv.gz"
            and all(p.ent.get(k) == v for k, v in own.items())
        ]
        rels = [first.rel] + [m.rel for m in index.matching(first, ".json")]
        rels += [p.rel for p in phase[:1]]
        for p in physio:
            rels.append(p.rel)
            rels += [m.rel for m in index.matching(p, ".json")]
        return list(dict.fromkeys(rels))

    def run(self, key, inputs, work, device):
        imgs = [r for r in inputs if r.endswith((".nii", ".nii.gz")) and "_physio" not in r]
        mag = next(r for r in imgs if "part-phase" not in r and not r.endswith("_phase.nii.gz"))
        phase = next((r for r in imgs if r != mag), None)
        meta: dict = {}
        for r, p in inputs.items():
            if r.endswith(".json") and "_physio" not in r:
                meta.update(json.loads(p.read_text()))
        st, tr = meta.get("SliceTiming"), meta.get("RepetitionTime")
        if not st or not tr:
            raise ValueError("SliceTiming and RepetitionTime are needed")
        st = np.asarray(st, dtype=float)
        tr = float(tr)

        rec: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for r, p in inputs.items():
            if r.endswith("_physio.tsv.gz"):
                own = r.removesuffix(".tsv.gz") + ".json"
                side = [j for j in inputs if j.endswith("_physio.json")]
                pm = json.loads(inputs[own if own in inputs else side[0]].read_text())
                for kind, v in _recording(_read_physio(p, pm), pm).items():
                    rec.setdefault(kind, v)
        if "cardiac" not in rec:
            raise ValueError("no cardiac column in the recordings")

        work.mkdir(parents=True, exist_ok=True)
        log = work / "commands.log"
        tp = work / "tpattern.txt"
        tp.write_text("\n".join(f"{t:.6f}" for t in st) + "\n")
        prefix = work / "ffs" / "run"
        cmd = ["ffs_physio", "-input", str(inputs[mag])]
        if phase:
            cmd += ["-phase", str(inputs[phase])]
        cmd += ["-tpattern", str(tp), "-TR", f"{tr:g}", "-prefix", str(prefix), "-no_qc"]
        res = UnitResult()
        t = timed([*cmd, "-device", device], work, log, timeout_s=self.timeout_s)
        res.timings.append(("ffs", device.split(",")[0], t, None))

        from ...processing.physio import CARDIAC_BAND, RESP_BAND, beat_phase, find_beats

        summary = json.loads((work / "ffs" / "run.json").read_text())
        cw = summary.get("cardiac_weighted") or summary.get("cardiac") or {}
        found = bool(cw.get("found", False))
        m: dict[str, float | None] = {"card_found": float(found)}
        for kind, band, short in (("cardiac", CARDIAC_BAND, "card"), ("respiratory", RESP_BAND, "resp")):  # fmt: skip
            if kind not in rec:  # respiration scored from magnitude too: docs call it unreliable
                continue
            phi_ffs = _slibase_phase(f"{prefix}.slibase.1D", short)
            nt = phi_ffs.shape[0]
            times = np.arange(nt)[:, None] * tr + st[None, : phi_ffs.shape[1]]
            t_rec, x_rec = rec[kind]
            keep = (t_rec >= times.min() - 10) & (t_rec <= times.max() + 10)
            peaks = find_beats(t_rec[keep], x_rec[keep], band)
            if peaks.size < 4:
                m[f"{short}_plv"] = None
                continue
            phi_rec = beat_phase(peaks, times.ravel()).reshape(times.shape)
            m[f"{short}_plv"] = plv(phi_ffs, phi_rec)
            m[f"{short}_rate_rec_per_min"] = float(60.0 / np.median(np.diff(peaks)))
        hr_ffs = cw.get("heart_rate_per_min_median", summary.get("heart_rate_per_min_median"))
        m["card_rate_ffs_per_min"] = hr_ffs
        m["resp_rate_ffs_per_min"] = summary.get("breathing_rate_per_min_median")
        if m.get("card_rate_rec_per_min") and hr_ffs:
            m["card_rate_abs_err_per_min"] = abs(float(hr_ffs) - m["card_rate_rec_per_min"])
        p = m.get("card_plv")
        m["card_right"] = float(found and p is not None and p > PLV_RIGHT)
        m["card_wrong"] = float(found and (p is None or p <= PLV_RIGHT))
        res.metrics["truth"] = m

        for f in sorted((work / "ffs").glob("run.*")):
            res.artifacts.append((f, "regenerable" if f.suffix == ".gz" else "keep"))
        res.artifacts += [(tp, "keep"), (log, "keep")]
        res.provenance["ffs"] = {"tool": "ffs_physio", **ffs_version(), "device": device,
                                 "phase_given": bool(phase),
                                 "sources": summary.get("sources")}  # fmt: skip
        res.provenance["truth"] = {
            "recordings": sorted(rec),
            "scoring": "PLV of slibase c1/s1 phase vs beat_phase(find_beats(recording))",
        }
        return res
