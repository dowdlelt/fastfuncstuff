"""Constant-modulus cardiac detection: per-alias table and candidate acceptance rules.

Image-only through the decision; the recording is opened afterwards to score every
alias and the decision. Writes one JSON line per run.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.io.afni import load_nifti
from fastfuncstuff.processing import physio as p
from fastfuncstuff.processing import physio_cm as cm
from scripts.physio_phase_review.run_taskforce import metrics


def align(Ys, phase, device):
    return p.slice_alignment(p._voxel_fits(Ys, phase, device))


def alias_table(Ys, times, start, tr, device, rng=None):
    """Per alias: fundamental and harmonic alignment, best non-harmonic control. With
    ``rng`` the excitation times are deranged first (the null)."""
    if rng is not None:
        G = start.shape[1]
        perm = rng.permutation(G)
        while np.any(perm == np.arange(G)):
            perm = rng.permutation(G)
        start = start[:, perm]
    alias = cm.complete_aliases(times, start, tr, p.CARDIAC_BAND)
    phases = [sign * start + 2 * np.pi * step * times / tr for sign, step in alias]
    a = p.slice_alignments(Ys, phases + [2 * q for q in phases], device)
    return [
        dict(sign=sign, step=step, rate=p._phase_rate(times, q), a1=float(a[i]),
             a2=float(a[len(phases) + i]), phase=q)
        for i, ((sign, step), q) in enumerate(zip(alias, phases, strict=True))
    ]  # fmt: skip


def controls(Ys, phase, device):
    return float(p.slice_alignments(Ys, [m * phase for m in p.HARMONIC_CONTROLS], device).max())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shuffles", type=int, default=30)
    args = ap.parse_args()
    device = torch.device(args.device)
    rows = [ln.split() for ln in args.list.read_text().splitlines() if ln.strip()]
    if args.labels:
        rows = [r for r in rows if r[0] in args.labels]
    done = set()
    if args.output.exists():
        done = {json.loads(ln)["run"] for ln in args.output.read_text().splitlines() if ln}
    for row in rows:
        if row[0] in done or len(row) < 7 or not Path(row[1]).exists():
            continue
        t0 = time.monotonic()
        side = json.loads(Path(row[3]).read_text())
        st, tr = np.asarray(side["SliceTiming"], float), float(side["RepetitionTime"])
        img = load_nifti(row[1])
        mag = np.asanyarray(img.dataobj)
        nt = mag.shape[-1] - p.trailing_noise_volumes(mag.mean(axis=(0, 1, 2)))
        mag = mag[..., :nt]
        ph = np.asanyarray(load_nifti(row[2]).dataobj)[..., :nt] if row[2] != "-" else None
        traces = p.excitation_traces(mag, st, tr, phase=ph, device=device)
        del ph
        times = traces.times
        tm, xm = traces.series("phase" if traces.phase is not None else "magnitude", uniform=True)
        _, rf, _ = p.peak_rate(tm, xm, p.RESP_BAND)
        rp = np.interp(times.ravel(), tm, p.cycle_phase(tm, xm, rf)).reshape(times.shape)
        mean = mag.mean(-1)
        mask = mean > 0.25 * np.percentile(mean[mean > 0], 98)
        host, _ = p._cardiac_voxels(mag, st, tr, mask, rp, p.REFINE_HIGHPASS_HZ)
        Ys, _ = p._store_voxels(host, device)
        del host
        traces_cm = cm.constant_modulus_traces(cm.temporal_gram(mag, mask, device))
        del mag
        rng = np.random.default_rng(0)
        starts = []
        for i, (ratio, c) in enumerate(traces_cm):
            start = cm.excitation_phase(c, times, tr)
            table = alias_table(Ys, times, start, tr, device)
            for r in table:
                r["controls"] = controls(Ys, r["phase"], device)
            null = [alias_table(Ys, times, start, tr, device, rng) for _ in range(args.shuffles)]
            starts.append(dict(label=f"cm{i}", ratio=ratio, table=table, null=null))
        # ---- recording opened here ----
        peaks = np.loadtxt(row[4]) / float(row[6]) + float(row[5])
        rec = dict(
            run=row[0],
            tr=tr,
            G=int(times.shape[1]),
            nt=int(nt),
            ref_hr=float(60 / np.median(np.diff(peaks))),
            starts=[],
        )
        for s in starts:
            out = dict(label=s["label"], ratio=s["ratio"], aliases=[], null=[])
            for r in s["table"]:
                m = metrics(r["phase"], times, peaks)
                out["aliases"].append(
                    {k: v for k, v in r.items() if k != "phase"}
                    | dict(lock=m["lock"], wlock=m["window_lock_median"])
                )
            out["null"] = [[dict(a1=r["a1"], a2=r["a2"]) for r in t] for t in s["null"]]
            rec["starts"].append(out)
        rec["seconds"] = time.monotonic() - t0
        with args.output.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        best = max((a for s in rec["starts"] for a in s["aliases"]), key=lambda a: a["lock"])
        print(
            row[0],
            "ref",
            round(rec["ref_hr"], 1),
            "best-alias lock",
            round(best["lock"], 3),
            "at",
            round(best["rate"], 1),
            f"{rec['seconds']:.0f}s",
            flush=True,
        )
        del Ys
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
