"""CPU diagnostics for short-TR cardiac rejection and magnitude respiratory combs.

Research only: no acceptance rule is changed. Recordings enter only after blind
candidates have been constructed. Reuse cached excitation traces between probes.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from tqdm.auto import tqdm

from fastfuncstuff.cli_help import FfsArgumentParser
from fastfuncstuff.io.afni import load_nifti
from fastfuncstuff.processing import physio as p
from fastfuncstuff.processing import physio_cm as cm
from scripts.physio_phase_review.resp_probe import belt, lock
from scripts.physio_phase_review.run_taskforce import metrics

CPU = torch.device("cpu")


def respiration(bold, output):
    side = json.loads(Path(str(bold).replace(".nii.gz", ".json")).read_text())
    tr = float(side["RepetitionTime"])
    cache = output.parent / (bold.name.replace(".nii.gz", "") + ".traces.npz")
    if cache.exists():
        z = np.load(cache)
        times, x = z["times"], z["x"]
    else:
        mag = np.asanyarray(load_nifti(str(bold)).dataobj)
        nt = mag.shape[-1] - p.trailing_noise_volumes(mag.mean((0, 1, 2)))
        traces = p.excitation_traces(mag[..., :nt], np.array(side["SliceTiming"]), tr, device=CPU)
        times, x = traces.times, traces.magnitude
        np.savez_compressed(cache, times=times, x=x)
    t, stitched = p._uniform(times.ravel(), x.ravel())
    variants = {"current": stitched, "notch": p.notch_tr_harmonics(t, stitched, tr, p.RESP_BAND)}
    # Each group has a separate gain. Interpolate its slow volume-rate trace
    # before averaging, rather than multiplying the signal by a periodic gain.
    aligned = np.stack([np.interp(t, times[:, g], x[:, g]) for g in range(x.shape[1])], 1)
    variants["aligned_mean"] = aligned.mean(1)
    # A sign-changing respiratory field can cancel in the unit-weight mean.
    sos = signal.butter(3, p.RESP_BAND, btype="bandpass", fs=p._fs(t), output="sos")
    band = signal.sosfiltfilt(sos, aligned, axis=0)
    _, vectors = np.linalg.eigh(band.T @ band)
    variants["aligned_pc"] = band @ vectors[:, -1]
    estimates = {}
    for name, y in variants.items():
        _, rf, _ = p.peak_rate(t, y, p.RESP_BAND)
        estimates[name] = (float(60 * np.median(rf)), p.cycle_phase(t, y, rf))
    # Recording is scoring only. No choice above uses the belt.
    pj = str(bold).replace("_bold.nii.gz", "_physio.json")
    rate, phase = belt(pj, t)
    result = dict(
        run=bold.name,
        tr=tr,
        volume_nyquist=0.5 / tr,
        belt_rate=rate,
        variants={k: dict(rate=r, lock=lock(q, phase)) for k, (r, q) in estimates.items()},
    )
    return result


def cardiac(row):
    label, bold, _, sidecar, peaks_path, offset, fs = row
    side = json.loads(Path(sidecar).read_text())
    st, tr = np.array(side["SliceTiming"]), float(side["RepetitionTime"])
    mag = np.asanyarray(load_nifti(bold).dataobj)
    nt = mag.shape[-1] - p.trailing_noise_volumes(mag.mean((0, 1, 2)))
    mag = mag[..., :nt]
    times = np.arange(nt)[:, None] * tr + p.excitation_groups(st)[1]
    mean = mag.mean(-1)
    mask = mean > 0.25 * np.percentile(mean[mean > 0], 98)
    traces = p.excitation_traces(mag, st, tr, mask=mask, device=CPU)
    t, x = traces.series("magnitude", uniform=True)
    _, rf, _ = p.peak_rate(t, x, p.RESP_BAND)
    rp = np.interp(times.ravel(), t, p.cycle_phase(t, x, rf)).reshape(times.shape)
    Ys, _ = p._cardiac_voxels(mag, st, tr, mask, rp, p.REFINE_HIGHPASS_HZ)
    Ys, _ = p._store_voxels(Ys, CPU)
    starts = cm.constant_modulus_traces(cm.temporal_gram(mag, mask, CPU))
    candidates = []
    for ratio, c in starts:
        phase = cm.excitation_phase(c, times, tr)
        for q, _a1, _a2 in cm._alias_rows(Ys, times, phase, tr, CPU, p.CARDIAC_BAND):
            candidates.append((f"cm{len(candidates)}", q, ratio))
    del mag
    # Recording is scoring only; also test the true clock as a recoverability oracle.
    peaks = np.loadtxt(peaks_path) / float(fs) + float(offset)
    recorded = p.beat_phase(peaks, times.ravel()).reshape(times.shape)
    candidates.append(("recorded_oracle", recorded, None))
    G = times.shape[1]
    perms = [q for q in itertools.permutations(range(G)) if all(i != v for i, v in enumerate(q))]
    out = dict(
        run=label,
        tr=tr,
        G=G,
        nt=nt,
        volume_nyquist=0.5 / tr,
        slice_span=float(np.ptp(times[0])),
        derangements=len(perms),
        candidates=[],
    )
    for name, phase, ratio in candidates:
        a1, a2 = p.slice_alignments(Ys, [phase, 2 * phase], CPU)
        shuffled = [phase[:, q] for q in perms]
        v = p.slice_alignments(Ys, shuffled + [2 * q for q in shuffled], CPU)
        null = v[: len(perms)] + v[len(perms) :]
        h, hnull = p._harmonic_alignment(Ys, phase, CPU)
        out["candidates"].append(
            dict(
                name=name,
                ratio=ratio,
                **metrics(phase, times, peaks),
                a1=float(a1),
                a2=float(a2),
                harmonic=float(h),
                harmonic_null=float(hnull),
                exact_null_min=float(null.min()),
                exact_null_max=float(null.max()),
                derangement_tail_fraction=float((1 + (null >= a1 + a2).sum()) / (len(null) + 1)),
            )
        )
    return out


def main():
    ap = FfsArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["cardiac", "respiration"], required=True)
    ap.add_argument("--list", type=Path)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "cardiac":
        rows = [s.split() for s in args.list.read_text().splitlines() if "on.ds004645." in s]
        jobs = [(r[0], lambda r=r: cardiac(r)) for r in rows]
    else:
        jobs = [
            (b.name, lambda b=b: respiration(b, args.output))
            for b in sorted(args.root.glob("sub-*/**/func/*_bold.nii.gz"))
        ]
    for label, job in tqdm(jobs, desc=args.mode, leave=True, disable=len(jobs) <= 1):
        result = job()
        with args.output.open("a") as f:
            f.write(json.dumps(result) + "\n")
        print(label, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
