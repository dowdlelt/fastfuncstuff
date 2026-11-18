"""Reproducible CPU pilot against TASKFORCE reviewed peaks (evaluation only).

Run from the repository root with its installed Python environment:
  python -m scripts.physio_phase_review.run_taskforce --output /tmp/physio-phase

No production behaviour is changed. Small selected-voxel caches allow subsequent
phase experiments without rereading the images. Baseline initialisation uses all
voxels; the spatial validation tests improvement conditional on that shared start,
not independent detection of a heartbeat.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from fastfuncstuff.io.afni import load_nifti
from fastfuncstuff.processing import physio as p
from scripts.physio_phase_review.direct_phase import fit_phase, prediction_score, select_voxels

ROOT = Path("/mnt/belegost/Data/nii_data/PROJECT_TASKFORCE/subjects/sub-pilot01")
DER = Path("/mnt/belegost/Projects/TASKFORCE/derivatives/sub-pilot01")
PILOT = [(1, "mvpsA", 1), (1, "rest", 1), (2, "mvpsA", 2), (3, "rest", 1), (4, "mvpsA", 2)]
CHECK = [(5, "mvpsB", 1), (6, "mvpsB", 4), (7, "mdtb", 7), (8, "mvpsB", 1), (10, "mdtb", 1)]
CPU = torch.device("cpu")


def prepare(ses, task, run, ica=True):
    session = f"ses-{ses:02d}"
    stem = ROOT / session / "func" / f"sub-pilot01_{session}_task-{task}_run-{run:02d}"
    with Path(f"{stem}_part-mag_bold.json").open() as f:
        side = json.load(f)
    st, tr = np.asarray(side["SliceTiming"]), float(side["RepetitionTime"])
    mag = np.asanyarray(load_nifti(f"{stem}_part-mag_bold.nii.gz").dataobj)
    phase = np.asanyarray(load_nifti(f"{stem}_part-phase_bold.nii.gz").dataobj)
    nt = mag.shape[-1] - p.trailing_noise_volumes(mag.mean(axis=(0, 1, 2)))
    mag, phase = mag[..., :nt], phase[..., :nt]
    traces = p.excitation_traces(mag, st, tr, phase=phase, device=CPU)
    del phase
    print(f"{session} {task} {run}: traces ready; running current cardiac detector", flush=True)
    ref = p.refine_cardiac(mag, traces, st, tr, device=CPU, ica=ica)
    print(
        f"{session} {task} {run}: current detector finished; preparing selected voxels", flush=True
    )
    initial = next(
        ph for tried, ph in zip(ref.tried, ref.tried_phase, strict=True) if tried[0] == ref.start
    )
    tm, xr = traces.series("phase", uniform=True)
    _, rf, _ = p.peak_rate(tm, xr, p.RESP_BAND)
    rp = np.interp(traces.times.ravel(), tm, p.cycle_phase(tm, xr, rf)).reshape(traces.times.shape)
    mean = mag.mean(axis=-1)
    mask = mean > 0.25 * np.percentile(mean[mean > 0], 98)
    ys, coordinates = p._cardiac_voxels(mag, st, tr, mask, rp, p.REFINE_HIGHPASS_HZ)
    del mag
    train, validation = [], []
    for y, xyz in zip(ys, coordinates, strict=True):
        split = xyz[:, 0] % 2 == 0
        train.append(y[split].numpy())
        validation.append(y[~split].numpy())
    train, conditioning = select_voxels(train, initial)
    validation, _ = select_voxels(validation, initial)
    record = dict(
        found=ref.found,
        rate=ref.heart_rate_per_min,
        alignment=ref.alignment,
        null=ref.alignment_null,
        harmonic=ref.harmonic,
        harmonic_null=ref.harmonic_null,
        start=ref.start,
        conditioning=conditioning,
        ica=ica,
    )
    data = dict(times=traces.times, initial=initial, tr=np.array(tr), baseline=json.dumps(record))
    for g, (a, b) in enumerate(zip(train, validation, strict=True)):
        data[f"train_{g}"] = a
        data[f"validation_{g}"] = b
    return data


def reference(ses, task, run):
    prefix = DER / f"ses-{ses:02d}" / "all_physio" / f"task-{task}_run{run:02d}"
    with Path(f"{prefix}_info.json").open() as f:
        info = json.load(f)
    return np.loadtxt(f"{prefix}_card_peaks_00.1D") / info["freq"] + info["start_time"]


def metrics(phase, times, peaks):
    truth = p.beat_phase(peaks, times.ravel()).reshape(times.shape)
    valid = (times > max(peaks[0], times.min() + 10)) & (times < min(peaks[-1], times.max() - 10))
    error = (phase - truth)[valid]
    windows = []
    for start in np.arange(times.min() + 10, times.max() - 30, 30):
        keep = valid & (times >= start) & (times < start + 30)
        windows.append(abs(np.mean(np.exp(1j * (phase - truth)[keep]))))
    rate = np.gradient(phase.ravel(), times.ravel()) / (2 * np.pi)
    return dict(
        lock=float(abs(np.mean(np.exp(1j * error)))),
        harmonic_lock=float(abs(np.mean(np.exp(2j * error)))),
        window_lock_median=float(np.median(windows)),
        outside_cardiac_band=float(
            np.mean((rate < p.CARDIAC_BAND[0]) | (rate > p.CARDIAC_BAND[1]))
        ),
        rate=float(p._phase_rate(times, phase)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run", nargs=3, metavar=("SESSION", "TASK", "RUN"))
    parser.add_argument("--mean-only", action="store_true", help="Skip ICA starts in the baseline")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    runs = CHECK if args.check else PILOT
    if args.run:
        ses, task, run = args.run
        runs = [(int(ses), task, int(run))]
    for ses, task, run in tqdm(runs[: args.limit], desc="TASKFORCE runs", leave=True):
        label = f"s{ses:02d}_{task}_{run:02d}"
        start = time.monotonic()
        cache = args.output / f"{label}.npz"
        if cache.exists():
            with np.load(cache) as z:
                data = dict(z)
        else:
            data = prepare(ses, task, run, ica=not args.mean_only)
            np.savez_compressed(cache, **data)
        times, initial, tr = data["times"], data["initial"], float(data["tr"])
        train = [data[f"train_{g}"] for g in range(times.shape[1])]
        validation = [data[f"validation_{g}"] for g in range(times.shape[1])]
        # Recordings are loaded only after the image-derived cache has been built.
        peaks = reference(ses, task, run)
        baseline = json.loads(str(data["baseline"]))
        out = dict(run=label, baseline=baseline, variants={})
        phases = {"current": initial}
        out["variants"]["current"] = metrics(initial, times, peaks)
        out["variants"]["current"]["prediction"] = prediction_score(validation, initial, tr)
        for knot in (2.0, 4.0, 8.0):
            smoothed, _ = fit_phase(train, times, initial, knot_s=knot, maxiter=0)
            smooth_score = metrics(smoothed, times, peaks)
            smooth_score["prediction"] = prediction_score(validation, smoothed, tr)
            out["variants"][f"smooth_{knot:g}s"] = smooth_score
            phases[f"smooth_{knot:g}s"] = smoothed
            fitted, diag = fit_phase(train, times, initial, knot_s=knot)
            score = metrics(fitted, times, peaks)
            score.update(diag)
            score["prediction"] = prediction_score(validation, fitted, tr)
            out["variants"][f"spline_{knot:g}s"] = score
            phases[f"spline_{knot:g}s"] = fitted
        out["seconds"] = time.monotonic() - start
        out["selected_by_prediction"] = max(
            out["variants"], key=lambda name: out["variants"][name]["prediction"]
        )
        with (args.output / f"{label}.json").open("w") as f:
            json.dump(out, f, indent=2)
        np.savez_compressed(args.output / f"{label}.phases.npz", **phases)
        print(
            label,
            "found",
            baseline["found"],
            {
                k: {m: round(v[m], 4) for m in ("lock", "harmonic_lock", "prediction")}
                for k, v in out["variants"].items()
            },
            f"{out['seconds']:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
