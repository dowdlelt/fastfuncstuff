"""Diagnostic upper bound: does recorded cardiac phase explain a missed run?

No discovery method sees these recordings. Compare preprocessing variants only
to locate lost information, not to choose an estimator using the true phase.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.io.afni import load_nifti
from fastfuncstuff.processing import physio as p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cpu = torch.device("cpu")
    row = next(
        line.split()
        for line in args.list.read_text().splitlines()
        if line.startswith(args.label + " ")
    )
    side = json.loads(Path(row[3]).read_text())
    st, tr = np.asarray(side["SliceTiming"]), float(side["RepetitionTime"])
    mag = np.asanyarray(load_nifti(row[1]).dataobj)
    nt = mag.shape[-1] - p.trailing_noise_volumes(mag.mean(axis=(0, 1, 2)))
    mag = mag[..., :nt]
    traces = p.excitation_traces(mag, st, tr, device=cpu)
    tm, xm = traces.series("magnitude", uniform=True)
    _, rf, _ = p.peak_rate(tm, xm, p.RESP_BAND)
    resp = np.interp(traces.times.ravel(), tm, p.cycle_phase(tm, xm, rf)).reshape(
        traces.times.shape
    )
    peaks = np.loadtxt(row[4]) / float(row[6]) + float(row[5])
    phase = p.beat_phase(peaks, traces.times.ravel()).reshape(traces.times.shape)
    mean = mag.mean(axis=-1)
    mask = mean > 0.25 * np.percentile(mean[mean > 0], 98)
    result = dict(
        run=args.label,
        recorded_rate_per_min=float(60 / np.median(np.diff(peaks))),
        estimated_respiration_per_min=float(60 * np.median(rf)),
        variants={},
    )
    for name, highpass, breathing, degree in (
        ("current", 0.05, resp, None),
        ("no_resp_projection", 0.05, None, None),
        ("drift_only", 0.0, None, None),
        ("mean_only", 0.0, None, 0),
    ):
        original_degree = p._drift_degree
        try:
            if degree is not None:
                p._drift_degree = lambda nt, tr, degree=degree: degree
            series, _ = p._cardiac_voxels(mag, st, tr, mask, breathing, highpass)
        finally:
            p._drift_degree = original_degree
        fits = p._voxel_fits(series, phase, cpu)
        r2 = np.concatenate([(fit[1] / (1 + fit[1])).numpy() for fit in fits if fit is not None])
        alignment = p.slice_alignment(fits)
        shuffled = p._shuffled_alignment(
            series, traces.times, phase, {0: (1, 0)}, tr, cpu, 30, np.random.default_rng(0)
        )
        harmonic, floor = p._harmonic_alignment(series, phase, cpu)
        metrics = dict(
            alignment=alignment,
            shuffled_max=float(shuffled.max()),
            harmonic=harmonic,
            harmonic_control_max=floor,
            r2_median=float(np.median(r2)),
            r2_p99=float(np.percentile(r2, 99)),
            fraction_r2_above_0p1=float(np.mean(r2 > 0.1)),
        )
        result["variants"][name] = metrics
        print(args.label, name, metrics, flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
