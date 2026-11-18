"""Run targeted discovery/confirmation experiments on missed TASKFORCE runs.

Hooks are confined to this single-process research runner and restored afterward.
The production library is untouched. Pulse recordings are loaded only after each
fit. Disabling respiratory exclusions is an ablation, not a proposed safe default.
"""

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.io.afni import load_nifti
from fastfuncstuff.processing import physio as p
from scripts.physio_phase_review.direct_phase import fit_phase, select_voxels
from scripts.physio_phase_review.run_taskforce import CPU, ROOT, metrics, reference


@contextmanager
def experiment_hooks(drop_resp_zones, phase_mode):
    original_avoid = p.cardiac_avoid
    original_voxels = p._cardiac_voxels
    original_crossed = p._crossed_phase
    training = []
    diagnostics = []

    def avoid(tr, f_resp=None, band=p.CARDIAC_BAND):
        return original_avoid(tr, None if drop_resp_zones else f_resp, band)

    def voxels(*args, **kwargs):
        series, xyz = original_voxels(*args, **kwargs)
        training[:] = [y[c[:, 0] % 2 == 0].numpy() for y, c in zip(series, xyz, strict=True)]
        return series, xyz

    def crossed(times, trace, avoid):
        phase = original_crossed(times, trace, avoid)
        if phase_mode == "current":
            return phase
        selected, _ = select_voxels(training, phase)
        phase, diag = fit_phase(
            selected,
            times,
            phase,
            knot_s=2.0,
            maxiter=0 if phase_mode == "smooth" else 150,
        )
        diagnostics.append(diag)
        return phase

    p.cardiac_avoid = avoid
    p._cardiac_voxels = voxels
    p._crossed_phase = crossed
    try:
        yield diagnostics
    finally:
        p.cardiac_avoid = original_avoid
        p._cardiac_voxels = original_voxels
        p._crossed_phase = original_crossed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=3, required=True, metavar=("SESSION", "TASK", "RUN"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop-resp-zones", action="store_true")
    parser.add_argument("--phase-mode", choices=("current", "smooth", "direct"), default="current")
    parser.add_argument("--mean-only", action="store_true")
    args = parser.parse_args()
    ses, task, run = int(args.run[0]), args.run[1], int(args.run[2])
    label = f"s{ses:02d}_{task}_{run:02d}"
    stem = ROOT / f"ses-{ses:02d}" / "func" / f"sub-pilot01_ses-{ses:02d}_task-{task}_run-{run:02d}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    started = time.monotonic()
    side = json.loads(Path(f"{stem}_part-mag_bold.json").read_text())
    st, tr = np.asarray(side["SliceTiming"]), float(side["RepetitionTime"])
    mag = np.asanyarray(load_nifti(f"{stem}_part-mag_bold.nii.gz").dataobj)
    phase = np.asanyarray(load_nifti(f"{stem}_part-phase_bold.nii.gz").dataobj)
    nt = mag.shape[-1] - p.trailing_noise_volumes(mag.mean(axis=(0, 1, 2)))
    mag, phase = mag[..., :nt], phase[..., :nt]
    traces = p.excitation_traces(mag, st, tr, phase=phase, device=CPU)
    del phase
    print(label, "traces ready", flush=True)
    with experiment_hooks(args.drop_resp_zones, args.phase_mode) as diagnostics:
        result = p.refine_cardiac(mag, traces, st, tr, device=CPU, ica=not args.mean_only)
    del mag
    # Oracle data cannot influence candidate generation, fitting, or selection.
    peaks = reference(ses, task, run)
    candidates = []
    for tried, phase in zip(result.tried, result.tried_phase, strict=True):
        candidates.append(
            dict(
                start=tried[0],
                alignment=tried[2],
                shuffled=tried[3],
                harmonic=tried[4],
                harmonic_null=tried[5],
                **metrics(phase, traces.times, peaks),
            )
        )
    chosen = next(candidate for candidate in candidates if candidate["start"] == result.start)
    record = dict(
        run=label,
        phase_mode=args.phase_mode,
        drop_resp_zones=args.drop_resp_zones,
        ica=not args.mean_only,
        found=result.found,
        selected=chosen,
        recorded_rate_per_min=float(60 / np.median(np.diff(peaks))),
        final_alignment_floor=result.alignment_null,
        candidates=candidates,
        fit_diagnostics=diagnostics,
        seconds=time.monotonic() - started,
    )
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".phases.npz"),
        times=traces.times,
        **{f"candidate_{i}": phase for i, phase in enumerate(result.tried_phase)},
    )
    print(
        label,
        "found",
        result.found,
        "selected",
        chosen,
        "best candidate lock",
        max(candidate["lock"] for candidate in candidates),
        f"{record['seconds']:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
