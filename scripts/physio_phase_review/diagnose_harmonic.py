"""Compare harmonic evidence on the cached, selected prediction voxels.

These are conditional diagnostics on a restricted voxel pool, not acceptance
decisions from the full detector. The recorded phase is an evaluation-only oracle.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.processing import physio as p
from scripts.physio_phase_review.run_taskforce import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cpu = torch.device("cpu")
    records = []
    for result_path in sorted(args.cache.glob("s*.json")):
        label = result_path.stem
        with np.load(args.cache / f"{label}.npz") as data:
            times = data["times"]
            series = [torch.from_numpy(data[f"validation_{g}"]) for g in range(times.shape[1])]
        with np.load(args.cache / f"{label}.phases.npz") as phases:
            candidates = {key: phases[key] for key in ("current", "smooth_2s", "spline_2s")}
        ses, task, run = label.split("_")
        peaks = reference(int(ses[1:]), task, int(run))
        candidates["recorded_oracle"] = p.beat_phase(peaks, times.ravel()).reshape(times.shape)
        record = dict(run=label, variants={})
        for key, phase in candidates.items():
            harmonic, floor = p._harmonic_alignment(series, phase, cpu)
            alignment = p.slice_alignment(p._voxel_fits(series, phase, cpu))
            record["variants"][key] = dict(
                alignment=alignment, harmonic=harmonic, control_max=floor, margin=harmonic - floor
            )
        records.append(record)
        print(
            label,
            {key: round(score["margin"], 4) for key, score in record["variants"].items()},
            flush=True,
        )
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
