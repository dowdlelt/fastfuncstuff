"""How the number of whitened components K changes the constant-modulus starts.

Image-only: one Gram per run, then the search at every K in a ladder (best 5 distinct
subspaces each). The recording is read afterwards, for scoring only: each trace's
volume-rate lock to the recorded pulse, best over sign and the run's slice times.
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

LADDER = (6, 8, 10, 12, 16, 20, 24, 32, 40, 56)


def volume_lock(c, times, peaks):
    """|mean exp(i(arg c_n -/+ phi_rec(n TR + s)))|, best over sign and slice time s."""
    ang = np.angle(c)
    best = 0.0
    for g in range(times.shape[1]):
        truth = p.beat_phase(peaks, times[:, g])
        ok = (times[:, g] > peaks[0]) & (times[:, g] < peaks[-1])
        for sgn in (1, -1):
            best = max(best, abs(np.mean(np.exp(1j * (ang - sgn * truth))[ok])))
    return float(best)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    done = set()
    if args.output.exists():
        done = {json.loads(ln)["run"] for ln in args.output.read_text().splitlines() if ln}
    for row in (ln.split() for ln in args.list.read_text().splitlines() if ln.strip()):
        if row[0] in done or len(row) < 7 or not Path(row[1]).exists():
            continue
        t0 = time.monotonic()
        side = json.loads(Path(row[3]).read_text())
        st, tr = np.asarray(side["SliceTiming"], float), float(side["RepetitionTime"])
        mag = np.asanyarray(load_nifti(row[1]).dataobj)
        nt = mag.shape[-1] - p.trailing_noise_volumes(mag.mean(axis=(0, 1, 2)))
        mag = mag[..., :nt]
        mean = mag.mean(-1)
        mask = mean > 0.25 * np.percentile(mean[mean > 0], 98)
        G = cm.temporal_gram(mag, mask, device)
        del mag
        evals = torch.linalg.eigvalsh(G.cpu()).flip(0).numpy()
        ladder = {}
        for k in LADDER:
            if k > nt // 3:
                continue
            ladder[k] = cm.constant_modulus_traces(G, n_keep=5, k=k)
        # ---- recording opened here ----
        offs = p.excitation_groups(st)[1]  # unique excitation times within the TR
        times = np.arange(nt)[:, None] * tr + offs[None, :]
        peaks = np.loadtxt(row[4]) / float(row[6]) + float(row[5])
        rec = dict(run=row[0], tr=tr, nt=int(nt), G=int(times.shape[1]),
                   ref_hr=float(60 / np.median(np.diff(peaks))),
                   evals=(evals[:80] / evals.sum()).tolist(), ladder={})  # fmt: skip
        for k, trs in ladder.items():
            rec["ladder"][k] = [dict(ratio=r, lock=volume_lock(c, times, peaks)) for r, c in trs]
        rec["seconds"] = time.monotonic() - t0
        with args.output.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        summary = {k: (round(v[0]["lock"], 2), max(range(len(v)), key=lambda i: v[i]["lock"]))
                   for k, v in rec["ladder"].items()}  # fmt: skip
        print(row[0], f"nt {nt}", summary, f"{rec['seconds']:.0f}s", flush=True)
        del G
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
