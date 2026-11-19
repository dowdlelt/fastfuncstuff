"""Respiration from magnitude at long TR: rate and phase lock to the belt, per variant.

Image-only estimates first; the belt (BIDS physio, 'respiratory' column) is read
afterwards for scoring: its rate (Welch peak) and Hilbert phase around that rate.
Variants: the current path, the trace with k/TR notched in the band, and each
constant-modulus start (volume-rate lock, best over excitation time and sign).
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal

from fastfuncstuff.io.afni import load_nifti
from fastfuncstuff.processing import physio as p
from fastfuncstuff.processing import physio_cm as cm


def belt(physio_json, times_flat):
    meta = json.loads(Path(physio_json).read_text())
    x = pd.read_csv(physio_json.replace(".json", ".tsv.gz"), sep="\t", header=None).values
    x = x[:, meta["Columns"].index("respiratory")].astype(float)
    fs = float(meta["SamplingFrequency"])
    t = meta["StartTime"] + np.arange(x.size) / fs
    f, P = signal.welch(x - x.mean(), fs, nperseg=int(fs * 60))
    band = (f > 0.08) & (f < 0.8)
    f0 = f[band][P[band].argmax()]
    b, a = signal.butter(2, [0.6 * f0, 1.6 * f0], btype="band", fs=fs)
    ph = np.unwrap(np.angle(signal.hilbert(signal.filtfilt(b, a, x - x.mean()))))
    return 60 * f0, np.interp(times_flat, t, ph)


def lock(a, b):
    return float(abs(np.mean(np.exp(1j * (a - b)))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="dataset root with sub-*/[ses-*/]func")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    for bold in sorted(glob.glob(f"{args.root}/sub-*/**/func/*_bold.nii.gz", recursive=True)):
        pj = glob.glob(bold.replace("_bold.nii.gz", "*physio.json"))
        pj = [q for q in pj if "ecg" not in q]
        if not pj or "part-phase" in bold:
            continue
        side = json.loads(Path(bold.replace(".nii.gz", ".json")).read_text())
        st, tr = np.asarray(side["SliceTiming"], float), float(side["RepetitionTime"])
        mag = np.asanyarray(load_nifti(bold).dataobj)
        nt = mag.shape[-1] - p.trailing_noise_volumes(mag.mean(axis=(0, 1, 2)))
        mag = mag[..., :nt]
        traces = p.excitation_traces(mag, st, tr, device=dev)
        t, x = traces.series("magnitude", uniform=True)
        est = {}
        _, rf, _ = p.peak_rate(t, x, p.RESP_BAND)
        est["current"] = (60 * np.median(rf), p.cycle_phase(t, x, rf))
        xn = p.notch_tr_harmonics(t, x, tr, p.RESP_BAND)
        _, rfn, _ = p.peak_rate(t, xn, p.RESP_BAND)
        est["notch"] = (60 * np.median(rfn), p.cycle_phase(t, xn, rfn))
        mean = mag.mean(-1)
        mask = mean > 0.25 * np.percentile(mean[mean > 0], 98)
        cms = cm.constant_modulus_traces(cm.temporal_gram(mag, mask, dev), n_keep=5)
        del mag
        # ---- belt read here ----
        times = traces.times
        rate_b, ph_b = belt(pj[0], t)
        rec = dict(run=Path(bold).name, tr=tr, nt=int(nt), belt_rate=rate_b)
        for k, (r, ph) in est.items():
            rec[k] = dict(rate=float(r), lock=lock(ph, ph_b))
        _, ph_bx = belt(pj[0], times.ravel())
        ph_bx = ph_bx.reshape(times.shape)
        rec["cm"] = []
        for ratio, c in cms:
            best = max(lock(np.angle(c), s * ph_bx[:, g])
                       for g in range(times.shape[1]) for s in (1, -1))  # fmt: skip
            rec["cm"].append(dict(ratio=ratio, lock=best))
        print(rec["run"][:28], f"belt {rate_b:.1f}",
              {k: (round(rec[k]["rate"], 1), round(rec[k]["lock"], 2)) for k in est},
              "cm", [(round(c["ratio"], 2), round(c["lock"], 2)) for c in rec["cm"]], flush=True)  # fmt: skip
        with args.output.open("a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
