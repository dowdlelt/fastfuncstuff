"""Respiration by the cardiac chain: constant-modulus starts, alias by slice alignment
inside the breathing band, derangement null. Image-only through the decision; the
belt is read afterwards to score every start's every alias."""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.io.afni import load_nifti
from fastfuncstuff.processing import physio as p
from fastfuncstuff.processing import physio_cm as cm
from scripts.physio_phase_review.resp_probe import belt, lock


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--k", type=int)
    args = ap.parse_args()
    dev = torch.device("cuda")
    band = p.RESP_BAND
    for bold in sorted(glob.glob(f"{args.root}/sub-*/**/func/*_bold.nii.gz", recursive=True)):
        pj = [q for q in glob.glob(bold.replace("_bold.nii.gz", "*physio.json")) if "ecg" not in q]
        if not pj or "part-phase" in bold:
            continue
        side = json.loads(Path(bold.replace(".nii.gz", ".json")).read_text())
        st, tr = np.asarray(side["SliceTiming"], float), float(side["RepetitionTime"])
        mag = np.asanyarray(load_nifti(bold).dataobj)
        nt = mag.shape[-1] - p.trailing_noise_volumes(mag.mean(axis=(0, 1, 2)))
        mag = mag[..., :nt]
        times = np.arange(nt)[:, None] * tr + p.excitation_groups(st)[1][None, :]
        mean = mag.mean(-1)
        mask = mean > 0.25 * np.percentile(mean[mean > 0], 98)
        host, _ = p._cardiac_voxels(mag, st, tr, mask, None, 0.05)
        Ys, _ = p._store_voxels(host, dev)
        cms = cm.constant_modulus_traces(
            cm.temporal_gram(mag, mask, dev), n_keep=args.keep, k=args.k
        )
        del mag, host
        rng = np.random.default_rng(0)
        starts = []
        for ratio, c in cms:
            start = cm.excitation_phase(c, times, tr, mid=0.3)
            rows = cm._alias_rows(Ys, times, start, tr, dev, band)
            if not rows:
                continue
            null = cm._null(Ys, times, start, tr, dev, band, 30, rng)
            starts.append((ratio, rows, null))
        # ---- belt read here ----
        rate_b, ph = belt(pj[0], times.ravel())
        ph = ph.reshape(times.shape)
        rec = dict(run=Path(bold).name, tr=tr, belt=rate_b, starts=[])
        for ratio, rows, null in starts:
            rec["starts"].append(dict(ratio=ratio, null=float(null.max()), aliases=[
                dict(rate=60 * p._phase_rate(times, q), a1=float(a1), a2=float(a2),
                     lock=lock(q, ph)) for q, a1, a2 in rows]))  # fmt: skip
        with args.output.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec["run"][:24], f"belt {rate_b:.1f}")
        for s in rec["starts"]:
            b = max(s["aliases"], key=lambda a: a["a1"] + a["a2"])
            print(f"   ratio {s['ratio']:.2f} null {s['null']:.2f} chosen {b['rate']:.1f}/min "
                  f"a1 {b['a1']:.2f} a2 {b['a2']:.2f} lock {b['lock']:.2f} | best lock "
                  f"{max(a['lock'] for a in s['aliases']):.2f}", flush=True)  # fmt: skip
        del Ys
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
