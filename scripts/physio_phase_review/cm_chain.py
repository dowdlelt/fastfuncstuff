"""Step-by-step audit of constant-modulus starts through alias choice and refinement.

Image-only steps first (CM traces, start phases, every complete alias with its slice
alignment, direct refinement); the recording is read afterwards to score each step.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.io.afni import load_nifti
from fastfuncstuff.processing import physio as p
from scripts.physio_phase_review.candidate_aliases import complete_alias_offsets
from scripts.physio_phase_review.cm_subspace import cm_traces, excitation_start
from scripts.physio_phase_review.direct_phase import fit_phase, select_voxels
from scripts.physio_phase_review.run_taskforce import metrics

CPU = torch.device("cpu")


def load(list_path, label, highpass):
    row = next(
        ln.split() for ln in list_path.read_text().splitlines() if ln.startswith(label + " ")
    )
    side = json.loads(Path(row[3]).read_text())
    st, tr = np.asarray(side["SliceTiming"], float), float(side["RepetitionTime"])
    mag = np.asanyarray(load_nifti(row[1]).dataobj)
    ph = np.asanyarray(load_nifti(row[2]).dataobj) if row[2] != "-" else None
    nt = mag.shape[-1] - p.trailing_noise_volumes(mag.mean(axis=(0, 1, 2)))
    mag = mag[..., :nt]
    ph = ph[..., :nt] if ph is not None else None
    traces = p.excitation_traces(mag, st, tr, phase=ph, device=CPU)
    tm, xm = traces.series("phase" if ph is not None else "magnitude", uniform=True)
    _, rf, _ = p.peak_rate(tm, xm, p.RESP_BAND)
    rp = np.interp(traces.times.ravel(), tm, p.cycle_phase(tm, xm, rf)).reshape(traces.times.shape)
    mean = mag.mean(-1)
    mask = mean > 0.25 * np.percentile(mean[mean > 0], 98)
    series, _ = p._cardiac_voxels(mag, st, tr, mask, rp, highpass)
    peaks = (row[4], float(row[5]), float(row[6]))
    return mag, mask, traces.times, tr, st, series, float(np.median(rf)), peaks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k", type=int)
    ap.add_argument("--highpass", type=float, default=p.REFINE_HIGHPASS_HZ)
    ap.add_argument("--refine", type=int, default=0, help="refine the best N aliases by alignment")
    args = ap.parse_args()
    torch.set_num_threads(2)
    mag, mask, times, tr, st, Ys, f_resp, peak_spec = load(args.list, args.label, args.highpass)
    cms = cm_traces(mag, mask, k=args.k, n=args.n)
    del mag
    Yn = [y.numpy() for y in Ys]
    steps = []
    for i, (ratio, c) in enumerate(cms):
        start = excitation_start(c, times, tr)
        offsets = complete_alias_offsets(times, start, tr)
        aliases = []
        for sgn, k in offsets.values():
            phase = sgn * start + 2 * np.pi * k * times / tr
            align = p.slice_alignment(p._voxel_fits(Ys, phase, CPU))
            aliases.append(
                dict(sign=sgn, step=k, rate=p._phase_rate(times, phase), align=align, phase=phase)
            )
        aliases.sort(key=lambda a: -a["align"])
        for a in aliases[: args.refine]:
            sel, _ = select_voxels([y[::2] for y in Yn], a["phase"])  # even rows train
            a["refined"], a["fit"] = fit_phase(sel, times, a["phase"], knot_s=2.0)
            a["refined_align"] = p.slice_alignment(p._voxel_fits(Ys, a["refined"], CPU))
        steps.append(dict(label=f"cm{i}", cm_ratio=ratio, aliases=aliases))
    # Production acceptance (refine_cardiac) on each start's best alias, unrefined:
    # beat 30 slice-time derangements re-choosing among the SAME complete alias set,
    # second harmonic above the non-harmonic controls, Bonferroni over the answers.
    rng = np.random.default_rng(0)
    answers = []
    for s in steps:
        a = s["aliases"][0]
        offs = complete_alias_offsets(times, a["phase"], tr)
        a["shuffled"] = p._shuffled_alignment(
            Ys, times, a["phase"], offs, tr, CPU, p.SLICE_PERMUTATIONS, rng
        )
        a["harmonic"], a["harmonic_null"] = p._harmonic_alignment(Ys, a["phase"], CPU)
        answers.append((s, a, offs))
    pulsed = [x for x in answers if x[1]["harmonic"] > x[1]["harmonic_null"]] or answers
    kept_s, kept, kept_offs = max(pulsed, key=lambda x: x[1]["align"] - x[1]["shuffled"].max())
    extra = p._shuffled_alignment(
        Ys, times, kept["phase"], kept_offs, tr, CPU, p.SLICE_PERMUTATIONS * (len(answers) - 1), rng
    )
    floor = float(np.concatenate([kept["shuffled"], extra]).max())
    found = bool(kept["align"] > floor and kept["harmonic"] > kept["harmonic_null"])
    decision = dict(
        found=found,
        kept=kept_s["label"],
        rate=kept["rate"],
        align=kept["align"],
        floor=floor,
        harmonic=kept["harmonic"],
        harmonic_null=kept["harmonic_null"],
    )
    # ---- recording opened here ----
    path, t0, fs = peak_spec
    peaks = np.loadtxt(path) / fs + t0
    out = dict(
        run=args.label,
        tr=tr,
        f_resp=f_resp,
        ref_hr=float(60 / np.median(np.diff(peaks))),
        avoid=p.cardiac_avoid(tr, f_resp),
        starts=[],
        decision=decision,
    )
    decision["lock"] = metrics(kept["phase"], times, peaks)["lock"]
    decision["window_lock_median"] = metrics(kept["phase"], times, peaks)["window_lock_median"]
    print(args.label, "ref_hr", round(out["ref_hr"], 1), "f_resp", round(f_resp, 3))
    for s in steps:
        rec = dict(label=s["label"], cm_ratio=s["cm_ratio"], aliases=[])
        for a in s["aliases"]:
            m = metrics(a["phase"], times, peaks)
            r = dict(
                sign=a["sign"], step=a["step"], rate=a["rate"], align=a["align"], lock=m["lock"]
            )
            if "shuffled" in a:
                r.update(
                    shuffled_max=float(a["shuffled"].max()),
                    harmonic=a["harmonic"],
                    harmonic_null=a["harmonic_null"],
                )
            if "refined" in a:
                mr = metrics(a["refined"], times, peaks)
                r.update(
                    refined_lock=mr["lock"],
                    refined_rate=mr["rate"],
                    refined_align=a["refined_align"],
                    outside=mr["outside_cardiac_band"],
                )
            rec["aliases"].append(r)
        out["starts"].append(rec)
        print(" ", s["label"], round(s["cm_ratio"], 3))
        for r in rec["aliases"]:
            print("    ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print(
        args.label,
        "DECISION",
        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in decision.items()},
        flush=True,
    )
    args.output.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
