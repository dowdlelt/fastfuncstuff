"""slicetime: ffs_slicetime against 3dTshift (wsinc9, tzero 0) on the raw run.

The frozen input is the raw first-echo magnitude and the SliceTiming its sidecars give
(BIDS inheritance resolved at planning, so both tools read the same list of times).
"""

from __future__ import annotations

import json
import os

from ..validation import compare_timeseries_4d
from .base import Check, UnitResult, afni_version, ffs_version, timed

REF_ARGS = ["-tzero", "0", "-wsinc9"]
FFS_ARGS = ["-tzero", "0", "-wsinc9"]


class SlicetimeCheck(Check):
    name = "slicetime"
    description = "Slice timing: ffs_slicetime vs 3dTshift (wsinc9), raw run as input"
    needs = [["slice_timing", "epi_2d"]]
    sampling = "breadth"
    reference = "AFNI 3dTshift"

    def unit_inputs(self, files, index):
        mags = [f for f in files if f.ent.get("part") != "phase" and f.suffix != "phase"]
        first = sorted(mags, key=lambda f: int(f.ent.get("echo", "1")))[0]
        # Sidecars shallowest first, so later ones override (BIDS inheritance order).
        return [first.rel] + [m.rel for m in index.matching(first, ".json")]

    def run(self, key, inputs, work, device):
        image = next(p for rel, p in inputs.items() if not rel.endswith(".json"))
        meta: dict = {}
        for rel, p in inputs.items():
            if rel.endswith(".json"):
                meta.update(json.loads(p.read_text()))
        st = meta.get("SliceTiming")
        if not st:
            raise ValueError("no SliceTiming in the run's sidecars")
        work.mkdir(parents=True, exist_ok=True)
        tp = work / "tpattern.txt"
        tp.write_text("\n".join(f"{float(t):.6f}" for t in st) + "\n")
        log = work / "commands.log"
        ref_out, ffs_out = work / "ref_tshift.nii", work / "ffs_tshift.nii"
        res = UnitResult()

        threads = int(os.environ.get("OMP_NUM_THREADS", "0")) or os.cpu_count()
        t_ref = timed(
            ["3dTshift", "-overwrite", "-prefix", str(ref_out), *REF_ARGS, "-tpattern", f"@{tp}", str(image)],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("ref", "cpu", t_ref, threads))
        t_ffs = timed(
            ["ffs_slicetime", "-input", str(image), *FFS_ARGS, "-tpattern", str(tp), "-prefix", str(ffs_out), "-device", device],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("ffs", device.split(",")[0], t_ffs, None))

        r = compare_timeseries_4d(str(ref_out), str(ffs_out), sample_frac=0.2)
        if "error" in r:
            raise RuntimeError(f"comparison failed: {r['error']}")
        res.metrics["reference"] = {
            k: r.get(k)
            for k in ("median_r", "mean_r", "min_r", "frac_above_0.95", "frac_above_0.99")
            if isinstance(r.get(k), (int, float))
        }
        from ..stages.slicetime import THRESHOLDS as T  # the old stage's gate

        res.metrics["reference"]["old_gate_pass"] = float(
            (r.get("median_r") or 0) >= T["median_r"]
            and (r.get("frac_above_0.95") or 0) >= T["frac_above_0.95"]
        )
        res.metrics["reference"]["n_slices"] = len(st)
        res.metrics["reference"]["n_shots"] = len({round(float(t), 4) for t in st})

        res.artifacts += [
            (ref_out, "regenerable"),
            (ffs_out, "regenerable"),
            (tp, "keep"),
            (log, "keep"),
        ]
        res.provenance["ref"] = {
            "tool": "3dTshift",
            "version": afni_version("3dTshift"),
            "args": REF_ARGS,
        }
        res.provenance["ffs"] = {
            "tool": "ffs_slicetime", **ffs_version(), "args": FFS_ARGS, "device": device,
            "note": "single-run process: timing includes interpreter start-up",
        }  # fmt: skip
        return res
