"""reml: ffs_reml against 3dREMLfit on one frozen design matrix.

Unit: one run with BIDS events. The frozen inputs are the run scaled to percent of its
mean inside a 3dAutomask, that mask, and the X matrix 3dDeconvolve builds from the
events (``-x1D_stop``). Both tools read exactly those three files, so the comparison is
the GLM and its ARMA(1,1) noise model alone: no design construction, no preprocessing.

Two ffs variants: ``-afni_mode`` (AFNI's banded R and (a,b) grid, the parity target) and
the default (ffs's exact noise model), whose distance from AFNI is a reported difference,
not a failure.

Rvar bricks compared in order a, b, lambda, stdev; a and b alone sit on a flat lambda
ridge, so lambda and stdev are the meaningful parity numbers (see stages/glm.py).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from ..validation import compare_bucket_volumes
from .base import Check, UnitResult, afni_version, ffs_version, timed

MAX_CONDITIONS = 12
MIN_EVENTS = 2
FFS_VARIANTS = {"ffs": ["-afni_mode"], "ffs_default": []}


# Non-BIDS spellings seen in private trees (MassedEvents9pt4T: event_onset, stimulus);
# BIDS names win when present. The columns used go into the unit's provenance.
ONSET_COLS = ("onset", "event_onset", "onset_s", "onsets")
CONDITION_COLS = ("trial_type", "stimulus", "condition", "event_type", "stim_type")


def event_columns(header: list[str]) -> tuple[str | None, str | None]:
    pick = lambda opts: next((c for c in opts if c in header), None)  # noqa: E731
    return pick(ONSET_COLS), pick(CONDITION_COLS)


def _conditions(events_tsv: Path) -> dict[str, list[tuple[float, float]]]:
    """condition -> [(onset, duration)], the commonest first, n/a dropped."""
    with open(events_tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        onset_col, cond_col = event_columns(list(reader.fieldnames or []))
    if onset_col is None:
        return {}
    out: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        try:
            onset = float(r[onset_col])
            dur = float(r.get("duration") or 0)
        except (KeyError, ValueError):
            continue
        tt = ((r.get(cond_col) if cond_col else None) or "event").strip()
        if tt in ("", "n/a"):
            tt = "event"
        out.setdefault(tt, []).append((onset, dur if dur == dur else 0.0))  # NaN -> 0
    keep = sorted((k for k in out if len(out[k]) >= MIN_EVENTS), key=lambda k: -len(out[k]))
    return {k: out[k] for k in keep[:MAX_CONDITIONS]}


def _label(tt: str, i: int) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in tt)[:24].strip("_")
    return safe or f"c{i}"


def _stim_args(conds: dict[str, list[tuple[float, float]]], tdir: Path) -> list[str]:
    args = ["-num_stimts", str(len(conds))]
    for i, (tt, ev) in enumerate(conds.items(), 1):
        durs = {round(d, 3) for _, d in ev}
        f = tdir / f"c{i:02d}.1D"
        if len(durs) == 1:
            d = durs.pop()
            f.write_text(" ".join(f"{o:.3f}" for o, _ in ev) + "\n")
            model = f"BLOCK({d:g},1)" if d >= 0.1 else "GAM"
            args += ["-stim_times", str(i), str(f), model]
        else:  # varying durations: married onset:duration, amplitude 1 per event
            f.write_text(" ".join(f"{o:.3f}:{d:.3f}" for o, d in ev) + "\n")
            args += ["-stim_times_AM1", str(i), str(f), "dmBLOCK(1)"]  # AM1 reads the :dur
        args += ["-stim_label", str(i), _label(tt, i)]
    return args


def _brick_stats(r: dict) -> tuple[float, float]:
    vals = sorted(x["r"] for x in r["per_brick"])
    return (vals[0], vals[len(vals) // 2]) if vals else (float("nan"), float("nan"))


class RemlCheck(Check):
    name = "reml"
    description = "GLM: ffs_reml vs 3dREMLfit on the same 3dDeconvolve X matrix"
    needs = [["events"]]
    sampling = "breadth"
    reference = "AFNI 3dREMLfit"

    def unit_inputs(self, files, index):
        mags = [f for f in files if f.ent.get("part") != "phase" and f.suffix != "phase"]
        first = sorted(mags, key=lambda f: int(f.ent.get("echo", "1")))[0]
        ev = index.matching(first, ".tsv", suffix="events")
        return (
            [first.rel]
            + [m.rel for m in index.matching(first, ".json")]
            + [m.rel for m in ev[-1:]]  # the most specific events file
        )

    def run(self, key, inputs, work, device):
        image = next(p for rel, p in inputs.items() if rel.endswith((".nii", ".nii.gz")))
        events = next((p for rel, p in inputs.items() if rel.endswith("_events.tsv")), None)
        if events is None:
            raise ValueError("no events file among the inputs")
        meta = self.read_meta(inputs)
        tr = meta.get("RepetitionTime")
        if meta.get("MRAcquisitionType") == "3D" and isinstance(tr, (int, float)) and tr < 0.3:
            # Some 3D sidecars (and headers) carry the per-shot TR; every onset would then
            # fall past the run's end and the design comes out all zeros.
            raise ValueError(
                f"3D run with RepetitionTime {tr} s: the shot TR, not the volume TR; "
                "fix the sidecar (or curated block) before this run can be modelled"
            )
        conds = _conditions(events)
        if not conds:
            raise ValueError(f"no trial_type with >= {MIN_EVENTS} events in {events.name}")
        work.mkdir(parents=True, exist_ok=True)
        log = work / "commands.log"
        tdir = work / "timing"
        tdir.mkdir(exist_ok=True)
        res = UnitResult()
        image, _ = self.trim_noise(image, work, log, res)

        # Frozen inputs: percent-of-mean series, its mask, the design.
        mean, mask, scaled = work / "mean.nii.gz", work / "mask.nii.gz", work / "scaled.nii.gz"
        timed(["3dTstat", "-overwrite", "-mean", "-prefix", str(mean), str(image)], work, log)
        timed(["3dAutomask", "-overwrite", "-prefix", str(mask), str(mean)], work, log)
        timed(
            ["3dcalc", "-overwrite", "-a", str(image), "-b", str(mean), "-c", str(mask),
             "-expr", "c*step(b)*min(200,a/b*100)", "-float", "-prefix", str(scaled)],
            work, log,
        )  # fmt: skip
        xmat = work / "X.xmat.1D"
        timed(
            ["3dDeconvolve", "-overwrite", "-input", str(scaled), "-polort", "A",
             # the sidecar TR (curated if wrong), not the header's: MassedEvents' header
             # carries the 3D shot TR too
             *(["-force_TR", f"{float(tr):g}"] if isinstance(tr, (int, float)) else []),
             *_stim_args(conds, tdir), "-x1D", str(xmat), "-x1D_stop"],
            work, log, timeout_s=600,
        )  # fmt: skip

        ref = work / "ref"
        ref.mkdir(exist_ok=True)
        threads = int(os.environ.get("OMP_NUM_THREADS", "0")) or os.cpu_count()
        t_ref = timed(
            ["3dREMLfit", "-overwrite", "-matrix", str(xmat), "-input", str(scaled),
             "-mask", str(mask), "-tout", "-nofdr", "-Obuck", str(ref / "ols.nii.gz"),
             "-Rbuck", str(ref / "reml.nii.gz"), "-Rvar", str(ref / "var.nii.gz")],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("ref", "cpu", t_ref, threads))

        for side, extra in FFS_VARIANTS.items():
            out = work / side
            out.mkdir(exist_ok=True)
            t = timed(
                ["ffs_reml", "-input", str(scaled), "-matrix", str(xmat), "-mask", str(mask),
                 "-tout", "-no_guard", *extra, "-Obuck", str(out / "ols.nii.gz"),
                 "-Rbuck", str(out / "reml.nii.gz"), "-Rvar", str(out / "var.nii.gz"),
                 "-device", device],
                work, log, timeout_s=self.timeout_s,
            )  # fmt: skip
            res.timings.append((side, device.split(",")[0], t, None))

        from ..stages.glm import THRESHOLDS as T

        for side in FFS_VARIANTS:
            sfx = "" if side == "ffs" else "_" + side.removeprefix("ffs_")
            out = work / side
            ols = compare_bucket_volumes(ref / "ols.nii.gz", out / "ols.nii.gz", mask_path=mask)
            reml = compare_bucket_volumes(ref / "reml.nii.gz", out / "reml.nii.gz", mask_path=mask)
            var = compare_bucket_volumes(ref / "var.nii.gz", out / "var.nii.gz", mask_path=mask)
            vr = {
                n: e["r"]
                for n, e in zip(("a", "b", "lambda", "stdev"), var["per_brick"], strict=False)
            }
            o_min, o_med = _brick_stats(ols)
            r_min, r_med = _brick_stats(reml)
            m = {
                f"ols_min_r{sfx}": o_min, f"ols_median_r{sfx}": o_med,
                f"reml_min_r{sfx}": r_min, f"reml_median_r{sfx}": r_med,
                f"lambda_r{sfx}": vr.get("lambda"), f"stdev_r{sfx}": vr.get("stdev"),
                f"a_r{sfx}": vr.get("a"), f"b_r{sfx}": vr.get("b"),
            }  # fmt: skip
            if side == "ffs":
                m["old_gate_pass"] = float(
                    o_min >= T["ols_beta_min_r"]
                    and r_min >= T["reml_beta_min_r"]
                    and (vr.get("lambda") or 0) >= T["reml_lambda_min_r"]
                    and (vr.get("stdev") or 0) >= T["reml_stdev_min_r"]
                )
            res.metrics.setdefault("reference", {}).update(m)
        res.metrics["reference"]["n_conditions"] = len(conds)
        first_row = next(ln for ln in xmat.read_text().splitlines() if ln and ln[0] != "#")
        res.metrics["reference"]["n_regressors"] = len(first_row.split())

        res.artifacts += [(scaled, "regenerable"), (mean, "regenerable")]
        for d in (ref, *(work / s for s in FFS_VARIANTS)):
            res.artifacts += [(d / f, "keep") for f in ("ols.nii.gz", "reml.nii.gz", "var.nii.gz")]
        res.artifacts += [(mask, "keep"), (xmat, "keep"), (log, "keep")]
        res.artifacts += [(p, "keep") for p in sorted(tdir.glob("*.1D"))]
        res.provenance["ref"] = {
            "tool": "3dREMLfit",
            "version": afni_version("3dREMLfit"),
            "design": "3dDeconvolve -polort A -x1D_stop",
        }
        res.provenance["ffs"] = {
            "tool": "ffs_reml", **ffs_version(), "variants": FFS_VARIANTS, "device": device,
        }  # fmt: skip
        res.provenance["inputs_prep"] = {
            "scaling": "3dcalc c*step(b)*min(200,a/b*100), b = 3dTstat mean, c = 3dAutomask",
            "conditions": {k: len(v) for k, v in conds.items()},
            "event_columns": event_columns(events.read_text().splitlines()[0].split("\t")),
        }
        return res
