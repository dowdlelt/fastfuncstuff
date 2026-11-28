"""melodic: ffs_ica against FSL MELODIC on the same smoothed, masked run.

ICA is not expected to match component for component. Each tool's FastICA starts from a
random unmixing matrix, so MELODIC does not even match itself. The check asks whether ffs
is within the spread MELODIC shows from run to run, and that it never falls apart.

The frozen input (prepared once, untimed, shared by every side):
1. the raw first-echo magnitude with its trailing noise volumes trimmed;
2. from the first steady-state volume on;
3. 3dAutomask of the mean;
4. a SMOOTH_FWHM mm blur inside that mask (3dBlurInMask).

The smoothing is the usual ICA preprocessing. MELODIC does none of its own, and doing it
once keeps the two tools on identical data. Both tools get the same mask, and neither
skull-strips (MELODIC ``--nobet``) or blurs again.

Sides:
- ``ref``: MELODIC, automatic dimensionality (Laplace). Its component count N is the one
  every other side uses.
- ``ref_repeat``: MELODIC again with ``-d N``. MELODIC against itself is the noise ceiling.
- ``ffs``: ffs_ica ``-num_comps N`` with MELODIC-style voxel variance normalisation. This is
  the matched comparison.
- ``ffs_default``: ffs_ica on its own defaults, including its own component count. Its
  count is recorded, not scored.

Judges: Hungarian matching on |spatial r| of the mixture-model z maps (melodic_IC against
ffs's ica_zmaps). Matched r is reported as mean and median, plus the share of MELODIC's
components that ffs finds at |r| >= 0.5 and >= 0.7, each with the MELODIC-vs-MELODIC
value beside it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..report_spec import Metric, ReportSpec
from .base import (
    Check,
    UnitResult,
    cpu_threads,
    ffs_threads,
    ffs_version,
    first_steady_volume,
    timed,
    volume_means,
)

SMOOTH_FWHM = 6.0
MIN_VOLUMES = 100  # fewer and the dimensionality estimate is not worth comparing


def match_components(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Hungarian matching of (n_a, v) and (n_b, v) maps on |r|; scores from a's side."""
    from scipy.optimize import linear_sum_assignment

    za = (a - a.mean(1, keepdims=True)) / np.maximum(a.std(1, keepdims=True), 1e-12)
    zb = (b - b.mean(1, keepdims=True)) / np.maximum(b.std(1, keepdims=True), 1e-12)
    r = np.abs(za @ zb.T) / a.shape[1]
    rows, cols = linear_sum_assignment(-r)
    matched = np.zeros(a.shape[0])
    matched[rows] = r[rows, cols]  # a component of a left unmatched (n_b < n_a) scores 0
    return {
        "matched_r_mean": float(matched.mean()),
        "matched_r_median": float(np.median(matched)),
        "coverage_0.5": float((matched >= 0.5).mean()),
        "coverage_0.7": float((matched >= 0.7).mean()),
    }


def _maps(path: Path, mask: np.ndarray) -> np.ndarray:
    import nibabel as nib

    d = np.asarray(nib.Nifti1Image.from_filename(str(path)).dataobj, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., None]
    return d[mask].T  # (n_comp, n_vox)


class MelodicCheck(Check):
    name = "melodic"
    description = "Single-run spatial ICA: ffs_ica vs FSL MELODIC on a smoothed, masked run"
    needs = [["rest"]]
    sampling = "breadth"
    reference = "FSL MELODIC (Laplace dimensionality)"
    timeout_s = 2 * 3600.0
    report = ReportSpec(
        summary=f"Single-run spatial ICA on the run blurred {SMOOTH_FWHM:g} mm in its "
        "automask: ffs_ica at MELODIC's component count against MELODIC, with MELODIC "
        "against itself (a second run) as the ceiling.",
        tools={"ffs": "ffs_ica", "ref_repeat": "MELODIC again"},
        timing={"ref": "MELODIC", "ffs": "ffs_ica (matched N)", "ffs_default": "ffs_ica"},
        references=("ref_repeat", "ref"),
        metrics=(
            Metric("reference.matched_r_median_{tool}", "Matched |r| with MELODIC, median",
                   better="higher", note="Each MELODIC component paired with one component "
                   "of the other run (Hungarian on |spatial r| of the z maps)."),
            Metric("reference.coverage_0.5_{tool}", "MELODIC components found at |r| >= 0.5",
                   better="higher"),
            Metric("reference.coverage_0.7_{tool}", "MELODIC components found at |r| >= 0.7",
                   better="higher"),
            Metric("intrinsic.n_comp_ratio", "ffs's own component count / MELODIC's",
                   better="none", line=1.0, log=True,
                   note="The two dimensionality estimators, on the same data."),
        ),
        speedup=(("ffs", "ref"), ("ffs_default", "ref")),
        difficulty=("intrinsic.n_comp_ref", "MELODIC components"),
        code=("fastfuncstuff.cli.ica", "fastfuncstuff.benchmark.checks.melodic"),
    )  # fmt: skip

    def unit_inputs(self, files, index):
        mags = [f for f in files if f.ent.get("part") != "phase" and f.suffix != "phase"]
        first = sorted(mags, key=lambda f: int(f.ent.get("echo", "1")))[0]
        return [first.rel] + [m.rel for m in index.matching(first, ".json")]

    def run(self, key, inputs, work, device):
        import nibabel as nib

        src = next(p for r, p in inputs.items() if r.endswith((".nii", ".nii.gz")))
        tr = float(self.read_meta(inputs).get("RepetitionTime") or 0) or float(
            nib.load(str(src)).header.get_zooms()[3]
        )
        work.mkdir(parents=True, exist_ok=True)
        log = work / "commands.log"
        res = UnitResult()
        src, _ = self.trim_noise(src, work, log, res)

        # -- frozen input: steady block, automask, blur in mask (untimed, shared) ----------
        means = volume_means(src)
        start = first_steady_volume(means)
        if len(means) - start < MIN_VOLUMES:
            raise ValueError(f"only {len(means) - start} steady volumes (< {MIN_VOLUMES})")
        block, mean = work / "block.nii.gz", work / "mean.nii.gz"
        mask_p, smooth = work / "mask.nii.gz", work / "smooth.nii.gz"
        timed(["3dTcat", "-overwrite", "-prefix", str(block), f"{src}[{start}..$]"], work, log)
        timed(["3dTstat", "-overwrite", "-mean", "-prefix", str(mean), str(block)], work, log)
        timed(["3dAutomask", "-overwrite", "-prefix", str(mask_p), str(mean)], work, log)
        timed(["3dBlurInMask", "-overwrite", "-input", str(block), "-mask", str(mask_p),
               "-FWHM", f"{SMOOTH_FWHM:g}", "-prefix", str(smooth)], work, log)  # fmt: skip
        mask = np.asarray(nib.Nifti1Image.from_filename(str(mask_p)).dataobj) > 0

        # -- MELODIC, twice ----------------------------------------------------------------
        threads = cpu_threads()
        env = {**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"}

        def melodic(out: Path, dim: list[str]) -> float:
            return timed(["melodic", "-i", str(smooth), "-o", str(out), "-m", str(mask_p),
                          "--nobet", f"--tr={tr:g}", *dim, "--Oall"],
                         work, log, timeout_s=self.timeout_s, env=env)  # fmt: skip

        ref, rep = work / "ref.ica", work / "ref_repeat.ica"
        res.timings.append(("ref", "cpu", melodic(ref, []), threads))
        ic_ref = _maps(ref / "melodic_IC.nii.gz", mask)
        n = ic_ref.shape[0]
        melodic(rep, ["-d", str(n)])  # the ceiling, not a timing side

        # -- ffs_ica, at MELODIC's N and on its own defaults ---------------------------------
        dev = device.split(",")[0]
        sides = {"ffs": ["-num_comps", str(n), "-voxel_norm"], "ffs_default": []}
        zmaps = {}
        for side, extra in sides.items():
            d = work / side
            d.mkdir(exist_ok=True)
            t = timed(["ffs_ica", "-input", str(smooth), "-mask", str(mask_p), "-tr", f"{tr:g}",
                       *extra, "-prefix", str(d / "ica"), "-device", device],
                      work, log, timeout_s=self.timeout_s)  # fmt: skip
            res.timings.append((side, dev, t, ffs_threads(device)))
            found = sorted(d.rglob("*_ica_zmaps.nii.gz"))
            if not found:
                raise FileNotFoundError(f"ffs_ica wrote no *_ica_zmaps.nii.gz under {d}")
            zmaps[side] = found[0]

        m: dict[str, float | None] = {}
        for tag, path in (("ffs", zmaps["ffs"]), ("ref_repeat", rep / "melodic_IC.nii.gz")):
            m.update(
                {f"{k}_{tag}": v for k, v in match_components(ic_ref, _maps(path, mask)).items()}
            )
        n_ffs = _maps(zmaps["ffs_default"], mask).shape[0]
        res.metrics["reference"] = m
        res.metrics["intrinsic"] = {
            "n_comp_ref": float(n), "n_comp_ffs_default": float(n_ffs),
            "n_comp_ratio": n_ffs / n, "n_volumes": float(len(means) - start),
            "mask_voxels": float(mask.sum()),
        }  # fmt: skip

        res.artifacts += [(p, "regenerable") for p in (block, smooth)]
        res.artifacts += [(mean, "keep"), (mask_p, "keep"), (log, "keep")]
        for d in (ref, rep):
            res.artifacts += [(p, "keep") for p in (d / "melodic_IC.nii.gz", d / "melodic_mix")]
            res.artifacts += [(p, "regenerable") for p in d.rglob("*") if p.is_file()
                              and p.name not in ("melodic_IC.nii.gz", "melodic_mix")]  # fmt: skip
        for side in sides:
            res.artifacts += [(zmaps[side], "keep")]
            res.artifacts += [(p, "regenerable") for p in (work / side).rglob("*.nii.gz")
                              if p != zmaps[side]]  # fmt: skip
        res.provenance["prep"] = {"first_volume": start, "smooth_fwhm_mm": SMOOTH_FWHM,
                                  "blur": "3dBlurInMask", "mask": "3dAutomask of the mean"}  # fmt: skip
        res.provenance["ref"] = {"tool": "melodic", "args": ["--nobet", "--Oall", "-m mask"],
                                 "version": _melodic_version()}  # fmt: skip
        res.provenance["ffs"] = {"tool": "ffs_ica", **ffs_version(), "sides": json.dumps(sides)}
        return res


def _melodic_version() -> str:
    import subprocess

    try:
        out = subprocess.run(["melodic", "--version"], capture_output=True, text=True, timeout=30)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return ""
