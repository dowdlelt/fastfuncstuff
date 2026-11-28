"""moco: ffs_moco against 3dvolreg on the raw run, both with heptic interpolation.

The frozen input is the raw first-echo magnitude itself, so nothing upstream can differ.
Judged against the reference (motion parameters, matrices, mean image, corrected series).
Synthetic-truth scoring (scripts/moco_truth) is the next judge to add.
"""

from __future__ import annotations

from ..report_spec import Metric, ReportSpec
from ..validation import compare_1d_params, compare_aff12_series, compare_moco_ssd, compare_volumes
from .base import (
    Check,
    UnitResult,
    afni_version,
    cpu_threads,
    ffs_threads,
    ffs_version,
    first_steady_volume,
    timed,
    volume_means,
)

MOTION_PARAM_NAMES = ["roll", "pitch", "yaw", "dS", "dL", "dP"]

# Same settings as the old stage (stages/moco.py), whose thresholds stay the fastbench gate.
# The base volume is chosen per run (first steady-state volume), not fixed at 0.
REF_ARGS = ["-heptic"]
FFS_ARGS = ["-interp", "heptic", "-final", "heptic", "-weight_automask", "-save_mean"]


class MocoCheck(Check):
    name = "moco"
    description = "Motion correction: ffs_moco vs 3dvolreg (heptic), raw run as input"
    needs = [[]]  # any functional run
    sampling = "breadth"
    reference = "AFNI 3dvolreg"
    report = ReportSpec(
        summary="Raw run, registered to its first steady-state volume: ffs_moco against 3dvolreg, "
        "both heptic.",
        tools={},
        timing={"ref": "3dvolreg", "ffs": "ffs_moco"},
        references=("ref",),
        metrics=(
            Metric("reference.aff12_max_trans_mm", "Largest translation difference (mm)",
                   line=0.5, line_label="gate", log=True,
                   note="Over all volumes, between the two tools' base-to-source matrices."),
            Metric("reference.aff12_max_rot", "Largest rotation-block difference",
                   line=0.05, line_label="gate", log=True,
                   note="Largest element difference of the matrices' 3x3 blocks."),
            Metric("reference.mean_image_r", "Mean corrected image, r with 3dvolreg's",
                   better="higher", line=0.98, line_label="gate"),
            Metric("reference.motion_min_r", "Motion parameters, lowest r over the six",
                   better="higher"),
        ),
        speedup=(("ffs", "ref"),),
        code=("fastfuncstuff.cli.moco", "fastfuncstuff.benchmark.checks.moco"),
    )  # fmt: skip

    def unit_inputs(self, files, index):
        mags = [f for f in files if f.ent.get("part") != "phase" and f.suffix != "phase"]
        first = sorted(mags, key=lambda f: int(f.ent.get("echo", "1")))[0]
        return [first.rel]

    def run(self, key, inputs, work, device):
        (src,) = inputs.values()
        is_cbv = "_cbv." in str(src).rsplit("/", 1)[-1]
        ref, ffs = work / "ref", work / "ffs"
        ref.mkdir(parents=True, exist_ok=True)
        ffs.mkdir(parents=True, exist_ok=True)
        log = work / "commands.log"
        res = UnitResult()
        src, _ = self.trim_noise(src, work, log, res)
        base = first_steady_volume(volume_means(src))
        res.provenance["base_volume"] = {
            "index": base,
            "rule": "first volume within 5% of the run median mean",
        }

        threads = cpu_threads()
        t_ref = timed(
            [
                "3dvolreg", "-overwrite", *REF_ARGS, "-base", str(base), "-prefix", str(ref / "moco.nii"),
                "-1Dfile", str(ref / "motion.1D"), "-1Dmatrix_save", str(ref / "moco.aff12.1D"),
                str(src),
            ],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        timed(
            ["3dTstat", "-overwrite", "-prefix", str(ref / "mean.nii"), str(ref / "moco.nii")],
            work,
            log,
        )
        res.timings.append(("ref", "cpu", t_ref, threads))

        # ffs_moco turns VASO mode on for a *_cbv input (ds004539); this check is plain rigid
        # moco against 3dvolreg, which sees one series.
        vaso_off = ["-no_vaso"] if is_cbv else []
        t_ffs = timed(
            [
                "ffs_moco", *FFS_ARGS, *vaso_off, "-base", str(base), "-input", str(src), "-prefix", str(ffs / "moco.nii"),
                "-1Dfile", str(ffs / "motion.1D"), "-1Dmatrix_save", str(ffs / "moco.aff12.1D"),
                "-device", device,
            ],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        (ffs / "mean_moco.nii").replace(ffs / "mean.nii")
        res.timings.append(("ffs", device.split(",")[0], t_ffs, ffs_threads(device)))

        mp = compare_1d_params(str(ref / "motion.1D"), str(ffs / "motion.1D"))
        af = compare_aff12_series(str(ref / "moco.aff12.1D"), str(ffs / "moco.aff12.1D"))
        mv = compare_volumes(str(ref / "mean.nii"), str(ffs / "mean.nii"))
        sd = compare_moco_ssd(str(ref / "moco.nii"), str(ffs / "moco.nii"))
        m: dict[str, float | None] = {
            "motion_min_r": mp.get("min_r"),
            "motion_mean_r": mp.get("mean_r"),
            "aff12_max_trans_mm": af.get("max_trans_diff"),
            "aff12_max_rot": af.get("max_rot_diff"),
            "mean_image_r": mv.get("r"),
            "series_nrmsd": sd.get("nrmsd"),
        }
        for name, r in zip(MOTION_PARAM_NAMES, mp.get("per_column_r", []), strict=False):
            m[f"motion_r_{name}"] = r
        # The old stage's gate (stages/moco.py THRESHOLDS), so results read on the known scale.
        from ..stages.moco import THRESHOLDS as T

        m["old_gate_pass"] = float(
            (m["mean_image_r"] or 0) >= T["mean_image_min_r"]
            and (m["motion_mean_r"] or 0) >= T["motion_param_mean_r"]
            and (m["aff12_max_trans_mm"] or 1e9) <= T["aff12_max_trans_diff_mm"]
            and (m["aff12_max_rot"] or 1e9) <= T["aff12_max_rot_diff"]
        )
        res.metrics["reference"] = m

        for p in (ref / "moco.nii", ffs / "moco.nii"):
            res.artifacts.append((p, "regenerable"))
        for p in (
            ref / "motion.1D", ffs / "motion.1D", ref / "moco.aff12.1D", ffs / "moco.aff12.1D",
            ref / "mean.nii", ffs / "mean.nii", log,
        ):  # fmt: skip
            res.artifacts.append((p, "keep"))

        res.provenance["ref"] = {
            "tool": "3dvolreg",
            "version": afni_version("3dvolreg"),
            "args": REF_ARGS,
        }
        res.provenance["ffs"] = {
            "tool": "ffs_moco", **ffs_version(), "args": FFS_ARGS, "device": device,
            "note": "single-run process: timing includes interpreter and compile warm-up",
        }  # fmt: skip
        return res
