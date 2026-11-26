"""warp_pair: every nonlinear warp tool on the same labelled pair, judged by transported labels.

Unit: an ordered pair (source, base) from a labelled cohort (NIREP16), drawn like the qwarp
check and prepared the same way (``qwarp.prep_pair``: 1 mm, one shared 3dAllineate affine),
so each tool fits only the nonlinear part from the same aligned source.

References:
- antsRegistration SyN on every pair, with the nonlinear stage of antsRegistrationSyN.sh:
  CC radius 4, SyN[0.1,3,0], 100x70x50x20 iterations, shrink 8x4x2x1, smooth 3x2x1x0 vox.
- 3dQwarp (the qwarp check's arguments) on one pair in QWARP_EVERY: it takes 25 min to 2 h.
ffs: qwarp (3dQwarp's arguments), formwarp with ANTs' SyN settings flag for flag (its
defaults; the engine reproduces antsRegistration -- Dice 0.6237 vs 0.6240 on these 48 pairs
-- so this row is a parity check, not a tuned fit), optiwarp at its defaults and with the hs
force (the autoproc MNI default).

Judges, per tool:
- truth: Dice of the source's traced parcels against the base's (mean / median / min) and
  the gain over the affine alone. It ranks every tool against the others, not only against
  its own reference.
- deformation: folding (det J <= 0) and the spread of log det J inside the base brain.
- nwarp: each AFNI-format warp applied to the labels by 3dNwarpApply and by ffs_nwarp (NN);
  both are deterministic, so agreement should be near total.
AFNI-format warps move labels with 3dNwarpApply and give det J from 3dNwarpFuncs; ANTs'
warp uses antsApplyTransforms and CreateJacobianDeterminantImage (ffs has no ITK-to-AFNI
warp converter yet). Both are nearest-neighbour label transports on the base grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..report_spec import Metric, ReportSpec
from .base import (
    UnitResult,
    _stable_rank,
    afni_version,
    cpu_threads,
    ffs_threads,
    ffs_version,
    timed,
    timed_cached,
)
from .qwarp import QWARP_ARGS, QwarpCheck, _dice_per_label, prep_pair

QWARP_EVERY = 4

ANTS_SYN = ["-t", "SyN[0.1,3,0]", "-c", "[100x70x50x20,1e-6,10]", "-f", "8x4x2x1",
            "-s", "3x2x1x0vox"]  # fmt: skip
# formwarp's flags for the same SyN stage, in the same units: gradient step 0.1, update
# variance 3, total variance 0, CC radius 4, same shrink / smooth / iterations / convergence.
# These are formwarp's defaults; spelled out so the mapping is on the record.
FORMWARP_AS_ANTS = ["-style", "ants", "-grad_step", "0.1", "-update_var", "3", "-total_var", "0",
                    "-metric", "cc", "-cc_radius", "4", "-shrink", "8x4x2x1", "-smooth", "3x2x1x0",
                    "-iters", "100x70x50x20", "-conv_thresh", "1e-6", "-conv_window", "10"]  # fmt: skip

# name -> (command head, extra args, output stem). Every tool writes STEM.nii.gz (warped
# source) and STEM_WARP.nii.gz (AFNI-format source->base warp) in its own folder.
FFS_TOOLS: dict[str, tuple[str, list[str], str]] = {
    "qwarp": ("ffs_qwarp", QWARP_ARGS, "qw"),
    "formwarp": ("ffs_formwarp", ["-save_warp", *FORMWARP_AS_ANTS], "fw"),
    "optiwarp": ("ffs_optiwarp", ["-save_warp"], "ow"),
    "optiwarp_hs": ("ffs_optiwarp", ["-save_warp", "-force", "hs"], "ow"),
}


def _ants_version() -> str:
    import subprocess

    try:
        out = subprocess.run(["antsRegistration", "--version"], capture_output=True, text=True)
        return out.stdout.splitlines()[0].split(":", 1)[-1].strip()
    except (OSError, IndexError):
        return ""


def _load(p: Path) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.Nifti1Image.from_filename(str(p)).dataobj)


def deformation_judges(detj: np.ndarray, inside: np.ndarray) -> dict[str, float]:
    """Folding, and how far local volume change reaches, inside the base brain.

    ``detj`` is det J itself (not AFNI's bulk = det J - 1) of the warp as every tool stores
    it: in base space, pulling each base voxel from the source. det J > 1 there means a base
    voxel draws on a larger patch of source, i.e. tissue squeezed, so it is inverted here:
    every number is the source tissue's own volume change (< 1 shrinks). The spread hides a
    squeezed corner, so the tails are reported too: 0.1/1/99/99.9th percentiles of log
    volume change and the share of voxels whose tissue more than halves or doubles."""
    d = tissue_volume_change(detj[inside])
    pos = d[d > 0]
    lj = np.log(pos) if pos.size else np.array([np.nan])
    p001, p01, p99, p999 = np.percentile(lj, [0.1, 1, 99, 99.9])
    return {
        "fold_frac": float((d <= 0).mean()),
        "logjac_sd": float(lj.std()),
        "logjac_p001": float(p001),
        "logjac_p01": float(p01),
        "logjac_p99": float(p99),
        "logjac_p999": float(p999),
        "frac_det_below_half": float((d < 0.5).mean()),
        "frac_det_above_2": float((d > 2.0).mean()),
    }


def tissue_volume_change(detj: np.ndarray) -> np.ndarray:
    """1 / det J of a base-space pullback warp: how much the source tissue that lands at each
    base voxel grew (> 1) or shrank (< 1). Folds (det J <= 0) stay non-positive."""
    d = np.asarray(detj, np.float64)
    with np.errstate(divide="ignore"):
        return np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), d)


LOST_RATIO = 0.5  # a label the warp leaves with under half its volume is "lost"


def label_judges(
    warped: np.ndarray,
    affine: np.ndarray,
    detj: np.ndarray,
    labels: np.ndarray,
    base: np.ndarray | None = None,
) -> tuple[dict[str, float], str]:
    """What the warp itself did to each label, beyond the affine.

    For every label: its volume after the warp over its volume after the affine alone (the
    warp's own share; shrinking a little may be right, vanishing is not), and the mean
    log tissue volume change where it lands (< 0 shrinks; see ``deformation_judges``). Returns summary judges and a per-label TSV (with Dice against
    ``base`` when a traced base exists)."""
    n_aff = np.array([(affine == lab).sum() for lab in labels], np.float64)
    n_w = np.array([(warped == lab).sum() for lab in labels], np.float64)
    ok = n_aff > 0
    ratio = np.where(ok, n_w / np.maximum(n_aff, 1), np.nan)
    lj = np.log(np.clip(tissue_volume_change(detj), 1e-6, None))
    mlj = np.array(
        [lj[warped == lab].mean() if n > 0 else np.nan for lab, n in zip(labels, n_w, strict=True)]
    )
    r = ratio[ok]
    j = {
        "label_vol_ratio_min": float(np.nanmin(r)),
        "label_vol_ratio_p10": float(np.nanpercentile(r, 10)),
        "label_vol_ratio_median": float(np.nanmedian(r)),
        "label_vol_ratio_max": float(np.nanmax(r)),
        "n_labels_shrunk_20pct": float((r < 0.8).sum()),
        "n_labels_lost": float((r < LOST_RATIO).sum()),
        "n_labels_vanished": float(((n_w == 0) & ok).sum()),
        "n_labels_grown_25pct": float((r > 1.25).sum()),
        "label_logjac_min": float(np.nanmin(mlj)),
        "label_logjac_max": float(np.nanmax(mlj)),
    }
    cols = ["label", "vox_affine", "vox_warped", "vol_ratio", "mean_logjac"]
    dice = None
    if base is not None:
        cols.append("dice")
        dice = _dice_per_label(warped.astype(np.int32), base.astype(np.int32), labels)
    rows = ["\t".join(cols)]
    for i, lab in enumerate(labels):
        row = [
            str(int(lab)),
            str(int(n_aff[i])),
            str(int(n_w[i])),
            f"{ratio[i]:.4f}",
            f"{mlj[i]:+.4f}",
        ]
        if dice is not None:
            row.append(f"{dice[i]:.4f}")
        rows.append("\t".join(row))
    return j, "\n".join(rows) + "\n"


def dice_judges(lab: np.ndarray, base_lab: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    d = _dice_per_label(lab.astype(np.int32), base_lab, labels)
    return {
        "dice_mean": float(np.nanmean(d)),
        "dice_median": float(np.nanmedian(d)),
        "dice_min": float(np.nanmin(d)),
    }


class WarpPairCheck(QwarpCheck):
    name = "warp_pair"
    description = "Nonlinear warp family on one pair: ANTs SyN / 3dQwarp vs every ffs warp tool"
    reference = "antsRegistration SyN (every pair), AFNI 3dQwarp (every 4th)"
    report = ReportSpec(
        summary="NIREP16 brain to brain, from one shared affine: ANTs SyN (every pair) and 3dQwarp "
        "(every 4th) against every ffs warp tool; manual labels judge the overlap.",
        tools={"affine": "affine only", "ants": "ANTs SyN", "3dqwarp": "3dQwarp", "qwarp": "ffs_qwarp",
               "optiwarp": "ffs_optiwarp", "optiwarp_hs": "ffs_optiwarp (hs)",
               "formwarp": "ffs_formwarp"},
        timing={"ref": "ANTs SyN", "ref_3dqwarp": "3dQwarp", "ffs_qwarp": "ffs_qwarp",
                "ffs_optiwarp": "ffs_optiwarp", "ffs_optiwarp_hs": "ffs_optiwarp (hs)",
                "ffs_formwarp": "ffs_formwarp"},
        references=("affine", "ants", "3dqwarp", "ref", "ref_3dqwarp"),
        metrics=(
            Metric("truth.dice_mean_{tool}", "Mean label Dice", better="higher",
                   note="Source labels carried through each warp, against the base's manual labels."),
            Metric("label_loss.label_vol_ratio_min_{tool}", "Smallest label kept (volume after warp / after affine)",
                   better="higher", line=0.5, line_label="lost"),
            Metric("deformation.frac_det_below_half_{tool}", "Brain squeezed below half its volume (share)"),
        ),
        speedup=(("ffs_qwarp", "ref_3dqwarp"), ("ffs_formwarp", "ref"), ("ffs_optiwarp_hs", "ref"),
                 ("ffs_optiwarp", "ref")),
        code=("fastfuncstuff.cli.qwarp", "fastfuncstuff.cli.optiwarp", "fastfuncstuff.cli.formwarp", "fastfuncstuff.benchmark.checks.warp_pair"),
    )  # fmt: skip
    timeout_s = 4 * 3600.0

    def run(self, key, inputs, work, device):
        src_img, src_lab, base_img, base_lab = (inputs[r] for r in list(inputs)[:4])
        work.mkdir(parents=True, exist_ok=True)
        log = work / "commands.log"
        prep = work / "prep"
        prep.mkdir(exist_ok=True)
        S, SL, B, BL, SA, SLA, aff = prep_pair(
            src_img, src_lab, base_img, base_lab, prep, work, log
        )

        res = UnitResult()
        threads = cpu_threads()
        base_lab_v = _load(BL).astype(np.int32)
        labels = np.unique(base_lab_v)
        labels = labels[labels != 0]
        inside = _load(B) > 0
        truth: dict[str, float | None] = {"n_labels": float(len(labels))}
        deform: dict[str, float | None] = {}
        nwarp: dict[str, float | None] = {}
        truth.update(
            {f"{k}_affine": v for k, v in dice_judges(_load(SLA), base_lab_v, labels).items()}
        )

        sla_v = _load(SLA).astype(np.int32)
        label_loss: dict[str, float | None] = {}

        def lab_judges(tool: str, d: Path, warped: np.ndarray, detj: np.ndarray) -> None:
            j, tsv = label_judges(warped.astype(np.int32), sla_v, detj, labels, base_lab_v)
            label_loss.update({f"{k}_{tool}": v for k, v in j.items()})
            (d / "labels.tsv").write_text(tsv)
            res.artifacts.append((d / "labels.tsv", "keep"))

        def afni_warp_judges(tool: str, d: Path, warp: Path) -> None:
            lab = d / "lab_warped.nii.gz"
            timed(["3dNwarpApply", "-overwrite", "-nwarp", str(warp), "-source", str(SLA),
                   "-master", str(B), "-ainterp", "NN", "-prefix", str(lab)], work, log)  # fmt: skip
            truth.update(
                {f"{k}_{tool}": v for k, v in dice_judges(_load(lab), base_lab_v, labels).items()}
            )
            jac, jac_b = d / "jac.nii.gz", d / "jac_base.nii.gz"
            timed(["3dNwarpFuncs", "-overwrite", "-nwarp", str(warp), "-bulk",
                   "-prefix", str(jac)], work, log)  # fmt: skip
            # warps sit on each tool's padded grid; score on the base grid
            timed(["3dresample", "-overwrite", "-master", str(B), "-rmode", "NN",
                   "-input", str(jac), "-prefix", str(jac_b)], work, log)  # fmt: skip
            detj = _load(jac_b).astype(np.float64) + 1.0  # bulk = det J - 1
            deform.update({f"{k}_{tool}": v for k, v in deformation_judges(detj, inside).items()})
            lab_judges(tool, d, _load(lab), detj)
            lab_ffs = d / "lab_ffs_nwarp.nii.gz"
            timed(["ffs_nwarp", "-source", str(SLA), "-nwarp", str(warp), "-master", str(B),
                   "-interp", "NN", "-prefix", str(lab_ffs), "-device", device], work, log)  # fmt: skip
            a, b = _load(lab).astype(np.int32), _load(lab_ffs).astype(np.int32)
            either = (a > 0) | (b > 0)
            nwarp[f"label_agree_{tool}"] = float((a[either] == b[either]).mean())
            res.artifacts.extend([(lab, "keep"), (jac, "regenerable"), (jac_b, "regenerable"),
                                  (lab_ffs, "regenerable")])  # fmt: skip

        # -- ANTs SyN (every pair) ------------------------------------------------
        ad = work / "ants"
        ad.mkdir(exist_ok=True)
        ants_warp = ad / "ants_0Warp.nii.gz"
        env = {"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(threads)}
        t = timed_cached(
            ["antsRegistration", "-d", "3", "-o", f"[{ad / 'ants_'},{ad / 'ants_warped.nii.gz'}]",
             "-n", "Linear", "-u", "0", "-z", "1", "-w", "[0.005,0.995]",
             "-m", f"CC[{B},{SA},1,4]", *ANTS_SYN],
            ad / "fit.timing.json", [ants_warp, ad / "ants_warped.nii.gz"],
            work, log, timeout_s=self.timeout_s, env=env,
        )  # fmt: skip
        res.timings.append(("ref", "cpu", t, threads))
        lab = ad / "lab_warped.nii.gz"
        timed(["antsApplyTransforms", "-d", "3", "-i", str(SLA), "-r", str(B), "-o", str(lab),
               "-n", "NearestNeighbor", "-t", str(ants_warp)], work, log, env=env)  # fmt: skip
        truth.update(
            {f"{k}_ants": v for k, v in dice_judges(_load(lab), base_lab_v, labels).items()}
        )
        jac = ad / "jac.nii.gz"
        timed(["CreateJacobianDeterminantImage", "3", str(ants_warp), str(jac), "0", "0"],
              work, log, env=env)  # fmt: skip
        detj_ants = _load(jac).astype(np.float64)
        deform.update({f"{k}_ants": v for k, v in deformation_judges(detj_ants, inside).items()})
        lab_judges("ants", ad, _load(lab), detj_ants)
        res.artifacts += [(lab, "keep"), (jac, "regenerable"), (ants_warp, "keep"),
                          (ad / "ants_0InverseWarp.nii.gz", "regenerable"),
                          (ad / "ants_warped.nii.gz", "keep"), (ad / "fit.timing.json", "keep")]  # fmt: skip

        # -- 3dQwarp (one pair in QWARP_EVERY) --------------------------------------
        if int(_stable_rank(key["src"], key["base"]), 16) % QWARP_EVERY == 0:
            qd = work / "afni_qwarp"
            qd.mkdir(exist_ok=True)
            t = timed_cached(
                ["3dQwarp", "-overwrite", "-base", str(B), "-source", str(SA),
                 "-prefix", str(qd / "qw.nii.gz"), *QWARP_ARGS],
                qd / "fit.timing.json", [qd / "qw.nii.gz", qd / "qw_WARP.nii.gz"],
                work, log, timeout_s=self.timeout_s,
            )  # fmt: skip
            res.timings.append(("ref_3dqwarp", "cpu", t, threads))
            afni_warp_judges("3dqwarp", qd, qd / "qw_WARP.nii.gz")
            # outputs timed_cached checks stay, so `work -redo` reuses this fit
            res.artifacts += [(qd / "qw_WARP.nii.gz", "keep"), (qd / "qw.nii.gz", "keep"),
                              (qd / "fit.timing.json", "keep")]  # fmt: skip

        # -- ffs tools ------------------------------------------------------------------
        dev = device.split(",")[0]
        for tool, (cli, args, stem) in FFS_TOOLS.items():
            d = work / f"ffs_{tool}"
            d.mkdir(exist_ok=True)
            prefix = d / stem if cli == "ffs_qwarp" else d / f"{stem}.nii.gz"
            warped, warp = d / f"{stem}.nii.gz", d / f"{stem}_WARP.nii.gz"
            t = timed_cached(
                [cli, "-base", str(B), "-source", str(SA), "-prefix", str(prefix), *args,
                 "-device", device],
                d / "fit.timing.json", [warped, warp], work, log, timeout_s=self.timeout_s,
            )  # fmt: skip
            res.timings.append((f"ffs_{tool}", dev, t, ffs_threads(device)))
            afni_warp_judges(tool, d, warp)
            res.artifacts += [
                (warp, "keep"),
                (warped, "keep"),
                (d / "fit.timing.json", "keep"),
            ]

        for tool in ["ants", "3dqwarp", *FFS_TOOLS]:
            if f"dice_mean_{tool}" in truth:
                truth[f"dice_gain_{tool}"] = truth[f"dice_mean_{tool}"] - truth["dice_mean_affine"]  # type: ignore[operator]
        res.metrics["truth"] = truth
        res.metrics["deformation"] = deform
        res.metrics["nwarp"] = nwarp
        res.metrics["label_loss"] = label_loss
        res.metrics["reference"] = {
            f"dice_mean_{t}": truth.get(f"dice_mean_{t}")
            for t in ("ants", "formwarp", "qwarp", "optiwarp")
        }
        res.artifacts += [(p, "regenerable") for p in (S, SL, B, SA)]
        res.artifacts += [(BL, "keep"), (SLA, "keep"), (aff, "keep"), (log, "keep")]
        res.provenance["ref"] = {
            "tool": "antsRegistration",
            "version": _ants_version(),
            "args": ["-m", "CC[base,source,1,4]", *ANTS_SYN, "-u", "0", "-w", "[0.005,0.995]"],
        }
        res.provenance["ref_3dqwarp"] = {"tool": "3dQwarp", "version": afni_version("3dQwarp"),
                                         "args": QWARP_ARGS, "every": QWARP_EVERY}  # fmt: skip
        res.provenance["ffs"] = {"tools": {k: [c, *a] for k, (c, a, _) in FFS_TOOLS.items()},
                                 **ffs_version(), "device": device}  # fmt: skip
        res.provenance["inputs_prep"] = {
            "resample": "3dresample -dxyz 1 (Cu image, NN labels)",
            "affine": "3dAllineate -cost ls -twopass -final wsinc5, shared by every tool",
            "label_transport": "3dNwarpApply NN (AFNI-format warps); antsApplyTransforms NN (ANTs)",
        }
        return res
