"""qwarp: ffs_qwarp against 3dQwarp, judged by transported manual labels (NIREP16).

Unit: an ordered pair (source, base) from a labelled cohort, sampled in depth: many pairs
of one dataset, because label agreement is a distribution. Frozen inputs: both brains and
tracings resampled to 1 mm (wsinc5 / NN), and one 3dAllineate affine of source to base,
shared by both nonlinear fits. Both tools get identical arguments.

The labels are carried through each tool's warp by the same applier (3dNwarpApply, NN),
so only the warp differs. Judges:
- truth: mean Dice over the base's traced parcels, against the affine-only baseline.
- deformation: share of voxels with Jacobian <= 0 for each warp (3dNwarpFuncs -bulk
  writes det(J) - 1, so the test is bulk <= -1).
- reference: correlation of the two warped brains inside the base mask.

The pairwise design follows ffs_tunewarp's cohort judge (wiki: tuning registration on
anatomical labels); NIREP: Christensen et al. 2006, LNCS 4057.
"""

from __future__ import annotations

import numpy as np

from .base import (
    Check,
    UnitResult,
    _stable_rank,
    afni_version,
    cpu_threads,
    ffs_threads,
    ffs_version,
    timed,
    timed_cached,
)

DXYZ = "1"
QWARP_ARGS = ["-minpatch", "9", "-blur", "0", "3"]


def _dice_per_label(a: np.ndarray, b: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = []
    for lab in labels:
        x, y = a == lab, b == lab
        s = x.sum() + y.sum()
        out.append(2.0 * (x & y).sum() / s if s else np.nan)
    return np.asarray(out)


def prep_pair(src_img, src_lab, base_img, base_lab, prep, work, log):
    """Both brains and tracings at 1 mm (cubic / NN) and one 3dAllineate affine
    of source to base, applied to the source image and labels. Every nonlinear tool of a
    pair starts from the same aligned source, so only the nonlinear part differs."""

    def resample(src, out, mode):
        timed(["3dresample", "-overwrite", "-dxyz", DXYZ, DXYZ, DXYZ, "-rmode", mode,
               "-input", str(src), "-prefix", str(out)], work, log)  # fmt: skip

    S, SL = prep / "src.nii.gz", prep / "src_lab.nii.gz"
    B, BL = prep / "base.nii.gz", prep / "base_lab.nii.gz"
    resample(src_img, S, "Cu")
    resample(src_lab, SL, "NN")
    resample(base_img, B, "Cu")
    resample(base_lab, BL, "NN")
    SA, SLA, aff = prep / "src_al.nii.gz", prep / "src_lab_al.nii.gz", prep / "src_al.aff12.1D"
    timed(
        ["3dAllineate", "-overwrite", "-base", str(B), "-source", str(S), "-prefix", str(SA),
         "-1Dmatrix_save", str(aff), "-cost", "ls", "-twopass", "-final", "wsinc5"],
        work, log, timeout_s=1800,
    )  # fmt: skip
    timed(
        ["3dAllineate", "-overwrite", "-master", str(B), "-source", str(SL),
         "-1Dmatrix_apply", str(aff), "-final", "NN", "-prefix", str(SLA)],
        work, log,
    )  # fmt: skip

    return S, SL, B, BL, SA, SLA, aff


class QwarpCheck(Check):
    name = "qwarp"
    description = "Nonlinear warp: ffs_qwarp vs 3dQwarp, Dice of transported manual labels"
    needs = [["@ds:labels_cohort"]]
    sampling = "depth"
    reference = "AFNI 3dQwarp"
    timeout_s = 4 * 3600.0  # 3dQwarp at minpatch 9 on a 1 mm brain: 2 h 13 min measured

    def plan(self, manifest, index, n):
        """``n`` ordered pairs, spread so sources and bases repeat as little as possible."""
        cohort = manifest.get("cohort") or []
        ds = manifest["dataset_id"]
        pairs = [(s, b) for s in cohort for b in cohort if s["id"] != b["id"]]
        pairs.sort(key=lambda p: _stable_rank(ds, p[0]["id"], p[1]["id"]))
        picked, used = [], set()
        for s, b in pairs:  # first pass: no subject twice
            if len(picked) < n and s["id"] not in used and b["id"] not in used:
                picked.append((s, b))
                used |= {s["id"], b["id"]}
        for p in pairs:
            if len(picked) < n and p not in picked:
                picked.append(p)
        return [
            ({"acq": manifest.get("cohort_acq", "T1w"), "src": s["id"], "base": b["id"]},
             [s["image"], s["labels"], b["image"], b["labels"]])
            for s, b in picked
        ]  # fmt: skip

    def run(self, key, inputs, work, device):
        import nibabel as nib

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
        sides = {"ref": work / "ref", "ffs": work / "ffs"}
        for d in sides.values():
            d.mkdir(exist_ok=True)
        t = timed_cached(
            ["3dQwarp", "-overwrite", "-base", str(B), "-source", str(SA),
             "-prefix", str(sides["ref"] / "qw.nii.gz"), *QWARP_ARGS],
            sides["ref"] / "fit.timing.json",
            [sides["ref"] / "qw.nii.gz", sides["ref"] / "qw_WARP.nii.gz"],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("ref", "cpu", t, threads))
        t = timed_cached(
            ["ffs_qwarp", "-base", str(B), "-source", str(SA),
             "-prefix", str(sides["ffs"] / "qw"), *QWARP_ARGS, "-device", device],
            sides["ffs"] / "fit.timing.json",
            [sides["ffs"] / "qw.nii.gz", sides["ffs"] / "qw_WARP.nii.gz"],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("ffs", device.split(",")[0], t, ffs_threads(device)))

        def load(p):
            return np.asarray(nib.Nifti1Image.from_filename(str(p)).dataobj)

        base_lab_v = load(BL).astype(np.int32)
        labels = np.unique(base_lab_v)
        labels = labels[labels != 0]
        truth: dict[str, float | None] = {}
        d_aff = _dice_per_label(load(SLA).astype(np.int32), base_lab_v, labels)
        truth["dice_mean_affine"] = float(np.nanmean(d_aff))
        warped = {}
        for side, d in sides.items():
            warp = next((w for w in sorted(d.glob("qw_WARP*.nii*")) if "INV" not in w.name), None)
            if warp is None:
                raise FileNotFoundError(f"no qw_WARP in {d}")
            lab_w = d / "lab_warped.nii.gz"
            timed(
                ["3dNwarpApply", "-overwrite", "-nwarp", str(warp), "-source", str(SLA),
                 "-master", str(B), "-ainterp", "NN", "-prefix", str(lab_w)],
                work, log,
            )  # fmt: skip
            dl = _dice_per_label(load(lab_w).astype(np.int32), base_lab_v, labels)
            truth[f"dice_mean_{side}"] = float(np.nanmean(dl))
            truth[f"dice_median_{side}"] = float(np.nanmedian(dl))
            truth[f"dice_min_{side}"] = float(np.nanmin(dl))
            jac = d / "jac.nii.gz"
            timed(["3dNwarpFuncs", "-overwrite", "-nwarp", str(warp), "-bulk",
                   "-prefix", str(jac)], work, log)  # fmt: skip
            # warps live on each tool's padded grid (3dQwarp 187x243x196, ffs 187x244x196
            # for a 179x210x179 base): score folding on the base grid
            jac_b = d / "jac_base.nii.gz"
            timed(["3dresample", "-overwrite", "-master", str(B), "-rmode", "NN",
                   "-input", str(jac), "-prefix", str(jac_b)], work, log)  # fmt: skip
            j = load(jac_b).astype(np.float64)
            inside = load(B) > 0
            truth[f"fold_frac_{side}"] = float((j[inside] <= -1).mean())  # bulk = det(J) - 1
            warped[side] = next(iter(sorted(d.glob("qw.nii*"))))
        truth["dice_gain_ffs_minus_ref"] = truth["dice_mean_ffs"] - truth["dice_mean_ref"]
        truth["n_labels"] = len(labels)
        res.metrics["truth"] = truth

        mask = load(B) > 0
        a, b = (load(warped[s]).astype(np.float64)[mask] for s in ("ref", "ffs"))
        a, b = a - a.mean(), b - b.mean()
        res.metrics["reference"] = {
            "warped_r": float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())),
            "dice_mean_ref": truth["dice_mean_ref"],
            "dice_mean_ffs": truth["dice_mean_ffs"],
        }

        for d in sides.values():
            # the warps are kept: the reference fit costs hours, and fold/Jacobian
            # judges may change after the night (they did, the first night)
            res.artifacts += [(p, "keep" if "WARP" in p.name else "regenerable")
                              for p in sorted(d.glob("qw*"))]  # fmt: skip
            res.artifacts += [
                (d / "jac.nii.gz", "regenerable"),
                (d / "jac_base.nii.gz", "regenerable"),
            ]
            res.artifacts += [(d / "lab_warped.nii.gz", "keep"), (d / "fit.timing.json", "keep")]
        res.artifacts += [(p, "regenerable") for p in (S, SL, B, SA)]
        res.artifacts += [(BL, "keep"), (SLA, "keep"), (aff, "keep"), (log, "keep")]
        res.provenance["ref"] = {"tool": "3dQwarp", "version": afni_version("3dQwarp"),
                                 "args": QWARP_ARGS}  # fmt: skip
        res.provenance["ffs"] = {"tool": "ffs_qwarp", **ffs_version(), "args": QWARP_ARGS,
                                 "device": device}  # fmt: skip
        res.provenance["inputs_prep"] = {
            "resample": f"3dresample -dxyz {DXYZ} (Cu image, NN labels)",
            "affine": "3dAllineate -cost ls -twopass -final wsinc5, shared by both",
            "label_transport": "3dNwarpApply -ainterp NN for both warps",
        }
        return res
