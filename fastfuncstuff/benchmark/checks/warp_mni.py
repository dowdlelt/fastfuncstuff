"""warp_mni: one subject to AFNI's MNI152_2009_template_SSW, the job @SSwarper was built for.

Unit: one subject of a whole-head labelled cohort (Mindboggle-101: unstripped T1w, a brain
extraction, manual DKT31 cortical labels on the native grid).

Arms:
- @SSwarper end to end (its defaults): its own skull strip, affine and 3dQwarp, on the head.
- antsRegistration rigid + affine + SyN (the antsRegistrationSyN.sh stages) to the template
  brain, from the cohort's own brain extraction.
- ffs as autoproc -do_mni runs it: ffs_allineate with autoproc's MNI affine options, then a
  nonlinear backend: optiwarp_hs (the autoproc default) and qwarp with -type MNI_T1a, and
  formwarp with ANTs' own SyN stage flag for flag (it reproduces antsRegistration's SyN),
  from the same brain extraction as ANTs.
ANTs and ffs share a strip neither made, so their comparison is registration only; @SSwarper
is judged whole, strip included, because that is how it is used.

Per-unit judges: folding and log det J spread of the nonlinear part inside the template
brain mask, Dice of the warped brain with the template mask, template correlation (reported,
not voting: the tools optimise it), time. The truth judge needs the whole cohort and runs
afterwards (``cohort_scores``): every subject's labels carried into template space, then
pairwise Dice between subjects per tool. Consistency alone rewards over-warping; pairwise
agreement of manual labels does not.

The ffs MNI_T1a presets were tuned on NIREP16, so Mindboggle tests them out of sample.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from ..report_spec import Metric, ReportSpec
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
from .warp_pair import (
    FORMWARP_AS_ANTS,
    _ants_version,
    _load,
    deformation_judges,
    label_judges,
)

TEMPLATE = "MNI152_2009_template_SSW.nii.gz"
# autoproc config DEFAULT_OPTS["mni"] / ["mni_nonlin"] and NL_BACKEND_OPTS, as of 2026-11-25
MNI_AFFINE = ["-affine", "-cost", "lpa", "-source_automask", "-autoweight", "-cmass",
              "-tbest", "16", "-final", "wsinc5"]  # fmt: skip
FFS_NL: dict[str, tuple[str, list[str]]] = {
    "optiwarp_hs": (
        "ffs_optiwarp",
        ["-force", "hs", "-metric", "lpa", "-final_interp", "wsinc5", "-type", "MNI_T1a"],
    ),
    "qwarp": (
        "ffs_qwarp",
        ["-optimizer", "gn", "-cost", "lpa", "-final", "wsinc5", "-type", "MNI_T1a"],
    ),
    # ANTs' SyN stage (ANTS_STAGES' last -t), flag for flag in ANTs' units, and no -type
    # preset: this arm is the ANTs-parity engine, not a tuned fit.
    "formwarp": ("ffs_formwarp", [*FORMWARP_AS_ANTS, "-final_interp", "wsinc5"]),
}
ANTS_STAGES = [
    "-t", "Rigid[0.1]", "-m", "MI[{T},{S},1,32,Regular,0.25]", "-c", "[1000x500x250x0,1e-6,10]",
    "-f", "8x4x2x1", "-s", "3x2x1x0vox",
    "-t", "Affine[0.1]", "-m", "MI[{T},{S},1,32,Regular,0.25]", "-c", "[1000x500x250x0,1e-6,10]",
    "-f", "8x4x2x1", "-s", "3x2x1x0vox",
    "-t", "SyN[0.1,3,0]", "-m", "CC[{T},{S},1,4]", "-c", "[100x70x50x20,1e-6,10]",
    "-f", "8x4x2x1", "-s", "3x2x1x0vox",
]  # fmt: skip
TOOLS = ("sswarper", "ants", *FFS_NL)


def _group(sid: str) -> str:
    return sid.rsplit("-", 1)[0]


def template_path() -> Path:
    from fastfuncstuff.cli_utils import find_afni_file

    p = find_afni_file(TEMPLATE)
    if p is None:
        raise FileNotFoundError(f"{TEMPLATE} not found beside the AFNI on PATH")
    return Path(p)


class WarpMniCheck(Check):
    name = "warp_mni"
    description = "Subject to MNI template: @SSwarper / ANTs SyN vs the ffs MNI chain"
    needs = [["@ds:labels_cohort_head"]]
    sampling = "depth"
    reference = "AFNI @SSwarper; antsRegistration rigid+affine+SyN"
    report = ReportSpec(
        summary="Subject T1 to the MNI152 2009 template: @SSwarper and ANTs SyN against the ffs chain "
        "(ffs_allineate, then each ffs warp tool); DKT31 manual labels follow each warp.",
        tools={"sswarper": "@SSwarper", "ants": "ANTs SyN", "qwarp": "ffs_qwarp",
               "optiwarp_hs": "ffs_optiwarp (hs)", "formwarp": "ffs_formwarp"},
        timing={"ref": "@SSwarper", "ref_ants": "ANTs SyN", "ffs_affine": "ffs_allineate",
                "ffs_qwarp": "ffs_qwarp", "ffs_optiwarp_hs": "ffs_optiwarp (hs)",
                "ffs_formwarp": "ffs_formwarp"},
        references=("sswarper", "ants", "ref", "ref_ants"),
        metrics=(
            Metric("label_loss.label_vol_ratio_min_{tool}", "Smallest label kept (volume after warp / after affine)",
                   better="higher", line=0.5, line_label="lost",
                   note="Per subject, the label that shrank most; below half counts as lost."),
            Metric("label_loss.n_labels_lost_{tool}", "Labels lost per subject", better="lower",
                   note="Labels kept below half their post-affine volume."),
            Metric("deformation.frac_det_below_half_{tool}", "Brain squeezed below half its volume (share)",
                   note="Share of brain voxels whose tissue the warp compresses more than 2x."),
            Metric("deformation.template_r_{tool}", "Warped brain, r with the template", better="higher"),
            Metric("deformation.brain_dice_{tool}", "Brain mask Dice with the template", better="higher"),
        ),
        speedup=(("ffs_affine+ffs_qwarp", "ref"), ("ffs_affine+ffs_formwarp", "ref"),
                 ("ffs_affine+ffs_optiwarp_hs", "ref"), ("ffs_affine+ffs_formwarp", "ref_ants")),
        code=("fastfuncstuff.cli.allineate", "fastfuncstuff.cli.qwarp", "fastfuncstuff.cli.optiwarp", "fastfuncstuff.cli.formwarp", "fastfuncstuff.benchmark.checks.warp_mni"),
    )  # fmt: skip
    timeout_s = 4 * 3600.0

    def plan(self, manifest, index, n):
        """``n`` subjects, round-robin over the cohort's groups in a stable hash order, so a
        small set already spans scanners and sites."""
        ds = manifest["dataset_id"]
        by_group: dict[str, list[dict]] = {}
        for s in sorted(manifest.get("cohort") or [], key=lambda s: _stable_rank(ds, s["id"])):
            by_group.setdefault(_group(s["id"]), []).append(s)
        picked: list[dict] = []
        for depth in itertools.count():
            row = [g[depth] for _, g in sorted(by_group.items()) if len(g) > depth]
            if not row or len(picked) >= n:
                break
            picked += row[: n - len(picked)]
        return [
            ({"acq": manifest.get("cohort_acq", "T1w-head"), "subject": s["id"]},
             [s["image"], s["brain"], s["labels"]])
            for s in picked
        ]  # fmt: skip

    def run(self, key, inputs, work, device):
        head, brain, labels = (inputs[r] for r in list(inputs)[:3])
        work.mkdir(parents=True, exist_ok=True)
        log = work / "commands.log"
        prep = work / "prep"
        prep.mkdir(exist_ok=True)
        tpl = template_path()
        T, TM = prep / "tmpl_brain.nii.gz", prep / "tmpl_mask.nii.gz"
        timed(["3dbucket", "-overwrite", "-prefix", str(T), f"{tpl}[0]"], work, log)
        timed(["3dbucket", "-overwrite", "-prefix", str(TM), f"{tpl}[3]"], work, log)
        tmask = _load(TM) > 0
        tvals = _load(T).astype(np.float64)

        res = UnitResult()
        threads = cpu_threads()
        dev = device.split(",")[0]
        per: dict[str, dict[str, float | None]] = {}

        lab_ids = np.unique(_load(labels).astype(np.int32))
        lab_ids = lab_ids[lab_ids != 0]
        loss: dict[str, float | None] = {}

        def judge(tool: str, warped: Path, lab_mni: Path, detj: np.ndarray, lab_aff: Path) -> None:
            j, tsv = label_judges(_load(lab_mni).astype(np.int32), _load(lab_aff).astype(np.int32),
                                  detj, lab_ids)  # fmt: skip
            loss.update({f"{k}_{tool}": v for k, v in j.items()})
            tsv_path = lab_mni.parent / "labels.tsv"
            tsv_path.write_text(tsv)
            res.artifacts += [(tsv_path, "keep"), (lab_aff, "regenerable")]
            w = _load(warped).astype(np.float64)
            m = w > 0.05 * np.percentile(w[w > 0], 99)
            a, b = w[tmask] - w[tmask].mean(), tvals[tmask] - tvals[tmask].mean()
            per[tool] = {
                **deformation_judges(detj, tmask),
                "brain_dice": float(2 * (m & tmask).sum() / (m.sum() + tmask.sum())),
                "template_r": float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())),
            }
            res.artifacts.append((lab_mni, "keep"))

        def afni_detj(d: Path, warp: Path) -> np.ndarray:
            jac, jac_t = d / "jac.nii.gz", d / "jac_tmpl.nii.gz"
            timed(["3dNwarpFuncs", "-overwrite", "-nwarp", str(warp), "-bulk",
                   "-prefix", str(jac)], work, log)  # fmt: skip
            timed(["3dresample", "-overwrite", "-master", str(T), "-rmode", "NN",
                   "-input", str(jac), "-prefix", str(jac_t)], work, log)  # fmt: skip
            res.artifacts += [(jac, "regenerable"), (jac_t, "regenerable")]
            return _load(jac_t).astype(np.float64) + 1.0  # bulk = det J - 1

        def afni_affine_labels(d: Path, aff12: Path) -> Path:
            out = d / "labels_affine_mni.nii.gz"
            timed(["3dAllineate", "-overwrite", "-master", str(T), "-source", str(labels),
                   "-1Dmatrix_apply", str(aff12), "-final", "NN", "-prefix", str(out)], work, log)  # fmt: skip
            return out

        def afni_labels(d: Path, chain: str) -> Path:
            out = d / "labels_mni.nii.gz"
            timed(["3dNwarpApply", "-overwrite", "-nwarp", chain, "-source", str(labels),
                   "-master", str(T), "-ainterp", "NN", "-prefix", str(out)], work, log)  # fmt: skip
            return out

        # -- @SSwarper, end to end on the head ------------------------------------------
        sd, sub = work / "sswarper", "s"
        sd.mkdir(exist_ok=True)
        qq, qwarp, qaff = (
            sd / f"anatQQ.{sub}.nii",
            sd / f"anatQQ.{sub}_WARP.nii",
            sd / f"anatQQ.{sub}.aff12.1D",
        )
        t = timed_cached(
            ["@SSwarper", "-input", str(head), "-base", str(tpl), "-subid", sub, "-odir", str(sd),
             "-extra_qc_off"],
            sd / "fit.timing.json", [qq, qwarp, qaff], work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("ref", "cpu", t, threads))
        judge(
            "sswarper",
            qq,
            afni_labels(sd, f"{qwarp} {qaff}"),
            afni_detj(sd, qwarp),
            afni_affine_labels(sd, qaff),
        )
        # every output timed_cached checks stays, so `work -redo` can refit one ffs tool
        # without paying for the references again
        res.artifacts += [(qwarp, "host"), (qaff, "keep"), (qq, "host"),
                          (sd / "fit.timing.json", "keep")]  # fmt: skip

        # -- ANTs rigid + affine + SyN, from the cohort's brain extraction ----------------
        ad = work / "ants"
        ad.mkdir(exist_ok=True)
        env = {"ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(threads)}
        aw, aaff, awarped = (
            ad / "ants_1Warp.nii.gz",
            ad / "ants_0GenericAffine.mat",
            ad / "ants_warped.nii.gz",
        )
        stages = [a.format(T=T, S=brain) for a in ANTS_STAGES]
        t = timed_cached(
            ["antsRegistration", "-d", "3", "-o", f"[{ad / 'ants_'},{awarped}]", "-n", "Linear",
             "-u", "0", "-z", "1", "-w", "[0.005,0.995]", "-r", f"[{T},{brain},1]", *stages],
            ad / "fit.timing.json", [aw, aaff, awarped], work, log,
            timeout_s=self.timeout_s, env=env,
        )  # fmt: skip
        res.timings.append(("ref_ants", "cpu", t, threads))
        alab, ajac = ad / "labels_mni.nii.gz", ad / "jac.nii.gz"
        timed(["antsApplyTransforms", "-d", "3", "-i", str(labels), "-r", str(T), "-o", str(alab),
               "-n", "NearestNeighbor", "-t", str(aw), "-t", str(aaff)], work, log, env=env)  # fmt: skip
        timed(["CreateJacobianDeterminantImage", "3", str(aw), str(ajac), "0", "0"],
              work, log, env=env)  # fmt: skip
        alab_aff = ad / "labels_affine_mni.nii.gz"
        timed(["antsApplyTransforms", "-d", "3", "-i", str(labels), "-r", str(T), "-o", str(alab_aff),
               "-n", "NearestNeighbor", "-t", str(aaff)], work, log, env=env)  # fmt: skip
        judge("ants", awarped, alab, _load(ajac).astype(np.float64), alab_aff)
        res.artifacts += [(aw, "host"), (aaff, "keep"), (awarped, "host"),
                          (ajac, "regenerable"), (ad / "ants_1InverseWarp.nii.gz", "regenerable"),
                          (ad / "fit.timing.json", "keep")]  # fmt: skip

        # -- ffs: autoproc's MNI affine, then each backend with -type MNI_T1a --------------
        fd = work / "ffs_affine"
        fd.mkdir(exist_ok=True)
        al, aff = fd / "brain_al.nii.gz", fd / "brain_al.aff12.1D"
        t_aff = timed_cached(
            ["ffs_allineate", "-base", str(T), "-source", str(brain), "-prefix", str(al),
             "-1Dmatrix_save", str(aff), *MNI_AFFINE, "-device", device],
            fd / "fit.timing.json", [al, aff], work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("ffs_affine", dev, t_aff, ffs_threads(device)))
        res.artifacts += [(aff, "keep"), (al, "host"), (fd / "fit.timing.json", "keep")]
        for tool, (cli, args) in FFS_NL.items():
            d = work / f"ffs_{tool}"
            d.mkdir(exist_ok=True)
            prefix = d / "nl" if cli == "ffs_qwarp" else d / "nl.nii.gz"
            warped, warp = d / "nl.nii.gz", d / "nl_WARP.nii.gz"
            extra = [] if cli == "ffs_qwarp" else ["-save_warp"]
            t = timed_cached(
                [cli, "-base", str(T), "-source", str(al), "-prefix", str(prefix), *args,
                 *extra, "-device", device],
                d / "fit.timing.json", [warped, warp], work, log, timeout_s=self.timeout_s,
            )  # fmt: skip
            # the user's cost is affine + nonlinear; the affine is shared, so add it here
            res.timings.append((f"ffs_{tool}", dev, t + t_aff, ffs_threads(device)))
            judge(
                tool,
                warped,
                afni_labels(d, f"{warp} {aff}"),
                afni_detj(d, warp),
                afni_affine_labels(d, aff),
            )
            res.artifacts += [
                (warp, "host"),
                (warped, "host"),
                (d / "fit.timing.json", "keep"),
            ]

        res.metrics["deformation"] = {f"{k}_{t}": v for t, m in per.items() for k, v in m.items()}
        res.metrics["label_loss"] = loss
        res.metrics["reference"] = {f"brain_dice_{t}": per[t]["brain_dice"] for t in per}
        res.artifacts += [(T, "regenerable"), (TM, "regenerable"), (log, "keep")]
        res.provenance["ref"] = {"tool": "@SSwarper", "version": afni_version("3dQwarp"),
                                 "template": str(tpl), "args": ["-extra_qc_off"]}  # fmt: skip
        res.provenance["ref_ants"] = {"tool": "antsRegistration", "version": _ants_version(),
                                      "stages": ANTS_STAGES, "input": "cohort brain extraction"}  # fmt: skip
        res.provenance["ffs"] = {"affine": MNI_AFFINE, "nonlinear": {k: [c, *a]
                                 for k, (c, a) in FFS_NL.items()}, **ffs_version(), "device": device}  # fmt: skip
        return res


def cohort_scores(unit_dirs: list[Path], tools: tuple[str, ...] = TOOLS) -> dict[str, dict]:
    """Pairwise Dice of the subjects' manual labels in template space, per tool.

    ``unit_dirs`` are finished warp_mni units of one cohort. For each tool, every pair of
    subjects scores the mean Dice over the labels both carry; reports the mean, median and
    5th percentile over pairs, and the number of subjects."""
    out: dict[str, dict] = {}
    folder = {"sswarper": "sswarper", "ants": "ants", **{t: f"ffs_{t}" for t in FFS_NL}}
    for tool in tools:
        labs = [_load(u / folder[tool] / "labels_mni.nii.gz").astype(np.int32)
                for u in unit_dirs if (u / folder[tool] / "labels_mni.nii.gz").exists()]  # fmt: skip
        if len(labs) < 2:
            continue
        # Per-label masks over the full template made this hours on one core (100 labels x
        # 4 passes x 8.5M voxels per pair). Instead: compact label indices on the voxels any
        # subject labels, per-subject counts once, and per pair one bincount of the voxels
        # where both carry the same label. Same Dice, ~1000x fewer passes.
        ids = np.unique(np.concatenate([np.unique(x) for x in labs]))
        ids = ids[ids != 0]
        n = len(ids)
        keep = np.zeros(labs[0].shape, bool)
        for x in labs:
            keep |= x != 0
        idx = [np.searchsorted(ids, x[keep]).astype(np.int32) for x in labs]
        for x, k in zip(labs, idx, strict=True):
            k[x[keep] == 0] = n  # background gets its own bin, dropped below
        del labs, keep
        vols = np.stack([np.bincount(k, minlength=n + 1)[:n] for k in idx]).astype(np.float64)
        pair = []
        for i, j in itertools.combinations(range(len(idx)), 2):
            a, b = idx[i], idx[j]
            inter = np.bincount(a[a == b], minlength=n + 1)[:n]
            common = (vols[i] > 0) & (vols[j] > 0)
            d = 2.0 * inter[common] / (vols[i][common] + vols[j][common])
            pair.append(float(d.mean()) if d.size else float("nan"))
        p = np.asarray(pair)
        # a tool that squeezes a structure shrinks it in every subject: keep each label's
        # median template-space volume (compare tools label by label) and its spread
        med = np.median(vols, axis=0)
        cv = vols.std(axis=0) / np.maximum(vols.mean(axis=0), 1)
        out[tool] = {"label_vol_cv_median": float(np.median(cv)),
                     "label_vol_median_by_label": {str(int(i)): float(v) for i, v in zip(ids, med, strict=True)}}  # fmt: skip
        out[tool] |= {"n_subjects": len(idx), "pair_dice_mean": float(p.mean()),
                     "pair_dice_median": float(np.median(p)), "pair_dice_p05": float(np.percentile(p, 5))}  # fmt: skip
    return out


def cohort_units(wl, root: Path, dataset: str, exclude_hosts: tuple[str, ...] = ()) -> list[Path]:
    """Finished warp_mni units of ``dataset`` here and on every ingested host set, the latest
    run of each unit (the same rule as the report), skipping hosts labelled ``exclude_hosts``."""
    from ..worklist import Worklist

    q = ("SELECT id, finished_at FROM units WHERE check_name='warp_mni' AND dataset=?"
         " AND status='done'")  # fmt: skip
    roots = [Path(root)] + [Path(h["path"]) for h in wl.db.execute(
        "SELECT label, path FROM hostsets ORDER BY ingested_at") if h["label"] not in exclude_hosts]  # fmt: skip
    best: dict[int, tuple[float, Path]] = {}
    for r in roots:
        db = wl if r == Path(root) else Worklist(r / "worklist.sqlite")
        for u in db.db.execute(q, (dataset,)).fetchall():
            if u["id"] not in best or (u["finished_at"] or 0) > best[u["id"]][0]:
                best[u["id"]] = (
                    u["finished_at"] or 0,
                    r / "units" / "warp_mni" / dataset / f"unit{u['id']:06d}",
                )
        if db is not wl:
            db.close()
    return [best[i][1] for i in sorted(best)]


def write_cohort(root: Path, dataset: str, units: list[Path]) -> Path:
    """Score the finished unit folders ``units`` of ``dataset`` and write the result under
    ROOT (the units may live on several ingested host sets)."""
    import time

    scores = cohort_scores(units)
    out = (
        Path(root) / "units" / "warp_mni" / dataset / f"cohort_{time.strftime('%Y%m%d-%H%M')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    rel = [str(u.relative_to(root)) if u.is_relative_to(root) else str(u) for u in units]
    out.write_text(json.dumps({"units": rel, "scores": scores}, indent=1))
    return out
