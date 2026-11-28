"""segment: ffs_segment against SPM25 unified segmentation (spm_preproc8) on T1w images.

ffs_segment is a port of SPM's New Segment, so SPM with its defaults and the same TPM
($FFS_TPM, SPM's tpm/TPM.nii) is the reference. Both write native-space posteriors
c1..c3 (GM, WM, CSF) on the T1's own grid.

Judges:
- reference: posterior r per class inside the head (either tool's c1+c2+c3 > 0.5), and
  Dice of the hard labels (argmax over c1..c3, "none" below 0.5) per class.
- truth, where the dataset has manual tissue labels on the T1 grid (ds005216 UltraCortex,
  9.4T MP2RAGE: FreeSurfer codes, 3/42 cortex and 2/41 white matter, nothing else labelled).
  Per tool: Dice of hard GM vs cortex and WM vs white matter over the whole volume, and
  the agreement on voxels the raters labelled. Whole-volume GM Dice counts subcortical
  grey matter (unlabelled by the raters) as a miss for every tool alike. The in-label
  agreement is free of that, so it is the headline.
"""

from __future__ import annotations

import gzip
import os
import shutil
from pathlib import Path

import numpy as np

from .. import census
from ..report_spec import Metric, ReportSpec
from .base import (
    Check,
    UnitResult,
    _stable_rank,
    cpu_threads,
    ffs_threads,
    ffs_version,
    timed_cached,
)

MATLAB = "/usr/local/MATLAB/R2025a/bin/matlab"
SPM_DIR = Path.home() / "Dropbox/Resources/code/matlab_toolboxes/spm25"
# dataset -> where its manual labels live, and the label codes per tissue
TRUTH = {
    "ds005216": {
        "dir": "derivatives/manual_segmentation",
        "name": "sub-{sub}_ses-{ses}_seg.nii",
        "gm": (3, 42),
        "wm": (2, 41),
    }
}


def hard_labels(post: list[np.ndarray], floor: float = 0.5) -> np.ndarray:
    """0 = none, 1..K = argmax class where the summed posterior reaches ``floor``."""
    p = np.stack(post)
    lab = p.argmax(0) + 1
    lab[p.sum(0) < floor] = 0
    return lab


def dice(a: np.ndarray, b: np.ndarray) -> float | None:
    n = a.sum() + b.sum()
    return float(2 * (a & b).sum() / n) if n else None


def truth_scores(lab: np.ndarray, manual: np.ndarray, spec: dict) -> dict[str, float | None]:
    gm_t, wm_t = np.isin(manual, spec["gm"]), np.isin(manual, spec["wm"])
    labelled = gm_t | wm_t
    agree = ((lab == 1) & gm_t) | ((lab == 2) & wm_t)
    return {
        "gm_dice": dice(lab == 1, gm_t),
        "wm_dice": dice(lab == 2, wm_t),
        "in_label_agreement": float(agree.sum() / labelled.sum()) if labelled.any() else None,
        "gm_dice_in_label": dice((lab == 1) & labelled, gm_t),
        "wm_dice_in_label": dice((lab == 2) & labelled, wm_t),
    }


def spm_script(t1: Path, tpm: str, threads: int) -> str:
    tis = " ".join(
        f"matlabbatch{{1}}.spm.spatial.preproc.tissue({k}).tpm = {{'{tpm},{k}'}}; "
        f"matlabbatch{{1}}.spm.spatial.preproc.tissue({k}).ngaus = {g}; "
        f"matlabbatch{{1}}.spm.spatial.preproc.tissue({k}).native = [{int(k <= 3)} 0]; "
        f"matlabbatch{{1}}.spm.spatial.preproc.tissue({k}).warped = [0 0];"
        for k, g in zip(range(1, 7), (1, 1, 2, 3, 4, 2), strict=True)
    )
    return (
        f"maxNumCompThreads({threads}); addpath('{SPM_DIR}'); spm('defaults','fmri'); "
        "spm_jobman('initcfg'); "
        f"matlabbatch{{1}}.spm.spatial.preproc.channel.vols = {{'{t1},1'}}; "
        "matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001; "
        "matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60; "
        "matlabbatch{1}.spm.spatial.preproc.channel.write = [0 0]; "
        + tis
        + " matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1; "
        "matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1; "
        "matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2]; "
        "matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni'; "
        "matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0; "
        "matlabbatch{1}.spm.spatial.preproc.warp.samp = 3; "
        "matlabbatch{1}.spm.spatial.preproc.warp.write = [0 0]; "
        "spm_jobman('run', matlabbatch);"
    )


class SegmentCheck(Check):
    name = "segment"
    description = "Tissue segmentation: ffs_segment vs SPM25 (spm_preproc8); manual labels as truth"
    needs = [["@ds:anat_t1"]]
    sampling = "depth"
    reference = "SPM25 unified segmentation"
    report = ReportSpec(
        summary="Unified segmentation of a T1w: ffs_segment against SPM25, and both against "
        "manual labels.",
        tools={"ref": "SPM25", "ffs": "ffs_segment"},
        timing={"ref": "SPM25", "ffs": "ffs_segment"},
        references=("ref",),
        metrics=(
            Metric("truth.gm_dice_{tool}", "GM Dice with the manual labels", better="higher"),
            Metric("truth.wm_dice_{tool}", "WM Dice with the manual labels", better="higher"),
            Metric("reference.gm_dice", "GM Dice with SPM", better="higher"),
            Metric("reference.csf_dice", "CSF Dice with SPM", better="higher"),
        ),
        speedup=(("ffs", "ref"),),
        code=("fastfuncstuff.cli.segment", "fastfuncstuff.benchmark.checks.segment"),
    )  # fmt: skip
    timeout_s = 3 * 3600.0

    def qualifies(self, manifest, acq):
        return "anat_t1" in manifest.get("offers", [])

    def plan(self, manifest, index, n):
        """Up to ``n`` subjects' first T1w (one session each); subjects with manual labels
        first, then a stable hash order."""
        if "anat_t1" not in manifest.get("offers", []):
            return []
        ds = manifest["dataset_id"]
        t1s: dict[tuple, census.BidsName] = {}
        cands = [
            nm
            for nm in index.names
            if nm.datatype == "anat" and nm.suffix == "T1w" and nm.ext in census._IMAGE_EXT
        ]
        # the plain T1w first; acq-/rec- variants (MP2RAGE parts, defaced copies) only if alone
        for nm in sorted(cands, key=lambda nm: (len(nm.entities), nm.rel)):
            t1s.setdefault((nm.ent.get("sub"), nm.ent.get("ses")), nm)
        truth = self._truth_files(manifest)
        order = sorted(t1s, key=lambda k: (self._truth_rel(ds, *k) not in truth,
                                           _stable_rank(ds, "segment", *map(str, k))))  # fmt: skip
        picked, seen = [], set()
        for k in order:
            if len(picked) < n and k[0] not in seen:
                picked.append(k)
                seen.add(k[0])
        units = []
        for k in picked:
            nm = t1s[k]
            rels = [nm.rel] + [m.rel for m in index.matching(nm, ".json")]
            tr = self._truth_rel(ds, *k)
            if tr in truth:
                rels.append(tr)
            units.append(({"acq": "anat-T1w", "sub": k[0], "ses": k[1]}, rels))
        return units

    @staticmethod
    def _truth_rel(ds: str, sub: str, ses: str | None) -> str | None:
        spec = TRUTH.get(ds)
        if not spec:
            return None
        return f"{spec['dir']}/" + str(spec["name"]).format(sub=sub, ses=ses)

    @staticmethod
    def _truth_files(manifest: dict) -> set[str]:
        spec = TRUTH.get(manifest["dataset_id"])
        if not spec:
            return set()
        from ..fetch import S3Source, source_from_manifest

        src = source_from_manifest(manifest)
        if isinstance(src, S3Source):  # derivatives are left out of list_files
            files, _ = src._list(f"{src.prefix}{spec['dir']}/")
            return {f"{spec['dir']}/{k.rsplit('/', 1)[-1]}" for k in files}
        root = src.root / spec["dir"]
        return {f"{spec['dir']}/{p.name}" for p in root.glob("*")} if root.is_dir() else set()

    def run(self, key, inputs, work, device):
        import nibabel as nib

        tpm = os.environ.get("FFS_TPM")
        if not tpm:
            raise RuntimeError("FFS_TPM (SPM's tpm/TPM.nii) is needed")
        imgs = [r for r in inputs if r.endswith((".nii", ".nii.gz"))]
        truth_rel = next((r for r in imgs if r.startswith("derivatives/")), None)
        t1_rel = next(r for r in imgs if r != truth_rel)
        work.mkdir(parents=True, exist_ok=True)
        log = work / "commands.log"
        res = UnitResult()

        # SPM reads plain .nii and writes c1<name>.nii beside it: its own uncompressed copy
        ref = work / "ref"
        ref.mkdir(exist_ok=True)
        t1 = ref / "t1.nii"
        if not t1.exists():
            src = inputs[t1_rel]
            if str(src).endswith(".gz"):
                with gzip.open(src, "rb") as fi, open(t1, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            else:
                shutil.copyfile(src, t1)
        threads = cpu_threads()
        matlab = os.environ.get("FFS_MATLAB", MATLAB)
        ref_c = [ref / f"c{k}t1.nii" for k in (1, 2, 3)]
        t_ref = timed_cached([matlab, "-batch", spm_script(t1, tpm, threads)],
                             ref / "stamp.json", ref_c, ref, log, timeout_s=self.timeout_s)  # fmt: skip
        res.timings.append(("ref", "cpu", t_ref, threads))

        ffs = work / "ffs"
        ffs.mkdir(exist_ok=True)
        prefix = ffs / "t1"
        cmd = ["ffs_segment", "-input", str(inputs[t1_rel]), "-tpm", tpm, "-prefix", str(prefix),
               "-no_invwarp", "-device", device]  # fmt: skip
        ffs_c = [Path(f"{prefix}_c{k}.nii.gz") for k in (1, 2, 3)]
        t = timed_cached(cmd, ffs / "stamp.json", ffs_c, work, log, timeout_s=self.timeout_s)
        res.timings.append(("ffs", device.split(",")[0], t, ffs_threads(device)))

        load = lambda p: np.asarray(nib.Nifti1Image.from_filename(str(p)).dataobj, np.float32)  # noqa: E731
        pr, pf = [load(p) for p in ref_c], [load(p) for p in ffs_c]
        if pr[0].shape != pf[0].shape:
            raise ValueError(f"grids differ: SPM {pr[0].shape}, ffs {pf[0].shape}")
        head = (sum(pr) > 0.5) | (sum(pf) > 0.5)
        lr, lf = hard_labels(pr), hard_labels(pf)
        ml = float(np.prod(nib.Nifti1Image.from_filename(str(t1)).header.get_zooms()[:3])) / 1e3
        m: dict[str, float | None] = {}
        for k, name in enumerate(("gm", "wm", "csf")):
            m[f"{name}_r"] = float(np.corrcoef(pr[k][head], pf[k][head])[0, 1])
            m[f"{name}_dice"] = dice(lr == k + 1, lf == k + 1)
            m[f"{name}_ml_ref"] = float(pr[k].sum()) * ml
            m[f"{name}_ml_ffs"] = float(pf[k].sum()) * ml
        res.metrics["reference"] = m

        if truth_rel:
            spec = next(s for s in TRUTH.values() if truth_rel.startswith(s["dir"]))
            manual = np.asarray(nib.Nifti1Image.from_filename(str(inputs[truth_rel])).dataobj)
            if manual.shape != lr.shape:
                raise ValueError(f"manual labels {manual.shape} not on the T1 grid {lr.shape}")
            tr: dict[str, float | None] = {}
            for side, lab in (("ref", lr), ("ffs", lf)):
                tr.update({f"{k}_{side}": v for k, v in truth_scores(lab, manual, spec).items()})
            res.metrics["truth"] = tr
            res.provenance["truth"] = {"labels": truth_rel, "gm_codes": spec["gm"], "wm_codes": spec["wm"],
                                       "note": "raters labelled cortex and white matter only"}  # fmt: skip

        for p in ref_c + ffs_c:
            res.artifacts.append((p, "regenerable"))
        res.artifacts += [(t1, "regenerable"), (log, "keep")]
        res.provenance["ref"] = {"tool": "SPM25 spm_preproc8 (batch defaults)", "spm": str(SPM_DIR),
                                 "tpm": tpm, "threads": threads}  # fmt: skip
        res.provenance["ffs"] = {"tool": "ffs_segment", **ffs_version(), "device": device,
                                 "args": ["-no_invwarp"]}  # fmt: skip
        return res
