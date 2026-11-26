"""allineate: ffs_allineate against AFNI's EPI-to-anatomy recipe, mean EPI to the T1w.

Unit: one run. Frozen inputs: the run's mean (3dTstat) and the subject's T1w stripped
once by ``ffs_segment -brain`` (bias-corrected, inside GM+WM+CSF > 0.95). Not
3dSkullStrip: on a 7T T1 (ds004645) it removed the cerebellum, occipital and inferior
temporal lobes, where the slab EPI sat.

Reference: align_epi_anat.py as afni_proc.py runs it (anat2epi, EPI strip by 3dAutomask,
-volreg off -tshift off, ``-anat_has_skull no`` as after @SSwarper) with the options
afni_proc's help recommends for EPI to anatomy, ``-cost lpc+ZZ -giant_move``. On an EPI
that covers only part of the brain the plain centre-of-mass shift drags the slab to the
brain's middle, so the matching ``-partial_*`` flag limits cmass to the covered axes
(AFNI's own advice; see ``partial_axes``). Its EPI-to-anatomy matrix is applied to the
mean with wsinc5 on the T1 grid, as ffs writes its own output. A bare 3dAllineate with
the arguments ffs gets (lpc, source automask, its defaults) runs too, for speed parity
and continuity with earlier runs; the judges score all three. ffs_allineate runs at its
defaults, since the defaults are what users get.

Judge (reference): RMS displacement in mm between ffs's and the reference's affines over
the stripped T1's brain voxels, and the correlation of the two aligned means. ffs_allineate's
search is stochastic (CMA-ES), so a unit is one draw: report the distribution over
units, not a per-unit pass.

Judge (yardstick): each tool's aligned mean scored against the stripped T1 by one
program (3dAllineate -allcostX -nocmass, no source automask: with it ls is 0.52 for an
image against itself). Histogram costs (hel, nmi, mi) are the comparison, since neither
tool optimises them; AFNI's allcostX lpc/lpa/ls are recorded too (lpc reads normally on
real pairs, -0.54 for 3dAllineate's fit on the first unit, but +24 for an image against
itself, so not for self-pairs). Lower is better except mi (AFNI prints it unsigned).

Judge (truth, tissue boundary): none of the above sees a fit that moves the data off the
brain. allcostX scores only the voxels the two images share, so on the ds004645 slab it
preferred 3dAllineate's fit (-0.17) with the slab off the brain over ffs's on-brain fit
(-0.07). Two checks that need no cost either tool optimises: the share of the aligned
EPI's bright voxels inside the T1 brain (dilated 3 mm), and the GM-WM contrast of the
aligned EPI across the T1's WM boundary (c2 = 0.5), sampled 1.5 mm either side along the
boundary normal. T2*-weighted GM is brighter than WM, so a right fit reads positive and a
wrong one near zero; only boundary points inside the EPI's field of view count, so slabs
are judged where they are.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import numpy as np

from .base import (
    Check,
    UnitResult,
    afni_version,
    cpu_threads,
    ffs_threads,
    ffs_version,
    timed,
    timed_cached,
)

ARGS = ["-cost", "lpc", "-source_automask", "-final", "wsinc5"]

# align_epi_anat.py as afni_proc.py writes it (db_mod.py, align block) plus the options
# its help recommends for EPI to anatomy; the -partial_* flag is added per unit.
AEA_ARGS = ["-anat2epi", "-epi2anat", "-epi_base", "0", "-epi_strip", "3dAutomask",
            "-anat_has_skull", "no", "-volreg", "off", "-tshift", "off",
            "-cost", "lpc+ZZ", "-giant_move", "-suffix", "_al"]  # fmt: skip

# An EPI axis shorter than this share of the brain's extent along the same world axis
# counts as partial. Measured on the benchmark's EPIs (11-25): whole-brain runs span
# 119-149 mm in I-S, slabs 30-104 mm (HiHiFaces, ds004645); a T1 brain is ~140 mm.
PARTIAL_FRAC = 0.8
PARTIAL_FLAG = {"z": "-partial_axial", "y": "-partial_coronal", "x": "-partial_sagittal"}


YARD_COSTS = ("hel", "nmi", "mi", "lpc", "lpa", "ls")

# Tissue-judge verdict. Set on 10 fits known right or wrong by eye (ds002785, ds006926,
# ds004645 slabs; 11-24): right fits read in-brain 0.87-0.97 and positive-contrast share
# 0.64-0.85, wrong ones 0.00-0.20 and ~0.50 (or too few boundary points to score).
IN_BRAIN_MIN = 0.7
POS_FRAC_MIN = 0.6


def parse_allcost(text: str) -> dict[str, float]:
    """``name = value`` lines of 3dAllineate -allcostX output."""
    out = {}
    for ln in text.splitlines():
        parts = ln.split("=")
        if len(parts) == 2 and parts[0].strip() in YARD_COSTS:
            out[parts[0].strip()] = float(parts[1])
    return out


def read_aff12(path) -> np.ndarray:
    """4x4 from the last row of an AFNI .aff12.1D (DICOM/RAI, base -> source)."""
    rows = [ln.split() for ln in open(path) if ln.strip() and not ln.lstrip().startswith("#")]
    m = np.eye(4)
    m[:3, :] = np.asarray(rows[-1][:12], dtype=float).reshape(3, 4)
    return m


def epi_in_brain(epi: np.ndarray, brain: np.ndarray, vox_mm, dilate_mm: float = 3.0) -> float:
    """Share of the aligned EPI's bright voxels (> 30% of its 98th percentile) inside
    ``brain`` dilated by ``dilate_mm`` (scalp and dura sit just outside a strip)."""
    from scipy import ndimage

    pos = epi[epi > 0]
    if pos.size == 0:
        return float("nan")
    bright = epi > 0.3 * np.percentile(pos, 98)
    it = max(1, round(dilate_mm / float(np.mean(vox_mm))))
    grown = ndimage.binary_dilation(brain, iterations=it)
    return float((bright & grown).sum() / max(1, bright.sum()))


def boundary_contrast(
    epi: np.ndarray,
    wm: np.ndarray,
    brain: np.ndarray,
    vox_mm,
    d_mm: float = 1.5,
    max_pts: int = 200_000,
) -> dict[str, float]:
    """GM-WM contrast of ``epi`` across the WM boundary of ``wm`` (posterior, same grid).

    Per boundary point: (gm - wm) / mean(gm, wm) of the EPI sampled ``d_mm`` outside and
    inside along the c2 gradient. Points whose samples fall outside the EPI's field of
    view (zero) are dropped. Returns the median contrast, the share of positive points
    and the point count (a slab that misses the brain has few).
    """
    from scipy import ndimage

    vox = np.asarray(vox_mm, dtype=float)
    g = np.stack(np.gradient(ndimage.gaussian_filter(wm.astype(np.float32), 1.0)), 0)
    g /= vox[:, None, None, None]  # per mm
    pts = np.argwhere((wm > 0.4) & (wm < 0.6) & brain)
    if len(pts) > max_pts:
        pts = pts[:: len(pts) // max_pts + 1]
    gp = g[:, pts[:, 0], pts[:, 1], pts[:, 2]].T
    norm = np.linalg.norm(gp, axis=1)
    keep = norm > 1e-6
    pts, n = pts[keep], gp[keep] / norm[keep, None]  # n points into WM
    step = d_mm * n / vox  # voxels
    out_wm = ndimage.map_coordinates(epi, (pts + step).T, order=1, mode="constant")
    out_gm = ndimage.map_coordinates(epi, (pts - step).T, order=1, mode="constant")
    ok = (out_wm > 0) & (out_gm > 0)
    if ok.sum() < 100:
        return {"contrast": float("nan"), "pos_frac": float("nan"), "n_pts": float(ok.sum())}
    c = (out_gm[ok] - out_wm[ok]) / (0.5 * (out_gm[ok] + out_wm[ok]))
    return {"contrast": float(np.median(c)), "pos_frac": float((c > 0).mean()),
            "n_pts": float(ok.sum())}  # fmt: skip


def _is_inversion(name) -> bool:
    return "inv" in name.ent or bool(re.fullmatch(r"inv-?\d+", name.ent.get("acq", "").lower()))


def partial_axes(epi_affine, epi_shape, brain_affine, brain_mask, frac=PARTIAL_FRAC) -> str:
    """World axes ("x", "y", "z", RAS order) along which the EPI's field of view is shorter
    than ``frac`` of the brain's extent.

    Each EPI voxel axis is assigned to the world axis it points along most (an oblique slab
    is still partial along its own short axis, whose world bounding box is inflated by the
    tilt); its length is compared to the brain mask's world extent on that axis."""
    from nibabel.affines import apply_affine

    a = np.asarray(epi_affine, dtype=float)[:3, :3]
    vox = np.sqrt((a**2).sum(0))
    ijk = np.argwhere(brain_mask)
    w = apply_affine(brain_affine, ijk)
    bvox = np.sqrt((np.asarray(brain_affine)[:3, :3] ** 2).sum(0))
    brain_ext = w.max(0) - w.min(0) + bvox.mean()
    out = set()
    for i in range(3):
        ax = int(np.argmax(np.abs(a[:, i])))
        if vox[i] * epi_shape[i] < frac * brain_ext[ax]:
            out.add("xyz"[ax])
    return "".join(sorted(out))


def partial_flag(axes: str) -> list[str]:
    """align_epi_anat.py's flag for ``partial_axes``: one partial axis has its own flag
    (cmass then moves only along the covered axes), more than one leaves AFNI to judge."""
    if not axes:
        return []
    return [PARTIAL_FLAG[axes]] if len(axes) == 1 else ["-partial_coverage"]


def rms_displacement(m1: np.ndarray, m2: np.ndarray, pts_rai: np.ndarray) -> float:
    d = (m1 - m2)[:3] @ np.c_[pts_rai, np.ones(len(pts_rai))].T
    return float(np.sqrt((d**2).sum(0).mean()))


class AllineateCheck(Check):
    name = "allineate"
    description = "Affine EPI->T1w: ffs_allineate vs align_epi_anat.py (and 3dAllineate)"
    needs = [["@ds:anat_t1"]]
    sampling = "breadth"
    reference = "AFNI align_epi_anat.py (lpc+ZZ -giant_move, -partial_* per unit)"

    def unit_inputs(self, files, index):
        mags = [f for f in files if f.ent.get("part") != "phase" and f.suffix != "phase"]
        first = sorted(mags, key=lambda f: int(f.ent.get("echo", "1")))[0]
        sub, ses = first.ent.get("sub"), first.ent.get("ses")
        t1s = [
            n
            for n in index.names
            if n.datatype == "anat" and n.suffix == "T1w" and n.ent.get("sub") == sub
            and n.ext in (".nii", ".nii.gz")
        ]  # fmt: skip
        if not t1s:
            return []
        # An MP2RAGE inversion saved as T1w (acq-inv1, Forest) is proton-density-like, not
        # the anatomy anyone aligns to; without this it wins the tie alphabetically over uni.
        t1s.sort(key=lambda n: (_is_inversion(n), n.ent.get("ses") != ses, len(n.entities), n.rel))
        return [first.rel, t1s[0].rel]

    def plan(self, manifest, index, n):
        return [(k, i) for k, i in super().plan(manifest, index, n) if len(i) == 2]

    def run(self, key, inputs, work, device):
        import nibabel as nib
        from nibabel.affines import apply_affine

        epi, t1 = (inputs[r] for r in list(inputs)[:2])
        work.mkdir(parents=True, exist_ok=True)
        log = work / "commands.log"
        mean = work / "epi_mean.nii.gz"
        seg = work / "seg" / "t1"
        brain, wm = Path(f"{seg}_brain.nii.gz"), Path(f"{seg}_c2.nii.gz")
        res = UnitResult()
        epi, _ = self.trim_noise(epi, work, log, res)
        timed(["3dTstat", "-overwrite", "-mean", "-prefix", str(mean), str(epi)], work, log)
        tpm = os.environ.get("FFS_TPM")
        if not tpm:
            raise RuntimeError("FFS_TPM (SPM's tpm/TPM.nii) is needed for the T1 strip")
        seg.parent.mkdir(exist_ok=True)
        timed(["ffs_segment", "-input", str(t1), "-tpm", tpm, "-prefix", str(seg), "-brain",
               "-no_invwarp", "-device", device], work, log, timeout_s=1800)  # fmt: skip

        threads = cpu_threads()
        out = {s: work / s for s in ("ref", "ffs", "3dallineate")}
        for d in out.values():
            d.mkdir(exist_ok=True)

        b = nib.Nifti1Image.from_filename(str(brain))
        mask = np.asarray(nib.Nifti1Image.from_filename(f"{seg}_brainmask.nii.gz").dataobj) > 0
        e = nib.Nifti1Image.from_filename(str(mean))
        axes = partial_axes(e.affine, e.shape[:3], b.affine, mask)
        aea_args = [*AEA_ARGS, *partial_flag(axes)]
        # align_epi_anat names its outputs after the inputs' prefixes, in -output_dir.
        # Cached (minutes per unit), so `work -redo allineate -tools ffs` refits ffs alone.
        t = timed_cached(
            ["align_epi_anat.py", "-overwrite", "-anat", str(brain), "-epi", str(mean),
             "-output_dir", str(out["ref"]), *aea_args],
            out["ref"] / "fit.timing.json", [out["ref"] / "epi_mean_al_mat.aff12.1D"],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        shutil.copy(out["ref"] / "epi_mean_al_mat.aff12.1D", out["ref"] / "al.aff12.1D")
        t += timed(
            ["3dAllineate", "-overwrite", "-base", str(brain), "-source", str(mean),
             "-1Dmatrix_apply", str(out["ref"] / "al.aff12.1D"), "-master", str(brain),
             "-final", "wsinc5", "-prefix", str(out["ref"] / "al.nii.gz")],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("ref", "cpu", t, threads))
        t = timed(
            ["3dAllineate", "-overwrite", "-base", str(brain), "-source", str(mean),
             "-prefix", str(out["3dallineate"] / "al.nii.gz"),
             "-1Dmatrix_save", str(out["3dallineate"] / "al.aff12.1D"), *ARGS],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("3dallineate", "cpu", t, threads))
        t = timed(
            ["ffs_allineate", "-base", str(brain), "-source", str(mean),
             "-prefix", str(out["ffs"] / "al.nii.gz"),
             "-1Dmatrix_save", str(out["ffs"] / "al.aff12.1D"), *ARGS, "-device", device],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("ffs", device.split(",")[0], t, ffs_threads(device)))

        ijk = np.argwhere(mask)[:: max(1, int(mask.sum() // 200000))]
        ras = apply_affine(b.affine, ijk)
        rai = ras * np.array([-1.0, -1.0, 1.0])
        mats = {s: read_aff12(out[s] / "al.aff12.1D") for s in out}
        imgs = {s: np.asarray(nib.Nifti1Image.from_filename(str(out[s] / "al.nii.gz")).dataobj)
                for s in out}  # fmt: skip
        m_ref, m_ffs = mats["ref"], mats["ffs"]
        a, c = imgs["ref"], imgs["ffs"]
        x, y = a[mask].astype(float), c[mask].astype(float)
        x, y = x - x.mean(), y - y.mean()
        res.metrics["reference"] = {
            "affine_rms_mm": rms_displacement(m_ref, m_ffs, rai),
            "aligned_r": float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())),
            "ref_rms_from_identity_mm": rms_displacement(m_ref, np.eye(4), rai),
            "ffs_rms_from_identity_mm": rms_displacement(m_ffs, np.eye(4), rai),
            # isotropic scale; ffs failures so far sit on its 0.8 bound (ds002603, ds004645)
            "scale_ref": float(abs(np.linalg.det(m_ref[:3, :3])) ** (1 / 3)),
            "scale_ffs": float(abs(np.linalg.det(m_ffs[:3, :3])) ** (1 / 3)),
            # the earlier reference, kept for continuity: how far plain 3dAllineate lands
            "3dallineate_rms_from_ref_mm": rms_displacement(mats["3dallineate"], m_ref, rai),
            "ffs_rms_from_3dallineate_mm": rms_displacement(m_ffs, mats["3dallineate"], rai),
        }
        # How much of the problem the EPI poses: a thin slab or partial field of view is the
        # hard case for both tools, so failures are read against coverage.
        vox = np.sqrt((e.affine[:3, :3] ** 2).sum(0))
        ext = vox * np.asarray(e.shape[:3])
        res.metrics["data"] = {
            "epi_extent_min_mm": float(ext.min()), "epi_extent_max_mm": float(ext.max()),
            "epi_n_min_axis": float(min(e.shape[:3])), "epi_voxel_max_mm": float(vox.max()),
            # share of the T1 brain inside the EPI's field of view under each fit
            "brain_coverage_ref": float((np.abs(a[mask]) > 0).mean()),
            "brain_coverage_ffs": float((np.abs(c[mask]) > 0).mean()),
            # axes align_epi_anat was told are partial (bit per axis: x=1, y=2, z=4)
            "partial_axes": float(sum(1 << "xyz".index(k) for k in axes)),
        }  # fmt: skip
        yard: dict[str, float | None] = {}
        for s in out:
            ylog = out[s] / "allcost.txt"
            timed(["3dAllineate", "-base", str(brain), "-source", str(out[s] / "al.nii.gz"),
                   "-cost", "lpc", "-nocmass", "-allcostX"], out[s], ylog)  # fmt: skip
            for k, v in parse_allcost(ylog.read_text()).items():
                yard[f"{k}_{s}"] = v
        for k in ("hel", "nmi", "mi"):
            for other in ("ref", "3dallineate"):
                if f"{k}_{other}" in yard and f"{k}_ffs" in yard:
                    yard[f"{k}_ffs_minus_{other}"] = yard[f"{k}_ffs"] - yard[f"{k}_{other}"]
        res.metrics["yardstick"] = yard

        # tissue judge: needs no cost either tool optimises (see the module docstring)
        wmv = np.asarray(nib.Nifti1Image.from_filename(str(wm)).dataobj, dtype=np.float32)
        bvox = np.sqrt((b.affine[:3, :3] ** 2).sum(0))
        truth: dict[str, float] = {}
        for s, img in imgs.items():
            e_al = np.asarray(img, dtype=np.float32)
            truth[f"epi_in_brain_{s}"] = epi_in_brain(e_al, mask, bvox)
            for k, v in boundary_contrast(e_al, wmv, mask, bvox).items():
                truth[f"wm_{k}_{s}"] = v
            pf = truth[f"wm_pos_frac_{s}"]
            truth[f"aligned_{s}"] = float(
                truth[f"epi_in_brain_{s}"] > IN_BRAIN_MIN and pf == pf and pf > POS_FRAC_MIN
            )
        res.metrics["truth"] = truth
        res.artifacts += [(out[s] / "allcost.txt", "keep") for s in out]
        res.artifacts += [(out[s] / "al.nii.gz", "regenerable") for s in out]
        res.artifacts += [(out[s] / "al.aff12.1D", "keep") for s in out]
        res.artifacts += [(mean, "keep"), (log, "keep")]
        # the cached reference fit, so a -redo of ffs reuses it on pruned host sets too
        res.artifacts += [
            (out["ref"] / f, "keep") for f in ("epi_mean_al_mat.aff12.1D", "fit.timing.json")
        ]
        res.artifacts += [(f, "regenerable") for f in sorted(seg.parent.glob("*.nii.gz"))]
        res.provenance["ref"] = {"tool": "align_epi_anat.py", "version": afni_version("3dAllineate"),
                                 "args": aea_args, "apply": "3dAllineate -1Dmatrix_apply -final wsinc5"}  # fmt: skip
        res.provenance["3dallineate"] = {"tool": "3dAllineate",
                                         "version": afni_version("3dAllineate"), "args": ARGS}  # fmt: skip
        res.provenance["ffs"] = {"tool": "ffs_allineate", **ffs_version(), "args": ARGS,
                                 "device": device}  # fmt: skip
        res.provenance["inputs_prep"] = {
            "epi": "3dTstat -mean",
            "t1": "ffs_segment -brain (bias-corrected, GM+WM+CSF > 0.95)",
            "tpm": tpm,
        }
        return res
