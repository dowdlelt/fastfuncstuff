"""allineate: ffs_allineate against 3dAllineate, mean EPI to the subject's T1w (lpc).

Unit: one run. Frozen inputs: the run's mean (3dTstat) and the subject's T1w stripped
once by 3dSkullStrip, given to both tools with identical arguments (lpc, source automask)
and each tool's default centring, since the defaults are what users get.

Judge (reference): RMS displacement in mm between the two fitted affines over the
stripped T1's brain voxels, and the correlation of the two aligned means. ffs_allineate's
search is stochastic (CMA-ES), so a unit is one draw: report the distribution over
units, not a per-unit pass.

Judge (yardstick): each tool's aligned mean scored against the stripped T1 by one
program (3dAllineate -allcostX -nocmass, no source automask: with it ls is 0.52 for an
image against itself). Histogram costs (hel, nmi, mi) are the comparison, since neither
tool optimises them; AFNI's allcostX lpc/lpa/ls are recorded too (lpc reads normally on
real pairs, -0.54 for 3dAllineate's fit on the first unit, but +24 for an image against
itself, so not for self-pairs). Lower is better except mi (AFNI prints it unsigned).
"""

from __future__ import annotations

import os

import numpy as np

from .base import Check, UnitResult, afni_version, ffs_version, timed

ARGS = ["-cost", "lpc", "-source_automask", "-final", "wsinc5"]


YARD_COSTS = ("hel", "nmi", "mi", "lpc", "lpa", "ls")


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


def rms_displacement(m1: np.ndarray, m2: np.ndarray, pts_rai: np.ndarray) -> float:
    d = (m1 - m2)[:3] @ np.c_[pts_rai, np.ones(len(pts_rai))].T
    return float(np.sqrt((d**2).sum(0).mean()))


class AllineateCheck(Check):
    name = "allineate"
    description = "Affine EPI->T1w: ffs_allineate vs 3dAllineate (lpc), same stripped T1"
    needs = [["@ds:anat_t1"]]
    sampling = "breadth"
    reference = "AFNI 3dAllineate"

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
        t1s.sort(key=lambda n: (n.ent.get("ses") != ses, len(n.entities), n.rel))
        return [first.rel, t1s[0].rel]

    def plan(self, manifest, index, n):
        return [(k, i) for k, i in super().plan(manifest, index, n) if len(i) == 2]

    def run(self, key, inputs, work, device):
        import nibabel as nib
        from nibabel.affines import apply_affine

        epi, t1 = (inputs[r] for r in list(inputs)[:2])
        work.mkdir(parents=True, exist_ok=True)
        log = work / "commands.log"
        mean, brain = work / "epi_mean.nii.gz", work / "t1_brain.nii.gz"
        timed(["3dTstat", "-overwrite", "-mean", "-prefix", str(mean), str(epi)], work, log)
        timed(["3dSkullStrip", "-overwrite", "-input", str(t1), "-prefix", str(brain)],
              work, log, timeout_s=1800)  # fmt: skip

        res = UnitResult()
        threads = int(os.environ.get("OMP_NUM_THREADS", "0")) or os.cpu_count()
        out = {s: work / s for s in ("ref", "ffs")}
        for d in out.values():
            d.mkdir(exist_ok=True)
        t = timed(
            ["3dAllineate", "-overwrite", "-base", str(brain), "-source", str(mean),
             "-prefix", str(out["ref"] / "al.nii.gz"),
             "-1Dmatrix_save", str(out["ref"] / "al.aff12.1D"), *ARGS],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("ref", "cpu", t, threads))
        t = timed(
            ["ffs_allineate", "-base", str(brain), "-source", str(mean),
             "-prefix", str(out["ffs"] / "al.nii.gz"),
             "-1Dmatrix_save", str(out["ffs"] / "al.aff12.1D"), *ARGS, "-device", device],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("ffs", device.split(",")[0], t, None))

        b = nib.Nifti1Image.from_filename(str(brain))
        vol = np.asarray(b.dataobj)
        ijk = np.argwhere(vol > 0)[:: max(1, int((vol > 0).sum() // 200000))]
        ras = apply_affine(b.affine, ijk)
        rai = ras * np.array([-1.0, -1.0, 1.0])
        m_ref = read_aff12(out["ref"] / "al.aff12.1D")
        m_ffs = read_aff12(out["ffs"] / "al.aff12.1D")
        a = np.asarray(nib.Nifti1Image.from_filename(str(out["ref"] / "al.nii.gz")).dataobj)
        c = np.asarray(nib.Nifti1Image.from_filename(str(out["ffs"] / "al.nii.gz")).dataobj)
        mask = vol > 0
        x, y = a[mask].astype(float), c[mask].astype(float)
        x, y = x - x.mean(), y - y.mean()
        res.metrics["reference"] = {
            "affine_rms_mm": rms_displacement(m_ref, m_ffs, rai),
            "aligned_r": float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())),
            "ref_rms_from_identity_mm": rms_displacement(m_ref, np.eye(4), rai),
        }
        yard: dict[str, float | None] = {}
        for s in out:
            ylog = out[s] / "allcost.txt"
            timed(["3dAllineate", "-base", str(brain), "-source", str(out[s] / "al.nii.gz"),
                   "-cost", "lpc", "-nocmass", "-allcostX"], out[s], ylog)  # fmt: skip
            for k, v in parse_allcost(ylog.read_text()).items():
                yard[f"{k}_{s}"] = v
        for k in ("hel", "nmi", "mi"):
            if f"{k}_ref" in yard and f"{k}_ffs" in yard:
                yard[f"{k}_ffs_minus_ref"] = yard[f"{k}_ffs"] - yard[f"{k}_ref"]
        res.metrics["yardstick"] = yard
        res.artifacts += [(out[s] / "allcost.txt", "keep") for s in out]
        res.artifacts += [(out[s] / "al.nii.gz", "regenerable") for s in out]
        res.artifacts += [(out[s] / "al.aff12.1D", "keep") for s in out]
        res.artifacts += [(mean, "keep"), (brain, "regenerable"), (log, "keep")]
        res.provenance["ref"] = {"tool": "3dAllineate", "version": afni_version("3dAllineate"),
                                 "args": ARGS}  # fmt: skip
        res.provenance["ffs"] = {"tool": "ffs_allineate", **ffs_version(), "args": ARGS,
                                 "device": device}  # fmt: skip
        res.provenance["inputs_prep"] = {"epi": "3dTstat -mean", "t1": "3dSkullStrip"}
        return res
