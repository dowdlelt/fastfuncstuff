"""fc: ffs_fc against AFNI 3dNetCorr on the same ROI set.

The frozen input is the raw first-echo magnitude (trailing noise volumes trimmed), its
3dAutomask, and a grid parcellation: cubes of about CUBE_MM on the EPI grid inside the
mask. Nobody's atlas, so nothing is resampled and both tools see identical labels.

3dNetCorr averages raw intensities and correlates without detrending. The parity variant
(``ffs``: ``-no_scale -polort -1``) must match it to rounding (the .netcc holds 4
decimals). The default variant (``ffs_default``: percent-signal scaling plus per-run
Legendre drift) answers a different question on purpose. Its distance from the parity
matrix is reported so the cost of 3dNetCorr's choices can be read, not scored as a
disagreement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Check, UnitResult, afni_version, cpu_threads, ffs_threads, ffs_version, timed

CUBE_MM = 15.0
MIN_ROI_VOXELS = 10
FFS_VARIANTS = {"ffs": ["-no_scale", "-polort", "-1"], "ffs_default": []}


def grid_rois(mask: np.ndarray, vox_mm: tuple[float, ...], cube_mm: float = CUBE_MM) -> np.ndarray:
    """Integer labels 1..K: cubes of ~cube_mm tiling the mask; cubes under
    MIN_ROI_VOXELS voxels are dropped and the survivors renumbered in order."""
    step = [max(2, round(cube_mm / v)) for v in vox_mm[:3]]
    idx = np.indices(mask.shape)
    cell = sum((idx[a] // step[a]) * int(np.prod([mask.shape[b] // step[b] + 1 for b in range(a + 1, 3)]))
               for a in range(3))  # fmt: skip
    cell = np.where(mask, cell + 1, 0)
    ids, counts = np.unique(cell[cell > 0], return_counts=True)
    keep = ids[counts >= MIN_ROI_VOXELS]
    lut = np.zeros(int(cell.max()) + 1, dtype=np.int32)
    lut[keep] = np.arange(1, keep.size + 1)
    return lut[cell]


def read_netcc(path: Path, block: str = "CC") -> np.ndarray:
    """The ``# CC`` (or ``# FZ``) matrix of a 3dNetCorr-format .netcc file."""
    rows: list[list[float]] = []
    on = False
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if s.startswith("#"):
            if on and rows:
                break
            on = s.lstrip("#").strip().split()[:1] == [block]
            continue
        if on and s:
            rows.append([float(x) for x in s.split()])
    if not rows:
        raise ValueError(f"no '# {block}' matrix in {path}")
    return np.asarray(rows)


def matrix_agreement(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    iu = np.triu_indices_from(a, k=1)
    x, y = a[iu], b[iu]
    return {"max_abs_diff": float(np.abs(x - y).max()), "mean_abs_diff": float(np.abs(x - y).mean()),
            "r": float(np.corrcoef(x, y)[0, 1])}  # fmt: skip


class FcCheck(Check):
    name = "fc"
    description = "Functional connectivity: ffs_fc vs 3dNetCorr on a grid parcellation"
    needs = [["rest"]]
    sampling = "breadth"
    reference = "AFNI 3dNetCorr"

    def unit_inputs(self, files, index):
        mags = [f for f in files if f.ent.get("part") != "phase" and f.suffix != "phase"]
        first = sorted(mags, key=lambda f: int(f.ent.get("echo", "1")))[0]
        return [first.rel] + [m.rel for m in index.matching(first, ".json")]

    def run(self, key, inputs, work, device):
        import nibabel as nib

        src = next(p for r, p in inputs.items() if r.endswith((".nii", ".nii.gz")))
        work.mkdir(parents=True, exist_ok=True)
        log = work / "commands.log"
        res = UnitResult()
        src, _ = self.trim_noise(src, work, log, res)

        mask_p, rois_p = work / "mask.nii.gz", work / "rois.nii.gz"
        timed(["3dAutomask", "-overwrite", "-prefix", str(mask_p), str(src)], work, log)
        mimg = nib.Nifti1Image.from_filename(str(mask_p))
        mask = np.asarray(mimg.dataobj) > 0
        rois = grid_rois(mask, mimg.header.get_zooms())
        if rois.max() < 3:
            raise ValueError(f"only {rois.max()} grid ROIs in the mask")
        nib.save(nib.Nifti1Image(rois.astype(np.int32), mimg.affine), rois_p)

        ref = work / "ref"
        ref.mkdir(exist_ok=True)
        threads = cpu_threads()
        t_ref = timed(["3dNetCorr", "-overwrite", "-inset", str(src), "-in_rois", str(rois_p),
                       "-mask", str(mask_p), "-fish_z", "-prefix", str(ref / "net")],
                      work, log, timeout_s=self.timeout_s)  # fmt: skip
        res.timings.append(("ref", "cpu", t_ref, threads))
        ref_netcc = next(ref.glob("net*.netcc"))
        cc_ref = read_netcc(ref_netcc)

        cc = {}
        for side, extra in FFS_VARIANTS.items():
            d = work / side
            d.mkdir(exist_ok=True)
            t = timed(["ffs_fc", "-input", str(src), "-atlas", str(rois_p), "-mask", str(mask_p),
                       "-prefix", str(d / "fc"), *extra, "-device", device, "-quiet"],
                      work, log, timeout_s=self.timeout_s)  # fmt: skip
            res.timings.append((side, device.split(",")[0], t, ffs_threads(device)))
            cc[side] = np.load(d / "fc_fc.npz")["CC"]
        if cc["ffs"].shape != cc_ref.shape:
            raise ValueError(f"ROI count differs: ref {cc_ref.shape}, ffs {cc['ffs'].shape}")

        m: dict[str, float | None] = {
            f"cc_{k}": v for k, v in matrix_agreement(cc_ref, cc["ffs"]).items()
        }
        m["n_rois"] = float(cc_ref.shape[0])
        fz_ref = read_netcc(ref_netcc, "FZ")
        m["fz_max_abs_diff"] = matrix_agreement(
            fz_ref, np.arctanh(np.clip(cc["ffs"], -0.999999, 0.999999))
        )["max_abs_diff"]
        res.metrics["reference"] = m
        res.metrics["intrinsic"] = {f"default_vs_parity_{k}": v
                                    for k, v in matrix_agreement(cc["ffs"], cc["ffs_default"]).items()}  # fmt: skip

        res.artifacts += [(rois_p, "keep"), (mask_p, "keep"), (ref_netcc, "keep"), (log, "keep")]
        for side in FFS_VARIANTS:
            res.artifacts += [
                (work / side / "fc_fc.npz", "keep"),
                (work / side / "fc.json", "keep"),
            ]
        res.provenance["ref"] = {"tool": "3dNetCorr", "version": afni_version("3dNetCorr"),
                                 "args": ["-fish_z"]}  # fmt: skip
        res.provenance["ffs"] = {"tool": "ffs_fc", **ffs_version(), "device": device,
                                 "variants": FFS_VARIANTS}  # fmt: skip
        res.provenance["inputs_prep"] = {"rois": f"grid cubes ~{CUBE_MM:g} mm in 3dAutomask, "
                                         f">= {MIN_ROI_VOXELS} voxels", "n_rois": int(rois.max())}  # fmt: skip
        return res
