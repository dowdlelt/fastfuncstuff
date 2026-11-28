"""locomoco: ffs_locomoco against per-volume ANTs SyN, on high-resolution 2-D EPI.

No tool does per-volume nonlinear motion correction, so the reference is built by hand
(wiki: Benchmark overhaul plan, "locomoco: per-volume SyN as the reference"):

1. A fixed block of ``N_VOLUMES`` volumes from the run's first steady-state volume is
   motion corrected once with ffs_moco (shared by both sides, untimed), and its temporal
   median is the target for both.
2. ffs side: ``ffs_locomoco -ref median`` on the moco'd block.
3. Reference side: split into 3-D volumes, ``antsRegistration`` SyN of each volume to the
   median with the deformation restricted to the phase-encode axis (``-g``, the physical
   axis nearest the voxel PE axis): locomoco's degrees of freedom exactly, so the judges
   compare fits, not models. Apply, merge. Timed in full, split and merge included.

The question (user, 2026-11-27): does the fast method correct the data the way the
best-in-class slow one does? There is no ground truth, so the headline judges compare
locomoco with ANTs directly, in the brain:
- agreement: the estimated PE displacements correlated volume by volume (median and worst
  volume; pooled r; median |difference|; each tool's median |displacement|); the
  *correction* each applied (corrected minus moco'd) correlated volume by volume; and
  every voxel's corrected time series correlated between the two tools; and the
  displacements again with both smoothed by ``SMOOTH_VOX`` (locomoco's field is rougher).
- intrinsic, suggestive and a little circular (the median is also the target), scored
  identically for the moco'd block (``moco``), ANTs and locomoco: each volume's spatial r
  with the median (mean, worst), median tSNR, mean-image sharpness; and whether the two
  tools improve the *same* volumes (r across volumes of each tool's gain in that r).

On an oblique slab the nearest physical axis is only approximately the PE axis; ANTs
restricts physical (LPS) components, not voxel axes.
"""

from __future__ import annotations

import time

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

N_VOLUMES = 40
MAX_ISO_MM = 1.8  # "high resolution": the in-plane distortion locomoco targets is a few voxels

MOCO_ARGS = ["-interp", "heptic", "-final", "heptic", "-weight_automask"]
FFS_ARGS = ["-ref", "median", "-no_movie", "-no_flow"]
# Same-contrast EPI to its own median: CC radius 2, three levels, modest SyN step.
# Fixed and recorded in provenance; deliberately not tuned against locomoco. The output
# is resampled with windowed sinc (ffs_locomoco's final resample is wsinc5): ANTs'
# default linear blurs, which lifts its tSNR and lowers its sharpness for free.
ANTS_ARGS = [
    "-d", "3", "--float", "1", "-v", "0", "-n", "LanczosWindowedSinc",
    "-t", "SyN[0.2,3,0]",
    "-c", "[50x30x10,1e-6,5]", "-f", "4x2x1", "-s", "2x1x0vox",
]  # fmt: skip
PE_AXIS = {"i": 0, "j": 1, "k": 2}
# The two fits differ most in fine detail: ffs_locomoco's field is rougher than SyN's
# (MindsEye 0.8 mm, 2026-11-27: raw per-volume r 0.38, 0.69 with both smoothed by 2 voxels,
# 0.74 at 4). Smoothing both alike compares the shape of the correction, not its texture.
SMOOTH_VOX = 2.0


def _rowwise_r(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r between matching rows of two (n, m) arrays (nan where one is constant)."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis=1) / den, np.nan)


def _agreement(
    d_ants: np.ndarray, d_ffs: np.ndarray, moco: np.ndarray, ants: np.ndarray, ffs: np.ndarray
) -> dict[str, float | None]:
    """locomoco vs ANTs. Displacements are (n_vox, t) PE mm; series (n_vox, t) intensities."""
    a, b = d_ants.ravel(), d_ffs.ravel()
    pooled = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    r_vol = _rowwise_r(d_ants.T, d_ffs.T)
    r_corr = _rowwise_r((ants - moco).T, (ffs - moco).T)
    r_ts = _rowwise_r(ants, ffs)

    def med(x: np.ndarray) -> float | None:
        return float(np.nanmedian(x)) if np.isfinite(x).any() else None

    def low(x: np.ndarray) -> float | None:
        return float(np.nanmin(x)) if np.isfinite(x).any() else None

    return {
        "pe_r_vol_median": med(r_vol),
        "pe_r_vol_min": low(r_vol),
        "pe_r": pooled,
        "pe_absdiff_median_mm": float(np.median(np.abs(a - b))),
        "pe_abs_median_mm_ants": float(np.median(np.abs(a))),
        "pe_abs_median_mm_ffs": float(np.median(np.abs(b))),
        "correction_r_vol_median": med(r_corr),
        "correction_r_vol_min": low(r_corr),
        "ts_r_median": med(r_ts),
    }


def _smoothed_pe_r(d_ants: np.ndarray, d_ffs: np.ndarray, mask: np.ndarray) -> float | None:
    """Median per-volume r of two (x, y, z, t) PE fields, both smoothed by ``SMOOTH_VOX``."""
    from scipy.ndimage import gaussian_filter

    s = (SMOOTH_VOX, SMOOTH_VOX, SMOOTH_VOX, 0.0)
    a = gaussian_filter(d_ants, s)[mask]
    b = gaussian_filter(d_ffs, s)[mask]
    r = _rowwise_r(a.T, b.T)
    return float(np.nanmedian(r)) if np.isfinite(r).any() else None


def score(
    moco: np.ndarray,
    ants: np.ndarray | None,
    ffs: np.ndarray,
    median: np.ndarray,
    mask: np.ndarray,
    d_ants: np.ndarray | None,
    d_ffs: np.ndarray,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Both judges from (x, y, z, t) series and PE displacement fields: (intrinsic, agreement).
    Without the reference (an ffs-only run, ``ants`` None) only the intrinsic judges of
    moco and ffs remain."""
    intr: dict[str, float | None] = {}
    vol_r = {}
    sides = (
        (("moco", moco), ("ants", ants), ("ffs", ffs))
        if ants is not None
        else (("moco", moco), ("ffs", ffs))
    )
    for tag, series in sides:
        for k, v in _intrinsic(series, median, mask).items():
            if k == "_vol_r":
                vol_r[tag] = v
            else:
                intr[f"{k}_{tag}"] = v
    if ants is None or d_ants is None:
        return intr, {}
    gain_a, gain_f = vol_r["ants"] - vol_r["moco"], vol_r["ffs"] - vol_r["moco"]
    same = _rowwise_r(gain_a[None], gain_f[None])[0]
    intr["same_volumes_r"] = float(same) if np.isfinite(same) else None
    agree = _agreement(d_ants[mask], d_ffs[mask], moco[mask], ants[mask], ffs[mask])
    agree["pe_r_vol_median_smooth"] = _smoothed_pe_r(d_ants, d_ffs, mask)
    return intr, agree


def _intrinsic(series: np.ndarray, median: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """tSNR, volume-to-median r and mean-image sharpness of a (x, y, z, t) series."""
    v = series[mask]  # (n_vox, t)
    mean = v.mean(axis=1)
    sd = v.std(axis=1)
    tsnr = np.median(mean / np.maximum(sd, 1e-6))
    m = median[mask]
    mc = m - m.mean()
    vc = v - v.mean(axis=0, keepdims=True)
    r = (mc @ vc) / (np.linalg.norm(mc) * np.linalg.norm(vc, axis=0) + 1e-12)
    mean_img = series.mean(axis=3)
    gx, gy, gz = np.gradient(mean_img)
    grad = np.sqrt(gx * gx + gy * gy + gz * gz)[mask]
    return {
        "_vol_r": r,  # per volume, for the same-volumes judge; not stored
        "tsnr_median": float(tsnr),
        "vol_r_mean": float(r.mean()),
        "vol_r_min": float(r.min()),
        "sharpness": float(grad.mean() / max(float(mean_img[mask].mean()), 1e-6)),
    }


class LocomocoCheck(Check):
    name = "locomoco"
    description = "Residual nonlinear motion: ffs_locomoco vs per-volume ANTs SyN (PE axis)"
    needs = [["epi_2d"]]
    sampling = "breadth"
    reference = "ANTs antsRegistration SyN per volume, deformation restricted to the PE axis"
    timeout_s = 4 * 3600.0
    ffs_only_ok = True  # ffs-only keeps the intrinsic judges; agreement needs ANTs
    report = ReportSpec(
        summary=f"{N_VOLUMES} volumes of high-resolution 2-D EPI, moco'd once and shared: "
        "ffs_locomoco against antsRegistration SyN per volume (PE axis only), both to the "
        "block's temporal median.",
        tools={"moco": "moco only", "ants": "ANTs SyN", "ffs": "ffs_locomoco"},
        timing={"ref": "ANTs per volume", "ffs": "ffs_locomoco"},
        references=("ref",),
        metrics=(
            Metric("agreement.pe_r_vol_median", "PE displacement, locomoco vs ANTs, r per volume "
                   "(median)", better="higher",
                   note="Spatial r of the two estimated PE displacement fields, volume by volume."),
            Metric("agreement.pe_r_vol_median_smooth", "PE displacement, locomoco vs ANTs, r per "
                   f"volume, both smoothed {SMOOTH_VOX:g} vox (median)", better="higher",
                   note="The shape of the two corrections; locomoco's field is rougher than SyN's."),
            Metric("agreement.correction_r_vol_median", "Correction applied, locomoco vs ANTs, "
                   "r per volume (median)", better="higher",
                   note="Corrected minus moco'd image, correlated between the tools per volume."),
            Metric("agreement.ts_r_median", "Corrected time series, locomoco vs ANTs, r per voxel "
                   "(median)", better="higher"),
            Metric("agreement.pe_absdiff_median_mm", "PE displacement, median |difference| (mm)",
                   log=True),
            Metric("intrinsic.vol_r_mean_{tool}", "Volume-to-median r, mean", better="higher",
                   note="Suggestive only: the median is also both tools' target."),
            Metric("intrinsic.same_volumes_r", "Do both tools improve the same volumes? (r)",
                   better="higher",
                   note="r across volumes of each tool's gain in volume-to-median r over moco."),
            Metric("intrinsic.tsnr_median_{tool}", "tSNR, median over the brain", better="higher"),
        ),
        speedup=(("ffs", "ref"),),
        code=("fastfuncstuff.cli.locomoco", "fastfuncstuff.benchmark.checks.locomoco"),
    )  # fmt: skip

    def qualifies(self, manifest, acq):
        vox = acq.get("voxel_mm") or []
        if len(vox) != 3 or (vox[0] * vox[1] * vox[2]) ** (1 / 3) >= MAX_ISO_MM:
            return False
        return super().qualifies(manifest, acq)

    def unit_inputs(self, files, index):
        mags = [f for f in files if f.ent.get("part") != "phase" and f.suffix != "phase"]
        first = sorted(mags, key=lambda f: int(f.ent.get("echo", "1")))[0]
        return [first.rel] + [m.rel for m in index.matching(first, ".json")]

    def run(self, key, inputs, work, device):
        import nibabel as nib

        image = next(p for rel, p in inputs.items() if not rel.endswith(".json"))
        meta = self.read_meta(inputs)
        ped = str(meta.get("PhaseEncodingDirection") or "")
        if not ped or ped[0] not in PE_AXIS:
            raise ValueError("no PhaseEncodingDirection in the run's sidecars")
        pe_vox = PE_AXIS[ped[0]]

        work.mkdir(parents=True, exist_ok=True)
        ffs, ants = work / "ffs", work / "ants"
        ffs.mkdir(exist_ok=True)
        ants.mkdir(exist_ok=True)
        log = work / "commands.log"
        res = UnitResult()
        src, _ = self.trim_noise(image, work, log, res)

        means = volume_means(src)
        start = first_steady_volume(means)
        stop = min(start + N_VOLUMES, len(means)) - 1
        if stop - start < 9:
            raise ValueError(f"only {stop - start + 1} steady volumes")
        block = work / "block.nii.gz"
        timed(["3dTcat", "-overwrite", "-prefix", str(block), f"{src}[{start}..{stop}]"],
              work, log)  # fmt: skip
        moco = work / "moco.nii.gz"
        timed(["ffs_moco", *MOCO_ARGS, "-base", "0", "-input", str(block), "-prefix", str(moco),
               "-device", device], work, log, timeout_s=self.timeout_s)  # fmt: skip
        median = work / "median.nii.gz"
        timed(["3dTstat", "-overwrite", "-median", "-prefix", str(median), str(moco)], work, log)
        mask_p = work / "mask.nii.gz"
        timed(["3dAutomask", "-overwrite", "-prefix", str(mask_p), str(median)], work, log)
        res.provenance["block"] = {
            "first": start, "last": stop, "n": stop - start + 1, "moco_args": MOCO_ARGS,
            "rule": "first steady-state volume (within 5% of the run median mean) onward",
        }  # fmt: skip

        # ffs side
        stem = ffs / "lm"
        t_ffs = timed(
            ["ffs_locomoco", "-input", str(moco), "-prefix", str(stem) + ".nii.gz",
             "-pe_dir", "xyz"[pe_vox], *FFS_ARGS, "-device", device],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("ffs", device.split(",")[0], t_ffs, ffs_threads(device)))
        ffs_corr = ffs / "lm_locomoco.nii.gz"
        ffs_warp = ffs / "lm_warp.nii.gz"

        img = nib.load(str(moco))
        # The physical (LPS) component nearest the voxel PE axis, chosen as AFNI's
        # deobliqued (cardinal) frame chooses it -- the frame locomoco writes its warp
        # in. ANTs' restriction and the agreement judge both use it; RAS and LPS share
        # the axis order.
        from ...processing.nwarpforge import compute_cardinal_affine

        comp = int(np.argmax(np.abs(compute_cardinal_affine(img.affine)[:3, pe_vox])))
        restrict = "x".join("1" if c == comp else "0" for c in range(3))
        data = np.asarray(img.dataobj, dtype=np.float32)
        n_t = data.shape[3]

        # reference side: split, register each volume, merge -- all of it timed
        ants_data = d_ants = None
        vols: list = []
        if self.with_reference:
            t0 = time.monotonic()
            for t in range(n_t):
                p = ants / f"v{t:04d}.nii.gz"
                nib.save(nib.Nifti1Image(data[..., t], img.affine, img.header), str(p))
                vols.append(p)
            split_s = time.monotonic() - t0
            reg_s = 0.0
            for t, p in enumerate(vols):
                out = ants / f"v{t:04d}_"
                reg_s += timed(
                    ["antsRegistration", *ANTS_ARGS, "-g", restrict, "-m", f"CC[{median},{p},1,2]",
                     "-o", f"[{out},{out}w.nii.gz]"],
                    work, log, timeout_s=self.timeout_s,
                )  # fmt: skip
            t1 = time.monotonic()
            ants_data = np.stack(
                [np.asarray(nib.load(str(ants / f"v{t:04d}_w.nii.gz")).dataobj, np.float32)
                 for t in range(n_t)], axis=3,
            )  # fmt: skip
            ants_corr = ants / "corrected.nii.gz"
            nib.save(nib.Nifti1Image(ants_data, img.affine, img.header), str(ants_corr))
            merge_s = time.monotonic() - t1
            res.timings.append(("ref", "cpu", split_s + reg_s + merge_s, cpu_threads()))
            # PE displacement in that component. Both fields are pull warps in LPS mm (ITK
            # and DICOM agree on the axes).
            d_ants = np.stack(
                [np.asarray(nib.load(str(ants / f"v{t:04d}_0Warp.nii.gz")).dataobj,
                            np.float32).reshape(*data.shape[:3], 3)[..., comp]
                 for t in range(n_t)], axis=3,
            )  # fmt: skip
            res.artifacts += [(ants_corr, "regenerable"), *((p, "regenerable") for p in vols)]
            res.artifacts += [(p, "regenerable") for p in ants.glob("v*_*")]
            res.provenance["ref"] = {
                "tool": "antsRegistration", "args": [*ANTS_ARGS, "-g", restrict],
                "metric": "CC[median, volume, 1, 2]",
                "seconds": {"split": split_s, "register": reg_s, "merge": merge_s},
            }  # fmt: skip

        # judges
        med = np.asarray(nib.load(str(median)).dataobj, np.float32)
        mask = np.asarray(nib.load(str(mask_p)).dataobj) > 0
        lm = np.asarray(nib.load(str(ffs_corr)).dataobj, np.float32)
        w = np.asarray(nib.load(str(ffs_warp)).dataobj, np.float32)
        d_ffs = w.reshape(*data.shape[:3], n_t, 3)[..., comp]
        intr, agree = score(data, ants_data, lm, med, mask, d_ants, d_ffs)
        res.metrics["intrinsic"] = intr
        if agree:
            res.metrics["agreement"] = agree

        for p in (block, moco, ffs_corr, ffs_warp):
            res.artifacts.append((p, "regenerable"))
        for p in (median, mask_p, log):
            res.artifacts.append((p, "keep"))
        res.provenance["ffs"] = {
            "tool": "ffs_locomoco", **ffs_version(), "args": FFS_ARGS, "device": device,
            "pe_dir": ped,
        }  # fmt: skip
        return res
