"""nordic: ffs_nordic against MATLAB NIFTI_NORDIC on magnitude + phase runs.

The frozen input is the raw first-echo magnitude and its phase. Both tools get the same
trailing noise-volume count: curated, else detected (``trailing_noise_count``). Unlike the
other checks, the noise volumes are kept, because NORDIC estimates thermal noise from them.

Judges:
- reference: the median per-voxel time-series r of the denoised signal volumes, and the
  g-factor map r. Settings are those of the old stage (stages/nordic.py), so the numbers
  read on the fastbench scale.
- intrinsic: median tSNR in the automask (signal volumes) for the raw run, the reference
  and each ffs variant. It is scored the same way for every side. tSNR alone cannot say
  which denoiser is right (removing signal also raises tSNR), but a large gap between the
  tools is a finding.

MATLAB holds the complex series in double, about 16 bytes x voxels x volumes several times
over. A run whose estimate exceeds MATLAB_MAX_GB fails the unit before MATLAB starts, so
an overnight run under a MemoryMax cap is never killed whole.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import numpy as np

from .base import Check, UnitResult, ffs_version, timed_cached, trailing_noise_count, volume_means

NORDIC_TOOLBOX = Path.home() / "Dropbox/Resources/code/matlab_toolboxes/NORDIC_Raw"
MATLAB = "/usr/local/MATLAB/R2025a/bin/matlab"
# under the overnight service MemoryMax (scripts/bench_overnight.sh, 200G) with room for
# the check's own float32 copies; override per host with FFS_MATLAB_MAX_GB
MATLAB_MAX_GB = float(os.environ.get("FFS_MATLAB_MAX_GB") or 160)
MATLAB_COPIES = 4  # complex double working copies NIFTI_NORDIC holds at its peak (estimate)

REF_ARGA = {"temporal_phase": 1, "phase_filter_width": 10, "NORDIC": 1, "save_gfactor_map": 1}
# ffs flags that reproduce MATLAB's estimator and g-factor overlap (old stage)
FFS_PARITY = ["-temporal-phase", "1", "-phase-filter-width", "10", "-noise-sigma-first-vol",
              "-gfactor-patch-overlap", "2", "-decomp-method", "eigh"]  # fmt: skip
FFS_VARIANTS = {"ffs": FFS_PARITY, "ffs_default": []}


def _is_phase(f) -> bool:
    return f.ent.get("part") == "phase" or f.suffix == "phase"


def ts_corr_median(a: np.ndarray, b: np.ndarray, mask: np.ndarray, keep: int) -> float:
    """Median per-voxel time-series r over the first ``keep`` volumes, slice by slice."""
    from ..stages.nordic import _ts_corr_median

    return _ts_corr_median(a, b, mask, keep)


def median_tsnr(x: np.ndarray, mask: np.ndarray, keep: int) -> float:
    v = x[..., :keep][mask].astype(np.float64)
    sd = v.std(axis=1)
    ok = sd > 0
    return float(np.median(v.mean(axis=1)[ok] / sd[ok])) if ok.any() else float("nan")


def matlab_gb(shape: tuple[int, ...]) -> float:
    return float(np.prod(shape)) * 16 * MATLAB_COPIES / 1e9


class NordicCheck(Check):
    name = "nordic"
    description = "NORDIC: ffs_nordic vs MATLAB NIFTI_NORDIC on magnitude + phase"
    needs = [["phase"]]
    sampling = "breadth"
    reference = "MATLAB NIFTI_NORDIC"
    timeout_s = 4 * 3600.0

    def unit_inputs(self, files, index):
        mags = [f for f in files if not _is_phase(f)]
        first = sorted(mags, key=lambda f: int(f.ent.get("echo", "1")))[0]
        echo = first.ent.get("echo", "1")
        phase = [f for f in files if _is_phase(f) and f.ent.get("echo", "1") == echo]
        rels = [first.rel] + [m.rel for m in index.matching(first, ".json")]
        return rels + [p.rel for p in phase[:1]]

    def run(self, key, inputs, work, device):
        import nibabel as nib

        imgs = [r for r in inputs if r.endswith((".nii", ".nii.gz"))]
        phase_rel = next(
            (r for r in imgs if "part-phase" in r or r.endswith("_phase.nii.gz")), None
        )
        mag_rel = next(r for r in imgs if r != phase_rel)
        if phase_rel is None:
            raise ValueError("no phase image among the inputs")
        mag, phase = inputs[mag_rel], inputs[phase_rel]
        shape = nib.Nifti1Image.from_filename(str(mag)).shape
        if matlab_gb(shape) > MATLAB_MAX_GB:
            raise ValueError(
                f"MATLAB NORDIC needs ~{matlab_gb(shape):.0f} GB for {shape} "
                f"(cap {MATLAB_MAX_GB:g} GB): too big for an unattended run"
            )
        work.mkdir(parents=True, exist_ok=True)
        log = work / "commands.log"
        res = UnitResult()

        curated = self.curation.get("noise_volumes")
        detected = trailing_noise_count(volume_means(mag))
        k = int(curated) if curated is not None else detected
        res.provenance["noise_volumes"] = {"used": k, "detected": detected, "curated": curated,
                                           "n_volumes": shape[3], "kept_for_denoising": True}  # fmt: skip

        # MATLAB writes beside its cwd: give it symlinks with plain names
        ref = work / "ref"
        ref.mkdir(exist_ok=True)
        ext = ".nii.gz" if str(mag).endswith(".gz") else ".nii"
        for name, src in (("mag", mag), ("phase", phase)):
            dst = ref / f"{name}{ext}"
            dst.unlink(missing_ok=True)
            dst.symlink_to(src)
        threads = int(os.environ.get("OMP_NUM_THREADS", "0")) or os.cpu_count() or 1
        toolbox = os.environ.get("NORDIC_MATLAB_PATH", str(NORDIC_TOOLBOX))
        arga = {**REF_ARGA, "noise_volume_last": k}
        script = (
            f"maxNumCompThreads({threads}); addpath(genpath('{toolbox}')); "
            + " ".join(f"ARGA.{a} = {v};" for a, v in arga.items())
            + f" NIFTI_NORDIC('mag{ext}', 'phase{ext}', 'nordic', ARGA);"
        )
        out_ref, gf_ref = ref / "nordic.nii", ref / "gfactor_nordic.nii"
        matlab = os.environ.get("FFS_MATLAB", MATLAB)
        t_ref = timed_cached(
            [matlab, "-batch", script], ref / "stamp.json", [out_ref], ref, log,
            timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("ref", "cpu", t_ref, threads))

        outs = {}
        for side, extra in FFS_VARIANTS.items():
            d = work / side
            d.mkdir(exist_ok=True)
            cmd = ["ffs_nordic", "-input-magn", str(mag), "-input-phase", str(phase),
                   "-prefix", str(d / "nordic"), "-noise-volume-last", str(k),
                   "-save-gfactor-map", *extra, "-device", device]  # fmt: skip
            t = timed_cached(cmd, d / "stamp.json", [d / "nordic.nii.gz"], work, log,
                             timeout_s=self.timeout_s)  # fmt: skip
            res.timings.append((side, device.split(",")[0], t, None))
            outs[side] = (d / "nordic.nii.gz", d / "nordic_gfactor.nii.gz")

        load = lambda p: np.asarray(nib.Nifti1Image.from_filename(str(p)).dataobj, np.float32)  # noqa: E731
        raw, r = load(mag), load(out_ref)
        keep = shape[3] - k
        import torch

        from ...processing.mask import automask

        cpu = torch.device("cpu")
        mean = torch.from_numpy(np.ascontiguousarray(raw[..., :keep].mean(axis=3)))
        mask = automask(mean, device=cpu).numpy().astype(bool)
        intr: dict[str, float | None] = {"tsnr_raw": median_tsnr(raw, mask, keep),
                                         "tsnr_ref": median_tsnr(r, mask, keep)}  # fmt: skip
        refm: dict[str, float | None] = {}
        g_ref = load(gf_ref).squeeze() if gf_ref.exists() else None
        for side, (p, gp) in outs.items():
            f = load(p)
            sfx = "" if side == "ffs" else f"_{side.removeprefix('ffs_')}"
            refm[f"ts_median_r{sfx}"] = ts_corr_median(
                r, f, mask & (r[..., :keep].std(3) > 0), keep
            )
            intr[f"tsnr_{side}"] = median_tsnr(f, mask, keep)
            if g_ref is not None and gp.exists():
                g = load(gp).squeeze()
                m = (g_ref > 0) & mask
                refm[f"gfactor_r{sfx}"] = (
                    float(np.corrcoef(g_ref[m], g[m])[0, 1]) if m.sum() > 10 else None
                )
        refm["noise_volumes"] = float(k)
        res.metrics["reference"] = refm
        res.metrics["intrinsic"] = intr

        res.artifacts += [(out_ref, "regenerable"), (log, "keep")]
        if gf_ref.exists():
            res.artifacts.append((gf_ref, "keep"))
        for p, gp in outs.values():
            res.artifacts += [(p, "regenerable"), (gp, "keep")]
        res.provenance["ref"] = {"tool": "NIFTI_NORDIC (MATLAB)", "toolbox": toolbox,
                                 "ARGA": arga, "threads": threads}  # fmt: skip
        res.provenance["ffs"] = {"tool": "ffs_nordic", **ffs_version(), "device": device,
                                 "variants": {s: shlex.join(a) for s, a in FFS_VARIANTS.items()}}  # fmt: skip
        return res
