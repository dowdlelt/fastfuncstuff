"""blipflip: ffs_blipflip against FSL topup (b02b0.cnf) on reverse phase-encoding pairs.

Unit: one session's ``fmap/*_dir-*_epi`` pair with opposite PhaseEncodingDirection. The
frozen input is the first volume of each, cropped to even dimensions when needed (topup's
subsampling demands it), written once and given to both tools.

Judges:
- reference: field (Hz) and corrected-mean agreement with topup, inside the brain.
- consistency: after correction the two polarities should be the same image. Each tool's
  corrected up vs down (r, and RMS difference over mean intensity) against the uncorrected
  pair's, scored identically for both tools, so this judge can say which is better.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .. import census
from .base import Check, UnitResult, _stable_rank, ffs_version, timed

TOPUP_CONFIG = "b02b0.cnf"
FFS_ARGS = ["-config", "standard", "-estmov"]  # b02b0 estimates movement at coarse levels
# The precision tier, scored alongside the default: standard regularises far harder than
# b02b0's numbers suggest, so the self-consistency judge needs both to be fair to ffs.
FFS_VARIANTS = {"ffs": FFS_ARGS, "ffs_workhard": ["-config", "workhard", "-estmov"]}


def _pe_vector(pe: str) -> list[int]:
    axis = {"i": 0, "j": 1, "k": 2, "x": 0, "y": 1, "z": 2}[pe[0]]
    v = [0, 0, 0]
    v[axis] = -1 if pe.endswith("-") else 1
    return v


def _readout(meta: dict, n_pe: int) -> float | None:
    if isinstance(meta.get("TotalReadoutTime"), (int, float)):
        return float(meta["TotalReadoutTime"])
    ees = meta.get("EffectiveEchoSpacing")
    if isinstance(ees, (int, float)):
        return float(ees) * (n_pe - 1)
    return None


def _meta(paths: list[Path]) -> dict:
    out: dict = {}
    for p in paths:
        out.update(json.loads(p.read_text()))
    return out


def _r(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


class BlipflipCheck(Check):
    name = "blipflip"
    description = "Distortion correction: ffs_blipflip vs FSL topup (b02b0) on reverse-PE pairs"
    needs = [["@ds:fmap_epi"]]
    sampling = "breadth"
    reference = "FSL topup"

    def plan(self, manifest, index, n):
        """Up to ``n`` pairs per fmap acquisition label, distinct subjects first."""
        groups: dict[tuple, list[census.BidsName]] = {}
        for nm in index.names:
            if nm.datatype != "fmap" or nm.suffix != "epi" or nm.ext not in census._IMAGE_EXT:
                continue
            if nm.ent.get("part") == "phase" or nm.ent.get("echo", "1") != "1":
                continue
            k = tuple(nm.ent.get(e) for e in ("sub", "ses", "acq", "run"))
            groups.setdefault(k, []).append(nm)
        by_acq: dict[str | None, list[tuple]] = {}
        for k, files in groups.items():
            dirs = {f.ent.get("dir") for f in files}
            if len(files) >= 2 and len(dirs) >= 2:
                by_acq.setdefault(k[2], []).append(k)
        ds = manifest["dataset_id"]
        units = []
        for acq, keys in sorted(by_acq.items(), key=lambda kv: str(kv[0])):
            order = sorted(keys, key=lambda k: _stable_rank(ds, str(acq), *map(str, k)))
            picked, subs = [], set()
            for k in order:
                if len(picked) < n and k[0] not in subs:
                    picked.append(k)
                    subs.add(k[0])
            for k in order:
                if len(picked) < n and k not in picked:
                    picked.append(k)
            for k in picked:
                files = sorted(groups[k], key=lambda f: f.ent.get("dir", ""))
                # One file per direction: the first of each.
                seen, pair = set(), []
                for f in files:
                    if f.ent.get("dir") not in seen:
                        seen.add(f.ent.get("dir"))
                        pair.append(f)
                pair = pair[:2]
                inputs = []
                for f in pair:
                    inputs.append(f.rel)
                    inputs += [m.rel for m in index.matching(f, ".json")]
                key = {"acq": f"fmap-{acq or 'epi'}", "sub": k[0], "ses": k[1], "run": k[3]}
                units.append((key, inputs))
        return units

    def run(self, key, inputs, work, device):
        import nibabel as nib

        images = [rel for rel in inputs if not rel.endswith(".json")]
        if len(images) != 2:
            raise ValueError(f"expected 2 images, got {images}")
        work.mkdir(parents=True, exist_ok=True)
        ref, ffs = work / "topup", work / "ffs"
        ref.mkdir(exist_ok=True)
        ffs.mkdir(exist_ok=True)
        log = work / "commands.log"

        # Frozen input: first volume of each, cropped to even dims, saved once.
        vols, pes, ro = [], [], []
        affine = None
        for rel in images:
            img = nib.Nifti1Image.from_filename(str(inputs[rel]))
            data = np.asarray(img.dataobj, dtype=np.float32)
            if data.ndim == 4:
                data = data[..., 0]
            meta = _meta(
                [inputs[r] for r in inputs if r.endswith(".json") and self._applies(r, rel)]
            )
            meta.update(self.curation.get("sidecar") or {})
            meta.update(self.file_overrides(rel))
            pe = meta.get("PhaseEncodingDirection")
            if not pe:
                raise ValueError(f"no PhaseEncodingDirection for {rel}")
            pes.append(pe)
            n_pe = data.shape[{"i": 0, "j": 1, "k": 2}[pe[0]]]
            ro.append(_readout(meta, n_pe))
            vols.append(data)
            affine = img.affine
        if pes[0][0] != pes[1][0] or pes[0] == pes[1]:
            raise ValueError(f"not a reverse-PE pair: {pes}")
        even = tuple(s - (s % 2) for s in vols[0].shape)
        vols = [v[: even[0], : even[1], : even[2]] for v in vols]
        readout = ro[0] or ro[1]
        ro_note = (
            "sidecar" if readout else "missing: 0.05 s assumed (Hz scale arbitrary, same for both)"
        )
        if readout and not 0.01 <= readout <= 0.2:
            # topup refuses outside 0.01-0.2 s (ds001499 sidecars say 0.00065 s); field r and
            # up/down consistency do not depend on the scale, only the Hz numbers do
            ro_note = f"implausible sidecar {readout:g} s: 0.05 s assumed (Hz scale arbitrary)"
            readout = None
        readout = readout or 0.05
        up, down = work / "up.nii.gz", work / "down.nii.gz"
        for p, v in ((up, vols[0]), (down, vols[1])):
            nib.save(nib.Nifti1Image(v, affine), p)
        both = work / "both.nii.gz"
        nib.save(nib.Nifti1Image(np.stack(vols, -1), affine), both)
        acqp = work / "acqparams.txt"
        acqp.write_text(
            "".join(f"{' '.join(map(str, _pe_vector(pe)))} {readout:.6f}\n" for pe in pes)
        )

        res = UnitResult()
        threads = int(os.environ.get("OMP_NUM_THREADS", "0")) or None
        t_ref = timed(
            ["topup", f"--imain={both}", f"--datain={acqp}", f"--config={TOPUP_CONFIG}",
             f"--out={ref / 'tu'}", f"--fout={ref / 'field'}", f"--iout={ref / 'unwarped'}"],
            work, log, timeout_s=self.timeout_s,
        )  # fmt: skip
        res.timings.append(("ref", "cpu", t_ref, threads))
        for side, args in FFS_VARIANTS.items():
            out = work / side
            out.mkdir(exist_ok=True)
            t_ffs = timed(
                ["ffs_blipflip", "-blip_up", str(up), "-blip_down", str(down), "-pe_dir", pes[0],
                 "-readout", f"{readout:.6f}", *args, "-prefix", str(out / "dc"), "-device", device],
                work, log, timeout_s=self.timeout_s,
            )  # fmt: skip
            res.timings.append((side, device.split(",")[0], t_ffs, None))

        def load(p: Path) -> np.ndarray:
            return np.asarray(nib.Nifti1Image.from_filename(str(p)).dataobj, dtype=np.float64)

        tu_field = load(ref / "field.nii.gz")
        tu_unw = load(ref / "unwarped.nii.gz")  # (x, y, z, 2)
        ff_field = load(ffs / "dc_field.nii.gz")
        ff_unw = load(ffs / "dc_unwarped.nii.gz")
        if ff_unw.ndim == 4 and ff_unw.shape[-1] != 2 and ff_unw.shape[0] == 2:
            ff_unw = np.moveaxis(ff_unw, 0, -1)
        raw = np.stack(vols, -1).astype(np.float64)

        mean_tu = tu_unw.mean(-1)
        thr = 0.1 * np.percentile(mean_tu, 98)
        mask = (mean_tu > thr) & (ff_unw.mean(-1) > thr)

        def consistency(pair: np.ndarray) -> tuple[float, float]:
            a, b = pair[..., 0][mask], pair[..., 1][mask]
            return _r(a, b), float(np.sqrt(np.mean((a - b) ** 2)) / np.mean((a + b) / 2))

        r_raw, nrms_raw = consistency(raw)
        r_tu, nrms_tu = consistency(tu_unw)
        r_ff, nrms_ff = consistency(ff_unw)
        wh_field = load(work / "ffs_workhard" / "dc_field.nii.gz")
        wh_unw = load(work / "ffs_workhard" / "dc_unwarped.nii.gz")
        if wh_unw.ndim == 4 and wh_unw.shape[-1] != 2 and wh_unw.shape[0] == 2:
            wh_unw = np.moveaxis(wh_unw, 0, -1)
        r_wh, nrms_wh = consistency(wh_unw)
        fdiff = (ff_field - tu_field)[mask]
        res.metrics["reference"] = {
            "field_r": _r(ff_field[mask], tu_field[mask]),
            "field_median_abs_diff_hz": float(np.median(np.abs(fdiff))),
            "field_p95_abs_diff_hz": float(np.percentile(np.abs(fdiff), 95)),
            "field_topup_p95_abs_hz": float(np.percentile(np.abs(tu_field[mask]), 95)),
            "corrected_mean_r": _r(ff_unw.mean(-1)[mask], mean_tu[mask]),
            "field_r_workhard": _r(wh_field[mask], tu_field[mask]),
            "field_median_abs_diff_hz_workhard": float(
                np.median(np.abs((wh_field - tu_field)[mask]))
            ),
            "mask_voxels": float(mask.sum()),
        }
        res.metrics["consistency"] = {
            "updown_r_raw": r_raw, "updown_r_topup": r_tu, "updown_r_ffs": r_ff,
            "updown_nrms_raw": nrms_raw, "updown_nrms_topup": nrms_tu, "updown_nrms_ffs": nrms_ff,
            "updown_r_ffs_workhard": r_wh, "updown_nrms_ffs_workhard": nrms_wh,
        }  # fmt: skip

        for p in (both, up, down, ref / "unwarped.nii.gz", ffs / "dc_unwarped.nii.gz",
                  ffs / "dc_warp.nii.gz", ffs / "dc_invwarp.nii.gz",
                  *(work / "ffs_workhard" / f"dc_{n}.nii.gz" for n in ("unwarped", "warp", "invwarp"))):  # fmt: skip
            res.artifacts.append((p, "regenerable"))
        for p in (ref / "field.nii.gz", ffs / "dc_field.nii.gz", ffs / "dc_mean.nii.gz",
                  work / "ffs_workhard" / "dc_field.nii.gz",
                  ref / "tu_movpar.txt", acqp, log):  # fmt: skip
            res.artifacts.append((p, "keep"))
        res.provenance["ref"] = {"tool": "topup", "config": TOPUP_CONFIG, "fsl": _fsl_version()}
        res.provenance["ffs"] = {
            "tool": "ffs_blipflip",
            **ffs_version(),
            "args": FFS_ARGS,
            "variants": FFS_VARIANTS,
            "device": device,
        }
        res.provenance["inputs_prep"] = {
            "volume": "first of each", "cropped_to": list(even), "pe": pes, "readout": ro_note,
        }  # fmt: skip
        return res

    @staticmethod
    def _applies(json_rel: str, image_rel: str) -> bool:
        j, i = census.parse_name(json_rel), census.parse_name(image_rel)
        return j is not None and i is not None and census._applies(j, i)


def _fsl_version() -> str:
    try:
        return (Path(os.environ.get("FSLDIR", "")) / "etc" / "fslversion").read_text().strip()
    except OSError:
        return ""
