"""Synthetic blur benchmark: native EPI to the surface in one step vs two.

[[Wang 2022]]'s question on a phantom where the answer is known. A spherical
"cortex" (white r=40 mm, pial r=43 mm) sits in a 1.5 mm EPI grid offset by a
different fraction of a voxel on each axis, so no intermediate grid lines up with
the acquisition. (Not oblique: an identity chain between an oblique EPI and a
cardinal anatomy is AFNI's obliquity-ignored convention, which would place the
phantom's EPI somewhere else; a real chain carries that rotation.) Two inputs go through each route:

* white noise (N frames): the SD left at each vertex, mapped to the equivalent
  Gaussian FWHM on the native grid (the paper's meter, as ``-surf_qc`` does it);
* bands of known wavelength along the cortex (radially constant, so depth does not
  matter): the amplitude that survives (the modulation transfer) and r with the truth.

Routes, all footprint-sampled and depth-averaged unless named otherwise:

  one-step            raw EPI -> surface, one wsinc5 read (ffs_nwarp -surf)
  one-step point mid  the same, one point per vertex at mid depth (no footprints)
  two-step EPI-res    raw -> cardinal 1.5 mm volume in anat space (wsinc5) -> surface
  two-step 0.5 mm     raw -> upsampled 0.5 mm volume (wsinc5) -> surface
  two-step trilinear  raw -> 1.5 mm volume -> trilinear point read at mid depth
                      (the classic vol2surf route)

Usage:
    python scripts/bench_surface_blur.py [-frames 64] [-level 6] [-device cpu]
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import nibabel as nib
import nibabel.freesurfer as nfs
import numpy as np
import torch

R_WHITE, R_PIAL = 40.0, 43.0
VOXEL = 1.5
WAVELENGTHS = (3.0, 5.0, 8.0)  # mm along the mid surface


def _icosphere(level: int) -> tuple[np.ndarray, np.ndarray]:
    t = (1 + 5**0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t)]
    v += [(0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4)]
    f += [(11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8)]
    f += [(3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, float) / np.linalg.norm(p) for p in v]
    faces = f
    for _ in range(level):
        mid: dict[tuple[int, int], int] = {}

        def m(a: int, b: int, mid: dict = mid) -> int:
            key = (min(a, b), max(a, b))
            if key not in mid:
                p = verts[a] + verts[b]
                verts.append(p / np.linalg.norm(p))
                mid[key] = len(verts) - 1
            return mid[key]

        faces = [
            g
            for a, b, c in faces
            for g in (
                (a, m(a, b), m(c, a)),
                (b, m(b, c), m(a, b)),
                (c, m(c, a), m(b, c)),
                (m(a, b), m(b, c), m(c, a)),
            )
        ]
    return np.asarray(verts), np.asarray(faces, np.int32)


def _vinfo() -> dict:
    return {
        "head": np.array([2, 0, 20], np.int32),
        "valid": "1  # volume info valid",
        "filename": "orig.mgz",
        "volume": np.array([256, 256, 256]),
        "voxelsize": np.array([1.0, 1.0, 1.0]),
        "xras": np.array([-1.0, 0, 0]),
        "yras": np.array([0, 0, -1.0]),
        "zras": np.array([0, 1.0, 0]),
        "cras": np.zeros(3),
    }


def _bands(direction: np.ndarray, wavelength: float) -> np.ndarray:
    """Rings round z: cos of the arc length from the pole on the mid surface."""
    theta = np.arccos(np.clip(direction[..., 2], -1, 1))
    return np.cos(2 * np.pi * theta * 0.5 * (R_WHITE + R_PIAL) / wavelength)


def _grid(n: int, voxel: float, offset_vox) -> np.ndarray:
    aff = np.eye(4) * voxel
    aff[3, 3] = 1.0
    aff[:3, 3] = -voxel * ((n - 1) / 2 + np.asarray(offset_vox))
    return aff


def _save(path: Path, data: np.ndarray, aff: np.ndarray) -> Path:
    img = nib.Nifti1Image(data.astype(np.float32), aff)
    img.header.set_xyzt_units("mm", "sec")
    if data.ndim == 4:
        img.header["pixdim"][4] = 1.0
    img.to_filename(str(path))
    return path


def build(tmp: Path, level: int, frames: int, seed: int) -> dict:
    subj = tmp / "subj"
    (subj / "surf").mkdir(parents=True)
    (subj / "label").mkdir()
    d, f = _icosphere(level)
    for name, r in (("white", R_WHITE), ("pial", R_PIAL)):
        nfs.write_geometry(str(subj / "surf" / f"lh.{name}"), r * d, f, volume_info=_vinfo())

    n = int(np.ceil(2 * (R_PIAL + 6) / VOXEL))
    epi_aff = _grid(n, VOXEL, (0.37, 0.21, 0.43))
    ijk = np.stack(np.meshgrid(*(np.arange(n),) * 3, indexing="ij"), -1).reshape(-1, 3)
    xyz = ijk @ epi_aff[:3, :3].T + epi_aff[:3, 3]
    direc = xyz / np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-6)
    bands = np.stack([_bands(direc, w) for w in WAVELENGTHS], -1).reshape(n, n, n, -1)
    rng = np.random.default_rng(seed)
    out = dict(
        subj=subj,
        directions=d,
        signal=_save(tmp / "bands.nii.gz", 1000.0 + 100.0 * bands, epi_aff),
        noise=_save(tmp / "noise.nii.gz", rng.normal(size=(n, n, n, frames)), epi_aff),
        ident=tmp / "ident.aff12.1D",
    )
    out["ident"].write_text("1 0 0 0 0 1 0 0 0 0 1 0\n")
    # The "anatomy": a cardinal grid the surfaces live in (only its affine matters).
    m = int(np.ceil(2 * (R_PIAL + 6)))
    anat = np.eye(4)
    anat[:3, 3] = -(m - 1) / 2
    out["anat"] = _save(tmp / "anat.nii.gz", np.zeros((m, m, m)), anat)
    for vox in (VOXEL, 0.5):
        k = int(np.ceil(2 * (R_PIAL + 6) / vox))
        g = np.eye(4) * vox
        g[3, 3] = 1.0
        g[:3, 3] = -vox * (k - 1) / 2
        out[f"grid{vox}"] = _save(tmp / f"grid{vox}.nii.gz", np.zeros((k, k, k)), g)
    return out


def _project(src, world, prefix, device, sample="footprint", depths=None, interp="wsinc5"):
    from fastfuncstuff.processing.surface_projection import DEFAULT_DEPTHS, project_to_surface

    project_to_surface(
        source_path=str(src), nwarp_specs=[str(world["ident"])], master_path=str(world["anat"]),
        subject_dir=world["subj"], prefix=str(prefix), meshes=("native",), hemis=("lh",),
        fractions=depths or DEFAULT_DEPTHS, sample=sample, verb=0, interp=interp,
        device=device, no_neg=False,
    )  # fmt: skip
    from fastfuncstuff.io.gifti import load_gifti_data

    data, _ = load_gifti_data(f"{prefix}.native.lh.func.gii")
    return data if data.ndim == 2 else data[:, None]


def _resample(src, world, grid, out, device):
    from fastfuncstuff.processing.nwarpforge import nwarpforge

    nwarpforge(
        source_path=str(src), nwarp_specs=[str(world["ident"])], prefix=str(out),
        master_path=str(world[grid]), interp="wsinc5", device=device, verb=0, auto_pad=False,
    )  # fmt: skip
    return out


def run(level: int = 6, frames: int = 64, device: str = "cpu", seed: int = 0) -> list[dict]:
    from fastfuncstuff.processing.surface_qc import noise_ratio_to_fwhm

    dev = torch.device(device)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        world = build(tmp, level, frames, seed)
        truth = np.stack([_bands(world["directions"], w) for w in WAVELENGTHS], 1)
        mid = [0.5]
        routes = {
            "one-step": dict(src="raw"),
            "one-step point mid": dict(src="raw", sample="point", depths=mid),
            "two-step EPI-res": dict(src="grid1.5"),
            "two-step 0.5 mm": dict(src="grid0.5"),
            "two-step trilinear": dict(src="grid1.5", sample="point", depths=mid, interp="linear"),
        }
        for name, cfg in routes.items():
            t0 = time.time()
            kw = {k: v for k, v in cfg.items() if k != "src"}
            ins = {}
            for what in ("noise", "signal"):
                src = world[what]
                if cfg["src"] != "raw":
                    src = _resample(
                        src, world, cfg["src"], tmp / f"{what}.{cfg['src']}.nii.gz", dev
                    )
                ins[what] = _project(
                    src, world, tmp / f"{name}.{what}".replace(" ", "_"), dev, **kw
                )
            ratio = np.sqrt((ins["noise"].astype(np.float64) ** 2).mean(1))
            fwhm = noise_ratio_to_fwhm(ratio, (VOXEL,) * 3)
            sig = ins["signal"] - 1000.0
            row = {"route": name, "noise_fwhm_mm": float(np.median(fwhm)),
                   "noise_fwhm_p5_p95": tuple(np.percentile(fwhm, (5, 95)).round(2)),
                   "seconds": round(time.time() - t0, 1)}  # fmt: skip
            for j, w in enumerate(WAVELENGTHS):
                x, y = truth[:, j], sig[:, j] / 100.0
                row[f"amp_{w:g}mm"] = float((x @ y) / (x @ x))
                row[f"r_{w:g}mm"] = float(np.corrcoef(x, y)[0, 1])
            rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("-level", type=int, default=6, help="icosphere level (6: 40,962 vertices)")
    p.add_argument("-frames", type=int, default=64, help="white-noise frames")
    p.add_argument("-device", default="cpu")
    a = p.parse_args()
    rows = run(a.level, a.frames, a.device)
    cols = ["route", "noise_fwhm_mm", "noise_fwhm_p5_p95"]
    cols += [f"amp_{w:g}mm" for w in WAVELENGTHS] + [f"r_{w:g}mm" for w in WAVELENGTHS]
    cols += ["seconds"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
