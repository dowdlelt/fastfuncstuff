"""Native EPI straight onto a cortical mesh, in one interpolation.

The surface twin of ``ffs_nwarp``'s volume output. The FreeSurfer subject gives white,
pial and ``sphere.reg``; a target mesh that lives on the registered sphere (onavg,
SUMA std.N, or the subject's own mesh) is placed in the subject through it
(:func:`surface.remesh.remesh_via_sphere`); :func:`surface.projection.build_sampling`
turns that into read points (vertex footprints at equivolume depths); nwarpforge
reads the EPI at those points through the whole chain; the reads fold back onto the
vertices. Both hemispheres go through one nwarp call, so the chain is composed once.

The chain must end in the anatomy the surfaces were built on: ``master`` is that
anatomy (``SUMA/brain.nii.gz`` or the T1w recon-all ran on), and the surface points
enter it through its real affine. See [[Surfaces as an analysis space]] section 1b.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fastfuncstuff.io.freesurfer import read_scanner_surface, read_surface
from fastfuncstuff.io.gifti import mesh_fingerprint, save_gifti_data
from fastfuncstuff.surface.projection import SurfaceSampling, build_sampling
from fastfuncstuff.surface.remesh import remesh_via_sphere
from fastfuncstuff.surface.topology import MeshBundle

__all__ = ["HemiTarget", "project_to_surface", "resolve_mesh", "surface_targets"]


@dataclass
class HemiTarget:
    """One hemisphere's target mesh, placed in the subject, and how to read it."""

    hemi: str
    mesh: str  # "native" or the template's folder name
    faces: np.ndarray
    sampling: SurfaceSampling
    cortex: np.ndarray | None  # target-mesh cortex mask, when the mesh has one
    meta: dict[str, str] = field(default_factory=dict)


def resolve_mesh(mesh: str, subject_dir: str | os.PathLike) -> Path | None:
    """The template folder for ``mesh``: a path, or a sibling of the subject (the
    usual SUBJECTS_DIR layout, e.g. ``onavg-ico64`` beside the subject). None = native."""
    if mesh == "native":
        return None
    p = Path(mesh)
    if p.is_dir():
        return p
    sib = Path(subject_dir).parent / mesh
    if sib.is_dir():
        return sib
    raise FileNotFoundError(f"-surf_mesh {mesh!r}: not a folder, and not beside {subject_dir}")


def _cortex(folder: Path, hemi: str, n: int) -> np.ndarray | None:
    path = folder / "label" / f"{hemi}.cortex.label"
    if not path.is_file():
        return None
    ids = np.loadtxt(path, skiprows=2, usecols=0, ndmin=1).astype(np.int64)
    mask = np.zeros(n, bool)
    mask[ids[ids < n]] = True
    return mask


def _sha(x: np.ndarray) -> str:
    import hashlib

    return hashlib.sha1(np.ascontiguousarray(x, np.float32).tobytes()).hexdigest()[:12]


def surface_targets(
    subject_dir: str | os.PathLike,
    mesh: str = "native",
    hemis=("lh", "rh"),
    fractions=(0.5,),
    voxel_face: float | None = None,
    white: str = "white",
    pial: str = "pial",
) -> list[HemiTarget]:
    """Each hemisphere's target mesh in the subject's scanner space, with its sampling."""
    subject_dir = Path(subject_dir)
    folder = resolve_mesh(mesh, subject_dir)
    out = []
    for hemi in hemis:
        surf = subject_dir / "surf"
        w, faces = read_scanner_surface(surf / f"{hemi}.{white}")
        p, pf = read_scanner_surface(surf / f"{hemi}.{pial}")
        if not np.array_equal(pf, faces):
            raise ValueError(f"{hemi}.{white} and {hemi}.{pial} are not the same mesh")
        # The subject's surface hashes, not just the topology: a geometric edit keeps
        # the fingerprint but makes an earlier projection stale.
        meta = {
            "subject": str(subject_dir),
            "white_sha1": _sha(w),
            "pial_sha1": _sha(p),
            "mesh": mesh,
        }
        if folder is None:
            tw, tp, tf = w, p, faces
            cortex = _cortex(subject_dir, hemi, len(w))
        else:
            reg = read_surface(surf / f"{hemi}.sphere.reg")
            tgt = read_surface(folder / "surf" / f"{hemi}.sphere.reg")
            bundle = MeshBundle(
                faces=faces.astype(np.int64),
                positions={
                    "surf:sphere.reg": reg.vertices.astype(np.float64),
                    "surf:white": w.astype(np.float64),
                    "surf:pial": p.astype(np.float64),
                },
                spherical={"surf:sphere.reg": np.zeros(3)},
            )
            placed, _ = remesh_via_sphere(bundle, tgt.vertices, tgt.faces)
            tw, tp, tf = placed.positions["surf:white"], placed.positions["surf:pial"], placed.faces
            cortex = _cortex(folder, hemi, len(tw))
        smp = build_sampling(tw, tp, tf, fractions, voxel_face)
        meta["mesh_fingerprint"] = mesh_fingerprint(tf, len(tw))
        out.append(HemiTarget(hemi, mesh, np.asarray(tf), smp, cortex, meta))
    return out


def _voxel_face(source_path: str) -> float:
    """Face area of the source voxel (mm^2), the geometric mean for anisotropic voxels."""
    from fastfuncstuff.io.headers import read_nifti_header

    zooms = np.asarray(read_nifti_header(source_path).get_zooms()[:3], np.float64)
    return float(np.prod(zooms) ** (2.0 / 3.0))


def project_to_surface(
    source_path: str,
    nwarp_specs: list[str],
    master_path: str,
    subject_dir: str | os.PathLike,
    prefix: str,
    mesh: str = "native",
    hemis=("lh", "rh"),
    fractions=(0.5,),
    depth_mean: bool = False,
    sample: str = "footprint",
    verb: int = 1,
    **nwarp_kwargs,
) -> list[Path]:
    """Sample ``source`` onto the surface through the chain and write GIfTI.

    Writes ``{prefix}.{hemi}.func.gii`` (vertices x time, depth-averaged when
    ``depth_mean`` or with one depth) or one ``{prefix}.{hemi}.depth-{f}.func.gii`` per
    depth, plus ``{prefix}.{hemi}.coverage.shape.gii``: the share of each vertex's
    footprint read inside the EPI in every frame. Returns the paths written.
    """
    from fastfuncstuff.io.afni import get_tr_from_file

    from .nwarpforge import nwarpforge

    if sample not in ("footprint", "point"):
        raise ValueError(f"sample must be 'footprint' or 'point', got {sample!r}")
    vface = _voxel_face(source_path) if sample == "footprint" else None
    targets = surface_targets(subject_dir, mesh, hemis, fractions, vface)
    counts = [t.sampling.points.shape[0] for t in targets]
    if verb >= 1:
        for t, n in zip(targets, counts, strict=True):
            print(
                f"  {t.hemi}: {mesh} mesh, {t.sampling.n_vertices} vertices x "
                f"{t.sampling.n_depths} depth(s) -> {n} reads ({sample})"
            )
    reads = nwarpforge(
        source_path=source_path,
        nwarp_specs=nwarp_specs,
        prefix="",
        master_path=master_path,
        points=np.concatenate([t.sampling.points for t in targets]),
        verb=verb,
        **nwarp_kwargs,
    )
    assert reads is not None
    r = reads.cpu().numpy()
    r = r[None] if r.ndim == 1 else r  # (T, P)
    tr = get_tr_from_file(source_path)
    written: list[Path] = []
    for t, chunk in zip(targets, np.split(r, np.cumsum(counts)[:-1], axis=1), strict=True):
        folded = t.sampling.fold(chunk)  # (K, V, T)
        cover = t.sampling.coverage(chunk)  # (K, V)
        meta = dict(t.meta)
        meta.update(
            source=str(source_path),
            sampling=sample,
            depths=" ".join(f"{f:g}" for f in t.sampling.fractions),
            equivolume="1",
        )
        if tr and tr > 0:
            meta["TR_seconds"] = f"{tr:g}"
        stem = f"{prefix}.{t.hemi}"
        if depth_mean or t.sampling.n_depths == 1:
            path = Path(f"{stem}.func.gii")
            save_gifti_data(path, folded.mean(axis=0), {**meta, "depth_mean": "1"})
            written.append(path)
        else:
            for k, f in enumerate(t.sampling.fractions):
                path = Path(f"{stem}.depth-{f:.2f}.func.gii")
                save_gifti_data(path, folded[k], {**meta, "depth": f"{f:g}"})
                written.append(path)
        cpath = Path(f"{stem}.coverage.shape.gii")
        save_gifti_data(cpath, cover.min(axis=0), meta, time_series=False)
        written.append(cpath)
    return written
