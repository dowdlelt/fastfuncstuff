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

__all__ = [
    "HemiTarget",
    "output_paths",
    "project_to_surface",
    "resolve_mesh",
    "resolve_subject",
    "space_name",
    "surface_targets",
]


@dataclass
class HemiTarget:
    """One hemisphere's target mesh, placed in the subject, and how to read it."""

    hemi: str
    mesh: str  # "native" or the template's folder name
    faces: np.ndarray
    white: np.ndarray  # (V, 3) the target's vertices placed in the subject, scanner mm
    pial: np.ndarray
    sampling: SurfaceSampling
    cortex: np.ndarray | None  # target-mesh cortex mask, when the mesh has one
    meta: dict[str, str] = field(default_factory=dict)


def _freesurfer_dirs() -> list[Path]:
    """Where FreeSurfer keeps subjects by name: ``$SUBJECTS_DIR``, then the copy of
    fsaverage & co. that ships in ``$FREESURFER_HOME/subjects``."""
    dirs = [os.environ.get("SUBJECTS_DIR")]
    home = os.environ.get("FREESURFER_HOME")
    dirs.append(str(Path(home) / "subjects") if home else None)
    return [Path(d) for d in dirs if d]


def resolve_subject(subject: str | os.PathLike) -> Path:
    """A FreeSurfer subject as a path, or as a name in :func:`_freesurfer_dirs`."""
    p = Path(subject)
    if (p / "surf").is_dir():
        return p
    for d in _freesurfer_dirs():
        if (d / p / "surf").is_dir():
            return d / p
    looked = ", ".join(str(d) for d in _freesurfer_dirs()) or "no $SUBJECTS_DIR set"
    raise FileNotFoundError(f"FreeSurfer subject {str(subject)!r}: no surf/ there or in {looked}")


def resolve_mesh(mesh: str, subject_dir: str | os.PathLike) -> Path | None:
    """The template folder for ``mesh``: a path, a sibling of the subject (the usual
    SUBJECTS_DIR layout, e.g. ``onavg-ico64`` beside the subject), or a subject of that
    name in ``$SUBJECTS_DIR`` / ``$FREESURFER_HOME/subjects``. None = native."""
    if mesh == "native":
        return None
    candidates = [Path(mesh), Path(subject_dir).parent / mesh]
    candidates += [d / mesh for d in _freesurfer_dirs()]
    for p in candidates:
        if p.is_dir():
            return p
    looked = ", ".join(str(p.parent) for p in candidates[1:])
    raise FileNotFoundError(f"-surf_mesh {mesh!r}: not a folder, and not in {looked}")


def space_name(mesh: str) -> str:
    """The name a target goes by in file names: ``native`` or the template folder's."""
    return "native" if mesh == "native" else Path(mesh).name


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
    subject_dir = resolve_subject(subject_dir)
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
            "mesh": space_name(mesh),
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
        out.append(HemiTarget(hemi, space_name(mesh), np.asarray(tf), tw, tp, smp, cortex, meta))
    return out


def _voxel_face(source_path: str) -> float:
    """Face area of the source voxel (mm^2), the geometric mean for anisotropic voxels."""
    from fastfuncstuff.io.headers import read_nifti_header

    zooms = np.asarray(read_nifti_header(source_path).get_zooms()[:3], np.float64)
    return float(np.prod(zooms) ** (2.0 / 3.0))


def output_paths(prefix: str, meshes, hemis, fractions, depth_mean: bool = False) -> list[str]:
    """Every file :func:`project_to_surface` writes, in order (for -batch_skip)."""
    out = []
    for mesh in meshes:
        for hemi in hemis:
            stem = f"{prefix}.{space_name(mesh)}.{hemi}"
            if depth_mean or len(fractions) == 1:
                out.append(f"{stem}.func.gii")
            else:
                out += [f"{stem}.depth-{float(f):.2f}.func.gii" for f in fractions]
            out += [f"{stem}.coverage.shape.gii", f"{stem}.mask.shape.gii"]
            out += [f"{stem}.{s}.surf.gii" for s in ("white", "pial", "midthickness")]
    return out


def project_to_surface(
    source_path: str,
    nwarp_specs: list[str],
    master_path: str,
    subject_dir: str | os.PathLike,
    prefix: str,
    meshes=("native",),
    hemis=("lh", "rh"),
    fractions=(0.5,),
    depth_mean: bool = False,
    sample: str = "footprint",
    verb: int = 1,
    **nwarp_kwargs,
) -> list[Path]:
    """Sample ``source`` onto every target mesh through the chain and write GIfTI.

    All targets and hemispheres share one nwarp call, so the chain is composed once
    and native and template outputs come from the same reads of the same data. Per
    target ``{prefix}.{space}.{hemi}`` (space = ``native`` or the template's folder
    name) gets:

    * ``.func.gii`` -- vertices x time (depth-averaged with ``depth_mean`` or one depth),
      or one ``.depth-{f}.func.gii`` per depth;
    * ``.coverage.shape.gii`` -- share of each footprint read inside the EPI in every
      frame; ``.mask.shape.gii`` -- cortex label (when the mesh has one) AND full
      coverage, the mask statistics should use;
    * ``.white/.pial/.midthickness.surf.gii`` -- the target's vertices placed in THIS
      subject (scanner mm): the geometry smoothing, cluster areas and display need.
    """
    from fastfuncstuff.io.afni import get_tr_from_file
    from fastfuncstuff.io.gifti import save_gifti_surface

    from .nwarpforge import nwarpforge

    if sample not in ("footprint", "point"):
        raise ValueError(f"sample must be 'footprint' or 'point', got {sample!r}")
    meshes = [meshes] if isinstance(meshes, str) else list(meshes)
    names = [space_name(m) for m in meshes]
    if len(set(names)) != len(names):
        raise ValueError(f"target meshes must have distinct names, got {names}")
    subject_dir = resolve_subject(subject_dir)
    vface = _voxel_face(source_path) if sample == "footprint" else None
    targets = [
        t for mesh in meshes for t in surface_targets(subject_dir, mesh, hemis, fractions, vface)
    ]
    counts = [t.sampling.points.shape[0] for t in targets]
    if verb >= 1:
        for t, n in zip(targets, counts, strict=True):
            print(
                f"  {t.mesh} {t.hemi}: {t.sampling.n_vertices} vertices x "
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
        cover = t.sampling.coverage(chunk).min(axis=0)  # (V,)
        stem = f"{prefix}.{t.mesh}.{t.hemi}"
        geom = {
            "white": t.white,
            "pial": t.pial,
            "midthickness": 0.5 * (t.white + t.pial),
        }
        for name, pos in geom.items():
            path = Path(f"{stem}.{name}.surf.gii")
            save_gifti_surface(path, pos, t.faces, {**t.meta, "surface": name})
        meta = dict(t.meta)
        meta.update(
            source=str(source_path),
            sampling=sample,
            depths=" ".join(f"{f:g}" for f in t.sampling.fractions),
            equivolume="1",
            # Relative, so the outputs can move together.
            geometry=Path(f"{stem}.midthickness.surf.gii").name,
        )
        if tr and tr > 0:
            meta["TR_seconds"] = f"{tr:g}"
        if depth_mean or t.sampling.n_depths == 1:
            path = Path(f"{stem}.func.gii")
            save_gifti_data(path, folded.mean(axis=0), {**meta, "depth_mean": "1"})
            written.append(path)
        else:
            for k, f in enumerate(t.sampling.fractions):
                path = Path(f"{stem}.depth-{f:.2f}.func.gii")
                save_gifti_data(path, folded[k], {**meta, "depth": f"{f:g}"})
                written.append(path)
        mask = cover > 0.99
        if t.cortex is not None:
            mask &= t.cortex
        for name, values in (("coverage", cover), ("mask", mask.astype(np.float32))):
            path = Path(f"{stem}.{name}.shape.gii")
            save_gifti_data(path, values, meta, time_series=False)
            written.append(path)
        written += [Path(f"{stem}.{name}.surf.gii") for name in geom]
    return written
