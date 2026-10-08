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
    "DEFAULT_DEPTHS",
    "HemiTarget",
    "SurfaceFold",
    "depth_weights",
    "output_paths",
    "project_to_surface",
    "resolve_mesh",
    "resolve_subject",
    "space_name",
    "subject_on_mesh",
    "surface_targets",
]


#: Default depths: the centres of five equivolume bins, combined with equal weights --
#: the midpoint rule for the ribbon mean, so every cortical volume element counts the
#: same and neither white matter (0) nor CSF (1) is read at the boundary itself.
DEFAULT_DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
DEPTH_COMBINE = ("mean", "none")


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
    fractions=DEFAULT_DEPTHS,
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


def subject_on_mesh(subject_dir: str | os.PathLike, mesh: str, hemis=("lh", "rh")) -> dict:
    """A FreeSurfer subject's hemispheres on a template mesh, for display.

    Every surface state the subject has (white, pial, smoothwm, inflated, sphere)
    and its curv / sulc / thickness are carried onto the template's vertices
    through ``?h.sphere.reg`` -- the same placement the projection used, so a
    result made on that template matches by fingerprint. The cortex mask is the
    template's own label. ``"native"`` is just the subject. The hemispheres have
    no file ``paths``: a template copy is not something to write edits back into.
    """
    from fastfuncstuff.io.freesurfer import Hemisphere, load_hemisphere, load_subject

    subject_dir = resolve_subject(subject_dir)
    folder = resolve_mesh(mesh, subject_dir)
    if folder is None:
        return load_subject(subject_dir, tuple(hemis))
    out = {}
    for hemi in hemis:
        if not (subject_dir / "surf" / f"{hemi}.white").exists():
            continue
        native = load_hemisphere(subject_dir, hemi, patches=False)
        reg = read_surface(subject_dir / "surf" / f"{hemi}.sphere.reg")
        tgt = read_surface(folder / "surf" / f"{hemi}.sphere.reg")
        positions = {f"surf:{k}": v.astype(np.float64) for k, v in native.states.items()}
        positions["surf:sphere.reg"] = reg.vertices.astype(np.float64)
        spherical = {"surf:sphere.reg": np.zeros(3)}
        if "surf:sphere" in positions:
            # the display sphere was shifted into scanner space with everything else
            spherical["surf:sphere"] = native.tkr_to_scanner[:3, 3].astype(np.float64)
        bundle = MeshBundle(
            faces=native.faces.astype(np.int64),
            positions=positions,
            scalars={k: v.astype(np.float64) for k, v in native.morph.items()},
            spherical=spherical,
        )
        placed, _ = remesh_via_sphere(bundle, tgt.vertices, tgt.faces)
        states = {
            name.removeprefix("surf:"): placed.positions[name].astype(np.float32)
            for name in positions
            if name != "surf:sphere.reg"
        }
        n = len(tgt.vertices)
        out[hemi] = Hemisphere(
            name=hemi,
            faces=np.asarray(tgt.faces, np.int32),
            states=states,
            tkr_to_scanner=native.tkr_to_scanner,
            morph={k: placed.scalars[k].astype(np.float32) for k in native.morph},
            cortex=_cortex(folder, hemi, n),
        )
    return out


def _voxel_face(source_path: str) -> float:
    """Face area of the source voxel (mm^2), the geometric mean for anisotropic voxels."""
    from fastfuncstuff.io.headers import read_nifti_header

    zooms = np.asarray(read_nifti_header(source_path).get_zooms()[:3], np.float64)
    return float(np.prod(zooms) ** (2.0 / 3.0))


def depth_weights(fractions, weights=None) -> np.ndarray:
    """Normalised per-depth weights: equal by default, else ``weights`` (one per depth)."""
    n = len(fractions)
    w = np.ones(n) if weights is None else np.asarray(weights, np.float64)
    if w.shape != (n,):
        raise ValueError(f"{w.size} depth weights for {n} depths")
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"depth weights must be >= 0 and not all 0, got {list(w)}")
    return w / w.sum()


def output_paths(
    prefix: str,
    meshes,
    hemis,
    fractions,
    depth_combine: str = "mean",
    qc: bool = False,
    geom_prefix: str | None = None,
) -> list[str]:
    """Every file :func:`project_to_surface` writes, in order (for -batch_skip)."""
    out = [f"{prefix}.{space_name(m)}.samples.nii.gz" for m in meshes] if qc else []
    for mesh in meshes:
        for hemi in hemis:
            stem = f"{prefix}.{space_name(mesh)}.{hemi}"
            if depth_combine == "mean" or len(fractions) == 1:
                out.append(f"{stem}.func.gii")
            else:
                out += [f"{stem}.depth-{float(f):.2f}.func.gii" for f in fractions]
            out += [f"{stem}.{n}.shape.gii" for n in ("coverage", "mask", "mean")]
            gstem = f"{geom_prefix or prefix}.{space_name(mesh)}.{hemi}"
            out += [f"{gstem}.{s}.surf.gii" for s in ("white", "pial", "midthickness")]
            if qc:
                out += [
                    f"{stem}.{n}.shape.gii" for n in ("voxel_volume", "blur_fwhm", "noise_ratio")
                ]
    return out


class SurfaceFold:
    """Fold every frame's reads onto every target's vertices as nwarp emits it.

    Per frame and ``(depth, vertex)`` row: the mean of the reads that landed inside
    the EPI (nwarp reads exactly 0 outside) -- renormalising keeps a vertex whose
    footprint pokes out of the slab at its true level instead of averaging in zeros.
    The depths are then combined with ``weights`` (over the depths that read
    anything that frame), and the smallest share of each footprint inside the EPI is
    kept across frames (:attr:`cover`, what the mask is built from).

    It runs on the reads' device as one sparse product per frame, so a frame hands
    back ``V`` values per target (``V + K * V`` with ``per_depth``) instead of every
    read: copying the reads to the host and folding there was half the wall time of
    a GPU pass.
    """

    def __init__(self, targets: list[HemiTarget], weights, per_depth: bool = False):
        from scipy import sparse

        self.targets = targets
        self.per_depth = per_depth
        self.rows = [t.sampling.operator.shape[0] for t in targets]
        self.shapes = [(t.sampling.n_depths, t.sampling.n_vertices) for t in targets]
        self.weights = np.asarray(weights, np.float32)
        self._op_cpu = sparse.block_diag([t.sampling.operator for t in targets], format="csr")
        self._op = None
        self._w = None
        self.cover = None  # (sum K * V,) min share inside, over frames

    def _setup(self, device):
        import torch

        from fastfuncstuff.surface.smooth import torch_csr
        from fastfuncstuff.utils import cpu_if_mps

        dev = cpu_if_mps(device, "sparse_csr_mm")
        if self._op is None or self._op.device != dev:
            self._op = torch_csr(self._op_cpu, dev)
            self._w = torch.as_tensor(self.weights, device=dev)[:, None]
        return dev

    def __call__(self, reads):
        import torch

        dev = self._setup(reads.device)
        r = reads.detach().to(dev, torch.float32).reshape(-1, 1)
        num = (self._op @ r).reshape(-1)
        cov = (self._op @ (r != 0).to(torch.float32)).reshape(-1)
        val = torch.where(cov > 0, num / cov.clamp_min(1e-12), torch.zeros_like(num))
        self.cover = cov if self.cover is None else torch.minimum(self.cover, cov)
        out = []
        for (k, v), vk, ck in zip(
            self.shapes, val.split(self.rows), cov.split(self.rows), strict=True
        ):
            vk, ck = vk.reshape(k, v), ck.reshape(k, v)
            w = self._w * (ck > 0)
            total = w.sum(0)
            out.append(torch.where(total > 0, (w * vk).sum(0) / total.clamp_min(1e-12), 0.0))
            if self.per_depth:
                out.append(vk.reshape(-1))
        return torch.cat(out)

    def reset(self) -> None:
        self.cover = None

    def split(self, out: np.ndarray):
        """``(T, M)`` frames -> per target ``(combined (V, T), per_depth (K, V, T) or
        None, cover (K, V))``."""
        out = out[None] if out.ndim == 1 else out
        cover = self.cover.cpu().numpy() if self.cover is not None else None
        sizes = [v + (k * v if self.per_depth else 0) for k, v in self.shapes]
        chunks = np.split(out, np.cumsum(sizes)[:-1], axis=1)
        covs = np.split(cover, np.cumsum(self.rows)[:-1]) if cover is not None else None
        for i, (t, (k, v), c) in enumerate(zip(self.targets, self.shapes, chunks, strict=True)):
            per = c[:, v:].T.reshape(k, v, -1) if self.per_depth else None
            yield t, np.ascontiguousarray(c[:, :v].T), per, covs[i].reshape(k, v) if covs else None


def project_to_surface(
    source_path: str,
    nwarp_specs: list[str],
    master_path: str,
    subject_dir: str | os.PathLike,
    prefix: str,
    meshes=("native",),
    hemis=("lh", "rh"),
    fractions=DEFAULT_DEPTHS,
    depth_combine: str = "mean",
    depth_weights_: list[float] | None = None,
    sample: str = "footprint",
    verb: int = 1,
    qc: bool = False,
    qc_frames: int = 64,
    geom_prefix: str | None = None,
    **nwarp_kwargs,
) -> list[Path]:
    """Sample ``source`` onto every target mesh through the chain and write GIfTI.

    All targets and hemispheres share one nwarp call, so the chain is composed once
    and native and template outputs come from the same reads of the same data. Per
    target ``{prefix}.{space}.{hemi}`` (space = ``native`` or the template's folder
    name) gets:

    * ``.func.gii`` -- vertices x time, the depths combined (``depth_combine="mean"``:
      equal weights, or ``depth_weights_``), or one ``.depth-{f}.func.gii`` per depth
      with ``"none"``;
    * ``.coverage.shape.gii`` -- the smallest share of the footprint inside the EPI in
      any frame and depth; ``.mask.shape.gii`` -- cortex label (when the mesh has one)
      AND full coverage; ``.mean.shape.gii`` -- the temporal mean (what a cross-run
      mask clips on);
    * ``.white/.pial/.midthickness.surf.gii`` -- the target's vertices placed in THIS
      subject (scanner mm): the geometry smoothing, cluster areas and display need.

    ``geom_prefix`` puts the placed geometry under another stem: it depends only on
    (subject, mesh), so every run of a subject can share one copy.

    ``qc`` adds :func:`surface_qc.surface_qc`'s maps (samples per native voxel,
    effective voxel volume, blur FWHM) from ``qc_frames`` noise volumes.
    """
    from fastfuncstuff.io.afni import get_tr_from_file
    from fastfuncstuff.io.gifti import save_gifti_surface

    from .nwarpforge import nwarpforge

    if sample not in ("footprint", "point"):
        raise ValueError(f"sample must be 'footprint' or 'point', got {sample!r}")
    if depth_combine not in DEPTH_COMBINE:
        raise ValueError(f"depth_combine must be one of {DEPTH_COMBINE}, got {depth_combine!r}")
    weights = depth_weights(fractions, depth_weights_)
    meshes = [meshes] if isinstance(meshes, str) else list(meshes)
    names = [space_name(m) for m in meshes]
    if len(set(names)) != len(names):
        raise ValueError(f"target meshes must have distinct names, got {names}")
    subject_dir = resolve_subject(subject_dir)
    vface = _voxel_face(source_path) if sample == "footprint" else None
    targets = [
        t for mesh in meshes for t in surface_targets(subject_dir, mesh, hemis, fractions, vface)
    ]
    if verb >= 1:
        for t in targets:
            print(
                f"  {t.mesh} {t.hemi}: {t.sampling.n_vertices} vertices x "
                f"{t.sampling.n_depths} depth(s) -> {t.sampling.points.shape[0]} reads ({sample})"
            )
    if qc and nwarp_kwargs.get("source_image") is None:
        from .io import load_image

        nwarp_kwargs["source_image"] = load_image(source_path, device=None)  # read once
    per_depth = depth_combine == "none" and len(fractions) > 1
    fold = SurfaceFold(targets, weights, per_depth=per_depth)
    out = nwarpforge(
        source_path=source_path,
        nwarp_specs=nwarp_specs,
        prefix="",
        master_path=master_path,
        points=np.concatenate([t.sampling.points for t in targets]),
        point_reducer=fold,
        verb=verb,
        **nwarp_kwargs,
    )
    assert out is not None
    tr = get_tr_from_file(source_path)
    written: list[Path] = []
    for t, combined, values, cov in fold.split(out.cpu().numpy()):
        stem = f"{prefix}.{t.mesh}.{t.hemi}"
        gstem = f"{geom_prefix or prefix}.{t.mesh}.{t.hemi}"
        geom = {
            "white": t.white,
            "pial": t.pial,
            "midthickness": 0.5 * (t.white + t.pial),
        }
        for name, pos in geom.items():
            path = Path(f"{gstem}.{name}.surf.gii")
            path.parent.mkdir(parents=True, exist_ok=True)
            save_gifti_surface(path, pos, t.faces, {**t.meta, "surface": name})
        meta = dict(t.meta)
        meta.update(
            source=str(source_path),
            sampling=sample,
            depths=" ".join(f"{f:g}" for f in t.sampling.fractions),
            equivolume="1",
            # Relative to the data file, so the outputs can move together.
            geometry=os.path.relpath(f"{gstem}.midthickness.surf.gii", Path(stem).parent),
        )
        if tr and tr > 0:
            meta["TR_seconds"] = f"{tr:g}"
        if not per_depth:
            path = Path(f"{stem}.func.gii")
            dmeta = {
                "depth_combine": "mean",
                "depth_weights": " ".join(f"{w:.4g}" for w in weights),
            }
            save_gifti_data(path, combined, {**meta, **dmeta})
            written.append(path)
        else:
            assert values is not None
            for k, f in enumerate(t.sampling.fractions):
                path = Path(f"{stem}.depth-{f:.2f}.func.gii")
                save_gifti_data(path, values[k], {**meta, "depth": f"{f:g}"})
                written.append(path)
        cover = cov.min(axis=0)  # (K, V) is already the min over frames
        mask = cover > 0.99
        if t.cortex is not None:
            mask &= t.cortex
        shapes = (
            ("coverage", cover),
            ("mask", mask.astype(np.float32)),
            ("mean", combined.mean(axis=1)),
        )
        for name, vals in shapes:
            path = Path(f"{stem}.{name}.shape.gii")
            save_gifti_data(path, vals.astype(np.float32), meta, time_series=False)
            written.append(path)
        written += [Path(f"{gstem}.{name}.surf.gii") for name in geom]
    if qc:
        from .surface_qc import surface_qc

        written += surface_qc(
            source_path, nwarp_specs, master_path, targets, prefix, weights,
            source_image=nwarp_kwargs["source_image"], n_frames=qc_frames,
            interp=nwarp_kwargs.get("interp", "wsinc5"),
            ainterp=nwarp_kwargs.get("ainterp", "cubic"),
            device=nwarp_kwargs.get("device"), verb=verb, geom_prefix=geom_prefix,
        )  # fmt: skip
    return written
