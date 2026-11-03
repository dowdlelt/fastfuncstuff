"""Volume runs onto a subject's cortical ribbon, as one dataset a voxel tool can fit.

The front end for "here are my volumes, fit the ribbon and give me GIfTI back".
:func:`sample_runs_to_ribbon` samples every run with
:func:`surface_projection.project_to_surface` (the ribbon mean over equivolume
depths) and then **merges the hemispheres into one mesh**, because the tools that
consume this fit one dataset at a time: two hemispheres as two datasets means two
super-grid passes, two noise pools and two of every hyperparameter search. The
merged mesh is two disconnected components, so nothing -- not a heat blur, not a
cluster -- crosses the midline.

The merge is recorded in the merged file's own metadata, so a result computed on
it can be split back into per-hemisphere files by :func:`split_merged_surface`
without the sampling plan being around any more. That is what makes
``{prefix}.lh.func.gii`` land next to a result fitted hours later in another tool.

The chain is the identity: this path assumes the runs are already in register with
the anatomy the surfaces were built on (the usual case after alignment to the
subject's anatomy). Data still in EPI space goes through ``ffs_nwarp -surf``, which
takes the real chain and writes the same per-hemisphere files.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fastfuncstuff.io.gifti import (
    load_gifti_data,
    load_gifti_surface,
    mesh_fingerprint,
    save_gifti_data,
    save_gifti_surface,
)

from .surface_projection import DEFAULT_DEPTHS, project_to_surface, resolve_subject, space_name

__all__ = [
    "MERGED_HEMIS_KEY",
    "RibbonSample",
    "default_master",
    "group_hemisphere_files",
    "hemisphere_of",
    "hemisphere_partner",
    "merge_hemisphere_runs",
    "sample_runs_to_ribbon",
    "split_merged_surface",
]

#: File metadata key holding the per-hemisphere vertex ranges of a merged mesh.
MERGED_HEMIS_KEY = "merged_hemis"

#: Where a FreeSurfer subject keeps a volume in the scanner space its surfaces
#: were built in, best first. ``-surf_master`` overrides it.
_MASTER_CANDIDATES = (
    "SUMA/brain.nii.gz",
    "SUMA/brain.nii",
    "mri/brain.mgz",
    "mri/orig.mgz",
)

_DATA_EXTENSIONS = (".func.gii", ".shape.gii", ".gii")


@dataclass
class RibbonSample:
    """Sampled runs on one merged mesh, and the geometry that describes them."""

    runs: list[str]  # one merged dataset per input run, the fit input
    geometry: str  # the merged midthickness mesh
    mask: str | None  # cortex AND full coverage in every run, merged
    hemis: list[dict]  # hemi, start, stop, mesh_fingerprint, geometry (per hemi)
    space: str  # "native" or the template mesh's folder name

    @property
    def n_vertices(self) -> int:
        return int(self.hemis[-1]["stop"]) if self.hemis else 0


def default_master(subject_dir: str | os.PathLike) -> Path:
    """A volume of the subject in the space its surfaces live in."""
    subject = resolve_subject(subject_dir)
    for rel in _MASTER_CANDIDATES:
        candidate = subject / rel
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{subject} has none of {', '.join(_MASTER_CANDIDATES)}: pass the anatomy the "
        "surfaces were built on explicitly"
    )


def _split_ext(path: str | os.PathLike) -> tuple[str, str]:
    """``("dir/prf", ".func.gii")`` -- the GIfTI extension is two components, not one."""
    text = os.fspath(path)
    for ext in _DATA_EXTENSIONS:
        if text.endswith(ext):
            return text[: -len(ext)], ext
    return text, ""


def _merge_meshes(paths: list[str], out_path: str) -> tuple[list[np.ndarray], str]:
    """Concatenate per-hemisphere meshes into one, returning each one's vertices."""
    vertices, faces, counts = [], [], 0
    for path in paths:
        v, f, _ = load_gifti_surface(path)
        faces.append(np.asarray(f, np.int32) + counts)
        vertices.append(np.asarray(v, np.float32))
        counts += len(v)
    merged_faces = np.concatenate(faces).astype(np.int32)
    merged_vertices = np.concatenate(vertices)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_gifti_surface(out_path, merged_vertices, merged_faces, {"surface": "midthickness"})
    return vertices, mesh_fingerprint(merged_faces, len(merged_vertices))


def _merge_data(
    paths: list[str], sizes: list[int], out_path: str, meta: dict[str, str], time_series: bool
) -> None:
    """Stack per-hemisphere arrays vertex-wise, one hemisphere alive at a time.

    A run's two hemispheres are the same size as the volume they came from, so the
    merge fills one preallocated array rather than concatenating a list of them.
    """
    merged: np.ndarray | None = None
    at = 0
    for path, size in zip(paths, sizes, strict=True):
        block, _ = load_gifti_data(path)
        block = block[:, None] if block.ndim == 1 else block
        if merged is None:
            merged = np.empty((sum(sizes), block.shape[1]), np.float32)
        merged[at : at + size] = block
        at += size
        del block
    assert merged is not None
    save_gifti_data(out_path, merged if merged.shape[1] > 1 else merged[:, 0], meta, time_series)


def _geometry(
    geom_prefix: str, space: str, hemis, merged_geometry: str, out: Path
) -> tuple[list[dict], list[int], str]:
    """Merge the placed per-hemisphere meshes and describe where each one landed."""
    paths = [f"{geom_prefix}.{space}.{hemi}.midthickness.surf.gii" for hemi in hemis]
    vertices, fingerprint = _merge_meshes(paths, merged_geometry)
    sizes = [len(block) for block in vertices]
    bounds, at = [], 0
    for hemi, size, path in zip(hemis, sizes, paths, strict=True):
        bounds.append(
            {
                "hemi": hemi,
                "start": at,
                "stop": at + size,
                "mesh_fingerprint": load_gifti_surface(path)[2]["mesh_fingerprint"],
                "geometry": os.path.relpath(path, out),
            }
        )
        at += size
    return bounds, sizes, fingerprint


def _reusable(merged: str, sidecar: Path, source: str, settings: dict) -> bool:
    """Was this run already sampled the same way, from a source no newer than it?"""
    if not Path(merged).is_file() or not sidecar.is_file():
        return False
    try:
        recorded = json.loads(sidecar.read_text())
    except ValueError:
        return False
    if recorded != {**settings, "source": source}:
        return False
    return Path(merged).stat().st_mtime >= Path(source).stat().st_mtime


def sample_runs_to_ribbon(
    sources: list[str],
    subject_dir: str | os.PathLike,
    out_dir: str | os.PathLike,
    master_path: str | os.PathLike | None = None,
    mesh: str = "native",
    hemis=("lh", "rh"),
    fractions=DEFAULT_DEPTHS,
    sample: str = "footprint",
    keep_hemi_runs: bool = False,
    reuse: bool = True,
    verb: int = 1,
    **nwarp_kwargs,
) -> RibbonSample:
    """Sample every run onto the subject's ribbon and merge the hemispheres.

    ``out_dir`` collects the sampled runs, the placed geometry (shared by every
    run: it depends only on subject and mesh) and the fit mask, which is cortex
    AND full footprint coverage **in every run** -- a vertex the EPI leaves in
    one run is not fit in any.

    ``reuse`` keeps a run that was already sampled with the same subject, mesh,
    depths and sampling mode, recorded in a sidecar beside it. Sampling is minutes
    per run, and a second pass over the same data (another model, another HRF
    setting) should not pay for it twice.
    """
    subject = resolve_subject(subject_dir)
    master = Path(master_path) if master_path else default_master(subject)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    space = space_name(mesh)
    hemis = tuple(hemis)

    # The composer refuses an empty chain, so the identity is a real one-link
    # chain: a file on disk, which also leaves the assumption visible afterwards.
    identity = out / "identity.aff12.1D"
    identity.write_text("1 0 0 0 0 1 0 0 0 0 1 0\n")
    geom_prefix = str(out / "geom")

    if verb >= 1:
        print(f"  Sampling {len(sources)} run(s) onto {subject.name} ({space}, {'+'.join(hemis)})")
        print(f"    master (assumed to be -input's space): {master}")

    runs: list[str] = []
    coverage: dict[str, np.ndarray] = {}
    merged_geometry = str(out / f"geom.{space}.midthickness.surf.gii")
    geometry: tuple[list[dict], list[int], str] | None = None
    settings = {
        "subject": str(subject),
        "mesh": space,
        "hemis": list(hemis),
        "depths": [float(f) for f in fractions],
        "sampling": sample,
    }
    for index, source in enumerate(sources, start=1):
        prefix = str(out / f"run-{index:02d}")
        merged = str(out / f"run-{index:02d}.{space}.func.gii")
        sidecar = Path(f"{prefix}.{space}.sample.json")
        reused = reuse and _reusable(merged, sidecar, str(source), settings)
        if not reused:
            project_to_surface(
                source_path=str(source),
                nwarp_specs=[str(identity)],
                master_path=str(master),
                subject_dir=subject,
                prefix=prefix,
                meshes=(mesh,),
                hemis=hemis,
                fractions=fractions,
                sample=sample,
                # Not quieter than the caller: sampling a run is minutes of work,
                # and project_to_surface's own spinner is its only sign of life.
                verb=verb,
                geom_prefix=geom_prefix,
                **nwarp_kwargs,
            )
        # The placed geometry depends only on (subject, mesh), so every run shares
        # one copy and the merge happens once.
        if geometry is None:
            geometry = _geometry(geom_prefix, space, hemis, merged_geometry, out)
        bounds, sizes, fingerprint = geometry

        for hemi in hemis:
            mask, _ = load_gifti_data(f"{prefix}.{space}.{hemi}.mask.shape.gii")
            keep = np.asarray(mask, np.float32) > 0
            coverage[hemi] = keep if hemi not in coverage else (coverage[hemi] & keep)
        if reused:
            runs.append(merged)
            if verb >= 1:
                print(f"    {Path(str(source)).name}: reusing {Path(merged).name}")
            continue

        per_hemi = [f"{prefix}.{space}.{hemi}.func.gii" for hemi in hemis]
        source_meta = load_gifti_data(per_hemi[0])[1]
        meta = {
            key: source_meta[key]
            for key in ("subject", "mesh", "depths", "equivolume", "sampling", "TR_seconds")
            if key in source_meta
        }
        meta.update(
            source=str(source),
            mesh_fingerprint=fingerprint,
            geometry=os.path.relpath(merged_geometry, out),
            **{MERGED_HEMIS_KEY: json.dumps(bounds)},
        )
        _merge_data(per_hemi, sizes, merged, meta, time_series=True)
        sidecar.write_text(json.dumps({**settings, "source": str(source)}, indent=1))
        runs.append(merged)
        if not keep_hemi_runs:
            # The merged file is the reusable artifact; the per-hemisphere copy of
            # the same time series would double the sampled data on disk.
            for path in per_hemi:
                Path(path).unlink(missing_ok=True)
        if verb >= 1:
            print(f"    {Path(str(source)).name} -> {Path(merged).name}")

    mask_path = None
    if coverage:
        mask_path = str(out / f"mask.{space}.shape.gii")
        values = np.concatenate([coverage[hemi] for hemi in hemis]).astype(np.float32)
        save_gifti_data(
            mask_path,
            values,
            {
                "mesh_fingerprint": fingerprint,
                "geometry": os.path.relpath(merged_geometry, out),
                MERGED_HEMIS_KEY: json.dumps(bounds),
            },
            time_series=False,
        )
        if verb >= 1:
            print(f"    fit mask: {int(values.sum())} of {values.size} vertices (cortex, covered)")

    return RibbonSample(
        runs=runs, geometry=merged_geometry, mask=mask_path, hemis=bounds, space=space
    )


def split_merged_surface(path: str | os.PathLike, out_prefix: str | None = None) -> list[str]:
    """Split a result on a merged mesh into one file per hemisphere.

    Reads the vertex ranges from the file's own ``merged_hemis`` metadata, so this
    works on any file that mesh produced, whenever it was produced. Each output
    carries its hemisphere's fingerprint and geometry pointer, so a reader can
    still refuse to pair it with the wrong mesh. Array names (the bucket labels)
    and intents are copied across untouched.
    """
    import nibabel as nib

    img = nib.load(os.fspath(path))
    assert isinstance(img, nib.gifti.GiftiImage)
    meta = dict(img.meta)
    if MERGED_HEMIS_KEY not in meta:
        raise ValueError(f"{path} is not on a merged mesh ({MERGED_HEMIS_KEY} metadata missing)")
    bounds = json.loads(meta[MERGED_HEMIS_KEY])
    stem, ext = _split_ext(out_prefix or path)
    written = []
    for part in bounds:
        start, stop = int(part["start"]), int(part["stop"])
        out_meta = {k: v for k, v in meta.items() if k != MERGED_HEMIS_KEY}
        out_meta["mesh_fingerprint"] = part["mesh_fingerprint"]
        out_meta["geometry"] = os.path.relpath(
            os.path.join(os.path.dirname(os.path.abspath(os.fspath(path))), part["geometry"]),
            os.path.dirname(os.path.abspath(f"{stem}.{part['hemi']}{ext}")) or ".",
        )
        out_meta["hemisphere"] = part["hemi"]
        arrays = []
        for array in img.darrays:
            arrays.append(
                nib.gifti.GiftiDataArray(
                    np.ascontiguousarray(np.asarray(array.data)[start:stop], np.float32),
                    intent=array.intent,
                    datatype="NIFTI_TYPE_FLOAT32",
                    meta=dict(array.meta),
                )
            )
        out_path = f"{stem}.{part['hemi']}{ext}"
        nib.save(
            nib.gifti.GiftiImage(darrays=arrays, meta=nib.gifti.GiftiMetaData(out_meta)), out_path
        )
        written.append(out_path)
    return written


#: A hemisphere named as its own dot/underscore/dash-delimited component of a
#: filename, which is how every ffs tool writes one (``...onavg-ico64.lh.func.gii``).
_HEMI_COMPONENT = re.compile(r"(?<![A-Za-z0-9])(lh|rh)(?![A-Za-z0-9])")


def hemisphere_of(path: str | os.PathLike) -> tuple[str, str] | None:
    """``("lh", "<the name with lh blanked out>")``, or None when it names neither.

    The second element is the key two hemispheres of the same data share, so
    grouping by it pairs them.

    Only the file's own name is read, so a parent directory called ``lh/`` is not
    a claim about the file in it. Within the name, two hemisphere components
    (``sub.lh.rh.func.gii``) is refused rather than guessed: taking the first is
    how one hemisphere's data ends up labelled as the other.
    """
    name = Path(os.fspath(path)).name
    found = list(_HEMI_COMPONENT.finditer(name))
    if len(found) != 1:
        return None
    at = found[0]
    parent = str(Path(os.fspath(path)).parent)
    return at.group(1), f"{parent}/{name[: at.start()]}\x00{name[at.end() :]}"


def hemisphere_partner(path: str | os.PathLike, hemi: str) -> str:
    """The same path with its hemisphere component replaced by ``hemi``."""
    found = hemisphere_of(path)
    if found is None:
        raise ValueError(f"{path}: names no single hemisphere (lh/rh) to pair")
    name = Path(os.fspath(path)).name
    at = next(_HEMI_COMPONENT.finditer(name))
    return str(Path(os.fspath(path)).parent / f"{name[: at.start()]}{hemi}{name[at.end() :]}")


def group_hemisphere_files(
    paths: list[str], hemis=("lh", "rh"), find_partner: bool = True
) -> tuple[list[list[str]], list[str]] | None:
    """``([per-run files in hemi order], hemis)`` for single-hemisphere inputs.

    Two shapes of input reach the same place. Both hemispheres listed (ten files
    for five runs) are grouped by the key they share; one hemisphere listed is
    completed from disk.

    None is the ordinary "this is not a pair" answer and never an error. It covers
    a merged dataset, volumes, a hemisphere whose partner is simply not there, and
    a one-hemisphere analysis someone means to run as one -- a dataset with no rh
    (or no lh) at all is a thing that happens, and pairing is a convenience, so it
    declines and the caller fits exactly what it was given.

    Run order follows the order the paths were given, taking each run's first
    appearance, so a fit's runs stay in the order the user listed them (-stimulus
    and -stim_groups are positional against it).
    """
    seen = [hemisphere_of(path) for path in paths]
    if not paths or any(found is None for found in seen):
        return None
    by_key: dict[str, dict[str, str]] = {}
    for path, found in zip(paths, seen, strict=True):
        assert found is not None
        hemi, key = found
        by_key.setdefault(key, {})[hemi] = path
    present = {hemi for run in by_key.values() for hemi in run}
    if not present <= set(hemis):
        return None
    if len(present) > 1:  # both listed: group them, complete nothing
        order = [hemi for hemi in hemis if hemi in present]
        missing = [
            f"{Path(run[next(iter(run))]).name} has no {hemi}"
            for run in by_key.values()
            for hemi in order
            if hemi not in run
        ]
        if missing:
            # Not a one-hemisphere dataset (which declines below, and is fine) but a
            # list that pairs some runs and not others: as one dataset those runs
            # would have different vertex counts, and quietly dropping the odd ones
            # would fit fewer runs than were asked for.
            raise FileNotFoundError(
                "-input lists both hemispheres for some runs and one for others ("
                + "; ".join(missing[:3])
                + "). List one hemisphere and the partner is found beside it, list both "
                "for every run, or pass -hemi_pair off to fit exactly what was given"
            )
        return [[run[hemi] for hemi in order] for run in by_key.values()], order
    if not find_partner:
        return None
    # One hemisphere listed: the other is the file beside it, or there is no pair.
    grouped = []
    for run in by_key.values():
        have = next(iter(run))
        files = []
        for hemi in hemis:
            path = run[have] if hemi == have else hemisphere_partner(run[have], hemi)
            if not Path(path).is_file():
                return None
            files.append(path)
        grouped.append(files)
    return grouped, list(hemis)


def _hemi_geometry(per_hemi: list[str], hemis: list[str], out: Path, space: str):
    """Merge the geometry the per-hemisphere data points at, when it is findable.

    Each file written by ``ffs_nwarp -surf`` carries a relative ``geometry``
    pointer in its own metadata, so the mesh the fit should blur on can be found
    without being named. When it cannot, the merge still happens -- the fit only
    needs a mesh for the operations that are geometric -- and ``geometry`` is None
    so the caller can refuse those specifically.
    """
    paths = []
    for path in per_hemi:
        meta = load_gifti_data(path)[1]
        pointer = meta.get("geometry")
        if not pointer:
            return None, None
        resolved = Path(path).parent / pointer
        if not resolved.is_file():
            return None, None
        paths.append(str(resolved))
    merged = str(out / f"geom.{space}.midthickness.surf.gii")
    vertices, fingerprint = _merge_meshes(paths, merged)
    bounds, at = [], 0
    for hemi, block, path in zip(hemis, vertices, paths, strict=True):
        bounds.append(
            {
                "hemi": hemi,
                "start": at,
                "stop": at + len(block),
                "mesh_fingerprint": load_gifti_surface(path)[2]["mesh_fingerprint"],
                "geometry": os.path.relpath(path, out),
            }
        )
        at += len(block)
    return (bounds, fingerprint), merged


def merge_hemisphere_runs(
    grouped: list[list[str]],
    hemis: list[str],
    out_dir: str | os.PathLike,
    space: str = "merged",
    mask: list[str] | None = None,
    reuse: bool = True,
    verb: int = 1,
) -> RibbonSample:
    """Already-sampled per-hemisphere runs, merged into the one dataset a fit reads.

    The same destination :func:`sample_runs_to_ribbon` reaches from volumes, and
    deliberately the same :class:`RibbonSample`: every consumer -- the fit, the
    mask, ``split_ribbon_outputs`` putting the results back into .lh/.rh files --
    then works without knowing which route the vertices came by. The hemispheres
    are one dataset because nothing crosses the midline anyway (the components are
    not connected), so one super-grid pass, one HRF selection and one noise pool
    serve both instead of two of each.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    geometry, merged_geometry = _hemi_geometry(grouped[0], hemis, out, space)
    sizes = [load_gifti_data(path)[0].shape[0] for path in grouped[0]]
    if geometry is None:
        # No mesh to point at: the ranges still come from the data, and each
        # hemisphere's own fingerprint keeps a result from being paired wrongly.
        bounds, at = [], 0
        for hemi, size, path in zip(hemis, sizes, grouped[0], strict=True):
            meta = load_gifti_data(path)[1]
            bounds.append(
                {
                    "hemi": hemi,
                    "start": at,
                    "stop": at + size,
                    "mesh_fingerprint": meta.get("mesh_fingerprint", ""),
                    "geometry": meta.get("geometry", ""),
                }
            )
            at += size
        fingerprint = ""
    else:
        bounds, fingerprint = geometry

    if verb >= 1:
        shape = " + ".join(f"{hemi} {size}" for hemi, size in zip(hemis, sizes, strict=True))
        print(f"  Pairing {len(grouped)} run(s) x {len(hemis)} hemisphere(s): {shape} vertices")

    runs = []
    for index, per_hemi in enumerate(grouped, start=1):
        merged = str(out / f"run-{index:02d}.{space}.func.gii")
        if reuse and Path(merged).is_file() and _no_older_than(merged, per_hemi):
            runs.append(merged)
            if verb >= 1:
                print(f"    reusing {Path(merged).name}")
            continue
        source_meta = load_gifti_data(per_hemi[0])[1]
        meta = {
            key: source_meta[key]
            for key in ("subject", "mesh", "depths", "equivolume", "sampling", "TR_seconds")
            if key in source_meta
        }
        meta.update(
            source=os.pathsep.join(per_hemi),
            mesh_fingerprint=fingerprint,
            **({"geometry": os.path.relpath(merged_geometry, out)} if merged_geometry else {}),
            **{MERGED_HEMIS_KEY: json.dumps(bounds)},
        )
        _merge_data(per_hemi, sizes, merged, meta, time_series=True)
        runs.append(merged)
        if verb >= 1:
            print(f"    {' + '.join(Path(p).name for p in per_hemi)} -> {Path(merged).name}")

    mask_path = None
    if mask:
        mask_path = str(out / f"mask.{space}.shape.gii")
        meta = {"mesh_fingerprint": fingerprint, MERGED_HEMIS_KEY: json.dumps(bounds)}
        if merged_geometry:
            meta["geometry"] = os.path.relpath(merged_geometry, out)
        _merge_data(mask, sizes, mask_path, meta, time_series=False)
        if verb >= 1:
            kept = int((np.asarray(load_gifti_data(mask_path)[0]) > 0).sum())
            print(f"    mask: {kept} of {sum(sizes)} vertices")

    return RibbonSample(
        runs=runs, geometry=merged_geometry or "", mask=mask_path, hemis=bounds, space=space
    )


def _no_older_than(merged: str, sources: list[str]) -> bool:
    """Is ``merged`` at least as new as every file it was made from?"""
    stamp = Path(merged).stat().st_mtime
    return all(Path(path).stat().st_mtime <= stamp for path in sources)
