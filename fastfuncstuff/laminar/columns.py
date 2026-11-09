"""Cortical columns made on the fly: overlapping geodesic disks of the ribbon,
each read through depth, with its own measured depth PSF.

The laminar model wants one depth-resolved response per region. Instead of a
region drawn ahead of time, cortex is covered by many overlapping disks centred
on responsive cortex, so every responsive vertex is fitted by several regions
that differ only in where their edges fall.

Everything is linear, so nothing is read per column:

1. the ribbon under all columns is tiled once into cells (:mod:`surface.slab`),
   each owned by a vertex and split over depth bins;
2. one sparse matrix reads the data grid into **vertex x depth** sums (cell
   volume x interpolated value), with their volumes and depth-PSF numerators
   (:func:`processing.roi_depth.mix_rows`);
3. a column is a set of vertices, so its curves, coverage and PSF are ratios of
   sums over its members: one sparse product for every column at once.

Step 3 is exactly the column read ``processing.roi_depth`` would do with the
column as an ROI: same cells, same weights, summed in a different order.
Fitting voxel curves first and sampling them after is exact too, because the
curves come from a linear estimator (a fixed-lambda smooth FIR,
``ffs_deconvolve -tent-smooth global``), so the column curve of the fits equals
the fit of the column mean.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from os import PathLike
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from fastfuncstuff.processing.roi_depth import (
    DepthOptions,
    _compose,
    _Roi,
    default_step,
    mix_rows,
    plan_depth_multi,
    psf_halo_mm,
)

_KEY = 1 << 40


def responsive_vertices(
    path: str | Sequence[str],
    surfaces,
    *,
    threshold: float | None = None,
    absolute: bool = True,
    fracs: tuple[float, ...] = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85),
    cluster_size: int | None = None,
    cluster_nn: int = 1,
    notes: list[str] | None = None,
) -> list[np.ndarray]:
    """Per surface, the vertices a responsive-cortex map marks.

    ``path`` is either a VOLUME (``file[N]`` picks brick N of a 4-D file) or one
    or two per-vertex ``.gii`` files, which are matched to the hemispheres by
    their metadata, then their filename, then their vertex count.

    From a volume, each column is read nearest-voxel at ``fracs`` (white 0 ..
    pial 1) and passes if ANY read does -- any depth, because a responsive patch
    need not span the ribbon (a thresholded map that peaks by pial). A surface
    map already is per vertex, so nothing is sampled and ``fracs`` is unused:
    that is the reason to prefer one when you have it, since the volume route
    spends a nearest-voxel read per vertex per depth to recover what the surface
    map states directly.

    A vertex passes if the value is non-zero (mask) or ``|value| >= threshold``
    (``value >= threshold`` with ``absolute=False``). The medial wall never does.

    ``cluster_size`` drops suprathreshold clusters smaller than that many
    SAMPLES before any column is placed -- voxels for a volume, vertices for a
    surface map, where ``cluster_nn`` does not apply because mesh adjacency is
    the only connectivity there (the note reports the kept area in mm^2 so the
    count can be calibrated). For a volume, ``cluster_nn`` 1/2/3 counts faces,
    faces + edges, or faces + edges + corners, as AFNI's NN does.

    Why it matters: a lone voxel at ``|z| >= 3`` is mostly a false positive, and
    ONE responsive vertex is enough to place a column -- which then claims every
    vertex within its radius. An unfiltered threshold therefore lets single noise
    voxels license whole neighbourhood fits, and the cost is paid in model
    inversions as well as in interpretation.
    """
    import re

    from fastfuncstuff.io.gifti import is_gifti

    paths = [path] if isinstance(path, (str, PathLike)) else [str(q) for q in path]
    if all(is_gifti(q) for q in paths):
        return _responsive_from_surfaces(
            paths, surfaces, threshold=threshold, absolute=absolute,
            cluster_size=cluster_size, notes=notes,
        )  # fmt: skip
    if any(is_gifti(q) for q in paths):
        raise ValueError(f"-responsive: mixed surface and volume inputs {paths}")
    if len(paths) != 1:
        raise ValueError(f"-responsive: one volume, or one .gii per hemisphere; got {paths}")
    path = paths[0]

    from fastfuncstuff.io.afni import load_nifti

    m = re.fullmatch(r"(.+)\[(\d+)\]", path)
    brick = int(m.group(2)) if m else 0
    img = load_nifti(m.group(1) if m else path)
    arr = np.asanyarray(img.dataobj)
    if arr.ndim == 4:
        arr = arr[..., brick]
    elif brick:
        raise ValueError(f"{path}: a 3-D volume has no brick {brick}")
    arr = np.asarray(arr, np.float64)
    if notes is not None and threshold is not None:
        notes.append(f"-responsive: brick {brick} of {path}")
        # An omnibus Full_Fstat / Full_Zstat is non-negative by construction, so asking for
        # the positive side of one is a no-op -- and such a brick marks every condition's
        # response, not the contrast people mean to localise with.  Brick 0 of an AFNI
        # stats bucket IS the omnibus, which is what a forgotten selector lands on.
        finite = arr[np.isfinite(arr)]
        if finite.size and finite.min() >= 0:
            notes.append(
                f"    WARNING: brick {brick} has no negative values, so it looks like an "
                "omnibus F/Z (every condition against baseline) rather than a signed "
                "contrast.  -responsive_positive does nothing on it, and it will mark far "
                "more cortex than a contrast would.  Check the brick with 3dinfo -label."
            )
    # Threshold the volume once rather than each vertex read: identical result, and it
    # is what a cluster filter needs to act on.
    if threshold is None:
        keep = arr != 0
    else:
        keep = (np.abs(arr) if absolute else arr) >= threshold
    if cluster_size is not None and cluster_size > 1:
        keep = _drop_small_clusters(keep, int(cluster_size), int(cluster_nn), notes)
    elif notes is not None and threshold is not None:
        # Costs one labelling and answers "would a cluster filter matter here?" without
        # making the user run the job twice.  On a real localiser the answer was yes:
        # 96% of the clusters were specks.
        notes.append(_speck_census(keep))
    inv = np.linalg.inv(np.asarray(img.affine, np.float64))
    out = []
    for s in surfaces:
        w = np.asarray(s.white, np.float64)
        p = np.asarray(s.pial, np.float64)
        hit = np.zeros(len(w), bool)
        for frac in fracs:
            ijk = np.rint((w + frac * (p - w)) @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
            inside = np.all((ijk >= 0) & (ijk < np.asarray(keep.shape[:3])), axis=1)
            v = np.zeros(len(w), bool)
            v[inside] = keep[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]
            hit |= v
        if s.cortex is not None:
            hit &= np.asarray(s.cortex, bool)
        out.append(hit)
    return out


def _responsive_from_surfaces(
    paths: list[str],
    surfaces,
    *,
    threshold: float | None,
    absolute: bool,
    cluster_size: int | None,
    notes: list[str] | None,
) -> list[np.ndarray]:
    """:func:`responsive_vertices` from per-vertex ``.gii`` maps, one per hemisphere."""
    import re

    from fastfuncstuff.io.gifti import load_gifti_data

    loaded = []
    for q in paths:
        data, meta = load_gifti_data(q)
        data = np.asarray(data, np.float64)
        if data.ndim > 1:  # a multi-column file: the first column, as a stat map
            data = data[:, 0]
        loaded.append((q, data, meta))

    assigned: dict[int, tuple[str, np.ndarray]] = {}
    for q, data, meta in loaded:
        primary = str(meta.get("AnatomicalStructurePrimary", "")).lower()
        name = Path(q).name.lower()
        want = None
        if "left" in primary or re.search(r"(^|[^a-z])lh([^a-z]|$)", name):
            want = "lh"
        elif "right" in primary or re.search(r"(^|[^a-z])rh([^a-z]|$)", name):
            want = "rh"
        hits = [
            h for h, s in enumerate(surfaces)
            if len(s.white) == data.size and (want is None or s.name == want)
        ]  # fmt: skip
        if not hits:
            have = ", ".join(f"{s.name} {len(s.white):,}" for s in surfaces)
            raise ValueError(
                f"{q}: {data.size:,} values match no surface"
                + (f" named {want}" if want else "")
                + f" (have {have})"
            )
        if len(hits) > 1:
            raise ValueError(
                f"{q}: {data.size:,} values fit several surfaces and the file says which "
                "hemisphere it is in neither its metadata nor its name; rename it lh/rh"
            )
        h = hits[0]
        if h in assigned:
            raise ValueError(f"{q} and {assigned[h][0]} are both the {surfaces[h].name} map")
        assigned[h] = (q, data)
    if notes is not None:
        notes.append(
            "-responsive: surface maps "
            + ", ".join(f"{surfaces[h].name} {Path(q).name}" for h, (q, _) in sorted(assigned.items()))
        )  # fmt: skip

    out, census = [], []
    for h, s in enumerate(surfaces):
        if h not in assigned:
            out.append(np.zeros(len(s.white), bool))
            continue
        v = assigned[h][1]
        hit = v != 0 if threshold is None else (np.abs(v) if absolute else v) >= threshold
        if s.cortex is not None:
            hit = hit & np.asarray(s.cortex, bool)
        if cluster_size is not None and cluster_size > 1:
            hit, note = _drop_small_surface_clusters(hit, s, int(cluster_size))
        else:
            note = _surface_speck_census(hit, s)
        census.append(f"{s.name}: {note}")
        out.append(hit)
    if notes is not None and census:
        notes += [f"    {c}" for c in census]
    return out


def _surface_components(hit: np.ndarray, surf) -> tuple[np.ndarray, int]:
    """Connected components of the marked vertices over the mesh edges."""
    import scipy.sparse as sp
    from scipy.sparse import csgraph

    faces = np.asarray(surf.faces, np.int64)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    both = np.asarray(hit, bool)[e].all(1)
    e = e[both]
    n = len(hit)
    adj = sp.coo_matrix(
        (np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)
    ).tocsr()  # fmt: skip
    n_comp, labels = csgraph.connected_components(adj + adj.T, directed=False)
    labels = np.where(hit, labels, -1)
    return labels, n_comp


def _cluster_sizes(labels: np.ndarray) -> np.ndarray:
    """Vertex count per component label (index 0 is label 0), zero for unmarked."""
    marked = labels >= 0
    return np.bincount(labels[marked], minlength=int(labels.max()) + 1 if marked.any() else 1)


def _drop_small_surface_clusters(hit, surf, min_vertices: int) -> tuple[np.ndarray, str]:
    from fastfuncstuff.surface.mesh import vertex_areas

    labels, _ = _surface_components(hit, surf)
    if not hit.any():
        return hit, "no suprathreshold vertices"
    sizes = _cluster_sizes(labels)
    big = sizes >= min_vertices
    out = hit & (labels >= 0) & big[np.clip(labels, 0, None)]
    area = vertex_areas(np.asarray(surf.white, np.float64), np.asarray(surf.faces, np.int64),
                        len(hit))  # fmt: skip
    return out, (
        f"cluster filter (mesh, >= {min_vertices} vertices): kept {int(big.sum()):,} of "
        f"{int((sizes > 0).sum()):,} clusters, {int(out.sum()):,} of {int(hit.sum()):,} "
        f"vertices ({area[out].sum():.0f} of {area[hit].sum():.0f} mm2)"
    )


def _surface_speck_census(hit, surf, min_vertices: int = 5) -> str:
    if not hit.any():
        return "no suprathreshold vertices"
    labels, _ = _surface_components(hit, surf)
    sizes = _cluster_sizes(labels)
    present = sizes > 0
    small = int((present & (sizes < min_vertices)).sum())
    return (
        f"no cluster filter: {int(hit.sum()):,} suprathreshold vertices in "
        f"{int(present.sum()):,} clusters, of which {small:,} are under {min_vertices} "
        "vertices -- see -responsive_cluster_size"
    )


def _drop_small_clusters(
    keep: np.ndarray, min_voxels: int, nn: int, notes: list[str] | None = None
) -> np.ndarray:
    """``keep`` with connected components under ``min_voxels`` voxels removed.

    ``nn`` is AFNI's: 1 = faces (6-connected), 2 = faces + edges (18),
    3 = faces + edges + corners (26).
    """
    from scipy import ndimage

    if nn not in (1, 2, 3):
        raise ValueError(f"cluster_nn must be 1, 2 or 3 (AFNI's NN), not {nn}")
    structure = ndimage.generate_binary_structure(3, nn)
    labels, n = ndimage.label(keep, structure=structure)
    if n == 0:
        return keep
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0  # background
    big = sizes >= min_voxels
    out = big[labels]
    if notes is not None:
        notes.append(
            f"    cluster filter (NN{nn}, >= {min_voxels} voxels): kept "
            f"{int(big.sum()):,} of {n:,} clusters, {int(out.sum()):,} of "
            f"{int(keep.sum()):,} suprathreshold voxels"
        )
    return out


def _speck_census(keep: np.ndarray, min_voxels: int = 5, nn: int = 1) -> str:
    """How much of a suprathreshold mask is specks, as a one-line note."""
    from scipy import ndimage

    labels, n = ndimage.label(keep, structure=ndimage.generate_binary_structure(3, nn))
    if n == 0:
        return "    no suprathreshold voxels"
    sizes = np.bincount(labels.reshape(-1))
    sizes[0] = 0
    small = int(((sizes < min_voxels) & (sizes > 0)).sum())
    return (
        f"    no cluster filter: {int(keep.sum()):,} suprathreshold voxels in {n:,} "
        f"NN{nn} clusters, of which {small:,} ({100 * small / n:.0f}%) are under "
        f"{min_voxels} voxels.  Each one that touches the ribbon places a column and "
        f"claims every vertex within -column_radius -- see -responsive_cluster_size."
    )


@dataclass
class ColumnSet:
    """Columns as vertex sets. ``members`` is ``(n_columns, n_vertices)`` over
    :attr:`vertices` (a ``(surface, vertex)`` list shared with the reads)."""

    centre_surface: np.ndarray  # (C,)
    centre_vertex: np.ndarray  # (C,)
    members: sp.csr_matrix  # (C, n_v) 0/1
    vertex_surface: np.ndarray  # (n_v,)
    vertex_id: np.ndarray  # (n_v,)
    radius_mm: float
    spacing_mm: float
    #: Region names (atlas labels); None for disks and clusters, which are numbered.
    names: list[str] | None = None

    @property
    def n_columns(self) -> int:
        return int(self.centre_vertex.size)

    def coverage_count(self) -> np.ndarray:
        """How many columns contain each vertex of :attr:`vertex_id`."""
        return np.asarray(self.members.sum(axis=0)).ravel()


def _midthickness(s) -> np.ndarray:
    return 0.5 * (np.asarray(s.white, np.float64) + np.asarray(s.pial, np.float64))


def make_columns(
    surfaces,
    responsive: list[np.ndarray],
    *,
    radius_mm: float,
    spacing_mm: float | None = None,
    seed: int = 0,
) -> ColumnSet:
    """Disks of geodesic ``radius_mm`` (on the midthickness) centred so that every
    responsive vertex is within ``spacing_mm`` of a centre.

    Centres are a greedy cover of the responsive vertices in a random order (a
    new centre wherever the cover has not reached), so their number is set by the
    responsive area and the spacing -- the data decide how many columns there
    are. ``spacing_mm`` defaults to half the radius: every vertex well inside
    responsive cortex then falls in several columns, and a column edge never
    decides a vertex alone. A column takes its whole disk, responsive or not:
    the responsive mask places columns, it does not trim them.
    """
    from fastfuncstuff.surface.mesh import MeshTopology, geodesic_ball

    spacing = float(spacing_mm) if spacing_mm else 0.5 * float(radius_mm)
    rng = np.random.default_rng(seed)
    cs, cv, member_lists = [], [], []
    for h, (s, resp) in enumerate(zip(surfaces, responsive, strict=True)):
        cand = np.flatnonzero(resp)
        if cand.size == 0:
            continue
        mid = _midthickness(s)
        topo = MeshTopology.from_faces(np.asarray(s.faces), mid.shape[0])
        cortex = None if s.cortex is None else np.asarray(s.cortex, bool)
        covered = np.zeros(mid.shape[0], bool)
        for v in rng.permutation(cand):
            if covered[v]:
                continue
            near, _ = geodesic_ball(mid, topo, int(v), spacing)
            covered[near] = True
            disk, _ = geodesic_ball(mid, topo, int(v), float(radius_mm))
            if cortex is not None:
                disk = disk[cortex[disk]]
            cs.append(h)
            cv.append(int(v))
            member_lists.append(h * _KEY + disk.astype(np.int64))
    if not cs:
        raise ValueError("no responsive vertex to centre a column on")
    keys = np.unique(np.concatenate(member_lists))
    rows = np.repeat(np.arange(len(member_lists)), [m.size for m in member_lists])
    cols = np.searchsorted(keys, np.concatenate(member_lists))
    members = sp.csr_matrix(
        (np.ones(cols.size), (rows, cols)), shape=(len(member_lists), keys.size)
    )
    return ColumnSet(
        np.asarray(cs), np.asarray(cv), members, (keys // _KEY).astype(np.int64),
        (keys % _KEY).astype(np.int64), float(radius_mm), spacing,
    )  # fmt: skip


def regions_from_map(
    surfaces,
    path: str,
    threshold: float,
    *,
    absolute: bool = False,
    min_area_mm2: float = 20.0,
    close_mm: float = 1.0,
) -> ColumnSet:
    """Regions as the connected surface clusters of a statistic: every patch where
    ``path`` (read along each column, as :func:`responsive_vertices`) passes
    ``threshold`` becomes one region, read through depth like a column.

    For a selectivity map (faces > objects at t > 3.1) this gives the data's own
    patches, typically several mm across, instead of disks of a chosen size. The
    patch is closed along the mesh by ``close_mm`` first (sub-threshold holes inside
    a patch come back, its outline does not move), and patches under
    ``min_area_mm2`` (midthickness) are dropped. Select on independent data: a
    localiser, never the contrast being tested.
    """
    from scipy.sparse.csgraph import connected_components

    from fastfuncstuff.surface.mesh import MeshTopology, close_patch, vertex_areas

    hits = responsive_vertices(path, surfaces, threshold=threshold, absolute=absolute)
    cs, cv, member_lists = [], [], []
    for h, (s, hit) in enumerate(zip(surfaces, hits, strict=True)):
        if not hit.any():
            continue
        mid = _midthickness(s)
        faces = np.asarray(s.faces)
        topo = MeshTopology.from_faces(faces, mid.shape[0])
        if close_mm > 0:
            hit = close_patch(mid, topo, hit, close_mm)
            if s.cortex is not None:
                hit &= np.asarray(s.cortex, bool)
        ids = np.flatnonzero(hit)
        sub = topo.adjacency()[ids][:, ids]
        n_comp, lab = connected_components(sub, directed=False)
        area = vertex_areas(mid, faces, mid.shape[0])
        for c in range(n_comp):
            verts = ids[lab == c]
            if area[verts].sum() < min_area_mm2:
                continue
            centroid = mid[verts].mean(0)
            cs.append(h)
            cv.append(int(verts[np.argmin(((mid[verts] - centroid) ** 2).sum(1))]))
            member_lists.append(h * _KEY + verts.astype(np.int64))
    if not cs:
        raise ValueError(
            f"{path}: no surface cluster of at least {min_area_mm2:g} mm2 passes {threshold:g}"
        )
    keys = np.unique(np.concatenate(member_lists))
    rows = np.repeat(np.arange(len(member_lists)), [m.size for m in member_lists])
    cols = np.searchsorted(keys, np.concatenate(member_lists))
    members = sp.csr_matrix(
        (np.ones(cols.size), (rows, cols)), shape=(len(member_lists), keys.size)
    )
    return ColumnSet(
        np.asarray(cs), np.asarray(cv), members, (keys // _KEY).astype(np.int64),
        (keys % _KEY).astype(np.int64), float("nan"), float("nan"),
    )  # fmt: skip


def regions_from_atlas(surfaces, paths: list[str], claim: str = "any") -> ColumnSet:
    """Regions from an atlas: each label of ``paths`` (``.annot``, ``.label``, GIfTI
    label map, or a volume atlas claimed along the column) is one region, read
    through depth like a column, keeping its name. The way to get the same regions
    in every subject -- a FreeSurfer parcellation, or a group atlas resampled to
    each subject -- so per-region values go straight into group tests.
    """
    from fastfuncstuff.processing.roi_depth import CLAIM_ANY, CLAIM_MID, read_depth_rois

    notes: list[str] = []
    rois = read_depth_rois(list(paths), surfaces, notes, CLAIM_ANY if claim == "any" else CLAIM_MID)
    cs, cv, names, member_lists = [], [], [], []
    for roi in rois:
        for h, member in sorted(roi.members.items()):
            verts = np.flatnonzero(member)
            if surfaces[h].cortex is not None:
                verts = verts[np.asarray(surfaces[h].cortex, bool)[verts]]
            if verts.size == 0:
                continue
            mid = _midthickness(surfaces[h])
            centroid = mid[verts].mean(0)
            cs.append(h)
            cv.append(int(verts[np.argmin(((mid[verts] - centroid) ** 2).sum(1))]))
            hemi = surfaces[h].name
            names.append(
                roi.name if roi.name.startswith((f"{hemi}.", f"{hemi}_")) else f"{hemi}.{roi.name}"
            )
            member_lists.append(h * _KEY + verts.astype(np.int64))
    if not cs:
        raise ValueError("no atlas region covers any cortex vertex of the surfaces given")
    keys = np.unique(np.concatenate(member_lists))
    rows = np.repeat(np.arange(len(member_lists)), [m.size for m in member_lists])
    cols = np.searchsorted(keys, np.concatenate(member_lists))
    members = sp.csr_matrix(
        (np.ones(cols.size), (rows, cols)), shape=(len(member_lists), keys.size)
    )
    return ColumnSet(
        np.asarray(cs), np.asarray(cv), members, (keys // _KEY).astype(np.int64),
        (keys % _KEY).astype(np.int64), float("nan"), float("nan"), names,
    )  # fmt: skip


@dataclass
class VertexDepthReads:
    """Vertex x depth reads of a data grid, per depth count ``K``.

    Rows are vertex-major, depth 1 (deepest) first, over the vertices of the
    :class:`ColumnSet` they were built for. ``weights[K] @ data[voxels]`` gives
    volume-weighted sums; ``volume[K]`` holds what they divide by.
    """

    ks: list[int]
    voxels: np.ndarray  # flat indices into the data grid that any read touches
    weights: dict[int, sp.csr_matrix]  # (n_v * K, voxels.size)
    volume: dict[int, np.ndarray]  # (n_v, K) read weight kept (mask / FOV drop reads)
    volume_all: dict[int, np.ndarray]  # (n_v, K) cell volume before any drop
    mixed: dict[int, np.ndarray] = field(default_factory=dict)  # (n_v, K, K) PSF numerators
    total: dict[int, np.ndarray] = field(default_factory=dict)  # (n_v, K)
    notes: list[str] = field(default_factory=list)


def vertex_depth_reads(
    surfaces,
    columns: ColumnSet,
    data_affine: np.ndarray,
    data_shape: tuple[int, ...],
    ks: list[int],
    opts: DepthOptions = DepthOptions(),
    *,
    mask: np.ndarray | None = None,
    psf: bool = True,
    device=None,
) -> VertexDepthReads:
    """Tile the ribbon under every column once and read it onto the data grid.

    ``mask`` (data grid) limits which voxels a read may use, as for ROI reads;
    the depth PSF is geometry and reads unmasked, as ``depth_sampling_multi``
    does.
    """
    shape3 = tuple(int(n) for n in data_shape[:3])
    aff = np.asarray(data_affine, np.float64)
    n_vox = int(np.prod(shape3))
    mask3 = None if mask is None else np.asarray(mask, bool).reshape(shape3)
    members = {}
    for h in np.unique(columns.vertex_surface):
        m = np.zeros(len(surfaces[int(h)].white), bool)
        m[columns.vertex_id[columns.vertex_surface == h]] = True
        members[int(h)] = m
    step = opts.dxyz or default_step(aff)
    ks = sorted({int(k) for k in ks})
    plans = plan_depth_multi(
        [_Roi(1, "columns", members)], surfaces, step, opts, ks,
        halo_mm=psf_halo_mm(opts.interp, default_step(aff)) if psf else 3.0,
        keep_all=psf, device=device,
    )  # fmt: skip
    cells = plans[0].cells
    key = cells.surface.astype(np.int64) * _KEY + cells.vertex
    vkeys = columns.vertex_surface * _KEY + columns.vertex_id
    slot = np.searchsorted(vkeys, key)
    slot_ok = (slot < vkeys.size) & (vkeys[np.minimum(slot, vkeys.size - 1)] == key)
    n_v = vkeys.size
    ops = {}
    for k in ks:
        cell, k_bin, share = cells.bin_shares(k)
        ok = slot_ok[cell]
        ops[k] = sp.csr_matrix(
            (cells.volume[cell[ok]] * share[ok], (slot[cell[ok]] * k + k_bin[ok], cell[ok])),
            shape=(n_v * k, len(cells)),
        )
    stacked = sp.vstack([ops[k] for k in ks]).tocsr()
    w_all, _ = _compose(stacked, cells.centre, aff, shape3, opts.interp, mask3, n_vox)
    voxels = np.flatnonzero(np.asarray((w_all != 0).sum(axis=0)).ravel() > 0)
    w_all = w_all[:, voxels].tocsr()
    out = VertexDepthReads(ks, voxels, {}, {}, {}, notes=[ln for ln in plans[0].notes])
    row0 = 0
    for k in ks:
        r1 = row0 + n_v * k
        out.weights[k] = w_all[row0:r1].tocsr()
        out.volume[k] = np.asarray(out.weights[k].sum(axis=1)).reshape(n_v, k)
        out.volume_all[k] = np.asarray(ops[k].sum(axis=1)).reshape(n_v, k)
        row0 = r1
    if psf:
        w_psf, _ = _compose(stacked, cells.centre, aff, shape3, opts.interp, None, n_vox)
        every = plans[0].all_cells
        assert every is not None
        inv = np.linalg.inv(aff)
        every_ijk = every.centre @ inv[:3, :3].T + inv[:3, 3]
        row0 = 0
        for k in ks:
            r1 = row0 + n_v * k
            mixed, total = mix_rows(
                every, k, w_psf[row0:r1].tocsr(), every_ijk, shape3, abs(np.linalg.det(aff[:3, :3]))
            )
            out.mixed[k] = mixed.reshape(n_v, k, k)
            out.total[k] = total.reshape(n_v, k)
            row0 = r1
    return out


def trim_to_coverage(
    columns: ColumnSet, reads: VertexDepthReads, min_coverage: float
) -> tuple[ColumnSet, np.ndarray]:
    """Drop from every column the vertices whose own depth reads are cut (FOV or mask
    under ``min_coverage`` at any bin, at any K). For named regions, which are large
    and fixed in advance: a slab that misses part of V1 should give V1's covered
    part, not lose V1. Returns the trimmed set and each column's kept share of
    ribbon volume."""
    keep = np.ones(columns.vertex_id.size, bool)
    for k in reads.ks:
        cov = np.divide(reads.volume[k], reads.volume_all[k], out=np.ones_like(reads.volume[k]),
                        where=reads.volume_all[k] > 0)  # fmt: skip
        keep &= cov.min(axis=1) >= min_coverage
    M = columns.members.tocsr()
    k0 = reads.ks[0]
    before = np.asarray(M @ reads.volume_all[k0].sum(1)).ravel()
    trimmed = sp.csr_matrix(M.multiply(keep[None, :]))
    trimmed.eliminate_zeros()
    after = np.asarray(trimmed @ reads.volume_all[k0].sum(1)).ravel()
    kept = np.divide(after, before, out=np.zeros_like(after), where=before > 0)
    return replace(columns, members=trimmed), kept


@dataclass
class ColumnCurves:
    """Every column's depth-resolved curves at one depth count."""

    K: int
    values: np.ndarray  # (C, K, *trailing) depth 1 deepest first
    coverage: np.ndarray  # (C, K) share of each bin's ribbon that was read
    psf: np.ndarray | None  # (C, K, K) rows normalised, depth 1 deepest first


def column_curves(
    reads: VertexDepthReads, columns: ColumnSet, data: np.ndarray, K: int
) -> ColumnCurves:
    """Column x depth means of ``data`` ``(reads.voxels.size, ...)`` -- any trailing
    shape (time x condition) -- with each column's coverage and PSF."""
    n_v = columns.vertex_id.size
    trailing = data.shape[1:]
    flat = np.asarray(data, np.float64).reshape(data.shape[0], -1)
    sums = (reads.weights[K] @ flat).reshape(n_v, K * flat.shape[1])
    M = columns.members
    col_sum = np.asarray(M @ sums).reshape(-1, K, flat.shape[1])
    vol = np.asarray(M @ reads.volume[K])
    vol_all = np.asarray(M @ reads.volume_all[K])
    values = np.divide(col_sum, vol[..., None], out=np.full_like(col_sum, np.nan),
                       where=vol[..., None] > 0)  # fmt: skip
    coverage = np.divide(vol, vol_all, out=np.zeros_like(vol), where=vol_all > 0)
    psf = None
    if K in reads.mixed:
        mixed = np.asarray(M @ reads.mixed[K].reshape(n_v, K * K)).reshape(-1, K, K)
        total = np.asarray(M @ reads.total[K])
        psf = np.divide(mixed, total[..., None], out=np.full_like(mixed, np.nan),
                        where=total[..., None] > 0)  # fmt: skip
        sm = psf.sum(axis=2, keepdims=True)
        psf = np.divide(psf, sm, out=np.full_like(psf, np.nan), where=sm > 0)
    return ColumnCurves(K, values.reshape(-1, K, *trailing), coverage, psf)


def paint_vertices(
    columns: ColumnSet, values: np.ndarray, valid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Each vertex's mean over the (valid) columns that contain it, and how many.

    ``values`` is ``(C, ...)``; returns ``(n_v, ...)`` (NaN where no valid column
    reaches) and ``(n_v,)`` counts. Overlapping columns are not independent, so
    the mean is a consensus of fits that share data, not a pooled estimate.
    """
    valid = np.ones(columns.n_columns, bool) if valid is None else np.asarray(valid, bool)
    M = columns.members[np.flatnonzero(valid)]
    vals = np.asarray(values, np.float64)[valid]
    flat = vals.reshape(vals.shape[0], -1)
    finite = np.isfinite(flat)
    sums = np.asarray(M.T @ np.where(finite, flat, 0.0))
    cnt = np.asarray(M.T @ finite.astype(np.float64))
    out = np.divide(sums, cnt, out=np.full_like(sums, np.nan), where=cnt > 0)
    n_cov = np.asarray(M.sum(axis=0)).ravel()
    return out.reshape(-1, *vals.shape[1:]), n_cov


def vertex_winner(
    columns: ColumnSet, best: np.ndarray, n_models: int, valid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Each vertex's most frequent winning model over the columns containing it (-1
    where none), and the share of those columns that agree -- the stability of
    the answer to where the column edges fall."""
    valid = np.ones(columns.n_columns, bool) if valid is None else np.asarray(valid, bool)
    valid = valid & (best >= 0)
    M = columns.members[np.flatnonzero(valid)]
    onehot = np.zeros((int(valid.sum()), n_models))
    onehot[np.arange(onehot.shape[0]), best[valid]] = 1.0
    counts = np.asarray(M.T @ onehot)
    tot = counts.sum(1)
    winner = np.where(tot > 0, counts.argmax(1), -1)
    agree = np.divide(counts.max(1), tot, out=np.full(tot.shape, np.nan), where=tot > 0)
    return winner, agree


def full_surface(columns: ColumnSet, vertex_values: np.ndarray, surface: int, n_vertices: int):
    """Values on the column vertices of one surface, scattered to the whole mesh (NaN
    elsewhere)."""
    sel = columns.vertex_surface == surface
    out = np.full((n_vertices, *vertex_values.shape[1:]), np.nan)
    out[columns.vertex_id[sel]] = vertex_values[sel]
    return out


def paint_ribbon(
    surfaces,
    columns: ColumnSet,
    vertex_values: np.ndarray,
    affine,
    shape,
    *,
    device=None,
    min_gm_fraction: float | None = None,
) -> np.ndarray:
    """Vertex values into the ribbon voxels of a grid, through each voxel's owning
    vertex (nearest midthickness point, as the depth sampling owns its cells):
    ``(X*Y*Z, ...)``, NaN off the columns. ``min_gm_fraction``: paint the
    fractional ribbon (``cortical_depth_volumes``), edge voxels by their GM part."""
    from fastfuncstuff.surface.volume_depth import cortical_depth_volumes

    dv = cortical_depth_volumes(list(surfaces), affine, tuple(shape[:3]), n_layers=3,
                                owners=True, device=device,
                                min_gm_fraction=min_gm_fraction)  # fmt: skip
    assert dv.owner_hemi is not None and dv.owner_vertex is not None
    key = dv.owner_hemi.astype(np.int64) * _KEY + dv.owner_vertex
    vkeys = columns.vertex_surface * _KEY + columns.vertex_id
    slot = np.searchsorted(vkeys, key)
    ok = (slot < vkeys.size) & (vkeys[np.minimum(slot, vkeys.size - 1)] == key)
    out = np.full((int(np.prod(shape[:3])), *vertex_values.shape[1:]), np.nan)
    out[dv.gm_idx[ok]] = vertex_values[slot[ok]]
    return out


__all__ = [
    "ColumnCurves",
    "ColumnSet",
    "VertexDepthReads",
    "column_curves",
    "full_surface",
    "make_columns",
    "paint_ribbon",
    "paint_vertices",
    "responsive_vertices",
    "trim_to_coverage",
    "vertex_depth_reads",
    "vertex_winner",
]
