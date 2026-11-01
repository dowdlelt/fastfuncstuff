"""ROI means along the cortical ribbon: one series per ROI per cortical depth.

The usual route to depth-resolved ROI signals is to upsample the data, build integer
layers on the fine grid (LayNii's LN2_LAYERS), intersect them with an ROI and average
the voxels. This does the same arithmetic straight from the white and pial meshes:

1. the ribbon under the ROIs is cut into small cells (:mod:`surface.slab`): about a
   voxel across along the surface (``-depth_dxyz``), much finer through the depth,
   each with its exact volume;
2. every cell gets its depth -- equidistant, or equivolume as a volume quantile
   within its cortical column, the definitions behind ``ffs_util_surf2layers`` --
   and its owning column (vertex). A cell across a bin boundary splits its volume;
3. a cell joins an ROI when its column is in the ROI. A surface ROI names the
   columns; a volume ROI claims the columns whose midthickness point lies in it, so
   every depth bin of an ROI averages the same patch of cortex;
4. each cell reads the data at its centre (cubic by default, ``-depth_interp``) and
   counts by its volume, so a bin's value is the volume-weighted mean over its slab
   of the ribbon -- what averaging an upsampled image within LayNii layers gives.

Interpolation is linear in the data, so the whole thing is one sparse
``(n_rois x n_depths, n_voxels)`` weight matrix: the ordinary weighted-ROI collapse,
and every output an ROI fit writes then comes out per ROI per depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from fastfuncstuff.processing.roi_collapse import RoiSet, _gifti_label_table, _stem
from fastfuncstuff.surface.slab import RibbonCells


@dataclass(frozen=True)
class DepthOptions:
    n_depths: int = 1
    #: ``equidistant`` or ``equivolume``.
    method: str = "equidistant"
    #: Read spacing along the surface, mm (half that through the depth); None is the
    #: data's voxel size.
    dxyz: float | None = None
    #: Rounds of mesh smoothing of each column's volume profile (equivolume).
    smooth: int = 1
    #: How each read interpolates the data grid (:data:`READ_KERNELS`).
    interp: str = "cubic"
    #: How a volume ROI claims a column: ``mid`` (its midthickness point) or ``any``
    #: (the ROI covers it at any depth).
    claim: str = "mid"

    @property
    def claim_fracs(self) -> tuple[float, ...]:
        return CLAIM_ANY if self.claim == "any" else CLAIM_MID


@dataclass
class _Roi:
    rid: int
    name: str
    #: index into the surfaces -> (V,) bool vertex membership.
    members: dict[int, np.ndarray]


def _read_label(path: str) -> tuple[np.ndarray, np.ndarray]:
    """``(vertex ids, white tkregister coords)`` of an ASCII FreeSurfer ``.label``."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[1].split()[0])
    rows = np.array([ln.split()[:4] for ln in lines[2 : 2 + n]], dtype=np.float64).reshape(-1, 4)
    return rows[:, 0].astype(np.int64), rows[:, 1:4]


def _candidates(path: str, surfaces, n_vertices: int | None) -> list[int]:
    """Surfaces a surface ROI file could belong to, by its lh/rh name and vertex count."""
    from fastfuncstuff.io.freesurfer import infer_label

    hemi = infer_label(Path(path).name)[0]
    out = [
        i
        for i, s in enumerate(surfaces)
        if (hemi is None or s.name == hemi) and (n_vertices is None or len(s.white) == n_vertices)
    ]
    if not out:
        have = ", ".join(f"{s.name} ({len(s.white):,} vertices)" for s in surfaces)
        raise ValueError(f"{path}: matches none of the surfaces given ({have})")
    return out


def _label_surface(path: str, surfaces, notes: list[str]) -> int:
    """The surface a ``.label`` was drawn on, checked by its own white coordinates."""
    ids, coords = _read_label(path)
    if ids.size == 0:
        raise ValueError(f"{path}: empty label")
    fits = [i for i in _candidates(path, surfaces, None) if ids.max() < len(surfaces[i].white)]
    if not fits:
        raise ValueError(f"{path}: vertex ids reach {ids.max()}, beyond every surface given")
    scored = []
    for i in fits:
        s = surfaces[i]
        if s.tkr_to_scanner is None:
            continue
        to_tkr = np.linalg.inv(np.asarray(s.tkr_to_scanner, np.float64))
        white = np.asarray(s.white, np.float64)[ids] @ to_tkr[:3, :3].T + to_tkr[:3, 3]
        scored.append((float(np.median(np.linalg.norm(white - coords, axis=1))), i))
    if scored:
        err, best = min(scored)
        if err > 2.0:
            raise ValueError(
                f"{path}: its coordinates sit {err:.1f} mm (median) from the white surface "
                "of every hemisphere given -- drawn on another subject or surface"
            )
        if err > 0.01:
            notes.append(
                f"{Path(path).name}: coordinates {err:.2f} mm (median) off this white "
                "surface -- drawn on an edited copy?"
            )
        return best
    if len(fits) > 1:
        raise ValueError(f"{path}: name its hemisphere (lh./rh.) -- both surfaces fit")
    notes.append(f"{Path(path).name}: no tkregister frame to check against, matched by name")
    return fits[0]


def _surface_rois(path: str, surfaces, notes: list[str]) -> list[_Roi]:
    """ROIs from a surface file: ``.label``, ``.annot`` or a GIfTI label/shape map."""
    from fastfuncstuff.io.freesurfer import NOT_A_REGION, read_annotation

    low = path.lower()
    if low.endswith(".label"):
        i = _label_surface(path, surfaces, notes)
        ids, _ = _read_label(path)
        member = np.zeros(len(surfaces[i].white), bool)
        member[ids] = True
        return [_Roi(1, _stem(path), {i: member})]

    if low.endswith(".annot"):
        import nibabel.freesurfer as nfs

        n = len(nfs.read_annot(path)[0])
        cands = _candidates(path, surfaces, n)
        if len(cands) > 1:
            raise ValueError(f"{path}: name its hemisphere (lh./rh.) -- both surfaces fit")
        annot = read_annotation(path, n_vertices=n)
        return [
            _Roi(k + 1, name, {cands[0]: annot.labels == k})
            for k, name in enumerate(annot.names)
            if name not in NOT_A_REGION and (annot.labels == k).any()
        ]

    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.io.gifti import mesh_fingerprint, surface_meta

    img = load_nifti(path)
    values = np.asanyarray(img.dataobj).reshape(-1)
    if not np.array_equal(values, np.round(values)):
        raise ValueError(f"{path}: a surface ROI map must hold integer labels")
    mine = (surface_meta(img.header) or {}).get("mesh_fingerprint")
    if mine:
        cands = [
            i for i, s in enumerate(surfaces) if mesh_fingerprint(s.faces, len(s.white)) == mine
        ]
        if not cands:
            raise ValueError(f"{path}: drawn on a mesh that is none of the surfaces given")
    else:
        cands = _candidates(path, surfaces, values.size)
        if len(cands) > 1:
            raise ValueError(f"{path}: name its hemisphere (lh./rh.) -- both surfaces fit")
        notes.append(f"{Path(path).name}: no mesh fingerprint, matched by name and vertex count")
    table = _gifti_label_table(path)
    labels = [int(v) for v in np.unique(values) if v != 0]
    one = len(labels) == 1 and labels[0] not in table
    return [
        _Roi(
            lab,
            _stem(path) if one else (table[lab].name if lab in table else f"{_stem(path)}_{lab}"),
            {cands[0]: values == lab},
        )
        for lab in labels
    ]


#: Depth fractions (white 0 to pial 1) a volume ROI is read at to claim a column.
#: One midthickness read is the depth-sampling default -- a column is in an ROI if
#: the ROI is there at mid-depth. The spread is for ``ffs_util_surf2layers``, where
#: the ROI is usually a thresholded statistic that only reaches part of the ribbon.
CLAIM_MID = (0.5,)
CLAIM_ANY = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)


def _volume_rois(
    path: str, surfaces, notes: list[str], claim: tuple[float, ...] = CLAIM_MID
) -> list[_Roi]:
    """A volume ROI, mapped to the columns whose ribbon lies inside it.

    Read on the ROI file's own grid (any resolution, in register with the surfaces),
    nearest voxel: a 3-D integer map gives one ROI per label, a 4-D stack one per
    volume (any value above zero is in).

    ``claim`` are the depth fractions (white 0 to pial 1) read along each column; the
    column joins if the ROI is there at *any* of them. With one fraction this is a
    point test -- the usual midthickness rule -- and with several it is the rule for
    an ROI that need not span the depth, such as a thresholded statistic that peaks
    near the pial surface and falls below threshold at the white matter.
    """
    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.io.labels import label_table

    img = load_nifti(path)
    arr = np.asanyarray(img.dataobj)
    if arr.ndim == 4 and arr.shape[3] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (3, 4):
        raise ValueError(f"{path}: ROI files must be 3-D or 4-D, got {arr.ndim}-D")
    inv = np.linalg.inv(np.asarray(img.affine, np.float64))
    fracs = np.asarray(claim, np.float64)
    sampled = []  # per surface: (V, len(claim)) or (V, len(claim), n) values on the column
    for s in surfaces:
        w = np.asarray(s.white, np.float64)
        p = np.asarray(s.pial, np.float64)
        vals = np.zeros((len(w), fracs.size, *arr.shape[3:]), arr.dtype)
        for d, frac in enumerate(fracs):
            pts = w + frac * (p - w)
            ijk = np.rint(pts @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
            inside = np.all((ijk >= 0) & (ijk < np.asarray(arr.shape[:3])), axis=1)
            vals[inside, d] = arr[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]
        sampled.append(vals)
    where = (
        "its midthickness point"
        if fracs.size == 1
        else f"any of {fracs.size} points along it (depth {fracs.min():g}-{fracs.max():g})"
    )
    notes.append(f"{Path(path).name}: volume ROI, each column joins by {where}")
    if arr.ndim == 4:
        return [
            _Roi(j + 1, f"{_stem(path)}_{j + 1}" if arr.shape[3] > 1 else _stem(path),
                 {i: (v[:, :, j] > 0).any(1) for i, v in enumerate(sampled)})
            for j in range(arr.shape[3])
        ]  # fmt: skip
    if not np.array_equal(arr, np.round(arr)):
        raise ValueError(f"{path}: a 3-D ROI file must hold integer labels; give weights as 4-D")
    table = label_table(path, img)
    labels = [int(v) for v in np.unique(arr) if v != 0]
    one = len(labels) == 1 and labels[0] not in table
    return [
        _Roi(
            lab,
            _stem(path) if one else (table[lab].name if lab in table else f"{_stem(path)}_{lab}"),
            {i: (v == lab).any(1) for i, v in enumerate(sampled)},
        )
        for lab in labels
    ]


def _read_rois(
    paths: list[str], surfaces, notes: list[str], claim: tuple[float, ...] = CLAIM_MID
) -> list[_Roi]:
    from fastfuncstuff.io.gifti import is_gifti

    rois: list[_Roi] = []
    for f in paths:
        plain = f.split("[")[0]
        if plain.lower().endswith((".label", ".annot")) or is_gifti(plain):
            rois += _surface_rois(plain, surfaces, notes)
        else:
            rois += _volume_rois(plain, surfaces, notes, claim)
    if len(paths) > 1:  # labels repeat across files; the table needs distinct ids
        for k, r in enumerate(rois):
            r.rid = k + 1
    return rois


#: Interpolants a ribbon read can use; ``linear`` is trilinear.
READ_KERNELS = ("linear", "cubic", "quintic", "heptic", "wsinc5")


def _read_taps(
    points: np.ndarray, affine: np.ndarray | None, shape, kernel: str, mask: np.ndarray | None
):
    """``(flat voxel indices, weights)``, each ``(N, taps)``, of reading a grid at
    ``points`` with a separable ``kernel`` (``affine`` None: ``points`` are already
    fractional voxel indices, e.g. where a transform chain lands them).

    Taps off the grid or outside ``mask`` are dropped and the rest renormalised, so a
    read never uses a voxel the analysis excludes; a read whose kept taps carry under
    half its weight is dropped (all-zero weights).

    Holds ``taps**3`` entries per read in several float64 temporaries (17 KB a read for
    wsinc5): call it in batches, as :func:`_compose` does, never on a whole ribbon.
    """
    import torch

    if affine is None:
        ijk = np.asarray(points, np.float64)
    else:
        inv = np.linalg.inv(np.asarray(affine, np.float64))
        ijk = points @ inv[:3, :3].T + inv[:3, 3]
    base = np.floor(ijk).astype(np.int64)
    frac = ijk - base
    if kernel == "linear":
        offsets = np.array([0, 1])
        axis_w = [np.stack([1.0 - frac[:, a], frac[:, a]], 1) for a in range(3)]
    else:
        from fastfuncstuff.processing.interp import _KERNELS, _kernel_half_width

        half = _kernel_half_width(kernel)
        offsets = np.arange(-(half - 1), half + 1)
        fn = _KERNELS[kernel][0]
        axis_w = [fn(torch.as_tensor(frac[:, a])).numpy() for a in range(3)]
    n = offsets.size
    dims = np.asarray(shape[:3])
    ax = [base[:, a, None] + offsets[None] for a in range(3)]  # (N, n) per axis
    ok_ax = [(ax[a] >= 0) & (ax[a] < dims[a]) for a in range(3)]
    flat = (
        np.clip(ax[0], 0, dims[0] - 1)[:, :, None, None] * (dims[1] * dims[2])
        + np.clip(ax[1], 0, dims[1] - 1)[:, None, :, None] * dims[2]
        + np.clip(ax[2], 0, dims[2] - 1)[:, None, None, :]
    ).reshape(len(points), n**3)
    w = axis_w[0][:, :, None, None] * axis_w[1][:, None, :, None] * axis_w[2][:, None, None, :]
    ok = ok_ax[0][:, :, None, None] & ok_ax[1][:, None, :, None] & ok_ax[2][:, None, None, :]
    w = np.where(ok, w, 0.0).reshape(len(points), n**3)
    if mask is not None:
        w = np.where(mask.reshape(-1)[flat], w, 0.0)
    total = w.sum(axis=1, keepdims=True)
    return flat, np.divide(w, total, out=np.zeros_like(w), where=total >= 0.5)


def _compose(operator: sp.csr_matrix, cells_xyz, affine, shape, kernel, mask, n_vox):
    """``operator @ taps``: rows over grid voxels, built a batch of reads at a time.
    Also returns which reads kept no tap."""
    n_taps = (2 if kernel == "linear" else 2 * _half(kernel)) ** 3
    batch = max(1024, (1 << 22) // n_taps)
    acc, parts, held = None, [], 0
    dropped = np.zeros(cells_xyz.shape[0], bool)
    op_csc = operator.tocsc()
    for s in range(0, cells_xyz.shape[0], batch):
        e = min(cells_xyz.shape[0], s + batch)
        flat, w = _read_taps(cells_xyz[s:e], affine, shape, kernel, mask)
        dropped[s:e] = ~(w != 0).any(axis=1)
        taps = sp.csr_matrix(
            (w.reshape(-1), flat.reshape(-1), np.arange(0, (e - s) * n_taps + 1, n_taps)),
            shape=(e - s, n_vox),
        )
        parts.append((op_csc[:, s:e] @ taps).tocoo())
        held += parts[-1].nnz
        if held > (1 << 25):
            acc = _merge(acc, parts, operator.shape[0], n_vox)
            parts, held = [], 0
    return _merge(acc, parts, operator.shape[0], n_vox), dropped


def _merge(acc, parts, n_rows, n_vox):
    if not parts:
        return acc if acc is not None else sp.csr_matrix((n_rows, n_vox))
    new = sp.coo_matrix(
        (np.concatenate([p.data for p in parts]),
         (np.concatenate([p.row for p in parts]), np.concatenate([p.col for p in parts]))),
        shape=(n_rows, n_vox),
    ).tocsr()  # fmt: skip
    return new if acc is None else acc + new


def _half(kernel: str) -> int:
    from fastfuncstuff.processing.interp import _kernel_half_width

    return _kernel_half_width(kernel)


def default_step(data_affine: np.ndarray) -> float:
    """The data's voxel size (mm, geometric mean): reads one voxel apart along the
    surface and half a voxel through the depth."""
    return float(abs(np.linalg.det(np.asarray(data_affine, np.float64)[:3, :3])) ** (1 / 3))


@dataclass
class DepthPlan:
    """ROI x depth reads before any data grid: where to read and how to average.

    ``operator`` is ``(len(rois) * n_depths, len(cells))`` -- each cell's volume,
    split over the depth bins it spans -- rows ROI-major, depth 1 (deepest) first.
    """

    rois: list[_Roi]
    cells: RibbonCells
    operator: sp.csr_matrix
    hemi_of: list[str]
    n_depths: int
    step: float
    notes: list[str]
    #: The whole tiling before cells were kept for their owners (``keep_all``): the
    #: ribbon around the ROIs as well, which shares their voxels.
    all_cells: RibbonCells | None = None
    #: Where each of ``cells`` sits in ``all_cells``.
    owned_index: np.ndarray | None = None

    def columns_per_row(self) -> np.ndarray:
        """Distinct cortical columns each operator row reads."""
        key = self.cells.surface.astype(np.int64) * (1 << 40) + self.cells.vertex
        coo = self.operator.tocoo()
        on = coo.data > 0
        pairs = np.unique(np.stack([coo.row[on], key[coo.col[on]]], 1), axis=0)
        return np.bincount(pairs[:, 0], minlength=self.operator.shape[0])


def _check_options(opts: DepthOptions) -> None:
    if opts.n_depths < 1:
        raise ValueError("-depths must be at least 1")
    if opts.smooth < 0:
        raise ValueError("-depth_smooth must be 0 or more")
    if opts.method not in ("equidistant", "equivolume"):
        raise ValueError(f"depth method must be equidistant or equivolume, not {opts.method!r}")
    if opts.interp not in READ_KERNELS:
        raise ValueError(f"read interpolation must be one of {READ_KERNELS}, not {opts.interp!r}")
    if opts.claim not in ("mid", "any"):
        raise ValueError(f"-roi_claim must be mid or any, not {opts.claim!r}")


def read_depth_rois(
    files: list[str], surfaces, notes: list[str], claim: tuple[float, ...] = CLAIM_MID
) -> list[_Roi]:
    """The ROIs of ``files`` that cover any vertex of ``surfaces``.

    One ROI per label, per 4-D volume or per ``.annot`` region, each holding the
    cortical columns it covers (``members``: surface index -> vertex mask). The one
    reading of ``-atlas`` / ``-rois`` against a ribbon, shared by the depth sampling
    and by ``ffs_util_surf2layers``'s ``-roi``; ``claim`` is :func:`_volume_rois`'s.
    """
    rois = [
        r
        for r in _read_rois(files, surfaces, notes, claim)
        if any(m.any() for m in r.members.values())
    ]
    if not rois:
        raise ValueError("no ROI covers any vertex of the surfaces given")
    return rois


def _faces_of(rois: list[_Roi], surfaces, halo_mm: float = 3.0) -> dict[int, np.ndarray]:
    """Faces to tile per surface so every cell the ROIs' columns own is among them
    (``halo_mm`` beyond a column's length)."""
    from fastfuncstuff.surface.slab import patch_faces

    members: dict[int, np.ndarray] = {}
    for r in rois:
        for i, m in r.members.items():
            members[i] = members.get(i, np.zeros(m.size, bool)) | m
    out = {}
    for i, m in sorted(members.items()):
        if m.any():
            s = surfaces[i]
            column = np.linalg.norm(np.asarray(s.pial) - np.asarray(s.white), axis=1)[m].max()
            out[i] = patch_faces(s, np.flatnonzero(m), float(column) + halo_mm)
    return out


def tile_costs(
    rois: list[_Roi], surfaces, step: float, n_depths: int, through: float
) -> np.ndarray:
    """Cells it takes to tile each ROI on its own, from the cut rule (nothing is built)."""
    from fastfuncstuff.surface.slab import estimate_cells

    cost = np.zeros(len(rois), np.int64)
    for k, r in enumerate(rois):
        for i, f in _faces_of([r], surfaces).items():
            cost[k] += estimate_cells(surfaces[i], f, step, n_depths, through=through)
    return cost


def pass_groups(costs: np.ndarray, max_reads: int | None) -> list[list[int]]:
    """Split ROIs into passes of at most ``max_reads`` cells (first fit, largest first).

    A pass tiles, reads and discards its own ribbon, so memory follows the cap and not
    the number of ROIs. An ROI whose own tiling exceeds the cap gets a pass of its own
    (the caller decides whether that is acceptable).
    """
    order = list(np.argsort(-costs, kind="stable"))
    if not max_reads or int(costs.sum()) <= max_reads:
        return [list(range(len(costs)))]
    groups: list[list[int]] = []
    load: list[int] = []
    for k in order:
        for g, used in enumerate(load):
            if used + int(costs[k]) <= max_reads:
                groups[g].append(int(k))
                load[g] += int(costs[k])
                break
        else:
            groups.append([int(k)])
            load.append(int(costs[k]))
    return [sorted(g) for g in groups]


def plan_depth_multi(
    rois: list[_Roi],
    surfaces,
    step: float,
    opts: DepthOptions = DepthOptions(),
    depths: list[int] | None = None,
    *,
    through: float = 0.5,
    halo_mm: float = 3.0,
    keep_all: bool = False,
    device=None,
) -> list[DepthPlan]:
    """One :class:`DepthPlan` per depth count in ``depths``, all from ONE tiling.

    A cell's depth metric does not depend on how many bins it is cut into, so the ribbon
    is tiled once, finely enough for the largest count (at least two sheets per bin),
    and each count's operator is cut from the same cells -- the reads are shared, which
    is what lets ``ffs_nwarp`` read the data once for every K.
    """
    from fastfuncstuff.surface.slab import ribbon_cells

    ks = sorted({int(k) for k in (depths or [opts.n_depths])})
    for k in ks:
        _check_options(DepthOptions(k, opts.method, opts.dxyz, opts.smooth, opts.interp))
    k_max = ks[-1]
    faces_of = _faces_of(rois, surfaces, halo_mm)
    log: list[str] = []
    cells = ribbon_cells(
        surfaces, faces_of, step, k_max, opts.method,
        smooth_rounds=opts.smooth, through=through, device=device, log=log.append,
    )  # fmt: skip
    members: dict[int, np.ndarray] = {}
    for r in rois:
        for i, m in r.members.items():
            members[i] = members.get(i, np.zeros(m.size, bool)) | m
    owned = np.zeros(len(cells), bool)
    for i, m in members.items():
        here = cells.surface == i
        owned[here] = m[cells.vertex[here]]
    inverted = cells.inverted_mm3
    everything = cells if keep_all else None
    cells = cells.subset(owned)
    cells.inverted_mm3 = inverted
    if not len(cells):
        raise ValueError("no ribbon under the ROIs: are the surfaces' cortex labels right?")

    inside_by_roi = []
    hemi_of = []
    for roi in rois:
        inside = np.zeros(len(cells), bool)
        for i, member in roi.members.items():
            here = cells.surface == i
            inside[here] = member[cells.vertex[here]]
        inside_by_roi.append(inside)
        hemi_of.append("+".join(sorted({surfaces[int(i)].name for i in np.unique(cells.surface[inside])})) or "-")  # fmt: skip
    plans = []
    for k_n in ks:
        cell, k_bin, share = cells.bin_shares(k_n)
        rows_i, cols_i, vals = [], [], []
        for r_idx, inside in enumerate(inside_by_roi):
            pick = np.flatnonzero(inside[cell])
            rows_i.append(r_idx * k_n + k_bin[pick])
            cols_i.append(cell[pick])
            vals.append(cells.volume[cell[pick]] * share[pick])
        operator = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(cols_i))),
            shape=(len(rois) * k_n, len(cells)),
        )
        plans.append(
            DepthPlan(
                rois, cells, operator, hemi_of, k_n, step, [ln.strip() for ln in log], everything,
                np.flatnonzero(owned) if keep_all else None,
            )
        )  # fmt: skip
    return plans


def plan_depth_rois(
    rois: list[_Roi],
    surfaces,
    step: float,
    opts: DepthOptions = DepthOptions(),
    *,
    through: float = 0.5,
    halo_mm: float = 3.0,
    keep_all: bool = False,
    device=None,
) -> DepthPlan:
    """Tile the ribbon under ``rois`` and cut it into ROI x depth operator rows."""
    return plan_depth_multi(
        rois, surfaces, step, opts, [opts.n_depths], through=through, halo_mm=halo_mm,
        keep_all=keep_all, device=device,
    )[0]  # fmt: skip


def build_depth_roi_sets(
    paths: str | list[str],
    surfaces,
    data_affine: np.ndarray,
    data_shape: tuple[int, ...],
    mask: np.ndarray | None = None,
    opts: DepthOptions = DepthOptions(),
    depths: list[int] | None = None,
    *,
    device=None,
) -> list[RoiSet]:
    """One ROI x depth :class:`RoiSet` per depth count in ``depths``, from one tiling.

    The ribbon is tiled once and read onto the data grid once: every count's operator
    is composed in a single pass over the taps. Each set's rows are ROI-major, depth 1
    (deepest, by white matter) to ``n_depths`` (by pial). ``surfaces`` are
    :class:`surface.volume_depth.RibbonSurfaces` in scanner RAS -- the data must be in
    register with the anatomy they were made from. ``mask`` (data grid) limits which
    voxels are ever read.
    """
    files = [paths] if isinstance(paths, str) else list(paths)
    _check_options(opts)
    shape3 = tuple(int(n) for n in data_shape[:3])
    aff = np.asarray(data_affine, np.float64)
    mask3 = None if mask is None else np.asarray(mask).astype(bool).reshape(shape3)
    notes: list[str] = []
    rois = read_depth_rois(files, surfaces, notes, opts.claim_fracs)
    step = opts.dxyz or default_step(aff)
    plans = plan_depth_multi(rois, surfaces, step, opts, depths, device=device)
    cells = plans[0].cells
    n_vox = int(np.prod(shape3))
    stacked = sp.vstack([p.operator for p in plans]).tocsr()
    w_all, dropped = _compose(stacked, cells.centre, aff, shape3, opts.interp, mask3, n_vox)
    inv = np.linalg.inv(aff)
    near = np.rint(cells.centre @ inv[:3, :3].T + inv[:3, 3])
    off_grid = ~np.all((near >= 0) & (near < np.asarray(shape3)), axis=1)
    out, row0 = [], 0
    for plan in plans:
        r1 = row0 + plan.operator.shape[0]
        out.append(
            _roi_set_from_plan(
                plan, w_all[row0:r1].tocsr(), dropped, off_grid, files, notes, opts, shape3,
                mask3 is not None, n_vox,
            )
        )  # fmt: skip
        row0 = r1
    return out


def build_depth_roi_set(
    paths: str | list[str],
    surfaces,
    data_affine: np.ndarray,
    data_shape: tuple[int, ...],
    mask: np.ndarray | None = None,
    opts: DepthOptions = DepthOptions(),
    *,
    device=None,
) -> RoiSet:
    """ROI x depth rows for data on ``(data_affine, data_shape)``, read along the ribbon;
    see :func:`build_depth_roi_sets` (this is its one-count form)."""
    return build_depth_roi_sets(
        paths, surfaces, data_affine, data_shape, mask, opts, [opts.n_depths], device=device
    )[0]


def _roi_set_from_plan(
    plan: DepthPlan, w, dropped, off_grid, files, notes0, opts, shape3, masked, n_vox
) -> RoiSet:
    """One count's ROI set from its plan and its rows of the composed reads."""
    rois, cells, operator, hemi_of = plan.rois, plan.cells, plan.operator, plan.hemi_of
    n_depths, step, log = plan.n_depths, plan.step, plan.notes
    notes = list(notes0)
    n_rows = operator.shape[0]
    read_op = operator.multiply(~dropped[None]).tocsr()
    w.eliminate_zeros()
    sizes = np.asarray(read_op.sum(axis=1)).ravel()
    alive = np.flatnonzero((sizes > 0) & (np.diff(w.indptr) > 0))
    touched = np.flatnonzero(np.asarray((w[alive] != 0).sum(axis=0)).ravel() > 0)
    w = w[alive][:, touched].tocsr()
    voxel_rows = np.zeros(n_vox, bool)
    voxel_rows[touched] = True
    # Columns per row: owners of the cells the row reads.
    key = cells.surface.astype(np.int64) * (1 << 40) + cells.vertex
    coo = read_op.tocoo()
    pairs = np.unique(np.stack([coo.row[coo.data > 0], key[coo.col[coo.data > 0]]], 1), axis=0)
    per_row = np.bincount(pairs[:, 0], minlength=n_rows)

    ids, names, index = [], [], {"hemi": [], "depth": [], "depth_frac": [], "n_columns": []}
    for row in alive:
        r_idx, k = divmod(int(row), n_depths)
        ids.append(rois[r_idx].rid)
        names.append(rois[r_idx].name)
        index["hemi"].append(hemi_of[r_idx])
        index["depth"].append(str(k + 1))
        index["depth_frac"].append(f"{(k + 0.5) / n_depths:.4g}")
        index["n_columns"].append(str(int(per_row[row])))
    empty = [f"{rois[int(r) // n_depths].name}@{int(r) % n_depths + 1}" for r in
             np.setdiff1d(np.arange(n_rows), alive)]  # fmt: skip

    total_mm3 = float(cells.volume.sum())
    lost_fov = float(cells.volume[off_grid].sum())
    lost_mask = float(cells.volume[dropped & ~off_grid].sum())
    names_used = "+".join(sorted({h for h in "+".join(hemi_of).split("+") if h != "-"})) or "-"
    notes += log
    notes += [
        f"depth sampling: {len(rois)} ROI(s) x {n_depths} {opts.method} depth(s) along the "
        f"{names_used} ribbon; depth 1 = deepest (white matter side), depth_frac = bin "
        f"centre as a fraction of {'thickness' if opts.method == 'equidistant' else 'column volume'}",
        f"ROI ribbon {total_mm3:,.0f} mm3 in {len(cells):,} cells (reads {step:.3g} mm apart "
        f"along the surface, finer through the depth), {opts.interp} interpolation",
        "the data must be in register with the anatomy the surfaces were made from",
    ]
    if cells.inverted_mm3 > 0.001 * total_mm3:
        notes.append(
            f"{cells.inverted_mm3:,.1f} mm3 where white and pial cross (inside-out cells) left out"
        )
    if lost_fov > 0.5 * total_mm3:
        notes.append(
            f"WARNING: {100 * lost_fov / total_mm3:.0f}% of the ROI ribbon lies outside "
            "the data -- partial coverage, or the surfaces are not in register"
        )
    elif lost_fov > 0:
        notes.append(f"{100 * lost_fov / total_mm3:.1f}% of the ROI ribbon is outside the data")
    if masked:
        notes.append(
            "ribbon reads limited to the analysis mask (-mask, or the missing-data guard's)"
            + (
                f": {100 * lost_mask / total_mm3:.1f}% of the ROI ribbon dropped"
                if lost_mask
                else ""
            )
        )
    if empty:
        notes.append(
            f"{len(empty)} ROI x depth row(s) hold no ribbon read, dropped: "
            + ", ".join(empty[:8])
            + (" ..." if len(empty) > 8 else "")
        )
    if not len(alive):
        raise ValueError("no ROI ribbon point reads any data voxel")
    return RoiSet(
        spec=w, ids=ids, names=names, sizes=sizes[alive], rows=voxel_rows.reshape(shape3),
        sources=[Path(f.split("[")[0]).name for f in files], notes=notes,
        index=index, size_name="volume_mm3", long_rows=True,
    )  # fmt: skip


def _mix_matrices(
    plan: DepthPlan,
    w: sp.csr_matrix,
    every_ijk: np.ndarray,
    shape3: tuple[int, int, int],
    voxel_mm3,
    normalise: bool,
) -> list[tuple[str, str, np.ndarray]]:
    """The core of the depth sampling matrices, for any grid the reads land on.

    ``w`` is the ROI rows' reads composed onto the grid's voxels, ``every_ijk`` where
    each cell of ``plan.all_cells`` lies on that grid (voxel indices), ``voxel_mm3``
    the volume of a grid voxel in the anatomy -- a number, or a callable of the flat
    indices of the voxels the reads touch (a distorted EPI voxel is not its nominal size).
    """
    k_n = plan.n_depths
    mixed, total = mix_rows(plan.all_cells, k_n, w, every_ijk, shape3, voxel_mm3)
    out = []
    for r_idx, roi in enumerate(plan.rois):
        rows = slice(r_idx * k_n, (r_idx + 1) * k_n)
        m = np.divide(mixed[rows], total[rows, None], out=np.full((k_n, k_n), np.nan),
                      where=total[rows, None] > 0)  # fmt: skip
        if normalise:
            sm = m.sum(axis=1, keepdims=True)
            m = np.divide(m, sm, out=np.full_like(m, np.nan), where=sm > 0)
        out.append((roi.name, plan.hemi_of[r_idx], m))
    return out


def mix_rows(
    every, k_n: int, w: sp.csr_matrix, every_ijk: np.ndarray, shape3, voxel_mm3
) -> tuple[np.ndarray, np.ndarray]:
    """``(mixed, total)`` per read row: ``mixed[r, j]`` is how much true depth ``j``
    row ``r``'s reads pick up (unnormalised), ``total[r]`` its read weight -- so
    ``mixed / total`` is the row's depth PSF. Linear in the rows, so the PSF of a
    union of rows is the ratio of their sums.

    ``every`` are all the cells that can sit in a read voxel (``DepthPlan.all_cells``),
    ``every_ijk`` where they lie on the read grid.
    """
    touched = np.flatnonzero(np.asarray((w != 0).sum(axis=0)).ravel() > 0)
    assert every is not None
    ijk = np.rint(every_ijk).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < np.asarray(shape3)), axis=1) & np.isfinite(every_ijk).all(1)
    flat = np.full(len(every), -1, np.int64)
    flat[inside] = np.ravel_multi_index(tuple(ijk[inside].T), shape3)
    slot = np.searchsorted(touched, flat)
    hit = (
        (flat >= 0) & (slot < touched.size) & (touched[np.minimum(slot, touched.size - 1)] == flat)
    )
    cell, k_bin, share = every.bin_shares(k_n)
    keep = hit[cell]
    v = np.zeros((touched.size, k_n))
    np.add.at(v, (slot[cell[keep]], k_bin[keep]), every.volume[cell[keep]] * share[keep])
    vol = voxel_mm3(touched) if callable(voxel_mm3) else np.full(touched.size, float(voxel_mm3))
    v /= np.where(vol > 0, vol, np.inf)[:, None]

    total = np.asarray(w.sum(axis=1)).ravel()
    mixed = np.asarray(w[:, touched] @ v)  # (rows, K)
    return mixed, total


def psf_halo_mm(interp: str, voxel_mm: float) -> float:
    """Halo that holds every cell inside any voxel the reads touch: the kernel's reach
    beyond the ROI's own voxels, plus a voxel, plus the usual pial-slide margin."""
    reach = 2 if interp == "linear" else (_half(interp) + 1)
    return 3.0 + (reach + 1) * voxel_mm


def depth_sampling_multi(
    paths: str | list[str],
    surfaces,
    data_affine: np.ndarray,
    data_shape: tuple[int, ...],
    opts: DepthOptions = DepthOptions(),
    depths: list[int] | None = None,
    *,
    normalise: bool = True,
    device=None,
) -> dict[int, list[tuple[int, str, str, np.ndarray]]]:
    """:func:`depth_sampling_matrices` for several depth counts from one tiling:
    ``{K: [(roi id, name, hemi, M), ...]}``, ROIs in the order the set has them."""
    files = [paths] if isinstance(paths, str) else list(paths)
    _check_options(opts)
    shape3 = tuple(int(n) for n in data_shape[:3])
    aff = np.asarray(data_affine, np.float64)
    rois = read_depth_rois(files, surfaces, [], opts.claim_fracs)
    step = opts.dxyz or default_step(aff)
    plans = plan_depth_multi(
        rois, surfaces, step, opts, depths, halo_mm=psf_halo_mm(opts.interp, default_step(aff)),
        keep_all=True, device=device,
    )  # fmt: skip
    n_vox = int(np.prod(shape3))
    stacked = sp.vstack([p.operator for p in plans]).tocsr()
    w_all, _ = _compose(stacked, plans[0].cells.centre, aff, shape3, opts.interp, None, n_vox)
    assert plans[0].all_cells is not None
    inv = np.linalg.inv(aff)
    every_ijk = plans[0].all_cells.centre @ inv[:3, :3].T + inv[:3, 3]
    out, row0 = {}, 0
    for plan in plans:
        r1 = row0 + plan.operator.shape[0]
        mats = _mix_matrices(
            plan, w_all[row0:r1].tocsr(), every_ijk, shape3,
            abs(np.linalg.det(aff[:3, :3])), normalise,
        )  # fmt: skip
        out[plan.n_depths] = [
            (roi.rid, name, hemi, m) for roi, (name, hemi, m) in zip(rois, mats, strict=True)
        ]
        row0 = r1
    return out


def depth_sampling_matrices(
    paths: str | list[str],
    surfaces,
    data_affine: np.ndarray,
    data_shape: tuple[int, ...],
    opts: DepthOptions = DepthOptions(),
    *,
    normalise: bool = True,
    device=None,
) -> list[tuple[str, str, np.ndarray]]:
    """How each measured depth bin mixes the true depths: ``(name, hemi, M)`` per ROI.

    ``M[k, j]`` (``K x K``, **depth 1 deepest first**, as the tables are) is the share
    of measured bin ``k`` that comes from true depth ``j`` when the data are on
    ``(data_affine, data_shape)`` and read as :func:`build_depth_roi_set` reads them.
    A voxel holds the volume of each true depth inside it (the box the voxel
    integrates over); the ROI's reads interpolate those voxels with ``opts.interp``
    and average with the cells' volumes. So ``M @ profile`` is what the pipeline
    measures for a depth profile that is the same along the cortex -- the laminar
    model's depth point-spread function, computed for this sampling rather than
    fitted as a Gaussian.

    Rows sum to the share of the read voxels that is cortex; ``normalise`` scales
    them to 1, the reference's convention (no dilution by white matter or CSF). The
    unnormalised form assumes every tissue has the same baseline signal per volume,
    which CSF does not -- so it is the cautious one only for its shape.
    """
    by_k = depth_sampling_multi(
        paths, surfaces, data_affine, data_shape, opts, [opts.n_depths],
        normalise=normalise, device=device,
    )  # fmt: skip
    return [(name, hemi, m) for _, name, hemi, m in by_k[opts.n_depths]]


def write_depth_psf(path: str, matrices, ids: list | None = None, ks: list | None = None) -> str:
    """The matrices as a tidy table: one line per ROI x measured depth x true depth,
    depth 1 deepest, unnormalised ``weight`` -- readers normalise if they want to.
    ``ks`` (one depth count per matrix) adds a ``K`` column after ``hemi``."""
    with open(path, "w") as fh:
        fh.write("roi\tname\themi\t" + ("K\t" if ks is not None else "")
                 + "depth\ttrue_depth\tweight\n")  # fmt: skip
        for i, (name, hemi, m) in enumerate(matrices):
            rid = ids[i] if ids is not None else i + 1
            kcol = f"{ks[i]}\t" if ks is not None else ""
            for k in range(m.shape[0]):
                for j in range(m.shape[1]):
                    fh.write(f"{rid}\t{name}\t{hemi}\t{kcol}{k + 1}\t{j + 1}\t{m[k, j]:.6g}\n")
    return path


def read_depth_psf_by_k(path: str) -> dict[int, dict[tuple[str, str], np.ndarray]]:
    """A ``*_depth_psf.tsv`` as ``{K: {(name, hemi): M}}``, each ``(K, K)``, depth 1
    deepest, weights as written. A file without a ``K`` column has one count."""
    import csv

    raw: dict[tuple[int, str, str], dict] = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        k_col = int(r["K"]) if "K" in r else 0
        raw.setdefault((k_col, r["name"], r["hemi"]), {})[
            (int(r["depth"]), int(r["true_depth"]))
        ] = float(r["weight"])
    out: dict[int, dict] = {}
    for (k_col, name, hemi), entries in raw.items():
        k = max(d for d, _ in entries)
        m = np.zeros((k, k))
        for (d, j), wgt in entries.items():
            m[d - 1, j - 1] = wgt
        out.setdefault(k_col or k, {})[(name, hemi)] = m
    return out


def depth_list(args) -> list[int]:
    """The depth counts ``-depths`` asked for (a bare int from older callers works too)."""
    d = getattr(args, "depths", 1)
    ks = [d] if isinstance(d, int) else [int(k) for k in d]
    return list(dict.fromkeys(ks)) or [1]


def options_from_args(args) -> DepthOptions:
    """The options of the first depth count; the others come from :func:`depth_list`."""
    return DepthOptions(
        n_depths=depth_list(args)[0],
        method=str(args.depth_method),
        dxyz=args.depth_dxyz,
        interp=str(getattr(args, "depth_interp", "cubic")),
        smooth=int(getattr(args, "depth_smooth", 1)),
        claim=str(getattr(args, "roi_claim", "mid")),
    )


def roi_set_from_args(
    args,
    volume_shape: tuple[int, ...],
    affine: np.ndarray | None,
    mask: np.ndarray | None = None,
    *,
    header=None,
    input_path: str | None = None,
    restrict_to_rois: bool = True,
    device=None,
) -> RoiSet:
    """The ``-atlas/-rois`` set a fit tool asked for: over ROI voxels, or -- when
    surfaces are given (``-fs_subj`` / ``-white_mesh``) -- per ROI per depth along the
    ribbon. Raises ``ValueError`` with a message for the user."""
    from fastfuncstuff.processing.roi_collapse import build_roi_set
    from fastfuncstuff.processing.roi_reads import SERIES, read_series_table, roi_set_from_series
    from fastfuncstuff.surface.ribbon_args import depth_requested

    if list(args.atlas or []) == [SERIES]:
        table = read_series_table(input_path or "")
        if table is None:
            raise ValueError("the input is not an ffs_nwarp ROI series")
        return roi_set_from_series(table, volume_shape, input_path or "")
    on_surface = header is not None and _surface_data(header)
    if not depth_requested(args):
        if depth_list(args) != [1] or getattr(args, "depth_method", None) == "equivolume":
            raise ValueError(
                "-depths / -equivolume sample along surfaces: give -fs_subj or -white_mesh"
            )
        return build_roi_set(
            args.atlas, volume_shape, mask=mask, restrict_to_rois=restrict_to_rois,
            reference_header=header, reference_path=input_path,
        )  # fmt: skip
    if on_surface:
        raise ValueError(
            "depth sampling reads a volume along the ribbon; surface data already lie on one "
            "surface -- drop -fs_subj / -white_mesh"
        )
    if affine is None:
        raise ValueError("depth sampling needs the data's affine")
    from fastfuncstuff.surface.volume_depth import resolve_ribbon_surfaces

    log_lines: list[str] = []
    surfaces = resolve_ribbon_surfaces(
        args.fs_subj, args.white, args.pial, args.white_mesh, args.pial_mesh,
        args.cortex_label, log=log_lines.append,
    )  # fmt: skip
    opts = options_from_args(args)
    ks = depth_list(args)
    sets = build_depth_roi_sets(
        args.atlas, surfaces, affine, volume_shape, mask, opts, ks, device=device
    )
    # The laminar model's depth PSF for exactly this sampling (unnormalised; the
    # analysis mask is left out -- it is geometry, not data).
    psf = depth_sampling_multi(
        args.atlas, surfaces, affine, volume_shape, opts, ks, normalise=False, device=device
    )
    roi_set = sets[0] if len(ks) == 1 else RoiSet.stack(sets, "K", sorted(ks))
    flat = [(k, rid, name, hemi, m) for k in sorted(ks) for rid, name, hemi, m in psf[k]]
    roi_set.depth_psf = [(name, hemi, m) for _, _, name, hemi, m in flat]
    roi_set.depth_psf_ids = [rid for _, rid, *_ in flat]
    roi_set.depth_psf_ks = [k for k, *_ in flat]
    roi_set.notes[:0] = [line.strip() for line in log_lines]
    return roi_set


def _surface_data(header) -> bool:
    from fastfuncstuff.io.gifti import surface_meta

    return surface_meta(header) is not None


__all__ = [
    "DepthOptions",
    "DepthPlan",
    "build_depth_roi_set",
    "build_depth_roi_sets",
    "READ_KERNELS",
    "default_step",
    "depth_list",
    "depth_sampling_matrices",
    "depth_sampling_multi",
    "psf_halo_mm",
    "read_depth_psf_by_k",
    "write_depth_psf",
    "pass_groups",
    "plan_depth_multi",
    "plan_depth_rois",
    "read_depth_rois",
    "tile_costs",
    "options_from_args",
    "roi_set_from_args",
]
