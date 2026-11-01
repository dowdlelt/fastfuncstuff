"""Fit on ROI means instead of voxels: the one place ``-atlas`` / ``-rois`` live.

Averaging a region's voxel timeseries *before* the fit (not fitting voxels and
averaging the estimates) hands the model the sqrt(N)-cleaner signal, and turns a
300k-voxel fit into a few hundred rows. Every tool that offers it shares this
module, so a label map means the same thing in ``ffs_varpart``, ``ffs_fitbasis``,
``ffs_deconvolve`` and ``ffs_reml``.

An ROI source is any of

* a 3-D integer **label map** -- one ROI per distinct non-zero value (disjoint);
* a 3-D **binary mask** -- the one-label case of the above, named after the file;
* a 4-D **stack** -- one volume per ROI, values used as weights (may overlap).

Several files stack into one ROI set (``-rois lh_v1.nii.gz rh_v1.nii.gz``). A
single 3-D file keeps the cheap grouped-mean path; anything else becomes an
``(n_rois, n_voxels)`` weight matrix. Only the disjoint case can be painted back
onto voxels without inventing structure, so only it is promised a "filled-in" map.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor

from fastfuncstuff.unique import sorted_unique


def build_roi_weights(
    atlas: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list, np.ndarray]:
    """Turn a parcellation into per-ROI voxel weights.

    Accepts either form an atlas usually ships as:

    * **3-D integer label map** — one ROI per distinct non-zero value. ROIs are
      disjoint, so collapsing is a grouped mean.
    * **4-D stack** — one volume per ROI, values used as weights. Binary masks and
      probabilistic/partial-volume maps both work, and ROIs may overlap.

    Returns ``(roi_index_or_weights, roi_ids, roi_sizes)``. For the 3-D case the first
    element is a ``(n_vox,)`` int array giving each voxel's ROI slot (``-1`` = unassigned),
    which lets the collapse run as an ``index_add`` instead of materialising an
    ``(n_rois, n_vox)`` matrix. For the 4-D case it is that ``(n_rois, n_vox)`` weight
    matrix, which is affordable because overlapping atlases carry far fewer ROIs.
    """
    if mask is not None:
        mask = np.asarray(mask).astype(bool)

    if atlas.ndim == 3:
        flat = atlas.reshape(-1)
        if mask is not None:
            flat = flat[mask.reshape(-1)]
        ids = [int(v) for v in sorted_unique(flat) if v != 0]
        if not ids:
            raise ValueError("atlas contains no non-zero labels")
        lookup = {v: i for i, v in enumerate(ids)}
        slot = np.full(flat.shape[0], -1, dtype=np.int64)
        for v, i in lookup.items():
            slot[flat == v] = i
        sizes = np.bincount(slot[slot >= 0], minlength=len(ids)).astype(np.float64)
        return slot, ids, sizes

    if atlas.ndim == 4:
        n_rois = atlas.shape[3]
        flat = atlas.reshape(-1, n_rois)
        if mask is not None:
            flat = flat[mask.reshape(-1)]  # crop before widening: float64 over the full grid is GBs
        w = np.clip(flat.T.astype(np.float64), 0.0, None)  # (n_rois, n_vox)
        sizes = w.sum(axis=1)
        if not np.any(sizes > 0):
            raise ValueError("4-D atlas has no ROI with positive weight inside the mask")
        ids = list(range(1, n_rois + 1))
        return w, ids, sizes

    raise ValueError(f"atlas must be 3-D (labels) or 4-D (per-ROI masks), got {atlas.ndim}-D")


def collapse_to_rois(
    betas: Tensor,
    roi_spec: np.ndarray,
    roi_sizes: np.ndarray,
    device: torch.device | None = None,
    chunk_size: int = 20000,
) -> Tensor:
    """Average ``(n_vox, n_cols)`` data (trial betas or timeseries) within each ROI.

    Returns ``(n_rois, n_cols)``. Empty ROIs come back as zeros rather than NaN so a
    partial atlas cannot poison downstream reductions.

    Collapsing before partitioning is not only a speed lever. Averaging voxels raises the
    noise ceiling roughly as sqrt(n_voxels) for signal shared across the ROI, which
    directly relaxes the ncsnr floor that rank selection needs -- an interaction rank that
    is undetectable per voxel is often resolvable per parcel.
    """
    if device is None:
        device = betas.device
    n_cols = betas.shape[1]
    sizes = torch.as_tensor(roi_sizes, dtype=torch.float32, device=device).clamp_min(1e-12)
    n_rois = len(roi_sizes)
    out = torch.zeros(n_rois, n_cols, device=device, dtype=torch.float32)

    # Chunk the voxel axis on both paths: a timeseries is 100x wider than a trial
    # table, and the whole of it should never have to sit on the card at once.
    if roi_spec.ndim == 1:
        slot_full = torch.as_tensor(roi_spec, dtype=torch.long)
        for c0 in range(0, slot_full.shape[0], chunk_size):
            slot = slot_full[c0 : c0 + chunk_size].to(device)
            keep = slot >= 0
            out.index_add_(0, slot[keep], betas[c0 : c0 + chunk_size].to(device)[keep].float())
        return out / sizes[:, None]

    if sp.issparse(roi_spec):
        # Depth sampling: a few nonzeros per row out of the whole ribbon's voxels.
        coo = sp.coo_matrix(roi_spec)
        w = torch.sparse_coo_tensor(
            np.vstack([coo.row, coo.col]),
            coo.data.astype(np.float32),
            coo.shape,
            device=device,
            check_invariants=False,
        )
        n_vox = coo.shape[1]
        for c0 in range(0, n_cols, chunk_size):
            c1 = min(c0 + chunk_size, n_cols)
            out[:, c0:c1] = torch.sparse.mm(w, betas[:, c0:c1].float().to(device))
        return out / sizes[:, None]

    w_full = torch.as_tensor(roi_spec, dtype=torch.float32)
    n_vox = w_full.shape[1]
    for c0 in range(0, n_vox, chunk_size):
        c1 = min(c0 + chunk_size, n_vox)
        out += w_full[:, c0:c1].to(device) @ betas[c0:c1].to(device).float()
    return out / sizes[:, None]


def paint_rois_to_voxels(
    values: Tensor | np.ndarray,
    roi_spec: np.ndarray,
    n_voxels: int,
    fill: float = 0.0,
) -> np.ndarray:
    """Broadcast per-ROI values back onto voxels, for display.

    ``values`` is ``(n_rois, ...)``; the result is ``(n_voxels, ...)``.

    The result carries no more spatial information than the ROI table it came from --
    every voxel in a parcel gets the same number. It exists so parcel results can be
    rendered on a brain; do not read within-parcel structure into it, because there
    isn't any.

    Overlapping 4-D atlases are resolved by weighted average of the ROIs covering each
    voxel, which reduces to the exact ROI value wherever coverage is disjoint and binary.
    Voxels no ROI covers get *fill*.
    """
    vals = np.asarray(values.cpu() if isinstance(values, Tensor) else values, dtype=np.float64)
    trailing = vals.shape[1:]

    if sp.issparse(roi_spec):
        raise ValueError("depth-sampled ROIs share voxels across depths; they cannot be painted")
    if roi_spec.ndim == 1:
        out = np.full((n_voxels, *trailing), fill, dtype=np.float32)
        assigned = roi_spec >= 0
        out[assigned] = vals[roi_spec[assigned]].astype(np.float32)
        return out

    w = np.asarray(roi_spec, dtype=np.float64)  # (n_rois, n_vox)
    denom = w.sum(axis=0)
    out = np.full((n_voxels, *trailing), fill, dtype=np.float32)
    covered = denom > 0
    mixed = np.tensordot(w.T, vals, axes=(1, 0))  # (n_vox, ...)
    out[covered] = (mixed[covered] / denom[covered].reshape(-1, *([1] * len(trailing)))).astype(
        np.float32
    )
    return out


# -- the set -----------------------------------------------------------------------


@dataclass
class RoiSet:
    """ROIs resolved against a grid, a mask and the voxels a tool will load.

    ``spec`` indexes the ``True`` voxels of ``rows`` in boolean-index order -- the
    same order ``data[rows]`` and the loader's ``mask_array`` give -- so collapsing
    data loaded with ``mask_array=rows`` needs no remapping.
    """

    spec: np.ndarray
    ids: list[int]
    names: list[str]
    sizes: np.ndarray
    rows: np.ndarray  # (X, Y, Z) bool
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Extra per-row columns written after ``name`` (depth sampling: hemi, depth, ...).
    index: dict[str, list] = field(default_factory=dict)
    #: Header of the ``sizes`` column.
    size_name: str = "n_voxels"
    #: Row files (curves, trials) one line per entry x ROI -- tidy -- rather than one
    #: column per ROI. Depth sampling turns it on: ROI x depth is too wide to read.
    long_rows: bool = False
    #: What ``collapse`` divides each row's weighted sum by, when that is not ``sizes``:
    #: an ROI series' elements are already means, but the table still reports their size.
    divisor: np.ndarray | None = None
    #: Depth-sampled sets: ``(name, hemi, K x K)`` per ROI, how each measured depth mixes
    #: the true depths (:func:`processing.roi_depth.depth_sampling_matrices`).
    depth_psf: list | None = None
    #: ROI id and depth count of each ``depth_psf`` matrix (parallel to it).
    depth_psf_ids: list | None = None
    depth_psf_ks: list | None = None

    @staticmethod
    def stack(sets: list[RoiSet], label: str, labels: list) -> RoiSet:
        """Several sparse sets of the same ROIs on one grid as one: every row of every set,
        with ``label`` (e.g. ``K``) as the first extra table column.

        The sets' voxel rows are united so a single load serves them all; each set's
        spec is re-indexed onto the union. The fit tools treat every row as an ROI, so a
        multi-depth-count run is one fit of ``sum_k n_rois * k`` series.
        """
        if not sets:
            raise ValueError("nothing to stack")
        shape = sets[0].rows.shape
        union = np.zeros(shape, bool)
        for st in sets:
            if not sp.issparse(st.spec) or st.rows.shape != shape:
                raise ValueError("only sparse ROI x depth sets on one grid can be stacked")
            union |= st.rows
        union_flat = np.flatnonzero(union.reshape(-1))
        specs = []
        for st in sets:
            touched = np.flatnonzero(st.rows.reshape(-1))
            coo = sp.coo_matrix(st.spec)
            cols = np.searchsorted(union_flat, touched)[coo.col]
            specs.append(
                sp.csr_matrix(
                    (coo.data, (coo.row, cols)), shape=(st.spec.shape[0], union_flat.size)
                )
            )
        index: dict[str, list] = {label: []}
        for key in sets[0].index:
            index[key] = []
        for st, lab in zip(sets, labels, strict=True):
            index[label] += [str(lab)] * len(st.ids)
            for key in sets[0].index:
                index[key] += list(st.index[key])
        first = sets[0]
        return RoiSet(
            spec=sp.vstack(specs).tocsr(),
            ids=[i for st in sets for i in st.ids],
            names=[n for st in sets for n in st.names],
            sizes=np.concatenate([np.asarray(st.sizes) for st in sets]),
            rows=union,
            sources=first.sources,
            notes=first.notes
            + [f"{len(sets)} depth counts ({', '.join(str(x) for x in labels)}) sampled together"],
            index=index,
            size_name=first.size_name,
            long_rows=True,
        )

    @property
    def n_rois(self) -> int:
        return len(self.ids)

    @property
    def n_rows(self) -> int:
        return int(self.rows.sum())

    @property
    def disjoint(self) -> bool:
        """A grouped-mean label map: the only case that paints back faithfully."""
        return not sp.issparse(self.spec) and self.spec.ndim == 1

    @property
    def n_voxels(self) -> np.ndarray:
        """Voxels per ROI (the weight total for a weighted ROI)."""
        return self.sizes

    def collapse(self, data: Tensor, device: torch.device | None = None) -> Tensor:
        """``(n_rows, n_cols)`` -> ``(n_rois, n_cols)`` ROI means."""
        if data.shape[0] != self.n_rows:
            raise ValueError(
                f"data has {data.shape[0]} voxel rows but the ROI set indexes {self.n_rows}"
            )
        div = self.sizes if self.divisor is None else self.divisor
        return collapse_to_rois(data, self.spec, div, device=device)

    def dominant(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each ROI's commonest of the per-voxel ``labels`` ``(n_rows,)`` ints >= 0, by weight.

        Returns ``(label, share)`` per ROI, ``share`` being the weight fraction the
        winner holds -- 1 where every voxel of the ROI carries the same label (an
        ``ffs_hrftdv -atlas`` index), less where the ROI averages over several.
        """
        labels = np.asarray(labels, np.int64).reshape(-1)
        if labels.size != self.n_rows:
            raise ValueError(f"{labels.size} labels for {self.n_rows} voxel rows")
        if labels.size and labels.min() < 0:
            raise ValueError("labels must be >= 0")
        if sp.issparse(self.spec):
            w = sp.csr_matrix(self.spec)
        elif self.spec.ndim == 1:
            j = np.flatnonzero(self.spec >= 0)
            w = sp.csr_matrix(
                (np.ones(j.size), (self.spec[j], j)), shape=(self.n_rois, self.n_rows)
            )
        else:
            w = sp.csr_matrix(np.asarray(self.spec, np.float64))
        n_lab = int(labels.max()) + 1 if labels.size else 1
        onehot = sp.csr_matrix(
            (np.ones(labels.size), (np.arange(labels.size), labels)), shape=(labels.size, n_lab)
        )
        mass = (w @ onehot).toarray()  # (n_rois, n_labels)
        total = mass.sum(1)
        best = mass.argmax(1)
        share = np.divide(mass[np.arange(mass.shape[0]), best], total, where=total > 0,
                          out=np.zeros(mass.shape[0]))  # fmt: skip
        return best, share

    def paint(self, values: Tensor | np.ndarray, fill: float = 0.0) -> np.ndarray:
        """``(n_rois, ...)`` -> ``(n_rows, ...)``, every voxel carrying its ROI's value."""
        return paint_rois_to_voxels(values, self.spec, self.n_rows, fill=fill)

    def to_volume(self, values: Tensor | np.ndarray, fill: float = 0.0) -> np.ndarray:
        """``(n_rois, ...)`` -> ``(X, Y, Z, ...)`` painted maps."""
        painted = self.paint(values, fill=fill)
        out = np.full((*self.rows.shape, *painted.shape[1:]), fill, dtype=np.float32)
        out[self.rows] = painted
        return out

    def describe(self) -> list[str]:
        sz = self.sizes
        if sp.issparse(self.spec):
            kind, unit = "ROI x depth rows", "mm3"
        else:
            kind = "label map" if self.disjoint else "weighted/overlapping masks"
            unit = "voxels"
        return [
            *self.notes,
            f"{self.n_rois} {kind} from {', '.join(self.sources)}: "
            f"min {sz.min():.0f}, median {np.median(sz):.0f}, max {sz.max():.0f} {unit}",
        ]

    def _row_keys(self, i: int) -> list:
        return [self.ids[i], self.names[i], *(self.index[k][i] for k in self.index)]

    def write_depth_psf(self, stem: str) -> str | None:
        """``{stem}_depth_psf.tsv`` when the set was sampled by depth, else None."""
        if not self.depth_psf:
            return None
        from fastfuncstuff.processing.roi_depth import write_depth_psf

        ks = self.depth_psf_ks if len(set(self.depth_psf_ks or [])) > 1 else None
        return write_depth_psf(f"{stem}_depth_psf.tsv", self.depth_psf, self.depth_psf_ids, ks=ks)

    def write_table(self, path: str | Path, columns: dict[str, np.ndarray | Tensor]) -> str:
        """One row per ROI: id, name, size, then each named column."""
        cols = {
            k: np.asarray(v.cpu() if isinstance(v, Tensor) else v).reshape(self.n_rois, -1)
            for k, v in columns.items()
        }
        header = ["roi", "name", *self.index, self.size_name]
        for k, v in cols.items():
            header += [k] if v.shape[1] == 1 else [f"{k}_{j}" for j in range(v.shape[1])]
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(header)
            for i in range(self.n_rois):
                row = [*self._row_keys(i), f"{self.sizes[i]:.6g}"]
                for v in cols.values():
                    row += [f"{x:.6g}" for x in v[i]]
                w.writerow(row)
        return str(path)

    def write_timecourses(
        self,
        path: str | Path,
        times: np.ndarray,
        values: np.ndarray | Tensor,
        value_name: str = "beta",
    ) -> str:
        """Plain text, time down the rows and one column per ROI (``values``: ROI x time)."""
        times_txt = [f"{float(t):.6g}" for t in times]
        return self.write_rows(path, {"time": times_txt}, values, value_name)

    def write_rows(
        self,
        path: str | Path,
        index: dict[str, list],
        values: np.ndarray | Tensor,
        value_name: str = "value",
    ) -> str:
        """Plain text, one row per entry of *index*'s columns, then one column per ROI.

        ``values`` is ``(n_rois, n_rows)``: response curves (rows = lags), single-trial
        betas (rows = trials). With :attr:`long_rows`, one line per entry per ROI
        instead -- *index*, then the ROI's own columns, then ``value_name`` -- which
        pandas / seaborn take as is.
        """
        vals = np.asarray(values.cpu() if isinstance(values, Tensor) else values)
        keys = list(index)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            if self.long_rows:
                w.writerow([*keys, "roi", "name", *self.index, value_name])
                for i in range(self.n_rois):
                    roi = self._row_keys(i)
                    for r in range(vals.shape[1]):
                        w.writerow([*(index[k][r] for k in keys), *roi, f"{vals[i, r]:.6g}"])
                return str(path)
            w.writerow([*keys, *("_".join(n.split()) for n in self.names)])
            for r, col in enumerate(vals.T):
                w.writerow([*(index[k][r] for k in keys), *(f"{x:.6g}" for x in col)])
        return str(path)


def _stem(path: str) -> str:
    name = Path(path.split("[")[0]).name
    for ext in (
        ".nii.gz", ".nii.zst", ".nii", ".mgz", ".mgh",
        ".label.gii", ".shape.gii", ".func.gii", ".gii", ".label", ".annot",
    ):  # fmt: skip
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


_FREESURFER_ROI = (".label", ".annot")


def _surface_reference(header, path) -> dict | None:
    """Mesh identity of the data an ROI must match: None for volume data."""
    from fastfuncstuff.io.freesurfer import infer_label
    from fastfuncstuff.io.gifti import surface_meta

    meta = None if header is None else surface_meta(header)
    if meta is None:
        return None
    hemi = infer_label(str(path))[0] if path else None
    return {"fingerprint": meta.get("mesh_fingerprint"), "hemi": hemi}


def _load_roi_file(f: str, shape3: tuple, surface: dict | None):
    """``(array, label table, note)`` for one ROI source, checked against the data.

    Surface data take GIfTI (``.label.gii`` / ``.shape.gii``) and FreeSurfer
    ``.label`` / ``.annot`` files; volume data take NIfTI. A GIfTI that names its
    mesh must name the data's; FreeSurfer files carry no mesh identity, so only the
    vertex count and the lh/rh in the file names can be checked.
    """
    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.io.freesurfer import infer_label
    from fastfuncstuff.io.gifti import is_gifti, surface_meta
    from fastfuncstuff.io.labels import LabelEntry, label_table

    plain = f.split("[")[0]
    is_fs = plain.lower().endswith(_FREESURFER_ROI)
    on_surface = is_fs or is_gifti(plain)
    if on_surface != (surface is not None):
        data_kind = "surface" if surface is not None else "volume"
        raise ValueError(f"{f}: a {'surface' if on_surface else 'volume'} ROI for {data_kind} data")
    note = None
    if surface is not None:
        hemi = infer_label(plain)[0]
        if hemi and surface["hemi"] and hemi != surface["hemi"]:
            raise ValueError(f"{f}: a {hemi} ROI for {surface['hemi']} data")

    if is_fs:
        import nibabel.freesurfer as nfs

        n_vert = shape3[0]
        if plain.lower().endswith(".annot"):
            from fastfuncstuff.io.freesurfer import NOT_A_REGION, read_annotation

            annot = read_annotation(plain, n_vertices=n_vert)
            real = [k for k, n in enumerate(annot.names) if n not in NOT_A_REGION]
            flat = np.zeros(n_vert, dtype=np.int64)
            for k in real:
                flat[annot.labels == k] = k + 1  # 0 stays background
            table = {
                k + 1: LabelEntry(k + 1, annot.names[k], tuple(int(c) for c in annot.rgba[k, :3]))
                for k in real
            }
        else:
            ids = np.asarray(nfs.read_label(plain), dtype=np.int64)
            if ids.size and (ids.min() < 0 or ids.max() >= n_vert):
                raise ValueError(f"{f}: vertex ids reach {ids.max()}, the mesh has {n_vert}")
            flat = np.zeros(n_vert, dtype=np.int64)
            flat[ids] = 1
            table = {}
        note = f"{Path(plain).name}: FreeSurfer file, mesh checked by vertex count only"
        return flat.reshape(shape3), table, note

    img = load_nifti(f)
    arr = np.asanyarray(img.dataobj)
    if arr.ndim == 4 and arr.shape[3] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (3, 4):
        raise ValueError(f"{f}: ROI files must be 3-D or 4-D, got {arr.ndim}-D")
    if arr.shape[:3] != shape3:
        what = "vertex count" if surface is not None else "grid"
        raise ValueError(
            f"{f}: ROI {what} {arr.shape[:3]} does not match the data's {shape3}"
            + ("" if surface is not None else "; resample it first (ffs_util_resample)")
        )
    table = label_table(f, img) if arr.ndim == 3 else {}
    if surface is not None:
        mine = (surface_meta(img.header) or {}).get("mesh_fingerprint")
        if mine and surface["fingerprint"] and mine != surface["fingerprint"]:
            raise ValueError(f"{f}: drawn on a different mesh from the data")
        if not mine:
            note = f"{Path(plain).name}: no mesh fingerprint, checked by vertex count only"
        if not table:
            table = _gifti_label_table(plain)
    return arr, table, note


def _gifti_label_table(path: str) -> dict:
    """A ``.label.gii``'s own LabelTable (``gifti_as_nifti`` keeps only the values)."""
    import nibabel as nib

    from fastfuncstuff.io.labels import LabelEntry

    img = nib.load(path)
    entries = getattr(getattr(img, "labeltable", None), "labels", None) or []
    out = {}
    for e in entries:
        if int(e.key) != 0 and e.label:
            rgb = None if e.red is None else tuple(int(round(255 * c)) for c in e.rgba[:3])
            out[int(e.key)] = LabelEntry(int(e.key), str(e.label), rgb)
    return out


def build_roi_set(
    paths: str | list[str],
    volume_shape: tuple[int, ...],
    mask: np.ndarray | None = None,
    *,
    restrict_to_rois: bool = True,
    reference_header=None,
    reference_path: str | None = None,
) -> RoiSet:
    """Resolve ROI files against *volume_shape* and an optional analysis *mask*.

    With ``restrict_to_rois`` (the fit tools) the rows are ``ROI cover & mask``: only
    voxels some ROI actually uses are ever loaded. Without it (``ffs_varpart``, whose
    data are already in memory) the rows are the mask itself and uncovered voxels hold
    slot ``-1``. ROIs that the mask empties are dropped and reported, not silently kept.

    ``reference_header`` (the data's header) and ``reference_path`` make surface data
    recognisable: then ROIs must be surface files on the same mesh and hemisphere.
    """
    files = [paths] if isinstance(paths, str) else list(paths)
    if not files:
        raise ValueError("no ROI files given")
    shape3 = tuple(volume_shape[:3])
    mask3 = None if mask is None else np.asarray(mask).astype(bool).reshape(shape3)

    surface = _surface_reference(reference_header, reference_path)
    # (file, array, label table) per source; arrays kept as loaded.
    loaded, file_notes = [], []
    for f in files:
        arr, table, note = _load_roi_file(f, shape3, surface)
        loaded.append((f, arr, table))
        if note:
            file_notes.append(note)
    unit = "vertices" if surface is not None else "voxels"

    cover = np.zeros(shape3, dtype=bool)
    for _, arr, _ in loaded:
        cover |= (arr != 0) if arr.ndim == 3 else np.any(arr > 0, axis=3)
    n_cover = int(cover.sum())
    if n_cover == 0:
        raise ValueError("the ROI file(s) contain no non-zero voxels")

    notes = [*file_notes, f"ROI cover: {n_cover:,} {unit}"]
    if mask3 is not None:
        n_in = int((cover & mask3).sum())
        notes.append(
            f"intersected with -mask: {n_in:,} of {n_cover:,} ROI {unit} are inside it "
            f"({n_cover - n_in:,} dropped)"
        )
        if n_in == 0:
            raise ValueError("no ROI voxel is inside -mask")
    rows = (
        (cover & mask3 if mask3 is not None else cover)
        if restrict_to_rois
        else (mask3 if mask3 is not None else np.ones(shape3, dtype=bool))
    )

    single_label_map = len(loaded) == 1 and loaded[0][1].ndim == 3
    if single_label_map:
        f, arr, table = loaded[0]
        if not np.array_equal(arr, np.round(arr)):
            raise ValueError(
                f"{f}: a 3-D ROI file must hold integer labels; give weights as a 4-D stack"
            )
        spec, ids, sizes = build_roi_weights(arr.astype(np.int64), mask=rows)
        if len(ids) == 1 and ids[0] not in table:
            names = [_stem(f)]
        else:
            names = [table[i].name if i in table else str(i) for i in ids]
    else:
        columns, names = [], []
        for f, arr, table in loaded:
            if arr.ndim == 3:
                for lab in (int(v) for v in sorted_unique(arr) if v != 0):
                    columns.append((arr == lab).astype(np.float32))
                    one = len({int(v) for v in sorted_unique(arr) if v != 0}) == 1
                    names.append(
                        _stem(f)
                        if one and lab not in table
                        else (table[lab].name if lab in table else f"{_stem(f)}_{lab}")
                    )
            else:
                for j in range(arr.shape[3]):
                    columns.append(np.asarray(arr[..., j], dtype=np.float32))
                    names.append(f"{_stem(f)}_{j + 1}" if arr.shape[3] > 1 else _stem(f))
        stack = np.stack(columns, axis=3)
        spec, ids, sizes = build_roi_weights(stack, mask=rows)
        ids = list(range(1, len(ids) + 1))
        del stack

    # A mask can empty an ROI entirely; keep only the ROIs that still have voxels.
    alive = np.flatnonzero(sizes > 0)
    if len(alive) < len(ids):
        gone = [names[i] for i in range(len(ids)) if sizes[i] <= 0]
        notes.append(
            f"-mask emptied {len(gone)} ROI(s), dropped: {', '.join(gone[:8])}"
            + (" ..." if len(gone) > 8 else "")
        )
        if spec.ndim == 1:
            remap = np.full(len(ids), -1, dtype=np.int64)
            remap[alive] = np.arange(len(alive))
            spec = np.where(spec >= 0, remap[np.clip(spec, 0, None)], -1)
        else:
            spec = spec[alive]
        ids = [ids[i] for i in alive]
        names = [names[i] for i in alive]
        sizes = sizes[alive]
        if not ids:
            raise ValueError("-mask leaves every ROI empty")

    return RoiSet(
        spec=spec, ids=ids, names=names, sizes=sizes, rows=rows,
        sources=[Path(f.split("[")[0]).name for f, _, _ in loaded], notes=notes,
    )  # fmt: skip


def add_roi_args(group) -> None:
    """Register ``-atlas`` / ``-rois`` (one flag, two spellings) on an argparse group."""
    group.add_argument(
        "-atlas",
        "-rois",
        dest="atlas",
        nargs="+",
        default=None,
        metavar="ROI_FILE",
        help=(
            "Fit ROI MEANS instead of voxels.  Each ROI's voxel timeseries is "
            "averaged BEFORE the fit.  Each file is a 3-D integer label map (one "
            "ROI per value), a 3-D binary mask (one ROI, named after the file) or a "
            "4-D stack (one weight volume per ROI; may overlap).  Surface (.gii) data "
            "take surface ROIs on the same mesh: .label.gii / .shape.gii, or "
            "FreeSurfer .label / .annot.  Several files stack into one set.  "
            "Intersected with -mask when one is given.  With surfaces (-fs_subj, or "
            "-white_mesh/-pial_mesh) volume data are read along the cortical ribbon "
            "instead: one row per ROI per depth (-depths, -equivolume), from surface "
            "ROIs or volume ROIs mapped to their columns.  "
            "Writes a per-ROI stats table and plain-text response curves; with a "
            "single label map, also paints each ROI's value back into its voxels."
        ),
    )
