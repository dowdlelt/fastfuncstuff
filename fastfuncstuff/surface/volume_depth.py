"""LayNii-compatible cortical depth volumes straight from white and pial meshes.

LN2_LAYERS takes a *rim* image -- GM plus one-voxel CSF/WM borders -- and
rebuilds the geometry from it: distances to the borders by voxel propagation,
equivolume factors from voxel-counted curvature, smoothed for hundreds of
iterations. A FreeSurfer subject already *has* that geometry, exactly: GM is
the solid between the white and pial meshes, the distances are point-to-mesh
distances, and a cortical column is the GM above a patch of white surface. So
every output here is computed from the meshes and only written onto voxels at
the end, at whatever grid is asked for.

Conventions follow LN2_LAYERS so the files drop into a LayNii workflow: the
metric is 0 at WM and 1 at CSF, layers are ``ceil(metric * N)`` (layer 1
deepest), borders are excluded from metric/layers/thickness, and midGM is a
one-voxel sheet at the voxel nearest each 0.5 crossing.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from fastfuncstuff.surface.mesh import MeshTopology
from fastfuncstuff.surface.voxelize import ClosestPoints, MeshDistance, winding_number

#: Rim labels, as LN2_LAYERS reads them.
RIM_CSF, RIM_WM, RIM_GM = 1, 2, 3

#: LN2_LAYERS's file tags, in the order they are written.
OUTPUT_TAGS = (
    "rim",
    "metric_equidist",
    "layers_equidist",
    "midGM_equidist",
    "metric_equivol",
    "layers_equivol",
    "midGM_equivol",
    "thickness",
)


@dataclass
class RibbonSurfaces:
    """One hemisphere's white and pial meshes in scanner RAS (same faces)."""

    name: str
    white: np.ndarray
    pial: np.ndarray
    faces: np.ndarray
    #: Cortex vertex mask (``?h.cortex.label``); the medial wall is excluded.
    cortex: np.ndarray | None = None

    @classmethod
    def from_hemisphere(cls, hemi, white: str = "white", pial: str = "pial") -> RibbonSurfaces:
        """From a loaded :class:`~fastfuncstuff.io.freesurfer.Hemisphere`, edits included."""
        for state in (white, pial):
            if state not in hemi.states:
                raise KeyError(f"{hemi.name} has no '{state}' surface")
        return cls(hemi.name, hemi.states[white], hemi.states[pial], hemi.faces, hemi.cortex)


@dataclass
class DepthVolumes:
    """LN2_LAYERS's outputs on one grid, ``(X, Y, Z)``, held sparse.

    Only the GM band is stored -- a few percent of a 0.2 mm grid -- and each
    dense volume is built when asked for, so writing them holds one at a time
    (whole-brain 0.2 mm peaked at 24.7 GB with all eight dense at once).
    """

    shape: tuple[int, int, int]
    n_layers: int
    #: Flat C-order indices of GM voxels (sorted), and the values there.
    gm_idx: np.ndarray
    rho: np.ndarray  # equidistant metric
    equivol: np.ndarray
    thick: np.ndarray
    #: Border voxels and whether each is on the WM side.
    border_idx: np.ndarray
    border_wm: np.ndarray
    #: Positions *within gm_idx* of the two midGM sheets.
    mid_equidist: np.ndarray
    mid_equivol: np.ndarray
    #: GM voxels thicker than ``thick_limit`` -- real cortex is not; these are
    #: surface errors (pial through a vessel or the dura, a bad edit).
    n_thick: int = 0
    thick_limit: float = 6.0
    #: GM-by-fill voxels dropped as medial wall.
    n_medial: int = 0
    #: GM voxels claimed by two hemispheres, or by one's GM and the other's WM.
    n_overlap: int = 0

    @property
    def n_gm(self) -> int:
        return int(self.gm_idx.size)

    def _dense(self, values, idx=None, dtype=np.float32) -> np.ndarray:
        out = np.zeros(int(np.prod(self.shape)), dtype)
        out[self.gm_idx if idx is None else idx] = values
        return out.reshape(self.shape)

    def _layers(self, metric: np.ndarray) -> np.ndarray:
        lay = np.clip(np.ceil(metric * self.n_layers), 1, self.n_layers)
        return self._dense(lay, dtype=np.int16)

    @property
    def rim(self) -> np.ndarray:
        out = self._dense(RIM_GM, dtype=np.int16).reshape(-1)
        out[self.border_idx] = np.where(self.border_wm, RIM_WM, RIM_CSF)
        return out.reshape(self.shape)

    @property
    def metric_equidist(self) -> np.ndarray:
        return self._dense(self.rho)

    @property
    def metric_equivol(self) -> np.ndarray:
        return self._dense(self.equivol)

    @property
    def layers_equidist(self) -> np.ndarray:
        return self._layers(self.rho)

    @property
    def layers_equivol(self) -> np.ndarray:
        return self._layers(self.equivol)

    @property
    def mid_gm_equidist(self) -> np.ndarray:
        return self._dense(1, self.gm_idx[self.mid_equidist], np.int16)

    @property
    def mid_gm_equivol(self) -> np.ndarray:
        return self._dense(1, self.gm_idx[self.mid_equivol], np.int16)

    @property
    def thickness(self) -> np.ndarray:
        return self._dense(self.thick)

    def outputs(self) -> Iterator[tuple[str, np.ndarray]]:
        """``(LN2_LAYERS file tag, volume)``, built one at a time."""
        for tag in OUTPUT_TAGS:
            yield tag, getattr(self, tag.replace("midGM", "mid_gm"))

    def thick_clusters(
        self, affine: np.ndarray, top: int | None = 5
    ) -> list[tuple[int, np.ndarray]]:
        """Connected clumps of over-thick GM, largest first: ``(n_voxels, centroid RAS)``.

        A clump is one place to look at the surfaces; a bare voxel count is not.
        """
        from scipy import ndimage

        over = self._dense(True, self.gm_idx[self.thick > self.thick_limit], bool)
        lab, n = ndimage.label(over)
        if n == 0:
            return []
        sizes = np.bincount(lab.ravel())[1:]
        order = np.argsort(-sizes, kind="stable")[:top]
        centres = ndimage.center_of_mass(over, lab, (order + 1).tolist())
        a = np.asarray(affine, np.float64)
        return [
            (int(sizes[i]), a[:3, :3] @ np.asarray(c) + a[:3, 3])
            for i, c in zip(order, centres, strict=True)
        ]


def _smooth_on_mesh(values: np.ndarray, faces: np.ndarray, n_iter: int) -> np.ndarray:
    """``n_iter`` rounds of 1-ring averaging (self included) of ``(V, ...)`` values."""
    out = np.asarray(values, np.float64)
    if n_iter <= 0:
        return out
    n = out.shape[0]
    topo = MeshTopology.from_faces(faces, n)
    i, j = topo.edges[:, 0], topo.edges[:, 1]
    adj = sp.coo_matrix(
        (np.ones(2 * i.size + n), (np.r_[i, j, np.arange(n)], np.r_[j, i, np.arange(n)])),
        shape=(n, n),
    ).tocsr()
    inv_deg = (1.0 / np.asarray(adj.sum(1)).ravel()).reshape((n,) + (1,) * (out.ndim - 1))
    for _ in range(n_iter):
        out = adj @ out * inv_deg
    return out


def volume_quantile_depth(
    rho: np.ndarray,
    foot: ClosestPoints,
    faces: np.ndarray,
    n_vertices: int,
    column_voxels: float = 64.0,
    bins: int = 64,
) -> np.ndarray:
    """Equivolume depth as each voxel's volume quantile within its cortical column.

    Equivolume layers are the depths that split every column's volume into
    equal parts. Rather than model a column's shape (Waehnert's linear area
    between paired white and pial vertices), measure it: the voxels whose
    nearest white point falls on a vertex's patch *are* that column, their
    count is its volume, and the volume fraction below a voxel's equidistant
    depth is its equivolume depth -- preserved by construction. LN2_LAYERS does
    the same at one depth (the fraction of each anchor's voxels below
    mid-depth); this does it at every depth, with sub-voxel columns.

    Why not the paired areas: FreeSurfer's pial vertex sits a median 1 mm (p90
    2.4 mm) sideways of its white partner, so "the column's pial area" is some
    other column's. On a folded synthetic with Bok ground truth and that much
    slide, paired areas did worse than equidistant (mean error 0.027 vs 0.022);
    this was 0.007.

    Columns are white-foot patches (a voxel's weight split over its foot face's
    vertices by barycentrics), depth histograms smoothed along the mesh until a
    column holds ~``column_voxels`` voxels: the measured optimum at both 0.25
    and 0.4 mm, where the rounds needed differed fourfold.
    """
    f = np.asarray(faces)[foot.face]
    b = np.minimum((rho * bins).astype(np.int64), bins - 1)
    hist = np.zeros(n_vertices * bins)
    for k in range(3):
        hist += np.bincount(f[:, k] * bins + b, foot.bary[:, k], n_vertices * bins)
    hist = hist.reshape(n_vertices, bins)
    per_vertex = rho.size / max(np.count_nonzero(hist.sum(1)), 1)
    n_iter = int(np.clip(np.ceil(column_voxels / per_vertex), 1, 500))
    hist = _smooth_on_mesh(hist, faces, n_iter)
    total = hist.sum(1, keepdims=True)
    cdf = np.concatenate([np.zeros((n_vertices, 1)), np.cumsum(hist, 1)], 1) / np.where(
        total > 0, total, 1.0
    )
    frac = rho * bins - b
    q = np.zeros(rho.size)
    for k in range(3):
        v = f[:, k]
        q += foot.bary[:, k] * (cdf[v, b] * (1 - frac) + cdf[v, b + 1] * frac)
    return np.clip(q, 0.0, 1.0).astype(np.float32)


def _neighbours(
    idx: np.ndarray, shape: tuple[int, int, int]
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Six face neighbours of flat C-order indices: yields ``(in_grid, neighbour)``.

    Flat-stride arithmetic, never an ``(N, 3)`` coordinate copy per direction:
    at 0.2 mm the ribbon is 70M voxels and those copies were most of the time.
    """
    strides = (shape[1] * shape[2], shape[2], 1)
    for axis in range(3):
        coord = (idx // strides[axis]) % shape[axis]
        for step in (-1, 1):
            ok = (coord > 0) if step < 0 else (coord < shape[axis] - 1)
            yield ok, idx + step * strides[axis]


def _mid_sheet(metric: np.ndarray, gm_idx: np.ndarray, shape) -> np.ndarray:
    """Positions in ``gm_idx`` of the one-voxel sheet where ``metric`` crosses 0.5.

    As LN2_LAYERS marks it: of two GM face neighbours on either side of 0.5,
    the one nearer 0.5 (both on a tie); a voxel at exactly 0.5 always.
    """
    # A dense lookup with NaN off GM answers "is the neighbour GM, and its
    # value" in one read.
    grid = np.full(int(np.prod(shape)), np.nan, np.float32)
    grid[gm_idx] = metric
    s = metric - np.float32(0.5)
    mark = s == 0
    for ok, nb in _neighbours(gm_idx, shape):
        t = np.where(ok, grid[np.where(ok, nb, 0)], np.nan) - np.float32(0.5)
        mark |= ~np.isnan(t) & (np.signbit(s) != np.signbit(t)) & (np.abs(s) <= np.abs(t))
    return np.flatnonzero(mark)


def cortical_depth_volumes(
    surfaces: list[RibbonSurfaces],
    affine: np.ndarray,
    shape: tuple[int, int, int],
    n_layers: int = 3,
    *,
    column_voxels: float = 64.0,
    thick_limit: float = 6.0,
    k: int = 24,
    device: torch.device | None = None,
    verbose: bool = False,
) -> DepthVolumes:
    """Rim, equidistant/equivolume metric + layers + midGM, and thickness.

    Parameters
    ----------
    surfaces : one entry per hemisphere.
    affine, shape : the output grid, voxel -> scanner RAS, ``(X, Y, Z)``.
    n_layers : LN2_LAYERS ``-nr_layers``.
    column_voxels : voxels per smoothed column for equivolume (volume_quantile_depth).
    thick_limit : GM voxels thicker than this (mm) are counted and located.
    k : candidate faces per voxel for the exact distance (see MeshDistance).
    """
    device = device or torch.device("cpu")
    shape = tuple(int(s) for s in shape)
    affine = np.asarray(affine, np.float64)
    t0 = time.time()

    def log(msg: str) -> None:
        if verbose:
            print(f"  [{time.time() - t0:6.1f}s] {msg}", flush=True)

    # 0 = not GM, h + 1 = GM of surfaces[h], -1 = GM by fill but medial wall.
    owner = np.zeros(shape, np.int8)
    wm = np.zeros(shape, bool)
    n_overlap = 0
    for h, s in enumerate(surfaces):
        in_white = winding_number(s.white, s.faces, affine, shape, device) > 0
        gm = winding_number(s.pial, s.faces, affine, shape, device) > 0
        gm &= ~in_white
        n_overlap += int((gm & (owner != 0)).sum())
        owner[gm & (owner == 0)] = h + 1
        wm |= in_white
        del in_white, gm
        log(f"{s.name}: filled white + pial")
    n_overlap += int((wm & (owner != 0)).sum())
    owner[wm] = 0  # Inside any white surface is WM.

    flat_owner = owner.reshape(-1)
    per_hemi = []
    for h, s in enumerate(surfaces):
        idx = np.flatnonzero(flat_owner == h + 1)
        if idx.size == 0:
            continue
        ijk = np.stack(np.unravel_index(idx, shape), 1).astype(np.float64)
        pts = ijk @ affine[:3, :3].T + affine[:3, 3]
        del ijk
        cw = MeshDistance(s.white, s.faces, k)(pts, device)
        cp = MeshDistance(s.pial, s.faces, k)(pts, device)
        del pts
        log(f"{s.name}: distances for {idx.size:,} GM voxels")
        keep = np.ones(idx.size, bool)
        if s.cortex is not None:
            keep = cw.interpolate(s.faces, s.cortex.astype(np.float32)) >= 0.5
            flat_owner[idx[~keep]] = -1
        dw, dp = cw.distance, cp.distance
        thick = dw + dp
        rho = np.clip(dw / np.maximum(thick, 1e-9), 0.0, 1.0)
        equivol = volume_quantile_depth(
            rho[keep], cw.subset(keep), s.faces, len(s.white), column_voxels
        )
        per_hemi.append((idx[keep], rho[keep], equivol, thick[keep]))
        del cw, cp, equivol
        log(f"{s.name}: depth metrics")

    if per_hemi:
        gm_idx = np.concatenate([p[0] for p in per_hemi])
        order = np.argsort(gm_idx, kind="stable")
        gm_idx = gm_idx[order]
        rho, equivol, thick = (
            np.concatenate([p[i] for p in per_hemi]).astype(np.float32)[order] for i in (1, 2, 3)
        )
    else:
        gm_idx = np.zeros(0, np.int64)
        rho = equivol = thick = np.zeros(0, np.float32)
    del per_hemi

    flat_wm = wm.reshape(-1)
    borders = []
    for ok, nb in _neighbours(gm_idx, shape):
        nb = nb[ok]
        borders.append(nb[flat_owner[nb] == 0])
    border_idx = np.unique(np.concatenate(borders)) if borders else np.zeros(0, np.int64)
    del borders
    log("rim borders")

    out = DepthVolumes(
        shape=shape,
        n_layers=int(n_layers),
        gm_idx=gm_idx,
        rho=rho,
        equivol=equivol,
        thick=thick,
        border_idx=border_idx,
        border_wm=flat_wm[border_idx],
        mid_equidist=_mid_sheet(rho, gm_idx, shape),
        mid_equivol=_mid_sheet(equivol, gm_idx, shape),
        n_thick=int((thick > thick_limit).sum()),
        thick_limit=float(thick_limit),
        n_medial=int((flat_owner == -1).sum()),
        n_overlap=n_overlap,
    )
    log("midGM sheets")
    return out


def regrid(
    affine: np.ndarray, shape: tuple[int, int, int], dxyz
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """The same field of view at voxel size ``dxyz`` (3dresample ``-dxyz``), ``(X, Y, Z)``."""
    from fastfuncstuff.processing.grid import resample_grid

    (nz, ny, nx), new_affine = resample_grid((shape[2], shape[1], shape[0]), affine, dxyz)
    return new_affine, (nx, ny, nz)


def crop_to_points(
    affine: np.ndarray, shape: tuple[int, int, int], points: np.ndarray, pad_mm: float = 1.0
) -> tuple[np.ndarray, tuple[int, int, int], tuple[slice, ...]]:
    """The sub-grid covering ``points`` plus ``pad_mm``, clipped to the grid.

    Whole voxels only, so the cropped grid's centres are a subset of the
    original's. Returns ``(affine, shape, slices into the original)``.
    """
    a = np.asarray(affine, np.float64)
    inv = np.linalg.inv(a)
    ijk = np.asarray(points, np.float64) @ inv[:3, :3].T + inv[:3, 3]
    pad = pad_mm / np.linalg.norm(a[:3, :3], axis=0)
    lo = np.maximum(np.floor(ijk.min(0) - pad), 0).astype(int)
    hi = np.minimum(np.ceil(ijk.max(0) + pad), np.asarray(shape) - 1).astype(int)
    if np.any(hi < lo):
        raise ValueError("the surfaces do not overlap the grid")
    out = a.copy()
    out[:3, 3] = a[:3, :3] @ lo + a[:3, 3]
    nx, ny, nz = (int(n) for n in hi - lo + 1)
    return out, (nx, ny, nz), tuple(slice(a, b + 1) for a, b in zip(lo, hi, strict=True))


def thick_report(out: DepthVolumes, affine: np.ndarray, top: int = 5) -> list[str]:
    """The over-thick warning as printable lines; empty when there is none."""
    if not out.n_thick:
        return []
    lines = [
        f"WARNING: {out.n_thick:,} GM voxels ({100 * out.n_thick / max(out.n_gm, 1):.2f}%) are "
        f"thicker than {out.thick_limit:g} mm -- check the pial surface there. "
        "Largest clusters (voxels @ scanner RAS mm):"
    ]
    for n, c in out.thick_clusters(affine, top):
        lines.append(f"    {n:7,d} @ ({c[0]:7.1f}, {c[1]:7.1f}, {c[2]:7.1f})")
    return lines


def export_depth_volumes(
    surfaces: list[RibbonSurfaces],
    master_affine: np.ndarray,
    master_shape: tuple[int, int, int],
    stem: str,
    ext: str = ".nii.gz",
    *,
    dxyz=None,
    autobox: bool = True,
    pad_mm: float = 1.0,
    n_layers: int = 3,
    column_voxels: float = 64.0,
    thick_limit: float = 6.0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple[DepthVolumes, np.ndarray, list[str]]:
    """Grid from a master, compute, and write ``{stem}_{tag}{ext}`` for every output.

    The whole of ffs_util_surf2layers below its argument parsing, shared with the
    viewer's export so the two cannot drift. Returns ``(volumes, grid affine,
    written paths)``.
    """
    from pathlib import Path

    from fastfuncstuff.io.afni import save_nifti

    affine, shape = np.asarray(master_affine, np.float64), tuple(master_shape)
    if dxyz is not None:
        affine, shape = regrid(affine, shape, dxyz)
    if autobox:
        pial = np.concatenate([s.pial for s in surfaces])
        affine, shape, _ = crop_to_points(affine, shape, pial, pad_mm)
    if verbose:
        vox = np.linalg.norm(affine[:3, :3], axis=0)
        print(
            f"{', '.join(s.name for s in surfaces)} onto {shape[0]}x{shape[1]}x{shape[2]} "
            f"at {' x '.join(f'{v:.3g}' for v in vox)} mm"
        )
    out = cortical_depth_volumes(
        surfaces,
        affine,
        shape,
        n_layers,
        column_voxels=column_voxels,
        thick_limit=thick_limit,
        device=device,
        verbose=verbose,
    )
    if out.n_gm == 0:
        raise ValueError("no GM voxel centres on this grid: is the master aligned to the anat?")
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    written = []
    for tag, vol in out.outputs():
        path = f"{stem}_{tag}{ext}"
        save_nifti(vol, path, affine=affine)
        written.append(path)
    return out, affine, written


__all__ = [
    "OUTPUT_TAGS",
    "DepthVolumes",
    "RibbonSurfaces",
    "cortical_depth_volumes",
    "crop_to_points",
    "export_depth_volumes",
    "regrid",
    "thick_report",
    "volume_quantile_depth",
]
