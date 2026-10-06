"""LayNii-compatible cortical depth volumes straight from white and pial meshes.

LN2_LAYERS takes a *rim* image -- GM plus one-voxel CSF/WM borders -- and
rebuilds the geometry from it: distances to the borders by voxel propagation,
equivolume factors from voxel-counted curvature, smoothed for hundreds of
iterations. A FreeSurfer subject already *has* that geometry, exactly: GM is
the solid between the white and pial meshes, the distances are point-to-mesh
distances, and the column's area at each end is the mesh's vertex area. So
every output here is computed from the meshes and only written onto voxels at
the end, at whatever grid is asked for.

Conventions follow LN2_LAYERS so the files drop into a LayNii workflow: the
metric is 0 at WM and 1 at CSF, layers are ``ceil(metric * N)`` (layer 1
deepest), borders are excluded from metric/layers/thickness, and midGM is a
one-voxel sheet at the voxel nearest each 0.5 crossing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from fastfuncstuff.surface.mesh import MeshTopology, vertex_areas
from fastfuncstuff.surface.profiles import volume_fraction
from fastfuncstuff.surface.voxelize import MeshDistance, winding_number

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
    """LN2_LAYERS's outputs on one grid, ``(X, Y, Z)``."""

    rim: np.ndarray
    metric_equidist: np.ndarray
    layers_equidist: np.ndarray
    mid_gm_equidist: np.ndarray
    metric_equivol: np.ndarray
    layers_equivol: np.ndarray
    mid_gm_equivol: np.ndarray
    thickness: np.ndarray
    #: GM voxels thicker than ``thick_limit`` -- real cortex is not; these are
    #: surface errors (pial through a vessel or the dura, a bad edit).
    n_thick: int = 0
    thick_limit: float = 6.0
    n_gm: int = 0
    #: GM-by-fill voxels dropped as medial wall.
    n_medial: int = 0
    #: GM voxels claimed by two hemispheres, or by one's GM and the other's WM.
    n_overlap: int = 0

    def outputs(self) -> dict[str, np.ndarray]:
        """LN2_LAYERS file tag -> volume."""
        return {tag: getattr(self, tag.replace("midGM", "mid_gm")) for tag in OUTPUT_TAGS}

    def thick_clusters(
        self, affine: np.ndarray, top: int | None = 5
    ) -> list[tuple[int, np.ndarray]]:
        """Connected clumps of over-thick GM, largest first: ``(n_voxels, centroid RAS)``.

        A clump is one place to look at the surfaces; a bare voxel count is not.
        """
        from scipy import ndimage

        lab, n = ndimage.label(self.thickness > self.thick_limit)
        if n == 0:
            return []
        sizes = np.bincount(lab.ravel())[1:]
        order = np.argsort(-sizes, kind="stable")[:top]
        centres = ndimage.center_of_mass(lab > 0, lab, (order + 1).tolist())
        a = np.asarray(affine, np.float64)
        return [
            (int(sizes[i]), a[:3, :3] @ np.asarray(c) + a[:3, 3])
            for i, c in zip(order, centres, strict=True)
        ]


def _smooth_vertex_values(values: np.ndarray, faces: np.ndarray, n_iter: int) -> np.ndarray:
    """``n_iter`` rounds of 1-ring averaging (self included).

    Barycentric vertex areas are noisy triangle to triangle; the equivolume
    correction depends on the white/pial ratio, which should vary on the scale
    of folding, not of the mesh.
    """
    if n_iter <= 0:
        return np.asarray(values, np.float64)
    topo = MeshTopology.from_faces(faces, values.shape[0])
    i, j = topo.edges[:, 0], topo.edges[:, 1]
    n = values.shape[0]
    adj = sp.coo_matrix(
        (np.ones(2 * i.size + n), (np.r_[i, j, np.arange(n)], np.r_[j, i, np.arange(n)])),
        shape=(n, n),
    ).tocsr()
    deg = np.asarray(adj.sum(1)).ravel()
    out = np.asarray(values, np.float64)
    for _ in range(n_iter):
        out = adj @ out / deg
    return out


def _neighbours(ijk: np.ndarray, shape: tuple[int, int, int]):
    """The six face neighbours of each voxel: yields ``(valid, flat_index)``."""
    for axis in range(3):
        for step in (-1, 1):
            nb = ijk.copy()
            nb[:, axis] += step
            valid = (nb[:, axis] >= 0) & (nb[:, axis] < shape[axis])
            flat = np.zeros(ijk.shape[0], np.int64)
            flat[valid] = np.ravel_multi_index(tuple(nb[valid].T), shape)
            yield valid, flat


def _mid_sheet(metric: np.ndarray, gm: np.ndarray, gm_idx: np.ndarray, shape) -> np.ndarray:
    """One-voxel sheet where ``metric`` crosses 0.5, as LN2_LAYERS marks it.

    Of two GM face neighbours on either side of 0.5, the one nearer 0.5 is
    marked (both on a tie); a voxel at exactly 0.5 always is.
    """
    flat_m = metric.reshape(-1)
    flat_gm = gm.reshape(-1)
    s = flat_m[gm_idx] - 0.5
    mark = s == 0
    ijk = np.stack(np.unravel_index(gm_idx, shape), 1)
    for valid, nb in _neighbours(ijk, shape):
        ok = valid.copy()
        ok[valid] = flat_gm[nb[valid]]
        t = flat_m[nb] - 0.5
        mark |= ok & (np.signbit(s) != np.signbit(t)) & (np.abs(s) <= np.abs(t))
    out = np.zeros(int(np.prod(shape)), np.int16)
    out[gm_idx[mark]] = 1
    return out.reshape(shape)


def cortical_depth_volumes(
    surfaces: list[RibbonSurfaces],
    affine: np.ndarray,
    shape: tuple[int, int, int],
    n_layers: int = 3,
    *,
    area_smooth: int = 10,
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
    area_smooth : 1-ring smoothing rounds on the vertex areas behind equivolume.
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
        log(f"{s.name}: filled white + pial")
    n_overlap += int((wm & (owner != 0)).sum())
    owner[wm] = 0  # Inside any white surface is WM.

    size = int(np.prod(shape))
    metric_ed = np.zeros(size, np.float32)
    metric_ev = np.zeros(size, np.float32)
    thickness = np.zeros(size, np.float32)
    flat_owner = owner.reshape(-1)
    for h, s in enumerate(surfaces):
        idx = np.flatnonzero(flat_owner == h + 1)
        if idx.size == 0:
            continue
        ijk = np.stack(np.unravel_index(idx, shape), 1).astype(np.float64)
        pts = ijk @ affine[:3, :3].T + affine[:3, 3]
        cw = MeshDistance(s.white, s.faces, k)(pts, device)
        cp = MeshDistance(s.pial, s.faces, k)(pts, device)
        log(f"{s.name}: distances for {idx.size:,} GM voxels")
        keep = np.ones(idx.size, bool)
        if s.cortex is not None:
            keep = cw.interpolate(s.faces, s.cortex.astype(np.float64)) >= 0.5
            flat_owner[idx[~keep]] = -1
        dw, dp = cw.distance.astype(np.float64), cp.distance.astype(np.float64)
        thick = dw + dp
        rho = np.clip(dw / np.maximum(thick, 1e-9), 0.0, 1.0)
        aw = _smooth_vertex_values(
            vertex_areas(s.white, s.faces, len(s.white)), s.faces, area_smooth
        )
        ap = _smooth_vertex_values(vertex_areas(s.pial, s.faces, len(s.pial)), s.faces, area_smooth)
        # Each closest point names a column; trust the one on the nearer surface
        # more, so the factor varies smoothly across the ribbon.
        alpha_w = volume_fraction(rho, cw.interpolate(s.faces, aw), cw.interpolate(s.faces, ap))
        alpha_p = volume_fraction(rho, cp.interpolate(s.faces, aw), cp.interpolate(s.faces, ap))
        equivol = np.clip((1.0 - rho) * alpha_w + rho * alpha_p, 0.0, 1.0)
        sel = idx[keep]
        metric_ed[sel] = rho[keep]
        metric_ev[sel] = equivol[keep]
        thickness[sel] = thick[keep]

    gm_idx = np.flatnonzero(flat_owner > 0)
    gm_mask = flat_owner.reshape(shape) > 0

    rim = np.zeros(size, np.int16)
    rim[gm_idx] = RIM_GM
    ijk = np.stack(np.unravel_index(gm_idx, shape), 1)
    flat_wm = wm.reshape(-1)
    for valid, nb in _neighbours(ijk, shape):
        nb = nb[valid]
        border = nb[flat_owner[nb] == 0]
        rim[border] = np.where(flat_wm[border], RIM_WM, RIM_CSF)

    def layers(metric: np.ndarray) -> np.ndarray:
        out = np.zeros(size, np.int16)
        out[gm_idx] = np.clip(np.ceil(metric[gm_idx] * n_layers), 1, n_layers)
        return out.reshape(shape)

    metric_ed = metric_ed.reshape(shape)
    metric_ev = metric_ev.reshape(shape)
    out = DepthVolumes(
        rim=rim.reshape(shape),
        metric_equidist=metric_ed,
        layers_equidist=layers(metric_ed.reshape(-1)),
        mid_gm_equidist=_mid_sheet(metric_ed, gm_mask, gm_idx, shape),
        metric_equivol=metric_ev,
        layers_equivol=layers(metric_ev.reshape(-1)),
        mid_gm_equivol=_mid_sheet(metric_ev, gm_mask, gm_idx, shape),
        thickness=thickness.reshape(shape),
        n_thick=int((thickness[gm_idx] > thick_limit).sum()),
        thick_limit=float(thick_limit),
        n_gm=int(gm_idx.size),
        n_medial=int((flat_owner == -1).sum()),
        n_overlap=n_overlap,
    )
    log("rim, layers, midGM")
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


__all__ = [
    "OUTPUT_TAGS",
    "DepthVolumes",
    "RibbonSurfaces",
    "cortical_depth_volumes",
    "crop_to_points",
    "regrid",
]
