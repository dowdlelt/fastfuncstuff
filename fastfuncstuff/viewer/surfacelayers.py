"""Surface results as layers: one stack, one set of controls, two ways to draw.

A per-vertex result (a ``.func.gii`` bucket from ``ffs_reml`` on a surface) is loaded
with the same LOAD as a volume and becomes an ordinary :class:`~viewer.layers.Layer`.
Its colormap, range, threshold (typed or as a p), OLAY / THR sub-bricks, sign and
alpha are the layer's, set by the same commands. What differs is where its values
live, and so how each view reads them:

* **slices**: the displayed sub-brick is painted into the cortical ribbon on the
  layer's grid through the voxel <-> vertex map (:mod:`surface.ribbon`), on demand,
  one volume at a time (the store's lazy entry). Slicing, readouts and every volume
  tool then work unchanged.
* **the 3-D window**: the vertices are coloured directly, through the same compose
  functions (:func:`vertex_rgba`), so a threshold dragged on the slices moves the
  surface too.

**Surface is king for what is computed.** Clusters are found on the mesh (edges,
areas in mm^2 on the midthickness) and corrected with the bucket's own SurfClustSim
table, then painted into voxels -- so a surface cluster is a volume ROI by
construction, with each voxel's cortical depth on the ribbon map. That is the
mapping the depth/laminar step needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fastfuncstuff.surface.ribbon import RibbonMap
from fastfuncstuff.surface.statmap import SurfaceData, load_surface_data

__all__ = [
    "SurfaceLayerData",
    "is_surface_path",
    "open_surface_parts",
    "surface_clusterize",
    "vertex_rgba",
]

_HEMI_TOKEN = re.compile(r"(?<=[._\-])(lh|rh)(?=[._\-])")


def is_surface_path(path: str | Path) -> bool:
    name = str(path).split("[")[0].lower()
    return name.endswith(".gii") and not name.endswith(".surf.gii")


@dataclass
class SurfaceLayerData:
    """A surface layer's values per hemisphere, and its ribbon maps on the layer grid."""

    parts: dict[str, SurfaceData]
    maps: dict[str, RibbonMap]
    #: The last clusterize, per hemisphere: (labels per vertex, dropped per vertex),
    #: so the 3-D window can show exactly the clusters the table lists.
    clusters: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def hemis(self) -> list[str]:
        return list(self.parts)

    @property
    def n_volumes(self) -> int:
        return next(iter(self.parts.values())).values.shape[1]

    def paint(self, index: int) -> np.ndarray:
        """Sub-brick ``index`` on the grid: every hemisphere's ribbon, 0 elsewhere."""
        rm = next(iter(self.maps.values()))
        vol = np.zeros(rm.shape, np.float32)
        for hemi, data in self.parts.items():
            k = max(0, min(int(index), data.values.shape[1] - 1))
            self.maps[hemi].paint(data.values[:, k], out=vol)
        return vol


def sibling_path(path: str | Path) -> Path | None:
    """The other hemisphere's file by name (``.lh.`` <-> ``.rh.``), when there is one."""
    p = Path(path)
    m = list(_HEMI_TOKEN.finditer(p.name))
    if not m:
        return None
    hit = m[-1]
    other = "rh" if hit.group(1) == "lh" else "lh"
    return p.with_name(p.name[: hit.start()] + other + p.name[hit.end() :])


def open_surface_parts(path: str | Path, match) -> dict[str, SurfaceData]:
    """The file and its sibling hemisphere, each matched to a loaded mesh.

    ``match(data, hint) -> hemi`` is the surface store's fingerprint match; the
    sibling is joined only if it matches the *other* loaded hemisphere.
    """
    from fastfuncstuff.io.freesurfer import infer_label

    data = load_surface_data(path)
    hemi = match(data, infer_label(str(path))[0] or "")
    parts = {hemi: data}
    sib = sibling_path(path)
    if sib is not None and sib.exists():
        try:
            other = load_surface_data(sib)
            h2 = match(other, infer_label(str(sib))[0] or "")
        except (ValueError, OSError):
            h2 = None
        if h2 is not None and h2 not in parts and other.values.shape[1] == data.values.shape[1]:
            parts[h2] = other
    return dict(sorted(parts.items()))


def vertex_rgba(layer, values: np.ndarray, stat: np.ndarray, keep: np.ndarray | None, device=None):
    """``(V, 4)`` uint8 colours of one hemisphere: :func:`layer_rgba` on its vertices."""
    return layer_rgba(layer, values, stat, keep, device)


def layer_rgba(
    layer, values: np.ndarray, stat: np.ndarray, keep: np.ndarray | None = None, device=None
):
    """``(..., 4)`` uint8 colours of any array of a layer's values, as
    :func:`viewer.compose.render_plane` colours a plane: the layer's LUT, range,
    sign mode, threshold and alpha ramp, then any cut (``keep``). Values exactly 0
    or NaN (off cortex, off a patch) stay clear. Shared by the surface window's
    vertices and CHEDI's overlays, so a threshold reads the same everywhere."""
    import torch

    from fastfuncstuff.viewer.colormap import apply_colormap, threshold_alpha
    from fastfuncstuff.viewer.compose import cached_lut

    dev = device or torch.device("cpu")
    shape = np.shape(values)
    raw = np.asarray(values, np.float32).reshape(-1)
    finite = np.isfinite(raw)
    v = torch.as_tensor(np.where(finite, raw, 0.0), device=dev)
    s = torch.as_tensor(np.nan_to_num(np.asarray(stat, np.float32).reshape(-1)), device=dev)
    lo = layer.range_lo if layer.range_lo is not None else 0.0
    hi = layer.range_hi if layer.range_hi is not None else 1.0
    rgb = apply_colormap(
        v,
        lut=cached_lut(layer.colormap, dev, reverse=layer.colormap_reversed),
        lo=float(lo),
        hi=float(hi),
        sign_mode=layer.sign_mode,
        n_panes=layer.n_panes,
    )
    alpha = threshold_alpha(s, layer.threshold, mode=layer.alpha_mode, sign_mode=layer.sign_mode)
    alpha = alpha * float(layer.opacity) * (v != 0).to(alpha.dtype)
    alpha = alpha * torch.as_tensor(finite, device=dev).to(alpha.dtype)
    if keep is not None:
        cut = np.asarray(keep, bool).reshape(-1)
        alpha = alpha * torch.as_tensor(cut, device=dev).to(alpha.dtype)
    out = np.zeros((v.shape[0], 4), np.uint8)
    out[:, :3] = np.clip(np.round(rgb.cpu().numpy() * 255), 0, 255).astype(np.uint8)
    out[:, 3] = np.clip(np.round(alpha.cpu().numpy() * 255), 0, 255).astype(np.uint8)
    return out.reshape(*shape, 4)


def _cluster_alpha(table: dict, pthr: float | None, area: float) -> float | None:
    """Corrected alpha of a cluster of ``area`` mm^2 from a SurfClustSim table row."""
    if table is None or pthr is None:
        return None
    p = np.asarray(table["pthr"], np.float64)
    a = np.asarray(table["athr"], np.float64)
    t = np.asarray(table["area_mm2"], np.float64)
    if not (p.min() <= pthr <= p.max()):
        return None
    order = np.argsort(p)
    row = np.array(
        [np.interp(np.log(pthr), np.log(p[order]), np.log(t[order, j])) for j in range(len(a))]
    )
    need = np.exp(row)  # area needed at each alpha (larger alpha, smaller area)
    ao = np.argsort(need)  # interpolate log alpha against log area
    return float(np.exp(np.interp(np.log(max(area, 1e-9)), np.log(need[ao]), np.log(a[ao]))))


def surface_clusterize(layer, sld: SurfaceLayerData, hemis_geometry, *, min_voxels: int = 1,
                       pthr: float | None = None):  # fmt: skip
    """Cluster a surface layer at its own threshold, on the mesh; a volume ClusterTable.

    ``hemis_geometry[hemi] = (white, pial, faces)``. Clusters are connected over mesh
    edges (positive and negative apart for a two-sided map), measured in mm^2 on the
    midthickness, given a corrected alpha from the bucket's SurfClustSim table, and
    painted into the layer grid through the ribbon map -- so ``labels`` and
    ``dropped`` are voxels, as every consumer of a cluster table expects. A cluster is
    "small" by its voxel count on that grid, the cluster window's own unit.
    """
    from fastfuncstuff.surface.mesh import vertex_areas
    from fastfuncstuff.surface.statmap import surviving_clusters
    from fastfuncstuff.viewer.clusters import SIDEDNESS, Cluster, ClusterTable
    from fastfuncstuff.viewer.layers import SignMode

    sided = SIDEDNESS[layer.sign_mode]
    rm0 = next(iter(sld.maps.values()))
    shape, aff = rm0.shape, rm0.affine
    voxel_mm3 = float(abs(np.linalg.det(aff[:3, :3])))
    inv = np.linalg.inv(aff)
    labels = np.zeros(shape, np.int32)
    dropped = np.zeros(shape, bool)
    found = []
    sld.clusters = {}
    alphas_seen = []
    for hemi, data in sld.parts.items():
        white, pial, faces = hemis_geometry[hemi]
        mid = 0.5 * (np.asarray(white, np.float64) + np.asarray(pial, np.float64))
        area = vertex_areas(mid, faces)
        vals = data.values[:, layer.volume_index]
        stat = data.values[:, layer.threshold_brick]
        if layer.sign_mode is SignMode.NEG:
            stat = -stat
        one = layer.sign_mode is not SignMode.BOTH
        _, lab = surviving_clusters(
            stat, float(layer.threshold), faces, area, None, sided=sided, one_sided_positive=one
        )
        rm = sld.maps[hemi]
        counts = np.bincount(lab[rm.vertex], minlength=int(lab.max()) + 1) if lab.max() else []
        vdropped = np.zeros(len(vals), bool)
        table = data.tables.get(sided)
        for c in range(1, int(lab.max()) + 1):
            ids = np.flatnonzero(lab == c)
            nvox = int(counts[c]) if c < len(counts) else 0
            if nvox < max(min_voxels, 1):
                vdropped[ids] = True
                continue
            a_mm2 = float(area[ids].sum())
            peak_v = ids[np.argmax(np.abs(vals[ids]))]
            w = np.abs(vals[ids])
            com = (mid[ids] * w[:, None]).sum(0) / max(w.sum(), 1e-12)
            alpha = _cluster_alpha(table, pthr, a_mm2)
            if alpha is not None:
                alphas_seen.append(alpha)
            found.append((nvox, hemi, ids, a_mm2, peak_v, mid[peak_v], com, alpha,
                          float(vals[peak_v]), float(vals[ids].mean())))  # fmt: skip
        dropped.reshape(-1)[rm.voxels_of(vdropped)] = True
        sld.clusters[hemi] = (lab, vdropped)
    clusters = []
    for i, (nvox, hemi, ids, a_mm2, _pv, peak_xyz, com, alpha, peak, mean) in enumerate(
        sorted(found, key=lambda x: -x[3]), start=1
    ):
        labels.reshape(-1)[sld.maps[hemi].voxels_of(ids)] = i
        pk = tuple(int(round(x)) for x in peak_xyz @ inv[:3, :3].T + inv[:3, 3])
        cm = tuple(float(x) for x in com @ inv[:3, :3].T + inv[:3, 3])
        clusters.append(
            Cluster(
                i,
                nvox,
                nvox * voxel_mm3,
                peak,
                pk,
                tuple(float(x) for x in peak_xyz),
                cm,
                tuple(float(x) for x in com),
                mean,
                alpha,
                area_mm2=a_mm2,
            )  # fmt: skip
        )
    tables = [d.tables.get(sided) for d in sld.parts.values()]
    note = "" if any(tables) else "no SurfClustSim table in this bucket: no corrected alpha"
    if pthr is None and any(tables):
        note = "the threshold sub-brick is not a statistic: no corrected alpha"
    arange = None
    if any(tables) and pthr is not None:
        t = next(x for x in tables if x)
        arange = (float(min(t["athr"])), float(max(t["athr"])))
    return ClusterTable(
        tuple(clusters), labels, float(layer.threshold), sided, 0, int(min_voxels), pthr,
        arange, note, dropped,
    )  # fmt: skip
