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
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from fastfuncstuff.memory import get_available_memory
from fastfuncstuff.surface.mesh import MeshTopology, vertex_areas
from fastfuncstuff.surface.voxelize import MeshDistance, winding_number
from fastfuncstuff.utils import linalg_device

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
    #: tkregister -> scanner RAS of a FreeSurfer mesh (None for GIfTI), so a
    #: ``.label``'s tkregister coordinates can be checked against this mesh.
    tkr_to_scanner: np.ndarray | None = None

    @classmethod
    def from_hemisphere(cls, hemi, white: str = "white", pial: str = "pial") -> RibbonSurfaces:
        """From a loaded :class:`~fastfuncstuff.io.freesurfer.Hemisphere`, edits included."""
        for state in (white, pial):
            if state not in hemi.states:
                raise KeyError(f"{hemi.name} has no '{state}' surface")
        return cls(
            hemi.name, hemi.states[white], hemi.states[pial], hemi.faces, hemi.cortex,
            hemi.tkr_to_scanner,
        )  # fmt: skip


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
    #: GM voxels dropped because their column is outside the region asked for.
    n_outside: int = 0
    #: ``(hemisphere index, rounds)`` of 1-ring mesh pooling the equivolume CDF used:
    #: how far along the surface each column's volume was measured.
    smooth_rounds: tuple[tuple[int, int], ...] = ()
    #: GM voxels claimed by two hemispheres, or by one's GM and the other's WM.
    n_overlap: int = 0
    #: With ``owners=True``: each GM voxel's column -- the index into ``surfaces``
    #: and the vertex owning it, by :mod:`surface.ribbon`'s rule (heaviest corner of
    #: the closest midthickness point), so a vertex set maps to the same voxels here
    #: as in the projection and the viewer.
    owner_hemi: np.ndarray | None = None
    owner_vertex: np.ndarray | None = None

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

    def gm_mask(self, value: int = 1) -> np.ndarray:
        """The GM voxels as an int16 volume holding ``value``: the region, cropped."""
        return self._dense(np.int16(value), dtype=np.int16)

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
    rho: torch.Tensor,
    face: torch.Tensor,
    bary: torch.Tensor,
    faces: np.ndarray,
    n_vertices: int,
    column_voxels: float = 64.0,
    bins: int = 64,
    rounds: int | None = None,
) -> tuple[torch.Tensor, int]:
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
    vertices by barycentrics: ``face``/``bary`` of its closest white point),
    depth histograms smoothed along the mesh until a column holds
    ~``column_voxels`` voxels: the measured optimum at both 0.25 and 0.4 mm,
    where the rounds needed differed fourfold.

    Returns the depths and the rounds of mesh smoothing they took -- how far along
    the surface a column's volume was pooled, which is how much cortex beyond a
    region of interest its equivolume depths depend on. ``rounds`` sets that count
    instead of deriving it from the voxels per column seen here.
    """
    cdf, rounds = column_depth_cdf(
        rho, face, bary, faces, n_vertices, column_voxels, bins, rounds=rounds
    )
    return column_quantile(cdf, rho, face, bary, faces), rounds


def column_smooth_rounds(n_samples: int, n_columns: int, column_voxels: float) -> int:
    """Rounds of 1-ring pooling for a column to hold ~``column_voxels`` samples.

    No samples pools once, not the 500-round maximum that "0 per column" asks for:
    there is nothing to pool, and the rounds are paid for whether or not there is.
    """
    if int(n_samples) == 0:
        return 1
    return _rounds_for(n_samples / max(int(n_columns), 1), column_voxels)


def _rounds_for(per_vertex: float, column_voxels: float) -> int:
    return int(np.clip(np.ceil(column_voxels / max(per_vertex, 1e-9)), 1, 500))


def column_depth_cdf(
    rho: torch.Tensor,
    face: torch.Tensor,
    bary: torch.Tensor,
    faces: np.ndarray,
    n_vertices: int,
    column_voxels: float = 64.0,
    bins: int = 64,
    weight: torch.Tensor | None = None,
    rho_hi: torch.Tensor | None = None,
    rounds: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Each column's volume CDF over equidistant depth, ``(V, bins + 1)``, and
    the rounds of mesh smoothing used.

    The first half of :func:`volume_quantile_depth`, for samples of the ribbon that
    are not equal-volume voxels: ``weight`` is each sample's volume, and with
    ``rho_hi`` a sample is a piece of ribbon spread evenly over depths ``[rho,
    rho_hi]``. Samples that sit at a few discrete depths (cells of a tiling) need
    that: as points they make every column's CDF a staircase. ``rounds`` of 1-ring mesh
    smoothing pool each column with its neighbours (default: enough that a column
    holds ~``column_voxels`` samples, for counting noise -- samples, not volume).
    """
    device = linalg_device(rho.device)  # float64 throughout; Metal has none
    rho, face, bary = rho.to(device), face.to(device), bary.to(device)
    weight = None if weight is None else weight.to(device)
    rho_hi = None if rho_hi is None else rho_hi.to(device)
    f = torch.as_tensor(faces, device=device)[face]  # (N, 3) vertex ids
    w = bary.double() if weight is None else bary.double() * weight.double()[:, None]
    hist = torch.zeros(n_vertices * bins, dtype=torch.float64, device=device)
    if rho_hi is None:
        pieces = [(f, torch.clamp((rho * bins).long(), max=bins - 1), w)]
    else:
        lo = torch.minimum(rho, rho_hi).double() * bins
        hi = torch.maximum(rho, rho_hi).double() * bins
        span = (hi - lo).clamp_min(1e-12)
        first = lo.floor().clamp(max=bins - 1)
        pieces = []
        for off in range(int((hi.ceil() - first).max().item()) + 1 if rho.numel() else 0):
            b = first + off
            share = (torch.minimum(hi, b + 1) - torch.maximum(lo, b)).clamp_min(0) / span
            share = torch.where(hi - lo < 1e-9, float(off == 0), share)
            ok = (share > 0) & (b < bins)
            pieces.append((f[ok], b[ok].long(), w[ok] * share[ok, None]))
    for fv, b, wb in pieces:
        for k in range(3):
            hist.index_add_(0, fv[:, k] * bins + b, wb[:, k])
    hist = hist.view(n_vertices, bins)
    n_iter = (
        column_smooth_rounds(rho.numel(), int((hist.sum(1) > 0).sum()), column_voxels)
        if rounds is None
        else int(rounds)
    )
    # (V, bins) is small; the sparse smoothing stays with scipy.
    smooth = _smooth_on_mesh(hist.cpu().numpy(), faces, n_iter)
    total = smooth.sum(1, keepdims=True)
    cdf = np.concatenate([np.zeros((n_vertices, 1)), np.cumsum(smooth, 1)], 1)
    out = torch.as_tensor(cdf / np.where(total > 0, total, 1.0), dtype=torch.float32, device=device)
    return out, n_iter


def column_quantile(
    cdf: torch.Tensor, rho: torch.Tensor, face: torch.Tensor, bary: torch.Tensor, faces: np.ndarray
) -> torch.Tensor:
    """Equivolume depth of points at equidistant depth ``rho`` with white feet
    ``face``/``bary``: their column CDF (:func:`column_depth_cdf`) read there."""
    home = rho.device
    rho, face, bary = rho.to(cdf.device), face.to(cdf.device), bary.to(cdf.device)
    bins = cdf.shape[1] - 1
    f = torch.as_tensor(faces, device=rho.device)[face]
    b = torch.clamp((rho * bins).long(), max=bins - 1)
    frac = rho * bins - b
    q = torch.zeros_like(rho)
    for k in range(3):
        v = f[:, k]
        q += bary[:, k] * (cdf[v, b] * (1 - frac) + cdf[v, b + 1] * frac)
    return q.clamp_(0.0, 1.0).to(home)


def _neighbours(idx, shape: tuple[int, int, int]) -> Iterator[tuple]:
    """Six face neighbours of flat C-order indices: yields ``(in_grid, neighbour)``.

    Flat-stride arithmetic on whatever array type comes in, never an ``(N, 3)``
    coordinate copy per direction: at 0.2 mm the ribbon is 70M voxels and those
    copies were most of the time.
    """
    strides = (shape[1] * shape[2], shape[2], 1)
    for axis in range(3):
        coord = (idx // strides[axis]) % shape[axis]
        for step in (-1, 1):
            ok = (coord > 0) if step < 0 else (coord < shape[axis] - 1)
            yield ok, idx + step * strides[axis]


def _chunks(n: int, device: torch.device, bytes_per_item: int) -> Iterator[slice]:
    step = max(1 << 16, get_available_memory(device, empty_cache=False) // bytes_per_item)
    for s in range(0, n, step):
        yield slice(s, min(n, s + step))


def _mid_sheet(metric: torch.Tensor, gm_idx: torch.Tensor, shape) -> torch.Tensor:
    """Positions in ``gm_idx`` of the one-voxel sheet where ``metric`` crosses 0.5.

    As LN2_LAYERS marks it: of two GM face neighbours on either side of 0.5,
    the one nearer 0.5 (both on a tie); a voxel at exactly 0.5 always.
    """
    size = int(np.prod(shape))
    # A dense lookup with NaN off GM answers "is the neighbour GM, and its
    # value" in one read.
    grid = torch.full((size,), float("nan"), dtype=torch.float32, device=metric.device)
    grid[gm_idx] = metric
    mark = torch.zeros(gm_idx.numel(), dtype=torch.bool, device=metric.device)
    for sl in _chunks(gm_idx.numel(), metric.device, 96):
        s = metric[sl] - 0.5
        m = s == 0
        for ok, nb in _neighbours(gm_idx[sl], shape):
            t = torch.where(ok, grid[nb.clamp(0, size - 1)], float("nan")) - 0.5
            m |= ~torch.isnan(t) & (torch.signbit(s) != torch.signbit(t)) & (s.abs() <= t.abs())
        mark[sl] = m
    return torch.nonzero(mark).squeeze(1)


def cortical_depth_volumes(
    surfaces: list[RibbonSurfaces],
    affine: np.ndarray,
    shape: tuple[int, int, int],
    n_layers: int = 3,
    *,
    column_voxels: float = 64.0,
    thick_limit: float = 6.0,
    k: int = 24,
    owners: bool = False,
    columns: list[np.ndarray] | None = None,
    column_rounds: int | None = None,
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
    k : candidate faces per voxel for the CPU distance search (see MeshDistance).
    owners : also record each GM voxel's owning vertex (one more distance search).
    columns : one vertex mask per hemisphere -- keep only the voxels those cortical
        columns own (implies ``owners``, by whose rule membership is decided). The
        depths are measured first and the voxels dropped after, so what is kept is
        exactly what an unrestricted run would have put there, as long as the grid
        holds enough cortex around the region for the equivolume CDF's pooling (see
        :func:`export_roi_depth_volumes`). Dropped voxels are not rim borders, as
        the medial wall's are not -- the patch's lateral edge is open rather than
        labelled a CSF border it is not.
    column_rounds : rounds of 1-ring pooling for the equivolume CDF, instead of the
        count derived from the voxels per column measured here. Cortical thickness
        and vertex density vary over a hemisphere, so a region's own measurement
        differs a little from a whole hemisphere's; pass the ``smooth_rounds`` of a
        whole-hemisphere run to reproduce its depths exactly.
    """
    device = device or torch.device("cpu")
    owners = owners or columns is not None
    if columns is not None and len(columns) != len(surfaces):
        raise ValueError(f"columns: {len(columns)} masks for {len(surfaces)} surfaces")
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
    # ~4 voxels a side: measured fastest at 0.35 mm (64 points fill a program).
    brick_mm = 4.0 * float(np.abs(np.linalg.det(affine[:3, :3])) ** (1 / 3))
    per_hemi = []
    rounds_used: list[tuple[int, int]] = []
    for h, s in enumerate(surfaces):
        idx = np.flatnonzero(flat_owner == h + 1)
        if idx.size == 0:
            continue
        if columns is not None and not np.asarray(columns[h], bool).any():
            # No column of the region is on this hemisphere at all: its voxels are
            # here only because the other one's crop reached over the midline, and
            # every one of them is about to be dropped. Don't measure them.
            flat_owner[idx] = -2
            log(f"{s.name}: not in the region, {idx.size:,} GM voxels dropped unmeasured")
            continue
        ijk = np.stack(np.unravel_index(idx, shape), 1).astype(np.float64)
        pts = ijk @ affine[:3, :3].T + affine[:3, 3]
        del ijk
        cw = MeshDistance(s.white, s.faces, k)(pts, device, brick_mm=brick_mm)
        cp = MeshDistance(s.pial, s.faces, k)(pts, device, brick_mm=brick_mm)
        del pts
        log(f"{s.name}: distances for {idx.size:,} GM voxels")
        face = torch.as_tensor(cw.face, device=device)
        bary = torch.as_tensor(cw.bary, device=device)
        dw = torch.as_tensor(cw.distance, device=device)
        thick = dw + torch.as_tensor(cp.distance, device=device)
        del cw, cp
        rho = (dw / thick.clamp_min(1e-9)).clamp_(0.0, 1.0)
        del dw
        idx_t = torch.as_tensor(idx, device=device)
        if owners:
            ijk = np.stack(np.unravel_index(idx, shape), 1).astype(np.float64)
            pts = ijk @ affine[:3, :3].T + affine[:3, 3]
            near = MeshDistance(0.5 * (s.white + s.pial), s.faces, k)(
                pts, device, brick_mm=brick_mm
            )
            del ijk, pts
            vertex = np.asarray(s.faces, np.int64)[near.face, np.argmax(near.bary, axis=1)]
            owner_v = torch.as_tensor(vertex, device=device)
            del near, vertex
        else:
            owner_v = torch.zeros(idx.size, dtype=torch.long, device=device)
        if s.cortex is not None:
            cortex = torch.as_tensor(s.cortex, dtype=torch.float32, device=device)
            vf = torch.as_tensor(s.faces, device=device)[face]
            keep = (cortex[vf] * bary).sum(1) >= 0.5
            del vf
            flat_owner[idx[~keep.cpu().numpy()]] = -1
            idx_t, rho, thick, face, bary, owner_v = (
                a[keep] for a in (idx_t, rho, thick, face, bary, owner_v)
            )
        keep_t = None
        rounds = column_rounds
        if columns is not None:
            keep_t = torch.as_tensor(np.asarray(columns[h], bool), device=device)[owner_v]
            if not bool(keep_t.any()):
                # None of this hemisphere is in the region -- the other one's crop
                # reaches across the midline. Nothing here survives, so measure nothing.
                flat_owner[np.asarray(idx_t.cpu())] = -2
                log(f"{s.name}: no column of the region on this grid")
                continue
            if rounds is None:
                # Pool as a whole-hemisphere run would. The surround exists so the
                # region's own columns are complete, but the surround's outermost
                # columns are cut off by the crop and read as sparse; measuring the
                # voxels per column over the region's own columns -- the same
                # estimator, over the columns this grid actually holds whole --
                # keeps the pooling independent of how much surround was taken.
                vf = torch.as_tensor(s.faces, device=device)[face[keep_t]]
                touched = vf[bary[keep_t] > 0]
                rounds = column_smooth_rounds(
                    int(keep_t.sum()), int(torch.unique(touched).numel()), column_voxels
                )
        equivol, rounds = volume_quantile_depth(
            rho, face, bary, s.faces, len(s.white), column_voxels, rounds=rounds
        )
        rounds_used.append((h, rounds))
        del face, bary
        if keep_t is not None:
            flat_owner[np.asarray(idx_t.cpu())[~keep_t.cpu().numpy()]] = -2
            idx_t, rho, equivol, thick, owner_v = (
                a[keep_t] for a in (idx_t, rho, equivol, thick, owner_v)
            )
        owner_h = torch.full_like(owner_v, h, dtype=torch.int8)
        per_hemi.append((idx_t, rho, equivol, thick, owner_h, owner_v))
        log(f"{s.name}: depth metrics ({rounds} pooling rounds)")

    if per_hemi:
        gm_idx = torch.cat([p[0] for p in per_hemi])
        gm_idx, order = torch.sort(gm_idx)
        rho, equivol, thick, owner_h, owner_v = (
            torch.cat([p[i] for p in per_hemi])[order] for i in (1, 2, 3, 4, 5)
        )
        del order
    else:
        gm_idx = torch.zeros(0, dtype=torch.long, device=device)
        rho = equivol = thick = torch.zeros(0, dtype=torch.float32, device=device)
        owner_h = torch.zeros(0, dtype=torch.int8, device=device)
        owner_v = torch.zeros(0, dtype=torch.long, device=device)
    del per_hemi

    owner_t = torch.as_tensor(flat_owner, device=device)
    borders = []
    for sl in _chunks(gm_idx.numel(), device, 64):
        for ok, nb in _neighbours(gm_idx[sl], shape):
            nb = nb[ok]
            borders.append(nb[owner_t[nb] == 0])
    del owner_t
    border_idx = torch.unique(torch.cat(borders)) if borders else gm_idx[:0]
    del borders
    border_wm = torch.as_tensor(wm.reshape(-1), device=device)[border_idx]
    log("rim borders")
    mid_ed, mid_ev = _mid_sheet(rho, gm_idx, shape), _mid_sheet(equivol, gm_idx, shape)
    log("midGM sheets")

    def host(t: torch.Tensor) -> np.ndarray:
        return t.cpu().numpy()

    return DepthVolumes(
        shape=shape,
        n_layers=int(n_layers),
        gm_idx=host(gm_idx),
        rho=host(rho),
        equivol=host(equivol),
        thick=host(thick),
        border_idx=host(border_idx),
        border_wm=host(border_wm),
        mid_equidist=host(mid_ed),
        mid_equivol=host(mid_ev),
        n_thick=int((thick > thick_limit).sum()),
        thick_limit=float(thick_limit),
        n_medial=int((flat_owner == -1).sum()),
        n_outside=int((flat_owner == -2).sum()),
        smooth_rounds=tuple(rounds_used),
        n_overlap=n_overlap,
        owner_hemi=host(owner_h) if owners else None,
        owner_vertex=host(owner_v) if owners else None,
    )


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
    column_rounds: int | None = None,
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
        column_rounds=column_rounds,
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


@dataclass
class RoiDepthResult:
    """One region's depth volumes on its own cropped grid."""

    name: str
    value: int
    out: DepthVolumes
    affine: np.ndarray
    written: list[str]
    #: Vertices the region covers, and after filling and growing, per hemisphere.
    n_own: int = 0
    n_filled: int = 0
    n_grown: int = 0
    surround_mm: float = 0.0
    #: How far along the surface the equivolume pooling reached, mm. Beyond
    #: ``surround_mm`` the pooling ran into the edge of the cropped grid.
    pooled_mm: float = 0.0

    @property
    def enough_surround(self) -> bool:
        return self.pooled_mm <= self.surround_mm + 1e-6


def estimate_column_reach(
    surfaces: list[RibbonSurfaces],
    affine: np.ndarray,
    column_voxels: float = 64.0,
    members: list[np.ndarray] | None = None,
    slack_rounds: int = 1,
) -> tuple[int, float]:
    """``(pooling rounds, how far they reach along the surface in mm)``, predicted.

    The equivolume CDF pools each column with its neighbours until a column holds
    ~``column_voxels`` voxels, so how far it reaches is set by how many voxels a
    column holds -- its thickness times its vertex's area over the voxel volume --
    and that is known from the meshes and the grid before anything is computed. Used
    to size the surround a region needs (:func:`export_roi_depth_volumes`); the run
    then reports what it actually pooled, which is the number that counts.

    ``slack_rounds`` is added to the prediction because the thing predicted is a
    small integer: the CDF counts a voxel into each of its foot face's corners, so
    it sees a little less per column than a vertex's own area says and rounds up one
    more time than this does -- measured on a synthetic shell, 3 predicted against 4
    taken. A round of slack is a millimetre of surround; being a round short is a
    wrong depth at the region's edge.
    """
    rounds, reach = 1, 0.0
    vox = float(np.abs(np.linalg.det(np.asarray(affine, np.float64)[:3, :3])))
    for h, s in enumerate(surfaces):
        m = np.ones(len(s.white), bool) if members is None else np.asarray(members[h], bool)
        if s.cortex is not None:
            m = m & np.asarray(s.cortex, bool)
        if not m.any():
            continue
        mid = 0.5 * (np.asarray(s.white, np.float64) + np.asarray(s.pial, np.float64))
        thick = np.linalg.norm(np.asarray(s.pial) - np.asarray(s.white), axis=1)
        area = vertex_areas(mid, s.faces, len(s.white))
        per_vertex = float((thick[m] * area[m]).mean()) / max(vox, 1e-12)
        r = _rounds_for(per_vertex, column_voxels) + int(slack_rounds)
        rounds = max(rounds, r)
        reach = max(reach, r * _mean_edge_mm(mid, s.faces))
    return rounds, reach


def _mean_edge_mm(positions: np.ndarray, faces: np.ndarray) -> float:
    """Mean edge length, mm: one round of 1-ring pooling reaches about this far."""
    topo = MeshTopology.from_faces(faces, len(positions))
    e = topo.edges
    return float(np.linalg.norm(positions[e[:, 0]] - positions[e[:, 1]], axis=1).mean())


#: What to do with a region that covers both hemispheres (``split_regions``).
ROI_HEMI_MODES = ("auto", "split", "major", "both")

#: ``auto`` splits when the smaller side holds at least this share of the columns.
#: A third: a bilateral V1 drawn on both banks is a region per hemisphere, while a
#: handful of columns across the midline is spill from a threshold or a millimetre
#: of misalignment, not a region.
HEMI_SPLIT_SHARE = 1.0 / 3.0


@dataclass
class Region:
    """One region to layer: a name, a label value, and its columns per hemisphere.

    The same three fields ``processing.roi_depth`` returns, after the hemisphere
    rule has been applied -- so ``export_roi_depth_volumes`` takes either.
    """

    name: str
    rid: int
    members: dict[int, np.ndarray]


def split_regions(
    rois,
    surfaces: list[RibbonSurfaces],
    mode: str = "auto",
    notes: list[str] | None = None,
) -> list[Region]:
    """Regions to layer, one hemisphere each, from ROIs that may cover both.

    A region spanning both hemispheres would be layered on one grid covering both,
    which for a bilateral V1 is a box of mostly air between the banks. So:

    * ``split`` -- one region per hemisphere, named ``<roi>_lh`` / ``<roi>_rh``;
    * ``major`` -- the hemisphere holding more of the region's columns, the other
      dropped;
    * ``auto`` (the default) -- ``split`` when the smaller side holds at least
      :data:`HEMI_SPLIT_SHARE` of the columns, else ``major``: a region on both
      banks on purpose is two regions, a few columns over the midline is spill;
    * ``both`` -- left alone, on one grid, hemisphere left out of the name.

    Either way the hemisphere goes in the name, so ``-hemi lh`` and ``-hemi rh``
    runs of a bilateral ROI write files that do not collide. What was dropped or
    split is appended to ``notes``.
    """
    if mode not in ROI_HEMI_MODES:
        raise ValueError(f"-roi_hemi must be one of {ROI_HEMI_MODES}, not {mode!r}")
    notes = [] if notes is None else notes
    out: list[Region] = []
    for roi in rois:
        counts = {
            h: int(np.asarray(m, bool).sum())
            for h, m in sorted(roi.members.items())
            if np.asarray(m, bool).any()
        }
        if mode == "both" or len(counts) < 2:
            name = roi.name
            if len(counts) == 1:
                name = _with_hemi(name, surfaces[next(iter(counts))].name)
            out.append(Region(name, roi.rid, dict(roi.members)))
            continue
        order = sorted(counts, key=lambda h: -counts[h])
        total = sum(counts.values())
        share = counts[order[-1]] / total
        split = mode == "split" or (mode == "auto" and share >= HEMI_SPLIT_SHARE)
        where = ", ".join(f"{surfaces[h].name} {counts[h]:,}" for h in order)
        if split:
            notes.append(f"{roi.name}: columns on both hemispheres ({where}) -- one region each")
            for h in order:
                out.append(
                    Region(
                        _with_hemi(roi.name, surfaces[h].name),
                        roi.rid,
                        {h: np.asarray(roi.members[h], bool)},
                    )
                )
        else:
            keep = order[0]
            notes.append(
                f"WARNING: {roi.name} has columns on both hemispheres ({where}); keeping "
                f"{surfaces[keep].name} and dropping {total - counts[keep]:,} column(s) on "
                f"{surfaces[order[-1]].name}. Run it once per -hemi to get both, or "
                "-roi_hemi split to write both from one run"
            )
            out.append(
                Region(
                    _with_hemi(roi.name, surfaces[keep].name),
                    roi.rid,
                    {keep: np.asarray(roi.members[keep], bool)},
                )
            )
    return out


def _with_hemi(name: str, hemi: str) -> str:
    """``name`` carrying ``hemi``, unless its own name already says so."""
    low = name.lower().replace(".", "_").replace("-", "_")
    if low == hemi or low.startswith(f"{hemi}_") or low.endswith(f"_{hemi}"):
        return name
    return f"{name}_{hemi}"


def grow_columns(
    surfaces: list[RibbonSurfaces],
    members: dict[int, np.ndarray],
    *,
    fill_mm: float = 2.0,
    surround_mm: float = 10.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """A region's columns, filled and grown: ``(own, padded)`` vertex masks per hemisphere.

    ``members`` maps a hemisphere's index to the vertices the region covers, as
    ``processing.roi_depth`` reads an ROI file. Two steps, both along the mesh:

    * **fill** (``fill_mm``): a morphological closing, so a region that came from a
      thresholded volume spans the depth as a column should. A voxel that fell below
      threshold leaves a hole in the patch, not a hole in the cortex.
    * **grow** (``surround_mm``): the surround the equivolume CDF needs. Each column's
      volume is pooled with its neighbours' for several rounds, so a column's depths
      depend on cortex a few millimetres away; computing the region alone would leave
      its border columns pooling against nothing. The grown patch only sets the grid
      that gets computed -- the filled patch is what gets written.
    """
    from fastfuncstuff.surface.mesh import close_patch, dilate_patch

    own, padded = [], []
    for h, s in enumerate(surfaces):
        mask = np.asarray(members.get(h, np.zeros(len(s.white), bool)), bool)
        if not mask.any():
            own.append(mask)
            padded.append(mask)
            continue
        mid = 0.5 * (np.asarray(s.white, np.float64) + np.asarray(s.pial, np.float64))
        topo = MeshTopology.from_faces(s.faces, len(s.white))
        o = close_patch(mid, topo, mask, fill_mm)
        if s.cortex is not None:  # the medial wall has no ribbon voxels to claim
            o &= np.asarray(s.cortex, bool)
        own.append(o)
        padded.append(dilate_patch(mid, topo, o, surround_mm))
    return own, padded


def export_roi_depth_volumes(
    surfaces: list[RibbonSurfaces],
    master_affine: np.ndarray,
    master_shape: tuple[int, int, int],
    stem: str,
    ext: str,
    rois,
    *,
    dxyz=None,
    autobox: bool = True,
    pad_mm: float = 1.0,
    fill_mm: float = 2.0,
    surround_mm: float | None = None,
    hemi_mode: str = "auto",
    n_layers: int = 3,
    column_voxels: float = 64.0,
    column_rounds: int | None = None,
    thick_limit: float = 6.0,
    notes: list[str] | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> list[RoiDepthResult]:
    """:func:`export_depth_volumes` for one region at a time, each on its own crop.

    Why a region cannot simply be cut out of a whole-hemisphere run's outputs: it can,
    and that is the expensive way -- a hemisphere at 0.2 mm is tens of millions of
    ribbon voxels for a patch that is thousands. The crop is taken *first*, around the
    region grown by ``surround_mm`` (:func:`grow_columns`), and the depths come out the
    same because every one of them is a local measurement: the fills, the point-to-mesh
    distances and the column CDFs all read geometry within a column's reach of the
    voxel. The one thing that is not local is how far the CDF pools along the surface,
    which the grown surround covers and :attr:`RoiDepthResult.pooled_mm` checks.

    A region covering both hemispheres becomes one region per hemisphere, or the
    hemisphere holding most of it -- see :func:`split_regions` for ``hemi_mode``.
    The hemisphere is in every region's name either way.

    ``surround_mm`` None sizes that surround per region from the grid and the meshes
    (:func:`estimate_column_reach`, plus a quarter for the estimate's own error): the
    pooling reaches furthest where a voxel is large next to the cortex, so one fixed
    default would be waste at 0.2 mm and too little at 0.8 mm.

    Each region writes ``{stem}_{name}_{tag}{ext}`` for every LN2_LAYERS output plus
    ``{stem}_{name}_roi{ext}``, the region itself as a mask -- cropped, not a grid of
    zeros around a patch, and on the master's own voxel centres so it lines up with the
    anatomical the surfaces came from.
    """
    from fastfuncstuff.io.afni import save_nifti

    edge_mm = [_mean_edge_mm(0.5 * (s.white + s.pial), s.faces) for s in surfaces]
    results = []
    said: list[str] = []
    rois = split_regions(rois, surfaces, hemi_mode, said)
    for note in said:
        # A dropped hemisphere is the user's business whether or not they asked for
        # a running commentary; the rest is commentary.
        if verbose or note.startswith("WARNING"):
            print(note if note.startswith("WARNING") else f"  {note}")
    if notes is not None:
        notes += said
    for k, roi in enumerate(rois, 1):
        name = _file_token(roi.name)
        # Where a bar would go: an .annot is 34 regions, each its own fill and
        # distance search, and the line below carries the grid a bar could not.
        count = f"[{k}/{len(rois)}] " if len(rois) > 1 else ""
        affine, shape = np.asarray(master_affine, np.float64), tuple(master_shape)
        if dxyz is not None:
            affine, shape = regrid(affine, shape, dxyz)
        members = [_members_of(roi, h, len(s.white)) for h, s in enumerate(surfaces)]
        surround = surround_mm
        if surround is None:
            _, reach = estimate_column_reach(surfaces, affine, column_voxels, members)
            surround = max(4.0, 1.25 * reach)
        own, padded = grow_columns(surfaces, roi.members, fill_mm=fill_mm, surround_mm=surround)
        if not any(m.any() for m in own):
            continue
        if autobox:
            # The grown patch's whole ribbon: its columns must be complete on this
            # grid, or they pool against voxels the crop cut off.
            pts = np.concatenate(
                [a[m] for s, m in zip(surfaces, padded, strict=True) for a in (s.white, s.pial)]
            )
            affine, shape, _ = crop_to_points(affine, shape, pts, pad_mm)
        n_own = sum(int(m.sum()) for m in own)
        n_grown = sum(int((p & ~o).sum()) for o, p in zip(own, padded, strict=True))
        if verbose:
            vox = np.linalg.norm(affine[:3, :3], axis=0)
            print(
                f"{count}{name}: {n_own:,} columns (+{n_grown:,} surround) onto "
                f"{shape[0]}x{shape[1]}x{shape[2]} at "
                f"{' x '.join(f'{v:.3g}' for v in vox)} mm"
            )
        out = cortical_depth_volumes(
            surfaces,
            affine,
            shape,
            n_layers,
            column_voxels=column_voxels,
            column_rounds=column_rounds,
            columns=own,
            thick_limit=thick_limit,
            device=device,
            verbose=verbose,
        )
        if out.n_gm == 0:
            raise ValueError(
                f"{name}: no GM voxel centre on this grid -- is the master aligned "
                "to the anatomical the surfaces came from?"
            )
        Path(stem).parent.mkdir(parents=True, exist_ok=True)
        written = []
        for tag, vol in out.outputs():
            path = f"{stem}_{name}_{tag}{ext}"
            save_nifti(vol, path, affine=affine)
            written.append(path)
        path = f"{stem}_{name}_roi{ext}"
        save_nifti(out.gm_mask(roi.rid), path, affine=affine)
        written.append(path)
        results.append(
            RoiDepthResult(
                name=name,
                value=int(roi.rid),
                out=out,
                affine=affine,
                written=written,
                n_own=n_own,
                n_filled=sum(int((m & ~mm).sum()) for m, mm in zip(own, members, strict=True)),
                n_grown=n_grown,
                surround_mm=float(surround),
                pooled_mm=max((r * edge_mm[h] for h, r in out.smooth_rounds), default=0.0),
            )
        )
    if not results:
        raise ValueError("no region covers any cortical column of the surfaces given")
    return results


def _file_token(name: str) -> str:
    """A region's name as one path component: ``cap.lh`` -> ``cap_lh``.

    The names come from label tables and file stems, so they carry dots, spaces and
    slashes; a dot in the middle of ``PREFIX_cap.lh_rim.nii.gz`` is legal but reads
    as an extension to the eye and to half the shell one-liners that will touch it.
    """
    out = "".join(c if (c.isalnum() or c in "+-") else "_" for c in name).strip("_")
    return out or "roi"


def _members_of(roi, h: int, n: int) -> np.ndarray:
    return np.asarray(roi.members.get(h, np.zeros(n, bool)), bool)


def roi_report(res: RoiDepthResult) -> list[str]:
    """One region's line, and the warning when its surround was too small."""
    o = res.out
    lines = [
        f"  {res.name}: {res.n_own:,} columns"
        + (f" (+{res.n_filled:,} filled)" if res.n_filled else "")
        + f", {o.n_gm:,} GM voxels, pooled {res.pooled_mm:.1f} mm "
        f"({'/'.join(str(r) for _, r in o.smooth_rounds)} rounds)"
    ]
    if not res.enough_surround:
        lines.append(
            f"WARNING: {res.name}'s equivolume depths pooled {res.pooled_mm:.1f} mm along "
            f"the surface but only {res.surround_mm:.1f} mm of surround was computed, so "
            "the columns near its edge pooled against cortex the crop cut off. "
            f"Re-run with -roi_surround {np.ceil(res.pooled_mm) + 2:.0f}"
        )
    return lines


def _by_hemi(paths: list[str], what: str) -> dict[str, Path]:
    """Files keyed by the hemisphere their names say; unnamed ones take lh, rh in order.

    One path that names no hemisphere and does not exist is a stem for both:
    ``surf/smoothwm`` finds ``surf/lh.smoothwm`` and ``surf/rh.smoothwm``.
    """
    from fastfuncstuff.io.freesurfer import infer_label

    if len(paths) == 1 and infer_label(paths[0])[0] is None and not Path(paths[0]).exists():
        stem = Path(paths[0])
        found = {h: stem.parent / f"{h}.{stem.name}" for h in ("lh", "rh")}
        found = {h: f for h, f in found.items() if f.exists()}
        if not found:
            raise ValueError(f"{what}: neither {paths[0]} nor lh./rh.{stem.name} beside it exists")
        return found
    if len(paths) > 2:
        raise ValueError(f"{what}: at most one file per hemisphere, got {len(paths)}")
    named = [infer_label(p)[0] for p in paths]
    if any(h is None for h in named):
        if any(h is not None for h in named):
            raise ValueError(f"{what}: name every file's hemisphere (lh./rh.) or none")
        named = ["lh", "rh"][: len(paths)]
    if len(set(named)) != len(named):
        raise ValueError(f"{what}: two files for {named[0]}")
    return {str(h): Path(p) for h, p in zip(named, paths, strict=True)}


def _cortex_mask(label, n_vertices: int) -> np.ndarray:
    import nibabel.freesurfer as nfs

    ids = nfs.read_label(str(label))
    mask = np.zeros(n_vertices, bool)
    mask[ids[(ids >= 0) & (ids < n_vertices)]] = True
    return mask


def resolve_ribbon_surfaces(
    fs_subj=None,
    white: str = "white",
    pial: str = "pial",
    white_mesh: list[str] | None = None,
    pial_mesh: list[str] | None = None,
    cortex_label: list[str] | None = None,
    hemi: str = "both",
    log=None,
) -> list[RibbonSurfaces]:
    """White/pial pairs from a FreeSurfer subject (``surf/?h.<white>``) or loose files.

    The one reading of the ``-fs_subj -white -pial -white_mesh -pial_mesh
    -cortex_label -hemi`` flags, shared by ffs_util_surf2layers and the ROI depth
    sampling. ``log`` (a print-like callable) hears what was skipped. Raises
    ``ValueError`` on anything inconsistent.
    """
    from fastfuncstuff.io.freesurfer import (
        load_hemisphere,
        read_scanner_surface,
        read_surface,
        tkr_to_scanner,
    )

    subj = Path(fs_subj) if fs_subj is not None else None
    meshes = white_mesh is not None or pial_mesh is not None
    if subj is None and not meshes:
        raise ValueError("give -fs_subj, or -white_mesh and -pial_mesh")
    if meshes:
        if white_mesh is None or pial_mesh is None:
            raise ValueError("-white_mesh and -pial_mesh go together")
        wfiles, pfiles = _by_hemi(white_mesh, "-white_mesh"), _by_hemi(pial_mesh, "-pial_mesh")
        if wfiles.keys() != pfiles.keys():
            raise ValueError(
                f"-white_mesh covers {', '.join(sorted(wfiles))} but -pial_mesh "
                f"covers {', '.join(sorted(pfiles))}"
            )
        labels = _by_hemi(cortex_label, "-cortex_label") if cortex_label else {}
        if labels.keys() - wfiles.keys():
            raise ValueError("-cortex_label names a hemisphere with no meshes")
        hemis = [h for h in ("lh", "rh") if h in wfiles and hemi in ("both", h)]
        if not hemis:
            raise ValueError(f"-hemi {hemi}: no meshes given for it")
        out = []
        for h in hemis:
            try:
                w, faces = read_scanner_surface(wfiles[h])
                p, pfaces = read_scanner_surface(pfiles[h])
            except OSError as err:
                raise ValueError(str(err)) from err
            if w.shape != p.shape or not np.array_equal(faces, pfaces):
                raise ValueError(
                    f"{wfiles[h].name} and {pfiles[h].name} are not the same mesh "
                    f"({w.shape[0]:,} vs {p.shape[0]:,} vertices): depth pairs them by vertex"
                )
            label = labels.get(h)
            if label is None and subj is not None:
                found = subj / "label" / f"{h}.cortex.label"
                label = found if found.exists() else None
            if label is None and log:
                log(f"  {h}: no cortex label -- medial wall kept")
            tkr = None
            if not wfiles[h].name.lower().endswith(".gii"):
                tkr = tkr_to_scanner(read_surface(wfiles[h]).volume_info)
            cortex = None if label is None else _cortex_mask(label, w.shape[0])
            out.append(RibbonSurfaces(h, w, p, faces, cortex, tkr))
        return out

    assert subj is not None
    if not (subj / "surf").is_dir():
        raise ValueError(f"{subj} has no surf/ directory; is it a FreeSurfer subject?")
    out = []
    for h in ("lh", "rh") if hemi == "both" else (hemi,):
        if not (subj / "surf" / f"{h}.white").exists():
            if hemi != "both":
                raise ValueError(f"{subj}/surf/{h}.white not found")
            continue
        # load_hemisphere requires white and pial whichever pair is asked for.
        states = tuple(dict.fromkeys(("white", "pial", white, pial)))
        hm = load_hemisphere(subj, h, states=states, morph=(), patches=False)
        for state in (white, pial):
            if state not in hm.states:
                raise ValueError(f"{subj}/surf/{h}.{state} not found")
        if hm.cortex is None and log:
            log(f"  {h}: no label/{h}.cortex.label -- medial wall kept")
        out.append(RibbonSurfaces.from_hemisphere(hm, white, pial))
    if not out:
        raise ValueError(f"{subj}/surf has no ?h.white")
    return out


__all__ = [
    "OUTPUT_TAGS",
    "DepthVolumes",
    "RibbonSurfaces",
    "ROI_HEMI_MODES",
    "Region",
    "RoiDepthResult",
    "column_smooth_rounds",
    "estimate_column_reach",
    "cortical_depth_volumes",
    "crop_to_points",
    "export_depth_volumes",
    "export_roi_depth_volumes",
    "grow_columns",
    "regrid",
    "resolve_ribbon_surfaces",
    "roi_report",
    "split_regions",
    "thick_report",
    "volume_quantile_depth",
]
