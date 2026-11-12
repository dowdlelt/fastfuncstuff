"""
Cluster-level inference shared by every tool that builds a null of z/t maps.

One driver, :func:`max_fom_null`, turns a *source* of null maps — permutations
(``ffs_perm``, the group tool), simulated random fields (``ffs_clustsim``),
blurred sign-flipped residuals (ETAC) — into the per-iteration largest-cluster
figure of merit for every (sidedness, NN, FOM power, threshold).  What differs
between tools is only the source, so the clustering, the batching and the
memory plan live here once.  See [[Group-level inference]] in the wiki.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from tqdm.auto import tqdm

from fastfuncstuff.memory import cluster_fom_batch


class VolumeGeometry:
    """Where the statistic lives: a 3-D mask on a voxel grid.

    Everything the cluster layer needs from a space -- neighbours, a measure per
    element (1 per voxel), a smoother -- comes through this interface, so the
    clustering, ETAC and ClustSim code is the same for volumes and surfaces.
    """

    kind = "volume"
    unit = "voxels"

    def __init__(self, mask: np.ndarray, voxdims=(1.0, 1.0, 1.0)):
        self.mask = np.asarray(mask, bool)
        self.voxdims = tuple(float(abs(v)) for v in voxdims)
        self.n = int(self.mask.sum())

    def neighbors(self, nn: int, device: torch.device) -> torch.Tensor:
        from fastfuncstuff.stats.cluster_gpu import build_neighbor_table

        return build_neighbor_table(self.mask, nn, device)

    def weights(self, device: torch.device) -> torch.Tensor | None:
        return None

    def smoother(self, fwhm: float, device: torch.device):
        """Callable blurring in-mask rows ``[k, n]`` on the device (3dBlurInMask-like)."""
        from fastfuncstuff.stats.blur_to import MaskedGaussian

        return MaskedGaussian(self.mask, self.voxdims, fwhm, device=device).rows


class SurfaceGeometry:
    """A masked triangle mesh: clusters over mesh edges, measured in mm^2.

    ``vertices`` should be the midthickness the data were sampled on (the
    measure of a vertex is its share of the triangle areas, and the heat
    kernel's distances are along this surface).  Neighbours are the same for any
    NN level -- a mesh has one connectivity.
    """

    kind = "surface"
    unit = "mm2"

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray | None = None):
        self.vertices = np.asarray(vertices, np.float64)
        self.faces = np.asarray(faces, np.int64)
        n_all = self.vertices.shape[0]
        self.mask = np.ones(n_all, bool) if mask is None else np.asarray(mask, bool).reshape(-1)
        if self.mask.size != n_all:
            raise ValueError(f"mask has {self.mask.size} vertices, the mesh {n_all}")
        self.n = int(self.mask.sum())
        self._nbr: np.ndarray | None = None

    def _table(self) -> np.ndarray:
        """``[n, K]`` in-mask neighbour indices over mesh edges, -1 padded."""
        if self._nbr is None:
            from fastfuncstuff.surface.smooth import mesh_edges

            e = mesh_edges(self.faces)
            e = e[self.mask[e[:, 0]] & self.mask[e[:, 1]]]
            remap = np.full(self.mask.size, -1, np.int64)
            remap[self.mask] = np.arange(self.n)
            a = remap[np.concatenate([e[:, 0], e[:, 1]])]
            b = remap[np.concatenate([e[:, 1], e[:, 0]])]
            order = np.argsort(a, kind="stable")
            a, b = a[order], b[order]
            deg = np.bincount(a, minlength=self.n)
            start = np.concatenate([[0], np.cumsum(deg)[:-1]])
            slot = np.arange(a.size) - start[a]
            tab = np.full((self.n, max(int(deg.max(initial=0)), 1)), -1, np.int32)
            tab[a, slot] = b.astype(np.int32)
            self._nbr = tab
        return self._nbr

    def neighbors(self, nn: int, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(self._table()).to(device)

    def weights(self, device: torch.device) -> torch.Tensor:
        from fastfuncstuff.surface.mesh import vertex_areas

        area = vertex_areas(self.vertices, self.faces)[self.mask]
        return torch.as_tensor(area, dtype=torch.float32, device=device)

    def smoother(self, fwhm: float, device: torch.device):
        """Callable heat-kernel blur of in-mask rows ``[k, n]`` (FWHM in mm along the mesh)."""
        from fastfuncstuff.surface.smooth import HeatSmoother

        dev = device if device.type == "cuda" else None  # exact direct solve on the CPU
        hs = HeatSmoother(self.vertices, self.faces, fwhm, mask=self.mask, device=dev)
        idx = torch.as_tensor(np.flatnonzero(self.mask))

        def rows(x: torch.Tensor) -> torch.Tensor:
            full = torch.zeros((self.mask.size, x.shape[0]), dtype=torch.float32, device=x.device)
            full[idx.to(x.device)] = x.T.float()
            if dev is None:
                sm = torch.as_tensor(hs(full.cpu().numpy()), dtype=torch.float32)
            else:
                sm = hs(full)
            return sm[idx.to(sm.device)].T.to(device=x.device, dtype=x.dtype).contiguous()

        return rows


Where = np.ndarray | VolumeGeometry | SurfaceGeometry


def as_geometry(where) -> VolumeGeometry | SurfaceGeometry:
    """A bare 3-D mask means a volume; geometry objects pass through."""
    if isinstance(where, (VolumeGeometry, SurfaceGeometry)):
        return where
    return VolumeGeometry(np.asarray(where, bool))


# A source returns ``nb`` null maps ``[nb, V]`` (float32, on ``device``)
# starting at iteration ``start``.  Deterministic sources make a run
# reproducible independently of the batch size the memory plan picks.
NullSource = Callable[[int, int], torch.Tensor]


def max_fom_null(
    source: NullSource,
    n_iter: int,
    mask,
    tcrits: dict[str, np.ndarray],
    *,
    nns: tuple[int, ...],
    sideds: tuple[str, ...],
    hpows: tuple[int, ...] = (0,),
    min_clust: int = 1,
    device: torch.device,
    batch: int | None = None,
    source_bytes: int | None = None,
    desc: str = "cluster null",
    verbose: bool = True,
) -> dict[tuple[str, int, int], np.ndarray]:
    """``{(sided, nn, h): [n_iter, npthr]}`` largest cluster FOM per null map.

    ``mask`` is a 3-D boolean mask or a :class:`SurfaceGeometry` (then FOMs
    are area-weighted: h=0 is the cluster area in mm^2).
    ``tcrits[sided]`` are the cluster-forming thresholds in the caller's order;
    columns come back in that order.  FOM ``h=0`` is the voxel count, ``1`` the
    mass, ``2`` the sum of squares (3dXClustSim).  Only one batch is resident;
    its size counts ``source_bytes`` per map (default: the source's own
    ``bytes_per_map()`` when it has one).
    """
    from fastfuncstuff.stats.cluster_gpu import cluster_fom_batched

    geom = as_geometry(mask)
    n_vox = geom.n
    if source_bytes is None:
        per_map = getattr(source, "bytes_per_map", None)
        source_bytes = int(per_map()) if callable(per_map) else 0
    if batch is None:
        batch = cluster_fom_batch(n_vox, len(hpows), device, source_bytes=source_bytes)
    batch = int(max(1, min(batch, n_iter)))

    # The batched labeller wants thresholds tightest-first.
    tc_dev: dict[str, torch.Tensor] = {}
    order: dict[str, np.ndarray] = {}
    for s in sideds:
        arr = np.asarray(tcrits[s], dtype=np.float64)
        idx = np.argsort(-arr, kind="stable")
        order[s] = idx
        tc_dev[s] = torch.from_numpy(arr[idx].copy()).to(device, torch.float32)
    n_pthr = len(next(iter(order.values())))

    nbr = {nn: geom.neighbors(nn, device) for nn in nns}
    weight = geom.weights(device)
    scratch = torch.full((batch * n_vox,), -1, device=device, dtype=torch.int32)
    out = {
        (s, nn, h): np.zeros((n_iter, n_pthr), dtype=np.float32)
        for s in sideds
        for nn in nns
        for h in hpows
    }
    bar = tqdm(total=n_iter, desc=desc, leave=True, disable=not verbose or n_iter < 2)
    try:
        for start in range(0, n_iter, batch):
            nb = min(batch, n_iter - start)
            fields = source(start, nb)
            view = scratch[: nb * n_vox]
            res = cluster_fom_batched(
                fields, nbr, nns, sideds, tc_dev, hpows, min_clust, view, weight=weight
            )
            del fields
            for (s, nn, h), tab in res.items():
                out[(s, nn, h)][start : start + nb, order[s]] = tab.cpu().numpy()
            bar.update(nb)
    finally:
        bar.close()
    return out


def rows_source(t: np.ndarray, device: torch.device) -> NullSource:
    """A :data:`NullSource` over the rows of an in-memory ``[P, V]`` matrix."""

    def source(start: int, nb: int) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(t[start : start + nb], dtype=np.float32)).to(
            device
        )

    return source
