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

# A source returns ``nb`` null maps ``[nb, V]`` (float32, on ``device``)
# starting at iteration ``start``.  Deterministic sources make a run
# reproducible independently of the batch size the memory plan picks.
NullSource = Callable[[int, int], torch.Tensor]


def max_fom_null(
    source: NullSource,
    n_iter: int,
    mask: np.ndarray,
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

    ``tcrits[sided]`` are the cluster-forming thresholds in the caller's order;
    columns come back in that order.  FOM ``h=0`` is the voxel count, ``1`` the
    mass, ``2`` the sum of squares (3dXClustSim).  Only one batch is resident;
    its size counts ``source_bytes`` per map (default: the source's own
    ``bytes_per_map()`` when it has one).
    """
    from fastfuncstuff.stats.cluster_gpu import build_neighbor_table, cluster_fom_batched

    n_vox = int(mask.sum())
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

    nbr = {nn: build_neighbor_table(mask, nn, device) for nn in nns}
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
            res = cluster_fom_batched(fields, nbr, nns, sideds, tc_dev, hpows, min_clust, view)
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
