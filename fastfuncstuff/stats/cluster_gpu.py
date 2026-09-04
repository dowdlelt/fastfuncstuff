"""
Batched connected-component cluster extents on the GPU.

The CPU path (:mod:`fastfuncstuff.stats.cluster_fast`) walks one volume at a
time with a sorted union-find, which is the right algorithm for one map and
the wrong shape for ten thousand of them.  Here every simulated volume in a
batch is labelled at once.

Two facts make this cheap, and both are properties of a *noise* null rather
than of a real stat map:

* **Work scales with the suprathreshold set, not the volume.**  At p=0.05
  exactly 5% of voxels survive by construction, so the labelling runs on a
  compacted list of active voxels — a dense pass over the whole volume would
  waste 20x.  Thresholds are nested, so summed over the nine ``pthr`` the
  total is only ~1.3x the loosest one.
* **Label propagation converges in about four rounds.**  The worry with
  Shiloach-Vishkin here is that the p=0.05 set percolates and a plain
  neighbour-max needs geodesic-diameter iterations.  Pointer jumping
  (``lab = lab[lab]``) collapses chains logarithmically: measured at 4 rounds
  even at FWHM ~7 voxels where the largest null cluster is 4257 voxels.

Components never span batch rows — the compacted adjacency only ever links
voxels within one volume — so a single flat labelling over the whole batch is
safe, and the per-volume maximum falls out of one scatter-reduce.

Calibration, so a future change can be judged against something real: on a
2.37 M-voxel 0.8 mm mask, 10 000 iterations take ~88 s here against a measured
12.8 hours for 3dClustSim on 6 threads.  That ratio is 26x larger than on a
64^3 test volume, because AFNI's cost is per-grid-voxel and this one's is per
*suprathreshold* voxel — the advantage is the compaction, not the GPU.  At that
size the labelling is ~68% of the run and sits at the random-gather roofline:
Morton-ordering the voxels for locality measured 0.99x and halving the pointer
jumps 1.06x, so neither is worth retrying.  The remaining cheap win is
``build_neighbor_table`` itself, 8.8 s of single-threaded numpy.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff.stats.cluster_fast import _offsets_for_nn


def build_neighbor_table(
    mask: np.ndarray,
    nn: int,
    device: torch.device,
) -> Tensor:
    """``[V, K]`` int32 table of in-mask neighbour indices, ``-1`` where none.

    Rows follow the in-mask voxel order (``np.flatnonzero(mask.ravel())``), so
    the table indexes the same compact voxel space the fields are stored in.
    Built once per (mask, NN) and reused for every iteration and threshold —
    it is the static half of the problem.
    """
    shape = mask.shape
    flat_idx = np.flatnonzero(mask.ravel())
    v_total = int(np.prod(shape))
    # volume flat index -> in-mask index, -1 outside
    lookup = np.full(v_total, -1, dtype=np.int64)
    lookup[flat_idx] = np.arange(flat_idx.size)

    ijk = np.stack(np.unravel_index(flat_idx, shape), axis=1)  # [V, 3]
    offsets = _offsets_for_nn(nn)
    out = np.full((flat_idx.size, offsets.shape[0]), -1, dtype=np.int32)
    for k, off in enumerate(offsets):
        nb = ijk + off
        inside = np.all((nb >= 0) & (nb < np.asarray(shape)), axis=1)
        nb_flat = np.ravel_multi_index(nb[inside].T, shape)
        out[inside, k] = lookup[nb_flat].astype(np.int32)
    return torch.from_numpy(out).to(device)


def _neighbor_min(lab: Tensor, adj_t: Tensor, big: int) -> Tensor:
    """Fused where available; the torch form is the fallback and the gate."""
    if lab.is_cuda:
        try:
            from fastfuncstuff.stats.cluster_gpu_triton import neighbor_min

            return neighbor_min(lab, adj_t, big)
        except ImportError:
            pass
    from fastfuncstuff.stats.cluster_gpu_triton import neighbor_min_torch

    return neighbor_min_torch(lab, adj_t, big)


def _connected_max_sizes(
    active_flat: Tensor,  # [T] int64, indices into the flattened [B*V] batch
    batch_id: Tensor,  # [T] int64
    adj_t: Tensor,  # [K, T] int32, compact neighbour ids or -1
    n_batch: int,
    max_rounds: int = 32,
) -> Tensor:
    """Per-volume largest component size, from a compacted adjacency.

    Shiloach-Vishkin: hook every root onto the smallest label that reaches it,
    then pointer-jump to compress the resulting forest.  The jumping is what
    keeps this logarithmic rather than proportional to a cluster's geodesic
    diameter — on this null it converges in about four rounds even where the
    p=0.05 set percolates.
    """
    n_act = int(active_flat.numel())
    device = active_flat.device
    lab = torch.arange(n_act, device=device, dtype=torch.int32)

    for _ in range(max_rounds):
        m = _neighbor_min(lab, adj_t, n_act)
        new = lab.clone()
        new.scatter_reduce_(0, lab.long(), m, reduce="amin")
        new = torch.minimum(new, m)
        # Path compression: [T]-sized, so the int64 index is cheap here in a
        # way it is not for the [K, T] neighbour gather above.
        for _ in range(4):
            new = new[new.long()]
        if torch.equal(new, lab):
            break
        lab = new

    counts = torch.bincount(lab.long(), minlength=n_act)
    out = torch.zeros(n_batch, device=device, dtype=torch.int64)
    out.scatter_reduce_(0, batch_id, counts, reduce="amax")
    return out


def _compact(eff: Tensor, tcrit: Tensor, rank_scratch: Tensor, n_vox: int):
    """Suprathreshold voxels of a batch as a flat compact list.

    Returns ``(active_flat, batch_id, vox, n_act)`` or ``None`` if empty.
    ``rank_scratch`` is left holding volume-slot -> compact-id for these
    voxels; the caller must hand it back to :func:`_uncompact`.
    """
    active_flat = (eff > tcrit).reshape(-1).nonzero(as_tuple=False).squeeze(1)
    n_act = int(active_flat.numel())
    if n_act == 0:
        return None
    rank_scratch[active_flat] = torch.arange(n_act, device=eff.device, dtype=torch.int32)
    batch_id = active_flat // n_vox
    return active_flat, batch_id, active_flat - batch_id * n_vox, n_act


def _uncompact(active_flat: Tensor, rank_scratch: Tensor) -> None:
    """Clear only the slots this threshold set — a whole-buffer memset here
    would dwarf the labelling it is resetting."""
    rank_scratch[active_flat] = -1


def _adjacency(
    nbr: Tensor, vox: Tensor, batch_id: Tensor, rank_scratch: Tensor, n_vox: int
) -> Tensor:
    """``[K, T]`` int32 compact neighbour ids, ``-1`` where the neighbour is
    below threshold or outside the mask.

    K-major so that each neighbour direction is a contiguous read in the
    labelling kernel.
    """
    if vox.is_cuda:
        try:
            from fastfuncstuff.stats.cluster_gpu_triton import adjacency

            return adjacency(
                vox.to(torch.int32), batch_id.to(torch.int32), nbr, rank_scratch, n_vox
            )
        except ImportError:
            pass
    nb = nbr[vox].to(torch.int64).T.contiguous()  # [K, T]
    gidx = batch_id.unsqueeze(0) * n_vox + nb.clamp(min=0)
    out = torch.where(nb >= 0, rank_scratch[gidx].to(torch.int64), torch.full_like(gidx, -1))
    return out.to(torch.int32)


def cluster_extent_batched(
    fields: Tensor,  # [B, V] float32 on device
    nbr_by_nn: dict[int, Tensor],
    nns: tuple[int, ...],
    sideds: tuple[str, ...],
    tcrits_by_sided: dict[str, Tensor],
    rank_scratch: Tensor | None = None,
) -> dict[tuple[str, int], Tensor]:
    """``{(sided, nn): [B, npthr]}`` largest cluster, in descending-tcrit order.

    Mirrors ``cluster_fast.cluster_extent_one_perm`` for a whole batch, but
    shares work the per-volume walk cannot:

    * **One compaction per threshold, not per connectivity.**  NN1/2/3 differ
      only in which neighbours they union — they see the identical
      suprathreshold set — so the ``nonzero`` scan over the batch is hoisted
      out of the NN loop.
    * **Bi-sided is one labelling, not two.**  Its threshold is the 2-sided
      one and its active set is the same ``|z| > zthr`` set; the only
      difference is that opposite-sign voxels may not join.  Masking
      cross-sign edges out of the adjacency labels both tails at once, and
      the per-volume maximum over the result is exactly ``max(siz_p, siz_m)``
      — so the second pass AFNI runs is unnecessary here.

      That shortcut needs ``zthr >= 0``, which every real p-value gives:
      ``{|z| > t}`` only partitions into ``{z > t}`` and ``{z < -t}`` when
      ``t`` is non-negative, and below zero the two tails overlap instead.
      A negative threshold therefore falls back to labelling each tail
      separately, as the per-volume walk does.
    """
    n_batch, n_vox = fields.shape
    device = fields.device
    if rank_scratch is None:
        rank_scratch = torch.full((n_batch * n_vox,), -1, device=device, dtype=torch.int32)

    want_pos = "1-sided" in sideds
    want_abs = ("2-sided" in sideds) or ("bi-sided" in sideds)
    n_pthr = int(next(iter(tcrits_by_sided.values())).numel())
    out = {
        (s, nn): torch.zeros((n_batch, n_pthr), device=device, dtype=torch.int64)
        for s in sideds
        for nn in nns
    }
    eff_abs = fields.abs() if want_abs else None
    positive = fields > 0 if "bi-sided" in sideds else None

    for ip in range(n_pthr):
        if want_pos:
            got = _compact(fields, tcrits_by_sided["1-sided"][ip], rank_scratch, n_vox)
            if got is not None:
                active_flat, batch_id, vox, _ = got
                for nn in nns:
                    adj = _adjacency(nbr_by_nn[nn], vox, batch_id, rank_scratch, n_vox)
                    out[("1-sided", nn)][:, ip] = _connected_max_sizes(
                        active_flat, batch_id, adj, n_batch
                    )
                _uncompact(active_flat, rank_scratch)

        if not want_abs:
            continue
        assert eff_abs is not None
        tc = tcrits_by_sided["2-sided" if "2-sided" in sideds else "bi-sided"][ip]
        want_bi = "bi-sided" in sideds
        # The sign-split shortcut needs a non-negative threshold; below zero
        # the two tails overlap and each must be labelled on its own.
        split_ok = want_bi and bool(tc >= 0)

        got = _compact(eff_abs, tc, rank_scratch, n_vox)
        if got is not None:
            active_flat, batch_id, vox, _ = got
            sign = (
                positive.reshape(-1)[active_flat] if (split_ok and positive is not None) else None
            )
            for nn in nns:
                adj = _adjacency(nbr_by_nn[nn], vox, batch_id, rank_scratch, n_vox)
                if "2-sided" in sideds:
                    out[("2-sided", nn)][:, ip] = _connected_max_sizes(
                        active_flat, batch_id, adj, n_batch
                    )
                if sign is not None:
                    same = sign.unsqueeze(0) == torch.where(
                        adj >= 0, sign[adj.clamp(min=0).long()], ~sign.unsqueeze(0)
                    )
                    out[("bi-sided", nn)][:, ip] = _connected_max_sizes(
                        active_flat,
                        batch_id,
                        torch.where(same, adj, torch.full_like(adj, -1)),
                        n_batch,
                    )
            _uncompact(active_flat, rank_scratch)

        if want_bi and not split_ok:
            for tail in (fields, -fields):
                got_t = _compact(tail, tc, rank_scratch, n_vox)
                if got_t is None:
                    continue
                af_t, bid_t, vox_t, _ = got_t
                for nn in nns:
                    adj_t = _adjacency(nbr_by_nn[nn], vox_t, bid_t, rank_scratch, n_vox)
                    out[("bi-sided", nn)][:, ip] = torch.maximum(
                        out[("bi-sided", nn)][:, ip],
                        _connected_max_sizes(af_t, bid_t, adj_t, n_batch),
                    )
                _uncompact(af_t, rank_scratch)

    return out
