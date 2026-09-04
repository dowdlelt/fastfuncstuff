"""Fused neighbour-minimum kernel for batched connected components.

The hot step of Shiloach-Vishkin label propagation is "for every active voxel,
the smallest label among itself and its neighbours".  Written in torch that is
``lab[adj].amin(0)``, which materialises a ``[K, T]`` int64 intermediate —
330 MB written and read back *every round* at a realistic batch size, for a
result that is only ``[T]``.  It was 77% of the labelling pass.

Here one program handles a block of voxels and folds the K neighbours in
registers: no intermediate, int32 labels and int32 addressing (int64
addressing measured 2.7x more expensive elsewhere in the toolbox), and the
adjacency stored ``[K, T]`` so each direction's read is coalesced.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from fastfuncstuff.triton_key import install_triton_key_cache

# Triton hashes its whole installation on the first kernel launch of every
# process (~1s).  Do this before any @triton.jit function can be launched.
install_triton_key_cache()


@triton.jit
def _neighbor_min_kernel(
    lab_ptr,
    adj_ptr,
    out_ptr,
    n_act,
    big,
    n_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    live = offs < n_act
    m = tl.load(lab_ptr + offs, mask=live, other=big)
    for k in range(n_k):
        nb = tl.load(adj_ptr + k * n_act + offs, mask=live, other=-1)
        ok = live & (nb >= 0)
        v = tl.load(lab_ptr + tl.where(ok, nb, 0), mask=ok, other=big)
        m = tl.minimum(m, v)
    tl.store(out_ptr + offs, m, mask=live)


def neighbor_min(lab: Tensor, adj_t: Tensor, big: int) -> Tensor:
    """``[T]`` int32 minimum of each voxel's own label and its neighbours'.

    ``adj_t`` is ``[K, T]`` int32, ``-1`` where a neighbour is absent or below
    threshold.
    """
    n_k, n_act = adj_t.shape
    out = torch.empty_like(lab)
    block = 256
    grid = (triton.cdiv(n_act, block),)
    _neighbor_min_kernel[grid](lab, adj_t, out, n_act, big, n_k=n_k, BLOCK=block)
    return out


def neighbor_min_torch(lab: Tensor, adj_t: Tensor, big: int) -> Tensor:
    """Reference implementation; the CPU path and the Triton correctness gate."""
    m = lab
    for k in range(adj_t.shape[0]):
        nb = adj_t[k]
        ok = nb >= 0
        m = torch.minimum(m, torch.where(ok, lab[nb.clamp(min=0).long()], big))
    return m


@triton.jit
def _adjacency_kernel(
    vox_ptr,
    batch_ptr,
    nbr_ptr,
    rank_ptr,
    adj_ptr,
    n_act,
    n_vox,
    n_k: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    live = offs < n_act
    v = tl.load(vox_ptr + offs, mask=live, other=0)
    b = tl.load(batch_ptr + offs, mask=live, other=0)
    for k in range(n_k):
        nb = tl.load(nbr_ptr + v * n_k + k, mask=live, other=-1)
        ok = live & (nb >= 0)
        g = b * n_vox + tl.where(ok, nb, 0)
        r = tl.load(rank_ptr + g, mask=ok, other=-1)
        tl.store(adj_ptr + k * n_act + offs, tl.where(ok, r, -1), mask=live)


def adjacency(vox: Tensor, batch_id: Tensor, nbr: Tensor, rank: Tensor, n_vox: int) -> Tensor:
    """``[K, T]`` int32 compact adjacency, built without a single temporary.

    The torch spelling of this needs a ``[T, K]`` gather, a transpose to
    K-major, ``[K, T]`` index arithmetic and a second gather — four int64
    intermediates for an int32 result, and it cost more than the labelling it
    feeds.
    """
    n_act = int(vox.numel())
    n_k = int(nbr.shape[1])
    out = torch.empty((n_k, n_act), dtype=torch.int32, device=vox.device)
    block = 256
    _adjacency_kernel[(triton.cdiv(n_act, block),)](
        vox, batch_id, nbr, rank, out, n_act, n_vox, n_k=n_k, BLOCK=block
    )
    return out
