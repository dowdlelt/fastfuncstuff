"""Fused CUDA kernel for the qwarp Gauss-Newton normal equations.

The portable path in :mod:`fastfuncstuff.processing.warp` materialises the
steepest-descent images -- ``(B, V, D*nb)``, the only tensor in qwarp carrying a
parameter axis on top of the volume -- and then contracts them twice. Profiled on
a 0.7 mm pair, assembling those columns was the slowest kernel at every pyramid
level (0.59-0.65 ms per 61 MiB against 0.08 ms for the gradient reduction),
because eager spends three kernels and five passes over the largest tensor in the
algorithm to produce it once.

The columns are only ever consumed by two sums over voxels, so they need not
exist. This kernel accumulates ``JᵀΩJ`` and ``Jᵀr`` straight into a per-patch
``NCOL x NCOL`` register tile, reading only the sampled gradient, the basis, and
three per-voxel scalars. At level 151^3 that turns ~13.7 GB of traffic per
iteration into ~0.5 GB and leaves the kernel compute-bound.

Determinism is kept deliberately. Voxels are split into a fixed ``NSPLIT``
ranges whose partial tiles are summed by torch afterwards, rather than reduced
with atomics: qwarp runs are otherwise bit-reproducible, and a registration whose
answer moves between runs cannot be A/B tested (see the allineate determinism
note). The split also keeps the GPU busy at the coarse levels, where there are
only a few dozen patches and one program per patch would leave most of the card
idle.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from fastfuncstuff.triton_key import install_triton_key_cache

# Triton hashes its whole installation on the first kernel launch of every
# process (~1s). Do this before any @triton.jit function can be launched.
install_triton_key_cache()

# Enough programs to fill a modern card once B * NSPLIT is reached; past this the
# split only adds partial tiles to sum.
_TARGET_PROGRAMS = 1024


@triton.jit
def _gn_normal_eqs_kernel(
    g_ptr,  # (D, B, V)
    bt_ptr,  # (V, NB)
    hw_ptr,  # (D,)
    mean_ptr,  # (B, NCOL_PAD)
    inv_ptr,  # (B, V) -- sqrt(omega) / scale
    rwres_ptr,  # (B, V) -- sqrt(omega) * residual
    h_ptr,  # (B, NSPLIT, NCOL_PAD, NCOL_PAD)
    grad_ptr,  # (B, NSPLIT, NCOL_PAD)
    V,
    n_split,
    stride_gd,
    stride_gb,
    NB: tl.constexpr,
    NCOL: tl.constexpr,
    NCOL_PAD: tl.constexpr,
    HAS_MEAN: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Contiguous voxel range for this split, rounded to whole BLOCK_V tiles so
    # every program's inner loop has the same shape.
    per_split = tl.cdiv(tl.cdiv(V, BLOCK_V), n_split) * BLOCK_V
    v_start = pid_s * per_split
    v_end = tl.minimum(v_start + per_split, V)

    offs_c = tl.arange(0, NCOL_PAD)
    c_mask = offs_c < NCOL
    d_idx = offs_c // NB
    n_idx = offs_c % NB
    hw = tl.load(hw_ptr + d_idx, mask=c_mask, other=0.0)
    if HAS_MEAN:
        mean = tl.load(mean_ptr + pid_b * NCOL_PAD + offs_c, mask=c_mask, other=0.0)
    else:
        mean = tl.zeros((NCOL_PAD,), dtype=tl.float32)

    acc = tl.zeros((NCOL_PAD, NCOL_PAD), dtype=tl.float32)
    gacc = tl.zeros((NCOL_PAD,), dtype=tl.float32)

    for v0 in tl.range(v_start, v_end, BLOCK_V):
        offs_v = v0 + tl.arange(0, BLOCK_V)
        v_mask = offs_v < v_end
        both = v_mask[:, None] & c_mask[None, :]

        inv = tl.load(inv_ptr + pid_b * V + offs_v, mask=v_mask, other=0.0)
        rwres = tl.load(rwres_ptr + pid_b * V + offs_v, mask=v_mask, other=0.0)
        # bt[v, n(c)] and g[d(c), b, v]: the gradient is re-read once per basis
        # function of its own direction, which is a cache hit, not a fetch.
        bt = tl.load(bt_ptr + offs_v[:, None] * NB + n_idx[None, :], mask=both, other=0.0)
        g = tl.load(
            g_ptr + d_idx[None, :] * stride_gd + pid_b * stride_gb + offs_v[:, None],
            mask=both,
            other=0.0,
        )

        q = (g * bt * hw[None, :] - mean[None, :]) * inv[:, None]
        q = tl.where(both, q, 0.0)

        # ieee, not the tensor-core default: this assembles a Hessian that is
        # solved for a step, and TF32 was measured to buy qwarp nothing anyway.
        acc += tl.dot(tl.trans(q), q, input_precision="ieee")
        gacc += tl.sum(q * rwres[:, None], axis=0)

    h_base = h_ptr + (pid_b * n_split + pid_s) * NCOL_PAD * NCOL_PAD
    tl.store(
        h_base + offs_c[:, None] * NCOL_PAD + offs_c[None, :],
        acc,
        mask=c_mask[:, None] & c_mask[None, :],
    )
    tl.store(grad_ptr + (pid_b * n_split + pid_s) * NCOL_PAD + offs_c, gacc, mask=c_mask)


def gn_normal_eqs_triton(
    g: Tensor,
    hw: Tensor,
    bt: Tensor,
    scale: Tensor,
    omega: Tensor,
    res: Tensor,
    mean_cols: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """``(hmat, grad)`` without ever assembling the steepest-descent columns.

    Args mirror :func:`fastfuncstuff.processing.warp._gn_accumulate`:
    ``g`` is (D, B, V), ``hw`` (D, 1, 1), ``bt`` (V, nb), ``scale`` (B, V) or
    (B, 1), ``omega`` and ``res`` (B, V), ``mean_cols`` (B, 1, D*nb) or None.
    """
    d, b, v = g.shape
    nb = bt.shape[1]
    ncol = d * nb
    # tl.dot needs at least a 16-wide tile; the padded columns are masked to zero
    # and sliced off the result.
    ncol_pad = max(16, triton.next_power_of_2(ncol))

    rw = omega.clamp_min(0.0).sqrt()
    inv = (rw / scale.expand(b, v)).contiguous()
    rwres = (rw * res).contiguous()

    if mean_cols is None:
        mean_pad = torch.empty(0, device=g.device, dtype=torch.float32)
    else:
        mean_pad = torch.zeros((b, ncol_pad), device=g.device, dtype=torch.float32)
        mean_pad[:, :ncol] = mean_cols.reshape(b, ncol)

    block_v = 64 if v < 4096 else 128
    n_split = max(1, min(triton.cdiv(v, block_v), triton.cdiv(_TARGET_PROGRAMS, b)))

    h_part = torch.zeros((b, n_split, ncol_pad, ncol_pad), device=g.device, dtype=torch.float32)
    g_part = torch.zeros((b, n_split, ncol_pad), device=g.device, dtype=torch.float32)

    g = g.contiguous()
    _gn_normal_eqs_kernel[(b, n_split)](
        g,
        bt.contiguous(),
        hw.reshape(-1).contiguous(),
        mean_pad,
        inv,
        rwres,
        h_part,
        g_part,
        v,
        n_split,
        g.stride(0),
        g.stride(1),
        NB=nb,
        NCOL=ncol,
        NCOL_PAD=ncol_pad,
        HAS_MEAN=mean_cols is not None,
        BLOCK_V=block_v,
    )
    return (
        h_part.sum(1)[:, :ncol, :ncol].contiguous(),
        g_part.sum(1)[:, :ncol].contiguous(),
    )
