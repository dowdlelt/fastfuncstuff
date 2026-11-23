"""Seed-to-voxel correlation maps, streamed over voxels.

One map per seed: every voxel's cleaned course against the seed's, which after
:func:`matrix.unit_rows` is a single ``(chunk, T) @ (T, S)`` product per voxel chunk.
The voxel matrix may sit on the host; only a chunk of it is on the device at a time
(3dNetCorr ``-ts_wb_corr``, 3dTcorr1D).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from fastfuncstuff.connectivity.matrix import unit_rows
from fastfuncstuff.utils import full_fp32_matmul


@full_fp32_matmul()
def seed_maps(
    voxels: Tensor,
    seeds: Tensor,
    *,
    weights: Tensor | np.ndarray | None = None,
    device: torch.device | None = None,
    chunk: int | None = None,
    verbose: bool = False,
) -> Tensor:
    """``(V, T)`` voxels x ``(S, T)`` seeds -> ``(V, S)`` r on the CPU (float32).

    Both inputs must already be cleaned the same way; ``weights`` (``(T,)``) makes it
    the weighted Pearson r. Constant voxels and constant seeds give r = 0.
    """
    device = device or voxels.device
    n_vox, n_t = voxels.shape
    if seeds.shape[1] != n_t:
        raise ValueError(f"seeds have {seeds.shape[1]} TRs, voxels {n_t}")
    s, _ = unit_rows(seeds.to(device), weights, dtype=torch.float32)
    if chunk is None:
        from fastfuncstuff.memory import estimate_chunk_size

        chunk = estimate_chunk_size(
            n_vox, n_t, seeds.shape[0], device, operation="paired_corr", n_trials=2
        )
    chunk = max(1, min(int(chunk), max(n_vox, 1)))
    out = torch.zeros(n_vox, seeds.shape[0], dtype=torch.float32)
    bar = tqdm(
        range(0, n_vox, chunk),
        desc="Seed maps",
        unit="chunk",
        leave=True,
        disable=not verbose or n_vox <= chunk,
    )
    for v0 in bar:
        v, _ = unit_rows(voxels[v0 : v0 + chunk].to(device), weights, dtype=torch.float32)
        out[v0 : v0 + chunk] = (v @ s.T).clamp_(-1.0, 1.0).cpu()
    return out
