"""Voxel-to-voxel temporal correlation BETWEEN datasets on the same grid.

AFNI's ``3dTcorrelate``, generalised to N datasets: each voxel's time course in one
series is correlated with the SAME voxel's time course in another -- two echoes of
one run, two runs of one task, a series before and after a correction -- giving one
r-map per pair.

Each dataset's voxel chunk is detrended and normalised to unit length ONCE, after
which every pair is a row-wise dot product. N datasets therefore cost N
normalisations and ``N(N-1)/2`` dot products; at that point reading the data is the
expensive part, so all-pairs is no dearer than one pair in practice.

Drift is removed with Legendre polynomials (projected out with an orthonormal basis)
before correlating. Shared slow drift is the classic way two series correlate for
reasons that have nothing to do with their content, and echoes of one run share all
of it.

Caveat for anyone grading a correction with this: thermal noise is independent
across echoes and runs, and ANY resampling averages it down while leaving the shared
signal, so blur alone raises these correlations. Read a rise in r alongside a
sharpness measure, never on its own.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

from fastfuncstuff import policy_linalg as plinalg
from fastfuncstuff.memory import estimate_chunk_size


def pair_list(n: int, mode: str = "all") -> list[tuple[int, int]]:
    """Index pairs for ``n`` datasets: ``all`` (i<j), ``adjacent`` (i, i+1), ``ref:K``.

    ``ref:K`` pairs dataset K (1-based) with every other one, in input order.
    """
    if n < 2:
        raise ValueError(f"need at least 2 datasets to correlate, got {n}")
    if mode == "all":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if mode == "adjacent":
        return [(i, i + 1) for i in range(n - 1)]
    if mode.startswith("ref:"):
        try:
            k = int(mode[4:]) - 1
        except ValueError as exc:
            raise ValueError(f"-pairs ref:K needs an integer K, got {mode!r}") from exc
        if not 0 <= k < n:
            raise ValueError(f"-pairs ref:K needs 1 <= K <= {n}, got {k + 1}")
        return [(k, j) for j in range(n) if j != k]
    raise ValueError(f"pairs must be 'all', 'adjacent' or 'ref:K', got {mode!r}")


def _detrend_basis(n_t: int, polort: int, device: torch.device) -> torch.Tensor:
    """Orthonormal ``(T, polort+1)`` Legendre basis; ``polort < 0`` gives ``(T, 0)``."""
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    if polort < 0:
        return torch.zeros(n_t, 0, device=device)
    poly = construct_polynomial_matrix(n_t, polort, torch.device("cpu"), dtype=torch.float64)
    q, _ = plinalg.qr(poly)
    return q.float().to(device)


def _f_ordered(s) -> bool:
    """Whether a 4-D array or tensor is laid out x-fastest (Fortran / nibabel order)."""
    if torch.is_tensor(s):
        st = s.stride()
        return st[0] == 1 and st[0] <= st[1] <= st[2] <= st[3] and not s.is_contiguous()
    return bool(s.flags.f_contiguous) and not bool(s.flags.c_contiguous)


def paired_voxel_correlation(
    series: Sequence[np.ndarray | torch.Tensor],
    pairs: Sequence[tuple[int, int]] | None = None,
    *,
    polort: int = 0,
    mask: np.ndarray | torch.Tensor | None = None,
    fisher: bool = False,
    device: torch.device | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Per-voxel Pearson r between paired datasets, ``(nx, ny, nz, P)`` float32 on CPU.

    ``series`` are ``(nx, ny, nz, T)`` arrays on one grid with one T; they stay where
    they are (host memory) and are streamed through ``device`` in x-slabs, so neither
    an F-ordered nibabel array nor a long run is ever copied whole. ``pairs`` defaults
    to all ``i < j``. ``polort`` is the Legendre drift degree removed first (0 = the
    mean only, as a plain Pearson r; -1 = nothing). ``mask`` restricts the work to its
    voxels; elsewhere, and wherever either side is constant, r is 0. ``fisher`` returns
    ``atanh(r)``.
    """
    if device is None:
        from fastfuncstuff.utils import get_device

        device = get_device()
    n = len(series)
    pairs = list(pairs) if pairs is not None else pair_list(n, "all")
    if n < 2:
        raise ValueError(f"need at least 2 datasets to correlate, got {n}")
    shape = tuple(series[0].shape)
    if len(shape) != 4:
        raise ValueError(f"each dataset must be 4-D (nx, ny, nz, T), got {shape}")
    for k, s in enumerate(series):
        if tuple(s.shape) != shape:
            raise ValueError(f"dataset {k + 1} has shape {tuple(s.shape)}, dataset 1 {shape}")
    for a, b in pairs:
        if not (0 <= a < n and 0 <= b < n) or a == b:
            raise ValueError(f"bad pair ({a}, {b}) for {n} datasets")
    nx, ny, nz, n_t = shape
    if n_t < polort + 3:
        raise ValueError(f"{n_t} timepoints cannot carry a degree-{polort} detrend and an r")

    keep = None
    if mask is not None:
        keep = torch.as_tensor(np.asarray(mask) if not torch.is_tensor(mask) else mask) != 0
        if tuple(keep.shape) != (nx, ny, nz):
            raise ValueError(f"mask shape {tuple(keep.shape)} != grid {(nx, ny, nz)}")

    basis = _detrend_basis(n_t, polort, device)
    chunk = estimate_chunk_size(nx * ny * nz, n_t, 0, device, operation="paired_corr", n_trials=n)
    out = torch.zeros(nx, ny, nz, len(pairs))

    # Slab along the slowest spatial axis in memory.  A C-ordered (x, y, z, T)
    # array is cut along x, each slab one contiguous span.  A nibabel array is
    # F-ordered -- x fastest, T slowest -- and an x-slab of it is a fully strided
    # gather (0.77 s per 20 planes of a 250x246x68x246 run: 20 of the 21 s of one
    # pair); a z-slab is T contiguous runs, copied as (T, z, y, x) at memcpy speed.
    by_z = all(_f_ordered(s) for s in series)
    if by_z:
        n_cut, plane = nz, nx * ny
    else:
        n_cut, plane = nx, ny * nz
    slab = max(1, chunk // plane)

    def _slab(s, a0: int, a1: int) -> torch.Tensor:
        """``(voxels, T)`` on the device, voxels in the slab's memory order."""
        if by_z:
            block = s[:, :, a0:a1]
            # (T, z, y, x): every T step is one contiguous run of the source.
            block = block.permute(3, 2, 1, 0) if torch.is_tensor(block) else block.T
            if not torch.is_tensor(block):
                block = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))
            return block.reshape(n_t, -1).to(device=device, dtype=torch.float32).T
        block = s[a0:a1]
        if not torch.is_tensor(block):
            block = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))
        return block.reshape(-1, n_t).to(device=device, dtype=torch.float32)

    def _voxels(t: torch.Tensor, a0: int, a1: int) -> torch.Tensor:
        """A ``(..., P)`` slab result as ``(x, y, z, P)``, from the slab's voxel order."""
        if by_z:
            return t.reshape(a1 - a0, ny, nx, -1).permute(2, 1, 0, 3)
        return t.reshape(a1 - a0, ny, nz, -1)

    def _sel(a0: int, a1: int):
        if keep is None:
            return None
        k = keep[:, :, a0:a1].permute(2, 1, 0) if by_z else keep[a0:a1]
        return k.reshape(-1).to(device)

    for a0 in tqdm(
        range(0, n_cut, slab),
        desc="voxel correlation",
        unit="slab",
        leave=True,
        disable=not verbose or n_cut <= slab,
    ):
        a1 = min(n_cut, a0 + slab)
        sel = _sel(a0, a1)
        if sel is not None and not bool(sel.any()):
            continue
        unit = []
        for s in series:
            x = _slab(s, a0, a1)
            if sel is not None:
                x = x[sel]
            scale = x.norm(dim=1, keepdim=True)
            if basis.shape[1]:
                x = x - (x @ basis) @ basis.T
            norm = x.norm(dim=1, keepdim=True)
            # A constant voxel has no direction to correlate; r = 0 there, not NaN. The
            # test is RELATIVE: float32 detrending leaves ~1e-7 of a constant behind,
            # which would otherwise normalise to a unit vector of rounding error.
            flat = norm <= 1e-5 * scale
            unit.append(torch.where(flat, torch.zeros_like(x), x / norm.clamp_min(1e-30)))
        r = torch.stack([(unit[a] * unit[b]).sum(dim=1) for a, b in pairs], dim=1)
        r = r.clamp(-1.0, 1.0)
        if fisher:
            r = torch.atanh(r.clamp(-0.999999, 0.999999))
        r = r.cpu()
        if sel is not None:
            dst = torch.zeros(sel.numel(), len(pairs))
            dst[sel.cpu()] = r
            r = dst
        if by_z:
            out[:, :, a0:a1] = _voxels(r, a0, a1)
        else:
            out[a0:a1] = _voxels(r, a0, a1)
        del unit
    return out
