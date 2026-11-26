"""Fused Numba kernels for displacement-field work on the CPU.

Imported lazily (Numba's import and first compile are not free), and only on the CPU
path: CUDA keeps the batched torch implementations, which are launch-bound in a way
this does not help.
"""

from __future__ import annotations

import numba
import numpy as np
import torch
from numba import njit, prange
from torch import Tensor

_F1 = np.float32(1.0)
_F2 = np.float32(2.0)
_F0 = np.float32(0.0)


@njit(inline="always")
def _border_index(coord, size):
    # grid_sampler_compute_source_index for align_corners=True, padding 'border':
    # normalise exactly as the torch caller does, unnormalise as ATen does, then clip.
    if size > 1:
        g = (coord * _F2) / np.float32(size - 1) - _F1
    else:
        g = coord * _F0
    ix = ((g + _F1) / _F2) * np.float32(size - 1)
    return min(np.float32(size - 1), max(ix, _F0))


@njit(inline="always")
def _trilinear3(f, x, y, z):
    """ATen's scalar 3-D trilinear (border, align_corners) on an interleaved
    (nz, ny, nx, 3) field: one cache line per corner instead of three."""
    nz, ny, nx = f.shape[0], f.shape[1], f.shape[2]
    ix = _border_index(x, nx)
    iy = _border_index(y, ny)
    iz = _border_index(z, nz)
    x0 = np.floor(ix)
    y0 = np.floor(iy)
    z0 = np.floor(iz)
    x1 = x0 + _F1
    y1 = y0 + _F1
    z1 = z0 + _F1
    tnw = (x1 - ix) * (y1 - iy) * (z1 - iz)
    tne = (ix - x0) * (y1 - iy) * (z1 - iz)
    tsw = (x1 - ix) * (iy - y0) * (z1 - iz)
    tse = (ix - x0) * (iy - y0) * (z1 - iz)
    bnw = (x1 - ix) * (y1 - iy) * (iz - z0)
    bne = (ix - x0) * (y1 - iy) * (iz - z0)
    bsw = (x1 - ix) * (iy - y0) * (iz - z0)
    bse = (ix - x0) * (iy - y0) * (iz - z0)
    i0, j0, k0 = int(x0), int(y0), int(z0)
    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
    ok_i1, ok_j1, ok_k1 = i1 < nx, j1 < ny, k1 < nz
    a = _F0
    b = _F0
    c = _F0
    # Accumulated in ATen's corner order so the float32 sums round identically.
    a += f[k0, j0, i0, 0] * tnw
    b += f[k0, j0, i0, 1] * tnw
    c += f[k0, j0, i0, 2] * tnw
    if ok_i1:
        a += f[k0, j0, i1, 0] * tne
        b += f[k0, j0, i1, 1] * tne
        c += f[k0, j0, i1, 2] * tne
    if ok_j1:
        a += f[k0, j1, i0, 0] * tsw
        b += f[k0, j1, i0, 1] * tsw
        c += f[k0, j1, i0, 2] * tsw
    if ok_i1 and ok_j1:
        a += f[k0, j1, i1, 0] * tse
        b += f[k0, j1, i1, 1] * tse
        c += f[k0, j1, i1, 2] * tse
    if ok_k1:
        a += f[k1, j0, i0, 0] * bnw
        b += f[k1, j0, i0, 1] * bnw
        c += f[k1, j0, i0, 2] * bnw
    if ok_k1 and ok_i1:
        a += f[k1, j0, i1, 0] * bne
        b += f[k1, j0, i1, 1] * bne
        c += f[k1, j0, i1, 2] * bne
    if ok_k1 and ok_j1:
        a += f[k1, j1, i0, 0] * bsw
        b += f[k1, j1, i0, 1] * bsw
        c += f[k1, j1, i0, 2] * bsw
    if ok_k1 and ok_i1 and ok_j1:
        a += f[k1, j1, i1, 0] * bse
        b += f[k1, j1, i1, 1] * bse
        c += f[k1, j1, i1, 2] * bse
    return a, b, c


@njit(parallel=True, cache=True)
def _invert_field_kernel(f, n_iter, ex, ey, ez, step_per_slice):
    nz, ny, nx = f.shape[0], f.shape[1], f.shape[2]
    for k in prange(nz):  # ty: ignore[not-iterable]
        worst = _F0
        kf = np.float32(k)
        for j in range(ny):
            jf = np.float32(j)
            for i in range(nx):
                fi = np.float32(i)
                e0 = -f[k, j, i, 0]
                e1 = -f[k, j, i, 1]
                e2 = -f[k, j, i, 2]
                for sweep in range(n_iter):
                    d0, d1, d2 = _trilinear3(f, fi + e0, jf + e1, kf + e2)
                    if sweep == n_iter - 1:
                        worst = max(worst, abs(-d0 - e0), abs(-d1 - e1), abs(-d2 - e2))
                    e0 = -d0
                    e1 = -d1
                    e2 = -d2
                ex[k, j, i] = e0
                ey[k, j, i] = e1
                ez[k, j, i] = e2
        step_per_slice[k] = worst


def invert_displacement_field_cpu(
    xd: Tensor, yd: Tensor, zd: Tensor, n_iter: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``e(x) = -d(x + e(x))`` by fixed-point iteration, every sweep fused per voxel.

    Each voxel's iterate depends only on its own previous value (the field it samples,
    ``d``, never changes), so the sweeps need no whole-volume barrier between them: one
    pass runs all ``n_iter`` per voxel while its neighbourhood is in cache. Same float32
    arithmetic as the torch sweep through ``grid_sample``.
    """
    if xd.dtype != torch.float32:
        raise TypeError("invert_displacement_field_cpu is float32-only")
    field = torch.stack([t.detach() for t in (xd, yd, zd)], dim=-1).numpy()
    out = [np.empty(field.shape[:3], dtype=np.float32) for _ in range(3)]
    steps = np.zeros(field.shape[0], dtype=np.float32)
    prev = numba.get_num_threads()
    numba.set_num_threads(max(1, min(torch.get_num_threads(), prev)))
    try:
        _invert_field_kernel(field, int(n_iter), *out, steps)
    finally:
        numba.set_num_threads(prev)
    ex, ey, ez = (torch.from_numpy(a) for a in out)
    return ex, ey, ez, torch.tensor(float(steps.max()) if steps.size else 0.0)


_HALF = np.float32(0.5)


@njit(inline="always")
def _cdiff(v, k, j, i, axis, min_n):
    # Central difference, one-sided on the faces, in float32 order. Zero on an axis
    # shorter than ``min_n``: 3 for formwarp._grad3, 2 for penalty._central_diff_batched
    # (which takes a one-sided difference on a 2-voxel axis where _grad3 gives zero).
    n = v.shape[axis]
    if axis == 0:
        if n < min_n:
            return _F0
        if k == 0:
            return v[1, j, i] - v[0, j, i]
        if k == n - 1:
            return v[n - 1, j, i] - v[n - 2, j, i]
        return _HALF * (v[k + 1, j, i] - v[k - 1, j, i])
    if axis == 1:
        if n < min_n:
            return _F0
        if j == 0:
            return v[k, 1, i] - v[k, 0, i]
        if j == n - 1:
            return v[k, n - 1, i] - v[k, n - 2, i]
        return _HALF * (v[k, j + 1, i] - v[k, j - 1, i])
    if n < min_n:
        return _F0
    if i == 0:
        return v[k, j, 1] - v[k, j, 0]
    if i == n - 1:
        return v[k, j, n - 1] - v[k, j, n - 2]
    return _HALF * (v[k, j, i + 1] - v[k, j, i - 1])


@njit(parallel=True, cache=True)
def _jacobian_kernel(xd, yd, zd, cut, want_owing, det_out, owing_out):
    nz, ny, nx = xd.shape
    min_n = 2 if want_owing else 3
    for k in prange(nz):  # ty: ignore[not-iterable]
        for j in range(ny):
            for i in range(nx):
                a11 = _F1 + _cdiff(xd, k, j, i, 2, min_n)
                a12 = _cdiff(xd, k, j, i, 1, min_n)
                a13 = _cdiff(xd, k, j, i, 0, min_n)
                a21 = _cdiff(yd, k, j, i, 2, min_n)
                a22 = _F1 + _cdiff(yd, k, j, i, 1, min_n)
                a23 = _cdiff(yd, k, j, i, 0, min_n)
                a31 = _cdiff(zd, k, j, i, 2, min_n)
                a32 = _cdiff(zd, k, j, i, 1, min_n)
                a33 = _F1 + _cdiff(zd, k, j, i, 0, min_n)
                det = (
                    a11 * (a22 * a33 - a23 * a32)
                    - a12 * (a21 * a33 - a23 * a31)
                    + a13 * (a21 * a32 - a22 * a31)
                )
                det_out[k, j, i] = det
                if want_owing:
                    dm = det - _F1
                    je = dm * dm
                    s1 = _HALF * (a12 + a21)
                    s2 = _HALF * (a13 + a31)
                    s3 = _HALF * (a23 + a32)
                    s4 = _HALF * (a12 - a21)
                    s5 = _HALF * (a13 - a31)
                    s6 = _HALF * (a23 - a32)
                    d1 = a11 - _F1
                    d2 = a22 - _F1
                    d3 = a33 - _F1
                    se = (
                        s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4 + s5 * s5 + s6 * s6
                        + _HALF * (d1 * d1 + d2 * d2 + d3 * d3)
                    )  # fmt: skip
                    ej = max(je - cut, _F0)
                    es = max(se - cut, _F0)
                    owing_out[k, j, i] = (ej * ej * ej * ej + es * es * es * es) > _F0


def jacobian_cpu(xd: Tensor, yd: Tensor, zd: Tensor, cut: float | None = None):
    """det(I + grad d) per voxel in one fused pass; with ``cut``, the voxels whose
    :func:`penalty.penalty_energy` is positive (the fold penalty's offenders) instead,
    and the determinant then follows the penalty's stencil (see ``_cdiff``).

    The torch version allocates nine derivative volumes and a few dozen temporaries for
    what is a 27-point stencil per voxel: 0.33 s against tens of ms here at 193x229x193.
    """
    arrays = [t.detach().contiguous().numpy() for t in (xd, yd, zd)]
    if arrays[0].dtype != np.float32:
        raise TypeError("jacobian_cpu is float32-only")
    det = np.empty_like(arrays[0])
    owing = np.empty(arrays[0].shape if cut is not None else (1, 1, 1), dtype=np.bool_)
    prev = numba.get_num_threads()
    numba.set_num_threads(max(1, min(torch.get_num_threads(), prev)))
    try:
        _jacobian_kernel(
            *arrays, np.float32(0.0 if cut is None else cut), cut is not None, det, owing
        )
    finally:
        numba.set_num_threads(prev)
    return torch.from_numpy(det), (torch.from_numpy(owing) if cut is not None else None)
