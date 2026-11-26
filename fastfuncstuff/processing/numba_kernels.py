"""Fused Numba kernels for the CPU: displacement-field work and separable filtering.

Imported lazily (Numba's import and first compile are not free), and only on the CPU
path: CUDA keeps the batched torch implementations, which are launch-bound in a way
this does not help.
"""

from __future__ import annotations

import functools
import platform

import numba
import numpy as np
import torch
from numba import prange
from torch import Tensor

njit = numba.njit

# torch's arm64 build compiles ATen with multiply-add contraction, so the float32
# arithmetic inside one fused ATen kernel (grid_sampler and its backward) rounds
# through FMA; the x86 wheels do not. LLVM only fuses when told to, so a kernel that
# replays grid_sampler to the bit has to match that per platform (measured: 7772 of
# 20000 trilinear samples one ULP off on an M-series without it, 0 with it). Kernels
# that replay a chain of separate torch elementwise ops (the Jacobian) must NOT
# contract: each of those ops rounds on its own.
_ATEN_CONTRACTS = platform.machine().lower() in {"arm64", "aarch64"}
njit_aten = functools.partial(numba.njit, fastmath={"contract"} if _ATEN_CONTRACTS else False)

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


@njit_aten(parallel=True, cache=True)
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


@njit(inline="always")
def _tap_index(p, n, zero_edges):
    # Replicate-clamped tap position, or -1 for a tap that falls off a zero-padded edge.
    if p < 0:
        return -1 if zero_edges else 0
    if p >= n:
        return -1 if zero_edges else n - 1
    return p


@njit(parallel=True, cache=True)
def _correlate_axis0(src, w, zero_edges, out):
    nz, ny, nx = src.shape
    r = w.shape[0] // 2
    for j in prange(ny):  # ty: ignore[not-iterable]
        acc = np.empty(nx, np.float64)
        for k in range(nz):
            acc[:] = 0.0
            for t in range(w.shape[0]):
                kk = _tap_index(k + t - r, nz, zero_edges)
                if kk >= 0:
                    wt = w[t]
                    for i in range(nx):
                        acc[i] += wt * src[kk, j, i]
            for i in range(nx):
                out[k, j, i] = acc[i]


@njit(parallel=True, cache=True)
def _correlate_axis1(src, w, zero_edges, out):
    nz, ny, nx = src.shape
    r = w.shape[0] // 2
    for k in prange(nz):  # ty: ignore[not-iterable]
        acc = np.empty(nx, np.float64)
        for j in range(ny):
            acc[:] = 0.0
            for t in range(w.shape[0]):
                jj = _tap_index(j + t - r, ny, zero_edges)
                if jj >= 0:
                    wt = w[t]
                    for i in range(nx):
                        acc[i] += wt * src[k, jj, i]
            for i in range(nx):
                out[k, j, i] = acc[i]


@njit(parallel=True, cache=True)
def _correlate_axis2(src, w, zero_edges, out):
    nz, ny, nx = src.shape
    r = w.shape[0] // 2
    for k in prange(nz):  # ty: ignore[not-iterable]
        buf = np.empty(nx + 2 * r, np.float64)
        acc = np.empty(nx, np.float64)
        for j in range(ny):
            for i in range(nx):
                buf[i + r] = src[k, j, i]
            for i in range(r):
                buf[i] = 0.0 if zero_edges else src[k, j, 0]
                buf[nx + r + i] = 0.0 if zero_edges else src[k, j, nx - 1]
            acc[:] = 0.0
            for t in range(w.shape[0]):
                wt = w[t]
                for i in range(nx):
                    acc[i] += wt * buf[i + t]
            for i in range(nx):
                out[k, j, i] = acc[i]


_CORRELATE_AXIS = (_correlate_axis0, _correlate_axis1, _correlate_axis2)


def correlate1d_cpu(arr: np.ndarray, weights: np.ndarray, axis: int, zero_edges: bool):
    """1-D correlation of each (D, H, W) volume of a (G, D, H, W) array along ``axis``
    (0..2 of the volume), edges replicated or zero. Accumulated in double, in tap order:
    the same numbers as ``scipy.ndimage.correlate1d`` (mode 'nearest' / 'constant').

    Each output row is a weighted sum of whole input rows on the two strided axes, and
    of shifted copies of one buffered row on the contiguous one, so the inner loop is a
    plain vector multiply-add: 13 ms per axis at 193x229x193 on 6 threads, against 25 ms
    for ndimage threaded over slabs (which walks every line through a scalar buffer).
    """
    out = np.empty_like(arr)
    w = np.ascontiguousarray(weights, dtype=np.float64)
    kernel = _CORRELATE_AXIS[axis]
    prev = numba.get_num_threads()
    numba.set_num_threads(max(1, min(torch.get_num_threads(), prev)))
    try:
        for g in range(arr.shape[0]):
            kernel(arr[g], w, zero_edges, out[g])
    finally:
        numba.set_num_threads(prev)
    return out


@njit(inline="always")
def _border_index_grad(coord, size):
    # _border_index plus d(index)/d(coord) as ATen and autograd compose it: the
    # unnormalise factor (size-1)/2 times the clip gate (0 at or beyond either face),
    # then autograd's chain back through the caller's ``2*x/(size-1) - 1``.
    if size <= 1:
        return _F0, _F0
    g = (coord * _F2) / np.float32(size - 1) - _F1
    ix = ((g + _F1) / _F2) * np.float32(size - 1)
    lim = np.float32(size - 1)
    mult = lim / _F2
    if ix <= _F0:
        return _F0, _F0
    if ix >= lim:
        return lim, _F0
    return ix, mult


@njit_aten(parallel=True, cache=True)
def _warp_linear_fwd(src, xd, yd, zd, out):
    nz, ny, nx = src.shape
    for k in prange(nz):  # ty: ignore[not-iterable]
        kf = np.float32(k)
        for j in range(ny):
            jf = np.float32(j)
            for i in range(nx):
                ix = _border_index(np.float32(i) + xd[k, j, i], nx)
                iy = _border_index(jf + yd[k, j, i], ny)
                iz = _border_index(kf + zd[k, j, i], nz)
                x0 = np.floor(ix)
                y0 = np.floor(iy)
                z0 = np.floor(iz)
                x1 = x0 + _F1
                y1 = y0 + _F1
                z1 = z0 + _F1
                i0, j0, k0 = int(x0), int(y0), int(z0)
                i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
                oi, oj, ok = i1 < nx, j1 < ny, k1 < nz
                a = src[k0, j0, i0] * ((x1 - ix) * (y1 - iy) * (z1 - iz))
                if oi:
                    a += src[k0, j0, i1] * ((ix - x0) * (y1 - iy) * (z1 - iz))
                if oj:
                    a += src[k0, j1, i0] * ((x1 - ix) * (iy - y0) * (z1 - iz))
                if oi and oj:
                    a += src[k0, j1, i1] * ((ix - x0) * (iy - y0) * (z1 - iz))
                if ok:
                    a += src[k1, j0, i0] * ((x1 - ix) * (y1 - iy) * (iz - z0))
                if ok and oi:
                    a += src[k1, j0, i1] * ((ix - x0) * (y1 - iy) * (iz - z0))
                if ok and oj:
                    a += src[k1, j1, i0] * ((x1 - ix) * (iy - y0) * (iz - z0))
                if ok and oi and oj:
                    a += src[k1, j1, i1] * ((ix - x0) * (iy - y0) * (iz - z0))
                out[k, j, i] = a


@njit_aten(parallel=True, cache=True)
def _warp_linear_bwd(src, xd, yd, zd, gout, gx_out, gy_out, gz_out):
    nz, ny, nx = src.shape
    sx2 = np.float32(nx - 1)
    sy2 = np.float32(ny - 1)
    sz2 = np.float32(nz - 1)
    for k in prange(nz):  # ty: ignore[not-iterable]
        kf = np.float32(k)
        for j in range(ny):
            jf = np.float32(j)
            for i in range(nx):
                ix, mx = _border_index_grad(np.float32(i) + xd[k, j, i], nx)
                iy, my = _border_index_grad(jf + yd[k, j, i], ny)
                iz, mz = _border_index_grad(kf + zd[k, j, i], nz)
                go = gout[k, j, i]
                x0 = np.floor(ix)
                y0 = np.floor(iy)
                z0 = np.floor(iz)
                x1 = x0 + _F1
                y1 = y0 + _F1
                z1 = z0 + _F1
                i0, j0, k0 = int(x0), int(y0), int(z0)
                i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
                oi, oj, ok = i1 < nx, j1 < ny, k1 < nz
                gix = _F0
                giy = _F0
                giz = _F0
                # ATen grid_sampler_3d_backward's corner terms, in its order.
                v = src[k0, j0, i0]
                gix -= v * (y1 - iy) * (z1 - iz) * go
                giy -= v * (x1 - ix) * (z1 - iz) * go
                giz -= v * (x1 - ix) * (y1 - iy) * go
                if oi:
                    v = src[k0, j0, i1]
                    gix += v * (y1 - iy) * (z1 - iz) * go
                    giy -= v * (ix - x0) * (z1 - iz) * go
                    giz -= v * (ix - x0) * (y1 - iy) * go
                if oj:
                    v = src[k0, j1, i0]
                    gix -= v * (iy - y0) * (z1 - iz) * go
                    giy += v * (x1 - ix) * (z1 - iz) * go
                    giz -= v * (x1 - ix) * (iy - y0) * go
                if oi and oj:
                    v = src[k0, j1, i1]
                    gix += v * (iy - y0) * (z1 - iz) * go
                    giy += v * (ix - x0) * (z1 - iz) * go
                    giz -= v * (ix - x0) * (iy - y0) * go
                if ok:
                    v = src[k1, j0, i0]
                    gix -= v * (y1 - iy) * (iz - z0) * go
                    giy -= v * (x1 - ix) * (iz - z0) * go
                    giz += v * (x1 - ix) * (y1 - iy) * go
                if ok and oi:
                    v = src[k1, j0, i1]
                    gix += v * (y1 - iy) * (iz - z0) * go
                    giy -= v * (ix - x0) * (iz - z0) * go
                    giz += v * (ix - x0) * (y1 - iy) * go
                if ok and oj:
                    v = src[k1, j1, i0]
                    gix -= v * (iy - y0) * (iz - z0) * go
                    giy += v * (x1 - ix) * (iz - z0) * go
                    giz += v * (x1 - ix) * (iy - y0) * go
                if ok and oi and oj:
                    v = src[k1, j1, i1]
                    gix += v * (iy - y0) * (iz - z0) * go
                    giy += v * (ix - x0) * (iz - z0) * go
                    giz += v * (ix - x0) * (iy - y0) * go
                gx_out[k, j, i] = ((mx * gix) / sx2) * _F2 if nx > 1 else _F0
                gy_out[k, j, i] = ((my * giy) / sy2) * _F2 if ny > 1 else _F0
                gz_out[k, j, i] = ((mz * giz) / sz2) * _F2 if nz > 1 else _F0


def _with_threads(fn, *args):
    prev = numba.get_num_threads()
    numba.set_num_threads(max(1, min(torch.get_num_threads(), prev)))
    try:
        fn(*args)
    finally:
        numba.set_num_threads(prev)


class WarpLinearCPU(torch.autograd.Function):
    """``out[p] = src[p + d(p)]``, trilinear with border clamping, differentiable in d.

    The numbers of ``formwarp._warp_diff`` (grid normalisation, ATen's sampler and its
    grid backward, autograd's chain back through the normalisation) in one pass each
    way, without the stacked grid, the lane split or the chain's temporaries.
    """

    @staticmethod
    def forward(ctx, src: Tensor, xd: Tensor, yd: Tensor, zd: Tensor) -> Tensor:
        arrays = [t.detach().contiguous().numpy() for t in (src, xd, yd, zd)]
        out = np.empty_like(arrays[0])
        _with_threads(_warp_linear_fwd, *arrays, out)
        ctx.arrays = arrays
        return torch.from_numpy(out)

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor):
        (grad,) = grad_outputs
        g = grad.detach().contiguous().numpy()
        outs = [np.empty_like(g) for _ in range(3)]
        _with_threads(_warp_linear_bwd, *ctx.arrays, g, *outs)
        return None, *(torch.from_numpy(o) for o in outs)


@njit(parallel=True, cache=True)
def _moments5_kernel(a, b, w, out):
    nz, ny, nx = a.shape
    for k in prange(nz):  # ty: ignore[not-iterable]
        for j in range(ny):
            for i in range(nx):
                wv = w[k, j, i]
                av = a[k, j, i]
                bv = b[k, j, i]
                wa = wv * av
                wb = wv * bv
                out[0, k, j, i] = wa
                out[1, k, j, i] = wb
                out[2, k, j, i] = wa * av
                out[3, k, j, i] = wb * bv
                out[4, k, j, i] = wa * bv


def moments5_cpu(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``w a, w b, w a a, w b b, w a b`` stacked, in torch's left-to-right op order."""
    out = np.empty((5, *a.shape), dtype=a.dtype)
    _with_threads(_moments5_kernel, a, b, w, out)
    return out


@njit(parallel=True, cache=True)
def _local_corr_kernel(s0, s, w, mode, want_grad, grads, part_total, part_w):
    nz, ny, nx = s0.shape
    if mode == 0:  # lpa: cost.py _local_pearson_from_moments
        eps = np.float32(1e-10)
    else:  # cc: formwarp._local_cc_cost
        eps = np.float32(1e-6)
    two = np.float32(2.0)
    for k in prange(nz):  # ty: ignore[not-iterable]
        tot = 0.0
        wt = 0.0
        for j in range(ny):
            for i in range(nx):
                wv = w[k, j, i]
                sw = max(s0[k, j, i], eps)
                mx = s[0, k, j, i] / sw
                my = s[1, k, j, i] / sw
                a0 = s[2, k, j, i] / sw - mx * mx
                b0 = s[3, k, j, i] / sw - my * my
                av = max(a0, eps)
                bv = max(b0, eps)
                c = s[4, k, j, i] / sw - mx * my
                q = np.sqrt(av * bv)
                r = c / q
                if mode == 0:
                    phi = abs(r)
                else:
                    phi = c * c / (av * bv)
                tot += wv * phi
                wt += wv
                if want_grad:
                    if mode == 0:
                        sgn = _F1 if r > _F0 else (-_F1 if r < _F0 else _F0)
                        gr = wv * sgn
                        gc = gr / q
                        ga = -gr * r / (two * av) if a0 >= eps else _F0
                        gb = -gr * r / (two * bv) if b0 >= eps else _F0
                    else:
                        gc = wv * two * c / (av * bv)
                        ga = -wv * phi / av if a0 >= eps else _F0
                        gb = -wv * phi / bv if b0 >= eps else _F0
                    grads[0, k, j, i] = (ga * (-two * mx) - gc * my) / sw
                    grads[1, k, j, i] = (gb * (-two * my) - gc * mx) / sw
                    grads[2, k, j, i] = ga / sw
                    grads[3, k, j, i] = gb / sw
                    grads[4, k, j, i] = gc / sw
        part_total[k] = tot
        part_w[k] = wt


def local_corr_cpu(s0: np.ndarray, s: np.ndarray, w: np.ndarray, mode: int, want_grad: bool):
    """Sum of ``w * phi`` and of ``w`` over the volume, plus d(w phi)/d(filtered moment)
    for the five moving moments when ``want_grad`` (phi = |r| for mode 0, r^2 for 1)."""
    nz = s0.shape[0]
    grads = np.empty((5, *s0.shape) if want_grad else (5, 1, 1, 1), dtype=s0.dtype)
    part_total = np.zeros(nz)
    part_w = np.zeros(nz)
    _with_threads(_local_corr_kernel, s0, s, w, mode, want_grad, grads, part_total, part_w)
    return float(part_total.sum()), float(part_w.sum()), grads


@njit(parallel=True, cache=True)
def _combine_grad_kernel(a, b, w, h, scale, ga, gb):
    nz, ny, nx = a.shape
    two = np.float32(2.0)
    for k in prange(nz):  # ty: ignore[not-iterable]
        for j in range(ny):
            for i in range(nx):
                av = a[k, j, i]
                bv = b[k, j, i]
                ws = w[k, j, i] * scale
                ga[k, j, i] = ws * (h[0, k, j, i] + two * av * h[2, k, j, i] + bv * h[4, k, j, i])
                gb[k, j, i] = ws * (h[1, k, j, i] + two * bv * h[3, k, j, i] + av * h[4, k, j, i])


def combine_local_corr_grad_cpu(a, b, w, h, scale: float):
    """Chain the adjoint-filtered moment gradients ``h`` back to the two images."""
    ga = np.empty_like(a)
    gb = np.empty_like(a)
    _with_threads(_combine_grad_kernel, a, b, w, h, np.float32(scale), ga, gb)
    return ga, gb


@njit(inline="always")
def _corner_weights(ix, iy, iz, nx, ny, nz):
    # ATen grid_sampler_3d's trilinear corner indices and weights, from a border-
    # clipped unnormalised position, in its expression order.
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
    return i0, j0, k0, i0 + 1 < nx, j0 + 1 < ny, k0 + 1 < nz, tnw, tne, tsw, tse, bnw, bne, bsw, bse


@njit(inline="always")
def _sample_planar(v, c, i0, j0, k0, oi, oj, ok, tnw, tne, tsw, tse, bnw, bne, bsw, bse):
    # One channel of a (C, nz, ny, nx) volume, accumulated in ATen's corner order.
    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
    a = _F0
    a += v[c, k0, j0, i0] * tnw
    if oi:
        a += v[c, k0, j0, i1] * tne
    if oj:
        a += v[c, k0, j1, i0] * tsw
    if oi and oj:
        a += v[c, k0, j1, i1] * tse
    if ok:
        a += v[c, k1, j0, i0] * bnw
    if ok and oi:
        a += v[c, k1, j0, i1] * bne
    if ok and oj:
        a += v[c, k1, j1, i0] * bsw
    if ok and oi and oj:
        a += v[c, k1, j1, i1] * bse
    return a


@njit_aten(parallel=True, cache=True)
def _compose_sample_kernel(
    src, warp, px, py, pz, bi, bj, bk, slo, sxhi, syhi, szhi, grad, want_grad, out, gout
):
    nz, ny, nx = src.shape[1], src.shape[2], src.shape[3]
    fx, fy, fz = np.float32(nx - 1), np.float32(ny - 1), np.float32(nz - 1)
    n = px.shape[0]
    block = 4096
    for blk in prange((n + block - 1) // block):  # ty: ignore[not-iterable]
        for p in range(blk * block, min(n, (blk + 1) * block)):
            hx, hy, hz = px[p], py[p], pz[p]
            # The query point, clamped as the torch path clamps before normalising.
            xq = min(max(bi[p] + hx, _F0), fx)
            yq = min(max(bj[p] + hy, _F0), fy)
            zq = min(max(bk[p] + hz, _F0), fz)
            i0, j0, k0, oi, oj, ok, tnw, tne, tsw, tse, bnw, bne, bsw, bse = _corner_weights(
                _border_index(xq, nx), _border_index(yq, ny), _border_index(zq, nz), nx, ny, nz
            )
            ax = hx + _sample_planar(
                warp, 0, i0, j0, k0, oi, oj, ok, tnw, tne, tsw, tse, bnw, bne, bsw, bse
            )
            ay = hy + _sample_planar(
                warp, 1, i0, j0, k0, oi, oj, ok, tnw, tne, tsw, tse, bnw, bne, bsw, bse
            )
            az = hz + _sample_planar(
                warp, 2, i0, j0, k0, oi, oj, ok, tnw, tne, tsw, tse, bnw, bne, bsw, bse
            )
            sx = min(max(ax + bi[p], slo), sxhi)
            sy = min(max(ay + bj[p], slo), syhi)
            sz = min(max(az + bk[p], slo), szhi)
            i0, j0, k0, oi, oj, ok, tnw, tne, tsw, tse, bnw, bne, bsw, bse = _corner_weights(
                _border_index(sx, nx), _border_index(sy, ny), _border_index(sz, nz), nx, ny, nz
            )
            out[0, p] = _sample_planar(
                src, 0, i0, j0, k0, oi, oj, ok, tnw, tne, tsw, tse, bnw, bne, bsw, bse
            )
            out[1, p] = ax
            out[2, p] = ay
            out[3, p] = az
            if want_grad:
                gout[0, p] = _sample_planar(
                    grad, 0, i0, j0, k0, oi, oj, ok, tnw, tne, tsw, tse, bnw, bne, bsw, bse
                )
                gout[1, p] = _sample_planar(
                    grad, 1, i0, j0, k0, oi, oj, ok, tnw, tne, tsw, tse, bnw, bne, bsw, bse
                )
                gout[2, p] = _sample_planar(
                    grad, 2, i0, j0, k0, oi, oj, ok, tnw, tne, tsw, tse, bnw, bne, bsw, bse
                )


def compose_and_sample_cpu(
    source: Tensor,
    warp_3ch: Tensor,
    patch_xd: Tensor,
    patch_yd: Tensor,
    patch_zd: Tensor,
    base_i: Tensor,
    base_j: Tensor,
    base_k: Tensor,
    grad_3ch: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """qwarp's compose-and-sample, ``interp.batched_compose_and_interpolate``, in one pass.

    Per patch voxel: sample the global warp at ``base + h``, add ``h``, sample the
    source at the composed position -- and, given ``grad_3ch``, the source gradient
    at that same position, which Gauss-Newton wants next. Bit-identical to the
    grid_sample route: same clamps, same normalise/unnormalise round trip, same
    corner order. Returns (B, V) warped values and composed displacements, plus the
    (3, B, V) gradient sample or None.
    """
    b, v = patch_xd.shape
    nz, ny, nx = source.shape
    flat = [t.detach().contiguous().reshape(-1).numpy() for t in (patch_xd, patch_yd, patch_zd)]
    bases = [t.contiguous().reshape(-1).numpy() for t in (base_i, base_j, base_k)]
    want = grad_3ch is not None
    out = np.empty((4, b * v), dtype=np.float32)
    gout = np.empty((3, b * v) if want else (3, 1), dtype=np.float32)
    grad = grad_3ch.contiguous().numpy() if want else np.empty((3, 1, 1, 1), np.float32)
    _with_threads(
        _compose_sample_kernel,
        source.contiguous().numpy()[None],
        warp_3ch.contiguous().numpy(),
        *flat,
        *bases,
        np.float32(-0.499),
        np.float32(nx - 0.501),
        np.float32(ny - 0.501),
        np.float32(nz - 0.501),
        grad,
        want,
        out,
        gout,
    )
    res = torch.from_numpy(out).reshape(4, b, v)
    g = torch.from_numpy(gout).reshape(3, b, v) if want else None
    return res[0], res[1], res[2], res[3], g
