"""Exact nearest-face search on CUDA, a brick of query points at a time.

The CPU path asks a KD-tree for each point's 24 nearest face centroids and
measured ~1 us per point -- most of a GPU run, with the GPU idle. Ribbon
voxels come in dense clumps whose nearest faces are nearly the same, so the
work is shared per *brick*: every face that could be nearest to any point of
the brick is found by a bound, and each such face is loaded once and tested
against all of the brick's points.

The bound is exact, which the per-point KD shortcut is not. With brick centre
``c``, every point within ``h`` of it, and ``d_c`` an upper bound on the
centre's distance to the mesh, a point's nearest distance is at most
``d_c + h``, so its nearest point ``q`` lies within ``d_c + 2h`` of ``c``. Each
face is listed in every cell of a 1 mm grid that its bounding box touches, so
``q``'s cell lists the face, and a cell farther than that from ``c`` can be
skipped outright -- with no allowance for face size, which for pial's few
large faces had made every brick scan a cube 3 mm wider. The bound tightens as
the brick's points find closer faces. (Sharing *candidates* per
brick without a bound missed the opposite bank of tight sulci by 0.1-0.8 mm.)
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from fastfuncstuff.triton_key import install_triton_key_cache

install_triton_key_cache()

BLOCK_V = 64
BLOCK_F = 32


@triton.jit
def _closest_d2(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    """Squared distance from points to triangles (Ericson 5.1.5), broadcast."""
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    acx = cx - ax
    acy = cy - ay
    acz = cz - az
    apx = px - ax
    apy = py - ay
    apz = pz - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    bpx = px - bx
    bpy = py - by
    bpz = pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    cpx = px - cx
    cpy = py - cy
    cpz = pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    tiny = 1e-30
    den = va + vb + vc
    den = tl.where(tl.abs(den) > tiny, den, tiny)
    v = vb / den
    w = vc / den
    u = 1.0 - v - w
    # Least to most specific region; later assignments win.
    t_den = (d4 - d3) + (d5 - d6)
    t = (d4 - d3) / tl.where(tl.abs(t_den) > tiny, t_den, tiny)
    m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
    u = tl.where(m, 0.0, u)
    v = tl.where(m, 1.0 - t, v)
    w = tl.where(m, t, w)
    t_den = d2 - d6
    t = d2 / tl.where(tl.abs(t_den) > tiny, t_den, tiny)
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    u = tl.where(m, 1.0 - t, u)
    v = tl.where(m, 0.0, v)
    w = tl.where(m, t, w)
    t_den = d1 - d3
    t = d1 / tl.where(tl.abs(t_den) > tiny, t_den, tiny)
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    u = tl.where(m, 1.0 - t, u)
    v = tl.where(m, t, v)
    w = tl.where(m, 0.0, w)
    m = (d6 >= 0) & (d5 <= d6)
    u = tl.where(m, 0.0, u)
    v = tl.where(m, 0.0, v)
    w = tl.where(m, 1.0, w)
    m = (d3 >= 0) & (d4 <= d3)
    u = tl.where(m, 0.0, u)
    v = tl.where(m, 1.0, v)
    w = tl.where(m, 0.0, w)
    m = (d1 <= 0) & (d2 <= 0)
    u = tl.where(m, 1.0, u)
    v = tl.where(m, 0.0, v)
    w = tl.where(m, 0.0, w)
    qx = u * ax + v * bx + w * cx - px
    qy = u * ay + v * by + w * cy - py
    qz = u * az + v * bz + w * cz - pz
    return qx * qx + qy * qy + qz * qz


@triton.jit
def _brick_nearest_kernel(
    pts_ptr,  # (N, 3) float32, grouped by segment
    seg_start_ptr,
    seg_count_ptr,
    seg_brick_ptr,
    centre_ptr,  # (B, 3) float32
    bound_ptr,  # (B,) float32: d_c + 2h
    tri_ptr,  # (P, 9) float32: one row per (cell, face) entry, sorted by cell
    face_id_ptr,  # (P,) int32: the face of each entry
    cell_ptr_ptr,  # (n_cells + 1,) int32
    gx0,
    gy0,
    gz0,
    gs,
    gnx,
    gny,
    gnz,
    half_diag,
    out_d2_ptr,
    out_face_ptr,
    BLOCK_V: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)
    start = tl.load(seg_start_ptr + pid)
    count = tl.load(seg_count_ptr + pid)
    b = tl.load(seg_brick_ptr + pid)
    ox = tl.load(centre_ptr + 3 * b)
    oy = tl.load(centre_ptr + 3 * b + 1)
    oz = tl.load(centre_ptr + 3 * b + 2)
    bound = tl.load(bound_ptr + b)

    rv = tl.arange(0, BLOCK_V)
    vm = rv < count
    # Local frame at the brick centre: float32 keeps sub-micron precision.
    px = (tl.load(pts_ptr + 3 * (start + rv), mask=vm, other=0.0) - ox)[:, None]
    py = (tl.load(pts_ptr + 3 * (start + rv) + 1, mask=vm, other=0.0) - oy)[:, None]
    pz = (tl.load(pts_ptr + 3 * (start + rv) + 2, mask=vm, other=0.0) - oz)[:, None]
    best = tl.full((BLOCK_V,), float("inf"), tl.float32)
    best_f = tl.zeros((BLOCK_V,), tl.int32)

    reach = bound
    cap = bound
    i0 = tl.maximum(((ox - reach - gx0) / gs).to(tl.int32), 0)
    i1 = tl.minimum(((ox + reach - gx0) / gs).to(tl.int32), gnx - 1)
    j0 = tl.maximum(((oy - reach - gy0) / gs).to(tl.int32), 0)
    j1 = tl.minimum(((oy + reach - gy0) / gs).to(tl.int32), gny - 1)
    k0 = tl.maximum(((oz - reach - gz0) / gs).to(tl.int32), 0)
    k1 = tl.minimum(((oz + reach - gz0) / gs).to(tl.int32), gnz - 1)
    rf = tl.arange(0, BLOCK_F)
    for k in range(k0, k1 + 1):
        for j in range(j0, j1 + 1):
            for i in range(i0, i1 + 1):
                cell = (k * gny + j) * gnx + i
                lo_x = gx0 + i * gs - ox
                lo_y = gy0 + j * gs - oy
                lo_z = gz0 + k * gs - oz
                ex = tl.maximum(tl.maximum(lo_x, -(lo_x + gs)), 0.0)
                ey = tl.maximum(tl.maximum(lo_y, -(lo_y + gs)), 0.0)
                ez = tl.maximum(tl.maximum(lo_z, -(lo_z + gs)), 0.0)
                box = tl.sqrt(ex * ex + ey * ey + ez * ez)
                f0 = tl.load(cell_ptr_ptr + cell)
                f1 = tl.load(cell_ptr_ptr + cell + 1)
                if (f1 > f0) & (box <= cap):
                    for fs in range(f0, f1, BLOCK_F):
                        fi = fs + rf
                        fm = fi < f1
                        base = tri_ptr + 9 * fi
                        ax = (tl.load(base, mask=fm, other=0.0) - ox)[None, :]
                        ay = (tl.load(base + 1, mask=fm, other=0.0) - oy)[None, :]
                        az = (tl.load(base + 2, mask=fm, other=0.0) - oz)[None, :]
                        bx = (tl.load(base + 3, mask=fm, other=0.0) - ox)[None, :]
                        by = (tl.load(base + 4, mask=fm, other=0.0) - oy)[None, :]
                        bz = (tl.load(base + 5, mask=fm, other=0.0) - oz)[None, :]
                        cx = (tl.load(base + 6, mask=fm, other=0.0) - ox)[None, :]
                        cy = (tl.load(base + 7, mask=fm, other=0.0) - oy)[None, :]
                        cz = (tl.load(base + 8, mask=fm, other=0.0) - oz)[None, :]
                        d2 = _closest_d2(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz)
                        d2 = tl.where(fm[None, :], d2, float("inf"))
                        m = tl.min(d2, axis=1)
                        am = tl.argmin(d2, axis=1)
                        upd = m < best
                        best = tl.where(upd, m, best)
                        best_f = tl.where(upd, tl.load(face_id_ptr + fs + am), best_f)
                    # |c - q| <= h + D_p for any point's nearest point q, and
                    # D_p is at most the brick's worst current best.
                    worst = tl.sqrt(tl.max(tl.where(vm, best, 0.0), axis=0))
                    cap = tl.minimum(cap, worst + half_diag)
    tl.store(out_d2_ptr + start + rv, best, mask=vm)
    tl.store(out_face_ptr + start + rv, best_f, mask=vm)


class BrickIndex:
    """A mesh's faces listed per cell for :func:`brick_nearest`, built once per mesh."""

    def __init__(
        self, vertices: np.ndarray, faces: np.ndarray, device: torch.device, cell_mm: float = 1.0
    ) -> None:
        v = np.asarray(vertices, np.float64)
        f = np.asarray(faces, np.int64)
        tri = v[f]  # (F, 3, 3)
        lo = tri.reshape(-1, 3).min(0) - cell_mm
        dims = np.ceil((tri.reshape(-1, 3).max(0) + cell_mm - lo) / cell_mm).astype(np.int64) + 1
        c0 = np.floor((tri.min(1) - lo) / cell_mm).astype(np.int64)
        c1 = np.floor((tri.max(1) - lo) / cell_mm).astype(np.int64)
        span = c1 - c0 + 1
        n = span.prod(1)
        face = np.repeat(np.arange(len(f)), n)
        local = np.arange(face.size) - np.repeat(np.cumsum(n) - n, n)
        sx, sy = span[face, 0], span[face, 1]
        ix = c0[face, 0] + local % sx
        iy = c0[face, 1] + (local // sx) % sy
        iz = c0[face, 2] + local // (sx * sy)
        flat = (iz * dims[1] + iy) * dims[0] + ix
        order = np.argsort(flat, kind="stable")
        n_cells = int(np.prod(dims))
        ptr = np.zeros(n_cells + 1, np.int64)
        np.cumsum(np.bincount(flat, minlength=n_cells), out=ptr[1:])
        face = face[order]
        self.tri = torch.as_tensor(tri[face].reshape(-1, 9), dtype=torch.float32, device=device)
        self.face_id = torch.as_tensor(face, dtype=torch.int32, device=device)
        self.cell_ptr = torch.as_tensor(ptr, dtype=torch.int32, device=device)
        self.origin = lo
        self.cell_mm = float(cell_mm)
        self.dims = dims


def brick_nearest(
    index: BrickIndex,
    points: torch.Tensor,
    counts: torch.Tensor,
    centres: torch.Tensor,
    centre_bound: torch.Tensor,
    brick_mm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact nearest face per point, on the device: ``(distance, face)``.

    ``points`` ``(N, 3)`` float32 grouped by brick, ``counts[b]`` points in
    brick ``b``, whose centre is ``centres[b]`` and ``centre_bound[b]`` an upper
    bound on that centre's distance to the mesh.
    """
    device = points.device
    n = points.shape[0]
    half = brick_mm * float(np.sqrt(3.0)) / 2.0
    starts = torch.cumsum(counts, 0) - counts
    # A brick with more points than a program holds is split; the pieces
    # share its centre and bound.
    pieces = (counts + BLOCK_V - 1) // BLOCK_V
    total = int(pieces.sum())
    seg_brick = torch.repeat_interleave(torch.arange(counts.numel(), device=device), pieces)
    within = torch.arange(total, device=device) - torch.repeat_interleave(
        torch.cumsum(pieces, 0) - pieces, pieces
    )
    seg_start = starts[seg_brick] + within * BLOCK_V
    seg_count = torch.clamp(starts[seg_brick] + counts[seg_brick] - seg_start, max=BLOCK_V)
    out_d2 = torch.empty(n, dtype=torch.float32, device=device)
    out_f = torch.empty(n, dtype=torch.int32, device=device)
    _brick_nearest_kernel[(total,)](
        points.contiguous(),
        seg_start.to(torch.int32),
        seg_count.to(torch.int32),
        seg_brick.to(torch.int32),
        centres.to(torch.float32).contiguous(),
        (centre_bound + 2 * half).to(torch.float32),
        index.tri,
        index.face_id,
        index.cell_ptr,
        float(index.origin[0]),
        float(index.origin[1]),
        float(index.origin[2]),
        index.cell_mm,
        int(index.dims[0]),
        int(index.dims[1]),
        int(index.dims[2]),
        float(half),
        out_d2,
        out_f,
        BLOCK_V=BLOCK_V,
        BLOCK_F=BLOCK_F,
    )
    return out_d2.sqrt(), out_f.long()


__all__ = ["BrickIndex", "brick_nearest"]
