"""
Fast cluster-extent null builder for [[ffs_perm]] — Numba-JIT'd
sorted-voxel union-find.

Idea
----
For one permutation, one NN connectivity, and one sidedness, naïvely we'd
threshold the t-map at every ``pthr`` and run a separate CCL pass per
threshold (the cc3d path).  Instead:

1. Apply sidedness (raw t for 1sided; ``|t|`` for 2sided).
2. Keep only voxels above the lowest pthr's tcrit.
3. Sort those voxels by stat **descending**.
4. Walk them: each voxel joins a disjoint-set forest, unioning with any
   already-active neighbours under the chosen NN connectivity.
5. Whenever we're about to add a voxel whose stat falls below the next
   tcrit, snapshot the current ``max(cluster size)`` for that pthr.

That processes all 9 default pthrs in **one pass per (NN, sided)**.  Mass
is intentionally skipped in v1 but the kernel layout (parent / size
arrays per root) makes adding a ``mass[root]`` accumulator a small
follow-up.

Performance budget (≈381 k mask voxels):
* sort: ~5 ms
* DSU walk with neighbour lookup: ~10–15 ms in Numba
* per (NN, sided) call: ~20 ms
* per perm × 6 combos: ~120 ms
* 1000 perms / 8 workers: ≈ 15 s
* 10 000 perms / 8 workers: ≈ 150 s
"""
from __future__ import annotations

import numpy as np
from numba import njit, types
from numba.typed import List as TypedList  # noqa: F401  (kept for future mass extension)

# ---------------------------------------------------------------------------
# Neighbour offset tables
# ---------------------------------------------------------------------------

def _offsets_for_nn(nn: int) -> np.ndarray:
    """``[K, 3]`` int8 array of (dx, dy, dz) offsets, K = 6/18/26."""
    if nn == 1:
        # 6 face neighbours
        return np.array(
            [(-1, 0, 0), (1, 0, 0),
             (0, -1, 0), (0, 1, 0),
             (0, 0, -1), (0, 0, 1)],
            dtype=np.int8,
        )
    if nn == 2:
        # 18 = faces + edges
        ofs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    if abs(dx) + abs(dy) + abs(dz) <= 2:
                        ofs.append((dx, dy, dz))
        return np.asarray(ofs, dtype=np.int8)
    if nn == 3:
        # 26 = full Moore neighbourhood
        ofs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    ofs.append((dx, dy, dz))
        return np.asarray(ofs, dtype=np.int8)
    raise ValueError(f"NN must be 1, 2, or 3; got {nn}")


# ---------------------------------------------------------------------------
# Numba kernel
# ---------------------------------------------------------------------------

@njit(cache=True, boundscheck=False, fastmath=True)
def _walk_dsu_extent(
    sorted_voxel_idx: np.ndarray,   # int64[n], flat indices, stat-descending
    sorted_voxel_stat: np.ndarray,  # float32[n], effective stat (signed/abs), descending
    parent: np.ndarray,             # int64[V_total], scratch, init = -1
    size: np.ndarray,               # int64[V_total], scratch
    offsets: np.ndarray,            # int8[K, 3] neighbour offsets
    nx: int, ny: int, nz: int,
    tcrits_desc: np.ndarray,        # float64[npthr], sorted *descending*
) -> np.ndarray:
    """Single-pass DSU; returns ``max_extent[npthr]`` int64."""
    n = sorted_voxel_idx.shape[0]
    npthr = tcrits_desc.shape[0]
    n_off = offsets.shape[0]
    # numpy C-order: for array of shape (nx, ny, nz), flat index
    # v = i*(ny*nz) + j*nz + k, with k fastest.
    yz = ny * nz

    max_extent = np.zeros(npthr, dtype=np.int64)
    cur_max = np.int64(0)
    ip = 0  # next pthr index to snapshot

    for kk in range(n):
        s = sorted_voxel_stat[kk]
        # Before adding this voxel, snapshot for every pthr whose tcrit
        # is now above s (i.e. this voxel is below their threshold).
        while ip < npthr and s <= tcrits_desc[ip]:
            max_extent[ip] = cur_max
            ip += 1
        if ip >= npthr:
            break  # all snapshots collected; remaining voxels irrelevant

        v = sorted_voxel_idx[kk]
        # Decompose v → (i, j, k) in C order; k fastest.
        i = v // yz
        rem = v - i * yz
        j = rem // nz
        k = rem - j * nz

        parent[v] = v
        size[v] = 1
        if cur_max < 1:
            cur_max = 1

        # Iterate neighbours
        for o in range(n_off):
            di = offsets[o, 0]
            dj = offsets[o, 1]
            dk = offsets[o, 2]
            i2 = i + di
            j2 = j + dj
            k2 = k + dk
            if i2 < 0 or i2 >= nx or j2 < 0 or j2 >= ny or k2 < 0 or k2 >= nz:
                continue
            nbr = i2 * yz + j2 * nz + k2
            if parent[nbr] < 0:
                continue
            # find(v) — v is its own root right now, but it may have been
            # unioned by an earlier neighbour in this same loop iteration.
            ra = v
            while parent[ra] != ra:
                parent[ra] = parent[parent[ra]]
                ra = parent[ra]
            # find(nbr)
            rb = nbr
            while parent[rb] != rb:
                parent[rb] = parent[parent[rb]]
                rb = parent[rb]
            if ra == rb:
                continue
            # union by size
            if size[ra] < size[rb]:
                parent[ra] = rb
                size[rb] += size[ra]
                if size[rb] > cur_max:
                    cur_max = size[rb]
            else:
                parent[rb] = ra
                size[ra] += size[rb]
                if size[ra] > cur_max:
                    cur_max = size[ra]

    # Tail: any remaining pthrs adopt the final cur_max
    while ip < npthr:
        max_extent[ip] = cur_max
        ip += 1
    return max_extent


# ---------------------------------------------------------------------------
# Top-level helper: one perm, all (NN, sided)
# ---------------------------------------------------------------------------

def cluster_extent_one_perm(
    stat3d: np.ndarray,
    mask_flat_idx: np.ndarray,
    shape_xyz: tuple[int, int, int],
    nns: tuple[int, ...],
    sideds: tuple[str, ...],
    tcrits_by_sided: dict[str, np.ndarray],
    offsets_by_nn: dict[int, np.ndarray],
    parent_scratch: np.ndarray,
    size_scratch: np.ndarray,
) -> dict[tuple[str, int], np.ndarray]:
    """One perm; returns ``{(sided, nn): max_extent_per_pthr[npthr]}``.

    Parameters
    ----------
    stat3d : float32 3-D array (already with mask scattered in).
    mask_flat_idx : int64[V_mask] — flat indices of in-mask voxels.
    tcrits_by_sided : dict mapping ``"1-sided"``/``"2-sided"`` to
        ``tcrits_desc`` (descending).
    offsets_by_nn : dict mapping NN → offset table.
    parent_scratch, size_scratch : preallocated int64[V_total] buffers,
        reset to -1 / 0 inside this function.
    """
    nx, ny, nz = shape_xyz
    stat_flat = stat3d.reshape(-1)
    stat_at_mask = stat_flat[mask_flat_idx]
    out: dict[tuple[str, int], np.ndarray] = {}

    for sided in sideds:
        if sided == "1-sided":
            eff_passes = [stat_at_mask]
        elif sided == "2-sided":
            eff_passes = [np.abs(stat_at_mask)]
        elif sided == "bi-sided":
            # Two independent passes: positive tail and negative tail.
            # Tail = max over the two; mass would be the sum within each
            # tail's connected components.
            eff_passes = [stat_at_mask, -stat_at_mask]
        else:
            raise ValueError(f"unknown sidedness: {sided}")

        tcrits_desc = tcrits_by_sided[sided]
        lowest = tcrits_desc[-1]

        for nn in nns:
            per_pass_extents = []
            for eff in eff_passes:
                above = eff > lowest
                idx_above = mask_flat_idx[above]
                stat_above = eff[above]
                order = np.argsort(-stat_above, kind="stable")
                idx_sorted = np.ascontiguousarray(idx_above[order])
                stat_sorted = np.ascontiguousarray(stat_above[order].astype(np.float32))
                parent_scratch.fill(-1)
                per_pass_extents.append(_walk_dsu_extent(
                    idx_sorted, stat_sorted,
                    parent_scratch, size_scratch,
                    offsets_by_nn[nn],
                    nx, ny, nz,
                    tcrits_desc,
                ))
            # Bi-sided: elementwise max over the +/- passes.  1-/2-sided
            # have a single pass so this is a no-op.
            stacked = np.stack(per_pass_extents, axis=0)
            out[(sided, nn)] = stacked.max(axis=0)

    return out


# ---------------------------------------------------------------------------
# Numba warm-up (compile cache hit on import)
# ---------------------------------------------------------------------------

def precompile() -> None:
    """Trigger Numba JIT once so the first real call isn't slow."""
    parent = np.full(8, -1, dtype=np.int64)
    size = np.zeros(8, dtype=np.int64)
    offsets = _offsets_for_nn(1)
    sorted_idx = np.array([0], dtype=np.int64)
    sorted_stat = np.array([5.0], dtype=np.float32)
    tcrits = np.array([4.0, 3.0], dtype=np.float64)
    _walk_dsu_extent(sorted_idx, sorted_stat, parent, size, offsets, 2, 2, 2, tcrits)


# Avoid `types` lint warning by referencing once (Numba may use it later).
_ = types
