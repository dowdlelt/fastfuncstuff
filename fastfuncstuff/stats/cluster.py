"""
3D connected-components clustering and 3dClustSim-style null tables.

AFNI nearest-neighbour conventions:

* ``NN1`` — face-touching only, 6-connectivity
* ``NN2`` — face + edge-touching, 18-connectivity
* ``NN3`` — face + edge + corner-touching, 26-connectivity

For every permutation we threshold the 3D t-map at each ``pthr`` for each
sidedness (1-sided positive tail, 2-sided ``|t|``) and each ``NN``, then
extract the maximum cluster size (voxel count) and maximum cluster mass
(sum of stat values above threshold).  Per-permutation maxima are the
null distribution; the cluster-size threshold at false-positive rate
``alpha`` is the ``(1 - alpha)`` quantile.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import cc3d
import numpy as np
import torch
from scipy.stats import t as student_t
from tqdm.auto import tqdm

# AFNI NN → cc3d connectivity
NN_TO_CONN = {1: 6, 2: 18, 3: 26}

# 3dClustSim defaults: 29 pthr × 10 athr × {1-sided, 2-sided, bi-sided} × NN1/2/3.
DEFAULT_PTHR = (
    0.10,
    0.09,
    0.08,
    0.07,
    0.06,
    0.05,
    0.04,
    0.03,
    0.02,
    0.015,
    0.01,
    0.007,
    0.005,
    0.003,
    0.002,
    0.0015,
    0.001,
    0.0007,
    0.0005,
    0.0003,
    0.0002,
    0.00015,
    0.0001,
    7e-5,
    5e-5,
    3e-5,
    2e-5,
    1.5e-5,
    1e-5,
)
DEFAULT_ATHR = (0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01)
DEFAULT_SIDED = ("1-sided", "2-sided", "bi-sided")
DEFAULT_NN = (1, 2, 3)


def _t_critical(pthr: float, dof: int, sidedness: str) -> float:
    """Critical t-value at uncorrected ``pthr`` for the given sidedness.

    AFNI convention (matched here):

    * ``1-sided`` — positive tail only.  ``tcrit = t.isf(pthr, dof)``.
    * ``2-sided`` — symmetric, ``P(|T|>tcrit)=pthr``.  ``tcrit = t.isf(pthr/2, dof)``.
    * ``bi-sided`` — two-tailed but clusters tails *separately*.  Threshold
      magnitude is the 2-sided ``tcrit = t.isf(pthr/2, dof)``; CCL is run
      independently on the positive- and negative-tail masks and the
      worst max-cluster is reported.

    For tcrit-lookup purposes ``2-sided`` and ``bi-sided`` use the same
    critical value.
    """
    if sidedness == "1-sided":
        return float(student_t.isf(pthr, dof))
    if sidedness in ("2-sided", "bi-sided"):
        return float(student_t.isf(pthr / 2.0, dof))
    raise ValueError(f"unknown sidedness: {sidedness}")


def _binarise(stat3d: np.ndarray, tcrit: float, sidedness: str) -> np.ndarray:
    if sidedness == "1-sided":
        return (stat3d > tcrit).astype(np.uint8, copy=False)
    # 2-sided and bi-sided both use |t|>tcrit for the binary mask; bi-sided
    # additionally splits + and - tails into independent CCL passes (handled
    # in _cluster_extent_mass_one).
    return (np.abs(stat3d) > tcrit).astype(np.uint8, copy=False)


def _cluster_extent_mass_one(
    stat3d: np.ndarray,
    tcrit: float,
    sidedness: str,
    nn: int,
) -> tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run CCL once and return (max_extent, max_mass, labels, sizes, masses).

    For ``bi-sided`` the positive and negative tails are clustered
    independently; ``labels`` is the union of the two label maps (with
    negative-tail labels offset to remain distinct), and the reported
    max is the max over both.
    """
    if sidedness == "bi-sided":
        # Positive tail
        binv_p = (stat3d > tcrit).astype(np.uint8, copy=False)
        # Negative tail
        binv_n = (stat3d < -tcrit).astype(np.uint8, copy=False)
        sub_results = []
        if binv_p.any():
            sub_results.append((binv_p, stat3d))
        if binv_n.any():
            sub_results.append((binv_n, -stat3d))
        if not sub_results:
            empty = np.zeros(0, dtype=np.int64)
            return 0, 0.0, np.zeros(stat3d.shape, dtype=np.int32), empty, empty.astype(np.float64)
        all_sizes, all_masses = [], []
        combined_labels = np.zeros(stat3d.shape, dtype=np.int32)
        offset = 0
        for binv, stat_for_mass in sub_results:
            lab = cc3d.connected_components(binv, connectivity=NN_TO_CONN[nn])
            n_lab = int(lab.max())
            if n_lab == 0:
                continue
            sizes = np.bincount(lab.ravel(), minlength=n_lab + 1)[1:]
            masses = np.bincount(
                lab.ravel(),
                weights=stat_for_mass.ravel(),
                minlength=n_lab + 1,
            )[1:]
            all_sizes.append(sizes)
            all_masses.append(masses)
            # Splice into combined label map with offset
            mask_nonzero = lab > 0
            combined_labels[mask_nonzero] = lab[mask_nonzero] + offset
            offset += n_lab
        if not all_sizes:
            empty = np.zeros(0, dtype=np.int64)
            return 0, 0.0, combined_labels, empty, empty.astype(np.float64)
        sizes = np.concatenate(all_sizes)
        masses = np.concatenate(all_masses)
        return int(sizes.max()), float(masses.max()), combined_labels, sizes, masses

    binv = _binarise(stat3d, tcrit, sidedness)
    if not binv.any():
        empty = np.zeros(0, dtype=np.int64)
        return 0, 0.0, np.zeros_like(binv, dtype=np.int32), empty, empty.astype(np.float64)
    labels = cc3d.connected_components(binv, connectivity=NN_TO_CONN[nn])
    n_labels = int(labels.max())
    if n_labels == 0:
        empty = np.zeros(0, dtype=np.int64)
        return 0, 0.0, labels.astype(np.int32), empty, empty.astype(np.float64)

    sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)[1:]
    stat_for_mass = stat3d if sidedness == "1-sided" else np.abs(stat3d)
    masses = np.bincount(
        labels.ravel(),
        weights=stat_for_mass.ravel(),
        minlength=n_labels + 1,
    )[1:]
    return int(sizes.max()), float(masses.max()), labels.astype(np.int32), sizes, masses


# ---------------------------------------------------------------------------
# Null distribution accumulator across (NN, sided, pthr)
# ---------------------------------------------------------------------------


@dataclass
class ClusterNull:
    """Accumulates per-permutation max extent/mass for each (sided, NN, pthr).

    Use ``record(perm_idx, stat3d, dof)`` after computing each permutation's
    3-D stat map.  After all permutations are seen, :meth:`extent_table` and
    :meth:`mass_table` return ``[len(pthr), len(athr)]`` matrices ready for
    NIML output.

    Attributes
    ----------
    pthr, athr : tuple[float, ...]
        Uncorrected per-voxel p thresholds and target FWE alphas.
    nns : tuple[int, ...]
        NN connectivities to record (1/2/3).
    sideds : tuple[str, ...]
        Sidednesses to record.
    n_perms : int
    """

    pthr: tuple[float, ...] = DEFAULT_PTHR
    athr: tuple[float, ...] = DEFAULT_ATHR
    nns: tuple[int, ...] = DEFAULT_NN
    sideds: tuple[str, ...] = DEFAULT_SIDED
    n_perms: int = 0
    # max_extent[sided][nn] -> [n_perms, len(pthr)]
    max_extent: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    max_mass: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)

    def init_storage(self, n_perms: int) -> None:
        self.n_perms = n_perms
        for sided in self.sideds:
            for nn in self.nns:
                self.max_extent[(sided, nn)] = np.zeros((n_perms, len(self.pthr)), dtype=np.int64)
                self.max_mass[(sided, nn)] = np.zeros((n_perms, len(self.pthr)), dtype=np.float64)

    def record(self, perm_idx: int, stat3d: np.ndarray, dof: int) -> None:
        """Record max extent/mass at every (sided, NN, pthr) for one perm."""
        for sided in self.sideds:
            for ip, p in enumerate(self.pthr):
                tcrit = _t_critical(p, dof, sided)
                # Re-binarise per NN to be safe (cc3d ignores connectivity-
                # dependent splits otherwise) — same binary mask is fine, but
                # cc3d label assignment depends on connectivity.
                for nn in self.nns:
                    ext, mass, _, _, _ = _cluster_extent_mass_one(stat3d, tcrit, sided, nn)
                    self.max_extent[(sided, nn)][perm_idx, ip] = ext
                    self.max_mass[(sided, nn)][perm_idx, ip] = mass

    def extent_table(self, sided: str, nn: int) -> np.ndarray:
        """``[len(pthr), len(athr)]`` cluster-size thresholds.

        Cell ``[i, j]`` = smallest cluster size whose null exceedance rate
        is ≤ ``athr[j]`` at pthr ``pthr[i]`` (i.e. the ``(1 - athr[j])``
        quantile of the null max-extent distribution).
        """
        nulls = self.max_extent[(sided, nn)]  # [P, npthr]
        return _quantile_table(nulls, self.athr)

    def mass_table(self, sided: str, nn: int) -> np.ndarray:
        nulls = self.max_mass[(sided, nn)]
        return _quantile_table(nulls, self.athr)


def _quantile_table(nulls: np.ndarray, athr: tuple[float, ...]) -> np.ndarray:
    """Return ``[npthr, nathr]`` table of (1-alpha)-quantiles of null max.

    Uses linear interpolation between order statistics, matching AFNI's
    3dClustSim (which produces fractional cluster-size thresholds like
    ``10935.33``).
    """
    out = np.zeros((nulls.shape[1], len(athr)), dtype=np.float64)
    # Exclude row 0 (identity / observed) from the null — standard practice.
    null_only = nulls[1:] if nulls.shape[0] > 1 else nulls
    for j, a in enumerate(athr):
        out[:, j] = np.quantile(null_only, 1.0 - a, axis=0, method="linear")
    return out


# ---------------------------------------------------------------------------
# Per-permutation utility: max stat for voxelwise FWE
# ---------------------------------------------------------------------------


def max_abs_t_per_perm(t_pv: torch.Tensor, sidedness: str) -> np.ndarray:
    """Return ``[P]`` of per-permutation peak stat for voxelwise FWE.

    1sided uses ``max(t)``; 2sided uses ``max(|t|)``.
    """
    if sidedness == "1-sided":
        return t_pv.max(dim=1).values.cpu().numpy()
    return t_pv.abs().max(dim=1).values.cpu().numpy()


def voxelwise_fwe_p(t_obs: np.ndarray, null_max: np.ndarray, sidedness: str) -> np.ndarray:
    """FWE-corrected p per voxel from the max-stat null.

    ``p_fwe(v) = (1 + #{null_max_i >= obs_v}) / (P+1)`` — standard 1/(P+1)
    plus-one rule so the smallest possible p is ``1/(P+1)``.
    """
    obs = t_obs if sidedness == "1-sided" else np.abs(t_obs)
    null = np.sort(null_max[1:])  # drop identity row
    # For each obs, count nulls >= obs.
    # np.searchsorted(null, obs, side='left') gives index of first null >= obs.
    idx = np.searchsorted(null, obs, side="left")
    ge = null.size - idx
    return (1.0 + ge) / (null.size + 1.0)


def uncorrected_p_from_perms(t_obs_pv: np.ndarray, sidedness: str) -> np.ndarray:
    """Per-voxel uncorrected p from the full ``[P, V]`` permutation matrix.

    ``p_unc(v) = #{t_perm_v >= t_obs_v} / P`` over all rows (the identity
    row contributes one count by construction).  This is algebraically the
    same as the ``(1 + B)/(P+1)`` "plus-one" form and bounded below by
    ``1/P`` — never zero.
    """
    p = t_obs_pv.shape[0]
    if sidedness == "1-sided":
        obs = t_obs_pv[0]
        ge = (t_obs_pv >= obs[None, :]).sum(axis=0).astype(np.float64)
    else:
        absp = np.abs(t_obs_pv)
        obs = absp[0]
        ge = (absp >= obs[None, :]).sum(axis=0).astype(np.float64)
    return ge / float(p)


def empirical_tcrits(
    perm_pseudo_t: np.ndarray,
    pthr: tuple[float, ...],
    sidedness: str,
) -> np.ndarray:
    """Cluster-defining tcrits from a permutation null when no parametric
    distribution applies (i.e. pseudo-t under variance smoothing).

    Pools the per-perm permutation null over voxels and perms (under H0
    each voxel's null is identically distributed within an EB, so pooling
    is exchangeability-justified) and returns the ``(1 - pthr)`` quantile
    at each pthr.

    * ``1-sided``: ``tcrit_i = quantile(t_null, 1 - pthr[i])``
    * ``2-sided`` / ``bi-sided``: ``tcrit_i = quantile(|t_null|, 1 - pthr[i])``
    """
    if sidedness == "1-sided":
        vals = perm_pseudo_t[1:].ravel()  # drop identity
    else:
        vals = np.abs(perm_pseudo_t[1:].ravel())
    qs = np.array([1.0 - p for p in pthr], dtype=np.float64)
    return np.quantile(vals, qs, method="linear").astype(np.float64)


def p_to_t(p: np.ndarray, dof: int, sidedness: str) -> np.ndarray:
    """Convert a per-voxel p back to a t-value for viewer thresholding.

    The FWE sub-brick is encoded as a t-stat with the original DoF so
    AFNI's threshold slider behaves naturally: thresholding at the t
    corresponding to ``p=0.05`` reveals FWE-significant voxels.
    """
    p = np.clip(p, 1e-30, 1.0)
    if sidedness == "1-sided":
        return student_t.isf(p, dof).astype(np.float32)
    return student_t.isf(p / 2.0, dof).astype(np.float32)


# ---------------------------------------------------------------------------
# Parallel null accumulation
# ---------------------------------------------------------------------------
#
# The CCL inner loop is embarrassingly parallel across permutations.  A
# process pool is the right tool: ``cc3d`` is a C++ extension and the
# Python orchestration overhead per CCL is small enough that the GIL
# matters; processes also sidestep any cc3d threading surprises.  On
# Linux we use ``fork`` so the mask + grids are inherited for free.

_NULL_WORKER_STATE: dict = {}


def _null_worker_init(
    mask: np.ndarray,
    dof: int,
    pthr: tuple[float, ...],
    nns: tuple[int, ...],
    sideds: tuple[str, ...],
    fast: bool,
    tcrits_override: dict[str, np.ndarray] | None = None,
) -> None:
    _NULL_WORKER_STATE["mask"] = mask
    _NULL_WORKER_STATE["dof"] = dof
    _NULL_WORKER_STATE["pthr"] = pthr
    _NULL_WORKER_STATE["nns"] = nns
    _NULL_WORKER_STATE["sideds"] = sideds
    _NULL_WORKER_STATE["fast"] = fast
    if tcrits_override is not None:
        _NULL_WORKER_STATE["tcrits"] = {
            (s, ip): float(tcrits_override[s][ip]) for s in sideds for ip in range(len(pthr))
        }
    else:
        _NULL_WORKER_STATE["tcrits"] = {
            (s, ip): _t_critical(p, dof, s) for s in sideds for ip, p in enumerate(pthr)
        }

    if fast:
        # Pre-compute fast-path scratch buffers and helpers.
        from fastfuncstuff.stats.cluster_fast import _offsets_for_nn, precompile

        precompile()
        _NULL_WORKER_STATE["mask_flat_idx"] = np.flatnonzero(mask.ravel()).astype(np.int64)
        _NULL_WORKER_STATE["offsets_by_nn"] = {nn: _offsets_for_nn(nn) for nn in nns}
        v_total = int(np.prod(mask.shape))
        _NULL_WORKER_STATE["parent_scratch"] = np.full(v_total, -1, dtype=np.int64)
        _NULL_WORKER_STATE["size_scratch"] = np.zeros(v_total, dtype=np.int64)
        # tcrits per sided, descending (the Numba kernel expects sorted desc)
        tcrits_by_sided = {}
        perm_to_orig = {}
        for s in sideds:
            if tcrits_override is not None:
                arr = np.asarray(tcrits_override[s], dtype=np.float64)
            else:
                arr = np.array(
                    [_t_critical(p, dof, s) for p in pthr],
                    dtype=np.float64,
                )
            # Sort descending by tcrit (stricter pthr → larger tcrit).
            order = np.argsort(-arr)
            tcrits_by_sided[s] = arr[order].copy()
            perm_to_orig[s] = order.astype(np.int64)
        _NULL_WORKER_STATE["tcrits_by_sided"] = tcrits_by_sided
        _NULL_WORKER_STATE["perm_to_orig"] = perm_to_orig


def _null_worker_chunk(
    args: tuple[int, np.ndarray],
) -> tuple[int, dict, dict]:
    """Process one chunk of permutations; return per-chunk max extent / mass.

    Returns ``(perm_start, max_extent, max_mass)`` where the dicts are
    keyed by ``(sided, nn)`` and have shape ``[Pc, npthr]``.  In fast
    mode ``max_mass`` is all-zero (not computed).
    """
    perm_start, t_chunk = args
    state = _NULL_WORKER_STATE
    mask: np.ndarray = state["mask"]
    pthr: tuple = state["pthr"]
    nns: tuple = state["nns"]
    sideds: tuple = state["sideds"]
    fast: bool = state["fast"]

    pc = t_chunk.shape[0]
    npth = len(pthr)
    max_extent = {(s, nn): np.zeros((pc, npth), dtype=np.int64) for s in sideds for nn in nns}
    max_mass = {(s, nn): np.zeros((pc, npth), dtype=np.float64) for s in sideds for nn in nns}

    if fast:
        from fastfuncstuff.stats.cluster_fast import cluster_extent_one_perm

        mask_flat_idx = state["mask_flat_idx"]
        offsets_by_nn = state["offsets_by_nn"]
        parent = state["parent_scratch"]
        size = state["size_scratch"]
        tcrits_by_sided = state["tcrits_by_sided"]
        perm_to_orig = state["perm_to_orig"]
        stat3d = np.zeros(mask.shape, dtype=np.float32)
        for i in range(pc):
            stat3d[:] = 0.0
            stat3d[mask] = t_chunk[i]
            res = cluster_extent_one_perm(
                stat3d,
                mask_flat_idx,
                mask.shape,
                nns,
                sideds,
                tcrits_by_sided,
                offsets_by_nn,
                parent,
                size,
            )
            # Map descending-order results back into original pthr column order
            for s in sideds:
                p2o = perm_to_orig[s]
                for nn in nns:
                    vec_desc = res[(s, nn)]  # [npthr] in descending-tcrit order
                    max_extent[(s, nn)][i, p2o] = vec_desc
        return perm_start, max_extent, max_mass

    # Slow / mass-aware fallback (cc3d per threshold).
    tcrits: dict = state["tcrits"]
    stat3d = np.zeros(mask.shape, dtype=np.float32)
    for i in range(pc):
        stat3d[:] = 0.0
        stat3d[mask] = t_chunk[i]
        for s in sideds:
            for ip in range(npth):
                tcrit = tcrits[(s, ip)]
                for nn in nns:
                    ext, mass, _, _, _ = _cluster_extent_mass_one(stat3d, tcrit, s, nn)
                    max_extent[(s, nn)][i, ip] = ext
                    max_mass[(s, nn)][i, ip] = mass

    return perm_start, max_extent, max_mass


def accumulate_cluster_null(
    t: np.ndarray,
    mask: np.ndarray,
    dof: int,
    nns: tuple[int, ...] = DEFAULT_NN,
    sideds: tuple[str, ...] = DEFAULT_SIDED,
    pthr: tuple[float, ...] = DEFAULT_PTHR,
    athr: tuple[float, ...] = DEFAULT_ATHR,
    n_jobs: int | None = None,
    verbose: bool = True,
    fast: bool = True,
    tcrits_override: dict[str, np.ndarray] | None = None,
) -> ClusterNull:
    """Build the ``ClusterNull`` over all permutations, in parallel.

    Parameters
    ----------
    t : np.ndarray, shape (P, V_in_mask), float32
        Permutation stat matrix; row 0 is the observed map.
    mask : np.ndarray, bool, shape (X, Y, Z)
        In-mask voxels (must satisfy ``mask.sum() == V_in_mask``).
    dof : int
        Degrees of freedom of the underlying t distribution.
    n_jobs : int or None
        Worker count.  ``None`` → ``os.cpu_count() - 1`` (capped at P).
        ``1`` runs single-process for debuggability.
    """
    n_perms, _ = t.shape
    null = ClusterNull(pthr=pthr, athr=athr, nns=nns, sideds=sideds)
    null.init_storage(n_perms)

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    n_jobs = min(n_jobs, n_perms)

    if n_jobs <= 1:
        _null_worker_init(mask, dof, pthr, nns, sideds, fast, tcrits_override)
        bar = tqdm(total=n_perms, desc="cluster null", leave=True, disable=not verbose)
        for p in range(n_perms):
            _, me, mm = _null_worker_chunk((p, t[p : p + 1]))
            for k, v in me.items():
                null.max_extent[k][p : p + 1] = v
            for k, v in mm.items():
                null.max_mass[k][p : p + 1] = v
            bar.update(1)
        bar.close()
        return null

    # Submit many small chunks (~8 per worker) so the tqdm bar updates
    # smoothly via as_completed.  Pickling cost is negligible (a few KB per
    # perm) compared to the CCL work.
    chunk_size = max(1, n_perms // (n_jobs * 8))
    tasks: list[tuple[int, np.ndarray]] = []
    for s in range(0, n_perms, chunk_size):
        e = min(s + chunk_size, n_perms)
        tasks.append((s, np.ascontiguousarray(t[s:e])))

    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    pool = ProcessPoolExecutor(
        max_workers=n_jobs,
        mp_context=ctx,
        initializer=_null_worker_init,
        initargs=(mask, dof, pthr, nns, sideds, fast, tcrits_override),
    )
    bar = tqdm(total=n_perms, desc="cluster null", leave=True, disable=not verbose)
    try:
        futures = [pool.submit(_null_worker_chunk, task) for task in tasks]
        for fut in as_completed(futures):
            perm_start, me, mm = fut.result()
            pc = next(iter(me.values())).shape[0]
            for k, v in me.items():
                null.max_extent[k][perm_start : perm_start + pc] = v
            for k, v in mm.items():
                null.max_mass[k][perm_start : perm_start + pc] = v
            bar.update(pc)
    finally:
        bar.close()
        pool.shutdown(wait=True)

    return null


def compute_observed_cluster_masks(
    t_obs_in_mask: np.ndarray,
    mask: np.ndarray,
    dof: int,
    nns: tuple[int, ...] = DEFAULT_NN,
    sideds: tuple[str, ...] = DEFAULT_SIDED,
    pthr: tuple[float, ...] = DEFAULT_PTHR,
) -> dict:
    """Cluster label maps for the observed (row-0) permutation only.

    Returns ``{(nn, pthr, "extent", sided): labels_3d_int32}`` matching the
    keys used by the ``-save-clust-masks`` output.
    """
    out: dict = {}
    stat3d = np.zeros(mask.shape, dtype=np.float32)
    stat3d[mask] = t_obs_in_mask
    for s in sideds:
        for p in pthr:
            tcrit = _t_critical(p, dof, s)
            for nn in nns:
                _, _, labels, _, _ = _cluster_extent_mass_one(stat3d, tcrit, s, nn)
                out[(nn, p, "extent", s)] = labels.astype(np.int32)
    return out
