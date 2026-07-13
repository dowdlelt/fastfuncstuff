"""
Nonparametric permutation tests for [[ffs_perm]].

Two test families with batched GPU kernels:

* **One-sample sign-flip** — null is "trial values are symmetric around 0".
  A permutation is a random sign vector ``s ∈ {-1,+1}^N`` applied trial-wise.
  Per-permutation t-stat uses the identity ``sum((s·y)²) = sum(y²)`` (signs
  cancel in the sum-of-squares), so per voxel we only need ``sum(y²)`` once
  and a single batched matmul ``S @ Y`` gives every permutation's mean.

* **Two-sample label-swap** — null is "the two groups are exchangeable".
  A permutation reassigns the binary group label.  Per voxel we need
  ``G @ Y`` and ``G @ Y²`` per permutation; Bessel-corrected pooled variance
  closes the form.

For both, the un-permuted statistic is row 0 of the perm matrix (identity
flip / identity group), so the same kernel computes the observed and the
null with no special-casing.

Block-restricted permutations (within-run exchangeability) are generated
by :func:`generate_label_swaps` when ``blocks`` is provided.  Sign flips
on a single trial are unaffected by block structure (a per-trial flip
preserves any partition), so :func:`generate_sign_flips` ignores blocks
unless ``flip_whole_block=True`` (the paired-style case, punted to v2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fastfuncstuff.memory import get_available_memory, get_memory_config


def _voxel_chunk(n_voxels: int, bytes_per_vox: int, device: torch.device) -> int:
    """Conservative voxel chunk for the permutation matmul working set.

    Uses the same available-memory + safety-factor model as the rest of
    fastfuncstuff (so ``MemoryConfig`` overrides apply), and clamps to
    ``min_chunk_size`` / ``max_chunk_size_gpu|cpu`` from the global config.
    """
    cfg = get_memory_config()
    budget = get_available_memory(device)  # already includes safety factor
    cap = cfg.max_chunk_size_gpu if device.type == "cuda" else cfg.max_chunk_size_cpu
    chunk = max(cfg.min_chunk_size, budget // max(bytes_per_vox, 1))
    return int(min(n_voxels, cap, chunk))


# ---------------------------------------------------------------------------
# Permutation generators
# ---------------------------------------------------------------------------


def generate_sign_flips(
    n_trials: int,
    n_perms: int,
    rng: np.random.Generator,
    exhaustive_threshold: int = 14,
) -> np.ndarray:
    """
    Generate a ``[P, N]`` int8 matrix of ±1 sign flips.

    Row 0 is the identity (all +1).  Subsequent rows are independent random
    sign vectors with no duplicate of the identity.  When ``2**n_trials`` is
    small enough (≤ ``2**exhaustive_threshold``) and would not exceed
    ``n_perms``, the full enumeration is returned (deterministic, exhaustive
    test).

    Parameters
    ----------
    n_trials : int
        Number of trials (rows of the data matrix).
    n_perms : int
        Requested number of permutations *including* the identity row.
    rng : np.random.Generator
        Source of randomness for sampling.
    exhaustive_threshold : int
        If ``n_trials <= exhaustive_threshold`` and ``2**n_trials <= n_perms``,
        return the exhaustive enumeration of all sign vectors.
    """
    total = 1 << n_trials  # 2**n_trials
    if n_trials <= exhaustive_threshold and total <= n_perms:
        # Exhaustive: every binary pattern, mapped to ±1, identity first.
        idx = np.arange(total, dtype=np.int64)
        bits = ((idx[:, None] >> np.arange(n_trials)[None, :]) & 1).astype(np.int8)
        signs = (1 - 2 * bits).astype(np.int8)  # 0→+1, 1→-1
        # Identity (all +1) is row 0 by construction (idx=0).
        return signs

    out = np.empty((n_perms, n_trials), dtype=np.int8)
    out[0] = 1
    out[1:] = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n_perms - 1, n_trials))
    return out


def generate_label_swaps(
    group: np.ndarray,
    n_perms: int,
    rng: np.random.Generator,
    blocks: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate a ``[P, N]`` int8 group-indicator matrix.

    Row 0 is the original group assignment.  Subsequent rows shuffle the
    group labels.  When ``blocks`` is given, shuffling is restricted to
    *within* each block (Nichols & Holmes restricted exchangeability — the
    canonical default for single-trial fMRI where the run is the obvious
    block).

    Parameters
    ----------
    group : np.ndarray[int]
        Length-N binary group indicator (1 for group A, 0 for group B).
    n_perms : int
        Number of permutations including the identity row.
    rng : np.random.Generator
        Source of randomness.
    blocks : np.ndarray[int], optional
        Length-N block labels.  If provided, each permutation shuffles
        group labels independently within each block.
    """
    g = np.asarray(group, dtype=np.int8)
    n = g.shape[0]
    out = np.empty((n_perms, n), dtype=np.int8)
    out[0] = g

    if blocks is None:
        for p in range(1, n_perms):
            out[p] = rng.permutation(g)
        return out

    blocks = np.asarray(blocks)
    block_ids = np.unique(blocks)
    block_idx = [np.where(blocks == b)[0] for b in block_ids]
    for p in range(1, n_perms):
        row = g.copy()
        for idx in block_idx:
            row[idx] = rng.permutation(row[idx])
        out[p] = row
    return out


def count_unique_label_perms(group: np.ndarray, blocks: np.ndarray | None) -> int:
    """
    Return the number of distinct within-block label permutations
    available for a 2-sample design (for the ``-q`` / how-many-perms path).

    Uses the multinomial coefficient per block:
    ``prod_b (n_b! / (nA_b! * nB_b!))``.
    """
    from math import comb

    g = np.asarray(group)
    if blocks is None:
        return int(comb(g.shape[0], int(g.sum())))
    total = 1
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        total *= comb(idx.size, int(g[idx].sum()))
    return total


# ---------------------------------------------------------------------------
# Batched statistic kernels
# ---------------------------------------------------------------------------


@dataclass
class PermStats:
    """Result of a permutation pass.

    Attributes
    ----------
    t : torch.Tensor
        Shape ``[P, V]`` of t-statistics (row 0 is the observed t).
    mean : torch.Tensor
        Observed mean per voxel (1-sample) or ``meanA - meanB`` (2-sample);
        shape ``[V]``.  Stored on CPU, float32.
    extras : dict
        Test-specific extras: for 2-sample, ``meanA`` and ``meanB``.
    dof : int
        Degrees of freedom of the t distribution (N-1 for 1-sample,
        nA + nB - 2 for 2-sample).
    """

    t: torch.Tensor
    mean: torch.Tensor
    extras: dict
    dof: int


def _device(device_str: str) -> torch.device:
    return torch.device(device_str)


def one_sample_t_perm(
    y: np.ndarray,
    sign_flips: np.ndarray,
    device: str = "cuda",
    perm_chunk: int | None = None,
    voxel_chunk: int | None = None,
    show_progress: bool = True,
    keep_perm_data: bool = False,
) -> PermStats:
    """
    Batched 1-sample t-stat across permutations.

    Parameters
    ----------
    y : np.ndarray, shape (N, V)
        Trial-by-voxel data.  ``y[0]`` is trial 0.
    sign_flips : np.ndarray[int8], shape (P, N)
        Output of :func:`generate_sign_flips`.  Row 0 is identity.
    device : str
        Torch device string.
    perm_chunk, voxel_chunk : int, optional
        Override the auto-chosen chunk sizes (useful for tests).

    Returns
    -------
    PermStats
        ``t`` has shape ``(P, V)``, ``mean`` has shape ``(V,)``.
    """
    n, v = y.shape
    p = sign_flips.shape[0]
    if sign_flips.shape[1] != n:
        raise ValueError(f"sign_flips trial dim {sign_flips.shape[1]} != y trials {n}")

    dev = _device(device)
    y_t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    sum_y2 = (y_t * y_t).sum(dim=0)  # [V], constant under sign flip

    if voxel_chunk is None:
        # Working set per voxel chunk on-device: y_chunk (N·4) + means/var (2·P·4)
        # + t scratch (P·4) ≈ (N + 3·P) · 4 bytes per voxel.
        bytes_per_vox = (n + 3 * p) * 4
        voxel_chunk = _voxel_chunk(v, bytes_per_vox, dev)
    if perm_chunk is None:
        perm_chunk = p  # the matmul cost is linear; let it run unless you set it.

    t_out = torch.empty((p, v), dtype=torch.float32)
    mean_obs = torch.empty(v, dtype=torch.float32)
    # Per-perm mean and the constant ``sum_y2`` together let downstream
    # code recompute the per-perm variance (var = (sum_y2 - n·m²)/(n-1))
    # without re-running the matmul.  Cheap to store; only allocated when
    # asked because at P=10k, V=300k it's 12 GB.
    perm_means = torch.empty((p, v), dtype=torch.float32) if keep_perm_data else None

    sqrt_n = float(np.sqrt(n))
    dof = n - 1

    try:
        from tqdm.auto import tqdm

        bar = tqdm(
            total=(v + voxel_chunk - 1) // voxel_chunk,
            desc="1-samp perm",
            leave=True,
            disable=not show_progress,
        )
    except ImportError:
        bar = None

    for v0 in range(0, v, voxel_chunk):
        v1 = min(v0 + voxel_chunk, v)
        y_chunk = y_t[:, v0:v1].to(dev, non_blocking=True)  # [N, Vc]
        sum_y2_chunk = sum_y2[v0:v1].to(dev, non_blocking=True)  # [Vc]

        # Identity mean (no flip) for the observed mean output.
        mean_obs[v0:v1] = y_chunk.mean(dim=0).cpu()

        for p0 in range(0, p, perm_chunk):
            p1 = min(p0 + perm_chunk, p)
            s = torch.from_numpy(sign_flips[p0:p1]).to(dev, dtype=torch.float32)
            # m[p, vox] = mean(s_p * y_vox) = (S @ Y) / N
            m = (s @ y_chunk) / n  # [Pc, Vc]
            var = (sum_y2_chunk[None, :] - n * m * m) / (n - 1)
            var = var.clamp_min(1e-30)  # guard against constant voxels
            t = m * sqrt_n / torch.sqrt(var)
            t_out[p0:p1, v0:v1] = t.cpu()
            if perm_means is not None:
                perm_means[p0:p1, v0:v1] = m.cpu()

        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()

    extras: dict = {}
    if perm_means is not None:
        extras["perm_means"] = perm_means
        extras["sum_y2"] = sum_y2  # per-voxel constant, [V] CPU tensor
    return PermStats(t=t_out, mean=mean_obs, extras=extras, dof=dof)


def two_sample_t_perm(
    y: np.ndarray,
    group_swaps: np.ndarray,
    device: str = "cuda",
    perm_chunk: int | None = None,
    voxel_chunk: int | None = None,
    show_progress: bool = True,
    welch: bool = False,
    keep_perm_data: bool = False,
) -> PermStats:
    """
    Batched 2-sample t-stat across permutations.

    Parameters
    ----------
    y : np.ndarray, shape (N, V)
    group_swaps : np.ndarray[int8], shape (P, N)
        Output of :func:`generate_label_swaps`.  Row 0 is the observed
        group assignment.  Values are 1 (group A) and 0 (group B).
    welch : bool, default False
        If True, use Welch's unequal-variance t-statistic
        ``t = (mA − mB) / sqrt(var_A/n_A + var_B/n_B)`` per permutation
        (Bessel-corrected per-group variance recomputed each perm).
        DoF reported is the conservative ``nA + nB − 2`` (Welch's
        Satterthwaite DoF varies per voxel and per perm; we use the
        pooled DoF for cluster-table tcrit lookup, matching randomise).
        If False, uses the pooled-variance two-sample t.
    """
    n, v = y.shape
    p = group_swaps.shape[0]
    if group_swaps.shape[1] != n:
        raise ValueError("group_swaps trial dim mismatch")

    # Group sizes are invariant across all permutations (they're shuffles).
    nA_arr = group_swaps.sum(axis=1)
    if not np.all(nA_arr == nA_arr[0]):
        raise ValueError("group_swaps rows must preserve group A size (shuffles only)")
    nA = int(nA_arr[0])
    nB = n - nA
    if nA < 2 or nB < 2:
        raise ValueError(f"need ≥2 trials per group, got nA={nA}, nB={nB}")

    dev = _device(device)
    y_t = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    y2_t = y_t * y_t

    if voxel_chunk is None:
        # On-device: y_chunk, y2_chunk (2·N·4) + sumA/ssqA/mA/mB/t (≥5·P·4).
        bytes_per_vox = (2 * n + 6 * p) * 4
        voxel_chunk = _voxel_chunk(v, bytes_per_vox, dev)
    if perm_chunk is None:
        perm_chunk = p

    t_out = torch.empty((p, v), dtype=torch.float32)
    meanA = torch.empty(v, dtype=torch.float32)
    meanB = torch.empty(v, dtype=torch.float32)
    perm_mA = torch.empty((p, v), dtype=torch.float32) if keep_perm_data else None
    perm_mB = torch.empty((p, v), dtype=torch.float32) if keep_perm_data else None
    perm_var = torch.empty((p, v), dtype=torch.float32) if keep_perm_data else None
    perm_varA = torch.empty((p, v), dtype=torch.float32) if (keep_perm_data and welch) else None
    perm_varB = torch.empty((p, v), dtype=torch.float32) if (keep_perm_data and welch) else None

    pool_denom = float(nA + nB - 2)
    pool_factor = float(1.0 / nA + 1.0 / nB)
    dof = nA + nB - 2

    try:
        from tqdm.auto import tqdm

        bar = tqdm(
            total=(v + voxel_chunk - 1) // voxel_chunk,
            desc="2-samp perm",
            leave=True,
            disable=not show_progress,
        )
    except ImportError:
        bar = None

    g_obs_np = group_swaps[0].astype(np.float32)

    for v0 in range(0, v, voxel_chunk):
        v1 = min(v0 + voxel_chunk, v)
        y_chunk = y_t[:, v0:v1].to(dev, non_blocking=True)  # [N, Vc]
        y2_chunk = y2_t[:, v0:v1].to(dev, non_blocking=True)  # [N, Vc]
        sum_y_chunk = y_chunk.sum(dim=0)  # [Vc]
        sum_y2_chunk = y2_chunk.sum(dim=0)  # [Vc]

        # Observed group means (for the meanA/meanB output sub-bricks).
        g_obs = torch.from_numpy(g_obs_np).to(dev)
        sumA_obs = g_obs @ y_chunk
        meanA[v0:v1] = (sumA_obs / nA).cpu()
        meanB[v0:v1] = ((sum_y_chunk - sumA_obs) / nB).cpu()

        for p0 in range(0, p, perm_chunk):
            p1 = min(p0 + perm_chunk, p)
            g = torch.from_numpy(group_swaps[p0:p1]).to(dev, dtype=torch.float32)
            sumA = g @ y_chunk  # [Pc, Vc]
            ssqA = g @ y2_chunk
            sumB = sum_y_chunk[None, :] - sumA
            ssqB = sum_y2_chunk[None, :] - ssqA
            mA = sumA / nA
            mB = sumB / nB
            if welch:
                # Bessel-corrected per-group variance, per perm
                var_A = (ssqA - nA * mA * mA) / (nA - 1)
                var_B = (ssqB - nB * mB * mB) / (nB - 1)
                denom = torch.sqrt(
                    (var_A / nA + var_B / nB).clamp_min(1e-30),
                )
            else:
                # pooled SS: (SSA - nA·mA²) + (SSB - nB·mB²)
                pooled_ss = ssqA - nA * mA * mA + ssqB - nB * mB * mB
                var_pool = pooled_ss / pool_denom
                denom = torch.sqrt((var_pool * pool_factor).clamp_min(1e-30))
            t = (mA - mB) / denom
            t_out[p0:p1, v0:v1] = t.cpu()
            if perm_mA is not None and perm_mB is not None:
                perm_mA[p0:p1, v0:v1] = mA.cpu()
                perm_mB[p0:p1, v0:v1] = mB.cpu()
                if welch and perm_varA is not None and perm_varB is not None:
                    perm_varA[p0:p1, v0:v1] = var_A.cpu()
                    perm_varB[p0:p1, v0:v1] = var_B.cpu()
                elif perm_var is not None:
                    perm_var[p0:p1, v0:v1] = var_pool.cpu()

        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()

    diff = meanA - meanB
    extras: dict = {"meanA": meanA, "meanB": meanB}
    if keep_perm_data:
        extras["perm_mA"] = perm_mA
        extras["perm_mB"] = perm_mB
        extras["_nA"] = nA
        extras["_nB"] = nB
        if welch:
            extras["perm_varA"] = perm_varA
            extras["perm_varB"] = perm_varB
        else:
            extras["perm_var"] = perm_var
            extras["pool_factor"] = pool_factor
    return PermStats(t=t_out, mean=diff, extras=extras, dof=dof)
