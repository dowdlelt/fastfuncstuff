"""
Combinatorial PC denoising: exhaustive evaluation of all 2^k subsets of noise PCs.

Unlike the sequential approach in denoise.py which tests prefix subsets
{0}, {0,1}, {0,1,2}, ..., this module tests ALL 2^k subsets:
{}, {0}, {1}, {0,1}, {2}, {0,2}, {1,2}, ...

This allows discovery of non-contiguous optimal PC subsets, e.g., selecting
PCs 0, 3, and 5 while skipping PCs 1, 2, and 4.

Algorithm (per outer fold):
1. Hold out one run, fit OLS betas on remaining N-1 runs
2. Inner LORO CV on N-1 runs to select criteria voxels
3. Extract k PCs from the held-out run's noise pool
4. Evaluate all 2^k combinations on the held-out run
5. Select optimal combination (argmax median CoD across criteria voxels)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from itertools import combinations as itertools_combinations

import numpy as np
import torch
from tqdm import tqdm

from fastfuncstuff.decomposition.pca import PCA
from fastfuncstuff.denoise.sequential import _compute_local_run_starts
from fastfuncstuff.glm.xval import (
    _cod_ratio,
    compute_xval_r2,
    generate_cv_splits,
    project_out_nuisance_per_run,
    slice_by_runs,
)
from fastfuncstuff.memory import estimate_chunk_size, get_available_memory, get_memory_config


def _combination_work_chunks(
    n_combos: int,
    n_timepoints: int,
    n_voxels: int,
    device: torch.device,
) -> tuple[int, int]:
    """Jointly size combination and voxel batches from the shared budget."""
    available = get_available_memory(device)
    cfg = get_memory_config()
    # A batch owns B projection matrices plus projected data, prediction, and
    # reduction scratch proportional to B*T*V. Split the already safety-scaled
    # budget so neither dimension can consume it all.
    projection_bytes = max(n_timepoints * n_timepoints * 4, 1)
    combo_batch = max(1, min(n_combos, 512, int(available * 0.35) // projection_bytes))
    transient_bytes_per_voxel = max(combo_batch * (3 * n_timepoints + 4) * 4, 1)
    voxel_cap = cfg.max_chunk_size_gpu if device.type in {"cuda", "mps"} else cfg.max_chunk_size_cpu
    voxel_chunk = max(
        1, min(n_voxels, voxel_cap, int(available * 0.55) // transient_bytes_per_voxel)
    )
    return combo_batch, voxel_chunk


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class CombinatorialDenoiseRunResult:
    """Results from combinatorial denoising for a single held-out run."""

    run_idx: int
    optimal_combination: tuple[int, ...]  # Selected PC indices
    # Median CoD of the SELECTED set. In singleton mode that set is never one of
    # the scored candidates (only singletons and the baseline are), so it is
    # scored once more on its own rather than reported as some other combo's
    # value -- which is what the old `median_cod[best_idx]` did, with best_idx
    # pinned to 0, i.e. the empty set.
    optimal_cod: float
    baseline_cod: float  # Median CoD with no PCs removed; optimal_cod - this is the gain
    all_cod: np.ndarray  # (2^k,) CoD per combo
    all_var_explained: np.ndarray  # (2^k,) variance per combo
    all_combinations: list[tuple[int, ...]]  # All 2^k subsets
    explained_variance_ratios: np.ndarray  # (k,) per-PC variance ratios
    n_criteria_voxels: int  # Number of criteria voxels used
    # Singleton null calibration (-null_surrogates). None when it was not run.
    null_thresholds: np.ndarray | None = None  # (k,) delta a PC had to beat
    pc_status: tuple[str, ...] | None = None  # (k,) selected|rejected_null|not_selected


@dataclass
class CombinatorialDenoiseResults:
    """Full results from combinatorial denoising across all runs."""

    per_run_results: list[CombinatorialDenoiseRunResult]
    noise_pool_mask: torch.Tensor  # (n_voxels,) bool
    initial_r2: torch.Tensor  # (n_voxels,) initial xval R2
    noise_pcs_per_run: list[torch.Tensor]  # Per-run PCs (all k)
    metadata: dict = field(default_factory=dict)


# ============================================================================
# Combination generation
# ============================================================================


def generate_all_pc_combinations(k: int) -> list[tuple[int, ...]]:
    """
    Generate all 2^k subsets of PC indices {0, ..., k-1}.

    Uses itertools.combinations for each subset size, producing subsets
    in order of increasing size: (), (0,), (1,), ..., (0,1), (0,2), ...

    Parameters
    ----------
    k : int
        Number of PCs. Must be >= 0.

    Returns
    -------
    list of tuple[int, ...]
        All 2^k subsets, starting with the empty set.
    """
    combos: list[tuple[int, ...]] = [()]  # Empty set first
    for size in range(1, k + 1):
        for combo in itertools_combinations(range(k), size):
            combos.append(combo)
    return combos


# ============================================================================
# PCA extraction with variance ratios
# ============================================================================


def extract_pcs_single_run_with_variance(
    run_data: torch.Tensor,
    noise_pool_mask: torch.Tensor,
    nuisance: torch.Tensor,
    max_components: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Extract PCs from one run's noise pool, returning PCs and variance ratios.

    Steps:
    1. Extract noise pool voxels
    2. Project out nuisance (polys, motion) via QR
    3. Unit-length normalize each voxel timeseries
    4. PCA to get PC timecourses
    5. Normalize PCs to unit variance

    Parameters
    ----------
    run_data : torch.Tensor
        Data for one run, (n_voxels, run_length).
    noise_pool_mask : torch.Tensor
        Boolean mask, (n_voxels,).
    nuisance : torch.Tensor
        Nuisance regressors for this run, (run_length, n_nuisance_cols).
    max_components : int
        Number of PCs to extract.
    device : torch.device
        Compute device.

    Returns
    -------
    pcs : torch.Tensor
        PC timecourses, (run_length, max_components), unit variance.
    variance_ratios : np.ndarray
        Explained variance ratio for each PC, (max_components,).
    """
    run_length = run_data.shape[1]

    # Extract noise pool voxels
    noise_data = run_data[noise_pool_mask, :].to(device)  # (n_noise, run_length)

    # Project out nuisance via QR
    nuis = nuisance.to(device)
    col_norms = nuis.abs().sum(dim=0)
    nonzero_cols = col_norms > 1e-10
    if nonzero_cols.any():
        nuis_clean = nuis[:, nonzero_cols]
        Q, _ = torch.linalg.qr(nuis_clean)
        # Project: noise_data = noise_data - noise_data @ Q @ Q.T
        noise_data = noise_data - (noise_data @ Q) @ Q.T

    # Unit-length normalize each voxel timeseries (GLMdenoise pattern)
    norms = noise_data.norm(dim=1, keepdim=True).clamp(min=1e-10)
    noise_data = noise_data / norms

    # PCA: (n_timepoints, n_noise_voxels) convention
    n_noise = noise_data.shape[0]
    actual_max = min(max_components, run_length - 1, n_noise)

    pca = PCA(n_components=actual_max, device=device)
    # Transpose: PCA expects (n_samples=timepoints, n_features=voxels)
    scores = pca.fit_transform(noise_data.T)  # (run_length, actual_max)

    # Normalize PCs to unit variance
    pc_std = scores.std(dim=0, keepdim=True).clamp(min=1e-10)
    scores = scores / pc_std

    assert pca.explained_variance_ratio_ is not None
    variance_ratios = pca.explained_variance_ratio_.cpu().numpy()

    # Pad if we got fewer components than requested
    if actual_max < max_components:
        padding = torch.zeros(run_length, max_components - actual_max, device=device)
        scores = torch.cat([scores, padding], dim=1)
        variance_ratios = np.concatenate(
            [
                variance_ratios,
                np.zeros(max_components - actual_max),
            ]
        )

    return scores, variance_ratios


# ============================================================================
# Core batch evaluation
# ============================================================================


def evaluate_all_combinations_for_run(
    run_data_criteria: torch.Tensor,
    run_design: torch.Tensor,
    betas_criteria: torch.Tensor,
    poly_nuisance: torch.Tensor,
    noise_pcs: torch.Tensor,
    combinations: list[tuple[int, ...]],
    variance_ratios: np.ndarray,
    device: torch.device,
    criteria_chunk_size: int | None = None,  # Will be computed adaptively if None
    verbose: bool = False,
    return_raw_cod: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate all 2^k PC combinations for one held-out run.

    Batches all combinations efficiently by precomputing projection matrices.
    Processes voxels in chunks to control memory.

    Parameters
    ----------
    run_data_criteria : torch.Tensor
        Held-out run data for criteria voxels, (n_criteria, run_length).
    run_design : torch.Tensor
        Task design for the held-out run, (run_length, n_conditions).
    betas_criteria : torch.Tensor
        OLS betas from training runs for criteria voxels, (n_criteria, n_conditions).
    poly_nuisance : torch.Tensor
        Polynomial nuisance for this run, (run_length, n_poly_cols).
    noise_pcs : torch.Tensor
        All k PC timecourses for this run, (run_length, k).
    combinations : list of tuple[int, ...]
        All 2^k subsets to evaluate.
    variance_ratios : np.ndarray
        Per-PC explained variance ratios, (k,).
    device : torch.device
        Compute device.
    criteria_chunk_size : int, default=2000
        Max criteria voxels to process at once (to control GPU memory).
        Reduced from 5000 to avoid OOM with 128 combos.

    Returns
    -------
    median_cod : np.ndarray
        Median CoD across criteria voxels for each combination, (n_combos,).
    var_explained : np.ndarray
        Total variance explained for each combination, (n_combos,).
    """
    n_combos = len(combinations)
    T = run_data_criteria.shape[1]
    V_criteria = run_data_criteria.shape[0]

    combo_batch_size, planned_voxel_chunk = _combination_work_chunks(
        n_combos, T, V_criteria, device
    )
    if criteria_chunk_size is None:
        chunk_size = planned_voxel_chunk
    else:
        chunk_size = criteria_chunk_size

    # Move small matrices to device
    poly_nuis = poly_nuisance.to(device)
    pcs = noise_pcs.to(device)
    design = run_design.to(device)
    betas = betas_criteria.to(device)

    # Clean poly_nuisance: remove zero columns
    col_norms = poly_nuis.abs().sum(dim=0)
    nonzero_cols = col_norms > 1e-10
    if nonzero_cols.any():
        poly_nuis_clean = poly_nuis[:, nonzero_cols]
    else:
        poly_nuis_clean = torch.zeros(T, 0, device=device)

    # =========================================================================
    # Q-based projection (no explicit T x T projectors)
    # =========================================================================
    # The projector onto the complement of span(poly u PCs) factorises:
    #     P = (I - Q_S Q_S')(I - Q_poly Q_poly')
    # where Q_S is an orthonormal basis of the *poly-projected* PC subset. So
    # the polynomial part is removed once, up front, instead of being rebuilt
    # inside every one of the 2^k combination projectors -- and each combination
    # only ever needs its (T x |S|) basis, never a (T x T) matrix.
    #
    # Cost per combination drops from O(T^2 * V) to O(T * |S| * V), and the
    # batch's memory from B*T*T to B*T*k (a factor of ~T/k), which lets far more
    # combinations share a batch.
    #
    # FFS_COMBO_LEGACY=1 restores the explicit-projector implementation.
    if os.environ.get("FFS_COMBO_LEGACY", "") != "1":
        if poly_nuis_clean.shape[1] > 0:
            q_poly, _ = torch.linalg.qr(poly_nuis_clean)
        else:
            q_poly = None

        def _strip_poly(mat: torch.Tensor) -> torch.Tensor:
            """Remove the polynomial subspace from (T, m) columns."""
            if q_poly is None:
                return mat
            return mat - q_poly @ (q_poly.T @ mat)

        design_poly = _strip_poly(design)
        pcs_poly = _strip_poly(pcs)  # (T, k)
        k_pcs = pcs_poly.shape[1]

        # Orthonormal basis per combination, zero-padded to k columns so that
        # combinations of different sizes share one batched tensor (zero columns
        # contribute nothing to Q Q'). Grouping by subset size lets the QR itself
        # be batched: k+1 calls instead of 2^k.
        q_by_combo = torch.zeros(n_combos, T, max(k_pcs, 1), device=device)
        size_groups: dict[int, list[int]] = {}
        for ci, combo in enumerate(combinations):
            size_groups.setdefault(len(combo), []).append(ci)

        for size, combo_ids in size_groups.items():
            if size == 0:
                continue  # empty combination: poly-only, Q stays all-zero
            cols = torch.tensor(
                [combinations[ci] for ci in combo_ids], dtype=torch.long, device=device
            )  # (n_group, size)
            sub = pcs_poly[:, cols.reshape(-1)].T.reshape(len(combo_ids), size, T)
            sub = sub.transpose(1, 2)  # (n_group, T, size)
            q_group, _ = torch.linalg.qr(sub)
            idx = torch.tensor(combo_ids, dtype=torch.long, device=device)
            q_by_combo[idx, :, :size] = q_group

        all_cod = torch.zeros(n_combos, V_criteria, device="cpu")

        # The residual never depends on the combination. With
        #     e = dp - design_poly @ betas          (poly-projected residual)
        # the cleaned residual is just (I - Q Q') e, so
        #     SS_res = ||e||^2 - ||Q' e||^2
        #     SS_tot = ||dp||^2 - ||Q' dp||^2 - T * mean^2
        # Every combination-dependent term is therefore a (k x V) projection,
        # never a (T x V) cleaned timeseries. That removes the (B, T, V)
        # intermediates entirely -- which is where the time actually went, since
        # those reductions are bandwidth-bound, not flop-bound.
        #
        # Voxel chunk outer, combinations inner: the chunk is uploaded and its
        # poly projection and residual computed once, then shared by every
        # combination batch, instead of being redone for each of them.
        q_all_t = q_by_combo.transpose(1, 2)  # (n_combos, k, T)
        q_all_col_sums = q_by_combo.sum(dim=1)  # (n_combos, k) == Q' 1

        chunk_iter = range(0, V_criteria, chunk_size)
        for chunk_start in tqdm(
            list(chunk_iter), desc="  Evaluating combos", leave=True, disable=not verbose
        ):
            chunk_end = min(chunk_start + chunk_size, V_criteria)
            data_chunk = run_data_criteria[chunk_start:chunk_end, :].to(device)
            betas_chunk = betas[chunk_start:chunk_end, :]

            dp = _strip_poly(data_chunk.T)  # (T, chunk)
            resid_poly = dp - design_poly @ betas_chunk.T  # (T, chunk)

            ss_e = (resid_poly * resid_poly).sum(dim=0, dtype=torch.float64)
            ss_dp = (dp * dp).sum(dim=0, dtype=torch.float64)
            sum_dp = dp.sum(dim=0, dtype=torch.float64)

            for combo_batch_start in range(0, n_combos, combo_batch_size):
                combo_batch_end = min(combo_batch_start + combo_batch_size, n_combos)
                q_batch_t = q_all_t[combo_batch_start:combo_batch_end]  # (B, k, T)
                q_col_sums = q_all_col_sums[combo_batch_start:combo_batch_end]  # (B, k)

                a_proj = q_batch_t @ dp  # (B, k, chunk)
                e_proj = q_batch_t @ resid_poly  # (B, k, chunk)

                # Reduce over k (a dozen terms) in float32, then subtract in
                # float64: the cancellation risk is in the subtraction, not in
                # the short sum, and float64 reductions over the (B, k, V)
                # tensors run at a small fraction of float32 rate on consumer
                # cards -- which is where the remaining time was going.
                ss_res = ss_e - (e_proj * e_proj).sum(dim=1).double()
                ss_clean = ss_dp - (a_proj * a_proj).sum(dim=1).double()
                mean_clean = (sum_dp - (q_col_sums.unsqueeze(-1) * a_proj).sum(dim=1).double()) / T
                ss_tot = ss_clean - T * mean_clean * mean_clean
                cod_chunk = _cod_ratio(ss_res, ss_tot).float()

                all_cod[combo_batch_start:combo_batch_end, chunk_start:chunk_end] = cod_chunk.cpu()
                del a_proj, e_proj, cod_chunk

            del data_chunk, dp, resid_poly

        if device.type == "cuda":
            torch.cuda.empty_cache()

        var_explained = np.array(
            [sum(variance_ratios[pc] for pc in combo) for combo in combinations]
        )
        if return_raw_cod:
            return all_cod.numpy(), var_explained
        return all_cod.median(dim=1).values.numpy(), var_explained

    # CRITICAL: For 8192 combos, P_all would be 7.4 GB - must batch combos
    # Process combos in batches: build projections, evaluate all voxels, accumulate
    all_cod = torch.zeros(n_combos, V_criteria, device="cpu")  # CPU accumulation

    I_T = torch.eye(T, device=device)

    for combo_batch_start in tqdm(
        range(0, n_combos, combo_batch_size),
        desc="  Evaluating combos",
        leave=True,
        disable=not verbose,
    ):
        combo_batch_end = min(combo_batch_start + combo_batch_size, n_combos)
        batch_combos = combinations[combo_batch_start:combo_batch_end]
        n_batch = len(batch_combos)

        # Build projection matrices for this batch only
        P_batch = torch.zeros(n_batch, T, T, device=device)
        for bi, combo in enumerate(batch_combos):
            if len(combo) == 0:
                nuisance_c = poly_nuis_clean
            else:
                pc_cols = pcs[:, list(combo)]
                if poly_nuis_clean.shape[1] > 0:
                    nuisance_c = torch.cat([poly_nuis_clean, pc_cols], dim=1)
                else:
                    nuisance_c = pc_cols

            if nuisance_c.shape[1] == 0:
                P_batch[bi] = I_T
            else:
                Q, _ = torch.linalg.qr(nuisance_c)
                P_batch[bi] = I_T - Q @ Q.T

        # Project design for this batch
        design_clean_batch = P_batch @ design  # (n_batch, T, n_conds)

        # Process all voxel chunks for this combo batch
        for chunk_start in range(0, V_criteria, chunk_size):
            chunk_end = min(chunk_start + chunk_size, V_criteria)
            data_chunk = run_data_criteria[chunk_start:chunk_end, :].to(device)
            betas_chunk = betas[chunk_start:chunk_end, :]

            # Project data and compute CoD
            data_clean_chunk = P_batch @ data_chunk.T  # (n_batch, T, chunk)
            pred_chunk = torch.einsum("cti,xi->ctx", design_clean_batch, betas_chunk)

            ss_res = ((data_clean_chunk - pred_chunk) ** 2).sum(dim=1)
            data_mean = data_clean_chunk.mean(dim=1, keepdim=True)
            ss_tot = ((data_clean_chunk - data_mean) ** 2).sum(dim=1)
            cod_chunk = _cod_ratio(ss_res, ss_tot)

            # Store directly to CPU
            all_cod[combo_batch_start:combo_batch_end, chunk_start:chunk_end] = cod_chunk.cpu()

            del data_chunk, data_clean_chunk, pred_chunk, cod_chunk

        # Clean up batch tensors before next combo batch
        del P_batch, design_clean_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Variance explained per combo
    var_explained = np.array([sum(variance_ratios[pc] for pc in combo) for combo in combinations])

    if return_raw_cod:
        return all_cod.numpy(), var_explained

    # Median CoD across criteria voxels
    median_cod = all_cod.median(dim=1).values.numpy()  # (n_combos,)

    return median_cod, var_explained


# A candidate direction counts as new only if the nuisance projection leaves this
# fraction of it standing. Set for float32 data, whose roundoff after projecting
# out a column it already contains lands around 1e-6 relative.
_SPAN_TOL = 1e-4


def evaluate_combinations_cross_run(
    data_criteria: torch.Tensor,
    run_starts: list[int],
    n_timepoints: int,
    nuisance_per_run: list[torch.Tensor],
    target_run: int,
    pcs: torch.Tensor,
    combinations: list[tuple[int, ...]],
    design: torch.Tensor | None,
    designs_by_hrf: dict[int, torch.Tensor] | None,
    criteria_hrf_indices: torch.Tensor | None,
    device: torch.device,
    variance_ratios: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score PC removal by whether it improves prediction of the *other* runs.

    The within-run criterion asks: with the training betas held fixed, does
    removing these PCs from run *r* improve how well they explain what is left
    of run *r*? That is not what the denoising is later used for — in the final
    fit the PCs come out of run *r* while run *r* is *contributing to the betas*.
    It also cannot be read cleanly, because it re-derives SS_tot from the
    cleaned data, so CoD rises whenever residual variance is removed.

    This criterion matches deployment instead. For each other run *h*:

    1. Fit betas on every run except *h*, with run *r*'s candidate PCs removed
       from run *r*'s contribution.
    2. Predict run *h*, which never has PCs removed — only its own polynomials.
    3. Accumulate SS_res and SS_tot across all *h*, then one CoD per voxel.

    Because the scored target never changes with the candidate, SS_tot is
    identical across candidates and the mechanical inflation disappears: a
    candidate can only win by producing better betas. Averaging over every
    *h* also puts N-1 runs of evidence behind each decision instead of one.

    Parameters
    ----------
    data_criteria : torch.Tensor
        ``(n_criteria, n_timepoints)`` — criteria voxels only, all runs.
    target_run : int
        The run whose PCs are being chosen.
    pcs : torch.Tensor
        ``(run_length, k)`` candidate PCs for ``target_run``.
    criteria_hrf_indices : torch.Tensor, optional
        Per-criteria-voxel HRF index, required with ``designs_by_hrf``.
    variance_ratios : np.ndarray, optional
        Per-PC variance, only used to fill the returned var-explained vector.

    Returns
    -------
    median_cod : np.ndarray
        ``(n_combos,)`` median across criteria voxels.
    var_explained : np.ndarray
        ``(n_combos,)`` summed variance ratios, or zeros when not supplied.
    """
    from fastfuncstuff.glm.moments import run_bounds
    from fastfuncstuff.glm.xval import compute_qr_projectors
    from fastfuncstuff.memory import estimate_chunk_size

    n_runs = len(run_starts)
    other_runs = [h for h in range(n_runs) if h != target_run]
    if not other_runs:
        raise ValueError("cross-run criterion needs at least 2 runs")

    n_criteria = data_criteria.shape[0]
    n_combos = len(combinations)

    if designs_by_hrf is not None:
        if criteria_hrf_indices is None:
            raise ValueError("designs_by_hrf requires criteria_hrf_indices")
        groups = [
            ((criteria_hrf_indices == h).cpu(), designs_by_hrf[h])
            for h in torch.unique(criteria_hrf_indices).tolist()
        ]
    else:
        assert design is not None
        groups = [(None, design)]

    ss_res = torch.zeros(n_combos, n_criteria, dtype=torch.float64)
    ss_tot = torch.zeros(n_criteria, dtype=torch.float64)

    # The nuisance does not vary with the HRF group or the candidate, so the QR
    # factors are built once for the whole call instead of once per candidate.
    q_factors = compute_qr_projectors(nuisance_per_run, run_starts, device=device)
    bounds = {r: run_bounds(run_starts, n_timepoints, r) for r in range(n_runs)}

    def _project(mat: torch.Tensor, run_idx: int) -> torch.Tensor:
        """Remove run ``run_idx``'s nuisance from a ``(run_length, m)`` matrix."""
        q = q_factors[run_idx]
        return mat if q is None else mat - q @ (q.T @ mat)

    run_len_target = bounds[target_run][1] - bounds[target_run][0]
    # Same column count for every group: the HRF library varies the response
    # shape, not the number of conditions.
    n_task_cols = groups[0][1].shape[1]

    for voxel_mask, group_design in groups:
        n_group = n_criteria if voxel_mask is None else int(voxel_mask.sum().item())
        if n_group == 0:
            continue
        group_idx = (
            torch.arange(n_criteria)
            if voxel_mask is None
            else torch.nonzero(voxel_mask, as_tuple=True)[0]
        )

        # ---- design side: no voxels involved, so once per group ----
        x_by_run = {
            r: _project(group_design[bounds[r][0] : bounds[r][1], :].to(device), r)
            for r in range(n_runs)
        }
        xtx_by_run = {r: x.T.double() @ x.double() for r, x in x_by_run.items()}
        sum_xtx_other = sum(xtx_by_run[r] for r in other_runs)

        # What each candidate adds BEYOND the base nuisance, as an orthonormal
        # basis. span(nuisance, pcs) == span(nuisance, q), so appending q
        # reproduces the joint projection exactly -- which is what turns the
        # per-candidate re-projection into a rank-m downdate of the base moments
        # and lets every candidate be built at once.
        pcs_proj = _project(pcs.to(device=device, dtype=x_by_run[target_run].dtype), target_run)
        x_target = x_by_run[target_run].double()

        # Singletons are the overwhelming case (-singleton_only, and the null pass
        # is k*N of them), and for a single column the orthonormal basis is just a
        # normalisation -- so they are built as one batch rather than one QR each.
        # A per-candidate QR loop here cost more than everything else combined once
        # the rest was batched.
        single_slot = [ci for ci, c in enumerate(combinations) if len(c) == 1]
        multi_slot = [ci for ci, c in enumerate(combinations) if len(c) > 1]

        q_cols = torch.zeros(
            pcs_proj.shape[0], len(combinations), device=device, dtype=torch.float64
        )
        a_rows = torch.zeros(len(combinations), n_task_cols, device=device, dtype=torch.float64)
        active = torch.zeros(len(combinations), dtype=torch.bool, device=device)

        if single_slot:
            picked = [combinations[ci][0] for ci in single_slot]
            cols = pcs_proj[:, picked].double()
            norms = cols.norm(dim=0)
            # "Did this column survive the projection?" is a question about the
            # column, so the tolerance is relative to its OWN pre-projection norm,
            # not to the matrix. A PC already inside the nuisance span projects to
            # float32 roundoff -- around 1e-6 of itself, which a matrix-relative
            # threshold happily accepts and then normalises into a unit vector of
            # pure noise, changing the answer by 1e-4.
            ok = norms > _SPAN_TOL * pcs[:, picked].to(cols).norm(dim=0).clamp(min=1e-30)
            cols = torch.where(ok.unsqueeze(0), cols / norms.clamp(min=1e-30), 0.0)
            slots = torch.tensor(single_slot, device=device)
            q_cols[:, slots] = cols
            a_rows[slots] = cols.T @ x_target
            active[slots] = ok

        # Multi-column candidates still need a real QR, but only the -singleton_only
        # off path builds any, and then only 2^k - k - 1 of them.
        multi_extra: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for ci in multi_slot:
            block = pcs_proj[:, list(combinations[ci])]
            q_block, r_block = torch.linalg.qr(block)
            # QR normalises every column it returns, including the ones it
            # manufactured for a rank-deficient block, so the Q norms carry no
            # information. The R diagonal is what says whether a column added
            # anything, measured against the source column as above.
            source = pcs[:, list(combinations[ci])].to(block).norm(dim=0).clamp(min=1e-30)
            keep = r_block.diagonal().abs() > _SPAN_TOL * source
            q_block = q_block[:, keep].double()
            if q_block.shape[1]:
                multi_extra[ci] = (q_block, q_block.T @ x_target)

        xtx_cand = xtx_by_run[target_run].unsqueeze(0).repeat(len(combinations), 1, 1)
        xtx_cand -= torch.where(
            active.view(-1, 1, 1), a_rows.unsqueeze(2) * a_rows.unsqueeze(1), 0.0
        )
        for ci, (_, a_blk) in multi_extra.items():
            xtx_cand[ci] = xtx_by_run[target_run] - a_blk.T @ a_blk

        n_cond = group_design.shape[1]
        # (n_combos, n_h, C, C) -- every system this group needs. Candidate n with
        # held-out h is fit on the other runs minus h, plus the target run
        # carrying candidate n, and all those terms are already in hand.
        a_stack = torch.stack(
            [sum_xtx_other - xtx_by_run[h] + xtx_cand for h in other_runs], dim=1
        ) + 1e-6 * torch.eye(n_cond, device=device, dtype=torch.float64)

        chunk = estimate_chunk_size(
            n_voxels=n_group,
            n_timepoints=run_len_target,
            n_regressors=n_cond,
            device=device,
            operation="cross_run_combos",
            n_trials=len(other_runs),
            n_designs=n_combos,
        )
        group_data = data_criteria if voxel_mask is None else data_criteria[voxel_mask, :]

        for lo in range(0, n_group, chunk):
            hi = min(lo + chunk, n_group)
            block = group_data[lo:hi, :].to(device)
            d_by_run = {
                r: _project(block[:, bounds[r][0] : bounds[r][1]].T, r) for r in range(n_runs)
            }
            xty_by_run = {r: x_by_run[r].T.double() @ d_by_run[r].double() for r in range(n_runs)}
            sum_xty_other = sum(xty_by_run[r] for r in other_runs)

            target_data = d_by_run[target_run].double()
            # (n_combos, C, chunk) in one shot: q'D for every candidate, then the
            # outer product with its a-row. No Python loop over candidates.
            qd = q_cols.T @ target_data
            xty_cand = xty_by_run[target_run].unsqueeze(0) - torch.einsum("nc,nv->ncv", a_rows, qd)
            for ci, (q_blk, a_blk) in multi_extra.items():
                xty_cand[ci] = xty_by_run[target_run] - a_blk.T @ (q_blk.T @ target_data)
            chunk_res = torch.zeros(n_combos, hi - lo, dtype=torch.float64, device=device)
            chunk_tot = torch.zeros(hi - lo, dtype=torch.float64, device=device)
            # Held-out runs are looped, candidates are batched. Batching h too
            # would hold (n_combos, n_h, C, chunk) twice -- right-hand sides and
            # betas -- for a handful of saved launches, and that product is what
            # sets the peak. Looping it divides the peak by n_h and costs nothing
            # measurable now that the candidate axis carries the batch.
            for slot, h in enumerate(other_runs):
                actual = d_by_run[h].double()
                beta = torch.linalg.solve(
                    a_stack[:, slot], sum_xty_other - xty_by_run[h] + xty_cand
                )
                # |y - Xb|^2 = |y|^2 - 2 b'X'y + b'X'Xb, entirely in condition
                # space: no (n_combos, run_length, chunk) prediction is ever built,
                # which is what keeps the batch affordable.
                chunk_res += (
                    (actual * actual).sum(dim=0).unsqueeze(0)
                    - 2.0 * (beta * xty_by_run[h].unsqueeze(0)).sum(dim=1)
                    + torch.einsum("ncv,cd,ndv->nv", beta, xtx_by_run[h], beta)
                )
                centred = actual - actual.mean(dim=0, keepdim=True)
                chunk_tot += (centred * centred).sum(dim=0)

            ss_res[:, group_idx[lo:hi]] = chunk_res.cpu()
            ss_tot[group_idx[lo:hi]] = chunk_tot.cpu()
            del block, d_by_run, xty_by_run, xty_cand, chunk_res, chunk_tot

        del x_by_run, xtx_by_run, xtx_cand, a_stack, q_cols, a_rows, multi_extra
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cod = _cod_ratio(ss_res, ss_tot.unsqueeze(0))
    median_cod = cod.median(dim=1).values.numpy().astype(np.float64)
    if variance_ratios is None:
        var_explained = np.zeros(len(combinations))
    else:
        var_explained = np.array(
            [sum(variance_ratios[pc] for pc in combo) for combo in combinations]
        )
    return median_cod, var_explained


def _evaluate_columns_for_run(
    columns: torch.Tensor,
    combos: list[tuple[int, ...]],
    variance_ratios: np.ndarray,
    held_data_criteria: torch.Tensor,
    betas_criteria: torch.Tensor,
    held_nuisance: torch.Tensor,
    held_start: int,
    held_end: int,
    design: torch.Tensor | None,
    designs_by_hrf: dict[int, torch.Tensor] | None,
    hrf_indices: torch.Tensor | None,
    criteria_mask: torch.Tensor,
    unique_hrf_indices: list[int] | None,
    device: torch.device,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a set of candidate nuisance columns against one held-out run.

    Split out so the real PCs and their null surrogates go through exactly the
    same evaluation — same criteria voxels, same training betas, same per-HRF
    aggregation. A null scored any other way would not be a null.
    """
    if unique_hrf_indices is not None:
        assert designs_by_hrf is not None and hrf_indices is not None
        all_group_cods = []
        criteria_hrf_indices = hrf_indices[criteria_mask]

        for hrf_idx in unique_hrf_indices:
            group_criteria = criteria_hrf_indices == hrf_idx
            if int(group_criteria.sum().item()) == 0:
                continue

            # Move mask to CPU for indexing CPU tensors
            group_criteria_cpu = group_criteria.cpu()
            raw_cod, _ = evaluate_all_combinations_for_run(
                run_data_criteria=held_data_criteria[group_criteria_cpu, :],
                run_design=designs_by_hrf[hrf_idx][held_start:held_end, :],
                betas_criteria=betas_criteria[group_criteria_cpu, :],
                poly_nuisance=held_nuisance,
                noise_pcs=columns,
                combinations=combos,
                variance_ratios=variance_ratios,
                device=device,
                return_raw_cod=True,
                verbose=verbose,
            )
            all_group_cods.append(raw_cod)  # (n_combos, V_group)

        # Aggregate across HRF groups
        all_cod_combined = np.concatenate(all_group_cods, axis=1)
        return (
            np.median(all_cod_combined, axis=1).astype(np.float64),
            np.array([sum(variance_ratios[pc] for pc in combo) for combo in combos]),
        )

    assert design is not None
    return evaluate_all_combinations_for_run(
        run_data_criteria=held_data_criteria,
        run_design=design[held_start:held_end, :],
        betas_criteria=betas_criteria,
        poly_nuisance=held_nuisance,
        noise_pcs=columns,
        combinations=combos,
        variance_ratios=variance_ratios,
        device=device,
        verbose=verbose,
    )


# ============================================================================
# Null calibration for singleton selection
# ============================================================================


def phase_randomize(
    timecourses: torch.Tensor,
    n_surrogates: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Phase-randomised surrogates: same spectrum and variance, no real structure.

    Randomising the Fourier phases while keeping the magnitudes preserves each
    timecourse's variance and autocorrelation exactly, so a surrogate costs the
    same degree of freedom and removes a comparable amount of residual variance
    as the PC it stands in for. It differs in one respect only: it carries no
    genuine relationship to the shared noise in the criteria voxels. That makes
    the distribution of surrogate deltas the right null for "did this PC help",
    absorbing both the CoD's mechanical response to variance removal and the
    single-run sampling noise without having to model either.

    Parameters
    ----------
    timecourses : torch.Tensor
        ``(T, k)`` real-valued columns to build surrogates for.
    n_surrogates : int
        Surrogates per column.
    generator : torch.Generator, optional
        Seeded RNG, so a rerun reproduces the same selection.

    Returns
    -------
    torch.Tensor
        ``(T, k * n_surrogates)``, column-major by source PC: columns
        ``i * n_surrogates ... (i+1) * n_surrogates - 1`` are surrogates of PC i.
    """
    n_time, k = timecourses.shape
    device = timecourses.device
    # Phase randomisation is not a numerically sensitive solve. Keeping it in
    # float32 avoids MPS's unsupported float64 FFT and consumer-CUDA slowdown.
    fft_input = timecourses.to(torch.float32)
    spectrum = torch.fft.rfft(fft_input, dim=0)  # (F, k)
    magnitude = spectrum.abs()
    n_freq = magnitude.shape[0]

    mag = magnitude.T.unsqueeze(1).expand(k, n_surrogates, n_freq)
    # Drawn on the generator's own device (CPU by default) and moved, so the
    # same seed gives the same surrogates whether the PCs live on GPU or CPU.
    gen_device = generator.device if generator is not None else torch.device("cpu")
    phases = (
        torch.rand(k, n_surrogates, n_freq, device=gen_device, generator=generator) * 2 * torch.pi
    ).to(device)
    # DC has no phase to randomise, and for even T neither does Nyquist —
    # rotating either would make the surrogate complex after the inverse FFT.
    phases[:, :, 0] = 0.0
    if n_time % 2 == 0:
        phases[:, :, -1] = 0.0

    surrogates = torch.fft.irfft(mag * torch.exp(1j * phases), n=n_time, dim=2)
    out = surrogates.permute(2, 0, 1).reshape(n_time, k * n_surrogates)
    return out.to(timecourses.dtype)


def select_singletons_against_null(
    median_cod: np.ndarray,
    null_cod: np.ndarray,
    k: int,
    n_surrogates: int,
    percentile: float = 95.0,
) -> tuple[tuple[int, ...], np.ndarray, tuple[str, ...]]:
    """Keep the PCs whose singleton delta beats their own surrogates' deltas.

    ``median_cod`` and ``null_cod`` share element 0 — the baseline (no columns
    removed) CoD — so the deltas are on the same footing.

    Returns the selection, the per-PC threshold, and a per-PC status of
    ``selected`` / ``rejected_null`` (delta was positive but inside the null) /
    ``not_selected`` (delta was not positive at all).
    """
    baseline = median_cod[0]
    deltas = median_cod[1 : k + 1] - baseline
    null_deltas = (null_cod[1:] - baseline).reshape(k, n_surrogates)
    thresholds = np.percentile(null_deltas, percentile, axis=1)

    selected: list[int] = []
    status: list[str] = []
    for i in range(k):
        if deltas[i] > thresholds[i]:
            selected.append(i)
            status.append("selected")
        elif deltas[i] > 0:
            status.append("rejected_null")
        else:
            status.append("not_selected")
    return tuple(selected), thresholds, tuple(status)


# ============================================================================
# Selection strategies
# ============================================================================


def select_optimal_combination(
    median_cod: np.ndarray,
    combinations: list[tuple[int, ...]],
    strategy: str = "argmax",
) -> tuple[int, tuple[int, ...]]:
    """
    Select the optimal PC combination from evaluation results.

    Parameters
    ----------
    median_cod : np.ndarray
        Median CoD for each combination, (n_combos,).
    combinations : list of tuple[int, ...]
        All evaluated subsets.
    strategy : str, default="argmax"
        Selection strategy:
        - "argmax": Pick combination with highest median CoD.
        - "parsimonious": Among combos within 1% of max CoD, pick fewest PCs.

    Returns
    -------
    best_idx : int
        Index into combinations/median_cod of the selected combo.
    best_combo : tuple[int, ...]
        The selected PC indices.
    """
    if strategy == "argmax":
        best_idx = int(np.argmax(median_cod))
    elif strategy == "parsimonious":
        max_cod = np.max(median_cod)
        threshold = max_cod - 0.01 * abs(max_cod) if max_cod > 0 else max_cod - 0.01
        # Among combos within threshold, pick fewest PCs (ties broken by highest CoD)
        candidates = np.where(median_cod >= threshold)[0]
        sizes = np.array([len(combinations[i]) for i in candidates])
        min_size = sizes.min()
        min_size_candidates = candidates[sizes == min_size]
        best_idx = int(min_size_candidates[np.argmax(median_cod[min_size_candidates])])
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")

    return best_idx, combinations[best_idx]


# ============================================================================
# Top-level pipeline
# ============================================================================


def parse_criteria_spec(spec: float | int | str) -> tuple[str, float]:
    """
    Parse a criteria-pool specification into ``(kind, value)``.

    Accepted forms::

        0.05      absolute inner-CV R² threshold  → ("abs", 0.05)
        "5%"      top 5% of voxels by inner R²    → ("pct", 5.0)
        "(1000)"  top 1000 voxels by inner R²     → ("topn", 1000.0)

    A bare number (float or numeric string) is always an absolute threshold;
    top-N must be parenthesised so it can never be confused with one.
    """
    if isinstance(spec, int | float):
        return ("abs", float(spec))

    s = str(spec).strip()
    if s.endswith("%"):
        pct = float(s[:-1])
        if not 0.0 < pct <= 100.0:
            raise ValueError(f"Criteria percentile must be in (0, 100], got {s}")
        return ("pct", pct)
    if s.startswith("(") and s.endswith(")"):
        n = float(s[1:-1])
        if n < 1:
            raise ValueError(f"Criteria top-N must be >= 1, got {s}")
        return ("topn", n)
    return ("abs", float(s))


def select_criteria_voxels(
    inner_r2: torch.Tensor,
    spec: float | int | str = 0.05,
    fallback_percentile: float = 5.0,
    min_criteria: int = 100,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Choose the criteria voxels — the responsive voxels the CoD is medianed over.

    An absolute R² threshold is the honest criterion but its yield swings with
    tSNR, so when it selects fewer than *min_criteria* voxels we fall back to the
    top *fallback_percentile* percent by inner-CV R². Percentile and top-N specs
    are self-sizing and never fall back.

    A too-permissive pool is the failure that matters: at threshold 0.0 the
    median lands on a voxel with no response, the singleton deltas collapse to
    ~1e-4, and PC selection becomes a coin flip.
    """
    kind, value = parse_criteria_spec(spec)
    n_voxels = inner_r2.numel()

    def _top_n(n: int) -> torch.Tensor:
        n = int(max(1, min(n, n_voxels)))
        mask = torch.zeros(n_voxels, dtype=torch.bool, device=inner_r2.device)
        mask[inner_r2.topk(n).indices] = True
        return mask

    if kind == "abs":
        mask = inner_r2 > value
        n_sel = int(mask.sum().item())
        if n_sel < min_criteria:
            n_fallback = max(min_criteria, int(round(n_voxels * fallback_percentile / 100.0)))
            if verbose:
                print(
                    f"  Criteria pool: R2 > {value:g} gives only {n_sel:,} voxels; "
                    f"falling back to top {fallback_percentile:g}% ({n_fallback:,} voxels)"
                )
            return _top_n(n_fallback)
        if verbose:
            print(f"  Criteria pool: R2 > {value:g}")
        return mask

    if kind == "pct":
        n_sel = max(1, int(round(n_voxels * value / 100.0)))
        if verbose:
            print(f"  Criteria pool: top {value:g}% by inner-CV R2 ({n_sel:,} voxels)")
        return _top_n(n_sel)

    if verbose:
        print(f"  Criteria pool: top {int(value):,} voxels by inner-CV R2")
    return _top_n(int(value))


def fit_combinatorial_denoising(
    data: torch.Tensor,
    design: torch.Tensor | None,
    run_starts: list[int],
    tr: float,
    nuisance_per_run: list[torch.Tensor],
    noise_pool_mask: torch.Tensor,
    initial_r2: torch.Tensor,
    max_pcs: int = 7,
    criteria_r2_threshold: float | int | str = 0.05,
    criteria_fallback_percentile: float = 5.0,
    selection_strategy: str = "argmax",
    singleton_only: bool = False,
    criterion: str = "cross_run",
    n_null_surrogates: int = 0,
    null_percentile: float = 95.0,
    null_seed: int = 0,
    designs_by_hrf: dict[int, torch.Tensor] | None = None,
    hrf_indices: torch.Tensor | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> CombinatorialDenoiseResults:
    """
    Run the full combinatorial PC denoising pipeline.

    For each held-out run:
    1. Fit OLS betas on N-1 training runs (with nuisance projected)
    2. Inner LORO CV on N-1 runs to determine criteria pool
    3. Extract k PCs from held-out run's noise pool
    4. Evaluate all 2^k combinations on held-out run
    5. Select optimal combination

    Parameters
    ----------
    data : torch.Tensor
        fMRI data, (n_voxels, n_timepoints).
    design : torch.Tensor
        Task design matrix, (n_timepoints, n_conditions).
    run_starts : list of int
        Starting timepoint for each run.
    tr : float
        Repetition time in seconds.
    nuisance_per_run : list of torch.Tensor
        Per-run nuisance regressors (polynomials + ortvec).
    noise_pool_mask : torch.Tensor
        Boolean mask for noise pool voxels, (n_voxels,).
    initial_r2 : torch.Tensor
        Initial cross-validated R2 per voxel, (n_voxels,).
    max_pcs : int, default=7
        Number of PCs to extract per run. 2^max_pcs combinations evaluated.
    criteria_r2_threshold : float or str, default=0.05
        Criteria-pool specification: a float (absolute inner-CV R2 threshold),
        ``"5%"`` (top 5% of voxels by inner-CV R2) or ``"(1000)"`` (top 1000
        voxels). See :func:`select_criteria_voxels`.
    criteria_fallback_percentile : float, default=5.0
        When an absolute threshold yields too few voxels, take this top
        percentile instead.
    selection_strategy : str, default="argmax"
        Strategy for selecting optimal combination (see select_optimal_combination).
    singleton_only : bool, default=False
        Evaluate each PC alone rather than all 2^k subsets.
    criterion : {"cross_run", "within_run"}, default="cross_run"
        How a candidate PC set is scored. ``cross_run`` matches deployment: the
        run's PCs are removed while it contributes to the betas, and the betas
        are scored on the other runs, which are never cleaned. The scored target
        is then identical across candidates, so SS_tot is fixed and only better
        betas can win. ``within_run`` is the historical rule — it holds the
        training betas fixed and asks how well they explain what is *left* of the
        held-out run after removal, which scores a removal on the scoring side
        while the denoising is later applied on the fitting side, and re-derives
        SS_tot from the cleaned data so CoD rises mechanically with any variance
        removed. Measured at +0.0078 against +0.0001 for the same PCs on the same
        data; it is kept for comparison, not as a default. Both need >= 3 runs,
        which combinatorial denoising already requires.
    n_null_surrogates : int, default=0
        Phase-randomised surrogates per PC for the singleton null. 0 keeps the
        historical bare ``delta > 0`` rule, which has no noise floor: CoD rises
        mechanically when residual variance is removed, so unrelated regressors
        clear zero roughly half the time. Singleton mode only.
    null_percentile : float, default=95.0
        Percentile of a PC's own surrogate deltas it must beat to be selected.
    null_seed : int, default=0
        Seed for surrogate generation, so a rerun reproduces the selection.
    device : torch.device, optional
        Compute device. Auto-detected if None.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    CombinatorialDenoiseResults
        Full results including per-run optimal combinations and diagnostics.
    """
    if device is None:
        from fastfuncstuff.utils import get_device

        device = get_device()

    # Validate design inputs
    if design is None and designs_by_hrf is None:
        raise ValueError("Either design or designs_by_hrf must be provided")
    if design is not None and designs_by_hrf is not None:
        raise ValueError("Cannot provide both design and designs_by_hrf")
    if designs_by_hrf is not None and hrf_indices is None:
        raise ValueError("hrf_indices required when designs_by_hrf is provided")

    per_hrf_mode = designs_by_hrf is not None
    if per_hrf_mode:
        assert designs_by_hrf is not None and hrf_indices is not None
        unique_hrf_indices_list = torch.unique(hrf_indices).tolist()
        n_conds = designs_by_hrf[unique_hrf_indices_list[0]].shape[1]
    else:
        assert design is not None
        unique_hrf_indices_list = []
        n_conds = design.shape[1]

    n_voxels, n_timepoints = data.shape
    n_runs = len(run_starts)

    # Generate combinations based on mode
    if singleton_only:
        # Only singletons: baseline + each PC alone
        combinations = [()] + [(i,) for i in range(max_pcs)]
        n_combos = max_pcs + 1
        mode_str = (
            f"Singleton-only (null-calibrated, {n_null_surrogates} surrogates)"
            if n_null_surrogates > 0
            else "Singleton-only (positive-delta PCs)"
        )
        mode_str += f", criterion={criterion}"
    else:
        combinations = generate_all_pc_combinations(max_pcs)
        n_combos = 2**max_pcs
        mode_str = "Exhaustive combinatorial"

    if verbose:
        print("\nCombinatorial PC Denoising")
        print(f"  Mode: {mode_str}")
        print(f"  Runs: {n_runs}")
        print(f"  Max PCs: {max_pcs} -> {n_combos} combinations per run")
        print(f"  Criteria pool spec: {criteria_r2_threshold}")
        if not singleton_only:
            print(f"  Selection strategy: {selection_strategy}")

    if n_null_surrogates > 0 and not singleton_only:
        raise ValueError("null calibration is a singleton-mode rule; pass singleton_only=True")
    if criterion not in ("within_run", "cross_run"):
        raise ValueError(f"criterion must be 'within_run' or 'cross_run', got {criterion!r}")
    if criterion == "cross_run" and n_runs < 3:
        # Two runs leaves one to fit and none to hold out once the target is in.
        raise ValueError(f"criterion='cross_run' needs at least 3 runs, got {n_runs}")
    # Seeded on the host so the selection reproduces regardless of device.
    null_generator = torch.Generator(device="cpu").manual_seed(null_seed)

    per_run_results = []
    noise_pcs_per_run = []

    # Outer loop: hold out each run
    for held_out_idx in range(n_runs):
        # Clear GPU memory from previous fold before starting this one
        torch.cuda.empty_cache()

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Outer fold: holding out run {held_out_idx}")
            print(f"{'=' * 60}")

        train_runs = [r for r in range(n_runs) if r != held_out_idx]

        # ----------------------------------------------------------------
        # Step 1: Fit OLS betas on N-1 training runs
        # ----------------------------------------------------------------
        train_local_starts = _compute_local_run_starts(train_runs, run_starts, n_timepoints)
        train_nuisance = [nuisance_per_run[i] for i in train_runs]
        betas_all = torch.zeros(n_voxels, n_conds, device=torch.device("cpu"))
        chunk_size = estimate_chunk_size(
            n_voxels=n_voxels,
            n_timepoints=sum(
                (run_starts[r + 1] if r < n_runs - 1 else n_timepoints) - run_starts[r]
                for r in train_runs
            ),
            n_regressors=n_conds,
            device=device,
            operation="glm",
        )

        if per_hrf_mode:
            assert designs_by_hrf is not None and hrf_indices is not None
            # Build training timepoint indices for slicing designs
            train_time_indices: list[int] = []
            for r in train_runs:
                start = run_starts[r]
                end = run_starts[r + 1] if r < n_runs - 1 else n_timepoints
                train_time_indices.extend(range(start, end))

            # Slice train data (all voxels, train timepoints only)
            train_data = data[:, train_time_indices]

            # Per-HRF OLS: loop over HRF groups
            for hrf_idx in unique_hrf_indices_list:
                voxel_mask_hrf = hrf_indices == hrf_idx
                group_data = train_data[voxel_mask_hrf, :]

                # Slice this HRF's design to training timepoints
                group_design = designs_by_hrf[hrf_idx][train_time_indices, :]

                # Project nuisance from both data and design
                group_data_clean, group_design_clean = project_out_nuisance_per_run(
                    data=group_data,
                    design=group_design,
                    nuisance_per_run=train_nuisance,
                    run_starts=train_local_starts,
                    device=device,
                )

                # OLS fit for this HRF group
                X = group_design_clean.to(device)
                XtX = X.T @ X + 1e-6 * torch.eye(n_conds, device=device)
                pinv = torch.linalg.inv(XtX) @ X.T

                group_indices = torch.where(voxel_mask_hrf)[0]
                n_group = group_data.shape[0]
                for vs in range(0, n_group, chunk_size):
                    ve = min(vs + chunk_size, n_group)
                    data_chunk = group_data_clean[vs:ve, :].to(device)
                    betas_chunk = (pinv @ data_chunk.T).T
                    betas_all[group_indices[vs:ve]] = betas_chunk.cpu()

                del X, XtX, pinv, group_data_clean, group_design_clean
        else:
            assert design is not None
            train_data, train_design, _ = slice_by_runs(data, design, run_starts, train_runs)

            # Project nuisance from both data and design
            train_data_clean, train_design_clean = project_out_nuisance_per_run(
                data=train_data,
                design=train_design,
                nuisance_per_run=train_nuisance,
                run_starts=train_local_starts,
                device=device,
            )

            # OLS: betas = (X'X)^-1 X'Y
            X = train_design_clean.to(device)
            XtX = X.T @ X
            XtX_reg = XtX + 1e-6 * torch.eye(XtX.shape[0], device=device)
            XtX_inv = torch.linalg.inv(XtX_reg)
            pinv = XtX_inv @ X.T

            for vs in range(0, n_voxels, chunk_size):
                ve = min(vs + chunk_size, n_voxels)
                data_chunk = train_data_clean[vs:ve, :].to(device)
                betas_chunk = (pinv @ data_chunk.T).T
                betas_all[vs:ve] = betas_chunk.cpu()

        if verbose:
            print(f"  Fitted OLS betas: {betas_all.shape}")

        # ----------------------------------------------------------------
        # Step 2: Inner LORO CV on N-1 training runs -> criteria pool
        # ----------------------------------------------------------------
        inner_cv_splits = generate_cv_splits(len(train_runs), strategy=1)

        if per_hrf_mode:
            assert designs_by_hrf is not None and hrf_indices is not None
            # Per-HRF inner CV: loop over groups, scatter R2 back
            inner_r2 = torch.zeros(n_voxels)
            for hrf_idx in unique_hrf_indices_list:
                voxel_mask_hrf = hrf_indices == hrf_idx
                group_data = train_data[voxel_mask_hrf, :]
                # Slice HRF design to training timepoints (train_data already sliced)
                group_design = designs_by_hrf[hrf_idx][train_time_indices, :]

                # Pre-project nuisance from data and design
                group_data_proj, group_design_proj = project_out_nuisance_per_run(
                    data=group_data,
                    design=group_design,
                    nuisance_per_run=train_nuisance,
                    run_starts=train_local_starts,
                    device=device,
                )

                group_r2_result = compute_xval_r2(
                    data=group_data_proj,
                    design_matrix=group_design_proj,
                    run_starts=train_local_starts,
                    stim_indices=list(range(n_conds)),
                    nuisance_indices=[],
                    cv_splits=inner_cv_splits,
                    metric="cod",
                    device=device,
                    verbose=False,
                )
                inner_r2[voxel_mask_hrf] = group_r2_result["r2"]
                del group_data_proj, group_design_proj
        else:
            assert design is not None
            # Build combined design with block-diagonal nuisance
            n_train_tp = train_data.shape[1]
            n_nuisance_cols = train_nuisance[0].shape[1]
            n_inner_runs = len(train_runs)

            combined_cols = n_conds + n_nuisance_cols * n_inner_runs
            combined_design = torch.zeros(n_train_tp, combined_cols, device=device)
            combined_design[:, :n_conds] = train_design.to(device)

            stim_indices = list(range(n_conds))
            nuisance_indices: list[int] = []

            for ri in range(n_inner_runs):
                start = train_local_starts[ri]
                end = train_local_starts[ri + 1] if ri < n_inner_runs - 1 else n_train_tp
                col_start = n_conds + ri * n_nuisance_cols
                col_end = col_start + n_nuisance_cols
                combined_design[start:end, col_start:col_end] = train_nuisance[ri].to(device)
                nuisance_indices.extend(range(col_start, col_end))

            inner_r2_result = compute_xval_r2(
                data=train_data,
                design_matrix=combined_design.cpu(),
                run_starts=train_local_starts,
                stim_indices=stim_indices,
                nuisance_indices=nuisance_indices,
                cv_splits=inner_cv_splits,
                metric="cod",
                device=device,
                verbose=False,
            )
            inner_r2 = inner_r2_result["r2"]

        criteria_mask = select_criteria_voxels(
            inner_r2,
            spec=criteria_r2_threshold,
            fallback_percentile=criteria_fallback_percentile,
            verbose=verbose,
        )
        n_criteria = int(criteria_mask.sum().item())

        if verbose:
            print(f"  Inner CV R2: median={inner_r2[criteria_mask].median().item():.4f}")
            print(f"  Criteria voxels: {n_criteria}")

        # ----------------------------------------------------------------
        # Step 3: Extract k PCs from held-out run's noise pool
        # ----------------------------------------------------------------
        held_start = run_starts[held_out_idx]
        held_end = run_starts[held_out_idx + 1] if held_out_idx < n_runs - 1 else n_timepoints
        held_data = data[:, held_start:held_end]
        held_nuisance = nuisance_per_run[held_out_idx]

        pcs, variance_ratios = extract_pcs_single_run_with_variance(
            run_data=held_data,
            noise_pool_mask=noise_pool_mask,
            nuisance=held_nuisance,
            max_components=max_pcs,
            device=device,
        )
        noise_pcs_per_run.append(pcs.cpu())

        if verbose:
            var_cumsum = np.cumsum(variance_ratios)
            print(f"  PCs extracted: {max_pcs}")
            print(f"  Variance explained: {var_cumsum[-1] * 100:.1f}% cumulative")

        # ----------------------------------------------------------------
        # Step 4: Evaluate all 2^k combinations
        # ----------------------------------------------------------------
        held_data_criteria = held_data[criteria_mask, :]
        betas_criteria = betas_all[criteria_mask, :]

        if verbose:
            print(f"  Evaluating {n_combos} combinations...")

        # CRITICAL: Free training data NOW - we only need betas + held-out run data
        if per_hrf_mode:
            del train_data
        else:
            del train_data, train_design, train_data_clean, train_design_clean
            del X, XtX, XtX_reg, XtX_inv, pinv
        torch.cuda.empty_cache()

        # One scoring path, so the null surrogates are evaluated exactly the way
        # the real PCs are — same criteria voxels, same betas, same aggregation.
        eval_kwargs = dict(
            held_data_criteria=held_data_criteria,
            betas_criteria=betas_criteria,
            held_nuisance=held_nuisance,
            held_start=held_start,
            held_end=held_end,
            design=design,
            designs_by_hrf=designs_by_hrf,
            hrf_indices=hrf_indices,
            criteria_mask=criteria_mask,
            unique_hrf_indices=unique_hrf_indices_list if per_hrf_mode else None,
            device=device,
            verbose=verbose,
        )
        cross_run_kwargs = dict(
            data_criteria=data[criteria_mask, :],
            run_starts=run_starts,
            n_timepoints=n_timepoints,
            nuisance_per_run=nuisance_per_run,
            target_run=held_out_idx,
            design=design,
            designs_by_hrf=designs_by_hrf,
            criteria_hrf_indices=hrf_indices[criteria_mask] if per_hrf_mode else None,
            device=device,
        )

        # kwargs bound at definition time: the closure is rebuilt each outer
        # fold and must not see the next fold's tensors.
        def _score(columns, combos, var_ratios, _cross=cross_run_kwargs, _within=eval_kwargs):
            if criterion == "cross_run":
                return evaluate_combinations_cross_run(
                    pcs=columns, combinations=combos, variance_ratios=var_ratios, **_cross
                )
            return _evaluate_columns_for_run(
                columns=columns, combos=combos, variance_ratios=var_ratios, **_within
            )

        median_cod, var_explained = _score(pcs, combinations, variance_ratios)

        # ----------------------------------------------------------------
        # Step 5: Select optimal combination
        # ----------------------------------------------------------------
        null_thresholds = None
        pc_status = None
        if singleton_only:
            baseline_cod = median_cod[0]
            if n_null_surrogates > 0:
                # A bare `delta > 0` is a sign test with no noise floor: CoD
                # rises mechanically whenever residual variance is removed
                # (d/dv[(R-v)/(T-v)] < 0 for R < T), and the single-run delta
                # is noisy on top of that, so unrelated regressors clear zero
                # about half the time. Each PC is scored against surrogates of
                # itself instead.
                surrogates = phase_randomize(pcs, n_null_surrogates, generator=null_generator)
                null_combos = [()] + [(i,) for i in range(surrogates.shape[1])]
                null_cod, _ = _score(surrogates, null_combos, np.zeros(surrogates.shape[1]))
                best_combo, null_thresholds, pc_status = select_singletons_against_null(
                    median_cod=median_cod,
                    null_cod=null_cod,
                    k=max_pcs,
                    n_surrogates=n_null_surrogates,
                    percentile=null_percentile,
                )
                var_exp_positive = float(sum(variance_ratios[pc] for pc in best_combo))
                if verbose:
                    n_rejected = sum(s == "rejected_null" for s in pc_status)
                    print(
                        f"  Null calibration ({n_null_surrogates} surrogates, "
                        f"p{null_percentile:g}): {len(best_combo)} PC(s) cleared, "
                        f"{n_rejected} positive-but-inside-null rejected"
                    )
                del surrogates
            else:
                # Select all PCs with positive delta vs baseline
                positive_pcs = []
                var_exp_positive = 0.0
                for i, combo in enumerate(combinations):
                    if len(combo) == 1:  # Singleton
                        delta = median_cod[i] - baseline_cod
                        if delta > 0:
                            positive_pcs.append(combo[0])
                            var_exp_positive += var_explained[i]
                best_combo = tuple(positive_pcs)
            # The chosen singletons are applied TOGETHER, and that joint set was
            # never among the scored candidates. One extra scoring call (a single
            # combination) buys a number that means what the summary says it does.
            if best_combo:
                selected_cod, _ = _score(pcs, [tuple(best_combo)], variance_ratios)
                combo_cod = float(selected_cod[0])
            else:
                combo_cod = float(baseline_cod)
            best_idx = 0  # combinations[0] is the empty set: the baseline
        else:
            best_idx, best_combo = select_optimal_combination(
                median_cod,
                combinations,
                strategy=selection_strategy,
            )

        if not singleton_only:
            combo_cod = float(median_cod[best_idx])

        if verbose:
            # 0-indexed, matching optimal_pcs.json, the saved PC columns and the
            # end-of-run summary. A 1-indexed display here made the same fold
            # look like it picked a different PC set in two places.
            best_combo_display = tuple(best_combo)
            if singleton_only:
                print(f"  Selected PCs: {best_combo_display}")
                print(f"  Baseline CoD: {median_cod[0]:.4f}")
                print(f"  Selected CoD: {combo_cod:.4f} ({combo_cod - median_cod[0]:+.4f})")
                if len(best_combo) > 0:
                    print(f"  Variance explained: {var_exp_positive * 100:.1f}%")
            else:
                print(f"  Optimal combination: PCs {best_combo_display}")
                print(f"  Optimal median CoD: {median_cod[best_idx]:.4f}")
                print(f"  Baseline (no PCs) CoD: {median_cod[0]:.4f}")
                if len(best_combo) > 0:
                    print(f"  Variance explained by selected: {var_explained[best_idx] * 100:.1f}%")

        per_run_results.append(
            CombinatorialDenoiseRunResult(
                run_idx=held_out_idx,
                optimal_combination=best_combo,
                optimal_cod=combo_cod,
                baseline_cod=float(median_cod[0]),
                all_cod=median_cod,
                all_var_explained=var_explained,
                all_combinations=combinations,
                explained_variance_ratios=variance_ratios,
                n_criteria_voxels=int(n_criteria),
                null_thresholds=null_thresholds,
                pc_status=pc_status,
            )
        )

        # CRITICAL: Clean up held-out run data before next fold
        del held_data, held_data_criteria, betas_criteria
        del pcs, variance_ratios
        del median_cod, var_explained
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    if verbose:
        print(f"\n{'=' * 60}")
        print("  Combinatorial Denoising Summary")
        print(f"{'=' * 60}")
        for res in per_run_results:
            combo_display = tuple(res.optimal_combination)
            gain = res.optimal_cod - res.baseline_cod
            line = (
                f"  Run {res.run_idx}: PCs {combo_display} "
                f"(CoD={res.optimal_cod:.4f}, {gain:+.4f} vs baseline)"
            )
            if res.pc_status is not None:
                n_rej = sum(s == "rejected_null" for s in res.pc_status)
                line += f", {n_rej} rejected by null"
            print(line)

    metadata = {
        "max_pcs": max_pcs,
        "n_combinations": n_combos,
        "criteria_r2_threshold": criteria_r2_threshold,
        "criteria_fallback_percentile": criteria_fallback_percentile,
        "selection_strategy": selection_strategy,
        "singleton_only": singleton_only,
        "criterion": criterion,
        "n_null_surrogates": n_null_surrogates,
        "null_percentile": null_percentile if n_null_surrogates > 0 else None,
        "null_seed": null_seed if n_null_surrogates > 0 else None,
        "n_runs": n_runs,
        "tr": tr,
    }

    return CombinatorialDenoiseResults(
        per_run_results=per_run_results,
        noise_pool_mask=noise_pool_mask,
        initial_r2=initial_r2,
        noise_pcs_per_run=noise_pcs_per_run,
        metadata=metadata,
    )


# ============================================================================
# Full-brain cross-validated R2 with per-run optimal PCs
# ============================================================================


def compute_optimized_xval_r2(
    data: torch.Tensor,
    design: torch.Tensor,
    run_starts: list[int],
    nuisance_per_run: list[torch.Tensor],
    noise_pcs_per_run: list[torch.Tensor],
    per_run_results: list[CombinatorialDenoiseRunResult],
    device: torch.device | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Compute full-brain LORO cross-validated R2 using per-run optimal PC combos.

    For each held-out run, augments nuisance with that run's selected PCs,
    then uses standard LORO CV (project nuisance from data+design, OLS, predict).

    Parameters
    ----------
    data : torch.Tensor
        fMRI data, (n_voxels, n_timepoints).
    design : torch.Tensor
        Task design, (n_timepoints, n_conditions).
    run_starts : list of int
        Starting timepoint for each run.
    nuisance_per_run : list of torch.Tensor
        Base nuisance per run (polys + motion).
    noise_pcs_per_run : list of torch.Tensor
        All PCs per run, (run_length, k).
    per_run_results : list of CombinatorialDenoiseRunResult
        Results with optimal_combination per run.
    device : torch.device, optional
        Compute device.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    r2 : torch.Tensor
        Per-voxel cross-validated R2, (n_voxels,).
    """
    if device is None:
        from fastfuncstuff.utils import get_device

        device = get_device()

    _n_voxels, n_timepoints = data.shape
    n_runs = len(run_starts)
    n_conds = design.shape[1]

    # Build augmented nuisance per run (base + selected PCs)
    augmented_nuisance = []
    max_nuis_cols = 0
    for run_idx in range(n_runs):
        base_nuis = nuisance_per_run[run_idx]
        selected_pcs_idx = per_run_results[run_idx].optimal_combination
        if len(selected_pcs_idx) > 0:
            selected_pcs = noise_pcs_per_run[run_idx][:, list(selected_pcs_idx)]
            aug = torch.cat([base_nuis, selected_pcs.to(base_nuis.device)], dim=1)
        else:
            aug = base_nuis
        augmented_nuisance.append(aug)
        max_nuis_cols = max(max_nuis_cols, aug.shape[1])

    # Pad to same number of columns
    for run_idx in range(n_runs):
        n_cols = augmented_nuisance[run_idx].shape[1]
        if n_cols < max_nuis_cols:
            pad = torch.zeros(
                augmented_nuisance[run_idx].shape[0],
                max_nuis_cols - n_cols,
                device=augmented_nuisance[run_idx].device,
            )
            augmented_nuisance[run_idx] = torch.cat([augmented_nuisance[run_idx], pad], dim=1)

    # Build combined design with block-diagonal nuisance
    combined_cols = n_conds + max_nuis_cols * n_runs
    combined_design = torch.zeros(n_timepoints, combined_cols, device=device)
    combined_design[:, :n_conds] = design.to(device)

    stim_indices = list(range(n_conds))
    nuisance_indices = []

    for run_idx in range(n_runs):
        start = run_starts[run_idx]
        end = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
        col_start = n_conds + run_idx * max_nuis_cols
        col_end = col_start + max_nuis_cols
        combined_design[start:end, col_start:col_end] = augmented_nuisance[run_idx].to(device)
        nuisance_indices.extend(range(col_start, col_end))

    cv_splits = generate_cv_splits(n_runs, strategy=1)

    if verbose:
        print("  Computing optimized full-brain xval R2...")

    r2_result = compute_xval_r2(
        data=data,
        design_matrix=combined_design.cpu(),
        run_starts=run_starts,
        stim_indices=stim_indices,
        nuisance_indices=nuisance_indices,
        cv_splits=cv_splits,
        metric="cod",
        device=device,
        verbose=verbose,
    )

    r2 = r2_result["r2"]
    assert isinstance(r2, torch.Tensor)
    return r2


def compute_optimized_xval_r2_3dDenoise_style(
    data: torch.Tensor,
    design: torch.Tensor | None,
    run_starts: list[int],
    nuisance_per_run: list[torch.Tensor],
    noise_pcs_per_run: list[torch.Tensor],
    per_run_results: list,
    designs_by_hrf: dict[int, torch.Tensor] | None = None,
    hrf_indices: torch.Tensor | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Compute full-brain LORO cross-validated R2 with per-run optimal PCs.

    This matches 3dDenoisefast.py's approach exactly:
    1. Build augmented nuisance (base + selected PCs) per run
    2. Pre-project augmented nuisance from both data and design
    3. Call compute_xval_r2 with cleaned data/design and nuisance_indices=[]

    Supports per-HRF mode: when designs_by_hrf and hrf_indices are provided,
    processes each HRF group separately with its own design.

    Parameters
    ----------
    data : torch.Tensor
        fMRI data, (n_voxels, n_timepoints).
    design : torch.Tensor or None
        Task design, (n_timepoints, n_conditions). None in per-HRF mode.
    run_starts : list of int
        Starting timepoint for each run.
    nuisance_per_run : list of torch.Tensor
        Base nuisance per run (polys + motion).
    noise_pcs_per_run : list of torch.Tensor
        All PCs per run, (run_length, k).
    per_run_results : list of CombinatorialDenoiseRunResult
        Results with optimal_combination per run.
    designs_by_hrf : dict, optional
        Per-HRF design matrices: {hrf_idx: (n_timepoints, n_conditions)}.
    hrf_indices : torch.Tensor, optional
        Per-voxel HRF index, (n_voxels,).
    device : torch.device, optional
        Compute device.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    r2 : torch.Tensor
        Per-voxel cross-validated R2, (n_voxels,).
    """
    from fastfuncstuff.glm.xval import (
        compute_xval_r2,
        generate_cv_splits,
        project_out_nuisance_per_run,
    )

    if device is None:
        from fastfuncstuff.utils import get_device

        device = get_device()

    n_voxels = data.shape[0]
    n_runs = len(run_starts)
    per_hrf_mode = designs_by_hrf is not None

    # Build augmented nuisance per run (base + selected PCs)
    augmented_nuisance = []
    for run_idx in range(n_runs):
        base_nuis = nuisance_per_run[run_idx]
        selected_pcs_idx = per_run_results[run_idx].optimal_combination
        if len(selected_pcs_idx) > 0:
            selected_pcs = noise_pcs_per_run[run_idx][:, list(selected_pcs_idx)]
            aug = torch.cat([base_nuis, selected_pcs.to(base_nuis.device)], dim=1)
        else:
            aug = base_nuis
        augmented_nuisance.append(aug)

    # Generate CV splits
    cv_splits = generate_cv_splits(n_runs, strategy=1)

    if verbose:
        print("  Computing optimized full-brain xval R2 (3dDenoisefast style)...")

    if per_hrf_mode:
        assert designs_by_hrf is not None and hrf_indices is not None
        # Per-HRF: project nuisance per group, xval per group, scatter back
        unique_hrf = torch.unique(hrf_indices).tolist()
        r2_all = torch.zeros(n_voxels)

        for hrf_idx in unique_hrf:
            voxel_mask = hrf_indices == hrf_idx
            group_data = data[voxel_mask, :]
            group_design = designs_by_hrf[hrf_idx]

            proj_data, proj_design = project_out_nuisance_per_run(
                data=group_data,
                design=group_design,
                nuisance_per_run=augmented_nuisance,
                run_starts=run_starts,
                device=device,
            )

            r2_result = compute_xval_r2(
                data=proj_data,
                design_matrix=proj_design,
                run_starts=run_starts,
                stim_indices=list(range(group_design.shape[1])),
                nuisance_indices=[],
                cv_splits=cv_splits,
                metric="cod",
                device=device,
                verbose=False,
            )
            r2_all[voxel_mask] = r2_result["r2"]
            del proj_data, proj_design

        return r2_all
    else:
        assert design is not None
        # Single-design: project nuisance from all data at once
        projected_data, projected_design = project_out_nuisance_per_run(
            data=data,
            design=design,
            nuisance_per_run=augmented_nuisance,
            run_starts=run_starts,
            device=device,
        )

        r2_result = compute_xval_r2(
            data=projected_data,
            design_matrix=projected_design,
            run_starts=run_starts,
            stim_indices=list(range(design.shape[1])),
            nuisance_indices=[],
            cv_splits=cv_splits,
            metric="cod",
            device=device,
            verbose=verbose,
        )

        r2 = r2_result["r2"]
        assert isinstance(r2, torch.Tensor)
        return r2


def compute_initial_xval_r2(
    data: torch.Tensor,
    design: torch.Tensor | None,
    run_starts: list[int],
    nuisance_per_run: list[torch.Tensor],
    designs_by_hrf: dict[int, torch.Tensor] | None = None,
    hrf_indices: torch.Tensor | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Compute full-brain LORO cross-validated R2 with polynomials only (no PCs).

    Supports per-HRF mode: when designs_by_hrf and hrf_indices are provided,
    loops over unique HRF groups and computes R2 per group, scattering results
    back into the full voxel array.

    Parameters
    ----------
    data : torch.Tensor
        fMRI data, (n_voxels, n_timepoints).
    design : torch.Tensor or None
        Task design, (n_timepoints, n_conditions). None in per-HRF mode.
    run_starts : list of int
        Starting timepoint for each run.
    nuisance_per_run : list of torch.Tensor
        Base nuisance per run (polys + motion).
    designs_by_hrf : dict, optional
        Per-HRF design matrices: {hrf_idx: (n_timepoints, n_conditions)}.
    hrf_indices : torch.Tensor, optional
        Per-voxel HRF index, (n_voxels,).
    device : torch.device, optional
        Compute device.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    r2 : torch.Tensor
        Per-voxel cross-validated R2, (n_voxels,).
    """
    if device is None:
        from fastfuncstuff.utils import get_device

        device = get_device()

    n_voxels, n_timepoints = data.shape
    n_runs = len(run_starts)
    per_hrf_mode = designs_by_hrf is not None

    cv_splits = generate_cv_splits(n_runs, strategy=1)

    if verbose:
        print("  Computing initial full-brain xval R2 (no PCs)...")

    if per_hrf_mode:
        assert designs_by_hrf is not None and hrf_indices is not None
        # Per-HRF mode: loop over groups, pre-project nuisance, scatter R2
        unique_hrf = torch.unique(hrf_indices).tolist()
        r2_all = torch.zeros(n_voxels)

        if verbose:
            print(f"  Processing {len(unique_hrf)} HRF groups...")

        for hrf_idx in unique_hrf:
            voxel_mask = hrf_indices == hrf_idx
            group_data = data[voxel_mask, :]
            group_design = designs_by_hrf[hrf_idx]

            # Pre-project nuisance from both data and design
            proj_data, proj_design = project_out_nuisance_per_run(
                data=group_data,
                design=group_design,
                nuisance_per_run=nuisance_per_run,
                run_starts=run_starts,
                device=device,
            )

            r2_result = compute_xval_r2(
                data=proj_data,
                design_matrix=proj_design,
                run_starts=run_starts,
                stim_indices=list(range(group_design.shape[1])),
                nuisance_indices=[],
                cv_splits=cv_splits,
                metric="cod",
                device=device,
                verbose=False,
            )
            r2_all[voxel_mask] = r2_result["r2"]
            del proj_data, proj_design

        return r2_all
    else:
        assert design is not None
        n_conds = design.shape[1]

        # Build combined design with block-diagonal nuisance
        max_nuis_cols = max(n.shape[1] for n in nuisance_per_run)
        combined_cols = n_conds + max_nuis_cols * n_runs
        combined_design = torch.zeros(n_timepoints, combined_cols, device=device)
        combined_design[:, :n_conds] = design.to(device)

        stim_indices = list(range(n_conds))
        nuisance_indices: list[int] = []

        for run_idx in range(n_runs):
            start = run_starts[run_idx]
            end = run_starts[run_idx + 1] if run_idx < n_runs - 1 else n_timepoints
            col_start = n_conds + run_idx * max_nuis_cols
            nuis = nuisance_per_run[run_idx]
            n_cols_actual = nuis.shape[1]
            combined_design[start:end, col_start : col_start + n_cols_actual] = nuis.to(device)
            nuisance_indices.extend(range(col_start, col_start + max_nuis_cols))

        r2_result = compute_xval_r2(
            data=data,
            design_matrix=combined_design.cpu(),
            run_starts=run_starts,
            stim_indices=stim_indices,
            nuisance_indices=nuisance_indices,
            cv_splits=cv_splits,
            metric="cod",
            device=device,
            verbose=verbose,
        )

        r2 = r2_result["r2"]
        assert isinstance(r2, torch.Tensor)
        return r2


# ============================================================================
# Visualization
# ============================================================================


def plot_singleton_contributions(
    results: CombinatorialDenoiseResults,
    output_prefix: str,
) -> list:
    """
    Plot individual PC contributions (singleton combinations).

    Shows each PC's effect alone as a bar plot. Bars above zero indicate
    the PC improves CoD compared to baseline (no PCs). Bars below zero
    indicate degradation. Useful for understanding interactions - if a PC
    is bad alone but good in combination, that suggests synergistic effects.

    Parameters
    ----------
    results : CombinatorialDenoiseResults
        Full combinatorial denoising results.
    output_prefix : str
        Output file prefix for saving plots.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        Generated figures (one per run).
    """
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    _n_runs = len(results.per_run_results)
    max_pcs = results.metadata.get("max_pcs", 7)

    figs = []

    for run_res in results.per_run_results:
        # With a null in play the deltas span orders of magnitude — one real PC
        # dwarfs the rest — so the margin against each PC's own threshold gets
        # its own symlog panel. Otherwise the near-threshold PCs, which are the
        # whole point of the null, are a flat line at the bottom of the plot.
        has_null = run_res.null_thresholds is not None
        if has_null:
            fig, (ax, ax_margin) = plt.subplots(1, 2, figsize=(max(8, max_pcs * 1.0 + 4), 4))
        else:
            fig, ax = plt.subplots(figsize=(max(4, max_pcs * 0.5 + 2), 4))
            ax_margin = None

        # Find baseline (no PCs) and singleton (one PC) combinations
        baseline_cod = None
        singleton_deltas = []

        for combo, cod in zip(run_res.all_combinations, run_res.all_cod, strict=False):
            if len(combo) == 0:
                baseline_cod = cod
            elif len(combo) == 1:
                singleton_deltas.append((combo[0], cod))

        if baseline_cod is None:
            baseline_cod = 0.0

        # Sort by PC index and compute delta
        singleton_deltas.sort()
        pc_indices = [pc for pc, _ in singleton_deltas]
        deltas = [cod - baseline_cod for _, cod in singleton_deltas]

        # Three states once a null is in play: kept, positive-but-inside-the-null
        # (the ones the old bare `delta > 0` rule would have swept in), and
        # negative. Without a null there are only the two.
        status = run_res.pc_status
        if status is not None:
            palette = {
                "selected": "#2ecc71",
                "rejected_null": "#f39c12",
                "not_selected": "#e74c3c",
            }
            colors = [palette[status[pc]] for pc in pc_indices]
        else:
            colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]

        ax.bar(range(len(pc_indices)), deltas, color=colors, edgecolor="black", alpha=0.7)

        # Draw the bar each PC actually had to clear.
        if run_res.null_thresholds is not None:
            for xi, pc in enumerate(pc_indices):
                ax.hlines(
                    run_res.null_thresholds[pc],
                    xi - 0.4,
                    xi + 0.4,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    zorder=3,
                )

        # Add zero line
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

        # Set fixed y-limits with some padding to prevent squishing
        y_min, y_max = min(deltas), max(deltas)
        if run_res.null_thresholds is not None:
            y_min = min(y_min, float(run_res.null_thresholds.min()))
            y_max = max(y_max, float(run_res.null_thresholds.max()))
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 0.001
        ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)

        # Labels
        ax.set_xticks(range(len(pc_indices)))
        ax.set_xticklabels([f"PC{pc + 1}" for pc in pc_indices])
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Δ CoD (vs. baseline)")
        if status is not None:
            from matplotlib.patches import Patch

            n_rej = sum(s == "rejected_null" for s in status)
            # Two panels: the run identity goes above both, so the axes titles
            # stay short enough not to collide.
            fig.suptitle(
                f"Run {run_res.run_idx}: Individual PC Contributions — "
                f"Baseline CoD = {baseline_cod:.4f} — "
                f"{n_rej} PC(s) positive but inside the null",
                fontsize=10,
            )
            ax.set_title("Δ CoD per PC", fontsize=9)
            ax.legend(
                handles=[
                    Patch(facecolor="#2ecc71", edgecolor="black", label="selected"),
                    Patch(facecolor="#f39c12", edgecolor="black", label="rejected by null"),
                    Patch(facecolor="#e74c3c", edgecolor="black", label="Δ ≤ 0"),
                    Line2D([0], [0], color="black", ls="--", label="null threshold"),
                ],
                fontsize=7,
                loc="best",
            )
        else:
            ax.set_title(
                f"Run {run_res.run_idx}: Individual PC Contributions\n"
                f"Baseline CoD = {baseline_cod:.4f}"
            )

        if ax_margin is not None:
            assert run_res.null_thresholds is not None
            margins = [deltas[i] - run_res.null_thresholds[pc] for i, pc in enumerate(pc_indices)]
            ax_margin.bar(
                range(len(pc_indices)), margins, color=colors, edgecolor="black", alpha=0.7
            )
            ax_margin.axhline(y=0, color="black", linewidth=1.0)
            finite = [abs(m) for m in margins if m != 0]
            ax_margin.set_yscale(
                "symlog", linthresh=max(float(np.median(finite)) if finite else 1e-6, 1e-9)
            )
            ax_margin.set_xticks(range(len(pc_indices)))
            ax_margin.set_xticklabels([f"PC{pc + 1}" for pc in pc_indices])
            ax_margin.set_xlabel("Principal Component")
            ax_margin.set_ylabel("Δ CoD − null threshold (symlog)")
            ax_margin.set_title("Margin over the null (>0 = kept)", fontsize=9)
            ax_margin.grid(alpha=0.3, axis="y")
            fig.tight_layout(rect=(0, 0, 1, 0.93))

        # Use bbox_inches instead of tight_layout to avoid warnings
        singleton_path = f"{output_prefix}_run{run_res.run_idx}_singleton_contributions.png"
        fig.savefig(singleton_path, dpi=150, bbox_inches="tight")
        figs.append(fig)
        plt.close(fig)

    return figs


def plot_plateau_curves(
    results: CombinatorialDenoiseResults,
    output_prefix: str,
) -> list:
    """
    Plot max CoD achievable with N PCs (cumulative maximum curve).

    Shows the best possible CoD using at most N PCs. Helps identify
    diminishing returns - if the curve plateaus, adding more PCs
    doesn't help much.

    Parameters
    ----------
    results : CombinatorialDenoiseResults
        Full combinatorial denoising results.
    output_prefix : str
        Output file prefix for saving plots.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        Generated figures (one per run).
    """
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Skip plateau plot in singleton-only mode (not meaningful)
    if results.metadata.get("singleton_only", False):
        return []

    max_pcs = results.metadata.get("max_pcs", 7)

    figs = []

    for run_res in results.per_run_results:
        fig, ax = plt.subplots(figsize=(6, 4))

        # Group CoD by number of PCs in combination
        cod_by_n = {n: [] for n in range(max_pcs + 1)}
        for combo, cod in zip(run_res.all_combinations, run_res.all_cod, strict=False):
            n_pcs = len(combo)
            if n_pcs <= max_pcs:
                cod_by_n[n_pcs].append(cod)

        # Find baseline and convert to delta
        baseline_cod = cod_by_n[0][0] if cod_by_n[0] else 0.0

        # Compute cumulative max: best achievable with at most N PCs
        n_pcs_sorted = sorted(cod_by_n.keys())
        max_cod_at_n = []
        for n in n_pcs_sorted:
            max_cod = max(cod_by_n[n]) if cod_by_n[n] else baseline_cod
            max_cod_at_n.append((n, max_cod - baseline_cod))  # Store delta

        # Separate x and y for plotting
        n_vals = [n for n, _ in max_cod_at_n]
        delta_vals = [d for _, d in max_cod_at_n]

        # Plot the curve
        ax.plot(
            n_vals,
            delta_vals,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=6,
            color="#3498db",
            label="Best achievable",
        )

        # Fill under curve
        ax.fill_between(n_vals, delta_vals, alpha=0.2, color="#3498db")

        # Mark the optimal combination
        best_idx = np.argmax(run_res.all_cod)
        best_combo = run_res.all_combinations[best_idx]
        best_n = len(best_combo)
        best_delta = run_res.all_cod[best_idx] - baseline_cod

        ax.scatter(
            [best_n],
            [best_delta],
            marker="*",
            s=200,
            color="red",
            edgecolors="black",
            zorder=10,
            label=f"Selected: {best_n} PCs",
        )

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xlabel("Number of PCs in combination")
        ax.set_ylabel("Δ CoD (vs. baseline)")
        ax.set_title(
            f"Run {run_res.run_idx}: CoD Plateau Curve\n"
            f"Best {best_n}-PC combo: Δ = {best_delta:.4f}"
        )
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plateau_path = f"{output_prefix}_run{run_res.run_idx}_plateau_curve.png"
        fig.savefig(plateau_path, dpi=150, bbox_inches="tight")
        figs.append(fig)
        plt.close(fig)

    return figs


def plot_inclusion_heatmap(
    results: CombinatorialDenoiseResults,
    output_prefix: str,
) -> list:
    """
    Plot PC inclusion heatmap with delta R² coloring.

    Shows each PC's delta R² as cell color, with X marks for included PCs.
    Works for both singleton and combinatorial modes.

    Parameters
    ----------
    results : CombinatorialDenoiseResults
        Full combinatorial denoising results.
    output_prefix : str
        Output file prefix for saving plots.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        Generated figures (single heatmap).
    """
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    n_runs = len(results.per_run_results)
    max_pcs = results.metadata.get("max_pcs", 7)

    # Summary heatmap: delta R2 per PC, with X marks for inclusion
    fig_heat, ax_heat = plt.subplots(figsize=(max(4, max_pcs * 0.6 + 1), max(3, n_runs * 0.5 + 1)))

    # Extract delta R2 for each singleton PC (same calculation as bar plot)
    delta_matrix = np.zeros((n_runs, max_pcs), dtype=np.float32)
    inclusion_mask = np.zeros((n_runs, max_pcs), dtype=bool)
    # Positive delta that did not clear its own null — the PCs the bare
    # `delta > 0` rule would have taken. Shown so an over-permissive selection
    # is visible as a field of open circles rather than silently absent.
    rejected_mask = np.zeros((n_runs, max_pcs), dtype=bool)

    for run_res in results.per_run_results:
        run_idx = run_res.run_idx

        # Find baseline and singleton CoDs
        baseline_cod = None
        singleton_deltas = {}
        for combo, cod in zip(run_res.all_combinations, run_res.all_cod, strict=False):
            if len(combo) == 0:
                baseline_cod = cod
            elif len(combo) == 1:
                singleton_deltas[combo[0]] = cod - baseline_cod

        # Fill in delta matrix and inclusion mask
        for pc_idx, delta in singleton_deltas.items():
            delta_matrix[run_idx, pc_idx] = delta
        for pc_idx in run_res.optimal_combination:
            inclusion_mask[run_idx, pc_idx] = True
        if run_res.pc_status is not None:
            for pc_idx, state in enumerate(run_res.pc_status):
                if state == "rejected_null":
                    rejected_mask[run_idx, pc_idx] = True

    # Find global min/max for consistent color scale (centered at 0 if possible)
    delta_min = delta_matrix.min()
    delta_max = delta_matrix.max()
    abs_max = max(abs(delta_min), abs(delta_max))

    # Create diverging colormap centered at 0
    cmap_heat = plt.get_cmap("RdYlGn")  # Red for negative, green for positive

    im = ax_heat.imshow(
        delta_matrix,
        cmap=cmap_heat,
        aspect="auto",
        vmin=-abs_max,
        vmax=abs_max,
    )

    # X = kept, o = positive but inside the null, blank = not a candidate
    for ri in range(n_runs):
        for pi in range(max_pcs):
            if not (inclusion_mask[ri, pi] or rejected_mask[ri, pi]):
                continue
            # Choose text color based on delta sign for visibility
            delta_val = delta_matrix[ri, pi]
            text_color = "white" if abs(delta_val) > abs_max * 0.5 else "black"
            ax_heat.text(
                pi,
                ri,
                "X" if inclusion_mask[ri, pi] else "o",
                ha="center",
                va="center",
                fontweight="bold",
                color=text_color,
                fontsize=10,
            )

    ax_heat.set_xlabel("PC Index (1-based)")
    ax_heat.set_ylabel("Run")
    ax_heat.set_xticks(range(max_pcs))
    ax_heat.set_xticklabels([f"PC{i + 1}" for i in range(max_pcs)], fontsize=8)
    ax_heat.set_yticks(range(n_runs))
    if rejected_mask.any():
        ax_heat.set_title(
            f"PC Delta R² (X = included, o = rejected by null; {int(rejected_mask.sum())} rejected)"
        )
    else:
        ax_heat.set_title("PC Delta R² (X = included)")

    # Colorbar
    cbar = fig_heat.colorbar(im, ax=ax_heat, shrink=0.8)
    cbar.set_label("Δ CoD (vs. baseline)")

    heatmap_path = f"{output_prefix}_combinatorial_heatmap.png"
    fig_heat.savefig(heatmap_path, dpi=150, bbox_inches="tight")

    return [fig_heat]


def plot_combinatorial_results(
    results: CombinatorialDenoiseResults,
    output_prefix: str,
) -> list:
    """
    Generate per-run scatter plots showing CoD vs variance explained.

    Creates a multi-panel figure with one subplot per run. Each point is a
    PC combination, colored by the number of PCs in the subset. The optimal
    combination is marked with a star.

    Parameters
    ----------
    results : CombinatorialDenoiseResults
        Full combinatorial denoising results.
    output_prefix : str
        Output file prefix for saving plots.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        Generated figures.
    """
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Skip scatter plot in singleton-only mode (just generates baseline + singletons)
    if results.metadata.get("singleton_only", False):
        return []

    n_runs = len(results.per_run_results)
    max_pcs = results.metadata.get("max_pcs", 7)

    figs = []

    # Per-run scatter plots (3 columns)
    n_cols = 3
    n_rows = (n_runs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    cmap = plt.get_cmap("viridis")

    for i, run_res in enumerate(results.per_run_results):
        ax = axes[i // n_cols, i % n_cols]

        var_exp = run_res.all_var_explained * 100  # To percent
        cod = run_res.all_cod
        n_pcs_per_combo = np.array([len(c) for c in run_res.all_combinations])

        # Find baseline (no PCs) and compute delta
        baseline_idx = next((j for j, c in enumerate(run_res.all_combinations) if len(c) == 0), 0)
        baseline_cod = cod[baseline_idx]
        cod_delta = cod - baseline_cod

        ax.scatter(
            var_exp,
            cod_delta,
            c=n_pcs_per_combo,
            cmap=cmap,
            s=15,
            alpha=0.6,
            vmin=0,
            vmax=max_pcs,
        )

        # Add zero line (baseline reference)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # Mark optimal
        best_idx = np.argmax(cod)
        best_combo_display = tuple(pc + 1 for pc in run_res.optimal_combination)
        ax.scatter(
            var_exp[best_idx],
            cod_delta[best_idx],
            marker="*",
            s=200,
            c="red",
            edgecolors="black",
            zorder=10,
        )

        ax.set_xlabel("Variance explained (%)")
        ax.set_ylabel("Δ Median CoD (vs. baseline)")
        ax.set_title(f"Run {run_res.run_idx}: PCs {best_combo_display}")

    # Remove empty subplots
    for i in range(n_runs, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, max_pcs), cmap=cmap),
        ax=axes,
        label="# PCs in subset",
        shrink=0.6,
    )

    fig.suptitle("Combinatorial PC Denoising: Δ CoD vs Variance Explained", fontsize=14)

    scatter_path = f"{output_prefix}_combinatorial_scatter.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    figs.append(fig)

    return figs
