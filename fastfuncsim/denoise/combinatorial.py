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

from dataclasses import dataclass, field
from itertools import combinations as itertools_combinations

import numpy as np
import torch
from tqdm import tqdm

from fastfuncsim.denoise.sequential import _compute_local_run_starts
from fastfuncsim.decomposition.pca import PCA
from fastfuncsim.glm.xval import (
    compute_xval_r2,
    generate_cv_splits,
    project_out_nuisance_per_run,
    slice_by_runs,
)

# ============================================================================
# Data structures
# ============================================================================


@dataclass
class CombinatorialDenoiseRunResult:
    """Results from combinatorial denoising for a single held-out run."""

    run_idx: int
    optimal_combination: tuple[int, ...]  # Selected PC indices
    optimal_cod: float  # Median CoD at optimal
    all_cod: np.ndarray  # (2^k,) CoD per combo
    all_var_explained: np.ndarray  # (2^k,) variance per combo
    all_combinations: list[tuple[int, ...]]  # All 2^k subsets
    explained_variance_ratios: np.ndarray  # (k,) per-PC variance ratios
    n_criteria_voxels: int  # Number of criteria voxels used


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

    # Adaptive chunk_size: scale down with more combinations to keep memory bounded
    # Target: keep (n_combos * T * chunk) at ~300M elements per tensor
    # For 128 combos: chunk=2000 → ~307 MB per tensor (with 2 tensors = ~614 MB)
    # For 1024 combos: chunk=250 → ~153 MB per tensor (with 2 tensors = ~306 MB)
    if criteria_chunk_size is None:
        target_elements = (
            300_000_000  # ~1.2 GB for both tensors (n_combos * T * chunk * 2 * 4 bytes)
        )
        chunk_size = max(100, target_elements // (n_combos * T * 2))
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

    # CRITICAL: For 8192 combos, P_all would be 7.4 GB - must batch combos
    # Process combos in batches: build projections, evaluate all voxels, accumulate
    combo_batch_size = 512  # (512, T, T) ≈ 464 MB, manageable
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
            cod_chunk = 1.0 - ss_res / ss_tot.clamp(min=1e-10)

            # Store directly to CPU
            all_cod[combo_batch_start:combo_batch_end, chunk_start:chunk_end] = cod_chunk.cpu()

            del data_chunk, data_clean_chunk, pred_chunk, cod_chunk

        # Clean up batch tensors before next combo batch
        del P_batch, design_clean_batch
        torch.cuda.empty_cache()

    # Variance explained per combo
    var_explained = np.array([sum(variance_ratios[pc] for pc in combo) for combo in combinations])

    if return_raw_cod:
        return all_cod.numpy(), var_explained

    # Median CoD across criteria voxels
    median_cod = all_cod.median(dim=1).values.numpy()  # (n_combos,)

    return median_cod, var_explained


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


def fit_combinatorial_denoising(
    data: torch.Tensor,
    design: torch.Tensor | None,
    run_starts: list[int],
    tr: float,
    nuisance_per_run: list[torch.Tensor],
    noise_pool_mask: torch.Tensor,
    initial_r2: torch.Tensor,
    max_pcs: int = 7,
    criteria_r2_threshold: float = 0.0,
    selection_strategy: str = "argmax",
    singleton_only: bool = False,
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
    criteria_r2_threshold : float, default=0.0
        Minimum inner-CV R2 for a voxel to be in the criteria pool.
    selection_strategy : str, default="argmax"
        Strategy for selecting optimal combination (see select_optimal_combination).
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
        from fastfuncsim.utils import get_device

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
        mode_str = "Singleton-only (positive-delta PCs)"
    else:
        combinations = generate_all_pc_combinations(max_pcs)
        n_combos = 2**max_pcs
        mode_str = "Exhaustive combinatorial"

    if verbose:
        print("\nCombinatorial PC Denoising")
        print(f"  Mode: {mode_str}")
        print(f"  Runs: {n_runs}")
        print(f"  Max PCs: {max_pcs} -> {n_combos} combinations per run")
        print(f"  Criteria R2 threshold: {criteria_r2_threshold}")
        if not singleton_only:
            print(f"  Selection strategy: {selection_strategy}")

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
        chunk_size = 10000

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

        # Criteria pool: voxels with inner R2 > threshold
        criteria_mask = inner_r2 > criteria_r2_threshold
        n_criteria = criteria_mask.sum().item()

        if n_criteria < 10:
            if verbose:
                print(
                    f"  WARNING: Only {n_criteria} criteria voxels. "
                    f"Lowering threshold to include more..."
                )
            sorted_r2, _ = inner_r2.sort(descending=True)
            fallback_threshold = sorted_r2[n_voxels // 2].item()
            criteria_mask = inner_r2 > fallback_threshold
            n_criteria = criteria_mask.sum().item()

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

        if per_hrf_mode:
            assert designs_by_hrf is not None and hrf_indices is not None
            # Per-HRF evaluation: get raw CoD per group, concatenate, take median
            all_group_cods = []
            criteria_hrf_indices = hrf_indices[criteria_mask]

            for hrf_idx in unique_hrf_indices_list:
                group_criteria = criteria_hrf_indices == hrf_idx
                n_group_criteria = int(group_criteria.sum().item())
                if n_group_criteria == 0:
                    continue

                # Move mask to CPU for indexing CPU tensors
                group_criteria_cpu = group_criteria.cpu()
                group_data = held_data_criteria[group_criteria_cpu, :]
                group_betas = betas_criteria[group_criteria_cpu, :]
                group_design = designs_by_hrf[hrf_idx][held_start:held_end, :]

                raw_cod, _ = evaluate_all_combinations_for_run(
                    run_data_criteria=group_data,
                    run_design=group_design,
                    betas_criteria=group_betas,
                    poly_nuisance=held_nuisance,
                    noise_pcs=pcs,
                    combinations=combinations,
                    variance_ratios=variance_ratios,
                    device=device,
                    return_raw_cod=True,
                    verbose=verbose,
                )
                all_group_cods.append(raw_cod)  # (n_combos, V_group)

            # Aggregate across HRF groups
            all_cod_combined = np.concatenate(all_group_cods, axis=1)
            median_cod = np.median(all_cod_combined, axis=1).astype(np.float64)
            var_explained = np.array(
                [sum(variance_ratios[pc] for pc in combo) for combo in combinations]
            )
        else:
            assert design is not None
            held_design = design[held_start:held_end, :]
            median_cod, var_explained = evaluate_all_combinations_for_run(
                run_data_criteria=held_data_criteria,
                run_design=held_design,
                betas_criteria=betas_criteria,
                poly_nuisance=held_nuisance,
                noise_pcs=pcs,
                combinations=combinations,
                variance_ratios=variance_ratios,
                device=device,
                verbose=verbose,
            )

        # ----------------------------------------------------------------
        # Step 5: Select optimal combination
        # ----------------------------------------------------------------
        if singleton_only:
            # Select all PCs with positive delta vs baseline
            baseline_cod = median_cod[0]
            positive_pcs = []
            var_exp_positive = 0.0
            for i, combo in enumerate(combinations):
                if len(combo) == 1:  # Singleton
                    delta = median_cod[i] - baseline_cod
                    if delta > 0:
                        positive_pcs.append(combo[0])
                        var_exp_positive += var_explained[i]
            best_combo = tuple(positive_pcs)
            best_idx = 0  # Not meaningful in singleton mode
        else:
            best_idx, best_combo = select_optimal_combination(
                median_cod,
                combinations,
                strategy=selection_strategy,
            )

        if verbose:
            best_combo_display = tuple(pc + 1 for pc in best_combo)
            if singleton_only:
                print(f"  Selected PCs: {best_combo_display}")
                print(f"  Baseline CoD: {median_cod[0]:.4f}")
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
                optimal_cod=float(median_cod[best_idx]),
                all_cod=median_cod,
                all_var_explained=var_explained,
                all_combinations=combinations,
                explained_variance_ratios=variance_ratios,
                n_criteria_voxels=int(n_criteria),
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
            combo_display = tuple(pc + 1 for pc in res.optimal_combination)
            print(f"  Run {res.run_idx}: PCs {combo_display} (CoD={res.optimal_cod:.4f})")

    metadata = {
        "max_pcs": max_pcs,
        "n_combinations": n_combos,
        "criteria_r2_threshold": criteria_r2_threshold,
        "selection_strategy": selection_strategy,
        "singleton_only": singleton_only,
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
        from fastfuncsim.utils import get_device

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

    return r2_result["r2"]


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
    from fastfuncsim.glm.xval import compute_xval_r2, generate_cv_splits, project_out_nuisance_per_run

    if device is None:
        from fastfuncsim.utils import get_device

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

        return r2_result["r2"]


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
        from fastfuncsim.utils import get_device

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

        return r2_result["r2"]


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

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    _n_runs = len(results.per_run_results)
    max_pcs = results.metadata.get("max_pcs", 7)

    figs = []

    for run_res in results.per_run_results:
        fig, ax = plt.subplots(figsize=(max(4, max_pcs * 0.5 + 2), 4))

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

        # Color bars by sign (positive=green, negative=red)
        colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]

        ax.bar(range(len(pc_indices)), deltas, color=colors, edgecolor="black", alpha=0.7)

        # Add zero line
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

        # Set fixed y-limits with some padding to prevent squishing
        y_min, y_max = min(deltas), max(deltas)
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 0.001
        ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.15 * y_range)

        # Labels
        ax.set_xticks(range(len(pc_indices)))
        ax.set_xticklabels([f"PC{pc + 1}" for pc in pc_indices])
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Δ CoD (vs. baseline)")
        ax.set_title(
            f"Run {run_res.run_idx}: Individual PC Contributions\nBaseline CoD = {baseline_cod:.4f}"
        )

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

    # Add X marks for included PCs
    for ri in range(n_runs):
        for pi in range(max_pcs):
            if inclusion_mask[ri, pi]:
                # Choose text color based on delta sign for visibility
                delta_val = delta_matrix[ri, pi]
                text_color = "white" if abs(delta_val) > abs_max * 0.5 else "black"
                ax_heat.text(
                    pi,
                    ri,
                    "X",
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
