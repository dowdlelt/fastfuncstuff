"""
Fully held-out evaluation of a denoising solution.

The combinatorial / GLMdenoise selections inside ``ffs_denoisatorial`` are
chosen by cross-validation *within* the input runs, so their cross-validated
R² is optimistic by construction: the same data picked the PCs and scored
them. This module answers the other question — do the task betas that came
out of the winning fit predict runs the selection never saw?

Procedure (one CV split, the same evaluator the internal loop uses):

1. Task betas are fit on the input runs with the winning denoising in place:
   each run's polynomials plus that run's selected noise PCs projected out.
2. The held-out runs get their polynomials projected out — nothing else. No
   PCs are extracted from them and none are removed: denoising is a property
   of the *fit*, and what is being tested is whether it produced better betas.
3. Those betas times the held-out design give a predicted timeseries, and R²
   is computed against the held-out data on the held-out timepoints only.

See [[LORO cross-validation]] for why the nuisance projection has to stay
fold-local — here it is per-run, so it always is.
"""

from __future__ import annotations

import torch

__all__ = ["heldout_prediction_r2"]


def _augmented_nuisance(
    base_nuisance: torch.Tensor,
    pcs: torch.Tensor,
    selection: tuple[int, ...],
) -> torch.Tensor:
    """Concatenate the selected PC columns onto a run's base nuisance."""
    if len(selection) == 0:
        return base_nuisance
    selected = pcs[:, list(selection)].to(base_nuisance.device, base_nuisance.dtype)
    return torch.cat([base_nuisance, selected], dim=1)


def heldout_prediction_r2(
    train_data: torch.Tensor,
    train_run_starts: list[int],
    train_nuisance_per_run: list[torch.Tensor],
    train_pcs_per_run: list[torch.Tensor],
    train_selections: list[tuple[int, ...]],
    test_data: torch.Tensor,
    test_run_starts: list[int],
    test_nuisance_per_run: list[torch.Tensor],
    train_design: torch.Tensor | None = None,
    test_design: torch.Tensor | None = None,
    train_designs_by_hrf: dict[int, torch.Tensor] | None = None,
    test_designs_by_hrf: dict[int, torch.Tensor] | None = None,
    hrf_indices: torch.Tensor | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Fit betas on the input runs, predict the held-out runs, return per-voxel R².

    ``train_*`` and ``test_*`` must describe the same voxels in the same
    order — the caller is responsible for applying the identical mask.

    Parameters
    ----------
    train_data, test_data : torch.Tensor
        ``(n_voxels, n_timepoints)`` for the input runs and the held-out runs.
    train_run_starts, test_run_starts : list of int
        Run start indices, each relative to its own dataset.
    train_nuisance_per_run, test_nuisance_per_run : list of torch.Tensor
        Base (polynomial + user) nuisance per run. The held-out runs get only
        this — no noise PCs are removed from the data being predicted.
    train_pcs_per_run : list of torch.Tensor
        ``(run_length, k)`` variance-ordered noise PCs per input run.
    train_selections : list of tuple of int
        PC indices to project out of each input run: the combinatorial fit's
        per-run winners, or ``range(k)`` for a GLMdenoise-style k.
    train_design, test_design : torch.Tensor, optional
        Task design for each dataset. Mutually exclusive with the per-HRF
        arguments.
    train_designs_by_hrf, test_designs_by_hrf : dict, optional
        Per-HRF designs, keyed by HRF index, for ``-hrf_opt`` mode.
    hrf_indices : torch.Tensor, optional
        Per-voxel HRF index; required with the per-HRF designs.
    device : torch.device, optional
        Compute device.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    torch.Tensor
        ``(n_voxels,)`` R² of the prediction on the held-out timepoints only.

    Notes
    -----
    Train and test data are concatenated into one array so the existing
    :func:`~fastfuncstuff.glm.xval.compute_xval_r2` evaluator can be reused
    verbatim with a single CV split. That costs one extra copy of the data.
    """
    from fastfuncstuff.glm.xval import compute_xval_r2, project_out_nuisance_per_run

    per_hrf_mode = train_designs_by_hrf is not None
    if per_hrf_mode:
        if test_designs_by_hrf is None or hrf_indices is None:
            raise ValueError(
                "per-HRF mode needs test_designs_by_hrf and hrf_indices alongside "
                "train_designs_by_hrf"
            )
    elif train_design is None or test_design is None:
        raise ValueError("provide either train/test_design or the per-HRF designs")

    if train_data.shape[0] != test_data.shape[0]:
        raise ValueError(
            f"train and test data must cover the same voxels: "
            f"{train_data.shape[0]} vs {test_data.shape[0]}"
        )

    if device is None:
        device = train_data.device

    n_train_runs = len(train_run_starts)
    n_test_runs = len(test_run_starts)
    n_train_tp = train_data.shape[1]

    # One dataset, one split: train = every input run, test = every held-out run.
    combined_data = torch.cat([train_data.cpu(), test_data.cpu()], dim=1)
    combined_run_starts = list(train_run_starts) + [n_train_tp + s for s in test_run_starts]
    cv_splits = [
        (
            list(range(n_train_runs)),
            list(range(n_train_runs, n_train_runs + n_test_runs)),
        )
    ]

    # Denoising applies to the fit only: input runs lose their selected PCs,
    # held-out runs lose nothing but their polynomials.
    nuisance = [
        _augmented_nuisance(train_nuisance_per_run[r], train_pcs_per_run[r], train_selections[r])
        for r in range(n_train_runs)
    ]
    nuisance += list(test_nuisance_per_run)

    if verbose:
        n_pcs = sorted({len(s) for s in train_selections})
        print(
            f"  Held-out prediction: fit on {n_train_runs} run(s) "
            f"({n_pcs} PCs removed per run), score on {n_test_runs} run(s)"
        )

    if per_hrf_mode:
        assert train_designs_by_hrf is not None and test_designs_by_hrf is not None
        assert hrf_indices is not None
        r2_all = torch.zeros(combined_data.shape[0])
        for hrf_idx in torch.unique(hrf_indices).tolist():
            voxel_mask = (hrf_indices == hrf_idx).cpu()
            group_design = torch.cat(
                [train_designs_by_hrf[hrf_idx], test_designs_by_hrf[hrf_idx]], dim=0
            )
            proj_data, proj_design = project_out_nuisance_per_run(
                data=combined_data[voxel_mask, :],
                design=group_design,
                nuisance_per_run=nuisance,
                run_starts=combined_run_starts,
                device=device,
            )
            r2_result = compute_xval_r2(
                data=proj_data,
                design_matrix=proj_design,
                run_starts=combined_run_starts,
                stim_indices=list(range(group_design.shape[1])),
                nuisance_indices=[],
                cv_splits=cv_splits,
                metric="cod",
                device=device,
                verbose=False,
            )
            r2_all[voxel_mask] = r2_result["r2"].cpu()
            del proj_data, proj_design
        return r2_all

    assert train_design is not None and test_design is not None
    combined_design = torch.cat([train_design, test_design], dim=0)
    projected_data, projected_design = project_out_nuisance_per_run(
        data=combined_data,
        design=combined_design,
        nuisance_per_run=nuisance,
        run_starts=combined_run_starts,
        device=device,
    )
    r2_result = compute_xval_r2(
        data=projected_data,
        design_matrix=projected_design,
        run_starts=combined_run_starts,
        stim_indices=list(range(combined_design.shape[1])),
        nuisance_indices=[],
        cv_splits=cv_splits,
        metric="cod",
        device=device,
        verbose=verbose,
    )
    return r2_result["r2"]
