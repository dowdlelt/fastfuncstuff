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
2. The held-out runs get their own base nuisance projected out — polynomials,
   plus whatever the caller supplied (``-test_ortvec``). No PCs are extracted
   from them and none are removed: denoising is a property of the *fit*, and
   what is being tested is whether it produced better betas.

   Both scored arms (with and without PCs in the fit) share this identical
   test-side processing, so SS_tot is the same for both and the comparison is
   unbiased. It is, however, low-powered: everything the held-out nuisance
   does not model stays in the denominator, which is why modelling held-out
   motion is worth the extra flag.
3. Those betas times the held-out design give a predicted timeseries, and R²
   is computed against the held-out data on the held-out timepoints only.

See [[LORO cross-validation]] for why the nuisance projection has to stay
fold-local — here it is per-run, so it always is.

Learning curve
--------------
A single held-out R² is a fair but low-powered read on denoising, because
reduced beta variance enters expected SS_res only as ``Var(β̂)·||x_test||²``
against the full held-out noise floor — a term that shrinks like
1/n_train_runs. :func:`heldout_learning_curve` recovers the power by *not*
using all the training runs: fitting on k runs inflates ``Var(β̂)`` by ~N/k,
so the arms separate at small k and converge as k → N. The convergence is
itself the confirmation, since that is what the ``Var(β̂) → 0`` limit predicts.
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np
import torch

from fastfuncstuff.glm.moments import (
    RunMoments,
    compute_run_moments,
    run_bounds,
    solve_from_moments,
)

__all__ = ["heldout_prediction_r2", "heldout_learning_curve", "plot_heldout_learning_curve"]


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


def _subset_time_indices(
    run_starts: list[int], n_timepoints: int, runs: list[int]
) -> tuple[list[int], list[int]]:
    """Timepoint indices for a subset of runs, plus their starts in the subset."""
    time_indices: list[int] = []
    local_starts: list[int] = []
    for r in runs:
        local_starts.append(len(time_indices))
        start, end = run_bounds(run_starts, n_timepoints, r)
        time_indices.extend(range(start, end))
    return time_indices, local_starts


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

    See Also
    --------
    heldout_learning_curve : the same prediction as a function of how many
        training runs the betas came from, which is far more sensitive to a
        denoising that only reduces beta variance.
    """
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

    # Denoising applies to the fit only: input runs lose their selected PCs,
    # held-out runs lose nothing but their own base nuisance.
    train_nuisance = [
        _augmented_nuisance(train_nuisance_per_run[r], train_pcs_per_run[r], train_selections[r])
        for r in range(n_train_runs)
    ]

    if verbose:
        n_pcs = sorted({len(s) for s in train_selections})
        print(
            f"  Held-out prediction: fit on {n_train_runs} run(s) "
            f"({n_pcs} PCs removed per run), score on {len(test_run_starts)} run(s)"
        )

    targets = _prepare_test_targets(
        test_data=test_data,
        test_run_starts=test_run_starts,
        test_nuisance=list(test_nuisance_per_run),
        test_design=test_design,
        test_designs_by_hrf=test_designs_by_hrf,
        hrf_indices=hrf_indices,
        device=device,
    )
    all_runs = list(range(n_train_runs))
    r2 = torch.zeros(train_data.shape[0])
    for target in targets:
        design = (
            train_design
            if train_designs_by_hrf is None
            else train_designs_by_hrf[_target_hrf_index(target, hrf_indices)]
        )
        assert design is not None
        moments = compute_run_moments(
            data=train_data,
            run_starts=train_run_starts,
            nuisance_per_run=train_nuisance,
            design=design,
            voxel_mask=target.voxel_mask,
            device=device,
            store_device=train_data.device,
        )
        scored = _solve_and_score(moments, target, all_runs, device)
        if target.voxel_mask is None:
            r2 = scored
        else:
            r2[target.voxel_mask] = scored
    return r2


def _target_hrf_index(target: _TestTarget, hrf_indices: torch.Tensor | None) -> int:
    """Recover the HRF index a per-HRF target belongs to from its voxel mask."""
    assert target.voxel_mask is not None and hrf_indices is not None
    return int(hrf_indices.cpu()[target.voxel_mask][0].item())


class _TestTarget(NamedTuple):
    """The held-out side, projected once and reused by every fit.

    Nothing about the held-out runs depends on which training runs a fit used,
    so this is loop-invariant — recomputing it per fit was most of the cost and
    most of the host↔device traffic in the learning curve.
    """

    voxel_mask: torch.Tensor | None  # None = every voxel (no per-HRF split)
    data_proj: torch.Tensor  # (V_g, T_test), on the storage device
    design_proj: torch.Tensor  # (T_test, C), on the compute device


def _prepare_test_targets(
    test_data: torch.Tensor,
    test_run_starts: list[int],
    test_nuisance: list[torch.Tensor],
    test_design: torch.Tensor | None,
    test_designs_by_hrf: dict[int, torch.Tensor] | None,
    hrf_indices: torch.Tensor | None,
    device: torch.device,
) -> list[_TestTarget]:
    from fastfuncstuff.glm.xval import project_out_nuisance_per_run

    def _one(data: torch.Tensor, design: torch.Tensor, mask: torch.Tensor | None) -> _TestTarget:
        data_proj, design_proj = project_out_nuisance_per_run(
            data=data,
            design=design,
            nuisance_per_run=test_nuisance,
            run_starts=test_run_starts,
            device=device,
        )
        return _TestTarget(mask, data_proj, design_proj.to(device))

    if test_designs_by_hrf is not None:
        assert hrf_indices is not None
        targets = []
        for hrf_idx in torch.unique(hrf_indices).tolist():
            mask = (hrf_indices == hrf_idx).cpu()
            targets.append(_one(test_data[mask, :], test_designs_by_hrf[hrf_idx], mask))
        return targets

    assert test_design is not None
    return [_one(test_data, test_design, None)]


def _solve_and_score(
    moments: RunMoments,
    target: _TestTarget,
    runs: list[int],
    device: torch.device,
    chunk_size: int = 20000,
) -> torch.Tensor:
    """Betas from the summed moments of *runs*, scored against the held-out side."""
    from fastfuncstuff.glm.xval import compute_r2_metric

    betas = solve_from_moments(
        [moments.xtx[r] for r in runs],
        [moments.xty[r] for r in runs],
        device=device,
        out_dtype=target.design_proj.dtype,
    )

    n_vox = betas.shape[1]
    r2 = torch.zeros(n_vox)
    for cs in range(0, n_vox, chunk_size):
        ce = min(cs + chunk_size, n_vox)
        pred = betas[:, cs:ce].T @ target.design_proj.T
        actual = target.data_proj[cs:ce, :].to(device)
        r2[cs:ce] = compute_r2_metric(actual, pred, metric="cod").cpu()
        del pred, actual
    return r2


def _choose_subsets(
    n_runs: int,
    k: int,
    max_subsets: int,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    """Up to *max_subsets* distinct k-run subsets, exhaustive when cheap.

    Enumerating C(n, k) is fine for the run counts fMRI actually has, but it
    is factorial, so past a cap we sample subsets directly instead.
    """
    from math import comb

    if comb(n_runs, k) <= max_subsets:
        return list(itertools.combinations(range(n_runs), k))
    if comb(n_runs, k) <= 100_000:
        all_combos = list(itertools.combinations(range(n_runs), k))
        picks = rng.choice(len(all_combos), size=max_subsets, replace=False)
        return [all_combos[i] for i in sorted(picks)]

    seen: set[tuple[int, ...]] = set()
    while len(seen) < max_subsets:
        seen.add(tuple(sorted(rng.choice(n_runs, size=k, replace=False).tolist())))
    return sorted(seen)


def heldout_learning_curve(
    train_data: torch.Tensor,
    train_run_starts: list[int],
    train_nuisance_per_run: list[torch.Tensor],
    train_pcs_per_run: list[torch.Tensor],
    arms: dict[str, list[tuple[int, ...]]],
    test_data: torch.Tensor,
    test_run_starts: list[int],
    test_nuisance_per_run: list[torch.Tensor],
    train_design: torch.Tensor | None = None,
    test_design: torch.Tensor | None = None,
    train_designs_by_hrf: dict[int, torch.Tensor] | None = None,
    test_designs_by_hrf: dict[int, torch.Tensor] | None = None,
    hrf_indices: torch.Tensor | None = None,
    arm_pcs_per_run: dict[str, list[torch.Tensor]] | None = None,
    subset_sizes: list[int] | None = None,
    max_subsets: int = 8,
    seed: int = 0,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Held-out R² as a function of how many training runs the betas came from.

    Fitting on k of N runs inflates ``Var(β̂)`` by roughly N/k, so a denoising
    that only reduces beta variance — leaving ``E[β̂]`` alone — still separates
    the arms at small k and converges as k → N. That separation is the signal
    a single all-runs held-out R² is too blunt to resolve.

    Every arm is scored on the *same* subsets, so the per-k differences stay
    paired: the subset draw cancels out of the comparison.

    Parameters
    ----------
    arms : dict of str -> list of tuple of int
        Named per-run PC selections to compare, e.g.
        ``{"denoised": sel, "initial": [()] * n_runs}``. Each list is indexed
        by *global* run index; subsetting picks the entries it needs.
    arm_pcs_per_run : dict, optional
        Per-arm override of ``train_pcs_per_run``, keyed by arm name. Arms not
        listed use the shared PCs.
    subset_sizes : list of int, optional
        Which k to evaluate. Defaults to every k from 1 to n_train_runs.
    max_subsets : int, default=8
        Subsets sampled per k. Exhaustive when C(n, k) is smaller.
    seed : int, default=0
        Seed for subset sampling.

    Returns
    -------
    dict
        ``"subset_sizes"``: the k values.
        ``"curves"``: arm name -> ``(n_k, n_voxels)`` median R² across subsets.
        ``"n_subsets"``: subsets actually evaluated per k.

    Notes
    -----
    The held-out side is projected once, and each arm's per-run normal-equation
    blocks are built once, so the per-subset work is a ``(C, C)`` solve plus one
    pass over the held-out timepoints. Cost scales with ``len(arms) × n_runs``
    projections, not with the number of subsets.
    """
    from tqdm import tqdm

    if device is None:
        device = train_data.device
    if train_data.shape[0] != test_data.shape[0]:
        raise ValueError(
            f"train and test data must cover the same voxels: "
            f"{train_data.shape[0]} vs {test_data.shape[0]}"
        )
    if not arms:
        raise ValueError("need at least one arm to compare")

    if train_designs_by_hrf is not None:
        if test_designs_by_hrf is None or hrf_indices is None:
            raise ValueError(
                "per-HRF mode needs test_designs_by_hrf and hrf_indices alongside "
                "train_designs_by_hrf"
            )
        n_cols = next(iter(train_designs_by_hrf.values())).shape[1]
    elif train_design is None or test_design is None:
        raise ValueError("provide either train/test_design or the per-HRF designs")
    else:
        n_cols = train_design.shape[1]

    n_train_runs = len(train_run_starts)
    n_voxels = train_data.shape[0]
    if subset_sizes is None:
        subset_sizes = list(range(1, n_train_runs + 1))
    bad = [k for k in subset_sizes if k < 1 or k > n_train_runs]
    if bad:
        raise ValueError(f"subset sizes {bad} outside 1..{n_train_runs}")

    rng = np.random.default_rng(seed)
    # Drawn once per k and shared by every arm — the comparison is paired, so
    # the subset draw must not vary between arms.
    subsets_by_k = {k: _choose_subsets(n_train_runs, k, max_subsets, rng) for k in subset_sizes}

    curves = {name: torch.zeros(len(subset_sizes), n_voxels) for name in arms}
    n_subsets = [len(subsets_by_k[k]) for k in subset_sizes]
    total_subsets = sum(n_subsets)

    targets = _prepare_test_targets(
        test_data=test_data,
        test_run_starts=test_run_starts,
        test_nuisance=list(test_nuisance_per_run),
        test_design=test_design,
        test_designs_by_hrf=test_designs_by_hrf,
        hrf_indices=hrf_indices,
        device=device,
    )
    store_device = _moment_store_device(
        n_voxels=n_voxels,
        n_cols=n_cols,
        n_runs=n_train_runs,
        device=device,
        fallback=train_data.device,
    )

    if verbose:
        print(
            f"  Learning curve: k={subset_sizes}, {n_subsets} subset(s) per k, "
            f"{len(arms)} arm(s) → {total_subsets * len(arms)} fits; "
            f"moments on {store_device.type}"
        )

    # The bar tracks the whole job, precompute included: on real data the
    # per-arm projection pass is the slow part and finishing it in silence is
    # exactly the churn this was reported for. Not gated on `verbose` — the
    # CLI defaults to -verb 0 and a long loop still needs to show progress.
    progress = tqdm(
        total=len(arms) * (n_train_runs + total_subsets),
        desc="  Learning curve",
        disable=len(arms) * (n_train_runs + total_subsets) < 8,
        leave=True,
    )

    for name, selections in arms.items():
        # An arm may carry its own PCs: a GLMdenoise reference must keep its
        # noise-pool components even when the arm under test used a different
        # source, or the comparison is against a straw man.
        pcs_for_arm = (
            train_pcs_per_run
            if arm_pcs_per_run is None or name not in arm_pcs_per_run
            else arm_pcs_per_run[name]
        )
        arm_nuisance = [
            _augmented_nuisance(train_nuisance_per_run[r], pcs_for_arm[r], selections[r])
            for r in range(n_train_runs)
        ]
        for target in targets:
            design = (
                train_design
                if train_designs_by_hrf is None
                else train_designs_by_hrf[_target_hrf_index(target, hrf_indices)]
            )
            assert design is not None
            moments = compute_run_moments(
                data=train_data,
                run_starts=train_run_starts,
                nuisance_per_run=arm_nuisance,
                design=design,
                voxel_mask=target.voxel_mask,
                device=device,
                store_device=store_device,
            )
            progress.update(n_train_runs)

            for k_idx, k in enumerate(subset_sizes):
                scored = torch.stack(
                    [
                        _solve_and_score(moments, target, list(runs), device)
                        for runs in subsets_by_k[k]
                    ],
                    dim=0,
                )
                progress.update(len(subsets_by_k[k]))
                median = scored.median(dim=0).values
                if target.voxel_mask is None:
                    curves[name][k_idx] = median
                else:
                    curves[name][k_idx][target.voxel_mask] = median
                del scored

            del moments
        if device.type == "cuda":
            torch.cuda.empty_cache()

    progress.close()
    return {"subset_sizes": subset_sizes, "curves": curves, "n_subsets": n_subsets}


def _moment_store_device(
    n_voxels: int,
    n_cols: int,
    n_runs: int,
    device: torch.device,
    fallback: torch.device,
) -> torch.device:
    """Where to keep the (C, V) per-run X'y blocks.

    They are reread once per subset, so the compute device is much preferred —
    but one arm's blocks are ``n_runs × n_cols × n_voxels`` floats and a
    many-condition design can outgrow VRAM. Fall back to wherever the data
    already lives rather than OOM mid-curve.
    """
    if device.type == "cpu":
        return device
    from fastfuncstuff.memory import get_available_memory

    needed = n_runs * n_cols * n_voxels * 4  # one arm at a time
    available = get_available_memory(device)
    # Half the budget: the scorer still needs room for the prediction chunks
    # and the held-out target on the same device.
    return device if needed < available * 0.5 else fallback


def plot_heldout_learning_curve(
    curve: dict[str, object],
    active_mask: torch.Tensor,
    output_path: str,
    title: str = "Held-out prediction vs training runs",
) -> None:
    """Plot median held-out R² over *active_mask* voxels against training-run count.

    The mask should be the union of the pre- and post-denoising active pools:
    scoring only on voxels the denoised fit calls active would hide the case
    that matters most — a voxel that improved in training and fell apart in
    held-out data.
    """
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    subset_sizes = curve["subset_sizes"]
    curves = curve["curves"]
    assert isinstance(subset_sizes, list) and isinstance(curves, dict)

    mask = active_mask.cpu()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax, ax_delta) = plt.subplots(1, 2, figsize=(11, 4.2))

    reference = curves.get("initial")
    for name, values in curves.items():
        active = values[:, mask]
        median = active.median(dim=1).values.numpy()
        q25 = active.quantile(0.25, dim=1).numpy()
        q75 = active.quantile(0.75, dim=1).numpy()
        line = ax.plot(subset_sizes, median, marker="o", label=name)[0]
        ax.fill_between(subset_sizes, q25, q75, alpha=0.15, color=line.get_color())

        if reference is not None and name != "initial":
            # Paired per voxel: same subsets, same test data, so the delta is
            # the comparison and its spread is what matters.
            delta = (values - reference)[:, mask]
            ax_delta.plot(
                subset_sizes,
                delta.median(dim=1).values.numpy(),
                marker="o",
                label=f"{name} − initial",
            )

    ax.set_xlabel("Training runs used for betas (k)")
    ax.set_ylabel(f"Median held-out R² ({int(mask.sum()):,} active voxels)")
    ax.set_title(title)
    ax.axhline(0, color="0.7", lw=0.8, zorder=0)
    ax.set_xticks(subset_sizes)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax_delta.axhline(0, color="0.5", lw=1.0)
    ax_delta.set_xlabel("Training runs used for betas (k)")
    ax_delta.set_ylabel("Δ median held-out R²")
    ax_delta.set_title("Improvement over no denoising")
    ax_delta.set_xticks(subset_sizes)
    ax_delta.legend(fontsize=8)
    ax_delta.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
