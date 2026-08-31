"""Two ways cross-validated R² claimed a result it had not earned.

1. The per-run fast path builds each fold's training statistics as
   ``all runs - held-out runs``.  That is the training set only when train is
   exactly the complement of test, which ``generate_cv_splits`` guarantees but
   a caller-supplied split does not.
2. A constant voxel has SS_tot == 0.  Divided by an epsilon, a perfect
   prediction of nothing scored 1.0 -- and then won HRF and hyperparameter
   selection.
"""

from __future__ import annotations

import torch

from fastfuncstuff.glm.xval import _cod_kernel, _cod_ratio, compute_xval_r2


def _three_run_problem(n_tp=30, n_voxels=8, seed=0, rogue_run=None):
    """Runs 0 and 1 share a response; ``rogue_run`` gets the opposite one.

    The rogue run is what makes the shortcut visible: if every run agreed,
    quietly training on an excluded run would still land on the right betas
    and the bug would score a perfect fold.
    """
    torch.manual_seed(seed)
    n_runs = 3
    run_starts = [0, n_tp, 2 * n_tp]
    total = n_runs * n_tp
    design = torch.zeros(total, 2)
    design[:, 0] = torch.sin(torch.arange(total, dtype=torch.float32) * 0.4)
    design[:, 1] = torch.cos(torch.arange(total, dtype=torch.float32) * 0.23)
    betas = torch.randn(n_voxels, 2)
    data = betas @ design.T + 0.01 * torch.randn(n_voxels, total)
    if rogue_run is not None:
        lo, hi = run_starts[rogue_run], run_starts[rogue_run] + n_tp
        data[:, lo:hi] = -3.0 * (betas @ design[lo:hi, :].T)
    return data, design, run_starts


def test_incomplete_split_is_not_scored_by_the_complement_shortcut():
    """Run 2 belongs to neither side; treating it as training gives nonsense."""
    data, design, run_starts = _three_run_problem(rogue_run=2)

    # A deliberately incomplete split: train on run 0, test on run 1, run 2 unused.
    results = compute_xval_r2(
        data=data,
        design_matrix=design,
        run_starts=run_starts,
        stim_indices=[0, 1],
        nuisance_indices=[],
        cv_splits=[([0], [1])],
        device=torch.device("cpu"),
        verbose=False,
    )
    r2 = results["r2"]
    assert torch.isfinite(r2).all()
    # The signal is near-noiseless, so an honest fold scores near 1.  The
    # complement shortcut scored this in the thousands-negative.
    assert r2.median() > 0.9, f"median R2 {r2.median():.4f} -- fast path took an invalid shortcut"


def test_complete_loro_split_still_uses_the_fast_path_and_agrees():
    data, design, run_starts = _three_run_problem()
    splits = [([1, 2], [0]), ([0, 2], [1]), ([0, 1], [2])]
    kwargs = dict(
        data=data,
        design_matrix=design,
        run_starts=run_starts,
        stim_indices=[0, 1],
        nuisance_indices=[],
        cv_splits=splits,
        device=torch.device("cpu"),
        verbose=False,
    )
    fast = compute_xval_r2(**kwargs)["r2"]

    import os

    os.environ["FFS_XVAL_LEGACY"] = "1"
    try:
        legacy = compute_xval_r2(**kwargs)["r2"]
    finally:
        del os.environ["FFS_XVAL_LEGACY"]

    assert torch.allclose(fast, legacy, atol=1e-4)


def test_constant_voxel_does_not_score_one():
    y = torch.full((1, 20), 7.0)
    assert _cod_kernel(y, y.clone()).item() == 0.0
    assert _cod_kernel(y, y + 1.0).item() == 0.0


def test_cod_ratio_is_unchanged_where_there_is_variance():
    ss_res = torch.tensor([1.0, 0.0, 4.0])
    ss_tot = torch.tensor([4.0, 4.0, 4.0])
    assert torch.allclose(_cod_ratio(ss_res, ss_tot), torch.tensor([0.75, 1.0, 0.0]))
