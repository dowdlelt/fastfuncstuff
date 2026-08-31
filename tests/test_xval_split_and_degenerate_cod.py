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

import pytest
import torch

from fastfuncstuff.glm.xval import (
    _cod_kernel,
    _cod_ratio,
    compute_xval_r2,
    validate_cv_splits,
)


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


def test_constant_voxel_does_not_score_one_in_the_kernel():
    y = torch.full((1, 20), 7.0)
    assert _cod_kernel(y, y.clone()).item() == 0.0
    assert _cod_kernel(y, y + 1.0).item() == 0.0


@pytest.mark.parametrize("r2_method", ["fast", "slow"])
def test_constant_voxel_does_not_score_one_through_compute_xval_r2(r2_method):
    """Through the public entry point, on BOTH finalisation paths.

    Patching _cod_kernel alone did not fix this: the LORO fast path finalises
    from its own sufficient-statistic accumulators and the slow path from its
    own averaged timeseries, and both carried the epsilon.  A test that calls
    the kernel directly passes while the default production path still returns
    0.9989 for a voxel with nothing in it.
    """
    _, design, run_starts = _three_run_problem()
    n_tp = design.shape[0]

    # One constant voxel and one real one, so a bug cannot hide in an
    # all-degenerate batch.
    real = (torch.randn(1, 2) @ design.T) + 0.01 * torch.randn(1, n_tp)
    data = torch.cat([torch.full((1, n_tp), 7.0), real], dim=0)

    r2 = compute_xval_r2(
        data=data,
        design_matrix=design,
        run_starts=run_starts,
        stim_indices=[0, 1],
        nuisance_indices=[],
        cv_splits=[([1, 2], [0]), ([0, 2], [1]), ([0, 1], [2])],
        device=torch.device("cpu"),
        r2_method=r2_method,
        verbose=False,
    )["r2"]

    assert r2[0].item() == 0.0, f"constant voxel scored {r2[0].item():.4f} on the {r2_method} path"
    assert r2[1].item() > 0.9  # the real voxel is unaffected


def test_constant_voxel_does_not_score_one_in_hrf_selection():
    """The HRF-selection scorers clamped SS_tot before dividing, same effect."""
    from fastfuncstuff.design.hrf_selection import _evaluate_hrfs_insample

    _, design, _ = _three_run_problem()
    n_tp = design.shape[0]
    real = (torch.randn(1, 2) @ design.T) + 0.01 * torch.randn(1, n_tp)
    data = torch.cat([torch.full((1, n_tp), 7.0), real], dim=0)

    r2 = _evaluate_hrfs_insample(
        projected_data=data,
        projected_designs=[design],
        device=torch.device("cpu"),
        verbose=False,
    )
    assert r2[0, 0].item() == 0.0
    assert r2[1, 0].item() > 0.9


def test_cod_ratio_is_unchanged_where_there_is_variance():
    ss_res = torch.tensor([1.0, 0.0, 4.0])
    ss_tot = torch.tensor([4.0, 4.0, 4.0])
    assert torch.allclose(_cod_ratio(ss_res, ss_tot), torch.tensor([0.75, 1.0, 0.0]))


# --------------------------------------------------------------------------
# Central split validation
# --------------------------------------------------------------------------


def test_out_of_range_run_index_is_refused():
    with pytest.raises(ValueError, match="outside"):
        validate_cv_splits([([0, 1], [5])], n_runs=3)


def test_duplicate_run_in_one_side_is_refused():
    """A duplicate passes a set-based complement check while double-weighting."""
    with pytest.raises(ValueError, match="more than once"):
        validate_cv_splits([([0, 0, 1], [2])], n_runs=3)


def test_run_in_both_train_and_test_is_refused():
    with pytest.raises(ValueError, match="BOTH train and test"):
        validate_cv_splits([([0, 1], [1])], n_runs=3)


def test_empty_splits_are_refused():
    with pytest.raises(ValueError, match="empty"):
        validate_cv_splits([], n_runs=3)


def test_split_shape_reports_complementarity_and_coverage():
    loro = [([1, 2], [0]), ([0, 2], [1]), ([0, 1], [2])]
    assert validate_cv_splits(loro, 3) == {"complementary": True, "covers_all_runs": True}

    # Run 2 in neither side: not complementary, and not full coverage.
    assert validate_cv_splits([([0], [1])], 3) == {
        "complementary": False,
        "covers_all_runs": False,
    }

    # Complementary but only two of three runs ever tested.
    partial = [([1, 2], [0]), ([0, 2], [1])]
    assert validate_cv_splits(partial, 3) == {
        "complementary": True,
        "covers_all_runs": False,
    }


def test_slow_path_ignores_never_tested_timepoints():
    """A timepoint no fold tested must not enter the denominator.

    Zero-filling it added no residual but pulled the mean towards zero and
    inflated SS_tot, flattering R² with data that was never held out.  The
    fast path accumulates only over what it tested, so it is the honest
    reference the slow path has to match.
    """
    torch.manual_seed(1)
    n_tp, total = 30, 90
    run_starts = [0, n_tp, 2 * n_tp]
    design = torch.zeros(total, 2)
    design[:, 0] = torch.sin(torch.arange(total, dtype=torch.float32) * 0.4)
    design[:, 1] = torch.cos(torch.arange(total, dtype=torch.float32) * 0.23)
    # Noisy on purpose: the honest R² is low, so an inflated one stands out.
    data = torch.randn(6, 2) @ design.T + 1.5 * torch.randn(6, total)

    # Two complementary folds over runs 0 and 1; run 2 is never tested.
    partial = [([1], [0]), ([0], [1])]

    def score(r2_method):
        return compute_xval_r2(
            data=data,
            design_matrix=design,
            run_starts=run_starts,
            stim_indices=[0, 1],
            nuisance_indices=[],
            cv_splits=partial,
            device=torch.device("cpu"),
            r2_method=r2_method,
            verbose=False,
        )["r2"]

    fast, slow = score("fast"), score("slow")
    assert torch.allclose(fast, slow, atol=1e-4), (
        f"slow path median {slow.median():.4f} vs fast {fast.median():.4f} -- "
        "the untested run is still in the denominator"
    )
