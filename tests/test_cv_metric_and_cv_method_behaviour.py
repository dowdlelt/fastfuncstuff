"""The wired flags have to change the arithmetic, not just be read.

The dead-flag tripwire proves a name appears somewhere outside its own
declaration.  That is not the same as being wired, so these check that
flipping each flag changes which folds are built and which summary is taken.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.design.matrices import fit_penalized_glm_cv, run_cv_folds

# --------------------------------------------------------------------------
# ffs_tps -cv-method
# --------------------------------------------------------------------------


def test_split_half_folds_partition_the_runs():
    assert run_cv_folds("loro", 4) == [[0], [1], [2], [3]]
    assert run_cv_folds("split_half", 4) == [[0, 1], [2, 3]]
    # Odd run counts put the extra run in the second half rather than dropping it.
    assert run_cv_folds("split_half", 5) == [[0, 1], [2, 3, 4]]

    for method, n_runs in [("loro", 4), ("split_half", 4), ("split_half", 5)]:
        folds = run_cv_folds(method, n_runs)
        covered = [r for fold in folds for r in fold]
        assert sorted(covered) == list(range(n_runs)), f"{method} does not cover every run once"


def test_split_half_needs_two_runs():
    with pytest.raises(ValueError, match="at least 2 runs"):
        run_cv_folds("split_half", 1)


def test_unknown_cv_method_is_refused():
    with pytest.raises(ValueError, match="must be 'loro' or 'split_half'"):
        run_cv_folds("bootstrap", 4)


def test_cv_method_changes_the_cv_error_curve():
    """Runs 0-1 carry one response, runs 2-3 the opposite.

    Under LORO three of the four training runs always disagree with the
    held-out run only partly; under split_half the training half never agrees
    with the test half at all.  The two must not produce the same errors.
    """
    torch.manual_seed(0)
    X = torch.randn(80, 4)
    D = np.eye(4, dtype=np.float32)
    run_boundaries = [(0, 20), (20, 40), (40, 60), (60, 80)]

    beta = torch.randn(4, 1)
    y = (X @ beta).T.clone()
    y[:, 40:] = -(X[40:, :] @ beta).T

    _, err_loro = fit_penalized_glm_cv(
        y, X, D, [1e-3, 1e3], run_boundaries, device=torch.device("cpu"), cv_method="loro"
    )
    _, err_split = fit_penalized_glm_cv(
        y, X, D, [1e-3, 1e3], run_boundaries, device=torch.device("cpu"), cv_method="split_half"
    )
    assert not np.allclose(err_loro, err_split), "split_half produced identical errors to loro"


# --------------------------------------------------------------------------
# ffs_pathfinder -cv_metric
# --------------------------------------------------------------------------


def _tiny_cv_problem(n_runs=3, n_tp=24, n_voxels=6, seed=0):
    torch.manual_seed(seed)
    total = n_runs * n_tp
    run_starts = [r * n_tp for r in range(n_runs)]

    design = torch.zeros(total, 2)
    design[:, 0] = torch.sin(torch.arange(total, dtype=torch.float32) * 0.5)
    design[:, 1] = torch.cos(torch.arange(total, dtype=torch.float32) * 0.31)

    betas = torch.randn(n_voxels, 2)
    data = betas @ design.T + 0.2 * torch.randn(n_voxels, total)
    # One fold's worth of the data is badly off, so mean and median across
    # folds cannot coincide.
    data[:, :n_tp] += 8.0 * torch.randn(n_voxels, n_tp)

    noise_pcs = [torch.randn(n_tp, 3) for _ in range(n_runs)]
    nuisance_per_run = [torch.ones(n_tp, 1) for _ in range(n_runs)]
    criteria_mask = torch.ones(n_voxels, dtype=torch.bool)
    cv_splits = [([r for r in range(n_runs) if r != held], [held]) for held in range(n_runs)]
    return dict(
        data=data,
        design_matrix=design,
        noise_pcs=noise_pcs,
        run_starts=run_starts,
        criteria_mask=criteria_mask,
        nuisance_per_run=nuisance_per_run,
        max_pcs=2,
        cv_splits=cv_splits,
        device=torch.device("cpu"),
    )


def test_cv_metric_changes_both_the_pc_curve_and_the_per_voxel_scores():
    """The per-voxel scores select the HRF downstream, so they follow the flag too.

    They used to be summed over folds and divided by n_splits -- always a mean
    -- so -cv_metric median picked the PC count by medians and the per-voxel
    HRF by means.
    """
    from fastfuncstuff.cli.pathfinder import cross_validate_denoising_for_hrf

    problem = _tiny_cv_problem()
    curve_mean, voxel_mean = cross_validate_denoising_for_hrf(**problem, cv_metric="mean")
    curve_median, voxel_median = cross_validate_denoising_for_hrf(**problem, cv_metric="median")

    assert not np.allclose(curve_mean, curve_median), "-cv_metric did not change the PC curve"
    assert not np.allclose(voxel_mean, voxel_median), (
        "-cv_metric did not reach the per-voxel scores that select the HRF"
    )


# --------------------------------------------------------------------------
# ffs_reml -beta_cv with -hrfopt_prefix
# --------------------------------------------------------------------------


def test_reml_refuses_beta_cv_with_a_per_voxel_hrf_library():
    """The combination used to run and quietly ignore the library.

    -beta_cv builds its single-trial design with the canonical HRF and passes
    no library, so -hrfopt_prefix affected nothing while appearing to.  A
    comment saying so is not a fix; the run is refused.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fastfuncstuff.cli.reml",
            "-input",
            "a.nii",
            "-onsets",
            "o.1D",
            "-durations",
            "2",
            "-Rbuck",
            "p",
            "-beta_cv",
            "-hrfopt_prefix",
            "h",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "does not support per-voxel HRFs" in result.stdout
