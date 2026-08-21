"""Tests for pack_for_shared_task_glm's handling of partially-supplied nuisance.

``append_nuisance_blocks`` pads every run to a common column count, so a block
supplied for one run only (``-ortvec_run motion_r1.1D motion 1``) leaves
genuinely empty columns in the *other* runs' block-diagonal slots. Projection
based callers never noticed -- QR discards zero columns -- but ffs_deconvolve
hands the packed design straight to an unpenalised ``fit_glm`` solve, where the
empty columns make X'X singular and the task betas blow up.
"""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.design.builder import pack_for_shared_task_glm


def _two_run_setup(n_tp: int = 60, n_task: int = 3, n_voxels: int = 5, seed: int = 0):
    """Two runs sharing a task design, each with its own 2-column nuisance block.

    ``extra_per_run`` is the shape ``append_nuisance_blocks`` produces from two
    ``-ortvec_run`` entries: 4 columns wide, but only 2 of them non-zero in any
    given run.
    """
    rng = np.random.default_rng(seed)
    task = [torch.from_numpy(rng.normal(size=(n_tp, n_task)).astype(np.float32)) for _ in range(2)]

    real = [rng.normal(size=(n_tp, 2)).astype(np.float32) for _ in range(2)]
    zeros = np.zeros((n_tp, 2), dtype=np.float32)
    extra_per_run = [
        torch.from_numpy(np.concatenate([real[0], zeros], axis=1)),  # run 0's block, then padding
        torch.from_numpy(np.concatenate([zeros, real[1]], axis=1)),  # padding, then run 1's block
    ]

    true_betas = rng.normal(size=(n_voxels, n_task)).astype(np.float32)
    data = [
        torch.from_numpy(true_betas @ t.numpy().T + rng.normal(scale=0.01, size=(n_voxels, n_tp)))
        .to(torch.float32)
        .contiguous()
        for t in task
    ]
    return data, task, extra_per_run, true_betas


def test_empty_nuisance_columns_are_kept_by_default():
    """Default stays byte-compatible with the projection-based callers."""
    data, task, extra, _ = _two_run_setup()
    packed = pack_for_shared_task_glm(data, task, polort=1, extra_regressors_per_run=extra)

    nuisance = packed.design_concat[:, packed.n_task_cols :]
    n_empty = int((nuisance.abs().amax(dim=0) == 0).sum())
    # 2 padding columns in each of the 2 runs' diagonal slots.
    assert n_empty == 4
    assert len(packed.column_labels) == packed.design_concat.shape[1]


def test_drop_empty_nuisance_removes_only_the_empty_columns():
    data, task, extra, _ = _two_run_setup()
    kept = pack_for_shared_task_glm(
        data, task, polort=1, extra_regressors_per_run=extra, drop_empty_nuisance=True
    )
    full = pack_for_shared_task_glm(data, task, polort=1, extra_regressors_per_run=extra)

    assert kept.n_task_cols == full.n_task_cols
    assert kept.design_concat.shape[1] == full.design_concat.shape[1] - 4
    assert (
        kept.design_concat[:, : kept.n_task_cols] == full.design_concat[:, : full.n_task_cols]
    ).all()
    assert kept.design_concat.abs().amax(dim=0).min() > 0
    assert len(kept.column_labels) == kept.design_concat.shape[1]
    # The surviving labels are the non-empty ones, in their original order.
    surviving = [
        lbl
        for lbl, col in zip(full.column_labels, full.design_concat.T, strict=True)
        if col.abs().max() > 0
    ]
    assert kept.column_labels == surviving


def test_ols_task_betas_survive_partial_per_run_nuisance():
    """The bug of record: without the drop, the OLS solve is rank deficient."""
    data, task, extra, true_betas = _two_run_setup()

    packed = pack_for_shared_task_glm(
        data, task, polort=1, extra_regressors_per_run=extra, drop_empty_nuisance=True
    )
    solved = torch.linalg.lstsq(packed.design_concat, packed.data_concat.T).solution.T
    est = solved[:, : packed.n_task_cols].numpy()
    assert np.allclose(est, true_betas, atol=1e-2)

    # Same design without the drop: X'X is singular, so the Cholesky path that
    # fit_glm prefers cannot factor it at all.
    singular = pack_for_shared_task_glm(data, task, polort=1, extra_regressors_per_run=extra)
    xtx = singular.design_concat.T @ singular.design_concat
    assert torch.linalg.matrix_rank(xtx) < xtx.shape[0]


def test_drop_is_a_no_op_when_every_column_carries_signal():
    data, task, _, _ = _two_run_setup()
    plain = pack_for_shared_task_glm(data, task, polort=1)
    dropped = pack_for_shared_task_glm(data, task, polort=1, drop_empty_nuisance=True)
    assert torch.equal(plain.design_concat, dropped.design_concat)
    assert plain.column_labels == dropped.column_labels
