"""Tests for slicewise (-slibase / -slibase_sm) regressors in the ARMA(1,1) GLM.

slibase supplies per-slice nuisance regressors (physiological noise) that fold INTO
the REML (a,b) estimate. Implemented by giving each z-slice its own
``[base | slice_block]`` design and dispatching through the generalized grouped
orchestrator. Tests cover:
- column de-interleave for both orderings (vs the AFNI help's worked examples),
- per-voxel slice-index derivation,
- recovery of task + per-slice nuisance betas through ``fit_glm_arma11_grouped``,
- slibase composing with dsort.
"""

import numpy as np
import torch

from fastfuncstuff.design.slibase import (
    deinterleave_slibase,
    voxel_slice_indices,
)
from fastfuncstuff.glm.arma import fit_glm_arma11_grouped


def test_deinterleave_cyclic_matches_afni_example():
    # 3 slices, 2 regressors, mat[:, c] == c so we can read which column lands where.
    mat = torch.arange(6).float().repeat(4, 1)  # (n_time=4, 6)
    blocks = deinterleave_slibase(mat, n_slices=3, slice_major=False)  # (3, 4, 2)
    # AFNI -slibase (cyclic): bb[r*n_slices + s] -> slice s, regressor r
    assert blocks[0, 0].tolist() == [0, 3]  # slice 0: cols 0,3
    assert blocks[1, 0].tolist() == [1, 4]  # slice 1: cols 1,4
    assert blocks[2, 0].tolist() == [2, 5]  # slice 2: cols 2,5


def test_deinterleave_slice_major_matches_afni_example():
    mat = torch.arange(6).float().repeat(4, 1)
    blocks = deinterleave_slibase(mat, n_slices=3, slice_major=True)  # (3, 4, 2)
    # AFNI -slibase_sm (blocked): bb[s*m + r] -> slice s, regressor r
    assert blocks[0, 0].tolist() == [0, 1]  # slice 0: cols 0,1
    assert blocks[1, 0].tolist() == [2, 3]  # slice 1: cols 2,3
    assert blocks[2, 0].tolist() == [4, 5]  # slice 2: cols 4,5


def test_deinterleave_rejects_bad_column_count():
    mat = torch.zeros(4, 5)  # 5 not a multiple of 3
    try:
        deinterleave_slibase(mat, n_slices=3, slice_major=False)
        raise AssertionError("expected ValueError for non-multiple column count")
    except ValueError:
        pass


def test_voxel_slice_indices_under_mask():
    # nz=3, 12 voxels, keep even flat indices -> 0,2,4,6,8,10 % 3
    mask = torch.zeros(12, dtype=torch.bool)
    mask[::2] = True
    idx = voxel_slice_indices(12, n_slices=3, mask_tensor=mask)
    assert idx.tolist() == [0, 2, 1, 0, 2, 1]
    # No mask -> straight arange % nz
    idx2 = voxel_slice_indices(6, n_slices=3, mask_tensor=None)
    assert idx2.tolist() == [0, 1, 2, 0, 1, 2]


def _base_design(n_time):
    t = np.arange(n_time)
    return np.stack(
        [np.ones(n_time), np.sin(2 * np.pi * t / 40.0), np.cos(2 * np.pi * t / 27.0)],
        axis=1,
    ).astype(np.float32)


def test_recovers_task_and_per_slice_nuisance():
    rng = np.random.default_rng(0)
    device = torch.device("cpu")
    n_time, n_slices, m = 200, 3, 2
    per_slice = 40  # voxels per slice
    X = _base_design(n_time)
    p = X.shape[1]
    # Distinct slice regressor blocks per slice.
    slice_blocks = rng.standard_normal((n_slices, n_time, m)).astype(np.float32)
    designs_by_group = {
        s: torch.from_numpy(np.concatenate([X, slice_blocks[s]], axis=1)) for s in range(n_slices)
    }
    # Two slices must get genuinely different designs.
    assert not torch.allclose(designs_by_group[0], designs_by_group[1])

    # Build data slice by slice with known task betas and per-slice nuisance betas.
    voxels, group_idx, beta_true, phi_true = [], [], [], []
    for s in range(n_slices):
        b = rng.standard_normal((per_slice, p)).astype(np.float32) * 2.0
        phi = rng.standard_normal((per_slice, m)).astype(np.float32) * 1.5
        sig = b @ X.T + np.einsum("vr,tr->vt", phi, slice_blocks[s])
        sig = sig + 0.05 * rng.standard_normal(sig.shape).astype(np.float32)
        voxels.append(sig)
        group_idx.append(np.full(per_slice, s))
        beta_true.append(b)
        phi_true.append(phi)
    data = torch.from_numpy(np.concatenate(voxels, axis=0))
    group_indices = torch.from_numpy(np.concatenate(group_idx)).long()
    beta_true = np.concatenate(beta_true, axis=0)
    phi_true = np.concatenate(phi_true, axis=0)

    res = fit_glm_arma11_grouped(
        data,
        designs_by_group,
        group_indices,
        tr=2.0,
        group_label="slice",
        device=device,
        verbose=False,
        use_double=True,
    )
    # No task_indices filter -> betas are all p+m columns.
    betas = res.betas.numpy()
    assert betas.shape == (n_slices * per_slice, p + m)
    assert np.abs(betas[:, :p] - beta_true).max() < 0.05  # task
    assert np.abs(betas[:, p:] - phi_true).max() < 0.05  # per-slice nuisance
    assert res.dof == n_time - (p + m)


def test_slibase_composes_with_dsort():
    rng = np.random.default_rng(1)
    device = torch.device("cpu")
    n_time, n_slices, m = 180, 2, 1
    per_slice = 30
    X = _base_design(n_time)
    p = X.shape[1]
    slice_blocks = rng.standard_normal((n_slices, n_time, m)).astype(np.float32)
    designs_by_group = {
        s: torch.from_numpy(np.concatenate([X, slice_blocks[s]], axis=1)) for s in range(n_slices)
    }
    n_vox = n_slices * per_slice
    dsort = rng.standard_normal((n_vox, 1, n_time)).astype(np.float32)
    gamma = rng.standard_normal((n_vox, 1)).astype(np.float32) * 2.0

    voxels, group_idx = [], []
    v = 0
    for s in range(n_slices):
        for _ in range(per_slice):
            b = rng.standard_normal(p).astype(np.float32)
            phi = rng.standard_normal(m).astype(np.float32)
            sig = b @ X.T + phi @ slice_blocks[s].T + gamma[v, 0] * dsort[v, 0]
            sig = sig + 0.05 * rng.standard_normal(n_time).astype(np.float32)
            voxels.append(sig)
            group_idx.append(s)
            v += 1
    data = torch.from_numpy(np.stack(voxels, axis=0))
    group_indices = torch.tensor(group_idx).long()

    res = fit_glm_arma11_grouped(
        data,
        designs_by_group,
        group_indices,
        tr=2.0,
        group_label="slice",
        device=device,
        verbose=False,
        use_double=True,
        dsort=torch.from_numpy(dsort),
    )
    # dsort coefficient recovered alongside the slicewise nuisance.
    assert res.dsort_betas is not None
    assert np.abs(res.dsort_betas.numpy() - gamma).max() < 0.05
    # DoF reflects base p + slice m + dsort q.
    assert res.dof == n_time - (p + m + 1)
