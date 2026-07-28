"""Voxel-restriction helpers shared by ffs_denoise / ffs_denoisatorial."""

import numpy as np
import torch

from fastfuncstuff.cli_utils import find_constant_voxels, restrict_voxels


def test_find_constant_voxels_flags_flat_in_any_run():
    # 3 voxels x 2 runs of 5 TRs: v0 fine, v1 flat in run 2 only, v2 all zero.
    data = torch.zeros(3, 10)
    data[0] = torch.arange(10, dtype=torch.float32)
    data[1, :5] = torch.arange(5, dtype=torch.float32)
    valid = find_constant_voxels(data, run_starts=[0, 5])
    assert valid.tolist() == [True, False, False]


def test_restrict_voxels_without_prior_mask():
    volume_shape = (2, 2, 2)
    data = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    keep = torch.tensor([True, False, True, False, True, False, True, False])

    new_data, mask, mask_flat, n_voxels = restrict_voxels(data, keep, volume_shape, None)

    assert n_voxels == 4
    assert new_data.shape == (4, 3)
    assert torch.equal(new_data, data[keep])
    assert mask.shape == volume_shape
    assert mask_flat.tolist() == keep.tolist()


def test_restrict_voxels_composes_with_existing_mask():
    # An earlier mask already dropped voxels 0 and 7; `keep` indexes the
    # surviving rows, but the returned masks must stay full-volume.
    volume_shape = (2, 2, 2)
    prior_flat = np.array([False, True, True, True, True, True, True, False])
    data = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    keep = torch.tensor([True, False, True, True, False, True])

    new_data, mask, mask_flat, n_voxels = restrict_voxels(data, keep, volume_shape, prior_flat)

    assert n_voxels == 4
    assert torch.equal(new_data, data[keep])
    # prior indices [1,2,3,4,5,6] filtered by keep → [1,3,4,6]
    assert np.flatnonzero(mask_flat).tolist() == [1, 3, 4, 6]
    assert mask_flat.sum() == mask.sum()
    # Dropped voxels unmask back to zero, which is the whole point.
    vol = np.zeros(mask_flat.size, dtype=np.float32)
    vol[mask_flat] = new_data[:, 0].numpy()
    assert vol[0] == 0.0 and vol[7] == 0.0
