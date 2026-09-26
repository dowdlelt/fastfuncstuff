"""-noise_comps: tissue-map resampling/erosion and per-run components."""

import nibabel as nib
import numpy as np
import torch

from fastfuncstuff.cli_utils import NuisanceBlock, append_nuisance_blocks_to_design_info
from fastfuncstuff.denoise.tissue_comps import tissue_mask_on_grid, tissue_noise_components

CPU = torch.device("cpu")


def _save(path, data, affine):
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))
    return str(path)


def test_erosion_is_in_data_voxels_after_resampling(tmp_path):
    # Data grid: 2 mm. Map: 1 mm covering the same space. A 6x6x6-data-voxel cube
    # eroded by ONE data voxel keeps 4x4x4 = 64; eroding one MAP voxel before
    # downsampling would keep the whole 6x6x6 cube.
    data_aff = np.diag([2.0, 2.0, 2.0, 1.0])
    map_aff = np.diag([1.0, 1.0, 1.0, 1.0])
    m = np.zeros((40, 40, 40))
    m[8:20, 8:20, 8:20] = 1.0  # data voxels 4..9 in each axis
    path = _save(tmp_path / "map.nii.gz", m, map_aff)
    mask, counts = tissue_mask_on_grid(path, (20, 20, 20), data_aff, erode=1, device=CPU)
    assert counts["resampled"]
    assert counts["above_threshold"] == 6**3
    assert counts["after_erode"] == 4**3
    assert mask[5:9, 5:9, 5:9].all()


def test_mask_keeps_nifti_axis_order(tmp_path):
    aff = np.eye(4)
    m = np.zeros((10, 7, 5))
    m[8, 1, 3] = 1.0  # distinct index on every axis
    path = _save(tmp_path / "map.nii.gz", m, aff)
    mask, counts = tissue_mask_on_grid(path, (10, 7, 5), aff, erode=0, device=CPU)
    assert not counts["resampled"]
    assert mask.shape == (10, 7, 5)
    assert np.argwhere(mask).tolist() == [[8, 1, 3]]


def test_components_are_what_motion_does_not_explain(tmp_path):
    # Tissue voxels carry motion AND a second shared source. With motion in the
    # projected nuisance, the first component must be the second source, not motion.
    rng = np.random.default_rng(0)
    T, n_runs = 120, 2
    shape = (6, 6, 6)
    tissue = np.zeros(shape, bool)
    tissue[1:5, 1:5, 1:5] = True
    files, nuis, novel = [], [], []
    for r in range(n_runs):
        motion = np.cumsum(rng.standard_normal(T))
        motion -= motion.mean()
        other = np.sin(np.linspace(0, 9 * np.pi, T) + r)
        vol = 100 + rng.standard_normal((*shape, T)) * 0.2
        w_m = rng.uniform(1, 3, tissue.sum())  # motion dominates the variance
        w_o = rng.uniform(0.5, 1, tissue.sum())
        vol[tissue] += w_m[:, None] * motion[None] + w_o[:, None] * other[None]
        files.append(_save(tmp_path / f"run{r}.nii.gz", vol, np.eye(4)))
        poly = np.column_stack([np.ones(T), np.linspace(-1, 1, T)])
        nuis.append(torch.as_tensor(np.column_stack([poly, motion]), dtype=torch.float32))
        novel.append(other)
    pcs = tissue_noise_components(files, tissue, 2, [0, T], nuis, device=CPU)
    assert [p.shape for p in pcs] == [(T, 2)] * n_runs
    for p, o, n in zip(pcs, novel, nuis, strict=True):
        assert abs(np.corrcoef(p[:, 0], o)[0, 1]) > 0.95
        assert abs(np.corrcoef(p[:, 0], n[:, 2].numpy())[0, 1]) < 0.2


def test_append_blocks_to_loaded_design_is_block_diagonal():
    X = np.ones((10, 2))
    info = {"matrix": X, "run_starts": [0, 6], "n_regressors": 2, "column_labels": ["a", "b"], "column_groups": [1, -1]}
    block = NuisanceBlock(label="ncomp_wm", per_run=[np.full((6, 1), 2.0), np.full((4, 1), 3.0)])
    added = append_nuisance_blocks_to_design_info(info, [block])
    assert added == 2
    M = info["matrix"]
    assert M.shape == (10, 4) and info["n_regressors"] == 4
    assert np.all(M[:6, 2] == 2) and np.all(M[6:, 2] == 0)
    assert np.all(M[:6, 3] == 0) and np.all(M[6:, 3] == 3)
    assert info["column_labels"][2:] == ["ncomp_wm_r1[0]", "ncomp_wm_r2[0]"]
    assert info["column_groups"][2:] == [0, 0]
