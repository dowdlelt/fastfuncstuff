"""Pure helpers of the nordic, fc and segment benchmark checks (no tools run)."""

from __future__ import annotations

import numpy as np


def test_grid_rois_tile_the_mask_and_drop_slivers():
    from fastfuncstuff.benchmark.checks.fc import MIN_ROI_VOXELS, grid_rois

    mask = np.zeros((40, 40, 20), bool)
    mask[5:35, 5:35, 2:18] = True
    rois = grid_rois(mask, (3.0, 3.0, 3.0), cube_mm=15.0)
    assert (rois[~mask] == 0).all()
    ids, counts = np.unique(rois[rois > 0], return_counts=True)
    assert ids.tolist() == list(range(1, ids.size + 1))  # renumbered 1..K
    assert counts.min() >= MIN_ROI_VOXELS
    assert counts.max() <= 5 * 5 * 5  # a 15 mm cube at 3 mm
    assert ids.size > 10


def test_read_netcc_picks_the_named_block(tmp_path):
    from fastfuncstuff.benchmark.checks.fc import matrix_agreement, read_netcc

    p = tmp_path / "x.netcc"
    p.write_text(
        "# 2  # N_ROI\n# 2  # N_NETMATS\n# WITH_ROI_LABELS\n  1  2\n# CC\n"
        " 1.0000 0.2500\n 0.2500 1.0000\n# FZ\n 4.0000 0.2554\n 0.2554 4.0000\n"
    )
    cc, fz = read_netcc(p), read_netcc(p, "FZ")
    assert cc.tolist() == [[1.0, 0.25], [0.25, 1.0]]
    assert abs(fz[0, 1] - 0.2554) < 1e-9
    a = np.array([[1, 0.2, 0.3], [0.2, 1, 0.4], [0.3, 0.4, 1]])
    assert matrix_agreement(a, a)["max_abs_diff"] == 0.0


def test_segment_hard_labels_and_truth_scores():
    from fastfuncstuff.benchmark.checks.segment import TRUTH, hard_labels, truth_scores

    gm = np.array([0.9, 0.1, 0.2, 0.05])
    wm = np.array([0.05, 0.8, 0.1, 0.05])
    csf = np.array([0.0, 0.05, 0.6, 0.1])
    lab = hard_labels([gm, wm, csf])
    assert lab.tolist() == [1, 2, 3, 0]  # last voxel sums to 0.2: none
    manual = np.array([3, 41, 0, 0])
    s = truth_scores(lab, manual, TRUTH["ds005216"])
    assert s["gm_dice"] == 1.0 and s["wm_dice"] == 1.0 and s["in_label_agreement"] == 1.0
    s = truth_scores(lab, np.array([2, 42, 0, 0]), TRUTH["ds005216"])  # swapped
    assert s["in_label_agreement"] == 0.0


def test_nordic_memory_guard_and_tsnr():
    from fastfuncstuff.benchmark.checks.nordic import MATLAB_MAX_GB, matlab_gb, median_tsnr

    assert matlab_gb((96, 96, 60, 300)) < MATLAB_MAX_GB
    assert matlab_gb((200, 200, 120, 400)) > MATLAB_MAX_GB
    rng = np.random.default_rng(0)
    x = 100 + rng.standard_normal((4, 4, 2, 200)).astype(np.float32)
    mask = np.ones((4, 4, 2), bool)
    assert 80 < median_tsnr(x, mask, 200) < 120
