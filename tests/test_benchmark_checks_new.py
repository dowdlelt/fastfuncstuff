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
    assert matlab_gb((256, 256, 160, 600)) > MATLAB_MAX_GB
    rng = np.random.default_rng(0)
    x = 100 + rng.standard_normal((4, 4, 2, 200)).astype(np.float32)
    mask = np.ones((4, 4, 2), bool)
    assert 80 < median_tsnr(x, mask, 200) < 120


def test_warp_mni_plan_spreads_groups_and_cohort_scores(tmp_path):
    import nibabel as nib

    from fastfuncstuff.benchmark.checks.warp_mni import WarpMniCheck, cohort_scores
    from fastfuncstuff.benchmark.checks.warp_pair import deformation_judges

    cohort = [{"id": f"{g}-{i}", "image": "h", "brain": "b", "labels": "l"}
              for g in ("OASIS-TRT-20", "NKI-RS-22", "MMRR-21") for i in range(1, 6)]  # fmt: skip
    units = WarpMniCheck().plan({"dataset_id": "MB", "cohort": cohort}, None, 4)
    groups = [k["subject"].rsplit("-", 1)[0] for k, _ in units]
    assert len(units) == 4 and len(set(groups[:3])) == 3  # one per group before any repeats

    lab = np.zeros((10, 10, 10), np.int16)
    lab[2:5, 2:8, 2:8], lab[5:8, 2:8, 2:8] = 1, 2
    shifted = np.roll(lab, 1, axis=1)
    for i, arr in enumerate((lab, lab, shifted)):
        d = tmp_path / f"unit{i:06d}" / "ants"
        d.mkdir(parents=True)
        nib.save(nib.Nifti1Image(arr, np.eye(4)), d / "labels_mni.nii.gz")
    s = cohort_scores(sorted(tmp_path.glob("unit*")), tools=("ants", "sswarper"))
    assert set(s) == {"ants"} and s["ants"]["n_subjects"] == 3
    shift_dice = s["ants"]["pair_dice_median"]  # two of the three pairs involve the shift
    assert 0.5 < shift_dice < 1.0
    assert abs(s["ants"]["pair_dice_mean"] - (1.0 + 2 * shift_dice) / 3) < 1e-12

    detj = np.array([1.0, 0.5, 2.0, -0.1])
    j = deformation_judges(detj, np.ones(4, bool))
    assert j["fold_frac"] == 0.25 and abs(j["logjac_sd"] - np.log([1, 0.5, 2]).std()) < 1e-12
    # A base-space pullback with det J = 2 draws two source voxels into one: tissue squeezed
    # to half. The judges speak of the tissue, so it counts as shrunk, not grown.
    sq = deformation_judges(np.array([1.0, 1.0, 2.5, 1.0]), np.ones(4, bool))
    assert sq["frac_det_below_half"] == 0.25 and sq["frac_det_above_2"] == 0.0
    assert sq["logjac_p001"] < 0


def test_cohort_scores_match_per_label_dice(tmp_path):
    """The bincount scorer gives the per-label-mask Dice: labels missing from some subjects,
    label ids that are not contiguous, background never counted."""
    import itertools

    import nibabel as nib

    from fastfuncstuff.benchmark.checks.qwarp import _dice_per_label
    from fastfuncstuff.benchmark.checks.warp_mni import cohort_scores

    rng = np.random.default_rng(0)
    labs = []
    for i in range(5):
        x = rng.choice(np.array([0, 3, 17, 41, 1002]), size=(12, 11, 10)).astype(np.int32)
        if i == 1:
            x[x == 41] = 0  # one subject lacks a label
        labs.append(x)
        d = tmp_path / f"unit{i:06d}" / "ants"
        d.mkdir(parents=True)
        nib.save(nib.Nifti1Image(x, np.eye(4)), d / "labels_mni.nii.gz")
    s = cohort_scores(sorted(tmp_path.glob("unit*")), tools=("ants",))["ants"]
    want = []
    for a, b in itertools.combinations(labs, 2):
        common = np.intersect1d(np.unique(a), np.unique(b))
        want.append(np.nanmean(_dice_per_label(a, b, common[common != 0])))
    assert abs(s["pair_dice_mean"] - np.mean(want)) < 1e-12
    assert abs(s["pair_dice_p05"] - np.percentile(want, 5)) < 1e-12
    assert s["label_vol_median_by_label"]["41"] == np.median([(x == 41).sum() for x in labs])
    assert "0" not in s["label_vol_median_by_label"]


def test_label_judges_flag_a_lost_label_and_write_a_table():
    from fastfuncstuff.benchmark.checks.warp_pair import deformation_judges, label_judges

    aff = np.zeros((12, 12, 12), np.int32)
    aff[1:6, 1:6, 1:6], aff[6:11, 6:11, 6:11], aff[1:3, 8:10, 8:10] = 1, 2, 3
    warped = aff.copy()
    warped[3:6, 1:6, 1:6] = 0  # label 1 keeps 2/5 of its volume: lost
    warped[aff == 3] = 0  # label 3 vanishes
    detj = np.ones(aff.shape)
    detj[warped == 2] = 2.0
    j, tsv = label_judges(warped, aff, detj, np.array([1, 2, 3]), base=aff)
    assert j["n_labels_lost"] == 2 and j["n_labels_vanished"] == 1
    assert abs(j["label_vol_ratio_min"]) < 1e-12 and j["label_vol_ratio_max"] == 1.0
    assert abs(j["label_logjac_min"] + np.log(2)) < 1e-9  # det J 2 = squeezed to half
    lines = tsv.splitlines()
    assert lines[0].split("\t") == [
        "label",
        "vox_affine",
        "vox_warped",
        "vol_ratio",
        "mean_logjac",
        "dice",
    ]
    assert lines[1].split("\t")[:4] == ["1", "125", "50", "0.4000"] and len(lines) == 4
    d = deformation_judges(np.r_[np.full(98, 1.0), 0.3, 3.0], np.ones(100, bool))
    assert d["frac_det_below_half"] == d["frac_det_above_2"] == 0.01


def test_locomoco_intrinsic_judge_and_hires_qualification():
    """A series that is its median plus noise scores worse than the clean one on every
    intrinsic judge; only voxels under 1.8 mm (isotropic equivalent) qualify."""
    import numpy as np

    from fastfuncstuff.benchmark.checks.locomoco import LocomocoCheck, _intrinsic

    rng = np.random.default_rng(0)
    z, y, x = np.mgrid[0:12, 0:12, 0:12]
    med = (100 + 50 * np.sin(x / 2.0) * np.cos(y / 3.0)).astype(np.float32)
    mask = np.ones(med.shape, bool)
    clean = np.repeat(med[..., None], 20, axis=3) + rng.normal(0, 1, (*med.shape, 20))
    noisy = clean + rng.normal(0, 10, clean.shape)
    a, b = _intrinsic(clean, med, mask), _intrinsic(noisy, med, mask)
    assert a["tsnr_median"] > b["tsnr_median"] and a["vol_r_mean"] > b["vol_r_mean"]
    assert a["vol_r_min"] <= a["vol_r_mean"] <= 1.0

    chk = LocomocoCheck.__new__(LocomocoCheck)
    ds = {"offers": []}
    acq = {"offers": ["epi_2d"], "entities": {}, "n_subjects": 1}
    assert chk.qualifies(ds, {**acq, "voxel_mm": [0.8, 0.8, 0.8]})
    assert not chk.qualifies(ds, {**acq, "voxel_mm": [2.0, 2.0, 2.0]})
    assert not chk.qualifies(ds, {**acq, "voxel_mm": [0.8, 0.8, 0.8], "offers": ["epi_3d"]})


def test_locomoco_agreement_judges():
    import pytest

    """Identical estimates agree perfectly; a sign-flipped one is anti-correlated."""
    import numpy as np

    from fastfuncstuff.benchmark.checks.locomoco import _agreement

    rng = np.random.default_rng(1)
    d = rng.normal(0, 0.3, (500, 12))
    moco = 100 + rng.normal(0, 5, (500, 12))
    fixed = moco + rng.normal(0, 1, (500, 12))
    same = _agreement(d, d.copy(), moco, fixed, fixed.copy())
    assert same["pe_r_vol_median"] == pytest.approx(1.0)
    assert same["correction_r_vol_median"] == pytest.approx(1.0)
    assert same["ts_r_median"] == pytest.approx(1.0) and same["pe_absdiff_median_mm"] == 0
    flip = _agreement(d, -d, moco, fixed, 2 * moco - fixed)
    assert flip["pe_r_vol_median"] < -0.99 and flip["correction_r_vol_median"] < -0.99


def test_locomoco_smoothed_pe_r_ignores_texture():
    """Same smooth field plus independent fine noise: raw r is low, smoothed r is high."""
    from scipy.ndimage import gaussian_filter

    from fastfuncstuff.benchmark.checks.locomoco import _rowwise_r, _smoothed_pe_r

    rng = np.random.default_rng(0)
    shape = (24, 24, 12, 4)
    smooth = gaussian_filter(rng.standard_normal(shape), (4, 4, 4, 0)) * 20
    a = smooth + rng.standard_normal(shape)
    b = smooth + rng.standard_normal(shape)
    mask = np.ones(shape[:3], bool)
    raw = float(np.median(_rowwise_r(a[mask].T, b[mask].T)))
    assert raw < 0.7
    assert _smoothed_pe_r(a, b, mask) > 0.9
