"""RoiSet: ROI files resolved against a grid and a mask (processing/roi_collapse.py)."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.processing.roi_collapse import build_roi_set

CPU = torch.device("cpu")
SHAPE = (4, 3, 2)


def _save(tmp_path, name, arr):
    path = str(tmp_path / name)
    nib.save(nib.Nifti1Image(np.asarray(arr), np.eye(4)), path)
    return path


@pytest.fixture
def labels(tmp_path):
    lab = np.zeros(SHAPE, dtype=np.int16)
    lab[:2, :, 0] = 3
    lab[2:, :, 0] = 7
    return _save(tmp_path, "atlas.nii.gz", lab)


def test_rows_are_roi_cover_and_collapse_is_the_mean(labels):
    roi = build_roi_set(labels, SHAPE)
    assert roi.ids == [3, 7]
    assert roi.rows.sum() == 12  # only z=0 is covered: nothing else would be loaded
    data = torch.arange(roi.n_rows * 2, dtype=torch.float32).reshape(roi.n_rows, 2)
    out = roi.collapse(data, device=CPU)
    flat = np.zeros(SHAPE, dtype=bool)
    flat[:2, :, 0] = True
    expected = data[torch.from_numpy(flat[roi.rows])].mean(dim=0)
    assert torch.allclose(out[0], expected)


def test_mask_intersection_is_reported_and_applied(labels):
    mask = np.ones(SHAPE, dtype=bool)
    mask[0] = False
    roi = build_roi_set(labels, SHAPE, mask=mask)
    assert roi.rows.sum() == 9
    assert not roi.rows[0].any()
    assert any("intersected with -mask" in n for n in roi.notes)


def test_mask_that_empties_every_roi_raises(labels):
    mask = np.zeros(SHAPE, dtype=bool)
    mask[:, :, 1] = True
    with pytest.raises(ValueError, match="no ROI voxel"):
        build_roi_set(labels, SHAPE, mask=mask)


def test_binary_masks_stack_and_are_named_after_the_file(tmp_path):
    a = np.zeros(SHAPE)
    a[0, 0, 0] = 1
    b = np.zeros(SHAPE)
    b[:2, 0, 0] = 1  # overlaps a
    roi = build_roi_set(
        [_save(tmp_path, "lh_v1.nii.gz", a), _save(tmp_path, "rh_v1.nii.gz", b)], SHAPE
    )
    assert roi.names == ["lh_v1", "rh_v1"]
    assert not roi.disjoint
    assert roi.sizes.tolist() == [1.0, 2.0]
    data = torch.tensor([[2.0], [4.0]])  # rows (0,0,0), (1,0,0)
    assert roi.collapse(data, device=CPU).flatten().tolist() == [2.0, 3.0]


def test_mask_emptying_one_roi_drops_it_and_keeps_the_others_index(tmp_path):
    a = np.zeros(SHAPE)
    a[0, 0, 0] = 1
    b = np.zeros(SHAPE)
    b[3, 2, 1] = 1
    mask = np.ones(SHAPE, dtype=bool)
    mask[3, 2, 1] = False
    roi = build_roi_set(
        [_save(tmp_path, "a.nii.gz", a), _save(tmp_path, "b.nii.gz", b)], SHAPE, mask=mask
    )
    assert roi.names == ["a"]
    assert roi.ids == [1]
    assert any("emptied" in n for n in roi.notes)


def test_paint_round_trips_the_roi_mean_and_leaves_uncovered_zero(labels):
    roi = build_roi_set(labels, SHAPE)
    vol = roi.to_volume(np.array([[1.0, 10.0], [2.0, 20.0]]))
    assert vol.shape == (*SHAPE, 2)
    assert vol[0, 0, 0].tolist() == [1.0, 10.0]
    assert vol[3, 2, 0].tolist() == [2.0, 20.0]
    assert not vol[..., 1, :].any()


def test_grid_mismatch_and_non_integer_labels_are_refused(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        build_roi_set(_save(tmp_path, "x.nii.gz", np.ones((2, 2, 2))), SHAPE)
    with pytest.raises(ValueError, match="integer"):
        build_roi_set(_save(tmp_path, "p.nii.gz", np.full(SHAPE, 0.5)), SHAPE)


def test_write_table_and_timecourses(labels, tmp_path):
    roi = build_roi_set(labels, SHAPE)
    tsv = roi.write_table(tmp_path / "t.tsv", {"r2": np.array([0.1, 0.2]), "b": np.ones((2, 2))})
    lines = open(tsv).read().splitlines()
    assert lines[0].split("\t") == ["roi", "name", "n_voxels", "r2", "b_0", "b_1"]
    assert lines[1].split("\t")[:3] == ["3", "3", "6"]
    txt = roi.write_timecourses(
        tmp_path / "c.tsv", np.array([0.0, 2.0]), np.arange(4.0).reshape(2, 2)
    )
    rows = open(txt).read().splitlines()
    assert rows[0].split("\t") == ["time", "3", "7"]
    assert rows[2].split("\t") == ["2", "1", "3"]


# -- surface ROIs ----------------------------------------------------------------
N_VERT = 30


@pytest.fixture
def lh_data(tmp_path):
    from fastfuncstuff.io.gifti import gifti_as_nifti, save_gifti_data

    path = tmp_path / "lh.bold.func.gii"
    save_gifti_data(path, np.zeros((N_VERT, 3), np.float32), {"mesh_fingerprint": "M1"})
    return str(path), gifti_as_nifti(path).header


def _shape_gii(tmp_path, name, values, fingerprint="M1"):
    from fastfuncstuff.io.gifti import save_gifti_data

    meta = {"mesh_fingerprint": fingerprint} if fingerprint else {}
    save_gifti_data(tmp_path / name, np.asarray(values, np.float32), meta, time_series=False)
    return str(tmp_path / name)


def _surface_set(paths, data):
    path, header = data
    return build_roi_set(paths, (N_VERT, 1, 1), reference_header=header, reference_path=path)


def test_surface_label_map_on_the_same_mesh(tmp_path, lh_data):
    lab = np.zeros(N_VERT)
    lab[:10], lab[10:15] = 1, 2
    roi = _surface_set(_shape_gii(tmp_path, "lh.parc.shape.gii", lab), lh_data)
    assert roi.ids == [1, 2] and roi.sizes.tolist() == [10.0, 5.0]
    assert any("vertices" in n for n in roi.notes)


def test_surface_roi_on_another_mesh_or_a_volume_roi_is_refused(tmp_path, lh_data, labels):
    other = _shape_gii(tmp_path, "lh.x.shape.gii", np.ones(N_VERT), fingerprint="M2")
    with pytest.raises(ValueError, match="different mesh"):
        _surface_set(other, lh_data)
    with pytest.raises(ValueError, match="volume ROI for surface data"):
        _surface_set(labels, lh_data)
    with pytest.raises(ValueError, match="surface ROI for volume data"):
        build_roi_set(other, SHAPE)


def test_freesurfer_label_and_annot(tmp_path, lh_data):
    import nibabel.freesurfer as nfs

    from fastfuncstuff.io.freesurfer import write_label

    ids = np.array([2, 3, 4])
    write_label(tmp_path / "V1.lh.label", ids, np.zeros((3, 3)), np.ones(3))
    roi = _surface_set(str(tmp_path / "V1.lh.label"), lh_data)
    assert roi.names == ["V1.lh"] and roi.rows.reshape(-1)[ids].all()
    assert any("vertex count only" in n for n in roi.notes)

    write_label(tmp_path / "rh.MT.label", ids, np.zeros((3, 3)), np.ones(3))
    with pytest.raises(ValueError, match="a rh ROI for lh data"):
        _surface_set(str(tmp_path / "rh.MT.label"), lh_data)

    annot_lab = np.zeros(N_VERT, np.int32)  # 0 = "unknown", not a region
    annot_lab[5:9], annot_lab[20:] = 1, 2
    ctab = np.array([[0, 0, 0, 0, 0], [255, 0, 0, 0, 1], [0, 255, 0, 0, 2]], np.int32)
    nfs.write_annot(str(tmp_path / "lh.aparc.annot"), annot_lab, ctab, ["unknown", "V1", "MT"])
    roi = _surface_set(str(tmp_path / "lh.aparc.annot"), lh_data)
    assert roi.names == ["V1", "MT"] and roi.sizes.tolist() == [4.0, 10.0]


def test_dominant_label_and_its_share_per_roi(labels, tmp_path):
    """Label map: ROI 3 is uniformly entry 5, ROI 7 is 4:2 entries 1:2. A weighted
    4-D set shifts the winner when the weights do."""
    roi = build_roi_set(labels, SHAPE)
    lab = np.zeros(SHAPE, np.int64)
    lab[:2, :, 0] = 5
    lab[2:, :2, 0] = 1
    lab[2:, 2, 0] = 2
    best, share = roi.dominant(lab[roi.rows])
    assert list(best) == [5, 1]
    np.testing.assert_allclose(share, [1.0, 4 / 6])
    w = np.zeros((*SHAPE, 2), np.float32)
    w[2:, :2, 0, 0] = 0.1
    w[2:, 2, 0, 0] = 1.0
    w[:2, :, 0, 1] = 1.0
    wroi = build_roi_set(_save(tmp_path, "w.nii.gz", w), SHAPE)
    best, share = wroi.dominant(lab[wroi.rows])
    assert list(best) == [2, 5] and share[0] == pytest.approx(2 / 2.4)
    with pytest.raises(ValueError, match=">= 0"):
        roi.dominant(np.full(roi.n_rows, -1))
