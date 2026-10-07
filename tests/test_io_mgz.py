"""FreeSurfer .mgz/.mgh: read like any other volume, header and voxels."""

from __future__ import annotations

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")


def _mgz(tmp_path, shape=(12, 10, 8), nt=None, name="T1.mgz"):
    aff = np.array([[-1.0, 0, 0, 6], [0, 0, 1, -4], [0, -1, 0, 5], [0, 0, 0, 1]])
    full = shape if nt is None else (*shape, nt)
    data = np.arange(np.prod(full), dtype=np.float32).reshape(full) % 251
    img = nib.MGHImage(data.astype(np.uint8), aff)
    img.header["tr"] = 2000.0
    path = tmp_path / name
    nib.save(img, str(path))
    return path, data, aff


def test_header_only_info_reads_shape_affine_and_ignores_a_t1_tr(tmp_path):
    from fastfuncstuff.io.dsetinfo import read_info

    path, _, aff = _mgz(tmp_path)
    info = read_info(path)
    assert info.storage == "MGH" and info.compression == "gzip"
    assert info.shape == (12, 10, 8, 1)
    np.testing.assert_allclose(info.affine, aff)
    assert info.tr == 0.0  # a 3-D volume's TR is the acquisition's, not a spacing
    series, _, _ = _mgz(tmp_path, nt=3, name="bold.mgh")
    assert read_info(series).tr == pytest.approx(2.0)
    assert read_info(series).compression is None


def test_loads_through_load_nifti_and_read_volume_unchanged(tmp_path):
    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.io.dsetinfo import read_volume

    path, data, aff = _mgz(tmp_path)
    img = load_nifti(path)
    assert isinstance(img, nib.Nifti1Image)
    np.testing.assert_array_equal(np.asarray(img.dataobj), data.astype(np.uint8))
    np.testing.assert_allclose(img.affine, aff)
    vol, _ = read_volume(path)
    np.testing.assert_array_equal(vol, data)


def test_the_viewer_lists_and_opens_one(tmp_path):
    import torch

    from fastfuncstuff.viewer.catalog import _looks_openable
    from fastfuncstuff.viewer.session import ViewerSession

    path, data, aff = _mgz(tmp_path)
    assert _looks_openable("T1.mgz") and _looks_openable("x.mgh")
    session = ViewerSession(device=torch.device("cpu"))
    try:
        key = session.load(str(path))
        layer = session.state.layers.get(key)
        np.testing.assert_allclose(layer.affine, aff)
        np.testing.assert_array_equal(np.asarray(session.volume(key, 0)), data)
    finally:
        session.close()
