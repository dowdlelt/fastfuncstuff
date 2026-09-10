"""The affine tunewarp caches must be a real AFNI .aff12.1D.

Step 0 used to write allineate's raw base-voxel matrix as a 4x4 block under
the ``.aff12.1D`` name, so the file could not be handed to ffs_nwarp or
3dNwarpApply for the head-to-head those matrices exist to support.
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.affine import load_matrix_1D
from fastfuncstuff.processing.nwarpforge import load_affine_1D
from fastfuncstuff.processing.tunewarp import SubjectPair, _migrate_legacy_matrix


def _write(tmp_path, name, affine, shape=(8, 9, 10)):
    nib = pytest.importorskip("nibabel")
    path = tmp_path / name
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), str(path))
    return path


def _pair(tmp_path):
    base = _write(tmp_path, "base.nii.gz", np.diag([1.0, 1.0, 1.0, 1.0]))
    src = _write(tmp_path, "src.nii.gz", np.diag([-0.7, 0.7, 0.7, 1.0]))
    return SubjectPair("s", str(base), str(src))


def test_legacy_block_cache_migrates_and_round_trips(tmp_path):
    pair = _pair(tmp_path)
    m_vox = np.eye(4)
    m_vox[:3, :3] = np.diag([-1.4, 1.4, 1.25])
    m_vox[:3, 3] = [257.6, -5.1, 39.5]

    mat_path = tmp_path / "s.aff12.1D"
    np.savetxt(mat_path, m_vox)

    assert _migrate_legacy_matrix(mat_path, pair) is True
    # One line of 12 numbers now -- which is the whole point.
    assert len(mat_path.read_text().split("\n")[0].split()) == 12
    assert mat_path.read_text().strip().count("\n") == 0

    from fastfuncstuff.io.dsetinfo import read_info

    back = load_matrix_1D(
        mat_path,
        base_affine=read_info(pair.base).affine,
        source_affine=read_info(pair.source).affine,
    ).numpy()
    assert np.abs(back - m_vox).max() < 1e-4

    # Already migrated: a second pass is a no-op, not a double conversion.
    assert _migrate_legacy_matrix(mat_path, pair) is False


def test_migrated_cache_is_readable_by_nwarp(tmp_path):
    pair = _pair(tmp_path)
    m_vox = np.eye(4)
    m_vox[:3, :3] = np.diag([-1.4, 1.4, 1.25])
    mat_path = tmp_path / "s.aff12.1D"
    np.savetxt(mat_path, m_vox)
    _migrate_legacy_matrix(mat_path, pair)

    from fastfuncstuff.io.dsetinfo import read_info

    xform = load_affine_1D(
        mat_path, read_info(pair.base).affine, device=torch.device("cpu")
    )
    assert xform.matrices.shape == (1, 4, 4)


def test_block_matrix_is_refused_with_a_useful_message(tmp_path):
    pair = _pair(tmp_path)
    mat_path = tmp_path / "s.aff12.1D"
    np.savetxt(mat_path, np.eye(4))

    from fastfuncstuff.io.dsetinfo import read_info

    with pytest.raises(ValueError, match="not AFNI .aff12.1D"):
        load_affine_1D(mat_path, read_info(pair.base).affine, device=torch.device("cpu"))
