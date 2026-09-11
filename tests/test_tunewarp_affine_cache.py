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

    xform = load_affine_1D(mat_path, read_info(pair.base).affine, device=torch.device("cpu"))
    assert xform.matrices.shape == (1, 4, 4)


def test_block_matrix_is_refused_with_a_useful_message(tmp_path):
    pair = _pair(tmp_path)
    mat_path = tmp_path / "s.aff12.1D"
    np.savetxt(mat_path, np.eye(4))

    from fastfuncstuff.io.dsetinfo import read_info

    with pytest.raises(ValueError, match="not AFNI .aff12.1D"):
        load_affine_1D(mat_path, read_info(pair.base).affine, device=torch.device("cpu"))


def test_method_slug_survives_a_config_label():
    from fastfuncstuff.processing.tunewarp import method_slug

    assert method_slug("ffs optiwarp_hs c62") == "ffs_optiwarp_hs_c62"
    assert method_slug("AFNI 3dQwarp") == "AFNI_3dQwarp"
    # Path separators would file the volume somewhere else entirely.
    assert "/" not in method_slug("a/b c")


def test_mean_sharpness_falls_when_the_mean_is_blurred():
    """The readout has to move the right way, or the column is decoration."""
    from fastfuncstuff.processing.tunewarp import mean_sharpness

    torch.manual_seed(0)
    sharp = torch.zeros(24, 24, 24)
    sharp[6:18, 6:18, 6:18] = 1.0
    blurred = torch.nn.functional.avg_pool3d(sharp[None, None], kernel_size=5, stride=1, padding=2)[
        0, 0
    ]

    assert mean_sharpness(sharp) > mean_sharpness(blurred)


def test_collect_unions_columns_across_methods(tmp_path):
    """A method with extra columns must not shift everyone else's rows."""
    from fastfuncstuff.processing.tunewarp import collect_diagnostics

    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "summary.tsv").write_text("method\tdice_mean\nold\t0.61\n")
    (tmp_path / "new").mkdir()
    (tmp_path / "new" / "summary.tsv").write_text(
        "method\tdice_mean\tmean_sharpness\nnew\t0.64\t0.21\n"
    )

    collect_diagnostics(tmp_path)
    lines = (tmp_path / "all_summary.tsv").read_text().strip("\n").split("\n")
    cols = lines[0].split("\t")
    assert cols == ["method", "dice_mean", "mean_sharpness"]
    rows = {r.split("\t")[0]: dict(zip(cols, r.split("\t"), strict=True)) for r in lines[1:]}
    assert rows["old"]["dice_mean"] == "0.61"
    assert rows["old"]["mean_sharpness"] == ""
    assert rows["new"]["mean_sharpness"] == "0.21"


def test_fix_reaches_the_diagnostics_config():
    """-fix on -diagnostics must move the knob, not just the recipe's tune list.

    It used to be applied only to the recipe, which searches nothing on this
    path, so the flag was accepted and silently did nothing -- a fold-floor
    sweep produced three bit-identical results.
    """
    from fastfuncstuff.processing.tunespec import fixed_for, parse_fix

    stored = {"hs_alpha": 1.0, "update_sigma": 1.0}
    pinned = fixed_for(parse_fix(["optiwarp.jac_floor=0.0"]), "optiwarp_hs")
    assert pinned == {"jac_floor": 0.0}
    assert {**stored, **pinned}["jac_floor"] == 0.0


def test_fit_cache_round_trips_image_field_and_time(tmp_path):
    """A cached fit must come back identical, with the ORIGINAL wall time.

    Reporting the read time would turn the seconds column into a measure of disk
    speed, and that column is how the frontier weighs a backend.
    """
    from fastfuncstuff.processing.tunewarp import (
        _fit_cache_key,
        _load_cached_fit,
        _save_cached_fit,
    )

    pair = _pair(tmp_path)
    cache = tmp_path / "cache"
    key = _fit_cache_key("optiwarp_hs", {"jac_floor": 0.0}, "common_T1", pair)

    assert _load_cached_fit(cache, pair, key, torch.device("cpu")) is None

    torch.manual_seed(0)
    warped = torch.rand(8, 9, 10)
    field = tuple(torch.rand(8, 9, 10) for _ in range(3))
    header = {"affine": np.diag([1.0, 1.0, 1.0, 1.0])}
    _save_cached_fit(cache, pair, key, warped, field, header, seconds=87.1)

    got = _load_cached_fit(cache, pair, key, torch.device("cpu"))
    assert got is not None
    back, back_field, secs = got
    assert torch.allclose(back, warped, atol=1e-5)
    assert back_field is not None
    for a, b in zip(back_field, field, strict=True):
        assert torch.allclose(a, b, atol=1e-5)
    assert secs == 87.1

    # A different config must not collide with this entry.
    other = _fit_cache_key("optiwarp_hs", {"jac_floor": 0.05}, "common_T1", pair)
    assert _load_cached_fit(cache, pair, other, torch.device("cpu")) is None
