"""Display-only resampling, and the written session report.

The resampling invariant is the one worth stating twice: it changes how a layer
is *painted* and nothing else. A 3 mm run drawn on a 1 mm anatomical's grid used
to be interpolated there, which painted a resolution the data does not have --
a thresholded cluster's edge landed between real voxels.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer import report
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.vocab import Load, SetMode, SetResample

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")


def _save(path, data, step, tr=0.0):
    aff = np.diag([step, step, step, 1.0])
    aff[:3, 3] = (-24.0, -27.0, -21.0)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), aff)
    if tr:
        img.header["pixdim"][4] = tr
        img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(path))
    return path


@pytest.fixture
def mixed(tmp_path):
    """A 1 mm anatomical and a 3 mm run on the same field of view."""
    rng = np.random.default_rng(4)
    _save(tmp_path / "anat.nii.gz", rng.random((48, 54, 42)) * 100, 1.0)
    _save(tmp_path / "bold.nii.gz", rng.normal(size=(16, 18, 14, 10)), 3.0, tr=2.0)
    return tmp_path


@pytest.fixture
def session(mixed):
    s = ViewerSession(device=CPU)
    s.do(Load(str(mixed / "anat.nii.gz")))
    s.do(Load(str(mixed / "bold.nii.gz")))
    yield s
    s.close()


def _layer(session, stem):
    return next(ly for ly in session.state.layers if ly.name.startswith(stem))


# ---------------------------------------------------------------------------
# which way auto goes
# ---------------------------------------------------------------------------


def test_a_coarse_layer_on_a_fine_grid_is_drawn_nearest(session):
    """The case this exists for: keep the run's voxels visible as voxels."""
    assert session.state.grid.shape == (48, 54, 42)
    assert session.resample_mode(_layer(session, "bold")) == "nearest"


def test_the_layer_defining_the_grid_is_drawn_linear(session):
    assert session.resample_mode(_layer(session, "anat")) == "linear"


def test_a_fine_layer_on_a_coarse_grid_is_drawn_linear(session):
    """The other direction: downsampling the anatomy, where nearest aliases."""
    session.state.layers.move(_layer(session, "bold").key, 0)
    session.state.adopt_grid(_layer(session, "bold").shape, _layer(session, "bold").affine)
    assert session.resample_mode(_layer(session, "anat")) == "linear"


def test_matching_grids_are_not_called_upsampling(tmp_path):
    """Two grids that differ by a rounding error are one grid."""
    rng = np.random.default_rng(5)
    _save(tmp_path / "a.nii.gz", rng.random((16, 18, 14)), 3.0)
    _save(tmp_path / "b.nii.gz", rng.random((16, 18, 14)), 3.0001)
    s = ViewerSession(device=CPU)
    try:
        s.do(Load(str(tmp_path / "a.nii.gz")))
        s.do(Load(str(tmp_path / "b.nii.gz")))
        assert s.resample_mode(s.state.layers.layers[-1]) == "linear"
    finally:
        s.close()


# ---------------------------------------------------------------------------
# it is a choice, and it is only visual
# ---------------------------------------------------------------------------


def test_the_automatic_choice_can_be_overridden_both_ways(session):
    run = _layer(session, "bold")
    session.do(SetResample(run.key, "linear"))
    assert session.resample_mode(_layer(session, "bold")) == "linear"
    session.do(SetResample(run.key, "nearest"))
    assert session.resample_mode(_layer(session, "bold")) == "nearest"
    session.do(SetResample(run.key, "auto"))
    assert session.resample_mode(_layer(session, "bold")) == "nearest"


def test_an_unknown_resample_mode_is_an_error(session):
    with pytest.raises(ValueError):
        session.do(SetResample(_layer(session, "bold").key, "cubic"))


def test_it_replays_through_a_script(session):
    run = _layer(session, "bold")
    session.do(SetResample(run.key, "linear"))
    assert f"SET_RESAMPLE {run.key} linear" in session.to_script()


def test_it_does_not_touch_the_data(session):
    """The worry on seeing an interpolation control beside a statistic map."""
    run = _layer(session, "bold")
    before = np.array(session.store.ensure_ram(run.key), copy=True)
    series = session.timeseries(run.key, (24, 27, 21))
    session.do(SetResample(run.key, "linear"))
    assert np.array_equal(np.asarray(session.store.ensure_ram(run.key)), before)
    assert np.array_equal(session.timeseries(run.key, (24, 27, 21)), series)


def test_it_does_not_invalidate_a_fit(session):
    """SLICES and nothing else: it is a repaint, not a recompute."""
    from fastfuncstuff.viewer.commands import Aspect

    dirty = session.do(SetResample(_layer(session, "bold").key, "nearest"))
    assert dirty & Aspect.SLICES
    assert not dirty & (Aspect.LAYERS | Aspect.GRAPH | Aspect.THRESHOLD)


def test_nearest_draws_only_values_the_data_holds(session):
    """The point of the whole thing, checked on the pixels.

    Interpolating a coarse layer onto a fine grid invents values between the
    voxels; nearest cannot, so every drawn sample is one that exists.
    """
    from fastfuncstuff.viewer.slicing import extract_plane
    from fastfuncstuff.viewer.state import Plane

    run = _layer(session, "bold")
    volume = session.display_volume(run.key)
    held = set(np.unique(np.asarray(volume)).tolist())

    near = extract_plane(volume, session.state.grid, run.affine, Plane.AXIAL, 21, mode="nearest")
    lin = extract_plane(volume, session.state.grid, run.affine, Plane.AXIAL, 21, mode="bilinear")
    assert set(np.unique(np.asarray(near)).tolist()) <= held | {0.0}
    invented = set(np.unique(np.asarray(lin)).tolist()) - held - {0.0}
    assert invented, "bilinear is expected to invent values; that is the problem"


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def test_the_report_names_the_grid_and_every_layer(session):
    text = report.describe(session)
    assert "DEFINES DISPLAY GRID" in text
    assert "anat.nii.gz" in text and "bold.nii.gz" in text
    assert "3.0000 x 3.0000 x 3.0000 mm" in text
    assert "nearest resample" in text


def test_the_report_says_what_the_mode_reads(session):
    session.do(SetMode("instacorr"))
    text = report.describe(session)
    assert "active       instacorr" in text
    assert "-> bold.nii.gz" in text


def test_the_report_carries_a_replayable_script(mixed):
    """The valuable half: a reproduction, not a description."""
    a = ViewerSession(device=CPU)
    try:
        a.do(Load(str(mixed / "anat.nii.gz")))
        a.do(Load(str(mixed / "bold.nii.gz")))
        a.do(SetResample(a.state.layers.layers[-1].key, "linear"))
        text = report.describe(a)
    finally:
        a.close()

    script = text.split("## script", 1)[1].split("\n", 2)[2]
    b = ViewerSession(device=CPU)
    try:
        b.run_script(script)
        assert [ly.name for ly in b.state.layers] == ["anat.nii.gz", "bold.nii.gz"]
        assert b.state.layers.layers[-1].resample == "linear"
        assert b.state.grid.shape == a.state.grid.shape if a.state.grid else True
    finally:
        b.close()


def test_the_report_works_on_an_empty_session():
    """It is a debugging tool; failing when there is a problem is the wrong time."""
    s = ViewerSession(device=CPU)
    try:
        text = report.describe(s)
        assert "(nothing loaded)" in text
    finally:
        s.close()


def test_the_report_flags_obliquity(tmp_path):
    """An oblique dataset is the thing you most want stated, not inferred."""
    rng = np.random.default_rng(6)
    theta = np.deg2rad(12.0)
    aff = np.eye(4)
    aff[:3, :3] = (
        np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        * 3.0
    )
    nib.save(
        nib.Nifti1Image(rng.random((12, 12, 10)).astype(np.float32), aff),
        str(tmp_path / "ob.nii.gz"),
    )
    s = ViewerSession(device=CPU)
    try:
        s.do(Load(str(tmp_path / "ob.nii.gz")))
        text = report.describe(s)
        assert "oblique 12.00 deg" in text
        assert "3.0000 x 3.0000 x 3.0000 mm" in text, "column norms, not the diagonal"
    finally:
        s.close()
