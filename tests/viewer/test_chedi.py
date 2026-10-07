"""CHEDI: cortex laid flat around the crosshair, the anatomy sampled at a depth."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

nib = pytest.importorskip("nibabel")
nfs = pytest.importorskip("nibabel.freesurfer")

from tests.viewer.test_surfaces import RADIUS, _shell_anat, _sphere, _subject  # noqa: E402

CPU = torch.device("cpu")
#: The phantom: WM inside 21 mm, GM to 24, CSF beyond; white at 20, pial at 24.
WM, GM, CSF = 110.25, 70.25, 20.25


def _with_shape(subject, name, scale):
    """Add ``lh.<name>`` (sphere or inflated) to a test subject."""
    info = nfs.read_geometry(str(subject / "surf" / "lh.white"), read_metadata=True)[2]
    v, f = _sphere(RADIUS)
    nfs.write_geometry(str(subject / "surf" / f"lh.{name}"), v * scale, f, volume_info=info)
    return subject


@pytest.fixture
def subject(tmp_path):
    return _with_shape(_subject(tmp_path), "sphere", 5.0)


def _hemi(subject):
    from fastfuncstuff.io.freesurfer import load_subject

    return load_subject(subject, ("lh",))["lh"]


def _sampler(tmp_path, mode="linear"):
    from fastfuncstuff.surface.sampling import VolumeSampler

    img = nib.load(str(_shell_anat(tmp_path)))
    return VolumeSampler(img.get_fdata(), img.affine, mode)


@pytest.mark.parametrize("shape", ["sphere", "inflated"])
def test_depth_walks_from_white_matter_through_grey_into_csf(tmp_path, shape):
    from fastfuncstuff.viewer.chedi import PatchSampler, build_patch

    subject = _with_shape(_subject(tmp_path), shape, 5.0 if shape == "sphere" else 1.5)
    h = _hemi(subject)
    patch = build_patch(h, 0, half_mm=6.0, size=48)
    assert patch.source == shape
    assert patch.inside.mean() > 0.95
    sampler = PatchSampler(patch)
    volume = _sampler(tmp_path)
    got = [float(np.nanmedian(sampler.sample(h, d, volume, 1))) for d in (0.0, 0.5, 1.5)]
    assert got == pytest.approx([WM, GM, CSF])


def test_flat_millimetres_are_roughly_cortical_millimetres(tmp_path, subject):
    """Not area-true, but a pixel 5 mm from the centre should be ~5 mm away on the cortex."""
    from fastfuncstuff.viewer.chedi import PatchSampler, build_patch

    h = _hemi(subject)
    patch = build_patch(h, 0, half_mm=10.0, size=64)
    pts = PatchSampler(patch).points(h, 0.5)
    mid = 0.5 * (h.states["white"][0] + h.states["pial"][0])
    flat = patch.to_pixels(np.array([5.0, 0.0]))
    k = int(np.count_nonzero(patch.inside.ravel()[: int(flat[0]) * 64 + int(flat[1])]))
    assert np.linalg.norm(pts[k] - mid) == pytest.approx(5.0, rel=0.25)
    # The centre vertex sits in the middle of the image.
    np.testing.assert_allclose(patch.to_pixels(patch.uv[patch.ids == 0][0]), (32, 32), atol=1.0)


def test_every_sampling_mode_reads_a_voxel_centre_as_that_voxel():
    from fastfuncstuff.surface.sampling import VolumeSampler

    data = np.random.default_rng(0).random((12, 12, 12)).astype(np.float32)
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    s = VolumeSampler(data, aff)
    centre = np.array([[6.0, 8.0, 10.0]])  # voxel (3, 4, 5)
    between = np.array([[7.0, 9.0, 11.0]])
    for mode in ("nearest", "linear", "cubic"):
        assert s.with_mode(mode)(centre)[0] == pytest.approx(data[3, 4, 5])
    # Between voxels, nearest returns a voxel and the others interpolate.
    assert s.with_mode("nearest")(between)[0] in data
    assert s.with_mode("linear")(between)[0] != s.with_mode("cubic")(between)[0]
    assert s.with_mode("cubic").data is s.data  # a mode shares the voxels
    with pytest.raises(ValueError, match="sampling"):
        s.with_mode("sinc")


def test_window_follows_the_crosshair_samples_the_overlay_and_steps_depth(tmp_path, subject):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.chediwindow import ChediWindow
    from fastfuncstuff.viewer.vocab import Load, LoadSurfaces, OpenView, SelectLayer, SetXYZ

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=CPU)
    anat = _shell_anat(tmp_path)
    session.load(str(anat))
    session.do(LoadSurfaces(str(subject), "lh"))
    session.do(OpenView("E1", "chedi", "axial"))
    win = ChediWindow("E1", session, session.do)

    def refresh(cmd=None):
        if cmd is not None:
            session.do(cmd)
        win.refresh()

    win._dispatch = refresh
    try:
        refresh(SetXYZ(0.0, 0.0, 22.0))  # mid-ribbon at the top
        assert win.patch is not None
        top = win.patch.centre
        assert session.surfaces.hemis["lh"].states["white"][top][2] > 19.0
        # Off cortex (the middle of the white matter): the patch stays put.
        refresh(SetXYZ(0.0, 0.0, 0.0))
        assert win.patch.centre == top
        # Onto cortex at the side: it follows.
        refresh(SetXYZ(22.0, 0.0, 0.0))
        assert win.patch.centre != top

        # Depth keys go through the bus: 0.5 -> 1.0 is grey -> the pial edge.
        win._depth_by(0.5)
        assert session.state.viewports.get("E1").depth == (1.0, 1.0)
        assert "SET_SURFACE_DEPTH E1" in session.to_script()

        # A second layer on top is what it samples, by name in the caption.
        import shutil

        shutil.copy(anat, tmp_path / "t2ish.nii.gz")
        session.do(Load(str(tmp_path / "t2ish.nii.gz"), "T2ish"))
        refresh()
        assert "t2ish" in win.canvas.caption
        assert win.layer().key == "T2ish"
        session.do(SelectLayer(session.state.layers.base.key))
        assert win.layer().key == "T2ish"  # selecting the base is not choosing it

        win._cycle_sampling()
        assert session.state.viewports.get("E1").sampling == "linear"
        assert "· linear" in win.canvas.caption
        assert "SET_VIEW_SAMPLING E1 linear" in session.to_script()

        # Double-click: the crosshair goes to that cortex, at the depth shown.
        win._set_depth(0.5)
        win._locate(10.0, 32.0)
        r = np.linalg.norm(session.state.crosshair_mm)
        assert 20.5 < r < 23.5
        app.processEvents()
    finally:
        win.close()
        session.close()

