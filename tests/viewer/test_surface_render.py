"""The surface view's pixels against the CPU reference -- the shader is right.

Renders through the real QRhiWidget (hidden, via grabFramebuffer), so it uses
the GPU and is marked ``gpu``. A flat sheet seen from above has its normal on
the view axis, which makes the lighting factor exactly 1: every interior pixel
must be the LUT colour of the voxel the fragment sampled at mid-depth. The
volume varies along x, y *and* z, so a transposed texture, a wrong depth or an
off-by-one LUT index all show up as mismatched pixels.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

nib = pytest.importorskip("nibabel")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


def _sheet(n: int = 41, size: float = 40.0, pial_scale: float = 1.0):
    """A square grid in z=0, as a one-hemisphere FreeSurfer-like object."""
    from fastfuncstuff.io.freesurfer import FlatPatch, Hemisphere

    xs = np.linspace(-size / 2, size / 2, n)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    white = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1).astype(np.float32)
    # Mid-depth z = 1.5 mm is inside voxel k = 5 (centre 1.0), not on a face --
    # at z = 2 the CPU and GPU nearest samplers legitimately round apart.
    pial = white * np.float32([pial_scale, pial_scale, 1.0]) + np.float32([0, 0, 3.0])
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [[a, a + n, a + 1], [a + 1, a + n, a + n + 1]]
    faces = np.asarray(faces, np.int32)
    coords = white.copy()
    patch = FlatPatch("flat", coords, np.ones(white.shape[0], bool), np.zeros(white.shape[0], bool))
    return Hemisphere(
        name="lh",
        faces=faces,
        states={"white": white, "pial": pial},
        tkr_to_scanner=np.eye(4),
        morph={"curv": np.zeros(white.shape[0], np.float32)},
        patches={"flat": patch},
    )


@pytest.mark.parametrize("pial_scale", [1.0, 1.5])
def test_rendered_pixels_match_the_cpu_colouring(tmp_path, pial_scale):
    """``pial_scale`` 1.5 makes pial's area 2.25x white's, so equivolume depth
    is not the identity and the shader's has to agree with the CPU twin."""
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        pytest.skip("QRhi needs a real platform plugin")
    from fastfuncstuff.viewer import surface3d as s3
    from fastfuncstuff.viewer.compose import cached_lut
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.surfacewindow import SurfaceWindow
    from fastfuncstuff.viewer.vocab import OpenView, SetRange, SetSurfaceShape

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    vox = 2.0
    shape = (30, 30, 10)
    aff = np.diag([vox, vox, vox, 1.0])
    aff[:3, 3] = [-29.0, -29.0, -9.0]
    ijk = np.stack(np.meshgrid(*[np.arange(k) for k in shape], indexing="ij"), -1)
    # Distinct along every axis, and coarse enough that nearest sampling is
    # not decided by float rounding except on voxel faces.
    data = (ijk[..., 0] * 7 + ijk[..., 1] * 3 + ijk[..., 2] * 11) % 64
    data = data.astype(np.float32)
    nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), aff), str(tmp_path / "base.nii.gz"))
    nib.save(nib.Nifti1Image(data, aff), str(tmp_path / "over.nii.gz"))

    session = ViewerSession(device=torch.device("cpu"))
    try:
        session.load(str(tmp_path / "base.nii.gz"))
        key = session.load(str(tmp_path / "over.nii.gz"))
        session.do(SetRange(key, 0.0, 63.0))
        session.surfaces.hemis = {"lh": _sheet(pial_scale=pial_scale)}
        session.surfaces.version = {"lh": 1}
        session.do(OpenView("S1", "surface", "axial"))
        session.do(SetSurfaceShape("S1", "flat"))
        win = SurfaceWindow("S1", session, session.do)
        win.canvas.resize(320, 320)
        win.apply(session.state.viewports.get("S1"))
        win.canvas._anim.stop()
        win.canvas.morph = 1.0
        grabbed = win.canvas.grabFramebuffer()
        img = grabbed.convertToFormat(grabbed.Format.Format_RGBA8888)
        if img.isNull():
            pytest.skip("no QRhi available to render with")
        px = np.array(img.constBits()).reshape(img.height(), img.width(), 4)[..., :3]

        inv = np.linalg.inv(aff)
        lut = cached_lut("gray", torch.device("cpu")).numpy()
        layer = session.state.layers.get(key)
        shade = s3.ShadeParams(lo=0.0, hi=63.0, opacity=1.0, has_data=True)
        assert layer.colormap == "gray"
        checked = agree = 0
        for y in range(40, 280, 9):
            for x in range(40, 280, 9):
                mm = win.canvas.pick_mm(_pt(x, y))
                if mm is None:
                    continue
                # Nearest voxel, as the shader's nearest sampler reads it.
                i, j, k = np.floor(inv[:3, :3] @ mm + inv[:3, 3] + 0.5).astype(int)
                v = data[i, j, k][None]
                rgb, alpha = s3.shade_reference(v, v, shade, np.c_[lut, np.ones(len(lut))])
                want = np.round(rgb[0] * 255)
                checked += 1
                agree += int(np.all(np.abs(px[y, x] - want) <= 2))
        assert checked > 300
        # Pixels on a voxel face can go either way between CPU and GPU.
        assert agree / checked > 0.97, f"{agree}/{checked} pixels match the CPU colouring"
    finally:
        session.close()
        app.processEvents()


def _pt(x: int, y: int):
    from PySide6.QtCore import QPointF

    return QPointF(x + 0.5, y + 0.5)


def test_label_layers_draw_their_palette_colour_unblended(tmp_path):
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        pytest.skip("QRhi needs a real platform plugin")
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.surfacewindow import SurfaceWindow
    from fastfuncstuff.viewer.vocab import OpenView, SetSurfaceShape

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shape = (30, 30, 10)
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = [-29.0, -29.0, -9.0]
    ijk = np.stack(np.meshgrid(*[np.arange(k) for k in shape], indexing="ij"), -1)
    labels = (1 + (ijk[..., 0] // 3 + 2 * (ijk[..., 1] // 3)) % 7).astype(np.int16)
    nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), aff), str(tmp_path / "base.nii.gz"))
    nib.save(nib.Nifti1Image(labels, aff), str(tmp_path / "rois.nii.gz"))
    session = ViewerSession(device=torch.device("cpu"))
    try:
        session.load(str(tmp_path / "base.nii.gz"))
        key = session.load(str(tmp_path / "rois.nii.gz"))
        assert session.state.layers.get(key).roi
        palette = session.roi_palette(key, torch.device("cpu")).numpy()
        session.surfaces.hemis = {"lh": _sheet()}
        session.surfaces.version = {"lh": 1}
        session.do(OpenView("S1", "surface", "axial"))
        session.do(SetSurfaceShape("S1", "flat"))
        win = SurfaceWindow("S1", session, session.do)
        win.canvas.resize(320, 320)
        win.apply(session.state.viewports.get("S1"))
        win.canvas._anim.stop()
        win.canvas.morph = 1.0
        grabbed = win.canvas.grabFramebuffer()
        if grabbed.isNull():
            pytest.skip("no QRhi available to render with")
        img = grabbed.convertToFormat(grabbed.Format.Format_RGBA8888)
        px = np.array(img.constBits()).reshape(img.height(), img.width(), 4)[..., :3]
        inv = np.linalg.inv(aff)
        checked = agree = 0
        for y in range(40, 280, 9):
            for x in range(40, 280, 9):
                mm = win.canvas.pick_mm(_pt(x, y))
                if mm is None:
                    continue
                i, j, k = np.floor(inv[:3, :3] @ mm + inv[:3, 3] + 0.5).astype(int)
                want = np.round(palette[labels[i, j, k]] * 255)
                checked += 1
                agree += int(np.all(np.abs(px[y, x] - want) <= 2))
        assert checked > 300
        assert agree / checked > 0.97, f"{agree}/{checked} label pixels match the palette"
    finally:
        session.close()
        app.processEvents()
