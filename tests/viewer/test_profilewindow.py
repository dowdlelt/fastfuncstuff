"""Profile window: a click goes to the vertex, the crosshair brings the column."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import torch

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
nib = pytest.importorskip("nibabel")

from tests.viewer.test_profilecolumn import _hemi, scene  # noqa: E402, F401


def test_click_locates_and_crosshair_follows(scene, tmp_path):  # noqa: F811
    from fastfuncstuff.viewer.commands import Aspect
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.profilewindow import ProfileWindow
    from fastfuncstuff.viewer.vocab import OpenView, SetProfileView, SetXYZ

    img, aff, u, faces = scene
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    nib.save(nib.Nifti1Image(img, aff), str(tmp_path / "t1.nii.gz"))
    session = ViewerSession(device=torch.device("cpu"))
    try:
        session.load(str(tmp_path / "t1.nii.gz"))
        session.surfaces.hemis = {"lh": _hemi(u, faces)}
        session.surfaces.version = {"lh": 1}
        session.do(OpenView("P1", "profile", "axial"))
        win = ProfileWindow("P1", session, session.do)
        win.resize(300, 600)
        win.apply(session.state.viewports.get("P1"))
        col = win.view.column
        assert col is not None and col.n_rows == len(u)

        got = []
        win.located.connect(lambda *mm: got.append(mm))
        win._locate(123)
        np.testing.assert_allclose(got[-1], col.row_mm(123))

        target = col.row_mm(2000)
        session.do(SetXYZ(*target))
        win.refresh(Aspect.CROSSHAIR)
        # The crosshair snaps to the display grid, so the nearest row is near.
        assert np.linalg.norm(np.subtract(col.row_mm(win.view.marked), target)) < 1.0
        lo, hi = win.view.visible_span()
        assert lo <= win.view.marked < hi

        session.do(SetProfileView("P1", "mm", "worst", 0.0))
        win.apply(session.state.viewports.get("P1"))
        assert win.view.column is not col and win.view.column.profiles.mode == "mm"
        assert "SET_PROFILE_VIEW P1 mm worst 0.0" in session.to_script()
        with pytest.raises(ValueError):
            session.do(SetProfileView("P1", "fraction", "nonsense", 0.5))
        win.close()
    finally:
        session.close()
        app.processEvents()
