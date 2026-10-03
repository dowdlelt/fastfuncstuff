"""Depth window: a region's profile against depth, and its export."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import torch

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
nib = pytest.importorskip("nibabel")

from tests.viewer.test_surfaces import _anat, _shell_anat, _subject  # noqa: E402


def test_profile_of_a_disc_reads_wm_gm_csf_and_exports(tmp_path):
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.depthwindow import DepthWindow, export_table
    from fastfuncstuff.viewer.vocab import LoadSurfaces, OpenView, SetDepthView, SetXYZ

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=torch.device("cpu"))
    try:
        session.load(str(_anat(tmp_path)))
        # Shell phantom (WM < 21, GM < 24, CSF) under a white=20/pial=24 pair.
        session.load(str(_shell_anat(tmp_path)))
        session.do(LoadSurfaces(str(_subject(tmp_path)), "lh"))
        session.do(SetXYZ(0.0, 0.0, 22.0))
        session.do(OpenView("D1", "depth", "axial"))
        session.do(SetDepthView("D1", "disc", 4.0, "", ""))
        win = DepthWindow("D1", session, session.do)
        win.apply(session.state.viewports.get("D1"))
        r = win.result
        assert r is not None and r.n_vertices > 3 and r.timecourses is None
        mean = np.nanmean(r.profiles, axis=0)
        f = r.fractions
        assert mean[np.argmin(abs(f + 0.3))] > 100  # WM below white
        assert 60 < mean[np.argmin(abs(f - 0.6))] < 80  # GM mid-ribbon
        assert mean[-1] < 30  # CSF past pial
        assert session.surfaces.depth_roi_vertices["lh"].size == r.n_vertices
        out = tmp_path / "p.tsv"
        export_table(r, out)
        rows = out.read_text().splitlines()
        assert rows[0].startswith("row\t") and rows[1].startswith("mean\t")
        assert len(rows[1].split("\t")) == f.size + 1
        assert "SET_DEPTH_VIEW D1 disc 4.0" in session.to_script()
        win.close()
        assert session.surfaces.depth_roi_vertices == {}
    finally:
        session.close()
        app.processEvents()
