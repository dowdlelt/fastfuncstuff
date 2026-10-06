"""LayNii export from the viewer: the surfaces on screen, edits and all."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.viewer.layerexport import export_dialog
from fastfuncstuff.viewer.vocab import Load, LoadSurfaces
from tests.viewer.test_surfaces import RADIUS, _anat, _subject, session  # noqa: F401

nib = pytest.importorskip("nibabel")


def test_exports_the_in_memory_surfaces_onto_the_underlay(session, tmp_path):  # noqa: F811
    session.do(Load(str(_anat(tmp_path))))
    session.do(LoadSurfaces(str(_subject(tmp_path))))
    hemi = session.surfaces.hemis["lh"]
    # An unsaved edit: pial pulled in from 1.2 R to 1.1 R. The files on disk
    # still say 4 mm thick; the export must see 2 mm.
    hemi.states["pial"][:] = hemi.states["white"] * 1.1

    spec = export_dialog(session)
    assert spec.blocked == ""
    params = dict(spec.params, prefix=str(tmp_path / "out" / "x"), dxyz=1.0, show=True)
    result = spec.run(params, None)
    assert isinstance(result, Load)
    session.do(result)
    assert any(ly.path.endswith("x_layers_equivol.nii.gz") for ly in session.state.layers)

    rim = nib.load(tmp_path / "out" / "x_rim.nii.gz")
    thick = np.asarray(nib.load(tmp_path / "out" / "x_thickness.nii.gz").dataobj)
    gm = np.asarray(rim.dataobj) == 3
    assert gm.sum() > 100
    assert abs(np.median(thick[gm]) - 0.1 * RADIUS) < 0.1
    np.testing.assert_allclose(np.abs(np.diag(rim.affine)[:3]), 1.0)
    assert "wrote 8 files" in spec.done()


def test_blocked_without_surfaces(session):  # noqa: F811
    assert export_dialog(session).blocked
