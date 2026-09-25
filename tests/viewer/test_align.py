"""Moving layers by hand, and handing the result to the command-line tools.

The bug this file exists to catch is a matrix that looks right in the viewer
and is wrong in ffs_allineate: the viewer draws through the real (possibly
oblique) affines while .aff12.1D is defined on cardinal ones, so a conversion
that skips the voxel step agrees on straight data and silently rotates oblique
data by the obliquity.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer import align
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.vocab import RemoveLayer, SetLayerFollows, SetLayerXform

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")


def _oblique(vox, degrees, origin):
    aff = np.eye(4)
    aff[:3, :3] = align.axis_rotation((1, 0.3, 0), degrees) @ np.diag(vox)
    aff[:3, 3] = origin
    return aff


BASE = _oblique((1.0, 1.0, 1.2), 0.0, (-90.0, -110.0, -70.0))
SOURCE = _oblique((2.5, 2.5, 3.0), 14.0, (-80.0, -95.0, -40.0))


def test_euler_round_trip():
    for angles in [(10, -20, 30), (0, 0, 0), (-170, 45, 5), (3, 89, -60)]:
        rot = align.euler_matrix(*angles)
        assert np.allclose(align.euler_matrix(*align.euler_angles(rot)), rot, atol=1e-9)


def test_decompose_reads_translation_as_how_far_the_pivot_moved():
    pivot = np.array([10.0, -5.0, 20.0])
    xform = align.compose((4, 0, -2, 0, 0, 25), pivot)
    moved = (xform @ np.append(pivot, 1.0))[:3]
    assert np.allclose(moved - pivot, (4, 0, -2))
    assert np.allclose(align.decompose(xform, pivot), (4, 0, -2, 0, 0, 25))


def test_compose_keeps_a_fitted_stretch():
    """Turning a 12-parameter result must not flatten its scale and shear."""
    pivot = np.zeros(3)
    fitted = np.eye(4)
    fitted[:3, :3] = align.euler_matrix(5, 0, 0) @ np.array(
        [[1.1, 0.05, 0], [0.05, 0.95, 0], [0, 0, 1.02]]
    )
    rot, stretch = align.polar(fitted[:3, :3])
    params = align.decompose(fitted, pivot)
    assert np.allclose(align.compose(params, pivot, stretch), fitted, atol=1e-9)


def test_aff12_round_trips_on_oblique_headers():
    xform = align.compose((6, -3, 9, 4, -7, 12), (0, 0, 0))
    dicom = align.to_aff12(xform, BASE, SOURCE)
    assert np.allclose(align.from_aff12(dicom, BASE, SOURCE), xform, atol=1e-9)


def test_a_saved_matrix_gives_allineate_the_voxel_map_the_viewer_draws(tmp_path):
    """What -1Dmatrix_apply reads back is base voxel -> the source voxel on screen."""
    from fastfuncstuff.processing.affine import load_matrix_1D

    xform = align.compose((6, -3, 9, 4, -7, 12), (0, 0, 0))
    path = tmp_path / "m.aff12.1D"
    align.save_aff12(path, xform, BASE, SOURCE)
    applied = load_matrix_1D(path, BASE, SOURCE).double().numpy()
    drawn = align.voxel_matrix(xform, BASE, SOURCE)
    assert np.allclose(applied, drawn, atol=1e-4)


# ---------------------------------------------------------------------------
# on the stack
# ---------------------------------------------------------------------------


@pytest.fixture
def session(tmp_path):
    s = ViewerSession(device=CPU)
    for name, aff in (("anat", BASE), ("epi", SOURCE), ("stat", SOURCE)):
        data = np.random.default_rng(0).random((12, 10, 8)).astype(np.float32)
        nib.save(nib.Nifti1Image(data, aff), str(tmp_path / f"{name}.nii.gz"))
        s.load(tmp_path / f"{name}.nii.gz", key=name)
    yield s
    s.close()


def test_a_follower_rides_with_its_parent(session):
    session.do(SetLayerFollows("stat", "epi"))
    xform = align.compose((5, 0, 0, 0, 10, 0), align.centre_mm(session.state.layers.get("epi")))
    session.do(SetLayerXform.of("epi", xform))
    stat = session.state.layers.get("stat")
    assert np.allclose(align.layer_xform(stat), xform)
    assert np.allclose(align.native_affine(stat), SOURCE)


def test_attaching_adopts_where_the_parent_already_is(session):
    xform = align.compose((0, 7, 0, 0, 0, 0), (0, 0, 0))
    session.do(SetLayerXform.of("epi", xform))
    session.do(SetLayerFollows("stat", "epi"))
    assert np.allclose(align.layer_xform(session.state.layers.get("stat")), xform)


def test_the_identity_puts_a_layer_back_exactly(session):
    loaded = session.state.layers.get("epi").affine.copy()
    session.do(SetLayerXform.of("epi", align.compose((1, 2, 3, 4, 5, 6), (0, 0, 0))))
    session.do(SetLayerXform.of("epi", np.eye(4)))
    epi = session.state.layers.get("epi")
    assert not epi.is_moved
    assert np.array_equal(epi.affine, loaded)


def test_removing_a_parent_leaves_its_follower_where_it_is(session):
    xform = align.compose((0, 0, 4, 0, 0, 0), (0, 0, 0))
    session.do(SetLayerFollows("stat", "epi"))
    session.do(SetLayerXform.of("epi", xform))
    session.do(RemoveLayer("epi"))
    stat = session.state.layers.get("stat")
    assert stat.follows is None
    assert np.allclose(align.layer_xform(stat), xform)


def test_a_moved_session_replays(session, tmp_path):
    session.do(SetLayerFollows("stat", "epi"))
    session.do(SetLayerXform.of("epi", align.compose((1, -2, 3, 4, -5, 6), (0, 0, 0))))
    script = session.to_script()
    again = ViewerSession(device=CPU)
    try:
        again.run_script(script)
        for key in ("epi", "stat"):
            assert np.allclose(
                again.state.layers.get(key).affine, session.state.layers.get(key).affine
            )
    finally:
        again.close()


def test_a_drag_records_one_line_per_layer(session):
    """Every mouse move sends a transform; only the last one is worth replaying."""
    for step in range(5):
        session.do(SetLayerXform.of("epi", align.compose((step, 0, 0, 0, 0, 0), (0, 0, 0))))
    assert session.to_script().count("SET_LAYER_XFORM") == 1


def test_chains_are_refused(session):
    session.do(SetLayerFollows("stat", "epi"))
    with pytest.raises(ValueError):
        session.do(SetLayerFollows("anat", "stat"))
