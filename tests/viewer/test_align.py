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


# ---------------------------------------------------------------------------
# the mode
# ---------------------------------------------------------------------------


def _align(session, moving="epi"):
    from fastfuncstuff.viewer.vocab import SetInput, SetMode

    session.do(SetMode("align"))
    if moving is not None:
        session.do(SetInput(moving))
    return session.mode


def test_the_moving_image_is_never_the_underlay_or_a_follower(session):
    session.do(SetLayerFollows("stat", "epi"))
    mode = _align(session, moving=None)
    offered = [ly.key for ly in session.input_candidates()]
    assert offered == ["epi"]
    assert mode.moving().key == "epi"


def test_a_slider_moves_the_image_and_what_rides_on_it(session):
    from fastfuncstuff.viewer.vocab import SetModeParam

    session.do(SetLayerFollows("stat", "epi"))
    mode = _align(session)
    session.do(SetModeParam("tx", "5"))
    session.do(SetModeParam("rz", "10"))
    pivot = mode.pivot()
    for key in ("epi", "stat"):
        xform = align.layer_xform(session.state.layers.get(key))
        assert np.allclose(align.decompose(xform, pivot), (5, 0, 0, 0, 0, 10), atol=1e-6)


def test_the_sliders_read_back_a_drag(session):
    mode = _align(session)
    session.do(mode.move_to(mode.shifted((0.0, 3.0, 0.0))))
    session.do(mode.move_to(mode.turned((0, 0, 1), 7.0)))
    assert mode.params["ty"] == pytest.approx(3.0)
    assert mode.params["rz"] == pytest.approx(7.0)


def test_turning_leaves_the_pivot_where_it_is_drawn(session):
    mode = _align(session)
    session.do(mode.move_to(mode.shifted((4.0, -2.0, 1.0))))
    before = mode.pivot_mm()
    session.do(mode.move_to(mode.turned((0.3, 1.0, 0.2), 25.0)))
    assert np.allclose(mode.pivot_mm(), before)


def test_pivot_here_turns_about_the_crosshair(session):
    from fastfuncstuff.viewer.vocab import ModeAction, SetIJK

    mode = _align(session)
    session.do(SetIJK(2, 3, 4))
    here = np.array(session.state.crosshair_mm)
    session.do(ModeAction("pivot"))
    session.do(mode.move_to(mode.turned((0, 0, 1), 30.0)))
    assert np.allclose(mode.pivot_mm(), here)


def test_the_automatic_moving_image_stays_put_while_another_is_selected(session):
    """Selecting the stat map to attach it must not make it the moving image."""
    from fastfuncstuff.viewer.vocab import ModeAction, SelectLayer, SetLayerVisible

    session.do(SetLayerVisible("stat", False))
    mode = _align(session, moving=None)
    first = mode.moving().key
    other = "epi" if first == "stat" else "stat"
    session.do(SelectLayer(other))
    session.do(ModeAction("attach"))
    assert mode.moving().key == first
    assert session.state.layers.get(other).follows == first


def test_a_slider_session_replays(session):
    from fastfuncstuff.viewer.vocab import SetModeParam

    _align(session)
    session.do(SetModeParam("tx", "4"))
    session.do(SetModeParam("ry", "-6"))
    again = ViewerSession(device=CPU)
    try:
        again.run_script(session.to_script())
        assert np.allclose(
            again.state.layers.get("epi").affine, session.state.layers.get("epi").affine
        )
    finally:
        again.close()


def test_save_then_load_gives_back_the_same_placement(session, tmp_path):
    from fastfuncstuff.viewer.vocab import ModeAction

    mode = _align(session)
    placed = align.compose((3, -4, 5, 6, -7, 8), mode.pivot())
    session.do(mode.move_to(placed))
    path = tmp_path / "hand.aff12.1D"
    save = mode.dialog_for("save")
    save.install(save.run({"path": str(path)}, None))
    session.do(ModeAction("reset"))
    assert not session.state.layers.get("epi").is_moved
    load = mode.dialog_for("load")
    load.install(load.run({"path": str(path)}, None))
    assert np.allclose(align.layer_xform(session.state.layers.get("epi")), placed, atol=1e-6)
    # And the load is in the recording, not only on screen.
    assert "SET_LAYER_XFORM epi" in session.to_script()


def test_the_allineate_button_refines_from_the_hand_placement(tmp_path):
    """A 2 mm / 3 degree hand placement goes to the truth."""
    rng = np.random.default_rng(4)
    vol = np.zeros((32, 32, 32), np.float32)
    vol[8:24, 10:22, 12:26] = 80.0
    vol[12:18, 14:18, 16:20] = 160.0
    vol += rng.normal(0, 1, vol.shape).astype(np.float32)
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = -32.0
    nib.save(nib.Nifti1Image(vol, aff), str(tmp_path / "fixed.nii"))
    nib.save(
        nib.Nifti1Image(np.pad(vol, 1), aff @ align.translation((-1, -1, -1))),
        str(tmp_path / "moving.nii"),
    )

    s = ViewerSession(device=CPU)
    try:
        s.load(tmp_path / "fixed.nii", key="fixed")
        s.load(tmp_path / "moving.nii", key="moving")
        mode = _align(s, moving=None)
        off = align.compose((2.0, -1.5, 1.0, 3.0, 0.0, -2.0), mode.pivot())
        s.do(mode.move_to(off))
        spec = mode.dialog_for("allineate")
        assert not spec.blocked
        spec.install(spec.run({"cost": "ls", "dof": "rigid", "small": True}, None))
        xform = align.layer_xform(s.state.layers.get("moving"))
        corners = np.array([[x, y, z] for x in (-30, 30) for y in (-30, 30) for z in (-30, 30)])
        moved = (xform[:3, :3] @ corners.T).T + xform[:3, 3]
        assert np.abs(moved - corners).max() < 1.0
    finally:
        s.close()


def test_cmass_puts_the_moving_centroid_on_the_fixed_one_and_keeps_the_turn(tmp_path):
    """An off-centre object, so the centroid is not just the box centre."""
    vol = np.zeros((30, 30, 30), np.float32)
    vol[4:14, 6:12, 18:26] = 50.0
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = -30.0
    moved = aff.copy()
    moved[:3, 3] += (18.0, -7.0, 5.0)  # the header is off
    nib.save(nib.Nifti1Image(vol, aff), str(tmp_path / "fixed.nii"))
    nib.save(nib.Nifti1Image(vol, moved), str(tmp_path / "moving.nii"))
    s = ViewerSession(device=CPU)
    try:
        s.load(tmp_path / "fixed.nii", key="fixed")
        s.load(tmp_path / "moving.nii", key="moving")
        mode = _align(s, moving="moving")
        from fastfuncstuff.viewer.vocab import ModeAction, SetModeParam

        s.do(SetModeParam("rz", "15"))
        turn = align.layer_xform(s.state.layers.get("moving"))[:3, :3].copy()
        s.do(ModeAction("cmass"))
        xform = align.layer_xform(s.state.layers.get("moving"))
        target = align.centre_of_mass_mm(vol, aff)
        drawn = xform @ np.append(align.centre_of_mass_mm(vol, moved), 1.0)
        assert np.allclose(drawn[:3], target, atol=1e-6)
        assert np.allclose(xform[:3, :3], turn)
        assert mode.params["rz"] == pytest.approx(15.0)
    finally:
        s.close()
