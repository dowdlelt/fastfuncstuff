"""Composing session state into pane images.

Runs entirely without Qt, which is the point: the pixels a screenshot or a
montage saves come from the same path the screen uses, so they cannot drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.compose import (
    plane_position,
    render_all,
    render_plane,
)
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.slicing import plane_shape
from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.vocab import (
    SetAlpha,
    SetBoxed,
    SetColormap,
    SetIJK,
    SetIndex,
    SetLayerVisible,
    SetThreshold,
    SetTimeLinked,
)

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")


def _write(tmp_path, name, data, tr=0.0):
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    aff[:3, 3] = [-30.0, -36.0, -28.0]
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), aff)
    if tr:
        img.header["pixdim"][4] = tr
        img.header.set_xyzt_units("mm", "sec")
    p = tmp_path / name
    nib.save(img, str(p))
    return p


@pytest.fixture
def anat(tmp_path):
    rng = np.random.default_rng(3)
    return _write(tmp_path, "anat.nii.gz", rng.random((12, 14, 10)) * 100)


@pytest.fixture
def stats(tmp_path):
    """A signed blob with a graded skirt.

    The skirt matters: alpha fades what sits *below* threshold, so a map whose
    background is exactly zero cannot demonstrate the difference.
    """
    zz, yy, xx = np.meshgrid(np.arange(10), np.arange(14), np.arange(12), indexing="ij")
    xx, yy, zz = xx.transpose(2, 1, 0), yy.transpose(2, 1, 0), zz.transpose(2, 1, 0)
    d1 = np.sqrt((xx - 4) ** 2 + (yy - 5) ** 2 + (zz - 4) ** 2)
    d2 = np.sqrt((xx - 9) ** 2 + (yy - 10) ** 2 + (zz - 7) ** 2)
    data = 8.0 * np.exp(-(d1**2) / 8.0) - 7.0 * np.exp(-(d2**2) / 8.0)
    return _write(tmp_path, "stats.nii.gz", data)


@pytest.fixture
def session():
    s = ViewerSession(device=CPU)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------------


def test_empty_session_renders_nothing_rather_than_black():
    """Callers must be able to tell "no data" from "data that is dark"."""
    s = ViewerSession(device=CPU)
    try:
        assert render_plane(s, Plane.AXIAL) is None
    finally:
        s.close()


def test_render_produces_rgba_of_the_planes_shape(session, anat):
    session.load(anat)
    for plane in Plane:
        img = render_plane(session, plane)
        assert img is not None
        assert img.rgba.dtype is torch.uint8
        assert img.rgba.shape == (*plane_shape(session.state.grid, plane), 4)


def test_render_all_covers_every_plane(session, anat):
    session.load(anat)
    assert set(render_all(session)) == set(Plane)


def test_plane_position_follows_the_crosshair(session, anat):
    session.load(anat)
    session.do(SetIJK(2, 5, 7))
    assert plane_position(session.state, Plane.SAGITTAL) == 2
    assert plane_position(session.state, Plane.CORONAL) == 5
    assert plane_position(session.state, Plane.AXIAL) == 7


def test_hidden_layers_are_not_drawn(session, anat):
    key = session.load(anat)
    before = render_plane(session, Plane.AXIAL)
    session.do(SetLayerVisible(key, False))
    assert render_plane(session, Plane.AXIAL) is None
    assert before is not None


# ---------------------------------------------------------------------------
# overlay behaviour
# ---------------------------------------------------------------------------


def test_an_overlay_changes_pixels_only_where_it_passes_threshold(session, anat, stats):
    """The core stats-review guarantee: thresholding must not tint everything."""
    session.load(anat)
    key = session.load(stats)
    session.do(SetColormap(key, "redblue"))
    session.do(SetThreshold(key, 4.0))
    session.do(SetIJK(4, 5, 4))

    with_overlay = render_plane(session, Plane.AXIAL)
    session.do(SetLayerVisible(key, False))
    without = render_plane(session, Plane.AXIAL)
    assert with_overlay is not None and without is not None

    diff = (with_overlay.rgba.int() - without.rgba.int()).abs().sum(-1)
    assert int((diff > 0).sum()) > 0, "overlay drew nothing"
    assert int((diff == 0).sum()) > 0, "overlay tinted the whole slice"


def test_raising_the_threshold_shrinks_the_drawn_region(session, anat, stats):
    session.load(anat)
    key = session.load(stats)
    session.do(SetIJK(4, 5, 4))

    def drawn(thr: float) -> int:
        session.do(SetThreshold(key, thr))
        on = render_plane(session, Plane.AXIAL)
        session.do(SetLayerVisible(key, False))
        off = render_plane(session, Plane.AXIAL)
        session.do(SetLayerVisible(key, True))
        diff = (on.rgba.int() - off.rgba.int()).abs().sum(-1)
        return int((diff > 0).sum())

    assert drawn(9.0) < drawn(2.0)


def test_alpha_mode_draws_more_than_a_hard_threshold(session, anat, stats):
    """Fading sub-threshold voxels must actually make them visible."""
    session.load(anat)
    key = session.load(stats)
    session.do(SetThreshold(key, 6.0))
    session.do(SetIJK(4, 5, 4))

    def drawn() -> int:
        on = render_plane(session, Plane.AXIAL)
        session.do(SetLayerVisible(key, False))
        off = render_plane(session, Plane.AXIAL)
        session.do(SetLayerVisible(key, True))
        return int(((on.rgba.int() - off.rgba.int()).abs().sum(-1) > 0).sum())

    session.do(SetAlpha(key, "off"))
    hard = drawn()
    session.do(SetAlpha(key, "linear"))
    assert drawn() > hard


def test_boxed_mode_adds_an_outline(session, anat, stats):
    session.load(anat)
    key = session.load(stats)
    session.do(SetThreshold(key, 4.0))
    session.do(SetIJK(4, 5, 4))
    plain = render_plane(session, Plane.AXIAL)
    session.do(SetBoxed(key, True))
    boxed = render_plane(session, Plane.AXIAL)
    assert not torch.equal(plain.rgba, boxed.rgba)


def test_layer_order_decides_what_wins(session, anat, stats):
    """The bottom layer is the underlay; moving one must change the picture."""
    session.load(anat)
    key = session.load(stats)
    session.do(SetThreshold(key, 4.0))
    session.do(SetIJK(4, 5, 4))
    top = render_plane(session, Plane.AXIAL)
    session.state.layers.move(key, 0)
    session.invalidate()
    bottom = render_plane(session, Plane.AXIAL)
    assert not torch.equal(top.rgba, bottom.rgba)


# ---------------------------------------------------------------------------
# time linkage
# ---------------------------------------------------------------------------


def test_a_dataset_with_a_tr_is_time_linked(session, tmp_path):
    rng = np.random.default_rng(1)
    path = _write(tmp_path, "bold.nii.gz", rng.random((8, 8, 6, 10)), tr=1.5)
    key = session.load(path)
    assert session.state.layers.get(key).time_linked


def test_a_3d_dataset_is_never_time_linked(session, anat):
    assert not session.state.layers.get(session.load(anat)).time_linked


def test_time_linkage_can_be_overridden(session, tmp_path):
    """4-D NIfTI cannot say whether its sub-bricks are time or contrasts."""
    data = np.zeros((8, 8, 6, 4), dtype=np.float32)
    key = session.load(_write(tmp_path, "multi.nii.gz", data, tr=1.0))
    assert session.state.layers.get(key).time_linked  # the 4-D default
    session.do(SetTimeLinked(key, False))
    assert not session.state.layers.get(key).time_linked


def test_labelled_subbricks_default_to_not_time_linked():
    """3dDeconvolve-style labels are the one usable signal that it is stats."""
    from fastfuncstuff.io.dsetinfo import DatasetInfo
    from fastfuncstuff.viewer.session import infer_time_linked

    info = DatasetInfo(path=Path("s.nii"), iname="s", exists=True, shape=(4, 4, 4, 6))
    assert infer_time_linked(info) is True
    info.labels = ["Full_R2", "task#0_Coef", "task#0_Tstat"]
    assert infer_time_linked(info) is False


def test_scrubbing_time_changes_what_a_time_linked_layer_shows(session, tmp_path):
    data = np.zeros((8, 8, 6, 6), dtype=np.float32)
    for v in range(6):
        data[..., v] = float(v + 1) * 20.0
    path = _write(tmp_path, "bold.nii.gz", data, tr=2.0)
    session.load(path)
    session.store.ensure_ram(session.state.layers.keys[0])
    session.invalidate()

    first = render_plane(session, Plane.AXIAL)
    session.do(SetIndex(5))
    later = render_plane(session, Plane.AXIAL)
    assert not torch.equal(first.rgba, later.rgba)


def test_scrubbing_does_not_disturb_a_non_time_linked_layer(session, anat, tmp_path):
    key = session.load(anat)
    rng = np.random.default_rng(2)
    session.load(_write(tmp_path, "bold.nii.gz", rng.random((12, 14, 10, 8)), tr=1.0))
    session.do(SetLayerVisible(session.state.layers.keys[1], False))

    before = render_plane(session, Plane.AXIAL)
    session.do(SetIndex(6))
    after = render_plane(session, Plane.AXIAL)
    assert torch.equal(before.rgba, after.rgba), "anatomy must not follow the time index"
    assert session.state.layers.get(key).time_linked is False


# ---------------------------------------------------------------------------
# residency signalling
# ---------------------------------------------------------------------------


def test_a_3d_layer_never_reads_as_pending(session, anat):
    """It is complete once previewed, so it must not sit at 'loading' forever."""
    key = session.load(anat)
    assert not session.resident(key).pending


def test_a_4d_layer_is_not_pending_once_loaded(session, tmp_path):
    rng = np.random.default_rng(4)
    path = _write(tmp_path, "bold.nii.gz", rng.random((8, 8, 6, 10)), tr=1.5)
    key = session.load(path)
    session.store.ensure_ram(key)
    assert not session.resident(key).pending


# ---------------------------------------------------------------------------
# display masks
# ---------------------------------------------------------------------------


def test_a_display_mask_hides_the_overlay_and_only_the_overlay(session, anat, stats):
    """Masked-out overlay voxels show the underlay, as if never drawn; the
    underlay itself is untouched."""
    session.load(anat)
    key = session.load(stats)
    session.do(SetColormap(key, "redblue"))
    session.do(SetThreshold(key, 4.0))
    session.do(SetIJK(4, 5, 4))
    shown = render_plane(session, Plane.AXIAL)

    layer = session.state.layers.get(key)
    session.set_display_mask("K1", key, np.zeros(layer.shape, dtype=bool))
    masked = render_plane(session, Plane.AXIAL)
    session.do(SetLayerVisible(key, False))
    underlay_only = render_plane(session, Plane.AXIAL)
    assert shown is not None and masked is not None and underlay_only is not None
    assert not torch.equal(shown.rgba, masked.rgba)
    assert torch.equal(masked.rgba, underlay_only.rgba)

    session.do(SetLayerVisible(key, True))
    session.set_display_mask("K1", None, None)  # the owner letting go restores it
    assert torch.equal(render_plane(session, Plane.AXIAL).rgba, shown.rgba)


def test_mask_resampling_lands_a_fine_mask_on_a_coarse_grid():
    """A 1 mm anat's mask on a 3 mm map: voxel centres, not corners."""
    from fastfuncstuff.viewer.session import resample_mask_nearest

    fine = np.zeros((30, 30, 30), dtype=bool)
    fine[:15] = True  # the left half, in 1 mm voxels
    coarse_aff = np.diag([3.0, 3.0, 3.0, 1.0])
    coarse_aff[:3, 3] = 1.0  # coarse voxel centres sit on fine voxels 1, 4, 7, ...
    out = resample_mask_nearest(fine, np.eye(4), (10, 10, 10), coarse_aff)
    assert out[:5].all() and not out[5:].any()
    # Outside the source is outside the mask, not wrapped or clamped in.
    shifted = coarse_aff.copy()
    shifted[0, 3] = -15.0
    assert not resample_mask_nearest(fine, np.eye(4), (10, 10, 10), shifted)[:2].any()
