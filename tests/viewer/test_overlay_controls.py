"""Colouring by one sub-brick and thresholding on another.

The stats case is a beta coloured and cut on its t, and every bug here came
from treating the two as one scale: a slider sized to the beta that could never
reach the t, an auto-range that read sub-brick 0 whatever was shown, and a
colour map chosen once from the F and kept for every signed coefficient after.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

CPU = torch.device("cpu")
LABELS = ["Full_Fstat", "A#0_Coef", "A#0_Tstat", "B#0_Coef", "B#0_Tstat"]


@pytest.fixture
def bucket_dir(tmp_path):
    """An anatomy and a bucket: F >= 0, betas near +-0.5, t near +-20."""
    from fastfuncstuff.io.afni import save_nifti

    rng = np.random.default_rng(4)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    shape = (8, 9, 6)
    stats = np.zeros((*shape, 5), np.float32)
    inside = (slice(1, 7), slice(1, 8), slice(1, 5))  # zero outside, as a masked bucket is
    stats[(*inside, 0)] = np.abs(rng.normal(size=(6, 7, 4))) * 10
    for coef in (1, 3):
        beta = rng.normal(size=(6, 7, 4)) * 0.5
        stats[(*inside, coef)] = beta
        stats[(*inside, coef + 1)] = beta * 40
    save_nifti(stats, tmp_path / "stats.nii.gz", affine=aff, brick_labels=LABELS)
    save_nifti(rng.random(shape).astype(np.float32) * 100, tmp_path / "anat.nii.gz", affine=aff)
    return tmp_path


@pytest.fixture
def session(bucket_dir):
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.vocab import SetOverlay, SetUnderlay

    s = ViewerSession(device=CPU)
    s.do(SetUnderlay(str(bucket_dir / "anat.nii.gz")))
    s.do(SetOverlay(str(bucket_dir / "stats.nii.gz")))
    yield s
    s.close()


def _stats(session):
    return session.state.layers.overlay


def _brick(session, index):
    from fastfuncstuff.io.dsetinfo import read_volume

    return read_volume(_stats(session).path, index)[0]


# ---------------------------------------------------------------------------
# the colour scale follows the sub-brick shown
# ---------------------------------------------------------------------------


def test_an_f_starts_hot_from_zero_with_alpha_off(session):
    layer = _stats(session)
    assert layer.colormap == "hot"
    assert layer.range_lo == 0.0 and layer.range_hi > 0.0
    assert layer.alpha_mode.value == "off"


def test_a_signed_sub_brick_turns_red_blue_and_symmetric(session):
    from fastfuncstuff.viewer.vocab import SetVolume

    session.do(SetVolume(_stats(session).key, 1))
    layer = _stats(session)
    assert layer.colormap == "redblue"
    assert layer.range_lo == pytest.approx(-layer.range_hi)
    # Sized to the beta, not left at the F's scale.
    assert layer.range_hi < 2.0

    session.do(SetVolume(layer.key, 0))
    assert _stats(session).colormap == "hot"


def test_a_colormap_someone_chose_survives_a_sub_brick_change(session):
    from fastfuncstuff.viewer.vocab import SetColormap, SetVolume

    key = _stats(session).key
    session.do(SetColormap(key, "viridis"))
    session.do(SetVolume(key, 1))
    assert _stats(session).colormap == "viridis"


def test_the_range_ignores_the_zeros_outside_the_mask(session):
    """Counting them put the 98th percentile of a masked beta near nothing."""
    from fastfuncstuff.viewer.vocab import SetVolume

    session.do(SetVolume(_stats(session).key, 2))
    beta_t = np.abs(_brick(session, 2))
    assert _stats(session).range_hi > np.percentile(beta_t[beta_t > 0], 50)


# ---------------------------------------------------------------------------
# which sub-brick the threshold reads
# ---------------------------------------------------------------------------


def test_plus_one_follows_each_coef_to_its_t(session):
    from fastfuncstuff.viewer.vocab import SetThresholdFollow, SetVolume

    key = _stats(session).key
    session.do(SetVolume(key, 1))
    session.do(SetThresholdFollow(key, "next"))
    assert _stats(session).threshold_brick == 2
    session.do(SetVolume(key, 3))
    assert _stats(session).threshold_brick == 4


def test_same_follows_the_overlay_itself(session):
    from fastfuncstuff.viewer.vocab import SetThresholdFollow, SetVolume

    key = _stats(session).key
    session.do(SetThresholdFollow(key, "next"))
    session.do(SetThresholdFollow(key, "same"))
    session.do(SetVolume(key, 4))
    assert _stats(session).threshold_brick == 4


def test_naming_a_threshold_sub_brick_stops_it_following(session):
    from fastfuncstuff.viewer.vocab import SetThresholdFollow, SetThresholdIndex, SetVolume

    key = _stats(session).key
    session.do(SetThresholdFollow(key, "next"))
    session.do(SetThresholdIndex(key, 2))
    session.do(SetVolume(key, 3))
    layer = _stats(session)
    assert (layer.threshold_follow, layer.threshold_brick) == ("fixed", 2)


def test_following_replays_from_a_script(session):
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.vocab import SetThresholdFollow, SetVolume

    key = _stats(session).key
    session.do(SetThresholdFollow(key, "next"))
    session.do(SetVolume(key, 3))
    replay = ViewerSession(device=CPU)
    try:
        replay.run_script(session.to_script())
        assert replay.state.layers.overlay.threshold_brick == 4
    finally:
        replay.close()


def test_an_unknown_follow_mode_is_refused(session):
    from fastfuncstuff.viewer.vocab import SetThresholdFollow

    with pytest.raises(ValueError):
        session.do(SetThresholdFollow(_stats(session).key, "previous"))


# ---------------------------------------------------------------------------
# mirror
# ---------------------------------------------------------------------------


def test_mirror_holds_min_at_minus_max(session):
    from fastfuncstuff.viewer.vocab import SetRange, SetRangeMirror, SetVolume

    key = _stats(session).key
    session.do(SetRangeMirror(key, True))
    layer = _stats(session)
    assert layer.range_lo == -layer.range_hi
    session.do(SetRange(key, 0.0, 5.0))
    assert (_stats(session).range_lo, _stats(session).range_hi) == (-5.0, 5.0)
    # An F is one-signed, and would otherwise re-derive to start at zero.
    session.do(SetVolume(key, 1))
    session.do(SetVolume(key, 0))
    layer = _stats(session)
    assert layer.range_lo == -layer.range_hi


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(qapp, bucket_dir):
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.window import ViewerWindow
    from fastfuncstuff.viewer.vocab import SetOverlay, SetUnderlay

    session = ViewerSession(device=CPU)
    w = ViewerWindow(session)
    w.read_directory(bucket_dir)
    w.refresh(session.do(SetUnderlay(str(bucket_dir / "anat.nii.gz"))))
    w.refresh(session.do(SetOverlay(str(bucket_dir / "stats.nii.gz"))))
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def _pick_brick(win, qapp, index):
    win.brick_box.setCurrentIndex(index)
    win.brick_box.activated.emit(index)
    qapp.processEvents()


def test_auto_reads_the_sub_brick_on_screen(win, qapp):
    """It read the time index, which on a bucket is always the F."""
    _pick_brick(win, qapp, 1)
    win.max_spin.setValue(99.0)
    qapp.processEvents()
    win.autorange_button.click()
    qapp.processEvents()
    layer = win.session.state.layers.overlay
    assert layer.range_hi < 2.0 and layer.range_lo == pytest.approx(-layer.range_hi)


def test_the_slider_reaches_the_t_when_cutting_on_it(win, qapp):
    """A slider sized to a beta of 0.5 could never reach a t of 20."""
    _pick_brick(win, qapp, 1)
    win.thr_next_check.click()
    qapp.processEvents()
    win.thr_slider.setValue(1000)
    qapp.processEvents()
    t = np.abs(_brick(win.session, 2))
    assert win.session.state.layers.overlay.threshold == pytest.approx(float(t.max()), rel=1e-3)


def test_the_bar_does_not_mark_a_threshold_in_other_units(win, qapp):
    _pick_brick(win, qapp, 1)
    win.thr_next_check.click()
    qapp.processEvents()
    before = win.session.state.layers.overlay.threshold
    assert win.colorbar._threshold == 0.0
    win.colorbar.clicked.emit(0.3)
    qapp.processEvents()
    assert win.session.state.layers.overlay.threshold == before


def test_follow_boxes_are_exclusive_and_both_can_be_off(win, qapp):
    assert win.thr_same_check.isChecked() and not win.thr_next_check.isChecked()
    win.thr_next_check.click()
    qapp.processEvents()
    assert win.thr_next_check.isChecked() and not win.thr_same_check.isChecked()
    win.thr_next_check.click()
    qapp.processEvents()
    assert not win.thr_next_check.isChecked() and not win.thr_same_check.isChecked()
    assert win.session.state.layers.overlay.threshold_follow == "fixed"


def test_the_threshold_menu_shows_what_a_follow_rule_chose(win, qapp):
    win.thr_next_check.click()
    _pick_brick(win, qapp, 3)
    assert win.thrbrick_box.currentData() == 4


def test_alpha_boxes_are_exclusive_and_both_can_be_off(win, qapp):
    layer = lambda: win.session.state.layers.overlay  # noqa: E731
    assert not win.alpha_linear_check.isChecked() and not win.alpha_quad_check.isChecked()
    win.alpha_linear_check.click()
    qapp.processEvents()
    assert layer().alpha_mode.value == "linear"
    win.alpha_quad_check.click()
    qapp.processEvents()
    assert layer().alpha_mode.value == "quadratic"
    assert not win.alpha_linear_check.isChecked()
    win.alpha_quad_check.click()
    qapp.processEvents()
    assert layer().alpha_mode.value == "off"


def test_mirror_fades_min_and_tracks_max(win, qapp):
    win.rangebar.mirror_check.click()
    qapp.processEvents()
    assert not win.min_spin.isEnabled()
    win.max_spin.setValue(3.5)
    qapp.processEvents()
    layer = win.session.state.layers.overlay
    assert (layer.range_lo, layer.range_hi) == (-3.5, 3.5)
    assert win.min_spin.value() == -3.5
