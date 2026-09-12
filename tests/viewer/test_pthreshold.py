"""Thresholding a statistic as a p-value.

Settled decision #4 of the viewer's scope is AFNI *value* parity: readouts,
thresholds and p<->stat conversions agree with AFNI even though the rendering
does not. This is the half that was missing -- `read_brick_stataux` had existed
in io/headers.py with no callers anywhere, so the viewer could not turn a t
into a p and you had to do the conversion in your head.

What is guarded here is that the bucket's own statement of its test reaches the
control, and that the control and the readout cannot disagree about it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")

from fastfuncstuff.io.afni import (  # noqa: E402
    _set_afni_brick_labels,
    _set_afni_brick_stataux,
    stat_type_to_stataux,
    stataux_to_stat_type,
)
from fastfuncstuff.viewer.session import ViewerSession  # noqa: E402
from fastfuncstuff.viewer.vocab import (  # noqa: E402
    SetOverlay,
    SetThreshold,
    SetThresholdIndex,
    SetUnderlay,
    SetVolume,
)

CPU = torch.device("cpu")


@pytest.fixture
def anat(tmp_path):
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    path = tmp_path / "anat.nii.gz"
    data = np.random.default_rng(2).random((6, 7, 5)).astype(np.float32) * 100
    nib.save(nib.Nifti1Image(data, aff), str(path))
    return path


@pytest.fixture
def bucket(tmp_path):
    """A stats bucket shaped like one ffs writes: coef, its t, and an F."""
    rng = np.random.default_rng(5)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    img = nib.Nifti1Image(rng.normal(size=(6, 7, 5, 3)).astype(np.float32), aff)
    _set_afni_brick_labels(img.header, ["Faces#0_Coef", "Faces#0_Tstat", "Full_Fstat"])
    _set_afni_brick_stataux(
        img.header,
        {
            1: stat_type_to_stataux("fitt", (120.0,)),
            2: stat_type_to_stataux("fift", (3.0, 120.0)),
        },
        3,
    )
    path = tmp_path / "stats.nii.gz"
    nib.save(img, str(path))
    return path


@pytest.fixture
def session(anat, bucket):
    sess = ViewerSession(device=CPU)
    try:
        sess.do(SetUnderlay(str(anat)))
        sess.do(SetOverlay(str(bucket)))
        yield sess
    finally:
        sess.close()


def _overlay(session):
    return session.state.layers.overlay


# ---------------------------------------------------------------------------
# the header reaches the layer
# ---------------------------------------------------------------------------


def test_the_code_mapping_inverts_itself():
    for name in ("fico", "fitt", "fift", "fizt", "fict"):
        code, _ = stat_type_to_stataux(name, (1.0, 1.0, 1.0))
        assert stataux_to_stat_type(code) == name
    assert stataux_to_stat_type(9999) is None


def test_stataux_survives_the_trip_from_header_to_layer(session):
    """It was read in io/headers.py and consumed by nothing at all."""
    assert _overlay(session).stataux == {1: (3, (120.0,)), 2: (4, (3.0, 120.0))}


def test_a_dataset_without_stat_metadata_carries_an_empty_map(anat):
    sess = ViewerSession(device=CPU)
    try:
        sess.do(SetUnderlay(str(anat)))
        assert sess.state.layers.base.stataux == {}
        assert sess.state.layers.base.stat_spec() is None
    finally:
        sess.close()


# ---------------------------------------------------------------------------
# which sub-brick the threshold is reading
# ---------------------------------------------------------------------------


def test_the_threshold_brick_is_the_displayed_one_until_told_otherwise(session):
    layer = _overlay(session)
    assert layer.threshold_brick == 0
    session.do(SetVolume(layer.key, 1))
    assert _overlay(session).threshold_brick == 1


def test_a_chosen_threshold_index_wins(session):
    """Colour by the coefficient, threshold on its t -- the whole stats case."""
    layer = _overlay(session)
    session.do(SetVolume(layer.key, 0))
    session.do(SetThresholdIndex(layer.key, 1))
    assert _overlay(session).threshold_brick == 1


def test_a_coefficient_has_no_stat_spec(session):
    """A p-value quoted for a beta is a number that means nothing."""
    assert _overlay(session).stat_spec() is None


def test_a_t_brick_reports_its_test_and_dof(session):
    layer = _overlay(session)
    session.do(SetThresholdIndex(layer.key, 1))
    assert _overlay(session).stat_spec() == ("fitt", 120.0)


def test_an_f_brick_reports_both_degrees_of_freedom(session):
    layer = _overlay(session)
    session.do(SetThresholdIndex(layer.key, 2))
    assert _overlay(session).stat_spec() == ("fift", (3.0, 120.0))


def test_a_code_we_cannot_convert_reports_nothing(session):
    """Naming a distribution is not the same as being able to invert it."""
    layer = _overlay(session)
    session.state.layers.update(layer.key, stataux={0: (2, (100.0, 1.0, 3.0))})  # fico
    assert _overlay(session).stat_spec() is None


# ---------------------------------------------------------------------------
# the control
# ---------------------------------------------------------------------------


def test_the_p_control_appears_only_for_a_statistic(session, qtbot=None):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from fastfuncstuff.viewer.ui.colorbar import RangeBar

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    bar = RangeBar()
    try:
        # isHidden rather than isVisible: the bar has no shown parent here, and
        # a child of a hidden widget is never "visible" whatever it was told.
        bar.configure(_overlay(session))  # brick 0: a coefficient
        assert bar._stat_host.isHidden()

        session.do(SetThresholdIndex(_overlay(session).key, 1))
        bar.configure(_overlay(session))
        assert not bar._stat_host.isHidden()
        assert "Faces#0_Tstat" in bar.stat_label.text()
        assert "2-sided" in bar.stat_label.text()
    finally:
        bar.deleteLater()


def test_typing_a_p_sets_the_threshold_afni_agrees_with(session):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from fastfuncstuff.viewer.ui.colorbar import RangeBar

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session.do(SetThresholdIndex(_overlay(session).key, 1))
    bar = RangeBar()
    seen: list[float] = []
    bar.threshold_changed.connect(seen.append)
    try:
        bar.configure(_overlay(session))
        bar.p_spin.setValue(0.001)
        assert seen and seen[-1] == pytest.approx(3.3735, abs=5e-4)
    finally:
        bar.deleteLater()


def test_the_p_follows_a_threshold_set_anywhere_else(session):
    """p is a view of the threshold, not a second stored value."""
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from fastfuncstuff.stats.fdr import stat_value_to_pvalue
    from fastfuncstuff.viewer.ui.colorbar import RangeBar

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    key = _overlay(session).key
    session.do(SetThresholdIndex(key, 1))
    session.do(SetThreshold(key, 2.0))
    bar = RangeBar()
    try:
        bar.configure(_overlay(session))
        assert bar.p_spin.value() == pytest.approx(
            stat_value_to_pvalue(2.0, "fitt", 120.0), abs=1e-6
        )
    finally:
        bar.deleteLater()


def test_an_f_threshold_is_not_halved(session):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from scipy.stats import f as scipy_f

    from fastfuncstuff.viewer.ui.colorbar import RangeBar

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session.do(SetThresholdIndex(_overlay(session).key, 2))
    bar = RangeBar()
    seen: list[float] = []
    bar.threshold_changed.connect(seen.append)
    try:
        bar.configure(_overlay(session))
        assert "1-sided" in bar.stat_label.text()
        bar.p_spin.setValue(0.01)
        assert seen[-1] == pytest.approx(scipy_f.isf(0.01, dfn=3.0, dfd=120.0), rel=1e-5)
    finally:
        bar.deleteLater()
