"""The ffs_dataviz panel: a slider's value can be clicked and typed."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6 import QtCore, QtTest, QtWidgets  # noqa: E402

from fastfuncstuff.viz.dataviz.app import Viz  # noqa: E402
from fastfuncstuff.viz.dataviz.features import compute_features  # noqa: E402
from fastfuncstuff.viz.dataviz.scenes import SCENE_NAMES  # noqa: E402

KEY = QtCore.Qt.Key


@pytest.fixture(scope="module")
def viz():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    data = 100.0 + np.random.default_rng(0).standard_normal((60, 10, 10, 8)).astype(np.float32)
    feats = compute_features(
        torch.from_numpy(data), np.ones((10, 10, 8), dtype=bool), np.eye(4), 2.0, [0],
        n_components=6, n_families=3, fwhm_mm=0.0, remove_global=False, verbose=False,
    )  # fmt: skip
    win = Viz(feats, "test", size=(64, 36), sound=False)
    win.timer.stop()
    win.dock.show()
    win.show()
    QtTest.QTest.qWaitForWindowExposed(win)
    yield win
    win.close()
    app.processEvents()


def type_into(viz, name: str, text: str, end=KEY.Key_Return):
    edit = viz.controls.widgets[name][1]
    edit.setFocus()
    edit.selectAll()
    QtTest.QTest.keyClicks(edit, text)
    QtTest.QTest.keyClick(edit, end)
    return edit


def test_typing_sets_the_exact_value_and_moves_the_slider(viz):
    slider = viz.controls.widgets["speed"][0]
    edit = type_into(viz, "speed", "3.37")
    assert viz.vis.p["speed"] == 3.37
    assert edit.text() == "3.37"
    assert slider.value() == round((3.37 - 0.1) / (20.0 - 0.1) * viz.controls.STEPS)


def test_a_typed_value_is_clamped_to_the_range(viz):
    edit = type_into(viz, "speed", "5000")
    assert viz.vis.p["speed"] == 20.0
    assert edit.text() == "20"


def test_an_integer_parameter_rounds(viz):
    type_into(viz, "components", "4.4")
    assert viz.vis.p["components"] == 4.0


@pytest.mark.parametrize("junk", ["abc", "", "nan", "inf"])
def test_junk_restores_the_value(viz, junk):
    type_into(viz, "speed", "2.5")
    edit = type_into(viz, "speed", junk)
    assert viz.vis.p["speed"] == 2.5
    assert edit.text() == "2.5"


def test_escape_cancels_rather_than_quitting(viz):
    type_into(viz, "speed", "2.5")
    edit = type_into(viz, "speed", "9", end=KEY.Key_Escape)
    assert viz.vis.p["speed"] == 2.5
    assert edit.text() == "2.5"
    assert viz.isVisible()


def test_typing_digits_does_not_switch_scene(viz):
    scene = viz.vis.scene_name
    type_into(viz, "speed", "3")
    assert viz.vis.scene_name == scene and scene in SCENE_NAMES


def test_leaving_an_untouched_field_applies_nothing(viz, monkeypatch):
    # The families field reruns the ICA, so merely passing through must not.
    edit = viz.controls.widgets["families"][1]
    calls = []
    monkeypatch.setattr(viz, "set_param", lambda *a, **k: calls.append(a))
    edit.setFocus()
    viz.canvas.setFocus()
    assert calls == []


def test_a_change_underneath_a_focused_field_survives_leaving_it(viz):
    # Remix's button takes no focus, so it can change a value under the cursor.
    edit = viz.controls.widgets["speed"][1]
    type_into(viz, "speed", "2.5")
    edit.setFocus()
    viz.set_param("speed", 7.0)
    viz.canvas.setFocus()
    assert viz.vis.p["speed"] == 7.0 and edit.text() == "7"


def test_a_click_selects_the_value_for_typing_over(viz):
    edit = viz.controls.widgets["speed"][1]
    type_into(viz, "speed", "2.5")
    QtTest.QTest.mouseClick(edit, QtCore.Qt.MouseButton.LeftButton)
    assert edit.hasFocus()
    QtWidgets.QApplication.processEvents()
    assert edit.selectedText() == "2.5"
    QtTest.QTest.keyClicks(edit, "4.25")
    QtTest.QTest.keyClick(edit, KEY.Key_Return)
    assert viz.vis.p["speed"] == 4.25
    assert viz.canvas.hasFocus()  # shortcuts work again straight away


def test_a_slider_drag_updates_the_field(viz):
    slider, edit = viz.controls.widgets["speed"]
    slider.setValue(viz.controls.STEPS)
    assert viz.vis.p["speed"] == 20.0 and edit.text() == "20"


def test_sound_values_are_typed_too(viz):
    box = next(b for b in viz.controls.findChildren(QtWidgets.QGroupBox) if b.title() == "sound")
    edit = box.findChildren(QtWidgets.QLineEdit)[0]  # volume
    edit.setFocus()
    edit.selectAll()
    QtTest.QTest.keyClicks(edit, "0.37")
    QtTest.QTest.keyClick(edit, KEY.Key_Return)
    assert viz.son.volume == 0.37
