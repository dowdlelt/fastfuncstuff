"""The right-click picker every dropdown gets.

What matters is that choosing in the picker is indistinguishable from choosing
in the combo's own popup -- the viewer's combos are wired to ``activated``, and
a picker that only set the index would move the box without changing anything.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from fastfuncstuff.viewer.ui.widgets import (  # noqa: E402
    ComboPicker,
    install_combo_pickers,
)

NAMES = ["gray", "hot", "viridis", "RdBu", "RdYlBu", "twilight", "twilight_shifted"]


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    install_combo_pickers(app)
    return app


@pytest.fixture
def form(qapp):
    host = QtWidgets.QWidget()
    layout = QtWidgets.QFormLayout(host)
    combo = QtWidgets.QComboBox()
    combo.addItems(NAMES)
    layout.addRow(QtWidgets.QLabel("[C]OLOR"), combo)
    host.show()
    qapp.processEvents()
    yield combo
    for picker in host.findChildren(ComboPicker):
        picker.close()
    host.close()


def right_click(combo, qapp):
    event = QtGui.QContextMenuEvent(
        QtGui.QContextMenuEvent.Reason.Mouse,
        QtCore.QPoint(5, 5),
        combo.mapToGlobal(QtCore.QPoint(5, 5)),
    )
    QtWidgets.QApplication.sendEvent(combo, event)
    qapp.processEvents()
    pickers = [w for w in combo.window().findChildren(ComboPicker) if w.isVisible()]
    return pickers[-1] if pickers else None


def test_right_click_opens_a_picker_of_every_item(form, qapp):
    picker = right_click(form, qapp)
    assert picker is not None
    assert [picker.list.item(r).text() for r in range(picker.list.count())] == NAMES
    assert picker.windowTitle() == "COLOR"


def test_choosing_goes_through_activated(form, qapp):
    seen = []
    form.activated.connect(seen.append)
    picker = right_click(form, qapp)
    picker.filter.setText("shift")
    QtWidgets.QApplication.sendEvent(
        picker.filter,
        QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.KeyboardModifier.NoModifier,
        ),
    )
    qapp.processEvents()
    assert form.currentText() == "twilight_shifted"
    assert seen == [NAMES.index("twilight_shifted")]
    assert not [w for w in form.window().findChildren(ComboPicker) if w.isVisible()]


def test_the_filter_hides_non_matches_and_arrows_skip_them(form, qapp):
    picker = right_click(form, qapp)
    picker.filter.setText("rd")
    shown = [
        picker.list.item(r).text()
        for r in range(picker.list.count())
        if not picker.list.item(r).isHidden()
    ]
    assert shown == ["RdBu", "RdYlBu"]
    assert picker.list.currentItem().text() == "RdBu"
    QtWidgets.QApplication.sendEvent(
        picker.filter,
        QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Down,
            QtCore.Qt.KeyboardModifier.NoModifier,
        ),
    )
    assert picker.list.currentItem().text() == "RdYlBu"


def test_a_disabled_combo_offers_no_picker(form, qapp):
    form.setEnabled(False)
    assert right_click(form, qapp) is None


def test_installing_twice_filters_once(qapp):
    install_combo_pickers(qapp)
    install_combo_pickers(qapp)
    from fastfuncstuff.viewer.ui.widgets import _ComboPickerFilter

    assert len(qapp.findChildren(_ComboPickerFilter)) == 1
