"""The generic control panel: layout, conditional visibility, path lists.

These are the properties a mode relies on without being able to see them. A
mode declares ``visible_when`` and trusts that the space is kept; it declares a
span and trusts that a half-width control stays half-width; it declares a path
list and trusts that the panel does not throw away a half-finished edit on the
next refit. Each of those is invisible in the mode's own tests and each of them
has a way of quietly stopping working.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

# Must be set before any QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets  # noqa: E402

from fastfuncstuff.viewer.modes.base import (  # noqa: E402
    FULL,
    HALF,
    THIRD,
    BoolControl,
    ChoiceControl,
    FloatControl,
    PathListControl,
)
from fastfuncstuff.viewer.ui.controls import ControlPanel  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _panel(qapp, controls, values, width=600):
    panel = ControlPanel()
    panel.rebuild(controls, values)
    panel.resize(width, 400)
    panel.show()
    qapp.processEvents()
    return panel


def _rows(panel):
    """Which control names landed on each row, in order."""
    out = []
    for i in range(panel._rows.count()):
        widget = panel._rows.itemAt(i).widget()
        if widget is None:
            continue
        names = [n for n, cell in panel._cells.items() if cell.parent() is widget]
        if names:
            out.append(names)
    return out


# -- layout -----------------------------------------------------------------


def test_controls_pack_across_a_row_until_the_budget_runs_out(qapp):
    specs = [
        ChoiceControl(name=f"c{i}", label=f"c{i}", choices=("a",), span=THIRD) for i in range(4)
    ]
    panel = _panel(qapp, specs, {})
    assert _rows(panel) == [["c0", "c1", "c2"], ["c3"]]


def test_newline_breaks_a_row_that_would_otherwise_have_fitted(qapp):
    specs = [
        ChoiceControl(name="a", label="a", choices=("x",), span=THIRD),
        ChoiceControl(name="b", label="b", choices=("x",), span=THIRD, newline=True),
    ]
    assert _rows(_panel(qapp, specs, {})) == [["a"], ["b"]]


def test_a_lone_half_width_control_keeps_half_the_row(qapp):
    """Without the padding stretch a span says nothing: the only control on a
    row takes all of it whatever it asked for."""
    wide = _panel(qapp, [ChoiceControl(name="a", label="a", choices=("x",), span=FULL)], {})
    narrow = _panel(qapp, [ChoiceControl(name="a", label="a", choices=("x",), span=HALF)], {})
    assert narrow._cells["a"].width() < wide._cells["a"].width()


# -- conditional visibility -------------------------------------------------


def test_a_dependent_control_is_hidden_but_keeps_its_place(qapp):
    specs = [
        ChoiceControl(name="basis", label="HRF", choices=("spmg1", "custom"), default="spmg1"),
        FloatControl(name="peak", label="peak", visible_when=("basis", ("custom",))),
        BoolControl(name="after", label="after"),
    ]
    panel = _panel(qapp, specs, {"basis": "spmg1"})
    assert not panel._cells["peak"].isVisible()
    hidden_y = panel._cells["after"].parent().y()

    panel._queue("basis", "custom", now=True)
    qapp.processEvents()
    assert panel._cells["peak"].isVisible()
    # Nothing below it moved, which is the whole reason the space is kept.
    assert panel._cells["after"].parent().y() == hidden_y


def test_visibility_follows_the_click_and_not_the_debounce(qapp):
    """The reveal has to happen on the press. Waiting for the debounce means
    waiting for the refit it also triggered, which is seconds on a real run."""
    specs = [
        ChoiceControl(name="basis", label="HRF", choices=("a", "b"), default="a"),
        BoolControl(name="dep", label="dep", visible_when=("basis", ("b",))),
    ]
    panel = _panel(qapp, specs, {"basis": "a"})
    seen = []
    panel.changed.connect(lambda n, v: seen.append((n, v)))
    panel._queue("basis", "b")  # debounced, not flushed
    assert panel._cells["dep"].isVisible()
    assert seen == []


# -- stability across rebuilds ----------------------------------------------


def test_an_unchanged_rebuild_keeps_the_same_widgets(qapp):
    """The panel is rebuilt on every refit. A path list mid-edit, a spin box
    with the caret in it and a slider under the thumb all die with it."""
    specs = [ChoiceControl(name="a", label="a", choices=("x", "y"), default="x")]
    panel = _panel(qapp, specs, {"a": "x"})
    before = panel._widgets["a"]
    panel.rebuild(specs, {"a": "x"})
    assert panel._widgets["a"] is before


def test_a_changed_choice_list_does_rebuild(qapp):
    """The column picker's choices follow the model, so this rebuild is real."""
    panel = _panel(qapp, [ChoiceControl(name="a", label="a", choices=("x",))], {})
    before = panel._widgets["a"]
    panel.rebuild([ChoiceControl(name="a", label="a", choices=("x", "y"))], {})
    assert panel._widgets["a"] is not before


def test_a_rebuild_writes_new_state_into_the_widgets(qapp):
    specs = [ChoiceControl(name="a", label="a", choices=("x", "y"), default="x")]
    panel = _panel(qapp, specs, {"a": "x"})
    panel.rebuild(specs, {"a": "y"})
    assert panel._widgets["a"].currentText() == "y"


# -- widgets ----------------------------------------------------------------


def test_a_float_can_be_typed_as_well_as_dragged(qapp):
    spec = FloatControl(name="peak", label="peak", lo=2.0, hi=12.0, default=6.0, step=0.25)
    panel = _panel(qapp, [spec], {})
    seen = []
    panel.changed.connect(lambda n, v: seen.append((n, float(v))))

    spin = panel._widgets["peak"].findChild(QtWidgets.QDoubleSpinBox)
    slider = panel._widgets["peak"].findChild(QtWidgets.QSlider)
    spin.setValue(9.0)
    qapp.processEvents()
    assert seen == [("peak", 9.0)]
    # And the slider went with it, rather than the two disagreeing.
    assert slider.value() == pytest.approx(700, abs=2)


def test_dragging_the_slider_moves_the_spin_box_without_echoing_back(qapp):
    spec = FloatControl(name="peak", label="peak", lo=0.0, hi=10.0, default=0.0, step=0.5)
    panel = _panel(qapp, [spec], {})
    seen = []
    panel.changed.connect(lambda n, v: seen.append(float(v)))
    panel._widgets["peak"].findChild(QtWidgets.QSlider).setValue(500)
    panel.flush_now()
    assert panel._widgets["peak"].findChild(QtWidgets.QDoubleSpinBox).value() == pytest.approx(5.0)
    assert seen == [5.0]


def test_radio_choices_emit_the_choice_they_name(qapp):
    spec = ChoiceControl(
        name="psc", label="units", choices=("swing", "per unit"), default="swing", style="radio"
    )
    panel = _panel(qapp, [spec], {})
    seen = []
    panel.changed.connect(lambda n, v: seen.append(v))
    buttons = panel._widgets["psc"].findChildren(QtWidgets.QRadioButton)
    assert [b.text() for b in buttons] == ["swing", "per unit"]
    buttons[1].setChecked(True)
    qapp.processEvents()
    assert seen == ["per unit"]


# -- path lists -------------------------------------------------------------


def test_a_path_list_round_trips_through_its_encoding():
    assert PathListControl.parse("+a.1D|-b.1D") == [("a.1D", True), ("b.1D", False)]
    assert PathListControl.enabled("+a.1D|-b.1D") == ["a.1D"]
    assert PathListControl.encode([("a.1D", True), ("b.1D", False)]) == "+a.1D|-b.1D"
    # What a script written before the list existed passes.
    assert PathListControl.parse("a.1D") == [("a.1D", True)]
    assert PathListControl.parse("") == []


def test_unticking_an_entry_keeps_it_in_the_value(qapp):
    spec = PathListControl(name="ortvec", label="ortvec")
    panel = _panel(qapp, [spec], {"ortvec": "+/tmp/a.1D|+/tmp/b.1D"})
    seen = []
    panel.changed.connect(lambda n, v: seen.append(v))

    listing = panel._widgets["ortvec"].findChild(QtWidgets.QListWidget)
    assert [listing.item(i).text() for i in range(2)] == ["a.1D", "b.1D"]
    listing.item(1).setCheckState(QtCore.Qt.CheckState.Unchecked)
    qapp.processEvents()
    assert seen == ["+/tmp/a.1D|-/tmp/b.1D"]


def _button(panel, name, text):
    return next(
        b for b in panel._widgets[name].findChildren(QtWidgets.QPushButton) if b.text() == text
    )


def test_the_derivative_button_adds_an_entry_under_its_source(qapp):
    spec = PathListControl(name="ortvec", label="ortvec", transforms=True, sample_interval=2.0)
    panel = _panel(qapp, [spec], {"ortvec": "+/tmp/motion.1D|+/tmp/resp.1D"})
    seen = []
    panel.changed.connect(lambda n, v: seen.append(v))
    listing = panel._widgets["ortvec"].findChild(QtWidgets.QListWidget)

    listing.item(0).setSelected(True)
    _button(panel, "ortvec", "∂").click()
    qapp.processEvents()
    assert seen[-1] == "+/tmp/motion.1D|+/tmp/motion.1D:deriv|+/tmp/resp.1D"
    assert listing.item(1).text() == "motion.1D  d/dt"

    # Pressed on the derivative, it gives the second one.
    listing.item(1).setSelected(True)
    _button(panel, "ortvec", "∂").click()
    qapp.processEvents()
    assert "+/tmp/motion.1D:deriv:deriv" in seen[-1].split("|")
    assert listing.item(2).text() == "motion.1D  d²/dt²"


def test_the_split_button_swaps_the_source_for_its_bands(qapp, monkeypatch):
    from fastfuncstuff.viewer.ui import bandsplit

    monkeypatch.setattr(
        bandsplit,
        "split_interactively",
        lambda entry, tr, parent: [f"{entry}:band=0-0.05", f"{entry}:band=0.05-nyq"],
    )
    spec = PathListControl(name="ortvec", label="ortvec", transforms=True, sample_interval=2.0)
    panel = _panel(qapp, [spec], {"ortvec": "+/tmp/motion.1D|+/tmp/resp.1D"})
    seen = []
    panel.changed.connect(lambda n, v: seen.append(v))
    listing = panel._widgets["ortvec"].findChild(QtWidgets.QListWidget)
    listing.item(0).setSelected(True)
    _button(panel, "ortvec", "≋").click()
    qapp.processEvents()
    assert seen[-1].split("|") == [
        "-/tmp/motion.1D",
        "+/tmp/motion.1D:band=0-0.05",
        "+/tmp/motion.1D:band=0.05-nyq",
        "+/tmp/resp.1D",
    ]
    assert listing.item(1).text() == "motion.1D  <0.05Hz"


def test_a_plain_path_list_has_no_transform_buttons(qapp):
    spec = PathListControl(name="files", label="files")
    panel = _panel(qapp, [spec], {"files": ""})
    texts = {b.text() for b in panel._widgets["files"].findChildren(QtWidgets.QPushButton)}
    assert texts == {"+", "−"}


def test_the_band_dialog_turns_clicks_into_entries(qapp):
    """Cuts placed on the spectrum and cuts typed are the same cuts, and only
    the ticked columns are split -- the rest come along unsplit."""
    import numpy as np

    from fastfuncstuff.viewer.ui.bandsplit import BandSplitDialog

    rng = np.random.default_rng(0)
    columns = np.cumsum(rng.normal(size=(100, 3)), axis=0)
    dialog = BandSplitDialog("m.1D", columns, ["x", "y", "z"], tr=1.0)
    ok = dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled(), "no cuts yet, nothing to add"

    dialog.cut_edit.setText("0.1, 0.3")
    dialog._cuts_from_text()
    assert dialog.cutoffs() == [0.1, 0.3]
    assert ok.isEnabled()
    dialog.column_list.item(1).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert dialog.entries() == [
        "m.1D:cols=0,2:band=0-0.1",
        "m.1D:cols=0,2:band=0.1-0.3",
        "m.1D:cols=0,2:band=0.3-nyq",
        "m.1D:cols=1",
    ]

    # A click on the plot adds a cut where it lands; the text follows.
    dialog.resize(800, 500)
    dialog.show()
    qapp.processEvents()
    view = dialog.spectrum
    x = view._to_x(0.2)
    from PySide6.QtTest import QTest

    QTest.mouseClick(view, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(int(x), 60))
    assert len(dialog.cutoffs()) == 3
    assert abs(dialog.cutoffs()[1] - 0.2) < 0.01
    assert dialog.cut_edit.text().count(",") == 2
    QTest.mouseClick(view, QtCore.Qt.MouseButton.RightButton, pos=QtCore.QPoint(int(x), 60))
    assert dialog.cutoffs() == [0.1, 0.3]
    dialog.close()


def test_a_path_list_is_as_tall_as_its_rows(qapp):
    """Measured after styling, not when the items are added: a row's height is
    not known until the theme's stylesheet has arrived."""
    from fastfuncstuff.viewer.ui import theme

    spec = PathListControl(name="ortvec", label="ortvec")
    panel = ControlPanel()
    panel.setStyleSheet(theme.stylesheet())
    panel.rebuild([spec], {"ortvec": "+/tmp/a.1D|+/tmp/b.1D"})
    panel.resize(600, 400)
    panel.show()
    qapp.processEvents()
    listing = panel._widgets["ortvec"].findChild(QtWidgets.QListWidget)
    assert listing.sizeHint().height() >= 2 * listing.sizeHintForRow(0)


def test_a_half_typed_number_survives_a_rebuild(qapp):
    """A refit lands while someone is typing. Overwriting the caret's line is
    how a control ends up fighting the person using it."""
    spec = FloatControl(name="peak", label="peak", lo=0.0, hi=10.0, default=1.0, step=0.5)
    panel = _panel(qapp, [spec], {"peak": 1.0})
    spin = panel._widgets["peak"].findChild(QtWidgets.QDoubleSpinBox)
    spin.lineEdit().setFocus()
    qapp.processEvents()
    panel.rebuild([spec], {"peak": 8.0})
    assert spin.value() == pytest.approx(1.0)


def test_a_focused_combo_still_follows_state(qapp):
    """Focus alone is not an edit: a combo holds it from the first click and
    commits on activation, so refusing to update one means the panel stops
    following state the moment anyone touches it."""
    spec = ChoiceControl(name="a", label="a", choices=("x", "y"), default="x")
    panel = _panel(qapp, [spec], {"a": "x"})
    panel._widgets["a"].setFocus()
    qapp.processEvents()
    panel.rebuild([spec], {"a": "y"})
    assert panel._widgets["a"].currentText() == "y"
