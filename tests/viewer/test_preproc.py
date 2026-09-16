"""Preproc: tools that make a dataset, and the form that runs one.

The invariants worth guarding are the ones that make the mode a teaching
surface rather than a file generator: the output lands next to its input so the
two can be flipped between, a second run replaces the first instead of piling
up, and nothing is written to disk unless someone asks for it.

Motion correction itself is not retested here -- ffs_moco has its own tests.
What is tested is the translation: the axis order across the boundary, and the
wiring from a declared button to an installed layer.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

# Must be set before any QApplication is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fastfuncstuff.viewer.modes import registry
from fastfuncstuff.viewer.modes.base import ChoiceControl
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.tools import registry as tools
from fastfuncstuff.viewer.tools.base import AuxVolume, Tool, ToolOutcome
from fastfuncstuff.viewer.tools.moco import base_index, qc_volumes
from fastfuncstuff.viewer.vocab import SetMode

nib = pytest.importorskip("nibabel")
pytest.importorskip("PySide6")
CPU = torch.device("cpu")

from PySide6 import QtWidgets  # noqa: E402  (after the importorskip above)

from fastfuncstuff.viewer.ui.window import ViewerWindow  # noqa: E402


def _write(d, name, data, tr=0.0):
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    aff[:3, 3] = (-12.0, -13.5, -10.0)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), aff)
    if tr:
        img.header["pixdim"][4] = tr
        img.header.set_xyzt_units("mm", "sec")
    p = d / name
    nib.save(img, str(p))
    return p


@pytest.fixture
def session(tmp_path):
    rng = np.random.default_rng(3)
    _write(tmp_path, "run1.nii.gz", rng.random((8, 9, 6, 5)) * 100, tr=2.0)
    _write(tmp_path, "run2.nii.gz", rng.random((8, 9, 6, 5)) * 100, tr=2.0)
    _write(tmp_path, "anat.nii.gz", rng.random((8, 9, 6)) * 100)
    s = ViewerSession(device=CPU)
    s.read_directory(tmp_path)
    yield s, tmp_path
    s.close()


@pytest.fixture
def preproc(session):
    s, d = session
    s.do(SetMode("preproc"))
    return s, d


# ---------------------------------------------------------------------------
# what the mode offers
# ---------------------------------------------------------------------------


def test_preproc_is_registered_and_makes_no_overlay(preproc):
    s, _ = preproc
    assert "preproc" in registry.names()
    assert s.mode.produces_overlay is False
    assert s.mode.compute() is None


def test_every_tool_gets_a_button(preproc):
    s, _ = preproc
    assert {a.name for a in s.mode.actions()} == {t.name for t in tools.all()}
    # produces_overlay is False, so the framework must not offer KEEP for a
    # layer the mode does not own.
    assert "keep" not in {a.name for a in s.mode.actions()}


def test_a_3d_dataset_is_never_offered_to_moco(preproc):
    s, d = preproc
    s.load(d / "run1.nii.gz")
    s.load(d / "anat.nii.gz")
    offered = s.mode.inputs_for(tools.find("moco"))
    assert "run1.nii.gz" in offered
    assert "anat.nii.gz" not in offered


def test_two_layers_of_the_same_name_stay_distinguishable(preproc):
    """Names come from files, so two directories can collide; keys cannot."""
    s, d = preproc
    a = s.load(d / "run1.nii.gz")
    b = s.load(d / "run1.nii.gz", key="second")
    offered = s.mode.inputs_for(tools.find("moco"))
    assert set(offered.values()) == {a, b}
    assert len(offered) == 2


def test_the_dialog_is_blocked_when_there_is_nothing_to_run_on(preproc):
    s, d = preproc
    s.load(d / "anat.nii.gz")
    spec = s.mode.dialog_for("moco")
    assert spec.blocked
    assert "4-D" in spec.blocked


def test_the_dialog_declares_the_input_and_the_tools_own_controls(preproc):
    s, d = preproc
    s.load(d / "run1.nii.gz")
    spec = s.mode.dialog_for("moco")
    assert not spec.blocked
    names = [c.name for c in spec.controls]
    assert names[0] == "input", "the input comes first, before anything tool-specific"
    assert {"base", "interp", "final_interp"} <= set(names)
    assert isinstance(spec.controls[0], ChoiceControl)
    assert spec.params["interp"] == "heptic"
    assert spec.params["final_interp"] == "wsinc5"


def test_a_button_that_is_not_a_tool_has_no_dialog(preproc):
    s, _ = preproc
    assert s.mode.dialog_for("not_a_tool") is None


def test_base_index_names_map_to_volumes():
    assert base_index("first", 10) == 0
    assert base_index("middle", 10) == 5
    assert base_index("last", 10) == 9


# ---------------------------------------------------------------------------
# running one, without running a real motion correction
# ---------------------------------------------------------------------------


class DoubleTool(Tool):
    """A stand-in: same contract, arithmetic cheap enough to assert on."""

    name = "double"
    label = "Double"
    tag = "DOUBLE"
    op = "double"
    input_kind = "4d"
    blurb = "Doubles every voxel."

    def run(self, session, params, progress=None):
        values = np.asarray(session.store.ensure_ram(params["input"]))
        return ToolOutcome(
            values=values * 2.0,
            detail="x2",
            aux=[
                AuxVolume(
                    slot="pair",
                    name="pair",
                    values=values[..., :2],
                    labels=("first", "last"),
                ),
                AuxVolume(
                    slot="signed",
                    name="signed",
                    values=(values[..., -1] - values[..., 0])[..., None],
                    labels=("diff",),
                    colormap="redblue",
                    symmetric=True,
                ),
            ],
        )


@pytest.fixture
def with_double():
    tools._tools["double"] = DoubleTool()
    yield
    del tools._tools["double"]


def _run(mode, tool_name, params):
    """Drive a spec the way the dialog does: run, then install."""
    spec = mode.dialog_for(tool_name)
    merged = {**spec.params, **params}
    return spec.install(spec.run(merged, None))


def _results(session, op="double"):
    """Layers that are a tool's result, not the QC volumes made beside it."""
    return [layer for layer in session.state.layers if layer.source.startswith(f"derived:{op}:")]


def test_the_output_lands_directly_above_its_input(preproc, with_double):
    s, d = preproc
    key = s.load(d / "run1.nii.gz")
    _run(s.mode, "double", {})
    keys = [layer.key for layer in s.state.layers]
    made = s.state.layers.find_by_source(f"derived:double:{key}")
    assert made is not None
    assert keys.index(made.key) == keys.index(key) + 1


def test_the_output_is_named_for_the_controller_and_the_tool(preproc, with_double):
    s, d = preproc
    s.load(d / "run1.nii.gz")
    _run(s.mode, "double", {})
    made = next(layer for layer in s.state.layers if layer.is_derived)
    assert made.name.endswith("_DOUBLE")


def test_running_again_replaces_rather_than_piles_up(preproc, with_double):
    """Ten looks at one interpolation choice must leave one layer, not ten."""
    s, d = preproc
    s.load(d / "run1.nii.gz")
    for _ in range(3):
        _run(s.mode, "double", {})
    assert len(_results(s)) == 1


def test_running_on_a_different_input_makes_a_second_layer(preproc, with_double):
    s, d = preproc
    s.load(d / "run1.nii.gz")
    s.load(d / "run2.nii.gz")
    offered = s.mode.inputs_for(tools.find("double"))
    for label in offered:
        _run(s.mode, "double", {"input": label})
    assert len(_results(s)) == 2


def test_the_result_stays_in_memory_and_writes_nothing(preproc, with_double, tmp_path):
    s, d = preproc
    before = {p.name for p in d.iterdir()}
    s.load(d / "run1.nii.gz")
    _run(s.mode, "double", {})
    assert {p.name for p in d.iterdir()} == before, "a tool must not write to the data directory"

    made = next(layer for layer in s.state.layers if layer.is_derived)
    assert s.store.get(made.key).memory_backed
    # ...but saving it is one call, and that is the only thing that writes.
    out = s.save_layer(made.key, tmp_path / "kept.nii.gz")
    assert out.exists()


def test_an_unknown_input_is_refused_rather_than_guessed(preproc, with_double):
    s, d = preproc
    s.load(d / "run1.nii.gz")
    spec = s.mode.dialog_for("double")
    with pytest.raises(ValueError, match="not a layer"):
        spec.run({**spec.params, "input": "gone.nii.gz"}, None)


# ---------------------------------------------------------------------------
# the moco tool itself: the boundary it is responsible for
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_moco_returns_the_viewer_axis_order_it_was_given(preproc):
    """The one real job: (nx, ny, nz, nt) out, whatever the library works in."""
    s, d = preproc
    key = s.load(d / "run1.nii.gz")
    shape = s.store.ensure_ram(key).shape
    spec = s.mode.dialog_for("moco")
    outcome = spec.run({**spec.params, "input": next(iter(spec.controls[0].choices))}, None)
    assert outcome.values.shape == shape
    assert outcome.values.dtype == np.float32
    assert np.isfinite(outcome.values).all()


# ---------------------------------------------------------------------------
# the dialog, offscreen
#
# Not pixel tests. What is checked is the wiring the mode cannot check for
# itself: that pressing a declared button opens a form instead of dispatching
# an action nobody implemented, and that the form does not outlive the session
# it was built against.
# ---------------------------------------------------------------------------


@pytest.fixture
def qapp():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(qapp, session):
    s, d = session
    w = ViewerWindow(s)
    w.read_directory(d)
    w.refresh(s.do(SetMode("preproc")))
    yield w, d
    w.close()


def test_pressing_a_tool_button_opens_its_form(win, qapp):
    w, d = win
    w.session.load(d / "run1.nii.gz")
    w._mode_action("moco")
    qapp.processEvents()
    assert "moco" in w._tool_dialogs
    dialog = w._tool_dialogs["moco"]
    assert dialog.isVisible()
    assert dialog.tool_name == "moco"


def test_pressing_it_again_reuses_the_same_window(win, qapp):
    w, d = win
    w.session.load(d / "run1.nii.gz")
    w._mode_action("moco")
    first = w._tool_dialogs["moco"]
    w._mode_action("moco")
    qapp.processEvents()
    assert w._tool_dialogs["moco"] is first


def test_a_form_does_not_survive_a_controller_switch(win, qapp):
    """Its spec closes over one session; running it against another is a bug."""
    w, d = win
    w.session.load(d / "run1.nii.gz")
    w._mode_action("moco")
    assert w._tool_dialogs

    w.new_controller()
    qapp.processEvents()
    assert not w._tool_dialogs


# ---------------------------------------------------------------------------
# QC volumes: the half of a preproc step that teaches
# ---------------------------------------------------------------------------


def test_qc_volumes_split_intensity_from_difference():
    """One stack cannot window both: intensity units against a signed map."""
    series = np.zeros((4, 4, 3, 5), np.float32)
    series[..., 0] = 10.0
    series[..., -1] = 14.0
    pair, diff = qc_volumes(series, "before")

    assert pair.values.shape == (4, 4, 3, 2)
    assert pair.labels == ("first", "last")
    assert not pair.symmetric

    assert diff.values.shape == (4, 4, 3, 1)
    assert np.allclose(diff.values[..., 0], 4.0), "signed last - first"
    assert diff.symmetric and diff.colormap == "redblue"


def test_qc_volumes_are_named_for_when_they_were_taken():
    before = {v.slot for v in qc_volumes(np.zeros((2, 2, 2, 3), np.float32), "before")}
    after = {v.slot for v in qc_volumes(np.zeros((2, 2, 2, 3), np.float32), "after")}
    assert not (before & after), "before and after must not replace each other"


def test_qc_layers_arrive_hidden(preproc, with_double):
    """Four QC volumes switched on at once would bury the anatomy."""
    s, d = preproc
    s.load(d / "run1.nii.gz")
    _run(s.mode, "double", {})
    made = [layer for layer in s.state.layers if layer.is_derived]
    assert len(made) == 3
    hidden = [layer for layer in made if not layer.visible]
    assert len(hidden) == 2, "the result shows; its QC volumes wait to be flipped to"


def test_the_result_stays_adjacent_to_its_source(preproc, with_double):
    """QC volumes must not push the result away from what [ and ] compare."""
    s, d = preproc
    key = s.load(d / "run1.nii.gz")
    _run(s.mode, "double", {})
    keys = [layer.key for layer in s.state.layers]
    result = s.state.layers.find_by_source(f"derived:double:{key}")
    assert keys.index(result.key) == keys.index(key) + 1


def test_a_signed_map_gets_its_own_window_not_the_runs(preproc, with_double):
    s, d = preproc
    s.load(d / "run1.nii.gz")
    _run(s.mode, "double", {})
    signed = next(layer for layer in s.state.layers if layer.name.endswith("signed"))
    assert signed.colormap == "redblue"
    assert signed.range_lo is not None and signed.range_lo == pytest.approx(-signed.range_hi)


def test_qc_volumes_do_not_follow_the_time_slider(preproc, with_double):
    """Its sub-bricks are 'first' and 'last', not time points."""
    s, d = preproc
    s.load(d / "run1.nii.gz")
    _run(s.mode, "double", {})
    pair = next(layer for layer in s.state.layers if layer.name.endswith("pair"))
    assert not pair.time_linked


def test_re_running_replaces_the_qc_volumes_too(preproc, with_double):
    s, d = preproc
    s.load(d / "run1.nii.gz")
    for _ in range(3):
        _run(s.mode, "double", {})
    assert sum(1 for layer in s.state.layers if layer.is_derived) == 3


def test_a_tool_with_no_qc_volumes_still_installs(preproc):
    """aux is optional; the contract must not require it."""

    class Bare(Tool):
        name, label, tag, op = "bare", "Bare", "BARE", "bare"
        blurb = ""

        def run(self, session, params, progress=None):
            return ToolOutcome(values=np.asarray(session.store.ensure_ram(params["input"])))

    tools._tools["bare"] = Bare()
    try:
        s, d = preproc
        s.load(d / "run1.nii.gz")
        _run(s.mode, "bare", {})
        assert len(_results(s, "bare")) == 1
        assert sum(1 for layer in s.state.layers if layer.is_derived) == 1
    finally:
        del tools._tools["bare"]


# ---------------------------------------------------------------------------
# the details pane: what the job printed, folded away until asked for
# ---------------------------------------------------------------------------


def test_line_writer_emits_whole_lines_not_print_fragments():
    """print() writes the text and the newline separately."""
    from fastfuncstuff.viewer.ui.work import _LineWriter

    seen = []
    w = _LineWriter(seen.append)
    print("hello", file=w)
    print("world", file=w)
    assert seen == ["hello", "world"]


def test_line_writer_flushes_a_trailing_line_without_a_newline():
    from fastfuncstuff.viewer.ui.work import _LineWriter

    seen = []
    w = _LineWriter(seen.append)
    w.write("no newline yet")
    assert seen == []
    w.flush()
    assert seen == ["no newline yet"]


class ChattyTool(Tool):
    name, label, tag, op = "chatty", "Chatty", "CHATTY", "chatty"
    blurb = "Prints while it works."

    def run(self, session, params, progress=None):
        print("chatty: starting")
        print("chatty: two things happened")
        return ToolOutcome(values=np.asarray(session.store.ensure_ram(params["input"])))


@pytest.fixture
def with_chatty():
    tools._tools["chatty"] = ChattyTool()
    yield
    del tools._tools["chatty"]


def _run_dialog(win, qapp, name):
    """Press a tool button and drive the worker to completion."""
    win._mode_action(name)
    dialog = win._tool_dialogs[name]
    dialog._start()
    for _ in range(2000):
        qapp.processEvents()
        if not dialog._running:
            break
    win.runner.wait(30_000)
    for _ in range(20):
        qapp.processEvents()
    return dialog


def test_what_a_tool_prints_reaches_the_details_pane(win, qapp, with_chatty):
    w, d = win
    w.session.load(d / "run1.nii.gz")
    dialog = _run_dialog(w, qapp, "chatty")
    assert "chatty: starting" in dialog._log.toPlainText()
    assert "chatty: two things happened" in dialog._log.toPlainText()


def test_the_pane_stays_folded_and_is_only_offered_once_there_is_output(win, qapp, with_chatty):
    w, d = win
    w.session.load(d / "run1.nii.gz")
    w._mode_action("chatty")
    assert not w._tool_dialogs["chatty"]._details.isVisible(), "nothing printed yet"

    dialog = _run_dialog(w, qapp, "chatty")
    assert dialog._details.isVisible(), "offered once there is something to read"
    assert not dialog._log.isVisible(), "but still folded away"

    dialog._details.setChecked(True)
    assert dialog._log.isVisible()


def test_a_silent_tool_never_grows_a_disclosure_arrow(win, qapp, with_double):
    w, d = win
    w.session.load(d / "run1.nii.gz")
    dialog = _run_dialog(w, qapp, "double")
    assert not dialog._details.isVisible()
    assert dialog._log.toPlainText() == ""


def test_the_log_describes_this_run_not_the_last_one(win, qapp, with_chatty):
    w, d = win
    w.session.load(d / "run1.nii.gz")
    first = _run_dialog(w, qapp, "chatty")
    assert first._log.toPlainText().count("chatty: starting") == 1

    second = _run_dialog(w, qapp, "chatty")
    assert second is first
    assert second._log.toPlainText().count("chatty: starting") == 1


# ---------------------------------------------------------------------------
# the motion plot
# ---------------------------------------------------------------------------


def test_motion_panels_split_rotation_from_translation():
    """One y-axis per panel, so a panel's lines must share units."""
    from fastfuncstuff.viewer.tools.moco import ROTATION, TRANSLATION, motion_panels

    panels = motion_panels(np.zeros((12, 6)))
    assert set(panels) == {ROTATION, TRANSLATION}
    assert [t.legend for t in panels[ROTATION]] == ["roll (deg)", "pitch (deg)", "yaw (deg)"]
    assert [t.legend for t in panels[TRANSLATION]] == ["dS (mm)", "dL (mm)", "dP (mm)"]
    assert all(t.x_label == "TR" for t in panels[ROTATION])


def test_the_plot_cannot_drift_from_the_1D_file(tmp_path):
    """The teaching failure this guards against is a plot that quietly lies.

    The solver reports the *correction*; everyone means the subject's motion,
    which is its negation in a different column order. A viewer that read the
    solver's columns directly would draw a picture disagreeing with the .1D
    written beside it from the same run.
    """
    from fastfuncstuff.processing.ffs_moco import save_moco_1D
    from fastfuncstuff.viewer.tools.moco import ROTATION, TRANSLATION, motion_panels

    rng = np.random.default_rng(4)
    params = rng.normal(0, 2, size=(20, 6))

    panels = motion_panels(params)
    plotted = np.concatenate(
        [
            np.stack([t.values for t in panels[ROTATION]], axis=1),
            np.stack([t.values for t in panels[TRANSLATION]], axis=1),
        ],
        axis=1,
    )
    save_moco_1D(params, str(tmp_path / "ref.1D"))
    # The file is %8.4f, so it round-trips only to that precision.
    assert np.allclose(plotted, np.loadtxt(tmp_path / "ref.1D"), atol=1e-4)


def test_moco_hands_the_layer_affine_to_the_solver(preproc, monkeypatch):
    """Without it the reported motion is in the wrong space, silently.

    Omitting header_info does not fail; it produces parameters that differ by
    millimetres, which is exactly the kind of wrong that gets believed.
    """
    import fastfuncstuff.processing.ffs_moco as ffs_moco

    s, d = preproc
    key = s.load(d / "run1.nii.gz")
    seen = {}
    real = ffs_moco.moco

    def spy(series, config, header_info=None, base_vol=None):
        seen["affine"] = None if header_info is None else header_info.get("affine")
        return real(series, config, header_info=header_info, base_vol=base_vol)

    monkeypatch.setattr(ffs_moco, "moco", spy)
    spec = s.mode.dialog_for("moco")
    spec.run({**spec.params, "interp": "linear", "final_interp": "linear"}, None)

    expected = s.state.layers.get(key).affine
    assert seen["affine"] is not None, "the solver was given no affine"
    assert np.allclose(seen["affine"], expected)


def test_preproc_has_no_plots_until_something_has_run(preproc, with_double):
    s, d = preproc
    assert s.mode.panel_names() == ()
    s.load(d / "run1.nii.gz")
    _run(s.mode, "double", {})
    assert s.mode.panel_names() == (), "a tool with no panels adds none"


def test_a_tools_plots_replace_the_last_tools(preproc):
    """Two tools' plots on screen at once is how the wrong one gets read."""

    class Plotter(Tool):
        name, label, tag, op = "plotter", "Plotter", "PLOT", "plotter"
        blurb = ""

        def __init__(self, panel):
            self._panel = panel

        def run(self, session, params, progress=None):
            from fastfuncstuff.viewer.modes.base import Trace

            return ToolOutcome(
                values=np.asarray(session.store.ensure_ram(params["input"])),
                panels={self._panel: [Trace(label="x", values=np.arange(8.0))]},
            )

    s, d = preproc
    s.load(d / "run1.nii.gz")
    tools._tools["plotter"] = Plotter("first plot")
    try:
        _run(s.mode, "plotter", {})
        assert s.mode.panel_names() == ("first plot",)
        tools._tools["plotter"] = Plotter("second plot")
        _run(s.mode, "plotter", {})
        assert s.mode.panel_names() == ("second plot",)
    finally:
        del tools._tools["plotter"]


def test_a_panel_draws_every_line_it_was_given(qapp):
    from fastfuncstuff.viewer.modes.base import Trace
    from fastfuncstuff.viewer.ui.tracewindow import PlotView

    view = PlotView()
    view.set_traces([Trace(label=f"l{i}", values=np.arange(10.0) * i) for i in range(1, 4)])
    assert len(view._traces) == 3

    # A line of one point cannot be drawn and must not blank the panel.
    view.set_traces(
        [Trace(label="ok", values=np.arange(10.0)), Trace(label="short", values=np.array([1.0]))]
    )
    assert [t.label for t in view._traces] == ["ok"]
