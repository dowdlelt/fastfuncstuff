"""What a mode reads, as its own question.

The invariant under all of this is that "what is computed from" and "what is on
screen" are independent. Before there was an input of its own, a mode found its
run by searching the stack for the topmost time series and the session hid
everything else when the result landed -- so loading an anatomical, a run and a
GLM meant you could not unticked the run without the next fit changing, and you
could not keep a second map beside the first at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.modes import registry
from fastfuncstuff.viewer.modes.base import ComputedOverlay, Mode, OverlayKind
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.vocab import (
    AddOverlay,
    SetInput,
    SetLayerVisible,
    SetMode,
    SetUnderlay,
)

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")


def _write(d, name, data, tr=0.0):
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    aff[:3, 3] = (-12.0, -13.5, -10.5)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), aff)
    if tr:
        img.header["pixdim"][4] = tr
        img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(d / name))
    return d / name


@pytest.fixture
def datadir(tmp_path):
    rng = np.random.default_rng(3)
    _write(tmp_path, "anat.nii.gz", rng.random((8, 9, 7)) * 100)
    _write(tmp_path, "run1.nii.gz", rng.random((8, 9, 7, 12)), tr=2.0)
    _write(tmp_path, "run2.nii.gz", rng.random((8, 9, 7, 12)), tr=2.0)
    _write(tmp_path, "stats.nii.gz", rng.normal(size=(8, 9, 7)))
    return tmp_path


class _Sum(Mode):
    """Writes the mean of whatever run it was pointed at into every voxel.

    Flat on purpose: the value identifies the input, so a test can tell which
    run the fit actually used rather than only that it produced something.
    """

    name = "test_sum"
    label = "Sum"
    overlay_kind = OverlayKind.VALUE

    def compute(self):
        layer = self.source_layer()
        if layer is None or self.session is None:
            return None
        data = np.asarray(self.session.store.ensure_ram(layer.key), dtype=np.float32)
        return ComputedOverlay(
            values=np.full(data.shape[:3], float(data.mean()), dtype=np.float32),
            affine=layer.affine,
            name=self.output_name(),
        )


@pytest.fixture(autouse=True)
def _register():
    registry.register(_Sum)
    yield


@pytest.fixture
def session(datadir):
    s = ViewerSession(device=CPU)
    s.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    s.do(AddOverlay(str(datadir / "run1.nii.gz")))
    s.do(AddOverlay(str(datadir / "run2.nii.gz")))
    yield s
    s.close()


def _keys(session):
    return {ly.name.split(".")[0]: ly.key for ly in session.state.layers}


def _output(session):
    layer = session.state.layers.find_by_source("mode:test_sum")
    return None if layer is None else float(session.store.get(layer.key).array.max())


def _mean_of(session, key):
    return float(np.asarray(session.store.ensure_ram(key)).mean())


# ---------------------------------------------------------------------------
# input is not visibility
# ---------------------------------------------------------------------------


def test_a_hidden_run_is_still_a_legal_input(session):
    """Unticking a run says something about the picture, not about the data."""
    run2 = _keys(session)["run2"]
    session.do(SetLayerVisible(run2, on=False))
    session.do(SetMode("test_sum"))
    assert session.mode.source_layer().key == run2
    assert _output(session) == pytest.approx(_mean_of(session, run2))


def test_hiding_the_input_after_a_fit_does_not_change_the_next_one(session):
    session.do(SetMode("test_sum"))
    before = _output(session)
    session.do(SetLayerVisible(session.mode.source_layer().key, on=False))
    assert session.refresh_mode() is not None
    assert _output(session) == pytest.approx(before)


def test_the_input_survives_being_buried_at_the_bottom(session, datadir):
    """Position is about drawing order; it must not re-point the fit."""
    run2 = _keys(session)["run2"]
    session.do(SetInput(run2))
    session.do(SetMode("test_sum"))
    session.state.layers.move(run2, 0)
    assert session.mode.source_layer().key == run2


# ---------------------------------------------------------------------------
# naming one
# ---------------------------------------------------------------------------


def test_the_default_input_is_the_topmost_candidate(session):
    session.do(SetMode("test_sum"))
    assert session.mode.source_layer().key == _keys(session)["run2"]


def test_naming_an_input_re_fits_on_it(session):
    keys = _keys(session)
    session.do(SetMode("test_sum"))
    session.do(SetInput(keys["run1"]))
    assert session.mode.source_layer().key == keys["run1"]
    assert _output(session) == pytest.approx(_mean_of(session, keys["run1"]))


def test_clearing_the_input_goes_back_to_the_default(session):
    keys = _keys(session)
    session.do(SetMode("test_sum"))
    session.do(SetInput(keys["run1"]))
    session.do(SetInput(""))
    assert session.mode.source_layer().key == keys["run2"]


def test_the_default_follows_the_stack_rather_than_sticking(session, datadir):
    """Unlike a selection: nobody named this one, so a newer run wins."""
    session.do(SetMode("test_sum"))
    assert session.mode.source_layer().name.startswith("run2")
    session.do(AddOverlay(str(datadir / "run1.nii.gz")))
    assert session.mode.source_layer().name.startswith("run1")


def test_a_named_input_does_not_move_when_a_run_is_loaded(session, datadir):
    keys = _keys(session)
    session.do(SetMode("test_sum"))
    session.do(SetInput(keys["run1"]))
    session.do(AddOverlay(str(datadir / "run2.nii.gz")))
    assert session.mode.source_layer().key == keys["run1"]


def test_naming_a_layer_that_is_not_a_run_falls_back(session):
    """A 3-D anatomical cannot be fitted; saying so beats fitting nothing."""
    session.do(SetMode("test_sum"))
    session.do(SetInput(_keys(session)["anat"]))
    assert session.mode.source_layer().key == _keys(session)["run2"]


def test_naming_a_layer_that_does_not_exist_is_an_error(session):
    with pytest.raises(KeyError):
        session.do(SetInput("nope"))


def test_the_input_replays_through_a_script(session, datadir):
    keys = _keys(session)
    session.do(SetMode("test_sum"))
    session.do(SetInput(keys["run1"]))
    assert f"SET_INPUT {keys['run1']}" in session.to_script()


# ---------------------------------------------------------------------------
# input is not selection
# ---------------------------------------------------------------------------


def test_selecting_a_layer_to_adjust_it_does_not_re_point_the_fit(session):
    """Changing an anatomical's colour scale must not move the GLM onto it."""
    keys = _keys(session)
    session.do(SetMode("test_sum"))
    before = session.mode.source_layer().key
    session.state.selected = keys["anat"]
    assert session.mode.source_layer().key == before


# ---------------------------------------------------------------------------
# what installing a result hides
# ---------------------------------------------------------------------------


def test_installing_a_result_hides_the_run_it_consumed(session):
    """A 4-D run drawn under its own map is noise over the anatomy."""
    run2 = _keys(session)["run2"]
    session.do(SetMode("test_sum"))
    assert session.state.layers.get(run2).visible is False


def test_installing_a_result_leaves_everything_else_alone(session, datadir):
    """The second map you loaded to compare against is the point of loading it."""
    session.do(AddOverlay(str(datadir / "stats.nii.gz")))
    stats = _keys(session)["stats"]
    run1 = _keys(session)["run1"]
    session.do(SetMode("test_sum"))
    assert session.state.layers.get(stats).visible is True
    assert session.state.layers.get(run1).visible is True


def test_a_run_switched_back_on_stays_on_across_recomputes(session):
    run2 = _keys(session)["run2"]
    session.do(SetMode("test_sum"))
    session.do(SetLayerVisible(run2, on=True))
    session.mode.invalidate()
    session.refresh_mode()
    assert session.state.layers.get(run2).visible is True


# ---------------------------------------------------------------------------
# the data stays loaded
# ---------------------------------------------------------------------------


def test_the_named_input_is_not_freed_while_the_mode_holds_it(session):
    """forget() on the input is what turns a re-prepare into a stale map."""
    run1 = _keys(session)["run1"]
    session.do(SetMode("test_sum"))
    session.do(SetInput(run1))
    session.store.ensure_ram(run1)
    session.forget(run1)
    assert run1 in session.store.keys()


def test_a_mode_that_reads_nothing_claims_no_input(session):
    assert session.mode.name == "plain"
    assert session.mode.source_layer() is None
    assert session.input_candidates() == []
