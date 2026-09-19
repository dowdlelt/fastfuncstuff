"""InstaGLM as a mode: the fit/view split, and the lines it hands the graphs.

The engine's own arithmetic is pinned in ``test_instaglm.py``. What matters here
is the wiring, and one property above all: **changing what you look at must not
refit**. That is the whole ergonomic claim of the mode -- pick a different beta,
a different column, a different unit, and the map redraws now -- and it is the
kind of thing that regresses silently, because a mode that refits on every view
change still shows the right answer, just slowly enough to stop being a teaching
tool.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.modes.base import OverlayKind
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.vocab import SetMode, SetOverlay, SetUnderlay

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")

TR = 2.0
N_TIME = 60
SOURCE = "mode:instaglm"
#: A voxel inside the bright box, so it survives the automask.
INSIDE = (4, 4, 3)


def _write(d, name, data, tr=0.0):
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    aff[:3, 3] = (-15.0, -18.0, -12.0)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), aff)
    if tr:
        img.header["pixdim"][4] = tr
        img.header.set_xyzt_units("mm", "sec")
    p = d / name
    nib.save(img, str(p))
    return p


def _events(path, onsets_by_condition):
    lines = ["onset\tduration\ttrial_type"]
    for name, onsets in onsets_by_condition.items():
        lines += [f"{o:g}\t2\t{name}" for o in onsets]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


@pytest.fixture
def glm_session(tmp_path):
    """A run with a bright box -- so the automask finds a brain -- carrying a
    drift the model can be made to explain."""
    rng = np.random.default_rng(4)
    nx, ny, nz = 9, 10, 7
    data = rng.normal(5.0, 0.5, (nx, ny, nz, N_TIME)).astype(np.float32)
    drift = np.linspace(0.0, 40.0, N_TIME, dtype=np.float32)
    data[2:7, 2:8, 1:6, :] += 1000.0 + drift

    _write(tmp_path, "anat.nii.gz", rng.random((nx, ny, nz)) * 100)
    _write(tmp_path, "bold.nii.gz", data, tr=TR)

    s = ViewerSession(device=CPU)
    s.do(SetUnderlay(str(tmp_path / "anat.nii.gz")))
    s.do(SetOverlay(str(tmp_path / "bold.nii.gz")))
    s.store.ensure_ram(s.state.layers.keys[1])
    s.events = _events(tmp_path / "sub-01_events.tsv", {"faces": [4, 30, 56, 82]})
    s.tmp = tmp_path
    yield s
    s.close()


def _enter(session, **params):
    session.do(SetMode("instaglm"))
    for name, value in params.items():
        session.set_mode_param(name, str(value))
    return session.mode


def _map(session):
    layer = session.state.layers.find_by_source(SOURCE)
    assert layer is not None, "the mode installed no overlay"
    return session.store.get(layer.key).array[..., 0]


# -- entering ---------------------------------------------------------------


def test_entering_fits_the_drift_only_model_without_an_events_file(glm_session):
    """Useful on its own: with polort 2 and nothing else, the map is how much
    of each voxel is drift, which is the first thing worth knowing about a run."""
    mode = _enter(glm_session)
    assert mode._fit is not None
    assert mode._fit.model.labels == ("Pol#0", "Pol#1", "Pol#2")
    assert _map(glm_session).shape == (9, 10, 7)


def test_the_planted_drift_is_explained_and_stepping_polort_up_explains_more(glm_session):
    mode = _enter(glm_session, show="R2", polort=0)
    flat = float(mode._fit.volume("R2")[INSIDE])
    glm_session.set_mode_param("polort", "1")
    linear = float(glm_session.mode._fit.volume("R2")[INSIDE])
    assert flat < 0.01
    assert linear > 0.95


def test_a_run_without_a_tr_still_fits_drift_but_refuses_events(tmp_path):
    """Refusing everything because a header is thin would block the first
    useful thing the mode does. Onsets are in seconds; polynomials are not."""
    rng = np.random.default_rng(1)
    data = rng.normal(500.0, 1.0, (8, 8, 6, N_TIME)).astype(np.float32)
    _write(tmp_path, "anat.nii.gz", rng.random((8, 8, 6)) * 100)
    path = _write(tmp_path, "bold.nii.gz", data)
    img = nib.load(str(path))
    img.header["pixdim"][4] = 0.0
    nib.save(img, str(path))

    s = ViewerSession(device=CPU)
    s.do(SetUnderlay(str(tmp_path / "anat.nii.gz")))
    s.do(SetOverlay(str(tmp_path / "bold.nii.gz")))
    s.do(SetMode("instaglm"))
    assert s.mode._fit is not None
    assert s.state.layers.find_by_source(SOURCE) is not None

    s.set_mode_param("events", _events(tmp_path / "e_events.tsv", {"faces": [4, 30]}))
    assert s.mode._fit is None
    assert "TR" in s.mode.status()
    s.close()


# -- the fit / view split ---------------------------------------------------


def test_changing_the_map_redraws_without_refitting(glm_session):
    """The mode's whole ergonomic claim. A view change must reuse the Fit."""
    mode = _enter(glm_session, events=glm_session.events)
    before = mode._fit
    beta = _map(glm_session).copy()

    glm_session.set_mode_param("show", "t")
    assert glm_session.mode._fit is before, "picking a different map refit the model"
    assert not np.allclose(_map(glm_session), beta)

    glm_session.set_mode_param("column", "Pol#1")
    assert glm_session.mode._fit is before, "picking a different column refit the model"

    glm_session.set_mode_param("psc", "per unit")
    assert glm_session.mode._fit is before, "changing units refit the model"


def test_changing_the_design_does_refit(glm_session):
    mode = _enter(glm_session, events=glm_session.events)
    before = mode._fit
    glm_session.set_mode_param("polort", "5")
    assert glm_session.mode._fit is not before
    assert glm_session.mode._fit.model.n_columns == before.model.n_columns + 3


def test_stepping_polort_does_not_reconvolve_the_events(glm_session):
    """Convolving events at microtime costs sixteen times the fit it feeds, and
    "step up through the polynomials and watch the fit improve" is the single
    gesture this mode exists for. It must not pay for a convolution per step."""
    mode = _enter(glm_session, events=glm_session.events)
    block = mode._task
    assert block is not None

    for changed in ("polort", "ort_deriv", "pcs"):
        glm_session.set_mode_param(changed, "3" if changed == "polort" else "1")
        assert glm_session.mode._task is block, f"{changed} reconvolved the events"


def test_moving_the_hrf_does_reconvolve_the_events(glm_session):
    mode = _enter(glm_session, events=glm_session.events, basis="custom", peak=5.0)
    block = mode._task
    glm_session.set_mode_param("peak", "9.0")
    assert glm_session.mode._task is not block


def test_the_two_psc_conventions_differ_by_the_regressors_swing(glm_session):
    mode = _enter(glm_session, events=glm_session.events, show="beta", column="faces")
    swing = _map(glm_session).copy()
    glm_session.set_mode_param("psc", "per unit")
    per_unit = _map(glm_session)
    scale = mode._fit.model.columns[0].swing
    assert np.allclose(swing[INSIDE], per_unit[INSIDE] * scale, rtol=1e-4)


# -- the model --------------------------------------------------------------


def test_an_events_file_adds_pickable_condition_columns(glm_session):
    mode = _enter(glm_session, events=glm_session.events)
    assert mode._fit.model.labels[0] == "faces"
    choices = next(c for c in mode.controls() if c.name == "column").choices
    assert choices[0] == "faces" and "Pol#0" in choices


def test_a_derivative_basis_adds_a_column_per_condition(glm_session):
    mode = _enter(glm_session, events=glm_session.events, basis="spmg2")
    assert mode._fit.model.labels[:2] == ("faces", "faces'")


def test_an_ortvec_is_fitted_and_its_derivative_is_optional(glm_session):
    path = glm_session.tmp / "motion.1D"
    rng = np.random.default_rng(2)
    np.savetxt(path, rng.normal(size=(N_TIME, 3)))

    mode = _enter(glm_session, ortvec=str(path))
    assert sum(c.group == "ort" for c in mode._fit.model.columns) == 3
    glm_session.set_mode_param("ort_deriv", "1")
    labels = glm_session.mode._fit.model.labels
    assert sum(c.group == "ort" for c in glm_session.mode._fit.model.columns) == 6
    assert any(lab.endswith("'") for lab in labels)


def test_an_ortvec_of_the_wrong_length_is_reported_not_raised(glm_session):
    """A file with the wrong number of rows is a normal mistake, and the status
    bar is where it belongs -- not a traceback out of a worker thread."""
    path = glm_session.tmp / "wrong.1D"
    np.savetxt(path, np.zeros((N_TIME + 7, 2)))
    mode = _enter(glm_session, ortvec=str(path))
    assert mode._fit is None
    assert "volumes" in mode.status()


def test_noise_pcs_add_columns_chosen_against_the_model_without_them(glm_session):
    mode = _enter(glm_session, events=glm_session.events, pcs=2)
    labels = mode._fit.model.labels
    assert "PC#0" in labels and "PC#1" in labels
    assert sum(c.group == "pc" for c in mode._fit.model.columns) == 2


def test_a_collinear_design_still_draws_and_says_so(glm_session):
    """polort 0 beside a constant ortvec is one click away."""
    path = glm_session.tmp / "flat.1D"
    np.savetxt(path, np.ones((N_TIME, 1)))
    mode = _enter(glm_session, ortvec=str(path), polort=0)
    assert "collinear" in mode.status()
    assert np.isfinite(_map(glm_session)).all()


# -- presentation -----------------------------------------------------------


def test_the_overlay_kind_follows_the_map_being_shown(glm_session):
    """A t map and an R2 map are not the same question, so the threshold
    control must not label them the same way."""
    mode = _enter(glm_session, events=glm_session.events, show="beta")
    assert mode.overlay_kind is OverlayKind.VALUE
    glm_session.set_mode_param("show", "t")
    assert glm_session.mode.overlay_kind is OverlayKind.STATISTIC
    glm_session.set_mode_param("show", "task F")
    assert glm_session.mode.overlay_kind is OverlayKind.STATISTIC


def test_the_layer_name_says_which_column_and_which_map(glm_session):
    _enter(glm_session, events=glm_session.events, show="beta", column="faces")
    layer = glm_session.state.layers.find_by_source(SOURCE)
    assert layer.name == "A_IGLM faces beta"
    glm_session.set_mode_param("show", "task R2")
    assert glm_session.state.layers.find_by_source(SOURCE).name == "A_IGLM task R2"


def test_the_map_goes_on_top_and_hides_the_run_under_it(glm_session):
    bold = glm_session.state.layers.keys[1]
    _enter(glm_session, events=glm_session.events)
    stack = glm_session.state.layers
    assert stack.layers[-1].source == SOURCE
    assert stack.find(bold) is not None and not stack.get(bold).visible


def test_leaving_the_mode_keeps_the_map_and_frees_the_gathered_run(glm_session):
    mode = _enter(glm_session, events=glm_session.events)
    glm_session.do(SetMode("plain"))
    assert glm_session.state.layers.find_by_source(SOURCE) is not None
    assert mode._prepared is None, "a gigabyte must not survive switching away"


# -- graph lines ------------------------------------------------------------


def test_every_line_is_contributed_at_a_voxel_in_the_mask(glm_session):
    mode = _enter(glm_session, events=glm_session.events)
    keys = {t.ident for t in mode.series(INSIDE)}
    assert keys == {"data", "signal", "fit", "resid", "column"}


def test_the_lines_reconstruct_the_measurement(glm_session):
    mode = _enter(glm_session, events=glm_session.events)
    lines = {t.ident: t.values for t in mode.series(INSIDE)}
    assert np.allclose(lines["signal"] - lines["fit"], lines["resid"], atol=1e-2)


def test_the_selected_column_line_follows_the_column_picker(glm_session):
    mode = _enter(glm_session, events=glm_session.events, column="faces")
    faces = next(t for t in mode.series(INSIDE) if t.ident == "column")
    assert faces.legend == "iglm faces"
    glm_session.set_mode_param("column", "Pol#1")
    drift = next(t for t in glm_session.mode.series(INSIDE) if t.ident == "column")
    assert drift.legend == "iglm Pol#1"
    assert not np.allclose(faces.values, drift.values)


def test_a_voxel_outside_the_mask_contributes_nothing(glm_session):
    mode = _enter(glm_session, events=glm_session.events)
    assert mode.series((0, 0, 0)) == []


# -- the HRF panel ----------------------------------------------------------


def test_the_hrf_panel_draws_the_curve_that_was_actually_fitted(glm_session):
    mode = _enter(glm_session, events=glm_session.events, basis="spmg2")
    traces = mode.panels()["hrf"]
    assert [t.label for t in traces] == ["HRF", "HRF'"]
    assert traces[0].x_label == "s"
    assert float(traces[0].values.max()) == pytest.approx(1.0, rel=1e-5)


def test_dragging_the_hrf_peak_moves_the_curve_and_the_map(glm_session):
    """The loop the mode exists to close: the shape you can see and the map it
    is redrawing have to move together."""
    _enter(glm_session, events=glm_session.events, basis="custom", peak=4.0, show="R2")
    early = glm_session.mode.panels()["hrf"][0].values
    early_map = _map(glm_session).copy()

    glm_session.set_mode_param("peak", "10.0")
    late = glm_session.mode.panels()["hrf"][0].values
    assert int(late.argmax()) > int(early.argmax())
    assert not np.allclose(_map(glm_session), early_map)


def test_the_panel_is_named_so_a_window_opens_for_it(glm_session):
    assert _enter(glm_session).panel_names() == ("hrf",)
