"""Design regressors in a graph: from a stats bucket to the column it describes.

Every failure worth pinning here draws something plausible. A regressor from
the wrong run, a column from a contrast, a run slice read off a censored xmat --
each would put a smooth, convolved curve in the graph, with nothing on screen to
say its events are in the wrong place. So the tests check that the chain
refuses rather than guesses.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.design import (
    Pin,
    fit_length,
    load_design,
    parse_provenance,
    run_of,
)

CPU = torch.device("cpu")


def write_xmat(path, columns, labels, groups, run_starts, good_list=None):
    """An AFNI xmat with the attributes the viewer reads."""
    n_rows, n_cols = columns.shape
    with open(path, "w") as f:
        f.write("# <matrix\n")
        f.write(f'#  ni_type = "{n_cols}*double"\n')
        f.write(f'#  ni_dimen = "{n_rows}"\n')
        f.write(f'#  ColumnLabels = "{" ; ".join(labels)}"\n')
        f.write(f'#  ColumnGroups = "{",".join(str(g) for g in groups)}"\n')
        f.write('#  RowTR = "2.0"\n')
        if good_list is not None:
            f.write(f'#  GoodList = "{",".join(str(g) for g in good_list)}"\n')
        f.write(f'#  RunStart = "{",".join(str(s) for s in run_starts)}"\n')
        f.write("# >\n")
        for row in columns:
            f.write(" ".join(f"{v:.10g}" for v in row) + "\n")
    return path


def two_run_design(tmp_path, lengths=(30, 40)):
    """Drift per run, one motion column, two stimuli whose values say their row."""
    total = sum(lengths)
    rows = np.arange(total, dtype=float)
    columns = np.stack(
        [
            np.r_[np.ones(lengths[0]), np.zeros(lengths[1])],
            np.r_[np.zeros(lengths[0]), np.ones(lengths[1])],
            np.sin(rows),
            rows,  # faces: value == row, so a slice shows where it came from
            -rows,  # houses
        ],
        axis=1,
    )
    return write_xmat(
        tmp_path / "design.xmat.1D",
        columns,
        labels=["Run#1Pol#0", "Run#2Pol#0", "roll", "faces#0", "houses#0"],
        groups=[-1, -1, 0, 1, 2],
        run_starts=[0, lengths[0]],
    )


# ---------------------------------------------------------------------------
# the design matrix
# ---------------------------------------------------------------------------


def test_a_run_is_the_rows_between_its_start_and_the_next(tmp_path):
    design = load_design(two_run_design(tmp_path))
    assert design.n_runs == 2
    assert design.run_length(1) == 30 and design.run_length(2) == 40
    assert design.column(1, 3)[0] == 0 and design.column(1, 3)[-1] == 29
    assert design.column(2, 3)[0] == 30 and design.column(2, 3)[-1] == 69


def test_the_menu_lists_stimuli_before_nuisance_before_drift(tmp_path):
    design = load_design(two_run_design(tmp_path))
    assert design.display_order() == [3, 4, 2, 0, 1]


def test_a_coef_or_tstat_names_its_one_column(tmp_path):
    design = load_design(two_run_design(tmp_path))
    assert design.column_for_brick("faces#0_Coef") == 3
    assert design.column_for_brick("houses#0_Tstat") == 4


@pytest.mark.parametrize("label", ["Full_Fstat", "faces_Fstat", "faces-houses_GLT#0_Coef"])
def test_a_statistic_over_several_columns_names_none(tmp_path, label):
    """An F or a contrast is not about one regressor; drawing one would misdescribe it."""
    design = load_design(two_run_design(tmp_path))
    assert design.column_for_brick(label) is None


def test_a_label_that_repeats_across_runs_names_no_column(tmp_path):
    columns = np.zeros((10, 2))
    path = write_xmat(
        tmp_path / "X.xmat.1D",
        columns,
        labels=["motion[0]", "motion[0]"],
        groups=[0, 0],
        run_starts=[0, 5],
    )
    assert load_design(path).column_for_brick("motion[0]_Coef") is None


def test_a_censored_design_refuses_to_be_cut_into_runs(tmp_path):
    """RunStart counts the full timeline; the rows of a censored xmat do not."""
    path = write_xmat(
        tmp_path / "X.xmat.1D",
        np.zeros((8, 1)),
        labels=["a#0"],
        groups=[1],
        run_starts=[0, 5],
        good_list=[0, 1, 2, 3, 5, 6, 7, 8],  # 9 TRs, one censored
    )
    with pytest.raises(ValueError, match="censored"):
        load_design(path)


def test_a_plain_1d_file_is_one_run_of_every_column(tmp_path):
    path = tmp_path / "motion.1D"
    np.savetxt(path, np.arange(24, dtype=float).reshape(12, 2))
    design = load_design(path)
    assert design.n_runs == 1 and design.n_columns == 2
    assert design.labels == ("motion#0", "motion#1")


def test_an_edited_design_is_read_again(tmp_path):
    path = two_run_design(tmp_path)
    assert load_design(path).run_length(1) == 30
    two_run_design(tmp_path, lengths=(20, 50))
    import os

    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert load_design(path).run_length(1) == 20


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def test_spec_names_the_xmat_ffs_reml_compiles_beside_it(tmp_path):
    history = (
        "[u@h: Thu Sep 17 12:39:38 2026] ffs_util_autobox -input a.nii.gz -prefix b.nii.gz\n"
        "[u@h: Thu Sep 17 12:45:52 2026] ffs_reml -input r1.nii.zst r2.nii.zst "
        "-spec designs/task.toml -Rbuck stats.nii.gz -tout"
    )
    prov = parse_provenance(history, stats_name="stats.nii.gz", base=tmp_path)
    assert prov is not None
    assert prov.design == str(tmp_path / "designs" / "task.xmat.1D")
    assert prov.inputs == (str(tmp_path / "r1.nii.zst"), str(tmp_path / "r2.nii.zst"))


def test_xmat_overrides_the_spec_derived_path(tmp_path):
    history = "ffs_reml -input r1.nii -spec task.toml -xmat /abs/X.xmat.1D -Rbuck s"
    assert parse_provenance(history, base=tmp_path).design == "/abs/X.xmat.1D"


def test_afni_programs_name_their_matrix_too(tmp_path):
    decon = "3dDeconvolve -input 'r1+orig.HEAD r2+orig.HEAD[2..$]' -x1D X.xmat.1D -bucket s"
    prov = parse_provenance(decon, base=tmp_path)
    assert prov.design == str(tmp_path / "X.xmat.1D")
    assert prov.inputs == (str(tmp_path / "r1+orig.HEAD"), str(tmp_path / "r2+orig.HEAD"))
    remlfit = "/opt/afni/3dREMLfit -matrix X.xmat.1D -input all_runs.nii -Rbuck s"
    assert parse_provenance(remlfit, base=tmp_path).design == str(tmp_path / "X.xmat.1D")


def test_either_flag_spelling_is_read():
    """FfsArgumentParser takes both, so a history can hold either."""
    prov = parse_provenance("ffs_reml -input r.nii -matrix X.1D -do-blur 3", base=None)
    assert prov is not None and prov.inputs[0].endswith("r.nii")


def test_the_fit_that_wrote_this_bucket_wins(tmp_path):
    """Two fits into one history (e.g. ols then reml); pick the one naming the file."""
    history = (
        "ffs_reml -input a.nii -matrix A.xmat.1D -Rbuck statsA.nii.gz\n"
        "ffs_reml -input b.nii -matrix B.xmat.1D -Rbuck other.nii.gz"
    )
    prov = parse_provenance(history, stats_name="statsA.nii.gz", base=tmp_path)
    assert prov.design.endswith("A.xmat.1D")


def test_no_glm_in_the_history_means_no_design():
    assert parse_provenance("ffs_nwarp -source a -prefix b") is None
    assert parse_provenance("ffs_reml -input a.nii -events e.tsv -Rbuck s") is None


def test_a_run_is_found_by_path_and_by_a_unique_name(tmp_path):
    history = "ffs_reml -input s1/run.nii s2/run.nii s2/run2.nii -matrix X.1D"
    prov = parse_provenance(history, base=tmp_path)
    assert run_of(prov, tmp_path / "s2" / "run.nii") == 2
    assert run_of(prov, "/elsewhere/run2.nii") == 3
    # The same name twice says nothing about which one this is.
    assert run_of(prov, "/elsewhere/run.nii") is None


# ---------------------------------------------------------------------------
# pins
# ---------------------------------------------------------------------------


def test_a_pin_survives_a_colon_in_its_path():
    pin = Pin("/data/C:odd/X.xmat.1D", run=3, col=31)
    assert Pin.decode(pin.encode()) == pin


def test_a_short_regressor_pads_with_its_baseline_and_a_long_one_is_cut():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert fit_length(values, 5).tolist() == [1.0, 2.0, 3.0, 0.0, 0.0]
    assert fit_length(values, 2).tolist() == [1.0, 2.0]
    assert fit_length(values, 0) is values


def test_pins_round_trip_through_a_script():
    from fastfuncstuff.viewer.commands import CommandBus, parse_line
    from fastfuncstuff.viewer.state import ViewerState
    from fastfuncstuff.viewer.vocab import OpenView, SetViewRegressors, install

    bus = install(CommandBus(ViewerState()))
    bus.dispatch(OpenView("G1", "graph", "axial"))
    specs = "2:3:/a b/X.xmat.1D,1:0:/c/Y.1D"
    line = SetViewRegressors("G1", specs).to_line()
    cmd = parse_line(line)
    assert cmd is not None
    bus.dispatch(cmd)
    assert bus.state.viewports.get("G1").regressors == ("2:3:/a b/X.xmat.1D", "1:0:/c/Y.1D")


# ---------------------------------------------------------------------------
# the session: which column follows the stats overlay
# ---------------------------------------------------------------------------


@pytest.fixture
def glm_dir(tmp_path, monkeypatch):
    """Two runs, their xmat, and a bucket whose history says it was fit from them."""
    from fastfuncstuff.io import afni

    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    rng = np.random.default_rng(0)
    for name, nt in (("run-01.nii.gz", 30), ("run-02.nii.gz", 40), ("other.nii.gz", 40)):
        afni.save_nifti(
            rng.normal(size=(4, 4, 3, nt)).astype(np.float32), tmp_path / name, affine=aff, tr=2.0
        )
    two_run_design(tmp_path)
    (tmp_path / "design.toml").write_text("")
    monkeypatch.setattr(
        afni,
        "_history_commandline",
        lambda: "ffs_reml -input run-01.nii.gz run-02.nii.gz -spec design.toml -Rbuck stats.nii.gz",
    )
    afni.save_nifti(
        rng.normal(size=(4, 4, 3, 5)).astype(np.float32),
        tmp_path / "stats.nii.gz",
        affine=aff,
        brick_labels=[
            "Full_Fstat",
            "faces#0_Coef",
            "faces#0_Tstat",
            "houses#0_Coef",
            "houses#0_Tstat",
        ],
    )
    return tmp_path


@pytest.fixture
def session():
    from fastfuncstuff.viewer.session import ViewerSession

    s = ViewerSession(device=CPU)
    yield s
    s.close()


def _load(session, d, *names):
    from fastfuncstuff.viewer.vocab import AddLayer

    for name in names:
        session.do(AddLayer(str(d / name), name.split(".")[0]))


def _show(session, key, index):
    from fastfuncstuff.viewer.vocab import SetVolume

    session.do(SetVolume(key, index))


def test_a_coef_on_screen_draws_its_column_for_the_graphed_run(glm_dir, session):
    _load(session, glm_dir, "run-02.nii.gz", "stats.nii.gz")
    _show(session, "stats", 3)  # houses#0_Coef
    design, run, col = session.auto_regressor()
    assert (run, col) == (2, 4)
    assert design.column(run, col)[0] == -30.0


def test_the_column_follows_the_sub_brick(glm_dir, session):
    _load(session, glm_dir, "run-01.nii.gz", "stats.nii.gz")
    _show(session, "stats", 2)
    assert session.auto_regressor()[1:] == (1, 3)
    _show(session, "stats", 4)
    assert session.auto_regressor()[1:] == (1, 4)


def test_an_f_stat_on_screen_draws_nothing(glm_dir, session):
    _load(session, glm_dir, "run-01.nii.gz", "stats.nii.gz")
    _show(session, "stats", 0)
    assert session.auto_regressor() is None


def test_several_runs_graphed_use_the_first_that_was_fit(glm_dir, session):
    _load(session, glm_dir, "other.nii.gz", "run-02.nii.gz", "run-01.nii.gz", "stats.nii.gz")
    _show(session, "stats", 1)
    assert session.auto_regressor()[1] == 2


def test_a_run_the_fit_never_saw_draws_nothing(glm_dir, session):
    _load(session, glm_dir, "other.nii.gz", "stats.nii.gz")
    _show(session, "stats", 1)
    assert session.auto_regressor() is None


def test_a_run_whose_length_disagrees_draws_nothing(glm_dir, session, monkeypatch):
    """A trimmed run (-drop_first) would put every event early."""
    two_run_design(glm_dir, lengths=(28, 42))
    import os

    st = os.stat(glm_dir / "design.xmat.1D")
    os.utime(glm_dir / "design.xmat.1D", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    _load(session, glm_dir, "run-01.nii.gz", "stats.nii.gz")
    _show(session, "stats", 1)
    assert session.auto_regressor() is None


def test_pins_and_the_automatic_line_are_listed_together(glm_dir, session):
    from fastfuncstuff.viewer.state import Plane
    from fastfuncstuff.viewer.viewports import ViewKind
    from fastfuncstuff.viewer.vocab import SetViewRegressors

    _load(session, glm_dir, "run-01.nii.gz", "stats.nii.gz")
    _show(session, "stats", 1)
    vid = session.open_view(ViewKind.GRAPH, Plane.AXIAL)
    xmat = str((glm_dir / "design.xmat.1D").resolve())
    session.do(SetViewRegressors(vid, f"2:4:{xmat}"))
    lines = session.regressor_series(session.state.viewports.get(vid))
    assert [r.legend for r in lines] == ["r1·faces#0 (auto)", "r2·houses#0"]
    assert lines[1].values.size == 40


# ---------------------------------------------------------------------------
# the graph window
# ---------------------------------------------------------------------------


@pytest.fixture
def graph(glm_dir, qapp):
    """A viewer on run 1 with the stats bucket over it and a graph open."""
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.gridgraph import GraphWindow
    from fastfuncstuff.viewer.ui.window import ViewerWindow
    from fastfuncstuff.viewer.vocab import SetOverlay, SetUnderlay, SetVolume

    session = ViewerSession(device=CPU)
    w = ViewerWindow(session)
    w.read_directory(glm_dir)
    w.refresh(session.do(SetUnderlay(str(glm_dir / "run-01.nii.gz"))))
    w.refresh(session.do(SetOverlay(str(glm_dir / "stats.nii.gz"))))
    run, stats = (ly.key for ly in session.state.layers)
    session.store.ensure_ram(run)
    w.refresh(session.do(SetVolume(stats, 1)))  # faces#0_Coef
    w.show()
    qapp.processEvents()
    w._new_graph()
    qapp.processEvents()
    g = next(x for x in w.manager.windows.values() if isinstance(x, GraphWindow))
    g.refresh()
    yield w, g, stats
    w.close()


@pytest.fixture(scope="module")
def qapp():
    import os

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _drawn(g, ident):
    return [v for cell in g.graph._cells for k, v in cell.traces if k == ident]


def test_the_graph_draws_the_shown_coef_column_undetrended(graph, qapp):
    from fastfuncstuff.viewer.design import AUTO_IDENT
    from fastfuncstuff.viewer.vocab import SetViewDetrend

    w, g, _ = graph
    assert AUTO_IDENT in g._trace_checks
    w.refresh(w.session.do(SetViewDetrend(g.vid, 1)))
    qapp.processEvents()
    curves = _drawn(g, AUTO_IDENT)
    assert curves, "the automatic column must be drawn in every cell"
    # Column faces#0 holds its own row number, so run 1 is exactly 0..29.
    assert np.array_equal(curves[0], np.arange(30, dtype=np.float32))


def test_the_menus_start_on_the_automatic_column_and_pin_it(graph, qapp):
    w, g, _ = graph
    assert g.design_box.currentData().endswith("design.xmat.1D")
    assert (g.run_box.currentData(), g.column_box.currentData()) == (1, 3)
    g.pin_button.click()
    qapp.processEvents()
    vp = w.session.state.viewports.find(g.vid)
    assert len(vp.regressors) == 1 and vp.regressors[0].startswith("1:3:")
    assert g.pin_button.text() == "unpin"
    assert "SET_VIEW_REGRESSORS" in w.session.to_script()


def test_a_pin_stays_when_the_sub_brick_moves_on(graph, qapp):
    from fastfuncstuff.viewer.design import AUTO_IDENT
    from fastfuncstuff.viewer.vocab import SetVolume

    w, g, stats = graph
    g.pin_button.click()
    w.refresh(w.session.do(SetVolume(stats, 0)))  # Full_Fstat: no column
    qapp.processEvents()
    assert AUTO_IDENT not in g._trace_checks
    pinned = [k for k in g._trace_checks if k.startswith("design:1:3:")]
    assert pinned and _drawn(g, pinned[0])


def test_a_pinned_line_unticks_like_any_other(graph, qapp):
    from fastfuncstuff.viewer.design import AUTO_IDENT

    w, g, _ = graph
    g._trace_checks[AUTO_IDENT].click()
    qapp.processEvents()
    assert AUTO_IDENT in w.session.state.viewports.find(g.vid).hidden
    assert not _drawn(g, AUTO_IDENT)
    # Unticking a design line must not narrow the window's layer selection.
    assert w.session.state.viewports.find(g.vid).traces == ()
