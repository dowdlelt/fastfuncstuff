"""Smoke tests for BSDS figures (headless Agg backend)."""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")

from fastfuncstuff.dynamics.bsds.model import fit_bsds  # noqa: E402
from fastfuncstuff.dynamics.graph import state_graph_metrics  # noqa: E402
from fastfuncstuff.dynamics.plots import (  # noqa: E402
    analysis_report,
    qc_report,
    save_publication_figures,
    state_colors,
)
from fastfuncstuff.dynamics.states import compute_state_stats  # noqa: E402
from fastfuncstuff.dynamics.switching import compute_switch_stats  # noqa: E402

sys.path.insert(0, "tests")
from test_bsds_model import _simulate  # noqa: E402


def _fit(k=3, d=6, seed=1):
    sessions, _, _, _ = _simulate(k=k, d=d, n_sessions=2, seed=seed)
    model = fit_bsds(sessions, n_states=k, max_ldim=3, n_init=2, n_init_iter=10, n_iter=50)
    return model, compute_state_stats(model, tr=0.72)


def test_state_colors_fixed_order_and_extension():
    assert state_colors(3) == ["#2a78d6", "#1baf7a", "#eda100"]
    assert len(state_colors(8)) == 8
    # Beyond 8 we switch to a golden-angle palette: K distinct colors, no cycling
    # (state 0 must not equal state 8), no warning.
    cols = state_colors(26)
    assert len(cols) == 26
    assert len(set(cols)) == 26  # all distinct — the whole point
    assert cols[0] != cols[8]

    from fastfuncstuff.dynamics.plots import condition_colors, golden_palette

    # 48 conditions (24 tasks + 24 instructions) must all be distinct.
    conds = condition_colors(48)
    assert len(conds) == 48 and len(set(conds)) == 48
    # Deterministic and valid hex.
    assert golden_palette(5) == golden_palette(5)
    assert all(c.startswith("#") and len(c) == 7 for c in golden_palette(30))


def test_qc_report_writes_png(tmp_path):
    model, stats = _fit()
    out = tmp_path / "sub_qc.png"
    qc_report(model, stats, out, tr=0.72)
    assert out.exists() and out.stat().st_size > 5000  # a real image, not empty


def test_publication_figures_written(tmp_path):
    model, stats = _fit()
    stem = str(tmp_path / "sub")
    gm = state_graph_metrics(stats.state_fc)
    ss = compute_switch_stats(model, tr=0.72)
    written = save_publication_figures(
        model, stats, stem, fmt="svg", tr=0.72, graph_metrics=gm, switch_stats=ss
    )
    # Each figure saved as both svg and png; timecourse per run.
    assert any(w.endswith("_transition.svg") for w in written)
    assert any(w.endswith("_state_fc.png") for w in written)
    assert any(w.endswith("_graph_metrics.svg") for w in written)
    assert any(w.endswith("_switching.png") for w in written)
    assert sum(w.endswith(".svg") for w in written) == sum(w.endswith(".png") for w in written)
    for w in written:
        import os

        assert os.path.getsize(w) > 0


def test_analysis_report_writes_png(tmp_path):
    model, stats = _fit()
    gm = state_graph_metrics(stats.state_fc)
    ss = compute_switch_stats(model, tr=0.72)
    out = tmp_path / "sub_analysis.png"
    analysis_report(model, gm, ss, out, tr=0.72)
    assert out.exists() and out.stat().st_size > 5000


def test_plot_restart_scores_renders_and_handles_empty():
    import matplotlib.pyplot as plt

    from fastfuncstuff.dynamics.plots import plot_restart_scores

    model, _ = _fit()
    # n_init=2 restarts -> two recorded scores, ranked-on free energy.
    assert len(model.restart_scores) == 2
    fig, ax = plt.subplots()
    plot_restart_scores(model, ax)
    assert ax.collections and ax.lines  # scatter + running-best step drawn
    plt.close(fig)

    # Models loaded from disk without the field must degrade gracefully.
    class _Empty:
        restart_scores: list[float] = []

    fig, ax = plt.subplots()
    plot_restart_scores(_Empty(), ax)
    plt.close(fig)


def test_restart_scores_round_trip(tmp_path):
    from fastfuncstuff.dynamics.bsds.model import load_bsds_model, save_bsds_model

    model, _ = _fit()
    p = str(tmp_path / "m.npz")
    save_bsds_model(model, p)
    reloaded = load_bsds_model(p)
    assert reloaded.restart_scores == model.restart_scores


def test_display_helpers_do_not_leak_open_figures(tmp_path):
    # Figure-returning helpers must close the pyplot-managed figure before
    # returning it, or %matplotlib inline renders it twice (repr + post-cell
    # auto-flush) — the "two images per cell" / slow-scroll bug. We assert the
    # returned Figure is not left in pyplot's registry (so inline can't
    # double-show it) while still being renderable.
    from io import BytesIO

    import matplotlib.pyplot as plt
    from matplotlib._pylab_helpers import Gcf

    from fastfuncstuff.dynamics.plots import plot_state_fc, plot_state_ribbons

    model, stats = _fit()
    for make in (lambda: plot_state_ribbons(model), lambda: plot_state_fc(stats)):
        plt.close("all")
        fig = make()
        assert fig not in [m.canvas.figure for m in Gcf.get_all_fig_managers()]
        buf = BytesIO()
        fig.savefig(buf, format="png")  # closed figure still renders once
        assert buf.getbuffer().nbytes > 1000
