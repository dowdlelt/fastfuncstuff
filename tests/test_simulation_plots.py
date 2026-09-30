"""The shared design-power figures render and save."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch

from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit, realize
from fastfuncstuff.simulation.plots import (
    plot_design,
    plot_design_comparison,
    plot_example_voxels,
    plot_power,
)
from fastfuncstuff.simulation.power import simulate_realizations_power


def test_figures_render(tmp_path):
    spec = ExperimentSpec(
        tr=2,
        units=[Unit.parse("A", "A:2", 8), Unit.parse("B", "B:2", 8)],
        isi=Interval.parse("exp:4,2,10"),
        post_fix=12,
    )
    reals = [realize(spec, s) for s in range(2)]
    noise = [
        {"label": "tSNR 40", "tsnr": 40.0, "phys_fraction": 0.5, "tau": 6.0},
        {"label": "tSNR 90", "tsnr": 90.0, "phys_fraction": 0.5, "tau": 6.0},
    ]
    contrasts = {"A": [1, 0], "A-B": [1, -1]}
    res = simulate_realizations_power(
        reals,
        2,
        contrasts,
        [1.0, 3.0],
        noise,
        beta_pattern=[1, 0],
        n_reps=20,
        device=torch.device("cpu"),
        progress=False,
    )
    labels = [n["label"] for n in noise]
    summary = {
        "facts": [("scan", "TR 2 s, 1 run"), ("model", "spmg1 " * 40)],  # wraps
        "notes": ["trimmed"],
        "header": ["noise", "A", "a long false-positive header"],
        "rows": [["tSNR 40", "1.00 [0.90-1.10]", "0.0010 / 0.0100"]],
        "footer": ["effect for 80% power"],
    }
    plot_power(res, labels, contrasts, [1, 0], effect=1.0, path=tmp_path / "p.png", summary=summary)
    plot_design(res, reals[0], 2, path=tmp_path / "d.png")
    plot_design_comparison({"x": res, "y": res}, "A", labels, path=tmp_path / "c.png")
    plot_example_voxels(reals[0], 2, noise, amplitude=[2.5, 1.0], path=tmp_path / "v.png")
    with pytest.raises(ValueError, match="2 noise levels"):
        plot_example_voxels(reals[0], 2, noise, amplitude=[1.0, 2.0, 3.0])
    arma = [{"label": "measured", "tsnr": 60.0, "arma": (0.0, 0.4)}]
    plot_example_voxels(reals[0], 2, arma, path=tmp_path / "v2.png")
    for name in ("p.png", "d.png", "c.png", "v.png", "v2.png"):
        assert (tmp_path / name).stat().st_size > 10_000


def test_design_spread_renders_for_one_and_several_conditions(tmp_path):
    from fastfuncstuff.simulation.plots import plot_design_spread
    from fastfuncstuff.simulation.power import realizations_design_quality

    noise = [{"label": "tSNR 50", "tsnr": 50.0, "phys_fraction": 0.5, "tau": 6.0}]
    for names in (["A"], ["A", "B", "C"]):
        spec = ExperimentSpec(
            tr=2,
            units=[Unit.parse(c, f"{c}:20", 2, "block") for c in names],
            isi=Interval.parse(10),
            post_fix=12,
        )
        reals = [realize(spec, s) for s in range(4)]
        q = realizations_design_quality(reals, 2, noise)
        out = tmp_path / f"s{len(names)}.png"
        plot_design_spread(q, reals, "tSNR 50", path=out)
        assert out.stat().st_size > 10_000


def test_tent_and_single_trial_figures_render(tmp_path):
    from fastfuncstuff.simulation.plots import plot_single_trials, plot_tent
    from fastfuncstuff.simulation.power import single_trial_example, tent_estimate

    spec = ExperimentSpec(
        tr=1.25,
        units=[Unit.parse("A", "A:0.5", 12), Unit.parse("B", "B:0.5", 12)],
        isi=Interval.parse("exp:4,2,10"),
        post_fix=15,
    )
    r = realize(spec, 0)
    noise = [{"label": "tSNR 80", "tsnr": 80.0, "phys_fraction": 0.5, "tau": 6.0}]
    t = tent_estimate(r, 1.25, noise, [1.0, 1.0])
    # the truth peaks at the planted amplitude; the knots start at onset
    assert t["truth"].max() == pytest.approx(1.0, abs=0.02) and t["knots"][0] == 0.0
    plot_tent(t, r.conditions, ["tSNR 80"], [1.0, 1.0], path=tmp_path / "t.png")
    ex = single_trial_example(r, 1.25, noise[0], sd_grid=(0.25, 1.0))
    assert ex["curve"]["lsa"][1] > ex["curve"]["lsa"][0]  # more variable trials: more reliable
    plot_single_trials(ex, r.conditions, 0.5, path=tmp_path / "s.png")
    for name in ("t.png", "s.png"):
        assert (tmp_path / name).stat().st_size > 10_000


def test_tstats_figure_shows_a_calibrated_null(tmp_path):
    from fastfuncstuff.simulation.plots import plot_tstats
    from fastfuncstuff.simulation.power import t_example

    spec = ExperimentSpec(
        tr=1.0,
        units=[Unit.parse("A", "A:0.5", 30), Unit.parse("B", "B:0.5", 30)],
        isi=Interval.parse("exp:4,2,10"),
        post_fix=15,
    )
    r = realize(spec, 0)
    noise = [{"label": "tSNR 60", "tsnr": 60.0, "phys_fraction": 0.5, "tau": 6.0}]
    out = t_example(r, 1.0, "A", [1, 0], noise, {"tSNR 60": 1.0}, n_reps=800)
    null_c, null_n = out["t"]["tSNR 60"]["null"]
    assert abs(null_c.std() - 1.0) < 0.1  # the corrected null is a unit t
    assert null_n.std() > 1.15 * null_c.std()  # the naive one is too wide
    plot_tstats(out, "A", ["tSNR 60"], 0.001, path=tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 10_000


def test_spectrum_puts_block_power_at_the_block_frequency(tmp_path):
    from fastfuncstuff.simulation.plots import plot_spectrum
    from fastfuncstuff.simulation.power import design_spectrum

    # A-B alternating 20 s blocks with no gap: the difference is a 40 s square wave
    spec = ExperimentSpec(
        tr=1.0,
        units=[Unit.parse("AB", "A:20:0, B:20:0", 8, "block")],
        isi=Interval.parse(0),
        initial_fix=10,
        post_fix=16,
    )
    r = realize(spec, 0)
    noise = {"label": "tSNR 60", "tsnr": 60.0, "phys_fraction": 0.5, "tau": 6.0}
    sp = design_spectrum(r, 1.0, {"A-B": [1, -1]}, noise)
    peak = sp["freq"][np.argmax(sp["power"]["A-B"])]
    assert peak == pytest.approx(1 / 40, abs=0.003)
    assert sp["removed"][0] == pytest.approx(1.0) and sp["removed"][-1] < 0.01
    assert sp["noise_psd"][0] == pytest.approx(1.0) and sp["noise_psd"][-1] < 0.5
    plot_spectrum(sp, ["A-B"], path=tmp_path / "s.png")
    assert (tmp_path / "s.png").stat().st_size > 10_000


def test_soa_sweep_and_liu_plane(tmp_path):
    from fastfuncstuff.simulation.plots import plot_liu, plot_soa
    from fastfuncstuff.simulation.power import RealizationScorer, soa_sweep

    spec = ExperimentSpec(
        tr=1.0,
        units=[Unit.parse("A", "A:1", 1), Unit.parse("B", "B:1", 1)],
        isi=Interval.parse("exp:4,1,12"),
        post_fix=15,
        scan_time=240,
    )
    r = realize(spec, 0)
    noise = [{"label": "t60", "tsnr": 60.0, "phys_fraction": 0.5, "tau": 6.0}]
    con = {"A": [1.0, 0.0], "A-B": [1.0, -1.0]}
    sc = RealizationScorer(1.0, con, noise)
    out = soa_sweep(r, 1.0, sc, "t60", 0.0, 15.0, n_soa=4, n_designs=1)
    assert out["this_soa"] == pytest.approx(5.0, abs=1.0)
    # random events are efficient estimators, long blocks detect: the Liu axes
    ev = out["families"]["jittered"][0][0]
    bl = out["blocks"][20.0][0]
    assert ev["xi"] > 3 * bl["xi"] and bl["liu_power"] > 2 * ev["liu_power"]
    plot_soa(out, con, [sc.score(r)], path=tmp_path / "s.png")
    plot_liu({"events": [(ev["xi"], ev["liu_power"])], "this design": [(0.2, 0.1)]}, 16, 2,
             path=tmp_path / "l.png")  # fmt: skip
    for name in ("s.png", "l.png"):
        assert (tmp_path / name).stat().st_size > 10_000


def test_tsnr_figure_reads_off_the_tsnr_an_effect_needs(tmp_path):
    from fastfuncstuff.simulation.plots import plot_tsnr

    grid = np.geomspace(10, 300, 40)
    fig = plot_tsnr({"A": 55.0 / grid}, grid, {"A": [(55.0, 1.0)]})
    text = " ".join(t.get_text() for t in fig.texts)
    assert "1% at tSNR 55" in text and "0.5% at tSNR 110" in text
    plot_tsnr({"A": 55.0 / grid}, grid, path=tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 10_000
