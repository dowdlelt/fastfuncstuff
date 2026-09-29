"""The shared design-power figures render and save."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import torch

from fastfuncstuff.simulation.experiment import ExperimentSpec, Interval, Unit, realize
from fastfuncstuff.simulation.plots import plot_design, plot_design_comparison, plot_power
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
    plot_power(res, labels, contrasts, [1, 0], effect=1.0, path=tmp_path / "p.png")
    plot_design(res, reals[0], 2, path=tmp_path / "d.png")
    plot_design_comparison({"x": res, "y": res}, "A", labels, path=tmp_path / "c.png")
    for name in ("p.png", "d.png", "c.png"):
        assert (tmp_path / name).stat().st_size > 10_000
