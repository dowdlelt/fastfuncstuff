"""Fitted REML must earn its power through an independent null check."""

import numpy as np
import pytest
import torch

from fastfuncstuff.simulation.power import amplitude_for_power, simulate_design_power


def test_reml_reuses_noise_fits_across_effects_and_tsnr(monkeypatch):
    import fastfuncstuff.glm.arma as arma

    torch.manual_seed(0)
    x = torch.sin(torch.arange(100, dtype=torch.float64) * 0.2)[:, None]
    calls = []
    original = arma.fit_glm_arma11

    def counted(*args, **kw):
        calls.append(args[0].shape[0])
        return original(*args, **kw)

    monkeypatch.setattr(arma, "fit_glm_arma11", counted)
    result = simulate_design_power(
        x,
        [100],
        1.0,
        {"A": [1]},
        [0.0, 0.5, 1.0],
        [
            {"label": "a", "tsnr": 50.0, "phys_fraction": 0.0},
            {"label": "b", "tsnr": 100.0, "phys_fraction": 0.0},
        ],
        n_reps=100,
        alpha=0.05,
        estimator="reml",
        null_reps=500,
        device=torch.device("cpu"),
        keep_t=True,
    )
    assert calls == [600]
    assert all(r["calibration"] == "checked" for r in result["table"])
    nulls = [r for r in result["table"] if r["amplitude"] == 0]
    assert all(r["null_rate"] == pytest.approx(0.05, abs=0.03) for r in nulls)
    assert amplitude_for_power(result)[("b", "A")] < amplitude_for_power(result)[("a", "A")]
    for r in result["table"]:
        corrected = result["t"][(r["noise"], r["amplitude"], "A")][0]
        assert corrected.mean() == pytest.approx(r["mean_t"])


def test_slow_noise_cannot_turn_liberal_reml_t_into_good_timing():
    n = 180
    x = torch.sin(torch.arange(n, dtype=torch.float64) * 0.06)
    x += 0.1 * torch.as_tensor(np.random.default_rng(2).normal(size=n))
    result = simulate_design_power(
        x[:, None],
        [n],
        1.0,
        {"A": [1]},
        [1.0, 2.0],
        [{"label": "fast", "tsnr": 50.0, "phys_fraction": 0.8, "tau": 6.0}],
        n_reps=100,
        null_reps=2048,
        alpha=0.05,
        poly_degree=2,
        seed=17,
        estimator="reml",
        device=torch.device("cpu"),
    )
    assert all(r["calibration"] == "inflated" for r in result["table"])
    assert all(np.isnan(r["power_validated"]) for r in result["table"])
    assert np.isnan(amplitude_for_power(result)[("fast", "A")])
    assert result["table"][0]["generating_a"] > 0.8


def test_insufficient_nulls_withhold_a_verdict():
    x = torch.sin(torch.arange(60, dtype=torch.float64) * 0.2)[:, None]
    result = simulate_design_power(
        x,
        [60],
        1.0,
        {"A": [1]},
        [2.0],
        [{"tsnr": 50.0, "phys_fraction": 0.0}],
        n_reps=10,
        null_reps=20,
        alpha=0.001,
        estimator="reml",
        device=torch.device("cpu"),
    )
    assert all(r["calibration"] == "limited" for r in result["table"])
    assert np.isnan(amplitude_for_power(result)[("noise0", "A")])


def test_near_unit_stationary_arma_is_supported():
    from fastfuncstuff.glm.arma import build_arma11_covariance, fit_glm_arma11

    cpu = torch.device("cpu")
    R = build_arma11_covariance(0.98, -0.7, 80, cpu, torch.float64)
    assert R is not None
    L = torch.linalg.cholesky(R)
    y = (L @ torch.randn(80, 8, generator=torch.Generator().manual_seed(5), dtype=torch.float64)).T
    fit = fit_glm_arma11(
        y,
        torch.ones(80, 1),
        1.0,
        a_grid=torch.tensor([0.0, 0.9, 0.98]),
        b_grid=torch.tensor([-0.7, 0.0]),
        device=cpu,
        verbose=False,
        use_double=True,
    )
    assert torch.isfinite(fit.tstats).all()
    assert build_arma11_covariance(1.0, 0.0, 80, cpu) is None
