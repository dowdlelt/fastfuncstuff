"""Monte-Carlo design power: calibrated false positives, and power that matches theory."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.simulation.core import simulate_bold
from fastfuncstuff.simulation.power import amplitude_for_power, simulate_design_power

CPU = torch.device("cpu")
TR = 1.25
RUNS = [240, 240]


def _design(shift_s: float = 0.0) -> torch.Tensor:
    rng = np.random.default_rng(0)
    on_a = [np.sort(rng.uniform(10, 280, 15)) + shift_s for _ in RUNS]
    on_b = [np.sort(rng.uniform(10, 280, 15)) + shift_s for _ in RUNS]
    return simulate_bold(
        [on_a, on_b], [2.0, 2.0], TR, RUNS, [0, 0], tsnr=50, n_voxels=1, device=CPU
    )["design"]


def _run(noise, amplitudes=(0.5, 1.0), n_reps=3000, **kw):
    return simulate_design_power(
        _design(),
        RUNS,
        TR,
        {"A": [1, 0], "A-B": [1, -1]},
        list(amplitudes),
        noise,
        beta_pattern=[1, 0],
        n_reps=n_reps,
        alpha=0.05,
        device=CPU,
        **kw,
    )


def _rows(res, noise, contrast):
    return {
        r["amplitude"]: r for r in res["table"] if r["noise"] == noise and r["contrast"] == contrast
    }


def test_white_noise_matches_theory():
    res = _run([{"label": "w", "tsnr": 40.0, "phys_fraction": 0.0}])
    for c in ("A", "A-B"):
        rows = _rows(res, "w", c)
        assert rows[0.0]["power"] == pytest.approx(0.05, abs=0.015)  # false-positive rate
        for amp in (0.5, 1.0):
            r = rows[amp]
            assert r["mean_est"] == pytest.approx(r["true_effect"], abs=0.03)
            assert r["sd_est"] == pytest.approx(r["sd_predicted"], rel=0.06)
            assert r["power"] == pytest.approx(r["power_predicted"], abs=0.04)


def test_autocorrelation_inflates_naive_ols_and_the_correction_restores_alpha():
    """The whole reason to carry the ARMA: naive OLS t is far too liberal."""
    res = _run([{"label": "ou", "tsnr": 40.0, "phys_fraction": 0.7, "tau": 6.0}])
    for c in ("A", "A-B"):
        null = _rows(res, "ou", c)[0.0]
        assert null["power_naive"] > 0.15
        assert null["power"] == pytest.approx(0.05, abs=0.025)
        one = _rows(res, "ou", c)[1.0]
        assert one["sd_est"] == pytest.approx(one["sd_predicted"], rel=0.06)
        assert one["power"] == pytest.approx(one["power_predicted"], abs=0.04)


def test_measured_arma_is_accepted_directly():
    res = _run([{"label": "ma", "tsnr": 40.0, "arma": (0.0, 0.4)}], amplitudes=(1.0,))
    null = _rows(res, "ma", "A")[0.0]
    assert null["power"] == pytest.approx(0.05, abs=0.025)


def test_hrf_mismatch_shows_as_bias():
    """Signal generated 2 s late, fitted with nominal onsets: the amplitude is underestimated."""
    res = _run(
        [{"label": "w", "tsnr": 40.0, "phys_fraction": 0.0}],
        amplitudes=(1.0,),
        true_design=_design(shift_s=2.0),
    )
    assert _rows(res, "w", "A")[1.0]["mean_est"] < 0.9


def test_noisier_bins_need_bigger_effects():
    res = _run(
        [
            {"label": "worst", "tsnr": 20.0, "phys_fraction": 0.3, "tau": 4.0},
            {"label": "best", "tsnr": 100.0, "phys_fraction": 0.7, "tau": 6.0},
        ],
        amplitudes=np.linspace(0.1, 3.0, 12),
        n_reps=200,
    )
    need = amplitude_for_power(res, target=0.8)
    assert need[("best", "A")] < need[("worst", "A")]
    assert np.isfinite(need[("best", "A")])


def test_collinear_conditions_are_refused():
    d = _design()
    with pytest.raises(ValueError, match="rank-deficient"):
        simulate_design_power(
            torch.cat([d[:, :1], d[:, :1]], 1),
            RUNS,
            TR,
            {"A": [1, 0]},
            [1.0],
            [{"tsnr": 40.0}],
            device=CPU,
            n_reps=10,
        )
