"""Thermal + physiological noise at a given tSNR, and sub-TR event simulation.

The physiological part is an Ornstein-Uhlenbeck process with a correlation time
in *seconds*; added to white noise and sampled at a TR it is exactly ARMA(1,1)
with a = exp(-TR / tau). These tests hold the generator to that algebra, and
hold the analytic contrast variance to what a GLS fit of simulated data shows.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.glm.arma import build_arma11_covariance, compute_arma_lambda
from fastfuncstuff.simulation.core import simulate_bold
from fastfuncstuff.simulation.metrics import design_contrast_variance
from fastfuncstuff.simulation.noise import (
    arma11_to_ou,
    generate_thermal_physio_noise,
    kruger_glover_tsnr,
    ou_to_arma11,
)

CPU = torch.device("cpu")


def _acf(noise: torch.Tensor, lags) -> np.ndarray:
    x = noise.double() - noise.double().mean(0)
    den = (x * x).sum(0)
    return np.array([((x[:-k] * x[k:]).sum(0) / den).mean().item() for k in lags])


class TestArmaMapping:
    @pytest.mark.parametrize("tr", [0.5, 1.25, 2.0])
    @pytest.mark.parametrize("f", [0.0, 0.3, 0.7, 1.0])
    def test_lag_one_is_phys_share_times_a(self, tr, f):
        a, b = ou_to_arma11(tr, 6.0, f)
        assert float(a) == pytest.approx(np.exp(-tr / 6.0))
        assert abs(float(b)) <= 1.0
        assert compute_arma_lambda(float(a), float(b)) == pytest.approx(f * float(a), abs=1e-9)

    def test_limits(self):
        a, b = ou_to_arma11(2.0, 5.0, 1.0)
        assert float(b) == pytest.approx(0.0, abs=1e-12)  # pure AR(1)
        a, b = ou_to_arma11(2.0, 5.0, 0.0)
        assert float(b) == pytest.approx(-float(a))  # cancels to white

    def test_round_trip(self):
        tau = torch.tensor([2.0, 6.0, 15.0])
        f = torch.tensor([0.2, 0.5, 0.9])
        a, b = ou_to_arma11(1.25, tau, f)
        back = arma11_to_ou(1.25, a, b)
        assert torch.allclose(back["tau"], tau.double(), rtol=1e-6)
        assert torch.allclose(back["phys_fraction"], f.double(), rtol=1e-6)
        assert bool(back["representable"].all())

    def test_positive_b_is_flagged_unrepresentable(self):
        assert not bool(arma11_to_ou(2.0, 0.5, 0.3)["representable"])


class TestThermalPhysioNoise:
    def test_tsnr_sets_the_noise_level(self):
        g = torch.Generator().manual_seed(0)
        n = generate_thermal_physio_noise(
            2000, 1.0, 40.0, 0.5, 6.0, n_voxels=300, device=CPU, generator=g
        )
        assert n.std(0).mean().item() == pytest.approx(100.0 / 40.0, rel=0.03)

    @pytest.mark.parametrize("tr", [0.5, 2.0])
    def test_autocorrelation_is_f_times_a_to_the_k(self, tr):
        g = torch.Generator().manual_seed(1)
        f, tau = 0.6, 6.0
        n = generate_thermal_physio_noise(
            4000, tr, 50.0, f, tau, n_voxels=200, device=CPU, generator=g
        )
        a = np.exp(-tr / tau)
        lags = [1, 2, 5]
        assert _acf(n, lags) == pytest.approx(f * a ** np.array(lags), abs=0.02)

    def test_shorter_tr_widens_the_autocorrelation_in_samples(self):
        """Same physiology, faster sampling: more samples inside one tau."""
        g = torch.Generator().manual_seed(2)
        slow = generate_thermal_physio_noise(
            3000, 2.0, 50.0, 0.6, 6.0, n_voxels=100, device=CPU, generator=g
        )
        fast = generate_thermal_physio_noise(
            3000, 0.5, 50.0, 0.6, 6.0, n_voxels=100, device=CPU, generator=g
        )
        assert _acf(fast, [4])[0] > _acf(slow, [4])[0] + 0.2

    def test_no_physiology_is_white(self):
        g = torch.Generator().manual_seed(3)
        n = generate_thermal_physio_noise(
            3000, 1.0, 30.0, 0.0, 6.0, n_voxels=100, device=CPU, generator=g
        )
        assert abs(_acf(n, [1])[0]) < 0.01

    def test_per_voxel_parameters(self):
        g = torch.Generator().manual_seed(4)
        tsnr = torch.tensor([20.0, 80.0])
        f = torch.tensor([0.0, 0.9])
        n = generate_thermal_physio_noise(4000, 1.0, tsnr, f, 5.0, device=CPU, generator=g)
        assert n.shape == (4000, 2)
        assert n[:, 0].std().item() == pytest.approx(5.0, rel=0.05)
        assert n[:, 1].std().item() == pytest.approx(1.25, rel=0.15)
        lag1 = _acf(n[:, :1], [1])[0], _acf(n[:, 1:], [1])[0]
        assert abs(lag1[0]) < 0.05 and lag1[1] > 0.7

    @pytest.mark.slow
    def test_reml_recovers_the_implied_arma(self):
        """The repo's own REML estimator reads back a = exp(-TR / tau)."""
        from fastfuncstuff.glm.arma import reml_grid_search

        g = torch.Generator().manual_seed(5)
        tr, tau, f = 2.0, 6.0, 0.6
        n = generate_thermal_physio_noise(
            300, tr, 50.0, f, tau, n_voxels=30, device=CPU, generator=g
        )
        a_true, b_true = (float(v) for v in ou_to_arma11(tr, tau, f))
        grid_a = torch.arange(0.05, 0.951, 0.05)
        grid_b = torch.arange(-0.8, 0.81, 0.05)
        est = np.array(
            [
                reml_grid_search(torch.ones(300, 1), n[:, v], grid_a, grid_b, device=CPU)[:2]
                for v in range(30)
            ]
        )
        assert np.median(est[:, 0]) == pytest.approx(a_true, abs=0.08)
        assert np.median(est[:, 1]) == pytest.approx(b_true, abs=0.12)

    def test_kruger_glover_saturates_at_one_over_lambda(self):
        tsnr, f = kruger_glover_tsnr(torch.tensor([10.0, 100.0, 1e6]), 0.01)
        assert float(tsnr[-1]) == pytest.approx(100.0, rel=1e-4)
        assert float(f[0]) < 0.02 and float(f[-1]) > 0.99  # thermal- vs physiology-dominated


class TestSimulateBold:
    def test_whole_second_events_at_tr_1_25(self):
        """Events every 2 s sampled at TR 1.25: onsets land exactly on the 0.05 s grid."""
        onsets = [[np.arange(10.0, 290.0, 16.0)], [np.arange(18.0, 290.0, 16.0)]]
        sim = simulate_bold(
            onsets,
            [2.0, 0.0],
            1.25,
            240,
            [1.0, 0.5],
            tsnr=60.0,
            phys_fraction=0.5,
            tau=6.0,
            n_voxels=400,
            device=CPU,
            generator=torch.Generator().manual_seed(6),
        )
        assert sim["microtime_dt"] == pytest.approx(0.05)
        assert sim["data"].shape == (400, 240)
        assert torch.allclose(sim["data"], 100.0 + sim["signal"] + sim["noise"])

        from fastfuncstuff.glm.core import construct_polynomial_matrix

        X = torch.cat(
            [sim["design"].double(), construct_polynomial_matrix(240, 2, CPU, torch.float64)], 1
        )
        beta = torch.linalg.lstsq(X, sim["data"].double().T).solution
        assert beta[0].mean().item() == pytest.approx(1.0, abs=0.03)
        assert beta[1].mean().item() == pytest.approx(0.5, abs=0.03)

    def test_sub_tr_shift_moves_the_regressor(self):
        """Half a TR of onset shift must change the design, not round away."""
        kwargs = dict(
            tr=2.0, n_timepoints_per_run=100, amplitude_psc=[1.0], tsnr=50.0, n_voxels=1, device=CPU
        )
        a = simulate_bold([[np.array([20.0])]], [0.0], **kwargs)["design"][:, 0]
        b = simulate_bold([[np.array([21.0])]], [0.0], **kwargs)["design"][:, 0]
        assert (a - b).abs().max().item() > 0.1


class TestContrastVarianceMatchesMonteCarlo:
    @pytest.mark.parametrize("f", [0.0, 0.7])
    def test_gls_variance(self, f):
        tr, tau, n_t, n_sim = 1.25, 6.0, 200, 2000
        onsets = [[np.arange(6.0, 230.0, 14.0)], [np.arange(13.0, 230.0, 14.0)]]
        sim = simulate_bold(
            onsets,
            [0.0, 0.0],
            tr,
            n_t,
            [0.0, 0.0],
            tsnr=40.0,
            phys_fraction=f,
            tau=tau,
            n_voxels=n_sim,
            device=CPU,
            generator=torch.Generator().manual_seed(7),
        )
        a, b = (float(v[0]) for v in (sim["arma_a"], sim["arma_b"]))
        contrasts = torch.tensor([[1.0, 0.0], [1.0, -1.0]])
        predicted = (
            design_contrast_variance(sim["design"], contrasts, [n_t], 2, a, b) * (100 / 40) ** 2
        )

        from fastfuncstuff.glm.core import construct_polynomial_matrix

        X = torch.cat(
            [sim["design"].double(), construct_polynomial_matrix(n_t, 2, CPU, torch.float64)], 1
        )
        R = (
            build_arma11_covariance(a, b, n_t, CPU, torch.float64)
            if f > 0
            else torch.eye(n_t, dtype=torch.float64)
        )
        L = torch.linalg.cholesky(R)
        Xw = torch.linalg.solve_triangular(L, X, upper=False)
        Yw = torch.linalg.solve_triangular(L, sim["data"].double().T, upper=False)
        beta = torch.linalg.lstsq(Xw, Yw).solution
        observed = torch.stack([beta[0].var(), (beta[0] - beta[1]).var()])
        assert observed.numpy() == pytest.approx(predicted.numpy(), rel=0.1)

    def test_identical_conditions_are_inestimable(self):
        d = torch.rand(100, 1).repeat(1, 2)
        v = design_contrast_variance(d, torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
        assert np.isinf(v[0].item()) and np.isfinite(v[1].item())


class TestDirectArma:
    def test_acf_round_trip(self):
        from fastfuncstuff.simulation.noise import arma11_from_acf

        for a, b in [(0.7, -0.3), (0.5, 0.2), (0.0, 0.4), (0.9, -0.6)]:
            lam = compute_arma_lambda(a, b)
            fa, fb = arma11_from_acf(lam, lam * a)
            assert float(fa) == pytest.approx(a, abs=1e-6)
            assert float(fb) == pytest.approx(b, abs=1e-6)

    def test_arma_noise_has_the_requested_level_and_correlation(self):
        g = torch.Generator().manual_seed(9)
        a, b = 0.0, 0.4
        n = generate_thermal_physio_noise(
            4000, 2.0, 50.0, n_voxels=200, device=CPU, generator=g, arma=(a, b)
        )
        assert n.std(0).mean().item() == pytest.approx(2.0, rel=0.03)
        assert _acf(n, [1, 2]) == pytest.approx([compute_arma_lambda(a, b), 0.0], abs=0.02)

    def test_simulate_bold_reports_the_arma_it_used(self):
        sim = simulate_bold(
            [[np.array([10.0])]],
            [0.0],
            2.0,
            50,
            [1.0],
            tsnr=50.0,
            n_voxels=3,
            device=CPU,
            arma=(0.3, 0.2),
        )
        assert torch.allclose(sim["arma_a"], torch.full((3,), 0.3, dtype=torch.float64))
