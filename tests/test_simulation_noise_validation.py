"""The simulator must deliver what it was asked for.

Every test here pins a *promise the API makes* against ground truth: ask for
rho=0.4 and the realised lag-1 autocorrelation is 0.4; ask for pink_exp=1.0 and
the realised spectrum has that slope; ask for TR=0.35 and the physiological peak
lands where TR=0.35 puts it.

This matters because the simulator is the substrate other tests are built on.
An untested generator turns every downstream failure into a two-suspect problem,
and each of the checks below corresponds to a bug that was actually present:
a decimation-aliasing artifact that halved the requested 1/f slope, a truncated
TR ratio that shifted every frequency at sub-second TRs, AR processes that began
at zero instead of at their stationary variance, and a shape heuristic that read
the voxel axis as time and reported rho ~ 0 for a planted 0.45.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.simulation.noise import (
    add_drift,
    estimate_noise_parameters_from_data,
    estimate_sfnr,
    generate_ar1_noise,
    generate_ar_noise,
    generate_arma_noise,
    generate_fmri_noise,
)

CPU = torch.device("cpu")


def _seeded(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _lag_acf(x: torch.Tensor, lag: int) -> float:
    """Mean lag-k autocorrelation over voxels. x: (n_timepoints, n_voxels)."""
    x = x - x.mean(dim=0, keepdim=True)
    num = (x[:-lag] * x[lag:]).sum(dim=0)
    den = (x * x).sum(dim=0)
    return float((num / den).mean().item())


def _mean_psd(series: torch.Tensor, tr: float):
    """One-sided periodogram averaged over voxels. series: (n_timepoints, n_voxels)."""
    n_t = series.shape[0]
    power = (torch.fft.rfft(series - series.mean(dim=0), dim=0).abs() ** 2).mean(dim=1)
    return torch.fft.rfftfreq(n_t, d=tr), power


def _log_log_slope(freqs: torch.Tensor, power: torch.Tensor) -> float:
    log_f = torch.log(freqs)
    log_p = torch.log(power)
    centred_f = log_f - log_f.mean()
    return float(-((centred_f * (log_p - log_p.mean())).sum() / (centred_f**2).sum()).item())


class TestAutoregressiveGenerators:
    @pytest.mark.parametrize("rho", [0.2, 0.4, 0.6, 0.9])
    def test_ar1_delivers_the_requested_rho(self, rho):
        noise = generate_ar1_noise(
            rho, n_timepoints=4000, n_voxels=200, device=CPU, generator=_seeded(1)
        )
        assert _lag_acf(noise, 1) == pytest.approx(rho, abs=0.01)

    def test_ar1_begins_at_its_stationary_variance(self):
        """No startup ramp: y[0] must seed the recursion, not be written after it.

        Assigning y[0] afterwards left the process starting from zero, so at
        rho=0.9 var(y[1]) was 0.19 against a stationary 1.0 and took ~50 samples
        to recover -- silently shrinking the noise at the start of every run.
        """
        noise = generate_ar1_noise(
            0.9,
            n_timepoints=400,
            n_voxels=6000,
            device=CPU,
            normalize=False,
            generator=_seeded(2),
        )
        early = noise[:5].var(dim=1).mean().item()
        late = noise[-100:].var(dim=1).mean().item()
        assert early == pytest.approx(late, rel=0.15)

    def test_arp_matches_the_yule_walker_solution(self):
        true_ar = [0.5, 0.2]
        noise = generate_ar_noise(
            true_ar, n_timepoints=6000, n_voxels=200, device=CPU, generator=_seeded(3)
        )
        r1, r2 = _lag_acf(noise, 1), _lag_acf(noise, 2)
        recovered = np.linalg.solve(np.array([[1.0, r1], [r1, 1.0]]), np.array([r1, r2]))
        assert recovered[0] == pytest.approx(true_ar[0], abs=0.02)
        assert recovered[1] == pytest.approx(true_ar[1], abs=0.02)

    def test_arp_is_stationary_from_the_first_sample(self):
        """The burn-in exists because seeding y[:p] from innovations alone gave
        those samples variance 1.0 while the process's own is 1.71."""
        noise = generate_ar_noise(
            [0.5, 0.2],
            n_timepoints=400,
            n_voxels=6000,
            device=CPU,
            normalize=False,
            generator=_seeded(4),
        )
        early = noise[:5].var(dim=1).mean().item()
        late = noise[-100:].var(dim=1).mean().item()
        assert early == pytest.approx(late, rel=0.15)

    def test_arma11_matches_its_theoretical_acf(self):
        ar, ma = 0.5, 0.3
        noise = generate_arma_noise(
            [ar], [ma], n_timepoints=6000, n_voxels=300, device=CPU, generator=_seeded(5)
        )
        expected_lag1 = (1 + ar * ma) * (ar + ma) / (1 + 2 * ar * ma + ma**2)
        assert _lag_acf(noise, 1) == pytest.approx(expected_lag1, abs=0.02)
        assert _lag_acf(noise, 2) == pytest.approx(ar * expected_lag1, abs=0.02)

    def test_single_voxel_shape_is_consistent_across_generators(self):
        """generate_arma_noise alone used to return (T, 1) where its siblings
        and its own docstring said (T,)."""
        shapes = {
            "ar1": generate_ar1_noise(0.3, 50, 1, device=CPU, generator=_seeded(6)).shape,
            "ar_p": generate_ar_noise([0.3], 50, 1, device=CPU, generator=_seeded(6)).shape,
            "arma": generate_arma_noise(
                [0.3], [0.2], 50, 1, device=CPU, generator=_seeded(6)
            ).shape,
        }
        assert set(shapes.values()) == {torch.Size([50])}, shapes


class TestSpectralGenerator:
    @pytest.mark.parametrize("pink_exp", [0.5, 1.0, 1.5])
    def test_realised_spectrum_matches_the_requested_one(self, pink_exp):
        """Compared against the analytic template, not against pink_exp itself.

        The 1/(f + 0.01) knee makes the fitted slope legitimately shallower than
        pink_exp; the template carries that knee too, so any *remaining* gap is
        the generator's fault. Decimation without an anti-alias filter used to
        halve the slope here (pink_exp=1.0 measured 0.48).
        """
        noise = generate_fmri_noise(
            tr=1.0,
            duration_s=2400,
            matrix_size=(24, 24),
            pink_exp=pink_exp,
            resp_strength=0.0,
            cardiac_strength=0.0,
            device=CPU,
            generator=_seeded(7),
        )
        series = noise.reshape(noise.shape[0], -1)
        freqs, power = _mean_psd(series, tr=1.0)
        band = (freqs > 0.02) & (freqs < 0.49)
        template = 1.0 / (freqs[band] + 0.01) ** pink_exp

        realised = _log_log_slope(freqs[band], power[band])
        expected = _log_log_slope(freqs[band], template)
        assert realised == pytest.approx(expected, abs=0.05)

    def test_periodogram_ordinates_are_chi_squared(self):
        """A Gaussian process, not a random-phase surrogate.

        Randomising only the phase leaves |X_k| deterministic, so every voxel
        gets an identical periodogram and the noise carries no spectral
        variability -- unrealistically stable for anything that estimates a
        spectrum or an AR coefficient.
        """
        noise = generate_fmri_noise(
            tr=1.0,
            duration_s=1200,
            matrix_size=(32, 32),
            resp_strength=0.0,
            cardiac_strength=0.0,
            device=CPU,
            generator=_seeded(8),
        )
        series = noise.reshape(noise.shape[0], -1)
        per_voxel = torch.fft.rfft(series - series.mean(dim=0), dim=0).abs() ** 2
        band = per_voxel[5:200]
        normalised = band / band.mean(dim=1, keepdim=True)
        # chi-squared with 2 dof has mean 1 and std 1.
        assert normalised.std().item() == pytest.approx(1.0, abs=0.15)

    @pytest.mark.parametrize("tr,resp_freq", [(1.0, 0.35), (2.0, 0.20), (0.5, 0.35)])
    def test_physio_peak_lands_at_the_requested_frequency(self, tr, resp_freq):
        noise = generate_fmri_noise(
            tr=tr,
            duration_s=2400,
            matrix_size=(16, 16),
            pink_exp=0.0,
            resp_freq=resp_freq,
            resp_width=0.05,
            resp_strength=10.0,
            cardiac_strength=0.0,
            device=CPU,
            generator=_seeded(9),
        )
        series = noise.reshape(noise.shape[0], -1)
        freqs, power = _mean_psd(series, tr=tr)
        peak = freqs[int(power[1:].argmax()) + 1].item()
        assert peak == pytest.approx(resp_freq, abs=0.02)

    def test_cardiac_aliases_when_it_sits_above_nyquist(self):
        """Deliberate, and physical: 1 Hz pulsation really is undersampled at
        TR=2 s, and real data carries it folded down to near DC."""
        noise = generate_fmri_noise(
            tr=2.0,
            duration_s=4800,
            matrix_size=(16, 16),
            pink_exp=0.0,
            resp_strength=0.0,
            cardiac_freq=1.0,
            cardiac_width=0.03,
            cardiac_strength=10.0,
            device=CPU,
            generator=_seeded(10),
        )
        series = noise.reshape(noise.shape[0], -1)
        freqs, power = _mean_psd(series, tr=2.0)
        peak = freqs[int(power[1:].argmax()) + 1].item()
        assert peak < 0.03  # 1.0 Hz folds to DC at a 0.25 Hz Nyquist

    @pytest.mark.parametrize("tr", [0.35, 0.25])
    def test_sub_second_tr_is_exact(self, tr):
        """int(fs_high * tr) truncated: TR=0.35 generated a 0.30 s series and
        put every requested frequency ~14% low."""
        resp_freq = 0.35
        noise = generate_fmri_noise(
            tr=tr,
            duration_s=2400,
            matrix_size=(16, 16),
            pink_exp=0.0,
            resp_freq=resp_freq,
            resp_width=0.03,
            resp_strength=10.0,
            cardiac_strength=0.0,
            device=CPU,
            generator=_seeded(11),
        )
        series = noise.reshape(noise.shape[0], -1)
        freqs, power = _mean_psd(series, tr=tr)
        peak = freqs[int(power[1:].argmax()) + 1].item()
        assert peak == pytest.approx(resp_freq, abs=0.02)


class TestNoiseParameterEstimation:
    """The 'measure your scanner, don't guess' path has to survive real shapes."""

    RHO = 0.45

    def _planted(self, n_timepoints, n_voxels, seed=12):
        noise = generate_ar1_noise(
            self.RHO,
            n_timepoints=n_timepoints,
            n_voxels=n_voxels,
            device=CPU,
            generator=_seeded(seed),
        )
        return noise * 10.0 + 100.0

    @pytest.mark.parametrize(
        "n_timepoints,n_voxels,transpose",
        [
            (400, 500, False),  # (time, voxels), voxels > time
            (500, 400, False),  # (time, voxels), time > voxels
            (400, 500, True),  # (voxels, time) -- what every ffs tool stores
            (500, 400, True),  # (voxels, time), the small-ROI long-run case
        ],
    )
    def test_round_trip_in_any_orientation(self, n_timepoints, n_voxels, transpose):
        """Shape cannot disambiguate this, so the axis is inferred from
        autocorrelation. 'Longer axis is time' read voxels as time on ordinary
        fMRI matrices and silently returned rho ~ 0."""
        data = self._planted(n_timepoints, n_voxels)
        if transpose:
            data = data.T
        params = estimate_noise_parameters_from_data(data, ar_order=1, device=CPU)
        assert params["n_timepoints"] == n_timepoints
        assert params["ar_coefficients"][0] == pytest.approx(self.RHO, abs=0.05)

    def test_round_trip_on_a_realistic_aspect_ratio(self):
        data = self._planted(300, 20000).T  # 20k voxels, 300 TRs
        params = estimate_noise_parameters_from_data(data, ar_order=1, device=CPU)
        assert params["n_timepoints"] == 300
        assert params["ar_coefficients"][0] == pytest.approx(self.RHO, abs=0.05)

    def test_round_trip_on_a_4d_volume(self):
        data = self._planted(300, 8 * 8 * 4).T.reshape(8, 8, 4, 300)
        params = estimate_noise_parameters_from_data(data, ar_order=1, device=CPU)
        assert params["n_timepoints"] == 300
        assert params["ar_coefficients"][0] == pytest.approx(self.RHO, abs=0.05)

    def test_explicit_time_axis_overrides_inference(self):
        data = self._planted(400, 500)  # (time, voxels)
        params = estimate_noise_parameters_from_data(data, ar_order=1, device=CPU, time_axis=0)
        assert params["n_timepoints"] == 400
        assert params["ar_coefficients"][0] == pytest.approx(self.RHO, abs=0.05)

    def test_ar2_round_trip(self):
        noise = generate_ar_noise(
            [0.5, 0.2], n_timepoints=2000, n_voxels=300, device=CPU, generator=_seeded(13)
        )
        params = estimate_noise_parameters_from_data(
            noise * 10.0 + 100.0, ar_order=2, device=CPU, time_axis=0
        )
        assert params["ar_coefficients"][0] == pytest.approx(0.5, abs=0.05)
        assert params["ar_coefficients"][1] == pytest.approx(0.2, abs=0.05)

    def test_documented_keys_are_actually_returned(self):
        """The docstring's own example indexed keys that did not exist."""
        params = estimate_noise_parameters_from_data(self._planted(200, 300), device=CPU)
        for key in ("ar_coefficients", "ar_coefficients_mean", "sfnr", "sfnr_mean", "noise_std"):
            assert key in params, f"{key} is documented but missing"


class TestSfnr:
    @pytest.mark.parametrize("target", [50.0, 150.0])
    def test_recovers_a_planted_sfnr(self, target):
        noise = torch.randn(1000, 200, generator=_seeded(14))
        data = 100.0 + noise * (100.0 / target)
        assert estimate_sfnr(data, device=CPU)["sfnr_mean"] == pytest.approx(target, rel=0.05)

    def test_detrending_protects_sfnr_from_drift(self):
        """Friedman & Glover take the fluctuation as the *detrended* residual;
        without that, drift lands in the denominator and depresses SFNR."""
        # estimate_sfnr takes time as the LAST axis, so the ramp runs along it.
        clean = 100.0 + torch.randn(300, 400, generator=_seeded(15)) * (100.0 / 150.0)
        ramp = torch.linspace(-3.0, 3.0, 400, device=CPU).unsqueeze(0)
        drifted = clean + ramp

        assert estimate_sfnr(drifted, device=CPU)["sfnr_mean"] == pytest.approx(150.0, rel=0.1)
        raw = estimate_sfnr(drifted, device=CPU, detrend=False)["sfnr_mean"]
        assert raw < 100.0  # the drift would otherwise dominate the denominator


class TestReproducibility:
    """Tests need to isolate streams; a global seed is not enough when a fixture
    and the code under test both draw."""

    def test_generator_makes_output_reproducible(self):
        first = generate_ar1_noise(0.4, 200, 20, device=CPU, generator=_seeded(16))
        second = generate_ar1_noise(0.4, 200, 20, device=CPU, generator=_seeded(16))
        assert torch.allclose(first, second)

    def test_generator_is_independent_of_global_seed(self):
        torch.manual_seed(1234)
        first = generate_fmri_noise(
            tr=1.0, duration_s=120, matrix_size=(4, 4), device=CPU, generator=_seeded(17)
        )
        torch.manual_seed(9999)
        second = generate_fmri_noise(
            tr=1.0, duration_s=120, matrix_size=(4, 4), device=CPU, generator=_seeded(17)
        )
        assert torch.allclose(first, second)

    def test_add_drift_accepts_a_generator(self):
        # A zeros base would make this vacuous: drift is scaled by the data's own
        # std, so it would be identically zero whatever the generator did.
        base = torch.randn(200, 50, generator=_seeded(18))
        first = add_drift(base, amplitude=1.0, device=CPU, generator=_seeded(19))
        second = add_drift(base, amplitude=1.0, device=CPU, generator=_seeded(19))
        assert torch.allclose(first, second)
        assert not torch.allclose(first, base)
