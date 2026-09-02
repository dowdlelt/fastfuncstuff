import numpy as np
import pytest
import torch

from fastfuncstuff.decomposition.tools import (
    apply_high_pass_fft,
    apply_polort_projection,
    component_condition_correlations,
    effective_rank_from_spectrum,
    mp_spikes_from_spectrum,
    parse_num_comps_spec,
)


def test_parse_num_comps_spec_int():
    assert parse_num_comps_spec("10") == 10
    assert isinstance(parse_num_comps_spec("10"), int)


def test_parse_num_comps_spec_float():
    result = parse_num_comps_spec("0.7")
    assert result == pytest.approx(0.7)
    assert isinstance(result, float)


def test_parse_num_comps_spec_string():
    assert parse_num_comps_spec("laplace") == "laplace"
    # the old name still parses, mapped to the new one
    assert parse_num_comps_spec("melodic") == "laplace"


def test_apply_polort_projection_removes_linear_trend():
    n_vox, n_t = 5, 100
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)
    trend = t.unsqueeze(0).expand(n_vox, -1)
    noise = torch.randn(n_vox, n_t, dtype=torch.float64) * 0.01
    data = trend + noise
    result = apply_polort_projection(data.clone(), polort=1, device=torch.device("cpu"))
    for v in range(n_vox):
        assert result[v].std() < 0.1


def test_apply_polort_projection_shape():
    data = torch.randn(20, 100)
    result = apply_polort_projection(data.clone(), polort=2, device=torch.device("cpu"))
    assert result.shape == (20, 100)


def test_apply_high_pass_fft_preserves_shape():
    data = torch.randn(20, 100)
    result = apply_high_pass_fft(data.clone(), tr=2.0, high_pass_hz=0.01)
    assert result.shape == (20, 100)


def test_apply_high_pass_fft_removes_dc():
    data = torch.ones(5, 200) * 3.0
    result = apply_high_pass_fft(data.clone(), tr=2.0, high_pass_hz=0.01)
    assert torch.abs(result).max().item() < 1e-6


def test_effective_rank_from_spectrum():
    signal = np.array([100.0, 80.0, 60.0])
    noise = np.ones(17) * 0.001
    evals = np.concatenate([signal, noise])
    rank = effective_rank_from_spectrum(evals)
    assert 2 <= rank <= 5


def test_mp_spikes_from_spectrum():
    n_features = 200
    n_samples = 50
    noise_evals = np.random.exponential(1.0, size=n_features)
    noise_evals = np.sort(noise_evals)[::-1]
    median_tail = np.median(noise_evals[100:])
    beta = n_features / n_samples
    lambda_plus = median_tail * (1.0 + np.sqrt(beta)) ** 2
    spikes_evals = noise_evals.copy()
    spikes_evals[0] = lambda_plus + 100.0
    spikes_evals[1] = lambda_plus + 50.0
    spikes_evals = np.sort(spikes_evals)[::-1]
    n_spikes = mp_spikes_from_spectrum(spikes_evals, n_samples=n_samples, n_features=n_features)
    assert n_spikes >= 2


def test_component_condition_correlations_shape():
    mixing = torch.randn(50, 5)
    design = torch.randn(50, 3)
    result = component_condition_correlations(mixing, design)
    assert result.shape == (5, 3)


def test_component_condition_correlations_diagonal():
    torch.manual_seed(42)
    base = torch.randn(50, 4)
    corr = component_condition_correlations(base, base)
    for i in range(4):
        assert corr[i, i] == pytest.approx(1.0, abs=1e-5)


def test_mixture_fit_returns_the_expected_parameter_set():
    """Shape/key contract of the mixture fit; behaviour is covered in test_mixture.py."""
    from fastfuncstuff.decomposition.mixture import fit_gaussian_gamma_mixture

    torch.manual_seed(0)
    result = fit_gaussian_gamma_mixture(torch.randn(3, 500, dtype=torch.float64))
    for key in (
        "mu_noise",
        "var_noise",
        "mu_pos",
        "var_pos",
        "mu_neg",
        "var_neg",
        "pi_noise",
        "pi_pos",
        "pi_neg",
        "converged",
    ):
        assert result[key].shape == (3,), key
    for key in ("p_noise", "p_pos", "p_neg", "x_std"):
        assert result[key].shape == (3, 500), key
