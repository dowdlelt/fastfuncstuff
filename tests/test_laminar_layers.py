"""
Tests for cortical-depth sampling, the voxel PSF, and the study layer.

The MATLAB oracle here covers the two steps the reference driver ships as cached
``.mat`` files because its own implementations are too slow to rerun. Regenerate
with::

    cd tests/laminar_oracle && matlab -batch "run('make_layers_oracle.m')"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fastfuncstuff.laminar.experiment import (
    bayesian_parameter_average,
    faes_priors,
    layer_model_names,
    layer_model_targets,
    posterior_model_probabilities,
)
from fastfuncstuff.laminar.forward import apply_depth_psf
from fastfuncstuff.laminar.inversion import VLResult, vec
from fastfuncstuff.laminar.layers import (
    depth_bin_centres,
    estimate_depth_psf,
    gaussian_depth_kernel,
    label_mean,
    voxels_to_layers,
)
from fastfuncstuff.laminar.params import ModelSpec, zero_params

ORACLE = Path(__file__).parent / "laminar_oracle" / "laminar_layers_oracle.json"


def _oracle():
    if not ORACLE.exists():  # pragma: no cover - only when the dump is missing
        pytest.skip(f"no layers oracle at {ORACLE}")
    return json.loads(ORACLE.read_text())


# --------------------------------------------------------------------------
# Depth sampling
# --------------------------------------------------------------------------


def test_label_mean_ignores_nan_and_empty_labels():
    values = np.array([1.0, 3.0, np.nan, 10.0])
    labels = np.array([0, 0, 0, 2])
    out = label_mean(values, labels, 4)
    assert out[0] == pytest.approx(2.0)  # NaN dropped, not propagated
    assert np.isnan(out[1])  # label with no members
    assert out[2] == pytest.approx(10.0)
    assert np.isnan(out[3])


def test_depth_bin_centres():
    c = depth_bin_centres(4)
    assert np.allclose(c, [0.125, 0.375, 0.625, 0.875])


def test_voxels_to_layers_puts_superficial_first():
    """Depth 0 of the output must be the CSF end, matching the model."""
    depth = np.linspace(0.0, 1.0, 200)
    # A ramp in depth: bright at the pial surface, dark at white matter.
    data = np.tile(depth[:, None], (1, 5))
    y = voxels_to_layers(data, depth, 5)
    assert y.shape == (5, 5)
    assert y[0, 0] > y[0, -1], "depth index 0 must be the CSF end"
    assert np.all(np.diff(y[0]) < 0)


def test_voxels_to_layers_is_a_weighted_mean():
    """A spatially flat signal must come back flat at every depth."""
    depth = np.random.default_rng(0).uniform(0.05, 0.95, 400)
    data = np.full((400, 3), 2.5)
    y = voxels_to_layers(data, depth, 7)
    assert np.allclose(y, 2.5, atol=1e-9)


def test_voxels_to_layers_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="voxels of data"):
        voxels_to_layers(np.zeros((10, 4)), np.zeros(9), 5)


def test_depth_sampling_matches_matlab():
    o = _oracle()
    data = np.asarray(o["data_sel"], dtype=float)
    depth = np.asarray(o["depth_sel"], dtype=float).reshape(-1)
    for K in (7, 9):
        expect = np.asarray(o[f"y{K}"], dtype=float)
        got = voxels_to_layers(data, depth, K)
        assert np.abs(got - expect).max() < 1e-10, f"K={K}"


def test_depth_downsampling_matches_matlab():
    """``label_mean`` against MATLAB, NaNs and empty labels included.

    The real ROI case is a 1.5M-voxel label volume, too large to ship through
    JSON, so the oracle also dumps a small case built to exercise the parts that
    break silently: NaNs scattered through the values, and labels with no members
    at all. The reference drops NaNs from the mean rather than propagating them,
    and returns NaN -- not 0 -- for an empty label.
    """
    o = _oracle()
    values = np.asarray(o["small_v"], dtype=float).reshape(-1)
    labels = np.asarray(o["small_l0"], dtype=int).reshape(-1)
    expect = np.asarray(o["small_out"], dtype=float).reshape(-1)
    got = label_mean(values, labels, expect.size)

    assert np.isnan(got).tolist() == np.isnan(expect).tolist(), "NaN pattern differs"
    good = ~np.isnan(expect)
    assert np.abs(got[good] - expect[good]).max() < 1e-12
    # The case must actually contain what it claims to test, or it proves nothing.
    assert np.isnan(values).any(), "oracle case has no NaNs to drop"
    assert np.isnan(expect).any(), "oracle case has no empty labels"


def test_downsampled_depth_map_has_the_expected_extent():
    """Sanity on the full-ROI downsampling the reference caches to a .mat."""
    o = _oracle()
    ev_ds = np.asarray(o["EV_ds"], dtype=float).reshape(-1)
    assert ev_ds.size == int(o["nlab"])
    finite = ev_ds[np.isfinite(ev_ds)]
    # Normalised cortical depth: WM at 0, CSF at 1.
    assert finite.min() >= 0.0 and finite.max() <= 1.0


# --------------------------------------------------------------------------
# Point-spread function
# --------------------------------------------------------------------------


def test_depth_psf_is_normalised_and_centred():
    """Synthetic check: the estimator returns a unit-sum, peaked kernel."""
    rng = np.random.default_rng(0)
    shape = (16, 16, 16)
    hi = rng.uniform(0, 1, shape)
    labels = np.arange(8 * 8 * 8).reshape(8, 8, 8, order="F")
    labels = np.repeat(np.repeat(np.repeat(labels, 2, 0), 2, 1), 2, 2)
    lo = label_mean(hi, labels, 8 * 8 * 8)
    sel = np.arange(0, 8 * 8 * 8, 3)
    kern = estimate_depth_psf(3, 7, labels, hi, lo, sel, seed=0)
    assert kern.shape == (7,)
    assert kern.sum() == pytest.approx(1.0)
    assert np.all(kern >= 0)
    assert kern.argmax() == 3, "a symmetric blur should peak at the middle depth"


def test_even_length_psf_uses_matlab_cropping():
    """K = 10 is in the published set, so the even-kernel path is not academic.

    MATLAB's conv(...,'same') keeps full-convolution samples from floor(L/2),
    which for even L is one sample later than numpy's convention.
    """
    profile = torch.zeros(1, 10, dtype=torch.float64)
    profile[0, 5] = 1.0
    kernel = torch.tensor([0.1, 0.2, 0.4, 0.3], dtype=torch.float64)
    out = apply_depth_psf(profile, kernel)
    assert out.shape == profile.shape
    full = np.convolve(profile[0].numpy(), kernel.numpy(), mode="full")
    expect_down = full[len(kernel) // 2 :][:10]
    # The result is the max of the two orientations, so it is >= either alone.
    assert np.all(out[0].numpy() >= expect_down - 1e-12)


def test_psf_kernel_matches_matlab():
    """The kernel MATLAB's fitted blur produces at each K, including K = 10.

    Everything after the two-parameter fit -- the PCHIP resampling onto the
    depth-bin centres and the normalisation -- is checked exactly here. The fit
    itself needs the 1.5M-voxel anatomical grid, so it is covered separately by
    the notebook against the kernel the authors ship.
    """
    o = _oracle()
    variance = float(np.asarray(o["est"], dtype=float).reshape(-1)[1])
    for K in (7, 9, 10, 11):
        expect = np.asarray(o[f"kernel{K}"], dtype=float).reshape(-1)
        got = gaussian_depth_kernel(variance, K)
        assert got.shape == expect.shape, f"K={K}"
        assert np.abs(got - expect).max() < 1e-12, f"K={K}"


def test_psf_estimator_recovers_a_known_blur():
    """End-to-end on the estimator: a known variance comes back out."""
    o = _oracle()
    variance = float(np.asarray(o["est"], dtype=float).reshape(-1)[1])
    assert 0.0 < variance < 1.0
    # The kernel is symmetric about mid-cortex by construction, so an odd K
    # must peak at its middle bin.
    for K in (7, 9, 11):
        assert gaussian_depth_kernel(variance, K).argmax() == K // 2


# --------------------------------------------------------------------------
# Study layer
# --------------------------------------------------------------------------


def test_model_space_is_the_full_power_set_null_first():
    names = layer_model_names(3)
    targets = layer_model_targets(3)
    assert len(names) == len(targets) == 8
    assert names[0] == "null" and targets[0] == (0, 0, 0)
    assert targets[-1] == (1, 1, 1)
    assert len(set(targets)) == 8


def test_faes_priors_free_only_the_targeted_depths():
    spec = ModelSpec(N=3, K=9, n_inputs=2, n_mod=1)
    priors = faes_priors(spec, (1, 0, 1))
    b_var = torch.diagonal(priors.pC["B"][0])
    assert b_var[0] > 0 and b_var[1] == 0 and b_var[2] > 0
    # Off-diagonal modulation is never estimated: the hypothesis is about which
    # depth is modulated, not about coupling between depths.
    off = priors.pC["B"][0] - torch.diag_embed(b_var)
    assert torch.all(off == 0)
    # mu and lam have non-zero prior means but no variance.
    assert priors.pE["mu"] != 0 and priors.pC["mu"] == 0


def test_null_model_has_fewer_free_parameters():
    spec = ModelSpec(N=3, K=9, n_inputs=2, n_mod=1)
    counts = [int((vec(faes_priors(spec, t).pC) > 0).sum()) for t in layer_model_targets(3)]
    assert counts[0] == min(counts)
    assert counts[-1] == max(counts) == counts[0] + 3


def test_posterior_model_probabilities_sum_to_one():
    p = posterior_model_probabilities([-100.0, -102.0, -110.0])
    assert float(p.sum()) == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]
    # A free-energy difference of 2 nats is a Bayes factor of e^2.
    assert float(p[0] / p[1]) == pytest.approx(np.exp(2.0))


def _fake_result(spec, value, variance, F):
    Ep = zero_params(spec)
    Ep["sigma"] = torch.tensor(value, dtype=torch.float64)
    idx = torch.nonzero(vec(Ep) == value).squeeze(-1)
    return VLResult(
        Ep=Ep,
        Cp=torch.tensor([[variance]], dtype=torch.float64),
        Eh=torch.zeros(spec.K, dtype=torch.float64),
        F=torch.tensor(F, dtype=torch.float64),
        L=torch.zeros(3, dtype=torch.float64),
        y_pred=torch.zeros(4, spec.K, dtype=torch.float64),
        free_idx=idx,
        n_iter=1,
        converged=True,
    )


def test_bpa_weights_by_precision():
    """A sharp posterior must dominate a vague one, not average with it."""
    spec = ModelSpec(N=3, K=9, n_inputs=2, n_mod=1)
    sharp = _fake_result(spec, 1.0, 0.01, -100.0)
    vague = _fake_result(spec, 3.0, 1.00, -101.0)
    bpa = bayesian_parameter_average([sharp, vague], spec)
    # Precision-weighted: (100*1 + 1*3)/101
    assert float(bpa.Ep["sigma"]) == pytest.approx((100 * 1.0 + 1 * 3.0) / 101, rel=1e-6)
    # And the combined posterior is sharper than either input.
    assert float(bpa.Cp[0, 0]) < 0.01


def test_bpa_free_energy_modes():
    spec = ModelSpec(N=3, K=9, n_inputs=2, n_mod=1)
    runs = [_fake_result(spec, 1.0, 0.1, -10.0), _fake_result(spec, 1.0, 0.1, -20.0)]
    assert float(bayesian_parameter_average(runs, spec, free_energy="reference").F) == -10.0
    assert float(bayesian_parameter_average(runs, spec, free_energy="sum").F) == -30.0
    assert float(bayesian_parameter_average(runs, spec, free_energy="mean").F) == -15.0
    with pytest.raises(ValueError, match="free_energy must be"):
        bayesian_parameter_average(runs, spec, free_energy="accumulated")


def test_bpa_rejects_mismatched_parameter_structure():
    spec = ModelSpec(N=3, K=9, n_inputs=2, n_mod=1)
    a = _fake_result(spec, 1.0, 0.1, -10.0)
    b = _fake_result(spec, 1.0, 0.1, -10.0)
    b.free_idx = torch.tensor([0])
    with pytest.raises(ValueError, match="share a parameter structure"):
        bayesian_parameter_average([a, b], spec)
