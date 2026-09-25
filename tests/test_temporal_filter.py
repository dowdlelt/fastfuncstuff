"""Temporal filters (processing/temporal_filter.py)."""

import numpy as np
import pytest
import torch
from scipy.ndimage import uniform_filter1d

from fastfuncstuff.processing.temporal_filter import (
    apply_temporal_filter,
    parse_temporal_filter,
)

CPU = torch.device("cpu")
TR = 1.0
T = 400
t = np.arange(T) * TR


def _sines(*freqs, offset=0.0):
    return torch.tensor(
        np.stack([np.sin(2 * np.pi * f * t) for f in freqs]).sum(0)[None] + offset,
        dtype=torch.float64,
    )


# Pass/stop-band checks stay clear of the run edges, where any finite filter
# smears whatever the extension joins there; the ramp test covers the edges.
MID = slice(40, T - 40)


def _filter(x, spec, **kw):
    return apply_temporal_filter(x, TR, parse_temporal_filter(spec), device=CPU, **kw)


def test_lowpass_keeps_slow_drops_fast_and_keeps_the_mean():
    slow = _sines(0.02, offset=100.0)
    out = _filter(slow + _sines(0.3), "lowpass:0.1")
    np.testing.assert_allclose(out[:, MID].numpy(), slow[:, MID].numpy(), atol=0.05)


def test_highpass_and_bandpass_remove_the_mean_and_out_of_band_sines():
    x = _sines(0.005, 0.05, 0.3, offset=100.0)
    for spec, keep in (("highpass:0.2", _sines(0.3)), ("bandpass:0.02,0.1", _sines(0.05))):
        out = _filter(x, spec)
        np.testing.assert_allclose(out[:, MID].numpy(), keep[:, MID].numpy(), atol=0.1)


def test_symmetric_extension_stops_the_run_end_wrapping_onto_its_start():
    ramp = torch.tensor(np.linspace(0, 10, T)[None], dtype=torch.float64)
    sym = _filter(ramp, "lowpass:0.1")
    wrapped = _filter(ramp, "lowpass:0.1", symmetric=False)
    assert (sym - ramp).abs().max() < 0.5  # a slow ramp passes a low-pass
    assert (wrapped - ramp).abs().max() > 3.0  # the periodic wrap sees a 10-unit step


def test_runs_are_filtered_independently():
    x = torch.cat([torch.zeros(1, T), torch.full((1, T), 10.0)], dim=1).double()
    for spec in ("lowpass:0.1", "movavg:7"):
        out = _filter(x, spec, run_starts=[0, T])
        torch.testing.assert_close(out, x)


def test_moving_average_matches_a_centred_uniform_filter():
    x = torch.randn(5, T, dtype=torch.float64)
    out = _filter(x, "movavg:7")
    ref = uniform_filter1d(x.numpy(), 7, axis=1, mode="nearest")
    np.testing.assert_allclose(out.numpy(), ref, atol=1e-12)


def test_float32_in_float32_out():
    x = torch.randn(30, T)
    out = _filter(x, "bandpass:0.01,0.1")
    assert out.dtype == torch.float32 and out.shape == x.shape


@pytest.mark.parametrize(
    "spec", ["lowpass", "lowpass:-1", "bandpass:0.1,0.01", "movavg:4", "notch:0.1"]
)
def test_bad_specs_are_rejected(spec):
    with pytest.raises(ValueError, match="temporal filter must be"):
        parse_temporal_filter(spec)


def test_edges_at_or_above_nyquist_are_rejected():
    with pytest.raises(ValueError, match="Nyquist"):
        _filter(torch.zeros(1, T), "lowpass:0.5")
