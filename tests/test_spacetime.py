"""Tests for joint space-time (slice-timing + motion) resampling.

The load-bearing check: with an identity spatial map the joint sampler must
reduce to a plain per-slice temporal shift -- i.e. exactly slice-timing
correction (3dTshift). That pins the temporal-coordinate math and sign
convention without any file IO.
"""

from __future__ import annotations

import math

import pytest
import torch

from fastfuncstuff.processing.spacetime import (
    apply_spacetime_sample,
    interp_slice_times,
    temporal_kernel_weights,
)


def _identity_coords(nz: int, ny: int, nx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    return ii, jj, kk  # sx, sy, sz


def _linear_interp_series(series: torch.Tensor, t: float) -> float:
    """Reference: linearly interpolate a 1-D series at fractional index t (edge-clamp)."""
    n = series.shape[0]
    tc = min(max(t, 0.0), n - 1.0)
    i0 = int(math.floor(tc))
    i0 = min(i0, n - 2)
    frac = tc - i0
    return float(series[i0] * (1.0 - frac) + series[i0 + 1] * frac)


def test_temporal_kernel_weights_support_and_symmetry():
    d = torch.linspace(-10, 10, 401)
    for method, half in (
        ("linear", 1),
        ("cubic", 2),
        ("quintic", 3),
        ("heptic", 4),
        ("wsinc5", 5),
        ("wsinc9", 9),
    ):
        w = temporal_kernel_weights(d, method)
        # zero outside the support
        assert torch.all(w[d.abs() >= half] == 0.0)
        # symmetric kernel
        wr = temporal_kernel_weights(-d, method)
        assert torch.allclose(w, wr, atol=1e-6)
        # interpolating: unit at the sample point, zero at other integer nodes
        assert abs(float(temporal_kernel_weights(torch.tensor([0.0]), method)) - 1.0) < 1e-6
        for k in range(1, half):
            assert abs(float(temporal_kernel_weights(torch.tensor([float(k)]), method))) < 1e-6


@pytest.mark.parametrize("method,half", [("quintic", 3), ("heptic", 4)])
def test_polynomial_temporal_weights_form_partition_of_unity(method, half):
    for tcoord in torch.linspace(-0.49, 0.49, 19):
        base = int(torch.floor(tcoord))
        frames = torch.arange(base - (half - 1), base + half + 1, dtype=torch.float32)
        weights = temporal_kernel_weights(tcoord - frames, method)
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0), atol=2e-6, rtol=0)


def test_interp_slice_times_linear():
    st = torch.tensor([0.0, 1.0, 2.0, 3.0])
    sz = torch.tensor([0.0, 0.5, 1.5, 3.0, 3.7, -1.0])
    got = interp_slice_times(sz, st)
    # fractional slices interpolate; out-of-range clamps to the ends
    exp = torch.tensor([0.0, 0.5, 1.5, 3.0, 3.0, 0.0])
    assert torch.allclose(got, exp, atol=1e-6)


def test_single_slice_constant_offset():
    st = torch.tensor([0.5])  # 1 slice
    sz = torch.rand(2, 3, 4) * 5.0
    got = interp_slice_times(sz, st)
    assert torch.allclose(got, torch.full_like(sz, 0.5))


def test_reduces_to_linear_slice_timing_shift():
    torch.manual_seed(0)
    nt, nz, ny, nx = 30, 5, 3, 3
    tr = 2.0
    # Smooth band-limited series per slice so linear interpolation is well-posed.
    t = torch.arange(nt, dtype=torch.float32)
    freqs = torch.rand(nz, ny, nx) * 0.15 + 0.02
    phase = torch.rand(nz, ny, nx) * 6.28
    source = torch.sin(freqs[None] * t[:, None, None, None] * 2 * math.pi + phase[None]) + 5.0

    # Ascending slice offsets across the TR (interleaved-agnostic: explicit times).
    slice_times = [k * tr / nz for k in range(nz)]
    st_t = torch.tensor(slice_times, dtype=torch.float32)
    tzero = float(sum(slice_times) / len(slice_times))

    sx, sy, sz = _identity_coords(nz, ny, nx)

    for j in range(nt):
        out = apply_spacetime_sample(
            source, sx, sy, sz, j, tr, tzero, st_t, tinterp="linear", interp="linear"
        )
        # Reference: each slice shifted by its own uniform temporal offset.
        for k in range(nz):
            tcoord = j + (tzero - slice_times[k]) / tr
            for yy in range(ny):
                for xx in range(nx):
                    expected = _linear_interp_series(source[:, k, yy, xx], tcoord)
                    assert abs(float(out[k, yy, xx]) - expected) < 1e-4


def test_zero_timing_is_spatial_passthrough():
    """All slices at tzero -> no temporal shift; identity spatial -> source[j]."""
    torch.manual_seed(1)
    nt, nz, ny, nx = 8, 4, 3, 3
    source = torch.rand(nt, nz, ny, nx) + 1.0
    slice_times = [0.3, 0.3, 0.3, 0.3]
    st_t = torch.tensor(slice_times)
    sx, sy, sz = _identity_coords(nz, ny, nx)
    for j in range(nt):
        out = apply_spacetime_sample(
            source, sx, sy, sz, j, 2.0, 0.3, st_t, tinterp="cubic", interp="linear"
        )
        assert torch.allclose(out, source[j], atol=1e-4)


def test_cubic_runs_finite_and_close_to_linear():
    torch.manual_seed(2)
    nt, nz, ny, nx = 20, 4, 3, 3
    tr = 1.5
    t = torch.arange(nt, dtype=torch.float32)
    source = torch.sin(0.1 * t[:, None, None, None] * 2 * math.pi) + torch.rand(nt, nz, ny, nx) * 0
    source = source.expand(nt, nz, ny, nx).contiguous() + 5.0
    slice_times = [k * tr / nz for k in range(nz)]
    st_t = torch.tensor(slice_times)
    tzero = float(sum(slice_times) / len(slice_times))
    sx, sy, sz = _identity_coords(nz, ny, nx)
    for j in (5, 10, 15):
        cub = apply_spacetime_sample(
            source, sx, sy, sz, j, tr, tzero, st_t, tinterp="cubic", interp="linear"
        )
        lin = apply_spacetime_sample(
            source, sx, sy, sz, j, tr, tzero, st_t, tinterp="linear", interp="linear"
        )
        assert torch.isfinite(cub).all()
        # Smooth low-freq signal: cubic and linear agree closely.
        assert (cub - lin).abs().max() < 0.05


def test_wsinc_no_neg_clamps_after_signed_temporal_accumulation():
    nt, nz, ny, nx = 20, 2, 1, 1
    source = torch.zeros(nt, nz, ny, nx)
    source[8] = 1.0
    sx, sy, sz = _identity_coords(nz, ny, nx)
    st = torch.tensor([0.0, 0.5])

    plain = apply_spacetime_sample(
        source, sx, sy, sz, 10, 1.0, 0.25, st, tinterp="wsinc5", interp="linear"
    )
    clamped = apply_spacetime_sample(
        source,
        sx,
        sy,
        sz,
        10,
        1.0,
        0.25,
        st,
        tinterp="wsinc5",
        interp="linear",
        no_neg=True,
    )

    assert plain.min() < 0.0
    assert clamped.min() >= 0.0
