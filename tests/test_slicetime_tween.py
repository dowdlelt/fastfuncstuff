"""Tests for ffs_slicetime -tween (inter-frame midpoint resampling).

The tween mode ignores slice timing and resamples a series onto the points between
consecutive frames: output frame k = the sample halfway between input frames k and
k+1. Its default (linear) makes each output the plain average of its two neighbours.
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.slicetime import tween_midpoints


def _series(nt=8, nz=3, ny=4, nx=5, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((nt, nz, ny, nx)).astype(np.float32))


def test_linear_tween_is_neighbour_average():
    """method='linear' → each midpoint frame is exactly the mean of its two neighbours."""
    vol = _series()
    nt = vol.shape[0]
    out = tween_midpoints(vol, method="linear", device=torch.device("cpu"))
    assert out.shape == (nt - 1, *vol.shape[1:])  # nt-1 midpoints, spatial dims unchanged
    expected = 0.5 * (vol[:-1] + vol[1:])
    assert torch.allclose(out, expected, atol=1e-5)


def test_tween_recovers_linear_ramp_exactly():
    """A signal linear in time is exact at the midpoints for any reasonable interp."""
    nt, nz, ny, nx = 10, 2, 2, 2
    slope = torch.arange(nz * ny * nx, dtype=torch.float32).reshape(nz, ny, nx)
    t = torch.arange(nt, dtype=torch.float32)
    vol = t[:, None, None, None] * slope[None] + 3.0  # value at time t = slope*t + 3
    for method in ("linear", "cubic", "fourier"):
        out = tween_midpoints(vol, method=method, device=torch.device("cpu"))
        # midpoint of frames k,k+1 sits at time k+0.5 → slope*(k+0.5)+3
        mids = (t[:-1] + 0.5)[:, None, None, None] * slope[None] + 3.0
        # fourier has mild circular-edge effects; check the interior it nails exactly.
        assert torch.allclose(out[1:-1], mids[1:-1], atol=1e-3), method


def test_tween_needs_two_frames():
    with pytest.raises(ValueError):
        tween_midpoints(_series(nt=1), device=torch.device("cpu"))


def test_tween_default_method_is_linear():
    """The library default is the neighbour-average (linear) path."""
    vol = _series(seed=1)
    default = tween_midpoints(vol, device=torch.device("cpu"))
    linear = tween_midpoints(vol, method="linear", device=torch.device("cpu"))
    assert torch.allclose(default, linear, atol=1e-6)
