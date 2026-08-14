"""Joint slice-timing over complex (phase) data.

The space-time samplers are linear in source values, so ``ffs_nwarp`` carries
phase through slice-timing by warping the complex channels and recombining. These
tests pin two things: (1) the multi-channel sampler is exactly N independent
single-channel samples sharing one space-time map, and (2) a spatially/temporally
constant phase survives the round trip (magnitude unchanged, phase preserved) --
which would have caught a channel mix-up or an accidental ``no_neg`` clamp on the
sign-bearing real/imag channels.
"""

from __future__ import annotations

import torch

from fastfuncstuff.processing.nwarpforge import _phase_spacetime_channels
from fastfuncstuff.processing.spacetime import (
    TissueFollowingSampler,
    apply_spacetime_sample,
)

DEV = torch.device("cpu")
NT, SNZ, SNY, SNX = 8, 6, 5, 5


def _coords(frame_shift: float = 0.3):
    """Absolute source coords for one output frame: identity grid + a small shift."""
    kk, jj, ii = torch.meshgrid(
        torch.arange(SNZ, dtype=torch.float32),
        torch.arange(SNY, dtype=torch.float32),
        torch.arange(SNX, dtype=torch.float32),
        indexing="ij",
    )
    return ii + frame_shift, jj - 0.2, kk + 0.1


def _slice_times():
    return torch.linspace(0.0, 0.9, SNZ, dtype=torch.float32)


def test_multi_channel_matches_per_channel_apply():
    torch.manual_seed(0)
    chans = [torch.randn(NT, SNZ, SNY, SNX) for _ in range(3)]
    sx, sy, sz = _coords()
    st = _slice_times()
    kw = dict(tinterp="cubic", interp="wsinc5")

    multi = apply_spacetime_sample(chans, sx, sy, sz, 4, 1.0, 0.45, st, **kw)
    assert isinstance(multi, list) and len(multi) == 3
    for c in range(3):
        single = apply_spacetime_sample(chans[c], sx, sy, sz, 4, 1.0, 0.45, st, **kw)
        assert torch.allclose(multi[c], single, atol=1e-6)


def test_multi_channel_matches_per_channel_follow():
    torch.manual_seed(1)
    chans = [torch.randn(NT, SNZ, SNY, SNX) for _ in range(2)]
    st = _slice_times()

    def coords_fn(_f):
        return _coords()

    common = dict(
        tr=1.0,
        tzero=0.45,
        slice_times=st,
        device=DEV,
        tinterp="cubic",
        interp="wsinc5",
    )
    shape = (SNZ, SNY, SNX)
    multi_sampler = TissueFollowingSampler(chans, coords_fn, shape, **common)
    out = multi_sampler.sample(4)
    assert isinstance(out, list) and len(out) == 2
    for c in range(2):
        solo = TissueFollowingSampler(chans[c], coords_fn, shape, **common)
        assert torch.allclose(out[c], solo.sample(4), atol=1e-6)


def test_follow_warped_tap_cache_matches_recomputation():
    """Overlapping output windows may reuse own-pose taps without changing values."""
    torch.manual_seed(11)
    source = torch.randn(NT, SNZ, SNY, SNX)
    st = _slice_times()

    def coords_fn(f):
        return _coords(frame_shift=0.03 * f)

    common = dict(
        tr=1.0,
        tzero=0.45,
        slice_times=st,
        device=DEV,
        tinterp="cubic",
        interp="cubic",
    )
    cached = TissueFollowingSampler(source, coords_fn, (SNZ, SNY, SNX), **common)
    recomputed = TissueFollowingSampler(source, coords_fn, (SNZ, SNY, SNX), **common)
    recomputed.cache_warped = False

    cached_out = [cached.sample(f) for f in range(NT)]
    recomputed_out = [recomputed.sample(f) for f in range(NT)]

    for actual, reference in zip(cached_out, recomputed_out, strict=True):
        assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-6)
    assert cached._warped
    assert len(cached._warped) <= 2 * cached.half + 2


def test_no_neg_is_per_channel():
    # A negative channel must NOT be clamped when its no_neg flag is False, even
    # though a sibling channel requests clamping.
    sx, sy, sz = _coords(frame_shift=0.0)
    st = _slice_times()
    neg = -torch.ones(NT, SNZ, SNY, SNX)
    pos = torch.ones(NT, SNZ, SNY, SNX)
    out = apply_spacetime_sample([pos, neg], sx, sy, sz, 4, 1.0, 0.45, st, no_neg=[True, False])
    assert out[1].min() < -0.5  # untouched negative channel
    assert out[0].min() >= 0.0  # clamped channel


def test_constant_phase_survives_roundtrip():
    # Constant magnitude field, spatially+temporally constant phase c: the complex
    # recombination must return the magnitude unchanged and phase == c.
    torch.manual_seed(2)
    mag = torch.rand(NT, SNZ, SNY, SNX) + 0.5  # strictly positive magnitude
    c = 0.7
    phase = torch.full((NT, SNZ, SNY, SNX), c)
    chans, no_neg, recombine = _phase_spacetime_channels(mag, phase, "complex", False)

    sx, sy, sz = _coords()
    st = _slice_times()
    warped = apply_spacetime_sample(chans, sx, sy, sz, 4, 1.0, 0.45, st, no_neg=no_neg)
    mag_out, phase_out = recombine(warped)

    mag_only = apply_spacetime_sample(mag, sx, sy, sz, 4, 1.0, 0.45, st)
    assert torch.allclose(mag_out, mag_only, atol=1e-5)
    assert torch.allclose(phase_out, torch.full_like(phase_out, c), atol=1e-5)
