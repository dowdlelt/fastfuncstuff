"""Tests for the optical-flow nonlinear registration engine (ffs_optiwarp)."""

from __future__ import annotations

import math

import pytest
import torch

from fastfuncstuff.processing.interp import warp_image_linear
from fastfuncstuff.processing.optiwarp import (
    NO_X_DISP,
    NO_Z_DISP,
    OptiwarpConfig,
    _exp_field,
    _grad3,
    jacobian_determinant,
    optiwarp,
    prep_intensity,
)


def _blobs(shape=(32, 40, 40), seed=0) -> torch.Tensor:
    """A textured phantom: enough structure at small scales for flow to lock onto."""
    from fastfuncstuff.processing.cost import _separable_smooth_3d

    g = torch.Generator().manual_seed(seed)
    vol = _separable_smooth_3d(torch.randn(shape, generator=g), 1.5)
    vol = (vol - vol.mean()) / vol.std()
    # A soft ellipsoid envelope so the phantom has a brain-like support.
    kk, jj, ii = torch.meshgrid(*[torch.linspace(-1, 1, n) for n in shape], indexing="ij")
    env = torch.exp(-1.5 * (kk**2 + jj**2 + ii**2))
    return (vol + 1.5) * env


def _smooth_field(shape, amp, seed=1):
    """A smooth analytic displacement field with peak magnitude bounded by ~``amp``.

    Deliberately not band-limited noise: a smoothed random field has a max/RMS ratio
    around 7, so scaling one to a plausible RMS leaves peaks of 15+ voxels — a tear,
    not the subtle warp this tool targets. Products of sines are bounded by
    construction and vanish at the boundary, where the field has no support anyway.
    """
    nz, ny, nx = shape
    kk, jj, ii = torch.meshgrid(
        torch.linspace(0, math.pi, nz),
        torch.linspace(0, math.pi, ny),
        torch.linspace(0, math.pi, nx),
        indexing="ij",
    )
    phase = 0.7 * seed
    env = torch.sin(kk) * torch.sin(jj) * torch.sin(ii)
    xd = amp * env * torch.cos(2 * jj + phase)
    yd = amp * env * torch.cos(2 * ii + 1.1 + phase)
    zd = amp * env * torch.cos(2 * kk + 2.2 + phase)
    return xd, yd, zd


def test_grad3_matches_analytic_ramp():
    """Gradient components map to the (x, y, z) = (last, middle, first) axis order."""
    nz, ny, nx = 6, 7, 8
    kk, jj, ii = torch.meshgrid(
        torch.arange(nz).float(), torch.arange(ny).float(), torch.arange(nx).float(), indexing="ij"
    )
    gx, gy, gz = _grad3(3.0 * ii + 5.0 * jj - 2.0 * kk)
    inner = (slice(1, -1),) * 3
    assert torch.allclose(gx[inner], torch.full_like(gx[inner], 3.0))
    assert torch.allclose(gy[inner], torch.full_like(gy[inner], 5.0))
    assert torch.allclose(gz[inner], torch.full_like(gz[inner], -2.0))


def test_exp_field_of_constant_velocity_is_that_displacement():
    """exp() of a constant velocity field is a pure translation of the same size."""
    shape = (12, 12, 12)
    v = (torch.full(shape, 1.7), torch.zeros(shape), torch.zeros(shape))
    ex, ey, ez = _exp_field(v)
    inner = (slice(2, -2),) * 3
    assert torch.allclose(ex[inner], torch.full_like(ex[inner], 1.7), atol=1e-4)
    assert ey.abs().max() < 1e-5
    assert ez.abs().max() < 1e-5


def test_jacobian_determinant_of_identity_is_one():
    zeros = torch.zeros(8, 8, 8)
    jac = jacobian_determinant(zeros, zeros, zeros)
    assert torch.allclose(jac, torch.ones_like(jac))


def test_jacobian_determinant_detects_expansion():
    """A field that stretches space has determinant != 1 in the stretched direction."""
    nz, ny, nx = 10, 10, 10
    _, _, ii = torch.meshgrid(
        torch.arange(nz).float(), torch.arange(ny).float(), torch.arange(nx).float(), indexing="ij"
    )
    # xd = 0.2*i means the sampled source coordinate advances 1.2 per output voxel:
    # a compression of the source, det = 1.2.
    jac = jacobian_determinant(0.2 * ii, torch.zeros_like(ii), torch.zeros_like(ii))
    inner = (slice(1, -1),) * 3
    assert torch.allclose(jac[inner], torch.full_like(jac[inner], 1.2), atol=1e-5)


def test_prep_localnorm_removes_a_bias_field():
    """Local z-scoring must undo a smooth multiplicative gain, which is its whole job."""
    vol = _blobs()
    kk, jj, ii = torch.meshgrid(*[torch.linspace(0, 1, n) for n in vol.shape], indexing="ij")
    bias = 1.0 + 1.5 * (ii + jj + kk) / 3.0
    core = (slice(8, -8),) * 3
    a = prep_intensity(vol, "localnorm", 6.0)[core].reshape(-1)
    b = prep_intensity(vol * bias + 20.0, "localnorm", 6.0)[core].reshape(-1)
    assert torch.corrcoef(torch.stack([a, b]))[0, 1] > 0.98


def test_prep_localnorm_inverts_under_contrast_inversion():
    """localnorm cannot fix a contrast flip — it negates. gradmag is the mode that can.

    Bug of record this pins down: an inverted-contrast pair prepped with localnorm has
    the flow force pointing backwards everywhere, so the default match mode is *not*
    the cross-modal answer even though it is the cross-session one.
    """
    vol = _blobs()
    inverted = -2.0 * vol + 100.0
    core = (slice(8, -8),) * 3

    def corr(mode):
        a = prep_intensity(vol, mode, 6.0)[core].reshape(-1)
        b = prep_intensity(inverted, mode, 6.0)[core].reshape(-1)
        return torch.corrcoef(torch.stack([a, b]))[0, 1]

    assert corr("localnorm") < -0.95  # negated map: the force would run in reverse
    assert corr("gradmag") > 0.99  # edge magnitude is sign-free


@pytest.mark.parametrize("force", ["demons", "lk", "hs"])
def test_recovers_a_known_smooth_warp(force):
    """The headline check: warp a phantom by a known field, then recover it.

    Each flow model must both (a) improve the image match a lot and (b) get the
    displacement field itself close to truth — matching the image while inventing the
    wrong deformation is the failure mode a correlation-only test would miss.
    """
    fixed = _blobs()
    truth = _smooth_field(fixed.shape, amp=2.5)
    moving_from_fixed = warp_image_linear(fixed, *truth)

    # `fixed` warped by `truth` is the moving image seen through the warp we want to
    # recover: registering that image back to `fixed` should return ~truth.
    cfg = OptiwarpConfig(
        force=force,
        match="none",
        shrink_factors=(4, 2, 1),
        smoothing_sigmas=(2.0, 1.0, 0.0),
        iterations=(60, 60, 60),
        metric="pearson",
        verb=0,
    )
    res = optiwarp(moving_from_fixed, fixed, config=cfg)

    core = (slice(6, -6),) * 3
    before = torch.corrcoef(
        torch.stack([moving_from_fixed[core].reshape(-1), fixed[core].reshape(-1)])
    )[0, 1]
    after = torch.corrcoef(
        torch.stack([res.warped[core].reshape(-1), moving_from_fixed[core].reshape(-1)])
    )[0, 1]
    assert after > before
    assert after > 0.95

    # Field accuracy: residual displacement error well under the truth's amplitude.
    err = torch.sqrt(sum((res.fwd[i][core] - truth[i][core]) ** 2 for i in range(3))).mean()
    truth_mag = torch.sqrt(sum(truth[i][core] ** 2 for i in range(3))).mean()
    assert err < 0.4 * truth_mag, f"{force}: mean field error {err:.3f} vs truth {truth_mag:.3f}"


def test_diffeo_step_keeps_the_field_foldless():
    """The diffeomorphic step exists to stop optical flow from tearing the image."""
    fixed = _blobs()
    truth = _smooth_field(fixed.shape, amp=3.0, seed=7)
    moving = warp_image_linear(fixed, *truth)

    cfg = OptiwarpConfig(
        step_mode="diffeo",
        match="none",
        total_sigma=0.5,
        shrink_factors=(2, 1),
        smoothing_sigmas=(1.0, 0.0),
        iterations=(50, 50),
        metric="pearson",
        verb=0,
    )
    res = optiwarp(moving, fixed, config=cfg)
    assert res.min_jacobian > 0.0
    assert math.isfinite(res.cost)


def test_axis_flags_zero_the_constrained_components():
    """-noXdis/-noZdis must hold exactly, so PE-only distortion fits stay 1-D."""
    fixed = _blobs()
    truth = _smooth_field(fixed.shape, amp=2.0, seed=3)
    moving = warp_image_linear(fixed, *truth)

    cfg = OptiwarpConfig(
        match="none",
        warp_flags=NO_X_DISP | NO_Z_DISP,
        shrink_factors=(2, 1),
        smoothing_sigmas=(1.0, 0.0),
        iterations=(30, 30),
        metric="pearson",
        verb=0,
    )
    res = optiwarp(moving, fixed, config=cfg)
    assert res.fwd[0].abs().max() == 0.0
    assert res.fwd[2].abs().max() == 0.0
    assert res.fwd[1].abs().max() > 0.1


def test_inverse_field_undoes_the_forward():
    """-save_inverse must actually invert, not just negate."""
    fixed = _blobs()
    truth = _smooth_field(fixed.shape, amp=2.0, seed=11)
    moving = warp_image_linear(fixed, *truth)

    cfg = OptiwarpConfig(
        match="none",
        shrink_factors=(2, 1),
        smoothing_sigmas=(1.0, 0.0),
        iterations=(40, 40),
        metric="pearson",
        verb=0,
    )
    res = optiwarp(moving, fixed, config=cfg)

    # Composing forward with inverse should land back near the identity.
    from fastfuncstuff.processing.optiwarp import _compose

    cx, cy, cz = _compose(res.fwd, res.inv)
    core = (slice(6, -6),) * 3
    resid = torch.sqrt(cx[core] ** 2 + cy[core] ** 2 + cz[core] ** 2).mean()
    assert resid < 0.25


def test_cross_contrast_pair_registers_with_gradmag():
    """End-to-end cross-modal case: opposite contrast, recovered via match='gradmag'."""
    fixed = _blobs()
    truth = _smooth_field(fixed.shape, amp=1.5, seed=5)
    warped_fixed = warp_image_linear(fixed, *truth)
    inverted = -2.0 * warped_fixed + 5.0  # same anatomy, opposite contrast

    cfg = OptiwarpConfig(
        match="gradmag",
        shrink_factors=(4, 2, 1),
        smoothing_sigmas=(2.0, 1.0, 0.0),
        iterations=(60, 60, 60),
        metric="pearson",
        verb=0,
    )
    res = optiwarp(inverted, fixed, config=cfg)

    core = (slice(6, -6),) * 3
    err = torch.sqrt(sum((res.fwd[i][core] - truth[i][core]) ** 2 for i in range(3))).mean()
    truth_mag = torch.sqrt(sum(truth[i][core] ** 2 for i in range(3))).mean()
    assert err < 0.5 * truth_mag, f"field error {err:.3f} vs truth {truth_mag:.3f}"
