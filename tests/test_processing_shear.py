"""Shear-based rigid resampling matches the general affine resampler.

The shear path (AFNI's THD_rota_vol method) must reproduce the same pull
mapping as resample_affine_fast for small rigid rotations + shifts, which is
the regime motion correction operates in.
"""

import math

import torch

from fastfuncstuff.processing.affine import (
    _build_homo_coords,
    params_to_matrix,
    resample_affine_fast,
)
from fastfuncstuff.processing.shear import rigid_matrix_to_shears, shear_resample


def _make_volume(nz=24, ny=30, nx=28, seed=0):
    g = torch.Generator().manual_seed(seed)
    # smooth-ish blob so interpolation differences are meaningful but bounded
    vol = torch.rand(nz, ny, nx, generator=g)
    # low-pass via a couple of box blurs
    for _ in range(3):
        vol = (
            vol
            + torch.roll(vol, 1, 0)
            + torch.roll(vol, -1, 0)
            + torch.roll(vol, 1, 1)
            + torch.roll(vol, -1, 1)
            + torch.roll(vol, 1, 2)
            + torch.roll(vol, -1, 2)
        ) / 7.0
    return vol


def _rigid_matrix(dx, dy, dz, rz, rx, ry):
    p = torch.zeros(12, dtype=torch.float32)
    p[0], p[1], p[2] = dx, dy, dz
    p[3], p[4], p[5] = rz, rx, ry
    p[6] = p[7] = p[8] = 1.0  # scales
    return params_to_matrix(p)


def test_shear_matches_affine_small_rotation_heptic():
    vol = _make_volume()
    shape = vol.shape
    coords = _build_homo_coords(shape, vol.device, vol.dtype)

    # realistic moco motion: all 6 params nonzero (degrees, voxels). Real
    # GN-fit params are never exactly axis-aligned, so the shear decomposition
    # is always well-conditioned here.
    cases = [
        (0.5, -0.3, 0.2, 1.0, -0.7, 0.4),
        (-1.2, 0.8, 0.4, 0.6, 1.5, -1.1),
        (1.0, 0.5, -1.0, 0.5, 0.5, 0.5),
        (2.0, -1.5, 1.0, -1.3, 0.9, 1.4),
    ]
    for c in cases:
        M = _rigid_matrix(*c)
        ref = resample_affine_fast(vol, M, coords, "heptic", shape, zero_outside=True)
        out = shear_resample(vol, M, shape, "heptic")
        assert out is not None, f"shear decomposition invalid for {c}"

        # compare on the interior (edges differ: shear zero-fills row-by-row,
        # the direct sampler zero-fills on the final 3D coordinate)
        interior = (slice(6, -6), slice(6, -6), slice(6, -6))
        a = out[interior]
        b = ref[interior]
        denom = b.abs().mean().clamp(min=1e-6)
        rel = (a - b).abs().mean() / denom
        assert rel < 0.02, f"case {c}: relative interior error {rel:.4f}"


def test_pitch_dominated_rotation_not_corrupted():
    """Pitch-dominated motion must not decompose into a spurious rotation.

    Regression: a rotation dominated by one axis (pitch) with tiny components on
    the others — exactly what a GN fit of pitch-dominated motion produces — drove
    the closed-form xzyx factorization into a regime where float32 loses ~7 digits
    in its cube-root/division chain, yielding a plan that passed the exact-zero
    validity guards yet applied a transform 10-100% off the request (a phantom
    pitch in the corrected image). Decomposing in float64 (as AFNI does) keeps
    these on the fast shear path *and* accurate. We shrink the off-axis terms
    toward the exact-degenerate limit; every well-posed case must stay valid and
    match the general resampler.
    """
    vol = _make_volume(seed=7)
    shape = vol.shape
    coords = _build_homo_coords(shape, vol.device, vol.dtype)
    interior = (slice(6, -6), slice(6, -6), slice(6, -6))

    for eps in (1e-2, 1e-3, 1e-4, 1e-5):
        M = _rigid_matrix(0.3, 0.2, 0.4, eps, 0.4, eps)  # rx=0.4 dominates
        ax, scl, sft, valid = rigid_matrix_to_shears(M, shape)
        assert bool(valid.all()), f"pitch-dominated eps={eps} wrongly flagged invalid"

        out = shear_resample(vol, M, shape, "heptic")
        ref = resample_affine_fast(vol, M, coords, "heptic", shape, zero_outside=True)
        a, b = out[interior], ref[interior]
        rel = (a - b).abs().mean() / b.abs().mean().clamp(min=1e-6)
        assert rel < 0.01, f"pitch-dominated eps={eps}: relative error {rel:.4f}"


def test_pure_axis_rotation_falls_back():
    # exactly axis-aligned rotations are degenerate for every xzyx ordering;
    # shear_resample returns None so the caller uses a general resample.
    vol = _make_volume()
    M = _rigid_matrix(0, 0, 0, 2.0, 0, 0)
    assert shear_resample(vol, M, vol.shape, "heptic") is None


def test_identity_is_near_exact():
    vol = _make_volume(seed=3)
    shape = vol.shape
    M = _rigid_matrix(0, 0, 0, 0, 0, 0)
    out = shear_resample(vol, M, shape, "heptic")
    assert out is not None
    interior = (slice(4, -4), slice(4, -4), slice(4, -4))
    assert torch.allclose(out[interior], vol[interior], atol=1e-4)


def test_decomposition_validity_mask_batched():
    shape = (24, 30, 28)
    Ms = torch.stack(
        [
            _rigid_matrix(0.3, -0.2, 0.1, 0.8, -0.5, 0.6),
            _rigid_matrix(0, 0, 0, 0, 0, 0),
        ]
    )
    _, _, _, valid = rigid_matrix_to_shears(Ms, shape)
    assert valid.shape == (2,)
    assert bool(valid.all())


def test_batched_moco_estimation_cuda():
    """Whole-batch shear GN estimation runs and tracks the per-volume path.

    CUDA-only (Triton). Builds known small motions, checks the corrected series
    aligns to base and that batched params agree with the per-volume solver to
    within interpolation-method tolerance (shear vs full-3D).
    """
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA/Triton required")
    from fastfuncstuff.processing.ffs_moco import MocoConfig, moco
    from fastfuncstuff.processing.shear_triton import shear_resample_triton

    torch.manual_seed(2)
    nt, nz, ny, nx = 16, 36, 48, 48
    base = _make_volume(nz, ny, nx, seed=5)
    truth = torch.zeros(nt, 12)
    truth[:, :3] = torch.randn(nt, 3) * 1.2  # clearly-detectable motion (vox)
    truth[:, 3:6] = torch.randn(nt, 3) * 1.5  # degrees
    truth[:, 6:9] = 1.0
    mats = torch.stack([params_to_matrix(truth[i]) for i in range(nt)]).cuda()
    ts = shear_resample_triton(
        base[None].expand(nt, nz, ny, nx).contiguous().cuda(), mats, (nz, ny, nx), "heptic"
    )[0].cpu()

    common = dict(
        device="cuda", verb=0, interp="heptic", final_interp="heptic", base_index=0, compile=False
    )
    rB = moco(ts, MocoConfig(use_shear=True, **common))  # batched shear path
    rP = moco(ts, MocoConfig(use_shear=False, **common))  # per-volume full-3D path

    assert torch.isfinite(rB.aligned).all()
    interior = (slice(6, -6), slice(6, -6), slice(6, -6))
    base_i = base[interior]
    err_aligned = (rB.aligned[1:, *interior] - base_i).abs().mean()
    err_raw = (ts[1:, *interior] - base_i).abs().mean()
    err_pervol = (rP.aligned[1:, *interior] - base_i).abs().mean()
    # registration reduces error, and the batched (shear) path is as good as the
    # validated per-volume (full-3D) path
    assert err_aligned < err_raw, f"{err_aligned:.4f} vs raw {err_raw:.4f}"
    assert err_aligned <= err_pervol * 1.2, f"batched {err_aligned:.4f} vs pervol {err_pervol:.4f}"
    # batched vs per-volume params agree to interpolation-method tolerance
    import numpy as np

    assert np.abs(rB.params - rP.params).mean() < 0.1
