"""Shear-based rigid resampling matches the general affine resampler.

The shear path (AFNI's THD_rota_vol method) must reproduce the same pull
mapping as resample_affine_fast for small rigid rotations + shifts, which is
the regime motion correction operates in.
"""

import pytest
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


def test_pure_axis_rotation_takes_the_paeth_path():
    # Exactly axis-aligned rotations are degenerate for every xzyx ordering, so
    # the 4-shear plan is invalid -- but a plane rotation needs only 3 shears.
    # Accuracy is pinned by test_single_axis_rotation_shears_like_its_neighbour.
    vol = _make_volume()
    M = _rigid_matrix(0, 0, 0, 2.0, 0, 0)
    assert not bool(rigid_matrix_to_shears(M, vol.shape)[3].all())
    assert shear_resample(vol, M, vol.shape, "heptic") is not None


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


@pytest.mark.gpu
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


# ---------------------------------------------------------------------------
# The six axis permutations are evaluated in one batched _shear_xzyx call.
# Per-call torch dispatch dominates that factorization (it is hundreds of
# elementwise ops on B lanes), so calling it six times cost 24.8s of a 104.4s
# ffs_moco CPU run. These pin that the batching is a pure speed change.
# ---------------------------------------------------------------------------


def _random_rigid(seed: int, batch: int = 1):
    """(B,3,3) rotation + (B,3) shift, at the small angles real moco produces."""
    g = torch.Generator().manual_seed(seed)
    mats, shifts = [], []
    for _ in range(batch):
        a = torch.randn(3, generator=g, dtype=torch.float64) * 0.03
        cx, sx = torch.cos(a[0]), torch.sin(a[0])
        cy, sy = torch.cos(a[1]), torch.sin(a[1])
        cz, sz = torch.cos(a[2]), torch.sin(a[2])
        rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float64)
        ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float64)
        rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float64)
        mats.append(rz @ ry @ rx)
        shifts.append(torch.randn(3, generator=g, dtype=torch.float64) * 0.5)
    return torch.stack(mats), torch.stack(shifts)


def test_shear_best_lanes_are_independent():
    """A batched call must decompose each matrix the way a lone call does.

    The six permutations are stacked along the batch, so a reduction that leaked
    across lanes would show up here and nowhere else. The PLAN (which axis order,
    and whether the lane is usable) must match exactly; the coefficients agree to
    float64 rounding, since BLAS blocks a 5-lane batch differently from a 1-lane
    one -- that was true of the per-permutation loop too (measured 9.3e-15).
    """
    from fastfuncstuff.processing.shear import _shear_best

    q, d = _random_rigid(7, batch=5)
    ax, scl, sft, valid = _shear_best(q, d)
    for i in range(q.shape[0]):
        ax_i, scl_i, sft_i, valid_i = _shear_best(q[i : i + 1], d[i : i + 1])
        assert torch.equal(ax[i : i + 1], ax_i)
        assert torch.equal(valid[i : i + 1], valid_i)
        assert torch.allclose(scl[i : i + 1], scl_i, rtol=0, atol=1e-12)
        assert torch.allclose(sft[i : i + 1], sft_i, rtol=0, atol=1e-12)


def test_batched_decomposition_still_reproduces_the_pull():
    """rigid_matrix_to_shears only marks a lane valid once it has checked that
    the plan composes back to the requested matrix, so an all-valid batch of
    honest rigid transforms is that safety net passing on the batched path."""
    shape = (24, 32, 32)
    q, d = _random_rigid(11, batch=4)
    matrices = torch.eye(4, dtype=torch.float64).repeat(4, 1, 1)
    matrices[:, :3, :3] = q
    matrices[:, :3, 3] = d
    ax, scl, sft, valid = rigid_matrix_to_shears(matrices, shape)
    assert bool(valid.all())
    for i in range(4):
        ax_i, scl_i, sft_i, valid_i = rigid_matrix_to_shears(matrices[i : i + 1], shape)
        assert torch.equal(ax[i : i + 1], ax_i)
        assert torch.equal(valid[i : i + 1], valid_i)
        assert torch.allclose(scl[i : i + 1], scl_i, rtol=0, atol=1e-10)


def test_degenerate_matrix_is_still_reported_invalid():
    """A singular input must not be rescued by the batching."""
    from fastfuncstuff.processing.shear import _shear_best

    q = torch.zeros(1, 3, 3, dtype=torch.float64)
    d = torch.zeros(1, 3, dtype=torch.float64)
    _, _, _, valid = _shear_best(q, d)
    assert not bool(valid.all())


@pytest.mark.parametrize("mode", ["nn", "linear", "cubic", "quintic", "heptic", "wsinc5"])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_rowconv_shear_matches_the_tap_loop(mode, dim):
    """The CPU row-conv pass must equal the reference tap loop, edges included.

    Its pad has to cover a whole kernel width: at H+1 a window starting just
    outside the row clamped onto real voxels and the edge rows came out wrong.
    """
    from fastfuncstuff.processing.shear import _interp_1d_along, _interp_1d_rowconv

    g = torch.Generator().manual_seed(dim)
    vol = torch.randn(9, 11, 13, generator=g)
    shp = [9, 11, 13]
    shp[dim] = 1
    af = torch.randn(shp, generator=g) * 4.0
    af.view(-1)[:3] = torch.tensor([40.0, -40.0, 10.5])  # rows shifted out of the FOV
    ref = _interp_1d_along(vol, dim, af, mode)
    out = _interp_1d_rowconv(vol, dim, af, mode)
    torch.testing.assert_close(out, ref, rtol=0, atol=2e-5)


@pytest.mark.parametrize(
    "setp",
    [{3: 0.5}, {4: -0.5}, {5: 0.7}, {4: 8.0, 1: 2.0}, {5: -3.0, 2: 1.5, 0: 0.4}],
)
def test_single_axis_rotation_shears_like_its_neighbour(setp):
    """An exact single-axis rotation used to come back None (NaN factorization).

    That is what moco's rotation derivative images request; the gather fallback
    took ~50 s on one CPU thread. The Paeth path must agree with the 4-shear plan
    of a rotation 1e-3 deg away, as closely as two shear plans agree at all.
    """
    from fastfuncstuff.processing.affine import identity_params, params_to_matrix

    g = torch.Generator().manual_seed(0)
    vol = torch.randn(1, 1, 30, 40, 36, generator=g)
    for _ in range(3):
        vol = torch.nn.functional.avg_pool3d(vol, 5, 1, 2)
    vol = vol[0, 0]
    p = identity_params(device=torch.device("cpu"), dtype=torch.float64)
    for k, v in setp.items():
        p[k] = v
    near = p.clone()
    near[3 + (max(k for k in setp if k >= 3) - 2) % 3] += 1e-3
    out = shear_resample(vol, params_to_matrix(p).float(), tuple(vol.shape), "heptic")
    ref = shear_resample(vol, params_to_matrix(near).float(), tuple(vol.shape), "heptic")
    assert out is not None and ref is not None
    inner = (slice(6, -6),) * 3
    err = (out - ref)[inner].pow(2).mean().sqrt() / vol[inner].std()
    assert err < 0.01


@pytest.mark.parametrize(
    "threads,available,enabled,use_rowconv",
    [
        (1, False, True, True),
        (8, False, True, False),
        (8, True, False, False),
        (8, True, True, True),
    ],
)
def test_cpu_shear_dispatch_avoids_threaded_generic_groups(
    monkeypatch, threads, available, enabled, use_rowconv
):
    from fastfuncstuff.processing import shear

    monkeypatch.setattr(torch, "get_num_threads", lambda: threads)
    monkeypatch.setattr(torch.backends.mkldnn, "is_available", lambda: available)
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", enabled)
    expected = shear._interp_1d_rowconv if use_rowconv else shear._interp_1d_along
    assert shear._get_shear_interp(torch.device("cpu")) is expected


@pytest.mark.parametrize("mode", ["heptic", "wsinc5"])
def test_cpu_shear_backends_agree_on_resampled_volume(monkeypatch, mode):
    from fastfuncstuff.processing import shear

    vol = _make_volume(seed=73)
    matrix = _rigid_matrix(0.5, -0.3, 0.2, 1.0, -0.7, 0.4)
    outputs = []
    for use_rowconv in (False, True):
        monkeypatch.setattr(shear, "prefer_cpu_row_convolution", lambda: use_rowconv)
        outputs.append(shear.shear_resample(vol, matrix, vol.shape, mode))
    assert all(out is not None for out in outputs)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=2e-6)


# ---------------------------------------------------------------------------
# Fourier shears. EPI rows are DFT-reconstructed, so a phase ramp shifts them
# exactly; polynomial kernels bias sub-voxel moco toward half-voxel positions.
# ---------------------------------------------------------------------------


def _bandlimited_inplane(nz=10, ny=36, nx=40, seed=0):
    """Random volume whose (y, x) spectrum stops below Nyquist, as an EPI row."""
    g = torch.Generator().manual_seed(seed)
    vol = torch.randn(nz, ny, nx, generator=g, dtype=torch.float64)
    spec = torch.fft.fftn(vol, dim=(1, 2))
    ky = torch.fft.fftfreq(ny, dtype=torch.float64).abs()[:, None]
    kx = torch.fft.fftfreq(nx, dtype=torch.float64).abs()[None, :]
    spec = spec * ((ky < 0.4) & (kx < 0.4))
    vol = torch.fft.ifftn(spec, dim=(1, 2)).real
    return vol + 0.0 * vol.mean()


def _exact_shift(vol, sy, sx):
    """Content moved by (sy, sx) voxels: the exact shift of a periodic band-limited image."""
    ny, nx = vol.shape[1:]
    ky = torch.fft.fftfreq(ny, dtype=torch.float64)[:, None]
    kx = torch.fft.fftfreq(nx, dtype=torch.float64)[None, :]
    ramp = torch.exp(-2j * torch.pi * (ky * sy + kx * sx))
    return torch.fft.ifftn(torch.fft.fftn(vol, dim=(1, 2)) * ramp, dim=(1, 2)).real


def test_fft_shift_rows_is_exact_on_a_trig_polynomial():
    from fastfuncstuff.processing.shear import fft_shift_rows

    n, af = 32, 0.37
    p = torch.arange(n, dtype=torch.float64)
    g = torch.Generator().manual_seed(1)
    coef = torch.randn(15, 2, generator=g, dtype=torch.float64)

    def signal(x):
        k = torch.arange(1, 16, dtype=torch.float64)[:, None]
        w = 2 * torch.pi * k * x[None] / n
        return (coef[:, :1] * torch.cos(w) + coef[:, 1:] * torch.sin(w)).sum(0)

    out = fft_shift_rows(signal(p)[None], 1, torch.tensor([[af]], dtype=torch.float64))
    torch.testing.assert_close(out[0], signal(p - af), rtol=0, atol=1e-10)


def test_fourier_shear_translates_inplane_exactly():
    """In-plane fractional shift through the shear plan equals the exact phase
    ramp; catches sign, axis-mapping and Nyquist mistakes in the Fourier step."""
    vol = _bandlimited_inplane()
    sx, sy = 0.37, -0.23
    # out(p) = source(p + t) moves content by -t.
    mat = torch.eye(4, dtype=torch.float64)
    mat[0, 3], mat[1, 3] = -sx, -sy
    out = shear_resample(vol, mat, tuple(vol.shape), "fourier")
    torch.testing.assert_close(out, _exact_shift(vol, sy, sx), rtol=0, atol=1e-8)
    heptic = shear_resample(vol, mat, tuple(vol.shape), "heptic")
    assert (heptic - _exact_shift(vol, sy, sx)).abs().max() > 100 * (
        out - _exact_shift(vol, sy, sx)
    ).abs().max()


def test_fourier_shear_rotation_tracks_heptic():
    """A small rotation through the Fourier plan agrees with the polynomial plan
    away from the border (they differ only in the 1D kernel)."""
    vol = _make_volume(20, 36, 40, seed=4).double()
    mat = _rigid_matrix(0.3, -0.2, 0.1, 1.5, 0.8, -0.6).double()
    four = shear_resample(vol, mat, tuple(vol.shape), "fourier")
    hept = shear_resample(vol, mat, tuple(vol.shape), "heptic")
    inner = (slice(4, -4), slice(6, -6), slice(6, -6))
    rel = (four - hept)[inner].abs().mean() / vol[inner].abs().mean()
    assert rel < 0.02, rel


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["fourier", "fourier3d"])
@pytest.mark.parametrize("perm", range(6))
def test_shared_fourier_path_matches_general_path(mode, perm):
    """The shared-order fast path (one FFT or Triton pass per step, fused ramp)
    must equal the general per-lane implementation on the same plan, for every
    axis order. Order z,x,y,z once returned uninitialized memory: a Triton step
    after a y-axis inverse FFT wrote into a reshape copy of a strided buffer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/Triton required")
    from fastfuncstuff.processing import shear as S
    from fastfuncstuff.processing.shear_triton import (
        _MODE_ID,
        _fourier_shears,
        _fourier_shears_shared,
        _unpack_ab,
    )

    nt, nz, ny, nx = 6, 20, 36, 40
    base = _make_volume(nz, ny, nx, seed=7)
    g = torch.Generator().manual_seed(3)
    mats = []
    for _ in range(nt):
        r = torch.randn(6, generator=g)
        mats.append(_rigid_matrix(*(r[:3] * 0.6).tolist(), *(r[3:] * 1.0).tolist()))
    M = torch.stack(mats).double()
    # the plan for one forced permutation, as _shear_best builds it
    c = torch.tensor([(nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0], dtype=torch.float64)
    A, t = M[:, :3, :3], M[:, :3, 3]
    q = torch.linalg.inv(A)
    d = -(q @ ((A @ c) + t - c).unsqueeze(-1)).squeeze(-1)
    pi = list(S._PERMS[perm])
    idx = torch.tensor(pi)
    ax, scl, sft, valid = S._shear_xzyx(q[:, idx][:, :, idx], d[:, idx])
    ax, scl, sft = S._permute_back(ax, scl, sft, pi)
    assert bool(valid.all())
    a4, b4 = _unpack_ab(ax, scl)
    fft_axes, poly = S.split_fourier_mode(mode)
    src = base[None].expand(nt, nz, ny, nx).contiguous().cuda()
    a4, b4, sft, ax = a4.float().cuda(), b4.float().cuda(), sft.float().cuda(), ax.cuda()
    order = ax[0].tolist()
    shared = _fourier_shears_shared(src, order, a4, b4, sft, fft_axes, _MODE_ID[poly])
    general = _fourier_shears(src, ax, a4, b4, sft, fft_axes, _MODE_ID[poly])
    torch.testing.assert_close(shared, general, rtol=1e-4, atol=1e-4)


def test_common_plan_rejects_badly_conditioned_lanes():
    """One axis order for the batch is only taken where it is well conditioned:
    near a lane's degenerate axis the factorization returns large cancelling
    shears (0.65 on a real 0.9 deg run), which sent GN 7 mm astray. Every lane
    left valid must have a shear norm within 2x (+0.01) of its own best."""
    from fastfuncstuff.processing import shear as S

    g = torch.Generator().manual_seed(5)
    mats = [
        _rigid_matrix(
            *(torch.randn(3, generator=g) * 0.3).tolist(),
            *(torch.randn(3, generator=g) * 0.6).tolist(),
        )
        for _ in range(12)
    ]
    mats.append(_rigid_matrix(0.1, 0.0, 0.0, 0.9, 1e-4, 1e-4))  # nearly single-axis
    mats.append(_rigid_matrix(0.0, 0.1, 0.0, 1e-4, 0.9, 1e-4))
    M = torch.stack(mats).double()
    ax, scl, sft, valid = rigid_matrix_to_shears(M, (24, 40, 44), common=True)
    # recompute each lane's per-permutation norms the way _shear_best does
    q = torch.linalg.inv(M[:, :3, :3])
    B = len(M)
    idxs = [torch.tensor(p) for p in S._PERMS]
    qq = torch.cat([q[:, i][:, :, i] for i in idxs])
    xx = torch.zeros(6 * B, 3, dtype=torch.float64)
    a6, s6, f6, v6 = S._shear_xzyx(qq, xx)
    norms = torch.stack(
        [
            S._norm_3shear(
                *S._permute_back(
                    a6[k * B : (k + 1) * B],
                    s6[k * B : (k + 1) * B],
                    f6[k * B : (k + 1) * B],
                    list(p),
                )[:2],
                v6[k * B : (k + 1) * B],
            )
            for k, p in enumerate(S._PERMS)
        ],
        dim=1,
    )
    chosen = S._norm_3shear(ax, scl, torch.ones(B, dtype=torch.bool))
    best = norms.min(dim=1).values
    assert bool((chosen[valid] <= 2 * best[valid] + 0.01 + 1e-9).all())


def test_plan_reproduces_mm_rotation_on_anisotropic_voxels():
    """A rotation in mm on 2.1x2.1x2.45 voxels is D^-1 R D in index space: unit
    determinant, not orthogonal. Taking A^T for A^-1 left such plans 4e-3 off in
    the linear part and 0.2 voxel in the shift while passing the old guard."""
    from scipy.spatial.transform import Rotation

    from fastfuncstuff.processing.shear import _reconstruct_pull

    shape = (48, 94, 94)
    vs = torch.tensor([2.1, 2.1, 2.45], dtype=torch.float64)
    R = torch.tensor(Rotation.from_euler("xyz", [0.86, 0.1, 0.05], degrees=True).as_matrix())
    M = torch.eye(4, dtype=torch.float64)
    M[:3, :3] = R * vs / vs[:, None]
    M[:3, 3] = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    ax, scl, sft, valid = rigid_matrix_to_shears(M[None], shape)
    assert bool(valid[0])
    Lt, bt = _reconstruct_pull(ax.long(), scl.double(), sft.double(), shape)
    torch.testing.assert_close(Lt[0], M[:3, :3], rtol=0, atol=1e-9)
    torch.testing.assert_close(bt[0], M[:3, 3], rtol=0, atol=1e-9)
