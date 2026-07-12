"""Tests for ffs_locomoco optical-flow residual motion correction.

Synthetic: a smooth phantom given a KNOWN per-frame global shift along the PE
axis. The tool must (a) recover ~ -shift as the pull displacement, (b) drop the
frame-to-reference error after correction, and (c) emit a 5-D PE-axis warp that
round-trips through the ffs_nwarp per-frame loader.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fastfuncstuff.processing.locomoco import (
    estimate_residual_flow,
    optical_flow_lk_2d,
    resolve_pe_axis,
)


def _phantom(nx=48, ny=48, nz=5):
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    base = np.sin(X / 5.0) * np.cos(Y / 4.0) + 0.5 * np.sin((X + Y) / 3.0) + Z * 0.02
    return (base - base.min() + 1.0).astype(np.float32)


def _shift_along_y(base, shift):
    """Shift a (nx,ny,nz) volume by `shift` voxels along y (axis 1), per z-slice."""
    nx, ny, nz = base.shape
    out = np.zeros_like(base)
    ys, xs = torch.meshgrid(
        torch.arange(nx, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([2 * (xs + shift) / (ny - 1) - 1, 2 * ys / (nx - 1) - 1], -1)[None]
    for z in range(nz):
        sl = torch.from_numpy(base[:, :, z])[None, None]
        out[:, :, z] = F.grid_sample(sl, grid, align_corners=True, padding_mode="border")[0, 0]
    return out


@pytest.fixture
def known_shift_series():
    base = _phantom()
    shifts = np.array([0.0, 0.5, 1.0, 1.5, -1.0, -0.5, 0.8, -1.3, 0.3, 2.0], np.float32)
    T = len(shifts)
    data = np.zeros((*base.shape, T), np.float32)
    for t, sh in enumerate(shifts):
        data[..., t] = _shift_along_y(base, float(sh))
    return data, shifts


def test_optical_flow_recovers_uniform_translation():
    base = _phantom(40, 40, 1)[:, :, 0]
    fixed = torch.from_numpy(base)[None]
    moving = torch.from_numpy(_shift_along_y(base[:, :, None], 1.5)[:, :, 0])[None]
    u, v = optical_flow_lk_2d(fixed, moving, n_levels=3, n_iters=8)
    # moving is `fixed` shifted +1.5 along W(=y); the pull flow to align is ~ -1.5 in u.
    interior = u[0, 5:-5, 5:-5]
    assert abs(float(interior.median()) + 1.5) < 0.15
    assert float(v[0, 5:-5, 5:-5].abs().median()) < 0.15  # no spurious H-flow


def test_recovers_per_frame_pe_shift(known_shift_series):
    data, shifts = known_shift_series
    res = estimate_residual_flow(
        data,
        pe_axis=1,
        slice_axis=2,
        ref_mode="first",
        n_iters=6,
        device=torch.device("cpu"),
        verbose=False,
    )
    pe = res.pe_displacement().numpy()  # (nx,ny,nz,T)
    est = np.median(pe.reshape(-1, pe.shape[-1]), axis=0)
    # Pull displacement recovers -shift (to resample moving back onto the reference).
    assert np.corrcoef(est, -shifts)[0, 1] > 0.99
    assert np.abs(est + shifts).max() < 0.3


@pytest.mark.parametrize("backend", ["phase", "xcorr"])
def test_alt_backends_recover_pe_shift(known_shift_series, backend):
    # The phase-correlation and cross-correlation searchlights must recover the same
    # per-frame PE shift as optical flow, to sub-voxel accuracy.
    data, shifts = known_shift_series
    res = estimate_residual_flow(
        data,
        pe_axis=1,
        slice_axis=2,
        ref_mode="first",
        backend=backend,
        device=torch.device("cpu"),
        verbose=False,
    )
    pe = res.pe_displacement().numpy()
    est = np.median(pe.reshape(-1, pe.shape[-1]), axis=0)
    assert np.corrcoef(est, -shifts)[0, 1] > 0.99
    assert np.abs(est + shifts).max() < 0.3


def test_xcorr_search_matches_translation():
    # Direct check of the cross-correlation searchlight primitive: a +1.2 shift along
    # W(u) must come back as a -1.2 pull, with no spurious H(v) component.
    from fastfuncstuff.processing.locomoco import xcorr_search_flow_2d

    base = _phantom(40, 40, 1)[:, :, 0]
    fixed = torch.from_numpy(base)[None]
    moving = torch.from_numpy(_shift_along_y(base[:, :, None], 1.2)[:, :, 0])[None]
    u, v = xcorr_search_flow_2d(fixed, moving, pe_is_u=True, max_shift=3.0)
    assert abs(float(u[0, 6:-6, 6:-6].median()) + 1.2) < 0.15
    assert float(v.abs().max()) == 0.0  # PE-only: orthogonal component is exactly zero


def _shift_xy(base, sx, sy):
    """Shift a (nx,ny,nz) volume by sx along x(axis0=H) and sy along y(axis1=W)."""
    nx, ny, nz = base.shape
    out = np.zeros_like(base)
    hh, ww = torch.meshgrid(
        torch.arange(nx, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        indexing="ij",
    )
    gx = 2 * (ww + sy) / (ny - 1) - 1
    gy = 2 * (hh + sx) / (nx - 1) - 1
    grid = torch.stack([gx, gy], -1)[None]
    for z in range(nz):
        sl = torch.from_numpy(base[:, :, z])[None, None]
        out[:, :, z] = F.grid_sample(sl, grid, align_corners=True, padding_mode="border")[0, 0]
    return out


@pytest.mark.parametrize("backend", ["flow", "phase", "xcorr"])
def test_dual_pe_recovers_both_axes(backend):
    # Two in-plane PE axes (x and y), slice along z. Each frame is shifted by a known
    # amount on BOTH axes; dual mode must recover both pull components (-shift each).
    base = _phantom(44, 44, 4)
    sx = np.array([0.0, 0.8, -0.6, 1.1, -0.9, 0.5], np.float32)  # along x (v / axis 0)
    sy = np.array([0.0, -0.5, 1.0, -0.8, 0.6, -1.1], np.float32)  # along y (u / axis 1)
    data = np.stack([_shift_xy(base, float(a), float(b)) for a, b in zip(sx, sy, strict=True)], -1)
    res = estimate_residual_flow(
        data,
        pe_axis=0,
        slice_axis=2,
        backend=backend,
        dual=True,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert res.dual
    comps = dict(res.warp_components())  # {axis: (nx,ny,nz,T)}
    assert set(comps) == {0, 1}  # both in-plane axes carried
    core = (slice(8, -8), slice(8, -8), slice(None))
    est_u = np.median(comps[1][core].reshape(-1, len(sy)), axis=0)  # y = W = u
    est_v = np.median(comps[0][core].reshape(-1, len(sx)), axis=0)  # x = H = v
    assert np.corrcoef(est_u, -sy)[0, 1] > 0.95
    assert np.corrcoef(est_v, -sx)[0, 1] > 0.95
    assert np.abs(est_u + sy).max() < 0.4 and np.abs(est_v + sx).max() < 0.4


def test_jacobian_det_linear_ramp():
    from fastfuncstuff.processing.locomoco import _jacobian_det

    # u = a*x along W → J = 1 + a everywhere (interior); v=0.
    nx = ny = 24
    xs = torch.arange(ny, dtype=torch.float32)
    for a in (0.2, -0.3):
        u = (a * xs)[None, None, None, :].expand(1, 1, nx, ny).contiguous()
        j = _jacobian_det(u, torch.zeros_like(u))
        assert abs(float(j[..., 2:-2].median()) - (1.0 + a)) < 1e-3


def test_local_field_recovers_spatial_variation():
    # A SMOOTH spatially-varying PE field (stretch where df/dy>0, squish where <0),
    # not a global shift: the estimate must track the field's spatial variation.
    rng = np.random.default_rng(0)
    nx, ny, nz = 40, 60, 3
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    base = (np.sin(xx / 4.0) * np.cos(yy / 5.0) + 0.6 * rng.standard_normal((nx, ny))).astype(
        np.float32
    )
    base = np.repeat(base[:, :, None], nz, axis=2)
    yline = np.arange(ny, dtype=np.float32)

    def distort(vol, f):  # moving(y) = base(y + f(y)), f varies along y (PE=axis1)
        out = np.empty_like(vol)
        hh, ww = torch.meshgrid(
            torch.arange(nx, dtype=torch.float32),
            torch.arange(ny, dtype=torch.float32),
            indexing="ij",
        )
        gx = 2 * (ww + torch.from_numpy(f)[None, :]) / (ny - 1) - 1
        gy = 2 * hh / (nx - 1) - 1
        grid = torch.stack([gx, gy], -1)[None]
        for z in range(nz):
            sl = torch.from_numpy(vol[:, :, z])[None, None]
            out[:, :, z] = F.grid_sample(sl, grid, align_corners=True, padding_mode="border")[0, 0]
        return out

    amps = [0.0, 1.0, -0.8, 1.3]
    fields = [(a * np.sin(2 * np.pi * 1.5 * yline / ny)).astype(np.float32) for a in amps]
    data = np.stack([distort(base, f) for f in fields], -1)
    res = estimate_residual_flow(
        data,
        pe_axis=1,
        slice_axis=2,
        ref_mode="first",
        n_iters=6,
        device=torch.device("cpu"),
        verbose=False,
    )
    est = res.pe_displacement().numpy()  # ~ -f(y)
    interior = (slice(8, -8), slice(8, -8), slice(None))
    for t in range(1, len(amps)):
        true = np.broadcast_to(-fields[t][None, :, None], (nx, ny, nz))
        # The recovered field tracks the true spatial variation (correlated, not flat).
        assert np.corrcoef(est[..., t][interior].ravel(), true[interior].ravel())[0, 1] > 0.9


def test_unknown_backend_raises(known_shift_series):
    data, _ = known_shift_series
    with pytest.raises(ValueError, match="Unknown backend"):
        estimate_residual_flow(
            data, pe_axis=1, slice_axis=2, backend="bogus", device=torch.device("cpu")
        )


def test_correction_reduces_frame_to_ref_error(known_shift_series):
    data, _ = known_shift_series
    res = estimate_residual_flow(
        data,
        pe_axis=1,
        slice_axis=2,
        ref_mode="first",
        n_iters=6,
        device=torch.device("cpu"),
        verbose=False,
    )
    corr = res.corrected_series().numpy()
    ref = data[..., 0]
    T = data.shape[-1]
    before = np.mean([np.abs(data[..., t] - ref).mean() for t in range(T)])
    after = np.mean([np.abs(corr[..., t] - ref).mean() for t in range(T)])
    assert after < 0.3 * before  # most residual motion removed


def test_warp_is_5d_pe_axis_and_roundtrips(tmp_path, known_shift_series):
    from fastfuncstuff.processing.medic import save_medic_warp
    from fastfuncstuff.processing.nwarpforge import (
        _is_time_varying_warp,
        load_time_varying_warp,
    )

    data, _ = known_shift_series
    res = estimate_residual_flow(
        data,
        pe_axis=1,
        slice_axis=2,
        ref_mode="first",
        n_iters=6,
        device=torch.device("cpu"),
        verbose=False,
    )
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    stem = str(tmp_path / "sub")
    path = save_medic_warp(res.pe_displacement(), 1, affine, stem, as_5d=True)

    import nibabel as nib

    warp = np.asarray(nib.load(path).get_fdata(dtype=np.float32))
    assert warp.ndim == 5 and warp.shape[-1] == 3
    assert warp.shape[3] == data.shape[-1]
    # Only the PE (y) component is populated; x and z stay zero.
    assert np.abs(warp[..., 0]).max() < 1e-6
    assert np.abs(warp[..., 2]).max() < 1e-6
    assert np.abs(warp[..., 1]).max() > 0.1
    assert _is_time_varying_warp(path)
    tv = load_time_varying_warp(path, device=torch.device("cpu"))
    assert tv.n_time == data.shape[-1]


def test_pe_only_mode_runs_and_matches(known_shift_series):
    data, shifts = known_shift_series
    res = estimate_residual_flow(
        data,
        pe_axis=1,
        slice_axis=2,
        ref_mode="first",
        n_iters=6,
        pe_only=True,
        device=torch.device("cpu"),
        verbose=False,
    )
    pe = res.pe_displacement().numpy()
    est = np.median(pe.reshape(-1, pe.shape[-1]), axis=0)
    assert np.corrcoef(est, -shifts)[0, 1] > 0.99


def test_pe_axis_equals_slice_axis_raises():
    data = np.zeros((8, 8, 8, 3), np.float32)
    with pytest.raises(ValueError, match="lie inside the slice plane"):
        estimate_residual_flow(data, pe_axis=2, slice_axis=2, device=torch.device("cpu"))


def test_resolve_pe_axis():
    assert resolve_pe_axis("AP") == 1
    assert resolve_pe_axis("PA") == 1
    assert resolve_pe_axis("LR") == 0
    assert resolve_pe_axis("z") == 2
    with pytest.raises(ValueError, match="Unknown -pe_dir"):
        resolve_pe_axis("QQ")


def test_split_prefix_strips_extensions():
    from fastfuncstuff.cli.locomoco import _split_prefix

    assert _split_prefix("sub") == ("sub", ".nii.gz")
    assert _split_prefix("sub.nii.gz") == ("sub", ".nii.gz")
    assert _split_prefix("sub.nii.zst") == ("sub", ".nii.zst")
    assert _split_prefix("sub.nii") == ("sub", ".nii")
    # Periods in the stem are preserved; only the imaging extension is stripped.
    assert _split_prefix("a/b.blur.2mm") == ("a/b.blur.2mm", ".nii.gz")


def test_strip_imaging_extension_handles_zst():
    from fastfuncstuff.glm.outputs import _strip_imaging_extension

    assert _strip_imaging_extension("errts.sub-01.nii.zst") == "errts.sub-01"
    assert _strip_imaging_extension("errts.sub-01.nii.gz") == "errts.sub-01"


def test_signed_flow_map(known_shift_series):
    data, shifts = known_shift_series
    res = estimate_residual_flow(
        data,
        pe_axis=1,
        slice_axis=2,
        ref_mode="first",
        n_iters=6,
        device=torch.device("cpu"),
        verbose=False,
    )
    flow = res.pe_displacement().numpy()  # signed, (nx,ny,nz,T)
    assert flow.shape == data.shape
    est = np.median(flow.reshape(-1, flow.shape[-1]), axis=0)
    # Signed: recovers -shift (pull), so sign flips with motion direction.
    assert np.corrcoef(est, -shifts)[0, 1] > 0.99
    assert (est < 0).any() and (est > 0).any()


@pytest.mark.parametrize("ref_mode", ["first_mean", "first_median"])
def test_progressive_reference_recovers_shift(known_shift_series, ref_mode):
    data, shifts = known_shift_series
    res = estimate_residual_flow(
        data,
        pe_axis=1,
        slice_axis=2,
        ref_mode=ref_mode,
        n_iters=6,
        device=torch.device("cpu"),
        verbose=False,
    )
    flow = res.pe_displacement().numpy()
    est = np.median(flow.reshape(-1, flow.shape[-1]), axis=0)
    # Frame 0 is the seed (zero warp); the rest register to the running template,
    # which stays ~frame 0 (all corrected frames are aligned back to it), so the
    # recovered pull flow tracks -shift like the static "first" reference.
    assert abs(est[0]) < 0.1
    assert np.corrcoef(est[1:], -shifts[1:])[0, 1] > 0.95


def test_automask_gates_flow_outside_brain():
    # A compact bright blob (the "brain") in a field of pure noise. Optical flow
    # invents large displacements in the noise; the automask must feather those to
    # ~0 outside the blob while preserving the recovered shift inside it.
    rng = np.random.default_rng(0)
    nx, ny, nz, T = 40, 40, 4, 6
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    blob = np.exp(-(((xx - 20) / 6.0) ** 2 + ((yy - 20) / 6.0) ** 2)).astype(np.float32)
    brain = (blob[:, :, None] * (5.0 + np.sin(xx / 3.0)[:, :, None])).astype(np.float32)
    shifts = np.array([0.0, 1.0, -1.0, 1.5, -0.8, 0.6], np.float32)
    data = np.zeros((nx, ny, nz, T), np.float32)
    for t, sh in enumerate(shifts):
        shifted = _shift_along_y(np.repeat(brain, nz, axis=2), float(sh))
        noise = 0.3 * rng.standard_normal((nx, ny, nz)).astype(np.float32)
        data[..., t] = shifted + noise

    common = dict(pe_axis=1, slice_axis=2, ref_mode="first", n_iters=6, verbose=False)
    res = estimate_residual_flow(
        data, automask=True, automask_dilate=3, automask_sigma=2.0, **common
    )
    flow = res.pe_displacement().numpy()  # (nx,ny,nz,T)
    inside = flow[17:23, 17:23]  # within the blob
    outside = flow[:5, :5]  # far corner, pure noise
    # The mask crushes the noisy corner flow but leaves the in-brain shift intact.
    assert np.abs(outside).max() < 0.1
    assert np.abs(inside).max() > 0.3


def test_spinner_silent_when_not_a_tty():
    import io

    from fastfuncstuff.cli_utils import spinner

    buf = io.StringIO()  # StringIO.isatty() is False -> no animation, no output
    ran = []
    with spinner("loading", stream=buf, interval=0.001):
        ran.append(True)
    assert ran == [True]
    assert buf.getvalue() == ""


def test_flow_movie_shape(known_shift_series):
    data, _ = known_shift_series
    res = estimate_residual_flow(
        data,
        pe_axis=1,
        slice_axis=2,
        ref_mode="first",
        n_iters=3,
        device=torch.device("cpu"),
        verbose=False,
    )
    movie = res.flow_movie()
    assert movie.ndim == 4 and movie.shape[0] == data.shape[-1] and movie.shape[-1] == 3
    assert movie.dtype == np.uint8


# ── rotation-aware (idea 2) ───────────────────────────────────────────────────
from fastfuncstuff.processing.locomoco import (  # noqa: E402
    compute_reproject_weights,
    estimate_residual_flow_rotaware,
    pe_tilt_degrees,
    _fuse_tridiag,
)


def _rotz(theta_deg):
    c, s = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return M


def test_reproject_weights_identity_reduces_to_pe():
    """Identity motion → all energy on the PE axis, zero leak, zero tilt."""
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    M = torch.eye(4).repeat(4, 1, 1)  # (T,4,4) identity
    for pe in (0, 1, 2):
        w = compute_reproject_weights(M, aff, pe_axis=pe)
        assert w.shape == (4, 3)
        assert abs(float(w[0, pe]) - 1.0) < 1e-5
        for a in (0, 1, 2):
            if a != pe:
                assert abs(float(w[0, a])) < 1e-5
        assert float(pe_tilt_degrees(M, aff, pe).max()) < 1e-3


def test_reproject_weights_known_rotation():
    """10° roll about z, isotropic voxels, PE=AP(y): PE→cosθ, leak onto x→sinθ, z→0."""
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    theta = 10.0
    M = torch.from_numpy(_rotz(theta)).repeat(3, 1, 1)
    w = compute_reproject_weights(M, aff, pe_axis=1)
    assert abs(float(w[0, 1]) - np.cos(np.radians(theta))) < 1e-4  # PE component
    assert abs(abs(float(w[0, 0])) - np.sin(np.radians(theta))) < 1e-4  # LR leak
    assert abs(float(w[0, 2])) < 1e-5  # no IS leak (roll is about z)
    assert abs(float(pe_tilt_degrees(M, aff, 1)[0]) - theta) < 1e-3


def test_reproject_leak_axis_depends_on_pe_dir():
    """A rotation about z leaks off PE differently for PE=x vs PE=y; PE=z is inert."""
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    M = torch.from_numpy(_rotz(12.0)).repeat(2, 1, 1)
    # PE=z is the rotation axis → no tilt, no leak.
    assert float(pe_tilt_degrees(M, aff, 2).max()) < 1e-3
    # PE=x and PE=y both tilt by 12°, leaking onto the other in-plane axis.
    assert abs(float(pe_tilt_degrees(M, aff, 0)[0]) - 12.0) < 1e-3
    assert abs(float(pe_tilt_degrees(M, aff, 1)[0]) - 12.0) < 1e-3


def test_fuse_tridiag_recovers_known_sequence():
    """Anchor + exact differentials of a known p(t) recover p(t) up to fit weights."""
    T = 20
    true_p = np.cumsum(np.random.RandomState(0).randn(T)) * 0.1
    fd = np.zeros((T, 1))
    fd[1:, 0] = np.diff(true_p)
    anchor = true_p[:, None].astype(np.float32)
    out = _fuse_tridiag(
        torch.from_numpy(fd).float(), torch.from_numpy(anchor).float(), w_anchor=1.0
    )
    assert np.corrcoef(out.numpy().ravel(), true_p)[0, 1] > 0.999


def test_rotaware_reduces_to_plain_at_zero_motion(known_shift_series):
    """With identity moco (raw==moco, no rotation) the rotation-aware warp must put all
    energy on the PE axis (leak ≈ 0) and its PE component must track the plain path."""
    data, _ = known_shift_series  # (nx,ny,nz,T), known PE shifts along axis 1
    T = data.shape[-1]
    M = torch.eye(4).repeat(T, 1, 1)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    res = estimate_residual_flow_rotaware(
        data,
        data,
        M,
        M,
        aff,
        pe_axis=1,
        slice_axis=2,
        ref_mode="mean",
        fuse="off",
        automask=False,
        device=torch.device("cpu"),
        verbose=False,
    )
    comps = dict((a, d) for a, d in res.warp_components())
    pe = comps[1].abs()
    leak = comps[0].abs().max() + comps[2].abs().max()
    assert float(pe.max()) > float(leak) * 5  # PE axis dominates
    assert float(leak) < 0.05  # near-zero off-axis at zero rotation
    # PE component agrees in sign/scale with the plain estimator.
    plain = estimate_residual_flow(
        data,
        pe_axis=1,
        slice_axis=2,
        ref_mode="mean",
        device=torch.device("cpu"),
        verbose=False,
    )
    corr = np.corrcoef(comps[1].numpy().ravel(), plain.pe_displacement().numpy().ravel())[0, 1]
    assert corr > 0.9


def test_max_reference_accepted_by_both_paths(known_shift_series):
    """'max' is a valid reference for the plain path and the rotation-aware default."""
    data, _ = known_shift_series
    T = data.shape[-1]
    plain = estimate_residual_flow(
        data,
        pe_axis=1,
        slice_axis=2,
        ref_mode="max",
        device=torch.device("cpu"),
        verbose=False,
    )
    assert np.isfinite(plain.pe_displacement().numpy()).all()
    M = torch.eye(4).repeat(T, 1, 1)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    import inspect

    # Rotation-aware default reference is temporal max.
    assert (
        inspect.signature(estimate_residual_flow_rotaware).parameters["ref_mode"].default == "max"
    )
    res = estimate_residual_flow_rotaware(
        data,
        data,
        M,
        M,
        aff,
        pe_axis=1,
        slice_axis=2,  # ref_mode defaults to max
        fuse="off",
        automask=False,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert np.isfinite(res.corrected_series().numpy()).all()


# ── 3D-acquired EPI (idea 3: -is_3dacq) ───────────────────────────────────────
from fastfuncstuff.processing.locomoco import (  # noqa: E402
    optical_flow_lk_3d,
    xcorr_search_flow_3d,
    _build_flow3d_fn,
    _shift3d_axis,
)


def _phantom3d(nx=40, ny=40, nz=16):
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    # Strong in-plane (x,y) structure, weak through-z structure — so the sagittal cut
    # (which pools over z) is worse-conditioned than the axial cut (pools over x).
    base = np.sin(X / 4.0) * np.cos(Y / 5.0) + 0.4 * np.sin((X + Y) / 3.0) + 0.03 * Z
    return (base - base.min() + 1.0).astype(np.float32)


def _yshift3d(vol, s):
    """Shift a (nx,ny,nz) volume by s voxels along y (axis 1) via the module warp."""
    return _shift3d_axis(torch.from_numpy(vol)[None], float(s), 1)[0].numpy()


def test_optical_flow_lk_3d_recovers_uniform_shift():
    vol = _phantom3d()
    moved = _yshift3d(vol, 1.3)  # content shifted +1.3 along y
    fx = torch.from_numpy(vol)[None]
    mv = torch.from_numpy(moved)[None]
    disp = optical_flow_lk_3d(fx, mv, pe_axis=1, n_iters=6)[0]  # pull: moving(x+disp)=fixed
    core = disp[6:-6, 6:-6, 3:-3]
    assert abs(float(core.mean()) - (-1.3)) < 0.15


def test_xcorr_3d_recovers_uniform_shift():
    vol = _phantom3d()
    moved = _yshift3d(vol, -0.8)
    disp = xcorr_search_flow_3d(
        torch.from_numpy(vol)[None], torch.from_numpy(moved)[None], pe_axis=1, max_shift=3
    )[0]
    core = disp[6:-6, 6:-6, 3:-3]
    assert abs(float(core.mean()) - 0.8) < 0.15


def test_3d_solve_beats_averaging_the_two_cuts():
    """The core -is_3dacq claim: one 3-D solve recovers the field better than either
    valid perpendicular 2-D cut OR their flat average, on a noisy anisotropic volume."""
    rng = np.random.RandomState(0)
    vol = _phantom3d()
    shifts = np.array([0.0, 0.6, 1.1, -0.5, -1.0, 0.8, 0.3, -0.7], dtype=np.float32)
    series = np.stack([_yshift3d(vol, s) for s in shifts], -1)
    series = series + rng.randn(*series.shape).astype(np.float32) * 0.10  # noise
    truth = -shifts  # uniform pull displacement per frame

    def err(res):
        d = res.pe_displacement().numpy()  # (nx,ny,nz,T)
        core = d[6:-6, 6:-6, 3:-3, :]
        return float(np.abs(core - truth[None, None, None, :]).mean())

    common = dict(
        pe_axis=1, backend="flow", ref_mode="mean", device=torch.device("cpu"), verbose=False
    )
    res3d = estimate_residual_flow(series, slice_axis=2, is_3dacq=True, **common)
    axial = estimate_residual_flow(series, slice_axis=2, **common)  # 2-D cut, pools x
    sagittal = estimate_residual_flow(series, slice_axis=0, **common)  # 2-D cut, pools z
    e3d, ea, es = err(res3d), err(axial), err(sagittal)
    davg = 0.5 * (axial.pe_displacement().numpy() + sagittal.pe_displacement().numpy())
    core = davg[6:-6, 6:-6, 3:-3, :]
    e_avg = float(np.abs(core - truth[None, None, None, :]).mean())
    # 3-D is no worse than the better cut and strictly better than the flat average.
    assert e3d <= min(ea, es) * 1.15
    assert e3d < e_avg


def test_phase_backend_has_no_3d_path():
    with pytest.raises(ValueError, match="phase backend has no 3-D"):
        _build_flow3d_fn(
            "phase", 1, n_levels=3, n_iters=4, window_sigma=2.0, max_shift=3, trial_step=0.5
        )


def test_3d_rotaware_runs_and_reduces(known_shift_series):
    """Rotation-aware + 3D acquisition: identity motion → PE-axis-dominant, finite."""
    data, _ = known_shift_series
    T = data.shape[-1]
    M = torch.eye(4).repeat(T, 1, 1)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    res = estimate_residual_flow_rotaware(
        data,
        data,
        M,
        M,
        aff,
        pe_axis=1,
        slice_axis=2,
        is_3dacq=True,
        fuse="off",
        automask=False,
        device=torch.device("cpu"),
        verbose=False,
    )
    comps = dict((a, d) for a, d in res.warp_components())
    assert np.isfinite(comps[1].numpy()).all()
    assert (
        float(comps[1].abs().max())
        > (float(comps[0].abs().max()) + float(comps[2].abs().max())) * 5
    )


def test_warp_time_pcs(known_shift_series):
    """Warp temporal PCs: (T, k) unit-variance regressors; empty warp → None."""
    from fastfuncstuff.processing.locomoco import warp_time_pcs

    data, _ = known_shift_series
    T = data.shape[-1]
    res = estimate_residual_flow(
        data, pe_axis=1, slice_axis=2, ref_mode="mean", device=torch.device("cpu"), verbose=False
    )
    scores, var = warp_time_pcs(res.warp_components(), n_pcs=3)
    assert scores.shape == (T, 3)
    assert abs(float(scores.std(dim=0).mean()) - 1.0) < 0.1  # normalised to unit variance
    assert var.shape == (3,) and 0.0 <= float(var.sum()) <= 1.0001
    empty, _ = warp_time_pcs([(1, torch.zeros(4, 4, 2, T))], n_pcs=3)
    assert empty is None
