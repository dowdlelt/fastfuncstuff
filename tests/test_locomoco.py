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


def test_flow_direction_and_magnitude_maps(known_shift_series):
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
    ang = res.flow_direction_deg().numpy()
    mag = res.flow_magnitude().numpy()
    assert ang.shape == data.shape and mag.shape == data.shape
    assert ang.min() >= 0.0 and ang.max() < 360.0
    assert mag.min() >= 0.0
    # For slice_axis=z the PE (y) axis maps to the u/W (column) flow component, so
    # a pure PE shift produces a HORIZONTAL flow direction (~0° or ~180°), never
    # vertical. Check the frame with the biggest shift where there is signal.
    t = int(np.argmax(np.abs(shifts)))
    strong = mag[..., t] > np.percentile(mag[..., t], 90)
    med_dir = float(np.median(ang[..., t][strong]))
    assert min(abs(med_dir - h) for h in (0.0, 180.0, 360.0)) < 30.0


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
