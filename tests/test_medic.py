"""Unit tests for the MEDIC warping/apply side we own (processing/medic.py).

Estimation is warpkit's (not exercised here). These cover our GPU tail:
field<->displacement conversion, 1-D inversion, undistortion apply, the
ffs_qwarp-compatible mm warp + its ffs_nwarp composition, and the
warpkit-fieldmap -> ffs_nwarp warp conversion tool.
"""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.processing.medic import (
    displacement_pe_to_field,
    field_to_displacement_pe,
    invert_displacement_pe,
    undistort_series,
)


def test_field_displacement_roundtrip():
    """field -> PE displacement (voxels) -> field is the identity."""
    field = torch.randn(8, 8, 8, 5) * 30.0
    trt = 0.032
    for pe in ("j", "j-", "i", "k-"):
        disp = field_to_displacement_pe(field, trt, pe)
        back = displacement_pe_to_field(disp, trt, pe)
        assert torch.allclose(back, field, atol=1e-4)
        # negative polarity must flip the displacement sign
        if pe.endswith("-"):
            assert torch.allclose(disp, -field * trt, atol=1e-4)
        else:
            assert torch.allclose(disp, field * trt, atol=1e-4)


def test_invert_constant_displacement():
    """Inverting a constant displacement gives its negation."""
    n = 32
    g = torch.full((1, 1, n), 2.5)
    h = invert_displacement_pe(g, pe_tensor_axis=2)
    # interior voxels (away from the clamped edges) should be exactly -2.5
    assert torch.allclose(h[0, 0, 4:-4], torch.full((n - 8,), -2.5), atol=1e-4)


def test_invert_linear_ramp():
    """Fixed-point inverse of an affine 1-D warp matches the closed form."""
    n = 64
    a = 0.2  # |a| < 1 keeps the map invertible
    x = torch.arange(n, dtype=torch.float32)
    g = a * x  # forward displacement g(x) = a*x
    h = invert_displacement_pe(g.view(1, 1, n), pe_tensor_axis=2).view(n)
    # T(x) = (1+a)x  =>  T^-1(y) = y/(1+a)  =>  h(y) = -a/(1+a) * y
    expected = -a / (1 + a) * x
    assert torch.allclose(h[8:-8], expected[8:-8], atol=0.05)


def test_invert_compose_identity():
    """Composing forward then inverse displacement returns ~identity."""
    n = 48
    x = torch.arange(n, dtype=torch.float32)
    g = 3.0 * torch.sin(2 * np.pi * x / n)  # smooth, |g'| < 1
    h = invert_displacement_pe(g.view(1, 1, n), pe_tensor_axis=2).view(n)

    # sample g at (x + h): should cancel h  ->  x + h + g(x+h) ~ x
    from fastfuncstuff.processing.medic import _interp_along_last_axis

    coord = (x + h).view(1, 1, n)
    g_at = _interp_along_last_axis(g.view(1, 1, n), coord).view(n)
    net = h + g_at
    assert net[8:-8].abs().max() < 0.1


def test_undistort_series_applies_pull_warp():
    """undistort_series shifts each frame by its per-frame PE displacement."""
    nx = ny = nz = 12
    nt = 3
    shifts = [0.0, 1.0, 2.0]
    # series[x, y, z, t] = y  (ramp along PE = NIfTI axis 1 = j)
    yy = torch.arange(ny, dtype=torch.float32).view(1, ny, 1).expand(nx, ny, nz)
    series = yy.unsqueeze(-1).repeat(1, 1, 1, nt).contiguous()
    disp = torch.zeros(nx, ny, nz, nt)
    for t in range(nt):
        disp[..., t] = shifts[t]

    out = undistort_series(series, disp, pe_nifti_axis=1, interp="linear", verbose=False)
    for t in range(nt):
        interior = out[:, 1:-2, :, t]
        expected = yy[:, 1:-2, :] + shifts[t]
        assert torch.allclose(interior, expected, atol=1e-3)

    # Streaming refactor: routing frames through an explicit compute device must not
    # change the result, and the output stays on the series' (host) device.
    streamed = undistort_series(
        series, disp, pe_nifti_axis=1, interp="linear", verbose=False, device=series.device
    )
    assert streamed.device == series.device
    assert torch.equal(streamed, out)


def test_time_chunk_frames_respects_budget():
    """_time_chunk_frames divides the memory budget into whole frames (>= 1)."""
    import fastfuncstuff.memory as mem
    from fastfuncstuff.processing import medic

    n_voxels = 1000
    n_fields = 6
    # Force a known byte budget: 10 frames' worth of 6 float32 fields.
    budget = 10 * n_fields * n_voxels * 4
    orig = mem.get_available_memory
    mem.get_available_memory = lambda device, *a, **k: budget
    try:
        chunk = medic._time_chunk_frames(n_voxels, torch.device("cpu"), n_fields=n_fields)
    finally:
        mem.get_available_memory = orig
    assert chunk == 10

    # A budget too small for even one field still yields a usable single frame.
    mem.get_available_memory = lambda device, *a, **k: 1
    try:
        assert medic._time_chunk_frames(n_voxels, torch.device("cpu")) == 1
    finally:
        mem.get_available_memory = orig


def test_field_to_pull_warp_matches_manual_inversion():
    """field_to_pull_warp reproduces the per-frame field->disp->invert tail (host)."""
    from fastfuncstuff.processing.medic import field_to_pull_warp

    torch.manual_seed(0)
    nx = ny = nz = 8
    nt = 5
    trt, pe_dir, pe_axis = 0.03, "j-", 1
    field = torch.randn(nx, ny, nz, nt) * 20.0  # Hz

    # Reference: the exact inline math the helper replaced.
    disp_native = field_to_displacement_pe(field, trt, pe_dir)
    ref_pull = torch.empty_like(disp_native)
    for t in range(nt):
        ref_pull[..., t] = invert_displacement_pe(disp_native[..., t], pe_axis)
    ref_undist = displacement_pe_to_field(ref_pull, trt, pe_dir)
    a, b = ref_undist[..., 0].reshape(-1), field[..., 0].reshape(-1)
    if torch.dot(a - a.mean(), b - b.mean()) < 0:
        ref_undist = -ref_undist

    disp_pull, field_undist = field_to_pull_warp(
        field, trt, pe_dir, pe_axis, torch.device("cpu"), verbose=False
    )
    assert disp_pull.device.type == "cpu" and field_undist.device.type == "cpu"
    assert torch.allclose(disp_pull, ref_pull, atol=1e-5)
    assert torch.allclose(field_undist, ref_undist, atol=1e-5)
    # Sign check guarantees a non-negative frame-0 correlation with the native field.
    a, b = field_undist[..., 0].reshape(-1), field[..., 0].reshape(-1)
    assert torch.dot(a - a.mean(), b - b.mean()) >= 0


def test_detrend_time_removes_mean_and_linear():
    """detrend_time annihilates per-voxel const+linear and is orthogonal to them."""
    from fastfuncstuff.processing.medic import detrend_time

    nx = ny = nz = 4
    nt = 40
    t = torch.arange(nt, dtype=torch.float32)
    osc = torch.sin(2 * torch.pi * t / 10.0)
    field = torch.zeros(nx, ny, nz, nt)
    for ix in range(nx):
        field[ix] = (5.0 + ix) + 0.3 * ix * t + osc  # per-voxel offset + slope + shared osc

    out = detrend_time(field, order=1)
    # Result is orthogonal to the removed basis: zero mean and zero linear trend.
    assert out.mean(dim=-1).abs().max() < 1e-3
    tc = t - t.mean()
    assert (out * tc).sum(dim=-1).abs().max() < 1e-2

    # Detrend is a linear projection, so the exactly-in-span const+linear vanish and
    # only the (shared) oscillation's residual remains — independent of voxel offset/slope.
    osc_field = osc.view(1, 1, 1, nt).expand(nx, ny, nz, nt)
    assert torch.allclose(out, detrend_time(osc_field, order=1), atol=1e-4)

    # A pure const+linear field detrends to ~0.
    poly = torch.zeros(nx, ny, nz, nt)
    for ix in range(nx):
        poly[ix] = (2.0 + ix) - 0.5 * ix * t
    assert detrend_time(poly, order=1).abs().max() < 1e-2

    # order < 0 is a no-op: the raw field map (mean + trend) is returned unchanged.
    assert torch.equal(detrend_time(field, order=-1), field)


def test_undistort_series_extra_axis_shifts_slice():
    """extra_disp on the k axis shifts along z, jointly with the PE-axis warp."""
    nx = ny = nz = 12
    nt = 2
    # ramp along k (z): series[x,y,z,t] = z; and along j for the PE warp: + y
    zz = torch.arange(nz, dtype=torch.float32).view(1, 1, nz).expand(nx, ny, nz)
    yy = torch.arange(ny, dtype=torch.float32).view(1, ny, 1).expand(nx, ny, nz)
    series = (yy + zz).unsqueeze(-1).repeat(1, 1, 1, nt).contiguous()
    pe_disp = torch.full((nx, ny, nz, nt), 1.0)  # +1 along j
    z_disp = torch.full((nx, ny, nz, nt), 2.0)  # +2 along k
    out = undistort_series(
        series,
        pe_disp,
        pe_nifti_axis=1,
        interp="linear",
        verbose=False,
        extra_disp=z_disp,
        extra_nifti_axis=2,
    )
    interior = out[:, 1:-2, 1:-2, 0]
    expected = (yy + zz)[:, 1:-2, 1:-2] + 1.0 + 2.0
    assert torch.allclose(interior, expected, atol=1e-3)


def test_undistort_series_extra_axis_requires_pair():
    series = torch.zeros(4, 4, 4, 1)
    disp = torch.zeros(4, 4, 4, 1)
    try:
        undistort_series(series, disp, 1, extra_disp=disp, verbose=False)
        raise AssertionError("expected ValueError for missing extra_nifti_axis")
    except ValueError:
        pass


def test_undistort_series_circular_phase():
    """Circular mode undistorts wrapped phase without smearing across wraps."""
    nx = ny = nz = 8
    # constant phase per frame near +pi; a small shift must not corrupt it
    series = torch.full((nx, ny, nz, 1), 3.0)
    disp = torch.full((nx, ny, nz, 1), 1.0)
    out = undistort_series(
        series, disp, pe_nifti_axis=1, interp="linear", circular=True, verbose=False
    )
    # constant field -> output stays the constant phase (interior)
    assert torch.allclose(out[:, 2:-2, :, 0], torch.full((nx, ny - 4, nz), 3.0), atol=1e-4)


def _ramp_phantom(nx, ny, nz, nt, voxel=2.0):
    """A 4D ramp along the j (PE) axis on a `voxel`-mm isotropic grid."""
    yy = torch.arange(ny, dtype=torch.float32).view(1, ny, 1).expand(nx, ny, nz)
    series = yy.unsqueeze(-1).repeat(1, 1, 1, nt).contiguous()
    affine = np.diag([voxel, voxel, voxel, 1.0]).astype(np.float64)
    return series, yy, affine


def test_medic_mm_warp_wildcard_matches_undistort(tmp_path):
    """ffs_medic mm warps (per-frame wildcard) applied by ffs_nwarp == undistort_series.

    This pins the qwarp mm convention end-to-end on a non-unit voxel grid.
    """
    import nibabel as nib

    from fastfuncstuff.processing.medic import save_medic_warp
    from fastfuncstuff.processing.nwarpforge import nwarpforge

    nx = ny = nz = 12
    nt = 3
    series, _yy, affine = _ramp_phantom(nx, ny, nz, nt, voxel=2.0)
    disp = torch.zeros(nx, ny, nz, nt)
    for t, s in enumerate([0.0, 1.5, -1.0]):  # per-frame, incl. negative
        disp[..., t] = s

    ref = undistort_series(series, disp, pe_nifti_axis=1, interp="linear", verbose=False)

    src_path = tmp_path / "src.nii"
    nib.Nifti1Image(series.numpy(), affine).to_filename(str(src_path))
    spec = save_medic_warp(disp, 1, affine, str(tmp_path / "dc"), ".nii", as_5d=False)

    out_path = tmp_path / "out.nii"
    nwarpforge(
        source_path=str(src_path),
        nwarp_specs=[spec],
        prefix=str(out_path),
        interp="linear",
        device=torch.device("cpu"),
        verb=0,
        auto_pad=False,
    )
    out = np.asarray(nib.load(str(out_path)).dataobj, dtype=np.float32)
    assert np.allclose(out[:, 2:-2, :, :], ref.numpy()[:, 2:-2, :, :], atol=1e-3)


def test_medic_mm_warp_5d_matches_undistort(tmp_path):
    """The 5D mm warp variant matches the per-frame files / undistort_series."""
    import nibabel as nib

    from fastfuncstuff.processing.medic import save_medic_warp
    from fastfuncstuff.processing.nwarpforge import nwarpforge

    nx = ny = nz = 12
    nt = 3
    series, _yy, affine = _ramp_phantom(nx, ny, nz, nt, voxel=2.0)
    disp = torch.zeros(nx, ny, nz, nt)
    for t, s in enumerate([0.0, 2.0, 1.0]):
        disp[..., t] = s
    ref = undistort_series(series, disp, pe_nifti_axis=1, interp="linear", verbose=False)

    src_path = tmp_path / "src.nii"
    nib.Nifti1Image(series.numpy(), affine).to_filename(str(src_path))
    spec = save_medic_warp(disp, 1, affine, str(tmp_path / "dc"), ".nii", as_5d=True)

    out_path = tmp_path / "out.nii"
    nwarpforge(
        source_path=str(src_path),
        nwarp_specs=[spec],
        prefix=str(out_path),
        interp="linear",
        device=torch.device("cpu"),
        verb=0,
        auto_pad=False,
    )
    out = np.asarray(nib.load(str(out_path)).dataobj, dtype=np.float32)
    assert np.allclose(out[:, 2:-2, :, :], ref.numpy()[:, 2:-2, :, :], atol=1e-3)


def test_medic_mm_warp_composes_to_master_grid(tmp_path):
    """The mm warp regrids to a different -master grid (the atlas-compose path)."""
    import nibabel as nib

    from fastfuncstuff.processing.medic import save_medic_warp
    from fastfuncstuff.processing.nwarpforge import nwarpforge

    nx = ny = nz = 14
    nt = 2
    series, yy, affine = _ramp_phantom(nx, ny, nz, nt, voxel=2.0)
    disp = torch.full((nx, ny, nz, nt), 1.0)  # constant 1-voxel pull along j

    src_path = tmp_path / "src.nii"
    nib.Nifti1Image(series.numpy(), affine).to_filename(str(src_path))
    # master: same affine/orientation, fewer voxels -> forces warp regridding
    master = np.zeros((10, 10, 10), dtype=np.float32)
    master_path = tmp_path / "master.nii"
    nib.Nifti1Image(master, affine).to_filename(str(master_path))
    spec = save_medic_warp(disp, 1, affine, str(tmp_path / "dc"), ".nii", as_5d=False)

    out_path = tmp_path / "out.nii"
    nwarpforge(
        source_path=str(src_path),
        nwarp_specs=[spec],
        prefix=str(out_path),
        master_path=str(master_path),
        interp="linear",
        device=torch.device("cpu"),
        verb=0,
        auto_pad=False,
    )
    out = np.asarray(nib.load(str(out_path)).dataobj, dtype=np.float32)
    assert out.shape[:3] == (10, 10, 10)
    # on the master grid the ramp is still y, shifted by +1 (interior)
    expected = yy.numpy()[:10, 1:8, :10] + 1.0
    assert np.allclose(out[:10, 1:8, :10, 0], expected, atol=1e-2)


def test_chain_time_varying_affine_and_warp_per_frame():
    """A chain with BOTH a time-varying affine and a time-varying warp composes
    per frame (the ffs_medic + ffs_moco one-step-to-atlas scenario).

    compose_chain([moco_affine, medic_warp], time_idx=t) must pick frame t from
    each: affine translation di[t] along i, warp translation dj[t] along j.
    """
    from fastfuncstuff.processing.nwarpforge import (
        AffineTransform,
        TimeVaryingWarp,
        compose_chain,
    )

    nz = ny = nx = 5
    nt = 3
    di = [0.0, 1.0, 2.0]  # per-frame i-shift from the affine
    dj = [0.0, 2.0, 4.0]  # per-frame j-shift from the warp
    affine = np.eye(4)

    # time-varying affine: per-frame translation di[t] along x (voxel space)
    mats = torch.eye(4).repeat(nt, 1, 1)
    for t in range(nt):
        mats[t, 0, 3] = di[t]
    aff = AffineTransform(matrices=mats)

    # time-varying warp: per-frame j-displacement dj[t] (nifti_mm, identity grid)
    zeros = torch.zeros(nt, nz, ny, nx)
    yd = torch.stack([torch.full((nz, ny, nx), dj[t]) for t in range(nt)])
    tvw = TimeVaryingWarp(
        xd=zeros.clone(),
        yd=yd,
        zd=zeros.clone(),
        header_info={"affine": affine},
        units="nifti_mm",
    )

    for t in range(nt):
        composed = compose_chain(
            [aff, tvw], (nz, ny, nx), affine, torch.device("cpu"), time_idx=t, verb=0
        )
        # interior, away from edges; displacement should be (di[t], dj[t], 0)
        assert torch.allclose(
            composed.xd[1:-1, 1:-1, 1:-1], torch.full((nz - 2, ny - 2, nx - 2), di[t]), atol=1e-3
        )
        assert torch.allclose(
            composed.yd[1:-1, 1:-1, 1:-1], torch.full((nz - 2, ny - 2, nx - 2), dj[t]), atol=1e-3
        )


def test_convert_medic_fieldmap_to_warp(tmp_path):
    """ffs_util_convert_medic: warpkit field map -> warp -> ffs_nwarp apply
    matches our undistort_series (the verified field-map conversion path)."""
    import nibabel as nib

    from fastfuncstuff.cli.util_convert_medic import main as convert_main
    from fastfuncstuff.processing.medic import (
        field_to_displacement_pe,
        invert_displacement_pe,
    )
    from fastfuncstuff.processing.nwarpforge import nwarpforge

    nx = ny = nz = 12
    nt = 2
    trt = 0.03
    affine = np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float64)

    field = torch.zeros(nx, ny, nz, nt)
    field[..., 0] = 20.0  # Hz
    field[..., 1] = -15.0
    nib.Nifti1Image(field.numpy(), affine).to_filename(str(tmp_path / "fmap.nii"))

    # expected pull warp + reference undistortion of a j-ramp phantom
    disp_native = field_to_displacement_pe(field, trt, "j-")
    disp_pull = torch.empty_like(disp_native)
    for t in range(nt):
        disp_pull[..., t] = invert_displacement_pe(disp_native[..., t], 1)
    yy = torch.arange(ny, dtype=torch.float32).view(1, ny, 1).expand(nx, ny, nz)
    phantom = yy.unsqueeze(-1).repeat(1, 1, 1, nt).contiguous()
    ref = undistort_series(phantom, disp_pull, 1, interp="linear", verbose=False)

    rc = convert_main(
        [
            "-fieldmap",
            str(tmp_path / "fmap.nii"),
            "-pe_dir",
            "j-",
            "-total_readout_time",
            str(trt),
            "-prefix",
            str(tmp_path / "dc"),
            "-verb",
            "0",
        ]
    )
    assert rc == 0

    nib.Nifti1Image(phantom.numpy(), affine).to_filename(str(tmp_path / "src.nii"))
    out_path = tmp_path / "out.nii"
    nwarpforge(
        source_path=str(tmp_path / "src.nii"),
        nwarp_specs=[str(tmp_path / "dc_warp" / "warp_*.nii.gz")],
        prefix=str(out_path),
        interp="linear",
        device=torch.device("cpu"),
        verb=0,
        auto_pad=False,
    )
    out = np.asarray(nib.load(str(out_path)).dataobj, dtype=np.float32)
    assert np.allclose(out[:, 2:-2, :, :], ref.numpy()[:, 2:-2, :, :], atol=1e-2)
