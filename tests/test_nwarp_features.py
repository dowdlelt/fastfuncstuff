"""Tests for the AFNI-parity features of ffs_nwarp / nwarpforge.

Covers: AFNI-faithful wsinc5 (M3 window, floor -4..+5 stencil), nearest-neighbor
interpolation, exposed cubic/quintic, the -no_neg clamp, -master WARP grid
selection, and the auto-pad that prevents data loss on a warp.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.processing import interp as I
from fastfuncstuff.processing.nwarpforge import (
    NonlinearWarp,
    _estimate_warp_padding,
    _pad_output_grid,
    apply_composed_warp,
    make_warp_apply_plan,
    nwarpforge,
)

DEV = torch.device("cpu")


def _diag_affine(vox: float = 2.0) -> np.ndarray:
    a = np.eye(4, dtype=np.float64)
    a[0, 0] = a[1, 1] = a[2, 2] = vox
    return a


# --------------------------------------------------------------------------
# wsinc5: AFNI-faithful kernel
# --------------------------------------------------------------------------


def _afni_wsinc5_weights(fx: float) -> np.ndarray:
    """Independent reimplementation of AFNI GA_interp_wsinc5p 1D weights."""
    irad, wrad = 5, 5.001
    offs = np.arange(-(irad - 1), irad + 1)  # -4..+5
    d = np.abs(fx - offs)
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(d < 1e-7, 1.0, np.sin(np.pi * d) / (np.pi * d))
    t = d / wrad
    m3 = 0.4243801 + 0.4973406 * np.cos(np.pi * t) + 0.0782793 * np.cos(2 * np.pi * t)
    w = sinc * m3
    return w / w.sum()


def test_wsinc5_kernel_matches_afni_formula():
    for fx in (0.0, 0.1, 0.37, 0.5, 0.83, 0.999):
        ours = I._wsinc5_kernel(torch.tensor([fx], dtype=torch.float32))[0].numpy()
        ref = _afni_wsinc5_weights(fx)
        assert ours.shape == (10,)
        assert np.allclose(ours, ref, atol=1e-5), (fx, ours, ref)


def test_wsinc5_partition_of_unity():
    fx = torch.linspace(0, 0.999, 50)
    w = I._wsinc5_kernel(fx)
    assert torch.allclose(w.sum(dim=1), torch.ones(50), atol=1e-5)


def _afni_wsinc5_weights_env(
    fx: float, irad: int = 5, wcut: float = 0.0, hamming: bool = False
) -> np.ndarray:
    """AFNI 1D weights honoring IRAD/WCUT/TAPERFUN (mri_genalign_util.c)."""
    wrad = 0.001 + irad
    offs = np.arange(-(irad - 1), irad + 1)
    d = np.abs(fx - offs)
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(d < 1e-7, 1.0, np.sin(np.pi * d) / (np.pi * d))
    xw = d / wrad
    arg = np.pi * (xw - wcut) / (1.0 - wcut)
    if hamming:
        win = 0.53836 + 0.46164 * np.cos(arg)
    else:
        win = 0.4243801 + 0.4973406 * np.cos(arg) + 0.0782793 * np.cos(2 * arg)
    win = np.where(xw > wcut, win, 1.0)
    w = sinc * win
    return w / w.sum()


@pytest.mark.parametrize(
    "env,irad,wcut,hamming",
    [
        ({"AFNI_WSINC5_RADIUS": "9"}, 9, 0.0, False),
        ({"AFNI_WSINC5_TAPERCUT": "0.5"}, 5, 0.5, False),
        ({"AFNI_WSINC5_TAPERFUN": "H"}, 5, 0.0, True),
        ({"AFNI_WSINC5_RADIUS": "3", "AFNI_WSINC5_TAPERCUT": "0.2"}, 3, 0.2, False),
    ],
)
def test_wsinc5_kernel_honors_env(monkeypatch, env, irad, wcut, hamming):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    I._wsinc5_params.cache_clear()
    try:
        for fx in (0.0, 0.37, 0.83):
            ours = I._wsinc5_kernel(torch.tensor([fx], dtype=torch.float32))[0].numpy()
            ref = _afni_wsinc5_weights_env(fx, irad=irad, wcut=wcut, hamming=hamming)
            assert ours.shape == (2 * irad,)
            assert np.allclose(ours, ref, atol=1e-5), (env, fx, ours, ref)
    finally:
        I._wsinc5_params.cache_clear()


def test_wsinc5_spherical_unsupported(monkeypatch):
    monkeypatch.setenv("AFNI_WSINC5_SPHERICAL", "YES")
    I._wsinc5_params.cache_clear()
    try:
        with pytest.raises(NotImplementedError):
            I._wsinc5_kernel(torch.tensor([0.3], dtype=torch.float32))
    finally:
        I._wsinc5_params.cache_clear()


def _einsum_resample_reference(source, x, y, z, kernel_name):
    """One-shot ntaps**3 einsum contraction -- the pre-S5 reference the
    three-pass _separable_resample_3d must reproduce exactly."""
    kernel_fn, _, _ = I._KERNELS[kernel_name]
    H = I._kernel_half_width(kernel_name)
    nz, ny, nx = source.shape
    offsets = torch.arange(-(H - 1), H + 1)
    xb, yb, zb = x.floor(), y.floor(), z.floor()
    wx, wy, wz = kernel_fn(x - xb), kernel_fn(y - yb), kernel_fn(z - zb)
    xi = (xb.long()[:, None] + offsets).clamp(0, nx - 1)
    yi = (yb.long()[:, None] + offsets).clamp(0, ny - 1)
    zi = (zb.long()[:, None] + offsets).clamp(0, nz - 1)
    neigh = source[zi[:, :, None, None], yi[:, None, :, None], xi[:, None, None, :]]
    out = torch.einsum("ctuv,ct,cu,cv->c", neigh, wz, wy, wx)
    oob = (x < -0.5) | (x > nx - 0.5) | (y < -0.5) | (y > ny - 0.5) | (z < -0.5) | (z > nz - 0.5)
    out[oob] = 0.0
    return out


@pytest.mark.parametrize("kernel", ["wsinc5", "cubic", "quintic", "heptic"])
def test_separable_resample_matches_einsum(kernel):
    torch.manual_seed(0)
    vol = torch.rand(12, 13, 11)
    # random interior coords (avoid the <1e-4 ISTINY band so this stays a pure
    # reassociation check) plus a few deliberately out-of-bounds points
    n = 500
    x = torch.rand(n) * 10.0 + 0.3
    y = torch.rand(n) * 12.0 + 0.3
    z = torch.rand(n) * 11.0 + 0.3
    x[:5], y[:5], z[:5] = -3.0, 50.0, -1.0  # OOB
    ours = I._separable_resample_3d(vol, x, y, z, kernel)
    ref = _einsum_resample_reference(vol, x, y, z, kernel)
    assert torch.allclose(ours, ref, atol=1e-5), (kernel, (ours - ref).abs().max())


@pytest.mark.parametrize("kernel", ["wsinc5", "cubic", "quintic", "heptic"])
def test_resample_istiny_matches_full_kernel(kernel):
    """Grid-node (ISTINY) points return the source value, == the full kernel."""
    torch.manual_seed(1)
    vol = torch.rand(10, 10, 10)
    # exact nodes + near-nodes (inside the 1e-4 band) in the interior
    base = torch.tensor([3.0, 4.0, 5.0, 6.0])
    x = base + torch.tensor([0.0, 1e-6, -2e-5, 5e-5])
    y = base + torch.tensor([0.0, -1e-6, 3e-5, 2e-5])
    z = base + torch.tensor([0.0, 2e-6, -4e-5, 1e-5])
    out = I._separable_resample_3d(vol, x, y, z, kernel)
    nodes = vol[base.long(), base.long(), base.long()]
    assert torch.allclose(out, nodes, atol=1e-4), (kernel, out, nodes)


@pytest.mark.parametrize("kernel", ["wsinc5", "cubic", "quintic", "heptic"])
def test_resample_commensurate_grid_matches_einsum(kernel):
    """Regular (0.5-step) grid -> few unique fracs -> S1 weight-cache path is
    taken; result must stay bit-identical to the one-shot einsum reference."""
    torch.manual_seed(2)
    vol = torch.rand(14, 14, 14)
    g = torch.arange(1.0, 11.0, 0.5)  # 20 points/axis, frac in {0.0, 0.5}
    zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
    x, y, z = xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)
    ours = I._separable_resample_3d(vol, x, y, z, kernel)
    ref = _einsum_resample_reference(vol, x, y, z, kernel)
    assert torch.allclose(ours, ref, atol=1e-6), (kernel, (ours - ref).abs().max())


def test_gather_contract_dispatch_gating(monkeypatch):
    """Eager when disabled, on an unsupported device, or before the budget is met.

    CPU and CUDA both compile once accumulated eager time covers what a warmup
    costs; a device that is neither (e.g. MPS) always stays eager. The budget and
    calibration themselves are covered in test_interp_compile_gate.py.
    """
    cpu = torch.device("cpu")
    monkeypatch.setattr(I, "_eager_seconds", {"cpu": 0.0, "cuda": 0.0})
    monkeypatch.setattr(I, "_compiled_gather_contract", {})
    # no eager time spent yet -> eager
    assert I._get_gather_contract(cpu) is I._gather_contract
    # disabled via env -> eager even with the budget long since met
    I._eager_seconds["cpu"] = 1e6
    monkeypatch.setenv("FFS_NWARP_NO_COMPILE", "1")
    assert I._get_gather_contract(cpu) is I._gather_contract
    monkeypatch.delenv("FFS_NWARP_NO_COMPILE")
    # an unsupported device (mps) stays eager regardless
    assert I._get_gather_contract(torch.device("mps")) is I._gather_contract
    # a caller-declared one-shot block stays eager too
    with I.no_gather_compile():
        assert I._get_gather_contract(cpu) is I._gather_contract


def test_compiled_resample_matches_eager():
    """The torch.compile CPU path must produce the same values as eager."""
    try:
        compiled = torch.compile(I._gather_contract, dynamic=False)
    except Exception:
        import pytest as _pytest

        _pytest.skip("torch.compile unavailable")
    torch.manual_seed(3)
    vol = torch.rand(16, 16, 16)
    n = 4000
    x = torch.rand(n) * 12 + 1.5
    y = torch.rand(n) * 12 + 1.5
    z = torch.rand(n) * 12 + 1.5
    H = I._kernel_half_width("wsinc5")
    offs = torch.arange(-(H - 1), H + 1)
    xb, yb, zb = x.floor(), y.floor(), z.floor()
    wx = I._wsinc5_kernel(x - xb)
    wy = I._wsinc5_kernel(y - yb)
    wz = I._wsinc5_kernel(z - zb)
    xi = (xb.long()[:, None] + offs).clamp(0, 15)
    yi = (yb.long()[:, None] + offs).clamp(0, 15)
    zi = (zb.long()[:, None] + offs).clamp(0, 15)
    eager = I._gather_contract(vol, xi, yi, zi, wx, wy, wz)
    comp = compiled(vol, xi, yi, zi, wx, wy, wz)
    assert torch.allclose(eager, comp, atol=1e-5), (eager - comp).abs().max()


def test_resample_all_out_of_bounds_returns_zero():
    vol = torch.rand(8, 8, 8)
    x = torch.tensor([-5.0, 20.0])
    y = torch.tensor([-5.0, 20.0])
    z = torch.tensor([-5.0, 20.0])
    out = I._separable_resample_3d(vol, x, y, z, "wsinc5")
    assert torch.all(out == 0.0)


def test_wsinc5_resample_identity_at_integers():
    vol = torch.rand(9, 9, 9)
    kk, jj, ii = torch.meshgrid(
        torch.arange(9.0), torch.arange(9.0), torch.arange(9.0), indexing="ij"
    )
    out = I.wsinc5_resample_3d(vol, ii.reshape(-1), jj.reshape(-1), kk.reshape(-1))
    # interior should reproduce source exactly (edges clamp)
    out = out.reshape(9, 9, 9)
    assert torch.allclose(out[2:-2, 2:-2, 2:-2], vol[2:-2, 2:-2, 2:-2], atol=1e-4)


# --------------------------------------------------------------------------
# nearest-neighbor + mode exposure
# --------------------------------------------------------------------------


def test_normalize_interp_mode():
    assert I.normalize_interp_mode("NN") == "nearest"
    assert I.normalize_interp_mode("WSINC5") == "wsinc5"
    try:
        I.normalize_interp_mode("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_nearest_preserves_integer_labels():
    labels = torch.randint(0, 5, (8, 8, 8)).float()
    # half-voxel shift in every axis -> interpolating kernels would blend labels
    xd = torch.full((8, 8, 8), 0.5)
    out = I.warp_image(labels, xd, xd.clone(), xd.clone(), mode="NN")
    uniq = set(out.unique().tolist())
    assert uniq.issubset(set(range(5)) | {0.0}), uniq


def test_all_modes_run_and_identity():
    vol = torch.rand(10, 10, 10)
    z = torch.zeros(10, 10, 10)
    for mode in ("nearest", "linear", "cubic", "quintic", "heptic", "wsinc5"):
        out = I.warp_image(vol, z, z.clone(), z.clone(), mode=mode)
        assert out.shape == vol.shape
        assert torch.allclose(out[3:-3, 3:-3, 3:-3], vol[3:-3, 3:-3, 3:-3], atol=1e-4), mode


@pytest.mark.parametrize("mode", ["nearest", "linear", "cubic", "wsinc5"])
def test_multi_channel_warp_matches_scalar_loop(mode):
    """Shared-coordinate batching must remain identical to frame-wise apply."""
    torch.manual_seed(73)
    sources = [torch.randn(9, 10, 11) for _ in range(3)]
    xd = torch.randn(9, 10, 11) * 0.15
    yd = torch.randn(9, 10, 11) * 0.15
    zd = torch.randn(9, 10, 11) * 0.15

    expected = [I.warp_image(src, xd, yd, zd, mode=mode) for src in sources]
    got = I.warp_image_multi(sources, xd, yd, zd, mode=mode)

    assert len(got) == len(expected)
    for actual, reference in zip(got, expected, strict=True):
        assert torch.allclose(actual, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("mode", ["nearest", "linear", "cubic", "wsinc5"])
def test_absolute_coordinate_sampling_matches_displacement_path(mode):
    torch.manual_seed(74)
    source = torch.randn(9, 10, 11)
    xd = torch.randn(9, 10, 11) * 0.15
    yd = torch.randn(9, 10, 11) * 0.15
    zd = torch.randn(9, 10, 11) * 0.15
    kk, jj, ii = torch.meshgrid(
        torch.arange(9, dtype=torch.float32),
        torch.arange(10, dtype=torch.float32),
        torch.arange(11, dtype=torch.float32),
        indexing="ij",
    )

    expected = I.warp_image_multi([source], xd, yd, zd, mode=mode)
    got = I.warp_image_multi(
        [source], xd, yd, zd, mode=mode, sample_coords=(ii + xd, jj + yd, kk + zd)
    )
    assert torch.allclose(got[0], expected[0], atol=2e-5, rtol=2e-5)


def test_warp_apply_plan_reuses_geometry_without_changing_result():
    src = torch.randn(9, 10, 11)
    aff = _diag_affine()
    zero = torch.zeros_like(src)
    warp = NonlinearWarp(zero, zero.clone(), zero.clone(), {"affine": aff})
    plan = make_warp_apply_plan(warp.shape, aff, aff, DEV)
    expected = apply_composed_warp(src, warp, aff, aff, interp="linear")
    got = apply_composed_warp(src, warp, aff, aff, interp="linear", plan=plan)
    assert torch.equal(got, expected)


# --------------------------------------------------------------------------
# -no_neg
# --------------------------------------------------------------------------


def _step_edge(n: int = 16) -> torch.Tensor:
    v = torch.zeros(n, n, n)
    v[:, :, n // 2 :] = 100.0
    return v


def test_no_neg_clamps_ringing():
    src = _step_edge()
    aff = _diag_affine()
    # half-voxel constant shift -> wsinc5 rings (negative undershoot) at the edge
    shift = torch.full_like(src, 0.5)
    warp = NonlinearWarp(
        xd=shift,
        yd=torch.zeros_like(src),
        zd=torch.zeros_like(src),
        header_info={"affine": aff},
        units="voxels",
    )
    plain = apply_composed_warp(src, warp, aff, aff, interp="wsinc5", no_neg=False)
    clamped = apply_composed_warp(src, warp, aff, aff, interp="wsinc5", no_neg=True)
    assert plain.min().item() < -1e-3, "expected wsinc5 undershoot to go negative"
    assert clamped.min().item() >= 0.0
    # clamp only touches the negatives; positive bulk unchanged
    assert torch.allclose(clamped.clamp_min(0), plain.clamp_min(0), atol=1e-4)


# --------------------------------------------------------------------------
# -jac phase-encode Jacobian intensity modulation
# --------------------------------------------------------------------------


def test_parse_pe_axis_spellings():
    from fastfuncstuff.processing.nwarpforge import parse_pe_axis

    for a0 in ("i", "x", "LR", "rl", "X"):
        assert parse_pe_axis(a0) == 0
    for a1 in ("j", "y", "AP", "pa", "Y"):
        assert parse_pe_axis(a1) == 1
    for a2 in ("k", "z", "IS", "si", "Z"):
        assert parse_pe_axis(a2) == 2
    with pytest.raises(ValueError):
        parse_pe_axis("q")


def test_compute_pe_jacobian_single_axis_and_value():
    from fastfuncstuff.processing.nwarpforge import compute_pe_jacobian
    from fastfuncstuff.processing.penalty import _central_diff_batched

    # A pure y (j) displacement ramp: disp(y) = 0.3*y -> jac = 1 + 0.3 everywhere interior.
    nz, ny, nx = 8, 10, 9
    jj = torch.arange(ny, dtype=torch.float32)[None, :, None].expand(nz, ny, nx).contiguous()
    yd = 0.3 * jj
    zero = torch.zeros_like(yd)
    warp = NonlinearWarp(
        xd=zero, yd=yd, zd=zero.clone(), header_info={"affine": _diag_affine()}, units="voxels"
    )
    jac, single, ratio = compute_pe_jacobian(warp, pe_axis=1)
    assert single and ratio < 1e-6
    expected = 1.0 + _central_diff_batched(yd, dim=1)
    assert torch.allclose(jac, expected)
    assert abs(float(jac[nz // 2, ny // 2, nx // 2]) - 1.3) < 1e-5

    # Off-axis displacement -> not single-axis -> caller must skip.
    warp_mixed = NonlinearWarp(
        xd=0.3 * jj, yd=yd, zd=zero.clone(), header_info={"affine": _diag_affine()}, units="voxels"
    )
    _, single_mixed, ratio_mixed = compute_pe_jacobian(warp_mixed, pe_axis=1)
    assert not single_mixed and ratio_mixed > 0.5


def test_jac_modulation_end_to_end():
    # A compression ramp warp: applying -jac must scale intensity by the Jacobian,
    # reproducing (geometry-only output) * jac.
    aff = _diag_affine()
    nz, ny, nx = 6, 12, 6
    src = torch.ones(nz, ny, nx)
    jj = torch.arange(ny, dtype=torch.float32)[None, :, None].expand(nz, ny, nx).contiguous()
    yd = 0.2 * (jj - ny / 2)  # linear PE displacement
    warp = NonlinearWarp(
        xd=torch.zeros_like(src),
        yd=yd,
        zd=torch.zeros_like(src),
        header_info={"affine": aff},
        units="voxels",
    )
    geom = apply_composed_warp(src, warp, aff, aff, interp="linear")
    from fastfuncstuff.processing.nwarpforge import compute_pe_jacobian

    jac, single, _ = compute_pe_jacobian(warp, pe_axis=1)
    assert single
    # interior voxels (avoid the boundary difference rows)
    m = torch.zeros_like(src, dtype=torch.bool)
    m[:, 2:-2, :] = True
    assert torch.allclose((geom * jac)[m], (geom[m] * jac[m]))
    assert (jac[m] - 1.2).abs().max() < 1e-5  # 1 + d(0.2*y)/dy


def test_jac_static_with_slice_timing(tmp_path):
    # -jac must work with -tpattern when a static PE distortion warp is present: the
    # constant Jacobian is applied to every slice-timing-corrected frame.
    n, nt = 8, 6
    aff = _diag_affine(2.0)
    rng = np.random.default_rng(0)
    src = rng.random((nt, n, n, n)).astype(np.float32) + 1.0  # 4D, all positive
    src_path = tmp_path / "src4d.nii"
    nib.Nifti1Image(np.moveaxis(src, 0, -1), aff).to_filename(str(src_path))

    warp_path = tmp_path / "pewarp.nii"
    jj = np.arange(n, dtype=np.float32)[None, :, None]
    disp_mm = np.broadcast_to(0.2 * (jj - n / 2) * 2.0, (n, n, n))  # PE(y) ramp, mm
    data = np.zeros((n, n, n, 3), dtype=np.float32)
    data[..., 1] = disp_mm
    nib.Nifti1Image(data, aff).to_filename(str(warp_path))

    st = list(np.linspace(0.0, 0.9, n))
    common = dict(
        source_path=str(src_path),
        nwarp_specs=[str(warp_path)],
        interp="linear",
        device=DEV,
        verb=0,
        slice_times=st,
        tr=1.0,
        auto_pad=False,
    )
    out = tmp_path / "out.nii"
    out_nojac = tmp_path / "out_nojac.nii"
    # Should not raise, and should run the joint slice-timing path with jac applied.
    nwarpforge(prefix=str(out), jac_axis=1, **common)
    nwarpforge(prefix=str(out_nojac), jac_axis=None, **common)
    assert out.exists()
    res = np.asarray(nib.load(str(out)).dataobj)
    res0 = np.asarray(nib.load(str(out_nojac)).dataobj)
    assert res.shape[:3] == (n, n, n) and np.isfinite(res).all()
    # The constant Jacobian scales every frame. disp_mm gradient 0.4/voxel over 2mm
    # voxels = 0.2 voxel/voxel; the AFNI mm->voxel load negates the y component, so the
    # applied gradient is -0.2 and jac = 1 - 0.2 = 0.8 (spatially constant for a ramp).
    assert not np.allclose(res, res0)
    interior = (slice(None), slice(2, n - 2), slice(None))
    ratio = res[interior] / np.clip(res0[interior], 1e-3, None)
    assert abs(float(np.median(ratio)) - 0.8) < 0.05
    assert float(ratio.std()) < 0.02  # constant modulation across the ramp


def _write_aff1d(path, rows):
    """Write an AFNI .aff12.1D file: one line per frame, 12 numbers (3x4)."""
    with open(path, "w") as fh:
        for r in rows:
            fh.write(" ".join(f"{v:.8f}" for v in r) + "\n")


_IDENT12 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]


def test_match_nwarp_spec():
    from fastfuncstuff.processing.nwarpforge import match_nwarp_spec

    specs = ["/a/ref2anat.aff12.1D", "/b/sub_fmap_warp.nii.gz", "/c/run1_motion.aff12.1D"]
    assert match_nwarp_spec(specs, "sub_fmap_warp.nii.gz") == 1  # basename
    assert match_nwarp_spec(specs, "fmap") == 1  # unique substring
    assert match_nwarp_spec(specs, "/c/run1_motion.aff12.1D") == 2  # exact path
    with pytest.raises(ValueError, match="no -nwarp entry"):
        match_nwarp_spec(specs, "nope")
    with pytest.raises(ValueError, match="multiple"):
        match_nwarp_spec(specs, ".1D")  # matches two


def test_jac_transport_with_upstream_motion(tmp_path):
    # Chain [fieldmap, per-frame motion]: motion is upstream (source side) of the fieldmap,
    # so -jac j:fmap must succeed and apply the (constant) transported Jacobian per frame.
    n, nt = 8, 4
    aff = _diag_affine(2.0)
    src = (np.random.default_rng(0).random((nt, n, n, n)) + 1.0).astype(np.float32)
    src_path = tmp_path / "s.nii"
    nib.Nifti1Image(np.moveaxis(src, 0, -1), aff).to_filename(str(src_path))

    fmap = tmp_path / "fmap.nii"
    ramp = np.zeros((n, n, n, 3), dtype=np.float32)
    ramp[..., 1] = 0.4 * (np.arange(n, dtype=np.float32)[None, :, None] - n / 2)  # PE(y) ramp, mm
    nib.Nifti1Image(ramp, aff).to_filename(str(fmap))  # constant jac (=/= 1)
    motion = tmp_path / "run_motion.aff12.1D"
    _write_aff1d(motion, [_IDENT12 for _ in range(nt)])  # per-frame (identity here)

    out = tmp_path / "o.nii"
    out0 = tmp_path / "o0.nii"
    common = dict(
        source_path=str(src_path),
        nwarp_specs=[str(fmap), str(motion)],
        interp="linear",
        device=DEV,
        verb=0,
        auto_pad=False,
    )
    nwarpforge(prefix=str(out), jac_axis=1, jac_match="fmap", **common)
    nwarpforge(prefix=str(out0), jac_axis=None, **common)
    res = np.asarray(nib.load(str(out)).dataobj)
    res0 = np.asarray(nib.load(str(out0)).dataobj)
    assert res.shape[:3] == (n, n, n)
    assert not np.allclose(res, res0)  # jac was applied
    interior = (slice(None), slice(2, n - 2), slice(None))
    ratio = res[interior] / np.clip(res0[interior], 1e-3, None)
    assert float(np.std(ratio)) < 0.02  # constant modulation across frames


def test_jac_transport_errors_on_downstream_perframe(tmp_path):
    # Chain [per-frame motion, fieldmap]: motion is DOWNSTREAM (output side) of the
    # fieldmap -> the transported Jacobian would be per-frame -> must error.
    n, nt = 6, 3
    aff = _diag_affine(2.0)
    src = (np.random.default_rng(1).random((nt, n, n, n)) + 1.0).astype(np.float32)
    src_path = tmp_path / "s.nii"
    nib.Nifti1Image(np.moveaxis(src, 0, -1), aff).to_filename(str(src_path))
    fmap = tmp_path / "fmap.nii"
    _write_mm_warp(fmap, (n, n, n), aff, (0.0, 8.0, 0.0))
    motion = tmp_path / "m.aff12.1D"
    _write_aff1d(motion, [_IDENT12 for _ in range(nt)])
    with pytest.raises(ValueError, match="downstream"):
        nwarpforge(
            source_path=str(src_path),
            nwarp_specs=[str(motion), str(fmap)],
            prefix=str(tmp_path / "o.nii"),
            interp="linear",
            device=DEV,
            verb=0,
            auto_pad=False,
            jac_axis=1,
            jac_match="fmap",
        )


def test_jac_timevarying_plus_slicetiming_errors(tmp_path):
    # A purely time-varying (5D) warp + slice timing + jac has no static Jacobian: error.
    import pytest as _pytest

    from fastfuncstuff.processing.nwarpforge import _is_time_varying_warp

    n, nt = 6, 4
    aff = _diag_affine(2.0)
    src = (np.random.default_rng(1).random((nt, n, n, n)) + 1.0).astype(np.float32)
    src_path = tmp_path / "s.nii"
    nib.Nifti1Image(np.moveaxis(src, 0, -1), aff).to_filename(str(src_path))
    # 5D per-frame warp (nx,ny,nz,T,3), PE ramp
    w = np.zeros((n, n, n, nt, 3), dtype=np.float32)
    w[..., 1] = (0.2 * (np.arange(n, dtype=np.float32)[None, :, None, None] - n / 2)) * 2.0
    wpath = tmp_path / "tvwarp.nii"
    nib.Nifti1Image(w, aff).to_filename(str(wpath))
    assert _is_time_varying_warp(str(wpath))
    with _pytest.raises(ValueError, match="static distortion warp"):
        nwarpforge(
            source_path=str(src_path),
            nwarp_specs=[str(wpath)],
            prefix=str(tmp_path / "o.nii"),
            interp="linear",
            device=DEV,
            verb=0,
            slice_times=list(np.linspace(0, 0.9, n)),
            tr=1.0,
            jac_axis=1,
        )


# --------------------------------------------------------------------------
# auto-pad helpers + end-to-end
# --------------------------------------------------------------------------


def test_estimate_padding_zero_for_zero_warp():
    zero = torch.zeros(6, 6, 6)
    warp = NonlinearWarp(
        xd=zero,
        yd=zero.clone(),
        zd=zero.clone(),
        header_info={"affine": _diag_affine()},
        units="nifti_mm",
    )
    pad = _estimate_warp_padding([warp], (6, 6, 6), _diag_affine())
    assert pad == (0, 0, 0)


def test_estimate_padding_translation():
    # 8 mm constant displacement along y, 2 mm voxels -> 4 voxels of pad in y.
    zero = torch.zeros(6, 6, 6)
    warp = NonlinearWarp(
        xd=zero,
        yd=torch.full((6, 6, 6), 8.0),
        zd=zero.clone(),
        header_info={"affine": _diag_affine()},
        units="nifti_mm",
    )
    pad = _estimate_warp_padding([warp], (6, 6, 6), _diag_affine())
    assert pad == (0, 4, 0), pad


def test_pad_output_grid_preserves_world_coords():
    shape = (6, 6, 6)
    aff = _diag_affine()
    new_shape, new_aff = _pad_output_grid(shape, aff, (1, 2, 3))
    assert new_shape == (6 + 6, 6 + 4, 6 + 2)  # (nz+2*pz, ny+2*py, nx+2*px)
    # world coordinate of original voxel (0,0,0) == new voxel (px,py,pz)
    old0 = aff @ np.array([0, 0, 0, 1.0])
    new_idx = new_aff @ np.array([1, 2, 3, 1.0])  # (px,py,pz)
    assert np.allclose(old0, new_idx)


def _write_mm_warp(path, shape, affine, disp_mm):
    """Write a constant-displacement warp in DICOM mm (nx,ny,nz,3)."""
    nx, ny, nz = shape
    data = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    data[..., 0] = disp_mm[0]
    data[..., 1] = disp_mm[1]
    data[..., 2] = disp_mm[2]
    nib.Nifti1Image(data, affine).to_filename(str(path))


def test_autopad_grows_grid(tmp_path):
    n = 12
    aff = _diag_affine(2.0)
    src = np.zeros((n, n, n), dtype=np.float32)
    src[:, n - 4 : n - 1, :] = 50.0  # bright slab near the high-y edge
    src_path = tmp_path / "src.nii"
    nib.Nifti1Image(src, aff).to_filename(str(src_path))

    warp_path = tmp_path / "warp.nii"
    # 10mm -> 5 vox pull in y: the high-y slab lands off the grid without padding.
    _write_mm_warp(warp_path, (n, n, n), aff, (0.0, 10.0, 0.0))

    out_pad = tmp_path / "out_pad.nii"
    nwarpforge(
        source_path=str(src_path),
        nwarp_specs=[str(warp_path)],
        prefix=str(out_pad),
        interp="linear",
        device=DEV,
        verb=0,
        auto_pad=True,
    )
    out_nopad = tmp_path / "out_nopad.nii"
    nwarpforge(
        source_path=str(src_path),
        nwarp_specs=[str(warp_path)],
        prefix=str(out_nopad),
        interp="linear",
        device=DEV,
        verb=0,
        auto_pad=False,
    )
    padded = np.asarray(nib.load(str(out_pad)).dataobj)
    plain = np.asarray(nib.load(str(out_nopad)).dataobj)
    # auto-pad grew the y dimension; no-pad kept the source grid
    assert plain.shape == (n, n, n)
    assert padded.shape[1] > n
    # without padding the slab is pushed off the grid and lost; padding recovers it
    assert plain.sum() < 1e-3
    assert padded.sum() > 1e3


def test_autopad_noop_when_source_fits(tmp_path):
    """No padding when the warped source stays within the output grid.

    Regression: padding was driven by raw displacement, so a warp with a large
    translation-like field grew the grid (and runtime) even though all data was
    captured. Here a small in-FOV warp must leave the grid untouched.
    """
    n = 16
    aff = _diag_affine(2.0)
    src = np.zeros((n, n, n), dtype=np.float32)
    src[4:12, 4:12, 4:12] = 30.0  # well inside the FOV, with margin
    src_path = tmp_path / "src.nii"
    nib.Nifti1Image(src, aff).to_filename(str(src_path))
    warp_path = tmp_path / "warp.nii"
    _write_mm_warp(warp_path, (n, n, n), aff, (0.0, 2.0, 0.0))  # 1-voxel pull, in-FOV

    out = tmp_path / "out.nii"
    nwarpforge(
        source_path=str(src_path),
        nwarp_specs=[str(warp_path)],
        prefix=str(out),
        interp="linear",
        device=DEV,
        verb=0,
        auto_pad=True,
    )
    got = np.asarray(nib.load(str(out)).dataobj)
    assert got.shape[:3] == (n, n, n)  # grid unchanged: nothing was at risk


def test_master_warp_uses_warp_grid(tmp_path):
    aff = _diag_affine(2.0)
    src = np.random.rand(8, 8, 8).astype(np.float32)
    src_path = tmp_path / "src.nii"
    nib.Nifti1Image(src, aff).to_filename(str(src_path))
    # warp defined on a *different* grid (10^3)
    warp_path = tmp_path / "warp.nii"
    _write_mm_warp(warp_path, (10, 10, 10), aff, (0.0, 0.0, 0.0))

    out = tmp_path / "out.nii"
    nwarpforge(
        source_path=str(src_path),
        nwarp_specs=[str(warp_path)],
        prefix=str(out),
        master_path="WARP",
        interp="linear",
        device=DEV,
        verb=0,
        auto_pad=False,
    )
    got = np.asarray(nib.load(str(out)).dataobj)
    assert got.shape[:3] == (10, 10, 10)


# --------------------------------------------------------------------------
# interleaved warp/affine chain (distortion + motion + cross-run + anat + MNI)
# --------------------------------------------------------------------------


def test_interleaved_chain_matches_sequential():
    """A mixed [warp, per-frame affine, warp, affine] chain composes to exactly
    the sequential application of each transform's map.

    Backs the realistic pipeline: distortion warp -> per-volume motion affine ->
    cross-run nonlinear -> affine-to-anat, with one nonlinear-to-MNI on top. The
    first-listed transform is innermost (applied first to the output coordinate),
    matching 3dNwarpApply's N(x) = last(...first(x)). Uses a unit affine so the
    stored NIfTI-mm warp equals the voxel displacement.
    """
    from fastfuncstuff.processing.interp import trilinear_interpolate
    from fastfuncstuff.processing.nwarpforge import AffineTransform, compose_chain

    torch.manual_seed(0)
    nz = ny = nx = 16
    aff = np.eye(4)

    def smooth_warp(scale):
        d = torch.randn(3, nz, ny, nx)
        d = (
            torch.nn.functional.conv3d(d[None], torch.ones(3, 3, 3, 3, 3) / 81, padding=1)[0]
            * scale
        )
        return NonlinearWarp(
            xd=d[0], yd=d[1], zd=d[2], header_info={"affine": aff}, units="nifti_mm"
        )

    def rand_affine(T):
        mats = torch.zeros(T, 4, 4)
        for t in range(T):
            m = torch.eye(4)
            m[:3, :3] += 0.03 * torch.randn(3, 3)
            m[:3, 3] = 0.5 * torch.randn(3) + torch.tensor([t * 0.2, 0.0, 0.0])
            mats[t] = m
        return AffineTransform(matrices=mats, base_affine=aff, source_affine=aff)

    transforms = [smooth_warp(0.8), rand_affine(4), smooth_warp(0.6), rand_affine(1)]

    def sample(w, p):
        return torch.stack(
            [
                trilinear_interpolate(w.xd, p[0], p[1], p[2]),
                trilinear_interpolate(w.yd, p[0], p[1], p[2]),
                trilinear_interpolate(w.zd, p[0], p[1], p[2]),
            ]
        )

    def manual_N(p, t):
        for xf in transforms:
            if isinstance(xf, AffineTransform):
                ph = torch.cat([p, torch.ones(1, p.shape[1])])
                p = (xf.at_time(t) @ ph)[:3]
            else:
                p = p + sample(xf, p)
        return p

    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    pts = torch.stack([ii.reshape(-1), jj.reshape(-1), kk.reshape(-1)])
    interior = (
        (pts[0] > 2)
        & (pts[0] < nx - 3)
        & (pts[1] > 2)
        & (pts[1] < ny - 3)
        & (pts[2] > 2)
        & (pts[2] < nz - 3)
    )

    for t in (0, 2, 3):  # 3 reuses the last (T=4) motion row; static affine fixed
        comp = compose_chain(transforms, (nz, ny, nx), aff, DEV, time_idx=t, verb=0)
        nmap = torch.stack(
            [(ii + comp.xd).reshape(-1), (jj + comp.yd).reshape(-1), (kk + comp.zd).reshape(-1)]
        )
        ref = manual_N(pts, t)
        err = (nmap[:, interior] - ref[:, interior]).abs().max().item()
        assert err < 1e-4, (t, err)


# --------------------------------------------------------------------------
# static-tail pre-reduction + higher-order warp interpolation
# --------------------------------------------------------------------------


def _smooth_mm_warp(n, scale, aff, seed):
    torch.manual_seed(seed)
    d = (
        torch.nn.functional.conv3d(
            torch.randn(1, 3, n, n, n), torch.ones(3, 1, 3, 3, 3) / 27, padding=1, groups=3
        )[0]
        * scale
    )
    return NonlinearWarp(xd=d[0], yd=d[1], zd=d[2], header_info={"affine": aff}, units="nifti_mm")


def test_reduce_chain_matches_full_per_frame():
    """reduce_chain (collapse static runs) yields the same composed warp as the
    full chain, frame by frame, with a time-dependent affine in the middle."""
    from fastfuncstuff.processing.nwarpforge import (
        AffineTransform,
        compose_chain,
        reduce_chain,
    )

    n = 16
    aff = np.eye(4)
    mats = torch.zeros(4, 4, 4)
    for t in range(4):
        m = torch.eye(4)
        m[:3, :3] += 0.02 * torch.randn(3, 3)
        m[:3, 3] = torch.tensor([t * 0.3, -0.2 * t, 0.1])
        mats[t] = m
    transforms = [
        _smooth_mm_warp(n, 0.7, aff, 1),  # static distortion
        AffineTransform(matrices=mats, base_affine=aff, source_affine=aff),  # motion
        _smooth_mm_warp(n, 0.5, aff, 2),  # static nonlinear-to-template
    ]
    reduced = reduce_chain(transforms, (n, n, n), aff, DEV, interp="cubic", verb=0)
    # one static warp on each side of the time-dependent affine -> 3 slots
    assert len(reduced) == 3
    for t in (0, 1, 3):
        full = compose_chain(transforms, (n, n, n), aff, DEV, time_idx=t, interp="cubic", verb=0)
        red = compose_chain(reduced, (n, n, n), aff, DEV, time_idx=t, interp="cubic", verb=0)
        m = slice(3, -3)
        assert torch.allclose(full.xd[m, m, m], red.xd[m, m, m], atol=1e-4)
        assert torch.allclose(full.yd[m, m, m], red.yd[m, m, m], atol=1e-4)
        assert torch.allclose(full.zd[m, m, m], red.zd[m, m, m], atol=1e-4)


def test_reduce_chain_all_static_single_slot():
    from fastfuncstuff.processing.nwarpforge import AffineTransform, reduce_chain

    n = 12
    aff = np.eye(4)
    one = torch.eye(4)[None]
    transforms = [
        _smooth_mm_warp(n, 0.5, aff, 3),
        AffineTransform(matrices=one, base_affine=aff, source_affine=aff),
        _smooth_mm_warp(n, 0.4, aff, 4),
    ]
    reduced = reduce_chain(transforms, (n, n, n), aff, DEV, interp="cubic", verb=0)
    assert len(reduced) == 1  # all static -> collapsed to one warp
    assert isinstance(reduced[0], NonlinearWarp)


def test_sample_field_edge_extends():
    """Higher-order warp-field sampling edge-extends (no zero-fill tear) so a
    constant displacement stays constant past the grid, like grid_sample border."""
    from fastfuncstuff.processing.nwarpforge import _sample_field

    field = torch.full((8, 8, 8), 3.0)
    # coords that run off the high edge in x
    x = torch.tensor([7.0, 8.5, 12.0])
    y = torch.tensor([4.0, 4.0, 4.0])
    z = torch.tensor([4.0, 4.0, 4.0])
    for mode in ("cubic", "quintic", "wsinc5"):
        out = _sample_field(field, x, y, z, mode)
        assert torch.allclose(out, torch.full((3,), 3.0), atol=1e-4), mode
