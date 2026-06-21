"""Tests for the AFNI-parity features of ffs_nwarp / nwarpforge.

Covers: AFNI-faithful wsinc5 (M3 window, floor -4..+5 stencil), nearest-neighbor
interpolation, exposed cubic/quintic, the -no_neg clamp, -master WARP grid
selection, and the auto-pad that prevents data loss on a warp.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import torch

from fastfuncstuff.processing import interp as I
from fastfuncstuff.processing.nwarpforge import (
    NonlinearWarp,
    _estimate_warp_padding,
    _pad_output_grid,
    apply_composed_warp,
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
        xd=shift, yd=torch.zeros_like(src), zd=torch.zeros_like(src),
        header_info={"affine": aff}, units="voxels",
    )
    plain = apply_composed_warp(src, warp, aff, aff, interp="wsinc5", no_neg=False)
    clamped = apply_composed_warp(src, warp, aff, aff, interp="wsinc5", no_neg=True)
    assert plain.min().item() < -1e-3, "expected wsinc5 undershoot to go negative"
    assert clamped.min().item() >= 0.0
    # clamp only touches the negatives; positive bulk unchanged
    assert torch.allclose(clamped.clamp_min(0), plain.clamp_min(0), atol=1e-4)


# --------------------------------------------------------------------------
# auto-pad helpers + end-to-end
# --------------------------------------------------------------------------


def test_estimate_padding_zero_for_zero_warp():
    zero = torch.zeros(6, 6, 6)
    warp = NonlinearWarp(
        xd=zero, yd=zero.clone(), zd=zero.clone(),
        header_info={"affine": _diag_affine()}, units="nifti_mm",
    )
    pad = _estimate_warp_padding([warp], (6, 6, 6), _diag_affine())
    assert pad == (0, 0, 0)


def test_estimate_padding_translation():
    # 8 mm constant displacement along y, 2 mm voxels -> 4 voxels of pad in y.
    zero = torch.zeros(6, 6, 6)
    warp = NonlinearWarp(
        xd=zero, yd=torch.full((6, 6, 6), 8.0), zd=zero.clone(),
        header_info={"affine": _diag_affine()}, units="nifti_mm",
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
        source_path=str(src_path), nwarp_specs=[str(warp_path)],
        prefix=str(out_pad), interp="linear", device=DEV, verb=0, auto_pad=True,
    )
    out_nopad = tmp_path / "out_nopad.nii"
    nwarpforge(
        source_path=str(src_path), nwarp_specs=[str(warp_path)],
        prefix=str(out_nopad), interp="linear", device=DEV, verb=0, auto_pad=False,
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
        source_path=str(src_path), nwarp_specs=[str(warp_path)],
        prefix=str(out), interp="linear", device=DEV, verb=0, auto_pad=True,
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
        source_path=str(src_path), nwarp_specs=[str(warp_path)],
        prefix=str(out), master_path="WARP", interp="linear", device=DEV,
        verb=0, auto_pad=False,
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
        d = torch.nn.functional.conv3d(
            d[None], torch.ones(3, 3, 3, 3, 3) / 81, padding=1
        )[0] * scale
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
            [trilinear_interpolate(w.xd, p[0], p[1], p[2]),
             trilinear_interpolate(w.yd, p[0], p[1], p[2]),
             trilinear_interpolate(w.zd, p[0], p[1], p[2])]
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
        (pts[0] > 2) & (pts[0] < nx - 3)
        & (pts[1] > 2) & (pts[1] < ny - 3)
        & (pts[2] > 2) & (pts[2] < nz - 3)
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
    d = torch.nn.functional.conv3d(
        torch.randn(1, 3, n, n, n), torch.ones(3, 1, 3, 3, 3) / 27, padding=1, groups=3
    )[0] * scale
    return NonlinearWarp(
        xd=d[0], yd=d[1], zd=d[2], header_info={"affine": aff}, units="nifti_mm"
    )


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
        _smooth_mm_warp(n, 0.7, aff, 1),     # static distortion
        AffineTransform(matrices=mats, base_affine=aff, source_affine=aff),  # motion
        _smooth_mm_warp(n, 0.5, aff, 2),     # static nonlinear-to-template
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
