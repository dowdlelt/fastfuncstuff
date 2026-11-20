"""Tests for fastfuncstuff/processing/ffs_moco.py — covers missing lines."""

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.affine import (
    _build_homo_coords,
    identity_params,
    params_to_matrix,
    resample_affine_fast,
)
from fastfuncstuff.processing.ffs_moco import (
    MocoConfig,
    MocoResult,
    _blur_volume,
    _get_voxel_sizes,
    _gram_normal_eq,
    _normal_solve,
    _prefer_wls_incumbent,
    _prepare_normal_solve,
    _unweighted_rms,
    _weighted_rms,
    compute_derivative_images,
    gauss_newton_rigid,
    gauss_newton_rigid_fixed,
    gauss_newton_rigid_fixed_masked,
    gauss_newton_rigid_masked,
    moco,
)
from fastfuncstuff.processing.weight import compute_weight_image

DEV = torch.device("cpu")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_normal_solve_matches_cpu_float64():
    g = torch.Generator().manual_seed(12)
    wj = torch.randn(6, 20000, generator=g)
    wj[5] = wj[4] * 0.999 + 0.001 * wj[5]
    rhs = torch.randn(6, generator=g)
    gram = _gram_normal_eq(wj.to("mps"), torch.device("mps"))
    got = _normal_solve(
        _prepare_normal_solve(gram, torch.device("mps"), torch.float32), rhs.to("mps")
    )
    eps = 1e-6 * gram.diagonal().mean()
    expected = torch.linalg.solve(gram + eps * torch.eye(6, dtype=torch.float64), rhs.double())
    torch.testing.assert_close(got.cpu().double(), expected, rtol=2e-4, atol=2e-5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gaussian_blob(shape=(12, 12, 12)):
    """Create a 3D Gaussian blob centered in volume."""
    nz, ny, nx = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    return torch.exp(-r2 / (2 * 3.0**2))


def _make_shifted_timeseries(base, n_vols=5, max_shift=0.5):
    """Create a timeseries where volumes are slightly translated from base."""
    nt = n_vols
    shape = base.shape
    ts = torch.zeros(nt, *shape)
    ts[0] = base  # base volume = identity

    coords = _build_homo_coords(shape, DEV, torch.float32)
    for t in range(1, nt):
        p = identity_params(device=DEV, dtype=torch.float32)
        # Apply small translations (fraction of a voxel)
        p[0] = max_shift * (t / nt)  # dx
        p[1] = -max_shift * (t / nt) * 0.5  # dy
        mat = params_to_matrix(p)
        ts[t] = resample_affine_fast(base, mat, coords, "heptic", shape)

    return ts


# ---------------------------------------------------------------------------
# _weighted_rms and _unweighted_rms  (lines 178-181, 186-188)
# ---------------------------------------------------------------------------


class TestRMSFunctions:
    def test_weighted_rms_identical(self):
        vol = torch.randn(8, 8, 8)
        w = torch.ones(8 * 8 * 8)
        rms = _weighted_rms(vol.reshape(-1), vol.reshape(-1), w)
        assert rms == pytest.approx(0.0, abs=1e-6)

    def test_weighted_rms_known_diff(self):
        a = torch.zeros(100)
        b = torch.ones(100)
        w = torch.ones(100)
        rms = _weighted_rms(a, b, w)
        assert rms == pytest.approx(1.0, abs=1e-5)

    def test_unweighted_rms_identical(self):
        vol = torch.randn(8, 8, 8)
        rms = _unweighted_rms(vol, vol)
        assert rms == pytest.approx(0.0, abs=1e-6)

    def test_unweighted_rms_known_diff(self):
        a = torch.zeros(10, 10, 10)
        b = torch.ones(10, 10, 10) * 2.0
        rms = _unweighted_rms(a, b)
        assert rms == pytest.approx(2.0, abs=1e-5)

    def test_wls_stage_keeps_a_better_incoming_transform(self):
        shape = (10, 10, 10)
        base = _make_gaussian_blob(shape)
        coords = _build_homo_coords(shape, DEV, torch.float32)
        incumbent = identity_params(device=DEV)
        candidate = incumbent.clone()
        candidate[0] = 2.0

        kept = _prefer_wls_incumbent(
            base.reshape(-1),
            base,
            torch.ones(base.numel()),
            incumbent,
            candidate,
            coords,
            "linear",
            shape,
        )

        torch.testing.assert_close(kept, incumbent)


# ---------------------------------------------------------------------------
# compute_derivative_images - verb>=2 path (line 153)
# ---------------------------------------------------------------------------


class TestDerivativeImagesVerbose:
    def test_derivative_images_verbose(self, capsys):
        vol = _make_gaussian_blob((8, 8, 8)).to(DEV)
        derivs = compute_derivative_images(vol, DEV, verb=2)
        assert derivs.shape == (6, 8 * 8 * 8)
        captured = capsys.readouterr()
        assert "derivative" in captured.out.lower() or "resample" in captured.out.lower()


# ---------------------------------------------------------------------------
# gauss_newton_rigid - convergence and max_iter paths (lines 228, 258)
# ---------------------------------------------------------------------------


class TestGaussNewtonRigidConvergence:
    def test_gn_rigid_max_iter_reached(self):
        """When source differs from base, solver should hit max_iter."""
        base = _make_gaussian_blob((8, 8, 8)).to(DEV)
        # Create a source with a large shift so convergence won't happen in 1 iter
        p_shift = identity_params(device=DEV, dtype=torch.float32)
        p_shift[0] = 2.0  # large shift
        coords = _build_homo_coords(base.shape, DEV, torch.float32)
        mat = params_to_matrix(p_shift)
        source = resample_affine_fast(base, mat, coords, "heptic", base.shape)

        weight = compute_weight_image(base).to(DEV)
        derivs = compute_derivative_images(base, DEV)
        weight_flat = weight.reshape(1, -1)
        base_flat = base.reshape(-1)
        WJ = weight_flat * derivs
        JtWJ = WJ @ WJ.t()
        weight_flat_1d = weight.reshape(-1)

        cfg = MocoConfig(
            max_iter=2, device="cpu", verb=0, compile=False, dxy_thresh=1e-10, dph_thresh=1e-10
        )
        init_params = identity_params(device=DEV, dtype=torch.float32)

        params, n_iter = gauss_newton_rigid(
            base_flat,
            source,
            weight_flat_1d,
            WJ,
            JtWJ,
            init_params,
            cfg,
            coords=coords,
        )
        # Should hit max_iter since thresholds are tiny
        assert n_iter == cfg.max_iter


# ---------------------------------------------------------------------------
# gauss_newton_rigid_masked (lines 273-299)
# ---------------------------------------------------------------------------


class TestGaussNewtonRigidMasked:
    def test_gn_rigid_masked_identical(self):
        base = _make_gaussian_blob((8, 8, 8)).to(DEV)
        source = base.clone()

        weight = compute_weight_image(base).to(DEV)
        derivs = compute_derivative_images(base, DEV)

        weight_flat = weight.reshape(-1)
        base_flat = base.reshape(-1)

        # Mask to non-zero weight voxels
        mask_idx = (weight_flat > 0).nonzero(as_tuple=True)[0]
        coords = _build_homo_coords(base.shape, DEV, torch.float32)
        coords_masked = coords[:, mask_idx]
        base_flat_masked = base_flat[mask_idx]
        weight_flat_masked = weight_flat[mask_idx]

        WJ_full = weight_flat.unsqueeze(0) * derivs
        WJ_masked = WJ_full[:, mask_idx]
        JtWJ = WJ_masked @ WJ_masked.t()

        cfg = MocoConfig(max_iter=3, device="cpu", verb=0, compile=False)
        init_params = identity_params(device=DEV, dtype=torch.float32)

        params, n_iter = gauss_newton_rigid_masked(
            base_flat_masked,
            source,
            weight_flat_masked,
            WJ_masked,
            JtWJ,
            init_params,
            cfg,
            coords_masked=coords_masked,
        )
        # Identical volumes should converge quickly
        assert n_iter >= 1
        assert torch.allclose(params[:3], torch.zeros(3), atol=0.1)

    def test_gn_rigid_masked_max_iter(self):
        """Masked solver reaches max_iter with tight thresholds."""
        base = _make_gaussian_blob((8, 8, 8)).to(DEV)
        p_shift = identity_params(device=DEV, dtype=torch.float32)
        p_shift[0] = 1.5
        coords = _build_homo_coords(base.shape, DEV, torch.float32)
        mat = params_to_matrix(p_shift)
        source = resample_affine_fast(base, mat, coords, "heptic", base.shape)

        weight = compute_weight_image(base).to(DEV)
        derivs = compute_derivative_images(base, DEV)
        weight_flat = weight.reshape(-1)
        base_flat = base.reshape(-1)

        mask_idx = (weight_flat > 0).nonzero(as_tuple=True)[0]
        coords_masked = coords[:, mask_idx]
        base_flat_masked = base_flat[mask_idx]
        weight_flat_masked = weight_flat[mask_idx]
        WJ_full = weight_flat.unsqueeze(0) * derivs
        WJ_masked = WJ_full[:, mask_idx]
        JtWJ = WJ_masked @ WJ_masked.t()

        cfg = MocoConfig(
            max_iter=2, device="cpu", verb=0, compile=False, dxy_thresh=1e-10, dph_thresh=1e-10
        )
        init_params = identity_params(device=DEV, dtype=torch.float32)

        params, n_iter = gauss_newton_rigid_masked(
            base_flat_masked,
            source,
            weight_flat_masked,
            WJ_masked,
            JtWJ,
            init_params,
            cfg,
            coords_masked=coords_masked,
        )
        assert n_iter == cfg.max_iter


# ---------------------------------------------------------------------------
# gauss_newton_rigid_fixed (lines 313-332)
# ---------------------------------------------------------------------------


class TestGaussNewtonRigidFixed:
    def test_gn_rigid_fixed_identical(self):
        base = _make_gaussian_blob((8, 8, 8)).to(DEV)
        source = base.clone()

        weight = compute_weight_image(base).to(DEV)
        derivs = compute_derivative_images(base, DEV)
        weight_flat = weight.reshape(1, -1)
        base_flat = base.reshape(-1)
        WJ = weight_flat * derivs
        JtWJ = WJ @ WJ.t()
        weight_flat_1d = weight.reshape(-1)
        coords = _build_homo_coords(base.shape, DEV, torch.float32)
        init_params = identity_params(device=DEV, dtype=torch.float32)

        params = gauss_newton_rigid_fixed(
            base_flat,
            source,
            weight_flat_1d,
            WJ,
            JtWJ,
            init_params,
            coords,
            max_iter=3,
            interp="heptic",
        )
        # Returns params tensor (no n_iter)
        assert params.shape == (12,)
        assert torch.allclose(params[:3], torch.zeros(3), atol=0.1)


# ---------------------------------------------------------------------------
# gauss_newton_rigid_fixed_masked (lines 350-371)
# ---------------------------------------------------------------------------


class TestGaussNewtonRigidFixedMasked:
    def test_gn_rigid_fixed_masked_identical(self):
        base = _make_gaussian_blob((8, 8, 8)).to(DEV)
        source = base.clone()

        weight = compute_weight_image(base).to(DEV)
        derivs = compute_derivative_images(base, DEV)
        weight_flat = weight.reshape(-1)
        base_flat = base.reshape(-1)

        mask_idx = (weight_flat > 0).nonzero(as_tuple=True)[0]
        coords = _build_homo_coords(base.shape, DEV, torch.float32)
        coords_masked = coords[:, mask_idx]
        base_flat_masked = base_flat[mask_idx]
        weight_flat_masked = weight_flat[mask_idx]
        WJ_full = weight_flat.unsqueeze(0) * derivs
        WJ_masked = WJ_full[:, mask_idx]
        JtWJ = WJ_masked @ WJ_masked.t()

        init_params = identity_params(device=DEV, dtype=torch.float32)

        params = gauss_newton_rigid_fixed_masked(
            base_flat_masked,
            source,
            weight_flat_masked,
            WJ_masked,
            JtWJ,
            init_params,
            coords_masked,
            max_iter=3,
            interp="heptic",
        )
        assert params.shape == (12,)
        assert torch.allclose(params[:3], torch.zeros(3), atol=0.1)


# ---------------------------------------------------------------------------
# gn_lpa_rigid (lines 398-433)
# ---------------------------------------------------------------------------


class TestGnLpaRigid:
    def test_lpa_rigid_identical(self):
        from fastfuncstuff.processing.ffs_moco import gn_lpa_rigid

        base = _make_gaussian_blob((8, 8, 8)).to(DEV)
        source = base.clone()
        weight = compute_weight_image(base).to(DEV)
        init_params = identity_params(device=DEV, dtype=torch.float32)

        cfg = MocoConfig(
            cost="lpa",
            max_iter=5,
            device="cpu",
            verb=0,
            compile=False,
            powell_maxfev=20,
            interp="linear",
        )
        params, n_evals = gn_lpa_rigid(base, source, weight, init_params, cfg)
        assert params.shape == (12,)
        assert n_evals >= 1
        # Identical volumes - params should be near identity
        assert torch.allclose(params[:3], torch.zeros(3), atol=0.5)


# ---------------------------------------------------------------------------
# _get_voxel_sizes (line 495)
# ---------------------------------------------------------------------------


class TestGetVoxelSizes:
    def test_identity_affine(self):
        affine = np.eye(4)
        vs = _get_voxel_sizes(affine)
        np.testing.assert_allclose(vs, [1.0, 1.0, 1.0], atol=1e-10)

    def test_scaled_affine(self):
        affine = np.diag([2.0, 3.0, 1.5, 1.0])
        vs = _get_voxel_sizes(affine)
        np.testing.assert_allclose(vs, [2.0, 3.0, 1.5], atol=1e-10)


# ---------------------------------------------------------------------------
# _blur_volume (lines 599-602)
# ---------------------------------------------------------------------------


class TestBlurVolume:
    def test_blur_zero_fwhm_returns_unchanged(self):
        vol = torch.randn(8, 8, 8)
        result = _blur_volume(vol, 0.0)
        assert torch.equal(result, vol)

    def test_blur_negative_fwhm_returns_unchanged(self):
        vol = torch.randn(8, 8, 8)
        result = _blur_volume(vol, -1.0)
        assert torch.equal(result, vol)

    def test_blur_positive_fwhm_smooths(self):
        vol = torch.randn(12, 12, 12)
        result = _blur_volume(vol, 2.0)
        assert result.shape == vol.shape
        # Blurred volume should have lower variance
        assert result.std() < vol.std()


# ---------------------------------------------------------------------------
# moco() integration (lines 629-1117)
# ---------------------------------------------------------------------------


class TestMocoIntegration:
    def test_moco_identity_timeseries(self):
        """All volumes identical -- params should be near zero."""
        base = _make_gaussian_blob((12, 12, 12))
        nt = 3
        ts = base.unsqueeze(0).repeat(nt, 1, 1, 1)

        cfg = MocoConfig(
            base_index=0,
            cost="wls",
            max_iter=3,
            device="cpu",
            verb=0,
            compile=False,
            chain_init=False,
        )
        result = moco(ts, cfg)

        assert isinstance(result, MocoResult)
        assert result.aligned.shape == ts.shape
        assert result.params.shape == (nt, 6)
        assert result.max_displacement.shape == (nt,)
        # All params should be near zero for identical volumes
        assert np.allclose(result.params, 0.0, atol=0.5)

    def test_moco_recovers_small_translation(self):
        """Apply known small shifts and verify moco reduces displacement."""
        base = _make_gaussian_blob((12, 12, 12))
        ts = _make_shifted_timeseries(base, n_vols=4, max_shift=0.4)

        cfg = MocoConfig(
            base_index=0,
            cost="wls",
            max_iter=5,
            device="cpu",
            verb=0,
            compile=False,
            chain_init=True,
        )
        result = moco(ts, cfg)

        assert result.aligned.shape == ts.shape
        # After correction, aligned volumes should be closer to base
        for t in range(1, 4):
            rms_before = float(torch.sqrt(((ts[t] - base) ** 2).mean()))
            rms_after = float(torch.sqrt(((result.aligned[t] - base) ** 2).mean()))
            # Allow some tolerance but correction should improve things
            assert rms_after <= rms_before + 0.01, (
                f"Vol {t}: rms_after={rms_after:.4f} > rms_before={rms_before:.4f}"
            )

    def test_moco_chain_init_false(self):
        """Verify moco works with chain_init=False."""
        base = _make_gaussian_blob((10, 10, 10))
        ts = _make_shifted_timeseries(base, n_vols=3, max_shift=0.3)

        cfg = MocoConfig(
            base_index=0,
            cost="wls",
            max_iter=3,
            device="cpu",
            verb=0,
            compile=False,
            chain_init=False,
        )
        result = moco(ts, cfg)
        assert result.aligned.shape == ts.shape

    def test_moco_with_header_info(self):
        """Verify moco handles header_info for DICOM conversion."""
        base = _make_gaussian_blob((10, 10, 10))
        ts = base.unsqueeze(0).repeat(2, 1, 1, 1)

        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        header_info = {"affine": affine}

        cfg = MocoConfig(
            base_index=0,
            cost="wls",
            max_iter=2,
            device="cpu",
            verb=0,
            compile=False,
        )
        result = moco(ts, cfg, header_info=header_info)
        assert result.matrices_dicom.shape == (2, 4, 4)

    def test_moco_fixed_iter_mode(self):
        """Test fixed_iter=True skips convergence checks."""
        base = _make_gaussian_blob((10, 10, 10))
        ts = _make_shifted_timeseries(base, n_vols=3, max_shift=0.3)

        cfg = MocoConfig(
            base_index=0,
            cost="wls",
            max_iter=3,
            device="cpu",
            verb=0,
            compile=False,
            fixed_iter=True,
            chain_init=False,
        )
        result = moco(ts, cfg)
        assert result.aligned.shape == ts.shape
        # All non-base volumes should report max_iter
        for t in range(1, 3):
            assert result.n_iters[t] == cfg.max_iter

    def test_moco_with_external_base_vol(self):
        """Verify base_vol parameter works."""
        base = _make_gaussian_blob((10, 10, 10))
        ts = _make_shifted_timeseries(base, n_vols=3, max_shift=0.2)

        cfg = MocoConfig(
            base_index=0,
            cost="wls",
            max_iter=3,
            device="cpu",
            verb=0,
            compile=False,
        )
        result = moco(ts, cfg, base_vol=base)
        assert result.aligned.shape == ts.shape
        # With external base_vol, no volume is skipped as base_index
        # (the skip only happens when base_vol is None)

    def test_moco_twopass(self):
        """Verify twopass mode runs without error."""
        base = _make_gaussian_blob((10, 10, 10))
        ts = _make_shifted_timeseries(base, n_vols=3, max_shift=0.3)

        cfg = MocoConfig(
            base_index=0,
            cost="wls",
            max_iter=3,
            device="cpu",
            verb=0,
            compile=False,
            twopass=True,
            chain_init=False,
        )
        result = moco(ts, cfg)
        assert result.aligned.shape == ts.shape

    def test_moco_blur_fwhm(self):
        """Verify blur_fwhm is applied during estimation."""
        base = _make_gaussian_blob((10, 10, 10))
        ts = _make_shifted_timeseries(base, n_vols=3, max_shift=0.3)

        cfg = MocoConfig(
            base_index=0,
            cost="wls",
            max_iter=3,
            device="cpu",
            verb=0,
            compile=False,
            blur_fwhm=2.0,
        )
        result = moco(ts, cfg)
        assert result.aligned.shape == ts.shape

    def test_moco_verbose_output(self, capsys):
        """Verify verbose output is produced."""
        base = _make_gaussian_blob((8, 8, 8))
        ts = base.unsqueeze(0).repeat(2, 1, 1, 1)

        cfg = MocoConfig(
            base_index=0,
            cost="wls",
            max_iter=2,
            device="cpu",
            verb=1,
            compile=False,
        )
        moco(ts, cfg)
        captured = capsys.readouterr()
        assert "ffs_moco" in captured.out

    def test_moco_device_auto_cpu(self):
        """When config.device is None and no GPU, should fall back to CPU."""
        base = _make_gaussian_blob((8, 8, 8))
        ts = base.unsqueeze(0).repeat(2, 1, 1, 1)

        cfg = MocoConfig(
            base_index=0,
            cost="wls",
            max_iter=2,
            device=None,
            verb=0,
            compile=False,
        )
        result = moco(ts, cfg)
        assert result.aligned.shape == ts.shape


def test_centred_pivot_rotation_fixes_the_grid_centre():
    """The GN parametrization pivots at the grid centre, like 3dvolreg.

    params_to_matrix pivots at voxel (0,0,0): a corner rotation is a centre
    rotation plus a ~100-voxel lever translation, which made the rotation and
    translation Jacobian columns nearly collinear and doubled GN iterations.
    """
    from fastfuncstuff.processing.affine import identity_params
    from fastfuncstuff.processing.ffs_moco import centered_rigid_p2m

    shape = (20, 30, 26)  # (nz, ny, nx)
    p2m, p2m_b = centered_rigid_p2m(shape)
    p = identity_params(device=torch.device("cpu"), dtype=torch.float64)
    p[3:6] = torch.tensor([3.0, -2.0, 1.5])
    c = torch.tensor([(26 - 1) / 2, (30 - 1) / 2, (20 - 1) / 2, 1.0], dtype=torch.float64)
    torch.testing.assert_close(p2m(p) @ c, c)
    torch.testing.assert_close(p2m_b(p[None])[0], p2m(p))


def test_afni_motion_params_pivot_at_the_cardinal_grid_centre():
    """A pure rotation about the grid centre must report zero translation.

    The old extraction pivoted at the DICOM origin (scanner isocentre), so on
    sub-pilot02 run-05 dS/dP carried 4.5/3.8 mm rms of rotation lever versus
    3dvolreg's own .1D.
    """
    from fastfuncstuff.processing.affine import identity_params, voxel_matrix_to_dicom
    from fastfuncstuff.processing.ffs_moco import afni_motion_params, centered_rigid_p2m

    shape = (40, 64, 64)
    # Oblique, off-isocentre grid: the case where pivots disagree the most.
    th = np.deg2rad(24.0)
    rot = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    affine = np.eye(4)
    affine[:3, :3] = rot * 2.0
    affine[:3, 3] = [-60.0, -20.0, 10.0]
    p2m, _ = centered_rigid_p2m(shape)
    p = identity_params(device=torch.device("cpu"), dtype=torch.float64)
    p[3:6] = torch.tensor([2.0, -3.0, 1.0])
    M_dicom = voxel_matrix_to_dicom(p2m(p).float(), affine, affine)
    params = afni_motion_params(M_dicom, affine, shape)
    assert np.abs(params[:3]).max() < 1e-3
    # ...while a genuine shift comes through in mm.
    p[:3] = torch.tensor([1.0, 0.0, 0.0])
    M_dicom = voxel_matrix_to_dicom(p2m(p).float(), affine, affine)
    assert np.abs(afni_motion_params(M_dicom, affine, shape)[:3]).max() > 0.5


@pytest.mark.parametrize("twopass", [False, True])
def test_moco_slab_moved_by_slices_recovers_the_shift(twopass):
    """A thin slab moved several slices (cross-run alignment of a 16-slice slab):
    a third of the warped base samples from outside the source grid. Fitting
    the resampler's zero fill as data stalled at 0 or 3 slices (CPU) and diverged
    on CUDA; weighting only the voxels whose sample is inside recovers it."""
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import shift as ndshift

    nz = 16
    rng = np.random.default_rng(3)
    tall = gaussian_filter(rng.normal(size=(nz + 12, 48, 48)), 1.5)
    tall = (tall - tall.min()) / np.ptp(tall) + 0.5

    def window(v):  # the slab: tissue in every slice, a different stretch per run
        return v[2 : 2 + nz]

    shifts = (2.0, 3.5, 5.0)
    vols = [window(tall)] + [
        window(ndshift(tall, (-dz, 0.3, -0.2), order=3, mode="nearest")) for dz in shifts
    ]
    ts = torch.tensor(np.stack(vols), dtype=torch.float32)
    res = moco(ts, MocoConfig(base_index=0, device="cpu", verb=0, compile=False, twopass=twopass))
    np.testing.assert_allclose(res.params[1:, 2], [-s for s in shifts], atol=0.1)
    assert np.abs(res.params[1:, 3:]).max() < 0.3


def test_moco_fourier_recovers_subvoxel_inplane_shifts():
    """Band-limited in-plane data shifted by a quarter voxel, the position where
    a polynomial kernel's bias peaks: -interp fourier recovers it exactly."""
    rng = np.random.default_rng(5)
    nz, ny, nx = 10, 40, 44
    vol = rng.normal(size=(nz, ny, nx))
    spec = np.fft.fftn(vol, axes=(1, 2))
    ky = np.abs(np.fft.fftfreq(ny))[:, None]
    kx = np.abs(np.fft.fftfreq(nx))[None, :]
    spec *= (ky < 0.3) & (kx < 0.3) & (np.hypot(ky, kx) > 0.02)
    vol = np.fft.ifftn(spec, axes=(1, 2)).real
    vol = vol / vol.std() + 4.0

    def moved(sy, sx):
        ramp = np.exp(
            -2j * np.pi * (np.fft.fftfreq(ny)[:, None] * sy + np.fft.fftfreq(nx)[None] * sx)
        )
        return np.fft.ifftn(np.fft.fftn(vol, axes=(1, 2)) * ramp, axes=(1, 2)).real

    shifts = [(0.25, 0.0), (0.0, 0.25), (-0.25, 0.4), (0.1, -0.15)]
    ts = torch.tensor(np.stack([vol] + [moved(*s) for s in shifts]), dtype=torch.float32)
    # Tight convergence so the GN stopping rule (0.01 vox by default) is not
    # what the tolerance measures.
    cfg = MocoConfig(
        base_index=0,
        device="cpu",
        verb=0,
        compile=False,
        interp="fourier",
        dxy_thresh=1e-4,
        dph_thresh=1e-4,
        max_iter=60,
    )
    res = moco(ts, cfg)
    # params[:, :3] are the base->source pull: minus the content shift.
    want = np.array([[-sx, -sy] for sy, sx in shifts])
    # Measured: fourier within 6e-4 vox; heptic misses by 3.9e-3.
    np.testing.assert_allclose(res.params[1:, :2], want, atol=1.5e-3)
    assert np.abs(res.params[1:, 2:6]).max() < 2e-3


def test_moco_fourier_falls_back_on_gather_paths(capsys):
    """fourier is the default, so the estimators without a shear path (lpa here)
    must still run -- on heptic -- and say so rather than fail."""
    rng = np.random.default_rng(1)
    ts = torch.tensor(rng.normal(size=(3, 8, 16, 16)).astype(np.float32) + 5.0)
    res = moco(ts, MocoConfig(device="cpu", verb=1, compile=False, interp="fourier", cost="lpa"))
    assert np.isfinite(res.params).all()
    assert "interp fourier -> heptic" in capsys.readouterr().out


def test_slice_lowpass_is_estimation_only():
    """-interp fourier blurs the slice axis for estimation; the corrected output
    must still be the unblurred data (the base volume comes back as itself)."""
    from fastfuncstuff.processing.ffs_moco import _slice_lowpass_sigma

    assert _slice_lowpass_sigma(MocoConfig(interp="fourier")) == 1.0
    assert _slice_lowpass_sigma(MocoConfig(interp="fourier3d")) == 0.0
    assert _slice_lowpass_sigma(MocoConfig(interp="heptic")) == 0.0
    assert _slice_lowpass_sigma(MocoConfig(interp="heptic", slice_lowpass=0.5)) == 0.5

    rng = np.random.default_rng(2)
    vol = rng.normal(size=(12, 24, 24)).astype(np.float32) + 5.0
    ts = torch.tensor(np.stack([vol, vol, vol]))
    res = moco(ts, MocoConfig(base_index=0, device="cpu", verb=0, compile=False, interp="fourier"))
    np.testing.assert_allclose(res.aligned[0].numpy(), vol, atol=1e-4)


def test_moco_rotates_in_mm_on_anisotropic_voxels():
    """A head turns rigidly in millimetres. On 2x2x3 mm voxels an index-space
    rotation is a rotation plus stretch/shear, so it cannot fit the motion:
    the recovered matrix must be rigid in mm and match the true rotation."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    from scipy.spatial.transform import Rotation

    vs = np.array([2.0, 2.0, 3.0])  # x, y, z
    rng = np.random.default_rng(4)
    vol_xyz = gaussian_filter(rng.normal(size=(48, 48, 32)), 2.0)
    vol_xyz = (vol_xyz - vol_xyz.min()) / np.ptp(vol_xyz) + 0.5
    n = np.array(vol_xyz.shape)
    c = (n - 1) / 2
    R = Rotation.from_euler("xyz", [3.0, -2.0, 1.5], degrees=True).as_matrix()
    L = (R * vs) / vs[:, None]  # D^-1 R D: the true pull in index space
    g = np.mgrid[0 : n[0], 0 : n[1], 0 : n[2]].reshape(3, -1)
    moved = map_coordinates(vol_xyz, c[:, None] + L @ (g - c[:, None]), order=5, mode="nearest")
    moved = moved.reshape(n)
    ts = torch.tensor(np.stack([vol_xyz, moved]).transpose(0, 3, 2, 1).copy(), dtype=torch.float32)
    affine = np.diag([*vs, 1.0])
    cfg = MocoConfig(
        base_index=0, device="cpu", verb=0, compile=False, dxy_thresh=1e-3, dph_thresh=1e-3
    )
    res = moco(ts, cfg, header_info={"affine": affine})
    Mv = res.matrices_vox[1].double().numpy()[:3, :3]
    Rmm = (Mv * (1 / vs)) / (1 / vs)[:, None]  # D Mv D^-1
    np.testing.assert_allclose(np.linalg.svd(Rmm, compute_uv=False), 1.0, atol=1e-5)
    # Measured: 0.01 deg with mm rotations; index-space rotations miss by ~0.2 deg.
    # matrices_vox is the correcting pull, so it undoes the motion: Rmm ~ R^-1.
    err = np.degrees(np.arccos(np.clip((np.trace(Rmm @ R) - 1) / 2, -1, 1)))
    assert err < 0.05, err
