"""Tests for processing/quadrature.py — quadrature phase-based registration."""

import math

import torch

from fastfuncstuff.processing.affine import identity_params
from fastfuncstuff.processing.quadrature import (
    apply_quadrature_filters_fft,
    build_phase_normal_equations,
    compute_phase_diff_and_certainty,
    design_quadrature_filters,
    precompute_filter_ffts,
    quadrature_gn_rigid,
    quadrature_gn_rigid_fixed,
    quadrature_phase_cost,
)

DEV = torch.device("cpu")


def _make_coords(vol_shape):
    """Build (4, N) homogeneous coordinate grid."""
    nz, ny, nx = vol_shape
    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32, device=DEV),
        torch.arange(ny, dtype=torch.float32, device=DEV),
        torch.arange(nx, dtype=torch.float32, device=DEV),
        indexing="ij",
    )
    N = nz * ny * nx
    coords = torch.stack(
        [
            ii.reshape(-1),
            jj.reshape(-1),
            kk.reshape(-1),
            torch.ones(N, device=DEV),
        ]
    )
    return coords


# ── design_quadrature_filters ──


class TestDesignQuadratureFilters:
    def test_output_shape(self):
        f = design_quadrature_filters(size=7, device=DEV)
        assert f.shape == (3, 7, 7, 7)

    def test_complex_dtype(self):
        f = design_quadrature_filters(size=7, device=DEV)
        assert f.is_complex()

    def test_different_sizes(self):
        for sz in [5, 7, 9]:
            f = design_quadrature_filters(size=sz, device=DEV)
            assert f.shape == (3, sz, sz, sz)

    def test_nonzero(self):
        f = design_quadrature_filters(size=7, device=DEV)
        assert f.abs().sum().item() > 0

    def test_custom_params(self):
        f = design_quadrature_filters(
            size=7,
            center_freq=math.pi / 4.0,
            bandwidth=1.5,
            device=DEV,
        )
        assert f.shape == (3, 7, 7, 7)


# ── precompute_filter_ffts ──


class TestPrecomputeFilterFFTs:
    def test_output_shape(self):
        filters = design_quadrature_filters(size=7, device=DEV)
        vol_shape = (10, 12, 14)
        spectra = precompute_filter_ffts(filters, vol_shape)
        assert spectra.shape == (3, 10, 12, 14)

    def test_complex_output(self):
        filters = design_quadrature_filters(size=7, device=DEV)
        spectra = precompute_filter_ffts(filters, (8, 8, 8))
        assert spectra.is_complex()


# ── apply_quadrature_filters_fft ──


class TestApplyQuadratureFiltersFFT:
    def test_output_shape(self):
        vol_shape = (10, 12, 14)
        filters = design_quadrature_filters(size=7, device=DEV)
        spectra = precompute_filter_ffts(filters, vol_shape)
        vol = torch.randn(*vol_shape, device=DEV)
        responses = apply_quadrature_filters_fft(vol, spectra)
        assert responses.shape == (3, *vol_shape)

    def test_complex_response(self):
        vol_shape = (8, 8, 8)
        filters = design_quadrature_filters(size=7, device=DEV)
        spectra = precompute_filter_ffts(filters, vol_shape)
        vol = torch.randn(*vol_shape, device=DEV)
        responses = apply_quadrature_filters_fft(vol, spectra)
        assert responses.is_complex()

    def test_zero_volume_gives_zero_response(self):
        vol_shape = (8, 8, 8)
        filters = design_quadrature_filters(size=7, device=DEV)
        spectra = precompute_filter_ffts(filters, vol_shape)
        vol = torch.zeros(*vol_shape, device=DEV)
        responses = apply_quadrature_filters_fft(vol, spectra)
        assert responses.abs().max().item() < 1e-10


# ── compute_phase_diff_and_certainty ──


class TestPhaseDiffAndCertainty:
    def test_identical_zero_phase_diff(self):
        vol_shape = (8, 8, 8)
        filters = design_quadrature_filters(size=7, device=DEV)
        spectra = precompute_filter_ffts(filters, vol_shape)
        torch.manual_seed(0)
        vol = torch.randn(*vol_shape, device=DEV) + 5.0
        q = apply_quadrature_filters_fft(vol, spectra)
        pd, cert = compute_phase_diff_and_certainty(q, q)
        N = 8 * 8 * 8
        assert pd.shape == (3, N)
        assert cert.shape == (3, N)
        # Phase diff should be zero for identical inputs
        assert pd.abs().max().item() < 1e-5

    def test_certainty_nonnegative(self):
        vol_shape = (8, 8, 8)
        filters = design_quadrature_filters(size=7, device=DEV)
        spectra = precompute_filter_ffts(filters, vol_shape)
        torch.manual_seed(1)
        v1 = torch.randn(*vol_shape, device=DEV)
        v2 = torch.randn(*vol_shape, device=DEV)
        q1 = apply_quadrature_filters_fft(v1, spectra)
        q2 = apply_quadrature_filters_fft(v2, spectra)
        _, cert = compute_phase_diff_and_certainty(q1, q2)
        assert (cert >= -1e-10).all()

    def test_high_certainty_for_identical(self):
        vol_shape = (8, 8, 8)
        filters = design_quadrature_filters(size=7, device=DEV)
        spectra = precompute_filter_ffts(filters, vol_shape)
        torch.manual_seed(2)
        vol = torch.randn(*vol_shape, device=DEV) + 5.0
        q = apply_quadrature_filters_fft(vol, spectra)
        _, cert = compute_phase_diff_and_certainty(q, q)
        # Should have high certainty where there's signal
        assert cert.mean().item() > 0


# ── build_phase_normal_equations ──


class TestBuildPhaseNormalEquations:
    def test_output_shapes(self):
        N = 100
        phase_diff = torch.randn(3, N, device=DEV)
        certainty = torch.ones(3, N, device=DEV)
        coords = torch.randn(4, N, device=DEV)
        coords[3] = 1.0
        vol_shape = (5, 5, 4)  # just for center computation
        A, h_vec = build_phase_normal_equations(phase_diff, certainty, coords, vol_shape)
        assert A.shape == (6, 6)
        assert h_vec.shape == (6,)

    def test_symmetric_A(self):
        N = 200
        phase_diff = torch.randn(3, N, device=DEV)
        certainty = torch.rand(3, N, device=DEV)
        coords = torch.randn(4, N, device=DEV)
        coords[3] = 1.0
        A, _ = build_phase_normal_equations(phase_diff, certainty, coords, (8, 8, 8))
        torch.testing.assert_close(A, A.T, atol=1e-5, rtol=1e-5)

    def test_zero_phase_diff_gives_zero_h(self):
        N = 100
        phase_diff = torch.zeros(3, N, device=DEV)
        certainty = torch.ones(3, N, device=DEV)
        coords = torch.randn(4, N, device=DEV)
        coords[3] = 1.0
        _, h_vec = build_phase_normal_equations(phase_diff, certainty, coords, (5, 5, 4))
        assert h_vec.abs().max().item() < 1e-10

    def test_positive_semidefinite_A(self):
        N = 300
        phase_diff = torch.randn(3, N, device=DEV)
        certainty = torch.rand(3, N, device=DEV).clamp(min=0)
        coords = torch.randn(4, N, device=DEV)
        coords[3] = 1.0
        A, _ = build_phase_normal_equations(phase_diff, certainty, coords, (8, 8, 8))
        eigenvalues = torch.linalg.eigvalsh(A)
        assert (eigenvalues >= -1e-6).all()


# ── quadrature_gn_rigid ──


class TestQuadratureGNRigid:
    def _setup(self, vol_shape=(10, 12, 14)):
        torch.manual_seed(3)
        vol = torch.randn(*vol_shape, device=DEV) + 5.0
        filters = design_quadrature_filters(size=7, device=DEV)
        spectra = precompute_filter_ffts(filters, vol_shape)
        q_base = apply_quadrature_filters_fft(vol, spectra)
        coords = _make_coords(vol_shape)
        init_p = identity_params(device=DEV)
        return vol, q_base, spectra, coords, init_p

    def test_identical_converges_immediately(self):
        vol, q_base, spectra, coords, init_p = self._setup()
        vol_shape = vol.shape
        params, n_iters = quadrature_gn_rigid(
            vol,
            vol,
            q_base,
            spectra,
            init_p,
            coords,
            vol_shape,
            max_iter=5,
        )
        assert params.shape == (12,)
        # Should converge quickly with no displacement
        assert n_iters <= 2

    def test_with_weight(self):
        vol, q_base, spectra, coords, init_p = self._setup()
        vol_shape = vol.shape
        weight = torch.ones(vol_shape, device=DEV)
        params, n_iters = quadrature_gn_rigid(
            vol,
            vol,
            q_base,
            spectra,
            init_p,
            coords,
            vol_shape,
            max_iter=3,
            weight=weight,
        )
        assert params.shape == (12,)

    def test_returns_max_iter_when_not_converged(self):
        vol, q_base, spectra, coords, init_p = self._setup((8, 8, 8))
        vol_shape = vol.shape
        # Use a different source so it doesn't converge in 1 iter
        source = torch.randn(*vol_shape, device=DEV)
        _, n_iters = quadrature_gn_rigid(
            vol,
            source,
            q_base,
            spectra,
            init_p,
            coords,
            vol_shape,
            max_iter=2,
            dxy_thresh=1e-10,
            dph_thresh=1e-10,
        )
        assert n_iters == 2


# ── quadrature_gn_rigid_fixed ──


class TestQuadratureGNRigidFixed:
    def test_runs_fixed_iterations(self):
        torch.manual_seed(4)
        vol_shape = (8, 10, 12)
        vol = torch.randn(*vol_shape, device=DEV) + 5.0
        filters = design_quadrature_filters(size=7, device=DEV)
        spectra = precompute_filter_ffts(filters, vol_shape)
        q_base = apply_quadrature_filters_fft(vol, spectra)
        coords = _make_coords(vol_shape)
        init_p = identity_params(device=DEV)

        params = quadrature_gn_rigid_fixed(
            vol,
            q_base,
            spectra,
            init_p,
            coords,
            vol_shape,
            max_iter=3,
            interp="heptic",
            weight_flat=None,
        )
        assert params.shape == (12,)

    def test_with_weight_flat(self):
        torch.manual_seed(5)
        vol_shape = (8, 8, 8)
        vol = torch.randn(*vol_shape, device=DEV) + 5.0
        filters = design_quadrature_filters(size=7, device=DEV)
        spectra = precompute_filter_ffts(filters, vol_shape)
        q_base = apply_quadrature_filters_fft(vol, spectra)
        coords = _make_coords(vol_shape)
        init_p = identity_params(device=DEV)
        N = 8 * 8 * 8
        weight_flat = torch.ones(1, N, device=DEV)

        params = quadrature_gn_rigid_fixed(
            vol,
            q_base,
            spectra,
            init_p,
            coords,
            vol_shape,
            max_iter=2,
            interp="heptic",
            weight_flat=weight_flat,
        )
        assert params.shape == (12,)


# ── quadrature_phase_cost ──


class TestQuadraturePhaseCost:
    def test_identical_gives_max_cost(self):
        torch.manual_seed(6)
        vol_shape = (8, 8, 8)
        vol = torch.randn(*vol_shape, device=DEV) + 5.0
        filters = design_quadrature_filters(size=7, device=DEV)
        spectra = precompute_filter_ffts(filters, vol_shape)
        q = apply_quadrature_filters_fft(vol, spectra)

        cost_same = quadrature_phase_cost(q, q)
        # Different source should give lower cost
        vol2 = torch.randn(*vol_shape, device=DEV) + 5.0
        q2 = apply_quadrature_filters_fft(vol2, spectra)
        cost_diff = quadrature_phase_cost(q, q2)

        assert cost_same.item() > cost_diff.item()

    def test_with_weight(self):
        torch.manual_seed(7)
        vol_shape = (8, 8, 8)
        vol = torch.randn(*vol_shape, device=DEV) + 5.0
        filters = design_quadrature_filters(size=7, device=DEV)
        spectra = precompute_filter_ffts(filters, vol_shape)
        q = apply_quadrature_filters_fft(vol, spectra)
        weight = torch.ones(vol_shape, device=DEV)

        cost = quadrature_phase_cost(q, q, weight=weight)
        assert cost.item() > 0

    def test_scalar_output(self):
        torch.manual_seed(8)
        vol_shape = (8, 8, 8)
        vol = torch.randn(*vol_shape, device=DEV)
        filters = design_quadrature_filters(size=7, device=DEV)
        spectra = precompute_filter_ffts(filters, vol_shape)
        q = apply_quadrature_filters_fft(vol, spectra)
        cost = quadrature_phase_cost(q, q)
        assert cost.ndim == 0
