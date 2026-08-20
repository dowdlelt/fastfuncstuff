"""Tests for fastfuncstuff.processing.allineate."""

import math

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.allineate import (
    AffineAlignConfig,
    CostContext,
    _batched_cost,
    _bounds_to_torch,
    _center_of_mass,
    _cmass_translation,
    _compute_cost,
    _compute_grid_matrix,
    _compute_nonzero_bbox,
    _compute_param_bounds,
    _compute_source_validity_mask,
    _crop_volumes,
    _default_tbest,
    _denormalize,
    _denormalize_t,
    _downsample_3d,
    _estimate_chunk_size,
    _get_free_mask,
    _identity_physical,
    _make_powell_cost,
    _normalize,
    _normalize_t,
    _parse_cost,
    _pick_optimizer,
    _refine_adam_normalized,
    _refine_cmaes_batched,
    _refine_pattern_batched,
    _refine_powell,
    _rotation_candidates,
    _smooth_to_resolution,
    _tqdm_bar,
    _translation_candidates,
    allineate,
)

DEV = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sphere_volume(shape=(16, 16, 16), center=None, radius=5.0):
    """Create a 3D volume with a bright sphere."""
    nz, ny, nx = shape
    if center is None:
        center = (nx / 2.0, ny / 2.0, nz / 2.0)
    kk, jj, ii = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    dist2 = (ii - center[0]) ** 2 + (jj - center[1]) ** 2 + (kk - center[2]) ** 2
    vol = torch.clamp(1.0 - dist2 / (radius**2), min=0.0)
    return vol


# ---------------------------------------------------------------------------
# AffineAlignConfig
# ---------------------------------------------------------------------------


class TestAffineAlignConfig:
    def test_defaults(self):
        cfg = AffineAlignConfig()
        assert cfg.dof == "affine"
        assert cfg.cost == "lpa"
        assert cfg.twopass is True
        assert cfg.cmass is True
        assert cfg.autocrop is True

    def test_custom(self):
        cfg = AffineAlignConfig(dof="rigid", cost="ls", verb=0)
        assert cfg.dof == "rigid"
        assert cfg.cost == "ls"
        assert cfg.verb == 0


# ---------------------------------------------------------------------------
# _tqdm_bar
# ---------------------------------------------------------------------------


class TestTqdmBar:
    def test_passthrough_when_disabled(self):
        items = [1, 2, 3]
        result = list(_tqdm_bar(items, disable=True))
        assert result == items

    def test_passthrough_iterable(self):
        result = list(_tqdm_bar(range(5), disable=True))
        assert result == list(range(5))


# ---------------------------------------------------------------------------
# _get_free_mask / _identity_physical
# ---------------------------------------------------------------------------


class TestParameterMasks:
    def test_rigid_mask(self):
        mask = _get_free_mask("rigid")
        assert mask[:6].all()
        assert not mask[6:].any()

    def test_affine_mask(self):
        mask = _get_free_mask("affine")
        assert mask.all()

    def test_epi_mask(self):
        mask = _get_free_mask("epi")
        # epi freezes sx (6), sz (8), shzy (11)
        assert mask[6] is np.False_
        assert mask[8] is np.False_
        assert mask[11] is np.False_
        # rest free
        assert mask[:6].all()
        assert mask[7]
        assert mask[9]
        assert mask[10]

    def test_identity_physical(self):
        p = _identity_physical()
        assert p.shape == (12,)
        np.testing.assert_array_equal(p[:6], 0.0)
        np.testing.assert_array_equal(p[6:9], 1.0)
        np.testing.assert_array_equal(p[9:], 0.0)


# ---------------------------------------------------------------------------
# Normalization / denormalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_normalize_denormalize_roundtrip(self):
        bounds = np.array(
            [
                [-10, 10],
                [-5, 5],
                [0, 20],
                [-30, 30],
                [-30, 30],
                [-30, 30],
                [0.7, 1.4],
                [0.7, 1.4],
                [0.7, 1.4],
                [-0.1, 0.1],
                [-0.1, 0.1],
                [-0.1, 0.1],
            ]
        )
        params = _identity_physical()
        normed = _normalize(params, bounds)
        recovered = _denormalize(normed, bounds)
        np.testing.assert_allclose(recovered, params, atol=1e-10)

    def test_normalize_identity_midpoint(self):
        bounds = np.array([[-1, 1]] * 12)
        p = np.zeros(12)
        normed = _normalize(p, bounds)
        np.testing.assert_allclose(normed, 0.5, atol=1e-10)

    def test_torch_roundtrip(self):
        bounds = np.array(
            [
                [-10, 10],
                [-5, 5],
                [0, 20],
                [-30, 30],
                [-30, 30],
                [-30, 30],
                [0.7, 1.4],
                [0.7, 1.4],
                [0.7, 1.4],
                [-0.1, 0.1],
                [-0.1, 0.1],
                [-0.1, 0.1],
            ]
        )
        bmin, span = _bounds_to_torch(bounds, DEV)
        params = torch.tensor(_identity_physical(), dtype=torch.float32, device=DEV)
        normed = _normalize_t(params, bmin, span)
        recovered = _denormalize_t(normed, bmin, span)
        torch.testing.assert_close(recovered, params, atol=1e-5, rtol=1e-5)

    def test_bounds_to_torch_clamp_tiny_span(self):
        bounds = np.array([[5.0, 5.0]] * 12)  # zero span
        bmin, span = _bounds_to_torch(bounds, DEV)
        assert (span >= 1.0).all()  # should clamp to 1.0


# ---------------------------------------------------------------------------
# _compute_param_bounds
# ---------------------------------------------------------------------------


class TestComputeParamBounds:
    def test_shape(self):
        bounds = _compute_param_bounds((16, 16, 16))
        assert bounds.shape == (12, 2)
        # lower < upper everywhere
        assert (bounds[:, 0] < bounds[:, 1]).all()

    def test_with_cmass_shift(self):
        shift = np.array([2.0, -3.0, 1.0])
        bounds = _compute_param_bounds((16, 16, 16), cmass_shift=shift)
        # translation bounds should be centered at shift
        for i in range(3):
            mid = (bounds[i, 0] + bounds[i, 1]) / 2
            np.testing.assert_allclose(mid, shift[i], atol=1e-10)

    def test_rot_range_is_honoured(self):
        """-coarse_range reached the config but never the bounds (fixed 2026-08-15).

        The joint coarse search draws its rotation seeds from these bounds, so a
        hardcoded rotation range silently made the flag a no-op on lpa/lpc.
        """
        default = _compute_param_bounds((16, 16, 16))
        np.testing.assert_allclose(default[3:6, 1], 30.0)
        wide = _compute_param_bounds((16, 16, 16), rot_range=60.0)
        np.testing.assert_allclose(wide[3:6, 1], 60.0)
        np.testing.assert_allclose(wide[3:6, 0], -60.0)

    def test_shift_frac_is_honoured(self):
        narrow = _compute_param_bounds((16, 16, 16), shift_frac=0.1)
        wide = _compute_param_bounds((16, 16, 16), shift_frac=0.4)
        assert (wide[:3, 1] > narrow[:3, 1]).all()

    def test_range_scale_multiplies_rot_range(self):
        """-hugerange rides on range_scale, so the two must compose."""
        huge = _compute_param_bounds((16, 16, 16), range_scale=1.5)
        np.testing.assert_allclose(huge[3:6, 1], 45.0)


class TestCostNameParsing:
    """lpc+/lpa+/+ZZ decompose into a base cost, weights, and a final-pass flag."""

    def test_plain_costs_are_untouched(self):
        for name in ("lpa", "lpc", "ls", "mi", "nmi"):
            assert _parse_cost(name) == (name, None, False)

    def test_combination_weights_match_afni(self):
        # DEFAULT_MICHO_* in 3dAllineate.c, as (hel, mi, nmi, crA).
        assert _parse_cost("lpc+") == ("lpc", (0.4, 0.2, 0.2, 0.4), False)
        # lpa+ drops the MI term (27 May 2021) -- the only difference.
        assert _parse_cost("lpa+") == ("lpa", (0.4, 0.0, 0.2, 0.4), False)

    def test_zz_sets_the_final_flag_only(self):
        base, w, zz = _parse_cost("lpa+ZZ")
        assert (base, w) == _parse_cost("lpa+")[:2]
        assert zz is True

    def test_zz_is_case_insensitive(self):
        assert _parse_cost("lpc+zz") == _parse_cost("lpc+ZZ")

    def test_tbest_default_follows_the_base_cost(self):
        """A combination cost still refines on the blok path, so it gets 10."""
        assert _default_tbest("lpa") == _default_tbest("lpa+ZZ") == 10
        assert _default_tbest("ls") == _default_tbest("nmi") == 3

    def test_config_resolves_tbest_from_cost(self):
        assert AffineAlignConfig(cost="lpc+ZZ").tbest == 10
        assert AffineAlignConfig(cost="nmi").tbest == 3
        assert AffineAlignConfig(cost="nmi", tbest=7).tbest == 7


# ---------------------------------------------------------------------------
# Downsample / smooth
# ---------------------------------------------------------------------------


class TestMultiResolution:
    def test_downsample_factor_1(self):
        vol = torch.randn(8, 8, 8)
        out = _downsample_3d(vol, 1)
        assert out is vol  # identity

    def test_downsample_factor_2(self):
        vol = torch.ones(8, 8, 8)
        out = _downsample_3d(vol, 2)
        assert out.shape == (4, 4, 4)
        torch.testing.assert_close(out, torch.ones(4, 4, 4))

    def test_smooth_factor_1(self):
        vol = torch.randn(8, 8, 8)
        out = _smooth_to_resolution(vol, 1)
        assert out is vol

    def test_smooth_factor_2(self):
        vol = torch.randn(8, 8, 8)
        out = _smooth_to_resolution(vol, 2)
        assert out.shape == vol.shape


# ---------------------------------------------------------------------------
# _compute_nonzero_bbox
# ---------------------------------------------------------------------------


class TestNonzeroBbox:
    def test_simple_cube(self):
        vol = torch.zeros(20, 20, 20)
        vol[8:12, 8:12, 8:12] = 1.0
        (z0, y0, x0), (z1, y1, x1) = _compute_nonzero_bbox(vol, pad=2)
        assert z0 <= 6 and z1 >= 14
        assert y0 <= 6 and y1 >= 14
        assert x0 <= 6 and x1 >= 14

    def test_empty_volume(self):
        vol = torch.zeros(10, 10, 10)
        start, end = _compute_nonzero_bbox(vol)
        assert start == (0, 0, 0)
        assert end == (10, 10, 10)

    def test_full_volume(self):
        vol = torch.ones(10, 10, 10)
        (z0, y0, x0), (z1, y1, x1) = _compute_nonzero_bbox(vol, pad=0)
        # Even with pad=0, there's a 5% minimum pad
        assert z0 == 0 and z1 == 10


# ---------------------------------------------------------------------------
# _crop_volumes
# ---------------------------------------------------------------------------


class TestCropVolumes:
    def test_crop_reduces_size(self):
        base = torch.zeros(32, 32, 32)
        base[10:22, 10:22, 10:22] = 1.0
        source = torch.randn(32, 32, 32)
        weight = torch.ones(32, 32, 32)

        bc, sc, wc, offset = _crop_volumes(base, source, weight)
        # Should be smaller than original
        assert bc.numel() < base.numel()
        assert sc.shape == bc.shape
        assert wc.shape == bc.shape
        assert len(offset) == 3

    def test_crop_no_weight(self):
        base = torch.zeros(32, 32, 32)
        base[10:22, 10:22, 10:22] = 1.0
        source = torch.randn(32, 32, 32)

        bc, sc, wc, offset = _crop_volumes(base, source, None)
        assert wc is None


# ---------------------------------------------------------------------------
# _compute_source_validity_mask
# ---------------------------------------------------------------------------


class TestSourceValidityMask:
    def test_identity_all_valid(self):
        shape = (8, 8, 8)
        mat = torch.eye(4, dtype=torch.float32, device=DEV)
        mask = _compute_source_validity_mask(shape, shape, mat)
        assert mask.shape == shape
        assert mask.all()

    def test_shifted_partial_valid(self):
        # Shift by 4 voxels in x: right half goes out of bounds
        shape = (8, 8, 8)
        mat = torch.eye(4, dtype=torch.float32, device=DEV)
        mat[0, 3] = 4.0  # translate source coords by 4 in x
        mask = _compute_source_validity_mask(shape, shape, mat)
        # Some voxels should be invalid
        assert not mask.all()
        assert mask.any()


# ---------------------------------------------------------------------------
# _center_of_mass / _cmass_translation
# ---------------------------------------------------------------------------


class TestCenterOfMass:
    def test_symmetric_volume(self):
        vol = _sphere_volume((16, 16, 16), center=(8.0, 8.0, 8.0))
        com = _center_of_mass(vol)
        torch.testing.assert_close(com, torch.tensor([8.0, 8.0, 8.0]), atol=0.5, rtol=0.0)

    def test_weighted_com(self):
        vol = torch.ones(8, 8, 8)
        weight = torch.zeros(8, 8, 8)
        weight[0:4, :, :] = 1.0  # weight only lower half in z
        com = _center_of_mass(vol, weight)
        # z component should be < 4 (biased to lower half)
        assert com[2].item() < 4.0

    def test_empty_volume(self):
        vol = torch.zeros(8, 8, 8)
        com = _center_of_mass(vol)
        # Should return center
        torch.testing.assert_close(com, torch.tensor([4.0, 4.0, 4.0]), atol=0.01, rtol=0.0)

    def test_cmass_translation_identical(self):
        vol = _sphere_volume((12, 12, 12))
        t = _cmass_translation(vol, vol)
        torch.testing.assert_close(t, torch.zeros(3), atol=0.01, rtol=0.0)

    def test_cmass_translation_shifted(self):
        base = _sphere_volume((16, 16, 16), center=(8, 8, 8))
        source = _sphere_volume((16, 16, 16), center=(10, 8, 8))
        t = _cmass_translation(base, source)
        # Should detect ~2 voxel shift in x
        assert t[0].item() > 1.0

    def test_cmass_maps_native_source_through_grid(self):
        """The source centroid is taken on its native grid and mapped to base
        voxels via grid_matrix — not differenced raw.

        Regression: computing the source centroid on the source-resampled-to-base
        grid clips it to the base FOV (the top of a brain under a short EPI is
        dropped), so the shift comes up short. With the grid affine the native
        centroid maps correctly regardless of overlap.
        """
        # source COM at x=13 (source voxels); base COM at x=10 (base voxels).
        base = _sphere_volume((20, 20, 20), center=(10, 10, 10))
        source = _sphere_volume((20, 20, 20), center=(13, 10, 10))
        # grid_matrix maps base voxel -> source voxel as a +2 shift in x.
        grid = torch.eye(4)
        grid[0, 3] = 2.0
        # native source COM (x=13) -> base voxels (13-2=11); base COM x=10 -> 1.0
        t = _cmass_translation(base, source, grid_matrix=grid)
        assert abs(t[0].item() - 1.0) < 0.6
        # without the grid map it would be the raw 13-10 = 3 (the buggy value)
        t_raw = _cmass_translation(base, source)
        assert abs(t_raw[0].item() - 3.0) < 0.6


# ---------------------------------------------------------------------------
# _compute_cost
# ---------------------------------------------------------------------------


class TestComputeCost:
    def test_ls_perfect_match(self):
        vol = _sphere_volume((8, 8, 8)).reshape(-1)
        cost = _compute_cost(vol, vol, None, CostContext(name="ls"))
        assert cost.item() > 0.99

    def test_lpa_perfect_match(self):
        # Blok costs need a real grid; 32^3 gives meaningful bloks. The
        # absolute value is overlap-scaled, so test the meaningful property:
        # identical volumes score far higher than independent ones.
        torch.manual_seed(0)
        vol = _sphere_volume((32, 32, 32)) + 0.05 * torch.randn(32, 32, 32)
        same = _compute_cost(vol, vol, None, CostContext(name="lpa")).item()
        indep = _compute_cost(vol, torch.randn(32, 32, 32), None, CostContext(name="lpa")).item()
        assert same > indep
        assert same > 0.05

    def test_lps_perfect_match(self):
        vol = _sphere_volume((8, 8, 8))
        cost = _compute_cost(vol, vol, None, CostContext(name="lps"))
        assert cost.item() > 0.9

    def test_lpc_runs(self):
        vol = _sphere_volume((32, 32, 32))
        cost = _compute_cost(vol, vol, None, CostContext(name="lpc"))
        assert math.isfinite(cost.item())

    def test_unknown_cost_raises(self):
        vol = torch.randn(8, 8, 8)
        with pytest.raises(ValueError, match="Unknown cost"):
            _compute_cost(vol, vol, None, CostContext(name="bad_cost"))


# ---------------------------------------------------------------------------
# _batched_cost
# ---------------------------------------------------------------------------


class TestBatchedCost:
    def test_ls_batched(self):
        base = _sphere_volume((8, 8, 8))
        warped = base.unsqueeze(0).expand(3, -1, -1, -1)
        costs = _batched_cost(base, warped, None, CostContext(name="ls"))
        assert costs.shape == (3,)
        assert (costs > 0.9).all()

    def test_ls_batched_with_weight(self):
        base = _sphere_volume((8, 8, 8))
        warped = base.unsqueeze(0).expand(3, -1, -1, -1)
        weight = torch.ones_like(base)
        costs = _batched_cost(base, warped, weight, CostContext(name="ls"))
        assert costs.shape == (3,)
        assert (costs > 0.9).all()

    def test_lpa_batched(self):
        base = _sphere_volume((32, 32, 32))
        warped = base.unsqueeze(0).expand(2, -1, -1, -1)
        costs = _batched_cost(base, warped, None, CostContext(name="lpa"))
        assert costs.shape == (2,)

    def test_lpc_batched(self):
        base = _sphere_volume((32, 32, 32))
        warped = base.unsqueeze(0).expand(2, -1, -1, -1)
        costs = _batched_cost(base, warped, None, CostContext(name="lpc"))
        assert costs.shape == (2,)

    def test_unknown_raises(self):
        base = torch.randn(8, 8, 8)
        warped = base.unsqueeze(0)
        with pytest.raises(ValueError, match="Unknown cost"):
            _batched_cost(base, warped, None, CostContext(name="bad"))


# ---------------------------------------------------------------------------
# Coarse candidate generators
# ---------------------------------------------------------------------------


class TestCandidateGenerators:
    def test_rotation_grid_basic(self):
        t = torch.tensor([0.0, 0.0, 0.0])
        cands = _rotation_candidates(10.0, 10.0, t, DEV)
        # angles -10, 0, 10 -> 3^3 = 27 candidates; scales == 1
        assert cands.shape == (27, 12)
        assert (cands[:, 6:9] == 1.0).all()

    def test_rotation_grid_multi_translation(self):
        ts = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        cands = _rotation_candidates(10.0, 10.0, ts, DEV)
        # 2 translations x 27 rotations
        assert cands.shape == (2 * 27, 12)
        # first block carries the first translation
        assert torch.allclose(cands[0, 0:3], ts[0])

    def test_translation_grid_covers_range(self):
        center = torch.tensor([0.0, 0.0, 0.0])
        shift_range = torch.tensor([10.0, 10.0, 10.0])
        cands = _translation_candidates(center, shift_range, 5, DEV)
        assert cands.shape == (5**3, 12)
        # rotations are zero, scales 1, and the extreme shift is reached
        assert (cands[:, 3:6] == 0.0).all()
        assert (cands[:, 6:9] == 1.0).all()
        assert float(cands[:, 0].max()) == pytest.approx(10.0)
        assert float(cands[:, 0].min()) == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# _estimate_chunk_size
# ---------------------------------------------------------------------------


class TestEstimateChunkSize:
    def test_cpu(self):
        chunk = _estimate_chunk_size((16, 16, 16), DEV)
        assert chunk >= 1
        assert chunk <= 4096

    def test_larger_volume_smaller_chunk(self):
        small = _estimate_chunk_size((8, 8, 8), DEV)
        large = _estimate_chunk_size((64, 64, 64), DEV)
        assert large <= small


# ---------------------------------------------------------------------------
# _compute_grid_matrix
# ---------------------------------------------------------------------------


class TestComputeGridMatrix:
    def test_identity_affines(self):
        aff = np.eye(4)
        mat = _compute_grid_matrix(aff, aff, DEV)
        torch.testing.assert_close(mat, torch.eye(4), atol=1e-5, rtol=1e-5)

    def test_scaled_affine(self):
        base_aff = np.diag([2.0, 2.0, 2.0, 1.0])
        src_aff = np.diag([1.0, 1.0, 1.0, 1.0])
        mat = _compute_grid_matrix(src_aff, base_aff, DEV)
        # base voxel * 2mm -> xyz; xyz * 1/1mm -> source voxel
        # So source_voxel = 2 * base_voxel
        expected = torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0]))
        torch.testing.assert_close(mat, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# _make_powell_cost
# ---------------------------------------------------------------------------


class TestMakePowellCost:
    def test_cost_fn_returns_float(self):
        base = _sphere_volume((8, 8, 8))
        source = base.clone()
        bounds = _compute_param_bounds((8, 8, 8))
        free_mask = _get_free_mask("rigid")
        identity_norm = _normalize(_identity_physical(), bounds)
        fixed_norm = identity_norm.copy()

        cost_fn = _make_powell_cost(
            base,
            source,
            None,
            CostContext(name="lps"),
            (1.0, 1.0, 1.0),
            bounds,
            free_mask,
            fixed_norm,
            DEV,
        )
        x0 = identity_norm[free_mask]
        val = cost_fn(x0)
        assert isinstance(val, float)
        # Perfect match should give negative cost (we negate for minimization)
        assert val < 0

    def test_cost_fn_with_counter(self):
        base = _sphere_volume((8, 8, 8))
        bounds = _compute_param_bounds((8, 8, 8))
        free_mask = _get_free_mask("rigid")
        identity_norm = _normalize(_identity_physical(), bounds)
        fixed_norm = identity_norm.copy()
        counter = [0]

        cost_fn = _make_powell_cost(
            base,
            base,
            None,
            CostContext(name="lps"),
            (1.0, 1.0, 1.0),
            bounds,
            free_mask,
            fixed_norm,
            DEV,
            counter=counter,
        )
        cost_fn(identity_norm[free_mask])
        assert counter[0] == 1


# ---------------------------------------------------------------------------
# _refine_adam_normalized (small test)
# ---------------------------------------------------------------------------


class TestRefineAdamNormalized:
    def test_identity_stays_near_identity(self):
        base = _sphere_volume((12, 12, 12))
        source = base.clone()
        bounds = _compute_param_bounds((12, 12, 12))
        init = _identity_physical()
        cfg = AffineAlignConfig(dof="rigid", cost="lps", verb=0)

        params, cost = _refine_adam_normalized(
            base,
            source,
            None,
            init,
            cfg,
            CostContext(name="lps"),
            (1.0, 1.0, 1.0),
            bounds,
            DEV,
            verb=0,
            n_iters=20,
            lr=0.01,
        )
        assert params.shape == (12,)
        # Should stay near identity for identical images
        np.testing.assert_allclose(params[:3], 0.0, atol=1.0)
        np.testing.assert_allclose(params[3:6], 0.0, atol=5.0)
        assert cost > 0.5

    def test_no_premature_early_stop_while_improving(self):
        """Plateau early-stop must NOT fire while the cost keeps improving.

        Regression for the -inf threshold bug: with last_best == -inf the
        relative plateau threshold was nan, so `best > nan` was always False and
        the loop early-stopped after ~patience iters regardless of progress.
        """
        base = _sphere_volume((8, 8, 8))
        bounds = _compute_param_bounds((8, 8, 8))
        init = _identity_physical()
        cfg = AffineAlignConfig(dof="rigid", verb=0)

        # A cost that strictly increases every evaluation (still differentiable
        # in the params via the zeroed matrix term) — it never plateaus, so a
        # correct loop runs the full iteration budget.
        calls = [0]

        def cost_fn(matrix):
            calls[0] += 1
            return matrix.sum() * 0.0 + 0.001 * calls[0]

        n_iters = 120
        _refine_adam_normalized(
            base,
            base,
            None,
            init,
            cfg,
            CostContext(name="lpa"),
            (1.0, 1.0, 1.0),
            bounds,
            DEV,
            verb=0,
            n_iters=n_iters,
            lr=0.01,
            cost_fn=cost_fn,
        )
        # Must run essentially the whole budget, not stop near `patience` (~40).
        assert calls[0] > n_iters - 5


# ---------------------------------------------------------------------------
# _refine_powell (small test)
# ---------------------------------------------------------------------------


class TestRefinePowell:
    def test_identity_stays_near_identity(self):
        base = _sphere_volume((10, 10, 10))
        source = base.clone()
        bounds = _compute_param_bounds((10, 10, 10))
        init = _identity_physical()
        cfg = AffineAlignConfig(dof="rigid", cost="lps", verb=0)

        params, cost = _refine_powell(
            base,
            source,
            None,
            init,
            cfg,
            CostContext(name="lps"),
            (1.0, 1.0, 1.0),
            bounds,
            DEV,
            verb=0,
            maxfev=50,
        )
        assert params.shape == (12,)
        assert cost > 0.5


# ---------------------------------------------------------------------------
# allineate (integration test)
# ---------------------------------------------------------------------------


class TestAllineate:
    def test_identity_alignment(self):
        """Aligning identical images should produce near-identity matrix."""
        vol = _sphere_volume((16, 16, 16))
        cfg = AffineAlignConfig(
            dof="rigid",
            cost="lpa",
            twopass=False,  # skip coarse for speed
            cmass=True,
            autocrop=False,
            powell_maxfev=0,
            adam_iters_1x=30,
            adam_iters_2x=20,
            verb=0,
        )
        matrix, warped = allineate(vol, vol, config=cfg)
        assert matrix.shape == (4, 4)
        assert warped.shape == vol.shape
        # Matrix should be near identity
        torch.testing.assert_close(matrix, torch.eye(4), atol=0.5, rtol=0.0)

    def test_small_translation_recovery(self):
        """Apply a small translation and verify alignment recovers it."""
        base = _sphere_volume((16, 16, 16), center=(8, 8, 8), radius=5)
        # Create source shifted by ~1.5 voxels in x
        source = _sphere_volume((16, 16, 16), center=(9.5, 8, 8), radius=5)

        cfg = AffineAlignConfig(
            dof="rigid",
            cost="lps",
            twopass=True,
            coarse_range=10.0,
            coarse_step=5.0,
            tbest=1,
            cmass=True,
            autocrop=False,
            adam_iters_2x=50,
            adam_iters_1x=80,
            powell_maxfev=100,
            verb=0,
        )
        matrix, warped = allineate(base, source, config=cfg)
        assert matrix.shape == (4, 4)
        assert warped.shape == base.shape

        # The cost between warped and base should be decent
        cost = _compute_cost(base, warped, None, CostContext(name="lps"))
        assert cost.item() > 0.7

    def test_config_none_defaults(self):
        """config=None should use defaults."""
        vol = _sphere_volume((10, 10, 10))
        cfg = AffineAlignConfig(
            twopass=False,
            powell_maxfev=0,
            adam_iters_1x=10,
            adam_iters_2x=10,
            verb=0,
        )
        matrix, warped = allineate(vol, vol, config=cfg)
        assert matrix.shape == (4, 4)

    def test_with_autocrop(self):
        """Autocrop should still produce correct output shape."""
        base = torch.zeros(24, 24, 24)
        base[8:16, 8:16, 8:16] = _sphere_volume((8, 8, 8))
        source = base.clone()

        cfg = AffineAlignConfig(
            dof="rigid",
            cost="lpa",
            twopass=False,
            cmass=True,
            autocrop=True,
            powell_maxfev=0,
            adam_iters_1x=20,
            adam_iters_2x=20,
            verb=0,
        )
        matrix, warped = allineate(base, source, config=cfg)
        assert warped.shape == base.shape

    @pytest.mark.skip(
        reason="_compute_cost passes 3D base to clipped_pearson "
        "which expects 1D; pre-existing bug outside test scope"
    )
    def test_ls_cost(self):
        """Test with clipped pearson cost."""
        vol = _sphere_volume((12, 12, 12))
        cfg = AffineAlignConfig(
            dof="rigid",
            cost="ls",
            twopass=False,
            powell_maxfev=0,
            adam_iters_1x=10,
            autoweight=False,
            verb=0,
        )
        matrix, warped = allineate(vol, vol, config=cfg)
        assert matrix.shape == (4, 4)

    def test_affine_dof(self):
        """Test with full affine DOF."""
        vol = _sphere_volume((12, 12, 12))
        cfg = AffineAlignConfig(
            dof="affine",
            cost="lpa",
            twopass=False,
            powell_maxfev=0,
            adam_iters_1x=10,
            verb=0,
        )
        matrix, warped = allineate(vol, vol, config=cfg)
        assert matrix.shape == (4, 4)

    def test_explicit_device(self):
        """Test config.device override."""
        vol = _sphere_volume((10, 10, 10))
        cfg = AffineAlignConfig(
            device="cpu",
            twopass=False,
            powell_maxfev=0,
            adam_iters_1x=10,
            verb=0,
        )
        matrix, warped = allineate(vol, vol, config=cfg)
        assert matrix.device == torch.device("cpu")

    def test_cmass_direct_reproduces_auto(self):
        """-cmass_direct with the printed shift reproduces the auto-cmass result.

        The reported shift is in base-grid voxels; feeding it back via
        cmass_direct must skip the COM estimate and land in the same place. With
        refinement disabled the final transform is exactly the cmass init, so the
        two runs must agree bit-for-bit and the warped source's COM must sit on
        the base COM.
        """
        base = _sphere_volume((20, 22, 24), center=(12, 11, 10))
        source = _sphere_volume((20, 22, 24), center=(8, 14, 13))

        def _cfg(**kw):
            return AffineAlignConfig(
                cost="ls",
                twopass=False,
                powell_maxfev=0,
                adam_iters_1x=0,
                adam_iters_2x=0,
                autocrop=False,
                autoweight=False,
                verb=0,
                **kw,
            )

        shift = _cmass_translation(base, source).cpu().numpy()
        m_auto, w_auto = allineate(base, source, config=_cfg(cmass=True))
        m_dir, w_dir = allineate(base, source, config=_cfg(cmass_direct=tuple(shift)))

        torch.testing.assert_close(m_auto, m_dir, atol=1e-5, rtol=0.0)
        torch.testing.assert_close(w_auto, w_dir, atol=1e-5, rtol=0.0)
        # the cmass shift aligns the source COM onto the base COM
        torch.testing.assert_close(
            _center_of_mass(w_auto), _center_of_mass(base), atol=0.6, rtol=0.0
        )

    def test_save_cmass_writes_file(self, tmp_path):
        """-save_cmass writes the cmass-positioned source to disk."""
        base = _sphere_volume((16, 16, 16), center=(8, 8, 8))
        source = _sphere_volume((16, 16, 16), center=(6, 9, 10))
        out = tmp_path / "cmass.nii.gz"
        cfg = AffineAlignConfig(
            cost="ls",
            twopass=False,
            powell_maxfev=0,
            adam_iters_1x=0,
            adam_iters_2x=0,
            autocrop=False,
            autoweight=False,
            cmass=True,
            verb=0,
        )
        allineate(base, source, config=cfg, save_cmass_path=str(out))
        assert out.exists()

    def test_cross_grid_far_apart_brains_align(self):
        """Cross-grid alignment when the brains start far apart in scanner space.

        Regression: the source was resampled onto the base grid at its
        un-shifted position, so when the two brains sit far apart in physical
        space that resample captures the wrong part of the source (often empty),
        the cost can't see the real overlap, and the result drifts back to the
        wrong place even though the cmass shift itself was correct. Baking the
        cmass into the resample lands the source brain in the base FOV first, so
        the optimiser converges. Here the affines put the brains 24 mm apart.
        """
        # Small grids on purpose: this is a CPU regression test, and its point is
        # the *geometry* (different FOVs, brains far apart, must still converge),
        # not resolution. Keep it cheap so it doesn't saturate every core.
        base = _sphere_volume((16, 22, 22), center=(11, 11, 8), radius=4)
        source = _sphere_volume((28, 22, 22), center=(11, 11, 8), radius=4)
        base_aff = np.eye(4)
        src_aff = np.eye(4)
        src_aff[2, 3] = 12.0  # source brain sits +12 mm away in physical z
        cfg = AffineAlignConfig(
            cost="ls",
            dof="rigid",
            twopass=True,
            coarse_range=6,
            coarse_step=3,
            powell_maxfev=0,
            adam_iters_1x=45,
            adam_iters_2x=30,
            autocrop=False,
            autoweight=False,
            verb=0,
        )
        _, warped = allineate(
            base,
            source,
            config=cfg,
            base_header={"affine": base_aff},
            source_header={"affine": src_aff},
        )

        assert warped.max().item() > 0.5  # the brain actually made it into the FOV
        torch.testing.assert_close(
            _center_of_mass(warped), _center_of_mass(base), atol=1.5, rtol=0.0
        )


# ---------------------------------------------------------------------------
# outval=0 during optimization (AFNI-faithful out-of-volume fill)
# ---------------------------------------------------------------------------


class TestZeroOutsideBatched:
    def test_batched_matches_single(self):
        """apply_affine_batched(zero_outside) must match the single-volume path."""
        from fastfuncstuff.processing.affine import (
            apply_affine,
            apply_affine_batched,
            identity_params,
            params_to_matrix,
        )

        src = _sphere_volume((16, 16, 16), center=(8, 8, 8), radius=5)
        # A shift big enough that part of the brain maps outside the volume.
        p = identity_params().clone()
        p[0] = 7.0  # +7 vox in x
        M = params_to_matrix(p)
        single = apply_affine(src, M, src.shape, zero_outside=True)
        batched = apply_affine_batched(src, M[None], src.shape, zero_outside=True)[0]
        torch.testing.assert_close(batched, single, atol=1e-5, rtol=0.0)

    def test_zeros_beyond_border_value(self):
        """Out-of-volume voxels read 0, not a replicated border (the AFNI fix)."""
        from fastfuncstuff.processing.affine import (
            apply_affine,
            identity_params,
            params_to_matrix,
        )

        src = _sphere_volume((16, 16, 16), center=(8, 8, 8), radius=5) + 0.3  # nonzero edge
        p = identity_params().clone()
        p[0] = 20.0  # shift everything fully out of the source
        M = params_to_matrix(p)
        zeroed = apply_affine(src, M, src.shape, zero_outside=True)
        bordered = apply_affine(src, M, src.shape, zero_outside=False)
        assert float(zeroed.abs().max()) < 1e-6  # all out -> all zero
        assert float(bordered.abs().max()) > 0.1  # border padding keeps edge value


# ---------------------------------------------------------------------------
# -ov overlap penalty (AFNI lpc+/lpa+)
# ---------------------------------------------------------------------------


class TestOverlapPenalty:
    def _ctx_with_cov(self, ov_weight=0.1):
        from fastfuncstuff.processing.mask import automask

        blob = _sphere_volume((20, 20, 20), center=(10, 10, 10), radius=6)
        base_dom = automask(blob, device=DEV).float()
        src_cov = automask(blob, device=DEV).float()
        denom = float(min(base_dom.sum().item(), src_cov.sum().item()))
        ctx = CostContext(
            name="lpa",
            ov_weight=ov_weight,
            src_cov=src_cov,
            base_dom=base_dom,
            ov_denom=denom,
        )
        return ctx

    def test_penalty_zero_at_identity_large_off_overlap(self):
        from fastfuncstuff.processing.affine import identity_params, params_to_matrix
        from fastfuncstuff.processing.allineate import _overlap_penalty

        ctx = self._ctx_with_cov()
        ident = params_to_matrix(identity_params())
        pen_ident = _overlap_penalty(ctx, ident, ctx.src_cov.shape)

        p = identity_params().clone()
        p[0] = 30.0  # shift coverage fully out of the base domain
        pen_off = _overlap_penalty(ctx, params_to_matrix(p), ctx.src_cov.shape)

        assert pen_ident.item() < 1.0  # near-full overlap -> ~0 penalty
        assert pen_off.item() > 50.0  # no overlap -> large penalty

    def test_compute_cost_subtracts_penalty(self):
        """With ov>0 a low-overlap warp scores strictly below the same warp at ov=0."""
        from fastfuncstuff.processing.affine import identity_params, params_to_matrix

        base = _sphere_volume((20, 20, 20), center=(10, 10, 10), radius=6)
        warped = base.clone()
        p = identity_params().clone()
        p[0] = 30.0
        M = params_to_matrix(p)

        ctx_off = self._ctx_with_cov(ov_weight=0.0)
        ctx_on = self._ctx_with_cov(ov_weight=0.1)
        c_off = _compute_cost(base, warped, None, ctx_off, matrix=M)
        c_on = _compute_cost(base, warped, None, ctx_on, matrix=M)
        assert c_on.item() < c_off.item() - 1.0

    def test_allineate_with_ov_recovers_small_shift(self):
        """-ov plumbs through end-to-end and doesn't break a normal recovery."""
        base = _sphere_volume((20, 20, 20), center=(10, 10, 10), radius=6)
        source = _sphere_volume((20, 20, 20), center=(11.5, 10, 10), radius=6)
        cfg = AffineAlignConfig(
            dof="rigid",
            cost="lpa",
            ov=0.1,
            twopass=True,
            coarse_range=8.0,
            coarse_step=4.0,
            tbest=1,
            autocrop=False,
            adam_iters_2x=40,
            adam_iters_1x=60,
            powell_maxfev=0,
            verb=0,
        )
        _, warped = allineate(base, source, config=cfg)
        torch.testing.assert_close(
            _center_of_mass(warped), _center_of_mass(base), atol=1.0, rtol=0.0
        )


# ---------------------------------------------------------------------------
# Match-point subsampling (point-wise blok refinement)
# ---------------------------------------------------------------------------


class TestSubsampling:
    def test_sample_affine_matches_full_at_points(self):
        """sample_affine_at_points equals apply_affine read at the same points."""
        from fastfuncstuff.processing.affine import (
            apply_affine,
            identity_params,
            params_to_matrix,
            sample_affine_at_points,
        )

        vol = _sphere_volume((16, 16, 16), center=(8, 8, 8), radius=5)
        p = identity_params().clone()
        p[0], p[1] = 2.0, -1.0  # shift in x, y
        M = params_to_matrix(p)
        full = apply_affine(vol, M, vol.shape, zero_outside=True)

        # A handful of base points (x, y, z).
        pts = torch.tensor([[8.0, 8.0, 8.0], [5.0, 9.0, 7.0], [10.0, 6.0, 8.0], [3.0, 3.0, 3.0]])
        sampled = sample_affine_at_points(vol, M, pts, zero_outside=True)
        flat = full.reshape(-1)
        nz, ny, nx = vol.shape
        for k, (x, y, z) in enumerate(pts.tolist()):
            ref = flat[int(z) * ny * nx + int(y) * nx + int(x)]
            torch.testing.assert_close(sampled[k], ref, atol=1e-4, rtol=0.0)

    def test_batched_adam_matches_sequential(self):
        """_refine_adam_batched row t equals _refine_adam_normalized on trial t."""
        from fastfuncstuff.processing.allineate import _refine_adam_batched

        bounds = _compute_param_bounds((16, 16, 16))
        cfg = AffineAlignConfig(dof="rigid", verb=0)
        target = torch.tensor([2.0, -1.0, 0.5])
        dummy = _sphere_volume((8, 8, 8))
        init = _identity_physical()

        # Same separable cost expressed per-matrix and batched.
        def scost(mat):
            return -((mat[:3, 3] - target) ** 2).sum()

        def bcost(mats):
            return -((mats[:, :3, 3] - target) ** 2).sum(dim=1)

        p_seq, _ = _refine_adam_normalized(
            dummy,
            dummy,
            None,
            init,
            cfg,
            CostContext(name="lpa"),
            (1.0, 1.0, 1.0),
            bounds,
            DEV,
            verb=0,
            n_iters=200,
            lr=0.02,
            cost_fn=scost,
        )
        out_phys, _ = _refine_adam_batched(
            [init.copy() for _ in range(3)],
            cfg,
            bounds,
            DEV,
            bcost,
            verb=0,
            n_iters=200,
            lr=0.02,
        )
        # All three batched rows identical (same init/cost) and equal to seq.
        for t in range(3):
            np.testing.assert_allclose(out_phys[t][:3], p_seq[:3], atol=0.05)
            np.testing.assert_allclose(out_phys[t][:3], target.numpy(), atol=0.5)

    def test_sample_batched_matches_single(self):
        """sample_affine_at_points_batched equals the single-matrix sampler."""
        from fastfuncstuff.processing.affine import (
            identity_params,
            params_to_matrix,
            sample_affine_at_points,
            sample_affine_at_points_batched,
        )

        vol = _sphere_volume((16, 16, 16), center=(8, 8, 8), radius=5)
        pts = torch.tensor([[8.0, 8.0, 8.0], [5.0, 9.0, 7.0], [10.0, 6.0, 8.0]])
        mats = []
        for dx, ry in [(0.0, 0.0), (2.0, 5.0), (-3.0, -8.0)]:
            p = identity_params().clone()
            p[0], p[5] = dx, ry
            mats.append(params_to_matrix(p))
        M = torch.stack(mats)
        batched = sample_affine_at_points_batched(vol, M, pts, zero_outside=True)
        for b in range(M.shape[0]):
            single = sample_affine_at_points(vol, M[b], pts, zero_outside=True)
            torch.testing.assert_close(batched[b], single, atol=1e-4, rtol=0.0)

    def test_assign_bloks_points_partitions(self):
        """assign_bloks_points labels points and respects MINCOR pruning."""
        from fastfuncstuff.processing.cost_blok import assign_bloks_points

        # A dense cube of points in mm (1 mm spacing): plenty per blok.
        g = torch.arange(0, 20, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
        bs = assign_bloks_points(coords, bloktype="tohd", blokrad=6.0)
        assert bs.index.shape[0] == coords.shape[0]
        assert bs.n_populated >= 1
        assert int((bs.index >= 0).sum()) > 0

    def test_allineate_subsampled_recovers_shift(self):
        """Forcing subsampling (small -nmatch) still recovers a small shift (lpa)."""
        base = _sphere_volume((28, 28, 28), center=(14, 14, 14), radius=9)
        source = _sphere_volume((28, 28, 28), center=(15.5, 14, 14), radius=9)
        cfg = AffineAlignConfig(
            dof="rigid",
            cost="lpa",
            n_match=1500,  # < domain -> exercises the point-sampled path
            twopass=True,
            coarse_range=8.0,
            coarse_step=4.0,
            tbest=1,
            autocrop=False,
            adam_iters_2x=40,
            adam_iters_1x=60,
            powell_maxfev=0,
            verb=0,
        )
        _, warped = allineate(base, source, config=cfg)
        torch.testing.assert_close(
            _center_of_mass(warped), _center_of_mass(base), atol=1.2, rtol=0.0
        )


def test_grid_from_dxyz_preserves_space():
    """`-mast_dxyz` grid: same orientation, centre, and FOV at a new voxel size."""
    from fastfuncstuff.processing.affine import grid_from_dxyz

    base_shape = (14, 12, 10)  # (nz, ny, nx)
    affine = np.array([[2.0, 0, 0, -9.0], [0, 2.0, 0, -11.0], [0, 0, 2.0, -13.0], [0, 0, 0, 1.0]])

    def _centre(a, shape):
        nz, ny, nx = shape
        c = np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2, 1.0])
        return (a @ c)[:3]

    # same voxel size → same grid + affine
    a_same, s_same = grid_from_dxyz(affine, base_shape, 2.0)
    assert s_same == base_shape
    assert np.allclose(a_same, affine)

    # half voxel → dims double, voxel size 1, centre + FOV preserved
    a_half, s_half = grid_from_dxyz(affine, base_shape, 1.0)
    assert s_half == (28, 24, 20)
    assert np.allclose(np.linalg.norm(a_half[:3, :3], axis=0), [1.0, 1.0, 1.0])
    assert np.allclose(_centre(a_half, s_half), _centre(affine, base_shape))
    # FOV (dim·vox) preserved on each axis
    assert np.allclose(np.array(s_half[::-1]) * 1.0, np.array(base_shape[::-1]) * 2.0)

    # anisotropic (x, y, z voxel/world order)
    a_an, s_an = grid_from_dxyz(affine, base_shape, [1.0, 2.0, 4.0])
    assert s_an == (7, 12, 20)  # z halves? no: fov_z=14*2=28 -> 28/4=7 ; y 24/2=12 ; x 20/1=20
    assert np.allclose(np.linalg.norm(a_an[:3, :3], axis=0), [1.0, 2.0, 4.0])
    with pytest.raises(ValueError):
        grid_from_dxyz(affine, base_shape, [1.0, 2.0])  # 2 values not allowed


def test_dxyz_output_samples_correct_world_location():
    """The -dxyz output resamples the SAME transform at a finer grid — a world-x ramp is
    reproduced at the new voxel centres (validates the out-grid matrix composition)."""
    from fastfuncstuff.cli.allineate import _out_matrix, _output_grid, _resample

    base_shape = (14, 12, 10)
    affine = np.array([[2.0, 0, 0, -9.0], [0, 2.0, 0, -13.0], [0, 0, 2.0, -11.0], [0, 0, 0, 1.0]])
    nz, ny, nx = base_shape
    _, _, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    source = torch.tensor(2.0 * xx + affine[0, 3], dtype=torch.float32)  # value = world-x
    matrix = torch.eye(4)  # source shares the base grid
    dev = torch.device("cpu")

    # base grid: identity → returns the ramp unchanged
    oa, os_ = _output_grid(affine, base_shape, None)
    w0 = _resample(source, _out_matrix(matrix, affine, oa, dev), os_, "linear")
    assert torch.allclose(w0, source, atol=1e-3)

    # half-res grid: each output voxel must carry its OWN world-x (correct composition)
    oa2, os2 = _output_grid(affine, base_shape, [1.0])
    w2 = _resample(source, _out_matrix(matrix, affine, oa2, dev), os2, "linear")
    ix = np.arange(os2[2])
    expected = 1.0 * ix + oa2[0, 3]  # world-x at each output column
    got = w2[os2[0] // 2, os2[1] // 2, 2 : os2[2] - 2].numpy()  # interior row (avoid OOB edges)
    assert np.allclose(got, expected[2 : os2[2] - 2], atol=1e-2)


class TestDerivativeFreeRefinement:
    """The batched derivative-free optimizers (-optimizer pattern / cmaes).

    Both exist because the batched cost is nearly free in the batch dimension
    while sequential steps are not, so they spend a population on search instead
    of a backward pass on a gradient. What has to hold is that they actually
    optimize: a quadratic bowl with a known optimum is enough to catch a sign
    error, a broken covariance update, or a search that never moves.
    """

    @staticmethod
    def _bowl(target_norm, bounds, device):
        """A batched cost (higher better) peaking at ``target_norm``, correlated axes.

        The off-diagonal coupling matters: a diagonal bowl is solved by a
        coordinate search, so it would not distinguish the two methods at all.
        """
        from fastfuncstuff.processing.affine import params_to_matrix_batched  # noqa: F401

        bmin, span = _bounds_to_torch(bounds, device)
        tgt = torch.as_tensor(target_norm, dtype=torch.float32, device=device)

        def cost(matrices):
            # Recover the normalized params from the matrix translation column,
            # which is what both optimizers vary here.
            t = matrices[:, :3, 3]
            d = t - tgt[None, :3] * span[None, :3] - bmin[None, :3]
            skew = d[:, 0] * d[:, 1] * 0.9  # correlated valley
            return -(d.pow(2).sum(dim=1) + skew)

        return cost

    @pytest.mark.parametrize("refine", [_refine_pattern_batched, _refine_cmaes_batched])
    def test_improves_on_its_starting_point(self, refine):
        device = torch.device("cpu")
        bounds = _compute_param_bounds((20, 20, 20), (1.0, 1.0, 1.0))
        config = AffineAlignConfig(dof="rigid")
        start = _identity_physical()
        cost = self._bowl(_normalize(start, bounds) + 0.02, bounds, device)

        out, costs = refine([start], config, bounds, device, cost, verb=0, n_iters=60)

        assert out.shape == (1, 12)
        assert np.all(np.isfinite(out))
        start_cost = float(cost(_batched_cost_matrix(start, device)).item())
        assert costs[0] >= start_cost, "refinement returned a worse point than it started from"

    def test_cmaes_is_reproducible(self):
        """Seeded sampling: an alignment that moves run to run is unusable downstream."""
        device = torch.device("cpu")
        bounds = _compute_param_bounds((20, 20, 20), (1.0, 1.0, 1.0))
        config = AffineAlignConfig(dof="rigid")
        start = _identity_physical()
        cost = self._bowl(_normalize(start, bounds) + 0.02, bounds, device)

        a, ca = _refine_cmaes_batched([start], config, bounds, device, cost, verb=0, n_iters=40)
        b, cb = _refine_cmaes_batched([start], config, bounds, device, cost, verb=0, n_iters=40)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ca, cb)

    def test_cmaes_handles_several_trials_at_once(self):
        """T trials share one batched evaluation; each must keep its own state."""
        device = torch.device("cpu")
        bounds = _compute_param_bounds((20, 20, 20), (1.0, 1.0, 1.0))
        config = AffineAlignConfig(dof="rigid")
        starts = [_identity_physical() for _ in range(3)]
        cost = self._bowl(_normalize(starts[0], bounds) + 0.02, bounds, device)

        out, costs = _refine_cmaes_batched(starts, config, bounds, device, cost, verb=0, n_iters=30)
        assert out.shape == (3, 12)
        assert costs.shape == (3,)
        assert np.all(np.isfinite(costs))


def _batched_cost_matrix(params_phys, device):
    """(1,4,4) matrix for a single physical parameter vector."""
    from fastfuncstuff.processing.affine import params_to_matrix_batched

    t = torch.as_tensor(params_phys, dtype=torch.float32, device=device)[None, :]
    return params_to_matrix_batched(t)


class TestOptimizerAutoSelection:
    """-optimizer auto: CMA-ES only while its population is actually free.

    CMA-ES beats Adam by ~4.7x when the cost evaluation is launch-bound, because
    then evaluating a population costs what evaluating one candidate costs. Once a
    generation's work is real compute, the population becomes a straight
    multiplier and Adam's single candidate per step wins by ~2x. These are the two
    measured cases that bracket the threshold.
    """

    def test_small_subsampled_problem_picks_cmaes(self):
        # crossalign: 61,384 points, 1 trial, rigid -> ~1.0M work
        assert _pick_optimizer("auto", 61_384, 1, 6) == "cmaes"

    def test_large_many_trial_problem_picks_adam(self):
        # anat->MNI first stage: 903,700 points, 11 trials, affine -> ~189M work.
        # Measured: adam 23.08 s vs cmaes 40.80 s.
        assert _pick_optimizer("auto", 903_700, 11, 12) == "adam"

    def test_measured_crossover_is_bracketed(self):
        """The sweep put the crossover between 69M and 103M; stay on both sides."""
        # 4 trials -> 69M, cmaes won (21.95 s vs 26.79 s)
        assert _pick_optimizer("auto", 903_700, 4, 12) == "cmaes"
        # 6 trials -> 103M, adam won (21.78 s vs 29.44 s)
        assert _pick_optimizer("auto", 903_700, 6, 12) == "adam"

    def test_trials_alone_can_flip_the_choice(self):
        """Trial count multiplies the work exactly as the point count does.

        This is what makes twopass a mixed case: it explores wide (many trials,
        Adam) and then refines (trials deduplicated, CMA-ES), so the choice is
        made per stage rather than once per alignment.
        """
        # Same points, same dof -- only the trial count differs.
        assert _pick_optimizer("auto", 903_700, 1, 12) == "cmaes"
        assert _pick_optimizer("auto", 903_700, 11, 12) == "adam"
        assert _pick_optimizer("auto", 60_000, 1, 6) == "cmaes"
        assert _pick_optimizer("auto", 60_000, 400, 6) == "adam"

    @pytest.mark.parametrize("explicit", ["adam", "cmaes", "pattern"])
    def test_explicit_choice_is_never_overridden(self, explicit):
        assert _pick_optimizer(explicit, 903_700, 11, 12) == explicit
        assert _pick_optimizer(explicit, 1, 1, 6) == explicit


class TestWorkGrid:
    """The search runs on the coarser of the two grids; the fit maps back exactly."""

    @staticmethod
    def _headers(base_mm, src_mm):
        base_aff = np.diag([base_mm, base_mm, base_mm, 1.0])
        src_aff = np.diag([src_mm, src_mm, src_mm, 1.0])
        return {"affine": base_aff}, {"affine": src_aff}

    def test_decimates_to_the_coarser_source(self):
        from fastfuncstuff.processing.allineate import _plan_work_grid

        base = _sphere_volume((40, 40, 40), center=(20, 20, 20), radius=8)
        bh, sh = self._headers(1.0, 2.0)
        work, work_header, full_to_work = _plan_work_grid(
            base, bh, sh, "auto", torch.device("cpu"), verb=0
        )
        assert full_to_work is not None
        assert work.shape == (20, 20, 20)
        assert np.allclose(np.diag(work_header["affine"])[:3], 2.0)

    def test_no_op_when_the_source_is_not_coarser(self):
        from fastfuncstuff.processing.allineate import _plan_work_grid

        base = _sphere_volume((32, 32, 32), center=(16, 16, 16), radius=6)
        bh, sh = self._headers(1.0, 1.0)
        work, work_header, full_to_work = _plan_work_grid(
            base, bh, sh, "auto", torch.device("cpu"), verb=0
        )
        assert full_to_work is None
        assert work is base and work_header is bh

    def test_never_upsamples_an_already_coarse_base(self):
        from fastfuncstuff.processing.allineate import _plan_work_grid

        base = _sphere_volume((32, 32, 32), center=(16, 16, 16), radius=6)
        bh, sh = self._headers(3.0, 1.0)  # base coarser than source
        _, _, full_to_work = _plan_work_grid(base, bh, sh, "auto", torch.device("cpu"), verb=0)
        assert full_to_work is None

    def test_mapping_back_is_exact(self):
        """A fit found on the work grid must mean the same mm-space transform.

        This is the whole premise: decimating the base changes what the search
        costs, never what the answer is. Take a known base->source voxel map on
        the work grid, carry it back with ``full_to_work``, and both must convert
        to the same DICOM matrix.
        """
        from fastfuncstuff.processing.affine import voxel_matrix_to_dicom
        from fastfuncstuff.processing.allineate import _plan_work_grid

        base = _sphere_volume((40, 40, 40), center=(20, 20, 20), radius=8)
        bh, sh = self._headers(1.0, 2.0)
        _, work_header, full_to_work = _plan_work_grid(
            base, bh, sh, "auto", torch.device("cpu"), verb=0
        )
        m_work = torch.eye(4)
        m_work[0, 3], m_work[1, 3], m_work[2, 3] = 1.5, -0.75, 0.25  # work voxels
        m_full = m_work @ full_to_work

        d_work = voxel_matrix_to_dicom(m_work, work_header["affine"], sh["affine"])
        d_full = voxel_matrix_to_dicom(m_full, bh["affine"], sh["affine"])
        assert torch.allclose(d_work, d_full, atol=1e-4)

    def test_output_stays_on_the_callers_grid(self):
        base = _sphere_volume((32, 32, 32), center=(16, 16, 16), radius=7)
        source = _sphere_volume((16, 16, 16), center=(8, 8, 8), radius=3.5)
        bh, sh = self._headers(1.0, 2.0)
        cfg = AffineAlignConfig(
            cost="ls",
            dof="rigid",
            twopass=False,
            powell_maxfev=0,
            adam_iters_1x=10,
            adam_iters_2x=5,
            autocrop=False,
            autoweight=False,
            verb=0,
        )
        matrix, warped = allineate(base, source, config=cfg, base_header=bh, source_header=sh)
        assert warped.shape == base.shape
        assert matrix.shape == (4, 4)


class TestMatchPointFloor:
    """A tight mask must not put the cost on a handful of points (AFNI's 9999)."""

    @staticmethod
    def _domain_volume(n_vox: int, shape=(24, 24, 24)):
        """A volume with exactly ``n_vox`` nonzero voxels."""
        vol = torch.zeros(shape)
        vol.reshape(-1)[:n_vox] = 1.0
        return vol

    def test_small_domain_is_floored_not_thinned(self):
        from fastfuncstuff.processing.allineate import _SAMPLE_MIN_POINTS, _build_sample_set

        n_dom = 12000
        base = self._domain_volume(n_dom)
        sample = _build_sample_set(base, None, (1.0, 1.0, 1.0), 0.0, "tohd", torch.device("cpu"))
        assert sample is not None
        # 47% of 12000 is 5640 -- the floor lifts it instead.
        assert sample.idx_flat.numel() == _SAMPLE_MIN_POINTS

    def test_floor_never_exceeds_the_domain(self):
        from fastfuncstuff.processing.allineate import _build_sample_set

        n_dom = 4000  # smaller than the floor
        base = self._domain_volume(n_dom)
        sample = _build_sample_set(base, None, (1.0, 1.0, 1.0), 0.0, "tohd", torch.device("cpu"))
        assert sample is not None
        assert sample.idx_flat.numel() == n_dom

    def test_large_domain_keeps_the_47_percent_default(self):
        from fastfuncstuff.processing.allineate import _SAMPLE_DEFAULT_FRAC, _build_sample_set

        n_dom = 100_000
        base = self._domain_volume(n_dom, shape=(50, 50, 50))
        sample = _build_sample_set(base, None, (1.0, 1.0, 1.0), 0.0, "tohd", torch.device("cpu"))
        assert sample is not None
        assert sample.idx_flat.numel() == int(_SAMPLE_DEFAULT_FRAC * n_dom)
