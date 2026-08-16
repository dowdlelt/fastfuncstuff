"""Tests for fastfuncstuff.processing.warp module.

Covers helper/utility functions, dataclasses, and algorithmic components
that can be tested in isolation without NIfTI files.
"""

import math

import pytest
import torch

from fastfuncstuff.processing.warp import (
    PatchSpec,
    QwarpConfig,
    WarpState,
    _autobox,
    _checkerboard_phases,
    _compute_hfactor,
    _compute_padding,
    _dedup_last_wins,
    _filter_patches,
    _generate_patch_grid,
    _get_basis_config,
    _maybe_compile,
    _pad_volume,
)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# QwarpConfig dataclass
# ---------------------------------------------------------------------------


class TestQwarpConfig:
    def test_defaults(self):
        cfg = QwarpConfig()
        assert cfg.minpatch == 25
        assert cfg.blur_base == 0.0
        assert cfg.blur_source == 0.0
        assert cfg.use_quintic is False
        assert cfg.use_lite is True
        assert cfg.cost_method == "pearclp"
        assert cfg.penalty_factor == 0.033  # matches AFNI Hpen_fbase
        assert cfg.penalty_first_level == 3  # matches AFNI Hpen_first_lev
        assert cfg.shrink == pytest.approx(0.749999)
        assert cfg.max_level == 666
        assert cfg.start_level == 0
        assert cfg.warp_flags == 0
        assert cfg.axis_weights == (1.0, 1.0, 1.0)
        assert cfg.verb == 1
        assert cfg.batch_optimizer_lr == pytest.approx(0.008)
        assert cfg.batch_optimizer_iters == 60
        assert cfg.hfactor_q == 0.5
        assert cfg.maxdisp == 0.0
        assert cfg.lpa_sigma == 4.0
        assert cfg.lpa_kernel == "gauss"
        assert cfg.level_stop_tol == 0.0
        assert cfg.compile is False
        assert cfg.pyramid_factor == 1
        assert cfg.reject_worse_levels is False

    def test_custom_values(self):
        cfg = QwarpConfig(
            minpatch=9,
            use_quintic=True,
            cost_method="lpa",
            warp_flags=3,
            axis_weights=(0.5, 1.0, 0.5),
        )
        assert cfg.minpatch == 9
        assert cfg.use_quintic is True
        assert cfg.cost_method == "lpa"
        assert cfg.warp_flags == 3
        assert cfg.axis_weights == (0.5, 1.0, 0.5)


# ---------------------------------------------------------------------------
# WarpState dataclass
# ---------------------------------------------------------------------------


class TestWarpState:
    def test_defaults(self):
        ws = WarpState()
        assert ws.nx == 0
        assert ws.ny == 0
        assert ws.nz == 0
        assert ws.cost == pytest.approx(666.666)
        assert ws.patches_done == 0
        assert ws.patches_skipped == 0
        assert ws.xd.numel() == 0
        assert ws.yd.numel() == 0
        assert ws.zd.numel() == 0

    def test_independent_defaults(self):
        """Each instance should get its own tensor (not shared mutable default)."""
        ws1 = WarpState()
        ws2 = WarpState()
        assert ws1.xd is not ws2.xd


# ---------------------------------------------------------------------------
# PatchSpec dataclass
# ---------------------------------------------------------------------------


class TestPatchSpec:
    def test_creation(self):
        p = PatchSpec(ibot=0, itop=10, jbot=5, jtop=15, kbot=2, ktop=8, gi=1, gj=0, gk=1)
        assert p.ibot == 0
        assert p.itop == 10
        assert p.gk == 1


# ---------------------------------------------------------------------------
# _compute_padding
# ---------------------------------------------------------------------------


class TestComputePadding:
    def test_small_volume(self):
        """Small volumes should get minimum padding of 3."""
        px, py, pz = _compute_padding(10, 10, 10)
        assert px >= 3
        assert py >= 3
        assert pz >= 3

    def test_formula(self):
        """Check the AFNI formula: ceil(0.1234 * dim) + 1, min 3."""
        nx, ny, nz = 64, 64, 32
        px, py, pz = _compute_padding(nx, ny, nz)
        assert px == max(3, math.ceil(0.1234 * nx) + 1)
        assert py == max(3, math.ceil(0.1234 * ny) + 1)
        assert pz == max(3, math.ceil(0.1234 * nz) + 1)

    def test_asymmetric(self):
        """Different dimensions should give different padding."""
        px, py, pz = _compute_padding(100, 50, 20)
        assert px > py > pz

    def test_very_small(self):
        """Tiny dimensions should still get padding of 3."""
        px, py, pz = _compute_padding(1, 1, 1)
        assert px == 3
        assert py == 3
        assert pz == 3


# ---------------------------------------------------------------------------
# _pad_volume / _unpad_volume
# ---------------------------------------------------------------------------


class TestPadVolume:
    def test_shape(self):
        vol = torch.randn(8, 10, 12)
        padded = _pad_volume(vol, 2, 3, 4)
        assert padded.shape == (8 + 2 * 4, 10 + 2 * 3, 12 + 2 * 2)

    def test_zero_padding_values(self):
        """Padded region should be zero."""
        vol = torch.ones(4, 4, 4)
        padded = _pad_volume(vol, 2, 2, 2)
        # Corners should be zero
        assert padded[0, 0, 0].item() == 0.0
        assert padded[-1, -1, -1].item() == 0.0

    def test_interior_preserved(self):
        """Interior should match original volume."""
        vol = torch.randn(5, 6, 7)
        px, py, pz = 3, 2, 4
        padded = _pad_volume(vol, px, py, pz)
        interior = padded[pz : pz + 5, py : py + 6, px : px + 7]
        assert torch.allclose(interior, vol)

    def test_zero_padding(self):
        """Zero pad amounts should return identical tensor."""
        vol = torch.randn(4, 5, 6)
        padded = _pad_volume(vol, 0, 0, 0)
        assert padded.shape == vol.shape
        assert torch.allclose(padded, vol)

    def test_roundtrip(self):
        """Padding then cropping should recover original."""
        vol = torch.randn(8, 10, 12)
        px, py, pz = 3, 4, 5
        padded = _pad_volume(vol, px, py, pz)
        recovered = padded[pz : pz + 8, py : py + 10, px : px + 12]
        assert torch.allclose(recovered, vol)


# ---------------------------------------------------------------------------
# _generate_patch_grid
# ---------------------------------------------------------------------------


class TestGeneratePatchGrid:
    def test_single_patch(self):
        """When window covers the entire range, should produce one patch."""
        patches = _generate_patch_grid(
            ibbb=0,
            ittt=9,
            jbbb=0,
            jttt=9,
            kbbb=0,
            kttt=9,
            xwid=10,
            ywid=10,
            zwid=10,
            xdel=5,
            ydel=5,
            zdel=5,
        )
        assert len(patches) == 1
        p = patches[0]
        assert p.ibot == 0
        assert p.itop == 9
        assert p.jbot == 0
        assert p.jtop == 9
        assert p.kbot == 0
        assert p.ktop == 9

    def test_multiple_patches(self):
        """Multiple patches with 50% overlap."""
        patches = _generate_patch_grid(
            ibbb=0,
            ittt=19,
            jbbb=0,
            jttt=19,
            kbbb=0,
            kttt=19,
            xwid=11,
            ywid=11,
            zwid=11,
            xdel=5,
            ydel=5,
            zdel=5,
        )
        assert len(patches) > 1
        # All patches should be within bounds
        for p in patches:
            assert p.ibot >= 0
            assert p.itop <= 19
            assert p.jbot >= 0
            assert p.jtop <= 19
            assert p.kbot >= 0
            assert p.ktop <= 19

    def test_patch_width(self):
        """Each patch should have the specified width."""
        patches = _generate_patch_grid(
            ibbb=0,
            ittt=29,
            jbbb=0,
            jttt=29,
            kbbb=0,
            kttt=29,
            xwid=9,
            ywid=9,
            zwid=9,
            xdel=4,
            ydel=4,
            zdel=4,
        )
        for p in patches:
            assert p.itop - p.ibot + 1 == 9
            assert p.jtop - p.jbot + 1 == 9
            assert p.ktop - p.kbot + 1 == 9

    def test_grid_indices_increment(self):
        """Grid indices (gi, gj, gk) should form a 3D grid."""
        patches = _generate_patch_grid(
            ibbb=0,
            ittt=29,
            jbbb=0,
            jttt=29,
            kbbb=0,
            kttt=29,
            xwid=9,
            ywid=9,
            zwid=9,
            xdel=4,
            ydel=4,
            zdel=4,
        )
        gi_vals = sorted(set(p.gi for p in patches))
        gj_vals = sorted(set(p.gj for p in patches))
        gk_vals = sorted(set(p.gk for p in patches))
        # Grid indices start at 0 and are contiguous
        assert gi_vals == list(range(len(gi_vals)))
        assert gj_vals == list(range(len(gj_vals)))
        assert gk_vals == list(range(len(gk_vals)))


# ---------------------------------------------------------------------------
# _checkerboard_phases
# ---------------------------------------------------------------------------


class TestCheckerboardPhases:
    def test_eight_phases(self):
        """Should produce exactly 8 phases."""
        patches = _generate_patch_grid(
            ibbb=0,
            ittt=29,
            jbbb=0,
            jttt=29,
            kbbb=0,
            kttt=29,
            xwid=9,
            ywid=9,
            zwid=9,
            xdel=4,
            ydel=4,
            zdel=4,
        )
        phases = _checkerboard_phases(patches)
        assert len(phases) == 8

    def test_all_patches_assigned(self):
        """Every patch should appear in exactly one phase."""
        patches = _generate_patch_grid(
            ibbb=0,
            ittt=29,
            jbbb=0,
            jttt=29,
            kbbb=0,
            kttt=29,
            xwid=9,
            ywid=9,
            zwid=9,
            xdel=4,
            ydel=4,
            zdel=4,
        )
        phases = _checkerboard_phases(patches)
        total = sum(len(ph) for ph in phases)
        assert total == len(patches)

    def test_phase_assignment_correct(self):
        """Check that checkerboard index formula is correct."""
        patches = [
            PatchSpec(0, 8, 0, 8, 0, 8, gi=0, gj=0, gk=0),
            PatchSpec(0, 8, 0, 8, 0, 8, gi=1, gj=0, gk=0),
            PatchSpec(0, 8, 0, 8, 0, 8, gi=0, gj=1, gk=0),
            PatchSpec(0, 8, 0, 8, 0, 8, gi=0, gj=0, gk=1),
            PatchSpec(0, 8, 0, 8, 0, 8, gi=1, gj=1, gk=1),
        ]
        phases = _checkerboard_phases(patches)
        # (0,0,0) -> phase 0
        assert patches[0] in phases[0]
        # (1,0,0) -> phase 4
        assert patches[1] in phases[4]
        # (0,1,0) -> phase 2
        assert patches[2] in phases[2]
        # (0,0,1) -> phase 1
        assert patches[3] in phases[1]
        # (1,1,1) -> phase 7
        assert patches[4] in phases[7]

    def test_empty_input(self):
        phases = _checkerboard_phases([])
        assert len(phases) == 8
        assert all(len(ph) == 0 for ph in phases)


# ---------------------------------------------------------------------------
# _filter_patches
# ---------------------------------------------------------------------------


class TestFilterPatches:
    def test_all_valid(self):
        """Patches with full weight/mask should pass."""
        nx, ny, nz = 20, 20, 20
        weight = torch.ones(nz, ny, nx)
        mask = torch.ones(nz, ny, nx, dtype=torch.uint8)
        patches = [
            PatchSpec(ibot=0, itop=8, jbot=0, jtop=8, kbot=0, ktop=8, gi=0, gj=0, gk=0),
            PatchSpec(ibot=5, itop=13, jbot=5, jtop=13, kbot=5, ktop=13, gi=1, gj=1, gk=1),
        ]
        valid = _filter_patches(patches, weight, mask, nx, ny, nz)
        assert len(valid) == 2

    def test_tiny_patch_filtered(self):
        """Patches smaller than 5x5x5 should be filtered out."""
        nx, ny, nz = 20, 20, 20
        weight = torch.ones(nz, ny, nx)
        mask = torch.ones(nz, ny, nx, dtype=torch.uint8)
        patches = [
            PatchSpec(ibot=0, itop=3, jbot=0, jtop=3, kbot=0, ktop=3, gi=0, gj=0, gk=0),
        ]
        valid = _filter_patches(patches, weight, mask, nx, ny, nz)
        assert len(valid) == 0

    def test_low_mask_coverage_filtered(self):
        """Patches with <33.3% mask coverage should be filtered."""
        nx, ny, nz = 20, 20, 20
        weight = torch.ones(nz, ny, nx)
        mask = torch.zeros(nz, ny, nx, dtype=torch.uint8)
        # Only a tiny corner is masked
        mask[0, 0, 0] = 1
        patches = [
            PatchSpec(ibot=0, itop=8, jbot=0, jtop=8, kbot=0, ktop=8, gi=0, gj=0, gk=0),
        ]
        valid = _filter_patches(patches, weight, mask, nx, ny, nz)
        assert len(valid) == 0

    def test_low_weight_filtered(self):
        """Patches with low weight sum should be filtered."""
        nx, ny, nz = 20, 20, 20
        weight = torch.ones(nz, ny, nx) * 0.01
        mask = torch.ones(nz, ny, nx, dtype=torch.uint8)
        # Most weight is outside the patch
        weight[10:, 10:, 10:] = 100.0
        patches = [
            PatchSpec(ibot=0, itop=8, jbot=0, jtop=8, kbot=0, ktop=8, gi=0, gj=0, gk=0),
        ]
        valid = _filter_patches(patches, weight, mask, nx, ny, nz)
        assert len(valid) == 0

    def test_mixed(self):
        """Mix of valid and invalid patches."""
        nx, ny, nz = 20, 20, 20
        weight = torch.ones(nz, ny, nx)
        mask = torch.ones(nz, ny, nx, dtype=torch.uint8)
        # Zero out mask for one region
        mask[:5, :5, :5] = 0
        patches = [
            # This patch has no mask coverage
            PatchSpec(ibot=0, itop=4, jbot=0, jtop=4, kbot=0, ktop=4, gi=0, gj=0, gk=0),
            # This patch has full coverage
            PatchSpec(ibot=6, itop=14, jbot=6, jtop=14, kbot=6, ktop=14, gi=1, gj=1, gk=1),
        ]
        valid = _filter_patches(patches, weight, mask, nx, ny, nz)
        assert len(valid) == 1
        assert valid[0].ibot == 6


# ---------------------------------------------------------------------------
# _compute_hfactor
# ---------------------------------------------------------------------------


class TestComputeHfactor:
    # Signature: _compute_hfactor(patch_size, patch_size_lev1, hfactor_q).
    # patch_size_lev1 is the COARSEST (level-1) patch size; finer levels
    # use smaller patch_size, yielding prat<1 and hfactor<1 (tighter bound).

    def test_at_lev1(self):
        """At level 1 (patch_size == lev1), hfactor should be 1.0."""
        assert _compute_hfactor(25, 25, 0.5) == pytest.approx(1.0)

    def test_above_lev1_clamps_to_one(self):
        """patch_size >= patch_size_lev1 clamps hfactor to 1.0."""
        assert _compute_hfactor(50, 25, 0.5) == pytest.approx(1.0)

    def test_hfactor_q_1(self):
        """With hfactor_q=1.0, always returns 1.0 (disabled)."""
        assert _compute_hfactor(10, 25, 1.0) == pytest.approx(1.0)

    def test_hfactor_q_too_small(self):
        """With hfactor_q < 0.1, returns 1.0."""
        assert _compute_hfactor(10, 25, 0.05) == pytest.approx(1.0)

    def test_finer_patch_smaller_hfactor(self):
        """Finer (smaller) patches than lev1 should get smaller hfactor."""
        h10 = _compute_hfactor(10, 25, 0.5)
        h5 = _compute_hfactor(5, 25, 0.5)
        assert h10 < 1.0
        assert h5 < h10

    def test_known_value(self):
        """Check against the formula at a finer-than-lev1 patch."""
        patch_size = 10
        patch_size_lev1 = 25
        hfactor_q = 0.5
        prat = patch_size / patch_size_lev1
        alpha = math.log(hfactor_q) / math.log(0.1)
        expected = prat**alpha
        assert _compute_hfactor(patch_size, patch_size_lev1, hfactor_q) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _get_basis_config
# ---------------------------------------------------------------------------


class TestGetBasisConfig:
    def test_cubic_lite(self):
        basis, hw, pmax = _get_basis_config("cubic_lite", 9, 9, 9, DEVICE)
        assert basis.ndim == 2  # (n_basis, n_voxels)
        assert basis.shape[1] == 9 * 9 * 9
        assert pmax == pytest.approx(0.0421)
        assert len(hw) == 3

    def test_cubic(self):
        basis, hw, pmax = _get_basis_config("cubic", 9, 9, 9, DEVICE)
        assert basis.shape[1] == 9 * 9 * 9
        assert pmax == pytest.approx(0.0280)

    def test_quintic_lite(self):
        basis, hw, pmax = _get_basis_config("quintic_lite", 9, 9, 9, DEVICE)
        assert basis.shape[1] == 9 * 9 * 9
        assert pmax == pytest.approx(0.0267)

    def test_quintic(self):
        basis, hw, pmax = _get_basis_config("quintic", 9, 9, 9, DEVICE)
        assert basis.shape[1] == 9 * 9 * 9
        assert pmax == pytest.approx(0.0099)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown basis type"):
            _get_basis_config("banana", 9, 9, 9, DEVICE)

    def test_hfactor_scales_param_max(self):
        _, _, pmax_full = _get_basis_config("cubic_lite", 9, 9, 9, DEVICE, hfactor=1.0)
        _, _, pmax_half = _get_basis_config("cubic_lite", 9, 9, 9, DEVICE, hfactor=0.5)
        assert pmax_half == pytest.approx(pmax_full * 0.5)

    def test_cubic_more_basis_than_lite(self):
        basis_lite, _, _ = _get_basis_config("cubic_lite", 9, 9, 9, DEVICE)
        basis_full, _, _ = _get_basis_config("cubic", 9, 9, 9, DEVICE)
        assert basis_full.shape[0] > basis_lite.shape[0]


# ---------------------------------------------------------------------------
# _maybe_compile
# ---------------------------------------------------------------------------


class TestMaybeCompile:
    def test_cpu_returns_original(self):
        """On CPU, should always return the original function."""
        fn = lambda x: x
        result = _maybe_compile(fn, "test_fn", torch.device("cpu"), do_compile=True)
        assert result is fn

    def test_compile_false_returns_original(self):
        fn = lambda x: x
        result = _maybe_compile(fn, "test_fn2", torch.device("cpu"), do_compile=False)
        assert result is fn


# ---------------------------------------------------------------------------
# _autobox
# ---------------------------------------------------------------------------


class TestAutobox:
    def test_all_nonzero(self):
        weight = torch.ones(8, 10, 12)
        imin, imax, jmin, jmax, kmin, kmax = _autobox(weight)
        assert imin == 0
        assert imax == 11  # nx-1
        assert jmin == 0
        assert jmax == 9
        assert kmin == 0
        assert kmax == 7

    def test_all_zero(self):
        """All-zero weight should return full volume bounds."""
        weight = torch.zeros(8, 10, 12)
        imin, imax, jmin, jmax, kmin, kmax = _autobox(weight)
        assert imin == 0
        assert imax == 11
        assert jmin == 0
        assert jmax == 9
        assert kmin == 0
        assert kmax == 7

    def test_partial_nonzero(self):
        """Bounding box should tightly wrap the nonzero region."""
        weight = torch.zeros(16, 16, 16)
        weight[3:7, 4:10, 2:8] = 1.0
        imin, imax, jmin, jmax, kmin, kmax = _autobox(weight)
        assert imin == 2
        assert imax == 7
        assert jmin == 4
        assert jmax == 9
        assert kmin == 3
        assert kmax == 6

    def test_single_voxel(self):
        weight = torch.zeros(8, 10, 12)
        weight[3, 5, 7] = 1.0
        imin, imax, jmin, jmax, kmin, kmax = _autobox(weight)
        assert imin == 7
        assert imax == 7
        assert jmin == 5
        assert jmax == 5
        assert kmin == 3
        assert kmax == 3


# ---------------------------------------------------------------------------
# Integration-style: patch grid + checkerboard + filter pipeline
# ---------------------------------------------------------------------------


class TestPatchPipeline:
    def test_grid_to_phases_roundtrip(self):
        """Generate patches, assign to phases, verify count matches."""
        patches = _generate_patch_grid(
            ibbb=1,
            ittt=30,
            jbbb=1,
            jttt=30,
            kbbb=1,
            kttt=30,
            xwid=11,
            ywid=11,
            zwid=11,
            xdel=5,
            ydel=5,
            zdel=5,
        )
        phases = _checkerboard_phases(patches)
        total_in_phases = sum(len(ph) for ph in phases)
        assert total_in_phases == len(patches)

    def test_checkerboard_separates_adjacent(self):
        """Adjacent patches (step=xdel) should land in different phases."""
        patches = _generate_patch_grid(
            ibbb=0,
            ittt=31,
            jbbb=0,
            jttt=31,
            kbbb=0,
            kttt=31,
            xwid=9,
            ywid=9,
            zwid=9,
            xdel=4,
            ydel=4,
            zdel=4,
        )
        phases = _checkerboard_phases(patches)
        # Build lookup: grid index -> phase
        grid_to_phase = {}
        for ph_idx, phase in enumerate(phases):
            for p in phase:
                grid_to_phase[(p.gi, p.gj, p.gk)] = ph_idx
        # Adjacent grid neighbors (differ by 1 in any axis) should be in different phases
        for key, ph in grid_to_phase.items():
            for d in range(3):
                neighbor = list(key)
                neighbor[d] += 1
                neighbor = tuple(neighbor)
                if neighbor in grid_to_phase:
                    assert grid_to_phase[neighbor] != ph, (
                        f"Adjacent patches {key} and {neighbor} in same phase {ph}"
                    )

    def test_filter_then_checkerboard(self):
        """Filtering first, then checkerboarding should still partition."""
        nx, ny, nz = 32, 32, 32
        weight = torch.ones(nz, ny, nx)
        mask = torch.ones(nz, ny, nx, dtype=torch.uint8)
        patches = _generate_patch_grid(
            ibbb=1,
            ittt=30,
            jbbb=1,
            jttt=30,
            kbbb=1,
            kttt=30,
            xwid=9,
            ywid=9,
            zwid=9,
            xdel=4,
            ydel=4,
            zdel=4,
        )
        valid = _filter_patches(patches, weight, mask, nx, ny, nz)
        phases = _checkerboard_phases(valid)
        total = sum(len(ph) for ph in phases)
        assert total == len(valid)
        assert total > 0


class TestPatchWriteBackDedup:
    """Patches in a phase overlap by a voxel, so write-backs must be deduplicated.

    The checkerboard is often described as partitioning into non-overlapping sets, but
    the sweep steps ``(width-1)//2``: same-parity patches are ``width-1`` apart and so
    share the plane where they abut. A raw scatter through the patch index is then an
    ``index_put_`` with duplicate targets, whose winner is unspecified -- that made two
    identical qwarp runs differ by tenths of a voxel.
    """

    def test_dedup_keeps_the_last_writer(self):
        dst = torch.tensor([5, 3, 5, 9, 3, 5])
        keep_dst, keep_src = _dedup_last_wins(dst)
        assert keep_dst.tolist() == [3, 5, 9]
        # last occurrence of each target: 3->index 4, 5->index 5, 9->index 3
        assert keep_src.tolist() == [4, 5, 3]

        values = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        out = torch.zeros(10)
        out[keep_dst] = values[keep_src]
        assert out[3] == 50.0 and out[5] == 60.0 and out[9] == 40.0

    def test_dedup_is_a_noop_on_disjoint_targets(self):
        dst = torch.tensor([4, 0, 7, 2])
        keep_dst, keep_src = _dedup_last_wins(dst)
        assert keep_dst.tolist() == [0, 2, 4, 7]
        assert dst[keep_src].tolist() == keep_dst.tolist()

    def test_phase_patches_really_do_overlap(self):
        """Guard: if the lattice ever becomes disjoint, the dedup is dead code."""
        patches = _generate_patch_grid(
            ibbb=1,
            ittt=30,
            jbbb=1,
            jttt=30,
            kbbb=1,
            kttt=30,
            xwid=9,
            ywid=9,
            zwid=9,
            xdel=4,
            ydel=4,
            zdel=4,
        )
        phases = _checkerboard_phases(patches)
        overlapping = 0
        for phase in phases:
            seen = set()
            for p in phase:
                vox = {
                    (k, j, i)
                    for k in range(p.kbot, p.ktop + 1)
                    for j in range(p.jbot, p.jtop + 1)
                    for i in range(p.ibot, p.itop + 1)
                }
                overlapping += len(seen & vox)
                seen |= vox
        assert overlapping > 0, "patches within a phase are disjoint -- dedup unnecessary?"

    @pytest.mark.slow
    def test_qwarp_batch_is_deterministic(self):
        """The batched multi-volume write-back must not depend on scatter luck."""
        from fastfuncstuff.processing.interp import warp_image
        from fastfuncstuff.processing.warp import qwarp_batch

        nz, ny, nx = 8, 20, 20
        g = torch.Generator().manual_seed(4)
        base = torch.rand(nz, ny, nx, generator=g)
        base = torch.nn.functional.avg_pool3d(base[None, None], 3, 1, 1)[0, 0] + 0.05
        zz, yy, xx = torch.meshgrid(
            torch.linspace(0, 1, nz),
            torch.linspace(0, 1, ny),
            torch.linspace(0, 1, nx),
            indexing="ij",
        )
        w = 1.2 * torch.sin(2 * torch.pi * xx) * torch.sin(torch.pi * yy) * torch.sin(torch.pi * zz)
        z0 = torch.zeros_like(w)
        sources = torch.stack([warp_image(base, z0, -s * w, z0, mode="linear") for s in (0.8, 1.0)])

        cfg = QwarpConfig(minpatch=9, cost_method="lpa", verb=0)
        a = qwarp_batch(base, sources, config=cfg)
        b = qwarp_batch(base, sources, config=cfg)
        for fa, fb in zip(a[1:], b[1:], strict=True):
            assert torch.equal(fa, fb), "qwarp_batch is not reproducible"

    def test_qwarp_cpu_level_zero_uses_resident_batched_optimizer(self, monkeypatch):
        """CPU level zero must not cross into SciPy/Powell for every evaluation."""
        import fastfuncstuff.processing.warp as warp_mod

        def fail_serial(*args, **kwargs):
            raise AssertionError("serial level-zero optimizer was called")

        monkeypatch.setattr(warp_mod, "_improve_warp_serial", fail_serial)
        base = torch.rand(12, 12, 12) + 0.1
        cfg = QwarpConfig(
            minpatch=9,
            max_level=0,
            cost_method="pearson",
            batch_optimizer_iters_lev0=1,
            verb=0,
        )
        warp_mod.qwarp(base, base.clone(), config=cfg, device=torch.device("cpu"), pad=False)


# ---------------------------------------------------------------------------
# WarpState mutation patterns
# ---------------------------------------------------------------------------


class TestPyramid:
    """Opt-in coarse-to-fine resolution pyramid (config.pyramid_factor)."""

    def _base_and_source(self):
        from fastfuncstuff.processing.interp import warp_image_linear

        torch.manual_seed(0)
        b = torch.rand(24, 26, 24)
        for _ in range(4):
            b = (
                b
                + torch.roll(b, 1, 0)
                + torch.roll(b, -1, 0)
                + torch.roll(b, 1, 1)
                + torch.roll(b, -1, 1)
                + torch.roll(b, 1, 2)
                + torch.roll(b, -1, 2)
            ) / 7.0
        zz, yy, xx = torch.meshgrid(
            *[torch.arange(n, dtype=torch.float32) for n in b.shape], indexing="ij"
        )
        s = warp_image_linear(
            b,
            1.5 * torch.sin(2 * math.pi * yy / b.shape[1]),
            1.0 * torch.cos(2 * math.pi * zz / b.shape[0]),
            0.8 * torch.sin(2 * math.pi * xx / b.shape[2]),
        )
        return b, s

    def test_default_off(self):
        assert QwarpConfig().pyramid_factor == 1

    @pytest.mark.slow
    def test_pyramid_runs_and_improves(self):
        # GPU-first project: exercise the full multi-level path on CUDA (fast),
        # and only a tiny smoke on CPU so the suite isn't pegged for seconds.
        from fastfuncstuff.processing.warp import _global_correlation, qwarp

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_level = 5 if dev.type == "cuda" else 1
        b, s = self._base_and_source()
        b, s = b.to(dev), s.to(dev)
        cfg = QwarpConfig(verb=0, minpatch=11, max_level=max_level, pyramid_factor=2)
        w, xd, yd, zd = qwarp(b, s, config=cfg, device=dev)
        assert torch.isfinite(w).all()
        # _global_correlation returns negated correlation (lower = better)
        ones = torch.ones_like(b)
        c_raw = _global_correlation(b, s, ones, None, None)
        c_warp = _global_correlation(b, w, ones, None, None)
        assert c_warp < c_raw, f"pyramid warp did not improve: {c_warp:.4f} vs raw {c_raw:.4f}"


class TestLevelDumper:
    """Per-level dump: -save_intermediates (folder) / -partial_warps (files) /
    -partials (concatenated 4D movie)."""

    def _dumper(self, tmp_path, **kw):
        from fastfuncstuff.cli.qwarp import _LevelDumper

        return _LevelDumper(str(tmp_path / "p"), None, None, 8, 8, 8, **kw)

    def test_folder_mode_writes_warp_and_image(self, tmp_path):
        v = torch.zeros(8, 8, 8)
        d = self._dumper(tmp_path, folder=True, warp_files=False, movie=False)
        d(0, v, v, v, v)
        assert (tmp_path / "p_levels" / "p_lev00.nii.gz").exists()
        assert (tmp_path / "p_levels" / "p_WARP_lev00.nii.gz").exists()

    def test_partial_warps_beside_prefix(self, tmp_path):
        v = torch.zeros(8, 8, 8)
        d = self._dumper(tmp_path, folder=False, warp_files=True, movie=False)
        d(1, v, v, v, v)
        assert (tmp_path / "p_WARP_lev01.nii.gz").exists()
        assert not (tmp_path / "p_lev01.nii.gz").exists()

    def test_partials_movie_is_one_4d_file(self, tmp_path):
        import nibabel as nib

        d = self._dumper(tmp_path, folder=False, warp_files=False, movie=True)
        for lev in range(3):
            d(
                lev,
                torch.zeros(8, 8, 8),
                torch.zeros(8, 8, 8),
                torch.zeros(8, 8, 8),
                torch.full((8, 8, 8), float(lev)),
            )
        d.finalize()
        out = tmp_path / "p_partials.nii.gz"
        assert out.exists()
        # one 4D file with a frame per level (no per-level files written)
        assert nib.load(str(out)).shape == (8, 8, 8, 3)
        assert not (tmp_path / "p_lev00.nii.gz").exists()


class TestWarpStateMutation:
    def test_warp_fields_independent(self):
        """Modifying one state's warp should not affect another."""
        ws1 = WarpState()
        ws2 = WarpState()
        ws1.xd = torch.ones(3, 3, 3)
        assert ws2.xd.numel() == 0

    def test_state_with_dimensions(self):
        ws = WarpState(nx=16, ny=16, nz=8)
        ws.xd = torch.zeros(8, 16, 16)
        ws.yd = torch.zeros(8, 16, 16)
        ws.zd = torch.zeros(8, 16, 16)
        ws.warped_source = torch.randn(8, 16, 16)
        assert ws.xd.shape == (8, 16, 16)
        assert ws.cost == pytest.approx(666.666)


class TestGaussNewtonPatchOptimizer:
    """GN replaces Adam's backward pass with normal equations built directly.

    Measured on a real 193^3 T1->MNI fit: backward was 58% of runtime and Adam
    needed ~24 evaluations per patch group. GN reaches AFNI-equivalent alignment in
    13.1s against Adam's 35.0s and 3dQwarp's 543s.
    """

    def _pair(self, n=24):
        import numpy as np
        import torch

        z, y, x = np.mgrid[0:n, 0:n, 0:n]
        c = n // 2

        def blob(cx, r, amp=100.0):
            return (
                amp * np.exp(-(((x - cx) ** 2 + (y - c) ** 2 + (z - c) ** 2) / (2 * r**2)))
            ).astype(np.float32)

        base = torch.from_numpy(blob(c, n / 5) + blob(c - n / 5, n / 12, 40.0))
        moving = torch.from_numpy(blob(c + 1.5, n / 4.5) + blob(c - n / 5 + 1.5, n / 12, 40.0))
        return base, moving

    def _run(self, **kw):
        import torch

        from fastfuncstuff.processing.warp import QwarpConfig, qwarp

        base, moving = self._pair()
        cfg = QwarpConfig(verb=0, cost_method="pearclp", minpatch=9, **kw)
        return qwarp(base, moving, config=cfg, device=torch.device("cpu"))

    def test_gauss_newton_improves_the_alignment(self):
        import torch

        from fastfuncstuff.processing.metrics import MetricInputs, evaluate_metrics

        base, moving = self._pair()
        warped, *_ = self._run(use_gauss_newton=True)
        before = evaluate_metrics(MetricInputs(base=base, moving=moving), ["ls"])["ls"]
        after = evaluate_metrics(MetricInputs(base=base, moving=warped), ["ls"])["ls"]
        assert after < before
        assert torch.isfinite(warped).all()

    def test_lands_near_the_adam_answer(self):
        """Different route, same destination -- within a tolerance, since the two
        optimisers do genuinely converge to different local points."""
        from fastfuncstuff.processing.metrics import MetricInputs, evaluate_metrics

        base, _ = self._pair()
        adam, *_ = self._run()
        gn, *_ = self._run(use_gauss_newton=True)
        a = evaluate_metrics(MetricInputs(base=base, moving=adam), ["ls"])["ls"]
        g = evaluate_metrics(MetricInputs(base=base, moving=gn), ["ls"])["ls"]
        assert abs(a - g) < 0.15, f"GN diverged from Adam: {g:.4f} vs {a:.4f}"

    def test_produces_a_sound_warp(self):
        from fastfuncstuff.processing.mask import automask
        from fastfuncstuff.processing.warpqc import regularity_verdict, warp_regularity

        base, _ = self._pair()
        _, xd, yd, zd = self._run(use_gauss_newton=True)
        off = [(a - b) // 2 for a, b in zip(xd.shape, base.shape, strict=True)]
        sl = tuple(slice(o, o + s) for o, s in zip(off, base.shape, strict=True))
        qc = warp_regularity(xd[sl], yd[sl], zd[sl], mask=automask(base))
        assert regularity_verdict(qc)[0] != "fail"

    def test_is_deterministic(self):
        import torch

        a = self._run(use_gauss_newton=True)[0]
        b = self._run(use_gauss_newton=True)[0]
        assert torch.equal(a, b)

    def test_falls_back_to_adam_for_costs_without_a_surrogate(self):
        """The descriptor costs have no least-squares residual, so GN cannot apply
        -- and asking for it must not silently produce a different (or broken)
        answer. lpa and lncc DO have one, via locally normalised residuals; see
        TestGaussNewtonLocalCosts."""
        import torch

        from fastfuncstuff.processing.warp import QwarpConfig, qwarp

        base, moving = self._pair()
        dev = torch.device("cpu")
        plain = qwarp(
            base, moving, config=QwarpConfig(verb=0, cost_method="mind", minpatch=9), device=dev
        )[0]
        asked = qwarp(
            base,
            moving,
            config=QwarpConfig(verb=0, cost_method="mind", minpatch=9, use_gauss_newton=True),
            device=dev,
        )[0]
        assert torch.equal(plain, asked)

    def test_adam_remains_the_default(self):
        """GN is opt-in: it lands on AFNI's answer where Adam lands slightly past
        it, so the safer route stays the default until the benchmark says otherwise."""
        from fastfuncstuff.processing.warp import QwarpConfig

        assert QwarpConfig().use_gauss_newton is False


class TestGaussNewtonLocalCosts:
    """The local-Pearson costs get GN through a locally normalised residual.

    lpa has no residual form of its own -- AFNI aggregates z*|z| over Fisher-z
    transformed local correlations -- but minimising the squared difference of
    locally normalised patches maximises local correlation, which points the same
    way. Measured on a real 193^3 fit: lpa Adam 96.7s -> GN 15.9s, with GN slightly
    *better* on ls, mi and lncc.
    """

    def _pair(self, n=24):
        import numpy as np
        import torch

        z, y, x = np.mgrid[0:n, 0:n, 0:n]
        c = n // 2

        def blob(cx, r, amp=100.0):
            return (
                amp * np.exp(-(((x - cx) ** 2 + (y - c) ** 2 + (z - c) ** 2) / (2 * r**2)))
            ).astype(np.float32)

        return (
            torch.from_numpy(blob(c, n / 5) + blob(c - n / 5, n / 12, 40.0)),
            torch.from_numpy(blob(c + 1.5, n / 4.5) + blob(c - n / 5 + 1.5, n / 12, 40.0)),
        )

    def _run(self, cost, **kw):
        import torch

        from fastfuncstuff.processing.warp import QwarpConfig, qwarp

        base, moving = self._pair()
        return qwarp(
            base,
            moving,
            config=QwarpConfig(verb=0, cost_method=cost, minpatch=9, **kw),
            device=torch.device("cpu"),
        )

    @pytest.mark.parametrize("cost", ["lpa", "lncc"])
    def test_local_gn_improves_alignment(self, cost):
        from fastfuncstuff.processing.metrics import MetricInputs, evaluate_metrics

        base, moving = self._pair()
        warped, *_ = self._run(cost, use_gauss_newton=True)
        before = evaluate_metrics(MetricInputs(base=base, moving=moving), ["ls"])["ls"]
        after = evaluate_metrics(MetricInputs(base=base, moving=warped), ["ls"])["ls"]
        assert after < before

    @pytest.mark.parametrize("cost", ["lpa", "lncc"])
    def test_local_gn_tracks_the_adam_answer(self, cost):
        from fastfuncstuff.processing.metrics import MetricInputs, evaluate_metrics

        base, _ = self._pair()
        a = evaluate_metrics(MetricInputs(base=base, moving=self._run(cost)[0]), ["ls"])["ls"]
        g = evaluate_metrics(
            MetricInputs(base=base, moving=self._run(cost, use_gauss_newton=True)[0]), ["ls"]
        )["ls"]
        assert abs(a - g) < 0.15, f"{cost}: GN {g:.4f} vs Adam {a:.4f}"

    def test_lpc_is_excluded_from_gauss_newton(self):
        """lpc rewards anti-correlation. A sum-of-squares residual between
        normalised patches can only pull them together, so the surrogate would
        point the wrong way -- it must fall back rather than optimise backwards."""
        import torch

        plain = self._run("lpc")[0]
        asked = self._run("lpc", use_gauss_newton=True)[0]
        assert torch.equal(plain, asked)

    def test_local_gn_produces_a_sound_warp(self):
        from fastfuncstuff.processing.mask import automask
        from fastfuncstuff.processing.warpqc import regularity_verdict, warp_regularity

        base, _ = self._pair()
        _, xd, yd, zd = self._run("lpa", use_gauss_newton=True)
        off = [(a - b) // 2 for a, b in zip(xd.shape, base.shape, strict=True)]
        sl = tuple(slice(o, o + s) for o, s in zip(off, base.shape, strict=True))
        qc = warp_regularity(xd[sl], yd[sl], zd[sl], mask=automask(base))
        assert regularity_verdict(qc)[0] != "fail"
