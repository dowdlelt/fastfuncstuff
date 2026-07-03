"""Tests for processing/cost.py — cost functions for image matching."""

import torch

from fastfuncstuff.processing.cost import (
    BatchedIncrementalCorrelation,
    IncrementalCorrelation,
    _auto_clip,
    _batched_separable_smooth_3d,
    _box_kernel_1d,
    _gauss_kernel_1d,
    _make_kernel_1d,
    _separable_smooth_3d,
    auto_box_radius,
    batched_lpa_cost,
    clipped_pearson_correlation,
    lpa_correlation,
    lpa_cost_patch,
    lpc_correlation,
    pearson_correlation,
)

DEV = torch.device("cpu")


# ── pearson_correlation ──


class TestPearsonCorrelation:
    def test_identical_signals(self):
        x = torch.randn(100, device=DEV)
        r = pearson_correlation(x, x)
        assert abs(r.item() - 1.0) < 1e-5

    def test_negated_signals(self):
        x = torch.randn(100, device=DEV)
        r = pearson_correlation(x, -x)
        assert abs(r.item() + 1.0) < 1e-5

    def test_uncorrelated(self):
        torch.manual_seed(42)
        x = torch.randn(10000, device=DEV)
        y = torch.randn(10000, device=DEV)
        r = pearson_correlation(x, y)
        assert abs(r.item()) < 0.05

    def test_with_weights(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEV)
        y = torch.tensor([2.0, 4.0, 6.0, 8.0], device=DEV)
        w = torch.tensor([1.0, 1.0, 1.0, 1.0], device=DEV)
        r = pearson_correlation(x, y, weight=w)
        assert abs(r.item() - 1.0) < 1e-5

    def test_zero_weights(self):
        x = torch.randn(10, device=DEV)
        y = torch.randn(10, device=DEV)
        w = torch.zeros(10, device=DEV)
        r = pearson_correlation(x, y, weight=w)
        assert r.item() == 0.0

    def test_constant_signal_returns_zero(self):
        x = torch.ones(50, device=DEV)
        y = torch.randn(50, device=DEV)
        r = pearson_correlation(x, y)
        assert abs(r.item()) < 1e-5


# ── clipped_pearson_correlation ──


class TestClippedPearsonCorrelation:
    def test_identical_signals(self):
        x = torch.randn(200, device=DEV)
        r = clipped_pearson_correlation(x, x)
        assert r.item() > 0.99

    def test_manual_clips(self):
        x = torch.randn(100, device=DEV)
        y = x + 0.01 * torch.randn(100, device=DEV)
        r = clipped_pearson_correlation(x, y, base_clip=(-2.0, 2.0), source_clip=(-2.0, 2.0))
        assert r.item() > 0.9


# ── _auto_clip ──


class TestAutoClip:
    def test_basic_range(self):
        d = torch.arange(100, dtype=torch.float32, device=DEV)
        lo, hi = _auto_clip(d)
        assert lo >= 0
        assert hi <= 99
        assert lo < hi

    def test_small_data(self):
        d = torch.tensor([1.0, 2.0], device=DEV)
        lo, hi = _auto_clip(d)
        assert lo == 1.0
        assert hi == 2.0

    def test_with_weight_mask(self):
        d = torch.arange(100, dtype=torch.float32, device=DEV)
        w = torch.zeros(100, device=DEV)
        w[20:80] = 1.0
        lo, hi = _auto_clip(d, weight=w)
        assert lo >= 20
        assert hi <= 79


# ── Kernel functions ──


class TestKernels:
    def test_gauss_kernel_normalized(self):
        k = _gauss_kernel_1d(2.0, DEV)
        assert abs(k.sum().item() - 1.0) < 1e-6

    def test_gauss_kernel_symmetric(self):
        k = _gauss_kernel_1d(3.0, DEV)
        n = k.shape[0]
        for i in range(n // 2):
            assert abs(k[i].item() - k[n - 1 - i].item()) < 1e-6

    def test_box_kernel_normalized(self):
        k = _box_kernel_1d(3, DEV)
        assert abs(k.sum().item() - 1.0) < 1e-6

    def test_box_kernel_uniform(self):
        k = _box_kernel_1d(2, DEV)
        assert k.shape[0] == 5  # 2*2+1
        expected = 1.0 / 5
        assert (k - expected).abs().max().item() < 1e-6

    def test_box_kernel_min_radius(self):
        k = _box_kernel_1d(0, DEV)
        assert k.shape[0] == 3  # radius clamped to 1

    def test_make_kernel_gauss(self):
        k = _make_kernel_1d("gauss", 2.0, DEV)
        assert abs(k.sum().item() - 1.0) < 1e-6

    def test_make_kernel_box(self):
        k = _make_kernel_1d("box", 3.0, DEV)
        assert k.shape[0] == 7  # 2*3+1


class TestAutoBoxRadius:
    def test_default_500(self):
        r = auto_box_radius(500)
        side = 2 * r + 1
        assert side**3 >= 500

    def test_small_target(self):
        r = auto_box_radius(8)
        assert r >= 1

    def test_monotonic(self):
        r1 = auto_box_radius(100)
        r2 = auto_box_radius(1000)
        assert r2 >= r1


# ── _separable_smooth_3d ──


class TestSeparableSmooth3D:
    def test_constant_volume_unchanged(self):
        vol = torch.ones(8, 10, 12, device=DEV)
        smoothed = _separable_smooth_3d(vol, 2.0)
        torch.testing.assert_close(smoothed, vol, atol=1e-5, rtol=1e-5)

    def test_reduces_variance(self):
        torch.manual_seed(0)
        vol = torch.randn(10, 12, 14, device=DEV)
        smoothed = _separable_smooth_3d(vol, 2.0)
        assert smoothed.var().item() < vol.var().item()

    def test_preserves_shape(self):
        vol = torch.randn(8, 10, 12, device=DEV)
        smoothed = _separable_smooth_3d(vol, 2.0)
        assert smoothed.shape == vol.shape

    def test_5d_input(self):
        vol = torch.randn(1, 1, 8, 10, 12, device=DEV)
        smoothed = _separable_smooth_3d(vol, 2.0)
        assert smoothed.shape == vol.shape

    def test_box_kernel(self):
        vol = torch.randn(8, 10, 12, device=DEV)
        smoothed = _separable_smooth_3d(vol, 2.0, kernel_type="box")
        assert smoothed.shape == vol.shape
        assert smoothed.var().item() < vol.var().item()


# ── lpa_correlation ──


class TestLPACorrelation:
    def test_identical_images_high_corr(self):
        torch.manual_seed(1)
        img = torch.randn(8, 10, 12, device=DEV) + 5.0
        r = lpa_correlation(img, img, sigma=2.0)
        assert r.item() > 0.95

    def test_unrelated_images_low_corr(self):
        torch.manual_seed(2)
        a = torch.randn(8, 10, 12, device=DEV)
        b = torch.randn(8, 10, 12, device=DEV)
        r = lpa_correlation(a, b, sigma=2.0)
        assert r.item() < 0.3

    def test_with_weight(self):
        torch.manual_seed(3)
        img = torch.randn(8, 10, 12, device=DEV) + 5.0
        w = torch.ones(8, 10, 12, device=DEV)
        r = lpa_correlation(img, img, weight=w, sigma=2.0)
        assert r.item() > 0.95

    def test_box_kernel(self):
        torch.manual_seed(4)
        img = torch.randn(8, 10, 12, device=DEV) + 5.0
        r = lpa_correlation(img, img, sigma=2.0, kernel_type="box")
        assert r.item() > 0.95


# ── lpc_correlation ──


class TestLPCCorrelation:
    def test_identical_images(self):
        torch.manual_seed(5)
        img = torch.randn(8, 10, 12, device=DEV) + 5.0
        r = lpc_correlation(img, img, sigma=2.0)
        # For identical images, z*|z| is positive (positive corr),
        # and we negate, so result should be negative (inverted convention).
        # Actually: identical → local_corr ≈ 1, z = atanh(0.99) ≈ 2.65,
        # z*|z| ≈ 7, mean > 0, negated → result < 0.
        # But we return -mean(z*|z|), where z*|z| is positive for positive corr.
        # So result < 0 for identical images.
        assert r.item() < 0  # identical = positive corr → negative return

    def test_with_weight(self):
        torch.manual_seed(6)
        img = torch.randn(8, 10, 12, device=DEV) + 5.0
        w = torch.ones_like(img)
        r = lpc_correlation(img, img, weight=w, sigma=2.0)
        assert isinstance(r, torch.Tensor)


# ── lpa_cost_patch ──


class TestLPACostPatch:
    def test_delegates_to_lpa_correlation(self):
        torch.manual_seed(7)
        patch = torch.randn(6, 8, 10, device=DEV) + 5.0
        w = torch.ones_like(patch)
        r = lpa_cost_patch(patch, patch, w, sigma=2.0)
        assert r.item() > 0.9


# ── _batched_separable_smooth_3d ──


class TestBatchedSeparableSmooth3D:
    def test_matches_single_smooth(self):
        """Batched smoothing should match individual smoothing."""
        torch.manual_seed(8)
        B = 3
        D, H, W = 6, 8, 10
        vols = torch.randn(B, 1, D, H, W, device=DEV)
        kernel = _gauss_kernel_1d(2.0, DEV)

        batched = _batched_separable_smooth_3d(vols, kernel)

        for i in range(B):
            single = _separable_smooth_3d(vols[i, 0], 2.0, kernel_type="gauss")
            torch.testing.assert_close(batched[i, 0], single, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        B = 4
        vols = torch.randn(B, 1, 5, 7, 9, device=DEV)
        kernel = _gauss_kernel_1d(1.5, DEV)
        out = _batched_separable_smooth_3d(vols, kernel)
        assert out.shape == vols.shape

    def test_constant_volume_unchanged(self):
        B = 2
        vols = torch.ones(B, 1, 6, 8, 10, device=DEV)
        kernel = _gauss_kernel_1d(2.0, DEV)
        out = _batched_separable_smooth_3d(vols, kernel)
        torch.testing.assert_close(out, vols, atol=1e-5, rtol=1e-5)


# ── batched_lpa_cost ──


class TestBatchedLPACost:
    def test_identical_patches_high_cost(self):
        torch.manual_seed(9)
        B = 3
        nzh, nyh, nxh = 6, 8, 10
        V = nzh * nyh * nxh
        patches = torch.randn(B, V, device=DEV) + 5.0
        w = torch.ones(B, V, device=DEV)
        costs = batched_lpa_cost(patches, patches, w, nzh, nyh, nxh, sigma=2.0)
        assert costs.shape == (B,)
        # z*|z| for perfect correlation → large positive values
        assert (costs > 0).all()

    def test_output_shape(self):
        B = 4
        V = 5 * 7 * 9
        base = torch.randn(B, V, device=DEV)
        src = torch.randn(B, V, device=DEV)
        w = torch.ones(B, V, device=DEV)
        costs = batched_lpa_cost(base, src, w, 5, 7, 9, sigma=1.5)
        assert costs.shape == (B,)

    def test_box_kernel(self):
        torch.manual_seed(10)
        B = 2
        nzh, nyh, nxh = 6, 8, 10
        V = nzh * nyh * nxh
        patches = torch.randn(B, V, device=DEV) + 5.0
        w = torch.ones(B, V, device=DEV)
        costs = batched_lpa_cost(patches, patches, w, nzh, nyh, nxh, sigma=2.0, kernel_type="box")
        assert costs.shape == (B,)
        assert (costs > 0).all()

    def test_differentiable(self):
        """batched_lpa_cost should be differentiable w.r.t. source patches."""
        torch.manual_seed(11)
        B = 2
        nzh, nyh, nxh = 5, 5, 5
        V = nzh * nyh * nxh
        base = torch.randn(B, V, device=DEV)
        src = torch.randn(B, V, device=DEV, requires_grad=True)
        w = torch.ones(B, V, device=DEV)
        costs = batched_lpa_cost(base, src, w, nzh, nyh, nxh, sigma=1.5)
        loss = costs.sum()
        loss.backward()
        assert src.grad is not None
        assert src.grad.shape == (B, V)


# ── IncrementalCorrelation ──


class TestIncrementalCorrelation:
    def test_evaluate_identical(self):
        ic = IncrementalCorrelation(method="pearclp")
        x = torch.randn(100, device=DEV)
        w = torch.ones(100, device=DEV)
        # No fixed part, just evaluate on patch
        r = ic.evaluate(x, x, w)
        assert abs(r - 1.0) < 1e-4

    def test_evaluate_with_fixed(self):
        ic = IncrementalCorrelation(method="pearclp")
        torch.manual_seed(12)
        # Fixed part
        x_fix = torch.randn(200, device=DEV)
        y_fix = x_fix + 0.01 * torch.randn(200, device=DEV)
        w_fix = torch.ones(200, device=DEV)
        ic.add_fixed(x_fix, y_fix, w_fix)

        # Variable part
        x_var = torch.randn(50, device=DEV)
        y_var = x_var + 0.01 * torch.randn(50, device=DEV)
        w_var = torch.ones(50, device=DEV)
        r = ic.evaluate(x_var, y_var, w_var)
        assert r > 0.9

    def test_set_clips(self):
        ic = IncrementalCorrelation(method="pearclp")
        ic.set_clips((-1.0, 1.0), (-2.0, 2.0))
        assert ic._base_clip == (-1.0, 1.0)
        assert ic._source_clip == (-2.0, 2.0)

    def test_empty_weight_returns_zero(self):
        ic = IncrementalCorrelation()
        x = torch.randn(10, device=DEV)
        w = torch.zeros(10, device=DEV)
        r = ic.evaluate(x, x, w)
        assert r == 0.0

    def test_clips_applied_in_evaluate(self):
        ic = IncrementalCorrelation()
        ic.set_clips((-0.5, 0.5), (-0.5, 0.5))
        x = torch.tensor([10.0, -10.0, 0.3], device=DEV)
        y = torch.tensor([10.0, -10.0, 0.3], device=DEV)
        w = torch.ones(3, device=DEV)
        r = ic.evaluate(x, y, w)
        # After clipping, x=y=[0.5, -0.5, 0.3], perfect correlation
        assert abs(r - 1.0) < 1e-4


# ── BatchedIncrementalCorrelation ──


class TestBatchedIncrementalCorrelation:
    def _make_test_data(self, B=3, nz=6, ny=8, nx=10):
        torch.manual_seed(13)
        base = torch.randn(nz, ny, nx, device=DEV)
        source = base + 0.01 * torch.randn(nz, ny, nx, device=DEV)
        weight = torch.ones(nz, ny, nx, device=DEV)
        # Create B small patches from interior
        patch_slices = []
        for i in range(B):
            ibot = i
            itop = i + 2
            patch_slices.append((ibot, itop, 1, 3, 1, 3))
        return base, source, weight, patch_slices

    def test_precompute_and_evaluate(self):
        base, source, weight, patch_slices = self._make_test_data()
        B = len(patch_slices)

        bic = BatchedIncrementalCorrelation(method="pearclp")
        bic.precompute_fixed_parts(base, source, weight, patch_slices)

        # Extract patches
        base_patches = torch.stack(
            [
                base[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)
                for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
            ]
        )
        source_patches = torch.stack(
            [
                source[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)
                for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
            ]
        )
        weight_patches = torch.stack(
            [
                weight[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)
                for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
            ]
        )

        corrs = bic.evaluate(base_patches, source_patches, weight_patches)
        assert corrs.shape == (B,)
        # All should be high since source ≈ base
        assert (corrs > 0.9).all()

    def test_with_clips(self):
        base, source, weight, patch_slices = self._make_test_data()
        bic = BatchedIncrementalCorrelation(
            method="pearclp",
            base_clip=(-2.0, 2.0),
            source_clip=(-2.0, 2.0),
        )
        bic.precompute_fixed_parts(base, source, weight, patch_slices)

        base_patches = torch.stack(
            [
                base[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)
                for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
            ]
        )
        source_patches = torch.stack(
            [
                source[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)
                for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
            ]
        )
        weight_patches = torch.stack(
            [
                weight[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)
                for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
            ]
        )

        corrs = bic.evaluate(base_patches, source_patches, weight_patches)
        assert corrs.shape == (len(patch_slices),)

    def test_with_pre_extracted_patches(self):
        base, source, weight, patch_slices = self._make_test_data()

        base_patches = torch.stack(
            [
                base[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)
                for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
            ]
        )
        weight_patches = torch.stack(
            [
                weight[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)
                for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
            ]
        )

        bic = BatchedIncrementalCorrelation()
        bic.precompute_fixed_parts(
            base,
            source,
            weight,
            patch_slices,
            base_patches=base_patches,
            weight_patches=weight_patches,
        )

        source_patches = torch.stack(
            [
                source[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)
                for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
            ]
        )

        corrs = bic.evaluate(base_patches, source_patches, weight_patches)
        assert corrs.shape == (len(patch_slices),)
        assert (corrs > 0.9).all()

    def test_differentiable(self):
        """Evaluate should be differentiable for autograd."""
        base, source, weight, patch_slices = self._make_test_data(B=2)

        bic = BatchedIncrementalCorrelation()
        bic.precompute_fixed_parts(base, source, weight, patch_slices)

        base_patches = torch.stack(
            [
                base[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)
                for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
            ]
        )
        source_patches = torch.stack(
            [
                source[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)
                for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
            ]
        ).requires_grad_(True)
        weight_patches = torch.stack(
            [
                weight[kbot : ktop + 1, jbot : jtop + 1, ibot : itop + 1].reshape(-1)
                for ibot, itop, jbot, jtop, kbot, ktop in patch_slices
            ]
        )

        corrs = bic.evaluate(base_patches, source_patches, weight_patches)
        corrs.sum().backward()
        assert source_patches.grad is not None
