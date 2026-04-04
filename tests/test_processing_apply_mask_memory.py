from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.apply import apply_warp, compose_warps, invert_warp
from fastfuncstuff.processing.mask import automask
from fastfuncstuff.processing.memory import (
    compute_chunk_plan,
    estimate_gpu_memory_bytes,
    estimate_gpu_memory_gb,
)
from fastfuncstuff.processing.quadrature import (
    apply_quadrature_filters_fft,
    compute_phase_diff_and_certainty,
    design_quadrature_filters,
    precompute_filter_ffts,
    quadrature_phase_cost,
)


class TestApplyWarp:
    def test_apply_warp_zero_displacement(self):
        vol = torch.randn(8, 8, 8)
        zero = torch.zeros(8, 8, 8)
        result = apply_warp(vol, zero, zero.clone(), zero.clone())
        assert result.shape == vol.shape
        assert torch.allclose(result, vol, atol=1e-5)

    def test_compose_warps_zero(self):
        shape = (8, 8, 8)
        z = torch.zeros(*shape)
        warp_a = (z.clone(), z.clone(), z.clone())
        warp_b = (z.clone(), z.clone(), z.clone())
        xc, yc, zc = compose_warps(warp_a, warp_b)
        assert torch.allclose(xc, z, atol=1e-5)
        assert torch.allclose(yc, z, atol=1e-5)
        assert torch.allclose(zc, z, atol=1e-5)

    def test_invert_warp_zero(self):
        shape = (8, 8, 8)
        z = torch.zeros(*shape)
        ix, iy, iz = invert_warp(z.clone(), z.clone(), z.clone(), n_iter=5)
        assert torch.allclose(ix, z, atol=1e-4)
        assert torch.allclose(iy, z, atol=1e-4)
        assert torch.allclose(iz, z, atol=1e-4)


class TestAutomask:
    def _make_sphere(self, shape=(20, 20, 20), radius=6):
        vol = torch.zeros(*shape, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(
            torch.arange(shape[0]),
            torch.arange(shape[1]),
            torch.arange(shape[2]),
            indexing="ij",
        )
        cz, cy, cx = shape[0] / 2, shape[1] / 2, shape[2] / 2
        dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        vol[dist2 <= radius**2] = 1.0
        return vol, dist2

    def test_automask_bright_center(self):
        vol, dist2 = self._make_sphere()
        mask = automask(vol)
        assert mask.sum() > 0
        n_sphere = int((dist2 <= 36).sum())
        assert mask.sum() >= n_sphere * 0.3

    def test_automask_zeros(self):
        vol = torch.zeros(10, 10, 10, dtype=torch.float32)
        mask = automask(vol)
        assert mask.sum() <= 5

    def test_automask_output_bool(self):
        vol = torch.randn(10, 10, 10, dtype=torch.float32).abs()
        mask = automask(vol)
        assert mask.dtype == torch.bool

    def test_automask_shape(self):
        vol = torch.randn(10, 10, 10, dtype=torch.float32).abs()
        mask = automask(vol)
        assert mask.shape == vol.shape


class TestMemory:
    def test_estimate_memory_keys(self):
        result = estimate_gpu_memory_bytes(64, 64, 64)
        expected_keys = {
            "base_image",
            "source_image",
            "weight",
            "mask",
            "warp_xd",
            "warp_yd",
            "warp_zd",
            "warped_source",
            "basis_matrix",
            "coord_grids",
            "patch_warp_temps",
            "jacobian_temps",
            "incor_temps",
            "pytorch_overhead",
            "total",
        }
        assert expected_keys <= set(result.keys())

    def test_estimate_memory_larger_volume(self):
        small = estimate_gpu_memory_bytes(32, 32, 32)
        large = estimate_gpu_memory_bytes(64, 64, 64)
        assert large["total"] > small["total"]

    def test_compute_chunk_plan(self):
        plan = compute_chunk_plan(64, 64, 64, nt=5)
        assert isinstance(plan, list)
        assert len(plan) == 5
        for i, (start, end) in enumerate(plan):
            assert start == i
            assert end == i + 1
        covered = set()
        for start, end in plan:
            covered.update(range(start, end))
        assert covered == set(range(5))

    def test_estimate_memory_gb_positive(self):
        gb = estimate_gpu_memory_gb(64, 64, 64)
        assert isinstance(gb, float)
        assert gb > 0


class TestQuadrature:
    def test_design_quadrature_filters_shape(self):
        size = 7
        filters = design_quadrature_filters(size=size)
        assert filters.shape == (3, size, size, size)
        assert filters.is_complex()

    def test_precompute_filter_ffts_shape(self):
        size = 7
        vol_shape = (16, 16, 16)
        filters = design_quadrature_filters(size=size)
        spectra = precompute_filter_ffts(filters, vol_shape)
        assert spectra.shape == (3, *vol_shape)

    def test_apply_filters_shape(self):
        vol_shape = (16, 16, 16)
        vol = torch.randn(*vol_shape)
        filters = design_quadrature_filters(size=7)
        spectra = precompute_filter_ffts(filters, vol_shape)
        responses = apply_quadrature_filters_fft(vol, spectra)
        assert responses.shape == (3, *vol_shape)

    def test_phase_diff_identical(self):
        vol_shape = (8, 8, 8)
        vol = torch.randn(*vol_shape)
        filters = design_quadrature_filters(size=7)
        spectra = precompute_filter_ffts(filters, vol_shape)
        q = apply_quadrature_filters_fft(vol, spectra)
        phase_diff, certainty = compute_phase_diff_and_certainty(q, q)
        assert phase_diff.shape[0] == 3
        nonzero = phase_diff[phase_diff.abs() > 1e-8]
        if nonzero.numel() > 0:
            assert nonzero.abs().max() < 0.1 or nonzero.numel() < phase_diff.numel() * 0.5

    def test_phase_cost_identical(self):
        vol_shape = (8, 8, 8)
        vol = torch.randn(*vol_shape)
        filters = design_quadrature_filters(size=7)
        spectra = precompute_filter_ffts(filters, vol_shape)
        q = apply_quadrature_filters_fft(vol, spectra)
        cost = quadrature_phase_cost(q, q)
        assert cost.item() > 0
