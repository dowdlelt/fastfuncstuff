"""Tests for processing/rbr.py — the nonlinear RBR stage (control-point FFD BBR).

The headline check recovers a known *local* (spatially varying) phase-encode
distortion — something the affine stage cannot represent — proving the nonlinear
warp does what the linear one can't.
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.bbr import (
    extract_boundary_normals,
    extract_edge_normals,
    gradient_field,
)
from fastfuncstuff.processing.interp import trilinear_interpolate
from fastfuncstuff.processing.rbr import (
    control_grid_shape,
    dense_field,
    eval_displacement,
    invert_displacement_field,
    membrane_penalty,
    optimize_rbr,
    resample_with_affine_field,
    upsample_ctrl,
)

DEV = torch.device("cpu")


def _curved_edge(nz=12, ny=48, nx=32, amp=3.0, sig=5.0, y0=20.0, width=1.5):
    """EPI-like volume whose WM/GM edge is bumped upward by a Gaussian in x."""
    x = torch.arange(nx, dtype=torch.float32)
    bump = amp * torch.exp(-((x - nx / 2) ** 2) / (2 * sig**2))  # (nx,)
    edge = y0 + bump
    yy = torch.arange(ny, dtype=torch.float32)[:, None]
    prof = 10 + 100 * 0.5 * (1 - torch.tanh((yy - edge[None, :]) / width))  # (ny, nx)
    vol = prof[None].expand(nz, ny, nx).contiguous()
    return vol, bump


# ── FFD building blocks ──────────────────────────────────────────────────────


class TestControlGridWarp:
    def test_zero_ctrl_zero_displacement(self):
        ctrl = torch.zeros(1, 3, 3, 3)
        pts = torch.rand(20, 3) * torch.tensor([31.0, 47.0, 11.0])
        d = eval_displacement(ctrl, pts, (12, 48, 32))
        assert d.shape == (20, 1)
        assert torch.allclose(d, torch.zeros_like(d))

    def test_constant_ctrl_constant_displacement(self):
        ctrl = torch.full((1, 4, 4, 4), 2.5)
        pts = torch.tensor([[5.0, 10.0, 3.0], [20.0, 30.0, 8.0]])
        d = eval_displacement(ctrl, pts, (12, 48, 32))
        assert torch.allclose(d, torch.full((2, 1), 2.5), atol=1e-4)

    def test_dense_field_axes_placement(self):
        ctrl = torch.ones(1, 2, 2, 2)
        field = dense_field(ctrl, (6, 8, 10), axes=[1])
        assert field.shape == (3, 6, 8, 10)
        assert torch.allclose(field[1], torch.ones(6, 8, 10), atol=1e-4)  # y filled
        assert torch.allclose(field[0], torch.zeros(6, 8, 10))  # x untouched
        assert torch.allclose(field[2], torch.zeros(6, 8, 10))  # z untouched

    def test_membrane_penalty_zero_for_constant(self):
        assert membrane_penalty(torch.full((1, 4, 4, 4), 3.0)).item() < 1e-10
        assert membrane_penalty(torch.randn(1, 4, 4, 4)).item() > 0

    def test_control_grid_shape_and_upsample(self):
        K = control_grid_shape((12, 48, 32), spacing=8.0)
        assert K == (3, 7, 5)  # round(12/8)+1, round(48/8)+1, round(32/8)+1
        up = upsample_ctrl(torch.zeros(1, 2, 2, 2), (3, 5, 4))
        assert up.shape == (1, 3, 5, 4)


# ── The nonlinear recovery (proof of principle) ──────────────────────────────


class TestRBRRecovery:
    def test_recovers_local_pe_distortion(self):
        vol, bump = _curved_edge()
        wm = np.zeros((12, 48, 32), dtype=bool)
        wm[:, :20, :] = True  # straight boundary; the EPI edge is curved
        pts, nrm = extract_boundary_normals(wm, device=DEV)

        res = optimize_rbr(
            vol,
            pts,
            nrm,
            axes=[1],
            spacings=[16, 8, 4],
            offsets=[4, 3, 2],  # wide coarse offset for the ~3-vox bump's capture range
            reg_weight=0.2,
            iters=250,
        )
        assert res["final_cost"] < res["init_cost"] - 0.1  # meaningfully better

        # Recovered y-displacement at the boundary tracks the true bump shape.
        rec = trilinear_interpolate(res["field"][1], pts[:, 0], pts[:, 1], pts[:, 2])
        truth = bump[pts[:, 0].round().long().clamp(0, 31)]
        corr = torch.corrcoef(torch.stack([rec, truth]))[0, 1].item()
        assert corr > 0.9  # affine (a single global shift) would give ~0 here
        # captures a real fraction of the bump amplitude (regularizer shrinks it)
        assert (rec.max() - rec.min()).item() > 0.4 * bump.max().item()

    def test_field_shape_and_pe_only(self):
        vol, _ = _curved_edge()
        wm = np.zeros((12, 48, 32), dtype=bool)
        wm[:, :20, :] = True
        pts, nrm = extract_boundary_normals(wm, device=DEV)
        res = optimize_rbr(vol, pts, nrm, axes=[1], spacings=[12, 6], iters=60)
        assert res["field"].shape == (3, 12, 48, 32)
        # PE-axis-only warp leaves x and z components exactly zero
        assert torch.count_nonzero(res["field"][0]) == 0
        assert torch.count_nonzero(res["field"][2]) == 0


# ── Inverse displacement field ───────────────────────────────────────────────


class TestInverseField:
    def test_compose_forward_inverse_is_identity(self):
        # Smooth pure-y field up to 1.5 vox; d⁻¹ must satisfy d(p+d⁻¹(p)) = -d⁻¹(p).
        nz, ny, nx = 8, 40, 24
        field = torch.zeros(3, nz, ny, nx)
        yy = torch.arange(ny).float()[None, :, None]
        field[1] = (1.5 * torch.sin(2 * np.pi * yy / ny)).expand(nz, ny, nx).contiguous()
        inv = invert_displacement_field(field, iters=25)

        zz, yg, xg = torch.meshgrid(
            torch.arange(nz).float(),
            torch.arange(ny).float(),
            torch.arange(nx).float(),
            indexing="ij",
        )
        px, py, pz = (xg + inv[0]).reshape(-1), (yg + inv[1]).reshape(-1), (zz + inv[2]).reshape(-1)
        d_at = trilinear_interpolate(field[1], px, py, pz).reshape(nz, ny, nx)
        # interior only (fixed-point is exact away from the clamp at the boundary)
        resid = (d_at + inv[1])[:, 5:-5, :].abs()
        assert resid.max().item() < 0.02

    def test_inverse_undistorts_back_to_source(self):
        # raw -> pull by d = undistorted; undistorted pulled by d⁻¹ recovers raw.
        vol, _ = _curved_edge(nz=8, ny=40, nx=24, amp=2.0)
        nz, ny, nx = vol.shape
        field = torch.zeros(3, nz, ny, nx)
        yy = torch.arange(ny).float()[None, :, None]
        field[1] = (1.2 * torch.sin(2 * np.pi * yy / ny)).expand(nz, ny, nx).contiguous()
        inv = invert_displacement_field(field, iters=25)
        eye = torch.eye(4)
        und = resample_with_affine_field(vol, eye, field, (nz, ny, nx))
        back = resample_with_affine_field(und, eye, inv, (nz, ny, nx))
        interior = (back - vol)[:, 6:-6, 3:-3].abs()
        assert interior.mean().item() < 0.5  # vol spans ~10..110


# ── NGF edge target in the nonlinear stage ───────────────────────────────────


class TestRBREdgeTarget:
    def test_edges_only_runs_and_improves(self):
        # Curved bright/dark edge; anat edge points sit on the *straight* boundary,
        # the warp must bend to the EPI's curved gradient. WM cloud left empty.
        vol, bump = _curved_edge()
        grad = gradient_field(vol)
        straight = np.zeros((12, 48, 32), dtype=np.float32)
        straight[:, :20, :] = 100.0  # straight edge at y=20; blur→gradient normals
        epts, enrm = extract_edge_normals(
            torch.from_numpy(straight), mask=None, blur=1.0, percentile=50.0, device=DEV
        )
        empty = torch.empty((0, 3))
        res = optimize_rbr(
            vol,
            empty,
            empty,
            axes=[1],
            spacings=[16, 8],
            offsets=3.0,
            reg_weight=0.2,
            edge_points=epts,
            edge_normals=enrm,
            grad_field=grad,
            iters=150,
        )
        assert res["final_cost"] < res["init_cost"] - 0.01

    def test_needs_at_least_one_cloud(self):
        vol, _ = _curved_edge(nz=6, ny=20, nx=12)
        empty = torch.empty((0, 3))
        with pytest.raises(ValueError, match="needs a WM boundary"):
            optimize_rbr(vol, empty, empty, axes=[1], spacings=[8])

    def test_combined_wm_and_tissue_field_improves(self):
        # The realistic path: a WM-BBR term provides capture range and constrains
        # the field, and a dense tissue (partial-volume) term rides along. A curved
        # edge with a straight WM mask + a straight 3-tissue design; the warp bends.
        from fastfuncstuff.processing.tissue import build_tissue_design, tissue_projector

        vol, _ = _curved_edge()
        nz, ny, nx = vol.shape
        wm = np.zeros((nz, ny, nx), dtype=bool)
        wm[:, :20, :] = True
        pts, nrm = extract_boundary_normals(wm, device=DEV)

        yy = torch.arange(ny, dtype=torch.float32)[None, :, None]

        def band(lo, hi, w=1.5):
            return 0.5 * (torch.tanh((yy - lo) / w) - torch.tanh((yy - hi) / w))

        tissues = [
            band(0, 20).expand(nz, ny, nx).contiguous(),
            band(20, 30).expand(nz, ny, nx).contiguous(),
            band(30, 48).expand(nz, ny, nx).contiguous(),
        ]
        coords, F = build_tissue_design(tissues, n_sample=None, device=DEV)
        Fpinv = tissue_projector(F)
        res = optimize_rbr(
            vol,
            pts,
            nrm,
            axes=[1],
            spacings=[16, 8],
            offsets=[4, 2],
            reg_weight=0.2,
            tissue_coords=coords,
            tissue_F=F,
            tissue_Fpinv=Fpinv,
            tissue_weight=0.001,  # small: WM drives, tissue refines
            iters=200,
        )
        assert res["final_cost"] < res["init_cost"] - 0.05
        assert res["field"].abs().max().item() < 6.0  # constrained, not diverged
        assert res["field"].shape == (3, nz, ny, nx)
