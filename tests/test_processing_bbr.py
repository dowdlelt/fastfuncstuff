"""Tests for processing/bbr.py — the BBR boundary-contrast cost (RBR Phase 1).

Synthetic-data checks that would catch real bugs:
  - normals point outward from a known surface,
  - the cost is minimized at the transform that puts the boundary back on the
    true edge (recovering a known phase-encode-axis displacement),
  - the cost is differentiable and a gradient step improves it,
  - the reverse-contrast switch flips the rewarded sign.
"""

import numpy as np
import torch

from fastfuncstuff.processing.affine import apply_affine
from fastfuncstuff.processing.bbr import (
    apply_transform,
    auto_polarity,
    bbr_cost,
    boundary_contrast,
    boundary_reliability,
    correct_sign_fraction,
    extract_boundary_normals,
    extract_edge_normals,
    gradient_field,
    greve_fischl_cost,
    identity_params,
    ngf_cost,
    optimize_bbr,
    rst_matrix,
)

DEV = torch.device("cpu")


def _sphere_mask(n=32, r=10.0):
    c = (n - 1) / 2.0
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    return ((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2) <= r**2, c


def _y_edge_volume(nz=16, ny=40, nx=16, y0=20.0, width=1.5, bright=100.0, base=10.0):
    """Smooth intensity step along y: bright for y < y0, dark above."""
    y = torch.arange(ny, dtype=torch.float32)
    prof = base + bright * 0.5 * (1.0 - torch.tanh((y - y0) / width))  # (ny,)
    return prof[None, :, None].expand(nz, ny, nx).contiguous()


# ── extract_boundary_normals ─────────────────────────────────────────────────


class TestExtractBoundaryNormals:
    def test_normals_point_outward_on_sphere(self):
        mask, c = _sphere_mask(n=32, r=10.0)
        pts, nrm = extract_boundary_normals(mask, device=DEV)
        assert pts.shape[1] == 3 and nrm.shape == pts.shape
        # unit length
        assert torch.allclose(nrm.norm(dim=1), torch.ones(len(nrm)), atol=1e-4)
        # radial outward: normal aligned with (p − centre)
        radial = pts - torch.tensor([c, c, c])
        radial = radial / radial.norm(dim=1, keepdim=True).clamp_min(1e-6)
        cos = (nrm * radial).sum(dim=1)
        assert cos.mean() > 0.95

    def test_points_lie_near_boundary(self):
        mask, c = _sphere_mask(n=32, r=10.0)
        pts, _ = extract_boundary_normals(mask, device=DEV, refine=True)
        radius = (pts - torch.tensor([c, c, c])).norm(dim=1)
        # refined points should sit close to r on the zero level set
        assert abs(radius.mean().item() - 10.0) < 0.75

    def test_empty_mask_raises(self):
        try:
            extract_boundary_normals(np.zeros((8, 8, 8)), device=DEV)
        except ValueError:
            return
        raise AssertionError("empty mask should raise")


# ── rst_matrix / apply_transform ─────────────────────────────────────────────


class TestTransform:
    def test_identity_is_identity(self):
        pts = torch.randn(20, 3)
        nrm = torch.randn(20, 3)
        nrm = nrm / nrm.norm(dim=1, keepdim=True)
        mat = rst_matrix(identity_params(), pivot=pts.mean(0))
        p2, n2 = apply_transform(pts, nrm, mat)
        assert torch.allclose(p2, pts, atol=1e-5)
        assert torch.allclose(n2, nrm, atol=1e-5)

    def test_translation_shifts_points_not_normals(self):
        pts = torch.zeros(5, 3)
        nrm = torch.tensor([[0.0, 1.0, 0.0]]).expand(5, 3).contiguous()
        p = identity_params()
        p[7] = 3.0  # ty
        mat = rst_matrix(p, pivot=pts.mean(0))
        p2, n2 = apply_transform(pts, nrm, mat)
        assert torch.allclose(p2[:, 1], torch.full((5,), 3.0), atol=1e-5)
        assert torch.allclose(n2, nrm, atol=1e-5)  # pure translation leaves normals

    def test_scale_about_pivot_fixes_pivot(self):
        pts = torch.tensor([[0.0, 5.0, 0.0], [0.0, 15.0, 0.0]])
        pivot = pts.mean(0)  # y = 10
        p = identity_params()
        p[4] = 2.0  # sy
        mat = rst_matrix(p, pivot=pivot)
        p2, _ = apply_transform(pts, pts * 0 + torch.tensor([0.0, 1.0, 0.0]), mat)
        # pivot maps to pivot + t (t=0); points scaled about it
        assert torch.allclose(p2[:, 1], torch.tensor([0.0, 20.0]), atol=1e-4)


# ── greve_fischl_cost ────────────────────────────────────────────────────────


class TestGreveFischlCost:
    def test_positive_contrast_low_cost(self):
        c = torch.full((100,), 0.5)
        assert greve_fischl_cost(c, reverse=False).item() < 0
        assert greve_fischl_cost(c, reverse=True).item() > 0

    def test_reverse_flips_sign(self):
        c = torch.linspace(-0.8, 0.8, 50)
        a = greve_fischl_cost(c, reverse=False)
        b = greve_fischl_cost(c, reverse=True)
        assert torch.allclose(a, -b, atol=1e-6)


# ── boundary_contrast + bbr_cost (the core behaviour) ────────────────────────


class TestBBRCostRecovery:
    def test_contrast_positive_at_true_edge(self):
        vol = _y_edge_volume()
        # boundary on the true edge, normal +y (WM below is bright)
        pts = torch.tensor([[8.0, 20.0, 8.0]])
        nrm = torch.tensor([[0.0, 1.0, 0.0]])
        c = boundary_contrast(vol, pts, nrm, offset=2.0)
        assert c.item() > 0.5  # bright(white) − dark(grey) → strong positive

    def test_recovers_known_pe_shift(self):
        vol = _y_edge_volume(y0=20.0)
        # WM mask edge deliberately displaced along y (edge at y=25)
        wm = np.zeros((16, 40, 16), dtype=bool)
        wm[:, :25, :] = True
        pts, nrm = extract_boundary_normals(wm, device=DEV)
        # normals should point +y (outward toward larger y)
        assert nrm[:, 1].mean() > 0.9

        boundary_y = pts[:, 1].mean().item()
        ty_star = 20.0 - boundary_y  # shift that lands the boundary on the true edge

        tys = torch.linspace(-10.0, 4.0, 57)
        costs = []
        for ty in tys:
            p = identity_params()
            p[7] = ty
            costs.append(bbr_cost(vol, pts, nrm, p, offset=2.0).item())
        best = tys[int(np.argmin(costs))].item()
        # Recovers the ~4.5-voxel displacement to within a voxel. The residual
        # offset is the normalized-contrast (white−grey)/(white+grey) asymmetry,
        # which biases the peak a fraction of a voxel toward the darker side —
        # a real property the iterative BBR washes out, not a placement bug.
        assert abs(best - ty_star) < 1.5

    def test_cost_is_differentiable_and_improves(self):
        vol = _y_edge_volume(y0=20.0)
        wm = np.zeros((16, 40, 16), dtype=bool)
        wm[:, :25, :] = True
        pts, nrm = extract_boundary_normals(wm, device=DEV)

        params = identity_params().clone().requires_grad_(True)
        # Start within the BBR capture range (offset sets it): far from the edge
        # both samples sit in a flat region and the gradient vanishes by design.
        with torch.no_grad():
            params[7] = -1.0
        opt = torch.optim.Adam([params], lr=0.3)
        c0 = bbr_cost(vol, pts, nrm, params, offset=3.0)
        assert c0.requires_grad
        cost_start = c0.item()
        for _ in range(60):
            opt.zero_grad()
            c = bbr_cost(vol, pts, nrm, params, offset=3.0)
            c.backward()
            assert torch.isfinite(params.grad).all()
            opt.step()
        cost_end = bbr_cost(vol, pts, nrm, params, offset=3.0).item()
        assert cost_end < cost_start - 1e-2
        # converged toward the boundary-onto-edge shift (~ −4)
        assert params[7].item() < -2.5

    def test_reverse_contrast_for_dark_wm(self):
        # GM-bright target (T2*/EPI-like): bright ABOVE the edge.
        vol = _y_edge_volume(y0=20.0, bright=-90.0, base=100.0)  # dark below, bright above
        pts = torch.tensor([[8.0, 20.0, 8.0]])
        nrm = torch.tensor([[0.0, 1.0, 0.0]])
        c = boundary_contrast(vol, pts, nrm, offset=2.0)
        assert c.item() < -0.5  # white side now darker → negative contrast
        # reverse=True rewards it (low cost); reverse=False penalizes it
        p = identity_params()
        assert bbr_cost(vol, pts, nrm, p, offset=2.0, reverse=True).item() < 0
        assert bbr_cost(vol, pts, nrm, p, offset=2.0, reverse=False).item() > 0


class TestOptimizeAndCompose:
    """optimize_bbr + the ffs_bbr affine-composition direction (A' = T @ A)."""

    def test_auto_polarity_detects_wm_bright(self):
        vol = _y_edge_volume(y0=20.0)  # WM bright below, GM dark above
        wm = np.zeros((16, 40, 16), dtype=bool)
        wm[:, :20, :] = True
        pts, nrm = extract_boundary_normals(wm, device=DEV)
        assert auto_polarity(vol, pts, nrm, offset=2.0) is False
        # invert brightness → GM brighter → reverse should be chosen
        assert auto_polarity(-vol + 200.0, pts, nrm, offset=2.0) is True

    def test_optimize_recovers_pure_shift(self):
        vol = _y_edge_volume(y0=19.5)
        wm = np.zeros((16, 40, 16), dtype=bool)
        wm[:, :23, :] = True  # boundary displaced ~3 vox from the edge
        pts, nrm = extract_boundary_normals(wm, device=DEV)
        res = optimize_bbr(vol, pts, nrm, mode="shift", offset=2.0, coarse_range=6.0)
        assert res["final_cost"] < res["init_cost"]
        # boundary at ~22.5, edge at 19.5 → ty ≈ −3 (± the contrast-asymmetry bias)
        assert res["params"][7].item() < -1.5

    def test_affine_composition_direction(self):
        """A' = T @ A removes a *known* alignment error and lands at a residual
        independent of that error — the definitive inv/compose direction check.
        A sign or inverse bug would instead land near 2·delta."""
        # anat grid == epi grid; true WM/GM edge at y=19.5, WM bright below.
        epi = _y_edge_volume(y0=19.5)
        wm = torch.zeros(16, 40, 16)
        wm[:, :20, :] = 1.0  # anat WM boundary ~19.5 (matches the true edge)

        residuals = []
        for delta in (3.0, -2.0):
            A = torch.eye(4, dtype=torch.float64)
            A[1, 3] = delta  # anat→epi voxel error: +delta along y (PE axis)
            A_inv = torch.linalg.inv(A)
            wm_epi = (
                apply_affine(wm, A_inv.float(), tuple(epi.shape), zero_outside=True) > 0.5
            ).float()
            pts, nrm = extract_boundary_normals(wm_epi, device=DEV)
            rev = auto_polarity(epi, pts, nrm, offset=2.0, pivot=pts.mean(0))
            assert rev is False  # WM bright
            res = optimize_bbr(
                epi, pts, nrm, mode="shift", offset=2.0, reverse=rev, coarse_range=6.0
            )
            assert res["final_cost"] < res["init_cost"]
            # boundary well-seated after refinement (QC is scale-free)
            f1 = correct_sign_fraction(epi, pts, nrm, res["params"], offset=2.0, reverse=rev)
            assert f1 > 0.8
            A_new = res["matrix"].double() @ A
            # residual alignment error is tiny and NOT ~2·delta (sign/inverse bug)
            assert abs(A_new[1, 3].item()) < 1.5
            residuals.append(A_new[1, 3].item())
        # same physical solution regardless of the injected error
        assert abs(residuals[0] - residuals[1]) < 0.5


class TestReliability:
    def test_flat_region_downweighted(self):
        # Left half has a sharp y-edge; right half is flat (no EPI edge).
        vol = torch.zeros(8, 40, 32)
        yy = torch.arange(40, dtype=torch.float32)
        edge = 100 * 0.5 * (1 - torch.tanh((yy - 20) / 1.5))
        vol[:, :, :16] = edge[None, :, None]  # x<16 has an edge; x>=16 stays flat 0
        pts_edge = torch.tensor([[8.0, 20.0, 4.0]])  # on the edge
        pts_flat = torch.tensor([[24.0, 20.0, 4.0]])  # in the flat region
        w_edge = boundary_reliability(vol, pts_edge)
        w_flat = boundary_reliability(vol, pts_flat)
        assert w_edge.item() > w_flat.item()
        assert w_flat.item() < 0.2  # flat → near the floor

    def test_weights_in_range(self):
        vol = _y_edge_volume()
        pts = torch.tensor([[8.0, 20.0, 8.0], [8.0, 5.0, 8.0]])
        w = boundary_reliability(vol, pts, floor=0.05)
        assert (w >= 0.05).all() and (w <= 1.0).all()


class TestEdgeAndNGF:
    def test_extract_edges_on_y_step(self):
        vol = _y_edge_volume(y0=20.0)  # edge in y at 20
        pts, nrm = extract_edge_normals(vol, blur=0.5, percentile=90.0, device=DEV)
        assert pts.shape[1] == 3 and nrm.shape == pts.shape
        assert torch.allclose(nrm.norm(dim=1), torch.ones(len(nrm)), atol=1e-3)
        assert nrm[:, 1].abs().mean() > 0.9  # normals point along y (the edge normal)
        assert abs(pts[:, 1].mean().item() - 20.0) < 2.0  # edge points cluster at y≈20

    def _edge_pts(self, y):
        xs = [6.0, 8.0, 10.0]
        zs = [6.0, 8.0, 10.0]
        pts = torch.tensor([[x, y, z] for x in xs for z in zs])
        nrm = torch.tensor([[0.0, 1.0, 0.0]]).expand(len(pts), 3).contiguous()
        return pts, nrm

    def test_ngf_recovers_shift_and_is_polarity_agnostic(self):
        best = {}
        for label, (bright, base) in {
            "wm_bright": (100.0, 10.0),
            "wm_dark": (-100.0, 110.0),
        }.items():
            vol = _y_edge_volume(y0=20.0, bright=bright, base=base)
            grad = gradient_field(vol)
            pts, nrm = self._edge_pts(23.0)  # anat edge displaced +3 vox from the EPI edge
            tys = torch.linspace(-8.0, 2.0, 41)
            costs = []
            for ty in tys:
                p = identity_params()
                p[7] = ty
                costs.append(ngf_cost(grad, pts, nrm, p).item())
            best[label] = tys[int(np.argmin(costs))].item()
            # moves the anat edge onto the EPI edge (ty ≈ −3)
            assert abs(best[label] - (-3.0)) < 1.0
        # polarity flip gives the same answer (NGF matches gradient direction, squared)
        assert abs(best["wm_bright"] - best["wm_dark"]) < 0.5

    def test_ngf_differentiable(self):
        vol = _y_edge_volume(y0=20.0)
        grad = gradient_field(vol)
        pts, nrm = self._edge_pts(22.0)
        p = identity_params().clone().requires_grad_(True)
        c = ngf_cost(grad, pts, nrm, p)
        c.backward()
        assert torch.isfinite(p.grad).all()


def test_offset_can_be_per_point_tensor():
    vol = _y_edge_volume()
    pts = torch.tensor([[8.0, 20.0, 8.0], [8.0, 20.0, 8.0]])
    nrm = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    off = torch.tensor([1.0, 3.0])
    c = boundary_contrast(vol, pts, nrm, offset=off)
    assert c.shape == (2,) and torch.isfinite(c).all()
