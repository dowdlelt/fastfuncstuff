"""Tests for processing/tissue.py — the partial-volume tissue-synthesis cost."""

import numpy as np
import torch

from fastfuncstuff.processing.bbr import identity_params
from fastfuncstuff.processing.tissue import (
    build_tissue_design,
    tissue_projector,
    tissue_synthesis_cost,
)

DEV = torch.device("cpu")


def _three_tissue_phantom(nz=8, ny=40, nx=32):
    """WM/GM/CSF fraction volumes (soft) + an EPI that is their intensity mixture.

    Layered along y: WM (dark) → GM (mid) → CSF (bright), with soft transitions
    so partial volume is present — the regime the synthesis cost is built for.
    """
    y = torch.arange(ny, dtype=torch.float32)[None, :, None]

    def band(lo, hi, w=1.5):
        return 0.5 * (torch.tanh((y - lo) / w) - torch.tanh((y - hi) / w))

    f_wm = band(4, 16).expand(nz, ny, nx).contiguous()
    f_gm = band(16, 24).expand(nz, ny, nx).contiguous()
    f_csf = band(24, 30).expand(nz, ny, nx).contiguous()
    # EPI intensity = tissue mixture (WM dark, GM mid, CSF bright) + DC.
    epi = 20.0 + 40.0 * f_wm + 120.0 * f_gm + 220.0 * f_csf
    return epi, [f_wm, f_gm, f_csf]


def _cost_at(epi, coords, F, Fpinv, ty):
    p = identity_params(DEV).clone()
    p[7] = ty  # translate along y
    return tissue_synthesis_cost(epi, coords, F, Fpinv, p).item()


def test_projector_profiles_out_means_exactly():
    epi, tissues = _three_tissue_phantom()
    coords, F = build_tissue_design(tissues, n_sample=None, device=DEV)
    Fpinv = tissue_projector(F)
    # At identity the EPI IS a mixture of these fractions, so the residual ~0.
    c0 = _cost_at(epi, coords, F, Fpinv, 0.0)
    assert c0 < 1.0  # EPI range ~20..340, so this is near-perfect explanation


def test_cost_minimized_at_true_alignment():
    epi, tissues = _three_tissue_phantom()
    coords, F = build_tissue_design(tissues, n_sample=None, device=DEV)
    Fpinv = tissue_projector(F)
    curve = {ty: _cost_at(epi, coords, F, Fpinv, ty) for ty in np.arange(-4, 4.01, 0.5)}
    best = min(curve, key=curve.get)
    assert abs(best) < 0.75  # argmin at the true (zero) displacement
    # Normalized cost is ≈1−R² (O(1)); a clear well means displaced ≫ aligned.
    assert curve[0.0] < 0.01  # near-perfect explanation at true alignment
    assert curve[3.0] > 0.1 and curve[-3.0] > 0.1  # much worse when misaligned


def test_recovers_injected_shift_by_gradient_descent():
    # The dense synthesis cost has a SHARP, narrow well (unlike the wider BBR/NGF),
    # so it needs a decaying LR to settle at the bottom — it's a refinement cost.
    epi, tissues = _three_tissue_phantom()
    coords, F = build_tissue_design(tissues, n_sample=None, device=DEV)
    Fpinv = tissue_projector(F)
    pivot = coords.mean(0)
    p = identity_params(DEV).clone()
    p[7] = 2.5  # start displaced along y
    p = p.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 600)
    for _ in range(600):
        opt.zero_grad()
        loss = tissue_synthesis_cost(epi, coords, F, Fpinv, p, pivot=pivot)
        loss.backward()
        opt.step()
        sched.step()
    assert abs(p[7].item()) < 0.3  # pulled back to true alignment


def test_subsample_is_deterministic_and_smaller():
    _, tissues = _three_tissue_phantom()
    c1, F1 = build_tissue_design(tissues, n_sample=500, device=DEV, seed=1)
    c2, F2 = build_tissue_design(tissues, n_sample=500, device=DEV, seed=1)
    assert c1.shape[0] == 500 and F1.shape == (500, 4)  # 3 tissues + intercept
    assert torch.equal(c1, c2)  # same seed → same subsample


def test_out_of_fov_penalized():
    # A huge translation samples off the grid (→0) and must cost MORE than aligned.
    epi, tissues = _three_tissue_phantom()
    coords, F = build_tissue_design(tissues, n_sample=None, device=DEV)
    Fpinv = tissue_projector(F)
    assert _cost_at(epi, coords, F, Fpinv, 100.0) > _cost_at(epi, coords, F, Fpinv, 0.0)
