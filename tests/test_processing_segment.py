"""Tests for the Unified Segmentation core (processing/segment.py)."""

from __future__ import annotations

import math
import pathlib
import tempfile

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.segment import (
    autobox_bounds,
    bias_field_shape,
    crop_affine,
    dct_basis,
    embed_in_full,
    eval_log_bias,
    fit_segment,
    fudge_factor,
    gmm_moments,
    gmm_responsibilities,
    gmm_update,
    update_tissue_weights,
    warp_penalty,
)


def _fit_gmm(corrected, prior, means, covs, mix, tissue_of, *, iters=60):
    """Run plain EM (fixed prior) to convergence; return final params + loglik."""
    cov_prior = torch.eye(corrected.shape[1], dtype=torch.float64) * 1e-4
    loglik = None
    for _ in range(iters):
        resp, ll = gmm_responsibilities(corrected, prior, means, covs, mix, tissue_of)
        count, sum1, sum2 = gmm_moments(corrected, resp)
        means, covs, mix = gmm_update(count, sum1, sum2, tissue_of, cov_prior)
        loglik = ll.sum()
    return means, covs, mix, loglik


def test_gmm_recovers_1d_three_tissue():
    """Three well-separated 1-channel tissues, one Gaussian each → EM recovers means."""
    torch.manual_seed(0)
    means_true = torch.tensor([10.0, 50.0, 90.0])
    sds = torch.tensor([3.0, 4.0, 5.0])
    per = 2000
    corrected = torch.cat([means_true[k] + sds[k] * torch.randn(per) for k in range(3)])[:, None]
    prior = torch.full((corrected.shape[0], 3), 1.0 / 3.0)  # uninformative

    means = torch.tensor([[20.0, 45.0, 70.0]])  # (n_chan=1, n_gauss=3), wrong on purpose
    covs = torch.ones(1, 1, 3) * 100.0
    mix = torch.ones(3)
    tissue_of = torch.tensor([0, 1, 2])

    means, covs, mix, _ = _fit_gmm(corrected, prior, means, covs, mix, tissue_of)
    got = torch.sort(means.flatten()).values.float()
    assert torch.allclose(got, means_true, atol=1.0), got
    got_sd = torch.sort(covs.flatten().sqrt()).values.float()
    assert torch.allclose(got_sd, sds, atol=1.0), got_sd


def test_gmm_responsibilities_sum_to_one():
    corrected = torch.randn(100, 2)
    prior = torch.rand(100, 3)
    means = torch.randn(2, 4)
    covs = torch.stack([torch.eye(2) for _ in range(4)], dim=-1)
    mix = torch.rand(4)
    tissue_of = torch.tensor([0, 0, 1, 2])
    resp, loglik = gmm_responsibilities(corrected, prior, means, covs, mix, tissue_of)
    assert torch.allclose(resp.sum(dim=1), torch.ones(100, dtype=resp.dtype), atol=1e-10)
    assert resp.min() >= 0.0
    assert loglik.shape == (100,)


def test_gmm_mixing_within_tissue_sums_to_one():
    """Two Gaussians on tissue 0, one each on 1/2 → mix sums to 1 within each tissue."""
    corrected = torch.randn(500, 1)
    resp = torch.rand(500, 4)
    resp = resp / resp.sum(dim=1, keepdim=True)
    count, sum1, sum2 = gmm_moments(corrected, resp)
    tissue_of = torch.tensor([0, 0, 1, 2])
    _, _, mix = gmm_update(count, sum1, sum2, tissue_of, torch.eye(1, dtype=torch.float64))
    assert torch.isclose(mix[:2].sum(), torch.tensor(1.0, dtype=mix.dtype))
    assert torch.isclose(mix[2], torch.tensor(1.0, dtype=mix.dtype))
    assert torch.isclose(mix[3], torch.tensor(1.0, dtype=mix.dtype))


def test_sample_tpm_prior_cubic_fractional_coords():
    """Cubic TPM sampling at fractional float64 coords (the warped output path)."""
    from fastfuncstuff.processing.segment import sample_tpm_prior

    torch.manual_seed(0)
    log_prior = torch.log(torch.rand(3, 12, 12, 12) + 1e-4)  # float32 source
    coords = torch.rand(50, 3, dtype=torch.float64) * 9 + 1  # fractional, in-bounds
    bg = torch.full((3,), 1.0 / 3)
    for kernel in ("linear", "cubic", "heptic"):
        prior = sample_tpm_prior(log_prior, coords, bg, bg, kernel=kernel)
        assert prior.shape == (50, 3)
        assert torch.allclose(prior.sum(1), torch.ones(50, dtype=prior.dtype), atol=1e-6)
        assert prior.min() >= 0.0


def test_gmm_prior_biases_assignment():
    """A voxel exactly between two tissue means goes to whichever the prior favours."""
    corrected = torch.tensor([[50.0]])
    means = torch.tensor([[0.0, 100.0]])
    covs = torch.ones(1, 1, 2) * 100.0
    mix = torch.ones(2)
    tissue_of = torch.tensor([0, 1])
    resp_lo, _ = gmm_responsibilities(
        corrected, torch.tensor([[0.9, 0.1]]), means, covs, mix, tissue_of
    )
    resp_hi, _ = gmm_responsibilities(
        corrected, torch.tensor([[0.1, 0.9]]), means, covs, mix, tissue_of
    )
    assert resp_lo[0, 0] > 0.5  # prior favours tissue 0
    assert resp_hi[0, 1] > 0.5  # prior favours tissue 1


# --- bias field ------------------------------------------------------------


def test_dct_basis_matches_spm_formula():
    """DCT-II basis: DC column is 1/√n and columns are orthonormal over the grid."""
    n = 32
    pos = torch.arange(n, dtype=torch.float64)
    basis = dct_basis(pos, n, 6)
    assert torch.allclose(basis[:, 0], torch.full((n,), 1.0 / n**0.5, dtype=torch.float64))
    gram = basis.T @ basis  # sampled on the full grid → orthonormal
    assert torch.allclose(gram, torch.eye(6, dtype=torch.float64), atol=1e-10)


def test_bias_recovers_known_low_freq_field():
    """Fit DCT coefficients (autograd) to a synthetic smooth multiplicative field."""
    torch.manual_seed(0)
    dim = (16, 20, 24)
    vox = (1.0, 1.0, 1.0)
    (nbx, nby, nbz), prec = bias_field_shape(dim, vox, fwhm=8.0, biasreg=1e-6)

    # ground-truth log-bias from a couple of low-frequency coefficients
    zz, yy, xx = torch.meshgrid(
        torch.arange(dim[2], dtype=torch.float64),
        torch.arange(dim[1], dtype=torch.float64),
        torch.arange(dim[0], dtype=torch.float64),
        indexing="ij",
    )
    pos_x, pos_y, pos_z = xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)
    bx = dct_basis(pos_x, dim[0], nbx)
    by = dct_basis(pos_y, dim[1], nby)
    bz = dct_basis(pos_z, dim[2], nbz)

    true_coef = torch.zeros(nbx, nby, nbz, dtype=torch.float64)
    true_coef[1, 0, 0] = 0.6
    true_coef[0, 1, 0] = -0.4
    target = eval_log_bias(true_coef, bx, by, bz)

    coef = torch.zeros(nbx, nby, nbz, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([coef], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        pred = eval_log_bias(coef, bx, by, bz)
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-5
    assert torch.allclose(coef.detach()[1, 0, 0], torch.tensor(0.6, dtype=torch.float64), atol=1e-2)


# --- full EM driver --------------------------------------------------------


def _phantom_three_tissue(n=24):
    """A 3-tissue block phantom + a smooth multiplicative bias; identity geometry."""
    vol = torch.zeros(n, n, n)
    zz, yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij")
    r = ((xx - n / 2) ** 2 + (yy - n / 2) ** 2 + (zz - n / 2) ** 2).sqrt()
    tissue = torch.zeros(3, n, n, n)
    tissue[0] = ((r >= 4) & (r < 8)).float()  # inner shell  = tissue 0
    tissue[1] = (r < 4).float()  # core         = tissue 1
    tissue[2] = ((r >= 8) & (r < 10)).float()  # outer shell  = tissue 2
    means = torch.tensor([80.0, 150.0, 40.0])
    for k in range(3):
        vol += means[k] * tissue[k]
    # smooth multiplicative bias in [0.6, 1.6]
    bias = 1.0 + 0.5 * torch.cos(torch.pi * xx / n) * torch.sin(torch.pi * yy / n)
    vol = vol * bias
    return vol, tissue, means, bias


def test_fit_segment_recovers_tissue_means_through_bias():
    torch.manual_seed(0)
    n = 24
    vol, tissue, means_true, _ = _phantom_three_tissue(n)
    # TPM = lightly-blurred tissue maps (a soft prior), stored as log(p+tiny)
    from fastfuncstuff.processing.segment import load_tpm  # noqa: F401  (using log form directly)

    prob = tissue.clone()
    prob = prob / prob.sum(0, keepdim=True).clamp_min(1e-6)  # normalise where any tissue
    # ensure a background floor so out-of-tissue voxels aren't degenerate
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))

    out = fit_segment(
        vol,
        eye,  # subj_affine = identity → world == voxel
        log_prior,
        eye,  # tpm_affine = identity
        bg,
        bg,
        eye,  # world_affine = identity
        ngaus=[1, 1, 1],
        biasfwhm=12.0,
        samp=1.0,
        n_iter=10,
        fit_warp=False,  # identity geometry: no deformation needed
        verbose=False,
    )
    got = torch.sort(out["means"].flatten()).values.float()
    assert torch.allclose(got, torch.sort(means_true).values, rtol=0.1), got


def test_fit_segment_pe_axis_constrains_warp():
    """PE-mode: only the phase-encode component of the deformation may be nonzero."""
    torch.manual_seed(0)
    n = 24
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))

    fit = fit_segment(
        vol,
        eye,
        log_prior,
        eye,
        bg,
        bg,
        eye,
        ngaus=[1, 1, 1],
        biasfwhm=12.0,
        samp=1.0,
        n_iter=6,
        fit_warp=True,
        pe_axis=1,
        warp_smooth=0.5,
        verbose=False,  # constrain to y (axis 1)
    )
    tw = fit["twarp"]  # (gz, gy, gx, 3)
    assert tw[..., 0].abs().max() < 1e-8  # x component pinned to 0
    assert tw[..., 2].abs().max() < 1e-8  # z component pinned to 0
    # the y component is free to move (may be ~0 on this symmetric phantom, but never forced)
    assert tw.shape[-1] == 3


def test_blur_log_prior_smooths_and_preserves_shape():
    """Blurring the TPM lowers spatial gradient energy but keeps a valid log-prior."""
    from fastfuncstuff.processing.segment import blur_log_prior

    torch.manual_seed(0)
    prob = torch.rand(3, 12, 12, 12)
    prob = prob / prob.sum(0, keepdim=True)  # a valid per-voxel prior
    log_prior = torch.log(prob + 1e-4)

    # sigma <= 0 is identity
    assert torch.equal(blur_log_prior(log_prior, 0.0), log_prior)

    blurred = blur_log_prior(log_prior, 2.0)
    assert blurred.shape == log_prior.shape
    # a valid log-prior: every probability is at least the tiny floor
    assert torch.exp(blurred).min() >= 1e-4 - 1e-6
    # blurring reduces the roughness of the underlying probabilities
    rough = lambda lp: torch.diff(torch.exp(lp), dim=1).pow(2).sum()  # noqa: E731
    assert rough(blurred) < rough(log_prior)


@pytest.mark.gpu
def test_dual_echo_reverse_pe_drives_one_warp():
    """Dual-echo PE mode: forward (+s) and reverse (−s) distortions drive a single
    PE-constrained warp; the reverse is applied with the opposite sign."""
    torch.manual_seed(0)
    n = 24
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))
    fwd = torch.roll(vol, shifts=3, dims=1)  # +PE distortion (dim1 = y = pe_axis 1)
    rev = torch.roll(vol, shifts=-3, dims=1)  # −PE distortion (opposite blip)
    kw = dict(  # noqa: C408
        ngaus=[1, 1, 1], biasfwhm=12.0, samp=1.0, n_iter=8, pe_axis=1,
        warp_anneal=False, verbose=False,
    )  # fmt: skip

    fit = fit_segment(fwd, eye, log_prior, eye, bg, bg, eye, reverse_volume=rev, **kw)
    tw = fit["twarp"]
    assert torch.isfinite(tw).all()
    assert (
        tw[..., 0].abs().max() < 1e-4 and tw[..., 2].abs().max() < 1e-4
    )  # PE-constrained (y only)
    assert tw[..., 1].abs().max() > 0.5  # both echoes drove the warp along PE

    with pytest.raises(ValueError):  # reverse without pe_axis is meaningless
        fit_segment(
            fwd, eye, log_prior, eye, bg, bg, eye, reverse_volume=rev,
            ngaus=[1, 1, 1], n_iter=2, verbose=False,
        )  # fmt: skip


@pytest.mark.gpu
def test_fit_segment_blur_tpms_runs_coarse_to_fine():
    """A coarse blurred-TPM pass still converges to a sensible, finite fit."""
    torch.manual_seed(0)
    n = 24
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))

    fit = fit_segment(
        vol, eye, log_prior, eye, bg, bg, eye,
        ngaus=[1, 1, 1], biasfwhm=12.0, samp=1.0, n_iter=8,
        fit_warp=True, blur_tpms=3.0, blur_frac=0.5, warp_smooth=0.5, verbose=False,
    )  # fmt: skip
    assert torch.isfinite(fit["twarp"]).all()
    assert torch.isfinite(fit["means"]).all()
    assert fit["twarp"].shape[-1] == 3


def test_fit_segment_chunking_matches_whole_batch():
    """Chunked sample processing reproduces the single-chunk tissue model exactly.

    The per-step accumulation (moment sums, bias/warp gradient accumulation) is
    mathematically exact, so the tissue means/covariances/mixing, the bias field, and
    the tissue weights are bit-for-bit identical (up to float64 round-off) whether the
    samples are processed in one chunk or many. A single outer iteration avoids the
    data-dependent early-stops confounding the comparison. (The dense *warp* is NOT
    asserted here: Adam normalises by sqrt(grad²), so in flat / under-determined
    regions it amplifies benign ~1e-13 grid_sample-backward round-off — a separate,
    expected effect, not a chunking bug.)
    """
    torch.manual_seed(0)
    n = 20
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))

    kw = dict(  # noqa: C408
        ngaus=[1, 1, 1], biasfwhm=10.0, samp=1.0, n_iter=1,
        fit_warp=True, warp_smooth=0.5, dtype=torch.float64, verbose=False,
    )  # fmt: skip
    whole = fit_segment(vol, eye, log_prior, eye, bg, bg, eye, fit_chunk=100000, **kw)
    chunked = fit_segment(vol, eye, log_prior, eye, bg, bg, eye, fit_chunk=97, **kw)

    assert whole["n_chunks"] == 1 and chunked["n_chunks"] > 1  # actually exercised chunking
    # tissue model + bias: computed before/independent of the warp, so chunk-exact
    assert torch.allclose(whole["means"], chunked["means"], atol=1e-9)
    assert torch.allclose(whole["covs"], chunked["covs"], atol=1e-7)
    assert torch.allclose(whole["mix"], chunked["mix"], atol=1e-9)
    assert torch.allclose(whole["coef"], chunked["coef"], atol=1e-9)
    # wp is re-estimated from the warped prior after the warp step, so it inherits the
    # warp's benign Adam-amplified drift — only loosely comparable, not chunk-exact.
    assert torch.allclose(whole["wp"], chunked["wp"], atol=5e-3)


@pytest.mark.gpu
def test_fit_segment_float32_tracks_float64():
    """The float32 hot path recovers essentially the same tissue means as float64."""
    torch.manual_seed(0)
    n = 24
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))
    kw = dict(ngaus=[1, 1, 1], biasfwhm=12.0, samp=1.0, n_iter=8, verbose=False)  # noqa: C408

    f64 = fit_segment(vol, eye, log_prior, eye, bg, bg, eye, dtype=torch.float64, **kw)
    f32 = fit_segment(vol, eye, log_prior, eye, bg, bg, eye, dtype=torch.float32, **kw)
    # means are O(50-150); float32 should track float64 to well under a grey level
    assert (f64["means"].sort().values - f32["means"].sort().values).abs().max() < 0.5


def test_wp_reg_prevents_weight_runaway():
    """SPM's observed/expected wp update self-corrects; wp_reg damps toward uniform.

    The tissue-weight update (:func:`update_tissue_weights`) is ``(observed + wp_reg)/
    (expected + wp_reg·Kb)``. Take a heavily over-observed tissue (observed mass far above
    what the model expects — the "growing brains" condition): a weak ``wp_reg`` lets that
    weight run away toward 1, while SPM's ``wp_reg=100`` holds every weight near uniform.
    Tested on the update directly so it isn't entangled with the deformation dynamics.
    """
    kb = 3
    # tissue 0 is massively over-observed vs its model-expected mass (would inflate)
    observed = torch.tensor([100.0, 5.0, 5.0], dtype=torch.float64)
    expected = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)

    weak = update_tissue_weights(observed, expected, wp_reg=1.0, n_tissue=kb)
    spm = update_tissue_weights(observed, expected, wp_reg=100.0, n_tissue=kb)

    assert weak.max() > 0.85  # weak reg → the over-observed tissue runs away toward 1
    assert spm.max() < 0.5  # SPM reg → stays bounded near uniform
    assert spm.min() > 0.25  # and no tissue collapses
    # larger wp_reg pulls every weight monotonically closer to uniform (1/Kb)
    uniform = 1.0 / kb
    assert (spm - uniform).abs().max() < (weak - uniform).abs().max()
    assert torch.allclose(weak.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_fit_segment_zero_reg_warp_runs():
    """A zero deformation penalty (reg all-zero) must not crash the warp backward.

    warp_penalty returns a non-grad constant when every reg weight is 0; the fit
    guards the separate ``.backward()`` on it. Regression for that guard.
    """
    torch.manual_seed(0)
    n = 20
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))
    fit = fit_segment(
        vol, eye, log_prior, eye, bg, bg, eye,
        ngaus=[1, 1, 1], biasfwhm=12.0, samp=1.0, n_iter=4,
        fit_warp=True, reg=(0.0, 0.0, 0.0), warp_smooth=0.5, verbose=False,
    )  # fmt: skip
    assert torch.isfinite(fit["twarp"]).all()
    assert fit["n_chunks"] >= 1 and fit["fit_chunk"] >= 1


def test_debridge_removes_thin_gm_bridge_keeps_thick_cortex():
    """A morphological opening deletes a 1-voxel GM bridge but keeps a thick GM blob,
    reassigning the removed probability to the other tissues."""
    from fastfuncstuff.processing.segment import debridge_gm

    n = 24
    post = torch.zeros(4, n, n, n)
    post[3] = 1.0  # background everywhere

    def set_gm(sl):  # GM with a little residual CSF so renorm has somewhere to go
        post[0][sl], post[2][sl], post[3][sl] = 0.8, 0.2, 0.0

    set_gm((slice(4, 14), slice(4, 14), slice(4, 14)))  # thick GM blob
    set_gm((8, 8, slice(14, 20)))  # thin (1-voxel) GM bridge sticking out

    out = debridge_gm(post, radius=1)
    assert out[0, 9, 9, 9] > 0.7  # blob interior GM kept
    assert out[0, 8, 8, 17] < 0.01  # thin bridge GM removed
    assert out[2, 8, 8, 17] > 0.9  # its probability reassigned to CSF
    assert torch.allclose(out.sum(0), torch.ones(n, n, n), atol=1e-4)  # still a valid posterior


def test_dura_cleanup_removes_moat_separated_gm_keeps_cortex():
    """WM-geodesic dura removal: a GM patch separated from the WM+cortex by a CSF moat is
    demoted (the front can't cross CSF), while GM tissue-connected to WM is kept."""
    from fastfuncstuff.processing.segment import dura_cleanup

    n = 32
    post = torch.zeros(4, n, n, n)
    post[3] = 1.0  # background everywhere (class 4)

    def fill(idx, sl, val=0.9):  # remainder in background so rows sum to 1 (renorm has room)
        post[idx][sl], post[3][sl] = val, 1.0 - val

    # WM slab + a GM "cortex" band directly abutting it (tissue-connected)
    fill(1, (slice(8, 16), slice(8, 24), slice(8, 24)))  # WM
    fill(0, (slice(16, 20), slice(8, 24), slice(8, 24)))  # cortex GM on the WM face
    # a CSF moat, then a detached GM "dura" sheet beyond it (must cross CSF to reach)
    fill(2, (slice(20, 23), slice(8, 24), slice(8, 24)))  # CSF moat
    fill(0, (slice(23, 25), slice(8, 24), slice(8, 24)))  # dura GM beyond the moat

    out = dura_cleanup(post, vox=(1.0, 1.0, 1.0), max_thick_mm=6.0)
    assert out[0, 17, 15, 15] > 0.8  # cortex (abuts WM) kept
    assert out[0, 24, 15, 15] < 0.05  # dura (beyond CSF moat) demoted
    assert torch.allclose(out.sum(0), torch.ones(n, n, n), atol=1e-4)  # valid posterior
    assert torch.allclose(dura_cleanup(post, (1.0, 1.0, 1.0), max_thick_mm=0.0), post)  # 0 disables


def test_dura_cleanup_csf_gap_fills_sheet_hole():
    """csf_gap method: an outer-shell GM voxel flanked by high CSF on both sides (a hole in the
    CSF sheet) is reassigned to CSF; CSF present on only one side (a concavity) is spared."""
    from fastfuncstuff.processing.segment import dura_cleanup

    n = 40
    post = torch.zeros(4, n, n, n)
    post[3] = 1.0

    def fill(idx, sl, val=0.9):
        post[idx][sl], post[3][sl] = val, 1.0 - val

    fill(1, (slice(4, 10), slice(10, 30), slice(10, 30)))  # WM slab (near edge)
    fill(0, (slice(10, 13), slice(10, 30), slice(10, 30)))  # cortex GM on the WM face
    # OUTER shell (far from WM, z≈30): a CSF sheet with a 2-voxel GM hole in the middle
    fill(2, (slice(28, 34), slice(10, 30), slice(10, 20)), 0.8)  # CSF, one side of the gap
    fill(0, (slice(28, 34), slice(10, 30), slice(20, 22)), 0.9)  # GM hole (the dura) in the sheet
    fill(2, (slice(28, 34), slice(10, 30), slice(22, 30)), 0.8)  # CSF, other side of the gap

    out = dura_cleanup(post, (1.0, 1.0, 1.0), max_thick_mm=6.0, method="csf_gap")
    assert out[0, 31, 20, 21] < 0.05  # GM hole flanked by CSF on both sides → reassigned
    assert out[2, 31, 20, 21] > 0.85  # ...to CSF (the sheet is closed)
    assert out[0, 11, 20, 20] > 0.8  # cortex near WM untouched (outer-shell restriction)
    assert torch.allclose(out.sum(0), torch.ones(n, n, n), atol=1e-4)


def test_dura_cleanup_csf_gap_fills_hole_in_thin_low_prob_sheet():
    """The subarachnoid sheet wrapping the dura is thin/partial-volumed — CSF only ~0.1, well
    below a confident (>0.4) threshold. The light-threshold search must still see the sheet and
    fill the dura hole (very-low-prob CSF); a >0.4 sheet test would find nothing to flank."""
    from fastfuncstuff.processing.segment import dura_cleanup

    n = 40
    post = torch.zeros(4, n, n, n)
    post[3] = 1.0

    def fill(idx, sl, val):
        post[idx][sl], post[3][sl] = val, 1.0 - val

    fill(1, (slice(4, 10), slice(10, 30), slice(10, 30)), 0.9)  # WM slab (near edge)
    fill(0, (slice(10, 13), slice(10, 30), slice(10, 30)), 0.9)  # cortex GM on the WM face
    # OUTER shell (z≈30): a THIN low-probability CSF sheet (0.1) with a very-low-CSF GM hole
    fill(2, (slice(28, 34), slice(10, 30), slice(10, 20)), 0.1)  # thin CSF, one side of the gap
    fill(0, (slice(28, 34), slice(10, 30), slice(20, 22)), 0.9)  # GM dura hole (CSF ≈ 0.01)
    fill(2, (slice(28, 34), slice(10, 30), slice(22, 30)), 0.1)  # thin CSF, other side of the gap

    out = dura_cleanup(post, (1.0, 1.0, 1.0), max_thick_mm=6.0, method="csf_gap")
    assert out[0, 31, 20, 21] < 0.05  # dura hole filled despite the flanking sheet being only 0.1
    assert out[2, 31, 20, 21] > 0.5  # ...reassigned to CSF
    # a confident (>0.4) sheet threshold would see no sheet here and leave the hole:
    out_hi = dura_cleanup(
        post, (1.0, 1.0, 1.0), max_thick_mm=6.0, method="csf_gap", csf_present=0.4
    )
    assert out_hi[0, 31, 20, 21] > 0.5  # unfilled — the regression this light threshold fixes


def test_segment_apply_precleanup_returned():
    """save_precleanup returns the posteriors before the mrf/debridge/cleanup passes."""
    from fastfuncstuff.processing.segment import segment_apply

    torch.manual_seed(0)
    n = 16
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))
    fit = fit_segment(
        vol, eye, log_prior, eye, bg, bg, eye, ngaus=[1, 1, 1], samp=1.0, n_iter=3, verbose=False
    )
    out = segment_apply(
        vol, log_prior, bg, bg, fit, mrf=1.0, cleanup=0, save_precleanup=True, verbose=False
    )
    assert "posteriors_precleanup" in out
    assert out["posteriors_precleanup"].shape == out["posteriors"].shape
    # without save_precleanup the key is absent
    out2 = segment_apply(vol, log_prior, bg, bg, fit, mrf=1.0, cleanup=0, verbose=False)
    assert "posteriors_precleanup" not in out2


def test_fit_segment_multichannel_recovers_tissues():
    """Two aligned channels with different contrast → joint GMM segments correctly."""
    from fastfuncstuff.processing.segment import segment_apply

    torch.manual_seed(0)
    n = 24
    vol, tissue, _, _ = _phantom_three_tissue(n)
    means2 = torch.tensor([40.0, 20.0, 120.0])  # channel 2: different tissue ordering
    vol2 = sum(means2[k] * tissue[k] for k in range(3))
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))

    fit = fit_segment(
        [vol, vol2], eye, log_prior, eye, bg, bg, eye,
        ngaus=[1, 1, 1], samp=1.0, n_iter=10, fit_warp=False, verbose=False,
    )  # fmt: skip
    assert fit["n_chan"] == 2
    assert fit["means"].shape == (2, 3)  # (n_chan, n_gauss)
    assert fit["covs"].shape == (2, 2, 3)  # joint n_chan × n_chan covariances
    assert fit["coef"].shape[0] == 2  # one bias field per channel

    out = segment_apply([vol, vol2], log_prior, bg, bg, fit, mrf=0, cleanup=0, verbose=False)
    assert out["corrected"].shape == (2, n, n, n)  # per-channel bias-corrected
    assert out["bias"].shape == (2, n, n, n)
    mask = tissue.sum(0) > 0
    agree = (out["posteriors"].argmax(0)[mask] == tissue.argmax(0)[mask]).float().mean()
    assert agree > 0.95


def test_fit_segment_single_in_list_equals_bare():
    """Passing [vol] is identical to passing vol (single-channel back-compat)."""
    torch.manual_seed(0)
    n = 20
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))
    kw = dict(ngaus=[1, 1, 1], samp=1.0, n_iter=4, fit_warp=False, verbose=False)  # noqa: C408
    bare = fit_segment(vol, eye, log_prior, eye, bg, bg, eye, **kw)
    listed = fit_segment([vol], eye, log_prior, eye, bg, bg, eye, **kw)
    assert torch.allclose(bare["means"], listed["means"], atol=1e-5)
    assert bare["coef"].shape == listed["coef"].shape == (1, *bare["bias_shape"])


@pytest.mark.gpu
def test_warp_aggressiveness_knobs_increase_displacement():
    """warp_lr / warp_iters up and reg down all make the deformation move more."""
    torch.manual_seed(0)
    n = 24
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))
    vol_shift = torch.roll(vol, shifts=3, dims=1)  # offset data vs prior → warp must move
    # anneal off + no bending in the base so the knob effects are isolated from SPM's
    # heavy-to-light schedule (which, over only 6 iters, would dominate the comparison).
    base = dict(  # noqa: C408
        ngaus=[1, 1, 1], biasfwhm=12.0, samp=1.0, n_iter=6, warp_solver="adam",
        fit_warp=True, warp_anneal=False, reg=(0.0, 0.0, 0.0), warp_smooth=0.5, verbose=False,
    )  # fmt: skip

    def mag(**kw):
        return (
            fit_segment(vol_shift, eye, log_prior, eye, bg, bg, eye, **{**base, **kw})["twarp"]
            .abs()
            .max()
        )

    assert mag(warp_lr=3.0) > mag(warp_lr=0.5)  # bigger Adam step → more warp
    assert mag(warp_iters=20) > mag(warp_iters=4)  # more steps → more warp
    assert mag(reg=(0.0, 0.0, 0.0)) > mag(reg=(0.0, 0.0, 0.5))  # less penalty → more warp


def test_autobox_roundtrip_preserves_geometry():
    """autobox crop + affine-shift + embed round-trips exactly: world coordinates of the
    kept voxels are unchanged, and embedding restores the original volume."""
    n = 20
    vol = torch.zeros(n, n, n)
    torch.manual_seed(0)
    vol[5:15, 6:16, 7:17] = 1.0 + torch.rand(10, 10, 10)  # a nonzero blob, offset from origin
    affine = torch.tensor(
        [[2.0, 0, 0, -10], [0, 2.0, 0, -12], [0, 0, 2.0, -14], [0, 0, 0, 1]], dtype=torch.float64
    )

    slices, offset = autobox_bounds(vol, pad=2)
    sz, sy, sx = slices
    cropped = vol[slices]
    caffine = crop_affine(affine, offset)

    # a known original voxel (array z,y,x = 5,6,7 → affine voxel x,y,z = 7,6,5) keeps its
    # world coordinate after the crop-affine shift
    w_full = affine @ torch.tensor([7.0, 6.0, 5.0, 1.0], dtype=torch.float64)
    cx, cy, cz = 7 - sx.start, 6 - sy.start, 5 - sz.start  # same voxel in the cropped grid
    w_crop = caffine @ torch.tensor([cx, cy, cz, 1.0], dtype=torch.float64)
    assert torch.allclose(w_full, w_crop)

    # embed restores the original (outside-box was zero); leading axes are preserved
    assert torch.allclose(embed_in_full(cropped, (n, n, n), slices), vol)
    stack = torch.stack([cropped, 2 * cropped])  # (2, nz', ny', nx')
    emb = embed_in_full(stack, (n, n, n), slices)
    assert emb.shape == (2, n, n, n)
    assert torch.allclose(emb[0], vol) and torch.allclose(emb[1], 2 * vol)
    # a bias-style fill: the margin takes the fill value, the box the data
    embf = embed_in_full(cropped, (n, n, n), slices, fill=1.0)
    assert embf[0, 0, 0] == 1.0 and torch.allclose(embf[slices], cropped)


def test_fudge_factor_matches_spm_formula():
    """`fudge_factor` reproduces SPM's ff = prod(4π(s/vx/sk)²+1)^½, s=(fwhm+mean vx)/√8ln2."""
    for vox, sk, fwhm in [((1.0, 1.0, 1.0), (3, 3, 3), 0.0), ((0.7, 0.7, 2.0), (4, 4, 1), 5.0)]:
        s = (fwhm + sum(vox) / 3.0) / math.sqrt(8.0 * math.log(2.0))
        expect = 1.0
        for vx, k in zip(vox, sk, strict=True):
            expect *= 4.0 * math.pi * (s / (vx * k)) ** 2 + 1.0
        expect = expect**0.5
        assert abs(fudge_factor(vox, sk, fwhm) - expect) < 1e-10
    # ff > 1 even at fwhm=0 (the mean(vx) term); denser sampling (smaller stride) means
    # more correlated neighbours → larger ff (coarser sampling tends toward independent → 1)
    assert fudge_factor((1.0, 1.0, 1.0), (3, 3, 3), 0.0) > 1.0
    assert fudge_factor((1.0, 1.0, 1.0), (1, 1, 1), 0.0) > fudge_factor(
        (1.0, 1.0, 1.0), (3, 3, 3), 0.0
    )


def test_warp_penalty_linear_elastic_and_tuple_compat():
    """reg[3]/reg[4] are SPM's ``mu`` (shear) and ``lambda`` (divergence), in that order.

    Per ``spm_diffeo.m``, param[7] (= reg[3], default 0.01) penalises the *symmetrised
    Jacobian* and param[8] (= reg[4], default 0.04) penalises the *divergence*. They were
    swapped here until 2026-07-22. A pure-divergence field cannot detect the swap (it has
    non-zero ``ε_ii`` too), so the discriminating case is a **shear field with zero
    divergence**.
    """
    torch.manual_seed(0)
    field = torch.randn(6, 6, 6, 3, dtype=torch.float64)
    vox = (1.0, 1.0, 1.0)
    p3 = warp_penalty(field, (0.0, 0.0, 0.1), vox)
    p5_zero = warp_penalty(field, (0.0, 0.0, 0.1, 0.0, 0.0), vox)
    assert torch.allclose(p3, p5_zero)  # 3-tuple back-compat = 5-tuple with le=0

    zz, yy, xx = torch.meshgrid(*[torch.arange(6, dtype=torch.float64)] * 3, indexing="ij")
    zero = torch.zeros_like(xx)
    # u = (y, 0, 0): ∂u_x/∂y = 1, every ∂u_i/∂x_i = 0 → pure shear, divergence-free
    shear = torch.stack([yy, zero, zero], dim=-1)
    assert warp_penalty(shear, (0.0, 0.0, 0.0, 1.0, 0.0), vox) > 0  # reg[3] = mu sees shear
    assert warp_penalty(shear, (0.0, 0.0, 0.0, 0.0, 1.0), vox) == 0  # reg[4] = lambda does not
    # and a pure-dilation field u = (x, y, z) is seen by both (divergence 3, ε_ii = 1)
    div_field = torch.stack([xx, yy, zz], dim=-1)
    assert warp_penalty(div_field, (0.0, 0.0, 0.0, 0.0, 1.0), vox) > 0
    assert warp_penalty(div_field, (0.0, 0.0, 0.0, 1.0, 0.0), vox) > 0
    # le terms strictly increase the total penalty on a generic field
    assert warp_penalty(field, (0.0, 0.0, 0.1, 0.02, 0.03), vox) > p3


def test_vel2mom_impulse_response_matches_spm_stencil():
    """``warp_prior_grad`` must reproduce ``spm_diffeo('vel2mom')``'s stencil coefficients.

    This is the test that was missing. Every previous check on the regulariser was a
    *self-consistency* check — the operator matches the autograd gradient of the energy,
    bending annihilates a harmonic field, the coarse-``samp`` warp does not freeze — and
    all of them passed while the operator was off from SPM's by a factor of 729 at
    ``samp 3`` / 1 mm. Nothing compared an absolute magnitude to the reference.

    Two things fixed on 2026-07-29 and pinned here (``src/shoot_regularisers.c:601``):

    - ``v_i = param_i²`` as a **multiplier** (SPM's ``v0 = s[0]*s[0]``), where the old code
      divided by the node spacing (``1/h²``).
    - ``vel2mom`` **always** divides the absolute/membrane/bending part of component ``c``
      by ``v_c``; only the ``kernel()`` helper has a ``mu==0 && lam==0`` shortcut.

    Read off an impulse response, which is the stencil by definition. Anisotropic ``param``
    so an axis mix-up cannot cancel.
    """
    from fastfuncstuff.processing.segment import warp_prior_grad

    param = (3.0, 2.0, 4.0)  # node spacing (SPM's sk.*vx), deliberately anisotropic
    lam0, lam1, lam2 = 0.011, 0.022, 0.1
    reg = (lam0, lam1, lam2, 0.0, 0.0)  # mu = lam = 0 → no cross-component coupling
    v0, v1, v2 = (p * p for p in param)
    tot = v0 + v1 + v2
    # SPM's scalars, transcribed
    w000 = (
        lam2 * (6.0 * (v0 * v0 + v1 * v1 + v2 * v2) + 8.0 * (v0 * v1 + v0 * v2 + v1 * v2))
        + lam1 * 2.0 * tot
        + lam0
    )
    w100 = lam2 * (-4.0 * v0 * tot) - lam1 * v0
    w010 = lam2 * (-4.0 * v1 * tot) - lam1 * v1
    w001 = lam2 * (-4.0 * v2 * tot) - lam1 * v2
    w200, w020, w002 = lam2 * v0 * v0, lam2 * v1 * v1, lam2 * v2 * v2
    w110, w101, w011 = lam2 * 2.0 * v0 * v1, lam2 * 2.0 * v0 * v2, lam2 * 2.0 * v1 * v2

    n = 13  # wide enough that the ±2 taps never reach the boundary
    c = n // 2
    for comp, vc in enumerate((v0, v1, v2)):
        field = torch.zeros(n, n, n, 3, dtype=torch.float64)
        field[c, c, c, comp] = 1.0
        g = warp_prior_grad(field, reg, param)[..., comp]

        # (dx, dy, dz) offset in SPM's (i, j, k) → array index (z, y, x)
        def at(dx, dy, dz, _g=g):
            return _g[c + dz, c + dy, c + dx].item()

        assert at(0, 0, 0) == pytest.approx(w000 / vc, rel=1e-12)
        assert at(1, 0, 0) == pytest.approx(w100 / vc, rel=1e-12)
        assert at(0, 1, 0) == pytest.approx(w010 / vc, rel=1e-12)
        assert at(0, 0, 1) == pytest.approx(w001 / vc, rel=1e-12)
        assert at(2, 0, 0) == pytest.approx(w200 / vc, rel=1e-12)
        assert at(0, 2, 0) == pytest.approx(w020 / vc, rel=1e-12)
        assert at(0, 0, 2) == pytest.approx(w002 / vc, rel=1e-12)
        assert at(1, 1, 0) == pytest.approx(w110 / vc, rel=1e-12)
        assert at(1, 0, 1) == pytest.approx(w101 / vc, rel=1e-12)
        assert at(0, 1, 1) == pytest.approx(w011 / vc, rel=1e-12)

    # the elastic terms couple components: mu/lam put w2 = (mu+lam)/4 on the mixed taps
    mu, lam = 0.013, 0.041
    field = torch.zeros(n, n, n, 3, dtype=torch.float64)
    field[c, c, c, 1] = 1.0  # a y-displacement impulse
    gx = warp_prior_grad(field, (0.0, 0.0, 0.0, mu, lam), param)[..., 0]
    w2 = 0.25 * (mu + lam)
    assert gx[c, c - 1, c + 1].item() == pytest.approx(w2, rel=1e-12)  # (dx,dy) = (+1,-1)
    assert gx[c, c + 1, c + 1].item() == pytest.approx(-w2, rel=1e-12)  # (+1,+1)

    # and the per-component diagonal helper agrees with SPM's wx000/wy000/wz000
    from fastfuncstuff.processing.segment import warp_prior_diagonal

    diag = warp_prior_diagonal((lam0, lam1, lam2, mu, lam), param, (n, n, n))
    w000_le = w000  # lam0/lam1/lam2 part is unchanged by mu/lam
    expected = (
        2.0 * mu * (2.0 * v0 + v1 + v2) / v0 + 2.0 * lam + w000_le / v0,
        2.0 * mu * (v0 + 2.0 * v1 + v2) / v1 + 2.0 * lam + w000_le / v1,
        2.0 * mu * (v0 + v1 + 2.0 * v2) / v2 + 2.0 * lam + w000_le / v2,
    )
    assert diag == pytest.approx(expected, rel=1e-12)


def test_warp_prior_symbol_diagonalises_the_operator():
    """The DCT-II symbol must equal ``L`` on its diagonal blocks.

    The Gauss-Newton warp solve depends on this: ``L`` is stiff and ill-conditioned in
    exactly the smooth modes the prior exists to produce, so CG needs the
    ``(shift·I + L)⁻¹`` preconditioner to converge in a sane number of iterations, and that
    preconditioner is only valid if the symbol really diagonalises the operator. A wrong
    symbol degrades silently into slow convergence — i.e. back into the spiky warp this
    was fixed to avoid — so pin it. ``mu = lam = 0`` because the cross-component coupling
    is deliberately excluded from the symbol (it maps cosine modes to sine modes).
    """
    from fastfuncstuff.processing.segment import (
        _dct3,
        _dct_matrix,
        warp_prior_grad,
        warp_prior_symbol,
    )

    torch.manual_seed(0)
    shape = (9, 11, 13)
    field = torch.randn(*shape, 3, dtype=torch.float64)
    for reg, param in (
        ((0.0, 0.0, 0.1, 0.0, 0.0), (3.0, 3.0, 3.0)),
        ((0.01, 0.02, 0.1, 0.0, 0.0), (3.0, 2.0, 4.0)),
        ((0.0, 0.5, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    ):
        mats = tuple(_dct_matrix(n, torch.device("cpu"), torch.float64) for n in shape)
        x = field.permute(3, 0, 1, 2).contiguous()
        assert torch.allclose(_dct3(_dct3(x, mats, inverse=False), mats, inverse=True), x)
        sym = warp_prior_symbol(reg, param, shape, dtype=torch.float64)
        got = _dct3(_dct3(x, mats, inverse=False) * sym, mats, inverse=True).permute(1, 2, 3, 0)
        expected = warp_prior_grad(field, reg, param)
        assert torch.allclose(got, expected, atol=1e-9, rtol=1e-7), (
            f"reg={reg} param={param}: max |Δ| = {(got - expected).abs().max():.3e}"
        )


def test_warp_penalty_is_half_the_quadratic_form():
    """SPM's warp prior is ``½·uᵀLu`` (``llr = -0.5*sum(Twarp.*vel2mom(Twarp))``).

    ``warp_penalty`` must return that half, so that its autograd gradient is ``Lu`` — the
    quantity SPM adds to ``Beta``. Checked via Euler's theorem: for a quadratic form,
    ``u·∂P/∂u = 2P``.
    """
    torch.manual_seed(0)
    field = torch.randn(5, 5, 5, 3, dtype=torch.float64, requires_grad=True)
    vox = (1.5, 1.5, 1.5)
    pen = warp_penalty(field, (0.01, 0.02, 0.1, 0.01, 0.04), vox)
    (grad,) = torch.autograd.grad(pen, field)
    assert torch.isclose((field * grad).sum(), 2.0 * pen)
    # and the half itself: doubling the field quadruples the energy, from a known scale.
    # SPM's absolute term enters as `lam0*c` inside the group divided by v_c = param_c²
    # (`shoot_regularisers.c:729`), so the energy is ½·lam0·Σu²/v — isotropic `vox` here,
    # so one factor covers all three components.
    field2 = field.detach() * 2.0
    v = vox[0] ** 2
    assert torch.isclose(
        warp_penalty(field2, (1.0, 0.0, 0.0), vox), 4.0 * 0.5 * (field**2).sum() / v
    )


@pytest.mark.parametrize(
    "reg",
    [
        (0.0, 0.0, 0.1, 0.01, 0.04),  # SPM default
        (1e-3, 0.0, 0.1, 0.01, 0.04),  # the GN variant (absolute floor for CG's SPD-ness)
        (0.5, 0.3, 0.0, 0.0, 0.0),  # abs + membrane only
        (0.0, 0.0, 0.0, 0.7, 0.0),  # shear only
        (0.0, 0.0, 0.0, 0.0, 0.7),  # divergence only
        (0.0, 0.0, 1.0, 0.0, 0.0),  # bending only
    ],
)
def test_warp_prior_grad_matches_autograd(reg):
    """The analytic ``L·u`` must equal ``∂/∂u warp_penalty(u)`` to round-off.

    ``warp_prior_grad`` replaced an autograd pass because the Gauss-Newton solver applies
    ``L`` eleven times per warp sub-iteration and the autograd route was the single largest
    contributor to a launch-bound EM loop. It is only safe if it is *exactly* the same
    operator — a silent mismatch would change the Newton direction without changing any
    result an existing test checks.
    """
    from fastfuncstuff.processing.segment import warp_prior_grad

    torch.manual_seed(0)
    field = torch.randn(7, 6, 5, 3, dtype=torch.float64)
    vox = (1.3, 0.9, 2.1)  # anisotropic, so an axis/spacing mix-up cannot cancel
    v = field.clone().requires_grad_(True)
    (expected,) = torch.autograd.grad(warp_penalty(v, reg, vox), v)
    got = warp_prior_grad(field, reg, vox)
    assert torch.allclose(got, expected, atol=1e-10, rtol=1e-8), (
        f"max |Δ| = {(got - expected).abs().max():.3e}"
    )


def test_warp_prior_grad_is_a_symmetric_linear_operator():
    """``L`` must be linear and self-adjoint — the conjugate-gradient solve assumes both."""
    from fastfuncstuff.processing.segment import warp_prior_grad

    torch.manual_seed(1)
    reg, vox = (1e-3, 0.05, 0.1, 0.01, 0.04), (1.0, 1.4, 2.2)
    a = torch.randn(5, 6, 4, 3, dtype=torch.float64)
    b = torch.randn(5, 6, 4, 3, dtype=torch.float64)
    la, lb = warp_prior_grad(a, reg, vox), warp_prior_grad(b, reg, vox)
    # linearity
    assert torch.allclose(warp_prior_grad(2.0 * a + 3.0 * b, reg, vox), 2.0 * la + 3.0 * lb)
    # self-adjointness: <a, Lb> == <La, b>
    assert torch.isclose((a * lb).sum(), (la * b).sum())
    # positive semi-definite (CG needs it), and the energy identity ½·uᵀLu = P(u)
    assert (a * la).sum() > 0
    assert torch.isclose(0.5 * (a * la).sum(), warp_penalty(a, reg, vox))


def test_bspline3_is_an_interpolating_spline():
    """The deformation upsampler must pass exactly through the fitted nodes.

    SPM expands ``Twarp`` with ``spm_bsplinc(...,[3 3 3 0 0 0])`` + a matching degree-3
    sample (``spm_preproc_write8.m:133-136``). Unlike the degree-2 TPM kernel — which is
    deliberately approximating because ``spm_bsplinc`` at degree 1 is a no-op — degree 3
    runs a real prefilter, so the result **interpolates**. At ``samp 3`` on 0.7 mm data
    this is a 4x upsample of the field that positions every tissue prior, so getting it
    wrong facets the deformation on the coarse lattice.
    """
    from fastfuncstuff.processing.segment import (
        bspline3_coefficients,
        bspline_interpolate_multi,
    )

    torch.manual_seed(0)
    v = torch.randn(3, 9, 8, 7, dtype=torch.float64)  # a 3-component displacement field
    coeffs = bspline3_coefficients(v)
    zz, yy, xx = torch.meshgrid(
        *[torch.arange(n, dtype=torch.float64) for n in (9, 8, 7)], indexing="ij"
    )
    got = bspline_interpolate_multi(
        coeffs, xx.reshape(-1), yy.reshape(-1), zz.reshape(-1), degree=3
    )
    # exact at every node INCLUDING the edges (mirror taps match the prefilter's boundary)
    assert (got.T.reshape(3, 9, 8, 7) - v).abs().max() < 1e-12

    one = lambda a: torch.tensor([a], dtype=torch.float64)  # noqa: E731
    ones = torch.ones(1, 9, 8, 7, dtype=torch.float64)
    mid = bspline_interpolate_multi(
        bspline3_coefficients(ones), one(2.3), one(3.7), one(4.5), degree=3
    )
    assert torch.isclose(mid, torch.ones_like(mid))  # partition of unity off-node
    # and a linear ramp is reproduced (degree-3 B-splines are exact up to cubic)
    ramp = bspline_interpolate_multi(
        bspline3_coefficients(xx[None].clone()), one(3.25), one(3.0), one(4.0), degree=3
    )
    assert abs(ramp.item() - 3.25) < 0.02


def test_load_tpm_adds_background_only_when_the_template_is_incomplete():
    """A limited-coverage TPM needs a class for 'everything else'; a complete one does not.

    A generative mixture has to explain every voxel. GM/WM/CSF-only priors leave air,
    skull and scalp with nowhere to go, so the model shoehorns them into a tissue class.
    SPM's own ``TPM.nii`` does not have this problem — its six classes sum to 1.0 — so the
    auto rule must be a no-op there.
    """
    import nibabel as nib

    from fastfuncstuff.processing.segment import load_tpm

    def _write(path, prob):  # prob: (K, nz, ny, nx) in [0,1]
        arr = np.ascontiguousarray(np.transpose(prob, (3, 2, 1, 0)))  # (nx,ny,nz,K)
        nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(path))

    n = 6
    rng = np.random.default_rng(0)
    brain = rng.random((3, n, n, n)) * 0.3  # sums to ~0.45 → 55% unexplained
    with tempfile.TemporaryDirectory() as td:
        partial = pathlib.Path(td) / "partial.nii"
        _write(partial, brain)
        lp, _, _, _, added = load_tpm(str(partial), verbose=False)
        assert added and lp.shape[0] == 4
        p = torch.exp(lp) - 1e-4
        assert torch.allclose(p.sum(0), torch.ones_like(p.sum(0)), atol=2e-4)  # now complete

        # a template that already sums to 1 gets nothing appended
        complete = np.concatenate([brain, (1.0 - brain.sum(0))[None]], axis=0)
        full = pathlib.Path(td) / "full.nii"
        _write(full, complete)
        lp2, _, _, _, added2 = load_tpm(str(full), verbose=False)
        assert not added2 and lp2.shape[0] == 4
        # and forcing it off is honoured
        _, _, _, _, added3 = load_tpm(str(partial), add_background="no", verbose=False)
        assert not added3


def test_tpm_coverage_mask_rejects_below_and_outside():
    """The field-of-view rules: SPM's inferior-only cut, and the full-box extension."""
    from fastfuncstuff.processing.segment import tpm_coverage_mask

    shape, vox = (40, 50, 60), (1.5, 1.5, 1.5)  # (nz, ny, nx), 1.5 mm iso
    pts = torch.tensor(
        [
            [30.0, 25.0, 20.0],  # well inside
            [30.0, 25.0, 2.0],  # z = 2 < 5/1.5 - 1 = 2.33 → below the template floor
            [30.0, 25.0, -40.0],  # far below (the neck)
            [200.0, 25.0, 20.0],  # far outside in +x, but at a legal height
        ],
        dtype=torch.float64,
    )
    spm_rule = tpm_coverage_mask(pts, shape, vox, bottom_mm=5.0)
    assert spm_rule.tolist() == [True, False, False, True]  # +x escapee survives
    both = tpm_coverage_mask(pts, shape, vox, bottom_mm=5.0, fov_mm=0.0)
    assert both.tolist() == [True, False, False, False]  # full box catches it
    assert tpm_coverage_mask(pts, shape, vox, bottom_mm=0.0).all()  # rules disabled


def test_clean_gwc_accepts_custom_and_multiple_class_roles():
    """Cleanup must not be hardwired to SPM's c1/c2/c3 ordering.

    Same phantom in two layouts — SPM order, and a permuted template with the GM mass
    split across two classes — must give the same brain extraction.
    """
    from fastfuncstuff.processing.segment import clean_gwc

    n = 24
    zz, yy, xx = torch.meshgrid(*[torch.arange(n, dtype=torch.float32)] * 3, indexing="ij")
    r = ((xx - n / 2) ** 2 + (yy - n / 2) ** 2 + (zz - n / 2) ** 2).sqrt()
    gm, wm, csf = ((r >= 4) & (r < 7)).float(), (r < 4).float(), ((r >= 7) & (r < 9)).float()
    blob = ((xx - 2) ** 2 + (yy - 2) ** 2 + (zz - 2) ** 2 < 4).float()  # detached "dura"
    other = (1.0 - gm - wm - csf - blob).clamp_min(0.0)

    spm_order = torch.stack([gm + blob, wm, csf, other])
    spm_order = spm_order / spm_order.sum(0, keepdim=True).clamp_min(1e-6)
    a = clean_gwc(spm_order, level=1)

    # permuted: WM=0, CSF=1, other=2, GM split across 3 and 4
    perm = torch.stack([wm, csf, other, gm * 0.5 + blob, gm * 0.5])
    perm = perm / perm.sum(0, keepdim=True).clamp_min(1e-6)
    b = clean_gwc(perm, level=1, gm_class=(3, 4), wm_class=(0,), csf_class=(1,))

    assert a[0][blob > 0].max() < 1e-3  # the detached blob is stripped in both layouts
    assert (b[3] + b[4])[blob > 0].max() < 1e-3
    assert torch.allclose(a[0][r < 6.5], (b[3] + b[4])[r < 6.5], atol=1e-4)  # cortex kept


def test_bspline2_is_spm_approximating_kernel():
    """``bspline2_interpolate_multi`` must be SPM's degree-2 B-spline on *unprefiltered*
    data — an approximating, not an interpolating, kernel.

    ``spm_load_priors8`` runs ``spm_bsplinc(..., deg=1)``, which is a no-op, then
    ``spm_sample_priors8`` samples with ``deg=2``. The signature of applying a degree-2
    basis to raw samples is exact reproduction of **linear** functions but a constant
    offset on quadratics: with taps at ``c±1`` and weights ``½(½∓d)², ¾-d²``,

        Σ w_k·(c+k)² = (c+d)² + ¼

    — the value is high by exactly ¼ everywhere, independent of ``d``. An interpolating
    kernel (trilinear, cubic Lagrange) would return ``x²`` on the nodes. That ``¼`` is the
    low-pass character the warp's Gauss-Newton gradient relies on.
    """
    from fastfuncstuff.processing.segment import bspline2_interpolate_multi

    n = 12
    zz, yy, xx = torch.meshgrid(*[torch.arange(n, dtype=torch.float64)] * 3, indexing="ij")
    ones = torch.ones_like(xx)
    vol = torch.stack([ones, xx, xx**2], dim=0)  # 3 "tissues": constant, linear, quadratic

    x = torch.tensor([3.0, 3.25, 3.5, 4.75, 6.0], dtype=torch.float64)
    y = torch.full_like(x, 5.0)
    z = torch.full_like(x, 5.0)
    got = bspline2_interpolate_multi(vol, x, y, z)

    assert torch.allclose(got[:, 0], torch.ones_like(x))  # partition of unity
    assert torch.allclose(got[:, 1], x)  # exact on linear
    assert torch.allclose(got[:, 2], x**2 + 0.25)  # +1/4 on quadratic → approximating

    # Continuous across the tap switch at a half-integer coordinate (where the stencil
    # shifts from {c-1,c,c+1} to {c,c+1,c+2}): the jump must shrink with the step, not sit
    # at a fixed cliff — the failure mode that stalled the Gauss-Newton line search.
    def _jump(eps: float) -> float:
        xs = torch.tensor([4.5 - eps, 4.5 + eps], dtype=torch.float64)
        v = bspline2_interpolate_multi(vol, xs, torch.full_like(xs, 5.0), torch.full_like(xs, 5.0))
        return (v[0] - v[1]).abs().max().item()

    assert _jump(1e-4) > 0.0
    assert _jump(1e-5) < 0.2 * _jump(1e-4)  # linear in the step ⇒ continuous


def test_tpm_bottom_cut_drops_samples_below_template():
    """SPM excludes samples the affine puts within 5 mm of the bottom of the TPM.

    ``spm_preproc8.m:233-237`` — on a whole-head T1 that is the neck and shoulders. Those
    voxels otherwise enter ``vr0``, the soft-tissue/bone/air Gaussians and the ``wp``
    masses. With an identity geometry and a 1 mm TPM the rule drops every plane with
    ``z <= 5/1 - 1 = 4``, i.e. z = 0..4.
    """
    torch.manual_seed(0)
    n = 16
    vol, tissue, _, _ = _phantom_three_tissue(n)
    vol = vol + 1.0  # no zero voxels, so the cut is the ONLY thing removing samples
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))
    base = dict(  # noqa: C408
        ngaus=[1, 1, 1], biasfwhm=12.0, samp=1.0, n_iter=1, n_inner=1,
        fit_warp=False, verbose=False,
    )  # fmt: skip

    kept_all = fit_segment(vol, eye, log_prior, eye, bg, bg, eye, tpm_bottom_mm=0.0, **base)
    kept_cut = fit_segment(vol, eye, log_prior, eye, bg, bg, eye, tpm_bottom_mm=5.0, **base)

    assert kept_all["n_samp"] == n**3
    assert kept_cut["n_samp"] == (n - 5) * n * n  # z = 0..4 removed


def test_split_gaussians_scales_jitter_by_per_tissue_covariance():
    """The ``ngaus`` split must use each tissue's OWN covariance, not a pooled one.

    SPM splits after the first [GMM, bias] round from the converged ``vr1(:,:,k1)``
    (``spm_preproc8.m:671``). Splitting at moment-init time — when every tissue still
    shares one pooled covariance — scatters a tight class as widely as a broad one, which
    is what lands the extra Gaussians of the multi-component classes (CSF x2, bone x3,
    soft x4) in the wrong places.
    """
    from fastfuncstuff.processing.segment import split_gaussians

    means1 = torch.tensor([[0.0, 0.0]], dtype=torch.float64)  # (n_chan=1, n_tissue=2)
    covs1 = torch.tensor([[[1.0, 100.0]]], dtype=torch.float64)  # tight vs broad tissue
    means, covs, mix, tissue_of = split_gaussians(means1, covs1, [4, 4])

    assert tissue_of.tolist() == [0] * 4 + [1] * 4
    assert torch.allclose(mix, torch.full((8,), 0.25, dtype=torch.float64))
    spread_tight = means[0, tissue_of == 0].std()
    spread_broad = means[0, tissue_of == 1].std()
    # jitter ∝ sqrt(vr_k), so the broad tissue's components must scatter ~10x wider
    assert spread_broad > 5.0 * spread_tight
    # component covariances scale with their own tissue too
    assert covs[0, 0, tissue_of == 1].min() > covs[0, 0, tissue_of == 0].max()


def test_split_is_deferred_but_still_applied_in_one_iteration():
    """``ngaus`` still takes effect even for a single outer iteration.

    The split moved from initialisation to the end of the first [GMM, bias] round, so a
    regression there would silently leave the fit with one Gaussian per tissue.
    """
    torch.manual_seed(0)
    n = 16
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))
    out = fit_segment(
        vol, eye, log_prior, eye, bg, bg, eye,
        ngaus=[2, 1, 3], biasfwhm=12.0, samp=1.0, n_iter=1, n_inner=1,
        fit_warp=False, verbose=False,
    )  # fmt: skip
    assert out["tissue_of"].tolist() == [0, 0, 1, 2, 2, 2]
    assert out["means"].shape[1] == 6
    assert out["covs"].shape[2] == 6
    assert out["mix"].numel() == 6


def test_split_after_waits_for_the_bias_before_expanding_gaussians():
    """``split_after`` must delay the expansion, and the delayed split must start from a
    **tighter** covariance than SPM's split-after-one-round schedule.

    The first ``[GMM, bias]`` round fits the GMM to the *uncorrected* image (``Tbias``
    starts at zero, the bias step comes after), so every tissue covariance is still
    inflated by the bias field. ``split_gaussians`` hands both children that inflated
    covariance, and on real data a too-broad pair bifurcates — one child contracts onto
    the tissue, the other expands onto the intensity tails and becomes an outlier-catcher.
    Measured on a reference T1 at ``samp 3``: split after 1 round starts from CSF sd 196
    and ends at means 284/1009 mixing 0.97/0.03; after 2 rounds it starts from sd 116 and
    ends at 226/340 mixing 0.41/0.59, against SPM's 224/349 at 0.43/0.57.

    Here the phantom carries a strong bias field so the same mechanism is visible: assert
    the split is genuinely deferred and that the covariance it starts from shrinks.
    """
    from fastfuncstuff.processing import segment as seg

    torch.manual_seed(0)
    n = 16
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))
    _, _, xx = torch.meshgrid(*[torch.arange(n, dtype=torch.float64)] * 3, indexing="ij")
    biased = vol * torch.exp(0.7 * torch.cos(math.pi * xx / (n - 1)))  # strong INU

    seen: dict[int, float] = {}
    real_split = seg.split_gaussians

    def spy(means1, covs1, ngaus, **kw):
        seen.setdefault(len(seen), float(covs1[0, 0, 0]))
        return real_split(means1, covs1, ngaus, **kw)

    widths = {}
    for after in (1, 4):
        seen.clear()
        seg.split_gaussians = spy
        try:
            out = fit_segment(
                biased, eye, log_prior, eye, bg, bg, eye,
                ngaus=[2, 1, 2], biasfwhm=12.0, samp=1.0, n_iter=2, n_inner=6,
                split_after=after, fit_warp=False, verbose=False,
            )  # fmt: skip
        finally:
            seg.split_gaussians = real_split
        assert len(seen) == 1, "split must happen exactly once"
        assert out["tissue_of"].tolist() == [0, 0, 1, 2, 2]  # ngaus still applied
        widths[after] = seen[0]

    assert widths[4] < widths[1], (
        f"deferring the split should start it from a tighter covariance, "
        f"got {widths[4]:.4g} (after 4) vs {widths[1]:.4g} (after 1)"
    )


@pytest.mark.gpu
def test_warp_reg_in_node_units_survives_a_coarser_samp():
    """The deformation prior lives in NODE units, so a coarser ``samp`` must not freeze it.

    SPM regularises ``Twarp./sk`` (``spm_preproc8.m:805,808``) with a data gradient in the
    same units. Penalising the displacement in *image-voxel* units instead — the behaviour
    here until 2026-07-22 — makes the prior ``sk²`` too strong, so the warp progressively
    under-deforms as ``samp`` coarsens (9x at ``sk=3``, 16x at ``sk=4``). Fit a known
    3-voxel shift at ``samp=1`` (``sk=1``, where the two conventions coincide) and at
    ``samp=2`` (``sk=2``, where they differ 4x) and require the coarse fit to keep most of
    the displacement.
    """
    torch.manual_seed(0)
    n = 24
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))
    vol_shift = torch.roll(vol, shifts=3, dims=1)
    base = dict(  # noqa: C408
        ngaus=[1, 1, 1], biasfwhm=12.0, n_iter=6, warp_iters=8, warp_anneal=False,
        tpm_bottom_mm=0.0, verbose=False,
    )  # fmt: skip

    fine = fit_segment(vol_shift, eye, log_prior, eye, bg, bg, eye, samp=1.0, **base)
    coarse = fit_segment(vol_shift, eye, log_prior, eye, bg, bg, eye, samp=2.0, **base)

    d_fine = fine["twarp"].abs().max().item()
    d_coarse = coarse["twarp"].abs().max().item()
    assert d_fine > 0.5, f"fine fit did not deform ({d_fine})"
    assert d_coarse > 0.5 * d_fine, f"coarse samp under-deformed: {d_coarse} vs {d_fine}"


@pytest.mark.gpu
def test_fit_recovers_known_bias_field():
    """The joint fit recovers a known smooth multiplicative bias — a regression guard for
    the change-of-variables Jacobian term (+Σ log|bias|) in the bias objective.

    Without that term the bias step can lower the negative log-likelihood by mis-scaling
    the field. The reported ``bias`` is the *correction* field ``exp(bias)`` applied to the
    observed data, so it is the inverse of the injected bias: the recovered log-field is
    strongly *anti*-correlated with the truth (up to a global offset the means absorb).
    """
    from fastfuncstuff.processing.segment import segment_apply

    torch.manual_seed(0)
    n = 24
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))
    _, _, xx = torch.meshgrid(*[torch.arange(n, dtype=torch.float64)] * 3, indexing="ij")
    true_logbias = 0.6 * torch.cos(math.pi * xx / (n - 1))  # smooth low-frequency shading
    biased = vol * torch.exp(true_logbias)

    fit = fit_segment(
        biased, eye, log_prior, eye, bg, bg, eye,
        ngaus=[1, 1, 1], biasfwhm=14.0, fit_warp=False, samp=1.0, n_iter=30, tol=0.0, verbose=False,
    )  # fmt: skip
    out = segment_apply(biased, log_prior, bg, bg, fit, mrf=0, cleanup=0, verbose=False)
    est = torch.log(out["bias"].to(torch.float64).clamp_min(1e-6))
    a = (est - est.mean()).reshape(-1)
    b = (true_logbias - true_logbias.mean()).reshape(-1)
    corr = float(a @ b / (a.norm() * b.norm()))
    # default analytic-GN bias recovers the field almost exactly (Adam plateaus near -0.54)
    assert corr < -0.85  # correction field is the (near-exact) inverse of the injected shading


@pytest.mark.gpu
def test_gauss_newton_warp_solver_deforms_and_improves():
    """The GN warp solver moves the field and raises the data log-likelihood.

    Regression guard for the boundary discontinuity that stalled GN at zero
    displacement: the hard in/out TPM substitution cliffed the objective at
    voxels sitting on the volume edge, so the global Armijo line search rejected
    every step. sample_tpm_prior now ramps the background over one voxel, keeping
    the (differentiable) linear path continuous.
    """
    torch.manual_seed(0)
    n = 24
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))
    vol_shift = torch.roll(vol, shifts=3, dims=1)  # offset data vs prior → warp must move
    # anneal off so the GN solver runs at the target reg from iter 1 (the heavy-to-light
    # schedule assumes SPM's ~30 iters; over 6 it would suppress the legitimate 3-vox warp).
    base = dict(  # noqa: C408
        ngaus=[1, 1, 1], biasfwhm=12.0, samp=1.0, n_iter=6, warp_iters=8,
        warp_anneal=False, verbose=False,
    )  # fmt: skip

    no_warp = fit_segment(vol_shift, eye, log_prior, eye, bg, bg, eye, fit_warp=False, **base)
    gn = fit_segment(
        vol_shift, eye, log_prior, eye, bg, bg, eye, fit_warp=True, warp_solver="gn", **base
    )

    tw = gn["twarp"]
    assert torch.isfinite(tw).all()
    assert tw.abs().max() > 0.5  # the solver actually deforms (not stuck at zero)
    assert gn["ll"] > no_warp["ll"]  # and the deformation improves the fit


def test_plot_intensity_fit_writes_png(tmp_path):
    """The histogram diagnostic renders and saves for single- and multi-channel fits."""
    from fastfuncstuff.processing.segment import plot_intensity_fit, segment_apply

    torch.manual_seed(0)
    n = 24
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))
    base = dict(ngaus=[1, 1, 1], biasfwhm=12.0, samp=1.0, n_iter=5, fit_warp=False, verbose=False)  # noqa: C408

    fit = fit_segment(vol, eye, log_prior, eye, bg, bg, eye, **base)
    out = segment_apply(vol, log_prior, bg, bg, fit, mrf=0, cleanup=0, verbose=False)
    p = tmp_path / "hist.png"
    ret = plot_intensity_fit(
        fit, out["corrected"], out["posteriors"], tissue_names=["GM", "WM", "CSF"], path=str(p)
    )
    assert ret is None  # saved-and-closed path returns None
    assert p.exists() and p.stat().st_size > 0

    # multi-channel: a second channel (contrast-inverted) → 2 panels, still valid
    vol2 = [vol, vol.max() - vol]
    fit2 = fit_segment(vol2, eye, log_prior, eye, bg, bg, eye, **base)
    out2 = segment_apply(vol2, log_prior, bg, bg, fit2, mrf=0, cleanup=0, verbose=False)
    assert fit2["n_chan"] == 2
    fig = plot_intensity_fit(fit2, out2["corrected"], out2["posteriors"])  # no path → Figure
    assert len(fig.axes) == 4  # 2 rows (full-max / 2nd-peak scale) × 2 channels


def test_warp_focus_relaxes_smoothing_locally():
    """_smooth_field keep_weight keeps raw displacement where the weight is 1."""
    from fastfuncstuff.processing.segment import _smooth_field

    torch.manual_seed(0)
    gz = gy = gx = 8
    field = torch.zeros(gz, gy, gx, 3, dtype=torch.float64)
    field[4, 4, 4, 1] = 5.0  # a lone spike on the y-component

    # full smoothing erases the spike toward its (zero) neighbours
    fully = _smooth_field(field, 1.0)
    assert fully[4, 4, 4, 1] < 2.5

    # keep_weight=1 at the spike preserves its raw value; elsewhere still smoothed
    kw = torch.zeros(gz, gy, gx, dtype=torch.float64)
    kw[4, 4, 4] = 1.0
    focused = _smooth_field(field, 1.0, kw)
    assert focused[4, 4, 4, 1] == field[4, 4, 4, 1]
    assert focused[0, 0, 0, 1].abs() < 1e-6  # untouched region unaffected


def test_fit_segment_warp_focus_runs():
    """The misfit-driven focus path produces a finite, valid fit."""
    torch.manual_seed(0)
    n = 24
    vol, tissue, _, _ = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))

    fit = fit_segment(
        vol, eye, log_prior, eye, bg, bg, eye,
        ngaus=[1, 1, 1], biasfwhm=12.0, samp=1.0, n_iter=8,
        fit_warp=True, warp_focus=0.7, focus_quantile=0.8, warp_smooth=0.5, verbose=False,
    )  # fmt: skip
    assert torch.isfinite(fit["twarp"]).all()
    assert fit["twarp"].shape[-1] == 3


def test_undistort_and_cast_identity():
    """Zero warp + identity geometry: undistort returns the input, cast returns source."""
    from fastfuncstuff.processing.segment import cast_template_to_input, undistort_input

    n = 10
    vol = torch.rand(n, n, n)
    fit = {
        "twarp": torch.zeros(n, n, n, 3, dtype=torch.float64),
        "sk": [1, 1, 1],
        "vox2vox": torch.eye(4, dtype=torch.float64),
    }
    undist = undistort_input(vol, fit, device="cpu")
    assert torch.allclose(undist, vol, atol=1e-4)

    src = torch.rand(n, n, n)
    eye = torch.eye(4, dtype=torch.float64)
    cast = cast_template_to_input(src, eye, eye, fit, (n, n, n), device="cpu")
    assert torch.allclose(cast, src, atol=1e-4)


def test_input_in_template_identity():
    """Zero warp + identity affine: input resampled into template space == input."""
    from fastfuncstuff.processing.segment import input_in_template

    n = 10
    vol = torch.rand(n, n, n)
    fit = {
        "twarp": torch.zeros(n, n, n, 3, dtype=torch.float64),
        "sk": [1, 1, 1],
        "vox2vox": torch.eye(4, dtype=torch.float64),
    }
    for use_warp in (False, True):
        out = input_in_template(vol, fit, (n, n, n), use_warp=use_warp, device="cpu")
        assert out.shape == (n, n, n)
        assert torch.allclose(out, vol, atol=1e-4)


def test_undistort_shifts_along_warp():
    """A constant PE displacement should shift the undistorted image by that amount."""
    from fastfuncstuff.processing.segment import undistort_input

    n = 16
    vol = torch.zeros(n, n, n)
    vol[:, :, 8] = 1.0  # a bright plane at x=8
    tw = torch.zeros(n, n, n, 3, dtype=torch.float64)
    tw[..., 0] = -2.0  # disp_x = -2 ⇒ undistorted(p)=vol(p - disp)=vol(p+2): plane moves to x=6
    fit = {"twarp": tw, "sk": [1, 1, 1], "vox2vox": torch.eye(4, dtype=torch.float64)}
    undist = undistort_input(vol, fit, device="cpu")
    assert undist[:, :, 6].mean() > 0.8  # plane pulled from x=8 to x=6
    assert undist[:, :, 8].mean() < 0.2


def test_segment_apply_full_resolution_labels_match():
    """fit → apply: full-res posteriors argmax should recover the tissue regions."""
    from fastfuncstuff.processing.segment import segment_apply

    torch.manual_seed(0)
    n = 24
    vol, tissue, _, bias_true = _phantom_three_tissue(n)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    log_prior = torch.log(prob.clamp(0, 1) + 1e-4)
    eye = torch.eye(4, dtype=torch.float64)
    bg = prob[:, 0].mean(dim=(1, 2))

    fit = fit_segment(
        vol,
        eye,
        log_prior,
        eye,
        bg,
        bg,
        eye,
        ngaus=[1, 1, 1],
        biasfwhm=12.0,
        samp=1.0,
        n_iter=10,
        fit_warp=False,
        verbose=False,
    )
    applied = segment_apply(vol, log_prior, bg, bg, fit)
    assert applied["posteriors"].shape == (3, n, n, n)
    assert applied["corrected"].shape == (n, n, n)

    # inside each tissue region, that tissue should dominate the posterior
    labels = applied["posteriors"].argmax(dim=0)
    for k in range(3):
        region = tissue[k] > 0.5
        agree = (labels[region] == k).float().mean()
        assert agree > 0.9, (k, agree.item())

    # the bias field should make intensities MORE uniform within a tissue
    core = tissue[1] > 0.5  # the homogeneous core
    raw_cov = vol[core].std() / vol[core].mean()
    corr_cov = applied["corrected"][core].std() / applied["corrected"][core].mean()
    assert corr_cov < raw_cov  # bias correction reduces within-tissue variation


def test_mrf_cleanup_removes_speckle():
    """MRF cleanup should flip an isolated misclassified voxel back to its neighbours."""
    from fastfuncstuff.processing.segment import mrf_cleanup

    n = 9
    post = torch.zeros(2, n, n, n)
    post[0] = 0.9  # tissue 0 fills the volume
    post[1] = 0.1
    # one borderline speckle voxel leaning to tissue 1 — neighbours should resolve it
    post[0, 4, 4, 4] = 0.45
    post[1, 4, 4, 4] = 0.55
    cleaned = mrf_cleanup(post, mrf=1.0, vox=(1.0, 1.0, 1.0), n_iter=10)
    assert cleaned[0, 4, 4, 4] > cleaned[1, 4, 4, 4]  # neighbours win
    assert torch.allclose(cleaned.sum(0), torch.ones(n, n, n), atol=1e-5)

    # mrf=0 is a no-op
    assert torch.equal(mrf_cleanup(post, mrf=0.0, vox=(1.0, 1.0, 1.0)), post)


def test_clean_gwc_strips_disconnected_tissue():
    """clean_gwc should remove a GM blob disconnected from the WM-seeded brain."""
    from fastfuncstuff.processing.segment import clean_gwc

    n = 24
    post = torch.zeros(6, n, n, n)
    post[5] = 1.0  # background everywhere by default (class 5 = air)
    # a central brain: WM core wrapped in GM
    zz, yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij")
    r = ((xx - n / 2) ** 2 + (yy - n / 2) ** 2 + (zz - n / 2) ** 2).sqrt()
    wm = (r < 5).float()
    gm = ((r >= 5) & (r < 8)).float()
    # a disconnected GM blob in the corner (dura/eyeball-like)
    blob = ((xx - 3) ** 2 + (yy - 3) ** 2 + (zz - 3) ** 2 < 4).float()
    gm = (gm + blob).clamp(0, 1)
    post[0] = gm
    post[1] = wm
    post[5] = 1.0 - gm - wm
    post = post / post.sum(0, keepdim=True).clamp_min(1e-6)

    cleaned = clean_gwc(post, level=1)
    blob_mask = blob > 0.5
    brain_gm = ((r >= 5) & (r < 8)) & ~blob_mask
    assert cleaned[0][blob_mask].mean() < 0.05  # disconnected blob removed
    assert cleaned[0][brain_gm].mean() > 0.5  # real cortical GM preserved
    # posteriors never exceed a simplex; where any tissue survives they sum to 1
    assert cleaned.sum(0).max() <= 1.0 + 1e-4
    kept = cleaned.sum(0) > 0.5
    assert torch.allclose(cleaned.sum(0)[kept], torch.ones(kept.sum()), atol=1e-4)


def test_separable_bias_assembly_matches_the_dense_normal_equations():
    """The Kronecker-separable GN assembly must equal ``Phiᵀ diag(wt2) Phi`` exactly.

    SPM never materialises the ``(n_samp, d3)`` DCT design matrix: the samples sit on a
    regular lattice and the basis is separable, so it contracts one axis at a time
    (``kron(b3*b3', spm_krutil(wt2,B1,B2,1))``). That is the same arithmetic at
    ``O(ngz·nbx²nby²nbz²)`` instead of ``O(n_samp·d3²)`` — on a reference T1 at
    ``-biasfwhm 30`` (d3 = 3094) about 3700x fewer operations, and the dense form grows
    *quadratically* as ``-biasfwhm`` is lowered. Since the two are supposed to be
    algebraically identical, pin that directly against the dense reference, including the
    ``(i,i',j,j',k,k') -> (i,j,k,i',j',k')`` index shuffle that is easy to get wrong.
    """
    from fastfuncstuff.processing.segment import dct_basis

    torch.manual_seed(0)
    nx, ny, nz = 20, 18, 16  # full image dims the DCT basis is defined over
    sk = (2, 3, 2)
    gx = torch.arange(0, nx, sk[0], dtype=torch.float64)
    gy = torch.arange(0, ny, sk[1], dtype=torch.float64)
    gz = torch.arange(0, nz, sk[2], dtype=torch.float64)
    ngx, ngy, ngz = len(gx), len(gy), len(gz)
    nbx, nby, nbz = 4, 3, 5  # deliberately unequal so an axis mix-up cannot cancel
    d3 = nbx * nby * nbz

    b1 = dct_basis(gx + 1.0, nx, nbx)
    b2 = dct_basis(gy + 1.0, ny, nby)
    b3 = dct_basis(gz + 1.0, nz, nbz)
    wt1 = torch.randn(ngz, ngy, ngx, dtype=torch.float64)
    wt2 = torch.rand(ngz, ngy, ngx, dtype=torch.float64) + 0.5  # PSD weights, as in eq. 34

    # separable: contract x, then y, then z (mirrors _assemble_separable)
    p1 = (b1[:, :, None] * b1[:, None, :]).reshape(ngx, -1)
    p2 = (b2[:, :, None] * b2[:, None, :]).reshape(ngy, -1)
    p3 = (b3[:, :, None] * b3[:, None, :]).reshape(ngz, -1)
    u1 = torch.einsum("zya,yb->zab", (wt1.reshape(-1, ngx) @ b1).reshape(ngz, ngy, nbx), b2)
    beta_sep = torch.einsum("zab,zc->abc", u1, b3).reshape(-1)
    u2 = torch.einsum("zya,yb->zab", (wt2.reshape(-1, ngx) @ p1).reshape(ngz, ngy, -1), p2)
    alpha_sep = (
        (u2.reshape(ngz, -1).T @ p3)
        .reshape(nbx, nbx, nby, nby, nbz, nbz)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(d3, d3)
    )

    # dense reference: one row per lattice node, flattened (z, y, x) as `kept_flat` indexes
    zz, yy, xx = torch.meshgrid(
        torch.arange(ngz), torch.arange(ngy), torch.arange(ngx), indexing="ij"
    )
    bx, by, bz = b1[xx.reshape(-1)], b2[yy.reshape(-1)], b3[zz.reshape(-1)]
    xy = (bx[:, :, None] * by[:, None, :]).reshape(bx.shape[0], -1)
    phi = (xy[:, :, None] * bz[:, None, :]).reshape(bx.shape[0], -1)  # (n, d3)
    beta_dense = phi.T @ wt1.reshape(-1)
    alpha_dense = (phi * wt2.reshape(-1)[:, None]).T @ phi

    assert torch.allclose(beta_sep, beta_dense, atol=1e-10, rtol=1e-9), (
        f"Beta max |Δ| = {(beta_sep - beta_dense).abs().max():.3e}"
    )
    assert torch.allclose(alpha_sep, alpha_dense, atol=1e-10, rtol=1e-9), (
        f"Alpha max |Δ| = {(alpha_sep - alpha_dense).abs().max():.3e}"
    )
    # and Alpha must stay symmetric PSD, since the GN solve assumes it
    assert torch.allclose(alpha_sep, alpha_sep.T, atol=1e-12)
    assert torch.linalg.eigvalsh(alpha_sep).min() > -1e-9
