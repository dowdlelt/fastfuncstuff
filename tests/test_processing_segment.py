"""Tests for the Unified Segmentation core (processing/segment.py)."""

from __future__ import annotations

import math

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
    """The 4th/5th reg terms (le1/le2) add divergence/shear energy; a 3-tuple = le1=le2=0."""
    torch.manual_seed(0)
    field = torch.randn(6, 6, 6, 3, dtype=torch.float64)
    vox = (1.0, 1.0, 1.0)
    p3 = warp_penalty(field, (0.0, 0.0, 0.1), vox)
    p5_zero = warp_penalty(field, (0.0, 0.0, 0.1, 0.0, 0.0), vox)
    assert torch.allclose(p3, p5_zero)  # 3-tuple back-compat = 5-tuple with le=0
    # a pure-divergence field (u = ∇·grid) is penalised by le1 but not by le2 (shear-free)
    zz, yy, xx = torch.meshgrid(*[torch.arange(6, dtype=torch.float64)] * 3, indexing="ij")
    div_field = torch.stack([xx, yy, zz], dim=-1)  # ∂u_i/∂x_i = 1, off-diagonals 0
    assert warp_penalty(div_field, (0.0, 0.0, 0.0, 1.0, 0.0), vox) > 0  # le1 sees divergence
    assert warp_penalty(div_field, (0.0, 0.0, 0.0, 0.0, 1.0), vox) > 0  # le2 sees ε_ii
    # le1/le2 strictly increase the total penalty on a generic field
    assert warp_penalty(field, (0.0, 0.0, 0.1, 0.02, 0.03), vox) > p3


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
