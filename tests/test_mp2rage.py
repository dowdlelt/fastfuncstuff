"""MP2RAGE joint denoising primitives (processing/mp2rage.py)."""

from __future__ import annotations

import torch

from fastfuncstuff.processing import mp2rage as M


def _phantom(n=48, sigma=6.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    zz, yy, xx = torch.meshgrid(*[torch.arange(n)] * 3, indexing="ij")
    r = ((zz - n / 2) ** 2 + (yy - n / 2) ** 2 + (xx - n / 2) ** 2).float().sqrt()
    a = torch.where(r < 9, -120.0, torch.where(r < 15, -30.0, 70.0))  # signed INV1
    b = torch.where(r < 9, 110.0, torch.where(r < 15, 175.0, 200.0))  # INV2
    x1 = a + sigma * torch.randn(n, n, n, generator=g)
    x2 = b + sigma * torch.randn(n, n, n, generator=g)
    return torch.stack([x1, x2]), (a, b), r


def test_uni_round_trip_and_polarity():
    inv1 = torch.tensor([50.0, 50.0, 0.0])
    inv2 = torch.tensor([100.0, 100.0, 100.0])
    s = torch.tensor([1.0, -1.0, 1.0])
    uni = M.uni_from_inversions(s * inv1, inv2)
    assert torch.equal(M.uni_polarity(uni), torch.tensor([1.0, -1.0, 1.0]))
    assert torch.allclose(uni[2], torch.tensor(M.UNI_OFFSET))  # INV1 at its null -> midpoint


def test_joint_nlm_reaches_sqrt_n_and_keeps_residuals_independent():
    raw, (a, b), r = _phantom()
    out, wsum = M.joint_nlm(raw, torch.tensor([6.0, 6.0]), rician=(False, False), beta=0.5)
    flat = r > 18
    err = (out[0] - a)[flat].std()
    neff = wsum[flat].median()
    assert neff > 30
    assert err < 1.3 * 6.0 / neff.sqrt()  # close to the ideal sigma/sqrt(Neff)
    # independent noise in, independent residuals out: structure was not removed
    mask = torch.ones_like(r, dtype=torch.bool)
    assert abs(M.residual_cross_correlation(raw, out, mask)) < 0.05
    # the CSF/GM/WM boundaries survive (no smearing across shells)
    assert (out[0] - a)[(r > 10) & (r < 13)].abs().median() < 3.0


def test_noise_sigma_survives_correlated_noise_along_one_axis():
    """PE-axis noise correlation made the first estimator read ~20% low; the readout axis
    (uncorrelated) must win."""
    g = torch.Generator().manual_seed(1)
    n, sigma = 64, 8.0
    w = torch.randn(n + 1, n, n, generator=g)
    noise = (w[1:] + w[:-1]) / 2**0.5  # unit variance, lag-1 corr 0.5 along axis 0 only
    x = 100.0 + sigma * noise
    mask = torch.ones(n, n, n, dtype=torch.bool)
    est = M.estimate_noise_sigma(x, mask)
    assert abs(est - sigma) / sigma < 0.06


def test_regularised_uni_darkens_air_and_spares_tissue():
    tissue = M.uni_from_inversions(torch.tensor([70.0]), torch.tensor([200.0]))
    tissue_reg = M.uni_from_inversions(torch.tensor([70.0]), torch.tensor([200.0]), reg=175.0)
    air_reg = M.uni_from_inversions(torch.tensor([3.0]), torch.tensor([-2.0]), reg=175.0)
    assert abs(float(tissue_reg - tissue)) < 30  # a few counts out of 4095
    assert float(air_reg) < 100  # air goes to black instead of a random +-0.5


def test_denoise_mp2rage_on_phantom():
    raw, (a, b), r = _phantom(n=40)
    uni = M.uni_from_inversions(a, b)  # noiseless scanner UNI: supplies the polarity
    out = M.denoise_mp2rage(raw[0].abs(), raw[1], uni, tissue_mask=r < 18)
    assert abs(out["k"] - 1.0) < 0.05
    assert abs(float(out["sigma"][0]) - 6.0) < 1.0
    err_raw = (M.uni_from_inversions(raw[0], raw[1]) - uni)[r > 16].std()
    err_den = (out["uni"] - uni)[r > 16].std()
    assert err_den < 0.6 * err_raw
