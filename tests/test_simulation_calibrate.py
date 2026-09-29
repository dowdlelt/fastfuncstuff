"""Noise profiles from ffs_reml outputs: tSNR bins with ARMA read back as (tau, f)."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.simulation.calibrate import noise_profile_from_maps, noise_profile_from_reml
from fastfuncstuff.simulation.noise import ou_to_arma11

TR = 1.25


def _planted_maps(shape=(10, 10, 10)):
    """tSNR rising along x; tau and phys share tied to it, as in cortex vs white matter."""
    rng = np.random.default_rng(0)
    n = int(np.prod(shape))
    tsnr = rng.uniform(10, 110, n)
    tau = np.where(tsnr < 60, 3.0, 8.0)
    f = np.where(tsnr < 60, 0.2, 0.7)
    a, b = (v.numpy() for v in ou_to_arma11(TR, torch.from_numpy(tau), torch.from_numpy(f)))
    return tsnr.reshape(shape), a.reshape(shape), b.reshape(shape)


def test_bins_recover_the_planted_parameters():
    tsnr, a, b = _planted_maps()
    profile = noise_profile_from_maps(tsnr, a, b, TR, quantiles=(0, 0.25, 0.5, 0.75, 1))
    assert profile.worst.tsnr < 30 and profile.best.tsnr > 90
    assert profile.worst.tau == pytest.approx(3.0, rel=1e-6)
    assert profile.worst.phys_fraction == pytest.approx(0.2, rel=1e-6)
    assert profile.best.tau == pytest.approx(8.0, rel=1e-6)
    assert profile.best.phys_fraction == pytest.approx(0.7, rel=1e-6)
    assert sum(bn.n_voxels for bn in profile.bins) == tsnr.size
    kw = profile.best.simulation_kwargs()
    assert set(kw) == {"tsnr", "phys_fraction", "tau"}


def test_mask_and_zeros_outside_are_excluded():
    tsnr, a, b = _planted_maps()
    tsnr[:2] = 0.0  # outside ffs_reml's automask
    mask = np.ones_like(tsnr, bool)
    mask[-2:] = False
    profile = noise_profile_from_maps(tsnr, a, b, TR, mask=mask)
    assert profile.n_voxels == 600


def test_grid_ceiling_is_reported():
    tsnr, a, b = _planted_maps()
    a = np.minimum(a, 0.8)  # what the default REML grid would return
    profile = noise_profile_from_maps(tsnr, a, b, TR, a_ceiling=0.8)
    assert profile.best.frac_a_at_ceiling > 0.9
    assert any("truncated" in n for n in profile.notes)


def test_lag_one_only_correlation_simulates_as_arma_at_its_tr():
    """a = 0, b > 0 (what real afni_proc Rvars are full of): MA(1), not white + OU."""
    tsnr, a, b = _planted_maps()
    a = np.zeros_like(a)
    b = np.full_like(b, 0.4)
    profile = noise_profile_from_maps(tsnr, a, b, TR)
    bn = profile.best
    assert not bn.representable and np.isnan(bn.tau)
    assert bn.acf_lag1 == pytest.approx(0.4 / 1.16) and bn.acf_lag2 == 0.0
    kw = bn.simulation_kwargs(tr=TR)
    assert kw["arma"] == pytest.approx((0.0, 0.4), abs=1e-6)
    with pytest.raises(ValueError, match="only defined at the measured TR"):
        bn.simulation_kwargs(tr=2.0)
    assert any("ARMA at this TR only" in n for n in profile.notes)


def test_a_bin_mixing_white_and_ou_is_summarised_by_its_acf():
    """AFNI folds near-white voxels onto a = b = 0; a median of a would read 0."""
    tsnr, a, b = _planted_maps()
    a, b = a.copy().ravel(), b.copy().ravel()
    a[::3], b[::3] = 0.0, 0.0  # a third white
    profile = noise_profile_from_maps(tsnr.ravel(), a, b, TR, quantiles=(0, 0.5, 1))
    bn = profile.best
    assert bn.frac_white == pytest.approx(1 / 3, abs=0.03)
    assert bn.representable and bn.arma_a > 0.5  # the tau = 8 s voxels still show
    assert bn.tau == pytest.approx(8.0, rel=0.05)


def _save(path, data, tr=None):
    img = nib.Nifti1Image(data.astype(np.float32), np.diag([2.5, 2.5, 2.5, 1.0]))
    if tr is not None:
        img.header.set_xyzt_units("mm", "sec")
        zooms = list(img.header.get_zooms())
        zooms[3] = tr
        img.header.set_zooms(zooms)
    nib.save(img, path)


def test_from_reml_files(tmp_path):
    tsnr, a, b = _planted_maps()
    rvar = np.stack([a, b, a * 0, a * 0 + 1], axis=-1)
    _save(tmp_path / "rvar.nii.gz", rvar)
    _save(tmp_path / "tsnr.nii.gz", tsnr)

    # Derived maps carry no TR: refuse rather than assume one.
    with pytest.raises(ValueError, match="no TR"):
        noise_profile_from_reml(tmp_path / "rvar.nii.gz", tmp_path / "tsnr.nii.gz")

    profile = noise_profile_from_reml(tmp_path / "rvar.nii.gz", tmp_path / "tsnr.nii.gz", tr=TR)
    assert profile.best.tau == pytest.approx(8.0, rel=1e-4)
    assert profile.voxel_size == (2.5, 2.5, 2.5)
    assert "TR 1.25" in profile.summary()


def test_header_tr_used_when_units_say_seconds(tmp_path):
    tsnr, a, b = _planted_maps()
    _save(tmp_path / "rvar.nii.gz", np.stack([a, b], axis=-1), tr=TR)
    _save(tmp_path / "tsnr.nii.gz", tsnr)
    profile = noise_profile_from_reml(tmp_path / "rvar.nii.gz", tmp_path / "tsnr.nii.gz")
    assert profile.tr == pytest.approx(TR)
    assert profile.tr_source == "header"


def test_a_stats_bucket_is_refused(tmp_path):
    tsnr, a, b = _planted_maps()
    _save(tmp_path / "bucket.nii.gz", np.stack([a * 40, b], axis=-1))
    _save(tmp_path / "tsnr.nii.gz", tsnr)
    with pytest.raises(ValueError, match="not ARMA"):
        noise_profile_from_reml(tmp_path / "bucket.nii.gz", tmp_path / "tsnr.nii.gz", tr=TR)
