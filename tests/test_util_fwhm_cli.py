"""Smoke tests for the ffs_util_fwhm CLI (3dFWHMx-style whole-volume smoothness)."""

from __future__ import annotations

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from fastfuncstuff.cli.util_fwhm import main


def _smoothed_noise_nii(tmp_path, vox_mm=2.0, fwhm_mm=6.0, shape=(24, 24, 24), nt=40, seed=0):
    """A 4-D dataset of spatially-smoothed white noise with a known-ish blur."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    sigma_vox = (fwhm_mm / 2.35482) / vox_mm
    data = rng.standard_normal((*shape, nt)).astype(np.float32)
    for t in range(nt):
        data[..., t] = gaussian_filter(data[..., t], sigma=sigma_vox, mode="wrap")
    affine = np.diag([vox_mm, vox_mm, vox_mm, 1.0])
    path = tmp_path / "noise.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)
    mpath = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, np.int16), affine), mpath)
    return str(path), str(mpath)


def test_cli_estimates_plausible_fwhm(tmp_path):
    pytest.importorskip("scipy")
    inp, mask = _smoothed_noise_nii(tmp_path, vox_mm=2.0, fwhm_mm=6.0)
    out = tmp_path / "acf.1D"
    main(["-input", inp, "-mask", str(mask), "-acf1D", str(out), "-device", "cpu", "-verb", "0"])

    a, b, c, fwhm = (float(x) for x in out.read_text().splitlines()[-1].split())
    # Smoothed with ~6 mm; the estimate should be finite, above the voxel size,
    # and in a sane neighbourhood of the truth (small volume -> loose bounds).
    assert np.isfinite(fwhm)
    assert 3.0 < fwhm < 12.0, f"implausible FWHM {fwhm}"
    assert 0.0 <= a <= 1.0


def test_cli_automask_and_detrend_run(tmp_path):
    pytest.importorskip("scipy")
    inp, _ = _smoothed_noise_nii(tmp_path, nt=30)
    # Mean-zero-ish noise: automask may keep little, but the path must not crash;
    # detrend must run and still produce a finite estimate under an explicit mask.
    out = tmp_path / "acf.1D"
    main(["-input", inp, "-detrend", "2", "-acf1D", str(out), "-device", "cpu", "-verb", "0"])
    fwhm = float(out.read_text().splitlines()[-1].split()[3])
    assert np.isfinite(fwhm) and fwhm > 0.0


def test_cli_unif_runs_and_nounif_disables(tmp_path):
    """-unif (MAD uniformization, matches 3dFWHMx -detrend) must run and give a
    finite estimate; -nounif must suppress the auto-unif that -detrend implies."""
    pytest.importorskip("scipy")
    inp, mask = _smoothed_noise_nii(tmp_path, vox_mm=2.0, fwhm_mm=6.0)

    o_unif = tmp_path / "u.1D"
    main(
        [
            "-input",
            inp,
            "-mask",
            mask,
            "-unif",
            "-acf1D",
            str(o_unif),
            "-device",
            "cpu",
            "-verb",
            "0",
        ]
    )
    fu = float(o_unif.read_text().splitlines()[-1].split()[3])
    assert np.isfinite(fu) and fu > 0.0

    # -detrend implies unif; -nounif turns it back off -> a different estimate.
    o_det = tmp_path / "d.1D"
    o_no = tmp_path / "n.1D"
    main(
        [
            "-input",
            inp,
            "-mask",
            mask,
            "-detrend",
            "2",
            "-acf1D",
            str(o_det),
            "-device",
            "cpu",
            "-verb",
            "0",
        ]
    )
    main(
        [
            "-input",
            inp,
            "-mask",
            mask,
            "-detrend",
            "2",
            "-nounif",
            "-acf1D",
            str(o_no),
            "-device",
            "cpu",
            "-verb",
            "0",
        ]
    )
    fd = float(o_det.read_text().splitlines()[-1].split()[3])
    fn = float(o_no.read_text().splitlines()[-1].split()[3])
    assert np.isfinite(fd) and np.isfinite(fn)
    assert fd != fn, "-nounif should change the result vs the unif-by-default -detrend"
