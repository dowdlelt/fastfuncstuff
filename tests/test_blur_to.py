"""Blurring TO a smoothness: the time-point sample, the residual master, the search."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from fastfuncstuff.stats.blur_to import residual_master, sample_columns
from fastfuncstuff.surface.acf import blur_to_fwhm, detrend
from fastfuncstuff.surface.mesh import vertex_areas
from fastfuncstuff.surface.smooth import HeatSmoother, surface_fwhm


def _sheet(n: int, s: float):
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v = np.c_[ii.ravel() * s, jj.ravel() * s, np.zeros(n * n)]
    idx = ii * n + jj
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
    f = np.r_[np.c_[a.ravel(), b.ravel(), c.ravel()], np.c_[b.ravel(), d.ravel(), c.ravel()]]
    return v, f.astype(np.int64)


@pytest.fixture(scope="module")
def sheet():
    v, f = _sheet(101, 0.8)
    inner = (np.abs(v[:, 0] - 40) < 30) & (np.abs(v[:, 1] - 40) < 30)
    return v, f, inner


def _noise(v, f, t, fwhm, seed):
    white = np.random.default_rng(seed).normal(size=(len(v), t))
    return HeatSmoother(v, f, fwhm)(white / np.sqrt(vertex_areas(v, f))[:, None])


def _design(lengths, task):
    """Per-run Legendre drift (degree 2) plus one task column across all runs."""
    t = sum(lengths)
    x = np.zeros((t, 3 * len(lengths) + 1))
    s = 0
    for r, n in enumerate(lengths):
        x[s : s + n, 3 * r : 3 * r + 3] = np.polynomial.legendre.legvander(np.linspace(-1, 1, n), 2)
        s += n
    x[:, -1] = task
    return x


def test_sample_spreads_over_runs_and_never_takes_a_censored_point():
    lengths = [100, 300, 200]
    good = np.r_[np.arange(0, 50), np.arange(100, 600)]  # second half of run 1 censored
    cols = sample_columns(lengths, 40, good=good, seed=3)
    assert len(cols) == 40 and len(np.unique(cols)) == 40
    assert np.isin(cols, good).all()
    per_run = np.histogram(cols, bins=[0, 100, 400, 600])[0]
    # kept lengths 50 : 300 : 200 -> about 3.6 : 21.8 : 14.5
    assert per_run.tolist() in ([4, 22, 14], [4, 21, 15], [3, 22, 15])
    assert sample_columns(lengths, 10_000, good=good).tolist() == good.tolist()


def test_residual_master_is_the_ols_residual_with_censored_rows_out_of_the_fit():
    rng = np.random.default_rng(0)
    lengths = [60, 80]
    x = _design(lengths, rng.normal(size=140))
    y = rng.normal(size=(7, 140)) + 50
    y[:, 10] += 1e4  # a spike, censored: it must not move the fit
    good = np.delete(np.arange(140), 10)
    cols = np.array([3, 70, 139])
    beta = np.linalg.lstsq(x[good], y[:, good].T, rcond=None)[0]
    want = (y - (x @ beta).T)[:, cols]
    np.testing.assert_allclose(residual_master(y, x, cols, good=good, chunk=3), want, atol=1e-6)


def test_blurring_commutes_with_the_residual_so_no_refit_is_needed(sheet):
    v, f, inner = sheet
    lengths = [50, 50]
    rng = np.random.default_rng(1)
    x = _design(lengths, rng.normal(size=100))
    y = _noise(v, f, 100, 2.0, 1) + np.outer(rng.normal(size=len(v)), x[:, -1])
    hs = HeatSmoother(v, f, 4.0, mask=inner)
    cols = sample_columns(lengths, 12)
    np.testing.assert_allclose(
        residual_master(hs(y), x, cols), hs(residual_master(y, x, cols)), atol=1e-4
    )


def test_a_detrended_master_mistakes_the_task_for_smoothness_and_residuals_do_not(sheet):
    # A broad task activation is spatially smooth: left in the master, it reads as
    # extra smoothness, and blur-to then under-blurs. The FWHM moves only a few
    # percent, but near the target the kernel is a difference of squares, so the
    # kernel moves several times as much (here ~17%).
    v, f, inner = sheet
    lengths = [120, 120]
    t = np.arange(240)
    task = ((t % 30) < 15).astype(float)
    x = _design(lengths, task)
    noise = _noise(v, f, 240, 3.0, 2)
    noise /= noise[inner].std()
    blob = np.exp(-((v[:, 0] - 40) ** 2 + (v[:, 1] - 40) ** 2) / (2 * 15.0**2))
    y = noise + 1.5 * np.outer(blob, task)
    truth = surface_fwhm(noise, v, f, inner)[0]
    cols = sample_columns(lengths, 40)
    resid = residual_master(y, x, cols)
    detr = detrend(y, lengths, degree=2)[:, cols]
    assert surface_fwhm(resid, v, f, inner)[0] == pytest.approx(truth, rel=0.03)
    assert surface_fwhm(detr, v, f, inner)[0] > 1.03 * truth
    _, k_resid, _ = blur_to_fwhm(resid, v, f, 6.0, inner)
    _, k_detr, _ = blur_to_fwhm(detr, v, f, 6.0, inner)
    assert k_detr < 0.9 * k_resid


@pytest.mark.slow  # 35s
def test_forty_time_points_pin_the_kernel(sheet):
    # The default -blur_to_columns: across seeds, the kernel 40 points choose stays
    # within a few percent of the one all 400 choose.
    v, f, inner = sheet
    lengths = [100] * 4
    x = _design(lengths, np.zeros(400))[:, :-1]
    y = _noise(v, f, 400, 2.5, 5)
    _, k_all, _ = blur_to_fwhm(
        residual_master(y, x, np.arange(400)), v, f, 6.0, inner, n_columns=400, tol=0.005
    )
    ks = [
        blur_to_fwhm(
            residual_master(y, x, sample_columns(lengths, 40, seed=s)), v, f, 6.0, inner, tol=0.005
        )[1]
        for s in range(5)
    ]
    assert np.max(np.abs(np.array(ks) / k_all - 1)) < 0.04


def test_blursurf_cli_by_and_to_with_a_residual_master(tmp_path):
    from fastfuncstuff.cli.util_blursurf import main
    from fastfuncstuff.io.gifti import (
        load_gifti_data,
        mesh_fingerprint,
        save_gifti_data,
        save_gifti_surface,
    )

    def run(argv):
        return main([*argv, "-device", "cpu"])

    v, f = _sheet(41, 1.0)
    fp = mesh_fingerprint(f, len(v))
    save_gifti_surface(tmp_path / "mid.surf.gii", v, f)
    lengths = [60, 60]
    task = (np.arange(120) % 20 < 10).astype(float)
    x = _design(lengths, task)
    blob = np.exp(-((v[:, 0] - 20) ** 2 + (v[:, 1] - 20) ** 2) / 50)
    y = 100 + _noise(v, f, 120, 2.0, 7) + 3 * np.outer(blob, task)
    meta = {"TR_seconds": "1", "mesh_fingerprint": fp, "geometry": "mid.surf.gii"}
    for r, s in enumerate((0, 60)):
        save_gifti_data(tmp_path / f"run{r}.func.gii", y[:, s : s + 60].astype(np.float32), meta)
    inputs = [str(tmp_path / "run0.func.gii"), str(tmp_path / "run1.func.gii")]
    good = np.delete(np.arange(120), [5, 6, 70])
    header = [
        '# RunStart = "0,60"',
        f'# GoodList = "{",".join(map(str, good))}"',
        '# NRowFull = "120"',
    ]
    rows = [" ".join(f"{c:.8g}" for c in row) for row in x]
    (tmp_path / "X.xmat.1D").write_text("\n".join(header + rows) + "\n")

    out = tmp_path / "out"
    out.mkdir()
    assert run(["-input", *inputs, "-do_blur", "3", "-prefix", str(out / "b3")]) == 0
    b3, m3 = load_gifti_data(out / "b3.r02.func.gii")
    np.testing.assert_allclose(b3, HeatSmoother(v, f, 3.0)(y[:, 60:]), rtol=1e-5, atol=1e-3)
    assert m3["blur_kernel_fwhm"] == "3" and m3["geometry"] == "../mid.surf.gii"

    argv = ["-input", *inputs, "-blur_to_fwhm", "5", "-matrix", str(tmp_path / "X.xmat.1D")]
    assert run([*argv, "-prefix", str(out / "bto5")]) == 0
    _, m5 = load_gifti_data(out / "bto5.r01.func.gii")
    assert float(m5["master_fwhm_after"]) == pytest.approx(5.0, rel=0.03)
    assert float(m5["master_fwhm_before"]) < 3.0 < float(m5["blur_kernel_fwhm"]) < 5.0

    loc = ["-blur_to_local", "-save_blur_fwhm", str(out / "loc.shape.gii")]
    assert run([*argv, *loc, "-prefix", str(out / "bl5")]) == 0
    _, ml = load_gifti_data(out / "bl5.r02.func.gii")
    assert float(ml["master_fwhm_after"]) == pytest.approx(5.0, rel=0.04)
    assert int(ml["blur_steps"]) >= 1 and "blur_kernel_fwhm" not in ml
    fmap, _ = load_gifti_data(out / "loc.shape.gii")
    assert fmap.shape == (len(v),) and np.nanmedian(fmap[fmap > 0]) > 3.5

    target = 2.0 * float(np.median(y.mean(1) / y.std(1)))
    tsnr_argv = ["-input", *inputs, "-blur_to_tsnr", f"{target:.1f}", "-blur_tsnr_max_fwhm", "8",
                 "-matrix", str(tmp_path / "X.xmat.1D"), "-prefix", str(out / "bt")]  # fmt: skip
    assert run(tsnr_argv) == 0
    _, mt = load_gifti_data(out / "bt.r01.func.gii")
    assert float(mt["tsnr_median_after"]) > 1.5 * float(mt["tsnr_median_before"])
    assert int(mt["blur_steps"]) >= 1

    with pytest.raises(SystemExit):
        run(["-input", *inputs, "-do_blur", "3", "-blur_to_fwhm", "5", "-prefix", "z"])
    with pytest.raises(SystemExit):
        run(["-input", *inputs, "-do_blur", "3", "-blur_to_local", "-prefix", "z"])
    with pytest.raises(SystemExit):
        run(["-input", *inputs, "-do_blur", "3", "-matrix", "X", "-prefix", "z"])


# ---------------------------------------------------------------------------
# Volume: the in-mask Gaussian and the 3dFWHMx-steered search
# ---------------------------------------------------------------------------


def _ball(shape=(36, 36, 30), r=13.0):
    g = np.indices(shape).astype(float)
    c = (np.array(shape)[:, None, None, None] - 1) / 2
    return ((g - c) ** 2).sum(0) < r**2


def _smooth_noise_vol(shape, t, fwhm_vox, seed):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    s = fwhm_vox / 2.3548
    vols = [gaussian_filter(rng.normal(size=shape), s, mode="wrap") for _ in range(t)]
    return np.stack(vols, -1).reshape(-1, t)


def test_masked_gaussian_keeps_a_constant_and_leaves_the_outside_alone():
    import torch

    from fastfuncstuff.stats.blur_to import MaskedGaussian

    mask = _ball()
    x = np.zeros((mask.size, 3), np.float32)
    x[mask.reshape(-1)] = 7.0
    x[~mask.reshape(-1)] = -50.0  # air must not bleed in
    out = MaskedGaussian(mask, (2.0, 2.0, 2.0), 6.0, device=torch.device("cpu"))(x)
    np.testing.assert_allclose(out[mask.reshape(-1)], 7.0, rtol=1e-5)
    np.testing.assert_array_equal(out[~mask.reshape(-1)], -50.0)


def test_volume_blur_to_lands_on_the_target_on_the_whole_data():
    import torch

    from fastfuncstuff.stats.blur_to import volume_blur_to_fwhm

    cpu = torch.device("cpu")
    mask = _ball()
    shape, vox = mask.shape, (2.0, 2.0, 2.5)
    y = _smooth_noise_vol(shape, 80, 1.5, 3)
    g, kernel, achieved = volume_blur_to_fwhm(y[:, :40], mask, vox, 7.0, device=cpu)
    assert g is not None and 0 < kernel < 7.0
    assert achieved == pytest.approx(7.0, rel=0.03)
    # on the other 40 time points, by plain 3dFWHMx (the noise is stationary, so no
    # -unif, whose MAD from 40 points would itself read ~9% low)
    from fastfuncstuff.stats.fwhmx import estimate_fwhmx_run

    rows = torch.as_tensor(g(y[:, 40:])[mask.reshape(-1)])
    res = estimate_fwhmx_run(rows, torch.as_tensor(mask), shape, vox, progress=False)
    assert res.classic_combined == pytest.approx(7.0, rel=0.04)
    none, _, start = volume_blur_to_fwhm(y[:, :40], mask, vox, 2.0, device=cpu)
    assert none is None and start > 2.0


@pytest.mark.slow  # 18s
def test_reml_volume_blur_to_reaches_the_target_in_the_fit_residuals(tmp_path, monkeypatch, capsys):
    import nibabel as nib

    from fastfuncstuff.cli import reml

    # Pure noise under the model, so the residuals are what the search measured: a
    # task the model misfits leaves a smooth blob in them that the post-fit 3dFWHMx
    # (spatially demeaned, MAD -unif) and the search weigh differently. 300 TRs keep
    # the post-fit -unif's own MAD bias to ~2%.
    mask = _ball((30, 30, 24), 11.0)
    shape, t_n, tr = mask.shape, 300, 2.0
    y = 1000 + 5 * _smooth_noise_vol(shape, t_n, 1.5, 9)
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    img = nib.Nifti1Image(y.reshape(*shape, t_n).astype(np.float32), aff)
    img.header.set_xyzt_units("mm", "sec")
    img.header["pixdim"][4] = tr
    img.to_filename(str(tmp_path / "bold.nii"))
    nib.Nifti1Image(mask.astype(np.float32), aff).to_filename(str(tmp_path / "mask.nii"))
    with open(tmp_path / "ev.tsv", "w") as f:
        f.write("onset\tduration\ttrial_type\n")
        for o in np.arange(0, t_n * tr, 40):
            f.write(f"{o}\t20\ttask\n")

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["ffs_reml", *argv, "-device", "cpu", "-verb", "0"])
        reml.main()

    common = ["-input", str(tmp_path / "bold.nii"), "-events", str(tmp_path / "ev.tsv"),
              "-polort", "2", "-blur_to_fwhm", "6"]  # fmt: skip
    with pytest.raises(SystemExit):
        run(*common, "-Rbuck", str(tmp_path / "x.nii"))  # a volume needs -mask
    run(*common, "-mask", str(tmp_path / "mask.nii"), "-Rbuck", str(tmp_path / "s.nii"),
        "-save_acf", str(tmp_path / "acf"))  # fmt: skip

    def classic(stem):
        avg = (tmp_path / f"{stem}.fwhmx_reml.txt").read_text().splitlines()[-1].split()
        return float(np.exp(np.mean(np.log([float(v) for v in avg[1:4]]))))

    assert classic("acf") == pytest.approx(6.0, rel=0.05)
    run(*common, "-mask", str(tmp_path / "mask.nii"), "-Rbuck", str(tmp_path / "l.nii"),
        "-blur_to_local", "-save_blur_fwhm", str(tmp_path / "loc.nii"),
        "-save_acf", str(tmp_path / "acfl"), "-clustsim", "-clustsim_niter", "200")  # fmt: skip
    # the safe default stays the ACF null, but a varying blur is told plainly to switch
    out = capsys.readouterr().out
    assert "Strongly recommended: rerun with -clustsim_null resid" in out
    assert "Gaussian sums of the residuals" not in out
    assert list((tmp_path / "l_clustsim").glob("*.NN1_bisided.1D"))
    assert classic("acfl") == pytest.approx(6.0, rel=0.05)
    fmap = np.asarray(nib.load(str(tmp_path / "loc.nii")).dataobj)
    assert fmap.shape == shape and np.median(fmap[mask]) == pytest.approx(6.0, rel=0.1)


@pytest.mark.slow  # 11s
def test_local_blur_to_evens_out_a_volume_that_is_rough_on_one_side():
    import torch

    from fastfuncstuff.stats.blur_to import (
        local_volume_fwhm,
        volume_blur_to_fwhm,
        volume_blur_to_fwhm_local,
        volume_fwhm,
    )

    cpu = torch.device("cpu")
    shape, vox = (48, 40, 20), (2.0, 2.0, 2.0)
    mask = np.ones(shape, bool)
    mask[[0, -1]] = mask[:, [0, -1]] = mask[:, :, [0, -1]] = False
    rough = _smooth_noise_vol(shape, 320, 1.0, 1).reshape(*shape, -1)
    smooth = _smooth_noise_vol(shape, 320, 3.0, 2).reshape(*shape, -1)
    smooth /= smooth.std()
    rough /= rough.std()
    y = np.concatenate([rough[:24], smooth[24:]], axis=0).reshape(-1, 320)
    left, right = np.zeros(shape, bool), np.zeros(shape, bool)
    left[4:20] = mask[4:20]
    right[28:44] = mask[28:44]

    def halves(x):
        loc = local_volume_fwhm(x, mask, vox, 8.0, cpu).numpy()
        return float(np.nanmedian(loc[left])), float(np.nanmedian(loc[right]))

    master, held = y[:, :160], y[:, 160:]
    ad, start, achieved, _ = volume_blur_to_fwhm_local(master, mask, vox, 9.0, device=cpu)
    assert ad is not None and achieved == pytest.approx(9.0, rel=0.03)
    g, _, _ = volume_blur_to_fwhm(master, mask, vox, 9.0, device=cpu)
    l_ad, r_ad = halves(ad(held))
    l_g, r_g = halves(g(held))
    # one kernel keeps the rough side rough; diffusion that stops where the target is
    # reached brings the two sides together, and the replay lands on held-out data
    assert abs(l_ad - r_ad) < 0.5 * abs(l_g - r_g)
    assert volume_fwhm(ad(held), mask, vox, cpu) == pytest.approx(9.0, rel=0.04)
    # outside the mask nothing moves
    out = ~mask.reshape(-1)
    np.testing.assert_allclose(ad(held)[out], held[out], rtol=1e-6)


@pytest.mark.slow  # 12s
def test_surface_local_blur_to_evens_out_a_sheet_that_is_rough_on_one_side(sheet):
    from fastfuncstuff.surface.acf import blur_to_fwhm, blur_to_fwhm_local, local_surface_fwhm

    v, f, inner = sheet
    rough = _noise(v, f, 240, 1.5, 11)
    smooth = _noise(v, f, 240, 4.0, 12)
    rough /= rough[inner].std()
    smooth /= smooth[inner].std()
    y = np.where((v[:, 0] < 40)[:, None], rough, smooth)
    left = inner & (v[:, 0] > 14) & (v[:, 0] < 32)
    right = inner & (v[:, 0] > 48) & (v[:, 0] < 66)

    def halves(x):
        loc = local_surface_fwhm(x, v, f, inner, 8.0)
        return float(np.nanmedian(loc[left])), float(np.nanmedian(loc[right]))

    master, held = y[:, :120], y[:, 120:]
    sad, _, achieved, _ = blur_to_fwhm_local(master, v, f, 7.0, inner)
    assert sad is not None and achieved == pytest.approx(7.0, rel=0.03)
    hs, _, _ = blur_to_fwhm(master, v, f, 7.0, inner, n_columns=120)
    l_ad, r_ad = halves(sad(held))
    l_hs, r_hs = halves(hs(held))
    assert abs(l_ad - r_ad) < 0.5 * abs(l_hs - r_hs)
    assert surface_fwhm(sad(held), v, f, inner)[0] == pytest.approx(7.0, rel=0.04)


# ---------------------------------------------------------------------------
# Blur to a residual tSNR
# ---------------------------------------------------------------------------


@pytest.mark.slow  # 12s
def test_volume_blur_to_tsnr_blurs_noisy_white_noise_not_clean_or_structured():
    import torch

    from fastfuncstuff.stats.blur_to import volume_blur_to_tsnr

    cpu = torch.device("cpu")
    shape, vox, t_n = (48, 40, 20), (2.0, 2.0, 2.0), 320
    mask = np.ones(shape, bool)
    mask[[0, -1]] = mask[:, [0, -1]] = mask[:, :, [0, -1]] = False
    white = _smooth_noise_vol(shape, t_n, 0.8, 1).reshape(*shape, t_n)
    smooth = _smooth_noise_vol(shape, t_n, 4.0, 2).reshape(*shape, t_n)
    noise = white / white.std() * np.where(np.arange(48) < 20, 40.0, 10.0)[:, None, None, None]
    noise[20:28] = 40 * smooth[20:28] / smooth.std()  # noisy, but spatially structured
    y = (1000 + noise).reshape(-1, t_n)
    mean = y.mean(1)
    ad, t0, t1, _ = volume_blur_to_tsnr(
        (y - mean[:, None])[:, :160], mean, mask, vox, 60.0, 10.0, device=cpu
    )
    assert ad is not None

    def region(a, sl):
        m = np.zeros(shape, bool)
        m[sl] = mask[sl]
        return float(np.nanmedian(np.asarray(a).reshape(shape)[m]))

    assert region(t0, slice(4, 16)) == pytest.approx(25, rel=0.1)
    assert 60 <= region(t1, slice(4, 16)) < 70  # reached, ~one geometric step over
    assert region(t1, slice(32, 44)) == pytest.approx(region(t0, slice(32, 44)), rel=0.02)
    assert region(t1, slice(21, 27)) < 55  # gave up where blur stopped paying
    held = ad(y[:, 160:])
    held_tsnr = held.mean(1) / held.std(1)
    assert 57 < region(held_tsnr, slice(4, 16)) < 70  # the replay does what was measured


def test_surface_blur_to_tsnr_blurs_noisy_white_noise_not_clean_or_structured(sheet):
    from fastfuncstuff.surface.acf import blur_to_tsnr

    v, f, inner = sheet
    w = _noise(v, f, 320, 1.0, 1)
    w /= w[inner].std()
    sm = _noise(v, f, 320, 6.0, 2)
    sm /= sm[inner].std()
    x = v[:, 0]
    noise = np.where((x < 30)[:, None], 40 * w, np.where((x < 50)[:, None], 40 * sm, 10 * w))
    y = 1000 + noise
    mean = y.mean(1)
    sad, t0, t1, _ = blur_to_tsnr((y - mean[:, None])[:, :160], mean, v, f, 60.0, 10.0, inner)
    assert sad is not None
    white, smooth, clean = (
        inner & (x > 12) & (x < 26),
        inner & (x > 34) & (x < 46),
        inner & (x > 54),
    )
    assert 60 <= np.nanmedian(t1[white]) < 70
    assert np.nanmedian(t1[clean]) == pytest.approx(np.nanmedian(t0[clean]), rel=0.02)
    assert np.nanmedian(t1[smooth]) < 40
    held = sad(y[:, 160:])
    assert 57 < np.median((held.mean(1) / held.std(1))[white]) < 72


def test_reml_volume_blur_to_tsnr_end_to_end(tmp_path, monkeypatch, capsys):
    import nibabel as nib

    from fastfuncstuff.cli import reml

    shape, t_n, tr = (32, 24, 16), 240, 2.0
    mask = np.ones(shape, bool)
    mask[[0, -1]] = mask[:, [0, -1]] = mask[:, :, [0, -1]] = False
    white = _smooth_noise_vol(shape, t_n, 0.8, 4).reshape(*shape, t_n)
    amp = np.where(np.arange(32) < 16, 40.0, 8.0)[:, None, None, None]
    y = 1000 + amp * white / white.std()
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    img = nib.Nifti1Image(y.astype(np.float32), aff)
    img.header.set_xyzt_units("mm", "sec")
    img.header["pixdim"][4] = tr
    img.to_filename(str(tmp_path / "bold.nii"))
    nib.Nifti1Image(mask.astype(np.float32), aff).to_filename(str(tmp_path / "mask.nii"))
    with open(tmp_path / "ev.tsv", "w") as f:
        f.write("onset\ttrial_type\tduration\n")
        for o in np.arange(0, t_n * tr, 40):
            f.write(f"{o}\ttask\t20\n")
    monkeypatch.setattr(sys, "argv", [
        "ffs_reml", "-input", str(tmp_path / "bold.nii"), "-events", str(tmp_path / "ev.tsv"),
        "-polort", "2", "-mask", str(tmp_path / "mask.nii"), "-blur_to_tsnr", "60",
        "-Rbuck", str(tmp_path / "s.nii"), "-save_tsnr", str(tmp_path / "tsnr.nii"),
        "-save_blur_fwhm", str(tmp_path / "fw.nii"), "-clustsim", "-clustsim_niter", "100",
        "-clustsim_null", "resid",
        "-device", "cpu", "-verb", "0",
    ])  # fmt: skip
    reml.main()
    out = capsys.readouterr().out
    assert "Blurring TO residual tSNR 60" in out and "Gaussian sums of the residuals" in out
    fw = np.asarray(nib.load(str(tmp_path / "fw.nii")).dataobj)
    noisy, clean = mask.copy(), mask.copy()
    noisy[14:], clean[:18] = False, False
    assert np.median(fw[noisy]) > 1.5 * np.median(fw[clean])  # the noisy half got the blur
