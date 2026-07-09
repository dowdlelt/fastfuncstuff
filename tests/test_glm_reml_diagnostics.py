"""Tests for fastfuncstuff.glm.reml_diagnostics."""

import numpy as np
import torch

from fastfuncstuff.glm.reml_diagnostics import (
    DatasetDiagnostics,
    blur_est_1D,
    fwhmx_report_text,
    global_acf_fit,
    per_run_acf,
    resolve_mask,
    temporal_mean,
    temporal_std,
    tsnr,
)


def test_temporal_mean_std_tsnr():
    # voxel 0: constant 10 (std 0), voxel 1: 8,12 -> mean 10 std sqrt(8)=2.828
    data = torch.tensor([[10.0, 10.0, 10.0], [8.0, 10.0, 12.0]])
    m = temporal_mean(data)
    assert torch.allclose(m, torch.tensor([10.0, 10.0]))
    s = temporal_std(data)  # ddof=1
    assert abs(s[0].item()) < 1e-6
    assert abs(s[1].item() - 2.0) < 1e-6  # std([8,10,12], ddof=1) = 2
    t = tsnr(m, s)
    assert t[0].item() == 0.0  # std 0 -> tsnr 0, not inf
    assert abs(t[1].item() - 5.0) < 1e-6


def test_resolve_mask_supplied_vs_automask(capsys):
    mean_vol = torch.zeros(12, 14, 14)
    mean_vol[3:9, 4:10, 4:10] = 100.0  # a bright blob to automask
    supplied = mean_vol > 50
    out = resolve_mask(supplied, mean_vol, verbose=True)
    assert bool((out == supplied).all())

    auto = resolve_mask(None, mean_vol, verbose=True)
    assert auto.dtype == torch.bool
    # automask should find the blob region and nothing in the empty corners
    assert auto.sum() > 0
    assert not bool(auto[0, 0, 0])
    captured = capsys.readouterr().out
    assert "automasked" in captured


def _smooth_time_vol(vol4d, sigma):
    """Separable Gaussian smooth each timepoint (nt, nz, ny, nx)."""
    from fastfuncstuff.processing.formwarp import _separable_smooth_3d

    return torch.stack([_separable_smooth_3d(vol4d[t], sigma) for t in range(vol4d.shape[0])])


def test_global_acf_recovers_known_fwhm():
    torch.manual_seed(0)
    nt, nz, ny, nx = 40, 28, 32, 32
    vox = 2.0  # mm isotropic
    sigma_vox = 1.8
    noise = torch.randn(nt, nz, ny, nx)
    sm = _smooth_time_vol(noise, sigma_vox)
    mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
    mask[4:24, 5:27, 5:27] = True  # interior, avoid edge roll-off

    a, b, c, fwhm = global_acf_fit(sm, mask, (vox, vox, vox), nbhd="SPHERE(-9.666)")
    expected = 2.3548 * sigma_vox * vox  # FWHM (mm) of the applied Gaussian
    # Single global curve on modest nt is noisier than the per-voxel path; a
    # generous band still catches a broken pipeline (wrong units, no decay, etc.)
    assert 0.6 * expected < fwhm < 1.5 * expected, (fwhm, expected)
    assert 0.0 <= a <= 1.0


def test_collector_end_to_end():
    torch.manual_seed(1)
    shape = (10, 12, 12)
    nvox = int(np.prod(shape))
    nt = 30
    run_starts = [0, 15]

    diag = DatasetDiagnostics(volume_shape=shape, run_starts=run_starts, voxdims=(3.0, 3.0, 3.0))

    raw = 100.0 + 5.0 * torch.randn(nvox, nt)
    diag.observe_raw(raw)
    assert diag.maps["grandmean"].shape == shape
    assert abs(float(diag.maps["grandmean"].mean()) - 100.0) < 2.0

    scaled = 100.0 + 3.0 * torch.randn(nvox, nt)
    diag.observe_scaled(scaled)
    assert "raw_tsnr" in diag.maps
    assert np.isfinite(diag.maps["raw_tsnr"]).all()

    # residuals for a subset of voxels (masked), two labels
    mask = torch.zeros(shape, dtype=torch.bool)
    mask.reshape(-1)[: nvox // 2] = True
    n_masked = int(mask.sum())
    resid = {
        "ols": 2.0 * torch.randn(n_masked, nt),
        "reml": 1.0 * torch.randn(n_masked, nt),
    }
    diag.observe_residuals(resid, mask, want_tsnr=True, want_fwhmx=True)
    assert "resid_tsnr_ols" in diag.maps
    assert "resid_tsnr_reml" in diag.maps
    # reml residuals have smaller std -> higher tsnr where masked
    mv = mask.numpy()
    assert diag.maps["resid_tsnr_reml"][mv].mean() > diag.maps["resid_tsnr_ols"][mv].mean()
    # outside the mask stays zero
    assert diag.maps["resid_tsnr_ols"][~mv].sum() == 0.0
    # per-run detail + run-averaged .1D, per label
    assert "fwhmx_ols" in diag.tables and "blur_est_ols" in diag.tables
    assert "fwhmx_reml" in diag.tables and "blur_est_reml" in diag.tables
    # blur_est .1D last line parses to 4 numbers (a b c FWHM)
    data_line = [ln for ln in diag.tables["blur_est_ols"].splitlines() if not ln.startswith("#")][-1]
    assert len(data_line.split()) == 4


def test_blur_est_is_run_mean():
    rows = [(1, 0.5, 2.0, 3.0, 8.0), (2, 0.7, 3.0, 5.0, 10.0)]
    txt = fwhmx_report_text(rows)
    assert "avg" in txt
    one_d = blur_est_1D(rows)
    a, b, c, fwhm = (float(x) for x in one_d.splitlines()[-1].split())
    assert abs(a - 0.6) < 1e-6 and abs(b - 2.5) < 1e-6 and abs(c - 4.0) < 1e-6 and abs(fwhm - 9.0) < 1e-6
    # per_run_acf is exercised end-to-end in test_collector_end_to_end
    assert callable(per_run_acf)
