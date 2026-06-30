"""Tests for the NORDIC single-echo factor-sweep residual-correlation diagnostic."""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from fastfuncstuff.denoise.nordic import NordicConfig
from fastfuncstuff.denoise.nordic_sweep import run_nordic_factor_sweep
from fastfuncstuff.stats.voxel_correlation import (
    analytic_r_null,
    corr_histogram_distance,
)


def _write_nifti(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)


# --- Primitive: streamed correlation matches np.corrcoef ---


def test_corr_histogram_matches_numpy():
    """Mean/|r| from the streamed primitive match dense np.corrcoef on small V."""
    rng = np.random.default_rng(0)
    v, t = 80, 60
    ts = torch.from_numpy(rng.standard_normal((v, t)).astype(np.float32))
    coords = torch.stack(
        [torch.arange(v) % 8, (torch.arange(v) // 8) % 8, torch.zeros(v, dtype=torch.long)],
        dim=1,
    ).float()

    summ = corr_histogram_distance(ts, coords, r_bins=400, n_dist_bins=10)

    cc = np.corrcoef(ts.numpy())
    iu = np.triu_indices(v, k=1)
    ref_pairs = cc[iu]
    assert summ.n_pairs == ref_pairs.size
    assert abs(summ.mean_r - ref_pairs.mean()) < 1e-3
    assert abs(summ.mean_abs_r - np.abs(ref_pairs).mean()) < 2e-3


def test_corr_histogram_excludes_self_pairs():
    """Self-correlations (r=1) must not inflate the histogram toward 1."""
    rng = np.random.default_rng(1)
    v, t = 50, 100
    ts = torch.from_numpy(rng.standard_normal((v, t)).astype(np.float32))
    coords = torch.zeros(v, 3)
    coords[:, 0] = torch.arange(v).float()
    summ = corr_histogram_distance(ts, coords, n_dist_bins=5)
    # Independent noise: mean |r| near the analytic null, nowhere near 1.
    null = analytic_r_null(t)
    assert abs(summ.mean_abs_r - null["mean_abs_r"]) < 0.02
    assert summ.mean_abs_r < 0.2


def test_analytic_null_decreases_with_timepoints():
    assert analytic_r_null(50)["mean_abs_r"] > analytic_r_null(500)["mean_abs_r"]
    assert analytic_r_null(100)["mean_r"] == 0.0


# --- Distance interaction: locally-correlated field reads near > far ---


def test_distance_interaction_detects_local_structure():
    """A spatially-smooth shared component makes near voxels correlate > far."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(2)
    side, t = 16, 80
    coords = (
        torch.stack(
            torch.meshgrid(torch.arange(side), torch.arange(side), torch.arange(1), indexing="ij"),
            dim=-1,
        )
        .reshape(-1, 3)
        .float()
    )
    v = coords.shape[0]
    # Local signal: a spatially-smooth (blurred) weight map times a shared time
    # course. Smooth => nearby voxels share weight => correlate more than distant.
    smooth = gaussian_filter(rng.standard_normal((side, side)), sigma=2.5)
    wx = smooth.reshape(-1)
    shared = rng.standard_normal(t)
    field = np.outer(wx, shared) * 4.0
    noise = rng.standard_normal((v, t))
    ts = torch.from_numpy((field + noise).astype(np.float32))

    summ = corr_histogram_distance(ts, coords, n_dist_bins=12)
    pop = summ.dist_count > 0
    near = summ.dist_mean_abs_r[pop][0].item()
    far = summ.dist_mean_abs_r[pop][-1].item()
    assert near > far + 0.05


# --- End-to-end sweep on synthetic single-echo data ---


def _run_sweep(tmp_path, data, prefix, max_voxels=None):
    magn = tmp_path / f"{prefix}_magn.nii.gz"
    _write_nifti(magn, data)
    cfg = NordicConfig(
        magnitude_only=True,
        nordic=True,
        temporal_phase=0,
        factor_sweep=True,
        factor_sweep_values=(0.5, 1.0, 1.5, 2.0),
        factor_sweep_max_voxels=max_voxels,
        verbose=False,
    )
    return run_nordic_factor_sweep(
        str(magn), None, str(tmp_path / prefix), cfg, device=torch.device("cpu")
    )


def test_sweep_pure_noise_stays_near_null(tmp_path):
    """Pure thermal noise: in-brain residual correlation hugs the null at every
    factor (no liftoff), and the diagnostic files are written."""
    rng = np.random.default_rng(3)
    data = rng.standard_normal((20, 20, 4, 60)).astype(np.float32) * 5.0 + 100.0
    summary = _run_sweep(tmp_path, data, "noise")

    null = summary["null"]
    band = null["mean_abs_r"] + 2 * null["ci95_r"]
    whole = summary["masks"]["whole"]
    # NaN entries = factors where nothing was removed (residual empty); ignore them.
    assert np.nanmax(whole["mean_abs_r"]) < band + 0.02

    out = summary["outputs"]
    for key in ("json", "tsv", "plot_summary", "plot_distributions"):
        assert Path(out[key]).exists()


def test_overremoval_puts_shared_signal_into_residual():
    """Drive the sweep engine directly: at a threshold above the whole spectrum
    NORDIC removes everything, so the residual equals the data. Where a shared
    smooth signal lives, the residual-magnitude correlation is then far above the
    timepoint null; in the surrounding noise it stays at null. This is the
    over-removal the sweep is built to flag, isolated from NORDIC's auto-lambda."""
    from scipy.ndimage import gaussian_filter

    from fastfuncstuff.denoise.nordic_sweep import (
        _cache_patch_svds,
        _residual_magnitude_at_factor,
    )

    rng = np.random.default_rng(5)
    nx, ny, nz, t = 20, 20, 2, 60
    dev = torch.device("cpu")
    noise = rng.standard_normal((nx, ny, nz, t)) + 1j * rng.standard_normal((nx, ny, nz, t))
    data = noise.astype(np.complex64)
    # Central block: rank-2 shared signal with a smooth spatial loading.
    block = (slice(5, 15), slice(5, 15))
    shared = rng.standard_normal((2, t))
    load = np.stack([gaussian_filter(rng.standard_normal((10, 10)), 2.0).ravel() for _ in range(2)])
    sig = (load.T @ shared).reshape(10, 10, 1, t)
    data[block[0], block[1], :, :] += (
        np.broadcast_to(sig, (10, 10, nz, t)).astype(np.complex64) * 8.0
    )
    data_t = torch.from_numpy(data).to(dev)

    cache = _cache_patch_svds(data_t, (7, 7, 2), 2, 256, False, dev)
    thr = float(cache.s.max()) * 2.0  # above the whole spectrum -> remove everything
    resid = _residual_magnitude_at_factor(cache, data_t, thr, 256, dev)  # (nvox, t)

    in_mask = torch.zeros(nx, ny, nz, dtype=torch.bool)
    in_mask[block[0], block[1], :] = True
    in_idx = torch.nonzero(in_mask.reshape(-1)).squeeze(1)
    out_idx = torch.nonzero(~in_mask.reshape(-1)).squeeze(1)
    ii, jj, kk = torch.meshgrid(torch.arange(nx), torch.arange(ny), torch.arange(nz), indexing="ij")
    coords = torch.stack([ii.ravel(), jj.ravel(), kk.ravel()], dim=1).float()

    in_summ = corr_histogram_distance(resid[in_idx], coords[in_idx], n_dist_bins=8)
    out_summ = corr_histogram_distance(resid[out_idx], coords[out_idx], n_dist_bins=8)
    null = analytic_r_null(t)["mean_abs_r"]
    assert in_summ.mean_abs_r > null + 0.1
    assert in_summ.mean_abs_r > out_summ.mean_abs_r + 0.1
    assert abs(out_summ.mean_abs_r - null) < 0.03
