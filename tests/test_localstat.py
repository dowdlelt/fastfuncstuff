"""Correctness for fastfuncstuff.stats.localstat (local spatial ACF, 3dLocalACF)."""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.stats.localstat import (
    FWHM_LABELS,
    acf_fwhm_batched,
    build_neighborhood,
    fit_acf_batched,
    local_acf,
    local_fwhm,
)

# --------------------------------------------------------------------------
# Neighborhood construction matches AFNI's voxel counts
# --------------------------------------------------------------------------


def test_sphere_voxel_counts():
    # AFNI help: SPHERE(1) on a 1mm^3 grid -> 7 voxels (center + 6 faces);
    # SPHERE(1.42) -> 19; SPHERE(1.74) -> 27.
    assert build_neighborhood("SPHERE(1)", (1.0, 1.0, 1.0)).offsets.shape[0] == 7
    assert build_neighborhood("SPHERE(1.42)", (1.0, 1.0, 1.0)).offsets.shape[0] == 19
    assert build_neighborhood("SPHERE(1.74)", (1.0, 1.0, 1.0)).offsets.shape[0] == 27


def test_rect_voxel_count():
    # RECT(0,0,2) on a 1mm^3 grid is a 5-voxel column (k = -2..2).
    nb = build_neighborhood("RECT(0,0,2)", (1.0, 1.0, 1.0))
    assert nb.offsets.shape[0] == 5


def test_negative_radius_is_voxel_units():
    # SPHERE(-2) uses index units regardless of voxel size; same count on a
    # 3 mm grid as SPHERE(2) on a 1 mm grid.
    a = build_neighborhood("SPHERE(-2)", (3.0, 3.0, 3.0)).offsets.shape[0]
    b = build_neighborhood("SPHERE(2)", (1.0, 1.0, 1.0)).offsets.shape[0]
    assert a == b


def test_radii_use_real_voxel_size_even_when_negative():
    # Inclusion is voxel-based but the ACF radius must use the true voxel size.
    nb = build_neighborhood("SPHERE(-2)", (3.0, 3.0, 3.0))
    # The nearest neighbor offset (one voxel along an axis) has radius 3 mm.
    nonzero = nb.radii[nb.radii > 0]
    assert abs(float(nonzero.min()) - 3.0) < 1e-6


def test_center_is_first_bin_radius_zero():
    nb = build_neighborhood("SPHERE(3)", (1.0, 1.0, 1.0))
    assert float(nb.radii[0]) == 0.0
    assert int(nb.bin_id[0]) == 0
    assert float(nb.bin_radius[0]) == 0.0
    # Bins increase monotonically in radius.
    br = nb.bin_radius.numpy()
    assert np.all(np.diff(br) > 0)


# --------------------------------------------------------------------------
# FWHM / FWQM bisection against closed forms
# --------------------------------------------------------------------------


def test_fwhm_pure_gaussian():
    # a = 1 -> ACF(r) = exp(-r^2 / 2b^2); FWHM = 2*sqrt(2 ln2)*b.
    b = torch.tensor([3.0], dtype=torch.float64)
    a = torch.tensor([1.0], dtype=torch.float64)
    c = torch.tensor([5.0], dtype=torch.float64)
    fwhm = acf_fwhm_batched(a, b, c, 0.5)
    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * 3.0
    assert abs(float(fwhm) - expected) < 1e-3


def test_fwhm_pure_exponential():
    # a = 0 -> ACF(r) = exp(-r/c); crosses 0.5 at r = c ln2 -> FWHM = 2 c ln2.
    a = torch.tensor([0.0], dtype=torch.float64)
    b = torch.tensor([2.0], dtype=torch.float64)
    c = torch.tensor([4.0], dtype=torch.float64)
    fwhm = acf_fwhm_batched(a, b, c, 0.5)
    expected = 2.0 * 4.0 * np.log(2.0)
    assert abs(float(fwhm) - expected) < 1e-3


# --------------------------------------------------------------------------
# Batched fit recovers known parameters from a clean curve
# --------------------------------------------------------------------------


def test_fit_recovers_known_params():
    a_true, b_true, c_true = 0.6, 3.0, 5.0
    radii = torch.linspace(0, 18, 40, dtype=torch.float64)
    model = a_true * torch.exp(-0.5 * radii**2 / b_true**2) + (1 - a_true) * torch.exp(
        -radii / c_true
    )
    y = model.view(1, -1).repeat(4, 1)  # 4 identical voxels
    w = torch.ones_like(y)
    a, b, c, ok = fit_acf_batched(radii, y, w, n_iter=80)
    assert bool(ok.all())
    # FWHM is the quantity users actually consume; check it is recovered tightly.
    fwhm = acf_fwhm_batched(a, b, c, 0.5)
    fwhm_true = acf_fwhm_batched(
        torch.tensor([a_true] * 4, dtype=torch.float64),
        torch.tensor([b_true] * 4, dtype=torch.float64),
        torch.tensor([c_true] * 4, dtype=torch.float64),
        0.5,
    )
    assert torch.allclose(fwhm, fwhm_true, atol=0.05)


# --------------------------------------------------------------------------
# End-to-end: smoother data -> larger measured ACF FWHM
# --------------------------------------------------------------------------


def _smoothed_volume(sigma: float, nt: int, shape, seed: int) -> torch.Tensor:
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    data = rng.standard_normal((nt, *shape)).astype(np.float32)
    if sigma > 0:
        for t in range(nt):
            data[t] = gaussian_filter(data[t], sigma=sigma)
    # Demean each voxel's time series (errts-like input).
    data -= data.mean(axis=0, keepdims=True)
    return torch.from_numpy(data)


def test_local_acf_runs_and_tracks_smoothing():
    shape = (16, 16, 16)
    nt = 60
    mask = torch.ones(shape, dtype=torch.bool)

    fwhms = []
    for sigma in (0.8, 2.0):
        data = _smoothed_volume(sigma, nt, shape, seed=int(sigma * 10))
        out = local_acf(
            data,
            voxdims=(1.0, 1.0, 1.0),
            nbhd="SPHERE(6)",
            mask=mask,
            device=torch.device("cpu"),
            do_median=False,
            verbose=0,
        )
        assert out.shape == (5, *shape)
        fwhm_map = out[3]
        # Interior voxels (avoid borders where the neighborhood is clipped).
        inner = fwhm_map[4:12, 4:12, 4:12]
        pos = inner[inner > 0]
        assert pos.numel() > 0
        assert torch.isfinite(pos).all()
        fwhms.append(float(pos.median()))

    # More spatial smoothing must yield a larger measured ACF FWHM.
    assert fwhms[1] > fwhms[0]


# --------------------------------------------------------------------------
# Local FWHM (Forman estimator, 3dLocalstat -stat fwhm)
# --------------------------------------------------------------------------


def _smoothed_3d(sigma: float, shape, seed: int) -> torch.Tensor:
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    vol = rng.standard_normal(shape).astype(np.float32)
    return torch.from_numpy(gaussian_filter(vol, sigma=sigma))


def test_local_fwhm_output_shape_and_labels():
    vol = _smoothed_3d(1.5, (20, 20, 20), seed=0)  # 3D input is accepted
    out = local_fwhm(
        vol,
        voxdims=(1.0, 1.0, 1.0),
        nbhd="SPHERE(-3)",
        device=torch.device("cpu"),
        do_median=False,
        verbose=0,
    )
    assert out.shape == (4, 20, 20, 20)
    assert FWHM_LABELS == ("FWHMx", "FWHMy", "FWHMz", "FWHMavg")
    assert torch.isfinite(out).all()


def test_local_fwhm_recovers_gaussian_smoothness():
    # Gaussian-smoothed white noise has smoothness FWHM = sqrt(8 ln2) * sigma.
    sigma = 2.0
    expected = float(np.sqrt(8 * np.log(2)) * sigma)  # ~4.71 voxels
    vol = _smoothed_3d(sigma, (28, 28, 28), seed=3)
    mask = torch.ones((28, 28, 28), dtype=torch.bool)
    out = local_fwhm(
        vol,
        voxdims=(1.0, 1.0, 1.0),
        nbhd="SPHERE(-4)",
        mask=mask,
        device=torch.device("cpu"),
        do_median=False,
        verbose=0,
    )
    bar = out[3]
    inner = bar[8:20, 8:20, 8:20]
    pos = inner[inner > 0]
    assert pos.numel() > 0
    med = float(pos.median())
    # The finite-difference estimator is biased a bit low at this smoothness;
    # accept a generous band around the analytic value.
    assert 0.7 * expected < med < 1.2 * expected, (med, expected)


def test_local_fwhm_tracks_smoothing():
    mask = torch.ones((24, 24, 24), dtype=torch.bool)
    meds = []
    for sigma in (1.0, 2.5):
        vol = _smoothed_3d(sigma, (24, 24, 24), seed=int(sigma * 7))
        out = local_fwhm(
            vol,
            voxdims=(1.0, 1.0, 1.0),
            nbhd="SPHERE(-4)",
            mask=mask,
            device=torch.device("cpu"),
            do_median=False,
            verbose=0,
        )
        inner = out[3][6:18, 6:18, 6:18]
        meds.append(float(inner[inner > 0].median()))
    assert meds[1] > meds[0]
