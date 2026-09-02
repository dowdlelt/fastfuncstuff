"""Known-displacement injection (-inject_jitter) and its recovery statistics."""

import numpy as np
import torch

from fastfuncstuff.processing.locomoco import (
    _brain_mask_from,
    apply_jitter,
    estimate_residual_flow,
    jitter_recovery,
    make_jitter_fields,
)

DEV = torch.device("cpu")


def _phantom(nx=40, ny=40, nz=24, nt=12, seed=0):
    rng = np.random.default_rng(seed)
    b = torch.from_numpy(rng.normal(size=(1, nx, ny, nz)).astype(np.float32))
    b = torch.nn.functional.avg_pool3d(b[None], 3, 1, 1)[0]
    g = np.mgrid[0:nx, 0:ny, 0:nz].astype(np.float32)
    blob = np.exp(-((g[0] - nx / 2) ** 2 + (g[1] - ny / 2) ** 2 + (g[2] - nz / 2) ** 2) / 150.0)
    base = (b + 8.0 * torch.from_numpy(blob)[None])[0].numpy()
    data = np.repeat(base[..., None], nt, axis=3)
    return data + 0.05 * rng.normal(size=data.shape).astype(np.float32)


def test_injected_field_hits_the_requested_rms_amplitude():
    f = make_jitter_fields((16, 16, 12, 8), [1, 2], [0.25, 0.1], [4.0, 4.0], 0, DEV)
    assert len(f) == 2
    assert f[0].shape == (16, 16, 12, 8)
    assert abs(float(f[0].pow(2).mean().sqrt()) - 0.25) < 1e-4
    assert abs(float(f[1].pow(2).mean().sqrt()) - 0.10) < 1e-4


def test_amplitude_survives_the_smoothing_scale():
    # Scaling has to happen AFTER the blur; blurring white noise cuts its variance by a
    # factor that depends on sigma, so scaling first would make the amplitude drift with
    # -inject_smooth and quietly invalidate a sweep over it.
    for sigma in (0.0, 2.0, 8.0):
        f = make_jitter_fields((16, 16, 12, 6), [1], [0.2], [sigma], 0, DEV)
        assert abs(float(f[0].pow(2).mean().sqrt()) - 0.2) < 1e-4, sigma


def test_seed_makes_the_injection_reproducible_across_runs():
    a = make_jitter_fields((12, 12, 8, 5), [1], [0.2], [3.0], 7, DEV)
    b = make_jitter_fields((12, 12, 8, 5), [1], [0.2], [3.0], 7, DEV)
    c = make_jitter_fields((12, 12, 8, 5), [1], [0.2], [3.0], 8, DEV)
    assert torch.equal(a[0], b[0])
    assert not torch.equal(a[0], c[0])


def test_recovery_slope_is_one_for_a_perfect_estimate():
    inj = make_jitter_fields((12, 12, 8, 6), [1], [0.2], [3.0], 0, DEV)[0]
    mask = torch.ones(12, 12, 8, dtype=torch.bool)
    # The estimator returns the PULL displacement that UNDOES the injected shift, so a
    # perfect estimate is the negation; jitter_recovery normalises that to +1.
    slope, r = jitter_recovery(-inj, inj, mask)
    assert abs(slope - 1.0) < 1e-6
    assert abs(r - 1.0) < 1e-6


def test_recovery_slope_is_zero_when_the_estimator_is_blind():
    inj = make_jitter_fields((12, 12, 8, 6), [1], [0.2], [3.0], 0, DEV)[0]
    mask = torch.ones(12, 12, 8, dtype=torch.bool)
    slope, _ = jitter_recovery(torch.zeros_like(inj), inj, mask)
    assert slope == 0.0


def test_unrelated_motion_lowers_r_but_does_not_bias_the_slope():
    # The whole reason injection works on real data that already contains motion: the
    # run's own residual is uncorrelated with the injection, so it adds variance to the
    # estimate without pulling the slope off 1.
    mask = torch.ones(16, 16, 12, dtype=torch.bool)
    slopes, rs = [], []
    for seed in range(8):
        inj = make_jitter_fields((16, 16, 12, 10), [1], [0.2], [3.0], seed, DEV)[0]
        other = make_jitter_fields((16, 16, 12, 10), [1], [0.4], [3.0], 100 + seed, DEV)[0]
        slope, r = jitter_recovery(-inj + other, inj, mask)
        slopes.append(slope)
        rs.append(r)
    # Unbiased in expectation, not per draw: a single pair of independent smooth fields
    # correlates by a few percent by chance, which is scatter around 1 rather than a pull
    # away from it. Averaging over draws is what distinguishes the two.
    assert abs(float(np.mean(slopes)) - 1.0) < 0.03, slopes
    assert max(rs) < 0.85  # the contamination shows up here, and only here


def test_apply_jitter_actually_moves_the_data():
    data = _phantom()
    f = make_jitter_fields(data.shape, [1, 2], [0.3, 0.3], [4.0, 4.0], 0, DEV)
    out = apply_jitter(data, f, [1, 2], DEV, warp_interp="bilinear")
    assert out.shape == data.shape
    assert np.abs(out - data).mean() > 1e-3


def test_end_to_end_injection_is_recovered():
    data = _phantom()
    axes = [1, 2]
    inj = make_jitter_fields(data.shape, axes, [0.3, 0.3], [5.0, 5.0], 0, DEV)
    moved = apply_jitter(data, inj, axes, DEV, warp_interp="bilinear")
    res = estimate_residual_flow(
        moved,
        pe_axis=1,
        slice_axis=0,
        backend="flow",
        n_levels=2,
        n_iters=4,
        dual=True,
        is_3dacq=True,
        pe_axis2=2,
        refine_rounds=0,
        verbose=False,
        device=DEV,
    )
    brain = _brain_mask_from(torch.from_numpy(np.abs(moved).mean(axis=3)))
    by_axis = {ax: f for _, ax, f in res.pe_displacements()}
    for ax, tru in zip(axes, inj, strict=True):
        slope, r = jitter_recovery(by_axis[ax], tru.cpu(), brain)
        assert slope > 0.4, (ax, slope)
        assert r > 0.4, (ax, r)
