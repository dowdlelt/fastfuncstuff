"""Per-sweep refine templates (-save_refine_templates)."""

import numpy as np
import torch

from fastfuncstuff.processing.locomoco import (
    apply_jitter,
    estimate_residual_flow,
    make_jitter_fields,
    refine_template_stats,
)

DEV = torch.device("cpu")


def _moved(nt=12, seed=0, amp=0.4):
    rng = np.random.default_rng(seed)
    nx, ny, nz = 32, 32, 20
    b = torch.from_numpy(rng.normal(size=(1, nx, ny, nz)).astype(np.float32))
    b = torch.nn.functional.avg_pool3d(b[None], 3, 1, 1)[0]
    g = np.mgrid[0:nx, 0:ny, 0:nz].astype(np.float32)
    blob = np.exp(-((g[0] - 16) ** 2 + (g[1] - 16) ** 2 + (g[2] - 10) ** 2) / 120.0)
    base = (b + 8.0 * torch.from_numpy(blob)[None])[0].numpy()
    data = np.repeat(base[..., None], nt, axis=3)
    data = data + 0.05 * rng.normal(size=data.shape).astype(np.float32)
    f = make_jitter_fields(data.shape, [1, 2], [amp, amp], [4.0, 4.0], 0, DEV)
    return apply_jitter(data, f, [1, 2], DEV, warp_interp="bilinear")


def test_stats_are_the_three_named_reductions():
    x = torch.rand(4, 5, 6, 9)
    st = refine_template_stats(x)
    assert set(st) == {"mean", "median", "std"}
    for v in st.values():
        assert v.shape == (4, 5, 6)
    assert torch.allclose(st["mean"], x.mean(dim=3))
    assert torch.allclose(st["median"], x.median(dim=3).values)
    # Unbiased: compared across sweeps and against the baseline, so the estimator has to
    # be consistent between them rather than merely cheap.
    assert torch.allclose(st["std"], x.std(dim=3, unbiased=True))


def _run(refine_rounds, save=True):
    return estimate_residual_flow(
        _moved(),
        pe_axis=1,
        slice_axis=0,
        backend="flow",
        n_levels=2,
        n_iters=4,
        dual=True,
        is_3dacq=True,
        pe_axis2=2,
        refine_rounds=refine_rounds,
        save_refine_templates=save,
        verbose=False,
        device=DEV,
    )


def test_one_template_per_sweep_plus_the_uncorrected_baseline():
    # 3 refine passes -> baseline + initial + 3 = 5. Without the baseline "it got sharper"
    # would have nothing to be sharper THAN.
    assert len(_run(3).refine_templates) == 5
    assert len(_run(0).refine_templates) == 2


def test_off_by_default():
    assert _run(2, save=False).refine_templates is None


def test_the_baseline_is_the_uncorrected_series():
    data = _moved()
    res = estimate_residual_flow(
        data,
        pe_axis=1,
        slice_axis=0,
        backend="flow",
        n_levels=2,
        n_iters=4,
        dual=True,
        is_3dacq=True,
        pe_axis2=2,
        refine_rounds=1,
        save_refine_templates=True,
        verbose=False,
        device=DEV,
    )
    raw = refine_template_stats(torch.from_numpy(data))
    assert torch.allclose(res.refine_templates[0]["mean"], raw["mean"], atol=1e-5)


def test_correction_drops_the_temporal_sd_below_the_uncorrected_baseline():
    # The point of the diagnostic: a sweep that removed motion shows up as a lower sd.
    tm = _run(2).refine_templates
    sd = [float(t["std"].mean()) for t in tm]
    assert sd[1] < sd[0], sd  # the initial estimate is where the work happens
    assert all(np.isfinite(sd))


def test_fwhm_ladder_has_one_entry_per_sweep_and_falls_on_correction():
    res = estimate_residual_flow(
        _moved(amp=0.6),
        pe_axis=1,
        slice_axis=0,
        backend="flow",
        n_levels=3,
        n_iters=5,
        dual=True,
        is_3dacq=True,
        pe_axis2=2,
        refine_rounds=2,
        est_lev_fwhm=True,
        voxdims=(2.0, 2.0, 2.0),
        fwhm_polort=4,
        verbose=False,
        device=DEV,
    )
    f = res.refine_fwhms
    assert len(f) == 4  # uncorrected + initial + 2 refine passes
    assert all(np.isfinite(f)) and all(v > 0 for v in f)
    # Uncorrected motion leaves spatially correlated structure, so taking it out has to
    # lower the estimate. This is the whole premise of the diagnostic.
    assert f[1] < f[0], f


def test_fwhm_is_off_by_default():
    assert _run(1).refine_fwhms is None


def test_fwhm_needs_voxel_sizes_to_report_mm():
    import pytest

    with pytest.raises(ValueError, match="voxdims"):
        estimate_residual_flow(
            _moved(),
            pe_axis=1,
            slice_axis=0,
            backend="flow",
            n_levels=2,
            n_iters=3,
            dual=True,
            is_3dacq=True,
            pe_axis2=2,
            refine_rounds=0,
            est_lev_fwhm=True,
            verbose=False,
            device=DEV,
        )


def test_detrend_uses_legendre_not_monomials():
    # A degree-8 monomial basis on 60 points is badly conditioned enough that the
    # "detrended" residual carries the conditioning error into the ACF. Pin that the
    # basis actually in use stays well conditioned at the doubled degree.
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    p = construct_polynomial_matrix(60, 8, device=DEV)
    assert float(torch.linalg.cond(p)) < 10.0
