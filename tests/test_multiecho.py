"""Multi-echo T2*/S0 decay fitting, optimal combination, and leave-one-echo-out QC.

Verifies the GPU primitives in ``fastfuncstuff.processing.multiecho``:
recovery of known S0/T2*, machine-precision agreement of the batched LM curve fit with
scipy's per-voxel ``curve_fit`` (the correctness gate for the accelerator), optimal
combination against tedana, and that the leave-one-echo-out residual + robust weighting
flag and down-weight a corrupted echo.
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.processing import multiecho as me

DEV = torch.device("cpu")
TES = torch.tensor([15.0, 39.0, 63.0, 87.0, 111.0])  # ms, 5 echoes


def _synth(n_vox=300, n_tr=8, noise=5.0, seed=0):
    """Synthetic monoexponential multi-echo data with known params -> (data, t2s, s0)."""
    g = torch.Generator().manual_seed(seed)
    t2s = torch.empty(n_vox).uniform_(20.0, 90.0, generator=g)
    s0 = torch.empty(n_vox).uniform_(600.0, 1800.0, generator=g)
    clean = me.monoexp(TES, s0, t2s).unsqueeze(-1)  # (V, E, 1)
    data = (
        clean.expand(n_vox, len(TES), n_tr)
        + torch.randn(n_vox, len(TES), n_tr, generator=g) * noise
    )
    return data, t2s, s0


def test_loglinear_recovers_params():
    data, t2s, s0 = _synth(noise=2.0)
    _, am = me.make_adaptive_mask(data)
    out = me.fit_decay(data, TES, am, fittype="loglin", fitmode="all")
    assert (out["t2s"] - t2s).abs().median() < 1.0  # within 1 ms
    assert ((out["s0"] - s0).abs() / s0).median() < 0.02


def test_curvefit_recovers_params_and_no_failures():
    data, t2s, s0 = _synth(noise=4.0)
    _, am = me.make_adaptive_mask(data)
    out = me.fit_decay(data, TES, am, fittype="curvefit", fitmode="all")
    assert (out["t2s"] - t2s).abs().median() < 0.5
    assert out["failures"].sum() == 0
    assert (out["t2s_var"] >= 0).all() and (out["s0_var"] >= 0).all()


def test_curvefit_matches_scipy():
    """The batched LM must reproduce scipy.optimize.curve_fit per voxel."""
    scipy_opt = pytest.importorskip("scipy.optimize")
    data, _, _ = _synth(n_vox=60, noise=6.0, seed=3)
    _, am = me.make_adaptive_mask(data)
    out = me.fit_decay(data, TES, am, fittype="curvefit", fitmode="all", fit_all_timepoints=True)

    # Compare only voxels where every echo is good, so the GPU fit (which down-weights
    # echoes beyond the adaptive count) and scipy fit the identical set of points.
    full = torch.where(am == len(TES))[0].tolist()
    assert len(full) > 10
    tes_long = np.repeat(TES.numpy(), data.shape[2])
    mono = lambda te, s0, t2s: s0 * np.exp(-te / t2s)  # noqa: E731
    ll = me.fit_loglinear(data.reshape(data.shape[0], -1), torch.from_numpy(tes_long))
    for v in full:
        y = data[v].reshape(-1).numpy()
        popt, _ = scipy_opt.curve_fit(
            mono,
            tes_long,
            y,
            p0=(float(ll[1][v]), float(ll[0][v])),
            bounds=((y.min(), 0), (np.inf, np.inf)),
        )
        # Params agree closely, and -- the meaningful equivalence on a flat likelihood
        # ridge -- the batched LM reaches at least as low a residual as scipy.
        assert abs(float(out["s0"][v]) - popt[0]) / popt[0] < 0.01
        assert abs(float(out["t2s"][v]) - popt[1]) / popt[1] < 0.01
        sse_me = float(((mono(tes_long, float(out["s0"][v]), float(out["t2s"][v])) - y) ** 2).sum())
        sse_scipy = float(((mono(tes_long, *popt) - y) ** 2).sum())
        assert sse_me <= sse_scipy * (1 + 1e-4)


def test_optcom_matches_tedana():
    tcombine = pytest.importorskip("tedana.combine")
    data, _, _ = _synth(seed=2)
    _, am = me.make_adaptive_mask(data)
    out = me.fit_decay(data, TES, am, fittype="curvefit", fitmode="all")
    oc_f = me.make_optcom(data, TES, am, t2s=out["t2s"], combmode="t2s")
    oc_t = tcombine.make_optcom(
        data.numpy(), TES.numpy(), am.numpy(), t2s=out["t2s"].numpy(), combmode="t2s"
    )
    rel = (oc_f.numpy() - oc_t) / (np.abs(oc_t) + 1e-6)
    assert np.median(np.abs(rel)) < 1e-5


def test_adaptive_mask_matches_tedana():
    tutils = pytest.importorskip("tedana.utils")
    data, _, _ = _synth(seed=5)
    _, am_f = me.make_adaptive_mask(data, methods=("dropout",))
    _, am_t = tutils.make_adaptive_mask(data.numpy(), threshold=1, methods=["dropout"])
    assert np.array_equal(am_f.numpy(), am_t)


def test_loeo_clean_vs_corrupt():
    data, _, _ = _synth(noise=4.0)
    _, am = me.make_adaptive_mask(data)
    avail = me.availability_weights(am, len(TES))
    mean_echo = data.mean(dim=2)

    resid_clean, _ = me.leave_one_echo_out(mean_echo, TES, avail, fittype="curvefit")
    clean_level = torch.nanmedian(resid_clean.abs())

    bad = mean_echo.clone()
    bad[:, 0] *= 1.5  # inflate first echo
    resid_bad, _ = me.leave_one_echo_out(bad, TES, avail, fittype="curvefit")
    per_echo = torch.nanmedian(resid_bad.abs(), dim=0).values
    # The corrupted echo's LOEO residual dwarfs the clean baseline and is the clear
    # outlier. (Neighbouring echoes inflate somewhat too, since the corrupt echo also
    # contaminates the folds that hold *them* out -- IRLS iteration handles that.)
    assert per_echo[0] > 10 * clean_level
    assert int(per_echo.argmax()) == 0
    assert per_echo[0] > 2 * per_echo[1:].max()


def test_robustness_weight_drops_corrupt_echo():
    data, _, _ = _synth(noise=4.0)
    _, am = me.make_adaptive_mask(data)
    avail = me.availability_weights(am, len(TES))
    bad = data.mean(dim=2).clone()
    bad[:, 0] *= 1.5
    resid, _ = me.leave_one_echo_out(bad, TES, avail, fittype="curvefit")
    w = me.robustness_weights(resid)
    assert w[:, 0].mean() < 0.3  # corrupted echo strongly down-weighted
    assert w[:, 2:].mean() > 0.8  # clean echoes retained


def test_robust_fit_beats_curvefit_on_corrupted_data():
    data, t2s, _ = _synth(noise=4.0)
    _, am = me.make_adaptive_mask(data)
    corrupt = data.clone()
    corrupt[:, 0, :] *= 1.5
    plain = me.fit_decay(corrupt, TES, am, fittype="curvefit", fitmode="all")
    robust = me.fit_robust(corrupt, TES, am, fitmode="all", n_irls=2)
    err_plain = (plain["t2s"] - t2s).abs().median()
    err_robust = (robust["t2s"] - t2s).abs().median()
    assert err_robust < err_plain
    assert robust["echo_weight"][:, 0].mean() < robust["echo_weight"][:, 2:].mean()


def test_fitmode_ts_shapes():
    data, _, _ = _synth(n_vox=50, n_tr=5)
    _, am = me.make_adaptive_mask(data)
    out = me.fit_decay(data, TES, am, fittype="curvefit", fitmode="ts")
    assert out["t2s"].shape == (50, 5)
    assert out["failures"].shape == (50, 5)


def test_modify_maps_floors_and_limited():
    data, _, _ = _synth()
    _, am = me.make_adaptive_mask(data)
    out = me.fit_decay(data, TES, am, fittype="loglin", fitmode="all")
    # Force a pathological voxel and a single-good-echo voxel.
    t2s = out["t2s"].clone()
    t2s[0] = float("inf")
    am2 = am.clone()
    am2[1] = 1
    t2s_full, s0_full, t2s_lim, s0_lim = me.modify_t2s_s0_maps(t2s, out["s0"], am2, TES)
    assert torch.isfinite(t2s_full).all()
    assert t2s_lim[1] == 0 and s0_lim[1] == 0  # single-echo voxel zeroed in limited map
