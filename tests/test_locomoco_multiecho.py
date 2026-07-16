"""Tests for ffs_locomoco multi-echo 3-D EPI (-me_3depi).

Synthetic: one shared partition-direction (PE-axis) displacement field, applied to
several echoes scaled by ``alpha_e ∝ TE_e``, each echo given its own T2*-like
contrast. The joint solve must (a) recover the per-echo scaling ratios (∝ TE),
(b) report a near-perfect linear-in-TE fit, and (c) correct every echo back onto a
common grid so its temporal variance collapses.
"""

import numpy as np
import torch

from fastfuncstuff.processing.locomoco import (
    _shift3d_axis,
    estimate_residual_flow_me_scaled,
    estimate_residual_flow_multiecho,
    optical_flow_lk_3d_multiecho,
    xcorr_search_flow_3d_multiecho,
)

PE = 2  # partition/slice axis being corrected


def _phantom(nx=24, ny=24, nz=10):
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    base = np.sin(x / 4.0) * np.cos(y / 3.5) + 0.5 * np.sin((x + z) / 3.0) + 0.3 * np.cos(z / 2.0)
    return (base - base.min() + 1.0).astype(np.float32)


def _make_multiecho(tes, shifts):
    """Build E moving series: echo e frame t = base shifted by alpha_e·s_t along PE."""
    base = _phantom()
    alpha = np.asarray(tes, np.float32) / tes[0]
    contrast = np.exp(-np.asarray(tes, np.float32) / 40.0)  # T2*-like per-echo dimming
    datas = []
    for a, c in zip(alpha, contrast, strict=True):
        series = np.zeros((*base.shape, len(shifts)), np.float32)
        for t, s in enumerate(shifts):
            shifted = _shift3d_axis(torch.from_numpy(base)[None], float(a * s), PE)[0].numpy()
            series[..., t] = c * shifted
        datas.append(series)
    return datas, alpha


def test_pooled_kernel_recovers_shared_shift():
    base = _phantom()
    tes = [12.0, 30.0, 48.0]
    alpha = torch.tensor([t / tes[0] for t in tes])
    fixed = [torch.from_numpy(base)[None] for _ in tes]
    moving = [_shift3d_axis(torch.from_numpy(base)[None], float(a * 0.7), PE) for a in alpha]
    w = optical_flow_lk_3d_multiecho(fixed, moving, alpha, PE, n_levels=3, n_iters=12)[0]
    # Shared field w such that echo e shifted by alpha_e·w undoes alpha_e·0.7 → w ≈ -0.7.
    core = base > np.percentile(base, 40)
    assert abs(float(w[core].median()) + 0.7) < 0.15


def test_multiecho_recovers_scaling_and_corrects():
    tes = [12.0, 30.0, 48.0]
    shifts = [0.0, 0.5, -0.4, 0.3, -0.5, 0.2]
    datas, alpha_true = _make_multiecho(tes, shifts)

    res = estimate_residual_flow_multiecho(
        datas,
        tes,
        pe_axis=PE,
        slice_axis=PE,
        n_levels=3,
        n_iters=8,
        device=torch.device("cpu"),
        verbose=False,
    )

    # Learned scaling (÷echo1) tracks the true TE ratios.
    a = res.alpha.numpy()
    assert np.allclose(a, alpha_true, rtol=0.15, atol=0.15), (a, alpha_true)
    # ...and the alpha-vs-TE relationship is essentially linear.
    assert res.linearity_r2 > 0.99

    # Every echo's temporal variance collapses after correction (frames realigned).
    brain = datas[0][..., 0] > np.percentile(datas[0][..., 0], 50)
    for j, corr in enumerate(res.per_echo):
        before = float(np.asarray(datas[j])[brain].std(axis=-1).mean())
        after = float(corr.corrected_series().numpy()[brain].std(axis=-1).mean())
        assert after < 0.5 * before, (j, before, after)


def test_refine_grows_magnitude_toward_truth():
    """Reference-refinement (the -refine knob) should push |w| UP toward the true shift.

    The initial reference is the motion-blurred frame mean, which biases displacement low;
    rebuilding it from the corrected series recovers more of the true magnitude.
    """
    tes = [12.0, 36.0]
    shifts = [0.0, 0.9, -0.8, 0.7, -0.9, 0.6, -0.5, 0.8]
    datas, _ = _make_multiecho(tes, shifts)
    common = dict(
        pe_axis=PE, slice_axis=PE, n_levels=3, n_iters=6, device=torch.device("cpu"), verbose=False
    )
    r0 = estimate_residual_flow_multiecho(datas, tes, refine_rounds=0, **common)
    r2 = estimate_residual_flow_multiecho(datas, tes, refine_rounds=3, **common)
    # True per-frame |shift| for echo 1 (alpha_1=1) is |shifts|; measure recovered vs it.
    true_mag = float(np.mean(np.abs([s for s in shifts if s != 0.0])))
    core = datas[0][..., 1] > np.percentile(datas[0][..., 1], 60)
    mag0 = float(np.abs(r0.w_field.numpy()[core]).mean())
    mag2 = float(np.abs(r2.w_field.numpy()[core]).mean())
    assert mag2 > mag0  # refinement grows the recovered displacement
    assert mag2 > 0.6 * true_mag  # ...and gets most of the way to truth


def test_fixed_scaling_uses_te_ratio():
    tes = [15.0, 45.0]
    shifts = [0.0, 0.6, -0.5, 0.4]
    datas, _ = _make_multiecho(tes, shifts)
    res = estimate_residual_flow_multiecho(
        datas,
        tes,
        pe_axis=PE,
        slice_axis=PE,
        learn_scaling=False,
        device=torch.device("cpu"),
        verbose=False,
    )
    # Fixed scaling pins alpha exactly to the TE ratio [1, 3].
    assert np.allclose(res.alpha.numpy(), [1.0, 3.0], atol=1e-5)
    brain = datas[0][..., 0] > np.percentile(datas[0][..., 0], 50)
    after = float(res.per_echo[1].corrected_series().numpy()[brain].std(axis=-1).mean())
    before = float(np.asarray(datas[1])[brain].std(axis=-1).mean())
    assert after < 0.6 * before


def test_pooled_xcorr_kernel_recovers_shared_shift():
    """Shared-parameter searchlight: all echoes trial-shifted by alpha_e·s, one field out."""
    base = _phantom()
    tes = [12.0, 30.0, 48.0]
    alpha = torch.tensor([t / tes[0] for t in tes])
    contrast = [float(np.exp(-t / 40.0)) for t in tes]
    fixed = [torch.from_numpy(c * base)[None] for c in contrast]
    # Echo e is shifted by alpha_e · 0.35 (echo-1 scale field = 0.35).
    moving = [
        _shift3d_axis(torch.from_numpy(c * base)[None], float(a * 0.35), PE)
        for a, c in zip(alpha, contrast, strict=True)
    ]
    w = xcorr_search_flow_3d_multiecho(fixed, moving, alpha, PE, max_shift=3.0, trial_step=0.05)[0]
    core = base > np.percentile(base, 40)
    # Correcting shift is -0.35 (echo-1 scale); pooled search should recover it.
    assert abs(float(w[core].median()) + 0.35) < 0.1


def test_fixed_scaling_xcorr_pools_all_echoes():
    """-me_fixed_scaling + xcorr uses the pooled searchlight and corrects every echo."""
    tes = [12.0, 30.0, 48.0]
    shifts = [0.0, 0.5, -0.4, 0.3, -0.5, 0.2]
    datas, _ = _make_multiecho(tes, shifts)
    res = estimate_residual_flow_multiecho(
        datas,
        tes,
        pe_axis=PE,
        slice_axis=PE,
        backend="xcorr",
        ref_mode="median",
        trial_step=0.1,
        refine_rounds=2,
        learn_scaling=False,  # enforce TE-linearity → pooled search
        device=torch.device("cpu"),
        verbose=False,
    )
    assert np.allclose(res.alpha.numpy(), [1.0, 2.5, 4.0], atol=1e-5)
    assert float(res.w_field.abs().max()) > 0.1
    brain = datas[0][..., 0] > np.percentile(datas[0][..., 0], 50)
    for j, corr in enumerate(res.per_echo):
        before = float(np.asarray(datas[j])[brain].std(axis=-1).mean())
        after = float(corr.corrected_series().numpy()[brain].std(axis=-1).mean())
        assert after < 0.6 * before, (j, before, after)


def test_scaled_from_one_echo_matches_te_ratio():
    """Estimate on the LAST echo, scale to the rest by TE ratio — alpha is exact, all corrected."""
    tes = [12.0, 30.0, 48.0]
    shifts = [0.0, 0.5, -0.4, 0.3, -0.5, 0.2]
    datas, _ = _make_multiecho(tes, shifts)
    res = estimate_residual_flow_me_scaled(
        datas,
        tes,
        estimate_idx=2,  # last echo (largest, easiest-to-detect shifts)
        pe_axis=PE,
        slice_axis=PE,
        backend="xcorr",
        ref_mode="median",
        trial_step=0.1,
        refine_rounds=2,
        device=torch.device("cpu"),
        verbose=False,
    )
    # Scaling is applied, not fitted → exact TE ratio and r²=1.
    assert np.allclose(res.alpha.numpy(), [1.0, 2.5, 4.0], atol=1e-5)
    assert res.linearity_r2 == 1.0
    # w is stored echo-1-scaled; the last echo's own field is 4× larger.
    assert float(res.w_field.abs().max()) > 0.1
    brain = datas[0][..., 0] > np.percentile(datas[0][..., 0], 50)
    for j, corr in enumerate(res.per_echo):
        before = float(np.asarray(datas[j])[brain].std(axis=-1).mean())
        after = float(corr.corrected_series().numpy()[brain].std(axis=-1).mean())
        assert after < 0.6 * before, (j, before, after)


def test_xcorr_backend_with_refine():
    """xcorr backend must run the shared-field solve AND honour -refine (was flow-only)."""
    tes = [12.0, 30.0, 48.0]
    shifts = [0.0, 0.5, -0.4, 0.3, -0.5, 0.2]
    datas, alpha_true = _make_multiecho(tes, shifts)
    res = estimate_residual_flow_multiecho(
        datas,
        tes,
        pe_axis=PE,
        slice_axis=PE,
        backend="xcorr",
        ref_mode="median",
        trial_step=0.1,
        refine_rounds=2,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert np.allclose(res.alpha.numpy(), alpha_true, rtol=0.15, atol=0.15)
    assert res.linearity_r2 > 0.99
    assert float(res.w_field.abs().max()) > 0.1  # not the near-zero, refine-skipped result
    brain = datas[0][..., 0] > np.percentile(datas[0][..., 0], 50)
    for j, corr in enumerate(res.per_echo):
        before = float(np.asarray(datas[j])[brain].std(axis=-1).mean())
        after = float(corr.corrected_series().numpy()[brain].std(axis=-1).mean())
        assert after < 0.6 * before, (j, before, after)
