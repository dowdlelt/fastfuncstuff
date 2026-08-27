"""Tests for ffs_locomoco multi-echo 3-D EPI (-me_3depi).

Synthetic: one shared partition-direction (PE-axis) displacement field, applied to
several echoes scaled by ``alpha_e ∝ TE_e``, each echo given its own T2*-like
contrast. The joint solve must (a) recover the per-echo scaling ratios (∝ TE),
(b) report a near-perfect linear-in-TE fit, and (c) correct every echo back onto a
common grid so its temporal variance collapses.
"""

import inspect

import numpy as np
import torch

from fastfuncstuff.processing.locomoco import (
    _fourier_shifter,
    _shift1d_windowed_sinc,
    _shift3d_axis,
    estimate_residual_flow_me_interecho,
    estimate_residual_flow_me_scaled,
    estimate_residual_flow_multiecho,
    make_raw_reference_me_result,
    optical_flow_lk_3d_multiecho,
    polish_me_result,
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


def test_raw_reference_builder_skips_flow_and_polishes():
    """`-backend qwarp` builds a median-of-raw reference WITHOUT a flow pass.

    `make_raw_reference_me_result` must (a) derive the fixed TE-ratio alpha + geometry,
    (b) hand back each raw echo as its own "corrected" series (so qwarp's reference is a
    plain median of raw) with a zero seed field, and (c) drive `polish_me_result(full=
    True)` to a finite per-echo field of the right shape. (Variance reduction is NOT
    asserted: with real residual motion a median-of-raw reference is blurry — this path
    is for already-moco'd input, where qwarp owns only the small residual distortion.)"""
    tes = [12.0, 30.0, 48.0]
    shifts = [0.0, 0.5, -0.4, 0.3, -0.5]
    datas, alpha_true = _make_multiecho(tes, shifts)
    nx, ny, nz, nt = datas[0].shape

    res = make_raw_reference_me_result(datas, tes, pe_axis=PE, slice_axis=PE, verbose=False)
    # Fixed TE-ratio scaling, no flow: alpha is exact (not estimated), field is zero.
    assert np.allclose(res.alpha.numpy(), alpha_true, atol=1e-5)
    assert float(res.w_field.abs().max()) == 0.0
    assert res.pe_axis == PE and len(res.per_echo) == len(tes)
    # The "corrected" series IS the raw echo (median of it becomes the qwarp reference).
    assert np.allclose(res.per_echo[0].corrected_series().numpy(), datas[0])

    polished = polish_me_result(
        res,
        minpatch=5,
        n_levels=1,
        iters=4,
        cost="ncc",
        optimizer="gn",
        full=True,
        raw_datas=datas,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert len(polished.per_echo) == len(tes)
    assert polished.w_field.shape == (nx, ny, nz, nt)
    assert torch.isfinite(polished.w_field).all()
    for corr in polished.per_echo:
        assert torch.isfinite(torch.as_tensor(corr.corrected_series())).all()


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


def test_want_corrected_false_skips_corrected_but_keeps_warp():
    """-no_corrected passes want_corrected=False: the warp/alpha are still produced but the
    corrected 4-D series is NOT materialized (pure waste when it won't be written). Refine,
    which builds corrected internally, must still work under the flag."""
    tes = [15.0, 45.0]
    shifts = [0.0, 0.6, -0.5, 0.4]
    datas, _ = _make_multiecho(tes, shifts)
    common = dict(
        pe_axis=PE,
        slice_axis=PE,
        learn_scaling=False,
        device=torch.device("cpu"),
        verbose=False,
    )

    on = estimate_residual_flow_multiecho(datas, tes, want_corrected=True, **common)
    off = estimate_residual_flow_multiecho(datas, tes, want_corrected=False, **common)

    # The estimate itself is identical — only the corrected output differs.
    assert np.allclose(on.w_field.numpy(), off.w_field.numpy(), atol=1e-6)
    assert np.allclose(on.alpha.numpy(), off.alpha.numpy(), atol=1e-6)
    for res in on.per_echo:
        assert res.corrected_nifti is not None
    for res in off.per_echo:
        assert res.corrected_nifti is None

    # Refine still converges with the flag off (it builds its own corrected internally).
    ref = estimate_residual_flow_multiecho(
        datas, tes, want_corrected=False, refine_rounds=2, **common
    )
    assert ref.per_echo[0].corrected_nifti is None
    assert float(np.abs(ref.w_field.numpy()).max()) > 0


def test_windowed_sinc_dc_scalar_field_and_integer_shift():
    """Resampler invariants: a flat volume keeps its intensity (DC-normalised weights, no
    brightness ripple), a scalar shift equals a constant-field shift, and an integer shift
    is an exact border-clamped roll (only the j=0 tap survives)."""
    # DC: warping a flat region by a sub-voxel amount must not modulate intensity.
    flat = torch.full((1, 12, 12, 12), 3.0)
    out = _shift1d_windowed_sinc(flat, 0.37, axis=PE, radius=5)
    assert torch.allclose(out, torch.full_like(out, 3.0), atol=1e-4)

    # Scalar shift == uniform-field shift of the same value.
    base = torch.from_numpy(_phantom())[None]
    field = torch.full_like(base, 0.3)
    assert torch.allclose(
        _shift1d_windowed_sinc(base, 0.3, PE, radius=3),
        _shift1d_windowed_sinc(base, field, PE, radius=3),
        atol=1e-6,
    )

    # Integer shift = exact roll with border clamp (sinc(integer)=0 except at 0).
    n = base.shape[PE + 1]
    rolled = _shift1d_windowed_sinc(base, 1.0, PE, radius=4)
    ref = base.index_select(PE + 1, torch.arange(1, n + 1).clamp(max=n - 1))
    assert torch.allclose(rolled, ref, atol=1e-5)


def test_windowed_sinc_beats_trilinear_on_subvoxel():
    """Against a sinc-exact (Fourier) sub-voxel shift of a smooth phantom, the Lanczos
    resampler tracks the truth markedly better than trilinear — the fidelity that keeps
    sub-voxel signal in the corrected output instead of blurring it away."""
    base = torch.from_numpy(_phantom(nx=32, ny=32, nz=32))[None]
    s = 0.3
    gt = _fourier_shifter(base, PE, pad=8)(s)  # sinc-exact reference shift
    lan = _shift1d_windowed_sinc(base, s, PE, radius=5)
    lin = _shift3d_axis(base, s, PE, mode="bilinear")
    # Compare on the interior (border handling differs between samplers).
    sl = (slice(None), slice(4, -4), slice(4, -4), slice(4, -4))
    err_lan = float((lan[sl] - gt[sl]).abs().mean())
    err_lin = float((lin[sl] - gt[sl]).abs().mean())
    assert err_lan < 0.6 * err_lin, (err_lan, err_lin)


def test_lanczos_correction_preserves_signal_better():
    """The payoff of the resampler is the CORRECTED output, not the estimate: correcting a
    distorted volume (undo a known sub-voxel shift) with Lanczos leaves less residual against
    the true undistorted signal than trilinear — i.e. it doesn't blur the data it realigns."""
    base = torch.from_numpy(_phantom(nx=32, ny=32, nz=32))[None]
    s = 0.3
    distorted = _fourier_shifter(base, PE, pad=8)(s)  # a faithful sub-voxel distortion
    corr_lan = _shift3d_axis(distorted, -s, PE, mode="lanczos", radius=5)
    corr_lin = _shift3d_axis(distorted, -s, PE, mode="bilinear")
    sl = (slice(None), slice(4, -4), slice(4, -4), slice(4, -4))
    err_lan = float((corr_lan[sl] - base[sl]).abs().mean())
    err_lin = float((corr_lin[sl] - base[sl]).abs().mean())
    assert err_lan < 0.6 * err_lin, (err_lan, err_lin)


def test_lanczos_flow_matches_bilinear_estimate_and_runs():
    """The Lanczos warp is a drop-in through the flow estimator: it produces a comparable
    shared field (the estimate is set by the pooling window, not the interpolator) — this
    guards the plumbing, not a magnitude claim."""
    tes = [12.0, 30.0, 48.0]
    shifts = [0.0, 0.18, -0.14, 0.2, -0.12, 0.16]
    datas, _ = _make_multiecho(tes, shifts)
    common = dict(
        pe_axis=PE,
        slice_axis=PE,
        learn_scaling=False,
        n_levels=3,
        n_iters=8,
        device=torch.device("cpu"),
        verbose=False,
    )
    r_lin = estimate_residual_flow_multiecho(datas, tes, warp_interp="bilinear", **common)
    r_lan = estimate_residual_flow_multiecho(
        datas, tes, warp_interp="lanczos", warp_radius=3, **common
    )
    core = datas[0][..., 1] > np.percentile(datas[0][..., 1], 60)
    mag_lin = float(np.abs(r_lin.w_field.numpy()[core]).mean())
    mag_lan = float(np.abs(r_lan.w_field.numpy()[core]).mean())
    assert abs(mag_lan - mag_lin) < 0.15 * max(mag_lin, 1e-6)  # same ballpark, plumbing works


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
    w_be, _conf = xcorr_search_flow_3d_multiecho(
        fixed, moving, alpha, PE, max_shift=3.0, trial_step=0.05
    )
    w = w_be[0]
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


def test_interecho_aligns_stack_to_anchor():
    """Inter-echo mode: echo n shifted by (n-1)·h_t from echo 1; recover & align to echo 1."""
    tes = [10.0, 20.0, 30.0]  # equal ΔTE → alpha steps [0, 1, 2]
    base = _phantom()
    contrast = [float(np.exp(-t / 40.0)) for t in tes]
    # Per-TR true echo1→echo2 shift h_t; echo n shifted by (n-1)·h_t (linear in TE).
    hs = [0.0, 0.6, -0.5, 0.4, -0.6, 0.3, -0.4, 0.5]
    datas = []
    for step, c in enumerate(contrast):
        series = np.zeros((*base.shape, len(hs)), np.float32)
        for t, h in enumerate(hs):
            series[..., t] = (
                c * _shift3d_axis(torch.from_numpy(base)[None], float(step * h), PE)[0].numpy()
            )
        datas.append(series)

    res = estimate_residual_flow_me_interecho(
        datas,
        tes,
        pe_axis=PE,
        slice_axis=PE,
        backend="xcorr",
        trial_step=0.1,
        device=torch.device("cpu"),
        verbose=False,
    )
    # alpha counts steps from echo 1: [0, 1, 2]; echo 1 is the untouched anchor.
    assert np.allclose(res.alpha.numpy(), [0.0, 1.0, 2.0], atol=1e-6)
    assert np.allclose(res.per_echo[0].pe_displacement().numpy(), 0.0, atol=1e-6)
    # w[...,t] recovers the echo1→echo2 step h_t (median over brain core).
    core = base > np.percentile(base, 40)
    for t in (1, 2, 4):
        rec = float(np.median(res.w_field.numpy()[core, t]))
        assert abs(rec - (-hs[t])) < 0.15, (t, rec, -hs[t])

    # Corrected later echoes land on echo 1's frame: cross-echo disagreement drops.
    def _spread(series_list, t):  # std across echoes (contrast-normalised) at frame t
        fr = np.stack([s[core, t] / np.mean(np.abs(s[core, t])) for s in series_list], 0)
        return float(fr.std(0).mean())

    corr_list = [r.corrected_series().numpy() for r in res.per_echo]
    for t in (2, 4):
        assert _spread(corr_list, t) < _spread(datas, t)


def test_special_multiecho_paths_default_to_lanczos():
    """Inter-echo and selected-echo paths must not regress to blurry linear sampling."""
    from fastfuncstuff.processing.locomoco import refine_interecho_temporally

    for fn in (
        estimate_residual_flow_me_interecho,
        estimate_residual_flow_me_scaled,
        refine_interecho_temporally,
    ):
        assert inspect.signature(fn).parameters["warp_interp"].default == "lanczos"


def _interecho_stack(tes, hs, decay=12.0):
    """Echo n = base shifted by (n−1)·h_t along PE, dimmed by a STEEP T2* decay.

    A short decay constant is the point: consecutive echoes then differ in brightness by
    far more than the displacement moves them, which is the regime that breaks LK.
    """
    base = _phantom()
    datas = []
    for step, te in enumerate(tes):
        c = float(np.exp(-te / decay))
        series = np.zeros((*base.shape, len(hs)), np.float32)
        for t, h in enumerate(hs):
            series[..., t] = (
                c * _shift3d_axis(torch.from_numpy(base)[None], float(step * h), PE)[0].numpy()
            )
        datas.append(series)
    return base, datas


def test_interecho_flow_matches_contrast_and_clamps_step():
    """Cross-TE LK diverges on raw intensities; matching + the step clamp keep it sane."""
    tes = [10.0, 20.0, 30.0, 40.0]
    hs = [0.0, 0.5, -0.4, 0.3]
    base, datas = _interecho_stack(tes, hs)
    common = dict(
        pe_axis=PE,
        slice_axis=PE,
        backend="flow",
        n_iters=8,
        max_shift=2.0,
        automask=False,
        device=torch.device("cpu"),
        verbose=False,
    )
    core = base > np.percentile(base, 40)

    matched = estimate_residual_flow_me_interecho(datas, tes, match="localnorm", **common)
    raw = estimate_residual_flow_me_interecho(datas, tes, match="none", **common)

    # The clamp is on the echo1→echo2 step field, in voxels — it holds for both.
    assert float(matched.w_field.abs().max()) <= 2.0 + 1e-5
    assert float(raw.w_field.abs().max()) <= 2.0 + 1e-5
    # Matching is what makes the recovered step track the truth; unmatched, the T2* step
    # dominates the residual and the field is driven by decay, not geometry.
    err_m = np.array(
        [abs(float(np.median(matched.w_field.numpy()[core, t])) + hs[t]) for t in (1, 2, 3)]
    )
    err_r = np.array(
        [abs(float(np.median(raw.w_field.numpy()[core, t])) + hs[t]) for t in (1, 2, 3)]
    )
    assert err_m.mean() < 0.15, err_m
    assert err_m.mean() < err_r.mean(), (err_m, err_r)


def test_interecho_temporal_refine_recovers_the_echo1_anchor_offset():
    """The refine pass must recover the leftover the inter-echo anchor leaves behind.

    Echo e truly sits at TE_e·g. Inter-echo removes the DIFFERENCES (TE_e − TE_1)·g, so
    every echo comes out carrying the same TE_1·g — a FLAT (common to all echoes)
    residual, which is why the refine defaults to flat scaling. Here the truth is built
    that way: a per-frame shift common to the whole stack on top of the TE ladder. Only
    the temporal pass can see it, and the composed result must show it on echo 1 too.
    """
    from fastfuncstuff.processing.locomoco import refine_interecho_temporally

    tes = [10.0, 20.0, 30.0]
    hs = [0.0, 0.5, -0.4, 0.3, -0.5]
    common_shift = [0.0, 0.4, 0.4, -0.3, -0.3]  # echo-1 wiggle, shared by the whole stack
    base = _phantom()
    datas = []
    for step, te in enumerate(tes):
        c = float(np.exp(-te / 40.0))
        series = np.zeros((*base.shape, len(hs)), np.float32)
        for t, (h, s) in enumerate(zip(hs, common_shift, strict=True)):
            series[..., t] = (
                c * _shift3d_axis(torch.from_numpy(base)[None], float(step * h + s), PE)[0].numpy()
            )
        datas.append(series)

    kw = dict(
        pe_axis=PE,
        slice_axis=PE,
        backend="xcorr",
        trial_step=0.1,
        automask=False,
        device=torch.device("cpu"),
        verbose=False,
    )
    ie = estimate_residual_flow_me_interecho(datas, tes, **kw)
    refined = refine_interecho_temporally(
        ie,
        datas,
        tes,
        PE,
        PE,
        refine_rounds=1,
        ref_mode="mean",
        **{k: v for k, v in kw.items() if k not in ("pe_axis", "slice_axis")},
    )

    # Inter-echo leaves echo 1 alone by construction; the temporal pass must not.
    assert float(ie.per_echo[0].pe_displacement().abs().max()) == 0.0
    core = base > np.percentile(base, 40)
    d1 = refined.per_echo[0].pe_displacement().numpy()
    for t in (1, 3):
        assert abs(float(np.median(d1[core, t])) + common_shift[t]) < 0.2, (
            t,
            np.median(d1[core, t]),
        )

    # Every echo's corrected series is flatter in time than the inter-echo-only result.
    for j in range(len(tes)):
        before = float(
            np.asarray(ie.per_echo[j].corrected_series().numpy())[core].std(axis=-1).mean()
        )
        after = float(
            np.asarray(refined.per_echo[j].corrected_series().numpy())[core].std(axis=-1).mean()
        )
        assert after < before, (j, before, after)

    # ...and 'flat' is the right default for it: a TE-proportional model cannot represent
    # a common-mode leftover and under-corrects the anchor echo it matters most for.
    te_scaled = refine_interecho_temporally(
        ie,
        datas,
        tes,
        PE,
        PE,
        refine_rounds=1,
        ref_mode="mean",
        scaling="te",
        **{k: v for k, v in kw.items() if k not in ("pe_axis", "slice_axis")},
    )
    d1_te = te_scaled.per_echo[0].pe_displacement().numpy()
    err_flat = np.mean([abs(float(np.median(d1[core, t])) + common_shift[t]) for t in (1, 3)])
    err_te = np.mean([abs(float(np.median(d1_te[core, t])) + common_shift[t]) for t in (1, 3)])
    assert err_flat < err_te, (err_flat, err_te)


def test_interecho_refine_keeps_the_te_scaling_coupled():
    """The refine pass must not let the echoes drift into independent per-echo warps.

    Both passes enforce "one shared field, scaled per echo", but with different anchors
    (inter-echo ∝ TE−TE₁, temporal ∝ TE), so the composed per-echo warps must lie exactly
    in the 2-D affine-in-TE family — that is the invariant, not proportionality. Also
    checks the refine pass really is a joint solve over EVERY echo: its own field is a
    single alpha·w, so the ratio between any two echoes' contributions is constant.
    """
    from fastfuncstuff.processing.locomoco import (
        _affine_in_te_r2,
        refine_interecho_temporally,
    )

    tes = [10.0, 22.0, 34.0, 46.0]  # 4 echoes: 2 more than the affine model has parameters
    hs = [0.0, 0.5, -0.4, 0.3, -0.5]
    common_shift = [0.0, 0.4, 0.3, -0.3, -0.2]
    base = _phantom()
    datas = []
    for step, te in enumerate(tes):
        c = float(np.exp(-te / 40.0))
        series = np.zeros((*base.shape, len(hs)), np.float32)
        for t, (h, s) in enumerate(zip(hs, common_shift, strict=True)):
            series[..., t] = (
                c * _shift3d_axis(torch.from_numpy(base)[None], float(step * h + s), PE)[0].numpy()
            )
        datas.append(series)

    kw = dict(backend="xcorr", trial_step=0.1, automask=False, device=torch.device("cpu"))
    ie = estimate_residual_flow_me_interecho(
        datas, tes, pe_axis=PE, slice_axis=PE, verbose=False, **kw
    )
    refined = refine_interecho_temporally(
        ie, datas, tes, PE, PE, refine_rounds=1, ref_mode="mean", verbose=False, **kw
    )

    # The composed per-echo warps are still ONE field scaled by echo time (affine family).
    totals = torch.stack([r.pe_displacement() for r in refined.per_echo], 0)
    r2 = _affine_in_te_r2(totals, torch.tensor(tes))
    assert r2 > 0.999, r2
    assert abs(refined.linearity_r2 - r2) < 1e-9  # ...and that is what gets reported

    # A per-echo-independent warp stack would NOT satisfy it — guard against the test
    # passing for a trivial reason (e.g. an all-zero or 2-echo field).
    scrambled = totals.clone()
    scrambled[2] = scrambled[2] * 3.0
    assert _affine_in_te_r2(scrambled, torch.tensor(tes)) < 0.99
    assert float(totals.abs().max()) > 0.1


def test_erode_6conn_peels_one_voxel_shell():
    from fastfuncstuff.processing.mask import _erode_6conn

    m = torch.zeros(9, 9, 9, dtype=torch.bool)
    m[2:7, 2:7, 2:7] = True  # solid 5³ block
    e1 = _erode_6conn(m, 1)
    assert bool(e1[4, 4, 4])  # core survives
    assert not bool(e1[2, 4, 4])  # face shell peeled
    assert int(e1.sum()) == 3 * 3 * 3  # 5³ eroded by 1 → 3³


def _dropout_echoes(tes, h, dropout_slabs):
    """Bright brain-blob echoes shifted by (n-1)·h along PE, with per-echo dropout slabs.

    ``dropout_slabs[e]`` is an (x0, x1) block zeroed in echo e (simulating T2* signal loss).
    """
    nx, ny, nz = 40, 40, 12
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    r = ((x - 20) / 15.0) ** 2 + ((y - 20) / 15.0) ** 2 + ((z - 6) / 5.0) ** 2
    tex = np.sin(x / 4.0) * np.cos(y / 3.0) + 0.5 * np.sin((x + z) / 3.0)
    base = ((r < 1.0) * (2.0 + 0.4 * tex)).astype(np.float32)  # bright blob, ~0 outside
    T = 6
    datas = []
    for step, te in enumerate(tes):
        c = float(np.exp(-te / 40.0))
        series = np.zeros((nx, ny, nz, T), np.float32)
        for t in range(T):
            series[..., t] = (
                c * _shift3d_axis(torch.from_numpy(base)[None], float(step * h), PE)[0].numpy()
            )
        if dropout_slabs.get(step) is not None:
            x0, x1 = dropout_slabs[step]
            series[x0:x1] = 0.0  # signal dropout
        datas.append(series)
    return datas


def test_interecho_automask_gates_dropout():
    """Deep dropout (echo 2 AND 3 gone) must be gated to ~0, not railed to max_shift."""
    tes = [10.0, 20.0, 30.0]
    h = 0.8
    # echo 2 loses x∈[6,12]; echo 3 loses the larger x∈[4,14] (⊇ echo 2's) — so x∈[6,12]
    # is deep dropout (both gone) and must be gated; echo 1 keeps full signal everywhere.
    slabs = {1: (6, 12), 2: (4, 14)}
    datas = _dropout_echoes(tes, h, slabs)
    common = dict(
        pe_axis=PE,
        slice_axis=PE,
        backend="xcorr",
        trial_step=0.1,
        max_shift=3.0,
        device=torch.device("cpu"),
        verbose=False,
    )

    railed = estimate_residual_flow_me_interecho(datas, tes, automask=False, **common)
    gated = estimate_residual_flow_me_interecho(datas, tes, automask=True, **common)

    brain = datas[0][..., 0] > 0.5  # echo-1 signal region (full blob)
    deep = np.zeros_like(brain)
    deep[6:12] = True
    deep &= brain  # deep-dropout voxels that are still inside the blob
    # Un-masked: the search rails in the dropout (large |w|). Masked: gated toward 0.
    assert float(np.abs(gated.w_field.numpy()[deep]).mean()) < 0.2
    assert float(np.abs(gated.w_field.numpy()[deep]).mean()) < 0.5 * float(
        np.abs(railed.w_field.numpy()[deep]).mean()
    )


def test_flat_scaling_shifts_all_echoes_equally():
    """-me_flat_scaling: alpha=1 for all echoes (TE-independent), still pooled."""
    tes = [12.0, 30.0, 48.0]
    # Build echoes that all shift by the SAME amount (flat, not TE-scaled).
    base = _phantom()
    shifts = [0.0, 0.6, -0.5, 0.4, -0.6, 0.3]
    contrast = [float(np.exp(-t / 40.0)) for t in tes]
    datas = []
    for c in contrast:
        series = np.zeros((*base.shape, len(shifts)), np.float32)
        for t, s in enumerate(shifts):
            series[..., t] = (
                c * _shift3d_axis(torch.from_numpy(base)[None], float(s), PE)[0].numpy()
            )
        datas.append(series)
    res = estimate_residual_flow_multiecho(
        datas,
        tes,
        pe_axis=PE,
        slice_axis=PE,
        backend="xcorr",
        ref_mode="median",
        trial_step=0.1,
        refine_rounds=2,
        flat_scaling=True,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert np.allclose(res.alpha.numpy(), [1.0, 1.0, 1.0], atol=1e-6)
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


def test_cli_want_pcs_written_with_no_warp(tmp_path):
    """-want_pcs must emit {stem}_locomoco_pcs.1D in the -me_3depi path even with -no_warp.

    Regression: the multi-echo path returned before ever handling -want_pcs, so the PCs
    (derived from the in-memory shared field) were silently dropped when the warp itself
    was not written.
    """
    import nibabel as nib

    from fastfuncstuff.cli.locomoco import main

    tes = [7.61, 21.71, 35.81]
    n_frames = 7
    datas, _ = _make_multiecho(tes, [0.0, 0.5, -0.4, 0.3, -0.5, 0.2, 0.1])
    paths = []
    for e, series in enumerate(datas):
        p = tmp_path / f"e{e + 1}.nii.gz"
        nib.save(nib.Nifti1Image(series, np.eye(4)), str(p))
        paths.append(str(p))
    stem = str(tmp_path / "out")

    rc = main(
        [
            "-input",
            *paths,
            "-prefix",
            stem,
            # -me_3depi corrects the PARTITION direction, now spelled -pe_dir2
            "-pe_dir2",
            "IS",
            "-backend",
            "flow",
            "-me_3depi",
            "-echo_times",
            *[str(t) for t in tes],
            "-is_3dacq",
            "-me_fixed_scaling",
            "-device",
            "cpu",
            "-no_warp",
            "-no_movie",
            "-no_corrected",
            "-no_flow",
            "-want_pcs",
            "4",
            "-levels",
            "2",
            "-iters",
            "2",
            "-refine",
            "0",
        ]
    )
    assert rc == 0
    pcs = tmp_path / "out_locomoco_pcs.1D"
    assert pcs.exists(), "want_pcs produced no .1D in the -me_3depi/-no_warp path"
    assert not list(tmp_path.glob("out*_warp*"))  # warp genuinely skipped, PCs still written
    arr = np.loadtxt(pcs)
    assert arr.ndim == 2 and arr.shape[0] == n_frames  # one row per frame
    assert 1 <= arr.shape[1] <= 4  # up to n_pcs columns


def test_polish_single_pass_is_sharper_than_double_resample():
    """The qwarp polish registers the CORRECTED series, so its own warped output has been
    resampled twice — the estimator's pass plus qwarp's. Given ``raw_datas`` it must
    instead warp the RAW echoes once by the composed total ``w + r``, which keeps edge
    energy the second pass cannot restore, and makes the returned series describe the
    same transform as the saved field.

    Temporal variance is deliberately NOT the assertion: the double-resampled series has
    the LOWER variance of the two, because interpolation blur suppresses variance without
    correcting anything. Sharpness is what separates them.
    """
    tes = [12.0, 30.0]
    shifts = [0.0, 0.7, -0.6, 0.4, -0.5]
    datas, _ = _make_multiecho(tes, shifts)
    res = estimate_residual_flow_multiecho(
        datas,
        tes,
        pe_axis=PE,
        slice_axis=PE,
        warp_interp="bilinear",  # the default, and the pass whose blur we refuse to keep
        device=torch.device("cpu"),
        verbose=False,
    )
    common = dict(
        minpatch=5,
        n_levels=1,
        iters=4,
        cost="ncc",
        optimizer="gn",
        full=False,
        device=torch.device("cpu"),
        verbose=False,
    )
    double = polish_me_result(res, raw_datas=None, **common)
    single = polish_me_result(res, raw_datas=datas, **common)

    # Same transform either way, so the saved fields must agree.
    assert torch.allclose(double.w_field, single.w_field, atol=1e-5)

    def sharpness(r):
        v = torch.as_tensor(r.per_echo[0].corrected_series()).float().numpy()
        return float(np.abs(np.diff(np.median(v, axis=3), axis=PE)).mean())

    assert sharpness(single) > sharpness(double)


def test_qwarp_refine_rebuilds_the_template_and_moves_the_field():
    """`-refine` under `-backend qwarp` must actually re-solve, not silently no-op.

    The first template is a reduction of the RAW echoes, still carrying the distortion,
    so it is blurred by it and biases the field low. A refine pass rebuilds the template
    from the corrected series and re-solves from seed 0, which has to land somewhere
    different — an identical field would mean the loop never ran.
    """
    tes = [12.0, 30.0, 48.0]
    datas, _ = _make_multiecho(tes, [0.0, 0.5, -0.4, 0.3, -0.5])
    res = make_raw_reference_me_result(datas, tes, pe_axis=PE, slice_axis=PE, verbose=False)
    common = dict(
        minpatch=5,
        n_levels=1,
        iters=4,
        cost="ncc",
        optimizer="gn",
        full=True,
        raw_datas=datas,
        device=torch.device("cpu"),
        verbose=False,
    )
    once = polish_me_result(res, refine=0, **common)
    twice = polish_me_result(res, refine=1, **common)
    assert twice.w_field.shape == once.w_field.shape
    assert torch.isfinite(twice.w_field).all()
    assert not torch.allclose(once.w_field, twice.w_field), "the refine pass did nothing"


def test_qwarp_ref_mode_changes_the_template():
    """The template reduction was pinned to the median; `-ref` has to reach it now."""
    tes = [12.0, 30.0, 48.0]
    datas, _ = _make_multiecho(tes, [0.0, 0.5, -0.4, 0.3, -0.5])
    res = make_raw_reference_me_result(datas, tes, pe_axis=PE, slice_axis=PE, verbose=False)
    common = dict(
        minpatch=5,
        n_levels=1,
        iters=4,
        cost="ncc",
        optimizer="gn",
        full=True,
        raw_datas=datas,
        device=torch.device("cpu"),
        verbose=False,
    )
    med = polish_me_result(res, ref_mode="median", **common)
    mean = polish_me_result(res, ref_mode="mean", **common)
    assert not torch.allclose(med.w_field, mean.w_field), "ref_mode never reached the template"
