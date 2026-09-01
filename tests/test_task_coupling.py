"""Task-coupling diagnostic: does it separate BOLD-as-motion from real motion?

Three statistics got replaced during review, and each replacement has a test that
pins WHY, so none of them can quietly come back:

* R2 -> signed r: a block design is one frequency, so a random-phase surrogate scores
  R2 ~ 0.5 against it.
* every per-voxel null -> none at all: a surrogate's p95 |r| equals the TRUE |r| for a
  block design, so no per-voxel significance exists to be had.
* sign consistency -> the kappa slope test: the field's sign flips at an edge and
  under negative BOLD, so sign alone is not evidence.
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.stats.task_coupling import (
    co_location,
    contamination_slope,
    default_polort,
    design_notch_bins,
    enrichment_curve,
    filter_task_band,
    format_task_coupling_report,
    notch_basis,
    pe_gradient,
    responding_mask,
    task_coupling,
    task_enrichment,
)


def _block_design(n_t: int, block: int = 10) -> torch.Tensor:
    box = ((np.arange(n_t) // block) % 2).astype(np.float64)
    k = np.exp(-np.arange(0, 12) / 3.0)
    conv = np.convolve(box, k / k.sum())[:n_t]
    return torch.tensor(conv - conv.mean())[:, None]


def _field(n_t: int, shape=(6, 6, 4), *, task=None, amp=0.0, drift=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(*shape, n_t, generator=g, dtype=torch.float64) * 0.1
    if amp and task is not None:
        f += amp * task[:, 0]
    if drift:
        t = torch.linspace(-1, 1, n_t, dtype=torch.float64)
        f += drift * (t**2)
    return f


# ── the statistic ────────────────────────────────────────────────────────────


def test_r2_would_average_half_under_the_null_which_is_why_r_is_used():
    """The reason R2 was abandoned, asserted on the numbers themselves.

    A block design is essentially one frequency (2 DoF). A random-phase signal at that
    frequency captures cos^2(dphi) of it — R2 ~ 0.5 — while signed r averages 0.
    """
    n_t, period = 160, 40
    t = np.arange(n_t)
    x = np.sin(2 * np.pi * t / period)
    rng = np.random.default_rng(0)
    rs = [
        np.corrcoef(x, np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi)))[0, 1]
        for _ in range(500)
    ]
    rs = np.array(rs)
    assert np.mean(rs**2) == pytest.approx(0.5, abs=0.06)  # R2 is useless here
    assert abs(np.mean(rs)) < 0.1  # signed r is not


def test_task_locked_field_is_detected():
    n_t = 120
    x = _block_design(n_t)
    tc = task_coupling(_field(n_t, task=x, amp=0.3), x, polort=2)
    assert tc.summary["conditions"][0]["abs_r_median"] > 0.6
    assert 0.1 < tc.summary["task_rms_median"] < 0.5  # in map units


def test_pure_noise_field_sits_at_the_null():
    n_t = 120
    x = _block_design(n_t)
    tc = task_coupling(_field(n_t, seed=3), x, polort=2)
    assert tc.summary["conditions"][0]["abs_r_median"] < 0.25


def test_drift_is_not_charged_to_the_task():
    """Residual motion drifts over a run and a long block is low-frequency."""
    n_t = 150
    x = _block_design(n_t, block=25)
    tc = task_coupling(_field(n_t, drift=2.0, seed=5), x, polort=2)
    assert tc.summary["conditions"][0]["abs_r_median"] < 0.25


# ── the null ─────────────────────────────────────────────────────────────────


def test_negating_the_design_leaves_the_magnitude_untouched():
    """Why no shift-based null can work: a P/2 shift of a block design negates it,
    and |r| does not notice. The null draws would BE the alternative."""
    n_t = 120
    x = _block_design(n_t)
    f = _field(n_t, task=x, amp=0.3)
    a = task_coupling(f, x, polort=2)
    b = task_coupling(f, -x, polort=2)
    assert torch.allclose(a.r, -b.r, atol=1e-12)
    assert torch.allclose(a.r.abs(), b.r.abs(), atol=1e-12)


def _contamination_case(n_t=120, *, bold_sign=1.0, flip_gradient=False, seed=0):
    """A field produced BY intensity change: d = beta_data * task / g."""
    x = _block_design(n_t)
    shape = (8, 8, 4)
    g = torch.full(shape, 20.0, dtype=torch.float64)  # PE gradient, intensity/voxel
    if flip_gradient:
        g[4:] *= -1  # the other side of an edge
    beta_data = torch.full(shape, 5.0 * bold_sign, dtype=torch.float64)
    field = (beta_data / g)[..., None] * x[:, 0]
    gen = torch.Generator().manual_seed(seed)
    field = field + 0.002 * torch.randn(*shape, n_t, generator=gen, dtype=torch.float64)
    data = beta_data[..., None] * x[:, 0]
    data = data + 0.05 * torch.randn(*shape, n_t, generator=gen, dtype=torch.float64)
    return x, field, data, g


def test_kappa_is_one_when_the_displacement_explains_the_intensity_change():
    x, field, data, g = _contamination_case()
    f_tc = task_coupling(field, x, polort=2)
    d_tc = task_coupling(data, x, polort=2)
    mask = torch.ones(g.shape, dtype=torch.bool)
    s = contamination_slope(f_tc.beta, d_tc.beta, g, mask)
    assert s["kappa"] == pytest.approx(1.0, abs=0.05)
    assert s["r2"] > 0.9


def test_negative_bold_is_still_scored_as_contamination():
    """Logan's point: negative BOLD is real, and it flips the field's sign.

    Both sides of the relation flip, so kappa is unchanged — which is exactly why the
    ratio is the statistic and the field's raw sign is not.
    """
    x, field, data, g = _contamination_case(bold_sign=-1.0)
    f_tc = task_coupling(field, x, polort=2)
    d_tc = task_coupling(data, x, polort=2)
    mask = torch.ones(g.shape, dtype=torch.bool)
    s = contamination_slope(f_tc.beta, d_tc.beta, g, mask)
    assert s["kappa"] == pytest.approx(1.0, abs=0.05)
    assert s["r2"] > 0.9


def test_gradient_flipping_at_an_edge_does_not_break_it():
    """The field's sign reverses across a boundary even under pure contamination.

    A sign-agreement statistic would call this incoherent; kappa does not move.
    """
    x, field, data, g = _contamination_case(flip_gradient=True)
    f_tc = task_coupling(field, x, polort=2)
    d_tc = task_coupling(data, x, polort=2)
    mask = torch.ones(g.shape, dtype=torch.bool)
    # The field's own sign is split down the middle...
    signs = (f_tc.beta[..., 0] > 0).float().mean()
    assert 0.3 < float(signs) < 0.7
    # ...and kappa is still 1.
    s = contamination_slope(f_tc.beta, d_tc.beta, g, mask)
    assert s["kappa"] == pytest.approx(1.0, abs=0.05)
    assert s["r2"] > 0.9


def test_real_motion_gives_kappa_near_zero():
    """A field that moves the brain but is unrelated to the BOLD response."""
    n_t = 120
    x = _block_design(n_t)
    shape = (8, 8, 4)
    gen = torch.Generator().manual_seed(3)
    g = torch.full(shape, 20.0, dtype=torch.float64)
    # task-locked displacement, but NOT produced by the intensity change
    field = 0.2 * x[:, 0].expand(*shape, n_t).clone()
    field = field + 0.01 * torch.randn(*shape, n_t, generator=gen, dtype=torch.float64)
    # ...and a BOLD response whose amplitude varies independently across voxels
    beta = torch.randn(*shape, generator=gen, dtype=torch.float64)
    data = beta[..., None] * x[:, 0] + 0.05 * torch.randn(
        *shape, n_t, generator=gen, dtype=torch.float64
    )
    f_tc = task_coupling(field, x, polort=2)
    d_tc = task_coupling(data, x, polort=2)
    s = contamination_slope(f_tc.beta, d_tc.beta, g, torch.ones(shape, dtype=torch.bool))
    assert s["r2"] < 0.2  # no BOLD-to-displacement relation


def test_pe_gradient_is_along_the_requested_axis():
    ref = torch.zeros(6, 6, 6, dtype=torch.float64)
    ref += torch.arange(6, dtype=torch.float64)[None, :, None] * 3.0
    assert float(pe_gradient(ref, 1).mean()) == pytest.approx(3.0, abs=0.6)
    assert float(pe_gradient(ref, 0).abs().max()) == 0.0


# ── strata, guards, plumbing ─────────────────────────────────────────────────


def test_responding_stratum_separates_what_the_median_buries():
    n_t = 140
    x = _block_design(n_t, block=20)
    shape = (10, 10, 6)
    f = _field(n_t, shape=shape, seed=7)
    f[:1] += 0.5 * x[:, 0]
    data_r = torch.zeros(*shape, 1, dtype=torch.float64)
    data_r[:1] = 0.6
    mask = torch.ones(shape, dtype=torch.bool)
    tc = task_coupling(f, x, polort=2, mask=mask)
    resp, quiet, _ = responding_mask(data_r, mask, 0.1)
    r, q = tc.summarize(resp), tc.summarize(quiet)
    assert r["conditions"][0]["abs_r_median"] > 3 * q["conditions"][0]["abs_r_median"]
    assert r["n_voxels"] + q["n_voxels"] == int(mask.sum())


def test_enrichment_is_one_when_the_field_ignores_where_the_task_is():
    """The headline number: 1.0x means the field's task coupling is spread like the
    brain, so it has nothing to do with where the task actually is."""
    n_t = 140
    x = _block_design(n_t, block=20)
    shape = (10, 10, 6)
    f = _field(n_t, shape=shape, seed=7) + 0.3 * x[:, 0]  # task-locked EVERYWHERE
    mask = torch.ones(shape, dtype=torch.bool)
    active = torch.zeros(shape, dtype=torch.bool)
    active[:1] = True  # ...but "the task" is only here
    tc = task_coupling(f, x, polort=2, mask=mask)
    e = task_enrichment(tc, active, mask)
    assert e["voxel_share"] == pytest.approx(0.1, abs=0.01)
    assert e["enrichment"] == pytest.approx(1.0, abs=0.25)


def test_enrichment_rises_when_the_field_moves_only_active_voxels():
    """Are we moving voxels, in a task-correlated way, where voxels are task-correlated?"""
    n_t = 140
    x = _block_design(n_t, block=20)
    shape = (10, 10, 6)
    f = _field(n_t, shape=shape, seed=7)
    active = torch.zeros(shape, dtype=torch.bool)
    active[:1] = True
    f[:1] += 1.0 * x[:, 0]  # only the active slab moves with the task
    mask = torch.ones(shape, dtype=torch.bool)
    tc = task_coupling(f, x, polort=2, mask=mask)
    e = task_enrichment(tc, active, mask)
    assert e["enrichment"] > 4.0
    assert e["energy_share"] > 0.4


def test_constant_voxels_score_zero_not_one():
    n_t = 80
    x = _block_design(n_t)
    f = _field(n_t, task=x, amp=0.3)
    f[0, 0, 0] = 0.0
    f[1, 0, 0] = 7.0  # constant, non-zero
    tc = task_coupling(f, x, polort=2)
    assert float(tc.r[0, 0, 0, 0]) == 0.0
    assert float(tc.r[1, 0, 0, 0]) == 0.0
    assert float(tc.task_rms[1, 0, 0]) == 0.0
    assert not bool(tc.valid[1, 0, 0])


def test_collinear_design_is_refused_not_silently_fitted():
    n_t = 60
    t = torch.linspace(-1, 1, n_t, dtype=torch.float64)
    with pytest.raises(ValueError, match="collinear"):
        task_coupling(_field(n_t), t[:, None], polort=3)


def test_co_location_uses_magnitudes():
    a = torch.rand(5, 5, 3, 1, dtype=torch.float64)
    mask = torch.ones(5, 5, 3, dtype=torch.bool)
    assert co_location(a, a, mask) == pytest.approx(1.0)
    assert co_location(a, -a, mask) == pytest.approx(1.0)  # sign-blind by design


def test_default_polort_matches_the_afni_rule():
    assert default_polort(100, 2.0) == 2
    assert default_polort(100, 1.0) == 1
    assert default_polort(500, 1.0) == 4


# ── de-tasking the field ─────────────────────────────────────────────────────


def test_project_task_out_removes_task_and_keeps_drift():
    """The polynomial rule: drift is FITTED so it cannot bias the task beta, but it is
    NOT subtracted — a slowly drifting displacement is real residual motion."""
    n_t = 140
    x = _block_design(n_t, block=20)
    shape = (6, 6, 4)
    t = torch.linspace(-1, 1, n_t, dtype=torch.float64)
    drift = 2.0 * t + 1.5 * t**2
    f = _field(n_t, shape=shape, seed=2) + 0.5 * x[:, 0] + drift

    from fastfuncstuff.stats.task_coupling import project_task_out

    cleaned, removed = project_task_out(f, x, polort=2)
    assert torch.allclose(cleaned + removed, f, atol=1e-9)

    # the task is gone from the cleaned field...
    before = task_coupling(f, x, polort=2).summary["conditions"][0]["abs_r_median"]
    after = task_coupling(cleaned, x, polort=2).summary["conditions"][0]["abs_r_median"]
    assert before > 0.5 and after < 1e-6

    # ...and the drift is still there (it would be ~0 if polys were subtracted too)
    kept = cleaned.reshape(-1, n_t).mean(dim=0)
    assert float(torch.corrcoef(torch.stack([kept, drift]))[0, 1]) > 0.95


def test_detasked_pcs_lose_the_task_the_raw_pcs_carry():
    """Why this exists: a task-correlated nuisance regressor eats real BOLD in a GLM.

    Measured against the DRIFT-RESIDUALIZED regressor, which is the honest target: the
    cleaned field is orthogonal to the task *after* drift removal, not to the raw
    regressor, because the raw regressor shares a slow component with the polynomials
    that we deliberately keep. Every ffs GLM carries polynomials too, so that shared
    part is not the task's to claim.
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix
    from fastfuncstuff.stats.task_coupling import _orthonormal_basis, project_task_out

    n_t = 140
    x = _block_design(n_t, block=20)
    f = _field(n_t, shape=(8, 8, 4), seed=6) + 0.6 * x[:, 0]
    cleaned, _ = project_task_out(f, x, polort=2)

    q_n = _orthonormal_basis(
        construct_polynomial_matrix(n_t, 2, device=torch.device("cpu"), dtype=torch.float64)
    )
    x_detrended = (x - q_n @ (q_n.T @ x))[:, 0]

    def top_pc_corr(v):
        m = v.reshape(-1, n_t)
        m = m - m.mean(dim=1, keepdim=True)
        pc = torch.linalg.svd(m, full_matrices=False)[2][0]
        return abs(float(torch.corrcoef(torch.stack([pc, x_detrended]))[0, 1]))

    assert top_pc_corr(f) > 0.9
    assert top_pc_corr(cleaned) < 1e-6


def test_detask_result_rebuilds_the_corrected_series_from_raw():
    """-detask must change the OUTPUT, not just write side files.

    Also pins the layout bug this hit first time out: the canonical field does not put
    time on a fixed axis across locomoco's paths, so the projection has to go through
    the NIfTI-order view and permute back.
    """
    from fastfuncstuff.processing.locomoco import LocomocoResult, detask_result

    n_t = 60
    nx, ny, nz = 8, 8, 4
    x = _block_design(n_t, block=10)
    rng = np.random.default_rng(0)
    data = rng.normal(100, 10, (nx, ny, nz, n_t)).astype(np.float32)

    # Inject the DRIFT-RESIDUALIZED regressor, so the projection can remove all of it.
    # Injecting the raw regressor would leave its linear component behind — correctly,
    # because drift is kept — and that residual is not what this test is about.
    from fastfuncstuff.glm.core import construct_polynomial_matrix
    from fastfuncstuff.stats.task_coupling import _orthonormal_basis

    q_n = _orthonormal_basis(
        construct_polynomial_matrix(n_t, 1, device=torch.device("cpu"), dtype=torch.float64)
    )
    x_detrended = (x - q_n @ (q_n.T @ x))[:, 0].float()

    # canonical layout with time NOT on axis 3 of the spatial-first order
    perm = [2, 0, 1, 3]
    field_nifti = torch.zeros(nx, ny, nz, n_t, dtype=torch.float32)
    field_nifti += 0.3 * x_detrended
    canon = field_nifti.permute(perm).contiguous()

    result = LocomocoResult(
        u_canon=canon,
        v_canon=torch.zeros_like(canon),
        corrected_canon=torch.zeros_like(canon),
        perm=perm,
        pe_flow_is_u=True,
        pe_axis=1,
        slice_axis=2,
        orig_shape=(nx, ny, nz, n_t),
        a0=0,
        a1=1,
    )
    cleaned, removed, note = detask_result(result, data, x, polort=1)
    assert note is None
    # the task is gone from the field...
    assert float(cleaned.pe_displacement().abs().max()) < 1e-5
    # ...it is accounted for in the removed component...
    assert float(removed[0][2].abs().max()) > 0.1  # 0.3 x the detrended regressor
    # ...and the corrected series was genuinely rebuilt (a zero field = the raw data)
    assert cleaned.corrected_nifti is not None
    assert torch.allclose(cleaned.corrected_nifti, torch.from_numpy(data), atol=1e-3)


def test_empty_active_mask_raises_something_actionable():
    """A dead data-coupling map means the design never lined up with the run.

    The real failure was a 3-D EPI header whose pixdim[4] held the shot time (0.0534 s),
    making a 120-frame run 6.4 s long. Every event fell past the end, the design came
    back empty, and the report indexed an empty condition list. It must say what is
    wrong instead.
    """
    from fastfuncstuff.stats.task_coupling import responding_mask

    mask = torch.ones(4, 4, 3, dtype=torch.bool)
    dead = torch.zeros(4, 4, 3, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="active mask is empty"):
        responding_mask(dead, mask, thresh=0.2)


def test_paired_templates_are_matched_in_task_state():
    """The condition-paired reference: one template per bin, each built from its own frames.

    The property that makes it work is that the BOLD response is CONSTANT within a bin,
    so it is common-mode between a frame and its template and cancels in the data term.
    """
    from fastfuncstuff.design.binning import design_state_bins
    from fastfuncstuff.processing.locomoco import paired_templates

    n_t = 120
    x = _block_design(n_t, block=20)
    bin_of, info = design_state_bins(x, bin_width=0.34, min_frames=4)
    # a series whose intensity follows the task exactly, with no motion at all
    series = torch.ones(4, 4, 3, n_t, dtype=torch.float64) * (100 + 20 * x[:, 0])
    tmpl = paired_templates(series, bin_of, "mean")

    assert tmpl.shape == (info["n_bins"], 4, 4, 3)
    for b in range(info["n_bins"]):
        frames = series[..., bin_of == b]
        # every frame in a bin is within a bin-width of its own template
        assert float((frames - tmpl[b][..., None]).abs().max()) < 20 * 0.34 + 1e-9
    # ...and distinct bins really do sit at different intensities
    means = sorted(float(tmpl[b].mean()) for b in range(info["n_bins"]))
    assert means[-1] - means[0] > 10


def test_paired_templates_median_and_max_fallback():
    """max falls back to the mean: a per-bin max over ~15 frames is a noise envelope."""
    from fastfuncstuff.processing.locomoco import paired_templates

    bin_of = torch.tensor([0, 0, 0, 1, 1, 1])
    series = torch.zeros(2, 2, 1, 6, dtype=torch.float64)
    series[..., :3] = torch.tensor([1.0, 2.0, 30.0])
    series[..., 3:] = torch.tensor([4.0, 5.0, 6.0])
    med = paired_templates(series, bin_of, "median")
    mx = paired_templates(series, bin_of, "max")
    assert float(med[0].mean()) == pytest.approx(2.0)  # median ignores the outlier
    assert float(mx[0].mean()) == pytest.approx(11.0)  # ...mean does not (max -> mean)


def test_chance_share_is_arithmetic_not_a_null():
    """The share a field with no task relation shows anyway: sqrt(K/df)."""
    n_t = 120
    x = _block_design(n_t)
    tc = task_coupling(_field(n_t, seed=1), x, polort=2)
    assert tc.n_timepoints == n_t
    assert tc.chance_share == pytest.approx((1 / (n_t - 2 - 1)) ** 0.5, rel=1e-6)


def test_pe_axis_name_never_calls_the_partition_the_primary_pe():
    """A ``-pe_dir2``-only run labels its one field ``pe1`` — it is still the partition.

    ``pe_displacements`` has no second axis to report on a single-axis run, so the
    label alone cannot say which axis it is. Naming that block "primary PE" told the
    reader the one thing it is not.
    """
    from types import SimpleNamespace

    from fastfuncstuff.cli.locomoco import _pe_axis_name

    partition_only = SimpleNamespace(pe_dir=None, pe_dir2="IS")
    name = _pe_axis_name("pe1", 2, partition_only)
    assert "partition" in name and "primary" not in name
    assert "pe1" in name and "IS" in name and "axis 2" in name

    both = SimpleNamespace(pe_dir=["AP"], pe_dir2="IS")
    assert _pe_axis_name("pe1", 1, both).startswith("pe1 (primary PE AP")
    assert _pe_axis_name("pe2", 2, both).startswith("pe2 (partition IS")


def test_report_is_compact_and_names_the_axis_it_is_about():
    """The report is read once per axis per run; it must fit on a screen.

    It used to be ~55 lines of methodology per axis, echoed to stdout AND written to
    the .txt, with the ``pe1``/``pe2`` suffix that names the files on disk appearing
    nowhere in it.
    """
    rng = np.random.default_rng(0)
    design = _block_design(60)
    field = torch.as_tensor(rng.normal(size=(6, 6, 4, 60)), dtype=torch.float32)
    tc = task_coupling(field, design, polort=2, labels=["check"])
    text = format_task_coupling_report(tc, label="pe2 (partition IS, axis 2 = z)")
    assert len(text.splitlines()) <= 15, text
    assert "pe2 (partition IS, axis 2 = z)" in text.splitlines()[0]


def _contaminated_and_innocent(rng, n_t=80):
    """A field contaminated only inside an 'activated' blob, and one that is not.

    Both are task-locked by the same amount overall. Only the first has its coupling
    CO-LOCATED with the response, which is the thing that distinguishes BOLD read as
    motion from real task-correlated head motion.
    """
    x = _block_design(n_t, block=10)
    xt = x[:, 0]
    shape = (16, 16, 10)
    data = torch.as_tensor(rng.normal(size=(*shape, n_t)), dtype=torch.float64)
    blob = torch.zeros(shape, dtype=torch.bool)
    # Sized to the responding decile on purpose: if the blob is much smaller than the
    # active mask, most of that mask is unresponsive noise and the enrichment ceiling
    # collapses to a few x regardless of how contaminated the field is.
    blob[4:12, 4:12, 3:7] = True  # 256 of 2560 voxels = the top decile
    data[blob] += 3.0 * xt

    contaminated = torch.as_tensor(rng.normal(size=(*shape, n_t)) * 0.3)
    contaminated[blob] += 3.0 * xt  # coupling exactly where the response is

    innocent = torch.as_tensor(rng.normal(size=(*shape, n_t)) * 0.3)
    elsewhere = torch.zeros(shape, dtype=torch.bool)
    elsewhere[12:16, 8:16, :] = True  # task-locked, but nowhere near the response
    innocent[elsewhere] += 3.0 * xt
    return x, data, contaminated, innocent


def test_enrichment_curve_separates_contamination_from_task_locked_motion():
    """The tail rises toward the ceiling for contamination and stays flat otherwise.

    The bug of record: the shipped verdict decided on the MEDIAN over the responding
    decile, which is a median over flat voxel interiors where the encode gradient is
    zero and no displacement can be induced at all. On a real 0.8mm checkerboard run
    it printed "no appreciable coupling" while the field's top percentile was ~9x
    concentrated on activated tissue.
    """
    rng = np.random.default_rng(7)
    x, data, contaminated, innocent = _contaminated_and_innocent(rng)
    mask = torch.ones(data.shape[:3])
    dtc = task_coupling(data, x, polort=2, mask=mask)
    resp, _, _ = responding_mask(dtc.r, mask, 0.1)

    bad = enrichment_curve(task_coupling(contaminated, x, polort=2, mask=mask), resp, mask)
    good = enrichment_curve(task_coupling(innocent, x, polort=2, mask=mask), resp, mask)
    assert bad and good
    # Contamination: the tail is strongly concentrated on the responding voxels, and
    # concentration RISES as the field-side threshold tightens.
    assert bad[-1]["enrichment"] > 3.0, bad
    assert bad[-1]["enrichment"] > bad[0]["enrichment"], bad
    # Task-locked motion somewhere else: no concentration at any threshold.
    assert good[-1]["enrichment"] < 1.5, good


def test_verdict_flags_contamination_the_median_statistic_missed():
    """End to end: the report says CONTAMINATION for the co-located field only."""
    rng = np.random.default_rng(7)
    x, data, contaminated, innocent = _contaminated_and_innocent(rng)
    mask = torch.ones(data.shape[:3])
    dtc = task_coupling(data, x, polort=2, mask=mask)
    resp, quiet, cut = responding_mask(dtc.r, mask, 0.1)

    def report(field):
        tc = task_coupling(field, x, polort=2, mask=mask)
        return format_task_coupling_report(
            tc,
            dtc,
            co_location(tc.r, dtc.r, mask),
            responding=tc.summarize(resp),
            quiet=tc.summarize(quiet),
            enrichment=task_enrichment(tc, resp, mask),
            active_thresh=cut,
            curve=enrichment_curve(tc, resp, mask),
        )

    assert "CONTAMINATION" in report(contaminated)
    assert "CONTAMINATION" not in report(innocent)


def test_notch_selects_one_line_for_a_block_design_not_a_shoulder():
    """Peak-relative, not median-relative — the bug this rule was rewritten to avoid.

    A 15-cycle 20s block design puts 91% of its power in ONE bin. Contrast against the
    median bin was the first rule and is not equivalent: the median sits ~2000x below
    the fundamental, so a 10x-the-median cut lands inside the low-frequency shoulder
    that drift removal leaves behind and selects nine bins instead of one.
    """
    n_t, tr, period = 120, 2.5, 20.0
    box = np.zeros(int(n_t * tr / 0.1))
    for on in range(10, int(n_t * tr), int(period)):
        box[int(on / 0.1) : int((on + period / 2) / 0.1)] = 1.0
    k = np.exp(-np.arange(0, 320) / 30.0)
    conv = np.convolve(box, k / k.sum())[: len(box)]
    design = torch.tensor(conv[(np.arange(n_t) * tr / 0.1).astype(int)])[:, None]

    bins, info = design_notch_bins(design, polort=3)
    fundamental = int(round(n_t * tr / period))  # 15 cycles over the run
    assert fundamental in bins
    # The shoulder is what the median-contrast rule wrongly swept up. Every kept bin
    # must be the fundamental or one of its harmonics, never the low-frequency skirt.
    assert not any(1 <= b < fundamental for b in bins), bins
    assert all(b % fundamental == 0 for b in bins), bins
    assert info["spectrum_frac"] < 0.05
    assert notch_basis(n_t, bins, polort=3).shape[1] == 2 * len(bins)

    # widen adds sidebands, for amplitude non-stationarity across blocks -- not for
    # HRF width, which can only narrow the design's spectrum.
    wide, _ = design_notch_bins(design, polort=3, widen=1)
    assert set(wide) == {b + d for b in bins for d in (-1, 0, 1)}
    assert notch_basis(n_t, wide, polort=3).shape[1] == 2 * len(wide)


def test_notch_refuses_a_broadband_design_instead_of_eating_the_data():
    """A jittered event-related design has no line to notch; saying so beats notching."""
    rng = np.random.default_rng(0)
    hi = np.zeros(120)
    for onset in rng.uniform(0, 290, 40):
        hi[int(onset / 2.5)] = 1.0
    k = np.exp(-np.arange(0, 12) / 3.0)
    design = torch.tensor(np.convolve(hi, k / k.sum())[:120])[:, None]
    with pytest.raises(ValueError, match="broadband"):
        design_notch_bins(design, polort=3)


def test_filter_task_band_removes_the_line_and_keeps_the_drift():
    """The split that matters: the task line goes, a slow drift stays.

    A drifting displacement is real residual motion -- the same reason project_task_out
    fits the polynomials but does not subtract them.
    """
    n_t = 120
    t = np.arange(n_t)
    design = torch.tensor(np.cos(2 * np.pi * 15 * t / n_t))[:, None]
    drift = 3.0 * np.linspace(-1, 1, n_t) ** 2
    rng = np.random.default_rng(1)
    series = torch.tensor(
        drift + 2.0 * np.cos(2 * np.pi * 15 * t / n_t) + rng.normal(0, 0.1, (4, 4, 3, n_t))
    )

    bins, _ = design_notch_bins(design, polort=3)
    basis = notch_basis(n_t, bins, polort=3)
    out = filter_task_band(series, basis)

    def line_power(x):
        v = np.asarray(x).reshape(-1, n_t)
        return float((np.abs(np.fft.rfft(v, axis=1))[:, 15] ** 2).mean())

    assert line_power(out) < 1e-3 * line_power(series)
    # The drift survives: correlation of each voxel's mean-removed course with it.
    d = drift - drift.mean()
    kept = np.asarray(out).reshape(-1, n_t).mean(0)
    kept = kept - kept.mean()
    assert float(np.corrcoef(kept, d)[0, 1]) > 0.99


def test_parse_detask_modes():
    from fastfuncstuff.cli.locomoco import parse_detask

    assert parse_detask(None) == (False, None)
    assert parse_detask("field") == (True, None)  # bare -detask, via const
    assert parse_detask("filter") == (False, 0)
    assert parse_detask("filter:2") == (False, 2)
    assert parse_detask("FILTER:0") == (False, 0)
    for bad in ("filter:x", "both", "filter:-1"):
        with pytest.raises(ValueError):
            parse_detask(bad)


@pytest.mark.slow
def test_cli_detask_filter_actually_reaches_the_estimator(tmp_path):
    """The notch must change the FIELD, not just print that it ran.

    Regression of record: the filter was wired into the multi-echo runner only, so a
    single-echo run parsed the flag, took the branch, and produced a field identical to
    an unfiltered one. Asserting on the printed line would have passed; asserting on
    the field is what catches it.
    """
    import nibabel as nib

    from fastfuncstuff.cli.locomoco import main

    n_t, tr, rng = 120, 2.5, np.random.default_rng(0)
    x = np.zeros(n_t)
    for onset in range(10, 300, 20):  # 15 whole cycles of a 20s period
        x[int(onset / tr) : int((onset + 10) / tr)] = 1.0
    series = rng.normal(0, 1, (20, 22, 12, n_t)).astype(np.float32)
    series[5:12, 6:14, 3:8] += (2.0 * x).astype(np.float32)
    series += 100.0
    src = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(series, np.eye(4)), str(src))
    ev = tmp_path / "ev.tsv"
    ev.write_text(
        "onset\tduration\ttrial_type\n" + "".join(f"{o}\t10\tcheck\n" for o in range(10, 300, 20))
    )

    def run(stem, *extra):
        assert (
            main(
                [
                    "-i",
                    str(src),
                    "-o",
                    f"{stem}.nii.gz",
                    "-pe_dir1",
                    "AP",
                    "-pe_dir2",
                    "IS",
                    "-backend",
                    "flow",
                    "-refine",
                    "1",
                    "-events",
                    str(ev),
                    "-tr",
                    str(tr),
                    "-device",
                    "cpu",
                    *extra,
                ]
            )
            == 0
        )
        return nib.load(f"{stem}_flow_pe1.nii.gz").get_fdata(dtype=np.float32)

    base = run(str(tmp_path / "base"))
    filt = run(str(tmp_path / "filt"), "-detask", "filter")
    assert base.shape == filt.shape
    assert not np.allclose(base, filt), "the notch did not reach the estimator"

    # And it removed the task band it claimed to: the fundamental is bin 15 over 120
    # frames with a 20s period at TR 2.5.
    def line_power(field):
        v = field.reshape(-1, n_t)
        v = v - v.mean(axis=1, keepdims=True)
        p = np.abs(np.fft.rfft(v, axis=1)) ** 2
        return float(p[:, 15].sum() / p.sum())

    assert line_power(filt) < 0.5 * line_power(base), (line_power(base), line_power(filt))
