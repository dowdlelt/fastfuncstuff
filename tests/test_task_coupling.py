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

from fastfuncstuff.glm.core import construct_polynomial_matrix
from fastfuncstuff.stats.task_coupling import (
    _orthonormal_basis,
    co_location,
    component_task_fit,
    contamination_slope,
    default_polort,
    design_fit_basis,
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


def test_notch_warns_between_the_cheap_and_the_refused_band():
    """15-50% of the spectrum is expensive but allowed, with a warning attached.

    Bulk head motion is spectrally BROAD, so removing a sixth of the spectrum can still
    leave it estimable. Whether it does is not knowable from the design -- only from
    whether the resulting field still tracks motion -- so the ceiling is permissive and
    the cost is announced rather than silently enforced.
    """
    rng = np.random.default_rng(3)
    hi = np.zeros(120)
    for onset in np.clip(np.arange(10, 300, 20) + rng.normal(0, 2.0, 15), 0, 295):
        hi[int(onset / 2.5) : int(onset / 2.5) + 4] = 1.0
    k = np.exp(-np.arange(0, 12) / 3.0)
    design = torch.tensor(np.convolve(hi, k / k.sum())[:120])[:, None]

    bins, info = design_notch_bins(design, polort=3)
    assert 0.15 < info["spectrum_frac"] <= 0.50, info["spectrum_frac"]
    assert info["warning"] and "CHECK" in info["warning"]
    # Below the warn level there is no warning at all.
    clean = np.zeros(120)
    for onset in range(10, 300, 20):
        clean[int(onset / 2.5) : int(onset / 2.5) + 4] = 1.0
    tidy = torch.tensor(np.convolve(clean, k / k.sum())[:120])[:, None]
    _, tidy_info = design_notch_bins(tidy, polort=3)
    assert tidy_info["warning"] is None, tidy_info


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

    # (clean_field, notch_widen, fit_deriv) -- three modes, and only one is ever set.
    assert parse_detask(None) == (False, None, None)
    assert parse_detask("field") == (True, None, None)  # bare -detask, via const
    assert parse_detask("filter") == (False, 0, None)
    assert parse_detask("filter:2") == (False, 2, None)
    assert parse_detask("FILTER:0") == (False, 0, None)
    assert parse_detask("fit") == (False, None, 0)
    assert parse_detask("fit:1") == (False, None, 1)
    assert parse_detask("FIT:2") == (False, None, 2)
    for bad in ("filter:x", "both", "filter:-1", "fit:x", "fit:-1"):
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


def test_parse_warp_recon_specs():
    from fastfuncstuff.cli.locomoco import parse_warp_recon

    assert parse_warp_recon(None) == (None, None, None)
    assert parse_warp_recon("pcs") == (None, None, "pcs")
    assert parse_warp_recon("pcs:all") == (None, None, "pcs")
    assert parse_warp_recon("pcs:5") == (5, None, "pcs")
    # Below 1 is a variance FRACTION -- unambiguous, since a count below 1 is meaningless.
    assert parse_warp_recon("pcs:0.8") == (None, 0.8, "pcs")
    # ICA moved to -detask ica. -warp_recon RECONSTRUCTS from principal components;
    # ICA never reconstructed at all -- it projects task-loaded time courses out of the
    # full-rank field, which is a de-tasking operation. The old spelling points there
    # rather than failing with a bare "unknown method".
    with pytest.raises(ValueError, match="moved to '-detask ica'"):
        parse_warp_recon("ica")
    with pytest.raises(ValueError, match="moved to '-detask ica'"):
        parse_warp_recon("ica:60")
    with pytest.raises(ValueError, match="only defined for -detask ica"):
        parse_warp_recon("pcs:sweep")
    for bad in ("pcs:x", "eigen", "pcs:0", "pcs:2.5"):
        with pytest.raises(ValueError):
            parse_warp_recon(bad)


def test_parse_detask_ica_specs():
    """The rank has an interior optimum, so bare 'ica' SEARCHES rather than guessing.

    Measured on a real contaminated run: 20 components missed the task source (1.75x
    enrichment), 60 found it (2.79x), the full 119 over-split it back down (2.27x). A
    variance fraction was the original default and resolved to 105 of 119 -- deep in
    the over-splitting regime -- which is why no fraction is the default now.
    """
    from fastfuncstuff.cli.locomoco import parse_detask_ica

    assert parse_detask_ica("ica") == ("sweep", None)
    assert parse_detask_ica("ica:sweep") == ("sweep", None)
    assert parse_detask_ica("ica:60") == (60, None)
    assert parse_detask_ica("ica:0.9") == (None, 0.9)
    for bad in ("ica:x", "ica:0", "ica:2.5", "ica:-3"):
        with pytest.raises(ValueError):
            parse_detask_ica(bad)


def test_warp_pc_basis_keeps_the_loading_per_axis_and_round_trips():
    """A component contaminated on ONE axis must score high there and low on the other.

    The whole reason the shared temporal basis is safe: each component carries its own
    spatial loading per encode axis, so rejection is a per-(component, axis) decision
    even though the dictionary is common. And keeping every component must reproduce
    the input exactly -- a reconstruction that quietly loses the temporal mean would
    move every voxel by its own average displacement.
    """
    from fastfuncstuff.processing.locomoco import warp_pc_basis, warp_pc_reconstruct
    from fastfuncstuff.stats.task_coupling import map_enrichment

    rng = np.random.default_rng(0)
    n_t, shape = 80, (12, 12, 6)
    t = np.arange(n_t)
    resp = np.sin(2 * np.pi * 3 * t / n_t)  # shared, brain-wide
    task = np.sin(2 * np.pi * 10 * t / n_t)
    blob = np.zeros(shape, bool)
    blob[3:7, 3:7, 2:4] = True  # 32 of 864 voxels

    f1 = rng.normal(0, 0.05, (*shape, n_t)) + 0.5 * resp
    f2 = rng.normal(0, 0.05, (*shape, n_t)) + 0.3 * resp
    f1 += 0.8 * blob[..., None] * task  # contamination on axis 1 ONLY
    comps = [(1, torch.tensor(f1)), (2, torch.tensor(f2))]

    u, loadings, _means, _var = warp_pc_basis(comps)
    mask, active = torch.ones(shape), torch.tensor(blob)
    scored = [
        (
            map_enrichment(loadings[0][1][..., i], active, mask)["enrichment"],
            map_enrichment(loadings[1][1][..., i], active, mask)["enrichment"],
        )
        for i in range(min(4, u.shape[1]))
    ]
    worst = max(range(len(scored)), key=lambda i: scored[i][0])
    assert scored[worst][0] > 5.0, scored  # loud on the contaminated axis
    assert scored[worst][1] < 2.0, scored  # quiet on the clean one

    full = warp_pc_reconstruct(comps, keep={})
    assert torch.allclose(full[0][1], torch.tensor(f1, dtype=torch.float32), atol=1e-4)
    assert torch.allclose(full[1][1], torch.tensor(f2, dtype=torch.float32), atol=1e-4)


@pytest.mark.slow
def test_write_pc_maps_is_one_4d_file_per_axis_with_labels(tmp_path):
    """The maps exist to be LOOKED at, so the identifying metadata has to survive.

    One file per encode axis, component on the 4th axis, and sub-brick k of the two
    files is the same temporal component seen on the two axes -- which is only useful
    if the brick labels say which component and how strong it is.
    """
    import nibabel as nib

    from fastfuncstuff.cli.locomoco import main
    from fastfuncstuff.io.headers import read_brick_labels

    n_t, tr, rng = 60, 2.5, np.random.default_rng(0)
    x = np.zeros(n_t)
    for onset in range(10, 150, 20):
        x[int(onset / tr) : int((onset + 10) / tr)] = 1.0
    series = rng.normal(0, 1, (14, 16, 8, n_t)).astype(np.float32)
    series[4:9, 5:11, 2:6] += (2.0 * x).astype(np.float32)
    series += 100.0
    src = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(series, np.eye(4)), str(src))
    ev = tmp_path / "ev.tsv"
    ev.write_text(
        "onset\tduration\ttrial_type\n" + "".join(f"{o}\t10\tcheck\n" for o in range(10, 150, 20))
    )
    stem = str(tmp_path / "out")
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
                "-write_pc_maps",
                "4",
            ]
        )
        == 0
    )
    for axis in ("pe1", "pe2"):
        img = nib.load(f"{stem}_pcmap_{axis}.nii.gz")
        assert img.shape[:3] == series.shape[:3]
        assert img.shape[3] == 4
        labels = read_brick_labels(img)
        assert len(labels) == 4
        assert labels[0].startswith("PC00 ") and "x" in labels[0]  # variance and enrichment
    table = (tmp_path / "out_pcmap_scores.1D").read_text().splitlines()
    assert table[0].startswith("# component") and "pe1" in table[0] and "pe2" in table[0]
    assert len(table) == 5  # header + 4 components


def test_warp_ica_basis_finds_what_pca_structurally_cannot():
    """ICA orders by independence, PCA by variance — and that is the whole difference.

    The bug of record is not a bug but a limit: contamination worth a fraction of a
    percent of the field cannot dominate any PRINCIPAL component, so -reject saw
    nothing on a run whose field was 8.8x task-enriched at its tail. Measured there, no
    PC exceeded 1.25x enrichment or 0.07 correlation with the design, while the best
    independent component reached 2.8-3.0x at 0.68. This pins the same contrast on a
    planted case, where the planted source is deliberately low-variance.
    """
    from fastfuncstuff.processing.locomoco import (
        warp_ica_basis,
        warp_pc_basis,
        warp_reconstruct,
    )
    from fastfuncstuff.stats.task_coupling import map_enrichment

    rng = np.random.default_rng(0)
    n_t, shape = 80, (12, 12, 6)
    t = np.arange(n_t)
    resp = np.sin(2 * np.pi * 3 * t / n_t)  # brain-wide, high variance
    task = np.sin(2 * np.pi * 10 * t / n_t)
    blob = np.zeros(shape, bool)
    blob[3:7, 3:7, 2:4] = True
    f1 = rng.normal(0, 0.05, (*shape, n_t)) + 0.5 * resp
    f2 = rng.normal(0, 0.05, (*shape, n_t)) + 0.3 * resp
    f1 += 0.8 * blob[..., None] * task
    comps = [(1, torch.tensor(f1)), (2, torch.tensor(f2))]
    mask, active = torch.ones(shape), torch.tensor(blob)

    def best_score(basis, loadings):
        k = basis.shape[1]
        scores = [
            map_enrichment(loadings[0][1][..., i], active, mask)["enrichment"] for i in range(k)
        ]
        i = int(np.argmax(scores))
        tc = basis[:, i].numpy() - basis[:, i].numpy().mean()
        x = task - task.mean()
        denom = np.linalg.norm(tc) * np.linalg.norm(x)
        return scores[i], abs(float(tc @ x / denom)) if denom > 0 else 0.0

    ica = warp_ica_basis(comps, pca_components=0.95, device=torch.device("cpu"))
    assert ica is not None
    e_ica, r_ica = best_score(ica[0], ica[1])
    assert e_ica > 5.0, e_ica
    assert r_ica > 0.8, r_ica

    # Reconstruction shares one formula with the PC path, and is LOSSY here by
    # construction: the PCA rank is chosen before the rotation, so 5% of the variance is
    # already gone. That is the trade the flag's help spells out.
    full = warp_reconstruct(ica[0], ica[1], ica[2], keep={})
    resid = float((full[0][1] - torch.tensor(f1, dtype=torch.float32)).pow(2).mean().sqrt())
    assert resid < 0.5 * float(np.std(f1)), resid

    # The PC basis at full rank round-trips exactly, unlike ICA.
    pcs = warp_pc_basis(comps, device=torch.device("cpu"))
    assert pcs is not None
    exact = warp_reconstruct(pcs[0], pcs[1], pcs[2], keep={})
    assert torch.allclose(exact[0][1], torch.tensor(f1, dtype=torch.float32), atol=1e-4)


def test_warp_project_out_removes_only_the_named_timecourses():
    """Reject by projection on the FULL-rank field, not by rebuilding from a truncation.

    The bug of record: -warp_recon ica at 95% variance rejected NOTHING on a real run
    and still cost 21.5% of the field's rms, because rebuilding from an ICA basis
    discards whatever the PCA reduction dropped. A projection cannot do that -- it
    touches only the span it is given.
    """
    from fastfuncstuff.processing.locomoco import warp_project_out

    rng = np.random.default_rng(0)
    n_t, shape = 100, (8, 8, 4)
    t = np.arange(n_t)
    bad = np.sin(2 * np.pi * 7 * t / n_t)
    good = np.cos(2 * np.pi * 3 * t / n_t)
    field = rng.normal(0, 0.1, (*shape, n_t)) + 1.0 * bad + 0.7 * good + 5.0
    comps = [(1, torch.tensor(field))]

    out = warp_project_out(comps, {1: torch.tensor(bad)[:, None].double()})
    cleaned = out[0][1].numpy()

    def amplitude(x, v):
        v = v - v.mean()
        flat = x.reshape(-1, n_t)
        flat = flat - flat.mean(axis=1, keepdims=True)
        return float(np.abs(flat @ v).mean() / np.linalg.norm(v) ** 2)

    assert amplitude(cleaned, bad) < 1e-5 * amplitude(field, bad)
    # Everything else survives untouched -- including the temporal mean, which is not a
    # component and would otherwise drag every voxel's average displacement with it.
    assert abs(amplitude(cleaned, good) - amplitude(field, good)) < 1e-4
    assert abs(float(cleaned.mean()) - float(field.mean())) < 1e-3
    # And an empty request is exactly identity, not a lossy round trip.
    same = warp_project_out(comps, {})
    assert np.abs(same[0][1].numpy() - field.astype(np.float32)).max() == 0.0


@pytest.mark.slow
def test_reject_writes_plottable_timecourses_and_an_after_map(tmp_path):
    """Rejection has to be inspectable: what was removed, and whether it worked.

    Two gaps a real run exposed. The dropped components existed only as a printed
    enrichment, so there was nothing to plot against the design; and the _taskr maps
    are measured BEFORE the fix by design, which left no after to compare them with.
    """
    import nibabel as nib

    from fastfuncstuff.cli.locomoco import main

    n_t, tr, rng = 120, 2.5, np.random.default_rng(0)
    x = np.zeros(n_t)
    for onset in range(10, 300, 20):
        x[int(onset / tr) : int((onset + 10) / tr)] = 1.0
    series = rng.normal(0, 1, (16, 18, 8, n_t)).astype(np.float32)
    series[4:10, 5:12, 2:6] += (2.0 * x).astype(np.float32)
    series += 100.0
    src = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(series, np.eye(4)), str(src))
    ev = tmp_path / "ev.tsv"
    ev.write_text(
        "onset\tduration\ttrial_type\n" + "".join(f"{o}\t10\tcheck\n" for o in range(10, 300, 20))
    )
    stem = str(tmp_path / "out")
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
                "-warp_recon",
                "ica:30",
                "-reject",
            ]
        )
        == 0
    )
    rejected = tmp_path / "out_locomoco_rejected.1D"
    assert rejected.exists()
    lines = rejected.read_text().splitlines()
    head = [ln for ln in lines if ln.startswith("#")][-1]  # the column-name line
    table = np.loadtxt(rejected)
    assert table.shape[0] == n_t and table.shape[1] >= 2
    # z-scored, so the columns are directly comparable on one axis.
    assert np.allclose(table.std(axis=0), 1.0, atol=1e-6)
    # ONE column per component, never one per (component, axis). The temporal basis is
    # shared across the encode axes, so a component rejected on both had its identical
    # time course written twice and read as two findings when it was one.
    names = head.replace("#", "").split()
    assert names[0] == "design"
    assert len(names) == len(set(names)), names
    for j in range(1, table.shape[1]):
        for i in range(1, j):
            assert not np.allclose(table[:, i], table[:, j]), (names[i], names[j])

    # The after maps exist and are a real second measurement, not a copy of the before.
    for axis in ("pe1", "pe2"):
        before = nib.load(f"{stem}_taskr_{axis}.nii.gz").get_fdata(dtype=np.float32)
        after = nib.load(f"{stem}_taskr_{axis}_after.nii.gz").get_fdata(dtype=np.float32)
        assert before.shape == after.shape
        assert not np.allclose(before, after)


def test_design_fit_basis_removes_the_task_and_keeps_the_drift():
    """The design-space cut must take the response out and leave slow drift alone.

    Drift is real residual motion, not something a task filter should eat — the same
    split :func:`notch_basis` and ``-detask field`` make.
    """
    n_t = 120
    design = _block_design(n_t)
    basis = design_fit_basis(design, polort=3)
    # One column per regressor: the whole point against the notch's two per line.
    assert basis.shape == (n_t, design.shape[1])
    assert torch.allclose(basis.T @ basis, torch.eye(basis.shape[1], dtype=basis.dtype), atol=1e-10)

    t = torch.arange(n_t, dtype=torch.float64)
    drift = 1.0 + 0.01 * t + 3e-5 * t**2
    series = (2.5 * design[:, 0] + drift).reshape(1, 1, 1, n_t)
    out = filter_task_band(series, basis).reshape(-1)

    # The task is gone. Orthogonality is against the BASIS -- the design after drift
    # removal -- not the raw design: polynomials go into the fit but not the
    # subtraction, so whatever the design shares with drift is deliberately left
    # behind. Asserting against the raw column would be asserting the drift was eaten.
    assert float((basis.T @ out).abs().max()) < 1e-8
    # ...and the drift survived, which a polort-eating basis would have flattened.
    assert float(out.std()) > 0.1 * float(drift.std())


def test_design_fit_basis_derivatives_cost_one_dof_each_and_absorb_latency():
    """``n_deriv`` widens the subspace by one column per regressor per derivative, and
    the extra columns are what let a latency-shifted response be removed at all."""
    n_t = 160
    design = _block_design(n_t, block=16)
    k = design.shape[1]
    assert design_fit_basis(design, polort=3, n_deriv=1).shape[1] == 2 * k
    assert design_fit_basis(design, polort=3, n_deriv=2).shape[1] == 3 * k

    # A SUB-FRAME latency shift is what the derivative column absorbs: to first order a
    # shifted regressor is x(t) - dt*x'(t), so the derivative spans the mismatch in the
    # small-dt limit. Measured in the DRIFT-ORTHOGONAL subspace, because the basis
    # deliberately leaves the design's drift-parallel part alone and that residual is
    # an order of magnitude larger than the latency term -- comparing raw norms hides
    # the effect entirely.
    q_n = _orthonormal_basis(
        construct_polynomial_matrix(n_t, 3, device=torch.device("cpu"), dtype=torch.float64)
    )

    def _perp(v):
        return v - q_n @ (q_n.T @ v)

    x = design[:, 0].double()
    ratios = []
    for dt in (0.1, 0.3, 0.5):
        late = ((1 - dt) * x + dt * torch.roll(x, 1)).reshape(1, 1, 1, n_t)
        plain = _perp(filter_task_band(late, design_fit_basis(design, polort=3)).reshape(-1))
        deriv = _perp(
            filter_task_band(late, design_fit_basis(design, polort=3, n_deriv=1)).reshape(-1)
        )
        ratios.append(float(deriv.norm()) / float(plain.norm()))

    # PARTIAL, not complete -- which is the point. A central difference is not exactly
    # matched to the shift, and this design's response is sharp, so a third of the
    # latency mismatch survives even with the derivative in. That residue is precisely
    # the "leftover variance" -detask fit warns about; a test asserting near-total
    # removal would be asserting something the method does not deliver.
    assert all(r < 0.75 for r in ratios), ratios
    # Linear in dt, so the ratio is scale-free -- confirms this is the first-order term
    # and not an artifact of one shift size.
    assert max(ratios) - min(ratios) < 0.02, ratios


def test_design_fit_basis_is_far_cheaper_than_the_notch_on_a_real_block_design():
    """The claim the -detask fit help makes, pinned: on an 18s-block design the notch
    costs 2 DoF per line while the fit costs 1 per regressor, and the gap is large.

    Built to match a real acquisition (OHBMPilot04: 5 conditions, 18s blocks at 20s SOA
    in triplets with 36s rest gaps, TR 3.5, 225 frames), which notched 38% of the
    spectrum -- past the warn threshold -- while spanning 5 design columns.
    """
    tr, n_t, up = 3.5, 225, 20
    onsets = {
        0: [12.0, 164.0, 360.0, 436.1, 572.1, 628.1],
        1: [88.0, 224.0, 284.0, 340.0, 648.1],
        2: [32.0, 108.0, 320.0, 456.1, 516.1, 592.1],
        3: [52.0, 128.0, 244.0, 396.1, 496.1, 552.1],
        4: [184.0, 204.0, 264.0, 416.1, 476.1, 668.1],
    }
    n_up = int(n_t * tr * up)
    hrf = np.exp(-np.arange(0, 32, 1 / up) / 4.0) * (np.arange(0, 32, 1 / up) ** 2)
    hrf = hrf / hrf.sum()
    cols = []
    for k in sorted(onsets):
        box = np.zeros(n_up)
        for o in onsets[k]:
            box[int(o * up) : int((o + 18.0) * up)] = 1.0
        cols.append(np.convolve(box, hrf)[:n_up][:: int(up * tr)][:n_t])
    design = torch.tensor(np.stack(cols, axis=1))

    fit_dof = design_fit_basis(design, polort=6).shape[1]
    bins, info = design_notch_bins(design, polort=6)
    notch_dof = notch_basis(n_t, bins, polort=6).shape[1]

    assert fit_dof == 5  # one per condition
    assert info["spectrum_frac"] > 0.15  # this design is NOT cheaply notchable
    assert notch_dof > 8 * fit_dof, (notch_dof, fit_dof)


def _ar1(n, k, seed, rho=0.6):
    """Autocorrelated noise: the thing that makes a spurious task fit possible."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, k))
    y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = rho * y[t - 1] + e[t]
    return torch.tensor(y)


def _pilot_design(n_t=225, tr=3.5, polort=6):
    """The real OHBMPilot04 design: 5 conditions, 18s blocks, 20s SOA in triplets."""
    onsets = {
        0: [12.0, 164.0, 360.0, 436.1, 572.1, 628.1],
        1: [88.0, 224.0, 284.0, 340.0, 648.1],
        2: [32.0, 108.0, 320.0, 456.1, 516.1, 592.1],
        3: [52.0, 128.0, 244.0, 396.1, 496.1, 552.1],
        4: [184.0, 204.0, 264.0, 416.1, 476.1, 668.1],
    }
    up = 20
    n_up = int(n_t * tr * up)
    t = np.arange(0, 32, 1 / up)
    hrf = np.exp(-t / 4.0) * t**2
    hrf /= hrf.sum()
    cols = []
    for k in sorted(onsets):
        box = np.zeros(n_up)
        for o in onsets[k]:
            box[int(o * up) : int((o + 18.0) * up)] = 1.0
        cols.append(np.convolve(box, hrf)[:n_up][:: int(up * tr)][:n_t])
    return torch.tensor(np.stack(cols, axis=1))


def test_component_task_fit_controls_the_familywise_error_rate():
    """A criterion that fires by default must not flag noise. With 60 components a
    nominal 0.05 per component would flag ~3 every run; the max-z null is what holds
    the FAMILYWISE rate at 0.05 instead.

    Measured over 40 trials at 400 surrogates: 3/40 runs flagged anything (0.075,
    inside binomial noise of 0.05). This test uses fewer trials to stay fast and
    asserts the loose bound that would still have caught an uncorrected criterion,
    which would fire on essentially every run.
    """
    design = _pilot_design()
    n_t, k = design.shape[0], 60
    fired = sum(
        len(
            component_task_fit(_ar1(n_t, k, 100 + s), design, polort=6, n_surrogates=300, seed=s)[
                "flagged"
            ]
        )
        > 0
        for s in range(12)
    )
    # Uncorrected, this would be ~12/12. Corrected it is ~1/12.
    assert fired <= 4, f"{fired}/12 noise runs flagged something — the correction is not holding"


def test_component_task_fit_finds_a_genuinely_task_locked_component():
    """The case the SPATIAL scorer misses: a task-locked component whose energy share
    is unremarkable. Power must rise with how strongly the task drives it."""
    design = _pilot_design()
    n_t, k, planted = design.shape[0], 60, 7

    hits = {}
    for snr in (0.3, 1.0):
        found = 0
        for s in range(8):
            rng = np.random.default_rng(500 + s)
            m = _ar1(n_t, k, 500 + s)
            mix = torch.tensor(design.numpy() @ rng.standard_normal(design.shape[1]))
            m[:, planted] = m[:, planted] + snr * mix / mix.std()
            res = component_task_fit(m, design, polort=6, n_surrogates=300, seed=s)
            found += planted in res["flagged"]
        hits[snr] = found

    assert hits[1.0] >= 6, hits  # a strong component is caught nearly always
    assert hits[1.0] > hits[0.3], hits  # and power is monotone in the driving strength


def test_component_task_fit_uses_the_omnibus_not_the_strongest_condition():
    """The statistic is the joint R^2 on the whole design. A component driven by a
    condition that is NOT the first must score just as well as one driven by the first
    — conditions activate different tissue and the strongest is not known in advance.
    """
    design = _pilot_design()
    n_t, k = design.shape[0], 20
    scores = []
    for cond in range(design.shape[1]):
        m = _ar1(n_t, k, 7)
        m[:, 3] = m[:, 3] + 1.2 * design[:, cond] / design[:, cond].std()
        res = component_task_fit(m, design, polort=6, n_surrogates=300, seed=0)
        assert 3 in res["flagged"], (cond, res["flagged"])
        scores.append(float(res["z"][3]))
    # No condition is privileged: every one is detected. The evidence is not equal --
    # condition 1 has five blocks where the others have six, and fewer blocks is
    # genuinely less power -- but no condition is structurally invisible, which is what
    # a per-condition statistic keyed on the "strongest" label would risk.
    assert min(scores) > 0.35 * max(scores), scores


def _periodic_block(n_t=225, tr=3.5, period=20.0, up=20):
    """A PURE periodic block design — the ~2 DoF case the module warns about."""
    n_up = int(n_t * tr * up)
    t = np.arange(0, 32, 1 / up)
    h = np.exp(-t / 4.0) * t**2
    h /= h.sum()
    box = np.zeros(n_up)
    for on in np.arange(10, n_t * tr, period):
        box[int(on * up) : int((on + period / 2) * up)] = 1.0
    return torch.tensor(np.convolve(box, h)[:n_up][:: int(up * tr)][:n_t])[:, None]


def test_component_task_fit_declines_when_the_component_shares_the_design_band():
    """ "Nothing flagged" must not mean two different things.

    The danger regime is NOT a block design as such — it is a spectral coincidence
    between a COMPONENT and the design. Measured, all against the same periodic 20 s
    block design:

        broadband (AR(1)) components   ~195 effective DoF   -> informative
        narrowband components at 20 s    ~5 effective DoF   -> declines

    A broadband component genuinely cannot fake a narrowband design, so the criterion
    is perfectly usable on a block design as long as the components are broadband. It
    is the narrowband case that must decline: random-phase sinusoids at the design's
    own frequency reach R^2 0.77-0.83 with no task relation whatsoever, and the max-z
    correction is anti-conservative there because such components are not independent.
    """
    design = _periodic_block()
    n_t = design.shape[0]

    broadband = component_task_fit(_ar1(n_t, 40, 11), design, polort=6, n_surrogates=400)
    assert broadband["informative"], broadband["uninformative_reason"]
    assert broadband["eff_dof"] > 50

    rng = np.random.default_rng(0)
    tt = np.arange(n_t) * 3.5
    narrow = torch.tensor(
        np.stack([np.sin(2 * np.pi * tt / 20.0 + rng.uniform(0, 2 * np.pi)) for _ in range(12)], 1)
    )
    res = component_task_fit(narrow, design, polort=6, n_surrogates=400)
    assert res["eff_dof"] < 10
    assert not res["informative"]
    # Declining means dropping NOTHING, not dropping the lucky ones. Some of these
    # sinusoids do cross the raw cut — that is precisely why the gate exists.
    assert res["flagged"] == []
    assert "effective DoF" in res["uninformative_reason"]

    # The same components against an irregular design are back in the safe regime,
    # which confirms the gate keys on the coincidence and not on the components alone.
    ok = component_task_fit(narrow, _pilot_design(), polort=6, n_surrogates=400)
    assert ok["informative"] and ok["eff_dof"] > 10


def test_warp_project_out_is_chunk_invariant_and_preserves_the_mean(monkeypatch):
    """The projection must give the same answer at any chunk size.

    Regression: this ran unchunked and materialised the whole (T, S) field in float64
    three times over. On a real 160x160x114x225 multi-echo run that is 14.7 GB, and it
    OOMed a 16 GB card after the ICA sweep had already filled it — the decomposition
    fits comfortably, so the entire memory cost lived here.

    Chunk-invariance is the property worth pinning, not the chunk size: a projection
    along time is independent per voxel, so any split must be exact, and a bug that
    mixed voxels across a chunk boundary would show up here and nowhere else.
    """
    import fastfuncstuff.memory as ffs_memory
    from fastfuncstuff.processing import locomoco as loco

    torch.manual_seed(0)
    nx, ny, nz, nt, m = 7, 6, 5, 40, 3
    # A large per-voxel mean beside a small displacement — the case float64 is for, and
    # the one where dropping/restoring the mean incorrectly would be visible.
    disp = torch.randn(nx, ny, nz, nt).double() * 0.3 + 1.7
    bad = torch.randn(nt, m).double()
    comps = [(2, disp)]

    results = []
    for chunk in (10**9, 97, 13, 1):  # one shot, then progressively silly splits
        # Patched at the source module: warp_project_out imports it inside the call.
        monkeypatch.setattr(ffs_memory, "estimate_chunk_size", lambda *a, **k: chunk)
        results.append(dict(loco.warp_project_out(comps, {2: bad}))[2])

    for r in results[1:]:
        assert torch.equal(results[0], r), "chunking changed the result"

    out = results[0]
    # The named time courses are gone...
    q, _ = torch.linalg.qr(bad - bad.mean(dim=0, keepdim=True))
    resid = out.reshape(-1, nt).T.double()
    resid = resid - resid.mean(dim=0, keepdim=True)
    assert float((q.T @ resid).abs().max()) < 1e-5
    # ...and the per-voxel temporal MEAN survived. It is not a component, and dropping
    # it would move every voxel by its own average displacement.
    assert torch.allclose(out.mean(-1), disp.mean(-1).float(), atol=1e-5)


def test_ica_rank_sweep_extends_past_a_grid_edge_winner():
    """An argmax on the grid boundary means the grid is wrong, not that the edge is best.

    Regression from a real run: the sweep's 0.15 floor (rank 33 of 224) won outright
    with enrichment falling monotonically above it, so the optimum was never searched.
    The sweep must now walk outward until the winner is interior.

    Driven with a stub scorer rather than real ICA fits: what is under test is the
    SEARCH, and a real decomposition would make this slow and non-deterministic.
    """
    from fastfuncstuff.cli import locomoco as cli

    tried = []

    def fake_ica(comps, n_components=None, pca_components=None, device=None):
        tried.append(n_components)
        # Monotonically better toward low rank, with the true optimum at 4 — well
        # below any grid floor the coarse pass would place.
        return (torch.zeros(8, 1), [(1, torch.zeros(2, 2, 2, 1))], [torch.zeros(2, 2, 2)], None)

    def fake_enrichment(values, active, mask):
        return {"enrichment": 100.0 / max(tried[-1], 1)}

    import fastfuncstuff.processing.locomoco as loco_proc
    import fastfuncstuff.stats.task_coupling as tc_mod

    old_ica, old_enr = loco_proc.warp_ica_basis, tc_mod.map_enrichment
    loco_proc.warp_ica_basis, tc_mod.map_enrichment = fake_ica, fake_enrichment
    try:
        best, _got = cli._ica_rank_sweep(
            [(1, torch.zeros(2, 2, 2, 8))], None, None, [33, 67, 100], torch.device("cpu")
        )
    finally:
        loco_proc.warp_ica_basis, tc_mod.map_enrichment = old_ica, old_enr

    # It must have gone BELOW the grid floor rather than stopping at 33.
    assert min(tried) < 33, tried
    assert best < 33, (best, tried)
    # ...and kept halving until it hit the floor of 2.
    assert best <= 4, (best, tried)


# ── the AFNI stat tag and the PSC amplitude the locomoco CLI writes ──────────────


def test_coupling_stataux_matches_afni_correl_t2p_dof():
    """fico params must give AFNI the dof this drift-partial correlation really has.

    correl_t2p(rho, nsam, nfit, nort) = incbeta(1-rho^2, (nsam-nfit-nort)/2, nfit/2),
    so the residual dof AFNI uses is nsam-nfit-nort. Our r correlates two vectors that
    have already had a (polort+1)-column drift basis projected out, against one
    regressor: T - (polort+1) - 1.
    """
    from fastfuncstuff.cli.task_events import coupling_stataux

    # n_ort counts EVERY column removed from both sides: polort+1 drift columns here.
    aux = coupling_stataux(n_t=120, n_ort=4, n_sub=2)
    assert set(aux) == {0, 1}
    for code, params in aux.values():
        assert code == 2  # FUNC_COR_TYPE
        nsam, nfit, nort = params
        assert (nsam, nfit, nort) == (120.0, 1.0, 4.0)
        assert nsam - nfit - nort == 120 - 3 - 2
    # A fit that also carries 5 nuisance regressors must declare their dof, or AFNI
    # credits it with dof it did not have.
    ((_c, p_pc),) = set(coupling_stataux(n_t=120, n_ort=4 + 5, n_sub=1).values())
    assert p_pc == (120.0, 1.0, 9.0)


def test_psc_betas_recover_a_planted_percent_signal_change():
    """A voxel modulated by 5% of its own mean must read ~5 %sig, whatever the units."""
    from fastfuncstuff.cli.task_events import psc_betas
    from fastfuncstuff.stats.task_coupling import task_coupling

    n_t = 120
    x = _block_design(n_t)
    base = 800.0
    shape = (4, 4, 3)
    series = torch.full((*shape, n_t), base)
    series[1, 1, 1] = base * (1.0 + 0.05 * (x[:, 0] - x[:, 0].mean()))
    mask = torch.zeros(shape)
    mask[1, 1, 1] = 1
    tc = task_coupling(series, x, polort=1, mask=mask)
    psc = psc_betas(tc, x.numpy(), series.mean(dim=3), mask)
    # The regressor swings 0->1, so beta IS the full modulation depth.
    assert abs(float(psc[1, 1, 1, 0]) - 5.0) < 0.2
    # Untouched voxels are flat, not noise: a constant voxel must not score.
    assert float(psc[0, 0, 0, 0]) == 0.0


def test_psc_betas_are_scale_free():
    """Doubling the intensity units must not change the percentage."""
    from fastfuncstuff.cli.task_events import psc_betas
    from fastfuncstuff.stats.task_coupling import task_coupling

    n_t = 100
    x = _block_design(n_t)
    shape = (3, 3, 2)
    mask = torch.ones(shape)
    out = []
    for scale in (1.0, 37.0):
        series = torch.full((*shape, n_t), 500.0 * scale)
        series[0, 0, 0] = 500.0 * scale * (1.0 + 0.03 * x[:, 0])
        tc = task_coupling(series, x, polort=1, mask=mask)
        out.append(float(psc_betas(tc, x.numpy(), series.mean(dim=3), mask)[0, 0, 0, 0]))
    assert abs(out[0] - out[1]) < 1e-3


def test_save_task_fit_interleaves_coef_and_stat_per_condition(tmp_path):
    """AFNI bucket order: the whole-model R, then view sub-brick 2k, threshold on 2k+1."""
    from fastfuncstuff.cli.task_events import save_task_fit
    from fastfuncstuff.io.afni import read_brick_labels, read_brick_stataux
    from fastfuncstuff.stats.task_coupling import task_coupling

    n_t = 80
    x = torch.stack([_block_design(n_t)[:, 0], _block_design(n_t, block=20)[:, 0]], dim=1)
    shape = (3, 3, 2)
    series = torch.full((*shape, n_t), 500.0)
    series[0, 0, 0] = 500.0 * (1.0 + 0.04 * x[:, 0] + 0.02 * x[:, 1])
    labels = ["faces", "houses"]
    tc = task_coupling(series, x, polort=1, mask=torch.ones(shape), labels=labels)
    psc = torch.zeros(*shape, 2)

    out = tmp_path / "fit.nii.gz"
    save_task_fit(str(out), tc, psc, labels, np.eye(4), 2, n_t)

    import nibabel as nib

    img = nib.load(out)
    assert read_brick_labels(img) == [
        "full_model_R",
        "faces_Coef",
        "faces_Correl",
        "houses_Coef",
        "houses_Correl",
    ]
    aux = read_brick_stataux(img)
    # Sub-brick 0 is the whole-model R and the Coef bricks stay untagged, so the stats
    # are 0 and every 2k+1 after it.
    assert set(aux) == {0, 2, 4}
    # nfit = the design rank for the omnibus, 1 for a single condition -- whose nort
    # absorbs the other n_fit-1 regressors, because the joint fit spent dof on them.
    assert aux[0][0] == 2 and tuple(aux[0][1]) == (n_t, 2.0, 2.0)
    for i in (2, 4):
        code, params = aux[i]
        assert code == 2 and tuple(params) == (n_t, 1.0, 3.0)


def test_component_variance_in_data_recovers_a_planted_share():
    """A component that IS one voxel's whole signal must account for its whole variance."""
    from fastfuncstuff.stats.task_coupling import component_variance_in_data

    n_t, shape = 60, (4, 4, 2)
    g = torch.Generator().manual_seed(3)
    a = torch.randn(n_t, generator=g)
    b = torch.randn(n_t, generator=g)
    data = torch.zeros(*shape, n_t)
    data[0, 0, 0] = 10.0 * a
    data[1, 1, 1] = 10.0 * b
    scores = torch.stack([a, b], dim=1)

    got = component_variance_in_data(scores, data, polort=1)
    # Two components spanning the only two signals present: JOINTLY they account for
    # everything. The per-component shares are marginal and only sum to the joint value
    # when the components are uncorrelated, which is why both are returned.
    assert got["joint_var_data"] > 0.95
    assert float(got["var_data"].sum()) > 0.95
    assert got["var_task"] is None and got["task_frac"] is None


def test_component_variance_in_data_separates_task_share_from_data_share():
    """A big component orthogonal to the design must take variance but not TASK variance."""
    from fastfuncstuff.stats.task_coupling import component_variance_in_data

    n_t, shape = 120, (4, 4, 2)
    x = _block_design(n_t)
    g = torch.Generator().manual_seed(11)
    # A large nuisance mode built to be orthogonal to the (detrended) design.
    nuis = torch.randn(n_t, generator=g).double()
    xd = (x[:, 0] - x[:, 0].mean()).double()
    nuis = nuis - xd * (nuis @ xd) / (xd @ xd)

    data = torch.zeros(*shape, n_t, dtype=torch.float64)
    data[0, 0, 0] = 20.0 * nuis + 1.0 * xd
    data[1, 1, 0] = 1.0 * xd
    scores = torch.stack([nuis, xd], dim=1)

    got = component_variance_in_data(scores, data, polort=1, design=x)
    vd, vt, tf = got["var_data"], got["var_task"], got["task_frac"]
    # The nuisance dominates the DATA variance but takes almost none of the TASK
    # variance -- which is exactly the split the table exists to show.
    assert float(vd[0]) > 0.9
    assert float(vt[0]) < 0.05
    assert float(tf[0]) < 0.05
    # The design-aligned component is the mirror image.
    assert float(vt[1]) > 0.9
    assert float(tf[1]) > 0.9


def test_component_variance_task_share_collapses_onto_design_overlap_at_one_condition():
    """With a single condition the two task columns are provably one number.

    The task subspace is then one direction, so the share of the data's task variance a
    component takes IS the share of the component lying in that direction. Reading them
    as two pieces of evidence would double-count; with two conditions they separate,
    because the data decides which task directions carry variance.
    """
    from fastfuncstuff.stats.task_coupling import component_variance_in_data

    n_t, shape = 100, (5, 5, 3)
    g = torch.Generator().manual_seed(0)
    x1 = _block_design(n_t)
    data = torch.randn(*shape, n_t, generator=g, dtype=torch.float64)
    u = torch.randn(n_t, 4, generator=g, dtype=torch.float64)

    one = component_variance_in_data(u, data, polort=2, design=x1)
    assert torch.allclose(one["var_task"], one["task_frac"], atol=1e-10)

    x2 = torch.cat([x1, torch.roll(x1, 7, 0)], dim=1)
    two = component_variance_in_data(u, data, polort=2, design=x2)
    assert not torch.allclose(two["var_task"], two["task_frac"], atol=1e-4)


def test_component_variance_task_frac_is_the_temporal_criterion_statistic():
    """task_frac must BE component_task_fit's r2 — for ICA too, not only for PCA.

    Bug of record: the components were run through _orthonormal_basis as a set, which
    returns the SVD directions of the span — an orthogonal ROTATION of the component
    space. Row k was then labelled "component k" while describing a mixture of all of
    them. Barely visible for PCA (already orthonormal) and wrong for ICA, whose mixing
    is not orthogonal at all.
    """
    from fastfuncstuff.stats.task_coupling import (
        component_task_fit,
        component_variance_in_data,
    )

    n_t, shape = 120, (5, 5, 3)
    g = torch.Generator().manual_seed(1)
    x = _block_design(n_t)
    data = torch.randn(*shape, n_t, generator=g, dtype=torch.float64)
    orthonormal = torch.linalg.qr(torch.randn(n_t, 4, generator=g, dtype=torch.float64))[0]
    oblique = torch.randn(n_t, 4, generator=g, dtype=torch.float64)  # an ICA-like mixing

    for u in (orthonormal, oblique):
        cv = component_variance_in_data(u, data, polort=2, design=x)
        tf = component_task_fit(u, x, polort=2, n_surrogates=50)
        assert torch.allclose(
            cv["task_frac"], torch.as_tensor(tf["r2"], dtype=torch.float64), atol=1e-9
        )


def test_joint_task_share_is_the_span_and_matches_the_sum_only_when_orthogonal():
    """The joint figure is the design's overlap with the component SPAN, not a sum.

    For components that are orthonormal AND already free of drift the two coincide
    exactly. Residualizing drift out of components that carry it makes them
    non-orthogonal, and the span then holds a DIFFERENT amount than the marginals sum
    to — in either direction, since correlated predictors can suppress as well as
    reinforce. That is why the joint is computed as a projection and reported
    separately, rather than left to be added up by eye.
    """
    from fastfuncstuff.glm.core import construct_polynomial_matrix
    from fastfuncstuff.stats.task_coupling import component_variance_in_data

    n_t, shape = 120, (5, 5, 3)
    g = torch.Generator().manual_seed(2)
    x = _block_design(n_t)
    data = torch.randn(*shape, n_t, generator=g, dtype=torch.float64)
    qn = torch.linalg.qr(
        construct_polynomial_matrix(n_t, 3, device=torch.device("cpu"), dtype=torch.float64)
    )[0]

    clean = torch.randn(n_t, 5, generator=g, dtype=torch.float64)
    clean = torch.linalg.qr(clean - qn @ (qn.T @ clean))[0]
    got = component_variance_in_data(clean, data, polort=3, design=x)
    assert abs(got["joint_var_task"] - float(got["task_frac"].sum())) < 1e-9

    drifty = torch.linalg.qr(torch.randn(n_t, 5, generator=g, dtype=torch.float64))[0]
    drifty = torch.linalg.qr(drifty + 2.0 * qn[:, :1] + 1.5 * qn[:, 1:2])[0]
    got = component_variance_in_data(drifty, data, polort=3, design=x)
    assert abs(got["joint_var_task"] - float(got["task_frac"].sum())) > 1e-6
    # Whichever way it went, the joint is a projection and so is bounded.
    assert 0.0 <= got["joint_var_task"] <= 1.0


def test_task_coupling_nuisance_columns_are_partialled_out():
    """Extra nuisance columns must enter the same projection the polynomials do."""
    from fastfuncstuff.stats.task_coupling import task_coupling

    n_t, shape = 120, (4, 4, 2)
    g = torch.Generator().manual_seed(5)
    x = _block_design(n_t)
    xd = (x[:, 0] - x[:, 0].mean()).double()
    nuis = torch.randn(n_t, generator=g).double()
    nuis = nuis - xd * (nuis @ xd) / (xd @ xd)  # orthogonal to the design

    data = torch.zeros(*shape, n_t, dtype=torch.float64)
    data[0, 0, 0] = 100.0 + 3.0 * xd + 8.0 * nuis
    mask = torch.zeros(shape)
    mask[0, 0, 0] = 1

    plain = task_coupling(data, x, polort=1, mask=mask)
    with_n = task_coupling(data, x, polort=1, mask=mask, nuisance=nuis[:, None])
    # The planted slope is 3. Carrying the nuisance column in the model recovers it
    # exactly; leaving it out biases the estimate, because a column orthogonal to the
    # RAW design is not orthogonal to it after the polynomials come out of both.
    assert abs(float(with_n.beta[0, 0, 0, 0]) - 3.0) < 1e-6
    assert abs(float(plain.beta[0, 0, 0, 0]) - 3.0) > 0.1
    # And the partial correlation rises: the variance the column removed was all
    # unexplained by the task.
    assert float(with_n.r[0, 0, 0, 0].abs()) > float(plain.r[0, 0, 0, 0].abs()) + 0.3
    # ...and the dof it costs is recorded, so the chance reference moves with it.
    assert with_n.n_nuisance == 1
    assert with_n.chance_share > plain.chance_share

    # A nuisance column that IS the design leaves nothing for the task to explain.
    with pytest.raises(ValueError, match="collinear"):
        task_coupling(data, x, polort=1, mask=mask, nuisance=xd[:, None])


# ── the joint fit ────────────────────────────────────────────────────────────


def _multi_design(n_t: int, n_k: int = 5, seed: int = 3) -> torch.Tensor:
    """Several jittered event-related conditions, the shape the marginal fit fails on."""
    rng = np.random.default_rng(seed)
    k = np.exp(-np.arange(0, 24) / 4.0)
    k /= k.sum()
    cols = []
    onsets = rng.permutation(np.arange(4, n_t - 24))[: n_k * 8].reshape(n_k, 8)
    for row in onsets:
        stick = np.zeros(n_t)
        stick[row] = 1.0
        c = np.convolve(stick, k)[:n_t]
        cols.append(c - c.mean())
    return torch.tensor(np.stack(cols, axis=1))


def test_joint_fit_matches_an_ordinary_least_squares_glm():
    """beta_joint and r_joint ARE the GLM's betas and partial correlations.

    The whole point of the joint path is that a reader used to 3dDeconvolve betas and
    t-stats can read these the same way, so it is pinned against an explicit lstsq.
    """
    n_t, n_k = 200, 5
    x = _multi_design(n_t, n_k)
    f = _field(n_t, shape=(8, 8, 4), seed=11)
    for k in range(n_k):
        f += (0.4 + 0.1 * k) * x[:, k]
    tc = task_coupling(f, x, polort=2)

    y = f.reshape(-1, n_t).double()
    pol = construct_polynomial_matrix(n_t, 2, torch.device("cpu"), dtype=torch.float64)
    d = torch.cat([x, pol], dim=1)
    b = torch.linalg.lstsq(d, y.T).solution
    dof = n_t - d.shape[1]
    s2 = ((y.T - d @ b) ** 2).sum(dim=0) / dof
    gi = torch.linalg.pinv(d.T @ d)
    t = torch.stack([b[k] / (s2 * gi[k, k]).sqrt() for k in range(n_k)], dim=1)

    assert tc.n_fit == n_k
    assert torch.allclose(tc.beta_joint.reshape(-1, n_k), b[:n_k].T, atol=1e-9)
    assert torch.allclose(tc.r_joint.reshape(-1, n_k), t / (t * t + dof).sqrt(), atol=1e-9)


def test_marginal_r_is_capped_by_the_design_but_the_full_model_r_is_not():
    """A NOISELESS copy of the full task response still reads low per condition.

    This is the bug that sent a real 8-condition run's stage03 maps to nothing while
    its 3-run GLM lit up: each marginal r converges to corr(x_k, sum_j x_j), which is a
    property of the DESIGN, so no amount of response can lift it.
    """
    n_t, n_k = 200, 5
    x = _multi_design(n_t, n_k)
    perfect = x.sum(dim=1)
    f = (perfect[None, None, None, :] + 100.0).repeat(4, 4, 2, 1).clone()
    tc = task_coupling(f, x, polort=2)

    # The ceiling is corr(x_k, sum_j x_j) measured the way the fit sees them: after the
    # same polort-2 projection, since that is what tc.r is a partial correlation to.
    q = torch.linalg.qr(
        construct_polynomial_matrix(n_t, 2, torch.device("cpu"), dtype=torch.float64)
    )[0]
    xd = x - q @ (q.T @ x)
    pd = perfect - q @ (q.T @ perfect)
    ceiling = torch.tensor(
        [float(torch.corrcoef(torch.stack([xd[:, k], pd]))[0, 1]) for k in range(n_k)],
        dtype=torch.float64,
    )
    assert float(ceiling.max()) < 0.75  # the design's own limit, not the data's
    assert torch.allclose(tc.r[0, 0, 0].abs(), ceiling.abs(), atol=0.02)
    assert float(tc.r_full[0, 0, 0]) == pytest.approx(1.0, abs=1e-6)
    assert float(tc.r_joint[0, 0, 0].abs().min()) > 0.99


def test_full_model_r_is_the_multiple_correlation_and_stays_bounded():
    n_t, n_k = 160, 4
    x = _multi_design(n_t, n_k)
    f = _field(n_t, shape=(6, 6, 4), seed=5) + 0.5 * x.sum(dim=1)
    tc = task_coupling(f, x, polort=2)
    r_full = tc.r_full.reshape(-1)
    assert float(r_full.min()) >= 0.0 and float(r_full.max()) <= 1.0
    # R of the whole design is never below the best single-condition |r| it contains.
    assert bool((r_full + 1e-9 >= tc.r.abs().amax(dim=-1).reshape(-1)).all())
    assert float(r_full.median()) > 0.4


def test_responding_mask_ranks_on_the_whole_model_not_one_condition():
    """A voxel responding a little to EVERY condition is what the active mask must find.

    Ranking on the largest single-condition |r| is what stage03 used to do, and it
    misses exactly this voxel — the one an ordinary GLM lights up hardest.
    """
    n_t, n_k = 200, 8
    x = _multi_design(n_t, n_k)
    shape = (10, 10, 4)
    f = _field(n_t, shape=shape, seed=2)
    f[:1] += 0.30 * x.sum(dim=1)  # responds to all of them, to none of them strongly
    mask = torch.ones(shape, dtype=torch.bool)
    tc = task_coupling(f, x, polort=2, mask=mask)

    def hit(active):
        return float(active[:1].double().mean())

    by_full, _, _ = responding_mask(tc.r_full, mask, 0.1)
    by_marginal, _, _ = responding_mask(tc.r, mask, 0.1)
    assert hit(by_full) > 0.9
    assert hit(by_full) - hit(by_marginal) > 0.1


def test_constant_voxels_score_zero_in_the_joint_fit_too():
    n_t, n_k = 120, 4
    x = _multi_design(n_t, n_k)
    f = _field(n_t, task=x[:, :1], amp=0.3)
    f[1, 0, 0] = 7.0  # constant, non-zero
    tc = task_coupling(f, x, polort=2)
    assert float(tc.r_joint[1, 0, 0].abs().max()) == 0.0
    assert float(tc.beta_joint[1, 0, 0].abs().max()) == 0.0
    assert float(tc.r_full[1, 0, 0]) == 0.0
