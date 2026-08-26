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
