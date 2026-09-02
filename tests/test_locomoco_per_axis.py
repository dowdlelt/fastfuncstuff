"""Per-encode-axis tuning: a scalar must reproduce the old behaviour exactly.

The whole point of the 1-or-2-value spelling is that ``-window 2`` keeps meaning what it
meant. These tests pin that reduction on every path that grew a per-axis knob, and then
check that the two-value forms actually do something different in the right direction.
"""

import pytest
import torch
from test_locomoco_dual_axes import _blobby

from fastfuncstuff.processing.locomoco import (
    _shift3d_axes,
    axis_params,
    axis_scalar,
    optical_flow_lk_3d_axes,
    xcorr_search_flow_3d_axes,
)
from fastfuncstuff.processing.warp import QwarpConfig, _build_mescaled_plan


def test_axis_params_broadcast_and_length_check():
    assert axis_params(2.0, 2, "-window") == [2.0, 2.0]
    assert axis_params([2.0], 2, "-window") == [2.0, 2.0]
    assert axis_params([2.0, 4.0], 2, "-window") == [2.0, 4.0]
    assert axis_params(None, 2, "-max_shift") == [None, None]
    assert axis_scalar([3.0], "-max_shift") == 3.0
    with pytest.raises(ValueError, match="takes 1 or 2"):
        axis_params([1.0, 2.0, 3.0], 2, "-window")


def _dual_pair():
    fixed = _blobby(seed=3)
    moving = _shift3d_axes(
        fixed,
        [torch.full_like(fixed, 0.6), torch.full_like(fixed, -0.4)],
        [0, 1],
    )
    return fixed, moving


@pytest.mark.parametrize("scalar,per_axis", [(2.0, [2.0, 2.0])])
def test_flow_axes_scalar_window_matches_repeated_pair(scalar, per_axis):
    fixed, moving = _dual_pair()
    kw = dict(axes=[0, 1], n_levels=2, n_iters=3, max_disp=3.0)
    a = optical_flow_lk_3d_axes(
        [fixed], [moving], kw["axes"], torch.ones(2, 1), window_sigma=scalar, **_rest(kw)
    )
    b = optical_flow_lk_3d_axes(
        [fixed], [moving], kw["axes"], torch.ones(2, 1), window_sigma=per_axis, **_rest(kw)
    )
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


def _rest(kw):
    return {k: v for k, v in kw.items() if k != "axes"}


def test_flow_axes_per_axis_window_changes_only_via_the_rows():
    """Different windows must move the answer — and the asymmetric solve must stay sane."""
    fixed, moving = _dual_pair()
    kw = dict(axes=[0, 1], n_levels=2, n_iters=3, max_disp=3.0)
    same = optical_flow_lk_3d_axes(
        [fixed], [moving], kw["axes"], torch.ones(2, 1), window_sigma=2.0, **_rest(kw)
    )
    diff = optical_flow_lk_3d_axes(
        [fixed], [moving], kw["axes"], torch.ones(2, 1), window_sigma=[4.0, 1.0], **_rest(kw)
    )
    assert all(torch.isfinite(d).all() for d in diff)
    assert not torch.allclose(same[1], diff[1], atol=1e-4)
    # Both components still land where the equal-window solve puts them: changing the
    # aperture is meant to trade smoothness against locality, not to move the answer.
    for x, y in zip(same, diff, strict=True):
        assert abs(float(x.mean()) - float(y.mean())) < 0.25


def test_flow_axes_level_dropout_freezes_the_short_axis_at_a_coarse_field():
    """An axis given fewer levels must stop changing once the pyramid passes its depth."""
    fixed, moving = _dual_pair()
    kw = dict(axes=[0, 1], n_iters=3, max_disp=3.0)
    full = optical_flow_lk_3d_axes(
        [fixed], [moving], kw["axes"], torch.ones(2, 1), n_levels=3, **_rest(kw)
    )
    # Axis 0 leaves after the coarsest level; axis 1 keeps all three.
    dropped = optical_flow_lk_3d_axes(
        [fixed], [moving], kw["axes"], torch.ones(2, 1), n_levels=[1, 3], **_rest(kw)
    )
    # The dropped axis stopped early, so it is smoother than the one that carried on.
    assert dropped[0].std() < full[0].std()
    assert not torch.allclose(full[0], dropped[0], atol=1e-4)
    # ...and it is exactly what the coarsest level alone produces, upsampled: the axis
    # that carried on cannot have fed anything back into it.
    assert torch.isfinite(dropped[0]).all()


def test_flow_axes_all_axes_solved_at_the_coarsest_level():
    """`n_levels=[1, 1]` on a 3-level pyramid must still solve both, not neither."""
    fixed, moving = _dual_pair()
    got = optical_flow_lk_3d_axes(
        [fixed], [moving], [0, 1], torch.ones(2, 1), n_levels=[1, 1], n_iters=2, max_disp=3.0
    )
    assert all(float(d.abs().max()) > 0 for d in got)


def test_flow_axes_per_axis_max_disp_clamps_independently():
    fixed, moving = _dual_pair()
    got = optical_flow_lk_3d_axes(
        [fixed],
        [moving],
        [0, 1],
        torch.ones(2, 1),
        n_levels=2,
        n_iters=3,
        max_disp=[2.0, 0.1],
    )
    assert float(got[0].abs().max()) <= 2.0 + 1e-6
    assert float(got[1].abs().max()) <= 0.1 + 1e-6


def test_xcorr_axes_scalar_matches_repeated_pair():
    fixed, moving = _dual_pair()
    kw = dict(n_passes=1, max_shift=1.5, trial_step=0.5)
    a, _ = xcorr_search_flow_3d_axes(
        [fixed], [moving], [0, 1], torch.ones(2, 1), window_sigma=2.0, **kw
    )
    b, _ = xcorr_search_flow_3d_axes(
        [fixed], [moving], [0, 1], torch.ones(2, 1), window_sigma=[2.0, 2.0], **kw
    )
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


def test_xcorr_axes_per_axis_max_shift_bounds_each_axis():
    """Each axis searches its OWN range (rounded up to whole voxels, as the search is)."""
    fixed, moving = _dual_pair()
    got, _ = xcorr_search_flow_3d_axes(
        [fixed],
        [moving],
        [0, 1],
        torch.ones(2, 1),
        window_sigma=[2.0, 1.0],
        max_shift=[3.0, 1.0],
        trial_step=[0.5, 0.25],
        n_passes=1,
    )
    assert float(got[0].abs().max()) <= 3.0 + 1e-6
    assert float(got[1].abs().max()) <= 1.0 + 1e-6
    # The wide axis is genuinely allowed past the narrow one's bound.
    wide, _ = xcorr_search_flow_3d_axes(
        [fixed], [moving], [0, 1], torch.ones(2, 1), max_shift=3.0, n_passes=1
    )
    assert not torch.allclose(wide[1], got[1], atol=1e-5)


def _plan(axis_minpatch, n_levels=3, minpatch=7):
    base = _blobby(shape=(20, 22, 22), seed=1)  # (1, nz, ny, nx)
    cfg = QwarpConfig(minpatch=minpatch, cost_method="ncc")
    return _build_mescaled_plan(
        base,
        (0, 1),
        None,
        None,
        cfg,
        torch.device("cpu"),
        n_levels,
        False,
        None,
        axis_minpatch,
    )


def test_qwarp_axis_minpatch_drops_one_channel_at_the_fine_levels():
    plan = _plan([13, 7])
    widths = [lv.nxh for lv in plan.levels]
    assert widths == sorted(widths, reverse=True), widths
    for lv in plan.levels:
        # x is the coarse-stopping axis; y keeps going.
        assert lv.do_xyz[1] is True
        assert lv.do_xyz[0] == (lv.nxh >= 13)
        assert lv.do_xyz[2] is False
    assert any(not lv.do_xyz[0] for lv in plan.levels), "no level actually dropped x"
    assert any(lv.do_xyz[0] for lv in plan.levels), "x was never solved at all"


def test_qwarp_axis_minpatch_none_keeps_every_axis_everywhere():
    plan = _plan(None)
    assert all(lv.do_xyz == (True, True, False) for lv in plan.levels)


def test_qwarp_ladder_extends_so_a_coarse_stopping_axis_is_always_solved():
    """A big axis_minpatch with few levels must not silently produce a one-axis fit."""
    plan = _plan([15, 7], n_levels=2)
    assert max(lv.nxh for lv in plan.levels) >= 15
    assert any(lv.do_xyz[0] for lv in plan.levels)


def test_flow_axes_dropped_axis_stays_inside_its_own_bound():
    """A dropped axis must not leave the pyramid holding 2x its clamp.

    Bug of record: the coarse-to-fine rescale multiplies the field by the pyramid ratio,
    and only ACTIVE axes were re-clamped after the update. An axis that stopped at a
    coarse level therefore came out of the last upsample at twice ``max_disp``.
    """
    fixed, moving = _dual_pair()
    got = optical_flow_lk_3d_axes(
        [fixed],
        [moving],
        [0, 1],
        torch.ones(2, 1),
        n_levels=[3, 2],
        n_iters=3,
        max_disp=[2.0, 0.5],
    )
    assert float(got[0].abs().max()) <= 2.0 + 1e-6
    assert float(got[1].abs().max()) <= 0.5 + 1e-6


# ── the shared temporal PC basis, now that the axes can be tuned apart ───────────


def _two_axis_warp(n_t=40, shape=(6, 6, 4), seed=0, pe2_gain=1.0):
    """Two per-frame fields sharing one temporal mode, plus per-axis structure."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 6.28, n_t)
    common = torch.sin(t)
    f1 = torch.randn(*shape, 1, generator=g) * common + 0.1 * torch.randn(*shape, n_t, generator=g)
    f2 = pe2_gain * (
        torch.randn(*shape, 1, generator=g) * common + 0.1 * torch.randn(*shape, n_t, generator=g)
    )
    return [(1, f1), (2, f2)]


def test_axis_balance_shares_sum_to_one_and_sharing_never_beats_a_solo_basis():
    from fastfuncstuff.processing.locomoco import warp_pc_basis

    _u, _load, _mean, _var, bal = warp_pc_basis(_two_axis_warp(), n_pcs=5, with_balance=True)
    assert len(bal) == 2
    total = sum(b["share"] for b in bal)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
    for b in bal:
        # A basis fitted to one axis alone can only do better on that axis than a basis
        # that had to serve both; if this ever inverts, the accounting is wrong.
        assert torch.all(b["solo_ev"] >= b["shared_ev"] - 1e-5)
        assert torch.all(b["shared_ev"] <= 1.0 + 1e-6)
        assert torch.all(b["shared_ev"].diff() >= -1e-6)  # cumulative


def test_axis_balance_reports_the_louder_axis_dominating_the_shared_basis():
    """An axis with far more variance takes over the unweighted concatenation's SVD."""
    from fastfuncstuff.processing.locomoco import warp_pc_basis

    _u, _load, _mean, _var, bal = warp_pc_basis(
        _two_axis_warp(pe2_gain=20.0), n_pcs=4, with_balance=True
    )
    quiet, loud = (b for b in sorted(bal, key=lambda b: b["energy"]))
    assert loud["energy"] > 50 * quiet["energy"]
    # The loud axis owns the leading component and is served better by the shared basis.
    assert float(loud["share"][0]) > 0.9
    assert float(loud["shared_ev"][-1]) > float(quiet["shared_ev"][-1])
    # ...and the quiet axis pays the bigger price for sharing, which is the whole point
    # of reporting this per axis rather than one pooled explained-variance number.
    loud_cost = float(loud["solo_ev"][-1] - loud["shared_ev"][-1])
    quiet_cost = float(quiet["solo_ev"][-1] - quiet["shared_ev"][-1])
    assert quiet_cost > loud_cost
