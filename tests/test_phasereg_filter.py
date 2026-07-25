"""Phase-filter wiring for ffs_phasereg.

Regression test for the bug where ``-phase_filter sgf`` was a silent no-op:
the CLI exposed the mode as ``"sgf"`` but ``_apply_sgf`` branched on the
internal name ``"fixed"``, so ``"sgf"`` matched no branch and the unfiltered
phase was returned — making sgf output bit-identical to none.
"""

import torch

from fastfuncstuff.phasereg.core import phase_regress


def _synth(n_vox=64, n_tp=120, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Smooth macrovascular phase drift + high-frequency noise so SGF has
    # something to remove.
    t = torch.linspace(0, 6.28, n_tp)
    slow = torch.sin(t)[None, :].repeat(n_vox, 1)
    noise = 0.3 * torch.randn(n_vox, n_tp, generator=g)
    phase = 4.5 + 0.4 * (slow + noise)
    magnitude = 1400 + 200 * slow + 50 * torch.randn(n_vox, n_tp, generator=g)
    return magnitude, phase


def test_sgf_filter_changes_output():
    mag, pha = _synth()
    common = dict(
        tr=2.5,
        task_removal="none",
        max_poly_degree=3,
        regression="deming",
        phi_method="residual",
        device=torch.device("cpu"),
        verbose=False,
    )
    res_none = phase_regress(mag.clone(), pha.clone(), phase_filter="none", **common)
    res_sgf = phase_regress(mag.clone(), pha.clone(), phase_filter="sgf", **common)

    # The whole point of SGF: filtered phase drives a different slope/correction.
    assert not torch.allclose(res_none.magnitude_corrected, res_sgf.magnitude_corrected), (
        "phase_filter='sgf' produced identical output to 'none' (filter is a no-op)"
    )


def test_explore_populates_param_maps():
    mag, pha = _synth()
    common = dict(
        tr=2.5,
        task_removal="none",
        max_poly_degree=3,
        device=torch.device("cpu"),
        verbose=False,
    )
    # sgf mode leaves the param maps unpopulated; explore fills them in.
    res_sgf = phase_regress(mag.clone(), pha.clone(), phase_filter="sgf", **common)
    assert res_sgf.sgf_window_map is None
    assert res_sgf.sgf_order_map is None

    res_ex = phase_regress(mag.clone(), pha.clone(), phase_filter="explore", **common)
    assert res_ex.sgf_window_map is not None
    assert res_ex.sgf_order_map is not None
    assert res_ex.sgf_window_map.shape == (mag.shape[0],)
    # Chosen windows are odd, or 0 where the unfiltered series won the search
    # (Barry & Gore step 3). Grid bounds default to n_tp//2 and window_max//4.
    n_tp = mag.shape[1]
    wm = res_ex.sgf_window_map
    om = res_ex.sgf_order_map
    unfiltered = wm == 0
    assert bool((wm[~unfiltered] % 2 == 1).all()), "chosen windows must be odd"
    assert int(wm.max()) <= n_tp // 2
    assert bool((om[unfiltered] == 0).all()), "order must be 0 where window is 0"
    assert int(om[~unfiltered].min()) >= 2
    assert int(om.max()) <= (n_tp // 2) // 4


def test_explore_grid_bounds_respected():
    mag, pha = _synth()
    res = phase_regress(
        mag.clone(),
        pha.clone(),
        tr=2.5,
        task_removal="none",
        phase_filter="explore",
        sgf_window_max=13,
        sgf_order_max=3,
        sgf_step=4,
        device=torch.device("cpu"),
        verbose=False,
    )
    assert int(res.sgf_window_map.max()) <= 13
    assert int(res.sgf_order_map.max()) <= 3


def test_unknown_filter_raises():
    mag, pha = _synth()
    try:
        phase_regress(
            mag,
            pha,
            tr=2.5,
            phase_filter="bogus",
            device=torch.device("cpu"),
            verbose=False,
        )
    except ValueError as e:
        assert "phase_filter" in str(e) or "sgf_mode" in str(e)
    else:
        raise AssertionError("unknown phase_filter should raise ValueError")
