"""Inner-LK convergence reporting (`N iters, conv @ a,b,c`) — see locomoco.py."""

import torch

from fastfuncstuff.processing.locomoco import (
    optical_flow_lk_3d_axes,
    summarize_flow_convergence,
)


def _steps(rows):
    """(n_calls, n_levels, n_iters) from a list of per-level per-iteration sequences."""
    return torch.tensor([rows], dtype=torch.float32)


def test_converged_level_reports_the_iteration_it_settled():
    # Geometric decay: 1.0 -> 0.5 -> 0.25 -> 0.125 -> 0.0625. The first step at or below
    # 10% of the peak is the 5th (0.0625 > 0.1? no: 0.0625 <= 0.1), so conv @ 5.
    s = _steps([[1.0, 0.5, 0.25, 0.125, 0.0625]])
    assert summarize_flow_convergence(s, 5) == "5 iters, conv @ 5"


def test_peak_not_first_step_is_the_denominator():
    # A level that opens small (field arrived already good from the coarser level) then
    # takes its real step second. Against step_0 this would never look converged.
    s = _steps([[0.01, 1.0, 0.5, 0.05]])
    assert "conv @ 4" in summarize_flow_convergence(s, 4)


def test_capped_level_is_flagged_and_carries_its_final_step():
    # Never falls to 10% of peak -> capped, and the final step is reported so a big
    # residual step ("raise -iters") is distinguishable from a negligible one.
    s = _steps([[1.0, 0.9, 0.8, 0.7]])
    out = summarize_flow_convergence(s, 4)
    assert out == "4 iters, conv @ 4*  (Δ 0.700 vox)"


def test_capped_at_a_negligible_step_reads_differently():
    # Same shape of failure, three orders of magnitude smaller: the cap is doing no harm
    # and the printed delta is what says so.
    s = _steps([[1e-3, 9e-4, 8e-4, 7e-4]])
    assert summarize_flow_convergence(s, 4) == "4 iters, conv @ 4*  (Δ 0.001 vox)"


def test_levels_read_coarsest_first_and_mix():
    s = _steps([[1.0, 0.05, 0.01, 0.01], [1.0, 1.0, 0.9, 0.9], [1.0, 0.5, 0.02, 0.01]])
    out = summarize_flow_convergence(s, 4)
    assert out.startswith("4 iters, conv @ 2,4*,3")


def test_one_stubborn_frame_does_not_flag_the_level():
    # 1 of 10 frames capped is under FLOW_CONV_CAPPED_FRAC: no star, no advice to change
    # -iters on the strength of a single frame.
    good = [1.0, 0.5, 0.05, 0.01]
    bad = [1.0, 1.0, 1.0, 1.0]
    s = torch.tensor([[good]] * 9 + [[bad]], dtype=torch.float32)
    assert summarize_flow_convergence(s, 4) == "4 iters, conv @ 3"


def test_empty_input_is_silent():
    assert summarize_flow_convergence(torch.zeros(0, 0, 0), 4) == ""


def test_conv_out_shape_matches_levels_and_iters():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    fixed = torch.rand(1, 16, 16, 16, device=dev)
    moving = torch.roll(fixed, shifts=1, dims=1)
    conv: list[torch.Tensor] = []
    optical_flow_lk_3d_axes(
        [fixed],
        [moving],
        [0],
        torch.ones(1, 1, device=dev),
        n_levels=2,
        n_iters=3,
        conv_out=conv,
    )
    assert len(conv) == 1
    assert conv[0].shape == (2, 3)  # (n_levels, n_iters), coarsest first
    assert torch.isfinite(conv[0]).all()


def test_conv_out_records_only_the_levels_an_axis_was_solved_on():
    # Per-axis -levels: the pyramid is as deep as the larger value, and every level still
    # solves SOMETHING, so the record is one row per pyramid level either way.
    torch.manual_seed(0)
    dev = torch.device("cpu")
    fixed = [torch.rand(1, 16, 16, 16, device=dev)]
    moving = [torch.roll(fixed[0], shifts=1, dims=1)]
    conv: list[torch.Tensor] = []
    optical_flow_lk_3d_axes(
        fixed,
        moving,
        [0, 1],
        torch.ones(2, 1, device=dev),
        n_levels=[1, 2],
        n_iters=3,
        conv_out=conv,
    )
    assert conv[0].shape == (2, 3)


def test_conv_out_is_none_by_default_and_costs_nothing():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    fixed = torch.rand(1, 16, 16, 16, device=dev)
    moving = torch.roll(fixed, shifts=1, dims=1)
    a = optical_flow_lk_3d_axes(
        [fixed], [moving], [0], torch.ones(1, 1, device=dev), n_levels=2, n_iters=3
    )
    conv: list[torch.Tensor] = []
    b = optical_flow_lk_3d_axes(
        [fixed],
        [moving],
        [0],
        torch.ones(1, 1, device=dev),
        n_levels=2,
        n_iters=3,
        conv_out=conv,
    )
    # Recording must not perturb the estimate it is reporting on.
    assert torch.equal(a[0], b[0])
