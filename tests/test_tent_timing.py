"""Sub-TR sample timing for FIR / TENT / CSPLIN designs.

A volume sampled at ``n * TR + t0`` (slice-time corrected to ``-tzero t0``)
sees each event ``t0`` later than one sampled at ``n * TR``.  The design has
to know that, or every estimated response is shifted by ``t0``.
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.design.builder import build_per_run_task_designs
from fastfuncstuff.design.matrices import (
    make_csplin_design,
    make_fir_design,
    make_tent_design,
    onsets_to_tr_matrix,
)

CPU = torch.device("cpu")


@pytest.mark.parametrize("fn", [make_tent_design, make_csplin_design])
def test_microtime_offset_equals_shifting_onsets_earlier(fn):
    onsets = np.array([3.3, 17.8, 31.1, 44.6])
    kw = dict(bot=0.0, top=12.0, tr=2.0, n_timepoints=40, device=CPU)
    shifted_samples = fn([onsets], microtime_offset=0.7, **kw)
    shifted_onsets = fn([onsets - 0.7], **kw)
    torch.testing.assert_close(shifted_samples, shifted_onsets, atol=1e-5, rtol=0)


def test_builder_fir_rounds_relative_to_sample_time():
    # 3.1 s at TR 2: nearest sample of an n*TR clock is n=2 (4.0 s); with the
    # volumes sampled 1 s into each TR the nearest sample is n=1 (3.0 s).
    kw = dict(n_timepoints_per_run=[20], tr=2.0, basis="FIR", fir_window_s=6.0, device=CPU)
    plain = build_per_run_task_designs([[np.array([3.1])]], **kw).per_run[0]
    offset = build_per_run_task_designs([[np.array([3.1])]], microtime_offset=1.0, **kw).per_run[0]
    assert plain[:, 0].nonzero().flatten().tolist() == [2]
    assert offset[:, 0].nonzero().flatten().tolist() == [1]


def test_builder_fir_fill_durations_matches_block_fir():
    """The GLM-family FIR (blocks, via onsets_to_tr_matrix) is reproduced per run."""
    onsets = [[np.array([2.0, 21.0]), np.array([5.0])]]
    n_tp = [30, 25]
    tr, dur, n_lags = 2.0, 5.0, 4
    got = build_per_run_task_designs(
        onsets,
        n_tp,
        tr,
        basis="FIR",
        fir_window_s=n_lags * tr,
        durations_per_condition=[dur],
        fir_fill_durations=True,
        device=CPU,
    ).per_run
    for r, n in enumerate(n_tp):
        mat, _ = onsets_to_tr_matrix([[onsets[0][r]]], [0], n, tr, durations=[dur], device=CPU)
        want = make_fir_design(mat, n_lags, n, device=CPU)
        torch.testing.assert_close(got[r], want)
