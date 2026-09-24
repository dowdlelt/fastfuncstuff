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


def _glm_family_design(model, bot, top, n_basis, onsets, run_starts, n_tp, tr=2.0, **kw):
    from fastfuncstuff.cli_utils import build_task_design_from_args

    design, _ = build_task_design_from_args(
        model,
        True,
        bot,
        top,
        n_basis,
        onsets,
        [0.0] * len(onsets),
        torch.zeros(1),
        len(onsets),
        n_tp,
        run_starts,
        tr,
        0.1,
        CPU,
        **kw,
    )
    return design


@pytest.mark.parametrize(
    ("spec", "n_cols"),
    [("FIR", 8), ("TENT", 9), ("TENTZERO", 7), ("TENT(0,16,5)", 5), ("TENTzero(0,16,5)", 3)],
)
def test_parsed_basis_count_is_the_design_width_per_condition(spec, n_cols):
    """reml/denoise label and slice betas by n_basis; the design must agree."""
    from fastfuncstuff.cli_utils import parse_hrf_model_args

    info = parse_hrf_model_args(spec, None, [0.0, 0.0], ["a", "b"], 2.0, fir_window_s=16.0)
    assert info["n_basis"] == n_cols
    assert len(info["condition_labels_full"]) == 2 * n_cols
    onsets = [[np.array([3.0, 41.0])], [np.array([20.0])]]
    design = _glm_family_design(
        info["hrf_model_name"], info["fir_bot"], info["fir_top"], info["n_basis"], onsets, [0], 60
    )
    assert design.shape == (60, 2 * n_cols)


def test_glm_family_tent_keeps_conditions_apart_and_runs_separate():
    # Condition b has no events in run 0; a late run-0 event of condition a
    # must not reach into run 1.
    onsets = [[np.array([110.0]), np.array([])], [np.array([]), np.array([10.0])]]
    design = _glm_family_design("TENT", 0.0, 16.0, 9, onsets, [0, 60], 120)
    a, b = design[:, :9], design[:, 9:]
    assert a[60:].abs().sum() == 0  # no spill across the run boundary
    assert b[:60].abs().sum() == 0 and b[60:].abs().sum() > 0


def test_convolved_model_microtime_offset_shifts_the_regressor():
    from fastfuncstuff.cli_utils import build_task_design_from_args

    def spmg1(onset, **kw):
        args = ([[np.array([onset])]], [0.0], torch.zeros(1), 1, 40, [0], 2.0, 0.1, CPU)
        design, _ = build_task_design_from_args("SPMG1", False, None, None, 1, *args, **kw)
        return design[:, 0]

    # Sampling 1 s later in the TR == the event happening 1 s earlier.
    torch.testing.assert_close(spmg1(10.0, microtime_offset=1.0), spmg1(9.0), atol=1e-5, rtol=0)


def test_microtime_offset_must_land_on_the_microtime_grid():
    from fastfuncstuff.cli_utils import microtime_offset_bins

    assert microtime_offset_bins(1.0, 2.0, 0.1) == 10
    with pytest.raises(ValueError, match="multiple"):
        microtime_offset_bins(0.25, 2.0, 0.1)
    with pytest.raises(ValueError, match="TR"):
        microtime_offset_bins(2.0, 2.0, 0.1)


_AFNI_TAXIS = (
    "<?xml version='1.0' ?>\n<AFNI_attributes\n  ni_form=\"ni_group\" >\n"
    '<AFNI_atr\n  ni_type="int"\n  ni_dimen="8"\n  atr_name="TAXIS_NUMS" >\n'
    " 10\n 4\n 77002\n -999\n -999\n -999\n -999\n -999\n</AFNI_atr>\n"
    '<AFNI_atr\n  ni_type="float"\n  ni_dimen="8"\n  atr_name="TAXIS_FLOATS" >\n'
    " 0\n 2\n 0\n 0\n 1\n -999999\n -999999\n -999999\n</AFNI_atr>\n"
    '<AFNI_atr\n  ni_type="float"\n  ni_dimen="4"\n  atr_name="TAXIS_OFFSETS" >\n'
    " 0\n 1\n 0.5\n 1.5\n</AFNI_atr>\n</AFNI_attributes>\n\x00"
)


@pytest.mark.parametrize("with_afni_ext", [False, True])
def test_slice_time_corrected_header_round_trips_the_sample_time(tmp_path, with_afni_ext):
    import nibabel as nib

    from fastfuncstuff.io.afni import mark_slice_time_corrected, save_nifti
    from fastfuncstuff.io.dsetinfo import read_info, sample_time_offset

    img = nib.Nifti1Image(np.zeros((3, 3, 4, 10), np.float32), np.eye(4))
    hdr = img.header
    hdr.set_zooms((2, 2, 2, 2.0))
    hdr.set_dim_info(slice=2)
    hdr.set_xyzt_units("mm", "sec")
    hdr["slice_code"] = 3  # alt+z
    hdr["slice_duration"] = 0.5
    hdr.set_slice_times([0.0, 1.0, 0.5, 1.5])
    if with_afni_ext:
        hdr.extensions.append(nib.nifti1.Nifti1Extension(4, _AFNI_TAXIS.encode()))
    raw = tmp_path / "raw.nii"
    save_nifti(np.asarray(img.dataobj), raw, header=hdr)
    assert sample_time_offset(read_info(raw)) is None  # slices still differ

    mark_slice_time_corrected(hdr, 0.75)
    fixed = tmp_path / "st.nii"
    save_nifti(np.asarray(img.dataobj), fixed, header=hdr)
    info = read_info(fixed)
    assert info.time_offset == pytest.approx(0.75)
    assert sample_time_offset(info) == pytest.approx(0.75)


def test_round_onsets_snaps_to_sample_times_so_round_plus_tent_is_fir():
    from fastfuncstuff.design.builder import round_onsets

    onsets = [[np.array([3.1, 10.9, 20.4])]]
    rounded = round_onsets(onsets, 2.0, threshold=0.5, microtime_offset=1.0)[0][0]
    np.testing.assert_allclose(rounded, [3.0, 11.0, 21.0])  # on n*2 + 1
    tent = make_tent_design([rounded], 0.0, 8.0, 2.0, 20, device=CPU, microtime_offset=1.0)
    # Each event now sits on a sample: every row holds a single 1, i.e. FIR.
    assert set(tent[tent != 0].tolist()) == {1.0}
    assert round_onsets(onsets, 2.0, threshold=0.5)[0][0].tolist() == [4.0, 10.0, 20.0]


def test_resolve_microtime_offset_prefers_flag_then_header(tmp_path):
    import nibabel as nib

    from fastfuncstuff.cli_utils import resolve_microtime_offset
    from fastfuncstuff.io.afni import mark_slice_time_corrected, save_nifti

    def run(name, tzero):
        img = nib.Nifti1Image(np.zeros((2, 2, 2, 5), np.float32), np.eye(4))
        img.header.set_xyzt_units("mm", "sec")
        if tzero is not None:
            mark_slice_time_corrected(img.header, tzero)
        path = tmp_path / name
        save_nifti(np.asarray(img.dataobj), path, header=img.header)
        return str(path)

    a, b, plain = run("a.nii", 0.5), run("b.nii", 0.5), run("c.nii", None)
    assert resolve_microtime_offset(None, [a, b], 2.0, verbose=False) == 0.5
    assert resolve_microtime_offset(None, [plain], 2.0, verbose=False) == 0.0
    assert resolve_microtime_offset(0.0, [a], 2.0, verbose=False) == 0.0
    with pytest.raises(ValueError, match="different sample times"):
        resolve_microtime_offset(None, [a, plain], 2.0, verbose=False)
