"""Tests for the benchmark robustness fixes (P0): missing-input handling.

The harness should turn a missing upstream artifact into a clean INCOMPLETE
result with an actionable message, not a raw I/O error after the expensive run.
"""

from __future__ import annotations

import types

import nibabel as nib
import numpy as np

from fastfuncstuff.benchmark import validation
from fastfuncstuff.benchmark.runner import (
    BenchmarkContext,
    StageResult,
    _missing_validation_inputs,
    run_stages,
)


def _write_vol(path, shape=(6, 6, 6)):
    nib.Nifti1Image(np.random.rand(*shape).astype(np.float32), np.eye(4)).to_filename(str(path))


# --------------------------------------------------------------------------
# compare_* helpers: structured error instead of raising
# --------------------------------------------------------------------------


def test_compare_volumes_missing_returns_structured_error(tmp_path):
    a = tmp_path / "a.nii"
    _write_vol(a)
    res = validation.compare_volumes(a, tmp_path / "does_not_exist.nii")
    assert res["error"] == "missing_input"
    assert str(tmp_path / "does_not_exist.nii") in res["missing"]
    assert np.isnan(res["r"])  # metric present (nan), so callers don't KeyError


def test_compare_timeseries_missing_returns_structured_error(tmp_path):
    res = validation.compare_timeseries_4d(tmp_path / "nope_a.nii", tmp_path / "nope_b.nii")
    assert res["error"] == "missing_input"
    assert len(res["missing"]) == 2
    assert np.isnan(res["median_r"])


def test_compare_volumes_present_still_works(tmp_path):
    a, b = tmp_path / "a.nii", tmp_path / "b.nii"
    _write_vol(a)
    _write_vol(b)
    res = validation.compare_volumes(a, b)
    assert "error" not in res
    assert "r" in res


# --------------------------------------------------------------------------
# StageResult.status
# --------------------------------------------------------------------------


def test_stage_result_status():
    assert StageResult("x", passed=True).status == "PASS"
    assert StageResult("x", passed=False).status == "FAIL"
    assert StageResult("x", passed=False, incomplete=True).status == "INCOMPLETE"


# --------------------------------------------------------------------------
# runner preflight
# --------------------------------------------------------------------------


def _ctx(tmp_path) -> BenchmarkContext:
    return BenchmarkContext(data_dir=tmp_path, dataset_id="testds")


def test_missing_validation_inputs_helper(tmp_path):
    present = tmp_path / "present.nii"
    present.write_text("x")
    stage = types.SimpleNamespace(
        name="fake",
        validation_inputs=lambda ctx: [present, tmp_path / "absent.nii"],
    )
    missing = _missing_validation_inputs(stage, _ctx(tmp_path))
    assert missing == [str(tmp_path / "absent.nii")]


def test_missing_validation_inputs_no_hook(tmp_path):
    stage = types.SimpleNamespace(name="fake")  # no validation_inputs
    assert _missing_validation_inputs(stage, _ctx(tmp_path)) == []


def test_run_stages_marks_incomplete_not_fail(tmp_path, capsys):
    """A stage whose validation_inputs are missing is INCOMPLETE, not FAIL, and
    validate() is not the thing that crashes the run."""
    absent = tmp_path / "from_upstream.nii"
    stage = types.SimpleNamespace(
        name="fake",
        check_prerequisites=lambda ctx: [],
        validate=lambda ctx: {"passed": True, "summary": "compared ok"},
        validation_inputs=lambda ctx: [absent],
        requires=["upstream"],
    )
    results = run_stages([stage], _ctx(tmp_path))
    assert len(results) == 1
    r = results[0]
    assert r.incomplete is True
    assert r.passed is False
    assert r.status == "INCOMPLETE"
    assert str(absent) in r.errors
    out = capsys.readouterr().out
    assert "INCOMPLETE" in out
    assert "upstream" in out  # the requires-based hint named the producing stage


def test_run_stages_passes_when_inputs_present(tmp_path):
    present = tmp_path / "ok.nii"
    present.write_text("x")
    stage = types.SimpleNamespace(
        name="fake",
        check_prerequisites=lambda ctx: [],
        validate=lambda ctx: {"passed": True, "summary": "ok"},
        validation_inputs=lambda ctx: [present],
    )
    r = run_stages([stage], _ctx(tmp_path))[0]
    assert r.incomplete is False
    assert r.passed is True
    assert r.status == "PASS"
