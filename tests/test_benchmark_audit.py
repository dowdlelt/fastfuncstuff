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
    # A run-phase crash is FAIL even if validation (on stale output) "passed".
    assert StageResult("x", passed=True, ffs_crashed=True).status == "FAIL"
    assert StageResult("x", passed=True, incomplete=True, ffs_crashed=True).status == "FAIL"


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


def test_run_stages_crash_fails_over_stale_validation(tmp_path, capsys):
    """If the FFS tool crashes but validation still 'passes' against stale output
    from an earlier run, the stage must report FAIL -- not a misleading PASS."""
    present = tmp_path / "stale.nii"
    present.write_text("x")

    def run_ffs(ctx):
        raise RuntimeError("Command failed (fake tool): exit code 1")

    stage = types.SimpleNamespace(
        name="fake",
        check_prerequisites=lambda ctx: [],
        run_ffs=run_ffs,
        # validate() succeeds against the leftover output -> would have said PASS
        validate=lambda ctx: {"passed": True, "summary": "r=0.99"},
        validation_inputs=lambda ctx: [present],
    )
    r = run_stages([stage], _ctx(tmp_path))[0]
    assert r.ffs_crashed is True
    assert r.passed is False
    assert r.status == "FAIL"
    assert "exit code 1" in r.summary
    assert "stale" in r.summary  # names that the r=0.99 was on stale output
    assert "FAIL" in capsys.readouterr().out


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


# --------------------------------------------------------------------------
# P1: dependency expansion + timing integrity
# --------------------------------------------------------------------------


def test_expand_with_deps_pulls_in_upstream():
    from fastfuncstuff.benchmark.stages import expand_with_deps

    expanded = expand_with_deps(["warp"])
    # warp requires moco, crossalign, align -> all present, in pipeline order
    for s in ("moco", "crossalign", "align", "warp"):
        assert s in expanded
    # ordered by ALL_STAGES: moco before warp
    assert expanded.index("moco") < expanded.index("warp")


def test_unsatisfied_deps_reports_missing():
    from fastfuncstuff.benchmark.stages import unsatisfied_deps

    deps = unsatisfied_deps(["warp"])  # warp alone -> all its requires unsatisfied
    assert set(deps["warp"]) == {"moco", "crossalign", "align"}
    # when the deps are included, nothing is unsatisfied
    assert unsatisfied_deps(["moco", "crossalign", "align", "warp"]) == {}


def test_partial_timing_flagged_and_excluded(tmp_path):
    """A partial run flags its timing and that timing is kept out of cached refs."""
    from fastfuncstuff.benchmark.timing_cache import (
        append_run,
        get_ref_timings_all_archs,
    )

    # full ref run -> baseline available
    append_run(tmp_path, {"warp": {"ref_seconds": 800.0}}, dataset_id="ds")
    assert get_ref_timings_all_archs(tmp_path, "warp")  # has a baseline

    # later partial ref run must not become the baseline
    append_run(
        tmp_path,
        {"warp": {"ref_seconds": 80.0, "ref_partial": True, "ref_ran": 1, "ref_total": 10}},
        dataset_id="ds",
    )
    refs = get_ref_timings_all_archs(tmp_path, "warp")
    secs = [s for _, s in refs]
    assert 800.0 in secs
    assert 80.0 not in secs  # partial excluded


def test_note_items_marks_partial_result(tmp_path):
    """A stage reporting ran<total via note_items surfaces as partial timing."""

    def run_ffs(ctx):
        ctx.note_items("ffs", 1, 10)
        return 42.0

    stage = types.SimpleNamespace(
        name="fake",
        check_prerequisites=lambda ctx: [],
        run_ffs=run_ffs,
        validate=lambda ctx: {"passed": True, "summary": "ok"},
    )
    r = run_stages([stage], _ctx(tmp_path))[0]
    assert r.partial.get("ffs") == {"ran": 1, "total": 10}
