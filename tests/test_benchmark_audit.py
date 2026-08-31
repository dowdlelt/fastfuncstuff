"""Tests for the benchmark robustness fixes (P0): missing-input handling.

The harness should turn a missing upstream artifact into a clean INCOMPLETE
result with an actionable message, not a raw I/O error after the expensive run.
"""

from __future__ import annotations

import types
from pathlib import Path

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


def test_benchmark_device_identity_honours_forced_cpu(tmp_path, monkeypatch):
    """Available MPS hardware must not label a forced-CPU timing as MPS."""
    from fastfuncstuff.benchmark.arch import get_ffs_arch_id

    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert get_ffs_arch_id("cpu") == "cpu"
    assert get_ffs_arch_id("cpu,8") == "cpu"
    assert get_ffs_arch_id("mps").startswith("mps-")
    assert BenchmarkContext(data_dir=tmp_path, device="cpu,8").ffs_tag == "_cpu"
    assert BenchmarkContext(data_dir=tmp_path, device="mps").ffs_tag == "_mps"

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert get_ffs_arch_id("auto") == "cpu"
    assert BenchmarkContext(data_dir=tmp_path, device="auto").ffs_tag == "_cpu"

    from fastfuncstuff.benchmark.timing_cache import append_run, load_cache

    append_run(tmp_path, {"fake": {"ffs_seconds": 1.0}}, device_spec="cpu,8")
    cached = load_cache(tmp_path)["runs"][-1]
    assert cached["ffs_arch_id"] == "cpu"
    assert cached["hardware"]["requested_device"] == "cpu,8"


def test_glmsingle_matlab_helper_is_packaged(tmp_path):
    from fastfuncstuff.benchmark.stages.glmsingle_matlab import _matlab_script

    script = _matlab_script(_ctx(tmp_path))
    assert script.exists()
    assert script.parent.name == "assets"


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


def test_run_stages_ref_crash_fails_over_stale_validation(tmp_path, capsys):
    """Same for the reference side.

    Only the FFS crash hard-overrode the verdict; a reference tool that raised
    was appended to errors and then validation compared two stale files and
    reported PASS -- which was also eligible to be cached.
    """
    present = tmp_path / "stale.nii"
    present.write_text("x")

    def run_ref(ctx):
        raise RuntimeError("reference crashed")

    stage = types.SimpleNamespace(
        name="fake",
        check_prerequisites=lambda ctx: [],
        run_ref=run_ref,
        run_ffs=lambda ctx: 1.0,
        validate=lambda ctx: {"passed": True, "summary": "r=0.99"},
        validation_inputs=lambda ctx: [present],
    )
    r = run_stages([stage], _ctx(tmp_path))[0]
    assert r.ref_crashed is True
    assert r.passed is False
    assert r.status == "FAIL"
    assert "reference crashed" in r.summary
    assert "stale" in r.summary
    assert "FAIL" in capsys.readouterr().out


def test_glmsingle_ridge_validator_fails_when_a_metric_is_not_computed(tmp_path, monkeypatch):
    """A stage whose outputs are absent must not validate as PASS.

    `passed` started at True and each threshold was skipped for a NaN, so an
    ffs_ridge that exited 0 without writing either output validated against
    nothing at all -- with valid MATLAB files on the other side.
    """
    from fastfuncstuff.benchmark import validation as bench_validation
    from fastfuncstuff.benchmark.stages import glmsingle_ridge

    gs = tmp_path / "glmsingle"
    gs.mkdir()
    ffs = tmp_path / "ffs_ridge"
    ffs.mkdir()

    # Reference side present and readable; FFS side absent entirely.
    monkeypatch.setattr(
        bench_validation,
        "_load_dataobj",
        lambda path, *a, **k: np.ones((4, 4, 4, 3), dtype=np.float32),
    )
    ctx = types.SimpleNamespace(glmsingle_dir=gs, ffs_ridge_dir=ffs)

    results = glmsingle_ridge.validate(ctx)
    assert np.isnan(results["fracvalue_corr"])
    assert np.isnan(results["beta_timeseries_corr"])
    assert results["passed"] is False
    assert "not computed" in results["summary"]


def test_glmsingle_ridge_declares_what_validate_reads():
    """Missing outputs should read INCOMPLETE with the file named."""
    from fastfuncstuff.benchmark.stages import glmsingle_ridge

    ctx = types.SimpleNamespace(glmsingle_dir=Path("/gs"), ffs_ridge_dir=Path("/ffs"))
    names = {p.name for p in glmsingle_ridge.validation_inputs(ctx)}
    assert "ridge_single_trial_betas.nii.gz" in names
    assert "glmsingle_betas_D.nii.gz" in names


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
