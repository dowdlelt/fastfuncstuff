"""Tests for opt-in benchmark CPU/CUDA profiling."""

from __future__ import annotations

import json
import types

import pytest

from fastfuncstuff.benchmark.profiling import (
    BenchmarkProfiler,
    StageProfiler,
    _is_io_row,
    capture_profile,
    command_argv,
)
from fastfuncstuff.benchmark.runner import BenchmarkContext, run_stages
from fastfuncstuff.cli.benchmark import parse_args
from fastfuncstuff.cli.profile import parse_args as parse_profile_args


def test_command_argv_accepts_only_direct_ffs_commands():
    assert command_argv('ffs_nwarp -input "run one.nii" -device cpu') == [
        "ffs_nwarp",
        "-input",
        "run one.nii",
        "-device",
        "cpu",
    ]
    assert command_argv(["/opt/bin/ffs_reml", "-input", "x.nii"]) == [
        "/opt/bin/ffs_reml",
        "-input",
        "x.nii",
    ]
    assert command_argv("3dvolreg -prefix out.nii in.nii") is None
    assert command_argv("ffs_info | tee report.txt") is None
    assert command_argv("ffs_benchmark -stages moco") is None


def test_stage_profiler_wraps_command_with_original_argv(tmp_path):
    profiler = StageProfiler(stage="warp", stage_dir=tmp_path)
    wrapped = profiler.wrap_command("ffs_nwarp -input x.nii -device cpu", "warp run 1")

    assert wrapped is not None
    assert wrapped[:3] == [
        pytest.importorskip("sys").executable,
        "-m",
        "fastfuncstuff.benchmark.profile_cli",
    ]
    assert wrapped[-6:] == ["--", "ffs_nwarp", "-input", "x.nii", "-device", "cpu"]
    assert wrapped[4].endswith("/invocations/001-warp-run-1.json")


def test_capture_profile_writes_compact_cpu_report(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "invocation.json"

    def work():
        return int((torch.ones(4) + 2).sum().item())

    assert (
        capture_profile(
            work,
            output,
            command=["ffs_test", "-device", "cpu"],
            label="tiny cpu work",
            stage="tiny",
        )
        == 12
    )
    payload = json.loads(output.read_text())
    assert payload["stage"] == "tiny"
    assert payload["wall_seconds"] >= 0
    assert payload["hardware"]["requested_device"] == "cpu"
    assert payload["hardware"]["cuda_available"] is False
    assert any(row["operator"].startswith("aten::") for row in payload["torch"])
    assert payload["python"]


def test_capture_profile_bounds_torch_events_but_profiles_full_python(tmp_path, monkeypatch):
    import time

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "bounded.json"

    def work():
        value = torch.ones(4)
        for _ in range(40):
            value.add_(1)
            time.sleep(0.005)
        return value

    capture_profile(
        work,
        output,
        command=["ffs_test", "-device", "cpu"],
        label="bounded cpu work",
        stage="tiny",
        torch_profile_seconds=0.03,
    )
    payload = json.loads(output.read_text())
    add_event = next(row for row in payload["torch"] if row["operator"] == "aten::add_")

    assert payload["tool_seconds"] >= 0.15
    assert payload["torch_collection_limited"] is True
    assert payload["torch_collection_seconds"] < payload["tool_seconds"]
    assert add_event["calls"] < 40
    assert payload["profiler_stop_seconds"] >= 0
    assert payload["report_build_seconds"] >= 0
    assert all(row["function"] != "stop" for row in payload["python"])


def test_io_classifier_avoids_threading_and_import_false_positives():
    assert not _is_io_row({"file": "/usr/lib/threading.py", "function": "wait"})
    assert not _is_io_row({"file": "<frozen importlib._bootstrap>", "function": "_find_and_load"})
    assert _is_io_row({"file": "/project/fastfuncstuff/io/images.py", "function": "load_image"})
    assert _is_io_row({"file": "/project/warp.py", "function": "save_warp_field"})


def test_benchmark_profiler_aggregates_invocations(tmp_path):
    benchmark = BenchmarkProfiler.create(tmp_path)
    assert _is_io_row({"file": "/numpy/io.py", "function": "loadtxt"})
    stage = benchmark.start_stage("glm")
    invocations = stage.stage_dir / "invocations"
    payload = {
        "label": "one",
        "command": ["ffs_reml"],
        "tool_seconds": 2.0,
        "profiler_stop_seconds": 0.2,
        "report_build_seconds": 0.3,
        "torch_collection_seconds": 1.0,
        "torch_collection_limited": True,
        "wall_seconds": 2.5,
        "error": None,
        "python": [
            {
                "file": "io.py",
                "line": 10,
                "function": "load_image",
                "calls": 2,
                "primitive_calls": 2,
                "self_cpu_seconds": 0.5,
                "cumulative_cpu_seconds": 1.5,
            }
        ],
        "torch": [
            {
                "operator": "aten::mm",
                "calls": 3,
                "self_cpu_seconds": 0.1,
                "cpu_seconds": 0.2,
                "self_device_seconds": 1.0,
                "device_seconds": 1.1,
            }
        ],
    }
    (invocations / "001-one.json").write_text(json.dumps(payload))

    payload["python_io"] = payload["python"]
    benchmark.finish_stage(stage)
    manifest = benchmark.finish()

    assert manifest["profiled_invocations"] == 1
    summary = json.loads((stage.stage_dir / "summary.json").read_text())
    assert summary["wall_seconds"] == 2.5
    assert summary["python"][0]["function"] == "load_image"
    assert summary["tool_seconds"] == 2.0
    assert summary["profiler_stop_seconds"] == 0.2
    assert summary["report_build_seconds"] == 0.3
    assert summary["io_python"][0]["function"] == "load_image"
    assert summary["torch"][0]["operator"] == "aten::mm"
    assert "Top sampled PyTorch operators" in (stage.stage_dir / "summary.txt").read_text()


def test_profiled_ffs_timing_is_not_cached(tmp_path, monkeypatch):
    from fastfuncstuff.benchmark import timing_cache

    recorded = {}

    def fake_append(data_dir, stage_timings, **kwargs):
        recorded["stage_timings"] = stage_timings

    monkeypatch.setattr(timing_cache, "append_run", fake_append)
    stage = types.SimpleNamespace(
        name="fake",
        check_prerequisites=lambda ctx: [],
        run_ffs=lambda ctx: 12.0,
        validate=lambda ctx: {"passed": True, "summary": "ok"},
    )
    ctx = BenchmarkContext(data_dir=tmp_path, dataset_id="test", profile=True)

    result = run_stages([stage], ctx)[0]

    assert result.ffs_time == 12.0
    assert recorded["stage_timings"] == {}
    manifest_path = next((tmp_path / "profiles").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["profiled_invocations"] == 0


def test_profile_cli_flags():
    args = parse_args(["-profile"])
    assert args.profile is True
    assert args.profile_trace is False
    with pytest.raises(SystemExit):
        parse_args(["-profile-trace"])


def test_standalone_profile_parser_accepts_command_after_separator():
    args = parse_profile_args(["--", "ffs_reml", "-device", "cuda"])
    assert args.command == ["ffs_reml", "-device", "cuda"]
    assert args.trace is False


def test_aggregate_stage_accepts_top_level_invocation(tmp_path):
    from fastfuncstuff.benchmark.profiling import aggregate_stage

    stage = tmp_path / "profile"
    stage.mkdir()
    (stage / "invocation.json").write_text(
        json.dumps({"label": "ffs_reml", "python": [], "torch": [], "wall_seconds": 1.0})
    )
    summary = aggregate_stage(stage)
    assert summary["invocation_count"] == 1
