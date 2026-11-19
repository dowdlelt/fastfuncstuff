"""benchmark/audit.py: the op census, sync attribution, and stage aggregation."""

from __future__ import annotations

import json
import traceback

import pytest
import torch

from fastfuncstuff.benchmark import audit


def _work() -> None:
    x = torch.randn(64, 32, device=torch.device("cpu"))
    for _ in range(3):
        torch.linalg.qr(x.double())
    (x.t() @ x).sum()


def test_ops_census_counts_real_ops_and_skips_views(tmp_path):
    out = tmp_path / "s" / "invocations" / "001-x.json"
    audit.capture_audit(_work, out, mode="ops", command=["ffs_x"], label="x", stage="s")
    rows = {(r["op"], r["dtype"], r["shapes"]): r for r in json.loads(out.read_text())["ops"]}
    qr = rows[("aten.linalg_qr.default", "float64", "64x32")]
    assert qr["calls"] == 3
    assert qr["cpu_seconds"] > 0  # CPU ops are timed
    # t() is a view: counting it would bury the real work under metadata ops.
    assert not any(op == "aten.t.default" for op, _, _ in rows)


def test_aggregate_and_report_merge_invocations(tmp_path):
    stage_dir = tmp_path / "s"
    for i in (1, 2):
        out = stage_dir / "invocations" / f"00{i}-x.json"
        audit.capture_audit(_work, out, mode="ops", command=["ffs_x"], label="x", stage="s")
    summary = audit.aggregate_stage(stage_dir, "s")
    qr = next(r for r in summary["ops"] if r["op"] == "aten.linalg_qr.default")
    assert qr["calls"] == 6
    report = audit.format_report({"s": summary}, "ops")
    assert "aten.linalg_qr.default" in report


def _inner_ffs_frame():
    return traceback.extract_stack()


def test_call_site_prefers_innermost_package_frame():
    stack = _inner_ffs_frame()
    # The test file is outside the package, so the fallback names it.
    assert "test_benchmark_speed_audit.py" in audit._call_site(stack)
    fake = traceback.StackSummary.from_list(
        [
            (audit._PKG_DIRS[0] + "glm/core.py", 10, "fit", None),
            (audit._PKG_DIRS[0] + "benchmark/audit.py", 5, "show", None),
            ("/x/lib/python3.14/warnings.py", 1, "warn", None),
        ]
    )
    # Frames in audit.py itself (the hook) and the stdlib are never the culprit.
    assert audit._call_site(fake).startswith("fastfuncstuff/glm/core.py:10")


def test_busy_summary_window_excludes_startup_and_tail():
    s = audit.BusySampler.__new__(audit.BusySampler)
    s.error, s.period = None, 0.1
    # 2 s of startup at 0%, then 50%/100% alternating, then 1 s of tail at 0%.
    s.samples = [(t / 10, 0, "load", "gzip.py:read") for t in range(20)]
    s.samples += [
        (2 + t / 10, 100 if t % 2 else 50, "fit" if t % 2 else "loop", "x.py:f") for t in range(20)
    ]
    s.samples += [(4 + t / 10, 0, "save", "gzip.py:write") for t in range(10)]
    got = s.summary(0.0, 5.0)
    assert got["first_kernel_s"] == pytest.approx(2.0)
    assert got["active_busy_pct"] == pytest.approx(75.0)
    assert got["mean_busy_pct"] < got["active_busy_pct"]
    # Idle inside the window is charged to where the host was: the 50% samples,
    # 10 of them x 0.1 s x half idle. Startup and tail are outside the window.
    sites = dict(got["idle_sites"])
    assert sites == {"loop": pytest.approx(0.5)}


def test_frame_sites_names_ffs_frame_and_leaf():
    import sys

    site, leaf = audit._frame_sites(sys._getframe())
    assert leaf.endswith(":test_frame_sites_names_ffs_frame_and_leaf")
    # tests/ is not the package, so the fallback names the caller's own frame.
    assert "test_benchmark_speed_audit.py" in site


def test_nvml_index_follows_device_spec_and_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert audit._nvml_index(None) == 0
    assert audit._nvml_index("cuda,1") == 1
    assert audit._nvml_index("cuda:1") == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,2")
    assert audit._nvml_index("cuda,1") == 2


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_idle_mode_counts_hidden_syncs_by_site(tmp_path):
    def work():
        x = torch.randn(256, 256, device="cuda")
        for _ in range(7):
            float(x.sum())

    out = tmp_path / "idle.json"
    audit.capture_audit(work, out, mode="idle", command=["ffs_x"], label="x", stage="s")
    syncs = json.loads(out.read_text())["syncs"]
    assert syncs["implicit_syncs"] >= 7
    site, _ = syncs["implicit_sites"][0]
    assert "test_benchmark_speed_audit.py" in site
    # The hook must come off again: later code in the process is not audited.
    assert torch.cuda.get_sync_debug_mode() == 0


def _replay_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "replay_op_census.py"
    spec = importlib.util.spec_from_file_location("replay_op_census", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_census_entry_rebuilds_and_runs(tmp_path):
    def work():
        x = torch.randn(40, 12, device=torch.device("cpu"))
        x.t().contiguous()  # memory_format kwarg
        torch.linalg.cholesky(x.t() @ x + 12 * torch.eye(12))  # needs SPD input
        x.unsqueeze(0).expand(3, -1, -1).sum(0)  # stride-0 operand
        x[x > 0]  # boolean index
        x.index_select(0, torch.arange(0, 40, 3))  # integer index
        x.to(torch.float64)  # dtype literal

    audit.capture_audit(
        work,
        tmp_path / "s" / "invocations" / "001-x.json",
        mode="ops",
        command=["x"],
        label="x",
        stage="s",
    )
    replay = _replay_module()
    rows, threads = replay.load_census(tmp_path, "cpu")
    assert threads is not None and len(rows) >= 6
    for row in rows:
        fn, _ = replay.build_call(row, torch.device("cpu"))
        fn()  # raises if any template does not round-trip


def test_env_hook_only_arms_for_ffs_clis(monkeypatch, tmp_path):
    from fastfuncstuff import _audit_hook

    armed = []
    monkeypatch.setattr(_audit_hook, "_install", lambda mode, tool: armed.append((mode, tool)))
    monkeypatch.setenv("FFS_AUDIT_DIR", str(tmp_path))

    monkeypatch.delenv("FFS_AUDIT", raising=False)
    monkeypatch.setattr("sys.argv", ["/usr/bin/ffs_moco", "-input", "x"])
    _audit_hook.maybe_install()
    monkeypatch.setenv("FFS_AUDIT", "idle")
    monkeypatch.setattr("sys.argv", ["/usr/bin/python", "-c", "x"])  # a notebook, a pool worker
    _audit_hook.maybe_install()
    assert armed == []

    monkeypatch.setattr("sys.argv", ["/usr/bin/ffs_moco", "-input", "x"])
    _audit_hook.maybe_install()
    assert armed == [("idle", "ffs_moco")]


def test_env_hook_never_breaks_the_tool(monkeypatch, capsys):
    from fastfuncstuff import _audit_hook

    monkeypatch.setenv("FFS_AUDIT", "bogus")
    monkeypatch.setattr("sys.argv", ["/usr/bin/ffs_moco"])
    _audit_hook.maybe_install()  # must not raise
    assert "not recording" in capsys.readouterr().err
