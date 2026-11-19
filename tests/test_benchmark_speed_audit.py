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
    s.error = None
    # 2 s of startup at 0%, then 50%/100% alternating, then 1 s of tail at 0%.
    s.samples = [(t / 10, 0) for t in range(20)]
    s.samples += [(2 + t / 10, 100 if t % 2 else 50) for t in range(20)]
    s.samples += [(4 + t / 10, 0) for t in range(10)]
    got = s.summary(0.0, 5.0)
    assert got["first_kernel_s"] == pytest.approx(2.0)
    assert got["active_busy_pct"] == pytest.approx(75.0)
    assert got["mean_busy_pct"] < got["active_busy_pct"]


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
