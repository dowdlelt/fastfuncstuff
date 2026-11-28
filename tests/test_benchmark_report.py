"""benchmark.report: figures and page rebuilt from the worklist plus ingested host sets."""

from __future__ import annotations

import time

import pytest

from fastfuncstuff.benchmark.report import (
    arch_label,
    build_section,
    load_check,
    speedups,
)
from fastfuncstuff.benchmark.report_spec import Metric, ReportSpec
from fastfuncstuff.benchmark.worklist import Worklist

pytest.importorskip("seaborn")

SPEC = ReportSpec(
    summary="toy",
    tools={"ref": "reftool", "ffs": "ffstool"},
    timing={"ref": "reftool", "ffs_a": "ffs a", "ffs_b": "ffs b"},
    references=("ref",),
    metrics=(
        Metric("truth.err_{tool}", "error", line=1.0),
        Metric("reference.dist", "distance", log=True),
        Metric("truth.ok_{tool}", "ok", better="higher", kind="rate"),
    ),
    speedup=(("ffs_a+ffs_b", "ref"),),
    difficulty=("data.size", "size"),
)


def _run(wl, uid, sha, err_ffs, device, cpu, ffs_s=(2.0, 3.0)):
    wl.record_metrics(uid, "truth", {"err_ref": 0.5, "err_ffs": err_ffs, "ok_ref": 1, "ok_ffs": 1})
    wl.record_metrics(uid, "reference", {"dist": 0.3 + uid})
    wl.record_metrics(uid, "data", {"size": 10.0 * uid})
    wl.record_timing(uid, "ref", "cpu", 50.0, 6)
    wl.record_timing(uid, "ffs_a", device, ffs_s[0], None)
    wl.record_timing(uid, "ffs_b", device, ffs_s[1], None)
    wl.record_provenance(uid, "host", {"cpu_model": cpu, "gpu": "NVIDIA GeForce RTX 9999"})
    wl.record_provenance(uid, "ffs", {"ffs_sha": sha})
    wl.finish(uid, "done")


def test_report_latest_run_per_unit_and_chain_speedup(tmp_path):
    root = tmp_path / "bench"
    wl = Worklist(root / "worklist.sqlite")
    wl.add_units("toy", "dsA", [({"sub": s}, ["x"]) for s in ("01", "02", "03")])
    ids = [wl.claim_next()["id"] for _ in range(3)]
    for uid in ids:
        _run(wl, uid, "oldsha00", 5.0, "cuda", "Intel(R) Xeon(R) W-2133 CPU @ 3.60GHz")
    # unit 1 again on another host, later, on newer code
    hs_dir = tmp_path / "hs"
    hs = Worklist(hs_dir / "worklist.sqlite")
    hs.db.execute(
        "INSERT INTO units (id, check_name, dataset, unit_key, inputs, added_at)"
        " SELECT id, check_name, dataset, unit_key, inputs, added_at FROM units WHERE 0"
    )
    hs.db.execute(
        "INSERT INTO units (id, check_name, dataset, unit_key, inputs, added_at)"
        " VALUES (?, 'toy', 'dsA', '{\"sub\": \"01\"}', '[\"x\"]', ?)",
        (ids[0], time.time()),
    )
    hs.db.commit()
    hs.claim_next()
    time.sleep(0.01)
    _run(hs, ids[0], "newsha00", 0.2, "cpu", "AMD EPYC 9555 64-Core Processor", (10.0, 15.0))
    hs.close()
    wl.db.execute(
        "INSERT INTO hostsets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("abc", "epyc", str(hs_dir), "srv", "AMD EPYC 9555", 1, 0, time.time()),
    )
    wl.db.commit()
    wl.close()

    allruns, df, tim = load_check(root, "toy")
    assert len(allruns) == 4 and len(df) == 3
    one = df.set_index("unit_id").loc[ids[0]]
    assert one["truth.err_ffs"] == 0.2 and one["host"] == "epyc"  # the latest run counts
    assert one["stale"] == False and df["stale"].sum() == 2  # noqa: E712

    sp = speedups(tim, SPEC)
    # a chain is timed in full: 50 / (2 + 3) on the GPU host, 50 / (10 + 15) on the CPU one
    assert sorted(sp.speedup.round(3)) == [2.0, 10.0, 10.0, 10.0]
    assert set(sp.ffs_arch) == {"RTX 9999 (CUDA)", "EPYC 9555 (CPU)"}

    out = tmp_path / "report"
    out.mkdir()
    import matplotlib

    matplotlib.use("Agg")
    s = build_section(root, "toy", SPEC, out)
    assert s is not None and s["mixed"]
    for stem in [f["stem"] for f in s["acc"] + s["tim"]]:
        for ext in ("svg", "pdf", "csv"):
            assert (out / f"{stem}.{ext}").stat().st_size > 0
    assert len(s["acc"]) == 4  # three metrics + difficulty
    err = s["acc"][0]
    assert err["nfail"] == 2 and "the 2 units" in err["caption"]  # two units past the line
    assert "agree" in s and "truth.err_ffs" in s["agree"]  # unit 1 ran on two hosts


def test_arch_labels_are_short():
    assert arch_label("cpu", "Intel(R) Xeon(R) W-2133 CPU @ 3.60GHz", None) == "Xeon W-2133 (CPU)"
    assert arch_label("cpu", "AMD EPYC 9555 64-Core Processor", None) == "EPYC 9555 (CPU)"
    assert arch_label("cuda", None, "NVIDIA GeForce RTX 5070 Ti") == "RTX 5070 Ti (CUDA)"


def test_estimated_speedup_from_calibration_units():
    """Chronos ran the reference on 3 units (twice the server's time), ffs on all 5: units
    4 and 5 get the server's time x 2, flagged as estimated; 1-3 are not re-estimated."""
    import pandas as pd

    from fastfuncstuff.benchmark.report import estimated_speedups

    rows = []
    for uid in range(1, 6):
        rows.append(("epyc", uid, "ref", "cpu", 100.0, "EPYC", "EPYC 9555 (CPU)"))
        rows.append(("local", uid, "ffs_a", "cuda", 4.0, "Xeon", "RTX 9999 (CUDA)"))
        rows.append(("local", uid, "ffs_b", "cuda", 6.0, "Xeon", "RTX 9999 (CUDA)"))
        if uid <= 3:
            rows.append(("local", uid, "ref", "cpu", 200.0, "Xeon", "W-2133 (CPU)"))
    tim = pd.DataFrame(rows, columns=["host", "unit_id", "side", "device", "seconds",
                                      "host.cpu_model", "arch"]).assign(dataset="dsA")  # fmt: skip
    est = estimated_speedups(tim, SPEC)
    assert sorted(est.unit_id) == [4, 5]
    assert (est.speedup == 20.0).all() and (est.calib_n == 3).all()
    assert set(est.ref_arch) == {"W-2133 (CPU)"} and set(est.calib_from) == {"EPYC 9555 (CPU)"}
    # with two calibration units the ratio is not trusted
    assert estimated_speedups(
        tim[~((tim.unit_id == 3) & (tim.side == "ref") & (tim.host == "local"))], SPEC
    ).empty
