"""QC folder: record naming, orientation, carpet rows, checks, report and table."""

from __future__ import annotations

import json
import types

import numpy as np
import pytest
import torch

from fastfuncstuff.qc import checks, plots, render
from fastfuncstuff.qc.record import QCRecord, load_records, metric_flagged, natural_key
from fastfuncstuff.qc.report import build_report
from fastfuncstuff.qc.table import build_table, parse_flag, robust_z, subject_name


def _rec(tmp_path, label, tool="moco"):
    return QCRecord.for_args(types.SimpleNamespace(qc=str(tmp_path), qc_label=label), tool, "x")


def test_no_qc_flag_means_no_record(tmp_path):
    assert QCRecord.for_args(types.SimpleNamespace(qc=None), "moco", "a.nii.gz") is None


def test_label_falls_back_to_input_name(tmp_path):
    rec = QCRecord.for_args(
        types.SimpleNamespace(qc=str(tmp_path)), "moco", "/d/sub-01_bold.nii.gz"
    )
    assert rec.label == "sub-01_bold"


def test_figure_names_sort_check_first_then_run(tmp_path):
    """Scrolling the folder must flip between runs on one check, and a later tool
    must sort after an earlier one whatever the labels."""
    names = []
    for label in ("run-10", "run-2"):
        rec = _rec(tmp_path, label)
        names += [rec.figure_path(2, "tsnr").name, rec.figure_path(1, "motion").name]
    names.append(_rec(tmp_path, "run-1", "allineate").figure_path(1, "edges").name)
    ordered = sorted(names, key=natural_key)
    assert ordered == [
        "20.moco.01-motion.run-2.png",
        "20.moco.01-motion.run-10.png",
        "20.moco.02-tsnr.run-2.png",
        "20.moco.02-tsnr.run-10.png",
        "50.allineate.01-edges.run-1.png",
    ]


def test_metric_limits_flag_and_roundtrip(tmp_path):
    rec = _rec(tmp_path, "run-1")
    rec.metric("censor_frac", 0.2, warn_gt=0.15)
    rec.metric("tsnr", np.float32(50.0), warn_lt=10)
    rec.metric("n", np.int64(3))
    rec.metric("bad", float("nan"))
    assert rec.flagged() == ["censor_frac"]
    rec.write(verb=0)
    (loaded,) = load_records(tmp_path)
    assert loaded["metrics"]["n"]["value"] == 3 and isinstance(loaded["metrics"]["n"]["value"], int)
    assert loaded["metrics"]["bad"]["value"] is None  # NaN is not valid JSON
    assert metric_flagged(loaded["metrics"]["censor_frac"])
    assert not metric_flagged(loaded["metrics"]["tsnr"])


def test_montage_orientation_follows_the_affine():
    """A blob at the subject's superior end must be at the top of the sagittal
    panel whether the file is stored superior-first or inferior-first."""
    nz, ny, nx = 20, 16, 12
    vol = np.zeros((nz, ny, nx), np.float32)
    vol[2:6, 6:10, 4:8] = 1.0  # low z index
    up = np.diag([2.0, 2.0, 3.0, 1.0])  # z index increases toward superior
    down = np.diag([2.0, 2.0, -3.0, 1.0])  # ... toward inferior
    tops = []
    for aff in (up, down):
        ras, zooms = render.ras_volume(vol, aff)
        grid = render.Grid(ras, zooms, None, n_cut=1)
        grid.cuts["sagittal"] = np.array([5])
        (sl,) = grid.slices(ras, "sagittal")
        rows = np.flatnonzero(sl.any(axis=1))
        tops.append(rows.mean() < sl.shape[0] / 2)
        assert grid.aspect("sagittal") == (3.0, 2.0)  # mm per row (S), per column (A)
    assert tops == [False, True]


def test_panel_keeps_physical_aspect():
    rgb = np.zeros((10, 40, 3), np.uint8)
    img = render.panel(rgb, (3.0, 1.0), 120)  # 30 mm tall, 40 mm wide
    assert img.size == (160, 120)


def test_carpet_rows_blocks_and_scale():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 5000)).astype(np.float32) + np.linspace(0, 10, 50)[:, None]
    rows, blocks = plots.carpet_rows(
        x, [("a", np.arange(1000)), ("b", np.arange(1000, 5000))], max_rows=100
    )
    assert rows.shape == (100, 50)
    assert blocks == [("a", 0, 20), ("b", 20, 100)]
    # Detrended: the shared ramp is gone, every row is z-scored.
    np.testing.assert_allclose(rows.mean(axis=1), 0, atol=1e-4)
    np.testing.assert_allclose(rows.std(axis=1), 1, atol=1e-3)


def _brain_series(nt=40, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[:16, :24, :24]
    brain = (((x - 12) / 9) ** 2 + ((y - 12) / 10) ** 2 + ((z - 8) / 6) ** 2) < 1
    base = np.where(brain, 1000.0, 2.0).astype(np.float32)
    ser = base[None] * (1 + noise * rng.standard_normal((nt, *base.shape)).astype(np.float32))
    return torch.from_numpy(ser)


def test_epi_check_figures_and_tsnr(tmp_path):
    rec = _rec(tmp_path, "run-1")
    ser = _brain_series()
    checks.epi_check(rec, 2, ser, np.diag([-3.0, -3.0, 3.0, 1.0]), tr=2.0, device="cpu")
    m = {k: v.value for k, v in rec.metrics.items()}
    # 1% white noise -> tSNR ~100 (detrending costs 2 dof of 40).
    assert 85 < m["tsnr_median"] < 115
    assert abs(m["gcor"]) < 0.05  # independent noise: no global correlation
    assert {f["check"] for f in rec.figures} == {"tsnr", "carpet"}
    for f in rec.figures:
        assert (tmp_path / f["file"]).stat().st_size > 10_000


def test_motion_check_censor_metrics(tmp_path):
    rec = _rec(tmp_path, "run-1")
    params = np.zeros((30, 6))
    params[20:, 3] = 2.0
    keep = np.ones(30, int)
    keep[20:26] = 0
    enorm = np.r_[0, np.linalg.norm(np.diff(params, axis=0), axis=1)]
    checks.motion_check(rec, 1, params, tr=1.0, enorm=enorm, keep=keep, motion_limit=0.3)
    assert rec.metrics["censor_frac"].value == pytest.approx(0.2)
    assert rec.metrics["trans_range"].value == pytest.approx(2.0)
    assert rec.flagged() == ["censor_frac"]
    assert rec.figure_path(1, "motion").exists()


def _fake_subject(root, name, tsnr, censor):
    d = root / name / "qc"
    for run in ("run-1", "run-2"):
        rec = QCRecord.for_args(types.SimpleNamespace(qc=str(d), qc_label=run), "moco", "x")
        rec.metric("tsnr_median", tsnr)
        rec.metric("censor_frac", censor, warn_gt=0.15)
        path = rec.figure_path(1, "motion")
        render.save_png(render.to_image(np.zeros((8, 8, 3), np.uint8)), path)
        rec.add_figure(1, "motion", "Motion parameters", path)
        rec.write(verb=0)
    return d


def test_report_tables_and_page(tmp_path):
    d = _fake_subject(tmp_path, "sub-01", 60.0, 0.3)
    out = build_report(d)
    tsv = out["tsv"].read_text().splitlines()
    assert tsv[0].split("\t") == ["label", "moco.tsnr_median", "moco.censor_frac"]
    assert len(tsv) == 3
    assert "** WARN" in out["txt"].read_text()
    page = out["html"].read_text()
    assert page.index("20.moco.01-motion.run-1.png") < page.index("20.moco.01-motion.run-2.png")
    assert "moco.censor_frac = 0.3" in page


def test_group_table_flags(tmp_path):
    dirs = [_fake_subject(tmp_path, f"sub-{i:02d}", 60.0 + i, 0.05) for i in range(8)]
    dirs.append(_fake_subject(tmp_path, "sub-99", 12.0, 0.2))
    assert subject_name(dirs[0]) == "sub-00"
    out = build_table(dirs, str(tmp_path / "group"), ["moco.tsnr_median < 20"], png=False)
    assert len(out["rows"]) == 18
    reasons = {(f.subject, f.column, f.reason.split()[0]) for f in out["flags"]}
    assert ("sub-99", "moco.censor_frac", "limit") in reasons
    assert ("sub-99", "moco.tsnr_median", "<") in reasons
    assert ("sub-99", "moco.tsnr_median", "robust") in reasons
    assert all(f.subject == "sub-99" for f in out["flags"])
    header = out["tsv"].read_text().splitlines()[0].split("\t")
    assert header[:2] == ["subject", "label"]


def test_group_strip_plot(tmp_path):
    dirs = [_fake_subject(tmp_path, f"sub-{i:02d}", 60.0 + i, 0.05 * i) for i in range(6)]
    out = build_table(dirs, str(tmp_path / "group"))
    assert out["png"].exists()


def test_parse_flag_and_robust_z():
    assert parse_flag("moco.censor_frac >= 0.1") == ("moco.censor_frac", ">=", 0.1)
    with pytest.raises(ValueError):
        parse_flag("censor_frac is big")
    assert np.isnan(robust_z(np.array([1.0, 2.0]))).all()  # too few rows to judge
    z = robust_z(np.array([10.0, 11, 9, 10, 10.5, 9.5, 40]))
    assert z[-1] > 10 and np.all(np.abs(z[:-1]) < 2)


def test_moco_cli_writes_qc(tmp_path):
    """End to end: ffs_moco -qc on a small series leaves figures and a record."""
    import nibabel as nib

    from fastfuncstuff.cli.moco import main

    ser = _brain_series(nt=12).numpy().transpose(3, 2, 1, 0)
    src = tmp_path / "bold.nii.gz"
    img = nib.Nifti1Image(ser, np.diag([-3.0, -3.0, 3.0, 1.0]))
    img.header["pixdim"][4] = 2.0
    nib.save(img, src)
    qc = tmp_path / "qc"
    main(
        [
            "-input",
            str(src),
            "-base",
            "0",
            "-device",
            "cpu",
            "-qc",
            str(qc),
            "-qc_label",
            "run-1",
            "-verb",
            "0",
        ]
    )
    (rec,) = load_records(qc)
    assert rec["label"] == "run-1"
    assert {f["check"] for f in rec["figures"]} == {"motion", "tsnr", "carpet"}
    assert rec["metrics"]["outlier_vol0"]["value"] is not None
    assert json.loads((qc / "records" / "20.moco.run-1.json").read_text())["tool"] == "moco"


def test_alignment_check_flags_a_shifted_source(tmp_path):
    """Coverage falls when the source sits off the base; the flicker is a real
    two-frame APNG."""
    from PIL import Image

    base = _brain_series(nt=1)[0].numpy()
    aff = np.diag([-3.0, -3.0, 3.0, 1.0])
    good = _rec(tmp_path, "good", "allineate")
    checks.alignment_check(good, 1, base, base * 0.5 + 3, aff)
    bad = _rec(tmp_path, "bad", "allineate")
    checks.alignment_check(bad, 1, base, np.roll(base, 6, axis=2), aff)
    assert good.metrics["coverage"].value > 0.99 and good.metrics["mask_dice"].value > 0.99
    assert bad.metrics["coverage"].value < 0.9 and "coverage" in bad.flagged()
    with Image.open(good.figure_path(2, "flicker")) as im:
        assert getattr(im, "n_frames", 1) == 2
