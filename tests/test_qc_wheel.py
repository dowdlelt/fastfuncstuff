"""QC of what a GLM left behind, and the session wheel drawn from it."""

from __future__ import annotations

import json
import types

import numpy as np

from fastfuncstuff.qc import checks, plots, wheel
from fastfuncstuff.qc import residuals as rs
from fastfuncstuff.qc.record import QCRecord, load_records
from fastfuncstuff.qc.report import build_report
from fastfuncstuff.qc.summary import _task_and_variant

NR, RUNS = 80, 3  # volumes per run, runs


def _rec(tmp_path, label, tool="reml"):
    return QCRecord.for_args(types.SimpleNamespace(qc=str(tmp_path), qc_label=label), tool, "x")


def _design(rng, extra: int = 0):
    """Per-run constant + linear drift, one block task, one motion-like nuisance."""
    t = RUNS * NR
    cols, labels, groups = [], [], []
    for r in range(RUNS):
        for j, shape in enumerate((np.ones(NR), np.linspace(-1, 1, NR))):
            c = np.zeros(t)
            c[r * NR : (r + 1) * NR] = shape
            cols.append(c)
            labels.append(f"Run#{r + 1}Pol#{j}")
            groups.append(-1)
    task = np.tile(((np.arange(NR) // 10) % 2).astype(float), RUNS)
    cols.append(task)
    labels.append("task#0")
    groups.append(1)
    for k in range(1 + extra):
        cols.append(rng.standard_normal(t))
        labels.append(f"nuis{k}")
        groups.append(0)
    return {
        "matrix": np.stack(cols, 1),
        "column_labels": labels,
        "column_groups": groups,
        "run_starts": [r * NR for r in range(RUNS)],
    }, task


def _volume(rng, task, bad_run: int | None = 2):
    """(x, y, z, t) series: the low-x half responds to the task; one run is noisier."""
    nx, ny, nz = 16, 12, 8
    data = 100 + rng.standard_normal((nx, ny, nz, RUNS * NR)).astype(np.float32)
    data[: nx // 2] += 2.0 * task.astype(np.float32)
    if bad_run is not None:
        sl = slice(bad_run * NR, (bad_run + 1) * NR)
        data[..., sl] += 3.0 * rng.standard_normal(NR).astype(np.float32)
    flat_index = np.arange(nx * ny * nz).reshape(nx, ny, nz).transpose(2, 1, 0)
    responds = np.zeros((nz, ny, nx), bool)
    responds[:, :, : nx // 2] = True
    return data, flat_index, responds


def test_more_regressors_do_not_buy_tsnr():
    """The reason both SDs are dof-corrected: 60 columns of pure noise remove ~25% of
    the variance of pure-noise data, which an uncorrected SD reports as a tSNR gain."""
    rng = np.random.default_rng(0)
    design, _ = _design(rng, extra=60)
    x = design["matrix"]
    drift = [i for i, g in enumerate(design["column_groups"]) if g < 0]
    data = 100 + rng.standard_normal((4000, x.shape[0])).astype(np.float32)
    st = rs.residual_stats(data, np.arange(4000), x, drift, design["run_starts"], device="cpu")
    assert st.rank_full == x.shape[1] and st.rank_drift == len(drift)
    gain = np.median(st.tsnr_res() / st.tsnr_in())
    assert abs(gain - 1.0) < 0.01  # 1.000 measured; 1.16 without the correction
    naive = np.sqrt((st.n_kept - st.rank_drift) / (st.n_kept - st.rank_full))
    assert naive > 1.15


def test_zero_column_is_not_charged_to_the_dof():
    """A condition with no events in the kept volumes is a zero column: rank, not
    column count, is what the residual lost."""
    rng = np.random.default_rng(1)
    design, _ = _design(rng)
    x = np.column_stack([design["matrix"], np.zeros(RUNS * NR)])
    assert rs.orthonormal_basis(x).shape[1] == x.shape[1] - 1


def test_fit_parts_partition_and_censored_volumes():
    """data = fitted + residual exactly, and a censored spike stays in the residual
    instead of being fitted or dropped."""
    rng = np.random.default_rng(2)
    design, task = _design(rng)
    x = design["matrix"]
    drift = [i for i, g in enumerate(design["column_groups"]) if g < 0]
    y = (100 + rng.standard_normal((x.shape[0], 50)) + 2 * task[:, None]).astype(np.float32)
    y[17] += 40.0
    keep = np.ones(x.shape[0], bool)
    keep[17] = False
    d, f, r = rs.fit_parts(y, x, drift, keep)
    np.testing.assert_allclose(d, f + r, atol=1e-4)
    assert abs(np.delete(r, 17, axis=0).mean()) < 0.05  # the run means went with the drift
    assert r[17].mean() > 30
    assert np.corrcoef(f.mean(1), task)[0, 1] > 0.9


def test_tissue_groups_need_a_confident_voxel_and_pool_names():
    mask = np.ones((2, 2, 3), bool)
    p = np.zeros((3, 2, 2, 3), np.float32)
    p[0, :, :, 0] = 0.95  # confident class 1
    p[1, :, :, 1] = 0.6  # partial volume: joins nothing
    p[2, :, :, 2] = 0.5
    p[1, :, :, 2] = 0.45  # 0.5 + 0.45 pooled under one name clears 0.9
    g = rs.tissue_groups(p, ["GM", "WM", "WM"], mask)
    assert g["GM"][:, :, 0].all() and not g["GM"][:, :, 1:].any()
    assert g["WM"][:, :, 2].all() and not g["WM"][:, :, 1].any()
    lab = np.array([[[1, 2, 3]]])
    g = rs.tissue_groups(lab, ["GM", "-", "CSF"], np.ones((1, 1, 3), bool))
    assert set(g) == {"GM", "CSF"} and g["CSF"][0, 0, 2]


def test_carpet_keeps_units_so_a_quiet_group_is_dimmer():
    """z-scoring (the raw-data carpet's choice) would make both groups equally loud;
    the model carpets must not, and must share one scale across panels."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((100, 2000)).astype(np.float32)
    x[:, 1000:] *= 0.25
    groups = [("loud", np.arange(1000)), ("quiet", np.arange(1000, 2000))]
    cp = plots.carpet_rows(x, groups, max_rows=100, standardize=False, order="none")
    (_, a0, a1), (_, b0, b1) = cp.blocks
    assert cp.rows[a0:a1].std() > 3 * cp.rows[b0:b1].std()
    z = plots.carpet_rows(x, groups, max_rows=100, order="none")
    assert z.rows[a0:a1].std() < 1.3 * z.rows[b0:b1].std()
    half = plots.carpet_rows(0.5 * x, groups, max_rows=100, standardize=False, scale=cp.scale)
    assert half.scale == cp.scale
    # An external score sets the row order: voxel 0 first, whatever its own data says.
    score = -np.arange(2000, dtype=np.float32)
    s = plots.carpet_rows(
        x, [("all", np.arange(2000))], max_rows=2000, standardize=False, score=score
    )
    np.testing.assert_allclose(s.rows[0] * s.scale, x[:, 0] - np.median(x[:, 0]), atol=1e-4)


def _fit(tmp_path, label, seed=0, bad_run=2, run_labels=None):
    rng = np.random.default_rng(seed)
    design, task = _design(rng)
    data, flat_index, responds = _volume(rng, task, bad_run)
    mask = np.ones(responds.shape, bool)
    rec = _rec(tmp_path, label)
    rec.metric("fstat_frac_sig", 0.2)
    checks.residual_check(
        rec,
        4,
        data,
        flat_index,
        mask,
        np.diag([-3.0, -3.0, 3.0, 1.0]),
        design,
        tr=1.5,
        device="cpu",
        tissue={"GM": responds, "WM": ~responds},
        run_labels=run_labels,
    )
    rec.write(verb=0)
    return rec


def test_residual_check_names_the_bad_run(tmp_path):
    """The run with added global noise has the lowest tSNR and the worst held-out
    fit; the layout mapping (x, y, z, t) -> (z, y, x) puts the response in GM."""
    rec = _fit(tmp_path, "task-a")
    assert [f["check"] for f in rec.figures] == ["tsnr_gain", "resid_carpet"]
    for f in rec.figures:
        assert (tmp_path / f["file"]).stat().st_size > 20000
    tin, tres, r2 = (rec.series[k] for k in ("run_tsnr_input", "run_tsnr_resid", "run_heldout_r2"))
    assert int(np.argmin(tres)) == 2 and tres[2] < 0.6 * tres[0]
    assert int(np.argmin(r2)) == 2 and r2[0] > 0.3  # 0.49 / 0.47 / 0.13 measured
    # The task is in half the voxels, so the model's gain is there and only there.
    assert tres[0] > 1.1 * tin[0]
    assert rec.metrics["tsnr_gain_GM"].value > 1.1 > rec.metrics["tsnr_gain_WM"].value
    with np.load(rec.sidecar_path(checks.WHEEL_SIDECAR)) as z:
        meta = json.loads(str(z["meta"]))
        assert z["tsnr"].ndim == 3
    assert meta["runs"] == ["run-01", "run-02", "run-03"] and meta["frac_sig"] == 0.2
    for q, med in zip(meta["run_tsnr_input_q"], meta["run_tsnr_input"], strict=True):
        assert q == sorted(q) and q[2] == med and q[0] < q[4]


def test_a_few_large_frames_stay_visible_in_the_motion_ring():
    """A still run that lurched 3 times in 300 frames: by share alone the lurches
    are 1% of the volumes, a sliver. They must still be marked as present, at
    their own value, and nowhere between them and the still frames."""
    rng = np.random.default_rng(0)
    still = 0.03 * np.exp(0.2 * rng.standard_normal(297))
    motion = np.r_[still, 1.5, 1.6, 1.4]
    grid = np.linspace(*np.log10(wheel.MOTION_AXIS), 97)
    dens, seen = wheel.enorm_density(motion, grid)
    at = lambda mm: np.argmin(np.abs(grid - np.log10(mm)))  # noqa: E731
    assert dens[at(1.5)] / wheel.ENORM_DENS_FULL < wheel.SEEN_WIDTH  # too thin alone
    assert seen[at(1.5)] and not seen[at(0.4)] and seen[at(0.03)]
    frac = np.r_[np.zeros(10), np.full(287, 0.002), 0.2, 0.25, 0.3]
    h = wheel.outlier_hist(frac)
    assert np.isclose(h.sum(), 1.0) and h[0] == 10 / 300  # zeros land in the first bin
    assert (h[-4:] > 0).any()


def test_wheel_layout_is_fixed_whatever_the_data():
    """Equal sectors, a fixed gap at the top, tasks in order: the geometry depends on
    the number of runs and on nothing measured."""
    runs = [wheel.WheelRun(f"run-{i}") for i in range(3)]
    tasks = [wheel.WheelTask("a", runs[:2]), wheel.WheelTask("b", runs[2:])]
    spans = wheel.layout(tasks)
    widths = [a1 - a0 for t in spans for a0, a1 in t]
    assert np.allclose(widths, widths[0])
    assert spans[0][0][0] == wheel.GAP_TOP / 2
    assert np.isclose(spans[1][0][1], 360 - wheel.GAP_TOP / 2)
    assert np.isclose(spans[1][0][0] - spans[0][1][1], wheel.GAP_TASK)
    assert wheel.short_run_label("ses-02.task-x.run-010", 0) == "s02 r10"
    assert wheel.short_run_label("anything", 4) == "r5"
    assert _task_and_variant("stage12.blur6.task-faces") == ("faces", "stage12.blur6")
    assert _task_and_variant("stage12.task-faces") == ("faces", "stage12")
    assert _task_and_variant("myfit") == ("myfit", "")


def test_report_draws_one_wheel_per_variant_and_finds_each_runs_motion(tmp_path):
    names = {
        t: [f"stage10.final.sub-1.task-{t}.run-{i + 1:02d}" for i in range(RUNS)] for t in "ab"
    }
    _fit(tmp_path, "stage12.task-a", seed=0, run_labels=names["a"])
    _fit(tmp_path, "stage12.task-b", seed=1, bad_run=None, run_labels=names["b"])
    _fit(tmp_path, "stage12.blur6.task-a", seed=0, run_labels=names["a"])
    rng = np.random.default_rng(5)
    for t in "ab":
        for i in range(RUNS):
            loud = t == "a" and i == 2
            params = np.cumsum((0.4 if loud else 0.02) * rng.standard_normal((NR, 6)), axis=0)
            enorm = np.r_[0, np.linalg.norm(np.diff(params, axis=0), axis=1)]
            rec = _rec(tmp_path, f"sub-1.task-{t}.run-{i + 1:02d}", "moco")
            checks.motion_check(
                rec,
                1,
                params,
                tr=1.0,
                enorm=enorm,
                outlier_frac=np.minimum(enorm / 10, 1.0),
                keep=(enorm < 0.3).astype(int),
                motion_limit=0.3,
                outlier_limit=0.05,
            )
            rec.write(verb=0)
    out = build_report(tmp_path)
    wheels = [s for s in out["summaries"] if "wheel" in s]
    assert wheels == ["00.session.00-wheel.stage12.png", "00.session.00-wheel.stage12.blur6.png"]
    page = out["html"].read_text()
    assert page.index("00-wheel.stage12.png") < page.index("20.moco")  # the summary comes first

    from fastfuncstuff.qc import summary

    seen = {}
    real = wheel.wheel

    def spy(tasks, slots, *a, **k):
        seen.setdefault("calls", []).append((tasks, slots, a[3] if len(a) > 3 else k.get("flags")))
        return real(tasks, slots, *a, **k)

    wheel.wheel = spy
    try:
        summary.session_wheel(tmp_path, load_records(tmp_path))
    finally:
        wheel.wheel = real
    tasks, slots, flags = seen["calls"][0]
    assert [t.name for t in tasks] == ["a", "b"] and [len(t.runs) for t in tasks] == [3, 3]
    assert slots == ["moco"]
    bad = tasks[0].runs[2]
    assert bad.motion is not None and bad.motion.size == NR and bad.motion_limit == 0.3
    assert bad.outlier is not None and bad.outlier.size == NR and bad.outlier_limit == 0.05
    assert bad.tsnr_in_q is not None and bad.tsnr_in_q[2] == bad.tsnr_in
    assert np.nanmean(bad.motion) > 5 * np.nanmean(tasks[0].runs[0].motion)
    assert bad.status == {"moco": "warn"} and tasks[1].runs[0].status == {"moco": "ok"}
    assert any(f.startswith("a r3: moco censor_frac") for f in flags)
    assert tasks[0].frac_sig == 0.2


def test_reml_cli_writes_the_residual_qc_with_tissue_from_another_grid(tmp_path):
    """End to end: ffs_reml -qc -qc_tissue. The segmentation sits on a finer grid
    than the data, so this also pins the (x, y, z, t) -> (z, y, x) mapping: get it
    wrong and the responding half is no longer the one called GM."""
    import subprocess
    import sys

    import nibabel as nib

    rng = np.random.default_rng(4)
    shape, n, tr = (10, 8, 6), 90, 2.0
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    t = np.arange(n) * tr
    paths, events = [], []
    for r in range(2):
        onsets = np.arange(8.0 + 4 * r, n * tr - 24, 24.0)
        resp = sum(np.exp(-0.5 * ((t - o - 6) / 3) ** 2) for o in onsets)
        arr = 100 + rng.normal(0, 1, (*shape, n))
        arr[:5] += 3 * resp
        img = nib.Nifti1Image(arr.astype(np.float32), aff)
        img.header.set_xyzt_units(xyz="mm", t="sec")
        img.header.set_zooms((3, 3, 3, tr))
        nib.save(img, tmp_path / f"sub-1_task-x_run-{r + 1}.nii.gz")
        (tmp_path / f"ev{r}.tsv").write_text(
            "onset\tduration\ttrial_type\n" + "".join(f"{o}\t6\ta\n" for o in onsets)
        )
        paths.append(f"sub-1_task-x_run-{r + 1}.nii.gz")
        events.append(f"ev{r}.tsv")
    fine = np.full((20, 16, 12), 2, np.int16)  # 1.5 mm labels: 1 where the task is
    fine[:10] = 1
    fa = np.diag([1.5, 1.5, 1.5, 1.0])
    fa[:3, 3] = -0.75
    nib.save(nib.Nifti1Image(fine, fa), tmp_path / "seg.nii.gz")
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", "-input", *paths, "-events", *events]
        + ["-hrf_model", "SPMG1", "-device", "cpu", "-Obuck", "stats", "-qc", "qc"]
        + ["-qc_label", "task-x", "-qc_tissue", "seg.nii.gz", "-qc_tissue_names", "GM", "WM"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    (rec,) = load_records(tmp_path / "qc")
    checks_made = [f["check"] for f in rec["figures"]]
    assert checks_made[-2:] == ["tsnr_gain", "resid_carpet"], res.stdout[-2000:]
    m = {k: v["value"] for k, v in rec["metrics"].items()}
    assert m["tsnr_gain_GM"] > 1.1 and abs(m["tsnr_gain_WM"] - 1.0) < 0.05  # 1.16 / 1.00
    with np.load(tmp_path / "qc" / "records" / "80.reml.task-x.wheel.npz") as z:
        meta = json.loads(str(z["meta"]))
    assert meta["runs"] == ["sub-1_task-x_run-1", "sub-1_task-x_run-2"]
    assert all(v is not None and v > 0.2 for v in meta["run_heldout_r2"])
