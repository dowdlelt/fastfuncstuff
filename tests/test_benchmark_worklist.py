"""benchmark.worklist + check planning: units, claims, records, pruning that spares originals."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np

from fastfuncstuff.benchmark import census as C
from fastfuncstuff.benchmark.checks import CHECKS
from fastfuncstuff.benchmark.fetch import fetch
from fastfuncstuff.benchmark.worklist import Worklist


def test_add_is_idempotent_and_claims_in_order(tmp_path):
    wl = Worklist(tmp_path / "w.sqlite")
    units = [({"sub": "01", "run": "1"}, ["a.nii.gz"]), ({"sub": "02", "run": "1"}, ["b.nii.gz"])]
    assert wl.add_units("moco", "dsX", units) == 2
    assert wl.add_units("moco", "dsX", units + [({"sub": "03"}, ["c"])]) == 1  # "more subjects"
    u1 = wl.claim_next()
    u2 = wl.claim_next()
    assert json.loads(u1["unit_key"])["sub"] == "01" and u1["status"] == "running"
    assert json.loads(u2["unit_key"])["sub"] == "02"
    wl.record_metrics(u1["id"], "reference", {"r": 0.99})
    wl.record_timing(u1["id"], "ref", "cpu", 10.0, 8)
    wl.record_timing(u1["id"], "ffs", "cuda", 1.0)
    wl.finish(u1["id"])
    wl.finish(u2["id"], "failed", "boom")
    (row,) = wl.results("moco")
    assert row["reference.r"] == 0.99 and row["seconds.ref.cpu"] == 10.0
    assert wl.reset("failed") == 1
    assert {(r["status"], r["n"]) for r in wl.status()} == {("done", 1), ("pending", 2)}


def test_claims_round_robin_over_check_and_dataset(tmp_path):
    """A night cut off early must have touched every (check, dataset), not finished one."""
    wl = Worklist(tmp_path / "w.sqlite")
    wl.add_units("moco", "dsA", [({"sub": f"{i}"}, []) for i in range(3)])
    wl.add_units("moco", "dsB", [({"sub": f"{i}"}, []) for i in range(2)])
    wl.add_units("blipflip", "dsA", [({"sub": "0"}, [])])
    order = [(u["check_name"], u["dataset"]) for u in iter(wl.claim_next, None)]
    assert order[:3] == [("moco", "dsA"), ("blipflip", "dsA"), ("moco", "dsB")]
    assert len(order) == 6
    wl.add_units("newcheck", "dsA", [({"sub": "0"}, [])])
    assert wl.claim_next(known_checks=["moco"]) is None  # unknown check stays pending
    assert wl.claim_next(known_checks=["moco", "newcheck"])["check_name"] == "newcheck"


def test_new_check_first_unit_jumps_the_queue(tmp_path):
    wl = Worklist(tmp_path / "w.sqlite")
    for ds in ("dsA", "dsB", "dsC"):
        wl.add_units("moco", ds, [({"sub": f"{i}"}, []) for i in range(2)])
    wl.add_units("reml", "dsC", [({"sub": f"{i}"}, []) for i in range(2)])
    first, second = wl.claim_next(), wl.claim_next()
    assert first["check_name"] == "moco" and second["check_name"] == "reml"  # smoke unit
    rest = [(u["check_name"], u["dataset"]) for u in iter(wl.claim_next, None)]
    assert rest[:2] == [("moco", "dsB"), ("moco", "dsC")]  # then round-robin resumes


def test_prune_removes_regenerable_only_and_keeps_rows(tmp_path):
    wl = Worklist(tmp_path / "w.sqlite")
    wl.add_units("moco", "dsX", [({"sub": "01"}, [])])
    u = wl.claim_next()
    big, small = tmp_path / "big.nii", tmp_path / "small.1D"
    big.write_bytes(b"x" * 1000)
    small.write_text("1 2 3")
    wl.record_artifact(u["id"], big, "regenerable")
    wl.record_artifact(u["id"], small, "keep")
    assert wl.prune(u["id"]) == 1000
    assert not big.exists() and small.exists()
    rows = wl.db.execute("SELECT path, pruned_at FROM artifacts").fetchall()
    assert len(rows) == 2 and any(r["pruned_at"] for r in rows)


def _bids(root):
    for sub in ("01", "02", "03"):
        f = root / f"sub-{sub}" / "func"
        f.mkdir(parents=True)
        for run in ("1", "2"):
            for echo in ("1", "2"):
                p = f / f"sub-{sub}_task-rest_run-{run}_echo-{echo}_bold.nii.gz"
                nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 3), np.int16), np.eye(4)), p)
    return root


def test_breadth_plan_spreads_subjects_and_is_stable(tmp_path):
    root = _bids(tmp_path / "ds")
    src = C.LocalSource(root)
    m = C.census(src)
    idx = C.Index(src.list_files())
    check = CHECKS["moco"]()
    two = check.plan(m, idx, 2)
    assert len({k["sub"] for k, _ in two}) == 2  # distinct subjects before repeats
    assert all(inp[0].endswith("echo-1_bold.nii.gz") for _, inp in two)  # first echo only
    four = check.plan(m, idx, 4)
    assert [k for k, _ in four][:2] == [k for k, _ in two]  # growing n keeps earlier picks
    assert len({k["sub"] for k, _ in four}) == 3


def test_local_fetch_never_offers_originals_for_pruning(tmp_path):
    root = _bids(tmp_path / "ds")
    m = C.census(C.LocalSource(root))
    rel = "sub-01/func/sub-01_task-rest_run-1_echo-1_bold.nii.gz"
    paths, downloaded = fetch(m, [rel], tmp_path / "cache")
    assert paths[rel] == root / rel and downloaded == []


def test_reml_events_to_stim_times(tmp_path):
    """Fixed durations -> BLOCK(d,1), varying -> married dmBLOCK, n/a and singletons dropped."""
    from fastfuncstuff.benchmark.checks.reml import _conditions, _stim_args

    ev = tmp_path / "x_events.tsv"
    ev.write_text(
        "onset\tduration\ttrial_type\n"
        "0\t2\tface\n10\t2\tface\n"
        "5\t1\thouse\n15\t3\thouse\n"
        "20\t0\tn/a\n25\t0\tn/a\n"
        "30\t1\tlonely\n"
    )
    conds = _conditions(ev)
    assert set(conds) == {"face", "house", "event"}  # n/a pooled as 'event', singleton dropped
    args = _stim_args(conds, tmp_path)
    assert args[:2] == ["-num_stimts", "3"]
    models = {args[i + 3]: args[i + 6] for i, a in enumerate(args) if a.startswith("-stim_times")}
    by_label = dict(zip(models.values(), models.keys(), strict=True))
    assert by_label["face"] == "BLOCK(2,1)"
    assert by_label["house"] == "dmBLOCK(1)"
    assert by_label["event"] == "GAM"
    assert args[args.index("dmBLOCK(1)") - 3] == "-stim_times_AM1"  # 3dDeconvolve refuses plain
    house_file = next(p for p in tmp_path.glob("c*.1D") if ":" in p.read_text())
    assert house_file.read_text().split() == ["5.000:1.000", "15.000:3.000"]


def test_allineate_affine_distance(tmp_path):
    """A 2 mm shift reads as 2 mm RMS; the aff12 row parses as base->source 3x4."""
    from fastfuncstuff.benchmark.checks.allineate import read_aff12, rms_displacement

    f = tmp_path / "m.aff12.1D"
    f.write_text("# 3dAllineate matrix\n 1 0 0 2  0 1 0 0  0 0 1 0\n")
    m = read_aff12(f)
    assert m[0, 3] == 2 and m[3, 3] == 1
    pts = np.random.default_rng(0).normal(size=(100, 3)) * 50
    assert abs(rms_displacement(m, np.eye(4), pts) - 2.0) < 1e-9
    rot = np.eye(4)
    rot[:2, :2] = [[0, -1], [1, 0]]  # 90 degrees: displacement grows with distance
    assert rms_displacement(rot, np.eye(4), pts) > 50


def test_reml_events_non_bids_columns(tmp_path):
    from fastfuncstuff.benchmark.checks.reml import _conditions

    ev = tmp_path / "y_events.tsv"
    ev.write_text(
        "event_onset\tduration\tstimulus\n1\t20\tcheck\n60\t20\tcheck\n3\t0.4\tletter\n9\t0.4\tletter\n"
    )
    assert {k: len(v) for k, v in _conditions(ev).items()} == {"check": 2, "letter": 2}


def test_timed_cached_reuses_recorded_fit(tmp_path):
    """A rerun with the same command and outputs returns the recorded seconds, no refit."""
    from fastfuncstuff.benchmark.checks.base import timed_cached

    out, stamp, log = tmp_path / "out.txt", tmp_path / "fit.timing.json", tmp_path / "log"
    cmd = ["sh", "-c", f"echo x >> {out}"]
    t1 = timed_cached(cmd, stamp, [out], tmp_path, log)
    t2 = timed_cached(cmd, stamp, [out], tmp_path, log)
    assert t2 == t1 and out.read_text() == "x\n"
    timed_cached([*cmd[:2], cmd[2] + " "], stamp, [out], tmp_path, log)  # new command: refits
    assert out.read_text() == "x\nx\n"


def test_parse_allcost_reads_named_costs():
    from fastfuncstuff.benchmark.checks.allineate import parse_allcost

    text = "++ allcost output: init #0\n   ls   = 0.2856\n   nmi  = 0.946878\n   lpc+ = 1\n   hel  = -0.0854437\n"
    assert parse_allcost(text) == {"ls": 0.2856, "nmi": 0.946878, "hel": -0.0854437}


def test_trailing_noise_count_finds_rf_off_volumes():
    """HiHiFaces-like: ~5280 signal, 3 trailing volumes at ~449 (8%); nothing else dropped."""
    from fastfuncstuff.benchmark.checks.base import trailing_noise_count

    sig = [5280.0 + (i % 7) for i in range(100)]
    assert trailing_noise_count(sig + [449.7, 449.1, 448.8]) == 3
    assert trailing_noise_count(sig) == 0
    assert trailing_noise_count([449.0] + sig) == 0  # leading low volume is not trailing noise


def test_unit_curation_and_read_meta_overrides(tmp_path):
    """Curated sidecar values beat the run's sidecar; per-acquisition entries beat the dataset's."""
    import json

    import pytest

    from fastfuncstuff.benchmark.census import apply_curation, unit_curation
    from fastfuncstuff.benchmark.checks.base import Check

    m = {"dataset_id": "X", "acquisitions": [], "curated": {
        "source": "dataset owner", "sidecar": {"RepetitionTime": 1.184},
        "noise_volumes": 3, "noise_volumes_acquisitions": {"task-b_bold": 0}}}  # fmt: skip
    a, b = unit_curation(m, "task-a_bold"), unit_curation(m, "task-b_bold")
    assert a == {"sidecar": {"RepetitionTime": 1.184}, "noise_volumes": 3,
                 "stimulus_model": None, "sidecar_files": [], "source": "dataset owner"}  # fmt: skip
    assert b["noise_volumes"] == 0
    side = tmp_path / "run_bold.json"
    side.write_text(json.dumps({"RepetitionTime": 0.037, "EchoTime": 0.02}))
    chk = Check()
    chk.curation = a
    assert chk.read_meta({"run_bold.json": side}) == {"RepetitionTime": 1.184, "EchoTime": 0.02}
    with pytest.raises(ValueError, match="source"):
        apply_curation({**m, "curated": {"sidecar": {"RepetitionTime": 1.0}}})


def test_sinusoid_regressors_phase_free():
    """sin and cos at 0.05 Hz span any phase; zero before the block, HRF-delayed."""
    import numpy as np

    from fastfuncstuff.benchmark.checks.reml import sinusoid_regressors

    tr, nt = 0.5, 600
    r = sinusoid_regressors([(14.0, 240.0)], 0.05, tr, nt)
    assert set(r) == {"block", "sin", "cos"} and all(v.shape == (nt,) for v in r.values())
    t = np.arange(nt) * tr
    assert np.abs(r["sin"][t < 14]).max() == 0 and r["block"][(t > 40) & (t < 250)].min() > 0
    # a 0.05 Hz drive at any phase is fit by the sin/cos pair
    shifted = sinusoid_regressors([(14.0, 240.0)], 0.05, tr, nt, dt=0.05)
    X = np.c_[r["sin"], r["cos"]]
    y = np.cos(0.7) * shifted["sin"] + np.sin(0.7) * shifted["cos"]
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.abs(resid).max() < 1e-9


def test_file_overrides_match_entities():
    """AP and PA fmaps of one acquisition get different PhaseEncodingDirection."""
    from fastfuncstuff.benchmark.checks.base import Check

    chk = Check()
    chk.curation = {"sidecar_files": [
        {"match": {"dir": "AP", "suffix": "epi"}, "set": {"PhaseEncodingDirection": "j-"}},
        {"match": {"dir": "PA", "suffix": "epi"}, "set": {"PhaseEncodingDirection": "j"}},
    ]}  # fmt: skip
    ap = "sub-01/ses-02/fmap/sub-01_ses-02_dir-AP_run-01_part-mag_epi.nii.gz"
    pa = ap.replace("dir-AP", "dir-PA")
    assert chk.file_overrides(ap) == {"PhaseEncodingDirection": "j-"}
    assert chk.file_overrides(pa) == {"PhaseEncodingDirection": "j"}
    assert chk.file_overrides("sub-01/func/sub-01_task-x_bold.nii.gz") == {}


def test_allineate_tissue_judge_separates_on_and_off_brain():
    import numpy as np

    from fastfuncstuff.benchmark.checks.allineate import boundary_contrast, epi_in_brain

    n = 48
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float32),) * 3, indexing="ij")
    r = np.sqrt((z - 24) ** 2 + (y - 24) ** 2 + (x - 24) ** 2)
    brain = r < 16
    wm = np.clip(10.0 - r, 0.0, 1.0)  # WM core r < 9.5, boundary at r ~ 9.5
    epi = np.where(brain, np.where(r < 9.5, 60.0, 100.0), 0.0).astype(np.float32)  # GM > WM
    vox = (1.0, 1.0, 1.0)
    right = boundary_contrast(epi, wm, brain, vox)
    assert right["pos_frac"] > 0.9 and right["contrast"] > 0.2
    assert epi_in_brain(epi, brain, vox) > 0.95
    off = np.roll(epi, 30, axis=2)  # the data moved off the brain
    assert epi_in_brain(off, brain, vox) < 0.5
    flat = np.where(brain, 80.0, 0.0).astype(np.float32)  # no tissue contrast seen
    assert abs(boundary_contrast(flat, wm, brain, vox)["contrast"]) < 1e-3
