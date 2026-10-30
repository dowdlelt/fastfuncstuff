"""ffs_hrftdv end to end on a sphere-shell ribbon whose response lag rises with depth,
then ffs_reml -hrf_opt on what it wrote."""

from __future__ import annotations

import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch

from tests.viewer.test_surfaces import _subject

VOX, N, TR, NT = 1.5, 36, 1.0, 160


def _curve(lag: float) -> np.ndarray:
    from scipy.stats import gamma

    t = np.arange(320) * 0.1 - lag
    h = np.where(t > 0, gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6, 0)
    return h / h.max()


def _write(tmp_path):
    from fastfuncstuff.design.matrices import build_event_design_microtime

    aff = np.diag([VOX, VOX, VOX, 1.0])
    aff[:3, 3] = -(N - 1) * VOX / 2
    ijk = np.stack(np.meshgrid(*[np.arange(N)] * 3, indexing="ij"), -1)
    r = np.linalg.norm(ijk * VOX + aff[:3, 3], axis=-1)
    ribbon = (r > 20) & (r < 24)
    frac = np.clip((r - 20) / 4, 0, 1)
    lag = np.round(-0.6 + 1.2 * frac, 1)  # later toward pial, as a vein would make it
    rng = np.random.default_rng(0)
    for run in (1, 2):
        on = np.round(np.cumsum(rng.uniform(5, 9, 30)), 1)
        on = on[on < NT * TR - 25]
        cond = np.arange(on.size) % 2
        (tmp_path / f"ev{run}.tsv").write_text(
            "onset\tduration\ttrial_type\n"
            + "".join(f"{o:.1f}\t2\t{'ab'[c]}\n" for o, c in zip(on, cond, strict=True))
        )
        data = 100 + rng.normal(0, 0.4, (N, N, N, NT))
        for lv in np.unique(lag[ribbon]):
            x = build_event_design_microtime(
                all_onsets=[[on[cond == 0]], [on[cond == 1]]],
                durations=[2.0, 2.0],
                hrf_bases=torch.tensor(_curve(lv), dtype=torch.float32),
                n_timepoints_per_run=[NT],
                tr=TR,
                microtime_dt=0.1,
                device=torch.device("cpu"),
            ).numpy()
            data[ribbon & (lag == lv)] += 2.0 * (x @ np.array([1.0, 0.7]))
        nib.save(nib.Nifti1Image(data.astype(np.float32), aff), str(tmp_path / f"run{run}.nii.gz"))
    return ribbon, frac


@pytest.mark.slow  # 14s
def test_hrftdv_finds_lag_rising_with_depth_and_reml_fits_its_pairs(tmp_path):
    ribbon, frac = _write(tmp_path)
    subj = _subject(tmp_path)
    common = ["-input", "run1.nii.gz", "run2.nii.gz", "-events", "ev1.tsv", "ev2.tsv"]
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.hrftdv", *common]
        + ["-fs_subj", str(subj), "-hemi", "lh", "-tr", "1", "-device", "cpu"]
        + ["-patch_mm", "6", "-prefix", "tdv"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    lib = torch.load(tmp_path / "tdv_hrf_library.pt", weights_only=False)
    assert lib["hrf_library"].ndim == 3 and lib["hrf_library"].shape[1] == 2
    assert lib["metadata"]["basis_labels"] == ["early", "late"]
    img = nib.load(tmp_path / "tdv_tdv.nii.gz")
    from fastfuncstuff.cli.util_concalc import _read_brick_labels

    labels = _read_brick_labels(img)
    b = np.asanyarray(img.dataobj)
    resp = (b[..., labels.index("responsive")] > 0) & ribbon
    assert resp.sum() > 100
    rel = b[..., labels.index("rel_lag")][resp]
    assert np.corrcoef(rel, frac[resp])[0, 1] > 0.4
    early = b[..., labels.index("unit_early_lag")][resp]
    late = b[..., labels.index("unit_late_lag")][resp]
    assert np.median(late - early) > 0.5
    index = np.asanyarray(nib.load(tmp_path / "tdv_hrf_index.nii.gz").dataobj)[..., 0]
    assert (index[ribbon] > 0).mean() > 0.95

    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", *common]
        + ["-TR", "1", "-device", "cpu", "-hrf_opt", "tdv", "-Obuck", "o"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    out = nib.load(tmp_path / "o.nii.gz")
    names = _read_brick_labels(out)
    assert {"a_early#0_Coef", "a_late#0_Coef", "b_early#0_Coef"} <= set(names)
    # Superficial voxels are later than their columns' median, so they load on the late curve.
    vol = np.asanyarray(out.dataobj)
    late_share = vol[..., names.index("a_late#0_Coef")] / (
        np.abs(vol[..., names.index("a_early#0_Coef")])
        + np.abs(vol[..., names.index("a_late#0_Coef")])
    )
    deep, sup = resp & (frac < 0.3), resp & (frac > 0.7)
    assert np.median(late_share[sup]) > np.median(late_share[deep]) + 0.2


@pytest.mark.slow  # 15s
def test_pooled_late_writes_an_early_library_and_dsort_runs_reml_fits(tmp_path):
    """-pooled_late: one early column per condition from {prefix}_early, one late
    column per voxel from the per-run -dsort files. The late sizes must grow toward
    pial (later voxels), and voxels outside every unit (zero dsort) must still fit."""
    from fastfuncstuff.cli.util_concalc import _read_brick_labels

    ribbon, frac = _write(tmp_path)
    subj = _subject(tmp_path)
    common = ["-input", "run1.nii.gz", "run2.nii.gz", "-events", "ev1.tsv", "ev2.tsv"]
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.hrftdv", *common]
        + ["-fs_subj", str(subj), "-hemi", "lh", "-tr", "1", "-device", "cpu"]
        + ["-patch_mm", "6", "-tile_mm", "0", "-pooled_late", "-prefix", "tdv"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    early = torch.load(tmp_path / "tdv_early_hrf_library.pt", weights_only=False)
    assert early["hrf_library"].ndim == 2
    dsort = [tmp_path / f"tdv_late_dsort.run-0{r}.nii.gz" for r in (1, 2)]
    assert all(p.exists() for p in dsort)
    assert nib.load(dsort[0]).shape == (N, N, N, NT)

    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", *common]
        + ["-TR", "1", "-device", "cpu", "-hrf_opt", "tdv_early"]
        + ["-dsort", *map(str, dsort), "-dsort_betas", "yes", "-Rbuck", "o"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    out = nib.load(tmp_path / "o.nii.gz")
    names = _read_brick_labels(out)
    vol = np.asanyarray(out.dataobj)
    assert np.isfinite(vol).all()
    a = vol[..., names.index("a#0_Coef")]
    resp = ribbon & (a > 0)
    assert resp.mean() > 0.5 * ribbon.mean()
    late = 0.5 * (
        vol[..., names.index("dsort0_run1#0_Coef")] + vol[..., names.index("dsort0_run2#0_Coef")]
    )
    deep, sup = resp & (frac < 0.3), resp & (frac > 0.7)
    assert np.median(late[sup]) > np.median(late[deep])


@pytest.mark.slow  # 23s
def test_spec_gives_the_events_fit_and_splits_its_contrasts(tmp_path):
    """-spec reaches both tools as the same model -events gives (ffs_autoproc fits
    the design TOML this way), and the spec's contrast comes out per pair column."""
    from fastfuncstuff.cli.util_concalc import _read_brick_labels
    from fastfuncstuff.design.spec import (
        ContrastSpec,
        EventSpec,
        MetaSpec,
        RunSpec,
        Spec,
        write_spec,
    )

    _write(tmp_path)
    subj = _subject(tmp_path)
    write_spec(
        Spec(
            meta=MetaSpec(
                runs=[RunSpec(bold=f"run{r}.nii.gz", events=f"ev{r}.tsv") for r in (1, 2)],
                tr=TR,
                n_timepoints_per_run=[NT, NT],
            ),
            events=[EventSpec(trial_type="a"), EventSpec(trial_type="b")],
            contrasts=[ContrastSpec(label="AvB", sym="SYM: +1*a -1*b")],
        ),
        tmp_path / "design.toml",
    )
    inputs = ["-input", "run1.nii.gz", "run2.nii.gz"]
    by = {"events": ["-events", "ev1.tsv", "ev2.tsv"], "spec": ["-spec", "design.toml"]}

    def run(tool, *args):
        res = subprocess.run(  # noqa: S603
            [sys.executable, "-m", f"fastfuncstuff.cli.{tool}", *inputs, *args, "-device", "cpu"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            cwd=tmp_path,
        )
        assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]

    for how, timing in by.items():
        run("hrftdv", *timing, "-fs_subj", str(subj), "-hemi", "lh", "-tr", "1", "-patch_mm", "6",
            "-prefix", f"tdv_{how}")  # fmt: skip
    assert (tmp_path / "tdv_spec_pairs.tsv").read_text() == (
        tmp_path / "tdv_events_pairs.tsv"
    ).read_text()

    for how, timing in by.items():
        run("reml", *timing, "-TR", "1", "-hrf_opt", "tdv_events", "-Obuck", f"o_{how}")
    ev, sp = (nib.load(tmp_path / f"o_{how}.nii.gz") for how in by)
    names_ev, names_sp = _read_brick_labels(ev), _read_brick_labels(sp)
    assert {"AvB_early#0_Coef", "AvB_late#0_Coef"} <= set(names_sp)
    vev, vsp = np.asanyarray(ev.dataobj), np.asanyarray(sp.dataobj)
    for name in ("a_early#0_Coef", "b_late#0_Coef"):
        np.testing.assert_allclose(
            vsp[..., names_sp.index(name)], vev[..., names_ev.index(name)], atol=1e-4
        )
    con = vsp[..., names_sp.index("AvB_early#0_Coef")]
    a = vsp[..., names_sp.index("a_early#0_Coef")] - vsp[..., names_sp.index("b_early#0_Coef")]
    np.testing.assert_allclose(con, a, atol=1e-4)


@pytest.mark.slow  # 13s
def test_atlas_units_give_one_pair_per_roi_and_a_table(tmp_path):
    """A volume label map (+x half = 1, -x/+y quarter = 2) as units: every ribbon voxel
    of an ROI shares one library entry, -atlas_only leaves the rest uncovered, the
    table's pair is early < late, and -atlas_unit patch splits an ROI into patches."""
    import csv

    ribbon, _ = _write(tmp_path)
    subj = _subject(tmp_path)
    aff = nib.load(tmp_path / "run1.nii.gz").affine
    ijk = np.stack(np.meshgrid(*[np.arange(N)] * 3, indexing="ij"), -1)
    xyz = ijk * VOX + aff[:3, 3]
    lab = np.zeros((N, N, N), np.int16)
    lab[xyz[..., 0] > 0] = 1
    lab[(xyz[..., 0] < 0) & (xyz[..., 1] > 0)] = 2
    nib.save(nib.Nifti1Image(lab, aff), tmp_path / "atlas.nii.gz")
    common = ["-input", "run1.nii.gz", "run2.nii.gz", "-events", "ev1.tsv", "ev2.tsv",
              "-fs_subj", str(subj), "-hemi", "lh", "-tr", "1", "-device", "cpu",
              "-atlas", "atlas.nii.gz", "-extend_mm", "0"]  # fmt: skip

    def run(*extra):
        res = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "fastfuncstuff.cli.hrftdv", *common, *extra],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            cwd=tmp_path,
        )
        assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]

    run("-atlas_only", "-prefix", "roi")
    index = np.asanyarray(nib.load(tmp_path / "roi_hrf_index.nii.gz").dataobj)[..., 0]
    for r in (1, 2):
        got = index[ribbon & (lab == r)]
        assert (got > 0).mean() > 0.9 and np.unique(got[got > 0]).size == 1
    # A voxel joins by its column's owner, so within half a voxel of the cut it can fall
    # on the other side; away from it nothing outside the atlas has a pair.
    far = (xyz[..., 0] < -VOX) & (xyz[..., 1] < -VOX)
    assert (index[ribbon & far] == 0).all()
    rows = list(csv.DictReader(open(tmp_path / "roi_roi.tsv"), delimiter="\t"))
    assert len(rows) == 2
    for row in rows:
        assert row["n_units"] == "1" and row["n_fallback"] == "0"
        assert float(row["early_lag"]) < 0 < float(row["late_lag"])
        assert float(row["early_peak_s"]) < float(row["column_peak_s"]) < float(row["late_peak_s"])

    run("-atlas_unit", "patch", "-patch_mm", "6", "-prefix", "patch")
    rows = list(csv.DictReader(open(tmp_path / "patch_roi.tsv"), delimiter="\t"))
    assert all(int(row["n_units"]) > 1 for row in rows)
