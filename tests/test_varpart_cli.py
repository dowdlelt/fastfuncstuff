"""End-to-end tests for the ffs_varpart CLI.

Covers the wiring the primitive tests cannot: sidecar parsing, mask/atlas handling,
sub-brick assembly, and the failure messages a user is most likely to hit.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

sys.path.insert(0, str(Path(__file__).parent))
from test_variance_partition import (  # noqa: E402
    N_STIM,
    N_TASK,
    centered,
    make_crossed_table,
)

from fastfuncstuff.cli.varpart import main  # noqa: E402


def _fixture(tmp_path, shape=(4, 4, 3), noise=1.0, seed=0, drop_col=None):
    factors, rep, run = make_crossed_table()
    rng = np.random.default_rng(seed)
    n_vox = int(np.prod(shape))
    n_tr = len(factors["stim"])
    a = centered(rng.normal(0, 1.0, size=(n_vox, N_STIM)), axis=1)
    b = centered(rng.normal(0, 1.0, size=(n_vox, N_TASK)), axis=1)
    s, t = factors["stim"], factors["task"]
    y = a[:, s] + b[:, t] + rng.normal(0, noise, size=(n_vox, n_tr))

    betas = tmp_path / "betas.nii.gz"
    nib.save(nib.Nifti1Image(y.reshape(*shape, n_tr).astype(np.float32), np.eye(4)), betas)

    tsv = tmp_path / "trials.csv"
    cols = ["stim", "task", "run", "repeat"]
    if drop_col:
        cols.remove(drop_col)
    with open(tsv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(n_tr):
            vals = {"stim": f"s{s[i]}", "task": f"t{t[i]}", "run": f"r{run[i]}", "repeat": rep[i]}
            w.writerow([vals[c] for c in cols])
    return betas, tsv, shape, n_tr


def _run(args):
    argv = sys.argv
    sys.argv = ["ffs_varpart", *args]
    try:
        return main()
    finally:
        sys.argv = argv


def test_voxel_mode_writes_expected_subbricks(tmp_path):
    betas, tsv, shape, _ = _fixture(tmp_path)
    out = tmp_path / "vp"
    assert (
        _run(
            [
                "-betas",
                str(betas),
                "-trials",
                str(tsv),
                "-factors",
                "stim,task",
                "-prefix",
                str(out),
                "-quiet",
            ]
        )
        == 0
    )

    img = nib.load(str(out) + ".nii.gz")
    assert img.shape[:3] == shape
    assert img.shape[3] == 14
    assert (tmp_path / "vp_varpart.json").exists()

    import json

    meta = json.loads((tmp_path / "vp_varpart.json").read_text())
    assert meta["unit"] == "voxel"
    assert meta["factors"] == ["stim", "task"]
    # Balanced crossed design: shared variance must come out ~0.
    assert abs(meta["diagnostics"]["shared_abs_median"]) < 0.02


def test_atlas_mode_writes_roi_table(tmp_path):
    betas, tsv, shape, _ = _fixture(tmp_path)
    atlas = tmp_path / "atlas.nii.gz"
    lab = (np.arange(int(np.prod(shape))) % 4 + 1).reshape(shape).astype(np.int16)
    nib.save(nib.Nifti1Image(lab, np.eye(4)), atlas)

    out = tmp_path / "vproi"
    assert (
        _run(
            [
                "-betas",
                str(betas),
                "-trials",
                str(tsv),
                "-factors",
                "stim,task",
                "-atlas",
                str(atlas),
                "-prefix",
                str(out),
                "-quiet",
            ]
        )
        == 0
    )

    rows = list(csv.DictReader(open(str(out) + "_roi.tsv"), delimiter="\t"))
    assert len(rows) == 4
    assert {"roi", "unique_stim", "unique_task", "interaction", "rank_E", "ncsnr"} <= set(rows[0])

    # ROI mode also paints a volume for figures: same grid and sub-bricks as voxel mode,
    # constant within each parcel.
    img = nib.load(str(out) + ".nii.gz")
    assert img.shape == (*shape, 14)
    painted = np.asanyarray(img.dataobj)
    names = list(rows[0])
    ustim = painted[..., names.index("unique_stim") - 1]
    for roi_i, row in enumerate(rows, start=1):
        vals = ustim[lab == roi_i]
        assert np.allclose(vals, vals.flat[0]), "parcel must be constant"
        assert np.isclose(vals.flat[0], float(row["unique_stim"]), atol=1e-5)


def test_permutation_adds_pvalue_columns(tmp_path):
    betas, tsv, shape, _ = _fixture(tmp_path)
    atlas = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.int16), np.eye(4)), atlas)

    out = tmp_path / "vpp"
    assert (
        _run(
            [
                "-betas",
                str(betas),
                "-trials",
                str(tsv),
                "-factors",
                "stim,task",
                "-atlas",
                str(atlas),
                "-perm",
                "20",
                "-prefix",
                str(out),
                "-quiet",
            ]
        )
        == 0
    )

    rows = list(csv.DictReader(open(str(out) + "_roi.tsv"), delimiter="\t"))
    assert "p_fwe_unique_stim" in rows[0]
    assert "p_unc_interaction" in rows[0]
    assert 0.0 < float(rows[0]["p_fwe_unique_stim"]) <= 1.0


def test_trial_count_mismatch_is_a_clear_error(tmp_path):
    betas, tsv, _, _ = _fixture(tmp_path)
    short = tmp_path / "short.csv"
    rows = list(csv.DictReader(open(tsv)))[:-5]
    with open(short, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with pytest.raises(SystemExit, match="One row per volume"):
        _run(
            [
                "-betas",
                str(betas),
                "-trials",
                str(short),
                "-factors",
                "stim,task",
                "-prefix",
                str(tmp_path / "x"),
                "-quiet",
            ]
        )


def test_missing_factor_column_names_available_columns(tmp_path):
    betas, tsv, _, _ = _fixture(tmp_path)
    with pytest.raises(SystemExit, match="column 'nope' not found"):
        _run(
            [
                "-betas",
                str(betas),
                "-trials",
                str(tsv),
                "-factors",
                "stim,nope",
                "-prefix",
                str(tmp_path / "x"),
                "-quiet",
            ]
        )


def test_requires_exactly_two_factors(tmp_path):
    betas, tsv, _, _ = _fixture(tmp_path)
    with pytest.raises(SystemExit, match="exactly 2 column names"):
        _run(
            [
                "-betas",
                str(betas),
                "-trials",
                str(tsv),
                "-factors",
                "stim",
                "-prefix",
                str(tmp_path / "x"),
                "-quiet",
            ]
        )


def test_runs_without_run_or_repeat_columns(tmp_path):
    """repeat is derivable from cell order; run only gates the locality check."""
    betas, tsv, _, _ = _fixture(tmp_path, drop_col="repeat")
    out = tmp_path / "norep"
    assert (
        _run(
            [
                "-betas",
                str(betas),
                "-trials",
                str(tsv),
                "-factors",
                "stim,task",
                "-prefix",
                str(out),
                "-quiet",
            ]
        )
        == 0
    )
    assert Path(str(out) + ".nii.gz").exists()
