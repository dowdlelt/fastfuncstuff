"""End-to-end tests for the ffs_varpart CLI.

Covers the wiring the primitive tests cannot: sidecar parsing, mask/atlas handling,
sub-brick assembly, and the failure messages a user is most likely to hit.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

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
    assert "oneminusp_fwe_unique_stim" in rows[0]
    assert "oneminusp_unc_interaction" in rows[0]
    # Stored as 1 - p, so significance is the high end: p in (0, 1] -> value in [0, 1).
    val = float(rows[0]["oneminusp_fwe_unique_stim"])
    assert 0.0 <= val < 1.0
    # No raw-p sub-brick survives, or a threshold at 0.95 would silently mean p > 0.95.
    assert not any(k.startswith(("p_unc", "p_fwe")) for k in rows[0])
    meta = json.loads(Path(f"{out}_varpart.json").read_text())
    assert "1 - p" in meta["p_map_convention"]


def test_pvalues_are_stored_complemented(tmp_path):
    """The written map must be exactly 1 - the library's p, not a rescaling of it."""
    from fastfuncstuff.stats.variance_partition import permutation_test

    betas, tsv, shape, _ = _fixture(tmp_path)
    atlas = tmp_path / "atlas2.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.int16), np.eye(4)), atlas)
    out = tmp_path / "vpc"
    args = [
        "-betas", str(betas), "-trials", str(tsv), "-factors", "stim,task",
        "-atlas", str(atlas), "-perm", "16", "-seed", "5", "-prefix", str(out), "-quiet",
    ]  # fmt: skip
    assert _run(args) == 0

    rows_tbl = list(csv.reader(open(tsv, newline="")))
    header = rows_tbl[0]
    factors = {
        name: np.array([r[header.index(name)] for r in rows_tbl[1:]]) for name in ("stim", "task")
    }
    run = np.array([r[header.index("run")] for r in rows_tbl[1:]])
    rep = np.array([int(r[header.index("repeat")]) for r in rows_tbl[1:]])
    data = np.asanyarray(nib.load(str(betas)).dataobj)
    y = torch.as_tensor(data.reshape(-1, data.shape[3])[None].mean(axis=1), dtype=torch.float32)

    res = permutation_test(y, factors, repeat=rep, run=run, n_perms=16, seed=5, verbose=False)
    row = list(csv.DictReader(open(str(out) + "_roi.tsv"), delimiter="\t"))[0]
    for key, col in (
        ("unique_a", "oneminusp_unc_unique_stim"),
        ("interaction", "oneminusp_unc_interaction"),
    ):
        assert float(row[col]) == pytest.approx(1.0 - float(res.p_uncorrected[key][0]), abs=1e-5)


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


def _base_args(betas, tsv, out):
    return [
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


def test_drop_trials_excludes_matching_rows(tmp_path, capsys):
    betas, tsv, _, n_tr = _fixture(tmp_path)
    out = tmp_path / "dropped"
    assert _run([*_base_args(betas, tsv, out), "-drop_trials", "run", "r0"]) == 0
    txt = capsys.readouterr().out
    assert "-drop_trials run=r0" in txt
    meta = json.loads(Path(f"{out}_varpart.json").read_text())
    assert meta["n_trials_in_table"] == n_tr
    assert meta["n_trials"] < n_tr
    assert meta["dropped_trials"] == [["run", "r0"]]


def test_drop_trials_is_repeatable(tmp_path):
    betas, tsv, _, n_tr = _fixture(tmp_path)
    out = tmp_path / "dropped2"
    assert (
        _run(
            [
                *_base_args(betas, tsv, out),
                "-drop_trials",
                "run",
                "r0",
                "-drop_trials",
                "run",
                "r1",
            ]
        )
        == 0
    )
    meta = json.loads(Path(f"{out}_varpart.json").read_text())
    assert 0 < meta["n_trials"] < n_tr


def test_drop_trials_rejects_a_label_that_matches_nothing(tmp_path):
    """A silent no-op here means analysing trials the user believes were excluded."""
    betas, tsv, _, _ = _fixture(tmp_path)
    with pytest.raises(SystemExit, match="no trial has that value"):
        _run([*_base_args(betas, tsv, tmp_path / "x"), "-drop_trials", "run", "r99"])


def test_drop_trials_rejects_an_unknown_column(tmp_path):
    betas, tsv, _, _ = _fixture(tmp_path)
    with pytest.raises(SystemExit, match="column 'session' not found"):
        _run([*_base_args(betas, tsv, tmp_path / "x"), "-drop_trials", "session", "1"])


def test_beta_count_is_checked_against_the_undropped_table(tmp_path):
    betas, tsv, _, _ = _fixture(tmp_path)
    short = tmp_path / "short.csv"
    rows = list(csv.reader(open(tsv, newline="")))
    with open(short, "w", newline="") as fh:
        csv.writer(fh).writerows(rows[:-3])
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


def test_free_text_levels_are_sanitized_and_recorded(tmp_path):
    """Levels with spaces/punctuation become identifiers; the mapping is written out."""
    betas, tsv, _, _ = _fixture(tmp_path)
    messy = tmp_path / "messy.csv"
    rows = list(csv.reader(open(tsv, newline="")))
    header = rows[0]
    ti = header.index("task")
    for row in rows[1:]:
        row[ti] = row[ti].replace("t", "task, number ")
    with open(messy, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    out = tmp_path / "messy_out"
    assert _run(_base_args(betas, messy, out)) == 0
    meta = json.loads(Path(f"{out}_varpart.json").read_text())
    mapping = meta["level_names"]["task"]
    assert all(" " not in v and "," not in v for v in mapping.values())
    assert len(set(mapping.values())) == len(mapping)
    assert any(k != v for k, v in mapping.items())


def _messy_question_table(tmp_path, tsv):
    """Rewrite the task column into free text, including a whitespace variant."""
    rows = list(csv.reader(open(tsv, newline="")))
    header = rows[0]
    ti = header.index("task")
    out = tmp_path / "freetext.csv"
    for i, row in enumerate(rows[1:]):
        n = row[ti][1:]
        # Every third trial of a level gets a stray double space -- same level, typed twice.
        sep = "  " if i % 3 == 0 else " "
        row[ti] = f"Where{sep}is this shown {n}"
    with open(out, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return out


def test_drop_trials_matches_free_text_across_whitespace_variants(tmp_path, capsys):
    """The label as typed must drop the whole level, not just the exactly-spelled rows."""
    betas, tsv, _, _ = _fixture(tmp_path)
    messy = _messy_question_table(tmp_path, tsv)
    out = tmp_path / "ft"
    assert (
        _run([*_base_args(betas, messy, out), "-drop_trials", "task", "Where is this shown 0"]) == 0
    )
    txt = capsys.readouterr().out
    assert "matched 2 spellings" in txt
    rows = list(csv.DictReader(open(messy, newline="")))
    remaining = [r for r in rows if "shown 0" in r["task"]]
    assert remaining  # the level really was present in the table
    meta = json.loads(Path(f"{out}_varpart.json").read_text())
    assert meta["n_trials"] == len(rows) - len(remaining)


def test_drop_trials_accepts_the_sanitized_identifier(tmp_path):
    """The identifier from level_names / map names is a legitimate thing to paste back in."""
    betas, tsv, _, _ = _fixture(tmp_path)
    messy = _messy_question_table(tmp_path, tsv)
    out = tmp_path / "ident"
    assert (
        _run([*_base_args(betas, messy, out), "-drop_trials", "task", "Where_is_this_shown_0"]) == 0
    )
    rows = list(csv.DictReader(open(messy, newline="")))
    meta = json.loads(Path(f"{out}_varpart.json").read_text())
    assert meta["n_trials"] == len(rows) - sum("shown 0" in r["task"] for r in rows)
