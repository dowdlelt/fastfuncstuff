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

from fastfuncstuff.cli.varpart import CEILING_FLOOR, _summarize_effects, main  # noqa: E402


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


def test_summary_pools_explained_sums_of_squares():
    rows = _summarize_effects(
        {"unique_task": torch.tensor([-0.2, 0.8])},
        heldout_sst=torch.tensor([1.0, 3.0]),
        noise_ceiling=torch.tensor([0.5, 1.0]),
    )

    row = rows[0]
    assert row["pooled_cv_r2"] == pytest.approx(0.55)
    assert row["pooled_frac_ceiling"] == pytest.approx(2.2 / 3.5)
    assert row["median_frac_ceiling"] == pytest.approx(0.2)
    assert row["q25_frac_ceiling"] == pytest.approx(-0.1)
    assert row["q75_frac_ceiling"] == pytest.approx(0.5)
    assert row["positive_frac_reliable"] == pytest.approx(0.5)
    assert row["n_units"] == row["n_reliable"] == 2


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
    assert img.shape[3] == 23
    assert (tmp_path / "vp_varpart.json").exists()
    summary = list(csv.DictReader(open(tmp_path / "vp_summary.tsv"), delimiter="\t"))
    assert [row["effect"] for row in summary] == [
        "unique_stim",
        "unique_task",
        "interaction",
        "shared",
        "r2_full",
    ]
    # Dominance is a property of the competing partition pieces only; shared and r2_full
    # are not competitors and must stay blank rather than reading as "never wins".
    by_effect = {row["effect"]: row for row in summary}
    assert by_effect["shared"]["n_dominant"] == ""
    assert by_effect["r2_full"]["frac_dominant"] == ""
    dominant = [
        int(by_effect[e]["n_dominant"]) for e in ("unique_stim", "unique_task", "interaction")
    ]
    assert sum(dominant) <= int(summary[0]["n_units"])

    import json

    meta = json.loads((tmp_path / "vp_varpart.json").read_text())
    assert meta["unit"] == "voxel"
    assert meta["nested_gamma"] is True
    assert meta["factors"] == ["stim", "task"]
    # Balanced crossed design: shared variance must come out ~0.
    assert abs(meta["diagnostics"]["shared_abs_median"]) < 0.02


def test_atlas_mode_writes_roi_table_and_summary(tmp_path, capsys):
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

    terminal = capsys.readouterr().out
    assert "Overall variance summary (parcels)" in terminal
    assert "unique_stim" in terminal

    summary = list(csv.DictReader(open(str(out) + "_summary.tsv"), delimiter="\t"))
    assert len(summary) == 5
    assert all(int(row["n_units"]) == 4 for row in summary)
    assert "partition check:" in terminal

    rows = list(csv.DictReader(open(str(out) + "_roi.tsv"), delimiter="\t"))
    assert len(rows) == 4
    assert {"roi", "unique_stim", "unique_task", "interaction", "rank_E", "ncsnr"} <= set(rows[0])

    # ROI mode also paints a volume for figures: same grid and sub-bricks as voxel mode,
    # constant within each parcel.
    img = nib.load(str(out) + ".nii.gz")
    assert img.shape == (*shape, 23)
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


def test_significance_summary_counts_parcels(tmp_path, capsys):
    betas, tsv, shape, _ = _fixture(tmp_path, noise=0.5)
    atlas = tmp_path / "atlas.nii.gz"
    lab = (np.arange(int(np.prod(shape))) % 4 + 1).reshape(shape).astype(np.int16)
    nib.save(nib.Nifti1Image(lab, np.eye(4)), atlas)

    out = tmp_path / "vpsig"
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
                "40",
                "-prefix",
                str(out),
                "-quiet",
            ]
        )
        == 0
    )
    terminal = capsys.readouterr().out
    assert "Significance (40 permutations, 4 parcels)" in terminal

    sig = list(csv.DictReader(open(str(out) + "_significance.tsv"), delimiter="\t"))
    assert {r["effect"] for r in sig} == {"unique_stim", "unique_task", "interaction"}
    for row in sig:
        assert int(row["n_units"]) == 4
        # FWE correction can only remove units, never add them.
        assert int(row["n_sig_fwe_p05"]) <= int(row["n_sig_unc_p05"]) <= 4
        assert int(row["n_sig_fwe_p01"]) <= int(row["n_sig_fwe_p05"])
        # 1/(N+1) is the smallest reachable p; the maps are float32, hence the slack.
        assert float(row["min_p_fwe"]) >= 1.0 / 41 - 1e-6

    # The per-parcel listing must agree with the counts, ROI ids and all.
    listed = list(csv.DictReader(open(str(out) + "_significant_rois.tsv"), delimiter="\t"))
    for row in sig:
        hits = [r for r in listed if r["effect"] == row["effect"]]
        assert len(hits) == int(row["n_sig_fwe_p05"])
        assert all(float(r["p_fwe"]) < 0.05 for r in hits)
        assert all(int(r["roi"]) in (1, 2, 3, 4) for r in hits)
        # Strongest first, so the top of the list is what a figure caption quotes.
        assert [float(r["value"]) for r in hits] == sorted(
            (float(r["value"]) for r in hits), reverse=True
        )

    meta = json.loads(Path(f"{out}_varpart.json").read_text())
    counts = {r["effect"]: r for r in meta["significance"]["per_effect"]}
    assert counts["unique_stim"]["n_sig_fwe_p05"] == len(
        meta["significance"]["significant_rois"]["unique_stim"]
    )


def test_too_few_permutations_is_called_out(tmp_path, capsys):
    betas, tsv, shape, _ = _fixture(tmp_path)
    atlas = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.int16), np.eye(4)), atlas)
    out = tmp_path / "vpfew"
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
                "5",
                "-prefix",
                str(out),
                "-quiet",
            ]
        )
        == 0
    )
    # 5 permutations floor p at 1/6: a table of zeros there means "too few permutations",
    # not "no effect", and the tool has to say which.
    assert "nothing can reach p < 0.05" in capsys.readouterr().out


def test_pvalues_are_stored_complemented(tmp_path):
    """A real effect must score HIGH and an absent one LOW -- the raw-p convention inverts
    both, so this ordering is what actually distinguishes the two conventions.

    Comparing against a recomputed p-value cannot do that job: with band shrinkage an
    absent effect gives a statistic of exactly zero, so its p sits on a knife-edge of ties
    and moves by a whole 1/(P+1) step under float noise.
    """
    betas, tsv, shape, _ = _fixture(tmp_path, noise=0.5)
    atlas = tmp_path / "atlas_p.nii.gz"
    lab = (np.arange(int(np.prod(shape))) % 4 + 1).reshape(shape).astype(np.int16)
    nib.save(nib.Nifti1Image(lab, np.eye(4)), atlas)

    out = tmp_path / "vpc"
    assert (
        _run([*_base_args(betas, tsv, out), "-atlas", str(atlas), "-perm", "40", "-seed", "5"]) == 0
    )

    rows = list(csv.DictReader(open(str(out) + "_roi.tsv"), delimiter="\t"))
    for row in rows:
        real = float(row["oneminusp_unc_unique_stim"])
        absent = float(row["oneminusp_unc_interaction"])
        assert real > 0.9, "a real main effect must land near 1 under the 1-p convention"
        assert real > absent
        # Every value sits on the (n_perms + 1) grid of achievable p-values.
        assert (1.0 - real) * 41 == pytest.approx(round((1.0 - real) * 41), abs=1e-3)


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


def test_requires_at_least_two_factors(tmp_path):
    betas, tsv, _, _ = _fixture(tmp_path)
    with pytest.raises(SystemExit, match="at least 2 column names"):
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
    assert (
        _run([*_base_args(betas, tsv, out), "-drop_trials", "run", "r0", "-no_nested_gamma"]) == 0
    )
    txt = capsys.readouterr().out
    assert "-drop_trials run=r0" in txt
    meta = json.loads(Path(f"{out}_varpart.json").read_text())
    assert meta["n_trials_in_table"] == n_tr
    assert meta["n_trials"] < n_tr
    assert meta["dropped_trials"] == [["run", "r0"]]
    assert meta["nested_gamma"] is False


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
                "-no_nested_gamma",
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


def test_frac_ceiling_maps_are_the_ratio_to_the_noise_ceiling(tmp_path):
    """R2 alone cannot say "is this good"; divided by the ceiling it can."""
    betas, tsv, shape, _ = _fixture(tmp_path)
    atlas = tmp_path / "atlas3.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.int16), np.eye(4)), atlas)
    out = tmp_path / "vpf"
    assert _run([*_base_args(betas, tsv, out), "-atlas", str(atlas)]) == 0

    row = list(csv.DictReader(open(str(out) + "_roi.tsv"), delimiter="\t"))[0]
    for src in ("unique_stim", "unique_task", "interaction", "r2_full"):
        assert f"{src}_frac_ceiling" in row
    ceiling = float(row["noise_ceiling"])
    assert ceiling > 0.01  # the fixture has real signal, so the ratio is defined
    for src in ("unique_stim", "interaction", "r2_full"):
        expected = float(row[src]) / ceiling
        assert float(row[f"{src}_frac_ceiling"]) == pytest.approx(expected, rel=1e-4)


def test_frac_ceiling_is_zero_where_nothing_is_obtainable(tmp_path):
    """Pure noise gives a ~0 ceiling; the ratio must not paint noise/noise over it."""
    rng = np.random.default_rng(0)
    factors, rep, run = make_crossed_table()
    n_tr = len(rep)
    shape = (4, 4, 3)
    n_vox = int(np.prod(shape))
    y = rng.normal(0, 1.0, size=(n_vox, n_tr))  # no signal at all
    betas = tmp_path / "noise.nii.gz"
    nib.save(nib.Nifti1Image(y.reshape(*shape, n_tr).astype(np.float32), np.eye(4)), betas)

    tsv = tmp_path / "noise_trials.csv"
    with open(tsv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stim", "task", "run", "repeat"])
        for i in range(n_tr):
            w.writerow([f"s{factors['stim'][i]}", f"t{factors['task'][i]}", f"r{run[i]}", rep[i]])

    # One ROI per voxel keeps the named-column table while leaving the units untouched.
    atlas = tmp_path / "atlas_noise.nii.gz"
    lab = (np.arange(n_vox) + 1).reshape(shape).astype(np.int16)
    nib.save(nib.Nifti1Image(lab, np.eye(4)), atlas)

    out = tmp_path / "vpn"
    assert _run([*_base_args(betas, tsv, out), "-atlas", str(atlas)]) == 0

    rows = list(csv.DictReader(open(str(out) + "_roi.tsv"), delimiter="\t"))
    dead = [r for r in rows if float(r["noise_ceiling"]) <= CEILING_FLOOR]
    assert dead, "expected pure noise to collapse the ceiling somewhere"
    for r in dead:
        for src in ("unique_stim", "interaction", "r2_full"):
            assert float(r[f"{src}_frac_ceiling"]) == 0.0


def test_anova_writes_its_own_table_and_subbricks(tmp_path, capsys):
    """-anova is an independent second opinion, so it must land in its own file.

    The fixture is purely additive, which makes the interaction row the informative one:
    its in-sample eta2 is large (380 df) while omega2 collapses to zero. If a future change
    ever reported eta2 alone, this catches it.
    """
    betas, tsv, _, _ = _fixture(tmp_path)
    out = tmp_path / "vp"
    assert _run(_base_args(betas, tsv, out) + ["-anova", "-device", "cpu"]) == 0

    rows = list(csv.DictReader(open(str(out) + "_anova.tsv"), delimiter="\t"))
    by_effect = {r["effect"]: r for r in rows}
    assert set(by_effect) == {"stim", "task", "stim:task", "r2_full"}
    assert int(by_effect["stim:task"]["df"]) == (N_STIM - 1) * (N_TASK - 1)

    inter = by_effect["stim:task"]
    assert float(inter["pooled_eta2"]) > 0.05  # df inflation, not signal
    assert abs(float(inter["pooled_omega2"])) < 0.02
    assert float(inter["pooled_eta2"]) == pytest.approx(
        float(inter["eta2_null_expected"]), abs=0.02
    )
    for factor in ("stim", "task"):
        assert float(by_effect[factor]["pooled_omega2"]) > 0.1

    img = nib.load(str(out) + ".nii.gz")
    meta = json.load(open(str(out) + "_varpart.json"))
    assert meta["anova"]["diagnostics"]["saturated"] is True
    assert img.shape[-1] > 0
    assert "Classical ANOVA" in capsys.readouterr().out


def test_anova_is_off_by_default(tmp_path):
    betas, tsv, _, _ = _fixture(tmp_path)
    out = tmp_path / "vp"
    assert _run(_base_args(betas, tsv, out) + ["-device", "cpu"]) == 0
    assert not (tmp_path / "vp_anova.tsv").exists()
    assert "anova" not in json.load(open(str(out) + "_varpart.json"))
