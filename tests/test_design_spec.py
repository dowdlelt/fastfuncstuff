"""Tests for the design.spec TOML schema, contrast resolver, and CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fastfuncstuff.design.spec import (
    ContrastSpec,
    EventsColumns,
    EventSpec,
    MetaSpec,
    NuisanceSpec,
    RunSpec,
    Spec,
    load_spec,
    resolve_contrast,
    resolve_contrast_row,
    write_spec,
)


def _make_spec(tmp_path: Path) -> Spec:
    return Spec(
        meta=MetaSpec(
            runs=[
                RunSpec(bold="r01.nii.gz", events="r01_events.tsv"),
                RunSpec(bold="r02.nii.gz", events="r02_events.tsv"),
            ],
            tr=1.5,
            n_timepoints_per_run=[150, 150],
            polort=[2, 2],
            events_columns=EventsColumns(),
            drop_trial_types=["rest"],
        ),
        events=[
            EventSpec(
                trial_type="face",
                duration="from_events",
                hrf="SPMG1(0)",
                mode="condition",
                round_onset="TR",
            ),
            EventSpec(trial_type="house", duration=2.0, hrf="SPMG1(2)", mode="condition"),
        ],
        nuisance=[NuisanceSpec(file="motion.1D", label="motion", scope="full")],
        contrasts=[
            ContrastSpec(label="FaceVHouse", sym="SYM: +1*face -1*house", balance="zero"),
        ],
    )


def test_spec_round_trip(tmp_path):
    """write_spec → load_spec preserves every field."""
    spec_in = _make_spec(tmp_path)
    path = tmp_path / "design.spec"
    write_spec(spec_in, path, header_comment="test")

    spec_out = load_spec(path)
    assert spec_out.meta.tr == 1.5
    assert spec_out.meta.n_timepoints_per_run == [150, 150]
    assert spec_out.meta.polort == [2, 2]
    assert spec_out.meta.drop_trial_types == ["rest"]
    assert len(spec_out.events) == 2
    assert spec_out.events[0].trial_type == "face"
    assert spec_out.events[0].duration == "from_events"
    assert spec_out.events[0].round_onset == "TR"
    assert spec_out.events[1].duration == 2.0
    assert spec_out.nuisance[0].label == "motion"
    assert spec_out.contrasts[0].balance == "zero"


def test_load_spec_validates_enums(tmp_path):
    bad = tmp_path / "bad.spec"
    bad.write_text(
        "[meta]\n"
        "tr = 1.5\nn_timepoints_per_run=[100]\n"
        'runs=[{bold="x.nii.gz",events="x.tsv"}]\n'
        '[[events]]\ntrial_type="a"\nmode="weird"\n'
    )
    with pytest.raises(ValueError, match="mode must be"):
        load_spec(bad)


def test_resolve_simple():
    row = resolve_contrast_row("SYM: +1*A -1*B", ["A", "B", "C"])
    assert row == {"A": (1.0, None), "B": (-1.0, None)}


def test_resolve_glob_divides_weight():
    """+1**_instruct over 2 matches -> 0.5 each."""
    labels = ["face_instruct", "house_instruct", "face_trial"]
    row = resolve_contrast_row("SYM: +1**_instruct", labels)
    assert row["face_instruct"] == (0.5, None)
    assert row["house_instruct"] == (0.5, None)
    assert "face_trial" not in row


def test_resolve_allothers_excludes_named():
    """+0.5*A +0.5*B -ALLOTHERS over 4 labels -> -1/2 each on C and D."""
    labels = ["A", "B", "C", "D"]
    row = resolve_contrast_row("SYM: +0.5*A +0.5*B -ALLOTHERS", labels)
    assert row["A"] == (0.5, None)
    assert row["B"] == (0.5, None)
    assert row["C"] == (-0.5, None)
    assert row["D"] == (-0.5, None)


def test_resolve_balance_zero():
    """balance='zero' shifts weights so they sum to zero."""
    labels = ["A", "B", "C"]
    row = resolve_contrast_row("SYM: +1*A", labels, balance="zero")
    # A=+1 alone, mean=1/1=1, but resolved dict has only A → shift to 0.
    # That's degenerate. Use a richer case:
    row = resolve_contrast_row("SYM: +1*A +1*B +1*C", labels, balance="zero")
    total = sum(w for w, _ in row.values())
    assert abs(total) < 1e-9


def test_resolve_balance_sum1():
    labels = ["A", "B", "C"]
    row = resolve_contrast_row("SYM: +2*A +2*B", labels, balance="sum1")
    total = sum(w for w, _ in row.values())
    assert abs(total - 1.0) < 1e-9


def test_resolve_f_test_via_contrastspec():
    """ContrastSpec with list[str] sym resolves to multiple rows."""
    cs = ContrastSpec(label="ABC_F", sym=["SYM: +1*A", "SYM: +1*B", "SYM: +1*C"])
    rows = resolve_contrast(cs, ["A", "B", "C"])
    assert len(rows) == 3
    assert rows[0] == {"A": (1.0, None)}
    assert rows[2] == {"C": (1.0, None)}


def test_resolve_missing_label_raises():
    with pytest.raises(ValueError, match="not in stim labels"):
        resolve_contrast_row("SYM: +1*Z", ["A", "B"])


def test_resolve_allothers_when_all_named_raises():
    with pytest.raises(ValueError, match="every stim label was already named"):
        resolve_contrast_row("SYM: +1*A +1*B -ALLOTHERS", ["A", "B"])


# ---------------------------------------------------------------------------
# Compile end-to-end (uses real builder; no GPU required)
# ---------------------------------------------------------------------------


def _write_events_tsv(path: Path, rows: list[tuple[float, float, str]]) -> None:
    with open(path, "w") as fh:
        fh.write("onset\tduration\ttrial_type\n")
        for onset, duration, tt in rows:
            fh.write(f"{onset}\t{duration}\t{tt}\n")


def test_compile_end_to_end(tmp_path):
    """A complete spec → compile → readable xmat with the expected columns
    and a resolved contrast."""
    from fastfuncstuff.cli.design_spec import _do_compile
    from fastfuncstuff.io.afni import read_afni_design_matrix

    # Two runs of 60 TRs at TR=2 (120s each).
    n_tp = 60
    tr = 2.0
    events_a = tmp_path / "r01_events.tsv"
    events_b = tmp_path / "r02_events.tsv"
    _write_events_tsv(
        events_a,
        [
            (10.0, 2.0, "face"),
            (40.0, 2.0, "face"),
            (20.0, 2.0, "house"),
            (80.0, 2.0, "house"),
        ],
    )
    _write_events_tsv(
        events_b,
        [
            (5.0, 2.0, "face"),
            (90.0, 2.0, "face"),
            (30.0, 2.0, "house"),
        ],
    )

    spec = Spec(
        meta=MetaSpec(
            runs=[
                RunSpec(bold="unused.nii.gz", events=str(events_a)),
                RunSpec(bold="unused.nii.gz", events=str(events_b)),
            ],
            tr=tr,
            n_timepoints_per_run=[n_tp, n_tp],
            polort=2,
        ),
        events=[
            EventSpec(trial_type="face", duration=2.0, hrf="SPMG1(2)", mode="condition"),
            EventSpec(trial_type="house", duration=2.0, hrf="SPMG1(2)", mode="condition"),
        ],
        contrasts=[
            ContrastSpec(label="F_vs_H", sym="SYM: +1*face -1*house"),
        ],
    )
    spec_path = tmp_path / "design.spec"
    write_spec(spec, spec_path)

    xmat_path = tmp_path / "X.xmat.1D"
    args = type(
        "A", (), {"spec": str(spec_path), "xmat": str(xmat_path), "verb": 0, "overwrite": True}
    )()
    rc = _do_compile(args)
    assert rc == 0
    assert xmat_path.exists()

    info = read_afni_design_matrix(str(xmat_path))
    labels = info["column_labels"]
    # Two stim labels + polort (per-run scaled) + run starts implicit.
    assert "face#0" in labels
    assert "house#0" in labels

    # Contrast resolved correctly.
    assert info["n_glt"] == 1
    assert info["glt_labels"] == ["F_vs_H"]
    glt = info["glt_matrices"][0]
    assert glt.shape[0] == 1  # t-test
    face_col = labels.index("face#0")
    house_col = labels.index("house#0")
    assert glt[0, face_col] == 1.0
    assert glt[0, house_col] == -1.0


def test_stub_nuisance_flags_match_ffs_reml():
    """`-ortvec` / `-ortvec_run` / `-ortvec_glob` populate [[nuisance]]."""
    from argparse import Namespace

    from fastfuncstuff.cli.design_spec import _build_nuisance_from_cli_args

    args = Namespace(
        ortvec=[("physio.1D", "physio")],
        ortvec_run=[("motion_r02.1D", "motion", "2")],
        ortvec_glob=[("motion_r0*.1D", "motion")],
        ortvec_concat=None,
    )
    n = _build_nuisance_from_cli_args(args, n_runs=3)

    assert len(n) == 3
    assert n[0].file == "physio.1D" and n[0].scope == "full"
    assert n[1].file == "motion_r02.1D" and n[1].scope == "run:2"
    assert n[2].scope == "glob" and n[2].pattern == "motion_r0*.1D"
    assert n[2].file is None and n[2].label == "motion"


def test_ortvec_concat_expands_to_full_length_entries(tmp_path):
    """-ortvec_concat expands a glob over N already-padded files into N
    scope='full' entries with auto-suffixed labels ordered by run index."""
    from argparse import Namespace

    from fastfuncstuff.cli.design_spec import _build_nuisance_from_cli_args

    for i in (1, 2, 3):
        (tmp_path / f"mot_demean.r{i:02d}.1D").write_text("0\n")

    args = Namespace(
        ortvec=None,
        ortvec_run=None,
        ortvec_glob=None,
        ortvec_concat=[(str(tmp_path / "mot_demean.r*.1D"), "motion")],
    )
    n = _build_nuisance_from_cli_args(args, n_runs=3)

    assert len(n) == 3
    assert [x.label for x in n] == ["motion01", "motion02", "motion03"]
    assert all(x.scope == "full" for x in n)
    # File paths track the run index (alphabetical glob happens to align here,
    # but the helper sorts by *inferred* run number).
    assert n[0].file.endswith("r01.1D")
    assert n[2].file.endswith("r03.1D")


def test_nuisance_glob_resolves_to_padortvec_at_compile(tmp_path):
    """scope='glob' expands to N padortvec entries using run-index inference."""
    from fastfuncstuff.cli.design_spec import _resolve_nuisance_for_compile
    from fastfuncstuff.design.spec import NuisanceSpec

    # Three per-run files, each 20 rows long.
    for i in (1, 2, 3):
        (tmp_path / f"motion_run-{i:02d}.1D").write_text(
            "\n".join("0.1 0.2 0.3" for _ in range(20)) + "\n"
        )
    nuisance = [
        NuisanceSpec(
            file=None,
            label="motion",
            scope="glob",
            pattern=str(tmp_path / "motion_run-*.1D"),
        )
    ]

    ortvec, padortvec = _resolve_nuisance_for_compile(
        nuisance,
        n_runs=3,
        n_timepoints_per_run=[20, 20, 20],
    )
    assert ortvec == []
    assert len(padortvec) == 3
    runs_assigned = sorted(r for _, _, r in padortvec)
    assert runs_assigned == [1, 2, 3]


def test_nuisance_glob_rejects_full_length_files(tmp_path):
    """Glob mode expects one-run-length files; full-length crashes early."""
    from fastfuncstuff.cli.design_spec import _resolve_nuisance_for_compile
    from fastfuncstuff.design.spec import NuisanceSpec

    # 60 rows = full length (3 runs × 20). Glob mode should refuse this and
    # tell the user to use scope='full' instead.
    for i in (1, 2, 3):
        (tmp_path / f"motion_run-{i:02d}.1D").write_text("\n".join("0.1" for _ in range(60)) + "\n")
    nuisance = [
        NuisanceSpec(
            file=None,
            label="motion",
            scope="glob",
            pattern=str(tmp_path / "motion_run-*.1D"),
        )
    ]
    with pytest.raises(ValueError, match="one-run-length"):
        _resolve_nuisance_for_compile(
            nuisance,
            n_runs=3,
            n_timepoints_per_run=[20, 20, 20],
        )


def test_bare_hrf_compiles_with_duration_from_events(tmp_path):
    """Bare ``SPMG1`` HRF picks up duration from the events spec."""
    from fastfuncstuff.cli.design_spec import _do_compile, _inject_duration
    from fastfuncstuff.io.afni import read_afni_design_matrix

    assert _inject_duration("SPMG1", 5.0) == "SPMG1(5)"
    assert _inject_duration("BLOCK", 10.0) == "BLOCK(10,1)"
    assert _inject_duration("SPMG1(3)", 99.0) == "SPMG1(3)"  # explicit wins

    ev = tmp_path / "r01.tsv"
    _write_events_tsv(ev, [(10.0, 3.0, "task"), (40.0, 3.0, "task")])
    spec = Spec(
        meta=MetaSpec(
            runs=[RunSpec(bold="x.nii.gz", events=str(ev))],
            tr=2.0,
            n_timepoints_per_run=[60],
            polort=2,
        ),
        events=[EventSpec(trial_type="task", duration=3.0, hrf="SPMG1")],
    )
    spec_path = tmp_path / "design.toml"
    write_spec(spec, spec_path)
    xmat = tmp_path / "X.xmat.1D"
    args = type(
        "A", (), {"spec": str(spec_path), "xmat": str(xmat), "verb": 0, "overwrite": True}
    )()
    assert _do_compile(args) == 0
    info = read_afni_design_matrix(str(xmat))
    assert "task#0" in info["column_labels"]


def test_stub_writes_examples_and_duration_stats(tmp_path):
    """Stub output carries duration-stats comments and example contrasts."""
    from argparse import Namespace

    from fastfuncstuff.cli.design_spec import _do_stub

    ev = tmp_path / "r01.tsv"
    _write_events_tsv(
        ev,
        [
            (5.0, 2.0, "stimA"),
            (15.0, 2.5, "stimA"),
            (25.0, 2.0, "stimA"),
            (35.0, 4.0, "stimB"),
        ],
    )
    # Tiny synthetic NIfTI for the header scan.
    import nibabel as nib

    nii = nib.Nifti1Image(np.zeros((2, 2, 2, 60), dtype=np.float32), np.eye(4))
    nii.header.set_zooms((1.0, 1.0, 1.0, 2.0))
    bold = tmp_path / "r01.nii.gz"
    nib.save(nii, bold)

    out_prefix = tmp_path / "stub_out"  # no extension; .toml should be appended
    args = Namespace(
        input=[str(bold)],
        events=[str(ev)],
        out=str(out_prefix),
        TR=None,
        event_cols=None,
        drop_trial_types=["rest", "Rest", "REST", "baseline"],
        default_hrf="SPMG1",
        ortvec=None,
        ortvec_run=None,
        ortvec_glob=None,
        ortvec_concat=None,
        overwrite=True,
    )
    rc = _do_stub(args)
    assert rc == 0

    written = tmp_path / "stub_out.toml"
    assert written.exists(), "Stub should auto-append .toml when missing"

    text = written.read_text()
    # duration stats appear as comments
    assert "observed durations" in text
    assert "stimA" in text and "stimB" in text
    # example contrasts present (commented)
    assert "# [[contrasts]]" in text
    assert "stimA_vs_stimB" in text
    # default HRF is bare
    assert 'hrf = "SPMG1"' in text


def test_compile_auto_appends_toml(tmp_path):
    """compile -spec design (no extension) finds design.toml."""
    from fastfuncstuff.cli.design_spec import _resolve_spec_path

    (tmp_path / "design.toml").write_text("dummy")
    resolved = _resolve_spec_path(str(tmp_path / "design"))
    assert resolved.name == "design.toml"

    # Direct .toml path also works.
    resolved2 = _resolve_spec_path(str(tmp_path / "design.toml"))
    assert resolved2.name == "design.toml"

    # Nonexistent → FileNotFoundError mentioning both attempts.
    with pytest.raises(FileNotFoundError, match=r"nope.*also tried.*nope\.toml"):
        _resolve_spec_path(str(tmp_path / "nope"))


def test_compile_refuses_to_overwrite_xmat(tmp_path):
    """Existing -xmat blocks compile in non-interactive mode without -overwrite."""
    from fastfuncstuff.cli.design_spec import _do_compile

    ev = tmp_path / "r01.tsv"
    _write_events_tsv(ev, [(10.0, 2.0, "task")])
    spec = Spec(
        meta=MetaSpec(
            runs=[RunSpec(bold="x.nii.gz", events=str(ev))],
            tr=2.0,
            n_timepoints_per_run=[60],
            polort=2,
        ),
        events=[EventSpec(trial_type="task", duration=2.0, hrf="SPMG1")],
    )
    spec_path = tmp_path / "design.toml"
    write_spec(spec, spec_path)

    xmat = tmp_path / "stale.xmat.1D"
    xmat.write_text("this is stale content\n")

    args = type(
        "A",
        (),
        {
            "spec": str(spec_path),
            "xmat": str(xmat),
            "verb": 0,
            "overwrite": False,
        },
    )()
    rc = _do_compile(args)
    assert rc == 1, "Should refuse without -overwrite"
    # Stale content untouched.
    assert xmat.read_text() == "this is stale content\n"


def test_compile_reorders_columns_to_afni_layout(tmp_path):
    """xmat columns come out polort → stim → nuisance, matching AFNI."""
    from fastfuncstuff.cli.design_spec import _do_compile
    from fastfuncstuff.io.afni import read_afni_design_matrix

    ev = tmp_path / "r01.tsv"
    _write_events_tsv(ev, [(10.0, 2.0, "task")])
    # 60-row full-length motion file (1 column).
    (tmp_path / "motion.1D").write_text("\n".join("0.1" for _ in range(60)) + "\n")

    spec = Spec(
        meta=MetaSpec(
            runs=[RunSpec(bold="x.nii.gz", events=str(ev))],
            tr=2.0,
            n_timepoints_per_run=[60],
            polort=2,
        ),
        events=[EventSpec(trial_type="task", duration=2.0, hrf="SPMG1")],
        nuisance=[
            NuisanceSpec(
                file=str(tmp_path / "motion.1D"),
                label="mot",
                scope="full",
            )
        ],
    )
    spec_path = tmp_path / "design.toml"
    write_spec(spec, spec_path)
    xmat = tmp_path / "X.xmat.1D"
    args = type(
        "A", (), {"spec": str(spec_path), "xmat": str(xmat), "verb": 0, "overwrite": True}
    )()
    assert _do_compile(args) == 0

    info = read_afni_design_matrix(str(xmat))
    labels = info["column_labels"]
    # polort first (3 cols for polort=2), then stim, then nuisance.
    polort_end = next(i for i, l in enumerate(labels) if not l.startswith("Run#"))
    task_col = labels.index("task#0")
    mot_col = labels.index("mot")
    assert task_col == polort_end, "stim should come right after polort"
    assert mot_col > task_col, "nuisance should come after stim"


def test_demean_rescale_subtracts_column_mean(tmp_path):
    """rescale='demean' preprocesses the nuisance file before the builder
    sees it; the resulting design columns sum to ~0."""
    from fastfuncstuff.cli.design_spec import _do_compile
    from fastfuncstuff.io.afni import read_afni_design_matrix

    ev = tmp_path / "r01.tsv"
    _write_events_tsv(ev, [(10.0, 2.0, "task")])
    # Non-zero-mean nuisance: 5.0 throughout. Demean → 0.0 throughout.
    (tmp_path / "motion.1D").write_text("\n".join("5.0" for _ in range(60)) + "\n")

    spec = Spec(
        meta=MetaSpec(
            runs=[RunSpec(bold="x.nii.gz", events=str(ev))],
            tr=2.0,
            n_timepoints_per_run=[60],
            polort=2,
        ),
        events=[EventSpec(trial_type="task", duration=2.0, hrf="SPMG1")],
        nuisance=[
            NuisanceSpec(
                file=str(tmp_path / "motion.1D"),
                label="mot",
                scope="full",
                rescale="demean",
            )
        ],
    )
    spec_path = tmp_path / "design.toml"
    write_spec(spec, spec_path)
    xmat = tmp_path / "X.xmat.1D"
    args = type(
        "A", (), {"spec": str(spec_path), "xmat": str(xmat), "verb": 0, "overwrite": True}
    )()
    assert _do_compile(args) == 0

    info = read_afni_design_matrix(str(xmat))
    mat = info["matrix"]
    mot_idx = info["column_labels"].index("mot")
    assert abs(mat[:, mot_idx].sum()) < 1e-6, "demeaned column should sum to zero"


def test_demean_round_trips_through_spec(tmp_path):
    """rescale field survives write_spec → load_spec."""
    spec = Spec(
        meta=MetaSpec(
            runs=[RunSpec(bold="x.nii.gz", events="x.tsv")],
            tr=2.0,
            n_timepoints_per_run=[10],
            polort=2,
        ),
        events=[],
        nuisance=[
            NuisanceSpec(file="a.1D", label="a", scope="full"),
            NuisanceSpec(file="b.1D", label="b", scope="full", rescale="demean"),
        ],
    )
    path = tmp_path / "design.toml"
    write_spec(spec, path)
    loaded = load_spec(path)
    assert loaded.nuisance[0].rescale == "as-is"
    assert loaded.nuisance[1].rescale == "demean"


def test_compile_refuses_hrfopt(tmp_path):
    """hrfopt:<lib> models can't be compiled to a single xmat."""
    from fastfuncstuff.cli.design_spec import _do_compile

    events = tmp_path / "r01.tsv"
    _write_events_tsv(events, [(10.0, 2.0, "task")])
    spec = Spec(
        meta=MetaSpec(
            runs=[RunSpec(bold="x.nii.gz", events=str(events))],
            tr=2.0,
            n_timepoints_per_run=[60],
            polort=2,
        ),
        events=[EventSpec(trial_type="task", hrf="hrfopt:lib.tsv")],
    )
    spec_path = tmp_path / "design.spec"
    write_spec(spec, spec_path)
    args = type(
        "A",
        (),
        {"spec": str(spec_path), "xmat": str(tmp_path / "X.xmat.1D"), "verb": 0, "overwrite": True},
    )()
    with pytest.raises(ValueError, match="hrfopt"):
        _do_compile(args)
