"""Scanner tests: entity parsing, sidecar inheritance, and the real-world
quirks ffs_autoproc must survive (non-contiguous runs, missing IntendedFor,
mislabeled phase, dataset-root events/json)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastfuncstuff.autoproc.bids import (
    find_events,
    pair_undetermined,
    parse_entities,
    parse_suffix,
    scan_subject,
)


def _touch(p: Path, sidecar: dict | None = None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    if sidecar is not None:
        p.with_name(p.name.replace(".nii.gz", ".json")).write_text(json.dumps(sidecar))


def test_parse_entities_and_suffix():
    n = "sub-001_ses-01_task-primary_run-03_part-mag_bold.nii.gz"
    ents = parse_entities(n)
    assert ents["sub"] == "001" and ents["task"] == "primary"
    assert ents["run"] == "03" and ents["part"] == "mag"
    assert parse_suffix(n) == "bold"


def test_noncontiguous_runs_and_root_sidecar(tmp_path: Path):
    """ds001555 shape: dataset-root task json (TR/PED inherited), gapped runs."""
    (tmp_path / "task-Foo_bold.json").write_text(
        json.dumps({"RepetitionTime": 2.0, "PhaseEncodingDirection": "i", "SliceTiming": [0, 1]})
    )
    func = tmp_path / "sub-001" / "ses-01" / "func"
    for run in ("003", "004", "006", "011"):  # non-contiguous
        _touch(func / f"sub-001_ses-01_task-Foo_run-{run}_bold.nii.gz")

    subj = scan_subject(tmp_path, "001")
    assert len(subj.sessions) == 1
    runs = subj.sessions[0].bold_runs
    assert [r.run for r in runs] == ["003", "004", "006", "011"]
    # TR/PED/SliceTiming inherited from the dataset-root sidecar.
    assert runs[0].tr == 2.0
    assert runs[0].pe_dir == "i"
    assert runs[0].has_slice_timing
    # TaskName absent → derived from entity.
    assert runs[0].task_name == "Foo"


def test_int_and_prefixed_subject_ids(tmp_path: Path):
    func = tmp_path / "sub-5" / "func"
    _touch(func / "sub-5_task-Foo_run-01_bold.nii.gz", {"RepetitionTime": 1.0})
    assert scan_subject(tmp_path, "5").subject == "5"
    assert scan_subject(tmp_path, "sub-5").subject == "5"


def test_sbref_and_phase_pairing_and_mislabeled_phase(tmp_path: Path):
    func = tmp_path / "sub-ME1" / "ses-SM" / "func"
    base = "sub-ME1_ses-SM_task-primary_run-01"
    _touch(func / f"{base}_part-mag_bold.nii.gz", {"RepetitionTime": 2.5})
    _touch(func / f"{base}_part-phase_bold.nii.gz", {"RepetitionTime": 2.5})
    _touch(func / f"{base}_part-mag_sbref.nii.gz", {"RepetitionTime": 2.5})
    # quirk: an sbref file that is actually PHASE (ImageType), must NOT be the sbref.
    _touch(func / f"{base}_part-sbref_bold.nii.gz", {"ImageType": ["P", "PHASE"]})

    subj = scan_subject(tmp_path, "ME1")
    run = subj.sessions[0].bold_runs[0]
    assert run.phase_path is not None and "part-phase" in run.phase_path.name
    assert run.sbref_path is not None and run.sbref_path.name.endswith("part-mag_sbref.nii.gz")


def test_bare_siemens_p_image_type_is_phase(tmp_path: Path):
    # HiHiFaces' older conversions mark phase only with the Siemens "P" token (no
    # "PHASE"), so their _sbref_ph / _phase fmaps were taken for magnitude candidates.
    from fastfuncstuff.autoproc.bids import _is_phase

    fmap = tmp_path / "sub-1" / "fmap"
    phase = fmap / "sub-1_dir-LR_run-1_sbref_ph.nii.gz"
    mag = fmap / "sub-1_dir-LR_run-1_sbref.nii.gz"
    _touch(phase, {"ImageType": ["ORIGINAL", "PRIMARY", "P", "ND", "MOSAIC"]})
    _touch(mag, {"ImageType": ["ORIGINAL", "PRIMARY", "M", "ND", "MOSAIC"]})
    assert _is_phase(phase, parse_entities(phase.name), tmp_path)
    assert not _is_phase(mag, parse_entities(mag.name), tmp_path)  # "PRIMARY" is not "P"


def test_fmap_intendedfor_and_task_fallback(tmp_path: Path):
    """floc fmap has IntendedFor (floc runs only); primary fmap has none →
    task-match. Scanning -task primary must keep only the primary fmap."""
    ses = tmp_path / "sub-ME1" / "ses-SM"
    for task in ("floc", "primary"):
        for run in ("01", "02"):
            _touch(
                ses / "func" / f"sub-ME1_ses-SM_task-{task}_run-{run}_part-mag_bold.nii.gz",
                {"RepetitionTime": 2.0, "PhaseEncodingDirection": "j-"},
            )
    # floc reverse fmap WITH IntendedFor pointing at floc runs.
    _touch(
        ses / "fmap" / "sub-ME1_ses-SM_task-floc_dir-PA_part-mag_sbref.nii.gz",
        {
            "PhaseEncodingDirection": "j",
            "TotalReadoutTime": 0.06,
            "IntendedFor": [
                "bids::/sub-ME1/ses-SM/func/sub-ME1_ses-SM_task-floc_run-01_part-mag_bold.nii.gz"
            ],
        },
    )
    # primary reverse fmap WITHOUT IntendedFor → task-match fallback.
    _touch(
        ses / "fmap" / "sub-ME1_ses-SM_task-primary_dir-PA_part-mag_sbref.nii.gz",
        {"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.08},
    )

    subj = scan_subject(tmp_path, "ME1", tasks=["primary"])
    fmaps = subj.sessions[0].fmaps
    assert len(fmaps) == 1  # floc fmap dropped (out of scope)
    assert fmaps[0].fmap_id == "primary"
    assert set(fmaps[0].run_ids) == {"01", "02"}
    assert fmaps[0].readout == 0.08

    # Unfiltered: BOTH fmaps survive, and the floc IntendedFor must parse to '01'
    # (a greedy run-(\w+) would yield '01_part' → drop the whole floc group).
    both = scan_subject(tmp_path, "ME1")
    ids = {f.fmap_id: set(f.run_ids) for f in both.sessions[0].fmaps}
    assert ids == {"floc": {"01"}, "primary": {"01", "02"}}


def test_time_based_fmap_assignment(tmp_path: Path):
    """No IntendedFor, multiple task-tagged fmaps with run numbers (SKILLED shape):
    assign each run to the most-recent-preceding fieldmap by AcquisitionTime, with
    task-aware (task, run) identity so run numbers can repeat across tasks."""
    ses = tmp_path / "sub-01" / "ses-01"
    # func: skilled run-01..03 + a rest run-01 (shares the number, different task).
    times = {
        ("skilled", "01"): "10:05:00",
        ("skilled", "02"): "10:35:00",
        ("skilled", "03"): "11:15:00",
        ("rest", "01"): "11:20:00",
    }
    for (task, run), t in times.items():
        _touch(
            ses / "func" / f"sub-01_ses-01_task-{task}_run-{run}_part-mag_bold.nii.gz",
            {"RepetitionTime": 1.5, "PhaseEncodingDirection": "j-", "AcquisitionTime": t},
        )
    # two fieldmaps (task-tagged, run-numbered), no IntendedFor.
    for frun, t in (("1", "10:00:00"), ("2", "11:00:00")):
        _touch(
            ses / "fmap" / f"sub-01_ses-01_task-skilled_dir-PA_run-{frun}_part-mag_sbref.nii.gz",
            {"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05, "AcquisitionTime": t},
        )

    fmaps = {f.fmap_id: f.intended_runs for f in scan_subject(tmp_path, "01").sessions[0].fmaps}
    # runs before 11:00 → fmap run-1; at/after → fmap run-2. rest/01 (11:20) → run-2,
    # distinct from skilled/01 (10:05) → run-1 (no run-number collision).
    assert set(fmaps["skilled-run1"]) == {("skilled", "01"), ("skilled", "02")}
    assert set(fmaps["skilled-run2"]) == {("skilled", "03"), ("rest", "01")}


def test_each_tasks_fmaps_split_its_runs_by_time(tmp_path: Path):
    """MindsEye shape: run-tagged fieldmaps for two tasks, one before each block,
    plus a bare task-primary duplicate of primary's first one (same second). Each
    run takes its own task's latest-preceding fieldmap. Bug of record: the bare
    duplicate claimed all six primary runs, which stopped the time assignment, so
    primary 4-6 took a 35-min-old field and floc took none of its own."""
    ses = tmp_path / "sub-01" / "ses-01"
    runs = {
        ("primary", "01"): "09:28:00",
        ("primary", "02"): "09:37:00",
        ("floc", "01"): "09:55:00",
        ("primary", "03"): "10:04:00",
        ("floc", "02"): "10:31:00",
    }
    for (task, run), t in runs.items():
        _touch(
            ses / "func" / f"sub-01_ses-01_task-{task}_run-{run}_part-mag_bold.nii.gz",
            {"RepetitionTime": 1.5, "PhaseEncodingDirection": "j-", "AcquisitionTime": t},
        )
    fm = {"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05}
    for name, t in (
        ("task-primary_dir-PA_run-01", "09:27:07"),
        ("task-primary_dir-PA", "09:27:07"),  # the duplicate
        ("task-floc_dir-PA_run-01", "09:54:00"),
        ("task-primary_dir-PA_run-02", "10:02:00"),
        ("task-floc_dir-PA_run-02", "10:29:00"),
    ):
        _touch(
            ses / "fmap" / f"sub-01_ses-01_{name}_part-mag_sbref.nii.gz",
            {**fm, "AcquisitionTime": t},
        )
    fmaps = {f.fmap_id: f.intended_runs for f in scan_subject(tmp_path, "01").sessions[0].fmaps}
    assert fmaps == {
        "primary-run01": [("primary", "01"), ("primary", "02")],
        "floc-run01": [("floc", "01")],
        "primary-run02": [("primary", "03")],
        "floc-run02": [("floc", "02")],
    }


def test_time_assignment_never_crosses_acquisitions(tmp_path: Path):
    """MindsEye ME191678 SM shape: untagged fieldmaps, two acquisitions interleaved.
    primary 4-6 come after floc's fieldmap, but it is floc's readout and grid, so
    they keep primary's own (older) one. Bug of record: by time alone, primary 4-6
    were corrected with floc's field."""
    ses = tmp_path / "sub-01" / "ses-01"
    runs = {
        ("primary", "01"): ("15:30:00", 0.082),
        ("floc", "01"): ("15:59:00", 0.064),
        ("primary", "04"): ("16:07:00", 0.082),
    }
    for (task, run), (t, ro) in runs.items():
        _touch(
            ses / "func" / f"sub-01_ses-01_task-{task}_run-{run}_part-mag_bold.nii.gz",
            {
                "RepetitionTime": 1.5,
                "PhaseEncodingDirection": "j-",
                "AcquisitionTime": t,
                "TotalReadoutTime": ro,
            },
        )
    for frun, t, ro in (("01", "15:28:00", 0.082), ("02", "15:57:00", 0.064)):
        _touch(
            ses / "fmap" / f"sub-01_ses-01_acq-sbref_dir-PA_run-{frun}_epi.nii.gz",
            {"PhaseEncodingDirection": "j", "TotalReadoutTime": ro, "AcquisitionTime": t},
        )
    fmaps = {f.fmap_id: f.intended_runs for f in scan_subject(tmp_path, "01").sessions[0].fmaps}
    assert {k: set(v) for k, v in fmaps.items()} == {
        "PA-run01": {("primary", "01"), ("primary", "04")},
        "PA-run02": {("floc", "01")},
    }


def test_opposite_pe_fmaps_merge_into_one_paired_group(tmp_path: Path):
    """fmap/ holding BOTH polarities of the same acquisition (dir-AP + dir-PA) is
    ONE self-contained fieldmap: the pair corrects itself, so the group carries its
    own forward image and never borrows a data run. Pairs still pair with runs by
    AcquisitionTime as usual."""
    ses = tmp_path / "sub-01" / "ses-01"
    for run, t in (("01", "10:05:00"), ("02", "11:30:00")):
        _touch(
            ses / "func" / f"sub-01_ses-01_task-check_run-{run}_bold.nii.gz",
            {"RepetitionTime": 1.5, "PhaseEncodingDirection": "j-", "AcquisitionTime": t},
        )
    # Two AP/PA pairs, run-numbered, no IntendedFor. Data is j- → AP (j-) is forward.
    for frun, t in (("01", "10:00:00"), ("02", "11:00:00")):
        for d, pe in (("AP", "j-"), ("PA", "j")):
            _touch(
                ses / "fmap" / f"sub-01_ses-01_acq-bold_dir-{d}_run-{frun}_epi.nii.gz",
                {"PhaseEncodingDirection": pe, "TotalReadoutTime": 0.05, "AcquisitionTime": t},
            )

    fmaps = scan_subject(tmp_path, "01").sessions[0].fmaps
    # One group per pair, not per polarity.
    assert sorted(f.fmap_id for f in fmaps) == ["run01", "run02"]
    by = {f.fmap_id: f for f in fmaps}
    # Forward = the polarity matching the runs (j-, i.e. AP); reverse = the other.
    assert "dir-AP" in by["run01"].forward_path.name
    assert "dir-PA" in by["run01"].reverse_path.name
    assert by["run01"].intended_runs == [("check", "01")]
    assert by["run02"].intended_runs == [("check", "02")]


def test_unpaired_pe_fmaps_stay_separate_groups(tmp_path: Path):
    """Only ONE polarity present → no pair to merge: the fmap is the reverse image
    and a data run supplies the forward, so the dir stays in the group id."""
    ses = tmp_path / "sub-01" / "ses-01"
    _touch(
        ses / "func" / "sub-01_ses-01_task-check_run-01_bold.nii.gz",
        {"RepetitionTime": 1.5, "PhaseEncodingDirection": "j-"},
    )
    _touch(
        ses / "fmap" / "sub-01_ses-01_dir-PA_epi.nii.gz",
        {"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.05},
    )
    (fg,) = scan_subject(tmp_path, "01").sessions[0].fmaps
    assert fg.fmap_id == "PA"
    assert fg.forward_path is None
    assert fg.intended_runs == [("check", "01")]


def test_same_polarity_fmaps_do_not_pair(tmp_path: Path):
    """Two dir labels but the SAME PE direction is not an opposite-PE pair —
    merging them would hand blipflip two identically-distorted images."""
    ses = tmp_path / "sub-01" / "ses-01"
    for run, t in (("01", "10:05:00"), ("02", "11:05:00")):
        _touch(
            ses / "func" / f"sub-01_ses-01_task-check_run-{run}_bold.nii.gz",
            {"RepetitionTime": 1.5, "PhaseEncodingDirection": "j-", "AcquisitionTime": t},
        )
    for d, t in (("AP", "10:00:00"), ("PA", "11:00:00")):
        _touch(
            ses / "fmap" / f"sub-01_ses-01_dir-{d}_epi.nii.gz",
            {"PhaseEncodingDirection": "j", "AcquisitionTime": t},
        )
    fmaps = scan_subject(tmp_path, "01").sessions[0].fmaps
    assert sorted(f.fmap_id for f in fmaps) == ["AP", "PA"]
    assert all(f.forward_path is None for f in fmaps)


def test_fmap_pe_dir_pairs_when_sidecars_omit_phase_encoding(tmp_path: Path):
    """quirk: some conversions write only InPlanePhaseEncodingDirectionDICOM (axis,
    no sign) on both fmaps and runs — nothing identifies a polarity, so the pair is
    recognisable but not orientable. Unpaired by default (and flagged);
    ``-fmap_pe_dir`` names the runs' polarity and pairs it."""
    ses = tmp_path / "sub-01" / "ses-01"
    for run, t in (("01", "10:00:30"), ("02", "10:05:00")):
        _touch(
            ses / "func" / f"sub-01_ses-01_task-check_run-{run}_part-mag_bold.nii.gz",
            {
                "RepetitionTime": 1.5,
                "InPlanePhaseEncodingDirectionDICOM": "COL",
                "AcquisitionTime": t,
            },
        )
    # Unpaired, these fall through to AcquisitionTime assignment (as the real data
    # does) — and each polarity ends up owning a different run, which is precisely
    # the wrong answer pairing fixes.
    for d, t in (("AP", "10:00:00"), ("PA", "10:01:00")):
        _touch(
            ses / "fmap" / f"sub-01_ses-01_dir-{d}_run-01_part-mag_epi.nii.gz",
            {
                "InPlanePhaseEncodingDirectionDICOM": "COL",
                "TotalReadoutTime": 0.05,
                "AcquisitionTime": t,
            },
        )

    plain = scan_subject(tmp_path, "01").sessions[0]
    assert sorted(f.fmap_id for f in plain.fmaps) == ["AP-run01", "PA-run01"]
    assert pair_undetermined(plain) == ["AP-run01", "PA-run01"]

    paired = scan_subject(tmp_path, "01", fmap_pe_dir="AP").sessions[0]
    (fg,) = paired.fmaps
    assert fg.fmap_id == "run01"
    assert "dir-AP" in fg.forward_path.name
    assert "dir-PA" in fg.reverse_path.name
    assert fg.intended_runs == [("check", "01"), ("check", "02")]  # one fmap, both runs
    assert pair_undetermined(paired) == []


def test_fmap_pe_dir_from_the_runs_own_dir_entity(tmp_path: Path):
    """A run carrying its own ``dir`` entity names the forward polarity itself — no
    flag needed."""
    ses = tmp_path / "sub-01" / "ses-01"
    _touch(
        ses / "func" / "sub-01_ses-01_task-check_dir-PA_run-01_bold.nii.gz",
        {"RepetitionTime": 1.5},
    )
    for d in ("AP", "PA"):
        _touch(ses / "fmap" / f"sub-01_ses-01_dir-{d}_epi.nii.gz", {"TotalReadoutTime": 0.05})
    (fg,) = scan_subject(tmp_path, "01").sessions[0].fmaps
    assert "dir-PA" in fg.forward_path.name  # matches the runs
    assert "dir-AP" in fg.reverse_path.name


def test_anat_prefers_uni(tmp_path: Path):
    ses = tmp_path / "sub-ME1" / "ses-WB"
    _touch(ses / "func" / "sub-ME1_ses-WB_task-x_run-01_bold.nii.gz", {"RepetitionTime": 1.5})
    _touch(ses / "anat" / "sub-ME1_ses-WB_acq-inv1_T1w.nii.gz", {})
    _touch(ses / "anat" / "sub-ME1_ses-WB_acq-uni_T1w.nii.gz", {})
    subj = scan_subject(tmp_path, "ME1")
    assert subj.sessions[0].anat is not None
    assert "acq-uni" in subj.sessions[0].anat.name


def test_anat_group_pools_repeats_of_one_acquisition(tmp_path: Path):
    """Several T1w may be averaged only if they are the same image twice. An
    MP2RAGE session files inv1/inv2/uni all as _T1w, so the group is the picked
    anat's own (acq, rec) — here the two uni runs, across both sessions."""
    from fastfuncstuff.cli.autoproc import anat_group

    for ses, runs in (("WB1", ("01", "02")), ("WB2", ("01",))):
        d = tmp_path / "sub-ME1" / f"ses-{ses}"
        _touch(d / "func" / f"sub-ME1_ses-{ses}_task-x_run-01_bold.nii.gz", {"RepetitionTime": 1.5})
        _touch(d / "anat" / f"sub-ME1_ses-{ses}_acq-inv1_T1w.nii.gz", {})
        for r in runs:
            _touch(d / "anat" / f"sub-ME1_ses-{ses}_acq-uni_run-{r}_T1w.nii.gz", {})
    group = anat_group(scan_subject(tmp_path, "ME1"))
    assert [p.name for p in group] == [
        "sub-ME1_ses-WB1_acq-uni_run-01_T1w.nii.gz",
        "sub-ME1_ses-WB1_acq-uni_run-02_T1w.nii.gz",
        "sub-ME1_ses-WB2_acq-uni_run-01_T1w.nii.gz",
    ]


def test_anat_group_keeps_reconstructions_apart(tmp_path: Path):
    """rec-norm and the raw reconstruction are not the same image; averaging them
    would average two different intensity normalisations."""
    from fastfuncstuff.cli.autoproc import anat_group

    d = tmp_path / "sub-01" / "ses-01"
    _touch(d / "func" / "sub-01_ses-01_task-x_run-01_bold.nii.gz", {"RepetitionTime": 1.5})
    _touch(d / "anat" / "sub-01_ses-01_rec-norm_run-1_T1w.nii.gz", {})
    _touch(d / "anat" / "sub-01_ses-01_rec-norm_run-2_T1w.nii.gz", {})
    _touch(d / "anat" / "sub-01_ses-01_run-3_T1w.nii.gz", {})
    group = anat_group(scan_subject(tmp_path, "01"))
    assert [p.name for p in group] == [
        "sub-01_ses-01_rec-norm_run-1_T1w.nii.gz",
        "sub-01_ses-01_rec-norm_run-2_T1w.nii.gz",
    ]


def test_anat_group_never_averages_across_echoes(tmp_path: Path):
    """A multi-echo MPRAGE files each echo AND the combination under _T1w. The
    combination is the anat, and the echoes are not repeats of it."""
    from fastfuncstuff.cli.autoproc import anat_group

    d = tmp_path / "sub-01" / "ses-anat"
    f = tmp_path / "sub-01" / "ses-fine"
    _touch(f / "func" / "sub-01_ses-fine_task-x_run-01_bold.nii.gz", {"RepetitionTime": 1.5})
    for run in ("1", "2"):
        for echo in ("1", "2", "3", "4"):
            _touch(d / "anat" / f"sub-01_ses-anat_rec-norm_run-{run}_echo-{echo}_T1w.nii.gz", {})
        _touch(d / "anat" / f"sub-01_ses-anat_rec-norm_run-{run}_T1w.nii.gz", {})
        _touch(d / "anat" / f"sub-01_ses-anat_rec-orig_run-{run}_T1w.nii.gz", {})
    group = anat_group(scan_subject(tmp_path, "01"), str(tmp_path))
    assert [p.name for p in group] == [
        "sub-01_ses-anat_rec-norm_run-1_T1w.nii.gz",
        "sub-01_ses-anat_rec-norm_run-2_T1w.nii.gz",
    ]


def test_anat_pick_prefers_the_corrected_reconstruction(tmp_path: Path):
    """Alphabetical order picks 'dist' over 'undist', which is exactly backwards.
    Both reconstructions of one scan are plain _T1w, so the label has to be read."""
    from fastfuncstuff.autoproc.bids import _pick_anat

    for good, bad in (("norm", "orig"), ("corr", "dist"), ("undist", "dist")):
        picked = _pick_anat([Path(f"s_rec-{r}_run-1_T1w.nii.gz") for r in (bad, good)])
        assert picked is not None and f"rec-{good}" in picked.name


def test_anat_group_finds_a_session_with_no_bold_runs(tmp_path: Path):
    """The structural usually lives in its own ses-anat, which has no BOLD runs
    and so is not a session of the scanned Subject at all."""
    from fastfuncstuff.cli.autoproc import anat_group

    _touch(
        tmp_path / "sub-01" / "ses-fine" / "func" / "sub-01_ses-fine_task-x_run-01_bold.nii.gz",
        {"RepetitionTime": 1.5},
    )
    _touch(tmp_path / "sub-01" / "ses-anat" / "anat" / "sub-01_ses-anat_T1w.nii.gz", {})
    subj = scan_subject(tmp_path, "01")
    assert [s.session for s in subj.sessions] == ["fine"]  # ses-anat is not one
    assert [p.name for p in anat_group(subj, str(tmp_path))] == ["sub-01_ses-anat_T1w.nii.gz"]


def test_find_events_ignores_image_only_entities(tmp_path: Path):
    """The bug of record: a `part-mag_bold.nii.gz` run pairs with an events TSV
    that carries no `part-` entity at all (part describes the image, not the
    task). Swapping `_bold`→`_events` looks for a file that BIDS forbids."""
    func = tmp_path / "sub-ME1" / "ses-SM" / "func"
    bold = func / "sub-ME1_ses-SM_task-floc_run-01_part-mag_bold.nii.gz"
    _touch(bold, {"RepetitionTime": 2.0})
    ev = func / "sub-ME1_ses-SM_task-floc_run-01_events.tsv"
    ev.write_text("onset\tduration\n1\t2\n")
    assert find_events(bold, tmp_path) == ev


def test_find_events_inherits_from_coarser_levels(tmp_path: Path):
    """Per-run file wins; without it the search widens (drops run, then ses/sub)
    and walks up to the dataset root."""
    func = tmp_path / "sub-ME1" / "ses-SM" / "func"
    bold = func / "sub-ME1_ses-SM_task-floc_run-01_part-mag_bold.nii.gz"
    _touch(bold, {"RepetitionTime": 2.0})

    root_ev = tmp_path / "task-floc_events.tsv"
    root_ev.write_text("onset\tduration\n1\t2\n")
    assert find_events(bold, tmp_path) == root_ev

    ses_ev = func / "sub-ME1_ses-SM_task-floc_events.tsv"  # no run entity
    ses_ev.write_text("onset\tduration\n1\t2\n")
    assert find_events(bold, tmp_path) == ses_ev

    run_ev = func / "sub-ME1_ses-SM_task-floc_run-01_events.tsv"
    run_ev.write_text("onset\tduration\n1\t2\n")
    assert find_events(bold, tmp_path) == run_ev


def test_find_events_hand_written_task_run_only(tmp_path: Path):
    """Hand-written timing is usually named `task-<T>_run-<N>_events.tsv`: it keeps
    run but drops sub/ses/acq, which a finest-first *prefix* of the drop order can
    never produce. Each run must still get its own file, not run-01 for all."""
    func = tmp_path / "sub-pilot01" / "ses-01" / "func"
    for run in ("01", "02"):
        bold = func / f"sub-pilot01_ses-01_task-fullcheck_acq-axial_run-{run}_part-mag_bold.nii.gz"
        _touch(bold, {"RepetitionTime": 2.0})
        (func / f"task-fullcheck_run-{run}_events.tsv").write_text("onset\tduration\n1\t2\n")

    for run in ("01", "02"):
        bold = func / f"sub-pilot01_ses-01_task-fullcheck_acq-axial_run-{run}_part-mag_bold.nii.gz"
        assert find_events(bold, tmp_path) == func / f"task-fullcheck_run-{run}_events.tsv"


def test_find_events_prefers_the_more_specific_stem(tmp_path: Path):
    """With subsets in play two candidates can sit in one directory; the one that
    drops fewer entities wins, so a full BIDS name beats a stripped one."""
    func = tmp_path / "sub-pilot01" / "ses-01" / "func"
    bold = func / "sub-pilot01_ses-01_task-fullcheck_run-01_part-mag_bold.nii.gz"
    _touch(bold, {"RepetitionTime": 2.0})
    (func / "task-fullcheck_run-01_events.tsv").write_text("onset\n1\n")
    full = func / "sub-pilot01_ses-01_task-fullcheck_run-01_events.tsv"
    full.write_text("onset\n1\n")
    assert find_events(bold, tmp_path) == full


def test_find_events_none_for_a_task_without_them(tmp_path: Path):
    func = tmp_path / "sub-ME1" / "ses-SM" / "func"
    bold = func / "sub-ME1_ses-SM_task-rest_bold.nii.gz"
    _touch(bold, {"RepetitionTime": 2.0})
    (func / "sub-ME1_ses-SM_task-floc_events.tsv").write_text("onset\n1\n")  # other task
    assert find_events(bold, tmp_path) is None


# ---------------------------------------------------------------------------
# GRE (B0) fieldmaps
# ---------------------------------------------------------------------------


def _b0_session(tmp_path: Path, fmap_files: dict[str, dict], n_runs: int = 2) -> Path:
    ses = tmp_path / "sub-01" / "ses-01"
    intended = []
    for run in [f"{i + 1:02d}" for i in range(n_runs)]:
        rel = f"ses-01/func/sub-01_ses-01_task-check_run-{run}_bold.nii.gz"
        _touch(
            ses / "func" / Path(rel).name,
            {
                "RepetitionTime": 1.5,
                "PhaseEncodingDirection": "j-",
                "TotalReadoutTime": 0.05,
            },
        )
        intended.append(rel)
    for name, js in fmap_files.items():
        _touch(ses / "fmap" / name, {"IntendedFor": intended, **js})
    return ses


def test_b0_phasediff_form_scanned(tmp_path: Path):
    """Siemens phasediff + two magnitudes: one GRE group, echo times in ms."""
    _b0_session(
        tmp_path,
        {
            "sub-01_ses-01_phasediff.nii.gz": {"EchoTime1": 0.00492, "EchoTime2": 0.00738},
            "sub-01_ses-01_magnitude1.nii.gz": {},
            "sub-01_ses-01_magnitude2.nii.gz": {},
        },
    )
    (fg,) = scan_subject(tmp_path, "01").sessions[0].fmaps
    assert fg.kind == "b0" and fg.is_b0
    assert fg.phasediff_path is not None and fg.phase_paths == []
    assert len(fg.magnitude_paths) == 2
    assert fg.te_ms == [4.92, 7.38]  # BIDS seconds → the tool's ms
    assert fg.intended_runs == [("check", "01"), ("check", "02")]
    # A GRE sidecar carries no EPI geometry — the emitter reads those from the run.
    assert fg.pe_dir is None and fg.readout is None


def test_b0_phase1_phase2_form_scanned(tmp_path: Path):
    _b0_session(
        tmp_path,
        {
            "sub-01_ses-01_phase1.nii.gz": {"EchoTime": 0.00492},
            "sub-01_ses-01_phase2.nii.gz": {"EchoTime": 0.00738},
            "sub-01_ses-01_magnitude1.nii.gz": {},
            "sub-01_ses-01_magnitude2.nii.gz": {},
        },
    )
    (fg,) = scan_subject(tmp_path, "01").sessions[0].fmaps
    assert [p.name for p in fg.phase_paths] == [
        "sub-01_ses-01_phase1.nii.gz",
        "sub-01_ses-01_phase2.nii.gz",
    ]
    assert fg.te_ms == [4.92, 7.38]


def test_b0_ready_made_hz_form_scanned(tmp_path: Path):
    """The ``fieldmap`` form is already in Hz — no echo times to find."""
    _b0_session(
        tmp_path,
        {
            "sub-01_ses-01_fieldmap.nii.gz": {"Units": "Hz"},
            "sub-01_ses-01_magnitude.nii.gz": {},
        },
    )
    (fg,) = scan_subject(tmp_path, "01").sessions[0].fmaps
    assert fg.fieldmap_path is not None and fg.te_ms == []


def test_magnitudes_alone_are_not_a_fieldmap(tmp_path: Path):
    """Magnitudes with nothing to derive a field from must not produce a group —
    silently planning distortion correction off an empty field is worse than none."""
    _b0_session(tmp_path, {"sub-01_ses-01_magnitude1.nii.gz": {}})
    assert scan_subject(tmp_path, "01").sessions[0].fmaps == []


def test_pepolar_wins_over_b0_unless_asked(tmp_path: Path):
    """A fmap/ offering both flavours: reverse-PE by default, GRE on request. Never
    both — one session gets one kind, so every group shares an estimation framework."""
    ses = _b0_session(
        tmp_path,
        {
            "sub-01_ses-01_phasediff.nii.gz": {"EchoTime1": 0.00492, "EchoTime2": 0.00738},
            "sub-01_ses-01_magnitude1.nii.gz": {},
        },
    )
    for d, pe in (("AP", "j-"), ("PA", "j")):
        _touch(
            ses / "fmap" / f"sub-01_ses-01_dir-{d}_epi.nii.gz",
            {"PhaseEncodingDirection": pe, "TotalReadoutTime": 0.05},
        )
    (auto,) = scan_subject(tmp_path, "01").sessions[0].fmaps
    assert auto.kind == "pepolar"
    (forced,) = scan_subject(tmp_path, "01", fmap_kind="b0").sessions[0].fmaps
    assert forced.kind == "b0"
    (forced_pe,) = scan_subject(tmp_path, "01", fmap_kind="pepolar").sessions[0].fmaps
    assert forced_pe.kind == "pepolar"


def test_b0_intendedfor_on_the_magnitude_only(tmp_path: Path):
    """quirk: some datasets put IntendedFor on the magnitude rather than the
    fieldmap image. Taking it from wherever it is beats dropping the fieldmap."""
    ses = tmp_path / "sub-01" / "ses-01"
    _touch(
        ses / "func" / "sub-01_ses-01_task-check_run-01_bold.nii.gz",
        {"RepetitionTime": 1.5, "PhaseEncodingDirection": "j-"},
    )
    _touch(
        ses / "fmap" / "sub-01_ses-01_phasediff.nii.gz",
        {"EchoTime1": 0.00492, "EchoTime2": 0.00738},
    )
    _touch(
        ses / "fmap" / "sub-01_ses-01_magnitude1.nii.gz",
        {"IntendedFor": ["ses-01/func/sub-01_ses-01_task-check_run-01_bold.nii.gz"]},
    )
    (fg,) = scan_subject(tmp_path, "01").sessions[0].fmaps
    assert fg.intended_runs == [("check", "01")]


def test_free_text_acq_label_does_not_hide_the_fieldmap(tmp_path: Path):
    """`acq` is a free-text protocol label (acq-fMRI, acq-2mm); only acq-bold /
    acq-sbref name the image FORM. Treating any acq as the form made
    acq-fMRI_dir-AP_epi unrecognisable, and the fieldmap vanished silently — the
    script then had no blip stage at all despite -recipe complete."""
    ses = tmp_path / "sub-01" / "ses-01"
    _touch(
        ses / "func" / "sub-01_ses-01_task-bar1_bold.nii.gz",
        {"RepetitionTime": 1.5, "PhaseEncodingDirection": "j"},
    )
    for d, pe in (("AP", "j-"), ("PA", "j")):
        _touch(
            ses / "fmap" / f"sub-01_ses-01_acq-fMRI_dir-{d}_epi.nii.gz",
            {
                "PhaseEncodingDirection": pe,
                "TotalReadoutTime": 0.05,
                "IntendedFor": ["ses-01/func/sub-01_ses-01_task-bar1_bold.nii.gz"],
            },
        )

    (fg,) = scan_subject(tmp_path, "01").sessions[0].fmaps
    # Forward = the polarity matching the data (j → PA); reverse = the other.
    assert "dir-PA" in fg.forward_path.name
    assert "dir-AP" in fg.reverse_path.name
    assert fg.intended_runs == [("bar1", "")]


def _mp2rage_session(root: Path, runs=("01", "02"), inv1=True) -> Path:
    d = root / "sub-ME1" / "ses-WB"
    _touch(d / "func" / "sub-ME1_ses-WB_task-x_run-01_bold.nii.gz", {"RepetitionTime": 1.5})
    for r in runs:
        for acq in ("uni", "inv2", *(("inv1",) if inv1 else ())):
            _touch(d / "anat" / f"sub-ME1_ses-WB_acq-{acq}_run-{r}_T1w.nii.gz", {})
    return d / "anat"


def test_mp2rage_partners_pair_each_uni_with_its_own_run(tmp_path: Path):
    """Two MP2RAGE runs: run-02's UNI must be MPRAGEised by run-02's INV2, never
    run-01's — they are different head positions."""
    from fastfuncstuff.autoproc.bids import mp2rage_partners

    anat = _mp2rage_session(tmp_path)
    inv2, inv1 = mp2rage_partners(anat / "sub-ME1_ses-WB_acq-uni_run-02_T1w.nii.gz")
    assert inv2 is not None and inv2.name == "sub-ME1_ses-WB_acq-inv2_run-02_T1w.nii.gz"
    assert inv1 is not None and inv1.name == "sub-ME1_ses-WB_acq-inv1_run-02_T1w.nii.gz"
    # not a UNI → no partners, even with inversions beside it
    assert mp2rage_partners(anat / "sub-ME1_ses-WB_acq-inv2_run-01_T1w.nii.gz") == (None, None)


def _presurf_args(**kw):
    from argparse import Namespace

    base = dict(
        presurf="auto", presurf_tpm="/t/TPM.nii", spm_tpm=None, tpm=None, presurf_denoise="off"
    )
    return Namespace(**{**base, **kw})


def test_presurf_resolves_only_for_scanned_mp2rage(tmp_path: Path, monkeypatch):
    from fastfuncstuff.cli.autoproc import _resolve_presurf

    monkeypatch.delenv("FFS_TPM", raising=False)
    anat = _mp2rage_session(tmp_path, runs=("01",), inv1=False)
    uni = [str(anat / "sub-ME1_ses-WB_acq-uni_run-01_T1w.nii.gz")]
    triples, tpm, _ = _resolve_presurf(_presurf_args(), uni, from_bids=True)
    assert triples == [(uni[0], uni[0].replace("acq-uni", "acq-inv2"), None)]
    assert tpm == "/t/TPM.nii"
    # -anat FILE means that file as it is; -presurf no opts out
    assert _resolve_presurf(_presurf_args(), uni, from_bids=False)[0] == []
    assert _resolve_presurf(_presurf_args(presurf="no"), uni, from_bids=True)[0] == []
    # no template anywhere → falls back to UNI, and says so
    triples, _, note = _resolve_presurf(_presurf_args(presurf_tpm=None), uni, from_bids=True)
    assert triples == [] and "FFS_TPM" in note


def test_anat_scan_is_case_blind(tmp_path: Path):
    """Scanner exports and hand-renamed trees write _t1w and acq-UNI; neither may
    hide the anat, and the UNI must still find its inversions."""
    from fastfuncstuff.autoproc.bids import mp2rage_partners

    ses = tmp_path / "sub-ME1" / "ses-WB"
    _touch(ses / "func" / "sub-ME1_ses-WB_task-x_run-01_bold.nii.gz", {"RepetitionTime": 1.5})
    _touch(ses / "anat" / "sub-ME1_ses-WB_acq-inv2_T1w.nii.gz", {})
    _touch(ses / "anat" / "sub-ME1_ses-WB_acq-INV1_t1w.nii.gz", {})
    _touch(ses / "anat" / "sub-ME1_ses-WB_acq-UNI_t1w.nii.gz", {})
    _touch(ses / "anat" / "sub-ME1_ses-WB_acq-UNI_t1w_Denoised.nii.gz", {})
    anat = scan_subject(tmp_path, "ME1").sessions[0].anat
    assert anat is not None and anat.name == "sub-ME1_ses-WB_acq-UNI_t1w.nii.gz"
    inv2, inv1 = mp2rage_partners(anat)
    assert inv2 is not None and inv2.name == "sub-ME1_ses-WB_acq-inv2_T1w.nii.gz"
    assert inv1 is not None and inv1.name == "sub-ME1_ses-WB_acq-INV1_t1w.nii.gz"


def test_t2w_partner_matches_the_t1s_own_reconstruction(tmp_path: Path):
    """A session with both reconstructions of both contrasts: the acq-corr T1w's
    mate is the acq-corr T2w, not the uncorrected one filed first alphabetically."""
    from fastfuncstuff.autoproc.bids import t2w_partner

    anat = tmp_path / "sub-A" / "ses-01" / "anat"
    for acq in ("corr", "dist"):
        for suf in ("T1w", "T2w"):
            _touch(anat / f"sub-A_ses-01_acq-{acq}_{suf}.nii.gz", {})
    t2 = t2w_partner(anat / "sub-A_ses-01_acq-corr_T1w.nii.gz")
    assert t2 is not None and t2.name == "sub-A_ses-01_acq-corr_T2w.nii.gz"
    assert t2w_partner(anat / "sub-A_ses-01_acq-dist_T1w.nii.gz") == (
        anat / "sub-A_ses-01_acq-dist_T2w.nii.gz"
    )
    # No T2w at all is the common case, and not an error.
    solo = tmp_path / "sub-B" / "ses-01" / "anat"
    _touch(solo / "sub-B_ses-01_T1w.nii.gz", {})
    assert t2w_partner(solo / "sub-B_ses-01_T1w.nii.gz") is None


def _tpm_args(**kw):
    from argparse import Namespace

    base = dict(
        tpm=None,
        spm_tpm="/t/TPM.nii",
        suma=None,
        anat_t2=None,
        anat_nonlin_method="segment",
        ref_file=None,
        grand_reference=None,
        ref_anat=None,
        ref_anat_nonlin_method="segment",
    )
    return Namespace(**{**base, **kw})


def test_tpm_resolution_builds_from_the_anat_when_there_is_no_freesurfer(tmp_path, monkeypatch):
    """The error this replaces: -anat_nonlin with a plain T1w and no -suma could not
    run at all. Order of preference is -tpm, then -suma's labels, then the anat."""
    from fastfuncstuff.cli.autoproc import _resolve_tpm

    monkeypatch.delenv("FFS_TPM", raising=False)
    anat = tmp_path / "sub-A" / "ses-01" / "anat"
    for suf in ("T1w", "T2w"):
        _touch(anat / f"sub-A_ses-01_acq-corr_{suf}.nii.gz", {})
    t1 = str(anat / "sub-A_ses-01_acq-corr_T1w.nii.gz")

    def res(**kw):
        return _resolve_tpm(_tpm_args(**kw), anat_nonlin=True, anat_path=t1, anat_skull=True)

    tpm, fs_tpm, anat_tpm, spm_tpm, t2 = res()
    assert tpm == "tpm_work/sub_specific_anatTPM.nii.gz" and anat_tpm and not fs_tpm
    assert spm_tpm == "/t/TPM.nii" and t2 is not None and t2.endswith("acq-corr_T2w.nii.gz")
    assert res(anat_t2="none")[4] is None  # T1 alone on request
    # -suma wins (FreeSurfer labels are sharper), and an explicit -tpm wins outright.
    assert res(suma="/fs/SUMA")[1:3] == (True, False)
    assert res(tpm="/my/tpm.nii")[:3] == ("/my/tpm.nii", False, False)
    # -ref_anat's segment reads the same TPM, so it is built without -anat_nonlin too;
    # register reads none.
    ref = dict(ref_anat="/r/7T.nii.gz")
    built = _resolve_tpm(_tpm_args(**ref), anat_nonlin=False, anat_path=t1, anat_skull=True)
    assert built[0] == "tpm_work/sub_specific_anatTPM.nii.gz"
    reg = _tpm_args(**ref, ref_anat_nonlin_method="register")
    assert _resolve_tpm(reg, anat_nonlin=False, anat_path=t1, anat_skull=True)[0] is None
    # Nothing to segment: no group TPM, a stripped anat, or a borrowed anat link.
    assert res(spm_tpm=None)[2] is False
    assert _resolve_tpm(_tpm_args(), anat_nonlin=True, anat_path=t1, anat_skull=False)[2] is False
    assert res(ref_file="/other/ref.nii.gz")[2] is False


def test_anat_skull_defaults_to_stripping_all_but_the_suma_brain():
    from argparse import Namespace

    from fastfuncstuff.cli.autoproc import _resolve_anat_skull

    def strip(skull, suma=None, presurf=False):
        return _resolve_anat_skull(Namespace(anat_skull=skull, suma=suma), presurf)

    assert strip("auto")  # BIDS-scanned or -anat FILE: assume a skull
    assert not strip("auto", suma="/fs/SUMA")  # brain.nii.gz is already stripped
    assert not strip("no")
    assert strip("no", presurf=True)  # the MPRAGEised head always has one


def test_acq_split_and_legacy_phase_suffix(tmp_path: Path):
    """ds003427 shape: acq-ge and acq-se runs of one task share run numbers, and
    the GE phase uses the pre-1.5 ``_phase`` suffix. Bug of record: (task, run)
    grouping let the SE magnitude silently replace the GE one in every run."""
    func = tmp_path / "sub-03" / "func"
    for run in ("01", "02"):
        for acq in ("ge", "se"):
            _touch(
                func / f"sub-03_task-cb_acq-{acq}_run-{run}_bold.nii.gz", {"RepetitionTime": 2.5}
            )
        _touch(func / f"sub-03_task-cb_acq-ge_run-{run}_phase.nii.gz")

    runs = scan_subject(tmp_path, "03").sessions[0].bold_runs
    by = {(r.task, r.run): r for r in runs}
    assert set(by) == {("cbGe", "01"), ("cbGe", "02"), ("cbSe", "01"), ("cbSe", "02")}
    assert "acq-ge" in by["cbGe", "01"].mag_path.name
    assert "acq-se" in by["cbSe", "01"].mag_path.name
    assert by["cbGe", "02"].phase_path is not None
    assert by["cbGe", "02"].phase_path.name.endswith("_run-02_phase.nii.gz")
    assert by["cbSe", "01"].phase_path is None


def test_single_acq_task_keeps_its_name(tmp_path: Path):
    func = tmp_path / "sub-1" / "func"
    _touch(func / "sub-1_task-Foo_acq-mb4_run-01_bold.nii.gz", {"RepetitionTime": 1.0})
    assert scan_subject(tmp_path, "1").sessions[0].bold_runs[0].task == "Foo"


def test_multiecho_echoes_land_on_one_run_in_echo_order(tmp_path: Path):
    """echo-1..3 (part-mag + part-phase): one BoldRun carrying every echo, ordered,
    TEs from each echo's own sidecar, and echo 1 as the run's main image. Before
    this, each echo overwrote the last and the run was its last echo."""
    func = tmp_path / "sub-01" / "func"
    tes = {1: 0.0076, 2: 0.0217, 3: 0.0358}
    for e in (3, 1, 2):  # filesystem order must not matter
        base = f"sub-01_task-foo_run-1_echo-{e}"
        _touch(func / f"{base}_part-mag_bold.nii.gz", {"EchoTime": tes[e], "RepetitionTime": 2.0})
        _touch(func / f"{base}_part-phase_bold.nii.gz", {"EchoTime": tes[e]})
    subj = scan_subject(tmp_path, "01")
    runs = [r for s in subj.sessions for r in s.bold_runs]
    assert len(runs) == 1
    r = runs[0]
    assert r.is_multiecho and r.n_echoes == 3
    assert [p.name for p in r.echo_mags] == [
        f"sub-01_task-foo_run-1_echo-{e}_part-mag_bold.nii.gz" for e in (1, 2, 3)
    ]
    assert all(p is not None and "part-phase" in p.name for p in r.echo_phases)
    assert r.echo_tes == pytest.approx([7.6, 21.7, 35.8])
    assert r.mag_path == r.echo_mags[0] and r.phase_path == r.echo_phases[0]


def test_single_echo_runs_are_unchanged(tmp_path: Path):
    func = tmp_path / "sub-01" / "func"
    _touch(func / "sub-01_task-foo_run-1_bold.nii.gz", {"EchoTime": 0.03, "RepetitionTime": 2.0})
    r = scan_subject(tmp_path, "01").sessions[0].bold_runs[0]
    assert not r.is_multiecho and r.n_echoes == 1 and r.echo_mags == []


def test_ref_anat_mp2rage_partners_are_found_beside_an_explicit_file(tmp_path, monkeypatch):
    """-ref_anat FILE still gets its INV2 looked up (unlike -anat FILE): the ref is
    the 7T MP2RAGE, and presurfer is what keeps UNI's noise out of its fits."""
    from argparse import Namespace

    from fastfuncstuff.cli.autoproc import _resolve_ref_presurf

    monkeypatch.delenv("FFS_TPM", raising=False)
    anat = tmp_path / "sub-A" / "ses-7T" / "anat"
    for acq in ("uni", "inv1", "inv2"):
        _touch(anat / f"sub-A_ses-7T_acq-{acq}_T1w.nii.gz", {})
    uni = str(anat / "sub-A_ses-7T_acq-uni_T1w.nii.gz")

    def res(**kw):
        base = dict(
            ref_anat=uni,
            ref_anat_inv2=None,
            ref_anat_inv1=None,
            presurf="auto",
            presurf_tpm=None,
            spm_tpm="/t/TPM.nii",
            tpm=None,
        )
        return _resolve_ref_presurf(Namespace(**{**base, **kw}))

    triple, tpm, _ = res()
    assert triple == (uni, uni.replace("acq-uni", "acq-inv2"), uni.replace("acq-uni", "acq-inv1"))
    assert tpm == "/t/TPM.nii"
    assert res(ref_anat_inv2="/x/inv2.nii.gz")[0][1] == "/x/inv2.nii.gz"  # override wins
    assert res(presurf="no")[0] is None
    assert res(spm_tpm=None)[0] is None  # no template: skipped, with a note
    plain = str(anat / "sub-A_ses-7T_acq-inv2_T1w.nii.gz")  # not a UNI: used as given
    assert res(ref_anat=plain)[0] is None
