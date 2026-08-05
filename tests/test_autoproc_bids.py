"""Scanner tests: entity parsing, sidecar inheritance, and the real-world
quirks ffs_autoproc must survive (non-contiguous runs, missing IntendedFor,
mislabeled phase, dataset-root events/json)."""

from __future__ import annotations

import json
from pathlib import Path

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
