"""Scanner tests: entity parsing, sidecar inheritance, and the real-world
quirks ffs_autoproc must survive (non-contiguous runs, missing IntendedFor,
mislabeled phase, dataset-root events/json)."""

from __future__ import annotations

import json
from pathlib import Path

from fastfuncstuff.autoproc.bids import parse_entities, parse_suffix, scan_subject


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
    assert set(fmaps[0].intended_runs) == {"01", "02"}
    assert fmaps[0].readout == 0.08

    # Unfiltered: BOTH fmaps survive, and the floc IntendedFor must parse to '01'
    # (a greedy run-(\w+) would yield '01_part' → drop the whole floc group).
    both = scan_subject(tmp_path, "ME1")
    ids = {f.fmap_id: set(f.intended_runs) for f in both.sessions[0].fmaps}
    assert ids == {"floc": {"01"}, "primary": {"01", "02"}}


def test_anat_prefers_uni(tmp_path: Path):
    ses = tmp_path / "sub-ME1" / "ses-WB"
    _touch(ses / "func" / "sub-ME1_ses-WB_task-x_run-01_bold.nii.gz", {"RepetitionTime": 1.5})
    _touch(ses / "anat" / "sub-ME1_ses-WB_acq-inv1_T1w.nii.gz", {})
    _touch(ses / "anat" / "sub-ME1_ses-WB_acq-uni_T1w.nii.gz", {})
    subj = scan_subject(tmp_path, "ME1")
    assert subj.sessions[0].anat is not None
    assert "acq-uni" in subj.sessions[0].anat.name
