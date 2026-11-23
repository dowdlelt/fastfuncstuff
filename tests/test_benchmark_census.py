"""benchmark.census: manifest facts come from the files, with BIDS inheritance."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from fastfuncstuff.benchmark import census as C


def _nii(path: Path, shape=(8, 8, 6, 5), zooms=(2.0, 2.0, 2.5, 1.0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.zeros(shape, np.int16), np.diag([*zooms[:3], 1.0]))
    img.header.set_zooms(zooms)
    nib.save(img, path)


def _json(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d))


def _tree(root: Path) -> Path:
    bids = root / "lab" / "subjects"
    # Top-level task sidecar: inherited by every run; per-echo files override EchoTime.
    _json(
        bids / "task-faces_bold.json",
        {
            "RepetitionTime": 1.5,
            "MagneticFieldStrength": 7,
            "SliceTiming": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],  # 6 slices, 3 shots -> MB 2
            "PhaseEncodingDirection": "j-",
        },
    )
    for sub in ("01", "02"):
        f = bids / f"sub-{sub}" / "func"
        for echo, te in ((1, 0.012), (2, 0.030)):
            stem = f"sub-{sub}_task-faces_run-1_echo-{echo}"
            _nii(f / f"{stem}_part-mag_bold.nii.gz")
            _nii(f / f"{stem}_part-phase_bold.nii.gz")
            _json(f / f"{stem}_part-mag_bold.json", {"EchoTime": te})
        (f / f"sub-{sub}_task-faces_run-1_events.tsv").write_text("onset\tduration\n1\t1\n")
    phys = bids / "sub-01" / "func" / "sub-01_task-faces_run-1_recording-x_physio.tsv.gz"
    with gzip.open(phys, "wt") as fh:
        fh.write("1\t2\n")
    _json(
        phys.with_name(phys.name.replace(".tsv.gz", ".json")),
        {"Columns": ["cardiac", "respiratory"]},
    )
    _nii(bids / "sub-01" / "anat" / "sub-01_T1w.nii.gz", shape=(8, 8, 8), zooms=(1.0, 1.0, 1.0))
    _nii(bids / "sub-01" / "misc" / "localizer.nii.gz", shape=(4, 4, 4), zooms=(1.0, 1.0, 1.0))
    return bids


def test_census_reads_inherited_sidecars_and_headers(tmp_path):
    bids = _tree(tmp_path)
    assert C.find_bids_root(tmp_path / "lab") == bids

    m = C.census(C.LocalSource(bids))
    assert m["dataset_id"] == "lab"  # generic 'subjects' folder skipped
    assert m["n_subjects"] == 2
    (acq,) = m["acquisitions"]  # echoes and parts fold into one acquisition
    assert acq["entities"] == {"task": "faces"}
    assert acq["n_runs"] == 2 and acq["n_subjects"] == 2
    assert acq["tr_s"] == 1.5 and acq["field_T"] == 7
    assert acq["te_ms"] == [12.0, 30.0]
    assert acq["multiband"] == 2
    assert acq["acq_type"] == "2D"
    assert acq["voxel_mm"] == [2.0, 2.0, 2.5] and acq["n_vols"] == 5
    assert acq["n_runs_events"] == 2
    assert acq["n_runs_physio"] == 1
    for tag in (
        "events",
        "phase",
        "multi_echo",
        "physio_cardiac",
        "physio_resp",
        "epi_2d",
        "multiband",
    ):
        assert tag in acq["offers"], tag
    assert "rest" not in acq["offers"] and "no_events" not in acq["offers"]
    assert "anat_t1" in m["offers"]

    issues = " | ".join(m["bids_issues"])
    assert "dataset_description.json" in issues
    assert "outside func/anat/fmap" in issues  # sub-01/misc/localizer.nii.gz
    assert "sidecar lacks" in issues and "TotalReadoutTime" in issues


def test_header_from_truncated_gzip(tmp_path):
    p = tmp_path / "x.nii.gz"
    _nii(p, shape=(10, 12, 7, 3), zooms=(1.5, 1.5, 3.0, 2.0))
    raw = p.read_bytes()[:400]  # far short of the full file
    h = C.header_from_bytes(raw, gz=True)
    assert h == {"matrix": [10, 12, 7], "n_vols": 3, "voxel_mm": [1.5, 1.5, 3.0]}


def test_gaps_report_flags_uncovered_tools(tmp_path):
    m = C.census(C.LocalSource(_tree(tmp_path)))
    rep = C.gaps_report([m])
    assert "physio" in rep
    t2 = next(line for line in rep.splitlines() if line.startswith("t2smap"))
    assert "lab" in t2
    blip = next(line for line in rep.splitlines() if line.startswith("blipflip"))
    assert "NONE" in blip  # no reverse-PE fieldmap in the tree
