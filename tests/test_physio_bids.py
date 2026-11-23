"""io.physio_bids: CMRR logs land beside the right run, matched by the scanner clock."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from fastfuncstuff.benchmark import census as C
from fastfuncstuff.cli import util_physio2bids
from fastfuncstuff.io import physio_bids as P


def _cmrr(d: Path, uuid: str, acq_tick: int, n_vols: int, lead_ticks: int = 8000) -> None:
    d.mkdir(parents=True, exist_ok=True)
    start = acq_tick - lead_ticks
    info = [
        f"UUID = {uuid}",
        f"NumVolumes  = {n_vols}",
        "",
        "VOLUME SLICE ACQ_START_TICS ACQ_FINISH_TICS ECHO",
    ]
    info += [f"{v} 0 {acq_tick + 400 * v} {acq_tick + 400 * v + 14} 0" for v in range(n_vols)]
    (d / f"Physio_x_{uuid}_Info.log").write_text("\n".join(info) + "\n")
    end = acq_tick + 400 * n_vols
    # PULS every 2 ticks with one dropped sample; RESP every 8 ticks.
    puls = [t for t in range(start, end, 2) if t != start + 10]
    (d / f"Physio_x_{uuid}_PULS.log").write_text(
        "ACQ_TIME_TICS CHANNEL VALUE\n" + "".join(f"{t} PULS {t % 97}\n" for t in puls)
    )
    (d / f"Physio_x_{uuid}_RESP.log").write_text(
        "".join(f"{t} RESP {t % 13}\n" for t in range(start, end, 8))
    )


def _run(func: Path, stem: str, acq_s: float, n_vols: int) -> Path:
    func.mkdir(parents=True, exist_ok=True)
    nii = func / f"{stem}_part-mag_bold.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, n_vols), np.int16), np.eye(4)), nii)
    h, rem = divmod(acq_s, 3600)
    m, s = divmod(rem, 60)
    (func / f"{stem}_part-mag_bold.json").write_text(
        json.dumps(
            {"AcquisitionTime": f"{int(h):02d}:{int(m):02d}:{s:09.6f}", "RepetitionTime": 1.0}
        )
    )
    return nii


def test_clock_matching_ignores_folder_names(tmp_path):
    logs, bids = tmp_path / "logs", tmp_path / "bids" / "sub-01"
    func = bids / "func"
    # Folder names deliberately swapped: only the clock may decide.
    _cmrr(logs / "run02_folder", "aaa", acq_tick=16_000_000, n_vols=5)
    _cmrr(logs / "run01_folder", "bbb", acq_tick=16_400_000, n_vols=5)
    r1 = _run(func, "sub-01_task-a_run-1", 16_000_000 * 0.0025, 5)
    r2 = _run(func, "sub-01_task-a_run-2", 16_400_000 * 0.0025, 5)
    _run(func, "sub-01_task-a_run-3", 16_900_000 * 0.0025, 5)  # no recording

    recs = P.find_cmrr_recordings(logs)
    pairs = {
        b.name: (r, why)
        for b, r, why in P.match_runs(
            recs,
            [
                (r1, json.loads(r1.with_name(r1.name.replace(".nii.gz", ".json")).read_text()), 5),
                (r2, json.loads(r2.with_name(r2.name.replace(".nii.gz", ".json")).read_text()), 5),
            ],
        )
    }
    assert pairs[r1.name][0].info.parent.name == "run02_folder"
    assert pairs[r2.name][0].info.parent.name == "run01_folder"

    util_physio2bids.main(["-logs", str(logs), "-bids", str(bids)])
    card = func / "sub-01_task-a_run-1_recording-cardiac_physio.json"
    meta = json.loads(card.read_text())
    assert meta["Columns"] == ["cardiac"]
    assert abs(meta["SamplingFrequency"] - 200.0) < 1e-6  # 2 ticks of 2.5 ms
    assert abs(meta["StartTime"] - (-8000 * 0.0025)) < 1e-6  # recording began 20 s early
    with gzip.open(str(card).replace(".json", ".tsv.gz"), "rt") as f:
        n = sum(1 for _ in f)
    assert n == (8000 + 400 * 5) // 2  # the dropped sample is interpolated back
    resp = json.loads((func / "sub-01_task-a_run-1_recording-respiratory_physio.json").read_text())
    assert abs(resp["SamplingFrequency"] - 50.0) < 1e-6
    assert not list(func.glob("sub-01_task-a_run-3_*physio*"))

    m = C.census(C.LocalSource(tmp_path / "bids"))
    (acq,) = m["acquisitions"]
    assert {"physio_cardiac", "physio_resp"} <= set(acq["offers"])
    assert acq["n_runs_physio"] == 2


def test_volume_count_mismatch_refuses(tmp_path):
    _cmrr(tmp_path / "l", "ccc", acq_tick=16_000_000, n_vols=6)
    nii = _run(tmp_path / "f", "sub-01_task-a_run-1", 16_000_000 * 0.0025, 5)
    meta = json.loads(nii.with_name(nii.name.replace(".nii.gz", ".json")).read_text())
    ((_, rec, why),) = P.match_runs(P.find_cmrr_recordings(tmp_path / "l"), [(nii, meta, 5)])
    assert rec is None and "volume count" in why
