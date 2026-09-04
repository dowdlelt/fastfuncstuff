"""Progress reporting: the spinner is the single voice for every write.

Bug of record: a big moco write printed four lines for one file — the writer's
own write announcement + "Wrote ... in 2.0s", the spinner's
"done (2.3s)", and the CLI's "Saved: ...".

Second bug of record: once that was fixed, one run still printed two FLAVOURS of
"I saved a file" — a two-line ``Writing (N GB uncompressed)`` / ``Wrote (M GB) in
2.2s`` pair from save_nifti's own big-write fallback, next to one-line spinner
lines from the callers that wrap their own saves. save_nifti now goes through the
same spinner, so there is one shape and one stream (stderr) for all of them.
"""

import numpy as np

from fastfuncstuff.cli_utils import spinner
from fastfuncstuff.io import afni as afni_io
from fastfuncstuff.utils import io_progress_suppressed, suppress_io_progress


def test_suppression_flag_nests_and_resets():
    assert not io_progress_suppressed()
    with suppress_io_progress():
        assert io_progress_suppressed()
        with suppress_io_progress():
            assert io_progress_suppressed()
        assert io_progress_suppressed()
    assert not io_progress_suppressed()


def test_spinner_suppresses_writer_announcement(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(afni_io, "_BIG_WRITE_BYTES", 1)  # every write is "big"
    data = np.zeros((4, 4, 4), dtype=np.float32)
    affine = np.eye(4)

    afni_io.save_nifti(data, str(tmp_path / "loud.nii"), affine=affine)
    cap = capsys.readouterr()
    # ONE line, on the spinner's stream, in the spinner's shape — not a
    # Writing/Wrote pair on stdout in a shape nothing else in the toolbox uses.
    loud = [ln for ln in cap.err.splitlines() if "loud.nii" in ln]
    assert len(loud) == 1, cap.err
    assert loud[0].startswith("Writing loud.nii") and "done (" in loud[0]
    assert cap.out == ""

    # An outer spinner still suppresses it, so a wrapped write stays one line.
    with spinner("Writing quiet.nii"):
        afni_io.save_nifti(data, str(tmp_path / "quiet.nii"), affine=affine)
    cap = capsys.readouterr()
    assert cap.out == ""
    assert len([ln for ln in cap.err.splitlines() if "quiet.nii" in ln]) == 1
