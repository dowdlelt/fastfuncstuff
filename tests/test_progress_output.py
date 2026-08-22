"""Progress reporting: the spinner is the single voice for a wrapped write.

Bug of record: a big moco write printed four lines for one file — the writer's
own write announcement + "Wrote ... in 2.0s", the spinner's
"done (2.3s)", and the CLI's "Saved: ...".
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
    loud = capsys.readouterr().out
    assert "Writing loud.nii" in loud
    assert "Wrote loud.nii" in loud
    assert "take a while" not in loud

    with spinner("Writing quiet.nii"):
        afni_io.save_nifti(data, str(tmp_path / "quiet.nii"), affine=affine)
    assert capsys.readouterr().out == ""
