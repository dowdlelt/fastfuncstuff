"""-1Dmatrix_init: start the search from a given matrix instead of the headers.

The use is a hand alignment handed over for refinement -- data whose headers
put the source too far away for the search to find. So the test puts the
source 24 mm off in its header, gives a start a couple of millimetres and
degrees from the truth, and checks the fit lands on the truth rather than
staying at the start or going back to the headers.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

from fastfuncstuff.cli.allineate import main
from fastfuncstuff.processing.affine import load_matrix_1D, read_aff12_rows
from fastfuncstuff.viewer import align


def _phantom(n=32):
    vol = np.zeros((n, n, n), np.float32)
    vol[8:24, 10:22, 12:26] = 80.0
    vol[12:18, 14:18, 16:20] = 160.0
    return vol


def test_read_aff12_rows_reads_every_row(tmp_path):
    path = tmp_path / "m.aff12.1D"
    rows = np.arange(24, dtype=float).reshape(2, 12)
    path.write_text("# two volumes\n" + "\n".join(" ".join(map(str, r)) for r in rows) + "\n")
    out = read_aff12_rows(path)
    assert out.shape == (2, 4, 4)
    assert np.allclose(out[1, :3, :4].ravel(), rows[1])
    assert np.allclose(out[:, 3], (0, 0, 0, 1))


def _pair(tmp_path, header_error_mm):
    """A base, and the same object on a padded grid whose header is wrong in x.

    A different grid on purpose: two volumes of one shape are taken as already
    voxel-aligned and the headers never enter, which would let a start that was
    ignored pass for one that was used.
    """
    vol = _phantom()
    base_aff = np.diag([2.0, 2.0, 2.0, 1.0])
    base_aff[:3, 3] = -32.0
    padded = np.pad(vol, 1)
    true_aff = base_aff.copy()
    true_aff[:3, 3] -= 2.0  # one 2 mm voxel of padding
    src_aff = true_aff.copy()
    src_aff[0, 3] += header_error_mm
    nib.save(nib.Nifti1Image(vol.transpose(2, 1, 0), base_aff), str(tmp_path / "base.nii"))
    nib.save(nib.Nifti1Image(padded.transpose(2, 1, 0), src_aff), str(tmp_path / "src.nii"))
    return base_aff, src_aff, true_aff


def _solve(tmp_path, *extra):
    out = tmp_path / "out.aff12.1D"
    main(
        [
            "-base", str(tmp_path / "base.nii"),
            "-source", str(tmp_path / "src.nii"),
            "-prefix", str(tmp_path / "out.nii"),
            "-1Dmatrix_save", str(out),
            "-rigid",
            "-cost", "ls",
            "-smallrange",
            "-device", "cpu",
            "-final", "linear",
            "-verb", "0",
            *extra,
        ]
    )  # fmt: skip
    return out


def _error_mm(path, base_aff, src_aff, true_aff):
    """Worst disagreement with the true voxel map, in mm, over the base's box."""
    solved = load_matrix_1D(path, base_aff, src_aff).double().numpy()
    truth = np.linalg.inv(true_aff) @ base_aff
    corners = np.array([[i, j, k, 1.0] for i in (0, 31) for j in (0, 31) for k in (0, 31)]).T
    return np.linalg.norm((true_aff @ ((solved - truth) @ corners))[:3], axis=0).max()


def test_the_search_refines_from_the_given_start(tmp_path):
    base_aff, src_aff, true_aff = _pair(tmp_path, 24.0)
    # The world move that undoes the header error, knocked a little off.
    truth = true_aff @ np.linalg.inv(src_aff)
    nudge = align.compose((2.0, -1.5, 1.0, 3.0, 0.0, -2.0), (0.0, 0.0, 0.0))
    start = tmp_path / "start.aff12.1D"
    align.save_aff12(start, nudge @ truth, base_aff, src_aff)

    out = _solve(tmp_path, "-1Dmatrix_init", str(start))
    assert _error_mm(out, base_aff, src_aff, true_aff) < 1.0


def test_without_a_start_the_header_error_is_not_recovered(tmp_path):
    """The control: the case above really does need the start it is given."""
    base_aff, src_aff, true_aff = _pair(tmp_path, 24.0)
    out = _solve(tmp_path, "-nocmass")
    assert _error_mm(out, base_aff, src_aff, true_aff) > 10.0
