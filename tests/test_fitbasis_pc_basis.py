"""``ffs_fitbasis -hrf pcs``: data-derived temporal PCs as a joint basis.

The PCs from ``ffs_librarian`` are fitted TOGETHER, exactly as SPMG3's three
columns are — the difference is that the columns come from the data instead of
being a canonical curve and its derivatives.  So this rides the existing
multi-basis path and ``-derivatives`` does not apply.

The file's sampling is the thing that has to survive the hand-off: the SVD
produces PCs on the FIR lag grid, ffs_librarian resamples them onto the same
0.1 s grid its library TSVs use, and the header states which.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from fastfuncstuff.cli.fitbasis import _load_pc_basis

LAG = np.arange(0, 36, 2.0)


def _native_pcs(n_pc=3, seed=0):
    """PCs on the FIR lag grid, smooth enough to survive resampling."""
    rng = np.random.default_rng(seed)
    t = LAG
    pcs = np.stack(
        [
            np.sin((k + 1) * np.pi * t / t[-1]) + 0.05 * rng.standard_normal(t.size)
            for k in range(n_pc)
        ]
    )
    return pcs / np.linalg.norm(pcs, axis=1, keepdims=True)


def _write_fine(tmp_path, pcs, dt=0.1, duration=34.0, name="sub01_pcs.tsv"):
    """Write PCs the way ffs_librarian does: fine grid, dt_s header."""
    times = np.arange(int(np.floor(duration / dt)) + 1) * dt
    fine = np.stack(
        [
            PchipInterpolator(LAG, pcs[k])(np.clip(times, LAG[0], LAG[-1]))
            for k in range(pcs.shape[0])
        ]
    )
    path = tmp_path / name
    np.savetxt(
        path,
        fine.T,
        fmt="%.10g",
        delimiter="\t",
        header=(
            "ffs_librarian temporal PCs — rows = time samples, columns = PCs\n"
            f"dt_s: {dt:g}\nn_samples: {fine.shape[1]}"
        ),
    )
    return path


def test_pc_basis_loads_at_the_requested_grid(tmp_path):
    path = _write_fine(tmp_path, _native_pcs())
    for dt, duration in ((0.1, 34.0), (0.05, 34.0), (0.1, 30.0)):
        basis = _load_pc_basis(str(path), dt, duration)
        assert basis.shape == (3, int(np.floor(duration / dt)) + 1)
        np.testing.assert_allclose(np.linalg.norm(basis, axis=1), 1.0, atol=1e-10)


def test_pc_basis_round_trips_the_native_curves(tmp_path):
    # Resampling to 0.1 s and back to the lag grid must not change the PCs:
    # they are the object the library was built from, and a basis that drifts
    # from them is not testing what the user thinks it is.
    pcs = _native_pcs()
    basis = _load_pc_basis(str(_write_fine(tmp_path, pcs)), 0.1, 34.0)
    times = np.arange(basis.shape[1]) * 0.1
    for k in range(pcs.shape[0]):
        back = np.interp(LAG, times, basis[k])
        assert abs(np.corrcoef(back, pcs[k])[0, 1]) > 0.999


def test_lag_times_header_still_works(tmp_path):
    # The brief intermediate format, where PCs were written on the raw FIR grid
    # with their lag times in the header.
    pcs = _native_pcs()
    path = tmp_path / "sub01_pcs.tsv"
    np.savetxt(
        path,
        pcs.T,
        fmt="%.10g",
        delimiter="\t",
        header="lag_times_s: " + " ".join(f"{x:g}" for x in LAG),
    )
    basis = _load_pc_basis(str(path), 0.1, 34.0)
    assert basis.shape == (3, 341)


def test_headerless_file_falls_back_and_says_so(tmp_path, capsys):
    pcs = _native_pcs()
    path = tmp_path / "legacy_pcs.tsv"
    np.savetxt(path, pcs.T, fmt="%.10g", delimiter="\t")
    basis = _load_pc_basis(str(path), 0.1, 34.0)
    assert basis.shape == (3, 341)
    assert "no sampling header" in capsys.readouterr().out


def test_single_pc_file_is_accepted(tmp_path):
    basis = _load_pc_basis(str(_write_fine(tmp_path, _native_pcs(n_pc=1))), 0.1, 34.0)
    assert basis.shape[0] == 1


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_pc_basis(str(tmp_path / "nope.tsv"), 0.1, 34.0)


def test_header_length_mismatch_raises(tmp_path):
    pcs = _native_pcs()
    path = tmp_path / "bad_pcs.tsv"
    np.savetxt(path, pcs.T, fmt="%.10g", delimiter="\t", header="lag_times_s: 0 2 4")
    with pytest.raises(ValueError, match="lag times"):
        _load_pc_basis(str(path), 0.1, 34.0)


def test_raw_pcs_file_points_at_the_smooth_one(tmp_path, capsys):
    # _pcs.tsv interpolates the FIR-grid estimate exactly, noise included;
    # _pcs_smooth.tsv is the penalized-spline version and is the better basis.
    # The note fires without switching files: which one was asked for is the
    # caller's decision.
    pcs = _native_pcs()
    _write_fine(tmp_path, pcs, name="sub01_pcs.tsv")
    _write_fine(tmp_path, pcs, name="sub01_pcs_smooth.tsv")
    _load_pc_basis(str(tmp_path / "sub01_pcs.tsv"), 0.1, 34.0)
    assert "pcs_smooth" in capsys.readouterr().out


def test_no_note_when_the_smooth_file_is_absent(tmp_path, capsys):
    _write_fine(tmp_path, _native_pcs(), name="sub01_pcs.tsv")
    _load_pc_basis(str(tmp_path / "sub01_pcs.tsv"), 0.1, 34.0)
    assert "pcs_smooth" not in capsys.readouterr().out
