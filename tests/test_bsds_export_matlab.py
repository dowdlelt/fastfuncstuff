"""MATLAB export for cross-checking against the reference BSDS implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from fastfuncstuff.dynamics.bsds.export_matlab import write_matlab_bsds_input


def test_write_matlab_bsds_input_roundtrip(tmp_path):
    sessions = [torch.randn(5, 20, dtype=torch.float64) for _ in range(3)]
    mat_path = tmp_path / "bsds_input.mat"

    m_path = write_matlab_bsds_input(
        sessions, str(mat_path), max_nstates=4, max_ldim=2, n_iter=50, tol=1e-3
    )

    assert Path(m_path).exists()
    script = Path(m_path).read_text()
    assert "BayesianSwitchingDynamicalSystems" in script
    # The reference uses globals it never clears between runs; the generated
    # script must reset them or a smaller/second fit in one session corrupts.
    assert "clear all" in script

    loaded = loadmat(str(mat_path))
    assert loaded["data"].shape == (1, 3)
    for i, s in enumerate(sessions):
        np.testing.assert_allclose(loaded["data"][0, i], s.numpy())
    assert loaded["max_nstates"].item() == 4
    assert loaded["max_ldim"].item() == 2


def test_write_matlab_bsds_input_default_ldim(tmp_path):
    sessions = [torch.randn(6, 15, dtype=torch.float64)]
    mat_path = tmp_path / "bsds_input2.mat"
    write_matlab_bsds_input(sessions, str(mat_path), max_nstates=3)
    loaded = loadmat(str(mat_path))
    assert loaded["max_ldim"].item() == 5  # D - 1


def test_write_matlab_bsds_input_equalizes_lengths(tmp_path):
    # Unequal run lengths (390 vs 391, like real MDTB data) must be truncated to
    # the shortest so the reference MATLAB's cell2mat can load them.
    sessions = [torch.randn(5, 20, dtype=torch.float64), torch.randn(5, 18, dtype=torch.float64)]
    mat_path = tmp_path / "bsds_uneq.mat"
    write_matlab_bsds_input(sessions, str(mat_path), max_nstates=3, max_ldim=2)
    loaded = loadmat(str(mat_path))
    assert loaded["data"][0, 0].shape == (5, 18)
    assert loaded["data"][0, 1].shape == (5, 18)


def test_write_matlab_bsds_input_no_truncate_when_equal(tmp_path):
    sessions = [torch.randn(5, 20, dtype=torch.float64), torch.randn(5, 20, dtype=torch.float64)]
    mat_path = tmp_path / "bsds_eq.mat"
    write_matlab_bsds_input(sessions, str(mat_path), max_nstates=3, max_ldim=2)
    loaded = loadmat(str(mat_path))
    assert loaded["data"][0, 0].shape == (5, 20)
    assert loaded["data"][0, 1].shape == (5, 20)
