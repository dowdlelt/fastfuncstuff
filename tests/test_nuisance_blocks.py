"""
Tests for the NuisanceBlock infrastructure in cli_utils.py.

Covers the three CLI input modes (full-length / per-run-file / glob),
the regex-priority run-index inference, the assembler's per-run demean
contract, and back-compat for callers still passing the legacy
``ortvec_files=[(path, label), ...]`` argument.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.cli_utils import (
    NuisanceAssembly,
    NuisanceBlock,
    _infer_run_indices_from_filenames,
    assemble_per_run_nuisance,
    build_nuisance_block_diag,
    build_nuisance_per_run,
    make_nuisance_block_from_full_length,
    make_nuisance_block_from_glob,
    make_nuisance_block_from_per_run_file,
)


CPU = torch.device("cpu")


def _write_1d(path, arr: np.ndarray) -> None:
    np.savetxt(path, arr)


# ---------------------------------------------------------------------------
# NuisanceBlock semantics
# ---------------------------------------------------------------------------

class TestNuisanceBlock:
    def test_n_columns_uses_widest_run(self):
        b = NuisanceBlock(
            label="pcs",
            per_run=[np.zeros((10, 3)), np.zeros((10, 5)), None],
        )
        assert b.n_columns == 5

    def test_get_run_zero_pads_shorter_columns(self):
        b = NuisanceBlock(
            label="pcs",
            per_run=[np.ones((10, 3)), np.full((10, 5), 2.0)],
        )
        run0 = b.get_run(0, 10)
        assert run0.shape == (10, 5)
        # First 3 columns are 1.0, last 2 are 0.0 (padded)
        assert (run0[:, :3] == 1.0).all()
        assert (run0[:, 3:] == 0.0).all()

    def test_get_run_none_yields_zeros(self):
        b = NuisanceBlock(label="x", per_run=[None, np.ones((10, 4))])
        run0 = b.get_run(0, 10)
        assert run0.shape == (10, 4)
        assert (run0 == 0.0).all()

    def test_get_run_wrong_length_raises(self):
        b = NuisanceBlock(label="x", per_run=[np.zeros((10, 2))])
        with pytest.raises(ValueError, match="design expects"):
            b.get_run(0, 12)

    def test_column_names_default_to_label_prefix(self):
        b = NuisanceBlock(label="motion", per_run=[np.zeros((5, 6))])
        names = b.get_column_names()
        assert names == [f"motion_{i:02d}" for i in range(6)]

    def test_column_names_explicit_overrides(self):
        b = NuisanceBlock(
            label="motion",
            per_run=[np.zeros((5, 3))],
            column_names=["roll", "pitch", "yaw"],
        )
        assert b.get_column_names() == ["roll", "pitch", "yaw"]

    def test_source_length_must_match_per_run(self):
        with pytest.raises(ValueError, match="source has"):
            NuisanceBlock(
                label="x",
                per_run=[None, None, None],
                source=["a.txt", "b.txt"],
            )


# ---------------------------------------------------------------------------
# Factory: full-length
# ---------------------------------------------------------------------------

class TestMakeBlockFullLength:
    def test_slices_into_runs(self, tmp_path):
        arr = np.arange(20, dtype=np.float32).reshape(20, 1)
        p = tmp_path / "ortvec.1D"
        _write_1d(p, arr)
        b = make_nuisance_block_from_full_length(
            p, "x", run_starts=[0, 10], n_timepoints=20,
        )
        assert b.label == "x"
        assert b.per_run[0].shape == (10, 1)
        assert b.per_run[1].shape == (10, 1)
        np.testing.assert_array_equal(b.per_run[0].ravel(), np.arange(10))
        np.testing.assert_array_equal(b.per_run[1].ravel(), np.arange(10, 20))
        assert b.source == [str(p), str(p)]

    def test_wrong_total_raises(self, tmp_path):
        p = tmp_path / "wrong.1D"
        _write_1d(p, np.zeros((15, 1)))
        with pytest.raises(ValueError, match="total timepoints"):
            make_nuisance_block_from_full_length(
                p, "x", run_starts=[0, 10], n_timepoints=20,
            )


# ---------------------------------------------------------------------------
# Factory: per-run file
# ---------------------------------------------------------------------------

class TestMakeBlockPerRunFile:
    def test_lands_in_correct_slot_others_none(self, tmp_path):
        p = tmp_path / "run2.1D"
        _write_1d(p, np.full((10, 2), 7.0))
        b = make_nuisance_block_from_per_run_file(
            p, "motion", run_idx_1based=2, run_starts=[0, 10], n_timepoints=20,
        )
        assert b.per_run[0] is None
        assert b.per_run[1].shape == (10, 2)
        assert b.source[0] is None
        assert b.source[1] == str(p)
        # Other runs zero-pad at assembly via get_run.
        np.testing.assert_array_equal(
            b.get_run(0, 10),
            np.zeros((10, 2), dtype=np.float32),
        )

    def test_out_of_range_run_raises(self, tmp_path):
        p = tmp_path / "x.1D"
        _write_1d(p, np.zeros((10, 1)))
        with pytest.raises(ValueError, match="out of range"):
            make_nuisance_block_from_per_run_file(
                p, "x", run_idx_1based=5,
                run_starts=[0, 10], n_timepoints=20,
            )

    def test_wrong_row_count_raises(self, tmp_path):
        p = tmp_path / "x.1D"
        _write_1d(p, np.zeros((7, 1)))  # run 1 has 10 rows, file has 7
        with pytest.raises(ValueError, match="run 1 has"):
            make_nuisance_block_from_per_run_file(
                p, "x", run_idx_1based=1,
                run_starts=[0, 10], n_timepoints=20,
            )


# ---------------------------------------------------------------------------
# _infer_run_indices_from_filenames — priority + ambiguity
# ---------------------------------------------------------------------------

class TestRunIndexInference:
    def test_bids_pattern_with_underscore(self):
        names = ["sub-01_run-01_motion.1D", "sub-01_run-02_motion.1D"]
        assert _infer_run_indices_from_filenames(names, n_runs=2) == [0, 1]

    def test_bids_pattern_no_trailing_underscore(self):
        names = ["sub-01_run-01.1D", "sub-01_run-02.1D"]
        assert _infer_run_indices_from_filenames(names, n_runs=2) == [0, 1]

    def test_underscore_runN_pattern(self):
        names = ["noise_run1_pcs.txt", "noise_run2_pcs.txt"]
        assert _infer_run_indices_from_filenames(names, n_runs=2) == [0, 1]

    def test_trailing_number_with_extension(self):
        names = ["motion01.1D", "motion02.1D"]
        assert _infer_run_indices_from_filenames(names, n_runs=2) == [0, 1]

    def test_non_sequential_runs(self):
        """Glob can match a subset (e.g. runs 1, 2, 4 with 3 missing)."""
        names = ["noise_run-1.1D", "noise_run-2.1D", "noise_run-4.1D"]
        assert _infer_run_indices_from_filenames(names, n_runs=5) == [0, 1, 3]

    def test_unmatched_files_error_lists_them(self):
        names = ["abc.txt", "def.txt"]  # no numbers
        with pytest.raises(ValueError, match="Could not infer per-run"):
            _infer_run_indices_from_filenames(names, n_runs=2)

    def test_unresolvable_duplicates_rejected(self):
        """When EVERY pattern yields duplicates, the function gives up
        with a message that lists which patterns saw which duplicates."""
        # Both names contain only "01" as their only digit run — every
        # pattern resolves to the same number for both files.
        names = ["motion_run01.txt", "motion_run01.txt"]
        with pytest.raises(ValueError) as excinfo:
            _infer_run_indices_from_filenames(names, n_runs=2)
        msg = str(excinfo.value)
        assert "Could not infer" in msg
        assert "duplicate run numbers" in msg

    def test_priority_takes_first_consistent_interpretation(self):
        """Two names where one pattern fails (duplicates) but a later
        pattern succeeds — the function should take the later one rather
        than reject. Demonstrates the priority-list philosophy."""
        # _run01.txt: every "run01" pattern resolves to 1.
        # _run01_take2.txt: "_run01_" pattern resolves to 1 (same → dup),
        #   but "trailing-number-before-extension" resolves to 2 (distinct).
        names = ["motion_run01.txt", "motion_run01_take2.txt"]
        # Some valid assignment is found; the exact mapping depends on
        # which pattern wins, but we should not raise.
        out = _infer_run_indices_from_filenames(names, n_runs=2)
        assert sorted(out) == [0, 1]

    def test_out_of_range_rejected(self):
        names = ["noise_run-1.1D", "noise_run-9.1D"]  # but n_runs=3
        with pytest.raises(ValueError, match="out-of-range"):
            _infer_run_indices_from_filenames(names, n_runs=3)

    def test_case_insensitive(self):
        names = ["sub-01_RUN-01_motion.1D", "sub-01_RUN-02_motion.1D"]
        assert _infer_run_indices_from_filenames(names, n_runs=2) == [0, 1]


# ---------------------------------------------------------------------------
# Factory: glob
# ---------------------------------------------------------------------------

class TestMakeBlockFromGlob:
    def test_glob_assembles_block_with_zero_runs_in_between(self, tmp_path):
        # 3 runs but only files for runs 1 and 3
        for r in (1, 3):
            _write_1d(tmp_path / f"noise_run-{r:02d}.1D", np.full((10, 2), float(r)))
        pattern = str(tmp_path / "noise_run-*.1D")
        b = make_nuisance_block_from_glob(
            pattern, "noisepcs",
            run_starts=[0, 10, 20], n_timepoints=30,
        )
        assert b.per_run[0] is not None
        assert b.per_run[1] is None     # run 2 absent → None → zero-padded at use
        assert b.per_run[2] is not None
        # And the contents are right.
        assert (b.per_run[0] == 1.0).all()
        assert (b.per_run[2] == 3.0).all()
        # Source provenance reflects which runs came from which file.
        assert b.source[0].endswith("noise_run-01.1D")
        assert b.source[1] is None
        assert b.source[2].endswith("noise_run-03.1D")

    def test_glob_with_no_matches_errors(self, tmp_path):
        with pytest.raises(ValueError, match="matched no files"):
            make_nuisance_block_from_glob(
                str(tmp_path / "nothing_*.1D"), "x",
                run_starts=[0, 10], n_timepoints=20,
            )

    def test_glob_with_mismatched_length_errors(self, tmp_path):
        _write_1d(tmp_path / "noise_run-1.1D", np.zeros((7, 1)))  # should be 10
        with pytest.raises(ValueError, match="run 1 has"):
            make_nuisance_block_from_glob(
                str(tmp_path / "noise_run-*.1D"), "x",
                run_starts=[0, 10], n_timepoints=20,
            )


# ---------------------------------------------------------------------------
# Assembler: demean contract + variable-cols handling + noise_pcs path
# ---------------------------------------------------------------------------

class TestAssemble:
    def test_polort_only_no_blocks(self):
        a = assemble_per_run_nuisance(
            blocks=[],
            run_starts=[0, 10],
            n_timepoints=20,
            polort=2,
            device=CPU,
        )
        assert len(a.per_run) == 2
        assert a.per_run[0].shape == (10, 3)  # polort=2 → 3 columns
        # Column names: r01_poly0..2, r02_poly0..2 (per-run polort labels).
        assert a.per_run_column_names[0] == ["r01_poly0", "r01_poly1", "r01_poly2"]

    def test_demeans_per_run(self):
        # Block whose per-run matrices have nonzero per-run mean.
        per_run = [np.full((10, 2), 5.0, dtype=np.float32),
                   np.full((10, 2), -3.0, dtype=np.float32)]
        block = NuisanceBlock(label="motion", per_run=per_run)
        a = assemble_per_run_nuisance(
            blocks=[block],
            run_starts=[0, 10],
            n_timepoints=20,
            polort=-1,  # no polynomials to isolate the demean check
            device=CPU,
        )
        # After demean, each run's nuisance columns should be (near) zero.
        for run_tensor in a.per_run:
            col_means = run_tensor.mean(dim=0)
            np.testing.assert_allclose(
                col_means.numpy(), np.zeros_like(col_means.numpy()), atol=1e-6,
            )

    def test_variable_cols_per_run(self):
        # Run 1 has 3 cols, run 2 has 5 — block has n_columns=5 and pads run 1.
        block = NuisanceBlock(
            label="pcs",
            per_run=[np.ones((10, 3), dtype=np.float32),
                     np.full((10, 5), 2.0, dtype=np.float32)],
        )
        a = assemble_per_run_nuisance(
            blocks=[block],
            run_starts=[0, 10],
            n_timepoints=20,
            polort=-1,
            device=CPU,
        )
        # Each run's nuisance tensor must have 5 nuisance columns.
        assert a.per_run[0].shape[1] == 5
        assert a.per_run[1].shape[1] == 5
        # The padded columns in run 1 (cols 3, 4) were zeros pre-demean and
        # stay zeros post-demean. Constant cols 0-2 demean to zero too.
        np.testing.assert_allclose(a.per_run[0].numpy(), np.zeros((10, 5)), atol=1e-6)

    def test_noise_pcs_path_still_works(self):
        """Legacy noise_pcs argument: gets concatenated per-run without demean."""
        noise_pcs = [
            torch.ones(10, 4, dtype=torch.float32),
            torch.full((10, 4), 2.0, dtype=torch.float32),
        ]
        a = assemble_per_run_nuisance(
            blocks=[],
            run_starts=[0, 10],
            n_timepoints=20,
            polort=-1,
            device=CPU,
            noise_pcs=noise_pcs,
        )
        # noise_pcs append after blocks; the values pass through unmodified
        # (PCs are presumed already polynomial-projected upstream).
        assert torch.equal(a.per_run[0], noise_pcs[0])
        assert torch.equal(a.per_run[1], noise_pcs[1])


# ---------------------------------------------------------------------------
# Back-compat: legacy ortvec_files / ortvec_data arguments
# ---------------------------------------------------------------------------

class TestBackCompatLegacyArgs:
    def test_build_nuisance_per_run_ortvec_files(self, tmp_path):
        # Same shape as a real user-supplied full-length motion file.
        arr = np.random.default_rng(0).standard_normal((20, 6)).astype(np.float32)
        p = tmp_path / "motion.1D"
        np.savetxt(p, arr)
        nuisance_per_run = build_nuisance_per_run(
            run_starts=[0, 10],
            n_timepoints=20,
            polort=-1,
            device=CPU,
            ortvec_files=[(str(p), "motion")],
        )
        assert len(nuisance_per_run) == 2
        # Demean contract: each per-run output equals input slice minus its mean.
        for i, m in enumerate(nuisance_per_run):
            start = i * 10
            expected = arr[start:start + 10]
            expected = expected - expected.mean(axis=0, keepdims=True)
            np.testing.assert_allclose(m.numpy(), expected, atol=1e-5)

    def test_build_nuisance_block_diag_legacy_ortvec(self, tmp_path):
        arr = np.random.default_rng(1).standard_normal((20, 3)).astype(np.float32)
        p = tmp_path / "ortvec.1D"
        np.savetxt(p, arr)
        out = build_nuisance_block_diag(
            run_starts=[0, 10],
            n_timepoints=20,
            polort=1,
            device=CPU,
            ortvec_files=[(str(p), "phys")],
        )
        # polort=1 → 2 cols per run (block-diag) → 4 cols, plus 3 ortvec cols.
        assert out.shape == (20, 4 + 3)
        # The ortvec part (last 3 cols) should be demeaned globally
        # (shared-across-runs layout).
        ortvec_part = out[:, -3:].numpy()
        np.testing.assert_allclose(
            ortvec_part.mean(axis=0),
            np.zeros(3),
            atol=1e-5,
        )
