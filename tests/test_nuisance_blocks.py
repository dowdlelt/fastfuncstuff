"""
Tests for the NuisanceBlock infrastructure in cli_utils.py.

Covers the three CLI input modes (full-length / per-run-file / glob),
the regex-priority run-index inference, the assembler's per-run demean
contract, and back-compat for callers still passing the legacy
``ortvec_files=[(path, label), ...]`` argument.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from fastfuncstuff.cli_utils import (
    NuisanceBlock,
    _infer_run_indices_from_filenames,
    add_ortvec_arguments,
    append_nuisance_blocks,
    apply_nuisance_transform,
    assemble_per_run_nuisance,
    build_nuisance_block_diag,
    build_nuisance_per_run,
    collect_nuisance_blocks,
    make_nuisance_block_from_full_length,
    make_nuisance_block_from_glob,
    make_nuisance_block_from_per_run_file,
    split_label_transform,
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
            p,
            "x",
            run_starts=[0, 10],
            n_timepoints=20,
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
                p,
                "x",
                run_starts=[0, 10],
                n_timepoints=20,
            )


# ---------------------------------------------------------------------------
# Factory: per-run file
# ---------------------------------------------------------------------------


class TestMakeBlockPerRunFile:
    def test_lands_in_correct_slot_others_none(self, tmp_path):
        p = tmp_path / "run2.1D"
        _write_1d(p, np.full((10, 2), 7.0))
        b = make_nuisance_block_from_per_run_file(
            p,
            "motion",
            run_idx_1based=2,
            run_starts=[0, 10],
            n_timepoints=20,
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
                p,
                "x",
                run_idx_1based=5,
                run_starts=[0, 10],
                n_timepoints=20,
            )

    def test_wrong_row_count_raises(self, tmp_path):
        p = tmp_path / "x.1D"
        _write_1d(p, np.zeros((7, 1)))  # run 1 has 10 rows, file has 7
        with pytest.raises(ValueError, match="run 1 has"):
            make_nuisance_block_from_per_run_file(
                p,
                "x",
                run_idx_1based=1,
                run_starts=[0, 10],
                n_timepoints=20,
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

    def test_short_rN_token(self):
        """AFNI-style `.r001_` short run token resolves without BIDS naming."""
        names = [f"nlmoco.r00{i}_locomoco_pcs.1D" for i in range(1, 7)]
        assert _infer_run_indices_from_filenames(names, n_runs=6) == [0, 1, 2, 3, 4, 5]

    def test_token_beats_lexical_sort_order(self):
        """Non-zero-padded r1..r10 sort as r1,r10,r2… lexically; the token
        parser must still map by numeric value, not sorted position."""
        names = sorted(f"motion_r{i}.1D" for i in range(1, 11))
        out = _infer_run_indices_from_filenames(names, n_runs=10, allow_sequential_fallback=True)
        # names[0] is motion_r1 (run 1 → idx 0), names[1] is motion_r10 (idx 9).
        assert out[0] == 0
        assert out[1] == 9

    def test_sequential_fallback_when_count_matches(self):
        """No parseable token, but one file per run → trust sorted order."""
        names = ["confounds_A.txt", "confounds_B.txt", "confounds_C.txt"]
        assert _infer_run_indices_from_filenames(
            names, n_runs=3, allow_sequential_fallback=True
        ) == [0, 1, 2]

    def test_sequential_fallback_disabled_by_default(self):
        names = ["confounds_A.txt", "confounds_B.txt", "confounds_C.txt"]
        with pytest.raises(ValueError, match="Could not infer per-run"):
            _infer_run_indices_from_filenames(names, n_runs=3)

    def test_sequential_fallback_requires_count_match(self):
        """Untokenised files with a count mismatch stay an error — sorted
        order can't tell which runs are present."""
        names = ["confounds_A.txt", "confounds_B.txt"]
        with pytest.raises(ValueError, match="counts must match"):
            _infer_run_indices_from_filenames(names, n_runs=3, allow_sequential_fallback=True)


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
            pattern,
            "noisepcs",
            run_starts=[0, 10, 20],
            n_timepoints=30,
        )
        assert b.per_run[0] is not None
        assert b.per_run[1] is None  # run 2 absent → None → zero-padded at use
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
                str(tmp_path / "nothing_*.1D"),
                "x",
                run_starts=[0, 10],
                n_timepoints=20,
            )

    def test_glob_with_mismatched_length_errors(self, tmp_path):
        _write_1d(tmp_path / "noise_run-1.1D", np.zeros((7, 1)))  # should be 10
        with pytest.raises(ValueError, match="run 1 has"):
            make_nuisance_block_from_glob(
                str(tmp_path / "noise_run-*.1D"),
                "x",
                run_starts=[0, 10],
                n_timepoints=20,
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
        per_run = [
            np.full((10, 2), 5.0, dtype=np.float32),
            np.full((10, 2), -3.0, dtype=np.float32),
        ]
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
                col_means.numpy(),
                np.zeros_like(col_means.numpy()),
                atol=1e-6,
            )

    def test_variable_cols_per_run(self):
        # Run 1 has 3 cols, run 2 has 5 — block has n_columns=5 and pads run 1.
        block = NuisanceBlock(
            label="pcs",
            per_run=[np.ones((10, 3), dtype=np.float32), np.full((10, 5), 2.0, dtype=np.float32)],
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
            expected = arr[start : start + 10]
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

    def test_block_diagonal_glob_expands_per_run(self):
        """A per-run (block_diagonal) block must get its OWN columns per run
        (zero outside that run), not collapse to shared columns."""
        rng = np.random.default_rng(2)
        r0 = rng.standard_normal((10, 3)).astype(np.float32) + 5.0  # non-zero mean
        r1 = rng.standard_normal((10, 3)).astype(np.float32) - 2.0
        block = NuisanceBlock(
            label="locomoco",
            per_run=[r0, r1],
            source=["r0.1D", "r1.1D"],
            block_diagonal=True,
        )
        out = build_nuisance_block_diag(
            run_starts=[0, 10],
            n_timepoints=20,
            polort=1,
            device=CPU,
            blocks=[block],
        )
        # polort=1 → 2 cols/run (block-diag) = 4, plus 2 runs × 3 = 6 block-diag ortvec.
        assert out.shape == (20, 4 + 6)
        ov = out[:, 4:].numpy()  # (20, 6): cols 0-2 = run0's block, 3-5 = run1's block
        # Run 0's columns are zero during run 1 and vice versa (block-diagonal).
        np.testing.assert_allclose(ov[10:, 0:3], 0.0, atol=1e-6)
        np.testing.assert_allclose(ov[:10, 3:6], 0.0, atol=1e-6)
        # Each run's block is per-run demeaned within its own rows.
        np.testing.assert_allclose(ov[:10, 0:3].mean(axis=0), np.zeros(3), atol=1e-5)
        np.testing.assert_allclose(ov[10:, 3:6].mean(axis=0), np.zeros(3), atol=1e-5)

    def test_block_diagonal_per_run_files_are_block_diag(self, tmp_path):
        """Two -ortvec_run-style single-run blocks land in disjoint run columns."""
        a = np.random.default_rng(3).standard_normal((10, 2)).astype(np.float32)
        b = np.random.default_rng(4).standard_normal((10, 2)).astype(np.float32)
        pa, pb = tmp_path / "a.1D", tmp_path / "b.1D"
        np.savetxt(pa, a)
        np.savetxt(pb, b)
        blk_a = make_nuisance_block_from_per_run_file(str(pa), "mot", 1, [0, 10], 20)
        blk_b = make_nuisance_block_from_per_run_file(str(pb), "mot", 2, [0, 10], 20)
        assert blk_a.block_diagonal and blk_b.block_diagonal
        out = build_nuisance_block_diag(
            run_starts=[0, 10], n_timepoints=20, polort=-1, device=CPU, blocks=[blk_a, blk_b]
        )
        # polort=-1 → no poly; 2 blocks × 2 cols = 4, each zero outside its run.
        assert out.shape == (20, 4)
        np.testing.assert_allclose(out[10:, 0:2].numpy(), 0.0, atol=1e-6)  # block a: run1 zero
        np.testing.assert_allclose(out[:10, 2:4].numpy(), 0.0, atol=1e-6)  # block b: run0 zero


# ---------------------------------------------------------------------------
# CLI surface: add_ortvec_arguments + collect_nuisance_blocks
# ---------------------------------------------------------------------------


class TestCliSurface:
    def _parser(self):
        p = argparse.ArgumentParser()
        add_ortvec_arguments(p)
        return p

    def test_parser_registers_three_flags(self):
        p = self._parser()
        args = p.parse_args(
            [
                "-ortvec",
                "a.1D",
                "phys",
                "-ortvec_run",
                "b.1D",
                "motion",
                "1",
                "-ortvec_glob",
                "mot_run-*.1D",
                "motion",
            ]
        )
        assert args.ortvec == [["a.1D", "phys"]]
        assert args.ortvec_run == [["b.1D", "motion", "1"]]
        assert args.ortvec_glob == [["mot_run-*.1D", "motion"]]

    def test_parser_dash_aliases(self):
        p = self._parser()
        # Dash form of the long names also resolves.
        args = p.parse_args(["-ortvec-run", "x.1D", "mot", "2"])
        assert args.ortvec_run == [["x.1D", "mot", "2"]]

    def test_collect_blocks_all_three_modes(self, tmp_path):
        # Mode 1: full-length file
        full = np.random.default_rng(0).standard_normal((20, 2)).astype(np.float32)
        full_path = tmp_path / "full.1D"
        np.savetxt(full_path, full)

        # Mode 2: per-run file (run 1, length 10)
        per_run_arr = np.random.default_rng(1).standard_normal((10, 1)).astype(np.float32)
        per_run_path = tmp_path / "phys_run1.1D"
        np.savetxt(per_run_path, per_run_arr)

        # Mode 3: glob (two BIDS-style files, run 1 and run 2)
        for i in (1, 2):
            arr = np.random.default_rng(10 + i).standard_normal((10, 3)).astype(np.float32)
            np.savetxt(tmp_path / f"mot_run-0{i}_motion.1D", arr)

        p = self._parser()
        args = p.parse_args(
            [
                "-ortvec",
                str(full_path),
                "phys_full",
                "-ortvec_run",
                str(per_run_path),
                "phys_r1",
                "1",
                "-ortvec_glob",
                str(tmp_path / "mot_run-*_motion.1D"),
                "mot",
            ]
        )
        blocks = collect_nuisance_blocks(args, run_starts=[0, 10], n_timepoints=20)
        assert [b.label for b in blocks] == ["phys_full", "phys_r1", "mot"]
        assert blocks[0].n_columns == 2  # full-length 2-col
        assert blocks[1].n_columns == 1  # per-run 1-col
        assert blocks[2].n_columns == 3  # glob 3-col
        # Per-run-only block has None for run 2
        assert blocks[1].per_run[0] is not None
        assert blocks[1].per_run[1] is None
        # Glob block populates both runs
        assert blocks[2].per_run[0] is not None
        assert blocks[2].per_run[1] is not None

    def test_collect_blocks_no_flags_returns_empty(self):
        p = self._parser()
        args = p.parse_args([])
        assert collect_nuisance_blocks(args, [0, 10], 20) == []

    def test_collect_blocks_bad_run_index_exits(self, tmp_path):
        arr = np.zeros((10, 1), dtype=np.float32)
        path = tmp_path / "x.1D"
        np.savetxt(path, arr)
        p = self._parser()
        args = p.parse_args(["-ortvec_run", str(path), "mot", "notanumber"])
        with pytest.raises(SystemExit):
            collect_nuisance_blocks(args, [0, 10], 20)


# ---------------------------------------------------------------------------
# blocks= passthrough through the two builders
# ---------------------------------------------------------------------------


class TestBuilderBlocksKwarg:
    def test_build_per_run_with_blocks(self, tmp_path):
        arr = np.random.default_rng(0).standard_normal((20, 2)).astype(np.float32)
        p = tmp_path / "motion.1D"
        np.savetxt(p, arr)
        block = make_nuisance_block_from_full_length(
            str(p),
            "motion",
            run_starts=[0, 10],
            n_timepoints=20,
        )
        out = build_nuisance_per_run(
            run_starts=[0, 10],
            n_timepoints=20,
            polort=0,
            device=CPU,
            blocks=[block],
        )
        # polort=0 → 1 column per run + 2 motion → 3 total per run
        assert all(m.shape[1] == 3 for m in out)

    def test_build_block_diag_per_run_block_zero_pads_missing(self, tmp_path):
        # Per-run block for run 1 only → block-diagonal: nonzero in run 1's rows,
        # exactly zero in run 2's rows (and per-run demeaned within run 1).
        arr = np.arange(10, dtype=np.float32).reshape(10, 1)  # non-constant
        path = tmp_path / "r1.1D"
        np.savetxt(path, arr)
        block = make_nuisance_block_from_per_run_file(
            str(path),
            "mot",
            run_idx_1based=1,
            run_starts=[0, 10],
            n_timepoints=20,
        )
        out = build_nuisance_block_diag(
            run_starts=[0, 10],
            n_timepoints=20,
            polort=-1,
            device=CPU,
            blocks=[block],
        )
        # No polys; one column from the block, occupying only run 1 (block-diag).
        assert out.shape == (20, 1)
        col = out[:, 0].numpy()
        np.testing.assert_allclose(col[10:], 0.0, atol=1e-6)  # run 2 rows are zero
        np.testing.assert_allclose(col[:10].mean(), 0.0, atol=1e-6)  # per-run demeaned
        # Shape preserved within run 1 (demean is just a constant shift).
        np.testing.assert_allclose(col[:10], arr[:, 0] - arr[:, 0].mean(), atol=1e-5)


# ---------------------------------------------------------------------------
# Per-run transforms (deriv)
# ---------------------------------------------------------------------------


class TestNuisanceTransforms:
    def test_deriv_matches_1d_tool_semantics(self):
        """Backward difference with a zero first row, length preserved — this is
        exactly afni_util.derivative(direct=0), which 1d_tool.py -derivative uses."""
        v = np.array([[1.0], [3.0], [6.0], [10.0]])
        back = apply_nuisance_transform(v, "deriv")
        np.testing.assert_allclose(back.ravel(), [0, 2, 3, 4])
        np.testing.assert_allclose(apply_nuisance_transform(v, "deriv_back"), back)
        # Forward difference: same values shifted one row, zero at the END.
        fwd = apply_nuisance_transform(v, "deriv_fwd")
        np.testing.assert_allclose(fwd.ravel(), [2, 3, 4, 0])
        assert back.shape == fwd.shape == v.shape
        assert apply_nuisance_transform(v, "none") is v

    def test_unknown_transform_is_rejected(self):
        with pytest.raises(ValueError, match="unknown nuisance transform"):
            apply_nuisance_transform(np.zeros((4, 1)), "integral")
        with pytest.raises(ValueError, match="unknown transform"):
            split_label_transform("motion:integral")

    def test_deriv_never_crosses_a_run_boundary(self, tmp_path):
        """The bug this guards: differencing a concatenated file across runs turns
        the between-run offset into a spike in the regressor."""
        run1 = np.arange(10, dtype=float).reshape(10, 1)
        run2 = 1000 + np.arange(10, dtype=float).reshape(10, 1)  # huge offset
        f1, f2 = tmp_path / "m_run-01.1D", tmp_path / "m_run-02.1D"
        _write_1d(f1, run1)
        _write_1d(f2, run2)

        block = make_nuisance_block_from_glob(
            str(tmp_path / "m_run-*.1D"), "motion", [0, 10], 20, transform="deriv"
        )
        r2 = block.get_run(1, 10)
        assert r2[0, 0] == 0.0  # run 2 starts fresh, no 991 spike
        np.testing.assert_allclose(r2[1:, 0], 1.0)

    def test_label_modifier_flows_through_the_cli(self, tmp_path):
        """`-ortvec_glob PATTERN motion:deriv` is the one syntax that works for
        every ortvec mode, so the same file can enter raw and differenced."""
        for r in (1, 2):
            _write_1d(tmp_path / f"m_run-0{r}.1D", np.arange(10, dtype=float).reshape(10, 1))
        parser = argparse.ArgumentParser()
        add_ortvec_arguments(parser)
        pattern = str(tmp_path / "m_run-*.1D")
        args = parser.parse_args(
            ["-ortvec_glob", pattern, "motion", "-ortvec_glob", pattern, "motion_d:deriv"]
        )
        blocks = collect_nuisance_blocks(args, [0, 10], 20)
        assert [(b.label, b.transform) for b in blocks] == [
            ("motion", "none"),
            ("motion_d", "deriv"),
        ]
        np.testing.assert_allclose(blocks[1].get_run(0, 10)[:, 0], [0] + [1] * 9)


# ---------------------------------------------------------------------------
# Prefixed flag family (-test_ortvec …) and the shared append helper
# ---------------------------------------------------------------------------


class TestPrefixedOrtvecFamily:
    """`ffs_denoisatorial` carries two independent nuisance sets — one for the
    input runs, one for `-test_input` — so the flags and the collector must
    stay separable."""

    def _parser(self):
        p = argparse.ArgumentParser()
        add_ortvec_arguments(p)
        add_ortvec_arguments(p, prefix="test_")
        return p

    def test_both_families_coexist(self, tmp_path):
        p = self._parser()
        args = p.parse_args(["-ortvec", "a.1D", "phys", "-test_ortvec", "b.1D", "phys"])
        assert args.ortvec == [["a.1D", "phys"]]
        assert args.test_ortvec == [["b.1D", "phys"]]

    def test_prefixed_dash_aliases(self):
        p = self._parser()
        args = p.parse_args(["-test-ortvec-run", "x.1D", "mot", "2"])
        assert args.test_ortvec_run == [["x.1D", "mot", "2"]]
        assert args.ortvec_run is None

    def test_collect_selects_only_its_own_family(self, tmp_path):
        train = tmp_path / "train.1D"
        test = tmp_path / "test.1D"
        _write_1d(train, np.arange(20, dtype=float).reshape(20, 1))
        _write_1d(test, np.arange(10, dtype=float).reshape(10, 1))

        p = self._parser()
        args = p.parse_args(["-ortvec", str(train), "mot", "-test_ortvec", str(test), "mot"])
        # Different datasets: 2 runs of 10 for the input, 1 run of 10 held out.
        train_blocks = collect_nuisance_blocks(args, [0, 10], 20)
        test_blocks = collect_nuisance_blocks(args, [0], 10, prefix="test_")

        assert [b.source[0] for b in train_blocks] == [str(train)]
        assert [b.source[0] for b in test_blocks] == [str(test)]
        assert len(train_blocks[0].per_run) == 2
        assert len(test_blocks[0].per_run) == 1

    def test_no_prefixed_flags_yields_no_blocks(self):
        p = self._parser()
        args = p.parse_args(["-ortvec", "a.1D", "phys"])
        assert collect_nuisance_blocks(args, [0], 10, prefix="test_") == []


class TestAppendNuisanceBlocks:
    def test_columns_are_demeaned_and_padded_to_equal_width(self, tmp_path):
        # Run 0 gets a block, run 1 does not: the widths must still match, or
        # the block-diagonal builders downstream see a ragged stack.
        f = tmp_path / "m_run-01.1D"
        _write_1d(f, (100.0 + np.arange(10, dtype=float)).reshape(10, 1))
        block = make_nuisance_block_from_glob(str(tmp_path / "m_run-*.1D"), "motion", [0, 10], 20)
        nuisance = [torch.ones(10, 2), torch.ones(10, 2)]

        out = append_nuisance_blocks(nuisance, [block], [0, 10], 20)

        assert out[0].shape == out[1].shape == (10, 3)
        assert abs(float(out[0][:, 2].mean())) < 1e-5  # demeaned, not 104.5
        np.testing.assert_allclose(out[1][:, 2].numpy(), 0.0, atol=1e-6)

    def test_no_blocks_is_a_no_op(self):
        nuisance = [torch.ones(10, 2), torch.ones(10, 2)]
        out = append_nuisance_blocks(nuisance, [], [0, 10], 20)
        assert out[0].shape == (10, 2)
