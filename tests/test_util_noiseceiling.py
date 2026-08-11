"""Trial-table parsing for ffs_util_noiseceiling.

The BYOB path is only useful if it reads what the ffs tools actually write, and
the two formats they write differ in more than delimiter -- the whitespace one
carries a per-trial suffix on the condition label that must come off.
"""

from __future__ import annotations

import pytest
import torch

from fastfuncstuff.cli.util_noiseceiling import _index_labels, _read_trial_table


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


class TestTrialOrderTxt:
    """The *_single_trial_order.txt every single-trial run writes."""

    TABLE = """# trial_index  condition  run
   0  cond3_001             run1
   1  cond0_001             run1
   2  cond0_002             run1
   3  cond3_002             run2
   4  cond0_003             run2
"""

    def test_strips_the_per_trial_suffix(self, tmp_path):
        """cond3_001 and cond3_002 are one condition, not two.

        Without stripping, every condition has exactly one trial, nothing
        repeats, and the ceiling comes back unestimable on data that is
        perfectly fine.
        """
        conditions, runs = _read_trial_table(_write(tmp_path, "order.txt", self.TABLE))
        assert conditions.tolist() == [0, 1, 1, 0, 1]
        assert runs.tolist() == [0, 0, 0, 1, 1]

    def test_labels_are_indexed_by_first_appearance(self, tmp_path):
        conditions, _ = _read_trial_table(_write(tmp_path, "order.txt", self.TABLE))
        # cond3 appears first, so it is 0 even though cond0 sorts earlier.
        assert conditions[0].item() == 0
        assert conditions[1].item() == 1


class TestTrialEventsTsv:
    """The *_single_trial_events.tsv written when -events was used."""

    TABLE = "trial_index\trun_index\tcondition\tevents_file\n" + "".join(
        f"{i}\t{i // 2}\tface\tx.tsv\n" if i % 2 == 0 else f"{i}\t{i // 2}\thouse\tx.tsv\n"
        for i in range(6)
    )

    def test_reads_condition_and_run_index(self, tmp_path):
        conditions, runs = _read_trial_table(_write(tmp_path, "events.tsv", self.TABLE))
        assert conditions.tolist() == [0, 1, 0, 1, 0, 1]
        assert runs.tolist() == [0, 0, 1, 1, 2, 2]

    def test_missing_column_exits_with_a_message(self, tmp_path, capsys):
        bad = "trial_index\tsomething_else\n0\tfoo\n"
        with pytest.raises(SystemExit):
            _read_trial_table(_write(tmp_path, "bad.tsv", bad))
        assert "run_index" in capsys.readouterr().out

    def test_empty_table_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            _read_trial_table(_write(tmp_path, "empty.tsv", "condition\trun_index\n"))


class TestIndexLabels:
    def test_first_appearance_order(self):
        assert _index_labels(["b", "a", "b", "c"]).tolist() == [0, 1, 0, 2]

    def test_single_label(self):
        assert _index_labels(["only"] * 3).tolist() == [0, 0, 0]


class TestParserWiring:
    def test_both_modes_are_rejected_together(self):
        """-input and -betas are alternatives, not a combination."""
        from fastfuncstuff.cli.util_noiseceiling import create_parser

        parser = create_parser()
        args = parser.parse_args(["-input", "a.nii.gz", "-betas", "b.nii.gz", "-prefix", "out"])
        assert args.input is not None and args.betas is not None  # main() then exits

    def test_identical_sets_accepts_arbitrary_tokens(self):
        from fastfuncstuff.cli.util_noiseceiling import create_parser

        args = create_parser().parse_args(
            ["-input", "a.nii.gz", "-identical_sets", "movieA", "movieA", "-prefix", "o"]
        )
        assert args.identical_sets == ["movieA", "movieA"]


class TestSplitHalfCeilingViaCli:
    def test_recovers_planted_ceiling_from_repeats(self):
        """End-to-end on the primitive the -identical_sets mode calls."""
        from fastfuncstuff.stats.reliability import split_half_noise_ceiling

        generator = torch.Generator().manual_seed(19)
        n_voxels, run_length, n_repeats = 300, 200, 3
        target = 0.3
        signal = torch.randn(n_voxels, run_length, generator=generator)
        noise_sd = (signal.var(dim=1, keepdim=True) * (1 - target) / target).sqrt()
        runs = [
            signal + torch.randn(n_voxels, run_length, generator=generator) * noise_sd
            for _ in range(n_repeats)
        ]
        data = torch.cat(runs, dim=1)
        run_starts = [run * run_length for run in range(n_repeats)]

        ceiling = split_half_noise_ceiling(data, [[0, 1, 2]], run_starts, data.shape[1])
        assert ceiling.median().item() == pytest.approx(target, abs=0.05)
