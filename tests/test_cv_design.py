"""Tests for the -single_trials / -cv_design split (cli_utils).

The bug these guard against is silent, not loud: beta-space CV happily returns
numbers for a design where no condition repeats across runs, and those numbers
are meaningless because a held-out trial has no same-condition training trial to
be scored against.
"""

import numpy as np
import pytest

from fastfuncstuff.cli_utils import (
    resolve_cv_design,
    summarize_trial_repeats,
)


def _onsets(per_condition_per_run):
    """[cond][run] -> ndarray of onsets, from [cond][run] -> count."""
    return [
        [np.arange(n, dtype=float) * 10.0 for n in per_run] for per_run in per_condition_per_run
    ]


class TestSummarizeTrialRepeats:
    def test_repeated_conditions_are_predictable(self):
        # 3 conditions x 4 runs, 5 trials each
        repeats = summarize_trial_repeats(_onsets([[5, 5, 5, 5]] * 3))
        assert repeats.n_trials == 60
        assert repeats.n_conditions == 3
        assert repeats.n_repeated_conditions == 3
        assert repeats.predictable_fraction == 1.0
        assert repeats.n_runs == 4

    def test_unique_trials_have_no_target(self):
        # Every "condition" is a one-off stimulus confined to a single run —
        # the pathological case for beta-space CV.
        onsets = []
        for run in range(4):
            for _ in range(10):
                per_run = [[] for _ in range(4)]
                per_run[run] = [0.0]
                onsets.append([np.asarray(x, dtype=float) for x in per_run])
        repeats = summarize_trial_repeats(onsets)
        assert repeats.n_trials == 40
        assert repeats.n_repeated_conditions == 0
        assert repeats.predictable_fraction == 0.0

    def test_condition_repeated_within_one_run_only(self):
        # Many trials, but all in run 0: still nothing to cross-validate against.
        repeats = summarize_trial_repeats(_onsets([[20, 0, 0]]))
        assert repeats.n_trials == 20
        assert repeats.runs_per_condition == [1]
        assert repeats.predictable_fraction == 0.0

    def test_mixed_design_fraction(self):
        # cond0: 10 trials across 2 runs (predictable); cond1: 30 in one run.
        repeats = summarize_trial_repeats(_onsets([[5, 5], [30, 0]]))
        assert repeats.n_trials == 40
        assert repeats.n_predictable_trials == 10
        assert repeats.predictable_fraction == pytest.approx(0.25)

    def test_empty_design(self):
        repeats = summarize_trial_repeats([])
        assert repeats.n_trials == 0
        assert repeats.predictable_fraction == 0.0


def _resolve(requested, single_trials, repeats, **kw):
    """resolve_cv_design with the ffs_denoise policy (both designs are CV)."""
    kw.setdefault("parameter", "PC count")
    kw.setdefault("manual_hint", "Set -pcstop -N.")
    kw.setdefault("verbose", False)
    return resolve_cv_design(requested, single_trials, repeats, **kw)


class TestResolveCvDesign:
    def test_auto_picks_single_when_repeats_support_it(self):
        repeats = summarize_trial_repeats(_onsets([[5, 5, 5, 5]] * 3))
        assert _resolve("auto", True, repeats) == "single"

    def test_auto_falls_back_to_condition_on_thin_repeats(self):
        # cond0 repeats across runs, cond1 is run-exclusive: 25% predictable.
        # Beta-space CV would score 75% of trials against nothing.
        repeats = summarize_trial_repeats(_onsets([[5, 5], [30, 0]]))
        assert _resolve("auto", True, repeats) == "condition"

    def test_auto_is_condition_without_single_trials(self):
        repeats = summarize_trial_repeats(_onsets([[5, 5, 5, 5]] * 3))
        assert _resolve("auto", False, repeats) == "condition"

    def test_explicit_condition_wins_over_good_repeats(self):
        # The whole point of the split: emit per-trial betas, select on conditions.
        repeats = summarize_trial_repeats(_onsets([[5, 5, 5, 5]] * 3))
        assert _resolve("condition", True, repeats) == "condition"

    def test_explicit_single_without_emit_exits(self):
        repeats = summarize_trial_repeats(_onsets([[5, 5, 5, 5]] * 3))
        with pytest.raises(SystemExit):
            _resolve("single", False, repeats)

    def test_explicit_single_with_thin_repeats_is_allowed(self, capsys):
        # 25% predictable: below the auto threshold, but the user asked for it.
        repeats = summarize_trial_repeats(_onsets([[5, 5], [30, 0]]))
        assert _resolve("single", True, repeats) == "single"
        assert "WARNING" in capsys.readouterr().out


class TestNoCrossRunStructure:
    """Zero cross-run overlap breaks *both* CV designs, not just beta space.

    Condition-level CV predicts a held-out run from the others; if no condition
    is shared, there is nothing to predict with. The tool must say so instead of
    dying inside compute_xval_r2 or, worse, returning a flat curve.
    """

    UNIQUE = _onsets([[5, 0], [0, 5]])

    def test_condition_is_not_a_rescue(self, capsys):
        repeats = summarize_trial_repeats(self.UNIQUE)
        with pytest.raises(SystemExit):
            _resolve("condition", True, repeats)
        assert "cannot be cross-validated" in capsys.readouterr().out

    def test_auto_errors_when_nothing_is_scoreable(self, capsys):
        repeats = summarize_trial_repeats(self.UNIQUE)
        with pytest.raises(SystemExit):
            _resolve("auto", True, repeats)
        out = capsys.readouterr().out
        assert "cannot be cross-validated" in out
        assert "-pcstop" in out  # the actionable way out

    def test_in_sample_single_trial_selection_survives(self):
        # ffs_hrfopt's single-trial HRF selection is in-sample, so it is the one
        # path that still has a valid criterion with zero repeats.
        repeats = summarize_trial_repeats(self.UNIQUE)
        assert (
            _resolve(
                "auto",
                True,
                repeats,
                parameter="HRF",
                single_needs_repeats=False,
                manual_hint="Add -single_trials.",
            )
            == "single"
        )

    def test_hrfopt_without_single_trials_still_errors(self, capsys):
        repeats = summarize_trial_repeats(self.UNIQUE)
        with pytest.raises(SystemExit):
            _resolve(
                "auto",
                False,
                repeats,
                parameter="HRF",
                single_needs_repeats=False,
                manual_hint="Add -single_trials.",
            )
        assert "Add -single_trials." in capsys.readouterr().out


class TestFlagWiring:
    """The two flags must exist, and be independent, in both CLIs."""

    @pytest.mark.parametrize("module", ["denoise", "hrfopt"])
    def test_parser_exposes_both_axes(self, module):
        import importlib

        parser = importlib.import_module(f"fastfuncstuff.cli.{module}").create_parser()
        dests = {a.dest for a in parser._actions}
        assert "single_trials" in dests
        assert "cv_design" in dests

        args = parser.parse_args(
            _minimal_args(module) + ["-single_trials", "-cv_design", "condition"]
        )
        assert args.single_trials is True
        assert args.cv_design == "condition"

        defaults = parser.parse_args(_minimal_args(module))
        assert defaults.cv_design == "auto"
        assert defaults.single_trials is False

    def test_hrfopt_legacy_alias_still_emits(self):
        import importlib

        parser = importlib.import_module("fastfuncstuff.cli.hrfopt").create_parser()
        args = parser.parse_args(_minimal_args("hrfopt") + ["-save_single_trial_betas"])
        assert args.single_trials is True


def _minimal_args(module):
    base = [
        "-input",
        "run1.nii.gz",
        "run2.nii.gz",
        "-onsets",
        "cond.txt",
        "-durations",
        "2.0",
        "-tr",
        "2.0",
        "-prefix",
        "out",
    ]
    return base
