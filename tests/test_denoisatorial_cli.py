"""Argument-parsing tests for ffs_denoisatorial's timing inputs."""

from fastfuncstuff.cli.denoisatorial import create_parser

BASE = ["-input", "run1.nii.gz", "-tr", "2.0", "-prefix", "out"]


def test_onsets_and_durations_are_optional():
    # Both timing paths are validated in main(), so argparse must not force either.
    args = create_parser().parse_args([*BASE, "-events", "run-01_events.tsv"])
    assert args.events == ["run-01_events.tsv"]
    assert args.onsets is None
    assert args.durations is None


def test_afni_timing_path_still_parses():
    args = create_parser().parse_args([*BASE, "-onsets", "a.txt", "-durations", "2.0"])
    assert args.onsets == ["a.txt"]
    assert args.durations == ["2.0"]
    assert args.events is None


def test_event_flags_accept_hyphen_and_underscore():
    for ignore, cols, rnd in (
        ("-event-ignore", "-event-cols", "-round-durations"),
        ("-event_ignore", "-event_cols", "-round_durations"),
    ):
        args = create_parser().parse_args(
            [
                *BASE,
                "-events",
                "e.tsv",
                ignore,
                "fixation",
                "null",
                cols,
                "start",
                "len",
                "cond",
                rnd,
                "1",
            ]
        )
        assert args.event_ignore == ["fixation", "null"]
        assert args.event_cols == ["start", "len", "cond"]
        assert args.round_durations == 1
