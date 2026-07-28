"""Per-stage option overrides (-<stage>_opts) and input-path absolutization.

The bug of record for the paths: a relative -grand_reference was baked into the
script verbatim, so moving the script (or running it from any other cwd) made
stage09 point at nothing."""

from __future__ import annotations

from pathlib import Path

import pytest

from fastfuncstuff.autoproc import config, optcheck
from fastfuncstuff.autoproc.bids import BoldRun, Session, Subject
from fastfuncstuff.autoproc.emit import write_script
from fastfuncstuff.autoproc.plan import Options, build_plan
from fastfuncstuff.cli.autoproc import _absolutize_inputs, _glue_opt_values, build_parser


def _subject():
    run = BoldRun(
        subject="X",
        session="01",
        task="foo",
        run="1",
        mag_path=Path("/bids/sub-X/ses-01/func/sub-X_ses-01_task-foo_run-1_bold.nii.gz"),
        json={"RepetitionTime": 2.0, "PhaseEncodingDirection": "j-"},
    )
    return Subject("X", [Session("01", [run])])


@pytest.mark.parametrize("key", config.STAGE_OPT_KEYS)
def test_every_default_opts_key_has_a_cli_flag(key):
    """DEFAULT_OPTS is what the emitter reads; a key without a flag is a stage
    the user silently cannot tune."""
    args = build_parser().parse_args(
        ["-bids_dir", "/bids", "-subject", "X", f"-{key}_opts", "-made -up"]
    )
    assert getattr(args, f"{key}_opts") == "-made -up"
    dashed = build_parser().parse_args(
        ["-bids_dir", "/bids", "-subject", "X", f"-{key.replace('_', '-')}-opts", "-made -up"]
    )
    assert getattr(dashed, f"{key}_opts") == "-made -up"


@pytest.mark.parametrize(
    ("key", "opts", "needle", "options"),
    [
        (
            "nordic",
            "-mppca -kernel_size_PCA 9 9 9",
            "-kernel_size_PCA 9 9 9",
            {"want_nordic": True},
        ),
        ("tshift", "-tzero 0 -interp fourier", "-interp fourier", {"slicetiming_method": "first"}),
        ("segment", "-niter 42 -samp 3.0", "-niter 42", {"anat_nonlin": True, "tpm": "/t.nii"}),
        ("moco", "-cost lpa", "-cost lpa", {}),
    ],
)
def test_stage_opts_reach_the_emitted_command(monkeypatch, key, opts, needle, options):
    monkeypatch.setitem(config.DEFAULT_OPTS, key, opts)
    plan = build_plan(_subject(), Options(go_to_anat=True, **options))
    script = write_script(plan, "wd", bids_root="/bids")
    assert needle in script
    assert config.DEFAULT_OPTS[key].split()[0] in script


def test_fs_tpm_segment_uses_its_own_opts_key(monkeypatch):
    """The FS-built TPM has 8 hard-edge classes and needs different ngaus than an
    SPM-style TPM — the two tunings must not share one override."""
    monkeypatch.setitem(config.DEFAULT_OPTS, "segment", "-niter 42")
    monkeypatch.setitem(config.DEFAULT_OPTS, "segment_fstpm", "-ngaus 9 9")
    plan = build_plan(
        _subject(), Options(go_to_anat=True, anat_nonlin=True, fs_tpm=True, tpm="/t.nii")
    )
    script = write_script(plan, "wd", bids_root="/bids")
    assert "-ngaus 9 9" in script
    assert "-niter 42" not in script


@pytest.mark.parametrize("value", ["-save_numcomps", "-nordic -save_numcomps", "-x"])
def test_single_flag_override_is_not_eaten_as_an_option(value):
    """argparse only tolerates a '-'-leading value when it contains a space, so a
    one-flag override used to die with "expected one argument"."""
    argv = _glue_opt_values(["-bids_dir", "/bids", "-subject", "X", "-nordic_opts", value])
    assert build_parser().parse_args(argv).nordic_opts == value


@pytest.mark.parametrize("key", sorted(optcheck.STAGE_TOOL))
def test_every_stage_default_passes_its_own_tools_parser(key):
    """The shipped defaults are the pipeline's opinion — they must at minimum be
    spelled the way the tool spells them. Doubles as a rename alarm."""
    if key == "glm":  # -glm_opts has no DEFAULT_OPTS entry (it appends, not replaces)
        pytest.skip("no default string")
    assert optcheck.check_opts(key, config.DEFAULT_OPTS[key]) == []


def test_every_opts_key_is_mapped_to_a_tool():
    unmapped = set(config.STAGE_OPT_KEYS) - set(optcheck.STAGE_TOOL) - {"unwrap"}
    assert not unmapped, f"no tool mapping (so no typo check) for: {sorted(unmapped)}"


def test_typo_is_caught_with_a_suggestion():
    errs = optcheck.check_opts("nordic", "-nordic -nodric")
    assert len(errs) == 1
    assert "-nodric" in errs[0]
    assert "-nordic" in errs[0].split("did you mean")[1]


def test_negative_number_values_are_not_mistaken_for_flags():
    """Values may start with '-' (ranges like -0.9,0.9,65); AFNI-style
    digit-leading flags (-1Dfile) still get checked."""
    assert optcheck.check_opts("glm", "-a_grid -0.9,0.9,65") == []
    assert optcheck.check_opts("moco", "-cost wls -1Dfile m.1D") == []
    assert optcheck.check_opts("moco", "-1Dfle m.1D") != []


def test_unknown_stage_and_external_tool_are_skipped():
    assert optcheck.check_opts("unwrap", "-t epi -v") == []
    assert optcheck.check_opts("not_a_stage", "-whatever") == []


def test_input_paths_are_absolutized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ref.results").mkdir()
    args = build_parser().parse_args(
        [
            "-bids_dir",
            "bids",
            "-subject",
            "X",
            "-grand_reference",
            "ref.results/",
            "-ref_transforms",
            "a.1D",
            "b.1D",
        ]
    )
    _absolutize_inputs(args)
    assert args.grand_reference == str(tmp_path / "ref.results")
    assert args.bids_dir == str(tmp_path / "bids")
    assert args.ref_transforms == [str(tmp_path / "a.1D"), str(tmp_path / "b.1D")]


def test_absolutize_leaves_unset_paths_alone():
    args = build_parser().parse_args(["-bids_dir", "/bids", "-subject", "X"])
    _absolutize_inputs(args)
    assert args.anat is None
    assert args.grand_reference is None
    assert args.events is None
