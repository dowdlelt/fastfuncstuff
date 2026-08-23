"""Batch-mode plumbing shared by ffs_moco, ffs_nwarp, ffs_allineate and ffs_formwarp.

Covers the shared runner (collect + skip-existing isolation), each tool's
expected-output enumeration used by -batch_skip, and the autoproc emitter's
batched stages (moco, cross-run, cross-session, final).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from fastfuncstuff.cli_utils import collect_batch_jobs, run_batch_jobs

# --------------------------------------------------------------------------
# collect_batch_jobs
# --------------------------------------------------------------------------


def test_collect_batch_jobs_file_and_inline(tmp_path):
    manifest = tmp_path / "runs.txt"
    manifest.write_text(
        "# a comment\n"
        "-input a.nii -prefix a_mc.nii\n"
        "\n"  # blank line ignored
        "-input b.nii -prefix b_mc.nii\n"
    )
    jobs = collect_batch_jobs(str(manifest), ["-input c.nii -prefix c_mc.nii"])
    # File runs first (with line-number labels), then inline runs.
    assert [label for label, _ in jobs] == ["line 2", "line 4", "run 1"]
    assert jobs[0][1] == "-input a.nii -prefix a_mc.nii"
    assert jobs[2][1] == "-input c.nii -prefix c_mc.nii"


def test_collect_batch_jobs_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        collect_batch_jobs(str(tmp_path / "nope.txt"), None)


def test_collect_batch_jobs_empty_file_exits(tmp_path):
    manifest = tmp_path / "empty.txt"
    manifest.write_text("# only comments\n\n")
    with pytest.raises(SystemExit):
        collect_batch_jobs(str(manifest), None)


# --------------------------------------------------------------------------
# run_batch_jobs: isolation + skip-existing
# --------------------------------------------------------------------------


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def test_run_batch_skips_when_outputs_exist(tmp_path, capsys):
    done = tmp_path / "done.nii"
    done.write_text("x")
    todo = tmp_path / "todo.nii"  # does not exist

    jobs = [("run 1", "done"), ("run 2", "todo")]
    dispatched: list[str] = []

    def parse_line(line: str, base=None) -> argparse.Namespace:
        return _ns(out=str(done if line == "done" else todo))

    run_batch_jobs(
        tool="ffs_test",
        jobs=jobs,
        device=torch.device("cpu"),
        parse_line=parse_line,
        dispatch=lambda a, d: dispatched.append(a.out),
        expected_outputs=lambda a: [a.out],
        skip_existing=True,
        verb=1,
    )
    # Only the run whose output was missing actually dispatched.
    assert dispatched == [str(todo)]
    assert "1 skipped" in capsys.readouterr().out


def test_run_batch_no_skip_runs_all(tmp_path):
    done = tmp_path / "done.nii"
    done.write_text("x")
    dispatched: list[str] = []
    run_batch_jobs(
        tool="ffs_test",
        jobs=[("run 1", "done")],
        device=torch.device("cpu"),
        parse_line=lambda line, base: _ns(out=str(done)),
        dispatch=lambda a, d: dispatched.append(a.out),
        expected_outputs=lambda a: [a.out],
        skip_existing=False,  # skip disabled → runs even though the file exists
        verb=0,
    )
    assert dispatched == [str(done)]


def test_run_batch_isolates_failures(capsys):
    dispatched: list[str] = []

    def dispatch(a, d):
        if a.tag == "bad":
            raise RuntimeError("boom")
        dispatched.append(a.tag)

    with pytest.raises(SystemExit):
        run_batch_jobs(
            tool="ffs_test",
            jobs=[("run 1", "good"), ("run 2", "bad"), ("run 3", "good2")],
            device=torch.device("cpu"),
            parse_line=lambda line, base: _ns(tag="good2" if line == "good2" else line),
            dispatch=dispatch,
            verb=1,
        )
    # A failing run does not sink the others; the batch still exits nonzero.
    assert dispatched == ["good", "good2"]
    assert "FAILED" in capsys.readouterr().out


def test_run_batch_rejects_nested_batch(capsys):
    with pytest.raises(SystemExit):
        run_batch_jobs(
            tool="ffs_test",
            jobs=[("run 1", "x")],
            device=torch.device("cpu"),
            parse_line=lambda line, base: _ns(nested=True),
            dispatch=lambda a, d: None,
            is_nested=lambda a: a.nested,
            verb=1,
        )
    assert "FAILED" in capsys.readouterr().out


# --------------------------------------------------------------------------
# ffs_moco._expected_outputs
# --------------------------------------------------------------------------


def test_moco_expected_outputs_single_echo():
    from fastfuncstuff.cli.moco import _expected_outputs, parse_args

    a = parse_args(["-input", "epi.nii.gz", "-prefix", "out", "-1Dfile", "m.1D", "-save_mean"])
    outs = _expected_outputs(a)
    assert "out.nii.gz" in outs
    assert "mean_out.nii.gz" in outs  # -save_mean derives from prefix
    assert "m.1D" in outs


def test_moco_reweight_tolerance_parses_and_validates(capsys):
    from fastfuncstuff.cli.moco import _validate_run_args, parse_args

    args = parse_args(
        ["-input", "epi.nii.gz", "-prefix", "out", "-reweight", "-reweight-tolerance", "1.35"]
    )
    assert args.reweight_tolerance == pytest.approx(1.35)
    _validate_run_args(args)

    bad = parse_args(
        ["-input", "epi.nii.gz", "-prefix", "out", "-reweight", "-reweight_tolerance", "0.9"]
    )
    with pytest.raises(SystemExit):
        _validate_run_args(bad)
    assert "must be >= 1" in capsys.readouterr().err


def test_moco_expected_outputs_multi_echo_prefixes_each():
    from fastfuncstuff.cli.moco import _expected_outputs, parse_args

    a = parse_args(
        [
            "-input",
            "e1.nii",
            "e2.nii",
            "-reg_echo",
            "1",
            "-prefix",
            "mc.nii.gz",
            "-save_tsnr",
            "-1Dmatrix_save",
            "mat.aff12.1D",
        ]
    )
    outs = _expected_outputs(a)
    # Per-echo series + QC get eN_ prefixes; the matrix is single-instance.
    assert "e1_mc.nii.gz" in outs
    assert "e2_mc.nii.gz" in outs
    assert "tsnr_e1_mc.nii.gz" in outs
    assert "tsnr_e2_mc.nii.gz" in outs
    assert "mat.aff12.1D" in outs
    assert outs.count("mat.aff12.1D") == 1


# --------------------------------------------------------------------------
# ffs_nwarp._expected_outputs
# --------------------------------------------------------------------------


def test_nwarp_expected_outputs_prefix_verbatim_plus_derived():
    from fastfuncstuff.cli.nwarp import _expected_outputs, parse_args

    a = parse_args(
        [
            "-source",
            "epi.nii.gz",
            "-nwarp",
            "w.nii",
            "-prefix",
            "out.nii.gz",
            "-save_mean",
            "-save_first_last",
            "-phase",
            "ph.nii.gz",
        ]
    )
    outs = _expected_outputs(a)
    assert outs[0] == "out.nii.gz"  # prefix used as-is (no parse_prefix)
    assert "out_phase.nii.gz" in outs
    assert "mean_out.nii.gz" in outs
    assert "firstlast_out.nii.gz" in outs


# --------------------------------------------------------------------------
# ffs_allineate / ffs_formwarp._expected_outputs
# --------------------------------------------------------------------------


def test_allineate_expected_outputs_and_optional_flags():
    from fastfuncstuff.cli.allineate import _expected_outputs, parse_args

    a = parse_args(
        [
            "-base",
            "ref.nii",
            "-source",
            "mov.nii",
            "-prefix",
            "out.nii.gz",
            "-1Dmatrix_save",
            "m.aff12.1D",
            "-save_weight",
            "w.nii.gz",
        ]
    )
    outs = _expected_outputs(a)
    assert outs[0] == "out.nii.gz"  # prefix verbatim (no parse_prefix)
    assert "m.aff12.1D" in outs and "w.nii.gz" in outs
    # An alignment with no matrix requested still has its warped image checked.
    bare = parse_args(["-base", "ref.nii", "-source", "mov.nii", "-prefix", "o.nii"])
    assert _expected_outputs(bare) == ["o.nii"]


def test_allineate_batch_makes_io_flags_optional():
    """-batch alone must parse: the per-run args live in the manifest."""
    from fastfuncstuff.cli.allineate import parse_args

    a = parse_args(["-batch", "runs.txt", "-device", "cpu"])
    assert a.batch == "runs.txt" and a.base is None and a.prefix is None


def test_formwarp_expected_outputs_names_the_warps():
    from fastfuncstuff.cli.formwarp import _expected_outputs, parse_args

    a = parse_args(
        [
            "-base",
            "f.nii",
            "-source",
            "m.nii",
            "-prefix",
            "w.nii.zst",
            "-save_warp",
            "-save_inverse",
        ]
    )
    outs = _expected_outputs(a)
    # Warp names follow the parsed prefix + its extension, as _dispatch_run writes them.
    assert outs == ["w.nii.zst", "w_WARP.nii.zst", "w_WARPINV.nii.zst"]


def test_optiwarp_expected_outputs_names_the_warps():
    from fastfuncstuff.cli.optiwarp import _expected_outputs, parse_args

    a = parse_args(
        [
            "-base",
            "f.nii",
            "-source",
            "m.nii",
            "-prefix",
            "w.nii.zst",
            "-save_warp",
            "-save_jacobian",
        ]
    )
    assert _expected_outputs(a) == ["w.nii.zst", "w_WARP.nii.zst", "w_JAC.nii.zst"]


def test_optiwarp_warp_prefix_moves_only_the_warps():
    """Several runs write their own warped image but share one field.

    Naming both off -prefix would have them overwrite each other's warp, which is
    why the autoproc cross-run stage needs the two stems to be separable.
    """
    from fastfuncstuff.cli.optiwarp import _expected_outputs, parse_args

    a = parse_args(
        [
            "-base",
            "f.nii",
            "-source",
            "m.nii",
            "-prefix",
            "laneA.nii.gz",
            "-warp_prefix",
            "shared",
            "-save_warp",
        ]
    )
    assert _expected_outputs(a) == ["laneA.nii.gz", "shared_WARP.nii.gz"]


def test_optiwarp_batch_makes_io_flags_optional():
    """-base/-source/-prefix come from the manifest line, not the outer command."""
    from fastfuncstuff.cli.optiwarp import parse_args

    a = parse_args(["-batch", "runs.txt"])
    assert a.base is None and a.source is None and a.prefix is None


def test_optiwarp_batch_run_rejects_a_run_missing_its_prefix():
    from fastfuncstuff.cli.optiwarp import _validate_batch_run, parse_args

    with pytest.raises(ValueError, match="-prefix"):
        _validate_batch_run(parse_args(["-base", "f.nii", "-source", "m.nii"]))


def test_nonlinear_backends_share_the_structural_flags():
    """The three flags autoproc's nonlinear stages need, on every backend.

    Backends differ in what they compute and in the knobs that shape it; they must
    NOT differ in how a pipeline addresses them. -matrix (solve in the source's
    frame), -warp_prefix (several images, one shared field) and the -batch trio
    (one process per stage) are the pipeline's requirements, so a backend without
    them cannot be offered as a choice at all.
    """
    import importlib

    for tool in ("formwarp", "optiwarp", "qwarp"):
        mod = importlib.import_module(f"fastfuncstuff.cli.{tool}")
        args = mod.parse_args(["-batch", "runs.txt"])
        for flag in ("matrix", "warp_prefix", "batch", "batch_run", "batch_skip"):
            assert hasattr(args, flag), f"{tool} is missing -{flag}"
        assert args.base is None and args.source is None and args.prefix is None, tool


# --------------------------------------------------------------------------
# autoproc emitter: batched stages
# --------------------------------------------------------------------------


def _tiny_plan(**opt_kw):
    from fastfuncstuff.autoproc.bids import BoldRun, Session, Subject
    from fastfuncstuff.autoproc.plan import Options, build_plan

    def _run(run):
        return BoldRun(
            subject="X",
            session="01",
            task="foo",
            run=run,
            mag_path=Path(f"/bids/sub-X/ses-01/func/sub-X_ses-01_task-foo_run-{run}_bold.nii.gz"),
            json={"RepetitionTime": 2.0, "PhaseEncodingDirection": "j-"},
        )

    subj = Subject("X", [Session("01", [_run("1"), _run("2")])])
    return build_plan(subj, Options(go_to_anat=True, **opt_kw))


def test_emit_moco_stage_is_batched():
    from fastfuncstuff.autoproc.emit import write_script

    s = write_script(_tiny_plan(), "wd", bids_root="/bids", script_stem="proc_sub-X")
    # One batched call over a manifest named after the script, no per-run tool call.
    assert 'mocobatch="proc_sub-X_mocobatch.txt"' in s
    assert 'ffs_moco -batch "$mocobatch"' in s
    # The manifest is (re)truncated then appended per run inside the loop.
    assert ': > "$mocobatch"' in s
    # Each run's args (incl. the motion params) are printf'd into the manifest.
    assert "printf " in s and '.motion.1D\\"" >> "$mocobatch"' in s


def test_emit_final_stage_is_batched():
    from fastfuncstuff.autoproc.emit import write_script

    s = write_script(_tiny_plan(), "wd", bids_root="/bids", script_stem="proc_sub-X")
    assert 'nwarpbatch="proc_sub-X_nwarpbatch.txt"' in s
    assert 'ffs_nwarp -batch "$nwarpbatch"' in s


def test_emit_xrun_stage_is_batched():
    """stage06 writes two manifests in the run loop, then launches one
    ffs_allineate and (with -xrun_nonlin) one ffs_formwarp for all runs."""
    from fastfuncstuff.autoproc.emit import write_script

    s = write_script(
        _tiny_plan(xrun_nonlin=True), "wd", bids_root="/bids", script_stem="proc_sub-X"
    )
    assert 'albatch="proc_sub-X_xrunbatch.txt"' in s
    assert 'fwbatch="proc_sub-X_xrunnlbatch.txt"' in s
    assert 'ffs_allineate -batch "$albatch"' in s
    assert 'ffs_formwarp -batch "$fwbatch"' in s
    # The loop only appends; no per-run tool call survives in the stage.
    stage = s.split("stage06:")[1].split("# ====")[0]
    assert "\n    ffs_allineate" not in stage and "\n    ffs_formwarp" not in stage
    # The linear batch must launch before the nonlinear one — its output is the
    # nonlinear source.
    assert s.index('ffs_allineate -batch "$albatch"') < s.index('ffs_formwarp -batch "$fwbatch"')


def test_emit_xses_stage_is_batched():
    from fastfuncstuff.autoproc.bids import BoldRun, Session, Subject
    from fastfuncstuff.autoproc.emit import write_script
    from fastfuncstuff.autoproc.plan import Options, build_plan

    def _run(ses):
        return BoldRun(
            subject="X",
            session=ses,
            task="foo",
            run="1",
            mag_path=Path(f"/bids/sub-X/ses-{ses}/func/sub-X_ses-{ses}_task-foo_run-1_bold.nii.gz"),
            json={"RepetitionTime": 2.0, "PhaseEncodingDirection": "j-"},
        )

    subj = Subject("X", [Session("01", [_run("01")]), Session("02", [_run("02")])])
    plan = build_plan(subj, Options(go_to_anat=True, ref_ses="01", xses_nonlin=True))
    s = write_script(plan, "wd", bids_root="/bids", script_stem="p")
    assert 'albatch="p_xsesbatch.txt"' in s and 'fwbatch="p_xsesnlbatch.txt"' in s
    assert 'ffs_allineate -batch "$albatch"' in s
    assert 'ffs_formwarp -batch "$fwbatch"' in s


def test_emit_single_session_writes_no_xses_batch():
    """One session → nothing to align across sessions, so no manifest and no
    launch (an empty -batch is an error in the tools)."""
    from fastfuncstuff.autoproc.emit import write_script

    s = write_script(_tiny_plan(), "wd", bids_root="/bids", script_stem="p")
    assert "xsesbatch.txt" not in s


def test_emit_skip_toggle_default_and_overwrite():
    from fastfuncstuff.autoproc.emit import write_script

    skip = write_script(_tiny_plan(), "wd", script_stem="p")
    assert "skip_moco=1" in skip and "skip_final=1" in skip
    over = write_script(_tiny_plan(batch_overwrite=True), "wd", script_stem="p")
    assert "skip_moco=0" in over and "skip_final=0" in over


def test_emitted_batch_stage_is_valid_bash(tmp_path):
    import shutil
    import subprocess

    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    from fastfuncstuff.autoproc.emit import write_script

    # Both shapes: the simple plan and the one where every batched stage is live.
    plans = [
        _tiny_plan(),
        _tiny_plan(xrun_nonlin=True, xses_nonlin=True, anat_nonlin=True, locomoco=True),
    ]
    for i, plan in enumerate(plans):
        s = write_script(plan, "wd", bids_root="/bids", script_stem="proc_sub-X")
        script = tmp_path / f"proc{i}.sh"
        script.write_text(s)
        # bash -n parses without executing: catches quoting/printf mistakes in the
        # batch-manifest construction.
        subprocess.run(["bash", "-n", str(script)], check=True)


# --------------------------------------------------------------------------
# ffs_util_automask._expected_outputs
# --------------------------------------------------------------------------


def test_automask_expected_outputs_is_the_prefix():
    from fastfuncstuff.cli.automask import _expected_outputs, parse_args

    a = parse_args(["-input", "mean.nii", "-prefix", "mask.nii.gz"])
    assert _expected_outputs(a) == ["mask.nii.gz"]


def test_automask_batch_makes_io_flags_optional():
    """-batch alone must parse: the per-run args live in the manifest."""
    from fastfuncstuff.cli.automask import parse_args

    a = parse_args(["-batch", "runs.txt", "-device", "cpu"])
    assert a.batch == "runs.txt" and a.input is None and a.prefix is None


def test_automask_batch_run_rejects_a_run_missing_its_output():
    from fastfuncstuff.cli.automask import _validate_batch_run, parse_args

    _validate_batch_run(parse_args(["-input", "m.nii", "-prefix", "o.nii"]))
    with pytest.raises(ValueError, match="-prefix"):
        _validate_batch_run(parse_args(["-input", "m.nii"]))


# --------------------------------------------------------------------------
# ffs_locomoco / ffs_blipflip / ffs_util_3dmath batch surfaces
# --------------------------------------------------------------------------


def test_locomoco_expected_outputs_covers_the_lane_reductions():
    """-batch_skip must check the images stage07 reads, not just the warp: a
    working directory from before -save_max existed has the warp but no max."""
    from fastfuncstuff.cli.locomoco import _expected_outputs, _parse

    a = _parse(["-input", "r.nii.gz", "-prefix", "nl.nii.zst", "-pe_dir", "y"])
    assert _expected_outputs(a) == ["nl_warp.nii.zst"]

    a = _parse(
        ["-input", "r.nii.gz", "-prefix", "nl.nii.zst", "-pe_dir", "y", "-save_max", "-save_mean"]
    )
    assert _expected_outputs(a) == [
        "nl_warp.nii.zst",
        "nl_locomoco_mean.nii.zst",
        "nl_locomoco_max.nii.zst",
    ]


def test_locomoco_batch_makes_io_flags_optional():
    from fastfuncstuff.cli.locomoco import _parse

    a = _parse(["-batch", "runs.txt", "-device", "cpu"])
    assert a.batch == "runs.txt" and a.input is None and a.prefix is None


def test_locomoco_batch_run_rejects_a_run_missing_its_output():
    from fastfuncstuff.cli.locomoco import _parse, _validate_batch_run

    _validate_batch_run(_parse(["-input", "r.nii", "-prefix", "o.nii", "-pe_dir", "y"]))
    with pytest.raises(ValueError, match="-prefix"):
        _validate_batch_run(_parse(["-input", "r.nii", "-pe_dir", "y"]))


def test_blipflip_expected_outputs_names_the_warp_and_unwarped():
    from fastfuncstuff.cli.blipflip import _expected_outputs, create_parser

    argv = ["-blip_up", "u.nii", "-blip_down", "d.nii", "-pe_dir", "j", "-prefix", "dc.nii.zst"]
    a = create_parser().parse_args(argv)
    assert _expected_outputs(a) == ["dc_warp.nii.zst", "dc_unwarped.nii.zst"]
    a = create_parser().parse_args([*argv, "-no_unwarped"])
    assert _expected_outputs(a) == ["dc_warp.nii.zst"]


def test_blipflip_batch_makes_io_flags_optional():
    from fastfuncstuff.cli.blipflip import create_parser

    a = create_parser().parse_args(["-batch", "runs.txt", "-device", "cpu"])
    assert a.batch == "runs.txt" and a.prefix is None and a.pe_dir is None


def test_3dmath_batch_requires_an_operation_per_run():
    from fastfuncstuff.cli.util_3dmath import _build_parser, _validate_batch_run

    p = _build_parser()
    _validate_batch_run(p.parse_args(["-input", "a.nii", "-mean", "-prefix", "o.nii"]))
    with pytest.raises(ValueError, match="no operation"):
        _validate_batch_run(p.parse_args(["-input", "a.nii", "-prefix", "o.nii"]))
    with pytest.raises(ValueError, match="-prefix"):
        _validate_batch_run(p.parse_args(["-input", "a.nii", "-mean"]))


def test_emit_locomoco_blip_and_runmean_stages_are_batched():
    """The three stages that used to be shell loops of one process per run."""
    from fastfuncstuff.autoproc.emit import write_script

    s = write_script(_tiny_plan(locomoco=True), "wd", bids_root="/bids", script_stem="proc_sub-X")
    assert 'lmbatch="proc_sub-X_locomocobatch.txt"' in s
    assert 'ffs_locomoco -batch "$lmbatch"' in s
    # stage07 pushes every coverage lane through ONE nwarp batch, not one per lane.
    assert 'rmbatch="proc_sub-X_runmeanbatch.txt"' in s
    assert 'ffs_nwarp -batch "$rmbatch"' in s
    stage07 = s[s.index("stage07: run + session means") : s.index("stage08:")]
    assert "ffs_nwarp \\\n" not in stage07, "stage07 still launches a solo ffs_nwarp per run"
    # session means and grandmeans are one ffs_util_3dmath each, not one per lane.
    assert 'smbatch="proc_sub-X_sesmeanbatch.txt"' in s
    assert 'gmeanbatch="proc_sub-X_grandmeanbatch.txt"' in s


def test_emit_qc_stacks_are_queued_and_flushed_per_stage():
    """qc_tcat only appends to a manifest; each stage's QC block ends in a flush
    so one ffs_util_3dmath builds every stack that stage queued."""
    from fastfuncstuff.autoproc.emit import write_script

    s = write_script(_tiny_plan(), "wd", bids_root="/bids", script_stem="proc_sub-X")
    assert 'qcbatch="proc_sub-X_qcbatch.txt"' in s
    assert 'ffs_util_3dmath -batch "$qcbatch"' in s
    # every emitted qc_tcat call is followed (eventually) by a flush
    assert s.count("qc_flush") >= 2  # the definition plus at least one call


def test_qwarp_expected_outputs_names_the_warp_gz_regardless_of_prefix_ext():
    """qwarp writes its warp .nii.gz even when the image is zstd -- mirror that.

    -batch_skip compares paths, so a guess that follows the prefix's extension
    would never match the file on disk and every job would re-run.
    """
    from fastfuncstuff.cli.qwarp import _expected_outputs, parse_args

    a = parse_args(["-base", "f.nii", "-source", "m.nii", "-prefix", "w.nii.zst"])
    assert _expected_outputs(a) == ["w.nii.zst", "w_WARP.nii.gz"]


def test_qwarp_warp_prefix_moves_only_the_warp():
    from fastfuncstuff.cli.qwarp import _expected_outputs, parse_args

    a = parse_args(
        ["-base", "f.nii", "-source", "m.nii", "-prefix", "laneA.nii.gz", "-warp_prefix", "shared"]
    )
    assert _expected_outputs(a) == ["laneA.nii.gz", "shared_WARP.nii.gz"]


def test_qwarp_batch_run_rejects_a_run_missing_its_prefix():
    """-source may be absent (single-file timeseries mode); -base/-prefix may not."""
    from fastfuncstuff.cli.qwarp import _validate_batch_run, parse_args

    _validate_batch_run(parse_args(["-base", "series.nii", "-prefix", "out.nii"]))
    with pytest.raises(ValueError, match="-prefix"):
        _validate_batch_run(parse_args(["-base", "f.nii", "-source", "m.nii"]))


def test_qwarp_batch_is_not_shadowed_by_its_optimizer_flags():
    """-batch_lr/-batch_iters/-batch_tol/-batch_patience are the Adam optimizer's.

    They predate the manifest flag and share its prefix, so before -batch existed
    argparse resolved a bare -batch as an ambiguous abbreviation of them. An exact
    match wins over abbreviation, but only while -batch itself is defined.
    """
    from fastfuncstuff.cli.qwarp import parse_args

    a = parse_args(["-batch", "runs.txt", "-batch_iters", "7"])
    assert a.batch == "runs.txt" and a.batch_iters == 7


# --------------------------------------------------------------------------
# Outer flags are per-run defaults
# --------------------------------------------------------------------------


def test_outer_flags_reach_every_run():
    """Bug of record: a flag outside -batch was silently dropped.

    `ffs_optiwarp -force lk -batch runs.txt` ran demons -- the default -- on every
    pair, and nothing said so. Under autoproc that meant asking for optiwarp_lk
    and getting demons: a whole study's worth of warps from the wrong engine, with
    a script that reads as though it asked for the right one.
    """
    seen = []
    run_batch_jobs(
        tool="t",
        jobs=[("line 1", "-prefix a"), ("line 2", "-prefix b")],
        device=torch.device("cpu"),
        parse_line=lambda line, base: _parse_toy(line, base),
        dispatch=lambda ra, dev: seen.append((ra.prefix, ra.force)),
        defaults=_parse_toy("-force lk"),
    )
    assert seen == [("a", "lk"), ("b", "lk")]


def test_a_run_line_overrides_the_outer_default():
    seen = []
    run_batch_jobs(
        tool="t",
        jobs=[("line 1", "-prefix a"), ("line 2", "-prefix b -force demons")],
        device=torch.device("cpu"),
        parse_line=lambda line, base: _parse_toy(line, base),
        dispatch=lambda ra, dev: seen.append((ra.prefix, ra.force)),
        defaults=_parse_toy("-force lk"),
    )
    assert seen == [("a", "lk"), ("b", "demons")]


def test_one_runs_settings_do_not_leak_into_the_next():
    """argparse writes into the namespace it is handed, so each run needs its own."""
    seen = []
    run_batch_jobs(
        tool="t",
        jobs=[("line 1", "-prefix a -force demons"), ("line 2", "-prefix b")],
        device=torch.device("cpu"),
        parse_line=lambda line, base: _parse_toy(line, base),
        dispatch=lambda ra, dev: seen.append((ra.prefix, ra.force)),
        defaults=_parse_toy("-force lk"),
    )
    assert seen == [("a", "demons"), ("b", "lk")]


def test_batch_only_flags_are_not_inherited_by_a_run():
    """-batch on the outer command must not make every line look nested."""
    seen = []
    run_batch_jobs(
        tool="t",
        jobs=[("line 1", "-prefix a")],
        device=torch.device("cpu"),
        parse_line=lambda line, base: _parse_toy(line, base),
        dispatch=lambda ra, dev: seen.append(ra.prefix),
        is_nested=lambda ra: getattr(ra, "batch", None) is not None,
        defaults=_parse_toy("-batch runs.txt -force lk"),
    )
    assert seen == ["a"]


def _parse_toy(line: str, base=None) -> argparse.Namespace:
    import shlex

    p = argparse.ArgumentParser()
    p.add_argument("-prefix", default=None)
    p.add_argument("-force", default="demons")
    p.add_argument("-batch", default=None)
    return p.parse_args(shlex.split(line), base)
