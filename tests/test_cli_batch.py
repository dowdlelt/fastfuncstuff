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

    def parse_line(line: str) -> argparse.Namespace:
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
        parse_line=lambda line: _ns(out=str(done)),
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
            parse_line=lambda line: _ns(tag="good2" if line == "good2" else line),
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
            parse_line=lambda line: _ns(nested=True),
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
