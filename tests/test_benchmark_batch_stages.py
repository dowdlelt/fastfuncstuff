"""Batched benchmark stages: one process per stage, nothing dropped.

The stages that loop over runs invoke their FFS tool once with a ``-batch``
manifest instead of once per run. Two things can go silently wrong there and
both are expensive: a flag that used to be on every per-run command line can
vanish from the batched command (the run still succeeds, with different
settings), and a partial rerun can go unflagged and poison the cached timing
baseline. These tests pin both, plus the manifest round-trip through
``shlex`` that the quoted ``-nwarp`` chain depends on.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from fastfuncstuff.benchmark.config import BenchmarkConfig
from fastfuncstuff.benchmark.runner import BenchmarkContext
from fastfuncstuff.benchmark.stages import automask, crossalign, moco, warp

TASKS = {"localizer": [1, 2], "rest": [1]}


@pytest.fixture
def ctx(tmp_path):
    c = BenchmarkContext(
        data_dir=tmp_path,
        device="cpu",
        config=BenchmarkConfig(dataset_id="ds000000", tasks=dict(TASKS)),
    )
    c.processing_dir.mkdir(parents=True, exist_ok=True)
    c._timing_meta = {}
    return c


@pytest.fixture
def commands(monkeypatch):
    """Capture every run_timed command instead of running it."""
    seen: list[str] = []

    def fake_run_timed(cmd, label, cwd, verbose=True):
        seen.append(cmd)
        return 1.0, None

    for mod in (moco, warp, crossalign, automask):
        monkeypatch.setattr(mod, "run_timed", fake_run_timed)
    return seen


def _manifest_lines(cmd: str, cwd: Path) -> list[str]:
    """The manifest a batched command points at, one argv-string per line."""
    argv = shlex.split(cmd)
    path = Path(argv[argv.index("-batch") + 1])
    if not path.is_absolute():
        path = cwd / path
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def test_moco_batches_all_runs_into_one_command(ctx, commands):
    ctx.func_dir.mkdir(parents=True, exist_ok=True)
    moco.run_ffs(ctx)

    assert len(commands) == 1, "one ffs_moco process for the whole stage"
    cmd = commands[0]
    # Settings shared by every run ride on the outer command, where the batch
    # runner turns them into each line's defaults.
    for flag in ("-interp heptic", "-final heptic", "-weight_automask", "-base 0", "-save_mean"):
        assert flag in cmd, f"{flag} dropped from the batched command"
    assert "-device cpu" in cmd

    lines = _manifest_lines(cmd, ctx.processing_dir)
    assert len(lines) == 3  # 2 localizer + 1 rest
    for line in lines:
        argv = shlex.split(line)
        for flag in ("-input", "-prefix", "-1Dfile", "-1Dmatrix_save"):
            assert flag in argv
    assert ctx._timing_meta["ffs"] == {"ran": 3, "total": 3}


def test_moco_skips_existing_and_flags_partial(ctx, commands):
    done = Path(moco._ffs_moco_path(ctx, "localizer", 1))
    done.parent.mkdir(parents=True, exist_ok=True)
    done.touch()

    moco.run_ffs(ctx)

    lines = _manifest_lines(commands[0], ctx.processing_dir)
    assert len(lines) == 2
    assert str(done) not in "\n".join(lines)
    # A rerun that skipped work is not a valid full-stage timing baseline.
    assert ctx._timing_meta["ffs"] == {"ran": 2, "total": 3}


def test_moco_all_present_runs_nothing(ctx, commands):
    for task, runs in ctx.all_task_run_pairs():
        for run in runs:
            p = Path(moco._ffs_moco_path(ctx, task, run))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

    assert moco.run_ffs(ctx) == 0.0
    assert commands == []
    assert ctx._timing_meta["ffs"] == {"ran": 0, "total": 3}


def test_moco_renames_means_after_the_batch(ctx, commands):
    """The mean lands beside -prefix as mean_<name>; the validator wants ours."""
    for task, runs in ctx.all_task_run_pairs():
        for run in runs:
            out = Path(moco._ffs_moco_path(ctx, task, run))
            (ctx.processing_dir / f"mean_{out.name}").touch()

    moco.run_ffs(ctx)

    for task, runs in ctx.all_task_run_pairs():
        for run in runs:
            assert Path(moco._ffs_mean_path(ctx, task, run)).exists()


def test_warp_manifest_keeps_the_quoted_nwarp_chain(ctx, commands, monkeypatch):
    """A warp chain is several space-separated paths in ONE -nwarp value.

    The manifest is shlex-split per line, so the quoting has to survive the
    round trip or the second path becomes a stray positional argument.
    """
    monkeypatch.setattr(warp, "_nwarp_chain", lambda c, t, r: "WARP.nii AFF.aff12.1D")

    warp.run_ffs(ctx)

    assert len(commands) == 1
    cmd = commands[0]
    for flag in ("-master", "-dxyz 3.0", "-interp wsinc5", "-no_autopad"):
        assert flag in cmd, f"{flag} dropped from the batched command"

    lines = _manifest_lines(cmd, ctx.processing_dir)
    assert len(lines) == 3
    for line in lines:
        argv = shlex.split(line)
        assert argv[argv.index("-nwarp") + 1] == "WARP.nii AFF.aff12.1D"
    assert ctx._timing_meta["ffs"] == {"ran": 3, "total": 3}


def test_crossalign_reports_items_for_both_roles(ctx, commands, monkeypatch):
    """Already batched, but it used to report nothing — so every partial rerun
    was cacheable as a full-stage baseline."""
    monkeypatch.setattr(crossalign, "_ref_mean", lambda c: ctx.processing_dir / "ref.nii")
    monkeypatch.setattr(crossalign, "_src_mean", lambda c, t, r: ctx.processing_dir / "src.nii")

    crossalign.run_ffs(ctx)
    assert ctx._timing_meta["ffs"]["total"] == len(crossalign._align_pairs(ctx))

    crossalign.run_ref(ctx)
    assert ctx._timing_meta["ref"]["total"] == len(crossalign._align_pairs(ctx))


def test_automask_reports_items_for_both_roles(ctx, commands):
    """Same gap as crossalign: batched, but silent about how much it ran."""
    n = sum(len(runs) for _, runs in ctx.all_task_run_pairs())

    automask.run_ffs(ctx)
    assert ctx._timing_meta["ffs"] == {"ran": n, "total": n}

    automask.run_ref(ctx)
    assert ctx._timing_meta["ref"] == {"ran": n, "total": n}
