"""Plan tests: reference resolution + warp-chain link-drop, and that the
emitted script is valid bash. Uses hand-built Subject trees (no disk)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from fastfuncstuff.autoproc.bids import BoldRun, FmapGroup, Session, Subject
from fastfuncstuff.autoproc.emit import write_script
from fastfuncstuff.autoproc.plan import Options, build_plan


def _run(session, task, run, tr=2.0, pe="j-"):
    return BoldRun(
        subject="X",
        session=session,
        task=task,
        run=run,
        mag_path=Path(
            f"/bids/sub-X/ses-{session}/func/sub-X_ses-{session}_task-{task}_run-{run}_bold.nii.gz"
        ),
        json={"RepetitionTime": tr, "PhaseEncodingDirection": pe},
    )


def _chain(plan, key):
    for pr in plan.runs:
        if (pr.bold.session, pr.bold.task, pr.bold.run) == key:
            return pr.warp_chain
    raise KeyError(key)


def test_no_fmap_single_session_first_run_anchor():
    subj = Subject("X", [Session("01", [_run("01", "foo", "1"), _run("01", "foo", "2")])])
    plan = build_plan(subj, Options(go_to_anat=True))
    # first run = anchor (identity): no wxrun, no blip, no xses.
    assert _chain(plan, ("01", "foo", "1")) == ["anat_lin", "moco"]
    # second run gets xrun-to-first-run.
    assert _chain(plan, ("01", "foo", "2")) == ["anat_lin", "wxrun_lin", "moco"]


def test_no_anat_stays_in_epi():
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")])])
    plan = build_plan(subj, Options(go_to_anat=False))
    assert _chain(plan, ("01", "foo", "1")) == ["moco"]


def test_fmap_anchored_all_runs_get_xrun():
    fmap = FmapGroup(
        "01", "foo", Path("/rev.nii.gz"), {"TotalReadoutTime": 0.06}, [("foo", "1"), ("foo", "2")]
    )
    subj = Subject("X", [Session("01", [_run("01", "foo", "1"), _run("01", "foo", "2")], [fmap])])
    plan = build_plan(subj, Options())
    # single (reference) fmap: xfmap dropped; blip + wxrun present for EVERY run.
    for run in ("1", "2"):
        assert _chain(plan, ("01", "foo", run)) == [
            "anat_lin",
            "blip_half",
            "wxrun_lin",
            "moco",
        ]


def test_multi_fmap_non_ref_gets_xfmap():
    f1 = FmapGroup("01", "a", Path("/a.nii.gz"), {}, [("a", "1")])
    f2 = FmapGroup("01", "b", Path("/b.nii.gz"), {}, [("b", "2")])
    subj = Subject("X", [Session("01", [_run("01", "a", "1"), _run("01", "b", "2")], [f1, f2])])
    plan = build_plan(subj, Options(fmap_ref=["a"]))
    assert "xfmap_lin" not in _chain(plan, ("01", "a", "1"))  # ref fmap
    assert "xfmap_lin" in _chain(plan, ("01", "b", "2"))  # non-ref fmap


def test_multi_fmap_xfmap_stage_and_premean():
    """Two fmap groups in one session: non-ref runs get xfmap in the chain, the
    xfmap stage is emitted, and premeans (fmap runs) feed the grandmean."""
    from fastfuncstuff.autoproc.bids import BoldRun

    def frun(task, run):
        return BoldRun(
            "X",
            "SM",
            task,
            run,
            Path(f"/bids/sub-X/ses-SM/func/x_{task}_{run}.nii.gz"),
            {"RepetitionTime": 2.0, "PhaseEncodingDirection": "j-"},
            sbref_path=Path(f"/sb_{task}_{run}.nii.gz"),
        )

    f_floc = FmapGroup(
        "SM", "floc", Path("/floc_rev.nii.gz"), {"TotalReadoutTime": 0.06}, [("floc", "1")]
    )
    f_prim = FmapGroup(
        "SM",
        "primary",
        Path("/prim_rev.nii.gz"),
        {"TotalReadoutTime": 0.08},
        [("primary", "1"), ("primary", "2")],
    )
    runs = [frun("floc", "1"), frun("primary", "1"), frun("primary", "2")]
    subj = Subject("X", [Session("SM", runs, [f_floc, f_prim])])
    plan = build_plan(subj, Options(fmap_ref=["floc"], xfmap_nonlin=True))

    # Task-preferred fmap assignment (run '1' exists in both tasks).
    by = {(r.bold.task, r.bold.run): r for r in plan.runs}
    assert by[("floc", "1")].fmap.fmap_id == "floc"
    assert by[("primary", "1")].fmap.fmap_id == "primary"  # not floc, despite run '1'

    # ref fmap (floc) drops xfmap; non-ref (primary) keeps it, in order.
    assert "xfmap_lin" not in by[("floc", "1")].warp_chain
    ch = by[("primary", "1")].warp_chain
    assert (
        ch.index("xfmap_nl") < ch.index("xfmap_lin") < ch.index("blip_half") < ch.index("wxrun_lin")
    )

    script = write_script(plan, "wd", bids_root="/bids")
    assert "stage05: cross-fmap" in script
    assert "stage04.blip.ses-SM.fmap-floc_mean" in script  # xfmap aligns to ref-fmap mean
    assert "stage07.premean." in script  # premeans feed the grandmean


def test_multi_session_xses_and_ref_drop():
    subj = Subject(
        "X",
        [
            Session("01", [_run("01", "foo", "1")]),
            Session("02", [_run("02", "foo", "1")]),
        ],
    )
    plan = build_plan(subj, Options(ref_ses="01", xses_nonlin=True))
    assert plan.ref_session == "01"
    assert "xses_lin" not in _chain(plan, ("01", "foo", "1"))  # reference session
    ch2 = _chain(plan, ("02", "foo", "1"))
    assert ch2.index("xses_nl") < ch2.index("xses_lin")  # nl before lin


def test_nonlinear_toggles_add_warps():
    fmap = FmapGroup("01", "foo", Path("/rev.nii.gz"), {}, [("foo", "1"), ("foo", "2")])
    subj = Subject("X", [Session("01", [_run("01", "foo", "1"), _run("01", "foo", "2")], [fmap])])
    plan = build_plan(subj, Options(xrun_nonlin=True, locomoco=True))
    ch = _chain(plan, ("01", "foo", "2"))
    assert ch == ["anat_lin", "blip_half", "wxrun_nl", "wxrun_lin", "locomoco", "moco"]


def test_grand_reference_chain_and_borrowed_anat():
    """primary→floc: anat matrix borrowed from the ref dir, grandmean aligned to
    the ref (xref), one last segment on current data (anat_nl)."""
    from fastfuncstuff.autoproc.emit import chain_files

    fmap = FmapGroup(
        "SM", "primary", Path("/rev.nii.gz"), {"TotalReadoutTime": 0.08}, [("primary", "1")]
    )
    subj = Subject("X", [Session("SM", [_run("SM", "primary", "1")], [fmap])])
    plan = build_plan(
        subj,
        Options(
            grand_reference="/floc.results",
            grand_reference_nonlin=True,
            anat_nonlin=True,
            tpm="/tpm.nii",
        ),
    )
    ch = _chain(plan, ("SM", "primary", "1"))
    assert ch == ["anat_lin", "xref_nl", "xref_lin", "anat_nl", "blip_half", "wxrun_lin", "moco"]
    files = chain_files(plan.runs[0], ".nii.zst", plan.options)
    # anat matrix is borrowed from the reference dir, not local.
    assert files[0] == "/floc.results/stage09.anat.aff12.1D"
    assert "stage09.xref.aff12.1D" in files
    assert "stage09.nlanat_invwarp.nii.zst" in files


def test_explicit_ref_file_uses_transforms_as_anat_link():
    """-ref_file/-ref_transforms replace the borrowed anat matrix with the
    user-supplied nwarp-order transforms."""
    from fastfuncstuff.autoproc.emit import chain_files

    subj = Subject("X", [Session("SM", [_run("SM", "primary", "1")])])
    plan = build_plan(
        subj,
        Options(ref_file="/ref_epi.nii.gz", ref_transforms=["/r2a_a.1D", "/r2a_b.1D"]),
    )
    files = chain_files(plan.runs[0], ".nii.zst", plan.options)
    assert files[:3] == ["/r2a_a.1D", "/r2a_b.1D", "stage09.xref.aff12.1D"]


def test_chain_files_are_produced_in_script():
    """Every in-script transform a CHAIN references must be produced by some stage
    command — i.e. the stage numbers in the chain builder and the stage bodies
    agree. This is the guard that keeps the emitted script runnable first try."""
    import re

    fmap = FmapGroup(
        "SM",
        "primary",
        Path("/rev.nii.gz"),
        {"TotalReadoutTime": 0.08},
        [("primary", "1"), ("primary", "2")],
    )
    runs = [_run("SM", "primary", "1"), _run("SM", "primary", "2")]
    subj = Subject("ME1", [Session("SM", runs, [fmap], anat=Path("/anat/T1w.nii.gz"))])
    plan = build_plan(subj, Options(want_nordic=True, locomoco=True, xrun_nonlin=True))
    script = write_script(plan, "workdir", bids_root="/bids")

    chain_by_key: dict[str, list[str]] = {}
    for m in re.finditer(r'CHAIN\[([^\]]+)\]="([^"]*)"', script):
        chain_by_key[m.group(1)] = m.group(2).split()
    frag_by_key = dict(re.findall(r"FRAG\[([^\]]+)\]=(\S+)", script))

    for tok in {t for toks in chain_by_key.values() for t in toks}:
        if not tok.startswith("stage"):
            continue  # borrowed/external path
        prefix = ".".join(tok.split(".")[:2])  # e.g. "stage02.moco"
        assert script.count(prefix) >= 2, f"{tok}: no producer with prefix {prefix}"

    # Full-coordinate guard: every run-specific CHAIN token (moco/nlmoco/xrun,
    # stages 02/03/06) must embed that run's exact FRAG coordinate — this is what
    # keeps the loop-written filenames (stage.label.${FRAG[$k]}) identical to the
    # CHAIN references. (A prefix-only check missed a real ses-01-vs-01 mismatch.)
    for key, toks in chain_by_key.items():
        frag = frag_by_key[key]
        for tok in toks:
            if re.match(r"stage0[236]\.", tok):
                assert frag in tok, f"{tok}: does not contain run coord {frag} (key {key})"


def test_warpmaster_defines_grid_mask_and_stats():
    """stage10a builds the warpmaster grid + epi_mask before stage10, stage10
    resamples onto it, and the GLM emits OLS+REML buckets masked by epi_mask.
    autobox3_brain (viewing only) is emitted only when we own the anat."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")], anat=Path("/anat/T1w.nii.gz"))])
    plan = build_plan(subj, Options(go_to_anat=True, run_glm=True))
    s = write_script(plan, "wd", bids_root="/bids")

    # warpmaster grid + mask are built, and defined before the resample stage.
    assert s.index("stage10a: warpmaster") < s.index("stage10: final compose")
    assert "ffs_util_autobox" in s and "ffs_util_resample" in s
    assert '-prefix "stage10.warpmaster.nii$FMT"' in s
    assert '-prefix "epi_mask.nii$FMT"' in s and "-dilate 2" in s
    assert '[ -f "autobox3_brain.nii.gz" ] ||' in s  # own anat → viewing brain

    # stage10 lands runs on the warpmaster (not the raw anat master).
    assert "-master stage10.warpmaster.nii$FMT" in s

    # GLM: OLS by default + REML, masked, scaled.
    assert '-Obuck "stage12.stats-ols.task-foo.nii$GLM_FMT"' in s
    assert '-Rbuck "stage12.stats-reml.task-foo.nii$GLM_FMT"' in s
    assert "-mask epi_mask.nii$FMT" in s and "-do_scale" in s

    # No-own-anat borrow mode: still a warpmaster, but no viewing brain.
    borrow = write_script(
        build_plan(subj, Options(grand_reference="/floc.results", run_glm=True)),
        "wd",
        bids_root="/bids",
    )
    assert '-prefix "stage10.warpmaster.nii$FMT"' in borrow
    assert '[ -f "autobox3_brain.nii.gz" ] ||' not in borrow


def test_output_format_flags_map_to_fmt_vars():
    """-format / -final_format / -glm_format set FMT / FINAL_FMT / GLM_FMT; the
    normalizer maps friendly spellings to the .nii-suffix."""
    from fastfuncstuff.cli.autoproc import _fmt_suffix

    assert _fmt_suffix("nii") == ""
    assert _fmt_suffix("gz") == ".gz" == _fmt_suffix("nii.gz") == _fmt_suffix(".gz")
    assert _fmt_suffix("zstd") == ".zst" == _fmt_suffix("zst") == _fmt_suffix("nii.zst")

    subj = Subject("X", [Session("01", [_run("01", "foo", "1")])])
    plan = build_plan(subj, Options(go_to_anat=True, fmt=".gz", final_fmt="", glm_fmt=".zst"))
    s = write_script(plan, "wd", bids_root="/bids")
    assert "\nFMT=.gz " in s
    assert "\nFINAL_FMT= " in s  # uncompressed .nii → empty suffix
    assert "\nGLM_FMT=.zst " in s

    # Defaults unchanged when the flags are absent.
    d = write_script(build_plan(subj, Options(go_to_anat=True)), "wd", bids_root="/bids")
    assert "\nFMT=.zst " in d and "\nFINAL_FMT=.gz " in d and "\nGLM_FMT=.gz " in d


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_emitted_script_is_valid_bash(tmp_path):
    fmap = FmapGroup(
        "SM",
        "primary",
        Path("/rev.nii.gz"),
        {"TotalReadoutTime": 0.08},
        [("primary", "1"), ("primary", "2")],
    )
    runs = [_run("SM", "primary", "1"), _run("SM", "primary", "2")]
    subj = Subject("ME1", [Session("SM", runs, [fmap], anat=Path("/anat/T1w.nii.gz"))])
    plan = build_plan(subj, Options(want_nordic=True, locomoco=True, xrun_nonlin=True))
    script = write_script(plan, "workdir", bids_root="/bids")
    sh = tmp_path / "proc.sh"
    sh.write_text(script)
    res = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
