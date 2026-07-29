"""Plan tests: reference resolution + warp-chain link-drop, and that the
emitted script is valid bash. Uses hand-built Subject trees (no disk)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from fastfuncstuff.autoproc.bids import BoldRun, FmapGroup, Session, Subject
from fastfuncstuff.autoproc.emit import write_script
from fastfuncstuff.autoproc.plan import Options, build_plan, effective_anat_source


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


def test_multi_fmap_xfmap_stage_and_runmean():
    """Two fmap groups in one session: non-ref runs get xfmap in the chain, the
    xfmap stage is emitted, and runmeans (fmap runs) feed the session mean."""
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
    assert "stage07.runmean." in script  # run means feed the session mean


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


def test_anat_nl_acts_after_xses():
    """The segment invwarp is estimated on the grandmean = REFERENCE-session
    space, so it must act on data xses has already brought there. In the
    leftmost-acts-first chain that means anat_nl is listed above xses_*."""
    subj = Subject(
        "X",
        [
            Session("01", [_run("01", "foo", "1")]),
            Session("02", [_run("02", "foo", "1")]),
        ],
    )
    plan = build_plan(subj, Options(ref_ses="01", xses_nonlin=True, anat_nonlin=True))
    ch = _chain(plan, ("02", "foo", "1"))
    assert ch.index("anat_lin") < ch.index("anat_nl") < ch.index("xses_nl")


def test_mean_levels_are_named_for_what_they_average():
    """runmean (one run) → sesmean (one session) → grandmean (everything). The
    grandmean is stage08 because it cannot exist before xses has aligned the
    sessions — and it has exactly ONE producer, single- or multi-session."""
    subj = Subject(
        "X",
        [
            Session("01", [_run("01", "foo", "1"), _run("01", "foo", "2")]),
            Session("02", [_run("02", "foo", "1")]),
        ],
    )
    s = write_script(build_plan(subj, Options(ref_ses="01")), "wd", bids_root="/bids")
    assert '-prefix "stage07.sesmean.ses-01.nii$FMT"' in s
    assert '-prefix "stage07.sesmean.ses-02.nii$FMT"' in s
    assert '-prefix "stage08.grandmean.nii$FMT"' in s
    # The old names are gone entirely.
    assert "premean" not in s and "stage07.grandmean" not in s
    # Session means are built before the alignment that makes the grandmean valid.
    assert s.index("stage07.sesmean.ses-02") < s.index("stage08: cross-session")
    assert s.index("stage08: cross-session") < s.index('-prefix "stage08.grandmean.nii$FMT"')
    # Exactly one command writes the grandmean.
    assert s.count('-prefix "stage08.grandmean.nii$FMT"') == 1


def test_single_session_grandmean_has_the_same_single_producer():
    """One session: stage08 still owns the grandmean (built from the lone sesmean),
    rather than a stage07 `cp` — same name, same stage, either way."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")])])
    s = write_script(build_plan(subj, Options()), "wd", bids_root="/bids")
    assert '-prefix "stage08.grandmean.nii$FMT"' in s
    assert s.count('-prefix "stage08.grandmean.nii$FMT"') == 1
    assert '"stage07.sesmean.ses-01.nii$FMT"' in s
    assert "cross-session alignment" not in s  # nothing to align


def test_reference_levels_get_a_role_ref_qc_copy():
    """Each alignment stage skips its own reference, leaving a gap when browsing.
    A `role-ref` copy fills it so the listing has one file per session/group/run."""
    f1 = FmapGroup("01", "a", Path("/a.nii.gz"), {"TotalReadoutTime": 0.06}, [("foo", "1")])
    f2 = FmapGroup("01", "b", Path("/b.nii.gz"), {"TotalReadoutTime": 0.06}, [("bar", "1")])
    subj = Subject(
        "X",
        [
            Session("01", [_run("01", "foo", "1"), _run("01", "bar", "1")], [f1, f2]),
            Session("02", [_run("02", "foo", "1")]),
        ],
    )
    s = write_script(
        build_plan(subj, Options(ref_ses="01", fmap_ref=["a"])), "wd", bids_root="/bids"
    )
    # xses: reference session has no transform, but appears in the listing.
    assert 'cp -f "stage07.sesmean.ses-01.nii$FMT" "stage08.xses.ses-01.role-ref.nii$FMT"' in s
    assert '-prefix \\"stage08.xses.ses-02.nii$FMT\\"' in s  # the real, aligned one (batched)
    # xfmap: same for the reference fieldmap group.
    assert 'cp -f "stage04.blip.ses-01.fmap-a_mean.nii$FMT" ' in s
    assert '"stage05.xfmap.ses-01.fmap-a.role-ref.nii$FMT"' in s
    # A reference has no nonlinear counterpart, by definition.
    assert "role-ref_nl" not in s
    # Markers are QC only — never fed into a warp chain or a mean.
    assert "-nwarp" in s and "role-ref" not in s.split("CHAIN[")[-1].split("\n\n")[0]


def test_xrun_anchor_gets_a_role_ref_copy_only_without_fieldmaps():
    """No fmaps → the session's first run is the anchor and has no xrun output, so
    it gets a marker. With fmaps EVERY run gets a real xrun; no gap, no marker."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1"), _run("01", "foo", "2")])])
    s = write_script(build_plan(subj, Options()), "wd", bids_root="/bids")
    assert '"stage06.xrun.ses-01.task-foo.run-1.role-ref.nii$FMT"' in s
    assert '"stage02.moco.ses-01.task-foo.run-1_mean.nii$FMT"' in s

    fmap = FmapGroup("01", "f", Path("/rev.nii.gz"), {}, [("foo", "1"), ("foo", "2")])
    with_fmap = Subject(
        "X", [Session("01", [_run("01", "foo", "1"), _run("01", "foo", "2")], [fmap])]
    )
    s2 = write_script(build_plan(with_fmap, Options()), "wd", bids_root="/bids")
    assert "stage06.xrun" in s2 and "xrun.ses-01.task-foo.run-1.role-ref" not in s2


def test_nonlinear_toggles_add_warps():
    fmap = FmapGroup("01", "foo", Path("/rev.nii.gz"), {}, [("foo", "1"), ("foo", "2")])
    subj = Subject("X", [Session("01", [_run("01", "foo", "1"), _run("01", "foo", "2")], [fmap])])
    plan = build_plan(subj, Options(xrun_nonlin=True, locomoco=True))
    ch = _chain(plan, ("01", "foo", "2"))
    assert ch == ["anat_lin", "blip_half", "wxrun_nl", "wxrun_lin", "locomoco", "moco"]


def test_locomoco_consumes_the_moco_timeseries_not_the_mean():
    """locomoco estimates a warp per volume, so stage02 must write its corrected 4D
    (-prefix) and stage03 must read that, not the mean. Without locomoco the 4D is
    not written at all — the only resample is stage10's."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")])])

    s = write_script(build_plan(subj, Options(locomoco=True)), "wd", bids_root="/bids")
    assert '-prefix \\"${mstem}.nii$FMT\\"' in s
    assert '-input "stage02.moco.${FRAG[$k]}.nii$FMT"' in s
    # and the mean that goes downstream is locomoco's, not moco's
    assert "stage03.nlmoco.ses-01.task-foo.run-1_locomoco_mean.nii$FMT" in s

    s2 = write_script(build_plan(subj, Options(locomoco=False)), "wd", bids_root="/bids")
    assert '-prefix \\"${mstem}.nii$FMT\\"' not in s2


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


def test_fieldmap_jacobian_rides_every_application_of_the_blip_warp():
    """ffs_blipflip's warp is geometry-only: applying it without ``-jac`` leaves
    the signal pile-up at compression edges. Every stage that re-applies it to
    data (stage07 runmeans, stage10 final + SBRef) must ask for the modulation,
    and must name the fieldmap so locomoco's PE warp is not picked instead."""
    import re

    fmap = FmapGroup(
        "SM", "PA", Path("/rev.nii.gz"), {"TotalReadoutTime": 0.08}, [("t", "1"), ("t", "2")]
    )
    runs = [_run("SM", "t", "1"), _run("SM", "t", "2")]
    for b in runs:
        b.sbref_path = Path(f"/bids/sb_{b.run}.nii.gz")
    subj = Subject("ME1", [Session("SM", runs, [fmap])])
    plan = build_plan(subj, Options(go_to_anat=False, locomoco=True))
    script = write_script(plan, "workdir", bids_root="/bids")

    jacs = set(re.findall(r'JAC\[[^\]]+\]="([^"]*)"', script))
    assert jacs == {"j:stage04.blip.ses-SM.fmap-PA_warp.nii$FMT"}
    # The named warp is actually in the chains it modulates.
    assert 'CHAIN[SM:t:1]="' in script
    for m in re.finditer(r'(?:CHAIN|SBCHAIN|PRECHAIN)\[[^\]]+\]="([^"]*)"', script):
        assert "stage04.blip.ses-SM.fmap-PA_warp.nii$FMT" in m.group(1).split()
    # stage07 runmean, stage10 BOLD and stage10 SBRef all pass it.
    assert script.count("${JAC[$k]:+") == 4  # two lanes in stage07, BOLD, SBRef

    # A run with no fieldmap has no JAC entry at all (nothing to modulate).
    plain = build_plan(
        Subject("ME1", [Session("SM", [_run("SM", "t", "1")])]), Options(go_to_anat=False)
    )
    assert "JAC[" not in write_script(plain, "workdir", bids_root="/bids").split("stage07")[0]


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


def _two_fmap_subject():
    """One session, two fieldmap groups (floc = reference), each with a run."""
    f1 = FmapGroup(
        "SM", "floc", Path("/floc_rev.nii.gz"), {"TotalReadoutTime": 0.06}, [("floc", "1")]
    )
    f2 = FmapGroup(
        "SM", "prim", Path("/prim_rev.nii.gz"), {"TotalReadoutTime": 0.08}, [("prim", "1")]
    )
    runs = [_run("SM", "floc", "1"), _run("SM", "prim", "1")]
    return Subject("X", [Session("SM", runs, [f1, f2], anat=Path("/anat/T1w.nii.gz"))])


def test_anat_source_grandmean_is_the_default():
    plan = build_plan(_two_fmap_subject(), Options(go_to_anat=True, fmap_ref=["floc"]))
    s = write_script(plan, "wd", bids_root="/bids")
    assert '-source "stage08.grandmean.nii$FMT"' in s
    assert "stage05.fmapmean" not in s  # not built when nothing asks for it


def test_anat_source_ref_fmap_uses_the_reference_blip_mean():
    """-anat_source ref_fmap aligns the anat to the image that DEFINES the space
    (the reference group's undistorted mean) instead of the grandmean."""
    plan = build_plan(
        _two_fmap_subject(), Options(go_to_anat=True, fmap_ref=["floc"], anat_source="ref_fmap")
    )
    s = write_script(plan, "wd", bids_root="/bids")
    assert '-source "stage04.blip.ses-SM.fmap-floc_mean.nii$FMT"' in s
    assert '-source "stage08.grandmean.nii$FMT"' not in s
    # The grandmean is still built (QC + xses/xref inputs) — only the anat moved.
    assert '-prefix "stage08.grandmean.nii$FMT"' in s


def test_anat_source_mean_fmap_averages_the_aligned_group_means():
    plan = build_plan(
        _two_fmap_subject(), Options(go_to_anat=True, fmap_ref=["floc"], anat_source="mean_fmap")
    )
    s = write_script(plan, "wd", bids_root="/bids")
    fm = "stage05.fmapmean.ses-SM.nii$FMT"
    assert f'-prefix "{fm}"' in s
    # Averaged: reference group's own blip mean + the non-ref group's ALIGNED mean.
    assert '"stage04.blip.ses-SM.fmap-floc_mean.nii$FMT"' in s
    assert '"stage05.xfmap.ses-SM.fmap-prim.nii$FMT"' in s
    assert f'-source "{fm}"' in s  # and it is what the anat aligns
    # Built after the xfmap alignment that puts the groups in one space.
    assert s.index("stage05: cross-fmap") < s.index(f'-prefix "{fm}"')


def test_anat_source_mean_fmap_uses_nonlinear_xfmap_result_when_available():
    plan = build_plan(
        _two_fmap_subject(),
        Options(go_to_anat=True, fmap_ref=["floc"], anat_source="mean_fmap", xfmap_nonlin=True),
    )
    s = write_script(plan, "wd", bids_root="/bids")
    assert '"stage05.xfmap.ses-SM.fmap-prim_nl.nii$FMT"' in s


def test_anat_source_falls_back_to_grandmean_without_fieldmaps():
    """No fmaps → the grandmean is the only EPI image there is (ffs_segment is
    what recovers the distortion), so both fmap choices degrade to it."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")], anat=Path("/anat/T1w.nii.gz"))])
    for mode in ("ref_fmap", "mean_fmap"):
        plan = build_plan(subj, Options(go_to_anat=True, anat_source=mode))
        assert effective_anat_source(plan) == "grandmean"
        s = write_script(plan, "wd", bids_root="/bids")
        assert '-source "stage08.grandmean.nii$FMT"' in s
        assert f"(-anat_source {mode} unavailable here → grandmean)" in s


def test_anat_source_mean_fmap_degenerates_to_ref_fmap_with_one_group():
    """Averaging a single group's mean is just that mean — don't emit a 3dmath."""
    fmap = FmapGroup("01", "only", Path("/rev.nii.gz"), {}, [("foo", "1")])
    subj = Subject(
        "X", [Session("01", [_run("01", "foo", "1")], [fmap], anat=Path("/anat/T1w.nii.gz"))]
    )
    plan = build_plan(subj, Options(go_to_anat=True, anat_source="mean_fmap"))
    assert effective_anat_source(plan) == "ref_fmap"
    s = write_script(plan, "wd", bids_root="/bids")
    assert "stage05.fmapmean" not in s
    assert '-source "stage04.blip.ses-01.fmap-only_mean.nii$FMT"' in s


def test_anat_source_does_not_change_the_warp_chain():
    """All three sources live on the reference-fmap grid, so the composed chain
    must be byte-identical whichever is chosen — only the image content differs."""
    subj = _two_fmap_subject()
    chains = []
    for mode in ("grandmean", "ref_fmap", "mean_fmap"):
        plan = build_plan(subj, Options(go_to_anat=True, fmap_ref=["floc"], anat_source=mode))
        chains.append([pr.warp_chain for pr in plan.runs])
    assert chains[0] == chains[1] == chains[2]


def test_segment_input_shares_the_anat_source_vocabulary():
    """-anat_nonlin_input accepts ref_fmap/mean_fmap too, and asking for mean_fmap
    there alone is enough to make stage05 build it."""
    plan = build_plan(
        _two_fmap_subject(),
        Options(
            go_to_anat=True,
            fmap_ref=["floc"],
            anat_nonlin=True,
            anat_nonlin_input="mean_fmap",
            tpm="/tpm.nii.gz",
        ),
    )
    s = write_script(plan, "wd", bids_root="/bids")
    fm = "stage05.fmapmean.ses-SM.nii$FMT"
    assert f'-prefix "{fm}"' in s  # built even though -anat_source stayed grandmean
    assert f'ffs_segment \\\n    -input "{fm}"' in s
    assert '-source "stage08.grandmean.nii$FMT"' in s  # anat linear step unchanged


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


def _phase_run(session, task, run, **kw):
    """A run that also has a part-phase bold (what -phase_proc requires)."""
    r = _run(session, task, run, **kw)
    r.phase_path = Path(str(r.mag_path).replace("_bold.nii.gz", "_part-phase_bold.nii.gz"))
    return r


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("nordic", [False, True])
@pytest.mark.parametrize("stc", ["integrate", "first"])
def test_phase_proc_unwraps_then_rides_the_final_resample(tmp_path, nordic, stc):
    """-phase_proc: ROMEO unwrap happens once up front (after NORDIC when NORDIC
    runs), the phase is untouched in between, and the SAME warp chain carries it
    at stage10 into a separately-labelled part-phase output."""
    subj = Subject("X", [Session("SM", [_phase_run("SM", "primary", "1")])])
    opt = Options(
        phase_proc=True,
        want_nordic=nordic,
        noise_vols=3,
        slicetiming_method=stc,
        go_to_anat=False,
        distortion=False,
    )
    s = write_script(build_plan(subj, opt), "wd", bids_root="/bids")

    assert (
        subprocess.run(  # noqa: S603
            ["bash", "-n", str(_write(tmp_path / "p.sh", s))], capture_output=True, text=True
        ).returncode
        == 0
    )

    # ROMEO is an external dependency; the script's preflight must catch a missing one.
    assert " romeo; do" in s
    assert "PHASE_FMT=.gz" in s  # ROMEO can't read .zst
    assert 'romeo -p "$ph" -m "$mag" -o "$uw" -t epi -v' in s

    unwrapped = "stage00.unwrap.${FRAG[$k]}.part-phase.nii$PHASE_FMT"
    if nordic:
        # NORDIC already wrote the denoised pair AND trimmed the noise volumes.
        assert 'ph="stage00.nordic.${FRAG[$k]}_phase.nii$PHASE_FMT"' in s
        assert "stage00.trim." not in s
        assert s.index("stage00: NORDIC") < s.index("stage00: phase unwrap")
    else:
        # No NORDIC: noise volumes must come off BOTH before unwrapping, and
        # ROMEO needs real files rather than [0..n] selectors.
        assert 'tp="stage00.trim.${FRAG[$k]}.part-phase.nii$PHASE_FMT"' in s
        assert '-input "${ph}[0..$last]"' in s

    # Slice timing done up front must be applied to the phase too, and that
    # tshifted phase is what reaches stage10.
    if stc == "first":
        assert f'-input "{unwrapped}"' in s
        phase_in = "stage01.tshift.${FRAG[$k]}.part-phase.nii$PHASE_FMT"
    else:
        phase_in = unwrapped
    assert f'-phase \\"{phase_in}\\" -phase_units rad -phase_warp direct' in s
    assert '-phase_prefix \\"$phoutf\\"' in s
    assert 'phoutf="stage10.final.${FRAG[$k]}.part-phase.nii$FINAL_FMT"' in s

    # Nothing between stage00 and stage10 touches the phase: no other stage
    # references a part-phase file.
    body = s[s.index("stage02: motion correction") : s.index("stage10: compose + resample")]
    assert "part-phase" not in body


def _write(path, text):
    path.write_text(text)
    return path


def test_phase_proc_off_emits_no_phase_machinery():
    subj = Subject("X", [Session("SM", [_phase_run("SM", "primary", "1")])])
    s = write_script(build_plan(subj, Options(go_to_anat=False, distortion=False)), "wd")
    # The PHASE[] data-array entry is always emitted (NORDIC may want it); the
    # unwrap stage, ROMEO, and the phase-side outputs must not be.
    assert "PHASE_FMT" not in s and "romeo" not in s
    assert "stage00.unwrap" not in s and "-phase_prefix" not in s


def _bids_with_events(tmp_path: Path, n_tp: int = 12):
    """Minimal on-disk BIDS: 2 part-mag runs of a task, each with an events TSV."""
    import nibabel as nib
    import numpy as np

    func = tmp_path / "sub-ME1" / "ses-SM" / "func"
    func.mkdir(parents=True)
    for r in ("01", "02"):
        base = f"sub-ME1_ses-SM_task-floc_run-{r}"
        img = nib.Nifti1Image(np.zeros((2, 2, 2, n_tp), dtype=np.float32), np.eye(4))
        img.header.set_zooms((1.0, 1.0, 1.0, 2.0))
        nib.save(img, func / f"{base}_part-mag_bold.nii.gz")
        (func / f"{base}_part-mag_bold.json").write_text(
            '{"RepetitionTime": 2.0, "PhaseEncodingDirection": "j-"}'
        )
        (func / f"{base}_events.tsv").write_text(
            "onset\tduration\ttrial_type\n2\t4\tface\n10\t4\tplace\n"
        )
    return func


def test_glm_stage_is_spec_driven_and_preflight_checks_its_inputs(tmp_path: Path):
    """The GLM's model is the design TOML, not the command line: stage12 runs
    `ffs_reml -spec`, and preflight checks BOTH the spec and the events files it
    names — a missing one should fail in the first seconds, not after an hour of
    preprocessing."""
    from fastfuncstuff.autoproc.bids import scan_subject
    from fastfuncstuff.autoproc.glm import write_design_specs

    _bids_with_events(tmp_path)
    subj = scan_subject(tmp_path, "ME1")
    plan = build_plan(subj, Options(run_glm=True, glm_ortvec=["motion", "motion_deriv"]))
    s = write_script(plan, str(tmp_path / "wd"), bids_root=str(tmp_path))

    assert "-spec stage11.design.task-floc.toml" in s
    assert "-events" not in s  # events live in the spec now
    preflight_block = s.split("MISSING INPUT")[0]
    assert "stage11.design.task-floc.toml" in preflight_block
    # Preflight checks the stimuli/ copies — those are what the spec names.
    for r in ("01", "02"):
        assert f"stimuli/sub-ME1_ses-SM_task-floc_run-{r}_events.tsv" in preflight_block

    rows = write_design_specs(plan, str(tmp_path), str(tmp_path / "wd"))
    assert [(t, st) for t, _, st in rows] == [("floc", "wrote")]
    spec = (tmp_path / "wd" / "stage11.design.task-floc.toml").read_text()
    # Nuisance blocks come from the -glm_ortvec registry, deriv included.
    assert 'label = "motion"' in spec and 'label = "motion_deriv"' in spec
    assert 'transform = "deriv"' in spec
    assert "locomoco" not in spec  # not requested, and locomoco is off anyway
    # Both trial types were scanned out of the events files.
    assert 'trial_type = "face"' in spec and 'trial_type = "place"' in spec


def test_design_spec_describes_the_preprocessed_runs_not_the_raw_ones(tmp_path: Path):
    """The spec is written before stage10 exists. It must name stage10's outputs
    while taking TR/length from the raw headers — and honour trimmed noise
    volumes, which are the one thing that changes run length."""
    from fastfuncstuff.autoproc.bids import scan_subject
    from fastfuncstuff.autoproc.glm import write_design_specs

    _bids_with_events(tmp_path, 12)
    subj = scan_subject(tmp_path, "ME1")
    plan = build_plan(subj, Options(run_glm=True, noise_vols=2))
    write_design_specs(plan, str(tmp_path), str(tmp_path / "wd"))

    from fastfuncstuff.design.spec import load_spec

    spec = load_spec(tmp_path / "wd" / "stage11.design.task-floc.toml")
    assert spec.meta.tr == 2.0
    assert spec.meta.n_timepoints_per_run == [10, 10]  # 12 acquired - 2 noise volumes
    assert [r.bold for r in spec.meta.runs] == [
        "stage10.final.ses-SM.task-floc.run-01.nii.gz",
        "stage10.final.ses-SM.task-floc.run-02.nii.gz",
    ]


def test_events_are_copied_into_stimuli_and_named_relatively(tmp_path: Path):
    """The timing that produced a stat map should travel with it: events are
    copied into <work_dir>/stimuli/ and the spec names the copy, relative to the
    working dir the script cds into — not the BIDS path, which may not be mounted
    (or may have been edited) by the time the GLM is re-run."""
    from fastfuncstuff.autoproc.bids import scan_subject
    from fastfuncstuff.autoproc.glm import write_design_specs
    from fastfuncstuff.design.spec import load_spec

    func = _bids_with_events(tmp_path)
    subj = scan_subject(tmp_path, "ME1")
    wd = tmp_path / "wd"
    plan = build_plan(subj, Options(run_glm=True))
    write_design_specs(plan, str(tmp_path), str(wd))

    spec = load_spec(wd / "stage11.design.task-floc.toml")
    assert [r.events for r in spec.meta.runs] == [
        "stimuli/sub-ME1_ses-SM_task-floc_run-01_events.tsv",
        "stimuli/sub-ME1_ses-SM_task-floc_run-02_events.tsv",
    ]
    for r in spec.meta.runs:
        copy = wd / r.events
        assert copy.is_file()
        assert copy.read_text() == (func / Path(r.events).name).read_text()
    # The trial types still came out of the copied files.
    assert {e.trial_type for e in spec.events} == {"face", "place"}


def _bids_two_tasks(tmp_path: Path, n_tp: int = 12):
    """Two tasks: 'floc' with BIDS columns, 'imagery' with a 'condition' column
    (and its trial_type column holding something coarser)."""
    import nibabel as nib
    import numpy as np

    func = tmp_path / "sub-ME1" / "ses-SM" / "func"
    func.mkdir(parents=True)
    rows = {
        "floc": "onset\tduration\ttrial_type\n2\t4\tface\n10\t4\tplace\n",
        "imagery": (
            "onset\tduration\ttrial_type\tcondition\n2\t4\tcue\tcue_A\n10\t4\tcue\tcue_B\n"
        ),
    }
    for task, text in rows.items():
        base = f"sub-ME1_ses-SM_task-{task}_run-01"
        img = nib.Nifti1Image(np.zeros((2, 2, 2, n_tp), dtype=np.float32), np.eye(4))
        img.header.set_zooms((1.0, 1.0, 1.0, 2.0))
        nib.save(img, func / f"{base}_part-mag_bold.nii.gz")
        (func / f"{base}_part-mag_bold.json").write_text(
            '{"RepetitionTime": 2.0, "PhaseEncodingDirection": "j-"}'
        )
        (func / f"{base}_events.tsv").write_text(text)
    return func


def test_event_cols_are_per_task_and_reach_the_design_toml(tmp_path: Path):
    """-spec_event_cols sets the columns for every task; -sep_spec_event_cols
    overrides one task, so a single oddly-columned task does not force the rest
    to be spelled out. Both end up as events_columns in that task's TOML."""
    from fastfuncstuff.autoproc.bids import scan_subject
    from fastfuncstuff.autoproc.glm import write_design_specs
    from fastfuncstuff.design.spec import load_spec

    _bids_two_tasks(tmp_path)
    subj = scan_subject(tmp_path, "ME1")
    plan = build_plan(
        subj,
        Options(run_glm=True, sep_spec_event_cols={"imagery": ("onset", "duration", "condition")}),
    )
    write_design_specs(plan, str(tmp_path), str(tmp_path / "wd"))

    imagery = load_spec(tmp_path / "wd" / "stage11.design.task-imagery.toml")
    assert imagery.meta.events_columns.trial_type == "condition"
    # The trial types were scanned through THAT column, not trial_type.
    assert {e.trial_type for e in imagery.events} == {"cue_A", "cue_B"}

    floc = load_spec(tmp_path / "wd" / "stage11.design.task-floc.toml")
    assert floc.meta.events_columns.trial_type == "trial_type"
    assert {e.trial_type for e in floc.events} == {"face", "place"}


def test_event_cols_fall_back_to_bids_defaults_with_a_warning(tmp_path: Path):
    """A column that is not in the file would compile to an empty design an hour
    of preprocessing later. It is caught against the real header, warned about,
    and the task falls back to the BIDS defaults — generation still finishes."""
    from fastfuncstuff.autoproc.bids import scan_subject
    from fastfuncstuff.autoproc.glm import resolve_event_cols, write_design_specs
    from fastfuncstuff.design.spec import load_spec

    func = _bids_two_tasks(tmp_path)
    subj = scan_subject(tmp_path, "ME1")
    opt = Options(run_glm=True, spec_event_cols=("onset", "duration", "nope"))
    plan = build_plan(subj, opt)

    cols, warn = resolve_event_cols(
        "floc", [func / "sub-ME1_ses-SM_task-floc_run-01_events.tsv"], opt
    )
    assert cols is None
    assert "nope" in warn and "onset/duration/trial_type" in warn

    rows = write_design_specs(plan, str(tmp_path), str(tmp_path / "wd"))
    assert all(status == "wrote" for _, _, status in rows)  # finished anyway
    spec = load_spec(tmp_path / "wd" / "stage11.design.task-floc.toml")
    assert spec.meta.events_columns.trial_type == "trial_type"
    assert {e.trial_type for e in spec.events} == {"face", "place"}


def test_existing_design_spec_is_never_clobbered(tmp_path: Path):
    """Re-running ffs_autoproc is routine; silently discarding an edited model
    would be the worst bug this tool could have."""
    from fastfuncstuff.autoproc.bids import scan_subject
    from fastfuncstuff.autoproc.glm import write_design_specs

    _bids_with_events(tmp_path)
    subj = scan_subject(tmp_path, "ME1")
    wd = str(tmp_path / "wd")
    plan = build_plan(subj, Options(run_glm=True))
    write_design_specs(plan, str(tmp_path), wd)

    dest = tmp_path / "wd" / "stage11.design.task-floc.toml"
    dest.write_text(dest.read_text() + "\n# my edit\n")

    rows = write_design_specs(plan, str(tmp_path), wd)
    assert rows[0][2] == "kept"
    assert "# my edit" in dest.read_text()

    plan2 = build_plan(subj, Options(run_glm=True, glm_spec_overwrite=True))
    rows = write_design_specs(plan2, str(tmp_path), wd)
    assert rows[0][2] == "wrote"
    assert "# my edit" not in dest.read_text()


def test_task_without_events_falls_back_to_flags_with_a_todo(tmp_path: Path):
    """No events → no spec to build. The script must say so rather than emit a
    command that looks like it works."""
    from fastfuncstuff.autoproc.bids import scan_subject
    from fastfuncstuff.autoproc.glm import write_design_specs

    func = _bids_with_events(tmp_path)
    for f in func.glob("*_events.tsv"):
        f.unlink()
    subj = scan_subject(tmp_path, "ME1")
    plan = build_plan(subj, Options(run_glm=True, glm_ortvec=["motion", "motion_deriv"]))

    rows = write_design_specs(plan, str(tmp_path), str(tmp_path / "wd"))
    assert rows[0][2].startswith("skipped")
    s = write_script(plan, str(tmp_path / "wd"), bids_root=str(tmp_path))
    assert "TODO task-floc" in s
    assert "-spec stage11" not in s
    # The registry still drives the fallback flags, transform modifier and all.
    assert "-ortvec_glob 'stage02.moco.*task-floc*.motion.1D' motion" in s
    assert "motion_deriv:deriv" in s


# ---------------------------------------------------------------------------
# SBRef lane
# ---------------------------------------------------------------------------


def _sbrun(session, task, run, sbref=True):
    b = _run(session, task, run)
    if sbref:
        b.sbref_path = Path(f"/bids/sb_{session}_{task}_{run}.nii.gz")
    return b


def test_sbref_lane_is_all_or_nothing_and_needs_moco_ref_sbref():
    """One run without an SBRef turns the lane off for everyone: a session mean
    that silently mixed SBRef and BOLD-mean contrast would be worse than either
    lane alone. So does a moco base that isn't the SBRef — the lane's whole
    premise is that the SBRef IS the run's post-moco space."""
    from fastfuncstuff.autoproc.plan import sbref_chain

    full = Subject("X", [Session("01", [_sbrun("01", "t", "1"), _sbrun("01", "t", "2")])])
    assert build_plan(full, Options()).use_sbref

    partial = Subject(
        "X", [Session("01", [_sbrun("01", "t", "1"), _sbrun("01", "t", "2", sbref=False)])]
    )
    assert not build_plan(partial, Options()).use_sbref

    assert not build_plan(full, Options(moco_ref="first")).use_sbref

    # The SBRef rides everything from the fieldmap level up, but never the
    # within-run motion tokens it defines rather than needs.
    plan = build_plan(full, Options(locomoco=True))
    pr = plan.runs[1]
    assert "moco" in pr.warp_chain and "locomoco" in pr.warp_chain
    assert "moco" not in sbref_chain(pr) and "locomoco" not in sbref_chain(pr)
    assert sbref_chain(pr) == [t for t in pr.warp_chain if t not in ("moco", "locomoco")]


def test_sbref_lane_estimates_transforms_and_keeps_the_mean_lane():
    """xrun/xses are estimated from SBRefs, but the BOLD-mean pyramid survives
    intact — same transforms, other lineage — so the two are comparable."""
    subj = Subject(
        "X",
        [
            Session("01", [_sbrun("01", "t", "1"), _sbrun("01", "t", "2")]),
            Session("02", [_sbrun("02", "t", "1")]),
        ],
    )
    s = write_script(build_plan(subj, Options(ref_ses="01")), "wd", bids_root="/bids")

    # xrun source is the SBRef, not the moco mean; the matrix stays lane-free so
    # both lanes are resampled by exactly the same file. (Batched stage: the args
    # live in a manifest line, so their quotes are backslash-escaped.)
    assert '-source \\"${SBREF[$k]}\\"' in s
    assert '-1Dmatrix_save \\"${xstem}.aff12.1D\\"' in s
    assert '-source "${MOCOMEAN[$k]}"' not in s.split("stage07")[0]

    # xses is estimated on the SBRef session means...
    assert '-source \\"stage07.sesmean.ses-02.src-sbref.nii$FMT\\"' in s
    # ...and re-applied to the BOLD-mean one (an ffs_nwarp manifest line), so both
    # grandmeans exist.
    assert '-prefix \\"stage08.xses.ses-02.nii$FMT\\"' in s
    assert '-prefix "stage08.grandmean.src-sbref.nii$FMT"' in s
    assert '-prefix "stage08.grandmean.nii$FMT"' in s

    # Both lanes' session means are built from their own runmeans.
    assert "stage07.runmean.ses-01.task-t.run-2.src-sbref.nii$FMT" in s
    assert "stage07.runmean.ses-01.task-t.run-2.nii$FMT" in s


def test_sbrefs_reach_the_final_space():
    """Every run's SBRef lands on the warpmaster grid alongside its timeseries —
    the sharpest per-run alignment QC the pipeline can produce."""
    subj = Subject("X", [Session("01", [_sbrun("01", "t", "1"), _sbrun("01", "t", "2")])])
    s = write_script(build_plan(subj, Options()), "wd", bids_root="/bids")
    assert 'sboutf="stage10.final.${FRAG[$k]}.src-sbref.nii$FINAL_FMT"' in s
    assert '-nwarp \\"${SBCHAIN[$k]}\\"' in s
    # SBRefs are load-bearing once the lane is on, so preflight checks them.
    assert "/bids/sb_01_t_1.nii.gz" in s.split("stage02")[0]

    # A run whose SBRef needs no transform at all (no anat, no fmap, anchor run)
    # cannot go through ffs_nwarp — there is no identity warp. It regrids instead.
    bare = write_script(
        build_plan(subj, Options(go_to_anat=False, distortion=False)), "wd", bids_root="/bids"
    )
    assert 'if [ -z "${SBCHAIN[$k]:-}" ]; then' in bare
    assert 'ffs_util_resample -input "${SBREF[$k]}"' in bare


def test_anat_source_sbmean_needs_the_lane():
    subj_sb = Subject("X", [Session("01", [_sbrun("01", "t", "1")])])
    subj_no = Subject("X", [Session("01", [_run("01", "t", "1")])])
    assert effective_anat_source(build_plan(subj_sb, Options(anat_source="sbmean"))) == "sbmean"
    assert effective_anat_source(build_plan(subj_no, Options(anat_source="sbmean"))) == "grandmean"

    s = write_script(build_plan(subj_sb, Options(anat_source="sbmean")), "wd", bids_root="/bids")
    assert '-source "stage08.grandmean.src-sbref.nii$FMT"' in s
