"""Plan tests: reference resolution + warp-chain link-drop, and that the
emitted script is valid bash. Uses hand-built Subject trees (no disk)."""

from __future__ import annotations

import re
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
    # xfmap aligns to the ref group's FORWARD image, not the pair mean.
    assert "stage04.blip.ses-SM.fmap-floc_unwarped.nii$FMT[0]" in script
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
    # xses: reference session has no transform, but appears in the listing. The
    # marker is the image that actually served as the alignment base — the primary
    # lane's session representative (the max lane, with no SBRefs here).
    assert (
        'cp -f "stage07.sesmean.ses-01.src-max.nii$FMT" "stage08.xses.ses-01.role-ref_lin.nii$FMT"'
        in s
    )
    # the real, aligned one (batched); tagged with the lane that estimated it.
    assert '-prefix \\"stage08.xses.ses-02.src-max_lin.nii$FMT\\"' in s
    # xfmap: same for the reference fieldmap group.
    assert '-input "stage04.blip.ses-01.fmap-a_unwarped.nii$FMT[0]"' in s
    assert '"stage05.xfmap.ses-01.fmap-a.role-ref_lin.nii$FMT"' in s
    # A reference has no nonlinear counterpart, by definition.
    assert "role-ref_nl" not in s
    # Markers are QC only — never fed into a warp chain or a mean.
    assert "-nwarp" in s and "role-ref" not in s.split("CHAIN[")[-1].split("\n\n")[0]


def test_alignment_images_pair_as_lin_and_nl_while_the_warp_stays_lane_free():
    """An alignment stage's two images differ only by `_lin` / `_nl`, so the pair
    reads as one thing refined twice and the nl file names what fed it. The warp
    beside the nl image is shared by every lane, so it keeps the lane-free stem —
    which is exactly what the chain references."""
    fmap = FmapGroup("01", "f", Path("/rev.nii.gz"), {"TotalReadoutTime": 0.06}, [("foo", "1")])
    subj = Subject(
        "X",
        [
            Session("01", [_run("01", "foo", "1")], [fmap]),
            Session("02", [_run("02", "foo", "1")]),
        ],
    )
    opts = Options(ref_ses="01", fmap_ref=["f"], xrun_nonlin=True, xses_nonlin=True)
    s = write_script(build_plan(subj, opts), "wd", bids_root="/bids")

    # xrun: bash-loop names, so assert on the emitted template.
    assert '-prefix \\"${xstem}${LANE}_lin.nii$FMT\\"' in s
    assert '-prefix \\"${xstem}${LANE}_nl.nii$FMT\\"' in s
    assert '-warp_prefix \\"${xstem}_nl\\"' in s
    # The nonlinear step refines the linear image, not the raw source.
    assert '-source \\"${xstem}${LANE}_lin.nii$FMT\\"' in s

    # xses: fully expanded names.
    xs = "stage08.xses.ses-02"
    assert f'-prefix \\"{xs}.src-max_lin.nii$FMT\\"' in s
    assert f'-prefix \\"{xs}.src-max_nl.nii$FMT\\"' in s
    assert f'-warp_prefix \\"{xs}_nl\\"' in s

    # Transforms carry neither the lane nor `_lin`, and the chain uses those names.
    assert f'-1Dmatrix_save \\"{xs}.aff12.1D\\"' in s
    assert "src-max_nl_WARP" not in s
    assert f"{xs}_nl_WARP.nii$FMT" in s
    assert "_lin.aff12.1D" not in s and "_lin_WARP" not in s


def test_xrun_anchor_gets_a_role_ref_copy_only_without_fieldmaps():
    """No fmaps → the session's first run is the anchor and has no xrun output, so
    it gets a marker. With fmaps EVERY run gets a real xrun; no gap, no marker."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1"), _run("01", "foo", "2")])])
    s = write_script(build_plan(subj, Options()), "wd", bids_root="/bids")
    # The marker is the anchor's primary-lane image: with no SBRefs that is the
    # moco MAX (the lane that estimates the transforms), not the mean.
    assert '"stage06.xrun.ses-01.task-foo.run-1.src-max.role-ref_lin.nii$FMT"' in s
    assert '"stage02.moco.ses-01.task-foo.run-1_max.nii$FMT"' in s

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


def test_paired_fmap_supplies_its_own_forward_image():
    """A self-contained AP/PA pair corrects itself: blipflip gets BOTH inputs from
    fmap/, and that forward image (not a data run's rep) is the xrun base — the
    distorted space the field was estimated in."""
    fmap = FmapGroup(
        "SM",
        "run01",
        Path("/bids/fmap/dir-PA_epi.nii.gz"),
        {"TotalReadoutTime": 0.08},
        [("t", "1"), ("t", "2")],
        forward_path=Path("/bids/fmap/dir-AP_epi.nii.gz"),
    )
    runs = [_run("SM", "t", "1"), _run("SM", "t", "2")]
    plan = build_plan(Subject("X", [Session("SM", runs, [fmap])]), Options(go_to_anat=False))
    s = write_script(plan, "wd", bids_root="/bids")
    assert "-blip_up /bids/fmap/dir-AP_epi.nii.gz" in s
    assert "-blip_down /bids/fmap/dir-PA_epi.nii.gz" in s
    # Every run's xrun base is the fmap's forward image, not its own rep.
    bases = set(re.findall(r'XRUNBASE\[[^\]]+\]="([^"]*)"', s))
    assert bases == {"/bids/fmap/dir-AP_epi.nii.gz"}
    assert "/bids/fmap/dir-AP_epi.nii.gz" in s.split("stage04")[0]  # preflight-checked


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
    # stage07 runmean (once per lane), stage10 BOLD and stage10 SBRef all pass it.
    assert script.count("${JAC[$k]:+") == 2 + len({"sbref", "max", "min", "mean"})

    # A run with no fieldmap has no JAC entry at all (nothing to modulate).
    plain = build_plan(
        Subject("ME1", [Session("SM", [_run("SM", "t", "1")])]), Options(go_to_anat=False)
    )
    assert "JAC[" not in write_script(plain, "workdir", bids_root="/bids").split("stage07")[0]


def test_header_carries_the_generating_command():
    """The ffs_autoproc call that made the script is in it, commented out, and
    every continuation line stays commented (a bare wrapped line would run)."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")], anat=Path("/anat/T1w.nii.gz"))])
    plan = build_plan(subj, Options(go_to_anat=True))
    cmd = "ffs_autoproc -bids /some/very/long/path/to/a/bids/dataset/root -sub 01 " + (
        "-anat /an/equally/long/path/to/the/freesurfer/SUMA/brain.nii.gz -recipe 9p4T -format .zst"
    )
    s = write_script(plan, "wd", bids_root="/bids", invocation=cmd)
    note = s.split("set -euo pipefail")[0].splitlines()
    quoted = [ln for ln in note if "ffs_autoproc -bids" in ln or ln.startswith("#   -")]
    assert quoted, "generating command missing from the header"
    assert all(ln.startswith("#") for ln in note)
    # The words survive the wrapping, in order.
    assert " ".join(w for ln in quoted for w in ln.lstrip("# ").rstrip(" \\").split()) == cmd

    assert "generated by (uncomment" not in write_script(plan, "wd", bids_root="/bids")


def test_warpmaster_defines_grid_mask_and_stats():
    """stage10a builds the warpmaster grid + epi_mask before stage10, stage10
    resamples onto it, and the GLM emits OLS+REML buckets masked by epi_mask.
    The anat underlays (whole-brain box + the EPI-FOV crop of it) are emitted only
    when we own the anat."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")], anat=Path("/anat/T1w.nii.gz"))])
    plan = build_plan(subj, Options(go_to_anat=True, run_glm=True))
    s = write_script(plan, "wd", bids_root="/bids")

    # warpmaster grid + mask are built, and defined before the resample stage.
    assert s.index("stage10a: warpmaster") < s.index("stage10: final compose")
    assert "ffs_util_autobox" in s and "ffs_util_resample" in s
    assert '-prefix "stage10.warpmaster.nii$FMT"' in s
    assert '-prefix "epi_mask.nii$FMT"' in s and "-dilate 2" in s
    assert '[ -f "stage09.anat_autobox.nii.gz" ] ||' in s  # own anat → whole-brain underlay
    assert '[ -f "stage10.anat_in_epi_fov.nii.gz" ] ||' in s  # ... and the EPI-FOV crop
    # The alignment base is the boxed anat, and its output is named for the source.
    assert '-base "stage09.anat_autobox.nii.gz"' in s
    assert '-prefix "stage09.grandmean_al_anat.nii$FMT"' in s

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
    assert "stage09.anat_autobox.nii.gz" not in borrow
    assert "stage10.anat_in_epi_fov.nii.gz" not in borrow


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


def test_anat_source_defaults_to_grandmean_without_sbrefs():
    """-anat_source auto has only one thing to fall back on when the dataset has
    no SBRef lane."""
    plan = build_plan(_two_fmap_subject(), Options(go_to_anat=True, fmap_ref=["floc"]))
    s = write_script(plan, "wd", bids_root="/bids")
    assert '-source "stage08.grandmean.nii$FMT"' in s
    assert "stage05.fmapmean" not in s  # not built when nothing asks for it


def test_anat_source_ref_fmap_uses_the_reference_blip_mean():
    """-anat_source ref_fmap aligns the anat to the image that DEFINES the space
    (the reference group's undistorted forward image) instead of the grandmean."""
    plan = build_plan(
        _two_fmap_subject(), Options(go_to_anat=True, fmap_ref=["floc"], anat_source="ref_fmap")
    )
    s = write_script(plan, "wd", bids_root="/bids")
    assert '-source "stage04.blip.ses-SM.fmap-floc_unwarped.nii$FMT[0]"' in s
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
    # Averaged: ref group's own forward image + the non-ref group's ALIGNED one.
    assert '"stage04.blip.ses-SM.fmap-floc_unwarped.nii$FMT[0]"' in s
    assert '"stage05.xfmap.ses-SM.fmap-prim_lin.nii$FMT"' in s
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
    assert '-source "stage04.blip.ses-01.fmap-only_unwarped.nii$FMT[0]"' in s


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
    assert '-source "${MOCO_MEAN[$k]}"' not in s.split("stage07")[0]

    # xses is estimated on the SBRef session means...
    assert '-source \\"stage07.sesmean.ses-02.src-sbref.nii$FMT\\"' in s
    # ...and every lane reaches the grandmean through that ONE transform, applied
    # per run in stage08b (not by re-warping each lane's session mean).
    assert '-prefix \\"stage08.gmrun.ses-02.task-t.run-1.nii$FMT\\"' in s
    assert '-prefix \\"stage08.gmrun.ses-02.task-t.run-1.src-sbref.nii$FMT\\"' in s
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


def test_nonlin_in_source_swaps_the_warp_past_its_own_affine():
    """A warp estimated in the source's frame acts on the data BEFORE its affine.

    _CHAIN_ORDER is written leftmost-acts-first on *coordinates*, so the default
    "affine then warp" reads as (nl, lin); estimating in the source frame is the
    other order. Only the one pair moves -- the other stages' links stay put, and a
    stage left in the default mode keeps its default order in the same chain.
    """
    subj = Subject(
        "X",
        [
            Session("01", [_run("01", "foo", "1")]),
            Session("02", [_run("02", "foo", "1")]),
        ],
    )
    opts = dict(ref_ses="01", xses_nonlin=True, xrun_nonlin=True, anat_nonlin=True)
    key = ("02", "foo", "1")

    plain = _chain(build_plan(subj, Options(**opts)), key)
    assert plain.index("xses_nl") < plain.index("xses_lin")

    swapped = _chain(build_plan(subj, Options(**opts, xses_nonlin_in_source=True)), key)
    assert swapped.index("xses_lin") < swapped.index("xses_nl")
    # Only that pair moved: the anat link keeps its place relative to the stage,
    # and xrun -- left in the default mode -- keeps the default order.
    assert swapped.index("anat_nl") < swapped.index("xses_lin")
    assert set(swapped) == set(plain)
    if "wxrun_nl" in swapped:
        assert swapped.index("wxrun_nl") < swapped.index("wxrun_lin")


def test_nonlin_in_source_passes_the_matrix_and_the_unaligned_source():
    """The emitted formwarp call must take -matrix and the *un*-allineated image.

    If the script kept feeding the linear stage's output while the chain swapped,
    the warp would be estimated in one space and composed as though it lived in
    another -- silently, and only visibly as a bad alignment.
    """
    subj = Subject(
        "X",
        [
            Session("01", [_run("01", "foo", "1")]),
            Session("02", [_run("02", "foo", "1")]),
        ],
    )
    opts = dict(ref_ses="01", xses_nonlin=True)

    plain = write_script(build_plan(subj, Options(**opts)), "wd", bids_root="/bids")
    swapped = write_script(
        build_plan(subj, Options(**opts, xses_nonlin_in_source=True)), "wd", bids_root="/bids"
    )

    assert "-matrix" not in plain
    nl = [ln for ln in swapped.splitlines() if "_nl.nii" in ln and "-matrix" in ln]
    assert nl, "no formwarp line carried -matrix"
    for ln in nl:
        assert ".aff12.1D" in ln
        # the source must be what the LINEAR stage consumed, not what it produced.
        # (manifest lines are printf'd, so the inner quotes arrive backslash-escaped)
        src = re.search(r'-source "([^"]+)"', ln.replace('\\"', '"')).group(1)
        assert "_nl" not in src and "sesmean" in src, src


# ---------------------------------------------------------------------------
# every run reaches the common grid; the coverage lanes; the grandmean checkpoint
# ---------------------------------------------------------------------------


def _fmap(session, fid, intended, pe="j-", readout=0.05):
    return FmapGroup(
        session,
        fid,
        Path(f"/{session}_{fid}_rev.nii.gz"),
        {"TotalReadoutTime": readout, "PhaseEncodingDirection": pe},
        list(intended),
    )


def test_unclaimed_run_inherits_the_session_reference_fieldmap():
    """A run no IntendedFor claims used to anchor on the session's FIRST RUN, which
    left its runmean on a different grid from its siblings' (and never undistorted)
    — so the session mean was averaging incompatible images. It inherits the
    reference group instead, and comes out with the same chain as its siblings."""
    subj = Subject(
        "X",
        [
            Session(
                "01",
                [_run("01", "t", "1"), _run("01", "t", "2")],
                [_fmap("01", "PA", [("t", "1")])],
            )
        ],
    )
    plan = build_plan(subj, Options(go_to_anat=False))
    claimed, orphan = plan.runs
    assert orphan.fmap is not None and orphan.fmap_inherited
    assert orphan.warp_chain == claimed.warp_chain
    s = write_script(plan, "wd", bids_root="/bids")
    # Both runs land on the ONE grid the session averages in.
    grids = set(re.findall(r'REFGRID\[[^\]]+\]="([^"]*)"', s))
    assert grids == {"stage04.blip.ses-01.fmap-PA_unwarped.nii$FMT[0]"}
    assert "inherited fmap" in s  # and the guess is stated in the header


def test_unclaimed_run_with_the_wrong_pe_still_lands_on_the_common_grid():
    """Applying an AP field to a PA run doubles the distortion, so a PE-incompatible
    run gets NO fieldmap — but it must still reach the session's common grid, or it
    drops out of the session mean. It aligns straight to the undistorted reference."""
    runs = [_run("01", "t", "1"), _run("01", "t", "2", pe="j")]
    subj = Subject("X", [Session("01", runs, [_fmap("01", "PA", [("t", "1")], pe="j-")])])
    plan = build_plan(subj, Options(go_to_anat=False))
    flipped = plan.runs[1]
    assert flipped.fmap is None  # no field applied...
    assert "blip_half" not in flipped.warp_chain
    assert flipped.ref_fmap_id == "PA"  # ...but still on the session's grid
    s = write_script(plan, "wd", bids_root="/bids")
    assert 'XRUNBASE[01:t:2]="stage04.blip.ses-01.fmap-PA_unwarped.nii$FMT[0]"' in s
    assert 'REFGRID[01:t:2]="stage04.blip.ses-01.fmap-PA_unwarped.nii$FMT[0]"' in s


def test_unknown_ref_ses_is_an_error_not_a_silent_first_session():
    """The reference session changes every warp chain in the script; a typo that
    quietly anchored on session one would produce a plausible, wrong pipeline."""
    subj = Subject(
        "X", [Session("01", [_run("01", "t", "1")]), Session("02", [_run("02", "t", "1")])]
    )
    with pytest.raises(ValueError, match="no such session"):
        build_plan(subj, Options(ref_ses="99"))
    assert build_plan(subj, Options(ref_ses="ses-02")).ref_session == "02"


def test_fmap_anat_sources_need_the_fieldmap_in_the_REFERENCE_session():
    """ref_fmap/mean_fmap name an image in the anchor group's space. When the
    reference session has no fieldmap, the anchor falls back to another session,
    whose blip mean is NOT in grandmean space — the anat matrix would be estimated
    in the wrong frame. Degrade to the grandmean instead."""
    subj = Subject(
        "X",
        [
            Session("01", [_run("01", "t", "1")], anat=Path("/anat/T1w.nii.gz")),
            Session("02", [_run("02", "t", "1")], [_fmap("02", "PA", [("t", "1")])]),
        ],
    )
    plan = build_plan(subj, Options(go_to_anat=True, ref_ses="01", anat_source="ref_fmap"))
    assert effective_anat_source(plan) == "grandmean"
    # ...and it IS available when the reference session owns the fieldmap.
    subj.sessions[0].fmaps = [_fmap("01", "PA", [("t", "1")])]
    plan = build_plan(subj, Options(go_to_anat=True, ref_ses="01", anat_source="ref_fmap"))
    assert effective_anat_source(plan) == "ref_fmap"


def test_ref_image_picks_the_cross_session_representative_per_session():
    """-ref_image is one vocabulary at two levels: it chooses what represents each
    session to the cross-session alignment, and (by default) what the anat aligns."""
    subj = Subject(
        "X",
        [
            Session("01", [_run("01", "t", "1")], [_fmap("01", "PA", [("t", "1")])]),
            Session("02", [_run("02", "t", "1")], [_fmap("02", "PB", [("t", "1")])]),
        ],
    )
    s = write_script(
        build_plan(subj, Options(go_to_anat=False, ref_ses="01", ref_image="ref_fmap")),
        "wd",
        bids_root="/bids",
    )
    # base and source are each session's own reference fieldmap, undistorted.
    assert 'REFGM="stage04.blip.ses-01.fmap-PA_unwarped.nii$FMT[0]"' in s
    assert '-source \\"stage04.blip.ses-02.fmap-PB_unwarped.nii$FMT[0]\\"' in s
    # A session with no fieldmap degrades to its own mean, on its own.
    subj.sessions[1].fmaps = []
    s2 = write_script(
        build_plan(subj, Options(go_to_anat=False, ref_ses="01", ref_image="ref_fmap")),
        "wd",
        bids_root="/bids",
    )
    assert 'REFGM="stage04.blip.ses-01.fmap-PA_unwarped.nii$FMT[0]"' in s2
    assert '-source \\"stage07.sesmean.ses-02.src-max.nii$FMT\\"' in s2


def test_coverage_lanes_are_built_from_moco_and_composite_by_their_own_rule():
    """max-of-maxes keeps the edges motion cost a single run; min-of-mins marks
    where every frame of every run has data. Averaging either would be wrong."""
    subj = Subject("X", [Session("01", [_run("01", "t", "1"), _run("01", "t", "2")])])
    s = write_script(build_plan(subj, Options(go_to_anat=False)), "wd", bids_root="/bids")
    # one moco pass writes all three reductions
    for which in ("mean", "max", "min"):
        assert f'-save_{which} \\"${{mstem}}_{which}.nii$FMT\\"' in s
    # ...and each lane composites with its own reduction, not -mean for all.
    for lane, flag in (("src-max", "-max"), ("src-min", "-min")):
        block = [b for b in s.split("ffs_util_3dmath") if f"sesmean.ses-01.{lane}" in b]
        assert block and flag in block[0], lane
    # The min lane resamples linearly (its information is the zero boundary, which
    # wsinc5 rings across); every other lane keeps the sharp default.
    blocks = {
        lane: [
            b for b in s.split("ffs_nwarp") if f'-prefix "stage07.runmean.${{FRAG[$k]}}{lane}' in b
        ]
        for lane in (".src-min", ".src-max")
    }
    assert "-interp linear" in blocks[".src-min"][0]
    assert "-interp wsinc5" in blocks[".src-max"][0]


def test_grandmean_is_rebuilt_from_moco_space_in_one_interpolation():
    """A mean of session means is three interpolations deep for every non-reference
    session (moco → runmean → xses), and the grandmean is what the anat step and
    every -grand_reference align to. Compose the pre-chain with xses and resample
    each run ONCE instead."""
    subj = Subject(
        "X",
        [
            Session("01", [_run("01", "t", "1")]),
            Session("02", [_run("02", "t", "1")]),
        ],
    )
    s = write_script(
        build_plan(subj, Options(go_to_anat=False, ref_ses="01")), "wd", bids_root="/bids"
    )
    gm = [ln for ln in s.splitlines() if "stage08.gmrun.ses-02" in ln and "-source" in ln]
    assert gm, "no single-resample job for the non-reference session's run"
    line = gm[0].replace('\\"', '"')
    # straight from the post-moco image, through xses ∘ pre-chain, in one call
    assert '-source "stage02.moco.ses-02.task-t.run-1_max.nii$FMT"' in line
    assert "stage08.xses.ses-02.aff12.1D" in line
    # the reference session needs no extra pass — its runmean already IS the image
    assert "stage08.gmrun.ses-01" not in s
    # and the grandmean is composited from runs, not from session means
    gmean = [b for b in s.split("ffs_util_3dmath") if 'prefix "stage08.grandmean.nii' in b][0]
    assert "gmrun.ses-02" in gmean and "sesmean" not in gmean


def test_single_session_needs_no_grandmean_checkpoint():
    """One session: every run's runmean is already in grandmean space, so the
    checkpoint stage is not emitted at all."""
    subj = Subject("X", [Session("01", [_run("01", "t", "1"), _run("01", "t", "2")])])
    s = write_script(build_plan(subj, Options(go_to_anat=False)), "wd", bids_root="/bids")
    assert "stage08b" not in s and "gmrun" not in s
    assert '-prefix "stage08.grandmean.nii$FMT"' in s


# ---------------------------------------------------------------------------
# QC stacks
# ---------------------------------------------------------------------------


def _qc(script: str) -> dict[str, list[str]]:
    """The emitted qc_tcat calls, as {output file: [inputs]}."""
    out: dict[str, list[str]] = {}
    for ln in script.splitlines():
        if not ln.startswith("qc_tcat "):
            continue
        parts = re.findall(r'"([^"]*)"', ln)
        out[parts[0]] = parts[2:]  # parts[1] is the label string
    return out


def _qc_labels(script: str, name: str) -> list[str]:
    for ln in script.splitlines():
        parts = re.findall(r'"([^"]*)"', ln)
        if ln.startswith("qc_tcat ") and parts[0] == name:
            return parts[1].split()
    raise KeyError(name)


def test_qc_stacks_group_the_images_a_stage_claims_to_have_aligned():
    """Two sessions, two fieldmap groups in the first: each level's stack holds
    exactly the images that level put in ONE space, and no more."""
    fA = FmapGroup("01", "A", Path("/revA.nii.gz"), {"TotalReadoutTime": 0.06}, [("t", "1")])
    fB = FmapGroup("01", "B", Path("/revB.nii.gz"), {"TotalReadoutTime": 0.06}, [("t", "2")])
    fC = FmapGroup("02", "C", Path("/revC.nii.gz"), {"TotalReadoutTime": 0.06}, [("t", "3")])
    subj = Subject(
        "X",
        [
            Session("01", [_run("01", "t", "1"), _run("01", "t", "2")], [fA, fB]),
            Session("02", [_run("02", "t", "3")], [fC]),
        ],
    )
    s = write_script(build_plan(subj, Options(go_to_anat=False)), "wd", bids_root="/bids")
    qc = _qc(s)

    # cross-fmap: the reference group's own mean, then each aligned group.
    xf = qc["stage05.QC.xfmap.ses-01_lin.nii.gz"]
    assert xf == [
        "stage04.blip.ses-01.fmap-A_unwarped.nii$FMT[0]",
        "stage05.xfmap.ses-01.fmap-B_lin.nii$FMT",
    ]
    assert _qc_labels(s, "stage05.QC.xfmap.ses-01_lin.nii.gz") == ["ref:fmap-A", "fmap-B"]
    # ses-02 has one group — nothing to compare, no stack.
    assert "stage05.QC.xfmap.ses-02_lin.nii.gz" not in qc

    # cross-run is grouped by ALIGNMENT BASE, not by session: each fieldmap group's
    # runs land on its own forward image and are only comparable within the group.
    assert "stage06.QC.xrun.ses-01.fmap-A_lin.nii.gz" in qc
    assert "stage06.QC.xrun.ses-01.fmap-B_lin.nii.gz" in qc
    assert "stage06.QC.xrun.ses-01_lin.nii.gz" not in qc

    # stage07 is where a session's runs first share a grid — one stack per session.
    assert _qc(s)["stage07.QC.runmean.ses-01.nii.gz"] == [
        "stage07.runmean.ses-01.task-t.run-1.nii$FMT",
        "stage07.runmean.ses-01.task-t.run-2.nii$FMT",
    ]

    # cross-session, reference session first; then every run of every session.
    assert _qc_labels(s, "stage08.QC.xses.src-max_lin.nii.gz") == ["ref:ses-01", "ses-02"]
    assert _qc_labels(s, "stage08.QC.grandmean.nii.gz") == [
        "ses-01.task-t.run-1",
        "ses-01.task-t.run-2",
        "ses-02.task-t.run-3",
    ]


def test_qc_final_stack_is_every_runs_mean_in_output_space():
    """-save_mean is on by default at the final resample, and those means are the
    dataset-wide stack: if anything moves here, no earlier stage fixed it."""
    subj = Subject("X", [Session("01", [_run("01", "t", "1"), _run("01", "t", "2")])])
    s = write_script(build_plan(subj, Options(go_to_anat=False)), "wd", bids_root="/bids")
    assert "-save_mean -prefix" in s
    assert _qc(s)["stage10.QC.final.nii.gz"] == [
        "stage10.warpmaster.nii$FMT",
        "mean_stage10.final.ses-01.task-t.run-1.nii$FINAL_FMT",
        "mean_stage10.final.ses-01.task-t.run-2.nii$FINAL_FMT",
    ]


def test_qc_skips_single_image_groups_and_honours_no_qc():
    """A one-image "stack" answers no alignment question, and -no_qc drops the
    machinery entirely."""
    subj = Subject("X", [Session("01", [_run("01", "t", "1")])])
    s = write_script(build_plan(subj, Options(go_to_anat=False)), "wd", bids_root="/bids")
    # one run: nothing to compare at the run/session levels...
    assert not [k for k in _qc(s) if k.startswith(("stage06", "stage07", "stage08"))]
    # ...but the final stack still pairs that run against the output grid.
    assert "stage10.QC.final.nii.gz" in _qc(s)

    off = write_script(build_plan(subj, Options(go_to_anat=False, qc=False)), "wd")
    assert "qc_tcat" not in off
    # The per-run mean survives -no_qc: it is a useful output in its own right,
    # not scaffolding for the stack.
    assert "-save_mean -prefix" in off


def test_qc_single_session_grandmean_stack_is_not_a_duplicate():
    """With one session, grandmean space IS the session's common grid, so the
    stage08 stack would repeat stage07's file for file."""
    subj = Subject("X", [Session("01", [_run("01", "t", "1"), _run("01", "t", "2")])])
    s = write_script(build_plan(subj, Options(go_to_anat=False)), "wd", bids_root="/bids")
    assert "stage07.QC.runmean.ses-01.nii.gz" in _qc(s)
    assert "stage08.QC.grandmean.nii.gz" not in _qc(s)


# ---------------------------------------------------------------------------
# GRE (B0) fieldmaps
# ---------------------------------------------------------------------------


def _stage04(script: str) -> str:
    """Just the stage04 block — the preflight lists every tool by name, so a bare
    substring search cannot tell which fieldmap tool the pipeline actually calls."""
    start = script.index("# ============================ stage04")
    return script[start : script.index("# ============================ stage0", start + 10)]


def _b0_run(session, task, run, pe="j-", readout=0.06):
    """A run whose sidecar carries the EPI geometry a GRE fieldmap does not."""
    r = _run(session, task, run, pe=pe)
    r.json["TotalReadoutTime"] = readout
    return r


def _b0_fmap(fmap_id="b0", intended=(("foo", "1"),), **kw):
    return FmapGroup(
        "01",
        fmap_id,
        Path("/fmap/sub-X_phasediff.nii.gz"),
        {"EchoTime1": 0.00492, "EchoTime2": 0.00738},
        list(intended),
        kind="b0",
        phasediff_path=Path("/fmap/sub-X_phasediff.nii.gz"),
        magnitude_paths=[Path("/fmap/sub-X_magnitude1.nii.gz")],
        te_ms=[4.92, 7.38],
        **kw,
    )


def test_b0_fmap_emits_b0fmap_and_its_own_undistorted_mean():
    """A GRE group must reach stage05+ with the same two products a blipflip group
    has: the PE warp and the undistorted mean that defines the common grid."""
    fmap = _b0_fmap(intended=[("foo", "1"), ("foo", "2")])
    subj = Subject(
        "X", [Session("01", [_b0_run("01", "foo", "1"), _b0_run("01", "foo", "2")], [fmap])]
    )
    plan = build_plan(subj, Options())
    s = write_script(plan, "wd", bids_root="/bids")
    st4 = _stage04(s)
    assert "ffs_util_b0fmap" in st4 and "ffs_blipflip" not in st4
    assert "-phasediff /fmap/sub-X_phasediff.nii.gz" in s
    assert "-te 4.92 7.38" in s
    # EPI geometry comes from the RUN: a GRE sidecar has neither of these.
    assert "-pe_dir j-" in s and "-readout 0.06" in s
    # The mean the rest of the pipeline consumes, made with the Jacobian so it
    # agrees with the runmeans about intensity where distortion was worst.
    assert 'stage04.blip.ses-01.fmap-b0_mean.nii$FMT"' in s
    assert "-jac j" in s
    # A GRE group has no pair to average, so that mean IS its forward image: the
    # pepolar '_unwarped[0]' target must not be used as a transform target here
    # (the b0 branch's own '_unwarped' is a scratch file it deletes).
    assert "fmap-b0_unwarped.nii$FMT[0]" not in s
    assert 'REFGRID[01:foo:2]="stage04.blip.ses-01.fmap-b0_mean.nii$FMT"' in s
    for run in ("1", "2"):
        assert "blip_half" in _chain(plan, ("01", "foo", run))


def test_b0_fmap_without_a_readout_aborts_rather_than_guessing():
    """A measured field has an absolute Hz scale, so the readout cannot be inferred
    the way blipflip can manage without one."""
    fmap = _b0_fmap()
    run = _run("01", "foo", "1")
    run.json.pop("TotalReadoutTime", None)
    subj = Subject("X", [Session("01", [run], [fmap])])
    s = write_script(build_plan(subj, Options()), "wd", bids_root="/bids")
    st4 = _stage04(s)
    assert "no TotalReadoutTime" in st4 and "ffs_util_b0fmap" not in st4


def test_b0_fmap_splits_per_pe_polarity():
    """One measured field, runs of both polarities: the field is polarity-free but
    the warp is not, so each polarity gets its own group, warp and undistorted
    space (which cross-fmap alignment then reconciles)."""
    fmap = _b0_fmap(intended=[("foo", "1"), ("foo", "2")])
    subj = Subject(
        "X",
        [
            Session(
                "01",
                [_b0_run("01", "foo", "1", pe="j-"), _b0_run("01", "foo", "2", pe="j")],
                [fmap],
            )
        ],
    )
    plan = build_plan(subj, Options())
    ids = {pr.bold.run: pr.fmap.fmap_id for pr in plan.runs}
    assert ids["1"] != ids["2"]
    s = write_script(plan, "wd", bids_root="/bids")
    flags = _stage04(s).replace(" \\", "")
    assert "-pe_dir j-\n" in flags and "-pe_dir j\n" in flags
    # Two groups → the non-reference one is brought onto the reference's grid.
    assert "xfmap_lin" in _chain(plan, ("01", "foo", "2"))


def test_pepolar_fmap_is_untouched_by_the_polarity_split():
    fmap = FmapGroup("01", "f", Path("/rev.nii.gz"), {"TotalReadoutTime": 0.06}, [("foo", "1")])
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")], [fmap])])
    plan = build_plan(subj, Options())
    assert plan.runs[0].fmap is not None and plan.runs[0].fmap.fmap_id == "f"
    assert "ffs_blipflip" in _stage04(write_script(plan, "wd", bids_root="/bids"))


def test_task_distinguished_runs_still_get_xrun_alignment():
    """Runs identified by TASK alone (task-bar1/bar2/..., no run- entity) all have
    run=None. Anchoring on the `run` value made every one of them 'the first run',
    so nothing was ever aligned to anything: no wxrun link in any chain."""
    subj = Subject(
        "X",
        [Session("01", [_run("01", "bar1", None), _run("01", "bar2", None)])],
    )
    plan = build_plan(subj, Options(go_to_anat=False))
    first, second = plan.runs
    assert first.is_ref_run and not second.is_ref_run
    assert first.warp_chain == ["moco"]
    assert second.warp_chain == ["wxrun_lin", "moco"]


# ---------------------------------------------------------------------------
# recipes: each named preset must actually put its stages in the script
# ---------------------------------------------------------------------------

# Marker → the stage that emits it, for a subject that HAS fieldmaps, an anat,
# and several runs, i.e. a dataset where every recipe's stages are all possible.
_STAGE_MARKERS = {
    # Section headers / chain tokens, never the preflight `command -v` list —
    # that names every tool the generator knows about, stage or not.
    "tshift": 'st_str="-tpattern',
    "blip": "stage04: fieldmap",
    "xrun": "stage06: cross-run alignment",
    "xrun_nl": "_nl_WARP",
    "anat": "stage09: anatomical alignment",
    "anat_nl": "nlanat_invwarp",
    "locomoco": "stage03: locomoco",
    "glm": "stage12: GLM",
}

# What each recipe claims (config.RECIPE_SUMMARY), as the stages it must emit.
_RECIPE_STAGES = {
    # xrun is in every recipe, bare_bones included: without it the runs are not
    # in a common space and a multi-run GLM is meaningless.
    "bare_bones": {"xrun", "glm"},
    "simple": {"tshift", "blip", "xrun", "anat", "glm"},
    "simple_nonlin": {"tshift", "blip", "xrun", "xrun_nl", "anat", "glm"},
    "complete": {"tshift", "blip", "xrun", "xrun_nl", "anat", "anat_nl", "glm"},
    "extreme": {"tshift", "blip", "xrun", "xrun_nl", "anat", "anat_nl", "locomoco", "glm"},
}


def _full_subject():
    """One session, two runs distinguished by task (the retinotopy shape), an
    AP/PA fieldmap pair, slice timing present."""
    runs = []
    for task in ("bar1", "bar2"):
        r = _run("01", task, None, pe="j")
        r.json["SliceTiming"] = [0.0, 0.5]
        runs.append(r)
    fmap = FmapGroup(
        "01",
        "PA-AP",
        Path("/fmap_AP.nii.gz"),
        {"TotalReadoutTime": 0.05, "PhaseEncodingDirection": "j"},
        [("bar1", None), ("bar2", None)],
        forward_path=Path("/fmap_PA.nii.gz"),
    )
    return Subject("X", [Session("01", runs, [fmap])])


@pytest.mark.parametrize("recipe", sorted(_RECIPE_STAGES))
def test_recipe_emits_exactly_the_stages_it_advertises(recipe):
    from fastfuncstuff.autoproc import config

    opt = Options(**config.RECIPES[recipe], recipe=recipe)
    opt.anat_path = "/anat.nii.gz"
    opt.tpm = "/tpm.nii.gz"
    script = write_script(build_plan(_full_subject(), opt), "wd", bids_root="/bids")
    want = _RECIPE_STAGES[recipe]
    for stage, marker in _STAGE_MARKERS.items():
        present = marker in script
        assert present == (stage in want), (
            f"{recipe}: stage {stage} {'missing from' if stage in want else 'unexpectedly in'} "
            "the emitted script"
        )


def test_anat_source_auto_prefers_the_sbref_grandmean():
    """SBRefs already estimate xrun and xses (emit._primary_lane). The anat step is
    the pipeline's one cross-modal lpc fit, so defaulting it to the BOLD-mean
    grandmean threw away the sharpest, least-interpolated image available — for
    the alignment that needs it most. `auto` follows the lane, and the segment
    input follows with it."""
    subj = Subject(
        "X",
        [
            Session(
                "01",
                [_sbrun("01", "t", "1"), _sbrun("01", "t", "2")],
                anat=Path("/anat/T1w.nii.gz"),
            )
        ],
    )
    plan = build_plan(subj, Options(go_to_anat=True, anat_nonlin=True, tpm="/tpm.nii.gz"))
    assert plan.use_sbref
    assert effective_anat_source(plan) == "sbmean"
    s = write_script(plan, "wd", bids_root="/bids")
    sb = "stage08.grandmean.src-sbref.nii$FMT"
    assert f'-source "{sb}"' in s  # linear anat (ffs_allineate)
    assert f'ffs_segment \\\n    -input "{sb}"' in s  # nonlinear anat
    # "auto" resolving to something is not a degraded request — no fallback note.
    assert "unavailable here →" not in s
    # ...and the explicit opt-out still works.
    s_gm = write_script(
        build_plan(subj, Options(go_to_anat=True, anat_source="grandmean")), "wd", bids_root="/bids"
    )
    assert '-source "stage08.grandmean.nii$FMT"' in s_gm


def _timed_run(task, run, at, pe="i"):
    r = _run("01", task, run, pe=pe)
    r.json["AcquisitionTime"] = at
    return r


def test_borrowed_forward_is_the_run_acquired_next_to_the_fieldmap():
    """A fmap folder with no matched-PE mate borrows a run's SBRef as the blip-up
    image. Anything the head did between the two is written straight into the
    field, so the pairing has to be by TIME, not by scan order (which is
    task-then-run and, in an interleaved session, tens of minutes off)."""
    runs = [
        _timed_run("expres", "1", "14:02:21"),  # first in task order — far away
        _timed_run("fncloc", "1", "13:48:59"),  # 2.5 min after the fieldmap
        _timed_run("fncloc", "2", "13:44:00"),  # closer still, but BEFORE it
    ]
    fmap = _fmap("01", "LR", [(r.task, r.run) for r in runs], pe="i-")
    fmap.json["AcquisitionTime"] = "13:46:29"
    plan = build_plan(Subject("X", [Session("01", runs, [fmap])]), Options())
    assert all(pr.fmap_forward.endswith("task-fncloc_run-1_bold.nii.gz") for pr in plan.runs)

    # ...and stage04 pairs blipflip with THAT image, not with the group's first
    # run in scan order (the emitter used to re-derive it and disagree).
    st4 = _stage04(write_script(plan, "wd", bids_root="/bids"))
    assert "-blip_up /bids/sub-X/ses-01/func/sub-X_ses-01_task-fncloc_run-1_bold.nii.gz" in st4

    # No AcquisitionTime anywhere: fall back to the first intended run.
    for r in runs:
        del r.json["AcquisitionTime"]
    del fmap.json["AcquisitionTime"]
    plan = build_plan(Subject("X", [Session("01", runs, [fmap])]), Options())
    assert all(pr.fmap_forward.endswith("task-expres_run-1_bold.nii.gz") for pr in plan.runs)
