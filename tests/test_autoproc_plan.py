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
    assert '-prefix \\"stage07.sesmean.ses-01.nii$FMT\\"' in s
    assert '-prefix \\"stage07.sesmean.ses-02.nii$FMT\\"' in s
    assert '-prefix \\"stage08.grandmean.nii$FMT\\"' in s
    # The old names are gone entirely.
    assert "premean" not in s and "stage07.grandmean" not in s
    # Session means are built before the alignment that makes the grandmean valid.
    assert s.index("stage07.sesmean.ses-02") < s.index("stage08: cross-session")
    assert s.index("stage08: cross-session") < s.index('-prefix \\"stage08.grandmean.nii$FMT\\"')
    # Exactly one command writes the grandmean.
    assert s.count('-prefix \\"stage08.grandmean.nii$FMT\\"') == 1


def test_single_session_grandmean_has_the_same_single_producer():
    """One session: stage08 still owns the grandmean (built from the lone sesmean),
    rather than a stage07 `cp` — same name, same stage, either way."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")])])
    s = write_script(build_plan(subj, Options()), "wd", bids_root="/bids")
    assert '-prefix \\"stage08.grandmean.nii$FMT\\"' in s
    assert s.count('-prefix \\"stage08.grandmean.nii$FMT\\"') == 1
    assert '\\"stage07.sesmean.ses-01.nii$FMT\\"' in s
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
    assert '-prefix \\"${mpfx}.nii$FMT\\"' in s
    assert 'lm_in="\\"stage02.moco.${FRAG[$k]}${E1[$k]:-}.nii$FMT\\""' in s
    assert "-input $lm_in" in s
    # and the mean that goes downstream is locomoco's, not moco's
    assert "stage03.nlmoco.ses-01.task-foo.run-1_locomoco_mean.nii$FMT" in s

    s2 = write_script(build_plan(subj, Options(locomoco=False)), "wd", bids_root="/bids")
    assert '-prefix \\"${mpfx}.nii$FMT\\"' not in s2


def test_locomoco_solves_the_partition_axis_too_for_3d():
    """3-D EPI: locomoco gets the partition axis (-pe_dir2) beside the in-plane PE
    one, so both directions are solved in one call; 2-D multi-slice does not."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")])])
    s3 = write_script(
        build_plan(subj, Options(locomoco=True, acq_3d=True)), "wd", bids_root="/bids"
    )
    s2 = write_script(
        build_plan(subj, Options(locomoco=True, acq_3d=False)), "wd", bids_root="/bids"
    )
    assert "-pe_dir2 k" in s3
    assert "-pe_dir2" not in s2


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
    for m in re.finditer(r'(?:CHAIN|AVGCHAIN|PRECHAIN)\[[^\]]+\]="([^"]*)"', script):
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
    note = s.split("# ---- Bash version")[0].splitlines()
    quoted = [ln for ln in note if "ffs_autoproc -bids" in ln or ln.startswith("#   -")]
    assert quoted, "generating command missing from the header"
    assert all(ln.startswith("#") for ln in note)
    # The words survive the wrapping, in order.
    assert " ".join(w for ln in quoted for w in ln.lstrip("# ").rstrip(" \\").split()) == cmd

    assert "generated by (uncomment" not in write_script(plan, "wd", bids_root="/bids")


def test_script_rejects_unsupported_shell_before_creating_outputs(tmp_path):
    plan = build_plan(
        Subject("X", [Session("01", [_run("01", "foo", "1")])]), Options(go_to_anat=False)
    )
    out_dir = tmp_path / "outputs"
    script = write_script(plan, str(out_dir))
    out = subprocess.run(["sh", "-c", script], capture_output=True, text=True)
    assert out.returncode == 1
    assert "require Bash >= 4" in out.stderr
    assert "brew install bash" in out.stderr
    assert "invalid option" not in out.stderr
    assert not out_dir.exists()


def _modern_bash() -> str | None:
    """A Bash >= 4 from the places the emitted preamble looks, or None."""
    for candidate in (shutil.which("bash"), "/opt/homebrew/bin/bash", "/usr/local/bin/bash"):
        if candidate and Path(candidate).exists():
            probe = subprocess.run(
                [candidate, "-c", '[ "${BASH_VERSINFO[0]}" -ge 4 ]'], capture_output=True
            )
            if probe.returncode == 0:
                return candidate
    return None


def test_script_run_by_an_older_shell_continues_under_a_modern_bash(tmp_path):
    """Stock macOS bash and plain sh both land here; neither should stop the pipeline."""
    if _modern_bash() is None:
        pytest.skip("no Bash >= 4 on this machine")
    plan = build_plan(
        Subject("X", [Session("01", [_run("01", "foo", "1")])]), Options(go_to_anat=False)
    )
    preamble = write_script(plan, str(tmp_path / "outputs")).split("\nOUT=", 1)[0]
    path = tmp_path / "s.sh"
    path.write_text(preamble + '\necho "major=${BASH_VERSINFO[0]} args=$*"\n')
    # sh: bash in POSIX mode on a Mac, dash on Debian. Either way not a Bash >= 4 script run.
    out = subprocess.run(["sh", str(path), "one", "two"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    major, args = out.stdout.strip().removeprefix("major=").split(" args=")
    assert int(major) >= 4
    assert args == "one two"


def test_script_preamble_checks_the_running_bash_version(tmp_path):
    plan = build_plan(
        Subject("X", [Session("01", [_run("01", "foo", "1")])]), Options(go_to_anat=False)
    )
    script = write_script(plan, str(tmp_path / "outputs"))
    preamble = script.split("\nOUT=", 1)[0]
    version = subprocess.run(
        ["bash", "-c", 'echo "${BASH_VERSINFO[0]}"'], capture_output=True, text=True, check=True
    )
    out = subprocess.run(["bash", "-c", preamble], capture_output=True, text=True)
    if int(version.stdout) < 4:
        assert out.returncode == 1
        assert "require Bash >= 4" in out.stderr
    else:
        assert out.returncode == 0
        assert out.stderr == ""


def test_warpmaster_defines_grid_mask_and_stats():
    """stage10a builds the warpmaster grid before stage10, stage10 resamples onto
    it, stage10b masks the data that came out, and the GLM emits OLS+REML buckets.
    The anat underlays (whole-brain box + the EPI-FOV crop of it) are emitted only
    when we own the anat."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")], anat=Path("/anat/T1w.nii.gz"))])
    plan = build_plan(subj, Options(go_to_anat=True, run_glm=True))
    s = write_script(plan, "wd", bids_root="/bids")

    # warpmaster grid is built, and defined before the resample stage.
    assert s.index("stage10a: warpmaster") < s.index("stage10: final compose")
    assert "ffs_util_autobox" in s and "ffs_util_resample" in s
    assert '-prefix "stage10.warpmaster.nii$FMT"' in s
    assert '[ -f "stage09.anat_autobox.nii.gz" ] ||' in s  # own anat → whole-brain underlay
    assert '[ -f "stage10.anat_in_epi_fov.nii.gz" ] ||' in s  # ... and the EPI-FOV crop
    # The alignment base is the boxed anat, and its output is named for the source.
    assert '-base "stage09.anat_autobox.nii.gz"' in s
    assert '-prefix "stage09.grandmean_al_anat.nii$FMT"' in s

    # stage10 lands runs on the warpmaster (not the raw anat master), correcting
    # the cardinal-frame slip against the grid the warpmaster was cut from: without
    # it an oblique slab's final data sat 1.2-1.7 mm off their header (ds003427).
    assert "-master stage10.warpmaster.nii$FMT" in s
    assert '-chain_base \\"$MASTER\\"' in s

    # GLM: OLS by default + REML, masked, scaled.
    assert '-Obuck "stage12.stats-ols.task-foo.nii$GLM_FMT"' in s
    assert '-Rbuck "stage12.stats-reml.task-foo.nii$GLM_FMT"' in s
    assert "-do_scale" in s
    # -glm_mask defaults to none: no mask flag at all, and nothing names a mask
    # the script does not write.
    assert "-mask " not in s

    # No-own-anat borrow mode: still a warpmaster, but no viewing brain.
    borrow = write_script(
        build_plan(subj, Options(grand_reference="/floc.results", run_glm=True)),
        "wd",
        bids_root="/bids",
    )
    assert '-prefix "stage10.warpmaster.nii$FMT"' in borrow
    assert "stage09.anat_autobox.nii.gz" not in borrow
    assert "stage10.anat_in_epi_fov.nii.gz" not in borrow


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        (["stage09.anat.nii.zst"], "stage09.anat.nii.zst"),  # pre-rename reference
        (
            ["stage09.anat.nii.zst", "stage09.grandmean_al_anat.nii.zst"],
            "stage09.grandmean_al_anat.nii.zst",
        ),
        (
            ["stage09.grandmean_al_anat.nii.zst", "stage09.grandmean_al_anat_nl.nii.zst"],
            "stage09.grandmean_al_anat_nl.nii.zst",
        ),
    ],
)
def test_grand_reference_master_resolves_old_and_new_names(tmp_path, present, expected):
    """The borrowed master is found whichever name the reference wrote its anchor
    under. A reference processed before the _al_anat rename only has stage09.anat,
    and stage10a used to stop there with "no master dataset"."""
    ref = tmp_path / "floc.results"
    ref.mkdir()
    for name in present:
        (ref / name).touch()
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")])])
    s = write_script(
        build_plan(subj, Options(grand_reference=str(ref), run_glm=True)), "wd", bids_root="/bids"
    )
    line = next(ln for ln in s.splitlines() if ln.startswith('MASTER="'))
    out = subprocess.run(
        ["bash", "-c", f'set -euo pipefail; {line}; echo "$MASTER"'],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    ).stdout.strip()
    assert out == f"{ref}/{expected}"


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
    assert '-prefix \\"stage08.grandmean.nii$FMT\\"' in s


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
@pytest.mark.parametrize("drift", [True, False])
def test_phase_proc_unwraps_then_rides_the_final_resample(tmp_path, nordic, stc, drift):
    """-phase_proc: the unwrap happens once up front (after NORDIC when NORDIC
    runs), per-slice field drift comes off in native space right after it, the
    phase is untouched in between, and the SAME warp chain carries it at stage10
    into a separately-labelled part-phase output."""
    subj = Subject("X", [Session("SM", [_phase_run("SM", "primary", "1")])])
    opt = Options(
        phase_proc=True,
        phase_drift=drift,
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

    preflight = next(line for line in s.splitlines() if line.startswith("for t in ffs_"))
    assert " ffs_phase_unwrap" in preflight
    # phase files follow FMT: every reader is an ffs tool now (ROMEO could not read .zst)
    assert "PHASE_FMT" not in s and "romeo" not in s
    assert 'ffs_phase_unwrap -phase "$ph" -magnitude "$mag" -prefix "$uw" -device "$DEVICE"' in s

    unwrapped = "stage00.unwrap.${FRAG[$k]}.part-phase.nii$FMT"
    if drift:
        # Native space, after the unwrap, before anything mixes the acquired slices.
        assert " ffs_phasedrift" in preflight
        assert 'ffs_phasedrift -input "$uw" -magnitude "$mag" -prefix "$pd"' in s
        assert (
            s.index("ffs_phase_unwrap -phase")
            < s.index("ffs_phasedrift -input")
            < s.index("stage02: motion")
        )
        unwrapped = "stage00.unwrapdrift.${FRAG[$k]}.part-phase.nii$FMT"
    else:
        assert "ffs_phasedrift" not in s
    if nordic:
        # NORDIC already wrote the denoised pair AND trimmed the noise volumes.
        assert 'ph="stage00.nordic.${FRAG[$k]}.part-phase.nii$FMT"' in s
        # resume: NORDIC is done only when its phase output exists too (a dir from
        # before the .part-phase name has _phase, and the unwrap would not find it)
        assert '[ -f "$outf" ] && { [ -z "$ph" ] || [ -f "$phout" ]; } && continue' in s
        assert s.index("stage00: NORDIC") < s.index("stage00: phase unwrap")
    else:
        # No NORDIC: noise volumes come off BOTH before unwrapping, as a selector
        # on the raw pair (ffs_phase_unwrap reads selectors, so nothing is copied).
        assert 'mag="${mag}[0..$last]"; ph="${ph}[0..$last]"' in s

    # Slice timing done up front must be applied to the phase too, and that
    # tshifted phase is what reaches stage10.
    if stc == "first":
        assert f'-input "{unwrapped}"' in s
        phase_in = "stage01.tshift.${FRAG[$k]}.part-phase.nii$FMT"
    else:
        phase_in = unwrapped
    assert f'-phase \\"{phase_in}\\" -phase_units rad -phase_warp direct' in s
    assert '-phase_prefix \\"$phoutf\\"' in s
    assert 'phoutf="stage10.final.${FRAG[$k]}.part-phase.nii$FINAL_FMT"' in s

    # Nothing between stage00 and stage10 touches the phase: no other stage
    # references a part-phase file.
    body = s[s.index("stage02: motion correction") : s.index("stage10: compose + resample")]
    assert "part-phase" not in body


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("stc", ["integrate", "first"])
def test_phase_proc_skips_runs_without_phase(tmp_path, stc):
    """A session mixing gradient-echo runs (phase) and spin-echo runs (none):
    every phase step is gated per run, so the SE run gets no unwrap, no phase
    tshift, and no -phase arguments at the final resample."""
    se = _run("SM", "primary", "2")
    subj = Subject("X", [Session("SM", [_phase_run("SM", "primary", "1"), se])])
    opt = Options(phase_proc=True, slicetiming_method=stc, go_to_anat=False, distortion=False)
    s = write_script(build_plan(subj, opt), "wd", bids_root="/bids")
    assert (
        subprocess.run(  # noqa: S603
            ["bash", "-n", str(_write(tmp_path / "p.sh", s))], capture_output=True
        ).returncode
        == 0
    )
    assert '[ -n "${PHASE[$k]:-}" ] || continue' in s
    assert '[ -n "${PHASE[$k]:-}" ] && pa=" -phase' in s
    assert '-prefix \\"$outf\\"$pa"' in s
    if stc == "first":
        assert '[ -z "${PHASE[$k]:-}" ] || [ -f "$phout" ]' in s

    # Run the stage10 argument builder for both runs: only the GE line has phase.
    for k, ph in (("ge", "x_phase.nii.gz"), ("se", "")):
        frag = f'declare -A PHASE FRAG; PHASE[{k}]="{ph}"; FRAG[{k}]={k}; k={k}; FINAL_FMT=.gz; FMT=.gz\n'
        lines = [ln for ln in s.splitlines() if "phoutf=" in ln or 'pa=" -phase' in ln]
        out = subprocess.run(  # noqa: S603
            ["bash", "-c", frag + "\n".join(lines) + '\necho "[$pa]"'],
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert ("-phase_prefix" in out) == bool(ph), out


def _write(path, text):
    path.write_text(text)
    return path


def test_phase_proc_off_emits_no_phase_machinery():
    subj = Subject("X", [Session("SM", [_phase_run("SM", "primary", "1")])])
    s = write_script(build_plan(subj, Options(go_to_anat=False, distortion=False)), "wd")
    # The PHASE[] data-array entry is always emitted (NORDIC may want it); the
    # unwrap stage and the phase-side outputs must not be.
    assert "ffs_phase_unwrap" not in s
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
    from fastfuncstuff.autoproc.plan import run_average_chain

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
    assert "moco" not in run_average_chain(pr) and "locomoco" not in run_average_chain(pr)
    assert run_average_chain(pr) == [t for t in pr.warp_chain if t not in ("moco", "locomoco")]


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
    assert '-prefix \\"stage08.grandmean.src-sbref.nii$FMT\\"' in s
    assert '-prefix \\"stage08.grandmean.nii$FMT\\"' in s

    # Both lanes' session means are built from their own runmeans.
    assert "stage07.runmean.ses-01.task-t.run-2.src-sbref.nii$FMT" in s
    assert "stage07.runmean.ses-01.task-t.run-2.nii$FMT" in s


def test_sbrefs_reach_the_final_space():
    """Every run's SBRef lands on the warpmaster grid alongside its timeseries —
    the sharpest per-run alignment QC the pipeline can produce."""
    subj = Subject("X", [Session("01", [_sbrun("01", "t", "1"), _sbrun("01", "t", "2")])])
    s = write_script(build_plan(subj, Options()), "wd", bids_root="/bids")
    assert 'sboutf="stage10.final.${FRAG[$k]}.src-sbref.nii$FINAL_FMT"' in s
    assert '-nwarp \\"${AVGCHAIN[$k]}\\"' in s
    # SBRefs are load-bearing once the lane is on, so preflight checks them.
    assert "/bids/sb_01_t_1.nii.gz" in s.split("stage02")[0]

    # A run whose SBRef needs no transform at all (no anat, no fmap, anchor run)
    # cannot go through ffs_nwarp — there is no identity warp. It regrids instead.
    bare = write_script(
        build_plan(subj, Options(go_to_anat=False, distortion=False)), "wd", bids_root="/bids"
    )
    assert 'if [ -z "${AVGCHAIN[$k]:-}" ]; then' in bare
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
        block = [ln for ln in s.splitlines() if f"sesmean.ses-01.{lane}" in ln and "-prefix" in ln]
        assert block and flag in block[0], lane
    # The min lane resamples linearly (its information is the zero boundary, which
    # wsinc5 rings across); every other lane keeps the sharp default.
    blocks = {
        lane: [
            ln for ln in s.splitlines() if f'-prefix \\"stage07.runmean.${{FRAG[$k]}}{lane}' in ln
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
    gmean = [ln for ln in s.splitlines() if 'prefix \\"stage08.grandmean.nii' in ln][0]
    assert "gmrun.ses-02" in gmean and "sesmean" not in gmean


def test_single_session_needs_no_grandmean_checkpoint():
    """One session: every run's runmean is already in grandmean space, so the
    checkpoint stage is not emitted at all."""
    subj = Subject("X", [Session("01", [_run("01", "t", "1"), _run("01", "t", "2")])])
    s = write_script(build_plan(subj, Options(go_to_anat=False)), "wd", bids_root="/bids")
    assert "stage08b" not in s and "gmrun" not in s
    assert '-prefix \\"stage08.grandmean.nii$FMT\\"' in s


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
    # The mean is named for its run, not prefixed with "mean_": stage first, as
    # every other file in the directory is.
    assert '-save_mean \\"$meanf\\"' in s
    assert _qc(s)["stage10.QC.final.nii.gz"] == [
        "stage10.warpmaster.nii$FMT",
        "stage10.final.ses-01.task-t.run-1_mean.nii$FINAL_FMT",
        "stage10.final.ses-01.task-t.run-2_mean.nii$FINAL_FMT",
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
    assert '-save_mean \\"$meanf\\"' in off


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
    # No anat: opposite polarities are two acquisition families, which with an anat
    # would run in separate folders (test_autoproc_family); here they stay one pipeline.
    plan = build_plan(subj, Options(go_to_anat=False))
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


# ---------------------------------------------------------------------------
# Which engine runs a nonlinear stage
# ---------------------------------------------------------------------------


def _two_session_subject() -> Subject:
    return Subject(
        "X",
        [Session("01", [_run("01", "foo", "1")]), Session("02", [_run("02", "foo", "1")])],
    )


def test_nonlinear_backend_is_per_stage():
    """One script can run different engines at different stages.

    The stages are different problems -- cross-run is a small residual between two
    images of one head minutes apart, cross-session can be a different day and a
    different shim -- so a study that measured them separately has to be able to
    say so.
    """
    subj = _two_session_subject()
    script = write_script(
        build_plan(
            subj,
            Options(
                ref_ses="01",
                xrun_nonlin=True,
                xses_nonlin=True,
                xrun_nl_backend="optiwarp_demons",
                xses_nl_backend="qwarp",
            ),
        ),
        "wd",
        bids_root="/bids",
    )
    launches = set(re.findall(r"(ffs_\w+(?: -\w+ \w+)*) -batch ", script))
    assert any(ln.startswith("ffs_optiwarp -force demons") for ln in launches), launches
    assert any(ln.startswith("ffs_qwarp") for ln in launches), launches
    assert not any(ln.startswith("ffs_formwarp") for ln in launches), launches


def test_the_force_model_travels_with_the_optiwarp_backend():
    """-force demons is not a knob, it is which engine this is.

    The three optical-flow backends are one executable told apart by it, so it has
    to ride with the command and not with the tunable op string -- otherwise
    picking optiwarp_lk and overriding -xrun_nl_opts would silently run demons.
    """
    subj = _two_session_subject()
    script = write_script(
        build_plan(subj, Options(xrun_nonlin=True, xrun_nl_backend="optiwarp_lk")),
        "wd",
        bids_root="/bids",
    )
    assert "ffs_optiwarp -force lk -batch" in script


def test_preflight_checks_the_backend_the_script_will_call():
    """A script that never runs formwarp must not fail preflight for lacking it."""
    subj = _two_session_subject()
    script = write_script(
        build_plan(subj, Options(xrun_nonlin=True, xrun_nl_backend="optiwarp_demons")),
        "wd",
        bids_root="/bids",
    )
    tools = re.search(r"for t in ([^;]+); do", script).group(1).split()
    assert "ffs_optiwarp" in tools and "ffs_formwarp" not in tools
    # and the identity flag is not mistaken for a command
    assert "-force" not in tools and "demons" not in tools


def test_a_linear_only_script_names_no_nonlinear_tool():
    subj = _two_session_subject()
    script = write_script(build_plan(subj, Options()), "wd", bids_root="/bids")
    tools = re.search(r"for t in ([^;]+); do", script).group(1).split()
    assert not [t for t in tools if t in ("ffs_formwarp", "ffs_optiwarp", "ffs_qwarp")], tools


def test_backend_choice_moves_the_default_op_string():
    """An unset -<stage>_nl_opts means 'this backend's baseline', not formwarp's.

    formwarp's string carries -cc_radius/-update_var, which optiwarp and qwarp do
    not have; handing it to them would fail at the stage rather than at generation.
    """
    from fastfuncstuff.autoproc import config

    for backend in config.nl_backends():
        opts = config.nl_stage_opts(backend, "xrun_nl")
        if backend != "formwarp":
            assert "-update_var" not in opts and "-cc_radius" not in opts, (backend, opts)


def test_every_backend_baseline_is_spelled_correctly():
    """The baselines are checked against each engine's own parser, as -*_opts are.

    A default nobody typed is exactly the string a typo survives longest in.
    """
    from fastfuncstuff.autoproc import config, optcheck

    for backend in config.nl_backends():
        module = config.nl_command(backend).split()[0].removeprefix("ffs_")
        for stage in config.NL_STAGE_KEYS:
            errs = optcheck.check_opts(stage, config.nl_stage_opts(backend, stage), module=module)
            assert not errs, errs


def test_warpmaster_anchor_carries_the_nonlinear_anat_link():
    """The grid/mask/QC anchor must be in the shape the DATA ends up in.

    ffs_allineate's ``_al_anat`` stops at the affine, but with -anat_nonlin every
    run's CHAIN also gets the segment invwarp — a real PE-axis deformation. Boxing
    and automasking the affine-only image gave an output FOV and a GLM mask for a
    brain shape nothing lands in, and put a differently-distorted sub-brick 0 at
    the head of the final QC stack. Bug of record: sub-3001 ses-fast, ±6.1 mm.
    """
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")], anat=Path("/anat/T1w.nii.gz"))])
    nl_plan = build_plan(subj, Options(go_to_anat=True, anat_nonlin=True, tpm="/tpm.nii.gz"))
    nl = write_script(nl_plan, "wd", bids_root="/bids")
    anchor = '-source "stage08.grandmean.nii$FMT"'
    # The nl anchor is built from the SAME source allineate took, through the head
    # of the chain (affine + invwarp), onto the anat grid.
    assert '-prefix "stage09.grandmean_al_anat_nl.nii$FMT"' in nl
    assert '-nwarp "stage09.anat.aff12.1D stage09.nlanat_invwarp.nii$FMT"' in nl
    assert nl.count(anchor) >= 2  # allineate's source and the nwarp's are one image
    # ... and it, not the affine-only image, is what the grid and mask come from.
    assert 'MASTER="${FFS_MASTER:-stage09.grandmean_al_anat_nl.nii$FMT}"' in nl
    # Its links are exactly the head of a run's chain, in order.
    chain = _chain(nl_plan, ("01", "foo", "1"))
    assert chain[: chain.index("anat_nl") + 1] == ["anat_lin", "anat_nl"]

    # Affine-only pipeline: no extra image, and the affine anchor still rules.
    lin = write_script(build_plan(subj, Options(go_to_anat=True)), "wd", bids_root="/bids")
    assert "_al_anat_nl" not in lin
    assert 'MASTER="${FFS_MASTER:-stage09.grandmean_al_anat.nii$FMT}"' in lin


def _bids_run(root: Path, session, task, run, *, events=True, tr=2.0):
    """A BoldRun whose paths exist on disk, so BIDS events inheritance can resolve."""
    func = root / f"sub-X/ses-{session}/func"
    func.mkdir(parents=True, exist_ok=True)
    stem = f"sub-X_ses-{session}_task-{task}_run-{run}"
    mag = func / f"{stem}_bold.nii.gz"
    mag.touch()
    if events:
        (func / f"{stem}_events.tsv").write_text("onset\tduration\ttrial_type\n0\t10\ta\n")
    return BoldRun(
        subject="X",
        session=session,
        task=task,
        run=run,
        mag_path=mag,
        json={"RepetitionTime": tr, "PhaseEncodingDirection": "j-"},
    )


def _locomoco_manifest(script: str) -> str:
    """The printf line stage03 appends to its batch manifest."""
    for line in script.splitlines():
        if '>> "$lmbatch"' in line:
            return line
    raise AssertionError("no locomoco manifest line in the script")


def test_locomoco_gets_the_run_events_for_the_task_coupling_report(tmp_path):
    """-locomoco_task_qc: the diagnostic that says whether the field is reading BOLD
    as motion. It needs the design, per RUN, not per task."""
    subj = Subject("X", [Session("01", [_bids_run(tmp_path, "01", "foo", "1")])])
    plan = build_plan(subj, Options(locomoco=True, locomoco_task_qc=True))
    s = write_script(plan, "wd", bids_root=str(tmp_path))

    # The stimuli/ copy, not the BIDS original: it is what the design TOML names.
    assert "EVENTS[01:foo:1]=stimuli/sub-X_ses-01_task-foo_run-1_events.tsv" in s
    line = _locomoco_manifest(s)
    assert '${ev:+-events \\"${ev}\\"}' in line
    assert '${ev:+${TR[$k]:+-tr \\"${TR[$k]}\\"}}' in line
    assert "${ev:+-task_qc}" in line
    assert 'ev="${EVENTS[$k]:-}"' in s


def test_locomoco_events_fall_back_to_the_bids_original_without_a_glm(tmp_path):
    """No GLM means no stimuli/ copy is written, so the script has to name the source."""
    subj = Subject("X", [Session("01", [_bids_run(tmp_path, "01", "foo", "1")])])
    plan = build_plan(subj, Options(locomoco=True, locomoco_task_qc=True, run_glm=False))
    s = write_script(plan, "wd", bids_root=str(tmp_path))
    src = tmp_path / "sub-X/ses-01/func/sub-X_ses-01_task-foo_run-1_events.tsv"
    assert f"EVENTS[01:foo:1]={src}" in s
    assert str(src) in s.split("stage: preflight")[1]  # and preflight checks it


def test_a_run_without_events_simply_goes_unscored(tmp_path):
    """An unset EVENTS key must not trip `set -u`, and the run still gets corrected."""
    subj = Subject("X", [Session("01", [_bids_run(tmp_path, "01", "rest", "1", events=False)])])
    opt = Options(locomoco=True, locomoco_task_qc=True)
    s = write_script(build_plan(subj, opt), "wd", bids_root=str(tmp_path))
    assert "EVENTS[01:rest:1]" not in s
    assert 'ev="${EVENTS[$k]:-}"' in s


def test_locomoco_events_are_not_emitted_without_locomoco(tmp_path):
    subj = Subject("X", [Session("01", [_bids_run(tmp_path, "01", "foo", "1")])])
    opt = Options(locomoco=False, locomoco_task_qc=True)
    s = write_script(build_plan(subj, opt), "wd", bids_root=str(tmp_path))
    assert "EVENTS[01:foo:1]" not in s


def test_detask_is_opt_in(tmp_path):
    """-detask CHANGES the field the correction applies; the report does not."""
    subj = Subject("X", [Session("01", [_bids_run(tmp_path, "01", "foo", "1")])])
    plain = write_script(build_plan(subj, Options(locomoco=True)), "wd", bids_root=str(tmp_path))
    assert "-detask" not in _locomoco_manifest(plain)

    on = write_script(
        build_plan(subj, Options(locomoco=True, locomoco_detask="filter:2")),
        "wd",
        bids_root=str(tmp_path),
    )
    line = _locomoco_manifest(on)
    assert "${ev:+-detask filter:2}" in line
    # -detask needs the design, but asks for no QC.
    assert "-events" in line and "-task_qc" not in line


def test_per_run_stages_read_the_glms_event_columns(tmp_path):
    """-sep_spec_event_cols / -event_filter_* reached ffs_reml but not stage00/03, so
    locomoco -detask modelled the task from trial_type while the GLM used the column
    asked for. Run through bash: a filter value with a space has to survive both the
    shlex-split manifest and stage00's direct call."""
    import shlex
    import subprocess

    from fastfuncstuff.design.bids_events import EventFilter

    pr = _bids_run(tmp_path, "01", "foo", "1")
    ev = tmp_path / "sub-X/ses-01/func/sub-X_ses-01_task-foo_run-1_events.tsv"
    ev.write_text("onset\tduration\tcondition\ttrial_type\n0\t10\tface\tevent\n")
    subj = Subject("X", [Session("01", [pr])])
    opt = Options(
        locomoco=True,
        locomoco_detask="fit:2",
        want_nordic=True,
        nordic_task_qc=True,
        sep_spec_event_cols={"foo": ("onset", "duration", "condition")},
        event_filters={"foo": [EventFilter(column="condition", values=["a b"], action="out")]},
    )
    s = write_script(build_plan(subj, opt), "wd", bids_root=str(tmp_path))
    evopts = next(ln for ln in s.splitlines() if ln.startswith("EVOPTS["))
    want = ["-event_cols", "onset", "duration", "condition"]
    want += ["-event_filter_out", "condition", "a b"]

    line = _locomoco_manifest(s).replace('>> "$lmbatch"', "")
    out = subprocess.run(
        ["bash", "-c", f"declare -A EVOPTS TR; k=01:foo:1; ev=x.tsv; {evopts}\n{line}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    toks = shlex.split(out)
    i = toks.index("-event_cols")
    assert toks[i : i + len(want)] == want

    nordic = _nordic_command(s)
    assert '"${evopts[@]}"' in nordic
    stage00 = s.split("stage00: NORDIC denoise")[-1]
    setup = next(ln for ln in stage00.splitlines() if "evopts=()" in ln)
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'declare -A EVOPTS; k=01:foo:1; ev=x.tsv; {evopts}\n{setup}\nprintf "%s\\n" "${{evopts[@]}}"',
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.splitlines() == want

    # BIDS-default columns and no filters: nothing extra is emitted.
    plain = write_script(
        build_plan(subj, Options(locomoco=True, locomoco_detask="fit:2")),
        "wd",
        bids_root=str(tmp_path),
    )
    assert "EVOPTS[01" not in plain


def _nordic_command(script: str) -> str:
    """The ffs_nordic invocation stage00 emits, joined onto one line."""
    lines = script.split("stage00: NORDIC denoise")[-1].splitlines()
    start = next(i for i, ln in enumerate(lines) if "ffs_nordic" in ln)
    out = []
    for ln in lines[start:]:
        out.append(ln.rstrip("\\").strip())
        if not ln.rstrip().endswith("\\"):
            break
    return " ".join(out)


def test_nordic_gets_the_run_events_for_the_task_leak_report(tmp_path):
    """NORDIC throws away what a patch's spectrum calls noise, and nothing in that
    decision knows the task. The report that says whether the response went with it
    is -nordic_task_qc: it needs -events per RUN, and the TR the design is sampled at."""
    subj = Subject("X", [Session("01", [_bids_run(tmp_path, "01", "foo", "1")])])
    opt = Options(want_nordic=True, nordic_task_qc=True)
    s = write_script(build_plan(subj, opt), "wd", bids_root=str(tmp_path))
    assert "EVENTS[01:foo:1]=stimuli/sub-X_ses-01_task-foo_run-1_events.tsv" in s
    cmd = _nordic_command(s)
    assert '${ev:+-events "${ev}"}' in cmd
    assert '${ev:+${TR[$k]:+-tr "${TR[$k]}"}}' in cmd


def test_nordic_task_rescue_is_opt_in(tmp_path):
    """The report MEASURES; -task_rescue changes what the denoised output contains."""
    subj = Subject("X", [Session("01", [_bids_run(tmp_path, "01", "foo", "1")])])
    plan = build_plan(subj, Options(want_nordic=True))
    assert "-task_rescue" not in _nordic_command(write_script(plan, "wd", bids_root=str(tmp_path)))
    plan = build_plan(subj, Options(want_nordic=True, nordic_task_rescue="ica"))
    cmd = _nordic_command(write_script(plan, "wd", bids_root=str(tmp_path)))
    # Guarded by ${ev:+…} like -events itself: a run with no events must not get a
    # flag whose own tool rejects it for having nothing to score against.
    assert "${ev:+-task_rescue ica}" in cmd


def test_a_nordic_run_without_events_simply_goes_unscored(tmp_path):
    subj = Subject("X", [Session("01", [_bids_run(tmp_path, "01", "rest", "1", events=False)])])
    s = write_script(build_plan(subj, Options(want_nordic=True)), "wd", bids_root=str(tmp_path))
    assert "EVENTS[01:rest:1]" not in s
    assert 'ev="${EVENTS[$k]:-}"' in s.split("stage00: NORDIC denoise")[-1]


def _dofloss_jobs(script: str) -> list[str]:
    """The printf lines that append a dof-map resample to the nwarp batch."""
    return [
        ln
        for ln in script.splitlines()
        if "printf" in ln and ("stage10.dofloss." in ln or "$dofoutf" in ln)
    ]


def test_nordic_dof_adjust_is_wired_end_to_end(tmp_path):
    """The whole point: a NORDIC run's stats come out comparable to a plain one's.

    Three links have to hold together or the correction is silently wrong — the map
    has to be SAVED, carried into the final space by the run's OWN chain, and summed
    across exactly the runs the model concatenates.
    """
    runs = [_bids_run(tmp_path, "01", "foo", str(i)) for i in (1, 2, 3)]
    subj = Subject("X", [Session("01", runs)])
    s = write_script(build_plan(subj, Options(want_nordic=True)), "wd", bids_root=str(tmp_path))

    assert "-save_num_comps" in _nordic_command(s)
    job = _dofloss_jobs(s)
    assert len(job) == 1, job  # one loop over RUN_KEYS, not one line per run
    # THAT run's chain and the same master its timeseries took — but minus the
    # within-run motion links: a run-pooled count has no volume-0 pose.
    assert '-nwarp \\"${AVGCHAIN[$k]}\\"' in job[0]
    assert "${CHAIN[$k]}" not in job[0]
    assert "stage00.nordic.${FRAG[$k]}${E1[$k]:-}_numcomps" in job[0]
    assert "-master stage10.warpmaster" in job[0]
    # A count, not an intensity: no ringing kernel and no Jacobian modulation.
    assert "-interp linear" in job[0] and "-no_neg" in job[0]
    assert "-jac" not in job[0]
    # Summed over exactly this task's three runs, and handed to the GLM.
    total = next(ln for ln in s.splitlines() if "ffs_util_3dmath" in ln and "dofloss" in ln)
    for i in (1, 2, 3):
        assert f"stage10.dofloss.ses-01.task-foo.run-{i}" in total
    assert "-sum" in total
    assert '-adjust_dof "stage11.dofloss.task-foo.nii$FMT"' in s


def test_no_nordic_dof_adjust_removes_every_link(tmp_path):
    """Off means off at all three stages, not a saved map nothing consumes."""
    subj = Subject("X", [Session("01", [_bids_run(tmp_path, "01", "foo", "1")])])
    plan = build_plan(subj, Options(want_nordic=True, nordic_dof_adjust=False))
    s = write_script(plan, "wd", bids_root=str(tmp_path))
    assert "-save_num_comps" not in _nordic_command(s)
    assert _dofloss_jobs(s) == []
    assert "-adjust_dof" not in s
    assert "stage11.dofloss" not in s


def test_dof_adjust_needs_nordic_to_have_run(tmp_path):
    """Without NORDIC there is no component count, so nothing may claim one."""
    subj = Subject("X", [Session("01", [_bids_run(tmp_path, "01", "foo", "1")])])
    s = write_script(build_plan(subj, Options(want_nordic=False)), "wd", bids_root=str(tmp_path))
    assert "-adjust_dof" not in s and "dofloss" not in s


def test_task_qc_is_opt_in_per_stage(tmp_path):
    """The two diagnostics are off by default, independent, and neither drags the other."""
    subj = Subject("X", [Session("01", [_bids_run(tmp_path, "01", "foo", "1")])])

    s = write_script(
        build_plan(subj, Options(want_nordic=True, locomoco=True)), "wd", bids_root=str(tmp_path)
    )
    assert "-events" not in _nordic_command(s) and "-task_qc" not in _nordic_command(s)
    assert "-events" not in _locomoco_manifest(s) and "-task_qc" not in _locomoco_manifest(s)

    plan = build_plan(subj, Options(want_nordic=True, locomoco=True, nordic_task_qc=True))
    s = write_script(plan, "wd", bids_root=str(tmp_path))
    assert "-events" in _nordic_command(s) and "${ev:+-task_qc}" in _nordic_command(s)
    assert "-events" not in _locomoco_manifest(s)

    plan = build_plan(subj, Options(want_nordic=True, locomoco=True, locomoco_task_qc=True))
    s = write_script(plan, "wd", bids_root=str(tmp_path))
    assert "-events" not in _nordic_command(s)
    assert "-events" in _locomoco_manifest(s) and "${ev:+-task_qc}" in _locomoco_manifest(s)


def test_no_events_consumer_means_no_events_array(tmp_path):
    """With no task step anywhere, nothing should resolve or preflight an events TSV."""
    subj = Subject("X", [Session("01", [_bids_run(tmp_path, "01", "foo", "1")])])
    plan = build_plan(subj, Options(want_nordic=True, locomoco=True, run_glm=False))
    s = write_script(plan, "wd", bids_root=str(tmp_path))
    assert "EVENTS[01:foo:1]" not in s


def test_nordic_task_rescue_brings_its_events_without_the_qc_flag(tmp_path):
    """-task_rescue scores against the design, so it gets events on its own."""
    subj = Subject("X", [Session("01", [_bids_run(tmp_path, "01", "foo", "1")])])
    plan = build_plan(subj, Options(want_nordic=True, nordic_task_rescue="ica"))
    cmd = _nordic_command(write_script(plan, "wd", bids_root=str(tmp_path)))
    assert "-events" in cmd and "${ev:+-task_rescue ica}" in cmd
    assert "-task_qc" not in cmd


def test_locomoco_opts_that_need_the_design_bring_events(tmp_path, monkeypatch):
    """-reject / -ref paired passed through -locomoco_opts read the design too."""
    from fastfuncstuff.autoproc import config

    subj = Subject("X", [Session("01", [_bids_run(tmp_path, "01", "foo", "1")])])
    for opts, want in (("-ref paired", True), ("-ref paired_mean", True), ("-ref mean", False)):
        monkeypatch.setitem(config.DEFAULT_OPTS, "locomoco", f"-backend flow {opts}")
        s = write_script(build_plan(subj, Options(locomoco=True)), "wd", bids_root=str(tmp_path))
        assert ("-events" in _locomoco_manifest(s)) is want, opts


def _skull_script(**kw):
    subj = Subject("X", [Session("01", [_run("01", "foo", "1"), _run("01", "foo", "2")])])
    opt = Options(go_to_anat=True, anat_path="/data/sub-X_T1w.nii.gz", **kw)
    return write_script(build_plan(subj, opt), "wd", bids_root="/bids")


def test_anat_skull_strips_before_anything_aligns_to_the_anat():
    """-anat_skull yes: everything downstream must see the BRAIN, not the head —
    the autobox, the allineate base and the final grid all descend from $ANAT."""
    s = _skull_script(anat_skull=True)
    assert 'mri_synthstrip -i "stage09.anat_head.nii.gz" -o "stage09.anat_brain.nii.gz"' in s
    assert 'ANAT="stage09.anat_brain.nii.gz"' in s
    assert 'cp -f "/data/sub-X_T1w.nii.gz" "stage09.anat_head.nii.gz"' in s  # kept as given
    assert "mri_synthstrip" in s.split("stage: preflight")[1].split("stage")[0]
    # The strip precedes the crop that every later stage inherits its FOV from.
    assert s.index("mri_synthstrip -i") < s.index('-prefix "stage09.anat_autobox.nii.gz"')


def test_anat_skull_no_uses_the_anat_as_given():
    s = _skull_script()
    assert "synthstrip" not in s
    assert 'ANAT="/data/sub-X_T1w.nii.gz"' in s


def test_multiple_anats_are_aligned_averaged_then_stripped():
    """Order matters: the strip and everything downstream must see the MEAN, and
    the mean has to be built from aligned images (a mean of unaligned T1w is a
    blur, and it is the thing every cross-modal cost then works from)."""
    s = _skull_script(anat_extra=["/data/sub-X_run-2_T1w.nii.gz"], anat_skull=True)
    assert '-base "/data/sub-X_T1w.nii.gz"' in s  # first anat is the alignment base
    assert '-prefix "stage09.anat_in2_al.nii.gz"' in s
    assert '-input "/data/sub-X_T1w.nii.gz" "stage09.anat_in2_al.nii.gz" \\\n    -mean' in s
    assert 'mri_synthstrip -i "stage09.anat_avg.nii.gz"' in s
    assert s.index('-prefix "stage09.anat_avg.nii.gz"') < s.index("mri_synthstrip -i")
    # No second copy of the average under the head name — it IS the strip's record.
    assert "stage09.anat_head" not in s


def test_multiple_anats_without_a_skull_still_average():
    """-anat_skull no is about stripping, not about averaging: repeats of an
    already-stripped anat are just as worth averaging."""
    s = _skull_script(anat_extra=["/data/sub-X_run-2_T1w.nii.gz"])
    assert "synthstrip" not in s
    assert 'ANAT="stage09.anat_avg.nii.gz"' in s


def test_presurf_mprageises_each_input_then_averages_then_strips():
    """MP2RAGE: UNI's amplified-noise background must never reach synthstrip or
    the average. Each repeat is MPRAGEised by its own INV2 first, the mean is of
    those, and the strip reads the mean."""
    presurf = [
        ("/b/acq-uni_run-01_T1w.nii.gz", "/b/acq-inv2_run-01_T1w.nii.gz", None),
        ("/b/acq-uni_run-02_T1w.nii.gz", "/b/acq-inv2_run-02_T1w.nii.gz", None),
    ]
    s = _skull_script(
        anat_extra=[presurf[1][0]], anat_skull=True, presurf=presurf, presurf_tpm="/t/TPM.nii"
    )
    assert (
        '-inv2 /b/acq-inv2_run-02_T1w.nii.gz \\\n    -tpm /t/TPM.nii \\\n    -prefix "presurf_work/anat2.nii.gz"'
        in s
    )
    assert "-inv1" not in s and "-denoise" not in s
    # the average is built from the MPRAGEised heads, based on the first
    assert '-base "presurf_work/anat1_MPRAGEised.nii.gz"' in s
    assert "-source presurf_work/anat2_MPRAGEised.nii.gz" in s
    order = [
        s.index('-prefix "presurf_work/anat2.nii.gz"'),
        s.index('-prefix "stage09.anat_avg.nii.gz"'),
        s.index('mri_synthstrip -i "stage09.anat_avg.nii.gz"'),
    ]
    assert order == sorted(order)
    assert "ffs_util_presurf" in s.split("stage: preflight")[1].split("stage")[0]


def test_presurf_denoise_reg_names_the_head_it_makes():
    presurf = [("/b/uni.nii.gz", "/b/inv2.nii.gz", "/b/inv1.nii.gz")]
    s = _skull_script(
        anat_skull=True, presurf=presurf, presurf_denoise="reg", presurf_tpm="/t/T.nii"
    )
    assert "-inv1 /b/inv1.nii.gz" in s and "-denoise reg" in s
    assert 'cp -f "presurf_work/anat1_UNIreg.nii.gz" "stage09.anat_head.nii.gz"' in s


def test_do_mni_puts_both_links_at_the_head_of_every_chain():
    """MNI is the LAST thing that happens to the data, which in this chain order
    (leftmost acts first on coordinates) means the two leftmost tokens — and it
    has to stay inside the one resample every run already gets."""
    s = _skull_script(anat_skull=True, do_mni=True, mni_template="/t/MNI.nii.gz")
    chain = next(line for line in s.splitlines() if line.startswith("CHAIN["))
    links = chain.split('="')[1].rstrip('"').split()
    assert links[:3] == [
        "stage09.mni_nl_WARP.nii.gz",
        "stage09.mni.aff12.1D",
        "stage09.anat.aff12.1D",
    ]
    # One resample, not two: the only ffs_nwarp -do_mni adds anywhere in the
    # script is the single anchor call, never one per run.
    assert s.count("ffs_nwarp") == _skull_script(anat_skull=True).count("ffs_nwarp") + 1


def test_do_mni_is_estimated_on_the_stripped_anat_against_the_template():
    """Plain affine-then-nonlinear, both from the anat to the template. The
    template's sub-brick 0 is its skull-off volume."""
    s = _skull_script(anat_skull=True, do_mni=True, mni_template="/t/MNI.nii.gz")
    assert "-base '/t/MNI.nii.gz[0]'" in s
    assert '-source "stage09.anat_autobox.nii.gz"' in s  # the stripped anat
    assert '-1Dmatrix_save "stage09.mni.aff12.1D"' in s
    # the nonlinear half starts from the affine result, and writes the chain's warp
    assert '-source "stage09.anat_mni_lin.nii.gz"' in s
    assert '-warp_prefix "stage09.mni_nl"' in s
    assert "-type MNI_T1a" in s


def test_do_mni_moves_the_final_grid_and_underlay_into_mni():
    """The warpmaster is cut out of the anchor in the OUTPUT space. If it stayed in
    anat space the data would land on a grid one warp away from where it is."""
    s = _skull_script(anat_skull=True, do_mni=True, mni_template="/t/MNI.nii.gz")
    assert 'MASTER="${FFS_MASTER:-stage09.grandmean_al_mni.nii$FMT}"' in s
    assert '-input "stage09.anat_mni.nii.gz"' in s  # underlay ancestor
    # and the anchor gets there through the same head-of-chain the data does
    assert '-nwarp "stage09.mni_nl_WARP.nii.gz stage09.mni.aff12.1D stage09.anat.aff12.1D"' in s


def test_do_mni_coexists_with_anat_nonlin():
    """Different jobs: anat_nl brings the EPI to the anat (PE-axis, from segment),
    mni_* takes the anat to the template. Both links, in that order."""
    s = _skull_script(
        anat_skull=True,
        do_mni=True,
        mni_template="/t/MNI.nii.gz",
        anat_nonlin=True,
        tpm="/t/tpm.nii.gz",
    )
    chain = next(line for line in s.splitlines() if line.startswith("CHAIN["))
    links = chain.split('="')[1].rstrip('"').split()
    assert links[:4] == [
        "stage09.mni_nl_WARP.nii.gz",
        "stage09.mni.aff12.1D",
        "stage09.anat.aff12.1D",
        "stage09.nlanat_invwarp.nii$FMT",
    ]


def test_do_mni_stays_out_of_the_grandmean_chain():
    """Bug of record: the grandmean is what the anat step ALIGNS, and the MNI warp
    is estimated from the result — so a run resampled into grandmean space must
    not carry it. Left in, stage08b asked for a warp that does not exist yet."""
    s = _skull_script(anat_skull=True, do_mni=True, mni_template="/t/MNI.nii.gz")
    gm = [line for line in s.splitlines() if "stage08.gmrun" in line or "$GMGRID" in line]
    assert not any("mni" in line for line in gm), gm
    # ...and the anat-space anchor stops at the anat, too.
    assert '-nwarp "stage09.mni_nl_WARP.nii.gz stage09.mni.aff12.1D"' not in s


def _chain_links(s: str) -> list[str]:
    chain = next(line for line in s.splitlines() if line.startswith("CHAIN["))
    return chain.split('="')[1].rstrip('"').split()


def test_ref_anat_links_sit_between_the_anat_and_the_epi_alignment():
    """-ref_anat: the anat stays the output space and the EPI aligns to the ref, so
    on the output coordinate the ref links act first (anat → ref), then the EPI's
    own anat_lin/anat_nl (ref → EPI); MNI stays outermost. The ref warp is
    segment's INVwarp — the forward warp maps ref → anat, the wrong way round."""
    s = _skull_script(
        anat_skull=True,
        anat_nonlin=True,
        tpm="/t/tpm.nii.gz",
        ref_anat="/data/sub-X_acq-7T_T1w.nii.gz",
        do_mni=True,
        mni_template="/t/MNI.nii.gz",
    )
    assert _chain_links(s)[:6] == [
        "stage09.mni_nl_WARP.nii.gz",
        "stage09.mni.aff12.1D",
        "stage09.refanat.aff12.1D",
        "stage09.refanat_seg_invwarp.nii.gz",
        "stage09.anat.aff12.1D",
        "stage09.nlanat_invwarp.nii$FMT",
    ]
    # Estimated on the anat as before: -ref_anat leaves the MNI hop alone.
    assert '-source "stage09.anat_autobox.nii.gz"' in s
    gm = [line for line in s.splitlines() if "stage08.gmrun" in line or "$GMGRID" in line]
    assert not any("refanat" in line for line in gm), gm


def test_ref_anat_is_what_the_epi_aligns_to_and_segments_against():
    """The EPI shares the ref's gradients, so both EPI fits use the ref: the anat
    matrix's base, and anat_nl's priors -- the anat's TPM as the ref's own
    posteriors, made by the same ffs_segment that finds the ref → anat warp."""
    s = _skull_script(
        anat_skull=True,
        anat_nonlin=True,
        tpm="/t/tpm.nii.gz",
        ref_anat="/data/sub-X_acq-7T_T1w.nii.gz",
    )
    ref_seg = s.index('-prefix "stage09.refanat_seg.nii.gz"')
    epi_lin = s.index('-1Dmatrix_save "stage09.anat.aff12.1D"')
    assert ref_seg < epi_lin  # the ref (and its TPM) exist before the EPI step
    seg = s[s.rindex("ffs_segment", 0, ref_seg) : ref_seg]
    assert '-input "stage09.refanat_autobox.nii.gz"' in seg  # the head, skull on
    assert "-tpm /t/tpm.nii.gz" in seg
    assert '-1Dmatrix "stage09.refanat.aff12.1D"' in seg
    assert '-matrix_base "stage09.anat_autobox.nii.gz"' in seg
    assert "-pe_axis" not in seg  # free on all three axes
    lin = s[s.rindex("ffs_allineate", 0, epi_lin) : epi_lin]
    assert '-base "stage09.refanat_brain.nii.gz"' in lin
    epi_seg = s[s.index('-prefix "stage09.nlanat.nii$FMT"') - 600 :]
    epi_seg = epi_seg[: epi_seg.index('-prefix "stage09.nlanat.nii$FMT"')]
    assert '-tpm "tpm_work/refanat_TPM.nii.gz"' in epi_seg
    assert '-matrix_base "stage09.refanat_brain.nii.gz"' in epi_seg


def test_ref_anat_tpm_keeps_only_the_tpms_own_classes_on_the_ref_grid():
    """Real-data bug: an FS TPM (8 classes, no air) makes the ref segment add a 9th,
    background posterior; tcat'd whole, the EPI fit then saw 9 "real" classes and
    refused -ngaus ×8. And -suma's SurfVol, the anat TPM's grid, is not the ref's."""
    s = _skull_script(
        anat_skull=True,
        anat_nonlin=True,
        tpm="/t/tpm.nii.gz",
        tpm_source="/fs/SUMA/sub_SurfVol.nii",
        ref_anat="/data/sub-X_acq-7T_T1w.nii.gz",
    )
    build = s[s.index('[ -f "tpm_work/refanat_TPM.nii.gz" ] ||') :]
    build = build[: build.index('-prefix "tpm_work/refanat_TPM.nii.gz"')]
    assert "tpm_n=$(ffs_info -nv /t/tpm.nii.gz)" in build
    assert 'sort -V | head -n "$tpm_n"' in build
    epi_seg = s[s.index('-prefix "stage09.nlanat.nii$FMT"') - 600 :]
    epi_seg = epi_seg[: epi_seg.index('-prefix "stage09.nlanat.nii$FMT"')]
    assert "-tpm_source stage09.refanat_autobox.nii.gz" in epi_seg
    assert "SurfVol" not in epi_seg


def test_ref_anat_keeps_the_anat_as_the_final_space():
    """Without anat_nl too: the anchor still has to go through the ref links, or
    the final grid would be cut from an image sitting on the ref."""
    s = _skull_script(anat_skull=True, ref_anat="/data/sub-X_acq-7T_T1w.nii.gz")
    assert 'MASTER="${FFS_MASTER:-stage09.grandmean_al_anat_nl.nii$FMT}"' in s
    anchor = s[: s.index('-prefix "stage09.grandmean_al_anat_nl.nii$FMT"')]
    anchor = anchor[anchor.rindex("ffs_nwarp") :]
    assert (
        '-nwarp "stage09.refanat.aff12.1D stage09.refanat_seg_invwarp.nii.gz stage09.anat.aff12.1D"'
        in anchor
    )
    assert '-master "stage09.anat_autobox.nii.gz"' in anchor
    assert '-input "stage09.anat_autobox.nii.gz"' in s  # underlay ancestor unchanged


def test_ref_anat_register_chains_the_backend_warp_in_the_same_slot():
    """register: the backend runs in the ref's frame (-matrix + the raw ref), so its
    warp takes segment's slot and nothing else in the chain moves."""
    s = _skull_script(
        anat_skull=True,
        ref_anat="/data/sub-X_acq-7T_T1w.nii.gz",
        ref_anat_nonlin_method="register",
    )
    assert _chain_links(s)[:3] == [
        "stage09.refanat.aff12.1D",
        "stage09.refanat_nl_WARP.nii.gz",
        "stage09.anat.aff12.1D",
    ]
    assert '-matrix "stage09.refanat.aff12.1D"' in s
    assert '-warp_prefix "stage09.refanat_nl"' in s
    # Nothing reads a TPM here (no -anat_nonlin), so no segment of the ref.
    assert "refanat_seg" not in s


def test_ref_anat_mp2rage_is_mprageised_before_anything_reads_it():
    """An MP2RAGE ref: presurfer's MPRAGEised head replaces the UNI from the autobox
    on, so the ref's strip, rigid fit and segmentation never see UNI's noise
    background. Its classes are not wanted (-no_classes); only the head is used."""
    uni = "/bids/sub-X/anat/sub-X_acq-uni_T1w.nii.gz"
    s = _skull_script(
        anat_skull=True,
        ref_anat=uni,
        ref_presurf=(uni, "/bids/sub-X/anat/sub-X_acq-inv2_T1w.nii.gz", None),
        presurf_tpm="/t/TPM.nii",
    )
    end = s.index('-prefix "presurf_work/refanat.nii.gz"')
    call = s[s.rindex("ffs_util_presurf", 0, end) : end + 200]
    assert f"-uni {uni}" in call
    assert "-inv2 /bids/sub-X/anat/sub-X_acq-inv2_T1w.nii.gz" in call
    assert "-no_classes" in call
    box = s[s.index("ffs_util_autobox", s.index("refanat.nii.gz")) :]
    assert box.split("\n")[1].strip() == '-input "presurf_work/refanat_MPRAGEised.nii.gz" \\'
    assert s.index("presurf_work/refanat.nii.gz") < s.index(
        '-1Dmatrix_save "stage09.refanat.aff12.1D"'
    )
    assert "ffs_util_presurf" in s.split("stage: preflight")[1].split("stage")[0]


def test_surface_target_samples_each_run_through_its_chain_up_to_the_anat():
    """-surface_target: the SAME one interpolation as stage10, read at surface points.
    The surfaces live in the anat's scanner space, so the MNI links must not be in
    the chain and the master is $ANAT, not the (MNI) warpmaster."""
    s = _skull_script(
        do_mni=True,
        mni_template="/t/MNI.nii.gz",
        surface_target=["native", "onavg-ico64"],
        fs_subject="/fs/sub-X",
    )
    surf = [line for line in s.splitlines() if line.startswith("SURFCHAIN[")]
    chain = [line for line in s.splitlines() if line.startswith("CHAIN[")]
    assert len(surf) == len(chain) == 2
    for sc, c in zip(surf, chain, strict=True):
        links = c.split('="')[1].rstrip('"').split()
        assert sc.split('="')[1].rstrip('"').split() == [x for x in links if "mni" not in x]
    assert "stage10s" in s and "skip_surface" not in s
    # The surface rides stage10's own per-run line: one process, one read of the source.
    line = next(x for x in s.splitlines() if "-surf " in x and "printf" in x)
    assert '-surf_master \\"$ANAT\\"' in line
    assert '-surf_nwarp \\"${SURFCHAIN[$k]}\\"' in line
    # Each pass corrects the cardinal-frame slip against its own chain's base: the
    # volume's (MNI) chain was fitted on $MASTER, SURFCHAIN's on the anat box.
    assert '-chain_base \\"$MASTER\\"' in line
    assert '-surf_chain_base \\"stage09.anat_autobox.nii.gz\\"' in line
    assert '-surf_prefix \\"stage10s.surf.${FRAG[$k]}\\"' in line
    assert "-surf /fs/sub-X -surf_mesh native onavg-ico64" in line
    # the volume half keeps its own (MNI) chain and warpmaster
    assert '-nwarp \\"${CHAIN[$k]}\\"' in line and "-master stage10.warpmaster" in line
    assert sum("printf" in x and "-surf " in x for x in s.splitlines()) == 1
    # cross-run masks, per space and hemisphere, after the runs are written
    for space in ("native", "onavg-ico64"):
        for hemi in ("lh", "rh"):
            assert f'-prefix "stage10s.surf.allruns.{space}.{hemi}"' in s
    assert s.index("== stage10: final") < s.index("== stage10s: surface masks")


def test_surface_chain_array_survives_bash():
    """Run keys are "ses:task:run"; an array missing from `declare -A` is indexed, and
    bash evaluates the key as arithmetic -- every SURFCHAIN assignment failed that way
    on real data while the string checks above passed."""
    import subprocess

    version = subprocess.run(
        ["bash", "-c", 'echo "${BASH_VERSINFO[0]}"'], capture_output=True, text=True, check=True
    )
    if int(version.stdout) < 4:
        pytest.skip("pipeline associative arrays require Bash >= 4; preamble rejection is tested")
    s = _skull_script(surface_target=["native"], fs_subject="/fs/sub-X")
    lines = [x for x in s.splitlines() if x.startswith(("declare -A", "SURFCHAIN[", "CHAIN["))]
    keys = [x.split("[", 1)[1].split("]", 1)[0] for x in lines if x.startswith("SURFCHAIN[")]
    assert keys
    probe = "\n".join(lines + [f'echo "${{SURFCHAIN[{k}]}}"' for k in keys])
    out = subprocess.run(["bash", "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0 and out.stderr == "", out.stderr
    assert all(line.strip() for line in out.stdout.splitlines())
    assert len(out.stdout.splitlines()) == len(keys)


def test_surface_qc_rides_the_same_line():
    s = _skull_script(surface_target=["native"], fs_subject="/fs/sub-X", surface_qc=True)
    line = next(x for x in s.splitlines() if "-surf " in x and "printf" in x)
    assert " -surf_qc" in line
    plain = _skull_script(surface_target=["native"], fs_subject="/fs/sub-X")
    assert "-surf_qc" not in plain


def test_no_surface_target_emits_no_surface_stage():
    s = _skull_script()
    assert "SURFCHAIN" not in s and "stage10s" not in s and "skip_surface" not in s


def test_glm_blur_tags_the_buckets_and_leaves_preprocessing_alone():
    """-glm_blur smooths inside ffs_reml and labels the outputs, so a second FWHM
    is a stage12-only re-run that does not overwrite the first fit."""
    from fastfuncstuff.autoproc.naming import blur_tag, glm_tag

    subj = Subject("X", [Session("01", [_run("01", "foo", "1")])])
    plain = write_script(build_plan(subj, Options(run_glm=True)), "wd", bids_root="/bids")
    assert "-do_blur" not in plain
    assert '-Rbuck "stage12.stats-reml.task-foo.nii$GLM_FMT"' in plain

    blurred = write_script(
        build_plan(subj, Options(run_glm=True, glm_blur=6.0)), "wd", bids_root="/bids"
    )
    assert "-do_blur 6" in blurred
    assert '-Obuck "stage12.blur6.stats-ols.task-foo.nii$GLM_FMT"' in blurred
    assert '-Rbuck "stage12.blur6.stats-reml.task-foo.nii$GLM_FMT"' in blurred
    # Only the GLM changes: every earlier stage is byte-identical, which is what
    # makes "edit one parameter, re-run stage12" true.
    assert plain.split("stage12: GLM")[0] == blurred.split("stage12: GLM")[0]

    # A fractional FWHM keeps the token dot-free (the naming scheme is dot-delimited).
    assert blur_tag(2.5) == "blur2p5"
    assert blur_tag(None) == "" and blur_tag(0) == ""

    # -glm_label names any other variant, in the same slot and a fixed order, so
    # the same two settings always land on the same name.
    labelled = write_script(
        build_plan(subj, Options(run_glm=True, glm_blur=6.0, glm_label="noloco")),
        "wd",
        bids_root="/bids",
    )
    assert '-Rbuck "stage12.blur6.noloco.stats-reml.task-foo.nii$GLM_FMT"' in labelled
    assert glm_tag(None, "fir") == "fir"
    assert glm_tag(2.5, "fir") == "blur2p5.fir"
    assert glm_tag(None, None) == "" and glm_tag(6.0, "  ") == "blur6"


def test_glm_blur_per_task_overrides_the_dataset_wide_one():
    """-glm_blur FWHM TASK smooths one task differently; the bare -glm_blur is
    what every task it does not name gets. The blur token keeps them apart."""
    subj = Subject(
        "X",
        [Session("01", [_run("01", "img", "1"), _run("01", "loc", "2"), _run("01", "rest", "3")])],
    )

    both = write_script(
        build_plan(subj, Options(run_glm=True, glm_blur=3.0, sep_glm_blur={"img": 6.0})),
        "wd",
        bids_root="/bids",
    )
    assert '-Rbuck "stage12.blur6.stats-reml.task-img.nii$GLM_FMT"' in both
    for task in ("loc", "rest"):
        assert f'-Rbuck "stage12.blur3.stats-reml.task-{task}.nii$GLM_FMT"' in both
    assert both.count("-do_blur 6") == 1 and both.count("-do_blur 3") == 2

    # No bare -glm_blur: the tasks nothing names are not smoothed at all, and
    # their buckets carry no token (they are the plain fit).
    only = write_script(
        build_plan(subj, Options(run_glm=True, sep_glm_blur={"img": 6.0})),
        "wd",
        bids_root="/bids",
    )
    assert only.count("-do_blur") == 1 and "-do_blur 6" in only
    assert '-Rbuck "stage12.stats-reml.task-loc.nii$GLM_FMT"' in only
    # The skip guard keys on the per-task name, so re-running after changing one
    # task's FWHM refits that task and leaves the others' buckets alone.
    assert '[ ! -f "stage12.blur6.stats-reml.task-img.nii$GLM_FMT" ]' in only
    assert '[ ! -f "stage12.stats-reml.task-rest.nii$GLM_FMT" ]' in only


def test_glm_blur_cli_forms():
    """-glm_blur takes FWHM, optionally a TASK, and repeats."""
    from fastfuncstuff.cli.autoproc import _resolve_glm_blur, build_parser

    p = build_parser()
    base = ["-bids_dir", "/b", "-subject", "X"]

    def res(*argv):
        return _resolve_glm_blur(p.parse_args(base + list(argv)))

    assert res() == (None, {})
    assert res("-glm_blur", "3") == (3.0, {})
    assert res("-glm_blur", "3", "-glm_blur", "6", "img") == (3.0, {"img": 6.0})
    # A task may be spelled with or without the BIDS entity prefix.
    assert res("-glm_blur", "6", "task-img") == (None, {"img": 6.0})
    assert res("-glm_blur", "6", "a", "-glm_blur", "2", "b") == (None, {"a": 6.0, "b": 2.0})
    # Two tasks in one entry is a typo, and so is putting the name first.
    for bad in (("-glm_blur", "6", "a", "b"), ("-glm_blur", "img", "6")):
        with pytest.raises(SystemExit):
            res(*bad)


def test_stage12_guard_makes_skip_stats_real():
    """skip_stats sat in the preamble switching nothing: stage12 always refit and
    silently replaced the previous buckets. Now it guards like every other stage."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1"), _run("01", "bar", "2")])])
    sh = write_script(build_plan(subj, Options(run_glm=True)), "wd", bids_root="/bids")

    assert "skip_stats=1" in sh  # still declared...
    for task in ("foo", "bar"):  # ...and now read, once per task
        guard = (
            f'if [ "$skip_stats" -ne 1 ] || [ ! -f "stage12.stats-reml.task-{task}.nii$GLM_FMT" ]'
        )
        assert guard in sh
    assert sh.count('if [ "$skip_stats" -ne 1 ]') == 2
    # A skipped GLM says so: silence would read as "the model ran".
    assert sh.count("skipping the GLM") == 2

    # The guard keys on the tagged name, so changing -glm_blur refits without a
    # toggle flip -- a different bucket is a different question, not a re-run.
    blurred = write_script(
        build_plan(subj, Options(run_glm=True, glm_blur=4.0)), "wd", bids_root="/bids"
    )
    assert '[ ! -f "stage12.blur4.stats-reml.task-foo.nii$GLM_FMT" ]' in blurred


def test_stage10b_masks_and_glm_mask_modes():
    """stage10b masks the FINAL data (not the pre-resample anchor), builds all
    three masks whatever -glm_mask says, and -glm_mask picks which one reaches
    ffs_reml."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")], anat=Path("/anat/T1w.nii.gz"))])

    def gen(**kw):
        return write_script(
            build_plan(subj, Options(go_to_anat=True, run_glm=True, **kw)),
            "wd",
            bids_root="/bids",
        )

    s = gen()
    # After the data exists, not before: an automask of the warpmaster is an
    # automask of the anchor.
    assert s.index("stage10: final compose") < s.index("stage10b: masks")
    assert '-prefix "stage10.meanall.nii$FMT"' in s
    for m in ("mask_epi", "mask_anat", "mask_brain"):
        assert f'-prefix "stage10.{m}.nii$FMT"' in s
    # The anat side is masked at ANAT resolution and brought down nearest-neighbour
    # (never automasked after the downsample, never interpolated as a binary map),
    # and the intersection is a product of the two masks, not a third automask.
    assert '-input "stage10.anat_in_epi_fov.nii.gz"' in s
    assert '-prefix "stage10.mask_anat_hires.nii$FMT"' in s
    assert '-input "stage10.mask_anat_hires.nii$FMT"' in s
    assert '-master "stage10.warpmaster.nii$FMT"' in s and "-rmode NN" in s
    assert "-expr 'step(a)*step(b)'" in s
    stage10b = s[s.index("stage10b: masks") :].split("# ====", 2)[0]
    assert stage10b.count("ffs_util_automask") == 2  # meanall + the anat, and no more
    # Default is no mask at all.
    assert "GLM_MASK=" not in s and "-mask " not in s

    for mode, want in (
        ("epi", "stage10.mask_epi.nii$FMT"),
        ("anat", "stage10.mask_anat.nii$FMT"),
        ("brain", "stage10.mask_brain.nii$FMT"),
    ):
        g = gen(glm_mask=mode)
        assert f'GLM_MASK="${{FFS_GLM_MASK:-{want}}}"' in g
        assert '-mask "$GLM_MASK"' in g

    # No anat of our own: the masks that need one are not emitted, and the anat
    # modes fall back to the EPI mask rather than naming a file nobody writes.
    borrow = write_script(
        build_plan(subj, Options(grand_reference="/floc.results", run_glm=True, glm_mask="brain")),
        "wd",
        bids_root="/bids",
    )
    assert "stage10.mask_anat" not in borrow and "stage10.mask_brain" not in borrow
    assert 'GLM_MASK="${FFS_GLM_MASK:-stage10.mask_epi.nii$FMT}"' in borrow


def test_clustsim_rides_the_reml_command_not_a_separate_stage():
    """-clustsim is one flag on stage12: the ACF comes from residuals ffs_reml
    already holds, so there is no errts to write and no stage13 to run."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")])])
    off = write_script(build_plan(subj, Options(run_glm=True)), "wd", bids_root="/bids")
    assert "-clustsim" not in off

    on = write_script(
        build_plan(subj, Options(run_glm=True, clustsim=True)), "wd", bids_root="/bids"
    )
    assert "-clustsim" in on
    # No residual dataset, and no extra stage: the mask it needs is already
    # passed to the same command.
    assert "-Rerrts" not in on and "stage13" not in on
    assert "ffs_clustsim" not in on
    assert on.count("-clustsim") == 1
    # -clustsim needs no mask flag: with -glm_mask none ffs_reml automasks the
    # grand mean for its own diagnostics, which is where the ACF comes from.
    assert "-mask " not in on
    # Only the GLM command changes.
    assert off.split("stage12: GLM")[0] == on.split("stage12: GLM")[0]


def test_round_flags_parse_per_task():
    from fastfuncstuff.cli.autoproc import _resolve_round_modes, build_parser

    p = build_parser()
    base = ["-bids_dir", "/b", "-subject", "X"]

    def res(*argv):
        return _resolve_round_modes(p.parse_args(base + list(argv)).round_durations, "-rd")

    assert res() == (None, {})
    assert res("-round_durations", "0") == (0, {})
    assert res("-round_durations", "1", "-round_durations", "TR", "task-a", "b") == (
        1,
        {"a": "TR", "b": "TR"},
    )
    for bad in (("-round_durations", "0.5"), ("-round_durations", "loc08pos", "0")):
        with pytest.raises(SystemExit):
            res(*bad)


def test_round_flags_reach_the_design_toml(tmp_path: Path):
    """A per-task entry wins over the dataset-wide value; both land in [[events]]."""
    from fastfuncstuff.autoproc.bids import scan_subject
    from fastfuncstuff.autoproc.glm import write_design_specs
    from fastfuncstuff.design.spec import load_spec

    _bids_two_tasks(tmp_path)
    plan = build_plan(
        scan_subject(tmp_path, "ME1"),
        Options(
            run_glm=True, round_durations=0, sep_round_durations={"floc": "TR"}, round_onsets=1
        ),
    )
    write_design_specs(plan, str(tmp_path), str(tmp_path / "wd"))

    floc = load_spec(tmp_path / "wd" / "stage11.design.task-floc.toml")
    assert {(e.round_onset, e.round_duration) for e in floc.events} == {(1, "TR")}
    imagery = load_spec(tmp_path / "wd" / "stage11.design.task-imagery.toml")
    assert {(e.round_onset, e.round_duration) for e in imagery.events} == {(1, 0)}


def test_event_ignore_is_per_task_and_reaches_the_design_toml(tmp_path: Path):
    """-event_ignore drops a trial type from the model for every task;
    -sep_event_ignore wins for the task it names. Both land in that task's TOML
    as drop_trial_types, and REPLACE the rest/baseline default."""
    from fastfuncstuff.autoproc.bids import scan_subject
    from fastfuncstuff.autoproc.glm import write_design_specs
    from fastfuncstuff.design.spec import load_spec

    func = _bids_two_tasks(tmp_path)
    (func / "sub-ME1_ses-SM_task-floc_run-01_events.tsv").write_text(
        "onset\tduration\ttrial_type\n2\t4\tface\n10\t4\tgray\n18\t4\trest\n"
    )
    plan = build_plan(
        scan_subject(tmp_path, "ME1"),
        Options(run_glm=True, event_ignore=["gray"], sep_event_ignore={"imagery": []}),
    )
    write_design_specs(plan, str(tmp_path), str(tmp_path / "wd"))

    floc = load_spec(tmp_path / "wd" / "stage11.design.task-floc.toml")
    assert floc.meta.drop_trial_types == ["gray"]
    # The default list is replaced, not added to: 'rest' is now modelled.
    assert {e.trial_type for e in floc.events} == {"face", "rest"}
    imagery = load_spec(tmp_path / "wd" / "stage11.design.task-imagery.toml")
    assert imagery.meta.drop_trial_types == []


def test_event_ignore_parses_and_defaults_to_none():
    from fastfuncstuff.cli.autoproc import _resolve_event_ignore, build_parser

    p = build_parser()
    base = ["-bids_dir", "/b", "-subject", "X"]
    # No flag at all leaves the spec's own default drop list in charge.
    assert _resolve_event_ignore(p.parse_args(base)) == (None, {})
    # A bare -event_ignore is "model everything", which is not the same as None.
    assert _resolve_event_ignore(p.parse_args([*base, "-event-ignore"])) == ([], {})
    wide, per_task = _resolve_event_ignore(
        p.parse_args(
            [*base, "-event_ignore", "gray", "-sep_event_ignore", "task-floc", "rest", "cue"]
        )
    )
    assert wide == ["gray"]
    assert per_task == {"floc": ["rest", "cue"]}


def test_event_filters_parse_per_task():
    from fastfuncstuff.cli.autoproc import _resolve_event_filters, build_parser

    p = build_parser()
    args = p.parse_args(
        ["-bids_dir", "/b", "-subject", "X"]
        + ["-event_filter_out", "task-loc08pos", "hemifield", "right"]
        + ["-event_filter_out", "loc08pos", "vertical_field", "upper", "lower"]
        + ["-event_filter_in", "floc", "trial_type", "face"]
    )
    got = {t: [f.describe() for f in fs] for t, fs in _resolve_event_filters(args).items()}
    assert got == {
        "loc08pos": [
            "-event_filter_out hemifield right",
            "-event_filter_out vertical_field upper lower",
        ],
        "floc": ["-event_filter_in trial_type face"],
    }
    with pytest.raises(SystemExit):
        _resolve_event_filters(
            p.parse_args(
                ["-bids_dir", "/b", "-subject", "X", "-event_filter_out", "loc08pos", "hemifield"]
            )
        )


def test_filtered_task_gets_a_labelled_toml_with_its_filters(tmp_path: Path):
    """Left and right variants of one task must not share a TOML: an existing one
    is kept, so the second variant would silently fit the first one's design."""
    from fastfuncstuff.autoproc.bids import scan_subject
    from fastfuncstuff.autoproc.glm import write_design_specs
    from fastfuncstuff.design.bids_events import EventFilter
    from fastfuncstuff.design.spec import load_spec

    _bids_two_tasks(tmp_path)
    subj = scan_subject(tmp_path, "ME1")
    wd = tmp_path / "wd"

    def run(label, value, overwrite=False):
        opt = Options(
            run_glm=True,
            glm_label=label,
            glm_spec_overwrite=overwrite,
            event_filters={"floc": [EventFilter("trial_type", [value], "out")]},
        )
        plan = build_plan(subj, opt)
        return plan, write_design_specs(plan, str(tmp_path), str(wd))

    plan, _ = run("noface", "face")
    run("noplace", "place")
    assert [
        e.trial_type for e in load_spec(wd / "stage11.design.noface.task-floc.toml").events
    ] == ["place"]
    assert [
        e.trial_type for e in load_spec(wd / "stage11.design.noplace.task-floc.toml").events
    ] == ["face"]
    # The unfiltered task keeps its plain name even under a label.
    assert (wd / "stage11.design.task-imagery.toml").is_file()

    s = write_script(plan, str(wd), bids_root=str(tmp_path))
    assert "-spec stage11.design.noface.task-floc.toml" in s

    # Same label, different filter: kept, but not silently.
    _, rows = run("noface", "place")
    status = dict((t, st) for t, _, st in rows)["floc"]
    assert status.startswith("kept") and "WARNING" in status


def test_prebuilt_contrasts_are_validated_and_pasted_verbatim(tmp_path: Path):
    from fastfuncstuff.autoproc.bids import scan_subject
    from fastfuncstuff.autoproc.glm import write_design_specs
    from fastfuncstuff.design.spec import load_spec

    _bids_two_tasks(tmp_path)
    subj = scan_subject(tmp_path, "ME1")
    good = tmp_path / "floc_contrasts.toml"
    good.write_text(
        "[[contrasts]]                       # t-test, classic A minus B\n"
        'label = "face_vs_place"\nsym = "SYM: +1*face -1*place"\nbalance = "none"\n'
    )
    wd = tmp_path / "wd"
    plan = build_plan(subj, Options(run_glm=True, prebuilt_contrasts={"floc": [str(good)]}))
    rows = write_design_specs(plan, str(tmp_path), str(wd))
    assert dict((t, st) for t, _, st in rows)["floc"] == "wrote"
    toml = wd / "stage11.design.task-floc.toml"
    assert "# t-test, classic A minus B" in toml.read_text()  # comments survive
    assert [c.label for c in load_spec(toml).contrasts] == ["face_vs_place"]
    assert load_spec(wd / "stage11.design.task-imagery.toml").contrasts == []

    # A kept TOML that lacks them says so.
    bad = tmp_path / "typo.toml"
    bad.write_text('[[contrasts]]\nlabel = "f"\nsym = "SYM: +1*fcae"\n')
    plan = build_plan(subj, Options(run_glm=True, prebuilt_contrasts={"floc": [str(bad)]}))
    status = dict((t, st) for t, _, st in write_design_specs(plan, str(tmp_path), str(wd)))
    assert "lacks the prebuilt contrast(s) f" in status["floc"]

    # A label the design will not build is an error, and nothing is written.
    plan = build_plan(
        subj,
        Options(run_glm=True, glm_spec_overwrite=True, prebuilt_contrasts={"floc": [str(bad)]}),
    )
    toml.unlink()
    status = dict((t, st) for t, _, st in write_design_specs(plan, str(tmp_path), str(wd)))
    assert status["floc"].startswith("error:") and "fcae" in status["floc"]
    assert not toml.exists()


def test_prebuilt_contrasts_flag_refuses_missing_or_non_contrast_files(tmp_path: Path):
    from fastfuncstuff.cli.autoproc import _resolve_prebuilt_contrasts, build_parser

    p = build_parser()
    base = ["-bids_dir", "/b", "-subject", "X"]
    ok = tmp_path / "c.toml"
    ok.write_text('[[contrasts]]\nlabel = "x"\nsym = "SYM: +1*a"\n')
    got = _resolve_prebuilt_contrasts(
        p.parse_args(base + ["-prebuilt_contrasts", "task-t", str(ok)])
    )
    assert got == {"t": [str(ok.resolve())]}
    notcon = tmp_path / "n.toml"
    notcon.write_text("[meta]\ntr = 2\n")
    for path in (tmp_path / "missing.toml", notcon):
        with pytest.raises(SystemExit):
            _resolve_prebuilt_contrasts(
                p.parse_args(base + ["-prebuilt_contrasts", "t", str(path)])
            )


def test_motsim_rides_along_with_stage02_and_feeds_the_glm():
    """MotSim is estimated by the correction that produced the motion, not by a
    stage of its own: that is what lets the backward variant re-register under
    the same settings. The GLM then globs the .1D stage02 wrote."""
    from fastfuncstuff.autoproc import config

    subj = Subject("X", [Session("01", [_run("01", "foo", "1")])])

    s = write_script(
        build_plan(subj, Options(motsim="forward,0.95", glm_ortvec=["motsim"])),
        "wd",
        bids_root="/bids",
    )
    assert "-motsim forward,0.95" in s
    assert '-motsim_1D \\"${mstem}.motsim.1D\\"' in s
    # No second tool invocation — the flags are on the stage02 ffs_moco line.
    assert "ffs_motsim" not in s

    off = write_script(build_plan(subj, Options()), "wd", bids_root="/bids")
    assert "-motsim" not in off

    # The GLM's glob has to match what stage02 named.
    pattern = config.GLM_ORTVEC["motsim"]["pattern"].replace("{task}", "foo")
    assert pattern.startswith("stage02.moco.")
    assert pattern.endswith(".motsim.1D")


def test_motsim_replaces_the_motion_params_only_in_a_default_set():
    """Its PCs come from those very parameters, so a default set carrying both
    spends degrees of freedom twice. Naming both by hand is still honoured."""
    from fastfuncstuff.autoproc.config import DEFAULT_GLM_ORTVEC, apply_motsim_default

    swapped = apply_motsim_default(list(DEFAULT_GLM_ORTVEC))
    assert "motsim" in swapped
    assert "motion" not in swapped and "motion_deriv" not in swapped
    assert "locomoco" in swapped  # unrelated sources survive

    # Idempotent, and a set with no motion params is left alone.
    assert apply_motsim_default(swapped) == swapped
    assert apply_motsim_default(["locomoco"]) == ["locomoco"]
    assert apply_motsim_default([]) == []


def _bids_uneven_runs(tmp_path: Path, phase: bool = False):
    """task 'floc' with runs of 14/13/11 volumes (events include a 'tail' trial at
    23 s, past a 12-volume cut in NO run), plus a 14-volume 'rest' run."""
    import nibabel as nib
    import numpy as np

    func = tmp_path / "sub-ME1" / "ses-SM" / "func"
    func.mkdir(parents=True)
    for task, lens in (("floc", (14, 13, 11)), ("rest", (14,))):
        for i, n in enumerate(lens, 1):
            base = f"sub-ME1_ses-SM_task-{task}_run-0{i}"
            img = nib.Nifti1Image(np.zeros((2, 2, 2, n), dtype=np.float32), np.eye(4))
            img.header.set_zooms((1.0, 1.0, 1.0, 2.0))
            parts = ("mag", "phase") if phase else ("mag",)
            for part in parts:
                nib.save(img, func / f"{base}_part-{part}_bold.nii.gz")
                (func / f"{base}_part-{part}_bold.json").write_text(
                    '{"RepetitionTime": 2.0, "PhaseEncodingDirection": "j-"}'
                )
            if task == "floc":
                (func / f"{base}_events.tsv").write_text(
                    "onset\tduration\ttrial_type\n2\t4\tface\n10\t4\tplace\n23\t2\ttail\n"
                )
    return func


def test_cut_task_vols_trims_only_the_longer_runs_of_that_task(tmp_path: Path):
    """-cut_task_vols floc 12 puts a [0..11] selector on the 14- and 13-volume floc
    runs, leaves the 11-volume run and the other task alone, sizes the design to
    the cut runs, and lets ffs_reml drop events past the new end for that task only."""
    from fastfuncstuff.autoproc.bids import scan_subject
    from fastfuncstuff.autoproc.glm import write_design_specs
    from fastfuncstuff.design.spec import load_spec

    _bids_uneven_runs(tmp_path)
    subj = scan_subject(tmp_path, "ME1")
    opt = Options(run_glm=True, go_to_anat=False, cut_task_vols={"floc": 12})
    plan = build_plan(subj, opt)
    s = write_script(plan, str(tmp_path / "wd"), bids_root=str(tmp_path))

    mags = dict(re.findall(r"^MAG\[SM:(\w+:\d+)\]=(.*)$", s, flags=re.M))
    assert mags["floc:01"].endswith("run-01_part-mag_bold.nii.gz[0..11]'")
    assert mags["floc:02"].endswith("run-02_part-mag_bold.nii.gz[0..11]'")
    assert mags["floc:03"].endswith("run-03_part-mag_bold.nii.gz")
    assert mags["rest:01"].endswith("run-01_part-mag_bold.nii.gz")
    assert "# -cut_task_vols: 14 -> 12 volumes" in s
    assert "# -cut_task_vols: 13 -> 12 volumes" in s

    glm = s[s.index("stage12") :]
    floc_cmd = glm[glm.index("task-floc: model") :]
    floc_cmd = floc_cmd[: floc_cmd.index("-device")]
    assert "-allow_late_events" in floc_cmd
    rest_cmd = glm[glm.index("TODO task-rest") :]
    assert "-allow_late_events" not in rest_cmd[: rest_cmd.index("-device")]

    write_design_specs(plan, str(tmp_path), str(tmp_path / "wd"))
    spec = load_spec(tmp_path / "wd" / "stage11.design.task-floc.toml")
    assert spec.meta.n_timepoints_per_run == [12, 12, 11]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_cut_task_vols_reaches_the_phase_and_the_unwrap_reads_it(tmp_path: Path):
    """The cut rides PHASE[] too (the pair must stay the same length), and without
    NORDIC the unwrap reads the selected pair in place: no copy of the raw data."""
    from fastfuncstuff.autoproc.bids import scan_subject

    _bids_uneven_runs(tmp_path, phase=True)
    subj = scan_subject(tmp_path, "ME1")
    opt = Options(go_to_anat=False, phase_proc=True, cut_task_vols={"floc": 12})
    s = write_script(build_plan(subj, opt), str(tmp_path / "wd"), bids_root=str(tmp_path))

    assert re.search(r"^PHASE\[SM:floc:01\]=.*part-phase_bold\.nii\.gz\[0\.\.11\]'$", s, re.M)
    assert 'mag="${MAG[$k]}"; ph="${PHASE[$k]}"' in s
    assert "stage00.trim." not in s
    assert (
        subprocess.run(  # noqa: S603
            ["bash", "-n", str(_write(tmp_path / "p.sh", s))], capture_output=True, text=True
        ).returncode
        == 0
    )


def test_cut_event_warnings_name_the_dropped_events_and_a_lost_condition(tmp_path: Path):
    from fastfuncstuff.autoproc.bids import find_events, scan_subject
    from fastfuncstuff.autoproc.glm import cut_event_warnings

    _bids_uneven_runs(tmp_path)
    subj = scan_subject(tmp_path, "ME1")
    runs = [
        (r, find_events(r.mag_path, str(tmp_path)))
        for sess in subj.sessions
        for r in sess.bold_runs
        if r.task == "floc"
    ]

    # Cut at 12 vols = 24 s: 'tail' at 23 s survives in the two cut runs; the
    # 11-volume run (22 s) was never cut but loses it too, and says so.
    warns = cut_event_warnings("floc", runs, Options(cut_task_vols={"floc": 12}))
    assert len(warns) == 1
    assert "run-03_events.tsv: 1 event(s) start after the run end" in warns[0]
    assert "run not cut" in warns[0] and "tail x1" in warns[0]

    # Cut at 11 vols = 22 s: 'tail' is gone from every run — an all-zero column.
    warns = cut_event_warnings("floc", runs, Options(cut_task_vols={"floc": 11}))
    assert sum("start after the cut" in w for w in warns) == 2
    assert "condition(s) tail have NO events" in warns[-1]

    assert cut_event_warnings("floc", runs, Options()) == []


def test_cut_task_vols_task_has_no_noise_vols(tmp_path: Path):
    """A cut task was stopped early, so its trailing volumes are task data, not noise
    scans: every run of it -- the uncut 11-volume one too -- gets NOISE=0, while the
    other task keeps -noise_vols for NORDIC, the trims and the design length."""
    from fastfuncstuff.autoproc.bids import scan_subject
    from fastfuncstuff.autoproc.glm import write_design_specs
    from fastfuncstuff.cli.autoproc import main
    from fastfuncstuff.design.spec import load_spec

    _bids_uneven_runs(tmp_path)
    argv = ["-bids_dir", str(tmp_path), "-subject", "ME1", "-no_anat"]
    argv += ["-out", str(tmp_path / "p.sh"), "-work_dir", str(tmp_path / "wd")]
    assert main([*argv, "-cut_task_vols", "task-floc", "12", "-noise_vols", "2"]) == 0
    s = (tmp_path / "p.sh").read_text()
    noise = dict(re.findall(r"^NOISE\[SM:(\w+:\d+)\]=(\d+)$", s, flags=re.M))
    assert noise == {"floc:01": "0", "floc:02": "0", "floc:03": "0", "rest:01": "2"}
    assert "[0..11]" in s
    assert "NOISE_VOLS" not in s  # no global left for a stage to read by mistake

    opt = Options(run_glm=True, go_to_anat=False, noise_vols=2, cut_task_vols={"floc": 12})
    subj = scan_subject(tmp_path, "ME1")
    rows = write_design_specs(build_plan(subj, opt), str(tmp_path), str(tmp_path / "wd2"))
    paths = {task: path for task, path, _ in rows}
    assert load_spec(paths["floc"]).meta.n_timepoints_per_run == [12, 12, 11]


def test_censoring_is_off_by_default_and_wired_through_moco_and_reml_when_on(tmp_path: Path):
    """-censor_motion / -censor_outliers: stage02's ffs_moco writes a keep mask per
    run, stage12 joins each task's in -input order and hands it to ffs_reml
    -censor (the gap-aware route, not spike regressors)."""
    import subprocess

    from fastfuncstuff.autoproc.bids import scan_subject

    _bids_with_events(tmp_path)
    subj = scan_subject(tmp_path, "ME1")
    off = write_script(
        build_plan(subj, Options(run_glm=True)), str(tmp_path / "wd0"), bids_root=str(tmp_path)
    )
    assert "-censor" not in off and "censor.1D" not in off

    plan = build_plan(subj, Options(run_glm=True, censor_motion=0.5, censor_outliers=0.1))
    s = write_script(plan, str(tmp_path / "wd"), bids_root=str(tmp_path))
    manifest = next(line for line in s.splitlines() if "mocobatch" in line and "printf" in line)
    for flag in ("-censor_motion 0.5", "-censor_outliers 0.1", "-outcount", "-enorm", "-censor "):
        assert flag in manifest
    cat = next(line for line in s.splitlines() if line.startswith("cat ") and "censor" in line)
    assert cat.index("run-01") < cat.index("run-02")  # same order as -input
    assert cat.endswith('> "stage12.censor.task-floc.1D"')
    assert '-censor "stage12.censor.task-floc.1D"' in s
    script = tmp_path / "proc.sh"
    script.write_text(s)
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


def _reml_calls(s: str, marker: str) -> list[list[str]]:
    """Every ffs_reml call after ``marker``, as argv (continuations joined)."""
    import shlex

    body = s[s.index(marker) :].replace("\\\n", " ")
    return [
        shlex.split(x.strip())[1:] for x in body.splitlines() if x.strip().startswith("ffs_reml")
    ]


def test_surface_glm_fits_each_target_and_hemisphere_with_its_own_options():
    from fastfuncstuff.cli.reml import create_parser

    s = _skull_script(
        surface_target=["native", "onavg-ico64"], fs_subject="/fs/sub-X", events=["/e/ev.tsv"],
        surface_glm=True, surface_glm_blur_to=6.0, surface_clustsim=True,
        surface_clustsim_niter=3000, glm_blur=4.0, clustsim=True, glm_opts="-polort 2",
        surface_glm_opts="-save_tsnr surf_tsnr",
    )  # fmt: skip
    calls = _reml_calls(s, "stage12s: surface GLM")
    assert len(calls) == 4  # one task x 2 targets x 2 hemispheres
    parser = create_parser()
    for argv in calls:
        a = parser.parse_args(argv)  # every flag stage12s writes is one ffs_reml has
        assert all(f.endswith(".func.gii") and "stage10s.surf." in f for f in a.input)
        assert a.mask.startswith("stage10s.surf.allruns.") and a.mask.endswith(".mask.shape.gii")
        assert a.blur_to_fwhm == 6.0 and a.do_blur is None  # not the voxel -glm_blur 4
        assert a.clustsim and a.clustsim_niter == 3000 and a.save_acf
        assert a.Rbuck.startswith("stage12s.blurto6.stats-reml.task-foo.")
        assert a.polort == 2 and a.save_tsnr == "surf_tsnr"  # shared + surface-only opts
    hemis = {c[c.index("-Rbuck") + 1].split(".")[-3] for c in calls}
    spaces = {c[c.index("-Rbuck") + 1].split(".")[-4] for c in calls}
    assert hemis == {"lh", "rh"} and spaces == {"native", "onavg-ico64"}
    # the voxel fit keeps its own smoothing
    vol = _reml_calls(s.split("stage12s: surface GLM")[0], "stage12: GLM")
    assert vol and all("-do_blur" in c and "-blur_to_fwhm" not in c for c in vol)


def test_glm_blur_to_targets_reach_ffs_reml_with_their_own_tokens_and_maps():
    from fastfuncstuff.autoproc.naming import smooth_tag
    from fastfuncstuff.cli.reml import create_parser

    assert smooth_tag(to=6.0) == "blurto6" and smooth_tag(to=6.0, local=True) == "blurto6local"
    assert smooth_tag(tsnr=80.0) == "tsnr80" and smooth_tag(by=2.5) == "blur2p5"
    assert smooth_tag() == ""
    parser = create_parser()

    s = _skull_script(events=["/e/ev.tsv"], glm_mask="brain", glm_blur_to=6.0,
                      glm_blur_local=True, glm_blur_nbhd=5.0, clustsim=True,
                      clustsim_null="resid")  # fmt: skip
    (argv,) = _reml_calls(s, "stage12: GLM")
    a = parser.parse_args(argv)
    assert a.blur_to_fwhm == 6.0 and a.blur_to_local and a.blur_to_nbhd == 5.0
    assert a.do_blur is None and a.mask and a.clustsim and a.clustsim_null == "resid"
    assert a.Rbuck.startswith("stage12.blurto6local.stats-reml.task-foo.")
    assert a.save_blur_fwhm == "stage12.blurto6local.blurfwhm.task-foo.nii.gz"

    s = _skull_script(events=["/e/ev.tsv"], glm_mask="brain", glm_blur_tsnr=80.0,
                      glm_blur_max_fwhm=6.0, clustsim=True)  # fmt: skip
    (argv,) = _reml_calls(s, "stage12: GLM")
    a = parser.parse_args(argv)
    assert a.blur_to_tsnr == 80.0 and a.blur_tsnr_max_fwhm == 6.0 and not a.blur_to_local
    assert a.clustsim_null is None  # the safe default: ffs_reml's acf
    assert a.Rbuck.startswith("stage12.tsnr80.stats-reml.")

    s = _skull_script(
        surface_target=["native"], fs_subject="/fs/sub-X", events=["/e/ev.tsv"],
        surface_glm=True, surface_glm_blur_tsnr=60.0, surface_glm_blur_nbhd=6.0,
        surface_glm_blur_max_fwhm=7.0, surface_clustsim=True, clustsim_null="resid",
    )  # fmt: skip
    for argv in _reml_calls(s, "stage12s: surface GLM"):
        a = parser.parse_args(argv)
        assert a.blur_to_tsnr == 60.0 and a.blur_to_nbhd == 6.0 and a.blur_tsnr_max_fwhm == 7.0
        assert a.clustsim_null == "resid" and a.save_blur_fwhm.endswith(".shape.gii")
        assert a.Rbuck.startswith("stage12s.tsnr60.stats-reml.")


def test_target_blur_flag_consistency():
    from fastfuncstuff.cli.autoproc import _target_blur_errors

    def errs(**kw):
        base = dict(by=False, to=None, tsnr=None, local=False, max_fwhm=None, nbhd=None)
        return _target_blur_errors("-glm_blur", **{**base, **kw})

    assert errs(to=6.0, local=True, nbhd=5.0) == []
    assert errs(tsnr=80.0, max_fwhm=6.0, nbhd=5.0) == []
    assert any("alternatives" in e for e in errs(by=True, to=6.0))
    assert any("alternatives" in e for e in errs(to=6.0, tsnr=80.0))
    assert any("needs -glm_blur_to" in e for e in errs(local=True, tsnr=80.0))
    assert any("needs -glm_blur_tsnr" in e for e in errs(to=6.0, max_fwhm=6.0))
    assert any("_nbhd is for" in e for e in errs(to=6.0, nbhd=5.0))
    assert any("positive" in e for e in errs(tsnr=-1.0))


@pytest.mark.parametrize("acq_3d", [False, True])
def test_phasedrift_fits_per_volume_for_3d(acq_3d):
    """3-D EPI has no acquired slices, so ffs_phasedrift must fit per volume."""
    subj = Subject("X", [Session("SM", [_phase_run("SM", "primary", "1")])])
    opt = Options(phase_proc=True, acq_3d=acq_3d, go_to_anat=False, distortion=False)
    s = write_script(build_plan(subj, opt), "wd", bids_root="/bids")
    assert ("-by volume" in s) is acq_3d
    assert ("-by slice" in s) is not acq_3d


def test_resolve_acq_3d_evidence_order(capsys):
    from types import SimpleNamespace

    from fastfuncstuff.cli.autoproc import _resolve_acq_3d

    def subj(*runs):
        return SimpleNamespace(sessions=[SimpleNamespace(bold_runs=list(runs))])

    def run(acq=None, st=False):
        return SimpleNamespace(mr_acquisition_type=acq, has_slice_timing=st)

    args = SimpleNamespace(acq="auto", slicetiming=None)
    assert _resolve_acq_3d(args, subj(run("3D", st=True))) is True  # explicit wins
    assert _resolve_acq_3d(args, subj(run("2D"))) is False
    assert _resolve_acq_3d(args, subj(run(st=True), run())) is False  # any SliceTiming
    assert _resolve_acq_3d(args, subj(run(), run())) is True  # no slice info at all
    assert "assuming 3D" in capsys.readouterr().err
    assert _resolve_acq_3d(SimpleNamespace(acq="auto", slicetiming="st.txt"), subj(run())) is False
    assert _resolve_acq_3d(SimpleNamespace(acq="2d", slicetiming=None), subj(run("3D"))) is False


def test_invocation_note_requotes_values_with_spaces():
    """Bug of record: -anat_opts "-rigid -cost lpa" came back unquoted, split
    over lines as if each word were a flag, so the regenerate line broke."""
    import shlex

    from fastfuncstuff.autoproc.emit import _invocation_note

    argv = ["ffs_autoproc", "-subject", "03", "-anat_opts", "-rigid -cost lpa -fast", "-exec"]
    note = _invocation_note(shlex.join(argv))
    cmd = " ".join(ln.lstrip("#").strip().rstrip("\\").strip() for ln in note.splitlines()[2:])
    assert shlex.split(cmd) == argv


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_stage04_writes_a_before_after_distortion_stack(tmp_path):
    """The corrected image alone cannot show a wrong-sign field (ds003427's made
    the distortion worse); the uncorrected image beside it on the same grid can."""
    fmap = _b0_fmap(intended=[("foo", "1")])
    subj = Subject("X", [Session("01", [_b0_run("01", "foo", "1")], [fmap])])
    s = write_script(build_plan(subj, Options()), "wd", bids_root="/bids")
    st4 = _stage04(s)
    assert "stage04.identity.aff12.1D" in st4
    assert 'qc_tcat "stage04.QC.blip.ses-01.fmap-b0.nii.gz" "uncorrected corrected"' in st4
    assert '"stage04.blip.ses-01.fmap-b0_mean.nii$FMT"' in st4
    assert (
        subprocess.run(  # noqa: S603
            ["bash", "-n", str(_write(tmp_path / "p.sh", s))], capture_output=True
        ).returncode
        == 0
    )


def test_stage04_pepolar_qc_is_the_fitted_pair_only():
    """A before/after stack for a reverse-PE pair duplicated volumes: its
    "uncorrected" is _padphase[0], its "corrected" is _unwarped[0]."""
    fmap = FmapGroup("01", "f", Path("/rev.nii.gz"), {"TotalReadoutTime": 0.06}, [("foo", "1")])
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")], [fmap])])
    st4 = _stage04(write_script(build_plan(subj, Options()), "wd", bids_root="/bids"))
    assert 'qc_tcat "stage04.QC.blip.ses-01.fmap-f_inputs.nii.gz" "blip_up blip_down"' in st4
    assert "stage04.QC.blip.ses-01.fmap-f.nii.gz" not in st4
    assert "identity.aff12" not in st4 and "_before" not in st4


def test_anat_nonlin_register_refines_along_pe_in_the_anchor_frame():
    """-anat_nonlin_method register: a thin-slab GRE anat has no tissue contrast a
    template can drive, so the anchor is registered to the anat instead. It is
    solved in the anchor's (EPI) frame via -matrix, so the PE axis is a grid axis
    and the other two are frozen exactly; the warp takes the existing anat_nl slot,
    acting after anat_lin, and no TPM is needed."""
    s = _skull_script(anat_skull=True, anat_nonlin=True, anat_nonlin_method="register")
    st9 = s[s.index("stage09: anatomical alignment") :]
    assert "ffs_segment \\" not in st9
    assert '-matrix "stage09.anat.aff12.1D"' in st9
    assert "-noXdis" in st9 and "-noZdis" in st9 and "-noYdis" not in st9
    chain = next(line for line in s.splitlines() if line.startswith("CHAIN["))
    links = chain.split('="')[1].rstrip('"').split()
    assert links[:2] == ["stage09.anat.aff12.1D", "stage09.nlanat_WARP.nii$FMT"]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_phasereg_stage_writes_into_its_folder_with_vein_mask_and_glm(tmp_path: Path):
    """-phasereg: post-preprocessing, so its outputs live in phasereg/ (no new
    stageNN files): the vein mask from the anat tissue fractions, the corrected
    runs with -spr confined to that mask, a QC stack, and the same GLM spec fit on
    the corrected runs. Without tissue maps there is no mask and no -spr."""
    from fastfuncstuff.autoproc.bids import scan_subject

    func = _bids_with_events(tmp_path)
    for mag in func.glob("*_part-mag_bold.nii.gz"):
        (func / mag.name.replace("part-mag", "part-phase")).write_bytes(mag.read_bytes())
    subj = scan_subject(tmp_path, "ME1")
    assert all(r.phase_path for s in subj.sessions for r in s.bold_runs)
    tissue = ["/t/c1.nii.gz", "/t/c2.nii.gz", "/t/c3.nii.gz"]
    opt = Options(
        run_glm=True, phase_proc=True, phasereg=True, tissue=tissue, phasereg_veins="auto"
    )
    s = write_script(build_plan(subj, opt), str(tmp_path / "wd"), bids_root=str(tmp_path))
    assert (
        subprocess.run(  # noqa: S603
            ["bash", "-n", str(_write(tmp_path / "p.sh", s))], capture_output=True
        ).returncode
        == 0
    )

    stage = s.split("phasereg: phase regression")[1].split("stage12: GLM")[0]
    assert "select_phasereg_dir" in stage and "phasereg_[0-9][0-9]" in stage
    assert "ffs_veinmask" in stage and "-tissue /t/c1.nii.gz /t/c2.nii.gz /t/c3.nii.gz" in stage
    assert '-prefix "$PHASEREG_DIR/veinmask"' in stage
    assert '-prefix "$PHASEREG_DIR/task-floc"' in stage and "-keep_drift" in stage
    assert '-spr_mask "$PHASEREG_DIR/veinmask_mask.nii$FINAL_FMT"' in stage
    assert "-verb 1" in stage and "phasereg: sPR inside the local vein mask" in stage
    # donors on task-residual coupling, with every run's events
    assert "-spr_select_on residual -task_removal fit:2" in stage and "-events " in stage
    assert "$PHASEREG_DIR/QC.veinmask.nii.gz" in stage
    import re

    written = re.findall(r'-prefix "([^"]+)"', stage) + re.findall(r'qc_tcat "([^"]+)"', stage)
    assert written and all(w.startswith("$PHASEREG_DIR/") for w in written)  # no new stageNN files
    glm = s.split("stage12: GLM")[1]
    assert '-input "$PHASEREG_DIR/task-floc_corrected_run-01.nii$FINAL_FMT"' in glm
    assert '-Rbuck "$PHASEREG_DIR/stats-reml.task-floc.nii$GLM_FMT"' in glm
    assert '-Rbuck "stage12.stats-reml.task-floc.nii$GLM_FMT"' in glm  # stage10 fit kept

    plain = write_script(
        build_plan(subj, Options(run_glm=True, phase_proc=True, phasereg=True)),
        str(tmp_path / "wd2"),
        bids_root=str(tmp_path),
    )
    assert "ffs_phasereg" in plain and "ffs_veinmask" not in plain
    assert not re.search(r"^\s+-spr", plain, re.M)  # the flag, not the comment saying so


def test_phasereg_vein_mask_takes_presurfs_classes_and_keeps_them_written():
    """With -presurf and no -tissue, the vein mask reads presurfer's UNI classes:
    presurf must then drop its default -no_classes, and rerun if an earlier run
    skipped them."""
    subj = Subject(
        "X",
        [Session("01", [_phase_run("01", "foo", "1")], anat=Path("/anat/uni.nii.gz"))],
    )
    opt = Options(
        go_to_anat=True,
        phase_proc=True,
        phasereg=True,
        phasereg_veins="auto",
        presurf=[("/a/uni.nii.gz", "/a/inv2.nii.gz", None)],
        presurf_tpm="/t/TPM.nii",
    )
    s = write_script(build_plan(subj, opt), "wd", bids_root="/bids")
    block = s.split("presurfer) ---")[1].split("\nfi")[0]
    assert "-no_classes" not in block
    assert '[ ! -f "presurf_work/anat1_uni_c1.nii.gz" ]' in block
    assert "-tissue presurf_work/anat1_uni_c1.nii.gz" in s
    # Without -phasereg nothing reads the classes: presurf keeps -no_classes.
    opt.phasereg = False
    assert "-no_classes" in write_script(build_plan(subj, opt), "wd", bids_root="/bids")


def test_phasereg_vein_mask_takes_the_suma_aseg_when_the_anat_is_sumas():
    """MindsEye bug of record: with -suma the anat is FreeSurfer's brain, presurfer
    is skipped, and -phasereg fell back to plain PR with no word in the log. The aseg
    labels sit on that anat, so they are the tissue maps (sulcal CSF included)."""
    subj = Subject("X", [Session("01", [_phase_run("01", "foo", "1")], anat=None)])
    opt = Options(
        go_to_anat=True,
        phase_proc=True,
        phasereg=True,
        phasereg_veins="auto",
        suma_dir="/fs/SUMA",
        anat_path="/fs/SUMA/brain.nii.gz",
    )
    s = write_script(build_plan(subj, opt), "wd", bids_root="/bids")
    stage = s.split("phasereg: phase regression")[1]
    assert "/fs/SUMA/aseg.auto.nii.gz'<4,14,43,24>'" in stage  # CSF incl. label 24
    assert '-prefix "phasereg_tissue/tissue_aseg_gm.nii.gz"' in stage
    assert "-tissue phasereg_tissue/tissue_aseg_gm.nii.gz" in stage and "-spr_mask" in stage
    # A user -anat that is not SUMA's: the aseg does not sit on it.
    opt.anat_path = "/elsewhere/t1.nii.gz"
    assert "tissue_aseg" not in write_script(build_plan(subj, opt), "wd", bids_root="/bids")


def test_phasereg_default_is_plain_pr_even_with_tissue():
    """The vein-masked sPR default rested on tissue maps 2-7 mm off the EPI and on a
    padded mask; rescored, it removed tissue response. Until it is re-earned,
    -phasereg means plain phase regression and the vein mask is opt-in."""
    subj = Subject("X", [Session("01", [_phase_run("01", "foo", "1")], anat=None)])
    opt = Options(go_to_anat=True, phase_proc=True, phasereg=True, tissue=["/t/c.nii.gz"])
    s = write_script(build_plan(subj, opt), "wd", bids_root="/bids")
    stage = s.split("phasereg: phase regression")[1]
    assert (
        "ffs_veinmask" not in stage
        and "-spr" not in stage.split("ffs_phasereg")[1].split("echo")[0]
    )
    assert "-phasereg_veins off" in stage


def test_phasereg_folders_are_numbered_per_setting_and_reused_on_rerun(tmp_path: Path):
    """Each phase-regression setting gets its own phasereg_NN with a settings.txt saying
    what was run: an identical rerun resumes in its folder, other -phasereg_* options
    make the next number. Runs the generated bash block itself."""
    from fastfuncstuff.autoproc.emit import _phasereg_dir_block

    def run(settings: str) -> str:
        block = "\n".join(_phasereg_dir_block(settings)) + '\necho "DIR=$PHASEREG_DIR"\n'
        r = subprocess.run(  # noqa: S603
            ["bash", "-c", "ffs_phasereg() { :; }\n" + block],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        return r.stdout.strip().splitlines()[-1].removeprefix("DIR=")

    a = "ffs_autoproc phase regression settings\n-phasereg_veins off\n\nffs_phasereg \\\n  -keep_drift"
    b = a.replace("off", "auto")
    assert run(a) == "phasereg_01"
    assert run(a) == "phasereg_01"  # same settings: resume, no new folder
    assert run(b) == "phasereg_02"
    assert run(a) == "phasereg_01"
    assert (tmp_path / "phasereg_02" / "settings.txt").read_text().strip() == b
    assert len((tmp_path / "phasereg_01" / "history.txt").read_text().splitlines()) == 3


def _me_run(session, task, run, n=3, phase=True):
    r = _run(session, task, run)
    base = str(r.mag_path).replace("_bold.nii.gz", "")
    r.echo_mags = [Path(f"{base}_echo-{e}_part-mag_bold.nii.gz") for e in range(1, n + 1)]
    r.echo_phases = [
        Path(f"{base}_echo-{e}_part-phase_bold.nii.gz") if phase else None for e in range(1, n + 1)
    ]
    r.echo_tes = [7.6, 21.7, 35.8, 49.9][:n]
    r.mag_path, r.phase_path = r.echo_mags[0], r.echo_phases[0]
    return r


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("acq_3d", [True, False])
def test_multiecho_estimates_from_echo1_warps_each_echo_and_combines(tmp_path, acq_3d):
    """Multi-echo: echo 1 drives the estimates; 3-D adds the per-echo partition
    correction (phase tables -> moco -me_3depi) and a joint per-echo locomoco; every
    echo is warped with its own chain at stage10 and optimally combined into the
    run's own stage10.final name."""
    subj = Subject("X", [Session("01", [_me_run("01", "foo", "1")])])
    opt = Options(
        phase_proc=True, locomoco=True, acq_3d=acq_3d, go_to_anat=False, distortion=False,
        slicetiming_method="none" if acq_3d else "integrate",
    )  # fmt: skip
    s = write_script(build_plan(subj, opt), "wd", bids_root="/bids")
    assert (
        subprocess.run(  # noqa: S603
            ["bash", "-n", str(_write(tmp_path / "p.sh", s))], capture_output=True, text=True
        ).returncode
        == 0
    )
    assert "NECHO[01:foo:1]=3" in s
    assert "TES[01:foo:1]='7.6 21.7 35.8'" in s
    # one echo/phase naming rule in every stage: <stem>.echo-N[.part-phase], echo 1
    # labeled too (E1), never an eN_ prefix, an _eN infix or a _phase suffix
    assert "E1[01:foo:1]=.echo-1" in s
    s_nordic = write_script(
        build_plan(subj, Options(phase_proc=True, want_nordic=True, acq_3d=acq_3d)), "wd"
    )
    assert "stage00.nordic.${FRAG[$k]}${E1[$k]:-}_numcomps.nii$FMT" in s_nordic
    old = r"\be(?:[0-9]+|\$\{e\}|\$e)_|_e(?:[0-9]+|\$\{e\}|\$e)[_.]|phase_e|_phase\.nii"
    assert not re.search(old, s), re.search(old, s)
    assert "MAGE['01:foo:1|3']=" in s and "PHASEE['01:foo:1|3']=" in s
    # one unwrap over every echo
    assert 'ffs_phase_unwrap -phase "${phs[@]}" -magnitude "${mags[@]}" -echo_times ${TES[$k]}' in s
    # every echo warped with ITS chain, then combined into the run's name
    assert '-nwarp \\"${ECHAIN[$k|$e]}\\"' in s
    assert 'ffs_t2smap -input "${ins[@]}" -tes ${TES[$k]}' in s
    assert '-optcom "$outf"' in s
    if acq_3d:
        assert "-save_partition_shift" in s and "-save_partition_stretch" in s
        assert "-reg_echo 1 -me_3depi -axis k" in s and "-1Dmatrix_shiftcorr" in s
        assert "-shift_table" in s and "-stretch_table" in s
        # phase into locomoco is an opt-in experiment: not passed by default
        assert (
            'lm_me="-me_3depi -echo_times ${TES[$k]} -me_fixed_scaling -me_shared_warp"' in s
            and "-phase$lm_ph" not in s
        )
        assert "stage02.moco.ses-01.task-foo.run-1.echo-3.aff12.1D" in s  # echo 3's matrices
        # one shared locomoco warp; echo 3 scales its partition displacement by TE3/TE1
        assert "VSCALE:1,1,4.71053:stage03.nlmoco.ses-01.task-foo.run-1_warp" in s
        assert "nlmoco.ses-01.task-foo.run-1.echo-" not in s
        assert 'ln -sf "${mstem}.echo-1_${which}.nii$FMT"' in s
    else:
        assert "-me_3depi" not in s and "-save_partition_shift" not in s
        # 2-D: one in-plane warp moves every echo alike -> echo chains share it
        ech = [ln for ln in s.splitlines() if ln.startswith("ECHAIN[")]
        assert len(ech) == 3 and len({ln.split("=", 1)[1] for ln in ech}) == 1


def test_multiecho_refuses_what_it_does_not_handle_yet():
    from fastfuncstuff.cli.autoproc import _multiecho_errors

    subj = Subject("X", [Session("01", [_me_run("01", "foo", "1")])])
    assert _multiecho_errors(Options(), subj) == []
    # slice timing first, -phasereg, -surface_target, 3-D -motsim: all wired 2026-10-18
    assert _multiecho_errors(Options(slicetiming_method="first", phasereg=True), subj) == []
    # ...but phasereg pools a task's runs echo by echo, so they must share the echoes
    two = _me_run("01", "foo", "2", n=2)
    mixed = Subject("X", [Session("01", [_me_run("01", "foo", "1"), two])])
    assert any("-phasereg" in e for e in _multiecho_errors(Options(phasereg=True), mixed))
    bad = _me_run("01", "foo", "2")
    bad.echo_tes = []
    assert any(
        "EchoTime" in e for e in _multiecho_errors(Options(), Subject("X", [Session("01", [bad])]))
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("acq_3d", [True, False])
def test_medic_reuses_the_unwrap_field_and_sits_rightmost_in_every_chain(tmp_path, acq_3d):
    """-medic: ffs_medic starts from the unwrap stage's field map (and mask) rather than
    a second estimate; moco reads its undistorted echoes (and, for 3-D locomoco, the
    phase undistorted with them); the time-varying warp is the rightmost link of every
    echo's chain, so stage10 still resamples once from the raw series. 3-D locomoco
    then solves the partition axis only, and the SBRef is not the moco base."""
    r = _me_run("01", "foo", "1")
    r.json["TotalReadoutTime"] = 0.0312
    r.sbref_path = Path("/bids/sub-X/ses-01/func/sub-X_ses-01_task-foo_run-1_sbref.nii.gz")
    subj = Subject("X", [Session("01", [r])])
    opt = Options(
        phase_proc=True, medic=True, locomoco=True, acq_3d=acq_3d, go_to_anat=False,
        slicetiming_method="none" if acq_3d else "integrate",
    )  # fmt: skip
    plan = build_plan(subj, opt)
    assert plan.runs[0].warp_chain[-2:] == ["moco", "medic"]
    # moco registered the undistorted echoes, so the lane images are already undistorted:
    # the grandmean chain must not apply MEDIC's warp to them again (it once did)
    from fastfuncstuff.autoproc.emit import _grandmean_tokens, _needs_gmrun

    assert "medic" not in _grandmean_tokens(plan.runs[0])
    assert not _needs_gmrun(plan.runs[0])  # one run: its runmean already is the image
    assert not plan.use_sbref
    s = write_script(plan, "wd", bids_root="/bids")
    assert (
        subprocess.run(  # noqa: S603
            ["bash", "-n", str(_write(tmp_path / "p.sh", s))], capture_output=True, text=True
        ).returncode
        == 0
    )
    assert "TRT[01:foo:1]=0.0312" in s and "\nSBREF[" not in s
    assert '-save_mask "stage00.unwrap.${FRAG[$k]}_mask.nii$FMT"' in s
    assert '-fieldmap "${uwe}_fieldmap.nii$FMT" -fieldmap_mask "${uwe}_mask.nii$FMT"' in s
    assert s.index("stage01: MEDIC") < s.index("stage02: moco")
    assert 'raw="stage01.medic.${FRAG[$k]}.echo-1_undist.nii$FMT"' in s
    medic_warp = "stage01.medic.ses-01.task-foo.run-1_warp.nii$FMT"
    ech = [ln for ln in s.splitlines() if ln.startswith("ECHAIN[")]
    assert len(ech) == 3 and all(ln.rstrip("'\"").endswith(medic_warp) for ln in ech)
    if acq_3d:
        assert 'mag="stage01.medic.${FRAG[$k]}.echo-${e}_undist.nii$FMT"' in s
        assert "-apply_phase" not in s and "part-phase_undist" not in s
        assert 'lm_pe="-pe_dir2 k"' in s
    else:
        assert "-apply_phase" not in s and 'lm_pe="-pe_dir2 k"' not in s


def test_medic_refuses_what_it_cannot_build():
    from fastfuncstuff.cli.autoproc import _medic_errors

    r = _me_run("01", "foo", "1")
    subj = Subject("X", [Session("01", [r])])
    assert any("-phase_proc" in e for e in _medic_errors(Options(medic=True), subj))
    errs = _medic_errors(Options(medic=True, phase_proc=True), subj)
    assert any("TotalReadoutTime" in e for e in errs)  # not guessed
    r.json["TotalReadoutTime"] = 0.03
    assert _medic_errors(Options(medic=True, phase_proc=True), subj) == []
    se = _run("01", "foo", "2")
    errs = _medic_errors(Options(medic=True, phase_proc=True), Subject("X", [Session("01", [se])]))
    assert any("not multi-echo" in e for e in errs)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
@pytest.mark.parametrize("medic", [False, True])
def test_multiecho_slice_timing_first_shifts_every_echo_and_its_phase(tmp_path, medic):
    """-slicetiming_method first with multi-echo: every echo and its unwrapped phase get
    the shift; MEDIC (after it) undistorts the shifted echoes; stage10 resamples the
    shifted echo e, with its shifted phase, and integrates no timing."""
    r = _me_run("01", "foo", "1")
    r.json["TotalReadoutTime"] = 0.03
    r.json["SliceTiming"] = [0.0, 0.5, 1.0, 1.5]
    opt = Options(
        phase_proc=True, medic=medic, acq_3d=False, go_to_anat=False, distortion=False,
        slicetiming_method="first",
    )  # fmt: skip
    s = write_script(build_plan(Subject("X", [Session("01", [r])]), opt), "wd", bids_root="/bids")
    assert (
        subprocess.run(  # noqa: S603
            ["bash", "-n", str(_write(tmp_path / "p.sh", s))], capture_output=True, text=True
        ).returncode
        == 0
    )
    shifted = 'mag="stage01.tshift.${FRAG[$k]}${esfx}.nii$FMT"'
    assert 'eoutf="stage01.tshift.${FRAG[$k]}${esfx}.nii$FMT"' in s
    assert '-input "stage00.unwrap.${FRAG[$k]}.echo-${e}.part-phase.nii$FMT"' in s
    final = s[s.index("stage10.final.${FRAG[$k]}.echo-$e") - 400 :]
    assert shifted in final
    assert '-phase \\"stage01.tshift.${FRAG[$k]}${esfx}.part-phase.nii$FMT\\"' in final
    if medic:
        assert s.index("stage01: slice timing") < s.index("stage01: MEDIC")
        medic_stage = s[s.index("stage01: MEDIC") : s.index("stage02: moco")]
        assert shifted in medic_stage


def test_multiecho_3d_motsim_rides_the_echo1_registration():
    """-motsim with 3-D multi-echo: ffs_moco simulates from echo 1's rigid motion (the
    TE-scaled partition shift is corrected in the data, not modelled as motion)."""
    from fastfuncstuff.cli.autoproc import _multiecho_errors

    r = _me_run("01", "foo", "1")
    subj = Subject("X", [Session("01", [r])])
    opt = Options(phase_proc=True, acq_3d=True, motsim="both,12", go_to_anat=False,
                  distortion=False, slicetiming_method="none")  # fmt: skip
    assert _multiecho_errors(opt, subj) == []
    s = write_script(build_plan(subj, opt), "wd", bids_root="/bids")
    line = next(ln for ln in s.splitlines() if "-1Dmatrix_save" in ln and "mocobatch" in ln)
    assert "-motsim both,12" in line and "$me_args" in line


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_multiecho_surface_reads_every_echo_and_combines_per_vertex(tmp_path):
    """-surface_target with multi-echo: each echo's stage10 job reads the surface
    through that echo's chain (still one interpolation); ffs_t2smap combines per
    vertex into the run's own surface name, which the masks and the GLM read."""
    r = _me_run("01", "foo", "1")
    subj = Subject("X", [Session("01", [r])])
    opt = Options(go_to_anat=True, anat_path="/data/sub-X_T1w.nii.gz", acq_3d=True,
                  phase_proc=True, locomoco=True, distortion=False, slicetiming_method="none",
                  surface_target=["native"], fs_subject="/fs/sub-X")  # fmt: skip
    from fastfuncstuff.cli.autoproc import _multiecho_errors

    assert _multiecho_errors(opt, subj) == []
    s = write_script(build_plan(subj, opt), "wd", bids_root="/bids")
    assert (
        subprocess.run(  # noqa: S603
            ["bash", "-n", str(_write(tmp_path / "p.sh", s))], capture_output=True, text=True
        ).returncode
        == 0
    )
    surf = {ln.split("=", 1)[0]: ln for ln in s.splitlines() if ln.startswith("SURFCHAIN[")}
    assert "SURFCHAIN['01:foo:1|3']" in surf
    assert "VSCALE:1,1,4.71053:stage03.nlmoco" in surf["SURFCHAIN['01:foo:1|3']"]  # echo 3
    assert '-surf_prefix \\"stage10s.surf.${FRAG[$k]}.echo-$e\\"' in s
    assert '-surf_nwarp \\"${SURFCHAIN[$k|$e]}\\"' in s
    assert "$pe_args$sa" in s
    # every echo shares one geometry copy; only echo 1 writes the QC
    sa = next(ln for ln in s.splitlines() if ln.strip().startswith('sa="'))
    lean, full = sa.split('; [ "$e" -eq 1 ] && sa=')
    assert "-surf_geom_prefix stage10s.surf" in lean and "-surf_qc" not in lean
    assert "-surf_geom_prefix stage10s.surf" in full
    assert (
        '-optcom "stage10s.surf.${FRAG[$k]}.native.lh.func.gii"' in s
        and '-optcom "stage10s.surf.${FRAG[$k]}.native.rh.func.gii"' in s
    )
    assert s.index("stage10s.t2s.") < s.index("== stage10s: surface masks")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_multiecho_phasereg_corrects_each_echo_then_combines(tmp_path):
    """-phasereg with multi-echo: each echo's magnitude against its own phase, pooled
    over the task's runs; then the corrected echoes are optimally combined into the
    names the GLM's phasereg variant reads."""
    runs = [_me_run("01", "foo", "1"), _me_run("01", "foo", "2")]
    opt = Options(phase_proc=True, phasereg=True, acq_3d=False, go_to_anat=False,
                  distortion=False, slicetiming_method="integrate")  # fmt: skip
    s = write_script(build_plan(Subject("X", [Session("01", runs)]), opt), "wd", bids_root="/bids")
    assert (
        subprocess.run(  # noqa: S603
            ["bash", "-n", str(_write(tmp_path / "p.sh", s))], capture_output=True, text=True
        ).returncode
        == 0
    )
    stage = s[s.index("== phasereg ==") :]
    for e in (1, 2, 3):
        assert f'-prefix "$PHASEREG_DIR/task-foo.echo-{e}"' in stage
        assert f"stage10.final.ses-01.task-foo.run-2.echo-{e}.part-phase.nii$FINAL_FMT" in stage
    assert stage.count("ffs_phasereg") >= 3
    assert "ffs_t2smap -input" in stage and "-tes 7.6 21.7 35.8" in stage
    assert "task-foo_corrected_run-02.nii$FINAL_FMT" in stage


@pytest.mark.parametrize("medic", [False, True])
def test_locomoco_phase_is_an_explicit_experimental_opt_in(medic):
    """-locomoco_phase hands ffs_locomoco every echo's unwrapped phase (and, under
    -medic, has the phase undistorted with its magnitude); without it none of that is
    emitted, and moco does not carry a phase it has no consumer for."""
    r = _me_run("01", "foo", "1")
    r.json["TotalReadoutTime"] = 0.0312
    subj = Subject("X", [Session("01", [r])])
    kw = dict(phase_proc=True, locomoco=True, acq_3d=True, go_to_anat=False, medic=medic,
              slicetiming_method="none")  # fmt: skip
    on = write_script(build_plan(subj, Options(locomoco_phase=True, **kw)), "wd", bids_root="/bids")
    off = write_script(build_plan(subj, Options(**kw)), "wd", bids_root="/bids")
    assert "-phase$lm_ph -phase_units rad" in on and "-phase$phs -phase_units rad" in on
    assert "-phase$lm_ph" not in off and "-phase$phs" not in off
    assert ("-apply_phase -phase_is_unwrapped" in on) is medic
    assert "-apply_phase" not in off
    # resume: MEDIC's skip waits for the phase it now has to write, not just the warp
    phase_done = '[ -f "${ms}.echo-${NECHO[$k]}.part-phase_undist.nii$FMT" ]'
    assert (phase_done in on) is medic and phase_done not in off


def test_locomoco_phase_refuses_what_it_cannot_use():
    from fastfuncstuff.cli.autoproc import _locomoco_phase_errors

    r = _me_run("01", "foo", "1")
    subj = Subject("X", [Session("01", [r])])
    ok = dict(locomoco=True, phase_proc=True, acq_3d=True, locomoco_phase=True)
    assert _locomoco_phase_errors(Options(**ok), subj) == []
    assert _locomoco_phase_errors(Options(locomoco_phase=False), subj) == []
    for drop, text in (
        ("locomoco", "-locomoco"),
        ("phase_proc", "-phase_proc"),
        ("acq_3d", "acq_3d"),
    ):
        errs = _locomoco_phase_errors(Options(**{**ok, drop: False}), subj)
        assert errs and text in errs[0]
    single = Subject("X", [Session("01", [_run("01", "foo", "1")])])
    assert "not multi-echo" in _locomoco_phase_errors(Options(**ok), single)[0]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_phasereg_me_field_regresses_the_combination_on_one_fitted_field(tmp_path):
    """-phasereg_me field: stage10's combine also writes the frequency-shift series fitted
    from every echo's phase, and phasereg runs ONCE per task on the combined echoes
    against it (Hz, so no temporal unwrap) instead of once per echo."""
    from fastfuncstuff.cli.autoproc import _phasereg_me_errors

    runs = [_me_run("01", "foo", "1"), _me_run("01", "foo", "2")]
    kw = dict(phase_proc=True, phasereg=True, acq_3d=False, go_to_anat=False,
              distortion=False, slicetiming_method="integrate")  # fmt: skip
    subj = Subject("X", [Session("01", runs)])
    on = write_script(build_plan(subj, Options(phasereg_me="field", **kw)), "wd", bids_root="/bids")
    off = write_script(build_plan(subj, Options(**kw)), "wd", bids_root="/bids")
    assert (
        subprocess.run(  # noqa: S603
            ["bash", "-n", str(_write(tmp_path / "p.sh", on))], capture_output=True, text=True
        ).returncode
        == 0
    )
    # the combine writes the field, from every echo's final phase
    assert 'fieldf="stage10.field.${FRAG[$k]}.nii$FINAL_FMT"' in on
    assert '-phase "${phs[@]}" -field_hz "$fieldf"' in on
    assert 'phs+=("stage10.final.${FRAG[$k]}.echo-$e.part-phase.nii$FINAL_FMT")' in on
    # one phasereg call per task, on the combination and the field, no per-echo calls
    stage = on[on.index("== phasereg ==") :]
    assert stage.count("ffs_phasereg") >= 1 and "task-foo.echo-" not in stage
    assert '-magnitude "stage10.final.ses-01.task-foo.run-1.nii$FINAL_FMT"' in stage
    assert '-phase "stage10.field.ses-01.task-foo.run-1.nii$FINAL_FMT"' in stage
    assert "-no_phase_unwrap" in stage
    assert "task-foo_corrected_run-02.nii$FINAL_FMT" in stage
    # off: nothing of it
    assert "-field_hz" not in off and "stage10.field" not in off
    assert _phasereg_me_errors(Options(phasereg_me="field", phasereg=True, phase_proc=True)) == []
    assert "-phasereg" in _phasereg_me_errors(Options(phasereg_me="field"))[0]
    assert "-phase_proc" in _phasereg_me_errors(Options(phasereg_me="field", phasereg=True))[0]
    assert _phasereg_me_errors(Options()) == []


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_me_combine_resumes_surfaces_on_their_own_and_fits_the_field_only_with_phase(tmp_path):
    """Run the emitted stage10 combine with a stub ffs_t2smap. A run whose volume is done
    but whose surfaces are not still combines its surfaces (the volume skip used to be a
    `continue`), and -phasereg_me field passes -phase only for a run that has phase (a
    run without it was handed phase files that do not exist)."""
    from fastfuncstuff.autoproc.emit import _final_me_combine

    runs = [_me_run("01", "foo", "1"), _me_run("01", "foo", "2")]
    opt = Options(phase_proc=True, phasereg=True, phasereg_me="field", acq_3d=False,
                  go_to_anat=False, distortion=False, surface_target=["fsnative"])  # fmt: skip
    block = _final_me_combine(build_plan(Subject("X", [Session("01", runs)]), opt))
    (tmp_path / "stage10.final.A.nii").write_text("")  # run A: volume combined already
    (tmp_path / "stage10.field.A.nii").write_text("")
    script = (
        "set -euo pipefail\n"
        'ffs_t2smap() { echo "T2SMAP $*" >> calls.log; }\n'
        "declare -A NECHO FRAG TES PHASE\n"
        "RUN_KEYS=(a b); NECHO[a]=3; NECHO[b]=3; FRAG[a]=A; FRAG[b]=B\n"
        "TES[a]='7.6 21.7 35.8'; TES[b]='7.6 21.7 35.8'; PHASE[a]=/p.nii\n"
        "skip_final=1; FINAL_FMT=''; DEVICE=cpu\n" + block
    )
    (tmp_path / "s.sh").write_text(script)
    r = subprocess.run(["bash", "s.sh"], cwd=tmp_path, capture_output=True, text=True)  # noqa: S603, S607
    assert r.returncode == 0, r.stderr
    calls = (tmp_path / "calls.log").read_text().splitlines()
    vol = [c for c in calls if "-optcom stage10.final." in c]
    surf = [c for c in calls if "stage10s.surf." in c and "-optcom" in c]
    assert len(vol) == 1 and "stage10.final.B.nii" in vol[0]  # A skipped, B combined
    assert "-phase" not in vol[0] and "-field_hz" not in vol[0]  # B has no phase
    assert any("stage10s.surf.A." in c for c in surf)  # A's surfaces still combined
    assert any("stage10s.surf.B." in c for c in surf)


@pytest.mark.parametrize("acq_3d", [True, False])
def test_medic_owns_inplane_unless_asked_to_keep_locomocos(acq_3d):
    """-medic corrects in-plane frame by frame: 3-D locomoco keeps only the partition
    shift, 2-D (no partition axis) skips locomoco -- chain, stage and lanes alike.
    -medic_inplane_locomoco keeps the in-plane solve on top in both."""
    r = _me_run("01", "foo", "1")
    r.json["TotalReadoutTime"] = 0.0312
    subj = Subject("X", [Session("01", [r])])
    kw = dict(phase_proc=True, medic=True, locomoco=True, acq_3d=acq_3d, go_to_anat=False,
              slicetiming_method="none" if acq_3d else "integrate")  # fmt: skip
    plan = build_plan(subj, Options(**kw))
    s = write_script(plan, "wd", bids_root="/bids")
    keep = build_plan(subj, Options(medic_inplane_locomoco=True, **kw))
    k = write_script(keep, "wd", bids_root="/bids")
    if acq_3d:
        assert "locomoco" in plan.runs[0].warp_chain
        assert 'lm_pe="-pe_dir2 k"' in s and 'lm_pe="-pe_dir2 k"' not in k
    else:
        assert "locomoco" not in plan.runs[0].warp_chain and not plan.options.locomoco
        assert "== stage03: locomoco ==" not in s
        assert "locomoco" in keep.runs[0].warp_chain and "== stage03: locomoco ==" in k


def test_nlanat_segment_is_told_which_grid_the_anat_matrix_was_fitted_on():
    """[[project_oblique_frame_slip]]: the anat aff12's -base is the AUTOBOX, while the
    subject TPM sits on the uncropped anat. An .aff12.1D is read in its base grid's
    cardinal mm frame, which keeps that grid's own origin, so on an oblique anat the
    two frames differ by a fixed shift (6.59 mm on the 5.04 deg subject that caught
    this) and ffs_segment starts the fit that far off, silently.
    """
    subj = Subject("X", [Session("01", [_run("01", "foo", "1")])])

    # Own anat: the matrix's base is the local autobox.
    s = write_script(
        build_plan(subj, Options(anat_nonlin=True, tpm="/tpm.nii", anat_path="/anat.nii.gz")),
        "wd",
        bids_root="/bids",
    )
    assert "-1Dmatrix stage09.anat.aff12.1D" in s
    assert '-matrix_base "stage09.anat_autobox.nii.gz"' in s
    # and that really is what the allineate used as -base
    assert '-base "stage09.anat_autobox.nii.gz"' in s

    # Borrowed anat matrix: its base is the autobox in the REFERENCE dir.
    s = write_script(
        build_plan(
            subj,
            Options(grand_reference="/floc.results", anat_nonlin=True, tpm="/tpm.nii"),
        ),
        "wd",
        bids_root="/bids",
    )
    assert '-matrix_base "/floc.results/stage09.anat_autobox.nii.gz"' in s

    # User-supplied -ref_transforms: we cannot know their base, so say nothing and let
    # ffs_segment refuse rather than slip (it only refuses on an oblique TPM).
    s = write_script(
        build_plan(
            subj,
            Options(
                ref_file="/ref_epi.nii.gz",
                ref_transforms=["/r2a.1D"],
                anat_nonlin=True,
                tpm="/tpm.nii",
            ),
        ),
        "wd",
        bids_root="/bids",
    )
    assert "-matrix_base" not in s


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_hrftdv_stage_fits_pairs_from_the_design_toml_into_its_folder(tmp_path: Path):
    """-hrftdv: post-preprocessing, so everything lands in hrftdv_NN/ (no stageNN
    files). ffs_hrftdv reads the task's design TOML and the FreeSurfer subject, and
    the same spec is fit with its pairs; -hrftdv_model pooled_late swaps in the early
    library and the per-run late -dsort columns."""
    from fastfuncstuff.autoproc.bids import scan_subject

    _bids_with_events(tmp_path)
    subj = scan_subject(tmp_path, "ME1")
    opt = Options(run_glm=True, hrftdv=True, fs_subject="/fs/sub-ME1")
    s = write_script(build_plan(subj, opt), str(tmp_path / "wd"), bids_root=str(tmp_path))
    assert (
        subprocess.run(  # noqa: S603
            ["bash", "-n", str(_write(tmp_path / "p.sh", s))], capture_output=True
        ).returncode
        == 0
    )
    stage = s.split("hrftdv: temporal deveining")[1]
    spec = "stage11.design.task-floc.toml"
    assert spec in s.split("stage12: GLM")[1].split("hrftdv:")[0]  # stage12's own spec
    assert "select_hrftdv_dir" in stage and "hrftdv_[0-9][0-9]" in stage
    # the TOML's content is part of the settings, read when the script runs
    assert f'HRFTDV_SETTINGS+=$\'\\n\\n== {spec} ==\\n\'"$(cat "{spec}")"' in stage
    tdv = stage.split("ffs_hrftdv")[2]
    assert f"-spec {spec}" in tdv and "-fs_subj /fs/sub-ME1" in tdv
    assert '-prefix "$HRFTDV_DIR/task-floc"' in tdv and '-tile_mask "stage10.mask_brain' in tdv
    glm = stage.split("ffs_reml")[1]
    assert f"-spec {spec}" in glm and '-hrf_opt "$HRFTDV_DIR/task-floc"' in glm
    assert '-Rbuck "$HRFTDV_DIR/stats-reml.task-floc.nii$GLM_FMT"' in glm
    written = re.findall(r'-(?:prefix|Rbuck|Obuck) "([^"]+)"', stage)
    written += re.findall(r'qc_tcat "([^"]+)"', stage)
    assert written and all(w.startswith("$HRFTDV_DIR/") for w in written)

    opt.hrftdv_model = "pooled_late"
    s = write_script(build_plan(subj, opt), str(tmp_path / "wd2"), bids_root=str(tmp_path))
    glm = s.split("hrftdv: temporal deveining")[1].split("ffs_reml")[1]
    assert '-hrf_opt "$HRFTDV_DIR/task-floc_early"' in glm and "-dsort_betas yes" in glm
    assert '-dsort "$HRFTDV_DIR/task-floc_late_dsort.run-01.nii.gz" "$HRFTDV_DIR/' in glm


def test_hrftdv_preflight_needs_the_anat_space_and_no_trim(tmp_path: Path):
    """The surfaces sit on the anat, so an MNI (or EPI) final space is refused; so is a
    GLM trim, which ffs_hrftdv could not follow."""
    from fastfuncstuff.cli.autoproc import _hrftdv_errors

    (tmp_path / "surf").mkdir()
    ok = Options(go_to_anat=True, hrftdv=True, fs_subject=str(tmp_path))
    assert _hrftdv_errors(ok) == []
    assert _hrftdv_errors(Options(go_to_anat=True)) == []
    for bad, words in [
        (dict(do_mni=True), "-do_mni"),
        (dict(go_to_anat=False), "-no_anat"),
        (dict(fs_subject=None), "FreeSurfer subject"),
        (dict(fs_subject=str(tmp_path / "nope")), "surf/"),
        (dict(glm_drop_first=4), "-glm_drop_first"),
    ]:
        opt = Options(**{**ok.__dict__, **bad})
        assert any(words in e for e in _hrftdv_errors(opt)), bad


def test_numbered_folder_moves_on_when_a_settings_file_is_edited(tmp_path: Path):
    """hrftdv_NN keys on the design TOML's content as well as the commands: pairs fit to
    an old model must not be resumed under an edited one. Runs the bash block itself."""
    from fastfuncstuff.autoproc.emit import _numbered_dir_block

    def run() -> str:
        block = _numbered_dir_block("hrftdv", "ffs_hrftdv", "cmd", content_files=["d.toml"])
        r = subprocess.run(  # noqa: S603
            ["bash", "-c", "\n".join(block) + '\necho "DIR=$HRFTDV_DIR"'],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        return r.stdout.strip().splitlines()[-1].removeprefix("DIR=")

    (tmp_path / "d.toml").write_text("[meta]\ntr = 2\n")
    assert run() == "hrftdv_01"
    assert run() == "hrftdv_01"
    (tmp_path / "d.toml").write_text("[meta]\ntr = 2\n# a contrast added\n")
    assert run() == "hrftdv_02"
    assert "# a contrast added" in (tmp_path / "hrftdv_02" / "settings.txt").read_text()


# --------------------------------------------------------------------------------
# -fs_subject without -suma: FreeSurfer's own volumes stand in for a SUMA dir
# --------------------------------------------------------------------------------


def _vol_script(**kw):
    """A script whose anat comes from a FreeSurfer volume dir rather than a T1w."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1"), _run("01", "foo", "2")])])
    return write_script(build_plan(subj, Options(go_to_anat=True, **kw)), "wd", bids_root="/bids")


def test_fs_staged_converts_the_recon_volumes_before_anything_reads_them():
    """The anat IS one of the staged files, so the conversion has to come first.

    AFNI is the reason the conversion exists at all: load_image reads .mgz, but the
    subject-TPM build is 3dcalc on aseg label sets and stock AFNI reports NO-DSET
    for .mgz.
    """
    s = _vol_script(
        anat_path="/w/fs_work/brain.nii.gz",
        suma_dir="/w/fs_work",
        fs_staged=True,
        fs_subject="/fs/sub-X",
        anat_nonlin=True,
        tpm="tpm_work/sub_specific_fsTPM.nii.gz",
        fs_tpm=True,
    )
    for name, mgz in (
        ("brain.nii.gz", "brain.mgz"),
        ("aseg.auto.nii.gz", "aseg.auto.mgz"),
        ("fs_SurfVol.nii.gz", "orig.mgz"),
    ):
        assert f"-input /fs/sub-X/mri/{mgz}" in s
        assert f"-prefix /w/fs_work/{name}" in s
        assert f"[ -f /w/fs_work/{name} ] ||" in s  # idempotent across re-runs
    assert s.index("FreeSurfer volumes -> NIfTI") < s.index('ANAT="/w/fs_work/brain.nii.gz"')


def test_the_staged_tpm_build_is_the_suma_one_with_a_different_folder():
    """The whole point of staging into a SUMA-shaped folder: no second code path.

    The paths must stay absolute -- the build runs inside a subshell that has cd'd
    into tpm_work, where a work-dir-relative path would resolve nowhere.
    """
    kw = dict(anat_nonlin=True, fs_tpm=True, tpm="tpm_work/sub_specific_fsTPM.nii.gz")
    staged = _vol_script(
        anat_path="/w/fs_work/brain.nii.gz", suma_dir="/w/fs_work", fs_staged=True,
        fs_subject="/fs/sub-X", **kw,
    )  # fmt: skip
    suma = _vol_script(anat_path="/S/brain.nii.gz", suma_dir="/S", **kw)
    assert "( cd tpm_work" in staged
    assert "-a /w/fs_work/aseg.auto.nii.gz'<2,41,28,60,16,252,251,253,254,255>'" in staged

    # Same build, same order, same label sets -- only the folder moved.
    def tpm(script):
        body = script[
            script.index("build subject TPM") : script.index("sub_specific_fsTPM.nii.gz ")
        ]
        return body.replace("/w/fs_work", "DIR").replace("/S", "DIR")

    assert tpm(staged) == tpm(suma)
    # The staged SurfVol is gzipped where SUMA's is not, so the skull class globs both.
    assert "-b DIR/*_SurfVol.nii* " in tpm(staged)


def test_the_staged_brain_is_not_skull_stripped_again():
    """FreeSurfer's brain.mgz is already stripped, exactly as -suma's brain.nii.gz
    is -- the -anat_skull auto default has to know that for both."""
    import argparse

    from fastfuncstuff.cli.autoproc import _resolve_anat_skull

    args = argparse.Namespace(anat_skull="auto", suma=None)
    assert _resolve_anat_skull(args, presurf=False, fs_anat=True) is False
    assert _resolve_anat_skull(args, presurf=False, fs_anat=False) is True
    # An explicit -anat_skull yes still wins: the user may have pointed -anat
    # somewhere else entirely.
    assert _resolve_anat_skull(argparse.Namespace(anat_skull="yes", suma=None), False, True) is True


def test_no_staging_without_fs_staged():
    s = _vol_script(suma_dir="/S", anat_path="/S/brain.nii.gz")
    assert "FreeSurfer volumes -> NIfTI" not in s
    assert "ffs_util_3dmath \\\n    -input /fs" not in s


def test_surface_target_refuses_an_anat_the_surfaces_do_not_sit_on(tmp_path, capsys):
    """The silent bug: -surface_target with an anat from another scan. Both have a
    valid scanner RAS, the projection runs, every map comes out of the wrong place.
    """
    from fastfuncstuff.cli.autoproc import _check_surface_anat
    from tests.test_surface_anat_check import _anat, _subject

    subj = _subject(tmp_path)
    opt = Options(fs_subject=str(subj), surface_target=["native"])
    assert _check_surface_anat(opt, str(_anat(tmp_path))) == []
    out = capsys.readouterr().err
    assert "separability" in out  # the numbers print on the way through, pass or fail

    wrong = str(_anat(tmp_path, shift=(15.0, 0.0, 0.0), name="wrong.nii.gz"))
    errs = _check_surface_anat(opt, wrong)
    assert len(errs) == 1
    assert "do not sit on this anat" in errs[0]
    assert "recon-all was run on" in errs[0]  # and says what to do about it


def test_the_staged_anat_is_checked_against_its_source_not_its_future_path(tmp_path):
    """With -fs_subject the anat does not exist yet -- the script writes it -- so
    the check reads FreeSurfer's own copy of the same voxels instead of silently
    skipping and leaving the projection unverified."""
    import nibabel as nib
    import numpy as np

    from fastfuncstuff.cli.autoproc import _check_surface_anat
    from tests.test_surface_anat_check import _anat, _subject

    subj = _subject(tmp_path)
    mri = subj / "mri"
    mri.mkdir()
    src = nib.load(str(_anat(tmp_path)))
    nib.save(nib.MGHImage(np.asarray(src.dataobj), src.affine), str(mri / "brain.mgz"))
    opt = Options(fs_subject=str(subj), surface_target=["native"], fs_staged=True)
    assert _check_surface_anat(opt, "/w/fs_work/brain.nii.gz") == []

    # And a recon whose brain.mgz is missing or unreadable must not take the whole
    # plan down with it: the check is advisory, every real error is reported elsewhere.
    (mri / "brain.mgz").write_bytes(b"not a volume")
    assert _check_surface_anat(opt, "/w/fs_work/brain.nii.gz") == []


def test_xrun_method_moco_stacks_each_anchors_runs_into_one_call():
    """-xrun_method moco: stage06's linear step is one ffs_moco -stack per anchor,
    writing the same matrix / _lin names the allineate path does, so the chain and
    the nonlinear refinement downstream are unchanged."""
    runs = [_run("01", "foo", str(i)) for i in (1, 2, 3)]
    subj = Subject("X", [Session("01", runs)])
    opts = Options(xrun_method="moco", xrun_nonlin=True)
    s = write_script(build_plan(subj, opts), "wd", bids_root="/bids")
    stage = s[s.index("== stage06: cross-run") : s.index("== QC: cross-run")]
    assert "ffs_allineate -batch" not in stage
    assert 'ffs_moco -stack "$base" "${xsrc[@]}" -stack_matrices "${xmat[@]}"' in stage
    assert 'XSTK_MAT["$base"]+="${xstem}.aff12.1D"' in stage
    assert "ffs_formwarp -batch" in stage  # the nonlinear refinement still follows
    default = write_script(build_plan(subj, Options()), "wd", bids_root="/bids")
    assert "-stack_matrices" not in default and "ffs_allineate -batch" in default
    if shutil.which("bash"):
        assert subprocess.run(["bash", "-n"], input=s, text=True).returncode == 0


def test_qc_folder_wired_into_moco_and_report():
    """-qc: every moco job writes into qc/ under its run fragment, and the script
    ends by stitching the folder; -no_qc drops both."""
    subj = Subject("X", [Session("01", [_run("01", "foo", "1"), _run("01", "foo", "2")])])
    s = write_script(build_plan(subj, Options()), "wd", bids_root="/bids")
    assert '-qc qc -qc_label \\"${FRAG[$k]}\\"' in s
    assert s.index("stage02: moco") < s.index("ffs_qc_report qc")
    assert "-qc qc -qc_label anat" in s  # the EPI <-> anat alignment edges
    assert 'ffs_qc_stack -qc qc "${have[@]}"' in s  # every QC stack -> ranking + flicker
    assert '-qc qc -qc_label "stage12.task-foo"' in s  # the GLM design + F map
    opts = Options()
    opts.qc = False
    s2 = write_script(build_plan(subj, opts), "wd", bids_root="/bids")
    assert "-qc qc" not in s2 and "ffs_qc_report" not in s2
