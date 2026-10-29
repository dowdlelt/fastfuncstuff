"""Acquisition-family detection: which runs share a grid and a distortion."""

from __future__ import annotations

import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from fastfuncstuff.autoproc.bids import BoldRun, FmapGroup, Session, Subject
from fastfuncstuff.autoproc.emit import write_script
from fastfuncstuff.autoproc.family import detect_families
from fastfuncstuff.autoproc.plan import Options, build_plan


def _bold(
    tmp: Path,
    task,
    run,
    vox=2.0,
    shape=(8, 8, 6),
    pe="j-",
    readout=0.04,
    ses="01",
    acq="",
    origin=(0.0, 0.0, 0.0),
):
    path = tmp / f"sub-X_ses-{ses}_task-{task}_acq-{acq}_run-{run}_bold.nii.gz"
    aff = np.diag([vox, vox, vox, 1.0])
    aff[:3, 3] = origin
    img = nib.Nifti1Image(np.zeros((*shape, 1), np.uint8), aff)
    img.header.set_zooms((vox, vox, vox, 2.0))
    nib.save(img, path)
    return BoldRun(
        subject="X",
        session=ses,
        task=task,
        run=run,
        mag_path=path,
        json={"RepetitionTime": 2.0, "PhaseEncodingDirection": pe, "TotalReadoutTime": readout},
        acq=acq,
    )


def test_two_resolutions_are_two_families_labelled_by_task(tmp_path):
    runs = [
        _bold(tmp_path, "floc", "1", vox=2.0),
        _bold(tmp_path, "prim", "1", vox=1.2, shape=(12, 12, 8), readout=0.05),
        _bold(tmp_path, "floc", "2", vox=2.0),
    ]
    fams = detect_families(runs)
    assert [f.label for f in fams] == ["floc", "prim"]
    assert [len(f.members) for f in fams] == [2, 1]
    assert fams[1].signature.voxel_mm == (1.2, 1.2, 1.2)


def test_identical_acquisition_across_tasks_is_one_family(tmp_path):
    # Separate fieldmaps per task do not make separate families; the acquisition does.
    runs = [_bold(tmp_path, "floc", "1"), _bold(tmp_path, "prim", "1")]
    fams = detect_families(runs)
    assert len(fams) == 1 and fams[0].label == "fam1"


def test_acq_names_a_family_its_tasks_cannot(tmp_path):
    # Two tasks share the high-res acquisition, so neither task names that family;
    # its acq does. The other family owns its task outright and takes that.
    runs = [
        _bold(tmp_path, "floc", "1", vox=1.2, acq="hires"),
        _bold(tmp_path, "prim", "1", vox=1.2, acq="hires"),
        _bold(tmp_path, "rest", "1", vox=2.0, acq="lores"),
    ]
    assert [f.label for f in detect_families(runs)] == ["hires", "rest"]


def test_shared_task_and_acq_fall_back_to_numbers(tmp_path):
    # One task, one acq label, two resolutions: nothing names either family.
    runs = [
        _bold(tmp_path, "a", "1", vox=2.0, acq="x"),
        _bold(tmp_path, "a", "2", vox=1.2, acq="x"),
    ]
    assert [f.label for f in detect_families(runs)] == ["fam1", "fam2"]


def test_polarity_alone_splits_a_family(tmp_path):
    # AP and PA residual distortions have opposite sign: one anat_nl cannot serve both.
    fams = detect_families([_bold(tmp_path, "a", "1", pe="j-"), _bold(tmp_path, "a", "2", pe="j")])
    assert len(fams) == 2
    # The task spans both, so neither family can take its name.
    assert [f.label for f in fams] == ["fam1", "fam2"]


def test_override_merges_and_flags_mixed_acquisitions(tmp_path):
    runs = [_bold(tmp_path, "floc", "1", vox=2.0), _bold(tmp_path, "prim", "1", vox=1.2)]
    fams = detect_families(runs, {"floc": "all", "prim": "all"})
    assert len(fams) == 1
    assert fams[0].label == "all" and fams[0].forced and fams[0].mixed


def test_auto_label_avoids_a_user_label(tmp_path):
    runs = [_bold(tmp_path, "floc", "1", vox=2.0), _bold(tmp_path, "prim", "1", vox=1.2)]
    fams = detect_families(runs, {"prim": "floc"})
    assert sorted(f.label for f in fams) == ["fam1", "floc"]
    assert next(f for f in fams if f.label == "floc").tasks == ["prim"]


def test_unreadable_header_does_not_split(tmp_path):
    # Plans are built from scans whose data may not be on disk; unknown grid is one key.
    runs = [_bold(tmp_path, "a", "1"), _bold(tmp_path, "a", "2")]
    for r in runs:
        r.mag_path = tmp_path / "missing" / r.mag_path.name
    assert len(detect_families(runs)) == 1


def test_plan_labels_runs_and_header_names_families_only_when_several(tmp_path):
    two = [_bold(tmp_path, "floc", "1", vox=2.0), _bold(tmp_path, "prim", "1", vox=1.2)]
    plan = build_plan(Subject("X", [Session("01", two)]), Options())
    assert [pr.family for pr in plan.runs] == ["floc", "prim"]
    assert "acq families      : 2" in write_script(plan, "wd", bids_root="/bids")

    one = [_bold(tmp_path, "floc", "1"), _bold(tmp_path, "floc", "2")]
    plan = build_plan(Subject("X", [Session("01", one)]), Options())
    assert "acq families" not in write_script(plan, "wd", bids_root="/bids")


def _two_family_subject(tmp: Path, prim_ses="01") -> Subject:
    """floc at 2 mm PE j-, prim at 1.2 mm PE i, each with its own fieldmap."""
    floc = _bold(tmp, "floc", "1", vox=2.0, pe="j-")
    prim = _bold(tmp, "prim", "1", vox=1.2, pe="i", ses=prim_ses)
    f1 = FmapGroup("01", "floc", tmp / "floc_rev.nii.gz", {}, [("floc", "1")])
    f2 = FmapGroup(prim_ses, "prim", tmp / "prim_rev.nii.gz", {}, [("prim", "1")])
    anat = tmp / "T1w.nii.gz"
    if prim_ses == "01":
        return Subject("X", [Session("01", [floc, prim], [f1, f2], anat=anat)])
    return Subject(
        "X", [Session("01", [floc], [f1], anat=anat), Session(prim_ses, [prim], [f2], anat=anat)]
    )


def test_each_family_plans_alone_with_only_its_own_fieldmaps(tmp_path):
    opt = Options(go_to_anat=True, family_ref="highest_res", anat_nonlin=True)
    plan = build_plan(_two_family_subject(tmp_path), opt)
    # The higher-res family is the root, and its folder runs first.
    assert [fam.label for fam, _ in plan.family_plans] == ["prim", "floc"]
    for fam, sub in plan.family_plans:
        assert [pr.bold.task for pr in sub.runs] == fam.tasks
        (pr,) = sub.runs
        # Each family is its own reference: no cross-family xfmap link, and the
        # anat_nl it carries is the one estimated on its own data.
        assert pr.is_ref_fmap and pr.fmap.fmap_id == fam.label
        assert not any(t.startswith("xfmap") for t in pr.warp_chain)
        assert "anat_nl" in pr.warp_chain
    # The whole-subject plan still links them, which is what the families replace.
    assert any(t.startswith("xfmap") for t in plan.runs[1].warp_chain)


def test_multi_family_script_runs_each_family_in_its_own_folder(tmp_path):
    opt = Options(go_to_anat=True, family_ref="highest_res", anat_nonlin=True)
    script = write_script(build_plan(_two_family_subject(tmp_path), opt), "wd", bids_root="/bids")
    sh = tmp_path / "proc.sh"
    sh.write_text(script)
    res = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    _, prim, floc = script.split("\n# family ")
    assert '(\nmkdir -p "fam-floc" && cd "fam-floc"' in floc
    assert '(\nmkdir -p "fam-prim" && cd "fam-prim"' in prim
    # Its own PE axis for the nonlinear anat correction, its own voxel size for the
    # output grid (read from its own first run).
    assert "-pe_axis y" in floc and "-pe_axis x" in prim
    assert "_task-floc_" in floc.split("FINAL_DXYZ=")[1].splitlines()[0]
    assert "_task-prim_" in prim.split("FINAL_DXYZ=")[1].splitlines()[0]
    assert "task-prim" not in floc and "task-floc" not in prim


def test_ref_ses_a_family_lacks_falls_back_to_its_own_first_session(tmp_path):
    subj = _two_family_subject(tmp_path, prim_ses="02")
    plan = build_plan(subj, Options(ref_ses="01"))
    assert {f.label: sub.ref_session for f, sub in plan.family_plans} == {
        "floc": "01",
        "prim": "02",
    }


def test_without_an_anat_families_stay_one_pipeline(tmp_path):
    # Split families meet only at the anat; with none they would never be aligned.
    plan = build_plan(_two_family_subject(tmp_path), Options(go_to_anat=False))
    assert len(plan.families) == 2 and not plan.family_plans
    script = write_script(plan, "wd", bids_root="/bids")
    assert "fam-" not in script and "ONE pipeline" in script


def _slab_subject(tmp: Path, prim_z0: float = 20.0) -> Subject:
    """Realistic slabs: floc 2 mm 96x96x40 (80 mm thick) from z=0, prim 1.2 mm
    160x160x50 (60 mm thick) starting at ``prim_z0`` — inside floc by default."""
    floc = _bold(tmp, "floc", "1", vox=2.0, shape=(96, 96, 40), readout=0.06)
    prim = _bold(
        tmp, "prim", "1", vox=1.2, shape=(160, 160, 50), readout=0.08, origin=(0, 0, prim_z0)
    )
    f1 = FmapGroup("01", "floc", tmp / "floc_rev.nii.gz", {}, [("floc", "1")])
    f2 = FmapGroup("01", "prim", tmp / "prim_rev.nii.gz", {}, [("prim", "1")])
    return Subject("X", [Session("01", [floc, prim], [f1, f2], anat=tmp / "T1w.nii.gz")])


def test_overlap_reads_fraction_and_slab_extent_from_headers(tmp_path):
    from fastfuncstuff.autoproc.family import overlap

    subj = _slab_subject(tmp_path)
    floc, prim = (b.mag_path for b in subj.sessions[0].bold_runs)
    frac, extent = overlap(prim, floc)  # prim lies wholly inside floc
    assert frac == 1.0 and abs(extent - 60) < 2
    frac, extent = overlap(floc, prim)  # floc is 80 mm thick; prim covers 60 of it
    assert 0.7 < frac < 0.8 and abs(extent - 60) < 3


def test_child_links_to_the_root_through_its_corrected_anchor(tmp_path):
    opt = Options(go_to_anat=True, family_ref="highest_res", anat_nonlin=True)
    plan = build_plan(_slab_subject(tmp_path), opt)
    links = {f.label: f.link for f in plan.families}
    assert links == {"prim": "root", "floc": "xfam"}
    (root_fam, root), (child_fam, child) = plan.family_plans
    assert root_fam.label == "prim" and child.xfam_root is not None
    # The child's chain gains the rigid link above its own anat_lin, keeping its
    # own anat_nl; the root's does not.
    toks = child.runs[0].warp_chain
    assert toks.index("xfam_lin") + 1 == toks.index("anat_lin") and "anat_nl" in toks
    assert "xfam_lin" not in root.runs[0].warp_chain

    script = write_script(plan, "wd", bids_root="/bids")
    sh = tmp_path / "proc.sh"
    sh.write_text(script)
    assert subprocess.run(["bash", "-n", str(sh)], capture_output=True).returncode == 0
    _, prim, floc = script.split("\n# family ")
    # Base = the root folder's corrected anchor; source = this family's own.
    assert '-base "../fam-prim/stage09.' in floc and "_al_anat_nl.nii$FMT" in floc
    assert "stage09.xfam.aff12.1D" in floc and "stage09.xfam" not in prim
    # The child's own anchor is built WITHOUT the link it is the source of...
    assert '-nwarp "stage09.anat.aff12.1D stage09.nlanat_invwarp.nii$FMT"' in floc
    # ...every run's chain leads with it...
    assert 'CHAIN[01:floc:1]="stage09.xfam.aff12.1D stage09.anat.aff12.1D ' in floc
    # ...and the output grid is cut from the anchor WITH it.
    assert 'MASTER="${FFS_MASTER:-stage09.xfam_lin.nii$FMT}"' in floc


def test_thin_overlap_falls_back_to_the_anat(tmp_path):
    # prim starts at z=70 of floc's 80 mm: 10 mm of shared slab cannot pin rotation.
    plan = build_plan(
        _slab_subject(tmp_path, prim_z0=70.0), Options(go_to_anat=True, family_ref="highest_res")
    )
    fam = {f.label: f for f in plan.families}["floc"]
    assert fam.link == "anat" and "20 mm" in fam.link_reason
    assert all(sub.xfam_root is None for _, sub in plan.family_plans)


def test_family_ref_and_link_overrides(tmp_path):
    subj = _slab_subject(tmp_path, prim_z0=70.0)
    plan = build_plan(subj, Options(go_to_anat=True, family_ref="floc"))
    assert {f.label: f.link for f in plan.families}["floc"] == "root"
    plan = build_plan(
        _slab_subject(tmp_path, prim_z0=70.0),
        Options(go_to_anat=True, family_ref="highest_res", family_links={"floc": "root"}),
    )
    assert {f.label: f.link for f in plan.families}["floc"] == "xfam"
    plan = build_plan(_slab_subject(tmp_path), Options(go_to_anat=True, family_ref="anat"))
    assert {f.link for f in plan.families} == {"anat"}
    assert all(sub.xfam_root is None for _, sub in plan.family_plans)


def test_anat_prep_and_mni_run_once_and_are_linked_into_each_family(tmp_path):
    opt = Options(
        go_to_anat=True,
        anat_nonlin=True,
        anat_tpm=True,
        anat_skull=True,
        anat_path=str(tmp_path / "T1w.nii.gz"),
        spm_tpm="/TPM.nii",
        do_mni=True,
        mni_template="/MNI.nii.gz",
    )
    script = write_script(build_plan(_slab_subject(tmp_path), opt), "wd", bids_root="/bids")
    sh = tmp_path / "proc.sh"
    sh.write_text(script)
    assert subprocess.run(["bash", "-n", str(sh)], capture_output=True).returncode == 0
    head, *families = script.split("\n# family ")
    shared = head.split("shared: anat prep")[1]
    # Once, in the parent folder: strip, autobox, subject TPM, MNI affine + warp.
    # (The calls, not the names: the preflight lists every tool, and every CHAIN
    # names the MNI matrix.)
    for step in (
        'mri_synthstrip -i "',
        '-prefix "stage09.anat_autobox.nii.gz"',
        '-prefix "tpm_work/anatseg"',
        '-1Dmatrix_save "stage09.mni.aff12.1D"',
    ):
        assert step in shared and all(step not in fam for fam in families), step
    for fam in families:
        assert 'ln -sfn "$f" .' in fam
        # Each family still aligns its own EPI and pushes its own anchor into MNI.
        assert '-1Dmatrix_save "stage09.anat.aff12.1D"' in fam
        assert "_al_mni" in fam


def test_families_align_to_the_anat_on_their_own_by_default(tmp_path):
    """Families can be very different acquisitions; by default none is a root, and
    each is aligned to the anat by its own anat_lin/anat_nl, onto its own grid."""
    plan = build_plan(_slab_subject(tmp_path), Options(go_to_anat=True, anat_nonlin=True))
    assert {f.label: f.link for f in plan.families} == {"prim": "anat", "floc": "anat"}
    for _, fplan in plan.family_plans:
        assert fplan.xfam_root is None
        assert all(not t.startswith("xfam_") for pr in fplan.runs for t in pr.warp_chain)


def test_ref_anat_link_is_fitted_once_and_shared_by_every_family(tmp_path):
    """Families meet at the anat, so they must reach it through ONE ref→anat warp:
    the ref block runs in the parent folder (like the MNI link), and each family
    still aligns its own EPI to the ref and chains the shared links."""
    opt = Options(
        go_to_anat=True,
        anat_nonlin=True,
        anat_tpm=True,
        anat_skull=True,
        anat_path=str(tmp_path / "T1w.nii.gz"),
        spm_tpm="/TPM.nii",
        ref_anat="/data/7T_T1w.nii.gz",
    )
    script = write_script(build_plan(_slab_subject(tmp_path), opt), "wd", bids_root="/bids")
    sh = tmp_path / "proc.sh"
    sh.write_text(script)
    assert subprocess.run(["bash", "-n", str(sh)], capture_output=True).returncode == 0
    head, *families = script.split("\n# family ")
    shared = head.split("shared: anat prep")[1]
    assert len(families) == 2
    for step in (
        '-1Dmatrix_save "stage09.refanat.aff12.1D"',
        '-prefix "stage09.refanat_seg.nii.gz"',
        '-prefix "tpm_work/refanat_TPM.nii.gz"',
    ):
        assert step in shared and all(step not in fam for fam in families), step
    # The TPM the ref segment reads is built before it.
    assert shared.index('-prefix "tpm_work/anatseg"') < shared.index("stage09.refanat_seg.nii.gz")
    for fam in families:
        assert '-base "stage09.refanat_brain.nii.gz"' in fam  # own EPI → the ref
        assert '-tpm "tpm_work/refanat_TPM.nii.gz"' in fam
        chain = next(x for x in fam.splitlines() if x.startswith("CHAIN["))
        assert chain.split('="')[1].startswith(
            "stage09.refanat.aff12.1D stage09.refanat_seg_invwarp.nii.gz stage09.anat.aff12.1D"
        )


def test_xfam_nonlin_fits_on_the_childs_own_grid_along_its_pe_axis(tmp_path):
    opt = Options(go_to_anat=True, family_ref="highest_res", anat_nonlin=True, xfam_nonlin=True)
    plan = build_plan(_slab_subject(tmp_path), opt)
    (_, root), (_, child) = plan.family_plans
    toks = child.runs[0].warp_chain
    # Between anat_lin and anat_nl: the frame anat_nl acts in, not the anat grid.
    i = toks.index("anat_lin")
    assert toks[i : i + 3] == ["anat_lin", "xfam_nl", "anat_nl"]
    assert toks[i - 1] == "xfam_lin" and "xfam_nl" not in root.runs[0].warp_chain

    script = write_script(plan, "wd", bids_root="/bids")
    sh = tmp_path / "proc.sh"
    sh.write_text(script)
    assert subprocess.run(["bash", "-n", str(sh)], capture_output=True).returncode == 0
    floc = script.split("\n# family ")[2]
    # The root's anchor comes back through INV(anat_lin) then INV(xfam)...
    assert '-nwarp "INV(stage09.anat.aff12.1D) INV(stage09.xfam.aff12.1D)"' in floc
    # ...and the fit holds the two non-PE axes still (floc is PE j-, so y moves).
    fit = floc.split('-warp_prefix "stage09.xfam_nl"')[1].split("-device")[0]
    assert "-noXdis" in fit and "-noZdis" in fit and "-noYdis" not in fit
    # The output grid is cut from the anchor through the whole head, warp included.
    assert '-nwarp "stage09.xfam.aff12.1D stage09.anat.aff12.1D stage09.xfam_nl_WARP' in floc
    assert 'MASTER="${FFS_MASTER:-stage09.xfam_nl_al_anat.nii$FMT}"' in floc
    # The family's own anchor still excludes both cross-family links.
    assert '-nwarp "stage09.anat.aff12.1D stage09.nlanat_invwarp.nii$FMT"' in floc


def test_cross_family_links_stay_out_of_the_grandmean_chain(tmp_path):
    # Bug of record: stage08's gmrun chain carried xfam_lin/xfam_nl, which stage09
    # makes from that very grandmean, so the script died on a missing matrix.
    from fastfuncstuff.autoproc.emit import _grandmean_tokens, _needs_gmrun

    opt = Options(go_to_anat=True, family_ref="highest_res", anat_nonlin=True, xfam_nonlin=True)
    plan = build_plan(_slab_subject(tmp_path), opt)
    (_, _), (_, child) = plan.family_plans
    toks = _grandmean_tokens(child.runs[0])
    assert not any(t.startswith(("xfam", "anat", "mni", "xref")) for t in toks), toks
    # The stray links are also what made a single-session run look as if it needed
    # its own gmrun resample at all.
    assert not _needs_gmrun(child.runs[0])


def test_every_family_cuts_its_own_warpmaster_and_mask(tmp_path):
    opt = Options(go_to_anat=True, family_ref="highest_res", anat_nonlin=True, xfam_nonlin=True)
    script = write_script(build_plan(_slab_subject(tmp_path), opt), "wd", bids_root="/bids")
    _, prim, floc = script.split("\n# family ")
    for fam, anchor in ((prim, "grandmean_al_anat_nl"), (floc, "xfam_nl_al_anat")):
        # Its own anchor (so its own FoV), its own voxel size, its own mask.
        assert f'MASTER="${{FFS_MASTER:-stage09.{anchor}.nii$FMT}}"' in fam
        assert "stage10.warpmaster.nii$FMT" in fam and "mask_epi.nii$FMT" in fam
    assert "_task-prim_" in prim.split("FINAL_DXYZ=")[1].splitlines()[0]
    assert "_task-floc_" in floc.split("FINAL_DXYZ=")[1].splitlines()[0]
