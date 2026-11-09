"""ffs_util_surf2layers end to end on a synthetic two-hemisphere subject."""

from __future__ import annotations

import nibabel as nib
import nibabel.freesurfer as nfs
import numpy as np

from fastfuncstuff.cli.util_surf2layers import main
from tests.test_io_freesurfer import CRAS, _volume_info
from tests.test_surface_voxelize import icosphere

R1, R2 = 6.0, 9.0


def _subject(tmp_path):
    surf = tmp_path / "subj" / "surf"
    surf.mkdir(parents=True)
    v, f = icosphere(3)
    for hemi, dx in (("lh", -11.0), ("rh", 11.0)):
        c = np.array([dx, 0.0, 0.0])
        for name, r in (("white", R1), ("pial", R2)):
            nfs.write_geometry(
                str(surf / f"{hemi}.{name}"), (v * r + c).astype(np.float32), f.astype(np.int32),
                volume_info=_volume_info(),
            )  # fmt: skip
    # Master: 1 mm RAS grid around both spheres (surfaces sit at tkr + CRAS).
    affine = np.eye(4)
    affine[:3, 3] = CRAS - np.array([24.0, 12.0, 12.0])
    master = tmp_path / "master.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((49, 25, 25), np.float32), affine), str(master))
    return tmp_path / "subj", master


def test_writes_laynii_set_on_upsampled_master(tmp_path, capsys):
    subj, master = _subject(tmp_path)
    prefix = tmp_path / "out" / "s"
    args = ["-fs_subj", str(subj), "-master", str(master), "-dxyz", "0.5", "-prefix", str(prefix)]
    assert main([*args, "-device", "cpu", "-quiet", "-thick_warn", "2"]) == 0
    tags = ("rim", "metric_equidist", "layers_equidist", "midGM_equidist", "metric_equivol",
            "layers_equivol", "midGM_equivol", "thickness")  # fmt: skip
    imgs = {t: nib.load(f"{prefix}_{t}.nii.gz") for t in tags}
    rim = np.asarray(imgs["rim"].dataobj)
    assert imgs["rim"].get_data_dtype() == np.int16
    np.testing.assert_allclose(np.abs(np.diag(imgs["rim"].affine)[:3]), 0.5)
    assert set(np.unique(rim)) == {0, 1, 2, 3}
    # Both hemispheres, cropped to them (autobox), and depth in range.
    ijk = np.argwhere(rim == 3)
    xyz = ijk @ imgs["rim"].affine[:3, :3].T + imgs["rim"].affine[:3, 3] - CRAS
    assert (xyz[:, 0] < -2).any() and (xyz[:, 0] > 2).any()
    assert rim.shape[0] < 2 * 49
    m = np.asarray(imgs["metric_equidist"].dataobj)[rim == 3]
    assert 0 <= m.min() and m.max() <= 1
    th = np.asarray(imgs["thickness"].dataobj)[rim == 3]
    assert abs(np.median(th) - (R2 - R1)) < 0.05
    # Every GM voxel is 3 mm thick, so a 2 mm limit warns, with locations.
    assert "WARNING" in capsys.readouterr().out

    main([*args[:-2], "-prefix", str(tmp_path / "lh"), "-hemi", "lh", "-device", "cpu", "-quiet"])
    lh = np.asarray(nib.load(f"{tmp_path / 'lh'}_rim.nii.gz").dataobj)
    assert 0.4 < (lh == 3).sum() / (rim == 3).sum() < 0.6


def test_roi_restricts_and_crops_to_the_patch(tmp_path, capsys):
    """``-roi``: one patch per region, cropped, with the same depths as the whole run."""
    import nibabel.freesurfer as nfs

    from fastfuncstuff.io.freesurfer import write_label

    subj, master = _subject(tmp_path)
    run = ["-fs_subj", str(subj), "-master", str(master), "-dxyz", "0.5", "-device", "cpu"]
    whole = tmp_path / "whole"
    assert main([*run, "-prefix", str(whole), "-quiet"]) == 0
    full = nib.load(f"{whole}_rim.nii.gz")
    full_rim, full_eq = (
        np.asarray(full.dataobj),
        np.asarray(nib.load(f"{whole}_metric_equidist.nii.gz").dataobj),
    )

    # A drawn surface ROI: the +z cap of the left hemisphere.
    lh_white = nfs.read_geometry(str(subj / "surf" / "lh.white"))[0]
    ids = np.flatnonzero(lh_white[:, 2] > 4.0)
    assert ids.size > 20
    label = tmp_path / "cap.lh.label"
    # A label carries tkregister coordinates, which is what read_geometry returns;
    # the reader checks them against the white surface to find the hemisphere.
    write_label(label, ids, lh_white[ids], np.zeros(ids.size))
    # A volume ROI: a box over the right hemisphere, as a 3-D mask on the master grid.
    vol = np.zeros(full_rim.shape[:3], np.int16)
    vol[full_rim.shape[0] // 2 :, :, full_rim.shape[2] // 2 :] = 1
    roi_path = tmp_path / "boxy.nii.gz"
    nib.save(nib.Nifti1Image(vol, full.affine), str(roi_path))

    prefix = tmp_path / "r" / "p"
    assert main([*run, "-prefix", str(prefix), "-roi", str(label), str(roi_path)]) == 0
    out = capsys.readouterr().out
    assert "cap_lh" in out and "boxy_rh" in out and "WARNING" not in out

    for name in ("cap_lh", "boxy_rh"):
        rim = nib.load(f"{prefix}_{name}_rim.nii.gz")
        gm = np.asarray(rim.dataobj) == 3
        # Cropped to the patch, and much smaller than the whole-cortex run.
        assert gm.sum() and gm.sum() < 0.4 * (full_rim == 3).sum()
        assert np.prod(rim.shape) < np.prod(full_rim.shape)
        # The region's own mask is exactly the GM that was written.
        mask = np.asarray(nib.load(f"{prefix}_{name}_roi.nii.gz").dataobj)
        np.testing.assert_array_equal(mask > 0, gm)
        # Whole columns: the patch spans the depth even where the ROI did not.
        metric = np.asarray(nib.load(f"{prefix}_{name}_metric_equidist.nii.gz").dataobj)
        assert metric[gm].min() < 0.15 and metric[gm].max() > 0.85
        # Same voxels, same equidistant depth as the whole-cortex run: the crop and
        # the restriction move nothing (exact -- these are point-to-mesh distances).
        inv = np.linalg.inv(full.affine) @ rim.affine
        lo = np.rint(inv[:3, 3]).astype(int)
        sl = tuple(slice(a, a + n) for a, n in zip(lo, rim.shape, strict=True))
        assert (full_rim[sl][gm] == 3).all()
        np.testing.assert_allclose(full_eq[sl][gm], metric[gm], atol=1e-6)

    # A BILATERAL ROI (both +z caps, as a drawn V1 would be): one box per
    # hemisphere, not one box of mostly air spanning both.
    both = np.zeros(full_rim.shape[:3], np.int16)
    both[:, :, full_rim.shape[2] // 2 :] = 1
    bi = tmp_path / "v1.nii.gz"
    nib.save(nib.Nifti1Image(both, full.affine), str(bi))
    bpre = tmp_path / "b" / "p"
    assert main([*run, "-prefix", str(bpre), "-roi", str(bi), "-quiet"]) == 0
    boxes = [nib.load(f"{bpre}_v1_{h}_rim.nii.gz") for h in ("lh", "rh")]
    for box in boxes:
        assert (np.asarray(box.dataobj) == 3).any()
        # Each box reaches nowhere near the other hemisphere's sphere, 22 mm away.
        assert abs(box.affine[0, 0]) * box.shape[0] < 22.0
    assert boxes[0].affine[0, 3] != boxes[1].affine[0, 3]

    # One of them alone, as -hemi asks for, lands in the same place under the same
    # name -- so two -hemi runs of a bilateral ROI do not collide or disagree.
    hpre = tmp_path / "h" / "p"
    assert main([*run, "-hemi", "rh", "-prefix", str(hpre), "-roi", str(bi), "-quiet"]) == 0
    alone = nib.load(f"{hpre}_v1_rh_rim.nii.gz")
    np.testing.assert_allclose(alone.affine, boxes[1].affine)
    np.testing.assert_array_equal(np.asarray(alone.dataobj), np.asarray(boxes[1].dataobj))

    # Spill across the midline is not a region: the bigger side, and a warning.
    spill = np.zeros(full_rim.shape[:3], np.int16)
    spill[: full_rim.shape[0] // 2, :, full_rim.shape[2] // 2 :] = 1  # all of lh's cap
    spill[full_rim.shape[0] // 2 :, 12:14, full_rim.shape[2] // 2 :] = 1  # a sliver of rh's
    sp = tmp_path / "lop.nii.gz"
    nib.save(nib.Nifti1Image(spill, full.affine), str(sp))
    capsys.readouterr()
    assert main([*run, "-prefix", str(tmp_path / "s" / "p"), "-roi", str(sp), "-quiet"]) == 0
    warn = capsys.readouterr().out
    assert "WARNING" in warn and "dropping" in warn and "-hemi" in warn
    assert (tmp_path / "s" / "p_lop_lh_rim.nii.gz").exists()
    assert not (tmp_path / "s" / "p_lop_rh_rim.nii.gz").exists()


def test_alternative_surface_names(tmp_path):
    subj, master = _subject(tmp_path)
    for hemi in ("lh", "rh"):
        (subj / "surf" / f"{hemi}.pial.ffsedit").write_bytes(
            (subj / "surf" / f"{hemi}.pial").read_bytes()
        )
    prefix = tmp_path / "alt"
    args = [
        "-fs_subj",
        str(subj),
        "-master",
        str(master),
        "-pial",
        "pial.ffsedit",
        "-prefix",
        str(prefix),
    ]
    assert main([*args, "-device", "cpu", "-quiet"]) == 0
    assert (np.asarray(nib.load(f"{prefix}_rim.nii.gz").dataobj) == 3).any()


def _rim(prefix):
    return np.asarray(nib.load(f"{prefix}_rim.nii.gz").dataobj)


def test_meshes_without_a_subject_match_the_subject_and_take_edits(tmp_path):
    import shutil

    import pytest

    from fastfuncstuff.io.freesurfer import write_label

    subj, master = _subject(tmp_path)
    run = ["-master", str(master), "-device", "cpu", "-quiet"]
    assert main(["-fs_subj", str(subj), "-prefix", str(tmp_path / "subj_out"), *run]) == 0
    # The same files, moved out of any subject: identical layers.
    loose = tmp_path / "loose"
    shutil.copytree(subj / "surf", loose)
    shutil.rmtree(subj)
    meshes = ["-white_mesh", str(loose / "lh.white"), str(loose / "rh.white")]
    pial = ["-pial_mesh", str(loose / "rh.pial"), str(loose / "lh.pial")]  # paired by name
    assert main([*meshes, *pial, "-prefix", str(tmp_path / "mesh_out"), *run]) == 0
    np.testing.assert_array_equal(_rim(tmp_path / "subj_out"), _rim(tmp_path / "mesh_out"))

    # An edited pial saved as a copy: thicker cortex, originals untouched.
    v, f = nfs.read_geometry(str(loose / "lh.pial"))
    c = np.array([-11.0, 0.0, 0.0])
    edited = loose / "lh.pial.ffsedit"
    nfs.write_geometry(str(edited), ((v - c) * 1.2 + c).astype(np.float32), f,
                       volume_info=_volume_info())  # fmt: skip
    lh = ["-white_mesh", str(loose / "lh.white"), "-pial_mesh", str(edited)]
    assert main([*lh, "-prefix", str(tmp_path / "edit"), *run]) == 0
    assert main([*lh[:2], "-pial_mesh", str(loose / "lh.pial"),
                 "-prefix", str(tmp_path / "orig"), *run]) == 0  # fmt: skip
    assert (_rim(tmp_path / "edit") == 3).sum() > 1.3 * (_rim(tmp_path / "orig") == 3).sum()

    # A cortex label drops what it leaves out (here, the top half).
    top = np.flatnonzero(nfs.read_geometry(str(loose / "lh.white"))[0][:, 2] < 0)
    write_label(tmp_path / "lh.cortex.label", top, np.zeros((top.size, 3)), np.zeros(top.size))
    lab = ["-cortex_label", str(tmp_path / "lh.cortex.label")]
    assert main([*lh[:2], "-pial_mesh", str(loose / "lh.pial"), *lab,
                 "-prefix", str(tmp_path / "lab"), *run]) == 0  # fmt: skip
    assert (_rim(tmp_path / "lab") == 3).sum() < 0.7 * (_rim(tmp_path / "orig") == 3).sum()

    with pytest.raises(SystemExit, match="-master is required"):
        main([*lh, "-prefix", str(tmp_path / "x"), "-device", "cpu", "-quiet"])
    with pytest.raises(SystemExit, match="covers"):
        main([*lh[:2], "-pial_mesh", str(loose / "rh.pial"), "-prefix", str(tmp_path / "x"), *run])
    other = loose / "lh.pial.other"
    w, wf = icosphere(2)
    nfs.write_geometry(str(other), (w * R2 + c).astype(np.float32), wf.astype(np.int32),
                       volume_info=_volume_info())  # fmt: skip
    with pytest.raises(SystemExit, match="not the same mesh"):
        main([*lh[:2], "-pial_mesh", str(other), "-prefix", str(tmp_path / "x"), *run])


def test_ribbon_frac_keeps_the_voxels_the_ribbon_clips(tmp_path):
    """By default a voxel is GM when >= 10% of it lies between white and pial: the
    centre-rule set plus the edge voxels it drops, with P_gm_frac holding each
    share; -ribbon_frac 0 is the old rule and writes no gm_frac."""
    subj, master = _subject(tmp_path)
    run = ["-fs_subj", str(subj), "-master", str(master), "-device", "cpu", "-quiet"]
    assert main([*run, "-prefix", str(tmp_path / "frac")]) == 0
    assert main([*run, "-prefix", str(tmp_path / "ctr"), "-ribbon_frac", "0"]) == 0
    frac = np.asarray(nib.load(f"{tmp_path / 'frac'}_rim.nii.gz").dataobj) == 3
    ctr = np.asarray(nib.load(f"{tmp_path / 'ctr'}_rim.nii.gz").dataobj) == 3
    share = np.asarray(nib.load(f"{tmp_path / 'frac'}_gm_frac.nii.gz").dataobj)
    assert not (tmp_path / "ctr_gm_frac.nii.gz").exists()
    assert (ctr & ~frac).sum() == 0 and (frac & ~ctr).sum() > 0.05 * ctr.sum()
    assert share[frac].min() >= 0.1 - 1e-6 and share[~frac].max() == 0
    # The added voxels are the ribbon's edges: mostly not in it.
    assert np.median(share[frac & ~ctr]) < 0.5 < np.median(share[ctr])
    # Their depth is measured at their GM part, so it stays inside [0, 1].
    m = np.asarray(nib.load(f"{tmp_path / 'frac'}_metric_equidist.nii.gz").dataobj)[frac]
    assert 0 <= m.min() and m.max() <= 1
