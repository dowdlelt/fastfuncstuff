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
