"""CLI-level tests for ffs_bbr's tissue (partial-volume) target and helpers."""

import numpy as np
import torch

from fastfuncstuff.cli.bbr import _normalize_targets, _ribbon_wm, main
from fastfuncstuff.processing.io import load_image, save_image


def test_normalize_targets_expands_both():
    assert _normalize_targets(["wm"]) == {"wm"}
    assert _normalize_targets(["both"]) == {"wm", "edges"}
    assert _normalize_targets(["wm", "tissue"]) == {"wm", "tissue"}
    assert _normalize_targets(["both", "tissue"]) == {"wm", "edges", "tissue"}


def _write(path, arr, aff):
    save_image(torch.from_numpy(arr.astype(np.float32)), str(path), header_info={"affine": aff})


def _phantom(tmp_path, D=40):
    z, y, x = np.mgrid[0:D, 0:D, 0:D]
    c = D / 2
    r = np.sqrt((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2)
    wm = (r < 9).astype(np.float32)
    gm = ((r >= 9) & (r < 13)).astype(np.float32)
    csf = ((r >= 13) & (r < 15)).astype(np.float32)
    epi = 20.0 + 90 * wm + 150 * gm + 220 * csf  # T2*-ish mixture
    aff = np.diag([2.5, 2.5, 2.5, 1.0])
    _write(tmp_path / "epi.nii.gz", epi, aff)
    _write(tmp_path / "wm.nii.gz", wm, aff)
    _write(tmp_path / "wm_pve.nii.gz", wm, aff)
    _write(tmp_path / "gm_pve.nii.gz", gm, aff)
    _write(tmp_path / "csf_pve.nii.gz", csf, aff)
    # Slightly-off anat->epi affine (DICOM mm), same grid.
    M = np.eye(4)
    M[0, 3] = 2.0
    np.savetxt(tmp_path / "aff.aff12.1D", M[:3, :].reshape(1, 12), fmt="%.6f")
    return aff


def test_ribbon_wm_union_labels(tmp_path):
    aff = _phantom(tmp_path)
    # FreeSurfer-labelled ribbon: 2 = lh WM, 41 = rh WM.
    lh = np.zeros((40, 40, 40), np.float32)
    lh[10:20, 10:30, 10:30] = 2
    rh = np.zeros((40, 40, 40), np.float32)
    rh[20:30, 10:30, 10:30] = 41
    _write(tmp_path / "lh.nii.gz", lh, aff)
    _write(tmp_path / "rh.nii.gz", rh, aff)
    wm, hdr = _ribbon_wm(
        str(tmp_path / "lh.nii.gz"), str(tmp_path / "rh.nii.gz"), torch.device("cpu")
    )
    assert wm.sum() > 0
    assert float(wm.max()) == 1.0  # binary union of the WM labels
    assert hdr is not None


def test_tissue_only_runs_and_writes_pve_outputs(tmp_path):
    _phantom(tmp_path)
    p = str(tmp_path / "bt")
    # No -wm_mask: the reference grid must come from a PVE.
    main(
        [
            "-epi",
            str(tmp_path / "epi.nii.gz"),
            "-1Dmatrix",
            str(tmp_path / "aff.aff12.1D"),
            "-wm_pve",
            str(tmp_path / "wm_pve.nii.gz"),
            "-gm_pve",
            str(tmp_path / "gm_pve.nii.gz"),
            "-csf_pve",
            str(tmp_path / "csf_pve.nii.gz"),
            "-target",
            "tissue",
            "-device",
            "cpu",
            "-prefix",
            p,
            "-verb",
            "0",
        ]
    )
    for name in ("wm", "gm", "csf"):
        out = tmp_path / f"bt_{name}_pve_in_epi.nii.gz"
        assert out.exists()
        v, _ = load_image(str(out))
        assert v.shape == (40, 40, 40)  # cast onto the EPI grid
    assert (tmp_path / "bt_epi2anat.aff12.1D").exists()
    # No WM boundary was requested, so no wm_in_epi overlay.
    assert not (tmp_path / "bt_wm_in_epi.nii.gz").exists()


def test_combined_wm_tissue_runs(tmp_path):
    _phantom(tmp_path)
    p = str(tmp_path / "bwt")
    main(
        [
            "-epi",
            str(tmp_path / "epi.nii.gz"),
            "-1Dmatrix",
            str(tmp_path / "aff.aff12.1D"),
            "-wm_mask",
            str(tmp_path / "wm.nii.gz"),
            "-wm_pve",
            str(tmp_path / "wm_pve.nii.gz"),
            "-gm_pve",
            str(tmp_path / "gm_pve.nii.gz"),
            "-csf_pve",
            str(tmp_path / "csf_pve.nii.gz"),
            "-target",
            "wm",
            "tissue",
            "-device",
            "cpu",
            "-prefix",
            p,
            "-verb",
            "0",
        ]
    )
    assert (tmp_path / "bwt_wm_in_epi.nii.gz").exists()
    assert (tmp_path / "bwt_gm_pve_in_epi.nii.gz").exists()
