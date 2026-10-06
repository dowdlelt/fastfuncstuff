"""presurfer workflow primitives (processing/presurf.py)."""

from __future__ import annotations

import torch

from fastfuncstuff.processing.presurf import (
    clean_mask,
    mprageise,
    scale_unit,
    strip_mask,
)


def _ball(shape, centre, r):
    zz, yy, xx = torch.meshgrid(*[torch.arange(n) for n in shape], indexing="ij")
    return (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2 <= r * r


def test_scale_unit_minmax_and_robust():
    x = torch.tensor([0.0, 1.0, 2.0, 4.0, 1000.0])
    assert torch.allclose(scale_unit(x), x / 1000.0)
    # one hot voxel should not set the top of the robust scale
    r = scale_unit(torch.cat([torch.linspace(1, 100, 10_000), torch.tensor([1e6])]), "robust")
    assert r[5000] > 0.4  # a mid-range value stays mid-range


def test_mprageise_suppresses_background():
    uni = torch.full((4,), 3000.0)
    inv2 = torch.tensor([0.0, 10.0, 500.0, 1000.0])  # dark background -> bright tissue
    out = mprageise(uni, inv2)
    assert out[0] == 0 and out[-1] == 3000.0


def test_strip_mask_is_complement_of_other_classes():
    post = torch.zeros(6, 1, 1, 3)
    post[0, 0, 0, 0] = 1.0  # GM
    post[3, 0, 0, 1] = 1.0  # bone
    post[0, 0, 0, 2], post[2, 0, 0, 2] = 0.3, 0.3  # sums to 0.6 -- others 0.3 -> kept
    assert strip_mask(post).flatten().tolist() == [True, False, True]


def test_clean_mask_prior_gate_removes_bridged_eye_and_fills_holes():
    shape = (40, 40, 60)
    brain = _ball(shape, (20, 20, 20), 12)
    eye = _ball(shape, (20, 20, 46), 6)
    bridge = torch.zeros(shape, dtype=torch.bool)
    bridge[19:21, 19:21, 30:42] = True  # an "optic nerve" joining eye to brain
    hole = _ball(shape, (20, 20, 20), 3)
    mask = (brain | eye | bridge) & ~hole

    # largest component alone keeps the eye: it is connected
    kept, _ = clean_mask(mask, fill_holes=False)
    assert kept[eye].all()

    prior = brain.float()  # template says brain only where the brain is
    cleaned, removed = clean_mask(mask, brain_prior=prior, prior_thresh=0.05)
    assert not cleaned[eye].any() and removed[eye].all()
    assert cleaned[brain].all()  # interior hole filled, brain intact
    assert not removed[brain & ~hole].any()


def test_clean_mask_opening_cuts_bridge_without_prior():
    shape = (40, 40, 60)
    brain = _ball(shape, (20, 20, 20), 12)
    eye = _ball(shape, (20, 20, 46), 6)
    bridge = torch.zeros(shape, dtype=torch.bool)
    bridge[19:21, 19:21, 30:42] = True
    cleaned, _ = clean_mask(brain | eye | bridge, open_radius=2)
    assert not cleaned[eye].any() and cleaned[brain].all()


def test_util_presurf_cli_end_to_end(tmp_path):
    """Smoke: the whole CLI on a phantom, into a not-yet-existing prefix directory."""
    import nibabel as nib
    import numpy as np

    from fastfuncstuff.cli.util_presurf import main

    n = 24
    r = (
        torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float32)] * 3, indexing="ij"))
        .sub(n / 2)
        .norm(dim=0)
    )
    tissue = torch.stack([(r < 5), (r >= 5) & (r < 8), (r >= 8) & (r < 11)]).float()
    tpm = (tissue + 0.01) / (tissue + 0.01).sum(0, keepdim=True)
    uni = (tissue * torch.tensor([2500.0, 3500.0, 1000.0])[:, None, None, None]).sum(0) + 2000
    inv2 = (tissue * torch.tensor([400.0, 300.0, 600.0])[:, None, None, None]).sum(0) + 5

    def write(path, arr):
        a = arr.numpy()
        a = a.transpose(2, 1, 0) if a.ndim == 3 else a.transpose(3, 2, 1, 0)
        nib.save(nib.Nifti1Image(np.ascontiguousarray(a), np.eye(4)), str(path))

    write(tmp_path / "uni.nii.gz", uni)
    write(tmp_path / "inv2.nii.gz", inv2)
    write(tmp_path / "tpm.nii.gz", tpm.permute(0, 1, 2, 3))
    prefix = tmp_path / "out" / "sub"
    rc = main(
        [
            "-uni", str(tmp_path / "uni.nii.gz"), "-inv2", str(tmp_path / "inv2.nii.gz"),
            "-tpm", str(tmp_path / "tpm.nii.gz"), "-prefix", str(prefix),
            "-ngaus", "1", "1", "1", "-samp", "1", "-affreg", "off",
            "-device", "cpu", "-quiet",
        ]
    )  # fmt: skip
    assert rc == 0
    for name in ("MPRAGEised_stripped", "stripmask", "stripmask_raw", "brainmask", "WMmask"):
        assert (tmp_path / "out" / f"sub_{name}.nii.gz").exists(), name


def test_util_presurf_cli_denoise_reg(tmp_path):
    """-denoise reg: joint INV1/INV2 denoising feeds both segmentations."""
    import nibabel as nib
    import numpy as np

    from fastfuncstuff.cli.util_presurf import main
    from fastfuncstuff.processing.mp2rage import uni_from_inversions

    n = 24
    g = torch.Generator().manual_seed(0)
    r = (
        torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float32)] * 3, indexing="ij"))
        .sub(n / 2)
        .norm(dim=0)
    )
    tissue = torch.stack([(r < 5), (r >= 5) & (r < 8), (r >= 8) & (r < 11)]).float()
    tpm = (tissue + 0.01) / (tissue + 0.01).sum(0, keepdim=True)
    inv1s = (tissue * torch.tensor([70.0, -30.0, -120.0])[:, None, None, None]).sum(0)
    inv2 = (tissue * torch.tensor([200.0, 175.0, 110.0])[:, None, None, None]).sum(0) + 2
    uni = uni_from_inversions(inv1s, inv2)
    inv1 = (inv1s + 3 * torch.randn(n, n, n, generator=g)).abs()
    inv2 = (inv2 + 3 * torch.randn(n, n, n, generator=g)).abs()

    def write(path, arr):
        a = arr.numpy()
        a = a.transpose(2, 1, 0) if a.ndim == 3 else a.transpose(3, 2, 1, 0)
        nib.save(nib.Nifti1Image(np.ascontiguousarray(a), np.eye(4)), str(path))

    for name, arr in (("uni", uni), ("inv1", inv1), ("inv2", inv2), ("tpm", tpm)):
        write(tmp_path / f"{name}.nii.gz", arr)
    rc = main(
        [
            "-uni", str(tmp_path / "uni.nii.gz"), "-inv2", str(tmp_path / "inv2.nii.gz"),
            "-inv1", str(tmp_path / "inv1.nii.gz"), "-denoise", "reg",
            "-tpm", str(tmp_path / "tpm.nii.gz"), "-prefix", str(tmp_path / "o" / "s"),
            "-ngaus", "1", "1", "1", "-samp", "1", "-affreg", "off",
            "-device", "cpu", "-quiet",
        ]
    )  # fmt: skip
    assert rc == 0
    for name in ("UNIreg", "UNIreg_stripped", "inv2_denoised", "stripmask", "brainmask"):
        assert (tmp_path / "o" / f"s_{name}.nii.gz").exists(), name
