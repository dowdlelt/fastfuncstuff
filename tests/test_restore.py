"""Returning a wrongly-removed component to a denoised series."""

import json

import nibabel as nib
import numpy as np
import torch

from fastfuncstuff.denoise.restore import restore_components


def _rank1_case(seed=0, nx=6, ny=5, nz=4, n_t=40, n_k=3):
    """A removed field built as a known sum of rank-1 terms, plus noise."""
    rng = np.random.RandomState(seed)
    n_vox = nx * ny * nz
    maps = rng.normal(size=(n_k, n_vox)).astype(np.float32)
    mixing = rng.normal(size=(n_t, n_k)).astype(np.float32)
    amps = np.array([3.0, 0.5, 2.0], dtype=np.float32)
    removed = sum(amps[c] * np.outer(maps[c], mixing[:, c]) for c in range(n_k))
    removed = removed.astype(np.float32).reshape(nx, ny, nz, n_t)
    denoised = rng.normal(loc=100.0, scale=1.0, size=(nx, ny, nz, n_t)).astype(np.float32)
    return denoised, removed, maps, mixing, amps


def test_restore_recovers_the_planted_amplitude_and_needs_the_joint_fit():
    """gamma must come from the DATA, not from the decomposition's arbitrary scale.

    ICA maps and mixing columns carry whatever whitening the decomposition applied, so
    adding a_c (x) s_c straight onto a magnitude image is off by an unknown factor.
    Here the planted amplitudes are known, so the fitted gammas must reproduce them.

    The components are deliberately NOT orthogonal, which is what makes this a test of
    the joint solve: the marginal projection each component would get on its own is
    computed alongside and has to be visibly wrong, or the joint fit is untested.
    """
    denoised, removed, maps, mixing, amps = _rank1_case()
    res = restore_components(
        torch.as_tensor(denoised),
        torch.as_tensor(removed),
        torch.as_tensor(maps),
        torch.as_tensor(mixing),
        [0, 1, 2],
    )
    np.testing.assert_allclose(res.gammas, amps, rtol=1e-4)
    assert res.dof_returned == 3

    lost = removed.reshape(-1, removed.shape[3]).astype(np.float64)
    marginal = np.array(
        [
            maps[c] @ (lost @ mixing[:, c]) / ((mixing[:, c] @ mixing[:, c]) * (maps[c] @ maps[c]))
            for c in range(3)
        ]
    )
    assert not np.allclose(marginal, amps, rtol=1e-2), (
        "the components are orthogonal enough that the marginal fit also works, so this "
        "case does not exercise the joint solve"
    )


def test_restore_puts_back_exactly_the_selected_terms():
    """Restoring every component reproduces denoised + removed; a subset does not."""
    denoised, removed, maps, mixing, _ = _rank1_case()
    d, r = torch.as_tensor(denoised), torch.as_tensor(removed)
    full = restore_components(d, r, torch.as_tensor(maps), torch.as_tensor(mixing), [0, 1, 2])
    np.testing.assert_allclose(full.restored.numpy(), denoised + removed, atol=1e-3)
    part = restore_components(d, r, torch.as_tensor(maps), torch.as_tensor(mixing), [1])
    assert not np.allclose(part.restored.numpy(), denoised + removed, atol=1e-3)


def test_restore_is_confined_to_the_component_map():
    """A voxel the map does not touch gets nothing back.

    That confinement is the whole reason rank-1 was chosen over regressing the time
    course into every voxel, so it is asserted on maps with genuinely disjoint support
    rather than inferred from small values of a dense random map.
    """
    rng = np.random.RandomState(2)
    nx, ny, nz, n_t = 6, 5, 4, 40
    n_vox = nx * ny * nz
    maps = np.zeros((2, n_vox), dtype=np.float32)
    maps[0, : n_vox // 2] = rng.uniform(0.5, 1.5, n_vox // 2)
    maps[1, n_vox // 2 :] = rng.uniform(0.5, 1.5, n_vox - n_vox // 2)
    mixing = rng.normal(size=(n_t, 2)).astype(np.float32)
    removed = (np.outer(maps[0], mixing[:, 0]) + np.outer(maps[1], mixing[:, 1])).reshape(
        nx, ny, nz, n_t
    )
    denoised = rng.normal(loc=50.0, scale=1.0, size=(nx, ny, nz, n_t)).astype(np.float32)
    res = restore_components(
        torch.as_tensor(denoised),
        torch.as_tensor(removed.astype(np.float32)),
        torch.as_tensor(maps),
        torch.as_tensor(mixing),
        [1],
    )
    delta = (res.restored.numpy() - denoised).reshape(-1, n_t)
    assert np.abs(delta[n_vox // 2 :]).max() > 0.5
    assert np.abs(delta[: n_vox // 2]).max() < 1e-4


def test_restore_variance_shares_sum_to_one_over_the_full_set():
    denoised, removed, maps, mixing, _ = _rank1_case()
    res = restore_components(
        torch.as_tensor(denoised),
        torch.as_tensor(removed),
        torch.as_tensor(maps),
        torch.as_tensor(mixing),
        [0, 1, 2],
    )
    assert abs(res.var_returned_total - 1.0) < 1e-3


def _write(path, arr, tr=None):
    img = nib.Nifti1Image(np.asarray(arr, dtype=np.float32), np.eye(4))
    if tr is not None:
        img.header["pixdim"][4] = tr
    nib.save(img, path)


def test_cli_selects_the_task_component_and_restores_it(tmp_path):
    """End to end: a task-locked component in the removed field is found and returned.

    The decoy matters. A second component with matched variance but no task structure
    has to be left where it is, or the tool is just undoing the denoising.
    """
    from fastfuncstuff.cli.util_restore import main

    rng = np.random.RandomState(4)
    nx, ny, nz, n_t, tr = 8, 7, 5, 96, 1.0
    n_vox = nx * ny * nz
    onsets = list(range(12, n_t - 12, 24))
    box = np.zeros(n_t, dtype=np.float32)
    for o in onsets:
        box[o : o + 12] = 1.0
    hrf = np.exp(-((np.arange(30) - 6.0) ** 2) / 18.0)
    task_tc = np.convolve(box, hrf / hrf.sum())[:n_t].astype(np.float32)
    task_tc -= task_tc.mean()
    decoy_tc = rng.normal(size=n_t).astype(np.float32)
    decoy_tc -= decoy_tc.mean()

    maps = np.zeros((2, n_vox), dtype=np.float32)
    maps[0, :40] = 1.0  # the task component, on a slab
    maps[1, 40:80] = 1.0  # the decoy, elsewhere
    mixing = np.stack([task_tc, decoy_tc], axis=1)
    removed = (np.outer(maps[0], task_tc) + np.outer(maps[1], decoy_tc)).reshape(nx, ny, nz, n_t)
    denoised = rng.normal(loc=100.0, scale=1.0, size=(nx, ny, nz, n_t)).astype(np.float32)

    _write(tmp_path / "den.nii.gz", denoised, tr=tr)
    _write(tmp_path / "rem.nii.gz", removed, tr=tr)
    _write(tmp_path / "maps.nii.gz", maps.T.reshape(nx, ny, nz, 2))
    np.savetxt(tmp_path / "tcs.1D", mixing, fmt="%.6f", delimiter="\t")
    ev = tmp_path / "ev.tsv"
    ev.write_text(
        "onset\tduration\ttrial_type\n" + "".join(f"{o}\t12\ttask\n" for o in onsets),
        encoding="utf-8",
    )

    main(
        [
            "-denoised",
            str(tmp_path / "den.nii.gz"),
            "-removed",
            str(tmp_path / "rem.nii.gz"),
            "-maps",
            str(tmp_path / "maps.nii.gz"),
            "-timecourses",
            str(tmp_path / "tcs.1D"),
            "-events",
            str(ev),
            "-prefix",
            str(tmp_path / "OUT"),
            "-surrogates",
            "300",
            "-device",
            "cpu",
        ]
    )
    meta = json.loads((tmp_path / "OUT_restore.json").read_text())
    assert meta["restored_components"] == [0], meta
    assert meta["dof_returned"] == 1

    out = nib.load(tmp_path / "OUT.nii.gz").get_fdata(dtype=np.float32)
    delta = (out - denoised).reshape(-1, n_t)
    # The task component came back where its map is, and the decoy's territory is
    # untouched -- restoring both would have been the failure worth catching.
    assert np.abs(delta[:40]).max() > 0.5
    assert np.abs(delta[40:80]).max() < 1e-3
