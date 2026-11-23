"""Tests for processing/reface.py and ffs_util_reface (AFNI @afni_refacer_run port)."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.reface import (
    SHELL_GAIN,
    apply_shell,
    remove_isolas,
    remove_isolated_mask_voxels,
    shell_gain,
    shell_on_subject,
)


def _isola_c(far: np.ndarray) -> np.ndarray:
    """3dcalc.c remove_isolated_stuff, one pass, transcribed loop for loop.

    ``far`` is (nz, ny, nx); FF(i,j,k) = far[k, j, i]. Kept literal -- including
    the neighbour list's repeated and missing rows -- so the vectorised port is
    held to what AFNI actually does, not to what the comment says it does.
    """
    nz, ny, nx = far.shape
    q = far.copy()

    def ff(i, j, k):
        return far[k, j, i]

    for k in range(1, nz - 1):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                if ff(i, j, k) != 0:
                    nb = [ff(i + di, j + dj, k - 1) for dj in (-1, 0, 1) for di in (-1, 0, 1)]
                    nb += [ff(i + di, j - 1, k) for di in (-1, 0, 1)]
                    nb += [ff(i + di, j, k) for di in (-1, 0, 1)]
                    nb += [ff(i + di, j - 1, k) for di in (-1, 0, 1)]  # AFNI repeats j-1
                    nb += [ff(i + di, j + 1, k + 1) for di in (-1, 0, 1)]
                    nb += [ff(i + di, j, k + 1) for di in (-1, 0, 1)]
                    nb += [ff(i + di, j + 1, k + 1) for di in (-1, 0, 1)]  # and j+1 here
                    if sum(v != 0 for v in nb) < 4:
                        q[k, j, i] = 0
    for k in range(nz):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                if q[k, j, i] != 0:
                    c = sum(ff(i + a, j + b, k) != 0 for b in (-1, 0, 1) for a in (-1, 0, 1))
                    if c < 2:
                        q[k, j, i] = 0
    for j in range(ny):
        for k in range(1, nz - 1):
            for i in range(1, nx - 1):
                if q[k, j, i] != 0:
                    c = sum(ff(i + a, j, k + b) != 0 for b in (-1, 0, 1) for a in (-1, 0, 1))
                    if c < 2:
                        q[k, j, i] = 0
    for i in range(nx):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                if q[k, j, i] != 0:
                    c = sum(ff(i, j + a, k + b) != 0 for b in (-1, 0, 1) for a in (-1, 0, 1))
                    if c < 2:
                        q[k, j, i] = 0
    return q


@pytest.mark.parametrize("density", [0.05, 0.15, 0.3])
def test_remove_isolas_matches_afni_loop(density):
    # Checked once against 3dcalc -ISOLA itself (0 mismatches at these densities);
    # a plain 3x3x3 stencil got 3-41 voxels wrong, so the quirk is load-bearing.
    rng = np.random.default_rng(int(density * 100))
    vol = (rng.random((9, 11, 10)) < density) * rng.integers(1, 9, (9, 11, 10))
    vol = vol.astype(np.float32)
    ours = remove_isolas(torch.from_numpy(vol)).numpy()
    np.testing.assert_array_equal(ours, _isola_c(vol))


def test_remove_isolated_mask_voxels_needs_a_face_neighbour():
    m = torch.zeros(5, 5, 5, dtype=torch.bool)
    m[1, 1, 1] = True  # alone
    m[3, 3, 2] = m[3, 3, 3] = True  # a face-adjacent pair
    m[1, 3, 3] = m[2, 4, 4] = True  # only diagonal to each other
    out = remove_isolated_mask_voxels(m)
    assert out.sum() == 2 and out[3, 3, 2] and out[3, 3, 3]


def _shell(shape=(20, 22, 24)) -> torch.Tensor:
    """Three-valued shell: -1 outside, 0 brain box, 100..300 face slab, 2 bricks."""
    s = -torch.ones(shape)
    s[4:16, 4:18, 4:20] = 0.0
    s[4:16, 4:18, 16:20] = torch.linspace(100, 300, 4)
    face_only = s.clone()
    face_only[4:16, 4:18, 18:20] = 0.0  # brick 1 replaces less
    return torch.stack([s, face_only])


def test_shell_on_subject_identity_and_shift():
    shell = _shell()
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    same = shell_on_subject(shell, aff, torch.eye(4), aff, tuple(shell.shape[1:]))
    torch.testing.assert_close(same, torch.stack([remove_isolas(b) for b in shell]))
    # Template->subject moving +4 mm in DICOM x (= -x in RAS, i.e. -2 voxels in i
    # on this RAS-diagonal grid): subject voxel i holds shell voxel i + 2.
    move = torch.eye(4)
    move[0, 3] = 4.0
    moved = shell_on_subject(shell, aff, move, aff, tuple(shell.shape[1:]))
    torch.testing.assert_close(moved[0, :, :, :-2], same[0, :, :, 2:])


def test_shell_gain_is_afni_ifac():
    shell = _shell()
    subj = torch.full(shell.shape[1:], 50.0)
    sel = shell > 99
    expected = SHELL_GAIN * 50.0 / float(shell[sel].double().mean())
    assert shell_gain(subj, shell) == pytest.approx(expected, rel=1e-9)


def test_apply_shell_modes():
    shell = _shell()
    torch.manual_seed(0)
    subj = 100.0 + 10.0 * torch.rand(shell.shape[1:])
    vox = (1.0, 1.0, 1.0)
    brain = shell[1] == 0

    deface = apply_shell(subj, shell, "deface", 0.5, vox)
    assert torch.equal(deface[brain], subj[brain])
    assert not deface[shell[1] != 0].any()  # face AND outside zeroed

    for mode, b in (("reface", 1), ("reface_plus", 0)):
        out = apply_shell(subj, shell, mode, 0.5, vox)
        c = shell[b]
        assert torch.equal(out[c == 0], subj[c == 0])  # the brain is never touched
        assert not out[c < 0].any()
        face = out[c > 0]
        # the shell's own face, scaled by the gain and blurred inside itself
        assert float(face.min()) >= 0.5 * 100 - 1e-3 and float(face.max()) <= 0.5 * 300 + 1e-3


def test_cli_output_names_follow_afni():
    from fastfuncstuff.cli.util_reface import _output_paths, parse_args

    one = parse_args(["-input", "t1.nii.gz", "-prefix", "out/sub_reface", "-mode_reface_plus"])
    assert _output_paths(one) == {
        "reface_plus": "out/sub_reface.nii.gz",
        "face_plus": "out/sub_reface.face_plus.nii.gz",
    }
    allm = parse_args(["-input", "t1.nii", "-prefix", "o.nii.zst", "-mode_all", "-no_face"])
    assert _output_paths(allm) == {
        "deface": "o.deface.nii.zst",
        "reface": "o.reface.nii.zst",
        "reface_plus": "o.reface_plus.nii.zst",
    }


def test_cli_requires_a_mode():
    from fastfuncstuff.cli.util_reface import _validate, parse_args

    args = parse_args(["-input", "t1.nii.gz", "-prefix", "o"])
    assert isinstance(args, argparse.Namespace)
    with pytest.raises(SystemExit, match="mode"):
        _validate(args)
