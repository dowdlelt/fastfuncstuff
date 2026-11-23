"""Deface / reface a T1w through a template-space face shell (AFNI ``@afni_refacer_run``).

The recipe is AFNI's, step for step: align the head to the MNI template, carry
the refacer shell back onto the subject grid through the inverse, and then either
zero what the shell marks (deface) or paint the shell's own synthetic face in its
place, scaled to the subject's brightness and lightly blurred (reface /
reface_plus). The shell is three-valued: ``-1`` outside the head, ``0`` over the
brain (keep the subject), ``> 0`` over the face/skull to replace. Sub-brick 0
(``refacer``) covers face + ears + skull, sub-brick 1 (``defacer``) face + ears.

All of AFNI's run time is its affine search (379 s of ~7 min on a 1 mm T1); this
module hands that to :func:`fastfuncstuff.processing.allineate.allineate` and
does the rest on the same device.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

REFACE_MODES = ("deface", "reface", "reface_plus")
TEMPLATE_NAME = "MNI152_2009_template_SSW.nii.gz"
SHELL_NAME = "afni_refacer_shell_sym_1.0.nii.gz"
TEMPLATE_HEAD_BRICK = 1  # SKon: the template with the skull on
TEMPLATE_WEIGHT_BRICK = 2  # SKweight
# Which shell sub-brick each mode reads.
SHELL_BRICK = {"reface_plus": 0, "reface": 1, "deface": 1}
# @afni_refacer_run's constants: the shell is scaled by GAIN * (subject mean /
# shell mean) over shell voxels above THRESHOLD, then blurred at BLUR_FWHM_MM.
SHELL_GAIN = 1.222
SHELL_THRESHOLD = 99.0
BLUR_FWHM_MM = 2.666


def _conv_count(ind: Tensor, kernel: Tensor) -> Tensor:
    """Correlate a 0/1 volume with an integer kernel, no padding (exact in float32)."""
    return F.conv3d(ind[None, None], kernel[None, None].to(ind))[0, 0]


def _isola_stencil(device: torch.device) -> Tensor:
    """The 27-neighbourhood ``3dcalc -ISOLA`` actually counts, as (dk, dj, di) weights.

    Not a plain 3x3x3: ``remove_isolated_stuff`` (3dcalc.c) lists the centre
    plane's ``j-1`` row twice and its ``j+1`` row never, and in the ``k+1`` plane
    the ``j+1`` row twice and the ``j-1`` row never. Reproduced, so a voxel is
    kept or dropped exactly when AFNI's would be.
    """
    w = torch.ones(3, 3, 3, device=device)
    w[1, 0, :], w[1, 2, :] = 2.0, 0.0
    w[2, 0, :], w[2, 2, :] = 0.0, 2.0
    return w


def remove_isolas(vol: Tensor, iterations: int = 1) -> Tensor:
    """``3dcalc -ISOLA``: zero non-zero voxels with too few non-zero neighbours.

    ``vol`` is ``(nz, ny, nx)``. Per pass, an interior voxel needs 4 non-zero voxels
    in its 3x3x3 neighbourhood (itself included), and every voxel needs 2 in the
    3x3 patch of each plane through it (edges of that plane excepted). All counts
    are taken on the volume as it stood at the start of the pass. ``-ISOLA`` is one
    pass; ``iterations`` repeats until nothing changes.
    """
    if sum(n < 5 for n in vol.shape) > 1:
        return vol
    out = vol
    for _ in range(iterations):
        ind = (out != 0).to(torch.float32)
        drop = torch.zeros_like(ind, dtype=torch.bool)
        inner = drop[1:-1, 1:-1, 1:-1]
        inner |= (ind[1:-1, 1:-1, 1:-1] > 0) & (_conv_count(ind, _isola_stencil(vol.device)) < 4)
        ones = torch.ones(3, 3, device=vol.device)
        # Each plane: (kernel shape, the slab of voxels it is evaluated on)
        planes = (
            (ones[None, :, :], (slice(None), slice(1, -1), slice(1, -1))),  # k fixed
            (ones[:, None, :], (slice(1, -1), slice(None), slice(1, -1))),  # j fixed
            (ones[:, :, None], (slice(1, -1), slice(1, -1), slice(None))),  # i fixed
        )
        for kernel, where in planes:
            drop[where] |= (ind[where] > 0) & (_conv_count(ind, kernel) < 2)
        if not bool(drop.any()):
            break
        out = out.masked_fill(drop, 0.0)
    return out


def remove_isolated_mask_voxels(mask: Tensor) -> Tensor:
    """``THD_mask_remove_isolas``: drop mask voxels with no face neighbour (6-conn)."""
    cross = torch.zeros(3, 3, 3, device=mask.device)
    cross[1, 1, 0] = cross[1, 1, 2] = cross[1, 0, 1] = cross[1, 2, 1] = 1.0
    cross[0, 1, 1] = cross[2, 1, 1] = 1.0
    ind = mask.to(torch.float32)
    nbrs = F.conv3d(ind[None, None], cross[None, None], padding=1)[0, 0]
    return mask & (nbrs > 0)


def shell_on_subject(
    shell: Tensor,
    shell_affine: np.ndarray,
    template_to_subject_dicom: Tensor,
    subject_affine: np.ndarray,
    subject_shape: tuple[int, int, int],
) -> Tensor:
    """The ``(B, nz, ny, nx)`` shell carried onto the subject grid, nearest neighbour.

    ``template_to_subject_dicom`` is the alignment as an ``.aff12.1D`` holds it
    (base = template, source = subject, AFNI DICOM mm). Nearest neighbour, as
    @afni_refacer_run uses: linear leaves a gap between the shell and the brain.
    Each sub-brick then gets one ``-ISOLA`` pass, as AFNI applies to it.
    """
    from fastfuncstuff.processing.affine import dicom_matrix_to_voxel
    from fastfuncstuff.processing.grid import resample_to_grid

    inv = torch.linalg.inv(template_to_subject_dicom.cpu().to(torch.float64)).to(torch.float32)
    # subject voxel -> shell voxel; resample_to_grid takes it as a header pair
    vox = dicom_matrix_to_voxel(inv, subject_affine, shell_affine).numpy()
    subject_affine = np.asarray(subject_affine, dtype=np.float64)
    pseudo = subject_affine @ np.linalg.inv(vox)
    moved = resample_to_grid(shell, pseudo, subject_shape, subject_affine, interp="nearest")
    if moved.ndim == 3:
        moved = moved[None]
    return torch.stack([remove_isolas(b) for b in moved])


def shell_gain(subject: Tensor, shell_subj: Tensor) -> float:
    """@afni_refacer_run's ``ifac``: ``1.222 * ibar / mbar``.

    ``ibar`` is the mean subject intensity over shell voxels above 99 (non-zero
    products only, as ``3dBrickStat -non-zero``), ``mbar`` the shell's own mean
    there; both pool every shell sub-brick, as the script's 3dBrickStat call does.
    """
    sel = shell_subj > SHELL_THRESHOLD
    prod = sel * subject
    hit = prod != 0
    if not bool(hit.any()):
        raise ValueError("the refacer shell lands on no subject voxels; is the alignment wrong?")
    ibar = prod[hit].double().mean()
    mbar = shell_subj[sel].double().mean()
    return float(SHELL_GAIN * ibar / mbar)


def apply_shell(
    subject: Tensor,
    shell_subj: Tensor,
    mode: str,
    gain: float,
    voxel_size_mm: tuple[float, float, float],
) -> Tensor:
    """One deface/reface output from the subject and its on-grid shell.

    ``voxel_size_mm`` is in tensor order ``(z, y, x)``.
    """
    from fastfuncstuff.stats.smooth3d import blur_in_mask

    if mode not in REFACE_MODES:
        raise ValueError(f"unknown mode {mode!r}; one of {REFACE_MODES}")
    c = shell_subj[SHELL_BRICK[mode]]
    if mode == "deface":
        # a*not(bool(c)): the outside (-1) goes too, not only the face.
        return remove_isolas(subject * (c == 0))
    # step(c)*c*ifac + iszero(c)*a*step(a)
    out = torch.where(c > 0, c * gain, torch.where(c == 0, subject.clamp_min(0.0), 0.0))
    out = remove_isolas(out)
    mask = remove_isolated_mask_voxels(c > 0)
    return blur_in_mask(out[None], mask, BLUR_FWHM_MM, voxel_size_mm)[0]
