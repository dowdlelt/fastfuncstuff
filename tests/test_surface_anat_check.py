"""check_anat: does the subject's ribbon actually sit on this anatomical volume?

The bug this guards is silent. -surface_target needs the chain to end in the
anatomy the surfaces were built on, and no header can say whether it does:
FreeSurfer's conform preserves scanner RAS, so the recon's volumes and the T1w it
ran on agree exactly, while a T1w of another session has an equally valid scanner
RAS that means a differently placed head. Only the data tells them apart.
"""

from __future__ import annotations

import nibabel as nib
import nibabel.freesurfer as nfs
import numpy as np

from fastfuncstuff.processing.surface_projection import (
    MIN_RIBBON_INSIDE,
    MIN_RIBBON_SEPARABILITY,
    check_anat,
)
from tests.test_io_freesurfer import CRAS, _volume_info
from tests.test_surface_voxelize import icosphere

R_WHITE, THICK = 14.0, 5.0
WM, GM = 110.0, 60.0
FOLD = 2.5  # amplitude of the corrugation standing in for gyri and sulci
# Angular frequencies putting the folds ~8 mm apart at this radius, comparable to
# the ribbon's own thickness as real cortex is. A long-wavelength corrugation is no
# better than a smooth sphere: a few mm of error stays on one bank and the depth
# ordering survives it.
FOLD_K, FOLD_M = 10.0, 8.0


def _white_radius(unit: np.ndarray) -> np.ndarray:
    """White-surface radius per direction: a corrugated sphere."""
    theta = np.arctan2(unit[:, 1], unit[:, 0])
    phi = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    return R_WHITE + FOLD * np.sin(FOLD_K * theta) * np.sin(FOLD_M * phi)


def _subject(tmp_path):
    surf = tmp_path / "subj" / "surf"
    surf.mkdir(parents=True)
    v, f = icosphere(5)  # ~1 mm edges: fine enough to carry the folds
    rw = _white_radius(v)[:, None]
    for hemi, dx in (("lh", -24.0), ("rh", 24.0)):
        c = np.array([dx, 0.0, 0.0])
        for name, r in (("white", rw), ("pial", rw + THICK)):
            nfs.write_geometry(
                str(surf / f"{hemi}.{name}"), (v * r + c).astype(np.float32), f.astype(np.int32),
                volume_info=_volume_info(),
            )  # fmt: skip
    return tmp_path / "subj"


def _anat(tmp_path, shift=(0.0, 0.0, 0.0), name="anat.nii.gz", shape=(96, 48, 48), scale=1.0):
    """A T1w in miniature: WM at the white surface, ramping to GM across the
    ribbon, 0 beyond the pial, plus noise.

    The ramp is the point. A razor-sharp WM/GM step reads the same tissue at every
    depth inside the ribbon and makes the measure blind; real data has a gradient
    there, which is what reading at 0.1 vs 0.9 of the depth resolves. The noise
    sets the IQR the separation is measured against.
    """
    # The content is laid out from the TRUE origin and the header is told the
    # shifted one: the head stays where it is and the affine lies about it. Moving
    # both (the obvious way to write this) moves nothing at all in world space.
    origin = CRAS - np.array([48.0, 24.0, 24.0])
    affine = np.eye(4)
    affine[:3, 3] = origin + np.asarray(shift)
    ijk = np.indices(shape).reshape(3, -1).T.astype(np.float64)
    xyz = ijk + origin
    vol = np.zeros(len(xyz))
    for dx in (-24.0, 24.0):
        d = xyz - (CRAS + np.array([dx, 0.0, 0.0]))
        r = np.linalg.norm(d, axis=1)
        rw = _white_radius(d / np.maximum(r, 1e-9)[:, None])
        ramp = np.clip((r - rw) / THICK, 0.0, 1.0)
        vol = np.where(r <= rw + THICK, WM + (GM - WM) * ramp, vol)
    vol += np.random.default_rng(0).normal(0.0, 8.0, vol.shape)
    path = tmp_path / name
    nib.save(nib.Nifti1Image((vol * scale).reshape(shape).astype(np.float32), affine), str(path))
    return path


def test_the_volume_the_surfaces_were_built_on_passes(tmp_path):
    chk = check_anat(_subject(tmp_path), _anat(tmp_path), stride=1)
    assert chk.ok, chk.report()
    assert chk.inside == 1.0
    # White reads WM, pial-side reads GM, beyond the pial reads background.
    assert chk.profile[0.1][1] > chk.profile[0.9][1] > chk.profile[1.3][1]


def test_a_header_that_misplaces_the_head_is_caught(tmp_path):
    """Measured on a real 0.8 mm recon, separability ran 1.3-2.2 placed correctly
    and 0.07-0.10 at 5 mm of offset -- a 20x gap, which is what sets the
    threshold. This phantom is far more forgiving than a brain and needs ~15 mm:
    its intensity is almost a function of distance from one centre, and a rigid
    error maps radius to radius smoothly, so the gradient partly survives. Real
    cortex is a 2.5 mm folded sheet where 5 mm crosses into WM on one bank and CSF
    on the other. So this asserts the mechanism, not the real-data sensitivity.
    """
    subj = _subject(tmp_path)
    off = check_anat(subj, _anat(tmp_path, shift=(15.0, 0.0, 0.0), name="off.nii.gz"), stride=1)
    assert off.separability < MIN_RIBBON_SEPARABILITY
    assert not off.ok
    good = check_anat(subj, _anat(tmp_path), stride=1)
    assert good.separability > 10 * off.separability


def test_inverted_contrast_passes_on_the_same_number(tmp_path):
    """A T2w anat is a legitimate target, so the measure must not assume which way
    up the tissue contrast is -- only ``profile`` says that."""
    subj = _subject(tmp_path)
    t1 = check_anat(subj, _anat(tmp_path), stride=1)
    img = nib.load(str(_anat(tmp_path, name="t2.nii.gz")))
    flipped = WM + GM - np.asarray(img.dataobj)
    nib.save(nib.Nifti1Image(flipped.astype(np.float32), img.affine), str(tmp_path / "t2.nii.gz"))
    t2 = check_anat(subj, tmp_path / "t2.nii.gz", stride=1)
    assert t2.ok
    assert t2.profile[0.1][1] < t2.profile[0.9][1]  # inverted, as a T2w is
    np.testing.assert_allclose(t2.separability, t1.separability, rtol=0.05)


def test_the_measure_does_not_move_with_the_intensity_scale(tmp_path):
    """Separability is a ratio of medians to IQR, so an anat in different units is
    the same anat. An absolute threshold here would fail on any real scanner."""
    subj = _subject(tmp_path)
    plain = check_anat(subj, _anat(tmp_path), stride=1)
    scaled = check_anat(subj, _anat(tmp_path, name="big.nii.gz", scale=1000.0), stride=1)
    np.testing.assert_allclose(scaled.separability, plain.separability, rtol=1e-3)


def test_an_anat_that_crops_the_cortex_fails_on_containment(tmp_path):
    """Separability can still be fine on the part that is in frame, so this is a
    second, independent condition rather than a refinement of the first."""
    chk = check_anat(_subject(tmp_path), _anat(tmp_path, shape=(40, 48, 48)), stride=1)
    assert chk.inside < MIN_RIBBON_INSIDE
    assert not chk.ok
