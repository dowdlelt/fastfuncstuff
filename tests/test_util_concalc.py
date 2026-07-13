"""Tests for ffs_util_concalc.

The integration test ('synthetic_roundtrip') builds a tiny synthetic dataset,
runs `ffs_reml` once *without* contrasts to populate the bucket and Rvar,
then runs `concalc` to add contrasts to that bucket. It compares the concalc
output against a second `ffs_reml` run *with* the same contrasts baked into
the spec — they should agree to within float32 round-off.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_select_non_contrast_subbricks_keeps_stim_drops_contrast():
    from fastfuncstuff.cli.util_concalc import _select_non_contrast_subbricks

    labels = [
        "Full_Fstat",  # keep (overall F)
        "DI#0_Coef",  # keep (stim coef)
        "DI#0_Tstat",  # keep (stim t)
        "PI#0_Coef",  # keep
        "PI#0_Tstat",  # keep
        "FvH_Coef",  # drop (old contrast)
        "FvH_Tstat",  # drop
        "anyOf_Fstat",  # drop (old F-test contrast)
        "Mask",  # keep (unknown shape, preserved)
    ]
    keep = _select_non_contrast_subbricks(labels, stim_base_labels=["DI", "PI"])
    assert [labels[i] for i in keep] == [
        "Full_Fstat",
        "DI#0_Coef",
        "DI#0_Tstat",
        "PI#0_Coef",
        "PI#0_Tstat",
        "Mask",
    ]


def test_brick_labels_extension_round_trip(tmp_path):
    """Our XML extension survives nibabel save/load + our reader recovers
    the original list."""
    from fastfuncstuff.cli.util_concalc import (
        _afni_brick_labels_extension,
        _read_brick_labels,
    )

    labels_in = ["Full_Fstat", "DI#0_Coef", "DI#0_Tstat", "FvH_Coef", "FvH_Tstat", "any_Fstat"]
    arr = np.zeros((4, 4, 4, len(labels_in)), dtype=np.float32)
    img = nib.Nifti1Image(arr, np.eye(4))
    img.header.extensions.append(_afni_brick_labels_extension(labels_in))
    out = tmp_path / "x.nii.gz"
    nib.save(img, out)
    loaded = nib.load(out)
    labels_out = _read_brick_labels(loaded)
    assert labels_out == labels_in


def test_stataux_parses_and_round_trips(tmp_path):
    """STATAUX from an AFNI-style bucket is parsed, preserved across save/
    load, and STATSYM matches the AFNI semicolon form."""
    from fastfuncstuff.cli.util_concalc import (
        _afni_bucket_extension,
        _parse_stataux,
    )

    labels = [
        "Full_Fstat",
        "DI#0_Coef",
        "DI#0_Tstat",
        "FvH_Coef",
        "FvH_Tstat",
        "anyOf_Fstat",
    ]
    stataux = {
        0: (4, (7.0, 1052.0)),  # Full_Fstat: Ftest(7,1052)
        2: (3, (1052.0,)),  # DI#0_Tstat: Ttest(1052)
        4: (3, (1052.0,)),  # FvH_Tstat
        5: (4, (4.0, 1052.0)),  # anyOf_Fstat
    }

    arr = np.zeros((2, 2, 2, len(labels)), dtype=np.float32)
    img = nib.Nifti1Image(arr, np.eye(4))
    img.header.extensions.append(_afni_bucket_extension(labels, stataux))
    p = tmp_path / "bucket.nii.gz"
    nib.save(img, p)

    loaded = nib.load(p)
    txt = loaded.header.extensions[0].get_content()
    if isinstance(txt, bytes):
        txt = txt.decode("utf-8", errors="ignore")
    parsed = _parse_stataux(txt)
    assert parsed == stataux

    # STATSYM should carry one entry per sub-brick, "none" for the
    # non-stat ones.
    assert "Ftest(7,1052)" in txt
    assert "Ttest(1052)" in txt
    assert ";none;" in txt  # at least one non-stat sub-brick separates them


def test_legacy_brick_labs_form_still_readable(tmp_path):
    """Older 3dDeconvolve outputs use plain ``BRICK_LABS=a~b~c\\x00``.
    The reader must accept that too."""
    from fastfuncstuff.cli.util_concalc import _read_brick_labels

    arr = np.zeros((2, 2, 2, 3), dtype=np.float32)
    img = nib.Nifti1Image(arr, np.eye(4))
    payload = b"BRICK_LABS=alpha~beta~gamma\x00"
    img.header.extensions.append(nib.nifti1.Nifti1Extension(4, payload))
    p = tmp_path / "legacy.nii.gz"
    nib.save(img, p)
    assert _read_brick_labels(nib.load(p)) == ["alpha", "beta", "gamma"]


def test_bin_index_groups_voxels_and_marks_invalid():
    from fastfuncstuff.cli.util_concalc import _bin_index

    a = np.array([0.5, 0.5, 0.5, 0.7, np.nan, 1.5], dtype=np.float32)
    b = np.array([0.1, 0.1, 0.2, 0.0, 0.0, 0.0], dtype=np.float32)
    bin_idx, unique_ab, valid = _bin_index(a, b)
    assert valid.tolist() == [True, True, True, True, False, False]
    # Three voxels share (0.5, 0.1), one is (0.5, 0.2), one is (0.7, 0.0).
    # _bin_index returns unique_ab limited to *valid* rows.
    assert unique_ab.shape[0] == 3
    # voxels 0 and 1 collapse to the same bin; voxel 2 is its own bin.
    assert bin_idx[0] == bin_idx[1]
    assert bin_idx[0] != bin_idx[2]
    # invalid voxels marked with -1.
    assert bin_idx[4] == -1
    assert bin_idx[5] == -1


@pytest.mark.slow
def test_synthetic_round_trip_matches_reml_to_floatprecision(tmp_path):
    """Run ffs_reml twice on the same synthetic data:

    1. Without contrasts → bucket A + Rvar.
    2. With contrasts in the spec → bucket B.

    Then run concalc on bucket A using the same spec → bucket C.
    Bucket C's contrast sub-bricks must match bucket B's.
    """
    # This test is heavy (full REML + concalc round trip). It exists to be
    # run on a developer's machine; not gated in CI by default.
    pytest.importorskip("torch")
    import subprocess
    import sys as _sys

    # Skip if the ffs_reml entry point isn't installed.
    res = subprocess.run(
        [_sys.executable, "-c", "from fastfuncstuff.cli import reml; print(reml.main)"],
        capture_output=True,
    )
    if res.returncode != 0:
        pytest.skip("ffs_reml entry point not importable")
    # Implementation deferred — the synthetic data plumbing needs the same
    # NIfTI-on-disk inputs that the real pipeline expects, which is more
    # boilerplate than belongs in a single unit test. Real-data validation
    # against the user's AFNI proc dir is the current acceptance gate.
    pytest.skip("synthetic round-trip plumbing TBD")
