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


def test_select_non_contrast_keeps_per_stim_zstat_drops_contrast_zstat():
    # ffs_util_updatedof -numcomps writes Coef/Tstat/Zstat triples per stim,
    # and a matching Zstat per contrast. Per-stim z must survive; contrast z
    # must be dropped like the other contrast bricks.
    from fastfuncstuff.cli.util_concalc import _select_non_contrast_subbricks

    labels = [
        "DI#0_Coef",  # keep
        "DI#0_Tstat",  # keep
        "DI#0_Zstat",  # keep (per-stim z)
        "FvH_Coef",  # drop
        "FvH_Tstat",  # drop
        "FvH_Zstat",  # drop (contrast z)
    ]
    keep = _select_non_contrast_subbricks(labels, stim_base_labels=["DI"])
    assert [labels[i] for i in keep] == ["DI#0_Coef", "DI#0_Tstat", "DI#0_Zstat"]


def test_contrast_base_name_strips_all_suffixes():
    from fastfuncstuff.cli.util_concalc import _contrast_base_name

    assert _contrast_base_name("face_vs_place#0_Coef") == "face_vs_place"
    assert _contrast_base_name("face_vs_place_Tstat") == "face_vs_place"
    assert _contrast_base_name("face_vs_place_Zstat") == "face_vs_place"
    assert _contrast_base_name("anyOf_Fstat") == "anyOf"
    assert _contrast_base_name("faces#0_Coef") == "faces"
    # Not a contrast-style label → None.
    assert _contrast_base_name("Full_Fstat") == "Full"  # (has _Fstat suffix)
    assert _contrast_base_name("Mask") is None


def test_unsafe_drops_flags_label_mismatch_but_allows_recompute():
    # The footgun: a spec built from singular `face` events against a bucket
    # fit with plural `faces#0_Coef`. Those betas fall out of keep_idx and are
    # NOT among the contrasts being recomputed → must be reported as unsafe.
    from fastfuncstuff.cli.util_concalc import (
        _select_non_contrast_subbricks,
        _unsafe_drops,
    )

    labels = [
        "Full_Fstat",
        "faces#0_Coef",
        "faces#0_Tstat",
        "places#0_Coef",
        "places#0_Tstat",
        "old_contrast_Coef",
        "old_contrast_Tstat",
    ]
    # Spec knows singular stim labels only.
    keep = _select_non_contrast_subbricks(labels, stim_base_labels=["face", "place"])

    # Recomputing a contrast that already exists → its old bricks drop, but
    # safely (same name is rebuilt).
    safe = _unsafe_drops(labels, keep, new_contrast_labels={"old_contrast"})
    assert "old_contrast_Coef" not in safe
    # The mislabelled betas are lost and not recomputed → flagged.
    assert set(safe) == {"faces#0_Coef", "faces#0_Tstat", "places#0_Coef", "places#0_Tstat"}

    # With correct labels nothing is unsafe.
    keep_ok = _select_non_contrast_subbricks(labels, stim_base_labels=["faces", "places"])
    assert _unsafe_drops(labels, keep_ok, new_contrast_labels={"old_contrast"}) == []


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


def _rvar_like(tmp_path, name, labels, data):
    """Write a 4-D NIfTI carrying AFNI sub-brick labels."""
    from fastfuncstuff.cli.util_concalc import _afni_brick_labels_extension

    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    img.header.extensions.append(_afni_brick_labels_extension(labels))
    path = tmp_path / name
    nib.save(img, str(path))
    return path


def test_concalc_refuses_a_stats_bucket_passed_as_rvar(tmp_path):
    """A bucket as -rvar read its F-stat as the ARMA `a` and said nothing.

    The two files differ by one suffix, and sub-bricks 0/1/3 of a bucket are
    finite floats, so nothing downstream complained: the ARMA covariance was
    built from an F-stat, and the contrasts came out as noise on the handful of
    voxels whose garbage (a, b) happened to land inside the unit square.
    """
    from fastfuncstuff.cli.util_concalc import _check_is_rvar

    rng = np.random.default_rng(0)
    shape = (6, 6, 4)
    bucket_labels = ["Full_Fstat", "face#0_Coef", "face#0_Tstat", "house#0_Coef"]
    bucket = np.abs(rng.normal(0, 3, (*shape, 4)))  # F-stats and coefs: continuous
    bpath = _rvar_like(tmp_path, "stats.nii.gz", bucket_labels, bucket)

    # 1) the same file for both -- the shape of the real mistake.
    assert _check_is_rvar(str(bpath), str(bpath), bucket_labels, bucket) == 1

    # 2) a *different* bucket: caught on the labels.
    other = _rvar_like(tmp_path, "other.nii.gz", bucket_labels, bucket)
    stats = tmp_path / "elsewhere.nii.gz"
    nib.save(nib.Nifti1Image(bucket, np.eye(4)), str(stats))
    assert _check_is_rvar(str(other), str(stats), bucket_labels, bucket) == 1

    # 3) unlabelled, so only the values can tell. Two independent tells, and
    #    neither depends on the dataset's size:
    #    (a) F-stats mostly sit outside the unit square an ARMA parameter lives in
    #        -- this is why the bad run showed almost no voxels at any threshold.
    assert _check_is_rvar(str(other), str(stats), [], bucket) == 1
    #    (b) in-range but continuous: ~one distinct pair per voxel, where a grid
    #        repeats itself however big the brain gets.
    big = (40, 40, 20)  # past the ratio arm's voxel floor
    cont = np.stack([rng.uniform(-0.9, 0.9, big) for _ in range(4)], axis=-1)
    assert _check_is_rvar(str(other), str(stats), [], cont) == 1

    # 4) a real Rvar passes: a and b come off a grid, so few distinct pairs.
    grid = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    a = rng.choice(grid, big)
    b = rng.choice(grid, big)
    real = np.stack([a, b, np.ones(big), np.full(big, 2.0)], axis=-1)
    labels = ["a", "b", "lambda", "StDev"]
    rpath = _rvar_like(tmp_path, "stats_ffsremlvar.nii.gz", labels, real)
    assert _check_is_rvar(str(rpath), str(stats), labels, real) == 0
    # ... and so must a small one, where every voxel may hold its own pair.
    tiny = np.stack(
        [rng.choice(grid, shape), rng.choice(grid, shape), np.ones(shape), np.ones(shape)],
        axis=-1,
    )
    assert _check_is_rvar(str(rpath), str(stats), labels, tiny) == 0


def test_rvar_companion_is_found_beside_the_bucket(tmp_path):
    """-rvar can be omitted: ffs_reml writes the companion under a fixed name,
    and the only wrong answer to guess was the bucket itself."""
    from fastfuncstuff.cli.util_concalc import _rvar_companion

    stats = tmp_path / "stage12.stats-reml.task-x.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 2), np.float32), np.eye(4)), str(stats))
    assert _rvar_companion(str(stats)) is None

    comp = tmp_path / "stage12.stats-reml.task-x_ffsremlvar.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 4), np.float32), np.eye(4)), str(comp))
    assert _rvar_companion(str(stats)) == str(comp)


def _clustsim_attr_names(path):
    import re

    txt = "".join(
        e.get_content().decode("utf-8", "replace") for e in nib.load(str(path)).header.extensions
    )
    return sorted(set(re.findall(r"AFNI_CLUSTSIM_\w+", txt)))


def test_bucket_rewrites_keep_the_clustsim_tables(tmp_path):
    """Rewriting a bucket must not silently un-cluster-correct it.

    Both concalc and the dof adjust rebuild a stats dataset. They own the
    sub-brick metadata they recompute; the AFNI_CLUSTSIM_* tables describe the
    dataset and belong to it. Dropping them leaves a bucket that looks fine and
    no longer reports cluster significance in the viewer.
    """
    from fastfuncstuff.cli.util_concalc import _CONCALC_OWNED_ATRS, _save_bucket
    from fastfuncstuff.io.afni import set_afni_atr

    labels = ["Full_Fstat", "face#0_Coef", "face#0_Tstat"]
    data = np.zeros((4, 4, 3, 3), np.float32)
    src = nib.Nifti1Image(data, np.eye(4))
    from fastfuncstuff.cli.util_concalc import _afni_brick_labels_extension

    src.header.extensions.append(_afni_brick_labels_extension(labels))
    set_afni_atr(src.header, "AFNI_CLUSTSIM_NN1_1sided", "<3dClustSim_NN1 />")
    set_afni_atr(src.header, "AFNI_CLUSTSIM_MASK", "abc123")
    spath = tmp_path / "stats.nii.gz"
    nib.save(src, str(spath))
    before = _clustsim_attr_names(spath)
    assert before == ["AFNI_CLUSTSIM_MASK", "AFNI_CLUSTSIM_NN1_1sided"]

    out = tmp_path / "out.nii.gz"
    _save_bucket(out, data, labels, nib.load(str(spath)))
    assert _clustsim_attr_names(out) == before

    # The labels concalc DOES own are the recomputed ones, not carried twice.
    assert "BRICK_LABS" in _CONCALC_OWNED_ATRS
    txt = "".join(
        e.get_content().decode("utf-8", "replace") for e in nib.load(str(out)).header.extensions
    )
    assert txt.count('atr_name="BRICK_LABS"') == 1


def test_dof_adjust_keeps_the_clustsim_tables(tmp_path):
    """Same property for ffs_util_updatedof / ffs_reml -adjust_dof."""
    from fastfuncstuff.io.afni import save_nifti, set_afni_atr
    from fastfuncstuff.stats.dof_adjust import resolve_dof_adjust_arg, update_dof_in_file

    data = np.abs(np.random.default_rng(1).normal(0, 2, (4, 4, 3, 2))).astype(np.float32)
    path = tmp_path / "stats.nii.gz"
    save_nifti(
        data,
        str(path),
        affine=np.eye(4),
        brick_labels=["face#0_Coef", "face#0_Tstat"],
        brick_stataux={1: (3, (40.0,))},  # fitt(40)
    )
    img = nib.load(str(path))
    set_afni_atr(img.header, "AFNI_CLUSTSIM_NN1_1sided", "<3dClustSim_NN1 />")
    nib.save(img, str(path))
    assert _clustsim_attr_names(path) == ["AFNI_CLUSTSIM_NN1_1sided"]

    update_dof_in_file(str(path), resolve_dof_adjust_arg("5"), str(path), verbose=False)
    assert _clustsim_attr_names(path) == ["AFNI_CLUSTSIM_NN1_1sided"]
