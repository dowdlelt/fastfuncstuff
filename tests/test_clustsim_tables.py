"""Reading ClustSim tables back out of a dataset that carries them.

A simulated table is only worth the hours it costs if something reads it
afterwards. These run the writer's output back through the reader, because the
two are the halves of one format and the way this breaks is that they drift --
at which point every corrected p silently becomes "none available".
"""

from __future__ import annotations

import numpy as np

from fastfuncstuff.stats.clustsim import (
    ClustSimTable,
    parse_clustsim_niml,
    read_clustsim_tables,
)
from fastfuncstuff.stats.niml import write_clustsim_niml

PTHR = (0.01, 0.005, 0.002, 0.001)
ATHR = (0.10, 0.05, 0.02, 0.01)


def _table():
    """A physically-shaped table: a stricter alpha demands a bigger cluster.

    ATHR runs 0.10 -> 0.01, so the row has to *rise* across it. Getting that
    backwards is exactly the mistake that makes every reported alpha wrong in
    the direction nobody checks.
    """
    return np.array([[22 + 6 * j + 5 * i for j in range(4)] for i in range(4)], dtype=float)


def _written(tmp_path, nn=1, sidedness="2-sided"):
    path = tmp_path / f"cs.NN{nn}.niml"
    write_clustsim_niml(
        path,
        _table(),
        nn=nn,
        sidedness=sidedness,
        commandline="ffs_clustsim -test",
        nxyz=(64, 64, 32),
        dxyz=(3.0, 3.0, 3.0),
        pthr=PTHR,
        athr=ATHR,
        n_perms=10000,
        mask_count=12345,
    )
    return path


def test_a_written_table_reads_back_identically(tmp_path):
    parsed = parse_clustsim_niml(_written(tmp_path).read_text())
    assert parsed is not None
    assert parsed.nn == 1
    assert parsed.sidedness == "2-sided"
    assert parsed.pthr == PTHR
    assert parsed.athr == ATHR
    assert parsed.n_iter == 10000
    assert np.array_equal(parsed.sizes, _table())


def test_it_survives_the_escaping_a_nifti_extension_adds(tmp_path):
    """The same table lives unescaped in a .niml file and escaped in a header."""
    import html

    raw = _written(tmp_path).read_text()
    assert parse_clustsim_niml(html.escape(raw)) is not None


def test_a_cluster_is_given_the_alpha_it_actually_earned(tmp_path):
    parsed = parse_clustsim_niml(_written(tmp_path).read_text())
    # Row for p=0.01: sizes 22, 28, 34, 40 at alpha 0.10, 0.05, 0.02, 0.01.
    assert parsed.size_for(0.01, 0.05) == 28.0
    assert parsed.alpha_for(0.01, 28) == 0.05
    # Between two tabulated sizes, between the two alphas.
    assert 0.02 < parsed.alpha_for(0.01, 31) < 0.05


def test_a_cluster_off_either_end_gets_a_bound_and_not_a_fitted_number(tmp_path):
    """Past the last row the truth is an inequality, so it is clamped.

    Extrapolating would invent an alpha from a curve fitted to nothing, and it
    would look exactly like a real one. alpha_range is what lets the caller
    print the "<" or the ">".
    """
    parsed = parse_clustsim_niml(_written(tmp_path).read_text())
    strictest, loosest = parsed.alpha_range
    assert (strictest, loosest) == (0.01, 0.10)
    assert parsed.alpha_for(0.01, 10_000) == strictest  # bigger than anything simulated
    assert parsed.alpha_for(0.01, 3) == loosest  # smaller than anything that survived


def test_a_dataset_without_tables_says_so_rather_than_guessing():
    class _Bare:
        extensions = ()

    assert read_clustsim_tables(_Bare()) == {}


def test_tables_are_found_by_nn_and_sidedness(tmp_path):
    """Three NNs times three sidednesses live in one header; the wrong one is
    a different cluster-size threshold, so the key has to be exact."""
    import nibabel as nib

    from fastfuncstuff.io.afni import set_afni_atr

    img = nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4))
    for nn, tag, sided in ((1, "1sided", "1-sided"), (2, "bisided", "bi-sided")):
        content = _written(tmp_path, nn=nn, sidedness=sided).read_text().rstrip()
        set_afni_atr(img.header, f"AFNI_CLUSTSIM_NN{nn}_{tag}", content, ni_type="String")

    found = read_clustsim_tables(img.header)
    assert set(found) == {(1, "1-sided"), (2, "bi-sided")}
    assert isinstance(found[(1, "1-sided")], ClustSimTable)
    assert found[(2, "bi-sided")].nn == 2
