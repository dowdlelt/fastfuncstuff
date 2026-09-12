"""Derived layers: projecting a design's nuisance out of a functional run.

The properties worth pinning are the ones that would look plausible while being
wrong: a projection that removes signal it should not, a rank-deficient design
that quietly eats a real dimension, and an output whose scale no longer shares
an axis with the series it came from -- which would defeat the one comparison
the feature exists to make.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.derive import (
    denoise,
    legendre_columns,
    orthonormal_basis,
    read_nuisance,
)

CPU = torch.device("cpu")


def _series(nt: int = 60, shape=(4, 5, 3), seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(*shape, nt)).astype(np.float32)


def _write_xmat(path, columns, groups, labels):
    """A minimal AFNI xmat: the header ffs writes, and the numbers."""
    n_rows, n_cols = columns.shape
    with open(path, "w") as f:
        f.write("# <matrix\n")
        f.write(f'#  ni_type = "{n_cols}*double"\n')
        f.write(f'#  ni_dimen = "{n_rows}"\n')
        f.write(f'#  ColumnLabels = "{" ; ".join(labels)}"\n')
        f.write(f'#  ColumnGroups = "{",".join(str(g) for g in groups)}"\n')
        f.write("# >\n")
        for row in columns:
            f.write(" ".join(f"{v:.10g}" for v in row) + "\n")
    return path


# ---------------------------------------------------------------------------
# which columns count as nuisance
# ---------------------------------------------------------------------------


def test_a_design_contributes_only_its_non_positive_groups(tmp_path):
    """ColumnGroups already says which columns are nuisance; do not guess."""
    rng = np.random.default_rng(1)
    columns = rng.normal(size=(40, 5))
    path = _write_xmat(
        tmp_path / "X.xmat.1D",
        columns,
        groups=[-1, -1, 0, 1, 1],  # drift, drift, motion, stim, stim
        labels=["Pol#0", "Pol#1", "roll", "faces#0", "faces#1"],
    )
    nuisance = read_nuisance(path, n_time=40)
    assert nuisance.n_columns == 3
    assert nuisance.labels == ("Pol#0", "Pol#1", "roll")
    assert np.allclose(nuisance.columns, columns[:, :3])


def test_a_plain_1d_file_is_nuisance_all_the_way_down(tmp_path):
    motion = np.random.default_rng(2).normal(size=(40, 6))
    path = tmp_path / "motion.1D"
    np.savetxt(path, motion)
    nuisance = read_nuisance(path, n_time=40)
    assert nuisance.n_columns == 6
    assert np.allclose(nuisance.columns, motion)


def test_the_row_count_has_to_match_the_dataset(tmp_path):
    path = tmp_path / "motion.1D"
    np.savetxt(path, np.zeros((30, 2)))
    with pytest.raises(ValueError, match="30 rows"):
        read_nuisance(path, n_time=40)


def test_polort_alone_is_a_valid_derivation():
    nuisance = read_nuisance(None, n_time=50, polort=2)
    assert nuisance.n_columns == 3
    assert "polort 2" in nuisance.description


def test_asking_for_nothing_is_an_error():
    with pytest.raises(ValueError, match="nothing to project out"):
        read_nuisance(None, n_time=50)


# ---------------------------------------------------------------------------
# the basis
# ---------------------------------------------------------------------------


def test_duplicated_columns_collapse_rather_than_inventing_a_direction():
    """An xmat carries drift, so a requested polort duplicates it exactly.

    QR would return a second orthonormal column spanning numerical noise, and
    projecting that out removes real signal from every voxel.
    """
    poly = legendre_columns(40, 2)
    doubled = np.concatenate([poly, poly], axis=1)
    assert doubled.shape[1] == 6
    assert orthonormal_basis(doubled).shape[1] == 3


def test_the_basis_is_orthonormal():
    basis = orthonormal_basis(np.random.default_rng(4).normal(size=(40, 5)))
    assert np.allclose(basis.T @ basis, np.eye(5), atol=1e-10)


def test_an_all_zero_nuisance_matrix_is_refused():
    with pytest.raises(ValueError, match="all zeros"):
        orthonormal_basis(np.zeros((20, 3)))


# ---------------------------------------------------------------------------
# the projection
# ---------------------------------------------------------------------------


def test_the_nuisance_is_gone_from_the_result():
    data = _series()
    nuisance = read_nuisance(None, n_time=data.shape[-1], polort=3)
    out = denoise(data, nuisance, device=CPU)
    basis = orthonormal_basis(nuisance.columns)
    residual = out.reshape(-1, data.shape[-1]) - out.reshape(-1, data.shape[-1]).mean(
        -1, keepdims=True
    )
    assert np.abs(residual @ basis).max() < 1e-3


def test_signal_orthogonal_to_the_nuisance_survives_untouched():
    """The property that says the projection is not just shrinking things.

    The signal is made orthogonal to the drift basis by construction rather
    than by choosing a frequency and hoping -- a sine at a whole number of
    cycles is only *nearly* orthogonal to a quadratic once it is sampled, and a
    test that tolerates that slack would also tolerate a real leak.
    """
    nt = 64
    nuisance = read_nuisance(None, n_time=nt, polort=2)
    basis = orthonormal_basis(nuisance.columns)

    rng = np.random.default_rng(17)
    raw = rng.normal(size=nt)
    signal = (raw - basis @ (basis.T @ raw)).astype(np.float32)
    drift = (basis @ np.array([300.0, -40.0, 12.0])).astype(np.float32)

    data = np.zeros((2, 2, 1, nt), dtype=np.float32)
    data[..., :] = signal + drift
    out = denoise(data, nuisance, device=CPU)

    recovered = out[0, 0, 0] - out[0, 0, 0].mean()
    assert np.allclose(recovered, signal - signal.mean(), atol=1e-3)


def test_the_temporal_mean_is_preserved_so_the_two_share_an_axis():
    """A mean-zero 'denoised BOLD' cannot be plotted against its own source."""
    data = _series() + 1000.0
    nuisance = read_nuisance(None, n_time=data.shape[-1], polort=1)
    out = denoise(data, nuisance, device=CPU)
    assert np.allclose(out.mean(-1), data.mean(-1), atol=1e-2)


def test_without_keep_mean_the_baseline_really_goes():
    data = _series() + 1000.0
    nuisance = read_nuisance(None, n_time=data.shape[-1], polort=1)
    out = denoise(data, nuisance, device=CPU, keep_mean=False)
    assert np.abs(out.mean(-1)).max() < 1e-2


def test_chunking_does_not_change_the_answer(monkeypatch):
    data = _series(nt=40, shape=(6, 6, 4))
    nuisance = read_nuisance(None, n_time=40, polort=2)
    whole = denoise(data, nuisance, device=CPU)

    import fastfuncstuff.viewer.derive as derive_mod

    monkeypatch.setattr(derive_mod, "estimate_chunk_size", None, raising=False)
    monkeypatch.setattr(
        "fastfuncstuff.memory.estimate_chunk_size", lambda **kwargs: 7, raising=True
    )
    in_bits = denoise(data, nuisance, device=CPU)
    assert np.allclose(whole, in_bits, atol=1e-5)


def test_the_shape_and_dtype_come_back_unchanged():
    data = _series(nt=30, shape=(3, 4, 2))
    out = denoise(data, read_nuisance(None, n_time=30, polort=1), device=CPU)
    assert out.shape == data.shape
    assert out.dtype == np.float32


def test_a_three_d_dataset_is_refused():
    with pytest.raises(ValueError, match="4-D"):
        denoise(np.zeros((3, 3, 3), np.float32), read_nuisance(None, n_time=3, polort=0))


def test_a_mismatched_nuisance_is_refused():
    data = _series(nt=30)
    with pytest.raises(ValueError, match="volumes"):
        denoise(data, read_nuisance(None, n_time=25, polort=1), device=CPU)


# ---------------------------------------------------------------------------
# a derived layer is a dataset, not a mode's output
# ---------------------------------------------------------------------------


@pytest.fixture
def session(tmp_path):
    nib = pytest.importorskip("nibabel")
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.vocab import SetOverlay, SetUnderlay

    rng = np.random.default_rng(23)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    nib.save(
        nib.Nifti1Image((rng.random((6, 7, 5)) * 100).astype(np.float32), aff),
        str(tmp_path / "anat.nii.gz"),
    )
    img = nib.Nifti1Image((rng.normal(size=(6, 7, 5, 40)) + 900).astype(np.float32), aff)
    img.header["pixdim"][4] = 2.0
    img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(tmp_path / "bold.nii.gz"))

    sess = ViewerSession(device=CPU)
    try:
        sess.do(SetUnderlay(str(tmp_path / "anat.nii.gz")))
        sess.do(SetOverlay(str(tmp_path / "bold.nii.gz")))
        sess.store.ensure_ram(sess.state.layers.overlay.key)
        yield sess
    finally:
        sess.close()


def _bold(session):
    return session.state.layers.overlay.key


def test_the_derived_layer_lands_directly_above_its_source(session):
    """Neighbours are what `[`, `]` and a soloed window flip between."""
    src = _bold(session)
    session.denoise(src, polort=2)
    keys = session.state.layers.keys
    assert keys.index(session.state.layers.find_by_source(f"derived:denoise:{src}").key) == (
        keys.index(src) + 1
    )


def test_it_inherits_how_its_source_is_drawn(session):
    """Auto-scaling each to itself would hide the difference you made it to see."""
    src = _bold(session)
    session.state.layers.update(src, range_lo=880.0, range_hi=920.0, colormap="viridis")
    session.denoise(src, polort=2)
    derived = session.state.layers.find_by_source(f"derived:denoise:{src}")
    assert (derived.range_lo, derived.range_hi) == (880.0, 920.0)
    assert derived.colormap == "viridis"


def test_it_is_graphable_beside_its_source(session):
    src = _bold(session)
    session.denoise(src, polort=2)
    names = [ly.key for ly in session.graph_layers()]
    assert src in names and len(names) == 2


def test_re_deriving_replaces_rather_than_growing_the_stack(session):
    src = _bold(session)
    session.denoise(src, polort=2)
    before = len(session.state.layers)
    session.denoise(src, polort=4)
    assert len(session.state.layers) == before


def test_it_survives_a_mode_switch(session):
    """The difference from a mode's overlay, which is taken back."""
    src = _bold(session)
    session.denoise(src, polort=2)
    session.set_mode("instacorr")
    session.set_mode("plain")
    assert session.state.layers.find_by_source(f"derived:denoise:{src}") is not None


def test_it_can_become_the_underlay(session):
    from fastfuncstuff.viewer.vocab import MoveLayer

    src = _bold(session)
    session.denoise(src, polort=2)
    derived = session.state.layers.find_by_source(f"derived:denoise:{src}")
    session.do(MoveLayer(derived.key, 0))
    assert session.state.layers.base.key == derived.key


def test_a_three_d_layer_cannot_be_denoised(session):
    base = session.state.layers.base.key
    with pytest.raises(ValueError, match="not a time series"):
        session.denoise(base, polort=2)


def test_the_command_replays(session, tmp_path):
    from fastfuncstuff.viewer.vocab import Denoise

    src = _bold(session)
    dirty = session.do(Denoise(src, polort=2))
    assert dirty
    assert "DENOISE" in session.to_script()
    assert session.state.layers.find_by_source(f"derived:denoise:{src}") is not None


def test_the_provenance_is_written_where_a_path_would_be(session):
    src = _bold(session)
    session.denoise(src, polort=3)
    derived = session.state.layers.find_by_source(f"derived:denoise:{src}")
    assert "polort 3" in derived.path
