"""The data selector and the mode framework.

The selector is the viewer's core, so the invariants here are the ones that
would make everything above it wrong: picking a new underlay must not teleport
the crosshair, swapping the overlay must not drop extra overlays, and a mode's
computed overlay must never consume the dataset it was computed from.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer import catalog as cat
from fastfuncstuff.viewer.catalog import Kind
from fastfuncstuff.viewer.modes import registry
from fastfuncstuff.viewer.modes.base import (
    ComputedOverlay,
    FloatControl,
    Mode,
    OverlayKind,
)
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.vocab import (
    AddOverlay,
    Read,
    SetMode,
    SetOverlay,
    SetSeed,
    SetUnderlay,
)

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")


def _write(d, name, data, tr=0.0, step=3.0, origin=(-30.0, -36.0, -28.0)):
    aff = np.diag([step, step, step, 1.0])
    aff[:3, 3] = origin
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), aff)
    if tr:
        img.header["pixdim"][4] = tr
        img.header.set_xyzt_units("mm", "sec")
    p = d / name
    nib.save(img, str(p))
    return p


@pytest.fixture
def datadir(tmp_path):
    rng = np.random.default_rng(11)
    _write(tmp_path, "anat.nii.gz", rng.random((12, 14, 10)) * 100)
    _write(tmp_path, "mean_epi.nii.gz", rng.random((8, 9, 7)) * 50)
    _write(tmp_path, "brainmask.nii.gz", (rng.random((12, 14, 10)) > 0.5) * 1.0)
    _write(tmp_path, "stats_tstat.nii.gz", rng.normal(size=(12, 14, 10)))
    (tmp_path / "notes.txt").write_text("not a dataset")
    return tmp_path


@pytest.fixture
def session():
    s = ViewerSession(device=CPU)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------


def test_scan_finds_datasets_and_ignores_other_files(datadir):
    entries = cat.scan(datadir)
    names = {e.name for e in entries}
    assert "anat.nii.gz" in names
    assert "notes.txt" not in names
    assert len(entries) == 4


def test_scan_is_header_only(datadir):
    """A catalog entry must not require reading any voxels."""
    entries = cat.scan(datadir)
    anat = next(e for e in entries if e.name == "anat.nii.gz")
    assert anat.shape == (12, 14, 10)
    assert anat.n_volumes == 1


def test_classification_separates_the_kinds(datadir):
    kinds = {e.name: e.kind for e in cat.scan(datadir)}
    assert kinds["brainmask.nii.gz"] is Kind.MASK
    assert kinds["stats_tstat.nii.gz"] is Kind.STATS
    assert kinds["anat.nii.gz"] is Kind.ANAT


def test_a_4d_dataset_classifies_as_func(tmp_path):
    rng = np.random.default_rng(2)
    _write(tmp_path, "run1.nii.gz", rng.random((6, 6, 5, 20)), tr=2.0)
    assert cat.scan(tmp_path)[0].kind is Kind.FUNC


def test_suggested_underlay_prefers_the_biggest_anatomical(datadir):
    assert cat.suggest_underlay(cat.scan(datadir)).name == "anat.nii.gz"


def test_scan_sorts_anat_before_stats(datadir):
    kinds = [e.kind for e in cat.scan(datadir)]
    assert kinds.index(Kind.ANAT) < kinds.index(Kind.STATS)


def test_scanning_a_non_directory_is_an_error(tmp_path):
    with pytest.raises(NotADirectoryError):
        cat.scan(tmp_path / "nope")


def test_an_unreadable_file_does_not_stop_the_scan(datadir):
    """A results directory routinely holds a half-written dataset."""
    (datadir / "truncated.nii.gz").write_bytes(b"not really gzip")
    assert len(cat.scan(datadir)) == 4


def test_read_command_populates_the_session_catalog(session, datadir):
    session.do(Read(str(datadir)))
    assert len(session.catalog) == 4
    assert session.catalog_dir == datadir


# ---------------------------------------------------------------------------
# underlay / overlay / +1
# ---------------------------------------------------------------------------


def test_underlay_becomes_the_bottom_layer(session, datadir):
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    assert session.state.layers.base.name == "anat.nii.gz"
    assert len(session.state.layers) == 1


def test_overlay_sits_above_the_underlay(session, datadir):
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    session.do(SetOverlay(str(datadir / "stats_tstat.nii.gz")))
    assert session.state.layers.keys[0].startswith("U")
    assert session.state.layers.overlay.name == "stats_tstat.nii.gz"


def test_setting_the_overlay_again_replaces_it(session, datadir):
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    session.do(SetOverlay(str(datadir / "stats_tstat.nii.gz")))
    session.do(SetOverlay(str(datadir / "brainmask.nii.gz")))
    assert len(session.state.layers) == 2
    assert session.state.layers.overlay.name == "brainmask.nii.gz"


def test_plus_one_adds_without_replacing(session, datadir):
    """+1 is the escape from AFNI's single-overlay limit; it must not replace."""
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    session.do(SetOverlay(str(datadir / "stats_tstat.nii.gz")))
    session.do(AddOverlay(str(datadir / "brainmask.nii.gz")))
    assert [ly.name for ly in session.state.layers] == [
        "anat.nii.gz",
        "stats_tstat.nii.gz",
        "brainmask.nii.gz",
    ]


def test_swapping_the_overlay_keeps_extra_overlays(session, datadir):
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    session.do(SetOverlay(str(datadir / "stats_tstat.nii.gz")))
    session.do(AddOverlay(str(datadir / "brainmask.nii.gz")))
    session.do(SetOverlay(str(datadir / "mean_epi.nii.gz")))
    assert [ly.name for ly in session.state.layers] == [
        "anat.nii.gz",
        "mean_epi.nii.gz",
        "brainmask.nii.gz",
    ]


def test_the_underlay_defines_the_display_grid(session, datadir):
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    assert session.state.grid.shape == (12, 14, 10)
    session.do(SetUnderlay(str(datadir / "mean_epi.nii.gz")))
    assert session.state.grid.shape == (8, 9, 7)


def test_swapping_the_underlay_keeps_the_anatomical_location(session, datadir):
    """Changing the base image must not teleport the crosshair."""
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    session.state.crosshair = (6, 7, 5)
    before = session.state.crosshair_mm
    session.do(SetUnderlay(str(datadir / "mean_epi.nii.gz")))
    after = session.state.crosshair_mm
    assert np.allclose(before, after, atol=3.0), (before, after)


def test_replacing_the_underlay_releases_the_old_one(session, datadir):
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    old = session.state.layers.base.key
    session.do(SetUnderlay(str(datadir / "mean_epi.nii.gz")))
    assert old not in session.store.keys()


# ---------------------------------------------------------------------------
# mode framework
# ---------------------------------------------------------------------------


class _Dummy(Mode):
    name = "test_dummy"
    label = "Dummy"
    overlay_kind = OverlayKind.STATISTIC

    def controls(self):
        return (FloatControl(name="gain", label="gain", lo=0.0, hi=10.0, default=2.0),)

    def compute(self):
        if self.session is None or self.session.state.grid is None:
            return None
        shape = self.session.state.grid.shape
        vals = np.full(shape, float(self.params["gain"]), dtype=np.float32)
        return ComputedOverlay(
            values=vals,
            affine=self.session.state.grid.affine,
            name="dummy",
            kind=OverlayKind.STATISTIC,
        )


@pytest.fixture(autouse=True)
def _register_dummy():
    registry.register(_Dummy)
    yield


def test_plain_is_the_default_mode_and_computes_nothing(session):
    assert session.mode.name == "plain"
    assert session.mode.produces_overlay is False


def test_switching_mode_installs_its_overlay(session, datadir):
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    session.do(SetMode("test_dummy"))
    layer = session.state.layers.find_by_source("mode:test_dummy")
    assert layer is not None and layer.is_computed


def test_switching_away_removes_the_computed_overlay(session, datadir):
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    session.do(SetMode("test_dummy"))
    key = session.state.layers.find_by_source("mode:test_dummy").key
    session.do(SetMode("plain"))
    assert session.state.layers.find_by_source("mode:test_dummy") is None
    assert key not in session.store.keys()


def test_a_mode_updates_its_overlay_in_place(session, datadir):
    """Recomputing must not grow the stack or reset the threshold."""
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    session.do(SetMode("test_dummy"))
    key = session.state.layers.find_by_source("mode:test_dummy").key
    n_before = len(session.state.layers)
    session.set_mode_param("gain", "7.0")
    assert len(session.state.layers) == n_before
    assert session.state.layers.find_by_source("mode:test_dummy").key == key
    assert session.store.get(key).array[..., 0].max() == pytest.approx(7.0)


def test_mode_params_are_coerced_from_text(session, datadir):
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    session.do(SetMode("test_dummy"))
    session.set_mode_param("gain", "3.5")
    assert session.mode.params["gain"] == 3.5
    assert isinstance(session.mode.params["gain"], float)


def test_an_unknown_mode_param_is_an_error(session):
    session.do(SetMode("test_dummy"))
    with pytest.raises(KeyError):
        session.set_mode_param("nonexistent", "1")


def test_an_unknown_mode_is_an_error(session):
    with pytest.raises(KeyError):
        session.do(SetMode("not_a_mode"))


def test_overlay_kind_is_declared_per_mode(session):
    session.do(SetMode("test_dummy"))
    assert session.mode.overlay_kind is OverlayKind.STATISTIC


# ---------------------------------------------------------------------------
# instacorr
# ---------------------------------------------------------------------------


@pytest.fixture
def corr_session(tmp_path):
    """A dataset where one blob genuinely shares a signal, so r is checkable."""
    rng = np.random.default_rng(5)
    nx, ny, nz, nt = 10, 12, 8, 60
    data = rng.normal(0, 1.0, (nx, ny, nz, nt)).astype(np.float32)
    signal = np.sin(2 * np.pi * np.arange(nt) / 15.0).astype(np.float32)
    data[2:5, 3:6, 2:4, :] += signal * 6.0
    _write(tmp_path, "anat.nii.gz", rng.random((nx, ny, nz)) * 100)
    _write(tmp_path, "bold.nii.gz", data, tr=2.0)

    s = ViewerSession(device=CPU)
    s.do(SetUnderlay(str(tmp_path / "anat.nii.gz")))
    s.do(SetOverlay(str(tmp_path / "bold.nii.gz")))
    s.store.ensure_ram(s.state.layers.keys[1])
    yield s
    s.close()


def test_instacorr_needs_a_seed_before_it_draws(corr_session):
    corr_session.do(SetMode("instacorr"))
    assert corr_session.state.layers.find_by_source("mode:instacorr") is None


def test_instacorr_correlates_the_shared_signal(corr_session):
    corr_session.do(SetMode("instacorr"))
    corr_session.set_mode_param("blur", "0")
    corr_session.set_mode_param("seed_radius", "0")
    corr_session.do(SetSeed(3, 4, 2))

    layer = corr_session.state.layers.find_by_source("mode:instacorr")
    assert layer is not None
    vol = corr_session.store.get(layer.key).array[..., 0]
    assert vol[3, 4, 2] == pytest.approx(1.0, abs=1e-4), "seed must correlate with itself"
    assert vol[3, 5, 3] > 0.6, "the shared-signal blob must correlate"
    assert abs(vol[8, 10, 6]) < 0.6, "unrelated noise must not"


def test_instacorr_stays_within_correlation_bounds(corr_session):
    corr_session.do(SetMode("instacorr"))
    corr_session.do(SetSeed(3, 4, 2))
    layer = corr_session.state.layers.find_by_source("mode:instacorr")
    vol = corr_session.store.get(layer.key).array[..., 0]
    assert np.nanmin(vol) >= -1.0001 and np.nanmax(vol) <= 1.0001


def test_instacorr_hides_its_input_but_does_not_consume_it(corr_session):
    """The 4-D source must survive: it is where the numbers came from."""
    bold_key = corr_session.state.layers.keys[1]
    corr_session.do(SetMode("instacorr"))
    corr_session.do(SetSeed(3, 4, 2))
    layer = corr_session.state.layers.find(bold_key)
    assert layer is not None, "the source dataset was consumed"
    assert not layer.visible, "the source should be hidden while its map is shown"


def test_leaving_instacorr_restores_the_input(corr_session):
    bold_key = corr_session.state.layers.keys[1]
    corr_session.do(SetMode("instacorr"))
    corr_session.do(SetSeed(3, 4, 2))
    corr_session.do(SetMode("plain"))
    assert corr_session.state.layers.get(bold_key).visible


def test_changing_a_preparation_parameter_changes_the_map(corr_session):
    """The bug this caught: a stale prepare silently returned the old map."""
    corr_session.do(SetMode("instacorr"))
    corr_session.set_mode_param("blur", "0")
    corr_session.do(SetSeed(3, 4, 2))
    key = corr_session.state.layers.find_by_source("mode:instacorr").key
    before = corr_session.store.get(key).array[..., 0].copy()

    corr_session.set_mode_param("blur", "6.0")
    after = corr_session.store.get(key).array[..., 0]
    assert not np.allclose(before, after), "blur change did not re-prepare"


def test_moving_the_seed_changes_the_map(corr_session):
    corr_session.do(SetMode("instacorr"))
    corr_session.do(SetSeed(3, 4, 2))
    key = corr_session.state.layers.find_by_source("mode:instacorr").key
    first = corr_session.store.get(key).array[..., 0].copy()
    corr_session.do(SetSeed(8, 10, 6))
    assert not np.allclose(first, corr_session.store.get(key).array[..., 0])


def test_instacorr_contributes_a_trace(corr_session):
    corr_session.do(SetMode("instacorr"))
    corr_session.do(SetSeed(3, 4, 2))
    traces = corr_session.mode_series((3, 4, 2))
    assert traces and traces[0].values.size == 60


def test_seed_radius_averages_rather_than_taking_one_voxel(corr_session):
    corr_session.do(SetMode("instacorr"))
    corr_session.set_mode_param("blur", "0")
    corr_session.set_mode_param("seed_radius", "0")
    corr_session.do(SetSeed(3, 4, 2))
    key = corr_session.state.layers.find_by_source("mode:instacorr").key
    single = corr_session.store.get(key).array[..., 0].copy()

    corr_session.set_mode_param("seed_radius", "8.0")
    assert not np.allclose(single, corr_session.store.get(key).array[..., 0])


# ---------------------------------------------------------------------------
# overlay defaults
# ---------------------------------------------------------------------------


def test_a_picked_overlay_does_not_hide_the_underlay(session, datadir):
    """Goal zero is checking two images line up; an opaque overlay defeats it."""
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    session.do(SetOverlay(str(datadir / "stats_tstat.nii.gz")))
    overlay = session.state.layers.overlay
    assert overlay.threshold > 0.0
    assert overlay.alpha_mode.value != "off"


def test_a_signed_overlay_gets_a_diverging_map(session, datadir):
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    session.do(SetOverlay(str(datadir / "stats_tstat.nii.gz")))
    assert session.state.layers.overlay.colormap == "redblue"


def test_an_all_positive_overlay_gets_a_sequential_map(session, tmp_path):
    rng = np.random.default_rng(9)
    _write(tmp_path, "anat.nii.gz", rng.random((8, 8, 6)))
    _write(tmp_path, "positive.nii.gz", rng.random((8, 8, 6)) + 1.0)
    session.do(SetUnderlay(str(tmp_path / "anat.nii.gz")))
    session.do(SetOverlay(str(tmp_path / "positive.nii.gz")))
    assert session.state.layers.overlay.colormap == "hot"


def test_the_underlay_stays_opaque(session, datadir):
    """The base image is the base image; it must not be thresholded away."""
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    base = session.state.layers.base
    assert base.threshold == 0.0
    assert base.colormap == "gray"


def test_a_computed_overlay_keeps_the_modes_own_defaults(session, datadir):
    """A mode sets its own range and threshold; the picker must not overwrite."""
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    session.do(SetMode("test_dummy"))
    layer = session.state.layers.find_by_source("mode:test_dummy")
    session.apply_overlay_defaults(layer.key)
    assert session.state.layers.get(layer.key).colormap == layer.colormap


# ---------------------------------------------------------------------------
# ICA mode -- the proof that a new mode is one file
# ---------------------------------------------------------------------------


@pytest.fixture
def ica_dir(tmp_path):
    """A MELODIC-compatible decomposition, one level down as ffs writes it."""
    rng = np.random.default_rng(21)
    out = tmp_path / "ica_out" / "melodic_compat"
    out.mkdir(parents=True)
    nx, ny, nz, k, t = 8, 9, 7, 4, 40
    maps = rng.normal(size=(nx, ny, nz, k)).astype(np.float32)
    for i in range(k):
        maps[i, i, i, i] = 20.0  # a distinct peak per component
    _write(out, "melodic_IC.nii.gz", maps)
    mix = np.column_stack([np.sin(2 * np.pi * f * np.arange(t)) for f in (0.02, 0.05, 0.1, 0.2)])
    np.savetxt(out / "melodic_mix", mix)
    np.savetxt(out / "melodic_FTmix", np.abs(np.fft.rfft(mix, axis=0))[1:])
    _write(tmp_path, "anat.nii.gz", rng.random((nx, ny, nz)) * 100)
    return tmp_path


def test_ica_finds_a_decomposition_one_level_down(ica_dir, session):
    session.do(SetUnderlay(str(ica_dir / "anat.nii.gz")))
    session.read_directory(ica_dir / "ica_out")
    session.do(SetMode("ica"))
    assert session.state.layers.find_by_source("mode:ica") is not None


def test_ica_component_control_spans_the_decomposition(ica_dir, session):
    session.do(SetUnderlay(str(ica_dir / "anat.nii.gz")))
    session.read_directory(ica_dir / "ica_out")
    session.do(SetMode("ica"))
    spec = next(c for c in session.mode.controls() if c.name == "component")
    assert spec.hi == 3


def test_stepping_components_swaps_the_map_and_the_name(ica_dir, session):
    """The name is identity: showing IC 0 while displaying IC 2 is a lie."""
    session.do(SetUnderlay(str(ica_dir / "anat.nii.gz")))
    session.read_directory(ica_dir / "ica_out")
    session.do(SetMode("ica"))
    key = session.state.layers.find_by_source("mode:ica").key
    first = session.store.get(key).array[..., 0].copy()

    session.set_mode_param("component", "2")
    layer = session.state.layers.find_by_source("mode:ica")
    assert layer.name == "IC 2"
    assert layer.key == key, "stepping must update in place, not add a layer"
    assert not np.allclose(first, session.store.get(key).array[..., 0])


def test_stepping_components_keeps_the_threshold_you_set(ica_dir, session):
    """Reviewing components at a fixed threshold is the whole workflow."""
    session.do(SetUnderlay(str(ica_dir / "anat.nii.gz")))
    session.read_directory(ica_dir / "ica_out")
    session.do(SetMode("ica"))
    key = session.state.layers.find_by_source("mode:ica").key
    session.state.layers.update(key, threshold=3.75)
    session.set_mode_param("component", "1")
    assert session.state.layers.get(key).threshold == 3.75


def test_ica_contributes_a_timecourse_and_a_spectrum(ica_dir, session):
    session.do(SetUnderlay(str(ica_dir / "anat.nii.gz")))
    session.read_directory(ica_dir / "ica_out")
    session.do(SetMode("ica"))
    traces = session.mode_series((1, 1, 1))
    assert [t.x_label for t in traces] == ["TR", "Hz"]
    assert traces[0].values.size == 40
    assert traces[1].values.size == 20  # single-sided, DC dropped


def test_the_spectrum_trace_can_be_turned_off(ica_dir, session):
    session.do(SetUnderlay(str(ica_dir / "anat.nii.gz")))
    session.read_directory(ica_dir / "ica_out")
    session.do(SetMode("ica"))
    session.set_mode_param("spectrum", "0")
    assert len(session.mode_series((1, 1, 1))) == 1


def test_ica_without_a_decomposition_says_so_rather_than_failing(session, datadir):
    session.do(SetUnderlay(str(datadir / "anat.nii.gz")))
    session.read_directory(datadir)
    session.do(SetMode("ica"))
    assert session.state.layers.find_by_source("mode:ica") is None
    assert "no decomposition" in session.mode.status()
