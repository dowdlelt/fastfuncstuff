"""Smoothing and slice timing: the two preproc steps with no QC volumes.

Neither needs a picture made for it -- the data is the picture -- so what is
tested here is that each does the thing it claims, and that the parts which can
silently be wrong are not. For smoothing that is the millimetre-to-voxel
conversion; for slice timing it is finding the right timing table and refusing
to guess when it cannot.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.tools import registry as tools
from fastfuncstuff.viewer.tools.slicetime import (
    dataset_stem,
    reference_time,
    repetition_time,
    sidecar_for,
    source_file,
)
from fastfuncstuff.viewer.tools.smooth import voxel_sizes
from fastfuncstuff.viewer.vocab import SetMode

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")

TR = 2.0
NX, NY, NZ, NT = 16, 16, 8, 24


@pytest.fixture
def raw(tmp_path):
    """A run with a hard edge, a per-slice phase ramp, and a BIDS sidecar."""
    rng = np.random.default_rng(0)
    t = np.arange(NT) * TR
    slice_times = [round(k * TR / NZ, 6) for k in range(NZ)]
    data = np.zeros((NX, NY, NZ, NT), np.float32)
    data[4:8] = 100.0
    data[8:12] = 20.0
    for k in range(NZ):
        data[..., k, :] += 30.0 * np.sin(2 * np.pi * 0.05 * (t + slice_times[k]))
    data += rng.normal(0, 1.0, data.shape).astype(np.float32)

    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    aff[:3, 3] = (-24.0, -24.0, -12.0)
    img = nib.Nifti1Image(data, aff)
    img.header["pixdim"][4] = TR
    img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(tmp_path / "sub-01_task-rest_bold.nii.gz"))
    (tmp_path / "sub-01_task-rest_bold.json").write_text(
        json.dumps({"RepetitionTime": TR, "SliceTiming": slice_times})
    )
    return tmp_path, data, slice_times


@pytest.fixture
def preproc(raw):
    directory, _, _ = raw
    s = ViewerSession(device=CPU)
    s.read_directory(directory)
    key = s.load(directory / "sub-01_task-rest_bold.nii.gz")
    s.do(SetMode("preproc"))
    yield s, directory, key
    s.close()


def _run(mode, tool_name, params):
    spec = mode.dialog_for(tool_name)
    return spec.install(spec.run({**spec.params, **params}, None))


def _made(session, op):
    return next(
        layer for layer in session.state.layers if layer.source.startswith(f"derived:{op}:")
    )


# ---------------------------------------------------------------------------
# both tools
# ---------------------------------------------------------------------------


def test_both_tools_are_registered_and_offered(preproc):
    s, _, _ = preproc
    assert {"smooth", "slicetime"} <= {t.name for t in tools.all()}
    assert {"smooth", "slicetime"} <= {a.name for a in s.mode.actions()}


@pytest.mark.parametrize("name", ["smooth", "slicetime"])
def test_neither_makes_a_qc_volume_or_a_plot(preproc, name):
    """The data is the picture; a second picture would only be in the way."""
    s, _, _ = preproc
    params = {"fwhm": 4.0} if name == "smooth" else {}
    _run(s.mode, name, params)
    assert sum(1 for layer in s.state.layers if layer.is_derived) == 1
    assert s.mode.panel_names() == ()


# ---------------------------------------------------------------------------
# smoothing
# ---------------------------------------------------------------------------


def test_voxel_sizes_come_from_the_column_norms():
    """An oblique affine's diagonal understates every spacing."""
    straight = np.diag([2.0, 3.0, 4.0, 1.0])
    assert voxel_sizes(straight) == pytest.approx((2.0, 3.0, 4.0))

    angle = np.deg2rad(30.0)
    rot = np.eye(4)
    rot[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    oblique = rot @ straight
    assert voxel_sizes(oblique) == pytest.approx((2.0, 3.0, 4.0))
    assert np.diag(oblique)[0] < 2.0, "the diagonal alone would under-blur this"


def test_smoothing_blunts_an_edge_and_keeps_the_shape(preproc):
    s, _, key = preproc
    before = np.asarray(s.store.ensure_ram(key))
    _run(s.mode, "smooth", {"fwhm": 6.0})
    after = np.asarray(s.store.ensure_ram(_made(s, "smooth").key))

    assert after.shape == before.shape
    sharp = np.abs(np.diff(before[:, :, :, 0], axis=0)).max()
    blunt = np.abs(np.diff(after[:, :, :, 0], axis=0)).max()
    assert blunt < sharp * 0.75, "a 6 mm kernel on 3 mm voxels must soften the boundary"


def test_a_zero_width_blur_is_refused_rather_than_copied(preproc):
    s, _, _ = preproc
    spec = s.mode.dialog_for("smooth")
    with pytest.raises(ValueError, match="greater than zero"):
        spec.run({**spec.params, "fwhm": 0.0}, None)


def test_the_blur_records_the_width_and_the_grid_it_was_applied_on(preproc):
    """4 mm means something different on 1 mm voxels; the layer should say."""
    s, _, _ = preproc
    _run(s.mode, "smooth", {"fwhm": 4.0})
    assert "FWHM 4 mm" in _made(s, "smooth").path
    assert "3x3x3" in _made(s, "smooth").path


# ---------------------------------------------------------------------------
# finding the timing table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "stem"),
    [
        ("sub-01_bold.nii.gz", "sub-01_bold"),
        ("sub-01_bold.nii", "sub-01_bold"),
        ("sub-01_bold.nii.zst", "sub-01_bold"),
        ("plain", "plain"),
    ],
)
def test_dataset_stem_strips_the_whole_extension(tmp_path, name, stem):
    """Path.stem leaves ".nii" on every gzipped NIfTI, which is the common case."""
    assert dataset_stem(tmp_path / name) == stem


def test_the_sidecar_is_found_beside_the_data(raw):
    directory, _, _ = raw
    found = sidecar_for(directory / "sub-01_task-rest_bold.nii.gz")
    assert found is not None and found.name == "sub-01_task-rest_bold.json"


def test_no_sidecar_is_not_an_error_by_itself(tmp_path):
    assert sidecar_for(tmp_path / "lonely.nii.gz") is None


def test_a_derived_layer_finds_the_file_it_descends_from(preproc):
    """Motion correction does not change when the slices were acquired."""
    s, directory, key = preproc
    _run(s.mode, "smooth", {"fwhm": 4.0})
    made = _made(s, "smooth")
    assert not made.path.endswith(".nii.gz"), "it has no file of its own"
    assert source_file(s, made.key) == directory / "sub-01_task-rest_bold.nii.gz"


def test_slicetiming_a_derived_layer_uses_the_originals_sidecar(preproc):
    s, _, _ = preproc
    _run(s.mode, "smooth", {"fwhm": 4.0})
    smoothed = _made(s, "smooth")

    # Installing selects it, so it is already what the next tool defaults to.
    spec = s.mode.dialog_for("slicetime")
    assert s.mode.inputs_for(tools.find("slicetime"))[spec.params["input"]] == smoothed.key

    spec.install(spec.run(spec.params, None))
    assert "sub-01_task-rest_bold.json" in _made(s, "slicetime").path


def test_a_layer_with_no_ancestry_says_where_it_looked(preproc):
    s, _, key = preproc
    s.install_derived(key, np.asarray(s.store.ensure_ram(key)), op="orphan", name="ORPHAN")
    orphan = next(layer for layer in s.state.layers if layer.name == "ORPHAN")
    s.state.layers.update(orphan.key, source="file", path="<nowhere>")

    spec = s.mode.dialog_for("slicetime")
    with pytest.raises(ValueError, match="no slice timing found"):
        spec.run({**spec.params, "input": "ORPHAN"}, None)


def test_repetition_time_is_read_from_the_sidecar(raw, tmp_path):
    directory, _, _ = raw
    assert repetition_time(directory / "sub-01_task-rest_bold.json") == pytest.approx(TR)
    (tmp_path / "silent.json").write_text("{}")
    assert repetition_time(tmp_path / "silent.json") == 0.0
    assert repetition_time(None) == 0.0


# ---------------------------------------------------------------------------
# the correction itself
# ---------------------------------------------------------------------------


def test_reference_names_map_to_times():
    timing = [0.0, 0.5, 1.0, 1.5]
    assert reference_time("start of TR", timing) == 0.0
    assert reference_time("first slice", timing) == 0.0
    assert reference_time("middle slice", timing) == 1.0
    assert reference_time("mean", timing) is None, "None lets the library use its own default"


def test_slice_timing_removes_the_lag_between_slices(preproc, raw):
    """Every slice sees the same wave; only the sampling instant differs."""
    _, data, _ = raw
    s, _, key = preproc
    _run(s.mode, "slicetime", {"reference": "start of TR"})
    after = np.asarray(s.store.ensure_ram(_made(s, "slicetime").key))

    def mean_lag(volume):
        ref = volume[6, 6, 0, 3:-3]
        lags = []
        for k in range(NZ):
            x = volume[6, 6, k, 3:-3]
            c = np.correlate(x - x.mean(), ref - ref.mean(), mode="same")
            lags.append(abs(int(np.argmax(c)) - len(c) // 2))
        return float(np.mean(lags))

    assert mean_lag(after) < mean_lag(np.asarray(data)) or mean_lag(after) == 0.0
    assert mean_lag(after) == pytest.approx(0.0, abs=0.2)


def test_a_timing_table_of_the_wrong_length_is_refused(preproc, tmp_path):
    """A table from a different protocol is the quiet way to corrupt a run."""
    s, _, _ = preproc
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"SliceTiming": [0.0, 0.1, 0.2]}))
    spec = s.mode.dialog_for("slicetime")
    with pytest.raises(ValueError, match="lists 3 slice times"):
        spec.run({**spec.params, "timing": str(wrong)}, None)


def test_a_missing_timing_file_is_named(preproc, tmp_path):
    s, _, _ = preproc
    spec = s.mode.dialog_for("slicetime")
    with pytest.raises(ValueError, match="no such timing file"):
        spec.run({**spec.params, "timing": str(tmp_path / "absent.json")}, None)


def test_the_correction_records_what_it_used(preproc):
    s, _, _ = preproc
    _run(s.mode, "slicetime", {"reference": "start of TR", "interp": "linear"})
    provenance = _made(s, "slicetime").path
    assert "start of TR" in provenance
    assert "linear" in provenance
    assert "TR 2s" in provenance
    assert "sub-01_task-rest_bold.json" in provenance
