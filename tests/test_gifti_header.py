"""gifti_header must agree with a full nibabel parse on everything it reports."""

import numpy as np
import pytest

from fastfuncstuff.io.gifti import gifti_header, load_gifti_data, save_gifti_data


@pytest.mark.parametrize("shape", [(37,), (37, 1), (37, 5)])
def test_header_matches_a_full_parse(tmp_path, shape):
    path = tmp_path / "x.func.gii"
    data = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    # Values that need escaping in XML, and an empty one.
    meta = {"mesh_fingerprint": "V37_F70_abc", "note": 'a < b & "c"', "empty": "", "TR_seconds": "1.5"}
    save_gifti_data(path, data, meta, time_series=len(shape) == 2)
    full, full_meta = load_gifti_data(path)
    header = gifti_header(path)
    assert header.meta == full_meta
    assert header.n_vertices == full.shape[0]
    assert header.n_arrays == (1 if full.ndim == 1 else full.shape[1])


def test_peek_run_length_reads_a_gifti_series_without_decoding(tmp_path):
    from fastfuncstuff.io.afni import _peek_run_length, load_nifti

    path = tmp_path / "run.func.gii"
    save_gifti_data(path, np.zeros((11, 7), np.float32), {"TR_seconds": "2"})
    assert _peek_run_length(path) == load_nifti(str(path)).shape[3] == 7
    single = tmp_path / "map.shape.gii"
    save_gifti_data(single, np.zeros(11, np.float32))
    assert _peek_run_length(single) == 1


def _nibabel_arrays(path):
    import nibabel as nib

    img = nib.load(str(path))
    return dict(img.meta), [(a.data, a.intent, dict(a.meta)) for a in img.darrays]


def _assert_same_as_nibabel(path):
    from fastfuncstuff.io.gifti import _read_gifti

    fast = _read_gifti(path)
    assert fast is not None
    meta, arrays = _nibabel_arrays(path)
    assert fast.meta == meta
    assert len(fast.darrays) == len(arrays)
    for got, (data, intent, array_meta) in zip(fast.darrays, arrays, strict=True):
        assert got.data.dtype == data.dtype and got.data.shape == data.shape
        np.testing.assert_array_equal(got.data, data)
        assert got.intent == intent
        assert got.meta == array_meta


def test_direct_reader_matches_nibabel_on_series_maps_and_surfaces(tmp_path):
    from fastfuncstuff.io.gifti import save_gifti_surface, save_nifti_data_as_gifti

    rng = np.random.default_rng(1)
    series = tmp_path / "s.func.gii"  # base64, one array per time point
    save_gifti_data(series, rng.standard_normal((50, 6)).astype(np.float32), {"a": "b & c"})
    _assert_same_as_nibabel(series)
    single = tmp_path / "m.shape.gii"  # gzip base64
    save_gifti_data(single, rng.standard_normal(50).astype(np.float32))
    _assert_same_as_nibabel(single)
    surface = tmp_path / "x.surf.gii"  # float32 points + int32 triangles, 2-D arrays
    save_gifti_surface(surface, rng.standard_normal((4, 3)), np.array([[0, 1, 2], [1, 2, 3]]))
    _assert_same_as_nibabel(surface)
    bucket = tmp_path / "b.func.gii"  # per-array names and stat codes
    save_nifti_data_as_gifti(
        bucket,
        rng.standard_normal((50, 1, 1, 2)).astype(np.float32),
        brick_labels=["coef", "tstat"],
        brick_stataux={1: (3, (20.0,))},
    )
    _assert_same_as_nibabel(bucket)


def test_direct_reader_declines_what_it_does_not_handle(tmp_path):
    import nibabel as nib

    from fastfuncstuff.io.gifti import _read_gifti, load_gifti_data

    path = tmp_path / "ascii.func.gii"
    values = np.arange(6, dtype=np.float32)
    array = nib.gifti.GiftiDataArray(values, datatype="NIFTI_TYPE_FLOAT32", encoding="ASCII")
    nib.save(nib.gifti.GiftiImage(darrays=[array]), str(path))
    assert _read_gifti(path) is None
    np.testing.assert_array_equal(load_gifti_data(path)[0], values)  # nibabel fallback


def test_external_series_round_trips_through_both_readers(tmp_path):
    import nibabel as nib

    from fastfuncstuff.io.afni import _peek_run_length, load_nifti
    from fastfuncstuff.io.gifti import (
        _read_gifti,
        save_gifti_series_external,
        surface_meta,
    )

    data = np.random.default_rng(2).standard_normal((40, 9)).astype(np.float32)
    path = tmp_path / "run-01.merged.func.gii"
    meta = {"mesh_fingerprint": "V40_x", "TR_seconds": "1", "odd": "<&>\"'"}
    dat = save_gifti_series_external(path, data, meta)
    assert dat.endswith("run-01.merged.func.dat")
    assert _read_gifti(path) is not None
    loaded, got_meta = load_gifti_data(path)
    np.testing.assert_array_equal(loaded, data)
    assert {k: got_meta[k] for k in meta} == meta
    img = nib.load(str(path))  # the standard reader agrees
    np.testing.assert_array_equal(np.stack([a.data for a in img.darrays], axis=1), data)
    assert dict(img.meta) == got_meta
    header = gifti_header(path)
    assert (header.n_arrays, header.n_vertices) == (9, 40) and header.meta == got_meta
    assert _peek_run_length(path) == 9
    as_nifti = load_nifti(str(path))
    np.testing.assert_array_equal(np.asarray(as_nifti.dataobj)[:, 0, 0, :], data)
    assert surface_meta(as_nifti.header)["mesh_fingerprint"] == "V40_x"
