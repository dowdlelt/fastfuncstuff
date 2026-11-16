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
