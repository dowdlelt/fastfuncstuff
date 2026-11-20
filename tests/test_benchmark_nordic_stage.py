"""Check that NORDIC and SAUNA use the caller's paired benchmark input."""

from types import SimpleNamespace

import pytest

from fastfuncstuff import _paths
from fastfuncstuff.benchmark.stages import nordic, sauna


def test_selected_pair_uses_matching_phase_and_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(_paths, "_BENCHMARK_DATA_DIR", tmp_path)
    data = tmp_path / "nordic_test_data"
    data.mkdir()
    magnitude = data / "sub-demo_task-rest_part-mag_bold.nii.gz"
    phase = data / "sub-demo_task-rest_part-phase_bold.nii.gz"
    reference = data / "NORDIC_sub-demo_task-rest_part-mag_bold.nii"
    for file in (magnitude, phase, reference):
        file.touch()
    ctx = SimpleNamespace(validate_only=False, force_ref=False)

    assert nordic._nordic_inputs(data) == (magnitude, phase)
    assert nordic.check_prerequisites(ctx) == []
    assert sauna.check_prerequisites(ctx) == []


def test_rejects_incomplete_or_ambiguous_pair(tmp_path):
    (tmp_path / "sub-a_part-mag_bold.nii.gz").touch()
    with pytest.raises(ValueError, match="Missing phase input"):
        nordic._nordic_inputs(tmp_path)
    (tmp_path / "sub-b_part-mag_bold.nii.gz").touch()
    with pytest.raises(ValueError, match="found 2"):
        nordic._nordic_inputs(tmp_path)
