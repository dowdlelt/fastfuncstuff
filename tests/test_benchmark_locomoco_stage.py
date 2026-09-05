"""The locomoco benchmark stage: paths, PE axis, and its reference-free checks.

The stage has no reference tool, so ``validate()`` is the only thing standing
between "ffs_locomoco ran" and "ffs_locomoco produced something usable". These
tests pin the three failures it exists to catch.
"""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.benchmark.config import BenchmarkConfig
from fastfuncstuff.benchmark.runner import BenchmarkContext
from fastfuncstuff.benchmark.stages import locomoco

SHAPE = (6, 6, 4, 5)


def _ctx(tmp_path) -> BenchmarkContext:
    config = BenchmarkConfig(
        dataset_id="dsTEST", subject="01", session="01", tasks={"localizer": [1]}
    )
    return BenchmarkContext(data_dir=tmp_path, device="cpu", config=config, verbose=False)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)).to_filename(str(path))


def _series(rng, scale=1.0):
    # Bright block on a dim background so _automask keeps a sensible interior.
    data = rng.random(SHAPE).astype(np.float32) * 10.0
    data[1:5, 1:5, 1:3] += 100.0 * scale
    return data


def _populate(tmp_path, flow=None, corrected_shape=None, pe_dir="j-"):
    """Write a full set of stage inputs/outputs and return the context."""
    ctx = _ctx(tmp_path)
    rng = np.random.default_rng(0)
    sidecar = ctx.func_dir / f"{ctx.bids_prefix('localizer', 1)}_bold.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({"PhaseEncodingDirection": pe_dir}))

    src = _series(rng)
    _write(locomoco._input_path(ctx, "localizer", 1), src)
    _write(
        locomoco._corrected_path(ctx, "localizer", 1),
        src if corrected_shape is None else rng.random(corrected_shape),
    )
    _write(
        locomoco._flow_path(ctx, "localizer", 1),
        rng.random(SHAPE).astype(np.float32) * 0.5 if flow is None else flow,
    )
    return ctx


def test_pe_axis_comes_from_the_sidecar(tmp_path):
    ctx = _populate(tmp_path, pe_dir="i-")
    assert locomoco._pe_axis(ctx, "localizer", 1) == "x"


def test_pe_axis_override_wins(tmp_path):
    ctx = _populate(tmp_path, pe_dir="j-")
    ctx.config.stage_params["locomoco"] = {"pe_dir": "z"}
    assert locomoco._pe_axis(ctx, "localizer", 1) == "z"


def test_validate_reports_diagnostics_without_a_reference(tmp_path):
    ctx = _populate(tmp_path)
    result = locomoco.validate(ctx)
    assert result["passed"]
    assert result["reference"] is None
    assert "no reference" in result["summary"]
    assert result["rms_flow_vox"] > 0


def test_validate_fails_on_an_identically_zero_flow(tmp_path):
    """The one thing a reference-free check can call wrong: nothing moved."""
    ctx = _populate(tmp_path, flow=np.zeros(SHAPE, dtype=np.float32))
    result = locomoco.validate(ctx)
    assert not result["passed"]
    assert "identically zero" in result["summary"]


def test_validate_fails_on_a_divergent_flow(tmp_path):
    flow = np.full(SHAPE, locomoco.MAX_PLAUSIBLE_FLOW_VOX + 1.0, dtype=np.float32)
    ctx = _populate(tmp_path, flow=flow)
    result = locomoco.validate(ctx)
    assert not result["passed"]
    assert "exceeds" in result["summary"]


def test_validate_fails_on_a_grid_mismatch(tmp_path):
    ctx = _populate(tmp_path, corrected_shape=(6, 6, 4, 4))
    result = locomoco.validate(ctx)
    assert not result["passed"]
    assert "Grid mismatch" in result["summary"]


def test_prerequisite_is_the_ffs_moco_output(tmp_path):
    ctx = _ctx(tmp_path)
    missing = locomoco.check_prerequisites(ctx)
    assert missing == [str(locomoco._input_path(ctx, "localizer", 1))]


def test_run_ffs_skips_when_the_output_exists(tmp_path, monkeypatch):
    ctx = _populate(tmp_path)

    def _boom(*args, **kwargs):
        pytest.fail("run_timed should not be called when the output already exists")

    monkeypatch.setattr(locomoco, "run_timed", _boom)
    assert locomoco.run_ffs(ctx) == 0.0
