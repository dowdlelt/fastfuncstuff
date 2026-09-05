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


def _populate_synthetic(ctx, *, recovered=None, corrected=None):
    """Write a synthetic case. By default a PERFECT recovery: the reported flow
    is exactly the negative of the imposed field, and correcting restores the
    clean series, so every threshold is met by construction."""
    rng = np.random.default_rng(1)
    clean = _series(rng)
    # Scaled like the real fixture (SYNTH_AMPLITUDE_VOX), so a fractional
    # amplitude error lands where it would on real data.
    truth = (rng.random(SHAPE).astype(np.float32) - 0.5) * 2.0 * locomoco.SYNTH_AMPLITUDE_VOX
    distorted = clean + 5.0 * truth  # any consistent 'damage' works for the ratio
    _write(locomoco._synth_clean_path(ctx, "localizer", 1), clean)
    _write(locomoco._synth_input_path(ctx, "localizer", 1), distorted)
    _write(locomoco._synth_truth_path(ctx, "localizer", 1), truth)
    _write(
        locomoco._synth_corrected_path(ctx, "localizer", 1),
        clean if corrected is None else corrected,
    )
    _write(
        locomoco._synth_flow_path(ctx, "localizer", 1),
        -truth if recovered is None else recovered,
    )
    return truth, clean


def test_pe_axis_comes_from_the_sidecar(tmp_path):
    ctx = _populate(tmp_path, pe_dir="i-")
    assert locomoco._pe_axis(ctx, "localizer", 1) == "x"


def test_pe_axis_override_wins(tmp_path):
    ctx = _populate(tmp_path, pe_dir="j-")
    ctx.config.stage_params["locomoco"] = {"pe_dir": "z"}
    assert locomoco._pe_axis(ctx, "localizer", 1) == "z"


def test_validate_passes_on_a_good_synthetic_recovery(tmp_path):
    ctx = _populate(tmp_path)
    _populate_synthetic(ctx)
    result = locomoco.validate(ctx)
    assert result["passed"], result["summary"]
    assert result["reference"] == "synthetic ground truth"
    assert result["synth_flow_r"] < -0.99  # recovered == -imposed
    assert result["synth_rms_err_vox"] < 1e-5
    assert result["synth_residual_ratio"] < 1e-5
    assert result["rms_flow_vox"] > 0  # the real run's diagnostic still reported


def test_validate_fails_when_the_synthetic_case_is_missing(tmp_path):
    """Without its own truth the stage has nothing to validate, and must say so
    rather than passing on diagnostics alone."""
    ctx = _populate(tmp_path)
    result = locomoco.validate(ctx)
    assert not result["passed"]
    assert "synthetic" in result["summary"]


def test_validate_catches_a_sign_flip(tmp_path):
    """|r| would score a flipped field perfectly; the threshold is signed."""
    ctx = _populate(tmp_path)
    truth, clean = _populate_synthetic(ctx)
    _write(locomoco._synth_flow_path(ctx, "localizer", 1), truth)  # not -truth
    result = locomoco.validate(ctx)
    assert not result["passed"]
    assert "flow r=" in result["summary"]


def test_validate_catches_an_amplitude_error_that_still_correlates(tmp_path):
    """Correlation is blind to scale; the RMS threshold is what catches this."""
    ctx = _populate(tmp_path)
    truth, clean = _populate_synthetic(ctx)
    # Recovers only a tenth of the amplitude: r is still -1.0 exactly.
    _write(locomoco._synth_flow_path(ctx, "localizer", 1), -truth * 0.1)
    result = locomoco.validate(ctx)
    assert result["synth_flow_r"] < -0.99  # correlation is perfect...
    assert not result["passed"]  # ...and the amplitude is still wrong
    assert "RMS err" in result["summary"]


def test_validate_catches_a_correction_that_did_not_help(tmp_path):
    """A perfect-looking field is worthless if the corrected series is no closer
    to the clean one; the residual ratio is the end-to-end check."""
    ctx = _populate(tmp_path)
    truth, clean = _populate_synthetic(ctx)
    distorted, _ = locomoco._load_vol(locomoco._synth_input_path(ctx, "localizer", 1))
    _write(locomoco._synth_corrected_path(ctx, "localizer", 1), distorted.numpy())
    result = locomoco.validate(ctx)
    assert not result["passed"]
    assert "residual ratio" in result["summary"]


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


def test_run_ffs_skips_only_when_BOTH_runs_are_on_disk(tmp_path, monkeypatch):
    """The stage runs locomoco twice -- the real run and the scored synthetic
    one -- so a directory holding only the real outputs is not done."""
    ctx = _populate(tmp_path)

    def _boom(*args, **kwargs):
        pytest.fail("run_timed should not be called when both runs already exist")

    monkeypatch.setattr(locomoco, "run_timed", _boom)
    # Synthetic outputs missing -> must NOT skip.
    calls = []
    monkeypatch.setattr(locomoco, "run_timed", lambda *a, **k: (calls.append(a), (0.0, None))[1])
    monkeypatch.setattr(locomoco, "_build_synthetic_case", lambda *a, **k: None)
    locomoco.run_ffs(ctx)
    assert calls, "a missing synthetic case must re-run the stage"

    # With both present, nothing runs.
    _write(locomoco._synth_flow_path(ctx, "localizer", 1), np.zeros(SHAPE, dtype=np.float32))
    monkeypatch.setattr(locomoco, "run_timed", _boom)
    assert locomoco.run_ffs(ctx) == 0.0
