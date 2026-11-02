"""ffs_moco -phase: unwrapped phase takes exactly the magnitude's resample."""

from pathlib import Path

import pytest
import torch

from fastfuncstuff.cli.moco import main
from fastfuncstuff.processing.io import load_image, save_image
from tests.test_moco_multiecho import _blob, _shifted_series

# 400 Hz across the blob: wider than one wrap period at TE 4 ms (250 Hz) even inside the
# automask, while neighbour steps stay far below half a period.
SCALE = 400.0
TE = "4"


def _write(path, tensor):
    save_image(tensor, str(path), header_info=None)


def _run(args):
    main([*args, "-device", "cpu", "-verb", "0"])


def test_single_echo_phase_gets_identical_resample(tmp_path):
    # Phase = SCALE x magnitude. Resampling is linear, so if the phase took exactly the
    # magnitude's matrices the outputs keep that factor voxel for voxel.
    mag = _shifted_series(_blob())
    _write(tmp_path / "m.nii.gz", mag)
    _write(tmp_path / "p.nii.gz", mag * SCALE)
    _run(
        [
            "-input",
            str(tmp_path / "m.nii.gz"),
            "-phase",
            str(tmp_path / "p.nii.gz"),
            "-echo_times",
            TE,
            "-prefix",
            str(tmp_path / "mc.nii.gz"),
        ]
    )
    a, _ = load_image(str(tmp_path / "mc.nii.gz"))
    p, _ = load_image(str(tmp_path / "mc.part-phase.nii.gz"))
    assert torch.allclose(p, a * SCALE, atol=1e-3 * SCALE)


def test_multiecho_phase_per_echo(tmp_path):
    mag1 = _shifted_series(_blob())
    mag2 = mag1 * 0.6
    for name, t in (("m1", mag1), ("m2", mag2), ("p1", mag1 * SCALE), ("p2", mag2 * 2 * SCALE)):
        _write(tmp_path / f"{name}.nii.gz", t)
    argv = [
        "-input",
        str(tmp_path / "m1.nii.gz"),
        str(tmp_path / "m2.nii.gz"),
        "-phase",
        str(tmp_path / "p1.nii.gz"),
        str(tmp_path / "p2.nii.gz"),
        "-reg_echo",
        "1",
        "-echo_times",
        TE,
        "20",
        "-prefix",
        str(tmp_path / "mc.nii.gz"),
    ]
    _run(argv)
    # -batch_skip's list names exactly what was written, the phase included: a run
    # listed without it would be skipped and never write the phase a later stage reads.
    from fastfuncstuff.cli.moco import _expected_outputs, parse_args

    expected = _expected_outputs(parse_args(argv))
    assert any(".part-phase" in o for o in expected)
    assert all(Path(o).exists() for o in expected), [o for o in expected if not Path(o).exists()]
    a1, _ = load_image(str(tmp_path / "mc.echo-1.nii.gz"))
    a2, _ = load_image(str(tmp_path / "mc.echo-2.nii.gz"))
    p1, _ = load_image(str(tmp_path / "mc.echo-1.part-phase.nii.gz"))
    p2, _ = load_image(str(tmp_path / "mc.echo-2.part-phase.nii.gz"))
    assert torch.allclose(p1, a1 * SCALE, atol=1e-3 * SCALE)
    assert torch.allclose(p2, a2 * 2 * SCALE, atol=1e-3 * SCALE)


def test_wrapped_phase_refused_before_registration(tmp_path):
    mag = _shifted_series(_blob())
    period = 250.0  # Hz at TE 4 ms
    wrapped = torch.remainder(mag * SCALE + period / 2, period) - period / 2
    _write(tmp_path / "m.nii.gz", mag)
    _write(tmp_path / "p.nii.gz", wrapped)
    with pytest.raises(SystemExit):
        _run(
            [
                "-input",
                str(tmp_path / "m.nii.gz"),
                "-phase",
                str(tmp_path / "p.nii.gz"),
                "-echo_times",
                TE,
                "-prefix",
                str(tmp_path / "mc.nii.gz"),
            ]
        )
    assert not (tmp_path / "mc.nii.gz").exists()


def test_hz_phase_needs_echo_times(tmp_path):
    mag = _shifted_series(_blob())
    _write(tmp_path / "m.nii.gz", mag)
    _write(tmp_path / "p.nii.gz", mag * SCALE)
    with pytest.raises(SystemExit):
        _run(
            [
                "-input",
                str(tmp_path / "m.nii.gz"),
                "-phase",
                str(tmp_path / "p.nii.gz"),
                "-prefix",
                str(tmp_path / "mc.nii.gz"),
            ]
        )
