"""3D EPI physio (processing/physio_3d.py): the field sampled once per partition cell.

Synthetic runs are built as the scanner fills them: each volume's partition lines are
acquired cell by cell across the TR, and a field change b(t) puts 2 pi TE b(t_line) on
each line. Breathing at 0.27 Hz aliases at TR 2 s (Nyquist 0.25 Hz); only the cell
timing recovers its true rate.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fastfuncstuff.processing import physio_3d

CPU = torch.device("cpu")
TR = 2.0
RZ = 3
F_RESP = 0.27  # Hz: 16.2/min, above the 0.25 Hz volume Nyquist
F_CARD = 1.15  # Hz: 69/min
TES = (0.010, 0.025, 0.040)


def _run(
    nt=150,
    shape=(20, 20, 36),
    card=0.3,
    resp=1.2,
    noise=0.002,
    seed=0,
    descending=False,
    static=False,
):
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    ncell = nz // RZ
    x, y, z = np.meshgrid(*(np.linspace(-1, 1, n) for n in shape), indexing="ij")
    obj = np.exp(-(x**2 + y**2 + z**2) / 0.5) * (1 + 0.3 * rng.standard_normal(shape))
    obj = obj * np.exp(1j * rng.uniform(-np.pi, np.pi, shape))
    K0 = np.fft.fftshift(np.fft.fft(obj, axis=2), axes=2)
    cell = np.arange(nz) // RZ
    if descending:
        cell = ncell - 1 - cell
    mags, phs = [], []
    for te in TES:
        mag = np.zeros((*shape, nt), np.float32)
        ph = np.zeros((*shape, nt), np.float32)
        for v in range(nt):
            # static: the whole volume sees one field (a 2D-like control, no line timing)
            t = v * TR + (0.5 * TR if static else (cell + 0.5) * TR / ncell)
            theta = 2 * np.pi * F_CARD * t
            # breathing wanders in rate and depth, as it does (a perfectly steady period
            # near 2 TR would make reversed volumes line up at their boundaries)
            rphase = 2 * np.pi * F_RESP * t + 0.6 * np.sin(2 * np.pi * 0.013 * t)
            depth = 1 + 0.3 * np.sin(2 * np.pi * 0.021 * t + 1.0)
            heart = np.cos(theta) + 0.5 * np.cos(2 * theta)
            b = resp * depth * np.sin(rphase) + card * heart
            img = np.fft.ifft(
                np.fft.ifftshift(K0 * np.exp(2j * np.pi * te * b)[None, None], axes=2), axis=2
            )
            img = img + noise * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
            mag[..., v] = np.abs(img)
            ph[..., v] = np.angle(img)
        mags.append(mag)
        phs.append(ph)
    return mags, phs


@pytest.fixture(scope="module")
def run():
    return _run()


def test_cell_trace_recovers_breathing_at_its_true_rate_and_the_fill_order(run):
    """The volume rate aliases 16.2/min to 13.8/min; the cell trace must not."""
    mags, phs = run
    pt = physio_3d.partition_field_traces(mags, phs, TR, RZ, TES, device=CPU)
    assert pt.order == "ascending" and pt.times.shape[1] == 12
    _, _, resp = physio_3d.breathing_cycle(pt)
    assert abs(resp - F_RESP * 60) < 1.0
    # the field is the TE slope: amplitude in Hz, not in radians of one echo
    assert 0.6 < pt.field.std() < 1.6


def test_fill_order_is_read_from_the_data():
    """Reversed volumes jump at every boundary; the true order is seamless. Needs content
    away from the volume Nyquist rate (here the heartbeat): breathing alone near Nyquist
    looks the same either way, which the second opinion then flags."""
    mags, phs = _run(nt=100, descending=True)
    pt = physio_3d.partition_field_traces(mags, phs, TR, RZ, TES, device=CPU)
    assert pt.order == "descending"


def test_a_heartbeat_is_found_by_its_cell_timing(run):
    mags, phs = run
    pt = physio_3d.partition_field_traces(mags, phs, TR, RZ, TES, device=CPU)
    c = physio_3d.cardiac(pt, TR, n_shuffles=49)
    assert c.found and c.prominence > c.shuffled_max
    assert abs(c.rate_per_min - F_CARD * 60) < 1.5


def test_no_heartbeat_is_not_found():
    """Breathing (and its harmonics, which reach the cardiac band) only: the strongest
    in-band peak must not beat the derangements of the cell times."""
    mags, phs = _run(card=0.0, seed=3)
    pt = physio_3d.partition_field_traces(mags, phs, TR, RZ, TES, device=CPU)
    assert not physio_3d.cardiac(pt, TR, n_shuffles=49).found


def test_kz_diagnostic_sees_line_timing_and_the_te_scaling(run):
    """Breathing (0.27 Hz) aliases to 0.23 Hz at TR 2 s. Sampled line by line across the
    TR, its component there advances in angle across kz, the same on every echo, with
    |A| in proportion to TE (the field, not coil or motion phase)."""
    mags, phs = run
    d = physio_3d.kz_diagnostic(
        physio_3d.partition_field_traces(mags, phs, TR, RZ, TES, device=CPU), TR
    )
    assert abs(d.alias_hz - (1 / TR - F_RESP)) < 0.01
    assert d.echo_agreement > 0.99
    np.testing.assert_allclose(d.amp_ratio, np.array(TES) / TES[-1], atol=0.05)
    # inner lines span ~2/3 of the TR: 2 pi 0.27 Hz x 1.33 s ~ 2.3 rad, in cell steps
    assert 1.5 < abs(d.phase_span_rad) < 3.0


def test_kz_diagnostic_span_collapses_without_line_timing():
    """Control: one field per volume (as if acquired at one instant) gives every line the
    same temporal phase -- the span is what the timing adds, not a fixed property of A."""
    mags, phs = _run(nt=100, static=True, card=0.0)
    d = physio_3d.kz_diagnostic(
        physio_3d.partition_field_traces(mags, phs, TR, RZ, TES, device=CPU), TR
    )
    assert abs(d.phase_span_rad) < 0.2


def test_cli_picks_3d_from_the_sidecar_and_writes_its_outputs(tmp_path, run):
    import nibabel as nib

    from fastfuncstuff.cli.physio import main

    mags, phs = run
    args = ["-input"]
    for e, te in enumerate(TES):
        name = tmp_path / f"run_echo-{e + 1}.nii.gz"
        nib.save(nib.Nifti1Image(mags[e], np.eye(4)), name)
        (tmp_path / f"run_echo-{e + 1}.json").write_text(json.dumps(
            {"MRAcquisitionType": "3D", "RepetitionTime": TR, "EchoTime": te,
             "ParallelReductionFactorOutOfPlane": RZ}))  # fmt: skip
        nib.save(nib.Nifti1Image(phs[e], np.eye(4)), tmp_path / f"run_echo-{e + 1}_ph.nii.gz")
        args.append(str(name))
    phase = ["-phase", *(str(tmp_path / f"run_echo-{e + 1}_ph.nii.gz") for e in range(3))]
    rc = main([*args, *phase, "-prefix", str(tmp_path / "o" / "run"), "-device", "cpu",
               "-verb", "0", "-no_qc", "-kz_plot"])  # fmt: skip
    assert rc == 0
    s = json.loads((tmp_path / "o" / "run.json").read_text())
    assert (tmp_path / "o" / "run.kz.png").exists() and s["partition"]["kz"][
        "echo_agreement"
    ] > 0.99
    assert s["acquisition"] == "3d" and s["partition"]["cells_per_volume"] == 12
    assert abs(s["breathing_rate_per_min_median"] - F_RESP * 60) < 1.0
    assert s["cardiac_3d"]["found"]
    for ext in (".slibase.1D", ".nuisance.1D", ".partition_shift.1D", ".traces.tsv"):
        assert (tmp_path / "o" / f"run{ext}").exists()
    assert np.loadtxt(tmp_path / "o" / "run.partition_shift.1D").shape == (150, 3)
    with pytest.raises(SystemExit, match="per-shot"):
        main(
            [
                *args,
                *phase,
                "-prefix",
                str(tmp_path / "o2"),
                "-TR",
                "0.05",
                "-device",
                "cpu",
                "-verb",
                "0",
            ]
        )
    with pytest.raises(SystemExit, match="needs -phase"):
        main([*args, "-prefix", str(tmp_path / "o3"), "-device", "cpu", "-verb", "0"])


def test_excitation_view_evaluates_every_slice_at_the_kspace_centre(run):
    mags, phs = run
    pt = physio_3d.partition_field_traces(mags, phs, TR, RZ, TES, device=CPU)
    tv = physio_3d.excitation_view(pt, 36)
    assert set(tv.group) == {pt.centre_cell}
    assert abs(pt.centre_offset - (pt.centre_cell + 0.5) * TR / 12) < 1e-9
