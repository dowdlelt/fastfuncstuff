"""ROI x depth curves from ffs_deconvolve -> the laminar model, refereed by the reference.

The MATLAB driver's own design (two conditions, 20 masked noise rows, one event each,
a modulatory pulse on the second) is in the inversion oracle. Writing its two responses
out the way ffs_deconvolve does -- tidy TSV, depth 1 deepest -- and reading them back
through the adapter must give the same input u, the same mask and the same data on
every fitted row. That pins the depth flip, the block layout and the timing at once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fastfuncstuff.laminar.forward import apply_depth_psf
from fastfuncstuff.laminar.io import pick_roi, read_iresp_table, two_condition_data

ORACLE = Path(__file__).parent / "laminar_oracle" / "laminar_inversion_oracle.json"


def _write_table(path, times, values_deep_first, name="V1", hemi="lh", cover=None):
    k = values_deep_first.shape[1]
    lines = ["time\troi\tname\themi\tdepth\tdepth_frac\tn_columns\tcoverage\tbeta"]
    for i, t in enumerate(times):
        for d in range(k):
            c = 1.0 if cover is None else cover[d]
            lines.append(
                f"{t:g}\t1\t{name}\t{hemi}\t{d + 1}\t{(d + 0.5) / k:.4g}\t100\t{c}\t"
                f"{values_deep_first[i, d]:.9g}"
            )
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def oracle():
    if not ORACLE.exists():  # pragma: no cover
        pytest.skip(f"no oracle at {ORACLE}")
    o = json.loads(ORACLE.read_text())
    y = np.asarray(o["y"], float).reshape(o["ns"], o["K"])
    return o, y


def test_the_reference_design_comes_back_from_deconvolve_style_tables(oracle, tmp_path):
    o, y = oracle
    tr, k = float(o["TR"]), int(o["K"])
    first, second = y[1:8], y[29:36]  # the reference's two responses, superficial first
    times = np.arange(7) * tr
    a = _write_table(tmp_path / "a_iresp_c1.tsv", times, first[:, ::-1])  # deep first, as we write
    b = _write_table(tmp_path / "b_iresp_c3.tsv", times, second[:, ::-1])
    data = two_condition_data(
        pick_roi(read_iresp_table(a), "V1"), pick_roi(read_iresp_table(b), "V1"), duration=1.6,
        delay_seconds=None, modulation="reference",
    )  # fmt: skip
    u_ref = torch.tensor(np.asarray(o["u"], float)).reshape(-1, 2)
    rows_ref = torch.tensor(np.asarray(o["mask_rows"], float)).reshape(-1).bool()
    assert data.spec.K == k and data.spec.TR == pytest.approx(tr)
    assert torch.equal(data.rows, rows_ref)
    assert torch.equal(data.u[: u_ref.shape[0]], u_ref) and data.u.shape[0] >= u_ref.shape[0]
    np.testing.assert_allclose(data.y[data.rows].numpy(), y[rows_ref.numpy()], rtol=1e-8)
    assert data.integrate_kwargs == {}  # None restores the reference's TR/2 delay


def test_tent_curves_default_to_no_sampling_delay(tmp_path):
    times = np.arange(8) * 2.0
    vals = np.random.default_rng(0).normal(0, 0.3, (8, 7))
    t = read_iresp_table(_write_table(tmp_path / "x.tsv", times, vals))
    data = two_condition_data(pick_roi(t, "V1"), pick_roi(t, "V1"), duration=2.0)
    assert data.integrate_kwargs == {"delay_seconds": 0.0} and data.spec.TR == 2.0
    np.testing.assert_allclose(data.y[1:9].numpy(), vals[:, ::-1])  # superficial first


def test_data_units_cut_profiles_and_uneven_knots_are_refused(tmp_path):
    times = np.arange(6) * 2.0
    vals = np.full((6, 7), 1000.0)
    t = read_iresp_table(_write_table(tmp_path / "raw.tsv", times, vals))
    with pytest.raises(ValueError, match="data units"):
        two_condition_data(pick_roi(t, "V1"), pick_roi(t, "V1"), duration=2.0)
    cut = read_iresp_table(
        _write_table(tmp_path / "cut.tsv", times, vals * 1e-3, cover=[1, 1, 1, 1, 1, 1, 0.7])
    )
    with pytest.raises(ValueError, match="inside the EPI"):
        two_condition_data(pick_roi(cut, "V1"), pick_roi(cut, "V1"), duration=2.0)
    uneven = read_iresp_table(
        _write_table(tmp_path / "u.tsv", np.r_[0, 2, 4, 7.0], vals[:4] * 1e-3)
    )
    with pytest.raises(ValueError, match="evenly spaced"):
        two_condition_data(pick_roi(uneven, "V1"), pick_roi(uneven, "V1"), duration=2.0)


def test_a_matrix_depth_kernel_is_a_plain_mixing_and_the_gaussian_path_is_untouched():
    lbr = torch.randn(4, 5, 7, dtype=torch.float64)
    np.testing.assert_allclose(apply_depth_psf(lbr, torch.eye(7, dtype=torch.float64)), lbr)
    m = torch.rand(7, 7, dtype=torch.float64)
    np.testing.assert_allclose(apply_depth_psf(lbr, m), torch.einsum("kj,...j->...k", m, lbr))
    with pytest.raises(ValueError, match=r"\(K, K\)"):
        apply_depth_psf(lbr, torch.eye(5, dtype=torch.float64))
    g = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
    assert apply_depth_psf(lbr, g).shape == lbr.shape  # the reference's 1-D Gaussian


def test_modulation_defaults_to_the_event_and_keeps_the_reference_pulse():
    from fastfuncstuff.laminar.io import two_condition_batch

    times = np.arange(8) * 1.85
    curves = np.zeros((1, 8, 5))
    ref = two_condition_batch(curves, curves, times, duration=1.0, gap=4, modulation="reference")
    ev = two_condition_batch(curves, curves, times, duration=1.0, gap=4)
    dt = ref.spec.dt
    on_ref = np.flatnonzero(ref.u[:, 0].numpy() > 0) * dt
    on_ev = np.flatnonzero(ev.u[:, 0].numpy() > 0) * dt
    onset_b = (ref.meta["block"] + 4 + 1) * 1.85
    # Reference: 0.1 s ending one sample after the second event (Faes's fourth tone).
    assert on_ref.min() == pytest.approx(onset_b + 1.85 - 0.1, abs=dt)
    assert on_ref.max() - on_ref.min() < 0.1
    # Event (the default): the whole second event; boxcars are end-inclusive.
    assert on_ev.min() == pytest.approx(onset_b, abs=dt)
    assert on_ev.max() == pytest.approx(onset_b + 1.0, abs=dt)
    span = two_condition_batch(curves, curves, times, duration=1.0, gap=4, modulation=(0.0, 1.0))
    torch.testing.assert_close(span.u, ev.u)
    # The driving input is untouched.
    torch.testing.assert_close(ref.u[:, 1], ev.u[:, 1])


def test_two_conditions_can_have_different_event_lengths():
    """The driving input carries each event's own length, and -modulation event
    follows the TEST event, since that is the event it modulates."""
    from fastfuncstuff.laminar.io import two_condition_batch

    times = np.arange(10) * 1.0
    curves = np.zeros((1, times.size, 3))
    one = two_condition_batch(curves, curves, times, duration=1.0, gap=4)
    pair = two_condition_batch(curves, curves, times, duration=(1.0, 4.0), gap=4)
    scalar_as_list = two_condition_batch(curves, curves, times, duration=[1.0], gap=4)

    # One value and a one-element list mean the same thing.
    assert torch.equal(one.u, scalar_as_list.u)
    # u is (n_micro, 2): modulatory first, then driving (both events).
    drive_one, drive_pair = one.u[:, 1], pair.u[:, 1]
    assert drive_pair.sum() > drive_one.sum(), "the longer test event should drive for longer"
    # The baseline event is unchanged; only the test event grew.
    half = drive_one.numel() // 2
    assert torch.equal(drive_pair[:half], drive_one[:half])
    # -modulation event follows the test event's own length.
    assert pair.u[:, 0].sum() > one.u[:, 0].sum()
    # ... and equals the test event's own boxcar when they coincide.
    assert pair.u[:, 0].sum() == pytest.approx(float(drive_pair[half:].sum()), rel=1e-6)


def test_duration_pair_is_validated():
    from fastfuncstuff.laminar.io import two_condition_batch

    times = np.arange(10) * 1.0
    curves = np.zeros((1, times.size, 3))
    with pytest.raises(ValueError, match="one value or two"):
        two_condition_batch(curves, curves, times, duration=(1.0, 2.0, 3.0), gap=4)
    with pytest.raises(ValueError, match="must be positive"):
        two_condition_batch(curves, curves, times, duration=(1.0, 0.0), gap=4)


def test_a_window_that_starts_after_the_onset_is_refused():
    """The event sits on the response's first sample, so a window starting at BOT > 0
    used to produce a model bit-identical to the same curves at BOT = 0 -- the offset
    was discarded and the response modelled as if it began at the event."""
    from fastfuncstuff.laminar.io import two_condition_batch

    curves = np.zeros((1, 8, 3))
    two_condition_batch(curves, curves, np.arange(8) * 2.0, duration=1.0, gap=4)  # bot 0: fine
    with pytest.raises(ValueError, match="start.*after the event|starts at 0"):
        two_condition_batch(curves, curves, np.arange(8) * 2.0 + 6.0, duration=1.0, gap=4)
