"""Physiological noise from the data (processing/physio.py).

Synthetic multiband runs carry a heartbeat in the magnitude and breathing in the
phase, each sampled at every slice's own acquisition time -- so the volume rate
aliases the heartbeat and only an excitation-rate reading can recover it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.processing import physio

CPU = torch.device("cpu")
TR = 1.0
F_CARD = 1.15  # Hz: aliased to 0.15 Hz at the 1 Hz volume rate
F_RESP = 0.25


def _mb_slice_times(nz: int, mb: int) -> np.ndarray:
    """Interleaved multiband timing: slice k and k + nz/mb share an excitation."""
    n_exc = nz // mb
    order = list(range(0, n_exc, 2)) + list(range(1, n_exc, 2))
    st = np.zeros(nz)
    for pos, g in enumerate(order):
        for m in range(mb):
            st[g + m * n_exc] = pos * TR / n_exc
    return st


def _synthetic_run(nt=240, shape=(12, 12, 20), mb=2, card=0.01, resp=0.15, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    st = _mb_slice_times(nz, mb)
    t = np.arange(nt)[None, :] * TR + st[:, None]  # (nz, nt) acquisition times
    base = 100.0 + 20.0 * rng.random(shape)
    heart = card * np.cos(2 * np.pi * F_CARD * t)  # fractional
    breath = resp * np.sin(2 * np.pi * F_RESP * t)  # radians of field phase
    mag = base[..., None] * (
        1.0 + heart[None, None] + noise * rng.standard_normal((nx, ny, nz, nt))
    )
    static = rng.uniform(-np.pi, np.pi, shape)
    ph = static[..., None] + breath[None, None] + 0.02 * rng.standard_normal((nx, ny, nz, nt))
    ph = np.angle(np.exp(1j * ph)) * (4096 / np.pi)  # raw Siemens units, wrapped
    return mag.astype(np.float32), ph.astype(np.float32), st, t


def test_excitation_groups_order_by_time_and_tolerate_rounding():
    st = np.array([0.5, 0.0, 0.25, 0.5004, 0.0, 0.2499])
    group, times = physio.excitation_groups(st)
    assert list(group) == [2, 0, 1, 2, 0, 1]
    np.testing.assert_allclose(times, [0.0, 0.2499, 0.5])


def test_trailing_noise_volumes_counts_only_the_collapsed_tail():
    means = np.r_[np.full(50, 100.0), 98.0, 3.0, 2.0, 4.0]
    assert physio.trailing_noise_volumes(means) == 3
    assert physio.trailing_noise_volumes(np.full(20, 100.0)) == 0


def test_heart_rate_comes_from_the_excitation_rate_not_the_aliased_volume_rate():
    """1.15 Hz is invisible at 1 Hz sampling (it lands at 0.15 Hz); at the excitation
    rate (10 Hz here) it is recovered, and so is breathing from phase."""
    mag, ph, st, _ = _synthetic_run()
    tr_ = physio.excitation_traces(mag, st, TR, phase=ph, device=CPU)
    t, x = tr_.series("magnitude")
    _, f_card, _ = physio.track_rate(t, x, physio.CARDIAC_BAND, 20.0)
    t, x = tr_.series("phase")
    _, f_resp, _ = physio.track_rate(t, x, physio.RESP_BAND, 40.0)
    assert abs(np.median(f_card) - F_CARD) < 0.02
    assert abs(np.median(f_resp) - F_RESP) < 0.02


def test_regressors_explain_the_signal_each_slice_actually_carried():
    """Each slice's cardiac terms must span the heartbeat AT THAT SLICE'S TIMES --
    which differ by up to a TR across slices, a full cycle at 1.15 Hz."""
    mag, ph, st, t = _synthetic_run()
    tr_ = physio.excitation_traces(mag, st, TR, phase=ph, device=CPU)
    X, labels, diag = physio.slice_regressors(tr_, "magnitude", "phase")
    nz = st.size
    m = X.shape[1] // nz
    assert m == 13 and len(labels) == X.shape[1]
    assert labels[:13] == [
        "s000.rvt00", "s000.rvt01", "s000.rvt02", "s000.rvt03", "s000.rvt04",
        "s000.card.c1", "s000.card.s1", "s000.card.c2", "s000.card.s2",
        "s000.resp.c1", "s000.resp.s1", "s000.resp.c2", "s000.resp.s2",
    ]  # fmt: skip
    for k in range(nz):
        truth = np.cos(2 * np.pi * F_CARD * t[k])
        A = np.c_[np.ones(len(truth)), X[:, k * m + 5 : k * m + 9]]
        fit = A @ np.linalg.lstsq(A, truth, rcond=None)[0]
        r2 = 1 - ((truth - fit) ** 2).sum() / ((truth - truth.mean()) ** 2).sum()
        assert r2 > 0.9, f"slice {k}: cardiac terms explain only {r2:.2f} of its heartbeat"
    # Slices sharing an excitation share their regressors exactly.
    group = tr_.group
    k0, k1 = np.flatnonzero(group == group[0])[:2]
    np.testing.assert_array_equal(X[:, k0 * m : (k0 + 1) * m], X[:, k1 * m : (k1 + 1) * m])


def test_tr_comb_from_a_slow_signal_seen_with_different_gains_is_removed():
    """A slow fluctuation scaled differently in each excitation, stitched in time order,
    is a line at 1/TR -- inside the cardiac band. Regressing on the slow volume mean
    removes it."""
    rng = np.random.default_rng(1)
    nt, G = 300, 10
    slow = np.sin(2 * np.pi * 0.01 * np.arange(nt))
    gain = 1.0 + 0.5 * rng.standard_normal(G)
    trace = np.outer(slow, gain) + 0.01 * rng.standard_normal((nt, G))

    def line_power(tr_):
        x = tr_.ravel()  # (nt, G) row-major = time order
        P = np.abs(np.fft.rfft(x - x.mean())) ** 2
        f = np.fft.rfftfreq(x.size, d=TR / G)
        return P[np.abs(f - 1.0 / TR) < 0.02].sum()

    assert line_power(physio.remove_tr_comb(trace, TR)) < 0.01 * line_power(trace)


def test_heart_rate_tracker_does_not_lock_onto_a_tr_harmonic():
    """A sharp line at 2/TR, stronger than the heartbeat, must not capture the track."""
    fs = 17.0
    t = np.arange(0, 360, 1 / fs)
    rng = np.random.default_rng(2)
    x = (
        0.5 * np.cos(2 * np.pi * 1.1 * t)
        + 2.0 * np.cos(2 * np.pi * 2.0 * t)
        + rng.standard_normal(t.size)
    )
    _, f, _ = physio.track_rate(t, x, physio.CARDIAC_BAND, 20.0, avoid=(1.0, 2.0))
    assert abs(np.median(f) - 1.1) < 0.02


@pytest.mark.parametrize("with_phase", [True, False])
def test_source_choice_follows_the_measured_mapping(with_phase):
    mag, ph, st, _ = _synthetic_run(nt=120)
    tr_ = physio.excitation_traces(mag, st, TR, phase=ph if with_phase else None, device=CPU)
    card, _ = physio.choose_source(tr_, physio.CARDIAC_BAND, 20.0)
    resp, _ = physio.choose_source(tr_, physio.RESP_BAND, 40.0)
    assert card == "magnitude"
    assert resp == ("phase" if with_phase else "magnitude")


def test_cli_writes_a_slibase_that_ffs_reml_reads_slice_major(tmp_path):
    """End to end: the file's 13-per-slice blocks land on the right slices when read
    the way ffs_reml -slibase_sm (and afni_proc -ricor_regs) reads them."""
    import json

    import nibabel as nib

    from fastfuncstuff.cli.physio import main
    from fastfuncstuff.design.slibase import deinterleave_slibase, load_1d_matrix

    mag, ph, st, _ = _synthetic_run(nt=150)
    mag = np.concatenate([mag, np.full(mag.shape[:3] + (2,), 1.0, np.float32)], -1)  # noise scans
    ph = np.concatenate([ph, np.zeros(ph.shape[:3] + (2,), np.float32)], -1)
    aff = np.eye(4)
    nib.save(nib.Nifti1Image(mag, aff), tmp_path / "run_mag.nii.gz")
    nib.save(nib.Nifti1Image(ph, aff), tmp_path / "run_ph.nii.gz")
    (tmp_path / "run_mag.json").write_text(
        json.dumps({"SliceTiming": st.tolist(), "RepetitionTime": TR})
    )
    prefix = tmp_path / "out" / "run"
    rc = main(["-input", str(tmp_path / "run_mag.nii.gz"), "-phase", str(tmp_path / "run_ph.nii.gz"),
               "-prefix", str(prefix), "-device", "cpu", "-verb", "0", "-no_qc"])  # fmt: skip
    assert rc == 0
    summary = json.loads((tmp_path / "out" / "run.json").read_text())
    assert summary["dropped_last"] == 2 and summary["n_volumes"] == 150
    assert abs(summary["heart_rate_per_min_median"] - F_CARD * 60) < 1.5
    mat = load_1d_matrix(tmp_path / "out" / "run.slibase.1D")
    assert mat.shape == (150, st.size * 13)
    blocks = deinterleave_slibase(mat, st.size, slice_major=True)  # (n_slices, nt, 13)
    group, _ = physio.excitation_groups(st)
    k0, k1 = np.flatnonzero(group == group[0])[:2]
    torch.testing.assert_close(blocks[k0], blocks[k1])  # same excitation, same regressors
    other = int(np.flatnonzero(group != group[0])[0])
    assert not torch.allclose(blocks[k0][:, 5:9], blocks[other][:, 5:9])  # different times


def test_regress_out_removes_the_heartbeat_and_keeps_mean_and_drift():
    """Cleaning must take the physiology and nothing a later step relies on: each
    voxel's mean and slow drift survive, the heartbeat at its slice's times goes."""
    mag, ph, st, t = _synthetic_run(nt=200, noise=0.002)
    drift = np.linspace(0.0, 5.0, mag.shape[-1], dtype=np.float32)
    mag = mag + drift
    tr_ = physio.excitation_traces(mag, st, TR, phase=ph, device=CPU)
    X, _, _ = physio.slice_regressors(tr_, "magnitude", "phase")
    clean, removed = physio.regress_out(mag, X, TR, device=CPU)
    np.testing.assert_allclose(clean.mean(-1), mag.mean(-1), rtol=1e-4)
    k = 7
    v = clean[3, 4, k]
    heart = np.cos(2 * np.pi * F_CARD * t[k])
    n = np.arange(v.size)

    def detrended(y):
        return y - np.polyval(np.polyfit(n, y, 1), n)

    before = abs(np.corrcoef(detrended(mag[3, 4, k]), heart)[0, 1])
    after = abs(np.corrcoef(detrended(v), heart)[0, 1])
    # The cardiac terms explain ~0.95 of this synthetic heartbeat, which leaves a residual
    # correlation near 0.12. An earlier < 0.1 bound passed only because the old envelope
    # RVT happened to carry a sliver of the aliased heartbeat (0.955 -> 0.994 with it).
    assert before > 0.5 and after < 0.2 and after < 0.25 * before, (before, after)
    assert abs(np.polyfit(np.arange(v.size), v, 1)[0] - drift[1]) < 0.2 * drift[1]  # drift kept
    assert 0.3 < float(np.median(removed)) < 1.0


def test_volume_regressors_at_tzero_equal_a_slice_acquired_at_tzero():
    """The GLM nuisance (one set at f*TR + tzero) and the per-slice terms are one model:
    at a slice acquired exactly at tzero all 13 columns must coincide, cardiac included."""
    mag, ph, st, _ = _synthetic_run(nt=150)
    tr_ = physio.excitation_traces(mag, st, TR, phase=ph, device=CPU)
    X, slabels, diag = physio.slice_regressors(tr_, "magnitude", "phase")
    k0 = int(np.flatnonzero(st == 0.0)[0])
    V, labels = physio.volume_regressors(diag, np.arange(150) * TR + 0.0)
    assert [f"s{k0:03d}.{lab}" for lab in labels] == slabels[k0 * 13 : (k0 + 1) * 13]
    np.testing.assert_allclose(V, X[:, k0 * 13 : (k0 + 1) * 13], atol=1e-6)
    V9, labels9 = physio.volume_regressors(diag, np.arange(150) * TR, cardiac=False)
    assert V9.shape[1] == 9 and not any(lab.startswith("card") for lab in labels9)


def test_cardiac_at_tzero_survives_slice_timing():
    """Slice timing shifts an aliased heartbeat by the wrong phase, but by a constant one
    per slice; the cardiac pair evaluated at tzero still fits the corrected data. Guards the
    reason cardiac is in the tzero nuisance (it was once left out as 'aliased')."""
    from fastfuncstuff.processing.slicetime import slicetime_correct

    nt = 240
    mag, ph, st, t = _synthetic_run(nt=nt, card=0.02, noise=0.004)
    tr_ = physio.excitation_traces(mag, st, TR, phase=ph, device=CPU)
    _, _, diag = physio.slice_regressors(tr_, "magnitude", "phase")
    tzero = float(st.mean())
    vol = torch.from_numpy(np.ascontiguousarray(mag.transpose(3, 2, 1, 0)))
    corr = slicetime_correct(vol, list(st), TR, tzero=tzero, method="heptic", device=CPU)
    corr = corr.numpy().transpose(3, 2, 1, 0)
    V, labels = physio.volume_regressors(diag, np.arange(nt) * TR + tzero)
    card = V[:, [i for i, lab in enumerate(labels) if lab.startswith("card")]]
    n = np.arange(nt, dtype=float)
    base = np.column_stack([np.ones(nt), n, n**2])
    for k in (0, st.size // 3, st.size - 1):  # slices far from tzero included
        y = corr[3, 4, k]
        r0 = y - base @ np.linalg.lstsq(base, y, rcond=None)[0]
        A = np.column_stack([base, card])
        r1 = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        assert 1 - (r1 @ r1) / (r0 @ r0) > 0.5, k


def test_response_functions_have_their_published_shapes():
    """Birn 2008 RRF peaks near 3 s and undershoots near 14 s; Chang 2009 CRF peaks
    near 4 s and undershoots near 12 s -- a typo in a constant moves these."""
    t = np.arange(0, 40, 0.01)
    rrf, crf = physio.respiration_response(t), physio.cardiac_response(t)
    assert 2.5 < t[np.argmax(rrf)] < 4.0 and 10 < t[np.argmin(rrf)] < 18 and rrf.min() < 0
    assert 3.0 < t[np.argmax(crf)] < 5.5 and 10 < t[np.argmin(crf)] < 14 and crf.min() < 0


def test_beat_phase_is_linear_between_beats_and_counts_cycles():
    peaks = np.array([0.0, 1.0, 1.8, 3.0])
    ph = physio.beat_phase(peaks, np.array([0.0, 0.5, 1.0, 1.4, 1.8, 2.4]))
    np.testing.assert_allclose(ph / (2 * np.pi), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])


def test_breath_rvt_follows_depth_and_rate():
    """A breath twice as deep, or twice as fast, doubles RVT."""
    fs = 50.0
    t = np.arange(0, 240, 1 / fs)
    depth = np.where(t < 120, 1.0, 2.0)
    x = depth * np.sin(2 * np.pi * 0.25 * t)
    bt, rvt = physio.breath_rvt(t, x, 0.25)
    early, late = np.median(rvt[(bt > 20) & (bt < 100)]), np.median(rvt[(bt > 140) & (bt < 220)])
    assert 1.8 < late / early < 2.2


def test_histogram_phase_spans_minus_pi_to_pi_and_tracks_the_cycle():
    fs = 50.0
    t = np.arange(0, 120, 1 / fs)
    x = np.sin(2 * np.pi * 0.25 * t)
    ph = physio.respiration_phase_histogram(t, x)
    assert ph.min() < -2.8 and ph.max() > 2.8
    # the cos of the phase follows the breath (peak inhale <-> phase near +-pi... sign-free check)
    assert abs(np.corrcoef(np.cos(ph[500:-500]), x[500:-500])[0, 1]) > 0.8


def test_standard_regressors_tapas_layout():
    """c3 r4 cr1 + HRV + RVT: TAPAS PhysIO's common 20-column set, in its group order."""
    fs = 100.0
    t = np.arange(0, 200, 1 / fs)
    card = np.sin(2 * np.pi * 1.1 * t) ** 21  # pulse-like
    resp = np.sin(2 * np.pi * 0.25 * t)
    model = physio.model_from_series(t, card, t, resp, source="test")
    X, labels = physio.standard_regressors(model, np.arange(10, 190, 1.0))
    assert X.shape == (180, 20) and np.isfinite(X).all()
    assert labels[:6] == ["card.c1", "card.s1", "card.c2", "card.s2", "card.c3", "card.s3"]
    assert labels[14:] == ["cxr.c1p", "cxr.s1p", "cxr.c1m", "cxr.s1m", "hrv.crf", "rvt.rrf"]


def test_field_gradient_trace_follows_breath_depth():
    """Breathing that reaches the head as an anterior-posterior field GRADIENT, its depth
    changing slowly: the field-gradient trace must follow that depth. (On real data the
    mean phase does not -- 91 TASKFORCE runs, belt RVT after RRF: gradient component
    +0.71, mean phase +0.20 -- but a synthetic this clean lets the mean see it too, so
    that half is not asserted here.)"""
    rng = np.random.default_rng(4)
    nt, (nx, ny, nz), mb = 240, (16, 16, 20), 2
    st = _mb_slice_times(nz, mb)
    t = np.arange(nt)[None, :] * TR + st[:, None]
    depth = 1.0 + 0.6 * np.sin(2 * np.pi * t / 80.0)  # breath depth, slow
    breath = 0.2 * depth * np.sin(2 * np.pi * F_RESP * t)  # rad per unit of y
    y = (np.arange(ny) - ny / 2) / (ny / 2)
    mag = (100.0 + rng.random((nx, ny, nz)))[..., None] * np.ones(nt, np.float32)
    mag = mag * (1 + 0.005 * rng.standard_normal(mag.shape))
    static = rng.uniform(-np.pi, np.pi, (nx, ny, nz))
    ph = (
        static[..., None]
        + y[None, :, None, None] * breath[None, None]
        + 0.02 * rng.standard_normal(mag.shape)
    )
    ph = np.angle(np.exp(1j * ph)) * (4096 / np.pi)
    traces = physio.excitation_traces(
        mag.astype(np.float32),
        st,
        TR,
        phase=ph.astype(np.float32),
        device=CPU,
        voxel_size=(2.0, 2.0, 2.0),
    )
    tf, xf = physio.field_breathing_trace(traces)
    bt, rvt = physio.breath_rvt(tf, xf, F_RESP)
    truth = 1.0 + 0.6 * np.sin(2 * np.pi * bt / 80.0)
    keep = (bt > 20) & (bt < nt - 20)
    assert np.corrcoef(rvt[keep], truth[keep])[0, 1] > 0.8
