"""RF spike detection/repair on synthetic magnitude series with injected k-space spikes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.processing import spikehunt as S

CPU = torch.device("cpu")


def _phantom_series(nt=40, nz=4, ny=32, nx=28, seed=0, spikes=(), scale_frames=()):
    """Magnitude series of a noisy ellipse; each spike adds a k-space delta before |.|.

    ``spikes``: (slice, vol, ky, kx, amplitude) with k offsets from the centre.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:ny, :nx]
    head = (((yy - ny / 2) / (ny * 0.38)) ** 2 + ((xx - nx / 2) / (nx * 0.38)) ** 2) < 1
    base = 100.0 * head + 5.0
    out = np.empty((nt, nz, ny, nx), dtype=np.float32)
    for t in range(nt):
        for z in range(nz):
            img = base * (1 + 0.01 * np.sin(0.3 * t + z)) + rng.normal(0, 3, (ny, nx))
            img = img + 1j * rng.normal(0, 3, (ny, nx))
            k = np.fft.fftshift(np.fft.fft2(img))
            for s, v, ky, kx, amp in spikes:
                if s == z and v == t:
                    k[ny // 2 + ky, nx // 2 + kx] += amp
            out[t, z] = np.abs(np.fft.ifft2(np.fft.ifftshift(k)))
    for t, f in scale_frames:
        out[t] *= f
    return torch.from_numpy(out)


# After |.| the spike is split into a mirrored pair and spread by the head: 1.6e4
# lands ~40x the per-k noise of the magnitude image, a moderate spike (the
# real ones on the test runs score up to 45). Below ~10x the log-blob score
# cannot separate a sharp spike from noise (measured: 4e3 scores 1-5 against
# 5 for noise frames). Near the image's DC term (~5e4) a spike would rectify
# into a background offset, which no real spike does.
AMP = 1.6e4


def test_detects_injected_spike_and_nothing_else():
    amp = AMP
    data = _phantom_series(spikes=[(2, 17, 13, -9, amp), (0, 30, -12, 6, amp)])
    scan = S.hunt_spikes(data, compute_device=CPU, progress=False)
    assert sorted(map(tuple, scan.hits.tolist())) == [(0, 30), (2, 17)]
    # the peak sits on the spike or its Hermitian mirror (magnitude image is real)
    assert abs(abs(scan.k_y[2, 17]) - 13) <= 1 and abs(abs(scan.k_x[2, 17]) - 9) <= 1


def test_global_intensity_step_is_not_a_spike():
    """A whole-volume brightness change (non-steady state, drift) is absorbed."""
    data = _phantom_series(scale_frames=[(0, 1.15), (1, 1.08)])
    scan = S.hunt_spikes(data, compute_device=CPU, progress=False)
    assert len(scan.hits) == 0


def test_clean_repairs_only_hit_frames_and_leaves_input_alone():
    data = _phantom_series(spikes=[(1, 20, 10, 7, AMP)])
    before = data.clone()
    out = data.clone()
    scan = S.hunt_spikes(data, clean=True, out=out, compute_device=CPU, progress=False)
    # on CPU float32, .to() aliases: the input must still be untouched
    assert torch.equal(data, before)
    assert scan.repair == {(1, 20): "kclip"}
    changed = (out != data).flatten(-2).any(-1)  # (T, Z)
    assert changed.nonzero().tolist() == [[20, 1]]
    # judged inside the head: in the noise-floor background |.| rectifies the
    # stripe into a DC offset (+11 here), and DC is never touched
    ny, nx = data.shape[-2:]
    yy, xx = np.mgrid[:ny, :nx]
    head = torch.from_numpy(
        (((yy - ny / 2) / (ny * 0.3)) ** 2 + ((xx - nx / 2) / (nx * 0.3)) ** 2) < 1
    )

    def dev(x, t):
        return float((x[t, 1] - 0.5 * (x[t - 1, 1] + x[t + 1, 1]))[head].std())

    normal = np.median([dev(data, t) for t in range(2, 38) if abs(t - 20) > 1])
    assert dev(data, 20) > 2.5 * normal
    assert dev(out, 20) < 1.25 * normal
    assert scan.residual is not None and len(scan.residual) == 0


def test_slice_repair_interpolates_across_consecutive_hits():
    nbrs = S._neighbours([5, 6], nt=10)
    assert nbrs[5] == (4, 7, pytest.approx(1 / 3))
    assert nbrs[6] == (4, 7, pytest.approx(2 / 3))
    assert S._neighbours([0], nt=4)[0][:2] == (1, 1)


def test_acquisition_window_and_mb_events():
    # MB2, 4 slices: slices 0/2 and 1/3 share an excitation; TR 2 s with a gap
    acq = S.acquisition_window(2.0, [0.0, 0.5, 0.0, 0.5])
    assert acq.ta == pytest.approx(1.0)
    assert acq.gap == pytest.approx(1.0)
    score = np.zeros((4, 10))
    score[0, 3] = score[2, 3] = 20.0  # one excitation, both MB partners
    score[1, 7] = 15.0
    scan = S.SpikeScan(
        score=score,
        raw=score,
        k_y=np.zeros((4, 10), int),
        k_x=np.zeros((4, 10), int),
        threshold=10.0,
        hits=np.argwhere(score > 10.0),
    )
    events = S.group_events(scan, acq)
    assert [(e.volume, e.slices) for e in events] == [(3, [0, 2]), (7, [1])]
    assert events[1].time == pytest.approx(7 * 2.0 + 0.5)
    assert events[1].within_tr == pytest.approx(0.5)


def test_no_slice_timing_reports_volume_onsets():
    acq = S.acquisition_window(1.5, None)
    assert acq.ta == 1.5 and acq.gap == 0.0
