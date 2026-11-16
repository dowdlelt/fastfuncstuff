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
    scan = S.hunt_spikes(
        data, clean=True, out=out, method="kclip", compute_device=CPU, progress=False
    )
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


def test_moco_hook_reports_scan_volumes_and_repairs_in_place(tmp_path, monkeypatch):
    """-spikehunt clean: hits in scan numbering (after -skip_first), data repaired
    before estimation, timing from the JSON sidecar."""
    import json

    import nibabel as nib

    from fastfuncstuff.cli import moco

    monkeypatch.chdir(tmp_path)
    data = _phantom_series(spikes=[(2, 17, 13, -9, AMP)])
    img = nib.Nifti1Image(data.numpy().transpose(3, 2, 1, 0), np.diag([3.0, 3.0, 3.0, 1.0]))
    img.header.set_xyzt_units("mm", "sec")
    img.header["pixdim"][4] = 2.0
    nib.save(img, tmp_path / "epi.nii.gz")
    (tmp_path / "epi.json").write_text(
        json.dumps({"RepetitionTime": 2.0, "SliceTiming": [0.0, 0.4, 0.8, 1.2]})
    )
    args = moco.parse_args(
        [
            "-input",
            "epi.nii.gz",
            "-1Dfile",
            "m.1D",
            "-prefix",
            "mc",
            "-spikehunt",
            "clean",
            "-skip_first",
            "2",
        ]  # fmt: skip
    )
    trimmed = data[2:].clone()
    moco._spikehunt(args, trimmed, "epi.nii.gz", CPU, verb=0)
    rows = (tmp_path / "mc_spikes.tsv").read_text().splitlines()
    assert len(rows) == 2
    vol, sl, time_s = rows[1].split("\t")[:3]
    assert (int(vol), int(sl)) == (17, 2)
    assert float(time_s) == pytest.approx(17 * 2.0 + 0.8)
    assert not torch.equal(trimmed[15, 2], data[17, 2])
    assert torch.equal(trimmed[14], data[16])
    assert "mc_spikes.tsv" in moco._expected_outputs(args)


def test_moco_spikehunt_refuses_multi_echo():
    from fastfuncstuff.cli import moco

    args = moco.parse_args(
        [
            "-input",
            "e1.nii.gz",
            "e2.nii.gz",
            "-reg_echo",
            "1",
            "-1Dfile",
            "m.1D",
            "-spikehunt",
            "detect",
        ]  # fmt: skip
    )
    with pytest.raises(SystemExit):
        moco._run_multi_echo(args, args.input_file, CPU, 0, False, 0)


def test_localized_inflow_blob_is_not_a_spike():
    """A bright localized change in the image (inflow, a vessel) is broad, low-k
    energy in k-space: whatever its score, the point check must reject it."""
    data = _phantom_series(spikes=[(2, 17, 13, -9, AMP)])
    ny, nx = data.shape[-2:]
    yy, xx = np.mgrid[:ny, :nx]
    blob = 60.0 * np.exp(-(((yy - 10) / 2.0) ** 2 + ((xx - 9) / 2.0) ** 2))
    data[25, 1] += torch.from_numpy(blob.astype(np.float32))
    scan = S.hunt_spikes(data, compute_device=CPU, progress=False)
    assert (1, 25) not in set(map(tuple, scan.hits.tolist()))
    assert (2, 17) in set(map(tuple, scan.hits.tolist()))


def test_kinterp_replaces_few_kpoints_and_only_in_the_hit_frame():
    data = _phantom_series(spikes=[(1, 20, 10, 7, AMP)])
    out = data.clone()
    scan = S.hunt_spikes(
        data, clean=True, out=out, method="kinterp", compute_device=CPU, progress=False
    )
    assert scan.repair == {(1, 20): "kinterp"}
    assert (out != data).flatten(-2).any(-1).nonzero().tolist() == [[20, 1]]
    # ~8% on the real runs; the phantom's rectified background spreads more
    # excess (measured 0.375), so this only guards against a whole-frame swap
    assert scan.clipped_frac[(1, 20)] < 0.5
    ref = 0.5 * (data[19, 1] + data[21, 1])
    assert (out[20, 1] - ref).std() < (data[20, 1] - ref).std()


def test_trailing_noise_volumes_are_counted_and_left_out():
    data = _phantom_series(spikes=[(2, 17, 13, -9, AMP)])
    data[-3:] = torch.rand_like(data[-3:]) * 4.0  # flip-angle-0 noise scans
    assert S.trailing_noise_volumes(data) == 3
    assert S.trailing_noise_volumes(data[:-3]) == 0
    scan = S.hunt_spikes(data[:-3], compute_device=CPU, progress=False)
    assert (2, 17) in set(map(tuple, scan.hits.tolist()))
