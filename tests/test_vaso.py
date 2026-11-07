"""Tests for the SS-SI VASO primitives (processing/vaso.py)."""

import numpy as np
import pytest
import torch

from fastfuncstuff.processing import vaso


def _labelled(nt: int, first: str) -> torch.Tensor:
    """(nt, 1, 1, 2) series whose voxel 0 is the raw index and voxel 1 the contrast."""
    d = torch.zeros(nt, 1, 1, 2)
    d[:, 0, 0, 0] = torch.arange(nt, dtype=torch.float32)
    is_nulled = (torch.arange(nt) % 2 == 0) == (first == vaso.NULLED)
    d[:, 0, 0, 1] = is_nulled.float()
    return d


@pytest.mark.parametrize("first", vaso.CONTRASTS)
@pytest.mark.parametrize("trim", [0, 3, 4, 5])
@pytest.mark.parametrize("nt", [20, 21])
def test_split_streams_pairs_start_on_nulled(first, trim, nt):
    s = vaso.split_streams(_labelled(nt, first), first=first, trim=trim)
    assert torch.all(s.nulled[:, 0, 0, 1] == 1)
    assert torch.all(s.bold[:, 0, 0, 1] == 0)
    # BOLD of pair k is the volume right after nulled k.
    assert torch.equal(s.bold[:, 0, 0, 0], s.nulled[:, 0, 0, 0] + 1)
    assert s.n_discarded >= trim and s.n_discarded - trim <= 1
    assert s.n_discarded + 2 * s.n_pairs + s.n_tail == nt
    assert s.n_tail in (0, 1)


def test_split_streams_rejects_too_short():
    with pytest.raises(ValueError, match="pair"):
        vaso.split_streams(_labelled(6, vaso.NULLED), trim=4)


def test_detect_steady_state_finds_both_streams():
    rng = np.random.default_rng(0)
    n = 80
    # Each parity relaxes to its own level; the odd stream takes longer.
    k = np.arange(n // 2)
    even = 500 + 400 * np.exp(-k / 0.6)
    odd = 450 - 150 * np.exp(-k / 1.5)
    means = np.empty(n)
    means[0::2], means[1::2] = even, odd
    means += rng.normal(0, 0.3, n)
    trim = vaso.detect_steady_state(means)
    # odd stream within 1% (4.5) of 450 once 150*exp(-k/1.5) < 4.5: k >= 6 -> raw 13
    assert trim == 13


def test_shift_series_recovers_a_delayed_sinusoid():
    """A BOLD stream sampled `delay` after the anchor lands on the anchor times."""
    pair_tr, delay, n = 2.5, 0.9, 120
    t = np.arange(n) * pair_tr
    f = lambda x: np.sin(2 * np.pi * x / 40.0) + 0.3 * np.cos(2 * np.pi * x / 23.0)  # noqa: E731
    acquired = torch.tensor(f(t + delay), dtype=torch.float32)[:, None, None, None]
    out = vaso.shift_series(acquired, delay / pair_tr, verb=0)[:, 0, 0, 0].numpy()
    # First output needs a value from before the first acquisition: a copy of it.
    assert out[0] == pytest.approx(float(acquired[0]))
    err = np.abs(out[1:] - f(t[1:]))
    # Heptic Lagrange on a signal at ~1/16 of the sampling rate: well under 1%.
    assert err.max() < 5e-3
    # Edge padding keeps the last few outputs as good as the middle (zero padding
    # would drag them towards 0 by tens of percent).
    assert err[-4:].max() < 5e-3


def test_shift_series_negative_shift_copies_last():
    d = torch.arange(10, dtype=torch.float32)[:, None, None, None]
    out = vaso.shift_series(d, -0.4, verb=0)[:, 0, 0, 0]
    assert out[-1] == 9
    # A linear ramp is reproduced exactly by any Lagrange order.
    assert torch.allclose(out[:-1], torch.arange(9, dtype=torch.float32) + 0.4, atol=1e-4)


def test_bold_correct_matches_ln_boco_rules():
    nulled = torch.tensor([[1.0, 0.0, 2.0, -1.0, 10.0, 3.0]])
    bold = torch.tensor([[2.0, 1.0, 0.0, 1.0, 1.0, 4.0]])
    out = vaso.bold_correct(nulled, bold)
    assert out.tolist() == [[0.5, 0.0, 0.0, 0.0, 5.0, 0.75]]


def test_compare_motion_offset_and_alignment():
    """Same smooth motion seen a volume-TR later, plus a base-pose offset."""
    pair_tr, delay, n = 2.5, 0.9, 150
    t = np.arange(n) * pair_tr
    traj = lambda x: np.stack([np.sin(x / (9.0 + i)) * (0.2 + 0.1 * i) for i in range(6)], 1)  # noqa: E731
    offset = np.array([0.01, -0.02, 0.0, 0.05, 0.0, -0.03])
    mn = traj(t)
    mb = traj(t + delay) - offset
    cmp = vaso.compare_motion(mn, mb, delay / pair_tr)
    np.testing.assert_allclose(cmp.offset, offset, atol=2e-3)
    assert np.all(cmp.r_aligned > 0.9999)
    assert np.all(cmp.r_aligned >= cmp.r_raw)
    assert np.all(cmp.rms_diff < 2e-3)
    assert "roll" in vaso.format_motion_comparison(cmp)


def test_brightness_check_orders_streams():
    nulled = torch.full((4, 2, 3, 3), 80.0)
    bold = torch.full((4, 2, 3, 3), 100.0)
    n, b = vaso.brightness_check(nulled, bold)
    assert n < b


def test_anchor_time_matches_the_acquisition_clock():
    pair_tr, delay = 2.57, 0.895
    # raw times when volume 0 is nulled: nulled at k*P, BOLD at k*P + delay
    assert vaso.anchor_time(4, vaso.NULLED, pair_tr, delay) == pytest.approx(2 * pair_tr)
    # raw 0 BOLD at 0, raw 1 nulled at P - delay, raw 5 nulled at 2P + P - delay
    assert vaso.anchor_time(1, vaso.BOLD, pair_tr, delay) == pytest.approx(pair_tr - delay)
    assert vaso.anchor_time(5, vaso.BOLD, pair_tr, delay) == pytest.approx(3 * pair_tr - delay)


# --- ffs_moco -vaso end to end -------------------------------------------------


def _vaso_run(nt: int = 15) -> torch.Tensor:
    """Static interleaved run (nulled first) whose true VASO ratio is 0.8 in tissue."""
    nz, ny, nx = 8, 12, 12
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    r2 = (xx - 5.5) ** 2 + (yy - 5.5) ** 2 + (zz - 3.5) ** 2
    bold = 100.0 * torch.exp(-r2 / 18.0) + 20.0
    data = torch.empty(nt, nz, ny, nx)
    data[0::2] = 0.8 * bold
    data[1::2] = bold
    return data


def _main_vaso(tmp_path, name: str, *extra: str):
    from fastfuncstuff.cli.moco import main
    from fastfuncstuff.processing.io import save_image

    inp = tmp_path / name
    save_image(_vaso_run(), str(inp), header_info=None)
    prefix = tmp_path / "out.nii.gz"
    main(
        ["-input", str(inp), "-prefix", str(prefix), "-device", "cpu", "-no_compile"]
        + ["-verb", "0", *extra]
    )
    return tmp_path / "out"


def test_moco_vaso_cli_writes_aligned_streams(tmp_path):
    import json

    import nibabel as nib

    stem = _main_vaso(tmp_path, "run_cbv.nii.gz", "-pair_tr", "2.5", "-vaso_delay", "0.9")
    vaso_img = nib.load(f"{stem}_vaso.nii.gz")
    # 15 volumes - 4 trimmed = 11 -> 5 pairs and one unpaired tail volume
    assert vaso_img.shape[3] == 5
    assert vaso_img.header.get_zooms()[3] == pytest.approx(2.5)
    centre = np.asarray(vaso_img.dataobj)[6, 6, 4, :]
    np.testing.assert_allclose(centre, 0.8, atol=1e-3)
    meta = json.loads((tmp_path / "out_vaso.json").read_text())
    assert meta["NumberOfVolumesDiscardedByUser"] == 4
    assert meta["StartTimeOffset"] == pytest.approx(5.0)
    for suffix in ("_null.nii.gz", "_bold.nii.gz", "_null_motion.1D", "_motion_compare.tsv"):
        assert (tmp_path / f"out{suffix}").exists()
    assert (tmp_path / "mean_out_vaso.nii.gz").exists()


def test_moco_vaso_requires_pair_tr(tmp_path):
    with pytest.raises(SystemExit):
        _main_vaso(tmp_path, "run_cbv.nii.gz")


def test_moco_vaso_refuses_unsupported_options(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _main_vaso(tmp_path, "run.nii.gz", "-vaso", "-pair_tr", "2.5", "-dfile", "x.1D")
    assert "-dfile" in capsys.readouterr().err


def test_moco_no_vaso_skips_cbv_detection(tmp_path):
    _main_vaso(tmp_path, "run_cbv.nii.gz", "-no_vaso")
    assert (tmp_path / "out.nii.gz").exists()
    assert not (tmp_path / "out_vaso.nii.gz").exists()


def test_trailing_noise_volumes():
    means = np.tile([333.0, 300.0], 20)
    assert vaso.trailing_noise_volumes(means) == 0
    assert vaso.trailing_noise_volumes(np.concatenate([means, [74.0, 73.7]])) == 2
    # a motion dip mid-run is not a noise map
    dip = means.copy()
    dip[10] = 100.0
    assert vaso.trailing_noise_volumes(dip) == 0


def test_moco_vaso_honours_skip_last(tmp_path):
    import nibabel as nib

    stem = _main_vaso(tmp_path, "run_cbv.nii.gz", "-pair_tr", "2.5", "-skip_last", "3")
    # 15 - 3 = 12 volumes, 4 trimmed -> 4 pairs
    assert nib.load(f"{stem}_vaso.nii.gz").shape[3] == 4


def test_resample_stream_upsamples_onto_the_anchor_grid():
    """x2 grid: anchor samples plus midpoints, BOLD delayed by `delay`."""
    pair_tr, delay, n = 2.5, 0.9, 100
    t = np.arange(n) * pair_tr
    f = lambda x: np.sin(2 * np.pi * x / 40.0)  # noqa: E731
    bold = torch.tensor(f(t + delay), dtype=torch.float32)[:, None, None, None]
    out = vaso.resample_stream(bold, delay / pair_tr, factor=2, verb=0)[:, 0, 0, 0].numpy()
    assert out.shape[0] == 2 * n - 1
    t_out = np.arange(out.shape[0]) * pair_tr / 2
    # the first output is the copied first volume; everything else interpolated
    np.testing.assert_allclose(out[1:], f(t_out[1:]), atol=5e-3)
    # factor 1 is the plain shift
    one = vaso.resample_stream(bold, delay / pair_tr, factor=1, verb=0)
    assert torch.equal(one, vaso.shift_series(bold, delay / pair_tr, verb=0))


def test_moco_vaso_upsample_halves_the_tr(tmp_path):
    import json

    import nibabel as nib

    stem = _main_vaso(
        tmp_path,
        "run_cbv.nii.gz",
        "-pair_tr",
        "2.5",
        "-vaso_upsample",
        "2",
        "-vaso_save_unshifted",
    )
    img = nib.load(f"{stem}_vaso.nii.gz")
    assert img.shape[3] == 2 * 5 - 1
    assert img.header.get_zooms()[3] == pytest.approx(1.25)
    np.testing.assert_allclose(np.asarray(img.dataobj)[6, 6, 4, :], 0.8, atol=1e-3)
    meta = json.loads((tmp_path / "out_vaso.json").read_text())
    assert meta["RepetitionTime"] == pytest.approx(1.25)
    assert meta["VasoPairRepetitionTime"] == pytest.approx(2.5)
    raw = nib.load(f"{stem}_bold_unshifted.nii.gz")
    assert raw.shape[3] == 5 and raw.header.get_zooms()[3] == pytest.approx(2.5)
