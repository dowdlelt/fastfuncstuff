"""ffs_dataviz core: features, drives, scenes and sound on a synthetic run.

The run is noise everywhere plus one patch carrying a shared 10 s on / 10 s off
block response -- the "only a small part of the brain has signal" case the
pooling and salience exist for.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.temporal_filter import apply_temporal_filter
from fastfuncstuff.viz.dataviz.drives import Drives, top_two, uniform_disc
from fastfuncstuff.viz.dataviz.engine import PARAM_BY_NAME, Visualizer, default_params
from fastfuncstuff.viz.dataviz.features import (
    DEFAULT_BANDPASS_HZ,
    band_dof,
    compute_features,
    member_counts,
    resolve_band,
    spike_score,
)
from fastfuncstuff.viz.dataviz.scenes import SCENE_NAMES
from fastfuncstuff.viz.dataviz.sound import Sonifier

CPU = torch.device("cpu")
SHAPE = (14, 14, 10)
N_TIME = 80
PATCH = (slice(3, 7), slice(3, 7), slice(3, 6))


@pytest.fixture(scope="module")
def run():
    rng = np.random.default_rng(0)
    block = (np.arange(N_TIME) // 10) % 2 * 2.0 - 1.0
    data = rng.standard_normal((N_TIME, *SHAPE)).astype(np.float32)
    data[(slice(None), *PATCH)] += 1.5 * block[:, None, None, None]
    data += 100.0
    mask = np.ones(SHAPE, dtype=bool)
    mask[0], mask[-1] = False, False
    feats = compute_features(
        torch.from_numpy(data),
        mask,
        np.eye(4),
        1.0,
        [0],
        n_components=8,
        n_families=4,
        # Explicit: these tests check pooling and families on known data. 6 mm
        # on a 14-voxel grid would swamp the patch, and the patch is a sizeable
        # part of this tiny brain's global mean.
        fwhm_mm=0.0,
        remove_global=False,
        # 80 s of tiny brain leaves 18 degrees of freedom in the default band, too
        # few to isolate one patch; test_bandpass_keeps_the_block has the band.
        bandpass_hz=None,
        verbose=False,
    )
    in_patch = np.zeros(SHAPE, dtype=bool)
    in_patch[PATCH] = True
    return feats, in_patch[tuple(feats.ijk.T)], block


def test_salience_ranks_the_signal_patch_above_noise(run):
    feats, in_patch, _ = run
    sal = feats.salience.numpy()
    assert np.median(sal[in_patch]) > np.quantile(sal[~in_patch], 0.9)


def test_a_family_recovers_the_block(run):
    feats, in_patch, block = run
    patch_family = int(torch.mode(feats.family[torch.from_numpy(in_patch)]).values)
    tc = feats.family_tcs[:, patch_family].numpy()
    assert abs(np.corrcoef(tc, block)[0, 1]) > 0.8


def test_neighbour_table_points_at_face_neighbours(run):
    feats, _, _ = run
    nb = feats.extras["neighbours"].numpy()
    d = np.abs(feats.ijk[nb] - feats.ijk[:, None, :]).sum(-1)
    # Each entry is a face neighbour (distance 1) or the voxel itself (0) where
    # the neighbour is outside the mask.
    assert set(np.unique(d)) <= {0, 1}
    assert (d == 1).sum(1).min() >= 3


def test_pooling_shrinks_noise_more_than_signal(run):
    feats, in_patch, _ = run
    params = default_params() | {"clarity": 0.0, "pool": 1.0, "reach": 2}
    dr = Drives(feats, params)
    raw = feats.z.float()
    pooled = torch.stack([dr.pooled(raw[t]) for t in range(N_TIME)])
    keep = torch.from_numpy(in_patch)
    noise_ratio = pooled[:, ~keep].std() / raw[:, ~keep].std()
    signal_ratio = pooled[:, keep].std() / raw[:, keep].std()
    assert noise_ratio < 0.6
    assert signal_ratio > noise_ratio + 0.2


@pytest.mark.parametrize("fwhm", [0.0, 6.0])
def test_a_clipped_voxel_does_not_look_salient(fwhm):
    # Found on real data: voxels stuck at the 12-bit ceiling detrend to round-off,
    # which unit variance turned into noise that ranked near the top for salience.
    rng = np.random.default_rng(1)
    data = 1000.0 + 20.0 * rng.standard_normal((N_TIME, *SHAPE)).astype(np.float32)
    data += 15.0 * rng.standard_normal((N_TIME, 1, 1, 1)).astype(np.float32)
    clipped = (slice(None), 7, 7, 5)
    data[clipped] = 4095.0
    mask = np.ones(SHAPE, dtype=bool)
    feats = compute_features(
        torch.from_numpy(data), mask, np.eye(4), 1.0, [0],
        n_components=8, n_families=4, fwhm_mm=fwhm, remove_global=False, verbose=False,
    )  # fmt: skip
    v = int(np.flatnonzero((feats.ijk == (7, 7, 5)).all(1))[0])
    assert feats.coherence.abs().max() <= 1.0
    assert feats.coherence[v] == 0.0
    if fwhm == 0.0:
        assert feats.z[:, v].abs().max() == 0.0
        assert feats.salience[v] < feats.salience.median()


def test_load_features_falls_back_to_the_cpu_when_the_device_runs_out(
    tmp_path, monkeypatch, capsys
):
    import nibabel as nib

    import fastfuncstuff.viz.dataviz.features as features

    data = 100.0 + np.random.default_rng(2).standard_normal((*SHAPE, 40)).astype(np.float32)
    path = tmp_path / "run.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    real = features.compute_features
    tried = []

    def fake(*args, device, **kwargs):
        tried.append(device.type)
        if device.type != "cpu":
            raise RuntimeError("MPS backend out of memory (MPS allocated: 8 GiB)")
        return real(*args, device=device, **kwargs)

    monkeypatch.setattr(features, "compute_features", fake)
    monkeypatch.setattr(features, "_release_cache", lambda _device: None)
    feats, _ = features.load_features(
        [path],
        torch.device("meta"),
        tr=1.0,
        verbose=False,
        n_components=4,
        n_families=2,
        fwhm_mm=0.0,
    )
    assert tried == ["meta", "cpu"]
    assert feats.z.device.type == "cpu"
    assert "continuing on the CPU" in capsys.readouterr().err

    def other_failure(*_args, **_kwargs):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(features, "compute_features", other_failure)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        features.load_features([path], torch.device("meta"), tr=1.0, verbose=False)


def test_top_two_agrees_with_topk():
    sim = torch.randn(500, 40)
    expect = sim.topk(2, dim=1).indices
    first, second = top_two(sim.clone())
    assert torch.equal(first, expect[:, 0]) and torch.equal(second, expect[:, 1])


def test_member_counts_are_exact():
    family = torch.tensor([0, 2, 2, 3, 3, 3])
    assert member_counts(family, 5).tolist() == [1, 0, 2, 3, 0]


def test_uniform_disc_spreads_radii_evenly():
    xy = torch.randn(10_000, 2) * torch.rand(10_000, 1) ** 4  # piled at the origin
    r = uniform_disc(xy).norm(dim=1)
    assert r.max() <= 1.0 + 1e-6
    # Uniform on the unit disc: P(r < x) = x^2, so the median radius is sqrt(0.5).
    assert abs(float(r.median()) - 0.5**0.5) < 0.02


@pytest.mark.parametrize("scene", SCENE_NAMES)
def test_every_scene_renders(run, scene):
    feats, _, _ = run
    vis = Visualizer(feats, 64, 36)
    vis.set_param("scene", SCENE_NAMES.index(scene))
    for _ in range(20):
        vis.step(1 / 30)
    img = vis.render_tensor()
    assert img.shape == (36, 64, 3) and img.dtype == torch.uint8
    vis.resize(48, 32)
    vis.step(1 / 30)
    assert vis.render_tensor().shape == (32, 48, 3)


def test_speed_is_a_multiple_of_real_time(run, monkeypatch):
    # A 3.2 s TR played at a fixed 2 TRs per second ran 6.4x real time, whatever
    # the status bar's "x2.0" said.
    feats, _, _ = run
    monkeypatch.setattr(feats, "tr", 3.2)
    dr = Drives(feats, default_params() | {"speed": 2.0})
    dr.advance(0.05)
    assert dr.t == pytest.approx(2.0 * 0.05 / 3.2)


def test_scene_params_are_unique_and_in_range():
    for prm in PARAM_BY_NAME.values():
        assert prm.lo <= prm.default <= prm.hi, prm.name


def test_sound_is_bounded_and_finite(run):
    feats, _, _ = run
    son = Sonifier(feats)
    out = np.concatenate([son.render(735, t * 0.1) for t in range(60)])
    assert out.shape == (735 * 60, 2)
    assert np.isfinite(out).all() and np.abs(out).max() <= 1.0
    assert np.abs(out).max() > 1e-3  # not silent


def test_cli_sets_take_choice_names():
    from fastfuncstuff.cli.dataviz import parse_sets

    out = parse_sets(["scene=sand", "speed=3"])
    assert out == {"scene": float(SCENE_NAMES.index("sand")), "speed": 3.0}
    with pytest.raises(SystemExit):
        parse_sets(["nonsense=1"])


def test_voxel_index_orients_slices_to_ras():
    from fastfuncstuff.viz.dataviz.inspector import VoxelIndex

    shape = (4, 5, 6)
    ijk = np.argwhere(np.ones(shape, dtype=bool))
    # i runs left (x flipped), j and k swapped: the RAS grid must undo both.
    affine = np.array([[-2.0, 0, 0, 0], [0, 0, 2.0, 0], [0, 2.0, 0, 0], [0, 0, 0, 1]])
    vi = VoxelIndex(ijk, shape, affine)
    assert vi.shape == (4, 6, 5)
    world = (affine[:3, :3] @ ijk.T).T
    # A larger RAS index must mean further right / anterior / superior.
    for axis in range(3):
        order = np.argsort(vi.ras[:, axis], kind="stable")
        assert np.all(np.diff(world[order, axis]) >= -1e-9)
    assert (vi.lookup >= 0).all()


def test_projection_silhouette_has_no_gap_rows(run):
    from fastfuncstuff.viz.dataviz.scenes.base import axial_cells

    feats, _, _ = run
    params = default_params()
    dr = Drives(feats, params)
    # Many more cells than voxels across: one cell per voxel would leave stripes.
    _, inside, _ = axial_cells(feats.anat, 120, 120, dr.spacing)
    rows = inside.any(1)
    first, last = int(rows.nonzero()[0]), int(rows.nonzero()[-1])
    assert rows[first : last + 1].all()


@pytest.mark.parametrize("scene", ["brain", "mandala", "flow", "galaxies"])
def test_clicking_a_drawn_voxel_picks_it(run, scene):
    feats, _, _ = run
    vis = Visualizer(feats, 160, 90)
    vis.set_param("scene", SCENE_NAMES.index(scene))
    for _ in range(3):
        vis.step(1 / 30)
    vis.render_tensor()
    sc = vis.scene
    v = int(torch.argmax(feats.salience))
    at = sc.locate(v)
    assert at is not None
    got = sc.pick(*at, radius=0.5)
    # Several voxels can share a pixel; the pick must land on one drawn there.
    assert got is not None
    assert sc.locate(got) is not None
    gx, gy = sc.locate(got)
    assert abs(gx - at[0]) < 0.75 and abs(gy - at[1]) < 0.75


def test_family_count_changes_live(run):
    import copy

    feats = copy.deepcopy(run[0])
    vis = Visualizer(feats, 48, 32)
    assert vis.p["families"] == feats.family_tcs.shape[1]
    vis.set_param("families", 3)
    assert feats.family_tcs.shape[1] == 3 and int(feats.family.max()) <= 2
    assert vis.drives.n_fam == 3 and vis.drives.family_salience.numel() == 3
    for scene in SCENE_NAMES:
        vis.set_param("scene", SCENE_NAMES.index(scene))
        vis.step(1 / 30)
        assert vis.render_tensor().shape == (32, 48, 3), scene


def test_flow_twist_does_not_collapse(run):
    feats, _, _ = run
    vis = Visualizer(feats, 64, 36)
    vis.set_param("scene", SCENE_NAMES.index("flow"))
    vis.set_param("stir", 6.0)
    sc = vis.scene

    def peak() -> float:
        # Coarse bins: the synthetic voxels are a lattice, and rotating a
        # lattice moves fine-bin coincidences around without compressing it.
        cell = ((sc.pos * 4).round().long() + 8).clamp(0, 15)
        return float(torch.bincount(cell[:, 0] * 16 + cell[:, 1], minlength=256).max())

    start = peak()
    for _ in range(120):
        vis.step(1 / 30)
    # A twist preserves area; the old vortex-plus-spring piled points 6-12x.
    assert peak() < start * 1.6


def test_weave_puts_the_correlated_patch_together(run):
    feats, in_patch, _ = run
    dr = Drives(feats, default_params())
    row, col, _, _ = dr.weave(16 / 9)
    xy = torch.stack([col, row], 1)
    keep = torch.from_numpy(in_patch)
    spread_patch = torch.cdist(xy[keep], xy[keep]).mean()
    spread_all = torch.cdist(xy, xy).mean()
    # The patch shares one signal; the SOM must gather it, not scatter it.
    assert spread_patch < 0.6 * spread_all


def test_remix_stays_in_range_and_in_scene(run):
    import copy

    vis = Visualizer(copy.deepcopy(run[0]), 48, 32)
    vis.set_param("scene", SCENE_NAMES.index("mandala"))
    before = vis.remix(seed=1)
    for name, value in vis.p.items():
        prm = PARAM_BY_NAME[name]
        assert prm.lo <= value <= prm.hi, name
        if prm.scene not in (None, "mandala"):
            assert value == before[name], f"remix touched {name} of another scene"
    assert vis.p["speed"] == before["speed"] and vis.p["families"] == before["families"]
    vis.restore(before)
    assert vis.p == before


def test_config_round_trip(run, tmp_path):
    import copy
    import json

    vis = Visualizer(copy.deepcopy(run[0]), 48, 32)
    vis.set_param("scene", SCENE_NAMES.index("glass"))
    vis.set_param("color", PARAM_BY_NAME["color"].choices.index("spectrum"))
    vis.set_param("panes", 40)
    vis.set_mute(1, True)
    path = tmp_path / "look.json"
    path.write_text(json.dumps(vis.to_config()))
    other = Visualizer(copy.deepcopy(run[0]), 48, 32)
    other.from_config(json.loads(path.read_text()))
    assert other.scene_name == "glass"
    assert other.p == vis.p
    assert bool(other.drives.mute[1]) and int(other.drives.mute.sum()) == 1


def test_components_and_mute(run):
    import copy

    feats = copy.deepcopy(run[0])
    vis = Visualizer(feats, 48, 32)
    assert feats.loadings_all.shape[1] > feats.k_active >= 2
    vis.set_param("components", 4)
    assert feats.loadings.shape[1] == 4 and feats.family_tcs.shape[1] <= 4
    assert len(feats.extras["family_flags"]) == feats.family_tcs.shape[1]
    for scene in SCENE_NAMES:
        vis.set_param("scene", SCENE_NAMES.index(scene))
        vis.step(1 / 30)
        assert vis.render_tensor().shape == (32, 48, 3), scene
    # A muted family's voxels go dark and its drive stops.
    vis.step(1 / 30)
    k = int(feats.family[torch.argmax(vis.drives.brightness())])
    vis.set_mute(k, True)
    vis.step(1 / 30)
    members = feats.family == k
    assert torch.all(vis.drives.voxel_mute()[members] < 0.1) and torch.all(
        vis.drives.voxel_mute()[~members] == 1
    )
    assert float(vis.drives.fam[k]) == 0.0 and float(vis.drives.fam_amp[k]) == 0.0


def test_smoothing_keeps_coherence_and_global_comes_out():
    rng = np.random.default_rng(3)
    block = (np.arange(N_TIME) // 10) % 2 * 2.0 - 1.0
    glob = rng.standard_normal(N_TIME).astype(np.float32) * 2
    data = rng.standard_normal((N_TIME, *SHAPE)).astype(np.float32) + glob[:, None, None, None]
    data[(slice(None), *PATCH)] += 1.5 * block[:, None, None, None]
    data += 100.0
    mask = np.ones(SHAPE, dtype=bool)
    kw = dict(n_components=8, n_families=4, verbose=False)
    plain = compute_features(
        torch.from_numpy(data), mask, np.eye(4), 1.0, [0], fwhm_mm=0.0, remove_global=False, **kw
    )
    smooth = compute_features(
        torch.from_numpy(data), mask, np.eye(4), 1.0, [0], fwhm_mm=3.0, remove_global=True, **kw
    )
    # Coherence is measured before smoothing, or it would only report the kernel.
    assert torch.allclose(plain.coherence, smooth.coherence, atol=1e-4)
    # With the global signal regressed out, no family rides it...
    g = smooth.global_signal
    fam = smooth.family_tcs
    r = ((fam - fam.mean(0)) * (g - g.mean())[:, None]).mean(0) / (fam.std(0) * g.std())
    assert float(r.abs().max()) < 0.3
    # ...but the whole-brain series is still there to drive tint and flashes.
    glob_band = apply_temporal_filter(
        torch.from_numpy(glob)[None], 1.0, resolve_band(DEFAULT_BANDPASS_HZ, 1.0, [0], N_TIME)
    )[0]
    assert abs(float(np.corrcoef(g.numpy(), glob_band.numpy())[0, 1])) > 0.9


@pytest.mark.parametrize("reach,clarity", [(0, 0.0), (2, 0.35), (4, 1.0)])
def test_shown_series_for_one_voxel_matches_the_whole_field(run, reach, clarity):
    feats, in_patch, _ = run
    params = default_params() | {"reach": reach, "clarity": clarity, "pool": 0.7}
    dr = Drives(feats, params)
    v = int(np.flatnonzero(in_patch)[3])
    one = dr.shown_series_voxel(v)
    whole = torch.stack([dr.pooled(dr.activity(float(t)))[v] for t in range(N_TIME)])
    assert torch.allclose(one, whole, atol=1e-5)


def test_interp_many_matches_interp_rows(run):
    from fastfuncstuff.viz.dataviz.drives import interp_many, interp_rows

    feats, _, _ = run
    ts = [0.0, 0.3, 5.5, 41.25, N_TIME - 1.0, N_TIME - 1.4]
    many = interp_many(feats.family_tcs, ts)
    each = torch.stack([interp_rows(feats.family_tcs, t) for t in ts])
    assert torch.allclose(many, each, atol=1e-5)


def test_spike_score_does_not_inflate_a_quiet_run():
    # Shaped like a real low-motion run (max FD 0.67 mm): the worst frame is only
    # 2.2x the median, which plain robust z scored 7.6, with 16% of frames "spikes".
    dv = torch.from_numpy(np.random.default_rng(0).gamma(40.0, 0.021, 254)).float()
    dv = dv.clamp_max(1.6)
    dv[33] = 1.8
    d = spike_score(dv)
    assert (d > 2.5).float().mean() < 0.05
    assert d.max() < 4.0


def test_spike_score_is_bounded_and_ordered():
    dv = torch.ones(100) + torch.linspace(0, 0.1, 100)
    dv[10], dv[20] = 3.0, 300.0
    d = spike_score(dv)
    assert 2.5 < d[10] < d[20] <= 6.0


@pytest.mark.parametrize("fps", [60, 30])
def test_fluid_survives_a_spike_at_the_ceiling(run, fps):
    # The burst's outward flow spread its white ink at full strength, because
    # semi-Lagrangian advection does not dilute: the whole screen stayed white
    # for 4 s of a 3-frame spike.
    feats, _, _ = run
    spiked = torch.full((N_TIME,), -1.0)
    spiked[20:23] = 6.0
    viz = Visualizer(replace(feats, dvars=spiked), width=160, height=90)
    viz.set_param("scene", SCENE_NAMES.index("fluid"))
    viz.seek(15.0)
    white = []
    for _ in range(8 * fps):
        viz.step(1.0 / fps)
        white.append(float((viz.render().min(axis=2) > 240).mean()))
    assert max(white) < 0.25


def _spiky_block_run(
    n_time=320, band=DEFAULT_BANDPASS_HZ, tr=1.0, run_starts=(0,), spike=6.0, **kw
):
    """A slow block in a patch, fast noise everywhere, and one whole-brain spike."""
    rng = np.random.default_rng(5)
    block = (np.arange(n_time) // 10) % 2 * 2.0 - 1.0
    data = rng.standard_normal((n_time, *SHAPE)).astype(np.float32)
    data[(slice(None), *PATCH)] += 1.5 * block[:, None, None, None]
    data[n_time // 2] += spike
    data += 100.0
    mask = np.ones(SHAPE, dtype=bool)
    mask[0], mask[-1] = False, False
    feats = compute_features(
        torch.from_numpy(data), mask, np.eye(4), tr, list(run_starts),
        n_components=8, n_families=4, fwhm_mm=0.0, remove_global=False,
        bandpass_hz=band, verbose=False, device=CPU, **kw,
    )  # fmt: skip
    in_patch = np.zeros(SHAPE, dtype=bool)
    in_patch[PATCH] = True
    return feats, in_patch[tuple(feats.ijk.T)], block


def test_bandpass_empties_the_drawn_series_but_not_what_reports_on_the_full_band():
    on, _, _ = _spiky_block_run()
    off, _, _ = _spiky_block_run(band=None)
    # On the mirror extension, as the filter sees it: a plain FFT of the run leaks
    # from its unjoined ends (0.5% here) and would measure that instead.
    mirrored = torch.cat([on.z.float(), on.z.float().flip(0)])
    freqs = torch.fft.rfftfreq(mirrored.shape[0], d=on.tr)
    power = torch.fft.rfft(mirrored, dim=0).abs().square().sum(1)
    assert float(power[freqs > 0.13].sum() / power.sum()) < 1e-4
    # The spike must not be filtered away, nor the fast band lost to spectrum colour.
    assert int(on.dvars.argmax()) in (on.n_time // 2, on.n_time // 2 + 1)
    assert torch.allclose(on.dvars, off.dvars, atol=1e-3)
    assert torch.allclose(on.tsnr, off.tsnr, atol=1e-3)
    assert torch.allclose(on.extras["bands"], off.extras["bands"], atol=1e-3)


def test_bandpass_keeps_the_block():
    feats, in_patch, block = _spiky_block_run(spike=0.0)
    fam = int(torch.mode(feats.family[torch.from_numpy(in_patch)]).values)
    # A square wave's fundamental alone correlates 0.90 with it; the band keeps only that.
    assert abs(np.corrcoef(feats.family_tcs[:, fam].numpy(), block)[0, 1]) > 0.85


def test_components_stop_at_the_bands_rank():
    feats, _, _ = _spiky_block_run(n_time=120)
    dof = band_dof(resolve_band(DEFAULT_BANDPASS_HZ, 1.0, [0], 120), 1.0, [0], 120)
    assert feats.loadings_all.shape[1] == dof
    assert np.linalg.matrix_rank(feats.z.float().numpy(), tol=1e-2 * np.sqrt(feats.n_voxels)) <= dof


def test_psc_scale_describes_the_filtered_series():
    # A sine in band loses nothing, so its drawn scale is what the full band says
    # about the sine alone: amplitude 2 in a mean of 100.
    n_time = 200
    wave = 2.0 * np.sin(2 * np.pi * 0.05 * np.arange(n_time))
    rng = np.random.default_rng(1)
    data = rng.standard_normal((n_time, *SHAPE)).astype(np.float32) * 0.5
    data += wave[:, None, None, None].astype(np.float32) + 100.0
    feats = compute_features(
        torch.from_numpy(data), np.ones(SHAPE, dtype=bool), np.eye(4), 1.0, [0],
        n_components=4, n_families=2, fwhm_mm=0.0, remove_global=False, verbose=False,
    )  # fmt: skip
    psc = feats.extras["sd"] / feats.extras["raw_mean"] * 100
    assert float(psc.median()) == pytest.approx(2.0 / np.sqrt(2), rel=0.1)


def test_flags_see_through_the_filter():
    # Family courses carrying a fast rhythm: invisible in band, still the members' signature.
    n_time = 300
    rng = np.random.default_rng(2)
    fast = np.sin(2 * np.pi * 0.2 * np.arange(n_time))
    slow = np.sin(2 * np.pi * 0.03 * np.arange(n_time))
    data = rng.standard_normal((n_time, *SHAPE)).astype(np.float32) * 0.3
    data[(slice(None), *PATCH)] += (6 * fast + slow)[:, None, None, None].astype(np.float32)
    data += 100.0
    feats = compute_features(
        torch.from_numpy(data), np.ones(SHAPE, dtype=bool), np.eye(4), 1.0, [0],
        n_components=4, n_families=2, fwhm_mm=0.0, remove_global=False, verbose=False,
    )  # fmt: skip
    in_patch = np.zeros(SHAPE, dtype=bool)
    in_patch[PATCH] = True
    fam = int(torch.mode(feats.family[torch.from_numpy(in_patch[tuple(feats.ijk.T)])]).values)
    assert "fast" in feats.extras["family_flags"][fam]


@pytest.mark.parametrize(
    "tr, band, want",
    [
        (6.0, (0.009, 0.1), "highpass"),  # the top is over Nyquist at a 6 s TR
        (0.1, (0.009, 0.1), "bandpass"),
        (1.0, (0.0, 0.1), "lowpass"),
        (1.0, (0.01, 0.0), "highpass"),
        (1.0, (0.0, 0.0), None),
        (1.0, None, None),
    ],
)
def test_resolve_band_handles_any_tr(tr, band, want):
    filt = resolve_band(band, tr, [0], 2000)
    assert (filt.kind if filt else None) == want


def test_resolve_band_skips_a_run_too_short_to_decompose(capsys):
    assert resolve_band(DEFAULT_BANDPASS_HZ, 1.0, [0], 20) is None
    assert "not filtering" in capsys.readouterr().err


@pytest.mark.parametrize("band", [(0.1, 0.01), (-0.1, 0.1), (0.6, 0.0)])
def test_resolve_band_rejects_nonsense(band):
    with pytest.raises(ValueError):
        resolve_band(band, 1.0, [0], 300)


def test_runs_are_filtered_separately():
    # Two runs at different baselines: one filter across the join would ring there.
    feats, _, _ = _spiky_block_run(run_starts=(0, 160))
    dof = sum(
        band_dof(resolve_band(DEFAULT_BANDPASS_HZ, 1.0, [0], 160), 1.0, [0], 160) for _ in "ab"
    )
    assert feats.loadings_all.shape[1] == min(64, dof)
