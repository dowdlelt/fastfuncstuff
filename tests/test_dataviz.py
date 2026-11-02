"""ffs_dataviz core: features, drives, scenes and sound on a synthetic run.

The run is noise everywhere plus one patch carrying a shared 10 s on / 10 s off
block response -- the "only a small part of the brain has signal" case the
pooling and salience exist for.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viz.dataviz.drives import Drives, uniform_disc
from fastfuncstuff.viz.dataviz.engine import PARAM_BY_NAME, Visualizer, default_params
from fastfuncstuff.viz.dataviz.features import compute_features
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
