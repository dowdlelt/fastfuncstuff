"""Tiered residency and the session that drives it.

The bugs these guard against are the ones that make a viewer feel slow: reading
a whole 4-D file to show one slice, inflating the same dataset twice because two
repaints both asked, and evicting the working set a compute is mid-way through.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.residency import Tier, VolumeStore
from fastfuncstuff.viewer.session import ViewerSession, derive_range
from fastfuncstuff.viewer.vocab import SetIJK, SetIndex

nib = pytest.importorskip("nibabel")

CPU = torch.device("cpu")


@pytest.fixture
def dataset(tmp_path):
    """A small 4-D NIfTI with a known, per-volume-distinct pattern."""
    rng = np.random.default_rng(0)
    nx, ny, nz, nv = 8, 9, 6, 12
    data = rng.normal(size=(nx, ny, nz, nv)).astype(np.float32)
    # Make each volume identifiable so an off-by-one in the seek is visible.
    for v in range(nv):
        data[0, 0, 0, v] = float(v) * 100.0
    affine = np.diag([3.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [-12.0, -13.5, -10.0]
    path = tmp_path / "series.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path, data


@pytest.fixture
def store():
    s = VolumeStore(device=CPU)
    yield s
    s.shutdown()


# ---------------------------------------------------------------------------
# tiers
# ---------------------------------------------------------------------------


def test_open_reads_the_header_only(store, dataset):
    path, data = dataset
    res = store.open(path)
    assert res.tier is Tier.HEADER
    assert res.array is None
    assert res.info.shape == data.shape


def test_preview_returns_volume_zero_without_a_full_load(store, dataset):
    path, data = dataset
    key = store.open(path).key
    vol = store.preview(key)
    assert vol.shape == data.shape[:3]
    assert np.allclose(vol, data[..., 0], atol=1e-5)
    assert store.get(key).array is None, "preview must not pull the whole series in"
    assert store.get(key).tier is Tier.PREVIEW


def test_preview_can_seek_to_a_later_volume(store, dataset):
    path, data = dataset
    key = store.open(path).key
    vol = store.preview(key, 7)
    assert np.allclose(vol, data[..., 7], atol=1e-5)


def test_preview_of_volume_zero_is_cached(store, dataset):
    path, _ = dataset
    key = store.open(path).key
    assert store.preview(key) is store.preview(key)


def test_ensure_ram_promotes_to_the_ram_tier(store, dataset):
    path, data = dataset
    key = store.open(path).key
    arr = store.ensure_ram(key)
    assert arr.shape == data.shape
    assert np.allclose(arr, data, atol=1e-5)
    assert store.get(key).tier is Tier.RAM


def test_concurrent_loads_share_one_inflate(store, dataset):
    """Two repaints asking at once must not inflate the file twice."""
    path, _ = dataset
    key = store.open(path).key
    f1 = store.load_async(key)
    f2 = store.load_async(key)
    assert f1 is f2
    f1.result()


def test_promote_moves_to_the_device_tier(store, dataset):
    path, data = dataset
    key = store.open(path).key
    t = store.promote(key)
    assert isinstance(t, torch.Tensor)
    assert t.shape == data.shape
    assert store.get(key).tier is Tier.DEVICE


def test_promote_is_idempotent(store, dataset):
    path, _ = dataset
    key = store.open(path).key
    assert store.promote(key) is store.promote(key)


def test_demote_keeps_ram_residency(store, dataset):
    path, _ = dataset
    key = store.open(path).key
    store.promote(key)
    store.demote(key)
    assert store.get(key).tier is Tier.RAM


def test_missing_file_is_an_error(store, tmp_path):
    with pytest.raises(FileNotFoundError):
        store.open(tmp_path / "nope.nii.gz")


def test_unopened_key_is_an_error(store):
    with pytest.raises(KeyError):
        store.get("never-opened")


# ---------------------------------------------------------------------------
# eviction
# ---------------------------------------------------------------------------


def test_device_eviction_drops_least_recently_used(tmp_path, dataset):
    """A budget that fits one dataset must evict the older one, not the new one."""
    path, data = dataset
    one_size = data.size * 4
    store = VolumeStore(device=CPU, device_budget=int(one_size * 1.5))
    try:
        a = store.open(path, key="a")
        b = store.open(path, key="b")
        store.promote(a.key)
        store.promote(b.key)
        assert store.get("a").tensor is None, "LRU should have evicted 'a'"
        assert store.get("b").tensor is not None
    finally:
        store.shutdown()


def test_eviction_never_drops_the_dataset_being_promoted(tmp_path, dataset):
    path, data = dataset
    store = VolumeStore(device=CPU, device_budget=1)  # nothing fits
    try:
        key = store.open(path).key
        t = store.promote(key)
        assert t is not None
        assert store.get(key).tensor is not None
    finally:
        store.shutdown()


def test_ram_eviction_spares_what_the_device_is_working_on(dataset):
    path, data = dataset
    store = VolumeStore(device=CPU, ram_budget=1)
    try:
        a = store.open(path, key="a")
        store.ensure_ram(a.key)
        store.promote(a.key)
        b = store.open(path, key="b")
        store.ensure_ram(b.key)
        assert store.get("a").array is not None, "must not evict a device working set"
    finally:
        store.shutdown()


# ---------------------------------------------------------------------------
# ranging
# ---------------------------------------------------------------------------


def test_derive_range_uses_percentiles_not_extremes():
    """One bright voxel must not flatten the whole map."""
    values = np.concatenate([np.random.default_rng(1).normal(size=10_000), [1e6]])
    lo, hi = derive_range(values)
    assert hi < 100.0


def test_derive_range_ignores_non_finite():
    values = np.array([np.nan, np.inf, 1.0, 2.0, 3.0], dtype=np.float32)
    lo, hi = derive_range(values)
    assert np.isfinite(lo) and np.isfinite(hi)


def test_derive_range_survives_a_constant_volume():
    lo, hi = derive_range(np.full(100, 5.0))
    assert hi > lo


def test_derive_range_survives_an_empty_volume():
    lo, hi = derive_range(np.array([np.nan, np.nan]))
    assert hi > lo


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------


@pytest.fixture
def session():
    s = ViewerSession(device=CPU)
    yield s
    s.close()


def test_loading_makes_a_layer_visible_before_the_full_inflate(session, dataset):
    path, data = dataset
    key = session.load(path)
    layer = session.state.layers.get(key)
    assert layer.shape == data.shape[:3]
    assert layer.n_volumes == data.shape[3]
    assert layer.range_lo is not None and layer.range_hi is not None
    assert session.state.grid is not None


def test_volume_falls_back_to_disk_before_residency(session, dataset):
    path, data = dataset
    key = session.load(path)
    session.store.release(key)
    session.do(SetIndex(5))
    assert np.allclose(session.volume(key), data[..., 5], atol=1e-5)


def test_volume_follows_the_time_index(session, dataset):
    path, data = dataset
    key = session.load(path)
    session.store.ensure_ram(key)
    session.do(SetIndex(9))
    assert np.allclose(session.volume(key), data[..., 9], atol=1e-5)


def test_timeseries_is_empty_until_resident_rather_than_blocking(session, dataset):
    """The graph pane asks on every crosshair move; it must never wait on I/O."""
    path, _ = dataset
    key = session.load(path)
    session.store.release(key)
    assert session.timeseries(key, (1, 1, 1)).size == 0


def test_timeseries_matches_the_source_once_resident(session, dataset):
    path, data = dataset
    key = session.load(path)
    session.store.ensure_ram(key)
    session.do(SetIJK(2, 3, 4))
    ts = session.timeseries(key)
    assert np.allclose(ts, data[2, 3, 4, :], atol=1e-5)


def test_timeseries_out_of_bounds_is_empty_not_an_error(session, dataset):
    path, _ = dataset
    key = session.load(path)
    session.store.ensure_ram(key)
    assert session.timeseries(key, (99, 99, 99)).size == 0


def test_a_session_replays_from_its_own_script(session, dataset, tmp_path):
    path, _ = dataset
    key = session.load(path)
    session.do(SetIJK(3, 4, 2))
    session.do(SetIndex(6))
    script = session.save_script(tmp_path / "session.ffs").read_text()

    replay = ViewerSession(device=CPU)
    try:
        replay.run_script(script)
        assert replay.state.crosshair == session.state.crosshair
        assert replay.state.time_index == session.state.time_index
        assert replay.state.layers.keys == [key]
    finally:
        replay.close()
