"""ffs_util_surfmask / surface.mask: the cross-run vertex mask."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.surface.mask import combine_run_masks


def _runs(rng, n=2000):
    cortex = np.ones(n, bool)
    cortex[:100] = False  # medial wall
    masks = []
    for lost in (slice(100, 150), slice(1900, 2000)):  # each run loses a slab edge
        m = cortex.copy()
        m[lost] = False
        masks.append(m.astype(np.float32))
    means = [rng.normal(1000, 80, n).astype(np.float32) for _ in masks]
    for m in means:
        m[500:560] = 150.0  # dropout: present in every run, but dark
    return masks, means


def test_intersection_then_clip_drops_dropout_not_bias():
    rng = np.random.default_rng(0)
    masks, means = _runs(rng)
    means[0][1000:1500] *= 0.7  # a smooth bias in one run is not dropout
    mask, meanall, clip = combine_run_masks(masks, means)
    assert not mask[:150].any() and not mask[1900:].any()  # union of what was lost
    assert not mask[500:560].any()
    assert mask[1000:1500].all() and mask[150:500].all()
    assert 150 < clip < 850 and meanall is not None
    only, _, _ = combine_run_masks(masks, means, clfrac=0)
    assert only[500:560].all()


def test_cli_writes_mask_and_meanall_and_refuses_mixed_meshes(tmp_path):
    from fastfuncstuff.cli.util_surfmask import main
    from fastfuncstuff.io.gifti import load_gifti_data, save_gifti_data

    masks, means = _runs(np.random.default_rng(1))
    paths = []
    for i, (m, a) in enumerate(zip(masks, means, strict=True)):
        meta = {"mesh_fingerprint": "abc", "geometry": "x.midthickness.surf.gii"}
        save_gifti_data(tmp_path / f"r{i}.mask.shape.gii", m, meta, time_series=False)
        save_gifti_data(tmp_path / f"r{i}.mean.shape.gii", a, meta, time_series=False)
        paths.append(i)
    m_args = [str(tmp_path / f"r{i}.mask.shape.gii") for i in paths]
    a_args = [str(tmp_path / f"r{i}.mean.shape.gii") for i in paths]
    main(["-mask", *m_args, "-mean", *a_args, "-prefix", str(tmp_path / "all.lh")])
    mask, meta = load_gifti_data(tmp_path / "all.lh.mask.shape.gii")
    assert meta["mesh_fingerprint"] == "abc" and meta["runs"] == "2"
    assert mask[200] == 1 and mask[520] == 0 and mask[120] == 0
    assert (tmp_path / "all.lh.meanall.shape.gii").is_file()
    save_gifti_data(tmp_path / "r1.mask.shape.gii", masks[1], {"mesh_fingerprint": "zzz"},
                    time_series=False)  # fmt: skip
    with pytest.raises(SystemExit, match="different meshes"):
        main(["-mask", *m_args, "-prefix", str(tmp_path / "bad")])
