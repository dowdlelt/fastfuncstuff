"""presurfer workflow primitives (processing/presurf.py)."""

from __future__ import annotations

import torch

from fastfuncstuff.processing.presurf import (
    clean_mask,
    mprageise,
    scale_unit,
    strip_mask,
)


def _ball(shape, centre, r):
    zz, yy, xx = torch.meshgrid(*[torch.arange(n) for n in shape], indexing="ij")
    return (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2 <= r * r


def test_scale_unit_minmax_and_robust():
    x = torch.tensor([0.0, 1.0, 2.0, 4.0, 1000.0])
    assert torch.allclose(scale_unit(x), x / 1000.0)
    # one hot voxel should not set the top of the robust scale
    r = scale_unit(torch.cat([torch.linspace(1, 100, 10_000), torch.tensor([1e6])]), "robust")
    assert r[5000] > 0.4  # a mid-range value stays mid-range


def test_mprageise_suppresses_background():
    uni = torch.full((4,), 3000.0)
    inv2 = torch.tensor([0.0, 10.0, 500.0, 1000.0])  # dark background -> bright tissue
    out = mprageise(uni, inv2)
    assert out[0] == 0 and out[-1] == 3000.0


def test_strip_mask_is_complement_of_other_classes():
    post = torch.zeros(6, 1, 1, 3)
    post[0, 0, 0, 0] = 1.0  # GM
    post[3, 0, 0, 1] = 1.0  # bone
    post[0, 0, 0, 2], post[2, 0, 0, 2] = 0.3, 0.3  # sums to 0.6 -- others 0.3 -> kept
    assert strip_mask(post).flatten().tolist() == [True, False, True]


def test_clean_mask_prior_gate_removes_bridged_eye_and_fills_holes():
    shape = (40, 40, 60)
    brain = _ball(shape, (20, 20, 20), 12)
    eye = _ball(shape, (20, 20, 46), 6)
    bridge = torch.zeros(shape, dtype=torch.bool)
    bridge[19:21, 19:21, 30:42] = True  # an "optic nerve" joining eye to brain
    hole = _ball(shape, (20, 20, 20), 3)
    mask = (brain | eye | bridge) & ~hole

    # largest component alone keeps the eye: it is connected
    kept, _ = clean_mask(mask, fill_holes=False)
    assert kept[eye].all()

    prior = brain.float()  # template says brain only where the brain is
    cleaned, removed = clean_mask(mask, brain_prior=prior, prior_thresh=0.05)
    assert not cleaned[eye].any() and removed[eye].all()
    assert cleaned[brain].all()  # interior hole filled, brain intact
    assert not removed[brain & ~hole].any()


def test_clean_mask_opening_cuts_bridge_without_prior():
    shape = (40, 40, 60)
    brain = _ball(shape, (20, 20, 20), 12)
    eye = _ball(shape, (20, 20, 46), 6)
    bridge = torch.zeros(shape, dtype=torch.bool)
    bridge[19:21, 19:21, 30:42] = True
    cleaned, _ = clean_mask(brain | eye | bridge, open_radius=2)
    assert not cleaned[eye].any() and cleaned[brain].all()
