"""Tests for slice-timing-aware motion correction (space-time realignment, 1a).

Two things to pin:
  1. Correctness — with no temporal signal, slice timing is inert, so the joint
     estimator must still recover known motion (aligned output ≈ base).
  2. The point of it — timing-blind moco on raw data attributes slice-timing-vs-
     BOLD intensity changes to *motion* (stimulus-correlated motion); the
     space-time estimator, which removes that confound, should report less.

Phantoms are deliberately **asymmetric** so registration is well-posed — a
symmetric blob carries no rotational information, making rotation params
unconstrained (for plain moco too), so metrics here use translations and the
aligned-output residual, which are well-determined.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from fastfuncstuff.processing.affine import (
    _build_homo_coords,
    identity_params,
    params_to_matrix,
    resample_affine_fast,
)
from fastfuncstuff.processing.ffs_moco import MocoConfig, moco, moco_spacetime

DEV = torch.device("cpu")


def _blob(shape, center, sigma, amp=1.0):
    nz, ny, nx = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    cz, cy, cx = center
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    return amp * torch.exp(-r2 / (2 * sigma**2))


def _asym_phantom(shape=(16, 20, 20)):
    """Anatomy with structure in all three axes (rotation is well-determined)."""
    nz, ny, nx = shape
    a = _blob(shape, (nz / 2, ny / 2, nx / 2), 4.0, 1.0)
    a += _blob(shape, (nz * 0.3, ny * 0.35, nx * 0.65), 1.6, 0.8)
    a += _blob(shape, (nz * 0.7, ny * 0.65, nx * 0.3), 2.2, 0.6)
    return a


def _st_config(slice_times, tr, st_iters=2, **kw):
    return MocoConfig(
        device="cpu",
        verb=0,
        compile=False,
        slice_times=slice_times,
        st_tr=tr,
        st_tzero=None,
        st_tinterp="cubic",
        st_iters=st_iters,
        interp="heptic",
        final_interp="wsinc5",
        **kw,
    )


def _realistic_slice_times(nz, tr):
    # Ascending, spanning ~half the TR (typical single-band / multiband spread).
    return [k * (0.5 * tr) / nz for k in range(nz)]


def test_recovers_translation_no_temporal_signal():
    """No time-varying signal ⇒ slice timing is inert; motion must be recovered."""
    base = _asym_phantom().to(DEV)
    shape = base.shape
    nt = 6
    coords = _build_homo_coords(shape, DEV, torch.float32)

    ts = torch.zeros(nt, *shape)
    ts[0] = base
    steps = []
    for t in range(1, nt):
        p = identity_params(device=DEV, dtype=torch.float32)
        p[0] = 0.7 * (t / nt)  # dx
        p[1] = -0.5 * (t / nt)  # dy
        p[2] = 0.3 * (t / nt)  # dz
        steps.append((float(p[0]), float(p[1]), float(p[2])))
        ts[t] = resample_affine_fast(base, params_to_matrix(p), coords, "heptic", shape)

    slice_times = _realistic_slice_times(shape[0], 2.0)
    res = moco_spacetime(ts, _st_config(slice_times, tr=2.0, st_iters=2), header_info=None)

    # The aligned series must collapse back onto the base.
    assert res.aligned.shape == (nt, *shape)
    for t in range(nt):
        err = float((res.aligned[t] - base).abs().max())
        assert err < 0.06, f"frame {t} residual too high: {err}"
    # Recovered translation magnitude should track the injected motion (sign is
    # the inverse-alignment convention; compare magnitudes).
    for t in range(1, nt):
        recov = np.abs(res.params[t][:3])
        inj = np.abs(np.array(steps[t - 1]))
        assert np.abs(recov - inj).max() < 0.2, f"frame {t}: recov={recov} inj={inj}"


def test_zero_motion_zero_signal_is_identity():
    base = _asym_phantom().to(DEV)
    ts = base.unsqueeze(0).repeat(5, 1, 1, 1)
    slice_times = _realistic_slice_times(base.shape[0], 2.0)
    res = moco_spacetime(ts, _st_config(slice_times, tr=2.0, st_iters=2), header_info=None)
    assert np.abs(res.params[:, :3]).max() < 0.05  # no translation
    assert float((res.aligned - ts).abs().max()) < 0.05


def test_reduces_stimulus_correlated_motion():
    """Timing-blind moco attributes slice-timing-staggered BOLD to motion; the
    space-time estimator removes that component and reports less spurious motion.

    The reduction is modest by construction: a task-locked BOLD *amplitude* change
    biases motion estimation on its own (not a slice-timing effect and not
    removable here), so slice timing only accounts for the extra z-staggered
    part. We assert a real, directional reduction — not a large one.
    """
    shape = (16, 20, 20)
    nz = shape[0]
    tr = 2.0
    nt = 24
    period_sec = 8.0

    anat = _asym_phantom(shape).to(DEV)
    slice_times = _realistic_slice_times(nz, tr)
    st_z = torch.tensor(slice_times, dtype=torch.float32)

    # Whole-anatomy modulation, z-staggered by slice timing (NO spatial motion):
    # raw[f, z] = anat[z] * (1 + amp·task(f*TR + slice_time[z])).
    ts = torch.zeros(nt, *shape)
    for f in range(nt):
        m = torch.sin(2 * math.pi * (f * tr + st_z) / period_sec)  # (nz,)
        ts[f] = anat * (1.0 + 0.1 * m[:, None, None])

    common = dict(device="cpu", verb=0, compile=False, interp="heptic")
    res_blind = moco(ts, MocoConfig(skip_resample=True, **common), header_info=None)
    res_st = moco_spacetime(
        ts, _st_config(slice_times, tr=tr, st_iters=2, skip_resample=True), header_info=None
    )

    # Spurious *translation* energy (rotation is ill-posed here; there is no real
    # motion, so all of this is artifact).
    blind = float(np.sqrt((res_blind.params[:, :3] ** 2).sum(axis=1)).mean())
    st = float(np.sqrt((res_st.params[:, :3] ** 2).sum(axis=1)).mean())
    assert blind > 0.1, f"test too weak; blind translation energy {blind}"
    assert st < 0.92 * blind, (
        f"space-time did not reduce stimulus-correlated motion: blind={blind:.4f} st={st:.4f}"
    )
