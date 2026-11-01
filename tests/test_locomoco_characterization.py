"""Characterization tests — pin the numeric output of every ffs_locomoco path.

These are the safety net for the locomoco harmonization refactor. They are NOT
correctness tests (the per-primitive tests do that); they assert that each path's
displacement field is *byte-for-byte stable* on a fixed CPU seed, so a refactor
that is meant to be behavior-preserving can be proven so. The baseline numbers
were captured from the pre-refactor code. If a change here is INTENTIONAL, update
the expected dict in the same commit and say why.
"""

import numpy as np
import torch

from fastfuncstuff.processing import locomoco as L

# The xcorr searchlight picks a discrete argmax, so a sub-ULP change to the shift/blur
# math (a faster-but-equivalent kernel, float reassociation) reselects between near-tied
# offsets at isolated voxels — a ~1e-3 aggregate wobble (amplified to ~5e-3 through the
# rank-1 alpha power-iteration of the learn path) that is NOT an accuracy change: the
# recovery tests in test_locomoco.py are the accuracy gate and hold exactly. So pin at the
# LOGIC level — real bugs move fields by >= 1e-2 (refine errors, alpha inflation, railing),
# well clear of this floor. Flow paths stay byte-stable regardless.
#
# The 3-D / multi-echo xcorr baselines were re-captured 2026-07-18 when the searchlight
# adopted (a) the sinc-exact Fourier trial-shift (no linear-interp blur → no spurious
# fractional-shift correlation) and (b) the first-peak finder replacing argmax (no-shift
# biased, no railing — fields come out tighter). The recovery tests confirmed accuracy
# held throughout. 2-D xcorr still uses the linear shift + argmax (untouched), unchanged.
ATOL = 6e-3

# Where one seed's tail statistics cannot be pinned at ATOL. Measured on an M4: rescaling the
# input by 1 +/- 2e-5, which cannot change the true displacement, moved these by up to
# 0.022 (q95), 0.014 (q05), 0.0095 (q50), 0.0047 (mean) and 0.18 (absmax) on the learned
# scaling path, more than the gap from the Linux baselines. Those baselines all sit inside
# that range. These are that spread with some margin, per path and statistic; everything
# else, including the fixed-scaling path, stays at ATOL, and test_locomoco.py is the accuracy gate.
ME_JOINT_TOL = {"mean": 8e-3, "q05": 0.025, "q50": 0.015, "q95": 0.035, "absmax": 0.25}
ME_INTERECHO_TOL = {"absmax": 0.2}
ME_SCALED_TOL = {"absmax": 0.015}
ME_JOINT_ALPHA_TOL = 0.02


def _phantom4d(nx=28, ny=28, nz=8, nt=6, seed=0, decay=1.0):
    rng = np.random.default_rng(seed)
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    base = (np.sin(X / 4) * np.cos(Y / 5) + 0.3 * np.sin((X + Y) / 3) + 1.5).astype(np.float32)
    v = np.zeros((nx, ny, nz, nt), np.float32)
    for t in range(nt):
        sh = 0.6 * np.sin(t * 0.9)  # deterministic per-frame shift along y
        shifted = L._shift3d_axis(torch.from_numpy(base)[None], float(sh), 1)[0].numpy()
        v[..., t] = decay * shifted + 0.01 * rng.standard_normal((nx, ny, nz)).astype(np.float32)
    return v


def _stats(field):
    f = field.detach().cpu().float().flatten()
    q = torch.quantile(f, torch.tensor([0.05, 0.5, 0.95]))
    return {
        "mean": float(f.mean()),
        "std": float(f.std()),
        "q05": float(q[0]),
        "q50": float(q[1]),
        "q95": float(q[2]),
        "absmax": float(f.abs().max()),
    }


def _assert_stats(field, expected, tol=None):
    got = _stats(field)
    tol = tol or {}
    for k, v in expected.items():
        limit = tol.get(k, ATOL)
        assert abs(got[k] - v) < limit, f"{k}: {got[k]:.6f} != {v:.6f} (Δ {got[k] - v:+.6f})"


_DEV = torch.device("cpu")
_TES = [7.6, 21.7, 35.8]


def _me_datas():
    return [_phantom4d(decay=dc, seed=s) for s, dc in enumerate([1.0, 0.6, 0.35])]


# ── single-echo ───────────────────────────────────────────────────────────────
def test_char_2d_flow():
    r = L.estimate_residual_flow(
        _phantom4d(), pe_axis=1, slice_axis=2, backend="flow", device=_DEV, verbose=False
    )
    _assert_stats(
        r.pe_displacement(),
        {
            "mean": -0.001030,
            "std": 0.411161,
            "q05": -0.527366,
            "q50": -0.058987,
            "q95": 0.675630,
            "absmax": 0.844810,
        },
    )


def test_char_2d_xcorr():
    r = L.estimate_residual_flow(
        _phantom4d(), pe_axis=1, slice_axis=2, backend="xcorr", device=_DEV, verbose=False
    )
    _assert_stats(
        r.pe_displacement(),
        {
            "mean": -0.007996,
            "std": 0.407173,
            "q05": -0.547944,
            "q50": -0.052291,
            "q95": 0.676604,
            "absmax": 0.791614,
        },
    )


def test_char_3d_xcorr():
    r = L.estimate_residual_flow(
        _phantom4d(),
        pe_axis=1,
        slice_axis=2,
        backend="xcorr",
        is_3dacq=True,
        device=_DEV,
        verbose=False,
    )
    _assert_stats(
        r.pe_displacement(),
        {
            "mean": 0.015923,
            "std": 0.398597,
            "q05": -0.519979,
            "q50": 0.000000,
            "q95": 0.683320,
            "absmax": 0.845521,
        },
    )


def test_char_3d_flow():
    r = L.estimate_residual_flow(
        _phantom4d(),
        pe_axis=1,
        slice_axis=2,
        backend="flow",
        is_3dacq=True,
        device=_DEV,
        verbose=False,
    )
    _assert_stats(
        r.pe_displacement(),
        {
            "mean": -0.000821,
            "std": 0.409817,
            "q05": -0.522195,
            "q50": -0.071100,
            "q95": 0.669497,
            "absmax": 0.756710,
        },
    )


# ── multi-echo ──────────────────────────────────────────────────────────────
def test_char_me_joint():
    r = L.estimate_residual_flow_multiecho(
        _me_datas(), _TES, pe_axis=2, slice_axis=2, backend="xcorr", device=_DEV, verbose=False
    )
    _assert_stats(
        r.w_field,
        {
            "mean": -0.018060,
            "std": 0.675731,
            "q05": -1.085326,
            "q50": -0.019396,
            "q95": 0.990209,
            "absmax": 2.845777,
        },
        ME_JOINT_TOL,
    )
    assert np.allclose(r.alpha.tolist(), [1.0, 0.19318, 0.17425], atol=ME_JOINT_ALPHA_TOL)


def test_char_me_fixed():
    r = L.estimate_residual_flow_multiecho(
        _me_datas(),
        _TES,
        pe_axis=2,
        slice_axis=2,
        backend="xcorr",
        learn_scaling=False,
        device=_DEV,
        verbose=False,
    )
    _assert_stats(
        r.w_field,
        {
            "mean": -0.000366,
            "std": 0.089310,
            "q05": -0.107882,
            "q50": -0.006136,
            "q95": 0.110331,
            "absmax": 0.604402,
        },
    )


def test_char_me_interecho():
    r = L.estimate_residual_flow_me_interecho(
        _me_datas(), _TES, pe_axis=2, slice_axis=2, backend="xcorr", device=_DEV, verbose=False
    )
    _assert_stats(
        r.w_field,
        {
            "mean": 0.046277,
            "std": 0.472227,
            "q05": -0.576482,
            "q50": 0.083549,
            "q95": 0.655723,
            "absmax": 1.603773,
        },
        ME_INTERECHO_TOL,
    )


def test_char_me_scaled():
    r = L.estimate_residual_flow_me_scaled(
        _me_datas(), _TES, 2, pe_axis=2, slice_axis=2, backend="xcorr", device=_DEV, verbose=False
    )
    _assert_stats(
        r.w_field,
        {
            "mean": -0.004341,
            "std": 0.088442,
            "q05": -0.109205,
            "q50": -0.008755,
            "q95": 0.113341,
            "absmax": 0.595628,
        },
        ME_SCALED_TOL,
    )
